"""Complete checkpoint text forwards through compiled native TP2/TP4 graphs."""

import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from tests.components.completion_support import completion_outputs
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS") != "1",
    reason="requires four assigned GPUs and FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS=1")


def _inputs(metadata, tokens, start):
    device = metadata.token_ids.device
    count = len(tokens)
    metadata.token_ids.zero_()
    metadata.token_ids[:count].copy_(torch.tensor(tokens, dtype=torch.int64, device=device))
    metadata.sample_indices.fill_(count - 1)
    delta, index = metadata.recurrent, metadata.indexed
    delta.num_seqs.fill_(1)
    delta.num_tokens.fill_(count)
    delta.query_start_loc[1:].fill_(count)
    delta.convolution_slots[0] = 1
    delta.fresh[0] = start == 0
    delta.state_indices[0, 0] = 1
    delta.initial_state_indices[0] = 1
    delta.final_state_indices[0] = 1
    index.state_slots[0] = 1
    index.query_start_loc[1:].fill_(count)
    index.block_table[0].copy_(torch.arange(index.block_table.shape[1] - 1, -1, -1,
                                           dtype=torch.int32, device=device))
    index.positions.fill_(-1)
    index.positions[:count].copy_(torch.arange(start, start + count, dtype=torch.int64, device=device))
    index.token_slots.fill_(-1)
    index.token_slots[:count].copy_(index.block_table[0, index.positions[:count] // 256].long()
                                   * 256 + index.positions[:count] % 256)


def _rank(rank, world, rendezvous, snapshot, capacity, generate=False, sockets=(), checkpoint=False):
    from transformers import AutoTokenizer

    from ignition.components.parallel import TensorParallelShard
    from ignition.components.sampling import (
        AdmissionBuffers,
        DeviceRequestState,
        RequestAdmission,
    )
    from ignition.components.state_transfer import StateCommands
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_gather_runtime,
        create_all_reduce_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.batch import BatchInputs, PreparedBatch
    from ignition.recipes.glm53.generation import PreparedGeneration
    from ignition.recipes.glm53.maintenance import PreparedMaintenance
    from ignition.recipes.glm53.pool import ContinuationPool
    from ignition.recipes.glm53.resume import PreparedResume
    from ignition.recipes.glm53.stack import DecoderScratch, PreparedDecoder
    from ignition.recipes.glm53.state import DecoderMetadata
    from ignition.recipes.glm53.weights import ModelWeights

    peers = []
    for index, (root, peer) in enumerate(sockets, start=1):
        if rank == 0:
            peers.append(root)
            peer.close()
        else:
            root.close()
            if rank == index:
                connection = peer
            else:
                peer.close()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    def progress(message):
        if rank == 0:
            print(f"TP{world}: {message}", flush=True)
    shard = TensorParallelShard(rank, world)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress)
    assert len(weights.layers) == 45
    head = weights.head if isinstance(weights.head, torch.Tensor) else weights.head.weight.values
    assert head.shape == (154880 // world, 4096)
    assert not head.is_meta
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=15))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group, timeout=timedelta(minutes=15))
    runtime = create_all_reduce_runtime(group, device, max_tokens=capacity, hidden_size=4096)
    vocabulary_runtime = create_all_gather_runtime(group, device, max_rows=1, columns=154880)
    worker = None
    graphs = {mode: torch.cuda.CUDAGraph() for mode in ("prefill", "decode")}
    try:
        pool = ContinuationPool.allocate(weights.tree.config.text_config.layer_types, world, 6, 64, device)
        state = pool.decoder
        arena = ScratchArena(device)
        requests = DeviceRequestState.allocate(device, vocab_size=154880,
            consumed_tokens=pool.consumed_tokens[1:5, 0]) if generate else None
        inputs = BatchInputs.allocate(capacity, 64, device) if generate else None
        records = torch.zeros((2, 4, 8), dtype=torch.int64, device=device) if generate else None
        frame = torch.zeros(1, dtype=torch.int32, device=device) if generate else None
        sequence = torch.zeros(1, dtype=torch.int64, device=device) if generate else None
        admission = AdmissionBuffers.allocate(device, vocab_size=154880) if checkpoint else None
        commands = StateCommands.allocate(device) if checkpoint else None
        with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                workload_schema="glm-text-stack-tp-v1", numerical_policy="default", coordinator=coordinator) as session:
            session.register_scratch(arena)
            decoders, metadata, scratch, forwards = {}, {}, {}, {}
            for mode, tokens in (("prefill", capacity), ("decode", 1)):
                metadata[mode] = DecoderMetadata.allocate(tokens, 1, 64, device)
                scratch[mode] = DecoderScratch.allocate(tokens, 64, mode, device, shard=shard, sample_rows=1)
                decoders[mode] = PreparedDecoder(weights, scratch[mode], state, metadata[mode], mode=mode,
                    runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator,
                    session=session, name=mode, workspace=arena,
                    progress=progress)
                forwards[mode] = decoders[mode]
                if generate:
                    batch = PreparedBatch(inputs, requests, metadata[mode], mode=mode, shard=shard, pages=64)
                    forwards[mode] = PreparedGeneration(decoders[mode], batch, completion_outputs(records), frame, sequence,
                        pool=pool, runtime=runtime, coordinator=coordinator, session=session, name=f"{mode}.generation")
            if checkpoint:
                forwards["resume"] = PreparedResume(weights, requests, pool, inputs, completion_outputs(records), frame, sequence,
                    rows=1, runtime=runtime, vocabulary_runtime=vocabulary_runtime,
                    coordinator=coordinator, session=session, name="resume")
                for mode in ("init", "restore", "snapshot"):
                    forwards[mode] = PreparedMaintenance(mode, requests, pool, commands, inputs, completion_outputs(records), frame, sequence,
                        runtime=runtime, coordinator=coordinator, session=session, name=mode, admission_buffers=admission)
                graphs.update({mode: torch.cuda.CUDAGraph() for mode in ("resume", "init", "restore", "snapshot")})
            progress("autotuning complete text forwards")
            session.tune()
            compiled = {}
            stream = torch.cuda.Stream(device=device)
            for mode, decoder in forwards.items():
                progress(f"compiling {mode}")
                compiled[mode] = decoder if isinstance(decoder, PreparedMaintenance) else compile_forward(decoder)
                compiled[mode]()
                torch.cuda.synchronize(device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graphs[mode], stream=stream):
                    compiled[mode]()
                torch.cuda.synchronize(device)
            session.freeze()
            buffers = dict(pool.partial_pages(), continuation=pool.backing)
            buffers.update({mode + ".hidden": value.hidden for mode, value in scratch.items()})
            resources = (weights, compiled, decoders, forwards, pool, metadata, scratch, session, arena, runtime, stream)
            if generate:
                buffers = dict(inputs.buffers(), completion_records=records, frame_index=frame, submission_sequence=sequence)
                signatures = {mode: GraphSignature(mode, 1, capacity if mode == "prefill" else 1, tuple(inputs.buffers()))
                              for mode in graphs}
                if checkpoint:
                    state_buffers = commands.buffers()
                    admission_buffers = {name: getattr(admission, name) for name in admission.__dataclass_fields__}
                    buffers.update(state_buffers)
                    buffers.update(admission_buffers)
                    rows = ("row_slot_ids", "row_generations", "live_rows")
                    signatures["resume"] = GraphSignature("resume", 1, 1, rows)
                    signatures["snapshot"] = GraphSignature("snapshot", 4, 1, (*rows, "page_tables", *state_buffers))
                    for mode in ("init", "restore"):
                        signatures[mode] = GraphSignature(mode, 4, 1, (*admission_buffers, "page_tables", *state_buffers))
                bundle = PreparedExecution(device=rank, graphs=graphs, buffers=buffers, signatures=signatures,
                    resources=(*resources, requests, inputs, admission, commands))
            else:
                bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers=buffers, resources=resources)
                worker = bundle.start_worker()
            pool.clear_for_preparation()
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            seed = tokenizer.encode("Explain how tensor parallelism divides a language model across GPUs.")
            tokens = (seed * ((capacity + 3 + len(seed) - 1) // len(seed)))[:capacity + 3]
            generated, finished = [], False
            if generate:
                content = "In one short sentence, what does tensor parallelism do?"
                if checkpoint:
                    content = "Context: " + "Tensor operations can run on multiple GPUs. " * 30 + "\n" + content
                tokens = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, reasoning_effort="low")
                assert 0 < len(tokens) <= capacity
                if checkpoint:
                    assert len(tokens) > 256 and len(tokens) % 4 != 0
                eos = json.loads((Path(snapshot) / "generation_config.json").read_text())["eos_token_id"]
                if not checkpoint:
                    requests.admit(0, RequestAdmission(1, 13, max_new_tokens=32, eos_token_ids=tuple(eos)), tokens)
                host = BatchInputs.allocate(capacity, 64, "cpu")
                host.row_slot_ids[0], host.row_generations[0] = 0, 1
                host.live_rows.fill_(1)
                host.sample_enabled[0] = 1
                host.page_tables[0].copy_(torch.arange(63, -1, -1, dtype=torch.int32))
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                before_handoff = torch.cuda.memory_stats(device)
                if rank == 0:
                    worker = bundle.start_executor(peers=peers)
                else:
                    bundle.serve_rank(connection, rank=rank, world=world)
            branches = {}
            if checkpoint and rank == 0:
                from tests.recipes.glm53_cached_generation import run_cached_generation

                branches = run_cached_generation(worker, bundle, pool, requests, scratch, forwards["resume"],
                    tokens, eos, capacity, progress)
                finished = True
            for index in range((0 if rank or checkpoint else 32) if generate else 4):
                mode = "prefill" if index == 0 else "decode"
                start = 0 if index == 0 else (len(tokens) if generate else capacity) + index - 1
                if generate:
                    ids = tokens if index == 0 else generated[-1:]
                    host.token_ids.zero_()
                    host.token_ids[:len(ids)].copy_(torch.tensor(ids))
                    host.query_lengths[0] = len(ids)
                    uploads = [(name, 0, value.numpy().tobytes()) for name, value in host.buffers().items()]
                else:
                    _inputs(metadata[mode], tokens[:capacity] if index == 0 else tokens[start:start + 1], start)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                if not generate:
                    dist.barrier(group=group)
                if generate:
                    ticket = worker.submit(bundle.graph_indices[mode], uploads)
                    completion = None
                    deadline = time.monotonic() + 120
                    while completion is None and time.monotonic() < deadline:
                        completion = worker.poll()
                        if completion is None:
                            time.sleep(.001)
                    assert ticket is not None and completion is not None and completion.fence == ticket
                else:
                    ticket = worker.submit(bundle.graph_indices[mode])
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                if generate:
                    record = completion.records[0]
                    assert record[:4] == [1, ticket, len(tokens) + index, index + 1 if rank == 0 else 0]
                    assert record[6] == 0 and record[5] in (1, 2)
                    token, finished = record[4], record[5] == 2
                    assert 0 <= token < 154880
                    generated.append(token)
                hidden = scratch[mode].hidden
                assert torch.isfinite(hidden).all() and hidden.count_nonzero()
                if generate:
                    torch.testing.assert_close(pool.last_hidden[1], hidden[0], rtol=0, atol=0)
                if not generate:
                    gathered = [None] * world
                    dist.all_gather_object(gathered, hidden.cpu(), group=group)
                    for value in gathered:
                        torch.testing.assert_close(value, hidden.cpu(), rtol=0, atol=0)
                if rank == 0:
                    logits = scratch[mode].logits
                    assert torch.isfinite(logits).all() and logits.count_nonzero()
                for layer_state in state.layers:
                    if hasattr(layer_state, "recurrent"):
                        assert not layer_state.recurrent[0].count_nonzero()
                        assert torch.isfinite(layer_state.recurrent[1]).all()
                progress(f"native {mode} completed at position {start}")
                if finished:
                    break
            if generate:
                if rank == 0:
                    assert (branches if checkpoint else generated) and finished
                    if checkpoint:
                        for slot, result in branches.items():
                            progress(f"cached branch {slot}: {tokenizer.decode(result)!r}")
                    else:
                        progress(f"sampled {len(generated)} tokens: {tokenizer.decode(generated)!r}")
                    worker.close()
                    assert worker.drained
                    worker = None
                after_handoff = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    # Rank-zero output checks allocate diagnostic tensors after each step.
                    if rank:
                        assert before_handoff[key] == after_handoff[key], key
                for value in scratch.values():
                    gathered = [None] * world
                    dist.all_gather_object(gathered, value.hidden.cpu(), group=group)
                    for hidden in gathered:
                        torch.testing.assert_close(hidden, value.hidden.cpu(), rtol=0, atol=0)
                counts = [{slot: len(result) for slot, result in branches.items()} if checkpoint else {0: len(generated)}]
                dist.broadcast_object_list(counts, src=0, group=group)
                for slot, count in counts[0].items():
                    assert requests.consumed_tokens[slot] == len(tokens) + count - 1
                    assert requests.generated_tokens[slot] == (count if rank == 0 else 0)
                last_slot = 3 if checkpoint else 1
                torch.testing.assert_close(pool.last_hidden[last_slot], scratch["decode"].hidden[0], rtol=0, atol=0)
                assert not pool.backing[[0, 4] if checkpoint else [0, 2, 3, 4, 5]].count_nonzero()
                if checkpoint:
                    assert pool.consumed_tokens[1] == len(tokens) and pool.consumed_tokens[5] == len(tokens)
                    torch.testing.assert_close(pool.last_hidden[5], scratch["prefill"].hidden[0], rtol=0, atol=0)
                    torch.testing.assert_close(forwards["resume"].state.hidden[0], pool.last_hidden[5], rtol=0, atol=0)
            progress(f"weight bytes {weights.allocated_bytes}; operator scratch bytes {arena.nbytes}")
            if worker is not None:
                worker.close()
                worker = None
            for graph in graphs.values():
                graph.reset()
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        vocabulary_runtime.close()
        runtime.close()
        dist.destroy_process_group()
        for pair in sockets:
            for connection in pair:
                connection.close()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("capacity", [65, 4096])
def test_complete_text_stack_native_tp_replay(world, capacity, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), capacity),
             nprocs=world, join=True)


@pytest.mark.parametrize("world", [2, 4])
def test_complete_generation_native_tp_replay(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    sockets = [socket.socketpair() for _ in range(world - 1)]
    processes = mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), 128, True, sockets),
                         nprocs=world, join=False)
    for pair in sockets:
        for connection in pair:
            connection.close()
    while not processes.join(timeout=1):
        pass


@pytest.mark.parametrize("world", [2, 4])
def test_complete_cached_generation_native_tp_replay(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    sockets = [socket.socketpair() for _ in range(world - 1)]
    processes = mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), 512, True, sockets, True),
                         nprocs=world, join=False)
    for pair in sockets:
        for connection in pair:
            connection.close()
    while not processes.join(timeout=1):
        pass
