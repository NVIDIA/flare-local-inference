"""Rank-zero sampling decisions survive divergent logits and native graph replay."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires explicitly assigned tensor-parallel GPUs")


class SamplingRound:
    def __init__(self, proposals, verify, ids, probabilities, valid, sampled, offset, seed):
        self.proposals, self.verify = proposals, verify
        self.ids, self.probabilities, self.valid = ids, probabilities, valid
        self.sampled, self.offset = sampled, offset
        self.seed = seed

    def __call__(self):
        for index, proposal in enumerate(self.proposals):
            self.offset.fill_(index * 4)
            proposal()
            self.ids[:, index].copy_(self.sampled)
            self.probabilities[:, index].copy_(proposal.probabilities[:, 0])
            self.valid[:, index].copy_(proposal.valid[:, 0])
        self.verify()

    def results(self):
        return (self.ids, self.probabilities, self.valid, self.sampled, self.offset, self.seed,
                self.verify.seed_bits, self.verify.offset,
                *(value for name, value in vars(self.verify.output).items() if name != 'survival'),
                self.verify.rng.generation, self.verify.rng.target_offset,
                self.verify.state.seed.view(torch.int64), self.verify.state.rng_counter.view(torch.int64))


def _rank(rank, world, rendezvous, result_channel=False):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.parallel_draft import prepare_sampling_broadcasts, prepare_sampling_scratch
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.speculation import PreparedStochasticVerification, SpeculativeAcceptance, SpeculativeRngState
    from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                                create_result_broadcast_runtime)
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    # TP2 and TP4 exercise multirow messages; TP16 uses a vocabulary-sized channel.
    runtime = create_all_gather_runtime(dist.group.WORLD, device, max_rows=64,
                                        columns=world * 64 if world <= 4 else 163840)
    result_runtime = (create_result_broadcast_runtime(dist.group.WORLD, device,
        layouts=(PreparedDraftDistribution.broadcast_layout(8), SpeculativeAcceptance.broadcast_layout(8, 6)))
        if result_channel else None)
    if result_channel:
        assert result_runtime.max_batch_size == 13
        assert result_runtime.heads_per_rank * result_runtime.query_head_dim * 2 == 512
    graphs, owners, references, calls = {}, [], [], []
    worker = None
    try:
        capacity = 16 if result_channel else 4
        batch_capacity = 8 if result_channel else 4
        state = DeviceRequestState.allocate(device, vocab_size=257, capacity=capacity)
        for slot in range(capacity):
            state.admit(slot, RequestAdmission(generation=7, seed=93 + slot, max_new_tokens=100,
                                               top_p=.95, eos_token_ids=()), [1, 2])
        state.grammar_enabled[2] = 1
        state.grammar_mask[2].fill_(0x55555555)
        slots = torch.tensor([3, 0, 2, 1, *range(4, batch_capacity)], dtype=torch.int32, device=device)
        epochs = torch.full((batch_capacity,), 7, dtype=torch.int64, device=device)
        epochs[3] = 99
        buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        with RecordedPreparationSession(device=device, checkpoint="sampling-broadcast",
                workload_schema="rank-sampling-v1", numerical_policy="default", coordinator=coordinator) as session:
            shapes = ([(batch, width, batch) for batch in (1, 2, 4, 8) for width in range(1, 7)]
                      + [(8, 6, 3), (8, 6, 0)] if result_channel else
                      [(4, 3, 4), (1, 1, 1), (4, 8, 3), (4, 3, 0)])
            for batch, width, active in shapes:
                live = torch.tensor([active], dtype=torch.int32, device=device)
                cursor = buffers.cursor()
                workspace = prepare_sampling_scratch(batch, width, 257, device, buffers=cursor)
                ids = torch.zeros((batch, width - 1), dtype=torch.int64, device=device)
                probabilities = cursor.zeros((batch, width - 1, 257), torch.float32)
                valid = torch.zeros((batch, width - 1), dtype=torch.bool, device=device)
                seed = torch.arange(batch, dtype=torch.int64, device=device) + 51
                offset, sampled = torch.zeros_like(seed), torch.zeros_like(seed)
                logits = torch.randn((batch, 1, 257), generator=torch.Generator().manual_seed(17 + rank)).to(device)
                proposals = tuple(PreparedDraftDistribution(logits, state, slots[:batch], epochs[:batch], live, ids,
                    prefix_offset=index, seed=seed, offset=offset, sampled=sampled,
                    workspace=workspace["proposal"]) for index in range(width - 1))
                for proposal in proposals:
                    proposal.prepare()
                target = torch.softmax(torch.randn((batch, width, 257),
                    generator=torch.Generator().manual_seed(31 + rank)), -1).to(device)
                rng = SpeculativeRngState.allocate(device, capacity=capacity)
                rng.generation.fill_(7)
                rng.target_offset.copy_(torch.arange(capacity, device=device) * 16)
                verify = PreparedStochasticVerification(probabilities, ids, target, state,
                    slots[:batch], epochs[:batch], live, rng=rng,
                    draft_valid=valid, workspace=workspace["verify"])
                verify.prepare()
                operation = SamplingRound(proposals, verify, ids, probabilities, valid, sampled, offset, seed)
                operation()
                reference = [tensor.cpu().clone() for tensor in operation.results()]
                dist.broadcast_object_list(reference, src=0)
                references.append(reference)
                prepare_sampling_broadcasts(proposals, verify,
                    collectives=dict(vocabulary_runtime=runtime, coordinator=coordinator), session=session,
                    name=f"sampling{batch}x{width}-active{active}", buffers=cursor, result_runtime=result_runtime)
                assert verify.broadcast.gather.runtime is (result_runtime if result_channel else runtime)
                for proposal in proposals:
                    assert proposal.broadcast.gather.runtime is (result_runtime if result_channel else runtime)
                owners.append(operation)
            session.tune(show_progress=False)
            for index, operation in enumerate(owners):
                call = compile_forward(operation)
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                graphs[str(index)] = graph
                calls.append(call)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers={},
                resources=(owners, calls, session, runtime, result_runtime, state, buffers))
            worker = bundle.start_worker()
            order = ((*range(len(owners)), *reversed(range(len(owners)))) if result_channel
                     else (0, 1, 2, 3, 0, 2, 1, 3))
            for index in order:
                # Poison shared scratch before switching shape; every replay must initialize it.
                for chunks in buffers.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else 255)
                torch.cuda.synchronize()
                dist.barrier()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                for actual, expected in zip(owners[index].results(), references[index], strict=True):
                    assert torch.equal(actual.cpu(), expected)
                verify = owners[index].verify
                if verify.drafts.shape[1]:
                    indices = verify.safe_ids.long().unsqueeze(-1)
                    draft = verify.safe_draft_probs.gather(-1, indices).squeeze(-1)
                    target = verify.safe_target_probs[:, :-1].gather(-1, indices).squeeze(-1)
                    expected = torch.where(draft > 0, (target / draft).clamp(max=1.), (target > 0).float()).cumprod(-1)
                    actual = verify.output.survival[:, :verify.drafts.shape[1]]
                    observed = actual >= 0
                    torch.testing.assert_close(actual[observed], expected[observed])
            worker.close()
            worker = None
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        if result_runtime is not None:
            result_runtime.close()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 16])
def test_native_sampling_broadcast(world, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)


@pytest.mark.parametrize("world", [2, 4])
def test_native_sampling_result_channel_catalog(tmp_path, world):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() >= world
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), True), nprocs=world, join=True)


def _byte_rank(rank, world, rendezvous):
    from ignition.components.collectives import PreparedBroadcast
    from ignition.preparation.collectives import CollectivePreparation, create_result_broadcast_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_result_broadcast_runtime(dist.group.WORLD, device, layouts=(
        {"ids": ((8, 64), torch.int64), "probabilities": ((8, 64), torch.float32),
         "seed": ((8,), torch.int64), "valid": ((8, 1), torch.bool)},))
    graphs, owners, calls, cases = {}, [], [], []
    worker = None
    try:
        # A sampling support payload spans 13 rows of this 512-byte rank partition.
        payload = (torch.empty((8, 64), dtype=torch.int64, device=device),
                   torch.empty((8, 64), dtype=torch.float32, device=device),
                   torch.empty(8, dtype=torch.int64, device=device),
                   torch.empty((8, 1), dtype=torch.bool, device=device))
        disjoint = tuple(tensor.view(torch.uint8).view(-1) for tensor in payload)
        alias = torch.full((16,), 2 + rank, dtype=torch.uint8, device=device)
        other = torch.full((16,), 2 + rank, dtype=torch.uint8, device=device)
        layouts = ((payload, disjoint),
                   ((alias[:8], alias[:8].view(torch.bool), alias[4:12]), (alias,)),
                   ((other[:8].view(torch.bool), other[:8]), (other,)))
        with RecordedPreparationSession(device=device, checkpoint="sampling-broadcast-bytes",
                workload_schema="ordered-byte-broadcast-v1", numerical_policy="bitwise",
                coordinator=coordinator) as session:
            for index, (tensors, backings) in enumerate(layouts):
                if index == 0:
                    for field, backing in enumerate(backings):
                        backing.copy_((torch.arange(backing.numel(), device=device) * 29 + field * 53 + rank).to(torch.uint8))
                    backings[-1].fill_(255)
                inputs = tuple(backing.cpu().clone() for backing in backings)
                expected = [tensor.clone() for tensor in inputs]
                packed = [((tensor.view(torch.uint8).view(-1) != 0).to(torch.uint8)
                           if tensor.dtype == torch.bool else tensor.view(torch.uint8).view(-1)).cpu()
                          for tensor in tensors]
                dist.broadcast_object_list(packed, src=0)
                for tensor, value in zip(tensors, packed, strict=True):
                    for backing, result in zip(backings, expected, strict=True):
                        offset = tensor.data_ptr() - backing.data_ptr()
                        if 0 <= offset and offset + value.numel() <= backing.numel():
                            result[offset:offset + value.numel()].copy_(value)
                            break
                    else:
                        raise AssertionError("broadcast field must belong to a retained backing")
                operation = PreparedBroadcast(tensors, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"bytes{index}")
                assert len(operation._unpack_groups) == (1, 3, 2)[index]
                owners.append(operation)
                cases.append((backings, inputs, expected))
            session.tune(show_progress=False)
            for index, operation in enumerate(owners):
                call = compile_forward(operation)
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                graphs[str(index)] = graph
                calls.append(call)
            session.freeze()
            bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers={},
                resources=(owners, calls, session, runtime, cases))
            worker = bundle.start_worker()
            for index in (0, 1, 2, 0, 2, 1):
                for backing, initial in zip(cases[index][0], cases[index][1], strict=True):
                    backing.copy_(initial)
                owners[index].send.fill_(173)
                owners[index].received.fill_(173)
                torch.cuda.synchronize()
                dist.barrier()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                for actual, expected in zip(cases[index][0], cases[index][2], strict=True):
                    assert torch.equal(actual.cpu(), expected)
            worker.close()
            worker = None
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_native_broadcast_bytes_and_overlapping_views(tmp_path, world):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() >= world
    mp.spawn(_byte_rank, args=(world, (tmp_path / "rendezvous").as_uri()), nprocs=world, join=True)
