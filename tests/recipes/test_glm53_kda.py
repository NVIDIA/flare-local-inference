"""GLM KDA state continuation through compiled graphs and native replay."""

import os
import time
from pathlib import Path

import pytest
import torch
from cuda.bindings import driver as cuda

from ignition.components.parallel import TensorParallelShard
from ignition.components.recurrent import DeltaMetadata, DeltaState
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.kda import KDAScratch, PreparedGlmKDA
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader


def runs_concurrently(graph):
    """Whether two nodes of a retained captured graph can run at once: unordered by any dependency path.

    Redundant dependencies, such as an event wait on the stream that recorded it, are not concurrency.
    """
    raw = cuda.CUgraph(graph.raw_cuda_graph())
    status, _, _, _, count = cuda.cuGraphGetEdges(raw)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, parents, children, _, _ = cuda.cuGraphGetEdges(raw, count)
    assert status == cuda.CUresult.CUDA_SUCCESS
    status, nodes, _ = cuda.cuGraphGetNodes(raw, cuda.cuGraphGetNodes(raw)[2])
    assert status == cuda.CUresult.CUDA_SUCCESS
    edges = [(int(parent), int(child)) for parent, child in zip(parents, children, strict=True)]
    waiting = {int(node): 0 for node in nodes}
    for _, child in edges:
        waiting[child] += 1
    ready = [node for node, count in waiting.items() if not count]
    while len(ready) == 1:
        node = ready.pop()
        for parent, child in edges:
            if parent == node:
                waiting[child] -= 1
                if not waiting[child]:
                    ready.append(child)
    return bool(ready)


@pytest.mark.parametrize("world", [2, 4])
def test_rank_state_and_scratch_geometry(world):
    heads = 64 // world
    state = DeltaState.allocate(6, "cpu", heads=heads, channels=3 * heads * 128)
    scratch = KDAScratch.allocate(65, heads, "cpu")
    assert state.recurrent.shape == (6, heads, 128, 128)
    assert state.convolution.shape == (6, 3 * heads * 128, 3)
    assert scratch.rows(1).projected_qkv.data_ptr() == scratch.projected_qkv.data_ptr()
    assert scratch.rows(1).output.shape == (1, 4096)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                   reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
# Without stream overlap the output-gate projections follow the input projections on the main stream.
@pytest.mark.parametrize("world,tokens,decode_capacity,overlap", [
    (1, 4096, 2, True), (2, 65, 2, True), (2, 65, 1, True), (4, 129, 2, True),
    (4, 4096, 2, True), (2, 65, 2, False), (4, 129, 2, False)])
@torch.no_grad()
def test_actual_kda_native_prefill_and_decode(world, tokens, decode_capacity, overlap):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    manifest = inspect_indexed_checkpoint(snapshot)
    model = create_model_tree(snapshot)
    module = model.model.language_model.layers[0].self_attn
    device = torch.device("cuda", 0)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, world),
                         prefixes=(f"{TEXT_PREFIX}.layers.0.self_attn.",)) as loader:
        weights = loader.load_attention(0)
    heads = 64 // world
    state = DeltaState.allocate(6, device, heads=heads, channels=3 * heads * 128)
    arena = ScratchArena(device, overlap=overlap)
    metadata = {mode: DeltaMetadata.allocate(2, device) for mode in ("prefill", "decode")}
    prefill_capacity = 1 << (tokens - 1).bit_length()
    inputs = {mode: torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
              for mode, capacity in (("prefill", prefill_capacity), ("decode", decode_capacity))}
    working = {mode: torch.zeros_like(source) for mode, source in inputs.items()}
    scratch = {mode: KDAScratch.allocate(source.shape[0], heads, device) for mode, source in inputs.items()}
    graphs, operators, compiled = {}, {}, {}
    worker = None
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                    workload_schema="glm-kda-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        for mode, source in working.items():
            operators[mode] = PreparedGlmKDA(module, weights, source, scratch[mode], state, metadata[mode],
                mode=mode, session=session, name=f"kda.{mode}", workspace=arena, checkpoint_export=mode == "prefill")
            assert (operators[mode].gate_overlap is not None) is overlap
        session.tune()
        try:
            for mode, operator in operators.items():
                source = inputs[mode]

                def forward(operator=operator, source=source, active=working[mode]):
                    active.copy_(source)
                    operator(active)
                    arena.tensor.fill_(255)
                    active.zero_()

                compiled[mode] = compile_forward(forward)
                compiled[mode]()
                torch.cuda.synchronize()
                graphs[mode] = torch.cuda.CUDAGraph(keep_graph=True)
                with session.capture(), torch.cuda.graph(graphs[mode]):
                    compiled[mode]()
                graphs[mode].instantiate()
                # The output-gate chain runs beside the input projections, or every node runs in order.
                assert runs_concurrently(graphs[mode]) is overlap
                torch.cuda.synchronize()
            session.freeze()
            control = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                       "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                       "completion_records": torch.zeros((2, 4, 8), dtype=torch.int64, device=device)}
            bundle = PreparedExecution(device=0, graphs=graphs, buffers=control,
                signatures={"prefill": GraphSignature("prefill", 1, prefill_capacity),
                            "decode": GraphSignature("decode", decode_capacity, 1)},
                resources=(compiled, operators, arena, session, inputs, working, metadata, scratch))
            worker = bundle.start_executor()
            generator = torch.Generator().manual_seed(53)
            for mode, count, repetitions in (("prefill", tokens, 1), ("decode", 1, 8), ("prefill", 3, 1)):
                source, meta = inputs[mode], metadata[mode]
                source.zero_()
                source[:count].copy_((torch.randn((count, 4096), generator=generator) * .1).to(torch.bfloat16))
                meta.query_start_loc.copy_(torch.tensor([0, count, count], dtype=torch.int32))
                meta.convolution_slots.copy_(torch.tensor([1, -1], dtype=torch.int64))
                meta.fresh.fill_(mode == "prefill" and count == tokens)
                meta.state_indices[0, 0] = 1
                meta.initial_state_indices[0] = 0 if count == tokens else 1
                meta.final_state_indices[0] = 1
                meta.checkpoint_state_indices[0] = 2 if count >= 16 else 0
                meta.checkpoint_offsets[0] = 16 if count >= 16 else 0
                meta.num_seqs.fill_(1)
                meta.num_tokens.fill_(count)
                initial = (state.convolution.clone(), state.recurrent.clone())
                for _ in range(repetitions):
                    operators[mode](source)
                expected = (scratch[mode].output.clone(), state.convolution.clone(), state.recurrent.clone())
                state.convolution.copy_(initial[0])
                state.recurrent.copy_(initial[1])
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                for _ in range(repetitions):
                    ticket = worker.submit(bundle.graph_indices[mode], [])
                    assert ticket is not None
                    deadline = time.monotonic() + 120
                    completion = worker.poll()
                    while completion is None and time.monotonic() < deadline:
                        time.sleep(.001)
                        completion = worker.poll()
                    assert completion is not None and completion.fence == ticket
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for actual, reference in zip((scratch[mode].output, state.convolution, state.recurrent), expected, strict=True):
                    assert bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert not state.recurrent[0].count_nonzero()
                assert not state.recurrent[3:].count_nonzero()
            worker.close()
            worker = None
        finally:
            if worker is not None:
                worker.close()
            for graph in graphs.values():
                graph.reset()
