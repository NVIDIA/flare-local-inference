"""Checkpoint GLM latent attention composes into allocation-stable native graphs."""

import os
import time
from pathlib import Path

import pytest
import torch
from b12x.attention import sparse_mla
from cuda.bindings import driver as cuda

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.indexer import IndexMetadata
from ignition.recipes.glm53.mla import MLAScratch, MLAState, PreparedGlmMLA
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


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


def assert_selection(actual, expected, indexer):
    differing = (actual.sort().values != expected.sort().values).any(1)
    if not differing.any():
        return
    assert indexer.mode == "prefill"
    # Prefill selects in row slices; recompute each slice's tile logits for its differing rows.
    logits = []
    for selector in indexer.selectors:
        selector()
        runtime = selector.binding.runtime
        block_k = runtime.prefill_block_k
        tiles = runtime.supertile_k // block_k
        logits.extend((runtime.scratch.get_indexer_contiguous_tile_logits()
                       .reshape(-1, tiles, 32, block_k).permute(0, 2, 1, 3)
                       .reshape(-1, tiles * block_k)[:selector.output.shape[0]]).clone())
    assert len(logits) == actual.shape[0]
    for row in differing.nonzero().flatten().tolist():
        observed, reference = actual[row].tolist(), expected[row].tolist()
        live = [index for index in observed if index >= 0]
        expected_live = [index for index in reference if index >= 0]
        assert len(live) == len(expected_live) == len(set(live))
        assert observed.count(-1) == reference.count(-1)
        length = int(indexer.scratch.pool_lengths[row])
        assert all(0 <= index < length for index in live)
        changed = sorted(set(live) ^ set(expected_live))
        cutoff = logits[row][expected_live].min()
        # CTA arrival order can choose different candidates tied at the cutoff.
        assert torch.all(logits[row][changed] == cutoff)


@pytest.fixture(scope="module", params=[1, 2, 4])
def model_weights(request):
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    tree = create_model_tree(snapshot)
    manifest = inspect_indexed_checkpoint(snapshot)
    world = request.param
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(world - 1, world),
            prefixes=(f"{TEXT_PREFIX}.layers.3.self_attn.",)) as loader:
        weights = loader.load_attention(3)
    return tree.model.language_model.layers[3].self_attn, weights, world


@pytest.mark.parametrize("mode,rows,overlap", [("decode", 1, True), ("decode", 2, True), ("decode", 1, False),
                                               ("decode", 2, False), ("prefill", 65, True), ("prefill", 4096, True)])
@torch.no_grad()
def test_glm_mla_checkpoint_native_replay(model_weights, mode, rows, overlap):
    module, weights, world = model_weights
    device = torch.device("cuda:0")
    capacity = 1 << (rows - 1).bit_length()
    sequences = rows if mode == "decode" else 1
    source = torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
    scratch = MLAScratch.allocate(capacity, 64 // world, 64, mode, device)
    state = MLAState.allocate(128, 4, device)
    metadata = IndexMetadata.allocate(capacity, sequences, 64, device)
    arena = ScratchArena(device, overlap=overlap)
    graph, worker = torch.cuda.CUDAGraph(keep_graph=True), None
    with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
            workload_schema="glm-mla-v1", numerical_policy="default") as session:
        try:
            session.register_scratch(arena)
            attention = PreparedGlmMLA(module, weights, source, scratch, state, metadata,
                mode=mode, session=session, name="mla", workspace=arena)
            # Decode projects and absorbs queries on an auxiliary stream unless overlap is disabled.
            assert (attention.query_chain is not None) is (overlap and mode == "decode")
            session.tune()

            value_expansion = attention.value_expansion
            def record_attended(values):
                # Observe the attention output before serial scratch reuse overwrites it.
                scratch.attended.copy_(values.transpose(0, 1))
                return value_expansion(values)
            attention.value_expansion = record_attended

            def forward():
                attention(source)
                arena.tensor.fill_(255)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graph.instantiate()
            assert runs_concurrently(graph) is (overlap and mode == "decode")
            session.freeze()
            control = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                       "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                       "completion_records": torch.zeros((2, 4, 8), dtype=torch.int64, device=device)}
            bundle = PreparedExecution(device=0, graphs={mode: graph}, buffers=control,
                signatures={mode: GraphSignature(mode, sequences, capacity if mode == "prefill" else 1)},
                resources=(compiled, attention, session, arena, source, scratch, state, metadata))
            worker = bundle.start_executor()
            state.cache.zero_()
            state.index.cache.zero_()
            state.index.tail.normal_(std=.1)
            metadata.block_table.copy_(torch.randperm(128, device=device, dtype=torch.int32)
                                       .reshape(2, 64)[:sequences])
            metadata.state_slots.copy_(torch.tensor([2, 1][:sequences], dtype=torch.int64, device=device))
            metadata.query_start_loc.copy_(torch.tensor(list(range(rows + 1)) if mode == "decode" else [0, rows],
                                                         dtype=torch.int32, device=device))
            if mode == "decode":
                metadata.requests.copy_(torch.arange(rows, dtype=torch.int32, device=device))
            stages = {key: getattr(scratch, key) for key in (
                "query_input", "query_residual", "latent_input", "latent", "query", "absorbed_query",
                "attended", "expanded", "output")}
            persistent = (state.cache, state.index.cache, state.index.tail)
            mismatches = []
            for step in range(3):
                source[:rows].normal_(std=.1)
                if mode == "decode":
                    metadata.positions.copy_(torch.tensor([255 + step, 3 + step][:rows],
                                                         dtype=torch.int64, device=device))
                else:
                    metadata.positions[:rows].copy_(torch.arange(rows, device=device, dtype=torch.int64) + 3 + step * rows)
                metadata.token_slots[:rows].copy_(
                    metadata.block_table[metadata.requests[:rows].long(), metadata.positions[:rows] // 256].long()
                    * 256 + metadata.positions[:rows] % 256)
                if mode == "decode" and step == 2:
                    metadata.state_slots[-1] = -1
                    metadata.positions[-1] = -1
                    metadata.token_slots[-1] = -1
                initial = tuple(value.clone() for value in persistent)
                forward()
                expected = {key: value.clone() for key, value in stages.items()}
                expected_state = tuple(value.clone() for value in persistent)
                expected_pools = scratch.index.pools.clone()
                expected_lengths = scratch.index.selected_lengths.clone()
                for value, saved in zip(persistent, initial, strict=True):
                    value.copy_(saved)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                ticket = worker.submit(0, [])
                assert ticket is not None
                deadline = time.monotonic() + 120
                completion = worker.poll()
                while completion is None and time.monotonic() < deadline:
                    time.sleep(.001)
                    completion = worker.poll()
                assert completion is not None and completion.fence == ticket
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                for value, expected_value in zip(persistent, expected_state, strict=True):
                    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
                torch.testing.assert_close(scratch.index.selected_lengths, expected_lengths, rtol=0, atol=0)
                if mode == "decode" and step == 2:
                    assert not scratch.output[-1].count_nonzero()
                actual = {key: value.clone() for key, value in stages.items()}
                actual_pools, actual_physical = scratch.index.pools.clone(), scratch.index.physical.clone()
                # Attention reduction must use the native selector's candidate order.
                attention.attention()
                attention.value_expansion(scratch.attended.transpose(0, 1))
                attention.projections["o_proj"](scratch.expanded.view(capacity, -1))
                for key in ("attended", "expanded", "output"):
                    expected[key] = stages[key].clone()
                assert_selection(actual_pools, expected_pools, attention.indexer)
                scratch.index.pools.copy_(actual_pools)
                if attention.indexer.logical:
                    from ignition.recipes.glm53.dcp import expand_tokens

                    expand_tokens(scratch.index.pools, metadata.positions,
                                  scratch.index.physical, scratch.index.selected_lengths)
                else:
                    sparse_mla.expand_pooled_topk_to_physical_slots(*attention.indexer.expand_arguments,
                        **attention.indexer.expand_options, plan=attention.indexer.expand_plan)
                torch.testing.assert_close(scratch.index.physical, actual_physical, rtol=0, atol=0)
                for key, value in actual.items():
                    assert torch.isfinite(value).all(), key
                    if not torch.equal(value, expected[key]):
                        difference = (value.float() - expected[key].float()).abs()
                        mismatches.append(f"step {step}, {key}: {difference.count_nonzero().item()} elements differ; "
                                          f"maximum absolute error {difference.max().item()}")
                        del difference
            assert not mismatches, "\n".join(mismatches)
        finally:
            if worker is not None:
                worker.close()
            graph.reset()
