"""Checkpoint GLM MLA part-output layout matches the row-concatenated layout bit for bit."""

import os
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from ignition.components.block_fp8 import BlockFP8Projection
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import GroupedLinearWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53 import mla as mla_module
from ignition.recipes.glm53.indexer import IndexMetadata
from ignition.recipes.glm53.mla import MLAScratch, MLAState, PreparedGlmMLA
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned GPU and the GLM checkpoint")

CAPACITIES = (4096, 128, 8, 1)
FIELDS = ("query_input", "latent_input", "query_residual", "latent", "query", "attended", "expanded", "output",
          "selected_lengths", "cache", "index_cache", "index_tail", "physical")


def same(left, right):
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def row_concatenated(scratch, weight):
    """The layout of a single allocation whose parts are column slices of every row."""
    consumers = [scratch.projected_input[:, start:end] for start, end in mla_module._QKV_A_COLUMNS]
    return scratch, scratch.projected_input, consumers[2], consumers[3]


def storage_of(scratch):
    return [getattr(scratch, f.name) for f in fields(scratch)
            if f.name not in ("projected_input", "index") and isinstance(getattr(scratch, f.name), torch.Tensor)
            and getattr(scratch, f.name).numel()]


@torch.no_grad()
def test_part_output_layout_equals_row_concatenated_layout_across_capacities(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT", "/data/checkpoints/GLM-5.3-Flash-FP4-Local"))
    if not snapshot.is_dir():
        pytest.skip("GLM checkpoint is unavailable")
    device = torch.device("cuda:0")
    module = create_model_tree(snapshot).model.language_model.layers[3].self_attn
    with GlmWeightLoader(inspect_indexed_checkpoint(snapshot), device, shard=TensorParallelShard(0, 1),
                         prefixes=(f"{TEXT_PREFIX}.layers.3.self_attn.",)) as loader:
        weights = loader.load_attention(3)
    if not isinstance(weights["qkv_a_proj"], GroupedLinearWeights):
        pytest.skip("checkpoint has a single layer-input projection format")
    capacity = max(CAPACITIES)
    source = torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device, overlap=False)
    variants = {}
    with RecordedPreparationSession(device=device, checkpoint=snapshot.name, workload_schema="mla-part-outputs-v1",
                                    numerical_policy="default", compile_workers=4) as session:
        session.register_scratch(arena)
        for layout, builder in (("parts", mla_module._input_projection_views), ("rows", row_concatenated)):
            monkeypatch.setattr(mla_module, "_input_projection_views", builder)
            parent = MLAScratch.allocate(capacity, 64, 64, "prefill", device)
            state = MLAState.allocate(128, 4, device)
            for tokens in CAPACITIES:
                metadata = IndexMetadata.allocate(tokens, 1, 64, device)
                operator = PreparedGlmMLA(module, weights, source[:tokens], parent.rows(tokens, "prefill"), state,
                    metadata, mode="prefill", session=session, name=f"mla.{layout}.{tokens}", workspace=arena)
                variants[layout, tokens] = (operator, metadata)
                blocks = [p for p in operator.projections["qkv_a_proj"].projections
                          if isinstance(p, BlockFP8Projection)]
                assert blocks
                print(f"{layout} layout, capacity {tokens}: block-FP8 part in_place =",
                      [p.in_place for p in blocks], "widths", [p.weight.values.shape[0] for p in blocks])
                # Contiguous aligned part outputs take the in-place GEMM path; column slices of several rows
                # copy a result (one row of a slice is contiguous). Only the SM103 GEMM takes an output.
                sm103 = torch.cuda.get_device_capability(0) == (10, 3)
                assert all(p.in_place is (sm103 and (layout == "parts" or tokens == 1)) for p in blocks)
            variants[layout] = parent, state
        session.tune()
        compiled = {}
        for key, (operator, _) in [(k, v) for k, v in variants.items() if len(k) == 2]:
            compiled[key] = compile_forward(lambda op=operator, x=source[:key[1]]: op(x))
        graphs = {}
        for tokens in CAPACITIES:
            graphs[tokens] = graph = torch.cuda.CUDAGraph()
            compiled["parts", tokens]()
            torch.cuda.synchronize()
            with session.capture(), torch.cuda.graph(graph):
                compiled["parts", tokens]()
        session.freeze()
        generator = torch.Generator().manual_seed(5312)
        try:
            for tokens in (4096, 8, 1, 128, 4096):
                for count in dict.fromkeys((tokens, max(1, tokens - 3), 0)):
                    results = {}
                    # Row-concatenated layout twice (its run-to-run repeatability), then part outputs eagerly and
                    # by graph replay.
                    for run, layout, replay in (("rows", "rows", False), ("rows_again", "rows", False),
                                                ("parts_eager", "parts", False), ("parts", "parts", True)):
                        operator, metadata = variants[layout, tokens]
                        parent, state = variants[layout]
                        generator.manual_seed(count * 7 + tokens)
                        source.copy_((torch.randn(source.shape, generator=generator) * .1).bfloat16().to(device))
                        state.cache.zero_()
                        state.index.cache.zero_()
                        state.index.tail.zero_()
                        metadata.block_table.copy_(torch.arange(64, dtype=torch.int32, device=device).flip(0)[None])
                        metadata.state_slots[0] = 1
                        metadata.query_start_loc.copy_(torch.tensor([0, count], dtype=torch.int32))
                        metadata.positions.fill_(-1)
                        metadata.token_slots.fill_(-1)
                        metadata.positions[:count].copy_(torch.arange(count, dtype=torch.int64, device=device))
                        metadata.token_slots[:count].copy_(
                            metadata.block_table[0, metadata.positions[:count] // 256].long() * 256
                            + metadata.positions[:count] % 256)
                        for value in storage_of(operator.scratch):
                            value.fill_(float("nan") if value.is_floating_point() else 0)
                        arena.tensor.fill_(255)
                        if replay:
                            before = torch.cuda.memory_stats()
                            graphs[tokens].replay()
                            torch.cuda.synchronize()
                            after = torch.cuda.memory_stats()
                            for counter in ("allocation.all.allocated", "allocation.all.freed",
                                            "segment.all.allocated", "segment.all.freed"):
                                assert before[counter] == after[counter]
                        else:
                            compiled[layout, tokens]()
                        torch.cuda.synchronize()
                        s = operator.scratch
                        results[run] = dict(zip(FIELDS, (value.clone() for value in (
                            s.query_input, s.latent_input, s.query_residual, s.latent, s.query, s.attended,
                            s.expanded, s.output, s.index.selected_lengths, state.cache, state.index.cache,
                            state.index.tail, s.index.physical.sort(-1).values)), strict=True))
                    reference = results["rows"]
                    repeatable = [name for name in FIELDS if same(results["rows_again"][name], reference[name])]
                    unstable = sorted(set(FIELDS) - set(repeatable))
                    if unstable:
                        print(f"capacity {tokens}, {count} active rows: run-to-run variation in {unstable}")
                    for run in ("parts_eager", "parts"):
                        for name in repeatable:
                            assert same(results[run][name], reference[name]), \
                                f"{run}: {name} differs at capacity {tokens}, {count} active rows"
                        # Sparse attention is not bitwise repeatable at this capacity even with an unchanged layout,
                        # so a layout may differ from the baseline by no more than the baseline differs from itself.
                        for name in unstable:
                            spread = (results["rows_again"][name].float() - reference[name].float()).abs().max()
                            difference = (results[run][name].float() - reference[name].float()).abs().max()
                            print(f"{run}: {name} max |difference| {float(difference):.3e}, baseline repeat spread "
                                  f"{float(spread):.3e}")
                            assert difference <= 4 * spread, \
                                f"{run}: {name} differs beyond run-to-run variation at capacity {tokens}"
        finally:
            for graph in graphs.values():
                graph.reset()
