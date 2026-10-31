"""One-rank GLM layers reading sub-block outputs in place match the copied reductions bit for bit."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53 import decoder as decoder_module
from ignition.recipes.glm53.decoder import PreparedGlmLayer
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.stack import DecoderScratch
from ignition.recipes.glm53.state import DecoderMetadata, DecoderState
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader, MoEWeights

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned GPU and the GLM checkpoint")

# Layer groups joined by deferred post mixing: a dense KDA layer, a sparse MLA layer and a routed KDA layer;
# and a sparse MLA layer followed by two routed KDA layers.
LAYER_GROUPS = {"dense_first": (2, 3, 4), "routed": (3, 4, 5)}
CAPACITIES = (4096, 128, 8, 1)
BASELINE_RUNS = ("copied", "copied_1", "copied_2", "copied_3")
COUNTERS = ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed")


def copying_reduction(source, output, **options):
    """The reduction with a separate output, as one rank built it before consumers read producers in place."""
    return PreparedAllReduce(source, torch.empty_like(source) if output is None else output, **options)


class CaptureAttention:
    """Run the real attention and keep a copy of its output."""

    def __init__(self, real, output, pinned):
        self.real, self.output, self.pinned = real, output, pinned

    def __call__(self, value):
        output = self.real(value)
        self.pinned.copy_(output)
        return output


class PinnedAttention:
    """Stand in for attention: its output is a previously captured tensor."""

    def __init__(self, real, output, pinned):
        self.output, self.pinned = output, pinned

    def __call__(self, value):
        self.output.copy_(self.pinned)
        return self.output


def depends_on_sparse_attention(name, mla_position):
    """Whether a compared tensor is computed from the sparse-attention layer's output.

    Layers run in order and each layer's state is written from its own input. The residual is the last
    layer's output. The attention layer's own caches are written from its input, which precedes its
    attention output, so only the states of layers after it, and the residual, depend on that output.
    """
    if name == "residual":
        return True
    position = int(name.split(".")[0].removeprefix("layer"))
    return position > mla_position


def set_metadata(metadata, count):
    recurrent, indexed = metadata.recurrent, metadata.indexed
    recurrent.query_start_loc.copy_(torch.tensor([0, count], dtype=torch.int32))
    recurrent.num_tokens.fill_(count)
    recurrent.num_seqs.fill_(int(count > 0))
    recurrent.convolution_slots.copy_(torch.tensor([1], dtype=torch.int64))
    recurrent.fresh.zero_()
    recurrent.state_indices.zero_()
    recurrent.state_indices[0, 0] = 1
    recurrent.initial_state_indices[0] = recurrent.final_state_indices[0] = 1
    recurrent.checkpoint_state_indices[0] = 2
    recurrent.checkpoint_offsets[0] = count // 32 * 16
    indexed.block_table[0].copy_(torch.arange(64, dtype=torch.int32, device=indexed.block_table.device).flip(0))
    indexed.state_slots[0] = 1
    indexed.query_start_loc.copy_(torch.tensor([0, count], dtype=torch.int32))
    indexed.positions.fill_(-1)
    indexed.token_slots.fill_(-1)
    positions = torch.arange(count, dtype=torch.int64, device=indexed.positions.device)
    indexed.positions[:count].copy_(positions)
    indexed.token_slots[:count].copy_(indexed.block_table[0, positions // 256].long() * 256 + positions % 256)


@pytest.mark.parametrize("group", list(LAYER_GROUPS))
@torch.no_grad()
def test_layers_reading_producers_in_place_equal_copied_reductions(monkeypatch, group):
    LAYERS = LAYER_GROUPS[group]
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT", "/data/checkpoints/GLM-5.3-Flash-FP4-Local"))
    if not snapshot.is_dir():
        pytest.skip("GLM checkpoint is unavailable")
    device = torch.device("cuda:0")
    tree = create_model_tree(snapshot)
    modules = tree.model.language_model.layers
    shard = TensorParallelShard(0, 1)
    with GlmWeightLoader(inspect_indexed_checkpoint(snapshot), device, shard=shard,
                         prefixes=tuple(f"{TEXT_PREFIX}.layers.{index}." for index in LAYERS)) as loader:
        weights = {index: (loader.load_attention(index), loader.load_mlp(index), loader.load_residual(index))
                   for index in LAYERS}
    mla_position = next(position for position, index in enumerate(LAYERS)
                        if modules[index].block_type == "indexed_attention")
    capacity = max(CAPACITIES)
    base = DecoderScratch.allocate(capacity, 64, "prefill", device, shard=shard, sample_rows=1, overlay_layers=True)
    states = DecoderState.allocate([modules[index].block_type for index in LAYERS], 1, 6, 128, device)
    arena = ScratchArena(device)
    layers, metadata, graphs, compiled = {}, {}, {}, {}
    with RecordedPreparationSession(device=device, checkpoint=snapshot.name, workload_schema="glm-layer-alias-v1",
                                    numerical_policy="default", compile_workers=4) as session:
        session.register_scratch(arena)
        for tokens in CAPACITIES:
            scratch = base.rows(tokens, "prefill", sample_rows=1)
            metadata[tokens] = DecoderMetadata.allocate(tokens, 1, 64, device)
            for variant in ("copied", "in_place"):
                monkeypatch.setattr(decoder_module, "PreparedAllReduce",
                                    copying_reduction if variant == "copied" else PreparedAllReduce)
                built = []
                for position, index in enumerate(LAYERS):
                    attention, mlp, residual = weights[index]
                    module = modules[index]
                    indexed = module.block_type == "indexed_attention"
                    built.append(PreparedGlmLayer(module, attention, mlp, residual,
                        scratch.residual if position == 0 else built[-1].result, scratch.residual,
                        scratch.layer(module.block_type, sparse=isinstance(mlp, MoEWeights)), states.layers[position],
                        metadata[tokens].indexed if indexed else metadata[tokens].recurrent, mode="prefill",
                        runtime=None, coordinator=None, session=session, name=f"layer{index}.{variant}.{tokens}",
                        workspace=arena, carry=None if position == 0 else scratch.mlp_hc,
                        defer_post=position < len(LAYERS) - 1))
                layers[variant, tokens] = built
                if variant == "in_place":
                    for layer in built:
                        assert layer.attention_reduce.output is layer.scratch.attention.output
                        assert layer.result is layer.scratch.mlp.output
                else:
                    assert all(layer.result is not layer.scratch.mlp.output for layer in built)
        session.tune()

        def forward(variant, tokens):
            def run():
                value = base.rows(tokens, "prefill", sample_rows=1).residual
                for layer in layers[variant, tokens]:
                    value = layer(value)
            return run

        for (variant, tokens) in layers:
            compiled[variant, tokens] = compile_forward(forward(variant, tokens))
            compiled[variant, tokens]()
        pinned_values, pinned_compiled, pinned_graphs = {}, {}, {}
        for tokens in CAPACITIES:
            pinned_values[tokens] = torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=device)
            for kind, variant in (("capture", "copied"), ("pinned", "copied"), ("pinned", "in_place")):
                attention_layer = layers[variant, tokens][mla_position]
                real = attention_layer.attention
                attention_layer.attention = (CaptureAttention if kind == "capture" else PinnedAttention)(
                    real, attention_layer.scratch.attention.output, pinned_values[tokens])
                try:
                    pinned_compiled[kind, variant, tokens] = compile_forward(forward(variant, tokens))
                    pinned_compiled[kind, variant, tokens]()
                finally:
                    attention_layer.attention = real
        torch.cuda.synchronize()
        for tokens in CAPACITIES:
            graphs[tokens] = graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled["in_place", tokens]()
            pinned_graphs[tokens] = graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                pinned_compiled["pinned", "in_place", tokens]()
        session.freeze()
        initial = [(value.clone(), value) for _, value in states.named_tensors()]
        generator = torch.Generator().manual_seed(5313)
        names = ("residual", *(name for name, _ in states.named_tensors()))

        def reset(tokens, count, seed):
            for saved, value in initial:
                value.copy_(saved)
            set_metadata(metadata[tokens], count)
            base.residual[:tokens].copy_(seed.bfloat16().to(device))
            # Poison every serial scratch tensor and kernel workspace before the layers write them.
            for tensor in (base.attention_hc, base.mlp_hc, base.kda, base.mla, base.dense, base.moe):
                for value in vars(tensor).values():
                    if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                        value.fill_(float("nan"))
            arena.tensor.fill_(255)
            torch.cuda.synchronize()

        def snapshot(tokens):
            return (base.residual[:tokens].clone(), *(value.clone() for _, value in states.named_tensors()))

        def same(left, right):
            return torch.equal(left.view(torch.uint8), right.view(torch.uint8))

        try:
            for tokens in (4096, 8, 1, 128, 4096):
                for count in dict.fromkeys((tokens, max(1, tokens - 3), 0)):
                    seed = torch.randn((tokens, 4, 4096), generator=generator) * .1
                    results = {}
                    # Four baseline runs of the copied reductions measure the baseline's own run-to-run spread.
                    for variant in (*BASELINE_RUNS, "in_place", "graph"):
                        reset(tokens, count, seed)
                        if variant == "graph":
                            before = torch.cuda.memory_stats(device)
                            graphs[tokens].replay()
                            torch.cuda.synchronize()
                            after = torch.cuda.memory_stats(device)
                            assert all(before[counter] == after[counter] for counter in COUNTERS)
                        else:
                            compiled["copied" if variant in BASELINE_RUNS else variant, tokens]()
                            torch.cuda.synchronize()
                        results[variant] = snapshot(tokens)
                    for variant, values in results.items():
                        print(f"{variant}: non-finite residual elements {int((~torch.isfinite(values[0])).sum())} "
                              f"at capacity {tokens}, {count} active rows")
                    assert all(bool(torch.isfinite(results[variant][0]).all()) for variant in ("in_place", "graph"))
                    for index, name in enumerate(names):
                        reference = results["copied"][index]
                        if depends_on_sparse_attention(name, mla_position):
                            # The sparse-attention output is not repeatable run to run, so these tensors are
                            # compared exactly under the pinned output below; the spread is printed only.
                            spread = max(float((results[run][index].float() - reference.float()).abs().max())
                                         for run in BASELINE_RUNS[1:])
                            for variant in ("in_place", "graph"):
                                difference = float((results[variant][index].float() - reference.float()).abs().max())
                                print(f"{variant}: {name} (depends on sparse attention) at capacity {tokens}, "
                                      f"{count} rows: max |difference| {difference:.3e}, baseline spread over "
                                      f"{len(BASELINE_RUNS) - 1} repeats {spread:.3e}")
                            continue
                        # Written from values that precede the sparse-attention output: bitwise.
                        for variant in ("in_place", "graph"):
                            assert same(results[variant][index], reference), (
                                f"{variant}: {name} differs from copied reductions at capacity {tokens}, "
                                f"{count} active rows")
                        assert all(same(results[run][index], reference) for run in BASELINE_RUNS[1:]), (
                            f"{name} is not repeatable in the baseline at capacity {tokens}")
                    # Pin the sparse-attention output to one captured tensor: with the only non-repeatable
                    # input fixed, every variant must agree bit for bit on every tensor.
                    reset(tokens, count, seed)
                    pinned_compiled["capture", "copied", tokens]()
                    torch.cuda.synchronize()
                    assert bool(torch.isfinite(pinned_values[tokens][:count]).all())
                    pinned_results = {}
                    for label, variant in (("copied", "copied"), ("copied_again", "copied"),
                                           ("in_place", "in_place"), ("graph", "graph")):
                        reset(tokens, count, seed)
                        if variant == "graph":
                            pinned_graphs[tokens].replay()
                        else:
                            pinned_compiled["pinned", variant, tokens]()
                        torch.cuda.synchronize()
                        pinned_results[label] = snapshot(tokens)
                    for index, name in enumerate(names):
                        copied = pinned_results["copied"][index]
                        assert same(pinned_results["copied_again"][index], copied), (
                            f"pinned baseline {name} is not repeatable at capacity {tokens}, {count} active rows")
                        for variant in ("in_place", "graph"):
                            assert same(pinned_results[variant][index], copied), (
                                f"pinned {variant} {name} differs from copied reductions at capacity {tokens}, "
                                f"{count} active rows")
                    print(f"pinned attention output at capacity {tokens}, {count} active rows: "
                          f"all {len(names)} tensors bitwise equal")
        finally:
            for graph in (*graphs.values(), *pinned_graphs.values()):
                graph.reset()
