"""One-rank GLM layers read each sub-block's output in place; the storage they leave live is never rewritten."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.glm53 import decoder as decoder_module
from ignition.recipes.glm53.decoder import LayerScratch, PreparedGlmLayer
from ignition.recipes.glm53.stack import DecoderScratch


def tensors(value):
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, torch.Tensor):
            if item.numel():
                yield item
        elif item is not None and hasattr(item, "__dataclass_fields__"):
            yield from tensors(item)


def span(tensor):
    return tensor.data_ptr(), tensor.data_ptr() + max(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())) * tensor.element_size() + tensor.element_size()


def overlaps(left, right):
    (a, b), (c, d) = span(left), span(right)
    return a < d and c < b


@pytest.mark.parametrize("tokens", [4096, 128, 8, 1])
def test_ops_between_a_producer_and_its_consumer_write_no_overlaid_layer_scratch(tokens):
    # Between an attention output and the FFN mixing that reads it, and between an MLP output and the
    # following mixing or post operation that reads it, the layer writes only mixing scratch and the residual.
    base = DecoderScratch.allocate(4096, 8, "prefill", "cpu", shard=TensorParallelShard(0, 1), sample_rows=1,
                                   overlay_layers=True)
    scratch = base.rows(tokens, "prefill", sample_rows=1)
    written = [*tensors(scratch.attention_hc), *tensors(scratch.mlp_hc), scratch.residual]
    layers = [tensor for group in (scratch.kda, scratch.mla, scratch.dense, scratch.moe) for tensor in tensors(group)]
    assert layers
    for tensor in written:
        assert not any(overlaps(tensor, layer) for layer in layers)


def stand_in(monkeypatch):
    calls = {}

    class Recorder:
        def __init__(self, inputs, *rest, name=None, **_):
            self.inputs, self.name = inputs, name
            calls[name.rsplit(".", 1)[-1]] = self
            self.output = getattr(self, "output", None)

        def __call__(self, *args):
            return self.output

    class Attention(Recorder):
        def __init__(self, module, weights, source, scratch, *rest, name=None, **_):
            super().__init__(source, name=name)
            self.output = scratch.output

    class Mlp(Recorder):
        def __init__(self, source, weights, gate_up, intermediate, output, *, name=None, **_):
            super().__init__(source, name=name)
            self.output = output

    class Post(Recorder):
        def __init__(self, source, residual, post, comb, output, *, name=None, **_):
            super().__init__(source, name=name)
            self.output = output

    monkeypatch.setattr(decoder_module, "PreparedMHC", Recorder)
    monkeypatch.setattr(decoder_module, "PreparedGlmKDA", Attention)
    monkeypatch.setattr(decoder_module, "PreparedMLP", Mlp)
    monkeypatch.setattr(decoder_module, "PreparedMHCPost", Post)
    return calls


def module_stub():
    norm = SimpleNamespace(eps=1e-6, variance_epsilon=1e-6)
    hc = SimpleNamespace(input_norm=norm, hc_eps=1e-6, hc_sinkhorn_iters=20)
    return SimpleNamespace(block_type="linear_attention", attn_hc=hc,
        ffn_hc=SimpleNamespace(input_norm=norm, hc_eps=1e-6, hc_sinkhorn_iters=20),
        input_layernorm=SimpleNamespace(variance_epsilon=1e-6), post_attention_layernorm=SimpleNamespace(),
        self_attn=None, mlp=SimpleNamespace(swiglu_limit=7.0))


def residual_weights():
    return {f"{prefix}_{name}": torch.zeros(1) for prefix in ("hc_attn", "hc_ffn") for name in ("fn", "scale", "base")} | {
        "input_layernorm.weight": torch.zeros(1), "post_attention_layernorm.weight": torch.zeros(1)}


@pytest.mark.parametrize("defer_post", [False, True])
def test_one_rank_layer_consumers_read_the_producer_outputs(monkeypatch, defer_post):
    calls = stand_in(monkeypatch)
    scratch = LayerScratch.allocate(8, 1, "cpu", sparse=False, block_type="linear_attention", mode="prefill")
    source, output = torch.zeros((8, 4, 4096), dtype=torch.bfloat16), torch.zeros((8, 4, 4096), dtype=torch.bfloat16)
    layer = PreparedGlmLayer(module_stub(), {}, {}, residual_weights(), source, output, scratch, None, None,
        mode="prefill", runtime=None, coordinator=None, session=None, name="layer", workspace=None,
        defer_post=defer_post)
    assert layer.attention_reduce.output is scratch.attention.output
    assert calls["ffn_hc"].inputs[0] is scratch.attention.output
    assert layer.result is scratch.mlp.output
    if defer_post:
        assert layer.post is None
    else:
        assert calls["post"].inputs is scratch.mlp.output


def test_ranks_reduce_into_the_layer_outputs(monkeypatch):
    # Tensor-parallel layers keep the separate reduction outputs; the reduction itself is built by the runtime.
    calls = stand_in(monkeypatch)
    built = []

    class Reduction:
        def __init__(self, source, output, **_):
            built.append((source, output))
            self.output = output

        def __call__(self, source):
            return self.output

    monkeypatch.setattr(decoder_module, "PreparedAllReduce", Reduction)
    scratch = LayerScratch.allocate(8, 2, "cpu", sparse=False, block_type="linear_attention", mode="prefill")
    source, output = torch.zeros((8, 4, 4096), dtype=torch.bfloat16), torch.zeros((8, 4, 4096), dtype=torch.bfloat16)
    layer = PreparedGlmLayer(module_stub(), {}, {}, residual_weights(), source, output, scratch, None, None,
        mode="prefill", runtime=object(), coordinator=None, session=None, name="layer", workspace=None)
    assert built[0][1] is scratch.attention_output and built[1][1] is scratch.mlp_output
    assert calls["ffn_hc"].inputs[0] is scratch.attention_output
    assert layer.result is scratch.mlp_output and calls["post"].inputs is scratch.mlp_output


@pytest.mark.parametrize("count", [0, 1, 5, 8])
def test_alias_gpu_test_metadata_matches_decoder_metadata_shapes(count):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).with_name("test_glm53_layer_reduction_alias_cuda.py")
    spec = importlib.util.spec_from_file_location("alias_cuda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from ignition.recipes.glm53.state import DecoderMetadata

    metadata = DecoderMetadata.allocate(8, 1, 64, "cpu")
    module.set_metadata(metadata, count)
    assert metadata.recurrent.query_start_loc.tolist() == [0, count]
    assert int(metadata.recurrent.num_tokens) == count
    assert metadata.indexed.positions.tolist() == list(range(count)) + [-1] * (8 - count)


def _alias_cuda_module():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).with_name("test_glm53_layer_reduction_alias_cuda.py")
    spec = importlib.util.spec_from_file_location("alias_cuda_classification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("group,mla_position,dependent", [
    ("dense_first", 1, {"residual", "layer2.convolution", "layer2.recurrent"}),
    ("routed", 0, {"residual", "layer1.convolution", "layer1.recurrent", "layer2.convolution", "layer2.recurrent"}),
])
def test_only_tensors_downstream_of_the_attention_output_use_the_spread_bound(group, mla_position, dependent):
    module = _alias_cuda_module()
    layers = module.LAYER_GROUPS[group]
    assert [position for position, index in enumerate(layers) if index == 3] == [mla_position]
    names = ["residual"]
    for position, index in enumerate(layers):
        names += ([f"layer{position}.{field}" for field in ("latent_cache", "index_cache", "index_tail")]
                  if index == 3 else [f"layer{position}.convolution", f"layer{position}.recurrent"])
    assert {name for name in names if module.depends_on_sparse_attention(name, mla_position)} == dependent
