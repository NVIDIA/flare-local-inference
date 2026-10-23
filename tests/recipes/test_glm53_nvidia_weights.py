"""NVIDIA GLM checkpoint partitions preserve packed values, scales, and plain weights."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ignition.components.block_fp8 import BlockFP8Weight
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import GroupedLinearWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.glm53.weights import GlmWeightLoader, TEXT_PREFIX


@pytest.fixture(scope="module")
def manifest():
    path = os.environ.get("FLARE_GLM53_SNAPSHOT")
    if path is None or not Path(path).is_dir():
        pytest.skip("set FLARE_GLM53_SNAPSHOT to the NVIDIA GLM checkpoint")
    snapshot = Path(path)
    declarations = json.loads((snapshot / "config.json").read_text())["quantization_config"]["quantized_layers"]
    if declarations.get("lm_head", {}).get("quant_algo") != "MXFP8":
        pytest.skip("requires the NVIDIA mixed-precision GLM checkpoint")
    return inspect_indexed_checkpoint(snapshot)


def read(manifest, name, extent=None):
    with safe_open(str(manifest.snapshot / manifest.tensors[name].shard), framework="pt") as source:
        return (source.get_tensor(name) if extent is None else source.get_slice(name)[extent]).clone()


def assert_projection(manifest, actual, names, rows=None, columns=None):
    rows = (None,) * len(names) if rows is None else rows
    weights, scales = [], []
    for name, row_range in zip(names, rows, strict=True):
        r = slice(None) if row_range is None else slice(*row_range)
        c = slice(None) if columns is None else slice(*columns)
        weights.append(read(manifest, name + ".weight", (r, c)))
        if not isinstance(actual, torch.Tensor):
            raw = read(manifest, name + ".weight_scale")
            block = 128 if isinstance(actual, BlockFP8Weight) else 32
            if block == 128:
                raw = raw.reshape(raw.shape[0], -1)
                r = slice(None) if row_range is None else slice(row_range[0] // 128, row_range[1] // 128)
            c = slice(None) if columns is None else slice(columns[0] // block, columns[1] // block)
            scales.append(raw[r, c])
    expected = torch.cat(weights)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.cpu(), expected)
        return
    values = actual.values if isinstance(actual, BlockFP8Weight) else actual.weight.values
    stored_scales = actual.scales if isinstance(actual, BlockFP8Weight) else actual.weight.scale_rows
    assert torch.equal(values.cpu().view(torch.uint8).reshape(expected.shape), expected.view(torch.uint8))
    expected_scales = torch.cat(scales)
    assert torch.equal(stored_scales.cpu().view(torch.uint8).reshape(expected_scales.view(torch.uint8).shape),
                       expected_scales.view(torch.uint8))


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@cuda
@pytest.mark.parametrize("world,rank", [(1, 0), (2, 1), (4, 3)])
def test_nvidia_attention_and_mlp_partitions(manifest, world, rank):
    shard = TensorParallelShard(rank, world)
    kda_prefix = f"{TEXT_PREFIX}.layers.0.self_attn"
    mla_prefix = f"{TEXT_PREFIX}.layers.3.self_attn"
    dense_prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    shared_prefix = f"{TEXT_PREFIX}.layers.3.mlp.shared_experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
            prefixes=tuple(prefix + "." for prefix in (kda_prefix, mla_prefix, dense_prefix, shared_prefix))) as loader:
        kda, mla, dense = loader.load_attention(0), loader.load_attention(3), loader.load_mlp(0)
        extent = shard.partition(2048, alignment=128)
        shared_up = loader.load_linear_group((shared_prefix + ".gate_proj", shared_prefix + ".up_proj"),
                                             row_ranges=(extent, extent))
        shared_down = loader.load_linear(shared_prefix + ".down_proj", columns=extent)
        assert loader._session.torch_copy_bytes == 0
    channels, heads = shard.partition(8192), shard.partition(64)
    assert isinstance(kda["input_proj"], GroupedLinearWeights)
    assert kda["input_proj"].widths == (3 * 8192 // world, 64 // world + 128)
    assert_projection(manifest, kda["input_proj"].parts[0],
                      tuple(kda_prefix + "." + name for name in ("q_proj", "k_proj", "v_proj")), (channels,) * 3)
    assert_projection(manifest, kda["input_proj"].parts[1],
                      (kda_prefix + ".b_proj", kda_prefix + ".f_a_proj"), (heads, None))
    for name in ("f_b_proj", "g_b_proj"):
        assert_projection(manifest, kda[name], (kda_prefix + "." + name,), (channels,))
    assert_projection(manifest, kda["g_a_proj"], (kda_prefix + ".g_a_proj",))
    assert_projection(manifest, kda["o_proj"], (kda_prefix + ".o_proj",), columns=channels)
    assert isinstance(mla["qkv_a_proj"], GroupedLinearWeights)
    assert mla["qkv_a_proj"].widths == (2048, 160)
    assert_projection(manifest, mla["qkv_a_proj"].parts[0],
                      (mla_prefix + ".q_a_proj", mla_prefix + ".kv_a_proj_with_mqa"))
    assert_projection(manifest, mla["qkv_a_proj"].parts[1],
                      (mla_prefix + ".indexer.wk", mla_prefix + ".indexer.weights_proj"))
    for name, size in (("q_b_proj", 16384), ("kv_b_proj", 32768)):
        assert_projection(manifest, mla[name], (mla_prefix + "." + name,), (shard.partition(size),))
    assert_projection(manifest, mla["o_proj"], (mla_prefix + ".o_proj",), columns=shard.partition(16384))
    assert_projection(manifest, mla["indexer.wq_b"], (mla_prefix + ".indexer.wq_b",))
    dense_extent = shard.partition(12288)
    assert_projection(manifest, dense.gate_up, (dense_prefix + ".gate_proj", dense_prefix + ".up_proj"),
                      (dense_extent, dense_extent))
    assert_projection(manifest, dense.down, (dense_prefix + ".down_proj",), columns=dense_extent)
    assert_projection(manifest, shared_up, (shared_prefix + ".gate_proj", shared_prefix + ".up_proj"),
                      (extent, extent))
    assert_projection(manifest, shared_down, (shared_prefix + ".down_proj",), columns=extent)


@cuda
def test_nvidia_vocabulary_and_draft_attention(manifest):
    prefix = f"{TEXT_PREFIX}.layers.45.self_attn"
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=(prefix + ".", "lm_head.")) as loader:
        head = loader.load_linear("lm_head", rows=(128, 384))
        draft = loader.load_attention(45)
    assert_projection(manifest, head, ("lm_head",), ((128, 384),))
    for name in ("indexer.wq_b", "q_b_proj", "kv_b_proj", "o_proj"):
        assert_projection(manifest, draft[name], (prefix + "." + name,))
    assert_projection(manifest, draft["qkv_a_proj"], tuple(prefix + "." + name for name in
        ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj")))
