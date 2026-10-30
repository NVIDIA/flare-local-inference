"""Stored MXFP8 projection grouping for the dense QAD checkpoint."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes import resolve
from ignition.recipes.qwen38_dense.config import TEXT_PREFIX
from ignition.reference.qwen38_quant import CheckpointTensors

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("names", [("in_proj_qkv", "in_proj_z"), ("in_proj_b", "in_proj_a")])
def test_qad_packed_groups_preserve_serialized_rows(names):
    snapshot = resolve("qwen3.8-27b-qad").snapshot(Path.home() / ".cache/huggingface/hub")
    manifest = inspect_indexed_checkpoint(snapshot)
    prefixes = tuple(f"{TEXT_PREFIX}.layers.0.linear_attn.{name}" for name in names)
    with DirectWeightLoader(manifest, "cuda:0", prefixes=prefixes) as loader:
        packed = loader.load_linear_group(prefixes)
    values = packed.weight.values.view(torch.uint8).cpu().reshape(packed.out_features, packed.padded_in_features)
    scales = packed.weight.scale_rows.view(torch.uint8).cpu().reshape(packed.out_features, -1)
    offset = 0
    with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
        for prefix in prefixes:
            original = reader.read(prefix + ".weight").view(torch.uint8)
            original_scales = reader.read(prefix + ".weight_scale").view(torch.uint8)
            rows, columns = original.shape
            assert torch.equal(values[offset:offset + rows, :columns], original)
            assert torch.equal(scales[offset:offset + rows, :columns // 32], original_scales[:rows, :columns // 32])
            offset += rows
    assert offset == packed.out_features
