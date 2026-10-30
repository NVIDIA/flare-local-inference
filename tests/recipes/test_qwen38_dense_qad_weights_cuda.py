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


@pytest.mark.parametrize("rows,role", [(8, "gate_up"), (16, "gate_up"), (32, "gate_up"),
                                       (64, "gate_up"), (64, "down")])
@torch.no_grad()
def test_packed_dense_projection_preserves_checkpoint_arithmetic(rows, role):
    from ignition.components.linear import prepare_projection
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38_dense.weights import _pack_dense_w4a16
    from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_linear_reference

    snapshot = resolve("qwen3.8-27b-qad").snapshot(Path.home() / ".cache/huggingface/hub")
    manifest = inspect_indexed_checkpoint(snapshot)
    fields = ("gate", "up") if role == "gate_up" else ("down",)
    names = tuple(f"{TEXT_PREFIX}.layers.0.mlp.{name}_proj" for name in fields)
    with DirectWeightLoader(manifest, "cuda:0", prefixes=names) as loader:
        packed = _pack_dense_w4a16(loader.load_linear_group(names))
    with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
        references = tuple(Nvfp4Projection(reader.read(name + ".weight"), reader.read(name + ".weight_scale"),
                                           reader.read(name + ".weight_scale_2"), None) for name in names)
    source = torch.randn((rows, packed.in_features), device="cuda", dtype=torch.bfloat16) * .1
    output = torch.empty((rows, packed.out_features), device="cuda", dtype=torch.bfloat16)
    arena = ScratchArena(source.device)
    with RecordedPreparationSession(device=source.device, checkpoint="qad-packed-dense",
            workload_schema="calibrated-bf16-input", numerical_policy="a16", autotune_effort="off") as session:
        session.register_scratch(arena)
        operation = prepare_projection(source, packed, output, session=session, name=role, workspace=arena)
        session.tune(show_progress=False)
        assert operation.activation_mode == "a16"
        compiled = compile_forward(lambda: operation(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for gain in (.1, .3, 0.):
            source.normal_(0, gain)
            source[-1].zero_()
            expected = torch.cat([nvfp4_linear_reference(source.cpu(), ref).bfloat16() for ref in references], -1).cuda()
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[name] == after[name]
            assert torch.isfinite(output).all() and not output[-1].any()
            relative = (output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
            assert relative <= .005
        graph.reset()
