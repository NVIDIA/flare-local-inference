# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint routed experts on SM103 through FlashInfer's native MoE kernels."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from b12x.preparation import PreparationSession

from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.moe_sm103 import CuteW4A16Experts, TrtllmNvfp4Experts
from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.mtp_weights import MTPWeightLoader
from ignition.recipes.qwen38.weights import QwenWeightLoader, allocate_expert_scale_scratch
from ignition.reference.qwen38_quant import (
    CheckpointTensors,
    decode_nvfp4,
    nvfp4_linear_reference,
    read_nvfp4_projection,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

IDS = (0, 7, 300, 511)


@pytest.fixture(scope="module")
def manifest():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 routed experts require a GB300/B300")
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    return inspect_qwen38_checkpoint(Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default)))


def _metrics(actual, expected):
    difference = actual.float().cpu() - expected.float().cpu()
    return {
        "max_absolute_error": difference.abs().max().item(),
        "relative_l2_error": (difference.norm() / expected.float().cpu().norm().clamp_min(1e-12)).item(),
    }


def _w4a16_reference(values, gate, up, down):
    def project(inputs, projection):
        weight = decode_nvfp4(projection.packed, projection.scales, torch.ones(()))
        return (inputs.to(torch.bfloat16).float() @ weight.T) * projection.weight_scale_2.reshape(())

    intermediate = (F.silu(project(values, gate)) * project(values, up)).to(torch.bfloat16)
    return project(intermediate, down)


def _run_component(experts, tokens, reference, *, compiled, source_scale=1.0):
    """Route, run, capture, and replay a small expert subset against a CPU reference."""
    count = experts.num_experts
    source = torch.zeros((tokens, experts.hidden_size), dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros((tokens, count), dtype=torch.bfloat16, device="cuda")
    route = RouterScratch.allocate(tokens, source.device, experts=count, top_k=2)
    output = torch.empty_like(source)
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        routing = PreparedRouter(logits, route, session=session, name=f"sm103_route:{tokens}")
        prepared = PreparedRoutedExperts(source, experts, route.ids, route.weights, output,
                                         session=session, name=f"sm103_experts:{tokens}", workspace=workspace)
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        assert prepared.plan.selection.source in ("tuned", "cached", "fixed")

        def forward():
            routing(logits)
            return prepared()

        run = compile_forward(forward) if compiled else forward
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        generator = torch.Generator().manual_seed(1031 + tokens)
        for iteration in range(2):
            cpu_source = (torch.randn((tokens, experts.hidden_size), generator=generator) * source_scale).to(torch.bfloat16)
            source.copy_(cpu_source)
            logits.copy_(torch.randn((tokens, count), generator=generator).to(torch.bfloat16))
            output.fill_(float("nan"))
            workspace.tensor.fill_(0x7F)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            ids, probabilities = route.ids.cpu(), route.weights.cpu()
            expected = torch.zeros_like(cpu_source, dtype=torch.float32)
            for expert in range(count):
                rows, slots = (ids == expert).nonzero(as_tuple=True)
                if rows.numel():
                    expected[rows] += reference(expert, cpu_source[rows]) * probabilities[rows, slots, None]
            metrics = _metrics(output, expected.to(torch.bfloat16))
            assert metrics["relative_l2_error"] < .008, metrics
            print(json.dumps({"component": type(experts).__name__, "tokens": tokens, "compiled": compiled,
                              "iteration": iteration, "implementation": prepared.binding.implementation,
                              **metrics}))


@pytest.mark.parametrize("tokens", [1, 4, 24, 512])
@pytest.mark.parametrize("compiled", [False, True])
def test_csf_nvfp4_experts_use_trtllm_gen(manifest, tokens, compiled):
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts"
    scratch = allocate_expert_scale_scratch("cuda:0", len(IDS))
    with QwenWeightLoader(manifest, "cuda:0", prefixes=tuple(f"{prefix}.{i}." for i in IDS)) as loader:
        experts = loader.load_experts(0, expert_ids=IDS, scale_scratch=scratch)
    assert isinstance(experts, TrtllmNvfp4Experts)
    with CheckpointTensors(manifest, byte_budget=64 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                             for name in ("gate_proj", "up_proj", "down_proj")) for expert in IDS]

    def reference(expert, values):
        # FlashInfer quantizes FC1 inputs, and trtllm-gen's FC1 epilogue quantizes the
        # FP32 SwiGLU output for FC2 without an intermediate BF16 rounding.
        gate, up, down = projections[expert]
        project = lambda inputs, projection: nvfp4_linear_reference(
            inputs, projection, quant_scale_math="flashinfer")
        return project(F.silu(project(values, gate)) * project(values, up), down)

    _run_component(experts, tokens, reference, compiled=compiled)


@pytest.mark.parametrize("tokens", [1, 24, 512])
def test_mtp_w4a16_experts_use_cute_dsl(manifest, tokens):
    prefix = "mtp.layers.0.mlp.experts"
    with MTPWeightLoader(manifest, "cuda:0", prefixes=tuple(f"{prefix}.{i}." for i in IDS)) as loader:
        experts = loader.load_experts(expert_ids=IDS)
    assert isinstance(experts, CuteW4A16Experts)
    with CheckpointTensors(manifest, byte_budget=64 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                             for name in ("gate_proj", "up_proj", "down_proj")) for expert in IDS]

    _run_component(experts, tokens, lambda expert, values: _w4a16_reference(values, *projections[expert]),
                   compiled=False)


@pytest.mark.parametrize("tokens", [1, 6, 24, 512])
def test_trtllm_reduction_into_shared_output_matches_finalize(manifest, tokens):
    """Staged FC2 rows reduced into the shared output equal trtllm's finalize plus a BF16 add."""
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts"
    with QwenWeightLoader(manifest, "cuda:0", prefixes=tuple(f"{prefix}.{i}." for i in IDS)) as loader:
        experts = loader.load_experts(0, expert_ids=IDS)
    _assert_shared_reduction(experts, tokens)


def _assert_shared_reduction(experts, tokens):
    count = experts.num_experts
    hidden = experts.hidden_size
    source = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros((tokens, count), dtype=torch.bfloat16, device="cuda")
    route = RouterScratch.allocate(tokens, source.device, experts=count, top_k=2)
    routed, shared, fused = (torch.empty_like(source) for _ in range(3))
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        routing = PreparedRouter(logits, route, session=session, name=f"sm103_reduce_route:{tokens}")
        prepared = PreparedRoutedExperts(source, experts, route.ids, route.weights, routed,
                                         session=session, name=f"sm103_reduce:{tokens}", workspace=workspace)
        session.freeze()
        workspace.finalize()
        workspace.freeze()
        assert prepared.reduces_into_shared

        def forward():
            routing(logits)
            prepared.stage_rows()
            return prepared.reduce(shared, fused)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        generator = torch.Generator().manual_seed(2207 + tokens)
        for iteration in range(2):
            source.copy_(torch.randn((tokens, hidden), generator=generator).to(torch.bfloat16))
            logits.copy_(torch.randn((tokens, count), generator=generator).to(torch.bfloat16))
            shared.copy_(torch.randn((tokens, hidden), generator=generator).to(torch.bfloat16))
            fused.fill_(float("nan"))
            graph.replay()
            routing(logits)
            prepared()
            torch.cuda.synchronize()
            assert torch.equal(fused, torch.add(routed, shared))


@pytest.fixture(scope="module")
def glm_manifest():
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("requires SM103")
    snapshot = os.environ.get("FLARE_GLM53_SNAPSHOT")
    if snapshot is None:
        pytest.skip("requires FLARE_GLM53_SNAPSHOT")
    return inspect_indexed_checkpoint(Path(snapshot))


@pytest.mark.parametrize("tokens", [1, 24])
def test_glm_experts_preserve_clamped_swiglu(glm_manifest, tokens):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.glm53.weights import GlmWeightLoader

    ids = (0, 7, 100, 287)
    prefix = "model.language_model.layers.3.mlp.experts"
    with GlmWeightLoader(glm_manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=tuple(f"{prefix}.{i}." for i in ids), expert_activation="a16") as loader:
        experts = loader.load_experts(3, expert_ids=ids)
    assert experts.swiglu_limit == 10.0
    with CheckpointTensors(glm_manifest, byte_budget=128 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                             for name in ("gate_proj", "up_proj", "down_proj")) for expert in ids]
    clamped = []

    def reference(expert, values):
        def project(inputs, projection):
            weight = decode_nvfp4(projection.packed, projection.scales, torch.ones(()))
            return (inputs.to(torch.bfloat16).float() @ weight.T) * projection.weight_scale_2.reshape(())
        gate, up, down = projections[expert]
        gate, up = project(values, gate), project(values, up)
        clamped.append(bool((gate > 10).any() or (up.abs() > 10).any()))
        intermediate = F.silu(gate.clamp(max=10)) * up.clamp(min=-10, max=10)
        return project(intermediate.bfloat16(), down)

    _run_component(experts, tokens, reference, compiled=True, source_scale=32.0)
    assert any(clamped), "reference inputs must exercise the checkpoint's clamp"


@pytest.mark.parametrize("tokens", [1, 24, 128, 512, 4096])
def test_glm_a4_experts_preserve_distinct_input_scales(glm_manifest, tokens):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.glm53.weights import GlmWeightLoader

    prefix = "model.language_model.layers.3.mlp.experts"
    ids = (0, 7, 100, 287)
    with GlmWeightLoader(glm_manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=tuple(f"{prefix}.{i}." for i in ids), expert_activation="a4") as loader:
        experts = loader.load_experts(3, expert_ids=ids)
    assert experts.input_global_scale.numel() == len(ids)
    assert experts.input_global_scale.unique().numel() > 1
    with CheckpointTensors(glm_manifest, byte_budget=128 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                             for name in ("gate_proj", "up_proj", "down_proj")) for expert in ids]

    def reference(expert, values):
        gate, up, down = projections[expert]
        project = lambda inputs, projection: nvfp4_linear_reference(
            inputs, projection, quant_scale_math="flashinfer")
        intermediate = F.silu(project(values, gate).clamp(max=10)) * project(values, up).clamp(min=-10, max=10)
        return project(intermediate, down)

    _run_component(experts, tokens, reference, compiled=True, source_scale=32.)


@pytest.mark.parametrize("tokens", [1, 512, 4093, 4096])
def test_glm_a4_shared_reduction_preserves_distinct_input_scales(glm_manifest, tokens):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.glm53.weights import GlmWeightLoader

    prefix = "model.language_model.layers.3.mlp.experts"
    ids = (0, 7, 100, 287)
    with GlmWeightLoader(glm_manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=tuple(f"{prefix}.{i}." for i in ids), expert_activation="a4") as loader:
        experts = loader.load_experts(3, expert_ids=ids)
    _assert_shared_reduction(experts, tokens)


@pytest.mark.parametrize("hidden", [2560, 4096])
@pytest.mark.parametrize("tokens,top_k,count", [(17, 2, 4), (301, 8, 288)])
def test_routed_nvfp4_quantization_matches_flashinfer(hidden, tokens, top_k, count):
    from ignition.components.moe_sm103 import (
        _QUANTIZE_BLOCK,
        _fp4_quantizer,
        _quantize_routes,
        route_quantization_grid,
    )

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("requires SM103")
    torch.manual_seed(1034)
    source = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    source[0].zero_()
    ids = torch.randint(count, (tokens, top_k), device="cuda", dtype=torch.int32)
    globals = (torch.tensor([.3, 1., 25., 100.], device="cuda") if count == 4
               else torch.rand(count, device="cuda") * 99.7 + .3)
    values = torch.empty(tokens * top_k, hidden // 2, device="cuda", dtype=torch.uint8)
    scales = torch.empty(tokens * top_k, hidden // 16, device="cuda", dtype=torch.uint8)
    _quantize_routes[route_quantization_grid(tokens, hidden)](
        source, ids, globals, values, scales, tokens * top_k, hidden, top_k, _QUANTIZE_BLOCK, num_warps=4)
    reference_values = torch.empty(1, hidden // 2, device="cuda", dtype=torch.uint8)
    reference_scales = torch.empty(16 * hidden // 16, device="cuda", dtype=torch.uint8)
    quantize = _fp4_quantizer().fp4_quantize
    for row, expert in enumerate(ids.flatten().tolist()):
        quantize(source[row // top_k:row // top_k + 1], globals[expert:expert + 1],
            reference_values, reference_scales, 16, False, False, False, False, True, 0)
        torch.testing.assert_close(values[row], reference_values[0], rtol=0, atol=0)
        torch.testing.assert_close(scales[row], reference_scales[:hidden // 16], rtol=0, atol=0)
