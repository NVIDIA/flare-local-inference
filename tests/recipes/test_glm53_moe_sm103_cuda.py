"""GLM 5.3 routed experts on SM103: clamped SwiGLU through FlashInfer's MoE kernels against checkpoint references."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

IDS = (0, 7, 150, 287)
LIMIT = 10.0


@pytest.fixture(scope="module")
def manifest():
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 routed experts require a GB300/B300")
    default = Path("/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD")
    return inspect_indexed_checkpoint(Path(os.environ.get("FLARE_GLM53_SNAPSHOT", default)))


def _run(experts, tokens, reference):
    """Route, run, capture and replay a small expert subset against a CPU reference."""
    from b12x.preparation import PreparationSession

    from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
    from ignition.preparation.scratch import ScratchArena

    count, hidden = experts.num_experts, experts.hidden_size
    source = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros((tokens, count), dtype=torch.bfloat16, device="cuda")
    route = RouterScratch.allocate(tokens, source.device, experts=count, top_k=2)
    output = torch.empty_like(source)
    workspace = ScratchArena("cuda:0")
    with torch.inference_mode(), PreparationSession(device=source.device, autotune=True,
                                                    compile_workers=0) as session:
        routing = PreparedRouter(logits, route, session=session, name=f"glm_sm103_route:{tokens}")
        prepared = PreparedRoutedExperts(source, experts, route.ids, route.weights, output,
                                         session=session, name=f"glm_sm103_experts:{tokens}", workspace=workspace)
        session.freeze()
        workspace.finalize()
        workspace.freeze()

        def forward():
            routing(logits)
            return prepared()

        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        generator = torch.Generator().manual_seed(4096 + tokens)
        cpu_source = (torch.randn((tokens, hidden), generator=generator) * 2).to(torch.bfloat16)
        source.copy_(cpu_source)
        logits.copy_(torch.randn((tokens, count), generator=generator).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ids, probabilities = route.ids.cpu(), route.weights.cpu()
        expected = torch.zeros_like(cpu_source, dtype=torch.float32)
        for expert in range(count):
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            if rows.numel():
                expected[rows] += reference(expert, cpu_source[rows]) * probabilities[rows, slots, None]
        difference = output.float().cpu() - expected.to(torch.bfloat16).float()
        relative = (difference.norm() / expected.norm().clamp_min(1e-12)).item()
        print(f"tokens {tokens}: relative L2 {relative:.2e}, max |error| {difference.abs().max().item():.2e}")
        assert relative < .008, relative


@pytest.mark.parametrize("tokens", [1, 6, 64])
@pytest.mark.parametrize("activation", ["a4", "a16"])
def test_clamped_swiglu_experts_match_checkpoint_reference(manifest, tokens, activation):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.glm53.weights import GlmWeightLoader
    from ignition.reference.qwen38_quant import (
        CheckpointTensors,
        decode_nvfp4,
        nvfp4_linear_reference,
        read_nvfp4_projection,
    )

    prefix = "model.language_model.layers.3.mlp.experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=tuple(f"{prefix}.{i}." for i in IDS), expert_activation=activation) as loader:
        experts = loader.load_experts(3, expert_ids=IDS)
    with CheckpointTensors(manifest, byte_budget=64 << 20) as reader:
        projections = [tuple(read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                             for name in ("gate_proj", "up_proj", "down_proj")) for expert in IDS]

    def a16(inputs, projection):
        weight = decode_nvfp4(projection.packed, projection.scales, torch.ones(()))
        return (inputs.to(torch.bfloat16).float() @ weight.T) * projection.weight_scale_2.reshape(())

    def a4(inputs, projection):
        if inputs.dtype == torch.bfloat16:
            from flashinfer.quantization import fp4_quantize

            # Match CUDA's approximate reciprocal at FP4 rounding boundaries;
            # CPU division can change codes for calibrated BF16 input values.
            packed, scales = fp4_quantize(inputs.cuda(), projection.input_scale.reciprocal().cuda(),
                is_sf_swizzled_layout=False, enable_pdl=False, nvfp4_4over6=None)
            rows, width = inputs.shape
            scales = scales.view(-1)[:rows * width // 16].reshape(rows, width // 16)
            values = decode_nvfp4(packed.cpu().view(torch.uint8), scales.cpu().view(torch.float8_e4m3fn),
                                  torch.ones(()))
            weight = decode_nvfp4(projection.packed, projection.scales, torch.ones(()))
            return (values @ weight.T) * (projection.weight_scale_2 / projection.input_scale.reciprocal())
        # FC2 quantizes the FP32 SwiGLU activation before its independent CPU GEMM.
        return nvfp4_linear_reference(inputs, projection, quant_scale_math="flashinfer")

    def reference(expert, values):
        gate, up, down = projections[expert]
        if activation == "a4":
            first = project = a4
        else:
            first = project = a16
        hidden = F.silu(first(values, gate).clamp(max=LIMIT)) * first(values, up).clamp(-LIMIT, LIMIT)
        if activation == "a16":
            hidden = hidden.to(torch.bfloat16)
        return project(hidden, down)

    _run(experts, tokens, reference)
