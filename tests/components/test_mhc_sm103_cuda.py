# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""mHC prefill on SM103 through FlashInfer's kernels against b12x's references."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

# GLM-5.3 and DeepSeek V4 both mix four 4096-wide streams; V4 normalizes with eps 1e-20.
MODELS = {
    "glm53": (4096, dict(rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-5)),
    "deepseek4": (4096, dict(rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-20)),
}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_composed_mhc_matches_reference(operation, model, norm_dtype):
    from b12x.preparation import PreparationSession
    from b12x.testing.mhc import post_reference, pre_reference

    from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
    from ignition.preparation.scratch import ScratchArena

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    torch.manual_seed(0)
    rows, (hidden, eps) = 1024, MODELS[model]
    weights = MHCWeights(torch.randn(24, 4 * hidden, device="cuda") * 0.02, torch.rand(3, device="cuda") * 0.5,
                         torch.randn(24, device="cuda") * 0.3,
                         (torch.rand(hidden, device="cuda") + 0.5).to(norm_dtype))
    residual = torch.randn(rows, 4, hidden, device="cuda").to(torch.bfloat16)
    inputs = (residual,)
    if operation == "post_pre":
        source = (torch.randn(rows, hidden, device="cuda") * 0.5).to(torch.bfloat16)
        post = torch.rand(rows, 4, device="cuda") * 2
        comb = torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1)
        inputs = (source, residual, post, comb)
    scratch = MHCScratch.allocate(rows, hidden, "cuda")
    arena = ScratchArena("cuda:0")
    with PreparationSession(device=torch.device("cuda"), autotune=True, compile_workers=0) as session:
        operator = PreparedMHC(inputs, weights, scratch, session=session, name=operation, workspace=arena, **eps)
        session.freeze()
    assert operator.composed is not None
    graph = torch.cuda.CUDAGraph()
    operator(*inputs)
    with torch.cuda.graph(graph):
        operator(*inputs)
    scratch.normalized.zero_()
    graph.replay()
    torch.cuda.synchronize()

    mixed = residual if operation == "pre" else post_reference(*inputs)
    collapsed, post_mix, comb_mix = pre_reference(mixed, weights.fn, weights.scale, weights.base,
        rms_eps=eps["rms_eps"], hc_eps=eps["hc_eps"], sinkhorn_iters=eps["sinkhorn_iters"])
    wide = collapsed.float()
    normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps["norm_eps"])
                  * weights.norm.float()).bfloat16()

    def relative(actual, expected):
        return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()

    assert relative(scratch.residual, mixed) < 1e-3
    # The projection reproduces FP32 weights to 16 bits (b12x uses TF32, 10).
    assert relative(scratch.post, post_mix) < 1e-3
    assert relative(scratch.comb, comb_mix) < 1e-3
    assert relative(scratch.normalized, normalized) < 5e-3


def _prepare_composed(operation, hidden, rows, norm_dtype, *, residual_scale=1.0):
    """Capture one composed mHC call; returns the operator, its scratch, inputs, weights and graph."""
    from b12x.preparation import PreparationSession

    from ignition.components import mhc_sm103
    from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
    from ignition.preparation.scratch import ScratchArena

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("composed mHC serves SM103")
    # The split-weight cache is keyed by the weight tensor's address, which a later test can reuse.
    mhc_sm103._SPLIT_WEIGHTS.clear()
    torch.manual_seed(0)
    eps = MODELS["glm53"][1]
    weights = MHCWeights(torch.randn(24, 4 * hidden, device="cuda") * 0.02, torch.rand(3, device="cuda") * 0.5,
                         torch.randn(24, device="cuda") * 0.3,
                         (torch.rand(hidden, device="cuda") + 0.5).to(norm_dtype))
    residual = (torch.randn(rows, 4, hidden, device="cuda") * residual_scale).to(torch.bfloat16)
    inputs = (residual,)
    if operation == "post_pre":
        source = (torch.randn(rows, hidden, device="cuda") * 0.5).to(torch.bfloat16)
        post = torch.rand(rows, 4, device="cuda") * 2
        comb = torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1)
        inputs = (source, residual, post, comb)
    scratch = MHCScratch.allocate(rows, hidden, "cuda")
    arena = ScratchArena("cuda:0")
    with PreparationSession(device=torch.device("cuda"), autotune=True, compile_workers=0) as session:
        operator = PreparedMHC(inputs, weights, scratch, session=session, name=operation, workspace=arena, **eps)
        session.freeze()
    assert operator.composed is not None
    graph = torch.cuda.CUDAGraph()
    operator(*inputs)
    with torch.cuda.graph(graph):
        operator(*inputs)
    return operator, scratch, inputs, weights, graph


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("rows", [4096, 1000])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_composed_mhc_replays_are_bit_exact(operation, rows, hidden, norm_dtype):
    """Replays of identical inputs agree bitwise when every output and projection scratch is poisoned first."""
    operator, scratch, inputs, _, graph = _prepare_composed(operation, hidden, rows, norm_dtype)
    composed = operator.composed
    outputs = (scratch.residual, scratch.post, scratch.comb, scratch.normalized)
    graph.replay()
    torch.cuda.synchronize()
    expected = [tensor.clone() for tensor in outputs]
    for tensor in expected:
        assert torch.isfinite(tensor.float()).all()
    for _ in range(10):
        for tensor in (*outputs, composed.dot, composed.squares, composed.collapsed):
            tensor.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("rows", [4096, 1000])
def test_composed_mhc_projection_matches_fp32_matmul(rows, hidden):
    """The BF16 high/low projection reproduces the FP32 product of the residual and the mixing weights."""
    operator, scratch, _, weights, graph = _prepare_composed("pre", hidden, rows, torch.bfloat16,
                                                              residual_scale=0.1)
    composed = operator.composed
    composed.dot.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = scratch.residual.view(rows, -1).float() @ weights.fn.T
    error = (composed.dot.sum(0) - expected).abs().max().item()
    assert error < 1e-4
