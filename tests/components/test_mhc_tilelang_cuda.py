"""Lagged mHC on SM120/121 matches the reference through the TileLang path."""

import os

import pytest
import torch
from b12x.testing.mhc import make_inputs, post_reference, pre_reference

from ignition.components import mhc_tilelang
from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")


@pytest.fixture(autouse=True)
def fp32_reference_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _options():
    return dict(rms_eps=1e-20, norm_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)


@pytest.mark.parametrize("operation", ("pre", "post_pre"))
@pytest.mark.parametrize("rows", (mhc_tilelang.MIN_ROWS, 7, 16, 33, 128, 1024))
@torch.no_grad()
def test_lagged_tilelang_mhc_matches_reference(operation, rows):
    device = torch.device("cuda", 0)
    if not mhc_tilelang.is_blackwell_consumer(device):
        pytest.skip("the TileLang mHC path serves SM120 and SM121")
    if mhc_tilelang.tiling(rows, device) is None:
        pytest.skip(f"{rows} rows keep b12x on this part")
    # The mixing operands come from the global RNG, and the normalized-output
    # budget is one BF16 ULP, so an unseeded draw can flip a rounding tie.
    torch.manual_seed(8110 + rows)
    hidden = 5120
    residual, branch, fn, scale, bias = make_inputs(
        tokens=rows, hidden_size=hidden, seed=8110 + rows, device=device)
    norm = torch.linspace(.5, 1.5, hidden, device=device).bfloat16()
    post = torch.rand((rows, 4), device=device) * 2
    comb = torch.softmax(torch.randn((rows, 4, 4), device=device), -1)
    inputs = (residual,) if operation == "pre" else (branch, residual, post, comb)
    incoming = torch.rand((rows, 4), device=device)
    predicted = torch.empty_like(incoming)
    scratch = MHCScratch.allocate(rows, hidden, device)
    options = _options()
    with RecordedPreparationSession(device=device, checkpoint="synthetic-mhc",
            workload_schema="mhc.tilelang.v1", numerical_policy="fp32-weights") as session:
        prepared = PreparedMHC(inputs, MHCWeights(fn, scale, bias, norm), scratch,
            session=session, name="tilelang", workspace=None,
            pre_mix=incoming, pre_out=predicted, **options)
        assert prepared.tilelang is not None
        assert prepared.tilelang.config.backend == "tilelang"
        session.tune()
        compiled = compile_forward(lambda: prepared(*inputs))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for shift in (0, 3):
                incoming.copy_(torch.eye(4, device=device)[(torch.arange(rows, device=device) + shift) % 4])
                predicted.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                mixed = residual if operation == "pre" else post_reference(*inputs)
                _, expected_post, expected_comb = pre_reference(
                    mixed, fn, scale, bias, rms_eps=options["rms_eps"],
                    hc_eps=options["hc_eps"], sinkhorn_iters=options["sinkhorn_iters"],
                    y_dtype=torch.float32)
                if operation == "pre":
                    # The pre operation carries the residual through unchanged.
                    torch.testing.assert_close(scratch.residual, residual, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(scratch.residual, mixed, rtol=0, atol=0)
                flat = mixed.flatten(1).float()
                projected = torch.nn.functional.linear(flat, fn) * torch.rsqrt(
                    flat.square().mean(-1, keepdim=True) + options["rms_eps"])
                expected_pre = torch.sigmoid(projected[:, :4] * scale[0] + bias[:4]) + options["hc_eps"]
                # The collapse consumes the lagged coefficients, not the pre the
                # projection just produced, and rounds the mixed state to BF16 first.
                y = (mixed.bfloat16().float() * incoming[:, :, None]).sum(1).bfloat16().float()
                expected_norm = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + options["norm_eps"])
                                 * norm.float()).bfloat16()
                torch.testing.assert_close(predicted, expected_pre, rtol=2e-5, atol=4e-5)
                torch.testing.assert_close(scratch.post, expected_post, rtol=2e-5, atol=4e-5)
                torch.testing.assert_close(scratch.comb, expected_comb, rtol=2e-5, atol=4e-5)
                # The RMS denominator is reduced in a different order than torch's
                # sum, which can flip a BF16 rounding tie; allow exactly one ULP.
                step = (torch.nextafter(expected_norm, torch.full_like(expected_norm, float("inf")))
                        - expected_norm).float()
                off = (scratch.normalized.float() - expected_norm.float()).abs()
                budget = step + 2e-5 * expected_norm.float().abs()
                assert torch.all(off <= budget), (off / step).max().item()
        finally:
            graph.reset()


@torch.no_grad()
def test_lagged_mhc_with_fp32_norm_weight_keeps_b12x():
    """An FP32 norm weight must not be narrowed to reach the TileLang pair."""
    device = torch.device("cuda", 0)
    if not mhc_tilelang.is_blackwell_consumer(device):
        pytest.skip("the TileLang mHC path serves SM120 and SM121")
    rows, hidden = 8, 5120
    residual, branch, fn, scale, bias = make_inputs(
        tokens=rows, hidden_size=hidden, seed=8111 + rows, device=device)
    norm = torch.linspace(.5, 1.5, hidden, device=device)
    post = torch.rand((rows, 4), device=device) * 2
    comb = torch.softmax(torch.randn((rows, 4, 4), device=device), -1)
    scratch = MHCScratch.allocate(rows, hidden, device)
    incoming = torch.rand((rows, 4), device=device)
    predicted = torch.empty_like(incoming)
    with RecordedPreparationSession(device=device, checkpoint="synthetic-mhc",
            workload_schema="mhc.tilelang.v1", numerical_policy="fp32-weights") as session:
        prepared = PreparedMHC((branch, residual, post, comb), MHCWeights(fn, scale, bias, norm),
            scratch, session=session, name="fp32-norm", workspace=None,
            pre_mix=incoming, pre_out=predicted, **_options())
        assert prepared.tilelang is None
        assert prepared.plan is not None
        with pytest.raises(ValueError, match="BF16 norm weight"):
            mhc_tilelang.PreparedTileLangMHC(
                (branch, residual, post, comb), MHCWeights(fn, scale, bias, norm), scratch,
                session=session, name="fp32-norm-direct", workspace=None,
                pre_mix=incoming, pre_out=predicted, **_options())


def test_tiling_selects_the_measured_profile():
    """The pair's useful hidden block follows the part's SM count, then rows."""
    device = torch.device("cuda", 0)
    if not mhc_tilelang.is_blackwell_consumer(device):
        pytest.skip("the TileLang mHC path serves SM120 and SM121")
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    assert mhc_tilelang.tiling(mhc_tilelang.MIN_ROWS - 1, device) is None
    if multiprocessors >= 128:
        # A wide part is under-filled at 160 CTAs, so coarser blocks only lose.
        assert mhc_tilelang.tiling(mhc_tilelang.MIN_ROWS, device) == mhc_tilelang.BLOCK_H
        assert mhc_tilelang.tiling(256, device) == mhc_tilelang.BLOCK_H
        assert mhc_tilelang.tiling(512, device) is None
        assert mhc_tilelang.tiling(8192, device) is None
    elif multiprocessors <= 64:
        # A narrow part saturates at 80 CTAs, and past 1024 rows the halved
        # split count pays; the middle band stays on b12x.
        assert mhc_tilelang.tiling(64, device) == mhc_tilelang.BLOCK_H
        assert mhc_tilelang.tiling(128, device) is None
        assert mhc_tilelang.tiling(512, device) is None
        assert mhc_tilelang.tiling(1024, device) == mhc_tilelang.PREFILL_BLOCK_H
        assert mhc_tilelang.tiling(8192, device) == mhc_tilelang.PREFILL_BLOCK_H


@torch.no_grad()
def test_lagged_mhc_in_the_middle_band_keeps_b12x():
    """Rows where neither tiling beats b12x keep its fused launches."""
    device = torch.device("cuda", 0)
    if not mhc_tilelang.is_blackwell_consumer(device):
        pytest.skip("the TileLang mHC path serves SM120 and SM121")
    rows, hidden = 512, 5120
    residual, branch, fn, scale, bias = make_inputs(
        tokens=rows, hidden_size=hidden, seed=8112 + rows, device=device)
    norm = torch.linspace(.5, 1.5, hidden, device=device).bfloat16()
    post = torch.rand((rows, 4), device=device) * 2
    comb = torch.softmax(torch.randn((rows, 4, 4), device=device), -1)
    scratch = MHCScratch.allocate(rows, hidden, device)
    incoming = torch.rand((rows, 4), device=device)
    predicted = torch.empty_like(incoming)
    with RecordedPreparationSession(device=device, checkpoint="synthetic-mhc",
            workload_schema="mhc.tilelang.v1", numerical_policy="fp32-weights") as session:
        prepared = PreparedMHC((branch, residual, post, comb), MHCWeights(fn, scale, bias, norm),
            scratch, session=session, name="middle-band", workspace=None,
            pre_mix=incoming, pre_out=predicted, **_options())
        assert prepared.tilelang is None
        assert prepared.plan is not None


@torch.no_grad()
def test_prepared_shapes_share_one_packed_projection():
    """Every shape using one weight set shares its packed planes."""
    device = torch.device("cuda", 0)
    if not mhc_tilelang.is_blackwell_consumer(device):
        pytest.skip("the TileLang mHC path serves SM120 and SM121")
    hidden, rows = 5120, 8
    residual, branch, fn, scale, bias = make_inputs(
        tokens=rows, hidden_size=hidden, seed=8113 + rows, device=device)
    norm = torch.linspace(.5, 1.5, hidden, device=device).bfloat16()
    post = torch.rand((rows, 4), device=device) * 2
    comb = torch.softmax(torch.randn((rows, 4, 4), device=device), -1)
    weights = MHCWeights(fn, scale, bias, norm)
    incoming = torch.rand((rows, 4), device=device)
    prepared = []
    with RecordedPreparationSession(device=device, checkpoint="synthetic-mhc",
            workload_schema="mhc.tilelang.v1", numerical_policy="fp32-weights") as session:
        for name in ("share-a", "share-b"):
            scratch = MHCScratch.allocate(rows, hidden, device)
            predicted = torch.empty_like(incoming)
            prepared.append(PreparedMHC((branch, residual, post, comb), weights, scratch,
                session=session, name=name, workspace=None,
                pre_mix=incoming, pre_out=predicted, **_options()))
        assert all(item.tilelang is not None for item in prepared)
        assert prepared[0].tilelang.high is prepared[1].tilelang.high
        assert prepared[0].tilelang.low is prepared[1].tilelang.low
        assert prepared[0].tilelang.config.block_h == mhc_tilelang.BLOCK_H


@torch.no_grad()
def test_tilelang_mhc_projection_split_preserves_weights():
    device = torch.device("cuda", 0)
    if not mhc_tilelang.is_blackwell_consumer(device):
        pytest.skip("the TileLang mHC path serves SM120 and SM121")
    generator = torch.Generator(device="cpu").manual_seed(4107)
    fn = (torch.randn((24, 4 * 5120), generator=generator) / 64).to(device=device)
    high, low = mhc_tilelang.pack_projection(fn, 5120)
    splits, k, padded = high.shape
    assert padded == mhc_tilelang.MIXES_PAD and k == 4 * mhc_tilelang.BLOCK_H
    reconstructed = torch.zeros_like(fn)
    block = mhc_tilelang.BLOCK_H
    for split in range(splits):
        for stream in range(4):
            window = slice(stream * block, (stream + 1) * block)
            offset = stream * 5120 + split * block
            reconstructed[:, offset:offset + block] = (
                high[split, window, :24].float()
                + low[split, window, :24].float() * mhc_tilelang.LOW_SCALE).T
    relative = (reconstructed - fn).norm() / fn.norm()
    assert relative < 2e-7, float(relative)
