"""Token-major NVFP4 route quantization equals the per-route-row kernel it replaced, bit for bit."""

from __future__ import annotations

import os

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)


@triton.jit
def _quantize_four_route_rows(Source, Ids, Globals, Values, Scales, ROWS: tl.constexpr,
                              HIDDEN: tl.constexpr, TOP_K: tl.constexpr):
    """Reference: one program per four route rows and 1024 columns, with the arithmetic of the contract."""
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    columns = tl.program_id(1) * 512 + tl.arange(0, 512)
    valid = rows < ROWS
    expert = tl.load(Ids + rows, valid, 0)
    global_scale = tl.load(Globals + expert)
    reciprocal = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
        args=[global_scale], dtype=tl.float32, is_pure=True, pack=1)
    a = tl.load(Source + (rows // TOP_K)[:, None] * HIDDEN + columns[None, :] * 2,
        valid[:, None] & (columns[None, :] * 2 < HIDDEN), 0).to(tl.float32).reshape(4, 64, 8)
    b = tl.load(Source + (rows // TOP_K)[:, None] * HIDDEN + columns[None, :] * 2 + 1,
        valid[:, None] & (columns[None, :] * 2 < HIDDEN), 0).to(tl.float32).reshape(4, 64, 8)
    maximum = tl.max(tl.maximum(tl.abs(a), tl.abs(b)), 2)
    scale = tl.minimum(global_scale[:, None] * (maximum * (1.0 / 6.0)), 448.).to(tl.float8e4nv)
    effective = scale.to(tl.float32) * reciprocal[:, None]
    multiplier = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
        args=[effective], dtype=tl.float32, is_pure=True, pack=1)
    multiplier = tl.where(maximum == 0, 0., multiplier)
    a, b = (a * multiplier[:, :, None]).reshape(4, 512), (b * multiplier[:, :, None]).reshape(4, 512)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        constraints="=r,f,f", args=[a, b], dtype=tl.uint32, is_pure=True, pack=1)
    tl.store(Values + rows[:, None] * (HIDDEN // 2) + columns[None, :], packed.to(tl.uint8),
        valid[:, None] & (columns[None, :] * 2 < HIDDEN))
    groups = tl.program_id(1) * 64 + tl.arange(0, 64)
    tl.store(Scales + rows[:, None] * (HIDDEN // 16) + groups[None, :],
        scale.to(tl.uint8, bitcast=True), valid[:, None] & (groups[None, :] < HIDDEN // 16))


@pytest.fixture(scope="module", autouse=True)
def device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    if torch.cuda.get_device_capability(0)[0] < 10:
        pytest.skip("NVFP4 conversion requires a Blackwell device")


def _sources(tokens, hidden, generator):
    """Random rows plus a zero row, a single-block row, saturating, underflowing and BF16-extreme rows."""
    source = torch.randn(max(tokens, 6), hidden, generator=generator)
    source[1] = torch.where(torch.arange(hidden) // 16 == 3, source[1], torch.zeros(hidden))
    source[0] = 0
    source[2] *= 3e4
    source[3] *= 1e-4
    source[4] *= torch.finfo(torch.bfloat16).max / 8
    source[5] *= 2.0 ** -30
    source = source[:tokens]
    if tokens > 6:
        source[tokens - 1].zero_()
    return source.to(torch.bfloat16).cuda()


def _quantize_both(source, ids, globals, top_k, poison):
    from ignition.components.moe_sm103 import (
        _QUANTIZE_BLOCK,
        _QUANTIZE_WARPS,
        _quantize_routes,
        route_quantization_grid,
    )

    tokens, hidden = source.shape
    rows = tokens * top_k
    padded = (rows + 15) // 16 * 16
    outputs = []
    for token_major in (False, True):
        values = torch.full((padded, hidden // 2), poison, dtype=torch.uint8, device="cuda")
        scales = torch.full((padded, hidden // 16), poison, dtype=torch.uint8, device="cuda")
        if token_major:
            _quantize_routes[route_quantization_grid(tokens, hidden)](
                source, ids, globals, values, scales, rows, hidden, top_k, _QUANTIZE_BLOCK,
                num_warps=_QUANTIZE_WARPS)
        else:
            _quantize_four_route_rows[((rows + 3) // 4, (hidden + 1023) // 1024, 1)](
                source, ids, globals, values, scales, rows, hidden, top_k, num_warps=4)
        outputs.append((values, scales))
    return outputs


@pytest.mark.parametrize("tokens", [1, 7, 128, 255, 256, 4093, 4096])
@pytest.mark.parametrize("hidden,top_k,experts", [(4096, 8, 288), (2560, 2, 32), (4096, 2, 7)])
def test_token_major_quantization_equals_four_row_kernel(tokens, hidden, top_k, experts):
    generator = torch.Generator().manual_seed(tokens * 31 + hidden + top_k)
    source = _sources(tokens, hidden, generator)
    ids = torch.randint(experts, (tokens, top_k), generator=generator, dtype=torch.int32).cuda()
    # Spread the per-expert scales so block scales saturate for some experts and underflow for others.
    globals = (torch.rand(experts, generator=generator) * 99.7 + .3).cuda()
    (old_values, old_scales), (new_values, new_scales) = _quantize_both(source, ids, globals, top_k, 0xA5)
    torch.testing.assert_close(new_values, old_values, rtol=0, atol=0)
    torch.testing.assert_close(new_scales, old_scales, rtol=0, atol=0)
    rows = tokens * top_k
    # The padding rows FlashInfer reserves, and every row past the live routes, stay untouched.
    assert bool((new_values[rows:] == 0xA5).all()) and bool((new_scales[rows:] == 0xA5).all())
    assert bool((new_values[:rows] != 0xA5).any())


def test_saturating_and_underflowing_block_scales_are_exercised():
    """The oracle rows reach both ends of the E4M3 block-scale range and the all-zero block."""
    tokens, hidden, top_k = 8, 4096, 2
    generator = torch.Generator().manual_seed(11)
    source = _sources(tokens, hidden, generator)
    ids = torch.zeros((tokens, top_k), dtype=torch.int32, device="cuda")
    globals = torch.tensor([100.], device="cuda")
    (_, old_scales), (_, new_scales) = _quantize_both(source, ids, globals, top_k, 0)
    torch.testing.assert_close(new_scales, old_scales, rtol=0, atol=0)
    scales = new_scales[:tokens * top_k].view(torch.float8_e4m3fn).float()
    assert bool((scales == 448).any()) and bool((scales == 0).any())
    assert bool(((scales > 0) & (scales < 2.0 ** -6)).any())


def test_quantization_is_repeatable_over_stale_inputs():
    """Two launches over differently poisoned outputs give identical bytes."""
    tokens, hidden, top_k = 300, 4096, 8
    generator = torch.Generator().manual_seed(5)
    source = _sources(tokens, hidden, generator)
    ids = torch.randint(288, (tokens, top_k), generator=generator, dtype=torch.int32).cuda()
    globals = (torch.rand(288, generator=generator) * 3 + .3).cuda()
    first = _quantize_both(source, ids, globals, top_k, 0x00)[1]
    second = _quantize_both(source, ids, globals, top_k, 0xFF)[1]
    rows = tokens * top_k
    for a, b in zip(first, second):
        assert torch.equal(a[:rows], b[:rows])
