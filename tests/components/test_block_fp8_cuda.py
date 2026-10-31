# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block-FP8 preparation, compilation and replay preserve arbitrary FP32 scales."""

import contextlib
import gc
import os
import weakref

import pytest
import torch

from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight, load_quantizer, quantize_token_groups
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


@pytest.mark.parametrize("rows", [1, 2, 4, 17, 31, 32, 33, 64, 65, 4096])
def test_block_fp8_compiled_replay(rows, monkeypatch):
    from flashinfer.quantization import per_token_group_quant_8bit

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(491)
    values = torch.randn((256, 512), device=device, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.tensor([[.1, .2, .7, .3], [.4, .8, .3, .6]], device=device)
    source = torch.randn((rows, 512), device=device, dtype=torch.bfloat16, generator=generator)
    output = torch.empty((rows, 256), device=device, dtype=torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint="block-fp8-projection",
            workload_schema="block-fp8-projection", numerical_policy="default") as session:
        projection = BlockFP8Projection(source, BlockFP8Weight(values, scales), session=session, name="projection")
        tuning_inputs = []

        def track_inputs(trial, *args, **kwargs):
            result = quantize_token_groups(trial, *args, **kwargs)
            tuning_inputs.extend(weakref.ref(value) for value in (trial, *result))
            return result

        with monkeypatch.context() as patch:
            patch.setattr("ignition.components.block_fp8.quantize_token_groups", track_inputs)
            session.tune()
        gc.collect()
        assert tuning_inputs and all(value() is None for value in tuning_inputs)

        def forward():
            output.copy_(projection(source))

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for multiplier in (1., .2, 2.):
            source.mul_(multiplier)
            quantized, activation_scale = per_token_group_quant_8bit(source, 128)
            expected = ((quantized.float() * activation_scale.repeat_interleave(128, 1))
                        @ (values.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)).T)
            before = torch.cuda.memory_stats(device)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats(device)
            torch.testing.assert_close(output.float(), expected, rtol=.02, atol=.02)
            assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
        graph.reset()


def quantize_loaded(source):
    """Quantize after loading the compiled program, as preparation does before kernels freeze."""
    load_quantizer(source.shape[1], source.device.index or 0)
    return quantize_token_groups(source)


def activation_cases(rows, k, device):
    """Activation matrices covering ordinary, zero, outlier, extreme and subnormal groups."""
    generator = torch.Generator(device=device).manual_seed(rows * 100003 + k)
    base = torch.randn((rows, k), device=device, generator=generator)
    magnitudes = torch.exp2(torch.randint(-20, 20, (rows, k // 128, 1), device=device,
                                          generator=generator).float())
    mixed = (base.view(rows, k // 128, 128) * magnitudes).view(rows, k)
    outlier = base.clone()
    outlier[:, ::257] *= 4096.
    zeros = base.clone()
    zeros[::2] = 0
    zeros[1::4, : min(k, 128)] = 0
    extreme = base.clone()
    extreme[:, 5::131] = torch.finfo(torch.bfloat16).max
    extreme[:, 7::131] = -torch.finfo(torch.bfloat16).max
    tiny = base * 1e-39
    ramp = torch.linspace(-448, 448, k, device=device).expand(rows, k).contiguous()
    return {name: value.to(torch.bfloat16).contiguous() for name, value in (
        ("random", base), ("mixed_magnitudes", mixed), ("outliers", outlier), ("zero_groups", zeros),
        ("bf16_extremes", extreme), ("subnormals", tiny), ("ramp", ramp), ("scaled_up", base * 300.))}


# 7168 is DeepSeek V4's hidden width (a partial last 16-group program); 5120 is DeepSeek V4.1's;
# 6144 and 2048 are MiMo V2 Pro's hidden width and a TP8 shard of its 16384-wide attention output.
@pytest.mark.parametrize("k", [128, 512, 1536, 2048, 4096, 5120, 6144, 7168, 16384])
@pytest.mark.parametrize("rows", [1, 7, 32, 33, 256, 4096])
def test_token_group_quantizer_matches_flashinfer_bitwise(rows, k):
    from flashinfer.quantization import per_token_group_quant_8bit

    device = torch.device("cuda", 0)
    for name, source in activation_cases(rows, k, device).items():
        expected_values, expected_scales = per_token_group_quant_8bit(source, 128)
        values, scales = quantize_loaded(source)
        assert values.shape == expected_values.shape and values.dtype == torch.float8_e4m3fn, name
        assert scales.shape == expected_scales.shape and scales.dtype == torch.float32, name
        assert scales.is_contiguous() and values.is_contiguous(), name
        assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8)), (name, rows, k)
        assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32)), (name, rows, k)


def test_token_group_quantizer_rows_are_independent_of_capacity():
    """A row quantizes identically in a small and a large capacity, whatever stale rows surround it."""
    device = torch.device("cuda", 0)
    k = 2048
    full = activation_cases(4096, k, device)["mixed_magnitudes"]
    values, scales = quantize_loaded(full)
    for rows in (1, 8, 128, 4096):
        part_values, part_scales = quantize_loaded(full[:rows].contiguous())
        assert torch.equal(part_values.view(torch.uint8), values[:rows].view(torch.uint8))
        assert torch.equal(part_scales.view(torch.int32), scales[:rows].view(torch.int32))


@pytest.mark.parametrize("rows", [1, 8, 128, 4096])
def test_token_group_quantizer_graph_replay_is_stable(rows):
    from flashinfer.quantization import per_token_group_quant_8bit

    device = torch.device("cuda", 0)
    k = 1536
    source = activation_cases(rows, k, device)["random"]
    quantize_loaded(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, scales = quantize_loaded(source)
    for _ in range(2):
        values.view(torch.uint8).fill_(0x55)
        scales.fill_(float("nan"))
        before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
        expected_values, expected_scales = per_token_group_quant_8bit(source, 128)
        assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
        assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))
    graph.reset()


def test_token_group_quantizer_rejects_unsupported_inputs():
    device = torch.device("cuda", 0)
    with pytest.raises(ValueError):
        quantize_token_groups(torch.zeros((4, 128), dtype=torch.float16, device=device))
    with pytest.raises(ValueError):
        quantize_token_groups(torch.zeros((4, 192), dtype=torch.bfloat16, device=device))
    with pytest.raises(ValueError):
        quantize_token_groups(torch.zeros((4, 256), dtype=torch.bfloat16, device=device)[:, :128])
    misaligned = torch.zeros(4 * 256 + 8, dtype=torch.bfloat16, device=device)[1:1 + 4 * 256].view(4, 256)
    with pytest.raises(ValueError, match="16-byte"):
        quantize_token_groups(misaligned)


def test_unloaded_quantizer_fails_instead_of_compiling():
    device = torch.device("cuda", 0)
    with pytest.raises(RuntimeError, match="not loaded during preparation"):
        quantize_token_groups(torch.zeros((4, 384), dtype=torch.bfloat16, device=device))


@pytest.fixture
def sm103():
    """Only the SM103 GEMM writes a caller-provided output."""
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("in-place block-FP8 output requires SM103")


@contextlib.contextmanager
def prepared_projection(rows, k, n, output_factory, seed=7):
    """A tuned projection writing into ``output_factory(source)``, with its source and session.

    The caller freezes the session once any graph is captured."""
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(seed)
    values = torch.randn((n, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand((n // 128, k // 128), device=device, generator=generator) + .1
    source = torch.randn((rows, k), device=device, dtype=torch.bfloat16, generator=generator)
    output = output_factory(source)
    with RecordedPreparationSession(device=device, checkpoint="block-fp8-projection",
            workload_schema="block-fp8-projection", numerical_policy="default") as session:
        projection = BlockFP8Projection(source, BlockFP8Weight(values, scales), output,
                                        session=session, name="projection")
        session.tune()
        # Resolve both entry points before kernel resolution freezes.
        projection(source)
        copy_path(projection, source)
        yield projection, source, output, session


def copy_path(projection, source):
    from ignition.components.block_fp8 import _project

    return _project(source, projection.weight.values, projection.weight.scales, projection.plan.handle)


def poison(output):
    output.view(torch.uint8).fill_(0xFF) if output.is_contiguous() else output.fill_(float("nan"))


SHAPES = [(512, 256), (2048, 4096), (4096, 2048), (1536, 4096), (4096, 4096)]


@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("rows", [1, 8, 32, 33, 128, 256, 4096])
def test_in_place_output_matches_result_tensor_bitwise(rows, k, n, sm103):
    device = torch.device("cuda", 0)
    with prepared_projection(rows, k, n, lambda source: torch.empty((rows, n), dtype=torch.bfloat16,
                                                                     device=device)) as (projection, source, output, session):
        session.freeze()
        assert projection.in_place
        expected = copy_path(projection, source)
        for _ in range(2):
            poison(output)
            assert projection(source) is output
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 33, 256])
def test_strided_output_keeps_values_and_leaves_neighbours_untouched(rows, sm103):
    """Column slices of a wider buffer keep the other columns intact, written in place only when contiguous."""
    device = torch.device("cuda", 0)
    k, n = 2048, 4096
    wide = torch.full((rows, n + 256), 3., dtype=torch.bfloat16, device=device)
    with prepared_projection(rows, k, n, lambda source: wide[:, 128:128 + n]) as (projection, source, output, session):
        session.freeze()
        assert projection.in_place == (rows == 1)  # a single row of a slice is contiguous
        expected = copy_path(projection, source)
        poison(output)
        projection(source)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert bool((wide[:, :128] == 3.).all() and (wide[:, 128 + n:] == 3.).all())


def test_in_place_output_may_alias_the_source(sm103):
    """The source is fully quantized before the GEMM writes, so a same-shape output may reuse its storage."""
    device = torch.device("cuda", 0)
    rows, k = 256, 2048
    with prepared_projection(rows, k, k, lambda source: source) as (projection, source, output, session):
        session.freeze()
        assert projection.in_place
        expected = copy_path(projection, source).clone()
        projection(source)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert device.type == "cuda"


@pytest.mark.parametrize("rows", [1, 8, 128, 4096])
def test_in_place_capture_is_allocation_stable_and_replays_identically(rows, sm103):
    device = torch.device("cuda", 0)
    k, n = 2048, 4096
    with prepared_projection(rows, k, n, lambda source: torch.empty((rows, n), dtype=torch.bfloat16,
                                                                     device=device)) as (projection, source, output, session):
        expected = copy_path(projection, source).clone()
        projection(source)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            projection(source)
        session.freeze()
        replays = []
        for _ in range(2):
            poison(output)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            replays.append(output.clone())
        assert torch.equal(replays[0].view(torch.int16), expected.view(torch.int16))
        assert torch.equal(replays[0].view(torch.int16), replays[1].view(torch.int16))
        graph.reset()


@pytest.mark.parametrize("rows", [1, 8, 128, 4096])
def test_in_place_compiled_forward_has_no_copy_after_the_gemm(rows, sm103):
    """The owned output is written by the GEMM itself, through compile_forward's storage-call lowering."""
    device = torch.device("cuda", 0)
    k, n = 2048, 4096
    with prepared_projection(rows, k, n, lambda source: torch.empty((rows, n), dtype=torch.bfloat16,
                                                                     device=device)) as (projection, source, output, session):
        assert projection.in_place
        expected = copy_path(projection, source).clone()
        compiled = compile_forward(lambda: projection(source))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        replays = []
        for _ in range(2):
            poison(output)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            replays.append(output.clone())
        assert torch.equal(replays[0].view(torch.int16), expected.view(torch.int16))
        assert torch.equal(replays[0].view(torch.int16), replays[1].view(torch.int16))
        poison(output)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            graph.replay()
            torch.cuda.synchronize()
        events = sorted((event for event in profile.events()
                         if event.device_type == torch.autograd.DeviceType.CUDA),
                        key=lambda event: event.time_range.start)
        names = [event.name for event in events]
        assert names, "the profiler recorded no device activity for the replay"
        assert not [name for name in names if "memcpy" in name.lower()], names
        # The GEMM is the last device activity: no copy of its result follows it. Up to 32 rows the
        # GEMM also takes transposed scale operands, which are copied before it, not after.
        assert "gemm" in names[-1].lower() and "direct_copy" not in names[-1], names[-1]
        if rows > 32:
            assert not [name for name in names if "direct_copy" in name], names
        graph.reset()
