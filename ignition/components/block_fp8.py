# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP8 projections with FP32 scales for 128-by-128 weight blocks."""

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall, plan_from_handle, require_prepared

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare

_GROUP = 128
_GROUPS_PER_PROGRAM = 16


@triton.jit
def _quantize_token_groups(Source, Values, Scales, K: tl.constexpr, GROUPS: tl.constexpr):
    """E4M3 values and FP32 scales for each 128-element group of a BF16 row.

    One program quantizes ``GROUPS`` consecutive groups of one row. Per group,
    ``scale = max(amax, 1e-10) / 448`` and ``value = E4M3(clamp(x * (1 / scale)))``,
    with IEEE round-to-nearest division, the arithmetic of FlashInfer's
    ``per_token_group_quant_8bit`` with row-major scales.
    """
    row = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    columns = tl.arange(0, 128)
    live = groups < K // 128
    offsets = row * K + groups[:, None] * 128 + columns[None, :]
    x = tl.load(Source + offsets, live[:, None], 0.).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
    scale = tl.math.div_rn(amax, 448.0)
    inverse = tl.math.div_rn(1.0, scale)
    quantized = tl.minimum(tl.maximum(x * inverse[:, None], -448.0), 448.0)
    tl.store(Values + offsets, quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True), live[:, None])
    tl.store(Scales + row * (K // 128) + groups, scale, live)


# b12x evicts compile-planning placeholders only from the Triton functions it tracks.
_NATIVE_JITS.add(_quantize_token_groups)


@program_cache(scope="preparation")
def compile_token_group_quantization(in_features, ordinal):
    """Compile the quantizer for rows of ``in_features`` BF16 values; the row count is the launch grid."""
    with torch.cuda.device(ordinal):
        return _quantize_token_groups.warmup(torch.bfloat16, torch.uint8, torch.float32,
            K=in_features, GROUPS=_GROUPS_PER_PROGRAM, grid=(1, 1, 1), num_warps=4)


def quantization_job(in_features, ordinal):
    return CompileJob.create("ignition.components.block_fp8:compile_token_group_quantization",
                             in_features, ordinal)


_LOADED = {}


def load_quantizer(in_features, ordinal):
    """Compile if needed and load the quantizer for rows of ``in_features`` on device ``ordinal``.

    Called where a projection's prepared state is materialized, before kernel resolution freezes.
    """
    key = (in_features, ordinal)
    if key not in _LOADED:
        _LOADED[key] = load_programs(compile_token_group_quantization(in_features, ordinal))
    return _LOADED[key]


def _quantizer(in_features, ordinal):
    try:
        return _LOADED[(in_features, ordinal)]
    except KeyError:
        raise RuntimeError(f"the block-FP8 activation quantizer for {in_features} input features was not "
                           "loaded during preparation; a missing compile job or load fails here, "
                           "never by compiling at run time") from None


def quantize_token_groups(source: torch.Tensor):
    """Row-major E4M3 values and FP32 scales ``[rows, K / 128]`` of a contiguous BF16 matrix."""
    rows, k = source.shape
    if source.dtype != torch.bfloat16 or k % _GROUP or not source.is_contiguous():
        raise ValueError("token-group quantization requires contiguous BF16 rows of whole 128-element groups")
    values = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=source.device)
    scales = torch.empty((rows, k // _GROUP), dtype=torch.float32, device=source.device)
    if rows:
        # Triton specializes pointer alignment; the compiled program assumes 16-byte-aligned operands.
        for operand in (source, values, scales):
            if operand.data_ptr() % 16:
                raise ValueError("token-group quantization operands must be 16-byte aligned")
        ordinal = source.device.index if source.device.index is not None else torch.cuda.current_device()
        _quantizer(k, ordinal)[(rows, triton.cdiv(k // _GROUP, _GROUPS_PER_PROGRAM), 1)](
            source, values.view(torch.uint8), scales)
    return values, scales


@dataclass(frozen=True)
class BlockFP8Weight:
    values: torch.Tensor
    scales: torch.Tensor

    @property
    def in_features(self):
        return self.values.shape[1]

    @property
    def out_features(self):
        return self.values.shape[0]


# Serialized block-FP8 operand declaration shared by tuning and execution.
GEMM_OPERANDS = dict(ab_dtype="float8_e4m3fn", sf_dtype="float32", c_dtype="bfloat16", sf_vec_size=128,
                     block_fp8=True, stream=None)


def _gemm(source, values, scales, handle, out):
    """Quantize ``source`` and multiply; with ``out``, the GEMM writes that buffer and returns it."""
    quantized, activation_scales = quantize_token_groups(source)
    from .block_fp8_sm103 import COMPONENT

    plan = plan_from_handle(handle)
    if plan.component_id == COMPONENT:
        return require_prepared(plan, COMPONENT, source.device).run_serialized(
            quantized, activation_scales, values, scales, None, out=out)
    if out is not None:
        raise RuntimeError("only the SM103 block-FP8 GEMM writes into a caller-provided output")
    return blockscaled.mm_block_fp8(quantized, activation_scales, values, scales,
                                    plan=plan, out_dtype=torch.bfloat16)


@torch.library.custom_op("flare::block_fp8_projection", mutates_args=())
def _project(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor, handle: int) -> torch.Tensor:
    return _gemm(source, values, scales, handle, None)


@_project.register_fake
def _fake_project(source, values, scales, handle):
    return source.new_empty((source.shape[0], values.shape[0]))


@torch.library.custom_op("flare::block_fp8_projection_into", mutates_args=("out",))
def _project_into(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor, out: torch.Tensor,
                  handle: int) -> None:
    _gemm(source, values, scales, handle, out)


@_project_into.register_fake
def _fake_project_into(source, values, scales, out, handle):
    pass


def writes_in_place(output):
    """Whether a GEMM can write ``output`` directly: contiguous and 16-byte aligned."""
    return output is not None and output.is_contiguous() and (output.data_ptr() % 16 == 0 or output.numel() == 0)


def takes_output(handle):
    """Whether the planned GEMM accepts a caller-provided output; only the SM103 GEMM does."""
    from .block_fp8_sm103 import COMPONENT

    return plan_from_handle(handle).component_id == COMPONENT


class BlockFP8Projection:
    def __init__(self, source, weight, output=None, *, session, name):
        n, k = weight.values.shape
        if (source.ndim != 2 or source.shape[1] != k or source.dtype != torch.bfloat16
                or weight.values.dtype != torch.float8_e4m3fn or weight.scales.dtype != torch.float32
                or k % 128 or n % 128 or weight.scales.shape != (n // 128, k // 128)
                or not weight.values.is_contiguous() or not weight.scales.is_contiguous()):
            raise ValueError("block-FP8 projection requires BF16 activations, E4M3 weights and FP32 block-128 scales")
        self.weight = weight
        self.bias = None
        for tensor in (weight.values, weight.scales):
            record_numeric_tensor(tensor)
        self.output = output
        if output is not None and (output.shape != (source.shape[0], n) or output.dtype != source.dtype
                                   or output.device != source.device):
            raise ValueError("block-FP8 output must match projection rows, columns, dtype and device")
        from .sm103 import is_sm103

        if is_sm103(source.device):
            from .block_fp8_sm103 import plan

            self.plan = plan(source.shape[0], k, n, source.device)
        else:
            gemm = blockscaled.plan(blockscaled.FixedBlockscaledQuery(
                recipe="block_fp8", call_kind="serialized", max_rows=source.shape[0],
                in_features=k, padded_in_features=k, out_features=n,
                input_dtype="float8_e4m3fn", output_dtype="bfloat16", expected_m=source.shape[0]))
            # The plan's compile phase must also resolve the activation quantizer.
            self.plan = replace(gemm, _compile_jobs=lambda config, detected: (
                *gemm._compile_jobs(config, detected), quantization_job(k, detected.ordinal)))
        # The SM103 GEMM writes a contiguous, 16-byte-aligned output in place. Strided outputs, such as
        # column slices of a grouped projection's buffer, and other architectures' GEMMs take the result
        # tensor and copy it.
        self.in_place = writes_in_place(output) and takes_output(self.plan.handle)

        def prepare(state):
            load_quantizer(k, source.device.index if source.device.index is not None
                           else torch.cuda.current_device())
            generator = torch.Generator(device=source.device).manual_seed(0xB12)
            trial = torch.randn(source.shape, dtype=source.dtype, device=source.device,
                                generator=generator).mul_(0.125)
            values, scales = quantize_token_groups(trial)

            def produce():
                produced, produced_scales = quantize_token_groups(trial)
                values.copy_(produced)
                scales.copy_(produced_scales)

            # Call closures own trial inputs; prepared plans must not retain them after priming.
            return PreparedCall(run=lambda: state.run_serialized(values, scales, weight.values,
                weight.scales, None, **GEMM_OPERANDS), produce=produce)

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=prepare),))

    def __call__(self, source):
        if self.in_place:
            _project_into(source, self.weight.values, self.weight.scales, self.output, self.plan.handle)
            return self.output
        result = _project(source, self.weight.values, self.weight.scales, self.plan.handle)
        return result if self.output is None else self.output.copy_(result)
