"""Prepared FP8 projections with FP32 scales for 128-by-128 weight blocks."""

from dataclasses import dataclass

import torch
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall, plan_from_handle, require_prepared
from flashinfer.quantization import per_token_group_quant_8bit

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare


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


@torch.library.custom_op("flare::block_fp8_projection", mutates_args=())
def _project(source: torch.Tensor, values: torch.Tensor, scales: torch.Tensor, handle: int) -> torch.Tensor:
    quantized, activation_scales = per_token_group_quant_8bit(source, 128)
    from .block_fp8_sm103 import COMPONENT

    plan = plan_from_handle(handle)
    if plan.component_id == COMPONENT:
        return require_prepared(plan, COMPONENT, source.device).run_serialized(
            quantized, activation_scales, values, scales, None)
    return blockscaled.mm_block_fp8(quantized, activation_scales, values, scales,
                                    plan=plan, out_dtype=torch.bfloat16)


@_project.register_fake
def _fake_project(source, values, scales, handle):
    return source.new_empty((source.shape[0], values.shape[0]))


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
            self.plan = blockscaled.plan(blockscaled.FixedBlockscaledQuery(
                recipe="block_fp8", call_kind="serialized", max_rows=source.shape[0],
                in_features=k, padded_in_features=k, out_features=n,
                input_dtype="float8_e4m3fn", output_dtype="bfloat16", expected_m=source.shape[0]))

        def prepare(state):
            generator = torch.Generator(device=source.device).manual_seed(0xB12)
            trial = torch.randn(source.shape, dtype=source.dtype, device=source.device,
                                generator=generator).mul_(0.125)
            values, scales = per_token_group_quant_8bit(trial, 128)

            def produce():
                produced, produced_scales = per_token_group_quant_8bit(trial, 128)
                values.copy_(produced)
                scales.copy_(produced_scales)

            # Call closures own trial inputs; prepared plans must not retain them after priming.
            return PreparedCall(run=lambda: state.run_serialized(values, scales, weight.values,
                weight.scales, None, ab_dtype="float8_e4m3fn", sf_dtype="float32",
                c_dtype="bfloat16", sf_vec_size=128, block_fp8=True, stream=None),
                produce=produce)

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=prepare),))

    def __call__(self, source):
        result = _project(source, self.weight.values, self.weight.scales, self.plan.handle)
        return result if self.output is None else self.output.copy_(result)
