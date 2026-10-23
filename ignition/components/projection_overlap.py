"""Capture a chain of dense projections alongside independent main-stream work."""

import torch

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer


@torch.library.custom_op("flare::start_projection_chain", mutates_args=("outputs", "workspace"))
def _start(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
           gains: list[torch.Tensor | None], activation_scales: list[torch.Tensor | None], scale_kinds: list[int],
           outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        value = source
        for weight, scale, gain, activation_scale, kind, output, plan in zip(
                weights, scales, gains, activation_scales, scale_kinds, outputs, plans, strict=True):
            if scale is None:
                bf16_out(value, weight, output, plan)
            else:
                packed_out_workspace(value, weight, scale, gain, activation_scale, output, workspace,
                                      ("none", "multiplier", "reciprocal")[kind], None, plan, "any")
            value = output


@_start.register_fake
def _fake_start(source, weights, scales, gains, activation_scales, scale_kinds, outputs, workspace, plans, stream):
    pass


@torch.library.custom_op("flare::join_projection_chain", mutates_args=("outputs", "workspace"))
def _join(source: torch.Tensor, dependency: torch.Tensor, outputs: list[torch.Tensor],
          workspace: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join.register_fake
def _fake_join(source, dependency, outputs, workspace, stream):
    pass


class ConcurrentProjectionChain:
    """Run dependent projections on an auxiliary stream with one scratch suffix.

    Each projection consumes its predecessor's complete output. The recipe
    declares every concurrent main-stream plan, starts the chain, and joins
    before consuming its output or reusing source and scratch storage.
    """

    def __init__(self, projections, *, parallel_plans, session, workspace: ScratchArena):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.projections, self.parallel_plans = tuple(projections), tuple(parallel_plans)
        if not isinstance(workspace, ScratchArena) or not self.projections or not self.parallel_plans:
            raise ValueError("projection overlap requires an arena, a projection chain, and concurrent plans")
        self.arena = workspace
        self.weights, self.scales, self.outputs, self.plans = [], [], [], []
        self.gains, self.activation_scales, self.scale_kinds = [], [], []
        for projection in self.projections:
            if not isinstance(projection, (QuantizedProjection, UnquantizedProjection)) or projection.bias is not None:
                raise ValueError("projection overlap requires unbiased BF16 or blockscaled projections")
            if isinstance(projection, QuantizedProjection):
                values, scales, gain, fp4 = _weight_parts(projection.weight)
                if scales.dtype == torch.float8_e8m0fnu:
                    scales = scales.view(torch.uint8)
                width = projection.weight.in_features
                activation_scale = projection.activation_scale
                kind = projection.weight.global_scale_kind if fp4 else "none"
            else:
                values, scales, gain = projection.weight, None, None
                width, activation_scale, kind = values.shape[1], None, "none"
            if self.outputs and self.outputs[-1].shape[1] != width:
                raise ValueError("projection chain output width must match the following input width")
            self.weights.append(values)
            self.scales.append(scales)
            self.gains.append(gain)
            self.activation_scales.append(activation_scale)
            self.scale_kinds.append(("none", "multiplier", "reciprocal").index(kind))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
            for main in self.parallel_plans:
                workspace.register_parallel(main, projection.plan)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in self.projections)
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def start(self, source):
        _start(source, self.weights, self.scales, self.gains, self.activation_scales, self.scale_kinds,
               self.outputs, self.workspace, self.plans, self.stream_handle)

    def finish(self, source, dependency):
        _join(source, dependency, self.outputs, self.workspace, self.stream_handle)
        return self.outputs[-1]
