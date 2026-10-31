# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared expert routing and packed expert execution with caller-owned storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import torch
from b12x.moe import fused_moe
from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from ignition.preparation.fixtures import fused_moe as moe_fixtures
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, races_candidates, register_finalizer


@dataclass(frozen=True)
class RouterScratch:
    """Caller-owned selected logits, expert IDs, and FP32 route weights."""

    selected_logits: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, experts: int, top_k: int, buffers=None) -> RouterScratch:
        if (any(type(value) is not int or value <= 0 for value in (tokens, experts, top_k))
                or top_k > experts or experts > 1024):
            raise ValueError("router requires positive integer dimensions and top_k <= experts <= 1024")
        def zeros(dtype):
            return (torch.zeros((tokens, top_k), dtype=dtype, device=device) if buffers is None
                    else buffers.zeros((tokens, top_k), dtype))
        return cls(zeros(torch.float32), zeros(torch.int32), zeros(torch.float32).fill_(1 / top_k))

    def rows(self, tokens: int) -> RouterScratch:
        if type(tokens) is not int or not 0 < tokens <= self.ids.shape[0]:
            raise ValueError("router rows exceed scratch capacity")
        return RouterScratch(*(getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


class PreparedRouter:
    """Native top-k routing, FP32 weights, and descending-ID ties.

    Route weights retain the library's FP32 precision without a BF16 roundtrip.
    The fixed route plan has no opaque scratch or mutable binding.
    A prepared image mask can select an image-specific correction bias per row
    within the same routing operation.
    A device-side valid-token count excludes the padded suffix of packed input;
    those rows receive expert ID -1 and zero weights.
    """

    def __init__(self, logits: torch.Tensor, scratch: RouterScratch, *,
                 session: PreparationSession, name: str, score_func="softmax",
                 renormalize=True, routed_scaling_factor=1.0, correction_bias=None,
                 image_correction_bias=None, image_mask=None, num_valid_tokens=None):
        if logits.ndim != 2 or logits.dtype not in (torch.bfloat16, torch.float32) or logits.stride(1) != 1:
            raise ValueError("router logits must be BF16 or FP32 [tokens, experts] with contiguous expert columns")
        tokens, experts = logits.shape
        shape = scratch.ids.shape
        if len(shape) != 2 or shape[0] != tokens or not 0 < shape[1] <= experts <= 1024:
            raise ValueError("router output geometry must match tokens and top_k <= experts <= 1024")
        for tensor, dtype in ((scratch.selected_logits, torch.float32),
                              (scratch.ids, torch.int32), (scratch.weights, torch.float32)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != logits.device or tensor.stride(1) != 1:
                raise ValueError("router outputs require matched shapes, native dtypes/device, and contiguous top-k columns")
        self.logits = logits
        self.scratch, self.device = scratch, logits.device
        self.dtype = logits.dtype
        self.options = {key: value for key, value in (
            ("correction_bias", correction_bias), ("image_correction_bias", image_correction_bias),
            ("image_mask", image_mask), ("num_valid_tokens", num_valid_tokens)) if value is not None}
        self.shape, self.stride = logits.shape, logits.stride()
        self.plan = fused_moe.plan_route_topk(fused_moe.RouteTopKInvocation(
            num_tokens=tokens, num_experts=experts, top_k=shape[1], logits_dtype=str(logits.dtype).removeprefix("torch."),
            score_func=score_func, renormalize=renormalize, routed_scaling_factor=routed_scaling_factor,
            has_correction_bias=correction_bias is not None,
            has_image_correction_bias=image_correction_bias is not None,
            has_image_mask=image_mask is not None,
            has_valid_tokens=num_valid_tokens is not None,
            logits_row_stride=logits.stride(0), topk_row_stride=scratch.selected_logits.stride(0),
            ids_row_stride=scratch.ids.stride(0), weights_row_stride=scratch.weights.stride(0)))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(logits, scratch.selected_logits, scratch.ids, scratch.weights, **self.options),
            owners=(logits, scratch)),
            benchmark_call=(lambda state: moe_fixtures.route_topk_benchmark_call(
                state, logits, scratch, self.options, session=session, name=name))
            if races_candidates(self.plan, logits.device) else None),))

    def __call__(self, logits: torch.Tensor) -> None:
        if (logits.dtype != self.dtype or logits.shape != self.shape
                or logits.stride() != self.stride or logits.device != self.device):
            raise ValueError("router logits must preserve the prepared dtype, shape, strides, and device")
        fused_moe.route_topk(self.plan, logits, self.scratch.selected_logits,
                             self.scratch.ids, self.scratch.weights, **self.options)


class PreparedRoutedExperts:
    """Prepared native expert execution with fixed source and route storage.

    Recipes supply expert geometry and enforce their complete model contract.
    Reduction order follows the execution plan selected by b12x.
    """

    def __new__(cls, source=None, experts=None, *args, **kwargs):
        # Experts loaded for SM103 execute through FlashInfer's native kernels. Unpickling, as compiler
        # workers do with a forward's metadata, creates the instance without arguments.
        from ignition.components.moe_sm103 import Sm103Experts, Sm103RoutedExperts

        if isinstance(experts, Sm103Experts):
            return Sm103RoutedExperts(source, experts, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, source, experts, ids, weights, output, *, session: PreparationSession, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None, router_plan=None,
                 a4_prefill=False, routing=None, router=None, route_output=None):
        # b12x owns the final weight/scale planes after repacking. Classify those
        # exact tensors, not opaque execution scratch or the discarded source.
        for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                      "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
            record_numeric_tensor(getattr(experts._impl, field))
        representation = experts._impl.representation
        if representation is not None:
            # A plan can keep alternate scale layouts beside its canonical planes.
            for field in ("w13_scale", "w2_scale", "w13_sfb", "w2_sfb",
                          "w13_global_scale", "w2_global_scale", "micro_w13_scale", "micro_w2_scale",
                          "micro_w13_global_scale", "micro_w2_global_scale"):
                tensor = getattr(representation.value, field, None)
                if tensor is not None:
                    record_numeric_tensor(tensor)
        decoder = experts._impl.nvfp4_csf
        if decoder is not None:
            # Compressed records contain numeric offsets, not device pointers.
            for plane in (decoder.first, decoder.second):
                for tensor in (plane.fixed, plane.exceptions, plane.task_offsets, plane.value_lut):
                    if tensor is not None:
                        record_numeric_tensor(tensor)
            record_numeric_tensor(decoder.active)
            for plane in decoder.indexed_scales or ():
                record_numeric_tensor(plane.storage)
        # A16 experts prepared with A4 capability run large prefill calls with quantized activations.
        precision = {} if experts.plan.activation.outer_scale == "token" else {"a4_prefill": a4_prefill}
        self.source = source
        self.output = output
        self.output_source = output
        if route_output is not None and experts.plan.activation.outer_scale == "token":
            from ignition.components.collectives import WeightedGroupSource

            output = route_output
            self.output = output
            self.output_source = WeightedGroupSource(output, weights, ids, experts.num_experts)
        self.scratch = None
        self.router = router
        self.route_inputs = {}
        if router is not None and experts.plan.activation.outer_scale == "token":
            query = router.plan.query
            routing = replace(routing or fused_moe.RoutingSpec(),
                logits_dtype=router.dtype, score_func=query.score_func,
                renormalize=query.renormalize, routed_scaling_factor=query.routed_scaling_factor,
                has_correction_bias=query.has_correction_bias,
                has_image_correction_bias=query.has_image_correction_bias,
                has_image_mask=query.has_image_mask, has_valid_tokens=query.has_valid_tokens)
            self.route_inputs = dict(router_logits=router.logits,
                selected_logits=router.scratch.selected_logits, **router.options)
        integrated_query = router.plan.query if self.route_inputs else None
        invocation = FrozenMapping({**precision, "tuning_route_pattern": TUNING_WORKLOAD_VERSION,
            "tuning_route_source": "b12x",
            **({"tuning_execution": "cuda_graph"} if experts.has_csf_blocks else {})})
        self.scale_plan = fused_moe.plan_scale_expansion(
            experts, routes=ids.numel(), ids_dtype=ids.dtype, invocation=invocation,
        ) if experts.has_csf_blocks else None
        self.plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=source.shape[0], top_k=ids.shape[1],
                route_num_experts=experts.num_experts if self.route_inputs else 0,
                output_layout="routes" if output.ndim == 3 else "token"),
            scale_expansion=self.scale_plan, invocation=invocation, routing=routing,
        )

        def prepare_call(state):
            scratch = planned_scratch(state.scratch.scratch_specs(), workspace)
            self.scratch = scratch
            route_inputs = dict(self.route_inputs)
            if route_inputs:
                route_inputs["router_logits"] = torch.zeros_like(router.logits)
                route_inputs["selected_logits"] = torch.zeros_like(router.scratch.selected_logits)
                if "num_valid_tokens" in route_inputs:
                    route_inputs["num_valid_tokens"] = torch.full_like(route_inputs["num_valid_tokens"], source.shape[0])
            binding = state.bind(a=source, topk_ids=torch.zeros_like(ids), topk_weights=torch.full_like(weights, 1 / ids.shape[1]),
                output=output, scratch=scratch, input_scales_static=True, **precision, **route_inputs)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        scale_requests = () if self.scale_plan is None else (self.scale_plan.request(
            name=f"{name}/scales",
            prepare_call=lambda state: moe_fixtures.scale_expansion_call(state, ids=torch.zeros_like(ids), experts=experts),
            benchmark_call=partial(moe_fixtures.scale_expansion_benchmark_call,
                source=source, experts=experts, top_k=ids.shape[1], session=session, name=name,
                allocate_router=RouterScratch.allocate, router_plan=router_plan,
                integrated_query=integrated_query, output_template=output)),)
        declare(session, (*scale_requests, self.plan.request(name=name, prepare_call=prepare_call,
            dependencies=tuple(request.name for request in scale_requests),
            benchmark_call=partial(moe_fixtures.benchmark_call, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate, router_plan=router_plan, precision=precision,
                integrated_query=integrated_query, output_template=output)),))

        def finalize():
            self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = fused_moe.bind(self.plan, a=source, topk_ids=ids, topk_weights=weights,
                output=output, scratch=self.scratch, input_scales_static=True, **precision, **self.route_inputs)

        register_finalizer(session, finalize, workspace=workspace)

    def route(self, logits: torch.Tensor) -> None:
        if self.route_inputs:
            if logits is not self.router.logits:
                raise ValueError("integrated expert routing requires the prepared logit storage")
        else:
            self.router(logits)

    def __call__(self) -> torch.Tensor:
        fused_moe.run(binding=self.binding)
        return self.output


