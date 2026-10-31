# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM sigmoid routing, NVFP4 experts, and the MXFP8 shared expert."""

from dataclasses import dataclass

import torch

from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch


@dataclass(frozen=True)
class MoeScratch:
    router_logits: torch.Tensor
    router: RouterScratch
    shared: MLPScratch
    routed: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world_size, device):
        if world_size not in (1, 2, 4):
            raise ValueError("GLM experts require one, two or four ranks")
        return cls(torch.zeros((tokens, 288), dtype=torch.float32, device=device),
                   RouterScratch.allocate(tokens, device, experts=288, top_k=8),
                   MLPScratch.allocate(tokens, 4096, 2048 // world_size, device),
                   torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=device),
                   torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=device))

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("GLM MoE rows exceed scratch capacity")
        return MoeScratch(self.router_logits[:tokens], self.router.rows(tokens), self.shared.rows(tokens),
                          self.routed[:tokens], self.output[:tokens])


class PreparedGlmMoe:
    """One rank's intermediate-sharded expert contribution before all-reduce."""

    def __init__(self, module, source, weights, scratch: MoeScratch, *, session, name, workspace, prefill=False):
        gate = module.gate
        if (gate.num_group != 1 or gate.topk_group != 1 or gate.top_k != 8 or gate.num_experts != 288
                or not gate.norm_topk_prob or gate.routed_scaling_factor != 2.5):
            raise ValueError("GLM routing requires one group, sigmoid top-8 of 288, normalization and scale 2.5")
        if (weights.experts.num_experts != 288 or weights.experts.hidden_size != 4096
                or weights.experts.intermediate_size not in (512, 1024, 2048)):
            raise ValueError("GLM MoE requires all 288 experts with a TP1/TP2/TP4 intermediate partition")
        if source.dtype != torch.bfloat16 or source.shape != scratch.output.shape or source.shape[1] != 4096:
            raise ValueError("GLM MoE input must match BF16 [tokens,4096] scratch")
        self.source, self.scratch, self.output = source, scratch, scratch.output
        gate.weight = torch.nn.Parameter(weights.router, requires_grad=False)
        gate.e_score_correction_bias = weights.correction
        self.router_projection = UnquantizedProjection(source, weights.router, scratch.router_logits,
            session=session, name=f"{name}.router")
        self.routing = PreparedRouter(scratch.router_logits, scratch.router, session=session,
            name=f"{name}.route_topk", score_func="sigmoid", renormalize=True,
            routed_scaling_factor=2.5, correction_bias=weights.correction)
        self.shared = PreparedMLP(source, weights.shared, scratch.shared.gate_up, scratch.shared.intermediate,
            scratch.shared.output, session=session, name=f"{name}.shared", workspace=workspace,
            swiglu_limit=module.config.swiglu_limit)
        self.experts = PreparedRoutedExperts(source, weights.experts, scratch.router.ids,
            scratch.router.weights, scratch.routed, session=session, name=f"{name}.experts", workspace=workspace,
            router_plan=self.routing.plan,
            a4_prefill=prefill and weights.experts_a4_prefill)
        self.concurrent_shared = None
        if source.shape[0] <= 256 and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.routing.plan, self.experts.plan),
                session=session, workspace=workspace, defer_launch=source.shape[0] == 1)

    def __call__(self, source):
        if source is not self.source and (source.data_ptr() != self.source.data_ptr() or source.shape != self.source.shape):
            raise ValueError("prepared GLM MoE requires its fixed source storage")
        self.router_projection(source)
        if self.concurrent_shared is not None:
            self.concurrent_shared.start(source)
        self.routing(self.scratch.router_logits)
        if self.concurrent_shared is None and getattr(self.experts, "reduces_into_shared", False):
            # SM103 serial order: the shared expert runs first, then one kernel sums the routed FC2
            # rows (rounded to BF16) with it, so the routed sum is never materialized.
            shared = self.shared(source)
            self.experts.stage_rows()
            return self.experts.reduce(shared, self.output)
        routed = self.experts()
        shared = (self.shared(source) if self.concurrent_shared is None
                  else self.concurrent_shared.finish(source, routed))
        torch.add(routed, shared, out=self.output)
        return self.output
