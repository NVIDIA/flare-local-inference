# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM decoder-layer composition with b12x operators and tensor-parallel reductions."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC, PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP

from .kda import KDAScratch, PreparedGlmKDA
from .mla import MLAScratch, PreparedGlmMLA
from .moe import MoeScratch, PreparedGlmMoe
from .weights import MoEWeights


@dataclass(frozen=True)
class LayerScratch:
    attention_hc: MHCScratch
    mlp_hc: MHCScratch
    attention: KDAScratch | MLAScratch
    attention_output: torch.Tensor
    mlp: MLPScratch | MoeScratch
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, world_size, device, *, sparse, block_type, mode, blocks=0):
        if world_size not in (1, 2, 4):
            raise ValueError("GLM layers require one, two or four ranks")
        def buffer(width):
            return torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
        if block_type == "linear_attention":
            attention = KDAScratch.allocate(tokens, 64 // world_size, device)
        elif block_type == "indexed_attention":
            attention = MLAScratch.allocate(tokens, 64 // world_size, blocks, mode, device)
        else:
            raise ValueError("GLM layers require linear_attention or indexed_attention")
        return cls(MHCScratch.allocate(tokens, 4096, device), MHCScratch.allocate(tokens, 4096, device),
                   attention, buffer(4096),
                   MoeScratch.allocate(tokens, world_size, device) if sparse else
                   MLPScratch.allocate(tokens, 4096, 12288 // world_size, device), buffer(4096))


def _install_mhc(module, norm, weights, prefix, norm_name):
    for attribute in ("fn", "scale", "base"):
        key = f"{prefix}_{attribute}"
        weights[key] = weights[key].float()
        setattr(module, attribute, torch.nn.Parameter(weights[key], requires_grad=False))
    norm.weight = torch.nn.Parameter(weights[f"{norm_name}.weight"], requires_grad=False)
    return MHCWeights(module.fn, module.scale, module.base, norm.weight)


class PreparedGlmLayer:
    """KDA or sparse MLA layer with replicated mHC and two output all-reduces.

    Each rank owns a head/intermediate partition and identical four-stream
    residuals. Post-attention mixing, FFN mixing prediction, and FFN RMSNorm
    share one fused b12x operation. Scratch may be reused by serialized layers;
    recurrent state and escaping outputs must remain separately owned.
    """

    def __init__(self, module, attention_weights, mlp_weights, residual_weights,
                 source, output, scratch: LayerScratch, state, metadata, *,
                 mode, runtime, coordinator, session, name, workspace, carry=None, defer_post=False,
                 max_cache_pages=None, context=None, kda_records=None):
        if module.block_type not in ("linear_attention", "indexed_attention"):
            raise ValueError("prepared GLM layer requires canonical KDA or sparse MLA attention")
        rows = scratch.attention.output.shape[0]
        if (source.shape != ((rows, 4, 4096) if carry is None else (rows, 4096))
                or output.shape != (rows, 4, 4096) or source.dtype != torch.bfloat16
                or output.dtype != source.dtype or output.device != source.device):
            raise ValueError("GLM layer residuals require BF16 [tokens, 4, 4096] on one device")
        self.module, self.scratch, self.output, self.carry = module, scratch, output, carry
        attention_mhc = _install_mhc(module.attn_hc, module.input_layernorm, residual_weights,
                                    "hc_attn", "input_layernorm")
        mlp_mhc = _install_mhc(module.ffn_hc, module.post_attention_layernorm, residual_weights,
                              "hc_ffn", "post_attention_layernorm")
        options = {"rms_eps": module.attn_hc.input_norm.eps,
                   "hc_eps": module.attn_hc.hc_eps, "sinkhorn_iters": module.attn_hc.hc_sinkhorn_iters,
                   "norm_eps": module.input_layernorm.variance_epsilon}
        inputs = (source,) if carry is None else (source, carry.residual, carry.post, carry.comb)
        self.attention_hc = PreparedMHC(inputs, attention_mhc, scratch.attention_hc,
            session=session, name=f"{name}.attn_hc", workspace=workspace, **options)
        attention_args = (module.self_attn, attention_weights, scratch.attention_hc.normalized,
                          scratch.attention, state, metadata)
        if module.block_type == "linear_attention":
            self.attention = PreparedGlmKDA(*attention_args, mode=mode, session=session, name=f"{name}.kda",
                                            workspace=workspace, checkpoint_export=mode == "prefill",
                                            records=kda_records)
        else:
            self.attention = PreparedGlmMLA(*attention_args, mode=mode, session=session, name=f"{name}.mla",
                                            workspace=workspace, max_cache_pages=max_cache_pages,
                                            context=context)
        # One rank reduces nothing, so its consumers read each sub-block's output in place. The attention
        # output is dead once the FFN mixing has read it, and the MLP output is dead once the post mixing
        # (here, or fused into the next layer's mixing) has read it; no operation between a producer and
        # its consumer writes either, so overlaid scratch cannot reuse their storage in between.
        self.attention_reduce = PreparedAllReduce(scratch.attention.output,
            None if runtime is None else scratch.attention_output, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.attention_reduce")
        self.mlp_hc = PreparedMHC((self.attention_reduce.output, scratch.attention_hc.residual,
                                  scratch.attention_hc.post, scratch.attention_hc.comb),
            mlp_mhc, scratch.mlp_hc, session=session, name=f"{name}.ffn_hc", workspace=workspace, **options)
        if isinstance(mlp_weights, MoEWeights):
            self.mlp = PreparedGlmMoe(module.mlp, scratch.mlp_hc.normalized, mlp_weights, scratch.mlp,
                                      session=session, name=f"{name}.moe", workspace=workspace,
                                      prefill=mode == "prefill")
        else:
            self.mlp = PreparedMLP(scratch.mlp_hc.normalized, mlp_weights, scratch.mlp.gate_up,
                scratch.mlp.intermediate, scratch.mlp.output, session=session, name=f"{name}.mlp", workspace=workspace,
                swiglu_limit=module.mlp.swiglu_limit)
        self.mlp_reduce = PreparedAllReduce(scratch.mlp.output,
            None if runtime is None else scratch.mlp_output, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.mlp_reduce")
        # The reduced MLP output the following layer's mixing, or the final post mixing, reads.
        self.result = self.mlp_reduce.output
        self.post = None if defer_post else PreparedMHCPost(self.result, scratch.mlp_hc.residual,
            scratch.mlp_hc.post, scratch.mlp_hc.comb, output, session=session, name=f"{name}.post")

    def __call__(self, source):
        s = self.scratch
        inputs = (source,) if self.carry is None else (
            source, self.carry.residual, self.carry.post, self.carry.comb)
        self.attention_reduce(self.attention(self.attention_hc(*inputs)))
        self.mlp_hc(self.attention_reduce.output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        self.mlp_reduce(self.mlp(s.mlp_hc.normalized))
        if self.post is None:
            return self.result
        return self.post(self.result, s.mlp_hc.residual, s.mlp_hc.post, s.mlp_hc.comb)
