"""Captured GLM vision tower with canonical blocks and prepared projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention


@dataclass(frozen=True)
class VisionMetadata:
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches, images, device):
        if patches <= 0 or patches % 4 or images <= 0:
            raise ValueError("GLM vision requires positive image capacity and patch capacity divisible by four")
        return cls(torch.zeros((patches, 2), dtype=torch.int64, device=device),
                   torch.zeros(images + 1, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor
    projected: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    merged: torch.Tensor
    merger_gate: torch.Tensor
    merger_up: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device):
        if patches <= 0 or patches % 4:
            raise ValueError("GLM vision patch capacity must be positive and divisible by four")
        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        return cls(zeros(patches, 1176), zeros(patches, 1024), zeros(patches, 3072),
                   zeros(patches, 16, 64), zeros(patches, 16, 64), zeros(patches, 16, 64),
                   zeros(patches, 1024), zeros(patches, 1024),
                   zeros(patches, 4096), zeros(patches, 4096),
                   zeros(patches // 4, 4096), zeros(patches // 4, 10240),
                   zeros(patches // 4, 10240), zeros(patches // 4, 4096))

    def rows(self, patches):
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("GLM vision view exceeds its patch capacity")
        merged = {"merged", "merger_gate", "merger_up", "output"}
        return type(self)(*(getattr(self, name)[:patches // 4 if name in merged else patches]
                            for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class VisionWeights:
    parameters: dict[str, torch.Tensor]
    projections: dict[str, object]
    downsample: torch.Tensor

    @classmethod
    def load(cls, module, loader):
        parameters, projections = {}, {}
        declarations = loader.quantization
        for path, _ in module.named_parameters():
            prefix = "model.visual." + path.removesuffix(".weight")
            if path.endswith(".weight") and prefix in declarations:
                projections[path.removesuffix(".weight")] = loader.load_linear(prefix)
            else:
                parameters[path] = loader.load_tensor("model.visual." + path)
        # Patch rows are ordered [merge_h, merge_w, channel]. Conv2d stores
        # weights as [out, channel, merge_h, merge_w]; transpose once at load.
        downsample = parameters["downsample.weight"].permute(0, 2, 3, 1).reshape(4096, 4096).contiguous()
        return cls(parameters, projections, downsample)


class _Linear(torch.nn.Module):
    def __init__(self, projection, bias=None):
        super().__init__()
        self.projection, self.bias = projection, bias

    def forward(self, source):
        result = self.projection(source)
        if self.bias is not None:
            result.add_(self.bias)
        return result


class _Attention(torch.nn.Module):
    def __init__(self, module, scratch, metadata, weights, *, session, name, shared=None,
                 projection_workspace=None, attention_workspace=None):
        super().__init__()
        self.scratch = scratch
        self.q_norm, self.k_norm = module.q_norm, module.k_norm
        self.qkv = QuantizedProjection(scratch.hidden, weights["qkv"], scratch.qkv,
            session=session, name=f"{name}.qkv", bias=module.qkv.bias, workspace=projection_workspace)
        self.proj = QuantizedProjection(scratch.attention, weights["proj"], scratch.projected,
            session=session, name=f"{name}.proj", bias=module.proj.bias, workspace=projection_workspace)
        self.attention = shared.attention if shared is not None else PreparedVarlenAttention(
            scratch.query, scratch.key, scratch.value, metadata.cu_seqlens,
            scratch.attention.view_as(scratch.query), session=session, workspace=attention_workspace, name=name)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        s = self.scratch
        projected = self.qkv(hidden_states).view(-1, 3, 16, 64)
        query, key = apply_rotary_pos_emb_vision(self.q_norm(projected[:, 0]), self.k_norm(projected[:, 1]),
                                               *position_embeddings)
        s.query.copy_(query)
        s.key.copy_(key)
        s.value.copy_(projected[:, 2])
        self.attention()
        return self.proj(s.attention)


class PreparedVision:
    """Encode packed patches with a fixed capacity and device-resident lengths.

    Canonical GLM blocks retain RMS normalization, Q/K normalization, rotary
    positions, clamped SwiGLU, and residual semantics. Prepared projections
    retain checkpoint precision. Each capacity owns a canonical module tree;
    weights and sequential scratch may be shared across capacities.
    """

    def __init__(self, module, scratch, metadata, weights, *, session, name="vision",
                 projection_workspace=None, attention_workspace=None):
        config = module.config
        expected = dict(hidden_size=1024, num_heads=16, intermediate_size=4096, depth=24,
                        out_hidden_size=4096, projection_intermediate_size=10240,
                        patch_size=14, temporal_patch_size=2, spatial_merge_size=2, in_channels=3)
        if any(getattr(config, key) != value for key, value in expected.items()):
            raise ValueError("GLM vision geometry differs from the checkpoint contract")
        capacity, device = scratch.hidden.shape[0], scratch.hidden.device
        if metadata.position_ids.shape != (capacity, 2):
            raise ValueError("GLM vision metadata and scratch capacities differ")
        if metadata.num_patches.item() or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("GLM vision preparation requires inactive metadata")
        self.module, self.scratch, self.metadata = module, scratch, metadata
        for path, tensor in weights.parameters.items():
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))
        self.patch_embed = _Linear(UnquantizedProjection(scratch.patches,
            module.patch_embed.proj.weight.flatten(1), scratch.hidden,
            session=session, name=f"{name}.patch_embed"), module.patch_embed.proj.bias)
        module.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(config).to(device)
        for index, block in enumerate(module.blocks):
            prefix = f"blocks.{index}"
            block.attn = _Attention(block.attn, scratch, metadata,
                {key: weights.projections[f"{prefix}.attn.{key}"] for key in ("qkv", "proj")},
                session=session, name=f"{name}.{prefix}.attn",
                shared=module.blocks[0].attn if index else None,
                projection_workspace=projection_workspace, attention_workspace=attention_workspace)
            for key, source, output in (("gate_proj", scratch.hidden, scratch.gate),
                                       ("up_proj", scratch.hidden, scratch.up),
                                       ("down_proj", scratch.gate, scratch.projected)):
                original = getattr(block.mlp, key)
                setattr(block.mlp, key, _Linear(QuantizedProjection(source,
                    weights.projections[f"{prefix}.mlp.{key}"], output,
                    session=session, name=f"{name}.{prefix}.mlp.{key}",
                    activation_mode="auto", bias=original.bias, workspace=projection_workspace)))
        self.downsample = _Linear(UnquantizedProjection(scratch.hidden.view(-1, 4096),
            weights.downsample, scratch.merged, session=session, name=f"{name}.downsample"), module.downsample.bias)
        for key, source, output in (("proj", scratch.merged, scratch.output),
                                   ("gate_proj", scratch.output, scratch.merger_gate),
                                   ("up_proj", scratch.output, scratch.merger_up),
                                   ("down_proj", scratch.merger_gate, scratch.output)):
            original = getattr(module.merger, key)
            setattr(module.merger, key, _Linear(QuantizedProjection(source,
                weights.projections[f"merger.{key}"], output,
                session=session, name=f"{name}.merger.{key}",
                activation_mode="auto", bias=original.bias, workspace=projection_workspace)))
        self.output_rows = torch.arange(capacity // 4, device=device)

    def __call__(self, pixels):
        module, scratch, metadata = self.module, self.scratch, self.metadata
        scratch.patches.copy_(pixels)
        hidden = self.patch_embed(scratch.patches)
        rope = module.rotary_pos_emb(hidden, metadata.position_ids)
        for block in module.blocks:
            hidden = block(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rope)
        hidden = module.post_layernorm(hidden)
        hidden = self.downsample(hidden.view(-1, 4096))
        output = module.merger(hidden)
        output.masked_fill_((self.output_rows >= metadata.num_patches // 4)[:, None], 0)
        return output
