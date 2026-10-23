"""CLIP vision encoder with axial RoPE and projection before spatial patch merging."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLVisionEncoderLayer,
    MiniMaxM3VLVisionRotaryEmbedding,
)

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.components.weights import DirectWeightLoader

PATCH_CAPACITIES = (256, 1024, 2304)
FEATURES = 2304
patches_per_feature = 4


@dataclass(frozen=True)
class VisionWeights:
    config: dict
    patch: torch.Tensor
    norm: tuple
    layers: tuple
    projector: tuple
    merger: tuple
    shard: object

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None):
        config = json.loads((manifest.metadata_path / "config.json").read_text())
        c = config["vision_config"]
        expected = dict(
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
            intermediate_size=5120,
            patch_size=14,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
        )
        if (
            any(c.get(k) != v for k, v in expected.items())
            or config["projector_hidden_size"] != 6144
        ):
            raise ValueError("unsupported MiniMax M3 vision geometry")
        if shard.world_size != 4:
            raise ValueError(
                "MiniMax M3 vision preparation requires tensor parallelism four"
            )
        with DirectWeightLoader(
            manifest,
            device,
            prefixes=("vision_tower.", "multi_modal_projector.", "patch_merge_mlp."),
            progress=progress,
            progress_label="Load vision weights",
        ) as loader:

            def tensor(name, extent=None, *, dtype=torch.bfloat16):
                value = loader._sources[name]
                if value.dtype != dtype:
                    raise ValueError(
                        f"{name}: vision weights require {dtype} checkpoint storage"
                    )
                return loader.load_tensor(name, extent=extent)

            def affine(prefix, rows=None, columns=None):
                return (
                    tensor(
                        prefix + ".weight",
                        (
                            slice(None) if rows is None else slice(*rows),
                            slice(None) if columns is None else slice(*columns),
                        ),
                    ),
                    tensor(
                        prefix + ".bias", slice(None) if rows is None else slice(*rows)
                    ),
                )

            def norm(prefix):
                return tensor(prefix + ".weight"), tensor(prefix + ".bias")

            h = shard.partition(1280)
            m = shard.partition(5120)
            p = shard.partition(6144)
            layers = []
            for index in range(32):
                prefix = f"vision_tower.vision_model.encoder.layers.{index}"
                projections = [
                    affine(prefix + ".self_attn." + role + "_proj", rows=h)
                    for role in ("q", "k", "v")
                ]
                layers.append(
                    dict(
                        norm1=norm(prefix + ".layer_norm1"),
                        norm2=norm(prefix + ".layer_norm2"),
                        qkv=(
                            torch.cat([w for w, _ in projections]),
                            torch.cat([b for _, b in projections]),
                        ),
                        output=affine(prefix + ".self_attn.out_proj", columns=h),
                        up=affine(prefix + ".mlp.fc1", rows=m),
                        down=affine(prefix + ".mlp.fc2", columns=m),
                    )
                )
            projectors = []
            for prefix in ("multi_modal_projector", "patch_merge_mlp"):
                projectors.append(
                    (
                        affine(prefix + ".linear_1", rows=p),
                        affine(prefix + ".linear_2", columns=p),
                    )
                )
            return cls(
                c,
                tensor(
                    "vision_tower.vision_model.embeddings.patch_embedding.weight",
                    dtype=torch.float32,
                ).flatten(1),
                norm("vision_tower.vision_model.pre_layrnorm"),
                tuple(layers),
                *projectors,
                shard,
            )


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device, patches=PATCH_CAPACITIES[-1]):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((patches, 1176), torch.float32),
            zeros((patches, 2), torch.int64),
            zeros((2,)),
            zeros((1,)),
            zeros((1,)),
            zeros((1,)) - 1,
            zeros((1,), torch.int64),
            zeros((1,)),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    reduced: torch.Tensor
    mlp_up: torch.Tensor
    activated: torch.Tensor
    projector_up: torch.Tensor
    projector_activated: torch.Tensor
    projector_output: torch.Tensor
    projector_reduced: torch.Tensor
    merger_up: torch.Tensor
    merger_activated: torch.Tensor
    merger_output: torch.Tensor
    merger_reduced: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, shard, device, *, buffers=None):
        def zeros(*shape, dtype=torch.bfloat16):
            return (
                torch.zeros(shape, dtype=dtype, device=device)
                if buffers is None
                else buffers.zeros(shape, dtype)
            )

        h, m, p = (
            1280 // shard.world_size,
            5120 // shard.world_size,
            6144 // shard.world_size,
        )
        return cls(
            zeros(patches, 1176, dtype=torch.float32),
            zeros(patches, 1280),
            zeros(patches, 1280),
            zeros(patches, 3 * h),
            *(zeros(patches, h // 80, 80) for _ in range(4)),
            zeros(patches, 1280),
            zeros(patches, 1280),
            zeros(patches, m),
            zeros(patches, m),
            zeros(patches, p),
            zeros(patches, p),
            zeros(patches, 6144),
            zeros(patches, 6144),
            zeros(patches // 4, p),
            zeros(patches // 4, p),
            zeros(patches // 4, 6144),
            zeros(patches // 4, 6144),
            zeros(patches // 4, p),
        )

    def rows(self, patches):
        return type(self)(
            *(
                getattr(self, name)[
                    : patches
                    // (4 if name.startswith("merger_") or name == "output" else 1)
                ]
                for name in self.__dataclass_fields__
            )
        )


def layer_norm(weights):
    weight, bias = weights
    module = nn.LayerNorm(weight.numel(), eps=1e-5, device="meta", dtype=weight.dtype)
    module.weight = nn.Parameter(weight, requires_grad=False)
    module.bias = nn.Parameter(bias, requires_grad=False)
    return module


def axial_rope(source, positions, inverse):
    """Partial NeoX rotary transform with FP32 trigonometry and BF16 output."""
    coordinates = torch.stack(
        (torch.zeros_like(positions[:, 0]), positions[:, 0], positions[:, 1]), -1
    )
    angles = (coordinates.float()[..., None] * inverse).flatten(1)[:, None]
    cosine, sine = angles.cos(), angles.sin()
    first, second = source[..., :39].float(), source[..., 39:78].float()
    rotated = torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), -1
    ).bfloat16()
    return torch.cat((rotated, source[..., 78:]), -1)


class VisionAttention(nn.Module):
    def __init__(
        self,
        weights,
        scratch,
        positions,
        inverse,
        cu,
        *,
        runtime,
        coordinator,
        session,
        workspace,
        name,
    ):
        super().__init__()
        self.scratch, self.positions, self.inverse = scratch, positions, inverse
        s = scratch
        self.qkv = UnquantizedProjection(
            s.normalized,
            weights["qkv"][0],
            s.qkv,
            bias=weights["qkv"][1],
            session=session,
            name=name + ".qkv",
        )
        self.attention = PreparedVarlenAttention(
            s.query,
            s.key,
            s.value,
            cu,
            s.attended,
            max_seqlen=s.query.shape[0],
            session=session,
            workspace=workspace,
            name=name + ".attention",
        )
        self.output = UnquantizedProjection(
            s.attended.flatten(1),
            weights["output"][0],
            s.projected,
            session=session,
            name=name + ".output",
        )
        self.reduce = PreparedAllReduce(
            s.projected,
            s.reduced,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".reduce",
        )
        self.bias = weights["output"][1]

    def forward(self, hidden_states, **kwargs):
        s = self.scratch
        s.normalized.copy_(hidden_states)
        q, k, v = self.qkv(s.normalized).chunk(3, -1)
        s.query.copy_(axial_rope(q.view_as(s.query), self.positions, self.inverse))
        s.key.copy_(axial_rope(k.view_as(s.key), self.positions, self.inverse))
        s.value.copy_(v.view_as(s.value))
        self.attention()
        return self.reduce(self.output(s.attended.flatten(1))) + self.bias, None


class VisionMLP(nn.Module):
    def __init__(
        self,
        weights,
        source,
        up,
        activated,
        output,
        reduced,
        *,
        runtime,
        coordinator,
        session,
        name,
    ):
        super().__init__()
        self.source, self.activated = source, activated
        self.up = UnquantizedProjection(
            source,
            weights[0][0],
            up,
            bias=weights[0][1],
            session=session,
            name=name + ".up",
        )
        self.down = UnquantizedProjection(
            activated, weights[1][0], output, session=session, name=name + ".down"
        )
        self.reduce = PreparedAllReduce(
            output,
            reduced,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".reduce",
        )
        self.bias = weights[1][1]

    def forward(self, hidden_states):
        self.source.copy_(hidden_states)
        self.activated.copy_(torch.nn.functional.gelu(self.up(self.source)))
        return self.reduce(self.down(self.activated)) + self.bias


class PreparedVision:
    def __init__(
        self,
        weights,
        scratch,
        inputs,
        *,
        runtime,
        coordinator,
        session,
        workspace,
        name,
        buffers=None,
    ):
        self.weights, self.scratch, self.inputs = weights, scratch, inputs
        s = scratch

        def zeros(shape, dtype=torch.int32):
            return (
                torch.zeros(shape, dtype=dtype, device=s.hidden.device)
                if buffers is None
                else buffers.zeros(shape, dtype)
            )

        self.positions = zeros((s.hidden.shape[0], 2), torch.int64)
        self.cu, self.num_patches, self.num_features = (
            zeros((2,)),
            zeros((1,)),
            zeros((1,)),
        )
        self.inverse, _ = (
            MiniMaxM3VLVisionRotaryEmbedding.compute_axial_rope_parameters(
                SimpleNamespace(
                    rope_parameters={"rope_theta": weights.config["rope_theta"]},
                    hidden_size=1280,
                    num_attention_heads=16,
                ),
                device=s.hidden.device,
            )
        )
        self.patch = UnquantizedProjection(
            s.patches, weights.patch, s.hidden, session=session, name=name + ".patch"
        )
        self.norm = layer_norm(weights.norm)
        common = dict(runtime=runtime, coordinator=coordinator, session=session)
        self.blocks = []
        for index, w in enumerate(weights.layers):
            block = MiniMaxM3VLVisionEncoderLayer.__new__(MiniMaxM3VLVisionEncoderLayer)
            nn.Module.__init__(block)
            block.layer_norm1, block.layer_norm2 = (
                layer_norm(w["norm1"]),
                layer_norm(w["norm2"]),
            )
            block.self_attn = VisionAttention(
                w,
                s,
                self.positions,
                self.inverse,
                self.cu,
                workspace=workspace,
                name=f"{name}.{index}.attention",
                **common,
            )
            block.mlp = VisionMLP(
                (w["up"], w["down"]),
                s.normalized,
                s.mlp_up,
                s.activated,
                s.projected,
                s.reduced,
                name=f"{name}.{index}.mlp",
                **common,
            )
            self.blocks.append(block)
        self.projector = VisionMLP(
            weights.projector,
            s.hidden,
            s.projector_up,
            s.projector_activated,
            s.projector_output,
            s.projector_reduced,
            name=name + ".projector",
            **common,
        )
        self.merger = VisionMLP(
            weights.merger,
            s.projector_reduced.view(-1, 4 * 6144),
            s.merger_up,
            s.merger_activated,
            s.merger_output,
            s.merger_reduced,
            name=name + ".merger",
            **common,
        )
        self.output_columns = torch.arange(
            *weights.shard.partition(6144), device=s.output.device, dtype=torch.int64
        )

    def __call__(self):
        s = self.scratch
        hidden = self.norm(self.patch(s.patches))
        for block in self.blocks:
            hidden = block(hidden, attention_mask=None)
        s.projector_reduced.copy_(self.projector(hidden))
        s.merger_reduced.copy_(self.merger(s.projector_reduced.view(-1, 4 * 6144)))
        s.output.copy_(s.merger_reduced.index_select(1, self.output_columns))
        return s.output

    def scratch_tensors(self):
        return self.positions, self.cu, self.num_patches, self.num_features


class PreparedVisionGeneration:
    def __init__(self, encoder, state, records, frame, sequence, *, buffers=None):
        self.encoder, self.state = encoder, state
        self.errors = (
            torch.zeros(1, dtype=torch.int32, device=records.device)
            if buffers is None
            else buffers.zeros((1,), torch.int32)
        )
        self.publish = PreparedFeaturePublication(
            state, encoder.inputs, records, frame, sequence
        )

    def __call__(self):
        e, state = self.encoder, self.state
        i, r = e.inputs, state.requests
        capacity = e.scratch.patches.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features, offset = (
            i.vision_num_patches,
            i.vision_num_features,
            i.vision_feature_offset,
        )
        valid = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        valid &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid &= (count > 0) & (count <= capacity) & (count == features * 4)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | (
            (state.generation[slot] == i.vision_generation)
            & (state.count[slot] == offset)
        )
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        valid &= (state.feature_slot >= 0) & (state.feature_slot < state.arena.shape[0])
        live = torch.arange(capacity, device=count.device) < count
        pixels, positions = (
            i.vision_pixel_values[:capacity],
            i.vision_position_ids[:capacity],
        )
        valid &= (~live[:, None] | torch.isfinite(pixels)).all()
        valid &= (~live[:, None] | ((positions >= 0) & (positions < capacity))).all()
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & live[:, None], pixels, 0))
        e.positions.copy_(torch.where(valid & live[:, None], positions, 0))
        e.cu.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        e.num_patches.copy_(torch.where(valid, count, 0))
        e.num_features.copy_(torch.where(valid, features, 0))
        output = e()
        active = torch.arange(capacity // 4, device=count.device) < features
        self.errors.copy_(
            self.errors
            | (~((~active[:, None]) | torch.isfinite(output)).all()).int() * 64
        )
        self.publish(output, e.num_features, self.errors)
