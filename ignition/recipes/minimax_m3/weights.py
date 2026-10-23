"""Direct MiniMax M3 transfers preserving MXFP8 and NVFP4 checkpoint scales."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights

from .config import ModelConfig, TEXT_PREFIX


@dataclass(frozen=True)
class LayerWeights:
    qkv_index: object
    output: object
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_q_norm: torch.Tensor | None
    index_k_norm: torch.Tensor | None
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights
    router: torch.Tensor | None
    correction: torch.Tensor | None
    experts: object | None


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    allocated_bytes: int
    draft: object = None
    vision: object = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, package_storage=None,
             expert_scale_compression=None, dspark=False, vision=True):
        before = torch.cuda.memory_allocated(device)
        loading = nullcontext if package_storage is None else package_storage.transient
        with loading(), MiniMaxWeightLoader(manifest, device, shard=shard, progress=progress,
                                          expert_scale_compression=expert_scale_compression) as loader:
            config = loader.config
            embedding = loader.load_tensor(TEXT_PREFIX + ".embed_tokens.weight",
                extent=(slice(None), slice(*shard.partition(config.hidden, alignment=64))))
            norm = loader.norm(TEXT_PREFIX + ".norm.weight", config.hidden)
            first, last, width = config.vocabulary_partition(shard)
            source = loader._sources["language_model.lm_head.weight"]
            if source.dtype != torch.bfloat16 or source.shape != (config.vocabulary, config.hidden):
                raise ValueError("MiniMax M3 vocabulary head must retain its BF16 checkpoint storage")
            head = torch.zeros((width, config.hidden), dtype=torch.bfloat16, device=device)
            loader._copy(head[:last - first], source[first:last])
            loader._session.flush()
            layers = tuple(loader.load_layer(index) for index in range(config.layers))
        draft_weights = vision_weights = None
        with loading():
            if dspark:
                from .draft import DraftConfig, DraftWeights

                config = replace(config, draft=DraftConfig.load(manifest.snapshot, config))
                draft_weights = DraftWeights.load(manifest, device, shard=shard, config=config.draft,
                    target_embedding=embedding, target_head=head, progress=progress)
            if vision:
                from .vision import VisionWeights

                vision_weights = VisionWeights.load(manifest, device, shard=shard, progress=progress)
        return cls(config, shard, embedding, norm, head, layers,
                   torch.cuda.memory_allocated(device) - before, draft_weights, vision_weights)


class MiniMaxWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, expert_scale_compression=None):
        super().__init__(manifest, device, prefixes=("language_model.",), progress=progress)
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        self.config.validate_shard(shard)
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("MiniMax M3 expert scale compression must be None or 'csf'")
        self.expert_scale_compression = expert_scale_compression
        self.expert_scale_scratch = {}

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name in (TEXT_PREFIX + ".embed_tokens.weight", TEXT_PREFIX + ".norm.weight",
                            "language_model.lm_head.weight") or name.startswith(TEXT_PREFIX + ".layers.")}

    @cached_property
    def quantization(self):
        return {name.removesuffix(".weight"): {"quant_algo": "MXFP8", "group_size": 32}
                for name in self._sources if name.endswith(".weight")
                and self.config.projection_format(name.removesuffix(".weight")) == "mxfp8"}

    def norm(self, name, width):
        value = self._sources[name]
        if value.shape != (width,) or value.dtype != torch.bfloat16:
            raise ValueError(f"{name}: expected a {width}-element BF16 normalization parameter")
        return self.load_tensor(name)

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None):
        prefixes = tuple(prefixes)
        formats = {self.config.projection_format(prefix) for prefix in prefixes}
        if len(formats) != 1 or "nvfp4" in formats:
            raise ValueError("MiniMax M3 projection fusion requires compatible BF16 or MXFP8 storage")
        if formats == {"mxfp8"}:
            return super().load_linear_group(prefixes, row_ranges=row_ranges, column_range=column_range,
                                            scale_name="weight_scale_inv")
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        values = []
        for prefix, rows in zip(prefixes, ranges, strict=True):
            source = self._sources[prefix + ".weight"]
            if source.ndim != 2 or source.dtype != torch.bfloat16:
                raise ValueError(f"{prefix}: checkpoint metadata requires BF16 projection storage")
            values.append(source[slice(*(rows or (0, source.shape[0]))),
                                 slice(*(column_range or (0, source.shape[1])))])
        if any(value.shape[1] != values[0].shape[1] for value in values):
            raise ValueError("fused MiniMax M3 projections must share an input width")
        result = torch.empty((sum(value.shape[0] for value in values), values[0].shape[1]),
                             dtype=torch.bfloat16, device=self.device)
        offset = 0
        for value in values:
            self._copy(result[offset:offset + value.shape[0]], value)
            offset += value.shape[0]
        self._session.flush()
        return result

    def mlp(self, prefix, intermediate):
        interval = self.shard.partition(intermediate, alignment=32)
        return MLPWeights(self.load_linear_group((prefix + ".gate_proj", prefix + ".up_proj"),
                                                row_ranges=(interval, interval)),
                          self.load_linear(prefix + ".down_proj", columns=interval))

    def load_layer(self, layer):
        c, shard = self.config.text, self.shard
        prefix = f"{TEXT_PREFIX}.layers.{layer}"
        attention = prefix + ".self_attn"
        q0, q1 = shard.partition(c["num_attention_heads"])
        k0, k1 = shard.partition(c["num_key_value_heads"])
        dimension = c["head_dim"]
        qr, kr = (q0 * dimension, q1 * dimension), (k0 * dimension, k1 * dimension)
        projections = tuple(attention + "." + name + "_proj" for name in ("q", "k", "v"))
        rows = (qr, kr, kr)
        index_q_norm = index_k_norm = None
        if self.config.sparse(layer):
            projections += (attention + ".index_q_proj", attention + ".index_k_proj")
            rows += (kr, None)
            index_q_norm = self.norm(attention + ".index_q_norm.weight", dimension)
            index_k_norm = self.norm(attention + ".index_k_norm.weight", dimension)
        qkv = self.load_linear_group(projections, row_ranges=rows)
        routed = self.config.routed(layer)
        ffn = prefix + (".block_sparse_moe.shared_experts" if routed else ".mlp")
        mlp = self.mlp(ffn, c["shared_intermediate_size"] if routed else c["dense_intermediate_size"])
        router = correction = experts = None
        if routed:
            block = prefix + ".block_sparse_moe"
            router = self.load_tensor(block + ".gate.weight")
            if router.shape != (c["num_local_experts"], c["hidden_size"]) or router.dtype != torch.float32:
                raise ValueError("MiniMax M3 router weights must retain their FP32 checkpoint precision")
            correction = self.load_tensor(block + ".e_score_correction_bias")
            if correction.shape != (c["num_local_experts"],) or correction.dtype != torch.float32:
                raise ValueError("MiniMax M3 correction bias must be one FP32 value per expert")
            names = tuple(f"{block}.experts.{index}" for index in range(c["num_local_experts"]))
            if any(self.config.projection_format(name + "." + part) != "nvfp4"
                   for name in names for part in ("w1", "w2", "w3")):
                raise ValueError("MiniMax M3 routed experts require declared NVFP4 storage")
            extent = shard.partition(c["intermediate_size"], alignment=32)
            scale_scratch = None
            if self.expert_scale_compression is not None:
                local = extent[1] - extent[0]
                key = len(names), c["hidden_size"], local
                if key not in self.expert_scale_scratch:
                    e, h, n = key
                    self.expert_scale_scratch[key] = tuple(
                        torch.empty(shape, dtype=torch.uint8, device=self.device)
                        for shape in ((e, 2 * n, h // 16), (e, h, n // 16)))
                scale_scratch = self.expert_scale_scratch[key]
            experts = self.load_fp4_experts(names, hidden_size=c["hidden_size"],
                intermediate_size=c["intermediate_size"], intermediate_range=extent,
                format="nvfp4", activation_mode="a4", scale_compression=self.expert_scale_compression,
                scale_scratch=scale_scratch,
                projection_names=("w1", "w3", "w2"), w13_layout="w31", weight_name="weight_packed",
                global_scale_name="weight_global_scale", global_scale_kind="reciprocal",
                input_scale_name="input_global_scale",
                activation="swigluoai_uninterleave", swiglu_alpha=c["swiglu_alpha"], swiglu_beta=1.,
                swiglu_limit=c["swiglu_limit"], activation_outer_scale="token")
        return LayerWeights(qkv, self.load_linear(attention + ".o_proj", columns=qr),
            self.norm(attention + ".q_norm.weight", dimension), self.norm(attention + ".k_norm.weight", dimension),
            index_q_norm, index_k_norm, self.norm(prefix + ".input_layernorm.weight", c["hidden_size"]),
            self.norm(prefix + ".post_attention_layernorm.weight", c["hidden_size"]), mlp, router, correction, experts)
