"""Native FP8/MXFP4 checkpoint transfers and tensor-parallel partitions."""

from dataclasses import dataclass, replace

import torch

from ignition.components.weights import MLPWeights
from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.components.vocabulary import nvfp4_draft_head

from .config import ModelConfig


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: object
    shared: MLPWeights
    vision_correction: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    experts: ExpertWeights
    residual: dict
    engram: dict | None


@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    norm: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: object
    confidence_head: torch.Tensor | None = None
    draft_head: object | None = None

    @classmethod
    def load(cls, loader, *, confidence=False):
        config = loader.config.draft
        last = f"mtp.{config.layers - 1}"
        return cls(tuple(loader.load_layer(i, draft=True) for i in range(config.layers)),
            loader.load_linear("mtp.0.main_proj"),
            loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(config.layers))),
            loader.load_tensor("mtp.0.main_norm.weight"),
            loader.load_tensor(last + ".norm.weight"), loader.load_tensor(last + ".markov_head.embed.weight"),
            loader.load_tensor(last + ".markov_head.head.weight"),
            loader.load_tensor(last + ".confidence_head.proj.weight") if confidence else None)


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    allocated_bytes: int
    vision: dict | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, dspark=False, confidence=False,
             expert_scale_compression=None, draft_head_quantization=None, markov_head_quantization=None):
        config = ModelConfig.load(manifest.metadata_path)
        before = torch.cuda.memory_allocated(device)
        with DeepSeekWeightLoader(manifest, device, shard=shard, progress=progress, dspark=dspark,
                                 expert_scale_compression=expert_scale_compression) as loader:
            embedding, norm = loader.load_tensors(("embed.weight", "norm.weight"))
            head = loader.load_padded_tensor("head.weight", rows=shard.padded_partition(129280, alignment=64))
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = DSparkWeights.load(loader, confidence=confidence) if dspark else None
            if draft is not None:
                if draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
            vision = ({name: loader.load_tensor(name) for name in loader._sources
                       if name.startswith(("vision.", "aligner.", "image_"))}
                      if config.vision is not None else None)
        return cls(config, shard, embedding, norm, head, layers, draft,
                   torch.cuda.memory_allocated(device) - before, vision)


class DeepSeekWeightLoader(BlockFP8WeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, dspark=False,
                 expert_scale_compression=None):
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("DeepSeek V4.1 expert scale compression must be None or 'csf'")
        self.dspark = dspark
        self.expert_scale_compression = expert_scale_compression
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.metadata_path)
        self.shard = shard
        self._expert_scale_scratch = {}

    def load_experts(self, prefixes, interval):
        text = self.config.text
        return self.load_mxfp4_experts(prefixes, hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"], intermediate_range=interval,
            swiglu_limit=text["swiglu_limit"], scale_name="scale", projection_names=("w3", "w1", "w2"),
            scale_compression=self.expert_scale_compression, scale_scratch=self._expert_scale_scratch)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if (name.startswith("layers.") or (self.dspark and name.startswith("mtp."))
                    or name.startswith(("vision.", "aligner.", "image_"))
                    or name in ("embed.weight", "norm.weight", "head.weight"))
                and ".engram.embed." not in name}

    def load_layer(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attn = prefix + ".attn"
        first, last = shard.padded_partition(8)
        heads = first * 8 * 512, last * 8 * 512
        groups = first * 1024, last * 1024
        result = {
            "qkv_a": self.load_linear_group((attn + ".wq_a", attn + ".wkv")),
            "q_b": self.load_linear_group((attn + ".wq_b",), row_ranges=(heads,), pad=True),
            "output": self.load_attention_output(attn, groups, pad=True),
            "q_norm": self.load_tensor(attn + ".q_norm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_norm.weight"),
            "sink": self.load_padded_tensor(attn + ".attn_sink", rows=(first * 8, last * 8)),
        }
        if not draft and layer in text["kv_source_layer_ids"]:
            ratio = text["compress_ratios"][layer]
            result["compressor"] = {
                "values": self.load_tensor(attn + ".compressor.wkv.weight"),
                "norm": self.load_tensor(attn + ".compressor.norm.weight", dtype=torch.float32),
            }
            if ratio == 2:
                result["compressor"]["gates"] = self.load_tensor(attn + ".compressor.wgate.weight")
            result["index_k"] = self.load_tensor(attn + ".indexer.wk.weight")
            result["index_norm"] = self.load_tensor(attn + ".indexer.k_norm.weight")
        if not draft and layer in text["index_source_layer_ids"]:
            # Replicated index heads avoid a context-sized TP score reduction.
            result["index_q"] = self.load_linear(attn + ".indexer.wq_b")
            result["index_weights"] = self.load_tensor(attn + ".indexer.weights_proj.weight")
        ffn = prefix + ".ffn"
        interval = shard.partition(text["moe_intermediate_size"], alignment=32)
        experts = self.load_experts(
            [f"{ffn}.experts.{i}" for i in range(self.config.draft.experts if draft else text["n_routed_experts"])],
            interval)
        shared = MLPWeights(self.load_linear_group((ffn + ".shared_experts.w1", ffn + ".shared_experts.w3"),
            row_ranges=(interval, interval)), self.load_linear(ffn + ".shared_experts.w2", columns=interval))
        routed = ExpertWeights(self.load_tensor(ffn + ".gate.weight"),
            self.load_tensor(ffn + ".gate.bias"), experts, shared,
            self.load_tensor(ffn + ".gate.bias_vl") if self.config.vision is not None else None)
        residual = {name: self.load_tensor(prefix + "." + name) for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "attn_norm.weight", "ffn_norm.weight")}
        engram = None
        if not draft and layer in text["engram_layer_ids"]:
            engram = {"projection": self.load_linear_columns(prefix + ".engram.wkv",
                ((head * 256, (head + 1) * 256) for head in range(shard.rank, 24, shard.world_size))),
                "weight": self.load_tensor(prefix + ".engram.q_weight").float()
                          * self.load_tensor(prefix + ".engram.k_weight").float()}
        return LayerWeights(result, routed, residual, engram)
