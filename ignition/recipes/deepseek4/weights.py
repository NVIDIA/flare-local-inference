"""Block-128 FP8, native MXFP4 experts, vision, and checkpoint-integrated DSpark."""

from dataclasses import dataclass, replace

import torch

from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.components.weights import MLPWeights

from .config import ModelConfig


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    vision_correction: torch.Tensor
    token_experts: torch.Tensor | None
    experts: object
    shared: MLPWeights


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    experts: ExpertWeights
    residual: dict


@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    norm: torch.Tensor
    head_mix: dict
    markov_embedding: torch.Tensor
    markov_head: object
    draft_head: object | None = None

    @classmethod
    def load(cls, loader):
        c, shard = loader.config.draft, loader.shard
        last = f"mtp.{c.layers - 1}"
        return cls(
            tuple(loader.load_layer(i, draft=True) for i in range(c.layers)),
            loader.load_linear("mtp.0.main_proj"),
            loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(c.layers))),
            loader.load_tensor("mtp.0.main_norm.weight"),
            loader.load_tensor(last + ".norm.weight"),
            loader.load_head_mix(last + "."),
            loader.load_tensor(last + ".markov_head.markov_w1.weight"),
            loader.load_tensor(
                last + ".markov_head.markov_w2.weight", extent=slice(*shard.partition(129280, alignment=64))
            ),
        )


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    head_mix: dict
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    vision: dict
    allocated_bytes: int

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, dspark=True,
             draft_head_quantization=None, markov_head_quantization=None):
        from ignition.components.vocabulary import nvfp4_draft_head
        config = ModelConfig.load(manifest.metadata_path)
        before = torch.cuda.memory_allocated(device)
        with DeepSeekWeightLoader(manifest, device, shard=shard, progress=progress, dspark=dspark) as loader:
            embedding, norm = loader.load_tensors(("embed.weight", "norm.weight"))
            head = loader.load_tensor("head.weight", extent=slice(*shard.partition(129280, alignment=64)))
            head_mix = loader.load_head_mix("")
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = DSparkWeights.load(loader) if dspark else None
            if draft is not None:
                if draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
            vision = {
                name: loader.load_tensor(name)
                for name in loader._sources
                if name.startswith(("vision.", "aligner.", "image_"))
            }
        return cls(
            config,
            shard,
            embedding,
            norm,
            head,
            head_mix,
            layers,
            draft,
            vision,
            torch.cuda.memory_allocated(device) - before,
        )


class DeepSeekWeightLoader(BlockFP8WeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, dspark=True):
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config, self.shard, self.dspark = ModelConfig.load(manifest.metadata_path), shard, dspark
        # Routed layers run serially; their overlapping dense shared experts use different scales.
        self._expert_scale_scratch = {}

    def _selected_names(self):
        return {name for name in super()._selected_names() if self.dspark or not name.startswith("mtp.")}

    def load_head_mix(self, prefix):
        return {name: self.load_tensor(prefix + "hc_head_" + name) for name in ("fn", "base", "scale")}

    def load_compressor(self, prefix):
        return {
            key: self.load_tensor(prefix + "." + suffix)
            for key, suffix in (
                ("values", "wkv.weight"),
                ("gates", "wgate.weight"),
                ("norm", "norm.weight"),
                ("ape", "ape"),
            )
        }

    def load_attention(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attn = prefix + ".attn"
        heads = shard.partition(64 * 512, alignment=512)
        groups = shard.partition(8 * 1024, alignment=1024)
        attention = {
            "qkv_a": self.load_linear_group((attn + ".wq_a", attn + ".wkv")),
            "q_b": self.load_linear(attn + ".wq_b", rows=heads),
            "output": self.load_attention_output(attn, groups),
            "q_norm": self.load_tensor(attn + ".q_norm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_norm.weight"),
            "sink": self.load_tensor(attn + ".attn_sink", extent=slice(*shard.partition(64))),
        }
        if not draft:
            ratio = text["compress_ratios"][layer]
            if ratio:
                attention["compressor"] = self.load_compressor(attn + ".compressor")
            if ratio == 4:
                attention["index_compressor"] = self.load_compressor(attn + ".indexer.compressor")
                # Replicated index heads avoid reducing a context-sized score matrix.
                attention["index_q"] = self.load_linear(attn + ".indexer.wq_b")
                attention["index_weights"] = self.load_tensor(attn + ".indexer.weights_proj.weight")
            if ratio:
                # Compressor value and gate projections share the attention input: pack them into one
                # BF16 matrix, main compressor first, so one GEMV produces every FP32 output.
                compressors = [attention[k] for k in ("compressor", "index_compressor") if k in attention]
                parts = [part for c in compressors for part in (c.pop("values"), c.pop("gates"))]
                if (parts[0].dtype not in (torch.bfloat16, torch.float32)
                        or any(part.dtype != parts[0].dtype or part.shape[1:] != parts[0].shape[1:]
                               for part in parts)):
                    raise ValueError("compressor projections require matching unquantized weight formats")
                attention["compression"] = torch.cat(parts)
        return attention

    def load_layer(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attention = self.load_attention(layer, draft=draft)
        ffn = prefix + ".ffn"
        interval = shard.partition(text["moe_intermediate_size"], alignment=128)
        experts = self.load_mxfp4_experts(
            [f"{ffn}.experts.{i}" for i in range(text["n_routed_experts"])],
            hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"],
            scale_scratch=self._expert_scale_scratch,
            intermediate_range=interval,
            swiglu_limit=text["swiglu_limit"],
            projection_names=("w3", "w1", "w2"),
            scale_name="scale",
        )
        shared = MLPWeights(
            self.load_linear_group(
                (ffn + ".shared_experts.w1", ffn + ".shared_experts.w3"), row_ranges=(interval, interval)
            ),
            self.load_linear(ffn + ".shared_experts.w2", columns=interval),
        )
        routed = ExpertWeights(
            self.load_tensor(ffn + ".gate.weight"),
            self.load_tensor(ffn + ".gate.bias"),
            self.load_tensor(ffn + ".gate.bias_vl"),
            self.load_tensor(ffn + ".gate.tid2eid", dtype=torch.int32)
            if not draft and layer < text["num_hash_layers"]
            else None,
            experts,
            shared,
        )
        residual = {
            key: self.load_tensor(prefix + "." + key)
            for key in (
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
                "attn_norm.weight",
                "ffn_norm.weight",
            )
        }
        return LayerWeights(attention, routed, residual)
