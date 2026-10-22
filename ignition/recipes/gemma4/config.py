"""Structural checkpoint configuration for the Gemma 4 model family."""

from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re

from ignition.preparation.quantization import quantized_layers

ARCHITECTURES = (
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)
DRAFT_ARCHITECTURES = (
    "Gemma4AssistantForCausalLM",
    "Gemma4UnifiedAssistantForCausalLM",
)
ASSISTANT_REPOSITORIES = {
    5376: ("google/gemma-4-31B-it-assistant", "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"),
    2816: ("google/gemma-4-26B-A4B-it-assistant", "6e5aaaf4c42b98394530b8fda2e95cadd65c151c"),
    3840: ("google/gemma-4-12B-it-assistant", "46d4c6f13f0ac0ad827b915669b8df9b81c64c51"),
    2560: ("google/gemma-4-E4B-it-assistant", "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"),
    1536: ("google/gemma-4-E2B-it-assistant", "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"),
}
DRAFT_DIRECTORY = "assistant"
PAGE_TOKENS = 128


@dataclass(frozen=True)
class DraftConfig:
    architecture: str
    backbone_hidden_size: int
    use_ordered_embeddings: bool
    num_centroids: int
    centroid_intermediate_top_k: int
    text: dict
    sliding_donor_layer: int = 0
    full_donor_layer: int = 0

    @classmethod
    def load(cls, snapshot, target: "ModelConfig") -> "DraftConfig":
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, raw: dict, target: "ModelConfig") -> "DraftConfig":
        architectures = raw.get("architectures")
        if not isinstance(architectures, list) or len(architectures) != 1 or architectures[0] not in DRAFT_ARCHITECTURES:
            raise ValueError(f"Gemma 4 Assistant requires one of {DRAFT_ARCHITECTURES}")
        if raw.get("model_type") not in ("gemma4_assistant", "gemma4_unified_assistant"):
            raise ValueError(f"unsupported Gemma 4 Assistant model_type: {raw.get('model_type')!r}")
        backbone_hidden = raw.get("backbone_hidden_size")
        if backbone_hidden != target.hidden:
            raise ValueError(f"Gemma 4 Assistant backbone_hidden_size {backbone_hidden!r} != {target.hidden}")
        text = raw.get("text_config")
        if not isinstance(text, dict) or text.get("model_type") not in ("gemma4_text", "gemma4_unified_text"):
            raise ValueError("Gemma 4 Assistant requires a gemma4_text text_config")
        if text.get("vocab_size") != target.vocabulary:
            raise ValueError("Gemma 4 Assistant vocab_size must match target vocabulary")
        if text.get("head_dim") != target.text["head_dim"] or text.get("global_head_dim") != target.text["global_head_dim"]:
            raise ValueError("Gemma 4 Assistant head dimensions must match target")
        if text.get("sliding_window") != target.sliding_window:
            raise ValueError("Gemma 4 Assistant sliding_window must match target")
        if text.get("num_key_value_heads") != target.text["num_key_value_heads"]:
            raise ValueError("Gemma 4 Assistant num_key_value_heads must match target")
        if text.get("num_global_key_value_heads") != target.text.get("num_global_key_value_heads"):
            raise ValueError("Gemma 4 Assistant num_global_key_value_heads must match target")
        if bool(text.get("attention_k_eq_v", False)) != target.attention_k_eq_v:
            raise ValueError("Gemma 4 Assistant attention_k_eq_v must match target")
        if text.get("hidden_size_per_layer_input", 0) != 0 or bool(text.get("enable_moe_block", False)):
            raise ValueError("Gemma 4 Assistant cannot enable PLE or MoE blocks")
        layers = text.get("layer_types")
        if (
            not isinstance(layers, list)
            or len(layers) != text.get("num_hidden_layers")
            or not layers
            or any(kind not in ("sliding_attention", "full_attention") for kind in layers)
        ):
            raise ValueError("Gemma 4 Assistant layer_types must describe every layer")
        ordered = bool(raw.get("use_ordered_embeddings", False))
        centroids = int(raw.get("num_centroids", 2048))
        top_k = int(raw.get("centroid_intermediate_top_k", 32))
        if ordered and (centroids <= 0 or target.vocabulary % centroids != 0 or not 0 < top_k <= centroids):
            raise ValueError("Gemma 4 Assistant ordered embeddings require valid centroid geometry")
        return cls(
            architectures[0],
            backbone_hidden,
            ordered,
            centroids,
            top_k,
            text,
            sliding_donor_layer=target.sliding_donor_layer,
            full_donor_layer=target.full_donor_layer,
        )

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.text["layer_types"])

    @property
    def num_layers(self) -> int:
        return int(self.text["num_hidden_layers"])

    @property
    def hidden(self) -> int:
        return int(self.text["hidden_size"])

    @property
    def intermediate_size(self) -> int:
        return int(self.text["intermediate_size"])

    @property
    def vocabulary(self) -> int:
        return int(self.text["vocab_size"])

    @property
    def cluster_size(self) -> int:
        return self.vocabulary // self.num_centroids

    @property
    def heads(self) -> int:
        return int(self.text["num_attention_heads"])

    @property
    def rms_norm_eps(self) -> float:
        return float(self.text["rms_norm_eps"])

    @property
    def sliding_window(self) -> int:
        return int(self.text["sliding_window"])

    @property
    def final_logit_softcapping(self) -> float | None:
        value = self.text.get("final_logit_softcapping")
        return None if value is None else float(value)

    @property
    def enable_moe(self) -> bool:
        return False

    @property
    def ple_dim(self) -> int:
        return 0

    @property
    def attention_k_eq_v(self) -> bool:
        return bool(self.text.get("attention_k_eq_v", False))

    def is_sliding(self, layer: int) -> bool:
        return self.layers[layer] == "sliding_attention"

    def is_kv_shared(self, layer: int) -> bool:
        return True

    def kv_source_layer(self, layer: int) -> int:
        return self.sliding_donor_layer if self.is_sliding(layer) else self.full_donor_layer

    def layer_head_dim(self, layer: int) -> int:
        return self.text["head_dim"] if self.is_sliding(layer) else self.text["global_head_dim"]

    def layer_kv_heads(self, layer: int) -> int:
        if not self.is_sliding(layer) and self.attention_k_eq_v:
            override = self.text.get("num_global_key_value_heads")
            if override is not None:
                return override
        return self.text["num_key_value_heads"]

    def local_heads(self, shard) -> int:
        return self.heads // shard.world_size

    def local_kv_heads(self, layer: int, shard) -> int:
        first, last = shard.replicated_partition(self.layer_kv_heads(layer))
        return last - first

    def q_range(self, layer: int, shard) -> tuple[int, int]:
        dim = self.layer_head_dim(layer)
        return shard.partition(self.heads * dim, alignment=dim)

    def layer_intermediate_size(self, layer: int) -> int:
        return self.intermediate_size

    def intermediate_range(self, layer: int, shard) -> tuple[int, int]:
        return shard.partition(self.intermediate_size, alignment=16)

    def rotary_inverse(self, layer_type: str, device):
        import torch

        dim = self.text["head_dim"] if layer_type == "sliding_attention" else self.text["global_head_dim"]
        spec = self.text["rope_parameters"][layer_type]
        theta = float(spec["rope_theta"])
        if spec["rope_type"] == "proportional":
            proportion = float(spec.get("partial_rotary_factor", 1.0))
            factor = float(spec.get("factor", 1.0))
            rope_angles = int(proportion * dim // 2)
            rotated = 1.0 / (
                theta ** (torch.arange(0, 2 * rope_angles, 2, device=device, dtype=torch.float32) / dim)
            )
            nope = torch.zeros(dim // 2 - rope_angles, device=device, dtype=torch.float32)
            return torch.cat((rotated, nope), dim=0) / factor
        return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    text_prefix: str
    text: dict
    quantization: dict
    draft: DraftConfig | None = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw: dict):
        architectures = raw.get("architectures")
        if not isinstance(architectures, list) or len(architectures) != 1 or architectures[0] not in ARCHITECTURES:
            raise ValueError(f"Gemma 4 preparation requires one of {ARCHITECTURES}")
        architecture = architectures[0]
        if architecture == "Gemma4UnifiedForConditionalGeneration":
            text = raw.get("text_config")
            text_prefix = "model.language_model"
            expected_model_types = ("gemma4_unified_text",)
        else:
            text = raw.get("text_config")
            text_prefix = "model.language_model"
            expected_model_types = ("gemma4_text",)
        if not isinstance(text, dict) or text.get("model_type") not in expected_model_types:
            raise ValueError(
                f"unsupported Gemma 4 text model_type: {None if not isinstance(text, dict) else text.get('model_type')!r}"
            )
        required = {
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "head_dim": 256,
            "global_head_dim": 512,
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"unsupported Gemma 4 {name}: {text.get(name)!r}")
        for name in (
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "sliding_window",
            "max_position_embeddings",
        ):
            if type(text.get(name)) is not int or text[name] <= 0:
                raise ValueError(f"Gemma 4 requires a positive integer {name}")
        layers = text.get("layer_types", ())
        if (
            not isinstance(layers, list)
            or len(layers) != text["num_hidden_layers"]
            or any(kind not in ("sliding_attention", "full_attention") for kind in layers)
            or layers[-1] != "full_attention"
        ):
            raise ValueError("Gemma 4 layer_types must describe every layer and end with full_attention")
        if text["num_attention_heads"] % text["num_key_value_heads"]:
            raise ValueError("Gemma 4 attention heads must be divisible by key/value heads")
        global_kv_heads = text.get("num_global_key_value_heads")
        if global_kv_heads is not None and (
            type(global_kv_heads) is not int
            or global_kv_heads <= 0
            or text["num_attention_heads"] % global_kv_heads
        ):
            raise ValueError("Gemma 4 global key/value heads must divide attention heads")
        shared_layers = text.get("num_kv_shared_layers", 0)
        if type(shared_layers) is not int or not 0 <= shared_layers < text["num_hidden_layers"]:
            raise ValueError("Gemma 4 num_kv_shared_layers must be in [0, num_hidden_layers)")
        if shared_layers > 0:
            non_shared = layers[: text["num_hidden_layers"] - shared_layers]
            for kind in set(layers[text["num_hidden_layers"] - shared_layers :]):
                if kind not in non_shared:
                    raise ValueError(f"Gemma 4 KV-shared layer type {kind!r} has no non-shared donor layer")
        ple_dim = text.get("hidden_size_per_layer_input", 0)
        if type(ple_dim) is not int or ple_dim < 0:
            raise ValueError("Gemma 4 hidden_size_per_layer_input must be a nonnegative integer")
        if ple_dim > 0:
            ple_vocab = text.get("vocab_size_per_layer_input")
            if type(ple_vocab) is not int or not 0 < ple_vocab <= text["vocab_size"]:
                raise ValueError("Gemma 4 vocab_size_per_layer_input must be in (0, vocab_size]")
        if bool(text.get("enable_moe_block", bool(text.get("num_experts")))):
            for name in ("num_experts", "top_k_experts", "moe_intermediate_size"):
                if type(text.get(name)) is not int or text[name] <= 0:
                    raise ValueError(f"Gemma 4 MoE requires a positive integer {name}")
            if text["top_k_experts"] > text["num_experts"]:
                raise ValueError("Gemma 4 top_k_experts cannot exceed num_experts")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("Gemma 4 requires rope_parameters for sliding_attention and full_attention")
        for kind in ("sliding_attention", "full_attention"):
            spec = rope.get(kind)
            if not isinstance(spec, dict) or spec.get("rope_type") not in ("default", "proportional"):
                raise ValueError(f"unsupported Gemma 4 RoPE configuration for {kind}: {spec!r}")
            if not 0 < float(spec.get("rope_theta", 0)):
                raise ValueError(f"Gemma 4 rope_theta for {kind} must be positive")
            proportion = float(spec.get("partial_rotary_factor", 1.0))
            if not 0 < proportion <= 1.0:
                raise ValueError(f"Gemma 4 partial_rotary_factor for {kind} must be in (0, 1]")
        if not 0 < float(text.get("rms_norm_eps", 0)):
            raise ValueError("Gemma 4 rms_norm_eps must be positive")
        softcap = text.get("final_logit_softcapping")
        if softcap is not None and not 0 < float(softcap):
            raise ValueError("Gemma 4 final_logit_softcapping must be positive when set")
        provisional = cls(architecture, text_prefix, text, {})
        return cls(architecture, text_prefix, text, _parse_quantization(raw, provisional))

    def validate_shard(self, shard) -> None:
        if shard.world_size not in (1, 2, 4):
            raise ValueError("Gemma 4 supports tensor-parallel world_size in (1, 2, 4)")
        if self.heads % shard.world_size != 0:
            raise ValueError("Gemma 4 attention heads must be divisible by world_size")
        if self.vocabulary % (shard.world_size * 64) != 0:
            raise ValueError("Gemma 4 vocabulary must partition into 64-aligned rank shards")
        for index in range(self.num_layers):
            if self.layer_intermediate_size(index) % (shard.world_size * 16) != 0:
                raise ValueError("Gemma 4 MLP intermediate_size must divide into 16-aligned rank shards")
        if self.enable_moe and self.moe_intermediate_size % (shard.world_size * 16) != 0:
            raise ValueError("Gemma 4 MoE intermediate_size must divide into 16-aligned rank shards")
        if self.draft is not None:
            if self.draft.heads % shard.world_size != 0:
                raise ValueError("Gemma 4 Assistant attention heads must be divisible by world_size")
            if self.draft.intermediate_size % (16 * shard.world_size) != 0:
                raise ValueError("Gemma 4 Assistant MLP intermediate_size must divide into 16-aligned rank shards")

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.text["layer_types"])

    @property
    def num_layers(self) -> int:
        return self.text["num_hidden_layers"]

    @property
    def hidden(self) -> int:
        return self.text["hidden_size"]

    @property
    def vocabulary(self) -> int:
        return self.text["vocab_size"]

    @property
    def heads(self) -> int:
        return self.text["num_attention_heads"]

    @property
    def rms_norm_eps(self) -> float:
        return float(self.text["rms_norm_eps"])

    @property
    def sliding_window(self) -> int:
        return self.text["sliding_window"]

    @property
    def final_logit_softcapping(self) -> float | None:
        value = self.text.get("final_logit_softcapping")
        return None if value is None else float(value)

    @property
    def ple_dim(self) -> int:
        return int(self.text.get("hidden_size_per_layer_input", 0))

    @property
    def ple_vocabulary(self) -> int:
        return int(self.text.get("vocab_size_per_layer_input", self.vocabulary))

    @property
    def enable_moe(self) -> bool:
        return bool(self.text.get("enable_moe_block", bool(self.text.get("num_experts"))))

    @property
    def attention_k_eq_v(self) -> bool:
        return bool(self.text.get("attention_k_eq_v", False))

    @property
    def num_experts(self) -> int:
        return int(self.text.get("num_experts", 0))

    @property
    def top_k_experts(self) -> int:
        return int(self.text.get("top_k_experts", 0))

    @property
    def moe_intermediate_size(self) -> int:
        return int(self.text.get("moe_intermediate_size", 0))

    @property
    def first_kv_shared_layer(self) -> int:
        return self.num_layers - int(self.text.get("num_kv_shared_layers", 0))

    @property
    def sliding_donor_layer(self) -> int:
        non_shared = self.layers[: self.first_kv_shared_layer]
        return len(non_shared) - 1 - non_shared[::-1].index("sliding_attention")

    @property
    def full_donor_layer(self) -> int:
        non_shared = self.layers[: self.first_kv_shared_layer]
        return len(non_shared) - 1 - non_shared[::-1].index("full_attention")

    def is_sliding(self, layer: int) -> bool:
        return self.layers[layer] == "sliding_attention"

    def is_kv_shared(self, layer: int) -> bool:
        return layer >= self.first_kv_shared_layer

    def kv_source_layer(self, layer: int) -> int:
        if not self.is_kv_shared(layer):
            return layer
        return self.sliding_donor_layer if self.is_sliding(layer) else self.full_donor_layer

    def layer_head_dim(self, layer: int) -> int:
        return self.text["head_dim"] if self.is_sliding(layer) else self.text["global_head_dim"]

    def layer_kv_heads(self, layer: int) -> int:
        if not self.is_sliding(layer) and self.attention_k_eq_v:
            override = self.text.get("num_global_key_value_heads")
            if override is not None:
                return override
        return self.text["num_key_value_heads"]

    def local_heads(self, shard) -> int:
        return self.heads // shard.world_size

    def local_kv_heads(self, layer: int, shard) -> int:
        first, last = shard.replicated_partition(self.layer_kv_heads(layer))
        return last - first

    def q_range(self, layer: int, shard) -> tuple[int, int]:
        dim = self.layer_head_dim(layer)
        return shard.partition(self.heads * dim, alignment=dim)

    def kv_range(self, layer: int, shard) -> tuple[int, int]:
        dim = self.layer_head_dim(layer)
        first, last = shard.replicated_partition(self.layer_kv_heads(layer))
        return first * dim, last * dim

    def intermediate_range(self, layer: int, shard) -> tuple[int, int]:
        return shard.partition(self.layer_intermediate_size(layer), alignment=16)

    def moe_intermediate_range(self, shard) -> tuple[int, int]:
        return shard.partition(self.moe_intermediate_size, alignment=16)

    def vocabulary_partition(self, shard) -> tuple[int, int, int]:
        start, end = shard.partition(self.vocabulary, alignment=64)
        return start, end, end - start

    def layer_has_v_proj(self, layer: int) -> bool:
        if self.is_kv_shared(layer):
            return False
        return not (self.attention_k_eq_v and not self.is_sliding(layer))

    def layer_intermediate_size(self, layer: int) -> int:
        double_wide = bool(self.text.get("use_double_wide_mlp", False)) and self.is_kv_shared(layer)
        return self.text["intermediate_size"] * (2 if double_wide else 1)

    def rotary_inverse(self, layer_type: str, device):
        import torch

        dim = self.text["head_dim"] if layer_type == "sliding_attention" else self.text["global_head_dim"]
        spec = self.text["rope_parameters"][layer_type]
        theta = float(spec["rope_theta"])
        if spec["rope_type"] == "proportional":
            proportion = float(spec.get("partial_rotary_factor", 1.0))
            factor = float(spec.get("factor", 1.0))
            rope_angles = int(proportion * dim // 2)
            rotated = 1.0 / (
                theta ** (torch.arange(0, 2 * rope_angles, 2, device=device, dtype=torch.float32) / dim)
            )
            nope = torch.zeros(dim // 2 - rope_angles, device=device, dtype=torch.float32)
            return torch.cat((rotated, nope), dim=0) / factor
        return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


def _candidate_linear_names(config: ModelConfig) -> list[str]:
    names = ["lm_head"]
    prefix = config.text_prefix
    if config.ple_dim > 0:
        names.extend((
            f"{prefix}.embed_tokens_per_layer",
            f"{prefix}.per_layer_model_projection",
        ))
    for index in range(config.num_layers):
        layer_prefix = f"{prefix}.layers.{index}"
        names.extend((
            f"{layer_prefix}.self_attn.q_proj",
            f"{layer_prefix}.self_attn.o_proj",
            f"{layer_prefix}.mlp.gate_proj",
            f"{layer_prefix}.mlp.up_proj",
            f"{layer_prefix}.mlp.down_proj",
        ))
        if not config.is_kv_shared(index):
            names.append(f"{layer_prefix}.self_attn.k_proj")
            if config.layer_has_v_proj(index):
                names.append(f"{layer_prefix}.self_attn.v_proj")
        if config.ple_dim > 0:
            names.extend((
                f"{layer_prefix}.per_layer_input_gate",
                f"{layer_prefix}.per_layer_projection",
            ))
        if config.enable_moe:
            names.append(f"{layer_prefix}.router.proj")
            for expert in range(config.num_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    names.append(f"{layer_prefix}.experts.{expert}.{proj}")
    return names


def _matches_pattern(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return bool(re.search(pattern[3:], name))
    return fnmatch.fnmatchcase(name, pattern)


def _parse_quantization(raw: dict, config: ModelConfig) -> dict:
    quant = raw.get("quantization_config")
    if quant is None:
        return {}
    if isinstance(quant, dict) and "quantized_layers" in quant:
        return dict(quantized_layers(raw))
    if not isinstance(quant, dict):
        raise ValueError(f"unsupported Gemma 4 quantization_config: {quant!r}")
    method = quant.get("quant_method")
    if method == "modelopt" and quant.get("quant_algo") == "NVFP4":
        groups = quant.get("config_groups")
        group_0 = groups.get("group_0") if isinstance(groups, dict) else None
        weights = group_0.get("weights") if isinstance(group_0, dict) else None
        if not isinstance(weights, dict) or weights.get("num_bits") != 4 or weights.get("group_size") != 16:
            raise ValueError("Gemma 4 NVFP4 quantization requires 4-bit group_size=16 weights")
        ignore = quant.get("ignore", ())
        if not isinstance(ignore, list) or any(not isinstance(pattern, str) for pattern in ignore):
            raise ValueError("Gemma 4 NVFP4 ignore list must contain glob strings")
        entry = {"quant_algo": "NVFP4", "group_size": 16}
        return {
            name: entry
            for name in _candidate_linear_names(config)
            if not any(_matches_pattern(name, pattern) for pattern in ignore)
        }
    if method == "compressed-tensors":
        groups = quant.get("config_groups")
        ignore = quant.get("ignore", ())
        if not isinstance(groups, dict) or not isinstance(ignore, list):
            raise ValueError("Gemma 4 compressed-tensors requires config_groups and ignore")
        group_specs = []
        for group in groups.values():
            if not isinstance(group, dict):
                raise ValueError("invalid compressed-tensors group")
            fmt = group.get("format")
            weights = group.get("weights") if isinstance(group.get("weights"), dict) else {}
            targets = group.get("targets", ())
            if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets):
                raise ValueError("compressed-tensors targets must be strings")
            if fmt == "float-quantized" and weights.get("num_bits") == 8 and weights.get("strategy") == "channel":
                spec = {"quant_algo": "FP8_CHANNEL", "group_size": 0}
            elif (
                fmt == "pack-quantized"
                and weights.get("num_bits") == 8
                and weights.get("strategy") == "channel"
                and weights.get("type") == "int"
            ):
                spec = {"quant_algo": "INT8_PACK_CHANNEL", "group_size": 0}
            elif fmt == "nvfp4-pack-quantized" and weights.get("num_bits") == 4 and weights.get("group_size") == 16:
                spec = {"quant_algo": "COMPRESSED_NVFP4", "group_size": 16}
            else:
                raise ValueError(f"unsupported compressed-tensors group format: {fmt!r}")
            group_specs.append((tuple(targets), spec))
        result = {}
        for name in _candidate_linear_names(config):
            if any(_matches_pattern(name, pattern) for pattern in ignore):
                continue
            for targets, spec in group_specs:
                if any(_matches_pattern(name, pattern) for pattern in targets):
                    result[name] = spec
                    break
        return result
    raise ValueError(f"unsupported Gemma 4 quantization_config: {quant!r}")
