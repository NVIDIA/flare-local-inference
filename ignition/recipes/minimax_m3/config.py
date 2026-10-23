"""Checkpoint-declared MiniMax M3 geometry and projection storage."""

import json
import math
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

PAGE_TOKENS = 128
TEXT_PREFIX = "language_model.model"


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    vision: dict
    target_layers: tuple[int, ...] = ()
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        snapshot = Path(snapshot)
        config = json.loads((snapshot / "config.json").read_text())
        draft = snapshot / "dspark" / "config.json"
        taps = ()
        if draft.is_file():
            declaration = json.loads(draft.read_text())
            text = declaration.get("text_config", declaration)
            taps = tuple(text.get("dspark_target_layer_ids", ()))
        return cls.from_dict(config, target_layers=taps)

    @classmethod
    def from_dict(cls, config, *, target_layers=()):
        if (config.get("model_type") != "minimax_m3_vl"
                or config.get("architectures") != ["MiniMaxM3SparseForConditionalGeneration"]):
            raise ValueError("MiniMax M3 preparation requires MiniMaxM3SparseForConditionalGeneration")
        c, q = dict(config["text_config"]), dict(config["quantization_config"])
        required = {"use_gemma_norm": True, "use_qk_norm": True, "qk_norm_type": "per_head",
            "attention_output_gate": False, "use_routing_bias": True, "scoring_func": "sigmoid",
            "hidden_act": "swigluoai", "tie_word_embeddings": False}
        if any(c.get(name) != value for name, value in required.items()):
            raise ValueError("unsupported MiniMax M3 normalization, routing, or activation contract")
        for name in ("hidden_size", "intermediate_size", "dense_intermediate_size", "shared_intermediate_size",
                     "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                     "vocab_size", "num_local_experts", "num_experts_per_tok", "max_position_embeddings"):
            if type(c.get(name)) is not int or c[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (c["head_dim"] != 128 or c["rotary_dim"] != 64
                or c["num_attention_heads"] != 16 * c["num_key_value_heads"]
                or c.get("n_shared_experts") != 1 or c["num_experts_per_tok"] > c["num_local_experts"]):
            raise ValueError("MiniMax M3 requires 128-channel GQA heads, rotary width 64, and one shared expert")
        rope = c.get("rope_parameters") or {}
        fraction = rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.))
        if fraction != c["rotary_dim"] / c["head_dim"] or rope.get("rope_type", "default") != "default":
            raise ValueError("MiniMax M3 requires canonical default partial rotary frequencies")
        for name in ("rms_norm_eps", "rope_theta", "swiglu_alpha", "swiglu_limit", "routed_scaling_factor"):
            if not isinstance(c.get(name), (int, float)) or not math.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"{name} must be positive and finite")
        sparse = c["sparse_attention_config"]
        expected = {"use_sparse_attention": True, "sparse_index_dim": 128,
            "sparse_num_index_heads": c["num_key_value_heads"], "sparse_topk_blocks": 16,
            "sparse_block_size": PAGE_TOKENS, "sparse_score_type": "max",
            "sparse_init_block": 0, "sparse_local_block": 1}
        if any(sparse.get(name) != value for name, value in expected.items()):
            raise ValueError("unsupported MiniMax M3 block-sparse indexer geometry")
        for name, flags in (("moe_layer_freq", c.get("moe_layer_freq")),
                            ("sparse_attention_freq", sparse.get("sparse_attention_freq")),
                            ("sparse_disable_index_value", sparse.get("sparse_disable_index_value"))):
            if (not isinstance(flags, list) or len(flags) != c["num_hidden_layers"]
                    or any(type(flag) is not int or flag not in (0, 1) for flag in flags)):
                raise ValueError(f"{name} must declare one binary flag per target layer")
        if any(not flag for flag in sparse["sparse_disable_index_value"]):
            raise ValueError("MiniMax M3 index-value projections are not supported")
        if not all(sparse["sparse_attention_freq"]):
            raise ValueError("MiniMax M3 preparation requires block-sparse attention on every target layer")
        if (q.get("quant_method") != "mxfp8" or q.get("weight_block_size") != [1, 32]
                or q.get("activation_scheme") != "dynamic" or q.get("group_size") != 16
                or q.get("moe_quant_algo") != "NVFP4"
                or q.get("moe_quant_format") != "nvfp4-pack-quantized"
                or not isinstance(q.get("ignored_layers"), list)):
            raise ValueError("MiniMax M3 requires declared MXFP8 projections and packed NVFP4 experts")
        taps = tuple(target_layers)
        if (tuple(sorted(set(taps))) != taps
                or any(type(index) is not int or not 0 <= index < c["num_hidden_layers"] for index in taps)):
            raise ValueError("draft target taps must be distinct, ordered target layer indices")
        return cls(c, q, dict(config.get("vision_config", {})), taps)

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    def sparse(self, layer):
        return bool(self.text["sparse_attention_config"]["sparse_attention_freq"][layer])

    def routed(self, layer):
        return bool(self.text["moe_layer_freq"][layer])

    def projection_format(self, prefix):
        names = (prefix, prefix.removeprefix("language_model."))
        if any(fnmatchcase(name, pattern) for name in names for pattern in self.quantization["ignored_layers"]):
            return "bf16"
        return "nvfp4" if ".experts." in prefix else "mxfp8"

    def validate_shard(self, shard):
        if shard.world_size != 4:
            raise ValueError("MiniMax M3 preparation requires tensor parallelism of four")
        c = self.text
        for extent, alignment in ((self.hidden, 64), (c["num_attention_heads"], 1),
                                  (c["num_key_value_heads"], 1), (c["dense_intermediate_size"], 32),
                                  (c["shared_intermediate_size"], 32), (c["intermediate_size"], 32)):
            shard.partition(extent, alignment=alignment)

    def vocabulary_partition(self, shard):
        width = -(-self.vocabulary // (shard.world_size * 64)) * 64
        return shard.rank * width, min((shard.rank + 1) * width, self.vocabulary), width

    def canonical_config(self):
        from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

        return MiniMaxM3VLTextConfig.from_dict(self.text)

    def rotary_frequencies(self, device=None):
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding

        canonical = self.canonical_config()
        if canonical.rope_parameters["rope_type"] != "default":
            raise ValueError("MiniMax M3 requires default rotary frequencies")
        inverse = MiniMaxM3VLRotaryEmbedding.compute_default_rope_parameters(canonical, device=device)[0]
        if inverse.numel() * 2 != self.text["rotary_dim"]:
            raise ValueError("canonical rotary frequencies disagree with the checkpoint's rotary dimension")
        return inverse
