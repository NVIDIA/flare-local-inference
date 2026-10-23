"""GLM checkpoint weights preserving per-projection precision across TP partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard
from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import IndexedCheckpoint

if TYPE_CHECKING:
    import torch
    from b12x.moe.fused_moe import PreparedExperts

TEXT_PREFIX = "model.language_model"


@dataclass(frozen=True)
class MoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: PreparedExperts
    shared: MLPWeights
    # A16 experts prepared with A4 capability: prefill may bind quantized activations.
    experts_a4_prefill: bool = False


@dataclass(frozen=True)
class LayerWeights:
    attention: dict[str, object]
    mlp: MLPWeights | MoEWeights
    residual: dict[str, torch.Tensor]


@dataclass(frozen=True)
class MTPWeights:
    attention: dict[str, object]
    mlp: MoEWeights
    tensors: dict[str, torch.Tensor]
    # Proposal head; None uses the target vocabulary projection at checkpoint precision.
    draft_head: object | None = None


@dataclass(frozen=True)
class ModelWeights:
    """One text rank's canonical modules and all packed decoder weights.

    Embeddings and normalization are replicated. Vocabulary weights partition
    their output rows. Routed layers either share one scale-expansion workspace
    and execute serially, or hold their CSF block scales expanded once at load.
    Vision weights are replicated when image graphs are prepared.
    """

    tree: torch.nn.Module
    shard: TensorParallelShard
    layers: tuple[LayerWeights, ...]
    expert_scale_scratch: tuple[torch.Tensor, torch.Tensor] | None
    allocated_bytes: int
    head: object
    mtp: MTPWeights | None = None
    vision: object | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, mtp=False, expert_activation="a4",
             expand_expert_scales=False, language_model_only=False):
        import torch

        from .model import create_model_tree

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GLM model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"GLM prepared execution requires {SUPPORTED_ARCHITECTURES}")
        tree = create_model_tree(manifest.snapshot)
        before = torch.cuda.memory_allocated(device)
        layers = []
        with GlmWeightLoader(manifest, device, shard=shard,
                             progress=progress if shard.rank == 0 else None,
                             expert_activation=expert_activation) as loader:
            names = [f"{TEXT_PREFIX}.embed_tokens.weight", f"{TEXT_PREFIX}.norm.weight"]
            for name in names:
                path, attribute = name.rsplit(".", 1)
                setattr(tree.get_submodule(path), attribute,
                        torch.nn.Parameter(loader.load_tensor(name), requires_grad=False))
            head = loader.load_linear("lm_head", rows=shard.partition(tree.lm_head.out_features, alignment=64))
            if isinstance(head, torch.Tensor):
                tree.lm_head.weight = torch.nn.Parameter(head, requires_grad=False)
            tree.lm_head.out_features = head.shape[0] if isinstance(head, torch.Tensor) else head.out_features
            scales = None if expand_expert_scales else loader.allocate_expert_scale_scratch()
            for index in range(len(tree.model.language_model.layers)):
                layers.append(LayerWeights(loader.load_attention(index), loader.load_mlp(index, scale_scratch=scales),
                                           loader.load_residual(index)))
            draft = loader.load_mtp(scales) if mtp else None
            if draft is not None and isinstance(head, torch.Tensor):
                from ignition.components.vocabulary import nvfp4_draft_head

                draft = replace(draft, draft_head=nvfp4_draft_head(head))
        vision = None
        if not language_model_only:
            from .vision import VisionWeights

            with DirectWeightLoader(manifest, device, prefixes=("model.visual.",),
                                    progress=progress if shard.rank == 0 else None,
                                    progress_label="Load vision weights") as loader:
                vision = VisionWeights.load(tree.model.visual, loader)
        torch.cuda.synchronize(device)
        return cls(tree, shard, tuple(layers), scales, torch.cuda.memory_allocated(device) - before, head, draft, vision)


class GlmWeightLoader(DirectWeightLoader):
    """Head and intermediate sharding with replicated latent attention and mHC.

    KDA and MLA output projections produce partial hidden vectors that require
    an all-reduce. Indexer projections, low-rank inputs, normalization and mHC
    remain replicated. Routed experts partition their intermediate dimension,
    preserving all 288 experts on every rank.
    """

    def __init__(self, manifest: IndexedCheckpoint, device, *, shard: TensorParallelShard,
                 prefixes=None, progress=None, expert_activation="a4"):
        super().__init__(manifest, device, prefixes=prefixes,
                         progress=progress)
        self.shard = shard
        if expert_activation not in ("a4", "a16"):
            raise ValueError("GLM NVFP4-CSF experts execute with A4 or A16 activations")
        # A16 keeps calibrated activation scales so prefill can still bind A4 launches.
        self.expert_activation = expert_activation
        config = json.loads((manifest.snapshot / "config.json").read_text())
        if (config.get("model_type") != "glm5_next"
                or config.get("architectures") != ["Glm5NextForConditionalGeneration"]):
            raise ValueError("GLM weights require Glm5NextForConditionalGeneration")
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

        text = Glm5NextConfig.from_dict(config).text_config.to_dict()
        for key, expected in {
            "hidden_size": 4096, "num_hidden_layers": 45, "intermediate_size": 12288,
            "moe_intermediate_size": 2048, "n_routed_experts": 288, "n_shared_experts": 1,
            "first_k_dense_replace": 3, "num_attention_heads": 64, "q_lora_rank": 1536,
            "kv_lora_rank": 512, "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
            "v_head_dim": 256, "linear_num_heads": 64, "linear_head_dim": 128,
            "linear_conv_kernel_dim": 4, "index_n_heads": 32, "index_head_dim": 128,
            "swiglu_limit": 10.0, "hc_mult": 4,
        }.items():
            if text.get(key) != expected:
                raise ValueError(f"GLM {key} requires {expected}, got {text.get(key)}")

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if name.startswith((TEXT_PREFIX + ".", "lm_head."))}

    def _layer(self, layer):
        if type(layer) is not int or not 0 <= layer <= 45:
            raise ValueError("GLM decoder layer must be in [0, 45]")
        return f"{TEXT_PREFIX}.layers.{layer}"

    def _projection(self, prefix, *, rows=None, columns=None):
        return self.load_linear(prefix, rows=rows, columns=columns)

    def load_attention(self, layer):
        prefix = self._layer(layer) + ".self_attn"
        if layer < 45 and layer % 4 != 3:
            return self._load_kda(prefix)
        head_channels = self.shard.partition(64 * 256, alignment=256)
        result = {"indexer.wq_b": self._projection(f"{prefix}.indexer.wq_b")}
        # Adjacent projections share a GEMM only when their stored precision is compatible.
        result["qkv_a_proj"] = self.load_mixed_linear_group(
            tuple(f"{prefix}.{name}" for name in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj")))
        result["q_b_proj"] = self._projection(f"{prefix}.q_b_proj", rows=head_channels)
        if self.linear_format(f"{prefix}.kv_b_proj") not in ("mxfp8", "bfloat16"):
            raise ValueError(f"{prefix}.kv_b_proj: MLA absorption requires BF16 or MXFP8 weights")
        result["kv_b_proj"] = self.load_linear_group(
            (f"{prefix}.kv_b_proj",), row_ranges=(self.shard.partition(64 * 512, alignment=512),))
        result["o_proj"] = self._projection(f"{prefix}.o_proj", columns=head_channels)
        for name in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight",
                     "indexer.k_norm.bias", "indexer.index_kpool_compress_ape", "indexer.index_kpool_compress_gate"):
            result[name] = self.load_tensor(f"{prefix}.{name}")
        return result

    def _load_kda(self, prefix):
        import torch

        channels = self.shard.partition(8192, alignment=128)
        heads = self.shard.partition(64)
        result = {name: self._projection(f"{prefix}.{name}", rows=channels)
                  for name in ("f_b_proj", "g_b_proj")}
        result["input_proj"] = self.load_mixed_linear_group(
            tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj")),
            row_ranges=(channels,) * 3 + (heads, None))
        result["g_a_proj"] = self._projection(f"{prefix}.g_a_proj")
        result["o_proj"] = self._projection(f"{prefix}.o_proj", columns=channels)
        result["A_log"] = self.load_tensor(f"{prefix}.A_log", extent=slice(*heads))
        result["dt_bias"] = self.load_tensor(f"{prefix}.dt_bias", extent=slice(*channels))
        result["o_norm.weight"] = self.load_tensor(f"{prefix}.o_norm.weight")
        result["conv1d.weight"] = torch.cat([
            self.load_tensor(f"{prefix}.{name}_conv1d.weight",
                             extent=slice(*channels)) for name in ("q", "k", "v")
        ]).squeeze(1)
        return result

    def load_mlp(self, layer, *, scale_scratch=None):
        prefix = self._layer(layer) + ".mlp"
        if layer < 3:
            extent = self.shard.partition(12288)
            gate_up = self.load_linear_group(tuple(f"{prefix}.{name}" for name in ("gate_proj", "up_proj")),
                                            row_ranges=(extent, extent))
            down = self._projection(f"{prefix}.down_proj", columns=extent)
            return MLPWeights(gate_up, down)
        extent = self.shard.partition(2048, alignment=128)
        shared_prefix = prefix + ".shared_experts"
        shared = MLPWeights(
            self.load_linear_group(tuple(f"{shared_prefix}.{name}" for name in ("gate_proj", "up_proj")),
                                 row_ranges=(extent, extent)),
            self._projection(f"{shared_prefix}.down_proj", columns=extent),
        )
        from ignition.preparation.quantization import expert_format

        csf = expert_format(self.quantization, prefix + ".experts")[0] == "nvfp4_csf"
        return MoEWeights(
            self.load_tensor(f"{prefix}.gate.weight"),
            self.load_tensor(f"{prefix}.gate.e_score_correction_bias"),
            self.load_experts(layer, scale_scratch=scale_scratch), shared,
            experts_a4_prefill=csf and self.expert_activation == "a16",
        )

    def load_experts(self, layer, *, expert_ids=None, scale_scratch=None):
        if layer < 3:
            raise ValueError("GLM routed experts begin at layer 3")
        prefix = self._layer(layer) + ".mlp.experts"
        ids = tuple(range(288)) if expert_ids is None else tuple(expert_ids)
        if (not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < 288 for i in ids)):
            raise ValueError("GLM expert IDs must be distinct integers in [0, 288)")
        prefixes = tuple(f"{prefix}.{i}" for i in ids)
        from ignition.preparation.quantization import expert_format

        fmt, mode = expert_format(self.quantization, prefix)
        if fmt != "nvfp4_csf":
            return self.load_fp4_experts(prefixes, hidden_size=4096, intermediate_size=2048,
                format=fmt, activation_mode=mode,
                intermediate_range=self.shard.partition(2048, alignment=128), swiglu_limit=10.0,
                scale_vectors=_expert_scales(self._sources, prefixes) if mode == "a4" else None)
        return self.load_nvfp4_experts(
            prefixes, hidden_size=4096, intermediate_size=2048,
            intermediate_range=self.shard.partition(2048, alignment=128),
            scale_vectors=_expert_scales(self._sources, prefixes), scale_scratch=scale_scratch,
            swiglu_limit=10.0, activation_mode=self.expert_activation,
        )

    def allocate_expert_scale_scratch(self, expert_count=288):
        import torch

        if type(expert_count) is not int or not 1 <= expert_count <= 288:
            raise ValueError("GLM expert count must be in [1, 288]")
        local = 2048 // self.shard.world_size
        return tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device) for shape in (
            (expert_count, 2 * local, 256), (expert_count, 4096, local // 16)))

    def load_mtp(self, scale_scratch):
        prefix = self._layer(45)
        return MTPWeights(self.load_attention(45), self.load_mlp(45, scale_scratch=scale_scratch),
            {name: self.load_tensor(f"{prefix}.{name}.weight") for name in (
                "enorm", "hnorm", "eh_proj", "input_layernorm", "post_attention_layernorm", "shared_head.norm")})

    def load_residual(self, layer):
        prefix = self._layer(layer)
        # mHC parameters execute in FP32; converting once lets every prepared graph variant share them.
        return {name: self.load_tensor(f"{prefix}.{name}").float() if name.startswith("hc_")
                else self.load_tensor(f"{prefix}.{name}") for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale")}


def _expert_scales(sources, prefixes):
    import torch

    w13, w2, a1, a2 = [], [], [], []
    for prefix in prefixes:
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                value = sources[name]
                if (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{name}: expected a finite positive FP32 scalar")
                values[projection, suffix] = value.item()
        if values["gate_proj", "weight_scale_2"] != values["up_proj", "weight_scale_2"]:
            raise ValueError(f"{prefix}: gate/up weight global scales differ")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(max(values["gate_proj", "input_scale"], values["up_proj", "input_scale"]))
        a2.append(values["down_proj", "input_scale"])
    return (torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
            torch.tensor(a1, dtype=torch.float32).reciprocal(), torch.tensor(a2, dtype=torch.float32).reciprocal())
