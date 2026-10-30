"""DFlash2 projections, grouped convolutions, and candidate selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights, quantize_bf16_linear_as_mxfp8
from ignition.preparation.checkpoint import inspect_indexed_checkpoint


if TYPE_CHECKING:
    from b12x.gemm.blockscaled import Weight


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: Weight | torch.Tensor
    output: Weight | torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights
    attention_conv: dict
    mlp_conv: dict


@dataclass(frozen=True)
class DraftWeights:
    projection: Weight | torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: Weight | torch.Tensor
    norm: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]
    selector: dict

    @classmethod
    def load(cls, snapshot, config, device, shard, *, progress=None, online_mxfp8=False):
        config.validate_shard(shard)
        manifest = inspect_indexed_checkpoint(snapshot)
        validate_metadata(manifest, config)
        c, a = config.text, config.attention
        q_range = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        first, last = shard.replicated_partition(a.kv_heads)
        kv_range = first * a.key_dim, last * a.key_dim
        intermediate = shard.partition(c["intermediate_size"], alignment=128)
        layers, context_parts = [], []

        def pack(weight):
            return quantize_bf16_linear_as_mxfp8(weight) if online_mxfp8 else weight

        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(prefix, rows=None, columns=None):
                return loader.load_tensor(prefix + ".weight", extent=(
                    slice(*(rows or (None, None))), slice(*(columns or (None, None)))))

            projection = pack(matrix("fc", rows=shard.partition(c["hidden_size"], alignment=64)))
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for i in range(config.layers):
                p = f"layers.{i}"
                qkv = torch.cat([matrix(p + ".self_attn." + n + "_proj", rows=r)
                    for n, r in zip(("q", "k", "v"), (q_range, kv_range, kv_range), strict=True)])
                context_kv = qkv[q_range[1] - q_range[0]:]
                context_parts.append(context_kv.clone() if online_mxfp8 else context_kv)
                del context_kv
                qkv = pack(qkv)
                output = pack(matrix(p + ".self_attn.o_proj", columns=q_range))
                qn, kn, inn, post = loader.load_tensors(tuple(p + n + ".weight" for n in
                    (".self_attn.q_norm", ".self_attn.k_norm", ".input_layernorm", ".post_attention_layernorm")))
                mlp = MLPWeights(pack(torch.cat([matrix(p + ".mlp." + n + "_proj", rows=intermediate)
                    for n in ("gate", "up")])), pack(matrix(p + ".mlp.down_proj", columns=intermediate)))
                def conv(name):
                    base, projection = loader.load_tensors((p + "." + name + ".base_kernel",
                                                           p + "." + name + ".kernel_projection.weight"))
                    return dict(base=base, projection=projection)
                layers.append(DraftLayerWeights(qkv, output, qn, kn, inn, post, mlp,
                                                conv("attention_conv"), conv("mlp_conv")))
            values = loader.load_tensors(("candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"))
        context = pack(torch.cat(context_parts))
        return cls(projection, hidden_norm, context, norm, tuple(layers),
                   dict(zip(("projection", "predecessor", "successor"), values, strict=True)))


def validate_metadata(manifest, config):
    c = config.text
    h, i = c["hidden_size"], c["intermediate_size"]
    q, kv, head = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"], c["head_dim"]
    d = c["dflash_config"]
    expected = {"fc.weight": (h, h * len(config.target_layers)), "hidden_norm.weight": (h,), "norm.weight": (h,),
        "candidate_selector.hidden_projection.weight": (d["selector_rank"], h),
        "candidate_selector.predecessor_codebook": (c["vocab_size"], d["selector_rank"]),
        "candidate_selector.successor_codebook": (c["vocab_size"], d["selector_rank"])}
    for layer in range(config.layers):
        p = f"layers.{layer}."
        for name, shape in {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h),
                "self_attn.v_proj": (kv, h), "self_attn.o_proj": (h, q), "self_attn.q_norm": (head,),
                "self_attn.k_norm": (head,), "input_layernorm": (h,), "post_attention_layernorm": (h,),
                "mlp.gate_proj": (i, h), "mlp.up_proj": (i, h), "mlp.down_proj": (h, i)}.items():
            expected[p + name + ".weight"] = shape
        for name in ("attention_conv", "mlp_conv"):
            expected[p + name + ".base_kernel"] = (2, d["conv_kernel_size"], h)
            expected[p + name + ".kernel_projection.weight"] = (2 * d["conv_kernel_size"] * (h // d["conv_group_size"]), h)
    if set(manifest.tensors) != set(expected):
        raise ValueError("DFlash2 checkpoint tensor names disagree with its architecture")
    for name, shape in expected.items():
        tensor = manifest.tensors[name]
        if tensor.shape != shape or tensor.dtype != "BF16":
            raise ValueError(f"{name}: DFlash2 requires BF16 storage with shape {shape}")
