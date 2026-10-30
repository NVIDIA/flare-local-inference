"""Canonical dense decoder parameters and checkpoint-calibrated projections."""

from dataclasses import dataclass, replace

import torch

from ignition.components.model import canonical_tree
from ignition.components.parallel import TensorParallelShard
from ignition.components.tensor_fp8 import TensorFP8Weight
from ignition.components.weights import DirectWeightLoader, quantize_bf16_linear_as_mxfp8

from .config import ModelConfig, TEXT_PREFIX


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    tree: torch.nn.Module
    projections: dict
    head: object
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    draft: object = None
    draft_head: object = None

    @classmethod
    def load(cls, manifest, device, *, progress=None, mtp=False, draft_snapshot=None,
             online_mxfp8=False, draft_head_quantization=None):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        config = ModelConfig.load(manifest.snapshot)
        if draft_head_quantization is not None and "lm_head" in config.quantization:
            raise ValueError("draft head conversion requires BF16 checkpoint vocabulary weights")
        if mtp:
            from .draft_config import DraftConfig
            config = replace(config, draft=DraftConfig.load(draft_snapshot, config))
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
        projections = {}
        before = torch.cuda.memory_allocated(device)
        with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".", "lm_head."),
                                progress=progress) as loader:
            mlp_groups = {
                f"layers.{index}.mlp.": tuple(TEXT_PREFIX + f".layers.{index}.mlp." + field
                                            for field in ("gate_proj", "up_proj"))
                for index in range(len(tree.layers))
                if all(loader.linear_format(TEXT_PREFIX + f".layers.{index}.mlp." + field) == "nvfp4"
                       for field in ("gate_proj", "up_proj"))
                and loader._nvfp4_calibration(TEXT_PREFIX + f".layers.{index}.mlp.gate_proj")
                    == loader._nvfp4_calibration(TEXT_PREFIX + f".layers.{index}.mlp.up_proj")
            }
            grouped_names = {part.removeprefix(TEXT_PREFIX + ".") + ".weight"
                             for parts in mlp_groups.values() for part in parts}
            for name, parameter in tree.named_parameters():
                if name in grouped_names:
                    stored = manifest.tensors[TEXT_PREFIX + "." + name].shape
                    if len(stored) != 2 or (stored[0], 2 * stored[1]) != tuple(parameter.shape):
                        raise ValueError(f"{name}: checkpoint projection shape differs from the model")
                    continue
                prefix = TEXT_PREFIX + "." + name.removesuffix(".weight")
                if name.endswith(".weight") and prefix in config.quantization:
                    weight = loader.load_linear(prefix)
                    packed = weight.packed if isinstance(weight, TensorFP8Weight) else weight
                    if (packed.out_features, packed.in_features) != tuple(parameter.shape):
                        raise ValueError(f"{prefix}: checkpoint projection shape differs from the model")
                    projections[name.removesuffix(".weight")] = weight
                    continue
                tensor = loader.load_tensor(TEXT_PREFIX + "." + name)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"{name}: checkpoint parameter shape differs from the model")
                parent, _, field = name.rpartition(".")
                if field == "weight" and isinstance(tree.get_submodule(parent), torch.nn.Linear):
                    convert = online_mxfp8 and tensor.dtype == torch.bfloat16
                    projections[parent] = quantize_bf16_linear_as_mxfp8(tensor) if convert else tensor
                    if convert:
                        continue
                setattr(tree.get_submodule(parent), field, torch.nn.Parameter(tensor, requires_grad=False))
            for prefix, parts in mlp_groups.items():
                combined = loader.load_linear_group(parts)
                expected = (2 * config.text["intermediate_size"], config.hidden)
                if (combined.out_features, combined.in_features) != expected:
                    raise ValueError(f"{prefix}gate_up: checkpoint projection shape differs from the model")
                projections[prefix + "gate_up"] = combined
            head = loader.load_linear("lm_head")
            head_shape = tuple(head.shape) if isinstance(head, torch.Tensor) else (head.out_features, head.in_features)
            if head_shape != (config.vocabulary, config.hidden):
                raise ValueError("Qwen vocabulary projection shape differs from the model")
            for index, layer in enumerate(tree.layers):
                if layer.block_type == "linear_attention":
                    prefix = f"layers.{index}.linear_attn."
                    parts = tuple(TEXT_PREFIX + "." + prefix + "in_proj_" + field for field in ("b", "a"))
                    combined = loader.load_linear_group(parts)
                    if online_mxfp8 and isinstance(combined, torch.Tensor):
                        combined = quantize_bf16_linear_as_mxfp8(combined)
                    projections[prefix + "in_proj_ba"] = combined
                    for field in ("b", "a"):
                        projections.pop(prefix + "in_proj_" + field, None)
                    qkvz = tuple(TEXT_PREFIX + "." + prefix + "in_proj_" + field for field in ("qkv", "z"))
                    if all(loader.linear_format(part) == "mxfp8" for part in qkvz):
                        projections[prefix + "in_proj_qkvz"] = loader.load_linear_group(qkvz)
                        for field in ("qkv", "z"):
                            projections.pop(prefix + "in_proj_" + field)
        draft = None
        draft_head = None
        if mtp:
            from ignition.components.dflash2_weights import DraftWeights
            draft = DraftWeights.load(draft_snapshot, config.draft, device, TensorParallelShard(0, 1),
                                      progress=progress, online_mxfp8=online_mxfp8)
            if draft_head_quantization is not None:
                from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head

                if not isinstance(head, torch.Tensor) or head.dtype != torch.bfloat16:
                    raise ValueError("draft head conversion requires BF16 checkpoint vocabulary weights")
                converters = {"mxfp8": mxfp8_draft_head, "nvfp4": nvfp4_draft_head}
                draft_head = converters[draft_head_quantization](head)
        return cls(config, tree, projections, head, torch.cuda.memory_allocated(device) - before,
                   draft=draft, draft_head=draft_head)
