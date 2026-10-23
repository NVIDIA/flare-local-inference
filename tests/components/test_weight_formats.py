"""Checkpoint quantization dispatch and projection fusion constraints."""

import json
from types import SimpleNamespace

import pytest

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.quantization import expert_format, linear_format, projection_group_format, quantized_layers
from ignition.recipes.qwen38.weights import QwenWeightLoader


def config(layers):
    return {"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": layers}}


@pytest.mark.parametrize("weight_name,activation,padded", [("weight", "silu", False),
                                                        ("weight_packed", "situ", True)])
@pytest.mark.parametrize("partition", [(0, 32), (32, 64)])
def test_mxfp4_csf_loader_preserves_packed_partitions_and_activation(
        monkeypatch, weight_name, activation, padded, partition):
    import torch
    from b12x.moe import fused_moe as moe

    loader = DirectWeightLoader(SimpleNamespace(), "cpu")
    loader._session = SimpleNamespace(flush=lambda: None)
    prefixes = ("expert0", "expert1")
    for expert, prefix in enumerate(prefixes):
        for index, projection in enumerate(("w1", "w3", "w2")):
            shape = (128, 32) if projection == "w2" else (64, 64)
            loader._sources[f"{prefix}.{projection}.{weight_name}"] = (
                torch.arange(4096).reshape(shape) + 7 * index + expert).to(torch.uint8)
    monkeypatch.setattr(loader, "_copy", lambda destination, source: destination.copy_(source))
    monkeypatch.setattr(loader, "_queue_csf_sources", lambda names, **selection: (names, selection))
    monkeypatch.setattr(moe, "plan_weights", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(moe, "prepare_weights", lambda **kwargs: SimpleNamespace(**kwargs))
    shapes = ((2, 256, 8), (2, 256, 8)) if padded else ((2, 4, 64), (2, 1, 128))
    scratch = tuple(torch.empty(shape, dtype=torch.uint8) for shape in shapes)
    options = dict(hidden_size=128, intermediate_size=64, intermediate_range=partition,
                   projection_names=("w1", "w3", "w2"), weight_name=weight_name,
                   activation=activation, scale_scratch=scratch)
    prepared = loader.load_mxfp4_csf_experts(prefixes, **options)
    assert prepared.plan.source.w13_layout.value == "w31"
    assert prepared.plan.activation.nonlinearity == activation
    first, last = partition
    for index, prefix in enumerate(prefixes):
        gate, up, down = (loader._sources[f"{prefix}.{projection}.{weight_name}"]
                          for projection in ("w1", "w3", "w2"))
        assert torch.equal(prepared.weights.w13[index], torch.cat((gate[first:last], up[first:last])))
        assert torch.equal(prepared.weights.w2[index], down[:, first // 2:last // 2])
    assert prepared.weights.w13_scales[0] == [f"{p}.{w}.weight_scale" for p in prefixes for w in ("w1", "w3")]
    assert prepared.weights.w13_scales[1]["row_range"] == partition
    assert prepared.weights.w2_scales[1]["column_range"] == (first // 32, last // 32)
    assert prepared.weights.w13_scale_scratch is scratch[0]
    assert prepared.weights.w2_scale_scratch is scratch[1]
    options["scale_scratch"] = (torch.empty(1, dtype=torch.uint8), scratch[1])
    with pytest.raises(ValueError, match="scale scratch"):
        loader.load_mxfp4_csf_experts(prefixes, **options)


@pytest.mark.parametrize("compression", [None, "csf"])
def test_mxfp4_storage_selection_preserves_projection_order_and_shared_scale_lifetime(monkeypatch, compression):
    names = [f"layer{layer}.expert{expert}" for layer in (0, 1) for expert in (0, 1)]
    compressed = {f"{name}.{projection}.scale" for name in names for projection in ("w1", "w3", "w2")}
    loader = DirectWeightLoader(SimpleNamespace(compressed_scales=compressed), "cpu")
    loader._session = object()
    monkeypatch.setattr(loader, "load_mxfp4_csf_experts", lambda prefixes, **kwargs: kwargs)
    monkeypatch.setattr(loader, "load_fp4_experts", lambda prefixes, **kwargs: kwargs)
    scratch = {}
    options = dict(hidden_size=128, intermediate_size=64, intermediate_range=(32, 64),
                   scale_scratch=scratch, scale_compression=compression,
                   projection_names=("w3", "w1", "w2"), scale_name="scale")
    first = loader.load_mxfp4_experts(names[:2], **options)
    second = loader.load_mxfp4_experts(names[2:], **options)
    assert first["projection_names"] == ("w1", "w3", "w2")
    assert first["scale_scratch"] is second["scale_scratch"]
    assert [tuple(tensor.shape) for tensor in first["scale_scratch"]] == [(2, 4, 64), (2, 1, 128)]
    ordinary = loader.load_mxfp4_experts(["draft.expert0"], **options)
    assert ordinary["projection_names"] == ("w3", "w1", "w2")
    assert ordinary["format"] == "mxfp4" and ordinary["activation_mode"] == "a8"
    if compression is None:
        assert "scale_scratch" not in ordinary and len(scratch) == 1
    else:
        assert ordinary["scale_compression"] == "csf" and len(scratch) == 2
        other_draft = loader.load_mxfp4_experts(["draft.expert1"], **options)
        assert ordinary["scale_scratch"] is other_draft["scale_scratch"]
        assert ordinary["scale_scratch"] is not first["scale_scratch"]
        converted = loader.load_mxfp4_experts(["native.expert0", "native.expert1"], **options)
        assert converted["scale_scratch"] is first["scale_scratch"]
    with pytest.raises(ValueError, match="complete MXFP4 scale encoding"):
        loader.load_mxfp4_experts([names[0], "draft.expert0"], **options)
    with pytest.raises(ValueError, match="scale compression"):
        loader.load_mxfp4_experts(names[:2], **(options | {"scale_compression": "zip"}))


@pytest.mark.parametrize("loader_type", [DirectWeightLoader, QwenWeightLoader])
@pytest.mark.parametrize("algo,group,selected", [("MXFP8", 32, "mxfp8"), ("MXFP4", 32, "mxfp4"), ("NVFP4", 16, "nvfp4"),
                                                ("W4A16_NVFP4", 16, "nvfp4")])
def test_checkpoint_selects_single_projection_format(tmp_path, monkeypatch, loader_type, algo, group, selected):
    prefix = "model.language_model.layers.0.linear_attn.out_proj"
    (tmp_path / "config.json").write_text(json.dumps(config({prefix: {"quant_algo": algo, "group_size": group}})))
    loader = loader_type(SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path), "cuda:0")
    loader._session = object()
    monkeypatch.setattr(loader, "load_linear_group", lambda prefixes, **kwargs: (projection_group_format(loader.quantization, prefixes), kwargs))
    result, slices = loader.load_linear(prefix, rows=(0, 32), columns=(16, 144))
    assert result == selected
    assert slices == {"row_ranges": ((0, 32),), "column_range": (16, 144)}


@pytest.mark.parametrize("formats", [("NVFP4", "NVFP4"), ("MXFP8", "NVFP4"),
                                      ("W4A16_NVFP4", "MXFP8")])
def test_nvfp4_fusion_rejected_before_tensor_access(tmp_path, formats):
    layers = {name: {"quant_algo": algo, "group_size": 32 if algo == "MXFP8" else 16}
              for name, algo in zip(("q", "k"), formats)}
    (tmp_path / "config.json").write_text(json.dumps(config(layers)))
    loader = DirectWeightLoader(SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path), "cuda:0")
    loader._session = object()
    with pytest.raises(ValueError, match="cannot fuse NVFP4.*only MoE expert w1/w3"):
        loader.load_linear_group(("q", "k"))


@pytest.mark.parametrize("entry", [None, {}, {"quant_algo": "AWQ", "group_size": 128},
    {"quant_algo": "MXFP8", "group_size": 16},
    {"quant_algo": "NVFP4", "group_size": 32},
    {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf_32x2_v1"}])
def test_unsupported_projection_encodings_rejected(entry):
    with pytest.raises(ValueError, match="projection|quantization|declaration"):
        linear_format({"proj": entry}, "proj")


def test_quantization_config_requires_supported_schema():
    with pytest.raises(ValueError, match="modelopt"):
        quantized_layers({"quantization_config": {"quant_method": "compressed-tensors"}})


@pytest.mark.parametrize("entry,expected", [
    ({"quant_algo": "MXFP8", "weight_block_size": [1, 32]}, "mxfp8"),
    ({"quant_algo": "MXFP8", "group_size": 32, "weight_block_size": [1, 32]}, "mxfp8"),
    ({"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128]}, "block_fp8"),
])
def test_modelopt_matrix_block_declarations(entry, expected):
    assert projection_group_format({"a": entry, "b": entry}, ("a", "b")) == expected


@pytest.mark.parametrize("entry", [
    {"quant_algo": "MXFP8", "weight_block_size": [128, 32]},
    {"quant_algo": "MXFP8", "group_size": 16, "weight_block_size": [1, 32]},
    {"quant_algo": "MXFP8", "group_size": 32, "weight_block_size": [1, 16]},
    {"quant_algo": "FP8_PB_WO", "weight_block_size": [32, 32]},
    {"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128], "weight_scale_encoding": "csf"},
])
def test_modelopt_conflicting_block_declarations_are_rejected(entry):
    with pytest.raises(ValueError, match="quantization"):
        linear_format({"projection": entry}, "projection")


@pytest.mark.parametrize("algo,group,encoding,expected", [
    ("NVFP4", 16, None, ("nvfp4", "a4")),
    ("NVFP4", 16, "csf_32x2_v1", ("nvfp4_csf", "a4")),
    ("NVFP4", 16, "csf", ("nvfp4_csf", "a4")),
    ("W4A16_NVFP4", 16, None, ("nvfp4", "a16")),
    ("MXFP4", 32, None, ("mxfp4", "a8")),
])
def test_expert_storage_and_activation_contract(algo, group, encoding, expected):
    entry = {"quant_algo": algo, "group_size": group}
    if encoding is not None:
        entry["weight_scale_encoding"] = encoding
    assert expert_format({"experts": entry}, "experts") == expected


def test_projection_fusion_requires_matching_formats():
    with pytest.raises(ValueError, match="same quantization format"):
        projection_group_format({"q": {"quant_algo": "MXFP4", "group_size": 32},
                                 "k": {"quant_algo": "MXFP8", "group_size": 32}}, ("q", "k"))


@pytest.mark.parametrize("algo,group,scale_dtype", [("NVFP4", 16, "F8_E4M3"),
    ("W4A16_NVFP4", 16, "F8_E4M3"), ("MXFP4", 32, "U8")])
def test_qwen_declares_ordinary_expert_storage(algo, group, scale_dtype):
    from ignition.preparation.qwen38 import _mlp

    layers = {"layer.mlp.experts": {"quant_algo": algo, "group_size": group}}
    tensors = {name: (dtype, shape) for name, dtype, shape in _mlp("layer.mlp", quantization=layers)}
    prefix = "layer.mlp.experts.0.gate_proj"
    assert tensors[prefix + ".weight"] == ("U8", (640, 1280))
    assert tensors[prefix + ".weight_scale"] == (scale_dtype, (640, 2560 // group))
    assert (prefix + ".input_scale" in tensors) == (algo == "NVFP4")
    assert (prefix + ".weight_scale_2" in tensors) == (algo != "MXFP4")
    assert not any("csf" in name for name in tensors)


@pytest.mark.parametrize("algo,group,mode", [("NVFP4", 16, "a4"),
    ("W4A16_NVFP4", 16, "a16"), ("MXFP4", 32, "a8")])
def test_mtp_experts_use_declared_format_and_mtp_scales(tmp_path, monkeypatch, algo, group, mode):
    import torch
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38.mtp_weights import MTPWeightLoader

    prefix = "mtp.layers.0.mlp.experts"
    (tmp_path / "config.json").write_text(json.dumps(config({prefix: {"quant_algo": algo, "group_size": group}})))
    loader = MTPWeightLoader(SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path), "cpu", shard=TensorParallelShard(1, 2))
    loader._session = object()
    loader._sources = {f"{prefix}.{expert}.{proj}.{suffix}": torch.tensor(value)
        for expert, value in ((7, .5), (4, .25))
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight_scale_2", "input_scale")}
    monkeypatch.setattr(loader, "load_fp4_experts", lambda prefixes, **kwargs: (prefixes, kwargs))
    prefixes, options = loader.load_experts(expert_ids=(7, 4))
    assert prefixes == (prefix + ".7", prefix + ".4")
    assert options["format"] == ("mxfp4" if algo == "MXFP4" else "nvfp4")
    assert options["activation_mode"] == mode
    assert options["intermediate_range"] == (320, 640)
    if mode == "a4":
        assert [v.tolist() for v in options["scale_vectors"]] == [[.5, .25], [.5, .25], [2., 4.], [2., 4.]]
    else:
        assert options["scale_vectors"] is None


@pytest.mark.parametrize("algo,group,payload", [("Q8_0", 32, 34), ("IQ2_XXS", 256, 66)])
def test_ggml_projection_and_expert_declarations(algo, group, payload):
    entry = dict(quant_algo=algo, group_size=group, block_payload_bytes=payload, packing="ggml")
    assert projection_group_format({"q": entry, "k": entry}, ("q", "k")) == algo.lower()
    assert expert_format({"expert": entry}, "expert") == (algo.lower(), "a16")
    for field, wrong in (("group_size", group // 2), ("block_payload_bytes", payload + 2),
                         ("packing", "row"), ("weight_scale_encoding", "csf")):
        with pytest.raises(ValueError, match="block encoding"):
            linear_format({"proj": entry | {field: wrong}}, "proj")


@pytest.mark.parametrize("invalid", [None, "gate_scale", "placeholder", "zero", "dtype"])
def test_nvfp4_reciprocal_globals_and_token_outer_scales_preserve_packed_tp_slices(monkeypatch, invalid):
    import torch
    from b12x.moe import fused_moe as moe
    from b12x._lib.intrinsics import reference

    loader = DirectWeightLoader(SimpleNamespace(), "cpu")
    loader._session = SimpleNamespace(flush=lambda: None)
    monkeypatch.setattr(loader, "_copy", lambda destination, source: destination.copy_(source))
    monkeypatch.setattr(reference, "swizzle_block_scale", lambda value: value)
    monkeypatch.setattr(moe, "plan_weights", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(moe, "prepare_weights", lambda **kwargs: SimpleNamespace(**kwargs))
    for index, projection in enumerate(("w1", "w3", "w2")):
        shape = (128, 64) if projection == "w2" else (64, 128)
        prefix = f"expert.{projection}"
        loader._sources[prefix + ".weight_packed"] = torch.arange(shape[0] * shape[1] // 2).reshape(shape[0], -1).to(torch.uint8)
        loader._sources[prefix + ".weight_scale"] = torch.full((shape[0], shape[1] // 16), index + 1.).to(torch.float8_e4m3fn)
        loader._sources[prefix + ".weight_global_scale"] = torch.tensor(8. if index == 2 else 4.)
        loader._sources[prefix + ".input_global_scale"] = torch.tensor(1.)
    if invalid == "gate_scale":
        loader._sources["expert.w3.weight_global_scale"].fill_(2.)
    elif invalid == "placeholder":
        loader._sources["expert.w2.input_global_scale"].fill_(2.)
    elif invalid == "zero":
        loader._sources["expert.w1.weight_global_scale"].zero_()
    elif invalid == "dtype":
        loader._sources["expert.w1.weight_global_scale"] = torch.tensor(4., dtype=torch.bfloat16)
    options = dict(hidden_size=128, intermediate_size=64, intermediate_range=(32, 64),
        format="nvfp4", activation_mode="a4", projection_names=("w1", "w3", "w2"),
        weight_name="weight_packed", w13_layout="w31", global_scale_name="weight_global_scale",
        global_scale_kind="reciprocal", input_scale_name="input_global_scale", activation_outer_scale="token",
        swiglu_limit=7., swiglu_alpha=1.702, swiglu_beta=1.)
    if invalid is not None:
        with pytest.raises(ValueError):
            loader.load_fp4_experts(["expert"], **options)
        return
    result = loader.load_fp4_experts(["expert"], **options)
    assert torch.equal(result.weights.w13[0], torch.cat([
        loader._sources[f"expert.{projection}.weight_packed"][32:64] for projection in ("w1", "w3")]))
    assert torch.equal(result.weights.w2[0], loader._sources["expert.w2.weight_packed"][:, 16:32])
    assert result.weights.w13_global_scales.tolist() == [.25]
    assert result.weights.w2_global_scales.tolist() == [.125]
    assert result.weights.input_scale is None and result.weights.intermediate_scale is None
    assert result.plan.activation.outer_scale == "token"
    assert result.plan.activation.swiglu_alpha == 1.702 and result.plan.activation.swiglu_beta == 1.


def test_mxfp8_projection_fusion_preserves_unsigned_exponent_payload(monkeypatch):
    import torch
    from b12x.gemm import blockscaled

    loader = DirectWeightLoader(SimpleNamespace(), "cpu")
    loader._session = SimpleNamespace(flush=lambda: None)
    loader.quantization = {name: {"quant_algo": "MXFP8", "group_size": 32} for name in ("q", "k")}
    for name, gain in (("q", 1.), ("k", 2.)):
        loader._sources[name + ".weight"] = torch.full((8, 64), gain).to(torch.float8_e4m3fn)
        loader._sources[name + ".weight_scale_inv"] = torch.arange(16, dtype=torch.uint8).reshape(8, 2) + 121
    monkeypatch.setattr(loader, "_copy", lambda destination, source: destination.copy_(source))
    monkeypatch.setattr(blockscaled, "pack_weight", lambda weight, scales, **kwargs: (weight, scales, kwargs))
    values, scales, _ = loader.load_linear_group(("q", "k"), row_ranges=((1, 5), (2, 6)),
        column_range=(32, 64), scale_name="weight_scale_inv")
    assert torch.equal(values.float(), torch.cat((torch.ones(4, 32), torch.full((4, 32), 2.))))
    assert torch.equal(scales, torch.cat((loader._sources["q.weight_scale_inv"][1:5, 1:],
                                         loader._sources["k.weight_scale_inv"][2:6, 1:])))
