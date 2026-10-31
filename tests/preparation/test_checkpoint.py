# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import json
import os
from collections import ChainMap
from dataclasses import replace
from pathlib import Path

import pytest

from ignition.preparation import CheckpointError, inspect_qwen38_checkpoint
from ignition.preparation import qwen38
from ignition.preparation.checkpoint import (
    _inspect_frontend_assets, _parse_json, _read_header, _validate_config, _validate_tensors,
)


def _write_header(path, header, payload=b"\0\0\0\0"):
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    return len(encoded) + 8


def test_header_inspection_never_reads_payload(tmp_path, monkeypatch):
    path = tmp_path / "tensor.safetensors"
    data_start = _write_header(path, {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    original_open = Path.open
    reads = []

    class HeaderOnlyFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, length):
            assert self.handle.tell() + length <= data_start
            reads.append(length)
            return self.handle.read(length)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: HeaderOnlyFile(original_open(self, *args, **kwargs)))
    shard, tensors = _read_header(path, path.name)
    assert sum(reads) == data_start
    assert shard.file_size == data_start + 4
    assert tensors["weight"].offset == data_start
    assert tensors["weight"].nbytes == 4


def test_single_safetensors_checkpoint_without_index(tmp_path):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    path = tmp_path / "model.safetensors"
    _write_header(path, {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    checkpoint = inspect_indexed_checkpoint(tmp_path)
    assert checkpoint.index_path is None
    assert len(checkpoint.shards) == 1
    assert checkpoint.tensors["weight"].shape == (1,)
    assert checkpoint.tensors["weight"].nbytes == 4


@pytest.mark.parametrize("dtype,shape,offsets", [
    ("F32", [2], [0, 4]),
    ("F32", [1], [0, 8]),
    ("BF16", [-1], [0, 4]),
    ("BF16", [True], [0, 2]),
    ("F32", [1], [True, 4]),
    ("FP4", [8], [0, 4]),
    (["U8"], [4], [0, 4]),
])
def test_invalid_tensor_ranges(tmp_path, dtype, shape, offsets):
    path = tmp_path / "bad.safetensors"
    _write_header(path, {"weight": {"dtype": dtype, "shape": shape, "data_offsets": offsets}})
    with pytest.raises(CheckpointError, match="weight"):
        _read_header(path, path.name)


@pytest.mark.parametrize("offsets,payload", [([1, 5], b"\0" * 5), ([0, 4], b"\0" * 5)])
def test_holes_and_trailing_payload_rejected(tmp_path, offsets, payload):
    path = tmp_path / "bad.safetensors"
    _write_header(path, {"weight": {"dtype": "U8", "shape": [4], "data_offsets": offsets}}, payload)
    with pytest.raises(CheckpointError, match="hole|cover"):
        _read_header(path, path.name)


def test_overlap_rejected(tmp_path):
    path = tmp_path / "bad.safetensors"
    _write_header(path, {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for name in ("a", "b")})
    with pytest.raises(CheckpointError, match="overlap"):
        _read_header(path, path.name)


@pytest.mark.parametrize("raw", [b"", b"\xff" * 8, (100 << 20 | 1).to_bytes(8, "little")])
def test_invalid_header_length(tmp_path, raw):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(raw)
    with pytest.raises(CheckpointError, match="length"):
        _read_header(path, path.name)


def test_duplicate_json_keys_rejected():
    with pytest.raises(CheckpointError, match="duplicate JSON key: weight"):
        _parse_json(b'{"weight": {}, "weight": {}}', "shard")


@pytest.fixture(scope="module")
def snapshot():
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    configured = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    path = Path(configured) if configured else default
    if not path.is_dir():
        if configured:
            pytest.fail(f"FLARE_QWEN38_SNAPSHOT does not exist: {path}")
        pytest.skip("exact Qwen3.8 checkpoint is not cached; set FLARE_QWEN38_SNAPSHOT")
    return path


@pytest.fixture(scope="module")
def manifest(snapshot):
    return inspect_qwen38_checkpoint(snapshot)


def test_actual_checkpoint_accounting(manifest):
    report = manifest.to_dict()
    assert report["tensor_count"] == 596671
    assert report["shard_count"] == 21
    assert report["payload_bytes"] == 102329644144
    assert report["gpu_stored_bytes"] == 71932785260
    assert report["ngram_stored_bytes"] == 28800138244
    assert report["excluded_stored_bytes"] == 1596720640
    assert sum(report[key] for key in ("gpu_stored_bytes", "ngram_stored_bytes", "excluded_stored_bytes")) == report["payload_bytes"]
    assert report["quantization"]["excluded_mtp_aliases"] == ["mtp.layers.48.mlp.experts"]
    assert report["validation"]["tensor_payloads_verified"] is False
    assert {name: asset["sha256"] for name, asset in report["frontend_assets"].items()} == qwen38.FRONTEND_SHA256


@pytest.mark.parametrize("changed", qwen38.FRONTEND_SHA256)
def test_modified_frontend_artifact_rejected(snapshot, tmp_path, changed):
    for name in qwen38.FRONTEND_SHA256:
        if name == changed:
            (tmp_path / name).write_bytes((snapshot / name).read_bytes() + b"\n")
        else:
            (tmp_path / name).symlink_to(snapshot / name)
    with pytest.raises(CheckpointError, match=changed):
        _inspect_frontend_assets(tmp_path)


@pytest.mark.parametrize("name,dtype,shape", [
    ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "F8_E4M3", (10240, 1280)),
    ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale.csf32_codes", "U8", (40, 1280)),
    ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale.csf32_exceptions", "I64", (200,)),
    ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight_scale", "U8", (2500012, 10)),
    ("model.visual.blocks.26.mlp.linear_fc2.weight", "U8", (1152, 4304)),
])
def test_actual_recipe_rejects_wrong_layouts(manifest, name, dtype, shape):
    modified = replace(manifest.tensors[name], dtype=dtype, shape=shape)
    with pytest.raises(CheckpointError, match="expected") as error:
        _validate_tensors(ChainMap({name: modified}, manifest.tensors))
    assert name in str(error.value)


def test_missing_expert_scale_rejected(manifest):
    name = "model.language_model.layers.0.mlp.experts.511.down_proj.input_scale"
    modified = dict(manifest.tensors)
    del modified[name]
    with pytest.raises(CheckpointError, match="missing required tensor") as error:
        _validate_tensors(modified)
    assert name in str(error.value)


def test_unmapped_tensor_rejected(manifest):
    extra = replace(manifest.tensors["lm_head.weight"], name="model.unexpected.weight")
    with pytest.raises(CheckpointError, match="unmapped tensor: model.unexpected.weight"):
        _validate_tensors(ChainMap({extra.name: extra}, manifest.tensors))


@pytest.mark.parametrize("field,value", [("hc_count", 2), ("ple_layer_ids", [1]), ("head_dim", 128)])
def test_changed_geometry_rejected(snapshot, field, value):
    config = json.loads((snapshot / "config.json").read_text())
    config["text_config"][field] = value
    with pytest.raises(CheckpointError, match=field):
        _validate_config(config)


def test_missing_quantization_recipe_rejected(snapshot):
    config = json.loads((snapshot / "config.json").read_text())
    name = "model.language_model.layers.0.mlp.experts"
    del config["quantization_config"]["quantized_layers"][name]
    with pytest.raises(CheckpointError, match="quantization") as error:
        _validate_config(config)
    assert name in str(error.value)


@pytest.mark.parametrize("entry", [
    {"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128]},
    {"quant_algo": "FP8"},
    {"quant_algo": "IQ2_XXS", "group_size": 256, "block_payload_bytes": 66, "packing": "ggml"},
    {"quant_algo": "Q8_0", "group_size": 32, "block_payload_bytes": 34, "packing": "ggml"},
])
def test_unsupported_dense_storage_rejected(snapshot, entry):
    config = json.loads((snapshot / "config.json").read_text())
    name = "model.language_model.layers.0.linear_attn.out_proj"
    config["quantization_config"]["quantized_layers"][name] = entry
    with pytest.raises(CheckpointError, match="dense") as error:
        _validate_config(config)
    assert name in str(error.value)


def test_checkpoint_metadata_does_not_import_torch(snapshot, monkeypatch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("checkpoint inspection imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    report = inspect_qwen38_checkpoint(snapshot).to_dict()
    assert report["reference_revision"] == qwen38.REVISION
    assert report["validation"]["tensor_payloads_read"] is False


@pytest.mark.parametrize("algo", ["NVFP4", "W4A16_NVFP4"])
def test_dense_storage_follows_quantization_config(snapshot, manifest, algo):
    config = json.loads((snapshot / "config.json").read_text())
    prefix = "model.language_model.layers.0.linear_attn.out_proj"
    config["quantization_config"]["quantized_layers"][prefix] = {"quant_algo": algo, "group_size": 16}
    _validate_config(config)
    weight = manifest.tensors[prefix + ".weight"]
    rows, columns = weight.shape
    tensors = dict(manifest.tensors)
    tensors[weight.name] = replace(weight, dtype="U8", shape=(rows, columns // 2))
    scales = tensors[prefix + ".weight_scale"]
    tensors[scales.name] = replace(scales, dtype="F8_E4M3", shape=(rows, columns // 16))
    name = prefix + ".weight_scale_2"
    tensors[name] = replace(weight, name=name, dtype="F32", shape=())
    if algo == "NVFP4":
        name = prefix + ".input_scale"
        tensors[name] = replace(weight, name=name, dtype="F32", shape=())
    _validate_tensors(tensors, config["quantization_config"]["quantized_layers"])
    with pytest.raises(CheckpointError, match="expected U8"):
        _validate_tensors(manifest.tensors, config["quantization_config"]["quantized_layers"])


def test_inspection_accepts_semantically_identical_config(snapshot, tmp_path):
    for path in snapshot.iterdir():
        if path.name != "config.json":
            (tmp_path / path.name).symlink_to(path)
    config = json.loads((snapshot / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config, sort_keys=True))
    manifest = inspect_qwen38_checkpoint(tmp_path)
    assert manifest.tensors["lm_head.weight"].shape == (248320, 2560)
