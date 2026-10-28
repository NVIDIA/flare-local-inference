"""Validate checkpoint metadata without importing torch or reading tensor payloads.

Tensor transfer belongs to b12x.loader.DirectWeightSession. Its manifest reader
requires a CUDA session; this preflight runs before CUDA initialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from . import qwen38
from .quantization import expert_format, linear_format, quantized_layers


class CheckpointError(ValueError):
    """Checkpoint metadata does not satisfy the prepared model's contract."""


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shard: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def component(self) -> str:
        if self.name.startswith("mtp."):
            return "mtp"
        if ".ngram_embedding." in self.name:
            return "ple_tables"
        if self.name.startswith("model.visual."):
            return "vision"
        if ".mlp.experts." in self.name:
            return "decoder_experts"
        return "decoder_other"


@dataclass(frozen=True, slots=True)
class ShardMetadata:
    path: str
    file_size: int
    header_size: int
    header_sha256: str
    tensor_count: int


@dataclass(frozen=True, slots=True)
class IndexedCheckpoint:
    snapshot: Path
    tensors: Mapping[str, TensorMetadata]
    shards: tuple[ShardMetadata, ...]

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return tuple(self.snapshot / shard.path for shard in self.shards)

    @property
    def index_path(self) -> Path | None:
        path = self.snapshot / "model.safetensors.index.json"
        return path if path.is_file() else None

    @property
    def metadata_path(self) -> Path:
        return self.snapshot


@dataclass(frozen=True, slots=True)
class CheckpointManifest(IndexedCheckpoint):
    metadata_sha256: Mapping[str, str]
    headers_sha256: str
    generation_config: Mapping

    def to_dict(self) -> dict:
        sizes = Counter()
        counts = Counter()
        csf_bytes = Counter()
        for tensor in self.tensors.values():
            sizes[tensor.component] += tensor.nbytes
            counts[tensor.component] += 1
            if ".csf32_" in tensor.name or ".nvfp4_csf_" in tensor.name:
                csf_bytes[tensor.name.rsplit(".", 1)[1]] += tensor.nbytes
        gpu_bytes = sum(sizes[key] for key in ("decoder_experts", "decoder_other", "vision"))
        return {
            "model_id": qwen38.MODEL_ID,
            "reference_revision": qwen38.REVISION,
            "architecture": qwen38.ARCHITECTURE,
            "snapshot": str(self.snapshot),
            "validation": {
                "tensor_payloads_read": False,
                "tensor_payloads_verified": False,
                "checks": [
                    "config", "generation_config", "quantization_map", "tensor_names",
                    "shapes", "dtypes", "file_ranges", "index_coverage", "header_fingerprints",
                    "frontend_asset_fingerprints",
                ],
                "unverified": [
                    "tensor_values", "scale_values", "csf_exception_contents",
                    "ple_hash_constants", "inference_correctness",
                ],
            },
            "metadata_sha256": dict(self.metadata_sha256),
            "frontend_assets": {
                name: {"path": str(self.snapshot / name), "sha256": self.metadata_sha256[name]}
                for name in qwen38.FRONTEND_SHA256
            },
            "headers_sha256": self.headers_sha256,
            "tensor_count": len(self.tensors),
            "shard_count": len(self.shards),
            "payload_bytes": sum(sizes.values()),
            "gpu_stored_bytes": gpu_bytes,
            "ngram_stored_bytes": sizes["ple_tables"],
            "excluded_stored_bytes": sizes["mtp"],
            "components": {
                key: {"tensor_count": counts[key], "stored_bytes": sizes[key]}
                for key in sorted(sizes)
            },
            "geometry": {
                "text": dict(qwen38.TEXT_GEOMETRY),
                "vision": dict(qwen38.VISION_GEOMETRY),
                "gdn_layers": [i for i in range(48) if i % 4 != 3],
                "sparse_attention_layers": list(range(3, 48, 4)),
            },
            "quantization": {
                "module_counts": dict(Counter(
                    entry["quant_algo"] for entry in quantized_layers(
                        _parse_json((self.snapshot / "config.json").read_bytes(), "config.json")
                    ).values()
                )),
                "excluded_mtp_aliases": [qwen38.EXCLUDED_QUANTIZATION_ALIAS],
                "csf_stored_bytes": dict(csf_bytes),
            },
            "ple": {
                "prefix": qwen38.PLE_PREFIX,
                "table_planes": 128,
                "rows_per_plane": 2500012,
                "packed_width": 80,
                "scale_width": 10,
                "logical_embedding_width": 160,
            },
            "generation_config": dict(self.generation_config),
            "shards": [
                {
                    "path": shard.path, "file_size": shard.file_size,
                    "header_size": shard.header_size, "header_sha256": shard.header_sha256,
                    "tensor_count": shard.tensor_count,
                }
                for shard in self.shards
            ],
        }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, source: str) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise CheckpointError(f"{source}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"{source}: expected a JSON object")
    return value


_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1,
                "I8": 1, "U8": 1, "U16": 2, "I32": 4, "U32": 4, "I64": 8}


def _read_header(path: Path, shard: str) -> tuple[ShardMetadata, dict[str, TensorMetadata]]:
    from .checkpoint_source import source_for

    remote = source_for(path)
    if remote is not None:
        name = remote.name(path)
        size = remote.files[name]["bytes"]
        content = remote.headers[name]
        prefix, raw = content[:8], content[8:]
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or length == 0 or length > 100 << 20 or length > size - 8 or len(raw) != length:
            raise CheckpointError(f"{shard}: invalid remote safetensors header")
    else:
        with path.open("rb", buffering=0) as source:
            size = os.fstat(source.fileno()).st_size
            prefix = source.read(8)
            if len(prefix) != 8:
                raise CheckpointError(f"{shard}: truncated safetensors length")
            length = int.from_bytes(prefix, "little")
            if length == 0 or length > 100 << 20 or length > size - 8:
                raise CheckpointError(f"{shard}: invalid safetensors header length")
            raw = source.read(length)
            if len(raw) != length:
                raise CheckpointError(f"{shard}: truncated safetensors header")
    header = _parse_json(raw, shard)
    tensors = {}
    data_start = 8 + length
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values()):
                raise CheckpointError(f"{shard}: invalid safetensors metadata")
            continue
        if not name or not isinstance(entry, dict):
            raise CheckpointError(f"{shard}: invalid tensor entry {name!r}")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        dtype = entry.get("dtype")
        if (
            not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(v) is not int or v < 0 for v in offsets)
            or not isinstance(dtype, str) or dtype not in _DTYPE_BYTES
        ):
            raise CheckpointError(f"{shard}: invalid shape, dtype, or offsets for {name}")
        start, end = offsets
        nbytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end - start != nbytes or end > size - data_start:
            raise CheckpointError(f"{shard}: invalid byte range for {name}")
        tensors[name] = TensorMetadata(
            name, shard, data_start + start, nbytes, tuple(shape), dtype
        )
    end = data_start
    for tensor in sorted(tensors.values(), key=lambda value: (value.offset, value.nbytes)):
        if tensor.offset != end:
            raise CheckpointError(f"{shard}: overlap or hole at {tensor.name}")
        end += tensor.nbytes
    if end != size:
        raise CheckpointError(f"{shard}: tensor ranges do not cover the file")
    metadata = ShardMetadata(shard, size, length, hashlib.sha256(raw).hexdigest(), len(tensors))
    return metadata, tensors


def _validate_config(config: dict) -> None:
    if (
        config.get("architectures") != [qwen38.ARCHITECTURE]
        or config.get("model_type") != "qwen4_exp"
    ):
        raise CheckpointError(f"expected architecture {qwen38.ARCHITECTURE}")
    for section, expected in (
        ("text_config", qwen38.TEXT_GEOMETRY), ("vision_config", qwen38.VISION_GEOMETRY)
    ):
        actual = config.get(section)
        if not isinstance(actual, dict):
            raise CheckpointError(f"missing {section}")
        for field, value in expected.items():
            if actual.get(field) != value:
                raise CheckpointError(
                    f"{section}.{field}: expected {value!r}, got {actual.get(field)!r}"
                )
    expected_layers = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)]
    if config["text_config"].get("layer_types") != expected_layers:
        raise CheckpointError(
            "text_config.layer_types: expected 36 GDN and 12 interleaved sparse-attention layers"
        )
    try:
        actual = quantized_layers(config)
        expected = qwen38.expected_quantization()
        for name, recipe in expected.items():
            if name.endswith(".experts"):
                fmt, mode = expert_format(actual, name)
                if name.startswith("mtp.") and fmt == "nvfp4_csf":
                    raise CheckpointError("Qwen MTP experts require ordinary NVFP4 or MXFP4 storage")
            elif name == f"{qwen38.PLE_PREFIX}.ngram_embedding":
                if actual.get(name) != recipe:
                    raise CheckpointError(
                        f"quantized_layers.{name}: expected {recipe!r}, got {actual.get(name)!r}"
                    )
            else:
                storage = linear_format(actual, name)
                if storage not in ("mxfp8", "mxfp4", "nvfp4"):
                    raise CheckpointError(
                        f"{name}: Qwen dense projections do not support {storage} storage"
                    )
    except ValueError as exc:
        raise CheckpointError(str(exc)) from exc
    extra = actual.keys() - expected.keys()
    if extra:
        raise CheckpointError(f"unmapped quantization entry: {min(extra)}")


def _validate_tensors(tensors: Mapping[str, TensorMetadata], quantization=None) -> None:
    unmatched = set(tensors)
    for name, dtype, shape in qwen38.expected_tensors(quantization):
        tensor = tensors.get(name)
        if tensor is None:
            raise CheckpointError(f"missing required tensor: {name}")
        if (
            tensor.dtype != dtype or len(tensor.shape) != len(shape)
            or any(expected is not None and actual != expected
                   for actual, expected in zip(tensor.shape, shape))
        ):
            raise CheckpointError(
                f"{name}: expected {dtype} {shape}, got {tensor.dtype} {tensor.shape}"
            )
        if shape == (None,) and tensor.shape[0] > 102400:
            raise CheckpointError(f"{name}: CSF exception count exceeds the scale element count")
        unmatched.remove(name)
    for name in tuple(unmatched):
        if not name.endswith(".input_scale") or quantization is None:
            continue
        entry = quantization.get(name.removesuffix(".input_scale"), {})
        if entry.get("quant_algo") != "NVFP4" or entry.get("weight_scale_encoding") is not None:
            continue
        tensor = tensors[name]
        if tensor.dtype != "F32" or tensor.shape not in ((), (1,)):
            raise CheckpointError(f"{name}: expected one F32 activation scale")
        unmatched.remove(name)
    if unmatched:
        raise CheckpointError(f"unmapped tensor: {min(unmatched)}")


def _inspect_frontend_assets(snapshot: Path) -> dict[str, str]:
    fingerprints = {}
    for name, expected in qwen38.FRONTEND_SHA256.items():
        with (snapshot / name).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise CheckpointError(
                f"{name}: SHA256 does not match reference revision {qwen38.REVISION}"
            )
        fingerprints[name] = digest
    return fingerprints


def inspect_indexed_checkpoint(snapshot: str | Path) -> IndexedCheckpoint:
    """Read indexed safetensors geometry and validate byte ranges without CUDA.

    Model recipes validate configuration, tensor names and expected precision.
    The direct loader revalidates source file identity before moving payloads.
    """
    snapshot = Path(snapshot).expanduser().absolute()
    from .checkpoint_source import source_for

    remote = source_for(snapshot)
    container = snapshot / "manifest.json"
    if container.is_file():
        description = _parse_json(container.read_bytes(), "manifest.json")
        if description.get("schema") == "lil-mxfp4-csf-checkpoint/1":
            from .csf_checkpoint import inspect_mxfp4_csf

            return inspect_mxfp4_csf(snapshot, description)
    if not (snapshot / "model.safetensors.index.json").exists() and (
            (snapshot / "model.safetensors").is_file() or remote is not None and "model.safetensors" in remote.files):
        shard, tensors = _read_header(snapshot / "model.safetensors", "model.safetensors")
        return IndexedCheckpoint(snapshot, MappingProxyType(tensors), (shard,))
    index = _parse_json((snapshot / "model.safetensors.index.json").read_bytes(),
                        "model.safetensors.index.json")
    return _inspect_indexed_checkpoint(snapshot, index)


def _inspect_indexed_checkpoint(snapshot: Path, index: dict) -> IndexedCheckpoint:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("model.safetensors.index.json: missing weight_map")
    shard_names = set()
    for name, shard in weight_map.items():
        if not name or not isinstance(shard, str):
            raise CheckpointError(f"invalid indexed shard for {name}: {shard!r}")
        if shard not in shard_names:
            path = PurePosixPath(shard)
            if path.is_absolute() or ".." in path.parts or not shard.endswith(".safetensors"):
                raise CheckpointError(f"invalid indexed shard for {name}: {shard!r}")
            shard_names.add(shard)
    tensors, shards = {}, []
    for shard in sorted(shard_names):
        record, entries = _read_header(snapshot / shard, shard)
        for name in entries:
            if name in tensors:
                raise CheckpointError(f"duplicate tensor across shards: {name}")
            if weight_map.get(name) != shard:
                raise CheckpointError(f"index/shard mismatch for {name}: found in {shard}")
        tensors.update(entries)
        shards.append(record)
    missing = weight_map.keys() - tensors.keys()
    if missing:
        raise CheckpointError(f"indexed tensor missing from shards: {min(missing)}")
    total = sum(tensor.nbytes for tensor in tensors.values())
    if (
        not isinstance(index.get("metadata"), dict)
        or index["metadata"].get("total_size") != total
    ):
        raise CheckpointError(f"index total_size does not match tensor payload bytes: {total}")
    return IndexedCheckpoint(snapshot, MappingProxyType(tensors), tuple(shards))


def inspect_qwen38_checkpoint(snapshot: str | Path) -> CheckpointManifest:
    """Validate model geometry and declared storage; tensor payloads are not read.

    The returned shard_paths and index_path can be passed to b12x's weight
    session. Exclude mtp.* while routing the remaining tensors into the recipe.
    File paths remain caller-owned and must not change between inspection and
    loading. The loader must perform its own range and source-lifetime checks.
    """
    snapshot = Path(snapshot).expanduser().absolute()
    metadata = _inspect_frontend_assets(snapshot)

    def read_json(name):
        raw = (snapshot / name).read_bytes()
        metadata[name] = hashlib.sha256(raw).hexdigest()
        return _parse_json(raw, name)

    config = read_json("config.json")
    _validate_config(config)
    index = read_json("model.safetensors.index.json")
    generation = read_json("generation_config.json")
    indexed = _inspect_indexed_checkpoint(snapshot, index)
    tensors, shards = indexed.tensors, indexed.shards
    _validate_tensors(tensors, quantized_layers(config))
    header_digest = hashlib.sha256()
    for shard in shards:
        header_digest.update(shard.path.encode() + b"\0" + bytes.fromhex(shard.header_sha256))
    return CheckpointManifest(
        snapshot, MappingProxyType(tensors), tuple(shards), MappingProxyType(metadata),
        header_digest.hexdigest(), MappingProxyType(generation),
    )
