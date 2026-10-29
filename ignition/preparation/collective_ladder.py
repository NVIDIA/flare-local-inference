"""Final compiled geometries and runtime profiles for plain PCIe reductions."""

import json
from pathlib import Path

from ignition.errors import UsageError

MODES = {
    "joint_latency": ("full", "payload_rsag", "pull", "serial_push", "queued_push"),
    "joint_bandwidth": ("payload_rsag", "bulk_push", "bulk_pull", "pull", "serial_push", "queued_push"),
}


def validate(data):
    """Validate the JSON contract without importing CUDA or preparation backends."""
    def require(condition, message):
        if not condition:
            raise UsageError(f"collective ladder: {message}")

    require(isinstance(data, dict), "expected an object")
    require(set(data) == {"version", "name", "compute_capability", "world_size", "dtype",
                          "inplace", "cache_policy", "min_bytes", "buckets"}, "invalid fields")
    require(type(data["version"]) is int and data["version"] == 1, "unsupported version")
    require(isinstance(data["name"], str) and bool(data["name"]), "name must be nonempty")
    require(isinstance(data["compute_capability"], list) and len(data["compute_capability"]) == 2
            and all(type(v) is int and v >= 0 for v in data["compute_capability"]), "invalid compute capability")
    require(type(data["world_size"]) is int and data["world_size"] in (2, 3, 4, 6, 8, 10), "unsupported TP size")
    require(data["dtype"] == "bfloat16", "only bfloat16 reductions are supported")
    require(type(data["inplace"]) is bool, "inplace must be boolean")
    require(data["cache_policy"] in ("cold", "warm"), "cache_policy must be cold or warm")
    require(type(data["min_bytes"]) is int and data["min_bytes"] >= 16
            and data["min_bytes"] % 16 == 0, "min_bytes must be positive and 16-byte aligned")
    require(isinstance(data["buckets"], list) and bool(data["buckets"]), "buckets must be nonempty")
    previous = data["min_bytes"] - 16
    for row in data["buckets"]:
        require(isinstance(row, dict) and set(row) == {"max_bytes", "implementation", "threads", "blocks", "profiles"},
                "invalid bucket fields")
        upper = row["max_bytes"]
        require(type(upper) is int and upper > previous and upper % 16 == 0,
                "bucket max_bytes must increase and be 16-byte aligned")
        previous = upper
        require(isinstance(row["implementation"], str) and row["implementation"] in MODES, "unknown joint kernel")
        require(type(row["threads"]) is int and row["threads"] in (32, 64, 128, 256, 512), "invalid threads")
        require(type(row["blocks"]) is int and 1 <= row["blocks"] <= 64, "invalid physical grid")
        require(isinstance(row["profiles"], dict) and bool(row["profiles"]), "profiles must be nonempty")
        for mode, counts in row["profiles"].items():
            require(mode in MODES[row["implementation"]], "mode does not belong to joint kernel")
            require(data["world_size"] > 2 or mode not in ("serial_push", "queued_push"),
                    "destination serialization requires more than two ranks")
            require(isinstance(counts, list) and bool(counts)
                    and all(type(v) is int and 1 <= v <= row["blocks"] for v in counts), "invalid active CTA counts")
            require(len(set(counts)) == len(counts), "duplicate active CTA counts")
            require(mode != "serial_push" or counts == [1], "serial_push requires one active CTA")
    return data


def load(path):
    try:
        return validate(json.loads(Path(path).read_text()))
    except (OSError, ValueError) as error:
        raise UsageError(f"cannot read collective ladder {path}: {error}") from error


def select(data, size_bytes, *, world_size, capability, inplace):
    if world_size != data["world_size"] or tuple(capability) != tuple(data["compute_capability"]):
        raise UsageError("collective ladder does not match the reduction's TP size or compute capability")
    if inplace != data["inplace"]:
        raise UsageError("collective ladder does not match the reduction's input/output aliasing")
    if type(size_bytes) is not int or size_bytes % 16 or not data["min_bytes"] <= size_bytes <= data["buckets"][-1]["max_bytes"]:
        raise UsageError(f"collective ladder has no bucket for {size_bytes} bytes")
    return next(row for row in data["buckets"] if size_bytes <= row["max_bytes"])


def resolve(mode, path):
    if mode == "runtime":
        if path is None:
            raise UsageError("--collective-tuning runtime requires --collective-ladder PATH")
        return load(path)
    if path is not None:
        raise UsageError("--collective-ladder requires --collective-tuning runtime")
    return None
