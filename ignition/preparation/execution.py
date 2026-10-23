"""Typed graph I/O metadata for the native two-frame execution ABI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ignition.components.capacity import BATCH_CAPACITIES, request_capacity

from .graphs import PreparedGraphBundle


@dataclass(frozen=True)
class GraphSignature:
    kind: str
    batch_capacity: int
    token_capacity: int
    inputs: tuple[str, ...] = ()
    sampling_mode: str = "general"
    speculation_enabled: bool | None = None
    prefill_capacity: int = 0
    packed_token_capacity: int = 0
    draft_tokens: int | None = None

    def __post_init__(self):
        if self.kind not in ("init", "restore", "snapshot", "prefill", "scored_prefill", "decode", "resume",
                             "vision", "speculative", "mixed", "proposal", "verification", "packed_speculative"):
            raise ValueError("unknown prepared graph operation")
        if self.kind == "scored_prefill" and (self.batch_capacity != 1
                or self.token_capacity < 64 or self.token_capacity % 64
                or "prompt_target_tokens" not in self.inputs):
            raise ValueError("scored prefill requires aligned prompt rows and next-token targets")
        if "prompt_logprobs" in self.inputs:
            raise ValueError("prompt logprob outputs cannot be uploaded")
        if self.sampling_mode != "general":
            raise ValueError("graphs use the general stochastic sampling path")
        if self.speculation_enabled is not None and type(self.speculation_enabled) is not bool:
            raise TypeError("graph speculation mode must be a boolean or None")
        if self.kind in ("speculative", "packed_speculative") and self.speculation_enabled is False:
            raise ValueError("speculative graphs require speculation")
        if self.kind == "decode" and self.speculation_enabled is True:
            raise ValueError("ordinary decode graphs require disabled speculation")
        if any(type(value) is not int or value <= 0 for value in (self.batch_capacity, self.token_capacity)):
            raise ValueError("graph capacities must be positive integers")
        if self.kind in ("speculative", "proposal", "verification", "packed_speculative") and (
                self.batch_capacity not in BATCH_CAPACITIES or not 1 <= self.token_capacity <= 8):
            raise ValueError("speculative graphs require power-of-two batch capacities through 64 and width 1 through 8")
        if type(self.prefill_capacity) is not int or self.prefill_capacity < 0:
            raise ValueError("prefill capacity must be a nonnegative integer")
        if self.kind == "mixed":
            if (self.batch_capacity not in BATCH_CAPACITIES[1:]
                    or not 1 <= self.token_capacity <= 8
                    or self.prefill_capacity < 64 or self.prefill_capacity % 64
                    or self.prefill_capacity + (self.batch_capacity - 1) * self.token_capacity >= 1 << 31
                    or self.speculation_enabled is None
                    or (not self.speculation_enabled and self.token_capacity != 1)):
                raise ValueError("mixed graphs require a prompt capacity, at least two rows, and an explicit decode mode")
            if "prefill_row" not in self.inputs:
                raise ValueError("mixed graphs require the prefill_row input")
        elif self.prefill_capacity:
            raise ValueError("only mixed graphs declare a separate prefill capacity")
        if (type(self.packed_token_capacity) is not int or self.packed_token_capacity < 0
                or (self.packed_token_capacity and (self.kind not in ("verification", "packed_speculative")
                    or not self.batch_capacity <= self.packed_token_capacity <= self.batch_capacity * self.token_capacity))):
            raise ValueError("packed verification capacity must fit the batch and per-request prefix bounds")
        if self.kind == "packed_speculative":
            if (type(self.draft_tokens) is not int or not 1 <= self.draft_tokens < self.token_capacity
                    or not self.packed_token_capacity):
                raise ValueError("packed speculative rounds require explicit draft depth and target-row capacity")
            if "query_lengths" in self.inputs or "speculative_prefix_limits" not in self.inputs:
                raise ValueError("packed speculative rounds bind prefix limits, not actual query lengths")
        elif self.draft_tokens is not None:
            raise ValueError("explicit draft depth belongs to packed speculative rounds")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("graph input names must be unique")
        if any(name in self.inputs for name in ("frame_index", "submission_sequence", "completion_records",
                                               "completion_logprobs", "logprob_requested", "logprob_token_ids",
                                               "logprob_token_id_counts", "readiness_timestamps",
                                               "completion_speculative_survival")):
            raise ValueError("reserved execution buffers are bound automatically")


class PreparedExecution(PreparedGraphBundle):
    """Retain immutable graph ownership and export the matching native ABI.

    Graph I/O and declared persistent-state views belong in ``buffers``. Their
    storage owners, packed weights, and graph scratch belong in ``resources``. Native
    validation rejects overlapping I/O spans, incompatible graph capacities,
    and malformed reserved buffers before its worker becomes ready.
    """

    def __init__(self, *, signatures: Mapping[str, GraphSignature], max_active: int = 4,
                 estimated_durations_ns=None, export_zero_ranges=(), export_events=None, **kwargs):
        super().__init__(**kwargs)
        request_capacity(max_active)
        if signatures.keys() != self.graphs.keys():
            raise ValueError("every prepared graph requires exactly one signature")
        if any(not isinstance(value, GraphSignature) for value in signatures.values()):
            raise TypeError("prepared graph signatures must be GraphSignature objects")
        required = {"frame_index", "submission_sequence", "completion_records"}
        if not required <= self.buffers.keys():
            raise ValueError("execution requires frame index, submission sequence, and completion buffers")
        if any(s.kind in ("mixed", "packed_speculative") for s in signatures.values()) and self.buffers["completion_records"].shape[-1] != 24:
            raise ValueError("mixed and packed speculative graphs require ABI3 completion vectors")
        for signature in signatures.values():
            if not set(signature.inputs) <= self.buffers.keys():
                raise ValueError("graph signature references an unregistered input buffer")
        from ignition.components.completion import validate_completion_buffers
        validate_completion_buffers(self.buffers, request_capacity(max_active))
        self.signatures = MappingProxyType(dict(signatures))
        durations = dict.fromkeys(self.graphs, 0) if estimated_durations_ns is None else dict(estimated_durations_ns)
        if durations.keys() != self.graphs.keys() or any(
                type(value) is not int or not 0 <= value < 1 << 64 for value in durations.values()):
            raise ValueError("graph duration estimates must cover the catalog with unsigned nanoseconds")
        self.estimated_durations_ns = MappingProxyType(durations)
        self.max_active = max_active
        self.export_zero_ranges = tuple(export_zero_ranges)
        self.export_events = dict(export_events or {})

    def manifest(self, *, all_variants=False) -> dict[str, Any]:
        context, handles, spans = self._validate_resources()
        descriptors = []
        offsets = {}
        cursor = 0
        reserved = ("frame_index", "submission_sequence")
        if "logprob_requested" in self.buffers:
            reserved += tuple(name for name in ("logprob_requested", "logprob_token_ids",
                                                "logprob_token_id_counts") if name in self.buffers)
        input_names = set(reserved)
        for signature in self.signatures.values():
            input_names.update(signature.inputs)
        for (name, tensor), (address, nbytes) in zip(self.buffers.items(), spans, strict=True):
            descriptors.append({"name": name, "address": address, "nbytes": nbytes,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape), "strides": list(tensor.stride())})
            if name in input_names:
                cursor = (cursor + 7) // 8 * 8
                offsets[name] = (cursor, nbytes)
                cursor += nbytes
        graphs = []
        speculative = any(signature.kind in ("speculative", "proposal", "verification", "packed_speculative") for signature in self.signatures.values())
        for (name, _), handle in zip(self.graphs.items(), handles, strict=True):
            signature = self.signatures[name]
            if not all_variants and signature.speculation_enabled not in (None, speculative):
                continue
            inputs = (*reserved, *signature.inputs)
            graphs.append({"name": name, "kind": signature.kind, "handle": handle,
                "sampling_mode": signature.sampling_mode,
                "batch_capacity": signature.batch_capacity, "token_capacity": signature.token_capacity,
                "prefill_capacity": signature.prefill_capacity,
                "estimated_duration_ns": self.estimated_durations_ns[name],
                "input_bindings": [{"buffer_name": field, "host_offset": offsets[field][0],
                                    "nbytes": offsets[field][1]} for field in inputs]})
            if signature.speculation_enabled is not None:
                graphs[-1]["speculation_enabled"] = signature.speculation_enabled
            if signature.packed_token_capacity:
                graphs[-1]["packed_token_capacity"] = signature.packed_token_capacity
            if signature.draft_tokens is not None:
                graphs[-1].update(draft_tokens=signature.draft_tokens,
                                  verify_tokens=signature.token_capacity, draft_source=1)
        abi_version = 3 if self.buffers["completion_records"].shape[-1] == 24 else 2
        return {"abi_version": abi_version, "context": context, "device": self.device,
                "max_active": self.max_active, "input_frame_bytes": (cursor + 7) // 8 * 8,
                "buffers": descriptors, "graphs": graphs}

    def start_executor(self, *, peers=None):
        """Synchronize preparation and consume the bundle into the native executor."""
        import flare_heliopause
        import torch

        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            arguments = {} if peers is None else {"peer_fds": [peer.fileno() for peer in peers]}
            executor = flare_heliopause.GraphExecutor(self, manifest, **arguments)
            self._consumed = True
            return executor

    def serve_rank(self, socket, *, rank, world):
        """Hand a peer to Rust until every rank drains; Python does no step work."""
        import flare_heliopause
        import torch

        with self._handoff_lock:
            if self._consumed:
                raise RuntimeError("graph bundle has already been consumed by a native worker")
            manifest = self.manifest()
            torch.cuda.synchronize(self.device)
            self._consumed = True
        flare_heliopause.serve_tensor_parallel_rank(self, manifest, socket.fileno(), rank, world)
