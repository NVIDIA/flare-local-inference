"""Captured admission, immutable-prefix restore, and exact-prefix publication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.components.capacity import BATCH_CAPACITIES
from ignition.components.sampling import (
    AdmissionBuffers,
    PreparedAdmission,
    _check_tensor,
    _compile,
    _completion_fields,
    _write_completion,
)

from .sampling import DeviceRequestState
from .state import (
    HybridStatePool,
    PreparedCheckpointCaches,
    PreparedStateTransfers,
    StateTransferMetadata,
)

TRANSFER_ERROR_BASE = 64
INVALID_COMMAND = 8
INVALID_ENDPOINT = 9


@triton.jit
def _stage(Slots, Epochs, Count, Sources, Destinations, Operations, RawSlots, Accepted, Expected, StateCount,
           Generation, Status, Consumed, AdmissionErrors, StateRows, RequestSlots, FilteredRequests,
           FilteredSources, FilteredDestinations, FilteredOperations, FilteredRawSlots, FilteredAccepted, FilteredExpected,
           FilteredCount, Errors, MODE: tl.constexpr, CAPACITY: tl.constexpr, MemoryCapacity,
           CS: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    count, state_count = tl.load(Count), tl.load(StateCount)
    if row == 0:
        tl.store(FilteredCount, tl.minimum(tl.maximum(count, 0), CAPACITY))
    slot, epoch = tl.load(Slots + row), tl.load(Epochs + row)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    operation, raw_slot, accepted = tl.load(Operations + row), tl.load(RawSlots + row), tl.load(Accepted + row)
    expected = tl.load(Expected + row)
    active = row < count
    slot_valid = (slot >= 0) & (slot < SLOTS)
    valid = (count >= 0) & (count <= CAPACITY) & (state_count == count) & slot_valid & (epoch > 0)
    model_slot = tl.load(StateRows + slot * 8, slot_valid, other=0)
    valid &= (model_slot > 0) & (model_slot < tl.load(MemoryCapacity)) & (tl.load(RequestSlots + row) == slot)
    if MODE == 0:
        valid &= (operation == 1) & (source == model_slot) & (destination == model_slot)
        valid &= (raw_slot == slot + 1) & (accepted == 1) & (expected == 0)
    elif MODE == 1:
        valid &= (operation == 0) & (source > 0) & (source != destination) & (source < tl.load(MemoryCapacity)) & (destination == model_slot)
        valid &= (raw_slot == slot + 1) & (accepted == 1) & (expected > 0)
    else:
        valid &= (operation == 0) & (source == model_slot) & (destination > 0) & (source != destination) & (destination < tl.load(MemoryCapacity))
        valid &= (raw_slot == -1) & (accepted == 0) & (expected > 0)
        valid &= epoch == tl.load(Generation + slot, slot_valid, other=0)
        status = tl.load(Status + slot, slot_valid, other=0)
        valid &= (status == 1) | (status == 2)
        other_rows = tl.arange(0, SLOTS)
        duplicates = (other_rows < count) & (other_rows < CAPACITY) & (tl.load(Slots + other_rows) == slot)
        valid &= tl.sum(duplicates.to(tl.int32), 0) == 1
    error = tl.where(active & ~valid, 8, 0)
    if MODE == 2:
        consumed = tl.load(Consumed + slot * CS, slot_valid, other=-1)
        valid_endpoint = expected == consumed
        error = tl.where(active & valid & ~valid_endpoint, 9, error)
        valid &= valid_endpoint
    else:
        admission_error = tl.load(AdmissionErrors + row)
        valid &= admission_error == 0
    tl.store(FilteredRequests + row, tl.where(active & valid, slot, -1))
    tl.store(Errors + row, error)
    enabled = active & valid
    tl.store(FilteredSources + row, tl.where(enabled, source, 0))
    tl.store(FilteredDestinations + row, tl.where(enabled, destination, 0))
    tl.store(FilteredOperations + row, operation)
    tl.store(FilteredRawSlots + row, tl.where(enabled, raw_slot, -1))
    tl.store(FilteredAccepted + row, tl.where(enabled, accepted, 0))
    tl.store(FilteredExpected + row, tl.where(enabled, expected, -1))


@triton.jit
def _merge_errors(Count, StageErrors, TransferErrors, AdmissionErrors, PublicErrors, CAPACITY: tl.constexpr, ADMIT: tl.constexpr, SLOTS: tl.constexpr = 4):
    rows = tl.arange(0, SLOTS)
    active = (rows < CAPACITY) & (rows < tl.load(Count))
    staged = tl.load(StageErrors + rows)
    transferred = tl.load(TransferErrors + rows)
    error = tl.where(staged != 0, staged, transferred)
    error = tl.where(active, error, 0)
    tl.store(PublicErrors + rows, error)
    if ADMIT:
        old = tl.load(AdmissionErrors + rows)
        tl.store(AdmissionErrors + rows, old | (error << 8))


@triton.jit
def _snapshot_complete(Slots, Epochs, Count, Errors, Generation, Consumed, Generated, Status, StateError,
                       Records, Frame, Sequence, CAPACITY: tl.constexpr, CS: tl.constexpr, RECORD_FIELDS: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    matches = (rows < CAPACITY) & (rows < tl.load(Count)) & (tl.load(Slots + rows) == slot)
    active = tl.sum(matches.to(tl.int32), 0) > 0
    row = tl.min(tl.where(matches, rows, SLOTS), 0)
    epoch, consumed, generated = tl.full((), 0, tl.int64), tl.full((), 0, tl.int64), tl.full((), 0, tl.int64)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        epoch = tl.load(Epochs + row)
        error = tl.load(Errors + row)
        if epoch == tl.load(Generation + slot):
            consumed, generated = tl.load(Consumed + slot * CS), tl.load(Generated + slot)
            status, code = tl.load(Status + slot), tl.load(StateError + slot)
        if error != 0:
            status, code = 3, 64 + error
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      -1, status, code, active, RECORD_FIELDS, SLOTS)


class PreparedMaintenance:
    """Compose validated slot transitions with the completion-frame ABI.

    Modes are ``init`` (reset live hybrid state), ``restore`` (copy immutable
    state to live), and ``snapshot`` (publish the exact consumed live state).
    Admission and transfer rows use identical packed order. Native request
    allocation reserves the live slot before init/restore and the immutable
    destination before publication; published immutable destinations must not
    be reused while referenced by the prefix cache.

    Prevalidation gates each transfer before any model-state write. Transfer
    errors suppress admission and appear as error_code=64+transfer_error. A
    failed operation never changes another active generation's session state.
    """

    def __init__(self, mode: str, requests: DeviceRequestState, pool: HybridStatePool,
                 metadata: StateTransferMetadata, completion, frame_index, submission_sequence,
                 *, admission_buffers: AdmissionBuffers | None = None, forward_inputs=None,
                 qsa_states=(), capacity: int = 4, speculation=None):
        if mode not in ("init", "restore", "snapshot") or capacity not in BATCH_CAPACITIES or capacity > requests.capacity:
            raise ValueError("maintenance requires init/restore/snapshot and a graph capacity within request storage")
        self.slots = requests.capacity
        if pool.backing.shape[0] < self.slots + 2:
            raise ValueError("maintenance requires null/live slots and an immutable snapshot slot")
        device = pool.backing.device
        if device.type != "cuda" or requests.generation.device != device:
            raise ValueError("maintenance pool and request state must share a CUDA device")
        for name, dtype in (("request_slots", torch.int64), ("source_slots", torch.int64), ("destination_slots", torch.int64),
                ("operation", torch.int32), ("qsa_slots", torch.int64), ("accepted_tokens", torch.int32),
                ("expected_consumed", torch.int64), ("num_ops", torch.int32), ("errors", torch.int32)):
            _check_tensor(getattr(metadata, name), (1,) if name == "num_ops" else (self.slots,), dtype, device, name)
        records = completion.records
        record_fields = _completion_fields(records, device, self.slots)
        for tensor, shape, dtype, name in ((frame_index, (1,), torch.int32, "frame_index"), (submission_sequence, (1,), torch.int64, "submission_sequence")):
            _check_tensor(tensor, shape, dtype, device, name)
        self.mode, self.requests, self.pool, self.metadata = mode, requests, pool, metadata
        self.admission = None
        if mode == "snapshot":
            if forward_inputs is None:
                raise ValueError("snapshot maintenance requires packed decoder slot/epoch/count inputs")
            slots, epochs, count = forward_inputs.row_slot_ids, forward_inputs.row_generations, forward_inputs.live_rows
            for tensor, shape, dtype, name in ((slots, (self.slots,), torch.int32, "row_slot_ids"),
                    (epochs, (self.slots,), torch.int64, "row_generations"), (count, (1,), torch.int32, "live_rows")):
                _check_tensor(tensor, shape, dtype, device, name)
            self.admission_errors = torch.zeros(self.slots, dtype=torch.int32, device=device)
        else:
            if admission_buffers is None:
                raise ValueError("init and restore maintenance require admission staging buffers")
            self.admission = PreparedAdmission(requests, admission_buffers, completion, frame_index,
                                              submission_sequence, capacity=capacity)
            slots, epochs, count = (admission_buffers.admission_slots, admission_buffers.admission_generations,
                                    admission_buffers.admission_count)
            self.admission_errors = self.admission.errors
        self.stage_errors = torch.zeros(self.slots, dtype=torch.int32, device=device)
        self.filtered = StateTransferMetadata.allocate(self.slots, device)
        if forward_inputs is None or not hasattr(forward_inputs, "page_tables"):
            raise ValueError("exact maintenance requires physical-slot page tables")
        self.qsa_states, self.forward_inputs, self.speculation = tuple(qsa_states), forward_inputs, speculation
        self._bind_transfers()
        self._rows = torch.arange(self.slots, device=device)
        m, f, r = metadata, self.filtered, requests
        self._stage_args = (slots, epochs, count, m.source_slots, m.destination_slots, m.operation, m.qsa_slots,
            m.accepted_tokens, m.expected_consumed, m.num_ops, r.generation, r.status, r.consumed_tokens, self.admission_errors, forward_inputs.request_state_rows, m.request_slots, f.request_slots, f.source_slots, f.destination_slots, f.operation,
            f.qsa_slots, f.accepted_tokens, f.expected_consumed, f.num_ops, self.stage_errors,
            ("init", "restore", "snapshot").index(mode), capacity, pool.memory_capacity,
            r.consumed_tokens.stride(0), self.slots)
        self._merge_args = (count, self.stage_errors, f.errors, self.admission_errors, m.errors, capacity, mode != "snapshot", self.slots)
        self._complete_args = (slots, epochs, count, m.errors, r.generation, r.consumed_tokens, r.generated_tokens,
            r.status, r.error_code, records, frame_index, submission_sequence, capacity, r.consumed_tokens.stride(0), record_fields, self.slots)
        from ignition.components.state import PreparedRequestHistory

        self.history = PreparedRequestHistory(pool, requests, self.filtered, snapshot=mode == "snapshot")
        self._prepared = False

    def _bind_transfers(self):
        exact = PreparedCheckpointCaches(self.pool, self.filtered, self.qsa_states,
            self.forward_inputs.page_tables, state_rows=self.forward_inputs.request_state_rows, speculation=self.speculation)
        self.transfers = PreparedStateTransfers(self.pool, self.filtered,
            qsa_states=self.qsa_states, exact=exact)

    def rebind_pages_for_preparation(self, cache):
        self.qsa_states = tuple(cache[index] for index in range(3, 48, 4))
        self._bind_transfers()
        self.transfers.prepare()

    def staging_buffers(self) -> dict[str, torch.Tensor]:
        """Exact names registered by the native manifest for state commands."""
        return {f"state_{field}": getattr(self.metadata, field) for field in self.metadata.__dataclass_fields__}

    def prepare(self) -> None:
        if self.admission is not None:
            self.admission.prepare()
        self.transfers.prepare()
        self._stage = _compile(_stage, self._stage_args, (self.slots, 1, 1))
        self._merge = _compile(_merge_errors, self._merge_args, (1, 1, 1))
        if self.mode == "snapshot":
            self._complete = _compile(_snapshot_complete, self._complete_args, (self.slots, 1, 1))
        self._prepared = True

    def __call__(self) -> None:
        if not self._prepared:
            raise RuntimeError("maintenance prepare() must complete before capture")
        if self.admission is not None:
            self.admission.validate()
        self._stage()
        self.filtered.errors.copy_(self.stage_errors)
        if self.mode == "snapshot":
            self.history()
        self.transfers()
        self._merge()
        if self.mode != "snapshot":
            self.history()
        if self.admission is not None:
            self.admission.apply()
            if self.speculation is not None:
                for slot in range(self.slots):
                    selected = ((self._rows < self.metadata.num_ops[0])
                        & (self.filtered.request_slots == slot) & (self.metadata.errors == 0)).any()
                    self.speculation.previous_accepted[slot].copy_(torch.where(selected, 1,
                        self.speculation.previous_accepted[slot]))
        else:
            self._complete()
