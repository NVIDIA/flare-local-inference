"""Captured exact-prefix row transfers with model-owned partial KV pages."""

from collections import defaultdict
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .capacity import BATCH_CAPACITIES
from .sampling import _check_tensor, _compile
from .state import StateStorage


@dataclass(frozen=True)
class StateCommands:
    """Copy (0) or reset (1) commands for live slots and immutable snapshots.

    Copy requires one live and one immutable slot, and a positive exact consumed
    endpoint. Reset requires source -1 and endpoint zero. Errors are 1=operation,
    2=slot/direction, 3=endpoint, 4=alias conflict, 5=count, 6=page mapping.
    Native ownership must reserve snapshot destinations before publication.
    """

    request_slots: torch.Tensor
    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    operation: torch.Tensor
    expected_consumed: torch.Tensor
    num_ops: torch.Tensor
    errors: torch.Tensor

    @classmethod
    def allocate(cls, device, *, capacity=4):
        if capacity not in BATCH_CAPACITIES:
            raise ValueError("state command capacity must be a power of two through 64")
        return cls(torch.full((capacity,), -1, dtype=torch.int64, device=device),
                   torch.full((capacity,), -1, dtype=torch.int64, device=device),
                   torch.full((capacity,), -1, dtype=torch.int64, device=device),
                   torch.zeros(capacity, dtype=torch.int32, device=device),
                   torch.full((capacity,), -1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device),
                   torch.zeros(capacity, dtype=torch.int32, device=device))

    def buffers(self):
        """Fixed native upload names; validation errors are device-owned output."""
        return {f"state_{name}": getattr(self, name) for name in self.__dataclass_fields__ if name != "errors"}


@triton.jit
def _validate(Sources, Destinations, Operations, Expected, Count, Errors, Consumed, Tables, RequestSlots, StateRows, RequestCounts, RCS: tl.constexpr,
              MemoryCapacity, CS: tl.constexpr, WIDTH: tl.constexpr, PAGE_TOKENS: tl.constexpr, LIVE: tl.constexpr = 4):
    row = tl.program_id(0)
    count = tl.load(Count)
    SLOTS, PAGES = tl.load(MemoryCapacity), tl.load(MemoryCapacity + 1)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    operation, endpoint = tl.load(Operations + row), tl.load(Expected + row)
    live = row < count
    slot = tl.load(RequestSlots + row)
    valid_slot = (slot >= 0) & (slot < LIVE)
    state = tl.load(StateRows + slot * 8, valid_slot, other=0)
    valid_row = valid_slot & (state > 0) & (state < SLOTS)
    snapshot = valid_row & (source == state) & (destination > 0) & (destination < SLOTS) & (source != destination)
    restore = valid_row & (destination == state) & (source > 0) & (source < SLOTS) & (source != destination)
    reset = valid_row & (operation == 1) & (source == -1) & (destination == state)
    copying = operation == 0
    valid_slots = (copying & (snapshot | restore)) | reset
    consumed = tl.load(Consumed + source * CS, live & copying & (source > 0) & (source < SLOTS), other=-1)
    consumed = tl.where(snapshot, tl.load(RequestCounts + slot * RCS, live & snapshot, other=-1), consumed)
    valid_endpoint = tl.where(copying, (endpoint > 0) & (endpoint == consumed)
                             & (endpoint <= WIDTH * PAGE_TOKENS), endpoint == 0)
    tail = copying & (endpoint > 0) & (endpoint % PAGE_TOKENS != 0)
    logical = endpoint // PAGE_TOKENS
    page = tl.load(Tables + slot * WIDTH + logical,
                   live & tail & valid_slots & (logical >= 0) & (logical < WIDTH), other=-1)

    others = tl.arange(0, LIVE)
    other_sources, other_destinations = tl.load(Sources + others), tl.load(Destinations + others)
    other_operations, other_endpoints = tl.load(Operations + others), tl.load(Expected + others)
    other_copy = other_operations == 0
    other_slots = tl.load(RequestSlots + others)
    other_state = tl.load(StateRows + other_slots * 8, (other_slots >= 0) & (other_slots < LIVE), other=0)
    other_snapshot = other_copy & (other_sources == other_state)
    other_restore = other_copy & (other_destinations == other_state)
    other_tail = (other_snapshot | other_restore) & (other_endpoints > 0) & (other_endpoints % PAGE_TOKENS != 0)
    other_logical = other_endpoints // PAGE_TOKENS
    other_live = (others < count) & (others != row)
    other_pages = tl.load(Tables + other_slots * WIDTH + other_logical,
        other_live & other_tail & (other_slots >= 0) & (other_slots < LIVE)
        & (other_logical >= 0) & (other_logical < WIDTH), other=-1)
    conflict = destination == other_destinations
    conflict |= copying & (source == other_destinations)
    conflict |= other_copy & (destination == other_sources)
    conflict |= tail & other_tail & (page >= 0) & (page == other_pages) & (restore | other_restore)

    error = tl.full((), 0, tl.int32)
    error = tl.where(live & tail & ((page < 0) | (page >= PAGES)), 6, error)
    error = tl.where(live & (tl.sum((other_live & conflict).to(tl.int32), 0) > 0), 4, error)
    error = tl.where(live & ~valid_endpoint, 3, error)
    error = tl.where(live & ~valid_slots, 2, error)
    error = tl.where(live & ((operation < 0) | (operation > 1)), 1, error)
    error = tl.where((count < 0) | (count > LIVE), 5, error)
    tl.store(Errors + row, error)


@triton.jit
def _validate_trailing(Sources, Destinations, Operations, Expected, Count, Errors, Tables, RequestSlots, StateRows,
                       WIDTH: tl.constexpr, MemoryCapacity, PAGE_TOKENS: tl.constexpr,
                       LIVE: tl.constexpr):
    row = tl.program_id(0)
    source = tl.load(Sources + row)
    PAGES = tl.load(MemoryCapacity + 1)
    endpoint = tl.load(Expected + row)
    slot = tl.load(RequestSlots + row)
    state = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < LIVE), other=0)
    snapshot = source == state
    active = (row < tl.load(Count)) & (tl.load(Errors + row) == 0) & (tl.load(Operations + row) == 0)
    others = tl.arange(0, LIVE)
    other_source = tl.load(Sources + others)
    other_slot = tl.load(RequestSlots + others)
    other_state = tl.load(StateRows + other_slot * 8, (other_slot >= 0) & (other_slot < LIVE), other=0)
    other_snapshot = other_source == other_state
    other_endpoint = tl.load(Expected + others)
    other_active = (others < tl.load(Count)) & (others != row) & (tl.load(Operations + others) == 0)
    other_active &= (other_slot >= 0) & (other_slot < LIVE)
    error = tl.load(Errors + row)
    for part in tl.static_range(2):
        logical = endpoint // PAGE_TOKENS + part - 1
        tail = active & (endpoint > 1) & (logical >= 0)
        if part == 1:
            tail &= endpoint % PAGE_TOKENS >= 2
        bounded = (logical >= 0) & (logical < WIDTH)
        page = tl.load(Tables + slot * WIDTH + logical, tail & bounded, other=-1)
        conflict = tl.full((), False, tl.int1)
        for other_part in tl.static_range(2):
            other_logical = other_endpoint // PAGE_TOKENS + other_part - 1
            used = other_active & (other_endpoint > 1) & (other_logical >= 0) & (other_logical < WIDTH)
            if other_part == 1:
                used &= other_endpoint % PAGE_TOKENS >= 2
            other_page = tl.load(Tables + other_slot * WIDTH + other_logical, used, other=-1)
            conflict |= tl.sum((used & (page == other_page) & (~snapshot | ~other_snapshot)).to(tl.int32), 0) > 0
        error = tl.where(tail & ((page < 0) | (page >= PAGES)), 6, error)
        error = tl.where(tail & conflict, 4, error)
    tl.store(Errors + row, error)


@triton.jit
def _copy_rows(Backing, Sources, Destinations, Operations, Count, Errors,
               ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr, EXCLUDED: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    copying = tl.load(Operations + row) == 0
    active = (row < tl.load(Count)) & (tl.load(Errors + row) == 0)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    mask = active & (index < ROW_BYTES)
    for span in tl.static_range(len(EXCLUDED)):
        mask &= (index < EXCLUDED[span][0]) | (index >= EXCLUDED[span][1])
    value = tl.load(Backing + source * ROW_BYTES + index, mask & copying, other=0)
    tl.store(Backing + destination * ROW_BYTES + index, value, mask)


@triton.jit
def _copy_pages(Pointers, Backing, Tables, Sources, Destinations, Operations, Expected, Count, Errors, RequestSlots, StateRows,
                ROW_BYTES: tl.constexpr, PAGE_BYTES: tl.constexpr, WIDTH: tl.constexpr,
                PAGE_TOKENS: tl.constexpr, BLOCK: tl.constexpr, LIVE: tl.constexpr = 4,
                DRAFT_PART: tl.constexpr = -1):
    row, region, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    source, destination = tl.load(Sources + row), tl.load(Destinations + row)
    endpoint = tl.load(Expected + row)
    slot = tl.load(RequestSlots + row)
    state = tl.load(StateRows + slot * 8, (slot >= 0) & (slot < LIVE), other=0)
    snapshot = source == state
    active = (row < tl.load(Count)) & (tl.load(Errors + row) == 0)
    active &= tl.load(Operations + row) == 0
    logical = endpoint // PAGE_TOKENS
    if DRAFT_PART < 0:
        active &= endpoint % PAGE_TOKENS != 0
    else:
        logical += DRAFT_PART - 1
        active &= (endpoint > 1) & (logical >= 0)
        if DRAFT_PART == 1:
            active &= endpoint % PAGE_TOKENS >= 2
    page = tl.load(Tables + slot * WIDTH + logical, active, other=0)
    cache = tl.load(Pointers + region * 2).to(tl.pointer_type(tl.uint8))
    offset = tl.load(Pointers + region * 2 + 1)
    if DRAFT_PART >= 0:
        offset += DRAFT_PART * PAGE_BYTES
    saved_slot = tl.where(snapshot, destination, source)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    saved = Backing + saved_slot * ROW_BYTES + offset + index
    external = cache + page.to(tl.int64) * PAGE_BYTES + index
    if snapshot:
        value = tl.load(external, active & (index < PAGE_BYTES), other=0)
        tl.store(saved, value, active & (index < PAGE_BYTES))
    else:
        value = tl.load(saved, active & (index < PAGE_BYTES), other=0)
        tl.store(external, value, active & (index < PAGE_BYTES))


class PreparedStateTransfer:
    """Validate then copy complete rows and each model's partial physical page.

    ``pages`` contains (state region name, contiguous uint8 [pages, page_bytes]
    cache) pairs. Cache owners and pointer tables remain alive until graph drain.
    ``trailing_pages`` uses a one-page-shifted table with a one-token lag. Each
    saved region holds two pages: the preceding boundary page, then the current
    one. Both may be private even when the preceding draft page is complete.
    These maintenance operations are captured directly. Tensor-parallel callers
    must combine validation failures across ranks between validate() and apply().
    A nonzero command error suppresses both row and external-page writes.
    """

    def __init__(self, pool: StateStorage, commands: StateCommands, tables, *, pages, page_tokens,
                 state_rows, requests, trailing_pages=(), trailing_tables=None, excluded_regions=()):
        self.capacity = commands.source_slots.numel()
        device = pool.backing.device
        if device.type != "cuda" or pool.backing.shape[0] < self.capacity + 2:
            raise ValueError("state transfers require CUDA null/live/snapshot storage")
        if type(page_tokens) is not int or page_tokens <= 0:
            raise ValueError("state transfers require positive physical page size")
        for name in commands.__dataclass_fields__:
            dtype = torch.int32 if name in ("operation", "num_ops", "errors") else torch.int64
            _check_tensor(getattr(commands, name), (1,) if name == "num_ops" else (self.capacity,), dtype, device, name)
        if tables.ndim != 2 or tables.shape[0] != self.capacity or tables.shape[1] <= 0:
            raise ValueError("state transfers require one page table per physical request slot")
        _check_tensor(tables, tables.shape, torch.int32, device, "page_tables")
        _check_tensor(state_rows, (self.capacity, 8), torch.int64, device, "request_state_rows")
        if requests.capacity != self.capacity:
            raise ValueError("request state and transfer command capacities must match")
        consumed = pool.region("consumed_tokens")
        if consumed.shape != (pool.backing.shape[0], 1) or consumed.dtype != torch.int64:
            raise ValueError("state rows require one int64 consumed endpoint")
        self.state_rows = state_rows
        self.pool, self.commands, self.tables, self.pages = pool, commands, tables, tuple(pages)
        groups, page_count = defaultdict(list), None
        names = set()
        for name, cache in self.pages:
            region = pool.layout.region(name)
            if name in names or region.dtype != torch.uint8 or cache.ndim != 2 or cache.shape[1] != region.nbytes:
                raise ValueError("partial-page regions must be distinct uint8 fields matching their cache records")
            names.add(name)
            _check_tensor(cache, cache.shape, torch.uint8, device, name)
            if cache.shape[0] <= 0 or (page_count is not None and page_count != cache.shape[0]):
                raise ValueError("partial-page caches must share positive physical page capacity")
            page_count = cache.shape[0]
            groups[region.nbytes].append((cache.data_ptr(), region.offset))
        if page_count is None:
            raise ValueError("state transfer requires physical KV page bindings")
        trailing_pages = tuple(trailing_pages)
        self.trailing_tables = trailing_tables
        trailing_groups = defaultdict(list)
        if trailing_pages:
            _check_tensor(trailing_tables, tables.shape, torch.int32, device, "trailing_page_tables")
            for name, cache in trailing_pages:
                region = pool.layout.region(name)
                if name in names or region.dtype != torch.uint8 or cache.shape != (page_count, region.nbytes // 2) or region.nbytes % 2:
                    raise ValueError("shifted draft pages require two saved pages per cache plane")
                _check_tensor(cache, cache.shape, torch.uint8, device, name)
                names.add(name)
                trailing_groups[cache.shape[1]].append((cache.data_ptr(), region.offset))
        self.trailing_pointers = {size: torch.tensor(values, dtype=torch.int64, device=device)
                                 for size, values in trailing_groups.items()}
        self.pointers = {size: torch.tensor(values, dtype=torch.int64, device=device) for size, values in groups.items()}
        c = commands
        self.validate_args = (c.source_slots, c.destination_slots, c.operation, c.expected_consumed, c.num_ops,
            c.errors, consumed, tables, c.request_slots, state_rows, requests.consumed_tokens, requests.consumed_tokens.stride(0), pool.memory_capacity, consumed.stride(0), tables.shape[1], page_tokens, self.capacity)
        self.row_args = (pool.backing, c.source_slots, c.destination_slots, c.operation, c.num_ops, c.errors,
                         pool.layout.row_bytes, 4096,
                         tuple((region.offset, region.offset + region.nbytes)
                               for name in excluded_regions for region in (pool.layout.region(name),)))
        self.page_args = [(pointer, pool.backing, tables, c.source_slots, c.destination_slots, c.operation,
            c.expected_consumed, c.num_ops, c.errors, c.request_slots, state_rows, pool.layout.row_bytes, size, tables.shape[1], page_tokens, 1024, self.capacity)
            for size, pointer in self.pointers.items()]
        self.page_args.extend((pointer, pool.backing, trailing_tables, c.source_slots, c.destination_slots, c.operation,
            c.expected_consumed, c.num_ops, c.errors, c.request_slots, state_rows, pool.layout.row_bytes, size, tables.shape[1], page_tokens,
            1024, self.capacity, part) for size, pointer in self.trailing_pointers.items() for part in (0, 1))
        self.trailing_validation_args = None if not trailing_pages else (
            c.source_slots, c.destination_slots, c.operation, c.expected_consumed, c.num_ops, c.errors,
            trailing_tables, c.request_slots, state_rows, tables.shape[1], pool.memory_capacity, page_tokens, self.capacity)
        self._trailing_validate = None
        self._validate, self._copies = None, []

    def prepare(self):
        if self.commands.num_ops.item() != 0:
            raise ValueError("state transfer preparation requires zero live commands")
        self._validate = _compile(_validate, self.validate_args, (self.capacity, 1, 1))
        if self.trailing_validation_args is not None:
            self._trailing_validate = _compile(_validate_trailing, self.trailing_validation_args, (self.capacity, 1, 1))
        self._copies = [_compile(_copy_rows, self.row_args, (self.capacity, triton.cdiv(self.pool.layout.row_bytes, 4096), 1))]
        for args in self.page_args:
            self._copies.append(_compile(_copy_pages, args, (self.capacity, args[0].shape[0], triton.cdiv(args[12], 1024))))

    def validate(self):
        if self._validate is None:
            raise RuntimeError("state transfer prepare() must complete before capture")
        self._validate()
        if self._trailing_validate is not None:
            self._trailing_validate()

    def apply(self):
        if self._validate is None:
            raise RuntimeError("state transfer prepare() must complete before capture")
        for operation in self._copies:
            operation()
