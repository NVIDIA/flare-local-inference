"""Exact endpoint cache copies and accepted-state normalization."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import triton
from triton.runtime.interpreter import InterpretedFunction

from ignition.recipes.qwen38 import state as module

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device")


def launch(kernel, grid, args, device):
    if device == "cpu":
        InterpretedFunction(kernel.fn)[grid](*args)
    else:
        kernel[grid](*args, num_warps=4)


def endpoint_case(endpoint, device):
    """Use real cache page shapes with private restored pages and retained shared pages."""
    from b12x.attention import qsa
    geometry = qsa.cache_requirements(main_page_size=64, position_axes=3, max_speculative_tokens=7,
                                      kv_dtype=torch.float8_e4m3fn)
    ring = geometry.raw_ring_capacity
    fields = (("raw_keys", geometry.raw_k_ring_shape, torch.bfloat16),
              ("raw_positions", (ring,), torch.int64),
              ("raw_rope", (ring, 3), torch.int64), ("raw_anchor", (1,), torch.int64))
    row_bytes = 2**19
    backing = torch.full((6, row_bytes), 0xA5, dtype=torch.uint8, device=device)
    sources = torch.tensor([3], dtype=torch.int64, device=device)
    destinations = torch.tensor([5], dtype=torch.int64, device=device)
    operations = torch.zeros(1, dtype=torch.int32, device=device)
    expected = torch.tensor([endpoint], dtype=torch.int64, device=device)
    count = torch.ones(1, dtype=torch.int32, device=device)
    errors = torch.zeros(1, dtype=torch.int32, device=device)
    requests = torch.tensor([2], dtype=torch.int64, device=device)
    raw_slots = torch.tensor([-1], dtype=torch.int64, device=device)
    all_args, outputs, owned = [], [], []
    offset = 0
    for draft in (False, True):
        consumed = max(endpoint - int(draft), 0)
        # Includes rejected future positions and valid preceding positions in their actual ring slots.
        positions = torch.arange(consumed - 5, consumed - 5 + ring, device=device)
        tags = torch.full((5, ring), -1, dtype=torch.int64, device=device)
        tags[3, positions.remainder(ring)] = positions
        for field, (_, shape, dtype) in enumerate(fields):
            size = torch.tensor(shape).prod().item()
            if field == 1:
                value = tags
            elif field == 3:
                value = torch.full((5,), 999, dtype=dtype, device=device)
            else:
                value = torch.arange(5 * size, device=device).reshape(5, *shape).to(dtype)
            pointers = torch.tensor([[value.data_ptr(), offset, tags.data_ptr()]], dtype=torch.int64, device=device)
            args = (pointers, backing, sources, destinations, operations, expected, count, errors, requests, raw_slots,
                    row_bytes, size, ring, field, draft, triton.next_power_of_2(size))
            all_args.append((module._checkpoint_raw, (1, 1, 1), args))
            valid = (tags[3] >= 0) & (tags[3] < consumed)
            if field == 3:
                wanted = torch.tensor([consumed - 1], dtype=dtype, device=device)
            else:
                expand = valid.reshape((ring,) + (1,) * (len(shape) - 1))
                wanted = torch.where(expand, value[3], 0 if field == 0 else -1)
            outputs.append((value, wanted.clone(), offset, size, dtype))
            owned.extend((value, pointers, tags))
            offset += (size * dtype.itemsize + 255) // 256 * 256
    page_expectations = []
    for part in (-1, 0, 1):
        logical = endpoint // 64 if part < 0 else endpoint // 64 + part - 1
        required = (endpoint % 64 != 0 if part < 0 else
                    endpoint > 1 and logical >= 0 and (part == 0 or endpoint % 64 >= 2))
        tables = torch.full((4, 4), -1, dtype=torch.int32, device=device)
        # Snapshot physical slot2; restore physical slot1, distinct private pages.
        if required:
            tables[2, logical], tables[1, logical] = 1, 3
        for shape, dtype in ((geometry.main_k_page_shape, geometry.kv_dtype),
                             (geometry.main_v_page_shape, geometry.kv_dtype),
                             (geometry.compressed_page_shape, torch.bfloat16)):
            size = torch.tensor(shape).prod().item() * dtype.itemsize
            cache = torch.arange(6, device=device).to(dtype).view(6, *([1] * len(shape))).expand(6, *shape).contiguous()
            pointers = torch.tensor([[cache.data_ptr(), offset]], dtype=torch.int64, device=device)
            args = (pointers, backing, tables, sources, destinations, operations, expected, count, errors, requests, raw_slots,
                    row_bytes, size, 4, part, 1024)
            all_args.append((module._checkpoint_pages, (1, 1, triton.cdiv(size, 1024)), args))
            page_expectations.append((cache, required, offset, size))
            owned.extend((cache, tables, pointers))
            offset += (size + 255) // 256 * 256
    assert offset <= row_bytes

    def invoke():
        for kernel, grid, args in all_args:
            launch(kernel, grid, args, device)

    invoke()
    for _, wanted, start, size, dtype in outputs:
        actual = backing[5, start:start + size * dtype.itemsize].view(dtype).view_as(wanted)
        assert torch.equal(actual, wanted)
    for cache, required, start, size in page_expectations:
        saved = backing[5, start:start + size]
        assert saved.view(cache.dtype).float().eq(1).all() if required else saved.eq(0xA5).all()
    immutable = backing[5].clone()
    for value, _, _, _, _ in outputs:
        value[2].fill_(77)
    sources.fill_(5); destinations.fill_(2)
    requests.fill_(1)
    raw_slots.fill_(0)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        graph.reset()
    else:
        invoke()
    assert torch.equal(backing[5], immutable)
    for value, wanted, *_ in outputs:
        assert torch.equal(value[2].reshape_as(wanted), wanted)
    for cache, required, *_ in page_expectations:
        cache = cache.float()
        assert cache[3].eq(1 if required else 3).all()
        assert cache[0].eq(0).all() and cache[1].eq(1).all() and cache[2].eq(2).all()
        assert cache[4].eq(4).all() and cache[5].eq(5).all()
    # Failed metadata must suppress every auxiliary cache mutation.
    errors.fill_(8)
    before = [value.clone() for value, *_ in outputs]
    invoke()
    assert all(torch.equal(value, original) for (value, *_), original in zip(outputs, before, strict=True))
    # Zero-consumed initialization invalidates both raw intervals without touching pages.
    errors.zero_(); operations.fill_(1); expected.zero_(); sources.fill_(2)
    before_pages = [cache.clone() for cache, *_ in page_expectations]
    invoke()
    for index, (value, _, _, _, _) in enumerate(outputs):
        assert value[2].eq(0 if index % 4 == 0 else -1).all()
    assert all(torch.equal(cache.view(torch.uint8), prior.view(torch.uint8))
               for (cache, *_), prior in zip(page_expectations, before_pages, strict=True))
    assert torch.equal(backing[5], immutable)


@pytest.mark.parametrize("endpoint", [1, 2, 3, 15, 63, 64, 65, 66, 67, 127, 128])
def test_exact_cache_endpoint_cpu(endpoint, monkeypatch):
    def unexpected_cuda_access():
        pytest.fail("CPU cache checks must not access CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected_cuda_access)
    endpoint_case(endpoint, "cpu")


@CUDA
@pytest.mark.parametrize("endpoint", [1, 2, 3, 15, 63, 64, 65, 66, 67, 127, 128])
def test_exact_cache_endpoint_cuda_graph(endpoint):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    endpoint_case(endpoint, "cuda")


def transfer_case(accepted, device):
    # Whole aligned tiles exercise the exact byte transfer and normalization kernel.
    block, row_bytes, slots = 4096, 16384, 34
    backing = torch.arange(slots * row_bytes, device=device).remainder(251).to(torch.uint8).view(slots, row_bytes)
    layout = torch.tensor([[1, 0, 0, 0], [2, block, 3, 10], [0, 0, 0, 0], [0, 0, 0, 0]], device=device, dtype=torch.int64)
    # Two full convolution channels cannot fit a 4096 tile with capacity10; a
    # production convolution spans many tiles, and the classifier uses region offsets.
    consumed = backing[:, 12288:12296].view(torch.int64)
    consumed[3] = 67
    previous = torch.tensor([1, 1, accepted, 1], dtype=torch.int32, device=device)
    source = torch.tensor([3], device=device, dtype=torch.int64)
    destination = torch.tensor([5], device=device, dtype=torch.int64)
    operation = torch.zeros(1, device=device, dtype=torch.int32)
    raw = torch.tensor([-1], device=device, dtype=torch.int64)
    count = torch.ones(1, device=device, dtype=torch.int32)
    expected = torch.tensor([67], device=device, dtype=torch.int64)
    errors = torch.zeros(1, device=device, dtype=torch.int32)
    tables = torch.tensor([[0, 1, 2, 3]] * 4, device=device, dtype=torch.int32)
    requests = torch.tensor([2], dtype=torch.int64, device=device)
    states = torch.zeros((4, 8), dtype=torch.int64, device=device)
    states[2, 0] = 3
    states[2, 1:] = torch.arange(20, 27, device=device)
    capacity = torch.tensor([slots, 6], dtype=torch.int64, device=device)
    args = (backing, consumed, source, destination, operation, raw, torch.ones_like(operation), count, errors,
            expected, previous, layout, tables, tables, requests, states, True, True, True, capacity, 4,
            row_bytes, 1, 5, 12304, consumed.stride(0), block, 1)
    original = backing.clone()
    launch(module._transfer, (1, 4, 1), args, device)
    assert errors.item() == 0
    # Recurrent state is committed into the live slot each round; snapshots copy it as is.
    assert torch.equal(backing[5, :block], original[3, :block])
    relative = torch.arange(block, device=device)
    keep = ((relative // 2) % 10) < 3
    expected_conv = torch.where(keep, original[3, block + relative + (accepted - 1) * 2], 0)
    assert torch.equal(backing[5, block:2 * block], expected_conv)
    assert torch.equal(backing[5, 2 * block:], original[3, 2 * block:])
    assert torch.equal(backing[:5], original[:5])
    frozen = backing.clone()
    for invalid in (-1, 66, 257):
        expected.fill_(invalid)
        launch(module._transfer, (1, 4, 1), args, device)
        assert errors.item() != 0 and torch.equal(backing, frozen)
    expected.fill_(67); tables[2, 1] = -1
    launch(module._transfer, (1, 4, 1), args, device)
    assert errors.item() == 8 and torch.equal(backing, frozen)


@pytest.mark.parametrize("accepted", [1, 2, 5, 8])
def test_committed_recurrence_and_selected_convolution_cpu(accepted):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(accepted)],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES=""),
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@CUDA
@pytest.mark.parametrize("accepted", [1, 2, 5, 8])
def test_committed_recurrence_and_selected_convolution_cuda(accepted):
    transfer_case(accepted, "cuda")


if __name__ == "__main__":
    transfer_case(int(sys.argv[1]), "cpu")
    assert not torch.cuda.is_initialized()


def test_admission_failure_does_not_reset_other_speculative_state():
    from ignition.recipes.qwen38.maintenance import PreparedMaintenance

    previous = torch.tensor([8, 7, 6, 5], dtype=torch.int32)
    generations = torch.tensor([11, 12, 13, 14], dtype=torch.int64)
    operation = PreparedMaintenance.__new__(PreparedMaintenance)
    operation._prepared = True
    operation.admission = SimpleNamespace(validate=lambda: None, apply=lambda: None)
    operation._stage = operation.transfers = operation._merge = lambda: None
    operation.metadata = SimpleNamespace(num_ops=torch.tensor([2]),
        destination_slots=torch.tensor([1, 3, 0, 0]), errors=torch.zeros(4, dtype=torch.int32))
    # Admission prevalidation rejected lane0 despite a syntactically valid destination.
    operation.filtered = SimpleNamespace(destination_slots=torch.tensor([0, 3, 0, 0]),
        request_slots=torch.tensor([-1, 2, -1, -1]), errors=torch.zeros(4, dtype=torch.int32))
    operation.stage_errors = torch.zeros(4, dtype=torch.int32)
    operation.mode, operation.history = "init", lambda: None
    operation.requests = SimpleNamespace(generation=generations)
    operation.speculation = SimpleNamespace(previous_accepted=previous)
    operation.slots = 4
    operation._rows = torch.arange(4)
    operation()
    assert previous.tolist() == [8, 7, 1, 5]
    previous[2] = 6
    operation.metadata.errors[1] = 8
    operation()
    assert previous.tolist() == [8, 7, 6, 5]


def test_actual_layout_normalization_map_covers_every_history_region():
    for speculative in (0, 7):
        layout = module.HybridStateLayout.build(speculative_tokens=speculative)
        tiles = module.checkpoint_copy_layout(layout)
        for region in layout.regions:
            # Committed recurrent state needs no normalization; histories shift by the accepted count.
            kind = 2 if region.name.endswith(".convolution") or region.name == "ple_convolution" else 0
            begin, end = region.offset // 4096, (region.offset + region.nbytes + 4095) // 4096
            assert tiles[begin:end, 0].eq(kind).all(), region.name
            if kind == 2:
                assert tiles[begin:end, 2].eq(9 if region.name == "ple_convolution" else 3).all()
                assert tiles[begin:end, 3].eq(region.shape[-1]).all()
