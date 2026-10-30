# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation storage preserves GPU capabilities and mapped mailbox publication."""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_export_keeps_host_mailbox_and_omits_unused_library_storage(tmp_path):
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.package_file import read_index

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "mailbox"
    result = subprocess.run([sys.executable, str(root / "helios/tests/engram_package_fixture.py"), str(output)],
                            cwd=root, timeout=180, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VRAM inventory at export" in result.stderr
    assert "Rank 0 / CUDA 0" in result.stderr
    assert "Graph I/O / persistent state" in result.stderr
    assert "not resident VRAM" in result.stderr
    index, _ = read_index(output / "package")
    allocations = index.ranks[0].allocations
    host, = (allocation for allocation in allocations if allocation.memory == proto.MAPPED_HOST)
    assert host.address == PackageStorage.HOST_BASE
    assert all(allocation.address >= PackageStorage.EXTERNAL_FLOOR
               for allocation in allocations if allocation.memory == proto.DEVICE)
    assert all(not allocation.name.startswith("external-") for allocation in allocations)
    assert all(allocation.name == host.name or allocation.address + allocation.nbytes <= host.address
               or allocation.address >= host.address + host.nbytes for allocation in allocations)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_package_weights_support_gpudirect_registration():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    subprocess.run([sys.executable, "-c", """
import ctypes as C
import torch
from ignition.preparation.package_storage import PackageStorage

torch.empty(1, device="cuda:0")
with PackageStorage(device=0) as storage:
    weight = torch.empty(4 << 20, dtype=torch.uint8, device="cuda:0")
    supported, capable = C.c_int(), C.c_uint()
    storage.cuda.call("cuDeviceGetAttribute", C.byref(supported), C.c_int(116), C.c_int(0))
    storage.cuda.call("cuPointerGetAttribute", C.byref(capable), C.c_int(15), C.c_uint64(weight.data_ptr()))
    assert bool(capable.value) == bool(supported.value), (capable.value, supported.value)
"""], check=True, timeout=60)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_projection_fixture_pools_release_after_trial_owners(mailbox_storage):
    import gc
    import torch
    from ignition.preparation.fixtures.projection import input_factory
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(0)
    storage = mailbox_storage
    with RecordedPreparationSession(device=0, checkpoint="fixtures",
            workload_schema="projection-fixtures", numerical_policy="default", benchmark_storage=storage) as session:
        session._tune_status = "tuning"
        source = torch.zeros((65536, 1024), dtype=torch.bfloat16, device="cuda:0")
        output = torch.empty((65536, 4608), dtype=source.dtype, device=source.device)
        baseline = torch.cuda.memory_allocated()
        pools = storage._preparation_pools.copy()
        acquire = input_factory(source, output, storage=session.benchmark_storage)
        first = acquire()
        second = acquire()
        assert first is second
        first.produce()
        assert torch.equal(first.source, first.seed)
        del first
        gc.collect()
        assert torch.cuda.memory_allocated() > baseline
        assert second.source.data_ptr() != source.data_ptr()
        second.produce()
        del second
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() == baseline
        fixture_pools = storage._preparation_pools - pools
        retained = [segment for segment in torch.cuda.memory_snapshot()
                    if tuple(segment["segment_pool_id"]) in fixture_pools]
        assert not retained, [(segment["segment_pool_id"], segment["total_size"]) for segment in retained]


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_attention_fixture_releases_before_projection_tuning(mailbox_storage):
    import gc
    import weakref
    import torch
    from ignition.preparation.fixtures import projection, varlen_attention

    storage = mailbox_storage
    source = torch.zeros((65536, 12, 128), dtype=torch.bfloat16, device="cuda:0")
    arguments = dict(q=source, k=source, v=source,
        cu_seqlens_q=torch.zeros(2, dtype=torch.int32, device=source.device),
        max_seqlen_q=65536, max_seqlen_k=65536)
    acquire = varlen_attention.input_factory(arguments, storage=storage)
    baseline = torch.cuda.memory_allocated()
    first, second = acquire(), acquire()
    assert first is second
    reference = weakref.ref(first)
    assert first.nbytes == 6 * source.nbytes + 8
    first.q.fill_(float("nan"))
    second.produce()
    assert torch.equal(second.q, second.seed_q)
    del first, second
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert reference() is None
    assert torch.cuda.memory_allocated() == baseline
    operands = projection.input_factory(source[:, 0], source[:, 0], storage=storage)()
    operands.produce()
    assert torch.isfinite(operands.source).all()
    assert not source.count_nonzero()


@pytest.fixture(scope="module")
def mailbox_storage():
    import torch
    from ignition.preparation.package_storage import PackageStorage

    torch.cuda.set_device(0)
    with PackageStorage(device=0) as storage:
        assert storage.HOST_BASE == storage._snapshot()["host_base"] == PackageStorage.HOST_BASE
        yield storage


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_retired_loading_pools_release_staging_and_preserve_graph_reachability(mailbox_storage, tmp_path):
    import ctypes as C
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.runtime_package import Allocation, RuntimePackage

    storage = mailbox_storage
    graph_pool = storage.graph_pool.id
    pools = storage._preparation_pools.copy()
    weights, retired = [], []
    for value in range(3):
        with storage.transient():
            # Each weight occupies a complete segment, so staging can retire independently.
            weights.append(torch.full((4 << 20,), value + 1., dtype=torch.float32, device="cuda"))
            staging = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
            retired.append(StorageWeakRef(staging.untyped_storage()))
            del staging
    assert all(reference.expired() for reference in retired)
    loading_pools = storage._preparation_pools - pools
    retained = [segment for segment in torch.cuda.memory_snapshot()
                if tuple(segment["segment_pool_id"]) in loading_pools]
    assert sum(segment["total_size"] for segment in retained) <= sum(t.nbytes for t in weights) + 3 * storage.granularity
    assert storage.graph_pool.id == graph_pool and graph_pool not in storage._preparation_pools

    with ModuleImages() as images:
        output = torch.empty_like(weights[-1])
        output.add_(0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
            storage.cuda.call("cuMemcpyDtoDAsync_v2", C.c_uint64(output.data_ptr()), C.c_uint64(weights[-1].data_ptr()),
                              C.c_size_t(output.nbytes), C.c_void_p(stream.cuda_stream))
            output.add_(0)
        graph.instantiate()
        mappings = storage.mappings
        allocations = tuple(Allocation(f"device-{i}", m.address, m.nbytes, preserve_address=True)
                            for i, m in enumerate(mappings.values()))
        package = RuntimePackage(allocations=allocations, modules={}, kernels=(), recorder=images.native,
                                 owners=(weights, output, storage, graph))
        package.write(tmp_path / "rank.json", {"weight-copy": graph}, sources=True)
        for tensor in (weights[-1], output):
            assert any(a.address <= tensor.data_ptr() < a.address + a.nbytes for a in package.allocations)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, weights[-1], rtol=0, atol=0)
        graph.reset()



@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_scratch_backing_released_for_compilation_and_restored():
    subprocess.run([sys.executable, "-c", """
import torch
from ignition.preparation.aot import aot_forward
from ignition.preparation.forward_metadata import dumps, loads
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.scratch import GraphBufferPool, ScratchArena

class Forward:
    def __init__(self, source, scratch):
        self.source, self.scratch = source, scratch
        self.output = torch.empty_like(source)

    def __call__(self):
        self.scratch.copy_(self.source.square())
        self.output.copy_(self.scratch + self.source)

torch.cuda.set_device(0)
with PackageStorage(device=0) as storage:
    source = torch.randn(256, 256, device='cuda')
    pool = storage.scratch_pool()
    buffers = GraphBufferPool('cuda', chunk_bytes=64 << 20, allocation_pool=pool.pool)
    cursor = buffers.cursor()
    scratch = cursor.zeros(source.shape, torch.float32)
    arena = ScratchArena('cuda', allocation_pool=pool.pool)
    arena.reserve(128 << 20)
    late = cursor.full((128 << 20,), 7, torch.uint8)
    arena.reserve(256 << 20)
    arena.trim(128 << 20)
    arena.tensor.fill_(1)
    tensors = (scratch, late, arena.tensor)
    forward = Forward(source, scratch)
    forward.buffers, forward.arena = buffers, arena
    metadata = loads(dumps(forward))
    assert metadata.scratch.shape == scratch.shape
    buffers.allocation_pool = arena.allocation_pool = None
    torch.cuda.synchronize()
    cached = torch.cuda.mem_get_info()[0]
    pool.retire()
    assert torch.cuda.mem_get_info()[0] >= cached + (256 << 20)
    addresses = tuple(t.data_ptr() for t in tensors)
    torch.cuda.synchronize()
    before = torch.cuda.mem_get_info()[0]
    pool.release_backing()
    assert torch.cuda.mem_get_info()[0] >= before + buffers.nbytes + arena.nbytes
    try:
        compiled = aot_forward(forward)
    finally:
        pool.materialize()
    assert addresses == tuple(t.data_ptr() for t in tensors)
    assert all(not t.any() for t in tensors)
    compiled(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(forward)
    allocated = torch.cuda.memory_allocated()
    for value in (3., 7., -2.):
        source.fill_(value)
        graph.replay()
        torch.testing.assert_close(forward.output, torch.full_like(source, value * value + value))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert storage.error is None
    graph.reset()
"""], check=True, timeout=120)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_package_allocator_oom_allows_retry():
    subprocess.run([sys.executable, "-c", """
import ctypes as C
import torch
from ignition.preparation.package_storage import PackageStorage

torch.cuda.set_device(0)
with PackageStorage(device=0) as storage:
    allocate = storage.library.flare_package_allocate
    allocate.argtypes = [C.c_size_t, C.c_int, C.c_void_p]
    allocate.restype = C.c_void_p
    total = torch.cuda.mem_get_info()[1]
    assert allocate(total + storage.granularity, 0, None) is None
    assert storage.error is None
    value = torch.full((1 << 20,), 37, dtype=torch.uint8, device='cuda')
    assert torch.all(value == 37)
    assert storage.error is None
"""], check=True, timeout=60)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_capture_releases_warmup_pools():
    subprocess.run([sys.executable, "-c", """
from contextlib import nullcontext
import torch
from ignition.preparation.capture import capture_forwards
from ignition.preparation.package_storage import PackageStorage

class Session:
    def capture(self):
        return nullcontext()

torch.cuda.set_device(0)
with PackageStorage(device=0) as storage:
    source = torch.ones(32 << 20, dtype=torch.float32, device='cuda')
    output = torch.empty(1, device='cuda')
    def forward():
        output.copy_(source.square().sum())
    existing = storage._preparation_pools.copy()
    graphs, _ = capture_forwards({'sum': forward}, {}, session=Session(), device=source.device,
        stream=torch.cuda.Stream(), pool=storage.graph_pool.id, compile_decoder=False,
        warmup_storage=storage)
    warmup_pools = storage._preparation_pools - existing
    assert warmup_pools
    assert not [segment for segment in torch.cuda.memory_snapshot()
                if tuple(segment['segment_pool_id']) in warmup_pools]
    allocated = torch.cuda.memory_allocated()
    source.fill_(2)
    graphs['sum'].replay()
    torch.cuda.synchronize()
    assert output.item() == source.numel() * 4
    assert torch.cuda.memory_allocated() == allocated
    assert storage.error is None
    graphs['sum'].reset()
"""], check=True, timeout=60)


class _WarmupForward:
    def __init__(self, rows):
        import torch

        self.source = torch.randn(rows, 1024, device="cuda")
        self.output = torch.empty_like(self.source)

    def __call__(self):
        self.output.copy_(self.source.square().sin())


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_compiled_variants_release_warmup_storage(tmp_path, mailbox_storage):
    from contextlib import contextmanager
    import torch
    from ignition.preparation.aot import ForwardCache
    from ignition.preparation.execution import GraphSignature

    storage = mailbox_storage
    functions = {"prefill4096": _WarmupForward(4096), "prefill1024": _WarmupForward(1024)}
    signatures = {name: GraphSignature("prefill", 1, forward.source.shape[0])
                  for name, forward in functions.items()}
    warmed = []

    @contextmanager
    def warmup():
        previous = storage._preparation_pools.copy()
        with storage.transient():
            temporary = torch.empty(4 << 20, dtype=torch.uint8, device="cuda")
            temporary.zero_()
            del temporary
            yield
        pools = storage._preparation_pools - previous
        assert pools
        assert not [segment for segment in torch.cuda.memory_snapshot()
                    if tuple(segment["segment_pool_id"]) in pools]
        assert storage._snapshot()["retained_handles"] > 0
        warmed.append(pools)

    cache = ForwardCache(namespace={"model": "warmup-storage"}, device="cuda:0", root=tmp_path)
    with storage.retain_transient_backing():
        compiled = cache.prepare_many(functions, signatures, materialize=lambda: None, warmup=warmup)
    assert storage._snapshot()["retained_handles"] == 0
    assert storage._snapshot()["retained_reference_bytes"] == 0
    assert len(warmed) == 2 * len(functions)
    graph = torch.cuda.CUDAGraph()
    for name, forward in functions.items():
        with torch.cuda.graph(graph, pool=storage.graph_pool.id):
            compiled[name]()
        allocated = torch.cuda.memory_allocated()
        forward.source.normal_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(forward.output, forward.source.square().sin())
        graph.reset()
    assert storage.error is None


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_transient_retention_reuses_retained_physical_backing(mailbox_storage):
    import torch

    storage = mailbox_storage

    segment_size = 16 << 20
    sizes = [segment_size] * 12
    reusable_address = None
    with storage.retain_transient_backing():
        for index, size in enumerate(sizes):
            with storage.transient():
                temporary = torch.empty(size, dtype=torch.uint8, device="cuda")
                mapping_extent = storage.mappings[temporary.data_ptr()].nbytes
                assert mapping_extent >= size
                if reusable_address is None:
                    reusable_address = temporary.data_ptr()
                    reusable_extent = mapping_extent
                else:
                    assert temporary.data_ptr() == reusable_address
                    assert mapping_extent == reusable_extent
                del temporary

            snapshot = storage._snapshot()
            assert snapshot["retained_handles"] == index + 1
            assert snapshot.get("retained_reusable_handles", 0) == 1
            assert snapshot.get("retained_reusable_bytes", 0) == reusable_extent
    snapshot = storage._snapshot()
    assert snapshot["retained_handles"] == 0
    assert snapshot["retained_reference_bytes"] == 0
    assert reusable_address not in storage.mappings
    assert storage.error is None


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_transient_mapping_inventory_records_full_pooled_extent(mailbox_storage):
    import ctypes as C
    import torch

    storage = mailbox_storage
    allocate = storage.library.flare_package_transient_allocate
    allocate.argtypes = [C.c_size_t, C.c_int, C.c_void_p]
    allocate.restype = C.c_void_p
    transient_free = storage.library.flare_package_transient_free
    transient_free.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
    transient_free.restype = None

    segment_size = 16 << 20
    smaller_size = segment_size // 2
    value = C.c_ubyte()
    free_before = torch.cuda.mem_get_info()[0]
    with storage.retain_transient_backing():
        first = allocate(segment_size, storage.device, None)
        assert first is not None
        assert storage.mappings[first].nbytes == segment_size
        transient_free(first, segment_size, storage.device, None)

        second = allocate(smaller_size, storage.device, None)
        assert second is not None
        assert storage.mappings[second].nbytes == segment_size
        storage.cuda.call("cuMemsetD8_v2", C.c_uint64(second), C.c_ubyte(19),
                          C.c_size_t(smaller_size))
        storage.cuda.call("cuMemcpyDtoH_v2", C.byref(value), C.c_uint64(second),
                          C.c_size_t(1))
        assert value.value == 19
        snapshot = storage._snapshot()
        assert snapshot["retained_handles"] == 1
        assert snapshot["retained_reusable_handles"] == 0
        transient_free(second, smaller_size, storage.device, None)
        snapshot = storage._snapshot()
        assert snapshot["retained_handles"] == 2
        assert snapshot["retained_reusable_handles"] == 1
        assert snapshot["retained_reusable_bytes"] == segment_size
        assert free_before - torch.cuda.mem_get_info()[0] < segment_size + storage.granularity
    snapshot = storage._snapshot()
    assert snapshot["retained_handles"] == 0
    assert snapshot["retained_reference_bytes"] == 0
    assert torch.cuda.mem_get_info()[0] >= free_before - storage.granularity
    assert storage.error is None


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
@pytest.mark.parametrize("count", [96, 65536])
def test_mapped_mailbox_publishes_rows_during_graph_replay(count, mailbox_storage):
    import torch
    from ignition.components.ngram_delivery import publish_requests

    with torch.cuda.device(0):
        storage = mailbox_storage
        region = storage.host_allocate("requests", (count + 1,), torch.int64)
        graph = torch.cuda.CUDAGraph()
        try:
            assert region.host_view.data_ptr() == region.device_view.data_ptr() == region.address
            ids = torch.arange(count, device="cuda", dtype=torch.int64)
            sequence = torch.ones(1, device="cuda", dtype=torch.int64)
            expected = torch.arange(count, dtype=torch.int64)
            publish_requests(ids, region.device_view, sequence)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                publish_requests(ids, region.device_view, sequence)
            for value in (17, 127, 2048):
                ids.add_(1)
                expected.add_(1)
                sequence.fill_(value)
                torch.cuda.synchronize()
                region.host_view.fill_(-1)
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                deadline = time.monotonic() + 10
                while region.host_view[-1].item() != value:
                    assert time.monotonic() < deadline, "mailbox publication timed out"
                    time.sleep(.001)
                torch.testing.assert_close(region.host_view[:-1], expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        finally:
            graph.reset()
            region.close()
