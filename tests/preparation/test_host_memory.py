# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate compiler concurrency beside mapped model storage."""

import pytest
from types import SimpleNamespace

from ignition.preparation import host_memory
from ignition.preparation.host_memory import worker_limit


def test_loading_reclaims_promoted_cycles_when_host_headroom_is_low(monkeypatch):
    import gc
    import weakref

    class Owner:
        pass

    enabled = gc.isenabled()
    gc.disable()
    try:
        owner = Owner()
        owner.cycle = owner
        observer = weakref.ref(owner)
        gc.collect(0)
        del owner
        monkeypatch.setattr(host_memory, "available_bytes", lambda: 2 * host_memory.COMPILE_HEADROOM_BYTES)
        host_memory.release_unused(generation=0)
        assert observer() is not None
        monkeypatch.setattr(host_memory, "available_bytes", lambda: host_memory.COMPILE_HEADROOM_BYTES - 1)
        host_memory.release_unused(generation=0)
        assert observer() is None
    finally:
        if enabled:
            gc.enable()


def test_cpu_worker_budget_uses_affinity_and_is_shared_by_local_ranks(monkeypatch):
    monkeypatch.setattr(host_memory, "os", SimpleNamespace(sched_getaffinity=lambda _: set(range(48))))
    assert host_memory.cpu_workers() == 48
    assert host_memory.cpu_workers(pools=4) == 12
    assert host_memory.cpu_workers(24, pools=4) == 6
    assert host_memory.cpu_workers(80, pools=4) == 20
    with pytest.raises(ValueError, match="one process"):
        host_memory.cpu_workers(2, pools=4)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_cpu_worker_budgets_are_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        host_memory.cpu_workers(count)


def test_cpu_worker_budget_without_affinity_uses_cpu_count(monkeypatch):
    monkeypatch.setattr(host_memory, "os", SimpleNamespace(cpu_count=lambda: 24))
    assert host_memory.cpu_workers(pools=2) == 12


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("gib", [16, 24, 64, 128])
def test_compiler_pools_leave_host_headroom(pools, gib):
    workers = worker_limit(8, pools=pools, available=gib << 30)
    assert 1 <= workers <= 8
    assert pools * workers * (2 << 30) + (8 << 30) <= gib << 30


def test_insufficient_host_ram_fails_before_workers_start():
    with pytest.raises(MemoryError, match="available host memory"):
        worker_limit(8, pools=4, available=15 << 30)


def test_unified_memory_budgets_host_and_device_storage_together():
    assert worker_limit(8, available=24 << 30) == 8
    assert worker_limit(8, available=24 << 30, device_bytes=2 << 30) == 4
    assert worker_limit(8, pools=2, available=24 << 30, device_bytes=2 << 30) == 2
    with pytest.raises(MemoryError, match="available host memory"):
        worker_limit(8, available=11 << 30, device_bytes=2 << 30)


@pytest.mark.parametrize("compile_headroom,expected", [(4 << 30, 5), (8 << 30, 4), (16 << 30, 2)])
def test_compile_headroom_sets_the_memory_compiler_pools_leave_free(compile_headroom, expected):
    assert worker_limit(8, available=24 << 30, device_bytes=2 << 30,
                        compile_headroom_bytes=compile_headroom) == expected
    # A larger preparation headroom still applies.
    assert worker_limit(8, available=24 << 30, device_bytes=2 << 30, headroom_bytes=20 << 30,
                        compile_headroom_bytes=compile_headroom) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        worker_limit(8, available=24 << 30, compile_headroom_bytes=-1)


def test_lower_compile_headroom_admits_a_compiler_on_a_loaded_unified_memory_host():
    # After DeepSeek V4.1 TP3 loads its weights, a GB10 rank has about 11 GiB left.
    with pytest.raises(MemoryError, match="found 11264 MiB [(]8192 MiB headroom, 4096 MiB per worker"):
        worker_limit(8, available=11 << 30, device_bytes=2 << 30)
    assert worker_limit(8, available=11 << 30, device_bytes=2 << 30, compile_headroom_bytes=6 << 30) == 1


@pytest.mark.parametrize("compile_headroom,threads", [(None, 2), (6 << 30, 6), (12 << 30, 1)])
def test_inductor_threads_follow_the_compile_headroom(compile_headroom, threads):
    options = {} if compile_headroom is None else {"compile_headroom_bytes": compile_headroom}
    assert host_memory.compile_threads(16, workers=1, available=13 << 30, device_bytes=2 << 30,
                                       **options) == threads


@pytest.mark.parametrize("compile_headroom,expected", [(None, (2, 7 << 28)), (4 << 30, (4, 11 << 28)),
                                                       (14 << 30, (0, 1 << 28))])
def test_integrated_tuning_follows_the_compile_headroom(monkeypatch, compile_headroom, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (16 << 30, 121 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 16 << 30)
    options = {} if compile_headroom is None else {"compile_headroom_bytes": compile_headroom}
    assert host_memory.tuning_limits("cuda:0", 8, **options) == expected
    with pytest.raises(MemoryError, match="kernel tuning"):
        host_memory.tuning_limits("cuda:0", 8, compile_headroom_bytes=15 << 30)


@pytest.mark.parametrize("compile_headroom,workers", [(None, 32), (40 << 30, 16)])
def test_discrete_tuning_compilers_leave_the_compile_headroom_in_host_ram(monkeypatch, compile_headroom, workers):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (24 << 30, 96 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    options = {} if compile_headroom is None else {"compile_headroom_bytes": compile_headroom}
    assert host_memory.tuning_limits("cuda:0", 64, headroom_bytes=1 << 30, **options)[0] == workers


@pytest.mark.parametrize("platform", ["posix", "nt"])
@pytest.mark.parametrize("available,expected", [(12 << 30, 12 << 30), (40 << 30, 32 << 30)])
def test_integrated_device_uses_available_ram(monkeypatch, available, expected, platform):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: available)
    monkeypatch.setattr(host_memory, "os", SimpleNamespace(name=platform))
    assert host_memory.device_memory_info("cuda:0") == (
        4 << 30 if platform == "nt" else expected, 32 << 30)


def test_windows_device_budget_includes_dedicated_memory_beyond_host_ram(monkeypatch):
    import torch

    monkeypatch.setattr(host_memory, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (105 << 30, 110 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 48 << 30)
    assert host_memory.device_memory_info("cuda:0") == (105 << 30, 110 << 30)
    workers, race = host_memory.tuning_limits("cuda:0", 8, headroom_bytes=8 << 30)
    assert workers * (2 << 30) + race + (1 << 30) + (8 << 30) <= 48 << 30


def test_discrete_device_budget_is_independent_of_host_ram(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (24 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: pytest.fail("discrete VRAM must not use host capacity"))
    assert host_memory.device_memory_info("cuda:0") == (24 << 30, 32 << 30)


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("race_budget", [None, 1 << 20, 64 << 30])
def test_discrete_tuning_keeps_device_headroom_without_dividing_vram_across_ranks(
    monkeypatch, pools, race_budget
):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 << 30, 96 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: 72 << 30)
    workers, race = host_memory.tuning_limits("cuda:0", 16, pools=pools,
        headroom_bytes=1 << 30, race_budget=race_budget)
    assert workers == min(16, 32 // pools)
    assert 0 < race <= 512 << 20
    assert 4 * race + (2 << 30) <= 4 << 30
    if race_budget is not None:
        assert race <= race_budget


def test_discrete_small_workloads_keep_a_positive_race_budget_without_workers(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (768 << 20, 24 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: pytest.fail("no compiler pool needs host RAM"))
    assert host_memory.tuning_limits("cuda:0", 0, headroom_bytes=1 << 30) == (0, 1)


@pytest.mark.parametrize("pools", [1, 2, 4])
@pytest.mark.parametrize("headroom", [0, 16 << 30])
@pytest.mark.parametrize("race_budget", [None, 1 << 20, 64 << 30])
def test_tuning_races_and_compilers_share_one_ram_budget(monkeypatch, pools, headroom, race_budget):
    import torch

    available = 40 << 30
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (available, 121 << 30))
    monkeypatch.setattr(host_memory, "available_bytes", lambda: available)
    workers, race = host_memory.tuning_limits("cuda:0", 8, pools=pools,
        headroom_bytes=headroom, race_budget=race_budget)
    assert workers > 0 and race > 0
    assert pools * (workers * (2 << 30) + race + (1 << 30)) + max(8 << 30, headroom) <= available
    if race_budget is not None:
        assert race <= race_budget
    if race_budget == 1 << 20:
        assert race == race_budget
