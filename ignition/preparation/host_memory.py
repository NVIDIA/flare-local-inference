# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Physical-memory budgets for preparation workers and kernel tuning."""

import ctypes
import gc
import os
from pathlib import Path

_WORKER_BYTES = 2 << 30
# System memory left free beside compiler processes and tuning races, unless a larger preparation
# headroom applies. `[preparation] compile_headroom` and `--compile-headroom` select another value.
COMPILE_HEADROOM_BYTES = 8 << 30
# Kernel modules and temporary compiler objects retained by each tuning rank.
_TUNING_GROWTH_BYTES = 1 << 30
# One Inductor compile subprocess inside a forward compiler.
_COMPILE_THREAD_BYTES = 512 << 20
# Host memory a preparing process holds for each compiled forward it restores to warm and capture:
# its generated module and kernel launchers. On GB10, both GLM TP2 ranks grew 3.47-3.50 GB (62 MiB
# per forward) from KV allocation through loading 56 forwards, warmup and capturing 80 graphs.
COMPILED_FORWARD_BYTES = 80 << 20


def cpu_workers(requested=None, *, pools=1):
    """Divide a host's compiler process budget among its preparation ranks."""
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError("compile workers must be a positive integer")
    if type(pools) is not int or pools < 1:
        raise ValueError("compiler pool count must be positive")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    budget = available if requested is None else requested
    if requested is not None and budget < pools:
        raise ValueError(f"compile workers must provide at least one process for each of {pools} local ranks")
    return max(1, budget // pools)


def release_unused(*, generation=2):
    """Collect preparation temporaries and release unused glibc arenas on Linux."""
    if generation == 0 and available_bytes() < COMPILE_HEADROOM_BYTES:
        generation = 2
    gc.collect(generation)
    if os.name == "nt":
        return
    trim = ctypes.CDLL(None).malloc_trim
    trim.argtypes = (ctypes.c_size_t,)
    trim.restype = ctypes.c_int
    trim(0)


def available_bytes():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "total_page_file",
                    "available_page_file", "total_virtual", "available_virtual", "extended_virtual")]

        status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return status.available_physical
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is required to size preparation workers")


def device_memory_info(device):
    """Report allocatable physical memory, including reclaimable RAM on integrated GPUs."""
    import torch

    free, total = torch.cuda.mem_get_info(device)
    if os.name != "nt" and torch.cuda.get_device_properties(device).is_integrated:
        free = min(total, available_bytes())
    return free, total


def virtual_device_memory(device, group=None, *, tuning_replica=None):
    """Use one device-memory extent for corresponding TP and tuning layouts.

    CUDA can report slightly different total bytes on otherwise identical GPUs.
    Reserve matching virtual cache shapes before constructing kernel declarations;
    physical preparation backing and runtime cache sizing remain independent.
    Call once per rank, before allocating its model cache ranges.
    """
    from .rank_group import minimum_across_ranks

    total = device_memory_info(device)[1]
    if tuning_replica is None:
        return minimum_across_ranks(total, group)
    from datetime import timedelta
    import torch.distributed as dist

    role = tuning_replica
    rank = 0 if group is None else dist.get_rank(group)
    store = dist.FileStore(role.rendezvous + '.device-memory', role.processes)
    store.set_timeout(timedelta(minutes=10))
    store.set(str(role.index * role.tensor_parallel + rank), str(total))
    keys = [str(member) for member in range(role.processes)]
    store.wait(keys)
    return min(int(store.get(key)) for key in keys)


def worker_limit(limit, *, pools=1, available=None, device_bytes=0, headroom_bytes=0,
                 compile_headroom_bytes=COMPILE_HEADROOM_BYTES):
    """Bound worker RAM after the greater of the compile headroom and the requested reserve."""
    if limit <= 0 or pools <= 0 or device_bytes < 0 or headroom_bytes < 0 or compile_headroom_bytes < 0:
        raise ValueError("compiler and pool limits must be positive; device bytes and headroom must be nonnegative")
    available = available_bytes() if available is None else available
    headroom = max(compile_headroom_bytes, headroom_bytes)
    worker_bytes = _WORKER_BYTES + device_bytes
    capacity = (available - headroom) // (pools * worker_bytes)
    if capacity < 1:
        raise MemoryError(f"preparation requires at least {(headroom + pools * worker_bytes) >> 20} MiB "
                          f"available host memory; found {available >> 20} MiB ({headroom >> 20} MiB headroom, "
                          f"{worker_bytes >> 20} MiB per worker in each of {pools} local pools)")
    return min(limit, capacity)


def compile_threads(limit, *, workers, available, device_bytes=0, headroom_bytes=0,
                    compile_headroom_bytes=COMPILE_HEADROOM_BYTES):
    """Bound each forward compiler's Inductor subprocesses by the RAM its workers leave.

    On GB10 a worker compiling a GLM forward holds about 2.8 GB, and each of its
    Inductor compile subprocesses about 0.4 GB; sixteen threads in one worker exhaust
    the memory a TP2 rank has left.
    """
    headroom = max(compile_headroom_bytes, headroom_bytes)
    spare = available - headroom - workers * (_WORKER_BYTES + device_bytes)
    return max(1, min(limit, spare // (workers * _COMPILE_THREAD_BYTES)))


def tuning_limits(device, workers, *, pools=1, headroom_bytes=0, race_budget=None,
                  compile_headroom_bytes=COMPILE_HEADROOM_BYTES):
    """Size compiler pools and candidate races against their physical memory supply."""
    import torch

    if not torch.cuda.get_device_properties(device).is_integrated:
        free = device_memory_info(device)[0]
        # Candidate fixtures and loaded modules grow after reserving the kernel scratch arena.
        # Bound each device independently; compiler processes still share the host RAM budget.
        usable = free - headroom_bytes - _TUNING_GROWTH_BYTES
        race = max(1, usable // 4)
        if race_budget is not None:
            race = min(race, race_budget)
        return worker_limit(workers, pools=pools, compile_headroom_bytes=compile_headroom_bytes) if workers else 0, race
    available = device_memory_info(device)[0]
    if os.name == "nt":
        available = min(available, available_bytes())
    headroom = max(compile_headroom_bytes, headroom_bytes)
    usable = (available - headroom) // pools - _TUNING_GROWTH_BYTES
    if usable <= 0:
        raise MemoryError(f"kernel tuning needs more than {(headroom + pools * _TUNING_GROWTH_BYTES) >> 20} MiB "
                          f"available shared memory; found {available >> 20} MiB")
    race = max(1, usable // 4)
    if race_budget is not None:
        race = min(race, race_budget)
    return min(workers, (usable - race) // _WORKER_BYTES), race


def release_page_cache(paths) -> None:
    """Drop files' clean pages from the page cache once their contents have been loaded.

    An integrated GPU's free memory excludes reclaimable page cache, so checkpoint pages
    read while loading weights would otherwise count against tuning, graphs and KV pages.
    """
    # Windows checkpoint reads bypass the filesystem cache.
    if os.name == "nt":
        return
    from .checkpoint_source import source_for

    for path in paths:
        if source_for(path) is not None:
            continue
        with open(path, "rb") as file:
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
