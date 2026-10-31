# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Trace and compile prepared forwards in workers without model storage."""

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from . import host_memory
from .compilation import _OPTIONS, _defer_full_collection
from .host_memory import COMPILE_HEADROOM_BYTES, compile_threads, cpu_workers, device_memory_info, worker_limit

# Device memory one compiler worker can hold: its CUDA context plus Inductor's constant folding.
_WORKER_DEVICE_BYTES = 2 << 30
# A discrete GPU's worker holds only device memory, about 0.7 GiB for Kimi K3's 93-layer forwards.
_DISCRETE_WORKER_DEVICE_BYTES = 1 << 30


@_defer_full_collection()
def _compile_forward(payload, device, options, path, num_threads, grad_enabled):
    from .aot import aot_forward
    from .device_binding import bind_device
    from .forward_metadata import loads

    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(grad_enabled)
    torch.cuda.set_device(device)
    # A lent forward's metadata names its owner's GPU; it loads bound to this worker's.
    with bind_device(device):
        forward = loads(payload)
    artifact = aot_forward(forward, options=options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        artifact.save_compiled_function(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CompilePool:
    def __init__(self, device, *, headroom_bytes=0, pools=1, options=None, compile_workers=None,
                 compile_headroom_bytes=COMPILE_HEADROOM_BYTES):
        """Size workers for `device`, beside `pools - 1` other ranks' pools compiling on this host.

        Host workers leave the greater of `compile_headroom_bytes` and `headroom_bytes` free.
        """
        self.device = torch.device(device)
        self.pool = None
        options = dict(_OPTIONS if options is None else options)
        if type(headroom_bytes) is not int or headroom_bytes < 0:
            raise ValueError("headroom_bytes must be a nonnegative byte count")
        if type(compile_headroom_bytes) is not int or compile_headroom_bytes < 0:
            raise ValueError("compile_headroom_bytes must be a nonnegative byte count")
        if type(pools) is not int or pools < 1:
            raise ValueError("pools must be a positive count")
        budget = cpu_workers(compile_workers, pools=pools)
        # Workers open contexts on the owning device; a nearly full device compiles with fewer of them.
        free = device_memory_info(self.device)[0]
        shared = torch.cuda.get_device_properties(self.device).is_integrated
        if compile_workers is None:
            budget = max(1, options['compile_threads'] // pools)
        if shared and os.name == "nt":
            free = min(free, host_memory.available_bytes())
        worker_bytes = _WORKER_DEVICE_BYTES if shared else _DISCRETE_WORKER_DEVICE_BYTES
        capacity = (free - headroom_bytes) // worker_bytes
        if capacity < 1:
            raise MemoryError("forward compilation requires memory for a CUDA worker beyond preparation headroom "
                              f"({free >> 20} MiB free, {headroom_bytes >> 20} MiB headroom, "
                              f"{worker_bytes >> 20} MiB per worker)")
        limit = min(8, budget) if compile_workers is None else budget
        self.workers = worker_limit(min(limit, capacity), pools=pools, available=free if shared else None,
                                   device_bytes=_WORKER_DEVICE_BYTES if shared else 0,
                                   headroom_bytes=headroom_bytes if shared else 0,
                                   compile_headroom_bytes=compile_headroom_bytes)
        # Each process compiles one forward; nested pools would multiply the host budget.
        threads = 1
        if compile_workers is None:
            threads = max(1, budget // self.workers)
            if shared:
                threads = compile_threads(threads, workers=pools * self.workers, available=free,
                    device_bytes=_WORKER_DEVICE_BYTES, headroom_bytes=headroom_bytes,
                    compile_headroom_bytes=compile_headroom_bytes)
        self.options = dict(options, compile_threads=threads)

    def submit(self, forward, path):
        from .forward_metadata import dumps

        return self.submit_payload(dumps(forward), path)

    def submit_payload(self, payload, path):
        """Compile a forward serialized by `forward_metadata.dumps`, possibly on another rank, into `path`."""
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool

        if self.pool is None:
            if os.name == "nt":
                self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"))
            else:
                self.pool = SubprocPool(self.workers, kind=SubprocKind.SPAWN)
        return self.pool.submit(_compile_forward, payload, self.device, self.options, str(path),
                                torch.get_num_threads(), torch.is_grad_enabled())

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.pool is not None:
            self.pool.shutdown()
