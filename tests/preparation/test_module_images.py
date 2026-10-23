"""Native CUPTI recorder loading and actual driver allocation lifetimes."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.preparation.module_images import ModuleImages


@pytest.mark.parametrize("present", [False, True])
def test_native_recorder_rejects_an_unloadable_library(tmp_path, present):
    path = tmp_path / "invalid-cupti.so"
    if present:
        path.write_bytes(b"not a shared library")
    with pytest.raises((OSError, RuntimeError), match="invalid-cupti"):
        ModuleImages(path)
    assert ModuleImages.active == 0


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_native_recorder_tracks_real_driver_allocation_and_context_lifetimes():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", r'''
import ctypes as C
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.runtime_package import _Cuda

cuda = _Cuda()
cuda.call("cuInit", C.c_uint(0))
context, device = C.c_void_p(), C.c_int()
cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
with ModuleImages() as images:
    assert ModuleImages.active == 1
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    try:
        pointer = C.c_uint64()
        cuda.call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(4096))
        assert images.allocations[pointer.value] == (4096, "device")
        assert images.allocation_contexts[pointer.value] == context.value
        cuda.call("cuMemFree_v2", pointer)
        assert pointer.value not in images.allocations
        cuda.call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(8192))
        assert pointer.value in images.allocations
    finally:
        cuda.call("cuCtxDestroy_v2", context)
    assert pointer.value not in images.allocations
assert ModuleImages.active == 0
'''], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
