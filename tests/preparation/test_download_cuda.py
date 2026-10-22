"""Package readback must preserve immutable bytes across repeated DMA transfers."""

import os
from random import Random

import pytest

from ignition.preparation.runtime_package import _Cuda


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_repeated_download_matches_cpu_bytes_for_normal_and_package_storage():
    import torch

    from ignition.preparation.package_storage import PackageStorage

    # The former pageable reader intermittently changed two four-byte words in
    # an otherwise unchanged 8 MiB block. Use an independent CPU source of truth.
    expected = Random(27).randbytes((8 << 20) + 73)
    source = torch.frombuffer(bytearray(expected), dtype=torch.uint8).cuda()
    storage = PackageStorage()
    with storage:
        packaged = torch.empty_like(source)
        packaged.copy_(source)
    torch.cuda.synchronize()
    cuda = _Cuda()
    for tensor in (source, packaged):
        for _ in range(1024):
            assert cuda.download(tensor.data_ptr() + 17, len(expected) - 23) == expected[17:-6]
        for size in (0, 1, 4095, 4096, 65537):
            assert cuda.download(tensor.data_ptr(), size) == expected[:size]
    with pytest.raises(ValueError, match="nonnegative"):
        cuda.download(source.data_ptr(), -1)
