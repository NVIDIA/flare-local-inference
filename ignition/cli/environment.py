# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Process environment that preparation libraries read, set from Flare's configuration.

PyTorch, b12x, Triton, and the CuTe DSL read these variables when they are
imported or first used, so `apply` runs before any of them is imported. Rank
and compiler processes inherit the result.
"""

import hashlib
import os
from importlib import util
from pathlib import Path

from ignition import patches

_DLL_DIRECTORIES = []


def flashinfer_workspace(kernel_cache: Path) -> Path:
    """Separate JIT build directories for independently editable installations."""
    package = util.find_spec("flashinfer")
    root = kernel_cache / "flashinfer"
    if package is None or package.origin is None:
        return root
    # FlashInfer keys build directories by version and GPU architecture. Distinct
    # checkouts can share both while requiring different sources and compiler flags.
    source = str(Path(package.origin).resolve()).encode()
    return root / hashlib.sha256(source).hexdigest()[:16]


def apply(configuration, gpus, b12x=()) -> None:
    """Select the plan's GPUs as process-local devices 0 through N-1, place every cache and set the recipe's
    b12x settings, which b12x reads only from its environment."""
    patches.apply()
    kernel_cache = configuration.directory("kernel_cache")
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in gpus),
        "HF_HUB_CACHE": str(configuration.directory("hub_cache")),
        "B12X_COMPILE_CACHE_DIR": str(kernel_cache / "b12x"),
        "FLASHINFER_WORKSPACE_BASE": str(flashinfer_workspace(kernel_cache)),
        "TRITON_CACHE_DIR": str(kernel_cache / "triton"),
        "TILELANG_CACHE_DIR": str(kernel_cache / "tilelang"),
        "TORCHINDUCTOR_CACHE_DIR": str(kernel_cache / "inductor"),
        **dict(b12x),
    })
    # cuTile discovers its compiler bundle independently of the native toolkit.
    cuda = configuration.directory("cuda")
    if cuda is not None:
        os.environ["CUDA_HOME"] = str(cuda)
        if os.name == "nt":
            os.environ["CUDA_PATH"] = str(cuda)
            os.environ["TRITON_PTXAS.EXE_PATH"] = str(cuda / "bin/ptxas.exe")
            for path in (cuda / "bin", cuda / "bin/x64", cuda / "lib/x64"):
                if path.is_dir():
                    _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
