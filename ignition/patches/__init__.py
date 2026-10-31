# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the pinned PyTorch Inductor and Triton fixes before preparation imports."""

import os
import stat
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from importlib.resources import files
from pathlib import Path

from ignition.errors import PreparationError

_VERSIONS = {"torch": "2.14.1+cu132", "triton": "3.8.0"}
_PATCHES = (
    "torch-inductor-triton-cache-filenames.patch",
    "torch-inductor-cooperative-static-launch.patch",
    "triton-platform-key-pointer-width.patch",
)


def _hunks(patch):
    """Read file names and exact replacement blocks from the bundled unified diffs."""
    path = None
    before = after = None
    for line in patch.splitlines(keepends=True):
        if line.startswith(("diff --git ", "@@ ")):
            if before is not None:
                yield path, "".join(before), "".join(after)
            before = after = None
            if line.startswith("diff --git "):
                _, _, source, target = line.split()
                allowed = target.startswith("b/torch/_inductor/") or target == "b/triton/runtime/build.py"
                if source[2:] != target[2:] or not allowed:
                    raise PreparationError(
                        "Preparation patches must modify existing torch/_inductor files or triton/runtime/build.py")
                path = target[2:]
            else:
                before, after = [], []
        elif before is not None:
            if line.startswith((" ", "-")):
                before.append(line[1:])
            if line.startswith((" ", "+")):
                after.append(line[1:])
    if before is not None:
        yield path, "".join(before), "".join(after)


def apply() -> tuple[Path, ...]:
    """Validate bundled patches, then atomically replace changed installation files.

    Matching uses complete hunk context without fuzz. Already applied
    hunks are accepted, including installations with only one patch applied.
    Replacing files breaks wheel-cache hardlinks instead of modifying shared bytes.
    """
    try:
        packages = {}
        for name, expected in _VERSIONS.items():
            try:
                package = distribution(name)
            except PackageNotFoundError:
                # torch requires triton only on Linux, so a Windows preparation
                # environment has none; its patches have no files to modify.
                continue
            if package.version != expected:
                raise PreparationError(
                    f"Preparation patches require {name}=={expected}; found {package.version}. "
                    "Install the locked preparation dependencies with uv sync --locked --extra prepare.")
            packages[name] = package

        sources, patched, modules = {}, {}, {}
        for name in _PATCHES:
            for relative, before, after in _hunks(files(__package__).joinpath(name).read_text(encoding="utf-8")):
                package = relative.split("/", 1)[0]
                if package not in packages:
                    continue
                path = Path(packages[package].locate_file(relative))
                modules[path] = relative.removesuffix(".py").replace("/", ".")
                if path not in sources:
                    sources[path] = path.read_text(encoding="utf-8")
                    patched[path] = sources[path]
                source = patched[path]
                counts = source.count(before), source.count(after)
                if counts == (1, 0):
                    patched[path] = source.replace(before, after, 1)
                elif counts != (0, 1):
                    raise PreparationError(
                        f"Cannot apply {name} to {path}: patch context is missing or ambiguous. "
                        f"Reinstall {package} with uv sync --locked --extra prepare --reinstall-package {package}.")

        changed = tuple(path for path in sources if patched[path] != sources[path])
        for path in changed:
            module = modules[path]
            if module in sys.modules:
                raise PreparationError(f"Cannot patch imported module {module}; restart Python before preparation.")
        for path in changed:
            # Keep replacement files beside their targets for atomic publication on one filesystem.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    output.write(patched[path].encode("utf-8"))
                    output.close()
                    temporary.chmod(stat.S_IMODE(path.stat().st_mode))
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return changed
    except PackageNotFoundError as error:
        raise PreparationError(
            "Preparation patches require the preparation dependencies; "
            "install them with uv sync --locked --extra prepare.") from error
    except OSError as error:
        raise PreparationError(f"Cannot apply preparation patches in the preparation environment: {error}") from error
