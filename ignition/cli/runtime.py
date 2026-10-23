"""Locate the native Flare executable used by checkpoint distribution."""

import os
from pathlib import Path
import shutil

from ignition.errors import UsageError


def binary():
    installed = shutil.which("flare")
    if installed:
        return Path(installed)
    root = Path(__file__).resolve().parents[2]
    executable = root / "flare/target/release" / ("flare.exe" if os.name == "nt" else "flare")
    if not executable.is_file():
        raise UsageError("flare executable is missing; build flare/Cargo.toml or install flare on PATH")
    return executable
