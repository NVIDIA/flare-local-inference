"""Exercise terminal rendering during native CUDA capture and publication."""

import errno
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform == "win32" or os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires a POSIX PTY and explicit CUDA test opt-in")
@pytest.mark.parametrize("rank", (0, 1))
def test_native_export_terminal_progress(tmp_path, rank):
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
    fixture = Path(__file__).parents[2] / "helios/tests/export_progress_fixture.py"
    environment = {**os.environ, "TERM": "xterm-256color"}
    chunks = []
    process = subprocess.Popen([sys.executable, str(fixture), str(tmp_path / "export"), str(rank)],
                               stdout=slave, stderr=slave, env=environment)
    os.close(slave)
    deadline = time.monotonic() + 120
    try:
        while True:
            if time.monotonic() >= deadline:
                process.kill()
                pytest.fail("native CUDA export exceeded 120 seconds")
            if not select.select([master], [], [], 1)[0]:
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                break
            if not chunk:
                break
            chunks.append(chunk)
        process.wait(timeout=10)
    finally:
        os.close(master)
        if process.poll() is None:
            process.kill()
            process.wait()
    output = b"".join(chunks).decode(errors="replace")
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    assert process.returncode == 0, plain
    assert f"EXPORTED RANK {rank}" in plain
    assert "████" not in plain
    assert not any("\u2800" <= character <= "\u28ff" for character in plain)
    labels = ("Clear disposable storage", "Inspect CUDA graph functions", "Serialize CUDA graphs",
              "Capture compiled modules and globals", "Scan and copy reachable storage",
              "Export frontend assets", "Hash and publish payload blobs",
              "Sign and publish package metadata")
    if rank == 0:
        assert plain.count("Export native package.") == 2, plain
        for label in labels:
            assert label in plain, plain
    else:
        for label in labels:
            assert label not in plain, plain
        assert "Export native package." not in plain, plain
