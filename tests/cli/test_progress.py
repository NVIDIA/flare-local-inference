"""Startup phase rows in consoles of different widths."""

import io
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rich.console import Console

from ignition.progress import StartupProgress, graph_catalog, phase


def rows(width, detail):
    stream = io.StringIO()
    progress = StartupProgress(stream=stream)
    progress.console = Console(file=stream, width=width, highlight=False)
    with phase(progress, "Select GPUs") as stage:
        stage.update(detail)
    return stream.getvalue().splitlines()


def test_a_wide_console_prints_the_whole_row():
    line = rows(120, "GPU 0 · concurrency 4 · up to 3 draft tokens")[-1]
    assert re.fullmatch(r"\[    OK    \] Select GPUs\.  GPU 0 · concurrency 4 · up to 3 draft tokens  \d+\.\ds", line)


def test_a_narrow_console_shortens_the_detail_and_keeps_status_and_timing():
    line = rows(60, "GPUs 0-1 of CUDA_VISIBLE_DEVICES=3,1 · concurrency 4 · exactly 2 draft tokens")[-1]
    assert len(line) == 60 and line.startswith("[    OK    ] Select GPUs.  GPUs 0-1 of CUDA_VISIBLE_D")
    assert re.search(r"…  \d+\.\ds$", line)


def signature(kind="decode", *, batch=1, width=1, packed=0, drafts=None, prefill=0, enabled=False):
    return SimpleNamespace(kind=kind, batch_capacity=batch, token_capacity=width,
                           packed_token_capacity=packed, draft_tokens=drafts,
                           prefill_capacity=prefill, speculation_enabled=enabled)


def test_catalog_plain_callback_retains_every_shape_without_inventing_cross_products():
    messages = []
    graph_catalog(messages.append, [signature(batch=1), signature(batch=2), signature(batch=1, width=2),
        signature("packed_speculative", batch=2, width=6, packed=4, drafts=5, enabled=True),
        signature("packed_speculative", batch=2, width=6, packed=12, drafts=5, enabled=True),
        signature("vision", width=576, enabled=None), signature("mixed", batch=2, prefill=128)])
    output = "\n".join(messages)
    assert "Graph catalog: 7 graphs (before compilation/capture)" in output
    rows = [line.split() for line in messages if line.strip().startswith("decode")]
    assert rows == [["decode", "off", "1,2", "1", "-", "-", "-", "2"],
                    ["decode", "off", "1", "2", "-", "-", "-", "1"]]
    packed = next(line.split() for line in messages if line.strip().startswith("packed_speculative"))
    assert packed == ["packed_speculative", "on", "2", "6", "4,12", "5", "-", "2"]
    assert any(line.split() == ["vision", "-", "1", "576", "-", "-", "-", "1"] for line in messages)
    assert any(line.split() == ["mixed", "off", "2", "1", "-", "-", "128", "1"] for line in messages)
    assert "\x1b" not in output


def test_catalog_counts_duplicate_variants_and_stays_quiet_without_progress():
    messages = []
    graph_catalog(messages.append, [signature(), signature()])
    row = next(line.split() for line in messages if line.strip().startswith("decode"))
    assert row[-1] == "2" and "Graph catalog: 2 graphs" in messages[0]
    def unavailable():
        raise AssertionError("quiet workers must not inspect graph metadata")
        yield
    graph_catalog(None, unavailable())


def test_catalog_is_persistent_with_an_active_nonterminal_phase():
    stream = io.StringIO()
    progress = StartupProgress(stream=stream)
    progress.console = Console(file=stream, width=120, color_system=None)
    with phase(progress, "Prepare serving graphs", live=False) as stage:
        stage.update("initial detail")
        graph_catalog(progress, [signature()])
        assert stage.detail == "initial detail"
        stage.update("later detail")
    output = stream.getvalue()
    assert output.count("Graph catalog: 1 graphs") == 1
    assert "decode" in output and "later detail" in output and "OK" in output


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX PTY")
def test_catalog_persists_above_an_active_terminal_phase():
    import errno
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    script = """
from types import SimpleNamespace
from ignition.progress import StartupProgress, graph_catalog, phase
progress = StartupProgress()
with phase(progress, 'Prepare serving graphs') as stage:
    graph_catalog(progress, [SimpleNamespace(kind='packed_speculative', batch_capacity=2,
        token_capacity=6, packed_token_capacity=12, draft_tokens=5,
        prefill_capacity=0, speculation_enabled=True)])
    stage.update('later work')
"""
    try:
        process = subprocess.run([sys.executable, "-c", script], stdout=slave, stderr=slave, timeout=10,
                                 env={**os.environ, "TERM": "xterm-256color"})
        os.close(slave)
        chunks = []
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(master)
    output = b"".join(chunks).decode()
    assert process.returncode == 0
    assert "\x1b[" in output
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    assert plain.count("Graph catalog: 1 graphs") == 1
    assert "packed_speculative" in plain and "later work" in plain and "OK" in plain
    assert "…" not in plain
