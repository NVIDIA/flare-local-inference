"""Native Qwen TP1 cache persistence and disabled-path regressions.

Requires FLARE_RUN_CUDA_TESTS=1, an assigned CUDA_VISIBLE_DEVICES, and
FLARE_OFFLOAD_TEST_PACKAGE pointing to a Qwen package with ordinary and MTP5
graphs. Build flare/target/release/flare before running. Use pytest --basetemp
on a filesystem supporting direct I/O; the test creates and removes its records.
"""

import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_OFFLOAD_TEST_PACKAGE")
    or not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="requires assigned GPUs and a compatible runtime package",
)
ROOT = Path(__file__).resolve().parents[2]


class Service:
    def __init__(self, url, process):
        self.url = url
        self.process = process

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=10) as response:
            return response.read().decode()

    def metrics(self):
        return {fields[0]: float(fields[1])
                for line in self.get("/metrics").splitlines()
                if not line.startswith("#") and len(fields := line.split()) == 2}

    def settled(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            counters = self.metrics()
            if all(counters[f"flare_host_cache_{direction}_in_flight"] == 0
                   for direction in ("outbound", "inbound")):
                return counters
            time.sleep(.01)
        pytest.fail("host transfers did not drain in 60 seconds")

    def ask(self, messages):
        model = json.loads(self.get("/v1/models"))["data"][0]["id"]
        body = {"model": model, "messages": messages, "temperature": 1,
                "seed": 1234, "max_tokens": 24, "enable_thinking": False}
        request = urllib.request.Request(self.url + "/v1/chat/completions",
                                         json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        assert result["choices"][0]["finish_reason"] != "length", result
        return result["choices"][0]["message"]["content"].strip()


@contextlib.contextmanager
def server(tmp_path, name, drafts=5, flags=()):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / f"{name}.log"
    devices = ",".join(str(rank) for rank in range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
    args = [str(ROOT / "flare/target/release/flare"), "run",
            os.environ["FLARE_OFFLOAD_TEST_PACKAGE"], "--gpus", devices,
            "--host", "127.0.0.1", "--port", str(port), "--mtp-draft-tokens", str(drafts), *flags]
    if publisher := os.environ.get("FLARE_TEST_PUBLISHER_KEYS"):
        args.extend(("--publisher-keys", publisher))
    with log_path.open("w") as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        service = Service(f"http://127.0.0.1:{port}", process)
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    service.get("/health")
                    break
                except OSError:
                    time.sleep(.1)
            else:
                pytest.fail(f"startup timed out: {log_path.read_text()}")
            yield service
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                pytest.fail(f"shutdown timed out: {log_path.read_text()}")
            assert process.returncode == 0, log_path.read_text()


def reference_messages(tag):
    return [{"role": "user", "content": f"Reference {tag}. "
             + " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(2600))
             + "\nWhat is the code recorded by item 211? Reply with only the number."}]


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_disabled_offload_preserves_gpu_prefix_reuse_without_host_resources(tmp_path, explicit_zero):
    directory = tmp_path / "disabled-records"
    flags = (["--prefix-cache-memory", "0", "--prefix-cache-disk", str(directory),
              "--prefix-cache-disk-limit", "0"] if explicit_zero else [])
    messages = reference_messages("disabled")
    with server(tmp_path, "disabled", flags=flags) as service:
        before = service.metrics()
        first = service.ask(messages)
        service.settled()
        second = service.ask(messages)
        after = service.settled()
        assert first == second
        assert after["flare_cache_reused_tokens_total"] - before["flare_cache_reused_tokens_total"] > 32000
        assert all(value == 0 for name, value in after.items() if name.startswith("flare_host_cache_"))
        threads = [path.read_text().strip() for path in Path(f"/proc/{service.process.pid}/task").glob("*/comm")]
        assert not any(name.startswith("flare-offload") for name in threads), threads
        assert not directory.exists()
    assert "Prepare host prefix cache" not in (tmp_path / "disabled.log").read_text()


def test_restart_separates_draft_state_and_streams_prefixes_larger_than_ram(tmp_path):
    directory = tmp_path / "records"
    disk = ["--prefix-cache-disk", str(directory), "--prefix-cache-disk-limit", "8G"]
    first = reference_messages("state-compatibility")
    with server(tmp_path, "ordinary", drafts=0, flags=disk) as service:
        service.ask(first)
        assert service.settled()["flare_host_cache_stored_total"] > 0
    ordinary = set(directory.iterdir())
    assert len(ordinary) == 1

    with server(tmp_path, "speculative", flags=disk) as service:
        first_answer = service.ask(first)
        counters = service.settled()
        assert counters["flare_host_cache_hits_total"] == 0
        assert counters["flare_host_cache_stored_total"] > 0
    assert len(set(directory.iterdir()) - ordinary) == 1

    composite = [*disk, "--prefix-cache-memory", "1G"]
    fresh = reference_messages("oversized-write")
    with server(tmp_path, "composite-read-write", flags=composite) as service:
        assert service.ask(first) == first_answer
        counters = service.settled()
        assert counters["flare_host_cache_fetched_total"] > 0
        assert counters["flare_host_cache_fetch_failed_total"] == 0
        assert counters["flare_host_cache_reused_tokens_total"] > 32000
        fresh_answer = service.ask(fresh)
        counters = service.settled()
        assert counters["flare_host_cache_store_failed_total"] == 0
        assert counters["flare_host_cache_stored_total"] > 0

    with server(tmp_path, "composite-restart", flags=composite) as service:
        assert service.ask(fresh) == fresh_answer
        counters = service.settled()
        assert counters["flare_host_cache_fetched_total"] > 0
        assert counters["flare_host_cache_fetch_failed_total"] == 0
        assert counters["flare_host_cache_reused_tokens_total"] > 32000

    # A damaged payload must fall back to prefill without exposing partial state.
    for record in directory.rglob("*.ckpt"):
        with record.open("r+b") as file:
            file.seek(-1, os.SEEK_END)
            byte = file.read(1)
            file.seek(-1, os.SEEK_END)
            file.write(bytes([byte[0] ^ 0xFF]))
    with server(tmp_path, "corrupt-record", flags=composite) as service:
        recomputed = service.ask(fresh)
        counters = service.settled()
        assert counters["flare_host_cache_fetch_failed_total"] > 0
        assert counters["flare_host_cache_fetched_total"] == 0
        assert counters['flare_graph_launches_total{kind="restore"}'] == 0
        assert counters['flare_graph_launches_total{kind="init"}'] == 1
        assert counters["flare_prefill_tokens_total"] > 32000
        assert service.ask(fresh) == recomputed


@pytest.mark.parametrize("tier", ["memory", "disk", "composite"])
def test_message_checkpoint_pressure_keeps_complete_continuations(tmp_path, tier):
    flags = []
    if tier in ("memory", "composite"):
        flags += ["--prefix-cache-memory", "1G"]
    if tier in ("disk", "composite"):
        flags += ["--prefix-cache-disk", str(tmp_path / "records"),
                  "--prefix-cache-disk-limit", "8G"]
    with server(tmp_path, f"message-pressure-{tier}", flags=flags) as service:
        before = service.metrics()
        capacity = int(before["flare_snapshot_capacity"])
        messages = [{"role": "system", "content": "Remember the access code 7319."}]
        for index in range(capacity + 8):
            messages += [{"role": "user", "content": f"Log entry {index}: the status is unchanged."},
                         {"role": "assistant", "content": "Recorded."}]
        messages += [{"role": "user", "content": "What is the access code? Reply with only the number."}]
        assert "7319" in service.ask(messages)
        first = service.settled()
        assert first["flare_snapshot_evictions_total"] > before["flare_snapshot_evictions_total"]
        assert "7319" in service.ask(messages)
        after = service.settled()
        assert after["flare_host_cache_fetch_failed_total"] == 0
        assert after["flare_cache_reused_tokens_total"] > first["flare_cache_reused_tokens_total"]
