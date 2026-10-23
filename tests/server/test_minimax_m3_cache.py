"""Complete MiniMax M3 continuations through native host offload and readmission.

Requires FLARE_RUN_CUDA_TESTS=1, four assigned CUDA_VISIBLE_DEVICES, and
FLARE_MINIMAX_TEST_PACKAGE, FLARE_MINIMAX_TEST_BINARY, and
FLARE_TEST_PUBLISHER_KEYS locating a TP4 package, release executable, and its
trusted local-development publisher profile. The package must contain ordinary
decode and seven-proposal DSpark graphs with concurrency four and prefill 4096.
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from ignition.preparation.package_file import read_index
from tests.server.test_prefix_cache_offload import Service


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or any(not os.environ.get(name) for name in (
        "CUDA_VISIBLE_DEVICES", "FLARE_MINIMAX_TEST_PACKAGE", "FLARE_MINIMAX_TEST_BINARY",
        "FLARE_TEST_PUBLISHER_KEYS")),
    reason="requires four assigned GPUs and a signed native MiniMax M3 package",
)


@contextlib.contextmanager
def server(tmp_path, drafts):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(devices) == 4, devices
    profile = Path(os.environ["FLARE_TEST_PUBLISHER_KEYS"])
    publisher, = [entry for entry in json.loads(profile.read_text())
                  if entry["namespace"] == "local-development" and entry.get("development")]
    fingerprint = hashlib.sha256(bytes(publisher["public_key"])).hexdigest()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = [os.environ["FLARE_MINIMAX_TEST_BINARY"], "run", os.environ["FLARE_MINIMAX_TEST_PACKAGE"],
            "--offline", "--gpus", "0,1,2,3", "--host", "127.0.0.1", "--port", str(port),
            "--publisher-keys", str(profile), "--dangerously-approve-key", fingerprint,
            "--dangerously-accept-publisher", publisher["namespace"],
            "--trust-store", str(tmp_path / "trust.toml"), "--verify-weights",
            "--max-concurrency", "4", "--mtp-draft-tokens", str(drafts),
            "--no-mtp-adaptive-verification", "--prefix-cache-memory", "8G",
            "--prefix-cache-disk-limit", "0"]
    log_path = tmp_path / f"offload-draft{drafts}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        service = Service(f"http://127.0.0.1:{port}", process)
        try:
            deadline = time.monotonic() + 300
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


def ask(service, model, prompt):
    body = dict(model=model, messages=[dict(role="user", content=prompt)], temperature=1, top_k=1,
                seed=1234, max_tokens=128, return_token_ids=True,
                chat_template_kwargs=dict(thinking_mode="disabled"))
    request = urllib.request.Request(service.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        pytest.fail(f"HTTP {error.code}: {error.read().decode()}")
    choice, = result["choices"]
    assert choice["finish_reason"] == "stop", result
    assert result["usage"]["completion_tokens"] == len(choice["token_ids"]), result
    return result


@pytest.mark.parametrize("drafts", [0, 7], ids=["ordinary", "dspark"])
def test_evicted_complete_continuation_is_readmitted_and_branched(tmp_path, drafts):
    package, _ = read_index(os.environ["FLARE_MINIMAX_TEST_PACKAGE"])
    assert len(package.ranks) == 4
    contract = package.contract
    assert contract.page_tokens == 128 and contract.prefill_chunk_size == 4096
    assert contract.snapshots > 0 and package.ranks[0].execution.max_active >= 4
    assert 8 in contract.speculative_widths
    prompt = "Reference offload. " + " ".join(
        f"Item {index} records code {index * 17 % 997}." for index in range(600))
    prompt += "\nWhat code does item 211 record? Reply with only the integer."
    with server(tmp_path, drafts) as service:
        model = json.loads(service.get("/v1/models"))["data"][0]["id"]
        cold = ask(service, model, prompt)
        if cold["usage"]["prompt_tokens"] % contract.page_tokens == 0:
            prompt = "Marker. " + prompt
            cold = ask(service, model, prompt)
        assert cold["usage"]["prompt_tokens"] > 4096, cold["usage"]
        assert cold["usage"]["prompt_tokens"] % contract.page_tokens != 0
        expected = cold["choices"][0]
        assert expected["message"]["content"].strip() == "596", cold
        resident = service.settled()
        assert resident["flare_host_cache_stored_total"] > 0
        for index in range(2 * contract.snapshots + 4):
            ask(service, model, f"Distinct record {index}. Reply with only the integer {index + 1000}.")
            before = service.settled()
            if (before["flare_snapshot_evictions_total"] - resident["flare_snapshot_evictions_total"]
                    >= resident["flare_cached_checkpoints"]):
                break
        else:
            pytest.fail("distinct requests did not retire the cold prefix's checkpoint count")
        before = service.settled()
        assert before["flare_snapshot_evictions_total"] > resident["flare_snapshot_evictions_total"]
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(lambda _: ask(service, model, prompt), range(2)))
        after = service.settled()
        names = ("flare_cached_checkpoints", "flare_snapshot_evictions_total", "flare_host_cache_memory_records",
                 "flare_host_cache_hits_total", "flare_host_cache_fetched_total", "flare_host_cache_reused_tokens_total")
        print({phase: {name: counters[name] for name in names}
               for phase, counters in (("resident", resident), ("evicted", before), ("readmitted", after))})
        for result in results:
            assert result["choices"][0]["token_ids"] == expected["token_ids"], result
            assert result["choices"][0]["message"]["content"].strip() == "596", result
            assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, result
        for name in ("flare_host_cache_hits_total", "flare_host_cache_fetched_total",
                     "flare_host_cache_reused_tokens_total", 'flare_graph_launches_total{kind="restore"}'):
            assert after[name] > before[name], (name, before, after)
        assert after["flare_host_cache_fetch_failed_total"] == 0
        assert after["flare_host_cache_store_failed_total"] == 0
        if drafts:
            assert after['flare_graph_launches_total{kind="proposal"}'] > 0
            assert after['flare_graph_launches_total{kind="verification"}'] > 0
