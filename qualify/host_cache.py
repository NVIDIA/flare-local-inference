"""Host prefix cache qualification: launch Flare, publish a conversation's
checkpoints, evict them from the GPU, and observe restores from host memory,
from disk across a restart, and shared by concurrent continuations.

Each scenario also runs against a server without a host store whose
continuations hit the GPU cache. Continuations ask for facts spread across the
whole restored context and must answer every one correctly; generated tokens
are not compared verbatim because two GPU-cache restores of one checkpoint
already diverge after a few tokens at temperature zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from qualify.server import Run, comparable_message, metric_delta, prefill_launches, resource_totals

ROOT = Path(__file__).resolve().parents[1]


class Server:
    """A `flare run` process bound to a local port."""

    def __init__(self, flare: Path, package: Path, gpus: str, port: int, log: Path, extra: list[str]):
        self.url = f"http://127.0.0.1:{port}"
        self.log = log.open("ab")
        command = [str(flare), "run", str(package), "--gpus", gpus,
                   "--host", "127.0.0.1", "--port", str(port), *extra]
        self.log.write((" ".join(command) + "\n").encode())
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, cwd=ROOT)

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"flare exited with status {self.process.returncode} before serving; see {self.log.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise TimeoutError("flare did not become healthy")

    def model_id(self) -> str:
        with urllib.request.urlopen(self.url + "/v1/models", timeout=10) as response:
            return json.load(response)["data"][0]["id"]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        if self.process.returncode != 0:
            raise RuntimeError(f"flare exited with status {self.process.returncode}; see {self.log.name}")


class Bound(Protocol):
    """A process or rank group bound to a local HTTP port."""

    url: str


def run_for(server: Bound, artifacts: Path, timeout: float) -> Run:
    artifacts.mkdir(parents=True, exist_ok=True)
    return Run(SimpleNamespace(base_url=server.url, artifacts=artifacts, timeout=timeout,
                               max_response_bytes=64 << 20, require_metrics=True, cases=None))


def fact_value(index: int) -> int:
    return index * 7919 % 1009


def long_prompt(nonce: str, sentences: int) -> str:
    return (f"Reference {nonce}. " + " ".join(
        f"Fact {index}: the {nonce[index % len(nonce)]} series records value {fact_value(index)}." for index in range(sentences))
        + "\nSummarize the facts above in one sentence.")


def retrieval(sentences: int, offset: int) -> tuple[str, list[int]]:
    """A question whose answer needs facts from every part of the context, and its answer."""
    facts = [(offset + step * sentences // 9) % sentences for step in range(9)]
    question = ("List the values recorded by facts " + ", ".join(str(fact) for fact in facts[:-1])
                + f" and {facts[-1]}, in that order, as comma-separated numbers only.")
    return question, [fact_value(fact) for fact in facts]


def answered(choice: dict, expected: list[int]) -> tuple[bool, list[int]]:
    numbers = [int(number) for number in re.findall(r"\d+", choice["message"].get("content") or "")][:len(expected)]
    return numbers == expected, numbers


def body(model_id: str, messages: list[dict], max_tokens: int = 48) -> dict:
    return {"model": model_id, "messages": messages, "temperature": 0, "seed": 1234, "max_tokens": max_tokens,
            "enable_thinking": False, "add_generation_prompt": True, "return_token_ids": True}


def wait_quiescent(run: Run, name: str, timeout: float) -> dict:
    """Idle metrics with no host store transfer in flight."""
    deadline = time.monotonic() + timeout
    while True:
        metrics = run.metrics(name)
        if metrics.get("flare_host_cache_outbound_in_flight", 0) == 0 and metrics.get("flare_host_cache_inbound_in_flight", 0) == 0:
            return metrics
        if time.monotonic() >= deadline:
            raise TimeoutError("host store transfers did not drain")
        time.sleep(0.05)


def evict_everything(run: Run, model_id: str, nonce: str, label: str, host: bool) -> None:
    """Publish more distinct prefixes than the GPU keeps, so every earlier checkpoint is evicted.

    Each filler publishes one terminal checkpoint, which the host store must retain alongside the
    conversation's records; the host store therefore needs more row slabs than the GPU has rows.
    """
    before = wait_quiescent(run, f"{label}-evict-before", run.args.timeout)
    capacity = resource_totals(before)["snapshots"]
    for index in range(capacity + 1):
        request = body(model_id, [{"role": "user", "content": f"Filler {nonce} {label} {index}. Reply with one word."}], 1)
        run.complete(run.request(f"{label}-evict-{index}", request))
    after = wait_quiescent(run, f"{label}-evict-after", run.args.timeout)
    delta = metric_delta(before, after)
    run.check(f"{label}.gpu_checkpoints_evicted", delta["flare_snapshot_evictions_total"] >= 1, delta=delta)
    if host:
        run.check(f"{label}.host_store_retains_every_record",
                  delta["flare_host_cache_stored_total"] == capacity + 1
                  and after["flare_host_cache_memory_records"] >= before["flare_host_cache_memory_records"] + capacity + 1,
                  before=before["flare_host_cache_memory_records"], after=after["flare_host_cache_memory_records"], delta=delta)


def conversation(run: Run, model_id: str, nonce: str, label: str, sentences: int) -> tuple[list[dict], dict, int]:
    messages = [{"role": "user", "content": long_prompt(nonce, sentences)}]
    response = run.request(f"{label}-turn-1", body(model_id, messages))
    choice = run.complete(response)
    prompt_tokens = response["body"]["usage"]["prompt_tokens"]
    question, _ = retrieval(sentences, 3)
    messages = messages + [{"role": "assistant", "content": choice["message"].get("content") or ""},
                           {"role": "user", "content": question}]
    return messages, choice, prompt_tokens


def continuation(run: Run, model_id: str, messages: list[dict], label: str) -> tuple[dict, dict, dict]:
    before = wait_quiescent(run, f"{label}-before", run.args.timeout)
    response = run.request(label, body(model_id, messages))
    choice = run.complete(response)
    after = wait_quiescent(run, f"{label}-after", run.args.timeout)
    return response, choice, metric_delta(before, after)


def check_host_restore(run: Run, label: str, response: dict, delta: dict, prompt_tokens: int, chunk: int, hits: int = 1) -> None:
    reused = response["body"].get("reused_prompt_tokens", 0)
    total = response["body"]["usage"]["prompt_tokens"]
    run.check(f"{label}.host_hit", delta["flare_host_cache_hits_total"] == hits
              and delta["flare_host_cache_fetched_total"] >= 1 and delta["flare_host_cache_fetch_failed_total"] == 0, delta=delta)
    run.check(f"{label}.reused_at_least_the_first_prompt", reused >= prompt_tokens, reused=reused, first_prompt_tokens=prompt_tokens)
    run.check(f"{label}.restored_not_recomputed", delta['flare_graph_launches_total{kind="restore"}'] == hits
              and delta['flare_graph_launches_total{kind="init"}'] == 0
              and delta['flare_graph_launches_total{kind="prefill"}'] <= hits * prefill_launches(total, reused, chunk), delta=delta,
              prompt_tokens=total, reused=reused)
    run.check(f"{label}.no_failures", delta["flare_requests_failed_total"] == delta["flare_requests_cancelled_total"] == 0, delta=delta)


def memory_scenario(args, flare_extra: list[str], label: str, expected: dict | None) -> dict:
    nonce = args.nonce
    server = Server(args.flare, args.package, args.gpus, args.port, args.artifacts / f"{label}.log", flare_extra)
    outputs = {}
    try:
        server.wait_ready(args.startup_timeout)
        run = run_for(server, args.artifacts / label, args.timeout)
        model_id = server.model_id()
        messages, first, prompt_tokens = conversation(run, model_id, nonce, "memory", args.sentences)
        outputs["first"] = comparable_message(first["message"])
        if expected is not None:
            evict_everything(run, model_id, nonce, "memory", True)
        response, choice, delta = continuation(run, model_id, messages, "memory-continue")
        outputs["continue"] = comparable_message(choice["message"])
        correct, numbers = answered(choice, retrieval(args.sentences, 3)[1])
        run.check("memory-continue.recalls_facts_across_the_context", correct,
                  expected=retrieval(args.sentences, 3)[1], observed=numbers)
        if expected is None:
            run.check("memory-continue.gpu_hit", delta["flare_cache_hits_total"] == 1
                      and delta['flare_graph_launches_total{kind="restore"}'] == 1, delta=delta)
        else:
            check_host_restore(run, "memory-continue", response, delta, prompt_tokens, args.prefill_chunk_size)
        # Two continuations of one evicted prefix admitted together share one restore.
        if expected is not None:
            evict_everything(run, model_id, nonce, "branch", False)
        questions = [retrieval(args.sentences, 11 + index) for index in range(2)]
        branches = [messages[:-1] + [{"role": "user", "content": question}] for question, _ in questions]
        before = wait_quiescent(run, "branch-before", args.timeout)
        results = run.concurrent("branch", [body(model_id, branch) for branch in branches])
        after = wait_quiescent(run, "branch-after", args.timeout)
        delta = metric_delta(before, after)
        outputs["branches"] = [comparable_message(run.complete(result)["message"]) for result in results]
        for index, (result, (_, values)) in enumerate(zip(results, questions)):
            correct, numbers = answered(result["body"]["choices"][0], values)
            run.check(f"branch-{index}.recalls_facts_across_the_context", correct, expected=values, observed=numbers)
        if expected is not None:
            run.check("branch.one_restore_serves_both", delta["flare_host_cache_hits_total"] + delta["flare_cache_hits_total"] == 2
                      and delta["flare_host_cache_fetched_total"] >= 1 and delta["flare_host_cache_fetch_failed_total"] == 0
                      and delta['flare_graph_launches_total{kind="init"}'] == 0, delta=delta)
        run.results["outputs"] = outputs
        (args.artifacts / f"{label}.json").write_text(json.dumps(run.results, indent=1))
        return outputs
    finally:
        server.stop()


def disk_scenario(args, memory: str, label: str) -> None:
    """Records survive a restart: a disk-only store with `memory` zero, or a composite otherwise.

    With a composite, the first server also evicts the GPU and restores from the memory tier,
    and the restarted server restores through the disk tier into memory.
    """
    nonce = args.nonce
    directory = args.artifacts / f"{label}-records"
    extra = [part for option in args.flare_option for part in option.split("=", 1)]
    extra += ["--prefix-cache-disk", str(directory), "--prefix-cache-disk-limit", args.disk_limit, "--prefix-cache-memory", memory]
    server = Server(args.flare, args.package, args.gpus, args.port, args.artifacts / f"{label}.log", extra)
    try:
        server.wait_ready(args.startup_timeout)
        run = run_for(server, args.artifacts / label, args.timeout)
        model_id = server.model_id()
        messages, first, prompt_tokens = conversation(run, model_id, nonce, label, args.sentences)
        metrics = wait_quiescent(run, f"{label}-written", args.timeout)
        run.check(f"{label}.records_written", metrics["flare_host_cache_disk_records"] >= 1
                  and metrics["flare_host_cache_stored_total"] >= 1, records=metrics["flare_host_cache_disk_records"])
        if memory != "0":
            evict_everything(run, model_id, nonce, f"{label}-memory", True)
            response, choice, delta = continuation(run, model_id, messages, f"{label}-memory-continue")
            check_host_restore(run, f"{label}-memory-continue", response, delta, prompt_tokens, args.prefill_chunk_size)
            correct, numbers = answered(choice, retrieval(args.sentences, 3)[1])
            run.check(f"{label}-memory-continue.recalls_facts_across_the_context", correct,
                      expected=retrieval(args.sentences, 3)[1], observed=numbers)
            metrics = wait_quiescent(run, f"{label}-memory-after", args.timeout)
            run.check(f"{label}.memory_tier_served_the_restore", metrics["flare_host_cache_bytes_in_total"] > 0
                      and metrics["flare_host_cache_disk_records"] >= metrics["flare_host_cache_memory_records"],
                      memory_records=metrics["flare_host_cache_memory_records"], disk_records=metrics["flare_host_cache_disk_records"])
        (args.artifacts / f"{label}.json").write_text(json.dumps(run.results, indent=1))
    finally:
        server.stop()
    server = Server(args.flare, args.package, args.gpus, args.port, args.artifacts / f"{label}-restart.log", extra)
    try:
        server.wait_ready(args.startup_timeout)
        run = run_for(server, args.artifacts / f"{label}-restart", args.timeout)
        model_id = server.model_id()
        response, choice, delta = continuation(run, model_id, messages, f"{label}-continue")
        check_host_restore(run, f"{label}-continue", response, delta, prompt_tokens, args.prefill_chunk_size)
        correct, numbers = answered(choice, retrieval(args.sentences, 3)[1])
        run.check(f"{label}-continue.recalls_facts_across_the_context", correct,
                  expected=retrieval(args.sentences, 3)[1], observed=numbers)
        if memory != "0":
            metrics = wait_quiescent(run, f"{label}-restart-after", args.timeout)
            run.check(f"{label}-continue.read_through_made_the_record_resident",
                      metrics["flare_host_cache_memory_records"] >= 1, memory_records=metrics["flare_host_cache_memory_records"])
        (args.artifacts / f"{label}-restart.json").write_text(json.dumps(run.results, indent=1))
    finally:
        server.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare", type=Path, default=ROOT / "flare/target/release/flare")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--memory", default="24G",
                        help="--prefix-cache-memory for the memory scenario; its row slabs must outnumber the GPU's cached rows")
    parser.add_argument("--disk-limit", default="50G")
    parser.add_argument("--sentences", type=int, default=400, help="facts in the first prompt; about 16 tokens each")
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--skip-disk", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-composite", action="store_true")
    parser.add_argument("--flare-option", action="append", default=[], metavar="FLAG=VALUE",
                        help="extra flare run option for every launch, such as --mtp-draft-tokens=0")
    args = parser.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("the artifact directory must not contain a previous run")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    # One nonce keeps every server's prompts identical so outputs are comparable.
    args.nonce = hashlib.sha256(f"host-cache-{time.time_ns()}".encode()).hexdigest()[:10]
    common = [part for option in args.flare_option for part in option.split("=", 1)]
    if not args.skip_memory:
        baseline = memory_scenario(args, common + ["--prefix-cache-memory", "0"], "baseline", None)
        memory_scenario(args, common + ["--prefix-cache-memory", args.memory], "memory", baseline)
    if not args.skip_disk:
        disk_scenario(args, "0", "disk")
    if not args.skip_composite:
        disk_scenario(args, args.memory, "composite")
    failed = 0
    for path in sorted(args.artifacts.glob("*.json")):
        results = json.loads(path.read_text())
        failed += sum(not check["passed"] for check in results["checks"]) + len(results["case_errors"])
    print(f"{'FAIL' if failed else 'PASS'}: {failed} failed checks across {args.artifacts}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
