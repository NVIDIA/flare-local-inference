"""Multi-host prefix cache qualification: run every rank of a multi-host package
as its own process, publish checkpoints, and confirm that each rank keeps its
own record files and that a restart restores from all of them.

Each rank runs on its own GPU with its own `--prefix-cache-disk` directory, so
the ranks exercise the sharded record transport that a package serving one rank
per host uses: rank zero indexes and plans, and every rank stores and restores
its own share of each record. The ranks stay on one host, which keeps the
scenario runnable without remote shells; a package whose ranks run on separate
hosts uses the same code path.

A continuation asks for facts spread across the whole restored context and must
answer every one correctly. Generated tokens are not compared verbatim: two
GPU-cache restores of one checkpoint already diverge after a few tokens at
temperature zero, so verbatim agreement does not distinguish a faithful restore
from an unfaithful one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from qualify.host_cache import (
    answered,
    check_host_restore,
    continuation,
    conversation,
    evict_everything,
    retrieval,
    run_for,
    wait_quiescent,
)

ROOT = Path(__file__).resolve().parents[1]


class Cluster:
    """Every rank of a multi-host package as its own `flare run` process."""

    def __init__(self, args: argparse.Namespace, label: str):
        self.args = args
        self.label = label
        self.url = f"http://127.0.0.1:{args.port}"
        self.processes: list[subprocess.Popen] = []
        self.logs = []

    def records(self, rank: int) -> Path:
        return self.args.artifacts / "records" / f"rank{rank}"

    def start(self) -> None:
        extra = [part for option in self.args.flare_option for part in option.split("=", 1)]
        rendezvous = ["--rendezvous", f"127.0.0.1:{self.args.rendezvous_port}"]
        # Rank zero binds the rendezvous; peers retry until it accepts them.
        self.spawn(
            0,
            [str(self.args.package), "--rank", "0", "--host", "127.0.0.1",
             "--port", str(self.args.port), *rendezvous, *extra],
        )
        for rank in range(1, len(self.args.gpus)):
            self.spawn(rank, ["--rank", str(rank), *rendezvous, *extra])

    def spawn(self, rank: int, arguments: list[str]) -> None:
        log = (self.args.artifacts / f"{self.label}-rank{rank}.log").open("ab")
        self.logs.append(log)
        command = [
            str(self.args.flare), "run", *arguments,
            "--gpus", "0",
            "--prefix-cache-memory", "0",
            "--prefix-cache-disk", str(self.records(rank)),
            "--prefix-cache-disk-limit", self.args.disk_limit,
        ]
        log.write((" ".join(command) + "\n").encode())
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.args.gpus[rank]))
        self.processes.append(
            subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             cwd=ROOT, env=environment)
        )

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for rank, process in enumerate(self.processes):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"rank {rank} exited with status {process.returncode} before serving; "
                        f"see {self.args.artifacts / f'{self.label}-rank{rank}.log'}"
                    )
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise TimeoutError("the cluster did not become healthy")

    def model_id(self) -> str:
        with urllib.request.urlopen(self.url + "/v1/models", timeout=10) as response:
            return json.load(response)["data"][0]["id"]

    def stop(self) -> None:
        """Stop rank zero first: it drains every rank, and the peers exit with it."""
        for process in self.processes[:1]:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=self.args.shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for process in self.processes[1:]:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in self.logs:
            log.close()
        failed = [rank for rank, process in enumerate(self.processes)
                  if process.returncode not in (0, -signal.SIGTERM)]
        self.processes = []
        self.logs = []
        if failed:
            raise RuntimeError(f"ranks {failed} exited abnormally; see {self.args.artifacts}")


def package_world(flare: Path, package: Path) -> int:
    """The rank count of a multi-host package, read from its metadata."""
    document = json.loads(subprocess.check_output(
        [str(flare), "inspect", "--unverified", "--json", str(package)], text=True))
    cluster = (document.get("package") or {}).get("cluster")
    if not cluster:
        raise SystemExit(f"{package} serves one host; run qualify/host_cache.py instead")
    return int(cluster["world"])


def record_names(cluster: Cluster, rank: int) -> list[str]:
    return sorted(path.name for path in cluster.records(rank).rglob("*.ckpt"))


def check_ranks_agree(run, cluster: Cluster, label: str, world: int) -> None:
    """Every rank holds the same record files, or the record is unusable."""
    names = [record_names(cluster, rank) for rank in range(world)]
    run.check(f"{label}.every_rank_holds_records", all(names), counts=[len(names_) for names_ in names])
    run.check(f"{label}.ranks_hold_the_same_records",
              all(names_ == names[0] for names_ in names), records=names[0])


def scenario(args: argparse.Namespace, world: int) -> int:
    nonce = args.nonce
    cluster = Cluster(args, "cluster")
    cluster.start()
    try:
        cluster.wait_ready(args.startup_timeout)
        run = run_for(cluster, args.artifacts / "cold", args.timeout)
        model_id = cluster.model_id()
        messages, _, prompt_tokens = conversation(run, model_id, nonce, "cold", args.sentences)
        wait_quiescent(run, "cold-written", args.timeout)
        check_ranks_agree(run, cluster, "cold", world)

        # The GPU holds the conversation's checkpoints, so a continuation reuses
        # them without touching any host's records.
        response, choice, delta = continuation(run, model_id, messages, "gpu-continue")
        run.check("gpu-continue.gpu_hit",
                  delta["flare_cache_hits_total"] == 1 and delta["flare_host_cache_hits_total"] == 0,
                  delta=delta)
        correct, numbers = answered(choice, retrieval(args.sentences, 3)[1])
        run.check("gpu-continue.recalls_facts_across_the_context", correct,
                  expected=retrieval(args.sentences, 3)[1], observed=numbers)

        # Evict the GPU copies; the continuation must come from every rank's records.
        evict_everything(run, model_id, nonce, "host", False)
        response, choice, delta = continuation(run, model_id, messages, "host-continue")
        check_host_restore(run, "host-continue", response, delta, prompt_tokens, args.prefill_chunk_size)
        correct, numbers = answered(choice, retrieval(args.sentences, 3)[1])
        run.check("host-continue.recalls_facts_across_the_context", correct,
                  expected=retrieval(args.sentences, 3)[1], observed=numbers)
        (args.artifacts / "cold.json").write_text(json.dumps(run.results, indent=1))
    finally:
        cluster.stop()

    # Restart every rank: the prefix can now come only from each rank's records.
    cluster = Cluster(args, "restart")
    cluster.start()
    try:
        cluster.wait_ready(args.startup_timeout)
        run = run_for(cluster, args.artifacts / "restart", args.timeout)
        model_id = cluster.model_id()
        before = wait_quiescent(run, "restart-joined", args.timeout)
        run.check("restart.every_rank_kept_its_records",
                  before["flare_host_cache_disk_records"] >= 1,
                  records=before["flare_host_cache_disk_records"])
        check_ranks_agree(run, cluster, "restart", world)
        response, choice, delta = continuation(run, model_id, messages, "disk-continue")
        check_host_restore(run, "disk-continue", response, delta, prompt_tokens, args.prefill_chunk_size)
        correct, numbers = answered(choice, retrieval(args.sentences, 3)[1])
        run.check("disk-continue.recalls_facts_across_the_context", correct,
                  expected=retrieval(args.sentences, 3)[1], observed=numbers)
        (args.artifacts / "restart.json").write_text(json.dumps(run.results, indent=1))
        passed = all(check["passed"] for check in run.results["checks"])
    finally:
        cluster.stop()
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare", type=Path, default=ROOT / "flare/target/release/flare")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3",
                        help="one GPU per rank, in rank order; the ranks run on this host")
    parser.add_argument("--port", type=int, default=8041, help="rank zero's HTTP port")
    parser.add_argument("--rendezvous-port", type=int, default=29941)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--disk-limit", default="50G")
    parser.add_argument("--sentences", type=int, default=400,
                        help="facts in the first prompt; about 16 tokens each")
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--shutdown-timeout", type=float, default=300)
    parser.add_argument("--nonce", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument("--flare-option", action="append", default=[], metavar="FLAG=VALUE",
                        help="extra serving option for every rank, as FLAG=VALUE")
    args = parser.parse_args()
    args.gpus = [int(gpu) for gpu in args.gpus.split(",")]
    args.artifacts.mkdir(parents=True, exist_ok=True)
    world = package_world(args.flare, args.package)
    if world != len(args.gpus):
        parser.error(f"the package serves {world} ranks; give {world} GPUs")
    return scenario(args, world)


if __name__ == "__main__":
    raise SystemExit(main())
