from types import SimpleNamespace

import pytest

from ignition.direct.executor import StepRunner


def test_eager_execution_requires_the_recipe_to_retain_its_forwards():
    with pytest.raises(RuntimeError, match="retains no prepared forwards"):
        StepRunner(SimpleNamespace(forwards=None), eager=True)


class Runner:
    """A rank's step runner that reports a fixed graph time."""

    def __init__(self, nanos):
        self.nanos, self.steps = nanos, []

    def run(self, step):
        self.steps.append(step["sequence"])
        return self.nanos

    def close(self):
        pass


def test_a_step_takes_its_slowest_ranks_graph_time(monkeypatch):
    import socket
    import threading

    from ignition.direct import executor

    peers = {f"rank{rank}": Runner(nanos) for rank, nanos in ((1, 7_000_000), (2, 9_500_000))}
    monkeypatch.setattr(executor, "StepRunner", lambda model, eager: peers[threading.current_thread().name])
    roots, threads = [], []
    for name in peers:
        root, peer = socket.socketpair()
        roots += [root, peer]
        threads.append(threading.Thread(target=executor.serve_peer, args=(None, peer), kwargs={"eager": False},
                                        name=name))
        threads[-1].start()
    steps = object.__new__(executor.RankSteps)
    steps.runner, steps.peers, steps.stopped = Runner(8_000_000), tuple(roots[::2]), False
    assert steps.run({"graph": "speculative16x6", "uploads": [], "frame": 1, "sequence": 1}) == 9_500_000
    peers["rank2"].nanos = 1
    assert steps.run({"graph": "speculative16x6", "uploads": [], "frame": 0, "sequence": 2}) == 8_000_000
    steps.stop()
    for thread in threads:
        thread.join(timeout=10)
    for connection in roots:
        connection.close()
    assert [peer.steps for peer in peers.values()] == [[1, 2], [1, 2]]


def test_combined_round_reads_terminal_confidence_without_a_separate_proposal():
    import torch
    from ignition.direct.executor import RankSteps
    steps = object.__new__(RankSteps)
    values = torch.tensor([[.25, .5], [.75, 1.]])
    steps.runner = SimpleNamespace(proposals={}, packed=frozenset({'combined'}),
                                   model=SimpleNamespace(proposal_confidence=values))
    assert steps.confidence('combined') == [[.25, .5], [.75, 1.]]
    assert steps.confidence('width-one') is None
