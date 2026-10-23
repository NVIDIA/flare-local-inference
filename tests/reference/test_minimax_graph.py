"""Verification replay compares persistent state and offered logits exactly."""

from types import SimpleNamespace

import torch


def test_parity_checks_only_offered_logits_and_all_participation(monkeypatch):
    import pytest
    from qualify.minimax_graph import _check_parity

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    logits = torch.zeros(6, 2)
    active = torch.tensor([True, False])
    lengths = torch.tensor([1, 0], dtype=torch.int32)
    state = torch.tensor([7])
    target = SimpleNamespace(width=3, active=active, metadata=SimpleNamespace(lengths=lengths),
        decoder=SimpleNamespace(scratch=SimpleNamespace(logits=logits)))

    class Forward:
        def __init__(self):
            self.target = target

        def __call__(self):
            logits.zero_()
            active.copy_(torch.tensor([True, False]))
            lengths.copy_(torch.tensor([1, 0]))

    def replay():
        logits.zero_()
        logits[1:].fill_(99)

    model = SimpleNamespace(device=torch.device("cpu"))
    forward, graph = Forward(), SimpleNamespace(replay=replay)
    saved = [(state, state.clone())]
    _check_parity(model, forward, [state], saved, graph=graph)
    for changed in ("offered", "active", "lengths", "state"):
        def invalid_replay():
            replay()
            if changed == "offered":
                logits[0, 0] = 1
            elif changed == "active":
                active[1] = True
            elif changed == "lengths":
                lengths[0] = 2
            else:
                state[0] = 8
        with pytest.raises(AssertionError):
            _check_parity(model, forward, [state], saved, graph=SimpleNamespace(replay=invalid_replay))
