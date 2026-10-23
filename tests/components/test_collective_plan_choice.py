"""Prepared reductions preserve the selected surface's capture and ownership contract."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("surface", ["PCIeTwoShotBF16.all_reduce", "OneshotAllReduce.all_reduce"])
def test_preparation_uses_selected_surface_and_retains_storage(monkeypatch, surface):
    from b12x.comm.pcie import _twoshot_preparation
    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    source = torch.full((32, 5120), 3, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    callbacks, calls, selected = [], [], []
    plan = SimpleNamespace(query=SimpleNamespace(surface=surface))
    plan.request = lambda **kwargs: callbacks.append(kwargs) or object()
    runtime = SimpleNamespace(algorithm="oneshot", world_size=4, single_channel=True,
                              exchange_group=object(), plan=lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    def two_call(state, *, payload, out):
        calls.append("two")
        assert payload is source and out is output
        call = PreparedCall(run=lambda: out.copy_(payload), output=out, capture_safe=False,
                            owners=(state, payload, out))
        selected.append(call)
        return call

    monkeypatch.setattr(_twoshot_preparation, "prepared_call", two_call)
    operation = collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=object(), coordinator=object(), name="hidden.reduce")
    state = SimpleNamespace(query=plan.query,
        bind_input=lambda tensor: calls.append("bind") or None,
        run_plain=lambda inp, out: out.copy_(inp))
    prepared = callbacks[0]["prepare_call"](state)
    if selected:
        assert prepared is selected[0]
    else:
        assert all(actual is expected for actual, expected in zip(
            prepared.owners, (runtime, source, output), strict=True))
    assert prepared.capture_safe is (surface != "PCIeTwoShotBF16.all_reduce")
    prepared.run()
    assert torch.equal(source, output)
    assert calls == (["two"] if surface == "PCIeTwoShotBF16.all_reduce" else ["bind"])
    assert operation.runtime is runtime and operation.plan is plan
