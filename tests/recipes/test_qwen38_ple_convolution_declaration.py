"""The PLE convolution request carries a representative benchmark.

On SM120 a prefill short convolution with at least 256 rows has many tile
candidates, and preparation rejects a multi-candidate request that has no
benchmark call.
"""

from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import DeviceIdentity
from b12x.sequence import short_conv
from b12x.sequence.short_conv._tuning import TUNING

from ignition.components import recurrent

SM120 = DeviceIdentity("nvidia", (12, 0), 170, "synthetic SM120")


def _plan(mode):
    return short_conv.plan(
        short_conv.Caps(
            device="cuda:0",
            mode=mode,
            channels=10240,
            kernel_size=4,
            dilation=3,
            max_tokens=4096,
            max_seqs=2,
            max_state_slots=4,
            max_speculative_tokens=0,
        )
    )


@pytest.mark.parametrize("mode", ["prefill", "mixed"])
def test_multi_candidate_ple_convolution_request_has_a_benchmark_call(
    monkeypatch, mode
):
    plan = _plan(mode)
    assert len(TUNING.eligible_plan(plan.query, SM120).candidates) > 1
    declared = []
    monkeypatch.setattr(
        recurrent, "declare", lambda session, requests: declared.extend(requests)
    )
    monkeypatch.setattr(recurrent, "register_finalizer", lambda *args, **kwargs: None)
    recurrent.prepare_sequence_binding(
        plan,
        {"conv_state": torch.empty((4, 10240, 9))},
        session=SimpleNamespace(),
        name="ple.convolution",
        public_bind=lambda plan, **kwargs: None,
        mode=mode,
        bind_plan=True,
        set_binding=lambda binding: None,
    )
    (request,) = declared
    assert request.benchmark_call is not None
