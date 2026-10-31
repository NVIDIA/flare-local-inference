"""Plans with several SM120 candidates are declared with representative benchmarks."""

import pytest
import torch
from b12x.preparation import DeviceIdentity

from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.components.weights import MLPWeights

SM120 = DeviceIdentity(vendor="nvidia", compute_capability=(12, 0), sm_count=170,
                       product_name="RTX PRO 6000 Blackwell Workstation Edition")


@pytest.fixture(autouse=True)
def sm120_device(monkeypatch):
    """Evaluate each plan's candidates against an SM120 identity on CPU tensors."""
    def races(plan, _device):
        return len(plan.contract.eligible_plan(plan.query, SM120).candidates) > 1

    for module in ("moe", "mlp"):
        monkeypatch.setattr(f"ignition.components.{module}.races_candidates", races)


class RecordingSession:
    """Collects the requests a component declares through the standalone path."""

    def __init__(self):
        self.requests = []

    def prepare(self, requests, **_):
        self.requests.extend(requests)


def assert_raced_requests_have_benchmarks(session):
    assert session.requests
    raced = 0
    for request in session.requests:
        plan = request.plan
        candidates = plan.contract.eligible_plan(plan.query, SM120).candidates
        if len(candidates) > 1:
            raced += 1
            assert request.benchmark_call is not None, request.name
    return raced


@pytest.mark.parametrize("score_func,bias", [("sigmoid", True), ("softmax", False)])
def test_router_race_declares_a_benchmark(score_func, bias):
    logits = torch.zeros(4096, 288)
    scratch = RouterScratch.allocate(4096, "cpu", experts=288, top_k=8)
    session = RecordingSession()
    PreparedRouter(logits, scratch, session=session, name="route_topk", score_func=score_func,
                   correction_bias=torch.zeros(288) if bias else None)
    assert assert_raced_requests_have_benchmarks(session) == 1


def test_mlp_swiglu_race_declares_a_benchmark():
    scratch = MLPScratch.allocate(4096, 256, 2048, "cpu")
    weights = MLPWeights(gate_up=torch.zeros(4096, 256, dtype=torch.bfloat16),
                         down=torch.zeros(256, 2048, dtype=torch.bfloat16))
    source = torch.zeros(4096, 256, dtype=torch.bfloat16)
    session = RecordingSession()
    PreparedMLP(source, weights, scratch.gate_up, scratch.intermediate, scratch.output,
                session=session, name="mlp", workspace=None)
    swiglu = [r for r in session.requests if r.name == "mlp.swiglu"]
    assert len(swiglu) == 1 and swiglu[0].benchmark_call is not None
    assert assert_raced_requests_have_benchmarks(session) >= 1


def test_guard_rejects_a_raced_plan_without_a_benchmark():
    logits = torch.zeros(4096, 288)
    scratch = RouterScratch.allocate(4096, "cpu", experts=288, top_k=8)
    session = RecordingSession()
    PreparedRouter(logits, scratch, session=session, name="route_topk", score_func="sigmoid")
    stripped = RecordingSession()
    stripped.requests = [type("R", (), {"name": r.name, "plan": r.plan, "benchmark_call": None})()
                         for r in session.requests]
    with pytest.raises(AssertionError):
        assert_raced_requests_have_benchmarks(stripped)


def test_fixed_or_non_cuda_plans_declare_no_benchmark():
    logits = torch.zeros(64, 288)
    scratch = RouterScratch.allocate(64, "cpu", experts=288, top_k=8)
    session = RecordingSession()
    PreparedRouter(logits, scratch, session=session, name="route_topk", score_func="sigmoid")
    from ignition.preparation.tuning import races_candidates

    assert races_candidates(session.requests[0].plan, "cpu") is False


def swiglu_candidates(gate_up, intermediate):
    source = torch.zeros(4096, 256, dtype=torch.bfloat16)
    weights = MLPWeights(gate_up=torch.zeros(4096, 256, dtype=torch.bfloat16),
                         down=torch.zeros(256, 2048, dtype=torch.bfloat16))
    scratch = MLPScratch.allocate(4096, 256, 2048, "cpu")
    session = RecordingSession()
    PreparedMLP(source, weights, gate_up if gate_up is not None else scratch.gate_up,
                intermediate if intermediate is not None else scratch.intermediate,
                scratch.output, session=session, name="mlp", workspace=None)
    (request,) = [r for r in session.requests if r.name == "mlp.swiglu"]
    plan = request.plan
    return plan.contract.eligible_plan(plan.query, SM120).candidates


def misaligned(like):
    """A contiguous BF16 view of the given shape starting two bytes past a 16-byte boundary."""
    flat = torch.zeros(like.numel() + 16, dtype=torch.bfloat16)
    start = (-flat.data_ptr() % 16) // 2 + 1
    view = flat[start:start + like.numel()].view(like.shape)
    assert view.data_ptr() % 16 == 2
    return view


@pytest.mark.parametrize("operand", ["gate_up", "intermediate"])
def test_misaligned_swiglu_operand_plans_without_the_vector_candidate(operand):
    scratch = MLPScratch.allocate(4096, 256, 2048, "cpu")
    aligned = swiglu_candidates(None, None)
    assert len(aligned) > 1
    views = {"gate_up": None, "intermediate": None}
    views[operand] = misaligned(getattr(scratch, operand))
    assert len(swiglu_candidates(views["gate_up"], views["intermediate"])) == 1
