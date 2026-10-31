"""GLM MoE folds the serial shared expert into the routed reduction when the experts can."""

from types import SimpleNamespace

import torch

from ignition.preparation.scratch import ScratchArena


class Operator:
    """A prepared operator stand-in; as experts it declares no staged reduction, like b12x's routed experts."""

    def __init__(self, launches, name, output=None):
        self.launches, self.name, self.output = launches, name, output
        self.plan = SimpleNamespace(name=name)

    def __call__(self, *operands):
        self.launches.append(self.name)
        return self.output


class StagedExperts(Operator):
    """Stand-in for SM103 experts: staged rows reduce into ``shared`` as BF16(routed) + shared."""

    reduces_into_shared = True

    def __init__(self, launches, routed):
        super().__init__(launches, "experts")
        self.rows = routed

    def stage_rows(self):
        self.launches.append("stage")

    def reduce(self, shared, output):
        self.launches.append("reduce")
        return torch.add(self.rows, shared, out=output)


def _build(monkeypatch, experts_factory, *, overlap, rows=4):
    from ignition.recipes.glm53 import moe

    launches = []
    routed = torch.full((rows, 4096), 2, dtype=torch.bfloat16)
    shared = torch.full((rows, 4096), 3, dtype=torch.bfloat16)
    monkeypatch.setattr(moe, "UnquantizedProjection", lambda *a, **k: Operator(launches, "router"))
    monkeypatch.setattr(moe, "PreparedRouter", lambda *a, **k: Operator(launches, "route"))
    monkeypatch.setattr(moe, "PreparedMLP", lambda *a, **k: Operator(launches, "shared", shared))
    monkeypatch.setattr(moe, "PreparedRoutedExperts", lambda *a, **k: experts_factory(launches, routed))
    gate = SimpleNamespace(num_group=1, topk_group=1, top_k=8, num_experts=288, norm_topk_prob=True,
                           routed_scaling_factor=2.5)
    weights = SimpleNamespace(router=torch.zeros((288, 4096), dtype=torch.bfloat16), correction=torch.zeros(288),
                              shared=None, experts_a4_prefill=False,
                              experts=SimpleNamespace(num_experts=288, hidden_size=4096, intermediate_size=1024))
    source = torch.zeros((rows, 4096), dtype=torch.bfloat16)
    prepared = moe.PreparedGlmMoe(SimpleNamespace(gate=gate, config=SimpleNamespace(swiglu_limit=None)), source,
                                  weights, moe.MoeScratch.allocate(rows, 2, "cpu"), session=None, name="moe",
                                  workspace=ScratchArena("cpu", overlap=overlap))
    return prepared, source, launches, routed, shared


def test_serial_path_reduces_staged_rows_into_the_shared_output(monkeypatch):
    prepared, source, launches, routed, shared = _build(monkeypatch, StagedExperts, overlap=False)
    output = prepared(source)
    assert launches == ["router", "route", "shared", "stage", "reduce"]
    assert output is prepared.output and torch.equal(output, routed + shared)


def test_serial_path_without_staged_reduction_adds_the_routed_output(monkeypatch):
    prepared, source, launches, routed, shared = _build(
        monkeypatch, lambda launches, routed: Operator(launches, "experts", routed), overlap=False)
    output = prepared(source)
    assert launches == ["router", "route", "experts", "shared"]
    assert torch.equal(output, routed + shared)
