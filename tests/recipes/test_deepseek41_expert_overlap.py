"""Decoder composition selects expert overlap from physical rows and execution mode."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41 import decoder


@pytest.mark.parametrize("rows", [33, 48, 64, 96, 128, 129])
@pytest.mark.parametrize("mode", ["decode", "verify", "draft", "prefill"])
@pytest.mark.parametrize("overlap", [False, True])
def test_layer_overlap_uses_physical_capacity_and_enabled_workspace(monkeypatch, rows, mode, overlap):
    scratch = decoder.Scratch.allocate(rows, 1, 4, "meta")
    calls = []

    def component(*args, **kwargs):
        return SimpleNamespace(plan=object())

    def concurrent(shared, **kwargs):
        calls.append((shared, kwargs))
        return object()

    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "UnquantizedProjection",
                 "PreparedRouter", "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost"):
        monkeypatch.setattr(decoder, name, component)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    monkeypatch.setattr(decoder, "ConcurrentSharedExpert", concurrent)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        vision_correction=None, router=None, correction=None, experts=None, shared=None))
    workspace = SimpleNamespace(overlap=overlap)
    session = object()
    layer = decoder.PreparedLayer(3, weights, None, scratch,
        SimpleNamespace(token_ids=torch.empty(rows, device="meta")),
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=4), mode=mode, runtime=None, coordinator=None,
        session=session, name="layer", workspace=workspace, carry=False, defer_post=True)
    expected = overlap and mode in ("decode", "verify", "draft") and rows <= 128
    assert (layer.concurrent_shared is not None) == expected
    if expected:
        shared, options = calls.pop()
        assert shared is layer.shared
        assert options == dict(parallel_plans=(layer.router_projection.plan, layer.router.plan,
            layer.experts.plan), session=session, workspace=workspace, defer_launch=True)
    assert not calls
