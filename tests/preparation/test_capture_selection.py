"""Diagnostic support forwards must not expand the compiled graph catalog."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ignition.preparation import capture


def test_support_forward_is_neither_compiled_nor_captured(monkeypatch):
    calls = []
    selected = lambda: calls.append("verification")
    support = lambda: calls.append("support")
    stream = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)

    class Cache:
        def __init__(self, **kwargs):
            pass

        def prepare_many(self, functions, signatures, **kwargs):
            assert functions == {"verification": selected}
            return functions

    monkeypatch.setattr(capture, "ForwardCache", Cache)
    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", lambda: "graph")
    monkeypatch.setattr(capture.torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    graphs, compiled = capture.capture_forwards(
        {"verification": selected, "support": support},
        {"verification": SimpleNamespace(kind="verification"), "support": SimpleNamespace(kind="prefill")},
        session=SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext),
        device="cuda:0", stream=stream, pool=None, compile_kinds=("verification", "prefill"),
        capture_names=("verification",))
    assert graphs == {"verification": "graph"}
    assert compiled == {"verification": selected}
    assert calls == ["verification"]


@pytest.mark.parametrize("selection", [("missing",), ("verification", "verification")])
def test_invalid_selection_rejected_before_cuda(selection):
    with pytest.raises(ValueError, match="capture selection"):
        capture.capture_forwards({"verification": lambda: None}, {}, session=None,
            device="cuda:0", stream=None, pool=None, capture_names=selection)


def test_width_one_speculation_can_be_selected_without_verification():
    from ignition.recipes.minimax_m3.bootstrap import _verification_variants

    shapes = ((8, 1), (8, 6), (1, 1))
    assert list(_verification_variants(shapes, {"speculative8x1"}, split=True)) == [
        ("speculative8x1", "speculative", 8, 1)]
    assert list(_verification_variants(shapes, {"verification8x6"}, split=True)) == [
        ("verification8x6", "verification", 8, 6)]
    assert [name for name, *_ in _verification_variants(shapes, None, split=True)] == [
        "verification8x6", "verification8x1", "speculative8x1", "verification1x1", "speculative1x1"]
    assert [name for name, *_ in _verification_variants(shapes, None, split=False)] == [
        "speculative8x1", "speculative1x1"]
