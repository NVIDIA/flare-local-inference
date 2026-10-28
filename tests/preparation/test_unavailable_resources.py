"""Placeholders for process-local resources survive the pickling of compiled-forward guards."""

import pickle

import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from ignition.preparation import forward_metadata
from ignition.preparation.forward_metadata import _Unavailable


def test_placeholder_round_trips_through_pickle():
    restored = pickle.loads(pickle.dumps(_Unavailable("RoCE proxy")))
    assert restored.description == "RoCE proxy"
    with pytest.raises(RuntimeError, match="RoCE proxy.post"):
        restored.post


def test_forward_metadata_replaces_cute_runtime_tensors_with_placeholders():
    # b12x bindings such as the TMEM KDA prefill binding keep CuTe DLPack descriptors of their scratch views.
    descriptor = from_dlpack(torch.zeros(4, 128, dtype=torch.bfloat16), assumed_align=16)
    restored = forward_metadata.loads(forward_metadata.dumps({"tmem_tensors": {"qd": descriptor}}))
    placeholder = restored["tmem_tensors"]["qd"]
    assert isinstance(placeholder, _Unavailable)
    with pytest.raises(RuntimeError, match="cutlass.cute.runtime._Tensor.shape"):
        placeholder.shape
