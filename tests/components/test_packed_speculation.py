"""Prepared round metadata must outlive both sequential sampling workspaces."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.packed_speculation import PreparedPackedSpeculativeRound
from ignition.preparation.scratch import GraphBufferPool


def operands(pool=None):
    b = 4
    owner = SimpleNamespace(inputs=SimpleNamespace(query_lengths=torch.zeros(b, dtype=torch.int32)),
        requests=SimpleNamespace(capacity=b), weights=SimpleNamespace(shard=SimpleNamespace(world_size=1)),
        collectives=dict(runtime=None, vocabulary_runtime=None, coordinator=None), session=None)
    source = SimpleNamespace(batch=b, width=3, owner=owner, active=torch.zeros(b, dtype=torch.bool),
        rows=torch.arange(b), buffers=None if pool is None else pool.cursor(),
        validator=SimpleNamespace(validation_input=torch.zeros((1, 8), dtype=torch.bfloat16)))
    target = SimpleNamespace(batch=b, width=6, owner=owner, target_token_capacity=8,
        buffers=None if pool is None else pool.cursor())
    storage = object()
    return SimpleNamespace(source=source, storage=storage), SimpleNamespace(target=target, storage=storage)


@pytest.mark.parametrize("independent_descriptor", [False, True])
def test_prefix_limits_cannot_alias_mutable_query_lengths(independent_descriptor):
    proposal, verification = operands()
    backing = torch.zeros(5, dtype=torch.int32)
    limits = backing[:4]
    queries = backing[1:]
    if independent_descriptor:
        queries = torch.empty(0, dtype=torch.int32).set_(backing.untyped_storage()[4:], 0, (4,), (1,))
        assert queries.untyped_storage()._cdata != limits.untyped_storage()._cdata
    proposal.source.owner.inputs.query_lengths = queries
    with pytest.raises(ValueError, match="must not overlap"):
        PreparedPackedSpeculativeRound(proposal, verification, limits, name="round")


def test_round_cursor_reserves_both_sampling_lifetimes_and_reuses_sequential_storage(monkeypatch):
    # This test exercises host storage planning only. Numerical CUDA allocation
    # is covered by the eager/captured component oracle on an assigned device.
    monkeypatch.setattr("ignition.components.packed_speculation.PreparedCausalPrefixAllocation",
                        lambda *args, **kwargs: None)
    pool = GraphBufferPool("cpu", chunk_bytes=4096)
    proposal, verification = operands(pool)
    scratch = []
    for cursor, count in ((proposal.source.buffers, 129), (verification.target.buffers, 513)):
        for dtype in (torch.float32, torch.bool, torch.int64, torch.bfloat16):
            scratch.append(cursor.zeros((count,), dtype))
    limits = torch.full((4,), 3, dtype=torch.int32)
    round_ = PreparedPackedSpeculativeRound(proposal, verification, limits, name="round", buffers=pool.cursor())
    fields = ("submitted", "source_valid", "admitted", "slots", "confidence", "validation_input", "validation_sum")
    retained = [getattr(round_, field) for field in fields]
    for value in retained:
        value.fill_(1)
    snapshots = [value.clone() for value in retained]
    for value in scratch:
        value.fill_(0)
    assert all(torch.equal(value, before) for value, before in zip(retained, snapshots))
    for i, left in enumerate(retained):
        for right in retained[i + 1:] + scratch:
            assert not (left.data_ptr() < right.data_ptr() + right.nbytes
                        and right.data_ptr() < left.data_ptr() + left.nbytes)
    allocated = pool.nbytes
    another = PreparedPackedSpeculativeRound(proposal, verification, limits, name="another", buffers=pool.cursor())
    assert pool.nbytes == allocated
    assert all(getattr(another, field).data_ptr() == getattr(round_, field).data_ptr() for field in fields)
