"""Prepared GPU prefix allocation must match the scalar frontier oracle exactly."""

import os
import random

import pytest
import torch

from ignition.components.causal_prefix import PreparedCausalPrefixAllocation
from tests.components.causal_prefix_oracle import allocate

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch,depth", [(1, 1), (1, 7), (8, 3), (32, 1), (32, 5), (64, 7)])
@torch.no_grad()
def test_eager_captured_prefix_oracle_and_allocation_stability(batch, depth):
    rng = random.Random(batch * 100 + depth)
    # Match a sliced host-mapped confidence plane with a larger retained depth.
    backing = torch.empty((batch, depth + 3), dtype=torch.float32, device="cuda")
    confidence = backing[:, :depth]
    slots = torch.empty(batch, dtype=torch.int64, device="cuda")
    submitted = torch.empty(batch, dtype=torch.bool, device="cuda")
    source_valid = torch.empty_like(submitted)
    limits = torch.empty(batch, dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(limits)
    valid = torch.empty_like(submitted)
    budget = 128 if (batch, depth) == (32, 1) else batch * (depth + 1) // 2
    operation = PreparedCausalPrefixAllocation(confidence, slots, submitted, source_valid, limits,
                                               lengths, valid, token_capacity=budget, request_capacity=64)
    confidence.fill_(1.)
    slots.copy_(torch.arange(batch, device="cuda"))
    submitted.fill_(True)
    source_valid.fill_(True)
    limits.fill_(depth + 1)
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    values = [0., -0., 1., .5, 2. ** -149, 2. ** -126, 1. - 2. ** -24]
    for iteration in range(40):
        probabilities = [[rng.choice(values + [rng.random()]) for _ in range(depth)] for _ in range(batch)]
        identities = rng.sample(range(64), batch)
        active = [rng.random() > .2 for _ in range(batch)]
        checked = [True] * batch
        caps = [rng.randint(1, depth + 1) for _ in range(batch)]
        if iteration % 7 == 0:
            active[0] = True
            probabilities[0][-1] = float("nan")
        elif iteration % 7 == 1:
            active[0] = True
            checked[0] = False
        elif iteration % 7 == 2 and batch > 1:
            active[:2] = [True, True]
            identities[1] = identities[0]
        confidence.copy_(torch.tensor(probabilities, dtype=torch.float32, device="cuda"))
        slots.copy_(torch.tensor(identities, device="cuda"))
        submitted.copy_(torch.tensor(active, device="cuda"))
        source_valid.copy_(torch.tensor(checked, device="cuda"))
        limits.copy_(torch.tensor(caps, dtype=torch.int32, device="cuda"))
        lengths.fill_(-111)
        valid.fill_(True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        assert (lengths.tolist(), valid.tolist()) == allocate(confidence.cpu().tolist(), identities,
                                                               active, checked, caps, budget, 64)
    graph.reset()


@torch.no_grad()
def test_signed_zero_frontier_order_and_fp64_underflow_survival():
    confidence = torch.tensor([[-0.] * 7, [0.] * 7, [2. ** -149] * 7], device="cuda")
    slots = torch.tensor([0, 1, 2], device="cuda")
    submitted = torch.ones(3, dtype=torch.bool, device="cuda")
    limits = torch.full((3,), 8, dtype=torch.int32, device="cuda")
    lengths = torch.empty_like(limits)
    valid = torch.empty_like(submitted)
    for budget in range(3, 25):
        operation = PreparedCausalPrefixAllocation(confidence, slots, submitted, submitted, limits,
            lengths, valid, token_capacity=budget, request_capacity=3)
        operation()
        assert (lengths.tolist(), valid.tolist()) == allocate(confidence.tolist(), slots.tolist(),
            submitted.tolist(), submitted.tolist(), limits.tolist(), budget, 3)
