"""Exact scalar cases for budget, total ordering, and invalid cohorts."""

import random

import pytest

from tests.components.causal_prefix_oracle import allocate


def test_zero_confidence_spends_budget_and_physical_slots_break_ties():
    assert allocate([[0.] * 3] * 2, [9, 2], [True] * 2, [True] * 2, [4] * 2, 5, 10)[0] == [2, 3]
    assert allocate([[-0.], [0.]], [0, 1], [True] * 2, [True] * 2, [2] * 2, 3, 2)[0] == [1, 2]


def test_pending_only_limits_and_unused_budget():
    assert allocate([[1.] * 3] * 2, [0, 1], [True] * 2, [True] * 2, [1, 2], 8, 2)[0] == [1, 2]
    assert allocate([[float("nan")]], [-1], [False], [False], [-99], 0, 1) == ([0], [False])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_invalid_confidence_beyond_admitted_cap_fails_whole_cohort(bad):
    assert allocate([[1., bad], [1., 1.]], [0, 1], [True] * 2, [True] * 2, [1, 3], 5, 2) == ([0, 0], [False, False])


@pytest.mark.parametrize("slots,valid,limits,budget", [([0, 0], [True, True], [2, 2], 4),
    ([0, 2], [True, True], [2, 2], 4), ([0, 1], [True, False], [2, 2], 4),
    ([0, 1], [True, True], [0, 2], 4), ([0, 1], [True, True], [2, 3], 4),
    ([0, 1], [True, True], [2, 2], 1)])
def test_invalid_identity_bounds_or_pending_budget_fails(slots, valid, limits, budget):
    assert allocate([[1.], [1.]], slots, [True] * 2, valid, limits, budget, 2) == ([0, 0], [False, False])


def test_slot_permutation_preserves_allocation_and_capacity():
    rng = random.Random(197)
    for _ in range(200):
        batch, depth = 8, 7
        confidence = [[rng.choice([0., -0., 1., .5, rng.random()]) for _ in range(depth)] for _ in range(batch)]
        slots = rng.sample(range(64), batch)
        submitted = [rng.choice([False, True]) for _ in range(batch)]
        limits = [rng.randint(1, depth + 1) for _ in range(batch)]
        budget = rng.randint(sum(submitted), batch * (depth + 1))
        lengths, valid = allocate(confidence, slots, submitted, [True] * batch, limits, budget, 64)
        permutation = rng.sample(range(batch), batch)
        moved = allocate([confidence[i] for i in permutation], [slots[i] for i in permutation],
                         [submitted[i] for i in permutation], [True] * batch,
                         [limits[i] for i in permutation], budget, 64)
        assert moved == ([lengths[i] for i in permutation], [valid[i] for i in permutation])
        assert sum(lengths) == min(budget, sum(limits[i] for i in range(batch) if submitted[i]))


def test_physical_budget_can_exceed_shallow_proposal_capacity():
    assert allocate([[1.]] * 32, list(range(32)), [True] * 32, [True] * 32,
                    [2] * 32, 128, 32) == ([2] * 32, [True] * 32)


def test_constructor_cuda_depth_bounds_accept_fake_cuda():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from ignition.components.causal_prefix import PreparedCausalPrefixAllocation

    def build(device, depth):
        confidence = torch.empty((2, depth), dtype=torch.float32, device=device)
        return PreparedCausalPrefixAllocation(confidence,
            torch.empty(2, dtype=torch.int64, device=device),
            torch.empty(2, dtype=torch.bool, device=device),
            torch.empty(2, dtype=torch.bool, device=device),
            torch.empty(2, dtype=torch.int32, device=device),
            torch.empty(2, dtype=torch.int32, device=device),
            torch.empty(2, dtype=torch.bool, device=device), token_capacity=16, request_capacity=2)

    with pytest.raises(ValueError, match="CUDA"):
        build("cpu", 1)
    with FakeTensorMode():
        build("cuda", 1)
        build("cuda", 7)
        for depth in (0, 8):
            with pytest.raises(ValueError, match="depth"):
                build("cuda", depth)
