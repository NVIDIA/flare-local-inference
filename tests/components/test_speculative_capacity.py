"""The same target-token envelope governs every speculative recipe."""

import pytest

from ignition.components.capacity import batch_capacities, proposal_shapes, speculative_shapes
from ignition.recipes.qwen38.bootstrap import PreparationConfig as Qwen
from ignition.recipes.glm53.bootstrap import PreparationConfig as Glm
from ignition.recipes.deepseek4.bootstrap import PreparationConfig as DeepSeek4
from ignition.recipes.deepseek41.bootstrap import PreparationConfig as DeepSeek41
from ignition.recipes.mimo_v2.bootstrap import PreparationConfig as Mimo
from ignition.recipes.step5.bootstrap import PreparationConfig as Step5


@pytest.mark.parametrize("concurrency", [1, 3, 4, 5, 8, 16, 33, 64])
@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("override", [None, 8, 32])
def test_recipes_share_bounded_target_coverage(concurrency, drafts, override):
    widths = tuple(range(1, drafts + 2))
    capacity = max(batch_capacities(concurrency)[-1], drafts + 1) if override is None else override
    expected = {(b, w) for b in batch_capacities(concurrency) for w in widths
                if w == 1 or b * w <= capacity}
    for recipe, enabled in ((Qwen, "mtp"), (Glm, "mtp"), (DeepSeek4, "dspark"),
                            (Mimo, "dflash")):
        config = recipe(max_active=concurrency, speculative_widths=widths,
                        speculative_token_capacity=override, **{enabled: True})
        assert set(config.speculative_shapes) == expected
    config = DeepSeek41(max_active=concurrency, speculative_widths=widths,
                       speculative_token_capacity=override)
    assert set(config.speculative_shapes) == {(b, 1) for b in batch_capacities(concurrency)}
    assert all(b < p <= min(capacity, b * (drafts + 1)) for b, p in config.packed_speculative_shapes)
    if drafts <= 3:
        config = Step5(max_active=concurrency, speculative_widths=widths,
                       speculative_token_capacity=override)
        assert set(config.speculative_shapes) == expected
    assert (1, drafts + 1) in expected
    assert all((b, 1) in expected for b in batch_capacities(concurrency))


def test_proposals_keep_full_depth_independently_of_target_coverage():
    assert len(proposal_shapes(4, 7)) == 3
    assert len(speculative_shapes(4, tuple(range(1, 9)), 8)) == 14
    assert set(proposal_shapes(4, 7)) == {(b, 7) for b in (1, 2, 4)}
    assert {w for b, w in speculative_shapes(4, tuple(range(1, 9)), 8) if b == 4} == {1, 2}


@pytest.mark.parametrize("capacity", [0, -1, 7, True, 8.0])
def test_insufficient_or_noninteger_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        speculative_shapes(4, tuple(range(1, 9)), capacity)
