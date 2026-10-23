"""GLM graph catalog capacities and rank geometry before loading weights."""

import pytest

from ignition.recipes.glm53.bootstrap import PreparationConfig


def test_default_catalog_covers_4096_prefill_and_four_requests():
    config = PreparationConfig()
    assert config.prefill_capacities == (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    assert config.batch_capacities == (4, 2, 1)
    assert config.page_capacity == 4 * (262144 // 256)
    assert PreparationConfig(prefill_chunk_size=3072).prefill_capacities[-2:] == (2560, 3072)
    assert PreparationConfig(max_active=1, pages=64).batch_capacities == (1,)
    assert PreparationConfig(max_active=3).batch_capacities == (4, 2, 1)
    assert PreparationConfig(max_active=1).image_concurrency == 1
    assert PreparationConfig(max_active=32).image_concurrency == 4
    assert PreparationConfig(max_active=32, image_concurrency=2).image_concurrency == 2


@pytest.mark.parametrize("values", [
    {"devices": ()}, {"devices": (0, 0)}, {"devices": (0, 1, 2)}, {"devices": (-1, 0)},
    {"context": 513}, {"context": 2097152}, {"prefill_chunk_size": 4097},
    {"context": 256, "prefill_chunk_size": 512}, {"max_active": 65}, {"pages": 0},
    {"snapshot_budget_bytes": 0}, {"headroom_bytes": -1}, {"expert_activation": "fp8"},
])
def test_invalid_prepared_geometry_rejects_before_gpu_use(values):
    with pytest.raises((ValueError, TypeError)):
        PreparationConfig(**values)


def test_mtp_catalog_defaults_to_five_drafts_and_reserves_draft_pages():
    config = PreparationConfig(mtp=True)
    assert config.speculative_widths == (1, 2, 3, 4, 5, 6)
    assert config.speculative_shapes == tuple((b, w) for b in (1, 2, 4) for w in range(6, 0, -1) if w == 1 or b * w <= 6)
    assert config.page_capacity == PreparationConfig().page_capacity + 4
    fixed = PreparationConfig(mtp=True, speculative_fixed_width=6)
    assert fixed.speculative_shapes == config.speculative_shapes
    large = PreparationConfig(mtp=True, max_active=64)
    assert len(large.speculative_shapes) == 31
    assert large.speculative_shapes[-1] == (64, 1)
