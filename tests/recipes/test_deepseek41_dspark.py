"""DSpark capacity declarations and packed verification coordinates."""

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.recipes.deepseek41.config import DSparkConfig
from ignition.recipes.deepseek41.recipe import RECIPE


def test_dspark_defaults_to_five_drafts_with_adaptive_verification():
    config = PreparationConfig(max_active=4)
    assert config.speculative_widths == (1, 2, 3, 4, 5, 6)
    assert config.draft_tokens == 5 and config.decode_tokens == 24
    assert config.split_speculation and config.packed_verification
    assert config.draft_prefix_selection == "gpu"
    assert RECIPE.speculation.draft_tokens == 5 and RECIPE.max_draft_tokens == 7
    assert RECIPE.speculation.adaptive and not RECIPE.history_copy
    assert PreparationConfig(dspark=False).speculative_shapes == ()
    assert PreparationConfig(dspark=False).draft_tokens == 0
    with pytest.raises(ValueError):
        PreparationConfig(speculative_widths=(1, 9))


@pytest.mark.parametrize("width", range(1, 9))
def test_explicit_draft_limit_sizes_the_parallel_block(width):
    config = PreparationConfig(max_active=4, speculative_widths=tuple(range(1, width + 1)))
    assert config.draft_tokens == width - 1
    assert config.decode_tokens == 4 * width
    assert config.packed_verification == (width > 1)


def test_native_dspark_geometry():
    text = dict(num_nextn_predict_layers=3, dspark_block_size=5, dspark_noise_token_id=128799,
        dspark_target_layer_ids=[37, 38, 39], dspark_markov_rank=256, dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3, num_hidden_layers=40, vocab_size=129280, compress_ratios=[0] * 43)
    assert DSparkConfig.from_text(text).target_layers == (37, 38, 39)
    assert DSparkConfig.from_text({**text, "dspark_block_size": 7}).block_size == 7
    with pytest.raises(ValueError):
        DSparkConfig.from_text({**text, "dspark_target_layer_ids": [39, 37, 38]})


def test_verification_packs_partial_and_inactive_requests_without_crossing_sequences():
    metadata = Metadata.allocate(24, 4, 8, "cpu", samples=24)
    tokens = torch.arange(24).view(4, 6) + 100
    pages = torch.arange(32, dtype=torch.int32).view(4, 8)
    metadata.set(torch.tensor([127, 80, 200, 4]), torch.tensor([3, 0, 1, 5]),
        torch.tensor([3, 4, 5, 6]), pages, tokens, torch.tensor([4]))
    assert metadata.live_counts.tolist() == [9, 4]
    assert metadata.query_start_loc.tolist() == [0, 3, 3, 4, 9]
    assert metadata.token_ids[:9].tolist() == [100, 101, 102, 112, 118, 119, 120, 121, 122]
    assert metadata.positions[:9].tolist() == [127, 128, 129, 200, 4, 5, 6, 7, 8]
    assert metadata.positions[9:].eq(-1).all()
    assert metadata.state_rows.tolist() == [3, 0, 5, 6]
    assert metadata.sample_indices.view(4, 6)[:, 0].tolist() == [0, 3, 3, 4]


def test_single_token_verification_packs_page_tables_across_inactive_rows():
    metadata = Metadata.allocate(4, 4, 8, "cpu")
    pages = torch.arange(32, dtype=torch.int32).view(4, 8)
    metadata.set(torch.tensor([127, 80, 200, 4]), torch.tensor([1, 0, 1, 0]),
        torch.tensor([3, 4, 5, 6]), pages, torch.tensor([[10], [11], [12], [13]]), torch.tensor([4]))
    assert metadata.token_ids.tolist() == [10, 12, 0, 0]
    assert metadata.positions.tolist() == [127, 200, -1, -1]
    assert torch.equal(metadata.token_page_table[:2], pages[[0, 2]])
    assert torch.equal(metadata.page_table, pages)


@pytest.mark.parametrize("device_pending", [False, True])
def test_decode_page_tables_follow_packed_tokens_across_inactive_rows(device_pending):
    from types import SimpleNamespace
    from ignition.components.paged_batch import BatchInputs, PreparedBatch

    inputs = BatchInputs.allocate(4, 8, "cpu", capacity=4)
    inputs.row_slot_ids.copy_(torch.tensor([3, 1, 0, 2]))
    inputs.live_rows.fill_(4)
    inputs.token_ids.copy_(torch.tensor([10, 11, 12, 13]))
    inputs.request_state_rows[:, 0].copy_(torch.tensor([1, 2, 3, 4]))
    inputs.page_tables.copy_(torch.arange(32).view(4, 8))
    batch = PreparedBatch.__new__(PreparedBatch)
    batch.inputs = inputs
    batch.requests = SimpleNamespace(capacity=4, status=torch.ones(4, dtype=torch.int32),
        error_code=torch.zeros(4, dtype=torch.int32))
    batch.metadata = Metadata.allocate(4, 4, 8, "cpu")
    batch.token_state = SimpleNamespace(gather=lambda: None, active=torch.tensor([1, 0, 1, 0]),
        positions=torch.tensor([127, 80, 200, 4]), pending_tokens=torch.tensor([20, 21, 22, 23]))
    batch.device_pending = device_pending
    batch.shard = SimpleNamespace(world_size=1 if device_pending else 4)
    batch.rows, batch.tokens, batch.mode = 4, 4, "decode"
    batch.present = torch.ones(4, dtype=torch.bool)
    batch.token_offsets = torch.arange(4)
    batch.begin(torch.tensor([[0, 0, 0, 0] + [batch.shard.world_size] * 4]))
    assert batch.metadata.token_ids.tolist() == ([20, 22, 0, 0] if device_pending else [10, 12, 0, 0])
    assert batch.metadata.positions.tolist() == [127, 200, -1, -1]
    assert torch.equal(batch.metadata.token_page_table[:2], inputs.page_tables[[3, 0]])
