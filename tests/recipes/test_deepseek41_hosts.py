"""DeepSeek TP4 host placement and channels preserve the local preparation contract."""

from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.recipes.deepseek41 import bootstrap, recipe


def test_four_hosts_may_each_use_their_own_gpu_zero():
    config = bootstrap.PreparationConfig(devices=(0,) * 4, hosts=("hertz", "faraday", "ampere", "maxwell"))
    assert config.devices == (0,) * 4
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 4)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 4, hosts=("hertz",) * 4)


@pytest.mark.parametrize("draft_tokens", [5, 7])
@pytest.mark.parametrize("remote", [False, True])
def test_recipe_dispatches_the_complete_plan_to_the_rank_group(monkeypatch, remote, draft_tokens):
    from ignition.preparation import rank_group
    from ignition.recipes.declaration import Speculation

    gpus = tuple(planning.Gpu(i, "GB10", f"GPU-{i}", 128 << 30, 120 << 30, (12, 1)) for i in range(4))
    plan = planning.fit(recipe.RECIPE, recipe.RECIPE.checkpoints[0], gpus, concurrency=16,
        reserve_bytes=6 << 30, speculation=Speculation(draft_tokens=draft_tokens, token_capacity=128,
            packed_verification=True, draft_prefix_selection="gpu"),
        expert_scale_compression="csf")
    ranks = SimpleNamespace(hosts=("hertz", "faraday", "ampere", "maxwell")) if remote else None
    calls, result, storage = [], object(), object()
    monkeypatch.setattr(rank_group, "prepare_group", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None, progress=None,
                          package_storage=storage, ranks=ranks) is result
    (snapshot, config, forward), kwargs = calls[0]
    assert snapshot == "checkpoint" and forward is bootstrap.prepare_rank
    assert config.devices == ((0,) * 4 if remote else (0, 1, 2, 3))
    assert config.hosts == (ranks.hosts if remote else None)
    assert config.max_active == 16 and config.draft_tokens == draft_tokens
    assert (16, 16 * (draft_tokens + 1)) in config.packed_speculative_shapes
    assert config.expert_scale_compression == "csf" and config.split_speculation
    assert kwargs["ranks"] is ranks and kwargs["checkpoint"] is plan.checkpoint
    assert kwargs["package_storage"] is storage


@pytest.mark.parametrize("draft_tokens", [0, 5, 7])
def test_remote_channels_cover_vocab_and_sampling_results_without_pcie(monkeypatch, draft_tokens):
    config = bootstrap.PreparationConfig(devices=(0,) * 4, hosts=("a", "b", "c", "d"), max_active=16,
        dspark=draft_tokens > 0, speculative_widths=tuple(range(1, draft_tokens + 2)),
        speculative_token_capacity=128)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    for name in ("create_all_reduce_runtime", "create_all_gather_runtime", "create_dma_all_reduce_runtime",
                 "create_result_broadcast_runtime"):
        monkeypatch.setattr(bootstrap, name, lambda *args, **kwargs: pytest.fail("host ranks cannot use CUDA IPC"))
    channels = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=True), package_storage=storage,
        verification_samples=config.decode_tokens)
    assert channels == (runtime, runtime, runtime, runtime if draft_tokens else None)
    assert calls[0]["package_storage"] is storage
    assert calls[0]["pipe_tokens"] == 4096
    assert calls[0]["max_tokens"] >= config.decode_tokens
    sample_rows = max(config.decode_tokens, config.request_capacity * draft_tokens)
    assert calls[0]["gather_bytes"] >= sample_rows * (129280 // 4) * 2
    if draft_tokens:
        from ignition.components.collectives import packed_broadcast_layout
        from ignition.components.draft_sampling import PreparedDraftDistribution
        from ignition.components.speculation import SpeculativeAcceptance

        layouts = (PreparedDraftDistribution.broadcast_layout(16),
                   SpeculativeAcceptance.broadcast_layout(16, draft_tokens + 1))
        assert calls[0]["gather_bytes"] >= max(packed_broadcast_layout(layout.values())[1] for layout in layouts)
