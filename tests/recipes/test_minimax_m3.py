"""MiniMax M3 checkpoint precision, logical pages, and continuation geometry."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes import resolve
from ignition.recipes.minimax_m3.config import ModelConfig
from ignition.recipes.minimax_m3.pool import ContinuationPool, continuation_layout, page_bytes


def configuration():
    return dict(model_type="minimax_m3_vl", architectures=["MiniMaxM3SparseForConditionalGeneration"],
        text_config=dict(hidden_size=256, intermediate_size=128, dense_intermediate_size=256,
            shared_intermediate_size=128, num_hidden_layers=2, num_attention_heads=64, num_key_value_heads=4,
            head_dim=128, rotary_dim=64, partial_rotary_factor=.5,
            vocab_size=512, num_local_experts=4, num_experts_per_tok=2,
            max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e7, swiglu_alpha=1.702,
            swiglu_limit=7., routed_scaling_factor=2., use_gemma_norm=True, use_qk_norm=True,
            qk_norm_type="per_head", attention_output_gate=False, use_routing_bias=True,
            scoring_func="sigmoid", hidden_act="swigluoai", tie_word_embeddings=False,
            n_shared_experts=1, moe_layer_freq=[0, 1], sparse_attention_config=dict(
                use_sparse_attention=True, sparse_index_dim=128, sparse_num_index_heads=4,
                sparse_topk_blocks=16, sparse_block_size=128, sparse_score_type="max",
                sparse_init_block=0, sparse_local_block=1, sparse_attention_freq=[1, 1],
                sparse_disable_index_value=[1, 1])), quantization_config=dict(quant_method="mxfp8",
            weight_block_size=[1, 32], activation_scheme="dynamic", group_size=16,
            moe_quant_algo="NVFP4", moe_quant_format="nvfp4-pack-quantized", ignored_layers=[
                "lm_head", "model.embed_tokens", "language_model.model.layers.*.block_sparse_moe.gate"]))


def test_checkpoint_precision_exclusions_apply_without_layer_allowlists():
    raw = configuration()
    config = ModelConfig.from_dict(raw)
    prefix = "language_model.model.layers.1.block_sparse_moe"
    assert config.projection_format(prefix + ".gate") == "bf16"
    assert config.projection_format(prefix + ".shared_experts.down_proj") == "mxfp8"
    assert config.projection_format(prefix + ".experts.0.w1") == "nvfp4"
    assert config.projection_format("language_model.lm_head") == "bf16"
    raw["quantization_config"]["ignored_layers"].append(prefix + ".shared_experts.*")
    assert ModelConfig.from_dict(raw).projection_format(prefix + ".shared_experts.down_proj") == "bf16"


@pytest.mark.parametrize("field,value", [("sparse_block_size", 64), ("sparse_index_dim", 64),
    ("sparse_num_index_heads", 1), ("sparse_score_type", "mean"), ("sparse_local_block", 2)])
def test_incompatible_sparse_geometry_rejects_before_loading(field, value):
    raw = configuration()
    raw["text_config"]["sparse_attention_config"][field] = value
    with pytest.raises(ValueError, match="indexer geometry"):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("taps", [(1, 0), (0, 0), (-1,), (2,)])
def test_invalid_target_taps_reject_before_allocating_state(taps):
    with pytest.raises(ValueError, match="target taps"):
        ModelConfig.from_dict(configuration(), target_layers=taps)


def test_tp4_full_resolution_cache_and_independent_partial_tails():
    config = ModelConfig.from_dict(configuration(), target_layers=(0, 1))
    shard = TensorParallelShard(0, 4)
    pool = ContinuationPool.for_model(config, shard, 3, 2, "cpu")
    assert page_bytes(config, shard) == 2 * (128 * 128 * 3)
    assert len(pool.pages) == 6
    for layer in range(2):
        key, value, index = (pool.cache(layer, name) for name in ("key", "value", "index"))
        assert key.shape == value.shape == (2, 128, 1, 128)
        assert index.shape == (2, 128, 128)
        assert key.dtype == value.dtype == index.dtype == torch.float8_e4m3fn
        assert key.data_ptr() == pool.pages[f"layer{layer}.key"].data_ptr()
        assert index.data_ptr() == pool.pages[f"layer{layer}.index"].data_ptr()
    tail = pool.layout.region("layer0.index")
    pool.backing[1, tail.offset:tail.offset + tail.nbytes].fill_(11)
    assert not pool.pages["layer0.index"].count_nonzero()
    assert pool.region("target_hidden").shape == (3, 2, 256)
    assert pool.layout.region("last_hidden").offset != pool.layout.region("target_hidden").offset
    assert all(region.offset % 256 == 0 for region in pool.layout.regions)


def test_target_tap_contract_follows_checkpoint_draft_declaration(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(configuration()))
    (tmp_path / "dspark").mkdir()
    (tmp_path / "dspark" / "config.json").write_text(json.dumps({"text_config": {"dspark_target_layer_ids": [0, 1]}}))
    assert ModelConfig.load(tmp_path).target_layers == (0, 1)


def test_rotary_table_and_long_position_angles_match_transformers_exactly():
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding

    raw = configuration()
    raw["text_config"]["max_position_embeddings"] = 1048576
    config = ModelConfig.from_dict(raw)
    canonical = MiniMaxM3VLRotaryEmbedding(config.canonical_config())
    inverse = config.rotary_frequencies()
    assert inverse.shape == (32,)
    assert torch.equal(inverse, canonical.inv_freq)
    positions = torch.tensor([[0, 4095, 65535, 1048575]])
    cosine, sine = canonical(torch.empty(1, dtype=torch.bfloat16), positions)
    frequencies = positions[..., None].float() * inverse
    angles = torch.cat((frequencies, frequencies), -1)
    assert torch.equal(angles.cos().bfloat16(), cosine)
    assert torch.equal(angles.sin().bfloat16(), sine)


def test_recipe_costs_cover_actual_tp4_checkpoint_cache_geometry():
    model = resolve("minimax-m3-1-flash")
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model.checkpoint.repository, "config.json", revision=model.checkpoint.revision)
    snapshot = Path(cached).parent if isinstance(cached, str) else Path(model.checkpoint.repository)
    if not (snapshot / "config.json").is_file():
        pytest.skip("MiniMax M3 checkpoint metadata is unavailable")
    config = ModelConfig.load(snapshot)
    shard = TensorParallelShard(0, 4)
    config.validate_shard(shard)
    assert model.recipe.plain_ranks[4].state_row_bytes == continuation_layout(config, shard).row_bytes
    from ignition.recipes.minimax_m3.draft import DraftConfig

    config = replace(config, draft=DraftConfig.load(snapshot, config))
    layout = continuation_layout(config, shard)
    assert model.recipe.ranks[4].state_row_bytes == layout.row_bytes
    for layer in range(config.draft.layers):
        for kind in ("key", "value"):
            region = layout.region(f"draft{layer}.{kind}")
            assert region.shape == (4096, 1, 128)
            assert region.dtype == torch.float8_e4m3fn
    assert model.recipe.ranks[4].page_bytes == page_bytes(config, shard)
    covered = []
    for rank in range(4):
        first, last, width = config.vocabulary_partition(TensorParallelShard(rank, 4))
        assert width % 64 == 0
        covered.extend(range(first, last))
    assert covered == list(range(config.vocabulary))
    assert model.recipe.sampling_defaults.as_dict() == dict(seed=0, temperature=1., top_p=.95, top_k=-1,
        min_p=0., repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)


def test_packed_tp4_defaults_match_deepseek_dspark_capacities():
    from ignition.recipes.minimax_m3.bootstrap import PreparationConfig
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig as DeepSeekConfig
    from ignition.recipes.minimax_m3.recipe import RECIPE

    config, deepseek = PreparationConfig(), DeepSeekConfig()
    assert config.max_active == RECIPE.concurrency == deepseek.max_active == 32
    assert config.speculative_widths == deepseek.speculative_widths
    assert config.speculative_token_capacity == deepseek.speculative_token_capacity == 128
    assert config.speculative_graph_budget == deepseek.speculative_graph_budget == 64
    assert config.speculative_shapes == deepseek.speculative_shapes
    assert config.packed_speculative_shapes == deepseek.packed_speculative_shapes
    assert config.packed_verification and config.draft_prefix_selection == "gpu"
    assert RECIPE.speculation.draft_tokens == 5
    assert RECIPE.checkpoints[0].repository == "MiniMaxAI/MiniMax-M3.1-Flash"
    assert PreparationConfig(dspark=False).speculative_shapes == ()
    assert PreparationConfig(dspark=False).packed_speculative_shapes == ()


def test_rectangular_verification_is_retained_for_explicit_diagnostics():
    from ignition.recipes.minimax_m3.bootstrap import PreparationConfig

    config = PreparationConfig(max_active=4, speculative_widths=tuple(range(1, 9)),
        speculative_token_capacity=32, diagnostic_graphs=("speculative4x8",), speculative_fixed_width=8)
    assert config.packed_speculative_shapes == ()
    assert set(config.speculative_shapes) == {(batch, width) for batch in (1, 2, 4) for width in range(1, 9)}


@pytest.mark.parametrize("options", [dict(packed_verification=False), dict(draft_prefix_selection="cpu"),
                                    dict(speculative_fixed_width=6)])
def test_serving_dspark_requires_adaptive_packed_gpu_rounds(options):
    from ignition.recipes.minimax_m3.bootstrap import PreparationConfig

    with pytest.raises(ValueError, match="adaptive packed GPU"):
        PreparationConfig(**options)


def test_multihost_placements_keep_the_complete_tp4_catalog():
    from ignition.recipes.minimax_m3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=(0, 0, 0, 0), hosts=("rank0", "rank1", "rank2", "rank3"),
        max_active=4, prefill_chunk_size=4096, image_concurrency=4, speculative_token_capacity=32)
    assert config.image_concurrency == 4
    assert set(config.speculative_shapes) == {(batch, 1) for batch in (1, 2, 4)}
    assert config.packed_speculative_shapes
    with pytest.raises(ValueError, match="placements"):
        replace(config, hosts=("rank0", "rank1", "rank1", "rank3"))
    with pytest.raises(ValueError, match="placements"):
        replace(config, hosts=None)
    with pytest.raises(ValueError, match="separate hosts"):
        PreparationConfig(roce_ring=True)


def test_recipe_routes_multihost_preparation_through_shared_rank_group(monkeypatch):
    from types import SimpleNamespace
    from ignition.recipes.minimax_m3.recipe import RECIPE, prepare
    import ignition.preparation.rank_group as rank_group

    rendezvous = SimpleNamespace(hosts=("rank0", "rank1", "rank2", "rank3"))
    plan = SimpleNamespace(speculation=RECIPE.speculation, compile_workers=2, autotune_effort="audit", tuning_replica=None,
        compile_headroom_bytes=6 << 30,
        native_dense_activations=True,
        image_concurrency=4, language_model_only=False, world_size=4, roce_ring=False,
        context=1048576, concurrency=4, prefill_chunk=4096, snapshots=4,
        costs=RECIPE.ranks[4], headroom_bytes=8 << 30, reserve_bytes=4 << 30,
        stream_overlap=True, expert_scale_compression=None, speculative_token_capacity=32,
        checkpoint=RECIPE.checkpoints[0])
    captured = {}
    def prepare_group(snapshot, config, prepare_rank, **kwargs):
        captured.update(snapshot=snapshot, config=config, **kwargs)
        return captured
    monkeypatch.setattr(rank_group, "prepare_group", prepare_group)
    result = prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None,
                     progress=None, ranks=rendezvous)
    assert result["ranks"] is rendezvous
    assert result["checkpoint"] is plan.checkpoint
    assert result["config"].devices == (0, 0, 0, 0)
    assert result["config"].hosts == rendezvous.hosts
    assert result["config"].vision and result["config"].dspark
    assert result["config"].prefill_chunk_size == 4096
    assert result["config"].speculative_token_capacity == 32
    assert result["config"].autotune_effort == "audit"
    assert result["config"].compile_headroom_bytes == 6 << 30
    assert result["config"].native_dense_activations
    assert result["config"].packed_verification
    assert result["config"].draft_prefix_selection == "gpu"
    assert result["config"].speculative_graph_budget == 64


@pytest.mark.parametrize("vision", [False, True])
def test_contract_preserves_cache_planes_and_complete_round_bindings(vision):
    from google.protobuf.json_format import ParseDict
    from ignition.preparation import contract as c, runtime_package_pb2 as p
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.minimax_m3.contract import prepare
    from ignition.recipes.minimax_m3.frontend import FRONTEND
    from types import SimpleNamespace

    pool = ContinuationPool.for_model(ModelConfig.from_dict(configuration()), TensorParallelShard(0, 4), 3, 2, "cpu")
    buffers = {"proposal_confidence": torch.empty((2, 4, 7))}
    signatures = {"speculative": GraphSignature("speculative", 4, 8,
        tuple(c.text_inputs("speculative")), speculation_enabled=True)}
    signatures["ragged"] = GraphSignature("speculative", 4, 6,
        tuple(field for field in c.text_inputs("speculative") if field != "query_lengths") + ("speculative_prefix_limits",),
        speculation_enabled=True, packed_token_capacity=10, draft_tokens=5)
    if vision:
        signatures["vision"] = GraphSignature("vision", 1, 256, ("vision_pixel_values", "vision_slot"))
    value = ParseDict(prepare(signatures, buffers, pool, vocabulary=512, frontend=FRONTEND,
        dspark=True), p.ModelContract())
    assert value.cache.page_tokens == 128
    assert len(value.cache.regions) == 7
    assert sum(region.kind == p.PAGED_LATENT for region in value.cache.regions) == 2
    for graph in value.graphs:
        assert all(not state.view.buffer.startswith("proposal_") for state in graph.state)
        if graph.kind == p.SPECULATIVE:
            assert graph.draft_cursor.lag_tokens == 0
            assert graph.state_rows_per_request == 1
        if graph.graph == "ragged":
            sources = {binding.buffer: binding.source for binding in graph.inputs}
            assert sources["speculative_prefix_limits"] == p.SPECULATIVE_PREFIX_LIMIT
            assert p.QUERY_LENGTH not in sources.values()
    assert value.vision.max_images == (4 if vision else 0)
    assert value.vision.patches_per_feature == (4 if vision else 0)


@pytest.mark.parametrize("world", [1, 2, 8, 16])
def test_unsupported_tensor_parallel_geometry_rejects(world):
    with pytest.raises(ValueError, match="parallelism of four"):
        ModelConfig.from_dict(configuration()).validate_shard(TensorParallelShard(0, world))
