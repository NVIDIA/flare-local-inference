"""Qwen3.8-27B DFlash2 geometry, precision, and complete continuation storage."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve

DRAFT_REPOSITORY = "incoai/Qwen3.8-27B-DFlash2"
DRAFT_REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"
DRAFT_CONFIG = {'architectures': ['DFlash2DraftModel'],
 'attention_bias': False,
 'attention_dropout': 0.0,
 'bos_token_id': None,
 'is_causal': False,
 'dflash_config': {'block_size': 8,
                   'conv_group_size': 16,
                   'conv_kernel_size': 2,
                   'mask_token_id': 248070,
                   'selector_rank': 256,
                   'selector_top_k': 16,
                   'target_layer_ids': [5, 19, 33, 47, 61]},
 'dtype': 'bfloat16',
 'eos_token_id': 248044,
 'head_dim': 128,
 'hidden_act': 'silu',
 'hidden_size': 5120,
 'initializer_range': 0.02,
 'intermediate_size': 17408,
 'layer_types': ['sliding_attention',
                 'sliding_attention',
                 'sliding_attention',
                 'sliding_attention',
                 'sliding_attention'],
 'max_position_embeddings': 262144,
 'max_window_layers': 5,
 'model_type': 'qwen3',
 'num_attention_heads': 32,
 'num_hidden_layers': 5,
 'num_key_value_heads': 8,
 'num_target_layers': 64,
 'pad_token_id': 248044,
 'rms_norm_eps': 1e-06,
 'rope_parameters': {'rope_theta': 10000000, 'rope_type': 'default'},
 'sliding_window': 2048,
 'tie_word_embeddings': False,
 'transformers_version': '5.15.0',
 'use_cache': True,
 'use_sliding_window': True,
 'vocab_size': 248320}


@pytest.fixture
def target():
    from ignition.recipes.qwen38_dense.config import ModelConfig
    text = dict(hidden_size=5120, vocab_size=248320, num_hidden_layers=64,
        num_key_value_heads=4, head_dim=256,
        layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)])
    return ModelConfig(text, {})


@pytest.fixture
def config(target):
    from ignition.recipes.qwen38_dense.draft_config import DraftConfig
    return replace(target, draft=DraftConfig.from_dict(copy.deepcopy(DRAFT_CONFIG), target))


def test_pinned_draft_geometry_and_raw_logits(config):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38_dense.draft_config import REPOSITORY, REVISION
    assert (REPOSITORY, REVISION) == (DRAFT_REPOSITORY, DRAFT_REVISION)
    draft = config.draft
    assert (draft.layers, draft.block_size, draft.noise_token) == (5, 8, 248070)
    assert draft.target_layers == (5, 19, 33, 47, 61)
    assert (draft.attention.heads, draft.attention.kv_heads, draft.attention.key_dim) == (32, 8, 128)
    assert (draft.attention.window, draft.attention.theta, draft.attention.rotary_dim) == (2048, 10000000, 128)
    assert not {"output_multiplier", "final_logit_softcapping"} & draft.text["dflash_config"].keys()
    draft.validate_shard(TensorParallelShard(0, 1))


@pytest.mark.parametrize("key,value", [
    ("hidden_size", 4096), ("num_attention_heads", 40), ("num_key_value_heads", 4),
    ("head_dim", 256), ("intermediate_size", 16384), ("vocab_size", 248321),
    ("dtype", "float32"), ("is_causal", True), ("sliding_window", 4096),
    ("num_target_layers", 63), ("rope_parameters", {"rope_type": "default", "rope_theta": 500000}),
])
def test_incompatible_draft_decoder_rejected(target, key, value):
    from ignition.recipes.qwen38_dense.draft_config import DraftConfig
    raw = copy.deepcopy(DRAFT_CONFIG)
    raw[key] = value
    with pytest.raises(ValueError):
        DraftConfig.from_dict(raw, target)


@pytest.mark.parametrize("key,value", [
    ("target_layer_ids", [5, 19, 33, 47, 62]), ("mask_token_id", 0),
    ("selector_top_k", 8), ("selector_rank", 128), ("conv_group_size", 32),
    ("conv_kernel_size", 3), ("block_size", 16),
])
def test_incompatible_draft_projector_rejected(target, key, value):
    from ignition.recipes.qwen38_dense.draft_config import DraftConfig
    raw = copy.deepcopy(DRAFT_CONFIG)
    raw["dflash_config"][key] = value
    with pytest.raises(ValueError):
        DraftConfig.from_dict(raw, target)


def test_complete_ring_and_target_page_geometry(config):
    from ignition.recipes.qwen38_dense.pool import ContinuationPool, continuation_layout, page_bytes
    recipe = resolve("qwen3.8-27b").recipe
    for drafting, expected_bytes in ((False, 158148864), (True, 186001664)):
        pool = ContinuationPool.for_model(config, 3, 2, "meta", mtp=drafting)
        assert continuation_layout(config, mtp=drafting).row_bytes == expected_bytes
        assert pool.layout.row_bytes == recipe.costs(1, recipe.speculation if drafting else None).state_row_bytes
        assert page_bytes(config, mtp=drafting) == recipe.costs(1, recipe.speculation if drafting else None).page_bytes == 4194304
        assert len(pool.pages) == 32
        assert all(not name.startswith("draft") for name in pool.pages)
        rings = [region for region in pool.layout.regions if region.name.startswith("draft")]
        assert len(rings) == (10 if drafting else 0)
        for region in rings:
            tensor = pool.reserved_region(region.name)
            assert tensor.shape == (3, 2048, 8, 128)
            assert tensor.dtype == torch.float8_e4m3fn
        for index in range(64):
            if index % 4 != 3:
                assert pool.reserved_region(f"layer{index}.convolution").shape[-1] == (10 if drafting else 3)


def test_parallel_speculative_catalog_and_explicit_ordinary_mode():
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig
    recipe = resolve("qwen3.8-27b").recipe
    assert recipe.speculation.draft_tokens == recipe.max_draft_tokens == 7
    assert recipe.speculation.adaptive and recipe.speculation.packed_verification
    assert recipe.speculation.token_capacity == 128
    assert (recipe.concurrency, recipe.max_concurrency) == (32, 64)
    config = PreparationConfig(draft_snapshot=Path("draft"), max_active=32)
    shapes = config.speculative_round_shapes
    assert {(batch, 1, batch) for batch in (1, 2, 4, 8, 16, 32)} <= set(shapes)
    assert any(batch == 32 and width == 8 and packed == 128 for batch, width, packed in shapes)
    assert all(packed <= 128 for _, _, packed in shapes)
    assert sum(width > 1 for _, width, _ in shapes) <= 64
    assert config.speculative_graph_budget == 64
    assert PreparationConfig(mtp=False).speculative_round_shapes == ()
    fixed = PreparationConfig(draft_snapshot=Path("draft"), speculative_fixed_width=4)
    assert {width for _, width, _ in fixed.speculative_round_shapes} == {1, 4}
    with pytest.raises(ValueError):
        PreparationConfig(draft_snapshot=Path("draft"), speculative_widths=(1, 9))


@pytest.mark.parametrize("location", ["decoder", "projector"])
@pytest.mark.parametrize("key,value", [("output_multiplier", .5), ("final_logit_softcapping", 20.)])
def test_nonraw_draft_logits_rejected(target, location, key, value):
    from ignition.recipes.qwen38_dense.draft_config import DraftConfig
    raw = copy.deepcopy(DRAFT_CONFIG)
    (raw if location == "decoder" else raw["dflash_config"])[key] = value
    with pytest.raises(ValueError):
        DraftConfig.from_dict(raw, target)


def test_pinned_checkpoint_metadata_rejects_precision_and_non_square_shape_errors(config):
    from ignition.components.dflash2_weights import validate_metadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from .test_qwen38_dense_dflash_cuda import DRAFT
    if not (DRAFT / "model.safetensors").is_file():
        pytest.skip("requires pinned Qwen DFlash2 checkpoint")
    manifest = inspect_indexed_checkpoint(DRAFT)
    validate_metadata(manifest, config.draft)
    assert manifest.tensors["layers.0.self_attn.o_proj.weight"].shape == (5120, 4096)
    for name, shape, dtype in (("fc.weight", (5120, 25600), "F32"),
            ("layers.0.self_attn.q_proj.weight", (5120, 5120), "BF16"),
            ("layers.0.self_attn.o_proj.weight", (5120, 5120), "BF16")):
        tensors = dict(manifest.tensors)
        tensors[name] = replace(tensors[name], shape=shape, dtype=dtype)
        with pytest.raises(ValueError, match=name):
            validate_metadata(replace(manifest, tensors=tensors), config.draft)
    tensors = dict(manifest.tensors)
    del tensors["candidate_selector.predecessor_codebook"]
    with pytest.raises(ValueError, match="tensor names"):
        validate_metadata(replace(manifest, tensors=tensors), config.draft)


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("head", [None, "mxfp8", "nvfp4"])
def test_qad_precision_options_reach_preparation_plan(online, head):
    from ignition import planning
    from ignition.cli import _parser
    from ignition.cli.preparation import speculation
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

    flags = (["--online-mxfp8"] if online else []) + (["--draft-head-quantization", head] if head else [])
    args = _parser().parse_args(["bench", "qwen3.8-27b-qad", *flags])
    model = resolve(args.model)
    assert model.checkpoint.repository == "local-inference-lab/Qwen3.8-27B-NVFP4-QAD"
    assert model.checkpoint.revision == "b9389787cbb5a09fbcf3517bdf673b5a56b16b02"
    assert resolve("qwen3.8-27b").checkpoint.repository == "nvidia/Qwen3.8-27B-NVFP4"
    gpus = (planning.Gpu(0, "SM120", "GPU-0", 96 << 30, 95 << 30, (12, 0)),)
    plan = planning.fit(model.recipe, model.checkpoint, gpus, concurrency=1, reserve_bytes=0,
                        online_mxfp8=args.online_mxfp8, speculation=speculation(args, model))
    assert plan.online_mxfp8 == online
    assert plan.speculation.draft_head_quantization == head
    assert plan.to_dict()["online_mxfp8"] == online
    assert plan.to_dict()["speculation"]["draft_head_quantization"] == head
    config = PreparationConfig(draft_snapshot=Path("draft"), online_mxfp8=online, draft_head_quantization=head)
    assert config.online_mxfp8 == online and config.draft_head_quantization == head
    if head:
        with pytest.raises(ValueError, match="requires DFlash2"):
            PreparationConfig(mtp=False, draft_head_quantization=head)


def test_packed_dense_w4a16_is_explicit_and_rejects_native_activations():
    from dataclasses import replace
    from ignition import planning
    from ignition.cli import _parser
    from ignition.cli.preparation import speculation
    from ignition.errors import UsageError
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig

    parser = _parser()
    plain = parser.parse_args(["bench", "qwen3.8-27b-qad"])
    assert not plain.packed_dense_w4a16
    args = parser.parse_args(["export", "qwen3.8-27b-qad", "--packed-dense-w4a16",
                              "--max-concurrency", "8", "--speculative-token-capacity", "64",
                              "--mtp-draft-tokens", "7", "--draft-head-quantization", "nvfp4"])
    assert args.packed_dense_w4a16
    model = resolve(args.model)
    gpus = (planning.Gpu(0, "SM121", "GPU-0", 128 << 30, 120 << 30, (12, 1)),)
    base = planning.fit(model.recipe, model.checkpoint, gpus, concurrency=8, reserve_bytes=0,
                        speculation=speculation(args, model))
    plan = replace(base, packed_dense_w4a16=True)
    assert plan.to_dict()["packed_dense_w4a16"]
    assert plan.speculative_token_capacity == 64
    with pytest.raises(UsageError, match="conflicts"):
        replace(plan, native_dense_activations=True)
    with pytest.raises(ValueError, match="conflicts"):
        PreparationConfig(draft_snapshot=Path("draft"), packed_dense_w4a16=True,
                          native_dense_activations=True)
