"""Unit tests for the Gemma 4 recipe, configuration, pool layout, and numerical kernels."""

import json
from pathlib import Path
import struct

import pytest
import torch

from ignition import planning
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes import resolve
from ignition.recipes.gemma4.attention import (
    apply_rotary_pos_emb,
    attention_descales,
    rms_norm,
    rms_norm_no_scale,
)
from ignition.recipes.gemma4.bootstrap import chat_template_asset
from ignition.recipes.gemma4.config import PAGE_TOKENS, ModelConfig
from ignition.recipes.gemma4.contract import FRONTEND, prepare as prepare_contract
from ignition.recipes.gemma4.decoder import softcap_logits
from ignition.recipes.gemma4.moe import dequantize_nvfp4_tensor, geglu_moe_reference
from ignition.recipes.gemma4.pool import ContinuationPool, continuation_layout, page_bytes
from ignition.recipes.gemma4.recipe import RECIPE
from ignition.recipes.gemma4.weights import BF16ExpertsWeight, NVFP4ExpertsWeight


def _rope_params():
    return {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
    }


def _e2b_raw():
    layer_types = ["full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(35)]
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "attention_k_eq_v": False,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": None,
            "num_kv_shared_layers": 20,
            "use_double_wide_mlp": True,
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "enable_moe_block": False,
            "sliding_window": 512,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "layer_types": layer_types,
            "rope_parameters": _rope_params(),
        },
    }


def _e4b_raw():
    layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(42)]
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "attention_k_eq_v": False,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "vocab_size": 262144,
            "num_hidden_layers": 42,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": None,
            "num_kv_shared_layers": 18,
            "use_double_wide_mlp": False,
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144,
            "enable_moe_block": False,
            "sliding_window": 512,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "layer_types": layer_types,
            "rope_parameters": _rope_params(),
        },
    }


def _gemma4_12b_raw():
    layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(48)]
    return {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "attention_k_eq_v": True,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "vocab_size": 262144,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False,
            "hidden_size_per_layer_input": 0,
            "enable_moe_block": False,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "layer_types": layer_types,
            "rope_parameters": _rope_params(),
        },
    }


def _moe_26b_raw(*, nvfp4=False):
    layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(30)]
    text = {
        "model_type": "gemma4_text",
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": False,
        "head_dim": 256,
        "global_head_dim": 512,
        "hidden_size": 2816,
        "intermediate_size": 2112,
        "moe_intermediate_size": 704,
        "num_experts": 128,
        "top_k_experts": 8,
        "vocab_size": 262144,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "sliding_window": 1024,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "layer_types": layer_types,
        "rope_parameters": _rope_params(),
        "attention_k_eq_v": True,
        "enable_moe_block": True,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
    }
    raw = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": text,
    }
    if nvfp4:
        raw["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                }
            },
            "ignore": [
                "lm_head",
                "*embed_vision*",
                "*mlp*",
                "*router*",
                "*self_attn*",
                "*vision_tower*",
            ],
        }
    return raw


def _dense_31b_raw(*, nvfp4=False):
    layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(60)]
    raw = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "attention_k_eq_v": True,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "vocab_size": 262144,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "num_global_key_value_heads": 4,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False,
            "hidden_size_per_layer_input": 0,
            "enable_moe_block": False,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "layer_types": layer_types,
            "rope_parameters": _rope_params(),
        },
    }
    if nvfp4:
        raw["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                }
            },
            "ignore": ["lm_head", "*embed_vision*", "*self_attn*", "*vision_tower*"],
        }
    return raw


@pytest.mark.parametrize(
    "alias,raw_factory",
    [
        ("gemma-4-e2b", _e2b_raw),
        ("gemma-4-e4b", _e4b_raw),
        ("gemma-4-12b", _gemma4_12b_raw),
        ("gemma-4-26b-a4b", lambda: _moe_26b_raw(nvfp4=False)),
        ("gemma-4-31b", lambda: _dense_31b_raw(nvfp4=False)),
        ("gemma-4-26b-a4b-nvfp4", lambda: _moe_26b_raw(nvfp4=True)),
        ("gemma-4-31b-nvfp4", lambda: _dense_31b_raw(nvfp4=True)),
    ],
)
def test_declared_checkpoints_match_pool_and_page_layouts(alias, raw_factory):
    resolved = resolve(alias)
    assert resolved.recipe is RECIPE
    config = ModelConfig.from_dict(raw_factory())
    assert config.architecture == resolved.checkpoint.architecture
    costs = RECIPE.costs(1, None, resolved.checkpoint)
    layout = continuation_layout(config)
    assert costs.state_row_bytes == layout.row_bytes
    assert costs.page_bytes == page_bytes(config)

    pool = ContinuationPool.for_model(config, 2, 3, "meta")
    assert pool.layout.row_bytes == costs.state_row_bytes
    assert sum(page[0].nbytes for page in pool.pages.values()) == costs.page_bytes

    for layer in range(config.num_layers):
        views = pool.attention(layer, config)
        donor = config.kv_source_layer(layer)
        assert all(view.dtype == torch.float8_e4m3fn for view in views)
        if config.is_sliding(layer):
            assert len(views) == 2
            assert views[0].shape == (3, PAGE_TOKENS, config.layer_kv_heads(donor), 256)
            assert views[1].shape == (3, PAGE_TOKENS, config.layer_kv_heads(donor), 256)
        else:
            assert len(views) == 3
            assert views[0].shape == (3, PAGE_TOKENS, config.layer_kv_heads(donor), 512)
            assert views[1].shape == (3, PAGE_TOKENS, config.layer_kv_heads(donor), 256)
            assert views[2].shape == (3, PAGE_TOKENS, config.layer_kv_heads(donor), 256)
        if config.is_kv_shared(layer):
            donor_views = pool.attention(donor, config)
            assert all(v.data_ptr() == dv.data_ptr() for v, dv in zip(views, donor_views, strict=True))


def test_e2b_kv_sharing_and_double_wide_mlp_geometry():
    config = ModelConfig.from_dict(_e2b_raw())
    assert config.first_kv_shared_layer == 15
    assert config.ple_dim == 256 and config.ple_vocabulary == 262144
    for layer in range(15):
        assert not config.is_kv_shared(layer)
        assert config.kv_source_layer(layer) == layer
        assert config.layer_intermediate_size(layer) == 6144
        assert config.layer_has_v_proj(layer)
    for layer in range(15, 35):
        assert config.is_kv_shared(layer)
        assert config.layer_intermediate_size(layer) == 12288
        assert not config.layer_has_v_proj(layer)
        if config.is_sliding(layer):
            assert config.kv_source_layer(layer) == 13
        else:
            assert config.kv_source_layer(layer) == 14


def test_moe_k_eq_v_and_quantization_expansion():
    moe_nvfp4 = ModelConfig.from_dict(_moe_26b_raw(nvfp4=True))
    assert moe_nvfp4.text_prefix == "model.language_model"
    assert moe_nvfp4.enable_moe and moe_nvfp4.attention_k_eq_v
    assert moe_nvfp4.layer_kv_heads(0) == 8 and moe_nvfp4.layer_has_v_proj(0)
    assert moe_nvfp4.layer_kv_heads(5) == 2 and not moe_nvfp4.layer_has_v_proj(5)
    assert len(moe_nvfp4.quantization) == 30 * 128 * 3
    assert "model.language_model.layers.0.experts.0.gate_proj" in moe_nvfp4.quantization

    dense_nvfp4 = ModelConfig.from_dict(_dense_31b_raw(nvfp4=True))
    assert len(dense_nvfp4.quantization) == 60 * 3
    assert "model.language_model.layers.0.mlp.gate_proj" in dense_nvfp4.quantization
    assert "model.language_model.layers.0.self_attn.q_proj" not in dense_nvfp4.quantization


def test_config_rejects_invalid_geometry():
    bad = _e2b_raw()
    bad["architectures"] = ["LlamaForCausalLM"]
    with pytest.raises(ValueError, match="Gemma 4 preparation requires"):
        ModelConfig.from_dict(bad)

    bad = _e2b_raw()
    bad["text_config"]["head_dim"] = 128
    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig.from_dict(bad)

    bad = _e2b_raw()
    bad["text_config"]["layer_types"][-1] = "sliding_attention"
    with pytest.raises(ValueError, match="end with full_attention"):
        ModelConfig.from_dict(bad)

    bad = _e2b_raw()
    bad["text_config"]["num_kv_shared_layers"] = 31
    with pytest.raises(ValueError, match="no non-shared donor layer"):
        ModelConfig.from_dict(bad)


def test_proportional_and_default_rope_and_norms():
    config = ModelConfig.from_dict(_moe_26b_raw())
    sliding_inv = config.rotary_inverse("sliding_attention", "cpu")
    full_inv = config.rotary_inverse("full_attention", "cpu")
    assert sliding_inv.shape == (128,)
    assert full_inv.shape == (256,)
    assert torch.all(full_inv[:64] > 0)
    assert torch.all(full_inv[64:] == 0)

    pos = torch.tensor([[3.0], [7.0]])
    f_freq = pos * full_inv[None, :]
    f_emb = torch.cat((f_freq, f_freq), dim=-1)
    cos, sin = f_emb.cos(), f_emb.sin()
    x = torch.randn(2, 4, 512)
    rotated = apply_rotary_pos_emb(x, cos, sin)
    assert torch.allclose(rotated[:, :, 64:256], x[:, :, 64:256], atol=1e-6)
    assert torch.allclose(rotated[:, :, 320:512], x[:, :, 320:512], atol=1e-6)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-4)

    weight = torch.full((16,), 2.0, dtype=torch.bfloat16)
    sample = torch.randn(3, 16, dtype=torch.bfloat16)
    normed = rms_norm(sample, weight, 1e-6)
    unscaled = rms_norm_no_scale(sample, 1e-6)
    assert normed.dtype == torch.bfloat16 and unscaled.dtype == torch.bfloat16
    assert torch.allclose(normed.float(), unscaled.float() * 2.0, atol=2e-2)

    unit, s_descale, f_descale = attention_descales(config, "cpu")
    assert unit.item() == 1.0
    assert s_descale.item() == 16.0
    assert abs(f_descale.item() - (512**0.5)) < 1e-5


def test_softcap_logits_and_moe_reference():
    logits = torch.tensor([[-100.0, 0.0, 15.0, 100.0]], dtype=torch.bfloat16)
    softcap_logits(logits, 30.0)
    expected = (torch.tanh(torch.tensor([[-100.0, 0.0, 15.0, 100.0]]) / 30.0) * 30.0).bfloat16()
    assert torch.equal(logits, expected)

    torch.manual_seed(0)
    tokens, hidden, inter, num_experts, top_k = 4, 32, 16, 6, 2
    residual = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    pre_norm_2 = torch.ones(hidden, dtype=torch.bfloat16)
    post_norm_2 = torch.ones(hidden, dtype=torch.bfloat16)
    router_scale = torch.ones(hidden, dtype=torch.bfloat16)
    per_expert_scale = torch.linspace(0.5, 1.5, num_experts, dtype=torch.bfloat16)
    router_proj = torch.randn(num_experts, hidden, dtype=torch.bfloat16)
    gate_up = torch.randn(num_experts, 2 * inter, hidden, dtype=torch.bfloat16) * 0.1
    down = torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16) * 0.1
    bf16_experts = BF16ExpertsWeight(gate_up_proj=gate_up, down_proj=down)

    out = geglu_moe_reference(
        residual,
        pre_norm_weight=pre_norm_2,
        post_norm_weight=post_norm_2,
        router_scale=router_scale,
        per_expert_scale=per_expert_scale,
        router_proj_weight=router_proj,
        experts=bf16_experts,
        top_k=top_k,
        eps=1e-6,
    )
    assert out.shape == (tokens, hidden) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    packed = torch.tensor([[[0x32] * (hidden // 2)] * (2 * inter)] * num_experts, dtype=torch.uint8)
    scale = torch.ones((num_experts, 2 * inter, hidden // 16), dtype=torch.float8_e4m3fn)
    scale_2 = torch.full((num_experts, 1, 1), 0.5, dtype=torch.float32)
    deq = dequantize_nvfp4_tensor(packed, scale, scale_2)
    assert deq.shape == (num_experts, 2 * inter, hidden)
    assert torch.allclose(deq[:, :, 0::2], torch.full_like(deq[:, :, 0::2], 0.5))
    assert torch.allclose(deq[:, :, 1::2], torch.full_like(deq[:, :, 1::2], 0.75))

    nvfp4_experts = NVFP4ExpertsWeight(
        gate_up_weight=packed,
        gate_up_scale=scale,
        gate_up_scale_2=torch.full((num_experts, 2), 0.5, dtype=torch.float32),
        down_weight=torch.tensor([[[0x21] * (inter // 2)] * hidden] * num_experts, dtype=torch.uint8),
        down_scale=torch.ones((num_experts, hidden, inter // 16), dtype=torch.float8_e4m3fn),
        down_scale_2=torch.full((num_experts,), 0.25, dtype=torch.float32),
    )
    out_nvfp4 = geglu_moe_reference(
        residual,
        pre_norm_weight=pre_norm_2,
        post_norm_weight=post_norm_2,
        router_scale=router_scale,
        per_expert_scale=per_expert_scale,
        router_proj_weight=router_proj,
        experts=nvfp4_experts,
        top_k=top_k,
        eps=1e-6,
    )
    assert out_nvfp4.shape == (tokens, hidden) and torch.isfinite(out_nvfp4).all()


def test_single_file_safetensors_checkpoint_resolution_and_inspection(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_e2b_raw()))
    header = json.dumps({
        "model.language_model.norm.weight": {
            "dtype": "BF16",
            "shape": [1536],
            "data_offsets": [0, 3072],
        }
    }).encode("utf-8")
    payload = b"\x00" * 3072
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)

    resolved = resolve(str(tmp_path))
    assert resolved.recipe is RECIPE
    assert resolved.cached(tmp_path)
    resolved.verify(tmp_path)
    assert resolved.checkpoint.stored_bytes == 8 + len(header) + len(payload)

    manifest = inspect_indexed_checkpoint(tmp_path)
    assert manifest.index_path is None
    assert len(manifest.shards) == 1
    assert "model.language_model.norm.weight" in manifest.tensors


def test_gemma4_recipe_adapter_configures_bootstrap(monkeypatch, tmp_path):
    seen = {}

    def fake_prepare(snapshot, config, *, progress, package_storage):
        seen.update(snapshot=snapshot, config=config, package_storage=package_storage)
        return "prepared"

    monkeypatch.setattr("ignition.recipes.gemma4.bootstrap.prepare", fake_prepare)
    model = resolve("gemma-4-26b-a4b-nvfp4")
    gpu = planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell", "GPU-0", 96 << 30, 95 << 30, (12, 0))
    plan = planning.fit(model.recipe, model.checkpoint, (gpu,), concurrency=4, reserve_bytes=0, speculation=None)
    assert model.recipe.prepare(
        tmp_path,
        plan,
        compile_forwards=True,
        kernel_cache=tmp_path / "cache",
        progress=None,
        package_storage="storage",
    ) == "prepared"
    assert seen["config"].max_active == 4
    assert seen["config"].context == 262144
    assert seen["config"].prefill_chunk_size == 4096
    assert seen["config"].compile_decoder is True
    assert seen["package_storage"] == "storage"


def test_frontend_contract_and_cache_regions():
    assert FRONTEND["assistant_marker"] == "<|turn>model\n"
    assert FRONTEND["reasoning"]["open"] == "<think>"
    assert FRONTEND["reasoning"]["close"] == "</think>"
    assert "<turn|>" in FRONTEND["message_end_markers"]

    raw = _moe_26b_raw()
    raw["text_config"]["hidden_size"] = 32
    raw["text_config"]["intermediate_size"] = 16
    raw["text_config"]["num_hidden_layers"] = 5
    raw["text_config"]["layer_types"] = ["sliding_attention"] * 4 + ["full_attention"]
    config = ModelConfig.from_dict(raw)

    pool = ContinuationPool.for_model(config, 2, 8, "cpu")
    buffers = {}
    contract = prepare_contract({}, buffers, pool, vocabulary=config.vocabulary)
    assert contract["vocabulary"] == config.vocabulary
    assert contract["cache"]["page_tokens"] == PAGE_TOKENS
    assert len(contract["cache"]["regions"]) == 1 + len(pool.pages)


@pytest.mark.parametrize(
    "alias,raw_fn,filename",
    [
        ("gemma-4-e2b", _e2b_raw, "chat_template.google-e2b-e4b.jinja"),
        ("gemma-4-e4b", _e4b_raw, "chat_template.google-e2b-e4b.jinja"),
        ("gemma-4-e4b-nvfp4", _e4b_raw, "chat_template.google-e2b-e4b.jinja"),
        ("gemma-4-12b", _gemma4_12b_raw, "chat_template.google.jinja"),
        ("gemma-4-12b-nvfp4", _gemma4_12b_raw, "chat_template.google.jinja"),
        ("gemma-4-26b-a4b", _moe_26b_raw, "chat_template.google.jinja"),
        ("gemma-4-26b-a4b-nvfp4", _moe_26b_raw, "chat_template.google.jinja"),
        ("gemma-4-31b", _dense_31b_raw, "chat_template.google.jinja"),
        ("gemma-4-31b-nvfp4", _dense_31b_raw, "chat_template.google.jinja"),
    ],
)
def test_model_sizes_select_canonical_chat_templates(alias, raw_fn, filename):
    assert resolve(alias).recipe is RECIPE
    config = ModelConfig.from_dict(raw_fn())
    asset = chat_template_asset(config.hidden)
    path = Path(asset["path"])
    assert path.name == filename
    assert path.read_bytes().count(b"{%- mark -%}") == 2


@pytest.mark.parametrize(
    "checkpoint",
    [checkpoint for checkpoint in RECIPE.checkpoints if checkpoint.repository.startswith("google/")],
    ids=lambda checkpoint: checkpoint.alias,
)
def test_canonical_chat_templates_match_pinned_google_sources(checkpoint):
    snapshot = (Path.home() / ".cache/huggingface/hub" / ("models--" + checkpoint.repository.replace("/", "--"))
                / "snapshots" / checkpoint.revision)
    if not all((snapshot / name).exists() for name in ("config.json", "chat_template.jinja")):
        pytest.skip("Google checkpoint configuration and chat template are not cached")
    source = (snapshot / "chat_template.jinja").read_bytes()
    config = ModelConfig.load(snapshot)
    annotated = Path(chat_template_asset(config.hidden)["path"]).read_bytes()
    assert annotated.replace(b"{%- mark -%}", b"") == source
    assert annotated.count(b"{%- mark -%}") == 2


def test_unknown_model_size_has_no_chat_template():
    with pytest.raises(ValueError, match="unsupported Gemma 4 chat template hidden size"):
        chat_template_asset(4096)


@pytest.mark.parametrize(
    "alias,raw_fn",
    [
        ("gemma-4-e2b", _e2b_raw),
        ("gemma-4-e4b", _e4b_raw),
        ("gemma-4-e4b-nvfp4", _e4b_raw),
        ("gemma-4-12b", _gemma4_12b_raw),
        ("gemma-4-12b-nvfp4", _gemma4_12b_raw),
        ("gemma-4-26b-a4b", _moe_26b_raw),
        ("gemma-4-26b-a4b-nvfp4", _moe_26b_raw),
        ("gemma-4-31b", _dense_31b_raw),
        ("gemma-4-31b-nvfp4", _dense_31b_raw),
    ],
)
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_gemma4_tp_sharding_and_declared_rank_costs(alias, raw_fn, world_size):
    from ignition.components.parallel import TensorParallelShard

    model = resolve(alias)
    config = ModelConfig.from_dict(raw_fn())
    for rank in range(world_size):
        shard = TensorParallelShard(rank, world_size)
        config.validate_shard(shard)
        pool = ContinuationPool.for_model(config, 2, 2, "meta", shard=shard)
        for spec in (None, model.recipe.speculation):
            costs = model.recipe.costs(world_size, spec, model.checkpoint)
            assert pool.layout.row_bytes == costs.state_row_bytes
            assert sum(page[0].nbytes for page in pool.pages.values()) == costs.page_bytes


def test_gemma4_draft_config_and_donor_kv_sharing():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.gemma4.config import DraftConfig

    target_31b = ModelConfig.from_dict(_dense_31b_raw())
    assert target_31b.sliding_donor_layer == 58
    assert target_31b.full_donor_layer == 59
    draft_raw = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 5376,
        "use_ordered_embeddings": False,
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "attention_k_eq_v": True,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "vocab_size": 262144,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_global_key_value_heads": 4,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "enable_moe_block": False,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": _rope_params(),
        },
    }
    draft_31b = DraftConfig.from_dict(draft_raw, target_31b)
    assert draft_31b.kv_source_layer(0) == 58
    assert draft_31b.kv_source_layer(1) == 59
    assert draft_31b.is_kv_shared(0) and draft_31b.is_kv_shared(1)
    from dataclasses import replace

    target_with_draft = replace(target_31b, draft=draft_31b)
    for w in (1, 2, 4):
        shard = TensorParallelShard(0, w)
        target_with_draft.validate_shard(shard)
        pool = ContinuationPool.for_model(target_31b, 2, 2, "meta", shard=shard)
        assert pool.attention(0, draft_31b, shard)[0].data_ptr() == pool.attention(58, target_31b, shard)[0].data_ptr()
        assert pool.attention(1, draft_31b, shard)[0].data_ptr() == pool.attention(59, target_31b, shard)[0].data_ptr()

    target_e4b = ModelConfig.from_dict(_e4b_raw())
    # E4B has 42 layers with 18 KV-shared layers; donor layers are 22 (sliding) and 23 (full).
    assert target_e4b.sliding_donor_layer == 22
    assert target_e4b.full_donor_layer == 23
    draft_e4b_raw = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": 2560,
        "use_ordered_embeddings": True,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "text_config": {
            **draft_raw["text_config"],
            "attention_k_eq_v": False,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": None,
            "sliding_window": 512,
        },
    }
    draft_e4b = DraftConfig.from_dict(draft_e4b_raw, target_e4b)
    assert draft_e4b.use_ordered_embeddings and draft_e4b.cluster_size == 128
    assert draft_e4b.kv_source_layer(0) == 22
    assert draft_e4b.kv_source_layer(1) == 23

    target_12b = ModelConfig.from_dict(_gemma4_12b_raw())
    draft_12b_raw = {
        **draft_raw,
        "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
        "model_type": "gemma4_unified_assistant",
        "backbone_hidden_size": 3840,
        "text_config": {
            **draft_raw["text_config"],
            "model_type": "gemma4_unified_text",
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "sliding_window": 1024,
        },
    }
    draft_12b = DraftConfig.from_dict(draft_12b_raw, target_12b)
    assert draft_12b.architecture == "Gemma4UnifiedAssistantForCausalLM"


def test_gemma4_unsloth_compressed_tensors_quantization_parsing():
    raw = _e4b_raw()
    raw["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0_fp8": {
                "format": "float-quantized",
                "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$"],
                "weights": {"num_bits": 8, "strategy": "channel", "type": "float"},
            },
            "group_0_nvfp4": {
                "format": "nvfp4-pack-quantized",
                "targets": ["re:.*mlp\\.(gate|up|down)_proj$"],
                "weights": {"num_bits": 4, "strategy": "tensor_group", "group_size": 16, "type": "float"},
                "input_activations": {"num_bits": 4, "strategy": "tensor_group", "group_size": 16, "type": "float"},
            },
            "group_2_ple": {
                "format": "pack-quantized",
                "targets": ["re:.*embed_tokens_per_layer$"],
                "weights": {"num_bits": 8, "strategy": "channel", "type": "int"},
            },
        },
        "ignore": ["lm_head", "re:.*per_layer_input_gate$"],
    }
    config = ModelConfig.from_dict(raw)
    assert config.quantization["model.language_model.layers.0.self_attn.q_proj"]["quant_algo"] == "FP8_CHANNEL"
    assert config.quantization["model.language_model.layers.0.mlp.gate_proj"]["quant_algo"] == "COMPRESSED_NVFP4"
    assert config.quantization["model.language_model.embed_tokens_per_layer"]["quant_algo"] == "INT8_PACK_CHANNEL"
    assert "model.language_model.layers.0.per_layer_input_gate" not in config.quantization


def test_gemma4_multi_rank_and_speculative_adapter(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import huggingface_hub

    seen = []
    monkeypatch.setattr(
        "ignition.preparation.rank_group.prepare_group",
        lambda snapshot, config, prepare, *, progress, package_storage=None, ranks=None, checkpoint=None: seen.append(
            (config, prepare, ranks, checkpoint)
        )
        or "group",
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: f"/hub/{repo}")

    model = resolve("gemma-4-31b-nvfp4")
    gpus = tuple(
        planning.Gpu(i, "NVIDIA GB10", f"GPU-{i}", 128 << 30, 120 << 30, (12, 1)) for i in range(4)
    )
    plan = planning.fit(model.recipe, model.checkpoint, gpus, concurrency=4, reserve_bytes=0, speculation=model.recipe.speculation)
    rendezvous = SimpleNamespace(hosts=("host0", "host1", "host2", "host3"))
    assert (
        model.recipe.prepare(
            tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None, ranks=rendezvous
        )
        == "group"
    )
    config, _, ranks, checkpoint = seen.pop()
    assert config.devices == (0, 0, 0, 0) and config.hosts == ("host0", "host1", "host2", "host3")
    assert config.draft_checkpoint == "/hub/google/gemma-4-31B-it-assistant"
    assert config.speculative_widths == (1, 2, 3, 4, 5, 6)
    assert ranks is rendezvous and checkpoint == model.checkpoint

