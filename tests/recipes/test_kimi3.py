"""Kimi K3 configuration, residual composition, and checkpoint-backed execution."""

from copy import deepcopy
import os
from pathlib import Path

import pytest

# The CSF and native checkpoints share configuration and BF16 attention weights.
CHECKPOINT = next((path for path in (Path("/data/models/Kimi-K3-MXFP4-CSF"), Path(
    "/data/models--moonshotai--Kimi-K3/snapshots/2496450e92e425c886db095102a52a6682ca3970")) if path.is_dir()),
    Path("/data/models/Kimi-K3-MXFP4-CSF"))


@pytest.fixture
def config():
    from ignition.recipes.kimi3.config import ModelConfig

    if not CHECKPOINT.is_dir():
        pytest.skip("Kimi K3 checkpoint is not installed")
    return ModelConfig.load(CHECKPOINT)


def test_checkpoint_geometry(config):
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.pool import continuation_layout, page_bytes

    config.validate_shard(TensorParallelShard(0, 16))
    assert config.layers.count("kda") == 69
    assert config.layers.count("mla") == 24
    assert config.layers[0] == "kda" and config.layers[3] == "mla"
    layout = continuation_layout(config, 16)
    assert layout.region("layer0.recurrent").shape == (6, 128, 128)
    # E4M3 latent records, as vLLM serves Kimi K3 with an FP8 KV cache.
    assert page_bytes(config) == 24 * 128 * 576


def test_quantization_and_layer_partition_rejected(config):
    from ignition.recipes.kimi3.config import ModelConfig

    raw = dict(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"],
               text_config=deepcopy(config.text), vision_config=config.vision,
               media_placeholder_token_id=163605, image_placeholder="<|kimi_image_placeholder|>")
    raw["text_config"]["linear_attn_config"]["kda_layers"].append(4)
    with pytest.raises(ValueError, match="partition"):
        ModelConfig.from_dict(raw)
    raw["text_config"] = deepcopy(config.text)
    raw["text_config"]["activation_situ_beta"] = 1.
    with pytest.raises(ValueError, match="activation_situ_beta"):
        ModelConfig.from_dict(raw)


def test_situ():
    import torch
    from ignition.recipes.kimi3.model import situ

    x = torch.tensor([[100., -100., 0., 100., 100., 100.]], dtype=torch.bfloat16)
    y = situ(x)
    assert torch.isfinite(y).all() and y[0, 0] == 100 and y[0, 2] == 0
    assert y[0, 1].abs() < 1e-30


def _reference_attention(weights):
    import torch
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

    result = {}
    for key, weight in weights.items():
        if isinstance(weight, torch.Tensor):
            result[key] = weight
        else:
            packed = weight.weight
            result[key] = dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows)[:, :weight.in_features].bfloat16()
    return result


@pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA checkpoint tests")
@pytest.mark.parametrize("online_mxfp8", [False, True])
@pytest.mark.parametrize("mode,tokens", [("decode", 1), ("prefill", 65), ("prefill", 4096)])
def test_checkpoint_kda(config, mode, tokens, online_mxfp8):
    import torch
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from ignition.recipes.kimi3.kda import PreparedKDA
    from ignition.components.recurrent import DeltaState, DeltaMetadata
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    with KimiWeightLoader(manifest, device, shard=TensorParallelShard(0, 16),
                         prefixes=("language_model.model.layers.0.",), online_mxfp8=online_mxfp8) as loader:
        weights = loader.load_layer(0)
    assert all(weight.dtype == torch.bfloat16 for weight in weights["mlp"].values())
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-kda", numerical_policy="default", compile_workers=4) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        state = DeltaState.allocate(3, device, heads=6, channels=2304)
        metadata = DeltaMetadata.allocate(1, device)
        torch.manual_seed(41)
        source = torch.randn((tokens, 7168), device=device, dtype=torch.bfloat16)
        graph_buffers = GraphBufferPool(device)
        op = PreparedKDA(weights["attention"], source, state, metadata, mode=mode, eps=1e-5,
                        session=session, name="kda", workspace=arena, buffers={}, allocator=graph_buffers.cursor())
        session.tune(show_progress=False)
        metadata.query_start_loc[1] = tokens
        metadata.convolution_slots[0] = 1
        metadata.state_indices[0, 0] = 1
        metadata.final_state_indices[0] = 1
        metadata.num_seqs.fill_(1)
        metadata.num_tokens.fill_(tokens)
        metadata.fresh.fill_(True)
        state.convolution.zero_()
        state.recurrent.zero_()
        output = op(source).clone()
        assert torch.isfinite(output).all() and output.abs().max() > 0
        expected_state = state.recurrent[1].clone()
        from b12x.sequence.kda_prefill.reference import recurrent_kda
        from torch.nn import functional as F

        w = _reference_attention(weights["attention"])
        projected = F.linear(source, w["input"])
        qkv, forget, gate, beta, _ = projected.split((2304, 128, 768, 6, 2), -1)
        convolution = F.conv1d(F.pad(qkv.T.float()[None], (3, 0)),
                               w["convolution"][:, None], groups=2304).bfloat16()
        q, k, v = F.silu(convolution)[0].T.reshape(tokens, 3, 6, 128).unbind(1)
        raw_g = F.linear(forget, w["forget"]).view(tokens, 6, 128)
        recurrent, reference_state, _ = recurrent_kda(q, k, v, raw_g, beta,
            w["A_log"], w["dt_bias"].view(6, 128), lower_bound=-5.,
            initial_state=torch.zeros_like(expected_state))
        normalized = recurrent.float() * torch.rsqrt(recurrent.float().square().mean(-1, keepdim=True) + 1e-5)
        gate = gate.reshape(tokens, 6, 128).float().sigmoid()
        reference = F.linear((normalized * w["norm"] * gate).bfloat16().flatten(1), w["output"])
        for actual, expected in ((output, reference), (expected_state, reference_state)):
            relative_rms = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
            assert relative_rms.item() < .015
        state.convolution.zero_()
        state.recurrent.zero_()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = op(source)
        for chunks in graph_buffers.chunks.values():
            for chunk in chunks:
                chunk.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(replayed, output, rtol=0, atol=0)
        torch.testing.assert_close(state.recurrent[1], expected_state, rtol=0, atol=0)
        assert not state.recurrent[0].count_nonzero()


@pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires CUDA checkpoint tests")
@pytest.mark.parametrize("online_mxfp8", [False, True])
@pytest.mark.parametrize("mode,tokens", [("decode", 1), ("prefill", 65), ("prefill", 4096)])
def test_checkpoint_mla(config, mode, tokens, online_mxfp8):
    import torch
    from torch.nn import functional as F
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from ignition.recipes.kimi3.attention import PreparedMLA
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(CHECKPOINT)
    # One rank projects every latent row itself; tensor-parallel ranks gather them through b12x.
    with KimiWeightLoader(manifest, device, shard=TensorParallelShard(0, 1),
                         prefixes=("language_model.model.layers.3.self_attn.",), online_mxfp8=online_mxfp8) as loader:
        weights = loader.load_attention(3)
    with RecordedPreparationSession(device=device, checkpoint=str(CHECKPOINT),
            workload_schema="kimi3-mla", numerical_policy="default", compile_workers=4) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        pages = max(2, -(-tokens // 128))
        metadata = Metadata.allocate(tokens, 1, pages, device)
        cache = torch.zeros((pages, 128, 576), dtype=torch.float8_e4m3fn, device=device)
        torch.manual_seed(41)
        source = torch.randn((tokens, 7168), device=device, dtype=torch.bfloat16)
        graph_buffers = GraphBufferPool(device)
        op = PreparedMLA(weights, source, metadata, cache, mode=mode, session=session,
                         name="mla", workspace=arena, buffers={}, allocator=graph_buffers.cursor())
        session.tune(show_progress=False)
        metadata.query_start_loc[1] = tokens
        metadata.lengths.fill_(tokens)
        metadata.page_table.copy_(torch.arange(pages, device=device)[None])
        metadata.positions.copy_(torch.arange(tokens, device=device))
        metadata.token_page_table.copy_(metadata.page_table.expand(tokens, -1))
        output = op(source).clone()
        w = _reference_attention(weights)
        projected = F.linear(source, w["latent_gate"][:2112])
        def norm(x, weight):
            return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
        heads = w["key"].shape[0]
        query = F.linear(norm(projected[:, :1536], w["query_norm"]), w["q_b"]).view(tokens, heads, 192)
        latent = norm(projected[:, 1536:-64], w["kv_norm"])
        gate = F.linear(source, w["latent_gate"][2112:]).sigmoid()
        keys = torch.einsum("tk,hdk->thd", latent.float(), w["key"].float()).bfloat16()
        keys = torch.cat((keys, projected[:, None, -64:].expand(-1, heads, -1)), -1)
        values = torch.einsum("tk,hkd->thd", latent.float(), w["value"].float()).bfloat16()
        expected = F.scaled_dot_product_attention(query.transpose(0, 1)[None], keys.transpose(0, 1)[None],
            values.transpose(0, 1)[None], is_causal=True).squeeze(0).transpose(0, 1).flatten(1)
        expected = F.linear(expected * gate, w["output"])
        error = (output.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert error.item() < .06
        # Compare implementation error separately from quantization: E4M3 records with a unit scale, and the
        # absorbed query that b12x quantizes to E4M3 the same way.
        absorbed = torch.cat((torch.bmm(query[:, :, :128].transpose(0, 1), w["key"]).transpose(0, 1),
                              query[:, :, 128:]), -1).float().clamp(-448., 448.).to(torch.float8_e4m3fn).float()
        records = torch.cat((latent, projected[:, -64:]), -1).to(torch.float8_e4m3fn).float()
        attended = F.scaled_dot_product_attention(absorbed.transpose(0, 1)[None], records[None, None],
            records[None, None, :, :512], is_causal=True, scale=192 ** -.5).squeeze(0).transpose(0, 1).bfloat16()
        expanded = torch.bmm(attended.transpose(0, 1), w["value"]).transpose(0, 1).flatten(1)
        quantized_reference = F.linear(expanded * gate, w["output"])
        error = (output.float() - quantized_reference.float()).norm() / quantized_reference.float().norm()
        assert error.item() < .02
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = op(source)
        for chunks in graph_buffers.chunks.values():
            for chunk in chunks:
                chunk.fill_(float("nan") if chunk.is_floating_point() else -1)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(replayed, output, rtol=0, atol=0)


DRAFT = Path(os.getenv("FLARE_KIMI3_REDHAT_DSPARK",
    "/data/cache/huggingface/hub/models--RedHatAI--Kimi-K3-speculator.dspark/snapshots/"
    "38a88101e0d46bb22134b9da340f381b954d40d4"))


def test_dspark_draft_geometry(config):
    import json

    import torch
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
    from ignition.recipes.kimi3.pool import continuation_layout

    if not DRAFT.is_dir():
        pytest.skip("Kimi K3 DSpark draft is not installed")
    raw = json.loads((DRAFT / "config.json").read_text())
    draft = DSparkConfig.from_dict(raw, config.text)
    draft.validate_shard(TensorParallelShard(0, 16))
    assert draft.auxiliary_layers == (24, 48, 72, 88, 92) and draft.noise_token == 163837
    assert (draft.layers, draft.heads, draft.kv_heads, draft.head_dim, draft.window) == (5, 96, 16, 64, 2048)
    speculative = ModelConfig(config.text, config.vision, draft)
    plain, widened = continuation_layout(speculative, 16), continuation_layout(speculative, 16, speculative_tokens=7)
    assert widened.region("layer1.convolution").shape == (2304, 10)
    assert widened.region("draft4.value").shape == (2048, 1, 64) and widened.region("draft4.value").dtype == torch.bfloat16
    assert widened.row_bytes > plain.row_bytes
    for name, value in (("aux_hidden_state_layer_ids", [0, 93]), ("sample_from_anchor", False),
                        ("sliding_window_non_causal", True)):
        with pytest.raises(ValueError, match="DSpark"):
            DSparkConfig.from_dict({**raw, name: value}, config.text)


def test_context_shards_place_interleaved_positions_on_their_owners():
    import torch

    from ignition.recipes.kimi3.config import PAGE_TOKENS
    from ignition.recipes.kimi3.dcp import ContextShards

    world, pages = 16, torch.tensor([[7, 3, -1]], dtype=torch.int32)
    positions = torch.tensor([-1, 0, 1, 15, 16, 17, PAGE_TOKENS * world + 5, PAGE_TOKENS * world * 2 + 1])
    tables = pages.expand(positions.numel(), -1)
    owned = {}
    for rank in range(world):
        shards = ContextShards.on(rank, world, "cpu")
        slots = shards.token_slots(positions, tables)
        for position, slot in zip(positions.tolist(), slots.tolist()):
            if slot >= 0:
                assert position % world == rank and position not in owned
                owned[position] = slot
        # Each rank holds its share of every prefix, interleaved one position at a time as vLLM's DCP stores it.
        lengths = torch.arange(0, 3 * world + 1)
        assert shards.local_tokens(lengths).tolist() == [sum(p % world == rank for p in range(n)) for n in range(len(lengths))]
    # Position t lives at local record t // 16 of its page; a third page without a table entry stores nothing.
    assert owned == {0: 7 * PAGE_TOKENS, 1: 7 * PAGE_TOKENS, 15: 7 * PAGE_TOKENS, 16: 7 * PAGE_TOKENS + 1,
                     17: 7 * PAGE_TOKENS + 1, PAGE_TOKENS * world + 5: 3 * PAGE_TOKENS}


def test_runtime_assets_preserve_converted_tokenizer_and_bundled_template_bytes(tmp_path):
    from tokenizers import Tokenizer, models
    from ignition.recipes.kimi3.tokenizer import runtime_assets

    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "世界": 2}, unk_token="<unk>"))
    expected = tmp_path / "tokenizer.json"
    tokenizer.save(str(expected))
    assets = runtime_assets(tokenizer)
    assert assets["tokenizer"]["text"].encode() == expected.read_bytes()
    template = Path(__file__).resolve().parents[2] / "ignition/recipes/kimi3/assets/chat_template.jinja"
    assert Path(assets["chat_template"]["path"]) == template
    assert len(list(tmp_path.iterdir())) == 1
