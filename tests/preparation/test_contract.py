"""Typed prepared values preserve graph dataflow independently of model identity."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.protobuf.json_format import ParseDict

from ignition.preparation import contract as c, runtime_package_pb2 as p
from ignition.preparation.execution import GraphSignature


@pytest.mark.parametrize("recipe,page,lag,rows", [
    ("qwen38", 64, 1, 1), ("glm53", 256, 1, 1),
    ("deepseek41", 128, 0, None), ("deepseek4", 128, 0, None), ("mimo_v2", 128, 0, 1),
])
def test_recipe_declares_sources_storage_cursors_and_frontend(recipe, page, lag, rows):
    module = importlib.import_module(f"ignition.recipes.{recipe}.contract")
    signatures = {kind: GraphSignature(kind, 1, 4 if kind == "speculative" else 1,
        tuple(c.text_inputs(kind)), speculation_enabled=kind == "speculative")
        for kind in ("init", "restore", "snapshot", "prefill", "decode", "resume", "speculative")}
    pool = SimpleNamespace(pages={}, layout=SimpleNamespace(row_bytes=32,
        region=lambda name: SimpleNamespace(offset=24, nbytes=8)))
    contract = module.prepare(signatures, {"page_tables": None, "draft_page_tables": None}, pool,
        page_tokens=page, draft_lag=lag, rows=rows)
    encoded = ParseDict(contract, p.ModelContract())
    assert encoded.version == 1 and encoded.input_vocabulary >= encoded.vocabulary
    assert "schema" not in contract
    decode = next(g for g in encoded.graphs if g.kind == p.DECODE)
    assert decode.HasField("pending_token") == (recipe == "qwen38")
    assert any(b.source == p.PENDING_TOKEN for b in decode.inputs) == (recipe != "qwen38")
    speculative = next(g for g in encoded.graphs if g.kind == p.SPECULATIVE)
    assert speculative.draft_cursor.lag_tokens == lag
    assert speculative.state_rows_per_request == (4 if rows is None else rows)
    assert speculative.draft_cursor.state.offset == 24
    init = next(g for g in encoded.graphs if g.kind == p.INIT)
    assert next(b for b in init.inputs if b.source == p.SAMPLING_SEED).indexing == p.ACTIVE_ROW
    assert encoded.frontend.tools.kind in (p.QWEN_CODER, p.GLM_ARGUMENT_TAGS, p.DSML)
    fixture = json.loads((Path(__file__).parents[2] / "helios/tests/support/contracts" / f"{recipe}.json").read_text())
    assert fixture["frontend"] == contract["frontend"]
    assert fixture["vision"] == contract["vision"]


def test_device_pending_token_has_read_and_producer_declarations():
    module = importlib.import_module("ignition.recipes.qwen38.contract")
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32, region=lambda _: SimpleNamespace(offset=0, nbytes=8)))
    signatures = {kind: GraphSignature(kind, 1, 1, tuple(c.text_inputs(kind))) for kind in ("init", "restore", "prefill", "decode")}
    value = module.prepare(signatures, {"page_tables": None}, pool, page_tokens=64, draft_lag=None)
    decode = next(g for g in value["graphs"] if g["kind"] == p.DECODE)
    pending = decode["pending_token"]
    assert pending["buffer"] == "request_pending_token"
    for graph in value["graphs"]:
        assert any(s["view"] == pending and s["access"] in (p.WRITE, p.READ_WRITE) for s in graph["state"])


def test_glm_vision_uses_shared_feature_bindings_and_preserves_text_only_contract():
    from ignition.recipes.glm53.contract import prepare

    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32, region=lambda _: SimpleNamespace(offset=0, nbytes=8)))
    text = {"prefill": GraphSignature("prefill", 1, 128,
        ("token_ids", "vision_token_indices", "vision_feature_slot"))}
    plain = ParseDict(prepare(text, {}, pool, page_tokens=256, draft_lag=None), p.ModelContract())
    assert plain.vision.max_images == 0 and not plain.frontend.HasField("image_layout")
    assert 154854 in plain.frontend.rejected_tokens
    signatures = {**text, "vision64": GraphSignature("vision", 1, 64,
        ("vision_pixel_values", "vision_position_ids", "vision_cu_seqlens", "vision_num_patches",
         "vision_slot", "vision_generation", "vision_feature_offset", "vision_feature_slot"))}
    vision = ParseDict(prepare(signatures, {}, pool, page_tokens=256, draft_lag=None, image_slots=2), p.ModelContract())
    assert vision.vision.feature_slots == 2 and vision.vision.features_per_request == 8192
    assert vision.frontend.image_layout == p.GLM_IMAGE_LAYOUT
    assert vision.frontend.image_preparation == p.GLM_IMAGE_PREPARATION
    assert vision.frontend.image_tokens.feature_token == 154854
    assert 154854 not in vision.frontend.rejected_tokens and 154855 in vision.frontend.rejected_tokens
    graph = next(graph for graph in vision.graphs if graph.kind == p.PREFILL)
    assert next(value for value in graph.inputs if value.buffer == "vision_token_indices").source == p.IMAGE_FEATURE_INDEX
    assert next(value for value in graph.inputs if value.buffer == "vision_feature_slot").source == p.IMAGE_FEATURE_SLOT


def test_history_plane_registers_actual_row_geometry():
    import torch
    tensor = torch.empty((8, 64, 12), dtype=torch.bfloat16)
    buffers = {}
    region = c.history_region(buffers, "keys", tensor, p.PAGED_KEY_VALUE)
    assert buffers["keys"] is tensor
    assert region["storage"] == dict(buffer="keys", offset=0, stride=1536, nbytes=1536)


@pytest.mark.parametrize("capacity", [1, 2, 4, 64])
def test_retained_proposal_views_cover_individual_request_rows(capacity):
    import torch
    from ignition.components.split_draft import PreparedProposalStorage
    from ignition.recipes.deepseek41.contract import prepare

    requests = SimpleNamespace(capacity=capacity, generation=torch.zeros(capacity, dtype=torch.int64), vocab_size=17)
    storage = PreparedProposalStorage(SimpleNamespace(requests=requests), 7)
    buffers = dict(storage.buffers(), proposal_confidence=torch.empty((capacity, 7)))
    pool = SimpleNamespace(pages={}, layout=SimpleNamespace(row_bytes=32,
        region=lambda name: SimpleNamespace(offset=24, nbytes=8)))
    signatures = {kind: GraphSignature(kind, capacity, 8 if kind == "proposal" else 1,
        tuple(c.text_inputs(kind)), speculation_enabled=True) for kind in ("proposal", "verification")}
    value = ParseDict(prepare(signatures, buffers, pool, page_tokens=128, draft_lag=0, rows=None), p.ModelContract())
    for graph in value.graphs:
        views = [s.view for s in graph.state if s.view.buffer.startswith("proposal_")]
        assert len(views) == len(buffers)
        for view in views:
            tensor = buffers[view.buffer]
            assert view.stride == tensor[0].numel() * tensor.element_size()
            assert (capacity - 1) * view.stride + view.offset + view.nbytes == tensor.nbytes


def test_unknown_packing_fields_do_not_acquire_sources_from_model_identity():
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32, region=lambda _: SimpleNamespace(offset=0, nbytes=8)))
    module = importlib.import_module("ignition.recipes.glm53.contract")
    value = module.prepare({"decode": GraphSignature("decode", 1, 1, ("token_ids", "unused_scratch"))},
        {}, pool, page_tokens=256, draft_lag=None)
    assert [v["buffer"] for v in value["graphs"][0]["inputs"]] == ["token_ids"]
    assert value["graphs"][0]["pending_token"] is None


def test_dense_qwen_preserves_device_decode_and_page_storage_in_typed_contract():
    import torch
    from ignition.components.paged_batch import BatchInputs
    from ignition.recipes.qwen38_dense.contract import prepare

    inputs = BatchInputs.allocate(128, 2, "cpu", capacity=4)
    buffers = inputs.buffers()
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=4096),
        pages={name: torch.empty((8, 128 * 4 * 256 * 2), dtype=torch.uint8)
               for name in ("layer3.key", "layer3.value")})
    signatures = {kind: GraphSignature(kind, 4, 128 if kind == "prefill" else 1, tuple(buffers))
                  for kind in ("prefill", "decode")}
    value = ParseDict(prepare(signatures, buffers, pool, vocabulary=248320), p.ModelContract())
    assert value.cache.page_tokens == 128 and value.vision.max_images == 0
    prefill, decode = value.graphs
    assert any(b.source == p.PROMPT_TOKEN and b.indexing == p.PACKED_TOKEN for b in prefill.inputs)
    assert all(b.source not in (p.PENDING_TOKEN, p.PROMPT_AND_PENDING_TOKEN) for b in decode.inputs)
    assert decode.pending_token.buffer == "request_pending_token"
    assert decode.pending_token.stride == decode.pending_token.nbytes == 8
    assert not decode.HasField("draft_cursor") and not decode.embedding_tables
    for graph in (prefill, decode):
        assert any(s.view == decode.pending_token and s.access == p.READ_WRITE for s in graph.state)
    assert [(t.buffer, t.page_offset) for t in value.cache.page_tables] == [("page_tables", 0)]
    pages = [r for r in value.cache.regions if r.kind == p.PAGED_KEY_VALUE]
    assert len(pages) == 2
    for region, (name, tensor) in zip(pages, pool.pages.items()):
        assert buffers[region.storage.buffer] is tensor
        assert region.storage.buffer == f"cache.{name}"
        assert region.storage.stride == region.storage.nbytes == 128 * 4 * 256 * 2


def test_prompt_scoring_is_an_additive_catalog_and_serialization_contract():
    from ignition.components.prompt_logprobs import prefill_graph_count, prefill_modes, scored_prefill_capacities
    from ignition.components.capacity import prefill_capacities

    prefills = prefill_capacities(4096)
    ordinary = prefill_modes(prefills, (1, 2, 4), None)
    assert ordinary == [('prefill', 'prefill', tuple(reversed(prefills))), ('decode', 'decode', (1, 2, 4))]
    scored = prefill_modes(prefills, (1, 2, 4), True)
    assert [part for part in scored if part[0] != 'scored_prefill'] == ordinary
    assert prefill_graph_count(prefills, None) == len(prefills)
    assert prefill_graph_count(prefills, True) == len(prefills) + 4
    assert scored_prefill_capacities(4096) == (128, 512, 2048, 4096)
    assert scored_prefill_capacities(64) == (64,)
    storage = c.state_view('state_pool', 32)
    def encode(kinds):
        signatures = {kind: GraphSignature(kind, 1, 128 if 'prefill' in kind else 1, tuple(c.text_inputs(kind)))
                      for kind in kinds}
        graphs = c.graph_contracts(signatures, {kind: c.text_inputs(kind) for kind in kinds},
                                   continuation=storage, pending=None)
        return [ParseDict(graph, p.GraphContract()) for graph in graphs]
    before = encode(['prefill', 'decode'])
    after = encode(['prefill', 'scored_prefill', 'decode'])
    assert [g.SerializeToString() for g in before] == [g.SerializeToString() for g in after if g.kind != p.SCORED_PREFILL]
    target = next(b for b in after[1].inputs if b.source == p.PROMPT_TARGET_TOKEN)
    assert target.indexing == p.PACKED_TOKEN and target.padding == -1
    assert all(b.source != p.PROMPT_TARGET_TOKEN for g in before for b in g.inputs)
    # Metadata exported before the new enum values retains its operation and inputs.
    old_prefill = p.GraphContract.FromString(before[0].SerializeToString())
    assert old_prefill.kind == p.PREFILL
    assert not any(b.source == p.PROMPT_TARGET_TOKEN for b in old_prefill.inputs)
