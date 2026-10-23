"""QSA candidate races use active private inputs with the serving tensor ABI."""

import gc
import os
from types import SimpleNamespace
from weakref import ref

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.preparation.fixtures import qsa as qsa_fixtures
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.attention import (
    PreparedAttention, QSAMetadata, QSAScratch, QSAState, RotaryTables,
)
from ignition.recipes.qwen38.tuning import preparation_session


def _operands(rows, batch, *, context=256, device="cpu", drafts=7):
    device = torch.device(device)
    state = QSAState.allocate(pages=1, slots=batch + 1, device=device,
                              draft_tokens=drafts, kv_heads=1)
    metadata = QSAMetadata.allocate(tokens=rows, batch=batch, context=context, device=device)
    scratch = QSAScratch.allocate(rows, device, world_size=2)
    binding = {**vars(state), **{name: getattr(metadata, name) for name in
        ("main_block_table", "compressed_block_table", "raw_state_slot_ids")},
        "output": scratch.attended, "selected_positions": scratch.selected_positions,
        "index_q_norm_weight": torch.zeros(128, device=device),
        "index_k_norm_weight": torch.zeros(128, device=device),
        "rope_cos": torch.ones(context, 32, device=device),
        "rope_sin": torch.zeros(context, 32, device=device),
        "k_descale": torch.ones(1, device=device), "v_descale": torch.ones(1, device=device)}
    dynamic = {"query": scratch.query,
        "index_query": scratch.projected_index[:, :512].view(rows, 4, 128),
        "raw_index_key": scratch.projected_index[:, 512:],
        **{name: getattr(metadata, name) for name in
           ("request_ids", "query_positions", "rope_positions", "sequence_lengths",
            "query_start_loc", "num_accepted_tokens", "is_prefilling")}}
    caps = SimpleNamespace(max_batch=batch, max_seq_len=context,
        main_page_size=64, max_speculative_tokens=drafts,
        raw_ring_capacity=state.raw_k_ring.shape[1],
        num_main_cache_pages=context // 64 + rows, num_compressed_cache_pages=context // 64 + rows)
    return caps, binding, dynamic


@pytest.mark.parametrize("rows,batch,prefill", [(65, 1, True), (4, 4, False), (18, 4, False)])
def test_active_qsa_fixture_preserves_inputs_layout_and_restores_selector_state(rows, batch, prefill):
    caps, binding, dynamic = _operands(rows, batch)
    fixture = qsa_fixtures.QSABenchmarkInputs(caps, binding, dynamic)
    d, b = fixture.dynamic, fixture.binding
    for name, value in dynamic.items():
        assert d[name].shape == value.shape and d[name].stride() == value.stride()
        assert d[name].data_ptr() != value.data_ptr()
    assert d["index_query"].stride(0) == d["raw_index_key"].stride(0) == 640
    assert d["query_start_loc"][-1] == rows
    assert torch.all(d["request_ids"] >= 0)
    assert torch.all(d["num_accepted_tokens"] == 1)
    assert torch.all(d["is_prefilling"] == prefill)
    for request in range(batch):
        start, end = d["query_start_loc"][request:request + 2].tolist()
        positions = d["query_positions"][start:end]
        assert d["request_ids"][start:end].tolist() == [request] * (end - start)
        assert positions.tolist() == list(range(positions[0], positions[0] + end - start))
        assert d["sequence_lengths"][request] == positions[-1] + 1
        assert positions[-1] < caps.max_seq_len
    history = d["query_positions"][0].item()
    table = b["main_block_table"]
    assert torch.equal(table, b["compressed_block_table"])
    suffixes = table[:, history // 64:]
    suffixes = suffixes[suffixes >= 0]
    assert suffixes.unique().numel() == suffixes.numel()
    for name in fixture.seeds:
        b[name].fill_(17)
    fixture.produce()
    for name, seed in fixture.seeds.items():
        assert torch.equal(b[name], seed)
    assert torch.all(dynamic["request_ids"] == -1)
    assert torch.all(dynamic["query_start_loc"] == 0)
    assert torch.all(binding["raw_logical_positions"] == -1)
    assert torch.all(binding["compressed_k_cache"] == 0)
    assert fixture.nbytes > 0


def test_qsa_candidate_inputs_release_private_pages_after_the_race():
    caps, binding, dynamic = _operands(16, 1)
    acquire = qsa_fixtures.input_factory(caps, binding, dynamic)
    fixture = acquire()
    assert acquire() is fixture
    owner = ref(fixture)
    del fixture
    gc.collect()
    assert owner() is None
    assert acquire() is not None


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")
@pytest.mark.parametrize("rows,batch", [(4096, 1), (32, 32), (128, 32)])
def test_qwen_qsa_request_tunes_active_candidates_and_replays(rows, batch):
    from b12x.attention import qsa

    device = torch.device("cuda:0")
    context = 262144
    drafts = 7
    state = QSAState.allocate(pages=1, slots=batch + 1, device=device, draft_tokens=drafts, kv_heads=1)
    metadata = QSAMetadata.allocate(tokens=rows, batch=batch, context=context, device=device)
    scratch = QSAScratch.allocate(rows, device, world_size=2)
    source = torch.zeros(rows, 2560, device=device, dtype=torch.bfloat16)
    norm = SimpleNamespace(weight=torch.zeros(128, device=device))
    module = SimpleNamespace(config=SimpleNamespace(rms_norm_eps=1e-6),
                             indexer=SimpleNamespace(q_layernorm=norm, k_layernorm=norm))
    weights = {"qkv_proj": torch.zeros(scratch.projected_qkv.shape[1], 2560,
                                      device=device, dtype=torch.bfloat16),
               "indexer.index_qk_proj": torch.zeros(640, 2560, device=device, dtype=torch.bfloat16),
               "o_proj": torch.zeros(2560, scratch.gated.shape[1], device=device, dtype=torch.bfloat16)}
    rotary = RotaryTables(torch.ones(context, 32, device=device), torch.zeros(context, 32, device=device),
                          torch.zeros(32, device=device, dtype=torch.int64))
    workspace = ScratchArena(device)
    session = preparation_session(device=device, checkpoint="qsa-benchmark-test", autotune_effort="medium")
    session.configure_compile_workers(2)
    try:
        attention = PreparedAttention(module, weights, source, scratch, state, metadata, rotary,
            session=session, name=f"prefill{rows}.layer3.qsa", draft_tokens=drafts,
            workspace=workspace, max_cache_pages=8192)
        request = session._recorded_requests[f"prefill{rows}.layer3.qsa.qsa"]
        assert request.benchmark_call is not None
        with PreparationSession.prepare(session, (request,)) as result:
            assert result.coverage[request.name]["effective_count"] == 3
            print(f"QSA {rows=} {batch=}: benchmarked={result.benchmarked_candidates}, "
                  f"restored={result.restored_candidates}, cache_hits={result.cache_hits}")
            fixture = qsa_fixtures.QSABenchmarkInputs(attention.plan.prepared.state.caps, attention.binding_kwargs,
                                                     attention.dynamic())
            storage = workspace.views(*((spec.shape, spec.dtype) for spec in attention.plan.scratch_specs()))[0]
            binding = qsa.bind(attention.plan, scratch=storage, **fixture.binding)
            fixture.produce()
            qsa.run(binding, **fixture.dynamic)
            torch.cuda.synchronize()
            expected = fixture.binding["output"].clone()
            selected = fixture.binding["selected_positions"].clone()
            assert torch.isfinite(expected).all()
            assert (selected >= 0).any()
            fixture.produce()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    qsa.run(binding, **fixture.dynamic)
                fixture.produce()
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                torch.testing.assert_close(fixture.binding["output"], expected, rtol=0, atol=0)
                assert torch.equal(fixture.binding["selected_positions"], selected)
            finally:
                graph.reset()
        assert torch.all(metadata.request_ids == -1)
        assert torch.all(state.raw_logical_positions == -1)
        assert torch.all(state.compressed_k_cache == 0)
    finally:
        session.close()
