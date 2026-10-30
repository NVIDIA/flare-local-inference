"""Prepared GQA retains strided cache layouts and live metadata under replay."""

import os
import gc
import weakref

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


def test_gqa_compact_requests_preserve_inactive_holes():
    device = torch.device("cuda", 0)
    cases = {}
    with RecordedPreparationSession(device=device, checkpoint="gqa-compact-requests",
            workload_schema="gqa-compact-requests", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        for selected in (False, True):
            for width in (1, 8):
                batch, heads, dim = 4, 16 if selected else 6, 128 if selected else 256
                query = torch.randn((batch * width, heads, dim), device=device, dtype=torch.bfloat16) * .1
                key = (torch.randn((8, 128, 1, dim), device=device).bfloat16() * .1).to(torch.float8_e4m3fn)
                value = (torch.randn((8, 128, 1, dim), device=device).bfloat16() * .1).to(torch.float8_e4m3fn)
                output = torch.empty_like(query)
                table = torch.arange(8, device=device, dtype=torch.int32).view(batch, 2)
                lengths = torch.zeros(batch, device=device, dtype=torch.int32)
                cu = torch.zeros(batch + 1, device=device, dtype=torch.int32)
                blocks = None
                if selected:
                    blocks = torch.full((1, batch * width, 16), -1, dtype=torch.int32, device=device)
                    blocks[:, :, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
                attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
                    mode="decode" if width == 1 else "verify", width=width, selected_blocks=blocks,
                    session=session, workspace=arena, name=f"attention-{selected}-{width}")
                cases[selected, width] = attention
        session.tune()
        graphs = {}
        for shape, attention in cases.items():
            compiled = compile_forward(attention)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[shape] = graph, compiled
        session.freeze()
        for (selected, width), attention in cases.items():
            graph, compiled = graphs[selected, width]
            a = attention.arguments
            query, output, cu = ((a["q"], a["output"], a["cu_seqlens_q"]) if attention.compact_decode is None
                                 else attention.compact_decode[:3])
            key, value, lengths = a["k_cache"], a["v_cache"], a["cache_seqlens"]
            batch, dim = lengths.numel(), query.shape[-1]
            for counts in ([width, 0, 1, width], [0, 1, width, 0], [width] * batch):
                lengths.copy_(torch.tensor([128 + n if n else 0 for n in counts], device=device))
                cu.copy_(torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device=device))
                output.fill_(float("nan"))
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                begin = 0
                for row, count in enumerate(counts):
                    for local in range(count):
                        end = 129 + local
                        k = key.view(batch, 256, 1, dim)[row, :end, 0].float()
                        v = value.view(batch, 256, 1, dim)[row, :end, 0].float()
                        scores = query[begin + local].float() @ k.T / dim ** .5
                        torch.testing.assert_close(output[begin + local].float(), scores.softmax(-1) @ v,
                                                   atol=.002, rtol=.02)
                    begin += count
            graph.reset()


def test_selected_attention_releases_private_tuning_inputs(monkeypatch):
    import ignition.components.paged_attention as module

    references = []
    original = module.benchmark_resource

    def tracked(session, key, factory):
        def allocate():
            operands, seed = factory()
            references.extend(weakref.ref(operands[name]) for name in
                ("q", "output", "page_table", "cache_seqlens", "cu_seqlens_q", "q2k_indices"))
            references.append(weakref.ref(seed))
            return operands, seed
        return original(session, key, allocate)

    monkeypatch.setattr(module, "benchmark_resource", tracked)
    device = torch.device("cuda", 0)
    query = torch.randn((33, 16, 128), device=device, dtype=torch.bfloat16)
    key = torch.randn((20, 128, 1, 128), device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    table = torch.arange(20, device=device, dtype=torch.int32)[None]
    lengths = torch.tensor([2048], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 33], device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    selected = torch.arange(16, device=device, dtype=torch.int32).expand(1, 33, 16).contiguous()
    with RecordedPreparationSession(device=device, checkpoint="paged-fixture-lifetime",
            workload_schema="paged-fixture-lifetime", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            mode="prefill", selected_blocks=selected, session=session, workspace=arena, name="attention")
        session.tune()
        gc.collect()
        assert references and all(reference() is None for reference in references)
        assert attention.plan.prepared.owners == (arena,)
        attention()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("metadata_layout", ["int32", "int64", "strided"])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_primary_paged_verification_declares_query_width(monkeypatch, metadata_layout, ragged, selected):
    from b12x.attention import paged_decode

    monkeypatch.setattr(paged_decode, "supports", lambda **kwargs: False)
    device = torch.device("cuda", 0)
    batch, width, heads, dimension, length = 2, 7, 16, 128, 137
    counts = (2, 3) if ragged else (width, width)
    query = torch.randn((sum(counts), heads, dimension), device=device, dtype=torch.bfloat16) * .1
    key = torch.randn((4, 128, 1, dimension), device=device, dtype=torch.bfloat16) * .1
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    table = torch.arange(4, device=device, dtype=torch.int32).reshape(batch, 2)
    lengths = torch.full((batch,), length, device=device, dtype=torch.int32)
    cu = torch.tensor([0, counts[0], sum(counts)], device=device, dtype=torch.int32)
    if metadata_layout == "int64":
        table, lengths, cu = (tensor.to(torch.int64) for tensor in (table, lengths, cu))
    elif metadata_layout == "strided":
        storage = torch.empty((batch, 4), device=device, dtype=torch.int32)
        storage[:, ::2].copy_(table)
        table = storage[:, ::2]
    initial_table = table.clone()
    blocks = None
    if selected:
        blocks = torch.full((1, query.shape[0], 16), -1, device=device, dtype=torch.int32)
        blocks[:, :, :2] = torch.tensor([0, 1], device=device, dtype=torch.int32)
    import ignition.components.paged_attention as module
    original = module.benchmark_resource

    def checked_resource(session, key, factory):
        def allocate():
            arguments, seed = factory()
            ends = arguments["cu_seqlens_q"].cpu()
            assert ends[0] == 0 and ends[-1] == query.shape[0]
            assert ((ends[1:] - ends[:-1]) <= width).all()
            return arguments, seed
        return original(session, key, allocate)

    monkeypatch.setattr(module, "benchmark_resource", checked_resource)
    with RecordedPreparationSession(device=device, checkpoint="primary-paged-verify",
            workload_schema="primary-paged-verify", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            mode="verify", width=width, selected_blocks=blocks, session=session, workspace=arena, name="attention")
        assert attention.arguments["q"].shape[0] == sum(counts)
        assert attention.arguments["output"].shape[0] == sum(counts)
        if selected:
            assert attention.arguments["q2k_indices"].shape[1] == sum(counts)
        assert not attention.decode
        session.tune()
        if selected:
            from b12x.attention import paged

            invalid = dict(attention.arguments, q2k_indices=blocks[:, :-1].contiguous())
            with pytest.raises(ValueError, match="smaller than the query tensor"):
                paged.bind(attention.plan, scratch=attention.scratch, **invalid, window_left=-1)
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for visible, reverse_pages in ((length, False), (71, True), (length, False)):
            lengths.fill_(visible)
            if ragged:
                counts = (4, 1) if reverse_pages else (2, 3)
                cu.copy_(torch.tensor([0, counts[0], sum(counts)], device=device, dtype=cu.dtype))
            table.copy_(initial_table.flip(1) if reverse_pages else initial_table)
            query.normal_(std=.1)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            begin = 0
            for row, count in enumerate(counts):
                for local in range(count):
                    end = visible - count + local + 1
                    k = key[table[row].long()].reshape(256, 1, dimension)[:end, 0].float()
                    v = value[table[row].long()].reshape(256, 1, dimension)[:end, 0].float()
                    scores = query[begin + local].float() @ k.T / dimension ** .5
                    expected = scores.softmax(-1) @ v
                    torch.testing.assert_close(output[begin + local].float(), expected,
                                               atol=.002, rtol=.02,
                                               msg=lambda detail: f"selected={selected} ragged={ragged} visible={visible} reverse={reverse_pages} row={row} local={local}: {detail}")
                begin += count
        graph.reset()


@pytest.mark.parametrize("mode,tokens", [("decode", 4), ("prefill", 64), ("prefill", 4096)])
@pytest.mark.parametrize("dimension,value_dimension,heads,cache_dtype", [
    (192, 128, 16, torch.bfloat16), (192, 192, 8, torch.bfloat16), (256, 256, 6, torch.bfloat16),
    (192, 128, 16, torch.float8_e4m3fn), (256, 256, 6, torch.float8_e4m3fn)])
def test_gqa_strided_pages_and_sinks(mode, tokens, dimension, value_dimension, heads, cache_dtype):
    device = torch.device("cuda", 0)
    rows = tokens if mode == "decode" else 1
    pages = max(2, (tokens + 127) // 128)
    key_size, value_size = 128 * dimension, 128 * value_dimension
    padding = 16384 if dimension == 256 else 12288 if value_dimension == 192 else 8192
    backing = (torch.randn((pages, key_size + value_size + padding), device=device, dtype=torch.bfloat16) * .1).to(cache_dtype)
    key = backing[:, :key_size].view(pages, 128, 1, dimension)
    value = backing[:, key_size:key_size + value_size].view(pages, 128, 1, value_dimension)
    if mode == "prefill":
        key, value = key.contiguous(), value.contiguous()
    query = torch.randn((tokens, heads, dimension), device=device, dtype=torch.bfloat16) * .1
    output = torch.empty((tokens, heads, value_dimension), device=device, dtype=torch.bfloat16)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None].repeat(rows, 1)
    lengths = torch.full((rows,), max(128, tokens), device=device, dtype=torch.int32)
    cu = torch.arange(rows + 1, device=device, dtype=torch.int32) * (tokens // rows)
    sink = torch.linspace(-1, 1, heads, device=device)
    with RecordedPreparationSession(device=device, checkpoint="paged-gqa",
            workload_schema="paged-gqa", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            sink=sink, mode=mode, window=127, session=session, workspace=arena, name="attention")
        session.tune()
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        cases = ([(length, tokens) for length in (128, 33, 1)] if mode == "decode"
                 else [(max(128, tokens), tokens), (tokens, tokens),
                       (max(128, tokens), tokens // 2), (17, 17)])
        for length, live_tokens in cases:
            lengths.fill_(length)
            if mode == "prefill":
                cu[1] = live_tokens
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # Sample both ends and interior query positions without a context-sized reference matrix.
            for token in sorted({0, live_tokens // 2, live_tokens - 1}):
                position = length - 1 if mode == "decode" else length - live_tokens + token
                first, last = max(0, position - 127), position + 1
                k = key.flatten(0, 1)[first:last, 0].float()
                v = value.flatten(0, 1)[first:last, 0].float()
                logits = query[token].float() @ k.T / (dimension ** .5)
                probabilities = torch.cat((logits, sink[:, None]), 1).softmax(1)[:, :-1]
                expected = probabilities @ v
                torch.testing.assert_close(output[token].float(), expected, atol=.002, rtol=.02)
        graph.reset()


@pytest.mark.parametrize('dimension,window,causal', [(192, 128, True), (128, 1024, False)])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_parallel_verification_has_per_query_windows(dimension, window, causal, cache_dtype):
    device = torch.device('cuda', 0)
    pages_per_row = (window + 8 + 127) // 128
    key = (torch.randn((4 * pages_per_row, 128, 1, dimension), device=device).bfloat16() * .1).to(cache_dtype)
    value = (torch.randn((4 * pages_per_row, 128, 1, 128), device=device).bfloat16() * .1).to(cache_dtype)
    query = torch.randn((32, 16, dimension), device=device).bfloat16() * .1
    output = torch.empty((32, 16, 128), dtype=torch.bfloat16, device=device)
    table = torch.arange(4 * pages_per_row, dtype=torch.int32, device=device).view(4, pages_per_row)
    lengths = torch.full((4,), window + 8, dtype=torch.int32, device=device)
    cu = torch.arange(5, dtype=torch.int32, device=device) * 8
    sink = torch.linspace(-1, 1, 16, device=device)
    with RecordedPreparationSession(device=device, checkpoint='parallel-gqa',
            workload_schema='parallel-gqa', numerical_policy='default') as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            sink=sink, mode='verify', width=8, window=window - 1, causal=causal,
            session=session, workspace=arena, name='attention')
        session.tune()
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for counts in ([8, 0, 3, 1], [1, 1, 1, 1], [8, 8, 0, 8]):
            lengths.copy_(torch.tensor([window + n if n else 0 for n in counts], device=device))
            cu.copy_(torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device=device))
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            begin = 0
            for row, count in enumerate(counts):
                for local in range(count):
                    first = local + 1
                    last = window + local + 1 if causal else window + count
                    k = key.view(4, -1, 1, dimension)[row, first:last, 0].float()
                    v = value.view(4, -1, 1, 128)[row, first:last, 0].float()
                    logits = query[begin + local].float() @ k.T / dimension ** .5
                    probabilities = torch.cat((logits, sink[:, None]), 1).softmax(1)[:, :-1]
                    torch.testing.assert_close(output[begin + local].float(), probabilities @ v,
                                               atol=.002, rtol=.02)
                begin += count
        graph.reset()


@pytest.mark.parametrize("mode", ["verify", "prefill"])
@pytest.mark.parametrize("selected", [False, True])
def test_attention_candidate_producer_restores_shared_schedule(monkeypatch, tmp_path, mode, selected):
    import inspect

    from b12x.attention.paged._preparation import _PagedState
    from b12x.preparation import _measurement

    bindings, checked, addresses = {}, [], set()
    original_bind = _PagedState.bind
    original_timer_init = _measurement._TimedCall.__init__

    def bind(state, *args, **kwargs):
        binding = original_bind(state, *args, **kwargs)
        if not torch.cuda.is_current_stream_capturing():
            fields = tuple(getattr(binding.scratch, name)
                           for name in ("kv_chunk_size_ptr", "o_indptr"))
            bindings[id(binding)] = (binding, fields, tuple(t.cpu().clone() for t in fields))
        return binding

    def timer_init(timer, call, *args, **kwargs):
        original_timer_init(timer, call, *args, **kwargs)
        if not inspect.isfunction(call.run):
            return
        binding = inspect.getclosurevars(call.run).nonlocals.get("binding")
        if id(binding) not in bindings:
            return
        owner, fields, expected = bindings[id(binding)]
        pointers = tuple(t.data_ptr() for t in fields)
        addresses.add(pointers[0])
        producers = []
        for original_producer in timer.producers:
            def produce(original_producer=original_producer):
                # Other candidates reuse these locations before this call is timed.
                for field in fields:
                    field.fill_(-1)
                original_producer()
                assert tuple(t.data_ptr() for t in fields) == pointers
                for field, value in zip(fields, expected):
                    torch.testing.assert_close(field.cpu(), value, rtol=0, atol=0)
                checked.append(id(owner))
            producers.append(produce)
        timer.producers = tuple(producers)

    monkeypatch.setattr(_PagedState, "bind", bind)
    monkeypatch.setattr(_measurement._TimedCall, "__init__", timer_init)
    device = torch.device("cuda", 0)
    batch, width, heads, dim = 2, 8, 16 if selected else 24, 128 if selected else 256
    kv_heads = 1 if selected else 4
    rows = batch * width if mode == "verify" else 66
    query = torch.randn((rows, heads, dim), device=device, dtype=torch.bfloat16) * .1
    key = (torch.randn((32, 128, kv_heads, dim), device=device) * .1).to(torch.float8_e4m3fn)
    value = (torch.randn_like(key.float()) * .1).to(key.dtype)
    pages = torch.zeros((batch, 2048), dtype=torch.int32, device=device)
    lengths = torch.full((batch,), 4096, dtype=torch.int32, device=device)
    cu = torch.tensor([0, rows // 2, rows], dtype=torch.int32, device=device)
    blocks = torch.arange(16, dtype=torch.int32, device=device).expand(kv_heads, rows, 16).contiguous() if selected else None
    with RecordedPreparationSession(device=device, checkpoint="attention-shared-schedule",
            workload_schema="attention-shared-schedule", numerical_policy="default",
            namespace={"test_directory": str(tmp_path)}) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, pages, lengths, cu,
            torch.empty_like(query), mode=mode, width=width, selected_blocks=blocks,
            session=session, workspace=arena, name="attention")
        assert not attention.decode
        session.tune()
    assert len(set(checked)) > 1
    assert len(addresses) == 1


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("physical_pages", [3, 64])
def test_attention_tuning_uses_available_distinct_pages(monkeypatch, tmp_path, dim, physical_pages):
    import ignition.components.paged_attention as module
    from b12x.preparation import autotune_effort

    device = torch.device("cuda", 0)
    query = torch.randn((16, 6, dim), device=device, dtype=torch.bfloat16) * .1
    key = (torch.randn((physical_pages, 128, 1, dim), device=device) * .1).to(torch.float8_e4m3fn)
    value = torch.empty_like(key).copy_(key)
    table = torch.zeros((2, 32), device=device, dtype=torch.int32)
    lengths = torch.full((2,), 128, device=device, dtype=torch.int32)
    cu = torch.tensor([0, 8, 16], device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    original = module.benchmark_resource
    checked = []

    def tracked(session, name, factory):
        def allocate():
            operands, seed = factory()
            expected = torch.arange(64, device=device, dtype=torch.int64).reshape(2, 32) % physical_pages
            torch.testing.assert_close(operands["page_table"], expected.to(table.dtype), rtol=0, atol=0)
            assert operands["page_table"].data_ptr() != table.data_ptr()
            assert operands["k_cache"] is key and operands["v_cache"] is value
            assert torch.all(operands["cache_seqlens"] == 4096)
            assert torch.all(table == 0) and torch.all(lengths == 128)
            checked.append(True)
            return operands, seed
        return original(session, name, allocate)

    monkeypatch.setattr(module, "benchmark_resource", tracked)
    effort = "audit" if dim == 128 else "default"
    with autotune_effort(effort), RecordedPreparationSession(device=device, checkpoint=f"attention-page-footprint-{dim}-{physical_pages}",
            workload_schema="attention-page-footprint", numerical_policy="default",
            autotune_effort=effort,
            namespace={"test_directory": str(tmp_path)}) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            mode="verify", width=8, session=session, workspace=arena, name="attention")
        assert attention.decode == (dim == 128)
        assert attention.plan.query.autotune_effort == effort
        session.tune()
        attention()
        torch.cuda.synchronize()
        assert checked and torch.isfinite(output).all()
        assert torch.all(table == 0) and torch.all(lengths == 128)


def test_attention_tuning_respects_virtual_cache_backing(monkeypatch, tmp_path):
    import ignition.components.paged_attention as module
    from ignition.preparation.virtual_storage import VirtualStorage

    device = torch.device("cuda", 0)
    query = torch.randn((32, 24, 256), device=device, dtype=torch.bfloat16) * .1
    output = torch.empty_like(query)
    storage = VirtualStorage(device)
    backed, reserved = 40, 256
    key = storage.device_allocate("key", (reserved, 128, 4, 256), torch.float8_e4m3fn,
                                  group="KV_PAGES", maximum=reserved, units=backed)
    value = storage.device_allocate("value", (reserved, 128, 4, 256), torch.float8_e4m3fn,
                                    group="KV_PAGES", maximum=reserved, units=backed)
    key[:backed].fill_(.125)
    value[:backed].fill_(.25)
    assert all(region.nbytes < region.extent for region in storage.dynamic_regions)
    table = torch.zeros((4, 32), dtype=torch.int32, device=device)
    lengths = torch.full((4,), 128, dtype=torch.int32, device=device)
    cu = torch.arange(5, dtype=torch.int32, device=device) * 8
    original = module.benchmark_resource
    checked = []

    def tracked(session, name, factory):
        def allocate():
            operands, seed = factory()
            expected = (torch.arange(128, device=device).reshape(4, 32) % backed).to(table.dtype)
            torch.testing.assert_close(operands["page_table"], expected, rtol=0, atol=0)
            assert operands["k_cache"] is key and operands["v_cache"] is value
            assert torch.all(operands["cache_seqlens"] == 4096)
            checked.append(True)
            return operands, seed
        return original(session, name, allocate)

    monkeypatch.setattr(module, "benchmark_resource", tracked)
    with RecordedPreparationSession(device=device, checkpoint="attention-backed-pages",
            workload_schema="attention-backed-pages", numerical_policy="default",
            namespace={"test_directory": str(tmp_path)}) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            mode="verify", width=8, tuning_page_count=backed,
            session=session, workspace=arena, name="attention")
        session.tune()
        attention()
        torch.cuda.synchronize()
        assert checked and torch.isfinite(output).all()
        torch.testing.assert_close(output, torch.full_like(output, .25), rtol=.02, atol=.002)
        assert torch.all(table == 0) and torch.all(lengths == 128)
        assert torch.all(key[:backed].float() == .125) and torch.all(value[:backed].float() == .25)
    storage.close()


@pytest.mark.parametrize("count", [0, -1, 5, True, 1.0])
def test_attention_rejects_invalid_tuning_page_capacity(count):
    query = torch.empty((8, 6, 256), dtype=torch.bfloat16, device="cuda")
    key = torch.empty((4, 128, 1, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="backed page count"):
        PreparedPagedAttention(query, key, key, None, None, None, query,
            mode="verify", width=8, tuning_page_count=count, session=None, workspace=None, name="invalid")


@pytest.mark.parametrize("recipe", ["mimo_v2", "muse_spark"])
@pytest.mark.parametrize("mode", ["decode", "verify", "prefill"])
def test_window_attention_tuning_uses_backed_continuation_rows(recipe, mode, tmp_path):
    import importlib
    from types import SimpleNamespace
    from ignition.components.block_fp8 import BlockFP8Weight
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.virtual_storage import VirtualStorage

    attention_module = importlib.import_module(f"ignition.recipes.{recipe}.attention")
    pool_module = importlib.import_module(f"ignition.recipes.{recipe}.pool")
    device, shard = torch.device("cuda", 0), TensorParallelShard(0, 1)
    geometry = SimpleNamespace(heads=8, kv_heads=2, key_dim=128, value_dim=128,
        rotary_dim=64, theta=10000, window=128, local=lambda shard: (8, 2))
    config = SimpleNamespace(layers=1, draft=None, attention=lambda layer: geometry,
        qkv_chunks=lambda layer, shard: (),
        text={"hidden_size": 128, "rms_norm_eps": 1e-5, "qk_scale_factor": 1.0})
    tokens = 1 if mode == "decode" else 8
    source = torch.zeros((tokens, 128), dtype=torch.bfloat16, device=device)
    storage = VirtualStorage(device)
    pool = pool_module.ContinuationPool.allocate(config, shard, 3, 7, device,
        storage=storage, maximum_pages=32, maximum_rows=64, backed_slots=2)
    region = next(r for r in storage.dynamic_regions if r.group == "STATE_ROWS")
    assert region.nbytes < region.extent
    pool.backing[:2].zero_()
    scratch = attention_module.AttentionScratch.allocate(config, shard, tokens, device)
    metadata = Metadata.allocate(tokens, 1, 32, device)
    if recipe == "mimo_v2":
        qkv = BlockFP8Weight(torch.zeros((1536, 128), dtype=torch.float8_e4m3fn, device=device),
                            torch.ones((12, 1), dtype=torch.float32, device=device))
    else:
        qkv = torch.zeros((2560, 128), dtype=torch.bfloat16, device=device)
    weights = SimpleNamespace(qkv=qkv, sink=None,
        output=torch.zeros((128, 1024), dtype=torch.bfloat16, device=device))
    with RecordedPreparationSession(device=device, checkpoint="window-backed-rows",
            workload_schema="window-backed-rows", numerical_policy="default",
            namespace={"test_directory": str(tmp_path)}) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        prepared = attention_module.PreparedAttention(0, weights, config, shard, source,
            scratch, metadata, pool, mode=mode, runtime=None, coordinator=None,
            session=session, name="window", workspace=arena)
        operation = prepared.attention
        if mode == "decode":
            expected = 2 if recipe == "mimo_v2" else pool.window_page_stride(0) + 1
            assert operation.tuning_page_count == expected
            key = pool.cache(0, "key") if recipe == "mimo_v2" else pool.window_pages(0, "key")
            assert expected < key.shape[0]
            assert key[:expected].float().count_nonzero() == 0
            assert int(pool.memory_capacity[1]) == 7
        else:
            assert operation.tuning_page_count == prepared.staged_key.shape[0]
            assert prepared.staged_key.untyped_storage().data_ptr() != region.address
    torch.cuda.synchronize()
    storage.close()
