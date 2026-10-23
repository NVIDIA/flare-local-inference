"""Packed attention preparation preserves FP8 casts, inactive rows and cache slots."""

import os
import weakref

import pytest
import torch

from ignition.components.msa_attention import PreparedPackedAttention
from ignition.components.msa_attention import PreparedMSAIndexer
from ignition.components.paged_attention import PreparedPagedAttention
from ignition.components.paged_batch import Metadata
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.scratch import ScratchArena

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU"
)


@pytest.fixture
def private_fixtures(monkeypatch):
    import ignition.components.msa_attention as module

    references = []
    original = module.benchmark_resource

    def tracked(session, key, factory):
        def allocate():
            values, seed = factory()
            indices = (0, 3, 4, 5, 6, 7, 8, 9, 10, 11) if key[0] == "packed_gqa_index_prepare" else range(7)
            references.extend(weakref.ref(values[index]) for index in indices)
            references.append(weakref.ref(seed))
            return values, seed
        return original(session, key, allocate)

    monkeypatch.setattr(module, "benchmark_resource", tracked)
    return references


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("start", [125, 1048569])
@pytest.mark.parametrize("separate_index_table", [False, True])
@torch.no_grad()
def test_packed_gemma_rope_index_cache_graph(cache_dtype, start, separate_index_table, private_fixtures):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding

    device = torch.device("cuda", 0)
    torch.manual_seed(411)
    rows, hq, hkv, hi, dim = 7, 16, 1, 1, 128
    projected = torch.randn(
        rows, (hq + 2 * hkv + hi + 1) * dim, device=device
    ).bfloat16()
    norms = tuple((torch.randn(dim, device=device) * 0.1).bfloat16() for _ in range(4))
    config = MiniMaxM3VLTextConfig(head_dim=128, rotary_dim=64,
        rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=.5))
    inverse = MiniMaxM3VLRotaryEmbedding.compute_default_rope_parameters(config, device=device)[0]
    blocks = max(4, -(-(start + rows) // 128))
    metadata = Metadata.allocate(rows, 1, blocks, device)
    metadata.requests.zero_()
    metadata.positions.copy_(torch.arange(start, start + rows, device=device))
    metadata.positions[-1] = -1
    metadata.page_table.copy_(torch.tensor([2, 0, 3, 1], device=device).repeat(-(-blocks // 4))[:blocks][None])
    index_pages = metadata.page_table.roll(1, dims=1) if separate_index_table else None
    q = torch.empty(rows, hq, dim, device=device, dtype=torch.bfloat16)
    iq = torch.empty(rows, hi, dim, device=device, dtype=torch.float8_e4m3fn)
    backing = torch.empty(
        (4, 2 * 128 * hkv * dim + 64), device=device, dtype=cache_dtype
    )
    kc = backing[:, : 128 * hkv * dim].view(4, 128, hkv, dim)
    kc.fill_(17)
    vc = backing[:, 128 * hkv * dim : 2 * 128 * hkv * dim].view(4, 128, hkv, dim)
    vc.fill_(23)
    index_backing = torch.full(
        (4, 128 * dim + 64), 29, device=device, dtype=torch.float8_e4m3fn
    )
    ik = index_backing[:, : 128 * dim].view(4, 128, dim)
    with RecordedPreparationSession(
        device=device,
        checkpoint="msa-packed",
        workload_schema="msa-packed",
        numerical_policy="default",
    ) as session:
        operation = PreparedPackedAttention(
            projected,
            *norms,
            inverse,
            metadata,
            q,
            iq,
            kc,
            vc,
            ik,
            epsilon=1e-6,
            session=session,
            name="packed",
            index_page_table=index_pages,
        )
        session.tune()
        assert private_fixtures and all(reference() is None for reference in private_fixtures)
        compiled = compile_forward(operation)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        kc.fill_(17)
        vc.fill_(23)
        ik.fill_(29)
        projected.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        x = projected.view(rows, -1, dim).float()
        role = torch.tensor(
            [0] * hq + [1] * hkv + [1] * hkv + [2] * hi + [3], device=device
        )
        weights = torch.stack(norms).float()[role]
        normalized = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + weights)
        )
        normalized[:, hq + hkv : hq + 2 * hkv] = x[:, hq + hkv : hq + 2 * hkv]
        angle = metadata.positions.float()[:, None] * inverse
        cosine, sine = (
            angle.cos().bfloat16().float()[:, None],
            angle.sin().bfloat16().float()[:, None],
        )
        a, b = normalized[..., :32].clone(), normalized[..., 32:64].clone()
        normalized[..., :32] = a * cosine - b * sine
        normalized[..., 32:64] = b * cosine + a * sine
        normalized[:, hq + hkv : hq + 2 * hkv] = x[:, hq + hkv : hq + 2 * hkv]
        torch.testing.assert_close(
            q[:-1].float(),
            normalized[:-1, :hq].bfloat16().float(),
            rtol=0.008,
            atol=0.008,
        )
        torch.testing.assert_close(
            iq[:-1].float(),
            normalized[:-1, hq + 2 * hkv : hq + 2 * hkv + hi]
            .bfloat16()
            .to(iq.dtype)
            .float(),
            rtol=0,
            atol=0,
        )
        assert bool((q[-1] == 0).all()) and bool((iq[-1].float() == 0).all())
        expected_k, expected_v, expected_i = (
            torch.full_like(kc, 17),
            torch.full_like(vc, 23),
            torch.full_like(ik, 29),
        )
        for row in range(rows - 1):
            pos = int(metadata.positions[row])
            page = int(metadata.page_table[0, pos // 128])
            expected_k[page, pos % 128] = normalized[row, hq : hq + hkv].to(cache_dtype)
            expected_v[page, pos % 128] = x[row, hq + hkv : hq + 2 * hkv].to(
                cache_dtype
            )
            index_page = page if index_pages is None else int(index_pages[0, pos // 128])
            expected_i[index_page, pos % 128] = normalized[row, -1].to(ik.dtype)
        torch.testing.assert_close(
            kc.float(), expected_k.float(), rtol=0.008, atol=0.016
        )
        torch.testing.assert_close(vc.float(), expected_v.float(), rtol=0, atol=0)
        torch.testing.assert_close(ik.float(), expected_i.float(), rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("invalid", ["batch", "width", "rank", "dtype", "integer_dtype", "layout", "device"])
def test_packed_attention_rejects_incompatible_index_page_table(invalid):
    device = torch.device("cuda", 0)
    rows, heads, dim = 2, 16, 128
    projected = torch.empty(rows, (heads + 4) * dim, device=device, dtype=torch.bfloat16)
    norms = tuple(torch.ones(dim, device=device, dtype=torch.bfloat16) for _ in range(4))
    inverse = torch.ones(32, device=device)
    metadata = Metadata.allocate(rows, 2, 2, device)
    index_pages = metadata.page_table.clone()
    if invalid == "batch":
        index_pages = index_pages[:1]
    elif invalid == "width":
        index_pages = index_pages[:, :1].contiguous()
    elif invalid == "rank":
        index_pages = index_pages.flatten()
    elif invalid == "dtype":
        index_pages = index_pages.float()
    elif invalid == "integer_dtype":
        index_pages = index_pages.long()
    elif invalid == "layout":
        index_pages = index_pages.t()
    else:
        index_pages = index_pages.cpu()
    query = torch.empty(rows, heads, dim, device=device, dtype=torch.bfloat16)
    index_query = torch.empty(rows, 1, dim, device=device, dtype=torch.float8_e4m3fn)
    key = torch.empty(4, 128, 1, dim, device=device, dtype=torch.bfloat16)
    index_key = torch.empty(4, 128, dim, device=device, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="index page table"):
        PreparedPackedAttention(projected, *norms, inverse, metadata, query, index_query,
            key, torch.empty_like(key), index_key, epsilon=1e-6, session=None,
            name="invalid.index_pages", index_page_table=index_pages)


@pytest.mark.parametrize(
    "mode,width",
    [("decode", 1), ("verify", 1), ("verify", 2), ("verify", 7), ("verify", 8),
     ("prefill", 33), ("prefill", 4096)],
)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_msa_indexer_attention_compiled_graph(mode, width, cache_dtype, private_fixtures):
    device = torch.device("cuda", 0)
    torch.manual_seed(412)
    batch, heads, pages = (1 if width == 4096 else 2), 16, max(20, (width + 255) // 128)
    padding = 3 if width > 1 else 0
    rows = batch * width + padding
    query = (torch.randn(rows, heads, 128, device=device) * 0.2).bfloat16()
    iq = torch.randn(rows, 1, 128, device=device).to(torch.float8_e4m3fn)
    backing = (torch.randn(batch * pages, 2 * 128 * 128 + 128 * 64, device=device) * 0.2).to(
        cache_dtype
    )
    key = backing[:, : 128 * 128].view(batch * pages, 128, 1, 128)
    value = backing[:, 128 * 128 : 2 * 128 * 128].view(batch * pages, 128, 1, 128)
    index_backing = torch.randn(batch * pages, 128 * 128 + 64, device=device).to(
        torch.float8_e4m3fn
    )
    ik = index_backing[:, : 128 * 128].view(batch * pages, 128, 128)
    table = torch.arange(batch * pages, device=device, dtype=torch.int32).view(
        batch, pages
    )
    lengths = torch.tensor(
        [max(2305, width + 129), 2049][:batch], device=device, dtype=torch.int32
    )
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    positions = torch.cat(
        tuple(
            torch.arange(int(lengths[i]) - width, int(lengths[i]), device=device)
            for i in range(batch)
        )
        + (torch.full((padding,), -1, device=device),)
    )
    selected = torch.empty(1, rows, 16, device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    with RecordedPreparationSession(
        device=device,
        checkpoint="msa-pipeline",
        workload_schema="msa-pipeline",
        numerical_policy="default",
    ) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        index = PreparedMSAIndexer(
            iq,
            ik,
            table,
            lengths,
            cu,
            positions,
            selected,
            session=session,
            workspace=arena,
            name="index",
        )
        attention = PreparedPagedAttention(
            query,
            key,
            value,
            table,
            lengths,
            cu,
            output,
            mode=mode,
            width=width,
            selected_blocks=selected,
            session=session,
            workspace=arena,
            name="attention",
        )
        session.tune()
        assert private_fixtures and all(reference() is None for reference in private_fixtures)

        def forward():
            index()
            attention()
            return output

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for shift in (0, -1):
            lengths.add_(shift)
            positions[: batch * width].add_(shift)
            arena.tensor.fill_(255)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            for req in range(batch):
                keys = key[table[req].long(), :, 0].float().flatten(0, 1)
                values = value[table[req].long(), :, 0].float().flatten(0, 1)
                for row in sorted({req * width, (req + 1) * width - 1}):
                    ids = selected[0, row].long()
                    ids = ids[ids >= 0]
                    tokens = (
                        ids[:, None] * 128 + torch.arange(128, device=device)
                    ).flatten()
                    tokens = tokens[tokens <= positions[row]]
                    weights = (
                        query[row].float() @ keys[tokens].T / (128**0.5)
                    ).softmax(-1)
                    expected = weights @ values[tokens]
                    torch.testing.assert_close(
                        output[row].float(), expected, atol=0.003, rtol=0.03
                    )
        if mode in ("decode", "verify"):
            cu[-1] = width
            lengths[-1] = 0
            positions[width:] = -1
            arena.tensor.fill_(255)
            graph.replay()
            torch.cuda.synchronize()
            assert bool((selected[:, width:] == -1).all())
        graph.reset()
