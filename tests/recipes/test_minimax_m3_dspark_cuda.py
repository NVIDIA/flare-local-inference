"""DSpark noncausal block windows and model-precision normalization."""

import os
from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.minimax_m3.draft import DraftConfig, norm_rope, stage_window

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("start", [3, 4096, 4103])
@torch.no_grad()
def test_noncausal_seven_position_block_preserves_each_query_window(start):
    torch.manual_seed(512)
    block, window, heads, dim = 7, 4096, 16, 128
    device = torch.device("cuda:0")
    history_key = (torch.randn((2, window, 1, dim), device=device) * .2).to(torch.float8_e4m3fn)
    history_value = (torch.randn_like(history_key.float()) * .2).to(torch.float8_e4m3fn)
    key = torch.randn((block, 1, dim), dtype=torch.bfloat16, device=device) * .2
    value = torch.randn_like(key) * .2
    query = (torch.randn((block, heads, dim), device=device) * .2).to(torch.float8_e4m3fn).to(torch.bfloat16)
    output = torch.empty_like(query)
    pages = -(-(window + block) // 128)
    staged_key = torch.empty((pages, 128, 1, dim), dtype=torch.float8_e4m3fn, device=device)
    staged_value = torch.empty_like(staged_key)
    metadata = SimpleNamespace(starts=torch.tensor([start], device=device),
        lengths=torch.tensor([block], dtype=torch.int32, device=device),
        state_rows=torch.ones(1, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, block], dtype=torch.int32, device=device))
    table = torch.arange(pages, dtype=torch.int32, device=device)[None]
    lengths = torch.tensor([min(start, window) + block], dtype=torch.int32, device=device)
    arena = ScratchArena(device, overlap=False)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        attention = PreparedPagedAttention(query, staged_key, staged_value, table, lengths,
            metadata.query_start_loc, output, mode="verify", width=block, causal=False,
            window=window - 1, session=session, workspace=arena, name="dspark.window")

        def forward():
            stage_window(history_key, key, staged_key, metadata, window)
            stage_window(history_value, value, staged_value, metadata, window)
            attention()

        compiled = torch.compile(forward, fullgraph=True)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for _ in range(3):
            query.neg_()
            value.neg_()
            staged_key.view(torch.uint8).fill_(0x7f)
            staged_value.view(torch.uint8).fill_(0x7f)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            prefix = min(start, window)
            positions = torch.arange(start - prefix, start, device=device)
            k = torch.cat((history_key[1, positions % window].float(), key.to(torch.float8_e4m3fn).float()))[:, 0]
            v = torch.cat((history_value[1, positions % window].float(), value.to(torch.float8_e4m3fn).float()))[:, 0]
            expected = []
            for index in range(block):
                lower = max(0, prefix + index - window + 1)
                scores = query[index].float() @ k[lower:].T / dim ** .5
                expected.append(scores.softmax(-1) @ v[lower:])
            expected = torch.stack(expected).to(torch.bfloat16)
            torch.testing.assert_close(output, expected, rtol=.025, atol=.0003)
        graph.reset()


@torch.no_grad()
def test_context_norm_rounding_and_fp8_query_values():
    device = torch.device("cuda:0")
    values = torch.randn((14, 1, 128), dtype=torch.bfloat16, device=device)
    weight = torch.randn(128, dtype=torch.bfloat16, device=device) * .1
    positions = torch.tensor([0, 1, 4090, 4095, 4096, 4103, 32768, 65535,
                              131071, 262143, 524287, 1048575, 2097151, 4194303], device=device)
    config = DraftConfig({"head_dim": 128, "rotary_dim": 64, "rope_theta": 10000000.,
                          "partial_rotary_factor": .5}, {}, ())
    inverse = config.rotary_frequencies(device)
    canonical = 1. / (10000000. ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    torch.testing.assert_close(inverse.cpu(), canonical, rtol=0, atol=0)
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRotaryEmbedding

    rotary = MiniMaxM3VLRotaryEmbedding(config.canonical_config()).to(device)
    cosine, sine = rotary(values, positions)
    output = torch.empty_like(values)
    for context in (False, True):
        norm_rope(values, weight, positions, inverse, 1e-6, output, fp8=not context,
                  round_normalized=context)
        x = values.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())
        if context:
            x = x.to(torch.bfloat16).float()
        c, s = cosine[..., :32].float()[:, None], sine[..., :32].float()[:, None]
        a, b = x[..., :32], x[..., 32:64]
        expected = torch.cat((a * c - b * s, b * c + a * s, x[..., 64:]), -1).bfloat16()
        if not context:
            expected = expected.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.no_grad()
def test_context_commits_only_accepted_prefixes_across_ring_boundary():
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.minimax_m3.draft import PreparedContext

    torch.manual_seed(171)
    device, rows, hidden, layers, window = torch.device("cuda:0"), 14, 128, 5, 4096
    config = SimpleNamespace(hidden=hidden, draft=DraftConfig(
        {"sliding_window": window, "num_hidden_layers": layers, "num_key_value_heads": 1,
         "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000000.,
         "rotary_dim": 64, "partial_rotary_factor": .5}, {}, ()))
    draft = SimpleNamespace(projection=torch.randn(hidden, layers * hidden, device=device).bfloat16() * .02,
        hidden_norm=torch.randn(hidden, device=device).bfloat16() * .1,
        context_kv=torch.randn(layers * 256, hidden, device=device).bfloat16() * .02,
        layers=tuple(SimpleNamespace(key_norm=torch.randn(128, device=device).bfloat16() * .1) for _ in range(layers)))
    weights = SimpleNamespace(config=config, draft=draft, shard=SimpleNamespace(world_size=1))
    specs = [(f"draft{i}.{role}", (window, 1, 128), torch.float8_e4m3fn)
             for i in range(layers) for role in ("key", "value")]
    storage = StateStorage.allocate(StateLayout.from_specs(specs), 3, device)
    pool = SimpleNamespace(backing=storage.backing, reserved_region=storage.region, region=storage.region)
    auxiliary = torch.randn(rows, layers, hidden, device=device).bfloat16()
    positions = torch.cat((torch.arange(4094, 4101), torch.arange(4090, 4097))).to(device)
    metadata = SimpleNamespace(positions=positions, starts=torch.tensor([4094, 4090], device=device),
        state_rows=torch.tensor([1, 2], device=device), requests=torch.arange(2, device=device).repeat_interleave(7))
    counts = torch.tensor([3, 5], dtype=torch.int32, device=device)
    buffers = GraphBufferPool(device).cursor()
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        context = PreparedContext(weights, auxiliary, metadata, pool, runtime=None, coordinator=None,
            session=session, name="context", workspace=None, buffers=buffers)
        torch.testing.assert_close(context.inverse, config.draft.rotary_frequencies(device), rtol=0, atol=0)
        from ignition.preparation.compilation import compile_forward

        compiled = compile_forward(lambda: context(auxiliary, counts))
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for accepted in ((3, 5), (0, 1), (7, 0)):
            pool.backing.fill_(0x7f)
            auxiliary.neg_()
            counts.copy_(torch.tensor(accepted, dtype=torch.int32, device=device))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            projected = torch.nn.functional.linear(auxiliary.flatten(1), draft.projection)
            normalized = projected.float() * torch.rsqrt(projected.float().square().mean(-1, keepdim=True) + 1e-6)
            normalized = (normalized * (1 + draft.hidden_norm.float())).bfloat16()
            kv = torch.nn.functional.linear(normalized, draft.context_kv).view(rows, layers, 2, 1, 128)
            matches = []
            for layer in range(layers):
                expected_key = torch.empty(rows, 1, 128, dtype=torch.bfloat16, device=device)
                norm_rope(kv[:, layer, 0], draft.layers[layer].key_norm, positions, context.inverse,
                          1e-6, expected_key, round_normalized=True)
                for request, count in enumerate(accepted):
                    for offset in range(7):
                        index = request * 7 + offset
                        slot = int(positions[index]) % window
                        for role, value in (("key", expected_key[index]), ("value", kv[index, layer, 1])):
                            actual = pool.region(f"draft{layer}.{role}")[request + 1, slot]
                            if offset < count:
                                expected = value.to(torch.float8_e4m3fn)
                                actual_bytes, expected_bytes = actual.view(torch.uint8), expected.view(torch.uint8)
                                equal = actual_bytes == expected_bytes
                                matches.append(equal.flatten())
                                distance = (actual_bytes.int() - expected_bytes.int()).abs()
                                zeros = (actual.float() == 0) & (expected.float() == 0)
                                assert ((distance <= 1) | zeros).all()
                            else:
                                assert actual.view(torch.uint8).eq(0x7f).all()
            if matches:
                assert torch.cat(matches).float().mean() >= .999
        graph.reset()
