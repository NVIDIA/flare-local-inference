"""Checkpoint draft attention constants survive shared scratch clearing and reuse."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.recipes.minimax_m3.config import ModelConfig
from ignition.recipes.minimax_m3.draft import (
    PREFIX, DraftConfig, DraftScratch, _Attention, _DraftLoader,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or "FLARE_MINIMAX_CHECKPOINT" not in os.environ,
    reason="requires an assigned CUDA device and the local MiniMax checkpoint",
)


@torch.no_grad()
def test_checkpoint_draft_attention_constants_survive_catalog_scratch_reuse():
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 4)
    snapshot = Path(os.environ["FLARE_MINIMAX_CHECKPOINT"])
    target = ModelConfig.load(snapshot)
    config = replace(target, draft=DraftConfig.load(snapshot, target))
    d = config.draft
    prefix = PREFIX + ".layers.0.decoder_layer.self_attn"
    qr = tuple(x * d.head_dim for x in shard.partition(d.heads))
    kr = tuple(x * d.head_dim for x in shard.partition(d.kv_heads))
    with _DraftLoader(inspect_indexed_checkpoint(snapshot / "dspark"), device,
                      prefixes=(prefix + ".",)) as loader:
        loader.config, loader.shard = d, shard
        weights = SimpleNamespace(
            qkv=loader.load_linear_group(tuple(prefix + "." + role + "_proj" for role in ("q", "k", "v")),
                                         row_ranges=(qr, kr, kr)),
            output=loader.load_linear(prefix + ".o_proj", columns=qr),
            query_norm=loader.norm(prefix + ".q_norm.weight", d.head_dim),
            key_norm=loader.norm(prefix + ".k_norm.weight", d.head_dim),
        )
    torch.manual_seed(183)
    history = {f"draft0.{role}": (torch.randn(3, d.window, 1, d.head_dim, device=device) * .2).to(torch.float8_e4m3fn)
               for role in ("key", "value")}
    pool = SimpleNamespace(reserved_region=history.__getitem__)
    buffers, arena = GraphBufferPool(device), ScratchArena(device, overlap=False)
    owners, graphs = [], []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for rows in (2, 1):
            cursor = buffers.cursor()
            tokens = rows * d.block_size
            scratch = DraftScratch.allocate(config, shard, tokens, tokens, device, buffers=cursor)
            starts = torch.tensor([4094, 4103][:rows], device=device)
            metadata = SimpleNamespace(
                starts=starts, lengths=torch.full((rows,), d.block_size, dtype=torch.int32, device=device),
                positions=(starts[:, None] + torch.arange(d.block_size, device=device)).flatten(),
                state_rows=torch.arange(1, rows + 1, device=device),
                query_start_loc=torch.arange(rows + 1, dtype=torch.int32, device=device) * d.block_size,
            )
            source = torch.randn(tokens, config.hidden, dtype=torch.bfloat16, device=device) * .1
            attention = _Attention(0, weights, config, shard, scratch, metadata, pool, block=d.block_size,
                runtime=None, coordinator=None, session=session, name=f"draft.storage.{rows}",
                workspace=arena, buffers=cursor)
            compiled = compile_forward(lambda a=attention, x=source: a(x))
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            owners.append((attention, source, compiled, attention.pages.clone()))
            graphs.append(graph)
        expected = []
        for graph, (attention, _, _, _) in zip(graphs, owners):
            graph.replay()
            expected.append(attention.s.output.clone())
        for fill in (0, 0x7f, 0):
            for chunks in buffers.chunks.values():
                for chunk in chunks:
                    chunk.view(torch.uint8).fill_(fill)
            for graph, (attention, _, _, pages), output in zip(graphs, owners, expected):
                torch.testing.assert_close(attention.pages, pages, rtol=0, atol=0)
                assert all(attention.pages.untyped_storage().data_ptr() != chunk.untyped_storage().data_ptr()
                           for chunks in buffers.chunks.values() for chunk in chunks)
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                torch.testing.assert_close(attention.s.output, output, rtol=0, atol=0)
        for graph in graphs:
            graph.reset()
