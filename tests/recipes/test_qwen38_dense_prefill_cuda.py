"""Captured Qwen GDN prefill with independent token and sequence counts."""

import os
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("checkpoint,online_mxfp8", [("qwen3.8-27b", False), ("qwen3.8-27b-qad", False), ("qwen3.8-27b-qad", True)])
@torch.no_grad()
def test_checkpoint_captured_chunked_prefill_preserves_counts_and_state(checkpoint, online_mxfp8):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import Metadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.qwen38_dense.decoder import PreparedDecoder, Scratch
    from ignition.recipes.qwen38_dense.pool import ContinuationPool
    from ignition.recipes.qwen38_dense.weights import ModelWeights

    device = torch.device("cuda:0")
    hub = Path.home() / ".cache/huggingface/hub"
    target = resolve(checkpoint).snapshot(hub)
    draft = hub / "models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"
    weights = ModelWeights.load(inspect_indexed_checkpoint(target), device, mtp=True, draft_snapshot=draft, online_mxfp8=online_mxfp8)
    pool = ContinuationPool.for_model(weights.config, 3, 2, device, mtp=True)
    metadata = Metadata.allocate(128, 1, 1, device)
    source, target_buffers, context_buffers = (GraphBufferPool(device, chunk_bytes=2 << 20) for _ in range(3))
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    rows = torch.ones(1, dtype=torch.int64, device=device)
    pages = torch.zeros((1, 1), dtype=torch.int32, device=device)
    tokens = torch.arange(11, 139, device=device).reshape(1, 128)
    live_rows = torch.ones(1, dtype=torch.int32, device=device)

    def inputs(start, count):
        starts.fill_(start)
        lengths.fill_(count)
        metadata.set(starts, lengths, rows, pages, tokens + start, live_rows)

    with PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)
        scratch = Scratch.allocate(128, 1, weights.config, device, buffers=source.cursor())
        decoder = PreparedDecoder(weights, scratch, metadata, pool, mode="prefill", vocabulary_runtime=None,
            coordinator=None, session=session, name="qwen.actual.prefill128", workspace=arena, buffers=target_buffers.cursor())
        context = PreparedContext(weights, decoder.auxiliary, metadata, pool, session=session,
            name="qwen.actual.prefill.context", workspace=arena, runtime=None, coordinator=None, buffers=context_buffers.cursor())
        arena.finalize()

        def forward():
            decoder()
            context(decoder.auxiliary, metadata.lengths)

        inputs(0, 17)
        compiled = compile_forward(forward)
        pool.clear_for_preparation()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        pool.clear_for_preparation()
        errors = []
        for start, count in ((0, 17), (17, 13)):
            inputs(start, count)
            before = pool.backing.clone()
            before_pages = {name: value.clone() for name, value in pool.pages.items()}
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert decoder.delta.num_seqs.item() == 1
            assert decoder.delta.num_tokens.item() == count
            actual = pool.backing.clone()
            actual_pages = {name: value.clone() for name, value in pool.pages.items()}
            actual_hidden = scratch.hidden.clone()
            actual_features = decoder.auxiliary[:count].clone()
            assert torch.equal(actual[0], before[0])
            assert torch.equal(actual[2], before[2])
            pool.backing.copy_(before)
            for name, value in before_pages.items():
                pool.pages[name].copy_(value)
            forward()
            torch.cuda.synchronize()
            relative = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
            errors.append(dict(start=start, tokens=count, hidden_relative_l2=relative(actual_hidden, scratch.hidden),
                feature_relative_l2=relative(actual_features, decoder.auxiliary[:count])))
            pool.backing.copy_(actual)
            for name, value in actual_pages.items():
                pool.pages[name].copy_(value)
        print(dict(chunked_prefill=errors, sequence_count=1, capacity=128), flush=True)
        graph.reset()
