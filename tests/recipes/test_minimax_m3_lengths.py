"""Graph-level cache lengths shared by every decoder layer."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.minimax_m3 import decoder as module


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fused", [False, True])
def test_decoder_layers_share_mutable_cache_lengths(monkeypatch, device, fused):
    if device != "cpu" and os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires an assigned CUDA device")
    rows, layers = 8, 60
    hidden = torch.zeros((rows, 4), device=device)
    observed = torch.empty((layers, rows), dtype=torch.int32, device=device)
    metadata = SimpleNamespace(lengths=torch.zeros(rows, dtype=torch.int32, device=device),
        starts=torch.zeros(rows, dtype=torch.int32, device=device),
        sample_indices=torch.arange(rows, device=device), live_counts=torch.tensor([rows], device=device),
        query_start_loc=torch.arange(rows + 1, device=device, dtype=torch.int32),
        token_ids=torch.zeros(rows, dtype=torch.int64, device=device))
    scratch = SimpleNamespace(embedding=hidden, local_embedding=hidden, partial=hidden,
        normalized=hidden, residual=hidden.clone(), auxiliary=hidden, hidden=hidden.clone(),
        attention=None, local_logits=hidden, logits=hidden,
        expert_routes=torch.zeros((rows, 4, 4), device=device),
        router=SimpleNamespace(ids=torch.zeros((rows, 4), dtype=torch.int32, device=device),
                               weights=torch.zeros((rows, 4), device=device)))
    config = SimpleNamespace(text={"rms_norm_eps": 1e-6, "num_local_experts": 128}, target_layers=(),
        rotary_frequencies=lambda device: torch.ones(32, device=device))
    weights = SimpleNamespace(config=config, shard=None, embedding=None, norm=None, head=None,
        draft=None, layers=[SimpleNamespace(input_norm=None) for _ in range(layers)])

    class Layer:
        def __init__(self, index, attention):
            self.attention, self.output = attention, observed[index]

        def normalize(self, update):
            return hidden

        def __call__(self, source):
            self.output.copy_(self.attention.lengths)
            return hidden

    monkeypatch.setattr(module.PreparedAllReduceRMSNorm, "fits", lambda *args: fused)
    monkeypatch.setattr(module, "PreparedShardedEmbedding", lambda *args, **kwargs: lambda: hidden)
    monkeypatch.setattr(module, "PreparedAttention", lambda *args, cache_lengths, **kwargs:
        SimpleNamespace(lengths=cache_lengths))
    monkeypatch.setattr(module, "PreparedFeedForward", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "CanonicalPreparedLayer", lambda index, attention, *args, **kwargs:
        Layer(index, attention))
    monkeypatch.setattr(module, "PreparedDecoderLayer", lambda attention, *args, name, **kwargs:
        Layer(int(name.rsplit("layer", 1)[1]), attention))
    monkeypatch.setattr(module, "PreparedIndexedRMSNorm", lambda *args, **kwargs: lambda source, indices: source)
    monkeypatch.setattr(module, "PreparedVocabularyProjection", lambda *args, **kwargs: lambda source: None)
    buffers = GraphBufferPool(device)
    decoder = module.PreparedDecoder(weights, scratch, metadata, None, mode="verify", runtime=None,
        embedding_runtime=None, vocabulary_runtime=None, coordinator=None, session=None,
        name="decoder", workspace=None, buffers=buffers.cursor())
    assert all(layer.attention.lengths is decoder.cache_lengths for layer in decoder.layers)
    assert decoder.scratch_tensors() == (scratch, decoder.cache_lengths, None)
    forward = decoder if device == "cpu" else compile_forward(decoder)
    graph = None
    if device != "cpu":
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
    starts = [0, 127, 128, 4095, 4096, 1048570, 65535, 131071]
    for counts in ([1, 6, 0, 1, 6, 6, 0, 2], [0] * rows, [6, 1, 6, 0, 1, 0, 6, 6]):
        metadata.starts.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
        metadata.lengths.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
        expected = torch.tensor([start + count if count > 0 else 0 for start, count in zip(starts, counts)],
            dtype=torch.int32, device=device).expand(layers, -1)
        decoder.cache_lengths.fill_(-197)
        observed.fill_(-197)
        allocated = torch.cuda.memory_allocated() if graph is not None else None
        if graph is None:
            forward()
        else:
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    if graph is not None:
        graph.reset()
