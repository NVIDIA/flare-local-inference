"""Window address equality across packed capacities and decoder replay bounds."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.window_indices import PreparedWindowIndices
from ignition.recipes.deepseek4.attention import AttentionIndices
from ignition.recipes.deepseek41.decoder import PreparedDecoderBoundary
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix
from ignition.recipes.deepseek41.attention import PreparedAttention
from ignition.recipes.deepseek41.attention import AttentionIndices as DeepSeek41Indices
from ignition.preparation.scratch import GraphBufferPool
from .window_indices_reference import LiteralDeepSeek4WindowIndices, LiteralDeepSeek41WindowIndices


FIELDS = ("window_slots", "window_indices", "window_lengths")


def assert_metadata_equal(actual, expected):
    for field in FIELDS:
        a, b = getattr(actual, field), getattr(expected, field)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b), field


def poison(indices):
    for field in FIELDS:
        getattr(indices, field).fill_(-777)
    if isinstance(indices, DeepSeek41Indices):
        indices.position.fill_(-777)
        for shared in indices.ratios.values():
            for value in vars(shared).values():
                value.fill_(-777)


def set_metadata(metadata, starts, lengths):
    device = metadata.positions.device
    starts = torch.tensor(starts, dtype=torch.int64, device=device)
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    rows, width = len(starts), metadata.max_query_tokens
    pages = torch.arange(rows * metadata.page_table.shape[1], dtype=torch.int32,
                         device=device).view_as(metadata.page_table)
    tokens = torch.arange(rows * width, device=device).view(rows, width)
    metadata.set(starts, lengths, torch.arange(rows, device=device) + 2, pages, tokens,
                 (lengths > 0).sum().view(1))


@pytest.mark.parametrize("mode,tokens,rows,query_capacity", [
    ("decode", 8, 8, 1), ("verify", 24, 8, 6), ("verify", 8, 8, 6),
    ("verify", 128, 32, 6), ("draft", 40, 8, 5), ("prefill", 4096, 1, 4096),
])
def test_deepseek41_window_formula_full_buffers(mode, tokens, rows, query_capacity):
    metadata = Metadata.allocate(tokens, rows, 40, "cpu", max_query_tokens=query_capacity)
    page_size = 128 if mode == "decode" else 128 + query_capacity
    width = page_size if mode == "draft" else 128
    replay_start = torch.zeros(rows, dtype=torch.int64) if mode == "prefill" else None
    actual = PreparedWindowIndices(metadata, mode=mode, tokens=tokens, page_size=page_size,
                                   width=width, replay_start=replay_start)
    expected = LiteralDeepSeek41WindowIndices(metadata, mode=mode, tokens=tokens, page_size=page_size,
                                             width=width, replay_start=replay_start)
    for start in (0, 127, 128, 4095):
        for partial in (False, True, None):
            per_row = min(query_capacity, tokens // rows)
            lengths = [per_row] * rows
            if partial is True:
                lengths[0] = max(0, per_row - 1)
                lengths[-1] = 0
            elif partial is None:
                lengths = [0] * rows
            set_metadata(metadata, [start + i for i in range(rows)], lengths)
            if replay_start is not None:
                replay_start.copy_(metadata.starts if partial is False else torch.zeros_like(replay_start))
            poison(actual)
            poison(expected)
            actual()
            expected()
            assert_metadata_equal(actual, expected)
    if tokens == 24 and rows == 8:
        assert actual.window_page_size == 134


@pytest.mark.parametrize("mode,visible", [("decode", False), ("verify", False),
    ("draft", False), ("prefill", False), ("prefill", True)])
def test_deepseek4_wrapper_preserves_literal_window_formula(mode, visible):
    rows, width = (1, 256) if mode == "prefill" else (3, 1 if mode == "decode" else 5)
    tokens = rows * width
    metadata = Metadata.allocate(tokens, rows, 40, "cpu")
    visibility = None
    if visible:
        visibility = (torch.arange(tokens) % 193, torch.arange(tokens).flip(0) % 193)
    buffers = GraphBufferPool(torch.device("cpu"))
    actual = AttentionIndices(metadata, mode=mode, tokens=tokens, visible=visibility, buffers=buffers.cursor())
    expected = LiteralDeepSeek4WindowIndices(metadata, mode=mode, tokens=tokens,
        page_size=actual.window_page_size, width=actual.offsets.numel(), visible=visibility)
    compressed = actual.compressed(4)
    for start in (0, 127, 128, 4095):
        for lengths in ([width] * rows, [width - 1] + [0] * (rows - 1), [0] * rows):
            set_metadata(metadata, [start] * rows, lengths)
            poison(actual)
            poison(expected)
            actual()
            expected()
            assert_metadata_equal(actual, expected)
            positions = metadata.positions.clamp_min(0)
            assert torch.equal(compressed.lengths, torch.where(metadata.positions >= 0, (positions + 1) // 4, 0))


@pytest.mark.parametrize("start", [0, 127, 128, 4095])
def test_decoder_boundary_refreshes_suffix_after_packing(start):
    metadata = Metadata.allocate(4096, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(metadata)
    suffix = packing.metadata
    shared = PreparedWindowIndices(suffix, mode="prefill", tokens=128, page_size=256,
        width=128, replay_start=packing.replay_start)
    expected = LiteralDeepSeek41WindowIndices(suffix, mode="prefill", tokens=128, page_size=256,
        width=128, replay_start=packing.replay_start)
    boundary = object.__new__(PreparedDecoderBoundary)
    boundary.packing, boundary.window_metadata = packing, shared
    boundary.incoming, boundary.rows, boundary.normalized = (), (), None
    observed = []
    def consumer(*args):
        expected()
        assert_metadata_equal(shared, expected)
        observed.append(shared.window_lengths.clone())
    boundary.mhc = consumer
    boundary.global_kv = consumer
    for count in (4096, 129, 128, 1, 127, 0, 4095):
        set_metadata(metadata, [start], [count])
        poison(shared)
        boundary(None)
        assert packing.replay_start.item() == (start + max(0, count - 128) if count > 128 else 0)
    assert len(observed) == 14


def test_shared_window_buffers_survive_layer_consumption_and_capacity_switches():
    groups = []
    for tokens in (8, 24, 128):
        metadata = Metadata.allocate(tokens, 8, 40, "cpu", max_query_tokens=6)
        shared = PreparedWindowIndices(metadata, mode="verify", tokens=tokens, page_size=134, width=128)
        oracle = LiteralDeepSeek41WindowIndices(metadata, mode="verify", tokens=tokens, page_size=134, width=128)
        groups.append((metadata, shared, oracle))
    # Distinct prepared capacities can run sequentially without retaining stale addresses.
    for turn, index in enumerate((0, 1, 2, 1, 0, 2, 0)):
        metadata, shared, oracle = groups[index]
        lengths = [min(6, metadata.positions.numel() // 8)] * 8
        if turn % 2:
            lengths[-1] = 0
        set_metadata(metadata, [4095 + turn] * 8, lengths)
        poison(shared)
        shared()
        oracle()
        before = tuple(getattr(shared, field).clone() for field in FIELDS)
        for _ in range(40):
            assert_metadata_equal(shared, oracle)
        for field, snapshot in zip(FIELDS, before):
            assert torch.equal(getattr(shared, field), snapshot)


@pytest.mark.parametrize("mismatch", ["metadata", "mode", "page_size", "width", "replay_start", "visible"])
def test_attention_rejects_incompatible_shared_window_indices(mismatch):
    metadata = Metadata.allocate(24, 8, 1, "cpu", max_query_tokens=6)
    other = Metadata.allocate(24, 8, 1, "cpu", max_query_tokens=6)
    arguments = dict(metadata=metadata, mode="verify", tokens=24, page_size=134, width=128)
    if mismatch == "metadata":
        arguments["metadata"] = other
    elif mismatch == "mode":
        arguments["mode"] = "prefill"
    elif mismatch == "page_size":
        arguments["page_size"] = 131
    elif mismatch == "width":
        arguments["width"] = 134
    elif mismatch == "replay_start":
        arguments["replay_start"] = torch.zeros_like(metadata.starts)
    else:
        arguments["visible"] = (torch.zeros(24), torch.zeros(24))
    helper = DeepSeek41Indices(**arguments)
    config = SimpleNamespace(text={"compress_ratios": [0], "rms_norm_eps": 1e-6}, source=lambda layer: None)
    pool = SimpleNamespace(config=config, window=lambda *args, **kwargs: torch.zeros((1, 128 * 528), dtype=torch.uint8))
    with pytest.raises(ValueError, match="shared window indices must match"):
        PreparedAttention(0, {}, torch.zeros((24, 1)), None, metadata, pool, None,
            mode="verify", world=1, session=None, name="attention", workspace=None, window_metadata=helper)


def test_standalone_attention_refreshes_its_own_window_metadata(monkeypatch):
    from ignition.recipes.deepseek41 import attention
    metadata = Metadata.allocate(24, 8, 1, "cpu", max_query_tokens=6)
    config = SimpleNamespace(text={"compress_ratios": [0], "rms_norm_eps": 1e-6}, source=lambda layer: None)
    pool = SimpleNamespace(config=config, window=lambda *args, **kwargs: torch.zeros((10, 128 * 528), dtype=torch.uint8))
    scratch = SimpleNamespace(projected=torch.zeros((24, 1792)),
        query_low=torch.zeros((24, 1280)), query=torch.zeros((24, 16, 512)),
        window_values=torch.zeros((24, 512)), attended=torch.zeros((24, 16, 512)),
        output=torch.zeros((24, 5120)))
    for name in ("QuantizedProjection", "PreparedRMSNorm", "PreparedGroupedOutputProjection", "PreparedCompressedCacheWriter"):
        monkeypatch.setattr(attention, name, lambda *args, **kwargs: lambda *args: None)
    consumed = []
    def prepare_consumer(query, cache, indices, lengths, *args, **kwargs):
        return lambda: consumed.append((indices.clone(), lengths.clone()))
    monkeypatch.setattr(attention, "PreparedCompressedMLA", prepare_consumer)
    weights = dict.fromkeys(("qkv_a", "q_b", "q_norm", "kv_norm", "output", "sink"))
    operation = PreparedAttention(0, weights, torch.zeros((24, 5120)), scratch, metadata, pool,
        SimpleNamespace(apply=lambda *args, **kwargs: None), mode="verify", world=4,
        session=None, name="attention", workspace=None)
    assert operation.owns_window_metadata
    oracle = LiteralDeepSeek41WindowIndices(metadata, mode="verify", tokens=24, page_size=134, width=128)
    for start, lengths in ((127, [3] * 8), (4095, [2, 3, 3, 3, 3, 3, 3, 0])):
        set_metadata(metadata, [start] * 8, lengths)
        poison(operation.window_metadata)
        operation(torch.zeros((24, 5120)))
        oracle()
        assert_metadata_equal(operation.window_metadata, oracle)
        assert torch.equal(consumed[-1][0], oracle.window_indices)
        assert torch.equal(consumed[-1][1], oracle.window_lengths)


def test_decoder_window_groups_survive_reused_prefill_layer_buffers(monkeypatch):
    from ignition.recipes.deepseek41 import decoder

    metadata = Metadata.allocate(256, 1, 4, "cpu")
    pool = GraphBufferPool("cpu", chunk_bytes=2 << 20)
    buffers = pool.cursor()
    scratch = SimpleNamespace(buffers=buffers, embedding=torch.zeros((256, 5120)),
        hidden=torch.zeros((1, 5120)), local_logits=torch.zeros((1, 1)), logits=torch.zeros((1, 1)))
    @dataclass
    class Layer:
        engram: object
        attention: dict

    layers = [Layer(engram=None, attention={}) for _ in range(3)]
    weights = SimpleNamespace(config=SimpleNamespace(decoder_start=1, sources=(), text={"compress_ratios": [0, 0, 0]}),
        layers=layers, draft=None, embedding=None, head=None, shard=SimpleNamespace(world_size=4))
    private = []

    def prepare_layer(*args, buffers, window_metadata, **kwargs):
        private.append(tuple(buffers.zeros((256, 128), dtype)
                             for dtype in (torch.int64, torch.int32)))
        return SimpleNamespace(window_metadata=window_metadata)

    monkeypatch.setattr(decoder, "PreparedLayer", prepare_layer)
    monkeypatch.setattr(decoder, "PreparedDecoderBoundary", lambda *args, **kwargs: None)
    for name in ("PreparedTableRequests", "PreparedEmbedding", "Rotary", "PreparedVocabularyProjection"):
        monkeypatch.setattr(decoder, name, lambda *args, **kwargs: None)
    operation = decoder.PreparedDecoder(weights, None, scratch, metadata, SimpleNamespace(),
        mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None,
        session=None, name="decoder", workspace=None, suffix_scratch=scratch)
    original, suffix = operation.window_metadata, operation.suffix_window_metadata
    assert operation.layers[0].window_metadata is original
    assert all(layer.window_metadata is suffix for layer in operation.layers[1:])
    assert private[0][0].data_ptr() == private[1][0].data_ptr() == private[2][0].data_ptr()
    for index, group in enumerate((original, suffix)):
        for field in FIELDS:
            getattr(group, field).fill_(index + 11)
    for layer_private in private:
        for value in layer_private:
            value.fill_(42)
    for index, group in enumerate((original, suffix)):
        for field in FIELDS:
            assert getattr(group, field).eq(index + 11).all()


def test_draft_window_group_survives_layer_buffers(monkeypatch):
    from ignition.recipes.deepseek41 import dspark

    metadata = Metadata.allocate(15, 3, 4, "cpu", max_query_tokens=5)
    buffers = GraphBufferPool("cpu", chunk_bytes=2 << 20).cursor()
    scratch = SimpleNamespace(buffers=buffers, embedding=torch.zeros((15, 5120)),
        hidden=torch.zeros((3, 5120)), local_logits=torch.zeros((3, 1)), logits=torch.zeros((3, 1)))
    weights = SimpleNamespace(config=None, embedding=None, head=None, shard=None,
        draft=SimpleNamespace(layers=(None, None), draft_head=None))
    private = []

    def prepare_layer(*args, window_metadata, **kwargs):
        private.append(tuple(buffers.zeros((15, 133), dtype)
                             for dtype in (torch.int64, torch.int32)))
        return SimpleNamespace(window_metadata=window_metadata)

    monkeypatch.setattr(dspark, "PreparedLayer", prepare_layer)
    for name in ("PreparedEmbedding", "Rotary", "PreparedVocabularyProjection"):
        monkeypatch.setattr(dspark, name, lambda *args, **kwargs: None)
    operation = dspark.PreparedDraft(weights, scratch, metadata, None, runtime=None,
        vocabulary_runtime=None, coordinator=None, session=None, name="draft", workspace=None)
    shared = operation.window_metadata
    assert all(layer.window_metadata is shared for layer in operation.layers)
    for field in FIELDS:
        getattr(shared, field).fill_(17)
    for layer_private in private:
        for value in layer_private:
            value.fill_(42)
    for field in FIELDS:
        assert getattr(shared, field).eq(17).all()
