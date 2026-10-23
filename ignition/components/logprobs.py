"""Requested raw-distribution reductions with exact cutoff tie ordering."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import prepare_triton

from .completion import (
    EMITTED_COUNT_FIELD, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS, MAX_OUTPUT_TOKENS,
    MAX_LOGPROB_TOKEN_IDS, MAX_TOP_LOGPROBS, TOKEN_FIELD, VECTOR_RECORD_FIELDS, VECTOR_TOKENS_FIELD,
)

_VOCAB_TILE = 1024
# A global top-20 token is in its tile's top 20; round up to 32 for the merge sort.
_TILE_CANDIDATES = triton.next_power_of_2(MAX_TOP_LOGPROBS)
_REDUCTION_WARPS = 4
_MERGE_WARPS = 8


@triton.jit
def _sort_key(value, ids):
    bits = tl.where(value == 0., 0., value).to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return (ordered.to(tl.uint64) << 32) | (0xffffffff - ids.to(tl.uint32)).to(tl.uint64)


@triton.jit
def _sort_pair(keys):
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    return (0xffffffff - (keys & 0xffffffff)).to(tl.int32), bits.to(tl.float32, bitcast=True)


@triton.jit
def _position(Records, Frame, Requested, Slots, Epochs, Live, row, col,
              SLOTS: tl.constexpr, FIELDS: tl.constexpr,
              VECTOR_FIELDS: tl.constexpr = VECTOR_RECORD_FIELDS,
              COUNT_FIELD: tl.constexpr = EMITTED_COUNT_FIELD,
              TOKENS_FIELD: tl.constexpr = VECTOR_TOKENS_FIELD,
              CHOSEN_FIELD: tl.constexpr = TOKEN_FIELD):
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS)
    record = (tl.load(Frame) * SLOTS + slot) * FIELDS
    valid &= tl.load(Requested + slot, valid, other=0) != 0
    valid &= tl.load(Records + record, valid, other=0) == tl.load(Epochs + row)
    if FIELDS == VECTOR_FIELDS:
        valid &= col < tl.load(Records + record + COUNT_FIELD, valid, other=0)
        token = tl.load(Records + record + TOKENS_FIELD + col, valid, other=-1)
    else:
        token = tl.load(Records + record + CHOSEN_FIELD, valid, other=-1)
    return slot, token, valid & (token >= 0)


@triton.jit
def _statistics(Logits, Records, Frame, Requested, Slots, Epochs, Live, Stats,
                V: tl.constexpr, WIDTH: tl.constexpr, TILES: tl.constexpr,
                SLOTS: tl.constexpr, FIELDS: tl.constexpr, BLOCK: tl.constexpr):
    row, col, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    _, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        ids = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(Logits + (row * WIDTH + col) * V + ids, ids < V, other=-float("inf")).to(tl.float32)
        maximum = tl.max(x, 0)
        mass = tl.sum(tl.exp(x - tl.where(maximum == -float("inf"), 0., maximum)), 0)
        index = (row * WIDTH + col) * TILES + tile
        tl.store(Stats + index * 2, maximum)
        tl.store(Stats + index * 2 + 1, mass)


@triton.jit
def _normalization(Records, Frame, Requested, Slots, Epochs, Live, Stats, Lse,
                   WIDTH: tl.constexpr, TILES: tl.constexpr, SLOTS: tl.constexpr,
                   FIELDS: tl.constexpr, BLOCK: tl.constexpr):
    row, col = tl.program_id(0), tl.program_id(1)
    _, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        base = (row * WIDTH + col) * TILES
        tiles = tl.arange(0, BLOCK)
        maxima = tl.load(Stats + (base + tiles) * 2, tiles < TILES, other=-float("inf"))
        masses = tl.load(Stats + (base + tiles) * 2 + 1, tiles < TILES, other=0.)
        maximum = tl.max(maxima, 0)
        lse = maximum + tl.log(tl.sum(masses * tl.exp(maxima - maximum), 0))
        tl.store(Lse + row * WIDTH + col, lse)


@triton.jit
def _candidates(Logits, Records, Frame, Requested, Slots, Epochs, Live, Lse, Keys,
                V: tl.constexpr, WIDTH: tl.constexpr, TILES: tl.constexpr,
                SLOTS: tl.constexpr, FIELDS: tl.constexpr, BLOCK: tl.constexpr,
                CANDIDATES: tl.constexpr):
    row, col, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    _, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        ids = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(Logits + (row * WIDTH + col) * V + ids, ids < V, other=-float("inf")).to(tl.float32)
        x -= tl.load(Lse + row * WIDTH + col)
        # Rank the reported FP32 values, including ties introduced by subtraction.
        keys = tl.sort(tl.where(ids < V, _sort_key(x, ids), 0), descending=True)
        index = (row * WIDTH + col) * TILES + tile
        k = tl.arange(0, BLOCK)
        tl.store(Keys + index * CANDIDATES + k, keys, k < CANDIDATES)


@triton.jit
def _finish(Logits, Records, Frame, Requested, Slots, Epochs, Live, Lse, Keys, Output, TokenIds, Counts,
            V: tl.constexpr, WIDTH: tl.constexpr, TILES: tl.constexpr,
            SLOTS: tl.constexpr, FIELDS: tl.constexpr, BLOCK: tl.constexpr,
            CANDIDATES: tl.constexpr, TOP: tl.constexpr, IDS: tl.constexpr,
            OUTPUT_TOKENS: tl.constexpr, OUTPUT_FIELDS: tl.constexpr, PAIR_FIELDS: tl.constexpr):
    row, col = tl.program_id(0), tl.program_id(1)
    slot, token, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        out = Output + ((tl.load(Frame) * SLOTS + slot) * OUTPUT_TOKENS + col) * OUTPUT_FIELDS
        chosen = tl.load(Logits + (row * WIDTH + col) * V + token).to(tl.float32) - tl.load(Lse + row * WIDTH + col)
        tl.store(out, token.to(tl.int32))
        tl.store(out + 1, chosen.to(tl.int32, bitcast=True))
        count = tl.load(Counts + slot)
        if count > 0:
            selected = tl.arange(0, IDS)
            requested_ids = tl.load(TokenIds + slot * IDS + selected, selected < count, other=-1)
            requested_values = tl.load(Logits + (row * WIDTH + col) * V + requested_ids,
                             (selected < count) & (requested_ids >= 0) & (requested_ids < V), other=float("nan")).to(tl.float32)
            requested_values -= tl.load(Lse + row * WIDTH + col)
            tl.store(out + (1 + selected) * PAIR_FIELDS, requested_ids, selected < count)
            tl.store(out + (1 + selected) * PAIR_FIELDS + 1, requested_values.to(tl.int32, bitcast=True), selected < count)
        else:
            k = tl.arange(0, BLOCK)
            keys = tl.load(Keys + (row * WIDTH + col) * TILES * CANDIDATES + k, k < TILES * CANDIDATES, other=0)
            keys = tl.sort(keys, descending=True)
            ids, values = _sort_pair(keys)
            tl.store(out + (1 + k) * PAIR_FIELDS, ids, k < TOP)
            tl.store(out + (1 + k) * PAIR_FIELDS + 1, values.to(tl.int32, bitcast=True), k < TOP)


class PreparedLogprobs:
    def __init__(self, logits, completion, slots, epochs, live, frame):
        if logits.ndim == 2:
            logits = logits[:, None, :]
        batch, width, vocab = logits.shape
        if not logits.is_contiguous() or logits.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("raw logprob logits require contiguous BF16 or FP32 storage")
        tiles = triton.cdiv(vocab, _VOCAB_TILE)
        cursor = completion.scratch.cursor()
        self.stats = cursor.zeros((batch, width, tiles, 2), torch.float32)  # Maximum and exponential mass.
        self.lse = cursor.zeros((batch, width), torch.float32)
        self.keys = cursor.zeros((batch, width, tiles, _TILE_CANDIDATES), torch.uint64)
        common = (logits, completion.records, frame, completion.requested, slots, epochs, live)
        dimensions = (vocab, width, tiles, completion.records.shape[1], completion.records.shape[2])
        self._preparation = (
            (_statistics, (*common, self.stats, *dimensions, _VOCAB_TILE), (batch, width, tiles), _REDUCTION_WARPS),
            (_normalization, (*common[1:], self.stats, self.lse, *dimensions[1:],
                triton.next_power_of_2(tiles)), (batch, width, 1), _REDUCTION_WARPS),
            (_candidates, (*common, self.lse, self.keys, *dimensions, _VOCAB_TILE,
                _TILE_CANDIDATES), (batch, width, tiles), _REDUCTION_WARPS),
            (_finish, (*common, self.lse, self.keys, completion.logprobs, completion.token_ids,
                completion.token_id_counts, *dimensions,
                triton.next_power_of_2(tiles * _TILE_CANDIDATES), _TILE_CANDIDATES,
                MAX_TOP_LOGPROBS, MAX_LOGPROB_TOKEN_IDS, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS),
                (batch, width, 1), _MERGE_WARPS),
        )
        self.statistics = self.normalization = self.candidates = self.finish = None

    def prepare(self):
        if self.statistics is not None:
            return
        operations = tuple(prepare_triton(kernel, args, grid, num_warps=warps)
                           for kernel, args, grid, warps in self._preparation)
        self.statistics, self.normalization, self.candidates, self.finish = operations
        self._preparation = ()

    def __call__(self):
        if self.statistics is None:
            raise RuntimeError("logprob reduction prepare() must finish before capture")
        self.statistics()
        self.normalization()
        self.candidates()
        self.finish()


@triton.jit
def _merge(Output, Prompt, Decode, Counts, PromptSlot, Frame,
           SLOTS: tl.constexpr, FIELDS: tl.constexpr, TOKENS: tl.constexpr,
           TOP: tl.constexpr, BLOCK: tl.constexpr):
    slot, col = tl.program_id(0), tl.program_id(1)
    field = tl.arange(0, BLOCK)
    width = tl.maximum(TOP, tl.load(Counts + slot))
    mask = field < 2 * (1 + width)
    offset = ((tl.load(Frame) * SLOTS + slot) * TOKENS + col) * FIELDS + field
    if slot == tl.load(PromptSlot):
        value = tl.load(Prompt + offset, mask, other=0)
    else:
        value = tl.load(Decode + offset, mask, other=0)
    tl.store(Output + offset, value, mask)
