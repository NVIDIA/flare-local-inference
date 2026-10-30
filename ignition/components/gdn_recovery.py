"""Gated DeltaNet verification with per-token correction records.

b12x's decode persists one FP32 recurrent checkpoint (heads x 128 x 128) per
verified token so that any accepted prefix can be resumed. Verification here
reads the committed state once and records, per token, the delta-rule
correction (one value vector per head), the normalized key and the decay.
After acceptance, one commit launch over every GDN layer replays the accepted
prefix into the committed slot, so each round reads and writes the recurrent
state once instead of writing it once per token.

The arithmetic follows b12x's CuTe Qwen recurrence: L2-normalized Q (scaled by
1/sqrt(128)) and K with epsilon 1e-6, decay exp(-exp(A_log) * softplus(a +
dt_bias)), the update gate sigmoid(b) rounded through BF16, then per token
S = decay * S; delta = (v - S k) * beta; S += delta k^T; o = S q.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ignition.preparation.triton import launch_triton, warmup_triton

# Verified tokens per round: the eight checkpoint columns of verification metadata.
WINDOW = 8
ROWS = 16
# One warp per request, value head and value-row tile. The launch configuration
# rides on the kernels so whole-forward compilation, which drops launch keywords, keeps it.
_CONFIG = triton.Config({}, num_warps=1, num_stages=2)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _verify(Mixed, A, B, ALog, DtBias, State, Starts, Slots, NumSeqs, Output, Delta, Keys, Decay,
            mixed_stride, a_stride, b_stride, state_stride, slot_stride,
            KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, SCALE: tl.constexpr,
            WINDOW: tl.constexpr, ROWS: tl.constexpr, IN_PLACE: tl.constexpr):
    tile, request_head = tl.program_id(0), tl.program_id(1)
    request = request_head // VALUE_HEADS
    head = request_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    if request < tl.load(NumSeqs):
        start = tl.load(Starts + request)
        end = tl.load(Starts + request + 1)
        slot = tl.load(Slots + request * slot_stride).to(tl.int64)
        rows = tile * ROWS + tl.arange(0, ROWS)
        keys = tl.arange(0, 128)
        if slot == 0:
            # A null slot carries no state: its tokens decode to zero.
            for position in tl.static_range(WINDOW):
                if position < end - start:
                    token = (start + position).to(tl.int64)
                    tl.store(Output + token * (VALUE_HEADS * 128) + head * 128 + rows,
                             tl.zeros((ROWS,), tl.bfloat16))
        else:
            offsets = slot * state_stride + head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
            state = tl.load(State + offsets)
            a_scale = tl.exp(tl.load(ALog + head).to(tl.float32))
            bias = tl.load(DtBias + head).to(tl.float32)
            for position in tl.range(0, WINDOW, loop_unroll_factor=1):
                if position < end - start:
                    token = (start + position).to(tl.int64)
                    q = tl.load(Mixed + token * mixed_stride + key_head * 128 + keys).to(tl.float32)
                    k = tl.load(Mixed + token * mixed_stride + KEY_HEADS * 128 + key_head * 128 + keys).to(tl.float32)
                    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
                    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
                    a = tl.load(A + token * a_stride + head).to(tl.float32) + bias
                    softplus = tl.where(a <= 20.0, libdevice.log1p(tl.exp(a)), a)
                    decay = tl.exp(-a_scale * softplus)
                    gate = tl.load(B + token * b_stride + head).to(tl.float32)
                    beta = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
                    value = tl.load(Mixed + token * mixed_stride + 2 * KEY_HEADS * 128 + head * 128 + rows).to(tl.float32)
                    state = state * decay
                    delta = (value - tl.sum(state * k[None, :], 1)) * beta
                    state = state + delta[:, None] * k[None, :]
                    output = tl.sum(state * q[None, :], 1)
                    tl.store(Output + token * (VALUE_HEADS * 128) + head * 128 + rows, output.to(tl.bfloat16))
                    if not IN_PLACE:
                        # Records are indexed by the request's row in the round, which the commit shares.
                        tl.store(Delta + ((request * VALUE_HEADS + head) * WINDOW + position) * 128 + rows, delta)
                        if (tile == 0) & (head % (VALUE_HEADS // KEY_HEADS) == 0):
                            tl.store(Keys + ((request * KEY_HEADS + key_head) * WINDOW + position) * 128 + keys, k)
                        if tile == 0:
                            tl.store(Decay + (request * VALUE_HEADS + head) * WINDOW + position, decay)
            if IN_PLACE:
                tl.store(State + offsets, state)


@triton.autotune(configs=[_CONFIG], key=[])
@triton.jit
def _commit(State, LayerOffsets, Delta, KeyRecords, Decay, Slots, Counts, NumSeqs, state_stride, slot_stride,
            ROWS_CAPACITY: tl.constexpr, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr, WINDOW: tl.constexpr,
            ROWS: tl.constexpr, STATE_ALIGNMENT: tl.constexpr):
    # Every argument a store reaches is a kernel operand, so whole-forward compilation sees the
    # mutation of the state backing: each layer's recurrent region sits at its offset in a row.
    tile, request, layer_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer = layer_head // VALUE_HEADS
    head = layer_head % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    if request < tl.load(NumSeqs):
        slot = tl.load(Slots + request * slot_stride).to(tl.int64)
        count = tl.load(Counts + request)
        if (slot != 0) & (count > 0):
            rows = tile * ROWS + tl.arange(0, ROWS)
            keys = tl.arange(0, 128)
            base = tl.multiple_of(tl.load(LayerOffsets + layer) + slot * state_stride, STATE_ALIGNMENT)
            offsets = base + head * (128 * 128) + rows[:, None] * 128 + keys[None, :]
            state = tl.load(State + offsets)
            # Records are stacked [layer, request row, head, position, 128].
            owner = layer * ROWS_CAPACITY + request
            for position in tl.range(0, WINDOW, loop_unroll_factor=1):
                if position < count:
                    k = tl.load(KeyRecords + ((owner * KEY_HEADS + key_head) * WINDOW + position) * 128 + keys)
                    delta = tl.load(Delta + ((owner * VALUE_HEADS + head) * WINDOW + position) * 128 + rows)
                    decay = tl.load(Decay + (owner * VALUE_HEADS + head) * WINDOW + position)
                    state = state * decay
                    state = state + delta[:, None] * k[None, :]
            tl.store(State + offsets, state)


@dataclass(frozen=True)
class GDNRecords:
    """Per-request correction, normalized key and decay records of one GDN layer's verified tokens."""

    delta: torch.Tensor
    keys: torch.Tensor
    decay: torch.Tensor


@dataclass(frozen=True)
class GDNLayerRecords:
    """Records of every GDN layer in stacked tensors, with one view per layer for verification."""

    layers: tuple[int, ...]
    delta: torch.Tensor
    keys: torch.Tensor
    decay: torch.Tensor

    @classmethod
    def allocate(cls, layers, rows: int, device, *, key_heads: int, value_heads: int) -> GDNLayerRecords:
        layers = tuple(layers)
        count = len(layers)
        return cls(layers,
                   torch.zeros((count, rows, value_heads, WINDOW, 128), dtype=torch.float32, device=device),
                   torch.zeros((count, rows, key_heads, WINDOW, 128), dtype=torch.float32, device=device),
                   torch.zeros((count, rows, value_heads, WINDOW), dtype=torch.float32, device=device))

    def __getitem__(self, layer: int) -> GDNRecords:
        index = self.layers.index(layer)
        return GDNRecords(self.delta[index], self.keys[index], self.decay[index])


class PreparedRecoveredRecurrence:
    """One layer's verification over packed requests, recording instead of checkpointing.

    Column zero of the verification metadata names each request's committed state
    row; records are indexed by the request's row in the round, so they hold one
    entry per request the round can verify.
    Output is the recurrence before the gated output norm.

    Without records, the recurrence advances the committed state in place: a
    single-token round has no drafts, so its token is accepted whenever it is emitted.
    """

    def __init__(self, *, mixed_qkv, a, b, A_log, dt_bias, state, metadata, output, records: GDNRecords | None,
                 key_heads: int, value_heads: int):
        sequences = metadata.query_start_loc.numel() - 1
        if (state.ndim != 4 or state.shape[1:] != (value_heads, 128, 128) or state.dtype != torch.float32
                or state.stride()[1:] != (128 * 128, 128, 1) or output.shape[-1] != value_heads * 128
                or output.dtype != torch.bfloat16 or not output.is_contiguous()
                or (records is not None and (records.delta.shape[1:] != (value_heads, WINDOW, 128)
                    or records.keys.shape[1:] != (key_heads, WINDOW, 128)
                    or records.keys.shape[0] != records.delta.shape[0] or records.delta.shape[0] < sequences))
                or metadata.state_indices.shape[1] > WINDOW or value_heads % key_heads
                or mixed_qkv.stride(1) != 1 or a.stride(1) != 1 or b.stride(1) != 1):
            raise ValueError("recovered GDN verification requires packed FP32 state slots, contiguous BF16 "
                             "output and per-request records for its head geometry")
        slots = metadata.state_indices[:, 0]
        # In place, the record operands are never stored through; the state stands in for them.
        delta, keys, decay = (state, state, state) if records is None else (records.delta, records.keys, records.decay)
        self.args = (mixed_qkv, a, b, A_log, dt_bias, state, metadata.query_start_loc, slots, metadata.num_seqs,
                     output, delta, keys, decay, mixed_qkv.stride(0), a.stride(0),
                     b.stride(0), state.stride(0), metadata.state_indices.stride(0))
        verify_rows = 8 if sequences * value_heads >= 192 else 4
        self.options = {"KEY_HEADS": key_heads, "VALUE_HEADS": value_heads, "SCALE": 1.0 / math.sqrt(128),
                        "WINDOW": WINDOW, "ROWS": verify_rows, "IN_PLACE": records is None}
        self.grid = (128 // verify_rows, sequences * value_heads, 1)
        self.compiled = warmup_triton(_verify.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)

    def __call__(self):
        launch_triton(_verify, self.compiled, self.args, self.grid, **self.options)


class PreparedGDNCommit:
    """Replay each request's accepted verification records into its committed state, for every layer.

    ``backing`` is the hybrid state pool's byte rows; ``states`` are its per-layer recurrent views.
    """

    def __init__(self, backing: torch.Tensor, states: Mapping[int, torch.Tensor], records: GDNLayerRecords,
                 metadata, counts: torch.Tensor, *, key_heads: int, value_heads: int):
        if set(states) != set(records.layers) or not records.layers:
            raise ValueError("GDN commit requires records for every committed layer")
        floats = backing.view(torch.float32)
        rows = floats.shape[0]
        for layer in records.layers:
            state = states[layer]
            if (state.untyped_storage().data_ptr() != backing.untyped_storage().data_ptr()
                    or state.stride(0) != floats.stride(0) or state.shape[0] != rows):
                raise ValueError("GDN commit requires recurrent views of the state backing rows")
        if counts.shape != (metadata.query_start_loc.numel() - 1,) or counts.numel() > records.delta.shape[1]:
            raise ValueError("GDN commit requires a count and records per request")
        layer_offsets = [states[layer].storage_offset() - floats.storage_offset() for layer in records.layers]
        state_alignment = math.gcd(4, floats.data_ptr() // 4, floats.stride(0), *layer_offsets)
        offsets = torch.tensor(layer_offsets, dtype=torch.int64, device=backing.device)
        self.args = (floats, offsets, records.delta, records.keys, records.decay, metadata.state_indices[:, 0],
                     counts, metadata.num_seqs, floats.stride(0), metadata.state_indices.stride(0))
        self.options = {"ROWS_CAPACITY": records.delta.shape[1], "KEY_HEADS": key_heads, "VALUE_HEADS": value_heads,
                        "WINDOW": WINDOW, "ROWS": ROWS, "STATE_ALIGNMENT": state_alignment}
        self.grid = (128 // ROWS, counts.numel(), len(records.layers) * value_heads)
        self.compiled = warmup_triton(_commit.fn, self.args, self.grid, num_warps=_CONFIG.num_warps,
                                      num_stages=_CONFIG.num_stages, **self.options)

    def __call__(self):
        launch_triton(_commit, self.compiled, self.args, self.grid, **self.options)
