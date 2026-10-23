"""Allocate a fixed token budget among causal speculative prefixes.

Each submitted row starts with its pending token. Further tokens are admitted
one at a time by descending cumulative survival, then shorter prefix, then
original request slot. Confidence is FP32; sequential products and ordering use
FP64, including the distinction between positive and negative zero.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _allocate(Confidence, Slots, Submitted, SourceValid, Limits, Lengths, Valid,
              ROW_STRIDE: tl.constexpr, COLUMN_STRIDE: tl.constexpr,
              B: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
              CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    inside = row < B
    submitted = tl.load(Submitted + row, inside, other=False)
    source_valid = tl.load(SourceValid + row, inside, other=False)
    slot = tl.load(Slots + row, inside, other=-1)
    limit = tl.load(Limits + row, inside, other=0)
    duplicates = tl.sum((submitted[:, None] & submitted[None, :]
                         & (slot[:, None] == slot[None, :])).to(tl.int32), 1)
    bad = submitted & (~source_valid | (slot < 0) | (slot >= CAPACITY)
                       | (limit < 1) | (limit > D + 1) | (duplicates != 1))
    for position in tl.static_range(D):
        value = tl.load(Confidence + row * ROW_STRIDE + position * COLUMN_STRIDE,
                        submitted & inside, other=0.)
        bad |= submitted & (~(value >= 0.) | ~(value <= 1.))
    count = tl.sum(submitted.to(tl.int32), 0)
    good = (tl.sum(bad.to(tl.int32), 0) == 0) & (count <= P)
    length = submitted.to(tl.int32)
    survival = tl.full((BLOCK,), 1., tl.float64)
    # Only the winner's immediate frontier advances. A global sort of all
    # cumulative products would not preserve the required causal tie order.
    for _ in range(tl.minimum(tl.maximum(P - count, 0), B * D)):
        eligible = submitted & (length < limit) & good
        confidence = tl.load(Confidence + row * ROW_STRIDE + (length - 1) * COLUMN_STRIDE,
                             eligible, other=0.).to(tl.float64)
        probability = survival * confidence
        bits = probability.to(tl.int64, bitcast=True)
        ordered = tl.where(bits < 0, bits ^ 0x7fffffffffffffff, bits)
        maximum = tl.max(tl.where(eligible, ordered, -9223372036854775808), 0)
        tied = eligible & (ordered == maximum)
        shortest = tl.min(tl.where(tied, length, 2147483647), 0)
        tied &= length == shortest
        first_slot = tl.min(tl.where(tied, slot, 9223372036854775807), 0)
        winner = tied & (slot == first_slot)
        survival = tl.where(winner, probability, survival)
        length += winner.to(tl.int32)
    tl.store(Lengths + row, tl.where(good, length, 0), inside)
    tl.store(Valid + row, submitted & good, inside)


class PreparedCausalPrefixAllocation:
    """Graph-safe prefix allocation over explicit caller-owned tensors.

    ``submitted`` is the expected cohort, independent of proposal validation.
    Every submitted row must have ``source_valid`` set, a unique slot in
    ``[0, request_capacity)``, a limit in ``[1, D + 1]``, and finite confidence
    in ``[0, 1]`` at all D positions. Padding is ignored. Any invalid submitted
    row, or more pending tokens than ``token_capacity``, clears both outputs
    for the entire cohort. Successful lengths include the pending token.

    The outputs must have disjoint live storage from the inputs and each other. The component mutates
    only the two outputs and allocates no GPU scratch. TP callers must supply
    identical confidence and cohort inputs on every participating rank.
    """

    def __init__(self, confidence, slots, submitted, source_valid, prefix_limits,
                 output_lengths, valid_output, *, token_capacity, request_capacity):
        if confidence.device.type != "cuda":
            raise ValueError("prefix allocation requires CUDA tensors")
        if confidence.ndim != 2 or confidence.dtype != torch.float32:
            raise ValueError("prefix confidence requires a two-dimensional FP32 tensor")
        batch, depth = confidence.shape
        if not 1 <= batch <= request_capacity <= 64 or not 1 <= depth <= 7:
            raise ValueError("prefix allocation requires 1 <= batch <= request_capacity <= 64 and 1 <= depth <= 7")
        if not isinstance(token_capacity, int) or not 0 <= token_capacity <= 2147483647:
            raise ValueError("prefix token capacity must be a nonnegative Int32 value")
        if confidence.stride(0) <= 0 or confidence.stride(1) <= 0:
            raise ValueError("prefix confidence requires positive strides")
        for tensor, dtype in ((slots, torch.int64), (submitted, torch.bool),
                              (source_valid, torch.bool), (prefix_limits, torch.int32),
                              (output_lengths, torch.int32), (valid_output, torch.bool)):
            if (tensor.shape != (batch,) or tensor.dtype != dtype
                    or tensor.device != confidence.device or not tensor.is_contiguous()):
                raise ValueError("prefix allocation vectors require matching contiguous shapes, dtypes, and devices")
        self.args = (confidence, slots, submitted, source_valid, prefix_limits, output_lengths, valid_output,
                     confidence.stride(0), confidence.stride(1), batch, depth, token_capacity,
                     request_capacity, triton.next_power_of_2(batch))

    def __call__(self):
        torch.library.wrap_triton(_allocate)[(1,)](*self.args, num_warps=4, enable_fp_fusion=False)
