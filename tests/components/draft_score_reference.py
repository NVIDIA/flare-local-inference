"""Unconditional draft scoring oracle for request-policy branch checks."""

import triton
import triton.language as tl


@triton.jit
def draft_scores_reference(Logits, Histogram, Candidates, Slots, Params, GrammarMask, GrammarEnabled,
                           Scores, BlockMax, hist_stride, mask_stride, candidate_stride,
                           CAPACITY: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                           BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    inside = token < V
    score = tl.load(Logits + row * V + token, mask=inside, other=0.).to(tl.float32)
    count = tl.load(Histogram + safe * hist_stride + token, mask=inside, other=0)
    for index in tl.static_range(PREFIX):
        draft = tl.load(Candidates + row * candidate_stride + index)
        count += ((token == draft) & (draft >= 0) & (draft < V)).to(tl.int32)
    temperature = tl.load(Params + safe * 6)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    adjusted = tl.where(score > 0, tl.math.div_rn(score, repetition), score * repetition)
    score = tl.where(count > 0, adjusted, score)
    score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
    score = tl.math.div_rn(score, tl.where(temperature > 0, temperature, 1.))
    word = tl.load(GrammarMask + safe * mask_stride + token // 32, mask=inside, other=0)
    allowed = (((word >> (token % 32)) & 1) != 0) | (tl.load(GrammarEnabled + safe) == 0)
    score = tl.where(allowed, score, -float("inf"))
    tl.store(Scores + row * V + token, score, mask=inside)
    score = tl.where(inside, score, -float("inf"))
    tl.store(BlockMax + row * BLOCKS + block, tl.max(score, 0))
