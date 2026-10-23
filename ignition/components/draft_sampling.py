"""Bounded top-k draft proposals: filtering and sampling over at most DRAFT_CANDIDATES tokens.

Proposals follow the request policy after penalties, grammar and temperature:
top-k, softmax, top-p and min-p, each retaining boundary ties among the
candidates, then renormalization. A request whose top-k exceeds
DRAFT_CANDIDATES, including disabled top-k, proposes from the
DRAFT_CANDIDATES highest scores. Chain verification receives exactly the
distribution each draft was drawn from, so verified output is unchanged;
only acceptance depends on how closely proposals follow the target policy.

The first kernel writes zeros over the dense proposal row and keeps each
block's highest scores. The second selects the row's candidates, applies the
policy, draws one Philox uniform, and scatters the surviving probabilities.
"""

import torch
import triton
import triton.language as tl

from ignition.components.speculation import PreparedSpeculativeDistributions

DRAFT_CANDIDATES = 64
BLOCK = 4096


@triton.jit
def _orderable(scores):
    # Signed integers ordered like the scores; NaN orders above infinity and is rejected later.
    bits = scores.to(tl.int32, bitcast=True)
    # IEEE signed zeros are equal scores and share the token-index tie order.
    bits = tl.where(scores == 0., 0, bits)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _kth_largest(ordered, k):
    # Largest t with at least k values >= t, by bisection over the 32-bit order.
    low = tl.full((), -2147483648, tl.int64)
    high = tl.full((), 2147483647, tl.int64)
    wide = ordered.to(tl.int64)
    for _ in range(32):
        middle = (low + high + 1) >> 1
        enough = tl.sum((wide >= middle).to(tl.int32), 0) >= k
        low = tl.where(enough, middle, low)
        high = tl.where(enough, high, middle - 1)
    return low


@triton.jit
def _candidate_positions(ordered, pivot, present, K: tl.constexpr):
    # Cutoff ties may exceed the bounded buffer. Every strictly higher score
    # must survive before ties consume its remaining slots.
    above = present & (ordered.to(tl.int64) > pivot)
    tied = present & (ordered.to(tl.int64) == pivot)
    higher_count = tl.sum(above.to(tl.int32), 0)
    position = tl.where(above, tl.cumsum(above.to(tl.int32), 0) - 1,
                        higher_count + tl.cumsum(tied.to(tl.int32), 0) - 1)
    take = (above | tied) & (position < K)
    count = tl.minimum(higher_count + tl.sum(tied.to(tl.int32), 0), K)
    return position, take, count


@triton.jit
def _block_candidates(Scores, Probabilities, TopK, Candidates, V: tl.constexpr, BLOCKS: tl.constexpr,
                      BLOCK: tl.constexpr, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    inside = tokens < V
    scores = tl.load(Scores + row * V + tokens, inside, other=-float("inf"))
    tl.store(Probabilities + row * V + tokens, tl.zeros((BLOCK,), tl.float32), inside)
    k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    ordered = tl.where(inside, _orderable(scores), -2147483648)
    position, take, count = _candidate_positions(ordered, _kth_largest(ordered, k), inside, K)
    keys = (ordered.to(tl.int64) << 32) | (2147483647 - tokens).to(tl.int64)
    base = Candidates + (row * BLOCKS + block) * K
    slots = tl.arange(0, K)
    tl.store(base + slots, tl.full((K,), -9223372036854775808, tl.int64), slots >= count)
    tl.store(base + position, keys, take)


@triton.jit
def _select_and_sample(Candidates, Selected, Probabilities, Valid, TopK, TopP, Params, Slots, Seed,
                       Offset, Sampled, V: tl.constexpr, BLOCKS: tl.constexpr, POOL: tl.constexpr, K: tl.constexpr,
                       SLOTS: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, POOL)
    pool = tl.load(Candidates + row * BLOCKS * K + index, index < BLOCKS * K, other=-9223372036854775808)
    top_k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    present = pool != -9223372036854775808
    ordered = (pool >> 32).to(tl.int32)
    position, take, _ = _candidate_positions(ordered, _kth_largest(ordered, top_k), present, K)
    rank = tl.arange(0, K)
    tl.store(Selected + row * K + rank, tl.full((K,), -9223372036854775808, tl.int64))
    tl.debug_barrier()
    tl.store(Selected + row * K + position, pool, take)
    tl.debug_barrier()
    keys = tl.sort(tl.load(Selected + row * K + rank), descending=True)
    ordered = (keys >> 32).to(tl.int32)
    scores = tl.where(ordered < 0, ordered ^ 2147483647, ordered).to(tl.float32, bitcast=True)
    tokens = 2147483647 - (keys & 2147483647).to(tl.int32)
    present = keys != -9223372036854775808
    finite = present & (scores == scores) & (scores != float("inf")) & (scores != -float("inf"))  # noqa: PLR0124

    slot = tl.minimum(tl.maximum(tl.load(Slots + row), 0), SLOTS - 1)
    min_p = tl.load(Params + slot * 6 + 2)
    top_p = tl.load(TopP + row)

    # Top-k keeps every candidate tied with the k-th score.
    pivot = tl.max(tl.where(rank == top_k - 1, scores, -float("inf")), 0)
    kept = finite & (scores >= pivot)
    maximum = tl.max(tl.where(kept, scores, -float("inf")), 0)
    weights = tl.where(kept, tl.exp(scores - maximum), 0.)
    probabilities = weights / tl.sum(weights, 0)
    # Top-p keeps the shortest descending prefix reaching top_p and ties with its last member.
    before = tl.cumsum(probabilities, 0) - probabilities
    cutoff = tl.min(tl.where(kept & (before < top_p), probabilities, float("inf")), 0)
    kept &= probabilities >= cutoff
    probabilities = tl.where(kept, probabilities, 0.)
    probabilities = probabilities / tl.sum(probabilities, 0)
    # Min-p against the largest remaining probability, then renormalize.
    kept &= (probabilities > 0.) & (probabilities >= tl.max(probabilities, 0) * min_p)
    probabilities = tl.where(kept, probabilities, 0.)
    mass = tl.sum(probabilities, 0)
    valid = (tl.load(Valid + row) != 0) & (mass > 0.) & (mass == mass) & (mass != float("inf"))  # noqa: PLR0124
    probabilities = probabilities / tl.where(mass > 0., mass, 1.)

    seed = tl.load(Seed + row)
    offset = (tl.load(Offset + row) & 4294967295).to(tl.uint32)
    uniform = tl.rand(seed, offset)
    cumulative = tl.cumsum(probabilities, 0)
    chosen = tl.sum(((cumulative <= uniform) & kept).to(tl.int32), 0)
    last = tl.max(tl.where(kept, rank, 0), 0)
    chosen = tl.minimum(chosen, last)
    sampled = tl.sum(tl.where(rank == chosen, tokens, 0), 0)
    sampled = tl.where(valid, sampled, 0)

    tl.store(Probabilities + row * V + tokens, probabilities, kept & valid)
    tl.store(Probabilities + row * V + sampled, 1., ~valid)
    tl.store(Valid + row, valid)
    tl.store(Sampled + row, sampled.to(tl.int64))


@triton.jit
def _draft_scores(Logits, Histogram, Candidates, Slots, Params, GrammarMask, GrammarEnabled, Scores, BlockMax,
                  hist_stride, mask_stride, candidate_stride, CAPACITY: tl.constexpr, V: tl.constexpr,
                  PREFIX: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    # One vocabulary block of one row: the committed histogram plus the row's earlier drafts give the penalty
    # counts; repetition, presence and frequency penalties, temperature and the grammar mask follow the
    # unfused order of operations. Each block records its maximum.
    row, block = tl.program_id(0), tl.program_id(1)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    inside = token < V
    score = tl.load(Logits + row * V + token, mask=inside, other=0.).to(tl.float32)
    temperature = tl.load(Params + safe * 6)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    # Nonnegative counts and positive-zero penalties leave signed logits unchanged.
    # Prefix validity remains a separate check in _draft_scores_finish.
    neutral = ((repetition == 1.) & (presence.to(tl.int32, bitcast=True) == 0)
        & (frequency.to(tl.int32, bitcast=True) == 0))
    if not neutral:
        count = tl.load(Histogram + safe * hist_stride + token, mask=inside, other=0)
        for index in tl.static_range(PREFIX):
            draft = tl.load(Candidates + row * candidate_stride + index)
            count += ((token == draft) & (draft >= 0) & (draft < V)).to(tl.int32)
        # IEEE division and no FMA contraction preserve the unfused arithmetic.
        adjusted = tl.where(score > 0, tl.math.div_rn(score, repetition), score * repetition)
        score = tl.where(count > 0, adjusted, score)
        score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
    score = tl.math.div_rn(score, tl.where(temperature > 0, temperature, 1.))
    if tl.load(GrammarEnabled + safe) != 0:
        word = tl.load(GrammarMask + safe * mask_stride + token // 32, mask=inside, other=0)
        score = tl.where(((word >> (token % 32)) & 1) != 0, score, -float("inf"))
    tl.store(Scores + row * V + token, score, mask=inside)
    score = tl.where(inside, score, -float("inf"))
    maximum = tl.max(score, 0)
    tl.store(BlockMax + row * BLOCKS + block, maximum)


@triton.jit
def _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY: tl.constexpr, BATCH: tl.constexpr):
    slot = tl.load(Slots + row).to(tl.int64)
    safe = tl.minimum(tl.maximum(slot, 0), CAPACITY - 1)
    selected = (row < live_rows) & (live_rows >= 0) & (live_rows <= BATCH) & (slot >= 0) & (slot < CAPACITY)
    return selected & (tl.load(Generation + safe) == tl.load(Generations + row)), slot, safe


@triton.jit
def _draft_scores_finish(BlockMax, Candidates, Slots, Generations, LiveRows, Generation, Status, TopK,
                         Params, Maximum, Valid, PackedTopK, PackedTopP, candidate_stride,
                         CAPACITY: tl.constexpr, BATCH: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                         BLOCKS: tl.constexpr, BLOCK_COUNT: tl.constexpr, RUNNING: tl.constexpr):
    # The row maximum, then the row's liveness (as speculation's _live), validity and filters.
    row = tl.program_id(0)
    block = tl.arange(0, BLOCK_COUNT)
    present = block < BLOCKS
    maxima = tl.load(BlockMax + row * BLOCKS + block, mask=present, other=-float("inf"))
    maximum = tl.max(maxima, 0)
    live_rows = tl.load(LiveRows)
    selected, slot, safe = _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY, BATCH)
    sharing = 0
    for other in tl.static_range(BATCH):
        other_selected, other_slot, _ = _selected_row(Slots, Generations, Generation, live_rows, other, CAPACITY, BATCH)
        sharing += (other_selected & (other_slot == slot)).to(tl.int32)
    live = selected & (sharing <= 1) & (tl.load(Status + safe) == RUNNING)
    prefix_valid = True
    for index in tl.static_range(PREFIX):
        draft = tl.load(Candidates + row * candidate_stride + index)
        prefix_valid &= (draft >= 0) & (draft < V)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    tl.store(Maximum + row, maximum)
    tl.store(Valid + row, live & prefix_valid & finite)
    top_k = tl.load(TopK + safe)
    tl.store(PackedTopK + row, tl.where(top_k > 0, top_k, V))
    tl.store(PackedTopP + row, tl.load(Params + safe * 6 + 1))


_SCORE_BLOCK = 4096


class PreparedDraftProposal:
    """Two captured launches over explicit request-policy inputs for one draft position."""

    def __init__(self, scores, probabilities, valid, top_k, top_p, params, slots, seed, offset, sampled,
                 *, capacity, candidates=None, selected=None):
        rows, vocab = scores.shape[0], scores.shape[-1]
        self.blocks = triton.cdiv(vocab, BLOCK)
        self.candidates = (torch.empty((rows, self.blocks, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
                           if candidates is None else candidates)
        self.selected = (torch.empty((rows, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
                         if selected is None else selected)
        for tensor, shape in ((self.candidates, (rows, self.blocks, DRAFT_CANDIDATES)),
                              (self.selected, (rows, DRAFT_CANDIDATES))):
            if (tensor.shape != shape or tensor.dtype != torch.int64
                    or tensor.device != scores.device or not tensor.is_contiguous()):
                raise ValueError("draft proposal scratch requires contiguous Int64 tensors on the score device")
        pool = triton.next_power_of_2(self.blocks * DRAFT_CANDIDATES)
        self._block_args = (scores, probabilities, top_k, self.candidates, vocab, self.blocks, BLOCK, DRAFT_CANDIDATES)
        self._select_args = (self.candidates, self.selected, probabilities, valid, top_k, top_p,
                             params, slots,
                             seed, offset, sampled, vocab, self.blocks, pool, DRAFT_CANDIDATES, capacity)
        self._grids = ((rows, self.blocks, 1), (rows, 1, 1))
        self._kernels = None

    def prepare(self):
        from ignition.preparation.triton import prepare_triton

        self._kernels = (
            prepare_triton(_block_candidates, self._block_args, self._grids[0], num_warps=8),
            prepare_triton(_select_and_sample, self._select_args, self._grids[1], num_warps=4),
        )
        self()

    def __call__(self):
        if self._kernels is None:
            raise RuntimeError("draft proposal prepare() must complete before capture")
        for kernel in self._kernels:
            kernel()



class PreparedDraftDistribution(PreparedSpeculativeDistributions):
    """A one-position conditional proposal whose filtering and draw use PreparedDraftProposal."""

    @staticmethod
    def broadcast_layout(batch):
        return dict(support_ids=((batch, DRAFT_CANDIDATES), torch.int64),
                    support_probabilities=((batch, DRAFT_CANDIDATES), torch.float32),
                    sampled=((batch,), torch.int64), valid=((batch, 1), torch.bool))

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return {**PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab),
            "block_max": ((batch, triton.cdiv(vocab, _SCORE_BLOCK)), torch.float32),
            "proposal_candidates": ((batch, triton.cdiv(vocab, BLOCK), DRAFT_CANDIDATES), torch.int64),
            "proposal_selected": ((batch, DRAFT_CANDIDATES), torch.int64)}

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids, *,
                 seed, offset, sampled, prefix_offset=0, workspace=None):
        super().__init__(logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                         prefix_offset=prefix_offset, workspace=workspace)
        if logits.shape[1] != 1:
            raise ValueError("draft proposals cover one position")
        # Fused scoring: per-block maxima of each row before the row reduction.
        self._score_kernels = None
        # Contiguous [B,1,...] storage is addressed by row; kernels receive the base tensors.
        self.proposal = PreparedDraftProposal(
            self.scores, self.probabilities, self.valid, self.packed_top_k, self.packed_top_p,
            requests.parameters, slots, seed, offset, sampled, capacity=requests.capacity,
            candidates=self.proposal_candidates, selected=self.proposal_selected)
        self.sampled = sampled
        self.broadcast = None

    def prepare_broadcast(self, *, runtime, session, coordinator, name, buffers=None):
        if runtime is None:
            return
        from .collectives import PreparedBroadcast

        def allocate(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=self.logits.device) if buffers is None
                    else buffers.zeros(shape, dtype))

        layout = self.broadcast_layout(self.logits.shape[0])
        self.support_ids = allocate(*layout["support_ids"])
        self.support_probabilities = allocate(*layout["support_probabilities"])
        self.broadcast = PreparedBroadcast(
            (self.support_ids, self.support_probabilities, self.sampled, self.valid),
            runtime=runtime, session=session, coordinator=coordinator, name=name, buffers=buffers)

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("draft proposals require CUDA")
        from ignition.components.sampling import RUNNING
        from ignition.preparation.triton import prepare_triton

        state = self.state
        batch, _, vocab = self.logits.shape
        blocks = self.block_max.shape[1]
        self._score_kernels = (
            prepare_triton(_draft_scores, (self.logits, state.histogram, self.candidates, self.slots,
                state.parameters, state.grammar_mask, state.grammar_enabled, self.scores, self.block_max,
                state.histogram.stride(0), state.grammar_mask.stride(0), self.candidates.stride(0)),
                (batch, blocks, 1), CAPACITY=state.capacity, V=vocab, PREFIX=self.prefix_offset, BLOCK=_SCORE_BLOCK,
                BLOCKS=blocks, num_warps=8, enable_fp_fusion=False),
            prepare_triton(_draft_scores_finish, (self.block_max, self.candidates, self.slots,
                self.generations, self.live_rows, state.generation, state.status, state.top_k, state.parameters,
                self.maximum, self.valid, self.packed_top_k,
                self.packed_top_p, self.candidates.stride(0)), (batch, 1, 1), CAPACITY=state.capacity, BATCH=batch,
                V=vocab, PREFIX=self.prefix_offset, BLOCKS=blocks, BLOCK_COUNT=triton.next_power_of_2(blocks),
                RUNNING=RUNNING, num_warps=1))
        self._prepared = True
        self.prepare_scores()
        self.proposal.prepare()

    def prepare_scores(self):
        """One-position scores in two kernels, without materialized penalty counts or a separate maximum."""
        for kernel in self._score_kernels:
            kernel()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("draft proposal requires prepare() before capture")
        self.prepare_scores()
        self.proposal()
        if self.broadcast is not None:
            keys = self.proposal_selected
            present = keys != -9223372036854775808
            ids = 2147483647 - (keys & 2147483647)
            self.support_ids.copy_(torch.where(present, ids, 0))
            probabilities = self.probabilities[:, 0].gather(1, self.support_ids)
            # Padding repeats token zero with its exact probability, so duplicate stores agree.
            # Invalid proposals use the sampler's point mass at token zero.
            self.support_probabilities.copy_(torch.where(self.valid, probabilities, 1.))
            self.support_ids.masked_fill_(~self.valid, 0)
            self.broadcast()
            self.probabilities.zero_()
            self.probabilities[:, 0].scatter_(1, self.support_ids, self.support_probabilities)
        return self.probabilities
