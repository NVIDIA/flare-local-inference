"""Proposal and verification graphs with persistent proposal ownership.

Proposal generation reserves its random draws and leaves target state unchanged.
Verification gathers request-owned prefixes independently of the proposal batch.
"""

import math
import heapq

import torch
import triton
import triton.language as tl

from .linear import UnquantizedProjection
from .sampling import PreparedCompletion, _write_completion
from .state import gather_rows, scatter_rows


@triton.jit
def _reject_stale(Valid, Slots, Epochs, Live, Generation, Consumed, Generated, Records, Frame, Sequence,
                  CS: tl.constexpr, SLOTS: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    if (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS) & ~tl.load(Valid + row):
        epoch = tl.load(Epochs + row)
        current = epoch == tl.load(Generation + slot)
        consumed = tl.load(Consumed + slot * CS, current, other=0)
        generated = tl.load(Generated + slot, current, other=0)
        # A diagnostic rejection reports failure without mutating the request occupying the slot.
        token = tl.full((), -1, tl.int64)
        _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                          token, 3, 256, True, 24, SLOTS)


def causal_prefix(confidences, threshold):
    """Admit position i using only confidence through i and stop at the first failure.

    Confidence i must depend only on history and proposals preceding i. Looking
    ahead to a later confidence when deciding whether to admit i biases sampling.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("confidence threshold must be finite and in (0, 1]")
    product, count = 1., 0
    for confidence in confidences:
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        product *= confidence
        if product < threshold:
            break
        count += 1
    return count


def lagged_budget(confidences, costs):
    """Choose a width from a completed previous round's confidence snapshot.

    Costs cover widths one through len(confidences)+1 and include the entire
    proposal, verification and feedback step. Current sampled drafts must not
    influence this decision.
    """
    if len(costs) != len(confidences) + 1 or any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("positive complete-step costs must cover every verification width")
    expected, survival = 1., 1.
    utilities = [expected / costs[0]]
    for confidence, cost in zip(confidences, costs[1:], strict=True):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        survival *= confidence
        expected += survival
        utilities.append(expected / cost)
    return max(range(len(utilities)), key=utilities.__getitem__) + 1


def allocate_draft_budget(confidences, budget):
    """Allocate a fixed draft count across causal request prefixes.

    The caller chooses the budget independently of these sampled proposals.
    A request's next confidence enters the frontier only after its preceding
    position has been admitted. Ties prefer the earlier position, then request.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("draft budget must be a nonnegative integer")
    rows = [tuple(row) for row in confidences]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for row in rows for p in row):
        raise ValueError("confidence must be finite and in [0, 1]")
    frontier = [(-row[0], 0, slot) for slot, row in enumerate(rows) if row]
    heapq.heapify(frontier)
    counts = [0] * len(rows)
    for _ in range(min(budget, sum(map(len, rows)))):
        probability, position, slot = heapq.heappop(frontier)
        counts[slot] += 1
        if position + 1 < len(rows[slot]):
            heapq.heappush(frontier, (probability * rows[slot][position + 1], position + 1, slot))
    return tuple(counts)


class PreparedDraftConfidence:
    """Preserve checkpoint storage precision while accumulating confidence in FP32."""

    def __init__(self, source, weight, output, *, name, bias=None, buffers=None):
        o, b, w = source.owner, source.batch, source.width
        hidden = source.draft.head_hidden
        markov = source.markov.embedding
        if weight.shape != (1, hidden.shape[1] + markov.shape[1]):
            raise ValueError("confidence weight must cover hidden and preceding-token Markov features")
        self.hidden = hidden.view(b, o.block_size, -1)[:, o.logit_offset:o.logit_offset + w - 1]
        buffers = getattr(source, "buffers", None) if buffers is None else buffers
        def allocate(shape, dtype):
            return (torch.empty(shape, dtype=dtype, device=markov.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.embeddings = allocate((b, w - 1, markov.shape[1]), markov.dtype)
        self.features = allocate((b, w - 1, weight.shape[1]), weight.dtype)
        self.logits = allocate((b, w - 1), torch.float32)
        self.output = output
        self.projection = UnquantizedProjection(self.features.view(b * (w - 1), -1), weight,
            self.logits.view(-1, 1), session=o.session, name=name, bias=bias)

    def record_embedding(self, position, embedding):
        self.embeddings[:, position].copy_(embedding)

    def __call__(self):
        self.features[:, :, :self.hidden.shape[-1]].copy_(self.hidden)
        self.features[:, :, self.hidden.shape[-1]:].copy_(self.embeddings)
        self.projection(self.features.view(-1, self.features.shape[-1]))
        self.output.copy_(self.logits.sigmoid())


class PreparedProposalStorage:
    """One retained block per physical request slot across all graph shapes."""

    def __init__(self, owner, drafts):
        self.owner, self.depth = owner, drafts
        c, device = owner.requests.capacity, owner.requests.generation.device
        self.ready = torch.zeros(c, dtype=torch.bool, device=device)
        self.generations = torch.zeros(c, dtype=torch.int64, device=device)
        self.state_rows = torch.zeros_like(self.generations)
        self.sequence = torch.zeros_like(self.generations)
        self.before = torch.zeros_like(self.generations)
        self.depths = torch.zeros_like(self.generations)
        self.base_offset = torch.zeros_like(self.generations)
        self.candidates = torch.zeros((c, drafts + 1), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((c, drafts), dtype=torch.int64, device=device)
        self.probabilities = torch.zeros((c, drafts, owner.requests.vocab_size), dtype=torch.float32, device=device)
        self.valid = torch.zeros((c, drafts), dtype=torch.bool, device=device)

    def buffers(self):
        return {"proposal_" + name: getattr(self, name) for name in (
            "ready", "generations", "state_rows", "sequence", "before", "depths",
            "base_offset", "candidates", "drafts", "probabilities", "valid")}


class PreparedDraftProposal:
    """Publish independently owned blocks and consume their RNG reservation once."""

    def __init__(self, source, confidence_output, storage):
        if source.width <= 1 or source.markov is None or source.width > storage.depth + 1:
            raise ValueError("split proposals require compatible retained storage and a confidence head")
        self.source, self.storage = source, storage
        o, b = source.owner, source.batch
        self.confidence = PreparedDraftConfidence(source, o.weights.draft.confidence_head,
            confidence_output[:b, :source.width - 1], name=f"split.proposal{b}x{source.width}.confidence",
            bias=getattr(o.weights.draft, "confidence_bias", None))
        self.allowed = torch.zeros_like(source.active)
        self.depth = torch.full_like(source.before, source.width - 1)
        self.completion = PreparedCompletion(o.requests, o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b], o.inputs.live_rows, o.completion, o.frame, o.sequence)
        self.completion.prepare()

    def __call__(self):
        s, p, o = self.source, self.storage, self.source.owner
        slots = o.inputs.row_slot_ids[:s.batch]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        self.allowed.copy_(~p.ready[safe] | (p.generations[safe] != o.requests.generation[safe]))
        s.propose(True, self.confidence, self.allowed)
        self.confidence()
        for source, destination in (
            (o.requests.generation[safe], p.generations),
            (o.inputs.request_state_rows[safe, 0], p.state_rows),
            (o.sequence.expand(s.batch), p.sequence),
            (self.depth, p.depths), (s.before, p.before), (s.base_offset, p.base_offset),
            (s.candidates, p.candidates), (s.drafts, p.drafts),
            (s.probabilities, p.probabilities), (s.valid, p.valid), (s.active, p.ready),
        ):
            scatter_rows(source, slots, destination, s.active)
        self.completion()
        torch.library.wrap_triton(_reject_stale)[(s.batch,)](
            self.allowed, slots, o.inputs.row_generations, o.inputs.live_rows, o.requests.generation,
            o.requests.consumed_tokens, o.requests.generated_tokens, o.completion.records, o.frame, o.sequence,
            o.requests.consumed_tokens.stride(0), o.requests.capacity)


class PreparedDraftVerification:
    """Gather and consume request-owned proposal prefixes exactly once."""

    def __init__(self, storage, target):
        if target.width > storage.depth + 1:
            raise ValueError("verification shape must fit retained proposal storage")
        self.storage, self.target = storage, target
        self.original_slots = torch.zeros_like(target.owner.inputs.row_slot_ids[:target.batch])
        self.rows = torch.arange(target.batch, device=target.active.device)
        self.admitted = torch.zeros_like(target.active)
        self.cleared = torch.zeros_like(target.active)

    def __call__(self, same_graph=False):
        p, t = self.storage, self.target
        o, w = t.owner, t.width
        slots = o.inputs.row_slot_ids[:t.batch]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        valid = p.ready[safe] & (p.sequence[safe] <= o.sequence if same_graph else p.sequence[safe] < o.sequence)
        valid &= (self.rows < o.inputs.live_rows) & (o.requests.status[safe] == 1)
        valid &= (slots >= 0) & (slots < o.requests.capacity)
        valid &= (o.inputs.row_generations[:t.batch] == p.generations[safe])
        valid &= (o.requests.generation[safe] == p.generations[safe])
        valid &= (o.requests.consumed_tokens[safe] == p.before[safe])
        valid &= (o.requests.pending_token[safe] == p.candidates[safe, 0])
        valid &= (o.inputs.request_state_rows[safe, 0] == p.state_rows[safe])
        lengths = o.inputs.query_lengths[:t.batch]
        valid &= (lengths >= 1) & (lengths <= w) & (lengths <= p.depths[safe] + 1)
        # A graph can admit long individual prefixes while bounding their sum.
        valid &= torch.where(valid, lengths, 0).sum() <= t.target_token_capacity
        self.admitted.copy_(valid)
        self.original_slots.copy_(slots)
        for source, destination in (
            (p.before, t.before), (p.base_offset, t.base_offset),
            (p.candidates, t.candidates), (p.drafts, t.drafts),
            (p.probabilities, t.probabilities), (p.valid, t.valid),
        ):
            if destination.numel():
                gather_rows(source, safe, destination)
        t.active.copy_(valid)
        # Invalid ownership must not let commit mutate a replacement request.
        slots.copy_(torch.where(valid, self.original_slots, -1))
        t.verify(valid)
        slots.copy_(self.original_slots)
        scatter_rows(self.cleared, slots, p.ready, self.admitted)
        r = o.requests
        torch.library.wrap_triton(_reject_stale)[(t.batch,)](
            self.admitted, slots, o.inputs.row_generations, o.inputs.live_rows, r.generation,
            r.consumed_tokens, r.generated_tokens, o.completion.records, o.frame, o.sequence,
            r.consumed_tokens.stride(0), r.capacity)


class PreparedCombinedDraftRound:
    """One-graph reference for the retained full-block proposal RNG contract."""

    def __init__(self, proposal, verification):
        self.proposal, self.verification = proposal, verification

    def __call__(self):
        self.proposal()
        self.verification(same_graph=True)
