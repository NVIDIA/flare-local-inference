"""One captured proposal, bounded causal allocation, and target commit.

The host supplies immutable per-request prefix ceilings and reserves their cache
extents before submission. Only the admitted lengths are device-owned; the
cohort, physical target capacity, page tables, and request epochs cannot change
inside the round.
"""

import torch

from .causal_prefix import PreparedCausalPrefixAllocation
from .collectives import PreparedAllReduce, PreparedBroadcast
from .state import scatter_rows


class PreparedPackedSpeculativeRound:
    """Compose retained proposals and verification without intermediate feedback."""

    def __init__(self, proposal, verification, prefix_limits, *, name, buffers=None):
        s, t, p = proposal.source, verification.target, proposal.storage
        if (verification.storage is not p or s.batch != t.batch
                or s.owner.inputs is not t.owner.inputs
                or s.owner.requests is not t.owner.requests
                or not 1 < s.width <= t.width):
            raise ValueError("packed rounds require matching proposal and target ownership")
        o, b = s.owner, s.batch
        if (prefix_limits.shape != (o.requests.capacity,) or prefix_limits.dtype != torch.int32
                or prefix_limits.device != s.active.device or not prefix_limits.is_contiguous()):
            raise ValueError("prefix limits must be contiguous int32 values for every request lane")
        query_lengths = o.inputs.query_lengths[:b]
        prefix_begin, query_begin = prefix_limits.data_ptr(), query_lengths.data_ptr()
        if (prefix_begin < query_begin + query_lengths.numel() * query_lengths.element_size()
                and query_begin < prefix_begin + prefix_limits.numel() * prefix_limits.element_size()):
            raise ValueError("host prefix limits must not overlap device-owned query lengths")
        self.proposal, self.verification, self.prefix_limits = proposal, verification, prefix_limits
        self.buffers = buffers
        if buffers is not None:
            # These values span proposal and target execution. Start beyond both
            # complete sampling workspaces, rather than aliasing either cursor.
            buffers.join(*(cursor for cursor in (s.buffers, t.buffers) if cursor is not None))
        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=s.active.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.rows = s.rows
        self.submitted = zeros((b,), torch.bool)
        self.source_valid = zeros((b,), torch.bool)
        self.admitted = zeros((b,), torch.bool)
        self.slots = zeros((b,), torch.int64)
        self.confidence = zeros((b, s.width - 1), torch.float32)
        self.validation_input = zeros(tuple(s.validator.validation_input.shape), s.validator.validation_input.dtype)
        self.validation_sum = zeros(tuple(self.validation_input.shape), self.validation_input.dtype)
        self.world = o.weights.shard.world_size
        self.validation = PreparedAllReduce(self.validation_input, self.validation_sum,
            runtime=o.collectives["runtime"], coordinator=o.collectives["coordinator"],
            session=o.session, name=name + ".allocation_validation")
        self.allocator = PreparedCausalPrefixAllocation(
            self.confidence, self.slots, self.submitted, self.source_valid, prefix_limits[:b],
            o.inputs.query_lengths[:b], self.admitted, token_capacity=t.target_token_capacity,
            request_capacity=o.requests.capacity)
        runtime = getattr(o, "result_runtime", None)
        if runtime is None:
            runtime = o.collectives["vocabulary_runtime"]
        self.broadcast = PreparedBroadcast((self.confidence,), runtime=runtime,
            session=o.session, coordinator=o.collectives["coordinator"], name=name + ".confidence", buffers=buffers)

    def __call__(self):
        proposal, verification = self.proposal, self.verification
        s, p, o = proposal.source, proposal.storage, proposal.source.owner
        b = s.batch
        slots = o.inputs.row_slot_ids[:b]
        safe = slots.long().clamp(0, o.requests.capacity - 1)
        self.slots.copy_(slots)
        self.submitted.copy_(self.rows < o.inputs.live_rows)
        # The existing proposal validator must see the host-reserved ceiling.
        # Allocation replaces this scratch only after proposal generation.
        o.inputs.query_lengths[:b].copy_(self.prefix_limits[:b])
        proposal.allowed.copy_(~p.ready[safe] | (p.generations[safe] != o.requests.generation[safe]))
        s.propose(False, proposal.confidence, proposal.allowed)
        # Sampling has consumed this captured block even if confidence
        # validation subsequently fails. Never reuse its draft draw reservation.
        s.commit_draft_rng()
        proposal.confidence()
        self.confidence.copy_(proposal.confidence.output)
        self.source_valid.copy_(s.active & proposal.allowed
            & torch.isfinite(self.confidence).all(1)
            & (self.confidence >= 0).all(1) & (self.confidence <= 1).all(1))
        self.broadcast()
        self.confidence.masked_fill_(~self.submitted[:, None], 0.)
        proposal.confidence.output.copy_(self.confidence)
        self.allocator()
        # Rank-zero confidence determines every prefix, but a validation failure
        # on any rank suppresses commit everywhere. Collectives execute even for
        # rejected cohorts so no peer can leave another inside a target collective.
        self.validation_input.zero_()
        self.validation_input[0, :b].copy_(self.admitted)
        agreement = self.validation(self.validation_input)
        cohort_valid = (~self.submitted | (agreement[0, :b] == self.world)).all()
        self.admitted.logical_and_(cohort_valid)
        o.inputs.query_lengths[:b].copy_(torch.where(self.admitted, o.inputs.query_lengths[:b], 0))
        # Invalid cohorts publish no retained proposals. Target RNG and state
        # advancement remain inside the ordinary admitted verification/commit.
        s.active.logical_and_(self.admitted)
        for source, destination in (
            (o.requests.generation[safe], p.generations),
            (o.inputs.request_state_rows[safe, 0], p.state_rows),
            (o.sequence.expand(b), p.sequence),
            (proposal.depth, p.depths), (s.before, p.before), (s.base_offset, p.base_offset),
            (s.candidates, p.candidates), (s.drafts, p.drafts),
            (s.probabilities, p.probabilities), (s.valid, p.valid), (s.active, p.ready),
        ):
            scatter_rows(source, slots, destination, s.active)
        # This performs the ordinary ownership checks and writes the sole
        # terminal completion, including admitted/offered counts and errors.
        verification(same_graph=True)
