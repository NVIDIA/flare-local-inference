"""Parallel draft proposals, stochastic target verification and accepted-state commits."""

import torch

from .candidate_selector import PreparedCandidateSelector
from .collectives import PreparedAllReduce
from .draft_sampling import PreparedDraftDistribution, PreparedDraftBlockBroadcast
from .paged_batch import Metadata, PreparedBatch
from .speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from .state import save_hidden


def prepare_sampling_scratch(batch, width, vocab, device, *, buffers=None, proposal=True, verification=True):
    """Share sequential sampling stages and retain target results through verification.

    Proposal probabilities must be copied to retained storage before the next
    proposal position. Target probabilities and validity stay disjoint from the
    verification workspace; filtering intermediates are dead before verification.
    """
    from ignition.preparation.scratch import bind_scratch

    layouts = {}
    if proposal and width > 1:
        layouts["proposal"] = PreparedDraftDistribution.scratch_layout(batch, 1, vocab)
    if verification:
        layouts["target"] = PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab)
        layouts["verify"] = PreparedStochasticVerification.scratch_layout(batch, width, vocab)
    if buffers is None:
        return {name: bind_scratch(layout, device=device) for name, layout in layouts.items()}
    proposal_buffers = buffers.fork()
    retained = (bind_scratch({key: layouts["target"].pop(key) for key in ("probabilities", "valid")},
                            device=device, workspace=buffers) if verification else {})
    cursors = {name: proposal_buffers if name == "proposal" else buffers.fork() for name in layouts}
    result = {name: bind_scratch(layout, device=device, workspace=cursors[name])
              for name, layout in reversed(tuple(layouts.items()))}
    if verification:
        result["target"].update(retained)
    buffers.join(*cursors.values())
    return result


def prepare_sampling_broadcasts(proposals, verification, *, collectives, session, name, buffers=None,
                                result_runtime=None):
    """Synchronize proposals before dependent work and acceptance before state commits."""
    cursors = []
    operations = [(f"proposal{index}", proposal) for index, proposal in enumerate(proposals)]
    if verification is not None:
        operations.append(("acceptance", verification))
    runtime = collectives["vocabulary_runtime"] if result_runtime is None else result_runtime
    for suffix, operation in operations:
        cursor = None if buffers is None else buffers.fork()
        operation.prepare_broadcast(runtime=runtime, session=session,
            coordinator=collectives["coordinator"], name=f"{name}.{suffix}", buffers=cursor)
        if cursor is not None:
            cursors.append(cursor)
    if buffers is not None:
        buffers.join(*cursors)


class PreparedParallelDraftRound:
    """Prepare verification and optionally its draft generator.

    A target-only forward receives proposals from its enclosing captured round;
    it allocates no draft decoder, Markov projection, or proposal samplers. An owner
    whose ``draft_argmax`` names a vocabulary-parallel argmax runtime proposes greedy
    drafts from the draft head's and Markov bias's vocabulary shards without gathering
    them; each proposal distribution is then one-hot, so verification stays exact.
    """

    @staticmethod
    def storage_layout(batch, width, vocab):
        return dict(probabilities=((batch, width - 1, vocab), torch.float32),
                    logits=((batch, 1, vocab), torch.float32))

    def __init__(self, owner, batch, width, name, *, generate_proposals=True, buffers=None, proposal_only=False,
                 target_token_capacity=None):
        self.owner, self.batch, self.width = owner, batch, width
        self.vocabulary_parallel = getattr(owner, "vocabulary_parallel_sampling", False)
        self.name, self.buffers = name, buffers
        tokens = batch * width if target_token_capacity is None else target_token_capacity
        if (type(tokens) is not int or not batch <= tokens <= batch * width
                or (proposal_only and target_token_capacity is not None)):
            raise ValueError("packed verification capacity must cover the request batch within its prefix bounds")
        self.target_token_capacity = tokens
        self.generate_proposals = generate_proposals
        self.proposal_only = proposal_only
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.sampling_scratch = {} if self.vocabulary_parallel else prepare_sampling_scratch(b, w, o.requests.vocab_size, device, buffers=buffers,
                                                        proposal=generate_proposals, verification=not proposal_only)
        self.rows = torch.arange(b, device=device)
        self.draft_positions = torch.arange(w - 1, device=device) if self.vocabulary_parallel else None
        self.metadata = Metadata.allocate(tokens, b, blocks, device, samples=b * w, max_query_tokens=w)
        self.validator = PreparedBatch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
            target_token_capacity=tokens,
            page_tokens=o.page_tokens,
        )
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.collectives["runtime"],
            coordinator=o.collectives["coordinator"],
            session=o.session,
            name=name + ".validation",
        )
        self.decoder = None if proposal_only else o.forwards.decoder(
            o.weights,
            o.tables,
            o.scratch.rows(tokens, b * w),
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            **o.collectives,
            **({"vocabulary_parallel": True} if self.vocabulary_parallel else {}),
        )
        self.context = None if proposal_only else o.forwards.context(
            o.weights,
            self.decoder.auxiliary,
            self.metadata,
            o.pool,
            session=o.session,
            name=name + ".context",
            workspace=o.workspace,
        )
        block = o.block_size
        self.draft_metadata = Metadata.allocate(b * block, b, blocks, device, samples=b * block)
        self.draft = (
            None
            if w == 1 or not generate_proposals
            else o.forwards.draft(
                o.weights,
                o.draft_scratch.rows(b * block, b * block),
                self.draft_metadata,
                o.pool,
                session=o.session,
                name=name + ".draft",
                workspace=o.workspace,
                **o.collectives,
                **({"vocabulary_parallel": True} if self.vocabulary_parallel else {}),
            )
        )
        self.selector = getattr(self.draft, "selector", None)
        self.draft_inputs = torch.full(
            (b, block), o.weights.config.draft.noise_token, dtype=torch.int64, device=device
        )
        self.markov_token = torch.zeros(b, dtype=torch.int64, device=device)
        self.markov = (
            None
            if self.draft is None or o.forwards.markov is None or self.vocabulary_parallel
            else o.forwards.markov(
                o.weights,
                self.markov_token,
                runtime=o.markov_runtime,
                coordinator=o.collectives["coordinator"],
                session=o.session,
                name=name + ".markov",
            )
        )
        if self.draft is not None and self.vocabulary_parallel:
            from .dspark import PreparedMarkovEmbedding

            self.markov = PreparedMarkovEmbedding(o.weights, self.markov_token,
                                                  session=o.session, name=name + ".markov")
        self.shortlisted_selector = (self.selector if self.markov is None
            and isinstance(self.selector, PreparedCandidateSelector) else None)
        self.before = torch.zeros(b, dtype=torch.int64, device=device)
        self.active = torch.zeros(b, dtype=torch.bool, device=device)
        self.candidates = torch.zeros((b, w), dtype=torch.int64, device=device)
        self.drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=device)
        self.argmax = None
        if self.draft is not None and getattr(o, "draft_argmax", None) is not None:
            from .collectives import PreparedVocabularyArgmax

            self.greedy_tokens = torch.zeros(b, dtype=torch.int64, device=device)
            self.local_logits = self.draft.local_logits.view(b, block, -1)
            self.argmax = PreparedVocabularyArgmax(self.local_logits[:, o.logit_offset], self.markov.local_bias,
                self.greedy_tokens, runtime=o.draft_argmax, session=o.session,
                coordinator=o.collectives["coordinator"], name=name + ".draft_argmax")
        from ignition.preparation.scratch import bind_scratch

        local_vocab = o.scratch.local_logits.shape[-1] if self.vocabulary_parallel else o.requests.vocab_size
        layout = self.storage_layout(b, w, local_vocab)
        if not generate_proposals or self.vocabulary_parallel:
            del layout["logits"]
        retained = bind_scratch(layout, device=device, workspace=buffers)
        self.logits, self.probabilities = retained.get("logits"), retained["probabilities"]
        for tensor in retained.values():
            tensor.zero_()
        self.valid = torch.zeros((b, w - 1), dtype=torch.bool, device=device)
        self.seed, self.offset, self.base_offset, self.sampled = (
            torch.zeros_like(self.before) for _ in range(4)
        )
        self.operations = None
        self.proposal_block = None
        self.full_proposal = None

    def prepare(self):
        if self.operations is not None:
            return
        o, b, w = self.owner, self.batch, self.width
        if self.vocabulary_parallel:
            self._prepare_vocabulary_parallel()
            return
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        target = None if self.proposal_only else PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1), *args, self.drafts,
            workspace=self.sampling_scratch.get("target")
        )
        head = getattr(getattr(o.weights, "draft", None), "markov_head", None)
        from .markov_sampling import RowScaledFP8MarkovWeight

        # Full-vocabulary transition fusion improves one-row proposals. Wider
        # rows retain the bounded sampler because transition-weight reuse wins.
        fuse_markov = (b == 1 and w > 1 and self.generate_proposals and self.argmax is None
            and getattr(o, "block_proposal_broadcast", False) and o.markov_runtime is None
            and self.selector is None and self.markov is not None
            and ((isinstance(head, torch.Tensor) and head.dtype == torch.bfloat16 and head.is_contiguous())
                 or isinstance(head, RowScaledFP8MarkovWeight)))
        proposals = tuple(
            PreparedDraftDistribution(
                self.logits,
                *args,
                self.drafts,
                prefix_offset=i,
                seed=self.seed,
                offset=self.offset,
                sampled=self.sampled,
                workspace=self.sampling_scratch.get("proposal"),
            )
            for i in range(w - 1 if self.generate_proposals and self.argmax is None and not fuse_markov else 0)
        )
        if target is not None:
            target.prepare()
        for index, proposal in enumerate(proposals):
            bias = None if self.markov is None else getattr(self.markov, "logits", None)
            if (self.draft is not None and self.selector is None
                    and (self.markov is None or bias is not None)):
                block = self.draft_inputs.shape[1]
                proposal.bind_logits(self.draft.scratch.logits.view(b, block, -1)[:, index + o.logit_offset],
                    bias)
            proposal.prepare()
        if fuse_markov:
            from .markov_sampling import PreparedMarkovProposals, PreparedDraftTokenBroadcast

            block = self.draft_inputs.shape[1]
            base = self.draft.scratch.logits.view(b, block, -1)[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedMarkovProposals(base, o.weights.draft.markov_embedding, head,
                self.markov_token, self.drafts, self.probabilities, self.valid, o.requests,
                o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
                self.seed, self.base_offset, workspace=self.buffers)
            self.full_proposal.prepare()
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftTokenBroadcast(self.drafts, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_tokens", buffers=self.buffers)
        broadcast_proposals = proposals
        if proposals and getattr(o, "block_proposal_broadcast", False):
            if o.markov_runtime is not None or self.selector is not None:
                raise ValueError("block proposal broadcast requires replicated Markov work without a selector")
            runtime = getattr(o, "result_runtime", None)
            if runtime is None:
                runtime = o.collectives["vocabulary_runtime"]
            self.proposal_block = PreparedDraftBlockBroadcast(
                proposals, self.drafts, self.probabilities, self.valid, self.sampled,
                runtime=runtime, session=o.session, coordinator=o.collectives["coordinator"],
                name=self.name + ".proposal_block", buffers=self.buffers)
            broadcast_proposals = ()
        if self.proposal_only:
            prepare_sampling_broadcasts(broadcast_proposals, None, collectives=o.collectives, session=o.session,
                                        name=self.name, buffers=self.buffers,
                                        result_runtime=getattr(o, "result_runtime", None))
            self.operations = None, proposals, None, None
            return
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=o.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=o.inputs.query_lengths[:b],
            workspace=self.sampling_scratch.get("verify"),
        )
        verify.prepare()
        prepare_sampling_broadcasts(broadcast_proposals, verify, collectives=o.collectives, session=o.session,
                                    name=self.name, buffers=self.buffers,
                                    result_runtime=getattr(o, "result_runtime", None))
        commit = PreparedSpeculativeCommit(
            o.requests,
            verify.output,
            self.candidates,
            o.inputs.row_slot_ids[:b],
            o.inputs.row_generations[:b],
            o.inputs.live_rows,
            o.completion,
            o.frame,
            o.sequence,
            logits=self.decoder.scratch.logits.view(b, w, -1), rng=o.rng,
            draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b],
        )
        commit.prepare()
        self.operations = target, proposals, verify, commit

    def _prepare_vocabulary_parallel(self):
        from .vocabulary_sampling import PreparedVocabularySoftmax, PreparedVocabularyBlockVerification

        o, b, w = self.owner, self.batch, self.width
        args = (o.requests, o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows)
        start, _ = o.weights.shard.padded_partition(o.requests.vocab_size, alignment=64)
        options = dict(start=start, runtime=o.sampling_runtime, session=o.session,
                       coordinator=o.collectives["coordinator"])
        if self.draft is not None:
            local = self.draft.scratch.local_logits.view(b, o.block_size, -1)
            base = local[:, o.logit_offset:o.logit_offset + w - 1]
            self.full_proposal = PreparedVocabularySoftmax(base, self.probabilities, self.valid, *args,
                self.drafts, **options, name=self.name + ".proposal", workspace=self.buffers,
                embedding=o.weights.draft.markov_embedding, head=o.weights.draft.markov_head,
                pending=self.markov_token, seeds=self.seed, offsets=self.base_offset, tokens=self.drafts)
            self.full_proposal.prepare()
        if self.proposal_only:
            self.operations = None, (), None, None
            return
        logits = self.decoder.local_logits.view(b, w, -1)
        from ignition.preparation.scratch import bind_scratch

        retained = bind_scratch(dict(probabilities=(tuple(logits.shape), torch.float32),
            valid=((b, w), torch.bool)), device=logits.device, workspace=self.buffers)
        target = PreparedVocabularySoftmax(logits, retained["probabilities"], retained["valid"], *args,
            self.drafts, **options, name=self.name + ".target", workspace=self.buffers)
        target.prepare()
        verify = PreparedVocabularyBlockVerification(self.probabilities, self.drafts, target.probabilities, *args,
            **options, name=self.name + ".verification", rng=o.rng, draft_valid=self.valid,
            target_valid=target.valid, token_limits=o.inputs.query_lengths[:b], workspace=self.buffers)
        verify.prepare()
        from .vocabulary_logprobs import PreparedVocabularyParallelLogprobs

        logprobs = PreparedVocabularyParallelLogprobs(logits, o.completion,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows, o.frame,
            vocabulary=o.requests.vocab_size, offset=start, runtime=o.logprob_runtime, session=o.session,
            coordinator=o.collectives["coordinator"], name=self.name + ".logprobs")
        commit = PreparedSpeculativeCommit(o.requests, verify.output, self.candidates,
            o.inputs.row_slot_ids[:b], o.inputs.row_generations[:b], o.inputs.live_rows,
            o.completion, o.frame, o.sequence, rng=o.rng, draft_context_lag=0,
            token_limits=o.inputs.query_lengths[:b], logprobs=logprobs)
        commit.prepare()
        self.operations = target, (), verify, commit

    def __call__(self):
        self.propose()
        self.verify()

    def propose(self, advance_rng=True, observer=None, allowed=None):
        if not self.generate_proposals:
            raise RuntimeError("verification-only preparation cannot generate proposals")
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        self.validator.begin(self.validation(self.validator.validate()))
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        self.active.copy_(self.validator.token_state.active.bool())
        if allowed is not None:
            self.active.logical_and_(allowed)
        self.before.copy_(o.requests.consumed_tokens[safe])
        limits = torch.where(self.active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        self.candidates[:, 0].copy_(o.requests.pending_token[safe])
        self.base_offset.copy_(
            torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0)
        )
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        if self.draft is not None:
            block = self.draft_inputs.shape[1]
            self.draft_inputs[:, 0].copy_(self.candidates[:, 0])
            self.draft_metadata.set(
                self.before,
                self.active.int() * block,
                states,
                o.inputs.page_tables[safe],
                self.draft_inputs,
                o.inputs.live_rows,
            )
            self.draft()
            if self.shortlisted_selector is not None:
                self.shortlisted_selector.prepare_shortlists(self.draft.scratch.logits.view(b, block, -1))
            self.markov_token.copy_(torch.where(self.active, self.candidates[:, 0], 0))
            if self.argmax is not None:
                # Greedy drafts: each position's shard logits plus its Markov bias, from the previous draft.
                for i in range(w - 1):
                    self.markov()
                    self.argmax(self.local_logits[:, i + o.logit_offset], self.markov.local_bias, self.greedy_tokens)
                    self.drafts[:, i].copy_(self.greedy_tokens)
                    self.valid[:, i].copy_(self.active & (i < limits - 1))
                    self.markov_token.copy_(self.greedy_tokens)
                self.probabilities.zero_()
                self.probabilities.scatter_(2, self.drafts[:, :, None], 1.0)
            if self.full_proposal is not None:
                self.full_proposal()
                if self.vocabulary_parallel:
                    self.valid.logical_and_(self.active[:, None] & (self.draft_positions[None, :] < limits[:, None] - 1))
                    self.sampled.copy_(self.drafts[:, -1])
                    self.markov_token.copy_(self.sampled)
            for i, proposal in enumerate(proposals):
                if proposal.source_logits is not None:
                    if self.markov is not None:
                        self.markov()
                else:
                    self.logits[:, 0].copy_(
                        self.draft.scratch.logits.view(b, block, -1)[:, i + o.logit_offset].float()
                    )
                    if self.markov is not None:
                        self.logits[:, 0].add_(self.markov())
                    if self.shortlisted_selector is not None:
                        self.shortlisted_selector.apply_shortlist(self.logits[:, 0], self.markov_token, i + o.logit_offset)
                    elif self.selector is not None:
                        self.selector(self.logits[:, 0], self.markov_token, i + o.logit_offset)
                if observer is not None:
                    observer.record_embedding(i, self.markov.embedding)
                self.offset.copy_(self.base_offset + 4 * i)
                proposal()
                self.drafts[:, i].copy_(self.sampled)
                if self.proposal_block is None:
                    self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
                    self.valid[:, i].copy_(proposal.valid[:, 0] & self.active & (i < limits - 1))
                else:
                    self.proposal_block.retain(i, proposal)
                self.markov_token.copy_(self.sampled)
            if self.proposal_block is not None:
                self.proposal_block.finish(self.active, limits)
                self.markov_token.copy_(self.sampled)
            if advance_rng:
                self.commit_draft_rng()
        self.candidates[:, 1:].copy_(self.drafts)

    def commit_draft_rng(self, drafts=None):
        o, b, w = self.owner, self.batch, self.width
        drafts = w - 1 if drafts is None else drafts
        slots = o.inputs.row_slot_ids[:b].long()
        slot_ids = torch.arange(o.requests.capacity, device=slots.device)
        matches = self.active[None, :] & (slots[None, :] == slot_ids[:, None])
        base = torch.where(matches, self.base_offset[None, :], 0).sum(1)
        o.offset.copy_(torch.where(matches.any(1), base + 4 * drafts, o.offset))
        o.generation.copy_(torch.where(matches.any(1), o.requests.generation, o.generation))

    def verify(self, proposal_valid=None):
        o, b, w = self.owner, self.batch, self.width
        target, _, verify, commit = self.operations
        slots = o.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, o.requests.capacity - 1)
        active = self.active if proposal_valid is None else self.active & proposal_valid
        limits = torch.where(active, o.inputs.query_lengths[:b], 0)
        states = o.inputs.request_state_rows[safe, 0]
        self.metadata.set(
            self.before, limits, states, o.inputs.page_tables[safe], self.candidates, o.inputs.live_rows
        )
        self.decoder()
        target()
        verify()
        if proposal_valid is not None:
            verify.output.valid.logical_and_(proposal_valid)
        commit()
        counts = commit.emitted_count
        if w > 1:
            self.decoder.commit(counts)
        self.context(self.decoder.auxiliary, counts)
        hidden = self.decoder.scratch.hidden.view(b, w, o.weights.config.text["hidden_size"])[
            self.rows, (counts - 1).long().clamp_min(0)
        ]
        save_hidden(hidden, o.pool.last_hidden, self.metadata.state_rows, counts > 0)
