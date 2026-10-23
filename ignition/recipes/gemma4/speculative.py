"""Gemma 4 Assistant frozen-KV multi-token proposals and stochastic verification."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.embedding import PreparedEmbedding
from ignition.components.paged_attention import PagedAttentionMetadata
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.parallel_draft import prepare_sampling_broadcasts, prepare_sampling_scratch
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import gather_rows, save_hidden
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import attention_descales, prepare_projection, rms_norm
from .config import PAGE_TOKENS
from .decoder import PreparedDecoder, PreparedLayer, Scratch, softcap_logits


class PreparedContext:
    """No-op draft context handler since PreparedGeneration already saves pool.last_hidden."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def scratch_tensors(self):
        return ()


class PreparedAssistant:
    """Single-position recurrent Gemma 4 Assistant reading from the frozen target KV cache."""

    def __init__(
        self,
        weights,
        pool,
        batch: int,
        blocks: int,
        *,
        runtime=None,
        vocabulary_runtime=None,
        coordinator=None,
        session,
        name: str,
        workspace,
        layer_buffers,
        buffers,
    ):
        self.weights = weights
        self.pool = pool
        self.batch = batch
        target_config = weights.config
        draft_config = weights.config.draft
        draft = weights.draft
        self.target_config = target_config
        self.draft_config = draft_config
        self.eps = draft_config.rms_norm_eps
        self.softcap = draft_config.final_logit_softcapping
        device = pool.backing.device

        self.metadata = Metadata.allocate(batch, batch, blocks, device, samples=batch, buffers=buffers)
        self.cache_metadata = PagedAttentionMetadata(self.metadata, page_tokens=PAGE_TOKENS, buffers=buffers)

        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)

        self.token_embed = zeros((batch, target_config.hidden))
        self.embedding = PreparedEmbedding(
            weights.embedding,
            self.metadata.token_ids,
            self.token_embed,
            num_rows=self.metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        )
        self.embed_scale = torch.tensor(target_config.hidden**0.5, dtype=torch.bfloat16, device=device)

        self.concat_in = zeros((batch, 2 * target_config.hidden))
        self.hidden_states = zeros((batch, draft_config.hidden))
        self.normed = zeros((batch, draft_config.hidden))
        self.sampled_normed = zeros((batch, draft_config.hidden))
        self.projected = zeros((batch, target_config.hidden))

        self.pre_projection = prepare_projection(
            self.concat_in,
            draft.pre_projection,
            self.hidden_states,
            session=session,
            name=name + ".pre_projection",
            workspace=workspace,
        )
        self.post_projection = prepare_projection(
            self.sampled_normed,
            draft.post_projection,
            self.projected,
            session=session,
            name=name + ".post_projection",
            workspace=workspace,
        )

        self.sliding_inv_freq = draft_config.rotary_inverse("sliding_attention", device)
        self.full_inv_freq = draft_config.rotary_inverse("full_attention", device)
        unit_scale, sliding_descale, full_descale = attention_descales(draft_config, device)

        self.layers = []
        for index, layer_weights in enumerate(draft.layers):
            cursor = layer_buffers.cursor()
            descale = sliding_descale if draft_config.is_sliding(index) else full_descale
            self.layers.append(
                PreparedLayer(
                    index,
                    layer_weights,
                    draft_config,
                    self.normed,
                    self.metadata,
                    pool,
                    shard=weights.shard,
                    runtime=runtime,
                    coordinator=coordinator,
                    mode="decode",
                    session=session,
                    name=f"{name}.layer{index}",
                    workspace=workspace,
                    buffers=cursor,
                    cache_metadata=self.cache_metadata,
                    unit_scale=unit_scale,
                    k_descale=descale,
                )
            )

        self.logits = zeros((batch, target_config.vocabulary))
        if draft_config.use_ordered_embeddings:
            self.head = None
            self.local_logits = None
            self.centroid_logits = zeros((batch, draft_config.num_centroids))
            self.centroid_proj = prepare_projection(
                self.sampled_normed,
                draft.centroids,
                self.centroid_logits,
                session=session,
                name=name + ".centroids",
                workspace=workspace,
            )
            self.top_k = draft_config.centroid_intermediate_top_k
            self.vocab_per_centroid = target_config.vocabulary // draft_config.num_centroids
            self.canonical_table = draft.token_ordering.long().view(
                draft_config.num_centroids, self.vocab_per_centroid
            )
            self.head_weight = draft.head
        else:
            self.centroid_proj = None
            local_vocab = target_config.vocabulary_partition(weights.shard)[2]
            self.local_logits = zeros((batch, local_vocab))
            self.head = PreparedVocabularyProjection(
                self.sampled_normed,
                draft.head,
                self.local_logits,
                self.logits,
                runtime=vocabulary_runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".head",
                activation_mode="auto",
                buffers=buffers,
            )

    def step(
        self,
        step_index: int,
        tokens: torch.Tensor,
        prev_hidden: torch.Tensor,
        before: torch.Tensor,
        states: torch.Tensor,
        pages: torch.Tensor,
        live_rows: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        m = self.metadata
        starts = (before - 1).clamp_min(0)
        lengths = active.int()
        safe_pages = torch.where(active[:, None], pages, -1)
        m.set(starts, lengths, states, safe_pages, tokens[:, None], live_rows)
        self.cache_metadata()

        pos = torch.where(m.positions >= 0, m.positions + 1 + step_index, 0).float()[:, None]
        s_freq = pos * self.sliding_inv_freq[None, :]
        s_emb = torch.cat((s_freq, s_freq), dim=-1)
        sliding_cos, sliding_sin = s_emb.cos().bfloat16(), s_emb.sin().bfloat16()

        f_freq = pos * self.full_inv_freq[None, :]
        f_emb = torch.cat((f_freq, f_freq), dim=-1)
        full_cos, full_sin = f_emb.cos().bfloat16(), f_emb.sin().bfloat16()

        token_embed = self.embedding() * self.embed_scale
        prev_packed = prev_hidden[m.requests]
        self.concat_in.copy_(torch.cat((token_embed, prev_packed), dim=-1))
        hidden = self.pre_projection(self.concat_in)

        for layer in self.layers:
            cos, sin = (sliding_cos, sliding_sin) if layer.is_sliding else (full_cos, full_sin)
            hidden = layer(hidden, cos, sin)

        normed = rms_norm(hidden, self.weights.draft.norm, self.eps)
        self.sampled_normed.copy_(normed[m.sample_indices])
        self.post_projection(self.sampled_normed)

        if self.head is not None:
            self.head(self.sampled_normed)
        else:
            b = self.batch
            self.centroid_proj(self.sampled_normed)
            _, top_k_indices = torch.topk(self.centroid_logits, k=self.top_k, dim=-1)
            canonical = self.canonical_table[top_k_indices].reshape(b, self.top_k * self.vocab_per_centroid)
            selected_embeds = self.head_weight[canonical.reshape(-1)].view(
                b, self.top_k * self.vocab_per_centroid, self.draft_config.hidden
            )
            selected_logits = torch.bmm(
                self.sampled_normed.unsqueeze(1), selected_embeds.transpose(1, 2)
            ).squeeze(1)
            mask_value = selected_logits.amin(dim=-1, keepdim=True) - 1.0
            self.logits.copy_(mask_value.expand_as(self.logits))
            self.logits.scatter_(dim=-1, index=canonical, src=selected_logits)
        softcap_logits(self.logits, self.softcap)

    def scratch_tensors(self):
        return (
            self.metadata,
            self.token_embed,
            self.concat_in,
            self.hidden_states,
            self.normed,
            self.sampled_normed,
            self.projected,
            self.logits,
            self.local_logits,
        )


class PreparedRound:
    """Multi-step Assistant proposal loop followed by parallel target verification and commit."""

    def __init__(self, owner, batch: int, width: int, name: str, *, buffers):
        o, b, w = owner, batch, width
        self.batch, self.width, self.buffers = b, w, buffers
        self.collectives, self.session, self.name = o.collectives, o.session, name
        self.inputs, self.requests, self.pool = o.inputs, o.requests, o.pool
        self.completion, self.frame, self.sequence = o.completion, o.frame, o.sequence
        self.rng, self.draft_offset, self.draft_generation = o.rng, o.offset, o.generation
        self.hidden = o.weights.config.hidden
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w, buffers=buffers)
        self.validator = PreparedBatch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device, buffers=buffers),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
            buffers=buffers,
        )
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.collectives["runtime"],
            coordinator=o.collectives["coordinator"],
            session=o.session,
            name=name + ".validation",
        )
        with o.scratch_pool:
            scratch = Scratch.allocate(
                b * w, b * w, o.weights.config, o.scratch_buffers.cursor(), shard=o.weights.shard
            )
        self.decoder = PreparedDecoder(
            o.weights,
            scratch,
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            layer_buffers=o.layer_buffers,
            buffers=buffers,
            **o.collectives,
        )
        self.draft = (
            None
            if w == 1
            else PreparedAssistant(
                o.weights,
                o.pool,
                b,
                blocks,
                session=o.session,
                name=name + ".draft",
                workspace=o.workspace,
                layer_buffers=o.layer_buffers,
                buffers=buffers,
                **o.collectives,
            )
        )

        def zeros(shape, dtype=torch.int64):
            return buffers.zeros(shape, dtype)

        self.previous = zeros((b, self.hidden), torch.bfloat16)
        self.before = zeros((b,))
        self.active = zeros((b,), torch.bool)
        self.candidates = zeros((b, w))
        self.drafts = zeros((b, w - 1))
        self.probabilities = zeros((b, w - 1, o.requests.vocab_size), torch.float32)
        self.valid = zeros((b, w - 1), torch.bool)
        self.seed, self.offset, self.base_offset, self.sampled = (zeros((b,)) for _ in range(4))
        self.sampling_scratch = prepare_sampling_scratch(
            b, w, o.requests.vocab_size, device, buffers=buffers
        )
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        b, w = self.batch, self.width
        args = (
            self.requests,
            self.inputs.row_slot_ids[:b],
            self.inputs.row_generations[:b],
            self.inputs.live_rows,
        )
        target = PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1),
            *args,
            self.drafts,
            workspace=self.sampling_scratch["target"],
        )
        proposals = (
            ()
            if w == 1
            else tuple(
                PreparedDraftDistribution(
                    self.draft.logits.view(b, 1, -1),
                    *args,
                    self.drafts,
                    prefix_offset=i,
                    seed=self.seed,
                    offset=self.offset,
                    sampled=self.sampled,
                    workspace=self.sampling_scratch["proposal"],
                )
                for i in range(w - 1)
            )
        )
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=self.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=self.inputs.query_lengths[:b],
            workspace=self.sampling_scratch["verify"],
        )
        verify.prepare()
        prepare_sampling_broadcasts(
            proposals,
            verify,
            collectives=self.collectives,
            session=self.session,
            name=self.name,
            buffers=self.buffers,
        )
        commit = PreparedSpeculativeCommit(
            self.requests,
            verify.output,
            self.candidates,
            self.inputs.row_slot_ids[:b],
            self.inputs.row_generations[:b],
            self.inputs.live_rows,
            self.completion,
            self.frame,
            self.sequence,
            logits=self.decoder.scratch.logits.view(b, w, -1),
            rng=self.rng,
            token_limits=self.inputs.query_lengths[:b],
            draft_context_lag=0,
            buffers=self.buffers,
        )
        commit.prepare()
        self.operations = target, proposals, verify, commit

    def __call__(self):
        b, w = self.batch, self.width
        target, proposals, verify, commit = self.operations
        self.validator.begin(self.validation(self.validator.validate()))
        slots = self.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, self.requests.capacity - 1)
        self.active.copy_(self.validator.token_state.active.bool())
        self.before.copy_(self.requests.consumed_tokens[safe])
        limits = torch.where(self.active, self.inputs.query_lengths[:b], 0)
        states = self.inputs.request_state_rows[safe, 0]
        pages = self.inputs.page_tables[safe]
        self.candidates[:, 0].copy_(self.requests.pending_token[safe])
        self.base_offset.copy_(
            torch.where(
                self.draft_generation[safe] == self.requests.generation[safe],
                self.draft_offset[safe],
                0,
            )
        )
        self.seed.copy_(self.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        if w > 1:
            gather_rows(self.pool.last_hidden, torch.where(self.active, states, 0), self.previous)
            for i, proposal in enumerate(proposals):
                tokens = self.candidates[:, 0] if i == 0 else self.drafts[:, i - 1]
                prev_hidden = self.previous if i == 0 else self.draft.projected
                proposing = self.active & (i < limits - 1)
                self.draft.step(
                    i, tokens, prev_hidden, self.before, states, pages, self.inputs.live_rows, self.active
                )
                self.offset.copy_(self.base_offset + 4 * i)
                proposal()
                self.drafts[:, i].copy_(self.sampled)
                self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
                self.valid[:, i].copy_(proposal.valid[:, 0] & proposing)
            matches = self.active[None] & (
                slots[None] == torch.arange(self.requests.capacity, device=slots.device)[:, None]
            )
            base = torch.where(matches, self.base_offset[None], 0).sum(1)
            self.draft_offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), self.draft_offset))
            self.draft_generation.copy_(torch.where(matches.any(1), self.requests.generation, self.draft_generation))
        self.candidates[:, 1:].copy_(self.drafts)
        self.metadata.set(self.before, limits, states, pages, self.candidates, self.inputs.live_rows)
        self.decoder()
        target()
        verify()
        commit()
        counts = commit.emitted_count
        hidden = self.decoder.scratch.hidden.view(b, w, self.hidden)[
            self.rows, (counts - 1).long().clamp_min(0)
        ]
        save_hidden(hidden, self.pool.last_hidden, self.metadata.state_rows, counts > 0)

    def scratch_tensors(self):
        return (
            self.metadata,
            self.decoder.scratch,
            self.previous,
            self.before,
            self.active,
            self.candidates,
            self.drafts,
            self.probabilities,
            self.valid,
            self.seed,
            self.offset,
            self.base_offset,
            self.sampled,
            () if self.draft is None else self.draft.scratch_tensors(),
        )
