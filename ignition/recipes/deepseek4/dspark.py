"""Checkpoint-integrated noncausal drafts with learned hyperconnection reduction."""

from functools import partial

from ignition.components.dspark import DSparkForwards, PreparedContext, PreparedMarkov
from ignition.components.embedding import PreparedEmbedding
from ignition.components.mhc import PreparedMHCHead
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.rotary import CompressedRotary
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import AttentionIndices
from .decoder import PreparedDecoder, PreparedLayer, Scratch


class PreparedDraft:
    def __init__(
        self,
        weights,
        scratch,
        metadata,
        pool,
        *,
        runtime,
        vocabulary_runtime,
        coordinator,
        session,
        name,
        workspace,
    ):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            scratch.embedding,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        )
        rotary = CompressedRotary(weights.config, scratch.embedding.device)
        self.indices = AttentionIndices(
            metadata, mode="draft", tokens=metadata.token_ids.numel(), visible=None, buffers=scratch.buffers
        )
        self.layers = tuple(
            PreparedLayer(
                i,
                layer,
                scratch,
                metadata,
                pool,
                rotary,
                shard=weights.shard,
                mode="draft",
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=f"{name}.layer{i}",
                workspace=workspace,
                carry=i > 0,
                defer_post=i + 1 < len(weights.draft.layers),
                indices=self.indices,
            )
            for i, layer in enumerate(weights.draft.layers)
        )
        self.head_hidden = scratch.collapsed
        self.collapse = PreparedMHCHead(
            scratch.head_residual,
            weights.draft.head_mix,
            self.head_hidden,
            rms_eps=weights.config.text["rms_norm_eps"],
            hc_eps=weights.config.text["hc_eps"],
            session=session,
            name=name + ".collapse",
        )
        self.norm = PreparedRMSNorm(
            self.head_hidden,
            weights.draft.norm,
            scratch.hidden,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".norm",
        )
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            weights.head if weights.draft.draft_head is None else weights.draft.draft_head,
            scratch.local_logits,
            scratch.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
        )

    def __call__(self):
        s = self.scratch
        s.residual.copy_(self.embedding()[:, None, :])
        self.indices()
        for layer in self.layers:
            layer()
        s.head_residual.copy_(s.residual[self.metadata.sample_indices])
        self.norm(self.collapse(s.head_residual))
        self.head(s.hidden)


FORWARDS = DSparkForwards(
    PreparedDecoder,
    Scratch,
    partial(PreparedContext, cache_format="deepseek_v4", compact_prefill=True),
    PreparedDraft,
    PreparedMarkov,
)
