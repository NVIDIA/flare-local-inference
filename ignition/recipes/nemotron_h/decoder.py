"""Nemotron-H heterogeneous residual blocks and vocabulary projection."""

from dataclasses import dataclass

import torch
import triton

from ignition.components.embedding import PreparedEmbedding
from ignition.components.mamba_speculation import PreparedMambaReplay
from ignition.components.paged_attention import PagedAttentionMetadata
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import PreparedAttention
from .config import PAGE_TOKENS
from .mamba import PreparedMamba
from .moe import PreparedLatentMoE
from .pool import REPLAY_RING


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, buffers):
        def zeros(shape):
            return buffers.zeros(shape, torch.bfloat16)
        return cls(zeros((tokens, config.hidden)), zeros((tokens, config.hidden)),
            zeros((rows, config.hidden)), zeros((rows, config.vocabulary)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.normalized[:tokens], self.hidden[:rows],
                          self.local_logits[:rows], self.logits[:rows])


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, session, name, workspace,
                 layer_buffers, coordinator=None, vocabulary_runtime=None, buffers=None, vision_state=None, scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision_state = vision_state if mode == "prefill" else None
        self.cache_metadata = PagedAttentionMetadata(metadata, page_tokens=PAGE_TOKENS, buffers=buffers)
        config, source = weights.config, scratch.normalized
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device)
        self.kv_scale = torch.ones(1, device=source.device, dtype=torch.float32)
        self.target_layers = () if weights.draft is None else config.draft.target_layers
        self.auxiliary = None if not self.target_layers else buffers.zeros(
            (source.shape[0], len(self.target_layers) * config.hidden), torch.bfloat16)
        self.width = metadata.max_query_tokens if mode == "verify" else 1
        self.replay = None
        if weights.draft is not None:
            indices = tuple(i for i, kind in enumerate(config.layers) if kind == "mamba")
            regions = [tuple(pool.layout.region(f"layer{i}.{field}") for field in
                             ("recurrent", "replay_x", "replay_b", "replay_dt")) for i in indices]
            self.replay = PreparedMambaReplay(pool.reserved_backing, pool.layout, regions,
                torch.stack([weights.layers[i]["A"] for i in indices]), metadata.state_rows, metadata.lengths,
                ring_region=pool.layout.region("mamba_ring_start"), count_region=pool.layout.region("mamba_pending_count"),
                ring_length=REPLAY_RING, buffers=buffers)
            convolutions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = torch.tensor([r.offset // 2 for r in convolutions], dtype=torch.int64, device=source.device)
            self.convolution_shape = convolutions[0].shape
        self.layers = []
        previous = None
        for index, w in enumerate(weights.layers):
            prefix, kind = f"{name}.layer{index}", config.layers[index]
            buffers = layer_buffers.cursor()
            norm = PreparedRMSNorm(scratch.activations, w["norm"], source,
                eps=config.text["layer_norm_epsilon"], residual=previous, session=session, name=prefix + ".norm")
            kwargs = dict(session=session, name=prefix, workspace=workspace, buffers=buffers)
            if kind == "mamba":
                block = PreparedMamba(w, source, pool.recurrent(index), self.delta, config, mode=mode,
                    replay=self.replay, records=pool.replay_records(index) if mode == "verify" else None,
                    verification_width=self.width, **kwargs)
            elif kind == "attention":
                key, value = pool.attention(index, config)
                block = PreparedAttention(w, source, metadata, key, value, config, mode=mode,
                    tuning_page_count=int(pool.memory_capacity[1].item()),
                    key_scale=self.kv_scale, value_scale=self.kv_scale, cache_metadata=self.cache_metadata, **kwargs)
            else:
                block = PreparedLatentMoE(w, source, config, index,
                    num_valid_tokens=metadata.live_counts[:1], **kwargs)
            # Every mixer returns a contiguous hidden-width matrix consumed by the following norm.
            previous = source
            self.layers.append((norm, block))
        self.final = PreparedRMSNorm(scratch.activations, weights.norm, source,
            eps=config.text["layer_norm_epsilon"], residual=previous, session=session, name=name + ".norm_f")
        self.scoring = None if scoring is None else scoring.prepare(scratch.normalized, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto")

    def commit(self, counts):
        self.replay.commit(counts, width=self.width)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, self.metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self):
        m, d = self.metadata, self.delta
        self.cache_metadata()
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        if self.replay is not None:
            self.replay.prepare(width=self.width)
        residual, output = self.embedding(), None
        if self.vision_state is not None:
            self.vision_state.inject(m, residual)
        for index, (norm, block) in enumerate(self.layers):
            normalized = norm(residual, output)
            if index - 1 in self.target_layers:
                offset = self.target_layers.index(index - 1) * residual.shape[1]
                self.auxiliary[:, offset:offset + residual.shape[1]].copy_(residual)
            output = block(normalized)
        normalized = self.final(residual, output)
        if len(self.layers) - 1 in self.target_layers:
            offset = self.target_layers.index(len(self.layers) - 1) * residual.shape[1]
            self.auxiliary[:, offset:offset + residual.shape[1]].copy_(residual)
        if self.scoring is not None:
            self.scoring()
        self.scratch.hidden.copy_(normalized[m.sample_indices])
        self.head(self.scratch.hidden)
