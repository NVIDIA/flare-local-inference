"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.ring_commit import PreparedRingCommit
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import RouterScratch, PreparedRouter, PreparedRoutedExperts
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.scratch import GraphBufferPool, GraphBuffers

from .attention import AttentionIndices, AttentionScratch, PreparedAttention, PreparedGlobalKV, Rotary, normalize
from .decoder_prefill import PreparedDecoderSuffix
from .engram import EngramScratch, PreparedEngram, PreparedTableRequests


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    residual: torch.Tensor
    engram_residual: torch.Tensor
    engram: EngramScratch
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    buffers: GraphBuffers

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if isinstance(value, GraphBuffers):
                return value.fork()
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, field.name), count) for field in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")
        return Scratch(*(select(getattr(self, field.name), samples if field.name in
            ("hidden", "local_logits", "logits") else tokens) for field in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, experts=384, top_k=6, buffers=None,
                 candidate_tokens=None):
        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        def zeros(*shape, dtype=torch.bfloat16, cursor=buffers):
            return cursor.zeros(shape, dtype)
        residual, engram_residual = zeros(tokens, 4, 5120), zeros(tokens, 4, 5120)
        pre_mix, attn_mix = zeros(tokens, 4, dtype=torch.float32), zeros(tokens, 4, dtype=torch.float32)
        attention_hc = MHCScratch.allocate(tokens, 5120, device, buffers=buffers)
        ffn_hc = MHCScratch.allocate(tokens, 5120, device, buffers=buffers)
        attention = AttentionScratch.allocate(tokens, world, device, buffers=buffers, candidate_tokens=candidate_tokens)
        attention_output, mlp_output = zeros(tokens, 5120), zeros(tokens, 5120)
        hidden, local_logits, logits = zeros(samples, 5120), zeros(samples, 129280 // world), zeros(samples, 129280)
        # Embedding, Engram projection, and feed-forward intermediates execute serially.
        embed, lookup, ffn = buffers.fork(), buffers.fork(), buffers.fork()
        embedding = zeros(tokens, 5120, cursor=embed)
        engram = EngramScratch.allocate(tokens, device, buffers=lookup, world=world)
        shared = MLPScratch.allocate(tokens, 5120, 2304 // world, device, buffers=ffn)
        router_logits = zeros(tokens, experts, dtype=torch.float32, cursor=ffn)
        router = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k, buffers=ffn)
        routed, partial = zeros(tokens, 5120, cursor=ffn), zeros(tokens, 5120, cursor=ffn)
        buffers.join(embed, lookup, ffn)
        return cls(embedding, residual, engram_residual, engram, pre_mix, attn_mix, attention_hc, ffn_hc,
            attention, attention_output, shared, router_logits, router, routed, partial, mlp_output,
            hidden, local_logits, logits, buffers)


def _mhc_weights(weights, kind, norm):
    residual = weights.residual
    return MHCWeights(residual[f"hc_{kind}_fn"], residual[f"hc_{kind}_scale"],
                      residual[f"hc_{kind}_base"], residual[norm + ".weight"])


class PreparedDecoderBoundary:
    """Produce complete global KV, then gather the decoder's bounded replay rows."""

    def __init__(self, index, weights, source, target, packing, pool, rotary, *, carry,
                 session, name, workspace, window_metadata, source_indices, buffers=None):
        self.packing = packing
        self.window_metadata = window_metadata
        self.incoming = ((source.mlp_output, source.ffn_hc.residual, source.ffn_hc.post, source.ffn_hc.comb)
                         if carry else (source.residual,))
        text = pool.config.text
        self.mhc = PreparedMHC(self.incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace,
            pre_mix=source.pre_mix, pre_out=source.attn_mix, rms_eps=text["rms_norm_eps"],
            norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"])
        self.normalized = source.attention_hc.normalized
        self.global_kv = PreparedGlobalKV(index, weights.attention, self.normalized, source.attention,
            packing.source, pool, rotary, session=session, name=name + ".global_kv",
            indices=source_indices, buffers=buffers)
        self.rows = tuple((getattr(source.attention_hc, field), getattr(target.attention_hc, field))
                          for field in ("normalized", "residual", "post", "comb"))
        self.rows += ((source.attn_mix, target.attn_mix),)

    def __call__(self, indexer):
        self.packing.begin()
        # Packed positions and replay bounds must precede their shared addresses.
        self.window_metadata()
        self.mhc(*self.incoming)
        self.global_kv(self.normalized, indexer)
        for source, output in self.rows:
            self.packing.gather(source, output)


class PreparedLayer:
    def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, *, shard, mode,
                 runtime, coordinator, session, name, workspace, carry, defer_post,
                 boundary=None, replay_start=None, window_staging=None, mla_compute_mode="auto", buffers=None,
                 window_metadata=None):
        s, text = scratch, pool.config.text
        self.scratch, self.carry = s, carry
        self.metadata = metadata
        buffers = scratch.buffers if buffers is None else buffers
        self.image_mask = (buffers.zeros(metadata.token_ids.shape, torch.bool)
                           if mode == "prefill" and weights.experts.vision_correction is not None else None)
        self.boundary = boundary
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator)
        source = s.engram_residual if self.engram is not None else s.residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (source,))
        self.attention_hc = None if boundary is not None else PreparedMHC(
            incoming, _mhc_weights(weights, "attn", "attn_norm"), s.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace, pre_mix=s.pre_mix, pre_out=s.attn_mix, **options)
        self.attention = PreparedAttention(index, weights.attention, s.attention_hc.normalized, s.attention,
            metadata, pool, rotary, mode=mode, world=shard.world_size,
            session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
            window_staging=window_staging, mla_compute_mode=mla_compute_mode, buffers=buffers,
            window_metadata=window_metadata)
        self.attention_reduce = PreparedAllReduce(s.attention.output, s.attention_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".attention_reduce")
        self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            _mhc_weights(weights, "ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
            pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
            image_mask=self.image_mask)
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, s.router.ids,
            s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
            router_plan=self.router.plan)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
            s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
        self.concurrent_shared = None
        if mode in ("decode", "verify", "draft") and s.ffn_hc.normalized.shape[0] <= 128 and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                session=session, workspace=workspace, defer_launch=True)
        self.mlp_reduce = PreparedAllReduce(s.mlp_partial, s.mlp_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".mlp_reduce")
        self.post = None if defer_post else PreparedMHCPost(s.mlp_output, s.ffn_hc.residual,
            s.ffn_hc.post, s.ffn_hc.comb, s.residual, session=session, name=name + ".post")

    def __call__(self):
        s = self.scratch
        if self.engram is not None:
            source = self.engram(s.residual)
        else:
            source = s.residual
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)
                    if self.carry else (source,))
        if self.boundary is None:
            self.attention_hc(*incoming)
        else:
            self.boundary(self.attention.selectors[0])
        self.attention_reduce(self.attention(s.attention_hc.normalized))
        self.ffn_hc(s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb)
        if self.concurrent_shared is not None:
            self.concurrent_shared.start(s.ffn_hc.normalized)
        if self.image_mask is not None:
            self.image_mask.copy_(self.metadata.token_ids >= 129280)
        self.router(self.router_projection(s.ffn_hc.normalized))
        self.experts()
        if self.concurrent_shared is None:
            self.shared(s.ffn_hc.normalized)
        else:
            self.concurrent_shared.finish(s.ffn_hc.normalized, s.routed)
        torch.add(s.routed, s.shared.output, out=s.mlp_partial)
        self.mlp_reduce(s.mlp_partial)
        if self.post is not None:
            self.post(s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb)


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace,
                 window_staging=None, suffix_scratch=None, vision=None, mla_compute_mode="auto", scoring=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        self.table_requests = PreparedTableRequests(tables, metadata, pool, session=session, name=name + ".table_requests")
        decoder_start = weights.config.decoder_start
        self.packing = None
        if mode == "prefill" and metadata.positions.numel() > 128 and decoder_start is not None and scoring is None:
            if (any(layer.engram is not None for layer in weights.layers[decoder_start:])
                    or any(source.layer > decoder_start for source in weights.config.sources)):
                raise ValueError("decoder suffix replay requires Engram and compressed KV sources before its boundary")
            self.packing = PreparedDecoderSuffix(metadata)
        self.output_metadata = metadata if self.packing is None else self.packing.metadata
        self.output_scratch = scratch if self.packing is None else (
            suffix_scratch if suffix_scratch is not None else Scratch.allocate(
                128, scratch.hidden.shape[0], weights.shard.world_size, scratch.embedding.device))
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        if self.packing is not None and any(layer <= decoder_start for layer in self.aux_layers):
            raise ValueError("bounded decoder feedback requires draft target layers after its global-KV boundary")
        buffers = scratch.buffers
        self.auxiliary = buffers.zeros((self.output_metadata.token_ids.numel(), len(self.aux_layers) * 5120), torch.bfloat16)
        self.safe_tokens = buffers.zeros(metadata.token_ids.shape, metadata.token_ids.dtype) if self.vision is not None else metadata.token_ids
        self.embedding = PreparedEmbedding(weights.embedding, self.safe_tokens, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.window_commit = None
        if mode == "verify":
            self.window_commit = PreparedRingCommit(pool.reserved_backing,
                [pool.layout.region(f"layer{index}.window").offset for index in range(len(weights.layers))],
                metadata.state_rows, metadata.starts,
                token_capacity=metadata.max_query_tokens,
                window_tokens=128, record_bytes=528, staging=window_staging)
        # These addresses stay live through every layer and precede reusable layer buffers.
        ratios = weights.config.text["compress_ratios"][:len(weights.layers)]
        prefix_ratios = ratios if self.packing is None else ratios[:decoder_start + 1]
        self.window_metadata = AttentionIndices(metadata, mode=mode,
            tokens=metadata.positions.numel(),
            page_size=128 if mode == "decode" else 128 + metadata.max_query_tokens, width=128,
            ratios=set(prefix_ratios) - {0}, buffers=buffers)
        self.suffix_window_metadata = None
        if self.packing is not None:
            self.suffix_window_metadata = AttentionIndices(self.output_metadata, mode=mode,
                tokens=self.output_metadata.positions.numel(),
                page_size=128 + self.output_metadata.max_query_tokens, width=128,
                replay_start=self.packing.replay_start, ratios=set(ratios[decoder_start:]) - {0}, buffers=buffers)
        self.layers, carry = [], False
        layer_cursors = []
        for index, layer in enumerate(weights.layers):
            # Prefill commits each layer before the following layer reuses its private buffers.
            layer_buffers = buffers.fork() if mode == "prefill" else buffers
            layer_cursors.append(layer_buffers)
            boundary = replay_start = None
            layer_scratch, layer_metadata = scratch, metadata
            if self.packing is not None and index >= decoder_start:
                layer_scratch, layer_metadata = self.output_scratch, self.output_metadata
                replay_start = self.packing.replay_start
                if index == decoder_start:
                    boundary = PreparedDecoderBoundary(index, layer, scratch, layer_scratch, self.packing,
                        pool, self.rotary, carry=carry, session=session, name=f"{name}.layer{index}", workspace=workspace,
                        buffers=layer_buffers, window_metadata=self.suffix_window_metadata,
                        source_indices=self.window_metadata)
                    layer = replace(layer, attention={key: value for key, value in layer.attention.items()
                                                      if key not in ("compressor", "index_k", "index_norm")})
            defer = (index + 1 < len(weights.layers) and weights.layers[index + 1].engram is None
                     and index + 1 not in self.aux_layers)
            self.layers.append(PreparedLayer(index, layer, tables, layer_scratch, layer_metadata, pool, self.rotary,
                shard=weights.shard, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=carry, defer_post=defer, boundary=boundary, replay_start=replay_start,
                window_staging=None if self.window_commit is None else self.window_commit.staging[index],
                mla_compute_mode=mla_compute_mode, buffers=layer_buffers,
                window_metadata=(self.window_metadata if layer_metadata is metadata else self.suffix_window_metadata)))
            carry = defer
        buffers.join(*layer_cursors)
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt", copy_source=True)
        self.compact_head = scratch.hidden.shape[0] > self.output_metadata.token_ids.numel()
        head_rows = self.output_metadata.token_ids.numel() if self.compact_head else scratch.hidden.shape[0]
        self.head_hidden = scratch.hidden[:head_rows]
        self.head = PreparedVocabularyProjection(self.head_hidden, weights.head, scratch.local_logits[:head_rows],
            scratch.logits, runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=self.output_metadata.sample_indices if self.compact_head else None)

    def __call__(self):
        s, m = self.scratch, self.metadata
        self.table_requests()
        if self.vision is not None:
            self.safe_tokens.copy_(m.token_ids.clamp(0, self.weights.embedding.shape[0] - 1))
        self.embedding()
        if self.vision is not None:
            self.vision.inject(m, s.embedding)
        s.residual.copy_(s.embedding[:, None, :])
        s.pre_mix.zero_()
        s.pre_mix[:, 0].fill_(1)
        self.window_metadata()
        for index, layer in enumerate(self.layers):
            if index in self.aux_layers:
                column = self.aux_layers.index(index)
                feedback = layer.scratch.residual.float().mean(1)
                if self.packing is not None:
                    feedback = torch.where(self.packing.live[:, None], feedback, 0)
                self.auxiliary[:, column * 5120:(column + 1) * 5120].copy_(feedback)
            layer()
        samples = self.output_metadata.sample_indices
        residual = (self.output_scratch.residual if self.compact_head
                    else self.output_scratch.residual[samples]).float()
        premix = (self.output_scratch.pre_mix if self.compact_head
                  else self.output_scratch.pre_mix[samples])[:, :, None]
        collapsed = (residual * premix).sum(1).bfloat16()
        if self.scoring is not None:
            collapsed = (s.residual.float() * s.pre_mix[:, :, None]).sum(1).bfloat16()
            normalize(collapsed, self.weights.norm, self.scoring.source, self.weights.config.text["rms_norm_eps"])
            self.scoring()
            self.head_hidden.copy_(self.scoring.source[samples])
        else:
            normalize(collapsed, self.weights.norm, self.head_hidden, self.weights.config.text["rms_norm_eps"])
        self.head(self.head_hidden)
        if self.compact_head:
            s.hidden.copy_(self.head_hidden[samples])
        if self.mode != "verify":
            self.table_requests.commit(m.lengths)
        return s.hidden

    def commit(self, counts):
        if self.window_commit is None:
            raise ValueError("accepted decoder commits require verification staging")
        self.window_commit(counts)
        for layer in self.layers:
            layer.attention.commit_pending(counts)
        self.table_requests.commit(counts)
