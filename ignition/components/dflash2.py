"""GQA DFlash2 context injection and parallel block proposals."""

from dataclasses import dataclass, field

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, rotate_half

from ignition.components.candidate_selector import PreparedCandidateSelector
from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.linear import prepare_projection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.window_staging import PreparedWindowStaging

PAGE_TOKENS = 128


def _norm(weight, eps):
    with torch.device("meta"):
        module = Qwen3RMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


def _inverse(geometry, device):
    return 1.0 / geometry.theta ** (
        torch.arange(0, geometry.rotary_dim, 2, dtype=torch.float32, device=device) / geometry.rotary_dim
    )


def _rotary(values, positions, inverse):
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    angles = torch.cat((frequencies, frequencies), -1)
    cos, sin = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
    return values * cos[:, None] + rotate_half(values) * sin[:, None]


class PreparedContext:
    """Write accepted target features into independent draft KV rings.

    The supplied cursor is dedicated to context scratch. Each graph variant forks
    its starting position; target auxiliary features must occupy disjoint storage.
    """

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 runtime, coordinator, buffers):
        self.metadata = metadata
        c, a = weights.config.draft, weights.config.draft.attention
        self.window = a.window
        tokens, hidden = auxiliary.shape[0], c.text["hidden_size"]
        scratch = buffers.fork()
        self.local = scratch.zeros((tokens, hidden // weights.shard.world_size), torch.bfloat16)
        self.projected = scratch.zeros((tokens, hidden), torch.bfloat16)
        _, kv_heads = a.local(weights.shard)
        self.kv = scratch.zeros((tokens, c.layers * kv_heads * (a.key_dim + a.value_dim)), torch.bfloat16)
        self.key = scratch.zeros((tokens, kv_heads, a.key_dim), torch.bfloat16)
        self.value = scratch.zeros((tokens, kv_heads, a.value_dim), torch.bfloat16)
        self.slots = scratch.full((tokens,), -1, torch.int64)
        self.inverse = _inverse(a, auxiliary.device)
        self.project = prepare_projection(auxiliary, weights.draft.projection, self.local,
            session=session, name=name + ".project", workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".gather")
        self.norm = _norm(weights.draft.hidden_norm, c.text["rms_norm_eps"])
        self.project_kv = prepare_projection(self.projected, weights.draft.context_kv, self.kv,
            session=session, name=name + ".kv", workspace=workspace)
        self.layers = []
        for index, layer in enumerate(weights.draft.layers):
            norm = _norm(layer.key_norm, c.text["rms_norm_eps"])
            writer = PreparedPagedKVWriter(self.key, self.value,
                pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value"),
                self.slots, session=session, name=f"{name}.{index}.write")
            self.layers.append((norm, writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        valid = (m.positions >= 0) & (offset >= 0) & (offset < count)
        valid &= offset >= (count - self.window).clamp_min(0)
        self.slots.copy_(torch.where(valid,
            m.state_rows[m.requests] * self.window + m.positions.remainder(self.window), -1))
        self.projected.copy_(self.norm(self.gather(self.project(auxiliary))))
        self.project_kv(self.projected)
        width = self.key.shape[1] * self.key.shape[2]
        chunks = self.kv.view(self.kv.shape[0], len(self.layers), 2 * width)
        for index, (norm, writer) in enumerate(self.layers):
            key = norm(chunks[:, index, :width].view_as(self.key))
            self.key.copy_(_rotary(key, m.positions, self.inverse))
            self.value.copy_(chunks[:, index, width:].view_as(self.value))
            writer()

    def scratch_tensors(self):
        return self.local, self.projected, self.kv, self.key, self.value, self.slots


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    gathered_logits: torch.Tensor
    logits: torch.Tensor
    convolution: dict
    retained: list = field(default_factory=list)
    selectors: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, buffers):
        if tokens <= 0 or tokens % config.block_size:
            raise ValueError("DFlash2 scratch requires complete proposal blocks")
        s = buffers.fork()
        a, c = config.attention, config.text
        heads, kv_heads = a.local(shard)
        rows = tokens // config.block_size
        pages = rows * ((a.window + config.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS)
        hidden = c["hidden_size"]
        vocab_width = ((c["vocab_size"] + shard.world_size - 1) // shard.world_size + 63) // 64 * 64
        def zeros(*shape, dtype=torch.bfloat16):
            return s.zeros(shape, dtype)
        convolution = dict(coefficients=zeros(tokens, 2 * 2 * hidden // 16),
                           before=zeros(tokens, hidden), after=zeros(tokens, hidden))
        return cls(zeros(tokens, hidden // shard.world_size), zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(tokens, heads * a.key_dim + kv_heads * (a.key_dim + a.value_dim)),
            zeros(tokens, heads, a.key_dim), zeros(tokens, kv_heads, a.key_dim),
            zeros(tokens, kv_heads, a.value_dim), zeros(tokens, heads, a.value_dim),
            zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(pages, PAGE_TOKENS, kv_heads, a.key_dim, dtype=torch.float8_e4m3fn),
            zeros(pages, PAGE_TOKENS, kv_heads, a.value_dim, dtype=torch.float8_e4m3fn),
            MLPScratch.allocate(tokens, hidden, c["intermediate_size"] // shard.world_size, device, buffers=s),
            zeros(tokens, hidden), zeros(tokens, vocab_width), zeros(tokens, vocab_width * shard.world_size),
            zeros(tokens, c["vocab_size"], dtype=torch.float32), convolution)

    def rows(self, tokens, samples):
        if (not 0 < tokens <= self.embedding.shape[0] or samples != tokens
                or tokens % 8):
            raise ValueError("DFlash2 scratch requires complete eight-token proposal blocks")
        return type(self)(*(getattr(self, name) if name in
            ("window_key", "window_value", "convolution", "retained", "selectors") else self.mlp.rows(tokens) if name == "mlp"
            else getattr(self, name)[:tokens] for name in self.__dataclass_fields__))

    def scratch_tensors(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__ if name != "selectors")


class DraftAttention:
    """Noncausal GQA over accepted context followed by a private eight-token block."""

    def __init__(self, index, weights, source, config, shard, scratch, metadata, pool, *,
                 runtime, coordinator, session, name, workspace, buffers):
        self.scratch, self.metadata = scratch, metadata
        a, self.window = config.attention, config.attention.window
        self.qkv = prepare_projection(source, weights.qkv, scratch.qkv, session=session, name=name + ".qkv", workspace=workspace)
        self.q_norm = _norm(weights.query_norm, config.text["rms_norm_eps"])
        self.k_norm = _norm(weights.key_norm, config.text["rms_norm_eps"])
        self.inverse = _inverse(a, source.device)
        self.history_key, self.history_value = (pool.reserved_region(f"draft{index}.{kind}")
                                               for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (a.window + config.block_size + PAGE_TOKENS - 1) // PAGE_TOKENS
        self.staged_key, self.staged_value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.staging = PreparedWindowStaging(self.history_key, self.history_value, scratch.key, scratch.value,
            self.staged_key, self.staged_value, metadata, session=session, name=name + ".staging")
        self.pages = torch.arange(rows * count, device=source.device, dtype=torch.int32).view(rows, count)
        self.lengths = buffers.zeros(metadata.lengths.shape, metadata.lengths.dtype)
        self.attention = PreparedPagedAttention(scratch.query, self.staged_key, self.staged_value,
            self.pages, self.lengths, metadata.query_start_loc, scratch.attended,
            mode="verify", width=config.block_size, window=a.window - 1, causal=False,
            session=session, workspace=workspace, name=name + ".attention")
        self.output = prepare_projection(scratch.attended.flatten(1), weights.output, scratch.partial,
            session=session, name=name + ".output", workspace=workspace)
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce")

    def __call__(self, source):
        s, m = self.scratch, self.metadata
        q, k, v = self.qkv(source).split((s.query[0].numel(), s.key[0].numel(), s.value[0].numel()), -1)
        s.query.copy_(_rotary(self.q_norm(q.view_as(s.query)), m.positions, self.inverse))
        s.key.copy_(_rotary(self.k_norm(k.view_as(s.key)), m.positions, self.inverse))
        s.value.copy_(v.view_as(s.value))
        self.staging()
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts.clamp(max=self.window) + m.lengths, 0).int())
        self.attention()
        return self.reduce(self.output(s.attended.flatten(1)))


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace, buffers, embedding, head,
                 output_multiplier, final_logit_softcapping):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c, s = weights.config.draft, scratch
        stages = buffers.fork()
        self.embedding = PreparedShardedEmbedding(embedding, metadata.token_ids, s.local_embedding, s.embedding,
            runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".embedding",
            num_rows=metadata.live_counts[:1])
        self.layers = []
        for index, layer in enumerate(weights.draft.layers):
            prefix = f"{name}.layer{index}"
            conv = {kind: PreparedGroupedConvolution(s.normalized, getattr(layer, kind), block_size=c.block_size,
                group_size=16, session=session, name=prefix + "." + kind, buffers=s.convolution,
                token_capacity=s.convolution["before"].shape[0]) for kind in ("attention_conv", "mlp_conv")}
            attention = DraftAttention(index, layer, conv["attention_conv"].before, c, weights.shard, s, metadata, pool,
                runtime=runtime, coordinator=coordinator, session=session, name=prefix + ".attention",
                workspace=workspace, buffers=stages)
            mlp = PreparedMLP(conv["mlp_conv"].before, layer.mlp, s.mlp.gate_up, s.mlp.intermediate, s.mlp.output,
                session=session, name=prefix + ".mlp", workspace=workspace)
            reduction = PreparedAllReduce(s.mlp.output, s.output, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp.reduce")
            self.layers.append((_norm(layer.input_norm, c.text["rms_norm_eps"]), conv["attention_conv"], attention,
                _norm(layer.post_attention_norm, c.text["rms_norm_eps"]), conv["mlp_conv"], mlp, reduction))
        self.norm = _norm(weights.draft.norm, c.text["rms_norm_eps"])
        self.head = PreparedVocabularyProjection(s.hidden, head, s.local_logits, s.gathered_logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            activation_mode="a16", workspace=workspace)
        selector_key = (s.hidden.shape[0], id(weights.draft.selector))
        if selector_key not in s.selectors:
            selector = PreparedCandidateSelector(s.hidden.view(-1, c.block_size, s.hidden.shape[-1]),
                weights.draft.selector, top_k=c.text["dflash_config"]["selector_top_k"], session=session,
                name=name + ".selector")
            s.selectors[selector_key] = selector
            s.retained.append(selector.projected)
        self.selector = s.selectors[selector_key]
        self.multiplier = output_multiplier
        self.softcap = final_logit_softcapping

    def __call__(self):
        s, c = self.scratch, self.weights.config.draft
        self.embedding()
        for norm, conv, attention, post_norm, mlp_conv, mlp, reduce in self.layers:
            s.normalized.copy_(norm(s.embedding))
            s.embedding.add_(conv.finish(attention(conv.prepare(s.normalized))))
            s.normalized.copy_(post_norm(s.embedding))
            s.embedding.add_(mlp_conv.finish(reduce(mlp(mlp_conv.prepare(s.normalized)))))
        s.hidden.copy_(self.norm(s.embedding[self.metadata.sample_indices]))
        self.head(s.hidden)
        # The unary transform consumes the head result in FP32.
        unary = s.gathered_logits[:, :c.text["vocab_size"]].float() * self.multiplier
        s.logits.copy_(unary if self.softcap is None else (unary / self.softcap).tanh() * self.softcap)
        self.selector.prepare(s.hidden.view(-1, c.block_size, s.hidden.shape[-1]))

    def scratch_tensors(self):
        return self.scratch.scratch_tensors(), self.selector.projected

