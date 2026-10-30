"""Canonical Qwen dense layers with prepared projections and persistent GPU caches."""

from copy import copy
from dataclasses import dataclass

import torch
import triton

from ignition.components.embedding import PreparedEmbedding
from ignition.components.gated_delta import GDNScratch, PreparedGDN
from ignition.components.linear import prepare_projection
from ignition.components.gdn_recovery import PreparedGDNCommit
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from ignition.components.vocabulary import PreparedVocabularyProjection

from .config import PAGE_TOKENS


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    gdn: GDNScratch
    quantized: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, device, *, buffers=None):
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        quantized_shape = (tokens * max(config.hidden, config.text["num_attention_heads"] * config.text["head_dim"], 6144),)
        quantized = (torch.zeros(quantized_shape, dtype=torch.uint8, device=device) if buffers is None
                     else buffers.zeros(quantized_shape, torch.uint8))
        return cls(zeros((tokens, config.hidden)), zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary)),
                   GDNScratch.allocate(tokens, device, hidden_size=config.hidden, buffers=buffers), quantized)

    def rows(self, tokens, rows):
        return Scratch(self.activations[:tokens], self.hidden[:rows], self.logits[:rows], self.gdn.rows(tokens), self.quantized)


class Projection(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, source):
        return self.operation(source)


def module_copy(module):
    result = copy(module)
    result._modules = module._modules.copy()
    return result


class RecurrentAttention(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, hidden_states, **kwargs):
        return self.operation(hidden_states)


class FullAttention(torch.nn.Module):
    def __init__(self, module, weights, source, metadata, pool, layer, *, mode, session, workspace, name, buffers,
                 projection_staging=None):
        super().__init__()
        self.canonical, self.metadata = module, metadata
        config = weights.config
        self.heads, self.kv_heads = config.text["num_attention_heads"], config.text["num_key_value_heads"]
        self.dim = config.text["head_dim"]
        tokens, device = source.shape[0], source.device

        def empty(field, width):
            if field not in buffers:
                buffers[field] = torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
            return buffers[field][:tokens]

        self.qg = empty("qg", 2 * self.heads * self.dim)
        self.k, self.v = (empty(field, self.kv_heads * self.dim) for field in ("k", "v"))
        self.q = empty("q", self.heads * self.dim).view(tokens, self.heads, self.dim)
        self.key = empty("key", self.kv_heads * self.dim).view(tokens, self.kv_heads, self.dim)
        self.attended, self.output = empty("attended", self.heads * self.dim), empty("output", config.hidden)
        prefix = f"layers.{layer}.self_attn."
        def projection(field, inputs, output):
            weight = weights.projections[prefix + field]
            if isinstance(weight, TensorFP8Weight):
                return TensorFP8Projection(inputs, weight, output, session=session, name=name + "." + field,
                                           staging=projection_staging)
            return prepare_projection(inputs, weight, output, session=session, name=name + "." + field,
                                      workspace=workspace)
        operands = (
            ("q_proj", source, self.qg), ("k_proj", source, self.k),
            ("v_proj", source, self.v), ("o_proj", self.attended, self.output))
        self.projections = {
            field: projection(field, inputs, output)
            for field, inputs, output in operands
        }
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.lengths = torch.zeros_like(metadata.lengths)
        key_cache, value_cache = pool.attention(layer, config)
        self.writer = PreparedPagedKVWriter(self.key, self.v.view(tokens, self.kv_heads, self.dim),
            key_cache, value_cache, self.slots, session=session, name=name + ".write")
        self.attention = PreparedPagedAttention(self.q, key_cache, value_cache, metadata.page_table,
            self.lengths, metadata.query_start_loc, self.attended.view(tokens, self.heads, self.dim),
            mode=mode, width=metadata.max_query_tokens if mode == "verify" else 1,
            session=session, workspace=workspace, name=name + ".attention")

    def write_cache(self, hidden_states, position_embeddings):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        for field in ("q_proj", "k_proj", "v_proj"):
            self.projections[field](hidden_states)
        query, gate = self.qg.view(-1, self.heads, 2 * self.dim).chunk(2, dim=-1)
        q, k = apply_rotary_pos_emb(self.canonical.q_norm(query),
            self.canonical.k_norm(self.k.view(-1, self.kv_heads, self.dim)), *position_embeddings)
        self.q.copy_(q)
        self.key.copy_(k)
        m = self.metadata
        page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
        self.slots.copy_(torch.where(m.positions >= 0, page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1))
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts + m.lengths, 0))
        self.writer()
        return gate

    def forward(self, hidden_states, position_embeddings, **kwargs):
        gate = self.write_cache(hidden_states, position_embeddings)
        self.attention()
        self.attended.mul_(gate.reshape_as(self.attended).sigmoid())
        return self.projections["o_proj"](self.attended), None


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, vocabulary_runtime,
                 coordinator, session, name, workspace, scoring=None, records=None, buffers=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.embedding = PreparedEmbedding(weights.tree.embed_tokens.weight, metadata.token_ids,
            scratch.activations, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), scratch.activations.device,
            verification_columns=metadata.max_query_tokens if mode == "verify" else 1)
        config, source = weights.config, scratch.activations
        def zeros(shape):
            return (torch.zeros(shape, dtype=source.dtype, device=source.device) if buffers is None
                    else buffers.zeros(shape, source.dtype))
        attention_buffers = {field: zeros((source.shape[0], width)) for field, width in (
            ("qg", 2 * config.text["num_attention_heads"] * config.text["head_dim"]),
            ("q", config.text["num_attention_heads"] * config.text["head_dim"]),
            ("attended", config.text["num_attention_heads"] * config.text["head_dim"]),
            ("k", config.text["num_key_value_heads"] * config.text["head_dim"]),
            ("v", config.text["num_key_value_heads"] * config.text["head_dim"]),
            ("key", config.text["num_key_value_heads"] * config.text["head_dim"]), ("output", config.hidden))}
        mlp_buffers = {field: zeros((source.shape[0], width))
                       for field, width in (("gate_proj", config.text["intermediate_size"]),
                                            ("up_proj", config.text["intermediate_size"]),
                                            ("down_proj", config.hidden))}
        mlp_input = zeros(mlp_buffers["gate_proj"].shape)
        self.layers = []
        for index, original in enumerate(weights.tree.layers):
            layer = module_copy(original)
            prefix = f"layers.{index}."
            if original.block_type == "linear_attention":
                inputs = (("in_proj_qkvz",) if prefix + "linear_attn.in_proj_qkvz" in weights.projections
                          else ("in_proj_qkv", "in_proj_z"))
                projections = {field: weights.projections[prefix + "linear_attn." + field]
                               for field in (*inputs, "in_proj_ba", "out_proj")}
                layer.linear_attn = RecurrentAttention(PreparedGDN(original.linear_attn, projections, source,
                    scratch.gdn, pool.recurrent(index), self.delta,
                    mode="decode" if mode == "verify" else mode, session=session,
                    records=records[index] if mode == "verify" else None,
                    name=name + "." + prefix + "gdn", workspace=workspace, projection_workspace=workspace,
                    projection_staging=scratch.quantized))
            else:
                layer.self_attn = FullAttention(original.self_attn, weights, source, metadata, pool, index,
                    mode=mode, session=session, workspace=workspace, name=name + "." + prefix + "attention",
                    buffers=attention_buffers, projection_staging=scratch.quantized)
            layer.mlp = module_copy(original.mlp)
            for field in ("gate_proj", "up_proj", "down_proj"):
                weight = weights.projections[prefix + "mlp." + field]
                inputs = source if field != "down_proj" else mlp_input
                output = mlp_buffers[field]
                setattr(layer.mlp, field, Projection(prepare_projection(inputs, weight, output,
                    activation_mode="auto", session=session, workspace=workspace, name=name + "." + prefix + field)))
            self.layers.append(layer)
        rotary = config.text["rope_parameters"]
        dim = int(config.text["head_dim"] * rotary["partial_rotary_factor"])
        self.inverse_frequency = 1.0 / (rotary["rope_theta"] ** (
            torch.arange(0, dim, 2, device=source.device, dtype=torch.float32) / dim))
        self.scoring = None if scoring is None else scoring.prepare(scratch.activations, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt", copy_source=True)
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto")
        self.auxiliary = (zeros((source.shape[0], config.hidden * len(config.draft.target_layers)))
                          if weights.draft is not None else None)
        self.replay = None
        if mode == "verify":
            self.commit_counts = torch.zeros(metadata.starts.numel(), dtype=torch.int32, device=source.device)
            states = {i: pool.region(f"layer{i}.recurrent") for i in records.layers}
            self.replay = PreparedGDNCommit(pool.backing, states, records, self.delta, self.commit_counts,
                                            key_heads=16, value_heads=48)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in records.layers]
            self.convolution_offsets = torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=source.device)
            self.convolution_shape = regions[0].shape

    def commit(self, counts):
        self.commit_counts.copy_(counts)
        self.replay()
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, self.metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self):
        m, d = self.metadata, self.delta
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        frequency = m.positions.float()[:, None] * self.inverse_frequency[None, :]
        embedding = torch.cat((frequency, frequency), dim=-1)
        position_embeddings = embedding.cos().bfloat16(), embedding.sin().bfloat16()
        hidden = self.embedding()
        taps = () if self.auxiliary is None else self.weights.config.draft.target_layers
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, position_embeddings=position_embeddings)
            if index in taps:
                column = taps.index(index) * hidden.shape[-1]
                self.auxiliary[:, column:column + hidden.shape[-1]].copy_(hidden)
        if self.scoring is not None:
            self.scoring.source.copy_(self.weights.tree.norm(hidden))
            self.scoring()
            self.scratch.hidden.copy_(self.scoring.source[self.metadata.sample_indices])
        else:
            self.scratch.hidden.copy_(self.weights.tree.norm(hidden[m.sample_indices]))
        self.head(self.scratch.hidden)
