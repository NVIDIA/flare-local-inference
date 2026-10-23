"""Learned GLM draft layer and target-feedback cache absorption."""

from copy import deepcopy
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x.attention.dsa_indexer import glm_pool
from flashinfer.triton.norm import rms_norm, rms_norm_add_residual

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.vocabulary import PreparedVocabularyProjection

from .indexer import IndexMetadata
from .mla import MLAScratch, PreparedGlmMLA
from .moe import MoeScratch, PreparedGlmMoe


@dataclass(frozen=True)
class MTPScratch:
    """Draft-layer intermediates that serial draft and absorption passes can share."""

    embedded: torch.Tensor
    joined: torch.Tensor
    residual: torch.Tensor
    normalized: torch.Tensor
    attention_output: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    attention: MLAScratch
    moe: MoeScratch | None

    @classmethod
    def allocate(cls, tokens, world_size, blocks, mode, device, *, moe=True, context_ranks=1, remote=False):
        def buffer(width):
            return torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
        return cls(buffer(4096), buffer(8192), buffer(4096), buffer(4096), buffer(4096), buffer(4096), buffer(4096),
                   MLAScratch.allocate(tokens, 64 // world_size, blocks, mode, device, context_ranks=context_ranks,
                                       remote=remote),
                   MoeScratch.allocate(tokens, world_size, device) if moe else None)

    @classmethod
    def absorbing(cls, decoder):
        """Views of decoder prefill scratch for cache absorption, which runs after the target prefill.

        Absorption reads only its own metadata and the separately owned target feedback, so the
        decoder's intermediates are dead by then. It writes no attention, MLP or hidden outputs.
        """
        tokens = decoder.embedding.shape[0]
        joined = decoder.residual.view(-1)[:tokens * 8192].view(tokens, 8192)
        return cls(decoder.embedding, joined, decoder.attention_output, decoder.mlp_output,
                   decoder.attention_output, decoder.mlp_output, decoder.embedding, decoder.mla, None)

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.hidden.shape[0]:
            raise ValueError("GLM draft rows exceed scratch capacity")
        views = {name: getattr(self, name)[:tokens] for name in (
            "embedded", "joined", "residual", "normalized", "attention_output", "mlp_output", "hidden")}
        return MTPScratch(**views, attention=self.attention.rows(tokens, mode),
                          moe=None if self.moe is None else self.moe.rows(tokens))


@triton.jit
def _draft_metadata(Lengths, Starts, Slots, Tokens, Hidden, Tables, StateRows, Rank, TokenIds, HiddenOut, Output,
                    NumTokens, QueryStart, StateSlots, Requests, Positions, BlockTable, TokenSlots,
                    token_stride_row, token_stride_column, hidden_stride_row, hidden_stride_column,
                    table_rows, state_stride, blocks,
                    REQUESTS: tl.constexpr, REQUEST_BLOCK: tl.constexpr, CAPACITY: tl.constexpr, ROWS: tl.constexpr,
                    SLOT_LIMIT: tl.constexpr, WORLD: tl.constexpr, HIDDEN: tl.constexpr, COLUMN_BLOCK: tl.constexpr):
    # Programs below ROWS place one draft row: its request, source token and hidden state, position and cache
    # slot. Later programs fill one request's output index, state slot, query start and page table.
    program = tl.program_id(0)
    request = tl.arange(0, REQUEST_BLOCK)
    lengths = tl.load(Lengths + request, mask=request < REQUESTS, other=0)
    ends = tl.cumsum(lengths, 0)
    total = tl.sum(lengths, 0)
    if program < ROWS:
        row = tl.minimum(tl.sum(((program >= ends) & (request < REQUESTS)).to(tl.int32), 0), REQUESTS - 1)
        row_length = tl.sum(tl.where(request == row, lengths, 0), 0)
        begin = tl.sum(tl.where(request == row, ends, 0), 0) - row_length
        local = program - begin
        active = (program < total) & (row_length > 0)
        source_row = tl.minimum(tl.maximum(row * CAPACITY + local, 0), SLOT_LIMIT - 1)
        source_request, source_column = source_row // CAPACITY, source_row % CAPACITY
        token = tl.load(Tokens + source_request * token_stride_row + source_column * token_stride_column)
        tl.store(TokenIds + program, token)
        channel = tl.arange(0, HIDDEN)
        value = tl.load(Hidden + source_request * hidden_stride_row + source_column * hidden_stride_column + channel)
        tl.store(HiddenOut + program * HIDDEN + channel, tl.where(active, value, 0.).to(value.dtype))
        tl.store(Requests + program, tl.where(active, row, 0).to(tl.int32))
        position = tl.load(Starts + row) + local
        tl.store(Positions + program, tl.where(active, position, -1))
        owner = (position // 4) % WORLD
        index = (position // (4 * WORLD)) * 4 + position % 4
        column = tl.minimum(tl.maximum(index // 256, 0), blocks - 1)
        slot_row = tl.minimum(tl.maximum(tl.load(Slots + row), 0), table_rows - 1)
        page = tl.load(Tables + slot_row * blocks + column).to(tl.int64)
        live = active & (position >= 0) & (page >= 0)
        if WORLD > 1:
            live &= owner == tl.load(Rank)
        tl.store(TokenSlots + program, tl.where(live, page * 256 + index % 256, -1))
    else:
        r = program - ROWS
        length = tl.sum(tl.where(request == r, lengths, 0), 0)
        end = tl.sum(tl.where(request == r, ends, 0), 0)
        tl.store(Output + r, tl.minimum(tl.maximum(end - 1, 0), ROWS - 1))
        slot_row = tl.minimum(tl.maximum(tl.load(Slots + r), 0), table_rows - 1)
        state = tl.load(StateRows + slot_row * state_stride)
        tl.store(StateSlots + r, tl.where(length > 0, state, -1))
        tl.store(QueryStart + 1 + r, end.to(tl.int32))
        if r == 0:
            tl.store(QueryStart, 0)
            tl.store(NumTokens, total.to(tl.int32))
        for start in tl.range(0, blocks, COLUMN_BLOCK):
            offsets = start + tl.arange(0, COLUMN_BLOCK)
            inside = offsets < blocks
            values = tl.load(Tables + slot_row * blocks + offsets, mask=inside)
            tl.store(BlockTable + r * blocks + offsets, values, mask=inside)


class DraftMetadata:
    def __init__(self, tokens, rows, blocks, device, *, shards=None):
        from .dcp import ContextShards

        self.shards = ContextShards() if shards is None else shards
        self.indexed = IndexMetadata.allocate(tokens, rows, blocks, device)
        self.token_ids = torch.zeros(tokens, dtype=torch.int64, device=device)
        self.hidden = torch.zeros((tokens, 4096), dtype=torch.bfloat16, device=device)
        self.indices = torch.arange(tokens, device=device)
        self.output_indices = torch.zeros(rows, dtype=torch.int64, device=device)
        self.num_tokens = torch.zeros(1, dtype=torch.int32, device=device)

    def set(self, slots, lengths, starts, tokens, hidden, tables, *, capacity, state_rows):
        """Place each request's draft rows in one kernel: tokens, hidden states, positions, slots and tables."""
        from .dcp import _launch

        if self.token_ids.device.type != "cuda":
            return self.set_reference(slots, lengths, starts, tokens, hidden, tables, capacity=capacity,
                                      state_rows=state_rows)
        m = self.indexed
        requests, rows = lengths.numel(), self.indices.numel()
        tokens = tokens.reshape(requests, -1)
        hidden = hidden.reshape(requests, -1, 4096)
        if (tokens.shape[1] != capacity or hidden.shape[1] != capacity or m.block_table.shape != (requests, tables.shape[1])
                or rows != requests * capacity or tables.stride(1) != 1 or hidden.stride(2) != 1):
            raise ValueError("draft metadata requires [requests, capacity] tokens, hidden states and page tables")
        rank = self.shards.index if self.shards.world > 1 else self.output_indices
        _launch(_draft_metadata, (rows + requests,), (lengths, starts, slots, tokens, hidden, tables, state_rows,
                rank, self.token_ids, self.hidden, self.output_indices, self.num_tokens, m.query_start_loc,
                m.state_slots, m.requests, m.positions, m.block_table, m.token_slots, tokens.stride(0),
                tokens.stride(1), hidden.stride(0), hidden.stride(1), tables.shape[0], state_rows.stride(0),
                tables.shape[1]), REQUESTS=requests, REQUEST_BLOCK=triton.next_power_of_2(requests), CAPACITY=capacity, ROWS=rows,
                SLOT_LIMIT=requests * capacity, WORLD=self.shards.world, HIDDEN=4096, COLUMN_BLOCK=1024, num_warps=4)

    def set_reference(self, slots, lengths, starts, tokens, hidden, tables, *, capacity, state_rows):
        m = self.indexed
        ends = lengths.cumsum(0)
        begins = ends - lengths
        self.output_indices.copy_((ends - 1).clamp(0, self.indices.numel() - 1))
        row = (self.indices[:, None] >= ends[None]).sum(1).clamp_max(lengths.numel() - 1)
        local = self.indices - begins[row]
        active = (self.indices < ends[-1]) & (lengths[row] > 0)
        source = (row * capacity + local).clamp(0, tokens.numel() - 1)
        self.token_ids.copy_(tokens.reshape(-1)[source])
        self.hidden.copy_(hidden.reshape(-1, 4096)[source])
        self.hidden.masked_fill_(~active[:, None], 0)
        self.num_tokens.copy_(ends[-1:])
        m.query_start_loc[0].zero_()
        m.query_start_loc[1:].copy_(ends)
        m.state_slots.copy_(torch.where(lengths > 0, state_rows[slots.long().clamp(0, tables.shape[0] - 1), 0], -1))
        m.requests.copy_(torch.where(active, row, 0))
        position = starts[row] + local
        m.positions.copy_(torch.where(active, position, -1))
        m.block_table.copy_(tables[slots.long().clamp(0, tables.shape[0] - 1)])
        m.token_slots.copy_(torch.where(active, self.shards.token_slots(position, m.block_table[row]), -1))


class PreparedMTP:
    """Shared embedding, normalized feedback, MLA, residual MoE and shared head.

    Cache absorption stops after input normalization, latent KV and C4 writes.
    The target's normalized output and the next token occupy the preceding
    target position in the draft cache. Position zero masks the token embedding.
    """

    def __init__(self, weights, state, metadata, *, mode, session, name, workspace,
                 runtime, vocabulary_runtime, coordinator, cache_only=False, max_cache_pages=None, scratch=None,
                 context=None):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention, Glm5NextTextMoE

        config = deepcopy(weights.tree.config.text_config)
        config.indexer_types = [*config.indexer_types, "full"]
        with torch.device("meta"):
            attention = Glm5NextTextAttention(config, 45).eval()
            moe = Glm5NextTextMoE(config).eval()
        self.metadata, self.weights, self.cache_only = metadata, weights.mtp, cache_only
        self.eps = config.rms_norm_eps
        rows, device = metadata.token_ids.numel(), metadata.hidden.device
        def buffer(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        if scratch is None:
            scratch = MTPScratch.allocate(rows, weights.shard.world_size, metadata.indexed.block_table.shape[1],
                                          mode, device, moe=not cache_only,
                                          context_ranks=1 if context is None else context.shards.world,
                                          remote=context is not None and context.remote)
        self.embedded, self.joined = scratch.embedded, scratch.joined
        self.residual, self.normalized = scratch.residual, scratch.normalized
        self.attention_output, self.mlp_output = scratch.attention_output, scratch.mlp_output
        self.hidden = scratch.hidden
        self.embedding = PreparedEmbedding(weights.tree.model.language_model.embed_tokens.weight,
            metadata.token_ids, self.embedded, num_rows=metadata.num_tokens, session=session, name=f"{name}.embedding")
        self.projection = UnquantizedProjection(self.joined, weights.mtp.tensors["eh_proj"], self.residual,
            session=session, name=f"{name}.eh_proj")
        self.attention = PreparedGlmMLA(attention, weights.mtp.attention, self.normalized, scratch.attention,
            state, metadata.indexed, mode=mode, session=session, name=f"{name}.mla", workspace=workspace,
            max_cache_pages=max_cache_pages, context=context)
        if not cache_only:
            self.attention_reduce = PreparedAllReduce(scratch.attention.output, self.attention_output,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.attention_reduce")
            self.moe = PreparedGlmMoe(moe, self.normalized, weights.mtp.mlp,
                scratch.moe, session=session, name=f"{name}.moe", workspace=workspace)
            self.mlp_reduce = PreparedAllReduce(self.moe.output, self.mlp_output,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.mlp_reduce")
            self.local_logits = buffer(rows, 154880 // weights.shard.world_size)
            self.logits = buffer(rows, 154880)
            head = weights.mtp.draft_head if weights.mtp.draft_head is not None else weights.head
            self.head = PreparedVocabularyProjection(self.hidden, head,
                self.local_logits, self.logits, runtime=vocabulary_runtime, coordinator=coordinator,
                session=session, name=f"{name}.lm_head")

    def rebind_pages(self):
        self.attention.rebind_pages()

    def __call__(self):
        w, m = self.weights.tensors, self.metadata
        self.embedding()
        self.embedded.masked_fill_((m.indexed.positions == 0)[:, None], 0)
        rms_norm(self.embedded, w["enorm"], self.joined[:, :4096], self.eps, round_before_weight=True)
        rms_norm(m.hidden, w["hnorm"], self.joined[:, 4096:], self.eps, round_before_weight=True)
        self.projection(self.joined)
        rms_norm(self.residual, w["input_layernorm"], self.normalized, self.eps, round_before_weight=True)
        if self.cache_only:
            a = self.attention
            a.projections["qkv_a_proj"](self.normalized)
            rms_norm(a.scratch.latent_input, a.module.kv_a_layernorm.weight, a.scratch.latent,
                self.eps, round_before_weight=True)
            a.writer()
            index = a.indexer
            index.scratch.normalized_key.copy_(index.module.k_norm(index.key_input))
            index.gate_projection(self.normalized)
            glm_pool.update_pools(*index.pool_arguments, plan=index.pool_plan)
            return
        self.attention_reduce(self.attention(self.normalized))
        rms_norm_add_residual(self.attention_output, self.residual, w["post_attention_layernorm"],
                              self.eps, x_out=self.normalized)
        self.mlp_reduce(self.moe(self.normalized))
        rms_norm_add_residual(self.mlp_output, self.residual, w["shared_head.norm"], self.eps, x_out=self.hidden)
        self.head(self.hidden)
        return self.hidden
