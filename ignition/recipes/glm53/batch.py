"""Fixed native uploads and captured request metadata for GLM text forwards."""

from dataclasses import dataclass

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import ERROR, RUNNING, DeviceRequestState
from ignition.components.token_state import PreparedTokenState

from .state import DecoderMetadata


@dataclass(frozen=True)
class BatchInputs:
    request_state_rows: torch.Tensor
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    live_rows: torch.Tensor
    query_lengths: torch.Tensor
    token_ids: torch.Tensor
    page_tables: torch.Tensor
    sample_enabled: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, device, *, capacity=4):
        if any(type(value) is not int or value <= 0 for value in (tokens, blocks)):
            raise ValueError("GLM upload capacities require positive token and page counts")
        return cls(torch.zeros((capacity, 8), dtype=torch.int64, device=device),torch.full((capacity,), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device), torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device), torch.zeros(tokens, dtype=torch.int64, device=device),
            torch.full((capacity, blocks), -1, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PreparedBatch:
    """Map request slots to packed KDA and sparse-MLA metadata inside a graph.

    Rust uploads prefill spans or one decode token per packed request to every
    rank. Rank zero checks decode tokens against its pending GPU samples; peers
    consume the same uploads without sampling. Validation precedes persistent
    model-state writes. Invalid or stale rows contribute no tokens or cache
    addresses. Recipes compile these tensor operations together with forward.
    """

    def __init__(self, inputs: BatchInputs, requests: DeviceRequestState, metadata: DecoderMetadata,
                 *, mode, shard: TensorParallelShard, pages, shards=None, validate_prefill=None):
        rows, tokens = metadata.sample_indices.numel(), metadata.token_ids.numel()
        self.capacity = requests.capacity
        if mode not in ("prefill", "decode") or not 1 <= rows <= self.capacity:
            raise ValueError("GLM batches require prefill/decode and rows within request capacity")
        if (mode == "prefill" and rows != 1) or (mode == "decode" and tokens != rows):
            raise ValueError("GLM prefill has one sequence; decode has one token per row")
        if tokens > inputs.token_ids.numel() or inputs.page_tables.shape[1] != metadata.indexed.block_table.shape[1]:
            raise ValueError("GLM upload and metadata capacities differ")
        if not isinstance(pages, torch.Tensor) or pages.shape != (1,) or pages.dtype != torch.int64 or requests.vocab_size != 154880:
            raise ValueError("GLM batches require positive cache capacity and a 154880-token vocabulary")
        self.inputs, self.requests, self.metadata = inputs, requests, metadata
        self.rows, self.tokens, self.pages, self.mode, self.shard = rows, tokens, pages, mode, shard
        from .dcp import ContextShards

        self.shards = ContextShards() if shards is None else shards
        self.validate_prefill = validate_prefill
        self.accept_unsampled = shard.rank != 0
        device = metadata.token_ids.device
        self.row_ids = torch.arange(rows, device=device)
        self.slot_ids = torch.arange(self.capacity, device=device)
        self.token_offsets = torch.arange(tokens, device=device)
        self.block_ids = torch.arange(inputs.page_tables.shape[1], device=device)
        self.validation_input = torch.zeros((1, max(8, 2 * self.capacity)), dtype=torch.bfloat16, device=device)
        self.validation_sum = torch.zeros_like(self.validation_input)
        self.present = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        self.token_state = PreparedTokenState(requests, metadata.token_ids, metadata.recurrent.query_start_loc,
            inputs.row_slot_ids[:rows], inputs.row_generations[:rows], inputs.live_rows)
        self.token_state.prepare()

    def validate(self):
        b, r = self.inputs, self.requests
        slots = b.row_slot_ids[:self.rows].long()
        safe_slots = slots.clamp(0, self.capacity - 1)
        resident = (self.row_ids < b.live_rows) & (slots >= 0) & (slots < self.capacity)
        resident &= b.row_generations[:self.rows] == r.generation[safe_slots]
        duplicate = ((slots[:, None] == slots[None, :]) & resident[:, None] & resident[None, :]).sum(1) > 1
        lengths = b.query_lengths[:self.rows] if self.mode == "prefill" else torch.ones_like(safe_slots)
        positions = r.consumed_tokens[safe_slots]
        ends = positions + lengths
        valid = ~duplicate & (b.live_rows <= self.rows) & (lengths > 0) & (lengths <= self.tokens)
        page_tokens = self.shards.page_tokens
        valid &= (positions >= 0) & (ends <= b.page_tables.shape[1] * page_tokens)
        pages = b.page_tables[safe_slots]
        valid &= (((pages >= 0) & (pages < self.pages))
                  | (self.block_ids >= (ends[:, None] + page_tokens - 1) // page_tokens)).all(1)
        pending = r.pending_token[safe_slots]
        uploaded = b.token_ids[:self.tokens]
        if self.mode == "prefill":
            invalid_tokens = ((uploaded < 0) | (uploaded >= r.vocab_size)) & (self.token_offsets < lengths[0])
            valid &= (pending < 0) & ~invalid_tokens.any()
            if self.validate_prefill is not None:
                valid &= self.validate_prefill(uploaded, lengths)
        else:
            valid &= (uploaded >= 0) & (uploaded < r.vocab_size)
            valid &= (pending == uploaded) | (self.accept_unsampled & (pending < 0))
        present = resident & (r.status[safe_slots] == RUNNING)
        bad = present & ~valid
        bad_slots = ((self.slot_ids[:, None] == slots[None, :]) & bad[None, :]).any(1)
        self.present.copy_(((self.slot_ids[:, None] == slots[None, :]) & present[None, :]).any(1))
        self.validation_input[0, :self.capacity].copy_(bad_slots)
        self.validation_input[0, self.capacity:2 * self.capacity].copy_(self.present)
        return self.validation_input

    def begin(self, validation_sum):
        participants = validation_sum[0, self.capacity:2 * self.capacity]
        mismatch = (participants > 0) & (participants < self.shard.world_size)
        bad_slots = self.present & ((validation_sum[0, :self.capacity] > 0) | mismatch)
        self.requests.status.copy_(torch.where(bad_slots, ERROR, self.requests.status))
        self.requests.error_code.copy_(torch.where(bad_slots, 32, self.requests.error_code))
        self.token_state.gather()
        b, m, t = self.inputs, self.metadata, self.token_state
        g, index = m.recurrent, m.indexed
        slots = b.row_slot_ids[:self.rows].long()
        active = t.active.bool()
        lengths = b.query_lengths[:self.rows] if self.mode == "prefill" else torch.ones_like(b.query_lengths[:self.rows])
        lengths = torch.where(active, lengths, 0)
        ends = lengths.cumsum(0, dtype=torch.int32)
        starts = ends - lengths
        g.query_start_loc[0].zero_()
        g.query_start_loc[1:].copy_(ends)
        state = b.request_state_rows[slots.clamp(0, self.capacity - 1), 0]
        g.convolution_slots.copy_(torch.where(active, state, -1))
        g.fresh.copy_(active & (t.positions == 0))
        g.accepted_tokens.fill_(1)
        g.state_indices[:, 0].copy_(torch.where(active, state, 0))
        g.initial_state_indices.copy_(torch.where(active & (t.positions > 0), state, 0))
        g.final_state_indices.copy_(torch.where(active, state, 0))
        g.checkpoint_state_indices.zero_()
        g.checkpoint_offsets.zero_()
        g.num_seqs.copy_(b.live_rows.clamp(0, self.rows))
        g.num_tokens.copy_(ends[-1:])
        m.sample_indices.copy_((ends - 1).clamp(min=0))
        index.query_start_loc.copy_(g.query_start_loc)
        index.state_slots.copy_(g.convolution_slots)
        index.block_table.copy_(torch.where(active[:, None], b.page_tables[slots.clamp(0, self.capacity - 1)], -1))
        request = (self.token_offsets[:, None] >= ends[None, :]).sum(1).clamp(max=self.rows - 1)
        live_token = self.token_offsets < ends[-1]
        position = t.positions[request] + self.token_offsets - starts[request]
        index.requests.copy_(torch.where(live_token, request, 0))
        index.positions.copy_(torch.where(live_token, position, -1))
        index.token_slots.copy_(torch.where(live_token,
            self.shards.token_slots(position, index.block_table[request]), -1))
        tokens = b.token_ids[:self.tokens] if self.mode == "prefill" else b.token_ids[request]
        m.token_ids.copy_(torch.where(live_token, tokens, 0))

    def finish(self):
        self.token_state.consume()
