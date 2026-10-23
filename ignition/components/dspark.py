"""Captured DSpark proposals, packed target verification, and accepted-state commits."""

from dataclasses import dataclass

import torch

from ignition.components.speculation import SpeculativeRngState

from .collectives import PreparedAllGather
from .compressed_mla import PreparedCompressedCacheWriter
from .embedding import PreparedEmbedding
from .linear import QuantizedProjection, UnquantizedProjection
from .rms_norm import PreparedRMSNorm
from .rotary import CompressedRotary as Rotary


@dataclass(frozen=True)
class DSparkForwards:
    decoder: type
    scratch: type
    context: type
    draft: type
    markov: type


class SpeculativePreparation:
    def __init__(
        self,
        weights,
        tables,
        pool,
        requests,
        inputs,
        scratch,
        *,
        draft_tokens,
        completion,
        frame,
        sequence,
        session,
        workspace,
        runtime,
        vocabulary_runtime,
        markov_runtime,
        coordinator,
        forwards,
        page_tokens=128,
        result_runtime=None,
        buffers=None,
    ):
        self.forwards = forwards
        self.page_tokens = page_tokens
        self.weights, self.tables, self.pool = weights, tables, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = {
            "runtime": runtime, "vocabulary_runtime": vocabulary_runtime, "coordinator": coordinator
        }
        self.markov_runtime = markov_runtime
        # Only sampling uses the result channel; model forwards receive the collectives above.
        self.result_runtime = result_runtime
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation = torch.zeros_like(requests.generation)
        self.offset = torch.zeros_like(requests.generation)
        draft = weights.config.draft
        self.logit_offset = 0
        self.block_size = max(1, draft_tokens)
        tokens = requests.capacity * self.block_size
        self.draft_scratch = forwards.scratch.allocate(
            tokens,
            tokens,
            weights.shard.world_size,
            pool.backing.device,
            experts=draft.experts,
            top_k=draft.top_k,
            buffers=buffers,
        )


class PreparedFeedbackCapture:
    """Retain target-layer stream means needed by the draft context window.

    Prefill accepts every valid input row, so one request can retain only its
    final 128 rows before reducing the hyperconnection streams. Verification
    retains every prepared row until acceptance is known.
    """

    def __init__(self, metadata, columns, buffers, *, compact_prefill=False):
        self.metadata = metadata
        rows = metadata.positions.numel()
        compact = compact_prefill and columns > 0 and metadata.starts.numel() == 1 and rows > 128
        self.offsets = self.indices = self.live = None
        if compact:
            rows = 128
            self.offsets = torch.arange(rows, device=metadata.positions.device)
            self.indices = buffers.zeros((rows,), torch.int64)
            self.live = buffers.zeros((rows,), torch.bool)
        self.output = buffers.zeros((rows, columns), torch.bfloat16)

    def begin(self):
        if self.offsets is not None:
            m = self.metadata
            count = m.lengths[0].long()
            kept = count.clamp(max=128)
            self.indices.copy_((m.query_start_loc[0] + count - kept + self.offsets).clamp(
                0, m.positions.numel() - 1))
            self.live.copy_((self.offsets < kept) & (m.positions[self.indices] >= 0))

    def __call__(self, residual, column):
        if self.indices is not None:
            values = residual[self.indices].float().mean(1)
            values = torch.where(self.live[:, None], values, 0)
        else:
            values = residual.float().mean(1)
        self.output[:, column:column + residual.shape[-1]].copy_(values)


class PreparedContext:
    """Project accepted target feedback into each draft stage's sliding window.

    Counts describe accepted prefixes of the metadata's packed rows. With
    compact_prefill, capacities above 128 rows for one request consume an
    already captured 128-row suffix. Packed verification keeps its prepared rows.
    """

    def __init__(
        self, weights, auxiliary, metadata, pool, *, session, name, workspace, cache_format="deepseek_v41",
        compact_prefill=False,
    ):
        self.metadata, self.pool = metadata, pool
        tokens, device = auxiliary.shape[0], auxiliary.device
        draft = weights.draft
        self.prefill_positions = self.prefill_offsets = None
        if compact_prefill and metadata.starts.numel() == 1 and metadata.positions.numel() > 128:
            # Only the accepted suffix can survive in the draft's 128-token ring.
            if tokens != 128:
                raise ValueError("compact prefill feedback must contain the retained 128-row suffix")
            self.prefill_positions = torch.zeros(tokens, dtype=metadata.positions.dtype, device=device)
            self.prefill_offsets = torch.arange(tokens, device=device)
        self.projected = torch.zeros(
            (tokens, weights.config.text["hidden_size"]), dtype=torch.bfloat16, device=device
        )
        self.main = torch.zeros_like(self.projected)
        self.kv = torch.zeros((tokens, 512 * len(draft.layers)), dtype=torch.bfloat16, device=device)
        self.normalized = torch.zeros((tokens, 512), dtype=torch.bfloat16, device=device)
        self.slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.project = QuantizedProjection(
            auxiliary,
            draft.main_projection,
            self.projected,
            session=session,
            name=name + ".main",
            workspace=workspace,
            activation_mode="auto",
        )
        self.norm = PreparedRMSNorm(
            self.projected,
            draft.main_norm,
            self.main,
            eps=weights.config.text["rms_norm_eps"],
            session=session,
            name=name + ".main_norm",
        )
        self.context_projection = QuantizedProjection(
            self.main,
            draft.context_kv,
            self.kv,
            session=session,
            name=name + ".kv",
            workspace=workspace,
            activation_mode="auto",
        )
        self.rotary = Rotary(weights.config, device)
        self.stages = []
        for index, layer in enumerate(draft.layers):
            part = self.kv[:, index * 512:(index + 1) * 512]
            norm = PreparedRMSNorm(
                part,
                layer.attention["kv_norm"],
                self.normalized,
                eps=weights.config.text["rms_norm_eps"],
                session=session,
                name=f"{name}.{index}.norm",
            )
            writer = PreparedCompressedCacheWriter(
                self.normalized,
                pool.window(index, draft=True),
                self.slots,
                page_size=128,
                kind="swa",
                session=session,
                name=f"{name}.{index}.write",
                cache_format=cache_format,
            )
            self.stages.append((part, norm, writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        if self.prefill_offsets is not None:
            count = counts[0].long()
            kept = count.clamp(max=128)
            indices = (m.query_start_loc[0] + count - kept + self.prefill_offsets).clamp(0, m.positions.numel() - 1)
            positions = m.positions[indices]
            live = (self.prefill_offsets < kept) & (positions >= 0)
            self.prefill_positions.copy_(torch.where(live, positions, -1))
            self.slots.copy_(torch.where(live, m.state_rows[0] * 128 + positions % 128, -1))
            positions = self.prefill_positions
        else:
            positions = m.positions
            offset = positions - m.starts[m.requests]
            count = counts[m.requests]
            live = (positions >= 0) & (offset < count) & (offset >= (count - 128).clamp_min(0))
            self.slots.copy_(torch.where(live, m.state_rows[m.requests] * 128 + positions % 128, -1))
        self.norm(self.project(auxiliary))
        self.context_projection(self.main)
        for part, norm, writer in self.stages:
            norm(part)
            self.rotary.apply(self.normalized, positions.clamp_min(0), compressed=False)
            writer(self.normalized)


class PreparedMarkov:
    """Project FP32 transition logits from replicated or vocabulary-sharded weights.

    A replicated head runs independently on each rank without a collective.
    A sharded head uses the supplied runtime to gather its FP32 bits unchanged.
    Quantized A16 projections produce BF16 logits, widened before bias addition.
    """

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name):
        rows, device = tokens.numel(), tokens.device
        self.embedding = torch.zeros(
            (rows, weights.config.draft.markov_rank), dtype=torch.bfloat16, device=device
        )
        head = weights.draft.markov_head
        columns = head.shape[0] if isinstance(head, torch.Tensor) else head.out_features
        expected_columns = 129280 if runtime is None else 129280 // weights.shard.world_size
        if columns != expected_columns:
            raise ValueError("Markov head vocabulary partition must match its gather runtime")
        self.local = torch.zeros((rows, columns), dtype=torch.float32, device=device)
        self.projected = self.local if isinstance(head, torch.Tensor) else torch.empty(
            (rows, columns), dtype=torch.bfloat16, device=device)
        self.logits = self.local if runtime is None else torch.zeros((rows, 129280), dtype=torch.float32, device=device)
        self.embed = PreparedEmbedding(
            weights.draft.markov_embedding, tokens, self.embedding, session=session, name=name + ".embed"
        )
        projection = UnquantizedProjection if isinstance(head, torch.Tensor) else QuantizedProjection
        options = {} if isinstance(head, torch.Tensor) else {"activation_mode": "a16"}
        self.projection = projection(self.embedding, head, self.projected,
                                     session=session, name=name + ".head", **options)
        self.gather = None if runtime is None else PreparedAllGather(
            self.local.view(torch.bfloat16),
            self.logits.view(torch.bfloat16),
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".gather",
        )

    def __call__(self):
        self.projection(self.embed())
        if self.projected is not self.local:
            self.local.copy_(self.projected)
        if self.gather is not None:
            self.gather(self.local.view(torch.bfloat16))
        return self.logits
