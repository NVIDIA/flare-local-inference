"""GLM C4 index projections, sequence pooling and sparse physical selection."""

from dataclasses import dataclass

import torch
from b12x.attention import dsa_indexer, sparse_mla
from b12x.attention.dsa_indexer import glm_pool
from b12x.preparation import PreparedCall

from ignition.components import sm103
from ignition.components.indexer import PreparedSparseIndexer
from ignition.components.indexer_sm103 import Sm103SparseIndexer
from ignition.components.linear import prepare_projection, UnquantizedProjection
from ignition.preparation.tuning import declare


_SELECT_ROWS = 512


@dataclass(frozen=True)
class IndexState:
    cache: torch.Tensor
    tail: torch.Tensor

    @classmethod
    def allocate(cls, pages, slots, device, *, ring=4):
        if pages <= 0 or slots <= 0 or ring < 4 or ring % 4:
            raise ValueError("GLM index state requires positive pages/slots and a ring divisible by four")
        return cls(torch.zeros((pages, 64, 132), dtype=torch.uint8, device=device),
                   torch.zeros((slots, 2, ring, 128), dtype=torch.bfloat16, device=device))


@dataclass(frozen=True)
class IndexMetadata:
    positions: torch.Tensor
    token_slots: torch.Tensor
    requests: torch.Tensor
    state_slots: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor

    @classmethod
    def allocate(cls, tokens, sequences, blocks, device):
        if min(tokens, sequences, blocks) <= 0:
            raise ValueError("GLM index metadata requires positive capacities")
        return cls(torch.full((tokens,), -1, dtype=torch.int64, device=device),
            torch.full((tokens,), -1, dtype=torch.int64, device=device),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            torch.zeros(sequences + 1, dtype=torch.int32, device=device),
            torch.full((sequences, blocks), -1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class IndexScratch:
    query: torch.Tensor
    query_fp8: torch.Tensor
    query_scales: torch.Tensor
    normalized_key: torch.Tensor
    gate: torch.Tensor
    weights: torch.Tensor
    pool_lengths: torch.Tensor
    pools: torch.Tensor
    physical: torch.Tensor
    selected_lengths: torch.Tensor
    page_table: torch.Tensor
    # Decode context parallel selection: local scores, packed and gathered candidates, global pools.
    scores: torch.Tensor
    candidates: torch.Tensor
    gathered: torch.Tensor
    global_pools: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, mode, device, *, context_ranks=1):
        if tokens <= 0 or blocks <= 0 or mode not in ("decode", "prefill"):
            raise ValueError("GLM index scratch requires positive capacities and decode/prefill mode")
        def tensor(width, dtype=torch.bfloat16):
            return torch.zeros((tokens, width), dtype=dtype, device=device)
        return cls(tensor(4096), tensor(4096, torch.float8_e4m3fn), tensor(32, torch.float32),
            tensor(128), tensor(128), tensor(32, torch.float32),
            torch.zeros(tokens, dtype=torch.int32, device=device),
            tensor(512, torch.int32),
            tensor(2051, torch.int32), torch.zeros(tokens, dtype=torch.int32, device=device),
            torch.zeros((tokens if mode == "decode" else 1, blocks), dtype=torch.int32, device=device),
            *cls._context_selection(tokens if context_ranks > 1 else 0, context_ranks, device))

    @staticmethod
    def _context_selection(tokens, ranks, device):
        return (torch.zeros((tokens, 512), dtype=torch.float32, device=device),
                torch.zeros((tokens, 512, 2), dtype=torch.float32, device=device),
                torch.zeros((tokens, ranks * 512, 2), dtype=torch.float32, device=device),
                torch.full((tokens, 512), -1, dtype=torch.int32, device=device))

    def rows(self, tokens, mode):
        if type(tokens) is not int or not 0 < tokens <= self.query.shape[0] or mode not in ("decode", "prefill"):
            raise ValueError("GLM index rows require valid capacity and decode/prefill mode")
        sequences = tokens if mode == "decode" else 1
        if sequences > self.page_table.shape[0]:
            raise ValueError("GLM index request rows exceed page-table capacity")
        return IndexScratch(self.query[:tokens], self.query_fp8[:tokens], self.query_scales[:tokens],
            self.normalized_key[:tokens], self.gate[:tokens], self.weights[:tokens], self.pool_lengths[:tokens],
            self.pools[:tokens], self.physical[:tokens], self.selected_lengths[:tokens],
            self.page_table[:sequences], self.scores[:tokens], self.candidates[:tokens], self.gathered[:tokens],
            self.global_pools[:tokens])


class PreparedGlmIndexer:
    """Replicated C4 selection for GLM sparse MLA with 256-token model pages.

    Position -1 masks an inactive query; negative state slots mask inactive
    requests. Token slots and page tables address the same physical model pages.
    Prefill admits one request, while decode supports packed per-request rows.
    Tails and cache outlive graph variants; intermediates can be shared by serial
    layers. Returned physical indices address a latent cache of 64-record pages.
    """

    def __init__(self, module, weights, source, query_residual, key_input, weight_input, scratch: IndexScratch,
                 state: IndexState, metadata: IndexMetadata, *, mode, session, name, workspace,
                 max_cache_pages=None, context=None):
        rows = source.shape[0]
        if (source.shape != (rows, 4096) or query_residual.shape != (rows, 1536)
                or source.dtype != torch.bfloat16 or query_residual.dtype != source.dtype
                or source.device != query_residual.device):
            raise ValueError("GLM index inputs require BF16 hidden[rows,4096] and query[rows,1536]")
        if (module.n_heads, module.head_dim, module.index_kpool, module.index_topk) != (32, 128, 4, 2048):
            raise ValueError("GLM indexing requires 32 heads, 128 channels, C4 pools and 2048 history tokens")
        if mode not in ("decode", "prefill") or (mode == "prefill" and metadata.state_slots.numel() != 1):
            raise ValueError("GLM index prefill requires one request")
        if (scratch.query.shape != (rows, 4096) or metadata.positions.shape != (rows,)
                or metadata.block_table.shape[1] != scratch.page_table.shape[1]
                or scratch.page_table.shape[0] != (1 if mode == "prefill" else rows)):
            raise ValueError("GLM index scratch and metadata capacities differ")
        self.module, self.scratch, self.state, self.metadata, self.mode = module, scratch, state, metadata, mode
        self.context = context
        # Layers of one forward share index scratch and metadata; only the first fills the selection tables.
        self.prepares_tables = True
        if context is not None and scratch.gathered.shape != (rows, context.shards.world * 512, 2):
            raise ValueError("GLM context-parallel selection requires candidate scratch for every row")
        # The index key and head weights arrive as columns of the layer's fused input projection.
        if key_input.shape != (rows, 128) or weight_input.shape != (rows, 32):
            raise ValueError("GLM index key and weight inputs require [rows, 128] and [rows, 32] projections")
        self.source, self.query_residual = source, query_residual
        self.key_input, self.weight_input = key_input, weight_input
        module.k_norm.weight = torch.nn.Parameter(weights["indexer.k_norm.weight"], requires_grad=False)
        module.k_norm.bias = torch.nn.Parameter(weights["indexer.k_norm.bias"], requires_grad=False)
        self.ape = weights["indexer.index_kpool_compress_ape"]
        self.query_projection = prepare_projection(query_residual, weights["indexer.wq_b"], scratch.query,
            session=session, name=f"{name}.query", workspace=workspace, activation_mode="auto")
        self.gate_projection = UnquantizedProjection(source, weights["indexer.index_kpool_compress_gate"], scratch.gate,
            session=session, name=f"{name}.gate")
        self.query_arguments = (scratch.query.view(-1, 128), scratch.query_fp8.view(-1, 128),
                                scratch.query_scales.view(-1), scratch.weights.view(-1))
        self.query_plan = glm_pool.plan_queries(*self.query_arguments)
        self.pool_arguments = (state.cache, state.tail, metadata.state_slots, metadata.query_start_loc,
            scratch.normalized_key, scratch.gate, self.ape, metadata.token_slots, metadata.positions)
        self.pool_plan = glm_pool.plan_pools(*self.pool_arguments, mode=mode)
        declare(session, (
            self.query_plan.request(name=f"{name}.rotate", prepare_call=lambda state:
                PreparedCall(run=lambda: state.quantize(*self.query_arguments))),
            self.pool_plan.request(name=f"{name}.pool", prepare_call=lambda state:
                PreparedCall(run=lambda: state.update(*self.pool_arguments))),
        ))
        page_table = scratch.page_table.expand(rows, -1) if mode == "prefill" else scratch.page_table
        # Prefill selection scratch grows with query rows (about 140 KB each), so long chunks select in slices.
        step = _SELECT_ROWS if mode == "prefill" else rows
        # SM103 selects through FlashInfer's paged FP8 logits and top-k; b12x's indexer serves SM120/SM121.
        selector = Sm103SparseIndexer if sm103.is_sm103(source.device) else PreparedSparseIndexer
        caps = None if selector is Sm103SparseIndexer else dsa_indexer.Caps(device=source.device, num_q_heads=32,
            max_q_rows=min(rows, step),
            max_page_table_width=page_table.shape[1], topk=512, mode=mode)
        cache = state.cache.view(state.cache.shape[0], -1)
        self.selectors = tuple(selector(scratch.query_fp8[start:end].view(end - start, 32, 128),
            scratch.weights[start:end], cache, page_table[start:end], scratch.pool_lengths[start:end],
            None, scratch.pools[start:end], caps=caps, session=session,
            name=f"{name}.select" if end - start == rows else f"{name}.select{start}", workspace=workspace,
            scores=None if context is None else scratch.scores[start:end])
            for start in range(0, rows, step) for end in (min(start + step, rows),))
        self.scratch_plans = (self.query_projection.plan,
            *(selector.plan for selector in self.selectors if isinstance(selector, PreparedSparseIndexer)))
        self.gathers = ()
        if context is not None:
            from ignition.components.collectives import PreparedAllGather

            columns, world = 1024, context.shards.world
            self.gathers = tuple(PreparedAllGather(scratch.candidates[start:end].view(torch.bfloat16).view(-1, 2 * columns),
                scratch.gathered[start:end].view(torch.bfloat16).view(-1, 2 * columns * world),
                runtime=context.candidates, session=session, coordinator=context.coordinator,
                name=f"{name}.candidates{start}") for start, end in context.slices(rows))
        # SM103 prefill attends over a slab of the request's context, so it selects positions, not records.
        self.logical = context is None and mode == "prefill" and sm103.is_sm103(source.device)
        self.expand_arguments = (scratch.pools, metadata.positions, metadata.requests,
                                 metadata.block_table, scratch.physical, scratch.selected_lengths)
        if self.logical:
            return
        self.expand_options = {"pool_size": 4, "block_size": 256, "block_stride_rows": 256,
                               "num_cache_blocks": state.cache.shape[0]}
        self.expand_plan = sparse_mla.plan_pooled_selection(device=source.device, max_rows=rows,
            page_size=256, max_page_table_width=metadata.block_table.shape[1],
            num_cache_blocks=max(state.cache.shape[0], max_cache_pages or 0))
        declare(session, (self.expand_plan.request(name=f"{name}.expand", prepare_call=lambda state:
            PreparedCall(run=lambda: state.run(*self.expand_arguments, **self.expand_options))),))

    def rebind_pages(self):
        """Address the state's current index cache; plans bound its page count from above."""
        cache = self.state.cache
        self.pool_arguments = (cache, *self.pool_arguments[1:])
        for selector in self.selectors:
            selector.rebind_cache(cache.view(cache.shape[0], -1))
        if not self.logical:
            self.expand_options["num_cache_blocks"] = cache.shape[0]

    def __call__(self):
        s, m = self.scratch, self.metadata
        self.query_projection(self.query_residual)
        s.normalized_key.copy_(self.module.k_norm(self.key_input))
        self.gate_projection(self.source)
        s.weights.copy_(self.weight_input)
        glm_pool.quantize_queries(*self.query_arguments, plan=self.query_plan)
        glm_pool.update_pools(*self.pool_arguments, plan=self.pool_plan)
        if self.prepares_tables:
            pools = (m.positions + 1) // 4
            s.pool_lengths.copy_(pools if self.context is None else self.context.shards.local_pools(pools))
            if self.mode == "prefill":
                s.page_table.copy_(m.block_table)
            else:
                s.page_table.copy_(m.block_table[m.requests.long()])
        for selector in self.selectors:
            selector()
        if self.logical:
            from .dcp import expand_tokens

            expand_tokens(s.pools, m.positions, s.physical, s.selected_lengths)
            return s.physical
        if self.context is None:
            sparse_mla.expand_pooled_topk_to_physical_slots(*self.expand_arguments,
                **self.expand_options, plan=self.expand_plan)
            return s.physical
        from .dcp import expand_tokens, merge_candidates, merge_expand_local, pack_candidates

        # Every rank merges identical global candidates, then keeps the records it owns.
        pack_candidates(s.pools, s.scores, s.candidates, self.context.shards)
        for gather, (start, end) in zip(self.gathers, self.context.slices(s.pools.shape[0]), strict=True):
            gather(s.candidates[start:end].view(torch.bfloat16).view(end - start, -1))
        if self.mode == "prefill":
            # Prefill attends over gathered context blocks addressed by global position.
            merge_candidates(s.gathered, s.global_pools)
            expand_tokens(s.global_pools, m.positions, s.physical, s.selected_lengths)
        else:
            merge_expand_local(s.gathered, s.global_pools, m.positions, m.requests, m.block_table, s.physical,
                               s.selected_lengths, self.context.shards)
        return s.physical
