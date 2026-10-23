"""Prepared b12x GQA over explicit cache pages, metadata and shared scratch."""

import torch
import triton
from b12x.attention import paged, paged_decode, paged_kv_write
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


def _scratch(specs, workspace):
    specs = tuple(specs)
    if not specs:
        return []
    values = planned_scratch(specs, workspace)
    return list(values) if isinstance(values, tuple) else [values]


@torch.library.custom_op("flare::paged_attention", mutates_args=("output", "scratch"))
def _attend(q: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pages: torch.Tensor,
            lengths: torch.Tensor, cu: torch.Tensor, sink: torch.Tensor | None,
            output: torch.Tensor, scratch: list[torch.Tensor], k_descale: torch.Tensor | None,
            v_descale: torch.Tensor | None, selected_blocks: torch.Tensor | None,
            handle: int, decode: bool, window: int,
            schedule_storage: torch.Tensor | None, schedule_slot: int) -> None:
    plan = plan_from_handle(handle)
    arguments = dict(q=q, k_cache=key, v_cache=value, page_table=pages, cache_seqlens=lengths,
                     cu_seqlens_q=cu, attention_sink_bias=sink, output=output, scratch=scratch,
                     k_descale=k_descale, v_descale=v_descale)
    if selected_blocks is not None:
        arguments["q2k_indices"] = selected_blocks
    if schedule_slot >= 0:
        from b12x.attention.paged.schedule import PrefillSchedule

        arguments["external_schedule"] = PrefillSchedule.from_plan(plan).bind(schedule_storage[schedule_slot])
    if decode:
        paged_decode.run(paged_decode.bind(plan, **arguments))
    else:
        paged.run(binding=paged.bind(plan, **arguments, window_left=window), plan=plan)


@_attend.register_fake
def _fake_attend(q, key, value, pages, lengths, cu, sink, output, scratch, k_descale, v_descale, selected_blocks, handle, decode, window, schedule_storage, schedule_slot):
    pass


class PreparedPagedAttention:
    def __init__(self, q, key, value, pages, lengths, cu, output, *, sink=None, mode,
                 window=-1, width=1, causal=True, k_descale=None, v_descale=None,
                 selected_blocks=None, session, workspace, name, schedules=None):
        self.schedule_storage, self.schedule_slot = None, -1
        if key.dtype == torch.float8_e4m3fn:
            if k_descale is None:
                k_descale = torch.ones(1, dtype=torch.float32, device=key.device)
            if v_descale is None:
                v_descale = torch.ones(1, dtype=torch.float32, device=value.device)
        self.selected_blocks = selected_blocks
        if selected_blocks is not None:
            if (selected_blocks.shape != (key.shape[2], q.shape[0], 16)
                    or selected_blocks.dtype != torch.int32 or not selected_blocks.is_contiguous()):
                raise ValueError("selected blocks must be contiguous int32 [kv_heads, query_capacity, 16]")
            if not causal or window != -1 or sink is not None:
                raise ValueError("selected-block attention requires causal full-context GQA without sinks")
            if mode in ("decode", "verify"):
                mode = "decode" if width == 1 else "verify"
            else:
                mode = "extend"
        self.decode = selected_blocks is None and mode in ("decode", "verify") and paged_decode.supports(
            num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
            head_dim_vo=value.shape[3], page_size=key.shape[1], kv_dtype=key.dtype,
            max_q_per_req=width, window_left=window, causal=causal, has_sinks=sink is not None)
        if not causal and not self.decode:
            raise ValueError("noncausal paged attention requires a supported paged-decode geometry")
        self.window, self.workspace = window, workspace
        self.arguments = dict(q=q, k_cache=key, v_cache=value, page_table=pages,
            cache_seqlens=lengths, cu_seqlens_q=cu, output=output, attention_sink_bias=sink,
            k_descale=k_descale, v_descale=v_descale)
        if selected_blocks is not None:
            self.arguments["q2k_indices"] = selected_blocks
        if self.decode:
            self.plan = paged_decode.plan(paged_decode.Caps(device=q.device, num_q_heads=q.shape[1],
                num_kv_heads=key.shape[2], head_dim_qk=q.shape[2], head_dim_vo=value.shape[3],
                page_size=key.shape[1], max_batch=pages.shape[0], max_q_per_req=width,
                window_left=window, has_sinks=sink is not None, causal=causal, kv_dtype=key.dtype))
        else:
            geometry = dict(device=q.device, q_dtype=q.dtype, kv_dtype=key.dtype,
                num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                head_dim_vo=value.shape[3], page_size=key.shape[1], batch=pages.shape[0],
                max_cache_page_count=pages.shape[1], window_left=window)
            if mode == "decode":
                capacity = paged.decode_graph_capacity(**geometry)
            elif mode == "verify":
                capacity = paged.verify_graph_capacity(**geometry, query_len=width)
            else:
                capacity = paged.extend_graph_capacity(**geometry, total_q_capacity=q.shape[0])
            caps = paged.Caps(device=q.device, mode=mode if mode in ("decode", "verify") else "extend",
                dtype=q.dtype, kv_dtype=key.dtype,
                num_q_heads=q.shape[1], num_kv_heads=key.shape[2], head_dim_qk=q.shape[2],
                head_dim_vo=value.shape[3], page_size=key.shape[1], max_total_q=q.shape[0],
                max_batch=pages.shape[0], max_page_table_width=pages.shape[1], num_cache_pages=key.shape[0],
                max_work_items=max(capacity.max_work_items, q.shape[0] * 32) if selected_blocks is not None and mode in ("decode", "verify") else capacity.max_work_items,
                max_partial_rows=max(getattr(capacity, "max_partial_rows", 0), q.shape[0] * 32) if selected_blocks is not None and mode in ("decode", "verify") else getattr(capacity, "max_partial_rows", 0),
                use_cuda_graph=True,
                copy_runtime_metadata=not all(t.dtype == torch.int32 and t.is_contiguous()
                                              for t in (pages, lengths, cu)),
                msa_block_sparse=selected_blocks is not None,
                msa_union_tile=selected_blocks is not None and mode == "extend")
            invocation = dict(paged.invocation_from_tensors(caps, **self.arguments), window_left=window)
            if mode == "verify":
                invocation["query_len"] = width
            if selected_blocks is not None and mode in ("decode", "verify"):
                invocation["force_split_kv"] = True
            self.plan = paged.plan(caps, invocation=invocation)
            if schedules is not None and mode in ("verify", "extend"):
                schedules.register(self, caps)

        def prepared_call(state, arguments):
            from b12x.attention.paged_decode._preparation import scratch_specs

            specs = scratch_specs(state.query, state.config, q.device) if self.decode else state.scratch_plan.scratch_specs()
            scratch = _scratch(specs, workspace)
            kwargs = {"plan": self.plan} if self.decode else {"window_left": window}
            binding = state.bind(scratch=scratch, **arguments, **kwargs)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            def inputs():
                arguments = dict(self.arguments)
                for field in ("q", "page_table", "cache_seqlens", "cu_seqlens_q", "output"):
                    value = arguments[field]
                    arguments[field] = torch.empty_strided(value.shape, value.stride(),
                                                         dtype=value.dtype, device=value.device)
                arguments["q"].normal_(std=.2)
                arguments["page_table"].zero_()
                arguments["cache_seqlens"].fill_(min(max(4096, q.shape[0]), pages.shape[1] * key.shape[1]))
                arguments["cu_seqlens_q"].copy_(torch.arange(cu.numel(), device=cu.device, dtype=cu.dtype)
                                                   * (width if mode == "verify" else q.shape[0] // pages.shape[0]))
                if selected_blocks is not None:
                    blocks = torch.arange(16, dtype=torch.int32, device=q.device)
                    extent = min(max(4096, q.shape[0]), pages.shape[1] * key.shape[1])
                    blocks = torch.where(blocks < triton.cdiv(extent, 128), blocks, -1)
                    arguments["q2k_indices"] = blocks.expand_as(selected_blocks).contiguous()
                return arguments, arguments["q"].clone()

            operands, seed = benchmark_resource(session, ("paged_attention", name), inputs)
            call = prepared_call(state, operands)
            return PreparedCall(run=call.run, produce=lambda: operands["q"].copy_(seed),
                                owners=(*call.owners, operands, seed), output=operands["output"])

        def prime(state):
            call = benchmark(state)
            return PreparedCall(run=call.run, produce=call.produce,
                                owners=prime_owners(workspace))

        declare(session, (self.plan.request(name=name,
            prepare_call=prime if selected_blocks is not None else lambda state: prepared_call(state, self.arguments),
            benchmark_call=benchmark),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = _scratch(self.plan.scratch_specs(), self.workspace)
        if not self.decode:
            paged.bind(self.plan, scratch=self.scratch, **self.arguments, window_left=self.window)

    def __call__(self):
        a = self.arguments
        _attend(a["q"], a["k_cache"], a["v_cache"], a["page_table"], a["cache_seqlens"],
                a["cu_seqlens_q"], a["attention_sink_bias"], a["output"], self.scratch,
                a["k_descale"], a["v_descale"], self.selected_blocks,
                self.plan.handle, self.decode, self.window, self.schedule_storage, self.schedule_slot)
        return a["output"]


class PreparedPagedKVWriter:
    def __init__(self, key, value, key_cache, value_cache, slots, *, session, name, k_scale=None, v_scale=None):
        if key_cache.dtype == torch.float8_e4m3fn:
            if k_scale is None:
                k_scale = torch.ones(1, dtype=torch.float32, device=key.device)
            if v_scale is None:
                v_scale = torch.ones(1, dtype=torch.float32, device=value.device)
        self.arguments = dict(key=key, value=value, k_cache=key_cache, v_cache=value_cache, slot_mapping=slots,
                              k_scale=k_scale, v_scale=v_scale)
        self.plan = paged_kv_write.plan(paged_kv_write.Caps(device=key.device, num_kv_heads=key.shape[1],
            head_dim_k=key.shape[2], head_dim_v=value.shape[2], page_size=key_cache.shape[1], kv_dtype=key_cache.dtype))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(**self.arguments))),))

    def __call__(self):
        paged_kv_write.run(self.plan, **self.arguments)


class PagedAttentionMetadata:
    """Cache-write slots and sequence extents shared by layers with one page geometry."""

    def __init__(self, metadata, *, page_tokens, buffers):
        self.metadata, self.page_tokens = metadata, page_tokens
        self.slots = buffers.zeros(metadata.positions.shape, torch.int64).fill_(-1)
        self.lengths = buffers.zeros(metadata.lengths.shape, torch.int32)

    def __call__(self):
        m = self.metadata
        columns = (m.positions.clamp_min(0) // self.page_tokens).clamp(max=m.token_page_table.shape[1] - 1)
        rows = torch.arange(m.positions.numel(), device=m.positions.device)
        pages = m.token_page_table[rows, columns]
        self.slots.copy_(torch.where((m.positions >= 0) & (pages >= 0),
            pages.long() * self.page_tokens + m.positions.remainder(self.page_tokens), -1))
        self.lengths.copy_(torch.where(m.lengths > 0, (m.starts + m.lengths).int(), 0))
