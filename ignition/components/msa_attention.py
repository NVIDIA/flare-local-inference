"""Prepared packed GQA/index projections and signed block selection."""

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.attention.dsa_indexer import MSACaps, plan_msa, run_msa
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


@triton.jit
def _prepare(
    X,
    W,
    Inverse,
    Requests,
    Positions,
    Pages,
    IndexPages,
    Q,
    IQ,
    K,
    V,
    IK,
    KScale,
    VScale,
    ROW_STRIDE: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    H_INDEX: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    INDEX_TABLE_WIDTH: tl.constexpr,
    BATCH: tl.constexpr,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    V_PAGE: tl.constexpr,
    V_TOKEN: tl.constexpr,
    IK_PAGE: tl.constexpr,
    IK_TOKEN: tl.constexpr,
):
    row, role_head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, 128)
    request = tl.load(Requests + row)
    position = tl.load(Positions + row)
    live = (
        (request >= 0)
        & (request < BATCH)
        & (position >= 0)
        & (position < TABLE_WIDTH * PAGE)
    )
    value_role = (role_head >= H_Q + H_KV) & (role_head < H_Q + 2 * H_KV)
    index_q_role = (role_head >= H_Q + 2 * H_KV) & (
        role_head < H_Q + 2 * H_KV + H_INDEX
    )
    index_k_role = role_head == H_Q + 2 * H_KV + H_INDEX
    norm_role = tl.where(
        role_head < H_Q, 0, tl.where(index_q_role, 2, tl.where(index_k_role, 3, 1))
    )
    value = tl.load(X + row * ROW_STRIDE + role_head * 128 + dim, live, other=0).to(
        tl.float32
    )
    weight = tl.load(W + norm_role * 128 + dim).to(tl.float32) + 1.0
    inverse_rms = tl.rsqrt(tl.sum(value * value, 0) / 128.0 + EPS)
    normalized = value * inverse_rms * weight
    half = ROTARY // 2
    partner_dim = tl.where(dim < half, dim + half, dim - half)
    partner = tl.load(
        X + row * ROW_STRIDE + role_head * 128 + partner_dim,
        live & (dim < ROTARY),
        other=0,
    ).to(tl.float32)
    partner_weight = (
        tl.load(W + norm_role * 128 + partner_dim, dim < ROTARY, other=0).to(tl.float32)
        + 1.0
    )
    partner = partner * inverse_rms * partner_weight
    frequency = tl.load(Inverse + dim % half, dim < ROTARY, other=0) * position.to(
        tl.float32
    )
    cosine = tl.extra.cuda.libdevice.cos(frequency).to(tl.bfloat16).to(tl.float32)
    sine = tl.extra.cuda.libdevice.sin(frequency).to(tl.bfloat16).to(tl.float32)
    rotated = normalized * cosine + tl.where(dim < half, -partner, partner) * sine
    result = tl.where(value_role, value, tl.where(dim < ROTARY, rotated, normalized))
    if role_head < H_Q:
        tl.store(Q + (row * H_Q + role_head) * 128 + dim, tl.where(live, result, 0))
    elif index_q_role:
        # The FP8 query casts through the model dtype; cache keys cast from FP32.
        rounded = tl.where(live, result, 0).to(tl.bfloat16).to(tl.float32)
        tl.store(
            IQ + (row * H_INDEX + role_head - H_Q - 2 * H_KV) * 128 + dim,
            tl.minimum(tl.maximum(rounded, -448.0), 448.0).to(tl.float8e4nv),
        )
    elif index_k_role:
        page = tl.load(
            IndexPages + request * INDEX_TABLE_WIDTH + position // PAGE, live, other=0
        )
        offset = page.to(tl.int64) * IK_PAGE + position % PAGE * IK_TOKEN + dim
        tl.store(
            IK + offset,
            tl.minimum(tl.maximum(result, -448.0), 448.0),
            live & (page >= 0),
        )
    elif value_role:
        page = tl.load(Pages + request * TABLE_WIDTH + position // PAGE, live, other=0)
        offset = (
            page.to(tl.int64) * V_PAGE
            + position % PAGE * V_TOKEN
            + (role_head - H_Q - H_KV) * 128
            + dim
        )
        result = result / tl.load(VScale)
        if V.dtype.element_ty == tl.float8e4nv:
            result = tl.minimum(tl.maximum(result, -448.0), 448.0)
        tl.store(V + offset, result, live & (page >= 0))
    else:
        page = tl.load(Pages + request * TABLE_WIDTH + position // PAGE, live, other=0)
        offset = (
            page.to(tl.int64) * K_PAGE
            + position % PAGE * K_TOKEN
            + (role_head - H_Q) * 128
            + dim
        )
        result = result / tl.load(KScale)
        if K.dtype.element_ty == tl.float8e4nv:
            result = tl.minimum(tl.maximum(result, -448.0), 448.0)
        tl.store(K + offset, result, live & (page >= 0))


@dataclass(frozen=True)
class PackedQuery:
    operands: tuple
    epsilon: float


PACKED = native_contract(
    component_id="flare.packed_gqa_index_prepare",
    query_type=PackedQuery,
    tactics=lambda q: (4, 1, 2, 8),
)


def _constants(tensors, q):
    x, w, inverse, requests, positions, pages, ipages, query, iq, kc, vc, ik, ks, vs = (
        tensors
    )
    return (
        x.stride(0),
        query.shape[1],
        kc.shape[2],
        iq.shape[1],
        2 * inverse.numel(),
        q.epsilon,
        kc.shape[1],
        pages.shape[1],
        ipages.shape[1],
        pages.shape[0],
        *kc.stride()[:2],
        *vc.stride()[:2],
        *ik.stride()[:2],
    )


@program_cache(scope="preparation")
def compile_packed(payload, warps, ordinal):
    from b12x.sequence.ple_hash._preparation import _CompilePointer

    q = PackedQuery(**dict(payload))
    tensors = tuple(map(metadata_tensor, q.operands))
    pointers = tuple(_CompilePointer(t.dtype, 16) for t in tensors)
    with torch.cuda.device(ordinal):
        return _prepare.warmup(
            *pointers,
            *_constants(tensors, q),
            grid=(tensors[0].shape[0], tensors[0].shape[1] // 128),
            num_warps=warps,
        )


@dataclass(frozen=True)
class PackedState:
    query: PackedQuery
    program: object

    def run(self, tensors):
        self.program[(tensors[0].shape[0], tensors[0].shape[1] // 128, 1)](
            *tensors, *_constants(tensors, self.query)
        )


@torch.library.custom_op(
    "flare::packed_gqa_index_prepare",
    mutates_args=("query", "index_query", "key", "value", "index_key"),
)
def _execute(
    projected: torch.Tensor,
    weights: torch.Tensor,
    inverse: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    pages: torch.Tensor,
    index_pages: torch.Tensor,
    query: torch.Tensor,
    index_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    handle: int,
) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, projected.device).run(
        (
            projected,
            weights,
            inverse,
            requests,
            positions,
            pages,
            index_pages,
            query,
            index_query,
            key,
            value,
            index_key,
            k_scale,
            v_scale,
        )
    )


@_execute.register_fake
def _fake(
    projected,
    weights,
    inverse,
    requests,
    positions,
    pages,
    index_pages,
    query,
    index_query,
    key,
    value,
    index_key,
    k_scale,
    v_scale,
    handle,
):
    pass


class PreparedPackedAttention:
    def __init__(
        self,
        projected,
        q_norm,
        k_norm,
        index_q_norm,
        index_k_norm,
        inverse,
        metadata,
        query,
        index_query,
        key_cache,
        value_cache,
        index_cache,
        *,
        epsilon,
        session,
        name,
        k_descale=None,
        v_descale=None,
        index_page_table=None,
    ):
        heads, kv_heads, index_heads = (
            query.shape[1],
            key_cache.shape[2],
            index_query.shape[1],
        )
        rows = projected.shape[0]
        if (
            projected.shape != (rows, (heads + 2 * kv_heads + index_heads + 1) * 128)
            or projected.dtype != torch.bfloat16
            or projected.stride(1) != 1
        ):
            raise ValueError(
                "packed attention requires BF16 [Q,K,V,index-Q,index-K] projection rows"
            )
        if (
            query.shape != (rows, heads, 128)
            or query.dtype != torch.bfloat16
            or not query.is_contiguous()
        ):
            raise ValueError("main query must be contiguous BF16 [rows, heads, 128]")
        if (
            index_query.shape != (rows, index_heads, 128)
            or index_query.dtype != torch.float8_e4m3fn
            or not index_query.is_contiguous()
        ):
            raise ValueError("index query must be contiguous E4M3 [rows, heads, 128]")
        if (
            key_cache.shape != value_cache.shape
            or key_cache.dtype != value_cache.dtype
            or key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        ):
            raise ValueError(
                "main cache must have matching BF16 or E4M3 key/value planes"
            )
        if (
            key_cache.stride()[1:] != (kv_heads * 128, 128, 1)
            or value_cache.stride()[1:] != key_cache.stride()[1:]
        ):
            raise ValueError("main cache token and head dimensions must be contiguous")
        if (
            index_cache.ndim != 3
            or index_cache.shape[1:] != (key_cache.shape[1], 128)
            or index_cache.dtype != torch.float8_e4m3fn
            or index_cache.stride(-1) != 1
        ):
            raise ValueError("index cache must be E4M3 [pages, page_size, 128]")
        if (
            inverse.dtype != torch.float32
            or inverse.ndim != 1
            or not 0 < inverse.numel() <= 64
            or not inverse.is_contiguous()
        ):
            raise ValueError(
                "partial NeoX rotary frequencies must be FP32 with at most 64 entries"
            )
        norms = (q_norm, k_norm, index_q_norm, index_k_norm)
        if any(w.shape != (128,) or w.dtype != torch.bfloat16 for w in norms):
            raise ValueError("head norm offsets must be BF16 vectors of length 128")
        for scale in (k_descale, v_descale):
            if scale is not None and (
                scale.dtype != torch.float32
                or scale.numel() != 1
                or not scale.is_contiguous()
            ):
                raise ValueError("KV descales must be contiguous FP32 scalars")
        index_page_table = metadata.page_table if index_page_table is None else index_page_table
        if (
            metadata.page_table.ndim != 2
            or not metadata.page_table.is_contiguous()
            or index_page_table.shape != metadata.page_table.shape
            or index_page_table.dtype != metadata.page_table.dtype
            or index_page_table.dtype not in (torch.int32, torch.int64)
            or not index_page_table.is_contiguous()
            or index_page_table.device != metadata.page_table.device
            or index_page_table.device != projected.device
        ):
            raise ValueError(
                "index page table must be contiguous integer storage matching the main "
                "page table shape, dtype and device"
            )
        self.weights = torch.stack(norms)
        unit = torch.ones(1, dtype=torch.float32, device=projected.device)
        self.arguments = (
            projected,
            self.weights,
            inverse,
            metadata.requests,
            metadata.positions,
            metadata.page_table,
            index_page_table,
            query,
            index_query,
            key_cache,
            value_cache,
            index_cache,
            unit if k_descale is None else k_descale,
            unit if v_descale is None else v_descale,
        )
        self.output = query
        q = PackedQuery(tuple(tensor_facts(t) for t in self.arguments), float(epsilon))
        payload = PACKED.encode_query(q)

        def materialize(warps):
            program = compile_packed(payload, warps, projected.device.index)
            load_programs(program)
            return attach_programs(PackedState(q, program), program)

        self.plan = native_plan(
            contract=PACKED,
            query=q,
            device=projected.device,
            scratch_bytes=lambda t: 0,
            materialize=materialize,
            compile_jobs=lambda t, d: (
                CompileJob.create(
                    "ignition.components.msa_attention:compile_packed",
                    payload,
                    t,
                    d.ordinal,
                ),
            ),
        )

        def benchmark(state):
            def inputs():
                values = list(self.arguments)
                batch, width = metadata.page_table.shape
                pages_per_request = min(
                    width, triton.cdiv(triton.cdiv(rows, batch), key_cache.shape[1])
                )
                count = batch * pages_per_request
                for i in (0, 7, 8):
                    t = values[i]
                    values[i] = torch.empty_strided(
                        t.shape, t.stride(), dtype=t.dtype, device=t.device
                    )
                values[0].normal_(std=0.2)
                values[3] = (
                    torch.arange(
                        rows, device=projected.device, dtype=metadata.requests.dtype
                    )
                    % batch
                )
                values[4] = (
                    torch.arange(
                        rows, device=projected.device, dtype=metadata.positions.dtype
                    )
                    // batch
                ) % (pages_per_request * key_cache.shape[1])
                table = (
                    torch.arange(
                        batch * width,
                        device=projected.device,
                        dtype=metadata.page_table.dtype,
                    ).view(batch, width)
                    % count
                )
                values[5], values[6] = table, table
                for i in (9, 10, 11):
                    t = values[i]
                    values[i] = torch.empty_strided(
                        (count, *t.shape[1:]),
                        t.stride(),
                        dtype=t.dtype,
                        device=t.device,
                    )
                return tuple(values), values[0].clone()

            values, seed = benchmark_resource(
                session, ("packed_gqa_index_prepare", name), inputs
            )
            return PreparedCall(
                run=lambda: state.run(values),
                owners=(values, seed),
                output=values[7],
                produce=lambda: values[0].copy_(seed),
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: replace(benchmark(state), owners=()),
                    benchmark_call=benchmark,
                ),
            ),
        )

    def __call__(self):
        _execute(*self.arguments, self.plan.handle)
        return self.output


class PreparedMSAIndexer:
    def __init__(
        self,
        query,
        cache,
        page_table,
        lengths,
        cu_seqlens_q,
        positions,
        output,
        *,
        session,
        workspace,
        name,
        max_q_per_request=None,
    ):
        self.arguments = dict(
            query=query,
            cache=cache,
            page_table=page_table,
            cache_lengths=lengths,
            cu_seqlens_q=cu_seqlens_q,
            positions=positions,
            output=output,
        )
        caps = MSACaps(
            device=query.device,
            max_q_rows=query.shape[0],
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            num_index_heads=query.shape[1],
            page_size=cache.shape[1],
            max_q_per_request=(
                1 if query.shape[0] == page_table.shape[0] else max_q_per_request
            ),
        )
        self.plan = plan_msa(
            caps, **{k: v for k, v in self.arguments.items() if k != "output"}
        )
        self.workspace, self.output = workspace, output

        def benchmark(state):
            def inputs():
                q = torch.empty_strided(
                    query.shape, query.stride(), device=query.device, dtype=query.dtype
                )
                q.copy_(torch.randn(query.shape, device=query.device))
                extent = min(page_table.shape[1] * cache.shape[1], 4096)
                count = max(1, min(cache.shape[0], triton.cdiv(extent, cache.shape[1])))
                k = torch.empty_strided(
                    (count, *cache.shape[1:]),
                    cache.stride(),
                    device=cache.device,
                    dtype=cache.dtype,
                )
                k.copy_(torch.randn(k.shape, device=k.device))
                pages = (
                    torch.arange(
                        page_table.numel(),
                        device=page_table.device,
                        dtype=page_table.dtype,
                    ).view_as(page_table)
                    % k.shape[0]
                )
                batch, rows = page_table.shape[0], query.shape[0]
                starts = (
                    torch.arange(
                        batch + 1, device=query.device, dtype=cu_seqlens_q.dtype
                    )
                    * rows
                    // batch
                )
                lens = torch.full_like(lengths, extent)
                pos = torch.full_like(positions, extent - 1)
                return (
                    q,
                    k,
                    pages,
                    lens,
                    starts,
                    pos,
                    torch.empty_like(output),
                ), q.clone()

            values, seed = benchmark_resource(session, ("msa_indexer", name), inputs)
            storage = planned_scratch(self.plan.scratch_specs(), workspace)
            scratch = storage if isinstance(storage, torch.Tensor) else storage[0]
            return PreparedCall(
                run=lambda: state.run(*values, scratch),
                owners=(*prime_owners(workspace, storage), values, seed),
                output=values[-1],
                produce=lambda: values[0].copy_(seed),
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: replace(
                        benchmark(state), owners=prime_owners(workspace)
                    ),
                    benchmark_call=benchmark,
                ),
            ),
        )
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def __call__(self):
        return run_msa(self.plan, **self.arguments, scratch=self.scratch)
