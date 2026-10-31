"""mHC pre and post/pre for SM120 and SM121 through a TileLang kernel pair.

b12x's fused post/pre runs the DeepSeek V4.1 decode shape with one CTA per
token and a scalar FP32 projection, which leaves Blackwell consumer parts well
below bandwidth: on a DGX Spark it costs 14.7 us for eight rows and 38.4 us for
128 rows, and the recipe calls it twice per layer. This path keeps the same
arithmetic and changes the execution. One CTA owns a hidden slice for every
row, so the projection weights are read once per call instead of once per row
tile; the mixing and the collapse stay in FP32 with explicit multiply and add
rounding; and the 24-wide projection runs on tensor cores as an FP16 high/low
split with the low plane shifted out of the FP16 subnormal range, which
reproduces the FP32 weights to about 1e-7 relative.

Arithmetic, identical to b12x's references:
  o_j = post[j]*x + sum_i comb[i,j]*residual[i]   (FP32, reference order)
  mixes = (o_flat @ fn^T) * rsqrt(mean(o_flat^2) + rms_eps)
  pre = sigmoid(mixes[:4]*scale[0] + base[:4]) + hc_eps
  post = 2*sigmoid(mixes[4:8]*scale[1] + base[4:8])
  comb = sinkhorn(mixes[8:24]*scale[2] + base[8:24], hc_eps, iterations)
  y = RMSNorm(sum_j pre_in[j] * o_j, norm_weight, norm_eps)
"""

import itertools
import weakref
from dataclasses import dataclass

import torch
import tilelang
from tilelang import language as T

MULT = 4
MIXES = 24
MIXES_PAD = 32
SLOTS = MIXES + 2
BLOCK_M = 16
BLOCK_H = 32
# Below this many rows b12x's fused pair is faster; its launch shape is one CTA
# per token, and the TileLang path's split reduction then costs more than it saves.
MIN_ROWS = 4
# The pair's grid is hidden/block_h CTAs, each looping row tiles serially, so the
# useful width follows the part: a wide part is already under-filled at 160 CTAs
# and only loses parallelism from coarser blocks, while a narrow part saturates at
# 80 CTAs and then pays half the partial traffic and half the finalize reduction.
PREFILL_BLOCK_H = 64
# The low plane holds the FP16 residual of the FP32 weights. mHC projections are
# O(1e-2), so that residual is O(1e-5) and would land in FP16 subnormals, or flush
# to zero; the plane is stored 2**LOW_SHIFT times larger and scaled back in FP32.
LOW_SHIFT = 12
LOW_SCALE = 2.0 ** -LOW_SHIFT
# The projection operand is staged in FP16 for the tensor-core GEMM, but the
# mixed residual is BF16 and can exceed FP16's 65504 ceiling. Scale it by a
# power of two (exact in FP32, so no mantissa loss) and undo the scale in FP32
# after the GEMM. 2**-8 covers activations up to 1.6e7. Values below 2**-6
# (0.0156) land in FP16 subnormals, adding at most 2**-17 (7.6e-6) absolute
# error, and values below 2**-17 flush to zero; both are negligible for a
# projection with O(1e-2) weights. The residual output and the RMS sums keep
# the unscaled values.
A_SHIFT = 8
A_SCALE = 2.0 ** -A_SHIFT
A_SCALE_INV = 2.0 ** A_SHIFT
LOW_SCALE_SCALED = A_SCALE_INV * LOW_SCALE
THREADS = 128
FINALIZE_THREADS = 128

_PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def _fadd(a, b):
    return T.call_extern("float32", "__fadd_rn", a, b)


def _fmul(a, b):
    return T.call_extern("float32", "__fmul_rn", a, b)


def _build_partial(rows, hidden, block_h, threads, pre_only):
    """Mix, project, and collapse one hidden slice for every row."""
    splits = hidden // block_h
    tiles = (rows + BLOCK_M - 1) // BLOCK_M
    k = MULT * block_h

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def kernel(
        fn_hi: T.Tensor[(splits, k, MIXES_PAD), T.float16],
        fn_lo: T.Tensor[(splits, k, MIXES_PAD), T.float16],
        x: T.Tensor[(rows, hidden), T.bfloat16],
        residual: T.Tensor[(rows, MULT, hidden), T.bfloat16],
        post_in: T.Tensor[(rows, MULT), T.float32],
        comb_in: T.Tensor[(rows, MULT, MULT), T.float32],
        pre_in: T.Tensor[(rows, MULT), T.float32],
        residual_out: T.Tensor[(rows, MULT, hidden), T.bfloat16],
        y_out: T.Tensor[(rows, hidden), T.bfloat16],
        partials: T.Tensor[(rows, splits, SLOTS), T.float32],
    ):
        with T.Kernel(splits, threads=threads) as pid_h:
            h0 = pid_h * block_h
            a_s = T.alloc_shared((BLOCK_M, k), T.float16)
            fn_s = T.alloc_shared((k, MIXES_PAD), T.float16)
            fn2_s = T.alloc_shared((k, MIXES_PAD), T.float16)
            dot = T.alloc_fragment((BLOCK_M, MIXES_PAD), T.float32)
            dot_lo = T.alloc_fragment((BLOCK_M, MIXES_PAD), T.float32)
            sq_part = T.alloc_fragment((BLOCK_M, block_h), T.float32)
            yq_part = T.alloc_fragment((BLOCK_M, block_h), T.float32)
            sqr = T.alloc_fragment((BLOCK_M,), T.float32)
            ysq = T.alloc_fragment((BLOCK_M,), T.float32)

            T.copy(fn_hi[pid_h, 0, 0], fn_s)
            T.copy(fn_lo[pid_h, 0, 0], fn2_s)
            for tile in T.serial(tiles):
                m0 = tile * BLOCK_M
                T.clear(dot)
                T.clear(dot_lo)

                for m, hh in T.Parallel(BLOCK_M, block_h):
                    h = h0 + hh
                    live = m0 + m < rows
                    xv = T.alloc_var(T.float32)
                    if not pre_only:
                        xv = T.if_then_else(live, T.Cast(T.float32, x[m0 + m, h]), 0.0)
                    r0 = T.alloc_var(T.float32)
                    r1 = T.alloc_var(T.float32)
                    r2 = T.alloc_var(T.float32)
                    r3 = T.alloc_var(T.float32)
                    r0 = T.if_then_else(live, T.Cast(T.float32, residual[m0 + m, 0, h]), 0.0)
                    r1 = T.if_then_else(live, T.Cast(T.float32, residual[m0 + m, 1, h]), 0.0)
                    r2 = T.if_then_else(live, T.Cast(T.float32, residual[m0 + m, 2, h]), 0.0)
                    r3 = T.if_then_else(live, T.Cast(T.float32, residual[m0 + m, 3, h]), 0.0)
                    b0 = T.alloc_var(T.bfloat16)
                    b1 = T.alloc_var(T.bfloat16)
                    b2 = T.alloc_var(T.bfloat16)
                    b3 = T.alloc_var(T.bfloat16)
                    if pre_only:
                        # The pre operation carries the residual through unchanged.
                        b0 = T.Cast(T.bfloat16, r0)
                        b1 = T.Cast(T.bfloat16, r1)
                        b2 = T.Cast(T.bfloat16, r2)
                        b3 = T.Cast(T.bfloat16, r3)
                    else:
                        b0 = _fadd(_fmul(post_in[m0 + m, 0], xv),
                                   _fadd(_fadd(_fadd(_fmul(comb_in[m0 + m, 0, 0], r0),
                                                     _fmul(comb_in[m0 + m, 1, 0], r1)),
                                               _fmul(comb_in[m0 + m, 2, 0], r2)),
                                         _fmul(comb_in[m0 + m, 3, 0], r3)))
                        b1 = _fadd(_fmul(post_in[m0 + m, 1], xv),
                                   _fadd(_fadd(_fadd(_fmul(comb_in[m0 + m, 0, 1], r0),
                                                     _fmul(comb_in[m0 + m, 1, 1], r1)),
                                               _fmul(comb_in[m0 + m, 2, 1], r2)),
                                         _fmul(comb_in[m0 + m, 3, 1], r3)))
                        b2 = _fadd(_fmul(post_in[m0 + m, 2], xv),
                                   _fadd(_fadd(_fadd(_fmul(comb_in[m0 + m, 0, 2], r0),
                                                     _fmul(comb_in[m0 + m, 1, 2], r1)),
                                               _fmul(comb_in[m0 + m, 2, 2], r2)),
                                         _fmul(comb_in[m0 + m, 3, 2], r3)))
                        b3 = _fadd(_fmul(post_in[m0 + m, 3], xv),
                                   _fadd(_fadd(_fadd(_fmul(comb_in[m0 + m, 0, 3], r0),
                                                     _fmul(comb_in[m0 + m, 1, 3], r1)),
                                               _fmul(comb_in[m0 + m, 2, 3], r2)),
                                         _fmul(comb_in[m0 + m, 3, 3], r3)))
                    if live:
                        residual_out[m0 + m, 0, h] = b0
                        residual_out[m0 + m, 1, h] = b1
                        residual_out[m0 + m, 2, h] = b2
                        residual_out[m0 + m, 3, h] = b3
                    a_s[m, 0 * block_h + hh] = T.Cast(
                        T.float16, _fmul(T.Cast(T.float32, b0), A_SCALE))
                    a_s[m, 1 * block_h + hh] = T.Cast(
                        T.float16, _fmul(T.Cast(T.float32, b1), A_SCALE))
                    a_s[m, 2 * block_h + hh] = T.Cast(
                        T.float16, _fmul(T.Cast(T.float32, b2), A_SCALE))
                    a_s[m, 3 * block_h + hh] = T.Cast(
                        T.float16, _fmul(T.Cast(T.float32, b3), A_SCALE))
                    yv = T.alloc_var(T.float32)
                    yv = _fadd(_fadd(_fadd(_fmul(pre_in[m0 + m, 0], T.Cast(T.float32, b0)),
                                           _fmul(pre_in[m0 + m, 1], T.Cast(T.float32, b1))),
                                      _fmul(pre_in[m0 + m, 2], T.Cast(T.float32, b2))),
                                _fmul(pre_in[m0 + m, 3], T.Cast(T.float32, b3)))
                    yb = T.alloc_var(T.bfloat16)
                    yb = yv
                    if live:
                        y_out[m0 + m, h] = yb
                    sq_part[m, hh] = (T.Cast(T.float32, b0) * T.Cast(T.float32, b0)
                                      + T.Cast(T.float32, b1) * T.Cast(T.float32, b1)
                                      + T.Cast(T.float32, b2) * T.Cast(T.float32, b2)
                                      + T.Cast(T.float32, b3) * T.Cast(T.float32, b3))
                    yq_part[m, hh] = T.Cast(T.float32, yb) * T.Cast(T.float32, yb)

                T.gemm(a_s, fn_s, dot)
                T.gemm(a_s, fn2_s, dot_lo)
                for m, j in T.Parallel(BLOCK_M, MIXES_PAD):
                    dot[m, j] = _fadd(_fmul(dot[m, j], A_SCALE_INV),
                                      _fmul(dot_lo[m, j], LOW_SCALE_SCALED))

                T.reduce_sum(sq_part, sqr, dim=1)
                T.reduce_sum(yq_part, ysq, dim=1)
                for m, j in T.Parallel(BLOCK_M, MIXES):
                    if m0 + m < rows:
                        partials[m0 + m, pid_h, 2 + j] = dot[m, j]
                for m in T.Parallel(BLOCK_M):
                    if m0 + m < rows:
                        partials[m0 + m, pid_h, 0] = sqr[m]
                        partials[m0 + m, pid_h, 1] = ysq[m]

    return kernel


def _build_finalize(rows, hidden, splits, rms_eps, hc_eps, sinkhorn_iters, norm_eps):
    """Reduce split partials, produce the mixes, and normalize the collapsed input."""

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def kernel(
        partials: T.Tensor[(rows, splits, SLOTS), T.float32],
        scale: T.Tensor[(3,), T.float32],
        base: T.Tensor[(MIXES,), T.float32],
        norm_w: T.Tensor[(hidden,), T.bfloat16],
        y_io: T.Tensor[(rows, hidden), T.bfloat16],
        pre_out: T.Tensor[(rows, MULT), T.float32],
        post_out: T.Tensor[(rows, MULT), T.float32],
        comb_out: T.Tensor[(rows, MULT, MULT), T.float32],
    ):
        with T.Kernel(rows, threads=FINALIZE_THREADS) as pid:
            sums = T.alloc_shared((SLOTS,), T.float32)
            ynorm = T.alloc_shared((1,), T.float32)
            rms_s = T.alloc_shared((1,), T.float32)
            mixes = T.alloc_shared((MIXES,), T.float32)
            tid = T.get_thread_binding()

            if tid < SLOTS:
                acc = T.alloc_var(T.float32)
                acc = 0.0
                for sp in T.serial(splits):
                    acc = _fadd(acc, partials[pid, sp, tid])
                sums[tid] = acc
            T.sync_threads()

            if tid < 32:
                if tid == 0:
                    rms_s[0] = T.rsqrt(_fmul(sums[0], 1.0 / (MULT * hidden)) + rms_eps)
                    ynorm[0] = T.rsqrt(_fmul(sums[1], 1.0 / hidden) + norm_eps)
                T.sync_threads()
                for j in T.Parallel(MIXES):
                    mixes[j] = _fmul(sums[2 + j], rms_s[0])
                T.sync_threads()
                for j in T.Parallel(MULT):
                    pre_out[pid, j] = T.sigmoid(_fadd(_fmul(mixes[j], scale[0]), base[j])) + hc_eps
                    post_out[pid, j] = 2.0 * T.sigmoid(
                        _fadd(_fmul(mixes[j + MULT], scale[1]), base[j + MULT]))
                cm = T.alloc_fragment((MULT, MULT), T.float32)
                for r, c in T.Parallel(MULT, MULT):
                    cm[r, c] = _fadd(_fmul(mixes[MULT * 2 + r * MULT + c], scale[2]),
                                     base[MULT * 2 + r * MULT + c])
                row_max = T.alloc_fragment((MULT,), T.float32)
                row_sum = T.alloc_fragment((MULT,), T.float32)
                col_sum = T.alloc_fragment((MULT,), T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for r, c in T.Parallel(MULT, MULT):
                    cm[r, c] = T.exp(cm[r, c] - row_max[r])
                T.reduce_sum(cm, row_sum, dim=1)
                for r, c in T.Parallel(MULT, MULT):
                    cm[r, c] = _fadd(cm[r, c] / row_sum[r], hc_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for r, c in T.Parallel(MULT, MULT):
                    cm[r, c] = cm[r, c] / _fadd(col_sum[c], hc_eps)
                for _ in T.serial(sinkhorn_iters - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for r, c in T.Parallel(MULT, MULT):
                        cm[r, c] = cm[r, c] / _fadd(row_sum[r], hc_eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for r, c in T.Parallel(MULT, MULT):
                        cm[r, c] = cm[r, c] / _fadd(col_sum[c], hc_eps)
                for r, c in T.Parallel(MULT, MULT):
                    comb_out[pid, r, c] = cm[r, c]

            T.sync_threads()
            for h in T.Parallel(hidden):
                # Reference order: (y * rsqrt(ms + eps)) * norm_weight.
                y_io[pid, h] = T.Cast(T.bfloat16, _fmul(_fmul(T.Cast(T.float32, y_io[pid, h]),
                                                              ynorm[0]), T.Cast(T.float32, norm_w[h])))

    return kernel


def pack_projection(fn, hidden, block_h=BLOCK_H):
    """Return chunk-major FP16 high and low planes of the FP32 projection weights.

    Each plane is a permutation of the checkpoint matrix, so the checkpoint's
    precision is preserved to about 1e-7 relative by the shifted FP16 pair.
    """
    if fn.shape != (MIXES, MULT * hidden) or fn.dtype != torch.float32:
        raise ValueError("mHC projection weights require FP32 [24, 4*hidden]")
    if hidden % block_h:
        raise ValueError("mHC hidden size must be divisible by the hidden block")
    high = fn.half()
    low = ((fn - high.float()) * (1 << LOW_SHIFT)).half()
    splits = hidden // block_h

    def pack(values):
        columns = [torch.stack([values[:, i * hidden + s * block_h: i * hidden + (s + 1) * block_h].T
                                for i in range(MULT)]) for s in range(splits)]
        out = torch.zeros((splits, MULT * block_h, MIXES_PAD), dtype=torch.float16, device=fn.device)
        out[:, :, :MIXES] = torch.stack(columns).reshape(splits, MULT * block_h, MIXES)
        return out.contiguous()

    return pack(high), pack(low)


def is_blackwell_consumer(device) -> bool:
    """True for SM120 and SM121, whose tensor cores have no tcgen05 path."""
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) in ((12, 0), (12, 1))


def tiling(rows, device):
    """Return the hidden block for a prepared row count, or None to keep b12x.

    Every measured profile is keyed on the part's SM count, because that is what
    decides whether the fixed split grid fills it. Rows then decide whether the
    pair's split reduction is worth its cost at all.
    """
    if rows < MIN_ROWS or not is_blackwell_consumer(device):
        return None
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    if multiprocessors >= 128:
        # 160 CTAs already leave this part under-filled, so a coarser block only
        # removes parallelism. Measured 0.82x at 128 rows and 1.13x at 512.
        return BLOCK_H if rows <= 256 else None
    if multiprocessors <= 64:
        # 160 CTAs need 3.3 waves here and 80 need 1.7. Above 1024 rows halving
        # the split count pays for itself: the FP32 partials fall from 130 to
        # 65 MiB at 8192 rows and the finalize reduction from 160 to 80 terms.
        # Measured 0.88x at 1024 rows and 0.80-0.84x through 8192; below 64 rows
        # the row tiles are too few for that to matter and b12x wins the middle.
        if rows <= 64:
            return BLOCK_H
        return PREFILL_BLOCK_H if rows >= 1024 else None
    return None


_PACKED: "dict[int, tuple]" = {}


def packed_projection(fn, hidden, block_h=BLOCK_H):
    """Return the packed planes for one weight tensor and hidden block.

    The planes are a pure function of the weights, so every prepared shape that
    uses them shares one copy. Packing per prepared call instead would hold one
    pair per graph variant per layer for the life of the catalog.
    """
    entry = _PACKED.get(id(fn))
    if entry is not None and entry[0] is fn and entry[1] == hidden and entry[2] == block_h:
        return entry[3], entry[4]
    high, low = pack_projection(fn, hidden, block_h)
    # Holding the weight tensor keeps its identity, and so this key, valid.
    _PACKED[id(fn)] = (fn, hidden, block_h, high, low)
    return high, low


@dataclass(frozen=True)
class TileLangMHCConfig:
    """Selected execution geometry, reported by the component."""

    backend: str = "tilelang"
    block_h: int = BLOCK_H


@dataclass(frozen=True)
class MHCQuery:
    """Everything that determines the compiled kernel pair."""

    operation: str
    rows: int
    hidden: int
    block_h: int
    threads: int
    rms_eps: float
    hc_eps: float
    sinkhorn_iters: int
    norm_eps: float


def _tactics(query):
    return ((query.block_h, query.threads),)


def _contract():
    from ignition.components.native_operator import native_contract

    return native_contract(component_id="flare.tilelang_mhc", query_type=MHCQuery,
                           tactics=_tactics)


_CONTRACT = None


_OWNERS: "weakref.WeakValueDictionary[int, PreparedTileLangMHC]" = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@torch.library.custom_op("flare::tilelang_mhc",
                         mutates_args=("residual_out", "y", "partials", "pre_out", "post", "comb"))
def _run(handle: int, high: torch.Tensor, low: torch.Tensor, x: torch.Tensor,
         residual: torch.Tensor, post_in: torch.Tensor, comb_in: torch.Tensor,
         pre_in: torch.Tensor, residual_out: torch.Tensor, y: torch.Tensor,
         partials: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
         norm: torch.Tensor, pre_out: torch.Tensor, post: torch.Tensor,
         comb: torch.Tensor) -> None:
    _OWNERS[handle].run()


@_run.register_fake
def _run_fake(handle, high, low, x, residual, post_in, comb_in, pre_in, residual_out, y,
              partials, scale, base, norm, pre_out, post, comb) -> None:
    pass


class _State:
    """Compiled TileLang pair for one prepared shape."""

    def __init__(self, query):
        self.partial = _build_partial(query.rows, query.hidden, query.block_h, query.threads,
                                      query.operation == "pre")
        self.finalize = _build_finalize(query.rows, query.hidden, query.hidden // query.block_h,
                                        query.rms_eps, query.hc_eps, query.sinkhorn_iters,
                                        query.norm_eps)

    def run(self, high, low, x, residual, post_in, comb_in, pre_in, residual_out, y, partials,
            scale, base, norm, pre_out, post, comb):
        self.partial(high, low, x, residual, post_in, comb_in, pre_in, residual_out, y, partials)
        self.finalize(partials, scale, base, norm, y, pre_out, post, comb)


class PreparedTileLangMHC:
    """Fused pre or post/pre for SM120/SM121, with lagged mixing coefficients.

    The pair declares one native plan per prepared shape, so its split partials
    come from the shared scratch arena and the kernels compile during the
    preparation batch rather than in the recipe.
    """

    def __init__(self, inputs, weights, scratch, *, session, name, workspace, rms_eps, hc_eps,
                 sinkhorn_iters, norm_eps, pre_mix, pre_out, block_h=BLOCK_H, threads=THREADS):
        from b12x.preparation import PreparedCall
        from b12x.preparation.types import require_prepared

        from ignition.components.native_operator import native_contract, native_plan
        from ignition.preparation.scratch import planned_scratch
        from ignition.preparation.tuning import declare, register_finalizer

        if len(inputs) == 1:
            self.operation = "pre"
            residual = inputs[0]
        elif len(inputs) == 4:
            self.operation = "post_pre"
            residual = inputs[1]
        else:
            raise ValueError("mHC requires residual or branch/residual/post/comb inputs")
        if residual.ndim != 3 or residual.shape[1] != MULT or residual.dtype != torch.bfloat16:
            raise ValueError("mHC residual requires BF16 [tokens, 4, hidden]")
        if pre_mix is None or pre_out is None:
            raise ValueError("the TileLang mHC path requires lagged mixing coefficients")
        rows, _, hidden = residual.shape
        if rows <= 0 or hidden % block_h or hidden % BLOCK_M:
            raise ValueError("mHC requires positive rows and a hidden size divisible by the block")
        splits = hidden // block_h
        global _CONTRACT
        if _CONTRACT is None:
            _CONTRACT = _contract()
        query = MHCQuery(self.operation, rows, hidden, block_h, threads, float(rms_eps),
                         float(hc_eps), int(sinkhorn_iters), float(norm_eps))
        self.plan = native_plan(
            contract=_CONTRACT, query=query, device=residual.device,
            scratch_bytes=lambda tactic: rows * splits * SLOTS * 4,
            materialize=lambda tactic: _State(query))
        self.config = TileLangMHCConfig(block_h=block_h)
        self.scratch, self.pre_mix, self.pre_out = scratch, pre_mix, pre_out
        self.scale, self.base = weights.scale, weights.base
        if weights.norm.dtype != torch.bfloat16:
            raise ValueError("the TileLang mHC path applies a BF16 norm weight")
        self.norm = weights.norm
        self.high, self.low = packed_projection(weights.fn, hidden, block_h)
        if self.operation == "pre":
            # The pre operation reads no branch input and no mixing coefficients;
            # caller-owned buffers supply well-typed operands its compiled branch
            # never loads.
            self._source = scratch.normalized
            self._post, self._comb = scratch.post, scratch.comb
        else:
            self._source, self._post, self._comb = inputs[0], inputs[2], inputs[3]
        self.partials = None
        self._residual = residual
        plan = self.plan
        state = {}
        self._state = state
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

        def prepare_call(prepared):
            state["prepared"] = prepared
            storage = torch.empty((rows, splits, SLOTS), dtype=torch.float32, device=residual.device)
            self.partials = storage
            return PreparedCall(run=self.run)

        declare(session, (plan.request(name=name, prepare_call=prepare_call),))

        def finalize():
            state["prepared"] = require_prepared(plan, plan.component_id, residual.device)
            self.partials = planned_scratch(plan.scratch_specs(), workspace).view(torch.float32)
            self.partials = self.partials.view(rows, splits, SLOTS)

        register_finalizer(session, finalize, workspace=workspace)

    def run(self):
        prepared = self._state["prepared"]
        prepared.run(self.high, self.low, self._source, self._residual, self._post, self._comb,
                     self.pre_mix, self.scratch.residual, self.scratch.normalized, self.partials,
                     self.scale, self.base, self.norm, self.pre_out, self.scratch.post,
                     self.scratch.comb)

    def __call__(self, *inputs):
        residual = inputs[0] if self.operation == "pre" else inputs[1]
        self._residual = residual
        _run(self.handle, self.high, self.low, self._source, residual, self._post, self._comb,
             self.pre_mix, self.scratch.residual, self.scratch.normalized, self.partials,
             self.scale, self.base, self.norm, self.pre_out, self.scratch.post, self.scratch.comb)
        return self.scratch.normalized
