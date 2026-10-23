"""Gemma 4 routed GeGLU MoE with BF16 and NVFP4 expert execution."""

import math
import torch
import triton
import triton.language as tl

from ignition.components.collectives import PreparedAllReduce
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.triton import prepare_triton

from .attention import prepare_projection, rms_norm, rms_norm_no_scale
from .weights import BF16ExpertsWeight, NVFP4ExpertsWeight

BLOCK_M = 16
BLOCK_N = 64
BLOCK_K = 64


@triton.jit
def _init_bins_kernel(
    expert_counts,
    tile_expert_ids,
    sorted_pair_ids,
    NUM_EXPERTS: tl.constexpr,
    MAX_TILES: tl.constexpr,
    MAX_PADDED_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_pair_ids + offsets, -1, mask=offsets < MAX_PADDED_ROWS)
    tl.store(tile_expert_ids + offsets, -1, mask=offsets < MAX_TILES)
    tl.store(expert_counts + offsets, 0, mask=offsets < NUM_EXPERTS)


@triton.jit
def _count_routes_kernel(
    top_k_index,
    num_valid_tokens,
    expert_counts,
    pair_ranks,
    TOP_K: tl.constexpr,
    TOTAL_PAIRS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair_ids = pid * BLOCK + tl.arange(0, BLOCK)
    valid_tokens = tl.load(num_valid_tokens).to(tl.int32)
    valid_pairs = valid_tokens * TOP_K
    mask = (pair_ids < TOTAL_PAIRS) & (pair_ids < valid_pairs)
    expert = tl.load(top_k_index + pair_ids, mask=mask, other=0).to(tl.int32)
    ranks = tl.atomic_add(expert_counts + expert, 1, mask=mask)
    tl.store(pair_ranks + pair_ids, ranks, mask=mask)


@triton.jit
def _scan_experts_kernel(
    expert_counts,
    expert_tile_starts,
    expert_tile_counts,
    NUM_EXPERTS: tl.constexpr,
    TILE_M: tl.constexpr,
):
    experts = tl.arange(0, NUM_EXPERTS)
    counts = tl.load(expert_counts + experts).to(tl.int32)
    tiles = (counts + TILE_M - 1) // TILE_M
    starts = tl.cumsum(tiles, axis=0) - tiles
    tl.store(expert_tile_starts + experts, starts)
    tl.store(expert_tile_counts + experts, tiles)


@triton.jit
def _scatter_routes_kernel(
    top_k_index,
    num_valid_tokens,
    pair_ranks,
    expert_tile_starts,
    sorted_pair_ids,
    tile_expert_ids,
    TOP_K: tl.constexpr,
    TOTAL_PAIRS: tl.constexpr,
    TILE_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair_ids = pid * BLOCK + tl.arange(0, BLOCK)
    valid_tokens = tl.load(num_valid_tokens).to(tl.int32)
    valid_pairs = valid_tokens * TOP_K
    mask = (pair_ids < TOTAL_PAIRS) & (pair_ids < valid_pairs)
    expert = tl.load(top_k_index + pair_ids, mask=mask, other=0).to(tl.int32)
    rank = tl.load(pair_ranks + pair_ids, mask=mask, other=0).to(tl.int32)
    tile_start = tl.load(expert_tile_starts + expert, mask=mask, other=0).to(tl.int32)
    dest = tile_start * TILE_M + rank
    tl.store(sorted_pair_ids + dest, pair_ids, mask=mask)
    tile_idx = tile_start + (rank // TILE_M)
    tl.store(tile_expert_ids + tile_idx, expert, mask=mask)


@triton.jit
def _gelu_tanh(x):
    c = 0.7978845608028654
    k = 0.044715
    inner = c * (x + k * x * x * x)
    return 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))


@triton.jit
def _dequant_e2m1(nibble):
    mag = nibble & 7
    sign = tl.where(nibble >= 8, -1.0, 1.0)
    mag_f = mag.to(tl.float32)
    val = tl.where(
        mag <= 4,
        0.5 * mag_f,
        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
    )
    return val * sign


@triton.jit
def _moe_bf16_stage1_kernel(
    source,
    gate_up_proj,
    sorted_pair_ids,
    tile_expert_ids,
    intermediate,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_id = tl.program_id(1)
    expert = tl.load(tile_expert_ids + tile_id).to(tl.int64)
    if expert < 0:
        return
    m_offs = tl.arange(0, TILE_M)
    n_offs = n_id * TILE_N + tl.arange(0, TILE_N)
    n_mask = n_offs < INTER
    pairs = tl.load(sorted_pair_ids + tile_id * TILE_M + m_offs).to(tl.int64)
    m_mask = pairs >= 0
    tokens = tl.where(m_mask, pairs // TOP_K, 0)

    gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    up_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    expert_base = gate_up_proj + expert * (2 * INTER) * HIDDEN

    for k_start in range(0, HIDDEN, TILE_K):
        k_offs = k_start + tl.arange(0, TILE_K)
        k_mask = k_offs < HIDDEN
        x_tile = tl.load(
            source + tokens[:, None] * HIDDEN + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_gate = tl.load(
            expert_base + n_offs[None, :] * HIDDEN + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        w_up = tl.load(
            expert_base + (INTER + n_offs)[None, :] * HIDDEN + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        gate_acc += tl.dot(x_tile, w_gate)
        up_acc += tl.dot(x_tile, w_up)

    gate_val = _gelu_tanh(gate_acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    up_val = up_acc.to(tl.bfloat16).to(tl.float32)
    out = (gate_val * up_val).to(tl.bfloat16)
    safe_pairs = tl.where(m_mask, pairs, 0)
    tl.store(
        intermediate + safe_pairs[:, None] * INTER + n_offs[None, :],
        out,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _moe_bf16_stage2_kernel(
    intermediate,
    down_proj,
    top_k_weights,
    sorted_pair_ids,
    tile_expert_ids,
    routed_out,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_id = tl.program_id(1)
    expert = tl.load(tile_expert_ids + tile_id).to(tl.int64)
    if expert < 0:
        return
    m_offs = tl.arange(0, TILE_M)
    n_offs = n_id * TILE_N + tl.arange(0, TILE_N)
    n_mask = n_offs < HIDDEN
    pairs = tl.load(sorted_pair_ids + tile_id * TILE_M + m_offs).to(tl.int64)
    m_mask = pairs >= 0
    safe_pairs = tl.where(m_mask, pairs, 0)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    expert_base = down_proj + expert * HIDDEN * INTER

    for k_start in range(0, INTER, TILE_K):
        k_offs = k_start + tl.arange(0, TILE_K)
        k_mask = k_offs < INTER
        x_tile = tl.load(
            intermediate + safe_pairs[:, None] * INTER + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_down = tl.load(
            expert_base + n_offs[None, :] * INTER + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x_tile, w_down)

    weights = tl.load(top_k_weights + safe_pairs, mask=m_mask, other=0.0).to(tl.float32)
    out = (acc.to(tl.bfloat16).to(tl.float32) * weights[:, None]).to(tl.bfloat16)
    tl.store(
        routed_out + safe_pairs[:, None] * HIDDEN + n_offs[None, :],
        out,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _moe_nvfp4_stage1_kernel(
    source,
    gate_up_weight,
    gate_up_scale,
    gate_up_scale_2,
    sorted_pair_ids,
    tile_expert_ids,
    intermediate,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_id = tl.program_id(1)
    expert = tl.load(tile_expert_ids + tile_id).to(tl.int64)
    if expert < 0:
        return
    m_offs = tl.arange(0, TILE_M)
    n_offs = n_id * TILE_N + tl.arange(0, TILE_N)
    n_mask = n_offs < INTER
    pairs = tl.load(sorted_pair_ids + tile_id * TILE_M + m_offs).to(tl.int64)
    m_mask = pairs >= 0
    tokens = tl.where(m_mask, pairs // TOP_K, 0)

    gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    up_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    HALF_K: tl.constexpr = TILE_K // 2
    PACKED_H: tl.constexpr = HIDDEN // 2
    SCALE_H: tl.constexpr = HIDDEN // 16

    w_base = gate_up_weight + expert * (2 * INTER) * PACKED_H
    s_base = gate_up_scale + expert * (2 * INTER) * SCALE_H
    g_gate = tl.load(gate_up_scale_2 + expert * 2).to(tl.float32)
    g_up = tl.load(gate_up_scale_2 + expert * 2 + 1).to(tl.float32)
    j_offs = tl.arange(0, HALF_K)

    for k_start in range(0, HIDDEN, TILE_K):
        k_even = k_start + 2 * j_offs
        k_odd = k_even + 1
        x_even = tl.load(
            source + tokens[:, None] * HIDDEN + k_even[None, :],
            mask=m_mask[:, None] & (k_even[None, :] < HIDDEN),
            other=0.0,
        )
        x_odd = tl.load(
            source + tokens[:, None] * HIDDEN + k_odd[None, :],
            mask=m_mask[:, None] & (k_odd[None, :] < HIDDEN),
            other=0.0,
        )
        byte_offs = (k_start // 2) + j_offs
        scale_offs = (k_start // 16) + (j_offs // 8)
        byte_mask = (byte_offs[:, None] < PACKED_H) & n_mask[None, :]

        packed_gate = tl.load(
            w_base + n_offs[None, :] * PACKED_H + byte_offs[:, None],
            mask=byte_mask,
            other=0,
        )
        scale_gate = (
            tl.load(
                s_base + n_offs[None, :] * SCALE_H + scale_offs[:, None],
                mask=byte_mask,
                other=0.0,
            ).to(tl.float32)
            * g_gate
        )
        w_gate_even = (_dequant_e2m1(packed_gate & 0x0F) * scale_gate).to(tl.bfloat16)
        w_gate_odd = (_dequant_e2m1(packed_gate >> 4) * scale_gate).to(tl.bfloat16)
        gate_acc += tl.dot(x_even, w_gate_even) + tl.dot(x_odd, w_gate_odd)

        packed_up = tl.load(
            w_base + (INTER + n_offs)[None, :] * PACKED_H + byte_offs[:, None],
            mask=byte_mask,
            other=0,
        )
        scale_up = (
            tl.load(
                s_base + (INTER + n_offs)[None, :] * SCALE_H + scale_offs[:, None],
                mask=byte_mask,
                other=0.0,
            ).to(tl.float32)
            * g_up
        )
        w_up_even = (_dequant_e2m1(packed_up & 0x0F) * scale_up).to(tl.bfloat16)
        w_up_odd = (_dequant_e2m1(packed_up >> 4) * scale_up).to(tl.bfloat16)
        up_acc += tl.dot(x_even, w_up_even) + tl.dot(x_odd, w_up_odd)

    gate_val = _gelu_tanh(gate_acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    up_val = up_acc.to(tl.bfloat16).to(tl.float32)
    out = (gate_val * up_val).to(tl.bfloat16)
    safe_pairs = tl.where(m_mask, pairs, 0)
    tl.store(
        intermediate + safe_pairs[:, None] * INTER + n_offs[None, :],
        out,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _moe_nvfp4_stage2_kernel(
    intermediate,
    down_weight,
    down_scale,
    down_scale_2,
    top_k_weights,
    sorted_pair_ids,
    tile_expert_ids,
    routed_out,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_id = tl.program_id(1)
    expert = tl.load(tile_expert_ids + tile_id).to(tl.int64)
    if expert < 0:
        return
    m_offs = tl.arange(0, TILE_M)
    n_offs = n_id * TILE_N + tl.arange(0, TILE_N)
    n_mask = n_offs < HIDDEN
    pairs = tl.load(sorted_pair_ids + tile_id * TILE_M + m_offs).to(tl.int64)
    m_mask = pairs >= 0
    safe_pairs = tl.where(m_mask, pairs, 0)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    HALF_K: tl.constexpr = TILE_K // 2
    PACKED_I: tl.constexpr = INTER // 2
    SCALE_I: tl.constexpr = INTER // 16

    w_base = down_weight + expert * HIDDEN * PACKED_I
    s_base = down_scale + expert * HIDDEN * SCALE_I
    g_down = tl.load(down_scale_2 + expert).to(tl.float32)
    j_offs = tl.arange(0, HALF_K)

    for k_start in range(0, INTER, TILE_K):
        k_even = k_start + 2 * j_offs
        k_odd = k_even + 1
        x_even = tl.load(
            intermediate + safe_pairs[:, None] * INTER + k_even[None, :],
            mask=m_mask[:, None] & (k_even[None, :] < INTER),
            other=0.0,
        )
        x_odd = tl.load(
            intermediate + safe_pairs[:, None] * INTER + k_odd[None, :],
            mask=m_mask[:, None] & (k_odd[None, :] < INTER),
            other=0.0,
        )
        byte_offs = (k_start // 2) + j_offs
        scale_offs = (k_start // 16) + (j_offs // 8)
        byte_mask = (byte_offs[:, None] < PACKED_I) & n_mask[None, :]

        packed_down = tl.load(
            w_base + n_offs[None, :] * PACKED_I + byte_offs[:, None],
            mask=byte_mask,
            other=0,
        )
        scale_down = (
            tl.load(
                s_base + n_offs[None, :] * SCALE_I + scale_offs[:, None],
                mask=byte_mask,
                other=0.0,
            ).to(tl.float32)
            * g_down
        )
        w_down_even = (_dequant_e2m1(packed_down & 0x0F) * scale_down).to(tl.bfloat16)
        w_down_odd = (_dequant_e2m1(packed_down >> 4) * scale_down).to(tl.bfloat16)
        acc += tl.dot(x_even, w_down_even) + tl.dot(x_odd, w_down_odd)

    weights = tl.load(top_k_weights + safe_pairs, mask=m_mask, other=0.0).to(tl.float32)
    out = (acc.to(tl.bfloat16).to(tl.float32) * weights[:, None]).to(tl.bfloat16)
    tl.store(
        routed_out + safe_pairs[:, None] * HIDDEN + n_offs[None, :],
        out,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def dequantize_nvfp4_tensor(
    weight: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor
) -> torch.Tensor:
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    low = lut[(weight & 0x0F).long()]
    high = lut[(weight >> 4).long()]
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    block_scale = scale.float().repeat_interleave(16, dim=-1)
    gain = scale_2.float()
    while gain.ndim < unpacked.ndim:
        gain = gain.unsqueeze(-1)
    return (unpacked * block_scale * gain).to(torch.bfloat16)


def geglu_moe_reference(
    residual: torch.Tensor,
    pre_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    router_scale: torch.Tensor,
    per_expert_scale: torch.Tensor,
    router_proj_weight: torch.Tensor,
    experts: BF16ExpertsWeight | NVFP4ExpertsWeight,
    *,
    top_k: int,
    eps: float,
) -> torch.Tensor:
    hidden_size = residual.shape[-1]
    router_in = rms_norm_no_scale(residual, eps) * router_scale * (hidden_size**-0.5)
    scores = torch.nn.functional.linear(router_in, router_proj_weight)
    probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    top_k_weights, top_k_index = torch.topk(probs, k=top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights * per_expert_scale[top_k_index]

    normed = rms_norm(residual, pre_norm_weight, eps)
    out = torch.zeros_like(normed)
    if isinstance(experts, BF16ExpertsWeight):
        gate_up_all = experts.gate_up_proj
        down_all = experts.down_proj
    else:
        inter = experts.down_weight.shape[-1] * 2
        gate_w = dequantize_nvfp4_tensor(
            experts.gate_up_weight[:, :inter],
            experts.gate_up_scale[:, :inter],
            experts.gate_up_scale_2[:, 0],
        )
        up_w = dequantize_nvfp4_tensor(
            experts.gate_up_weight[:, inter:],
            experts.gate_up_scale[:, inter:],
            experts.gate_up_scale_2[:, 1],
        )
        gate_up_all = torch.cat((gate_w, up_w), dim=1)
        down_all = dequantize_nvfp4_tensor(experts.down_weight, experts.down_scale, experts.down_scale_2)

    for token_idx in range(normed.shape[0]):
        x = normed[token_idx : token_idx + 1]
        accum = torch.zeros_like(x)
        for k_pos in range(top_k):
            expert_idx = int(top_k_index[token_idx, k_pos].item())
            weight = top_k_weights[token_idx, k_pos].to(x.dtype)
            gate, up = torch.nn.functional.linear(x, gate_up_all[expert_idx]).chunk(2, dim=-1)
            act = torch.nn.functional.gelu(gate, approximate="tanh") * up
            proj = torch.nn.functional.linear(act, down_all[expert_idx])
            accum = accum + (proj * weight).to(x.dtype)
        out[token_idx] = accum[0]
    return rms_norm(out, post_norm_weight, eps)


class PreparedMoE:
    def __init__(
        self,
        weights: dict,
        config,
        source: torch.Tensor,
        num_valid_tokens: torch.Tensor,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
        runtime=None,
        coordinator=None,
        session,
        name: str,
        workspace,
        buffers,
    ):
        self.config = config
        self.eps = config.rms_norm_eps
        self.hidden = config.hidden
        start, end = config.moe_intermediate_range(shard)
        self.inter = end - start
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.root_scale = math.sqrt(config.hidden) ** -1
        self.pre_norm = weights["pre_feedforward_layernorm_2"]
        self.post_norm = weights["post_feedforward_layernorm_2"]
        self.router_scale = weights["router_scale"]
        self.per_expert_scale = weights["per_expert_scale"]
        self.num_valid_tokens = num_valid_tokens
        tokens = source.shape[0]
        total_pairs = tokens * self.top_k
        max_tiles = min(total_pairs, triton.cdiv(total_pairs, BLOCK_M) + self.num_experts)
        max_padded_rows = max_tiles * BLOCK_M

        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)

        self.router_input = zeros((tokens, self.hidden))
        self.router_logits = zeros((tokens, self.num_experts))
        self.router_proj = prepare_projection(
            self.router_input,
            weights["router_proj"],
            self.router_logits,
            session=session,
            name=name + ".router_proj",
            workspace=workspace,
        )
        self.moe_input = zeros((tokens, self.hidden))
        self.top_k_index = zeros((total_pairs,), torch.int32)
        self.top_k_weights = zeros((total_pairs,), torch.float32)
        self.expert_counts = zeros((self.num_experts,), torch.int32)
        self.expert_tile_starts = zeros((self.num_experts,), torch.int32)
        self.expert_tile_counts = zeros((self.num_experts,), torch.int32)
        self.pair_ranks = zeros((total_pairs,), torch.int32)
        self.sorted_pair_ids = zeros((max_padded_rows,), torch.int32)
        self.tile_expert_ids = zeros((max_tiles,), torch.int32)
        self.intermediate = zeros((total_pairs, self.inter))
        self.routed_out = zeros((total_pairs, self.hidden))
        self.partial_out = zeros((tokens, self.hidden))
        self.reduced_out = zeros((tokens, self.hidden))
        self.reduce = PreparedAllReduce(
            self.partial_out,
            self.reduced_out,
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".reduce",
        )

        init_cap = max(max_padded_rows, max_tiles, self.num_experts)
        self.init_bins = prepare_triton(
            _init_bins_kernel,
            (
                self.expert_counts,
                self.tile_expert_ids,
                self.sorted_pair_ids,
                self.num_experts,
                max_tiles,
                max_padded_rows,
                256,
            ),
            (triton.cdiv(init_cap, 256), 1, 1),
            num_warps=4,
        )
        self.count_routes = prepare_triton(
            _count_routes_kernel,
            (
                self.top_k_index,
                self.num_valid_tokens,
                self.expert_counts,
                self.pair_ranks,
                self.top_k,
                total_pairs,
                256,
            ),
            (triton.cdiv(total_pairs, 256), 1, 1),
            num_warps=4,
        )
        self.scan_experts = prepare_triton(
            _scan_experts_kernel,
            (
                self.expert_counts,
                self.expert_tile_starts,
                self.expert_tile_counts,
                self.num_experts,
                BLOCK_M,
            ),
            (1, 1, 1),
            num_warps=4,
        )
        self.scatter_routes = prepare_triton(
            _scatter_routes_kernel,
            (
                self.top_k_index,
                self.num_valid_tokens,
                self.pair_ranks,
                self.expert_tile_starts,
                self.sorted_pair_ids,
                self.tile_expert_ids,
                self.top_k,
                total_pairs,
                BLOCK_M,
                256,
            ),
            (triton.cdiv(total_pairs, 256), 1, 1),
            num_warps=4,
        )

        experts = weights["experts"]
        if isinstance(experts, BF16ExpertsWeight):
            self.stage1 = prepare_triton(
                _moe_bf16_stage1_kernel,
                (
                    self.moe_input,
                    experts.gate_up_proj,
                    self.sorted_pair_ids,
                    self.tile_expert_ids,
                    self.intermediate,
                    self.top_k,
                    self.hidden,
                    self.inter,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                (max_tiles, triton.cdiv(self.inter, BLOCK_N), 1),
                num_warps=4,
            )
            self.stage2 = prepare_triton(
                _moe_bf16_stage2_kernel,
                (
                    self.intermediate,
                    experts.down_proj,
                    self.top_k_weights,
                    self.sorted_pair_ids,
                    self.tile_expert_ids,
                    self.routed_out,
                    self.hidden,
                    self.inter,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                (max_tiles, triton.cdiv(self.hidden, BLOCK_N), 1),
                num_warps=4,
            )
        else:
            self.stage1 = prepare_triton(
                _moe_nvfp4_stage1_kernel,
                (
                    self.moe_input,
                    experts.gate_up_weight,
                    experts.gate_up_scale,
                    experts.gate_up_scale_2,
                    self.sorted_pair_ids,
                    self.tile_expert_ids,
                    self.intermediate,
                    self.top_k,
                    self.hidden,
                    self.inter,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                (max_tiles, triton.cdiv(self.inter, BLOCK_N), 1),
                num_warps=4,
            )
            self.stage2 = prepare_triton(
                _moe_nvfp4_stage2_kernel,
                (
                    self.intermediate,
                    experts.down_weight,
                    experts.down_scale,
                    experts.down_scale_2,
                    self.top_k_weights,
                    self.sorted_pair_ids,
                    self.tile_expert_ids,
                    self.routed_out,
                    self.hidden,
                    self.inter,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                (max_tiles, triton.cdiv(self.hidden, BLOCK_N), 1),
                num_warps=4,
            )

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        tokens = residual.shape[0]
        r_in = rms_norm_no_scale(residual, self.eps) * self.router_scale * self.root_scale
        self.router_input.copy_(r_in)
        self.router_proj(self.router_input)
        probs = torch.softmax(self.router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        self.top_k_index.copy_(top_k_index.reshape(-1).to(torch.int32))
        self.top_k_weights.copy_(top_k_weights.reshape(-1).to(torch.float32))
        self.moe_input.copy_(rms_norm(residual, self.pre_norm, self.eps))

        self.init_bins()
        self.count_routes()
        self.scan_experts()
        self.scatter_routes()
        self.stage1()
        self.stage2()

        self.partial_out.copy_(self.routed_out.view(tokens, self.top_k, self.hidden).sum(dim=1))
        reduced = self.reduce(self.partial_out)
        return rms_norm(reduced, self.post_norm, self.eps)
