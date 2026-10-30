"""Normalized gated attention with paged FP8 and accepted circular histories."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.window_attention import stage_windows, commit_window

from .pool import KV_DTYPE, PAGE_TOKENS


@dataclass(frozen=True)
class AttentionScratch:
    projected: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    gated: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, requests=1, buffers=None):
        geometries = [config.attention(i) for i in range(config.layers)]
        if len({(a.heads, a.kv_heads, a.key_dim, a.value_dim) for a in geometries}) != 1:
            raise ValueError("Muse Spark attention scratch requires matching global and sliding head geometry")
        geometry = geometries[0]
        heads, kv_heads = geometry.local(shard)
        def zeros(*shape, dtype=torch.bfloat16):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        window = max(a.window for a in geometries)
        pages = max((tokens + max(window, 0) + PAGE_TOKENS - 1) // PAGE_TOKENS,
                    requests * ((max(window, 0) + 8 + PAGE_TOKENS - 1) // PAGE_TOKENS))
        return cls(zeros(tokens, 2 * heads * geometry.key_dim + kv_heads * (geometry.key_dim + geometry.value_dim)),
            zeros(tokens, heads, geometry.key_dim), zeros(tokens, kv_heads, geometry.key_dim),
            zeros(tokens, kv_heads, geometry.value_dim), zeros(tokens, heads, geometry.value_dim),
            zeros(tokens, heads, geometry.value_dim),
            zeros(tokens, config.text["hidden_size"]), zeros(tokens, config.text["hidden_size"]),
            zeros(pages, PAGE_TOKENS, kv_heads, geometry.key_dim, dtype=KV_DTYPE),
            zeros(pages, PAGE_TOKENS, kv_heads, geometry.value_dim, dtype=KV_DTYPE))

    def rows(self, tokens):
        return type(self)(*(getattr(self, name) if name.startswith("window_") else getattr(self, name)[:tokens]
                            for name in self.__dataclass_fields__))


def head_norm(values, eps):
    value = values.float()
    return (value * torch.pow(value.square().mean(-1, keepdim=True) + eps, -0.5)).to(values.dtype)


class PreparedAttention(torch.nn.Module):
    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 runtime, coordinator, session, name, workspace, buffers=None):
        super().__init__()
        self.geometry, self.scratch, self.metadata = config.attention(layer), scratch, metadata
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError(f"unsupported Muse Spark attention mode: {mode}")
        self.mode, self.pool, self.layer = mode, pool, layer
        self.projection = UnquantizedProjection(source, weights.qkv, scratch.projected,
            session=session, name=name + ".qkv_gate")
        self.output_projection = UnquantizedProjection(scratch.gated.flatten(1), weights.output, scratch.partial,
            session=session, name=name + ".output")
        self.reduction = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime,
                                           coordinator=coordinator, session=session, name=name + ".reduce")
        geometry = self.geometry
        self.inverse = (1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
            dtype=torch.float32, device=source.device) / geometry.rotary_dim))
            if geometry.rotary_dim else torch.empty(0, dtype=torch.float32, device=source.device))
        self.eps = config.text["rms_norm_eps"]
        self.query_factor = config.text["qk_scale_factor"]
        self.lengths = (torch.zeros_like(metadata.lengths) if buffers is None
                        else buffers.zeros(metadata.lengths.shape, metadata.lengths.dtype))
        self.slots = (torch.full_like(metadata.positions, -1) if buffers is None
                      else buffers.zeros(metadata.positions.shape, metadata.positions.dtype).fill_(-1))
        self.key_cache, self.value_cache = pool.cache(layer, "key"), pool.cache(layer, "value")
        self.commit_writer = None
        window = -1
        if geometry.window > 0:
            if mode in ("prefill", "verify"):
                key, value = scratch.window_key, scratch.window_value
                rows = metadata.lengths.numel()
                if mode == "verify":
                    pages = (geometry.window + metadata.max_query_tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
                    key, value = key[:rows * pages], value[:rows * pages]
                    self.retained_key = (torch.zeros_like(scratch.key) if buffers is None
                                         else buffers.zeros(scratch.key.shape, scratch.key.dtype))
                    self.retained_value = (torch.zeros_like(scratch.value) if buffers is None
                                           else buffers.zeros(scratch.value.shape, scratch.value.dtype))
                    self.commit_writer = PreparedPagedKVWriter(self.retained_key, self.retained_value,
                        self.key_cache, self.value_cache, self.slots, session=session, name=name + ".commit")
                self.staged_key, self.staged_value = key, value
                self.pages = torch.arange(key.shape[0], dtype=torch.int32, device=source.device).view(rows, -1)
                window = geometry.window - 1
            else:
                key, value = pool.window_pages(layer, "key"), pool.window_pages(layer, "value")
                self.window_row_stride = pool.window_page_stride(layer)
                self.window_columns = torch.arange(geometry.window // PAGE_TOKENS,
                    dtype=torch.int32, device=source.device)
                self.pages = torch.zeros((metadata.lengths.numel(), self.window_columns.numel()),
                    dtype=torch.int32, device=source.device)
        else:
            key, value, self.pages = self.key_cache, self.value_cache, metadata.page_table
        self.writer = None
        if geometry.window < 0 or mode == "decode":
            self.writer = PreparedPagedKVWriter(scratch.key, scratch.value, key, value, self.slots,
                                                  session=session, name=name + ".cache_write")
        if geometry.window > 0:
            tuning_page_count = (key.shape[0] if mode in ("prefill", "verify") else
                                 (int(pool.memory_capacity[0].item()) - 1) * self.window_row_stride + geometry.window // PAGE_TOKENS)
        else:
            tuning_page_count = int(pool.memory_capacity[1].item())
        self.attention = PreparedPagedAttention(scratch.query, key, value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, mode=mode, window=window,
            tuning_page_count=tuning_page_count,
            width=1 if mode == "prefill" else metadata.max_query_tokens,
            session=session, workspace=workspace, name=name + ".attention")

    def _rotary(self, values, cos, sin):
        rotary, tail = values.split((self.geometry.rotary_dim, values.shape[-1] - self.geometry.rotary_dim), -1)
        left, right = rotary.chunk(2, -1)
        rotated = torch.cat((-right, left), -1)
        return torch.cat((rotary * cos[:, None] + rotated * sin[:, None], tail), -1)

    def forward(self, hidden_states, **kwargs):
        s, m, a = self.scratch, self.metadata, self.geometry
        projected = self.projection(hidden_states.reshape(-1, hidden_states.shape[-1]))
        heads, kv_heads = s.query.shape[1], s.key.shape[1]
        q, k, v, gate = projected.split((heads * a.key_dim, kv_heads * a.key_dim,
                                        kv_heads * a.value_dim, heads * a.value_dim), -1)
        q = head_norm(q.view_as(s.query), self.eps) * self.query_factor
        k = head_norm(k.view_as(s.key), self.eps)
        if a.rotary_dim:
            frequencies = m.positions.clamp_min(0).float()[:, None] * self.inverse[None]
            angles = torch.cat((frequencies, frequencies), -1)
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
        s.query.copy_(q)
        s.key.copy_(k)
        s.value.copy_(v.view_as(s.value))
        if a.window < 0:
            columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
            rows = torch.arange(m.positions.numel(), device=m.positions.device)
            pages = m.token_page_table[rows, columns]
            self.slots.copy_(torch.where(m.positions >= 0, pages.long() * PAGE_TOKENS
                                        + m.positions.remainder(PAGE_TOKENS), -1))
            self.lengths.copy_((m.starts + m.lengths).int())
        elif self.mode == "decode":
            self.pages.copy_(m.state_rows[:, None] * self.window_row_stride + self.window_columns[None])
            self.lengths.copy_((m.starts + m.lengths).clamp(max=a.window).int())
            self.slots.copy_(torch.where(m.positions >= 0,
                m.state_rows[m.requests] * self.window_row_stride * PAGE_TOKENS + m.positions.remainder(a.window), -1))
        elif self.mode == "prefill":
            stage_windows(self.key_cache, s.key, s.window_key, m)
            stage_windows(self.value_cache, s.value, s.window_value, m)
            self.lengths.copy_(m.starts.clamp(max=a.window).int() + m.lengths)
        else:
            self.retained_key.copy_(s.key)
            self.retained_value.copy_(s.value)
            stage_windows(self.key_cache, s.key, self.staged_key, m)
            stage_windows(self.value_cache, s.value, self.staged_value, m)
            self.lengths.copy_(m.starts.clamp(max=a.window).int() + m.lengths)
        if self.writer is not None:
            self.writer()
        self.attention()
        if a.window > 0 and self.mode == "prefill":
            commit_window(self.key_cache, s.key, m.state_rows[0], m.starts[0], m.lengths[0])
            commit_window(self.value_cache, s.value, m.state_rows[0], m.starts[0], m.lengths[0])
        s.gated.copy_(head_norm(s.attended, self.eps) * torch.sigmoid(gate.view_as(s.gated)))
        s.partial.copy_(self.output_projection(s.gated.flatten(1)))
        return self.reduction(s.partial).view_as(hidden_states), None

    def commit(self, counts):
        if self.commit_writer is not None:
            m, window = self.metadata, self.geometry.window
            offset = m.positions - m.starts[m.requests]
            self.slots.copy_(torch.where((m.positions >= 0) & (offset >= 0) & (offset < counts[m.requests]),
                m.state_rows[m.requests] * window + m.positions.remainder(window), -1))
            self.commit_writer()

    def scratch_tensors(self):
        retained = () if self.commit_writer is None else (self.retained_key, self.retained_value)
        return self.slots, self.lengths, *retained
