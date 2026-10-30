"""Unrotated grouped-query attention with paged BF16 or FP8 history."""

import torch

from ignition.components.linear import QuantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter


class PreparedAttention:
    def __init__(self, weights, source, metadata, key_cache, value_cache, config, *, mode,
                 session, name, workspace, buffers, cache_metadata, key_scale=None, value_scale=None, tuning_page_count=None):
        c, tokens = config.text, source.shape[0]
        heads, groups, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]

        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)

        self.projected = zeros((tokens, (heads + 2 * groups) * dim))
        self.projection = QuantizedProjection(source, weights["qkv"], self.projected, session=session,
            name=name + ".qkv", activation_mode="auto", workspace=workspace)
        self.query = self.projected[:, :heads * dim].view(tokens, heads, dim)
        key, value = (tensor.view(tokens, groups, dim) for tensor in self.projected[:, heads * dim:].split(groups * dim, -1))
        attended, output = zeros((tokens, heads, dim)), zeros((tokens, config.hidden))
        self.writer = PreparedPagedKVWriter(key, value, key_cache, value_cache, cache_metadata.slots,
            session=session, name=name + ".cache_write", k_scale=key_scale, v_scale=value_scale)
        self.attention = PreparedPagedAttention(self.query, key_cache, value_cache, metadata.page_table,
            cache_metadata.lengths, metadata.query_start_loc, attended, mode=mode, tuning_page_count=tuning_page_count,
            width=1 if mode == "prefill" else metadata.max_query_tokens,
            session=session, name=name + ".attention", workspace=workspace,
            k_descale=key_scale, v_descale=value_scale)
        self.output = QuantizedProjection(attended.flatten(1), weights["output"], output, session=session,
            name=name + ".output", activation_mode="auto", workspace=workspace)
        self.attended = attended.flatten(1)

    def __call__(self, source):
        self.projection(source)
        self.writer()
        self.attention()
        return self.output(self.attended)
