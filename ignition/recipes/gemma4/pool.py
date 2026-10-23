"""Device-resident continuation state and paged KV storage for Gemma 4."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.parallel import TensorParallelShard
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
    for index in range(config.first_kv_shared_layer):
        kv_heads = config.local_kv_heads(index, shard)
        dim = config.layer_head_dim(index)
        if config.is_sliding(index):
            size = PAGE_TOKENS * kv_heads * dim
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (size,)
        else:
            key_size = PAGE_TOKENS * kv_heads * dim
            half_value_size = PAGE_TOKENS * kv_heads * (dim // 2)
            yield f"layer{index}.key", (key_size,)
            yield f"layer{index}.value_lo", (half_value_size,)
            yield f"layer{index}.value_hi", (half_value_size,)


def page_bytes(config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
    regions = [(name, shape, torch.uint8) for name, shape in page_specs(config, shard)]
    regions.extend((
        ("last_hidden", (config.hidden,), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
    ))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(
        cls,
        config,
        slots,
        pages,
        device,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
        **kwargs,
    ):
        return cls.allocate(
            continuation_layout(config, shard),
            page_specs(config, shard),
            slots,
            pages,
            device,
            **kwargs,
        )

    def attention(self, layer, config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
        donor = config.kv_source_layer(layer)
        kv_heads = config.local_kv_heads(layer, shard)
        dim = config.layer_head_dim(layer)
        if config.is_sliding(layer):
            shape = (-1, PAGE_TOKENS, kv_heads, dim)
            return tuple(
                self.pages[f"layer{donor}.{field}"].view(torch.float8_e4m3fn).view(shape)
                for field in ("key", "value")
            )
        key_shape = (-1, PAGE_TOKENS, kv_heads, dim)
        value_shape = (-1, PAGE_TOKENS, kv_heads, dim // 2)
        return (
            self.pages[f"layer{donor}.key"].view(torch.float8_e4m3fn).view(key_shape),
            self.pages[f"layer{donor}.value_lo"].view(torch.float8_e4m3fn).view(value_shape),
            self.pages[f"layer{donor}.value_hi"].view(torch.float8_e4m3fn).view(value_shape),
        )
