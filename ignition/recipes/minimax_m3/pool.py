"""Full-resolution GQA/index histories and exact partial-page continuations."""

from dataclasses import dataclass

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def page_specs(config, shard):
    heads = config.text["num_key_value_heads"] // shard.world_size
    for layer in range(config.layers):
        for kind in ("key", "value"):
            yield f"layer{layer}.{kind}", (PAGE_TOKENS * heads * config.text["head_dim"],)
        if config.sparse(layer):
            yield f"layer{layer}.index", (PAGE_TOKENS * config.text["sparse_attention_config"]["sparse_index_dim"],)


def page_bytes(config, shard):
    return sum(shape[0] for _, shape in page_specs(config, shard))


def continuation_layout(config, shard):
    regions = [(name, shape, torch.uint8) for name, shape in page_specs(config, shard)]
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    if config.target_layers:
        regions.append(("target_hidden", (len(config.target_layers), config.hidden), torch.bfloat16))
    if config.draft is not None:
        from .draft import continuation_specs

        regions.extend(continuation_specs(config, shard))
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(PagedStateStorage):
    config: object
    shard: object

    @classmethod
    def for_model(cls, config, shard, slots, pages, device, **kwargs):
        config.validate_shard(shard)
        backing = PagedStateStorage.allocate(continuation_layout(config, shard), page_specs(config, shard),
            slots, pages, device, **kwargs)
        return cls(backing.backing, backing.layout, backing.memory_capacity, backing.pages,
                   backing.reserved_backing, config, shard)

    def cache(self, layer, kind):
        value = self.pages[f"layer{layer}.{kind}"].view(torch.float8_e4m3fn)
        if kind == "index":
            return value.view(-1, PAGE_TOKENS, self.config.text["sparse_attention_config"]["sparse_index_dim"])
        heads = self.config.text["num_key_value_heads"] // self.shard.world_size
        return value.view(-1, PAGE_TOKENS, heads, self.config.text["head_dim"])
