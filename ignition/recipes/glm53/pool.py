"""Exact-prefix GLM state: KDA history, C4 tails, hidden output, and partial pages."""

from dataclasses import dataclass

import torch

from ignition.components.recurrent import DeltaState
from ignition.components.state import StateLayout, StateStorage

from .indexer import IndexState
from .mla import MLAState
from .state import DecoderState


def continuation_layout(layer_types, world_size, *, mtp=False):
    if world_size not in (1, 2, 4):
        raise ValueError("GLM continuation storage requires one, two or four ranks")
    layer_types = tuple(layer_types)
    if not layer_types:
        raise ValueError("GLM continuation storage requires attention layers")
    heads = 64 // world_size
    regions = []
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            regions.extend((
                (f"layer{index}.recurrent", (heads, 128, 128), torch.float32),
                (f"layer{index}.convolution", (3 * heads * 128, 10 if mtp else 3), torch.bfloat16),
            ))
        elif kind == "indexed_attention":
            regions.extend((
                (f"layer{index}.index_tail", (2, 12 if mtp else 4, 128), torch.bfloat16),
                (f"layer{index}.latent_page", (4, 64, 528), torch.uint8),
                (f"layer{index}.index_page", (64, 132), torch.uint8),
            ))
        else:
            raise ValueError(f"unsupported canonical GLM attention type: {kind}")
    if mtp:
        regions.extend((("draft.index_tail", (2, 12, 128), torch.bfloat16),
                        ("draft.latent_page", (2, 4, 64, 528), torch.uint8),
                        ("draft.index_page", (2, 64, 132), torch.uint8)))
    regions.extend((("last_hidden", (4096,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


@dataclass(frozen=True)
class ContinuationPool(StateStorage):
    """One byte row per null/live/snapshot slot; complete KV pages remain separate.

    Native configuration assigns execution and prefix-cache rows at load time.
    KDA and C4 kernels bind directly to strided views into these rows. Every MLA
    snapshot has space for the complete 256-token partial page and its index page.
    """

    decoder: DecoderState
    consumed_tokens: torch.Tensor
    last_hidden: torch.Tensor
    draft: MLAState | None = None

    @classmethod
    def allocate(cls, layer_types, world_size, slots, pages, device, *, mtp=False, storage=None, maximum_pages=None,
                 context_ranks=1, virtual=None):
        """`storage` places everything in a package; `virtual`, an in-process VirtualStorage, holds
        only the KV pages. Either gives each page pool `maximum_pages` of shape and backs `pages`."""
        layer_types = tuple(layer_types)
        if type(pages) is not int or pages <= 0:
            raise ValueError("GLM continuation storage requires positive page capacity")
        package_storage = storage
        storage = StateStorage.allocate(continuation_layout(layer_types, world_size, mtp=mtp), slots, device, storage=storage)
        page_storage = package_storage if package_storage is not None else virtual
        maximum = pages if page_storage is None else maximum_pages
        if maximum is None or maximum < pages:
            raise ValueError("GLM virtual page capacity must cover prepared pages on every rank")
        def page_tensor(name, shape):
            if page_storage is None:
                return torch.zeros((pages, *shape), dtype=torch.uint8, device=device)
            return page_storage.device_allocate(name, (maximum, *shape), torch.uint8,
                group="KV_PAGES", maximum=maximum, units=pages)
        layers = []
        for index, kind in enumerate(layer_types):
            if kind == "linear_attention":
                layers.append(DeltaState(storage.region(f"layer{index}.convolution"),
                                         storage.region(f"layer{index}.recurrent")))
            else:
                cache = page_tensor(f"kv-{index}-latent", (4, 64, 528)).view(-1, 64, 528)
                index_cache = page_tensor(f"kv-{index}-index", (64, 132))
                peers = (torch.full((context_ranks,), cache.data_ptr(), dtype=torch.int64, device=device)
                         if context_ranks > 1 else None)
                layers.append(MLAState(cache, IndexState(index_cache, storage.region(f"layer{index}.index_tail")), peers))
        draft = None
        if mtp:
            draft = MLAState(page_tensor("kv-draft-latent", (4, 64, 528)).view(-1, 64, 528),
                IndexState(page_tensor("kv-draft-index", (64, 132)),
                           storage.region("draft.index_tail")))
        storage.memory_capacity[1] = pages
        return cls(storage.backing, storage.layout, storage.memory_capacity, DecoderState(tuple(layers)),
                   storage.region("consumed_tokens"), storage.region("last_hidden"), draft)

    def share_pages(self, storage, group):
        """Bind peer cache addresses after the package's physical page prefixes have been allocated."""
        states = [layer for layer in self.decoder.layers if isinstance(layer, MLAState)]
        regions = {region.address: region for region in storage.dynamic_regions}
        addresses = storage.peer_storage.share_regions(group, [regions[state.cache.data_ptr()] for state in states])
        for state, pointers in zip(states, addresses, strict=True):
            state.peers.copy_(torch.tensor(pointers, dtype=torch.int64, device=self.backing.device))

    def replace_pages(self, pages, *, metadata_mode=None, ranks=1, share=None):
        """Give every MLA state, including the draft layer, `pages` empty physical pages.

        The states are updated in place, so their holders see the new pools; operators that
        bound the previous pools must rebind before capture. A metadata mode describes the
        pools without device memory for forward compilation. With several context ranks,
        `share(nbytes)` allocates decoder latent caches that peers read through CUDA IPC;
        the shared allocations are returned for closing after native replay drains.
        """
        if type(pages) is not int or pages <= 0:
            raise ValueError("GLM continuation storage requires positive page capacity")
        device = self.backing.device

        def empty(shape, dtype=torch.uint8):
            if metadata_mode is not None:
                from ignition.preparation.deferred import deferred_tensor

                return deferred_tensor(shape, dtype=dtype, device=device, mode=metadata_mode)
            return torch.zeros(shape, dtype=dtype, device=device)

        shared = []
        states = [layer for layer in self.decoder.layers if isinstance(layer, MLAState)]
        for state in (*states, *((self.draft,) if self.draft is not None else ())):
            object.__setattr__(state, "cache", None)
            object.__setattr__(state.index, "cache", None)
            shape, peers = (pages * 4, 64, 528), None
            if ranks > 1 and state is not self.draft and metadata_mode is None and share is not None:
                pool = share(pages * 4 * 64 * 528)
                shared.append(pool)
                cache, peers = pool.tensor.view(shape), pool.addresses
            else:
                cache = empty(shape)
                if ranks > 1 and state is not self.draft:
                    peers = empty((ranks,), torch.int64)
                    if metadata_mode is None:
                        # Unshared preparation pools: every rank's address points at this rank's cache.
                        peers.fill_(cache.data_ptr())
            object.__setattr__(state, "cache", cache)
            object.__setattr__(state, "peers", peers)
            object.__setattr__(state.index, "cache", empty((pages, 64, 132)))
        return shared

    def clear_for_preparation(self):
        self.backing.zero_()
        pages = int(self.memory_capacity[1].item())
        for layer in (*self.decoder.layers, *((self.draft,) if self.draft is not None else ())):
            if isinstance(layer, MLAState):
                layer.cache[:pages * 4].zero_()
                layer.index.cache[:pages].zero_()

    def partial_pages(self):
        """Pair each immutable page field with its live physical-page records."""
        for index, layer in enumerate(self.decoder.layers):
            if isinstance(layer, MLAState):
                pages = layer.index.cache.shape[0]
                yield f"layer{index}.latent_page", layer.cache.view(pages, -1)
                yield f"layer{index}.index_page", layer.index.cache.view(pages, -1)

    def draft_pages(self):
        if self.draft is not None:
            pages = self.draft.index.cache.shape[0]
            yield "draft.latent_page", self.draft.cache.view(pages, -1)
            yield "draft.index_page", self.draft.index.cache.view(pages, -1)
