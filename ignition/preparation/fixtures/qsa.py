"""Private paged QSA workloads with shared history and disjoint request suffixes."""

from contextlib import nullcontext

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from .resources import candidate_inputs


class QSABenchmarkInputs:
    def __init__(self, caps, binding, dynamic):
        rows = dynamic["query"].shape[0]
        active = min(rows, caps.max_batch)
        width = (rows + active - 1) // active
        suffix_pages = (width + caps.main_page_size - 1) // caps.main_page_size
        history_pages = min((caps.max_seq_len - width) // caps.main_page_size,
                            caps.num_main_cache_pages - active * suffix_pages,
                            caps.num_compressed_cache_pages - active * suffix_pages)
        if history_pages < 0:
            raise ValueError("QSA benchmark page capacity cannot hold the active request intervals")
        pages = history_pages + active * suffix_pages
        history = history_pages * caps.main_page_size
        shared = {"index_q_norm_weight", "index_k_norm_weight", "rope_cos", "rope_sin",
                  "k_descale", "v_descale"}
        self.binding = {}
        for name, value in binding.items():
            if name in shared:
                self.binding[name] = value
            else:
                shape = ((pages, *value.shape[1:]) if name in
                         ("main_k_cache", "main_v_cache", "compressed_k_cache") else value.shape)
                self.binding[name] = torch.empty_strided(shape, value.stride(),
                                                       dtype=value.dtype, device=value.device)
        self.dynamic = {name: torch.empty_strided(value.shape, value.stride(),
                         dtype=value.dtype, device=value.device) for name, value in dynamic.items()}
        device = dynamic["query"].device
        generator = torch.Generator(device=device).manual_seed(530129)
        for name in ("main_k_cache", "main_v_cache", "compressed_k_cache", "raw_k_ring"):
            value = self.binding[name]
            value.copy_(torch.randn(value.shape, dtype=torch.bfloat16, device=device,
                                    generator=generator).to(value.dtype))
        for name in ("query", "index_query", "raw_index_key"):
            self.dynamic[name].normal_(generator=generator)
        bounds = torch.arange(caps.max_batch + 1, device=device).clamp_max(active) * rows // active
        lengths = bounds[1:] - bounds[:-1]
        requests = torch.repeat_interleave(torch.arange(caps.max_batch, device=device), lengths)
        positions = history + torch.arange(rows, device=device) - bounds[requests]
        d = self.dynamic
        d["request_ids"].copy_(requests)
        d["query_positions"].copy_(positions)
        d["rope_positions"].copy_(positions[:, None].expand_as(d["rope_positions"]))
        d["query_start_loc"].copy_(bounds)
        d["sequence_lengths"].copy_(torch.where(lengths > 0, history + lengths, 0))
        d["num_accepted_tokens"].copy_((lengths > 0).to(torch.int32))
        d["is_prefilling"].copy_(lengths > 1 + caps.max_speculative_tokens)
        for name in ("main_block_table", "compressed_block_table"):
            table = self.binding[name]
            table.fill_(-1)
            table[:active, :history_pages].copy_(torch.arange(history_pages, device=device))
            suffix = torch.arange(active * suffix_pages, device=device).view(active, suffix_pages)
            table[:active, history_pages:history_pages + suffix_pages].copy_(suffix + history_pages)
        self.binding["raw_state_slot_ids"].fill_(-1)
        self.binding["raw_state_slot_ids"][:active].copy_(torch.arange(active, device=device))
        tags = self.binding["raw_logical_positions"]
        rope_tags = self.binding["raw_rope_positions"]
        tags.fill_(-1)
        rope_tags.fill_(-1)
        previous = torch.arange(max(0, history - caps.raw_ring_capacity), history, device=device)
        ring = previous % caps.raw_ring_capacity
        tags[:active, ring] = previous
        rope_tags[:active, ring] = previous[:, None]
        self.binding["raw_interval_start_positions"].fill_(-1)
        self.binding["raw_interval_start_positions"][:active].fill_(history - 1)
        self.seeds = {name: self.binding[name].clone() for name in
            ("compressed_k_cache", "raw_k_ring", "raw_logical_positions",
             "raw_rope_positions", "raw_interval_start_positions")}
        self._private = tuple(value for name, value in self.binding.items() if name not in shared)

    @property
    def nbytes(self):
        return sum(value.untyped_storage().nbytes() for value in
                   (*self._private, *self.dynamic.values(), *self.seeds.values()))

    def produce(self):
        for name, value in self.seeds.items():
            self.binding[name].copy_(value)


def input_factory(caps, binding, dynamic, *, storage=None):
    def allocate():
        with (nullcontext() if storage is None else storage.transient()):
            return QSABenchmarkInputs(caps, binding, dynamic)

    return candidate_inputs(allocate)


def benchmark_call(state, *, inputs, workspace):
    inputs = inputs()
    scratch = planned_scratch(state.scratch_specs(), workspace)
    binding = state.bind_for_preparation(scratch=scratch, **inputs.binding)
    return PreparedCall(run=lambda: state.run_for_preparation(binding, **inputs.dynamic),
                        produce=inputs.produce, owners=(inputs, scratch, binding))
