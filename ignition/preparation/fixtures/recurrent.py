"""Finite recurrent benchmark operands with serving layouts and serialized state reuse."""

from functools import partial

import torch
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch
from ignition.preparation.tuning import benchmark_resource


class BenchmarkStateArena:
    """One full-stride semantic-state arena reused by serialized trials."""

    def __init__(self, template, *, storage=None):
        self.nbytes = template.shape[0] * template.stride(0) * template.element_size()
        self.borrowed_nbytes = 0 if storage is None else self.nbytes
        self.allocated_nbytes = self.nbytes - self.borrowed_nbytes
        if storage is None:
            storage = torch.empty(self.nbytes, dtype=torch.uint8, device=template.device)
        elif (not isinstance(storage, torch.Tensor) or storage.dtype != torch.uint8
                or not storage.is_contiguous() or storage.nbytes != self.nbytes
                or storage.device != template.device):
            raise ValueError("borrowed recurrent arena requires contiguous uint8 storage with the exact extent and device")
        self.storage = storage.view(-1)

    def view(self, template):
        if (template.device != self.storage.device
                or template.shape[0] * template.stride(0) * template.element_size() != self.nbytes):
            raise ValueError("benchmark state shape/device differs from its strided arena")
        extent = sum((size - 1) * stride for size, stride in zip(template.shape, template.stride())) + 1
        if extent * template.element_size() > self.nbytes:
            raise ValueError("benchmark state view exceeds its strided arena")
        return self.storage.view(template.dtype).as_strided(template.shape, template.stride())


def _benchmark_state_key(template):
    return ("delta_state", str(template.device), template.shape[0],
            template.stride(0) * template.element_size())


def register_benchmark_state(session, template, *, storage, restore):
    """Borrow an unused hybrid pool during tuning and restore it before capture."""
    from ignition.preparation.tuning import RecordedPreparationSession

    if not isinstance(session, RecordedPreparationSession):
        raise TypeError("borrowing recurrent state requires RecordedPreparationSession")
    arena = BenchmarkStateArena(template, storage=storage)
    session.register_benchmark_resource(_benchmark_state_key(template), arena,
                                        restore=restore)


def _strided_empty(template):
    return torch.empty_strided(template.shape, template.stride(), dtype=template.dtype, device=template.device)


def _packed_qkv(original):
    tensors = tuple(original[name] for name in ("q", "k", "v"))
    if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != 1:
        raise ValueError("recurrent prefill benchmark requires packed QKV storage")
    first = min(tensor.storage_offset() for tensor in tensors)
    end = max(tensor.storage_offset() + 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    ) for tensor in tensors)
    storage = torch.empty(end - first, dtype=tensors[0].dtype, device=tensors[0].device)
    views = {
        name: storage.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset() - first)
        for name, tensor in zip(("q", "k", "v"), tensors)
    }
    return storage, views


class RecurrentBenchmarkInputs:
    """Active recurrence/convolution operands with exact serving tensor layouts."""

    def __init__(self, original, *, mode, arena):
        self.mode, self.arena = mode, arena
        self.state_name = "conv_state" if "conv_state" in original else "recurrent_state"
        self.output_name = "out" if "out" in original else "output"
        immutable = {"A_log", "A", "D", "dt_bias", "norm_weight", "conv_weight", "conv_bias"}
        # Verification records are transient scratch the trials may share with serving.
        immutable |= {"correction_cache", "kg_cache"}
        # Recording verification reads one live row per sequence instead of a checkpoint per column.
        self.recording = "correction_cache" in original
        self.packed_source, packed_views = _packed_qkv(original) if "q" in original else (None, {})
        self.args = {}
        for name, value in original.items():
            if name in immutable:
                self.args[name] = value
            elif name == self.state_name:
                self.args[name] = arena.view(value)
            elif name in packed_views:
                self.args[name] = packed_views[name]
            else:
                self.args[name] = _strided_empty(value)
        self.state = self.args[self.state_name]
        source_name = "x" if "x" in original else "mixed_qkv" if "mixed_qkv" in original else "q"
        self.tokens = original[source_name].shape[0]
        self.sequences = original.get("query_start_loc", original.get("cu_seqlens")).numel() - 1
        self.live_sequences = min(self.tokens, self.sequences)
        quotient, remainder = divmod(self.tokens, self.live_sequences)
        self.lengths = tuple(quotient + (index < remainder) for index in range(self.live_sequences))
        columns = original["state_indices"].shape[1] if "state_indices" in original else 1
        # Padded verification columns address the null row; only live checkpoints need storage.
        widths = self.lengths if "state_indices" in original and not self.recording else (1,) * self.live_sequences
        if max(widths) > columns:
            raise ValueError("benchmark sequence length exceeds its state-index columns")
        self.live_slots, self.state_rows = [], 1
        for width in widths:
            self.live_slots.append(self.state_rows)
            self.state_rows += width
        self.live_slots = tuple(self.live_slots)
        if mode == "prefill" and "checkpoint_state_indices" in original:
            self.state_rows += self.live_sequences
        if self.state_rows > self.state.shape[0]:
            raise ValueError("benchmark state pool cannot hold live and checkpoint slots")
        generator = torch.Generator(device="cpu").manual_seed(104729)
        initial = torch.randn((self.state_rows, *self.state.shape[1:]), generator=generator) * .1
        initial[0].zero_()
        self.initial_state = initial.to(device=self.state.device, dtype=self.state.dtype)
        self.sources = {}
        self.packed_fixture = None
        if self.packed_source is not None:
            value = torch.randn(self.packed_source.shape, generator=generator) * .1
            self.packed_fixture = value.to(device=self.state.device, dtype=self.packed_source.dtype)
        for name, target in self.args.items():
            if name in immutable or name == self.state_name:
                continue
            if target.is_floating_point() and name != self.output_name and name not in packed_views:
                value = torch.randn(tuple(target.shape), generator=generator) * .1
                self.sources[name] = value.to(device=target.device, dtype=target.dtype)
            target.zero_()
        self._metadata(columns)
        self.producers = (partial(self.produce, False), partial(self.produce, True)) if mode == "prefill" else (partial(self.produce, False),)
        self._owned_tensors = (
            self.initial_state, self.packed_source, self.packed_fixture, *self.sources.values(),
            *(value for name, value in self.args.items() if name not in immutable and name != self.state_name),
        )

    @property
    def nbytes(self):
        storage = {(value.device, value.untyped_storage().data_ptr()): value.untyped_storage()
                   for value in self._owned_tensors if value is not None}
        return sum(value.nbytes() for value in storage.values())

    def _metadata(self, columns):
        a = self.args
        a["num_tokens"].fill_(self.tokens)
        a["num_seqs"].fill_(self.live_sequences)
        bounds = a.get("query_start_loc", a.get("cu_seqlens"))
        bounds.fill_(self.tokens)
        bounds[0] = 0
        offset = 0
        for row, length in enumerate(self.lengths):
            offset += length
            bounds[row + 1] = offset
            if "state_slot_ids" in a:
                a["state_slot_ids"][row] = self.live_slots[row]
            if "state_indices" in a:
                for column in range(columns):
                    a["state_indices"][row, column] = (0 if column >= length or (self.recording and column)
                                                       else self.live_slots[row] + column)
            if "final_state_indices" in a:
                a["final_state_indices"][row] = self.live_slots[row]
            if "checkpoint_offsets" in a:
                boundary = length // 64 * 64
                a["checkpoint_offsets"][row] = boundary
                a["checkpoint_state_indices"][row] = self.live_sequences + 1 + row if boundary else 0
        if "num_accepted_tokens" in a:
            a["num_accepted_tokens"].fill_(1)
        if "state_slot_ids" in a and self.live_sequences < self.sequences:
            a["state_slot_ids"][self.live_sequences:].fill_(-1)

    def reset(self):
        self.state[:self.state_rows].copy_(self.initial_state)
        self.args[self.output_name].zero_()

    def produce(self, fresh):
        if self.packed_source is not None:
            self.packed_source.copy_(self.packed_fixture)
        for name, value in self.sources.items():
            self.args[name].copy_(value)
        if "state_is_fresh" in self.args:
            self.args["state_is_fresh"].fill_(fresh)
        if "initial_state_indices" in self.args:
            self.args["initial_state_indices"].zero_()
            if not fresh:
                for row, slot in enumerate(self.live_slots):
                    self.args["initial_state_indices"][row] = slot


def benchmark_call(prepared, *, arguments, session, name, mode, scratch_specs,
                   workspace, run_options, binding_plan=None, bind_method="bind"):
    template = arguments["conv_state"] if binding_plan is not None else arguments["recurrent_state"]
    key = _benchmark_state_key(template)

    def create_inputs():
        arena = benchmark_resource(session, key, lambda: BenchmarkStateArena(template))
        return RecurrentBenchmarkInputs(arguments, mode=mode, arena=arena)

    # Looking up the state on every candidate evicts the active input fixture when
    # its arena is privately allocated. The input fixture owns that arena already.
    operands = benchmark_resource(session, ("delta_inputs", name), create_inputs)
    scratch = planned_scratch(scratch_specs(prepared), workspace)
    binding = getattr(prepared, bind_method)(scratch=scratch, **operands.args,
                            **({"plan": binding_plan} if binding_plan is not None else {}))
    return PreparedCall(run=lambda: prepared.run(binding, **run_options),
        output=operands.args[operands.output_name], produce=operands.producers[0],
        benchmark_producers=operands.producers, reset=operands.reset,
        restore=operands.reset, owners=(operands, scratch, binding))
