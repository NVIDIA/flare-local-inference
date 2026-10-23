"""Typed numeric storage and scalar launch arguments available during export."""

from contextvars import ContextVar
import gc
import inspect
import math


_NUMERIC_STORAGE = ContextVar("flare_numeric_storage", default=None)


def merge_ranges(ranges):
    """Return sorted disjoint half-open byte intervals."""
    result = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract_ranges(ranges, exclusions):
    result, index = [], 0
    for start, end in merge_ranges(ranges):
        while index < len(exclusions) and exclusions[index][1] <= start:
            index += 1
        cursor, other = start, index
        while other < len(exclusions) and exclusions[other][0] < end:
            left, right = exclusions[other]
            if cursor < left:
                result.append((cursor, left))
            cursor = max(cursor, right)
            other += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def tensor_ranges(tensor):
    """Describe elements without including strides' holes or allocator padding.

    Highly fragmented layouts remain unclassified rather than creating an
    unbounded interval catalog during export.
    """
    if not tensor.numel():
        return []
    start, width = tensor.data_ptr(), tensor.element_size()
    dimensions = sorted((stride * width, size) for size, stride in zip(tensor.shape, tensor.stride())
                        if size > 1 and stride)
    outer = []
    for stride, size in dimensions:
        if stride < 0:
            return []
        if not outer and stride <= width:
            width += (size - 1) * stride
        else:
            outer.append((stride, size))
    if math.prod(size for _, size in outer) > 65536:
        return []
    spans = [(start, start + width)]
    for stride, size in outer:
        spans = [(left + index * stride, right + index * stride)
                 for index in range(size) for left, right in spans]
    return merge_ranges(spans)


class NumericStorage:
    """Retain numeric tensor roles without retaining their storage."""

    def __init__(self):
        self.entries = {}

    def __enter__(self):
        self.token = _NUMERIC_STORAGE.set(self)
        return self

    def __exit__(self, *exc):
        _NUMERIC_STORAGE.reset(self.token)

    def record(self, tensor):
        from torch.multiprocessing.reductions import StorageWeakRef

        storage = tensor.untyped_storage()
        key = storage._cdata
        reference, address, nbytes, spans = self.entries.setdefault(
            key, (StorageWeakRef(storage), storage.data_ptr(), storage.nbytes(), []))
        if (address, nbytes) != (storage.data_ptr(), storage.nbytes()):
            raise ValueError("numeric checkpoint storage was resized during preparation")
        spans.extend(tensor_ranges(tensor))

    def ranges(self):
        import torch

        result = []
        for reference, address, nbytes, spans in self.entries.values():
            storage = torch.UntypedStorage._new_with_weak_ptr(reference.cdata)
            if storage is None:
                continue
            if (storage.data_ptr(), storage.nbytes()) != (address, nbytes):
                raise ValueError("numeric checkpoint storage was resized during preparation")
            result.extend(spans)
        return merge_ranges(result)


def record_numeric_tensor(tensor):
    registry = _NUMERIC_STORAGE.get()
    if registry is not None:
        registry.record(tensor)


def _triton_parameters(kernel):
    from triton.backends.nvidia.driver import expand_signature
    from triton.compiler import ASTSource

    if not isinstance(kernel.src, ASTSource) or not kernel.function:
        return None
    ir = kernel.asm.get("ttir")
    if ir is None or any(operation in ir for operation in (
            "tt.int_to_ptr", "tt.elementwise_inline_asm", "tt.extern_elementwise", "tt.call")):
        return None
    types = []

    def flatten(value):
        if isinstance(value, (tuple, list)):
            for child in value:
                flatten(child)
        elif value != "constexpr":
            types.append(value)

    descriptor = {}
    if "descriptor_type" in inspect.signature(expand_signature).parameters:
        descriptor["descriptor_type"] = "nvTmaDesc"
    flatten(expand_signature(kernel.src.signature.values(),
                             getattr(kernel.metadata, "tensordesc_meta", None), **descriptor))
    types.extend(("*i8", "*i8"))
    sizes = {"i1": 1, "u1": 1, "i8": 1, "u8": 1, "i16": 2, "u16": 2,
             "i32": 4, "u32": 4, "i64": 8, "u64": 8, "fp16": 2, "bf16": 2,
             "fp32": 4, "f32": 4, "fp64": 8, "nvTmaDesc": 128}
    if any(not value.startswith("*") and value not in sizes for value in types):
        return None
    return tuple((8, False) if value.startswith("*") else (sizes[value], value != "nvTmaDesc")
                 for value in types)


def live_metadata(allocations, declared, buffers, *, zero_ranges=()):
    """Collect exact tensor leaves and compiler-declared scalar arguments.

    Integer and byte tensors can contain pointers or tensor maps. They remain
    opaque unless their checkpoint or native-I/O role establishes numeric data.
    Integer aliases also prevent inferring numeric data from a floating view.
    Overlapping driver mappings describe coverage, with zero-initialized and
    declared disposable storage excluded. Views crossing coverage boundaries still classify their covered bytes.
    """
    from bisect import bisect_right
    import torch
    from triton.compiler import CompiledKernel

    bounds = subtract_ranges(
        merge_ranges((a.address, a.address + a.nbytes) for a in allocations if a.initialize != "zero"),
        merge_ranges([(a.address, a.address + a.nbytes) for a in allocations if a.initialize == "zero"]
                     + [(address, address + size) for address, size in zero_ranges]))
    starts = [start for start, _ in bounds]
    numeric, opaque, arguments = [], [], {}
    seen = set()
    for value in gc.get_objects():
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            if value.device.type != "cuda" or value.layout != torch.strided or not value.numel():
                continue
            address = value.data_ptr()
            end = address + value.element_size() * (1 + sum((size - 1) * stride
                                                           for size, stride in zip(value.shape, value.stride())))
            index = max(0, bisect_right(starts, address) - 1)
            if index < len(bounds) and bounds[index][1] <= address:
                index += 1
            if index >= len(bounds) or bounds[index][0] >= end:
                continue
            floating = value.dtype.is_floating_point or value.dtype.is_complex
            layout = address, value.element_size(), tuple(value.shape), tuple(value.stride()), floating
            if layout in seen:
                continue
            seen.add(layout)
            spans = tensor_ranges(value)
            if not floating and not spans:
                spans = [(address, end)]
            classified = numeric if floating else opaque
            for left, right in spans:
                index = max(0, bisect_right(starts, left) - 1)
                while index < len(bounds) and bounds[index][0] < right:
                    start, stop = bounds[index]
                    if left < stop:
                        classified.append((max(left, start), min(right, stop)))
                    index += 1
        elif type(value) is CompiledKernel:
            layout = _triton_parameters(value)
            if layout is not None:
                arguments[value.function] = layout
    numeric = subtract_ranges(numeric, merge_ranges(opaque))
    numeric.extend(declared.ranges())
    numeric.extend((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in buffers.values())
    return merge_ranges(numeric), arguments
