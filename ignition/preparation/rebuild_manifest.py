"""Describe preparation inputs and named views of stored weights without copying payloads.

The catalog describes prepared layouts, not checkpoint tensors. Reading it does
not instantiate Python objects or establish that a recipe can rebuild from it.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math

from . import runtime_package_pb2 as proto


def from_plan(model, plan, *, compiled_forwards):
    from .source_identity import source_identity

    result = dict(version=1, recipe=plan.recipe.name, source_identity=source_identity(),
        checkpoint_repository=plan.checkpoint.repository,
        checkpoint_revision=plan.checkpoint.revision if model.directory is None else '',
        local_checkpoint=model.directory is not None, tensor_parallel=plan.world_size,
        max_concurrency=plan.concurrency, prefill_chunk=plan.prefill_chunk,
        image_concurrency=plan.image_concurrency, context_parallel=plan.context_parallel,
        stream_overlap=plan.stream_overlap, mixed_prefill=plan.mixed_prefill,
        mla_compute_mode=plan.mla_compute_mode, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain,
        online_mxfp8=plan.online_mxfp8, language_model_only=plan.language_model_only,
        compiled_forwards=compiled_forwards, reserve_bytes=plan.reserve_bytes)
    if plan.expert_scale_compression is not None:
        result['expert_scale_compression'] = plan.expert_scale_compression
    if (s := plan.speculation) is not None:
        result['speculation'] = dict(draft_tokens=s.draft_tokens,
            adaptive=s.adaptive, proposal_source=s.proposal_source, history_drafts=s.history_drafts,
            token_capacity=plan.speculative_token_capacity, packed_verification=s.packed_verification,
            draft_prefix_selection=s.draft_prefix_selection)
    return result


def weight_views(roots):
    """Walk named weight owners without evaluating properties, closures, or module code.

    Paths identify fields within the recorded source identity. Aliased tensors keep
    each field name; cycles in container ownership do not recurse indefinitely.
    """
    import torch

    def walk(value, name, owner, ancestors):
        if isinstance(value, torch.Tensor):
            yield name, owner, value
            return
        if value is None or id(value) in ancestors:
            return
        ancestry = ancestors | {id(value)}
        if isinstance(value, Mapping):
            children = ((f'[{key!r}]', child) for key, child in value.items()
                        if isinstance(key, (str, int)))
        elif isinstance(value, (tuple, list)):
            children = ((f'[{index}]', child) for index, child in enumerate(value))
        elif isinstance(value, torch.nn.Module):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{key}', child) for group in (value._parameters, value._buffers, value._modules)
                        for key, child in group.items())
        elif is_dataclass(value) and not isinstance(value, type):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{field.name}', getattr(value, field.name)) for field in fields(value))
        elif type(value).__module__.startswith(('b12x.', 'ignition.')) and hasattr(value, '__dict__'):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{key}', child) for key, child in vars(value).items())
        else:
            return
        for suffix, child in children:
            yield from walk(child, name + suffix, owner, ancestry)

    for name, value in sorted(roots.items()):
        yield from walk(value, name, '', set())


def rebuild_weights_metadata(roots):
    """Describe tensor views for native resolution against retained storage."""
    import torch

    tensors, unavailable = [], []
    for name, owner, tensor in weight_views(roots):
        if (tensor.device.type == 'meta' or tensor.layout != torch.strided or not tensor.numel()
                or any(stride < 0 for stride in tensor.stride())):
            unavailable.append(name)
            continue
        extent = tensor.element_size() * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)))
        tensors.append(dict(name=name, dtype=str(tensor.dtype).removeprefix('torch.'),
            element_bytes=tensor.element_size(), shape=list(tensor.shape), stride=list(tensor.stride()),
            layout_type=owner, address=tensor.data_ptr(), extent=extent))
    return dict(version=1, tensors=tensors, unavailable=unavailable)


def validate_weights(bundle):
    """Check catalog references before a preparation tool uses them; serving ignores this metadata."""
    if not bundle.HasField('rebuild_weights'):
        return
    catalog = bundle.rebuild_weights
    if catalog.version != 1:
        raise ValueError(f'unsupported prepared-weight catalog version {catalog.version}')
    allocations = {a.name: a for a in bundle.allocations}
    names = set()
    widths = {'bool': 1, 'uint8': 1, 'int8': 1, 'int16': 2, 'uint16': 2, 'float16': 2,
              'bfloat16': 2, 'int32': 4, 'uint32': 4, 'float32': 4, 'int64': 8, 'uint64': 8,
              'float64': 8, 'complex32': 4, 'complex64': 8, 'complex128': 16,
              'float8_e4m3fn': 1, 'float8_e5m2': 1, 'float8_e4m3fnuz': 1,
              'float8_e5m2fnuz': 1, 'float8_e8m0fnu': 1}
    for tensor in catalog.tensors:
        if (not tensor.name or tensor.name in names or not tensor.HasField('storage')
                or tensor.element_bytes != widths.get(tensor.dtype)
                or len(tensor.shape) != len(tensor.stride) or not math.prod(tensor.shape)):
            raise ValueError('invalid prepared-weight tensor description')
        names.add(tensor.name)
        extent = tensor.element_bytes * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride, strict=True)))
        allocation = allocations.get(tensor.storage.allocation)
        if (allocation is None or allocation.HasField('sizing') or allocation.memory == proto.PEER
                or tensor.storage.offset + extent > allocation.nbytes):
            raise ValueError(f'prepared-weight tensor {tensor.name!r} exceeds fixed stored allocation')
    for name in catalog.unavailable:
        if not name or name in names:
            raise ValueError('duplicate or empty unavailable prepared-weight name')
        names.add(name)
