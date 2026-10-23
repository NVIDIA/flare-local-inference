"""Compile prepared forwards with explicit storage mutation boundaries."""

import hashlib
import gc
import math
import operator
import os
import pickle
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch._higher_order_ops.triton_kernel_wrap import get_mutated_tensors, kernel_side_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_unflatten

_INVOCATIONS = ContextVar('flare_storage_invocations')
_OPTIONS = {'max_autotune': True, 'emulate_precision_casts': True,
            'triton.cudagraphs': False, 'triton.autotune_at_compile_time': False,
            'triton.cooperative_reductions': True,
            'compile_threads': min(16, os.cpu_count() or 1)}
if os.name == 'nt':
    _OPTIONS['worker_start_method'] = 'spawn'
# Compiled forwards and their guards persist beside b12x kernels and tuning selections.
DEFAULT_CACHE_ROOT = Path.home() / '.cache' / 'flare' / 'kernels'


@contextmanager
def _defer_full_collection():
    thresholds = gc.get_threshold()
    # Repeated full-heap scans dominate collection while FX graphs are constructed.
    gc.set_threshold(thresholds[0], thresholds[1], max(thresholds[2], 1000))
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)


@contextmanager
def _compiler_recursion(module):
    # Inductor follows one recursive rename per aliased storage mutation.
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, 2 * len(module.graph.nodes) + 1024))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


@dataclass
class _Invocation:
    target: object
    leaves: list
    tree: object
    operands: list
    integers: list = field(default_factory=list)

    def run(self, reads, writes, integers=()):
        leaves = self.leaves.copy()
        for position, value in zip(self.integers, integers, strict=True):
            leaves[position] = value
        for position, mutable, index, view in self.operands:
            tensor = writes[index] if mutable else reads[index]
            if view is not None:
                dtype, shape, stride, offset = view
                tensor = tensor.view(dtype).as_strided(shape, stride, offset)
            leaves[position] = tensor
        args, kwargs = tree_unflatten(leaves, self.tree)
        self.target(*args, **kwargs)


@torch.library.custom_op('flare::storage_call', mutates_args=('writes',))
def _storage_call(reads: list[torch.Tensor], writes: list[torch.Tensor], handle: int,
                  integers: list[int] | None = None) -> None:
    _INVOCATIONS.get()[handle].run(reads, writes, () if integers is None else integers)


@_storage_call.register_fake
def _fake_storage_call(reads, writes, handle, integers=None):
    pass


@torch.library.custom_op('flare::snapshot', mutates_args=())
def _snapshot(source: torch.Tensor) -> torch.Tensor:
    return source.clone()


@_snapshot.register_fake
def _fake_snapshot(source):
    return torch.empty_like(source)


def _snapshot_lowering(source):
    from torch._inductor.lowering import clone

    # A clone can have been folded into its source before custom lowering.
    result = clone(source)
    result.realize()
    return result


@torch.library.custom_op('flare::mean', mutates_args=())
def _mean(source: torch.Tensor, dim: list[int] | None, keepdim: bool = False,
          dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.mean(source, dim, keepdim, dtype=dtype)


@_mean.register_fake
def _fake_mean(source, dim, keepdim=False, dtype=None):
    return torch.mean(source, dim, keepdim, dtype=dtype)


@torch.library.custom_op('flare::sigmoid', mutates_args=())
def _sigmoid(source: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(source)


@_sigmoid.register_fake
def _fake_sigmoid(source):
    return torch.sigmoid(source)


@torch.library.custom_op('flare::gelu', mutates_args=())
def _gelu(source: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    return torch.nn.functional.gelu(source, approximate=approximate)


@_gelu.register_fake
def _fake_gelu(source, approximate='none'):
    return torch.empty_like(source)


@torch.library.custom_op('flare::native_layer_norm', mutates_args=())
def _native_layer_norm(source: torch.Tensor, shape: list[int], weight: torch.Tensor | None,
                       bias: torch.Tensor | None, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.aten.native_layer_norm.default(source, shape, weight, bias, eps)


@_native_layer_norm.register_fake
def _fake_native_layer_norm(source, shape, weight, bias, eps):
    return torch.ops.aten.native_layer_norm.default(source, shape, weight, bias, eps)


@torch.library.custom_op('flare::native_rms_norm', mutates_args=())
def _native_rms_norm(source: torch.Tensor, shape: list[int], weight: torch.Tensor | None,
                     eps: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.aten._fused_rms_norm.default(source, shape, weight, eps)


@_native_rms_norm.register_fake
def _fake_native_rms_norm(source, shape, weight, eps):
    return torch.ops.aten._fused_rms_norm.default(source, shape, weight, eps)


def _byte_storage(tensor):
    storage = tensor.untyped_storage()
    return tensor.new_empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


@contextmanager
def _artifact_kernel_table():
    # Native launches are runtime bindings; artifacts must not restore their IDs.
    names = ('id_to_kernel', 'kernel_to_id', 'constant_args')
    with kernel_side_table.lock:
        tables = [getattr(kernel_side_table, name) for name in names]
        try:
            for name in names:
                setattr(kernel_side_table, name, {})
            yield
        finally:
            for name, table in zip(names, tables):
                setattr(kernel_side_table, name, table)


def _returned_alias(node, index):
    if not isinstance(node.target, torch._ops.OpOverload):
        return None
    schema = node.target._schema
    if len(schema.returns) == 1 and isinstance(schema.returns[0].type, torch.ListType):
        return None
    result = schema.returns[index]
    if result.alias_info is None:
        return None
    for position, argument in enumerate(schema.arguments):
        if argument.alias_info is not None and result.alias_info.before_set & argument.alias_info.before_set:
            value = node.args[position] if position < len(node.args) else node.kwargs.get(argument.name)
            if isinstance(value, torch.fx.Node):
                return value
    return None


@dataclass
class CompiledForward:
    artifact: object
    inputs: list
    invocations: list
    owner: object
    output_tree: object
    cache_hit: bool

    def __call__(self):
        token = _INVOCATIONS.set(self.invocations)
        try:
            result = self.artifact(*self.inputs)
            return tree_unflatten(result, self.output_tree)
        finally:
            _INVOCATIONS.reset(token)


def _compile_module(module, calls, owner, output_tree):
    import copy

    from torch._inductor import standalone_compile
    from torch._inductor.codecache import torch_key

    from .aot import _load_inductor
    from .device_binding import bind_device

    graph = module.graph
    inputs = []
    for node in list(graph.nodes):
        if node.op != 'get_attr':
            continue
        tensor = operator.attrgetter(node.target)(module)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'compiled forward constant must be a tensor: {node.target}')
        with graph.inserting_before(next(iter(graph.nodes))):
            argument = graph.placeholder(f'buffer_{len(inputs)}')
        argument.meta = node.meta.copy()
        node.replace_all_uses_with(argument)
        graph.erase_node(node)
        inputs.append(tensor)
    # Placeholders and runtime operands must retain the same order.
    inputs.reverse()
    module = torch.fx.GraphModule({}, graph)
    storages = {}
    metadata = []
    for tensor in inputs:
        storage = tensor.untyped_storage()._cdata
        alias = storages.setdefault(storage, len(storages))
        metadata.append((tuple(tensor.shape), tensor.stride(), tensor.storage_offset(),
                         tensor.untyped_storage().nbytes(), tensor.dtype, tensor.device.type,
                         tensor.requires_grad, alias))
    cache_graph = copy.deepcopy(module.graph)
    def portable_device(value):
        return torch.device(value.type) if isinstance(value, torch.device) else value
    for node in cache_graph.nodes:
        node.args = torch.fx.node.map_aggregate(node.args, portable_device)
        node.kwargs = torch.fx.node.map_aggregate(node.kwargs, portable_device)
    cache_code = torch.fx.GraphModule({}, cache_graph).code
    key = hashlib.sha256(pickle.dumps(('compiled-forward-portable-device-v1', cache_code, metadata, _OPTIONS, torch_key(),
        torch.cuda.get_device_capability(), torch.cuda.get_device_name()))).hexdigest()
    path = DEFAULT_CACHE_ROOT / 'compiled' / (key + '.bin')
    cache_hit = path.is_file()
    if cache_hit:
        with path.open('rb') as file:
            original, serialized = pickle.load(file)
        with bind_device(torch.device('cuda', torch.cuda.current_device())), _artifact_kernel_table():
            artifact, _ = _load_inductor(serialized, original)
    else:
        with _compiler_recursion(module):
            artifact = standalone_compile(module, inputs, dynamic_shapes='from_example_inputs',
                options={'config_patches': _OPTIONS}, aot=True, donate_graph_module=True)
    token = _INVOCATIONS.set(calls)
    try:
        # Warm caller-owned state on both paths so TP ranks issue equal collectives.
        artifact(*inputs)
    finally:
        _INVOCATIONS.reset(token)
    if not cache_hit:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _artifact_kernel_table():
            serialized = artifact.serialize()
        from torch._inductor.codecache import write_atomic
        write_atomic(path, pickle.dumps((torch.device('cuda', torch.cuda.current_device()), serialized)))
    return CompiledForward(artifact, inputs, calls, owner, output_tree, cache_hit)


class ForwardTracer:
    """Reuse persistent tensor metadata across a fixed-buffer graph catalog."""

    def __init__(self):
        from torch._functorch import config
        from torch._subclasses.fake_tensor import FakeTensorMode

        with config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            self.mode = FakeTensorMode(allow_non_fake_inputs=True, static_shapes=True)
        self.inputs = {}

    def __call__(self, fn):
        with self.mode:
            module = make_fx(fn, tracing_mode='fake', _allow_non_fake_inputs=True,
                             _disable_torch_fn_metadata_mode=True)()
        for node in module.graph.nodes:
            if node.op == 'get_attr':
                value = operator.attrgetter(node.target)(module)
                if isinstance(value, torch.Tensor):
                    self.inputs[id(value)] = (value, self.mode.from_tensor(value))
        return module


def compile_forward(fn, *, tracer=None):
    """Trace and compile one fixed-buffer forward before CUDA graph capture.

    Tracing uses tensor metadata without executing kernels. Mutating native operators
    receive explicit storage operands for all captured tensor aliases.
    Their typed views are reconstructed inside each opaque invocation so
    Inductor can preserve mutations without cloning strided state regions.
    Pointwise tensor operations remain visible to ordinary Inductor compilation.
    Mean reductions, FP32 sigmoid, exact GELU, LayerNorm and RMSNorm retain ATen's rounding.
    Both compilation and cache restoration execute one warmup on caller-owned state.
    Compiled artifacts take persistent tensors as inputs and are cached without
    weight payloads or process-local kernel handles. Each callable retains its
    own prepared invocations, buffers, and forward owner.
    """
    output_tree = None

    def trace():
        nonlocal output_tree
        leaves, output_tree = tree_flatten(fn())
        return tuple(leaves)

    module = (ForwardTracer() if tracer is None else tracer)(trace)
    calls = _lower_storage(module)
    return _compile_module(module, calls, fn, output_tree)


def _lower_storage(module):
    from torch._inductor.lowering import register_lowering

    register_lowering(torch.ops.flare.snapshot.default)(_snapshot_lowering)
    graph = module.graph
    # FP32 rounding differences can cross a BF16 rounding boundary.
    for node in graph.nodes:
        if node.op == 'call_function' and node.target is torch.ops.aten.mean.dim:
            node.target = torch.ops.flare.mean.default
        elif (node.op == 'call_function' and node.target is torch.ops.aten.sigmoid.default
                and node.meta['val'].dtype == torch.float32):
            node.target = torch.ops.flare.sigmoid.default
        elif (node.op == 'call_function' and node.target is torch.ops.aten.gelu.default
                and node.kwargs.get('approximate', 'none') == 'none'):
            node.target = torch.ops.flare.gelu.default
        elif node.op == 'call_function' and node.target is torch.ops.aten.native_layer_norm.default:
            node.target = torch.ops.flare.native_layer_norm.default
        elif node.op == 'call_function' and node.target is torch.ops.aten._fused_rms_norm.default:
            node.target = torch.ops.flare.native_rms_norm.default
    for node in reversed(list(graph.nodes)):
        if (node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload)
                and node.target._schema.name in (
                    'profiler::_record_function_enter', 'profiler::_record_function_enter_new',
                    'profiler::_record_function_exit')):
            graph.erase_node(node)
    originals = list(graph.nodes)
    bases = {}
    keys = {}
    for node in originals:
        value = node.meta.get('val')
        if not isinstance(value, torch.Tensor):
            continue
        keys[node] = value.untyped_storage()._cdata
        if node.op == 'get_attr':
            actual = operator.attrgetter(node.target)(module)
            keys[node] = actual.untyped_storage()._cdata
            node.meta['val'] = actual
        elif node.op == 'call_function' and node.target is operator.getitem:
            source, index = node.args
            if isinstance(source, torch.fx.Node) and isinstance(index, int):
                alias = _returned_alias(source, index)
                if alias is not None:
                    keys[node] = keys[alias]
                    node.meta['val'] = alias.meta['val']
                elif (isinstance(source.target, torch._ops.OpOverload)
                      and len(source.target._schema.returns) == 1
                      and isinstance(source.target._schema.returns[0].type, torch.ListType)
                      and source.target._schema.returns[0].alias_info is not None
                      and not source.target._schema.is_mutable):
                    # List-valued views retain each element's offset and stride.
                    values = source.target(
                        *torch.fx.map_arg(source.args, lambda arg: arg.meta['val']),
                        **torch.fx.map_arg(source.kwargs, lambda arg: arg.meta['val']))
                    view = values[index]
                    leaves, _ = tree_flatten((source.args, source.kwargs))
                    for argument in leaves:
                        if (isinstance(argument, torch.fx.Node) and argument in keys
                                and argument.meta['val'].untyped_storage()._cdata == view.untyped_storage()._cdata):
                            keys[node] = keys[argument]
                            node.meta['val'] = view
                            break
                    else:
                        raise ValueError(f'cannot identify input storage for list view {source.target}')
        elif node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):
            schema = node.target._schema
            if len(schema.returns) == 1 and schema.returns[0].alias_info is not None:
                aliases = schema.returns[0].alias_info.before_set
                for index, argument in enumerate(schema.arguments):
                    if argument.alias_info is not None and aliases & argument.alias_info.before_set:
                        arg = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
                        if isinstance(arg, torch.fx.Node) and arg in keys:
                            keys[node] = keys[arg]
                            if schema.is_mutable:
                                node.meta['val'] = arg.meta['val']
                            else:
                                node.meta['val'] = node.target(
                                    *torch.fx.map_arg(node.args, lambda arg: arg.meta['val']),
                                    **torch.fx.map_arg(node.kwargs, lambda arg: arg.meta['val']))
                            break
    mutations = {node: _mutated_tensors(node) for node in originals if node.op == 'call_function'}
    materialize = _reads_across_mutations(originals, keys, mutations)
    dtypes, storage_references = {}, {}
    for node, key in keys.items():
        dtypes.setdefault(key, set()).add(node.meta['val'].dtype)
        storage_references.setdefault(key, set()).add(node)

    for node in originals:
        if node.op != 'get_attr' or not isinstance(node.meta.get('val'), torch.Tensor):
            continue
        key = keys[node]
        if len(dtypes[key]) <= 1 and len(storage_references[key]) <= 1:
            continue
        if key not in bases:
            tensor = operator.attrgetter(node.target)(module)
            name = '_storage_' + str(len(bases))
            storage = _byte_storage(tensor)
            if len(dtypes[key]) == 1:
                storage = storage.view(tensor.dtype)
            module.register_buffer(name, storage)
            with graph.inserting_before(originals[0]):
                bases[key] = graph.get_attr(name)

    def tensor_nodes(value):
        leaves, _ = tree_flatten(value)
        return [node for node in leaves if isinstance(node, torch.fx.Node)
                and isinstance(node.meta.get('val'), torch.Tensor)]

    def key_for(node):
        return keys[node]

    calls = []
    for node in originals:
        if node.op != 'call_function':
            continue
        target = node.target
        mutated = mutations[node]
        if not mutated:
            continue
        # Keep pointwise tensor work visible unless its operands alias shared storage.
        is_external = target is torch.ops.higher_order.triton_kernel_wrapper_mutation or (
            isinstance(target, torch._ops.OpOverload) and target.namespace != 'aten')
        operand_nodes = tensor_nodes((node.args, node.kwargs))
        if not is_external and all(arg.meta['val'].is_contiguous() for arg in mutated) and not any(
            key_for(arg) in bases
            for arg in operand_nodes):
            continue
        mutable_keys = {key_for(arg) for arg in mutated}
        reads, writes, read_indices, write_indices = [], [], {}, {}
        leaves, tree = tree_flatten((node.args, node.kwargs))
        operands, integers, integer_positions = [], [], []
        for position, arg in enumerate(leaves):
            if not isinstance(arg, torch.fx.Node):
                continue
            value = arg.meta.get('val')
            if isinstance(value, torch.SymInt):
                integers.append(arg)
                integer_positions.append(position)
                leaves[position] = None
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'non-tensor dynamic operand: {arg}')
            key = key_for(arg)
            mutable = key in mutable_keys
            array, indices = (writes, write_indices) if mutable else (reads, read_indices)
            base = bases.get(key)
            identity = key if base is not None else arg
            if identity not in indices:
                indices[identity] = len(array)
                array.append(base if base is not None else arg)
            descriptor = (value.dtype, tuple(map(int, value.shape)), tuple(map(int, value.stride())),
                          int(value.storage_offset())) if base is not None else None
            operands.append((position, mutable, indices[identity], descriptor))
            leaves[position] = None
        invocation = _Invocation(target, leaves, tree, operands, integer_positions)
        handle = len(calls)
        calls.append(invocation)
        with graph.inserting_before(node):
            arguments = (reads, writes, handle, integers) if integers else (reads, writes, handle)
            graph.call_function(torch.ops.flare.storage_call.default, arguments)
        if node.users:
            schema = getattr(target, '_schema', None)
            aliases = [] if schema is None else [
                _returned_alias(node, index) for index in range(len(schema.returns))]
            if not aliases or any(alias is None for alias in aliases):
                raise RuntimeError(f'unsupported returned value: {node}')
            replacement = aliases[0] if len(aliases) == 1 else tuple(aliases)
            for user in list(node.users):
                user.args = torch.fx.map_arg(user.args, lambda arg, node=node, replacement=replacement:
                                             replacement if arg is node else arg)
                user.kwargs = torch.fx.map_arg(user.kwargs, lambda arg, node=node, replacement=replacement:
                                               replacement if arg is node else arg)
        graph.erase_node(node)

    for node in originals:
        if node.op != 'get_attr' or not isinstance(node.meta.get('val'), torch.Tensor):
            continue
        if key_for(node) not in bases:
            continue
        value = node.meta['val']
        shape = tuple(map(int, value.shape))
        stride = tuple(map(int, value.stride()))
        offset = int(value.storage_offset())
        with graph.inserting_before(node):
            typed = graph.call_function(torch.ops.aten.view.dtype, (bases[key_for(node)], value.dtype))
            if value.is_contiguous():
                sliced = graph.call_function(torch.ops.aten.slice.Tensor,
                    (typed, 0, offset, offset + math.prod(shape)))
                view = graph.call_function(torch.ops.aten.view.default, (sliced, shape))
            else:
                view = graph.call_function(torch.ops.aten.as_strided.default,
                    (typed, shape, stride, offset))
        node.replace_all_uses_with(view)
        graph.erase_node(node)
    for node in materialize:
        with graph.inserting_after(node):
            value = graph.call_function(torch.ops.flare.snapshot.default, (node,))
            value.meta = node.meta.copy()
        node.replace_all_uses_with(value)
        value.args = (node,)
    graph.eliminate_dead_code()
    graph.lint()
    return calls


def _mutated_tensors(node):
    if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
        kwargs = node.kwargs
        names = get_mutated_tensors(kwargs['kernel_idx'], kwargs['constant_args_idx'],
            torch.fx.map_arg(kwargs['kwargs'], lambda arg: arg.meta['val']),
            kwargs['tma_descriptor_metadata'])
        return [kwargs['kwargs'][name] for name in names]
    mutated = []
    if isinstance(node.target, torch._ops.OpOverload):
        for index, argument in enumerate(node.target._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
                leaves, _ = tree_flatten(value)
                mutated.extend(leaf for leaf in leaves if isinstance(leaf, torch.fx.Node)
                               and isinstance(leaf.meta.get('val'), torch.Tensor))
    return mutated


def _reads_across_mutations(nodes, keys, mutations):
    """Materialize computed values that outlive an overwrite of their inputs."""
    positions = {node: index for index, node in enumerate(nodes)}
    last_use = {node: max((positions[user] for user in node.users), default=-1) for node in nodes}
    # Inductor can inline an expression into a later consumer after its input was overwritten.
    for node in reversed(nodes):
        if (isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace == 'aten' and not node.target._schema.is_mutable):
            for operand in node.all_input_nodes:
                last_use[operand] = max(last_use[operand], last_use[node])
    mutations = {node: {keys[operand] for operand in operands}
                 for node, operands in mutations.items() if operands}
    mutable = set().union(*mutations.values()) if mutations else set()
    live, fresh, required = {}, set(), set()
    for index, node in enumerate(nodes):
        operands = node.all_input_nodes
        reads = set().union(*(live.get(arg, set()) for arg in operands)) if operands else set()
        reads.update(keys[arg] for arg in operands if keys.get(arg) in mutable)
        if node in mutations:
            for previous, pending in live.items():
                if previous in fresh and last_use[previous] > index and pending & mutations[node]:
                    required.add(previous)
                    pending.clear()
        live = {value: pending for value, pending in live.items() if pending and last_use[value] > index}
        if (node in keys and node.op == 'call_function'
                and isinstance(node.target, torch._ops.OpOverload) and node.target.namespace == 'aten'):
            if keys[node] not in {keys[arg] for arg in operands if arg in keys}:
                fresh.add(node)
            live[node] = reads
        elif node in keys and keys[node] in mutable:
            live[node] = {keys[node]}
    return [node for node in nodes if node in required]
