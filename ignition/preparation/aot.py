"""Serializable compiled forwards with caller-owned storage and runtime handles."""

import hashlib
import importlib
import inspect
import operator
import os
import pickle
import tempfile
import zlib
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import RLock
from types import MappingProxyType

import torch
from filelock import FileLock
from torch._dynamo.aot_compile_types import SerializableCallable
from torch._dynamo.guards import GuardsStatePickler, _Missing
from torch.fx.experimental.proxy_tensor import make_fx
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_dumps, treespec_loads

from .compilation import (DEFAULT_CACHE_ROOT, _INVOCATIONS, _OPTIONS, _Invocation, _artifact_kernel_table,
                          _byte_storage, _storage_region, _lower_storage, _compiler_recursion)
from .device_binding import bind_device, relocate_inductor, restore_device, restore_prepared_device
from .host_memory import COMPILE_HEADROOM_BYTES, release_unused

_GUARD_LOCK = RLock()
_GUARD_TENSOR_MODE = ContextVar("flare_guard_tensor_mode", default=None)
_DEFERRED_BINDINGS = ContextVar("flare_deferred_bindings", default=False)
_CHECK_BINDINGS = ContextVar("flare_check_bindings", default=False)


def _load_inductor(data, original, *, frozen=False):
    from torch._functorch._aot_autograd.aot_autograd_result import deserialize_bundled_cache_entry
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    entry, kernels, constants = pickle.loads(data)
    if kernels or constants:
        raise ValueError("compiled native launches must use runtime kernel bindings")
    relocate_inductor(entry, original, freeze_tuning=frozen)
    if frozen and entry.compiled_fw.result._triton_bundle is not None:
        from .inductor_tuning import _materialize_selected_cubin

        for tuner in entry.compiled_fw.result._triton_bundle.static_autotuners:
            for result in tuner.kernel.compile_results:
                _materialize_selected_cubin(result)
    serialized = zlib.compress(pickle.dumps((entry, kernels, constants)), level=1)
    compiled = deserialize_bundled_cache_entry(entry)
    if frozen and entry.compiled_fw.result._triton_bundle is not None:
        from .device_binding import _freeze_tuning_winners

        # Torch adds config defaults while rebuilding launchers from a static bundle.
        _freeze_tuning_winners(entry.compiled_fw.result._triton_bundle)
    # Loading materializes bundled files in Triton's cache. Keep the portable copy compressed;
    # live kernel owners retain their binaries, launchers, and argument metadata independently.
    for result in (entry, compiled.serialize()):
        graph = result.compiled_fw.result
        if graph._triton_bundle is not None:
            graph._triton_bundle.kernel_artifacts.clear()
    artifact = AOTCompiledArtifact(compiled)
    artifact._flare_graph = entry.compiled_fw.result
    return artifact, serialized


class _BindingsChecked(Exception):
    pass


class BindingMismatch(ValueError):
    """Prepared tensor storage differs from the compiled input contract."""


def _guard_object(cls):
    return object.__new__(cls)


class _GuardPickler(GuardsStatePickler):
    """Exclude unguarded kernel resources from prepared Python objects."""

    @classmethod
    def _unpickle_tensor(cls, meta_tensor, device, pytype, dispatch_keys_raw, grad, deferred=False):
        if deferred and _DEFERRED_BINDINGS.get():
            pytype = torch._subclasses.FakeTensor
        mode = _GUARD_TENSOR_MODE.get()
        if mode is None:
            return super()._unpickle_tensor(meta_tensor, device, pytype, dispatch_keys_raw, grad)
        value = mode.fake_tensor_converter.from_meta_and_device(
            mode, meta_tensor, device, pytype,
            torch._C.DispatchKeySet.from_raw_repr(dispatch_keys_raw))
        value.grad = grad
        return value

    def reducer_override(self, obj):
        from triton.runtime import JITFunction
        from torch._dynamo.aot_compile_types import _deserialize_triton_kernel, _serialize_triton_kernel

        if isinstance(obj, torch.device):
            return restore_device, (obj.type, obj.index)
        if isinstance(obj, torch._subclasses.FakeTensor):
            with torch._C._DisableTorchDispatch():
                meta = torch.empty_strided(obj.shape, obj.stride(), dtype=obj.dtype,
                                          device="meta", requires_grad=obj.requires_grad)
            from .forward_metadata import _TENSOR_KINDS
            kinds = _TENSOR_KINDS.get()
            kind = torch._subclasses.FakeTensor if kinds is None else kinds.get(obj, torch._subclasses.FakeTensor)
            deferred = kind is torch._subclasses.FakeTensor
            return type(self)._unpickle_tensor, (meta, obj.device, torch.Tensor if deferred else kind,
                torch._C._dispatch_keys(obj).raw_repr(), obj.grad, deferred)
        if isinstance(obj, JITFunction):
            return _deserialize_triton_kernel, (_serialize_triton_kernel(obj),)
        if (hasattr(obj, '__dict__')
                and type(obj).__module__.startswith(('ignition.', 'b12x.'))
                and not inspect.isclass(obj) and not isinstance(obj, (torch.Tensor, torch.nn.Module))):
            state = {key: value if id(value) in self.guard_tree_values
                     else _Missing('unguarded attribute')
                     for key, value in vars(obj).items()}
            from b12x.preparation.device import DetectedDevice

            if isinstance(obj, DetectedDevice):
                return restore_prepared_device, (type(obj), state)
            return _guard_object, (type(obj),), state
        return super().reducer_override(obj)


@contextmanager
def _guard_serialization():
    from torch._dynamo import guards

    with _GUARD_LOCK:
        original = guards.GuardsStatePickler
        guards.GuardsStatePickler = _GuardPickler
        try:
            yield
        finally:
            guards.GuardsStatePickler = original


@contextmanager
def _guard_code_cache(root):
    from torch._dynamo import guards

    with _GUARD_LOCK:
        original = guards.build_guard_function

        def build(code_parts, closure_args):
            key = hashlib.sha256(pickle.dumps((tuple(code_parts), closure_args))).hexdigest()
            path = root / (key + '.bin')
            if path.is_file():
                return pickle.loads(path.read_bytes())
            result = original(code_parts, closure_args)
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=root, suffix='.tmp', delete=False) as file:
                temporary = Path(file.name)
                try:
                    pickle.dump(result, file)
                    file.close()
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            return result

        token = _GUARD_TENSOR_MODE.set(torch._subclasses.FakeTensorMode())
        guards.build_guard_function = build
        try:
            yield
        finally:
            guards.build_guard_function = original
            _GUARD_TENSOR_MODE.reset(token)


def _signature(inputs):
    aliases, result = {}, []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            alias = aliases.setdefault(storage._cdata, len(aliases))
            result.append((tuple(map(int, value.shape)), tuple(map(int, value.stride())),
                           int(value.storage_offset()), value.dtype, value.device, alias))
        elif isinstance(value, (int, torch.SymInt)):
            result.append(int)
        else:
            raise TypeError(f'unsupported compiled input: {type(value)}')
    return tuple(result)


@dataclass
class _Input:
    index: int
    view: tuple | None = None
    region: tuple[int, int] | None = None

    def bind(self, args):
        value = args[self.index]
        if self.view is not None:
            dtype, shape, stride, offset = self.view
            storage = _byte_storage(value) if self.region is None else _storage_region(value, *self.region)
            value = storage.view(dtype).as_strided(shape, stride, offset)
        return value


def _serialized_triton_launch(kernel, constants, arguments):
    """Preserve one traced launch config without serializing an autotuner."""
    from triton.runtime import JITFunction
    from triton.runtime.autotuner import Autotuner, Config

    if isinstance(kernel, JITFunction):
        return kernel, dict(constants)
    if type(kernel) is not Autotuner or len(kernel.configs) != 1 or not isinstance(kernel.fn, JITFunction):
        raise ValueError("opaque Triton serialization requires a JIT function or one fixed launch config")
    config = kernel.configs[0]
    if (type(config) is not Config or config.pre_hook is not None
            or kernel.user_defined_pre_hook or kernel.user_defined_post_hook
            or kernel.reset_to_zero or kernel.restore_value
            or kernel.early_config_prune is not None or kernel.perf_model is not None
            or kernel._do_bench is not None or kernel.num_warmups is not None
            or kernel.num_reps is not None or kernel.use_cuda_graph):
        raise ValueError("opaque Triton serialization cannot discard autotuner hooks or tuning behavior")
    options = config.all_kwargs()
    conflicts = options.keys() & (constants.keys() | arguments.keys())
    if conflicts:
        raise ValueError("opaque Triton launch config conflicts with arguments: " + ", ".join(sorted(conflicts)))
    return kernel.fn, {**constants, **options}


def _deserialized_triton_launch(kernel_info):
    """Resolve the raw JIT whose fixed options were serialized separately."""
    from torch._dynamo.aot_compile_types import _deserialize_triton_kernel, _serialize_triton_kernel
    from triton.runtime import JITFunction
    from triton.runtime.autotuner import Autotuner, Heuristics

    kernel = _deserialize_triton_kernel(kernel_info)
    # Import restores the module global, including decorators removed from the
    # traced launch. Reapplying them can tune again or duplicate fixed options.
    seen = set()
    while type(kernel) in (Autotuner, Heuristics):
        if id(kernel) in seen:
            raise ValueError("serialized fixed Triton launch has cyclic decorators")
        seen.add(id(kernel))
        kernel = kernel.fn
    if not isinstance(kernel, JITFunction):
        raise ValueError("serialized fixed Triton launch does not resolve to a JIT function")
    if _serialize_triton_kernel(kernel) != kernel_info:
        raise ValueError("serialized fixed Triton launch resolves to a different function")
    return kernel


class StorageBackend(SerializableCallable):
    def __init__(self, module, examples, *, options=None):
        from torch._guards import detect_fake_mode
        from torch._inductor import standalone_compile

        module = make_fx(module, tracing_mode='fake', _allow_non_fake_inputs=True,
                         _disable_torch_fn_metadata_mode=True)(*examples)
        graph = module.graph
        arguments = [node for node in graph.nodes if node.op == 'placeholder']
        examples = [node.meta['val'] for node in arguments]
        self.signature = _signature(examples)
        storages, scalar_inputs = {}, {}
        for index, node in enumerate(arguments):
            value = node.meta['val']
            if not isinstance(value, torch.Tensor):
                scalar_inputs[node] = _Input(index)
                continue
            storages.setdefault(value.untyped_storage()._cdata, index)
            name = f'_input_{index}'
            module.register_buffer(name, value)
            with graph.inserting_before(node):
                replacement = graph.get_attr(name)
            replacement.meta = node.meta.copy()
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

        self.invocations = _lower_storage(module)
        bindings = dict(scalar_inputs)
        for node in list(graph.nodes):
            if node.op != 'get_attr':
                continue
            value = operator.attrgetter(node.target)(module)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'compiled constant must be a tensor: {node.target}')
            origin = node.meta.get('storage_origin')
            index = storages[value.untyped_storage()._cdata if origin is None else origin[0]]
            with graph.inserting_before(next(iter(graph.nodes))):
                argument = graph.placeholder(f'buffer_{len(bindings)}')
            argument.meta['val'] = value
            bindings[argument] = _Input(index, (value.dtype, tuple(map(int, value.shape)),
                                              tuple(map(int, value.stride())), int(value.storage_offset())),
                                        None if origin is None else origin[1:])
            node.replace_all_uses_with(argument)
            graph.erase_node(node)
        graph.eliminate_dead_code()
        graph.lint()
        arguments = [node for node in graph.nodes if node.op == 'placeholder']
        self.inputs = [bindings[node] for node in arguments]
        examples = [node.meta['val'] for node in arguments]
        module = torch.fx.GraphModule({}, graph)
        with _compiler_recursion(module):
            self.artifact = standalone_compile(module, examples, dynamic_shapes='from_example_inputs',
                fake_mode=detect_fake_mode(examples), options={'config_patches': dict(_OPTIONS if options is None else options)},
                aot=True, donate_graph_module=True)
        self._serialized_artifact = None
        self._tuning_frozen = False
        self._bound = None

    def __call__(self, *args):
        identity = tuple((id(value), value.untyped_storage()._cdata, value.storage_offset())
                         if isinstance(value, torch.Tensor) else value for value in args)
        if self._bound is None or identity != self._bound[0]:
            signature = _signature(args)
            if signature != self.signature:
                differences = [(index, expected, actual) for index, (expected, actual)
                               in enumerate(zip(self.signature, signature)) if expected != actual]
                raise BindingMismatch('compiled forward input shapes, layouts, or storage aliases changed: '
                    f'{len(self.signature)} expected inputs, {len(signature)} supplied; '
                    f'first differences (index, expected, actual): {differences[:4]}')
            if _CHECK_BINDINGS.get():
                raise _BindingsChecked
            self._bound = identity, [item.bind(args) for item in self.inputs]
        if _CHECK_BINDINGS.get():
            raise _BindingsChecked
        token = _INVOCATIONS.set(self.invocations)
        try:
            return self.artifact(*self._bound[1])
        finally:
            _INVOCATIONS.reset(token)

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        from torch._library.custom_ops import _maybe_get_opdef
        from torch._dynamo.aot_compile_types import _serialize_triton_kernel

        with _artifact_kernel_table():
            artifact = (fn.artifact.serialize() if fn._serialized_artifact is None
                        else zlib.decompress(fn._serialized_artifact))
        calls = []
        for call in fn.invocations:
            target = call.target
            leaves = call.leaves
            if target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
                args, kwargs = tree_unflatten(leaves, call.tree)
                kwargs = dict(kwargs)
                kernel = kernel_side_table.get_kernel(kwargs['kernel_idx'])
                constants = kernel_side_table.get_constant_args(kwargs['constant_args_idx'])
                kernel, constants = _serialized_triton_launch(kernel, constants, kwargs['kwargs'])
                name = ('triton', _serialize_triton_kernel(kernel), constants)
                kwargs['kernel_idx'] = kwargs['constant_args_idx'] = None
                leaves, _ = tree_flatten((args, kwargs))
            elif isinstance(target, torch._ops.OpOverload):
                definition = _maybe_get_opdef(target)
                registration = None if definition is None else definition._init_fn.__module__
                name = ('op', target.namespace, target._opname, target._overloadname, registration)
            else:
                raise TypeError(f'cannot serialize native operator: {target}')
            calls.append((name, leaves, treespec_dumps(call.tree), call.operands, call.integers))
        return pickle.dumps((artifact, fn.inputs, calls, fn.signature, fn._tuning_frozen))

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        fn = cls.__new__(cls)
        artifact, fn.inputs, calls, fn.signature, fn._tuning_frozen = pickle.loads(data)
        devices = {item[4] for item in fn.signature if isinstance(item, tuple) and item[4].type == "cuda"}
        if len(devices) != 1:
            raise ValueError("compiled forwards require exactly one CUDA device")
        original_device = devices.pop()
        fn.signature = tuple((*item[:4], restore_device(item[4].type, item[4].index), item[5])
                             if isinstance(item, tuple) else item for item in fn.signature)
        fn.invocations = []
        for name, leaves, tree, operands, integers in calls:
            tree = treespec_loads(tree)
            if name[0] == 'triton':
                _, kernel, constants = name
                target = torch.ops.higher_order.triton_kernel_wrapper_mutation
                args, kwargs = tree_unflatten(leaves, tree)
                kwargs = dict(kwargs)
                kwargs['kernel_idx'] = kernel_side_table.add_kernel(_deserialized_triton_launch(kernel))
                kwargs['constant_args_idx'] = kernel_side_table.add_constant_args(constants)
                leaves, _ = tree_flatten((args, kwargs))
            else:
                _, namespace, opname, overload, registration = name
                if registration is not None:
                    importlib.import_module(registration)
                target = getattr(getattr(getattr(torch.ops, namespace), opname), overload)
            fn.invocations.append(_Invocation(target, leaves, tree, operands, integers))
        with _artifact_kernel_table():
            fn.artifact, fn._serialized_artifact = _load_inductor(artifact, original_device,
                                                               frozen=fn._tuning_frozen)
        fn._bound = None
        return fn


def _execute(forward):
    return forward()


def aot_forward(forward, *, options=None):
    options = dict(_OPTIONS if options is None else options)
    sources = r'.*\.(?:_?handle|stream_handle|launch_order|plans\[\d+\])$'
    with _guard_serialization(), torch.compiler.config.patch(unbacked_sources=sources):
        return torch.compile(_execute, backend=partial(StorageBackend, options=options), fullgraph=True,
                             dynamic=False).aot_compile(((forward,), {}))


@dataclass
class _GuardedForwardCall:
    function: object
    guards: object
    backend: StorageBackend

    def __call__(self, owner):
        arguments = {"forward": owner}
        if not self.guards.check(arguments):
            raise BindingMismatch(f"prepared forward no longer matches compiled guards: "
                                  f"{self.guards.check_verbose(arguments)}")
        return self.function(owner)


@dataclass
class BoundForward:
    artifact: object
    owner: object
    cache_hit: bool
    pending: object = None

    def __call__(self):
        return self.artifact(self.owner)

    def prepare_capture(self):
        """Retain execution guards and kernel owners after portable cache writes finish."""
        state = self.artifact._artifacts
        if (tuple(state.signature.parameters) != ("forward",) or state.runtime_env.closure
                or state.guard_manager is None or not isinstance(state.compiled_fn, StorageBackend)):
            raise RuntimeError("capture requires a storage-bound, closed compiled forward")
        backend = state.compiled_fn
        released = len(state.guards_state) + len(backend._serialized_artifact or b"")
        backend._serialized_artifact = None
        self.artifact = _GuardedForwardCall(self.artifact.fn, state.guard_manager, backend)
        return released


def _completed(pending):
    while True:
        for index, item in enumerate(pending):
            future = item[-2].pending
            if future is None or future.done():
                return pending.pop(index)
        wait([item[-2].pending for item in pending], return_when=FIRST_COMPLETED)


# Command-line, planning, display, and diagnostic direct-execution sources do not change compiled forwards.
_COMMAND_SOURCES = ('cli', 'direct', 'errors.py', 'planning.py', 'progress.py')
# Nor does writing and signing the package file after preparation.
_PUBLICATION_SOURCES = ('preparation/model_package.py', 'preparation/package_file.py', 'preparation/protection.py')


def _variant_compilers(selections):
    """Assign each selected forward variant to the least loaded rank that selects it.

    `selections[i]` maps each forward name to the variant the group's rank `i` binds. Returns
    {(name, variant): i}; ties go to the lowest rank, so every rank derives the same assignment.
    """
    assigned, compilers = [0] * len(selections), {}
    for name in selections[0]:
        for variant in dict.fromkeys(chosen[name] for chosen in selections):
            compiler = min((index for index, chosen in enumerate(selections) if chosen[name] == variant),
                           key=assigned.__getitem__)
            assigned[compiler] += 1
            compilers[name, variant] = compiler
    return compilers


def _balance_compilations(counts, workers):
    """Spread forwards missing from one host's cache over its ranks' compilers.

    Rank `i` must compile `counts[i]` forwards with `workers[i]` workers; a rank without workers,
    whose GPU lacks room for one, lends all of them. Returns, per rank, the (owner, index)
    forwards it compiles: each rank keeps its own until lending one finishes sooner elsewhere.
    Every rank derives the same assignment.
    """
    if not any(workers):
        raise MemoryError("no rank on this host has room for a forward compiler")
    assigned = [[(owner, index) for index in range(count)] for owner, count in enumerate(counts)]
    able = [rank for rank, count in enumerate(workers) if count]

    def finish(rank, extra=0):
        held = len(assigned[rank]) + extra
        return -(-held // workers[rank]) if workers[rank] else float("inf") if held else 0

    while True:
        busiest = max(range(len(counts)), key=lambda rank: (finish(rank), -rank))
        idlest = min(able, key=lambda rank: (finish(rank, 1), rank))
        if finish(idlest, 1) >= finish(busiest):
            return assigned
        assigned[idlest].append(assigned[busiest].pop())


def _share_artifacts(owned, group, ranks):
    """Copy every rank's compiled forwards to ranks on other hosts, which load them by the same path.

    `owned[i]` lists the artifacts rank `ranks[i]` compiled. Each rank keeps its own cache
    directory; a rank receives only artifacts it lacks or whose contents differ from their
    compiler's.
    """
    import torch.distributed as dist

    me = ranks.index(dist.get_rank())
    digests = [None] * len(ranks)
    dist.all_gather_object(digests, {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                     for path in owned[me]}, group=group)

    def current(path, digest):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        except OSError:
            return False

    missing = {path for index, compiled in enumerate(digests) if index != me
               for path, digest in compiled.items() if not current(path, digest)}
    wanted = [None] * len(ranks)
    dist.all_gather_object(wanted, sorted(missing), group=group)
    wanted = set().union(*wanted)
    for index, compiled in enumerate(digests):
        for path in sorted(wanted.intersection(compiled)):
            payload = [Path(path).read_bytes() if index == me else None]
            dist.broadcast_object_list(payload, src=ranks[index], group=group)
            if path in missing:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as file:
                    file.write(payload[0])
                os.replace(file.name, target)
                missing.discard(path)


class ForwardCache:
    """Load complete compiled forwards before tracing the serving catalog."""

    def __init__(self, *, namespace, device, root=None, headroom_bytes=0, options=None, compile_workers=None,
                 lock_gpu_clocks=False, compile_headroom_bytes=COMPILE_HEADROOM_BYTES):
        import flashinfer
        from torch._inductor.codecache import torch_key

        self.device = torch.device(device)
        self.headroom_bytes = headroom_bytes
        self.compile_workers = compile_workers
        self.lock_gpu_clocks = lock_gpu_clocks
        self.compile_headroom_bytes = compile_headroom_bytes
        self.options = MappingProxyType(dict(_OPTIONS if options is None else options))
        properties = torch.cuda.get_device_properties(device)
        options = {name: value for name, value in self.options.items() if name != 'compile_threads'}
        key = hashlib.sha256(pickle.dumps((namespace, options, torch_key(),
            properties.name, properties.major, properties.minor, properties.multi_processor_count)))
        for source in (Path(__file__).resolve().parents[1], Path(flashinfer.__file__).resolve().parent):
            for path in sorted(source.rglob('*.py')):
                if source.name == 'ignition' and (path.name == '__main__.py'
                                               or path.relative_to(source).parts[0] in _COMMAND_SOURCES
                                               or path.relative_to(source).as_posix() in _PUBLICATION_SOURCES):
                    continue
                key.update(str(path.relative_to(source)).encode())
                key.update(path.read_bytes())
        root = DEFAULT_CACHE_ROOT if root is None else Path(root)
        self.root = root / 'forwards' / key.hexdigest()

    def prepare(self, forward, *, name, signature, progress=None):
        """Bind and warm one callable on the caller's preparation stream."""
        with self._publication_lock(), _guard_code_cache(self.root / 'guards'):
            bound, path = self._prepare(forward, name=name, signature=signature, progress=progress)
            result = self._warm(bound, path, name=name, progress=progress)
            release_unused()
            return result

    def bind_group(self, template, forwards, *, signature, coordinator=None):
        """Bind one compiled forward to owners with matching guards and storage aliases.

        A grouped caller first prepares its representative through ``prepare_many``.
        Every rank checks every owner before any grouped graph is captured; a
        mismatch fails on all ranks and cannot trigger compilation on peers.
        """
        if (not forwards or not isinstance(template, BoundForward) or template.pending is not None
                or not isinstance(template.artifact._artifacts.compiled_fn, StorageBackend)):
            raise ValueError("group binding requires a completed storage-bound forward")
        artifact = template.artifact
        failures = {}
        for name, owner in forwards.items():
            try:
                if type(owner) is not type(template.owner):
                    raise BindingMismatch("grouped forward type differs from its compiled representative")
                if not artifact.guard_check(owner):
                    reason = artifact._artifacts.guard_manager.check_verbose(
                        artifact.prepare_f_locals(owner))
                    raise BindingMismatch(f"compiled forward guard differs: {reason}")
                token = _CHECK_BINDINGS.set(True)
                try:
                    try:
                        artifact(owner)
                    except _BindingsChecked:
                        pass
                    else:
                        raise RuntimeError("grouped forward did not validate storage bindings")
                finally:
                    _CHECK_BINDINGS.reset(token)
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
        if coordinator is None:
            gathered = ((tuple(forwards), signature, failures),)
            ranks = (0,)
        else:
            import torch.distributed as dist

            ranks = coordinator.ranks
            gathered = [None] * len(ranks)
            dist.all_gather_object(gathered, (tuple(forwards), signature, failures), group=coordinator.group)
        if any(names != tuple(forwards) or selected != signature for names, selected, _ in gathered):
            raise BindingMismatch("tensor-parallel ranks declared different grouped forward owners or signatures")
        if any(items for _, _, items in gathered):
            detail = [(rank, name, reason) for rank, (_, _, items) in zip(ranks, gathered, strict=True)
                      for name, reason in sorted(items.items())]
            raise BindingMismatch(f"grouped forward bindings differ from compiled guards: {detail[:4]}")
        return {name: BoundForward(artifact, owner, template.cache_hit)
                for name, owner in forwards.items()}

    @contextmanager
    def _publication_lock(self, coordinator=None):
        """Hold this host's cache against other launches from compilation through binding.

        Cached variants share filenames, so another launch must not replace a catalog still in
        use. Ranks on one host share its cache directory: the host's first rank holds the lock for
        all of them. No rank compiles before every host's lock is held, and no host releases it
        before every rank has bound its forwards.
        """
        host, hosts, holder = None, [None], True
        if coordinator is not None:
            import torch.distributed as dist

            index = coordinator.ranks.index(dist.get_rank())
            host = coordinator.hosts[index]
            holder = coordinator.hosts.index(host) == index
            hosts = sorted(set(coordinator.hosts))
        lock = None
        try:
            # Hosts lock in one order, so launches sharing hosts cannot each hold a lock the other awaits.
            for turn in hosts:
                if holder and turn == host:
                    self.root.mkdir(parents=True, exist_ok=True)
                    lock = FileLock(self.root / 'publication.lock')
                    lock.acquire()
                if coordinator is not None:
                    dist.barrier(group=coordinator.group)
            yield
            # A failure skips this barrier: the failing rank's peers learn of it collectively.
            if coordinator is not None:
                dist.barrier(group=coordinator.group)
        finally:
            if lock is not None:
                lock.release()

    def prepare_many(self, functions, signatures, *, progress=None, coordinator=None, materialize=None,
                     warmup=nullcontext, before_run=None):
        """Trace and compile in workers, then bind and warm on the owning rank.

        ``before_run`` restores mutable input state before every parent-rank
        binding check and warmup; worker compilation uses tensor metadata.
        """
        # Compiler workers own parallel compilation; artifact binding needs no second worker pool.
        with self._publication_lock(coordinator), torch._inductor.config.patch(compile_threads=1):
            return self._prepare_many(functions, signatures, progress=progress,
                                      coordinator=coordinator, materialize=materialize, warmup=warmup,
                                      before_run=before_run)

    def _prepare_many(self, functions, signatures, *, progress, coordinator, materialize, warmup,
                      before_run):
        from ignition.progress import phase
        from .compile_pool import CompilePool
        from .inductor_tuning import InductorSelections

        if coordinator is not None:
            return self._prepare_ranks(functions, signatures, coordinator, progress=progress,
                                       materialize=materialize, warmup=warmup, before_run=before_run)
        selections = InductorSelections(self.device, lock_gpu_clocks=self.lock_gpu_clocks)
        compiled, paths, pending = {}, {}, []
        with phase(progress, "Compile model forwards", total=len(functions)) as stage, CompilePool(self.device,
                headroom_bytes=self.headroom_bytes, options=self.options, compile_workers=self.compile_workers,
                compile_headroom_bytes=self.compile_headroom_bytes) as compiler:
            def finish():
                name, bound, path = _completed(pending)
                try:
                    with _guard_code_cache(self.root / 'guards'):
                        if bound.pending is not None:
                            bound.pending.result()
                        if materialize is None:
                            if not bound.cache_hit:
                                bound = self._load(bound.owner, path, required=True,
                                                   before_run=None if before_run is None else lambda: before_run(name))
                                bound.cache_hit = False
                            compiled[name] = self._warm(bound, path, name=name, progress=None,
                                                       selections=selections, warmup=warmup,
                                                       before_run=None if before_run is None else lambda name=name: before_run(name))
                        else:
                            paths[name] = path
                except Exception as error:
                    raise RuntimeError(f"failed to prepare compiled forward {name}") from error
                stage.update(advance=1)

            for name, forward in functions.items():
                with _guard_code_cache(self.root / 'guards'):
                    bound, path = self._prepare(forward, name=name, signature=signatures[name], compiler=compiler,
                                                deferred=materialize is not None,
                                                before_run=None if before_run is None else lambda name=name: before_run(name))
                pending.append((name, bound, path))
                if len(pending) >= compiler.workers:
                    finish()
            while pending:
                finish()
        release_unused()
        if materialize is not None:
            materialize()
            with phase(progress, "Warm compiled forwards", total=len(functions)) as stage:
                for name, forward in functions.items():
                    with _guard_code_cache(self.root / 'guards'):
                        try:
                            bound = self._load(forward, paths[name], required=True,
                                               before_run=None if before_run is None else lambda name=name: before_run(name))
                        except BindingMismatch:
                            bound = BoundForward(aot_forward(forward, options=self.options), forward, False)
                        compiled[name] = self._warm(bound, paths[name], name=name, progress=None,
                                                   selections=selections, warmup=warmup,
                                                   before_run=None if before_run is None else lambda name=name: before_run(name))
                    stage.update(advance=1)
        release_unused()
        return compiled

    def _prepare_ranks(self, functions, signatures, coordinator, *, progress, materialize, warmup,
                       before_run):
        """Compile the catalog for a rank group, then bind and warm it on every rank.

        Compiled forwards serve every rank, so the catalog compiles once: each forward variant a
        rank selects through `compilation_variant` compiles on a rank that selects it, from that
        rank's own forward, unless an idle rank on the same host borrows it and compiles it from
        the owner's traced forward metadata. Compiler pools are budgeted for the host's ranks.
        Ranks on one host share its cache directory; a rank on another host receives the
        artifacts it lacks from the ranks that compiled them. Every rank then binds the same
        artifacts.
        """
        import torch.distributed as dist
        from ignition.progress import phase
        from .compile_pool import CompilePool
        from .inductor_tuning import InductorSelections

        group, ranks = coordinator.group, coordinator.ranks
        share = ranks.index(dist.get_rank())
        selections = [None] * len(ranks)
        dist.all_gather_object(selections, {name: getattr(forward, "compilation_variant", "shared")
                                            for name, forward in functions.items()}, group=group)
        if any(chosen.keys() != selections[0].keys() for chosen in selections):
            raise RuntimeError("ranks prepared different forward catalogs")
        compilers = _variant_compilers(selections)
        work = [(name, variant, functions[name]) for (name, variant), compiler in compilers.items() if compiler == share]
        selected, outcome, error, compiler, paths, lent, sent = {}, (None, None), None, None, {}, set(), []
        unavailable, loans = None, []
        with ExitStack() as scope:
            stage = scope.enter_context(phase(progress, "Compile model forwards", total=len(work)))
            try:
                compiler = scope.enter_context(CompilePool(
                    self.device, headroom_bytes=self.headroom_bytes, pools=coordinator.local_ranks,
                    options=self.options, compile_workers=self.compile_workers,
                    compile_headroom_bytes=self.compile_headroom_bytes))
            except MemoryError as failure:
                # A rank whose GPU has no room for a compiler lends its forwards to the others on this host.
                unavailable = f"{type(failure).__name__}: {failure}"
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            # Variants that only one rank selects, such as rank zero's sampling, can leave that rank most of
            # the catalog. Ranks on one host share its cache, so a rank lends forwards it cannot restore, whether
            # missing or bound differently by another launch, to ranks whose compilers would otherwise finish
            # first; the borrower compiles the owner's metadata.
            restored, missing = {}, []
            try:
                for index, (name, variant, forward) in enumerate(work):
                    with _guard_code_cache(self.root / 'guards'):
                        bound, path = self._restore(forward, name=f"{name}.{variant}", signature=signatures[name],
                                                    deferred=materialize is not None,
                                                    before_run=None if before_run is None else lambda name=name: before_run(name))
                    if bound is None:
                        missing.append(index)
                    else:
                        restored[index] = bound, path
            except Exception as failure:
                error = error or f"{type(failure).__name__}: {failure}"
            demand = [None] * len(ranks)
            dist.all_gather_object(demand, (len(missing), 0 if compiler is None else compiler.workers), group=group)
            if not coordinator.spans_hosts and any(workers for _, workers in demand):
                assigned = _balance_compilations([count for count, _ in demand], [workers for _, workers in demand])
                loans = [(borrower, missing[index]) for borrower, forwards in enumerate(assigned)
                         for owner, index in forwards if owner == share and borrower != share]
                lent = {work[index][:2] for _, index in loans}
                try:
                    from .forward_metadata import dumps

                    jobs = self.root / 'lent'
                    jobs.mkdir(parents=True, exist_ok=True)
                    for borrower, index in loans:
                        name, variant, forward = work[index]
                        path = self._path(forward, name=f"{name}.{variant}", signature=signatures[name])
                        job = jobs / f"{path.stem}.forward"
                        with tempfile.NamedTemporaryFile(dir=jobs, suffix='.tmp', delete=False) as file:
                            pickle.dump((dumps(forward), str(path)), file)
                        os.replace(file.name, job)
                        sent.append((borrower, str(job)))
                        # The borrower compiles it before every rank gathers the outcome below.
                        paths[name, variant] = str(path), True
                except Exception as failure:
                    error = error or f"{type(failure).__name__}: {failure}"
            mailed = [None] * len(ranks)
            dist.all_gather_object(mailed, sent, group=group)
            borrowed = [job for jobs in mailed for borrower, job in jobs if borrower == share]
            try:
                if error is not None:
                    raise RuntimeError(error)
                if compiler is None and (borrowed or any(index not in {i for _, i in loans} for index in missing)
                                         if not coordinator.spans_hosts else missing):
                    raise RuntimeError(unavailable)
                pending = []

                def finish():
                    name, variant, bound, path = _completed(pending)
                    if bound.pending is not None:
                        bound.pending.result()
                    elif materialize is None:
                        # Restored and bound to this rank's own forward already.
                        selected[name] = bound
                    if name is not None:
                        # Each forward's path, and whether this launch compiled it rather than restoring it.
                        paths[name, variant] = str(path), bound.pending is not None
                        stage.update(advance=1)

                for index, (name, variant, forward) in enumerate(work):
                    if (name, variant) in lent:
                        continue
                    if index in restored:
                        bound, path = restored[index]
                    else:
                        path = self._path(forward, name=f"{name}.{variant}", signature=signatures[name])
                        bound = BoundForward(None, forward, False, compiler.submit(forward, path))
                    pending.append((name, variant, bound, path))
                    if compiler is not None and len(pending) >= compiler.workers:
                        finish()
                for job in borrowed:
                    with open(job, 'rb') as file:
                        payload, path = pickle.load(file)
                    os.unlink(job)
                    future = compiler.submit_payload(payload, Path(path))
                    pending.append((None, None, BoundForward(None, None, False, future), path))
                    if len(pending) >= compiler.workers:
                        finish()
                while pending:
                    finish()
                outcome = paths, None
            except Exception as failure:
                outcome = None, f"{type(failure).__name__}: {failure}"
            outcomes = [None] * len(ranks)
            dist.all_gather_object(outcomes, outcome, group=group)
            # Forwards lent to other ranks are complete once every rank reports.
            stage.update(advance=len(lent))
        failures = [(index, failure) for index, (_, failure) in zip(ranks, outcomes) if failure is not None]
        if failures:
            raise RuntimeError(f"forward compilation failed: {failures}")
        restored.clear()
        bound = None
        paths = {key: value for compiled, _ in outcomes for key, value in compiled.items()}
        # Ranks on one host compiled into its cache directory; only other hosts lack their shares.
        if coordinator.spans_hosts:
            _share_artifacts([sorted({path for path, _ in compiled.values()}) for compiled, _ in outcomes],
                             group, ranks)
        error = None
        try:
            release_unused()
            if materialize is not None:
                materialize()
            with phase(progress, "Load compiled forwards", total=len(functions) - len(selected)) as stage:
                for name, forward in functions.items():
                    if name in selected:
                        continue
                    path, compiled = paths[name, getattr(forward, "compilation_variant", "shared")]
                    with _guard_code_cache(self.root / 'guards'):
                        selected[name] = self._load(forward, Path(path), required=True,
                            before_run=None if before_run is None else lambda name=name: before_run(name))
                    selected[name].cache_hit = not compiled
                    stage.update(advance=1)
        except Exception as failure:
            import traceback

            traceback.print_exc(limit=-40)
            error = f"{type(failure).__name__}: {failure}"
        errors = [None] * len(ranks)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"compiled forward binding failed: {errors}")
        release_unused()
        selections = InductorSelections(self.device, coordinator=coordinator,
                                        lock_gpu_clocks=self.lock_gpu_clocks)
        failure = None
        try:
            for bound in selected.values():
                selections.register(bound)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        selections._check(failure)
        with phase(progress, "Warm compiled forwards", total=len(functions)) as stage:
            for name in functions:
                coordinator.barrier(f"compiled warmup {name}", ranks)
                try:
                    selections.warm(selected[name], warmup,
                                    before_run=None if before_run is None else lambda name=name: before_run(name))
                except Exception as error:
                    error.add_note(f"Warming compiled forward {name!r}")
                    raise
                stage.update(advance=1)
        published, failure = [], None
        try:
            for name, bound in selected.items():
                selections.freeze(bound)
                variant = getattr(functions[name], 'compilation_variant', 'shared')
                if compilers[name, variant] == share:
                    path, _ = paths[name, variant]
                    bound.artifact.save_compiled_function(path)
                    published.append(path)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        selections._check(failure)
        if coordinator.spans_hosts:
            _share_artifacts(selections._gather(published), group, ranks)
        return selected

    def _load(self, forward, path, *, required=False, before_run=None):
        with bind_device(self.device, freeze_tuning=self.options.get(
                'triton.autotune_at_compile_time', False)), path.open('rb') as file:
            artifact = torch.compiler.load_compiled_function(file)
        # Bound forwards remain live for the complete catalog. Collect young
        # deserialization temporaries here; scan the retained heap once at each
        # preparation phase boundary instead of once per restored forward.
        release_unused(generation=0)
        if not artifact.guard_check(forward):
            if required:
                reason = artifact._artifacts.guard_manager.check_verbose(artifact.prepare_f_locals(forward))
                raise BindingMismatch(f"rank-local forward does not match compiled guards: {reason}")
            return None
        # Dynamo guards omit some offsets and aliases checked by the storage backend.
        token = _CHECK_BINDINGS.set(True)
        try:
            if before_run is not None:
                before_run()
            artifact(forward)
        except _BindingsChecked:
            return BoundForward(artifact, forward, True)
        except BindingMismatch:
            if required:
                raise
            return None
        finally:
            _CHECK_BINDINGS.reset(token)
        raise RuntimeError("compiled forward did not validate its storage bindings")

    def _path(self, forward, *, name, signature):
        identity = (type(forward).__module__, type(forward).__qualname__, name, signature)
        return self.root / (hashlib.sha256(pickle.dumps(identity)).hexdigest() + '.bin')

    def _restore(self, forward, *, name, signature, deferred=False, report=None,
                 before_run=None):
        """The cached forward bound to `forward`, or None when it is missing or its guards reject it, and its path."""
        path = self._path(forward, name=name, signature=signature)
        if not path.is_file():
            return None, path
        if report is not None:
            report("restoring compiled forward")
        token = _DEFERRED_BINDINGS.set(deferred)
        try:
            bound = self._load(forward, path, before_run=before_run)
        finally:
            _DEFERRED_BINDINGS.reset(token)
        if report is not None:
            report("checking compiled bindings")
        return bound, path

    def _prepare(self, forward, *, name, signature, compiler=None, progress=None, deferred=False,
                 before_run=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")
        bound, path = self._restore(forward, name=name, signature=signature, deferred=deferred,
                                    report=report, before_run=before_run)
        if bound is not None:
            return bound, path
        report("tracing and compiling forward")
        if compiler is not None:
            return BoundForward(None, forward, False, compiler.submit(forward, path)), path
        return BoundForward(aot_forward(forward, options=self.options), forward, False), path

    def _warm(self, bound, path, *, name, progress=None, selections=None, warmup=nullcontext,
              before_run=None):
        def report(operation):
            if progress is not None:
                progress.update(f"{name} · {operation}")

        from .inductor_tuning import InductorSelections

        if selections is None:
            selections = InductorSelections(self.device, lock_gpu_clocks=self.lock_gpu_clocks)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(bound.artifact._artifacts.compiled_fn.artifact, '_flare_graph'):
            bound.artifact.save_compiled_function(str(path))
            cached = bound.cache_hit
            bound = self._load(bound.owner, path, required=True, before_run=before_run)
            bound.cache_hit = cached
        report("warming restored forward" if bound.cache_hit else "warming compiled forward")
        selections.register(bound)
        selections.warm(bound, warmup, before_run=before_run)
        selections.freeze(bound)
        report("saving compiled forward")
        bound.artifact.save_compiled_function(str(path))
        return bound
