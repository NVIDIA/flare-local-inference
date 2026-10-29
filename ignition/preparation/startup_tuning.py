"""Captured collective consumers and their prepared startup selector programs."""

from contextlib import contextmanager
from contextvars import ContextVar

from .execution import TunedBinding, TuningGraph

_CAPTURE = ContextVar("startup_tuning_capture", default=None)


def replay_startup_tuning(model, stream):
    """Initialize diagnostic graph bindings before requests on every rank."""
    import torch
    import os
    import time

    manifest = model.manifest()
    referenced = {binding["tuning_graph"] for graph in manifest["graphs"]
                  for binding in graph.get("tuned_bindings", ())}
    names = tuple(graph["name"] for graph in manifest.get("tuning_graphs", ())
                  if graph["name"] in referenced)
    buffers = {buffer["name"]: buffer for buffer in manifest["buffers"]}
    storage = {field for graph in manifest.get("tuning_graphs", ()) if graph["name"] in referenced
               for field in (*graph["outputs"], graph["status_buffer"])}
    signature = (
        tuple((graph["name"], tuple(graph["outputs"]), graph["status_buffer"])
              for graph in manifest.get("tuning_graphs", ()) if graph["name"] in referenced),
        tuple((graph["name"], tuple((binding["buffer"], binding["tuning_graph"])
                                   for binding in graph.get("tuned_bindings", ())))
              for graph in manifest["graphs"]),
        tuple((name, buffers[name]["dtype"], buffers[name]["nbytes"],
               tuple(buffers[name]["shape"]), tuple(buffers[name]["strides"])) for name in sorted(storage)),
    )
    coordinator = getattr(model.session, "_coordinator", None)
    if coordinator is not None and not coordinator.same_on_every_rank(signature):
        raise RuntimeError("diagnostic startup tuning graph order or binding schema differs across ranks")
    for name in names:
        tuner = model.tuning_graphs[name]
        status = model.buffers[tuner.status_buffer]
        error = None
        try:
            with torch.cuda.stream(stream):
                status.fill_(-1)
                tuner.graph.replay()
        except BaseException as failure:
            error = failure
        deadline = time.monotonic() + 120
        try:
            while not stream.query():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"diagnostic startup tuning graph {name!r} did not drain")
                time.sleep(.001)
        except BaseException:
            # A rank may still access its peers' storage; process teardown must reclaim it.
            os.abort()
        if error is not None:
            # A partial launch can leave a peer waiting inside its graph.
            os.abort()
        value = int(status.item())
        agreed = coordinator is None or coordinator.same_on_every_rank(value)
        if value != 0 or not agreed:
            raise RuntimeError(f"diagnostic startup tuning graph {name!r} failed: local status {value}")


def record_consumer(owner):
    capture = _CAPTURE.get()
    if capture is not None:
        capture.add(id(owner))


@contextmanager
def capture_consumers(session, graph):
    consumers = set()
    token = _CAPTURE.set(consumers)
    try:
        yield
    finally:
        _CAPTURE.reset(token)
    registry = getattr(session, "_startup_collectives", None)
    if registry is not None:
        registry.captures[id(graph)] = consumers


class CollectiveStartupTuning:
    def __init__(self, session):
        self.session = session
        self.owners = []
        self.captures = {}
        self.groups = {}
        self.owner_groups = {}
        self.group_selectors = {}

    def prepare(self):
        from b12x.comm.pcie import joint_startup_spec, prepare_joint_startup_tuner

        groups, owner_groups = {}, {}
        for owner in self.owners:
            spec = joint_startup_spec(owner.plan, profiles=owner.runtime_profiles)
            groups.setdefault(spec.key, []).append(spec)
            owner_groups[id(owner)] = spec.key
        selectors = {key: tuple(spec.selector.data_ptr() for spec in specs) for key, specs in groups.items()}
        policies = {owner_groups[id(owner)]: owner.runtime_cache_policy for owner in self.owners}
        self.groups = {key: self.groups[key] if self.group_selectors.get(key) == selectors[key]
                       else prepare_joint_startup_tuner(specs, cache_policy=policies[key]) for key, specs in groups.items()}
        self.group_selectors = selectors
        self.owner_groups = owner_groups

    def memory_report(self):
        from b12x.preparation import MemoryRequirements

        memory = MemoryRequirements.sequential(producer.memory_requirements() for producer in self.groups.values())
        # The catalog adds one aggregate status and one int32 scratch scalar.
        return {"startup_tuning_bytes": sum(entry.resident_nbytes for entry in memory.persistent) + (8 if self.groups else 0),
                "startup_tuning_producers": len(self.groups)}

    def catalog(self, graphs):
        import torch
        from b12x.comm.pcie import run_joint_startup_tuners

        consumers = {name: self.captures.get(id(graph), set()) for name, graph in graphs.items()}
        needed = {self.owner_groups[owner] for owners in consumers.values() for owner in owners
                  if owner in self.owner_groups}
        storage = self.session.benchmark_storage
        if needed and storage is None:
            raise RuntimeError("collective startup capture requires package storage")
        buffers, tuners, bindings, resources = {}, {}, {}, []
        statuses, all_outputs = [], []
        tuning_name = "collective-tuning"
        for index, (key, producer) in enumerate(self.groups.items()):
            if key not in needed:
                continue
            name = f"collective-tuning-{index}"
            outputs, addresses = [], {}
            for spec in producer.specs:
                pointer = spec.selector.data_ptr()
                if pointer not in addresses:
                    field = f"{name}-selector-{len(outputs)}"
                    addresses[pointer] = field
                    outputs.append(field)
                    buffers[field] = spec.selector.view(torch.int32)
            status = f"{name}-status"
            buffers[status] = producer.status
            all_outputs.extend(outputs)
            statuses.append((index, producer.status))
            resources.append(producer)
            for serving, owners in consumers.items():
                selected = {id(owner): owner for owner in self.owners if id(owner) in owners
                            and self.owner_groups[id(owner)] == key}
                fields = {addresses[owner.plan.prepared.state.selector.data_ptr()] for owner in selected.values()}
                bindings.setdefault(serving, []).extend(TunedBinding(field, tuning_name) for field in sorted(fields))
        if not resources:
            return buffers, tuners, bindings, ()
        device = self.session.device.ordinal
        status_name = f"{tuning_name}-status"
        scratch_name = f"{status_name}-scratch"
        coordinator = getattr(self.session, "_coordinator", None)
        if coordinator is not None:
            coordinator.barrier(f"capture {tuning_name}", coordinator.ranks)
        with torch.cuda.device(device):
            status = resources[0].status.new_full((1,), -1)
            scratch = status.new_empty((1,))
            buffers[status_name], buffers[scratch_name] = status, scratch
            _publish_group_status(statuses, status, scratch)
            status.fill_(-1)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with self.session.capture(), torch.cuda.graph(graph, pool=storage.graph_pool.id, stream=stream):
                status.fill_(-1)
                run_joint_startup_tuners(resources)
                _publish_group_status(statuses, status, scratch)
            graph.instantiate()
            stream.synchronize()
        tuners[tuning_name] = TuningGraph(graph, tuple(all_outputs), status_name)
        torch.cuda.synchronize(device)
        return buffers, tuners, bindings, tuple(resources)


def _publish_group_status(groups, status, scratch):
    """Publish zero or the highest failing prepared-group index plus one.

    Per-group status buffers retain the original failure values. The aggregate
    identifies a failing group without interpreting its protocol in native code.
    All operations write caller-owned scalars and are captured with the race.
    """
    import torch

    status.zero_()
    for index, group_status in groups:
        torch.ne(group_status, 0, out=scratch)
        scratch.mul_(index + 1)
        torch.maximum(status, scratch, out=status)


def register_collective(owner, session):
    from .tuning import RecordedPreparationSession, register_finalizer

    registry = getattr(session, "_startup_collectives", None)
    if registry is None:
        registry = session._startup_collectives = CollectiveStartupTuning(session)
        registry.owners.append(owner)
        register_finalizer(session, registry.prepare)
    else:
        registry.owners.append(owner)
        if not isinstance(session, RecordedPreparationSession):
            registry.prepare()
