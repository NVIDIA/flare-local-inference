"""Persistent startup outputs remain separate from serving inputs and capacities."""

from types import SimpleNamespace
import os

import pytest
import torch

from ignition.preparation.execution import GraphSignature, PreparedExecution, TunedBinding, TuningGraph
from ignition.preparation.startup_tuning import capture_consumers, record_consumer


def execution(**changes):
    values = dict(device=0, max_active=4, graphs={"decode": object()},
        signatures={"decode": GraphSignature("decode", 1, 1, ("tokens",))},
        buffers={"frame_index": torch.zeros(1, dtype=torch.int32),
                 "submission_sequence": torch.zeros(1, dtype=torch.int64),
                 "completion_records": torch.zeros((2, 4, 24), dtype=torch.int64),
                 "tokens": torch.zeros(1, dtype=torch.int64),
                 "selector": torch.ones(1, dtype=torch.int32),
                 "status": torch.ones(1, dtype=torch.int32)},
        tuning_graphs={"choose": TuningGraph(object(), ("selector",), "status")},
        tuned_bindings={"decode": (TunedBinding("selector", "choose"),)})
    return PreparedExecution(**(values | changes))


def test_manifest_has_separate_tuners_without_host_uploads(monkeypatch):
    bundle = execution()
    spans = [(4096 + index * 4096, tensor.numel() * tensor.element_size())
             for index, tensor in enumerate(bundle.buffers.values())]
    monkeypatch.setattr(bundle, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = bundle.manifest()
    assert manifest["abi_version"] == 5
    assert set(bundle.signatures) == set(bundle.estimated_durations_ns) == {"decode"}
    assert set(bundle.graphs) == {"decode", "choose"}
    assert manifest["tuning_graphs"] == [dict(name="choose", handle=12,
                                              outputs=["selector"], status_buffer="status")]
    assert manifest["graphs"][0]["tuned_bindings"] == [dict(buffer="selector", tuning_graph="choose")]
    assert {binding["buffer_name"] for binding in manifest["graphs"][0]["input_bindings"]} == {
        "frame_index", "submission_sequence", "tokens"}


def test_invalid_startup_dependencies_and_uploads_are_rejected():
    with pytest.raises(ValueError, match="output of its tuning graph"):
        execution(tuned_bindings={"decode": (TunedBinding("tokens", "choose"),)})
    with pytest.raises(ValueError, match="host uploads"):
        execution(signatures={"decode": GraphSignature("decode", 1, 1, ("selector",))})
    with pytest.raises(ValueError, match="int32"):
        execution(tuning_graphs={"choose": TuningGraph(object(), ("selector",), "tokens")})
    with pytest.raises(ValueError, match="distinct"):
        execution(tuning_graphs={"decode": TuningGraph(object(), ("selector",), "status")})


def test_capture_dependencies_follow_invoked_consumers_and_graph_identity():
    registry = SimpleNamespace(captures={})
    session = SimpleNamespace(_startup_collectives=registry)
    first, second, unused = object(), object(), object()
    graph, other_graph = object(), object()
    record_consumer(unused)
    with capture_consumers(session, graph):
        record_consumer(first)
        record_consumer(first)
    with capture_consumers(session, other_graph):
        record_consumer(second)
    assert registry.captures == {id(graph): {id(first)}, id(other_graph): {id(second)}}


def test_registry_reuses_producer_until_its_consumers_change(monkeypatch):
    from b12x.comm import pcie
    from ignition.preparation.startup_tuning import CollectiveStartupTuning

    prepared = []
    def prepare(specs, *, cache_policy):
        assert cache_policy == "cold"
        producer = SimpleNamespace(specs=tuple(specs))
        prepared.append(producer)
        return producer
    def spec(plan, *, profiles):
        assert profiles == (("full", 1),)
        return plan
    monkeypatch.setattr(pcie, "joint_startup_spec", spec, raising=False)
    monkeypatch.setattr(pcie, "prepare_joint_startup_tuner", prepare, raising=False)
    registry = CollectiveStartupTuning(object())
    first = SimpleNamespace(key=("latency", 256), selector=torch.zeros(1))
    registry.owners.append(SimpleNamespace(plan=first, runtime_profiles=(("full", 1),), runtime_cache_policy="cold"))
    registry.prepare()
    registry.prepare()
    assert len(prepared) == 1
    second = SimpleNamespace(key=first.key, selector=torch.zeros(1))
    registry.owners.append(SimpleNamespace(plan=second, runtime_profiles=(("full", 1),), runtime_cache_policy="cold"))
    registry.prepare()
    assert len(prepared) == 2
    assert registry.groups[first.key].specs == (first, second)
    registry.prepare()
    assert len(prepared) == 2


def test_standalone_registration_prepares_each_declared_consumer(monkeypatch):
    from ignition.preparation.startup_tuning import CollectiveStartupTuning, register_collective

    snapshots = []
    monkeypatch.setattr(CollectiveStartupTuning, "prepare", lambda self: snapshots.append(tuple(self.owners)))
    session = SimpleNamespace()
    first, second = object(), object()
    register_collective(first, session)
    register_collective(second, session)
    assert snapshots == [(first,), (first, second)]


def test_catalog_retains_exportable_graph_and_selector_storage(monkeypatch):
    from contextlib import contextmanager, nullcontext
    from ignition.preparation.runtime_package import _graphs
    from ignition.preparation.startup_tuning import CollectiveStartupTuning
    from b12x.comm import pcie

    events = []
    class Graph:
        def __init__(self, *, keep_graph):
            self.keep_graph = keep_graph
            self.instantiated = False
        def instantiate(self):
            self.instantiated = True
            events.append("instantiate")
        def raw_cuda_graph(self):
            assert self.keep_graph
            return 11
        def raw_cuda_graph_exec(self):
            assert self.instantiated
            return 12
    class Stream:
        def __init__(self, *, device):
            assert device == 2
        def wait_stream(self, stream):
            assert stream == "current-2"
            events.append("wait")
        def synchronize(self):
            events.append("drain")
    @contextmanager
    def device(ordinal):
        assert ordinal == 2
        events.append("device-enter")
        yield
        events.append("device-exit")
    @contextmanager
    def capture(graph, *, stream, pool):
        assert isinstance(stream, Stream)
        assert pool == (1, 7)
        events.append("capture-enter")
        yield
        events.append("capture-exit")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda ordinal: f"current-{ordinal}")
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda ordinal: None)
    monkeypatch.setattr(pcie, "run_joint_startup_tuners", lambda producers: events.append("run"))
    selector = torch.tensor([7], dtype=torch.uint32)
    owner = SimpleNamespace(plan=SimpleNamespace(prepared=SimpleNamespace(state=SimpleNamespace(selector=selector))))
    producer = SimpleNamespace(specs=(SimpleNamespace(selector=selector),),
        status=torch.ones(1, dtype=torch.int32), run=lambda: events.append("run"))
    registry = CollectiveStartupTuning(SimpleNamespace(device=SimpleNamespace(ordinal=2), capture=nullcontext,
        benchmark_storage=SimpleNamespace(graph_pool=SimpleNamespace(id=(1, 7)))))
    serving = object()
    registry.owners = [owner]
    registry.groups = {"key": producer}
    registry.owner_groups = {id(owner): "key"}
    registry.captures[id(serving)] = {id(owner)}
    buffers, tuners, bindings, resources = registry.catalog({"decode": serving})
    output = buffers[bindings["decode"][0].buffer]
    assert output.dtype == torch.int32 and output.data_ptr() == selector.data_ptr()
    output.fill_(5)
    assert selector.item() == 5
    assert resources == (producer,)
    assert _graphs({name: value.graph for name, value in tuners.items()}) == [
        dict(name="collective-tuning", graph=11, executable=12)]
    assert buffers["collective-tuning-0-status"] is producer.status
    assert tuners["collective-tuning"].status_buffer == "collective-tuning-status"
    assert buffers["collective-tuning-status"].data_ptr() != producer.status.data_ptr()
    assert events == ["device-enter", "wait", "capture-enter", "run", "capture-exit",
                      "instantiate", "drain", "device-exit"]


def test_storage_compiled_collective_records_actual_consumer(monkeypatch):
    from ignition.components.collectives import _OWNERS, _all_reduce
    from ignition.preparation.aot import StorageBackend

    class Owner:
        pass
    owner = Owner()
    owner.plan = object()
    owner.runtime = SimpleNamespace(all_reduce=lambda source, *, out, plan: out.copy_(source))
    handle = 987654321
    monkeypatch.setitem(_OWNERS, handle, owner)
    source, output = torch.arange(8, dtype=torch.float32), torch.zeros(8)
    def forward(source, output):
        _all_reduce(source, output, handle)
        return output
    compiled = StorageBackend(forward, (source, output))
    registry = SimpleNamespace(captures={})
    session = SimpleNamespace(_startup_collectives=registry)
    graph = object()
    with capture_consumers(session, graph):
        compiled(source, output)
    torch.testing.assert_close(output, source)
    assert registry.captures[id(graph)] == {id(owner)}


def test_manifest_emits_explicit_empty_startup_catalog(monkeypatch):
    bundle = execution(tuning_graphs={}, tuned_bindings={})
    spans = [(4096 + index * 4096, tensor.numel() * tensor.element_size())
             for index, tensor in enumerate(bundle.buffers.values())]
    monkeypatch.setattr(bundle, "_validate_resources", lambda: (9, [11], spans))
    manifest = bundle.manifest()
    assert manifest["tuning_graphs"] == []
    assert manifest["graphs"][0]["tuned_bindings"] == []


def test_memory_report_deduplicates_shared_producer_allocations():
    from b12x.preparation import MemoryRequirements, PersistentMemory
    from ignition.preparation.startup_tuning import CollectiveStartupTuning

    shared = PersistentMemory("eviction", 4096, 4096)
    def producer(key):
        memory = MemoryRequirements(persistent=(shared, PersistentMemory(key, 128, 128)))
        return SimpleNamespace(memory_requirements=lambda: memory)
    registry = CollectiveStartupTuning(object())
    registry.groups = {"a": producer("a"), "b": producer("b")}
    assert registry.memory_report() == dict(startup_tuning_bytes=4360, startup_tuning_producers=2)


def test_aggregate_status_preserves_group_failures_across_success_and_failure_replays():
    from ignition.preparation.startup_tuning import _publish_group_status

    first = torch.tensor([0], dtype=torch.int32)
    second = torch.tensor([0], dtype=torch.int32)
    status = torch.tensor([99], dtype=torch.int32)
    scratch = torch.tensor([99], dtype=torch.int32)
    groups = ((1, first), (5, second))
    addresses = tuple(value.data_ptr() for value in (first, second, status, scratch))
    for a, b, expected in ((0, 0, 0), (1, 0, 2), (0, 0, 0), (0, -1, 6), (-1, 9, 6), (0, 0, 0)):
        first.fill_(a)
        second.fill_(b)
        status.zero_()
        scratch.fill_(-7)
        _publish_group_status(groups, status, scratch)
        assert status.item() == expected
        assert first.item() == a and second.item() == b
        assert addresses == tuple(value.data_ptr() for value in (first, second, status, scratch))


def test_catalog_prunes_unused_groups_and_preserves_individual_consumer_fields(monkeypatch):
    from contextlib import nullcontext
    from b12x.comm import pcie
    from ignition.preparation.startup_tuning import CollectiveStartupTuning

    captures, launched = [], []
    class Graph:
        def __init__(self, *, keep_graph):
            assert keep_graph
            captures.append(self)
        def instantiate(self):
            pass
    class Stream:
        def __init__(self, *, device):
            assert device == 0
        def wait_stream(self, stream):
            pass
        def synchronize(self):
            pass
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, stream, pool: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(pcie, "run_joint_startup_tuners", lambda producers: launched.append(tuple(producers)))
    registry = CollectiveStartupTuning(SimpleNamespace(device=SimpleNamespace(ordinal=0), capture=nullcontext,
        benchmark_storage=SimpleNamespace(graph_pool=SimpleNamespace(id=(1, 7)))))
    owners, producers = [], []
    for index in range(3):
        selector = torch.tensor([index + 1], dtype=torch.uint32)
        owner = SimpleNamespace(plan=SimpleNamespace(prepared=SimpleNamespace(state=SimpleNamespace(selector=selector))))
        producer = SimpleNamespace(specs=(SimpleNamespace(selector=selector),), status=torch.zeros(1, dtype=torch.int32))
        owners.append(owner)
        producers.append(producer)
    registry.owners = owners
    registry.groups = dict(enumerate(producers))
    registry.owner_groups = {id(owner): index for index, owner in enumerate(owners)}
    decode, prefill, unused = object(), object(), object()
    registry.captures = {id(decode): {id(owners[0])}, id(prefill): {id(owners[2])}, id(unused): {id(owners[1])}}
    buffers, tuners, bindings, resources = registry.catalog(dict(decode=decode, prefill=prefill))
    assert len(captures) == len(tuners) == 1
    assert resources == launched[0] == (producers[0], producers[2])
    assert tuners["collective-tuning"].outputs == ("collective-tuning-0-selector-0", "collective-tuning-2-selector-0")
    assert bindings == {
        "decode": [TunedBinding("collective-tuning-0-selector-0", "collective-tuning")],
        "prefill": [TunedBinding("collective-tuning-2-selector-0", "collective-tuning")],
    }
    assert "collective-tuning-1-status" not in buffers
    assert buffers["collective-tuning-0-status"] is producers[0].status
    assert buffers["collective-tuning-2-status"] is producers[2].status
    registry.session.benchmark_storage = None
    with pytest.raises(RuntimeError, match="requires package storage"):
        registry.catalog(dict(decode=decode))
    captures.clear()
    assert registry.catalog({"unrecorded": object()}) == ({}, {}, {}, ())
    assert not captures


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")
def test_captured_aggregate_status_reads_changed_group_status_without_allocation():
    from ignition.preparation.startup_tuning import _publish_group_status

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA devices explicitly"
    device = torch.cuda.current_device()
    first = torch.zeros(1, dtype=torch.int32, device=device)
    second = torch.zeros(1, dtype=torch.int32, device=device)
    status = torch.empty(1, dtype=torch.int32, device=device)
    scratch = torch.empty_like(status)
    groups = ((1, first), (5, second))
    _publish_group_status(groups, status, scratch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        status.fill_(-1)
        _publish_group_status(groups, status, scratch)
    for a, b, expected in ((0, 0, 0), (1, 0, 2), (0, 0, 0), (0, -1, 6), (1, 1, 6), (0, 0, 0)):
        first.fill_(a)
        second.fill_(b)
        status.zero_()
        scratch.fill_(99)
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert status.item() == expected
        assert first.item() == a and second.item() == b
    graph.reset()
