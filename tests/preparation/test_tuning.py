from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from threading import Event
from types import SimpleNamespace
import weakref

import pytest
import torch
from b12x.preparation import FrozenMapping, PreparationSession, MemoryRequirements, PreparedCall
from b12x._lib.scratch import ScratchBufferSpec

from ignition.preparation.scratch import ScratchArena, scratch_storage

from ignition.preparation.tuning import (
    RecordedPreparationSession,
    benchmark_resource,
    declare,
    register_finalizer,
    _local_tuning_catalog,
)


def make_session(**kwargs):
    return RecordedPreparationSession(workload_schema="test-operators-v1",
        numerical_policy="default", **kwargs)


@pytest.mark.parametrize("borrowed_state", [False, True])
def test_recurrent_candidates_share_inputs_with_active_resource_eviction(backend, borrowed_state):
    from ignition.preparation.fixtures.recurrent import benchmark_call, register_benchmark_state

    state = torch.zeros(3, 1, 2, 2)
    arguments = {
        "mixed_qkv": torch.zeros(1, 6, dtype=torch.bfloat16),
        "recurrent_state": state, "output": torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        "query_start_loc": torch.zeros(2, dtype=torch.int32),
        "num_tokens": torch.zeros(1, dtype=torch.int32), "num_seqs": torch.zeros(1, dtype=torch.int32),
        "num_accepted_tokens": torch.ones(1, dtype=torch.int32),
        "state_indices": torch.zeros(1, 1, dtype=torch.int64),
    }
    workspace = torch.empty(64, dtype=torch.uint8)
    specs = (SimpleNamespace(shape=(64,), nbytes=64, dtype=torch.uint8, device=workspace.device),)
    prepared = SimpleNamespace(bind=lambda **kwargs: SimpleNamespace(**kwargs), run=lambda binding: None)
    session = make_session(device="cuda:0", checkpoint="recurrent-fixture-reuse")
    if borrowed_state:
        register_benchmark_state(session, state, storage=torch.empty(state.nbytes, dtype=torch.uint8),
                                 restore=lambda: None)
    session._tune_status = "tuning"
    try:
        calls = [benchmark_call(prepared, arguments=arguments, session=session, name="decode",
            mode="decode", scratch_specs=lambda prepared: specs, workspace=workspace, run_options={})
            for _ in range(4)]
        assert all(call.owners[0] is calls[0].owners[0] for call in calls)
        assert all(call.output is calls[0].output for call in calls)
    finally:
        session.close()


@pytest.fixture
def backend(monkeypatch):
    events = []
    controls = SimpleNamespace(cancel=False, skip_prime=set(), close_error=False)

    def initialize(self, *, device, namespace, **kwargs):
        assert type(kwargs["autotune"]) is bool and kwargs["compile_workers"] == 0
        self.autotune = kwargs["autotune"]
        self.prime = kwargs.get("prime", True)
        self.device = SimpleNamespace(ordinal=0)
        self.namespace = FrozenMapping(namespace)
        self._stop = Event()
        self.compile_workers = 0
        self.race_budget = kwargs.get("race_budget")
        self._published_owners = {}

    def prepare(self, requests, *, autotune, **kwargs):
        events.append(("prepare", autotune, tuple(item.name for item in requests)))
        retained_trials = []
        for request in requests:
            if autotune and request.benchmark_call is not None:
                trial = request.benchmark_call(request.plan)
                if getattr(controls, "retain_trials", False):
                    retained_trials.append(trial)
            request.plan.selection = SimpleNamespace(
                source=request.plan.source if autotune else "default", config={"tile": 32})
            if request.name not in controls.skip_prime:
                call = request.prepare_call(request.plan)
                if call is not None and self.prime:
                    if call.produce is not None:
                        call.produce()
                    call.invoke()
                # b12x PreparationJob._publish retains this tuple until plan release.
                self._published_owners[request.name] = () if call is None else tuple(call.owners)
        if autotune and controls.cancel:
            self._stop.set()

        def close():
            events.append(("result_close", autotune))
            if controls.close_error:
                raise RuntimeError("result close failed")
            if autotune and getattr(controls, "retain_trials", False):
                controls.before_trial_close()
                retained_trials.clear()

        return SimpleNamespace(coverage={}, cache_hits=2 if autotune else 0,
            benchmarked_candidates=7 if autotune else 0, restored_candidates=3 if autotune else 0,
            rejected_candidates=0, elapsed_seconds=1.25,
            parent_cute_compilations=0, parent_triton_compilations=0, close=close)

    def envelope(self, requests):
        events.append(("envelope", tuple(item.name for item in requests)))
        return MemoryRequirements.sequential(MemoryRequirements(
            scratch=(ScratchBufferSpec(name="workspace", shape=(item.plan.bytes,),
                                       dtype=torch.uint8, device=torch.device("cpu")),),
            persistent=getattr(item.plan, "persistent", ())) for item in requests)

    def workers(self, count):
        events.append(("workers", count))
        self.compile_workers = 8 if count is None else count

    @contextmanager
    def capture(self):
        events.append(("capture",))
        yield

    monkeypatch.setattr(PreparationSession, "__init__", initialize)
    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    monkeypatch.setattr(PreparationSession, "candidate_memory_envelope", envelope)
    monkeypatch.setattr(PreparationSession, "configure_compile_workers", workers)
    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: 128 << 30)
    monkeypatch.setattr(PreparationSession, "_check_thread", lambda self: None)
    monkeypatch.setattr(PreparationSession, "_synchronize", lambda self: events.append(("drain",)))
    monkeypatch.setattr(PreparationSession, "capture", capture)
    monkeypatch.setattr(PreparationSession, "freeze", lambda self: events.append(("freeze",)))
    def close_session(self):
        events.append(("close",))
        self._published_owners.clear()

    monkeypatch.setattr(PreparationSession, "close", close_session)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (24 << 30, 32 << 30))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty_cache",)))
    return events, controls


@dataclass
class Request:
    name: str
    plan: object
    prepare_call: object
    benchmark_call: object = None
    collective: object = None
    retain_benchmark_call: bool = False
    dependencies: tuple[str, ...] = ()
    finalist_policy: object = None


def test_context_choice_rebinds_declared_plan_owner_and_parallel_scratch(backend):
    session = make_session(device=0, checkpoint="context-bindings")
    session.device.identity = "test-device"
    arena = ScratchArena("cpu")
    arena.reserve(1024)
    session.register_scratch(arena)

    class Plan:
        def __init__(self, *, override=None, size=256):
            self.override = override
            self.prepared = SimpleNamespace(device=SimpleNamespace(identity="test-device"))
            self.selection = SimpleNamespace(source="tuned" if override is None else "override",
                                              config={"tile": 32 if override is None else override})
            self.size = size

        def same_declaration_except_override(self, other):
            return isinstance(other, Plan)

        def memory_requirements(self):
            return SimpleNamespace(scratch_nbytes=self.size)

    original, alternative, writer = Plan(), Plan(override=64, size=512), Plan(size=128)
    request = Request("q_b", original, lambda state: None, finalist_policy="families")
    session.declare((request,))
    arena.register_parallel(original, writer)
    owner = SimpleNamespace(plan=original)
    session.register_context_plan("q_b", original, lambda plan: setattr(owner, "plan", plan))
    session._context_original_requests = dict(session._recorded_requests)
    session._tune_status = "contextual"

    previous = session.bind_context_plans({"q_b": alternative})
    assert previous == {"q_b": original}
    assert owner.plan is alternative and session._recorded_requests["q_b"].plan is alternative
    assert session._recorded_requests["q_b"].finalist_policy is None
    assert arena.parallel_nbytes(lambda plan: plan.memory_requirements().scratch_nbytes) == 768
    session.preview_context_bindings()
    assert not arena.frozen and session._context_scratch_owner is arena.tensor
    session.close()


def test_finalist_policy_applies_to_every_equivalent_declaration(backend):
    session = make_session(device=0, checkpoint="context-families")
    session.declare((declaration("first"), declaration("second"), declaration("different", size=256)))
    session._set_finalist_policies({"first": "candidate-families"})
    assert session._recorded_requests["first"].finalist_policy == "candidate-families"
    assert session._recorded_requests["second"].finalist_policy == "candidate-families"
    assert session._recorded_requests["different"].finalist_policy is None
    session.close()


def declaration(name, *, component="sequence.gdn_prefill", size=128, selected_size=None, source="tuned",
                prepare=lambda state: None, benchmark=None):
    requirement = MemoryRequirements(scratch=(ScratchBufferSpec(name="workspace",
        shape=(size if selected_size is None else selected_size,), dtype=torch.uint8,
        device=torch.device("cpu")),))
    plan = SimpleNamespace(component_id=component, bytes=size, source=source, selection=None,
        memory_requirements=lambda: requirement,
        query=size, invocation=FrozenMapping(), override=None, _device=None,
        contract=SimpleNamespace(component_id=component, query_schema_version=1,
            config_schema_version=1, semantic_version=1, candidate_contract_version=1,
            encode_query=lambda query: {"size": query},
            config_payload=lambda config: FrozenMapping(config)))
    return Request(name=name, plan=plan, prepare_call=prepare, collective=None,
                           benchmark_call=benchmark, retain_benchmark_call=False)


def test_declarations_do_no_work_and_prepare_in_one_tuned_batch(backend):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision")
    assert session.prime is False
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    calls = []
    requests = []
    for name in ("projection", "attention", "experts", "recurrence"):
        request = declaration(name, prepare=lambda state: calls.append("prime"),
                              benchmark=lambda state: calls.append("benchmark"))
        requests.append(request)
        declare(session, (request,))
    register_finalizer(session, lambda: calls.append("bind"), workspace=arena)
    assert not events and not calls and arena.nbytes == 0
    assert all(request.plan.selection is None for request in requests)
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    report = session.tune()
    assert [event for event in events if event[0] == "prepare"] == [
        ("prepare", True, ("projection", "attention", "experts", "recurrence")),
    ]
    assert calls == ["benchmark", "prime"] * 4 + ["bind"]
    assert report["cache_hits"] == 2 and report["benchmarked_candidates"] == 7
    assert report["restored_candidates"] == 3 and report["rejected_candidates"] == 0
    assert events.count(("result_close", True)) == 1
    assert arena.frozen


def test_preparation_failure_prevents_binding_or_capture(backend):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision")
    session.declare((declaration("projection"),))

    def fail(state):
        raise RuntimeError("cache preparation failed")

    session.declare((declaration("cache", prepare=fail),))
    session.register_finalizer(lambda: pytest.fail("binding before kernel preparation succeeded"))
    with pytest.raises(RuntimeError, match="cache preparation failed"):
        session.tune()
    assert not session._recorded_requests
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()


def test_terminal_progress_remains_live_for_every_tuning_job(backend, monkeypatch):
    from io import StringIO

    from b12x.preparation import PreparationProgress
    from rich.console import Console

    from ignition.preparation import tuning
    from ignition.progress import StartupProgress, phase

    session = make_session(device=0, checkpoint="terminal-progress")
    requests = (declaration("first"), declaration("second"))
    session.declare(requests)
    monkeypatch.setattr(session, "_jobs", lambda requests: tuple((item,) for item in requests))
    displays, statuses = [], []
    display_type = tuning.PreparationDisplay

    def display(**options):
        instance = display_type(**options)
        displays.append(instance)
        return instance

    monkeypatch.setattr(tuning, "PreparationDisplay", display)
    prepare = session._prepare_job

    def prepare_job(requests, *, progress, autotune):
        status = PreparationProgress(False, False, (), False,
                                     request_name=requests[0].name, phase="planning")
        progress(status)
        assert displays[-1]._live.is_started
        result = prepare(requests, progress=progress, autotune=autotune)
        progress(replace(status, done=True))
        assert not displays[-1]._live.is_started
        return result

    monkeypatch.setattr(session, "_prepare_job", prepare_job)
    startup = StartupProgress()
    startup.console = Console(file=StringIO(), force_terminal=True, width=120)
    with phase(startup, "Prepare and tune kernels") as stage:
        session.tune(progress=statuses.append, show_progress=True, console=stage.console)
        assert stage.live.is_started
    assert len(displays) == 2
    assert [item._title for item in displays] == [
        "b12x / kernel autotuning · job 1/2", "b12x / kernel autotuning · job 2/2"]
    assert [(status.request_name, status.done) for status in statuses] == [
        ("first", False), ("first", True), ("second", False), ("second", True)]


def test_standalone_preparation_still_precedes_binding(backend):
    session = PreparationSession(device=0, namespace={}, autotune=True, compile_workers=0)
    events = []
    request = declaration("operator", prepare=lambda state: events.append("prime"))
    # Standalone component tests can choose b12x's preparation policy directly.
    session.prepare = lambda requests: PreparationSession.prepare(session, requests, autotune=True)
    declare(session, (request,))
    register_finalizer(session, lambda: events.append("bind"))
    assert events == ["prime", "bind"]




def test_benchmark_fixture_cache_releases_completed_shapes_but_keeps_live_owners(backend):
    session = make_session(device=0, checkpoint="revision")

    class Fixture:
        nbytes = 16

    registered = Fixture()
    restored = []
    session.register_benchmark_resource("state", registered, restore=lambda: restored.append(True))
    session._tune_status = "tuning"
    first = benchmark_resource(session, "shape-a", Fixture)
    reference = weakref.ref(first)
    second = benchmark_resource(session, "shape-b", Fixture)
    assert reference() is first
    assert benchmark_resource(session, "shape-b", Fixture) is second
    assert benchmark_resource(session, "state", Fixture) is registered
    del first
    assert reference() is None
    session._restore_benchmark_resources()
    assert restored == [True]
    session.close()


def test_benchmark_resource_shared_across_requests_and_released_before_capture(backend):
    session = make_session(device=0, checkpoint="revision")
    observed = []
    references = []

    class Arena:
        nbytes = 512

    def create():
        value = Arena()
        references.append(weakref.ref(value))
        return value

    def trial(state):
        observed.append(id(benchmark_resource(session, "strided-state", create)))

    session.declare((declaration("a", benchmark=trial), declaration("b", benchmark=trial)))
    report = session.tune()
    assert len(references) == 1 and observed[0] == observed[1]
    assert references[0]() is None
    assert report["shared_benchmark_bytes"] == 512
    with pytest.raises(RuntimeError, match="only during"):
        benchmark_resource(session, "late", create)


def test_indexed_normalization_releases_private_priming_operands(backend, monkeypatch):
    from ignition.components import rms_norm

    session = make_session(device=0, checkpoint="indexed-normalization")
    plan = declaration("indexed").plan
    plan.request = lambda **kwargs: Request(plan=plan, **kwargs)
    monkeypatch.setattr(rms_norm.hyperconnection, "Caps", lambda **kwargs: None)
    monkeypatch.setattr(rms_norm.hyperconnection, "plan", lambda *args, **kwargs: plan)
    references = []
    original = rms_norm.HyperconnectionBenchmarkInputs

    def operands(*args):
        value = original(*args)
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(rms_norm, "HyperconnectionBenchmarkInputs", operands)

    def launch(source, weight, output, *, eps, indices):
        assert torch.isfinite(source).all()
        assert ((indices >= 0) & (indices < source.shape[0])).all()
        output.copy_(source[indices])

    plan.launch = launch
    source = torch.full((257, 128), torch.nan, dtype=torch.bfloat16)
    indices = torch.full((8,), 1 << 20, dtype=torch.int32)
    weight = torch.ones(128, dtype=torch.bfloat16)
    output = torch.full((8, 128), torch.nan, dtype=torch.bfloat16)
    rms_norm.PreparedIndexedRMSNorm(source, indices, weight, output,
        eps=1e-6, session=session, name="indexed")
    session.tune()
    assert references and all(reference() is None for reference in references)
    assert not session._published_owners["indexed"]
    assert torch.isnan(source).all() and torch.isnan(output).all()
    assert (indices == 1 << 20).all()
    session.close()






@pytest.mark.parametrize("source", ["fixed", "override", "cached", "tuned"])
def test_completed_selection_sources_are_accepted_and_reported(backend, source):
    session = make_session(device=0, checkpoint="revision")
    session.declare((declaration("op", source=source),))
    assert session.tune()["selection_counts"] == {source: 1}


@pytest.mark.parametrize("failure", ["default", "cancel", "finalizer", "result_close"])
def test_incomplete_tuning_never_allows_capture_or_freeze(backend, failure):
    events, controls = backend
    session = make_session(device=0, checkpoint="revision")
    session.declare((declaration("op", source="default" if failure == "default" else "tuned"),))
    controls.cancel = failure == "cancel"
    controls.close_error = failure == "result_close"
    if failure == "finalizer":
        def fail():
            raise RuntimeError("binding refresh failed")
        session.register_finalizer(fail)
    with pytest.raises(RuntimeError):
        session.tune()
    assert session._tune_status == "failed"
    assert not session._recorded_requests and not session._benchmark_resources
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()


def test_disabled_tuning_does_not_silently_keep_defaults(backend):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision")
    session.declare((declaration("op"),))
    session._stop.set()
    with pytest.raises(RuntimeError, match="enabled, uncancelled"):
        session.tune()
    assert not any(event[0] == "prepare" and event[1] is True for event in events)


def test_names_registration_and_one_shot_lifecycle_are_strict(backend):
    session = make_session(device=0, checkpoint="revision")
    item = declaration("op")
    with pytest.raises(ValueError, match="unique"):
        session.declare((item, item))
    session.declare((item,))
    with pytest.raises(ValueError, match="unique"):
        session.declare((item,))
    session.tune()
    for action in (lambda: session.tune(), lambda: session.declare((declaration("late"),)),
                   lambda: session.register_finalizer(lambda: None)):
        with pytest.raises(RuntimeError, match="closed after tuning"):
            action()
    session.close()
    assert session._tune_status == "closed"


def test_namespace_is_exact_and_reserved_policy_cannot_be_overridden(backend):
    session = make_session(device=0, checkpoint="revision", namespace={"headers": "digest"})
    assert session.namespace["checkpoint"] == "revision"
    assert session.namespace["headers"] == "digest"
    for key in ("checkpoint", "workload_schema", "numerical_policy"):
        with pytest.raises(ValueError, match="cannot override"):
            make_session(device=0, checkpoint="revision", namespace={key: "different"})


def test_recipe_policy_owns_cache_identity_and_report_limits(backend):
    limits = {"projection": "all eligible backends"}
    session = RecordedPreparationSession(device=0, checkpoint="revision",
        workload_schema="glm53-test-operators-v1", numerical_policy="default-tp2",
        search_limits=limits)
    limits["projection"] = "changed after construction"
    session.declare((declaration("op"),))
    report = session.tune()
    assert report["workload_schema"] == "glm53-test-operators-v1"
    assert report["namespace"]["numerical_policy"] == "default-tp2"
    assert report["search_limits"] == {"projection": "all eligible backends"}


def test_qwen_recipe_preserves_preparation_cache_identity(backend):
    from ignition.recipes.qwen38.tuning import preparation_session

    session = preparation_session(device=0, checkpoint="revision", namespace={"headers": "digest"})
    assert session.namespace.to_dict() == {
        "checkpoint": "revision", "headers": "digest",
        "workload_schema": "qwen38-active-operators-v1",
        "numerical_policy": "qsa16-gdn-sequential-k1-dense-mxfp8-auto-vision-nvfp4-a16-moe-a4-route-fp32",
    }


def test_worker_budget_accounts_for_all_preparing_ranks(backend, monkeypatch):
    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: 24 << 30)
    limits = []

    def configure(session, count):
        limits.append(count)
        session.compile_workers = 8 if count is None else count

    monkeypatch.setattr(PreparationSession, "configure_compile_workers", configure)
    coordinator = SimpleNamespace(ranks=(0, 1, 2, 3), barrier=lambda: None, local_ranks=4, spans_hosts=False,
                                  same_on_every_rank=lambda value: False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(is_integrated=False, name="NVIDIA GB10"))
    session = make_session(device=0, checkpoint="revision", coordinator=coordinator)
    session.declare((declaration("op"),))
    report = session.tune()
    assert limits == [None, 2]
    assert report["compile_workers"] == 2


@pytest.mark.parametrize("compile_headroom,workers", [(None, 2), (0, 3), (16 << 30, 1)])
def test_compile_headroom_bounds_tuning_compilers_and_is_reported(backend, monkeypatch, compile_headroom, workers):
    from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES

    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: 24 << 30)
    limits = []

    def configure(session, count):
        limits.append(count)
        session.compile_workers = 8 if count is None else count

    monkeypatch.setattr(PreparationSession, "configure_compile_workers", configure)
    coordinator = SimpleNamespace(ranks=(0, 1, 2, 3), barrier=lambda: None, local_ranks=4, spans_hosts=False,
                                  same_on_every_rank=lambda value: False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(is_integrated=False, name="NVIDIA GB10"))
    options = {} if compile_headroom is None else {"compile_headroom_bytes": compile_headroom}
    session = make_session(device=0, checkpoint="revision", coordinator=coordinator, **options)
    session.declare((declaration("op"),))
    report = session.tune()
    assert limits == [None, workers]
    assert report["compile_headroom_bytes"] == (COMPILE_HEADROOM_BYTES if compile_headroom is None
                                                else compile_headroom)


@pytest.mark.parametrize("value", [-1, 1.5, None])
def test_invalid_compile_headroom_is_rejected(backend, value):
    with pytest.raises(ValueError, match="compile_headroom_bytes"):
        make_session(device=0, checkpoint="revision", compile_headroom_bytes=value)


def test_explicit_cpu_budget_reaches_b12x_without_environment_switch(backend):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision", compile_workers=24)
    session.declare((declaration("op"),))
    report = session.tune()
    assert report["compile_workers"] == 24
    assert ("workers", 24) in events




@pytest.mark.parametrize("matching_catalog", [True, False])
@pytest.mark.parametrize("identical", [True, False])
@pytest.mark.parametrize("workers", [None, 4])
@pytest.mark.parametrize("spans_hosts", [False, True])
def test_tp_ranks_shard_tuning_across_identical_gpus(backend, monkeypatch, identical, matching_catalog, workers, spans_hosts):
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(is_integrated=False, name="NVIDIA GB10"))
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    shards, shared, coordinated = [], [], []
    monkeypatch.setattr(PreparationSession, "configure_tuning_shard",
                        lambda self, rank, ranks: shards.append((rank, ranks)), raising=False)
    monkeypatch.setattr(PreparationSession, "_selection_cache", lambda self: "rank cache", raising=False)

    def prepare(session, requests, *, progress, autotune):
        coordinated.append(tuple(request.name for request in requests))
        return PreparationSession.prepare(session, requests, autotune=autotune)

    coordinator = SimpleNamespace(ranks=(0, 1), barrier=lambda: None,
        local_ranks=1 if spans_hosts else 2, spans_hosts=spans_hosts,
        same_on_every_rank=lambda value: identical and matching_catalog and value ==
            ("NVIDIA GB10", _local_tuning_catalog((declaration("op"),))),
        share_selections=shared.append, agree_selections=lambda selections: None, prepare=prepare)
    session = make_session(device=0, checkpoint="revision", coordinator=coordinator, compile_workers=workers)
    session.declare((declaration("op"),))
    report = session.tune()
    assert session.autotune
    assert report["compile_workers"] == (8 if workers is None else workers // coordinator.local_ranks)
    # Sharded races exchange results through the coordinator even in jobs without collectives.
    expected = ([(1, (0, 1))], [], [("op",)]) if identical and matching_catalog else (
        [], ["rank cache"] if spans_hosts else [], [])
    assert (shards, shared, coordinated) == expected


def test_helpers_preserve_ordinary_session_behavior():
    session = object.__new__(PreparationSession)
    value = object()
    called = []
    register_finalizer(session, lambda: called.append(True))
    assert called == [True]
    assert benchmark_resource(session, "arena", lambda: value) is value


@pytest.mark.parametrize("source", ["cached", "tuned"])
def test_borrowed_state_restores_after_result_close_and_owner_release(backend, monkeypatch, source):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision")
    storage = torch.zeros(32, dtype=torch.uint8)
    resource = SimpleNamespace(nbytes=32, allocated_nbytes=0, borrowed_nbytes=32)
    calls = []

    def restore():
        assert not session._benchmark_resources and not session._recorded_requests
        assert not session._finalizers
        assert events[-2:] == [("result_close", True), ("drain",)]
        assert session._tune_status == "releasing"
        assert storage[0].item() == 9
        storage.zero_()
        calls.append("restore")

    session.register_benchmark_resource("state", resource, restore=restore)
    original_prepare = PreparationSession.prepare

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        close = result.close

        def final_trial_restore():
            close()
            storage.fill_(9)

        result.close = final_trial_restore
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)

    def trial(state):
        assert benchmark_resource(session, "state", lambda: pytest.fail("borrow ignored")) is resource
        storage.fill_(7)

    session.declare((declaration("op", source=source,
        benchmark=trial if source == "tuned" else None),))
    report = session.tune()
    assert calls == ["restore"] and not storage.count_nonzero()
    assert events.index(("result_close", True)) < len(events) - 1
    assert report["shared_benchmark_bytes"] == report["borrowed_benchmark_bytes"] == 32
    assert not session._benchmark_restorers
    with session.capture():
        assert events[-2][0] == "empty_cache"
        assert not storage.count_nonzero()


@pytest.mark.parametrize("failure", ["prepare", "finalizer", "result_close"])
def test_borrowed_state_restores_on_tuning_failures(backend, monkeypatch, failure):
    _, controls = backend
    session = make_session(device=0, checkpoint="revision")
    storage = torch.full((8,), 7, dtype=torch.uint8)
    session.register_benchmark_resource("state", SimpleNamespace(nbytes=8), restore=storage.zero_)
    session.declare((declaration("op"),))
    if failure == "prepare":
        def fail(*args, **kwargs):
            raise RuntimeError("candidate failed")
        monkeypatch.setattr(PreparationSession, "prepare", fail)
    elif failure == "finalizer":
        def fail():
            raise RuntimeError("finalizer failed")
        session.register_finalizer(fail)
    else:
        controls.close_error = True
    with pytest.raises(RuntimeError):
        session.tune()
    assert not storage.count_nonzero()
    assert not session._benchmark_restorers
    assert session._tune_status == "failed"
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()


@pytest.mark.parametrize("failure", ["reset", "pre_sync", "post_sync"])
def test_failed_restore_retains_owner_and_close_retries_before_freeze(backend, monkeypatch, failure):
    session = make_session(device=0, checkpoint="revision")
    attempts = []

    class Owner:
        def restore(self):
            attempts.append("reset")
            if failure == "reset" and len(attempts) == 1:
                raise RuntimeError("reset failed")

    owner = Owner()
    reference = weakref.ref(owner)
    session.register_benchmark_resource("state", SimpleNamespace(nbytes=16), restore=owner.restore)
    del owner
    original_sync = session._synchronize
    sync_attempts = []
    if failure in ("pre_sync", "post_sync"):
        def sync():
            if session._tune_status == "releasing":
                sync_attempts.append("sync")
                if len(sync_attempts) == (1 if failure == "pre_sync" else 2):
                    raise RuntimeError("restoration fence failed")
            original_sync()
        monkeypatch.setattr(session, "_synchronize", sync)
    session.declare((declaration("op"),))
    with pytest.raises(RuntimeError):
        session.tune()
    assert session._tune_status == "failed" and reference() is not None
    assert not session._benchmark_resources and session._benchmark_restorers
    assert len(attempts) == (0 if failure == "pre_sync" else 1)
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()
    monkeypatch.setattr(session, "_synchronize", original_sync)
    session.close()
    assert len(attempts) == (1 if failure == "pre_sync" else 2) and reference() is None
    assert session._tune_status == "closed" and not session._benchmark_restorers


def test_borrowed_registration_is_unique_collecting_only_and_restores_on_close(backend):
    session = make_session(device=0, checkpoint="revision")
    resource = SimpleNamespace(nbytes=4)
    restored = []
    with pytest.raises(TypeError, match="restore callback"):
        session.register_benchmark_resource("state", resource, restore=None)
    session.register_benchmark_resource("state", resource, restore=lambda: restored.append(True))
    with pytest.raises(ValueError, match="unique"):
        session.register_benchmark_resource("state", resource, restore=lambda: None)
    session.close()
    assert restored == [True]
    with pytest.raises(RuntimeError, match="closed after tuning"):
        session.register_benchmark_resource("late", resource, restore=lambda: None)


def test_gdn_registration_reuses_exact_strided_key_and_drops_arena_before_reset(backend):
    from ignition.preparation.fixtures.recurrent import register_benchmark_state

    session = make_session(device=0, checkpoint="revision")
    storage = torch.full((2, 32), 7, dtype=torch.uint8)
    template = torch.empty_strided((2, 4), (8, 1))
    references = []

    def restore():
        assert references[0]() is None
        storage.zero_()

    register_benchmark_state(session, template, storage=storage, restore=restore)
    key = ("delta_state", "cpu", 2, 32)
    references.append(weakref.ref(session._benchmark_resources[key]))

    def trial(state):
        arena = benchmark_resource(session, key, lambda: pytest.fail("registered arena not reused"))
        assert arena.view(template).data_ptr() == storage.data_ptr()
        assert arena.view(template).stride() == template.stride()
        arena.view(template).fill_(23)

    session.declare((declaration("op", benchmark=trial),))
    report = session.tune()
    assert not storage.count_nonzero()
    assert report["borrowed_benchmark_bytes"] == storage.nbytes
    assert report["allocated_shared_benchmark_bytes"] == 0


def test_close_retries_restoration_after_base_session_has_closed(backend, monkeypatch):
    session = make_session(device=0, checkpoint="revision")
    session.state = "OPEN"
    attempts = []

    def base_close(self):
        self.state = "CLOSED"

    def check_thread(self):
        if self.state == "CLOSED":
            raise RuntimeError("preparation session is closed")

    def restore():
        attempts.append("restore")
        if len(attempts) == 1:
            raise RuntimeError("reset failed after base close")

    monkeypatch.setattr(PreparationSession, "close", base_close)
    monkeypatch.setattr(PreparationSession, "_check_thread", check_thread)
    session.register_benchmark_resource("state", SimpleNamespace(nbytes=4), restore=restore)
    with pytest.raises(RuntimeError, match="reset failed"):
        session.close()
    assert session.state == "CLOSED" and session._tune_status == "failed"
    assert session._benchmark_restorers
    session.close()
    session.close()
    assert attempts == ["restore", "restore"]
    assert session._tune_status == "closed" and not session._benchmark_restorers


@pytest.mark.parametrize("failure", [None, "eviction", "restoration"])
def test_terminal_collection_reclaims_tuning_owners_before_restoration(monkeypatch, failure):
    import gc
    from b12x.preparation import DetectedDevice
    from b12x.preparation import session as preparation

    class Owner:
        def __init__(self):
            self.cycle = self

    session = make_session(device=DetectedDevice(None, None), checkpoint="terminal-collection")
    owner = Owner()
    reference = weakref.ref(owner)
    events = []
    attempts = 0

    def restore():
        nonlocal attempts
        attempts += 1
        events.append(("restore", reference() is None))
        if failure == "restoration" and attempts == 1:
            raise RuntimeError("restoration failed")

    session.register_benchmark_resource("borrowed", owner, restore=restore)
    del owner
    session._reclaim_programs()
    assert reference() is not None

    if failure == "eviction":
        def fail_eviction(keep):
            raise RuntimeError("eviction failed")
        monkeypatch.setattr(preparation, "evict_unretained", fail_eviction)

    def observe(phase, info):
        if phase == "stop" and info["generation"] == 2:
            events.append(("collect", reference() is None))

    automatic = gc.isenabled()
    gc.disable()
    gc.callbacks.append(observe)
    try:
        if failure is None:
            session.close()
        else:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                session.close()
        session.close()
        session.close()
    finally:
        gc.callbacks.remove(observe)
        if automatic:
            gc.enable()
    expected = [("collect", True), ("restore", True)]
    if failure == "restoration":
        expected.append(("restore", True))
    assert events == expected
    assert session._tune_status == "closed" and not session._benchmark_restorers








def test_all_component_envelope_reserves_one_arena_before_tuning_and_refreshes_aliases(backend):
    events, controls = backend
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(4)
    session.register_scratch(arena)
    owners = [SimpleNamespace(binding=None) for _ in range(3)]
    primes = []

    def declaration_for(index, component, size, selected_size):
        def prime(state):
            value = scratch_storage(arena, nbytes=4, device="cpu")
            primes.append((index, value.nbytes))
            owners[index].binding = value
            assert not arena.frozen

        request = declaration(f"layer{index}", component=component, size=size,
                              selected_size=selected_size, prepare=prime)
        session.declare((request,))
        register_finalizer(session, lambda: setattr(owners[index], "binding", arena.tensor))
        return request

    requests = [declaration_for(0, "sequence.gdn_prefill", 300, 120),
                declaration_for(1, "attention.qsa", 513, 257),
                declaration_for(2, "moe.decode", 400, 129)]
    controls.skip_prime.add("layer1")
    report = session.tune()
    assert primes == [(0, 768), (2, 768)]
    assert arena.nbytes == 512 and arena.frozen
    assert all(owner.binding is arena.tensor for owner in owners)
    assert report["scratch_arena"] == {"initial_bytes": 256,
        "candidate_scratch_bytes": 768, "candidate_reserved_bytes": 768,
        "parallel_candidate_scratch_bytes": 0, "parallel_selected_scratch_bytes": 0,
        "selected_scratch_bytes": 512, "reserved_bytes": 512, "pending_persistent_bytes": 0}
    envelope = ("envelope", tuple(request.name for request in requests))
    assert events.count(envelope) == 1
    assert events.index(envelope) < events.index(("prepare", True, envelope[1]))
    assert report["selection_counts"] == {"tuned": 3}
    assert report["cache_hits"] == 2 and report["benchmarked_candidates"] == 7
    with session.capture():
        assert all(owner.binding is arena.tensor for owner in owners)
    session.freeze()


def test_parallel_workspace_peak_is_reserved_then_trimmed_without_summing_serial_groups(backend):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    requests = [declaration("main", size=700, selected_size=300),
                declaration("auxiliary", size=300, selected_size=20),
                declaration("following", size=400, selected_size=100)]
    session.declare(requests)
    main, auxiliary, following = (request.plan for request in requests)
    arena.register_parallel(main, auxiliary)
    arena.register_parallel(auxiliary, following)
    report = session.tune()["scratch_arena"]
    assert report["candidate_reserved_bytes"] == 1280
    assert report["parallel_candidate_scratch_bytes"] == 1280
    assert report["parallel_selected_scratch_bytes"] == 768
    assert arena.nbytes == 768 and arena.frozen
    one, two = arena.views(((300,), torch.uint8), ((20,), torch.uint8))
    one.fill_(7)
    two.fill_(11)
    assert one.eq(7).all() and two.eq(11).all()
    with pytest.raises(RuntimeError, match="frozen"):
        arena.register_parallel(main, auxiliary)


def test_parallel_candidate_envelope_bounds_all_equivalent_bindings_once(backend):
    from collections import Counter
    from b12x.preparation import PersistentMemory

    events, _ = backend
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    requests = [declaration("a", size=300), declaration("b", size=700),
                declaration("c", size=100), declaration("serial", size=1800)]
    for request in requests[:2]:
        request.plan.query = "same workload"
    requests[0].plan.persistent = (PersistentMemory("shared", 400, 100),)
    requests[-1].plan.persistent = (PersistentMemory("shared", 700, 100),)
    session.declare(requests)
    a, b, c = (request.plan for request in requests[:3])
    arena.register_parallel(a, c)
    arena.register_parallel(b, c)
    report = session.tune()["scratch_arena"]
    assert report["parallel_candidate_scratch_bytes"] == 1024
    assert report["candidate_scratch_bytes"] == 2048
    assert report["pending_persistent_bytes"] == 600
    inspected = Counter(name for event in events if event[0] == "envelope" for name in event[1])
    assert inspected == Counter(request.name for request in requests)
    assert events.count(("envelope", ("a", "b"))) == 1
    assert ("envelope", ("a",)) not in events
    assert ("envelope", ("b",)) not in events


def test_parallel_scratch_subsets_preserve_dependencies_in_tuning(backend, monkeypatch):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    scales = declaration("experts/scales", size=64)
    experts = replace(declaration("experts", size=700), dependencies=(scales.name,))
    shared = declaration("shared", size=300)
    following = replace(declaration("following", size=1600), dependencies=(experts.name,))
    session.declare((scales, experts, shared, following))
    arena.register_parallel(experts.plan, shared.plan)
    envelope = session.candidate_memory_envelope
    prepare = session._prepare_job

    def inspect(requests):
        names = {request.name for request in requests}
        assert all(set(request.dependencies) <= names for request in requests)
        return envelope(requests)

    def tune(requests, **kwargs):
        assert requests == (scales, experts, shared, following)
        assert experts.dependencies == (scales.name,)
        assert following.dependencies == (experts.name,)
        return prepare(requests, **kwargs)

    monkeypatch.setattr(session, "candidate_memory_envelope", inspect)
    monkeypatch.setattr(session, "_prepare_job", tune)
    report = session.tune()["scratch_arena"]
    assert report["parallel_candidate_scratch_bytes"] == 1280
    assert report["candidate_scratch_bytes"] == 1792


def test_default_storage_released_after_alias_rebind_and_trial_cleanup_before_report(backend, monkeypatch):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    original = arena.reserve(4)
    reference = weakref.ref(original)
    owner = SimpleNamespace(binding=original)
    session.register_scratch(arena)
    session.declare((declaration("op", size=513, selected_size=120, prepare=lambda state, saved=original: None),))
    session.register_finalizer(lambda saved=original: setattr(owner, "binding", arena.tensor))
    del original

    def empty_cache():
        assert reference() is None
        assert owner.binding is arena.tensor and arena.nbytes == 256

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    session.tune()
    assert reference() is None


@pytest.mark.parametrize("source", ["cached", "tuned"])
@pytest.mark.parametrize("borrowed_state", [False, True])
def test_selected_arena_retained_after_trials_restoration_and_capture(backend, monkeypatch, source, borrowed_state):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    references, restored = {}, []
    original_prepare = PreparationSession.prepare

    class Trial:
        def __init__(self, scratch):
            self.scratch = scratch

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        if kwargs["autotune"]:
            trial = Trial(arena.tensor)
            references["trial"] = weakref.ref(trial)
            references["tensor"] = weakref.ref(trial.scratch)
            close = result.close

            def close_trial():
                assert references["tensor"]() is arena.tensor
                trial.scratch.fill_(17)
                close()

            result.close = close_trial
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    session.declare((declaration("expert", component="moe.decode", source=source),))

    def restore():
        assert references["trial"]() is None
        assert references["tensor"]() is arena.tensor
        assert arena.tensor.eq(17).all()
        assert events[-1] == ("drain",)
        restored.append(True)

    if borrowed_state:
        session.register_benchmark_resource("state", SimpleNamespace(nbytes=0), restore=restore)

    def empty_cache():
        assert references["trial"]() is None
        assert references["tensor"]() is arena.tensor

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    report = session.tune()
    assert restored == ([True] if borrowed_state else [])
    assert report["scratch_arena"] == {"initial_bytes": 0, "candidate_scratch_bytes": 256,
        "parallel_candidate_scratch_bytes": 0, "parallel_selected_scratch_bytes": 0,
        "candidate_reserved_bytes": 256, "selected_scratch_bytes": 256,
        "reserved_bytes": 256, "pending_persistent_bytes": 0}
    with session.capture():
        assert references["tensor"]() is arena.tensor
    session.freeze()
    session.close()
    assert references["tensor"]() is arena.tensor
    assert arena.tensor.eq(17).all()


def test_arena_registration_is_unique_collecting_only_and_optional(backend):
    session = make_session(device=0, checkpoint="revision")
    with pytest.raises(TypeError):
        session.register_scratch(torch.empty(4, dtype=torch.uint8))
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    with pytest.raises(ValueError):
        session.register_scratch(arena)
    session.declare((declaration("op"),))
    session.tune()
    with pytest.raises(RuntimeError, match="closed after tuning"):
        session.register_scratch(ScratchArena("cpu"))
    ordinary = make_session(device=0, checkpoint="revision")
    ordinary.declare((declaration("op"),))
    assert ordinary.tune()["scratch_arena"] == {}


def test_failed_close_retains_arena_owner_until_successful_drain(backend, monkeypatch):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(17)
    reference = weakref.ref(arena)
    session.register_scratch(arena)
    del arena
    attempts = []

    def close(self):
        attempts.append("close")
        if len(attempts) == 1:
            raise RuntimeError("drain failed")

    sync_attempts = []

    def synchronize():
        sync_attempts.append("sync")
        if len(sync_attempts) == 1:
            raise RuntimeError("fallback drain failed")

    monkeypatch.setattr(PreparationSession, "close", close)
    monkeypatch.setattr(session, "_synchronize", synchronize)
    with pytest.raises(RuntimeError, match="drain failed"):
        session.close()
    assert reference() is not None and session._tune_status == "failed"
    session.close()
    assert reference() is None and session._tune_status == "closed"
    assert attempts == ["close", "close"] and sync_attempts == ["sync", "sync"]


def test_arena_binding_finalizers_precede_other_finalizers_and_close_trial(backend):
    events, controls = backend
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(1)
    session.register_scratch(arena)
    owners = [SimpleNamespace(binding=arena.tensor) for _ in range(2)]
    session.declare((declaration("owner", size=512), declaration("alias", size=512)))
    controls.skip_prime.update(("owner", "alias"))
    for index, owner in enumerate(owners):
        def finalize(index=index, owner=owner):
            owner.binding = arena.tensor
            events.append(("arena_finalize", index))
        register_finalizer(session, finalize, workspace=arena)

    def inspect_bindings():
        assert all(owner.binding is arena.tensor for owner in owners)
        events.append(("inspect_bindings",))

    register_finalizer(session, inspect_bindings)
    session.tune()
    ordered = [("arena_finalize", 0), ("arena_finalize", 1),
               ("inspect_bindings",), ("result_close", True)]
    assert [events.index(event) for event in ordered] == sorted(events.index(event) for event in ordered)
    assert all(owner.binding.nbytes == 512 for owner in owners)


def test_arena_binding_failure_keeps_callback_owner_until_session_cleanup(backend):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    session.declare((declaration("op"),))

    class Owner:
        def finalize(self):
            raise RuntimeError("selected binding failed")

    owner = Owner()
    reference = weakref.ref(owner)
    register_finalizer(session, owner.finalize, workspace=arena)
    del owner
    with pytest.raises(RuntimeError, match="selected binding failed"):
        session.tune()
    assert reference() is not None and session._tune_status == "failed"
    with pytest.raises(RuntimeError, match="completed tuning"):
        session.freeze()
    session.close()
    assert reference() is None
    assert arena.nbytes == 256


def test_candidate_storage_survives_trial_close_then_dies_before_selected_memory_report(backend, monkeypatch):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    references, order = {}, []
    owner = SimpleNamespace(binding=None)
    original_prepare = PreparationSession.prepare

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        if kwargs["autotune"]:
            candidate = arena.tensor
            references["candidate"] = weakref.ref(candidate)
            close = result.close

            def close_trial():
                assert candidate is not arena.tensor
                assert owner.binding is arena.tensor
                assert candidate.nbytes == 1024 and arena.nbytes == 512
                assert arena.tensor.eq(23).all()
                candidate.fill_(17)
                order.append("close_trial")
                close()

            result.close = close_trial
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    session.declare((declaration("expert", component="moe.decode", size=1024, selected_size=257),))

    def finalize():
        assert references["candidate"]() is not None
        assert arena.nbytes == 512 and arena.frozen
        arena.tensor.fill_(23)
        owner.binding = arena.tensor
        order.append("finalize")

    register_finalizer(session, finalize, workspace=arena)

    def empty_cache():
        assert references["candidate"]() is None
        assert owner.binding is arena.tensor and arena.tensor.eq(23).all()
        order.append("empty_cache")

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    report = session.tune()
    assert order == ["finalize", "close_trial", "empty_cache"]
    assert report["scratch_arena"]["candidate_reserved_bytes"] == 1024
    assert report["scratch_arena"]["selected_scratch_bytes"] == report["scratch_arena"]["reserved_bytes"] == 512


def test_undeclared_candidate_growth_fails_before_freeze_finalization_or_capture(backend):
    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    finalized = []

    def benchmark(state):
        arena.reserve(513)

    session.declare((declaration("op", size=128, benchmark=benchmark),))
    register_finalizer(session, lambda: finalized.append(True), workspace=arena)
    with pytest.raises(RuntimeError):
        session.tune()
    assert not finalized and not arena.frozen
    assert session._tune_status == "failed"
    with pytest.raises(RuntimeError, match="completed tuning"):
        with session.capture():
            pass
    session.close()


@pytest.mark.parametrize("retain_arena_owner", [False, True])
def test_published_priming_owners_can_release_candidate_storage_only_when_arena_is_retained(
        backend, monkeypatch, retain_arena_owner):
    from ignition.preparation.scratch import prime_owners

    session = make_session(device=0, checkpoint="revision")
    arena = ScratchArena("cpu")
    arena.reserve(16)
    session.register_scratch(arena)
    source, output = torch.ones(4), torch.zeros(4)
    owners = [SimpleNamespace(binding=None), SimpleNamespace(binding=None)]
    references = {}

    def prime(state):
        scratch = arena.tensor
        binding = SimpleNamespace(scratch=scratch, source=source, output=output)
        for owner in owners:
            owner.binding = binding
        references[state.selection.source] = weakref.ref(scratch)
        retained = prime_owners(arena, scratch, binding) if retain_arena_owner else (scratch, binding)
        return PreparedCall(run=lambda: binding.output.copy_(binding.source), owners=retained)

    session.declare((declaration("operator", size=1024, selected_size=257, prepare=prime),))
    assert not session._published_owners
    for owner in owners:
        def finalize(owner=owner):
            owner.binding = SimpleNamespace(scratch=arena.tensor, source=source, output=output)
        register_finalizer(session, finalize, workspace=arena)
    del finalize
    session.tune()
    assert arena.nbytes == 512 and arena.frozen
    assert "default" not in references
    assert all(owner.binding.scratch is arena.tensor for owner in owners)
    assert output.eq(0).all()
    if retain_arena_owner:
        assert session._published_owners["operator"] == (arena,)
        assert references["tuned"]() is None
    else:
        assert references["tuned"]() is not None
        assert session._published_owners["operator"][0] is references["tuned"]()
    session.close()
    assert not session._published_owners
    assert references["tuned"]() is None
    assert all(owner.binding.scratch is arena.tensor for owner in owners)


@pytest.mark.parametrize("operator", ["gdn", "fused_moe", "hyperconnection", "varlen_attention", "blockscaled_gemm", "bf16_gemv", "mtp_feedback"])
def test_operator_benchmark_owners_release_after_trial_close(backend, operator):
    from ignition.preparation.fixtures import (
        blockscaled_gemm, bf16_gemv, fused_moe, recurrent, hyperconnection, varlen_attention, mtp_feedback,
    )
    from ignition.components.moe import RouterScratch

    events, controls = backend
    controls.retain_trials = True
    session = make_session(device=0, checkpoint="revision")
    source = torch.zeros(4, 32, dtype=torch.bfloat16)
    output = torch.zeros_like(source)
    references = []
    trial_alive_at_close = []
    controls.before_trial_close = lambda: trial_alive_at_close.append(all(ref() is not None for ref in references))
    specs = lambda: (ScratchBufferSpec(name="workspace", shape=(0,),
        dtype=torch.uint8, device=torch.device("cpu")),)
    state = SimpleNamespace(scratch=SimpleNamespace(scratch_specs=specs),
        scratch_plan=SimpleNamespace(scratch_specs=specs),
        layout=SimpleNamespace(scratch_specs=specs), required_workspace=0,
        bind=lambda **kwargs: SimpleNamespace(**kwargs), run=lambda *args, **kwargs: None,
        launch=lambda *args, **kwargs: None)

    def benchmark(_):
        if operator == "gdn":
            arguments = dict(x=source, out=output, conv_state=torch.zeros(3, 32, 3),
                conv_weight=torch.ones(32, 4), query_start_loc=torch.zeros(2, dtype=torch.int32),
                state_slot_ids=torch.full((1,), -1, dtype=torch.int64),
                state_is_fresh=torch.zeros(1, dtype=torch.bool),
                num_accepted_tokens=torch.ones(1, dtype=torch.int32),
                num_seqs=torch.zeros(1, dtype=torch.int32), num_tokens=torch.zeros(1, dtype=torch.int32))
            call = recurrent.benchmark_call(state, arguments=arguments, session=session, name="op",
                mode="prefill", scratch_specs=lambda _: specs(), workspace=None,
                run_options={}, binding_plan=object())
        elif operator == "fused_moe":
            call = fused_moe.benchmark_call(state, source=source,
                experts=SimpleNamespace(num_experts=512, has_csf_blocks=False), top_k=10, session=session, name="op",
                workspace=None, allocate_router=RouterScratch.allocate, router_plan=None)
        elif operator == "hyperconnection":
            call = hyperconnection.benchmark_call(state, "scaled_silu", (source, output), {})
        elif operator == "varlen_attention":
            arguments = dict(q=source.view(4, 2, 16), k=source.view(4, 2, 16),
                v=source.view(4, 2, 16), cu_seqlens_q=torch.zeros(3, dtype=torch.int32),
                max_seqlen_q=2, max_seqlen_k=2)
            call = varlen_attention.benchmark_call(state, arguments=arguments,
                inputs=varlen_attention.input_factory(arguments), workspace=None)
        elif operator == "mtp_feedback":
            call = mtp_feedback.benchmark_call(state,
                arguments=dict(token_embedding=source, multi_state=source, output=output),
                session=session, name="op", workspace=None, eps=1e-6)
        elif operator == "bf16_gemv":
            from ignition.preparation.fixtures.projection import input_factory
            call = bf16_gemv.benchmark_call(state, inputs=input_factory(source, output),
                weight=torch.ones(32, 32, dtype=torch.bfloat16), bias=None)
        else:
            from ignition.preparation.fixtures.projection import input_factory
            call = blockscaled_gemm.benchmark_call(state, inputs=input_factory(source, output),
                values=torch.ones(32, 32), scales=None, gain=None, bias=None,
                weight=object(), selected_workspace=lambda _: None)
        if call.reset is not None:
            call.reset()
        call.produce()
        call.invoke()
        references.append(weakref.ref(call.owners[0]))
        return call

    session.declare((declaration("op", benchmark=benchmark),))
    session.tune()
    assert trial_alive_at_close == [True]
    assert references and all(ref() is None for ref in references)
    assert not session._benchmark_resources
    assert not source.count_nonzero() and not output.count_nonzero()
    with session.capture():
        pass


GIB = 1 << 30


def test_projection_candidates_share_private_strided_operands(backend):
    from ignition.preparation.fixtures.bf16_gemv import benchmark_call
    from ignition.preparation.fixtures.projection import input_factory

    allocations = []

    @contextmanager
    def transient():
        allocations.append("enter")
        yield
        allocations.append("exit")

    session = make_session(device=0, checkpoint="projection-fixtures",
                           benchmark_storage=SimpleNamespace(transient=transient))
    session._tune_status = "tuning"
    source = torch.zeros(5, 64, dtype=torch.bfloat16)[:, :32]
    output = torch.zeros(5, 48, dtype=torch.float32)[:, :16]
    weight = torch.ones(16, 32, dtype=torch.bfloat16)
    state = SimpleNamespace(run=lambda x, w, out, bias: out.copy_(x.float() @ w.float().T))
    acquire = input_factory(source, output, storage=session.benchmark_storage)
    first, second = (benchmark_call(state, inputs=acquire, weight=weight, bias=None) for _ in range(2))
    assert allocations == ["enter", "exit"]
    assert first.owners[0] is second.owners[0]
    inputs = first.owners[0]
    assert inputs.source.stride() == source.stride() and inputs.output.stride() == output.stride()
    first.produce()
    first.invoke()
    expected = first.output.clone()
    inputs.source.fill_(float("nan"))
    inputs.output.fill_(float("nan"))
    second.produce()
    second.invoke()
    torch.testing.assert_close(second.output, expected, rtol=0, atol=0)
    assert not source.count_nonzero() and not output.count_nonzero()
    session.close()


@pytest.mark.parametrize(("free_gib", "race_bytes", "workers"), [
    # One compiler process runs even without room for one: compiling in the tuning
    # process would keep its programs through serving.
    (16, 7 * GIB // 4, 2), (12, 3 * GIB // 4, 1), (10, GIB // 4, 1)])
def test_integrated_tuning_automatically_budgets_shared_ram(backend, monkeypatch, free_gib, race_bytes, workers):
    events, _ = backend
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (GIB, 121 * GIB))
    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: free_gib * GIB)
    session = make_session(device=0, checkpoint="revision", headroom_bytes=6 * GIB)
    declare(session, (declaration("projection"),))
    report = session.tune()
    assert session.race_budget == report["race_budget_bytes"] == race_bytes
    assert session.compile_workers == report["compile_workers"] == workers
    assert report["headroom_bytes"] == 6 * GIB
    assert [event for event in events if event[0] == "workers"] == [("workers", None), ("workers", workers)]


def test_integrated_tuning_fails_before_using_headroom(backend, monkeypatch):
    events, _ = backend
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=True))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (GIB, 121 * GIB))
    monkeypatch.setattr("ignition.preparation.host_memory.available_bytes", lambda: 12 * GIB)
    session = make_session(device=0, checkpoint="revision", headroom_bytes=12 * GIB)
    declare(session, (declaration("projection"),))
    with pytest.raises(MemoryError, match="available shared memory"):
        session.tune()
    assert not [event for event in events if event[0] == "prepare"]


@pytest.mark.parametrize("race_budget", [None, GIB])
def test_discrete_gpu_tuning_bounds_races_without_reducing_compiler_pool(backend, monkeypatch, race_budget):
    events, _ = backend
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (12 * GIB, 32 * GIB))
    session = make_session(device=0, checkpoint="revision", headroom_bytes=6 * GIB, race_budget=race_budget)
    declare(session, (declaration("projection"),))
    report = session.tune()
    expected = 5 * GIB // 4 if race_budget is None else race_budget
    assert session.race_budget == report["race_budget_bytes"] == expected
    assert [event for event in events if event[0] == "workers"] == [("workers", None)]



def test_off_prepares_defaults_warms_and_finalizes_without_benchmarks(backend):
    events, _ = backend
    session = make_session(device=0, checkpoint="revision", autotune_effort="off")
    arena = ScratchArena("cpu")
    session.register_scratch(arena)
    calls = []
    session.declare((declaration("op", size=1024, selected_size=512,
                                prepare=lambda state: calls.append("warmup"),
                                benchmark=lambda state: pytest.fail("OFF benchmarked")),))
    session.register_finalizer(lambda: calls.append("bind"))
    report = session.tune()
    assert not session.autotune
    assert report["autotune_effort"] == "off"
    assert report["selection_counts"] == {"default": 1}
    assert report["cache_hits"] == report["benchmarked_candidates"] == 0
    assert calls == ["warmup", "bind"]
    assert ("prepare", False, ("op",)) in events
    assert report["scratch_arena"]["selected_scratch_bytes"] == 512
    assert arena.frozen and arena.nbytes == 512


def test_bootstrap_effort_scope_captures_real_contract_and_restores():
    from b12x.gemm._tuning import DenseGemmQuery
    from b12x.preparation import autotune_effort
    from b12x.preparation._efficiency import capture_autotune_effort

    from ignition.preparation.tuning import with_autotune_effort

    @with_autotune_effort
    def prepare(snapshot, config, *, fail=False):
        query = DenseGemmQuery(recipe="mxfp8", entry_point="gemm.mm", weight_storage="native",
                              output_dtype="bfloat16", batch=1, max_rows=8,
                              in_features=1280, out_features=32768)
        if fail:
            raise RuntimeError("bootstrap failed")
        return query

    config = SimpleNamespace(autotune_effort="audit")
    with autotune_effort("off"):
        query = prepare("snapshot", config)
        assert query.autotune_effort == "audit"
        assert capture_autotune_effort() == "off"
        with pytest.raises(RuntimeError, match="bootstrap failed"):
            prepare("snapshot", config, fail=True)
        assert capture_autotune_effort() == "off"
    assert query.autotune_effort == "audit"



def test_off_multihost_keeps_coordinator_warmup_without_tuning_cache(backend, monkeypatch):
    events, _ = backend
    agreed = []
    coordinator = SimpleNamespace(spans_hosts=True, ranks=(0,), local_ranks=1, group=None,
        barrier=lambda *args: None,
        same_on_every_rank=lambda *args: pytest.fail("OFF configured tuning shards"),
        share_selections=lambda *args: pytest.fail("OFF shared tuning cache"),
        agree_selections=lambda selections: agreed.append(selections),
        prepare=lambda session, requests, **kwargs: PreparationSession.prepare(session, requests, **kwargs))
    session = make_session(device=0, checkpoint="revision", autotune_effort="off", coordinator=coordinator)
    monkeypatch.setattr(session, "_selection_cache", lambda: pytest.fail("OFF accessed tuning cache"))
    request = declaration("collective")
    request.collective = object()
    calls = []
    request.prepare_call = lambda state: calls.append("warmup")
    session.declare((request,))
    assert session.tune()["selection_counts"] == {"default": 1}
    assert calls == ["warmup"]
    assert len(agreed) == 1 and agreed[0][0]["source"] == "default"
    assert ("prepare", False, ("collective",)) in events


@pytest.mark.parametrize("enabled,effort", [(False, "default"), (True, "off"), (True, "default")])
def test_clock_lock_scopes_tuning_without_changing_cache(backend, monkeypatch, enabled, effort):
    from ignition.preparation import clocks

    events = []

    @contextmanager
    def lock(uuid):
        events.append(("lock", uuid))
        try:
            yield
        finally:
            events.append(("unlock", uuid))

    monkeypatch.setattr(clocks, "locked_gpu_clocks", lock)
    session = make_session(device=0, checkpoint="revision", lock_gpu_clocks=enabled, autotune_effort=effort)
    session.device.uuid = "GPU-test"
    ordinary = make_session(device=0, checkpoint="revision", autotune_effort=effort)
    assert session.namespace == ordinary.namespace
    ordinary.close()
    active = enabled and effort != "off"
    request = declaration("projection", benchmark=lambda state: events.append("benchmark"))
    session.declare((request,))
    session.register_finalizer(lambda: events.append("bind"))
    session.tune()
    if active:
        assert events.index(("lock", "GPU-test")) < events.index("benchmark")
        assert events.index("benchmark") < events.index(("unlock", "GPU-test")) < events.index("bind")
    else:
        assert not any(isinstance(event, tuple) for event in events)
    session.close()


def test_clock_lock_released_on_tuning_failure(backend, monkeypatch):
    from ignition.preparation import clocks

    events = []

    @contextmanager
    def lock(uuid):
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    def fail(*args, **kwargs):
        raise RuntimeError("race failed")

    monkeypatch.setattr(clocks, "locked_gpu_clocks", lock)
    session = make_session(device=0, checkpoint="revision", lock_gpu_clocks=True)
    session.device.uuid = "GPU-test"
    session.declare((declaration("projection"),))
    monkeypatch.setattr(session, "_prepare_job", fail)
    with pytest.raises(RuntimeError, match="race failed"):
        session.tune()
    assert events == ["lock", "unlock"]
    session.close()


def test_local_tuning_catalog_preserves_semantics_without_bound_device_ordinals():
    first, peer = declaration("projection"), declaration("projection")
    first.plan._device, peer.plan._device = 0, 1
    assert _local_tuning_catalog((first,)) == _local_tuning_catalog((peer,))
    assert _local_tuning_catalog((first,)) != _local_tuning_catalog(())
    peer.plan.query = 256
    assert _local_tuning_catalog((first,)) != _local_tuning_catalog((peer,))
    collective = declaration("reduce")
    collective.collective = object()
    assert _local_tuning_catalog((first, collective)) == _local_tuning_catalog((first,))


@pytest.mark.parametrize("mode", ["static", "runtime"])
def test_collective_selection_is_reported_without_partitioning_operator_cache(backend, tmp_path, mode):
    from ignition.preparation.collective_ladder import load
    ladder = load("ignition/preparation/ladders/gb202_pcie_tp4.json") if mode == "runtime" else None
    session = make_session(device=0, checkpoint="collective-policy", collective_tuning=mode, collective_ladder=ladder)
    assert session.collective_tuning == mode
    from b12x.preparation._cache import SelectionCache

    static = make_session(device=0, checkpoint="collective-policy")
    assert session.namespace == static.namespace
    identity = dict(schema_version=23, tuning_cache_version=1, namespace=session.namespace.to_dict())
    original = dict(schema_version=23, tuning_cache_version=1,
                    namespace=dict(checkpoint="collective-policy", workload_schema="test-operators-v1",
                                   numerical_policy="default"))
    assert SelectionCache(tmp_path, identity).path == SelectionCache(tmp_path, original).path
    static.close()
    session.declare((declaration("op"),))
    assert session.tune()["collective_tuning"] == mode
    session.close()
