"""Declare operators, prepare one kernel batch, then bind serving storage."""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Hashable
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from functools import wraps
from threading import get_ident

import torch
from b12x.preparation import (
    AutotuneEffort, MemoryRequirements, PreparationDisplay, PreparationSession,
    autotune_effort as b12x_autotune_effort,
)
from b12x.preparation.session import _declaration_key
from b12x.preparation.types import _CompositePlan

from .scratch import ScratchArena
from .host_memory import cpu_workers, release_unused, tuning_limits


# Requests per tuning job on an integrated GPU: collective declarations with a coordinator,
# otherwise all requests. Each job's compiler processes then stay within about a gigabyte.
_COLLECTIVES_PER_JOB = 96
_REQUESTS_PER_JOB = 192


def _local_tuning_catalog(requests):
    def identity(plan):
        if isinstance(plan, _CompositePlan):
            return (plan.component_id, plan.composite_semantic_version, plan.capacity_metadata,
                    tuple((count, identity(child)) for count, child in plan.variants.items()))
        # Bound device ordinals differ across ranks but do not change the race.
        return _declaration_key(plan)[:-1]

    return tuple((request.name, identity(request.plan), request.dependencies)
                 for request in requests if request.collective is None)


def _prepare_compiler_toolchain():
    """Resolve forward math libraries before native program identities are recorded."""
    from torch._inductor import config
    from torch._inductor.runtime.compile_tasks import _set_triton_libdevice_path

    from .compilation import _OPTIONS

    # Inductor selects the CUDA toolkit's libdevice for eager precision. Native
    # discovery and later materialization must see that same library identity.
    with config.patch(emulate_precision_casts=_OPTIONS["emulate_precision_casts"]):
        _set_triton_libdevice_path()


def with_autotune_effort(prepare):
    """Apply the preparation config's effort throughout bootstrap declarations."""
    @wraps(prepare)
    def scoped(snapshot, config, *args, **kwargs):
        with b12x_autotune_effort(config.autotune_effort):
            return prepare(snapshot, config, *args, **kwargs)

    return scoped


class RecordedPreparationSession(PreparationSession):
    """Retain declarations, tune them together, and refresh selected bindings.

    Construction records requests without compiling or launching their kernels.
    All kernels prepare in one batch using b12x's declaration coalescing and
    compiler pool. Finalizers bind the selected plans after the shared scratch
    arena has its serving capacity.

    Selected plans materialize without per-operation priming. Complete forwards
    warm their execution paths before capture; tuning candidates still prime.
    """

    def __init__(self, *, device, checkpoint: str, workload_schema: str,
                 numerical_policy: str, search_limits=None, namespace=None, race_budget=None,
                 coordinator=None, headroom_bytes=0, benchmark_storage=None, compile_workers=None,
                 tuning_replica=None, autotune_effort="default", collective_tuning="static", collective_ladder=None, native_dense_activations=False, lock_gpu_clocks=False):
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("tuning requires an exact checkpoint revision")
        if any(not isinstance(value, str) or not value for value in (workload_schema, numerical_policy)):
            raise ValueError("tuning requires a workload schema and numerical policy")
        if collective_tuning not in ("static", "runtime"):
            raise ValueError("collective_tuning must be static or runtime")
        from .collective_ladder import validate

        if collective_tuning == "runtime" and collective_ladder is None:
            raise ValueError("runtime collective tuning requires a collective ladder")
        if collective_ladder is not None and collective_tuning != "runtime":
            raise ValueError("collective ladder requires runtime collective tuning")
        self.collective_ladder = None if collective_ladder is None else validate(collective_ladder)
        self.collective_tuning = collective_tuning
        identity = dict(namespace or {})
        reserved = {"checkpoint", "workload_schema", "numerical_policy"}
        if reserved.intersection(identity):
            raise ValueError("tuning namespace cannot override its workload identity")
        identity.update(checkpoint=checkpoint, workload_schema=workload_schema,
                        numerical_policy=numerical_policy)
        self.autotune_effort = AutotuneEffort(autotune_effort).value
        self.native_dense_activations = native_dense_activations
        self.lock_gpu_clocks = lock_gpu_clocks
        _prepare_compiler_toolchain()
        with b12x_autotune_effort(self.autotune_effort):
            super().__init__(device=device, autotune=self.autotune_effort != "off", compile_workers=0,
                             namespace=identity, race_budget=race_budget, prime=False,
                             collective_barrier=None if coordinator is None else coordinator.barrier)
        if type(headroom_bytes) is not int or headroom_bytes < 0:
            raise ValueError("headroom_bytes must be a nonnegative byte count")
        self.headroom_bytes = headroom_bytes
        cpu_workers(compile_workers)
        self.requested_compile_workers = compile_workers
        self.tuning_replica = tuning_replica
        self._replica_coordinator = None
        self._coordinator = coordinator
        self._recorded_requests = {}
        self._context_plan_bindings = {}
        self._context_original_requests = None
        self._context_touched = set()
        self._context_scratch_owner = None
        self._finalizers = []
        self._scratch_arena = None
        self._benchmark_resources = {}
        self.benchmark_storage = benchmark_storage
        self._benchmark_restorers = {}
        self._preparation_thread = get_ident()
        self._tune_status = "collecting"
        self._terminal_collected = False
        self._sharded = False
        self._workload_schema = workload_schema
        self._search_limits = dict(search_limits or {})
        self.tuning_report = None

    def _require_collecting(self):
        self._check_thread()
        if self._tune_status != "collecting":
            raise RuntimeError("preparation declarations are closed after tuning begins")

    def declare(self, requests, *, coordinator=None):
        self._require_collecting()
        requests = tuple(requests)
        if coordinator is not None and coordinator is not self._coordinator:
            raise ValueError("collective declarations must use the session's bootstrap coordinator")
        names = [request.name for request in requests]
        if len(set(names)) != len(names) or any(name in self._recorded_requests for name in names):
            raise ValueError("preparation request names must be unique across the model")
        if any(request.retain_benchmark_call for request in requests):
            raise ValueError("serving preparation cannot retain benchmark calls after tuning")
        self._recorded_requests.update(zip(names, requests))

    @property
    def _integrated(self):
        import torch

        return torch.cuda.get_device_properties(self.device.ordinal).is_integrated

    def _jobs(self, requests):
        """Split tuning into jobs on an integrated GPU, whose compiler processes share system memory.

        b12x's compiler processes keep every program they compile, about 2.4 MB each, and
        last for one job; across a catalog the size of GLM's they would exhaust GB10's memory.
        Jobs keep dependency groups whole. With a coordinator, every job ends after the same
        collective declarations on every rank, so ranks enter matching jobs together.
        """
        if not self._integrated:
            return (requests,)
        names = {request.name: index for index, request in enumerate(requests)}
        parent = list(range(len(requests)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for index, request in enumerate(requests):
            for dependency in request.dependencies:
                parent[root(names[dependency])] = root(index)
        collective = self._coordinator is not None
        total = sum(request.collective is not None for request in requests) if collective else len(requests)
        per_job = _COLLECTIVES_PER_JOB if collective else _REQUESTS_PER_JOB
        last = max(0, (total - 1) // per_job)
        counted, chunk = 0, []
        for request in requests:
            chunk.append(min(last, counted // per_job))
            counted += not collective or request.collective is not None
        group_chunk = {}
        for index in range(len(requests)):
            group_chunk[root(index)] = max(group_chunk.get(root(index), 0), chunk[index])
        jobs = [[] for _ in range(last + 1)]
        for index, request in enumerate(requests):
            jobs[group_chunk[root(index)]].append(request)
        return tuple(tuple(job) for job in jobs if job)

    @contextmanager
    def _tuning_clocks(self):
        if not self.lock_gpu_clocks or not self.autotune:
            yield
            return
        from ignition.errors import PreparationError
        from .clocks import locked_gpu_clocks

        with ExitStack() as stack:
            failure = None
            try:
                stack.enter_context(locked_gpu_clocks(self.device.uuid))
            except Exception as error:
                failure = error
            ready = failure is None
            if self._replica_coordinator is not None:
                ready = all(self._replica_coordinator.gather(ready))
            elif self._coordinator is not None:
                ready = self._coordinator.same_on_every_rank(ready) and ready
            if failure is not None:
                raise failure
            if not ready:
                raise PreparationError("GPU clock locking failed on another preparation rank")
            yield

    def _prepare_job(self, requests, *, progress, autotune):
        autotune = autotune and self.autotune
        if self._replica_coordinator is not None:
            return self._replica_coordinator.prepare(self, requests, progress=progress, autotune=autotune)
        if self._coordinator is None or not self._sharded and not any(
                request.collective is not None for request in requests):
            return super().prepare(requests, progress=progress, autotune=autotune)
        return self._coordinator.prepare(self, requests, progress=progress, autotune=autotune)

    def register_finalizer(self, callback):
        self._require_collecting()
        if not callable(callback):
            raise TypeError("binding finalizer must be callable")
        self._finalizers.append(callback)

    def register_context_plan(self, name, plan, bind):
        """Track a prepared component whose plan may be chosen with its layer."""
        self._require_collecting()
        request = self._recorded_requests.get(name)
        if request is None or request.plan is not plan or name in self._context_plan_bindings:
            raise ValueError("context plan owner must match one declared request")
        if not callable(bind):
            raise TypeError("context plan binding must be callable")
        self._context_plan_bindings[name] = bind

    @property
    def context_plan_names(self):
        """Names whose component owners follow a contextual Plan selection."""
        return frozenset(self._context_plan_bindings)

    def bind_context_plans(self, choices):
        """Bind prepared alternatives to all declared owners before graph capture."""
        self._check_thread()
        if self._tune_status != "contextual" or self._context_original_requests is None:
            raise RuntimeError("context plans can be bound only during contextual preparation")
        if not choices.keys() <= self._context_plan_bindings.keys():
            raise ValueError("context choices must name registered component plans")
        replacements, requests = {}, {}
        for name, new_plan in choices.items():
            original = self._context_original_requests[name].plan
            if (not original.same_declaration_except_override(new_plan)
                    or new_plan is not original and new_plan.override is None):
                raise ValueError(f"context plan {name!r} changes the declared operator contract")
            if new_plan.selection is None or new_plan.prepared is None:
                raise ValueError(f"context plan {name!r} is not prepared")
            if new_plan is not original and new_plan.selection.source != "override":
                raise ValueError(f"context plan {name!r} must be an explicit prepared override")
            if new_plan.prepared.device.identity != self.device.identity:
                raise ValueError(f"context plan {name!r} changes the device")
            old_plan = self._recorded_requests[name].plan
            if old_plan in replacements and replacements[old_plan] is not new_plan:
                raise ValueError("one shared plan cannot bind two context alternatives")
            replacements[old_plan] = new_plan
            requests[name] = replace(self._recorded_requests[name], plan=new_plan,
                                     finalist_policy=None)
        previous = {name: self._recorded_requests[name].plan for name in choices}
        bound = []
        try:
            for name, request in requests.items():
                bound.append(name)
                self._context_plan_bindings[name](request.plan)
            if self._scratch_arena is not None:
                self._scratch_arena.replace_parallel_plans(replacements)
        except BaseException:
            for name in reversed(bound):
                self._context_plan_bindings[name](previous[name])
            raise
        self._recorded_requests.update(requests)
        self._context_touched.update(choices)
        return previous

    def preview_context_bindings(self):
        """Bind trial storage without freezing or consuming finalizers."""
        self._check_thread()
        if self._tune_status != "contextual":
            raise RuntimeError("trial bindings require contextual preparation")
        if self._scratch_arena is not None:
            self._scratch_arena.finalize(group=None if self._coordinator is None else self._coordinator.group,
                                         retain=True)
            if (self._context_scratch_owner is not None
                    and self._scratch_arena.tensor is not self._context_scratch_owner):
                raise RuntimeError("trial scratch moved after a contextual graph was captured")
            self._context_scratch_owner = self._scratch_arena.tensor
        for callback in self._finalizers:
            callback()
        self._synchronize()

    def prepare_context_plans(self, requests):
        """Prepare explicit legal alternatives in the open model session."""
        self._check_thread()
        if self._tune_status != "contextual":
            raise RuntimeError("context alternatives require contextual preparation")
        requests = tuple(requests)
        if any(request.collective is not None for request in requests):
            raise ValueError("context alternatives must use prepared local operators")
        return super().prepare(requests, autotune=False)

    def register_scratch(self, arena):
        self._require_collecting()
        if not isinstance(arena, ScratchArena):
            raise TypeError("planned operator scratch requires a ScratchArena")
        if self._scratch_arena is not None:
            raise ValueError("operator scratch arena is already registered")
        self._scratch_arena = arena

    def benchmark_resource(self, key: Hashable, factory):
        self._check_thread()
        if self._tune_status != "tuning":
            raise RuntimeError("shared benchmark resources exist only during candidate tuning")
        if key not in self._benchmark_resources:
            # Calls own their operands until their trials drain. Cache only the active
            # fixture; keeping every completed shape's inputs can exhaust GPU memory.
            self._benchmark_resources = {key: value for key, value in self._benchmark_resources.items()
                                         if key in self._benchmark_restorers}
            # Retired pools release fixture segments once their final trial owner drops them.
            with (nullcontext() if self.benchmark_storage is None else self.benchmark_storage.transient()):
                self._benchmark_resources[key] = factory()
        return self._benchmark_resources[key]

    def register_benchmark_resource(self, key: Hashable, resource, *, restore):
        """Borrow preparation-only storage until trial cleanup and restoration finish."""
        self._require_collecting()
        if key in self._benchmark_resources:
            raise ValueError("benchmark resource keys must be unique")
        if not callable(restore):
            raise TypeError("borrowed benchmark resources require a restore callback")
        self._benchmark_resources[key] = resource
        self._benchmark_restorers[key] = restore

    def _restore_benchmark_resources(self):
        if not self._benchmark_restorers:
            return
        self._synchronize()
        failures = []
        for restore in self._benchmark_restorers.values():
            try:
                restore()
            except BaseException as error:
                failures.append(error)
        try:
            self._synchronize()
        except BaseException as error:
            failures.append(error)
        if failures:
            # Keep callbacks and their state owners for a subsequent close attempt.
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("benchmark state restoration failed", failures)
        self._benchmark_restorers.clear()

    def _reserve_scratch(self, requests):
        if self._scratch_arena is None:
            return {}
        arena = self._scratch_arena
        groups, keys, envelopes = {}, {}, {}
        for request in requests:
            key = _declaration_key(request.plan)
            keys[id(request.plan)] = key
            # Memory bounds are independent of execution order. Each subset must
            # stand alone while the tuning batch retains its dependency graph.
            if request.dependencies:
                request = replace(request, dependencies=())
            groups.setdefault(key, []).append(request)

        def scratch_bytes(plan):
            key = keys[id(plan)]
            if key not in envelopes:
                # Bound every binding in the group while enumerating configs once.
                envelopes[key] = self.candidate_memory_envelope(groups[key])
            return envelopes[key].scratch_nbytes

        parallel = arena.parallel_nbytes(scratch_bytes)
        remaining = [request for key, group in groups.items() if key not in envelopes for request in group]
        requirements = list(envelopes.values())
        if remaining:
            requirements.append(self.candidate_memory_envelope(remaining))
        envelope = MemoryRequirements.sequential(requirements)
        before = arena.nbytes
        required = max(envelope.scratch_nbytes, parallel)
        arena.reserve(required)
        return {"initial_bytes": before, "candidate_scratch_bytes": required,
                "parallel_candidate_scratch_bytes": parallel,
                "candidate_reserved_bytes": arena.nbytes,
                "pending_persistent_bytes": envelope.pending_persistent_nbytes}

    def _set_finalist_policies(self, policies):
        """Give declaration-equivalent requests the same shortlist contract."""
        if not policies.keys() <= self._recorded_requests.keys():
            raise ValueError("finalist policy names must identify declared requests")
        by_declaration = {}
        for name, policy in policies.items():
            key = _declaration_key(self._recorded_requests[name].plan)
            if key in by_declaration and by_declaration[key] != policy:
                raise ValueError("one operator declaration cannot use conflicting finalist policies")
            by_declaration[key] = policy
        for name, request in tuple(self._recorded_requests.items()):
            policy = by_declaration.get(_declaration_key(request.plan))
            if policy is not None:
                declared = getattr(request, "finalist_policy", None)
                if declared is not None and declared != policy:
                    raise ValueError("declared finalist policy conflicts with its operator group")
                self._recorded_requests[name] = replace(request, finalist_policy=policy)

    def tune(self, *, progress=None, show_progress=False, console=None, contextual=None,
             finalist_policies=None):
        self._require_collecting()
        if not self._recorded_requests:
            raise RuntimeError("tuning requires recorded preparation requests")
        if finalist_policies is None and contextual is not None:
            finalist_policies = contextual.finalist_policies(dict(self._recorded_requests))
        if finalist_policies:
            self._set_finalist_policies(finalist_policies)
        self._tune_status = "tuning"
        results = []
        try:
            if self._stop.is_set():
                raise RuntimeError("full serving preparation requires enabled, uncancelled b12x tuning")
            requests = tuple(self._recorded_requests.values())
            if self.tuning_replica is not None:
                import torch.distributed as dist
                from .tuning_replicas import ReplicaPreparation

                if self._coordinator is not None and self._coordinator.spans_hosts:
                    raise ValueError("local tuning replicas cannot replace multi-host sharding")
                ranks = (0,) if self._coordinator is None else self._coordinator.ranks
                rank = 0 if self._coordinator is None else ranks.index(dist.get_rank())
                self._replica_coordinator = ReplicaPreparation(self.tuning_replica, rank, ranks=ranks)
                identities = self._replica_coordinator.gather(self.device.identity)
                corresponding = identities[rank::self.tuning_replica.tensor_parallel]
                if any(identity != corresponding[0] for identity in corresponding[1:]):
                    raise ValueError("corresponding tuning GPUs must have identical hardware geometry")
                if self.autotune:
                    self.configure_tuning_shard(self.tuning_replica.index, tuple(range(self.tuning_replica.replicas)))
            elif self.autotune and self._coordinator is not None:
                import torch.distributed as dist

                # Candidate shards must cover the same local races on every rank.
                name = torch.cuda.get_device_properties(self.device.ordinal).name
                catalog = _local_tuning_catalog(requests)
                same_name = self._coordinator.same_on_every_rank((name, catalog))
                # A Max-Q card and its Workstation Edition sibling report different names for the
                # same silicon. Ranks that differ only by that token must still share rank zero's
                # selections and race new candidates together; independent races chose different
                # MoE plans, and compiled forwards bind one rank's choices.
                self._sharded = same_name or self._coordinator.same_on_every_rank(
                    (name.replace(" Max-Q ", " "), catalog))
                if self._sharded:
                    if not same_name:
                        self._coordinator.share_selections(self._selection_cache())
                    self.configure_tuning_shard(dist.get_rank(), self._coordinator.ranks)
                elif self._coordinator.spans_hosts:
                    self._coordinator.share_selections(self._selection_cache())
            scratch_arena = self._reserve_scratch(requests)
            release_unused()
            # Compiler pools share the memory and CPUs of ranks on this host only.
            pools = 1 if self._coordinator is None else self._coordinator.local_ranks
            if self.tuning_replica is not None:
                pools = self.tuning_replica.processes
            preserve_default = self.requested_compile_workers is None and self.tuning_replica is None
            self.configure_compile_workers(None if preserve_default else
                cpu_workers(self.requested_compile_workers, pools=pools))
            bounded, self.race_budget = tuning_limits(self.device.ordinal, self.compile_workers,
                pools=pools, headroom_bytes=self.headroom_bytes, race_budget=self.race_budget)
            if self._integrated and self.compile_workers:
                # Compiling in this process would keep every program it compiles through serving;
                # one compiler process returns its memory at the end of each tuning job.
                bounded = max(1, bounded)
            if bounded < self.compile_workers:
                self.configure_compile_workers(bounded)
            jobs = self._jobs(requests)
            with self._tuning_clocks():
                for index, job in enumerate(jobs, 1):
                    # A completed b12x job stops its live display. Each job needs its own lifecycle.
                    with ExitStack() as display_stack:
                        update = progress
                        if show_progress:
                            title = "b12x / kernel autotuning"
                            if len(jobs) > 1:
                                title += f" · job {index}/{len(jobs)}"
                            display = display_stack.enter_context(PreparationDisplay(
                                global_rank=0, console=console, transient=True, title=title))

                            def update(status):
                                display.update(status)
                                if progress is not None:
                                    progress(status)

                        results.append(self._prepare_job(job, progress=update, autotune=self.autotune))
            if self._stop.is_set():
                raise RuntimeError("b12x tuning was cancelled before serving preparation completed")
            context_report = None
            if contextual is not None:
                self._tune_status = "contextual"
                self._context_original_requests = dict(self._recorded_requests)
                with self._tuning_clocks():
                    choices, context_report = contextual(self, dict(self._recorded_requests),
                        {name: finalists for result in results
                         for name, finalists in getattr(result, "finalists", {}).items()})
                    if not self._context_touched <= choices.keys():
                        raise ValueError("context result omits a plan bound during trial capture")
                    self.bind_context_plans(choices)
                    retired = {original.plan for name, original in self._context_original_requests.items()
                               if name in choices and original.plan is not choices[name]}
                    if retired:
                        self.release_many(retired)
                requests = tuple(self._recorded_requests.values())
                self._context_original_requests = None
            selections = []
            coverage = {name: value for result in results for name, value in result.coverage.items()}
            for request in requests:
                selection = request.plan.selection
                if selection is None or (self.autotune and selection.source == "default"):
                    raise RuntimeError(f"request {request.name!r} has no completed tuning selection")
                selections.append({"name": request.name, "component": request.plan.component_id,
                    "source": selection.source,
                    "config": request.plan.contract.config_payload(selection.config).to_dict(),
                    "coverage": dict(coverage.get(request.name, {}))})
            if self._coordinator is not None and (self._sharded or self._coordinator.spans_hosts):
                self._coordinator.agree_selections(selections)
            self._tune_status = "finalizing"
            if self._scratch_arena is not None:
                if self._scratch_arena.nbytes != scratch_arena["candidate_reserved_bytes"]:
                    raise RuntimeError("candidate binding exceeded the planned scratch envelope")
                self._synchronize()
                selected = MemoryRequirements.sequential(request.plan.memory_requirements() for request in requests)
                parallel = self._scratch_arena.parallel_nbytes(lambda plan:
                    plan.memory_requirements().scratch_nbytes)
                required = max(selected.scratch_nbytes, parallel)
                self._scratch_arena.trim(required)
                if self._coordinator is None:
                    self._scratch_arena.freeze()
                self._scratch_arena.finalize(group=None if self._coordinator is None else self._coordinator.group)
                self._scratch_arena.freeze()
                scratch_arena.update(selected_scratch_bytes=required,
                                     parallel_selected_scratch_bytes=parallel,
                                     reserved_bytes=self._scratch_arena.nbytes)
            for callback in self._finalizers:
                callback()
            self._synchronize()
            self.tuning_report = {
                "workload_schema": self._workload_schema, "namespace": self.namespace.to_dict(),
                "autotune_effort": self.autotune_effort,
                "collective_tuning": self.collective_tuning,
                "collective_ladder": self.collective_ladder,
                "native_dense_activations": self.native_dense_activations,
                "requests": len(requests), "compile_workers": self.compile_workers,
                "tuning_replicas": 1 if self.tuning_replica is None else self.tuning_replica.replicas,
                "race_budget_bytes": self.race_budget, "headroom_bytes": self.headroom_bytes,
                "cache_hits": sum(result.cache_hits for result in results),
                "benchmarked_candidates": sum(result.benchmarked_candidates for result in results),
                "restored_candidates": sum(result.restored_candidates for result in results),
                "rejected_candidates": sum(result.rejected_candidates for result in results),
                "elapsed_seconds": sum(result.elapsed_seconds for result in results),
                "parent_cute_compilations": sum(result.parent_cute_compilations for result in results),
                "parent_triton_compilations": sum(result.parent_triton_compilations for result in results),
                "selection_counts": dict(Counter(item["source"] for item in selections)),
                "selections": selections, "scratch_arena": scratch_arena,
                "shared_benchmark_bytes": sum(int(getattr(value, "nbytes", 0))
                                              for value in self._benchmark_resources.values()),
                "borrowed_benchmark_bytes": sum(int(getattr(value, "borrowed_nbytes", 0))
                                                for value in self._benchmark_resources.values()),
                "allocated_shared_benchmark_bytes": sum(int(getattr(value, "allocated_nbytes",
                    getattr(value, "nbytes", 0))) for value in self._benchmark_resources.values()),
                "search_limits": dict(self._search_limits),
            }
            registry = getattr(self, "_startup_collectives", None)
            if registry is not None:
                self.tuning_report.update(registry.memory_report())
            if context_report is not None:
                self.tuning_report["contextual_tuning"] = context_report
            self._tune_status = "releasing"
            return self.tuning_report
        except BaseException:
            self._tune_status = "failed"
            raise
        finally:
            try:
                failures = []
                for result in reversed(results):
                    try:
                        result.close()
                    except BaseException as error:
                        failures.append(error)
                if len(failures) == 1:
                    raise failures[0]
                if failures:
                    raise BaseExceptionGroup("preparation result cleanup failed", failures)
            except BaseException:
                self._tune_status = "failed"
                raise
            finally:
                result = None
                results.clear()
                requests = job = jobs = ()
                request = None
                callback = None
                self._clear_tuning_owners()
                try:
                    gc.collect()
                    self._restore_benchmark_resources()
                except BaseException:
                    self._tune_status = "failed"
                    raise
            if self._tune_status == "releasing":
                try:
                    with torch.cuda.device(self.device.ordinal):
                        torch.cuda.empty_cache()
                    self._tune_status = "ready"
                except BaseException:
                    self._tune_status = "failed"
                    raise
            self._replica_coordinator = None
            if self._tune_status == "ready" and self.tuning_replica is not None and self.tuning_replica.index:
                from .tuning_replicas import TuningComplete

                if self._coordinator is not None:
                    self._coordinator.barrier("tuning_complete", self._coordinator.ranks)
                self.close()
                raise TuningComplete()

    @contextmanager
    def capture(self):
        if self._tune_status not in ("ready", "contextual"):
            raise RuntimeError("graph capture requires completed tuning and binding finalization")
        with super().capture():
            yield

    def freeze(self):
        if self._tune_status != "ready":
            raise RuntimeError("preparation freeze requires completed tuning and binding finalization")
        return super().freeze()

    def _clear_tuning_owners(self):
        self._benchmark_resources.clear()
        self._finalizers.clear()
        self._recorded_requests.clear()
        self._context_plan_bindings.clear()
        self._context_original_requests = None
        self._context_touched.clear()
        self._context_scratch_owner = None

    def _reclaim_programs(self):
        terminal = self.state == "CLOSED"
        if terminal:
            # Include Ignition's tuning owners in b12x's terminal heap collection.
            self._clear_tuning_owners()
        super()._reclaim_programs()
        if terminal:
            self._terminal_collected = True

    def close(self):
        if get_ident() != self._preparation_thread:
            raise RuntimeError("preparation cleanup requires the owning thread")
        if self._tune_status == "closed":
            return
        try:
            try:
                super().close()
            finally:
                self._clear_tuning_owners()
                if not self._terminal_collected:
                    gc.collect()
                    self._terminal_collected = True
                self._restore_benchmark_resources()
                self._synchronize()
                if self._scratch_arena is not None:
                    self._scratch_arena._finalizers.clear()
                self._scratch_arena = None
        except BaseException:
            self._tune_status = "failed"
            raise
        else:
            self._tune_status = "closed"


def declare(session, requests, *, coordinator=None):
    """Collect model requests, or prepare immediately in a standalone b12x session."""
    if isinstance(session, RecordedPreparationSession):
        session.declare(requests, coordinator=coordinator)
    else:
        _prepare_compiler_toolchain()
        options = {} if coordinator is None else {"coordinator": coordinator}
        session.prepare(requests, **options)


def register_finalizer(session, callback, *, workspace=None):
    """Bind after model tuning, or immediately after standalone preparation."""
    if not isinstance(session, RecordedPreparationSession):
        callback()
    if isinstance(workspace, ScratchArena):
        workspace.register_finalizer(callback)
    elif isinstance(session, RecordedPreparationSession):
        session.register_finalizer(callback)


def register_context_plan(session, name, owner):
    """Let model preparation rebind a declared component before final capture."""
    if isinstance(session, RecordedPreparationSession):
        session.register_context_plan(name, owner.plan, lambda plan: setattr(owner, "plan", plan))


def benchmark_resource(session, key, factory):
    if isinstance(session, RecordedPreparationSession):
        return session.benchmark_resource(key, factory)
    return factory()
