"""Shared caller-owned replay schedules for equivalent paged attention geometry."""

import torch
from b12x.attention.paged.schedule import (
    PrefillSchedule,
    schedule_counts,
    schedule_nbytes,
)
from b12x.preparation import plan_from_handle, require_prepared

from ignition.preparation.tuning import register_finalizer


def _schedule(plan):
    state = require_prepared(plan, "attention.gqa")
    caps, selected = state.scratch_plan.caps, state.scratch_plan.plan
    if caps.mode not in ("verify", "extend") or not selected.split_kv:
        return None
    return PrefillSchedule.from_scratch_plan(state.scratch_plan, state.metadata)


@torch.library.custom_op("flare::paged_schedules", mutates_args=("storage",))
def _update_schedules(
    lengths: torch.Tensor, cu: torch.Tensor, storage: torch.Tensor, handles: list[int]
) -> None:
    seen = set()
    for index, handle in enumerate(handles):
        schedule = _schedule(plan_from_handle(handle))
        if schedule is not None and schedule.key not in seen:
            seen.add(schedule.key)
            schedule.update(storage[index], lengths, cu)


@_update_schedules.register_fake
def _fake_update_schedules(lengths, cu, storage, handles):
    pass


class PreparedPagedSchedules:
    """Keep scheduling storage outside numerical workspaces for a complete layer loop."""

    def __init__(self, lengths, cu, layers, *, buffers):
        if any(t.dtype != torch.int32 or not t.is_contiguous() for t in (lengths, cu)):
            raise ValueError(
                "shared attention schedules require contiguous int32 live metadata"
            )
        self.lengths, self.cu, self.layers, self.buffers = lengths, cu, layers, buffers
        self.storage, self.counts = None, None
        self.components, self.groups = [], ()

    def finalize(self, session):
        register_finalizer(session, self.rebind)

    def register(self, component, caps):
        if (
            component.arguments["cache_seqlens"] is not self.lengths
            or component.arguments["cu_seqlens_q"] is not self.cu
        ):
            raise ValueError(
                "shared attention schedules require identical live metadata sources"
            )
        if len(self.components) >= self.layers:
            raise ValueError(
                "shared attention schedules exceed the declared layer capacity"
            )
        counts = schedule_counts(caps)
        if self.storage is None:
            self.counts = counts
            self.storage = self.buffers.zeros(
                (self.layers, schedule_nbytes(caps)), torch.uint8
            )
        elif counts != self.counts:
            raise ValueError(
                "shared attention schedule capacities must match across layers"
            )
        self.components.append(component)
        component.schedule_storage = self.storage

    def rebind(self):
        groups, owners = {}, []
        for index, component in enumerate(self.components):
            schedule = _schedule(component.plan)
            if schedule is None:
                component.schedule_slot = -1
            else:
                schedule.views(self.storage[index])
                component.schedule_slot = groups.setdefault(schedule.key, index)
                owners.append(schedule)
        self.groups = tuple(owners)

    def __call__(self):
        if self.storage is not None:
            _update_schedules(
                self.lengths,
                self.cu,
                self.storage,
                [component.plan.handle for component in self.components],
            )
