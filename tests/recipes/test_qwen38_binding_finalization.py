# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU binding-lifetime checks around batched preparation selection."""

from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.fixtures import varlen_attention as attention_fixtures
from ignition.components import recurrent, varlen_attention
from ignition.recipes.qwen38 import attention, vision
from ignition.preparation.scratch import ScratchArena


class Plan:
    def __init__(self, nbytes=8):
        self.nbytes, self.version = nbytes, 0

    def scratch_specs(self):
        return (
            SimpleNamespace(
                name="workspace",
                nbytes=self.nbytes,
                shape=(self.nbytes,),
                dtype=torch.uint8,
                device=torch.device("cpu"),
            ),
        )

    def request(self, **kwargs):
        return SimpleNamespace(plan=self, **kwargs)


class Materialized:
    def __init__(self, plan, *, nbytes=None):
        self.version = plan.version
        self.scratch_plan = Plan(plan.nbytes if nbytes is None else nbytes)
        self.layout = self.scratch_plan
        self.bindings = []

    def scratch_specs(self):
        return self.scratch_plan.scratch_specs()

    def bind(self, **kwargs):
        binding = SimpleNamespace(version=self.version, **kwargs)
        self.bindings.append(binding)
        return binding

    bind_for_preparation = bind

    def run(self, *args, **kwargs):
        return None

    run_for_preparation = run


class Session:
    def __init__(self):
        self.requests, self.finalizers, self.arenas = {}, [], []

    def declare(self, requests, *, coordinator=None):
        for request in requests:
            self.requests[request.name] = request

    def finalize(self):
        for arena in self.arenas:
            arena.finalize()
        for callback in self.finalizers:
            callback()


@pytest.fixture
def session(monkeypatch):
    from ignition.preparation import tuning

    monkeypatch.setattr(tuning, "RecordedPreparationSession", Session)
    value = Session()
    def finalizer(session, callback, *, workspace=None):
        if isinstance(workspace, ScratchArena):
            workspace.register_finalizer(callback)
            if workspace not in session.arenas:
                session.arenas.append(workspace)
        else:
            session.finalizers.append(callback)

    # PLE binds through the shared sequence-binding helper, which owns its finalizer.
    for module in (attention, vision, recurrent, varlen_attention):
        monkeypatch.setattr(module, "register_finalizer", finalizer)
    for module in (attention, vision):
        monkeypatch.setattr(module, "QuantizedProjection", lambda *args, **kwargs: None)

    return value


def test_qsa_binding_refreshes_selected_payload_and_resized_workspace(
    monkeypatch, session
):
    plan, writer = Plan(), Plan()
    captured = {}
    monkeypatch.setattr(
        attention.qsa, "Caps", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        attention.qsa, "invocation_from_tensors", lambda *args, **kwargs: kwargs
    )

    def declare(caps, **kwargs):
        captured["caps"] = caps
        return plan

    monkeypatch.setattr(attention.qsa, "plan", declare)
    monkeypatch.setattr(
        attention.qsa,
        "bind",
        lambda plan, **kwargs: SimpleNamespace(version=plan.version, **kwargs),
    )
    monkeypatch.setattr(attention.paged_kv_write, "Caps", lambda **kwargs: kwargs)
    monkeypatch.setattr(attention.paged_kv_write, "plan", lambda caps: writer)
    device = torch.device("cpu")
    source = torch.zeros((16, 2560), dtype=torch.bfloat16)
    scratch = attention.QSAScratch.allocate(16, device)
    state = attention.QSAState.allocate(pages=1, slots=1, device=device)
    metadata = attention.QSAMetadata.allocate(
        tokens=16, batch=1, context=128, device=device
    )
    module = SimpleNamespace(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        indexer=SimpleNamespace(
            q_layernorm=SimpleNamespace(weight=torch.zeros(128)),
            k_layernorm=SimpleNamespace(weight=torch.zeros(128)),
        ),
    )
    rotary = attention.RotaryTables(
        torch.ones(128, 32), torch.zeros(128, 32), torch.zeros(32, dtype=torch.int64)
    )
    weights = {
        name: object()
        for name in ("qkv_proj", "o_proj", "indexer.index_qk_proj")
    }
    arena = ScratchArena("cpu")
    operator = attention.PreparedAttention(
        module,
        weights,
        source,
        scratch,
        state,
        metadata,
        rotary,
        session=session,
        name="qsa",
        workspace=arena,
        max_cache_pages=2,
    )
    assert not hasattr(operator, "binding")
    assert arena.nbytes == 0
    assert not hasattr(captured["caps"], "attention_splits")
    assert (metadata.request_ids == -1).all() and (metadata.slot_mapping == -1).all()
    plan.version, plan.nbytes = 2, 32
    grown = arena.reserve(512)
    session.finalize()
    assert operator.binding.version == 2
    assert operator.binding.main_k_cache.shape[0] == 1
    assert operator.binding.scratch.data_ptr() == operator.workspace.data_ptr() == grown.data_ptr()
    assert operator.binding.scratch.shape == (32,)
    replacement = state.resize_pages_for_preparation(2)
    operator.rebind_pages_for_preparation(replacement)
    assert operator.binding.main_k_cache is replacement.main_k_cache
    assert operator.binding.scratch.data_ptr() == grown.data_ptr()
    assert operator.write_kwargs["k_cache"] is replacement.main_k_cache
    assert (metadata.request_ids == -1).all() and (metadata.slot_mapping == -1).all()


def test_vision_benchmark_owns_active_operands_and_aliases_follow_selected_binding(
    monkeypatch, session
):
    plan = Plan()
    monkeypatch.setattr(varlen_attention.varlen, "plan", lambda *args, **kwargs: plan)
    module = SimpleNamespace(
        qkv=SimpleNamespace(bias=None),
        proj=SimpleNamespace(bias=None),
        scaling=72**-0.5,
    )
    scratch = vision.VisionScratch.allocate(256, torch.device("cpu"))
    metadata = vision.VisionMetadata.allocate(256, 2, torch.device("cpu"))
    weights = {"qkv": object(), "proj": object()}
    arena = ScratchArena("cpu")
    owner = vision._Attention(
        module, scratch, metadata, weights, session=session, name="vision.owner",
        attention_workspace=arena,
    )
    aliases = [
        vision._Attention(
            module,
            scratch,
            metadata,
            weights,
            session=session,
            name=f"vision.alias{i}",
            shared=owner,
            attention_workspace=arena,
        )
        for i in range(2)
    ]
    assert not hasattr(owner.attention, "scratch")
    assert all(alias.attention is owner.attention for alias in aliases)
    assert arena.nbytes == 0
    request = session.requests["vision.owner"]
    candidates = [Materialized(plan, nbytes=64) for _ in range(2)]
    calls = [request.benchmark_call(candidate) for candidate in candidates]
    inputs = calls[0].owners[0]
    for candidate, call in zip(candidates, calls, strict=True):
        bound = candidate.bindings[-1]
        assert bound.scratch.numel() == 64
        assert bound.scratch.untyped_storage().data_ptr() == arena.tensor.data_ptr()
        assert bound.cu_seqlens_q.tolist() == [0, 256, 256]
        assert bound.cu_seqlens_q is not metadata.cu_seqlens
        for name, production in (
            ("q", scratch.query),
            ("k", scratch.key),
            ("v", scratch.value),
        ):
            value = getattr(bound, name)
            assert value.data_ptr() != production.data_ptr()
            assert value.stride() == production.stride()
            assert torch.isfinite(value).all() and value.count_nonzero() > 0
            value.fill_(float("nan"))
        assert call.produce is not None
        call.produce()
        for name in ("q", "k", "v"):
            value, seed = getattr(bound, name), getattr(inputs, "seed_" + name)
            assert value.data_ptr() != seed.data_ptr()
            assert torch.isfinite(seed).all() and seed.count_nonzero() > 0
            assert torch.equal(value, seed)
        call.run()
    assert all(call.owners[0] is inputs for call in calls)
    assert inputs.nbytes == 6 * 256 * 16 * 72 * 2 + 3 * 4
    for name in ("q", "k", "v"):
        assert getattr(candidates[0].bindings[-1], name) is getattr(
            candidates[1].bindings[-1], name
        )
        assert torch.equal(
            getattr(candidates[0].bindings[-1], name),
            getattr(candidates[1].bindings[-1], name),
        )
    assert (
        metadata.cu_seqlens.count_nonzero() == metadata.num_patches.count_nonzero() == 0
    )
    assert (
        scratch.query.count_nonzero()
        == scratch.key.count_nonzero()
        == scratch.value.count_nonzero()
        == 0
    )
    plan.version, plan.nbytes = 2, 32
    plan.handle = 123
    grown = arena.reserve(512)
    session.finalize()
    assert owner.attention.plan is plan
    assert owner.attention.scratch.data_ptr() == grown.data_ptr()
    assert owner.attention.scratch.shape == (32,)
    launches = []
    monkeypatch.setattr(varlen_attention, "_attention", lambda *args: launches.append(args))
    for operator in (owner, *aliases):
        assert operator.attention() is owner.attention.out
    assert len(launches) == 3
    assert all(args[5] is owner.attention.scratch and args[7] == 123 for args in launches)


def test_vision_private_producer_preserves_padded_operand_strides():
    arguments = {
        name: torch.zeros((8, 16, 80), dtype=torch.bfloat16)[..., :72]
        for name in ("q", "k", "v")
    }
    arguments["cu_seqlens_q"] = torch.zeros(2, dtype=torch.int32)
    arguments["max_seqlen_q"] = 8
    inputs = attention_fixtures.attention_benchmark_inputs(arguments)
    seeds = {name: getattr(inputs, "seed_" + name).clone() for name in ("q", "k", "v")}
    for name in seeds:
        getattr(inputs, name).zero_()
    inputs.produce()
    for name, seed in seeds.items():
        assert getattr(inputs, name).stride() == arguments[name].stride()
        assert torch.equal(getattr(inputs, name), seed)
        assert torch.equal(getattr(inputs, "seed_" + name), seed)
        assert arguments[name].count_nonzero() == 0
    assert inputs.nbytes == sum(
        value.untyped_storage().nbytes() for value in vars(inputs).values()
    )


@pytest.mark.parametrize("bind_plan", [False, True])
def test_ple_singleton_binding_refreshes_without_reinvoking_prepare_factory(session, bind_plan):
    arena = ScratchArena("cpu")
    plan = Plan()
    output = torch.zeros((2, 2560), dtype=torch.bfloat16)
    published = []
    returned = recurrent.prepare_sequence_binding(
        plan,
        {"out": output},
        session=session,
        name="ple.binding",
        public_bind=lambda plan, **kwargs: SimpleNamespace(version=plan.version, **kwargs),
        mode="prefill",
        bind_plan=bind_plan,
        workspace=arena,
        set_binding=published.append,
    )
    assert returned is None and not published
    assert arena.nbytes == 0
    request = session.requests["ple.binding"]
    # The request carries a representative benchmark for multi-candidate plans.
    assert request.benchmark_call is not None and request.prepare_call is not None
    candidate = Materialized(plan)
    plan.version, plan.nbytes = 3, 32
    grown = arena.reserve(512)
    session.finalize()
    arena.freeze()
    (refreshed,) = published
    assert refreshed.version == 3
    assert refreshed.out is output
    assert refreshed.scratch.shape == (32,)
    assert refreshed.scratch.data_ptr() == grown.data_ptr()
    # Refreshing the selected binding never re-invokes the prepare factory.
    assert candidate.bindings == []


def test_plain_preparation_session_keeps_workspace_without_finalizer_protocol():
    from b12x.preparation import PreparationSession

    from ignition.preparation.tuning import register_finalizer
    from ignition.preparation.scratch import scratch_storage

    ordinary = object.__new__(PreparationSession)
    workspace = torch.empty(8, dtype=torch.uint8)
    invoked = []
    register_finalizer(ordinary, lambda: invoked.append(True))
    assert invoked == [True]
    assert scratch_storage(workspace, nbytes=8, device="cpu") is workspace
