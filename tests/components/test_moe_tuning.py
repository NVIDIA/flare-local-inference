"""Serialized expert candidates share bounded scratch and exact routing inputs."""

from types import SimpleNamespace

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec, scratch_tensor

from ignition.components import moe
from ignition.preparation.fixtures import fused_moe as moe_fixtures
from ignition.preparation.scratch import ScratchArena


@pytest.mark.parametrize("a4_prefill", [False, True])
@pytest.mark.parametrize("tokens", [4, 17])
@pytest.mark.parametrize("storage", ["arena", "tensor"])
def test_candidate_cohort_reuses_exact_operands_and_one_arena(monkeypatch, tokens, storage, a4_prefill):
    source = torch.full((tokens, 32), 7, dtype=torch.bfloat16)
    ids = torch.zeros((tokens, 10), dtype=torch.int32)
    weights = torch.zeros((tokens, 10), dtype=torch.float32)
    output = torch.zeros_like(source)
    planes = {name: torch.zeros(16, dtype=torch.uint8) for name in
              ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
               "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale")}
    experts = SimpleNamespace(num_experts=512, has_csf_blocks=False,
                              _impl=SimpleNamespace(**planes, representation=None, nvfp4_csf=None))
    arena = ScratchArena("cpu")
    arena.reserve(256)
    arena.freeze()
    workspace = arena if storage == "arena" else arena.tensor
    resources, requests, runs = {}, [], []
    session = SimpleNamespace(prepare=lambda declarations: requests.extend(declarations))

    def specs(size):
        return (ScratchBufferSpec(name="workspace", shape=(size,),
            dtype=torch.uint8, device=torch.device("cpu")),)

    plan = SimpleNamespace(scratch_specs=lambda: specs(16),
        request=lambda **kwargs: SimpleNamespace(**kwargs))
    def plan_execution(**kwargs):
        assert kwargs["invocation"]["a4_prefill"] is a4_prefill
        assert kwargs["invocation"]["tuning_route_source"] == "b12x"
        assert kwargs["invocation"]["tuning_route_pattern"] == moe.TUNING_WORKLOAD_VERSION
        return plan

    monkeypatch.setattr(moe.fused_moe, "plan_execution", plan_execution)
    monkeypatch.setattr(moe.fused_moe, "bind", lambda plan, **kwargs: SimpleNamespace(**kwargs))
    finalizers = []
    monkeypatch.setattr(moe, "register_finalizer", lambda session, callback, **kwargs: finalizers.append(callback))
    router_plan = SimpleNamespace(query=SimpleNamespace(routed_scaling_factor=2.5))

    def route(*args, **kwargs):
        pytest.fail("expert tuning must use b12x synthetic routes")

    monkeypatch.setattr(moe.fused_moe, "route_topk", route)

    def resource(session, key, factory):
        if key not in resources:
            resources[key] = factory()
        return resources[key]

    monkeypatch.setattr(moe_fixtures, "benchmark_resource", resource)
    operator = moe.PreparedRoutedExperts(source, experts, ids, weights, output,
        session=session, name="prefill.layer0.experts", workspace=workspace,
        router_plan=router_plan, a4_prefill=a4_prefill)
    assert operator.scratch is None

    def candidate(index, size):
        def bind(**kwargs):
            scratch_tensor(kwargs["scratch"], specs(size), owner="TP MoE")
            assert kwargs["a4_prefill"] is a4_prefill
            return SimpleNamespace(**kwargs)

        def run(binding):
            runs.append(index)
            binding.output.copy_(binding.a)
            binding.scratch.fill_(index + 1)

        return SimpleNamespace(scratch=SimpleNamespace(scratch_specs=lambda: specs(size)),
            bind=bind, run=run)

    calls = [requests[0].benchmark_call(candidate(index, size))
             for index, size in enumerate((32, 128, 256))]
    fixture = resources[("moe_inputs", "prefill.layer0.experts")]
    reference = moe_fixtures.ExpertBenchmarkInputs(source, experts=512, top_k=10,
        allocate_router=moe.RouterScratch.allocate,
        router_plan=router_plan)
    assert len(resources) == 1
    assert all(call.owners[0] is fixture and call.owners[1].data_ptr() == arena.tensor.data_ptr() for call in calls)
    assert all(call.output is fixture.output and call.owners[2].scratch.data_ptr() == arena.tensor.data_ptr() for call in calls)
    assert all(call.owners[3] is experts and call.owners[2].input_scales_static for call in calls)
    assert all(call.owners[2].a4_prefill is a4_prefill for call in calls)
    assert all(call.reset is None and call.restore is None for call in calls)
    assert all(call.benchmark_producers is fixture.producers for call in calls)
    assert operator.scratch is None
    assert len(fixture.producers) == 6
    for index in (0, 1, 2, 0):
        call = calls[index]
        for produce, expected_produce in zip(call.benchmark_producers, reference.producers, strict=True):
            fixture.source.fill_(float("nan"))
            fixture.router.ids.fill_(-1)
            fixture.router.weights.fill_(float("nan"))
            fixture.output.fill_(float("nan"))
            produce()
            expected_produce()
            assert torch.equal(fixture.source, reference.source)
            assert torch.equal(fixture.router.ids, reference.router.ids)
            assert torch.equal(fixture.router.weights, reference.router.weights)
            call.run()
            assert torch.equal(fixture.output, reference.source)
    assert runs == [index for index in (0, 1, 2, 0) for _ in fixture.producers]
    assert torch.all(source == 7) and not ids.count_nonzero() and not weights.count_nonzero()
    assert not output.count_nonzero()
    requests[0].prepare_call(candidate(0, 16))
    for finalize in finalizers:
        finalize()
    assert operator.scratch.data_ptr() == arena.tensor.data_ptr()
    assert operator.binding.a4_prefill is a4_prefill
    with pytest.raises((ValueError, RuntimeError), match=f"requires {512 if storage == 'arena' else 257}"):
        requests[0].benchmark_call(candidate(3, 257))
    tensors = (fixture.fixture, fixture.source, fixture.output, fixture.routes,
        *(getattr(fixture.router, field) for field in fixture.router.__dataclass_fields__),
        fixture.route_weights)
    storage = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
               for value in tensors if value is not None}
    assert fixture.nbytes == sum(storage.values())


@pytest.mark.parametrize("a4_prefill", [False, True])
@pytest.mark.parametrize("tokens", [1, 128])
def test_expert_precision_reaches_the_canonical_execution_query(monkeypatch, tokens, a4_prefill):
    from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights, _PreparedWeightRepresentation
    from b12x.moe.fused_moe.weights import PreparedExperts

    api = moe.fused_moe
    weight_plan = api.plan_weights(
        source=api.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=api.ActivationSpec(mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=api.MoEGeometry(num_experts=2, hidden_size=256, intermediate_size=256))
    w13 = torch.empty((2, 512, 128), dtype=torch.uint8, device="meta")
    w2 = torch.empty((2, 256, 128), dtype=torch.uint8, device="meta")
    scales13 = torch.empty((2, 512, 16), dtype=torch.float8_e4m3fn, device="meta")
    scales2 = torch.empty((2, 256, 16), dtype=torch.float8_e4m3fn, device="meta")
    global_scales = torch.empty(2, device="meta")
    representation = _PreparedWeightRepresentation(quant_mode="w4a16",
        layout=weight_plan._impl.required_weight_layout("w4a16"),
        value=SimpleNamespace(w13=w13, w2=w2, w13_scale=scales13, w2_scale=scales2))
    experts = PreparedExperts(plan=weight_plan, _impl=B12XFP4ExpertWeights(
        plan=weight_plan._impl, w1_fp4=w13, w2_fp4=w2,
        w1_blockscale=scales13, w2_blockscale=scales2, w1_alphas=global_scales,
        w2_alphas=global_scales, a1_gscale=global_scales, a2_gscale=global_scales,
        representation=representation, a4_prefill_scales=True))
    declarations, finalizers = [], []
    session = SimpleNamespace(prepare=lambda requests: declarations.extend(requests))
    monkeypatch.setattr(moe, "register_finalizer", lambda _, callback, **kwargs: finalizers.append(callback))
    source = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    operation = moe.PreparedRoutedExperts(source, experts,
        torch.empty((tokens, 1), dtype=torch.int32, device="meta"),
        torch.empty((tokens, 1), dtype=torch.float32, device="meta"), torch.empty_like(source),
        session=session, name="experts", a4_prefill=a4_prefill,
        routing=api.RoutingSpec(deterministic_output=False))
    assert experts._impl.a4_prefill_scales
    assert operation.plan.query.a4_prefill is a4_prefill
    assert operation.plan.invocation["a4_prefill"] is a4_prefill
    assert operation.plan.query.quant_modes == ("w4a16",)
    assert declarations[0].plan is operation.plan
    assert len(finalizers) == 1
