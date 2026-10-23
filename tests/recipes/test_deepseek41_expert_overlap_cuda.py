"""Compiled shared/routed expert overlap preserves outputs and scratch lifetimes."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import PreparedMLP
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader
from ignition.recipes.deepseek41.decoder import Scratch


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.fixture(scope="module")
def expert_weights():
    snapshot = os.environ.get("FLARE_DS41_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_DS41_SNAPSHOT must identify the checkpoint")
    with DeepSeekWeightLoader(inspect_indexed_checkpoint(Path(snapshot)), "cuda:0",
            shard=TensorParallelShard(0, 4), prefixes=("layers.3.",)) as loader:
        weights = loader.load_layer(3).experts
        config = loader.config.text
    return weights, config


@pytest.mark.parametrize("rows", [16, 24, 32, 33, 48, 64, 96, 128])
@torch.no_grad()
def test_shared_expert_join_precedes_input_and_scratch_reuse(expert_weights, rows):
    weights, config = expert_weights
    device = torch.device("cuda:0")
    arena = ScratchArena(device, overlap=True)
    source = torch.randn((rows, 5120), device=device, dtype=torch.bfloat16)
    storage = Scratch.allocate(rows, rows, 4, device)
    live, logits = storage.ffn_hc.normalized, storage.router_logits
    route, shared_buffers, routed = storage.router, storage.shared, storage.routed
    output, shared_output = torch.empty_like(source), torch.empty_like(source)
    with RecordedPreparationSession(device=device, checkpoint="ds41-expert-overlap",
            workload_schema="tp4-shard", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        projection = UnquantizedProjection(live, weights.router, logits, session=session, name="router")
        router = PreparedRouter(logits, route, session=session, name="route",
            score_func=config["scoring_func"], renormalize=config["norm_topk_prob"],
            routed_scaling_factor=config["routed_scaling_factor"], correction_bias=weights.correction)
        experts = PreparedRoutedExperts(live, weights.experts, route.ids, route.weights, routed,
            session=session, name="experts", workspace=arena, router_plan=router.plan)
        shared = PreparedMLP(live, weights.shared, shared_buffers.gate_up, shared_buffers.intermediate,
            shared_buffers.output, session=session, name="shared", workspace=arena,
            swiglu_limit=config["swiglu_limit"], round_silu=False, activation_mode="auto")
        concurrent = ConcurrentSharedExpert(shared, parallel_plans=(projection.plan, router.plan, experts.plan),
            session=session, workspace=arena, defer_launch=True)
        session.tune(show_progress=False)

        def forward():
            live.copy_(source)
            concurrent.start(live)
            router(projection(live))
            experts()
            concurrent.finish(live, routed)
            output.copy_(routed + shared.output)
            shared_output.copy_(shared.output)
            storage.engram.gathered.fill_(42)
            arena.tensor.fill_(255)
            live.zero_()

        def serial_forward():
            live.copy_(source)
            router(projection(live))
            experts()
            shared(live)
            output.copy_(routed + shared.output)
            shared_output.copy_(shared.output)
            storage.engram.gathered.fill_(42)
            arena.tensor.fill_(255)
            live.zero_()

        serial_compiled = compile_forward(serial_forward)
        serial_compiled()
        torch.cuda.synchronize()
        serial_graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(serial_graph):
            serial_compiled()
        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        scratch_bytes = arena.tensor.nbytes
        for _ in range(12):
            source.normal_()
            live.copy_(source)
            router(projection(live))
            expected_routed = experts().clone()
            expected_shared = shared(live).clone()
            serial_graph.replay()
            torch.cuda.synchronize()
            serial_reference = output.clone()
            repetitions = []
            for _ in range(3):
                serial_graph.replay()
                torch.cuda.synchronize()
                repetitions.append(output.clone())
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert arena.tensor.nbytes == scratch_bytes
            torch.testing.assert_close(shared_output, expected_shared, atol=0, rtol=0)
            # The materialized MoE kernel sums BF16 expert contributions
            # atomically. Bound overlap error by repeated serial execution of
            # the same selected plans, while keeping the shared branch exact.
            reference = serial_reference.float()
            differences = [value.float() - reference
                           for value in [expected_routed + expected_shared, *repetitions]]
            floor = max(float(value.norm()) for value in differences)
            peak = max(float(value.abs().max()) for value in differences)
            error = output.float() - reference
            if floor == 0:
                torch.testing.assert_close(output, serial_reference, atol=0, rtol=0)
            else:
                assert float(error.norm()) <= 1.25 * floor, (rows, float(error.norm()), floor)
                assert float(error.abs().max()) <= 2 * peak, (rows, float(error.abs().max()), peak)
        timings = {"serial": [], "overlap": []}
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for repeat in range(5):
            order = (("serial", serial_graph), ("overlap", graph))
            for name, captured in order[::(-1 if repeat % 2 else 1)]:
                begin.record()
                for _ in range(100):
                    captured.replay()
                end.record()
                end.synchronize()
                timings[name].append(begin.elapsed_time(end) * 10)
        print(f"expert graph rows={rows} scratch_bytes={scratch_bytes} microseconds={timings}")
        graph.reset()
        serial_graph.reset()
