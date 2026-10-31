"""GLM MoE layer on SM103: the serial shared expert folds into the routed reduction bit for bit."""

import os
from pathlib import Path

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.moe import MoeScratch, PreparedGlmMoe
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.fixture(scope="module")
def layer():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 routed experts require a GB300/B300")
    snapshot = os.environ.get("FLARE_GLM53_SNAPSHOT")
    if snapshot is None:
        pytest.skip("FLARE_GLM53_SNAPSHOT must identify the checkpoint")
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    tree = create_model_tree(manifest.snapshot)
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                         prefixes=(f"{TEXT_PREFIX}.layers.3.mlp.",)) as loader:
        scales = loader.allocate_expert_scale_scratch()
        weights = loader.load_mlp(3, scale_scratch=scales)
    return tree.model.language_model.layers[3].mlp, weights, scales


def _kernel_names(graph):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        graph.replay()
        torch.cuda.synchronize()
    return [event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]


# 256 rows with overlap runs the concurrent shared expert and keeps the add; the other cases are serial.
@pytest.mark.parametrize("rows,overlap", [(256, False), (256, True), (4093, True), (4096, True)])
@torch.no_grad()
def test_serial_shared_expert_folds_into_route_reduction(layer, rows, overlap):
    module, weights, _scale_owners = layer
    arena = ScratchArena("cuda:0", overlap=overlap)
    source = torch.randn(rows, 4096, dtype=torch.bfloat16, device="cuda")
    input_copy = source.clone()
    output_copy = torch.empty_like(source)
    scratch = MoeScratch.allocate(rows, 1, "cuda:0")
    with PreparationSession(device="cuda:0", autotune=True, compile_workers=0) as session:
        moe = PreparedGlmMoe(module, input_copy, weights, scratch, session=session,
                             name=f"serial_reduction:{rows}", workspace=arena)
        arena.freeze()
        arena.finalize()
        session.freeze()
        folded = moe.concurrent_shared is None
        assert folded is (not (overlap and rows <= 256))
        assert moe.experts.reduces_into_shared

        def forward():
            input_copy.copy_(source)
            moe(input_copy)
            output_copy.copy_(moe.output)

        call = compile_forward(forward)
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        generator = torch.Generator().manual_seed(rows)
        for _ in range(3):
            source.copy_(torch.randn((rows, 4096), generator=generator).to(torch.bfloat16))
            input_copy.copy_(source)
            expected_shared = moe.shared(input_copy).clone()
            moe.router_projection(input_copy)
            moe.routing(scratch.router_logits)
            routed = moe.experts().clone()
            expected = torch.add(routed, expected_shared)
            arena.tensor.fill_(0x7F)
            output_copy.fill_(float("nan"))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            assert torch.equal(output_copy, expected)
        names = _kernel_names(graph)
        reductions = sum("_reduce_routes" in name for name in names)
        adds = sum("CUDAFunctor_add" in name for name in names)
        # Folded: one reduction kernel carries the shared add and no separate add runs.
        assert (reductions, adds) == ((1, 0) if folded else (1, 1)), names
