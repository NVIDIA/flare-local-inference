"""A grouped AOT artifact preserves distinct prepared BF16 kernel launches."""

import os
from dataclasses import replace

import pytest
import torch

from b12x.gemm import bf16_gemv
from b12x.preparation import PreparationSession, PreparedCall

from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class ProjectionForward:
    def __init__(self, source, weight, output, scratch, plan):
        self.source, self.weight, self.output = source, weight, output
        self.scratch, self.plan = scratch, plan

    def __call__(self):
        self.scratch.copy_(self.source)
        return bf16_gemv.mm(self.scratch, self.weight, out=self.output, plan=self.plan)


def _kernel_grids(graph):
    from cuda.bindings import driver

    raw = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(raw)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, nodes, count = driver.cuGraphGetNodes(raw, count)
    assert status == driver.CUresult.CUDA_SUCCESS
    kernels = []
    for node in nodes[:count]:
        status, kind = driver.cuGraphNodeGetType(node)
        assert status == driver.CUresult.CUDA_SUCCESS
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        status, params = driver.cuGraphKernelNodeGetParams(node)
        assert status == driver.CUresult.CUDA_SUCCESS
        status, name = driver.cuFuncGetName(params.func)
        assert status == driver.CUresult.CUDA_SUCCESS
        kernels.append((name.decode() if isinstance(name, bytes) else str(name),
                        (params.gridDimX, params.gridDimY, params.gridDimZ)))
    return kernels


def _validate_projection(forward):
    source = forward.source.double()
    weight = forward.weight.double()
    exact = source @ weight.T
    magnitude = source.abs() @ weight.abs().T
    u = source.shape[1] * 2.0 ** -24
    actual = forward.output
    below = torch.nextafter(actual, torch.full_like(actual, -float("inf"))).double()
    above = torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
    bound = (u / (1.0 - u)) * magnitude
    bound += .5 * torch.maximum(actual.double() - below, above - actual.double())
    assert torch.equal(forward.scratch, forward.source)
    assert bool(torch.isfinite(actual).all())
    assert bool(((actual.double() - exact).abs() <= bound).all())


def test_grouped_bf16_plans_capture_distinct_grids_with_one_artifact(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not bf16_gemv.is_supported(device):
        pytest.skip("prepared BF16 GEMV requires SM120/SM121")
    rows, k, n = 8, 5120, 512
    torch.manual_seed(5120512)
    weight = (torch.randn((n, k), device=device) * .125).bfloat16()
    seeds = tuple((torch.randn((rows, k), device=device) * .125).bfloat16()
                  for _ in range(2))
    inputs = tuple(seed.clone() for seed in seeds)
    scratch = tuple(torch.empty_like(value) for value in inputs)
    outputs = tuple(torch.empty((rows, n), device=device, dtype=torch.bfloat16)
                    for _ in inputs)
    query = bf16_gemv.query_from_call(scratch[0], weight, out=outputs[0])
    base = bf16_gemv.GemvConfig(backend="simt", rows_per_tile=8, block_threads=128)
    plans = (bf16_gemv.plan(query, override=base),
             bf16_gemv.plan(query, override=replace(base, rows_per_tile=2)))
    forwards = {name: ProjectionForward(value, weight, output, work, plan)
                for name, value, output, work, plan in zip(
                    ("eight-rows", "two-rows"), inputs, outputs, scratch, plans, strict=True)}
    signature = GraphSignature("decode", rows, 1)
    graphs = {}
    with PreparationSession(device=device, autotune=False, compile_workers=2,
                            prime=False) as session:
        requests = []
        for (name, forward), plan in zip(forwards.items(), plans, strict=True):
            def prepare(state, forward=forward):
                return PreparedCall(run=lambda: state.run(forward.scratch, weight,
                    out=forward.output), owners=(forward, weight))
            requests.append(plan.request(name=name, prepare_call=prepare))
        result = session.prepare(tuple(requests), autotune=False)
        result.close()
        try:
            graphs, compiled = capture_forwards(forwards,
                {name: signature for name in forwards}, session=session,
                device=device, stream=torch.cuda.Stream(device=device),
                pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path,
                compile_group="bf16-gemv-plan-family", retain_graphs=True,
                before_run=lambda name: forwards[name].source.copy_(
                    seeds[0 if name == "eight-rows" else 1]))
            assert compiled["eight-rows"].artifact.backend is compiled["two-rows"].artifact.backend
            kernels = {name: _kernel_grids(graph) for name, graph in graphs.items()}
            eight = [grid for kernel, grid in kernels["eight-rows"] if "gemv" in kernel.lower()]
            two = [grid for kernel, grid in kernels["two-rows"] if "gemv" in kernel.lower()]
            assert len(eight) == len(two) == 1
            assert eight[0][1] == 1 and two[0][1] == 4
            session.freeze()
            for factor in (.5, 1.5):
                for name, forward in forwards.items():
                    forward.source.copy_(seeds[0 if name == "eight-rows" else 1] * factor)
                    forward.scratch.fill_(float("nan"))
                    forward.output.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    before = torch.cuda.memory_stats(device)
                    graphs[name].replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    _validate_projection(forward)
        finally:
            for graph in graphs.values():
                graph.reset()
