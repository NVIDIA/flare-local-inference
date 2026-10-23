"""Output-aware PCIe reductions through compiled Flare forwards and native replay."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _check_factory_cleanup(device, rank):
    from unittest.mock import patch

    from b12x.comm.pcie import pcie_allreduce as dispatch
    from ignition.preparation.collectives import create_all_reduce_runtime

    original_factory = dispatch.PCIeOneshotAllReducePool.from_exchange_group
    original_support = dispatch._bf16_twoshot_supported
    original_properties = torch.cuda.get_device_properties
    created = []

    def factory(**kwargs):
        runtime = original_factory(**kwargs)
        channels = tuple(runtime._all_channels) + tuple(runtime._channels.values())
        assert channels
        created.append((runtime, channels))
        return runtime

    def support(**kwargs):
        failed = False
        def properties(queried_device):
            nonlocal failed
            if rank == 1 and not failed:
                failed = True
                raise RuntimeError("injected rank-1 optional capability failure")
            return original_properties(queried_device)
        with patch.object(torch.cuda, "get_device_properties", side_effect=properties):
            return original_support(**kwargs)

    with patch.object(dispatch.PCIeOneshotAllReducePool, "from_exchange_group", side_effect=factory), \
            patch.object(dispatch, "_bf16_twoshot_supported", side_effect=support):
        with pytest.raises(RuntimeError, match="injected rank-1 optional capability failure"):
            create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=8, hidden_size=4096)
    assert len(created) == 1
    runtime, channels = created[0]
    assert runtime._closed and not runtime._all_channels and not runtime._channels
    for channel in channels:
        assert channel._coordinated_close_complete
        assert channel._ipc_imports_closed and channel._ipc_exports_freed
        assert not channel._owned_buffers
    dist.barrier()


def _rank(rank, rendezvous):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=3))
    coordinator = CollectivePreparation(dist.group.WORLD)
    _check_factory_cleanup(device, rank)
    runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=257, hidden_size=4096)
    worker = None
    graphs = {}
    try:
        with RecordedPreparationSession(device=device, checkpoint="pcie-dispatch",
                workload_schema="aliased-and-disjoint-bf16", numerical_policy="exact-integer-sums",
                coordinator=coordinator, compile_workers=4) as session:
            cases = []
            for rows, width, alias in ((1, 4096, False), (8, 4096, False),
                    (256, 4096, False), (257, 4096, False), (4096, 256, False),
                    (8, 4096, True), (256, 4096, True)):
                name = f"reduce-{rows}-{width}-{'alias' if alias else 'disjoint'}"
                source = torch.zeros((rows, width), dtype=torch.bfloat16, device=device)
                output = source if alias else torch.empty_like(source)
                result = torch.empty_like(source)
                operation = PreparedAllReduce(source, output, runtime=runtime, session=session,
                                              coordinator=coordinator, name=name)
                expected_surface = ("PCIeTwoShotBF16.all_reduce" if not alias
                    and 65536 <= source.numel() * 2 <= 2 << 20 else "OneshotAllReduce.all_reduce")
                assert operation.plan.query.surface == expected_surface, (name, operation.plan.query.surface)
                cases.append((name, source, output, result, operation))
            session.tune(show_progress=False)
            # Unused declarations must not invalidate capture of the prepared catalog.
            _, source, output, _, _ = cases[1]
            unused = runtime.plan(source, out=output)
            assert unused._prepared is None
            compiled = []
            stream = torch.cuda.Stream(device=device)
            for name, source, output, result, operation in cases:
                def forward(source=source, result=result, operation=operation):
                    result.copy_(operation(source) * 2)
                function = compile_forward(forward)
                compiled.append(function)
                source.fill_(rank + 1)
                dist.barrier()
                function()
                torch.cuda.synchronize(device)
                source.fill_(rank + 1)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                    function()
                stream.synchronize()
                graphs[name] = graph
            session.freeze()
            buffers = {f"{name}-{role}": value for name, source, output, result, _ in cases
                       for role, value in (("source", source), ("output", output), ("result", result))}
            bundle = PreparedGraphBundle(device=rank, graphs=graphs, buffers=buffers,
                resources=(runtime, session, stream, cases, compiled))
            worker = bundle.start_worker()
            for step in range(4):
                for name, source, output, result, _ in cases[::1 if step % 2 else -1]:
                    source.fill_(rank + 1 + step)
                    result.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    before = torch.cuda.memory_stats(device)
                    dist.barrier()
                    ticket = worker.submit(bundle.graph_indices[name])
                    completion = worker.poll(timeout_ms=120000)
                    assert completion is not None and completion.ticket == ticket
                    assert completion.error is None
                    after = torch.cuda.memory_stats(device)
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key], (name, key)
                    torch.testing.assert_close(result, torch.full_like(result, 2 * (10 + 4 * step)), rtol=0, atol=0)
                    dist.barrier()
            worker.close()
            worker = None
            for graph in graphs.values():
                graph.reset()
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


def test_native_compiled_output_aware_all_reduce(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)
