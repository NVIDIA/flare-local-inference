"""Package-owned TP4 reductions preserve arithmetic across graph variants."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires four explicitly assigned GPUs")


def _rank(rank, rendezvous):
    from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllReduce
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=5))
    try:
        with PackageStorage(rank=rank, world=4, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=128,
                                               hidden_size=5120, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="tp4-reductions",
                        workload_schema="bf16-fp32-reductions", numerical_policy="default",
                        coordinator=coordinator, compile_workers=4) as session:
                    operations = []
                    cases = [(n, torch.bfloat16) for n in (1, 4, 8, 16, 24, 32, 40, 48, 64, 128, 4096)]
                    for rows, dtype in [*cases, (8, torch.float32)]:
                        generator = torch.Generator().manual_seed(1701 + rank)
                        host = (torch.randint(-2048, 2049, (rows, 5120), generator=generator).float() / 1024).to(dtype)
                        peers = [torch.empty_like(host) for _ in range(4)]
                        dist.all_gather(peers, host)
                        source, output = host.to(device), torch.empty_like(host, device=device)
                        component = PreparedChunkedAllReduce if rows == 4096 else PreparedAllReduce
                        operation = component(source, output, runtime=runtime, session=session,
                                              coordinator=coordinator, name=f"reduce.{rows}.{dtype}")
                        if rows != 4096:
                            expected_surface = ("PCIeTwoShotBF16.all_reduce"
                                if dtype == torch.bfloat16 and 8 <= rows <= 128 else "OneshotAllReduce.all_reduce")
                            assert operation.plan.query.surface == expected_surface
                        expected = torch.stack(peers).float().sum(0).to(dtype).to(device)
                        operations.append((source, output, expected, operation))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream()
                    graphs = []
                    for indices in ((0, 2, 5, 1, 8, 11, 9), (7, 0, 4, 6, 3), (10,)):
                        for index in indices:
                            source, _, _, operation = operations[index]
                            operation(source)
                        torch.cuda.synchronize()
                        stream.wait_stream(torch.cuda.current_stream())
                        graph = torch.cuda.CUDAGraph()
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                            for index in indices:
                                source, _, _, operation = operations[index]
                                operation(source)
                        graphs.append((graph, indices))
                    session.freeze()
                    for step in range(12):
                        for source, output, expected, _ in operations:
                            source.neg_()
                            expected.neg_()
                            output.fill_(float("nan"))
                        torch.cuda.synchronize()
                        for graph, indices in graphs[::(-1 if step % 2 else 1)]:
                            allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
                            for index in indices:
                                _, output, expected, _ = operations[index]
                                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                        dist.barrier()
                    for graph, _ in graphs:
                        graph.reset()
            finally:
                runtime.close()
    finally:
        dist.destroy_process_group()


def test_package_owned_tp4_reductions(tmp_path):
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)
