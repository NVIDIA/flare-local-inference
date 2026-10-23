"""Ordered expert route groups consumed by collective normalization staging."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _rank(rank, world, rendezvous):
    from b12x.comm.pcie.pcie_oneshot import _eager_storage_shards
    from ignition.components.collectives import PreparedAllReduceRMSNorm, WeightedGroupSource
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=8))
    graphs = []
    try:
        with PackageStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4096,
                hidden_size=6144, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="weighted-group-staging",
                        workload_schema="ordered-group-bf16", numerical_policy="zero-centered",
                        coordinator=coordinator) as session:
                    calls = []
                    for rows in (1, 48, 49, 4096):
                        for ids_dtype in (torch.int32, torch.int64):
                            values = torch.zeros((rows, 4, 6144), device=device, dtype=torch.bfloat16)
                            weights = torch.zeros((rows, 4), device=device)
                            ids = torch.zeros((rows, 4), device=device, dtype=ids_dtype)
                            grouped = WeightedGroupSource(values, weights, ids, 128)
                            source = torch.zeros((rows, 6144), device=device, dtype=torch.bfloat16)
                            addend = torch.zeros_like(source)
                            residuals = (torch.zeros_like(source), torch.zeros_like(source))
                            outputs = (torch.empty_like(source), torch.empty_like(source))
                            gain = torch.linspace(-.5, .5, 6144, device=device, dtype=source.dtype)
                            operations = tuple(PreparedAllReduceRMSNorm(inp, residual, gain, output,
                                eps=1e-6, runtime=runtime, session=session, coordinator=coordinator,
                                name=f"groups:{rows}:{ids_dtype}:{index}", zero_centered=True,
                                input_addend=addend, reduction_dtype=torch.bfloat16)
                                for index, (inp, residual, output) in enumerate(zip(
                                    (source, grouped), residuals, outputs, strict=True)))
                            calls.append((values, weights, ids, source, addend, residuals, outputs, operations))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream(device=device)
                    for values, _, _, source, _, residuals, _, operations in calls:
                        for operation, residual in zip(operations, residuals, strict=True):
                            operation(source if operation.group_id_limit == 0 else values, residual)
                            torch.cuda.synchronize(device)
                            graph = torch.cuda.CUDAGraph()
                            stream.wait_stream(torch.cuda.current_stream(device))
                            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                                operation(source if operation.group_id_limit == 0 else values, residual)
                            graphs.append(graph)
                    session.freeze()
                    channel = runtime.for_stream()
                    native = channel._ext._state(channel._ptr)
                    staging_bytes = channel.eager_buffer_bytes * _eager_storage_shards(world, native.transport_policy)
                    for iteration in range(4):
                        for index, (values, weights, ids, source, addend, residuals, outputs, _) in enumerate(calls):
                            values.normal_()
                            weights.uniform_(-1, 1)
                            ids.random_(-1, 130)
                            ids[:, 1].copy_(ids[:, 0])
                            if iteration == 1:
                                ids.fill_(-1)
                            elif iteration == 2:
                                ids[source.shape[0] // 2:].fill_(128)
                            invalid = (ids < 0) | (ids >= 128)
                            values[invalid] = float("nan")
                            reduced = torch.zeros_like(source, dtype=torch.float32)
                            for group in range(4):
                                term = torch.where(~invalid[:, group, None], values[:, group].float(), 0.)
                                reduced = reduced + term * weights[:, group, None]
                            source.copy_(reduced)
                            addend.normal_(std=.02)
                            residuals[0].normal_(std=.1)
                            residuals[1].copy_(residuals[0])
                            for offset, graph in enumerate(graphs[index * 2:index * 2 + 2]):
                                outputs[offset].fill_(float("nan"))
                                torch.cuda.synchronize(device)
                                dist.barrier()
                                for pointers in channel._eager_ptrs:
                                    channel._ipc.cudaMemset(pointers[rank], 255, staging_bytes)
                                torch.cuda.synchronize(device)
                                dist.barrier()
                                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                                graph.replay()
                                torch.cuda.synchronize(device)
                                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            torch.testing.assert_close(residuals[1], residuals[0], rtol=0, atol=0)
                            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
            finally:
                for graph in graphs:
                    graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


def test_weighted_group_staging_matches_precomputed_ordered_sum(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned GPUs")
    mp.spawn(_rank, args=(4, (tmp_path / "groups").as_uri()), nprocs=4, join=True)
