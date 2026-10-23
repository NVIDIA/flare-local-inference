"""Final residual reduction and indexed head normalization preserve canonical boundaries."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _rank(rank, world, rendezvous):
    from ignition.components.collectives import PreparedAllReduce, PreparedAllReduceAdd, WeightedGroupSource
    from ignition.components.rms_norm import PreparedIndexedRMSNorm, PreparedRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    graphs = []
    try:
        with PackageStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4096,
                hidden_size=6144, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="indexed-head",
                        workload_schema="final-residual-head", numerical_policy="source-dtype",
                        coordinator=coordinator, compile_workers=4) as session:
                    calls = []
                    for rows, samples in ((48, 48), (1, 1), (47, 8), (49, 8), (4096, 8)):
                        source = torch.zeros(rows, 4, 6144, device=device, dtype=torch.bfloat16)
                        weights = torch.zeros(rows, 4, device=device)
                        ids = torch.zeros(rows, 4, device=device, dtype=torch.int32)
                        addend = torch.zeros(rows, 6144, device=device, dtype=torch.bfloat16)
                        residual, seed = torch.zeros_like(addend), torch.zeros_like(addend)
                        local, reduced = torch.zeros_like(addend), torch.zeros_like(addend)
                        control = torch.zeros_like(addend)
                        indices = torch.zeros(samples, device=device, dtype=torch.int64)
                        output = torch.zeros(samples, 6144, device=device, dtype=torch.bfloat16)
                        gathered, reference = torch.zeros_like(output), torch.zeros_like(output)
                        gain = torch.linspace(-.5, .5, 6144, device=device).bfloat16()
                        collective = PreparedAllReduceAdd(WeightedGroupSource(source, weights, ids, 128), residual,
                            runtime=runtime, session=session, coordinator=coordinator,
                            name=f"final:{rows}", reduction_dtype=torch.bfloat16, input_addend=addend)
                        plain = PreparedAllReduce(local, reduced, runtime=runtime, session=session,
                            coordinator=coordinator, name=f"plain:{rows}")
                        norm = PreparedIndexedRMSNorm(residual, indices, gain, output, eps=1e-6,
                            zero_centered=True, session=session, name=f"indexed:{rows}")
                        canonical = PreparedRMSNorm(gathered, gain, reference, eps=1e-6,
                            zero_centered=True, session=session, name=f"canonical:{rows}")
                        calls.append((source, weights, ids, addend, residual, seed, local, reduced,
                                      control, indices, output, gathered, reference, collective, plain, norm, canonical))
                    session.tune(show_progress=False)
                    source, weights, ids, addend, residual, *rest = calls[0]
                    collective = calls[0][-4]
                    overlapping = source.view(-1)[:residual.numel()].view_as(residual)
                    with pytest.raises(ValueError, match="overlap collective contributions"):
                        runtime.all_reduce_fused_add(source, overlapping, plan=collective.plan,
                            input_addend=addend, group_weights=weights, group_ids=ids, group_id_limit=128)
                    stream = torch.cuda.Stream(device=device)
                    for call in calls:
                        source, _, _, _, residual, _, _, _, _, indices, _, _, _, collective, _, norm, _ = call
                        def forward(source=source, residual=residual, indices=indices, collective=collective, norm=norm):
                            collective(source, residual)
                            return norm(residual, indices)
                        compiled = compile_forward(forward)
                        compiled()
                        graph = torch.cuda.CUDAGraph()
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                            compiled()
                        graphs.append(graph)
                    session.freeze()
                    channel = runtime.for_stream()
                    native = channel._ext._state(channel._ptr)
                    from b12x.comm.pcie.pcie_oneshot import _eager_storage_shards
                    stage_bytes = channel.eager_buffer_bytes * _eager_storage_shards(world, native.transport_policy)
                    for turn in range(3):
                        for call, graph in zip(calls, graphs, strict=True):
                            (source, weights, ids, addend, residual, seed, local, reduced,
                             control, indices, output, gathered, reference, collective, plain, norm, canonical) = call
                            source.normal_(std=.2)
                            source.mul_(rank + 1)
                            weights.copy_(torch.tensor([.25, -.5, .75, .125], device=device))
                            ids.copy_(torch.tensor([0, 3, -1, 128], device=device, dtype=ids.dtype))
                            ids[::3, 2] = 5
                            addend.normal_(std=.1)
                            seed.normal_(std=.2)
                            seed.add_(rank * .125)
                            residual.copy_(seed)
                            control.copy_(seed)
                            accumulator = torch.zeros_like(local, dtype=torch.float32)
                            for group in range(4):
                                values = torch.where(((ids[:, group] >= 0) & (ids[:, group] < 128))[:, None],
                                                     source[:, group].float(), 0.)
                                product = values * weights[:, group, None]
                                accumulator.add_(product)
                            local.copy_((accumulator.bfloat16().float() + addend.float()).bfloat16())
                            plain(local)
                            control.add_(reduced)
                            indices.copy_((torch.arange(indices.numel(), device=device).flip(0) + turn * 7) % source.shape[0])
                            if indices.numel() > 1:
                                indices[1::2] = indices[0]
                            gathered.copy_(control[indices])
                            canonical(gathered)
                            output.fill_(float("nan"))
                            torch.cuda.synchronize()
                            dist.barrier()
                            for pointers in channel._eager_ptrs:
                                channel._ipc.cudaMemset(pointers[rank], 255, stage_bytes)
                            torch.cuda.synchronize()
                            dist.barrier()
                            allocated = torch.cuda.memory_allocated()
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.cuda.memory_allocated() == allocated
                            torch.testing.assert_close(residual, control, atol=0, rtol=0)
                            torch.testing.assert_close(output, reference, atol=.002, rtol=.01)
                    for graph in graphs:
                        graph.reset()
                    graphs.clear()
            finally:
                runtime.close()
    finally:
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group()


def test_final_collective_and_indexed_head(tmp_path):
    mp.spawn(_rank, args=(4, "file://" + str(tmp_path / "rendezvous")), nprocs=4, join=True)
