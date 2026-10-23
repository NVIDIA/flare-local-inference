"""Recipient-ordered owner reduction matches staged pull under graph transitions."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _rank(rank, world, rendezvous):
    from dataclasses import replace
    from types import MappingProxyType

    from b12x._lib.compile_plan import load_programs
    from b12x.comm.pcie._oneshot_preparation import (
        _OneshotExecutionState, compile_oneshot_surface,
    )
    from b12x.comm.pcie._tuning import TUNING
    from b12x.comm.pcie.pcie_oneshot import _eager_storage_shards
    from b12x.preparation import FrozenMapping
    from b12x.preparation.types import require_prepared
    from ignition.components.collectives import PreparedAllReduceRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    graphs, controls = [], []
    try:
        with PackageStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4096,
                hidden_size=6144, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="recipient-ordered-owner",
                        workload_schema="rounded-cyclic-reduction", numerical_policy="source-dtype",
                        coordinator=coordinator, compile_workers=4) as session:
                    calls = []
                    for rows, hidden in ((48, 6144), (1, 6144), (47, 6144), (49, 6144),
                                         (4096, 6144), (1, 520), (49, 520), (1, 136)):
                        for dtype in (torch.bfloat16, torch.float16):
                            source = torch.zeros((rows, hidden), device=device, dtype=dtype)
                            addend = torch.zeros_like(source) if rows % 2 == 0 else None
                            residual = torch.zeros_like(source)
                            reference_residual = torch.zeros_like(source)
                            output, reference_output = torch.empty_like(source), torch.empty_like(source)
                            gain = torch.linspace(-.5, .5, hidden, device=device).to(dtype)
                            operation = PreparedAllReduceRMSNorm(source, residual, gain, output,
                                eps=1e-6, runtime=runtime, session=session, coordinator=coordinator,
                                name=f"owner:{rows}:{hidden}:{dtype}", zero_centered=True,
                                input_addend=addend, reduction_dtype=dtype)
                            calls.append((source, addend, residual, reference_residual,
                                          output, reference_output, gain, operation))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream(device=device)
                    for source, addend, residual, reference_residual, output, reference_output, gain, operation in calls:
                        state = require_prepared(operation.plan, "comm.pcie", device)
                        selected = state.query.call["mode"]
                        assert selected == ("stage_remote_push" if source.shape == (1, 6144) else "stage_tp4_owner")
                        modes = [None] * world
                        dist.all_gather_object(modes, selected)
                        assert len(set(modes)) == 1
                        query = replace(state.query, call=FrozenMapping(dict(state.query.call, mode="stage_pull")))
                        programs = load_programs(compile_oneshot_surface(TUNING.encode_query(query), rank))
                        control = _OneshotExecutionState(query, state.runtime, MappingProxyType(programs))
                        control.bind_input(source)
                        controls.append(control)
                        operation(source, residual)
                        control.run_fused(source, reference_residual, gain, reference_output,
                                          reference_residual, 1e-6, addend)
                        torch.cuda.synchronize(device)
                        for invoke in (lambda operation=operation, source=source, residual=residual:
                                       operation(source, residual),
                                       lambda control=control, source=source, reference_residual=reference_residual,
                                       gain=gain, reference_output=reference_output, addend=addend:
                                       control.run_fused(source, reference_residual, gain, reference_output,
                                                         reference_residual, 1e-6, addend)):
                            graph = torch.cuda.CUDAGraph()
                            stream.wait_stream(torch.cuda.current_stream(device))
                            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                                invoke()
                            graphs.append(graph)
                    session.freeze()
                    channel = runtime.for_stream()
                    native = channel._ext._state(channel._ptr)
                    stage_bytes = channel.eager_buffer_bytes * _eager_storage_shards(world, native.transport_policy)
                    for turn in range(3):
                        for index, (source, addend, residual, reference_residual, output, reference_output, _, _) in enumerate(calls):
                            source.normal_(std=.125)
                            amplitude = 2.**(24 if source.dtype == torch.bfloat16 else 15)
                            pattern = (amplitude, 2.**-12, -amplitude, 2.**-12)
                            source[:, ::2] = pattern[(rank + turn) % world]
                            if addend is not None:
                                addend.zero_()
                                addend[:, 1::2].fill_(rank / 256)
                            residual.fill_((rank + 1) / 8192)
                            reference_residual.copy_(residual)
                            output.fill_(float("nan"))
                            reference_output.fill_(float("nan"))
                            combined = source if addend is None else (source.float() + addend.float()).to(source.dtype)
                            gathered = [torch.empty_like(combined) for _ in range(world)]
                            dist.all_gather(gathered, combined)
                            reduced = gathered[rank].float()
                            for offset in range(1, world):
                                reduced = reduced + gathered[(rank + offset) % world].float()
                            expected = (residual.float() + reduced.to(source.dtype).float()).to(source.dtype)
                            for graph in graphs[index * 2:index * 2 + 2]:
                                torch.cuda.synchronize(device)
                                dist.barrier()
                                for pointers in channel._eager_ptrs:
                                    channel._ipc.cudaMemset(pointers[rank], 255, stage_bytes)
                                torch.cuda.synchronize(device)
                                dist.barrier()
                                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                                graph.replay()
                                torch.cuda.synchronize(device)
                                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            torch.testing.assert_close(residual, expected, atol=0, rtol=0)
                            torch.testing.assert_close(residual, reference_residual, atol=0, rtol=0)
                            torch.testing.assert_close(output, reference_output, atol=0, rtol=0)
            finally:
                for graph in graphs:
                    graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


def test_owner_preserves_recipient_order_and_graph_storage(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned GPUs")
    mp.spawn(_rank, args=(4, (tmp_path / "owner").as_uri()), nprocs=4, join=True)
