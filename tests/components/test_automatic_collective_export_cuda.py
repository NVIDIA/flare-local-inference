"""Export automatic collective startup graphs and their default cold-cache storage."""

from datetime import timedelta
import os
from pathlib import Path
import sys

import pytest


def _rank(rank, directory, readiness):
    import torch
    import torch.distributed as dist

    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collective_ladder import load
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.model import PreparedModel
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.tuning import RecordedPreparationSession

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "helios/tests"))
    from collective_package_fixture import _configuration

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(directory / "gloo").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(minutes=5))
    runtime = session = prepared = None
    graphs = {}
    try:
        with ModuleImages() as images, PackageStorage(rank=rank, world=4, device=rank) as storage:
            try:
                coordinator = CollectivePreparation(dist.group.WORLD)
                runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=13,
                    hidden_size=2560, ipc=storage.peer_storage, collective_tuning="runtime")
                session = RecordedPreparationSession(device=device, checkpoint="collective-startup-export",
                    workload_schema="tp4-bf16-hidden", numerical_policy="fp32-single-rounding",
                    coordinator=coordinator, collective_tuning="runtime", compile_workers=4, benchmark_storage=storage,
                    collective_ladder=load(Path(__file__).resolve().parents[2] / "ignition/preparation/ladders/gb202_pcie_tp4.json"))
                if readiness:
                    from ignition.components.readiness import PreparedReadinessWait, ReadinessTimings

                    timings = ReadinessTimings(torch.zeros(1, dtype=torch.int32, device=device), 1)
                    wait = PreparedReadinessWait(timings, 0, runtime=runtime, session=session,
                        coordinator=coordinator, name="readiness-barrier")
                    source, output, operation = wait.input, wait.output, wait.barrier
                    assert source.numel() * source.element_size() == 16
                    assert operation.runtime_profiles == (("full", 1),)
                else:
                    source = torch.full((13, 2560), rank + 1, dtype=torch.bfloat16, device=device)
                    output = torch.empty_like(source)
                    operation = PreparedAllReduce(source, output, runtime=runtime, session=session,
                        coordinator=coordinator, name="shared-reduction")
                session.tune(show_progress=False)
                signatures = {name: GraphSignature(name, 1, 1) for name in ("init", "decode")}
                stream = torch.cuda.Stream(device=device)
                graphs, _ = capture_forwards(dict.fromkeys(signatures, lambda: operation(source)), signatures,
                    session=session, device=device, stream=stream, pool=storage.graph_pool.id,
                    collective=runtime, coordinator=coordinator, compile_decoder=False, retain_graphs=True)
                session.freeze()
                source.fill_(rank + 1)
                torch.cuda.synchronize(device)
                dist.barrier()
                graphs["decode"].replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(output, torch.full_like(output, 10), rtol=0, atol=0)
                registry = session._startup_collectives
                assert len(registry.groups) == 1
                producer = next(iter(registry.groups.values()))
                minimum_eviction = max(2 * torch.cuda.get_device_properties(device).L2_cache_size, 64 << 20)
                assert producer.eviction is not None
                assert producer.eviction.numel() * producer.eviction.element_size() == minimum_eviction
                buffers = {
                    "source": source, "output": output,
                    "frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": torch.zeros((2, 1, 24), dtype=torch.int64, device=device),
                }
                print(f"rank {rank}: capture automatic startup catalog", flush=True)
                prepared = PreparedModel(device=rank, graphs=graphs, buffers=buffers,
                    signatures=signatures, max_active=1, session=session, memory_report={},
                    server_config=_configuration(directory, signatures), close_resources=(runtime,),
                    resources=(operation, source, output, storage, stream))
                assert len(prepared.tuning_graphs) == 1
                assert prepared.tuned_bindings["init"] == prepared.tuned_bindings["decode"]
                assert len(prepared.tuned_bindings["decode"]) == 1
                binding = prepared.tuned_bindings["decode"][0]
                tuner = prepared.tuning_graphs[binding.tuning_graph]
                assert tuner.outputs == (binding.buffer,)
                assert prepared.buffers[binding.buffer].data_ptr() == operation.plan.prepared.state.selector.data_ptr()
                assert prepared.buffers["collective-tuning-0-status"] is producer.status
                aggregate = prepared.buffers[tuner.status_buffer]
                assert aggregate is not producer.status
                assert aggregate.dtype == torch.int32 and aggregate.shape == (1,)
                assert set(prepared.graphs) == {*signatures, binding.tuning_graph}
                from ignition.preparation.startup_tuning import replay_startup_tuning

                replay_startup_tuning(prepared, stream)
                output.fill_(-1)
                torch.cuda.synchronize(device)
                dist.barrier()
                graphs["decode"].replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(output, torch.full_like(output, 10), rtol=0, atol=0)
                dist.barrier()
                storage.peer_storage.reset()
                torch.cuda.synchronize(device)
                dist.barrier()
                print(f"rank {rank}: export automatic startup storage", flush=True)
                package = export_model(prepared, images, storage, directory / f"rank-{rank}.json",
                    options={}, bundle_only=True, sources=True)
                assert package.export_counts["modules"] > 0
                assert (directory / f"rank-{rank}.json").is_file()
                from ignition.preparation.package_file import read_sources_record
                from ignition.preparation import runtime_package_pb2 as proto

                index, _, _ = read_sources_record(directory / f"rank-{rank}.json")
                allocations = index.ranks[0].allocations
                outside = [(allocation.name, hex(allocation.address), allocation.nbytes)
                    for allocation in allocations if allocation.memory == proto.DEVICE
                    and not (storage.DEVICE_BASE <= allocation.address
                             and allocation.address + allocation.nbytes
                             <= storage.DEVICE_BASE + storage.RESERVATION_BYTES)]
                assert not outside, outside
                print(f"rank {rank}: automatic startup export complete", flush=True)
                dist.barrier()
            finally:
                if prepared is not None:
                    prepared.close()
                else:
                    for graph in graphs.values():
                        graph.reset()
                    if session is not None:
                        session.close()
                    if runtime is not None:
                        runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                   reason="requires four assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")
@pytest.mark.parametrize("readiness", [False, True], ids=["activation", "readiness-16b"])
def test_prepared_model_exports_automatic_collective_startup_storage(tmp_path, readiness):
    import torch
    import torch.multiprocessing as mp

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=(tmp_path, readiness), nprocs=4, join=True)
