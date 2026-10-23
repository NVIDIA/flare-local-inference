"""Package-owned TP4 reductions across retained graph variants and native replay."""

from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1",
)


def _operands(rows, columns, step):
    generator = torch.Generator().manual_seed(4181 + columns + step)
    values = [(torch.randn((rows, columns), generator=generator)
               * 2. ** torch.randint(-12, 13, (rows, columns), generator=generator)).bfloat16()
              for _ in range(4)]
    # Cancellation exposes the FP32 addition order before the final BF16 cast.
    values[2][:rows // 2] = -values[0][:rows // 2]
    values[3][:rows // 2] = -values[1][:rows // 2]
    return values


def _expected(values, rank, surface):
    cyclic = []
    for first in range(4):
        value = values[first].float()
        for offset in range(1, 4):
            value = value + values[(first + offset) % 4].float()
        cyclic.append(value.bfloat16())
    if surface == "PCIeTwoShotBF16.all_reduce":
        # Each quarter is summed once by its owner, then copied to every rank.
        elements = values[0].numel()
        quarter = elements // 4
        return torch.cat([cyclic[owner].flatten()[owner * quarter:(owner + 1) * quarter]
                          for owner in range(4)]).view_as(values[0])
    assert surface == "OneshotAllReduce.all_reduce"
    return cyclic[rank]


def _rank(rank, rendezvous, directory, widths):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=5))
    max_rows = max((2 << 20) // (width * 2) + 1 for width in widths)
    runtime = worker = None
    graphs = {}
    try:
        with ModuleImages() as images, PackageStorage(rank=rank, world=4, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=max_rows,
                hidden_size=max(widths), ipc=storage.peer_storage)
            with RecordedPreparationSession(device=device, checkpoint="tp4-collectives",
                    workload_schema="bf16-hidden-reduction", numerical_policy="fp32-single-rounding",
                    coordinator=coordinator) as session:
                # Hidden widths share one runtime, which
                # transports flat BF16 packs independently of tensor row geometry.
                sources = {width: torch.empty((max_rows, width), dtype=torch.bfloat16, device=device)
                           for width in widths}
                outputs, operations = {}, {}
                for width, source in sources.items():
                    upper = (2 << 20) // (width * 2)
                    row_counts = (1, 8, 12, 13, 32, 33, 128, upper, upper + 1)
                    for rows in row_counts:
                        key = width, rows
                        outputs[key] = torch.empty_like(source[:rows])
                        operations[key] = PreparedAllReduce(source[:rows], outputs[key], runtime=runtime,
                            session=session, coordinator=coordinator, name=f"reduce-{width}-{rows}")
                session.tune(show_progress=False)
                surfaces = {key: operation.plan.query.surface for key, operation in operations.items()}
                assert set(surfaces.values()) == {
                    "OneshotAllReduce.all_reduce", "PCIeTwoShotBF16.all_reduce"}
                for (width, rows), surface in surfaces.items():
                    nbytes = rows * width * 2
                    expected = ("PCIeTwoShotBF16.all_reduce" if 64 << 10 <= nbytes <= 2 << 20
                                else "OneshotAllReduce.all_reduce")
                    assert surface == expected, (width, rows, nbytes)
                if 2560 in widths:
                    # BF16[12,2560] is 61,440 bytes; 13 rows first exceed 64 KiB.
                    assert surfaces[2560, 12] == "OneshotAllReduce.all_reduce"
                    assert surfaces[2560, 13] == "PCIeTwoShotBF16.all_reduce"
                stream = torch.cuda.Stream(device=device)
                for epoch in range(2):
                    for (width, rows), operation in operations.items():
                        sources[width].fill_(rank + 1)
                        operation(sources[width][:rows])
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                                graph, stream=stream, pool=storage.graph_pool.id):
                            operation(sources[width][:rows])
                        graph.instantiate()
                        graphs[f"reduce-{width}-{rows}-{epoch}"] = graph
                session.freeze()

                # Both transports must use the same explicit VMM ownership path
                # that the runtime-package exporter uses for peer resources.
                peer = storage.peer_storage
                assert len(peer.owned) >= 2 and len(peer.imported) >= 6
                for address, size in peer.owned.items():
                    assert address in storage.mappings
                    assert storage.mappings[address].nbytes == size
                for parts in peer.imported_mappings.values():
                    for address, size in parts:
                        assert images.allocations[address] == (size, "device")
                owners = [None] * 4
                dist.all_gather_object(owners, dict(peer.owned))
                other_owners = {address: size for index, owned in enumerate(owners) if index != rank
                                for address, size in owned.items()}
                assert dict(peer.imported) == other_owners

                buffers = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": torch.zeros((2, 1, 8), dtype=torch.int64, device=device)}
                buffers.update({f"source-{width}": value for width, value in sources.items()})
                buffers.update({f"output-{width}-{rows}": value
                                for (width, rows), value in outputs.items()})
                bundle = PreparedExecution(device=rank, graphs=graphs, buffers=buffers,
                    signatures={name: GraphSignature("decode", 1, 1) for name in graphs}, max_active=1,
                    resources=(runtime, session, stream, storage, operations, sources, outputs))
                bundle.server_config = {}
                bundle.package_data = None
                torch.cuda.synchronize(device)
                dist.barrier()
                peer.reset()
                torch.cuda.synchronize(device)
                dist.barrier()
                # Stage a tiny unsigned rank bundle, not a model checkpoint.
                package = export_model(bundle, images, storage,
                    directory / f"rank-{rank}.json", options={}, bundle_only=True, sources=True)
                assert package.modules
                exported = {(allocation.address, allocation.nbytes) for allocation in package.allocations
                            if allocation.memory == "peer"}
                assert exported == set(peer.imported.items())
                worker = bundle.start_executor()
                names = list(graphs)
                # Retain both capture epochs while interleaving all capacities.
                for step in range(3):
                    for index in (list(range(len(names))) if step % 2 == 0
                                  else list(reversed(range(len(names))))):
                        _, width, rows, _ = names[index].split("-")
                        width, rows = int(width), int(rows)
                        values = _operands(max_rows, width, step)
                        sources[width].copy_(values[rank])
                        for output in outputs.values():
                            output.fill_(float("nan"))
                        expected_source = sources[width].clone()
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        before = torch.cuda.memory_stats(device)
                        ticket = worker.submit(index, [])
                        assert ticket is not None
                        deadline = time.monotonic() + 60
                        completion = worker.poll()
                        while completion is None and time.monotonic() < deadline:
                            time.sleep(.001)
                            completion = worker.poll()
                        assert completion is not None and completion.fence == ticket
                        after = torch.cuda.memory_stats(device)
                        for key in ("allocation.all.allocated", "allocation.all.freed",
                                    "segment.all.allocated", "segment.all.freed"):
                            assert before[key] == after[key]
                        assert torch.equal(sources[width].view(torch.uint8), expected_source.view(torch.uint8))
                        expected = _expected([value[:rows] for value in values], rank, surfaces[width, rows])
                        assert torch.equal(outputs[width, rows].cpu().view(torch.uint8), expected.view(torch.uint8))
                worker.close()
                assert worker.drained
                worker = None
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
                runtime.close()
                runtime = None
                assert not peer.owned and not peer.imported and not peer.imported_mappings
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("widths", [(4096, 5120), (2560,)], ids=["4096-5120", "qwen-2560"])
def test_package_owned_tp4_collective_graph_variants(tmp_path, widths):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    assert torch.cuda.device_count() == 4
    mp.spawn(_rank, args=((tmp_path / "gloo").as_uri(), tmp_path, widths), nprocs=4, join=True)


def test_exported_tp4_collectives_replay_after_preparation_exit(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four CUDA devices explicitly"
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "helios" / "target"))
    subprocess.run(["cargo", "build", "-j", "4", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=600)
    result = subprocess.run([sys.executable, str(root / "helios/tests/collective_package_fixture.py"),
        str(tmp_path)], cwd=root, text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all four preparation ranks exited" in result.stdout
    # subprocess.run waits for the Python exporter; the Rust reader independently
    # rejects any still-live recorded preparation PID before initializing CUDA.
    result = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(tmp_path), "--tp4-collective"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native TP4 collective package replay passed after Python exit" in result.stdout
