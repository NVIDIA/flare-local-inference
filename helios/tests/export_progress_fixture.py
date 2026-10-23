"""Capture and export a CUDA graph through the preparation CLI progress surface."""

import argparse
from pathlib import Path

import torch

from fixture_frontend import server_config
from signing import publisher
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.runtime_package import export_model
from ignition.progress import StartupProgress, phase


def export(root, rank):
    root.mkdir(parents=True)
    with ModuleImages() as images:
        torch.cuda.set_device(0)
        with PackageStorage(rank=rank, world=rank + 1) as storage:
            source = torch.arange(1024, dtype=torch.float32, device="cuda")
            output = torch.zeros_like(source)
            buffers = {
                "frame_index": torch.zeros(1, dtype=torch.int32, device="cuda"),
                "submission_sequence": torch.zeros(1, dtype=torch.int64, device="cuda"),
                "completion_records": torch.zeros((2, 1, 8), dtype=torch.int64, device="cuda"),
                "source": source, "output": output,
            }
            torch.add(source, 1, out=output)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=storage.graph_pool.id):
                torch.add(source, 1, out=output)
            graph.instantiate()
            graph.replay()
            torch.cuda.synchronize()
            prepared = PreparedExecution(device=0, graphs={"decode": graph}, buffers=buffers,
                signatures={"decode": GraphSignature("decode", 1, 1, ("source",))}, max_active=1,
                resources=(storage, source, output))
            prepared.server_config = server_config(root)
            prepared.package_data = None
    path = root / "package"
    if rank:
        export_model(prepared, images, storage, path, options={}, sources=True, bundle_only=True)
    else:
        progress = StartupProgress()
        with phase(progress, "Export native package", live=False) as stage:
            export_model(prepared, images, storage, path, options={}, publisher=publisher(), progress=progress)
            stage.update(path)
    assert path.is_file()
    print(f"EXPORTED RANK {rank}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("rank", type=int, choices=(0, 1))
    args = parser.parse_args()
    export(args.directory, args.rank)
