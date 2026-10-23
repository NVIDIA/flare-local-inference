"""Export a compact b12x Engram graph and file-backed table for native replay tests."""

from pathlib import Path
from signing import publisher
from fixture_frontend import server_config
import json
import sys

import torch
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.runtime_package import FileSource, export_model


@torch.no_grad()
def prepare(root, shard_size=64 << 30, table_world=1):
    root.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    with ModuleImages() as images, PackageStorage() as storage:
        from b12x.preparation import PreparationSession
        from b12x.sequence import engram
        from ignition.components.engram import EngramInputs, PreparedEngramLookup
        from ignition.components.parallel import TensorParallelShard
        from ignition.components.ngram_delivery import publish_requests
        from ignition.components.readiness import ReadinessTimings
        from ignition.preparation.execution import PreparedExecution, GraphSignature
        from ignition.recipes.deepseek41.engram import Tables

        geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
        rows = geometry.num_embeddings[0]
        weight = (torch.arange(rows * 256).remainder(15).float() - 7).to(torch.float8_e4m3fn)
        scales = torch.full((rows * 8,), 127, dtype=torch.uint8)
        sources = []
        for name, tensor in [('weights', weight), ('scales', scales)]:
            path = root / name
            path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
            sources.append(FileSource(path, 0, tensor.nbytes))
        device = torch.device('cuda:0')
        # Preparation can initialize library storage that the serving graph does not reference.
        warmup = torch.ones((32, 32), device=device)
        torch.mm(warmup, warmup)
        torch.cuda.synchronize()
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        inputs = EngramInputs.allocate(4, device, timings=ReadinessTimings.allocate(frame, 1))
        output = torch.empty((4, 6144), dtype=torch.bfloat16, device=device)
        count = torch.full((1,), 4, dtype=torch.int32, device=device)
        source = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        transfer_source = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
        transfer_target = torch.empty_like(transfer_source)
        mailbox = storage.host_allocate("engram_1_requests", (4 * 24 + 1,), torch.int64)
        mailbox.host_view.zero_()
        records = torch.zeros((2, 4, 8), dtype=torch.int64, device=device)
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            lookup = PreparedEngramLookup(inputs, output, count, layer=1, token_map=tuple(range(32)),
                geometry=geometry, session=session, name='engram')

            def run():
                transfer_target.copy_(transfer_source)
                source.add_(1)
                publish_requests(inputs.hashes, mailbox.device_view, sequence)
                lookup(source)
                records.zero_()
                records[:, 0, 0] = 1
                records[:, 0, 1].copy_(sequence[0].expand(2))
                records[:, 0, 4] = -1
                records[:, 0, 5] = 1

            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph, pool=storage.graph_pool.id):
                run()
            graph.instantiate()
            # Session cleanup unloads compiled kernels; retain it until export has read the graph.
            try:
                tables = Tables(geometry, tuple(range(32)), {1: tuple(sources)}, {1: inputs}, {1: mailbox}, sequence,
                    TensorParallelShard(0, table_world), {},
                    torch.arange(6144, device=device))
                reference_rows = [row for rank in range(table_world)
                                  for first, last in tables.partition(0, rank).source_rows
                                  for row in range(first, last)]
                (root / 'reference_rows.json').write_text(json.dumps(reference_rows))
                buffers = dict(tables.buffers(), output=output, transfer_source=transfer_source, frame_index=frame,
                               submission_sequence=sequence, completion_records=records)
                prepared = PreparedExecution(device=0, graphs={'decode': graph}, buffers=buffers,
                    signatures={'decode': GraphSignature('decode', 4, 1, ('engram_1_hashes', 'transfer_source'))},
                    max_active=4, resources=(storage, lookup), export_events=tables.events())
                prepared.server_config = server_config(root)
                prepared.package_data = tables
                if table_world == 1:
                    export_model(prepared, images, storage, root / 'package', options={}, publisher=publisher(),
                                 shard_size=shard_size)
                else:
                    from ignition.preparation import runtime_package_pb2 as proto
                    from ignition.preparation.package_file import publish, read_index
                    from ignition.preparation.protection import resource_roles

                    staged = root / 'table-source'
                    export_model(prepared, images, storage, staged, options={})
                    index, start = read_index(staged)
                    # The replay test binds one graph to each partition separately. Empty
                    # peer bundles provide storage owners without allocating more GPUs.
                    index.ranks.extend(proto.BundleIndex() for _ in range(table_world - 1))
                    payload = sorted((span.offset, staged, start + span.offset, span.nbytes)
                                     for _, _, span in resource_roles(index))
                    publish(root / 'package', index, payload, publisher=publisher(), shard_size=shard_size)
                    staged.unlink()
            finally:
                graph.reset()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shard-size", type=int, default=64 << 30)
    parser.add_argument("--table-world", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    prepare(args.directory, args.shard_size, args.table_world)
