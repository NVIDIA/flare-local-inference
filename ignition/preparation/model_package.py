"""Publish a model package containing one native resource bundle per rank."""

from pathlib import Path

from .runtime_package import export_model
from .package_file import DEFAULT_SHARD_SIZE, combine
from .memory_inventory import print_inventory


def export_package(prepared, images, storage, path, *, options, publisher, progress=None, console=None,
                   staging=None, shard_size=DEFAULT_SHARD_SIZE):
    """Export every rank, close preparation resources, and assemble the model file.

    Ranks reuse immutable allocation blobs and stage execution resources under `staging`
    (default: beside `path`). Publication links payload blobs beside the metadata primary.
    """
    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")
    import torch.distributed as dist

    path = Path(path)
    import flare_heliopause

    native_stage = flare_heliopause.NativeExportStage(path, staging)
    stage = Path(native_stage.path)
    from .package_snapshot import _OPTIONS, ExportOptions
    token = _OPTIONS.set(_OPTIONS.get() or ExportOptions(path.parent / '.flare-cache'))
    try:
        roots = [f'rank-{rank}.json' for rank in range(storage.world)]
        if storage.world > 1:
            for rank, control in enumerate(prepared.controls, 1):
                control.send(('export', stage / roots[rank], options, _OPTIONS.get()))
            dist.barrier()
            storage.peer_storage.reset()
            dist.barrier()
        model = prepared if storage.world == 1 else prepared.model
        package = export_model(model, images, storage, stage / roots[0], options=options, progress=progress,
                               sources=True)
        inventories = [package.memory_inventory]
        memory = [[]] if storage.world == 1 else [list(storage.peer_storage.owned.items())]
        if storage.world > 1:
            for control in prepared.controls:
                reply = control.recv()
                memory.append(reply['peer_memory'])
                inventories.append(reply['memory_inventory'])
        peers = []
        for rank, allocations in enumerate(memory):
            for address, size in allocations:
                peers.append(dict(owner=rank, address=address, nbytes=size, readers=list(range(storage.world))))
        prepared.close()
        for region in storage.host_regions:
            region.close()
        combine(path, [stage / root for root in roots], peers, publisher=publisher, shard_size=shard_size)
        print_inventory(inventories, console=console or getattr(progress, "console", None))
    finally:
        _OPTIONS.reset(token)
        native_stage.close()
