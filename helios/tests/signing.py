"""Public test credentials shared by Python exporters and native fixture readers."""

from pathlib import Path
from ignition.preparation.runtime_package import export_model
from ignition.preparation.memory_inventory import print_inventory
from ignition.preparation.package_file import DEFAULT_SHARD_SIZE

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ignition.preparation.protection import Publisher
if __package__:
    from . import legacy_package
else:
    import legacy_package

publish_legacy = legacy_package.publish_legacy
extend_cluster_legacy = legacy_package.extend_cluster
resource_key = legacy_package.resource_key
chunk_aad = legacy_package.chunk_aad


def publisher():
    return Publisher(Ed25519PrivateKey.from_private_bytes(bytes([42]) * 32),
                     'flare-tests', 'https://huggingface.co/flare-tests/fixture')


def export_cluster_package(prepared, images, storage, path, *, peer_files, options, export_id, world,
                           publisher, peer_inventories, progress=None, shard_size=DEFAULT_SHARD_SIZE):
    """Create signed cluster containers for version 4/5 compatibility fixtures."""

    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)

    try:
        package = export_model(prepared, images, storage, partial, options=options, progress=progress)
        extend_cluster_legacy(partial, peer_files, (world, export_id))
        index, start = legacy_package.read_index(partial)
        payload = sorted((span.offset, partial, start + span.offset, span.nbytes)
                         for _, _, span in legacy_package.resource_roles(index))
        publish_legacy(path, index, payload, publisher=publisher, shard_size=shard_size)
        print_inventory([package.memory_inventory, *peer_inventories], console=getattr(progress, "console", None))
    finally:
        partial.unlink(missing_ok=True)
