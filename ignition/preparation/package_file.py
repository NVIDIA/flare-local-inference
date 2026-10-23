"""Metadata adapters for the native package publisher and reader."""

from pathlib import Path
import json

from . import runtime_package_pb2 as proto

ALIGNMENT = 4096
MAX_HEADER_BYTES = 64 << 10
MAX_INDEX_BYTES = 128 << 20
DEFAULT_SHARD_SIZE = 64 << 30


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def validate_filename(filename):
    import flare_heliopause
    flare_heliopause.export_validate_layout(json.dumps({'version': 1, 'files': {'': [], filename: []}}))
    return filename


def container_header(length, index_size=0, format='flare-data'):
    import flare_heliopause
    return flare_heliopause.export_container_header(length, index_size, format)


def _path(source):
    if isinstance(source, (str, Path)):
        return str(source)
    import os
    return f'/proc/self/fd/{source.fileno()}' if os.name == 'posix' else source.name


def _payload(payload):
    return json.dumps([(destination, _path(source), offset, size) for destination, source, offset, size in payload])


def _options(path):
    from .package_snapshot import _OPTIONS, ExportOptions
    options = _OPTIONS.get() or ExportOptions(Path(path).parent / '.flare-cache')
    return dict(cache=str(options.cache), no_weights=options.no_weights)


def read_index(path):
    import flare_heliopause
    encoded, base = (flare_heliopause.export_read_index(str(path)) if isinstance(path, (str, Path))
                     else flare_heliopause.export_read_index_fd(path.fileno()))
    return proto.ModelIndex.FromString(encoded), base


def publish(path, index, payload, *, publisher=None, shard_size=DEFAULT_SHARD_SIZE):
    import flare_heliopause
    from types import SimpleNamespace
    if Path(path).exists():
        raise FileExistsError(path)
    value = flare_heliopause.export_publish(str(path), index.SerializeToString(deterministic=True),
        _payload(payload), None if publisher is None else json.dumps(publisher.native_request),
        json.dumps(_options(path)), shard_size)
    return SimpleNamespace(**json.loads(value))


def write_sources(path, index, payload):
    import flare_heliopause
    flare_heliopause.export_write_sources(str(path), index.SerializeToString(deterministic=True), _payload(payload))


def read_sources(path):
    import flare_heliopause
    record = json.loads(flare_heliopause.export_read_sources(str(path)))
    index = proto.ModelIndex.FromString(bytes(record['index']))
    def locate(span):
        return tuple((Path(source), offset, size) for source, offset, size in
                     json.loads(flare_heliopause.export_locate_sources(str(path), span.offset, span.nbytes)))

    return index, locate


def combine(path, ranks, peer_memory, *, publisher=None, shard_size=DEFAULT_SHARD_SIZE, cluster=None):
    import flare_heliopause
    from types import SimpleNamespace
    if Path(path).exists():
        raise FileExistsError(path)
    options = _options(path)
    if cluster is not None:
        cluster = dict(world=cluster[0], export_id=cluster[1])
    value = flare_heliopause.export_combine(str(path), [str(rank) for rank in ranks], json.dumps(peer_memory),
        None if publisher is None else json.dumps(publisher.native_request), shard_size,
        None if cluster is None else json.dumps(cluster), options['cache'], options['no_weights'])
    return SimpleNamespace(**json.loads(value))


class _PayloadReader:
    def __init__(self, files, filename):
        self.files, self.filename, self.position = files, filename, 0

    def read_at(self, size, offset):
        if size < 0 or offset < 0:
            raise ValueError('invalid logical read')
        return self.files.read_at(self.filename, offset, size)

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.position
        elif whence == 2:
            import flare_heliopause
            offset += self.files.length(self.filename)
        elif whence:
            raise ValueError('invalid seek')
        if offset < 0:
            raise ValueError('negative seek')
        self.position = offset
        return offset

    def read(self, size=-1):
        if size < 0:
            import flare_heliopause
            size = max(0, self.files.length(self.filename) - self.position)
        data = self.read_at(size, self.position)
        self.position += len(data)
        return data


class PackageFiles:
    def __init__(self, path, primary, base, length):
        import flare_heliopause
        self.files, self.readers = flare_heliopause.NativePackageFiles(str(path)), {}

    def get(self, filename):
        import flare_heliopause
        reader = self.readers.setdefault(filename, _PayloadReader(self.files, filename))
        return reader, 0, self.files.length(filename)

    def close(self):
        self.files.close()
        self.readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
