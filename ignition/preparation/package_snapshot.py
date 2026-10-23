"""Publishing options and adapters for native immutable runtime snapshots."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import json

from .package_file import ALIGNMENT, DEFAULT_SHARD_SIZE, MAX_HEADER_BYTES, MAX_INDEX_BYTES

MAX_BLOBS = 65536
MAX_PUBLISHED_BLOB_BYTES = 64_000_000_000
BLOCK = 8 << 20
_OPTIONS = ContextVar('package_snapshot_options', default=None)


@dataclass(frozen=True)
class ExportOptions:
    cache: Path
    no_weights: bool = False


@contextmanager
def export_options(cache, *, no_weights=False):
    token = _OPTIONS.set(ExportOptions(Path(cache), no_weights))
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def read_container(source):
    import flare_heliopause
    record = json.loads(flare_heliopause.export_read_container(str(source)) if isinstance(source, (str, Path))
                        else flare_heliopause.export_read_container_fd(source.fileno()))
    return record['header'], record['base'], bytes(record['envelope']), record['layout']



def validate_layout(layout):
    import flare_heliopause
    flare_heliopause.export_validate_layout(json.dumps(layout))


def write_primary(path, envelope, layout):
    import flare_heliopause
    flare_heliopause.export_write_primary(str(path), envelope, json.dumps(layout))


def convert(source, destination, *, profiles, cache=None, shard_size=DEFAULT_SHARD_SIZE):
    import flare_heliopause
    from types import SimpleNamespace
    if Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_convert(str(source), str(destination), json.dumps(profiles),
                                          None if cache is None else str(cache), shard_size)
    return SimpleNamespace(**json.loads(value))
