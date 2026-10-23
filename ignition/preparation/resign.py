"""Promote development packages with native authentication and signing."""

from pathlib import Path
import json

from .package_file import DEFAULT_SHARD_SIZE


def resign(source, destination, *, publisher, development_profile, progress=None,
           shard_size=DEFAULT_SHARD_SIZE):
    import flare_heliopause
    if Path(destination).exists():
        raise FileExistsError(destination)
    from .package_snapshot import _OPTIONS, ExportOptions
    options = _OPTIONS.get() or ExportOptions(Path(destination).parent / '.flare-cache')
    result = flare_heliopause.export_resign(str(source), str(destination),
                json.dumps(publisher.native_request), json.dumps(development_profile),
                json.dumps({'cache': str(options.cache), 'no_weights': options.no_weights}), shard_size)
    from types import SimpleNamespace
    return SimpleNamespace(**json.loads(result))
