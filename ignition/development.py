"""Per-user signing credentials managed by the native publisher."""

from pathlib import Path
import json

from ignition.preparation.protection import Publisher


def publisher(configuration):
    import flare_heliopause
    result = json.loads(flare_heliopause.export_development_publisher(str(configuration.path.parent)))
    return Publisher(None, 'local-development', 'local-development', development=True,
                     _native={'key_path': result['key_path'], 'password': None},
                     _public_key=bytes(result['public_key']), _fingerprint=bytes(result['fingerprint']))


def profile(configuration, signer):
    """Return the native development publisher's public profile."""
    publisher(configuration)
    return configuration.path.parent / 'development-publisher.json'
