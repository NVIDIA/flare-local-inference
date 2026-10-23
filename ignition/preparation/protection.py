"""Publisher identity metadata for the native package writer."""

from dataclasses import dataclass, field
from pathlib import Path

CHUNK = 4 << 20
VERSION = 4


def validate_namespace(value):
    import flare_heliopause
    flare_heliopause.export_validate_publisher(value, None)


def validate_source(namespace, source):
    import flare_heliopause
    flare_heliopause.export_validate_publisher(namespace, source)


@dataclass(frozen=True)
class License:
    name: str
    link: str = ''

    def __post_init__(self):
        import flare_heliopause
        flare_heliopause.export_validate_license(self.name, self.link)


@dataclass(frozen=True)
class Publisher:
    key: object
    namespace: str
    official_source: str
    development: bool = False
    license: License | None = None
    _native: dict | None = field(default=None, repr=False, compare=False)
    _public_key: bytes | None = field(default=None, repr=False, compare=False)
    _fingerprint: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.development:
            if self.namespace != 'local-development' or self.official_source != 'local-development':
                raise ValueError('invalid local development publisher')
        else:
            validate_source(self.namespace, self.official_source)
        if self.license is not None and not isinstance(self.license, License):
            raise ValueError('publisher license must be a License')
        if self._native is None:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            if not isinstance(self.key, Ed25519PrivateKey):
                raise ValueError('publishing requires an Ed25519 private key')

    @classmethod
    def load(cls, key_path, namespace, official_source, *, development=False):
        import json
        import flare_heliopause
        from ignition.signing import passphrase

        path = str(Path(key_path).expanduser())
        info = json.loads(flare_heliopause.export_key_info(path, None, namespace, official_source))
        password = None
        if info.get('encrypted'):
            password = passphrase(f'Passphrase for {path}: ')
            info = json.loads(flare_heliopause.export_key_info(path, list(password), namespace, official_source))
        return cls(None, namespace, official_source, development=development,
                   _native={'key_path': path, 'password': None if password is None else list(password)},
                   _public_key=bytes(info['public_key']), _fingerprint=bytes(info['fingerprint']))

    @property
    def native_request(self):
        from dataclasses import asdict
        request = dict(self._native) if self._native is not None else {'seed': list(self.key.private_bytes_raw())}
        request.update(expected_public_key=list(self.public_key), namespace=self.namespace, official_source=self.official_source,
                       development=self.development, license=None if self.license is None else asdict(self.license))
        return request

    @property
    def public_key(self):
        return self._public_key if self._public_key is not None else self.key.public_key().public_bytes_raw()

    @property
    def fingerprint(self):
        if self._fingerprint is not None:
            return self._fingerprint
        import hashlib
        return hashlib.sha256(self.public_key).digest()


def signature_message(package_id, signer, kind, body):
    import flare_heliopause
    return flare_heliopause.export_signature_message(package_id, signer, kind, body)


def resource_roles(index):
    import json
    import flare_heliopause
    from . import runtime_package_pb2 as proto
    return [(kind, owner, proto.Span(**span)) for kind, owner, span in
            json.loads(flare_heliopause.export_roles(index.SerializeToString(deterministic=True)))]
