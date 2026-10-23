"""Native Ed25519 key import against independent PKCS#8 encoders."""

import hashlib
import json
import shutil
import struct
import subprocess
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import publish
from ignition.preparation.package_snapshot import read_container
from ignition.preparation.protection import Publisher

PASSWORD = b'native-import-fixture-passphrase'


def _write(path, value):
    path.write_bytes(value)
    path.chmod(0o600)


def _verify_signatures(path, public):
    with path.open('rb') as source:
        envelope = proto.ProtectedIndex.FromString(read_container(source)[2])
    assert envelope.signer == hashlib.sha256(public.public_bytes_raw()).digest()
    for group in [*envelope.groups, *envelope.ranks]:
        values = (b'flare-package-signature-v4', struct.pack('<I', 4), envelope.package_id,
                  envelope.signer, struct.pack('<I', group.kind), group.body)
        message = b''.join(struct.pack('<Q', len(value)) + value for value in values)
        public.verify(group.signature, message)


def _publish_with_key(tmp_path, monkeypatch, path, key):
    from ignition import signing
    monkeypatch.setattr(signing, 'passphrase', lambda _: PASSWORD)
    publisher = Publisher.load(path, 'flare-tests', 'https://huggingface.co/flare-tests/fixture')
    assert publisher.public_key == key.public_key().public_bytes_raw()
    output = tmp_path / 'model.safetensors'
    publish(output, proto.ModelIndex(version=4, ranks=[proto.BundleIndex()]), [], publisher=publisher)
    _verify_signatures(output, key.public_key())


@pytest.mark.parametrize('encrypted', [False, True])
def test_cryptography_pkcs8_keys_sign_native_packages(tmp_path, monkeypatch, encrypted):
    key = Ed25519PrivateKey.from_private_bytes(bytes([23]) * 32)
    encryption = serialization.BestAvailableEncryption(PASSWORD) if encrypted else serialization.NoEncryption()
    path = tmp_path / 'signing-key.pem'
    _write(path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption))
    _publish_with_key(tmp_path, monkeypatch, path, key)


@pytest.mark.parametrize('cipher,prf,scrypt', [
    *[(cipher, prf, False) for cipher in ('aes-128-cbc', 'aes-192-cbc', 'aes-256-cbc')
      for prf in ('hmacWithSHA1', 'hmacWithSHA224', 'hmacWithSHA256', 'hmacWithSHA384', 'hmacWithSHA512')],
    ('des3', 'hmacWithSHA256', False),
    ('aes-128-cbc', None, True), ('aes-256-cbc', None, True),
])
def test_openssl_pbes2_keys_match_cryptography_and_native_import(tmp_path, monkeypatch, cipher, prf, scrypt):
    executable = shutil.which('openssl')
    if executable is None:
        pytest.skip('OpenSSL CLI is required for independent PBES2 algorithm fixtures')
    key = Ed25519PrivateKey.from_private_bytes(bytes([31]) * 32)
    plain, encrypted = tmp_path / 'plain.pem', tmp_path / 'encrypted.pem'
    _write(plain, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    arguments = [executable, 'pkcs8', '-topk8', '-in', str(plain), '-out', str(encrypted),
                 '-v2', cipher, '-passout', 'stdin']
    arguments += ['-scrypt'] if scrypt else ['-v2prf', prf]
    subprocess.run(arguments, input=PASSWORD + b'\n', check=True, capture_output=True)
    encrypted.chmod(0o600)
    oracle = serialization.load_pem_private_key(encrypted.read_bytes(), PASSWORD)
    assert oracle.public_key().public_bytes_raw() == key.public_key().public_bytes_raw()
    _publish_with_key(tmp_path, monkeypatch, encrypted, key)


def test_native_development_key_profile_matches_existing_tools(tmp_path):
    from ignition.development import publisher
    directory = tmp_path / 'config'
    configuration = SimpleNamespace(path=directory / 'config.toml')
    first = publisher(configuration)
    path = directory / 'development-signing-key.pem'
    encoded = path.read_bytes()
    key = serialization.load_pem_private_key(encoded, None)
    assert isinstance(key, Ed25519PrivateKey)
    assert encoded.startswith(b'-----BEGIN PRIVATE KEY-----')
    profile = json.loads((directory / 'development-publisher.json').read_text())
    assert profile == [{'name': 'Local development signer', 'namespace': 'local-development',
                        'public_key': list(key.public_key().public_bytes_raw()), 'development': True}]
    assert first.public_key == key.public_key().public_bytes_raw()
    assert publisher(configuration).fingerprint == first.fingerprint
    assert path.read_bytes() == encoded


def test_existing_cryptography_development_key_is_preserved(tmp_path):
    from ignition.development import publisher
    key = Ed25519PrivateKey.from_private_bytes(bytes([19]) * 32)
    path = tmp_path / 'development-signing-key.pem'
    encoded = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                               serialization.NoEncryption())
    _write(path, encoded)
    signer = publisher(SimpleNamespace(path=tmp_path / 'config.toml'))
    assert signer.public_key == key.public_key().public_bytes_raw()
    assert path.read_bytes() == encoded

@pytest.mark.parametrize('algorithm', ['PBE-SHA1-3DES', 'PBE-MD5-DES'])
def test_legacy_pkcs8_password_algorithms_preserve_import_support(tmp_path, monkeypatch, algorithm):
    executable = shutil.which('openssl')
    if executable is None:
        pytest.skip('OpenSSL CLI is required for independent legacy PKCS#8 fixtures')
    key = Ed25519PrivateKey.from_private_bytes(bytes([37]) * 32)
    plain, encrypted = tmp_path / 'plain.pem', tmp_path / 'encrypted.pem'
    _write(plain, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    subprocess.run([executable, 'pkcs8', '-provider', 'default', '-provider', 'legacy', '-topk8',
                    '-in', str(plain), '-out', str(encrypted), '-v1', algorithm, '-passout', 'stdin'],
                   input=PASSWORD + b'\n', check=True, capture_output=True)
    encrypted.chmod(0o600)
    oracle = serialization.load_pem_private_key(encrypted.read_bytes(), PASSWORD)
    assert oracle.public_key().public_bytes_raw() == key.public_key().public_bytes_raw()
    _publish_with_key(tmp_path, monkeypatch, encrypted, key)


def test_signing_key_rotation_during_preparation_rejects_native_publication(tmp_path):
    path = tmp_path / 'signing-key.pem'
    first = Ed25519PrivateKey.from_private_bytes(bytes([17]) * 32)
    second = Ed25519PrivateKey.from_private_bytes(bytes([29]) * 32)
    def encode(key):
        return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption())
    _write(path, encode(first))
    signer = Publisher.load(path, 'flare-tests', 'https://huggingface.co/flare-tests/fixture')
    _write(path, encode(second))
    output = tmp_path / 'model.safetensors'
    with pytest.raises(ValueError, match='signing key changed'):
        publish(output, proto.ModelIndex(version=4, ranks=[proto.BundleIndex()]), [], publisher=signer)
    assert not output.exists()
