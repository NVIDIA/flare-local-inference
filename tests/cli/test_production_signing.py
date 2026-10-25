"""Production keys stay encrypted on disk and are unlocked through a terminal."""

import json
import os

from cryptography.hazmat.primitives import serialization
import pytest

from ignition import signing
from ignition.preparation.protection import Publisher


def test_encrypted_key_and_public_profile(tmp_path, monkeypatch):
    directory = tmp_path / 'keys'
    password = b'fixture-passphrase-not-a-production-secret'
    fingerprint = signing.generate(directory, 'Example publisher', 'example', password)
    path = directory / 'signing-key.pem'
    encoded = path.read_bytes()
    assert encoded.startswith(b'-----BEGIN ENCRYPTED PRIVATE KEY-----')
    with pytest.raises(TypeError):
        serialization.load_pem_private_key(encoded, None)
    with pytest.raises(ValueError):
        serialization.load_pem_private_key(encoded, b'incorrect')
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: password)
    publisher = Publisher.load(path, 'example', 'https://huggingface.co/example/model')
    assert publisher.fingerprint.hex() == fingerprint
    profile = json.loads((directory / 'publisher.json').read_text())[0]
    assert bytes(profile['public_key']) == publisher.public_key
    assert not profile['development']
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from tests.preparation.payload_fixture import publish, verify_signatures

    package = publish(tmp_path, publisher)
    public_key = Ed25519PublicKey.from_public_bytes(bytes(profile['public_key']))
    verify_signatures(package, public_key)
    with pytest.raises(FileExistsError):
        signing.generate(directory, 'Example publisher', 'example', password)
    assert path.read_bytes() == encoded
    if os.name != 'nt':
        assert directory.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
        path.chmod(0o644)
        with pytest.raises(ValueError, match='owner-only'):
            signing.load_key(path)


def test_no_passphrase_fallback_to_piped_input(monkeypatch):
    monkeypatch.setattr(signing.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(signing.getpass, 'getpass', lambda prompt: pytest.fail('must not prompt'))
    with pytest.raises(ValueError, match='interactive terminal'):
        signing.passphrase('secret')


def test_invalid_generation_leaves_no_directory(tmp_path):
    directory = tmp_path / 'keys'
    with pytest.raises(ValueError, match='at least 16'):
        signing.generate(directory, 'Example publisher', 'example', b'short')
    assert not directory.exists()


def test_terminal_echo_failure_cannot_expose_passphrase(monkeypatch):
    import warnings

    monkeypatch.setattr(signing.sys.stdin, 'isatty', lambda: True)

    def no_echo(prompt):
        warnings.warn('echo unavailable', signing.getpass.GetPassWarning)
        pytest.fail('must not read an echoing passphrase')

    monkeypatch.setattr(signing.getpass, 'getpass', no_echo)
    with pytest.raises(ValueError, match='cannot disable terminal echo'):
        signing.passphrase('secret')


def test_key_password_cli_preserves_identity_and_signed_messages(tmp_path, monkeypatch, capsys):
    from ignition.cli import main

    directory = tmp_path / 'keys'
    old, new = b'original-fixture-passphrase', b'replacement-fixture-passphrase'
    fingerprint = signing.generate(directory, 'Example publisher', 'example', old)
    path = directory / 'signing-key.pem'
    original = path.read_bytes()
    key = serialization.load_pem_private_key(original, old)
    signature = key.sign(b'previously signed package')
    profile = (directory / 'publisher.json').read_bytes()
    passwords = iter((old, new, new))
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: next(passwords))
    assert main(['key-password', '--signing-key', str(path)]) == 0
    result = path.read_bytes()
    assert result != original and result.startswith(b'-----BEGIN ENCRYPTED PRIVATE KEY-----')
    with pytest.raises(ValueError):
        serialization.load_pem_private_key(result, old)
    changed = serialization.load_pem_private_key(result, new)
    assert changed.public_key().public_bytes_raw() == key.public_key().public_bytes_raw()
    changed.public_key().verify(signature, b'previously signed package')
    assert (directory / 'publisher.json').read_bytes() == profile
    assert sorted(p.name for p in directory.iterdir()) == ['publisher.json', 'signing-key.pem']
    output = capsys.readouterr().out
    assert fingerprint in output
    assert old.decode() not in output and new.decode() not in output
    if os.name != 'nt':
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('failure', ['wrong_password', 'mismatch', 'short', 'replace', 'interrupted'])
def test_password_change_failure_preserves_original_key(tmp_path, monkeypatch, failure):
    old, new = b'original-fixture-passphrase', b'replacement-fixture-passphrase'
    directory = tmp_path / 'keys'
    signing.generate(directory, 'Example publisher', 'example', old)
    path = directory / 'signing-key.pem'
    original = path.read_bytes()
    passwords = iter((b'wrong' if failure == 'wrong_password' else old,
                      b'short' if failure == 'short' else new,
                      b'short' if failure == 'short' else b'different' if failure == 'mismatch' else new))
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: next(passwords))

    def failed_replace(*args):
        if failure == 'interrupted':
            raise KeyboardInterrupt()
        raise OSError('replacement failed')

    if failure in ('replace', 'interrupted'):
        monkeypatch.setattr(signing.os, 'replace', failed_replace)
    with pytest.raises((ValueError, OSError, KeyboardInterrupt)):
        signing.change_password(path)
    assert path.read_bytes() == original
    assert sorted(p.name for p in directory.iterdir()) == ['publisher.json', 'signing-key.pem']


def test_password_change_refuses_to_overwrite_a_replaced_key(tmp_path, monkeypatch):
    directory = tmp_path / 'keys'
    old, new = b'original-fixture-passphrase', b'replacement-fixture-passphrase'
    signing.generate(directory, 'Example publisher', 'example', old)
    path = directory / 'signing-key.pem'
    original = path.read_bytes()
    calls = 0

    def password(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return old
        if calls == 3:
            path.rename(directory / 'moved.pem')
            path.write_bytes(b'another key')
        return new

    monkeypatch.setattr(signing, 'passphrase', password)
    with pytest.raises(ValueError, match='key changed'):
        signing.change_password(path)
    assert path.read_bytes() == b'another key'
    assert (directory / 'moved.pem').read_bytes() == original
    assert not list(directory.glob('.signing-key.pem-*'))


@pytest.mark.skipif(os.name != 'posix', reason='POSIX directory synchronization')
def test_password_change_reports_replacement_when_directory_sync_fails(tmp_path, monkeypatch):
    directory = tmp_path / 'keys'
    old, new = b'original-fixture-passphrase', b'replacement-fixture-passphrase'
    signing.generate(directory, 'Example publisher', 'example', old)
    path = directory / 'signing-key.pem'
    passwords = iter((old, new, new))
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: next(passwords))
    fsync = signing.os.fsync

    def fail_directory_sync(descriptor):
        if signing.stat.S_ISDIR(signing.os.fstat(descriptor).st_mode):
            raise OSError('directory sync failed')
        return fsync(descriptor)

    monkeypatch.setattr(signing.os, 'fsync', fail_directory_sync)
    with pytest.raises(OSError, match='passphrase was changed'):
        signing.change_password(path)
    serialization.load_pem_private_key(path.read_bytes(), new)
    assert not list(directory.glob('.signing-key.pem-*'))


def test_sign_promotes_the_identity_and_keeps_the_exported_license(tmp_path, monkeypatch):
    """A signed package shows the same license terms the export recorded from the model card."""
    from dataclasses import replace
    from types import SimpleNamespace
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from ignition.cli import publishing
    from ignition.preparation.protection import License
    from tests.preparation.payload_fixture import manifest, publish, schema, verify_signatures
    from tests.preparation.test_package_protection import development_publisher

    license_link = 'https://huggingface.co/example/model/blob/abcdef/LICENSE'
    signer = replace(development_publisher(), license=License('apache-2.0', license_link))
    source = publish(tmp_path / 'exported', signer)

    password = b'fixture-passphrase-not-a-production-secret'
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: password)
    keys = tmp_path / 'keys'
    signing.generate(keys, 'Example publisher', 'example', password)
    profile_path = tmp_path / 'development-publisher.json'
    profile_path.write_text(json.dumps([{'namespace': signer.namespace, 'development': signer.development,
                                         'public_key': list(signer.public_key)}]))
    config = tmp_path / 'config.toml'
    config.write_text('')
    output = tmp_path / 'signed'
    args = SimpleNamespace(package=source, output=output, config=config,
                           signing_key=keys / 'signing-key.pem', namespace='example',
                           official_source='https://huggingface.co/example/model',
                           development_publisher=profile_path)
    publishing.sign(args)

    publisher = schema()('PackageBody').FromString(manifest(output).package_body).publisher
    assert publisher.namespace == 'example'
    assert publisher.official_source == 'https://huggingface.co/example/model'
    assert not publisher.development
    assert (publisher.license_name, publisher.license_link) == ('apache-2.0', license_link)
    profile = json.loads((keys / 'publisher.json').read_text())[0]
    verify_signatures(output, Ed25519PublicKey.from_public_bytes(bytes(profile['public_key'])))
