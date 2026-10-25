"""Payload cache conversion, combination, promotion, and native replay."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from helios.tests.signing import publisher
from tests.preparation import test_package_protection as fixtures


@pytest.fixture(scope='module')
def native():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(['cargo', 'build', '--manifest-path', str(root / 'helios/Cargo.toml'),
                    '--features', 'runtime-package,test-tools', '--bin', 'test_package_protection'],
                   check=True, capture_output=True, text=True)
    return root / 'helios/target/debug/test_package_protection'


def profile():
    signer = publisher()
    return {'namespace': signer.namespace, 'development': False, 'public_key': list(signer.public_key)}


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="payload fixture generation requires explicit CUDA test opt-in")
def test_payload_cache_convert_combine_and_promote(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from ignition.preparation.package_snapshot import (
        payload_combine, payload_convert_cache, payload_form, payload_promote)

    root = Path(__file__).resolve().parents[2]
    original = tmp_path / "original"
    result = subprocess.run([
        sys.executable, str(root / "helios/tests/runtime_package_fixture.py"), str(original),
    ], cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    source = original / "package"
    assert payload_form(source) == "directory"

    subprocess.run([
        "cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools",
        "--bin", "test_runtime_package",
    ], check=True, cwd=root, timeout=180)

    def replay(artifact, name):
        import shutil
        run = tmp_path / f"run-{name}"
        shutil.copytree(artifact, run / "package")
        for fixture_file in original.iterdir():
            if fixture_file.is_file():
                shutil.copyfile(fixture_file, run / fixture_file.name)
        result = subprocess.run(
            [str(root / "helios/target/debug/test_runtime_package"), str(run)],
            cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "192 graph launches" in result.stdout

    cache = tmp_path / "cache"
    exported = tmp_path / "exported"
    entry = payload_convert_cache(source, cache, exported, profiles=[profile()])
    entry_path = Path(entry.entry_path)
    assert entry_path.is_dir() and entry.entry and entry.model
    assert not entry.entry.startswith("entry-")
    assert (entry_path / "model.safetensors").is_file()
    # The exported directory is a hardlink view of the committed cache entry.
    assert os.stat(exported / "model.safetensors").st_ino ==         os.stat(entry_path / "model.safetensors").st_ino
    companions = [path for path in entry_path.rglob("*.safetensors")
                  if path.name != "model.safetensors"]
    assert companions
    for companion in companions:
        relative = companion.relative_to(entry_path)
        assert os.stat(exported / relative).st_ino == os.stat(companion).st_ino
    assert payload_form(exported) == "directory"

    # A native conversion whose standalone export fails must not select the new
    # entry or disturb the prior default. Call the Pyfunction directly to bypass
    # the Python wrapper's FileExistsError preflight.
    import flare_heliopause
    ref = cache / "refs" / entry.model
    versions_before = sorted(path.name for path in (cache / "models" / entry.model).iterdir())
    assert ref.read_text().strip() == entry.entry
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_bytes(b"keep")
    with pytest.raises(ValueError) as error:
        flare_heliopause.export_payload_convert_cache(
            str(source), str(cache), str(occupied), json.dumps([profile()]), "directory")
    assert str(error.value) == f"export destination {occupied} already exists"
    assert (occupied / "keep").read_bytes() == b"keep"
    assert ref.read_text().strip() == entry.entry
    assert sorted(path.name for path in (cache / "models" / entry.model).iterdir()) == versions_before

    replay(exported, "exported")

    # Packed and primary-file inputs are accepted directly by the shared cache
    # import, and concurrent same-minute imports reserve distinct readable names.
    from ignition.preparation.package_snapshot import convert

    packed_source = tmp_path / "source-packed.safetensors"
    convert(source, packed_source, profiles=[profile()], cache=cache, form="packed")
    packed_entry = payload_convert_cache(packed_source, cache, profiles=[profile()])
    assert Path(packed_entry.entry_path).is_dir()
    primary_entry = payload_convert_cache(
        source / "model.safetensors", cache, profiles=[profile()])
    assert Path(primary_entry.entry_path).is_dir()
    assert packed_entry.entry != primary_entry.entry
    with pytest.raises(ValueError, match="directory"):
        payload_convert_cache(source, cache, form="packed", profiles=[profile()])
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(payload_convert_cache, source, cache, profiles=[profile()]),
            pool.submit(payload_convert_cache, packed_source, cache, profiles=[profile()]),
        ]
        concurrent = [future.result() for future in futures]
    assert len({entry.entry_path for entry in concurrent}) == 2
    assert all(not entry.entry.startswith("entry-") for entry in concurrent)
    assert all(Path(entry.entry_path).is_dir() for entry in concurrent)

    combined = tmp_path / "combined"
    payload_combine([exported], combined, publisher=publisher(), profiles=[profile()],
                    accepted=publisher().namespace, target_accepted=publisher().namespace)
    assert (combined / "model.safetensors").is_file()
    assert payload_form(combined) == "directory"
    replay(combined, "combined")

    promoted = tmp_path / "promoted"
    target = {
        "namespace": publisher().namespace,
        "official_source": publisher().official_source,
        "development": False,
    }
    payload_promote(combined, promoted, publisher=publisher(), profiles=[profile()],
                    accepted=publisher().namespace, target_accepted=publisher().namespace,
                    target_publisher=target)
    assert (promoted / "model.safetensors").is_file()
    assert payload_form(promoted) == "directory"
    replay(promoted, "promoted")


def assert_native(native, path):
    result = fixtures.read(native, path, weights=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not (os.environ.get("FLARE_TEST_PAYLOAD_PACKAGE") and os.environ.get("FLARE_TEST_SOURCE_PROFILE")),
    reason="requires an actual payload package and its source publisher profile",
)
def test_payload_sign_uses_external_source_profile_namespace(native, tmp_path):
    """The source profile namespace authenticates the source; target stays distinct."""
    import subprocess
    import sys
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    source = Path(os.environ["FLARE_TEST_PAYLOAD_PACKAGE"])
    source_profile = Path(os.environ["FLARE_TEST_SOURCE_PROFILE"])
    target_key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "target-key.pem"
    key_path.write_bytes(target_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    key_path.chmod(0o600)
    target_profile = tmp_path / "target-profile.json"
    target_profile.write_text(json.dumps([{
        "name": "Production target",
        "namespace": "production-target",
        "public_key": list(target_key.public_key().public_bytes_raw()),
        "development": False,
    }]))
    config = tmp_path / "config.toml"
    config.write_text(
        '[publishing]\nnamespace = "production-target"\n'
        'official_source = "https://huggingface.co/production-target/model"\n'
        '[paths]\npackage_cache = "' + str(tmp_path / "cache") + '"\n')
    output = tmp_path / "promoted"
    result = subprocess.run([
        sys.executable, "-m", "ignition", "sign", str(source),
        "--output", str(output), "--config", str(config),
        "--signing-key", str(key_path), "--namespace", "production-target",
        "--official-source", "https://huggingface.co/production-target/model",
        "--development-publisher", str(source_profile),
    ], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    package = output / "model.safetensors"
    assert package.is_file()
    verified = fixtures.read(native, package, namespace="production-target",
                             weights=True, profile=target_profile)
    assert verified.returncode == 0, verified.stdout + verified.stderr
