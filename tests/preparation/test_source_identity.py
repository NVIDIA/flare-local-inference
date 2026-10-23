"""Native source hashing preserves preparation's path and content identity."""

import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def native_identity():
    import flare_heliopause

    return flare_heliopause.export_source_identity


def reference(roots):
    digest = hashlib.sha256()
    for source in roots:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".rs", ".c", ".cu", ".cuh", ".h"):
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def test_native_identity_matches_pathlib_order_suffixes_and_root_order(tmp_path, native_identity):
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
    for name, contents in (("nested.py", b"root"), ("nested/a.py", b"child"),
                           (".hidden.cu", b"hidden"), ("UPPER.PY", b"excluded"),
                           ("data.bin", b"excluded"), ("μ.cuh", bytes(range(256)) * 1000)):
        (first / name).write_bytes(contents)
    (second / "kernel.rs").write_bytes(b"native")
    roots = [first, second]
    identity = native_identity(roots)
    assert identity == reference(roots)
    assert native_identity(roots[::-1]) == reference(roots[::-1])
    assert native_identity(roots) != native_identity(roots[::-1])
    (first / "data.bin").write_bytes(b"changed")
    assert native_identity(roots) == identity == reference(roots)
    (first / "nested.py").write_bytes(b"changed")
    assert native_identity(roots) == reference(roots)
    assert native_identity(roots) != identity


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="requires filesystem symlinks")
def test_native_identity_preserves_file_symlinks_and_ignores_directory_cycles(tmp_path, native_identity):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "target.py").write_bytes(b"target")
    try:
        (root / "linked.py").symlink_to(tmp_path / "target.py")
        (root / "cycle").symlink_to(tmp_path, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"filesystem symlink creation is unavailable: {error}")
    assert native_identity([root]) == reference([root])
