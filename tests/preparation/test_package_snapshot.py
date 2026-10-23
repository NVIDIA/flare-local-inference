"""Blob snapshots preserve signed packages and reuse immutable payload files."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from helios.tests.signing import publisher
from ignition.preparation.package_file import PackageFiles, read_index
from ignition.preparation.package_snapshot import (convert, export_options, read_container)
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


def legacy(path, index, payload, *, publisher, shard_size=0):
    from helios.tests.signing import publish_legacy
    return publish_legacy(path, index, payload, publisher=publisher, shard_size=shard_size)


def assert_native(native, path):
    result = fixtures.read(native, path, weights=True)
    assert result.returncode == 0, result.stderr


def test_snapshot_primary_is_metadata_only_and_native_reads_payloads(native, tmp_path):
    path, data = fixtures.fixture(tmp_path)
    with path.open('rb') as source:
        header, base, _, layout = read_container(source)
        assert header['flare.payload']['shape'] == [0]
        assert header['__metadata__']['format'] == 'flare-snapshot'
        index, _ = read_index(path)
        with PackageFiles(path, source, base, 0) as files:
            span = index.ranks[0].resources['weights']
            source, base, _ = files.get(span.filename)
            assert source.read_at(span.nbytes, base + span.offset) == data['weights']
    for parts in layout['files'].values():
        for _, name in parts:
            if name:
                blob = path.parent / name
                assert hashlib.sha256(blob.read_bytes()).hexdigest() == name[:-12]
                assert blob.stat().st_size <= 64 << 30
                assert not blob.stat().st_mode & 0o222
    assert_native(native, path)


def test_snapshot_companion_links_support_native_loading_and_conversion(native, tmp_path):
    path, _ = fixtures.fixture(tmp_path)
    with path.open('rb') as source:
        layout = read_container(source)[3]
    storage = tmp_path / 'payload-storage'
    storage.mkdir()
    names = {name for parts in layout['files'].values() for _, name in parts if name}
    assert names
    for name in names:
        companion = path.parent / name
        target = storage / name
        companion.rename(target)
        companion.symlink_to(target)
    assert_native(native, path)
    destination = tmp_path / 'converted' / 'model.safetensors'
    convert(path, destination, profiles=[profile()], cache=tmp_path / 'converted-cache')
    assert_native(native, destination)


@pytest.mark.parametrize('sharded', [False, True])
def test_converter_preserves_original_signature_and_reuses_blobs(native, tmp_path, monkeypatch, sharded):
    def publish(*args, **kwargs):
        return legacy(*args, **kwargs, shard_size=16384 if sharded else 0)
    monkeypatch.setattr(fixtures, 'publish', publish)
    source, _ = fixtures.fixture(tmp_path, size=100)
    with source.open('rb') as file:
        original = read_container(file)[2]
    cache = tmp_path / 'cache'
    first = tmp_path / 'first' / 'model.safetensors'
    convert(source, first, profiles=[profile()], cache=cache)
    second = tmp_path / 'second' / 'model.safetensors'
    report = convert(first, second, profiles=[profile()], cache=cache)
    assert report.written == 0
    assert report.reused > 0
    with second.open('rb') as file:
        _, _, signed, layout = read_container(file)
        assert signed == original
    for parts in layout['files'].values():
        for _, name in parts:
            if name:
                assert os.path.samefile(first.parent / name, second.parent / name)
                assert os.path.samefile(cache / 'blobs' / name, second.parent / name)
    assert_native(native, first)
    assert_native(native, second)


def test_export_no_weights_requires_existing_exact_payload(native, tmp_path):
    first = tmp_path / 'first'
    first.mkdir()
    cache = tmp_path / 'cache'
    with export_options(cache):
        source, _ = fixtures.fixture(first, size=100)
    second = tmp_path / 'second'
    second.mkdir()
    with export_options(cache, no_weights=True):
        destination, _ = fixtures.fixture(second, size=200)
    assert_native(native, destination)
    first_weights = read_index(source)[0].ranks[0].resources['weights']
    with source.open('rb') as file:
        layout = read_container(file)[3]
    weight_name = layout['files'][first_weights.filename][0][1]
    assert os.path.samefile(first / weight_name, second / weight_name)
    third = tmp_path / 'third'
    third.mkdir()
    with export_options(tmp_path / 'empty', no_weights=True), pytest.raises(ValueError, match='cached payload'):
        fixtures.fixture(third, size=100)
    assert not (third / 'model.safetensors').exists()


def test_converter_rejects_tampered_weights_and_unknown_publisher(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, 'publish', legacy)
    source, _ = fixtures.fixture(tmp_path, size=100)
    with pytest.raises(ValueError, match='unknown'):
        convert(source, tmp_path / 'unknown.safetensors', profiles=[])
    _, base = read_index(source)
    with source.open('r+b') as file:
        file.seek(base)
        file.write(b'changed')
    with pytest.raises(ValueError, match='digest mismatch'):
        convert(source, tmp_path / 'bad.safetensors', profiles=[profile()])
    assert not (tmp_path / 'bad.safetensors').exists()


def test_shard_limit_is_hard_and_primary_is_never_published_on_failure(tmp_path):
    from ignition.preparation import package_file
    from ignition.preparation import runtime_package_pb2 as proto
    with pytest.raises(ValueError, match='shard limit'):
        package_file.publish(tmp_path / 'bad.safetensors', proto.ModelIndex(version=4, ranks=[proto.BundleIndex()]),
                             [], publisher=publisher(), shard_size=0)


def test_native_logical_reads_cross_blob_boundaries_and_zero_padding(tmp_path):
    path, _ = fixtures.fixture(tmp_path, size=100)
    with path.open('rb') as primary:
        _, base, _, layout = read_container(primary)
        with PackageFiles(path, primary, base, 0) as files:
            for name, parts in layout['files'].items():
                expected = b''.join((path.parent / filename).read_bytes()[4096:]
                                    if filename else bytes(size) for size, filename in parts)
                source, start, length = files.get(name)
                assert length == len(expected)
                source.seek(start)
                assert source.read() == expected
                boundary = 0
                for size, _ in parts:
                    boundary += size
                    offset = max(0, boundary - 7)
                    source.seek(start + offset)
                    assert source.read(14) == expected[offset:offset + 14]


def test_native_export_rejects_corrupt_cached_weight_without_replacement(tmp_path):
    cache = tmp_path / 'cache'
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    with export_options(cache):
        path, _ = fixtures.fixture(first, size=100)
    span = read_index(path)[0].ranks[0].resources['weights']
    with path.open('rb') as source:
        layout = read_container(source)[3]
    name = next(filename for _, filename in layout['files'][span.filename] if filename)
    blob = cache / 'blobs' / name
    blob.chmod(0o644)
    with blob.open('r+b') as source:
        source.seek(4096)
        source.write(b'corrupt')
    contents = blob.read_bytes()
    with export_options(cache), pytest.raises(ValueError, match='cached blob'):
        fixtures.fixture(second, size=100)
    assert not (second / 'model.safetensors').exists()
    assert blob.read_bytes() == contents


def test_native_snapshot_segment_budget_rejects_metadata_before_payload_io(tmp_path):
    import struct
    from ignition.preparation.package_file import align

    source, _ = fixtures.fixture(tmp_path, size=100)
    with source.open('rb') as primary:
        _, _, envelope, _ = read_container(primary)
    filename = 'a' * 64 + '.safetensors'
    layout = json.dumps({'version': 1, 'files': {'': [[1, filename]] * 65537}}).encode()
    header = json.dumps({'__metadata__': {'format': 'flare-snapshot'},
        'flare.payload': {'dtype': 'U8', 'shape': [0], 'data_offsets': [0, 0]},
        'flare.metadata': {'dtype': 'U8', 'shape': [len(envelope)], 'data_offsets': [0, len(envelope)]},
        'flare.layout': {'dtype': 'U8', 'shape': [len(layout)],
                         'data_offsets': [len(envelope), len(envelope) + len(layout)]}},
        separators=(',', ':')).encode()
    header = header.ljust(align(8 + len(header)) - 8, b' ')
    invalid = tmp_path / 'invalid.safetensors'
    invalid.write_bytes(struct.pack('<Q', len(header)) + header + envelope + layout)
    destination = tmp_path / 'converted' / 'model.safetensors'
    with pytest.raises(ValueError, match='segment'):
        convert(invalid, destination, profiles=[profile()])
    assert not destination.exists()


@pytest.mark.parametrize('stage', ['payload', 'primary'])
def test_native_publication_failure_preserves_unrelated_files(tmp_path, stage):
    import shutil
    import sys

    tracer = shutil.which('strace')
    if tracer is None:
        pytest.skip('requires strace to inject a native link failure')
    code = ('from pathlib import Path; import sys; '
            'from tests.preparation.test_package_protection import fixture; '
            'fixture(Path(sys.argv[1]),size=100)')
    baseline, failing = tmp_path / 'baseline', tmp_path / 'failing'
    baseline.mkdir()
    failing.mkdir()
    trace = tmp_path / 'links.txt'
    completed = subprocess.run([tracer, '-e', 'trace=link,linkat', '-o', str(trace), sys.executable,
        '-c', code, str(baseline)], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    links = [line for line in trace.read_text().splitlines() if line.startswith(('link(', 'linkat('))]
    assert links and 'model.safetensors' in links[-1]
    foreign = failing / 'unrelated.safetensors'
    foreign.write_bytes(b'keep')
    occurrence = 1 if stage == 'payload' else len(links)
    operation = links[-1].split('(', 1)[0]
    assert all(line.startswith(operation + '(') for line in links)
    result = subprocess.run([tracer, '-e', f'inject={operation}:error=EIO:when={occurrence}',
        '-o', str(tmp_path / 'failed-links.txt'), sys.executable, '-c', code, str(failing)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Input/output error' in result.stderr
    assert not (failing / 'model.safetensors').exists()
    assert not list(failing.glob('.snapshot-*'))
    assert foreign.read_bytes() == b'keep'


def test_snapshot_transfers_only_selected_rank_blobs(native, tmp_path):
    path, _ = fixtures.shard_fixture(tmp_path, ranks=2, shard_size=32768)
    result = subprocess.run([str(native), str(path), 'ranges', '1'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ranges = json.loads(result.stdout)
    destination = tmp_path / 'peer'
    destination.mkdir()
    for name, (length, parts) in ranges.items():
        assert parts == [[0, length]]
        (destination / (name or path.name)).write_bytes((path.with_name(name) if name else path).read_bytes())
    result = fixtures.read(native, destination / path.name, weights=True, rank=1)
    assert result.returncode == 0, result.stderr
    assert fixtures.read(native, destination / path.name, weights=True).returncode != 0


def test_snapshot_rejects_swapped_payload_and_wrong_blob_length(native, tmp_path):
    from ignition.preparation.package_snapshot import write_primary
    path, _ = fixtures.fixture(tmp_path)
    with path.open('rb') as source:
        _, _, envelope, layout = read_container(source)
    parts = layout['files']['']
    nonempty = [part for part in parts if part[1]]
    nonempty[0][1], nonempty[-1][1] = nonempty[-1][1], nonempty[0][1]
    bad = tmp_path / 'swapped.safetensors'
    write_primary(bad, envelope, layout)
    result = fixtures.read(native, bad, weights=True)
    assert result.returncode != 0


def test_snapshot_signing_rejects_weight_corruption(tmp_path):
    from ignition.preparation.resign import resign

    signer = fixtures.development_publisher()
    source, _ = fixtures.fixture(tmp_path, signer=signer)
    with source.open('rb') as file:
        layout = read_container(file)[3]
    span = read_index(source)[0].ranks[0].resources['weights']
    weight = tmp_path / layout['files'][span.filename][0][1]
    weight.chmod(0o644)
    with weight.open('r+b') as file:
        file.seek(4096)
        file.write(b'bad')
    destination = tmp_path / 'production.safetensors'
    with pytest.raises(ValueError, match='stored-data digest mismatch'):
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer))
    assert not destination.exists()


def test_snapshot_layout_preserves_64_gib_read_limit():
    from ignition.preparation.package_snapshot import validate_layout
    validate_layout({'version': 1, 'files': {'': [[(64 << 30) - 4096, 'a' * 64 + '.safetensors']]}})


def test_packed_weights_preserve_padding_signatures_and_cache_reuse(native, tmp_path):
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.package_file import publish, align
    from ignition.preparation.resign import resign

    cache = tmp_path / 'cache'
    signer = fixtures.development_publisher()

    def export(directory, graph):
        directory.mkdir()
        index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex(), proto.BundleIndex()])
        index.cluster.world, index.cluster.export_id = 2, 'packed-weights'
        payload, cursor = [], 0
        for owner, rank in enumerate(index.ranks):
            for i in range(4):
                name = f'weight-{i}'
                data = directory / f'{owner}-{name}'
                data.write_bytes(bytes([owner + i + 1]) * (101 + i))
                rank.allocations.add(name=name, file=name)
                rank.resources[name].CopyFrom(proto.Span(offset=cursor, nbytes=data.stat().st_size))
                payload.append((cursor, data, 0, data.stat().st_size))
                cursor = align(cursor + data.stat().st_size)
            data = directory / f'graph-{owner}'
            data.write_bytes(graph)
            rank.graphs.add(name='decode', resource='graph')
            rank.resources['graph'].CopyFrom(proto.Span(offset=cursor, nbytes=len(graph)))
            payload.append((cursor, data, 0, len(graph)))
            cursor = align(cursor + len(graph))
        path = directory / 'model.safetensors'
        publish(path, index, payload, publisher=signer, shard_size=16384)
        return path

    with export_options(cache):
        source = export(tmp_path / 'first', b'graph')
    with source.open('rb') as stream:
        original_layout = read_container(stream)[3]
    # Four allocations per rank fill two bounded shards, rather than four files.
    weights = {name for logical, parts in original_layout['files'].items() if logical
               for _, name in parts if name}
    assert len(weights) == 4
    assert all((source.parent / name).stat().st_size <= 16384 for name in weights)
    development = fixtures.profiles(tmp_path / 'development.json', signer)
    result = fixtures.read(native, source, namespace='local-development', weights=True, profile=development)
    assert result.returncode == 0, result.stderr

    with export_options(cache, no_weights=True):
        repeated = export(tmp_path / 'second', b'changed graph')
    for name in weights:
        assert os.path.samefile(source.parent / name, repeated.parent / name)
    signed = tmp_path / 'signed' / 'model.safetensors'
    with export_options(cache):
        resign(source, signed, publisher=publisher(),
               development_profile=fixtures.development_profile(signer), shard_size=16384)
    assert_native(native, signed)

    repacked = tmp_path / 'repacked' / 'model.safetensors'
    convert(signed, repacked, profiles=[profile()], cache=cache, shard_size=65536)
    with signed.open('rb') as before, repacked.open('rb') as after:
        assert read_container(before)[2] == read_container(after)[2]
    assert_native(native, repacked)


def test_signing_large_resource_catalog_stays_within_snapshot_segment_budget(native, tmp_path):
    """Resource-per-blob layout would require more than 65,536 segments."""
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.package_file import publish, align
    from ignition.preparation.resign import resign

    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex(), proto.BundleIndex()])
    index.cluster.world, index.cluster.export_id = 2, 'large-resource-catalog'
    data = tmp_path / 'payload'
    data.write_bytes(b'graph')
    payload = []
    cursor = 0
    for owner, rank in enumerate(index.ranks):
        rank.allocations.add(name='weights', file='weights')
        rank.resources['weights'].CopyFrom(proto.Span(offset=cursor, nbytes=5))
        payload.append((cursor, data, 0, 5))
        cursor = align(cursor + 5)
        module = rank.modules.add(name='kernel', file='resource-0')
        for i in range(31201):
            name = f'resource-{i}'
            if i % 2:
                module.globals.add(symbol=name, file=name, nbytes=5)
            else:
                rank.graphs.add(name=name, resource=name)
            rank.resources[name].CopyFrom(proto.Span(offset=cursor, nbytes=5))
            payload.append((cursor, data, 0, 5))
            cursor = align(cursor + 5)
    signer = fixtures.development_publisher()
    source = tmp_path / 'development.safetensors'
    destination = tmp_path / 'production.safetensors'
    limit = 8 << 20
    with export_options(tmp_path / 'cache'):
        publish(source, index, payload, publisher=signer, shard_size=limit)
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer),
               shard_size=limit)
    with destination.open('rb') as file:
        header, _, _, layout = read_container(file)
    assert header['flare.payload']['shape'] == [0]
    assert sum(len(parts) for parts in layout['files'].values()) < 100
    assert all((tmp_path / name).stat().st_size <= limit
               for parts in layout['files'].values() for _, name in parts if name)
    assert_native(native, destination)
    for rank in (0, 1):
        result = subprocess.run([str(native), str(destination), 'ranges', str(rank)], capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        selected = json.loads(result.stdout)
        peer = tmp_path / f'peer-{rank}'
        peer.mkdir()
        for name in selected:
            os.link(destination.with_name(name) if name else destination,
                    peer / (name or destination.name))
        result = fixtures.read(native, peer / destination.name, weights=True, rank=rank)
        assert result.returncode == 0, result.stderr
        assert fixtures.read(native, peer / destination.name, weights=True, rank=1 - rank).returncode != 0
    converted = tmp_path / 'converted' / 'model.safetensors'
    convert(destination, converted, profiles=[profile()], cache=tmp_path / 'cache', shard_size=limit)
    assert_native(native, converted)


def test_resource_larger_than_hard_limit_does_not_publish(tmp_path):
    with pytest.raises(ValueError, match='resource exceeds'):
        fixtures.shard_fixture(tmp_path, sizes=(17000,), shard_size=12288)
    assert not (tmp_path / 'model.safetensors').exists()


def test_snapshot_promotion_reuses_weights_and_native_verifies(native, tmp_path, monkeypatch):
    from ignition.preparation.resign import resign
    signer = fixtures.development_publisher()
    monkeypatch.setattr(fixtures, 'publisher', lambda: signer)
    cache = tmp_path / 'cache'
    with export_options(cache):
        source, _ = fixtures.fixture(tmp_path)
        destination = tmp_path / 'production' / 'model.safetensors'
        destination.parent.mkdir()
        resign(source, destination, publisher=publisher(), development_profile=fixtures.development_profile(signer))
    with source.open('rb') as file:
        first = read_container(file)[3]
    with destination.open('rb') as file:
        second = read_container(file)[3]
    shared = {name for parts in first['files'].values() for _, name in parts if name} & {
              name for parts in second['files'].values() for _, name in parts if name}
    assert shared
    assert all(os.path.samefile(source.parent / name, destination.parent / name) for name in shared)
    assert_native(native, destination)


def test_convert_cli_requires_no_torch_or_private_key(tmp_path, monkeypatch, capsys):
    import sys
    from ignition.cli import main
    monkeypatch.setattr(fixtures, 'publish', legacy)
    source, _ = fixtures.fixture(tmp_path, size=100)
    keys = tmp_path / 'publisher.json'
    keys.write_text(json.dumps([profile()]))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert main(['convert', str(source), '--publisher-keys', str(keys)]) == 0
    output = capsys.readouterr().out
    assert 'Converted package:' in output and 'Payload bytes written:' in output
    converted, = (tmp_path / '.cache/flare/packages/models--model/snapshots').glob('*/model.safetensors')
    with converted.open('rb') as file:
        assert read_container(file)[0]['flare.payload']['shape'] == [0]
