"""Publisher authentication across the Python exporter and native resource reader."""

import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import struct
import subprocess

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from helios.tests.signing import publisher
from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import align, publish, read_index
from ignition.preparation.protection import CHUNK, License, Publisher, signature_message

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def legacy_container_fixtures(monkeypatch):
    """Retain coverage of published version 4/5 container bytes and signatures."""
    from helios.tests.signing import publish_legacy
    from ignition.preparation import package_file
    monkeypatch.setattr(package_file, 'publish', publish_legacy)
    monkeypatch.setattr(__import__(__name__, fromlist=['publish']), 'publish', publish_legacy)


@pytest.fixture(scope='module')
def native():
    subprocess.run(['cargo', 'build', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'runtime-package,test-tools', '--bin', 'test_package_protection'],
                   check=True, capture_output=True, text=True)
    return ROOT / 'helios/target/debug/test_package_protection'


def read(native, path, namespace='flare-tests', weights=False, profile=None, rank=None):
    args = [str(native), str(path), namespace, 'verify-weights' if weights else 'skip-weights']
    if rank is not None:
        args.append(f'rank={rank}')
    if profile is not None:
        args.append(str(profile))
    return subprocess.run(args, capture_output=True, text=True, timeout=30)


def fixture(tmp_path, *, size=CHUNK + 39, signer=None):
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
    rank = index.ranks[0]
    rank.allocations.add(name='weights', file='weights')
    module = rank.modules.add(name='kernel', file='kernel')
    module.globals.add(symbol='constant', file='constant', nbytes=13)
    rank.graphs.add(name='decode', resource='graph')
    data = {'weights': bytes(range(251)) * 31, 'kernel': (b'kernel-code-' * (size // 12 + 1))[:size],
            'constant': b'constant-data', 'graph': b'graph-records', 'tokenizer': b'tokenizer-data'}
    payload, cursor = [], 0
    for name, contents in data.items():
        source = tmp_path / name
        source.write_bytes(contents)
        span = proto.Span(offset=cursor, nbytes=len(contents))
        if name == 'tokenizer':
            index.assets.add(name=name, payload=span, sha256=hashlib.sha256(contents).hexdigest())
        else:
            rank.resources[name].CopyFrom(span)
        payload.append((cursor, source, 0, len(contents)))
        cursor = align(cursor + len(contents))
    path = tmp_path / 'model.safetensors'
    publish(path, index, payload, publisher=signer or publisher())
    return path, data


def envelope(path):
    data = path.read_bytes()
    start = 8 + struct.unpack_from('<Q', data)[0]
    header = json.loads(data[8:start])
    end = start + header['flare.payload']['shape'][0]
    metadata_start, metadata_end = header['flare.metadata']['data_offsets']
    return (proto.ProtectedIndex.FromString(data[start + metadata_start:start + metadata_end]),
            data[start:end])


def replace_envelope(path, value, payload):
    metadata = value.SerializeToString(deterministic=True)
    size = len(payload)
    header = json.dumps({'__metadata__': {'format': 'flare'},
        'flare.payload': {'dtype': 'U8', 'shape': [size], 'data_offsets': [0, size]},
        'flare.metadata': {'dtype': 'U8', 'shape': [len(metadata)],
                           'data_offsets': [size, size + len(metadata)]}}, separators=(',', ':')).encode()
    header = header.ljust(align(8 + len(header)) - 8, b' ')
    path.write_bytes(struct.pack('<Q', len(header)) + header + payload + metadata)


def resign(value, signer):
    value.signer = signer.fingerprint
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    metadata.namespace, metadata.official_source = signer.namespace, signer.official_source
    metadata.development = signer.development
    del metadata.group_hashes[:]
    metadata.group_hashes.extend(hashlib.sha256(g.body).digest() for g in value.groups[1:])
    value.groups[0].body = metadata.SerializeToString(deterministic=True)
    for group in chain(value.groups, value.ranks):
        group.signature = signer.key.sign(signature_message(value.package_id, value.signer, group.kind, group.body))


def profiles(path, signer, name='Fixture publisher'):
    path.write_text(json.dumps([{'name': name, 'namespace': signer.namespace,
        'public_key': list(signer.public_key), 'development': signer.development}]))
    return path


def test_multichunk_export_is_read_and_authenticated_by_rust(native, tmp_path):
    path, data = fixture(tmp_path)
    result = read(native, path, weights=True)
    assert result.returncode == 0, result.stderr
    index, start = read_index(path)
    expected = {span.offset: hashlib.sha256(data[name]).hexdigest()
                for name, span in index.ranks[0].resources.items() if name != 'weights'}
    expected[index.assets[0].payload.offset] = hashlib.sha256(data['tokenizer']).hexdigest()
    assert dict(json.loads(result.stdout)) == expected
    contents = path.read_bytes()
    kernel = index.ranks[0].resources['kernel']
    assert data['kernel'] not in contents
    assert contents[start:start + len(data['weights'])] == data['weights']
    assert kernel.nbytes == len(data['kernel']) + 32


def cluster_package(tmp_path, monkeypatch):
    from helios.tests import signing as model_package
    from types import SimpleNamespace

    def export(rank, path):
        index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
        bundle = index.ranks[0]
        bundle.roce.world, bundle.roce.rank = 2, rank
        bundle.allocations.add(name='weights', file='weights')
        bundle.modules.add(name='kernel', file='kernel')
        payload = []
        for offset, name in [(0, 'weights'), (4096, 'kernel')]:
            source = tmp_path / f'{name}-{rank}'
            source.write_bytes(f'{name}-{rank}'.encode())
            bundle.resources[name].CopyFrom(proto.Span(offset=offset, nbytes=source.stat().st_size))
            payload.append((offset, source, 0, source.stat().st_size))
        publish(path, index, payload)
        return SimpleNamespace(memory_inventory={'rank': rank})

    peer = tmp_path / 'peer.safetensors'
    export(1, peer)
    monkeypatch.setattr(model_package, 'export_model',
                        lambda prepared, images, storage, path, **kwargs: export(0, path))
    path = tmp_path / 'cluster.safetensors'
    def inventory(reports, **kwargs):
        assert path.is_file()
        assert reports == [{'rank': 0}, {'rank': 1}]
    monkeypatch.setattr(model_package, 'print_inventory', inventory)
    with peer.open('rb') as stream:
        model_package.export_cluster_package(None, None, None, path, peer_files=[stream], options={},
                                             export_id='cluster-test', world=2, publisher=publisher(),
                                             peer_inventories=[{'rank': 1}])
    return path


def rank_bytes(path, rank):
    """File ranges holding `rank`'s signed stored data and its executable and supporting resources."""
    value, _ = envelope(path)
    index, start = read_index(path)
    data = proto.RankProtection.FromString(value.ranks[rank].body).data
    spans = [(data.offset, data.nbytes)] + [(span.offset, span.nbytes)
                                            for name, span in index.ranks[rank].resources.items() if name != 'weights']
    return [(start + offset, nbytes) for offset, nbytes in spans]


def overwrite(path, ranges, value=None):
    """Flip one byte of each range, or zero the ranges as a copy without them reads."""
    path.chmod(path.stat().st_mode | 0o200)
    with path.open('r+b') as output:
        for offset, nbytes in ranges:
            output.seek(offset)
            contents = output.read(1 if value is None else nbytes)
            output.seek(offset)
            output.write(bytes([contents[0] ^ 1]) if value is None else bytes(nbytes))


def test_multi_host_export_signs_cluster_metadata_and_each_ranks_resources(native, tmp_path, monkeypatch):
    from helios.tests.signing import extend_cluster_legacy as extend_cluster

    path = cluster_package(tmp_path, monkeypatch)
    result = read(native, path, weights=True)
    assert result.returncode == 0, result.stderr
    index, _ = read_index(path)
    assert (index.cluster.world, index.cluster.export_id) == (2, 'cluster-test')
    assert [bundle.roce.rank for bundle in index.ranks] == [0, 1]
    expected = {bundle.resources['kernel'].offset: hashlib.sha256(f'kernel-{rank}'.encode()).hexdigest()
                for rank, bundle in enumerate(index.ranks)}
    assert dict(json.loads(result.stdout)) == expected
    assert max(b.resources['weights'].offset + b.resources['weights'].nbytes for b in index.ranks) <= min(
        b.resources['kernel'].offset for b in index.ranks)
    assert not list(tmp_path.glob('*.partial'))
    with pytest.raises(ValueError, match='unsigned rank staging'):
        extend_cluster(path, [], (2, 'different-cluster'))
    value, payload = envelope(path)
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    metadata.index.cluster.export_id = 'tampered-cluster'
    value.groups[0].body = metadata.SerializeToString(deterministic=True)
    replace_envelope(path, value, payload)
    assert read(native, path).returncode != 0


@pytest.mark.parametrize('resource', ['kernel', 'graph', 'constant', 'tokenizer'])
def test_tampering_with_execution_or_support_fails_without_weight_scan(native, tmp_path, resource):
    path, _ = fixture(tmp_path, size=41)
    index, start = read_index(path)
    span = index.assets[0].payload if resource == 'tokenizer' else index.ranks[0].resources[resource]
    with path.open('r+b') as output:
        output.seek(start + span.offset)
        value = output.read(1)
        output.seek(-1, 1)
        output.write(bytes([value[0] ^ 1]))
    result = read(native, path)
    assert result.returncode != 0 and 'content verification failed' in result.stderr


@pytest.mark.parametrize('padding', [False, True])
def test_optional_data_scan_covers_weights_and_alignment_padding(native, tmp_path, padding):
    path, data = fixture(tmp_path, size=41)
    _, start = read_index(path)
    with path.open('r+b') as output:
        output.seek(start + (len(data['weights']) if padding else 0))
        output.write(b'X')
    assert read(native, path).returncode == 0
    result = read(native, path, weights=True)
    assert result.returncode != 0 and 'stored-data content verification failed' in result.stderr


def test_a_rank_only_copy_verifies_that_rank_completely(native, tmp_path, monkeypatch):
    path = cluster_package(tmp_path, monkeypatch)
    # Rank one's host holds no byte of rank zero's.
    overwrite(path, rank_bytes(path, 0), value=0)
    result = read(native, path, weights=True, rank=1)
    assert result.returncode == 0, result.stderr
    index, _ = read_index(path)
    assert dict(json.loads(result.stdout)) == {
        index.ranks[1].resources['kernel'].offset: hashlib.sha256(b'kernel-1').hexdigest()}
    assert 'verification failed' in read(native, path, weights=True).stderr


def test_another_ranks_tampering_leaves_a_rank_verifiable(native, tmp_path, monkeypatch):
    path = cluster_package(tmp_path, monkeypatch)
    overwrite(path, rank_bytes(path, 0))
    value, payload = envelope(path)
    value.ranks[0].signature = bytes(64)
    replace_envelope(path, value, payload)
    result = read(native, path, weights=True, rank=1)
    assert result.returncode == 0, result.stderr
    assert 'signature failed for rank 0' in read(native, path, rank=0).stderr


@pytest.mark.parametrize('target', ['data', 'kernel', 'signature'])
def test_tampering_with_a_ranks_own_bytes_fails_its_verification(native, tmp_path, monkeypatch, target):
    path = cluster_package(tmp_path, monkeypatch)
    data, kernel = rank_bytes(path, 1)
    if target == 'signature':
        value, payload = envelope(path)
        value.ranks[1].signature = bytes(64)
        replace_envelope(path, value, payload)
    else:
        overwrite(path, [data if target == 'data' else kernel])
    result = read(native, path, weights=True, rank=1)
    expected = {'data': 'stored-data content verification failed', 'kernel': 'content verification failed',
                'signature': 'signature failed for rank 1'}[target]
    assert result.returncode != 0 and expected in result.stderr
    assert read(native, path, weights=True, rank=0).returncode == 0


def test_only_a_multi_host_package_verifies_one_rank(native, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    assert 'not a separately served rank' in read(native, path, rank=0).stderr


@pytest.mark.parametrize('group', range(4))
def test_every_signature_group_is_verified_before_access(native, tmp_path, group):
    path, _ = fixture(tmp_path, size=41)
    value, payload = envelope(path)
    (value.groups[group] if group < 3 else value.ranks[0]).signature = bytes(64)
    replace_envelope(path, value, payload)
    result = read(native, path)
    label = f'group {group + 1}' if group < 3 else 'rank 0'
    assert result.returncode != 0 and f'signature failed for {label}' in result.stderr


def test_the_accepted_string_is_the_key_material(native, tmp_path):
    # The exporter encrypts under the publisher namespace; any other accepted string opens nothing.
    path, _ = fixture(tmp_path, size=41)
    result = read(native, path, namespace='yes')
    assert result.returncode != 0 and 'executable decryption failed' in result.stderr
    assert read(native, path).returncode == 0


def test_invalid_license_metadata_is_rejected(native, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    value, payload = envelope(path)
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    metadata.license.name, metadata.license.link = 'apache-2.0', 'http://example.com/LICENSE'
    value.groups[0].body = metadata.SerializeToString(deterministic=True)
    signer = publisher()
    for group in chain(value.groups, value.ranks):
        group.signature = signer.key.sign(signature_message(value.package_id, value.signer, group.kind, group.body))
    replace_envelope(path, value, payload)
    result = read(native, path)
    assert result.returncode != 0 and 'invalid package license metadata' in result.stderr
    with pytest.raises(ValueError, match='https URL'):
        License('apache-2.0', 'http://example.com/LICENSE')
    with pytest.raises(ValueError, match='printable'):
        License('apache\x1b[31m-2.0')


@pytest.mark.parametrize('change', ['key', 'namespace'])
def test_resigning_ciphertext_under_different_identity_cannot_decrypt(native, tmp_path, change):
    path, _ = fixture(tmp_path, size=41)
    original = publisher()
    signer = Publisher(Ed25519PrivateKey.generate() if change == 'key' else original.key,
                       'another-publisher', 'https://huggingface.co/another-publisher/fixture')
    value, payload = envelope(path)
    resign(value, signer)
    replace_envelope(path, value, payload)
    profile = profiles(tmp_path / 'publisher.json', signer)
    result = read(native, path, namespace=signer.namespace, profile=profile)
    assert result.returncode != 0 and 'executable decryption failed' in result.stderr


def test_independently_prepared_publisher_is_supported(native, tmp_path):
    signer = Publisher(Ed25519PrivateKey.generate(), 'another-publisher',
                       'https://huggingface.co/another-publisher/fixture')
    path, _ = fixture(tmp_path, size=41, signer=signer)
    assert 'unknown package publisher' in read(native, path).stderr
    profile = profiles(tmp_path / 'publisher.json', signer)
    result = read(native, path, namespace=signer.namespace, weights=True, profile=profile)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['overlap', 'unclassified', 'reclassify', 'groups'])
def test_authenticated_but_inconsistent_index_is_rejected(native, tmp_path, change):
    path, _ = fixture(tmp_path, size=41)
    value, payload = envelope(path)
    signed = proto.RankProtection.FromString(value.ranks[0].body)
    rank = signed.bundle
    if change == 'overlap':
        rank.resources['graph'].CopyFrom(rank.resources['kernel'])
    elif change == 'unclassified':
        rank.resources['unused'].CopyFrom(rank.resources['kernel'])
    elif change == 'reclassify':
        rank.allocations[0].file = 'kernel'
    value.ranks[0].body = signed.SerializeToString(deterministic=True)
    resign(value, publisher())
    if change == 'groups':
        value.groups[1].kind = 3
    replace_envelope(path, value, payload)
    assert read(native, path).returncode != 0


def test_native_loader_rejects_unsigned_staging(native, tmp_path):
    path = tmp_path / 'stage'
    publish(path, proto.ModelIndex(version=4, ranks=[proto.BundleIndex()]), [])
    result = read(native, path)
    assert result.returncode != 0 and 'not a Flare safetensors package' in result.stderr


@pytest.fixture(scope='module')
def flare():
    subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(ROOT / 'flare/Cargo.toml')],
                   check=True, capture_output=True, text=True)
    return ROOT / 'flare/target/release/flare'


def launch(flare, path, profile, store, *args):
    # A rank-count mismatch stops this CPU fixture after trust handling and before CUDA. Offline,
    # the publisher's repository is not consulted, so identity rests on the supplied profile.
    keys = [] if profile is None else ['--publisher-keys', str(profile)]
    return subprocess.run([str(flare), 'run', str(path), *keys, '--offline',
        '--trust-store', str(store), '--gpus', '0,1', *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1'})


def approval(signer):
    return ['--dangerously-approve-key', signer.fingerprint.hex(),
            '--dangerously-accept-publisher', signer.namespace]


def test_external_key_warns_once_and_remembers_exact_pair(flare, tmp_path):
    signer = publisher()
    path, _ = fixture(tmp_path, size=41)
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    first = launch(flare, path, profile, store)
    assert 'Signing key acceptance' in first.stderr and 'Identity:   Unverified' in first.stderr
    assert 'Executable GPU code can access your process and data.' in first.stderr
    assert 'unattended first use requires' in first.stderr and not store.exists()
    accepted = launch(flare, path, profile, store, *approval(signer))
    assert 'Identity:   Unverified' in accepted.stderr
    assert 'model rank count must match distinct selected devices' in accepted.stderr and store.exists()
    remembered = launch(flare, path, profile, store)
    assert 'Signing key acceptance' not in remembered.stderr
    assert 'model rank count must match distinct selected devices' in remembered.stderr
    assert store.stat().st_mode & 0o777 == 0o600
    revoked = subprocess.run([str(flare), 'trust', 'revoke', signer.fingerprint.hex(),
                             '--trust-store', str(store)], capture_output=True, text=True)
    assert revoked.returncode == 0, revoked.stderr
    assert 'unattended first use requires' in launch(flare, path, profile, store).stderr


@pytest.mark.parametrize('mismatch', ['key', 'publisher', 'missing'])
def test_explicit_approval_never_ignores_mismatch_or_missing_value(flare, tmp_path, mismatch):
    path, _ = fixture(tmp_path, size=41)
    signer = publisher()
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    args = approval(signer)
    if mismatch == 'key':
        args[1] = '0' * 64
    elif mismatch == 'publisher':
        args[3] = 'someone-else'
    else:
        args = args[:2]
    result = launch(flare, path, profile, store, *args)
    assert result.returncode != 0 and not store.exists()
    assert 'model rank count must match distinct selected devices' not in result.stderr


def test_rotation_does_not_inherit_namespace_trust(flare, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    first = publisher()
    profile = profiles(tmp_path / 'publisher.json', first)
    store = tmp_path / 'trust.toml'
    launch(flare, path, profile, store, *approval(first))
    directory = tmp_path / 'rotation'
    directory.mkdir()
    second = Publisher(Ed25519PrivateKey.generate(), first.namespace, first.official_source)
    changed, _ = fixture(directory, size=41, signer=second)
    profiles(profile, second)
    result = launch(flare, changed, profile, store)
    assert 'Signing key acceptance' in result.stderr and 'Identity:   Unverified' in result.stderr
    assert 'unattended first use requires' in result.stderr


def test_stamped_profiles_verify_signatures_without_conferring_identity(flare, tmp_path):
    from ignition.release import stamp
    signer = publisher()
    path, _ = fixture(tmp_path, size=41)
    profile = profiles(tmp_path / 'publisher.json', signer, 'Example publisher fixture')
    executable = tmp_path / 'flare-publisher'
    stamp(flare, executable, json.loads(profile.read_text()))
    store = tmp_path / 'trust.toml'
    unknown = launch(flare, path, None, store, *approval(signer))
    assert unknown.returncode != 0 and 'publish the key in publisher.json' in unknown.stderr
    assert 'https://huggingface.co/flare-tests/fixture' in unknown.stderr and not store.exists()
    known = launch(executable, path, None, store, *approval(signer))
    assert 'Publisher:  Example publisher fixture' in known.stderr
    assert 'Identity:   Unverified' in known.stderr and 'Verified via' not in known.stderr
    assert 'model rank count must match distinct selected devices' in known.stderr
    with pytest.raises(ValueError, match='not an empty release slot'):
        stamp(executable, tmp_path / 'restamped', json.loads(profile.read_text()))


def test_license_is_disclosed_and_accepted_by_name(flare, tmp_path):
    link = 'https://huggingface.co/flare-tests/fixture/blob/main/LICENSE'
    signer = Publisher(publisher().key, 'flare-tests', 'https://huggingface.co/flare-tests/fixture',
                       license=License('apache-2.0', link))
    path, _ = fixture(tmp_path, size=41, signer=signer)
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    result = launch(flare, path, profile, store, *approval(signer))
    assert 'License:    apache-2.0' in result.stderr and link in result.stderr
    assert 'agrees to the license terms at the link above' in result.stderr
    assert 'model rank count must match distinct selected devices' in result.stderr
    # A package exported before licenses were recorded discloses none and prompts as before.
    directory = tmp_path / 'unlicensed'
    directory.mkdir()
    plain, _ = fixture(directory, size=41)
    result = launch(flare, plain, profile, directory / 'trust.toml', *approval(signer))
    assert 'License:' not in result.stderr and 'license terms' not in result.stderr
    assert 'model rank count must match distinct selected devices' in result.stderr


def test_remembered_acceptance_keeps_the_key_and_needs_no_profile(flare, tmp_path):
    import tomllib
    signer = publisher()
    path, _ = fixture(tmp_path, size=41)
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    accepted = launch(flare, path, profile, store, *approval(signer))
    assert 'model rank count must match distinct selected devices' in accepted.stderr
    [entry] = tomllib.loads(store.read_text())['trusted']
    assert entry == {'fingerprint': signer.fingerprint.hex(), 'namespace': 'flare-tests',
                     'public_key': signer.public_key.hex(), 'name': 'Fixture publisher', 'identity': 'profile'}
    remembered = launch(flare, path, None, store)
    assert 'Signing key acceptance' not in remembered.stderr
    assert 'model rank count must match distinct selected devices' in remembered.stderr


def test_trust_entries_without_a_key_still_apply_and_gain_one(flare, tmp_path):
    import tomllib
    signer = publisher()
    path, _ = fixture(tmp_path, size=41)
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    store.write_text(f'[[trusted]]\nfingerprint = "{signer.fingerprint.hex()}"\nnamespace = "flare-tests"\n')
    result = launch(flare, path, profile, store)
    assert 'Signing key acceptance' not in result.stderr
    assert 'model rank count must match distinct selected devices' in result.stderr
    [entry] = tomllib.loads(store.read_text())['trusted']
    assert entry['public_key'] == signer.public_key.hex() and entry['identity'] == 'profile'
    tampered = store.read_text().replace(signer.public_key.hex(), '11' * 32)
    store.write_text(tampered)
    result = launch(flare, path, profile, store)
    assert result.returncode != 0 and 'does not match its fingerprint' in result.stderr


def test_invalid_signature_cannot_record_trust_even_with_approval(flare, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    signer = publisher()
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    value, payload = envelope(path)
    value.groups[0].signature = bytes(64)
    replace_envelope(path, value, payload)
    result = launch(flare, path, profile, store, *approval(signer))
    assert 'signature failed' in result.stderr and not store.exists()


def test_corrupt_trust_store_fails_closed(flare, tmp_path):
    path, _ = fixture(tmp_path, size=41)
    signer = publisher()
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    store.write_text('trusted = "everything"\n')
    result = launch(flare, path, profile, store, *approval(signer))
    assert result.returncode != 0 and 'model rank count must match distinct selected devices' not in result.stderr
    assert store.read_text() == 'trusted = "everything"\n'


def test_tensor_parallel_assembly_authenticates_all_ranks(native, tmp_path):
    from ignition.preparation.package_file import combine, write_sources
    ranks = []
    for rank in range(3):
        index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
        bundle = index.ranks[0]
        bundle.allocations.add(name='weight', file='weight')
        bundle.allocations.add(name='scale', file='scale')
        bundle.graphs.add(name='decode', resource='graph')
        payload = []
        # Stored data spans several copy blocks and leaves padding inside and between ranks.
        for name, offset, data in [('weight', 0, bytes([rank]) * 301),
                                   ('scale', 4096, bytes(range(256)) * (rank * 70000 + 9)),
                                   ('graph', 64 << 20, bytes([rank + 2]) * 41)]:
            source = tmp_path / f'{name}-{rank}'
            source.write_bytes(data)
            bundle.resources[name].CopyFrom(proto.Span(offset=offset, nbytes=len(data)))
            payload.append((offset, source, 0, len(data)))
        path = tmp_path / f'rank-{rank}.json'
        write_sources(path, index, payload)
        ranks.append(path)
    output = tmp_path / 'tp3.safetensors'
    combine(output, ranks, [], publisher=publisher())
    result = read(native, output, weights=True)
    assert result.returncode == 0, result.stderr
    assert {digest for _, digest in json.loads(result.stdout)} == {
        hashlib.sha256(bytes([rank + 2]) * 41).hexdigest() for rank in range(3)}
    index, start = read_index(output)
    from ignition.preparation.package_file import PackageFiles
    with output.open('rb') as package, PackageFiles(output, package, start, 0) as files:
        for rank, bundle in enumerate(index.ranks):
            span = bundle.resources['scale']
            source, base, _ = files.get(span.filename)
            source.seek(base + span.offset)
            assert source.read(span.nbytes) == bytes(range(256)) * (rank * 70000 + 9)
    # A stored-data byte flipped after publication fails the rank's weight scan.
    value, _ = envelope(output)
    span = index.ranks[1].resources['scale']
    from ignition.preparation.package_snapshot import read_container
    with output.open('rb') as package:
        layout = read_container(package)[3]
    position = span.offset + span.nbytes - 1
    for length, filename in layout['files'][span.filename]:
        if position < length:
            overwrite(output.parent / filename, [(4096 + position, 1)])
            break
        position -= length
    else:
        pytest.fail('stored resource is absent from the snapshot layout')
    assert read(native, output, weights=True).returncode != 0


def test_signed_assembly_stores_a_table_read_from_several_checkpoint_shards(native, tmp_path):
    from ignition.preparation.package_file import combine, write_sources
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
    bundle = index.ranks[0]
    bundle.allocations.add(name='weight', file='weight')
    bundle.graphs.add(name='decode', resource='graph')
    weight, scales, graph = tmp_path / 'weight', tmp_path / 'scales', tmp_path / 'graph'
    weight.write_bytes(b'weight-data')
    scales.write_bytes(b'table-scales')
    graph.write_bytes(b'graph-records')
    bundle.resources['weight'].CopyFrom(proto.Span(offset=0, nbytes=11))
    # The table's rows lie in three shards, each behind its own header, and span several copy blocks.
    parts = [bytes(range(251)) * (part * 70000 + 3) for part in range(3)]
    table = index.engram.tables.add(layer=1)
    table.weights.CopyFrom(proto.Span(offset=4096, nbytes=sum(map(len, parts))))
    payload, cursor = [(0, weight, 0, 11)], 4096
    for number, part in enumerate(parts):
        shard = tmp_path / f'shard-{number}'
        shard.write_bytes(b'header' * number + part)
        payload.append((cursor, shard, 6 * number, len(part)))
        cursor += len(part)
    cursor = align(cursor)
    table.scales.CopyFrom(proto.Span(offset=cursor, nbytes=12))
    payload.append((cursor, scales, 0, 12))
    cursor = align(cursor + 12)
    bundle.resources['graph'].CopyFrom(proto.Span(offset=cursor, nbytes=13))
    payload.append((cursor, graph, 0, 13))
    recorded = tmp_path / 'rank-0.json'
    write_sources(recorded, index, payload)
    output = tmp_path / 'model.safetensors'
    combine(output, [recorded], [], publisher=publisher())
    result = read(native, output, weights=True)
    assert result.returncode == 0, result.stderr
    combined, start = read_index(output)
    span = combined.engram.tables[0].weights
    from ignition.preparation.package_file import PackageFiles
    with output.open('rb') as package, PackageFiles(output, package, start, 0) as files:
        source, base, _ = files.get(span.filename)
        source.seek(base + span.offset)
        assert source.read(span.nbytes) == b''.join(parts)
    # Executables are one range each.
    split = tmp_path / 'split.json'
    write_sources(split, index, [*payload[:-1], (cursor, graph, 0, 6), (cursor + 6, graph, 6, 7)])
    with pytest.raises(ValueError, match='protected resources require one contiguous source'):
        combine(tmp_path / 'split.safetensors', [split], [], publisher=publisher())


def test_native_record_reader_selects_independently_and_rejects_malformed_records(native, tmp_path):
    node = proto.Node(empty=proto.Empty()).SerializeToString()
    valid = struct.pack('<I', len(node)) + node
    data = {'selected': valid, 'oversize': struct.pack('<I', (1 << 20) + 1),
            'truncated': struct.pack('<I', 100) + b'x', 'trailing': valid + b'x'}
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex(architecture=[12, 0], context=proto.Context())])
    bundle = index.ranks[0]
    payload = []
    for ordinal, (name, contents) in enumerate(data.items()):
        path = tmp_path / name
        path.write_bytes(contents)
        bundle.resources[name].CopyFrom(proto.Span(offset=ordinal * 4096, nbytes=len(contents)))
        bundle.graphs.add(name=name, resource=name, node_count=1)
        payload.append((ordinal * 4096, path, 0, len(contents)))
    output = tmp_path / 'records.safetensors'
    publish(output, index, payload, publisher=publisher())
    for name in data:
        result = subprocess.run([str(native), str(output), 'records', name], capture_output=True, text=True)
        assert (result.returncode == 0) == (name == 'selected'), result.stderr


@pytest.mark.parametrize('field,value', [
    (('flare.payload', 'dtype'), 'F32'),
    (('flare.payload', 'shape'), [2, 2]),
    (('flare.payload', 'data_offsets'), [1, 5]),
    (('flare.metadata', 'data_offsets'), [3, 2**64 - 1]),
    (('flare.metadata', 'shape'), [(128 << 20) + 1]),
])
def test_native_header_validation_rejects_bad_shapes_and_extents(native, tmp_path, field, value):
    path, _ = fixture(tmp_path, size=41)
    contents = path.read_bytes()
    start = 8 + struct.unpack_from('<Q', contents)[0]
    header = json.loads(contents[8:start])
    header[field[0]][field[1]] = value
    encoded = json.dumps(header, separators=(',', ':')).encode().ljust(start - 8, b' ')
    path.write_bytes(contents[:8] + encoded + contents[start:])
    assert read(native, path).returncode != 0


def development_publisher():
    return Publisher(Ed25519PrivateKey.generate(), 'local-development', 'local-development', development=True)


def development_profile(signer):
    return {'namespace': signer.namespace, 'development': signer.development,
            'public_key': list(signer.public_key)}


def test_development_profile_requires_acceptance_and_remembers_the_key(flare, tmp_path):
    signer = development_publisher()
    path, _ = fixture(tmp_path, size=41, signer=signer)
    profile = profiles(tmp_path / 'publisher.json', signer)
    store = tmp_path / 'trust.toml'
    first = launch(flare, path, profile, store)
    assert 'Local development key' in first.stderr
    assert 'unattended first use requires' in first.stderr and not store.exists()
    accepted = launch(flare, path, profile, store, *approval(signer))
    assert 'model rank count must match distinct selected devices' in accepted.stderr
    assert store.exists()
    remembered = launch(flare, path, None, store)
    assert 'Signing key acceptance' not in remembered.stderr
    assert 'model rank count must match distinct selected devices' in remembered.stderr


def test_development_profile_must_match_the_signed_package(flare, tmp_path):
    signer = development_publisher()
    path, _ = fixture(tmp_path, size=41, signer=signer)
    profile = profiles(tmp_path / 'publisher.json', development_publisher())
    store = tmp_path / 'trust.toml'
    wrong_key = launch(flare, path, profile, store, *approval(signer))
    assert 'unknown package publisher' in wrong_key.stderr and not store.exists()
    profiles(profile, signer)
    value, payload = envelope(path)
    value.groups[0].signature = bytes(64)
    replace_envelope(path, value, payload)
    invalid = launch(flare, path, profile, store, *approval(signer))
    assert 'signature failed' in invalid.stderr and not store.exists()


def test_promote_development_package_preserves_resources(native, tmp_path):
    from ignition.preparation.resign import resign as promote

    original = development_publisher()
    source, data = fixture(tmp_path, signer=original)
    before = source.read_bytes()
    output = tmp_path / 'production.safetensors'
    target = publisher()
    promote(source, output, publisher=target, development_profile=development_profile(original))
    result = read(native, output, weights=True)
    assert result.returncode == 0, result.stderr
    index, start = read_index(output)
    contents = output.read_bytes()
    from ignition.preparation.package_file import PackageFiles
    with output.open('rb') as primary, PackageFiles(output, primary, start, 0) as files:
        for name in ('weights', 'constant'):
            span = index.ranks[0].resources[name]
            reader, base, _ = files.get(span.filename)
            reader.seek(base + span.offset)
            assert reader.read(span.nbytes) == data[name]
    assert dict(json.loads(result.stdout)) == {
        index.ranks[0].resources[name].offset: hashlib.sha256(data[name]).hexdigest()
        for name in ('kernel', 'graph', 'constant')} | {
        index.assets[0].payload.offset: hashlib.sha256(data['tokenizer']).hexdigest()}
    old, _ = envelope(source)
    new, _ = envelope(output)
    assert new.package_id != old.package_id and new.signer == target.fingerprint
    assert not proto.PublisherMetadata.FromString(new.groups[0].body).development
    assert source.read_bytes() == before
    assert not list(tmp_path.glob('.flare-resign-*'))
    with pytest.raises(FileExistsError):
        promote(source, output, publisher=target, development_profile=development_profile(original))
    assert output.read_bytes() == contents


@pytest.mark.parametrize('corruption', ['weights', 'kernel', 'graph', 'constant', 'tokenizer', 'padding', 'signature', 'profile'])
def test_promotion_rejects_tampering_without_publishing(tmp_path, corruption):
    from ignition.preparation.resign import resign as promote

    original = development_publisher()
    source, data = fixture(tmp_path, size=41, signer=original)
    index, start = read_index(source)
    profile = development_profile(original)
    if corruption == 'profile':
        profile = development_profile(development_publisher())
    elif corruption == 'signature':
        value, payload = envelope(source)
        value.ranks[0].signature = bytes(64)
        replace_envelope(source, value, payload)
    else:
        offset = (len(data['weights']) if corruption == 'padding' else
                  index.assets[0].payload.offset if corruption == 'tokenizer' else
                  index.ranks[0].resources[corruption].offset)
        overwrite(source, [(start + offset, 1)])
    output = tmp_path / 'production.safetensors'
    with pytest.raises(ValueError):
        promote(source, output, publisher=publisher(), development_profile=profile)
    assert not output.exists()
    assert not list(tmp_path.glob('.flare-resign-*'))
    assert not list(tmp_path.glob('.production.safetensors-*'))


def test_sign_cli_with_encrypted_key_and_tensor_parallel_package(native, tmp_path, monkeypatch):
    from ignition import signing
    from ignition.cli import main
    from ignition.preparation.package_file import combine, write_sources

    original = development_publisher()
    staged = []
    for rank in range(2):
        index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
        bundle = index.ranks[0]
        bundle.allocations.add(name='weight', file='weight')
        bundle.graphs.add(name='decode', resource='graph')
        payload = []
        for name, offset, data in [('weight', 0, bytes([rank]) * 301),
                                   ('graph', 4096, bytes([rank + 2]) * 41)]:
            path = tmp_path / f'{name}-{rank}'
            path.write_bytes(data)
            bundle.resources[name].CopyFrom(proto.Span(offset=offset, nbytes=len(data)))
            payload.append((offset, path, 0, len(data)))
        path = tmp_path / f'rank-{rank}.json'
        write_sources(path, index, payload)
        staged.append(path)
    source = tmp_path / 'development.safetensors'
    combine(source, staged, [], publisher=original)
    profiles(tmp_path / 'development-publisher.json', original)
    password = b'encrypted-fixture-key-password'
    signing.generate(tmp_path / 'keys', 'Fixture publisher', 'example', password)
    monkeypatch.setattr(signing, 'passphrase', lambda prompt: password)
    config = tmp_path / 'config.toml'
    config.write_text('[publishing]\nnamespace = "example"\n'
                      'official_source = "https://huggingface.co/example/default"\n'
                      '[paths]\npackage_cache = ' + json.dumps(str(tmp_path / 'cache')) + '\n')
    output = tmp_path / 'production.safetensors'
    args = ['sign', str(source), '--output', str(output), '--config', str(config),
            '--signing-key', str(tmp_path / 'keys/signing-key.pem'),
            '--official-source', 'https://huggingface.co/example/override']
    assert main(args) == 0
    value, _ = envelope(output)
    metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
    assert metadata.official_source == 'https://huggingface.co/example/override'
    assert metadata.ranks == 2
    result = read(native, output, namespace='example', weights=True, profile=tmp_path / 'keys/publisher.json')
    assert result.returncode == 0, result.stderr
    assert {digest for _, digest in json.loads(result.stdout)} == {
        hashlib.sha256(bytes([rank + 2]) * 41).hexdigest() for rank in range(2)}
    assert main(args) != 0


def test_promotion_requires_development_source_and_production_target(tmp_path):
    from ignition.preparation.resign import resign as promote

    source, _ = fixture(tmp_path, size=41)
    output = tmp_path / 'production.safetensors'
    with pytest.raises(ValueError, match='local-development'):
        promote(source, output, publisher=publisher(), development_profile=development_profile(publisher()))
    with pytest.raises(ValueError, match='production publisher'):
        promote(source, output, publisher=development_publisher(), development_profile=development_profile(publisher()))
    assert not output.exists()


def shard_fixture(tmp_path, *, sizes=(6000, 8000, 17000, 100), shard_size=12288, signer=None,
                  name='model', ranks=1):
    index = proto.ModelIndex(version=4)
    if ranks > 1:
        index.cluster.world, index.cluster.export_id = ranks, 'sharded-test'
    payload, cursor, expected = [], 0, {}
    for rank in range(ranks):
        bundle = index.ranks.add()
        for i, size in enumerate(sizes):
            key = f'weight-{i}'
            contents = bytes([rank * 16 + i + 1]) * size
            source = tmp_path / f'{name}-source-{rank}-{i}'
            source.write_bytes(contents)
            bundle.allocations.add(name=key, file=key)
            bundle.resources[key].CopyFrom(proto.Span(offset=cursor, nbytes=size))
            payload.append((cursor, source, 0, size))
            expected[rank, key] = contents
            cursor = align(cursor + size)
        source = tmp_path / f'{name}-code-{rank}'
        source.write_bytes(b'kernel-code')
        bundle.modules.add(name='kernel', file='kernel')
        bundle.resources['kernel'].CopyFrom(proto.Span(offset=cursor, nbytes=11))
        payload.append((cursor, source, 0, 11))
        cursor = align(cursor + 11)
    path = tmp_path / f'{name}.safetensors'
    publish(path, index, payload, publisher=signer or publisher(), shard_size=shard_size)
    return path, expected


def assert_shard_contents(path, expected):
    from ignition.preparation.package_file import PackageFiles
    index, base = read_index(path)
    _, primary = envelope(path)
    with path.open('rb') as source, PackageFiles(path, source, base, len(primary)) as files:
        for (rank, key), contents in expected.items():
            span = index.ranks[rank].resources[key]
            file, start, _ = files.get(span.filename)
            file.seek(start + span.offset)
            assert file.read(span.nbytes) == contents
    return index


def test_shards_hold_whole_resources_and_keep_code_primary(native, tmp_path):
    path, expected = shard_fixture(tmp_path)
    index = assert_shard_contents(path, expected)
    assert index.version == 5
    assert not index.ranks[0].resources['kernel'].filename
    names = sorted({span.filename for span in index.ranks[0].resources.values()} - {''})
    assert names == [f'model-data-{i:02}-of-04.safetensors' for i in range(1, 5)]
    assert [path.with_name(name).stat().st_size for name in names] == [10096, 12096, 21096, 4196]
    result = read(native, path, weights=True)
    assert result.returncode == 0, result.stderr
    value, _ = envelope(path)
    data = proto.RankProtection.FromString(value.ranks[0].body).data
    assert data.offset == 0 and data.nbytes == sum(part.nbytes for part in data.parts)
    assert [part.filename for part in data.parts] == names


@pytest.mark.parametrize('limit', [0, 1 << 30])
def test_small_or_explicitly_monolithic_packages_keep_legacy_encoding(native, tmp_path, limit):
    path, expected = shard_fixture(tmp_path, shard_size=limit)
    index = assert_shard_contents(path, expected)
    assert index.version == 4
    assert not list(tmp_path.glob('*-data-*.safetensors'))
    value, _ = envelope(path)
    assert not proto.RankProtection.FromString(value.ranks[0].body).data.parts
    assert read(native, path, weights=True).returncode == 0


def test_shard_boundary_and_two_digit_limit(tmp_path):
    path, _ = shard_fixture(tmp_path, sizes=(4096, 4096, 4096), shard_size=12288)
    index, _ = read_index(path)
    a, b, c = (index.ranks[0].resources[f'weight-{i}'] for i in range(3))
    assert a.filename == b.filename != c.filename
    assert path.with_name(a.filename).stat().st_size == 12288
    path, _ = shard_fixture(tmp_path, sizes=(1,) * 99, shard_size=1, name='ninety-nine')
    assert path.with_name('ninety-nine-data-99-of-99.safetensors').exists()
    with pytest.raises(ValueError, match='more than 99'):
        shard_fixture(tmp_path, sizes=(1,) * 100, shard_size=1, name='hundred')
    assert not list(tmp_path.glob('hundred*.safetensors'))


@pytest.mark.parametrize('name', ['../escape.safetensors', '/root.safetensors', r'a\b.safetensors',
                                  'C:stream.safetensors', '.hidden.safetensors', 'a..b.safetensors',
                                  'CON.safetensors', 'x\0.safetensors', 'a/b.safetensors'])
def test_signed_companion_names_cannot_select_paths(native, tmp_path, name):
    path, _ = shard_fixture(tmp_path)
    value, payload = envelope(path)
    rank = proto.RankProtection.FromString(value.ranks[0].body)
    previous = rank.bundle.resources['weight-0'].filename
    rank.bundle.resources['weight-0'].filename = name
    next(part for part in rank.data.parts if part.filename == previous).filename = name
    value.ranks[0].body = rank.SerializeToString(deterministic=True)
    resign(value, publisher())
    replace_envelope(path, value, payload)
    assert read(native, path).returncode != 0


@pytest.mark.parametrize('relative', [False, True])
def test_linked_companions_validate_targets_and_stored_data(native, tmp_path, relative):
    path, expected = shard_fixture(tmp_path)
    storage = tmp_path / 'storage'
    storage.mkdir()
    for companion in tmp_path.glob('model-data-*.safetensors'):
        target = storage / companion.name
        companion.rename(target)
        companion.symlink_to(target.relative_to(tmp_path) if relative else target)
    index = assert_shard_contents(path, expected)
    result = read(native, path, weights=True)
    assert result.returncode == 0, result.stderr
    target = storage / index.ranks[0].resources['weight-0'].filename
    overwrite(target, [(4096, 1)])
    assert read(native, path, weights=False).returncode == 0
    assert 'stored-data content verification failed' in read(native, path, weights=True).stderr
    target.write_bytes(target.read_bytes()[:-1])
    assert 'payload length' in read(native, path, weights=False).stderr
    with pytest.raises(ValueError, match='data length'):
        assert_shard_contents(path, expected)


def test_companion_handles_survive_link_retargeting(native, tmp_path):
    from ignition.preparation.package_file import PackageFiles
    path, expected = shard_fixture(tmp_path)
    index, base = read_index(path)
    _, primary = envelope(path)
    span = index.ranks[0].resources['weight-0']
    companion = path.with_name(span.filename)
    target = companion.with_suffix('.held')
    companion.rename(target)
    companion.symlink_to(target)
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'invalid replacement data')
    with path.open('rb') as source, PackageFiles(path, source, base, len(primary)) as files:
        opened, start, _ = files.get(span.filename)
        result = subprocess.run([str(native), str(path), 'retarget-companion', span.filename,
                                 str(replacement)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert companion.resolve() == replacement
        assert files.get(span.filename)[0] is opened
        opened.seek(start + span.offset)
        assert opened.read(span.nbytes) == expected[0, 'weight-0']
    assert read(native, path).returncode != 0


@pytest.mark.parametrize('target_kind', ['missing', 'loop', 'directory', 'fifo'])
def test_companion_links_reject_unreadable_or_nonregular_targets(native, tmp_path, target_kind):
    path, expected = shard_fixture(tmp_path)
    index, _ = read_index(path)
    companion = path.with_name(index.ranks[0].resources['weight-0'].filename)
    companion.unlink()
    target = tmp_path / 'target'
    if target_kind == 'loop':
        target.symlink_to(companion)
    elif target_kind == 'directory':
        target.mkdir()
    elif target_kind == 'fifo':
        os.mkfifo(target)
    companion.symlink_to(target)
    assert read(native, path).returncode != 0
    with pytest.raises((OSError, ValueError)):
        assert_shard_contents(path, expected)


@pytest.mark.parametrize('corruption', ['missing', 'truncate', 'trailing', 'header', 'weights'])
def test_companion_validation_and_optional_weight_scan(native, tmp_path, corruption):
    path, _ = shard_fixture(tmp_path)
    index, _ = read_index(path)
    companion = path.with_name(index.ranks[0].resources['weight-0'].filename)
    if corruption == 'missing':
        companion.unlink()
    elif corruption == 'truncate':
        companion.write_bytes(companion.read_bytes()[:-1])
    elif corruption == 'trailing':
        with companion.open('ab') as output:
            output.write(b'x')
    elif corruption == 'header':
        with companion.open('r+b') as output:
            output.write(struct.pack('<Q', 1 << 30))
    else:
        overwrite(companion, [(4096, 1)])
    assert read(native, path, weights=True).returncode != 0
    assert (read(native, path, weights=False).returncode == 0) == (corruption == 'weights')


@pytest.mark.parametrize('source_limit,destination_limit', [(0, 32768), (12288, 64 << 30), (12288, 65536)])
def test_resigning_repacks_verified_resources(native, tmp_path, source_limit, destination_limit):
    from ignition.preparation.resign import resign as promote
    signer = development_publisher()
    path, expected = shard_fixture(tmp_path, shard_size=source_limit, signer=signer)
    output = tmp_path / 'production.safetensors'
    promote(path, output, publisher=publisher(), development_profile=development_profile(signer),
            shard_size=destination_limit)
    assert_shard_contents(output, expected)
    result = read(native, output, weights=True)
    assert result.returncode == 0, result.stderr


def test_publishing_does_not_overwrite_existing_companions(tmp_path):
    occupied = tmp_path / 'model-data-02-of-04.safetensors'
    occupied.write_bytes(b'keep this file')
    with pytest.raises(FileExistsError):
        shard_fixture(tmp_path)
    assert occupied.read_bytes() == b'keep this file'
    assert sorted(tmp_path.glob('*.safetensors')) == [occupied]


def test_native_file_qualified_rank_ranges_verify_after_sparse_transfer(native, tmp_path):
    path, _ = shard_fixture(tmp_path, ranks=2)
    result = subprocess.run([str(native), str(path), 'ranges', '1'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ranges = json.loads(result.stdout)
    assert '' in ranges and len(ranges) > 1
    destination = tmp_path / 'peer'
    destination.mkdir()
    for name, (length, parts) in ranges.items():
        source = path.with_name(name) if name else path
        output = destination / (name or path.name)
        with source.open('rb') as stream, output.open('wb') as copy:
            copy.truncate(length)
            for offset, size in parts:
                stream.seek(offset)
                copy.seek(offset)
                copy.write(stream.read(size))
    result = read(native, destination / path.name, weights=True, rank=1)
    assert result.returncode == 0, result.stderr
    assert read(native, destination / path.name, weights=True).returncode != 0


@pytest.mark.parametrize('shard_size', [0, 4096])
def test_embedding_partitions_transfer_only_the_selected_rank(native, tmp_path, shard_size):
    """Signed table partitions survive sparse transfer without any peer table bytes."""
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex() for _ in range(4)])
    index.cluster.world, index.cluster.export_id = 4, 'partitioned-tables'
    table = index.engram.tables.add(layer=1, rows=40, request_buffer='requests')
    payload, cursor = [], 0
    expected = {}
    for rank in range(4):
        shard = table.shards.add(rank=rank, first_row=rank * 10, rows=10)
        for name, size in (('weights', 2560), ('scales', 80)):
            data = bytes([rank * 2 + (name == 'scales') + 1]) * size
            source = tmp_path / f'{rank}-{name}'
            source.write_bytes(data)
            getattr(shard, name).CopyFrom(proto.Span(offset=cursor, nbytes=size))
            payload.append((cursor, source, 0, size))
            expected[rank, name] = data
            cursor = align(cursor + size)
    path = tmp_path / 'tables.safetensors'
    publish(path, index, payload, publisher=publisher(), shard_size=shard_size)
    assert read(native, path, weights=True).returncode == 0
    exported, _ = read_index(path)
    for rank in range(4):
        result = subprocess.run([str(native), str(path), 'ranges', str(rank)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        ranges = json.loads(result.stdout)
        destination = tmp_path / f'peer-{rank}'
        destination.mkdir()
        for name, (length, parts) in ranges.items():
            source = path.with_name(name) if name else path
            output = destination / (name or path.name)
            with source.open('rb') as stream, output.open('wb') as copy:
                copy.truncate(length)
                for offset, size in parts:
                    stream.seek(offset)
                    copy.seek(offset)
                    copy.write(stream.read(size))
        result = read(native, destination / path.name, weights=True, rank=rank)
        assert result.returncode == 0, result.stderr
        assert read(native, destination / path.name, weights=True).returncode != 0
        for other, shard in enumerate(exported.engram.tables[0].shards):
            for name in ('weights', 'scales'):
                span = getattr(shard, name)
                source = destination / (span.filename or path.name)
                if not source.exists():
                    assert other != rank
                    continue
                with source.open('rb') as stream:
                    start = 8 + struct.unpack('<Q', stream.read(8))[0]
                    stream.seek(start + span.offset)
                    data = stream.read(span.nbytes)
                assert data == (expected[other, name] if other == rank else bytes(span.nbytes))


@pytest.mark.parametrize('corruption', ['protected-filename', 'destination-filename', 'part-length',
                                      'part-overlap', 'span-overflow', 'wrong-file', 'legacy-version'])
def test_signed_shard_geometry_is_validated_without_scanning_weights(native, tmp_path, corruption):
    path, _ = shard_fixture(tmp_path)
    value, payload = envelope(path)
    rank = proto.RankProtection.FromString(value.ranks[0].body)
    weights = rank.bundle.resources['weight-0']
    if corruption == 'protected-filename':
        rank.bundle.resources['kernel'].filename = weights.filename
    elif corruption == 'destination-filename':
        rank.bundle.allocations[0].data_spans.add(offset=0, nbytes=1, filename=weights.filename)
    elif corruption == 'part-length':
        rank.data.nbytes += 1
    elif corruption == 'part-overlap':
        rank.data.parts.add().CopyFrom(rank.data.parts[0])
        rank.data.nbytes += rank.data.parts[0].nbytes
    elif corruption == 'span-overflow':
        weights.offset = 2**64 - 1
    elif corruption == 'wrong-file':
        weights.filename = 'absent.safetensors'
    else:
        metadata = proto.PublisherMetadata.FromString(value.groups[0].body)
        metadata.index.version = 4
        value.groups[0].body = metadata.SerializeToString(deterministic=True)
    value.ranks[0].body = rank.SerializeToString(deterministic=True)
    resign(value, publisher())
    replace_envelope(path, value, payload)
    assert read(native, path, weights=False).returncode != 0
