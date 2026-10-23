"""Native publication preserves signed bytes, crypto geometry, and source coverage."""

import hashlib
import json
from pathlib import Path
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import pytest

from helios.tests.signing import publisher
from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import publish
from ignition.preparation.package_snapshot import convert, export_options

CHUNK = 4 << 20


def fields(*values):
    return b''.join(struct.pack('<Q', len(value)) + value for value in values)


def container(path):
    raw = Path(path).read_bytes()
    size, = struct.unpack_from('<Q', raw)
    return json.loads(raw[8:8 + size]), raw[8 + size:]


def snapshot(path):
    header, raw = container(path)
    start, end = header['flare.metadata']['data_offsets']
    encoded = raw[start:end]
    start, end = header['flare.layout']['data_offsets']
    layout = json.loads(raw[start:end])
    logical = {}
    for filename, segments in layout['files'].items():
        pieces = []
        for length, name in segments:
            if name:
                blob_header, blob = container(Path(path).parent / name)
                first, last = blob_header['flare.payload']['data_offsets']
                assert last - first == length
                pieces.append(blob[first:last])
            else:
                pieces.append(bytes(length))
        logical[filename] = b''.join(pieces)
    return encoded, logical


def fixture(directory):
    directory.mkdir(parents=True)
    index = proto.ModelIndex(version=4)
    payload, contents = [], {}
    cursor = 0
    for owner in range(2):
        rank = index.ranks.add()
        rank.allocations.add(name='weights', file='weights')
        module = rank.modules.add(name='kernel', file='kernel')
        module.globals.add(symbol='constant', file='constant', nbytes=13)
        for name, data in [('weights', bytes([owner + 1]) * 8193),
                           ('kernel', bytes(range(251)) * ((CHUNK + 39) // 251 + 1) if owner == 0 else b'rank-one-code'),
                           ('constant', b'constant-data')]:
            if name == 'kernel' and owner == 0:
                data = data[:CHUNK + 39]
            source = directory / f'{owner}-{name}.bin'
            source.write_bytes(data)
            rank.resources[name].CopyFrom(proto.Span(offset=cursor, nbytes=len(data)))
            payload.append((cursor, source, 0, len(data)))
            contents[owner, name] = data
            cursor = (cursor + len(data) + 4095) & ~4095
    data = b'tokenizer-data'
    source = directory / 'tokenizer.bin'
    source.write_bytes(data)
    index.assets.add(name='tokenizer', payload=proto.Span(offset=cursor, nbytes=len(data)),
                     sha256=hashlib.sha256(data).hexdigest())
    payload.append((cursor, source, 0, len(data)))
    return index, payload, contents


def test_native_signatures_and_multichunk_ciphertext_match_independent_crypto(tmp_path):
    index, payload, contents = fixture(tmp_path / 'inputs')
    signer = publisher()
    output = tmp_path / 'model.safetensors'
    publish(output, index, payload, publisher=signer)
    encoded, logical = snapshot(output)
    envelope = proto.ProtectedIndex.FromString(encoded)
    assert envelope.version == 4 and len(envelope.package_id) == 32
    assert envelope.signer == hashlib.sha256(signer.public_key).digest()
    for kind, group in [*enumerate(envelope.groups, 1), *((4, group) for group in envelope.ranks)]:
        assert group.kind == kind
        message = fields(b'flare-package-signature-v4', struct.pack('<I', 4), envelope.package_id,
                         envelope.signer, struct.pack('<I', kind), group.body)
        signer.key.public_key().verify(group.signature, message)
    shared = proto.PublisherMetadata.FromString(envelope.groups[0].body)
    assert list(shared.group_hashes) == [hashlib.sha256(group.body).digest() for group in envelope.groups[1:]]
    assert not shared.index.ranks and shared.ranks == 2
    stored = [proto.StoredData.FromString(envelope.groups[2].body)]
    for owner, group in enumerate(envelope.ranks):
        rank = proto.RankProtection.FromString(group.body)
        assert rank.rank == owner
        stored.append(rank.data)
        weights = rank.bundle.resources['weights']
        assert logical[weights.filename][weights.offset:weights.offset + weights.nbytes] == contents[owner, 'weights']
        for name in ['kernel', 'constant']:
            span = rank.bundle.resources[name]
            catalog = rank.executables if name == 'kernel' else rank.support
            resource = next(entry for entry in catalog.resources if entry.offset == span.offset)
            ciphertext = logical[span.filename][span.offset:span.offset + span.nbytes]
            if name == 'constant':
                assert ciphertext == contents[owner, name]
                assert list(resource.sha256) == [hashlib.sha256(ciphertext).digest()]
                continue
            key = HKDF(algorithm=hashes.SHA256(), length=32, salt=envelope.package_id,
                       info=fields(b'flare-executable-v3', struct.pack('<Q', span.offset), signer.namespace.encode())).derive(signer.public_key)
            cipher = AESGCM(key)
            plaintext = bytearray()
            for counter, offset in enumerate(range(0, len(ciphertext), CHUNK + 16)):
                chunk = ciphertext[offset:offset + CHUNK + 16]
                assert hashlib.sha256(chunk).digest() == resource.sha256[counter]
                aad = fields(b'flare-executable-chunk-v3', envelope.package_id,
                             struct.pack('<Q', span.offset), struct.pack('<Q', resource.plain_bytes),
                             struct.pack('<Q', counter))
                plaintext.extend(cipher.decrypt(counter.to_bytes(12, 'big'), chunk, aad))
                with pytest.raises(InvalidTag):
                    cipher.decrypt(counter.to_bytes(12, 'big'), chunk, aad + b'changed')
            assert bytes(plaintext) == contents[owner, name]
    for data in stored:
        parts = list(data.parts) or [proto.Span(offset=data.offset, nbytes=data.nbytes)]
        raw = b''.join(logical[part.filename][part.offset:part.offset + part.nbytes] for part in parts)
        assert hashlib.sha256(raw).digest() == data.sha256


def test_native_owner_hashes_separate_shared_table_planes_and_rank_weights(tmp_path):
    index, payload, contents = fixture(tmp_path / 'inputs')
    cursor = (payload[-1][0] + payload[-1][3] + 4095) & ~4095
    table = index.engram.tables.add(layer=1)
    planes = {}
    for name, data in [('weights', b'shared-table-weights'), ('scales', b'shared-table-scales')]:
        path = tmp_path / f'table-{name}'
        path.write_bytes(data)
        getattr(table, name).CopyFrom(proto.Span(offset=cursor, nbytes=len(data)))
        payload.append((cursor, path, 0, len(data)))
        planes[name] = data
        cursor = (cursor + len(data) + 4095) & ~4095
    output = tmp_path / 'shared.safetensors'
    publish(output, index, payload, publisher=publisher())
    encoded, logical = snapshot(output)
    envelope = proto.ProtectedIndex.FromString(encoded)
    metadata = proto.PublisherMetadata.FromString(envelope.groups[0].body)
    shared = proto.StoredData.FromString(envelope.groups[2].body)
    assert shared.nbytes > 0
    for name, expected in planes.items():
        span = getattr(metadata.index.engram.tables[0], name)
        assert logical[span.filename][span.offset:span.offset + span.nbytes] == expected
        assert shared.offset <= span.offset < shared.offset + shared.nbytes
    protected = [(None, shared)] + [(owner, proto.RankProtection.FromString(group.body).data)
                                   for owner, group in enumerate(envelope.ranks)]
    for owner, data in protected:
        parts = list(data.parts) or [proto.Span(offset=data.offset, nbytes=data.nbytes)]
        raw = b''.join(logical[part.filename][part.offset:part.offset + part.nbytes] for part in parts)
        assert hashlib.sha256(raw).digest() == data.sha256
        if owner is not None:
            assert raw.startswith(contents[owner, 'weights'])


@pytest.mark.parametrize('malformed', ['unclassified', 'prefix', 'overlap', 'unordered', 'straddle',
                                       'source-overflow', 'destination-overflow', 'gap'])
def test_native_publication_rejects_malformed_source_coverage(tmp_path, malformed):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'abcdefgh')
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
    rank = index.ranks[0]
    rank.allocations.add(name='weights', file='weights')
    rank.resources['weights'].CopyFrom(proto.Span(nbytes=4))
    payload = [(0, source, 0, 4)]
    if malformed == 'unclassified':
        payload.append((4096, source, 4, 4))
    elif malformed == 'prefix':
        rank.resources['weights'].offset = 4096
        payload = [(0, source, 0, 4), (4096, source, 4, 4)]
    elif malformed == 'overlap':
        payload.append((2, source, 2, 2))
    elif malformed == 'unordered':
        rank.resources['weights'].nbytes = 8
        payload = [(4, source, 4, 4), (0, source, 0, 4)]
    elif malformed == 'straddle':
        payload = [(0, source, 0, 8)]
    elif malformed == 'source-overflow':
        payload = [(0, source, (1 << 64) - 2, 4)]
    elif malformed == 'destination-overflow':
        payload = [((1 << 64) - 2, source, 0, 4)]
    else:
        rank.resources['weights'].nbytes = 8
        payload = [(0, source, 0, 3), (4, source, 4, 4)]
    output = tmp_path / 'invalid.safetensors'
    with pytest.raises(ValueError):
        publish(output, index, payload, publisher=publisher())
    assert not output.exists()
    assert not list(tmp_path.glob('.flare-protection-*'))


def test_conversion_preserves_unknown_envelope_fields_without_resigning(tmp_path):
    index, payload, _ = fixture(tmp_path / 'inputs')
    signer = publisher()
    source = tmp_path / 'source.safetensors'
    publish(source, index, payload, publisher=signer)
    header, raw = container(source)
    first, last = header['flare.metadata']['data_offsets']
    encoded = raw[first:last] + b'\xf8\x07\x01'
    first, last = header['flare.layout']['data_offsets']
    layout = raw[first:last]
    header['flare.metadata']['shape'] = [len(encoded)]
    header['flare.metadata']['data_offsets'] = [0, len(encoded)]
    header['flare.layout']['data_offsets'] = [len(encoded), len(encoded) + len(layout)]
    header_bytes = json.dumps(header, separators=(',', ':')).encode()
    header_bytes = header_bytes.ljust(((8 + len(header_bytes) + 4095) & ~4095) - 8, b' ')
    source.write_bytes(struct.pack('<Q', len(header_bytes)) + header_bytes + encoded + layout)
    destination = tmp_path / 'converted' / 'model.safetensors'
    convert(source, destination, profiles=[dict(namespace=signer.namespace,
                                               public_key=list(signer.public_key), development=False)])
    assert snapshot(destination)[0] == encoded


def test_native_package_close_releases_pinned_files_and_invalidates_retained_readers(tmp_path):
    from ignition.preparation.package_file import PackageFiles, read_index

    index, payload, contents = fixture(tmp_path / 'inputs')
    output = tmp_path / 'model.safetensors'
    publish(output, index, payload, publisher=publisher())
    published, base = read_index(output)
    span = published.ranks[0].resources['weights']
    with output.open('rb') as primary:
        files = PackageFiles(output, primary, base, 0)
        reader, start, _ = files.get(span.filename)
        assert reader.read_at(span.nbytes, start + span.offset) == contents[0, 'weights']
        files.close()
        with pytest.raises(ValueError, match='package reader is closed'):
            reader.read_at(span.nbytes, start + span.offset)
        files.close()
    header, raw = container(output)
    first, last = header['flare.layout']['data_offsets']
    layout = json.loads(raw[first:last])
    for name in {name for parts in layout['files'].values() for _, name in parts if name}:
        target = tmp_path / name
        replacement = tmp_path / 'replacement.bin'
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
    replacement = tmp_path / 'replacement-primary.safetensors'
    replacement.write_bytes(output.read_bytes())
    replacement.replace(output)
