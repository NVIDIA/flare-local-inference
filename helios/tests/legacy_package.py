"""Version 4/5 physical package fixtures, independent of the native exporter.

Executable encryption uses AES-GCM chunks; Ed25519 signatures bind the publisher,
resource catalogs, stored-data digests, and per-rank bundles. Stored allocations
remain in the primary payload or numbered companion files.
"""

from bisect import bisect_right
from contextlib import nullcontext
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import re
import struct
import tempfile

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ignition.preparation import runtime_package_pb2 as proto

ALIGNMENT = 4096
MAX_HEADER_BYTES = 64 << 10
MAX_INDEX_BYTES = 128 << 20
DEFAULT_SHARD_SIZE = 64 << 30
COPY_BLOCK = 16 << 20
COPY_THREADS = 16
CHUNK = 4 << 20
VERSION = 4
_ZEROS = bytes(4 << 20)


def _open_source(source):
    return nullcontext(source) if hasattr(source, 'read') else Path(source).open('rb')


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors field: {key}")
        result[key] = value
    return result


def _byte_tensor(tensor):
    if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
        raise ValueError("invalid package byte tensor")
    offsets, shape = tensor["data_offsets"], tensor["shape"]
    if (tensor["dtype"] != "U8" or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
            or offsets[1] < offsets[0] or not isinstance(shape, list) or len(shape) != 1
            or type(shape[0]) is not int or shape[0] != offsets[1] - offsets[0]):
        raise ValueError("invalid package byte tensor")
    return offsets


def validate_filename(filename):
    """Accept a portable sibling basename, never a path or device name."""
    if (not isinstance(filename, str) or len(filename) > 255
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors', filename)
            or '..' in filename
            or filename.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('invalid package companion filename')
    return filename


def container_header(length, index_size=0, format='flare-data'):
    header = {
        "__metadata__": {"format": format},
        "flare.payload": {"dtype": "U8", "shape": [length], "data_offsets": [0, length]},
    }
    if format != 'flare-data':
        header["flare.metadata"] = {"dtype": "U8", "shape": [index_size],
                                    "data_offsets": [length, length + index_size]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded = encoded.ljust(align(8 + len(encoded)) - 8, b" ")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError('package header exceeds its size limit')
    return struct.pack('<Q', len(encoded)) + encoded


def embedding_spans(index):
    """Yield each table plane with its owning rank, or None for shared planes."""
    for table in index.engram.tables:
        if table.shards:
            if table.HasField("weights") or table.HasField("scales"):
                raise ValueError("partitioned embedding tables cannot also hold shared planes")
            for shard in table.shards:
                for name in ("weights", "scales"):
                    if not shard.HasField(name):
                        raise ValueError("embedding partition lacks a payload plane")
                    yield shard.rank, getattr(shard, name)
        else:
            for name in ("weights", "scales"):
                yield None, getattr(table, name)


def _validate_spans(index, length):
    if index.version not in (4, 5) or not index.ranks:
        raise ValueError("unsupported model index")
    spans = chain(
        (span for rank in index.ranks for span in rank.resources.values()),
        (asset.payload for asset in index.assets),
        (span for _, span in embedding_spans(index)),
    )
    for span in spans:
        if span.filename:
            validate_filename(span.filename)
            if index.version != 5:
                raise ValueError('companion references require model index 5')
        elif span.offset + span.nbytes > length:
            raise ValueError("resource exceeds model package payload")
    if any(span.filename for rank in index.ranks for allocation in rank.allocations
           for span in allocation.data_spans):
        raise ValueError('allocation destination spans cannot name a file')


def _publish(path, index, payload, protection=None):
    """Sync companions before publishing the primary, without replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = {'': list(payload)} if protection is None else protection.payloads
    names = [name for name in payloads if name] + ['']
    destinations = {name: path.parent / name if name else path for name in names}
    for destination in destinations.values():
        if os.path.lexists(destination):
            raise FileExistsError(destination)
    encoded = (index.SerializeToString(deterministic=True) if protection is None
               else protection.envelope())
    if not 0 < len(encoded) <= MAX_INDEX_BYTES:
        raise ValueError(f"model index is {len(encoded)} bytes; metadata limit is {MAX_INDEX_BYTES} bytes")
    published = []
    try:
        for name in names:
            if not name and len(names) > 1:
                _sync_directory(path.parent)
            entries, length = payloads[name], 0
            for destination, _, offset, size in entries:
                if destination < length or offset < 0 or size < 0:
                    raise ValueError("package payload ranges must be ordered and disjoint")
                length = destination + size
            if not name:
                _validate_spans(index, length)
            header = container_header(length, 0 if name else len(encoded),
                                      'flare-data' if name else 'flare' if protection else 'flare-staging')
            destination = destinations[name]
            descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(header)
                    output.flush()
                    _copy_payload(output.fileno(), len(header), entries, protection, name)
                    output.truncate(len(header) + length)
                    if not name:
                        if protection is not None:
                            signed = protection.envelope(final=True)
                            if len(signed) != len(encoded):
                                raise ValueError("signed envelope size changed during publication")
                            encoded = signed
                        output.seek(len(header) + length)
                        output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                os.link(temporary, destination)
                info = os.stat(temporary)
                published.append((destination, info.st_dev, info.st_ino))
            finally:
                Path(temporary).unlink(missing_ok=True)
        _sync_directory(path.parent)
    except BaseException:
        for destination, device, inode in reversed(published):
            try:
                info = destination.lstat()
                if (info.st_dev, info.st_ino) == (device, inode):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise


def _sync_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_all(fd, data, position):
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, position)
        view, position = view[written:], position + written


def _absorb_zeros(digest, count):
    while count > 0:
        block = _ZEROS[:min(count, len(_ZEROS))]
        digest.update(block)
        count -= len(block)


def _copy_payload(fd, data_start, payload, protection, filename=''):
    """Copy in parallel across owners; hash each owner's parts in publication order."""
    from concurrent.futures import ThreadPoolExecutor

    parts = [] if protection is None else [(owner, part) for owner, ranges in protection.ranges.items()
                                           for part in ranges if part.filename == filename]
    stored_jobs = [(protection.hashes[owner], part, [], protection.source_verifiers.get(owner))
                   for owner, part in parts]
    jobs = list(stored_jobs)
    for entry in payload:
        destination, _, _, size = entry
        job = next((job for job in stored_jobs if job[1].offset <= destination
                    and destination + size <= job[1].offset + job[1].nbytes), None)
        if job is None:
            jobs.append((None, None, [entry], None))
        else:
            job[2].append(entry)

    def copy(digest, stored, entries, verifier):
        cursor = stored.offset if stored else 0
        for destination, source, offset, size in entries:
            if digest is not None:
                _absorb_zeros(digest, destination - cursor)
            with _open_source(source) as contents:
                copied = 0
                while copied < size:
                    length = min(COPY_BLOCK, size - copied)
                    block = os.pread(contents.fileno(), length, offset + copied)
                    if len(block) != length:
                        raise ValueError(f"truncated package resource: {source}")
                    if verifier is not None:
                        verifier.consume(contents, offset + copied, block)
                    _write_all(fd, block, data_start + destination + copied)
                    if digest is not None:
                        digest.update(block)
                    copied += length
            cursor = destination + size
        if digest is not None:
            _absorb_zeros(digest, stored.offset + stored.nbytes - cursor)

    if jobs:
        with ThreadPoolExecutor(min(COPY_THREADS, len(jobs))) as pool:
            for future in [pool.submit(copy, *job) for job in jobs]:
                future.result()


def _copy(source, output, nbytes):
    while nbytes:
        block = source.read(min(nbytes, 16 << 20))
        if not block:
            raise ValueError("truncated rank export")
        output.write(block)
        nbytes -= len(block)


def extend_cluster(path, peers, cluster):
    """Append ranks 1 onward to rank zero's private staging file at `path`.

    `peers` yields binary streams, each a complete exported rank file of one bundle, in rank order;
    each is read once, front to back. Every rank's payload stays contiguous, rank zero's where its
    export wrote it, so a host reads one region and this disk never holds a second copy of any
    rank. `cluster` is (world, export id): the package holds every rank, and each host loads the
    one it serves.
    """
    index, start = read_index(path)
    with Path(path).open("rb") as source:
        source.seek(8)
        header = json.loads(source.read(start - 8), object_pairs_hook=_unique_fields)
    if header["__metadata__"] != {"format": "flare-staging"}:
        raise ValueError("cluster assembly requires unsigned rank staging")
    if len(index.ranks) != 1:
        raise ValueError("rank export must contain one resource bundle")
    cursor = _byte_tensor(header["flare.payload"])[1]
    result = proto.ModelIndex()
    result.CopyFrom(index)
    with Path(path).open("r+b") as output:
        for stream in peers:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size <= MAX_HEADER_BYTES or (8 + size) % ALIGNMENT:
                raise ValueError("invalid rank export header")
            peer = json.loads(stream.read(size), object_pairs_hook=_unique_fields)
            if peer["__metadata__"] != {"format": "flare-staging"}:
                raise ValueError("cluster assembly requires unsigned rank staging")
            length = _byte_tensor(peer["flare.payload"])[1]
            index_start, end = _byte_tensor(peer["flare.metadata"])
            base = align(cursor)
            output.seek(start + base)
            _copy(stream, output, length)
            peer_index = proto.ModelIndex.FromString(stream.read(end - index_start))
            _validate_spans(peer_index, length)
            if len(peer_index.ranks) != 1 or peer_index.HasField("engram") or peer_index.assets:
                raise ValueError("a peer rank export must hold one bundle and no shared tables or assets")
            bundle = result.ranks.add()
            bundle.CopyFrom(peer_index.ranks[0])
            for span in bundle.resources.values():
                span.offset += base
            cursor = base + length
        if len(result.ranks) != cluster[0]:
            raise ValueError("a multi-host package holds every rank of its cluster")
        result.cluster.world, result.cluster.export_id = cluster
        _validate_spans(result, cursor)
        encoded = result.SerializeToString(deterministic=True)
        if not 0 < len(encoded) <= MAX_INDEX_BYTES:
            raise ValueError("model index exceeds its metadata limit")
        header = json.dumps({
            "__metadata__": {"format": "flare-staging"},
            "flare.payload": {"dtype": "U8", "shape": [cursor], "data_offsets": [0, cursor]},
            "flare.metadata": {"dtype": "U8", "shape": [len(encoded)],
                               "data_offsets": [cursor, cursor + len(encoded)]},
        }, separators=(",", ":")).encode()
        if 8 + len(header) > start:
            raise ValueError("package header outgrew rank zero's reserved space")
        output.seek(start + cursor)
        output.write(encoded)
        output.truncate()
        output.seek(0)
        output.write(struct.pack("<Q", start - 8) + header.ljust(start - 8, b" "))
        output.flush()
        os.fsync(output.fileno())


def fields(*values):
    return b''.join(struct.pack('<Q', len(value)) + value for value in values)


def signature_message(package_id, signer, kind, body):
    return fields(b'flare-package-signature-v4', struct.pack('<I', VERSION),
                  package_id, signer, struct.pack('<I', kind), body)


def resource_key(public_key, package_id, offset, namespace):
    # Bind ciphertext keys to the signing key, publisher, package, and resource.
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=package_id,
                info=fields(b'flare-executable-v3', struct.pack('<Q', offset), namespace.encode())).derive(public_key)


def chunk_aad(package_id, offset, size, chunk):
    return fields(b'flare-executable-chunk-v3', package_id,
                  struct.pack('<Q', offset), struct.pack('<Q', size), struct.pack('<Q', chunk))


def resource_roles(index):
    """Classify every payload span from its actual consumer, rejecting ambiguity.

    Returns (kind, owner, span) for each span: owner is its rank, or None for what every rank shares.
    """
    resources = []
    for owner, rank in enumerate(index.ranks):
        roles = {}
        for kind, names in (
            (2, [m.file for m in rank.modules] + [g.resource for g in rank.graphs]),
            (3, [g.file for m in rank.modules for g in m.globals]),
            (4, [a.file for a in rank.allocations if a.HasField('file')]),
        ):
            for name in names:
                if name not in rank.resources or (name in roles and roles[name] != kind):
                    raise ValueError('missing or conflicting resource classification')
                roles[name] = kind
        if set(roles) != set(rank.resources):
            raise ValueError('unclassified package resource')
        resources.extend((roles[name], owner, span) for name, span in rank.resources.items())
    resources.extend((3, None, asset.payload) for asset in index.assets)
    resources.extend((4, owner, span) for owner, span in embedding_spans(index))
    for kind, _, span in resources:
        if span.filename:
            validate_filename(span.filename)
            if index.version != 5 or kind != 4:
                raise ValueError('only stored data in model index 5 may reference a companion')
    ranges = sorted((s.filename, s.offset, s.nbytes) for _, _, s in resources)
    end, previous = 0, None
    for filename, offset, size in ranges:
        if filename != previous:
            end, previous = 0, filename
        if size <= 0 or offset < end:
            raise ValueError('empty or overlapping package resources')
        end = offset + size
    return resources


def _read_at(contents, position, length, error):
    block = os.pread(contents.fileno(), length, position)
    if len(block) != length:
        raise ValueError(error)
    return block


def _encrypt(cipher, package_id, offset, source, source_offset, size, encrypted):
    """Encrypt a resource chunk by chunk into `encrypted`; returns each ciphertext chunk's SHA-256.

    Sources are read by position, so jobs sharing an open source (a package being re-signed) do not race.
    """

    digests = []
    with _open_source(source) as contents, encrypted.open('xb') as output:
        for counter, start in enumerate(range(0, size, CHUNK)):
            block = _read_at(contents, source_offset + start, min(CHUNK, size - start), 'truncated executable resource')
            ciphertext = cipher.encrypt(counter.to_bytes(12, 'big'), block, chunk_aad(package_id, offset, size, counter))
            digests.append(hashlib.sha256(ciphertext).digest())
            output.write(ciphertext)
    return digests


def _digest(source, source_offset, size):

    digest = hashlib.sha256()
    with _open_source(source) as contents:
        for start in range(0, size, CHUNK):
            digest.update(_read_at(contents, source_offset + start, min(CHUNK, size - start),
                                   'truncated supporting resource'))
    return [digest.digest()]


class Protection:
    """Transform a final rank assembly and build its signed envelope.

    What every rank shares (the index without its bundles, assets and tables) is signed once, and
    each rank's bundle, executables, supporting resources and stored data separately, so a host
    holding one rank's bytes verifies that rank completely.
    """

    def __init__(self, index, payload, publisher, stage, *, shard_size=0, basename='model',
                 source_verifiers=None):
        from concurrent.futures import ThreadPoolExecutor
        if type(shard_size) is not int or shard_size < 0:
            raise ValueError('shard size must be a nonnegative byte count')
        self.publisher = publisher
        self.package_id = os.urandom(32)
        self.index = index
        self.source_verifiers = source_verifiers or {}
        owners = [None, *range(len(index.ranks))]
        roles = resource_roles(index)
        if any(span.filename for _, _, span in roles):
            raise ValueError('publication inputs must use a contiguous staging address space')
        by_offset = {span.offset: (kind, owner, span) for kind, owner, span in roles}
        starts = sorted(by_offset)
        ranges = {offset: [] for offset in starts}
        for entry in sorted(payload, key=lambda entry: entry[0]):
            if not starts or entry[0] < starts[0]:
                raise ValueError('publication range has no classified resource')
            ranges[starts[bisect_right(starts, entry[0]) - 1]].append(entry)
        for offset, entries in ranges.items():
            kind, _, span = by_offset[offset]
            ends = [destination + size for destination, _, _, size in entries]
            if (not entries or [entry[0] for entry in entries] != [offset, *ends[:-1]]
                    or kind != 4 and len(entries) != 1 or ends[-1] - offset != span.nbytes):
                raise ValueError('publication ranges differ from classified resources')
        ordered = sorted(ranges, key=lambda offset: (by_offset[offset][0] != 4,
                         owners.index(by_offset[offset][1]) if by_offset[offset][0] == 4 else 0, offset))
        sizes = {offset: by_offset[offset][2].nbytes for offset in ordered}
        self.hashes = {owner: hashlib.sha256() for owner in owners}

        def layout(sharded):
            self.code = {owner: proto.ResourceCatalog() for owner in owners[1:]}
            self.support = {owner: proto.ResourceCatalog() for owner in owners}
            self.ranges = {owner: [] for owner in owners}
            self.payloads = {'': []}
            positions = {'': 0}
            locations, catalog_entries = {}, {}
            data_file, data_files = '', []
            for offset in ordered:
                kind, owner, span = by_offset[offset]
                size = sizes[offset]
                stored = size + (16 * -(-size // CHUNK) if kind == 2 else 0)
                if sharded and kind == 4:
                    if not data_file or (positions[data_file] and align(positions[data_file]) + size + 4096 > shard_size):
                        data_file = str(len(data_files) + 1)
                        data_files.append(data_file)
                        if len(data_files) > 99:
                            raise ValueError('package requires more than 99 data files')
                        self.payloads[data_file], positions[data_file] = [], 0
                    filename = data_file
                else:
                    filename = ''
                cursor = align(positions[filename])
                span.offset, span.nbytes, span.filename = cursor, stored, filename
                positions[filename] = cursor + stored
                locations[offset] = (filename, cursor)
                if kind == 4:
                    parts = self.ranges[owner]
                    if parts and parts[-1].filename == filename:
                        parts[-1].nbytes = cursor + stored - parts[-1].offset
                    else:
                        parts.append(proto.Span(filename=filename, offset=cursor, nbytes=stored))
                else:
                    catalog = self.code[owner] if kind == 2 else self.support[owner]
                    catalog_entries[offset] = catalog.resources.add(
                        offset=cursor, plain_bytes=size, stored_bytes=stored,
                        sha256=[bytes(32)] * (-(-size // CHUNK) if kind == 2 else 1))
            # Assign alignment gaps to the preceding stored-data part, as in single-file packages.
            for filename in positions:
                parts = sorted((part for parts in self.ranges.values() for part in parts
                                if part.filename == filename), key=lambda part: part.offset)
                end = min((s.offset for k, _, s in roles if k != 4 and s.filename == filename),
                          default=positions[filename])
                for part, following in zip(parts, [p.offset for p in parts[1:]] + [end]):
                    part.nbytes = following - part.offset
            if sharded:
                names = {f: f'{basename}-data-{i:02}-of-{len(data_files):02}.safetensors'
                         for i, f in enumerate(data_files, 1)}
                for name in names.values():
                    validate_filename(name)
                for _, _, span in roles:
                    span.filename = names.get(span.filename, '')
                for parts in self.ranges.values():
                    for part in parts:
                        part.filename = names.get(part.filename, '')
                locations = {offset: (names.get(f, ''), pos) for offset, (f, pos) in locations.items()}
                self.payloads = {names.get(f, ''): [] for f in positions}
            index.version = 5 if sharded else 4
            return locations, positions[''], catalog_entries

        locations, length, catalog_entries = layout(False)
        encoded = self.envelope()
        total = len(container_header(length, len(encoded), 'flare')) + length + len(encoded)
        if shard_size and total > shard_size and any(kind == 4 for kind, _, _ in roles):
            locations, _, catalog_entries = layout(True)
        jobs = []
        for offset in ordered:
            kind, owner, span = by_offset[offset]
            filename, cursor = locations[offset]
            entries = ranges[offset]
            if kind != 4:
                ((_, source, source_offset, size),) = entries
                resource = catalog_entries[offset]
                if kind == 2:
                    encrypted = Path(stage) / f'executable-{cursor}'
                    cipher = AESGCM(resource_key(publisher.public_key, self.package_id, cursor, publisher.namespace))
                    jobs.append((resource, _encrypt, (cipher, self.package_id, cursor, source, source_offset,
                                                       size, encrypted)))
                    entries = [(offset, encrypted, 0, resource.stored_bytes)]
                else:
                    jobs.append((resource, _digest, (source, source_offset, size)))
            for _, source, source_offset, size in entries:
                self.payloads[filename].append((cursor, source, source_offset, size))
                cursor += size
        if jobs:
            with ThreadPoolExecutor(COPY_THREADS) as pool:
                futures = [(resource, pool.submit(work, *arguments)) for resource, work, arguments in jobs]
                for resource, future in futures:
                    del resource.sha256[:]
                    resource.sha256.extend(future.result())
        self.payload = self.payloads['']

    def envelope(self, final=False):
        """The signed envelope; before every payload byte is absorbed, one of the same size."""
        if final:
            for verifier in self.source_verifiers.values():
                verifier.finish()
        def stored(owner):
            parts = self.ranges[owner]
            digest = self.hashes[owner].digest() if final else bytes(32)
            if self.index.version == 4:
                part = parts[0] if parts else proto.Span()
                return proto.StoredData(offset=part.offset, nbytes=part.nbytes, sha256=digest)
            return proto.StoredData(nbytes=sum(p.nbytes for p in parts), sha256=digest, parts=parts)

        result = proto.ProtectedIndex(version=VERSION, package_id=self.package_id, signer=self.publisher.fingerprint)

        def sign(group, kind, body):
            signature = self.publisher.key.sign(signature_message(self.package_id, result.signer, kind, body))
            group.add(kind=kind, body=body, signature=signature)

        shared = proto.ModelIndex()
        shared.CopyFrom(self.index)
        del shared.ranks[:]
        bodies = [self.support[None].SerializeToString(deterministic=True),
                  stored(None).SerializeToString(deterministic=True)]
        metadata = proto.PublisherMetadata(index=shared, namespace=self.publisher.namespace,
            official_source=self.publisher.official_source, development=self.publisher.development,
            group_hashes=[hashlib.sha256(body).digest() for body in bodies], ranks=len(self.index.ranks))
        if self.publisher.license is not None:
            metadata.license.name = self.publisher.license.name
            metadata.license.link = self.publisher.license.link
        bodies.insert(0, metadata.SerializeToString(deterministic=True))
        for kind, body in enumerate(bodies, 1):
            sign(result.groups, kind, body)
        for rank, bundle in enumerate(self.index.ranks):
            sign(result.ranks, 4, proto.RankProtection(rank=rank, bundle=bundle, executables=self.code[rank],
                support=self.support[rank], data=stored(rank)).SerializeToString(deterministic=True))
        return result.SerializeToString(deterministic=True)


def publish_legacy(path, index, payload, *, publisher=None, shard_size=DEFAULT_SHARD_SIZE):
    """Write unsigned staging or signed physical version 4/5 package fixtures."""
    if publisher is None:
        return _publish(path, index, payload)
    with tempfile.TemporaryDirectory(prefix="flare-fixture-protection-") as stage:
        copied = type(index)()
        copied.CopyFrom(index)
        protection = Protection(copied, list(payload), publisher, stage,
                                shard_size=shard_size, basename=Path(path).stem)
        return _publish(path, copied, protection.payload, protection)


def read_index(path):
    """Decode fixture metadata from a physical version 4/5 container."""
    with Path(path).open('rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors header length')
        size, = struct.unpack('<Q', prefix)
        if not 0 < size <= MAX_HEADER_BYTES or (size + 8) % ALIGNMENT:
            raise ValueError('invalid or unaligned safetensors header')
        header = json.loads(source.read(size), object_pairs_hook=_unique_fields)
        if (set(header) != {'__metadata__', 'flare.payload', 'flare.metadata'}
                or header['__metadata__'] not in ({'format': 'flare'}, {'format': 'flare-staging'})):
            raise ValueError('not a Flare safetensors package')
        begin, length = _byte_tensor(header['flare.payload'])
        index_start, end = _byte_tensor(header['flare.metadata'])
        if begin or index_start != length or size + 8 + end != os.fstat(source.fileno()).st_size:
            raise ValueError('invalid package tensor ranges')
        if not 0 < end - index_start <= MAX_INDEX_BYTES:
            raise ValueError('invalid model index size')
        source.seek(size + 8 + index_start)
        encoded = source.read(end - index_start)
    if header['__metadata__']['format'] == 'flare-staging':
        result = proto.ModelIndex.FromString(encoded)
    else:
        envelope = proto.ProtectedIndex.FromString(encoded)
        if envelope.version != 4 or len(envelope.groups) != 3 or not envelope.ranks:
            raise ValueError('invalid protected package envelope')
        result = proto.PublisherMetadata.FromString(envelope.groups[0].body).index
        for rank in envelope.ranks:
            result.ranks.add().CopyFrom(proto.RankProtection.FromString(rank.body).bundle)
    _validate_spans(result, length)
    return result, size + 8
