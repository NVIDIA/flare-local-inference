"""Serve immutable checkpoint ranges while peer ranks retain only metadata."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import queue
import secrets
import socket
import stat
import struct
import tempfile
import threading
import time


BLOCK_BYTES = 4 << 20
METADATA_BYTES = 256 << 20
MARKER = ".flare-remote-source.json"
_OWNER = ContextVar("flare_checkpoint_source", default=None)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _exact(connection, length):
    data = bytearray()
    while len(data) < length:
        block = connection.recv(min(length - len(data), BLOCK_BYTES))
        if not block:
            raise ConnectionError("checkpoint source disconnected")
        data.extend(block)
    return bytes(data)


def _packet(connection, data):
    if len(data) > BLOCK_BYTES:
        raise ValueError("checkpoint frame exceeds its bounded capacity")
    connection.sendall(struct.pack("<I", len(data)) + data)


def _receive(connection):
    length, = struct.unpack("<I", _exact(connection, 4))
    if length > BLOCK_BYTES:
        raise ValueError("checkpoint frame exceeds its bounded capacity")
    return _exact(connection, length)


def token(path):
    """Read the shared cluster credential without putting it on the wire."""
    path = Path(path).expanduser()
    if path.stat().st_mode & 0o077:
        raise ValueError("checkpoint token file must be private to its owner")
    value = path.read_bytes().strip()
    if len(value) < 32:
        raise ValueError("checkpoint token must contain at least 32 bytes")
    return value


class Channel:
    """Authenticate each bounded frame, its direction, and its sequence number."""

    def __init__(self, connection, key, *, server):
        self.connection, self.key = connection, key
        self.send_role, self.receive_role = (b"root", b"peer") if server else (b"peer", b"root")
        self.sent = self.received = 0

    def send(self, kind, payload=b"", **fields):
        header = _json(dict(kind=kind, sequence=self.sent, bytes=len(payload), **fields))
        if len(header) > 65536 or len(payload) > BLOCK_BYTES:
            raise ValueError("checkpoint message exceeds its bounded capacity")
        signature = hmac.digest(self.key, self.send_role + header + payload, "sha256")
        _packet(self.connection, header)
        self.connection.sendall(payload + signature)
        self.sent += 1

    def receive(self, expected=None):
        header = _receive(self.connection)
        if len(header) > 65536:
            raise ValueError("checkpoint message header exceeds capacity")
        message = json.loads(header)
        length = message.get("bytes")
        if type(length) is not int or not 0 <= length <= BLOCK_BYTES:
            raise ValueError("invalid checkpoint payload length")
        payload, signature = _exact(self.connection, length), _exact(self.connection, 32)
        actual = hmac.digest(self.key, self.receive_role + header + payload, "sha256")
        if not hmac.compare_digest(signature, actual):
            raise ValueError("checkpoint message authentication failed")
        if message.get("sequence") != self.received:
            raise ValueError("checkpoint message sequence differs")
        if expected is not None and message.get("kind") != expected:
            raise ValueError(f"expected checkpoint {expected}, received {message.get('kind')}")
        self.received += 1
        if message.get("kind") == "error":
            raise RuntimeError(f"checkpoint source rejected access: {message.get('error')}")
        return message, payload


def _proof(key, role, challenge, nonce):
    return hmac.digest(key, role + _json(challenge) + nonce.encode(), "sha256").hex()


def authenticate(connection, key, *, rank, identity, catalog=None):
    """Bind both peers to fresh nonces, rank, source code and catalog identity."""
    server = catalog is not None
    if server:
        challenge = dict(version=1, rank=rank, identity=identity, nonce=secrets.token_hex(32),
                         catalog_sha256=hashlib.sha256(_json(catalog)).hexdigest())
        _packet(connection, _json(challenge))
        reply = json.loads(_receive(connection))
        nonce = reply.get("nonce", "")
        if len(nonce) != 64 or not hmac.compare_digest(reply.get("proof", ""), _proof(key, b"peer", challenge, nonce)):
            raise ValueError("checkpoint peer authentication failed")
        _packet(connection, _json(dict(proof=_proof(key, b"root", challenge, nonce))))
    else:
        challenge = json.loads(_receive(connection))
        if (challenge.get("version"), challenge.get("rank"), challenge.get("identity")) != (1, rank, identity):
            raise ValueError("checkpoint source rank or code identity differs")
        nonce = secrets.token_hex(32)
        _packet(connection, _json(dict(nonce=nonce, proof=_proof(key, b"peer", challenge, nonce))))
        reply = json.loads(_receive(connection))
        if not hmac.compare_digest(reply.get("proof", ""), _proof(key, b"root", challenge, nonce)):
            raise ValueError("checkpoint root authentication failed")
    session = hmac.digest(key, b"checkpoint-session" + _json(challenge) + nonce.encode(), "sha256")
    return Channel(connection, session, server=server), challenge["catalog_sha256"]


def relative_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError("checkpoint path must be a canonical source-relative path")
    return name


def file_identity(descriptor):
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("checkpoint source must be a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def row_extent(size, offset, width, rows, stride):
    if any(type(value) is not int or value < 0 or value >= 1 << 63 for value in (size, offset, width, rows, stride)):
        raise ValueError("checkpoint row geometry must use nonnegative signed integers")
    if rows < 1 or width * rows > BLOCK_BYTES:
        raise ValueError("checkpoint request exceeds its bounded capacity")
    extent = (rows - 1) * stride + width
    if extent > BLOCK_BYTES or offset + extent > size:
        raise ValueError("checkpoint request lies outside its source or bounded extent")
    return extent


@dataclass
class HeldFile:
    path: Path
    descriptor: int
    identity: tuple

    def validate(self):
        if file_identity(self.descriptor) != self.identity or file_identity_path(self.path) != self.identity:
            raise RuntimeError("checkpoint source changed after admission")


def file_identity_path(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def release_read_pages(descriptor, offset=0, length=0):
    """Release completed buffered reads from the integrated GPU's shared RAM budget."""
    page = os.sysconf("SC_PAGESIZE")
    first = offset // page * page
    size = ((offset + length + page - 1) // page * page - first) if length else 0
    os.posix_fadvise(descriptor, first, size, os.POSIX_FADV_DONTNEED)


class RootSource:
    """Own authenticated source descriptors until all peer loading sessions end."""

    def __init__(self, snapshot, checkpoint):
        self.snapshot = Path(snapshot).resolve()
        if (self.snapshot / MARKER).exists():
            raise ValueError("a metadata-only remote source cannot publish a checkpoint")
        self.files = {}
        entries, metadata_bytes = [], 0
        try:
            for path in sorted(self.snapshot.rglob("*")):
                if not path.is_file():
                    continue
                name = relative_name(path.relative_to(self.snapshot).as_posix())
                descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                held = HeldFile(path, descriptor, file_identity(descriptor))
                self.files[name] = held
                # Selective reads release their completed ranges; read-ahead outside those ranges
                # would remain cached in the same DRAM needed by the integrated GPU.
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_RANDOM)
                size = held.identity[2]
                header = None
                if name.endswith(".safetensors"):
                    prefix = os.pread(descriptor, 8, 0)
                    header = 8 + int.from_bytes(prefix, "little")
                    if len(prefix) != 8 or not 8 < header <= min(size, (100 << 20) + 8):
                        raise ValueError(f"invalid safetensors source header: {name}")
                extent = size if header is None else header
                metadata_bytes += extent
                if metadata_bytes > METADATA_BYTES:
                    raise ValueError("checkpoint metadata and headers exceed the bounded admission budget")
                digest = hashlib.sha256()
                for offset in range(0, extent, BLOCK_BYTES):
                    data = os.pread(descriptor, min(BLOCK_BYTES, extent - offset), offset)
                    if len(data) != min(BLOCK_BYTES, extent - offset):
                        raise ValueError("checkpoint metadata read was truncated")
                    digest.update(data)
                held.validate()
                release_read_pages(descriptor)
                entries.append(dict(path=name, bytes=size, header_bytes=header, sha256=digest.hexdigest()))
            self.catalog = dict(repository=checkpoint.repository, revision=checkpoint.revision,
                                name=self.snapshot.name, files=entries)
            if len(_json(self.catalog)) > BLOCK_BYTES:
                raise ValueError("checkpoint catalog exceeds capacity")
        except BaseException:
            self.close()
            raise

    def read(self, name, offset, width, rows=1, stride=0):
        held = self.files[relative_name(name)]
        extent = row_extent(held.identity[2], offset, width, rows, stride)
        held.validate()
        raw = os.pread(held.descriptor, extent, offset)
        if len(raw) != extent:
            raise ConnectionError("checkpoint source read was truncated")
        held.validate()
        result = b"".join(raw[index * stride:index * stride + width] for index in range(rows))
        release_read_pages(held.descriptor, offset, extent)
        return result

    def close(self):
        for held in self.files.values():
            release_read_pages(held.descriptor)
            os.close(held.descriptor)
        self.files.clear()


class SourceServer:
    """Own one checkpoint channel until its authenticated loading drain."""

    def __init__(self, source, connection, key, rank, identity, *, transport=None):
        self.source, self.connection, self.key, self.rank, self.identity = source, connection, key, rank, identity
        self.transport = transport
        self.ready = threading.Event()
        self.error = None
        self.stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
        self.thread = threading.Thread(target=self.run, name=f"flare-checkpoint-rank{rank}", daemon=True)

    def run(self):
        try:
            channel, _ = authenticate(self.connection, self.key, rank=self.rank, identity=self.identity,
                                      catalog=self.source.catalog)
            self.channel = channel
            channel.send("catalog", _json(self.source.catalog), transport=self.transport)
            while True:
                message, payload = channel.receive()
                if payload:
                    raise ValueError("checkpoint requests cannot carry payload bytes")
                kind = message["kind"]
                if kind == "ready":
                    channel.send("ready")
                    self.connection.settimeout(None)
                    self.ready.set()
                elif kind == "finished":
                    if not self.ready.is_set() or message["stats"] != self.stats:
                        raise ValueError("checkpoint source completion byte accounting differs")
                    for held in self.source.files.values():
                        held.validate()
                    channel.send("finished", stats=self.stats)
                    return
                elif kind == "read":
                    fields = message["read"]
                    data = self.source.read(**fields)
                    self.stats["payload_bytes" if fields["name"].endswith(".safetensors") and
                               fields["offset"] >= next(e["header_bytes"] for e in self.source.catalog["files"]
                                                       if e["path"] == fields["name"]) else "metadata_bytes"] += len(data)
                    self.stats["requests"] += 1
                    channel.send("data", data)
                else:
                    raise ValueError(f"unknown checkpoint source operation {kind}")
        except BaseException as error:
            self.error = error
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        finally:
            self.ready.set()

    def wait_ready(self):
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError(f"rank {self.rank} checkpoint admission failed") from self.error

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"rank {self.rank} checkpoint reader failed") from self.error


class RemoteSource:
    """A scoped source directory containing real metadata and no tensor files."""

    def __init__(self, connection, key, rank, identity, checkpoint, *, directory=None, on_created=None, fabric=None):
        self.channel, digest = authenticate(connection, key, rank=rank, identity=identity)
        admission, payload = self.channel.receive("catalog")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("checkpoint catalog identity differs")
        catalog = json.loads(payload)
        if (catalog["repository"], catalog["revision"]) != (checkpoint.repository, checkpoint.revision):
            raise ValueError("checkpoint source differs from its preparation assignment")
        self.files = {relative_name(entry["path"]): entry for entry in catalog["files"]}
        if len(self.files) != len(catalog["files"]):
            raise ValueError("duplicate checkpoint source path")
        metadata = 0
        for entry in self.files.values():
            size, header = entry["bytes"], entry["header_bytes"]
            if type(size) is not int or not 0 <= size < 1 << 63:
                raise ValueError("invalid checkpoint source extent")
            if header is not None and (type(header) is not int or not 8 < header <= min(size, (100 << 20) + 8)):
                raise ValueError("invalid checkpoint header extent")
            metadata += size if header is None else header
        if metadata > METADATA_BYTES:
            raise ValueError("checkpoint metadata exceeds admission capacity")
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="flare-checkpoint-", dir=directory)
        if on_created is not None:
            on_created(self)
        name = relative_name(catalog["name"])
        if "/" in name:
            raise ValueError("checkpoint snapshot name must be a basename")
        self.snapshot = Path(self.temporary.name) / name
        self.snapshot.mkdir()
        self.headers = {}
        self.stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
        self.control_stats = self.stats
        self.channels = None
        self.fabric_connections = []
        self.channel_stats = []
        self.stats_lock = threading.Lock()
        self.streams = 1
        self.read_started = self.read_finished = None
        self.finished = False
        (self.snapshot / MARKER).write_text(json.dumps(dict(catalog_sha256=digest)))
        try:
            for name, entry in self.files.items():
                size = entry["bytes"] if entry["header_bytes"] is None else entry["header_bytes"]
                content = bytearray()
                for offset in range(0, size, BLOCK_BYTES):
                    content.extend(self.read(name, offset, min(BLOCK_BYTES, size - offset)))
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError("checkpoint metadata fingerprint differs")
                if entry["header_bytes"] is not None:
                    self.headers[name] = bytes(content)
                else:
                    target = self.snapshot / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            if admission.get("transport") is not None:
                self._connect_fabric(fabric, admission["transport"], key, rank, identity, digest)
            self.channel.send("ready")
            self.channel.receive("ready")
        except BaseException:
            self.close()
            raise

    def _connect_fabric(self, fabric, transport, key, rank, identity, digest):
        streams = transport["streams"]
        if fabric is None or streams != fabric.streams:
            raise ValueError("checkpoint fabric differs from its admitted stream count")
        self.control_stats = dict(self.stats)
        self.channels = queue.Queue()
        self.streams = streams
        for stream in range(streams):
            connection = fabric.connect(stream)
            self.fabric_connections.append(connection)
            try:
                channel, actual = authenticate(connection, key, rank=rank, identity=identity)
                _, payload = channel.receive("catalog")
                if actual != digest or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("checkpoint fabric catalog differs from its admitted source")
                channel.send("ready")
                channel.receive("ready")
            except BaseException:
                self.fabric_connections.pop().close()
                raise
            connection.settimeout(None)
            stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
            self.channel_stats.append((channel, stats))
            self.channels.put((channel, stats))

    @contextmanager
    def activate(self):
        value = _OWNER.set(self)
        try:
            yield self.snapshot
        finally:
            _OWNER.reset(value)

    def name(self, path):
        return Path(path).resolve().relative_to(self.snapshot).as_posix()

    def read(self, name, offset, width, rows=1, stride=0):
        if self.finished:
            raise RuntimeError("checkpoint reader has already drained")
        entry = self.files[relative_name(name)]
        row_extent(entry["bytes"], offset, width, rows, stride)
        channel, stats = self.channels.get() if self.channels is not None else (self.channel, self.control_stats)
        started = time.monotonic()
        try:
            channel.send("read", read=dict(name=name, offset=offset, width=width, rows=rows, stride=stride))
            _, data = channel.receive("data")
            if len(data) != width * rows:
                raise ConnectionError("checkpoint source returned a different byte extent")
            field = ("payload_bytes" if entry["header_bytes"] is not None and offset >= entry["header_bytes"]
                     else "metadata_bytes")
            with self.stats_lock:
                stats[field] += len(data)
                stats["requests"] += 1
                if stats is not self.stats:
                    self.stats[field] += len(data)
                    self.stats["requests"] += 1
                if field == "payload_bytes":
                    self.read_started = started if self.read_started is None else min(self.read_started, started)
                    self.read_finished = time.monotonic()
            return data
        finally:
            if self.channels is not None:
                self.channels.put((channel, stats))

    def finish(self, *, keep_transport=False):
        total = {field: sum(stats[field] for _, stats in [*self.channel_stats, (self.channel, self.control_stats)])
                 for field in self.stats}
        if total != self.stats:
            raise ValueError("checkpoint stream accounting differs from its aggregate")
        for channel, stats in [*self.channel_stats, (self.channel, self.control_stats)]:
            channel.send("finished", stats=stats)
            message, _ = channel.receive("finished")
            if message["stats"] != stats:
                raise ValueError("checkpoint completion accounting differs")
        self.finished = True
        if not keep_transport:
            self.close_transport()

    def close_transport(self):
        while self.fabric_connections:
            self.fabric_connections.pop().close()

    def report(self):
        seconds = 0 if self.read_started is None else self.read_finished - self.read_started
        return dict(self.stats, transport="fabric-tcp" if self.channels is not None else "rendezvous-tcp",
                    streams=self.streams, payload_seconds=seconds,
                    payload_gbps=self.stats["payload_bytes"] * 8 / seconds / 1e9 if seconds else 0)

    def abort_reads(self):
        """Wake all concurrent reads before their owners release transfer buffers."""
        connections = ([channel.connection for channel, _ in self.channel_stats] if self.channels is not None
                       else [self.channel.connection])
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self):
        while getattr(self, "fabric_connections", []):
            self.fabric_connections.pop().close()
        self.temporary.cleanup()


def source_for(path):
    """Resolve only the active owner's paths; marked directories require that owner."""
    path = Path(path).resolve()
    owner = _OWNER.get()
    if owner is not None and path.is_relative_to(owner.snapshot):
        return owner
    for parent in (path, *path.parents):
        if (parent / MARKER).is_file():
            raise RuntimeError("remote checkpoint access requires its active source owner")
    return None


def source_size(path):
    owner = source_for(path)
    return Path(path).stat().st_size if owner is None else owner.files[owner.name(path)]["bytes"]


def tensor_dtype(name):
    import torch

    names = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32",
             "I64": "int64", "U16": "uint16", "U32": "uint32", "U64": "uint64", "F16": "float16",
             "BF16": "bfloat16", "F32": "float32", "F64": "float64", "F8_E4M3": "float8_e4m3fn",
             "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu"}
    return getattr(torch, names[name])


def read_small_tensor(manifest, name, *, limit):
    """Read a declared remote CPU scalar without opening a nonexistent tensor file."""
    import torch

    source = source_for(manifest.snapshot)
    if source is None:
        raise ValueError("remote scalar reads require an owned source")
    tensor = manifest.tensors[name]
    if tensor.nbytes > min(limit, BLOCK_BYTES):
        raise ValueError("remote scalar exceeds its declared read budget")
    if not tensor.nbytes:
        return torch.empty(tensor.shape, dtype=tensor_dtype(tensor.dtype), device="cpu")
    content = bytearray(source.read(tensor.shard, tensor.offset, tensor.nbytes))
    return torch.frombuffer(content, dtype=tensor_dtype(tensor.dtype)).reshape(tensor.shape)
