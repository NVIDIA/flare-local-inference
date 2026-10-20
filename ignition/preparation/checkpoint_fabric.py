"""Rank-owned fabric TCP channels for checkpoint ranges and prepared exports."""

import json
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading

from .checkpoint_source import SourceServer, _exact


class FabricTransport:
    """One native relay per rank; stdin lifetime bounds every listener and worker."""

    def __init__(self, rank, world, *, ring, streams, chain=False, source_port=0):
        from ignition.cli.runtime import binary

        if not 1 <= streams <= 64:
            raise ValueError("checkpoint transfer streams must be between 1 and 64")
        self.rank, self.streams = rank, streams
        self.temporary = tempfile.TemporaryDirectory(prefix="flare-fabric-")
        self.path = str(Path(self.temporary.name) / "channels.sock")
        self.log = tempfile.TemporaryFile(mode="w+")
        self.process = None
        try:
            self.process = subprocess.Popen([str(binary()), "checkpoint-transport", "--socket", self.path],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True)
            self.endpoint = self._exchange(dict(rank=rank, world=world, ring=ring, chain=chain, streams=streams,
                                               source_port=source_port))
        except BaseException:
            self.close(check=False)
            raise

    def _exchange(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            self.log.seek(0)
            raise RuntimeError(f"checkpoint fabric helper stopped: {self.log.read()}")
        return json.loads(response)

    def configure(self, endpoints):
        if self._exchange(endpoints) != {"ready": True}:
            raise RuntimeError("checkpoint fabric helper did not confirm its endpoints")

    def connect(self, stream):
        connection = socket.socket(socket.AF_UNIX)
        try:
            connection.settimeout(30)
            connection.connect(self.path)
            connection.sendall(struct.pack("<II", self.rank, stream))
            return connection
        except BaseException:
            connection.close()
            raise

    def close(self, *, check=True):
        result = 0
        try:
            if self.process is not None:
                if not self.process.stdin.closed:
                    self.process.stdin.close()
                try:
                    result = self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        result = self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        result = self.process.wait()
                self.process.stdout.close()
                self.process = None
            if check and result:
                self.log.seek(0)
                raise RuntimeError(f"checkpoint fabric helper failed ({result}): {self.log.read()}")
        finally:
            self.log.close()
            self.temporary.cleanup()


class FabricSource:
    """Serve the admitted rank/stream slots from a private loopback checkpoint source."""

    def __init__(self, source, key, identity, world, *, ring, streams, chain=False):
        self.source, self.key, self.identity = source, key, identity
        self.world, self.streams = world, streams
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(1)
        self.readers = {}
        self.error = None
        self.stopping = threading.Event()
        try:
            self.transport = FabricTransport(0, world, ring=ring, chain=chain, streams=streams,
                                             source_port=self.listener.getsockname()[1])
        except BaseException:
            self.listener.close()
            raise
        self.thread = threading.Thread(target=self.run, name="flare-checkpoint-fabric", daemon=True)
        self.thread.start()

    def run(self):
        try:
            while len(self.readers) < (self.world - 1) * self.streams and not self.stopping.is_set():
                try:
                    connection, _ = self.listener.accept()
                except TimeoutError:
                    continue
                try:
                    connection.settimeout(30)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    rank, stream = struct.unpack("<II", _exact(connection, 8))
                    if not 1 <= rank < self.world or not 0 <= stream < self.streams or (rank, stream) in self.readers:
                        raise ValueError("checkpoint channel is outside its admitted stream slots")
                    reader = SourceServer(self.source, connection, self.key, rank, self.identity)
                    self.readers[rank, stream] = reader
                    reader.thread.start()
                except BaseException:
                    connection.close()
                    raise
        except BaseException as error:
            if not self.stopping.is_set():
                self.error = error
        finally:
            self.listener.close()

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("checkpoint fabric admission failed") from self.error
        for reader in self.readers.values():
            reader.join()

    def close(self):
        self.stopping.set()
        try:
            self.transport.close()
        finally:
            self.thread.join()
            for reader in self.readers.values():
                try:
                    reader.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            for reader in self.readers.values():
                reader.thread.join()
                reader.connection.close()
