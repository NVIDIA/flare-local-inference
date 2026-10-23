"""Native export transfers preserve authenticated channels and atomic cache publication."""

from contextlib import ExitStack
import hashlib
import os
import json
from pathlib import Path
import resource
import socket
import struct
import threading
import time
from types import SimpleNamespace

import flare_heliopause
import pytest

from ignition.preparation import host_ranks
from ignition.preparation.checkpoint_source import BLOCK_BYTES, Channel
from ignition.preparation.package_file import container_header, read_sources, write_sources
from ignition.preparation.runtime_package_pb2 import BundleIndex, ModelIndex, Span


def sockets(stack):
    left, right = socket.socketpair()
    for connection in (left, right):
        stack.enter_context(connection)
        connection.settimeout(5)
    return left, right


def channels(stack, count):
    roots, peers = [], []
    for _ in range(count):
        left, right = sockets(stack)
        roots.append(Channel(left, b'k' * 32, server=True))
        peers.append(Channel(right, b'k' * 32, server=False))
    return roots, peers


def package(path):
    return SimpleNamespace(native_output=flare_heliopause.NativeRankOutput.from_sources(path), memory_inventory={})


def manifest(directory, count, *, large=False, blob=True):
    directory.mkdir(parents=True, exist_ok=True)
    payload, contents = [], {}
    bundle = BundleIndex()
    offset = 0
    for i in range(count):
        data = bytes([i % 256]) * (BLOCK_BYTES + 17 if large and i < 2 else i + 1)
        raw = container_header(len(data)) + data if blob else data
        name = directory / (hashlib.sha256(raw).hexdigest() + '.safetensors' if blob else f'global-{i}.bin')
        name.write_bytes(raw)
        payload.append((offset, name, len(raw) - len(data), len(data)))
        bundle.resources[str(i)].CopyFrom(Span(offset=offset, nbytes=len(data)))
        bundle.allocations.add(name=str(i), file=str(i))
        contents[str(i)] = data
        offset += (len(data) + 4095) // 4096 * 4096
    path = directory / 'peer.json'
    write_sources(path, ModelIndex(version=4, ranks=[bundle]), payload)
    return path, contents


def launch(function, connections):
    failures = []

    def run():
        try:
            function()
        except BaseException as error:
            failures.append(error)
            for connection in connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    thread = threading.Thread(target=run)
    thread.start()
    return thread, failures


def receive(control, roots, destination, cache, *, no_weights=False):
    request = dict(control_fd=control.fileno(), channels=host_ranks._export_channels(roots),
                   destination=str(destination))
    result = json.loads(flare_heliopause.export_receive_ranks(
        json.dumps([request]), cache, no_weights, (control, roots)))[0]
    host_ranks._restore_export_channels(roots, result['channels'])
    return result['stats']


def transfer(path, destination, cache, *, streams=4, no_weights=False):
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, streams)
        thread, failures = launch(lambda: host_ranks.send_export_sources(peer, package(path), {}, channels=peers),
                                  [peer, *(channel.connection for channel in peers)])
        receiving_error = None
        try:
            report = receive(control, roots, destination, cache, no_weights=no_weights)
        except BaseException as error:
            receiving_error = error
            raise
        finally:
            thread.join(10)
            if receiving_error is not None:
                for failure in failures:
                    receiving_error.add_note(f"Native sender failed: {failure}")
        assert not thread.is_alive() and not failures
        return report


def control_send(connection, value):
    payload = json.dumps(value).encode()
    connection.sendall(b'N' + struct.pack('<Q', len(payload)) + payload)


def exact(connection, count):
    result = bytearray()
    while len(result) < count:
        block = connection.recv(count - len(result))
        if not block:
            raise ConnectionError('native export control disconnected')
        result.extend(block)
    return bytes(result)


def control_receive(connection):
    assert exact(connection, 1) == b'N'
    length, = struct.unpack('<Q', exact(connection, 8))
    assert length <= 256 << 20
    return json.loads(exact(connection, length))


def description(path):
    output = json.loads(Path(path).read_text())
    files = {}
    for part in output['payload']:
        name = part['source']['path']
        data = Path(name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        files[name] = dict(size=len(data), sha256=digest, blob=Path(name).name == digest + '.safetensors')
    return dict(version=1, output=output, files=files, inventory={})


def assert_contents(path, contents):
    index, locate = read_sources(path)
    for name, span in index.ranks[0].resources.items():
        location, offset, size = locate(span)[0]
        with Path(location).open('rb') as source:
            source.seek(offset)
            assert source.read(size) == contents[name]


@pytest.mark.parametrize('streams', [1, 4, 8])
def test_parallel_export_and_verified_cache_reuse(tmp_path, streams):
    path, contents = manifest(tmp_path / 'peer', 11, large=True)
    expected = sum(info['size'] for info in description(path)['files'].values())
    for iteration in range(2):
        destination = tmp_path / f'received-{iteration}.json'
        report = transfer(path, destination, tmp_path / 'root-cache', streams=streams, no_weights=iteration == 1)
        assert report['streams'] == streams and report['transport'] == 'fabric-tcp'
        assert report['files'] == (len(contents) if iteration == 0 else 0)
        assert report['payload_bytes'] == (expected if iteration == 0 else 0)
        assert_contents(destination, contents)


def test_export_rejects_management_payload_fallback(tmp_path):
    path, _ = manifest(tmp_path / 'peer', 1)
    with ExitStack() as stack:
        control, _ = sockets(stack)
        with pytest.raises(RuntimeError, match='established fabric channels'):
            host_ranks.send_export_sources(control, package(path), {}, channels=[])
        request = dict(control_fd=control.fileno(), channels=[], destination=str(tmp_path / 'rank.json'))
        with pytest.raises(RuntimeError, match='management payload transfer is forbidden'):
            flare_heliopause.export_receive_ranks(json.dumps([request]), tmp_path / 'cache', False, (control,))



def test_export_requires_retained_socket_owners(tmp_path):
    path, _ = manifest(tmp_path / 'peer', 1)
    with pytest.raises(ValueError, match='retained socket owners'):
        flare_heliopause.export_send_rank(package(path).native_output, '{}', -1, '[]', None)
    with pytest.raises(ValueError, match='retained socket owners'):
        flare_heliopause.export_receive_ranks('[]', tmp_path / 'cache', False, None)


def test_native_export_requires_missing_weights_and_verifies_cached_content(tmp_path):
    path, _ = manifest(tmp_path / 'peer', 1)
    with pytest.raises(RuntimeError, match='--no-weights requires'):
        transfer(path, tmp_path / 'missing.json', tmp_path / 'cache', no_weights=True)
    assert not (tmp_path / 'missing.json').exists()
    name, info = next(iter(description(path)['files'].items()))
    target = tmp_path / 'cache/blobs' / Path(name).name
    target.write_bytes(bytes(info['size']))
    with pytest.raises(RuntimeError, match='cached blob content does not match its hash'):
        transfer(path, tmp_path / 'corrupt.json', tmp_path / 'cache')
    assert not (tmp_path / 'corrupt.json').exists()

def test_export_thousands_of_files_with_small_descriptor_limit(tmp_path):
    path, contents = manifest(tmp_path / 'peer', 1200)
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(limits[0], 128), limits[1]))
        report = transfer(path, tmp_path / 'root.json', tmp_path / 'root-cache', streams=8)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    assert report['files'] == 1200
    assert_contents(tmp_path / 'root.json', contents)


@pytest.mark.parametrize('disconnect', [False, True])
def test_large_export_frames_progress_with_scarce_descriptors(tmp_path, disconnect):
    payload = []
    bundle = BundleIndex()
    size = 2 * BLOCK_BYTES + 17
    for index in range(8):
        source = tmp_path / f'global-{index}.bin'
        source.write_bytes(bytes([index + 1]) * size)
        offset = index * (2 * BLOCK_BYTES + 4096)
        payload.append((offset, source, 0, size))
        bundle.resources[str(index)].CopyFrom(Span(offset=offset, nbytes=size))
        bundle.allocations.add(name=str(index), file=str(index))
    path = tmp_path / 'peer.json'
    write_sources(path, ModelIndex(version=4, ranks=[bundle]), payload)
    prepared = package(path)
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 8)
        for channel in (*roots, *peers):
            channel.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        # Match the available descriptors after preparation has initialized NVIDIA libraries.
        while len(os.listdir('/proc/self/fd')) - 1 < 116:
            stack.enter_context(open('/dev/null', 'rb'))
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(limits[0], 128), limits[1]))
            thread, failures = launch(
                lambda: host_ranks.send_export_sources(peer, prepared, {}, channels=peers),
                [peer, *(channel.connection for channel in peers)])
            try:
                time.sleep(0.2)
                assert thread.is_alive()
                if disconnect:
                    roots[3].connection.shutdown(socket.SHUT_RDWR)
                    with pytest.raises(RuntimeError):
                        receive(control, roots, tmp_path / 'root.json', tmp_path / 'cache')
                else:
                    report = receive(control, roots, tmp_path / 'root.json', tmp_path / 'cache')
                    assert report['streams'] == 8 and report['files'] == 8
                    assert all(channel.received >= 4 for channel in roots)
            finally:
                thread.join(10)
            assert not thread.is_alive()
            assert bool(failures) == disconnect
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    if disconnect:
        assert not (tmp_path / 'root.json').exists()
        assert not list(tmp_path.rglob('.rank-blob-*'))


@pytest.mark.parametrize('fault', ['offset', 'digest', 'truncated', 'accounting', 'authentication', 'replay'])
def test_failed_stream_aborts_siblings_and_removes_partial_files(tmp_path, fault):
    path, _ = manifest(tmp_path / 'peer', 1)
    document = description(path)
    name, info = next(iter(document['files'].items()))
    data = Path(name).read_bytes()
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 4)

        def transmit():
            control_send(peer, document)
            assert control_receive(peer)['files'] == [name]
            if fault == 'truncated':
                peers[0].connection.shutdown(socket.SHUT_RDWR)
                return
            if fault == 'authentication':
                peers[0].key = b'wrong' * 8
            if fault == 'replay':
                peers[0].sent += 1
            peers[0].send('export-data', bytes([data[0] ^ 1]) + data[1:] if fault == 'digest' else data,
                          name=name, offset=1 if fault == 'offset' else 0)
            if fault == 'accounting':
                peers[0].send('export-finished', stats=dict(files=1, payload_bytes=0))

        thread, failures = launch(transmit, [peer, *(channel.connection for channel in peers)])
        try:
            with pytest.raises(RuntimeError) as caught:
                receive(control, roots, tmp_path / 'root.json', tmp_path / 'root-cache')
            assert 'timed out' not in str(caught.value)
            assert not (tmp_path / 'root.json').exists()
            assert not list(tmp_path.rglob('.rank-blob-*'))
            if fault != 'accounting':
                assert not list((tmp_path / 'root-cache').rglob('*.safetensors'))
        finally:
            thread.join(5)
        assert not thread.is_alive() and not failures


@pytest.mark.parametrize('change', ['mutate', 'replace', 'truncate'])
def test_export_rejects_sources_changed_after_metadata_scan(tmp_path, change):
    path, _ = manifest(tmp_path / 'peer', 1)
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 4)
        thread, failures = launch(lambda: host_ranks.send_export_sources(peer, package(path), {}, channels=peers),
                                  [peer, *(channel.connection for channel in peers)])
        document = control_receive(control)
        name = next(iter(document['files']))
        target = Path(name)
        if change == 'replace':
            changed = tmp_path / 'replacement'
            changed.write_bytes(target.read_bytes())
            changed.replace(target)
        elif change == 'truncate':
            target.write_bytes(b'short')
        else:
            with target.open('r+b') as source:
                source.write(b'changed')
        control_send(control, dict(version=1, files=[name]))
        with pytest.raises(ConnectionError):
            roots[0].receive()
        thread.join(5)
        assert not thread.is_alive()
        assert len(failures) == 1 and 'changed before transfer' in str(failures[0])


def test_checkpoint_completion_hands_authenticated_channel_to_native_export(tmp_path):
    from tests.preparation.test_checkpoint_source import connected

    path, contents = manifest(tmp_path / 'peer', 3, large=True)
    with connected(tmp_path) as (client, server, offset), ExitStack() as stack:
        assert client.read('weights.safetensors', offset, 8) == bytes(range(8))
        client.finish(keep_transport=True)
        server.join()
        assert client.channel.sent == server.channel.received
        assert client.channel.received == server.channel.sent
        before = (client.channel.sent, client.channel.received)
        control, peer = sockets(stack)
        thread, failures = launch(lambda: host_ranks.send_export_sources(peer, package(path), {}, channels=[client.channel]),
                                  [peer, client.channel.connection])
        report = receive(control, [server.channel], tmp_path / 'root.json', tmp_path / 'root-cache')
        thread.join(5)
        assert report['files'] == 3 and report['transport'] == 'fabric-tcp'
        assert not thread.is_alive() and not failures
        assert client.channel.sent > before[0] and client.channel.received > before[1]
        assert client.channel.sent == server.channel.received
        assert client.channel.received == server.channel.sent
        client.channel.send('after-native-export', b'authenticated continuation')
        message, payload = server.channel.receive('after-native-export')
        assert message['bytes'] == len(payload) and payload == b'authenticated continuation'
        server.channel.send('continuation-ack')
        client.channel.receive('continuation-ack')
        assert_contents(tmp_path / 'root.json', contents)


def test_native_export_reports_original_failure_before_aborting_siblings(tmp_path, capfd):
    path, _ = manifest(tmp_path / 'peer', 2)
    document = description(path)
    names = sorted(document['files'])
    diagnostics = []
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 2)

        def observe_abort():
            with pytest.raises(ConnectionError):
                peers[0].receive()
            diagnostics.append(capfd.readouterr().err)

        observer, observed_failures = launch(observe_abort, [])

        def transmit():
            control_send(peer, document)
            assert control_receive(peer)['files'] == names
            peers[1].send('export-data', Path(names[1]).read_bytes(), name=names[1], offset=1)

        sender, failures = launch(transmit, [peer, peers[1].connection])
        with pytest.raises(RuntimeError, match='payload range differs'):
            receive(control, roots, tmp_path / 'root.json', tmp_path / 'cache')
        sender.join(5)
        observer.join(5)
        assert not sender.is_alive() and not observer.is_alive()
        assert not failures and not observed_failures
        assert len(diagnostics) == 1 and 'rank export payload range differs' in diagnostics[0]


@pytest.mark.parametrize('cached', [False, True])
def test_concurrent_native_rank_reuse_publishes_one_complete_blob(tmp_path, cached):
    path, contents = manifest(tmp_path / 'peer', 1, large=True)
    document = description(path)
    name = next(iter(document['files']))
    if cached:
        transfer(path, tmp_path / 'warm.json', tmp_path / 'cache', streams=1)
    ready = threading.Barrier(2, timeout=5)
    with ExitStack() as stack:
        workers, requests, root_channels, controls = [], [], [], []
        for rank in range(2):
            control, peer = sockets(stack)
            controls.append(control)
            roots, peers = channels(stack, 1)
            root_channels.append(roots)
            requests.append(dict(control_fd=control.fileno(), channels=host_ranks._export_channels(roots),
                                 destination=str(tmp_path / f'rank-{rank}.json')))

            def transmit(peer=peer, channel=peers[0]):
                control_send(peer, document)
                requested = control_receive(peer)['files']
                assert requested == ([] if cached else [name])
                ready.wait()
                if requested:
                    data = Path(name).read_bytes()
                    for offset in range(0, len(data), BLOCK_BYTES):
                        channel.send('export-data', data[offset:offset + BLOCK_BYTES], name=name, offset=offset)
                stats = dict(files=len(requested), payload_bytes=0 if cached else document['files'][name]['size'])
                channel.send('export-finished', stats=stats)
                message, payload = channel.receive('export-finished')
                assert not payload and message['stats'] == stats

            workers.append(launch(transmit, [peer, peers[0].connection]))
        received = json.loads(flare_heliopause.export_receive_ranks(
            json.dumps(requests), tmp_path / 'cache', cached, (controls, root_channels)))
        for thread, failures in workers:
            thread.join(5)
            assert not thread.is_alive() and not failures
        assert [item['stats']['files'] for item in received] == ([0, 0] if cached else [1, 1])
        for roots, result in zip(root_channels, received, strict=True):
            host_ranks._restore_export_channels(roots, result['channels'])
        assert not list(tmp_path.rglob('.rank-blob-*'))
        blobs = list((tmp_path / 'cache/blobs').glob('*.safetensors'))
        assert len(blobs) == 1 and blobs[0].stat().st_nlink == 1
        for rank in range(2):
            assert_contents(tmp_path / f'rank-{rank}.json', contents)
