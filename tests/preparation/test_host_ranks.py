"""Rank zero admits each peer rank once, from its own host, running rank zero's sources."""

import threading

import pytest

from ignition.preparation import host_ranks
from ignition.preparation.host_ranks import HostRendezvous, knock


def topology(boot):
    return {"host": boot, "boot_id": boot, "rdma": [],
            "gpu": {"name": "NVIDIA GB10", "uuid": f"GPU-{boot}", "total_bytes": 1 << 37,
                    "free_bytes": 1 << 36, "capability": [12, 1]}}


def rendezvous(world=2):
    owner = HostRendezvous("127.0.0.1:0", world, identity="same", topology=topology("zero"))
    return owner, f"127.0.0.1:{owner.listener.getsockname()[1]}"


def join(address, rank, world=2, *, identity="same", boot=None, results=None):
    try:
        sockets = knock(address, rank, world, identity=identity, topology=topology(boot or f"peer{rank}"),
                        timeout=10)
        if results is not None:
            results.append(sockets)
    except RuntimeError as error:
        if results is not None:
            results.append(error)


def test_admits_every_peer_rank_with_its_executor_connection():
    owner, address = rendezvous(4)
    results = []
    peers = [threading.Thread(target=join, args=(address, rank, 4), kwargs={"results": results})
             for rank in (3, 1, 2)]
    for peer in peers:
        peer.start()
    topologies = owner.accept(timeout=10)
    for peer in peers:
        peer.join()
    assert [item["boot_id"] for item in topologies] == ["zero", "peer1", "peer2", "peer3"]
    assert owner.hosts == ("zero", "peer1", "peer2", "peer3")
    assert all(isinstance(result, tuple) for result in results)
    group = host_ranks.HostRankGroup.__new__(host_ranks.HostRankGroup)
    group.rendezvous = owner
    assert len(group.connections) == len(group.controls) == 3


@pytest.mark.parametrize("arguments, reason", [
    ({"identity": "other"}, "source identity"),
    ({"world": 4}, "world size"),
])
def test_rejects_peers_that_differ_from_rank_zero(arguments, reason):
    owner, address = rendezvous()
    results = []
    peer = threading.Thread(target=join, args=(address, 1), kwargs={**arguments, "results": results})
    peer.start()
    with pytest.raises(TimeoutError):
        owner.accept(timeout=2)
    peer.join()
    assert isinstance(results[0], RuntimeError) and reason in str(results[0])


def test_rejects_a_second_process_for_an_admitted_rank():
    owner, address = rendezvous(4)
    failures = []

    def admit():
        try:
            owner.accept(timeout=3)
        except TimeoutError as error:
            failures.append(error)

    accepting = threading.Thread(target=admit)
    accepting.start()
    results = []
    join(address, 1, 4, results=results)
    join(address, 1, 4, results=results)
    accepting.join()
    assert isinstance(results[0], tuple) and failures
    assert isinstance(results[1], RuntimeError) and "already joined" in str(results[1])


def test_requires_one_rank_per_host():
    owner, address = rendezvous()
    peer = threading.Thread(target=join, args=(address, 1), kwargs={"boot": "zero"})
    peer.start()
    with pytest.raises(ValueError, match="different host"):
        owner.accept(timeout=10)
    peer.join()


def test_finds_the_interface_holding_an_address():
    assert host_ranks.interface_of("127.0.0.1") == "lo"
    with pytest.raises(RuntimeError):
        host_ranks.interface_of("203.0.113.7")


def test_multi_host_preparation_requires_a_hub_checkpoint(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="hub cache"):
        owner.prepare(tmp_path / "local-model", None, None,
                      checkpoint=SimpleNamespace(repository="org/model", revision="rev"))


def test_local_checkpoint_source_must_match_its_declared_directory(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="declared source"):
        owner.prepare(tmp_path / "source", None, None,
                      checkpoint=SimpleNamespace(repository=str(tmp_path / "other"), revision="path-id"))



def test_export_manifest_transfers_only_absent_files(tmp_path):
    from ignition.preparation.package_file import container_header, read_sources, write_sources
    from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span
    from tests.preparation.test_export_transfer import transfer
    import hashlib

    data = container_header(8) + b'weights!'
    weights = tmp_path / (hashlib.sha256(data).hexdigest() + '.safetensors')
    weights.write_bytes(data)
    code = tmp_path / 'code.bin'
    code.write_bytes(b'kernel')
    bundle = BundleIndex(resources={'weights': Span(nbytes=8), 'kernel': Span(offset=4096, nbytes=6)})
    bundle.allocations.add(name='weights', file='weights')
    bundle.modules.add(name='kernel', file='kernel')
    manifest = tmp_path / 'rank.json'
    write_sources(manifest, ModelIndex(version=4, ranks=[bundle]),
                  [(0, weights, 4096, 8), (4096, code, 0, 6)])
    for iteration in range(2):
        destination = tmp_path / f'received-{iteration}' / 'rank.json'
        report = transfer(manifest, destination, tmp_path / 'cache', streams=2, no_weights=iteration == 1)
        assert report['files'] == (2 if iteration == 0 else 1)
        assert report['payload_bytes'] == (len(data) + 6 if iteration == 0 else 6)
        index, locate = read_sources(destination)
        for name, expected in [('weights', b'weights!'), ('kernel', b'kernel')]:
            path, offset, count = locate(index.ranks[0].resources[name])[0]
            assert path.read_bytes()[offset:offset + count] == expected


def test_export_subprocess_reopens_more_sources_than_descriptor_limit(tmp_path):
    import subprocess
    import sys
    from contextlib import ExitStack
    from tests.preparation.test_export_transfer import channels, manifest, receive, sockets, assert_contents

    path, contents = manifest(tmp_path / 'peer', 256, blob=False)
    code = """
import resource, socket, sys
from ignition.preparation.host_ranks import send_export_sources
from ignition.preparation.checkpoint_source import Channel
from tests.preparation.test_export_transfer import package
resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
with socket.socket(fileno=int(sys.argv[1])) as control:
    control.settimeout(10)
    connections = [socket.socket(fileno=int(fd)) for fd in sys.argv[3:]]
    channels = [Channel(connection, b'k' * 32, server=False) for connection in connections]
    send_export_sources(control, package(sys.argv[2]), {}, channels=channels)
"""
    with ExitStack() as stack:
        control, peer = sockets(stack)
        roots, peers = channels(stack, 2)
        inherited = (peer.fileno(), *(channel.connection.fileno() for channel in peers))
        child = subprocess.Popen([sys.executable, '-c', code, str(peer.fileno()), str(path),
                                  *(str(channel.connection.fileno()) for channel in peers)],
                                 pass_fds=inherited, stderr=subprocess.PIPE)
        peer.close()
        for channel in peers:
            channel.connection.close()
        try:
            report = receive(control, roots, tmp_path / 'received.json', tmp_path / 'cache')
            _, errors = child.communicate(timeout=10)
            assert child.returncode == 0, errors.decode()
            assert report['files'] == 256
            assert_contents(tmp_path / 'received.json', contents)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def test_cluster_export_combines_native_received_manifests_into_signed_snapshot(tmp_path):
    import hashlib
    from helios.tests.signing import publisher
    from ignition.preparation.package_file import combine, container_header, write_sources, read_index, PackageFiles
    from ignition.preparation.package_snapshot import export_options
    from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span
    from tests.preparation.test_export_transfer import transfer

    def sources(rank):
        contents = f'weights-{rank}'.encode()
        raw = container_header(len(contents)) + contents
        blob = tmp_path / (hashlib.sha256(raw).hexdigest() + '.safetensors')
        blob.write_bytes(raw)
        code = tmp_path / f'code-{rank}.bin'
        code.write_bytes(b'kernel')
        bundle = BundleIndex(resources={'weights': Span(nbytes=len(contents)), 'kernel': Span(offset=4096, nbytes=6)})
        bundle.allocations.add(name='weights', file='weights')
        bundle.modules.add(name='kernel', file='kernel')
        bundle.roce.world, bundle.roce.rank = 2, rank
        path = tmp_path / f'peer-{rank}.json'
        write_sources(path, ModelIndex(version=4, ranks=[bundle]),
                      [(0, blob, 4096, len(contents)), (4096, code, 0, 6)])
        return path

    local, remote = sources(0), sources(1)
    received = tmp_path / 'received' / 'rank-1.json'
    report = transfer(remote, received, tmp_path / 'cache', streams=2)
    assert report['files'] == 2
    output = tmp_path / 'model.safetensors'
    with export_options(tmp_path / 'cache'):
        combine(output, [local, received], [], publisher=publisher(), cluster=(2, 'snapshot-cluster'))
    index, start = read_index(output)
    assert index.cluster.world == 2 and index.cluster.export_id == 'snapshot-cluster'
    with output.open('rb') as primary, PackageFiles(output, primary, start, 0) as files:
        for rank in range(2):
            span = index.ranks[rank].resources['weights']
            source, base, _ = files.get(span.filename)
            assert source.read_at(span.nbytes, base + span.offset) == f'weights-{rank}'.encode()
