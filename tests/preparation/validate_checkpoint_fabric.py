"""Exercise rank-owned checkpoint reads and export channels on a configured Spark fabric.

Run with --rank 0..N-1 and the same --ranks, --rendezvous, --streams and --ring or --chain on
all hosts. The source fixture occupies 16 MiB; --mib controls bytes read per peer.
The preparation token is the existing shared credential. No Orbit agent is used.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import struct
import tempfile
import time
from types import SimpleNamespace

from ignition.preparation.checkpoint_fabric import FabricSource, FabricTransport
from ignition.preparation.checkpoint_source import BLOCK_BYTES, RemoteSource, RootSource, SourceServer, token
from ignition.preparation.host_ranks import receive_object, send_object


def counters():
    return {p.name: {kind: int((p / "statistics" / kind).read_text())
                     for kind in ("rx_bytes", "tx_bytes")}
            for p in Path('/sys/class/net').iterdir() if (p / 'statistics').is_dir()}


def run(args):
    key = token(args.token_file)
    identity = "checkpoint-fabric-validation"
    host, port = args.rendezvous.rsplit(':', 1)
    before = counters()
    with tempfile.TemporaryDirectory(prefix='flare-fabric-validation-') as directory:
        directory = Path(directory)
        fabric = source = client = None
        controls, readers = [], []
        try:
            if args.rank == 0:
                snapshot = directory / 'checkpoint'
                snapshot.mkdir()
                content = bytes(range(256)) * (BLOCK_BYTES // 256)
                header = json.dumps({'weight': {'dtype': 'U8', 'shape': [4 * BLOCK_BYTES],
                                               'data_offsets': [0, 4 * BLOCK_BYTES]}}).encode()
                (snapshot / 'weights.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + content * 4)
                (snapshot / 'config.json').write_text('{}')
                checkpoint = SimpleNamespace(repository=str(snapshot), revision='fixture')
                source = RootSource(snapshot, checkpoint)
                fabric = FabricSource(source, key, identity, args.ranks, ring=args.ring, chain=args.chain, streams=args.streams)
                with socket.create_server((host, int(port))) as listener:
                    listener.settimeout(120)
                    peers = {}
                    for _ in range(args.ranks - 1):
                        control, _ = listener.accept()
                        control.settimeout(120)
                        rank, endpoint = receive_object(control)
                        assert 1 <= rank < args.ranks and rank not in peers
                        peers[rank] = (control, endpoint)
                controls = [peers[rank][0] for rank in range(1, args.ranks)]
                endpoints = [fabric.transport.endpoint, *(peers[rank][1] for rank in range(1, args.ranks))]
                for control in controls:
                    send_object(control, (endpoints, checkpoint))
                fabric.transport.configure(endpoints)
                for rank, control in enumerate(controls, 1):
                    reader = SourceServer(source, control, key, rank, identity, transport={'streams': args.streams})
                    reader.thread.start()
                    readers.append(reader)
                for reader in readers:
                    reader.join()
                fabric.join()
                # The same authenticated streams carry prepared export bytes back to rank zero.
                def receive_export(item):
                    slot, reader = item
                    channel = reader.channel
                    for _ in range(args.export_blocks):
                        message, data = channel.receive('validation-export')
                        assert data == content
                    channel.send('validation-complete')
                with ThreadPoolExecutor(max_workers=(args.ranks - 1) * args.streams) as pool:
                    list(pool.map(receive_export, fabric.readers.items()))
                reports = [receive_object(control) for control in controls]
                for control in controls:
                    send_object(control, 'close')
                print(json.dumps({'rank': 0, 'ring': args.ring, 'chain': args.chain, 'streams': args.streams, 'peers': reports}), flush=True)
            else:
                fabric = FabricTransport(args.rank, args.ranks, ring=args.ring, chain=args.chain, streams=args.streams)
                deadline = time.monotonic() + 120
                while True:
                    try:
                        control = socket.create_connection((host, int(port)), timeout=10)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.2)
                controls = [control]
                send_object(control, (args.rank, fabric.endpoint))
                endpoints, checkpoint = receive_object(control)
                fabric.configure(endpoints)
                client = RemoteSource(control, key, args.rank, identity, checkpoint, directory=directory, fabric=fabric)
                extent = len(client.headers['weights.safetensors'])
                expected = bytes(range(256)) * (BLOCK_BYTES // 256)
                started = time.monotonic()
                def read(index):
                    data = client.read('weights.safetensors', extent + (index % 4) * BLOCK_BYTES, BLOCK_BYTES)
                    assert data == expected
                blocks = args.mib * (1 << 20) // BLOCK_BYTES
                with ThreadPoolExecutor(max_workers=args.streams) as pool:
                    list(pool.map(read, range(blocks)))
                seconds = time.monotonic() - started
                client.finish(keep_transport=True)
                def export(item):
                    channel, _ = item
                    for _ in range(args.export_blocks):
                        channel.send('validation-export', expected)
                    channel.receive('validation-complete')
                with ThreadPoolExecutor(max_workers=args.streams) as pool:
                    list(pool.map(export, client.channel_stats))
                send_object(control, dict(rank=args.rank, bytes=blocks * BLOCK_BYTES, seconds=seconds,
                                          gbps=blocks * BLOCK_BYTES * 8 / seconds / 1e9))
                assert receive_object(control) == 'close'
                client.close_transport()
        finally:
            if client is not None:
                client.close()
            if fabric is not None:
                fabric.close()
            if source is not None:
                source.close()
            for control in controls:
                control.close()
    after = counters()
    print(json.dumps({'rank': args.rank, 'interfaces': {name: {k: after[name][k] - values[k] for k in values}
                      for name, values in before.items() if name in after}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--ranks', type=int, required=True)
    parser.add_argument('--rendezvous', required=True)
    parser.add_argument('--streams', type=int, default=8, choices=range(1, 65))
    parser.add_argument('--ring', action='store_true')
    parser.add_argument('--chain', action='store_true')
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--mib', type=int, default=1024)
    parser.add_argument('--export-blocks', type=int, default=8)
    run(parser.parse_args())
