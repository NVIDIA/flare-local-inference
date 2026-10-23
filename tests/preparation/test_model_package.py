"""Atomic publication and rank ownership in native model packages."""

from types import SimpleNamespace

import pytest

from helios.tests.signing import publisher
from ignition.preparation import model_package
from ignition.preparation.runtime_package_pb2 import Asset, ModelIndex, BundleIndex, Span
from ignition.preparation.package_file import combine, read_index, write_sources


@pytest.mark.parametrize('world', [1, 2])
def test_publication_contains_one_bundle_per_rank(monkeypatch, tmp_path, world):
    import torch
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    exported = []
    closed = []
    printed = []
    def export(prepared, images, storage, path, **kwargs):
        write_sources(path, ModelIndex(version=4, ranks=[BundleIndex()]), [])
        exported.append(path.name)
        return SimpleNamespace(memory_inventory={'rank': 0})
    monkeypatch.setattr(model_package, 'export_model', export)
    class Control:
        def send(self, command):
            assert command[0] == 'export'
            export(None, None, None, command[1])
        def recv(self):
            return {'peer_memory': [(0x210000000000, 65536)], 'memory_inventory': {'rank': 1}}
    prepared = SimpleNamespace(model=object(), controls=[Control()] if world == 2 else [],
                               close=lambda: closed.append('model'))
    storage = SimpleNamespace(world=world, peer_storage=SimpleNamespace(
        owned={0x200000000000: 65536}, reset=lambda: None),
        host_regions=[SimpleNamespace(close=lambda: closed.append('host'))])
    def assemble(*args, **kwargs):
        assert closed == ['model', 'host']
        assert len(exported) == world
        combine(*args, **kwargs)
    monkeypatch.setattr(model_package, 'combine', assemble)
    output = tmp_path / 'model'
    def inventory(reports, **kwargs):
        assert output.is_file() and closed == ['model', 'host']
        printed.extend(reports)
    monkeypatch.setattr(model_package, 'print_inventory', inventory)
    model_package.export_package(prepared, None, storage, output, options={}, publisher=publisher())
    descriptor, start = read_index(output)
    assert descriptor.version == 4
    assert len(descriptor.ranks) == world
    assert sorted(exported) == [f'rank-{rank}.json' for rank in range(world)]
    assert output.is_file()
    assert printed == [{'rank': rank} for rank in range(world)]
    assert len(descriptor.peer_memory) == (2 if world == 2 else 0)
    for owner, peer in enumerate(descriptor.peer_memory):
        assert peer.owner == owner and list(peer.readers) == list(range(world))
    assert {p.name for p in tmp_path.iterdir()} == {output.name, '.flare-cache'}


def test_failed_rank_export_leaves_no_published_model(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('missing module image')
    monkeypatch.setattr(model_package, 'export_model', fail)
    with pytest.raises(RuntimeError, match='missing module image'):
        model_package.export_package(object(), None, SimpleNamespace(world=1), tmp_path / 'model', options={}, publisher=publisher())
    assert not list(tmp_path.iterdir())


def test_combined_file_preserves_rank_resources_and_frontend_ranges(tmp_path):
    ranks = []
    for rank, data in enumerate([b'rank-zero-tokenizer', b'rank-one']):
        source = tmp_path / f'payload-{rank}'
        source.write_bytes(data)
        index = ModelIndex(version=4, ranks=[BundleIndex(
            resources={'weights': Span(offset=3, nbytes=len(data))})])
        if rank == 0:
            index.assets.append(Asset(name='tokenizer', payload=Span(offset=13, nbytes=9)))
            index.contract.model_id = "test"
        path = tmp_path / f'rank-{rank}.json'
        write_sources(path, index, [(3, source, 0, len(data))])
        ranks.append(path)
    model = tmp_path / 'model.safetensors'
    combine(model, ranks, [])
    index, start = read_index(model)
    with model.open('rb') as source:
        for rank, expected in zip(index.ranks, [b'rank-zero-tokenizer', b'rank-one']):
            span = rank.resources['weights']
            source.seek(start + span.offset)
            assert source.read(span.nbytes) == expected
        source.seek(start + index.assets[0].payload.offset)
        assert source.read(index.assets[0].payload.nbytes) == b'tokenizer'
    assert index.contract.model_id == 'test'
    with model.open('r+b') as source:
        source.truncate(model.stat().st_size - 1)
    with pytest.raises(ValueError, match='truncated export source'):
        read_index(model)


def test_offloaded_engram_planes_are_stored_once_and_bounds_checked(tmp_path):
    source = tmp_path / 'table'
    source.write_bytes(b'weights-and-scales')
    ranks = []
    for rank in range(2):
        index = ModelIndex(version=4, ranks=[BundleIndex(events=['engram_ready'])])
        payload = []
        if rank == 0:
            table = index.engram.tables.add(layer=1)
            table.weights.CopyFrom(Span(offset=4096, nbytes=7))
            table.scales.CopyFrom(Span(offset=8192, nbytes=6))
            payload = [(4096, source, 0, 7), (8192, source, 12, 6)]
        path = tmp_path / f'rank-{rank}.json'
        write_sources(path, index, payload)
        ranks.append(path)
    output = tmp_path / 'model.safetensors'
    combine(output, ranks, [])
    index, start = read_index(output)
    assert len(index.engram.tables) == 1
    with output.open('rb') as package:
        for field, expected in [('weights', b'weights'), ('scales', b'scales')]:
            span = getattr(index.engram.tables[0], field)
            package.seek(start + span.offset)
            assert package.read(span.nbytes) == expected
    with pytest.raises(ValueError, match='stored once on rank zero'):
        combine(tmp_path / 'duplicated', [ranks[0], ranks[0]], [])
    with source.open('r+b') as table:
        table.truncate(12 + 5)
    with pytest.raises(ValueError, match='source range exceeds file'):
        combine(tmp_path / 'truncated', ranks, [])
    assert not (tmp_path / 'truncated').exists()
