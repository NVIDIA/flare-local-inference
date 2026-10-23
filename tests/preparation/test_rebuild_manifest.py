"""Signed preparation metadata names stored layouts without changing serving payloads."""

from dataclasses import dataclass

import pytest
import torch

from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.package_file import combine, read_index, write_sources
from ignition.preparation.package_snapshot import convert, read_container
from ignition.preparation.rebuild_manifest import rebuild_weights_metadata, validate_weights, weight_views
from helios.tests.signing import publisher
from tests.preparation.test_package_snapshot import native, profile, assert_native
from tests.preparation import test_package_protection as fixtures


@dataclass
class PackedWeight:
    values: torch.Tensor
    scales: torch.Tensor

    @property
    def forbidden(self):
        raise AssertionError('cataloging must not evaluate properties')


def test_prepared_views_preserve_names_aliases_strides_and_packing_owner():
    storage = torch.arange(48, dtype=torch.uint8)
    matrix = storage.view(6, 8)
    weights = PackedWeight(matrix[1:4, ::2], matrix[4:, :4])
    roots = {'packed': weights, 'alias': weights.values}
    catalog = rebuild_weights_metadata(roots)
    entries = {t['name']: t for t in catalog['tensors']}
    assert entries['packed.values']['shape'] == [3, 4]
    assert entries['packed.values']['stride'] == [8, 2]
    assert entries['packed.values']['address'] == storage.data_ptr() + 8
    assert entries['packed.values']['address'] == entries['alias']['address']
    assert entries['packed.values']['extent'] == 23
    assert entries['packed.scales']['address'] == storage.data_ptr() + 32
    assert entries['packed.values']['layout_type'].endswith('.PackedWeight')
    assert entries['packed.values']['dtype'] == 'uint8'
    assert not catalog['unavailable']


def test_tensor_metadata_bounds_cycles_and_defers_storage_resolution():
    stored = torch.ones(8)
    disposable = torch.zeros(8)
    roots = {'stored': stored, 'unused': torch.ones(3), 'scratch': disposable,
             'meta': torch.empty(3, device='meta'), 'empty': torch.empty(0)}
    roots['cycle'] = roots
    catalog = rebuild_weights_metadata(roots)
    assert {entry['name'] for entry in catalog['tensors']} == {
        'scratch', 'stored', 'unused', "cycle['scratch']", "cycle['stored']", "cycle['unused']"}
    assert set(catalog['unavailable']) == {'meta', 'empty', "cycle['meta']", "cycle['empty']"}
    assert len(list(weight_views(roots))) < 20


@pytest.mark.parametrize('change', ['version', 'dtype', 'shape', 'duplicate', 'dynamic', 'peer', 'missing'])
def test_invalid_weight_catalog_cannot_be_used_for_preparation(change):
    tensor = torch.ones(4)
    catalog = proto.RebuildWeights(version=1)
    catalog.tensors.add(name='w', dtype='float32', element_bytes=4, shape=[4], stride=[1],
                        storage=proto.Reference(allocation='weights'))
    bundle = proto.BundleIndex(allocations=[proto.Allocation(name='weights', nbytes=tensor.nbytes)],
                              rebuild_weights=catalog)
    entry = bundle.rebuild_weights.tensors[0]
    if change == 'version': bundle.rebuild_weights.version = 99
    elif change == 'dtype': entry.dtype = 'unknown'
    elif change == 'shape': entry.shape.append(3)
    elif change == 'duplicate': bundle.rebuild_weights.tensors.add().CopyFrom(entry)
    elif change == 'dynamic': bundle.allocations[0].sizing.maximum = 1024
    elif change == 'peer': bundle.allocations[0].memory = proto.PEER
    else: entry.storage.allocation = 'absent'
    with pytest.raises(ValueError): validate_weights(bundle)


def test_signed_manifest_survives_rank_combination_and_conversion(native, tmp_path):
    roots = []
    for rank in range(2):
        source = tmp_path / f'weights-{rank}'
        source.write_bytes(bytes([rank + 1]) * 64)
        bundle = proto.BundleIndex(allocations=[proto.Allocation(name='weights', nbytes=64, file='weight-bytes')])
        bundle.resources['weight-bytes'].CopyFrom(proto.Span(nbytes=64))
        bundle.rebuild_weights.version = 1
        bundle.rebuild_weights.tensors.add(name='layers[0].weight', dtype='uint8', element_bytes=1,
            shape=[8, 8], stride=[8, 1], layout_type='packed', storage={'allocation': 'weights'})
        index = proto.ModelIndex(version=4, ranks=[bundle])
        if rank == 0:
            index.rebuild.CopyFrom(proto.RebuildManifest(version=1, recipe='test', source_identity='a'*64,
                checkpoint_repository='test/weights', checkpoint_revision='b'*40, tensor_parallel=2))
        path = tmp_path / f'rank-{rank}.json'
        write_sources(path, index, [(0, source, 0, 64)])
        roots.append(path)
    package = tmp_path / 'package.safetensors'
    combine(package, roots, [], publisher=publisher(), cluster=(2, 'test-export'))
    index, _ = read_index(package)
    assert index.rebuild.recipe == 'test' and index.rebuild.tensor_parallel == 2
    assert len(index.ranks) == 2
    for rank in index.ranks:
        validate_weights(rank)
        assert rank.rebuild_weights.tensors[0].name == 'layers[0].weight'
    with package.open('rb') as f:
        header, _, envelope, layout = read_container(f)
    assert header['flare.payload']['shape'] == [0]
    assert all((package.parent / name).stat().st_size <= 64_000_000_000
               for parts in layout['files'].values() for _, name in parts if name)
    assert_native(native, package)
    converted = tmp_path / 'converted' / 'package.safetensors'
    convert(package, converted, profiles=[profile()], cache=tmp_path / 'cache')
    with converted.open('rb') as f:
        assert read_container(f)[2] == envelope
    assert read_index(converted)[0] == index
    assert_native(native, converted)

    from ignition.preparation.resign import resign
    development = fixtures.development_publisher()
    development_path = tmp_path / 'development.safetensors'
    combine(development_path, roots, [], publisher=development, cluster=(2, 'test-export'))
    signed_path = tmp_path / 'production.safetensors'
    resign(development_path, signed_path, publisher=publisher(),
           development_profile=fixtures.development_profile(development))
    signed_index, _ = read_index(signed_path)
    assert signed_index.rebuild == index.rebuild
    for actual, expected in zip(signed_index.ranks, index.ranks, strict=True):
        assert actual.rebuild_weights == expected.rebuild_weights
    assert_native(native, signed_path)


def test_manifest_is_authenticated_with_shared_and_rank_metadata(native, tmp_path):
    index = proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])
    index.rebuild.version = 1
    index.rebuild.recipe = 'signed-recipe'
    from ignition.preparation.package_file import publish
    path = tmp_path / 'model.safetensors'
    publish(path, index, [], publisher=publisher())
    assert_native(native, path)
    raw = path.read_bytes()
    assert raw.count(b'signed-recipe') == 1
    path.write_bytes(raw.replace(b'signed-recipe', b'forged-recipe'))
    result = fixtures.read(native, path, weights=True)
    assert result.returncode != 0


def test_absent_catalog_is_not_inferred_from_allocation_names():
    bundle = proto.BundleIndex.FromString(b'\x12\x0b\x0a\x07weights\x10\x40')
    assert not bundle.HasField('rebuild_weights')
    validate_weights(bundle)
    assert bundle.allocations[0].nbytes == 64
