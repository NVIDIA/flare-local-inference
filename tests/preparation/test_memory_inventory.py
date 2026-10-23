"""Physical memory attribution conserves bytes across aliases and virtual capacities."""

from io import StringIO
import os
from types import SimpleNamespace

import pytest
from rich.console import Console

from ignition.preparation.memory_inventory import Claim, attribute, collect_inventory, owner_claims, print_inventory


def test_aliases_partition_backing_without_inflating_or_losing_holes():
    rows = attribute([(100, 200), (180, 250), (400, 500)], [
        Claim(100, 200, "weights", "weights", priority=30),
        Claim(150, 250, "scratch", "pool", priority=80),
        Claim(150, 190, "", "vision.qkv", "vision.py:5", tensor=True),
        Claim(150, 210, "", "decode.hidden", "decode.py:7", tensor=True),
        Claim(220, 250, "slack", "padding", priority=100),
        Claim(400, 1 << 40, "cache", "paged KV", priority=110),
    ])
    assert sum(row['nbytes'] for row in rows) == 250
    assert {kind: sum(row['nbytes'] for row in rows if row['category'] == kind)
            for kind in ('weights', 'scratch', 'slack', 'cache')} == {
                'weights': 50, 'scratch': 70, 'slack': 30, 'cache': 100}
    assert next(row for row in rows if row['owner'] == 'vision.qkv')['nbytes'] == 40
    assert next(row for row in rows if row['owner'] == 'decode.hidden')['nbytes'] == 20


def test_unknown_bytes_and_known_owners_are_distinct():
    rows = attribute([(0, 30)], [Claim(10, 20, '', 'features.arena', 'vision.py:1', tensor=True)])
    assert rows == [dict(category='Other owned tensors', owner='features.arena', source='vision.py:1', nbytes=10),
                    dict(category='Unknown storage', owner='unattributed', source='', nbytes=20)]


def test_compiler_fake_tensors_do_not_claim_physical_addresses():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        virtual = torch.empty(1024, device='cuda:0')
    prepared = SimpleNamespace(device=0, resources=(virtual,), buffers={})
    assert owner_claims(prepared) == []


def test_shared_owner_graph_does_not_queue_duplicate_paths():
    import tracemalloc
    import torch

    shared = SimpleNamespace(value=torch.empty(1))
    parents = [[shared] * 1000 for _ in range(1000)]
    prepared = SimpleNamespace(device=0, resources=(parents,), buffers={})
    tracemalloc.start()
    try:
        assert owner_claims(prepared) == []
        assert tracemalloc.get_traced_memory()[1] < 8 << 20
    finally:
        tracemalloc.stop()


def test_compiled_forward_traverses_runtime_owner_without_compiler_metadata():
    from ignition.preparation.aot import BoundForward
    from ignition.preparation.scratch import ScratchArena

    class CompilerMetadata:
        @property
        def __dict__(self):
            raise AssertionError("compiler metadata must not be traversed for storage ownership")

    arena = ScratchArena("cpu")
    arena.reserve(4096)
    forward = BoundForward(CompilerMetadata(), SimpleNamespace(workspace=arena), True,
                           pending=CompilerMetadata())
    claims = owner_claims(SimpleNamespace(device=0, resources=(forward,), buffers={}))
    claim, = claims
    assert claim.end - claim.start == 4096
    assert claim.category == "Operator scratch"
    assert ".owner" in claim.owner and ".workspace" in claim.owner


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned CUDA device')
def test_repeated_strided_views_have_one_owner_and_preserve_holes():
    import torch

    backing = torch.empty((4096, 32), device='cuda:0')
    views = [backing[:, :16] for _ in range(200)]
    prepared = SimpleNamespace(device=0, resources=tuple(views), buffers={})
    claims = owner_claims(prepared)
    assert len(claims) == 4096
    assert {claim.owner for claim in claims} == {'resources[0]'}
    start = backing.data_ptr()
    rows = attribute([(start, start + backing.nbytes)], claims)
    assert sum(row['nbytes'] for row in rows if row['owner'] == 'resources[0]') == backing.nbytes // 2
    assert sum(row['nbytes'] for row in rows if row['category'] == 'Unknown storage') == backing.nbytes // 2


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned CUDA device')
def test_overlapping_views_bound_claims_and_preserve_first_owner():
    import torch

    backing = torch.empty((4096, 32), device='cuda:0')
    first, second = backing[:, :16], backing[1024:3072, 8:24]
    views = [first, second, *(backing[:rows] for rows in range(128, 4097, 128))]
    prepared = SimpleNamespace(device=0, resources=tuple(views), buffers={})
    claims = owner_claims(prepared)
    ordered = sorted((claim.start, claim.end) for claim in claims)
    assert all(end <= start for (_, end), (start, _) in zip(ordered, ordered[1:]))
    assert len(claims) <= 3 * backing.shape[0]
    rows = attribute([(backing.data_ptr(), backing.data_ptr() + backing.nbytes)], claims)
    sizes = {row['owner']: row['nbytes'] for row in rows}
    assert sizes['resources[0]'] == first.nbytes
    assert sizes['resources[1]'] == 2048 * 8 * backing.element_size()
    assert sum(sizes.values()) == backing.nbytes
    assert not any(row['category'] == 'Unknown storage' for row in rows)


def test_inventory_separates_physical_cache_host_peers_and_package_alignment(monkeypatch):
    import flare_heliopause
    import torch
    from ignition.preparation import memory_inventory
    from ignition.preparation.runtime_package import Allocation

    monkeypatch.setattr(flare_heliopause, 'memory_inventory', lambda: dict(
        device=0, total_bytes=10000, free_bytes=8000, mappings=[(100, 300)]))
    monkeypatch.setattr(torch.cuda, 'memory_snapshot', lambda: [])
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 300)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 300)
    monkeypatch.setattr(memory_inventory, 'owner_claims', lambda prepared: [
        Claim(100, 200, '', 'layer.weight', 'weights.py:7', tensor=True)])
    prepared = SimpleNamespace(export_zero_ranges=[(200, 200)], buffers={}, graphs={'decode': None})
    storage = SimpleNamespace(rank=0, device=0, numeric_storage=SimpleNamespace(ranges=lambda: [(100, 200)]),
        graph_pool=SimpleNamespace(id=(1, 1)), dynamic_regions=[SimpleNamespace(
            address=1000, nbytes=200, extent=1 << 40, name='KV')],
        peer_storage=SimpleNamespace(owned={300: 100}, imported_mappings={9000: [(9000, 500)]}),
        host_regions=[SimpleNamespace(extent=4000, closed=False)])
    package = SimpleNamespace(allocations=[Allocation('fixed', 100, 300),
        Allocation('external', 500, 100), Allocation('cache', 1000, 1 << 40, sizing={'capacity': 'KV'}),
        Allocation('peer', 9000, 500, memory='peer'), Allocation('host', 10000, 4000, memory='mapped_host')],
        module_globals=[(700, 10, 'kernel.constant')],
        export_counts={'kernels': 1, 'modules': 1, 'cubin_bytes': 3})
    report = collect_inventory(prepared, storage, package, external=[(520, 20)])
    assert report['fixed_package_bytes'] == 400
    assert report['cache_backing_bytes'] == 200
    assert report['package_padding_bytes'] == 80
    assert report['tracked_physical_bytes'] == 530
    assert report['device_unattributed_bytes'] == 1470
    assert sum(row['nbytes'] for row in report['entries']) == 330
    assert sum(row['nbytes'] for row in report['preparation_entries']) == 200
    assert report['mapped_host_bytes'] == 4000
    assert report['imported_peer_bytes'] == 500
    output = StringIO()
    print_inventory([report], console=Console(file=output, width=250))
    text = output.getvalue()
    assert 'layer.weight' in text and 'weights.py:7' in text
    assert 'Weights / numeric constants' in text and 'Preparation cache backing' in text
    assert 'other processes' in text
    assert 'not resident VRAM' in text


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned CUDA device')
def test_cuda_inventory_tracks_shared_scratch_and_bounded_cache():
    import subprocess
    import sys

    assert os.environ.get('CUDA_VISIBLE_DEVICES')
    subprocess.run([sys.executable, '-c', '''
from dataclasses import dataclass
from types import SimpleNamespace
import torch
import flare_heliopause
from ignition.preparation.memory_inventory import collect_inventory, owner_claims
from ignition.preparation.package_storage import PackageStorage
from ignition.preparation.runtime_package import Allocation
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tensor_metadata import record_numeric_tensor

# A dataclass retained through a closure exercises owners outside Ignition modules.
@dataclass
class Forward:
    hidden: torch.Tensor

torch.cuda.set_device(0)
with PackageStorage() as storage:
    weight = torch.ones(1024, device='cuda')
    record_numeric_tensor(weight)
    pool = GraphBufferPool('cuda:0', chunk_bytes=4096)
    prefill = Forward(pool.cursor().zeros(1024, torch.float32))
    decode = Forward(pool.cursor().zeros(8, torch.float32))
    def forward(owner=prefill):
        return owner.hidden
    arena = ScratchArena('cuda:0')
    arena.reserve(512)
    cache = storage.device_allocate('KV', (1 << 20, 256), torch.uint8,
                                    group='kv_pages', maximum=1 << 20, units=4)
    region, = storage.dynamic_regions
    prepared = SimpleNamespace(resources=(weight, pool, forward, decode, arena),
        buffers={'cache': cache}, device=0, export_zero_ranges=(), graphs={'prefill': None, 'decode': None})
    allocations = [Allocation(str(i), a.address, a.nbytes) for i, a in enumerate(storage.mappings.values())]
    allocations.append(Allocation('KV', region.address, region.extent, sizing={'capacity': 'kv_pages'}))
    package = SimpleNamespace(allocations=allocations, module_globals=[],
        export_counts={'kernels': 0, 'modules': 0, 'cubin_bytes': 0})
    report = collect_inventory(prepared, storage, package)
    def size(category, field='entries'):
        return sum(row['nbytes'] for row in report[field] if row['category'] == category)
    assert size('Weights / numeric constants') == weight.nbytes
    assert size('Graph scratch pools') == pool.nbytes == 4096
    assert size('Operator scratch') == arena.nbytes == 512
    assert size('Preparation cache backing', 'preparation_entries') == region.nbytes < region.extent
    assert report['tracked_physical_bytes'] == sum(size for _, size in flare_heliopause.memory_inventory()['mappings']) + region.nbytes
    claims = owner_claims(prepared)
    assert any('defaults[0]<Forward>.hidden' in claim.owner for claim in claims)
    assert any('scratch.py:' in claim.source for claim in claims)
    assert prefill.hidden.data_ptr() == decode.hidden.data_ptr()
'''], check=True, timeout=60)
