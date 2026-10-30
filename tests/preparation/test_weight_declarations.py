"""Semantic weight declarations: grouping, aliases, tuned and unclassified views."""

from dataclasses import dataclass
import itertools

import pytest
import torch

from ignition.preparation.rebuild_manifest import (STABLE, TUNED, declare_container_fields,
    declare_weights, declared_views, rebuild_weights_metadata, weight_payload_declaration,
    weight_payload_request)


@dataclass(frozen=True)
class Layer:
    attention: dict
    mlp: torch.Tensor
    scratch: torch.Tensor | None = None


LAYER_KEY = f'{Layer.__module__}.{Layer.__qualname__}'
declare_container_fields(LAYER_KEY, ('attention', 'mlp'))


@dataclass(frozen=True)
class Model:
    embedding: torch.Tensor
    head: torch.Tensor
    layers: tuple
    allocated_bytes: int
    vision: dict | None = None
    draft: torch.Tensor | None = None
    tree: object | None = None
    projections: dict | None = None


def build():
    layers = tuple(Layer({'q': torch.zeros(4, 4)}, torch.zeros(4, 4),
                         torch.zeros(2)) for _ in range(2))
    embedding = torch.zeros(8, 4)
    return Model(embedding, embedding, layers, 0, {'patch': torch.zeros(2)},
                 torch.zeros(2, 2))


COMPONENTS = ('embedding', 'head', 'vision', 'draft')


def test_layers_and_components_group_by_declaration():
    views = list(declared_views(declare_weights(build(), components=COMPONENTS)))
    groups = {name: group for name, _owner, _tensor, group, _storage in views}
    assert groups['weights.embedding'] == 'misc'
    assert groups['weights.layers[0].attention[\'q\']'] == 'layer-000'
    assert groups['weights.layers[1].mlp'] == 'layer-001'
    assert groups['weights.vision[\'patch\']'] == 'misc'
    assert groups['weights.draft'] == 'misc'
    assert all(storage == STABLE for _n, _o, _t, _g, storage in views)


def test_undeclared_fields_are_not_weight_views():
    views = list(declared_views(declare_weights(build(), components=('embedding',), layers=())))
    names = {name for name, _owner, _tensor, _group, _storage in views}
    assert names == {'weights.embedding'}


def test_tuned_group_is_declared_not_inferred():
    views = list(declared_views(declare_weights(build(), components=COMPONENTS, tuned=('draft',))))
    storage = {name: kind for name, _owner, _tensor, _group, kind in views}
    assert storage['weights.draft'] == TUNED
    assert storage['weights.embedding'] == STABLE
    assert storage['weights.layers[0].attention[\'q\']'] == STABLE


def test_declared_alias_is_recorded_once_as_a_binding():
    declaration = declare_weights(build(), components=COMPONENTS,
        aliases=(('weights.head', 'weights.embedding'),))
    views = {view['name']: view for view in weight_payload_declaration(declaration)['views']}
    assert views['weights.head']['alias_of'] == 'weights.embedding'
    assert views['weights.head']['address'] == views['weights.embedding']['address']
    assert 'alias_of' not in views['weights.embedding']


def test_alias_requires_identical_declared_storage():
    weights = build()
    with pytest.raises(ValueError):
        weight_payload_declaration(declare_weights(weights, components=COMPONENTS,
            aliases=(('weights.head', 'weights.vision[\'patch\']'),)))
    with pytest.raises(ValueError):
        weight_payload_declaration(declare_weights(weights, components=COMPONENTS,
            aliases=(('weights.head', 'weights.missing'),)))


def test_declared_container_fields_exclude_runtime_scratch():
    declare_container_fields('tests.Scratchy', ('w13', 'w2'))

    class Scratchy:
        def __init__(self):
            self.w13 = torch.zeros(2)
            self.w2 = torch.zeros(2)
            self.w13_scale_scratch = torch.zeros(2)

    Scratchy.__module__ = 'tests'
    Scratchy.__qualname__ = 'Scratchy'

    @dataclass(frozen=True)
    class Holder:
        experts: object

    views = list(declared_views(declare_weights(Holder(Scratchy()), components=('experts',))))
    names = {name for name, _owner, _tensor, _group, _storage in views}
    assert names == {'weights.experts.w13', 'weights.experts.w2'}


def test_declared_catalog_matches_legacy_schema():
    catalog = rebuild_weights_metadata(declare_weights(build(), components=COMPONENTS))
    assert catalog['version'] == 1
    assert set(catalog) == {'version', 'tensors', 'unavailable'}
    entry = catalog['tensors'][0]
    assert set(entry) == {'name', 'dtype', 'element_bytes', 'shape', 'stride',
                          'layout_type', 'address', 'extent'}
    assert catalog['unavailable'] == []


def test_legacy_root_mapping_still_walks_generically():
    catalog = rebuild_weights_metadata({'weights': build()})
    names = {entry['name'] for entry in catalog['tensors']}
    assert 'weights.layers[0].attention[\'q\']' in names


def test_meta_tensors_are_unavailable():
    weights = build()
    with torch.device('meta'):
        meta = torch.zeros(2, 2)
    replacement = Model(weights.embedding, weights.head,
                        (Layer({'q': meta}, weights.layers[0].mlp),), 0)
    catalog = rebuild_weights_metadata(declare_weights(replacement,
        components=('embedding', 'head', 'layers')))
    assert 'weights.layers[0].attention[\'q\']' in catalog['unavailable']


def test_declarations_require_components():
    with pytest.raises(ValueError):
        declare_weights(build(), components=())


def test_weight_payload_request_is_only_for_declarations():
    assert weight_payload_request({'weights': build()}) is None
    assert weight_payload_request(None) is None
    request = weight_payload_request(declare_weights(build(), components=COMPONENTS))
    assert request['version'] == 1
    assert {view['group'] for view in request['views']} >= {'misc', 'layer-000'}
    assert 'weight_payload' not in rebuild_weights_metadata({'weights': build()})


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Parameter(torch.zeros(4, 4))
        self.q_norm = torch.nn.Parameter(torch.zeros(4))
        self.register_buffer('inv_freq', torch.zeros(4), persistent=False)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.input_layernorm = torch.nn.Parameter(torch.zeros(4))


class Tree(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Parameter(torch.zeros(8, 4))
        self.layers = torch.nn.ModuleList(Decoder() for _ in range(2))


def tree_model():
    return Model(torch.zeros(2), torch.zeros(2), (), 0, tree=Tree(), projections={})


def test_dotted_layers_group_attention_norms_and_skip_module_buffers():
    views = list(declared_views(declare_weights(tree_model(),
        components={'tree.embed_tokens': 'embedding'},
        layers=('tree.layers',))))
    groups = {name: group for name, _owner, _tensor, group, _storage in views}
    assert groups['weights.tree.embed_tokens'] == 'misc'
    assert groups['weights.tree.layers[0].self_attn.q_proj'] == 'layer-000'
    assert groups['weights.tree.layers[0].self_attn.q_norm'] == 'layer-000'
    assert groups['weights.tree.layers[1].input_layernorm'] == 'layer-001'
    assert not any('inv_freq' in name for name in groups)


def test_mapping_component_is_walked_whole():
    weights = tree_model()
    weights.projections['layers.0.q_proj'] = torch.zeros(2, 2)
    views = list(declared_views(declare_weights(weights,
        components={'projections': 'projections'}, layers=())))
    groups = {name: group for name, _owner, _tensor, group, _storage in views}
    assert groups["weights.projections['layers.0.q_proj']"] == 'misc'


def test_unresolved_component_path_fails_loudly():
    with pytest.raises(ValueError):
        list(declared_views(declare_weights(tree_model(),
            components={'tree.missing.weight': 'embedding'}, layers=())))


def test_thin_declaration_omits_unmaterialized_views():
    weights = build()
    with torch.device('meta'):
        meta = torch.zeros(2, 2)
    replacement = Model(weights.embedding, weights.head,
                        (Layer({'q': meta}, weights.layers[0].mlp),), 0)
    declaration = declare_weights(replacement, components=('embedding', 'layers'))
    thin = weight_payload_declaration(declaration)
    assert all(view['name'] != "weights.layers[0].attention['q']" for view in thin['views'])
    catalog = rebuild_weights_metadata(declaration)
    assert "weights.layers[0].attention['q']" in catalog['unavailable']


def test_declared_layer_scratch_field_is_not_weight_storage():
    views = list(declared_views(declare_weights(build(), components=COMPONENTS)))
    names = {name for name, _owner, _tensor, _group, _storage in views}
    assert 'weights.layers[0].scratch' not in names
    assert 'weights.layers[0].mlp' in names


def test_registered_scratch_field_becomes_weight_storage():
    declare_container_fields(LAYER_KEY, ('attention', 'mlp', 'scratch'))
    try:
        views = list(declared_views(declare_weights(build(), components=COMPONENTS)))
        names = {name for name, _owner, _tensor, _group, _storage in views}
        assert 'weights.layers[0].scratch' in names
    finally:
        declare_container_fields(LAYER_KEY, ('attention', 'mlp'))


def test_provenance_ranges_cover_every_axis_in_order():
    import gc
    from ignition.preparation import provenance

    shape = (8, 4, 2)
    assert provenance.ranges_of(None, shape) == ((0, 8), (0, 4), (0, 2))
    assert provenance.ranges_of((slice(2, 6),), shape) == ((2, 6), (0, 4), (0, 2))
    assert provenance.ranges_of((slice(None), slice(1, 3)), shape) == (
        (0, 8), (1, 3), (0, 2))
    assert provenance.ranges_of(slice(0, 0), shape) == ((0, 0), (0, 4), (0, 2))
    # Unsupported indexing is explicitly unprovenanced, never guessed.
    assert provenance.ranges_of((slice(0, 8, 2),), shape) is None
    assert provenance.ranges_of((slice(-1, 2),), shape) is None
    assert provenance.ranges_of((slice(0, 9),), shape) is None
    assert provenance.ranges_of((3,), shape) is None
    assert provenance.ranges_of((slice(0, 8), slice(0, 4), slice(0, 2), slice(0, 1)), shape) is None


def test_provenance_follows_tensor_lifetime_and_identity():
    import gc
    from ignition.preparation import provenance

    tensor = torch.zeros(4, 4)
    provenance.record(tensor, tensor_name="model.embed.weight",
                      ranges=((0, 4), (0, 4)), packing="bfloat16")
    assert provenance.lookup(tensor)["sources"][0]["ranges"] == ((0, 4), (0, 4))
    del tensor
    gc.collect()
    assert not any(entry for entry in provenance._ENTRIES.values()
                   if entry["reference"]() is None)
    replacement = torch.zeros(4, 4)
    assert provenance.lookup(replacement) is None


def test_projection_extent_records_full_column_range():
    import gc
    from ignition.preparation import provenance

    tensor = torch.zeros(6, 4)
    provenance.record(tensor, tensor_name="layers.0.mlp.down_proj.weight",
                      ranges=provenance.ranges_of((slice(None), slice(1, 3)), (6, 4)),
                      packing="bfloat16")
    recorded = provenance.lookup(tensor)
    assert recorded["sources"][0]["ranges"] == ((0, 6), (1, 3))
    assert recorded["packing"] == "bfloat16"
    del tensor, recorded
    gc.collect()


def test_fused_projection_keeps_one_ordered_source_per_part():
    import gc
    from ignition.preparation import provenance

    left = torch.zeros(4, 4)
    right = torch.zeros(4, 4)
    provenance.record(left, tensor_name="layers.0.gate_proj.weight",
                      ranges=((0, 4), (0, 4)), packing="bfloat16")
    provenance.record(right, tensor_name="layers.0.up_proj.weight",
                      ranges=((0, 4), (0, 4)), packing="bfloat16")
    fused = torch.zeros(8, 4)
    provenance.record_sources(
        fused,
        [source for tensor in (left, right)
         for source in provenance.lookup(tensor)["sources"]],
        packing="bfloat16", concatenation_axis=0)
    recorded = provenance.lookup(fused)
    assert [source["tensor"] for source in recorded["sources"]] == [
        "layers.0.gate_proj.weight", "layers.0.up_proj.weight"]
    assert all(source["concatenation_axis"] == 0 for source in recorded["sources"])
    assert all(source["ranges"] == ((0, 4), (0, 4)) for source in recorded["sources"])
    del left, right, fused, recorded
    gc.collect()


def test_provenance_records_bounded_swizzle_byte_piece():
    import gc
    from ignition.preparation import provenance

    rows, columns = 2304, 448
    source = torch.arange(rows * columns, dtype=torch.uint8).reshape(rows, columns)
    owner = (source.reshape(rows // 128, 4, 32, columns // 4, 4)
             .permute(0, 3, 2, 1, 4).contiguous().reshape(rows, columns))
    piece = dict(source_offset=0, destination_offset=0, contiguous_bytes=4,
                 extents=(9, 4, 32, 112),
                 source_strides=(57344, 14336, 448, 4),
                 destination_strides=(57344, 4, 16, 512))
    provenance.record_sources(
        owner,
        [dict(tensor="model.experts.0.up_proj.weight_scale",
              ranges=((0, rows), (0, columns)), pieces=(piece,))],
        packing="nvfp4", statistics_domain="shard")
    recorded = provenance.lookup(owner)
    assert recorded["sources"][0]["pieces"] == (piece,)
    flat_source = source.reshape(-1)
    flat_owner = owner.reshape(-1)
    for rb, a, b, cb in itertools.product(range(9), range(4), range(32), range(112)):
        source_base = rb * 57344 + a * 14336 + b * 448 + cb * 4
        destination_base = rb * 57344 + a * 4 + b * 16 + cb * 512
        for lane in range(4):
            assert flat_owner[destination_base + lane] == flat_source[source_base + lane]
    del owner, source, recorded
    gc.collect()


def test_q8_0_producer_does_not_claim_iq2_metadata():
    import gc
    from types import SimpleNamespace
    from ignition.components import weights as weight_components
    from ignition.preparation import provenance

    owners = {name: torch.zeros(8, dtype=torch.uint8)
              for name in ('w13', 'w13_scale', 'w2', 'w2_scale')}
    prepared = SimpleNamespace(**owners)
    weight_components._record_iq2_expert_provenance(
        prepared, prefixes=('e0',), sources={}, projection_names=('up_proj', 'down_proj'),
        weight_name='weight', first=0, last=8, hidden_size=8, local=8, codec='q8_0')
    assert all(provenance.lookup(tensor) is None for tensor in owners.values())
    del owners, prepared
    gc.collect()


def test_provenance_records_piece_encoding_without_bytes():
    import gc
    from ignition.preparation import provenance

    owner = torch.zeros(4, dtype=torch.uint8)
    provenance.record_sources(owner, [dict(
        tensor="experts.0.up_proj.weight", ranges=((0, 4), (0, 4)),
        piece_encoding="w4a16_mma_k16_n64_v1")])
    recorded = provenance.lookup(owner)
    assert recorded["sources"][0]["piece_encoding"] == "w4a16_mma_k16_n64_v1"
    assert recorded["sources"][0]["pieces"] == ()
    del owner, recorded
    gc.collect()


def test_fp4_expert_producer_maps_retained_owners_to_checkpoint_bytes():
    from types import SimpleNamespace
    from b12x._lib.intrinsics.reference import swizzle_block_scale as swizzle
    from ignition.components import weights as weight_components
    from ignition.preparation import provenance

    hidden, local, group = 128, 128, 16
    up = torch.arange(local * (hidden // 2), dtype=torch.uint8).reshape(local, hidden // 2)
    gate = torch.arange(local * (hidden // 2), dtype=torch.uint8).reshape(local, hidden // 2) + 7
    down = torch.arange(hidden * (local // 2), dtype=torch.uint8).reshape(hidden, local // 2)
    up_scale = torch.arange(local * (hidden // group), dtype=torch.uint8).reshape(local, hidden // group)
    gate_scale = torch.arange(local * (hidden // group), dtype=torch.uint8).reshape(local, hidden // group) + 3
    down_scale = torch.arange(hidden * (local // group), dtype=torch.uint8).reshape(hidden, local // group)
    w13 = torch.cat((up, gate), dim=0).unsqueeze(0)
    w2 = down.unsqueeze(0)
    s13 = swizzle(torch.cat((up_scale, gate_scale), dim=0)).unsqueeze(0)
    s2 = swizzle(down_scale).unsqueeze(0)
    prepared = SimpleNamespace(_impl=SimpleNamespace(
        w1_fp4=w13, w2_fp4=w2, w1_blockscale=s13, w2_blockscale=s2))
    prefix = 'e0'
    weight_components._record_fp4_expert_owner_provenance(
        prepared, prefixes=(prefix,),
        sources={f'{prefix}.{name}.{suffix}': value
                 for name, suffix, value in (
                     ('up_proj', 'weight', up), ('gate_proj', 'weight', gate),
                     ('down_proj', 'weight', down),
                     ('up_proj', 'weight_scale', up_scale),
                     ('gate_proj', 'weight_scale', gate_scale),
                     ('down_proj', 'weight_scale', down_scale))},
        projection_names=('up_proj', 'gate_proj', 'down_proj'),
        weight_name='weight', scale_name='weight_scale', first=0, last=local,
        hidden_size=hidden, local=local, group=group, format='nvfp4')
    for owner, count in ((w13, 2), (w2, 1), (s13, 2), (s2, 1)):
        recorded = provenance.lookup(owner)
        assert recorded is not None
        assert len(recorded['sources']) == count
    w13_piece = provenance.lookup(w13)['sources'][0]['pieces'][0]
    assert w13_piece['contiguous_bytes'] == hidden // 2
    assert w13_piece['extents'] == (local,)
    assert w13_piece['destination_offset'] + local * w13_piece['destination_strides'][0] <= w13.numel()
    for row in (0, local - 1):
        source_offset = w13_piece['source_offset'] + row * w13_piece['source_strides'][0]
        destination_offset = (w13_piece['destination_offset']
                              + row * w13_piece['destination_strides'][0])
        assert (w13.reshape(-1)[destination_offset:destination_offset + hidden // 2].tolist()
                == up.reshape(-1)[source_offset:source_offset + hidden // 2].tolist())
    scale_piece = provenance.lookup(s13)['sources'][0]['pieces'][0]
    assert scale_piece['destination_strides'] == (1024, 4, 16, 512)

    # A gate-first w31 owner is flipped in place by b12x before it is retained;
    # the recorded source order must follow the retained [up; gate] bytes.
    from b12x.moe.fused_moe import _impl as b12x_impl
    key = (w13.data_ptr(), s13.data_ptr())
    b12x_impl._W13_NORMALIZED_STORAGES[key] = (w13, s13)
    try:
        weight_components._record_fp4_expert_owner_provenance(
            prepared, prefixes=(prefix,),
            sources={f'{prefix}.{name}.{suffix}': value
                     for name, suffix, value in (
                         ('up_proj', 'weight', up), ('gate_proj', 'weight', gate),
                         ('down_proj', 'weight', down),
                         ('up_proj', 'weight_scale', up_scale),
                         ('gate_proj', 'weight_scale', gate_scale),
                         ('down_proj', 'weight_scale', down_scale))},
            projection_names=('gate_proj', 'up_proj', 'down_proj'),
            weight_name='weight', scale_name='weight_scale', first=0, last=local,
            hidden_size=hidden, local=local, group=group, format='nvfp4',
            w13_layout='w31')
        flipped = provenance.lookup(w13)
        assert flipped['sources'][0]['tensor'].endswith('up_proj.weight')
        assert flipped['sources'][1]['tensor'].endswith('gate_proj.weight')
    finally:
        b12x_impl._W13_NORMALIZED_STORAGES.pop(key, None)
    for rb, a, b, cb in ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
        source_base = (rb * scale_piece['source_strides'][0]
                       + a * scale_piece['source_strides'][1]
                       + b * scale_piece['source_strides'][2]
                       + cb * scale_piece['source_strides'][3])
        destination_base = (scale_piece['destination_offset']
                            + rb * scale_piece['destination_strides'][0]
                            + a * scale_piece['destination_strides'][1]
                            + b * scale_piece['destination_strides'][2]
                            + cb * scale_piece['destination_strides'][3])
        for lane in range(4):
            assert s13.reshape(-1)[destination_base + lane] == up_scale.reshape(-1)[source_base + lane]

    # A compressed scale representation can be absent or non-byte; valid
    # retained weight owners must still record independently.
    w13_only, w2_only = w13.clone(), w2.clone()
    weight_components._record_fp4_expert_owner_provenance(
        SimpleNamespace(_impl=SimpleNamespace(w1_fp4=w13_only, w2_fp4=w2_only,
                                             w1_blockscale=None, w2_blockscale=None)),
        prefixes=(prefix,),
        sources={f'{prefix}.up_proj.weight': up, f'{prefix}.gate_proj.weight': gate,
                 f'{prefix}.down_proj.weight': down},
        projection_names=('up_proj', 'gate_proj', 'down_proj'),
        weight_name='weight', scale_name='weight_scale', first=0, last=local,
        hidden_size=hidden, local=local, group=group, format='nvfp4',
        scale_compression='csf')
    assert provenance.lookup(w13_only) is not None
    assert provenance.lookup(w2_only) is not None

    # Deferred metadata is immutable source addressing, not the loader's dict.
    live = {f'{prefix}.up_proj.weight': up, f'{prefix}.gate_proj.weight': gate,
            f'{prefix}.down_proj.weight': down,
            f'{prefix}.up_proj.weight_scale': up_scale,
            f'{prefix}.gate_proj.weight_scale': gate_scale,
            f'{prefix}.down_proj.weight_scale': down_scale}
    snapshot = weight_components._snapshot_fp4_sources(live, tuple(live))
    live.clear()
    w13_snapshot, w2_snapshot = w13.clone(), w2.clone()
    weight_components._record_fp4_expert_owner_provenance(
        SimpleNamespace(_impl=SimpleNamespace(w1_fp4=w13_snapshot, w2_fp4=w2_snapshot,
                                             w1_blockscale=None, w2_blockscale=None)),
        prefixes=(prefix,), sources=snapshot,
        projection_names=('up_proj', 'gate_proj', 'down_proj'),
        weight_name='weight', scale_name='weight_scale', first=0, last=local,
        hidden_size=hidden, local=local, group=group, format='nvfp4')
    assert provenance.lookup(w13_snapshot) is not None
    assert provenance.lookup(w2_snapshot) is not None


def qwen_dense_weights(first_layer_seed=1.0):
    from types import SimpleNamespace

    class LayerModule(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.q_proj = torch.nn.Parameter(torch.full((2, 2), float(seed)))
            self.q_norm = torch.nn.Parameter(torch.zeros(2))

    tree = torch.nn.Module()
    tree.embed_tokens = torch.nn.Parameter(torch.zeros(4, 2))
    tree.norm = torch.nn.Parameter(torch.zeros(2))
    tree.layers = torch.nn.ModuleList([LayerModule(1), LayerModule(2)])
    return SimpleNamespace(
        tree=tree, head=torch.zeros(2), draft=None, draft_head=None,
        projections={
            'layers.0.self_attn.q_proj': torch.full((2, 2), first_layer_seed),
            'layers.1.self_attn.q_proj': torch.full((2, 2), 2.0),
        })


def declared_group_bytes(weights):
    from ignition.recipes.qwen38_dense.bootstrap import _declared_weight_roots

    roots, components = _declared_weight_roots(weights)
    groups = {}
    for _name, _owner, tensor, group, _storage in declared_views(
            declare_weights(roots, components=components, layers=())):
        groups.setdefault(group, []).append(tensor.detach().clone().reshape(-1))
    return {group: torch.cat(values).clone() for group, values in groups.items()}


def test_qwen_dense_layer_projections_are_grouped_per_layer():
    from ignition.recipes.qwen38_dense.bootstrap import _declared_weight_roots

    weights = qwen_dense_weights()
    roots, components = _declared_weight_roots(weights)
    groups = {}
    for _name, _owner, tensor, group, _storage in declared_views(
            declare_weights(roots, components=components, layers=())):
        groups.setdefault(group, []).append(tensor.detach().clone())
    assert set(groups) == {'misc', 'layer-000', 'layer-001'}
    assert 'projections' not in groups
    def contains(group, value):
        return any(tensor.shape == value.shape and torch.equal(tensor, value)
                   for tensor in groups[group])
    assert contains('layer-000', torch.full((2, 2), 1.0))
    assert contains('layer-001', torch.full((2, 2), 2.0))
    assert not contains('layer-000', torch.full((2, 2), 2.0))
    assert not contains('layer-001', torch.full((2, 2), 1.0))


def test_qwen_dense_single_layer_change_keeps_other_layer_payloads():
    before = declared_group_bytes(qwen_dense_weights(first_layer_seed=1.0))
    after = declared_group_bytes(qwen_dense_weights(first_layer_seed=9.0))
    assert torch.equal(before['misc'], after['misc'])
    assert torch.equal(before['layer-001'], after['layer-001'])
    assert before['layer-000'].shape == after['layer-000'].shape
    assert not torch.equal(before['layer-000'], after['layer-000'])


def test_provenance_statistic_domain_reaches_the_declaration():
    """A shard-local quantization domain survives into the export declaration."""
    declaration = declare_weights({"weights": {"expert": torch.zeros(4, 4)}},
                                 components=("weights",))
    from ignition.preparation import provenance

    expert = declaration.roots["weights"]["expert"]
    provenance.record(expert, tensor_name="layers.0.experts.mlp.w1",
                      ranges=((0, 4), (0, 4)), packing="nvfp4",
                      statistics_domain="shard",
                      packing_parameters=dict(format="nvfp4", group_size=16))
    request = weight_payload_request(declaration)
    view = next(view for view in request["views"] if "expert" in view["name"])
    assert view["packing"] == "nvfp4"
    assert view["statistics_domain"] == "shard"
    assert view["packing_parameters"] == {"format": "nvfp4", "group_size": 16}

    # Without provenance the declaration records no packing or domain at all.
    plain = declare_weights({"weights": {"other": torch.zeros(2, 2)}}, components=("weights",))
    request = weight_payload_request(plain)
    view = next(view for view in request["views"] if "other" in view["name"])
    assert "statistics_domain" not in view
    assert "packing" not in view
    assert "packing_parameters" not in view


def test_nvfp4_linear_provenance_uses_packed_source_columns():
    from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight
    from ignition.components.weights import _record_nvfp4_linear_provenance
    from ignition.preparation import provenance

    # Real _load_nvfp4_linear geometry: logical columns 2048 -> stored_k 2048 ->
    # 1024 packed uint8 columns, with nonzero row and column source offsets.
    values = torch.zeros((64, 1024), dtype=torch.uint8)
    _record_nvfp4_linear_provenance(values, "layers.0.mlp.down_proj",
                                    r0=128, r1=192, c0=2048, c1=4096)
    assert values.shape == (64, 1024) and values.stride() == (1024, 1)
    owner = NVFP4LinearWeight(values, torch.zeros(256, dtype=torch.uint8),
                              torch.zeros((), dtype=torch.float32), "multiplier", 2048, 64)
    assert provenance.lookup(owner.values) is not None

    declaration = declare_weights({"weights": owner}, components=("weights",), layers=())
    views = {view["name"]: view for view in weight_payload_declaration(declaration)["views"]}
    packed = views["weights.values"]
    assert packed["sources"][0]["tensor"] == "layers.0.mlp.down_proj.weight"
    assert packed["sources"][0]["ranges"] == ((128, 192), (1024, 2048)), packed["sources"]
    assert packed["shape"] == [64, 1024] and packed["stride"] == [1024, 1]
    assert packed["packing"] == "nvfp4"
    assert packed["packing_parameters"] == {"format": "nvfp4", "group_size": 16}
    # The discarded pre-swizzle scale tensor is not an exported owner.
    assert not any("scale" in name and name.endswith("scales") for name in views)


def test_nvfp4_linear_padding_is_not_claimed_to_be_covered():
    from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight
    from ignition.components.weights import _record_nvfp4_linear_provenance

    # Logical 2032 columns pad to stored_k 2048: 1016 packed columns are sourced,
    # 8 packed columns are zero padding the record does not claim.
    values = torch.zeros((64, 1024), dtype=torch.uint8)
    _record_nvfp4_linear_provenance(values, "layers.0.mlp.up_proj",
                                    r0=0, r1=64, c0=0, c1=2032)
    owner = NVFP4LinearWeight(values, torch.zeros(256, dtype=torch.uint8),
                              torch.zeros((), dtype=torch.float32), "multiplier", 2032, 64)
    declaration = declare_weights({"weights": owner}, components=("weights",), layers=())
    view = {v["name"]: v for v in weight_payload_declaration(declaration)["views"]}["weights.values"]
    assert view["sources"][0]["ranges"] == ((0, 64), (0, 1016))
    assert view["shape"] == [64, 1024]
    assert "padding" not in view


def test_nonstandard_layer_field_uses_layer_files_and_other_weights_use_misc():
    @dataclass
    class Network:
        blocks: tuple
        output: torch.Tensor

    weights = Network((Layer({'q': torch.ones(4)}, torch.ones(4)),), torch.ones(8))
    views = list(declared_views(declare_weights(weights, components=('output',), layers=('blocks',))))
    assert {group for _, _, _, group, _ in views} == {'misc', 'layer-000'}
    assert {name for name, _, _, group, _ in views if group == 'misc'} == {'weights.output'}


def test_misc_preserves_stable_and_tuned_storage_classes():
    declaration = weight_payload_declaration(declare_weights(build(), components=COMPONENTS, tuned=('draft',)))
    misc = {view['name']: view['storage'] for view in declaration['views'] if view['group'] == 'misc'}
    assert misc == {'weights.embedding': STABLE, 'weights.head': STABLE,
                    "weights.vision['patch']": STABLE, 'weights.draft': TUNED}


def test_packed_nvfp4_declaration_describes_retained_storage():
    from b12x.gemm.blockscaled import PackedNVFP4LinearWeight
    from ignition.components import weights as _registered_weights
    from ignition.preparation import provenance

    source = torch.zeros((64, 16), dtype=torch.uint8)
    provenance.record(source, tensor_name="mlp.gate_up.weight", ranges=((0, 64), (0, 16)),
                      packing="nvfp4", packing_parameters={"format": "nvfp4", "group_size": 16})
    packed = PackedNVFP4LinearWeight(torch.zeros((64, 4), dtype=torch.int32),
                                     torch.zeros((1, 2, 64), dtype=torch.uint8), torch.ones(()), 32, 64)
    declaration = declare_weights({"projection": packed}, components=("projection",), layers=())
    views = {view["name"]: view for view in weight_payload_declaration(declaration)["views"]}
    assert set(views) == {"projection." + field for field in ("values", "scale_mma", "global_scale")}
    values = views["projection.values"]
    assert values["shape"] == [64, 4] and values["stride"] == [4, 1]
    assert "packing" not in values and "sources" not in values
    assert provenance.lookup(packed.values) is None
    assert provenance.lookup(source)["packing"] == "nvfp4"
    tensors = {name: tensor for name, _owner, tensor, _group, _storage in declared_views(declaration)}
    assert tensors["projection.values"] is packed.values
    assert tensors["projection.scale_mma"] is packed.scale_mma
    assert tensors["projection.global_scale"] is packed.global_scale
