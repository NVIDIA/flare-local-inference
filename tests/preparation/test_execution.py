from types import SimpleNamespace

import pytest

from ignition.preparation.execution import GraphSignature, PreparedExecution


def _tensor(dtype, shape):
    strides, running = [], 1
    for extent in reversed(shape):
        strides.insert(0, running)
        running *= extent
    return SimpleNamespace(dtype=f"torch.{dtype}", shape=shape, stride=lambda: tuple(strides))


def _execution(*, model_class=PreparedExecution, **changes):
    arguments = dict(device=0, graphs={"decode": object(), "prefill": object()},
        signatures={"decode": GraphSignature("decode", 1, 1, ("tokens",)),
                    "prefill": GraphSignature("prefill", 1, 16, ("tokens", "fresh"))},
        buffers={"frame_index": _tensor("int32", (1,)),
                 "submission_sequence": _tensor("int64", (1,)),
                 "completion_records": _tensor("int64", (2, 4, 8)),
                 "tokens": _tensor("int64", (16,)), "fresh": _tensor("bool", (1,))})
    return model_class(**(arguments | changes))


def test_manifest_offsets_are_stable_across_graphs(monkeypatch):
    execution = _execution(estimated_durations_ns={"decode": 123_000, "prefill": 456_000})
    spans = [(4096, 4), (8192, 8), (12288, 512), (16384, 128), (20480, 1)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = execution.manifest()
    assert manifest["context"] == 9
    assert manifest["abi_version"] == 2
    assert manifest["input_frame_bytes"] == 152
    decode, prefill = manifest["graphs"]
    assert decode["estimated_duration_ns"] == 123_000
    assert prefill["estimated_duration_ns"] == 456_000
    from ignition.preparation.runtime_package_pb2 import ExecutionGraph

    encoded = ExecutionGraph(**{key: value for key, value in decode.items() if key != "handle"})
    assert ExecutionGraph.FromString(encoded.SerializeToString()).estimated_duration_ns == 123_000
    assert decode["sampling_mode"] == prefill["sampling_mode"] == "general"
    assert decode["input_bindings"] == prefill["input_bindings"][:3]
    assert decode["input_bindings"][-1] == {"buffer_name": "tokens", "host_offset": 16, "nbytes": 128}
    assert prefill["input_bindings"][-1] == {"buffer_name": "fresh", "host_offset": 144, "nbytes": 1}
    assert all(binding["buffer_name"] != "completion_records" for graph in manifest["graphs"]
               for binding in graph["input_bindings"])
    assert manifest["buffers"][-1]["dtype"] == "bool"


def test_verification_keeps_prefix_width_separate_from_total_rows(monkeypatch):
    signature = GraphSignature('verification', 4, 6, ('tokens',), packed_token_capacity=12)
    execution = _execution(graphs={'verify':object()}, signatures={'verify':signature})
    spans = [(4096,4),(8192,8),(12288,512),(16384,128),(20480,1)]
    monkeypatch.setattr(execution, '_validate_resources', lambda:(9,[11],spans))
    graph = execution.manifest()['graphs'][0]
    assert (graph['batch_capacity'],graph['token_capacity'],graph['packed_token_capacity']) == (4,6,12)
    for capacity in (3,25):
        with pytest.raises(ValueError,match='packed verification capacity'):
            GraphSignature('verification',4,6,packed_token_capacity=capacity)
    with pytest.raises(ValueError,match='packed verification capacity'):
        GraphSignature('decode',4,1,packed_token_capacity=4)


def test_execution_manifest_preserves_declared_legacy_producer_geometry(monkeypatch):
    signatures = {f"proposal{width}": GraphSignature("proposal", 4, width, ("tokens",),
                  speculation_enabled=True) for width in (2, 4, 6)}
    execution = _execution(graphs={name: object() for name in signatures}, signatures=signatures)
    spans = [(4096, 4), (8192, 8), (12288, 512), (16384, 128), (20480, 1)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11, 12, 13], spans))
    assert [g["token_capacity"] for g in execution.manifest()["graphs"]] == [2, 4, 6]


def test_resident_binding_is_retained_for_diagnostics_and_excluded_from_export(monkeypatch):
    import flare_heliopause
    from ignition.preparation.model import PreparedModel

    catalog, binding, events = {"vocabulary": 32, "tables": []}, object(), {"ready": 123}
    calls = []

    class Resident:
        def __init__(self, supplied, *, descriptors):
            calls.append((supplied, descriptors))

        def bind(self, manifest, supplied_events):
            assert "engram" not in manifest
            assert supplied_events == events
            return binding

    monkeypatch.setattr(flare_heliopause, "EmbeddingTables", Resident)
    model = _execution(model_class=PreparedModel, server_config={}, memory_report={}, session=None,
                       embedding_catalog=catalog, export_events=events)
    spans = [(4096, 4), (8192, 8), (12288, 512), (16384, 128), (20480, 1)]
    monkeypatch.setattr(model, "_validate_resources", lambda: (9, [11, 12], spans))
    owner = model.prepare_embedding_tables(descriptors=[31, 32])
    assert calls == [(catalog, [31, 32])]
    assert model.embedding_tables is owner and model.manifest()["engram"] is binding
    assert "engram" not in model.manifest(all_variants=True)
    with pytest.raises(RuntimeError, match="existing attachment"):
        model.prepare_embedding_tables()


def test_resident_tables_reject_an_extension_without_table_registration_before_cuda(monkeypatch):
    import flare_heliopause
    from ignition.preparation.model import PreparedModel

    monkeypatch.delattr(flare_heliopause, "EmbeddingTables", raising=False)
    model = _execution(model_class=PreparedModel, server_config={}, memory_report={}, session=None,
                       embedding_catalog={"vocabulary": 32, "tables": []})
    with pytest.raises(RuntimeError, match="rebuilding the installed Heliopause extension"):
        model.prepare_embedding_tables()
    assert model.embedding_tables is None and not model.consumed


def test_speculative_manifest_exports_vector_abi(monkeypatch):
    execution = _execution()
    buffers = dict(execution.buffers)
    buffers["completion_records"] = _tensor("int64", (2, 4, 24))
    execution = _execution(buffers=buffers,
        signatures={"decode": GraphSignature("speculative", 1, 8, ("tokens",)),
                    "prefill": GraphSignature("prefill", 1, 16, ("tokens", "fresh"))})
    spans = [(4096, 4), (8192, 8), (12288, 1536), (16384, 128), (20480, 1)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11, 12], spans))
    manifest = execution.manifest()
    assert manifest["abi_version"] == 3
    assert manifest["graphs"][0]["kind"] == "speculative"
    assert manifest["graphs"][0]["token_capacity"] == 8
    assert manifest["input_frame_bytes"] == 152


def test_export_includes_ordinary_graphs_while_diagnostics_select_speculation(monkeypatch):
    signatures = {
        "decode": GraphSignature("decode", 1, 1, speculation_enabled=False),
        "prefill": GraphSignature("prefill", 1, 16, speculation_enabled=True),
        "prefill-no-mtp": GraphSignature("prefill", 1, 16, speculation_enabled=False),
        "speculative": GraphSignature("speculative", 1, 2, speculation_enabled=True),
    }
    execution = _execution(graphs=dict.fromkeys(signatures, object()), signatures=signatures)
    spans = [(4096, 4), (8192, 8), (12288, 512), (16384, 128), (20480, 1)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11, 12, 13, 14], spans))
    exported = execution.manifest(all_variants=True)["graphs"]
    assert {g["name"]: g["speculation_enabled"] for g in exported} == {
        name: signature.speculation_enabled for name, signature in signatures.items()}
    assert [g["name"] for g in execution.manifest()["graphs"]] == ["prefill", "speculative"]


@pytest.mark.parametrize("width", (None, 1, 4))
def test_diagnostic_manifest_selects_fixed_width_without_discarding_export_graphs(monkeypatch, width):
    from ignition.preparation.model import PreparedModel

    signatures = {"prefill": GraphSignature("prefill", 1, 16)}
    signatures.update({f"verify-{w}": GraphSignature("speculative", 1, w) for w in range(1, 5)})
    model = _execution(model_class=PreparedModel, graphs=dict.fromkeys(signatures, object()),
        signatures=signatures, server_config={"speculative_fixed_width": width}, memory_report={}, session=None)
    spans = [(4096, 4), (8192, 8), (12288, 512), (16384, 128), (20480, 1)]
    monkeypatch.setattr(model, "_validate_resources", lambda: (9, list(range(11, 16)), spans))
    selected = model.manifest()["graphs"]
    assert selected[0]["kind"] == "prefill"
    assert [g["token_capacity"] for g in selected[1:]] == ([1, 2, 3, 4] if width is None else sorted({1, width}))
    assert [g["token_capacity"] for g in model.manifest(all_variants=True)["graphs"][1:]] == [1, 2, 3, 4]
    assert tuple(model.graphs) == tuple(signatures)


@pytest.mark.parametrize("signature", [
    lambda: GraphSignature("callback", 1, 1),
    lambda: GraphSignature("decode", True, 1),
    lambda: GraphSignature("decode", 1, 0),
    lambda: GraphSignature("decode", 1, 1, ("tokens", "tokens")),
    lambda: GraphSignature("decode", 1, 1, ("completion_records",)),
    lambda: GraphSignature("decode", 1, 1, sampling_mode="automatic"),
    lambda: GraphSignature("speculative", 3, 4),
    lambda: GraphSignature("speculative", 1, 9),
])
def test_invalid_execution_signature_fails_before_cuda(signature):
    with pytest.raises(ValueError):
        signature()


def test_signature_references_must_cover_exact_bundle():
    with pytest.raises(ValueError, match="exactly one signature"):
        _execution(signatures={"decode": GraphSignature("decode", 1, 1)})
    with pytest.raises(ValueError, match="unregistered"):
        _execution(signatures={"decode": GraphSignature("decode", 1, 1, ("missing",)),
                               "prefill": GraphSignature("prefill", 1, 16)})


@pytest.mark.parametrize("durations", [
    {"decode": 100}, {"decode": -1, "prefill": 100},
    {"decode": True, "prefill": 100}, {"decode": 1 << 64, "prefill": 100},
])
def test_invalid_graph_durations_fail_before_cuda(durations):
    with pytest.raises(ValueError, match="duration estimates"):
        _execution(estimated_durations_ns=durations)


@pytest.mark.parametrize("prefill,width,enabled", [(64, 1, False), (64, 1, True), (512, 8, True)])
def test_mixed_manifest_preserves_both_capacities_and_decode_mode(monkeypatch, prefill, width, enabled):
    signature = GraphSignature("mixed", 4, width, ("prefill_row",),
                               speculation_enabled=enabled, prefill_capacity=prefill)
    execution = _execution()
    buffers = dict(execution.buffers)
    buffers["completion_records"] = _tensor("int64", (2, 4, 24))
    buffers["prefill_row"] = _tensor("int32", (1,))
    execution = _execution(graphs={"mixed": object()}, signatures={"mixed": signature}, buffers=buffers)
    spans = [(4096, 4), (8192, 8), (12288, 1536), (16384, 128), (20480, 1), (24576, 4)]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11], spans))
    manifest = execution.manifest(all_variants=True)
    assert manifest["abi_version"] == 3
    graph, = manifest["graphs"]
    from ignition.preparation.runtime_package_pb2 import ExecutionGraph
    encoded = ExecutionGraph(**{k: v for k, v in graph.items() if k != "handle"})
    decoded = ExecutionGraph.FromString(encoded.SerializeToString())
    assert (decoded.batch_capacity, decoded.token_capacity, decoded.prefill_capacity) == (4, width, prefill)
    assert decoded.speculation_enabled == enabled
    assert "prefill_row" in [b.buffer_name for b in decoded.input_bindings]


def test_verification_protobuf_preserves_physical_rows_and_legacy_width_bounds():
    from ignition.preparation.runtime_package_pb2 import ExecutionGraph
    legacy = ExecutionGraph.FromString(b"\x12\x0cverification\x20\x08\x28\x06")
    assert (legacy.batch_capacity, legacy.token_capacity, legacy.packed_token_capacity) == (8, 6, 0)
    graph = ExecutionGraph(kind="verification", batch_capacity=8, token_capacity=6, packed_token_capacity=24)
    decoded = ExecutionGraph.FromString(graph.SerializeToString())
    assert (decoded.batch_capacity, decoded.token_capacity, decoded.packed_token_capacity) == (8, 6, 24)


@pytest.mark.parametrize("changes", [
    {"batch_capacity": 1}, {"batch_capacity": 3}, {"prefill_capacity": 0},
    {"prefill_capacity": 65}, {"prefill_capacity": 1 << 31}, {"prefill_capacity": True}, {"token_capacity": 9},
    {"speculation_enabled": None}, {"speculation_enabled": False, "token_capacity": 2},
    {"inputs": ()}, {"kind": "prefill"},
])
def test_mixed_signature_rejects_ambiguous_or_invalid_row_contracts(changes):
    args = dict(kind="mixed", batch_capacity=4, token_capacity=1, inputs=("prefill_row",),
                speculation_enabled=True, prefill_capacity=64)
    with pytest.raises((ValueError, TypeError)):
        GraphSignature(**(args | changes))


def test_mixed_execution_requires_vector_completions():
    with pytest.raises(ValueError, match="ABI3"):
        _execution(signatures={"decode": GraphSignature("mixed", 2, 1, ("prefill_row",),
                    speculation_enabled=False, prefill_capacity=64),
                    "prefill": GraphSignature("prefill", 1, 16)})
