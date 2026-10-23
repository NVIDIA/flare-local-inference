"""Packed rounds declare proposal depth separately from verifier width and rows."""

import pytest

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p
from ignition.preparation.execution import GraphSignature
from tests.preparation.test_execution import _execution, _tensor


def signature(depth=5, **changes):
    arguments = dict(kind="packed_speculative", batch_capacity=32, token_capacity=6,
        inputs=("row_slot_ids", "row_generations", "live_rows", "speculative_prefix_limits",
                "token_ids", "page_tables", "request_state_rows", "sample_enabled"),
        packed_token_capacity=128, draft_tokens=depth, speculation_enabled=True)
    return GraphSignature(**(arguments | changes))


def test_depth_and_physical_capacity_are_independent():
    for depth in range(1, 6):
        for rows in (32, 64, 96, 128):
            graph = signature(depth, packed_token_capacity=rows)
            assert (graph.draft_tokens, graph.token_capacity, graph.packed_token_capacity) == (depth, 6, rows)


@pytest.mark.parametrize("changes", [
    {"draft_tokens": None}, {"draft_tokens": 0}, {"draft_tokens": 6},
    {"draft_tokens": True}, {"packed_token_capacity": 0}, {"packed_token_capacity": 31},
    {"packed_token_capacity": 193}, {"inputs": ("query_lengths", "speculative_prefix_limits")},
    {"inputs": ("token_ids",)}, {"speculation_enabled": False},
])
def test_invalid_round_geometry_or_host_query_lengths_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        signature(**changes)


def test_manifest_preserves_shallow_depth_and_requires_vector_completion(monkeypatch):
    graph = signature(1)
    buffers = {"frame_index": _tensor("int32", (1,)), "submission_sequence": _tensor("int64", (1,)),
        "completion_records": _tensor("int64", (2, 32, 24))}
    buffers.update((name, _tensor("int32", (32,))) for name in graph.inputs)
    execution = _execution(graphs={"round": object()}, signatures={"round": graph},
                           buffers=buffers, max_active=32)
    spans = [(4096 * (i + 1), 4) for i in range(len(buffers))]
    monkeypatch.setattr(execution, "_validate_resources", lambda: (9, [11], spans))
    manifest = execution.manifest()
    encoded = p.ExecutionGraph(**{k: v for k, v in manifest["graphs"][0].items() if k != "handle"})
    decoded = p.ExecutionGraph.FromString(encoded.SerializeToString())
    assert manifest["abi_version"] == 3
    assert (decoded.draft_tokens, decoded.verify_tokens, decoded.packed_token_capacity) == (1, 6, 128)
    assert decoded.draft_source == p.MTP
    buffers["completion_records"] = _tensor("int64", (2, 32, 8))
    with pytest.raises(ValueError, match="ABI3"):
        _execution(graphs={"round": object()}, signatures={"round": graph}, buffers=buffers, max_active=32)


def test_semantic_contract_retains_host_pending_and_cache_dependencies():
    graph = signature(2)
    cursor = dict(state=c.state_view("state", 512, offset=16, nbytes=8), lag_tokens=0)
    result, = c.graph_contracts({"round": graph}, {"packed_speculative": c.text_inputs("packed_speculative")},
        continuation=c.state_view("state", 512), pending=c.state_view("pending", 8),
        cursor=cursor, rows=None, embedding_tables=[0, 1])
    assert result["kind"] == p.PACKED_SPECULATIVE
    sources = {v["buffer"]: v["source"] for v in result["inputs"]}
    assert sources["speculative_prefix_limits"] == p.SPECULATIVE_PREFIX_LIMIT
    assert p.QUERY_LENGTH not in sources.values()
    assert sources["token_ids"] == p.PENDING_TOKEN
    assert result["pending_token"] is None
    assert result["state_rows_per_request"] == 6
    assert result["draft_cursor"] == cursor
    assert result["embedding_tables"] == [0, 1]
    assert all(access["access"] == p.READ_WRITE for access in result["state"])


def test_legacy_graphs_do_not_acquire_explicit_depth_or_prefix_limits():
    for kind in ("speculative", "proposal", "verification"):
        graph = GraphSignature(kind, 4, 6)
        assert graph.draft_tokens is None
        values = c.text_inputs(kind)
        assert values["query_lengths"]["source"] == p.QUERY_LENGTH
        assert "speculative_prefix_limits" not in values
        with pytest.raises(ValueError, match="explicit draft depth"):
            GraphSignature(kind, 4, 6, draft_tokens=5)
