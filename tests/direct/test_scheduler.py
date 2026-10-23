"""Direct execution's scheduling, prefix bookkeeping and Helios-equivalent input packing, on the CPU."""

import numpy as np
import pytest

from ignition.direct.scheduler import (
    Acceptance,
    AdmissionError,
    CompletionError,
    Layout,
    Request,
    SamplingParameters,
    Scheduler,
)

VOCABULARY, EOS = 1000, (7,)


def layout(**overrides):
    graphs = {("prefill", 1, tokens): f"prefill{tokens}" for tokens in (128, 256, 512)}
    for kind in ("init", "restore", "snapshot", "resume"):
        graphs.update({(kind, batch, 1): f"{kind}{batch}" for batch in (1, 2, 4)})
    graphs.update({("speculative", batch, width): f"speculative{batch}x{width}"
                   for batch in (1, 2, 4) for width in (1, 2, 3)})
    values = dict(capacity=4, max_active=4, context=4096, page_tokens=256, table_width=16, pages=64,
                  snapshots=2, prefill_chunk=512, vocabulary=VOCABULARY, eos=EOS, graphs=graphs,
                  widths=(1, 2, 3), state_rows_per_request=1, draft_pages=1, record_fields=24)
    values.update(overrides)
    from ignition.preparation import contract as c, runtime_package_pb2 as p
    bindings = {name: list(c.text_inputs(key[0]).values()) for key, name in values["graphs"].items()}
    specs = {}
    for fields in bindings.values():
        for b in fields:
            source, name = b["source"], b["buffer"]
            width = {p.SAMPLING_PARAMETERS: 6, p.EOS_TOKEN_IDS: 8, p.LIVE_STATE_ROWS: 8,
                     p.PROMPT_HISTOGRAM: values["vocabulary"], p.GRAMMAR_MASK: -(-values["vocabulary"] // 32),
                     p.PAGE_TABLE: values["table_width"], p.COPY_TOKEN_IDS: 7,
                     p.COPY_GRAMMAR_MASK: 8 * -(-values["vocabulary"] // 32), p.PROMPT_POSITION: 3}.get(source, 1)
            dtype = np.float32 if source == p.SAMPLING_PARAMETERS else np.int64 if name in _INT64 or name.startswith("copy_") and source in (p.COPY_TOKEN_IDS, p.COPY_CONSUMED_TOKENS, p.COPY_GENERATION) or source == p.PROMPT_POSITION else np.int32
            rows = values["prefill_chunk"] if b["indexing"] == p.PACKED_TOKEN else values["capacity"]
            shape = (1,) if b["indexing"] == p.SCALAR else (rows, width) if width > 1 else (rows,)
            if name not in specs or np.prod(shape) > np.prod(specs[name][1]):
                specs[name] = (np.dtype(dtype), shape)
    return Layout(**values, bindings=bindings, buffer_specs=specs,
                  page_views={"page_tables": 0, "draft_page_tables": 1})


def request(ident, prompt, limit=8, seed=0):
    return Request(ident, prompt, limit, SamplingParameters(seed=seed, temperature=1., top_p=.95))


def uploads(scheduler, step):
    """Decode a step's uploads into {buffer: {element offset: values}}."""
    types = {"admission_parameters": np.float32}
    result = {}
    for name, offset, data in scheduler.pack(step):
        dtype = types.get(name, np.int64 if name in _INT64 else np.int32)
        values = np.frombuffer(data, dtype=dtype)
        result.setdefault(name, {})[offset // np.dtype(dtype).itemsize] = values.tolist()
    return result


_INT64 = {"request_state_rows", "row_generations", "token_ids", "state_request_slots", "state_source_slots",
          "state_destination_slots", "state_expected_consumed", "admission_generations", "admission_seed_bits",
          "admission_max_new_tokens", "admission_eos_token_ids"}


def records(scheduler, step, results):
    """Completion rows for `results`: {request: (status, consumed, generated tokens, error, offered, accepted)}."""
    rows = np.zeros((scheduler.layout.capacity, scheduler.layout.record_fields), dtype=np.int64)
    rows[:, 4] = -1
    rows[:, 16:] = -1
    for request, (status, consumed, tokens, error, *acceptance) in results.items():
        row = rows[request.slot]
        emitted = len(tokens)
        row[:8] = (request.generation, step.sequence, consumed, len(request.generated) + emitted,
                   tokens[-1] if tokens else -1, status, error, 0)
        row[8] = emitted
        row[9:11] = acceptance or (0, 0)
        row[12], row[13], row[15] = -1, max(consumed - 1, 0), step.width
        row[16:16 + emitted] = tokens
    return rows


def test_a_cold_request_initializes_prefills_in_chunks_snapshots_and_decodes():
    scheduler = Scheduler(layout())
    first = request(1, list(range(10, 710)), limit=4, seed=3)
    scheduler.submit(first)

    step = scheduler.plan()
    assert (step.kind, step.graph, [lane.request for lane in step.lanes]) == ("init", "init1", [first])
    packed = uploads(scheduler, step)
    assert packed["request_state_rows"] == {0: [1, 0, 0, 0, 0, 0, 0, 0]}
    assert packed["state_source_slots"] == {0: [-1, -1, -1, -1]}
    assert packed["state_destination_slots"] == {0: [1, -1, -1, -1]}
    assert packed["state_operation"] == {0: [1, 0, 0, 0]} and packed["state_expected_consumed"] == {0: [0, -1, -1, -1]}
    assert packed["admission_slots"] == {0: [0, -1, -1, -1]} and packed["admission_generations"] == {0: [1, 0, 0, 0]}
    assert packed["admission_seed_bits"] == {0: [3]} and packed["admission_max_new_tokens"] == {0: [4]}
    assert packed["admission_eos_token_ids"] == {0: [7, -1, -1, -1, -1, -1, -1, -1]}
    assert packed["admission_eos_count"] == {0: [1]}
    assert np.allclose(packed["admission_parameters"][0], [1., .95, 0., 1., 0., 0.])
    histogram = packed["admission_histogram"][0]
    assert len(histogram) == VOCABULARY and histogram[10] == 1 and sum(histogram) == 700
    assert packed["admission_grammar_mask"] == {0: [-1] * 32}
    # Pages through prompt, output limit and the widest round, plus the draft page: ceil(706 / 256) + 1.
    assert first.pages == [0, 1, 2, 3]
    assert packed["page_tables"] == {0: [0, 1, 2, 3]} and packed["draft_page_tables"] == {0: [1, 2, 3]}
    assert "row_slot_ids" not in packed
    assert scheduler.complete(step, records(scheduler, step, {first: (1, 0, [], 0)})) == []

    step = scheduler.plan()
    assert (step.graph, step.lanes[0].start, step.lanes[0].tokens) == ("prefill512", 0, 512)
    packed = uploads(scheduler, step)
    assert packed["token_ids"] == {0: list(range(10, 522))}
    assert packed["query_lengths"] == {0: [512, 0, 0, 0]} and packed["sample_enabled"] == {0: [0, 0, 0, 0]}
    assert packed["row_slot_ids"] == {0: [0, -1, -1, -1]} and packed["live_rows"] == {0: [1]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 512, [], 0)}))

    step = scheduler.plan()
    assert (step.graph, step.lanes[0].start, step.lanes[0].tokens) == ("prefill256", 512, 188)
    assert uploads(scheduler, step)["sample_enabled"] == {0: [1, 0, 0, 0]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 700, [42], 0)}))
    assert first.phase == "snapshot" and first.generated == [42]

    step = scheduler.plan()
    assert step.graph == "snapshot1"
    packed = uploads(scheduler, step)
    assert packed["state_source_slots"] == {0: [1, -1, -1, -1]}
    assert packed["state_destination_slots"] == {0: [5, -1, -1, -1]}
    assert packed["state_expected_consumed"] == {0: [700, -1, -1, -1]}
    assert packed["row_slot_ids"] == {0: [0, -1, -1, -1]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 700, [], 0)}))
    checkpoint = scheduler.cache[tuple(first.prompt)]
    assert (checkpoint.snapshot, checkpoint.pages, checkpoint.consumed) == (0, (0, 1), 700)

    step = scheduler.plan()
    assert (step.kind, step.graph, step.lanes[0].tokens) == ("speculative", "speculative1x3", 3)
    packed = uploads(scheduler, step)
    assert packed["token_ids"] == {0: [42, 0, 0, 0]} and packed["query_lengths"] == {0: [3, 0, 0, 0]}
    assert packed["sample_enabled"] == {0: [1, 0, 0, 0]}
    scheduler.complete(step, records(scheduler, step, {first: (1, 702, [43, 44], 0, 2, 1)}))
    assert first.generated == [42, 43, 44] and first.consumed == 702 and first.pending_token == 44

    step = scheduler.plan()
    # One output token remains, so the row offers no drafts though the graph is wider.
    assert step.lanes[0].tokens == 1
    retired = scheduler.complete(step, records(scheduler, step, {first: (2, 703, [7], 0)}))
    assert retired == [first] and first.finish_reason == "stop" and first.slot == 0
    assert scheduler.idle and scheduler.page_pool.available == 64 - 2


def test_an_identical_prompt_restores_its_checkpoint_and_shares_its_complete_pages():
    scheduler = Scheduler(layout())
    prompt = list(range(10, 610))
    first = request(1, prompt, limit=1)
    scheduler.submit(first)
    for result in ((1, 0, []), (1, 512, []), (2, 600, [42])):
        step = scheduler.plan()
        scheduler.complete(step, records(scheduler, step, {first: (*result, 0)}))
    step = scheduler.plan()
    assert step.graph == "snapshot1"
    assert scheduler.complete(step, records(scheduler, step, {first: (2, 600, [], 0)})) == [first]
    shared = scheduler.cache[tuple(prompt)].pages
    assert len(shared) == 2 and scheduler.page_pool.references[shared[0]] == 1

    copies = [request(2 + index, prompt, limit=6, seed=index) for index in range(3)]
    for copy in copies:
        scheduler.submit(copy)
    step = scheduler.plan()
    assert (step.graph, [lane.request for lane in step.lanes]) == ("restore4", copies)
    packed = uploads(scheduler, step)
    assert packed["state_source_slots"] == {0: [5, 5, 5, -1]}
    assert packed["state_destination_slots"] == {0: [1, 2, 3, -1]}
    assert packed["state_expected_consumed"] == {0: [600, 600, 600, -1]}
    assert packed["admission_count"] == {0: [3]} and packed["admission_slots"] == {0: [0, 1, 2, -1]}
    assert packed["admission_generations"] == {0: [2, 1, 1, 0]}
    for copy in copies:
        assert copy.pages[:2] == list(shared) and len(set(copy.pages[2:]) & set(shared)) == 0
        assert copy.reused == 600 and scheduler.page_pool.references[shared[0]] == 4
    tables = packed["page_tables"]
    assert tables[0][:2] == list(shared) and tables[16][:2] == list(shared)
    scheduler.complete(step, records(scheduler, step, {copy: (1, 600, [], 0) for copy in copies}))

    step = scheduler.plan()
    assert (step.graph, [lane.tokens for lane in step.lanes]) == ("resume4", [0, 0, 0])
    packed = uploads(scheduler, step)
    assert "page_tables" not in packed and "query_lengths" not in packed
    assert packed["row_slot_ids"] == {0: [0, 1, 2, -1]}
    scheduler.complete(step, records(scheduler, step, {copy: (1, 600, [50 + i], 0) for i, copy in enumerate(copies)}))

    step = scheduler.plan()
    assert step.graph == "speculative4x3"
    assert uploads(scheduler, step)["token_ids"] == {0: [50, 51, 52, 0]}
    assert scheduler.stats["prefix_hits"] == 3


def respond(scheduler, step):
    """Plausible device results: prefill consumes its chunk, rounds accept every draft."""
    results = {}
    for lane in step.lanes:
        current = lane.request
        if step.kind == "prefill":
            end = lane.start + lane.tokens
            results[current] = (1, end, [9] if end == len(current.prompt) else [], 0)
        elif step.kind in ("speculative", "decode"):
            results[current] = (1, current.consumed + lane.tokens, [8] * lane.tokens, 0, lane.tokens - 1,
                                lane.tokens - 1)
        elif step.kind == "resume":
            results[current] = (1, current.consumed, [9], 0)
        else:
            results[current] = (1, current.consumed, [], 0)
    return records(scheduler, step, results)


def test_device_errors_retire_the_failed_request_and_free_its_pages():
    scheduler = Scheduler(layout())
    failing, healthy = request(1, [1, 2, 3], limit=64), request(2, [4, 5, 6], limit=64)
    scheduler.submit(failing)
    scheduler.submit(healthy)
    while not (failing.phase == healthy.phase == "decode"):
        step = scheduler.plan()
        scheduler.complete(step, respond(scheduler, step))
    step = scheduler.plan()
    assert step.graph == "speculative2x3"
    before = dict(scheduler.rounds)
    retired = scheduler.complete(step, records(scheduler, step, {
        failing: (3, failing.consumed, [], 6), healthy: (1, healthy.consumed + 2, [10, 11], 0, 2, 2)}))
    assert retired == [failing] and failing.error == 6 and failing.finish_reason == "error"
    assert scheduler.device_errors == {6: 1} and scheduler.rounds[(2, 3)] == before.get((2, 3), 0) + 1
    assert failing.pages == [] and scheduler.slots[0] is None
    assert scheduler.plan().graph == "speculative1x3"


def test_completion_records_must_match_the_submitted_step():
    scheduler = Scheduler(layout())
    current = request(1, [1, 2, 3])
    scheduler.submit(current)
    step = scheduler.plan()
    stale = records(scheduler, step, {current: (1, 0, [], 0)})
    stale[current.slot, 1] -= 1
    with pytest.raises(CompletionError, match="inactive or stale"):
        scheduler.complete(step, stale)
    stray = records(scheduler, step, {current: (1, 0, [], 0)})
    stray[3, 5] = 1
    with pytest.raises(CompletionError, match="outside step"):
        scheduler.complete(step, stray)


def test_admission_rejects_what_the_prepared_capacities_cannot_hold():
    scheduler = Scheduler(layout(pages=8))
    with pytest.raises(AdmissionError, match="exceeds this server's context"):
        scheduler.submit(request(1, [1] * 4000, limit=200))
    with pytest.raises(AdmissionError, match="more KV pages"):
        scheduler.submit(request(2, [1] * 3000, limit=8))
    with pytest.raises(AdmissionError, match="vocabulary"):
        scheduler.submit(request(3, [VOCABULARY]))
    with pytest.raises(AdmissionError, match="greedy"):
        SamplingParameters(temperature=0.)


def test_requests_wait_for_slots_and_pages_and_rounds_alternate_with_prefill():
    scheduler = Scheduler(layout(max_active=2, pages=12))
    requests = [request(index, list(range(1, 600)), limit=4) for index in range(3)]
    for current in requests:
        scheduler.submit(current)
    step = scheduler.plan()
    # Each request reserves ceil(605 / 256) + 1 pages; the third waits for a slot.
    assert step.graph == "init2" and len(scheduler.queue) == 1
    scheduler.complete(step, records(scheduler, step, {r: (1, 0, [], 0) for r in requests[:2]}))
    step = scheduler.plan()
    assert step.lanes[0].request is requests[0]
    scheduler.complete(step, records(scheduler, step, {requests[0]: (1, 512, [], 0)}))
    step = scheduler.plan()
    scheduler.complete(step, records(scheduler, step, {requests[0]: (1, 599, [5], 0)}))
    step = scheduler.plan()
    assert step.graph == "snapshot1"
    scheduler.complete(step, records(scheduler, step, {requests[0]: (1, 599, [], 0)}))
    kinds = []
    for _ in range(4):
        step = scheduler.plan()
        kinds.append(step.kind)
        lanes = {lane.request: lane for lane in step.lanes}
        if step.kind == "prefill":
            results = {requests[1]: (1, lanes[requests[1]].start + lanes[requests[1]].tokens, [], 0)}
            if results[requests[1]][1] == 599:
                results[requests[1]] = (1, 599, [6], 0)
        else:
            results = {r: (1, r.consumed + 1, [8], 0, 2, 0) for r in lanes}
        scheduler.complete(step, records(scheduler, step, results))
    assert kinds == ["speculative", "prefill", "speculative", "prefill"]
    # The second prompt is cached already, so it decodes without a snapshot.
    assert requests[1].phase == "decode"


def decoding(scheduler, count, limit=1000):
    """`count` requests decoding from the first round, as Helios's policy tests set them up."""
    requests = [request(index, [1, 2, 3], limit=limit) for index in range(count)]
    for current in requests:
        scheduler.submit(current)
    scheduler._admit()
    for current in requests:
        current.phase, current.consumed, current.generated = "decode", 3, [5]
    return requests


def decode_round(scheduler, nanos, accepted):
    """Plan, time and complete one round; each row accepts up to `accepted` drafts."""
    step = scheduler.plan()
    scheduler.advance_to(scheduler.now_ns + nanos(step))
    results = {}
    for lane in step.lanes:
        offered = lane.tokens - 1
        emitted = min(accepted, offered) + 1
        results[lane.request] = (1, lane.request.consumed + emitted, [8] * emitted, 0, offered, emitted - 1)
    scheduler.complete(step, records(scheduler, step, results), nanos(step))
    return step.batch, step.width, len(step.lanes)


def test_decode_rounds_explore_unmeasured_graphs_then_follow_graph_time_and_acceptance():
    scheduler = Scheduler(layout())
    decoding(scheduler, 2)
    cost = lambda step: (2 + step.width + step.batch) * 1_000_000
    # Every unmeasured graph is explored once before any measured one runs, the most valuable first:
    # both rows at the widest verification in the smallest batch that holds them.
    explored = [decode_round(scheduler, cost, 2) for _ in range(9)]
    assert explored[0] == (2, 3, 2)
    assert sorted(shape[:2] for shape in explored) == [(batch, width) for batch in (1, 2, 4) for width in (1, 2, 3)]
    # Then the measured graph with the best expected value per graph time serves both rows.
    assert decode_round(scheduler, cost, 2) == (2, 3, 2)
    # Rejected drafts lower the expected tokens until the narrowest round is worth the most per time.
    assert [decode_round(scheduler, cost, 0) for _ in range(40)][-1] == (2, 1, 2)
    # Estimates expire after 30 s without an observation, and expired graphs are explored again.
    scheduler.advance_to(scheduler.now_ns + 31_000_000_000)
    assert decode_round(scheduler, cost, 0) == (2, 3, 2)


def test_a_fixed_width_still_chooses_batch_capacity_and_rows_by_value():
    scheduler = Scheduler(layout(fixed_width=2))
    requests = decoding(scheduler, 3)
    # Three rows need batch four; a two-row graph costing a quarter as much carries the two most valuable rows.
    cost = lambda step: {1: 10, 2: 10, 4: 40}[step.batch] * 1_000_000
    shapes = [decode_round(scheduler, cost, 1) for _ in range(12)]
    assert {width for _, width, _ in shapes} == {2} and scheduler.rounds[(4, 2)] == 1
    assert shapes[3:] == [(2, 2, 2)] * 9
    # Fairness prices and output age rotate the rows, so no request waits indefinitely.
    assert all(len(current.generated) > 4 for current in requests)


def test_acceptance_follows_helios_depth_estimates():
    acceptance = Acceptance()
    assert acceptance.expected(1) == 1 and acceptance.expected(3) == 1.75
    acceptance.observe(2, 0, True)
    assert acceptance.depths[0][0] == pytest.approx(.5 - .5 / 3)
    acceptance.observe(0, 0, True)
    assert acceptance.depths[0][1] == 3


def test_layout_selects_the_serving_variant_of_each_graph():
    from ignition.preparation.execution import GraphSignature

    def Signature(kind, batch, tokens, enabled):
        return GraphSignature(kind, batch, tokens, speculation_enabled=enabled)

    class Tensor:
        def __init__(self, *shape):
            self.shape = shape

        def numel(self):
            return int(np.prod(self.shape))

    class Model:
        max_active = 3
        signatures = {"prefill128": Signature("prefill", 1, 128, True),
                      "prefill128-no-mtp": Signature("prefill", 1, 128, False),
                      "decode4": Signature("decode", 4, 1, False), "init4": Signature("init", 4, 1, True),
                      "speculative4x2": Signature("speculative", 4, 2, True)}
        buffers = {"row_slot_ids": Tensor(4), "page_tables": Tensor(4, 32), "draft_page_tables": Tensor(4, 32),
                   "completion_records": Tensor(2, 4, 24)}
        server_config = {"max_sequence_tokens": 8192, "page_tokens": 256, "pages": 100, "snapshots": 5,
                         "prefill_chunk_size": 128, "model_contract": {"vocabulary": VOCABULARY},
                         "frontend": {"eos_token_ids": [7, 8]}, "speculative_widths": [1, 2],
                         "speculative_fixed_width": None, "state_rows_per_request": 1}

    fixture = layout(context=8192, table_width=32, graphs={
        (v.kind, v.batch_capacity, v.token_capacity): name for name, v in Model.signatures.items()
        if v.speculation_enabled in (None, True)})
    Model.server_config["model_contract"].update(cache={"page_tokens": 256,
        "page_tables": [{"buffer": name, "page_offset": shift} for name, shift in fixture.page_views.items()]},
        graphs=[{"graph": name, "inputs": fields, "state_rows_per_request": 1} for name, fields in fixture.bindings.items()])
    for name, (dtype, shape) in fixture.buffer_specs.items():
        tensor = Tensor(*shape)
        tensor.dtype = dtype
        Model.buffers[name] = tensor
    result = Layout.of(Model())
    assert result.graphs == {("prefill", 1, 128): "prefill128", ("init", 4, 1): "init4",
                             ("speculative", 4, 2): "speculative4x2"}
    assert (result.capacity, result.max_active, result.table_width, result.draft_pages) == (4, 3, 32, 1)
    assert (result.execution_width, result.state_row(2), result.snapshot_row(1)) == (2, 3, 5)

    Model.signatures["proposal4x3"] = Signature("proposal", 4, 3, True)
    Model.signatures["verification4x2"] = Signature("verification", 4, 2, True)
    Model.signatures["speculative4x1"] = Signature("speculative", 4, 1, True)
    result = Layout.of(Model())
    assert result.graphs["speculative", 4, 2] == "verification4x2"
    assert result.graphs["speculative", 4, 1] == "speculative4x1"
    assert result.paired_graphs == frozenset({"verification4x2"})
    assert not any(kind == "proposal" for kind, _, _ in result.graphs)


def test_native_confidence_uses_conditional_survival_without_observed_acceptance_updates():
    acceptance = Acceptance()
    acceptance.confidence([.8, .25, .5])
    assert acceptance.expected(4) == pytest.approx(1 + .8 + .8 * .25 + .8 * .25 * .5)
    acceptance.observe(3, 0, True)
    assert acceptance.expected(4) == pytest.approx(2.1)


def packed_layout(*, combined=True):
    kind = "packed_speculative" if combined else "speculative"
    graphs = {("speculative", batch, 1): f"width-one-{batch}" for batch in (1, 2, 4)}
    graphs.update({(kind, 4, 6, capacity, 5 if combined else 0): f"packed-{capacity}"
                   for capacity in (4, 8, 16, 24)})
    return layout(graphs=graphs, widths=(1, 6),
                  paired_graphs=frozenset() if combined else frozenset(f"packed-{p}" for p in (4, 8, 16, 24)),
                  completion_cursors={f"packed-{p}": 1 for p in (4, 8, 16, 24)})


def ready_packed(*, combined=True, capacity=4, remaining=(20, 20, 20)):
    scheduler = Scheduler(packed_layout(combined=combined), prefix_cache=False)
    requests = [request(i, [10, 11], limit=n + 1) for i, n in enumerate(remaining)]
    for current in requests:
        scheduler.submit(current)
    scheduler._admit()
    for current in requests:
        current.phase, current.consumed, current.generated = "decode", 2, [42]
    for shape in scheduler.layout.graphs:
        scheduler.policy.costs.observe(shape, 1_000_000 if len(shape) == 5 and shape[3] == capacity else 100_000_000)
    return scheduler, requests


def packed_records(scheduler, step, offers=(1, 0, 0)):
    rows = records(scheduler, step, {
        lane.request: (1, lane.start + offered + 1, [43] * (offered + 1), 0, offered, offered)
        for lane, offered in zip(step.lanes, offers)})
    return rows


def test_packed_round_selects_each_physical_capacity_and_uploads_immutable_ceilings():
    from dataclasses import FrozenInstanceError
    for capacity in (4, 8, 16, 24):
        scheduler, requests = ready_packed(capacity=capacity)
        step = scheduler.plan()
        assert (step.kind, step.graph, step.batch, step.width, step.packed_tokens, step.draft_tokens) == (
            "packed_speculative", f"packed-{capacity}", 4, 6, capacity, 5)
        assert step.sequence == 1 and tuple(lane.tokens for lane in step.lanes) == (6, 6, 6)
        packed = uploads(scheduler, step)
        assert packed["speculative_prefix_limits"] == {0: [6, 6, 6, 0]}
        assert "query_lengths" not in packed
        with pytest.raises(FrozenInstanceError):
            step.lanes[0].tokens = 1
        assert all(len(r.pages) >= scheduler._pages_needed(r, 0) for r in requests)


def test_packed_round_commits_ragged_terminal_feedback_and_confidence_once():
    scheduler, requests = ready_packed()
    step = scheduler.plan()
    rows = packed_records(scheduler, step)
    confidence = [[.8] * 5, [.4] * 5, [.2] * 5, [float('nan')] * 5]
    scheduler.complete(step, rows, 2_000_000, confidence=confidence)
    assert [len(r.generated) for r in requests] == [3, 2, 2]
    assert [r.consumed for r in requests] == [4, 3, 3]
    assert [r.rounds for r in requests] == [1, 1, 1]
    assert requests[0].acceptance.expected(3) == pytest.approx(1 + .8 + .8 ** 2)
    assert scheduler.policy.costs.duration(*step.shape) > 1_000_000
    with pytest.raises(CompletionError, match="stale"):
        scheduler.complete(step, rows, confidence=confidence)
    assert [len(r.generated) for r in requests] == [3, 2, 2]


@pytest.mark.parametrize('damage', ['generation', 'sequence', 'status', 'negative-offer', 'ceiling', 'accepted',
                                   'physical', 'width', 'token', 'tail', 'reserved', 'base-reserved', 'model-reserved', 'copy-mask',
                                   'cursor', 'output', 'confidence', 'sidecar'])
def test_invalid_packed_terminal_batch_changes_no_request_or_cost(damage):
    scheduler, requests = ready_packed()
    step = scheduler.plan()
    rows = packed_records(scheduler, step)
    confidence = np.full((4, 5), .75)
    row = rows[requests[-1].slot]
    if damage == 'generation': row[0] += 1
    elif damage == 'sequence': row[1] += 1
    elif damage == 'status': row[5] = 0
    elif damage == 'negative-offer': row[9] = -1
    elif damage == 'ceiling': row[9] = 6
    elif damage == 'accepted': row[10] = 1
    elif damage == 'physical': row[9] = 1
    elif damage == 'width': row[15] = 5
    elif damage == 'token': row[4] = row[16] = VOCABULARY
    elif damage == 'tail': row[17] = 0
    elif damage == 'reserved': row[14] = 1
    elif damage == 'base-reserved': row[7] = 1
    elif damage == 'model-reserved': row[12] = 0
    elif damage == 'copy-mask':
        rows = packed_records(scheduler, step, (0, 0, 1))
        rows[requests[-1].slot, 11] = 1
    elif damage == 'cursor': row[13] += 1
    elif damage == 'output': row[3] = requests[-1].max_new_tokens + 1
    elif damage == 'confidence': confidence[2, 4] = float('nan')
    elif damage == 'sidecar': confidence = None
    before = [(r.consumed, list(r.generated), r.rounds, r.acceptance.expected(6)) for r in requests]
    costs = scheduler.policy.costs.estimates()
    with pytest.raises(CompletionError):
        scheduler.complete(step, rows, 7_000_000, confidence=confidence)
    assert before == [(r.consumed, r.generated, r.rounds, r.acceptance.expected(6)) for r in requests]
    assert scheduler.policy.costs.estimates() == costs
    assert not scheduler.rounds and not scheduler.steps


def test_packed_ceilings_bound_output_context_and_device_errors_remain_visible():
    scheduler, requests = ready_packed(remaining=(1, 2, 20))
    requests[2].consumed = scheduler.layout.context - 2
    step = scheduler.plan()
    assert step.kind == 'packed_speculative'
    assert [lane.tokens for lane in step.lanes] == [1, 2, 2]
    rows = packed_records(scheduler, step, (0, 0, 0))
    rows[requests[0].slot, 5] = 2
    row = rows[requests[1].slot]
    row[:8] = (requests[1].generation, step.sequence, requests[1].consumed, len(requests[1].generated), -1, 3, 256, 0)
    row[8:16] = (0, 0, 0, 0, -1, -1, 0, 1)
    row[16:] = -1
    confidence = np.full((4, 5), .75)
    confidence[1] = float('nan')
    retired = scheduler.complete(step, rows, confidence=confidence)
    assert retired == requests[:2]
    assert requests[0].finish_reason == 'length' and requests[1].error == 256
    assert scheduler.device_errors[256] == 1


def test_separate_packed_verification_keeps_capacity_and_allocates_host_lengths():
    scheduler, _ = ready_packed(combined=False)
    step = scheduler.plan()
    assert (step.kind, step.graph, step.sequence, step.packed_tokens) == ('speculative', 'packed-4', 2, 4)
    assert sum(lane.tokens for lane in step.lanes) == 4
    packed = uploads(scheduler, step)
    assert packed['query_lengths'] == {0: [2, 1, 1, 0]}
    assert 'speculative_prefix_limits' not in packed


def test_layout_reads_real_combined_catalog_and_retains_all_c32_p128_shapes():
    from types import SimpleNamespace
    import torch
    from ignition.components.capacity import speculative_shapes
    from ignition.preparation.execution import GraphSignature

    shapes = speculative_shapes(32, (1, 2, 3, 4, 5, 6), 128)
    graphs = {('speculative', batch, 1): f'width-one-{batch}' for batch, width in shapes if width == 1}
    graphs.update({('packed_speculative', batch, 6, batch * width, 5): f'packed-{batch}-{batch * width}'
                   for batch, width in shapes if width > 1})
    fixture = layout(capacity=32, max_active=32, widths=(1, 2, 3, 4, 5, 6), graphs=graphs)
    signatures = {name: GraphSignature(*key[:3], tuple(fixture.bindings[name][i]['buffer']
                  for i in range(len(fixture.bindings[name]))), speculation_enabled=True,
                  packed_token_capacity=key[3] if len(key) == 5 else 0,
                  draft_tokens=key[4] if len(key) == 5 else None)
                  for key, name in graphs.items()}
    buffers = {name: torch.from_numpy(np.zeros(shape, dtype=dtype))
               for name, (dtype, shape) in fixture.buffer_specs.items()}
    buffers['completion_records'] = torch.zeros((2, 32, 24), dtype=torch.int64)
    config = dict(max_sequence_tokens=4096, pages=64, snapshots=2, prefill_chunk_size=512,
                  speculative_widths=[1, 2, 3, 4, 5, 6], frontend=dict(eos_token_ids=[7]),
                  model_contract=dict(vocabulary=VOCABULARY,
                    cache=dict(page_tokens=256, page_tables=[dict(buffer=name, page_offset=offset)
                               for name, offset in fixture.page_views.items()]),
                    graphs=[dict(graph=name, inputs=bindings, state_rows_per_request=1, draft_cursor=dict(lag_tokens=1))
                            for name, bindings in fixture.bindings.items()]))
    actual = Layout.of(SimpleNamespace(server_config=config, buffers=buffers,
                                       signatures=signatures, max_active=32))
    assert actual.graphs == graphs and len(actual.packed_shapes) == 28
    assert all(lag == 1 for lag in actual.completion_cursors.values())
    assert len(actual.paired_graphs) == 0
    scheduler = Scheduler(actual, prefix_cache=False)
    requests = [request(i, [10, 11], limit=20) for i in range(32)]
    for current in requests:
        scheduler.submit(current)
    scheduler._admit()
    for current in requests:
        current.phase, current.consumed, current.generated = 'decode', 2, [42]
    step = scheduler.plan()
    assert (step.kind, step.batch, step.width, step.packed_tokens, step.draft_tokens) == (
        'packed_speculative', 32, 6, 128, 5)
    assert len(step.lanes) == 32 and sum(lane.tokens for lane in step.lanes) == 192
    assert uploads(scheduler, step)['speculative_prefix_limits'] == {0: [6] * 32}


@pytest.mark.parametrize('bad_cursor', [False, True])
def test_packed_completion_honors_explicit_absence_of_a_draft_cursor(bad_cursor):
    scheduler, requests = ready_packed()
    step = scheduler.plan()
    scheduler.layout.completion_cursors[step.graph] = None
    rows = packed_records(scheduler, step)
    for lane in step.lanes:
        rows[lane.request.slot, 13] = -1
    if bad_cursor:
        rows[requests[-1].slot, 13] = 0
        with pytest.raises(CompletionError, match='history cursor'):
            scheduler.complete(step, rows, confidence=np.full((4, 5), .75))
        assert all(r.generated == [42] for r in requests)
    else:
        scheduler.complete(step, rows, confidence=np.full((4, 5), .75))
        assert [len(r.generated) for r in requests] == [3, 2, 2]
