"""Diagnostic rejection reports failure without touching device request state."""

import os

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.split_draft import _reject_stale
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_captured_rejection_preserves_replacement_and_inactive_requests():
    valid = torch.zeros(1, dtype=torch.bool, device="cuda")
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    epoch = torch.tensor([9], dtype=torch.int64, device="cuda")
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    generation = epoch.clone()
    consumed = torch.tensor([127], device="cuda")
    generated = torch.tensor([3], device="cuda")
    records = torch.zeros((2, 1, 24), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.tensor([19], device="cuda")

    def forward():
        torch.library.wrap_triton(_reject_stale)[(1,)](valid, slot, epoch, live, generation,
            consumed, generated, records, frame, sequence, 1, 1)

    operation = compile_forward(forward)
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    bundle = PreparedGraphBundle(device=0, graphs={"reject": graph}, buffers={},
        resources=(valid, slot, epoch, live, generation, consumed, generated, records,
                   frame, sequence, operation))
    worker = bundle.start_worker()
    try:
        for admitted, active, request_epoch in ((False, 1, 9), (True, 1, 9), (False, 0, 9), (False, 1, 10)):
            valid.fill_(admitted)
            live.fill_(active)
            generation.fill_(request_epoch)
            records.fill_(71)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0)
            completion = worker.poll(10000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert generation.item() == request_epoch and consumed.item() == 127 and generated.item() == 3
            if admitted or not active:
                assert records.eq(71).all()
            else:
                record = records[0, 0].tolist()
                assert record[:8] == [9, 19, 127 if request_epoch == 9 else 0,
                                      3 if request_epoch == 9 else 0, -1, 3, 256, 0]
                assert record[8] == 0 and record[16:] == [-1] * 8
    finally:
        worker.close()
        graph.reset()


@pytest.fixture(scope="module")
def package_storage():
    from ignition.preparation.package_storage import PackageStorage
    storage = PackageStorage(world=1)
    yield storage
    for region in storage.host_regions:
        region.close()


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("depth", [1, 3, 7])
@torch.no_grad()
def test_native_proposal_feedback_copies_host_mapped_confidence_without_allocating(batch, depth, package_storage):
    import struct
    import time
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    storage = package_storage
    confidence = storage.host_allocate("proposal_confidence", (batch, 7), torch.float32)
    confidence.device_view.zero_()
    values = torch.zeros((batch, 7), device="cuda")
    records = torch.zeros((2, batch, 24), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        confidence.device_view[:, :depth].copy_(values[:, :depth])
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        confidence.device_view[:, :depth].copy_(values[:, :depth])
    model = PreparedExecution(device=0, max_active=batch, graphs={"proposal": graph},
        signatures={"proposal": GraphSignature("proposal", batch, depth + 1, ("values",), speculation_enabled=True)},
        buffers=dict(frame_index=frame, submission_sequence=sequence, completion_records=records,
                     proposal_confidence=confidence.device_view, values=values), resources=(storage, confidence))
    executor = model.start_executor()
    try:
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for value in (.25, .75, 0., 1.) * 8:
            fence = executor.submit(0, [("values", 0, struct.pack(f"{batch * 7}f", *([value]*(batch * 7))))], True)
            deadline = time.monotonic() + 10
            while (completed := executor.poll()) is None:
                assert time.monotonic() < deadline
            assert completed.fence == fence
            assert [list(row) for row in completed.proposal_confidence] == [[value] * depth + [0.] * (7 - depth)] * batch
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    finally:
        executor.close()
        graph.reset()


@pytest.mark.parametrize("with_survival", [False, True])
@torch.no_grad()
def test_native_verification_feedback_preserves_frames_and_skips_width_one(with_survival):
    import struct
    import time
    from ignition.preparation.execution import GraphSignature, PreparedExecution

    records = torch.zeros((2, 4, 24), dtype=torch.int64, device="cuda")
    feedback = torch.full((2, 4, 7), -1., device="cuda")
    values = torch.full((4, 7), -1., device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.zeros(1, dtype=torch.int64, device="cuda")
    graphs, signatures, owners = {}, {}, []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for width in (1, 3):
        row = torch.zeros((4, 24), dtype=torch.int64, device="cuda")
        row[:, [4, 12, 13, *range(16, 24)]] = -1
        row[2, :16] = torch.tensor([7, 0, 3, 1, 20, 1, 0, 0, 1, width - 1, 0, 0, -1, -1, 0, width], device="cuda")
        row[2, 16] = 20
        stream.wait_stream(torch.cuda.current_stream())

        def forward():
            row[2, 1].copy_(sequence[0])
            records.index_copy_(0, frame.long(), row.unsqueeze(0))
            if with_survival and width > 1:
                feedback.index_copy_(0, frame.long(), values.unsqueeze(0))

        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        graphs[f"verify{width}"] = graph
        signatures[f"verify{width}"] = GraphSignature("verification", 4, width, ("values",), speculation_enabled=True)
        owners.append(row)
    buffers = dict(frame_index=frame, submission_sequence=sequence, completion_records=records, values=values)
    if with_survival:
        buffers["completion_speculative_survival"] = feedback
    model = PreparedExecution(device=0, max_active=4, graphs=graphs, signatures=signatures,
        buffers=buffers, resources=tuple(owners))
    executor = model.start_executor()
    try:
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for index in range(24):
            wide = index % 3 != 1
            expected = [[-1.] * 7 for _ in range(4)]
            expected[2][:2] = [.75, .25] if index % 2 else [.5, .125]
            payload = struct.pack("28f", *(p for row in expected for p in row))
            fence = executor.submit(int(wide), [("values", 0, payload)], True)
            deadline = time.monotonic() + 10
            while (completed := executor.poll()) is None:
                assert time.monotonic() < deadline
            assert completed.fence == fence
            assert completed.records[2][1] == fence
            assert completed.records[2][9] == (2 if wide else 0)
            actual = completed.speculative_survival
            assert (None if actual is None else [list(row) for row in actual]) == (expected if with_survival and wide else None)
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    finally:
        executor.close()
        for graph in graphs.values():
            graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("target_capacity", [9, 12])
def test_retained_proposals_regroup_prefixes_and_reject_stale_ownership(target_capacity):
    from types import SimpleNamespace
    from ignition.components.split_draft import PreparedDraftVerification, PreparedProposalStorage

    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device="cuda")

    requests = SimpleNamespace(capacity=4, vocab_size=13,
        generation=tensor([11, 12, 13, 14]), consumed_tokens=tensor([10, 20, 30, 40]),
        generated_tokens=tensor([1, 1, 1, 1]), pending_token=tensor([4, 5, 6, 7]),
        status=tensor([1, 1, 1, 1]))
    inputs = SimpleNamespace(row_slot_ids=tensor([3, 0, 2, 1], torch.int32),
        row_generations=tensor([14, 11, 13, 12]), live_rows=tensor([4], torch.int32),
        request_state_rows=tensor([[1], [2], [3], [4]]), query_lengths=tensor([3, 1, 2, 3], torch.int32))
    owner = SimpleNamespace(requests=requests, inputs=inputs, sequence=tensor([6]),
        frame=tensor([0], torch.int32), completion=completion_outputs(torch.zeros((2, 4, 24), dtype=torch.int64, device="cuda")))
    retained = PreparedProposalStorage(owner, 7)
    retained.ready.fill_(True)
    retained.generations.copy_(requests.generation)
    retained.before.copy_(requests.consumed_tokens)
    retained.state_rows.copy_(inputs.request_state_rows[:, 0])
    retained.sequence.copy_(tensor([2, 3, 4, 5]))
    retained.depths.fill_(7)
    retained.candidates.copy_(torch.arange(32, device="cuda").view(4, 8))
    retained.candidates[:, 0].copy_(requests.pending_token)
    retained.drafts.copy_(retained.candidates[:, 1:])
    retained.probabilities.copy_(torch.arange(4 * 7 * 13, device="cuda").view(4, 7, 13))
    retained.valid.fill_(True)
    target = SimpleNamespace(owner=owner, batch=4, width=3, target_token_capacity=target_capacity,
        active=torch.zeros(4, dtype=torch.bool, device="cuda"), before=torch.empty_like(retained.before),
        base_offset=torch.empty_like(retained.before), candidates=torch.empty((4, 3), dtype=torch.int64, device="cuda"),
        drafts=torch.empty((4, 2), dtype=torch.int64, device="cuda"),
        probabilities=torch.empty((4, 2, 13), device="cuda"), valid=torch.empty((4, 2), dtype=torch.bool, device="cuda"))
    observed = torch.empty_like(inputs.row_slot_ids)
    target.verify = lambda valid: observed.copy_(inputs.row_slot_ids)
    operation = compile_forward(PreparedDraftVerification(retained, target))
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    retained.ready.fill_(True)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    assert observed.tolist() == [3, 0, 2, 1]
    assert not retained.ready.any()
    assert target.probabilities.is_contiguous()
    torch.testing.assert_close(target.probabilities, retained.probabilities[[3, 0, 2, 1], :2])
    torch.testing.assert_close(target.candidates, retained.candidates[[3, 0, 2, 1], :3])

    # Duplicate verification never reaches a request, and leaves its inputs intact.
    graph.replay()
    torch.cuda.synchronize()
    assert observed.tolist() == [-1] * 4
    assert inputs.row_slot_ids.tolist() == [3, 0, 2, 1]
    assert owner.completion.records[0, :, 6].tolist() == [256] * 4

    # Different invalidation causes coexist in one batch without consuming the bank.
    retained.ready.fill_(True)
    requests.generation[3] += 1
    requests.consumed_tokens[0] += 1
    inputs.request_state_rows[2, 0] += 1
    retained.sequence[1] = owner.sequence[0]
    graph.replay()
    torch.cuda.synchronize()
    assert observed.tolist() == [-1] * 4
    assert retained.ready.all()
    assert requests.generation.tolist() == [11, 12, 13, 15]
    assert requests.consumed_tokens.tolist() == [11, 20, 30, 40]

    requests.generation.copy_(retained.generations)
    requests.consumed_tokens.copy_(retained.before)
    inputs.request_state_rows[:, 0].copy_(retained.state_rows)
    owner.sequence += 1
    inputs.live_rows.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert observed.tolist() == [3, 0, -1, -1]
    assert retained.ready.tolist() == [False, True, True, False]
    if target_capacity == 9:
        retained.ready.fill_(True)
        inputs.live_rows.fill_(4)
        inputs.query_lengths.fill_(3)
        graph.replay()
        torch.cuda.synchronize()
        assert observed.tolist() == [-1] * 4
        assert retained.ready.all()
    graph.reset()


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_confidence_head_preserves_fp32_bias_and_preceding_token_features(with_bias, weight_dtype):
    from types import SimpleNamespace
    from ignition.components.split_draft import PreparedDraftConfidence
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.manual_seed(811)
    batch, block, hidden, rank = 4, 7, 6144, 256
    head = torch.randn(batch * block, hidden, device="cuda").bfloat16()
    markov = torch.randn(batch, block, rank, device="cuda").bfloat16()
    weight = (torch.randn(1, hidden + rank, device="cuda") * .01).to(weight_dtype)
    bias = torch.tensor([.71], device="cuda") if with_bias else None
    output = torch.empty(batch, block, device="cuda")
    with RecordedPreparationSession(device=torch.device("cuda", 0), checkpoint="confidence",
                                    workload_schema="confidence", numerical_policy="default") as session:
        source = SimpleNamespace(owner=SimpleNamespace(block_size=block, logit_offset=0, session=session),
            batch=batch, width=block + 1, draft=SimpleNamespace(head_hidden=head),
            markov=SimpleNamespace(embedding=markov[:, 0]))
        confidence = PreparedDraftConfidence(source, weight, output, bias=bias, name="confidence")

        def forward():
            for position in range(block):
                confidence.record_embedding(position, markov[:, position])
            confidence()

        session.tune()
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for shift in (0., .25, -.5):
            head.add_(shift)
            markov.add_(shift)
            features = torch.cat((head.view(batch, block, hidden).float(), markov.float()), dim=-1)
            expected = (features @ weight.float().T).squeeze(-1)
            if bias is not None:
                expected.add_(bias)
            expected.sigmoid_()
            confidence.features.fill_(float("nan"))
            confidence.embeddings.fill_(float("nan"))
            confidence.logits.fill_(float("nan"))
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
        graph.reset()
