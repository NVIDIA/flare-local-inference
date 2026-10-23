"""A deterministic proposal backbone drives real validation and stochastic commit.

The model-free fixture substitutes only proposal tokens and causal confidence.
Request validation, retained storage, target filtering, stochastic verification,
commit, and graph-buffer allocation use the shared production components.
"""

import os
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.packed_speculation import PreparedPackedSpeculativeRound
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.parallel import TensorParallelShard
from ignition.components.parallel_draft import PreparedParallelDraftRound
from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (PreparedSpeculativeCommit, PreparedSpeculativeDistributions,
                                             PreparedStochasticVerification, SpeculativeRngState)
from ignition.components.split_draft import PreparedDraftVerification, PreparedProposalStorage
from ignition.components.state import scatter_rows
from ignition.preparation.scratch import GraphBufferPool
from ignition.preparation.compilation import compile_forward
from tests.components.causal_prefix_oracle import allocate
from tests.components.completion_support import completion_outputs

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


class _Confidence:
    def __init__(self, batch, depth, device):
        self.values = torch.full((batch, depth), .75, device=device)
        self.output = torch.empty_like(self.values)

    def __call__(self):
        self.output.copy_(self.values)


class _ProposalBackbone:
    """Fixed one-hot proposals with the production admission and RNG mechanisms."""

    def __init__(self, owner, depth, cursor):
        self.owner, self.batch, self.width, self.buffers = owner, 4, depth + 1, cursor
        self.rows = torch.arange(4, device="cuda")
        self.active = cursor.zeros((4,), torch.bool)
        self.before = cursor.zeros((4,), torch.int64)
        self.base_offset = cursor.zeros((4,), torch.int64)
        self.candidates = cursor.zeros((4, depth + 1), torch.int64)
        self.drafts = cursor.zeros((4, depth), torch.int64)
        self.probabilities = cursor.zeros((4, depth, 256), torch.float32)
        self.valid = cursor.zeros((4, depth), torch.bool)
        self.validator = PreparedBatch(owner.inputs, owner.requests, Metadata.allocate(4, 4, 2, "cuda"),
            mode="decode", shard=owner.weights.shard, pool=owner.pool, verification_width=depth + 1)
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=owner.collectives["runtime"], coordinator=owner.collectives["coordinator"],
            session=owner.session, name="test.proposal.validation")

    def propose(self, advance_rng=True, observer=None, allowed=None):
        o = self.owner
        self.validator.begin(self.validation(self.validator.validate()))
        safe = o.inputs.row_slot_ids.long().clamp(0, 3)
        self.active.copy_(self.validator.token_state.active.bool())
        if allowed is not None:
            self.active.logical_and_(allowed)
        self.before.copy_(o.requests.consumed_tokens[safe])
        self.base_offset.copy_(torch.where(o.generation[safe] == o.requests.generation[safe], o.offset[safe], 0))
        self.candidates[:, 0].copy_(o.requests.pending_token[safe])
        for position in range(self.width - 1):
            self.drafts[:, position].fill_(5 + position)
            self.valid[:, position].copy_(self.active & (position < o.inputs.query_lengths - 1))
        self.candidates[:, 1:].copy_(self.drafts)
        self.probabilities.zero_()
        self.probabilities.scatter_(2, self.drafts[:, :, None], 1.)
        if advance_rng:
            self.commit_draft_rng()

    commit_draft_rng = PreparedParallelDraftRound.commit_draft_rng


class _VerificationTarget(SimpleNamespace):
    def verify(self, proposal_valid):
        self.distributions()
        self.stochastic()
        self.stochastic.output.valid.logical_and_(proposal_valid)
        self.commit()


def _fixture(depth, *, token_capacity=12, rank=0, world=1, runtime=None, broadcast=None, coordinator=None, session=None):
    batch, width, vocab, device = 4, 8, 256, "cuda"
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    for slot in range(batch):
        state.admit(slot, RequestAdmission(generation=11 + slot, seed=41800 + slot,
            max_new_tokens=100, top_k=-1, top_p=1.), [4, 5, 4])
    state.pending_token.fill_(4)
    state.consumed_tokens.fill_(10)
    state.generated_tokens.fill_(3)
    state.rng_counter.fill_(9)
    inputs = BatchInputs.allocate(batch, 2, device, capacity=batch)
    inputs.row_slot_ids.copy_(torch.tensor([3, 0, 2, 1], dtype=torch.int32, device=device))
    inputs.row_generations.copy_(state.generation[inputs.row_slot_ids.long()])
    inputs.live_rows.fill_(batch)
    inputs.request_state_rows[:, 0].copy_(torch.arange(1, 5, device=device))
    inputs.page_tables.copy_(torch.arange(8, dtype=torch.int32, device=device).view(4, 2))
    inputs.token_ids.fill_(4)
    inputs.sample_enabled.fill_(1)
    prefix_limits = torch.full((batch,), depth + 1, dtype=torch.int32, device=device)
    inputs.query_lengths.copy_(prefix_limits)
    records = torch.empty((2, batch, 24), dtype=torch.int64, device=device)
    owner = SimpleNamespace(requests=state, inputs=inputs, sequence=torch.tensor([6], device=device),
        frame=torch.zeros(1, dtype=torch.int32, device=device), completion=completion_outputs(records),
        offset=torch.tensor([8, 12, 16, 20], device=device), generation=state.generation.clone(),
        weights=SimpleNamespace(shard=TensorParallelShard(rank, world)),
        pool=SimpleNamespace(memory_capacity=torch.tensor([16, 16], device=device)),
        collectives=dict(runtime=runtime, vocabulary_runtime=broadcast, coordinator=coordinator), session=session)
    pool = GraphBufferPool(device, chunk_bytes=1 << 20)
    cursor = pool.cursor()
    source = _ProposalBackbone(owner, depth, cursor)
    retained = PreparedProposalStorage(owner, 7)
    confidence = _Confidence(batch, depth, device)
    proposal = SimpleNamespace(source=source, storage=retained, confidence=confidence,
        allowed=torch.empty_like(source.active), depth=torch.full_like(source.before, depth))
    target_cursor = pool.cursor()
    target = _VerificationTarget(owner=owner, batch=batch, width=width, target_token_capacity=token_capacity,
        buffers=target_cursor, active=target_cursor.zeros((batch,), torch.bool),
        before=target_cursor.zeros((batch,), torch.int64), base_offset=target_cursor.zeros((batch,), torch.int64),
        candidates=target_cursor.zeros((batch, width), torch.int64), drafts=target_cursor.zeros((batch, width - 1), torch.int64),
        probabilities=target_cursor.zeros((batch, width - 1, vocab), torch.float32),
        valid=target_cursor.zeros((batch, width - 1), torch.bool))
    logits = torch.zeros((batch, width, vocab), device=device)
    distributions = PreparedSpeculativeDistributions(logits, state, inputs.row_slot_ids,
        inputs.row_generations, inputs.live_rows, target.drafts)
    distributions.prepare()
    rng = SpeculativeRngState.allocate(device, capacity=batch)
    rng.generation.copy_(state.generation)
    rng.target_offset.copy_(torch.tensor([120, 88, 4, 52], device=device))
    stochastic = PreparedStochasticVerification(target.probabilities, target.drafts,
        distributions.probabilities, state, inputs.row_slot_ids, inputs.row_generations, inputs.live_rows,
        rng=rng, draft_valid=target.valid, target_valid=distributions.valid, token_limits=inputs.query_lengths)
    stochastic.prepare()
    commit = PreparedSpeculativeCommit(state, stochastic.output, target.candidates, inputs.row_slot_ids,
        inputs.row_generations, inputs.live_rows, owner.completion, owner.frame, owner.sequence,
        rng=rng, logits=logits, token_limits=inputs.query_lengths, draft_context_lag=0)
    commit.prepare()

    target.distributions, target.stochastic, target.commit = distributions, stochastic, commit
    verification = PreparedDraftVerification(retained, target)
    operation = PreparedPackedSpeculativeRound(proposal, verification, prefix_limits,
                                              name="test.packed", buffers=pool.cursor())
    return SimpleNamespace(**locals())


def _publish(f, admitted):
    s, p, o = f.source, f.retained, f.owner
    slots = o.inputs.row_slot_ids
    safe = slots.long().clamp(0, 3)
    s.active.logical_and_(admitted)
    for value, destination in ((o.requests.generation[safe], p.generations),
        (o.inputs.request_state_rows[safe, 0], p.state_rows), (o.sequence.expand(4), p.sequence),
        (f.proposal.depth, p.depths), (s.before, p.before), (s.base_offset, p.base_offset),
        (s.candidates, p.candidates), (s.drafts, p.drafts), (s.probabilities, p.probabilities),
        (s.valid, p.valid), (s.active, p.ready)):
        scatter_rows(value, slots, destination, s.active)


@pytest.mark.parametrize("depth,compiled,token_capacity", [
    (1, False, 12), (3, False, 12), (5, False, 12), (3, True, 12),
    (3, False, 5), (3, True, 7), (5, False, 13), (5, True, 13),
])
@torch.no_grad()
def test_captured_packed_round_matches_split_oracle_state_rng_and_poison(depth, compiled, token_capacity):
    f = _fixture(depth, token_capacity=token_capacity)
    state, inputs, owner = f.state, f.inputs, f.owner
    originals = {name: value.clone() for name, value in vars(state).items()}
    rng_originals = {name: value.clone() for name, value in vars(f.rng).items()}
    input_originals = {field.name: getattr(inputs, field.name).clone() for field in fields(inputs)}
    offset = owner.offset.clone()
    generation = owner.generation.clone()
    generator = torch.Generator().manual_seed(719)
    dense_logits = torch.randn(f.logits.shape, generator=generator).cuda()
    f.logits.copy_(dense_logits)

    def restore(case, poison):
        for name, value in originals.items():
            getattr(state, name).copy_(value)
        for name, value in rng_originals.items():
            getattr(f.rng, name).copy_(value)
        for name, value in input_originals.items():
            getattr(inputs, name).copy_(value)
        owner.offset.copy_(offset)
        owner.generation.copy_(generation)
        f.prefix_limits.fill_(depth + 1)
        f.prefix_limits[0] = 1
        f.confidence.values.fill_(.75)
        f.logits.copy_(dense_logits)
        if case == "accept":
            f.logits.fill_(-torch.inf)
            for position in range(8):
                f.logits[:, position, 5 + position] = 8.
        if case == "confidence":
            f.confidence.values[1, -1] = torch.nan
        elif case == "stale":
            inputs.row_generations[1] += 1
        elif case == "duplicate":
            inputs.row_slot_ids[1] = inputs.row_slot_ids[0]
            inputs.row_generations[1] = inputs.row_generations[0]
        elif case == "inactive":
            inputs.live_rows.fill_(2)
            f.confidence.values[2:] = torch.nan
        f.retained.ready.zero_()
        for tensor in (f.retained.probabilities, f.retained.candidates, f.retained.drafts,
                       f.retained.valid, f.target.probabilities):
            tensor.fill_(0)
        if poison:
            f.retained.probabilities[:, depth:] = torch.nan
            f.retained.drafts[:, depth:] = 999
            f.retained.candidates[:, depth + 1:] = -777
        owner.completion.records.fill_(-91)
        f.distributions.probabilities.fill_(torch.nan)

    operation = compile_forward(f.operation) if compiled else f.operation
    restore("valid", False)
    operation()
    restore("valid", False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()

    def result():
        return [value.cpu().clone() for value in (*vars(state).values(), *vars(f.rng).values(),
            owner.offset, owner.generation, owner.completion.records, f.commit.emitted_count,
            inputs.query_lengths, f.retained.ready)]

    try:
        for case in ("valid", "accept", "confidence", "stale", "duplicate", "inactive"):
            for poison in (False, True):
                restore(case, poison)
                inputs.query_lengths.copy_(f.prefix_limits)
                safe = inputs.row_slot_ids.long().clamp(0, 3)
                f.proposal.allowed.copy_(~f.retained.ready[safe])
                f.source.propose(True, f.confidence, f.proposal.allowed)
                f.confidence()
                submitted = (f.source.rows < inputs.live_rows).tolist()
                lengths, admitted = allocate(f.confidence.output.tolist(), inputs.row_slot_ids.tolist(),
                    submitted, f.source.active.tolist(), f.prefix_limits.tolist(), token_capacity, 4)
                inputs.query_lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
                _publish(f, torch.tensor(admitted, dtype=torch.bool, device="cuda"))
                f.verification(same_graph=True)
                expected = result()
                restore(case, poison)
                frozen_limits = f.prefix_limits.clone()
                frozen_slots = inputs.row_slot_ids.clone()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                assert torch.equal(f.prefix_limits, frozen_limits)
                assert torch.equal(inputs.row_slot_ids, frozen_slots)
                for left, right in zip(expected, result(), strict=True):
                    assert torch.equal(left, right), (depth, token_capacity, case, poison, left.shape, left.dtype)
                assert inputs.query_lengths.sum().item() <= token_capacity
                if case == "accept":
                    assert torch.equal(f.commit.emitted_count, inputs.query_lengths)
                if case == "confidence":
                    assert torch.equal(owner.offset, offset + 4 * depth)
                    for name, value in originals.items():
                        assert torch.equal(getattr(state, name), value), name
                if case in ("confidence", "stale", "duplicate"):
                    assert not f.retained.ready.any()
                    assert torch.equal(f.rng.target_offset, rng_originals["target_offset"])
    finally:
        graph.reset()


@pytest.mark.parametrize("survival", [False, True])
@torch.no_grad()
def test_native_packed_signature_prefix_upload_and_terminal_feedback(survival):
    import struct
    import time
    from ignition.preparation.execution import GraphSignature, PreparedExecution

    f = _fixture(3)
    initial_state = {name: value.clone() for name, value in vars(f.state).items()}
    initial_rng = {name: value.clone() for name, value in vars(f.rng).items()}
    initial_offset, initial_generation = f.owner.offset.clone(), f.owner.generation.clone()
    f.logits.fill_(-torch.inf)
    for position in range(8):
        f.logits[:, position, 5 + position] = 8.

    def restore(limits):
        for name, value in initial_state.items():
            getattr(f.state, name).copy_(value)
        for name, value in initial_rng.items():
            getattr(f.rng, name).copy_(value)
        f.owner.offset.copy_(initial_offset)
        f.owner.generation.copy_(initial_generation)
        f.retained.ready.zero_()
        f.retained.probabilities[:, 3:] = torch.nan
        f.retained.drafts[:, 3:] = 999
        f.retained.candidates[:, 4:] = -777
        f.prefix_limits.copy_(torch.tensor(limits, dtype=torch.int32, device="cuda"))
        f.inputs.query_lengths.fill_(-111)
        f.target.probabilities.fill_(torch.nan)
        f.owner.completion.records.fill_(-91)
        f.owner.completion.speculative_survival.fill_(-91.)
        f.logits[:, :, 101].fill_(8. - float(limits[0]))

    def snapshot():
        return [tensor.cpu().clone() for tensor in (*vars(f.state).values(), *vars(f.rng).values(),
                f.owner.offset, f.owner.generation, f.inputs.query_lengths, f.commit.emitted_count)]

    operation = compile_forward(f.operation)
    variants = ([1, 4, 2, 4], [4, 1, 4, 3], [2, 2, 2, 2])
    expected = []
    for limits in variants:
        restore(limits)
        operation()
        torch.cuda.synchronize()
        expected.append((snapshot(), f.owner.completion.records[0].cpu().clone(),
                         f.owner.completion.speculative_survival[0].cpu().clone()))
    restore(variants[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    model = PreparedExecution(device=torch.cuda.current_device(), max_active=4, graphs={"packed": graph},
        signatures={"packed": GraphSignature("packed_speculative", 4, 8, ("speculative_prefix_limits",),
            speculation_enabled=True, packed_token_capacity=12, draft_tokens=3)},
        buffers=dict(frame_index=f.owner.frame, submission_sequence=f.owner.sequence,
            completion_records=f.owner.completion.records, speculative_prefix_limits=f.prefix_limits,
            proposal_confidence=f.operation.confidence,
            **({"completion_speculative_survival": f.owner.completion.speculative_survival} if survival else {})),
        resources=(f, operation))
    manifest = model.manifest()
    assert "query_lengths" not in {entry["buffer_name"] for entry in manifest["graphs"][0]["input_bindings"]}
    executor = model.start_executor()
    try:
        with pytest.raises(RuntimeError, match="not bound"):
            executor.submit(0, [("query_lengths", 0, bytes(16))])
        for limits, (expected_state, expected_records, expected_survival) in zip(variants, expected, strict=True):
            restore(limits)
            f.prefix_limits.zero_()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = executor.submit(0, [("speculative_prefix_limits", 0, struct.pack("4i", *limits))])
            assert fence is not None
            deadline = time.monotonic() + 30
            completion = None
            while completion is None:
                assert time.monotonic() < deadline
                completion = executor.poll()
                if completion is None:
                    time.sleep(.001)
            assert completion.fence == fence
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            for left, right in zip(expected_state, snapshot(), strict=True):
                assert torch.equal(left, right), (left.shape, left.dtype)
            actual_records = torch.tensor(completion.records)
            # Native transport owns the submission sequence and active frame.
            actual_records[:, 1] = expected_records[:, 1]
            assert torch.equal(actual_records, expected_records)
            assert completion.proposal_confidence == [[.75] * 3 + [0.] * 4] * 4
            if survival:
                assert torch.equal(torch.tensor(completion.speculative_survival), expected_survival)
                for record, row in zip(completion.records, completion.speculative_survival, strict=True):
                    observed = record[9] if record[8] > record[10] else record[10]
                    assert all((value >= 0) == (depth < observed) for depth, value in enumerate(row))
            else:
                assert completion.speculative_survival is None
            assert f.prefix_limits.tolist() == list(limits)
            assert executor.outstanding == 0
    finally:
        executor.close()
        assert executor.drained
        graph.reset()


def _tp_rank(rank, rendezvous, sockets):
    from contextlib import ExitStack
    import struct
    import time
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from datetime import timedelta
    import torch.distributed as dist
    from ignition.preparation.collectives import (CollectivePreparation, create_all_reduce_runtime,
                                                  create_result_broadcast_runtime)
    from ignition.preparation.tuning import RecordedPreparationSession

    peers, connection = [], None
    for index, (root, peer) in enumerate(sockets, start=1):
        if rank == 0:
            peers.append(root)
            peer.close()
        else:
            root.close()
            if rank == index:
                connection = peer
            else:
                peer.close()
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=3))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_all_reduce_runtime(dist.group.WORLD, torch.device("cuda", rank), max_tokens=1, hidden_size=8)
    broadcast = create_result_broadcast_runtime(dist.group.WORLD, torch.device("cuda", rank),
        layouts=[dict(confidence=((4, 3), torch.float32))])
    graph, native_graph, executor = None, None, None
    try:
        with RecordedPreparationSession(device=torch.device("cuda", rank), checkpoint="model-free-prefix-validation",
                workload_schema="causal-prefix-tp4", numerical_policy="FP64-frontier", coordinator=coordinator,
                compile_workers=4) as session:
            f = _fixture(3, rank=rank, world=4, runtime=runtime, broadcast=broadcast,
                         coordinator=coordinator, session=session)
            session.tune()
            initial_state = {name: value.clone() for name, value in vars(f.state).items()}
            initial_rng = {name: value.clone() for name, value in vars(f.rng).items()}
            initial_offset = f.owner.offset.clone()
            f.prefix_limits.copy_(torch.tensor([1, 4, 4, 4], dtype=torch.int32, device="cuda"))
            f.logits.fill_(-torch.inf)
            for position in range(8):
                f.logits[:, position, 5 + position] = 8.

            def restore(case):
                for name, value in initial_state.items():
                    getattr(f.state, name).copy_(value)
                for name, value in initial_rng.items():
                    getattr(f.rng, name).copy_(value)
                f.owner.offset.copy_(initial_offset)
                f.retained.ready.zero_()
                f.owner.completion.records.fill_(-91 - rank)
                f.inputs.live_rows.fill_(2 if case == "partial" else 4)
                f.prefix_limits[2:].fill_(0 if case == "partial" else 4)
                f.confidence.values.fill_(.25 * (rank + 1))
                if case == "rank2_nan" and rank == 2 or case == "root_nan" and rank == 0:
                    f.confidence.values[1, -1] = torch.nan
                if case == "partial":
                    f.confidence.values[2:] = torch.nan

            restore("valid")
            f.operation()
            torch.cuda.synchronize()
            restore("valid")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                stack.enter_context(runtime.capture(stream=stream))
                stack.enter_context(session.capture())
                stack.enter_context(torch.cuda.graph(graph, stream=stream))
                f.operation()
            for case in ("valid", "partial", "rank2_nan", "root_nan"):
                restore(case)
                torch.cuda.synchronize()
                dist.barrier()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                actual_lengths = f.inputs.query_lengths.tolist()
                all_lengths = [None] * 4
                dist.all_gather_object(all_lengths, actual_lengths)
                assert all_lengths == [actual_lengths] * 4
                if case == "root_nan":
                    assert torch.isnan(f.operation.confidence[1, -1])
                else:
                    live = 2 if case == "partial" else 4
                    assert f.operation.confidence[:live].eq(.25).all()
                    assert f.operation.confidence[live:].eq(0).all()
                expected_offset = initial_offset.clone()
                expected_offset[[3, 0] if case == "partial" else [0, 1, 2, 3]] += 12
                assert torch.equal(f.owner.offset, expected_offset)
                if case in ("rank2_nan", "root_nan"):
                    assert actual_lengths == [0] * 4
                    assert not f.retained.ready.any()
                    for name, value in initial_state.items():
                        assert torch.equal(getattr(f.state, name), value), name
                    for name, value in initial_rng.items():
                        assert torch.equal(getattr(f.rng, name), value), name
                else:
                    expected, _ = allocate([[.25] * 3] * 4, [3, 0, 2, 1],
                                          [True, True, False, False] if case == "partial" else [True] * 4,
                                          [True] * 4, [1, 4, 4, 4], 12, 4)
                    assert actual_lengths == expected
                    assert f.commit.emitted_count.tolist() == expected
                    if case == "partial":
                        for name, value in initial_state.items():
                            assert torch.equal(getattr(f.state, name)[1:3], value[1:3]), name
                        assert torch.equal(f.owner.completion.records[0, [1, 2]], f.commit.idle_record.expand(2, -1))
            # Reset inside the graph so native peer workers can replay
            # independent cohorts without Python touching their owned state.
            failure = torch.zeros(1, dtype=torch.int32, device="cuda")
            initial_generation = f.owner.generation.clone()

            def native_forward():
                for name, value in initial_state.items():
                    getattr(f.state, name).copy_(value)
                for name, value in initial_rng.items():
                    getattr(f.rng, name).copy_(value)
                f.owner.offset.copy_(initial_offset)
                f.owner.generation.copy_(initial_generation)
                f.retained.ready.zero_()
                f.owner.completion.records.fill_(-91 - rank)
                f.confidence.values.fill_(.25 * (rank + 1))
                if rank == 2:
                    f.confidence.values.masked_fill_(failure[0] == 1, torch.nan)
                f.confidence.values[2:].masked_fill_(failure[0] == 2, torch.nan)
                f.operation()

            native_forward()
            torch.cuda.synchronize()
            stream.wait_stream(torch.cuda.current_stream())
            native_graph = torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                stack.enter_context(runtime.capture(stream=stream))
                stack.enter_context(session.capture())
                stack.enter_context(torch.cuda.graph(native_graph, stream=stream))
                native_forward()
            session.freeze()
            model = PreparedExecution(device=rank, max_active=4, graphs={"packed": native_graph},
                signatures={"packed": GraphSignature("packed_speculative", 4, 8,
                    ("speculative_prefix_limits", "confidence_failure", "live_rows"), speculation_enabled=True,
                    packed_token_capacity=12, draft_tokens=3)},
                buffers=dict(frame_index=f.owner.frame, submission_sequence=f.owner.sequence,
                    completion_records=f.owner.completion.records, speculative_prefix_limits=f.prefix_limits,
                    proposal_confidence=f.operation.confidence, confidence_failure=failure, live_rows=f.inputs.live_rows,
                    completion_speculative_survival=f.owner.completion.speculative_survival),
                resources=(f, native_forward, initial_state, initial_rng, session, runtime, broadcast, stream))
            torch.cuda.synchronize()
            dist.barrier()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            if rank:
                model.serve_rank(connection, rank=rank, world=4)
            else:
                executor = model.start_executor(peers=peers)
                for case, mode, live in (("valid", 0, 4), ("partial", 2, 2), ("rank2_nan", 1, 4)):
                    limits = (1, 4, 0, 0) if case == "partial" else (1, 4, 4, 4)
                    fence = executor.submit(0, [("speculative_prefix_limits", 0, struct.pack("4i", *limits)),
                        ("confidence_failure", 0, struct.pack("i", mode)), ("live_rows", 0, struct.pack("i", live))])
                    assert fence is not None
                    deadline, completion = time.monotonic() + 30, None
                    while completion is None:
                        assert time.monotonic() < deadline
                        completion = executor.poll()
                        if completion is None:
                            time.sleep(.001)
                    assert completion.fence == fence
                    assert completion.proposal_confidence == ([[.25] * 3 + [0.] * 4] * live
                                                               + [[0.] * 7] * (4 - live))
                    assert completion.speculative_survival is not None
                    if case == "rank2_nan":
                        assert all(row[5:7] == [3, 256] for row in completion.records)
                        assert completion.speculative_survival == [[-1.] * 7] * 4
                    else:
                        admitted, _ = allocate([[.25] * 3] * 4, [3, 0, 2, 1],
                            [row < live for row in range(4)], [True] * 4, [1, 4, 4, 4], 12, 4)
                        by_slot = [0] * 4
                        for row, slot in enumerate([3, 0, 2, 1]):
                            by_slot[slot] = admitted[row]
                        assert [record[8] for record in completion.records] == by_slot
                        if case == "partial":
                            assert all(completion.records[slot] == f.commit.idle_record.tolist() for slot in (1, 2))
                            assert all(completion.speculative_survival[slot] == [-1.] * 7 for slot in (1, 2))
                executor.close()
                assert executor.drained
                executor = None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert f.inputs.query_lengths.eq(0).all()
            assert torch.equal(f.owner.offset, initial_offset + 12)
            for name, value in initial_state.items():
                assert torch.equal(getattr(f.state, name), value), name
            for name, value in initial_rng.items():
                assert torch.equal(getattr(f.rng, name), value), name
            dist.barrier()
    finally:
        if executor is not None:
            executor.close()
        if native_graph is not None:
            native_graph.reset()
        if graph is not None:
            graph.reset()
        broadcast.close()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                    reason="requires four explicitly assigned helper GPUs")
def test_tp4_rank_zero_confidence_and_unanimous_rejection(tmp_path):
    import socket
    import torch.multiprocessing as mp
    assert torch.cuda.device_count() == 4
    sockets = [socket.socketpair() for _ in range(3)]
    try:
        mp.spawn(_tp_rank, args=((tmp_path / "rendezvous").as_uri(), sockets), nprocs=4, join=True)
    finally:
        for pair in sockets:
            for connection in pair:
                connection.close()


@torch.no_grad()
def test_worker_serialized_packed_round_restores_and_captures(tmp_path, monkeypatch):
    from b12x.preparation import PreparationSession
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature
    import ignition.preparation.aot as aot
    from ignition.preparation.compile_pool import CompilePool

    submitted = []
    submit_payload = CompilePool.submit_payload

    def observe_submission(compiler, payload, path):
        assert not path.exists()
        submitted.append(path)
        return submit_payload(compiler, payload, path)

    monkeypatch.setattr(CompilePool, "submit_payload", observe_submission)

    def reject_parent_retrace(*args, **kwargs):
        raise AssertionError("the owning process must bind the worker artifact without retracing")

    monkeypatch.setattr(aot, "aot_forward", reject_parent_retrace)
    f = _fixture(3)
    f.prefix_limits.copy_(torch.tensor([1, 4, 0, 0], dtype=torch.int32, device="cuda"))
    f.inputs.live_rows.fill_(2)
    f.confidence.values[2:] = torch.nan
    f.owner.completion.records.fill_(-91)
    f.logits.fill_(-torch.inf)
    for position in range(8):
        f.logits[:, position, 5 + position] = 8.
    initial_state = {name: value.clone() for name, value in vars(f.state).items()}
    initial_rng = {name: value.clone() for name, value in vars(f.rng).items()}
    initial_offset, initial_generation = f.owner.offset.clone(), f.owner.generation.clone()
    materialized = []

    def restore():
        for name, value in initial_state.items():
            getattr(f.state, name).copy_(value)
        for name, value in initial_rng.items():
            getattr(f.rng, name).copy_(value)
        f.owner.offset.copy_(initial_offset)
        f.owner.generation.copy_(initial_generation)
        f.retained.ready.zero_()
        f.owner.completion.records.fill_(-91)
        f.inputs.query_lengths.fill_(-111)

    def materialize():
        materialized.append(True)
        restore()

    graphs = {}
    # This fixture has no tunable b12x operators. A drained standalone session
    # still supplies the production kernel-resolution guard during capture.
    with PreparationSession(device=torch.device("cuda", 0), compile_workers=1,
            namespace=dict(fixture="causal-prefix-worker")) as session:
        session.requested_compile_workers = 1
        stream = torch.cuda.Stream()
        signatures = {"packed": GraphSignature("packed_speculative", 4, 8,
            ("speculative_prefix_limits", "live_rows"), speculation_enabled=True,
            packed_token_capacity=12, draft_tokens=3)}
        graphs, compiled = capture_forwards({"packed": f.operation}, signatures, session=session,
            device=torch.device("cuda", 0), stream=stream, pool=torch.cuda.graph_pool_handle(),
            materialize=materialize, cache_root=tmp_path, compile_kinds=("packed_speculative",))
        assert materialized == [True]
        assert len(submitted) == 1 and submitted[0].is_file()
        assert list(tmp_path.rglob("*.bin")) == submitted
        try:
            for _ in range(2):
                restore()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graphs["packed"].replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                assert f.inputs.query_lengths.tolist() == [1, 4, 0, 0]
                assert f.commit.emitted_count.tolist() == [1, 4, 0, 0]
                assert f.operation.confidence[:2].eq(.75).all()
                assert f.operation.confidence[2:].eq(0).all()
                assert torch.equal(f.owner.completion.records[0, [1, 2]], f.commit.idle_record.expand(2, -1))
                for name, value in initial_state.items():
                    assert torch.equal(getattr(f.state, name)[1:3], value[1:3]), name
        finally:
            for graph in graphs.values():
                graph.reset()
