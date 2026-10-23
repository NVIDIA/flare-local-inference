"""The prepared reduction reads raw logits and only the committed positions."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='CUDA opt-in')

@pytest.mark.parametrize('vocab', [71, 4096, 248320])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_raw_reduction_ties_frames_and_replay(vocab, dtype):
    from ignition.components.completion import CompletionOutputs
    from ignition.components.logprobs import PreparedLogprobs
    device = 'cuda'
    outputs = CompletionOutputs.allocate((2, 4, 24), device=device)
    logits = torch.randn((2, 3, vocab), device=device, dtype=dtype)
    # More than twenty equal maxima exercises the cutoff's ID ordering.
    logits[0, :, :32] = 20
    slots = torch.tensor([2, 0], dtype=torch.int32, device=device)
    epochs = torch.tensor([9, 10], dtype=torch.int64, device=device)
    live = torch.tensor([2], dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    outputs.requested[2] = 1
    record = outputs.records[0, 2]
    record[0], record[5], record[8] = 9, 1, 2
    record[16:18] = torch.tensor([vocab - 1, 0], device=device)
    outputs.logprobs.fill_(-777)
    op = PreparedLogprobs(logits, outputs, slots, epochs, live, frame)
    op.prepare()
    op()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op()
    for iteration in range(3):
        logits[0, :, 40:] += .5
        graph.replay()
        result = outputs.logprobs.cpu()
        assert torch.all(result[1] == -777)
        assert torch.all(result[0, :2] == -777)
        assert torch.all(result[0, 3] == -777)
        assert torch.all(result[0, 2, 2:] == -777)
        for position, token in enumerate([vocab - 1, 0]):
            _assert_entry(result[0, 2, position], logits[0, position], token, atol=1e-5)
    outputs.requested.zero_()
    outputs.logprobs.fill_(-777)
    graph.replay()
    assert torch.all(outputs.logprobs == -777)


def _snapshot(state):
    tensors = list(vars(state).values())
    saved = [(tensor, tensor.clone()) for tensor in tensors if isinstance(tensor, torch.Tensor)]
    def restore():
        for tensor, value in saved:
            tensor.copy_(value)
    return restore


def _assert_entry(output, logits, token, *, atol=2e-5, ids=None):
    raw = logits.float()
    expected = raw - torch.logsumexp(raw, 0)
    count = 20 if ids is None else len(ids)
    row = output.cpu()[:2 + 2 * count]
    assert row[0] == token
    torch.testing.assert_close(row[1:2].view(torch.float32), expected[token:token+1].cpu(), atol=atol, rtol=1e-5)
    ids = torch.argsort(expected, descending=True, stable=True)[:20] if ids is None else torch.tensor(ids, device=logits.device)
    torch.testing.assert_close(row[2::2].long(), ids.cpu())
    torch.testing.assert_close(row[3::2].contiguous().view(torch.float32), expected[ids].cpu(), atol=atol, rtol=1e-5)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('count', [0, 128])
def test_sampling_reports_raw_scores_before_every_sampler_transform(compiled, count):
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState, PreparedSampling, RequestAdmission
    from ignition.preparation.compilation import compile_forward
    outputs = CompletionOutputs.allocate((2, 4, 24), device='cuda')
    state = DeviceRequestState.allocate('cuda', vocab_size=4096)
    state.admit(2, RequestAdmission(9, 42, top_k=1, temperature=.3, top_p=.9,
        min_p=.1, repetition_penalty=1.8, presence_penalty=2, frequency_penalty=.5), [1, 2, 2])
    # Grammar forces a token outside raw top twenty.
    state.grammar_mask[2].zero_()
    state.grammar_mask[2, 1] = 1
    state.grammar_enabled[2] = 1
    logits = torch.linspace(10, -10, 4096, device='cuda')[None, :]
    slots = torch.tensor([2], dtype=torch.int32, device='cuda')
    epochs = torch.tensor([9], dtype=torch.int64, device='cuda')
    live = torch.tensor([1], dtype=torch.int32, device='cuda')
    frame = torch.tensor([0], dtype=torch.int32, device='cuda')
    sequence = torch.tensor([1], dtype=torch.int64, device='cuda')
    outputs.requested[2] = 1
    ids = list(reversed(range(count)))
    outputs.token_id_counts[2] = count
    outputs.token_ids[2, :count] = torch.tensor(ids, dtype=torch.int32, device='cuda')
    sampler = PreparedSampling(logits, state, slots, epochs, live, outputs, frame, sequence)
    sampler.prepare()
    restore = _snapshot(state)
    call = compile_forward(sampler) if compiled else sampler
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        restore()
        graph.replay()
        assert state.pending_token[2] == 32
        _assert_entry(outputs.logprobs[0, 2, 0], logits[0], 32, ids=ids or None)


@pytest.mark.parametrize('width', [1, 3, 8])
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('count', [0, 21])
def test_speculative_logprobs_follow_committed_target_positions_and_token_limits(width, compiled, count):
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.speculation import PreparedSpeculativeCommit, SpeculativeAcceptance
    from ignition.preparation.compilation import compile_forward
    device = 'cuda'
    state = DeviceRequestState.allocate(device, vocab_size=4096)
    for slot in range(4):
        state.admit(slot, RequestAdmission(9, 10, max_new_tokens=20), [1, 2], consumed_tokens=2)
        state.pending_token[slot] = 10
        state.generated_tokens[slot] = 1
    state.max_new_tokens[1] = 2
    inputs = torch.arange(10, 10+width, device=device).repeat(4, 1)
    acceptance = SpeculativeAcceptance.allocate(4, width, device)
    acceptance.token_ids.copy_(inputs + 1)
    acceptance.proposed_count.fill_(width)
    acceptance.accepted_drafts.fill_(width-1)
    acceptance.offered_drafts.fill_(width-1)
    acceptance.valid.fill_(True)
    slots = torch.tensor([2, 1, 3, 0], dtype=torch.int32, device=device)
    epochs = torch.tensor([9, 9, 8, 9], dtype=torch.int64, device=device)
    live = torch.tensor([3], dtype=torch.int32, device=device)
    limits = torch.tensor([width, 1, width, width], dtype=torch.int32, device=device)
    outputs = CompletionOutputs.allocate((2, 4, 24), device=device)
    outputs.requested.fill_(1)
    ids = list(reversed(range(count)))
    outputs.token_id_counts.fill_(count)
    outputs.token_ids[:, :count] = torch.tensor(ids, dtype=torch.int32, device=device)
    outputs.logprobs.fill_(-777)
    logits = torch.randn((4, width, 4096), dtype=torch.bfloat16, device=device)
    frame = torch.tensor([1], dtype=torch.int32, device=device)
    sequence = torch.tensor([7], dtype=torch.int64, device=device)
    commit = PreparedSpeculativeCommit(state, acceptance, inputs, slots, epochs, live,
        outputs, frame, sequence, logits=logits, token_limits=limits)
    commit.prepare()
    restore = _snapshot(state)
    def run():
        commit()
    call = compile_forward(run) if compiled else run
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        restore()
        outputs.logprobs.fill_(-777)
        graph.replay()
        assert outputs.records[1, 2, 8] == width
        assert outputs.records[1, 1, 8] == 1
        for position in range(width):
            _assert_entry(outputs.logprobs[1, 2, position], logits[0, position], 11+position, ids=ids or None)
        _assert_entry(outputs.logprobs[1, 1, 0], logits[1, 0], 11, ids=ids or None)
        assert torch.all(outputs.logprobs[0] == -777)
        assert torch.all(outputs.logprobs[1, 1, 1:] == -777)
        assert torch.all(outputs.logprobs[1, [0, 3]] == -777)
    # Duplicate live ownership is rejected by commit, so neither row may publish.
    restore()
    slots[1] = 2
    outputs.logprobs.fill_(-777)
    graph.replay()
    assert outputs.records[1, 2, 8] == 0
    assert torch.all(outputs.logprobs == -777)


def test_subtraction_rounding_ties_choose_lowest_ids_even_at_cutoff():
    from ignition.components.completion import CompletionOutputs
    from ignition.components.logprobs import PreparedLogprobs
    device = 'cuda'
    outputs = CompletionOutputs.allocate((2, 1, 8), device=device)
    outputs.requested.fill_(1)
    outputs.records[0, 0, 0] = 9
    outputs.records[0, 0, 4] = 3
    # Distinct logits become equal FP32 logprobs after normalization.
    logits = torch.arange(4096, device=device, dtype=torch.float32)[None, :] * 1e-12
    slot = torch.tensor([0], dtype=torch.int32, device=device)
    epoch = torch.tensor([9], device=device)
    live = torch.tensor([1], dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    reduction = PreparedLogprobs(logits, outputs, slot, epoch, live, frame)
    reduction.prepare()
    reduction()
    torch.testing.assert_close(outputs.logprobs[0, 0, 0, 2:42:2].cpu(), torch.arange(20, dtype=torch.int32))


@pytest.mark.parametrize('legacy_width', [None, 1, 7, 20])
@pytest.mark.parametrize('mapped', [False, True])
def test_native_two_frame_readback_copies_requesting_frames_and_decodes_requesting_lanes(legacy_width, mapped):
    import struct
    import time
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState, PreparedSampling, RequestAdmission
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    device = 'cuda'
    outputs = CompletionOutputs.allocate((2, 4, 24), device=device)
    state = DeviceRequestState.allocate(device, vocab_size=4096)
    for slot in (0, 2):
        state.admit(slot, RequestAdmission(9, 42, max_new_tokens=1000, top_k=1), [1])
    logits = torch.arange(4096, device=device, dtype=torch.float32).neg()[None, :].repeat(2, 1)
    slots = torch.tensor([2, 0], dtype=torch.int32, device=device)
    epochs = torch.tensor([9, 9], dtype=torch.int64, device=device)
    live = torch.tensor([2], dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    sequence = torch.tensor([1], dtype=torch.int64, device=device)
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    from dataclasses import replace
    mapping = None
    if mapped:
        shape = outputs.logprobs.shape if legacy_width is None else (2, 4, 8, 2 + 2 * legacy_width)
        mapping = MappedHostAllocation(shape, torch.int32, torch.device('cuda', 0))
        mapping.device_view.zero_()
    if legacy_width is None:
        if mapping is not None:
            outputs = replace(outputs, logprobs=mapping.device_view)
        legacy = None
    else:
        legacy = mapping.device_view if mapping is not None else torch.empty(
            (2, 4, 8, 2 + 2 * legacy_width), dtype=torch.int32, device=device)
    sampler = PreparedSampling(logits, state, slots, epochs, live, outputs, frame, sequence)
    sampler.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler()
        state.pending_token.fill_(-1)
        if legacy is not None:
            index = frame.long()
            legacy.index_copy_(0, index, outputs.logprobs.index_select(0, index)[..., :legacy.shape[-1]])
    buffers = outputs.buffers()
    if legacy is not None:
        buffers['completion_logprobs'] = legacy
        del buffers['logprob_token_ids'], buffers['logprob_token_id_counts']

    bundle = PreparedExecution(device=0, max_active=4, graphs={'sample': graph},
        signatures={'sample': GraphSignature('decode', 2, 1)},
        buffers={**buffers, 'frame_index': frame, 'submission_sequence': sequence},
        resources=(sampler, state, slots, epochs, live, outputs, mapping))
    executor = bundle.start_executor()
    def poll():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            value = executor.poll()
            if value is not None:
                return value
        pytest.fail('native frame transfer did not complete')
    expected = float(torch.log_softmax(logits[0], -1)[0])
    def submit(flags):
        uploads = [('logprob_requested', 0, struct.pack('4i', *flags))]
        if legacy is None:
            uploads.append(('logprob_token_id_counts', 0, bytes(16)))
        return executor.submit(0, uploads)
    try:
        allocated = torch.cuda.memory_stats()['allocation.all.allocated']
        assert submit([0, 0, 1, 0]) == 1
        assert submit([1, 0, 0, 0]) == 2
        for fence, slot in [(1, 2), (2, 0)]:
            result = poll()
            assert result.fence == fence
            assert result.logprobs[slot][0][0] == 0
            assert abs(result.logprobs[slot][0][1] - expected) < 2e-5
            assert len(result.logprobs[slot][0][2]) == (legacy_width or 20)
            assert all(entry is None for i, entries in enumerate(result.logprobs) if i != slot for entry in entries)
        if legacy is None:
            for fence, slot, ids in [(3, 2, list(reversed(range(1, 129)))), (4, 0, [0])]:
                counts = [0] * 4
                counts[slot] = len(ids)
                flags = [int(i == slot) for i in range(4)]
                assert executor.submit(0, [
                    ('logprob_requested', 0, struct.pack('4i', *flags)),
                    ('logprob_token_id_counts', 0, struct.pack('4i', *counts)),
                    ('logprob_token_ids', slot * 512, struct.pack(f'{len(ids)}i', *ids)),
                ]) == fence
            for fence, slot, expected_ids in [(3, 2, list(reversed(range(1, 129))) + [0]), (4, 0, [0])]:
                result = poll()
                assert result.fence == fence
                assert [id for id, _ in result.logprobs[slot][0][2]] == expected_ids
                assert result.logprobs[1 - slot // 2][0] is None
        # Stale payloads remain resident but this frame must not copy or decode them.
        assert submit([0, 0, 0, 0]) == (5 if legacy is None else 3)
        assert poll().logprobs is None
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
    finally:
        executor.close()


@pytest.mark.parametrize('compiled', [False, True])
def test_mixed_output_merge_preserves_the_inactive_frame_during_capture(compiled):
    from ignition.components.completion import CompletionOutputs
    from ignition.preparation.compilation import compile_forward
    outputs = CompletionOutputs.allocate((2, 4, 24), device='cuda')
    cursor = outputs.private_storage.cursor()
    prompt, decode = outputs.private(cursor), outputs.private(cursor)
    prompt.records.fill_(11)
    prompt.logprobs.fill_(12)
    prompt.speculative_survival.fill_(.875)
    decode.records.fill_(21)
    decode.logprobs.fill_(22)
    decode.speculative_survival.fill_(.5)
    outputs.token_id_counts.copy_(torch.tensor([1, 21, 128, 0], dtype=torch.int32, device='cuda'))
    frame = torch.tensor([1], dtype=torch.int32, device='cuda')
    slot = torch.tensor(2, device='cuda')
    def run():
        outputs.merge(prompt, decode, slot, frame)
    call = compile_forward(run) if compiled else run
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        outputs.records[0].fill_(-7)
        outputs.logprobs.fill_(-8)
        outputs.speculative_survival[0].fill_(.25)
        graph.replay()
        assert torch.all(outputs.records[0] == -7)
        assert torch.all(outputs.logprobs[0] == -8)
        assert torch.all(outputs.speculative_survival[0] == .25)
        assert torch.all(outputs.records[1, 2] == 11)
        assert torch.all(outputs.logprobs[1, 2, :, :42] == 12)
        for i, width in enumerate([20, 21, 128, 20]):
            assert torch.all(outputs.logprobs[1, i, :, :2 + 2 * width] == (12 if i == 2 else 22))
            assert torch.all(outputs.logprobs[1, i, :, 2 + 2 * width:] == -8)
        assert torch.all(outputs.speculative_survival[1, 2] == -1)
        assert torch.all(outputs.speculative_survival[1, [0, 1, 3]] == .5)


def _tp_logprob_rank(rank, world, sockets):
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState, PreparedSampling, RequestAdmission
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    import struct
    import time
    peers = []
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
    device = torch.device('cuda', rank)
    outputs = CompletionOutputs.allocate((2, 4, 8), device=device)
    state = DeviceRequestState.allocate(device, vocab_size=4096)
    state.admit(2, RequestAdmission(9, 42, top_k=1, max_new_tokens=100), [1])
    logits = -torch.arange(4096, dtype=torch.float32, device=device)[None, :]
    if rank:
        logits[0, 0] = 1000
    expected = float(torch.log_softmax(logits[0], -1)[0])
    slots = torch.tensor([2], dtype=torch.int32, device=device)
    epochs = torch.tensor([9], dtype=torch.int64, device=device)
    live = torch.tensor([1], dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    sequence = torch.tensor([1], dtype=torch.int64, device=device)
    sampler = PreparedSampling(logits, state, slots, epochs, live, outputs, frame, sequence)
    sampler.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler()
        state.pending_token.fill_(-1)
    bundle = PreparedExecution(device=rank, graphs={'sample': graph},
        signatures={'sample': GraphSignature('decode', 1, 1)},
        buffers={**outputs.buffers(), 'frame_index': frame, 'submission_sequence': sequence},
        resources=(sampler, state, slots, epochs, live))
    if rank:
        bundle.serve_rank(connection, rank=rank, world=world)
    else:
        executor = bundle.start_executor(peers=peers)
        try:
            with pytest.raises(RuntimeError, match='invalid logprob token ID count'):
                executor.submit(0, [
                    ('logprob_requested', 0, struct.pack('4i', 0, 0, 1, 0)),
                    ('logprob_token_id_counts', 0, struct.pack('4i', 0, 0, 129, 0)),
                ])
            assert executor.outstanding == 0
            for count in [0, 1, 20, 21, 127, 128, 0]:
                ids = list(reversed(range(1, count + 1)))
                uploads = [
                    ('logprob_requested', 0, struct.pack('4i', 0, 0, 1, 0)),
                    ('logprob_token_id_counts', 0, struct.pack('4i', 0, 0, count, 0)),
                ]
                if count:
                    uploads.append(('logprob_token_ids', 2 * 512, struct.pack(f'{count}i', *ids)))
                assert executor.submit(0, uploads) is not None
                result = None
                deadline = time.monotonic() + 30
                while result is None and time.monotonic() < deadline:
                    result = executor.poll()
                assert result is not None
                assert abs(result.logprobs[2][0][1] - expected) < 2e-5
                assert all(result.gpu_ns[i] is not None for i in range(world))
                if count:
                    alternatives = result.logprobs[2][0][2]
                    assert [id for id, _ in alternatives] == ids + [0]
                    reference = logits[0].double().log_softmax(0).cpu()
                    assert max(abs(lp - reference[id].item()) for id, lp in alternatives) < 2e-5
        finally:
            executor.close()
    torch.cuda.synchronize(device)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1', reason='requires assigned GPUs')
@pytest.mark.parametrize('world', [2, 4])
def test_native_tensor_parallel_reports_only_the_coordinating_ranks_values(world):
    import socket
    import torch.multiprocessing as mp
    sockets = [socket.socketpair() for _ in range(world - 1)]
    try:
        mp.spawn(_tp_logprob_rank, args=(world, sockets), nprocs=world, join=True)
    finally:
        for pair in sockets:
            for sock in pair:
                sock.close()


def test_packed_columns_beyond_a_short_request_never_read_another_requests_logits():
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.speculation import PreparedSpeculativeCommit, SpeculativeAcceptance
    from ignition.preparation.compilation import compile_forward
    device = 'cuda'
    state = DeviceRequestState.allocate(device, vocab_size=4096, capacity=2)
    for slot in (0, 1):
        state.admit(slot, RequestAdmission(9, 42, max_new_tokens=100), [1, 2], consumed_tokens=2)
        state.pending_token[slot] = 10
        state.generated_tokens[slot] = 1
    inputs = torch.tensor([[10, 11, 12], [10, 11, 12]], device=device)
    acceptance = SpeculativeAcceptance.allocate(2, 3, device)
    acceptance.token_ids.copy_(inputs + 1)
    acceptance.proposed_count.fill_(3)
    acceptance.accepted_drafts.fill_(2)
    acceptance.offered_drafts.fill_(2)
    acceptance.valid.fill_(True)
    packed = torch.randn((4, 4096), dtype=torch.bfloat16, device=device)
    # Materialize overlapping packed rows: the short request's unused columns
    # contain the following request's real rows, without sharing storage.
    logits = packed[torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)].contiguous()
    slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    epochs = torch.tensor([9, 9], device=device)
    live = torch.tensor([2], dtype=torch.int32, device=device)
    limits = torch.tensor([1, 3], dtype=torch.int32, device=device)
    outputs = CompletionOutputs.allocate((2, 2, 24), device=device)
    outputs.requested.fill_(1)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    sequence = torch.tensor([1], device=device)
    commit = PreparedSpeculativeCommit(state, acceptance, inputs, slots, epochs, live,
        outputs, frame, sequence, logits=logits, token_limits=limits)
    commit.prepare()
    restore = _snapshot(state)
    def run():
        commit()
    call = compile_forward(run)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    restore()
    outputs.logprobs.fill_(-777)
    graph.replay()
    _assert_entry(outputs.logprobs[0, 0, 0], packed[0], 11)
    assert torch.all(outputs.logprobs[0, 0, 1:] == -777)
    for col in range(3):
        _assert_entry(outputs.logprobs[0, 1, col], packed[1+col], 11+col)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('count', [0, 128])
def test_verification_reuses_filtering_scratch_but_reports_the_raw_target_logits(compiled, count):
    from ignition.components.completion import CompletionOutputs
    from ignition.components.parallel_draft import prepare_sampling_scratch
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.components.speculation import (PreparedSpeculativeDistributions,
        PreparedStochasticVerification, PreparedSpeculativeCommit, SpeculativeRngState)
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool
    device = 'cuda'
    state = DeviceRequestState.allocate(device, vocab_size=4096)
    state.admit(2, RequestAdmission(9, 42, top_k=1, max_new_tokens=20), [1, 2], consumed_tokens=2)
    state.pending_token[2], state.generated_tokens[2] = 10, 1
    inputs = torch.tensor([[10, 11, 12]], device=device)
    drafts = inputs[:, 1:].contiguous()
    raw = torch.randn((1, 3, 4096), dtype=torch.bfloat16, device=device)
    for position in range(3):
        raw[0, position, 11 + position] = 20
    slots = torch.tensor([2], dtype=torch.int32, device=device)
    epochs = torch.tensor([9], dtype=torch.int64, device=device)
    live = torch.tensor([1], dtype=torch.int32, device=device)
    pool = GraphBufferPool(device)
    scratch = prepare_sampling_scratch(1, 3, 4096, device, buffers=pool.cursor(), proposal=False)
    target = PreparedSpeculativeDistributions(raw, state, slots, epochs, live, drafts, workspace=scratch['target'])
    target.prepare()
    probabilities = torch.zeros((1, 2, 4096), device=device)
    probabilities.scatter_(2, drafts[:, :, None], 1)
    rng = SpeculativeRngState.allocate(device)
    verify = PreparedStochasticVerification(probabilities, drafts, target.probabilities,
        state, slots, epochs, live, rng=rng, target_valid=target.valid, workspace=scratch['verify'])
    verify.prepare()
    assert target.scores.data_ptr() == verify.safe_draft_probs.data_ptr()
    outputs = CompletionOutputs.allocate((2, 4, 24), device=device)
    outputs.requested[2] = 1
    ids = list(reversed(range(count)))
    outputs.token_id_counts[2] = count
    outputs.token_ids[2, :count] = torch.tensor(ids, dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    sequence = torch.tensor([1], dtype=torch.int64, device=device)
    commit = PreparedSpeculativeCommit(state, verify.output, inputs, slots, epochs, live,
        outputs, frame, sequence, logits=raw, rng=rng)
    commit.prepare()
    restore = _snapshot(state)
    def forward():
        target()
        verify()
        commit()
    call = compile_forward(forward) if compiled else forward
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    restore()
    graph.replay()
    assert outputs.records[0, 2, 8] == 3
    for position in range(3):
        _assert_entry(outputs.logprobs[0, 2, position], raw[0, position], 11 + position, ids=ids or None)
    restore()
    raw[0, 0, 11] = -20
    raw[0, 0, 101] = 20
    graph.replay()
    assert outputs.records[0, 2, 8] == 1
    _assert_entry(outputs.logprobs[0, 2, 0], raw[0, 0], 101, ids=ids or None)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_explicit_ids_replay_changing_widths_against_fp64(dtype):
    from ignition.components.completion import CompletionOutputs
    from ignition.components.logprobs import PreparedLogprobs
    device = 'cuda'
    outputs = CompletionOutputs.allocate((2, 2, 24), device=device)
    logits = torch.randn((2, 3, 4096), device=device, dtype=dtype) * 8
    slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    epochs = torch.tensor([9, 10], device=device)
    live = torch.tensor([2], dtype=torch.int32, device=device)
    frame = torch.tensor([0], dtype=torch.int32, device=device)
    outputs.requested.fill_(1)
    outputs.records[0, 1, 0] = 9
    outputs.records[0, 1, 8] = 2
    outputs.records[0, 1, 16:18] = 4000
    # The other row has a stale epoch, despite a live gate and count.
    outputs.records[0, 0, 0] = 9
    outputs.records[0, 0, 8] = 3
    outputs.token_id_counts[0] = 128
    op = PreparedLogprobs(logits, outputs, slots, epochs, live, frame)
    op.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op()
    for count in [128, 1, 20, 21, 127, 0, 128]:
        ids = torch.arange(count, device=device, dtype=torch.int32).flip(0) * 7
        outputs.token_id_counts[1] = count
        outputs.token_ids[1, :count] = ids
        outputs.logprobs.fill_(-777)
        graph.replay()
        result = outputs.logprobs.cpu()
        assert torch.all(result[1] == -777)
        assert torch.all(result[0, 0] == -777)
        assert torch.all(result[0, 1, 2:] == -777)
        for position in range(2):
            row = result[0, 1, position]
            raw = logits[0, position].double().cpu()
            reference = raw - torch.logsumexp(raw, 0)
            reported = row[1:2].view(torch.float32).double()
            # FP32 subtraction and the two-stage FP32 logsumexp dominate error.
            bound = 8 * torch.finfo(torch.float32).eps * (raw.abs().max() + torch.logsumexp(raw, 0).abs())
            assert abs(reported[0] - reference[4000]) <= bound
            if count:
                torch.testing.assert_close(row[2:2 + 2 * count:2], ids.cpu())
                values = row[3:2 + 2 * count:2].contiguous().view(torch.float32).double()
                assert (values - reference[ids.long().cpu()]).abs().max() <= bound
                assert torch.all(row[2 + 2 * count:] == -777)
            else:
                _assert_entry(row, logits[0, position], 4000)
    outputs.requested.zero_()
    outputs.logprobs.fill_(-777)
    graph.replay()
    assert torch.all(outputs.logprobs == -777)
