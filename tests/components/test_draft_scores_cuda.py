"""Fused one-position draft scores match the unfused speculative score preparation."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not (os.environ.get("FLARE_RUN_CUDA_TESTS") and torch.cuda.is_available()),
                                reason="set FLARE_RUN_CUDA_TESTS=1 on a CUDA host")

VOCAB = 10000


def _state(generator):
    from ignition.components.sampling import RUNNING, DeviceRequestState

    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=4)
    state.generation.copy_(torch.tensor([7, 8, 9, 10]))
    state.status.fill_(RUNNING)
    state.status[3] = 0
    state.histogram.copy_(torch.randint(0, 3, (4, VOCAB), generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([[0.7, 0.9, 0.0, 1.3, 0.2, 0.05],
                                         [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                                         [1.2, 0.8, 0.1, 0.8, -0.1, 0.1],
                                         [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]))
    state.top_k.copy_(torch.tensor([0, 40, 0, 5], dtype=torch.int32))
    masks = torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape, generator=generator, dtype=torch.int64)
    state.grammar_mask.copy_(masks.to(torch.int32))
    state.grammar_enabled.copy_(torch.tensor([0, 1, 0, 1], dtype=torch.int32))
    return state


@pytest.mark.parametrize("prefix", [0, 1, 3])
@pytest.mark.parametrize("live", [4, 3])
@pytest.mark.parametrize("layout", ["duplicate", "stopped"])
def test_fused_draft_scores_match_reference(prefix, live, layout):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.speculation import PreparedSpeculativeDistributions

    generator = torch.Generator().manual_seed(prefix * 10 + live)
    state = _state(generator)
    batch = 4
    # Rows: ordinary, grammar-constrained, stale generation, then a duplicate of row 0 or a stopped slot.
    last = 0 if layout == "duplicate" else 3
    slots = torch.tensor([0, 1, 2, last], dtype=torch.int32, device="cuda")
    generations = torch.tensor([7, 8, 99, 7 if layout == "duplicate" else 10], dtype=torch.int64, device="cuda")
    live_rows = torch.tensor([live], dtype=torch.int32, device="cuda")
    logits = (torch.randn((batch, 1, VOCAB), generator=generator) * 4).to(torch.bfloat16).cuda()
    candidates = torch.randint(0, VOCAB, (batch, 4), generator=generator).cuda()
    candidates[1, 0] = -1
    reference = PreparedSpeculativeDistributions(logits, state, slots, generations, live_rows, candidates,
                                                 prefix_offset=prefix)
    reference.prepare_scores()
    seed, offset = torch.zeros(batch, dtype=torch.int64, device="cuda"), torch.zeros(batch, dtype=torch.int64,
                                                                                       device="cuda")
    sampled = torch.zeros(batch, dtype=torch.int64, device="cuda")
    fused = PreparedDraftDistribution(logits, state, slots, generations, live_rows, candidates,
                                      seed=seed, offset=offset, sampled=sampled, prefix_offset=prefix)
    fused.prepare()
    fused.prepare_scores()
    torch.cuda.synchronize()
    assert torch.equal(fused.valid, reference.valid), (fused.valid, reference.valid)
    rows = reference.valid[:, 0]
    assert torch.equal(fused.scores[rows], reference.scores[rows])
    assert torch.equal(fused.maximum[rows], reference.maximum[rows])
    assert torch.equal(fused.packed_top_k, reference.packed_top_k)
    assert torch.equal(fused.packed_top_p, reference.packed_top_p)


def _exact(actual, expected):
    if actual.dtype.is_floating_point:
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    else:
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch,vocab,dtype,compiled", [
    (1, 17, torch.float32, False),
    (4, 10000, torch.bfloat16, True),
    (8, 154880, torch.float32, True),
    (32, 129280, torch.bfloat16, True),
    (32, 248320, torch.bfloat16, False),
])
@pytest.mark.parametrize("prefix", [0, 3, 7])
@torch.no_grad()
def test_draft_policy_branches_preserve_scores_draws_and_native_replay(batch, vocab, dtype, compiled, prefix):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.triton import prepare_triton
    from .draft_score_reference import draft_scores_reference

    generator = torch.Generator().manual_seed(319 + prefix)
    capacity = max(4, batch)
    state = DeviceRequestState.allocate("cuda:0", vocab_size=vocab, capacity=capacity)
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.histogram.copy_(torch.randint(0, 10000, state.histogram.shape, generator=generator, dtype=torch.int32))
    state.grammar_mask.copy_(torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape,
        generator=generator, dtype=torch.int64).to(torch.int32))
    state.parameters.copy_(torch.tensor([1., .95, 0., 1., 0., 0.], device="cuda"))
    state.grammar_enabled.zero_()
    state.top_k.fill_(20 if vocab == 248320 else 0)
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    generations = torch.full((batch,), 7, dtype=torch.int64, device="cuda")
    live = torch.tensor([batch], dtype=torch.int32, device="cuda")
    candidates = torch.tensor([[7, 7, 0, 3, 4, 5, 6]] * batch, dtype=torch.int64, device="cuda")
    logits = torch.randn((batch, 1, vocab), generator=generator).to(device="cuda", dtype=dtype)
    initial_logits = logits.clone()
    seed = torch.arange(batch, dtype=torch.int64, device="cuda") + 731
    offset = torch.zeros(batch, dtype=torch.int64, device="cuda")
    operations = []
    for reference in (True, False):
        operation = PreparedDraftDistribution(logits, state, slots, generations, live, candidates,
            seed=seed, offset=offset, sampled=torch.zeros_like(seed), prefix_offset=prefix)
        operation.prepare()
        if reference:
            score, finish = operation._score_kernels
            operation._score_kernels = (prepare_triton(draft_scores_reference,
                score.args, score.grid, **score.options), finish)
        operations.append(operation)
    calls = [compile_forward(operation) if compiled else operation for operation in operations]
    graphs = {}
    for name, call in zip(("reference", "candidate"), calls, strict=True):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={}, resources=(operations, calls, state))
    worker = bundle.start_worker()
    modes = ("neutral", "penalties", "grammar", "penalties_and_grammar", "mixed", "negative_zero_presence",
        "negative_zero_frequency", "tiny_temperature", "nonfinite", "invalid_prefix",
        "inactive", "invalid_rows", "neutral")
    try:
        for phase, mode in enumerate(modes):
            state.parameters.copy_(torch.tensor([1., .95, 0., 1., 0., 0.], device="cuda"))
            state.grammar_enabled.zero_()
            state.status.fill_(RUNNING)
            slots.copy_(torch.arange(batch, dtype=torch.int32, device="cuda"))
            generations.fill_(7)
            live.fill_(batch)
            candidates.copy_(torch.tensor([[7, 7, 0, 3, 4, 5, 6]] * batch, device="cuda"))
            logits.copy_(initial_logits)
            logits[:, :, 0] = 0.
            logits[:, :, 1] = -0.
            if mode in ("penalties", "penalties_and_grammar", "mixed"):
                selected = torch.arange(capacity, device="cuda") % 2 == 1
                if mode != "mixed":
                    selected.fill_(True)
                state.parameters[selected] = torch.tensor([.7, .8, .1, 1.3, -.2, -.1], device="cuda")
            if mode in ("grammar", "penalties_and_grammar"):
                state.grammar_enabled.fill_(1)
            elif mode == "mixed":
                state.grammar_enabled[::2] = 1
            if mode == "negative_zero_presence":
                state.parameters[:, 4] = -0.
            if mode == "negative_zero_frequency":
                state.parameters[:, 5] = -0.
            if mode == "tiny_temperature":
                state.parameters[:, 0] = torch.finfo(torch.float32).tiny
            if mode == "nonfinite":
                logits[:, :, 2:5] = torch.tensor([float("nan"), float("inf"), -float("inf")],
                    dtype=dtype, device="cuda")
            if mode == "invalid_prefix":
                candidates[:, 0] = -1
            if mode == "inactive":
                live.fill_(batch - 1)
            if mode == "invalid_rows":
                slots[0] = -1
                if batch > 1:
                    slots[1] = capacity
                    generations[-1] = 99
                    state.status[2] = 0
            offset.fill_(phase * 28)
            for operation in operations:
                operation.scores.fill_(float("nan"))
                operation.probabilities.fill_(float("nan"))
                operation.block_max.fill_(float("nan"))
                operation.sampled.fill_(-1)
            before = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
            seed_before, offset_before = seed.clone(), offset.clone()
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            for name in ("candidate", "reference", "candidate"):
                ticket = worker.submit(bundle.graph_indices[name], repetitions=3)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            reference, candidate = operations
            for name in ("scores", "block_max", "maximum", "valid", "packed_top_k", "packed_top_p",
                         "probabilities", "sampled"):
                _exact(getattr(candidate, name), getattr(reference, name))
            for name, expected in before.items():
                actual = getattr(state, name)
                if actual.dtype == torch.uint64:
                    actual, expected = actual.view(torch.int64), expected.view(torch.int64)
                assert torch.equal(actual, expected), name
            assert torch.equal(seed, seed_before) and torch.equal(offset, offset_before)
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
