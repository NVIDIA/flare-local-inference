"""Batched unary shortlists preserve sequential predecessor corrections."""
import os

import pytest
import torch

from ignition.components.candidate_selector import PreparedCandidateSelector
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA GPU')


@pytest.mark.parametrize('batch', [1, 2, 4, 8])
@torch.no_grad()
def test_batched_shortlist_mutable_graph(batch):
    _check_shortlist_replay(batch, compiled_control=False)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@torch.no_grad()
def test_batched_shortlist_compiled_control(batch):
    _check_shortlist_replay(batch, compiled_control=True)


def _check_shortlist_replay(batch, *, compiled_control):
    torch.manual_seed(913)
    vocab, width, rank = 248320, 8, 256
    selector = object.__new__(PreparedCandidateSelector)
    selector.top_k = 16
    selector.weights = {name: torch.randn((vocab, rank), device='cuda', dtype=torch.bfloat16)
                        for name in ('predecessor', 'successor')}
    selector.projected = torch.randn((batch, width, rank), device='cuda', dtype=torch.bfloat16)
    selector.unary = torch.empty((batch, width, 16), device='cuda', dtype=torch.float32)
    selector.ids = torch.empty((batch, width, 16), device='cuda', dtype=torch.int64)
    source = torch.empty((batch, width, vocab), device='cuda', dtype=torch.float32)
    predecessor = torch.arange(batch, device='cuda', dtype=torch.int64)
    output = torch.empty((batch, width - 1, vocab), device='cuda', dtype=torch.float32)
    reference = torch.empty_like(output)

    def forward():
        selector.prepare_shortlists(source)
        for position in range(1, width):
            selector.apply_shortlist(output[:, position - 1], predecessor, position)

    source.normal_()
    compiled = compile_forward(forward)
    compiled()
    def sequential():
        for position in range(1, width):
            logits = reference[:, position - 1]
            logits.copy_(source[:, position])
            selector(logits, predecessor, position)
    control = compile_forward(sequential) if compiled_control else None
    if control is not None:
        control()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
        if control is not None:
            control()
    for mode in ('random', 'ties', 'zeros', 'infinity', 'nan', 'random'):
        source.normal_()
        if mode == 'ties':
            source.round_()
        elif mode == 'zeros':
            source.zero_()
        elif mode in ('infinity', 'nan'):
            source[..., :20] = float('inf') if mode == 'infinity' else float('nan')
        selector.projected.normal_()
        predecessor.add_(7)
        before = torch.cuda.memory_stats()['allocation.all.allocated']
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == before
        if compiled_control:
            torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)
            continue
        for position in range(1, width):
            logits = source[:, position].clone()
            unary, ids = logits.topk(16, dim=-1, sorted=False)
            coded = selector.weights["predecessor"][predecessor] * selector.projected[:, position]
            correction = torch.bmm(selector.weights["successor"][ids], coded.unsqueeze(-1)).squeeze(-1)
            logits.fill_(-torch.inf)
            logits.scatter_(1, ids, unary + correction.float())
            reference[:, position - 1].copy_(logits)
        torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)
