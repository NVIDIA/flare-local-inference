"""SM103 index selection reads the operands of each compiled replay, not pointers bound at construction."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA_VISIBLE_DEVICES",
)

HEADS, PAGES_PER_ROW, TOPK = 32, 64, 512
WIDTH = PAGES_PER_ROW * 64
# The kernel sums the weighted head scores in a different FP32 order than the reference, so a logit may differ
# by about one rounding per head; indices whose reference logits lie this close to the TOPK-th value are
# interchangeable, and exact ties at that value occur with these inputs.
TIE_RTOL = HEADS * 2.0 ** -24


@pytest.fixture(scope="module")
def device():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 index selection requires a GB300/B300")
    return torch.device("cuda", 0)


def assert_selection(output, logits, lengths, context):
    """Each row selects min(TOPK, extent) distinct indices inside its extent that form a top-k of the reference."""
    for row, count in enumerate(lengths.tolist()):
        selected = output[row].tolist()
        chosen = [index for index in selected if index >= 0]
        k = min(TOPK, count)
        assert len(chosen) == len(set(chosen)) == k, f"{context}, row {row}"
        assert sum(index < 0 for index in selected) == TOPK - k, f"{context}, row {row}"
        if not k:
            continue
        assert max(chosen) < count, f"{context}, row {row}: selected an index past the extent"
        values = logits[row, :count]
        kth = float(torch.topk(values, k).values[-1])
        slack = TIE_RTOL * abs(kth)
        inside = torch.zeros(count, dtype=torch.bool, device=values.device)
        inside[chosen] = True
        assert float(values[inside].min()) >= kth - slack, f"{context}, row {row}: selected a logit below the top-k"
        if k < count:
            assert float(values[~inside].max()) <= kth + slack, f"{context}, row {row}: skipped a top-k logit"


def requests(generator, device, mode, rows, pages, step):
    """Inputs of one replay: each step is a different request with its own extent and page table."""
    if mode == "decode":
        # One request per row; extents vary from a single record to the table's full width.
        lengths = torch.randint(1, WIDTH + 1, (rows,), generator=generator).int()
        lengths[step % rows] = 1 if step % 2 else WIDTH
        table = torch.stack([torch.randperm(pages, generator=generator)[:PAGES_PER_ROW] for _ in range(rows)]).int()
    else:
        # One request on every row: causal extents from a start position, tail rows inactive.
        start = int(torch.randint(0, WIDTH // 2, (1,), generator=generator)) if step % 2 else 0
        active = max(1, rows - 3 * (step % 3))
        lengths = torch.zeros(rows, dtype=torch.int32)
        lengths[:active] = torch.arange(start + 1, start + active + 1).clamp(max=WIDTH).int()
        table = torch.randperm(pages, generator=generator)[:PAGES_PER_ROW].view(1, -1).int()
    return lengths.to(device), table.to(device)


@pytest.mark.parametrize("mode,rows", [("decode", 1), ("decode", 8), ("prefill", 128), ("prefill", 512)])
@torch.no_grad()
def test_compiled_replays_select_from_current_lengths_and_page_tables(device, mode, rows):
    from b12x.attention.dsa_indexer.reference import (
        pack_index_k_cache_reference,
        paged_decode_logits_reference,
    )

    from ignition.components.indexer_sm103 import Sm103SparseIndexer
    from ignition.preparation.compilation import compile_forward

    generator = torch.Generator().manual_seed(53 + rows)
    device_generator = torch.Generator(device).manual_seed(53 + rows)
    pages = rows * PAGES_PER_ROW + 5 if mode == "decode" else 2 * PAGES_PER_ROW + 5
    cache = pack_index_k_cache_reference(torch.randn(pages * 64, 128, device=device, generator=device_generator),
                                         page_size=64).view(pages, 64 * 132)
    query = (torch.randn(rows, HEADS, 128, device=device, generator=device_generator) * .5).to(torch.float8_e4m3fn)
    # GLM produces the selector's FP32 weights in-graph from a strided column view of a BF16 projection.
    weights = torch.zeros(rows, HEADS, device=device)
    projected = torch.zeros((rows, 160), dtype=torch.bfloat16, device=device)
    output = torch.full((rows, TOPK), -7, dtype=torch.int32, device=device)
    # Persistent scratch the selector is bound to; its contents arrive from the inputs by in-graph copies.
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    table = torch.zeros((rows if mode == "decode" else 1, PAGES_PER_ROW), dtype=torch.int32, device=device)
    next_lengths = torch.zeros_like(lengths)
    next_table = torch.zeros_like(table)
    selector = Sm103SparseIndexer(query, weights, cache, table, lengths, None, output, caps=None,
                                  session=None, name="select", workspace=None)

    def forward():
        weights.copy_(projected[:, 128:])
        lengths.copy_(next_lengths)
        table.copy_(next_table)
        selector()

    def reference():
        # The reference reads the weights the graph wrote, which must equal the converted source columns.
        assert torch.equal(weights, projected[:, 128:].float())
        batch = torch.arange(rows, device=device) if mode == "decode" else torch.zeros(rows, device=device).long()
        return paged_decode_logits_reference(q_fp8=query, weights=weights, index_k_cache=cache,
            real_page_table=next_table, query_row_to_batch=batch, seqlens_per_query=next_lengths, page_size=64)

    def inputs(step):
        projected.copy_(torch.rand((rows, 160), generator=generator).bfloat16().to(device))
        new_lengths, new_table = requests(generator, device, mode, rows, pages, step)
        next_lengths.copy_(new_lengths)
        next_table.copy_(new_table)

    inputs(0)
    compiled = compile_forward(forward)
    compiled()
    # The operator body itself allocates no device memory once warm.
    selector()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    selector()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    for counter in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated",
                    "segment.all.freed"):
        assert before[counter] == after[counter], counter
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    torch.cuda.synchronize()
    assert_selection(output, reference(), next_lengths, "compiled warmup")
    for step in range(1, 9):
        inputs(step)
        output.fill_(-7)
        # Logits cells past a row's extent must never decide a selection.
        selector.logits.fill_(1e30)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert_selection(output, reference(), next_lengths, f"{mode} replay {step}")
    graph.reset()
