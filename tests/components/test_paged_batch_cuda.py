"""Queued single-rank decode consumes GPU-owned pending tokens."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState, RequestAdmission, RUNNING, ERROR
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows,capacity,length_cases", [
    (2, 8, ((6, 2), (2, 6))),
    (2, 7, ((5, 2), (2, 5))),
    (4, 5, ((2, 1, 1, 1), (1, 1, 2, 1))),
    (4, 13, ((6, 3, 2, 1), (1, 0, 6, 4))),
])
@torch.no_grad()
def test_packed_prefixes_keep_individual_lengths_and_total_rows_independent(rows, capacity, length_cases):
    device = torch.device("cuda", 0)
    metadata = Metadata.allocate(capacity, rows, 2, device, samples=rows * 6, max_query_tokens=6)
    lengths = torch.tensor(length_cases[0], dtype=torch.int32, device=device)
    starts = torch.arange(rows, dtype=torch.int64, device=device) * 128 + 126
    states = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    pages = torch.arange(rows * 2, dtype=torch.int32, device=device).reshape(rows, 2)
    tokens = torch.arange(rows * 6, dtype=torch.int64, device=device).reshape(rows, 6)
    live = torch.tensor([rows], dtype=torch.int32, device=device)
    compiled = compile_forward(lambda: metadata.set(starts, lengths, states, pages, tokens, live))
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for values in length_cases:
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device=device))
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected_tokens = [row * 6 + column for row, count in enumerate(values) for column in range(count)]
            expected_positions = [126 + row * 128 + column
                                  for row, count in enumerate(values) for column in range(count)]
            padding = capacity - sum(values)
            ends = [sum(values[:row + 1]) for row in range(rows)]
            sample_indices = [min(sum(values[:row]) + column, capacity - 1)
                              for row in range(rows) for column in range(6)]
            assert metadata.max_query_tokens == 6
            assert metadata.token_ids.tolist() == expected_tokens + [0] * padding
            assert metadata.positions.tolist() == expected_positions + [-1] * padding
            assert metadata.query_start_loc.tolist() == [0, *ends]
            assert metadata.sample_indices.tolist() == sample_indices
            assert metadata.state_rows.tolist() == [row + 1 if count else 0 for row, count in enumerate(values)]
            assert metadata.live_counts.tolist() == [sum(values), rows]
    finally:
        graph.reset()


@torch.no_grad()
def test_queued_decode_uses_pending_device_token_and_rejects_invalid_state():
    device = torch.device("cuda", 0)
    requests = DeviceRequestState.allocate(device, vocab_size=256, capacity=2)
    inputs = BatchInputs.allocate(2, 2, device, capacity=2)
    metadata = Metadata.allocate(2, 2, 2, device)
    pool = SimpleNamespace(memory_capacity=torch.tensor([3, 4], device=device))
    batch = PreparedBatch(inputs, requests, metadata, mode="decode",
                          shard=TensorParallelShard(0, 1), pool=pool)

    def step():
        batch.begin(batch.validate())
        batch.finish()
        # A deterministic device sampler supplies the next token without a host round trip.
        requests.pending_token.copy_(torch.where(requests.status == RUNNING,
                                                 metadata.token_ids + 1, requests.pending_token))

    compiled = compile_forward(step)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    requests.admit(0, RequestAdmission(generation=1, seed=0), [7], consumed_tokens=1)
    requests.pending_token[0] = 11
    inputs.request_state_rows[0, 0] = 1
    inputs.row_slot_ids[0] = 0
    inputs.row_generations[0] = 1
    inputs.live_rows.fill_(1)
    inputs.page_tables[0].copy_(torch.tensor([0, 1], device=device))
    inputs.token_ids.fill_(11)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    assert requests.status[0].item() == RUNNING
    assert requests.consumed_tokens[0].item() == 5
    assert requests.pending_token[0].item() == 15
    assert metadata.token_ids.tolist() == [14, 0]
    assert requests.consumed_tokens[1].item() == 0
    requests.pending_token[0] = 256
    graph.replay()
    torch.cuda.synchronize()
    assert requests.status[0].item() == ERROR
    assert requests.error_code[0].item() == 32
    assert requests.consumed_tokens[0].item() == 5
    graph.reset()
