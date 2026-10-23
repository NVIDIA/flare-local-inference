"""Complete GQA/index tails and target tap preservation through captured fork/restore."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import DeviceRequestState
from ignition.components.state_transfer import PreparedStateTransfer, StateCommands
from ignition.recipes.minimax_m3.config import ModelConfig
from ignition.recipes.minimax_m3.pool import ContinuationPool

from .test_minimax_m3 import configuration

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("endpoint", [1, 127, 128, 129, 255, 256])
@pytest.mark.parametrize("draft", [False, True])
@torch.no_grad()
def test_captured_snapshot_forks_isolate_gqa_index_and_saved_target_taps(endpoint, draft):
    config = ModelConfig.from_dict(configuration(), target_layers=(0, 1))
    if draft:
        config = replace(config, draft=SimpleNamespace(layers=2, window=4096, kv_heads=4, head_dim=128))
    pool = ContinuationPool.for_model(config, TensorParallelShard(0, 4), 8, 5, "cuda:0")
    commands = StateCommands.allocate("cuda:0", capacity=4)
    tables = torch.full((4, 3), -1, dtype=torch.int32, device="cuda:0")
    consumed = pool.region("consumed_tokens")
    requests = DeviceRequestState.allocate("cuda:0", vocab_size=512, capacity=4,
                                           consumed_tokens=consumed[1:5, 0])
    rows = torch.arange(1, 5, device="cuda:0")[:, None].expand(4, 8).contiguous()
    transfer = PreparedStateTransfer(pool, commands, tables, pages=pool.partial_pages(), page_tokens=128,
                                     state_rows=rows, requests=requests)
    transfer.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer.validate()
        transfer.apply()

    def replay(operations):
        commands.num_ops.fill_(len(operations))
        commands.errors.zero_()
        for index, (source, destination, request) in enumerate(operations):
            commands.source_slots[index] = source
            commands.destination_slots[index] = destination
            commands.operation[index] = 0
            commands.expected_consumed[index] = endpoint
            commands.request_slots[index] = request
        before = torch.cuda.memory_stats()
        graph.replay()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert not commands.errors.count_nonzero()
        for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
            assert before[name] == after[name]

    pool.backing[1].random_(0, 256)
    consumed[1] = endpoint
    tables[0].copy_(torch.tensor([3, 0, 4], device="cuda:0"))
    tables[1].fill_(2)
    tables[2].fill_(1)
    for plane in pool.pages.values():
        plane.random_(0, 256)
    tail = tables[0, endpoint // 128].item() if endpoint % 128 else None
    expected = pool.backing[1].clone()
    tails = {name: plane[tail].clone() for name, plane in pool.partial_pages()} if tail is not None else {}
    for name, value in tails.items():
        region = pool.layout.region(name)
        expected[region.offset:region.offset + region.nbytes].copy_(value)
    replay([(1, 5, 0)])
    assert torch.equal(pool.backing[5], expected)
    pool.backing[1].fill_(255)
    for plane in pool.pages.values():
        plane.zero_()
    replay([(5, 2, 1), (5, 3, 2)])
    for row in (2, 3, 5):
        assert torch.equal(pool.backing[row], expected)
    for name, plane in pool.partial_pages():
        if tail is not None:
            assert torch.equal(plane[1], tails[name])
            assert torch.equal(plane[2], tails[name])
        assert not plane[3:].count_nonzero()
    pool.region("target_hidden")[2].zero_()
    assert torch.equal(pool.backing[5], expected)
    graph.reset()
