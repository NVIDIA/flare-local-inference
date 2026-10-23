"""Persistent collective row ownership across changing CUDA graph shapes."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _persistent_rows_rank(rank, world, rendezvous):
    from b12x.comm.pcie.pcie_oneshot import _eager_storage_shards
    from ignition.components.collectives import PreparedAllReduceRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    graphs = []
    try:
        with PackageStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4096,
                hidden_size=6144, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="collective-persistent-rows",
                        workload_schema="bf16-residual", numerical_policy="zero-centered",
                        coordinator=coordinator) as session:
                    calls = []
                    for rows in (48, 1, 49, 4096):
                        for dtype in (torch.bfloat16, torch.float32):
                            for with_addend in (False, True):
                                source = torch.zeros((rows, 6144), device=device, dtype=dtype)
                                input_addend = torch.zeros_like(source) if with_addend else None
                                residual = torch.zeros_like(source, dtype=torch.bfloat16)
                                weight = torch.linspace(-.5, .5, 6144, device=device, dtype=torch.bfloat16)
                                output = torch.empty_like(residual)
                                operation = PreparedAllReduceRMSNorm(source, residual, weight, output,
                                    eps=1e-6, runtime=runtime, session=session, coordinator=coordinator,
                                    name=f"persistent:{rows}:{dtype}:{with_addend}", zero_centered=True,
                                    input_addend=input_addend,
                                    reduction_dtype=torch.bfloat16)
                                calls.append((source, residual, weight, output, operation, input_addend))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream(device=device)
                    for source, residual, _, _, operation, _ in calls:
                        operation(source, residual)
                        torch.cuda.synchronize(device)
                        graph = torch.cuda.CUDAGraph()
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                            operation(source, residual)
                        graphs.append(graph)
                    session.freeze()
                    channel = runtime.for_stream()
                    native = channel._ext._state(channel._ptr)
                    staging_bytes = channel.eager_buffer_bytes * _eager_storage_shards(
                        world, native.transport_policy)
                    for iteration in range(3):
                        for (source, residual, weight, output, _, input_addend), graph in zip(calls, graphs, strict=True):
                            values = torch.arange(source.numel(), device=device).reshape(source.shape)
                            source.copy_(((values + rank + iteration) % 33 - 16).float() / 32)
                            residual.copy_(((values + rank * 3 + iteration) % 17 - 8).float() / 32)
                            if input_addend is not None:
                                input_addend.copy_(((values + rank * 7 + iteration) % 37 - 18).float() / 512)
                            combined = source if input_addend is None else (source.float() + input_addend.float()).to(source.dtype)
                            reduced = combined.to(torch.float32, copy=True)
                            dist.all_reduce(reduced)
                            expected_residual = (residual.float() + reduced.bfloat16().float()).bfloat16()
                            normalized = expected_residual.float() * torch.rsqrt(
                                expected_residual.float().square().mean(-1, keepdim=True) + 1e-6)
                            expected = (normalized * (1 + weight.float())).bfloat16()
                            output.fill_(float("nan"))
                            torch.cuda.synchronize(device)
                            dist.barrier()
                            for pointers in channel._eager_ptrs:
                                channel._ipc.cudaMemset(pointers[rank], 255, staging_bytes)
                            torch.cuda.synchronize(device)
                            dist.barrier()
                            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                            graph.replay()
                            torch.cuda.synchronize(device)
                            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
                            torch.testing.assert_close(output, expected, rtol=.008, atol=.008)
            finally:
                for graph in graphs:
                    graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


def test_persistent_collective_rows_preserve_bf16_residual_rounding(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned GPUs")
    mp.spawn(_persistent_rows_rank, args=(4, (tmp_path / "persistent").as_uri()), nprocs=4, join=True)


def _layer_handoff_rank(rank, world, rendezvous):
    from types import SimpleNamespace

    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLDecoderLayer, MiniMaxM3VLRMSNorm,
    )
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.minimax_m3.decoder import PreparedDecoderLayer

    class ReferenceBranch(torch.nn.Module):
        def __init__(self, output, observed, *, attention):
            super().__init__()
            self.output, self.observed, self.attention = output, observed, attention

        def forward(self, hidden_states, **kwargs):
            self.observed.copy_(hidden_states)
            return (self.output, None) if self.attention else self.output

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    graphs = []
    try:
        with PackageStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=49,
                hidden_size=6144, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="minimax-layer-handoff",
                        workload_schema="two-layer-bf16-residual", numerical_policy="zero-centered",
                        coordinator=coordinator) as session:
                    cases = []
                    for rows in (48, 1, 49):
                        def matrix(dtype=torch.bfloat16):
                            return torch.zeros((rows, 6144), device=device, dtype=dtype)

                        initial = matrix()
                        scratch = SimpleNamespace(residual=matrix(), normalized=matrix(), reduced=matrix(),
                            partial=matrix(), dense=SimpleNamespace(output=matrix()),
                            attention=SimpleNamespace(output=matrix()), experts=matrix(),
                            shared=SimpleNamespace(output=matrix()))
                        gains = tuple(((torch.arange(6144, device=device) + index) % 13 - 6).bfloat16() / 16
                                      for index in range(4))
                        sources = [(matrix(), matrix(), matrix()), (matrix(), matrix(), matrix())]
                        seen = [[matrix(), matrix()] for _ in range(2)]
                        reference_seen = [[matrix(), matrix()] for _ in range(2)]
                        reference_outputs = [[matrix(), matrix()] for _ in range(2)]
                        layers, canonical = [], []
                        for index in range(2):
                            def attention(source, *, index=index, seen=seen, scratch=scratch, sources=sources):
                                seen[index][0].copy_(source)
                                scratch.attention.output.copy_(sources[index][0])
                                return scratch.attention.output

                            def feed_forward(source, *, combine_output=True, index=index, seen=seen, scratch=scratch, sources=sources):
                                seen[index][1].copy_(source)
                                if index == 0:
                                    scratch.experts.copy_(sources[index][1])
                                    scratch.shared.output.copy_(sources[index][2])
                                    if not combine_output:
                                        return scratch.experts
                                    scratch.partial.copy_(scratch.experts.float() + scratch.shared.output.float())
                                    return scratch.partial
                                scratch.dense.output.copy_(sources[index][1])
                                return scratch.dense.output

                            feed_forward.experts = SimpleNamespace(output_source=scratch.experts)
                            weights = SimpleNamespace(input_norm=gains[index * 2],
                                post_attention_norm=gains[index * 2 + 1], experts=object() if index == 0 else None)
                            layers.append(PreparedDecoderLayer(attention, feed_forward, weights, scratch,
                                epsilon=1e-6, runtime=runtime, coordinator=coordinator, session=session,
                                name=f"handoff:{rows}:{index}", first=index == 0, fused=True,
                                next_norm=gains[2] if index == 0 else None))
                            reference = MiniMaxM3VLDecoderLayer.__new__(MiniMaxM3VLDecoderLayer)
                            torch.nn.Module.__init__(reference)
                            reference.input_layernorm = MiniMaxM3VLRMSNorm(6144, eps=1e-6).to(device)
                            reference.post_attention_layernorm = MiniMaxM3VLRMSNorm(6144, eps=1e-6).to(device)
                            reference.input_layernorm.weight = torch.nn.Parameter(gains[index * 2], requires_grad=False)
                            reference.post_attention_layernorm.weight = torch.nn.Parameter(gains[index * 2 + 1], requires_grad=False)
                            reference.self_attn = ReferenceBranch(reference_outputs[index][0],
                                reference_seen[index][0], attention=True)
                            reference.mlp = ReferenceBranch(reference_outputs[index][1],
                                reference_seen[index][1], attention=False)
                            canonical.append(reference)

                        def forward(initial=initial, scratch=scratch, layers=layers):
                            scratch.residual.copy_(initial)
                            update = None
                            for layer in layers:
                                update = layer(layer.normalize(update))
                            if update is not None:
                                scratch.residual.add_(update)

                        cases.append((initial, scratch, sources, seen, reference_seen,
                                      reference_outputs, canonical, forward))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream(device=device)
                    for *_, forward in cases:
                        forward()
                        torch.cuda.synchronize(device)
                        graph = torch.cuda.CUDAGraph()
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                            forward()
                        graphs.append(graph)
                    session.freeze()
                    for iteration in range(3):
                        for case, graph in zip(cases, graphs, strict=True):
                            initial, scratch, sources, seen, reference_seen, reference_outputs, canonical, _ = case
                            columns = torch.arange(initial.numel(), device=device).reshape(initial.shape)
                            active = ((torch.arange(initial.shape[0], device=device) + iteration) % 4 != 0)[:, None]
                            initial.copy_(((columns + rank * 3 + iteration) % 29 - 14).float() / 32)
                            for index, pair in enumerate(sources):
                                for branch, source in enumerate(pair):
                                    source.copy_((((columns + rank * 13 + iteration + index * 7 + branch) % 521 - 260)
                                                  .float() / 1024) * active)
                                for branch in (0, 1):
                                    local = pair[branch]
                                    if index == 0 and branch == 1:
                                        local = (pair[1].float() + pair[2].float()).bfloat16()
                                    reduced = local.to(torch.float32, copy=True)
                                    dist.all_reduce(reduced)
                                    reference_outputs[index][branch].copy_(reduced)
                            expected = initial
                            with torch.no_grad():
                                for layer in canonical:
                                    expected = layer.forward(expected)
                            for pair in seen:
                                for value in pair:
                                    value.fill_(float("nan"))
                            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                            graph.replay()
                            torch.cuda.synchronize(device)
                            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            torch.testing.assert_close(scratch.residual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(scratch.residual[~active[:, 0]], initial[~active[:, 0]], rtol=0, atol=0)
                            for actual_pair, expected_pair in zip(seen, reference_seen, strict=True):
                                for actual, reference in zip(actual_pair, expected_pair, strict=True):
                                    torch.testing.assert_close(actual, reference, rtol=.008, atol=.008)
            finally:
                for graph in graphs:
                    graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


def test_minimax_fused_layer_handoff_matches_canonical(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned GPUs")
    mp.spawn(_layer_handoff_rank, args=(4, (tmp_path / "handoff").as_uri()), nprocs=4, join=True)
