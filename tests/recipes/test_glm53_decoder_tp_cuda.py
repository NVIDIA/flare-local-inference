# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM KDA/MLA and dense/MoE layers in compiled TP graphs with native rank replay."""

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four explicitly assigned CUDA devices and FLARE_RUN_TP_CUDA_TESTS=1")


def _rank(rank, world, rendezvous, snapshot, mode, capacity, layer_index, following_dense=False):
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.recurrent import DeltaMetadata, DeltaState
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.decoder import LayerScratch, PreparedGlmLayer
    from ignition.recipes.glm53.indexer import IndexMetadata
    from ignition.recipes.glm53.mla import MLAState
    from ignition.recipes.glm53.model import create_model_tree
    from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = create_model_tree(Path(snapshot))
    manifest = inspect_indexed_checkpoint(snapshot)
    selected = (layer_index, layer_index + 1) if following_dense else (layer_index,)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(rank, world),
                         prefixes=tuple(f"{TEXT_PREFIX}.layers.{index}." for index in selected)) as loader:
        attention, mlp, residual = loader.load_attention(layer_index), loader.load_mlp(layer_index), loader.load_residual(layer_index)
        if following_dense:
            following_weights = (loader.load_attention(layer_index + 1), loader.load_mlp(layer_index + 1),
                                 loader.load_residual(layer_index + 1))
        assert loader._session.torch_copy_bytes == 0
    count = capacity if mode == "prefill" else 1
    capacity = 1 << (capacity - 1).bit_length()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_all_reduce_runtime(group, device, max_tokens=capacity, hidden_size=4096)
    worker = None
    graph = torch.cuda.CUDAGraph()
    try:
        heads = 64 // world
        module = model.model.language_model.layers[layer_index]
        is_mla = module.block_type == "indexed_attention"
        if is_mla:
            state = MLAState.allocate(128, 6, device)
            metadata = IndexMetadata.allocate(capacity, 1 if mode == "prefill" else capacity, 64, device)
            persistent = {"latent_cache": state.cache, "index_cache": state.index.cache, "tail": state.index.tail}
        else:
            state = DeltaState.allocate(6, device, heads=heads, channels=3 * heads * 128)
            metadata = DeltaMetadata.allocate(2, device)
            persistent = {"state": state.recurrent, "history": state.convolution}
        if following_dense:
            following_state = DeltaState.allocate(6, device, heads=heads, channels=3 * heads * 128)
            persistent.update(following_state=following_state.recurrent, following_history=following_state.convolution)
        scratch = LayerScratch.allocate(capacity, world, device, sparse=layer_index >= 3,
            block_type=module.block_type, mode=mode, blocks=64)
        source = torch.zeros((capacity, 4, 4096), dtype=torch.bfloat16, device=device)
        output = torch.empty_like(source)
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint="GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
                                        workload_schema="glm-layer-tp-v1", numerical_policy="default",
                                        coordinator=coordinator) as session:
            session.register_scratch(arena)
            layer = PreparedGlmLayer(module, attention, mlp, residual,
                source, output, scratch, state, metadata, mode=mode, runtime=runtime, coordinator=coordinator,
                session=session, name=f"layer{layer_index}", workspace=arena, defer_post=following_dense)
            following = None
            if following_dense:
                following = PreparedGlmLayer(model.model.language_model.layers[layer_index + 1], *following_weights,
                    layer.result, output, scratch, following_state, metadata, mode=mode, runtime=runtime,
                    coordinator=coordinator, session=session, name=f"layer{layer_index + 1}", workspace=arena,
                    carry=scratch.mlp_hc)
            session.tune()

            def forward():
                value = layer(source)
                if following is not None:
                    following(value)
                arena.tensor.fill_(255)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                compiled()
            torch.cuda.synchronize(device)
            session.freeze()
            control = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                       "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                       "completion_records": torch.zeros((2, 4, 24), dtype=torch.int64, device=device)}
            bundle = PreparedExecution(device=rank, graphs={mode: graph}, buffers=control,
                signatures={mode: GraphSignature(mode, capacity if mode == "decode" else 1,
                                                capacity if mode == "prefill" else 1)},
                resources=(compiled, layer, following, runtime, session, arena, stream, metadata,
                           source, output, persistent, scratch))
            worker = bundle.start_executor()
            generator = torch.Generator().manual_seed(530)
            source[:count].copy_((torch.randn((count, 4, 4096), generator=generator) * .1).to(torch.bfloat16))
            if is_mla:
                metadata.block_table[0].copy_(torch.arange(64, dtype=torch.int32, device=device).flip(0))
                metadata.state_slots[0] = 1
                metadata.query_start_loc[1:].fill_(count)
                metadata.positions[:count].copy_(torch.arange(count, dtype=torch.int64, device=device))
                metadata.token_slots[:count].copy_(metadata.block_table[0, metadata.positions[:count] // 256].long()
                                                  * 256 + metadata.positions[:count] % 256)
            else:
                metadata.query_start_loc.copy_(torch.tensor([0, count, count], dtype=torch.int32))
                metadata.convolution_slots.copy_(torch.tensor([1, -1], dtype=torch.int64))
                metadata.state_indices[0, 0] = 1
                metadata.initial_state_indices[0] = 1
                metadata.final_state_indices[0] = 1
                metadata.checkpoint_state_indices[0] = 2 if count >= 16 else 0
                metadata.checkpoint_offsets[0] = 16 if count >= 16 else 0
                metadata.num_seqs.fill_(1)
                metadata.num_tokens.fill_(count)
            stages = {
                "attention_norm": scratch.attention_hc.normalized,
                "attention": layer.attention_reduce.output,
                "mlp_input": scratch.mlp_hc.normalized,
            }
            if layer_index >= 3:
                stages.update(router_logits=scratch.mlp.router_logits, ids=scratch.mlp.router.ids,
                              weights=scratch.mlp.router.weights, shared=scratch.mlp.shared.output)
            for repetitions in (1, 4):
                initial = {key: value.clone() for key, value in persistent.items()}
                for _ in range(repetitions):
                    forward()
                expected_output = output.clone()
                expected_state = {key: value.clone() for key, value in persistent.items()}
                expected_stages = {name: value.clone() for name, value in stages.items()}
                for key, value in persistent.items():
                    value.copy_(initial[key])
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                dist.barrier(group=group)
                for _ in range(repetitions):
                    ticket = worker.submit(0, [])
                    assert ticket is not None
                    deadline = time.monotonic() + 120
                    completion = worker.poll()
                    while completion is None and time.monotonic() < deadline:
                        time.sleep(.001)
                        completion = worker.poll()
                    assert completion is not None and completion.fence == ticket
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for key, value in persistent.items():
                    assert bool(torch.isfinite(value).all())
                    torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
                if not is_mla:
                    assert not state.recurrent[0].count_nonzero()
                gathered = [None] * world
                dist.all_gather_object(gathered, output.cpu(), group=group)
                for other in gathered:
                    torch.testing.assert_close(other, output.cpu(), rtol=0, atol=0)
                if layer_index >= 3:
                    dist.all_gather_object(gathered, scratch.mlp.router.ids.cpu(), group=group)
                    for other in gathered:
                        torch.testing.assert_close(other, scratch.mlp.router.ids.cpu(), rtol=0, atol=0)
                assert bool(torch.isfinite(output).all())
                for name, value in stages.items():
                    torch.testing.assert_close(value, expected_stages[name], rtol=0, atol=0,
                        msg=lambda message, name=name: f"{name}: {message}")
                torch.testing.assert_close(output, expected_output,
                    rtol=.01 if layer_index >= 3 else 0,
                    atol=.001 if layer_index >= 3 else 0)
            worker.close()
            worker = None
            graph.reset()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("layer_index", [0, 3, 4])
@pytest.mark.parametrize("world,mode,capacity", [(2, "decode", 2), (4, "decode", 2),
                                               (2, "prefill", 65), (4, "prefill", 4096)])
def test_actual_layer_native_tp_graph(world, mode, capacity, layer_index, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() == 4
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), str(snapshot), mode, capacity, layer_index),
             nprocs=world, join=True)


@pytest.mark.parametrize("mode,capacity", [("decode", 2), ("prefill", 65), ("prefill", 4096)])
def test_adjacent_dense_layers_fuse_mhc_native_tp_graph(mode, capacity, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    assert torch.cuda.device_count() >= 2
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    mp.spawn(_rank, args=(2, (tmp_path / "rendezvous").as_uri(), str(snapshot), mode, capacity, 0, True),
             nprocs=2, join=True)
