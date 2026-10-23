"""Real checkpoint vision features through compiled TP4 encoder graphs."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from ignition.recipes import resolve
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS") != "1",
    reason="requires four assigned GPUs and the MiniMax checkpoint",
)


def _rotary_reference(source, positions):
    inverse = 1.0 / (10000.0 ** (torch.arange(0, 26, 2, dtype=torch.float32) / 26))
    inverse = inverse.to(source.device)
    phase = torch.cat(
        (
            torch.zeros_like(positions[:, :1]) * inverse,
            positions[:, :1] * inverse,
            positions[:, 1:] * inverse,
        ),
        -1,
    )
    phase = torch.cat((phase, phase), -1)[:, None]
    x = source[..., :78].float()
    rotated = torch.cat((-x[..., 39:], x[..., :39]), -1)
    result = (x * phase.cos() + rotated * phase.sin()).bfloat16()
    return torch.cat((result, source[..., 78:]), -1)


def _reference(weights, encoder, pixels, positions):
    f = torch.nn.functional

    def norm(x, w):
        return f.layer_norm(x, (x.shape[-1],), *w, eps=1e-5)

    def reduce(values, destination, operation, bias):
        destination[: values.shape[0]].copy_(values)
        if values.shape[0] < destination.shape[0]:
            destination[values.shape[0] :].zero_()
        return operation(destination)[: values.shape[0]] + bias

    hidden = norm(f.linear(pixels, weights.patch).bfloat16(), weights.norm)
    for index, w in enumerate(weights.layers):
        block = encoder.blocks[index]
        q, k, v = f.linear(norm(hidden, w["norm1"]), *w["qkv"]).chunk(3, -1)
        shape = (hidden.shape[0], -1, 80)
        q, k = (
            _rotary_reference(q.view(shape), positions),
            _rotary_reference(k.view(shape), positions),
        )
        attended = (
            f.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.view(shape).transpose(0, 1)[None],
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        attended = attended.flatten(1)
        output = reduce(
            f.linear(attended, w["output"][0]),
            encoder.scratch.projected,
            block.self_attn.reduce,
            w["output"][1],
        )
        hidden = hidden + output
        output = f.linear(
            f.gelu(f.linear(norm(hidden, w["norm2"]), *w["up"])), w["down"][0]
        )
        output = reduce(
            output, encoder.scratch.projected, block.mlp.reduce, w["down"][1]
        )
        hidden = hidden + output
    values = f.linear(
        f.gelu(f.linear(hidden, *weights.projector[0])), weights.projector[1][0]
    )
    values = reduce(
        values,
        encoder.scratch.projector_output,
        encoder.projector.reduce,
        weights.projector[1][1],
    )
    values = values.reshape(-1, 4 * 6144)
    values = f.linear(
        f.gelu(f.linear(values, *weights.merger[0])), weights.merger[1][0]
    )
    values = reduce(
        values,
        encoder.scratch.merger_output,
        encoder.merger.reduce,
        weights.merger[1][1],
    )
    first, last = weights.shard.partition(6144)
    return values[:, first:last].contiguous()


@torch.no_grad()
def _rank(rank, rendezvous, snapshot):
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import (
        MiniMaxM3VLImageProcessor,
    )
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_dma_all_reduce_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.minimax_m3.vision import (
        VisionWeights,
        VisionScratch,
        VisionInputs,
        PreparedVision,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=30),
    )
    coordinator = CollectivePreparation(dist.group.WORLD, timeout=timedelta(minutes=30))
    weights = VisionWeights.load(
        inspect_indexed_checkpoint(snapshot), device, shard=TensorParallelShard(rank, 4)
    )
    capacity = 2304
    inputs = VisionInputs.allocate(device)
    scratch = VisionScratch.allocate(capacity, weights.shard, device)
    runtime = create_dma_all_reduce_runtime(
        dist.group.WORLD, device, max_tokens=capacity, hidden_size=6144
    )
    graph = None
    try:
        with RecordedPreparationSession(
            device=device,
            checkpoint=str(snapshot),
            workload_schema="minimax-vision-tp4",
            numerical_policy="default",
            coordinator=coordinator,
            compile_workers=4,
        ) as session:
            arena = ScratchArena(device)
            session.register_scratch(arena)
            encoder = PreparedVision(
                weights,
                scratch,
                inputs,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                workspace=arena,
                name="vision",
            )
            from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
                MiniMaxM3VLVisionConfig,
            )
            from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
                MiniMaxM3VLVisionRotaryEmbedding,
            )

            canonical = MiniMaxM3VLVisionRotaryEmbedding(MiniMaxM3VLVisionConfig())
            assert torch.equal(encoder.inverse, canonical.inv_freq.to(device))
            session.tune()
            compiled = compile_forward(encoder)
            encoder.cu.copy_(
                torch.tensor([0, capacity], device=device, dtype=torch.int32)
            )
            compiled()
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                compiled()
            session.freeze()
            failed = False
            for side in (56, 672, 56):
                image = (
                    (torch.arange(side * side * 3, dtype=torch.int64) % 256)
                    .to(torch.uint8)
                    .view(side, side, 3)
                    .permute(2, 0, 1)
                )
                processed = MiniMaxM3VLImageProcessor()(image, return_tensors="pt")
                pixels = processed["pixel_values"].to(device)
                count = pixels.shape[0]
                grid = side // 14
                positions = torch.tensor(
                    [
                        (y + dy, x + dx)
                        for y in range(0, grid, 2)
                        for x in range(0, grid, 2)
                        for dy in range(2)
                        for dx in range(2)
                    ],
                    device=device,
                )
                scratch.patches.zero_()
                scratch.patches[:count].copy_(pixels)
                encoder.positions.zero_()
                encoder.positions[:count].copy_(positions)
                encoder.cu[1] = count
                arena.tensor.fill_(197)
                dist.barrier()
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                after = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                assert after == before
                actual = scratch.output[: count // 4].clone()
                eager = encoder()[: count // 4].clone()
                torch.testing.assert_close(actual, eager, rtol=0, atol=0)
                expected = _reference(weights, encoder, pixels, positions)
                a, b = actual.float(), expected.float()
                error = (a - b).abs()
                metrics = dict(
                    rank=rank,
                    side=side,
                    patches=count,
                    features=count // 4,
                    max_abs=error.max().item(),
                    mean_abs=error.mean().item(),
                    rel_l2=((a - b).norm() / b.norm()).item(),
                    cosine=torch.nn.functional.cosine_similarity(
                        a.flatten(), b.flatten(), dim=0
                    ).item(),
                )
                print("VISION_NUMERICS", metrics, flush=True)
                assert torch.isfinite(a).all()
                failed |= metrics["rel_l2"] >= 0.035 or metrics["cosine"] <= 0.999
                dist.barrier()
            failures = torch.tensor(int(failed))
            dist.all_reduce(failures)
            assert failures.item() == 0, "complete vision numerical tolerance failed"
            graph.reset()
            graph = None
    finally:
        if graph is not None:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


def test_real_checkpoint_vision_tp4(tmp_path):
    snapshot = Path(
        os.environ.get(
            "FLARE_MINIMAX_M3_SNAPSHOT", str(resolve("minimax-m3-1-flash").snapshot(Path(HF_HUB_CACHE)))
        )
    )
    if not snapshot.is_dir():
        pytest.skip("requires MiniMax checkpoint")
    assert torch.cuda.device_count() >= 4
    mp.spawn(
        _rank, args=((tmp_path / "rendezvous").as_uri(), snapshot), nprocs=4, join=True
    )
