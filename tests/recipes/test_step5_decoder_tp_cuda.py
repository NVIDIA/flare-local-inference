"""Step-5 dense decoder collectives with BF16 transport and FP32 residuals."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires four explicitly assigned GPUs")


def _rank(rank, world, rendezvous, snapshot):
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.rms_norm import PreparedRMSNorm
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.step5.attention import PreparedAttention
    from ignition.recipes.step5.decoder import (
        PreparedDecoderLayer,
        PreparedFeedForward,
        Scratch,
    )
    from ignition.recipes.step5.pool import ContinuationPool
    from ignition.recipes.step5.weights import Step5WeightLoader

    torch.cuda.set_device(rank)
    device, shard = torch.device("cuda", rank), TensorParallelShard(rank, world)
    manifest = inspect_indexed_checkpoint(snapshot)
    with Step5WeightLoader(manifest, device, shard=shard, prefixes=("model.layers.0.",)) as loader:
        weights, config = loader.load_layer(0), loader.config
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    try:
        with PackageStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=49,
                hidden_size=config.hidden, element_bytes=4, ipc=storage.peer_storage)
            graph = torch.cuda.CUDAGraph()
            try:
                arena = ScratchArena(device)
                pools = [ContinuationPool.for_model(config, shard, 2, 1, device, mtp=False) for _ in range(2)]
                with RecordedPreparationSession(device=device, checkpoint="step5-dense-decoder",
                        workload_schema="bf16-transport", numerical_policy="fp32-residual",
                        coordinator=coordinator, compile_workers=4) as session:
                    session.register_scratch(arena)
                    common = {"runtime": runtime, "coordinator": coordinator, "session": session, "workspace": arena}
                    pairs = []
                    for rows in (1, 4, 7, 36, 48, 49):
                        metadata = Metadata.allocate(rows, 1, 1, device)
                        metadata.positions.copy_(torch.arange(rows, device=device))
                        metadata.state_rows.fill_(1)
                        metadata.lengths.fill_(rows)
                        metadata.query_start_loc[1] = rows
                        paths = []
                        for fused, pool in zip((False, True), pools):
                            scratch = Scratch.allocate(config, shard, rows, 1, 128, device)
                            name = f"dense:{rows}:{fused}"
                            attention = PreparedAttention(0, weights, config, shard, scratch.normalized,
                                scratch.attention, metadata, pool, mode="prefill", name=name + ".attention",
                                reduce_output=not fused, **common)
                            mlp = PreparedFeedForward(0, weights, config, scratch, name=name + ".mlp",
                                overlap=False, reduce_output=not fused, **common)
                            layer = PreparedDecoderLayer(weights, attention, mlp, scratch,
                                config.text["rms_norm_eps"], session=session, name=name, runtime=runtime,
                                coordinator=coordinator, next_norm=weights.input_norm if fused else None)
                            norm = None if fused else PreparedRMSNorm(scratch.residual, weights.input_norm,
                                scratch.normalized, eps=config.text["rms_norm_eps"], session=session,
                                name=name + ".following_norm", zero_centered=True)
                            paths.append((scratch, layer, norm))
                        pairs.append(paths)
                    session.tune(show_progress=False)

                    def forward():
                        for paths in pairs:
                            for scratch, layer, norm in paths:
                                update = layer(scratch.residual)
                                if norm is not None:
                                    scratch.residual.add_(update.float())
                                    norm(scratch.residual)
                        arena.tensor.fill_(255)

                    compiled = compile_forward(forward)
                    compiled()
                    torch.cuda.synchronize()
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream())
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                        compiled()
                    session.freeze()
                    failures = []
                    for iteration in range(20):
                        generator = torch.Generator().manual_seed(913 + iteration)
                        for pool in pools:
                            pool.backing.zero_()
                        for paths in pairs:
                            initial = torch.randn(paths[0][0].residual.shape, generator=generator) * .3
                            for scratch, _, _ in paths:
                                scratch.residual.copy_(initial)
                        allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                        graph.replay()
                        torch.cuda.synchronize()
                        try:
                            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                            for paths in pairs:
                                reference, actual = (path[0] for path in paths)
                                for field in ("residual", "normalized"):
                                    expected, result = getattr(reference, field).float(), getattr(actual, field).float()
                                    # BF16 rounding in RMSNorm propagates through the nonlinear MLP.
                                    error = (result - expected).norm(dim=-1) / expected.norm(dim=-1)
                                    assert (error < .006).all(), (
                                        f"rows={result.shape[0]}, iteration={iteration}, field={field}, "
                                        f"maximum relative error={error.max().item()}")
                        except AssertionError as exc:
                            failures.append(str(exc))
                    assert not failures, "\n".join(failures)
            finally:
                graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


def test_dense_decoder_bf16_transport(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned GPUs")
    default = (Path.home() / ".cache/huggingface/hub/models--stepfun-ai--Step-5-Preview-NVFP4"
               / "snapshots/bc246cc2848e17b37f377eda8fd825c5b3018d1f")
    snapshot = Path(os.getenv("FLARE_STEP5_SNAPSHOT", default))
    if not (snapshot / "config.json").exists():
        pytest.skip("requires the Step-5 checkpoint")
    mp.spawn(_rank, args=(4, (tmp_path / "rendezvous").as_uri(), snapshot), nprocs=4, join=True)
