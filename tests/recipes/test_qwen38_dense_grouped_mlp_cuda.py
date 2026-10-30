"""Calibrated target MLP grouping with canonical BF16 activation and pooled replay."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize("down_mode", ["auto", "a16", "quantized"])
@torch.no_grad()
def test_checkpoint_grouped_mlp_matches_separate_projections(rows, down_mode, monkeypatch):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

    from ignition.components.linear import prepare_projection
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes import resolve
    from ignition.recipes.qwen38_dense.config import ModelConfig, TEXT_PREFIX
    from ignition.recipes.qwen38_dense.decoder import GroupedMLP, Projection

    snapshot = resolve("qwen3.8-27b-qad").snapshot(Path.home() / ".cache/huggingface/hub")
    manifest = inspect_indexed_checkpoint(snapshot)
    config = ModelConfig.load(snapshot)
    names = tuple(f"{TEXT_PREFIX}.layers.0.mlp.{field}" for field in ("gate_proj", "up_proj", "down_proj"))
    with DirectWeightLoader(manifest, "cuda:0", prefixes=names) as loader:
        separate = tuple(loader.load_linear(name) for name in names)
        combined = loader.load_linear_group(names[:2])
    with torch.device("meta"):
        canonical = Qwen3_5MLP(Qwen3_5TextConfig.from_dict(config.text), config.text["intermediate_size"])
    device, width, intermediate = torch.device("cuda:0"), config.hidden, config.text["intermediate_size"]
    pool = GraphBufferPool(device, chunk_bytes=2 << 20)
    buffers = pool.cursor()
    guard_before = buffers.zeros((1024,), torch.bfloat16)
    source = buffers.zeros((rows, width), torch.bfloat16)
    packed = buffers.zeros((rows, 2 * intermediate), torch.bfloat16)
    hidden = buffers.zeros((rows, intermediate), torch.bfloat16)
    output = buffers.zeros((rows, width), torch.bfloat16)
    gate, up = (buffers.zeros((rows, intermediate), torch.bfloat16) for _ in range(2))
    reference_hidden = buffers.zeros((rows, intermediate), torch.bfloat16)
    reference_output = buffers.zeros((rows, width), torch.bfloat16)
    guard_after = buffers.zeros((1024,), torch.bfloat16)
    arena = ScratchArena(device)
    if down_mode != "auto":
        from dataclasses import asdict
        from b12x.gemm import blockscaled
        from b12x.gemm._tuning import DenseGemmConfig

        original_plan = blockscaled.plan

        def plan(query, **kwargs):
            if query.in_features == intermediate and query.out_features == width:
                if down_mode == "a16":
                    config = blockscaled.BlockscaledConfig(mode="a16", tile_m=16, tile_n=64,
                                                          tile_k=64, split_k=2)
                else:
                    dense = DenseGemmConfig(backend="cutedsl", tile_m=64, tile_n=64, tile_k=512,
                        load_path="tma", split_k_slices=None, large_m_unroll=None,
                        swap_ab=rows != 32, target_occupancy={8: 2, 16: 3, 32: 2}[rows])
                    config = blockscaled.BlockscaledConfig(mode="quantized", dense_config=asdict(dense),
                        fuse_swiglu=query.source_activation == "swiglu_bf16")
                kwargs["override"] = config
            return original_plan(query, **kwargs)

        monkeypatch.setattr(blockscaled, "plan", plan)
    with RecordedPreparationSession(device=device, checkpoint="qad-grouped-target-mlp",
            workload_schema="canonical-swiglu", numerical_policy="auto", autotune_effort="off",
            native_dense_activations=down_mode != "a16") as session:
        session.register_scratch(arena)
        operation = GroupedMLP(canonical, source, combined, separate[2], packed, hidden, output,
                              session=session, name="grouped", workspace=arena)
        for field, weight, x, y in zip(("gate_proj", "up_proj", "down_proj"), separate,
                (source, source, reference_hidden), (gate, up, reference_output), strict=True):
            setattr(canonical, field, Projection(prepare_projection(x, weight, y, session=session,
                name="separate." + field, workspace=arena)))
        session.tune(show_progress=False)

        def run():
            operation(source)
            canonical(source)

        compiled = compile_forward(run)
        source.normal_()
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for factor in (.5, 1., 1.5):
            source.normal_(0, factor)
            source[-1].zero_()
            guard_before.fill_(17)
            guard_after.fill_(-19)
            output.fill_(float("nan"))
            reference_output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            assert torch.all(guard_before == 17) and torch.all(guard_after == -19)
            assert torch.all(output[-1] == 0) and torch.all(reference_output[-1] == 0)
            assert operation.down_source_is_gate_up
            assert torch.all(hidden == 0)
            for actual, expected in ((packed, torch.cat((gate, up), -1)), (output, reference_output)):
                a, b = actual.float(), expected.float()
                assert torch.isfinite(a).all()
                relative = float((a-b).norm() / b.norm().clamp_min(1e-12))
                cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
                assert relative <= .005 and cosine >= .9999, (rows, factor, relative, cosine)
        graph.reset()
