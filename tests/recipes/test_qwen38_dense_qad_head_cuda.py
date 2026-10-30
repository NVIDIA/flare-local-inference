"""Captured proposal projections from the QAD checkpoint vocabulary weights."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA GPU")


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
@torch.no_grad()
def test_qad_proposal_head_captured_arithmetic(format):
    from b12x.moe._shared.kernels.w4a16.host import unswizzle_block_scale
    from ignition.components.vocabulary import PreparedVocabularyProjection, mxfp8_draft_head, nvfp4_draft_head
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes import resolve

    snapshot = resolve("qwen3.8-27b-qad").snapshot(Path.home() / ".cache/huggingface/hub")
    with DirectWeightLoader(inspect_indexed_checkpoint(snapshot), "cuda:0", prefixes=("lm_head.",)) as loader:
        target = loader.load_tensor("lm_head.weight")
    assert target.dtype == torch.bfloat16
    saved = target.clone()
    weight = (nvfp4_draft_head if format == "nvfp4" else mxfp8_draft_head)(target)
    vocabulary, hidden = target.shape
    if format == "nvfp4":
        quantization_gain = (6.0 * 448.0) / target.abs().amax().float().clamp_min(1e-8)
        scale_storage = weight.scale_mma.permute(5, 2, 4, 0, 1, 3)
        scales = unswizzle_block_scale(scale_storage, vocabulary, hidden // 16)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    else:
        scales = weight.weight.scale_rows.view(torch.float8_e8m0fnu).reshape(vocabulary, hidden // 32).float()
    torch.manual_seed(41019)
    source = torch.randn((8, hidden), device="cuda", dtype=torch.bfloat16) * .1
    local, output = (torch.empty((8, vocabulary), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    with RecordedPreparationSession(device="cuda:0", checkpoint="qwen-qad-proposal-head",
            workload_schema="stored-quantized-head", numerical_policy=format, compile_workers=4) as session:
        operation = PreparedVocabularyProjection(source, weight, local, output,
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)
        assert operation.projection.plan.prepared.selection.config.mode == "a16"
        compiled = compile_forward(lambda: operation(source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5, 0.):
            source.mul_(factor)
            expected = torch.empty_like(output, dtype=torch.float32)
            baseline = torch.empty_like(output, dtype=torch.float32)
            for start in range(0, vocabulary, 4096):
                stop = min(start + 4096, vocabulary)
                if format == "nvfp4":
                    scale_reference = (target[start:stop].float().reshape(-1, hidden // 16, 16).abs().amax(-1)
                                       / 6.0 * quantization_gain).clamp(max=448).to(torch.float8_e4m3fn).float()
                    assert torch.equal(scales[start:stop], scale_reference)
                    packed = weight.values[start:stop]
                    values = lut[torch.stack((packed & 15, packed >> 4), -1).long()].flatten(-2)
                    restored = values * scales[start:stop].repeat_interleave(16, -1) * weight.global_scale
                else:
                    restored = (weight.weight.values[start:stop].float().reshape(-1, hidden // 32, 32)
                                * scales[start:stop, :, None]).reshape(-1, hidden)
                expected[:, start:stop] = source.float() @ restored.T
                baseline[:, start:stop] = source.float() @ target[start:stop].float().T
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            error = (output.float() - expected).norm() / expected.norm().clamp_min(1e-20)
            assert error < .008
            assert torch.equal(target, saved)
            drift = (expected - baseline).norm() / baseline.norm().clamp_min(1e-20)
            print(f"QAD {format} factor={factor}: stored arithmetic relative L2={error.item():.8f}, BF16 conversion relative L2={drift.item():.8f}")
        graph.reset()
