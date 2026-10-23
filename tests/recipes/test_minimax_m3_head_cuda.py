"""Sampled target normalization with shared activation storage and checkpoint gains."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from ignition.recipes import resolve
from safetensors import safe_open
import torch
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLRMSNorm

from ignition.components.parallel import TensorParallelShard
from ignition.components.rms_norm import PreparedIndexedRMSNorm
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import GraphBufferPool
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.minimax_m3.config import ModelConfig, TEXT_PREFIX
from ignition.recipes.minimax_m3.decoder import PreparedDecoder, Scratch

CHECKPOINT = Path(os.environ.get("FLARE_MINIMAX_CHECKPOINT", str(resolve("minimax-m3-1-flash").snapshot(Path(HF_HUB_CACHE)))))
pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not CHECKPOINT.is_dir(),
                              reason="requires an assigned GPU and the MiniMax M3 checkpoint")


@pytest.mark.parametrize("samples", [1, 4, 32])
@torch.no_grad()
def test_sampled_normalization_reads_residual_rows_with_disjoint_output(samples):
    config = ModelConfig.load(CHECKPOINT)
    device = torch.device("cuda:0")
    pool = GraphBufferPool(device, chunk_bytes=2 << 20)
    scratch = Scratch.allocate(config, TensorParallelShard(0, 4), 4096, samples, device,
                               buffers=pool.cursor())
    name = TEXT_PREFIX + ".norm.weight"
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
    with safe_open(CHECKPOINT / index["weight_map"][name], framework="pt", device="cpu") as file:
        weight = file.get_tensor(name).to(device)
    canonical = MiniMaxM3VLRMSNorm(config.hidden, eps=config.text["rms_norm_eps"]).to(device)
    canonical.weight.copy_(weight)
    indices = torch.linspace(0, 4095, samples, device=device).long()
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    decoder.scratch, decoder.metadata = scratch, SimpleNamespace(sample_indices=indices)
    assert scratch.residual.data_ptr() != scratch.hidden.data_ptr()
    with RecordedPreparationSession(device=device, checkpoint="minimax-head",
            workload_schema="minimax-head", numerical_policy="default", compile_workers=1) as session:
        decoder.norm = PreparedIndexedRMSNorm(scratch.residual, indices, weight, scratch.hidden,
            eps=config.text["rms_norm_eps"], zero_centered=True, session=session, name="head_norm")
        session.tune()
        forward = compile_forward(decoder.normalize_head)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        for shift in (0, 113, 511):
            for chunks in pool.chunks.values():
                for chunk in chunks:
                    chunk.view(torch.uint8).fill_(0xFF)
            scratch.residual.normal_(std=.2)
            indices.copy_((torch.linspace(0, 4095, samples, device=device).long() + shift) % 4096)
            if shift == 113 and samples > 1:
                indices[1::2] = indices[0]
            elif shift == 511:
                indices.copy_(indices.flip(0))
            expected = canonical(scratch.residual[indices])
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(scratch.hidden, expected, atol=.002, rtol=.01)
        graph.reset()
