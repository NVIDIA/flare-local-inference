"""Checkpoint affine precision and shared target normalization scratch on replay."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("tokens", [8, 16, 32, 64])
@torch.no_grad()
def test_target_hidden_norm_checkpoint_compiled_replay(tokens):
    from b12x.preparation import PreparationSession
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes import resolve
    from ignition.recipes.qwen38_dense.config import TEXT_PREFIX
    from ignition.recipes.qwen38_dense.decoder import RMSNorm
    from ignition.reference.qwen38_quant import CheckpointTensors

    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = resolve("qwen3.8-27b-qad").snapshot(hub)
    manifest = inspect_indexed_checkpoint(snapshot)
    import json
    config = json.loads((snapshot / "config.json").read_text())
    config = config.get("text_config", config)
    width, layers = config["hidden_size"], config["num_hidden_layers"]
    assert width == 5120 and layers == 64
    device = torch.device("cuda:0")
    pool = GraphBufferPool(device, chunk_bytes=2 << 20)
    buffers = pool.cursor()
    source = buffers.zeros((tokens, width), torch.bfloat16)
    output = buffers.zeros((tokens, width), torch.bfloat16)
    saved = torch.empty((2 * layers, tokens, width), dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device=device).manual_seed(5100 + tokens)
    canonical, operations = [], []
    with CheckpointTensors(manifest) as reader, PreparationSession(
            device=device, autotune=False, compile_workers=4) as session:
        for layer in range(layers):
            for field in ("input_layernorm", "post_attention_layernorm"):
                name = f"layers.{layer}.{field}"
                weight = reader.read(f"{TEXT_PREFIX}.{name}.weight")
                assert weight.dtype == torch.float32
                norm = Qwen3_5RMSNorm(width, eps=config["rms_norm_eps"]).to(device)
                norm.weight.copy_(weight)
                canonical.append(norm)
                operations.append(RMSNorm(norm, source, output, session=session, name=name))

        def run():
            for index, operation in enumerate(operations):
                saved[index].copy_(operation(source))

        compiled = torch.compile(run, fullgraph=True)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for gain in (1, -.5, 2):
            source.copy_(torch.randn(source.shape, device=device, generator=generator) * gain)
            source[-1].zero_()
            original = source.clone()
            output.fill_(float("nan"))
            saved.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(source, original)
            del original
            assert torch.cuda.memory_allocated() == allocated
            for index, norm in enumerate(canonical):
                expected = norm(source)
                lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
                upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
                assert ((saved[index] >= lower) & (saved[index] <= upper)).all()
                assert (saved[index] != expected).float().mean() < .001
                assert torch.count_nonzero(saved[index, -1]) == 0
                del expected, lower, upper
        graph.reset()
