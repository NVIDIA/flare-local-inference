"""Indexed normalization preserves the canonical reduction and mutable row selection."""

import os

import pytest
import torch
from b12x.preparation import PreparationSession

from ignition.components.rms_norm import PreparedIndexedRMSNorm, PreparedRMSNorm

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows,samples", [(1, 1), (47, 8), (48, 48), (49, 8), (4096, 8)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_indexed_norm_matches_canonical_mutable_replay(rows, samples, dtype):
    device = torch.device("cuda:0")
    source = torch.randn(rows, 6152, device=device, dtype=torch.bfloat16)[:, 8:]
    indices = torch.arange(samples, device=device, dtype=dtype) % rows
    gain = torch.linspace(-.5, .5, 6144, device=device).bfloat16()
    output = torch.empty(samples, 6144, device=device, dtype=torch.bfloat16)
    gathered, reference = torch.empty_like(output), torch.empty_like(output)
    with PreparationSession(device=device, autotune=False, compile_workers=4) as session:
        operation = PreparedIndexedRMSNorm(source, indices, gain, output, eps=1e-6,
            zero_centered=True, session=session, name="indexed")
        canonical = PreparedRMSNorm(gathered, gain, reference, eps=1e-6,
            zero_centered=True, session=session, name="canonical")
        forward = torch.compile(lambda: operation(source, indices), fullgraph=True)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for shift in (0, 7, 13):
            source.normal_(std=.2)
            indices.copy_((torch.arange(samples, device=device, dtype=dtype).flip(0) + shift) % rows)
            if samples > 1:
                indices[1::2] = indices[0]
            gathered.copy_(source[indices.long()])
            canonical(gathered)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
        graph.reset()


@torch.no_grad()
def test_indexed_norm_drops_private_operands_before_capture(tmp_path):
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    source = torch.full((4096, 6144), torch.nan, device=device, dtype=torch.bfloat16)
    indices = torch.full((8,), 1 << 20, device=device, dtype=torch.int32)
    weight = torch.ones(6144, device=device, dtype=torch.bfloat16)
    output = torch.full((8, 6144), torch.nan, device=device, dtype=torch.bfloat16)
    with RecordedPreparationSession(device=device, checkpoint=str(tmp_path),
            workload_schema="indexed-rms-memory", numerical_policy="default", compile_workers=4) as session:
        operation = PreparedIndexedRMSNorm(source, indices, weight, output,
            eps=1e-6, session=session, name="indexed")
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        session.tune()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(device) - before < 1 << 20
        assert torch.isnan(source).all() and torch.isnan(output).all()
        assert (indices == 1 << 20).all()
        source.fill_(2)
        indices.zero_()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            operation(source, indices)
        allocated = torch.cuda.memory_allocated(device)
        for value in (3, -2):
            source.fill_(value)
            output.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated
            torch.testing.assert_close(output, torch.full_like(output, 1 if value > 0 else -1), atol=0, rtol=0)
        graph.reset()
