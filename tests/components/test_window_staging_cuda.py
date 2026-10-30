"""Prepared circular staging preserves mutable metadata and exact FP8 output bytes."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.window_attention import stage_windows
from ignition.components.window_staging import PreparedWindowStaging
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


def _fixture(heads, dim, batch):
    device = torch.device("cuda:0")
    window, capacity = 2048, 2176
    backing = torch.randn((batch + 1, 2, window, heads, dim), device=device).to(torch.float8_e4m3fn)
    history = backing[:, 0], backing[:, 1]
    for h in history:
        h[0, 0].view(torch.uint8).flatten()[:min(256, heads * dim)].copy_(
            torch.arange(min(256, heads * dim), device=device).to(torch.uint8))
    values = tuple(torch.randn((batch * 8, heads, dim), device=device).bfloat16() for _ in range(2))
    special = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 0., 1.0625, 449., -449.], device=device).bfloat16()
    for t in values:
        t[0].flatten()[:8].copy_(special)
    output_storage = torch.full((2, batch, capacity, heads, dim + 4), 17., device=device, dtype=torch.float8_e4m3fn)
    outputs = output_storage[0, ..., :dim], output_storage[1, ..., :dim]
    m = SimpleNamespace(starts=torch.zeros(batch, device=device, dtype=torch.int64),
        state_rows=torch.arange(batch, device=device, dtype=torch.int64),
        query_start_loc=torch.arange(batch + 1, device=device, dtype=torch.int32) * 8,
        lengths=torch.full((batch,), 8, device=device, dtype=torch.int32))
    return history, values, outputs, m, output_storage


@pytest.mark.parametrize('heads,dim,batch', [(8, 128, 1), (8, 128, 4), (16, 64, 2), (1, 128, 2), (2, 64, 1)])
def test_compiled_mutable_staging(heads, dim, batch):
    torch.manual_seed(823)
    history, values, outputs, m, storage = _fixture(heads, dim, batch)
    padding = storage[..., dim:].view(torch.uint8).clone()
    counts = m.lengths.clone()
    live = torch.ones_like(m.lengths)
    observed = tuple(torch.empty_like(t, dtype=torch.bfloat16) for t in outputs)
    references = tuple(torch.empty_like(t) for t in outputs)
    with RecordedPreparationSession(device=values[0].device, checkpoint="paired-window-staging",
            workload_schema="paired-window-staging", numerical_policy="exact", compile_workers=1) as session:
        operation = PreparedWindowStaging(*history, *values, *outputs, m, counts=counts,
            first_position=0, session=session, name="staging")
        session.tune()

        def forward():
            m.lengths.copy_(live * 8)
            operation()
            for out, consumed in zip(outputs, observed):
                consumed.copy_(out)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for mode in range(5):
            m.starts.fill_((0, 1, 2047, 2048, 4096)[mode])
            m.state_rows.copy_(torch.arange(batch - 1, -1, -1, device=live.device) - (batch + 1 if mode == 4 else 0))
            live.copy_(torch.tensor([int(mode != 0 and not (mode == 3 and r % 2)) for r in range(batch)], device=live.device))
            counts.fill_((0, 8, 3, 7, 8)[mode])
            for tensor in values:
                tensor[1:].normal_()
            for tensor in history:
                tensor[:, 2:4].copy_(torch.randn(tensor[:, 2:4].shape, device=live.device).to(tensor.dtype))
            original = tuple(t.clone() for t in (*history, *values))
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for h, v, out, consumed, ref in zip(history, values, outputs, observed, references):
                stage_windows(h, v, ref, m, counts=counts)
                torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)
                torch.testing.assert_close(consumed, ref.bfloat16(), rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(storage[..., dim:].view(torch.uint8), padding, rtol=0, atol=0)
            for t, saved in zip((*history, *values), original):
                torch.testing.assert_close(t.view(torch.uint8), saved.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize('failure', ['misaligned', 'stride', 'overlap', 'external_overlap', 'empty', 'dtype'])
def test_reject_unsupported_staging_views(failure):
    history, values, outputs, m, storage = _fixture(8, 128, 1)
    if failure == 'misaligned':
        outputs = storage[0, ..., 1:129], outputs[1]
    elif failure == 'stride':
        base = torch.empty((1, 2176, 8, 130), device='cuda', dtype=torch.float8_e4m3fn)
        outputs = base[..., :128], outputs[1]
    elif failure == 'overlap':
        outputs = outputs[0], outputs[0]
    elif failure == 'external_overlap':
        offset = outputs[1]
        outputs = offset, torch.from_dlpack(offset.view(torch.uint8)).view(torch.float8_e4m3fn)
        assert outputs[0].untyped_storage().data_ptr() != outputs[1].untyped_storage().data_ptr()
    elif failure == 'empty':
        values = values[0][:0], values[1][:0]
    elif failure == 'dtype':
        values = values[0].float(), values[1]
    with pytest.raises(ValueError):
        PreparedWindowStaging(*history, *values, *outputs, m, session=None, name='invalid')
