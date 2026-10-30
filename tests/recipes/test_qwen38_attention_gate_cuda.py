"""Strided per-element attention gates through the existing prepared head gate."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("tokens", [8, 16, 32, 64])
@torch.no_grad()
def test_strided_attention_gate_preserves_rounding_and_captured_storage(tokens):
    from b12x.preparation import PreparationSession
    from ignition.components.head_gate import PreparedHeadGate
    from ignition.preparation.scratch import GraphBufferPool

    pool = GraphBufferPool("cuda", chunk_bytes=2 << 20)
    buffers = pool.cursor()
    before = buffers.zeros((1024,), torch.bfloat16)
    qg = buffers.zeros((tokens, 24, 512), torch.bfloat16)
    values = buffers.zeros((tokens, 24, 256), torch.bfloat16)
    after = buffers.zeros((1024,), torch.bfloat16)
    before.fill_(17)
    after.fill_(-19)
    gate = qg[..., 256:]
    flat_gate = gate.view(tokens * 24, 256)
    assert flat_gate.stride() == (512, 1)
    assert flat_gate.data_ptr() == gate.data_ptr()
    with PreparationSession(device="cuda", autotune=False) as session:
        operation = PreparedHeadGate(values.view(tokens * 24, 256, 1), flat_gate,
                                     session=session, name="attention.output_gate")
        compiled = torch.compile(operation, fullgraph=True)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for gain in (0., .5, 2.):
            qg.normal_()
            values.normal_()
            values.mul_(gain)
            values[-1].zero_()
            original = qg.clone()
            expected = values * gate.sigmoid()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(qg, original)
            assert torch.all(before == 17) and torch.all(after == -19)
            assert not values[-1].any()
            torch.testing.assert_close(values, expected, rtol=2**-6, atol=2e-2)
        graph.reset()


def test_head_gate_rejects_aliasing_logits():
    from ignition.components.head_gate import PreparedHeadGate
    values = torch.zeros(2, 256, 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="overlap"):
        PreparedHeadGate(values, values.view(2, 256), session=None, name="invalid")
