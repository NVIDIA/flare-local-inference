"""Replicated Markov heads retain FP32 logits through compiled graph replay."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.dspark import PreparedMarkov
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 8, 32, 160])
@pytest.mark.parametrize("quantization", [None, "nvfp4"])
@torch.no_grad()
def test_replicated_markov_head_on_a_tensor_parallel_rank(rows, quantization):
    device = torch.device("cuda:0")
    torch.manual_seed(4101)
    head = torch.randn((129280, 256), device=device, dtype=torch.bfloat16) * .02
    if quantization:
        from ignition.components.vocabulary import nvfp4_draft_head

        # Every block has the same maximum; all entries are exactly NVFP4-representable.
        values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=device) / 32
        head.copy_(values[torch.randint(0, 8, head.shape, device=device)])
        head[:, ::16] = 6 / 32
        packed = nvfp4_draft_head(head)
    embedding = torch.randn_like(head) * .1
    weights = SimpleNamespace(
        config=SimpleNamespace(draft=SimpleNamespace(markov_rank=256)),
        shard=SimpleNamespace(world_size=4),
        draft=SimpleNamespace(markov_head=packed if quantization else head, markov_embedding=embedding))
    tokens = torch.arange(rows, device=device, dtype=torch.int64)
    with RecordedPreparationSession(device=device, checkpoint="markov-projection",
            workload_schema="replicated-fp32", numerical_policy=quantization or "checkpoint",
            compile_workers=8) as session:
        operation = PreparedMarkov(weights, tokens, runtime=None, coordinator=None,
                                   session=session, name="markov")
        session.tune(show_progress=False)
        assert operation.logits is operation.local
        assert operation.gather is None
        compiled = torch.compile(operation, fullgraph=True)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for _ in range(3):
            tokens.add_(17)
            expected = torch.nn.functional.linear(embedding[tokens].float(), head.float())
            if quantization:
                expected = expected.bfloat16().float()
                operation.projected.fill_(float("nan"))
            operation.logits.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert operation.logits.dtype == torch.float32
            torch.testing.assert_close(operation.logits, expected, atol=2e-5,
                                       rtol=.008 if quantization else 2e-5)
        graph.reset()
