"""NVFP4 vocabulary shards preserve target weights and replay with bounded storage."""

import os

import pytest
import torch

from ignition.components.vocabulary import PreparedVocabularyProjection, nvfp4_draft_head
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 5, 20, 40, 160])
@pytest.mark.parametrize("hidden", [5120, 6144])
@torch.no_grad()
def test_nvfp4_vocabulary_shard_keeps_target_weights(rows, hidden):
    device = torch.device("cuda:0")
    torch.manual_seed(1017)
    # A TP4 DeepSeek vocabulary shard has 64-aligned rows, requiring scale padding.
    target = torch.randn((32320, hidden), device=device, dtype=torch.bfloat16) * .02
    original = target.clone()
    weight = nvfp4_draft_head(target)
    assert torch.equal(target, original)
    source = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16) * .1
    local = torch.empty((rows, 32320), device=device, dtype=torch.bfloat16)
    logits = torch.empty_like(local)
    with RecordedPreparationSession(device=device, checkpoint="draft-vocabulary-shard",
            workload_schema="nvfp4-a16", numerical_policy="nvfp4", compile_workers=4) as session:
        operation = PreparedVocabularyProjection(source, weight, local, logits,
            runtime=None, coordinator=None, session=session, name="draft-head")
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: operation(source))
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1., -.5):
            source.mul_(factor)
            expected = torch.nn.functional.linear(source.float(), original.float())
            operation(source)
            eager = logits.clone()
            logits.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(logits, eager, atol=0, rtol=0)
            assert torch.isfinite(logits).all()
            assert (logits.float() - expected).norm() / expected.norm() < .15
            assert torch.equal(target, original)
        graph.reset()
