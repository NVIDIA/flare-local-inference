"""Grouped projections preserve caller-owned part destinations through compiled replay."""

import os

import pytest
import torch

from ignition.components.linear import prepare_projection
from ignition.components.weights import GroupedLinearWeights


@pytest.mark.parametrize("invalid", ["count", "rows", "width", "dtype", "device", "type"])
def test_explicit_projection_outputs_reject_mismatched_parts(invalid):
    source = torch.empty((3, 128), dtype=torch.bfloat16)
    weights = GroupedLinearWeights((torch.empty((128, 128)), torch.empty((64, 128))), (128, 64))
    outputs = (torch.empty((3, 128), dtype=torch.bfloat16), torch.empty((3, 64), dtype=torch.bfloat16))
    changes = {
        "count": outputs[:1],
        "rows": (outputs[0], torch.empty((2, 64), dtype=torch.bfloat16)),
        "width": (outputs[0], torch.empty((3, 65), dtype=torch.bfloat16)),
        "dtype": (outputs[0], outputs[1].float()),
        "device": (outputs[0], torch.empty((3, 64), dtype=torch.bfloat16, device="meta")),
        "type": (outputs[0], None),
    }
    with pytest.raises((TypeError, ValueError), match="grouped projection outputs"):
        prepare_projection(source, weights, changes[invalid], session=None, name="invalid", workspace=None)


def test_separate_outputs_require_grouped_weights():
    source, weight, output = (torch.empty((2, 2), dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(TypeError, match="separate projection outputs require grouped weights"):
        prepare_projection(source, weight, (output,), session=None, name="invalid", workspace=None)


@pytest.mark.parametrize("alias", ["parts", "row_wrap", "source", "other_weight"])
def test_grouped_outputs_reject_overlapping_storage_before_preparation(alias):
    source = torch.zeros((3, 128), dtype=torch.bfloat16)
    weights = GroupedLinearWeights((torch.ones((128, 128), dtype=torch.bfloat16),
                                   torch.ones((64, 128), dtype=torch.bfloat16)), (128, 64))
    backing = torch.full((3 * 192 + 64,), -123, dtype=torch.bfloat16)[:3 * 192].view(3, 192)
    outputs = (backing[:, :128], backing[:, 128:])
    if alias == "parts":
        outputs = (outputs[0], backing[:, 64:128])
    elif alias == "row_wrap":
        # The second destination reaches the following row's first destination.
        outputs = (outputs[0], backing.as_strided((3, 64), (192, 1), 160))
    elif alias == "source":
        outputs = (source, outputs[1])
    else:
        outputs = (weights.parts[1][:3], outputs[1])
    before = [value.clone() for value in (source, backing, *weights.parts)]
    with pytest.raises(ValueError, match="must not overlap"):
        prepare_projection(source, weights, outputs, session=None, name="alias", workspace=None)
    for actual, expected in zip((source, backing, *weights.parts), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an explicitly assigned GPU")
@pytest.mark.parametrize("rows", [1, 65])
@torch.no_grad()
def test_mixed_projection_parts_compile_and_replay_with_guards(rows):
    from b12x.gemm import blockscaled

    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    device = torch.device("cuda:0")
    source = torch.ones((rows, 128), dtype=torch.bfloat16, device=device)
    quantized = blockscaled.pack_weight(
        torch.ones((128, 128), dtype=torch.float8_e4m3fn, device=device),
        torch.full((128, 4), 127, dtype=torch.uint8, device=device), recipe="mxfp8")
    weights = GroupedLinearWeights((quantized, torch.full((64, 128), 0.5, dtype=torch.bfloat16, device=device)),
                                  (128, 64))
    guard = torch.full((rows * 192 + 96,), -123, dtype=torch.bfloat16, device=device)
    outputs = (guard[32:32 + rows * 128].view(rows, 128), guard[64 + rows * 128:-32].view(rows, 64))
    concatenated = torch.empty((rows, 192), dtype=torch.bfloat16, device=device)
    staging = torch.empty((rows, 128), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device, overlap=False)
    with RecordedPreparationSession(device=device, checkpoint="mixed-projection", workload_schema="parts-v1",
                                    numerical_policy="default", compile_workers=2) as session:
        session.register_scratch(arena)
        parts = prepare_projection(source, weights, outputs, session=session, name="parts", workspace=arena)
        packed = prepare_projection(source, weights, concatenated, session=session, name="packed", workspace=arena,
                                    staging=staging)
        row_parts = prepare_projection(source, weights, concatenated.split(weights.widths, -1), session=session,
                                       name="row_parts", workspace=arena, staging=staging)
        session.tune()
        assert parts(source) is outputs
        assert packed(source) is concatenated
        forward = compile_forward(lambda: (parts(source), packed(source), row_parts(source)))
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        try:
            for fill in (1, 2, -1, 0):
                source.fill_(fill)
                guard.fill_(-123)
                arena.tensor.fill_(255)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for counter in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                    assert before[counter] == after[counter]
                for output, expected, scale in zip(outputs, concatenated.split(weights.widths, -1), (128, 64), strict=True):
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    assert bool(output.eq(fill * scale).all())
                for edge in (guard[:32], guard[32 + rows * 128:64 + rows * 128], guard[-32:]):
                    assert bool(edge.eq(-123).all())
        finally:
            graph.reset()
