"""MiniMax vision projection, partial axial rotary, and attention graph contracts."""

import os

import pytest
import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.minimax_m3.vision import axial_rope

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU"
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_vision_patch_projection_conv3d_graph(dtype):
    device = torch.device("cuda", 0)
    torch.manual_seed(703)
    source = torch.randn(256, 1176, device=device, dtype=dtype)
    weight = (torch.randn(1280, 1176, device=device) * 0.03).to(dtype)
    output = torch.empty(256, 1280, device=device, dtype=torch.bfloat16)
    with RecordedPreparationSession(
        device=device,
        checkpoint="vision-patch",
        workload_schema="vision-patch",
        numerical_policy="default",
    ) as session:
        projection = UnquantizedProjection(
            source, weight, output, session=session, name="patch"
        )
        session.tune()

        def forward():
            return projection(source)

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for gain in (1.0, 0.5):
            source.mul_(gain)
            graph.replay()
            reference = (
                torch.nn.functional.conv3d(
                    source.view(256, 3, 2, 14, 14), weight.view(1280, 3, 2, 14, 14)
                )
                .view(256, 1280)
                .bfloat16()
            )
            torch.testing.assert_close(output, reference, atol=0.03125, rtol=0.016)
        graph.reset()


@torch.no_grad()
def test_vision_axial_rope_attention_compiled_graph():
    device = torch.device("cuda", 0)
    torch.manual_seed(704)
    rows, heads, dim = 256, 4, 80
    packed = (torch.randn(rows, 3 * heads * dim, device=device) * 0.3).bfloat16()
    positions = torch.stack(
        (
            torch.arange(rows, device=device) // 16,
            torch.arange(rows, device=device) % 16,
        ),
        -1,
    )
    inverse = 10000 ** (-torch.arange(13, device=device, dtype=torch.float32) * 2 / 26)
    q, k, v, out = [
        torch.empty(rows, heads, dim, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    ]
    cu = torch.tensor([0, 256], device=device, dtype=torch.int32)
    with RecordedPreparationSession(
        device=device,
        checkpoint="vision-axial",
        workload_schema="vision-axial",
        numerical_policy="default",
    ) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedVarlenAttention(
            q, k, v, cu, out, session=session, workspace=arena, name="attention"
        )
        session.tune()

        def forward():
            qs, ks, vs = packed.chunk(3, -1)
            q.copy_(axial_rope(qs.view_as(q), positions, inverse))
            k.copy_(axial_rope(ks.view_as(k), positions, inverse))
            v.copy_(vs.view_as(v))
            return attention()

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for length in (256, 28, 144):
            cu[1] = length
            positions[:, 1].add_(1)
            arena.tensor.fill_(231)
            graph.replay()
            torch.cuda.synchronize()
            qs, ks, vs = packed.chunk(3, -1)
            for source, actual in ((qs.view_as(q), q), (ks.view_as(k), k)):
                angles = torch.cat(
                    (
                        torch.zeros(rows, 13, device=device),
                        positions[:, 0, None] * inverse,
                        positions[:, 1, None] * inverse,
                    ),
                    -1,
                )
                c, s = angles.cos()[:, None], angles.sin()[:, None]
                a, b = source[..., :39].float(), source[..., 39:78].float()
                expected = torch.cat(
                    (a * c - b * s, b * c + a * s, source[..., 78:].float()), -1
                ).bfloat16()
                torch.testing.assert_close(
                    actual, expected, atol=0.00390625, rtol=0.008
                )
                assert torch.equal(actual[..., 78:], source[..., 78:])
            reference = (
                torch.nn.functional.scaled_dot_product_attention(
                    q[:length].transpose(0, 1)[None],
                    k[:length].transpose(0, 1)[None],
                    v[:length].transpose(0, 1)[None],
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            torch.testing.assert_close(
                out[:length], reference, atol=0.00390625, rtol=0.016
            )
        graph.reset()


@torch.no_grad()
def test_axial_canonical_frequency_table_and_large_coordinates():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLVisionConfig,
    )
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLVisionRotaryEmbedding,
    )

    device = torch.device("cuda", 0)
    canonical = MiniMaxM3VLVisionRotaryEmbedding(MiniMaxM3VLVisionConfig())
    inverse = canonical.inv_freq.to(device)
    expected_table = (
        1.0 / (10000.0 ** (torch.arange(0, 26, 2, dtype=torch.float32) / 26))
    ).to(device)
    assert torch.equal(inverse, expected_table)
    torch.manual_seed(705)
    source = torch.randn(4, 4, 80, device=device, dtype=torch.bfloat16)
    positions = torch.tensor(
        [[0, 0], [47, 47], [1151, 1151], [1048575, 1048575]], device=device
    )
    output = torch.empty_like(source)
    compiled = compile_forward(
        lambda: output.copy_(axial_rope(source, positions, inverse))
    )
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    graph.replay()
    phases = torch.cat(
        (
            torch.zeros(4, 13, device=device),
            positions[:, :1] * expected_table,
            positions[:, 1:] * expected_table,
        ),
        -1,
    )[:, None]
    a, b = source[..., :39].float(), source[..., 39:78].float()
    reference = torch.cat(
        (
            a * phases.cos() - b * phases.sin(),
            a * phases.sin() + b * phases.cos(),
            source[..., 78:].float(),
        ),
        -1,
    ).bfloat16()
    torch.testing.assert_close(output, reference, rtol=0.008, atol=0.0078125)
    graph.reset()
