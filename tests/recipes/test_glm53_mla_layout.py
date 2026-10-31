"""Per-capacity MLA input-projection views give each weight part a contiguous output."""

import pytest
import torch

from ignition.components.block_fp8 import writes_in_place
from ignition.components.weights import GroupedLinearWeights
from ignition.recipes.glm53.mla import MLAScratch, _input_projection_views

COLUMNS = (1536, 512, 128, 32)


def grouped(widths):
    return GroupedLinearWeights(tuple(torch.empty((size, 4), dtype=torch.bfloat16) for size in widths), widths)


@pytest.mark.parametrize("widths", [(2048, 160), (1536, 512, 160), (2048, 128, 32), (1536, 512, 128, 32),
                                    (1536, 672), (2176, 32), (1000, 1208)])
def test_parts_are_contiguous_and_every_consumer_reads_one_part(widths):
    parent = MLAScratch.allocate(65, 64, 8, "prefill", "cpu")
    strides = parent.projected_input.stride(), parent.query_input.stride(), parent.latent_input.stride()
    for tokens in (65, 8, 1, 17, 65):
        scratch = parent.rows(tokens, "prefill")
        local, outputs, key, gate = _input_projection_views(scratch, grouped(widths))
        parent.projected_input.fill_(-123)
        crossing = widths == (1000, 1208)
        if crossing:
            # The query latent spans both parts, so the row-concatenated layout remains.
            assert local is scratch and outputs is scratch.projected_input
            targets = outputs.split(widths, -1)
        else:
            assert isinstance(outputs, tuple) and [o.shape[1] for o in outputs] == list(widths)
            assert all(o.is_contiguous() for o in outputs)
            targets = outputs
        expected = []
        for index, (target, width) in enumerate(zip(targets, widths, strict=True)):
            values = torch.arange(tokens).view(-1, 1) % 64 + 64 * index
            target.copy_(values)
            expected.append(values.expand(tokens, width))
            assert target.untyped_storage().data_ptr() == parent.projected_input.untyped_storage().data_ptr()
        reference = torch.cat(expected, -1)
        consumers = (local.query_input, local.latent_input, key, gate)
        for actual, columns in zip(consumers, reference.split(COLUMNS, -1), strict=True):
            torch.testing.assert_close(actual.float(), columns.float(), rtol=0, atol=0)
        # Rows beyond the capacity are not written.
        assert bool(parent.projected_input[tokens:].eq(-123).all())
        assert local.projected_input is scratch.projected_input
        assert local.output is scratch.output and local.latent is scratch.latent
    assert (parent.projected_input.stride(), parent.query_input.stride(), parent.latent_input.stride()) == strides


def test_part_outputs_tile_the_projected_input_exactly():
    parent = MLAScratch.allocate(33, 64, 8, "prefill", "cpu")
    for tokens in (33, 5, 1):
        scratch = parent.rows(tokens, "prefill")
        _, outputs, _, _ = _input_projection_views(scratch, grouped((2048, 160)))
        # Every part is a contiguous, aligned destination, so a block-FP8 GEMM writes it in place.
        assert all(writes_in_place(o) for o in outputs)
        spans = sorted((o.data_ptr(), o.data_ptr() + o.numel() * o.element_size()) for o in outputs)
        base = scratch.projected_input.data_ptr()
        assert spans[0][0] == base and spans[0][1] == spans[1][0]
        assert spans[1][1] == base + scratch.projected_input.numel() * scratch.projected_input.element_size()


def test_ungrouped_projection_keeps_row_concatenated_layout():
    scratch = MLAScratch.allocate(8, 64, 8, "prefill", "cpu")
    weight = torch.empty((2208, 4), dtype=torch.bfloat16)
    local, output, key, gate = _input_projection_views(scratch, weight)
    assert local is scratch and output is scratch.projected_input
    assert key.data_ptr() == scratch.projected_input[:, 2048:2176].data_ptr()
    assert gate.data_ptr() == scratch.projected_input[:, 2176:2208].data_ptr()


def test_partitions_must_cover_the_projected_input():
    scratch = MLAScratch.allocate(8, 64, 8, "prefill", "cpu")
    with pytest.raises(ValueError, match="partitions must cover"):
        _input_projection_views(scratch, grouped((2048, 128)))
