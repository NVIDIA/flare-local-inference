"""A reduction on one rank copies into its output, or binds consumers to the producer's storage."""

import pytest
import torch

from ignition.components.collectives import PreparedAllReduce


def single(source, output):
    return PreparedAllReduce(source, output, runtime=None, session=None, coordinator=None, name="reduce")


def test_single_rank_reduction_copies_into_its_output():
    source, output = torch.arange(12, dtype=torch.bfloat16).view(3, 4), torch.full((3, 4), -1, dtype=torch.bfloat16)
    reduction = single(source, output)
    assert reduction(source) is output and not reduction.aliased
    assert torch.equal(output, source)
    assert output.data_ptr() != source.data_ptr()


def test_aliased_single_rank_reduction_runs_no_operation_and_exposes_the_source():
    source = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
    before = source.clone()
    reduction = single(source, None)
    assert reduction.aliased and reduction.output is source
    assert reduction(source) is source
    assert torch.equal(source, before)


def test_aliased_reduction_binds_a_view_of_the_producer_without_copying():
    backing = torch.arange(32, dtype=torch.bfloat16)
    source = backing[:24].view(6, 4)[:3]
    reduction = single(source, None)
    backing.fill_(7)
    assert reduction.output.data_ptr() == source.data_ptr()
    assert bool(reduction(source).eq(7).all())


@pytest.mark.parametrize("change", ["shape", "dtype", "storage"])
def test_aliased_reduction_rejects_a_different_producer_value(change):
    source = torch.zeros((3, 4), dtype=torch.bfloat16)
    reduction = single(source, None)
    other = {"shape": torch.zeros((2, 4), dtype=torch.bfloat16), "dtype": source.float(),
             "storage": torch.zeros((3, 4), dtype=torch.bfloat16)}[change]
    with pytest.raises(ValueError, match="bound to"):
        reduction(other)


def test_single_rank_copy_still_requires_a_matching_output():
    with pytest.raises(ValueError, match="matching output"):
        single(torch.zeros((3, 4), dtype=torch.bfloat16), torch.zeros((3, 5), dtype=torch.bfloat16))


def test_reduction_across_ranks_requires_an_output():
    with pytest.raises(ValueError, match="requires an output"):
        PreparedAllReduce(torch.zeros((3, 4), dtype=torch.bfloat16), None, runtime=object(), session=None,
                          coordinator=None, name="reduce")
