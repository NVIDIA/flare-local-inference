"""IEEE arithmetic conditions for omitting neutral conditional-count penalties."""

import pytest
import torch


@pytest.mark.parametrize("temperature", [0., 1., .7, torch.finfo(torch.float32).tiny])
def test_positive_zero_penalties_are_identity_for_nonnegative_counts(temperature):
    generator = torch.Generator().manual_seed(721)
    bits = torch.randint(-2**31, 2**31 - 1, (32768,), generator=generator, dtype=torch.int32)
    bits[:12] = torch.tensor([0, -2**31, 1, -2**31 + 1, 0x007fffff, -0x7f800001,
        0x00800000, -0x7f800000, 0x3f800000, -0x40800000, 0x7f800000, -0x00800000],
        dtype=torch.int32)
    logits = bits.view(torch.float32)
    for count in (0, 1, 7, 1048576):
        counts = torch.full(logits.shape, count, dtype=torch.int32)
        adjusted = torch.where(logits > 0, logits / 1., logits * 1.)
        penalized = torch.where(counts > 0, adjusted, logits)
        penalized = penalized - (counts > 0).float() * 0. - counts.float() * 0.
        divisor = temperature if temperature > 0 else 1.
        expected, actual = penalized / divisor, logits / divisor
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        defined = ~torch.isnan(expected)
        assert torch.equal(actual.view(torch.int32)[defined], expected.view(torch.int32)[defined])


@pytest.mark.parametrize("presence,frequency", [(-0., 0.), (0., -0.), (-0., -0.)])
def test_negative_zero_penalties_require_the_unconditional_arithmetic(presence, frequency):
    logits = torch.tensor([-0.], dtype=torch.float32)
    counts = torch.ones(1, dtype=torch.int32)
    penalized = logits - (counts > 0).float() * presence - counts.float() * frequency
    assert penalized.view(torch.int32).item() == 0
    assert logits.view(torch.int32).item() == -2**31
