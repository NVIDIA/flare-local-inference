"""Completion storage remains shared across serial mixed graph variants."""

import os

import pytest
import torch

from ignition.components.completion import (
    CompletionOutputs, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS, MAX_OUTPUT_TOKENS,
    validate_completion_buffers,
)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='CUDA opt-in')
def test_mixed_roles_merge_only_the_active_frame_and_reuse_the_catalog_pool():
    outputs = CompletionOutputs.allocate((2, 4, 24), device='cuda')
    cursor = outputs.private_storage.cursor()
    prompt, decode = outputs.private(cursor), outputs.private(cursor)
    assert prompt.requested is decode.requested is outputs.requested
    prompt.records.fill_(11)
    prompt.logprobs.fill_(12)
    decode.records.fill_(21)
    decode.logprobs.fill_(22)
    outputs.records.fill_(-7)
    outputs.logprobs.fill_(-8)
    prompt.speculative_survival.fill_(.9)
    decode.speculative_survival.fill_(.5)
    outputs.merge(prompt, decode, torch.tensor(2, device='cuda'), torch.tensor([1], dtype=torch.int32, device='cuda'))
    assert torch.all(outputs.records[0] == -7)
    assert torch.all(outputs.logprobs[0] == -8)
    assert torch.all(outputs.records[1, 2] == 11)
    assert torch.all(outputs.logprobs[1, 2, :, :42] == 12)
    assert torch.all(outputs.records[1, [0, 1, 3]] == 21)
    assert torch.all(outputs.logprobs[1, [0, 1, 3], :, :42] == 22)
    assert torch.all(outputs.logprobs[1, :, :, 42:] == -8)
    assert torch.all(outputs.speculative_survival[0] == -1)
    assert torch.all(outputs.speculative_survival[1, 2] == -1)
    assert torch.all(outputs.speculative_survival[1, [0, 1, 3]] == .5)
    size = outputs.private_storage.nbytes
    for _ in range(20):
        cursor = outputs.private_storage.cursor()
        a, b = outputs.private(cursor), outputs.private(cursor)
        assert a.logprobs.data_ptr() == prompt.logprobs.data_ptr()
        assert b.logprobs.data_ptr() == decode.logprobs.data_ptr()
    assert outputs.private_storage.nbytes == size
    assert outputs.zero_ranges()


@pytest.mark.parametrize('change', ['missing_output', 'missing_flags', 'wrong_dtype', 'wrong_frame', 'wrong_slots', 'wrong_k'])
def test_malformed_logprob_declarations_are_rejected(change):
    outputs = CompletionOutputs.allocate((2, 4, 24), device='meta')
    buffers = outputs.buffers()
    validate_completion_buffers(buffers, 4)
    if change == 'missing_output':
        del buffers['completion_logprobs']
    elif change == 'missing_flags':
        del buffers['logprob_requested']
    elif change == 'wrong_dtype':
        buffers['completion_logprobs'] = buffers['completion_logprobs'].float()
    else:
        shape = {
            'wrong_frame': (1, 4, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS),
            'wrong_slots': (2, 2, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS),
            'wrong_k': (2, 4, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS + LOGPROB_PAIR_FIELDS),
        }[change]
        buffers['completion_logprobs'] = torch.empty(shape, dtype=torch.int32, device='meta')
    with pytest.raises(ValueError) as raised:
        validate_completion_buffers(buffers, 4)
    message = str(raised.value)
    if change.startswith('missing_'):
        missing = 'completion_logprobs' if change == 'missing_output' else 'logprob_requested'
        assert missing in message
    else:
        assert 'completion_logprobs' in message
        assert 'expected shape' in message and 'dtype torch.int32' in message
        assert 'got shape' in message and 'dtype torch.' in message
    # Optional named buffers preserve the older export format.
    validate_completion_buffers({'completion_records': outputs.records}, 4)


@pytest.mark.parametrize('shape,dtype', [((1, 4, 7), torch.float32),
    ((2, 3, 7), torch.float32), ((2, 4, 8), torch.float32), ((2, 4, 7), torch.float64)])
def test_optional_survival_requires_two_float32_frames(shape, dtype):
    outputs = CompletionOutputs.allocate((2, 4, 24), device='meta')
    validate_completion_buffers(outputs.buffers(), 4)
    buffers = {'completion_records': outputs.records,
               'completion_speculative_survival': torch.empty(shape, dtype=dtype, device='meta')}
    with pytest.raises(ValueError, match='completion_speculative_survival'):
        validate_completion_buffers(buffers, 4)
    assert 'completion_speculative_survival' not in CompletionOutputs.allocate((2, 4, 8), device='meta').buffers()
