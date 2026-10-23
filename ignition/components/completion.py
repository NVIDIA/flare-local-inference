"""Persistent two-frame generation outputs and their request flags."""

from dataclasses import dataclass

import torch

from ignition.preparation.disposable import storage_ranges
from ignition.preparation.scratch import GraphBufferPool

MAX_TOP_LOGPROBS = 20
MAX_LOGPROB_TOKEN_IDS = 128
MAX_OUTPUT_TOKENS = 8
COMPLETION_FRAMES = 2
RECORD_FIELDS = 8
VECTOR_RECORD_FIELDS = 24
TOKEN_FIELD = 4
EMITTED_COUNT_FIELD = 8
VECTOR_TOKENS_FIELD = 16
# Each token/logprob pair occupies two int32 words, starting with the chosen token.
LOGPROB_PAIR_FIELDS = 2
LOGPROB_FIELDS = LOGPROB_PAIR_FIELDS * (1 + MAX_LOGPROB_TOKEN_IDS)
# Small chunks bound unused tails for reductions and mixed-role outputs; the
# general graph pool default would reserve far more than these buffers need.
_POOL_CHUNK_BYTES = 256 << 10


@dataclass(frozen=True)
class CompletionOutputs:
    """Own persistent outputs shared by the serial graph catalog.

    ``records`` is int64 [2, slots, 8 or 24] for ABI2 or ABI3. The runtime
    alternates frames so readback can retain one while the next graph writes.
    ``logprobs`` is int32 [2, slots, 8, 258]: the chosen token followed by up to
    128 requested alternatives, each packed as [token ID, FP32 logprob bits]. Per-slot
    int32 ``requested`` flags gate reductions; only committed positions are valid.
    ``scratch`` shares reduction storage across serial variants. ``private_storage``
    keeps mixed-role outputs disjoint until their active-frame merge.
    """
    records: torch.Tensor
    logprobs: torch.Tensor
    requested: torch.Tensor
    token_ids: torch.Tensor
    token_id_counts: torch.Tensor
    scratch: GraphBufferPool
    private_storage: GraphBufferPool
    speculative_survival: torch.Tensor | None = None

    @classmethod
    def allocate(cls, shape, *, device):
        frames, slots, fields = shape
        if frames != COMPLETION_FRAMES or fields not in (RECORD_FIELDS, VECTOR_RECORD_FIELDS):
            raise ValueError("completion records require two ABI2 or ABI3 frames")
        return cls(torch.zeros(shape, dtype=torch.int64, device=device),
            torch.zeros((frames, slots, MAX_OUTPUT_TOKENS, LOGPROB_FIELDS), dtype=torch.int32, device=device),
            torch.zeros(slots, dtype=torch.int32, device=device),
            torch.zeros((slots, MAX_LOGPROB_TOKEN_IDS), dtype=torch.int32, device=device),
            torch.zeros(slots, dtype=torch.int32, device=device), GraphBufferPool(device, chunk_bytes=_POOL_CHUNK_BYTES),
            GraphBufferPool(device, chunk_bytes=_POOL_CHUNK_BYTES),
            torch.full((frames, slots, MAX_OUTPUT_TOKENS - 1), -1., dtype=torch.float32, device=device)
            if fields == VECTOR_RECORD_FIELDS else None)

    def buffers(self):
        result = dict(completion_records=self.records, completion_logprobs=self.logprobs,
                      logprob_requested=self.requested, logprob_token_ids=self.token_ids,
                      logprob_token_id_counts=self.token_id_counts)
        if self.speculative_survival is not None:
            result["completion_speculative_survival"] = self.speculative_survival
        return result

    @property
    def nbytes(self):
        return sum(t.nbytes for t in self.buffers().values()) + self.scratch.nbytes + self.private_storage.nbytes

    def zero_ranges(self):
        return storage_ranges([self.buffers(), self.scratch.chunks, self.private_storage.chunks])

    def private(self, buffers):
        """Mixed roles share flags but retain disjoint outputs until their merge."""
        return CompletionOutputs(buffers.zeros(self.records.shape, torch.int64),
            buffers.zeros(self.logprobs.shape, torch.int32), self.requested, self.token_ids, self.token_id_counts,
            self.scratch, self.private_storage,
            buffers.zeros(self.speculative_survival.shape, torch.float32).fill_(-1.)
            if self.speculative_survival is not None else None)

    def merge(self, prompt, decode, prompt_slot, frame):
        slots = torch.arange(self.records.shape[1], device=self.records.device)
        role = slots == prompt_slot
        index = frame.long()
        for output, a, b in ((self.records, prompt.records, decode.records),
                             (self.speculative_survival, prompt.speculative_survival, decode.speculative_survival)):
            if output is None:
                continue
            mask = role.view(1, -1, *([1] * (output.ndim - 2)))
            output.index_copy_(0, index, torch.where(mask,
                -1. if output is self.speculative_survival else a.index_select(0, index), b.index_select(0, index)))
        from .logprobs import _merge
        torch.library.wrap_triton(_merge)[(self.records.shape[1], MAX_OUTPUT_TOKENS)](
            self.logprobs, prompt.logprobs, decode.logprobs, self.token_id_counts,
            prompt_slot, frame, self.records.shape[1], LOGPROB_FIELDS,
            MAX_OUTPUT_TOKENS, MAX_TOP_LOGPROBS, 512,
        )


def validate_completion_buffers(buffers, capacity):
    if (survival := buffers.get("completion_speculative_survival")) is not None:
        shape = (COMPLETION_FRAMES, capacity, MAX_OUTPUT_TOKENS - 1)
        if tuple(survival.shape) != shape or str(survival.dtype) != "torch.float32":
            raise ValueError(f"completion_speculative_survival: expected shape {shape}, dtype torch.float32")
    names = ("completion_logprobs", "logprob_requested", "logprob_token_ids", "logprob_token_id_counts")
    present = [name in buffers for name in names]
    if not any(present):
        return
    required = names if any(present[2:]) else names[:2]
    for name in required:
        if name not in buffers:
            raise ValueError(f"generation logprobs missing buffer {name}")
    fields = buffers[names[0]].shape[-1]
    if (any(present[2:]) and fields != LOGPROB_FIELDS) or (
            not any(present[2:]) and fields not in range(4, 2 * (1 + MAX_TOP_LOGPROBS) + 1, 2)):
        raise ValueError("completion_logprobs: expected shape with a valid alternative count and dtype torch.int32; "
                         f"got shape {tuple(buffers[names[0]].shape)}, dtype {buffers[names[0]].dtype}")
    for name, shape in ((names[0], (COMPLETION_FRAMES, capacity, MAX_OUTPUT_TOKENS, fields)),
                        (names[1], (capacity,)), (names[2], (capacity, MAX_LOGPROB_TOKEN_IDS)),
                        (names[3], (capacity,))):
        if name not in buffers:
            continue
        tensor = buffers[name]
        if tuple(tensor.shape) != shape or str(tensor.dtype) != "torch.int32":
            raise ValueError(f"{name}: expected shape {shape}, dtype torch.int32; "
                             f"got shape {tuple(tensor.shape)}, dtype {tensor.dtype}")
