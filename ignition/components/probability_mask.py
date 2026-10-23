"""Prepared masked stores into caller-owned verification probability buffers."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import prepare_triton


@triton.jit
def _mask_probabilities(Source, Destination, Valid, Limits, Dummy,
                        ELEMENTS: tl.constexpr, COLUMNS: tl.constexpr, VOCAB: tl.constexpr,
                        LIMITED: tl.constexpr, LIMIT_OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = index < ELEMENTS
    row = index // (COLUMNS * VOCAB)
    column = index // VOCAB % COLUMNS
    valid = tl.load(Valid + row, inside, other=0).to(tl.int1)
    if LIMITED:
        limit = tl.load(Limits + row, inside, other=0)
        valid = valid & (column < limit - LIMIT_OFFSET)
    value = tl.load(Source + index, inside & valid, other=0)
    dummy = tl.load(Dummy + index % VOCAB, inside, other=0)
    tl.store(Destination + index, tl.where(valid, value, dummy), inside)


class PreparedProbabilityMask:
    """Copy admitted columns and fill all others with a valid dummy distribution.

    Probability and destination tensors are contiguous FP32 views. The owner
    retains their storage and the row-validity, limit and dummy tensors through
    replay; validation of those operands belongs to the verification component.
    """

    def __init__(self, source, destination, valid, limits, dummy, *, limit_offset, outputs):
        self.source, self.destination = source, destination
        self.valid, self.limits, self.dummy = valid, limits, dummy
        self.limit_offset = limit_offset
        self.kernel = None
        begin, end = source.data_ptr(), source.data_ptr() + source.numel() * source.element_size()
        overlaps = any(source.numel() and output.numel() and source.device == output.device
            and begin < output.data_ptr() + output.numel() * output.element_size()
            and output.data_ptr() < end for output in outputs)
        # Explicit workspace views may overlap probability inputs. Preserve the
        # functional snapshot before the corresponding destination is written.
        self.snapshot = torch.empty_like(source) if overlaps else None
        self.prepared = False

    def prepare(self):
        batch, columns, vocab = self.source.shape
        self.prepared = True
        if not columns:
            return
        self.kernel = prepare_triton(_mask_probabilities,
            (self.source if self.snapshot is None else self.snapshot, self.destination, self.valid,
             self.valid if self.limits is None else self.limits, self.dummy),
            (triton.cdiv(batch * columns * vocab, 512), 1, 1),
            ELEMENTS=batch * columns * vocab, COLUMNS=columns, VOCAB=vocab,
            LIMITED=self.limits is not None, LIMIT_OFFSET=self.limit_offset,
            BLOCK=512, num_warps=8)

    def stage(self):
        if self.snapshot is not None:
            self.snapshot.copy_(self.source)

    def __call__(self):
        if not self.prepared:
            raise RuntimeError("probability masking requires prepare() before capture")
        if self.kernel is not None:
            self.kernel()
