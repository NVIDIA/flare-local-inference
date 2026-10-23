"""Captured chain acceptance and request-state commit.

Model verification produces one row for the pending input and each draft.
Commit emits a clipped prefix and consumes the equally long prefix of those
inputs. Model checkpoints and MTP history are selected by the caller using the
returned counts; these operations never reconstruct model state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.flashinfer_sampling import filter_probabilities_out, probability_filter_workspace, verify_chain

from ignition.components.capacity import BATCH_CAPACITIES
from ignition.components.logprobs import PreparedLogprobs
from ignition.components.sampling import (
    DUPLICATE_SLOT,
    ERROR,
    FINISHED,
    INVALID_DISTRIBUTION,
    RUNNING,
)

from ignition.components.sampling import DeviceRequestState
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.scratch import bind_scratch

INVALID_SPECULATION = 6
TARGET_RNG_DOMAIN = 0x4D54505F56455249
DRAFT_RNG_DOMAIN = 0x4D54505F44524146
WIDTHS = tuple(range(1, 9))
# Vocabulary-wide checks reduce each row in slices first, so compiled forwards spread
# them over many programs instead of one serial program per row.
VOCABULARY_SLICES = 128


def _sliced(values: torch.Tensor) -> torch.Tensor:
    slices = VOCABULARY_SLICES if values.shape[-1] % VOCABULARY_SLICES == 0 else 1
    return values.unflatten(-1, (slices, values.shape[-1] // slices))


def _tensor(value, shape, dtype, device, name):
    if value.shape != torch.Size(shape) or value.dtype != dtype or value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} requires contiguous {dtype} {tuple(shape)} on {device}")


def _geometry(batch, width):
    if type(batch) is not int or batch not in BATCH_CAPACITIES or type(width) is not int or width not in WIDTHS:
        raise ValueError("speculation requires power-of-two batch capacity through 64 and width 1 through 8")


def _metadata(state, slots, generations, live_rows, batch, device):
    if state.generation.device != device:
        raise ValueError("request state and speculative operands must share a device")
    _tensor(slots, (batch,), torch.int32, device, "slots")
    _tensor(generations, (batch,), torch.int64, device, "generations")
    _tensor(live_rows, (1,), torch.int32, device, "live_rows")


def _live(state, slots, generations, live_rows, rows):
    safe = slots.long().clamp(0, state.capacity - 1)
    selected = (rows < live_rows[0]) & (live_rows[0] >= 0) & (live_rows[0] <= slots.numel())
    selected &= (slots >= 0) & (slots < state.capacity) & (state.generation[safe] == generations)
    duplicates = ((slots[:, None] == slots[None, :]) & selected[None, :]).sum(-1) > 1
    return safe, selected & ~duplicates & (state.status[safe] == RUNNING)


@dataclass
class SpeculativeAcceptance:
    token_ids: torch.Tensor
    proposed_count: torch.Tensor
    accepted_drafts: torch.Tensor
    offered_drafts: torch.Tensor
    valid: torch.Tensor
    rng_increment: torch.Tensor
    survival: torch.Tensor

    @classmethod
    def storage_layout(cls, batch, width):
        _geometry(batch, width)
        return dict(token_ids=((batch, width), torch.int64),
                    proposed_count=((batch,), torch.int32), accepted_drafts=((batch,), torch.int32),
                    offered_drafts=((batch,), torch.int32), valid=((batch,), torch.bool),
                    rng_increment=((batch,), torch.int64), survival=((batch, 7), torch.float32))

    @classmethod
    def broadcast_layout(cls, batch, width):
        return {name: layout for name, layout in cls.storage_layout(batch, width).items() if name != "survival"}

    @classmethod
    def allocate(cls, batch, width, device):
        return cls(**{name: torch.full(shape, -1 if name in ("token_ids", "survival") else 0, dtype=dtype, device=device)
                      for name, (shape, dtype) in cls.storage_layout(batch, width).items()})

    def validate(self, batch, width, device):
        for name, (shape, dtype) in self.storage_layout(batch, width).items():
            _tensor(getattr(self, name), shape, dtype, device, name)


@dataclass
class SpeculativeRngState:
    """Physical-slot target-verification stream, independent of proposal draws."""

    generation: torch.Tensor
    target_offset: torch.Tensor

    @classmethod
    def allocate(cls, device, *, capacity=4):
        return cls(torch.zeros(capacity, dtype=torch.int64, device=device),
                   torch.zeros(capacity, dtype=torch.int64, device=device))


class PreparedCopyProposals:
    """Validate physical-slot copy commands before verification mutates model state.

    A copied token is a deterministic proposal distribution, including when the
    target sampler is stochastic. Only target verification consumes RNG draws.
    """

    def __init__(self, requests, slots, generations, live_rows, *, copy_ids,
                 copy_count, copy_expected, copy_generation, grammar_mask_count,
                 draft_ids, draft_probabilities, draft_valid, copy_mask,
                 previous_accepted, accepted_generation, token_limits=None):
        batch, drafts = draft_ids.shape
        _geometry(batch, drafts + 1)
        device = draft_ids.device
        _metadata(requests, slots, generations, live_rows, batch, device)
        for value, shape, dtype, name in (
                (copy_ids, (requests.capacity, 7), torch.int64, "copy_ids"),
                (copy_count, (requests.capacity,), torch.int32, "copy_count"),
                (copy_expected, (requests.capacity,), torch.int64, "copy_expected"),
                (copy_generation, (requests.capacity,), torch.int64, "copy_generation"),
                (grammar_mask_count, (requests.capacity,), torch.int32, "grammar_mask_count"),
                (draft_ids, (batch, drafts), torch.int64, "draft_ids"),
                (draft_probabilities, (batch, drafts, requests.vocab_size), torch.float32, "draft_probabilities"),
                (draft_valid, (batch, drafts), torch.bool, "draft_valid"),
                (copy_mask, (batch,), torch.int64, "copy_mask"),
                (previous_accepted, (requests.capacity,), torch.int32, "previous_accepted"),
                (accepted_generation, (requests.capacity,), torch.int64, "accepted_generation")):
            _tensor(value, shape, dtype, device, name)
        self.state, self.slots, self.generations, self.live = requests, slots, generations, live_rows
        self.ids, self.count, self.expected, self.generation = copy_ids, copy_count, copy_expected, copy_generation
        self.mask_count = grammar_mask_count
        if token_limits is not None:
            _tensor(token_limits, (batch,), torch.int32, device, "token_limits")
        self.token_limits = token_limits
        self.columns = torch.arange(drafts, device=device)
        self.drafts, self.probabilities, self.valid, self.copy_mask = draft_ids, draft_probabilities, draft_valid, copy_mask
        self.previous_accepted, self.accepted_generation = previous_accepted, accepted_generation
        self.rows = torch.arange(batch, device=device)
        self.active = torch.zeros(batch, dtype=torch.bool, device=device)

    def __call__(self):
        state, drafts = self.state, self.drafts.shape[1]
        safe, live = _live(state, self.slots, self.generations, self.live, self.rows)
        ids = self.ids[safe, :drafts]
        required = drafts if self.token_limits is None else self.token_limits - 1
        used = self.columns < required if self.token_limits is None else self.columns[None, :] < required[:, None]
        valid = ((self.generation[safe] == self.generations)
            & (self.expected[safe] == state.consumed_tokens[safe])
            & (self.count[safe] >= required) & (self.count[safe] <= 7)
            & (((ids >= 0) & (ids < state.vocab_size)) | ~used).all(-1))
        constrained = state.grammar_enabled[safe].bool()
        valid &= torch.where(constrained,
            (self.mask_count[safe] >= required + 1) & (self.mask_count[safe] <= 8),
            self.mask_count[safe] == 0)
        self.active.copy_(live & valid)
        for slot in range(state.capacity):
            failed = (live & ~valid & (self.slots == slot)).any()
            state.status[slot].copy_(torch.where(failed, ERROR, state.status[slot]))
            state.error_code[slot].copy_(torch.where(failed, INVALID_SPECULATION, state.error_code[slot]))
            reset = (self.active & (self.slots == slot)).any() & (self.accepted_generation[slot] != state.generation[slot])
            self.previous_accepted[slot].copy_(torch.where(reset, 1, self.previous_accepted[slot]))
            self.accepted_generation[slot].copy_(torch.where(reset, state.generation[slot], self.accepted_generation[slot]))
        self.drafts.copy_(torch.where(self.active[:, None] & used, ids, 0))
        self.valid.copy_(self.active[:, None].expand_as(self.valid))
        self.probabilities.zero_()
        self.probabilities.scatter_(2, self.drafts[:, :, None], 1.)
        self.copy_mask.copy_(torch.where(self.active, (1 << required) - 1, 0))
        return self


class PreparedSpeculativeDistributions:
    """Conditional request filtering using the ordinary FlashInfer policy.

    Row j includes candidate IDs before j+prefix_offset in its penalty counts.
    The committed histogram already contains the incoming pending token.
    Optional physical-slot grammar masks describe each conditional prefix.
    """

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return dict(scores=((batch, width, vocab), torch.float32),
            probabilities=((batch, width, vocab), torch.float32),
            maximum=((batch, width), torch.float32), valid=((batch, width), torch.bool),
            packed_top_k=((batch * width,), torch.int32), packed_top_p=((batch * width,), torch.float32))

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                 *, prefix_offset=0, grammar_masks=None, grammar_mask_count=None, workspace=None):
        batch, width, vocab = logits.shape
        _geometry(batch, width)
        device = logits.device
        if logits.dtype not in (torch.float32, torch.bfloat16) or not logits.is_contiguous() or vocab != requests.vocab_size:
            raise ValueError("conditional logits require contiguous FP32/BF16 [B,W,vocabulary]")
        _metadata(requests, slots, generations, live_rows, batch, device)
        if candidate_prefix_ids.ndim != 2 or candidate_prefix_ids.shape[0] != batch:
            raise ValueError("candidate_prefix_ids requires [B,K]")
        _tensor(candidate_prefix_ids, candidate_prefix_ids.shape, torch.int64, device, "candidate_prefix_ids")
        if type(prefix_offset) is not int or prefix_offset < 0 or width - 1 + prefix_offset > candidate_prefix_ids.shape[1]:
            raise ValueError("prefix_offset must keep every conditional prefix within candidate IDs")
        self.logits, self.state, self.slots, self.generations, self.live_rows = logits, requests, slots, generations, live_rows
        self.candidates, self.prefix_offset = candidate_prefix_ids, prefix_offset
        if (grammar_masks is None) != (grammar_mask_count is None):
            raise ValueError("conditional grammar masks and counts must be provided together")
        if grammar_masks is not None:
            _tensor(grammar_masks, (requests.capacity, 8, (vocab + 31) // 32), torch.int32, device, "grammar_masks")
            _tensor(grammar_mask_count, (requests.capacity,), torch.int32, device, "grammar_mask_count")
            if width + prefix_offset > 8:
                raise ValueError("conditional grammar prefix exceeds eight mask rows")
        self.grammar_masks, self.grammar_mask_count = grammar_masks, grammar_mask_count
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.vocab_ids = torch.arange(vocab, device=device)
        for tensor in (self.rows, self.columns, self.vocab_ids):
            record_numeric_tensor(tensor)
        for name, tensor in bind_scratch(self.scratch_layout(batch, width, vocab),
                                        device=device, workspace=workspace).items():
            setattr(self, name, tensor)
        from ignition.components.probability_normalization import PreparedProbabilityNormalization
        self.normalization = PreparedProbabilityNormalization(self.probabilities, self.valid)
        self._prepared = False

    def prepare_scores(self):
        state, columns = self.state, self.columns
        safe, live = _live(state, self.slots, self.generations, self.live_rows, self.rows)
        counts = state.histogram[safe, None, :]
        prefix = columns + self.prefix_offset
        prefix_valid = torch.ones_like(self.valid)
        for index in range(columns.numel() - 1 + self.prefix_offset):
            ids = self.candidates[:, index]
            used = index < prefix
            valid = (ids >= 0) & (ids < state.vocab_size)
            prefix_valid &= valid[:, None] | ~used[None, :]
            matches = self.vocab_ids[None, None, :] == ids[:, None, None]
            additions = matches & valid[:, None, None] & used[None, :, None]
            counts = counts + additions.to(torch.int32)
        params = state.parameters[safe]
        scores = self.logits.float()
        penalty = params[:, None, 3:4]
        adjusted = torch.where(scores > 0, scores / penalty, scores * penalty)
        scores = torch.where(counts > 0, adjusted, scores)
        scores = scores - (counts > 0) * params[:, None, 4:5] - counts * params[:, None, 5:6]
        temperature = params[:, None, :1]
        scores = scores / torch.where(temperature > 0, temperature, 1)
        if self.grammar_masks is None:
            words = state.grammar_mask[safe][:, None, self.vocab_ids // 32]
        else:
            words = self.grammar_masks[safe][:, prefix][:, :, self.vocab_ids // 32]
        allowed = ((words >> (self.vocab_ids % 32)) & 1).bool() | ~state.grammar_enabled[safe, None, None].bool()
        self.scores.copy_(torch.where(allowed, scores, -torch.inf))
        self.maximum.copy_(_sliced(self.scores).amax(-1).amax(-1))
        self.valid.copy_(live[:, None] & prefix_valid & torch.isfinite(self.maximum))
        if self.grammar_masks is not None:
            count = self.grammar_mask_count[safe, None]
            self.valid.logical_and_(~state.grammar_enabled[safe, None].bool()
                                    | ((prefix[None, :] < count) & (count <= 8)))
        elif columns.numel() > 1:
            self.valid.logical_and_(~state.grammar_enabled[safe, None].bool() | (prefix[None, :] == 0))
        self.packed_top_k.copy_(torch.where(state.top_k[safe] > 0, state.top_k[safe], state.vocab_size)[:, None].expand_as(self.valid).reshape(-1))
        self.packed_top_p.copy_(params[:, 1:2].expand_as(self.valid).reshape(-1))
        return safe, params

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("FlashInfer probability filtering requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        self.library_scratch = (
            sampling._get_cache_buf(f"top_k_mask_logits_row_states_{self.logits.device}", 1024 * 1024, self.logits.device, zero_init=True),
            sampling._get_cache_buf("softmax_workspace", 1024 * 1024, self.logits.device),
        )
        self.top_p_workspace = probability_filter_workspace(
            self.packed_top_p.numel(), self.state.vocab_size, self.logits.device)
        self._prepared = True
        self()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("probability filtering requires prepare() before capture")
        _, params = self.prepare_scores()
        dummy = torch.where(self.vocab_ids == 0, 0., -torch.inf)
        self.scores.copy_(torch.where(self.valid[:, :, None], self.scores, dummy))
        flat = self.scores.view(-1, self.state.vocab_size)
        filter_probabilities_out(flat, self.packed_top_k, self.packed_top_p,
            *self.library_scratch, self.probabilities.view_as(flat), self.top_p_workspace)
        return self.normalization(self.probabilities, params)


class PreparedStochasticVerification:
    """Fixed-width one-row FlashInfer calls with device-owned RNG offsets.

    K+2 bounds scalar Philox draws, including the independent-acceptance metric.
    Reservations are rounded to four draws. Functional library outputs remain
    graph-private and are copied into the persistent acceptance buffers.
    """

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return dict(safe_draft_probs=((batch, width - 1, vocab), torch.float32),
            safe_target_probs=((batch, width, vocab), torch.float32),
            safe_ids=((batch, width - 1), torch.int32), seed_bits=((batch,), torch.int64),
            offset=((batch,), torch.int64), independent_accepted=((batch,), torch.int32),
            emitted_drafts=((batch,), torch.int32))

    def __init__(self, draft_probs, draft_token_ids, target_probs, requests, slots, generations,
                 live_rows, *, rng, draft_valid=None, target_valid=None, output=None, token_limits=None,
                 workspace=None):
        batch, width, vocab = target_probs.shape
        _geometry(batch, width)
        device = target_probs.device
        for value, shape, dtype, name in ((draft_probs, (batch, width - 1, vocab), torch.float32, "draft_probs"),
                (draft_token_ids, (batch, width - 1), torch.int64, "draft_token_ids"),
                (target_probs, (batch, width, vocab), torch.float32, "target_probs")):
            _tensor(value, shape, dtype, device, name)
        _metadata(requests, slots, generations, live_rows, batch, device)
        if vocab != requests.vocab_size:
            raise ValueError("verification vocabulary must match request state")
        for name in ("generation", "target_offset"):
            _tensor(getattr(rng, name), (requests.capacity,), torch.int64, device, f"rng.{name}")
        self.draft_probs, self.drafts, self.target_probs = draft_probs, draft_token_ids, target_probs
        self.state, self.slots, self.generations, self.live_rows, self.rng = requests, slots, generations, live_rows, rng
        self.draft_valid = torch.ones((batch, width - 1), dtype=torch.bool, device=device) if draft_valid is None else draft_valid
        self.target_valid = torch.ones((batch, width), dtype=torch.bool, device=device) if target_valid is None else target_valid
        _tensor(self.draft_valid, (batch, width - 1), torch.bool, device, "draft_valid")
        _tensor(self.target_valid, (batch, width), torch.bool, device, "target_valid")
        self.output = SpeculativeAcceptance.allocate(batch, width, device) if output is None else output
        self.output.validate(batch, width, device)
        self.rows, self.columns = torch.arange(batch, device=device), torch.arange(width, device=device)
        self.dummy = (torch.arange(vocab, device=device) == 0).float()
        for name, tensor in bind_scratch(self.scratch_layout(batch, width, vocab),
                                        device=device, workspace=workspace).items():
            setattr(self, name, tensor)
        self.draw_reservation = 4 * ((width + 1 + 3) // 4)
        if token_limits is not None:
            _tensor(token_limits, (batch,), torch.int32, device, "token_limits")
        self.token_limits = token_limits
        self._prepared = False
        self.broadcast = None
        from .probability_mask import PreparedProbabilityMask

        outputs = (self.safe_draft_probs, self.safe_target_probs)
        self.probability_masks = (
            PreparedProbabilityMask(self.draft_probs, self.safe_draft_probs, self.output.valid,
                token_limits, self.dummy, limit_offset=1, outputs=outputs),
            PreparedProbabilityMask(self.target_probs, self.safe_target_probs, self.output.valid,
                token_limits, self.dummy, limit_offset=0, outputs=outputs),
        )

    def prepare_broadcast(self, **kwargs):
        from .collectives import PreparedBroadcast

        self.broadcast = PreparedBroadcast(tuple(value for name, value in vars(self.output).items()
                                                 if name != "survival"), **kwargs)

    def prepare(self):
        if self.target_probs.device.type != "cuda":
            raise ValueError("FlashInfer chain verification requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        for mask in self.probability_masks:
            mask.prepare()
        self._prepared = True
        self()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("chain verification requires prepare() before capture")
        out = self.output
        safe, live = _live(self.state, self.slots, self.generations, self.live_rows, self.rows)
        if self.token_limits is None:
            valid = live & self.draft_valid.all(-1) & self.target_valid.all(-1)
            valid &= ((self.drafts >= 0) & (self.drafts < self.state.vocab_size)).all(-1)
            draft_probs, target_probs, ids = self.draft_probs, self.target_probs, self.drafts
        else:
            target_used = self.columns[None, :] < self.token_limits[:, None]
            draft_used = self.columns[None, :-1] < self.token_limits[:, None] - 1
            valid = live & (self.draft_valid | ~draft_used).all(-1) & (self.target_valid | ~target_used).all(-1)
            valid &= (((self.drafts >= 0) & (self.drafts < self.state.vocab_size)) | ~draft_used).all(-1)
            valid &= (self.token_limits >= 1) & (self.token_limits <= self.target_probs.shape[1])
            draft_probs = torch.where(draft_used[:, :, None], self.draft_probs, self.dummy)
            target_probs = torch.where(target_used[:, :, None], self.target_probs, self.dummy)
            ids = torch.where(draft_used, self.drafts, 0)
        for probabilities in (draft_probs, target_probs):
            slices = _sliced(probabilities)
            entries = (torch.isfinite(slices) & (slices >= 0)).all(-1).all(-1)
            mass = slices.sum(-1).sum(-1)
            valid &= (entries & torch.isfinite(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)
        out.valid.copy_(valid)
        if self.token_limits is not None:
            for mask in self.probability_masks:
                mask.stage()
        for mask in self.probability_masks:
            if self.token_limits is None:
                mask.stage()
            mask()
        self.safe_ids.copy_(torch.where(valid[:, None], ids, 0))
        self.seed_bits.copy_(self.state.seed.view(torch.int64)[safe] ^ TARGET_RNG_DOMAIN)
        self.offset.copy_(torch.where(self.rng.generation[safe] == self.generations, self.rng.target_offset[safe], 0))
        self.independent_accepted.zero_()
        self.emitted_drafts.zero_()
        tokens, _, _ = verify_chain(self.safe_draft_probs, self.safe_ids, self.safe_target_probs,
            self.independent_accepted, self.emitted_drafts, seed=self.seed_bits, offset=self.offset)
        accepted = self.emitted_drafts
        if self.token_limits is None:
            out.offered_drafts.fill_(self.drafts.shape[1])
        else:
            # The fixed-width library may accept dummy columns beyond a short
            # tail. Only the real prefix belongs to the commit contract.
            out.offered_drafts.copy_((self.token_limits - 1).clamp(0, self.drafts.shape[1]))
            accepted = torch.minimum(accepted, out.offered_drafts)
        out.token_ids.copy_(tokens)
        out.accepted_drafts.copy_(torch.where(valid, accepted, 0))
        out.proposed_count.copy_(torch.where(valid, accepted + 1, 0))
        out.rng_increment.copy_(torch.where(valid, self.draw_reservation, 0))
        out.token_ids.masked_fill_(~valid[:, None], -1)
        if self.drafts.shape[1]:
            indices = self.safe_ids.long().unsqueeze(-1)
            draft = self.safe_draft_probs.gather(-1, indices).squeeze(-1)
            target = self.safe_target_probs[:, :-1].gather(-1, indices).squeeze(-1)
            ratio = torch.where(draft > 0, (target / draft).clamp(max=1.), (target > 0).float())
            # Serial products preserve monotonicity despite FP32 rounding at unit ratios.
            prefixes = [ratio[:, 0]]
            for depth in range(1, self.drafts.shape[1]):
                prefixes.append(prefixes[-1] * ratio[:, depth])
            out.survival.fill_(-1.)
            out.survival[:, :self.drafts.shape[1]].copy_(torch.where(valid[:, None], torch.stack(prefixes, -1), -1.))
        if self.broadcast is not None:
            self.broadcast()
        return out


class PreparedSpeculativeCommit:
    """Commit a verified chain prefix and emit physical-slot ABI3 records.

    A round emitting t tokens commits verification input t-1, leaving emitted
    token t-1 pending. Exact snapshot maintenance selects the committed state
    column after the request's model submissions have drained.
    """

    def __init__(self, requests: DeviceRequestState, acceptance: SpeculativeAcceptance,
                 input_ids, slots, generations, live_rows, completion, frame_index,
                 submission_sequence, *, logits, rng=None, copy_offered_mask=None, draft_context_lag=1, token_limits=None, buffers=None):
        records = completion.records
        batch, width = input_ids.shape
        _geometry(batch, width)
        device = input_ids.device
        if draft_context_lag is not None and (type(draft_context_lag) is not int or draft_context_lag < 0):
            raise ValueError("draft_context_lag must be a nonnegative token count or None")
        self.draft_context_lag = draft_context_lag
        self.history = getattr(requests, "ple_history", None)
        if token_limits is not None:
            _tensor(token_limits, (batch,), torch.int32, device, "token_limits")
        self.token_limits = token_limits
        if logits.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("target_logits requires BF16 or FP32 storage")
        _tensor(logits, (batch, width, requests.vocab_size), logits.dtype, device, "target_logits")
        _tensor(input_ids, (batch, width), torch.int64, device, "input_ids")
        _metadata(requests, slots, generations, live_rows, batch, device)
        acceptance.validate(batch, width, device)
        _tensor(records, (2, requests.capacity, 24), torch.int64, device, "completion_records")
        _tensor(frame_index, (1,), torch.int32, device, "frame_index")
        _tensor(submission_sequence, (1,), torch.int64, device, "submission_sequence")
        self.copy_offered_mask = torch.zeros(batch, dtype=torch.int64, device=device) if copy_offered_mask is None else copy_offered_mask
        _tensor(self.copy_offered_mask, (batch,), torch.int64, device, "copy_offered_mask")
        if rng is not None:
            for name in ("generation", "target_offset"):
                _tensor(getattr(rng, name), (requests.capacity,), torch.int64, device, f"rng.{name}")
        self.logprobs = PreparedLogprobs(logits, completion, slots, generations, live_rows, frame_index)
        self.state, self.acceptance, self.inputs, self.slots = requests, acceptance, input_ids, slots
        self.generations, self.live, self.records, self.frame, self.sequence = generations, live_rows, records, frame_index, submission_sequence
        self.rng = rng
        self.rows, self.columns = torch.arange(batch, device=device), torch.arange(width, device=device)
        self.eos_columns = torch.arange(requests.eos_token_ids.shape[1], device=device)
        self.history_columns = torch.arange(2, device=device)
        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.emitted_ids = zeros(input_ids.shape).fill_(-1)
        self.committed_input_ids = zeros(input_ids.shape).fill_(-1)
        self.emitted_count = zeros((batch,), torch.int32)
        self.accepted_count = zeros((batch,), torch.int32)
        self.accepted_drafts = zeros((batch,), torch.int32)
        self.commit_column = zeros((batch,), torch.int32).fill_(-1)
        self.consumed_before = zeros((batch,))
        self.generated_before = zeros((batch,))
        self.history_before = zeros((batch, 2))
        # Pool-backed request state is staged here, so compiled commits write the pool without copying it.
        self.consumed_after = zeros((requests.capacity,))
        self.history_after = zeros((requests.capacity, 2))
        # One extra entry receives the rows that own no slot.
        self.owners = zeros((requests.capacity + 1,))
        self._records = zeros((requests.capacity, 24))
        self.survival = completion.speculative_survival
        self._survival = zeros((requests.capacity, 7), torch.float32) if self.survival is not None and width > 1 else None
        self.survival_columns = torch.arange(7, device=device)
        self.idle_record = torch.zeros(24, dtype=torch.int64, device=device)
        self.idle_record[[4, 12, *range(16, 24)]] = -1
        if draft_context_lag is None:
            self.idle_record[13] = -1

    def prepare(self):
        self.logprobs.prepare()

    def __call__(self):
        state, acceptance, columns = self.state, self.acceptance, self.columns
        width = columns.numel()
        safe = self.slots.long().clamp(0, state.capacity - 1)
        selected = ((self.rows < self.live[0]) & (self.live[0] >= 0) & (self.live[0] <= self.rows.numel())
            & (self.slots >= 0) & (self.slots < state.capacity) & (state.generation[safe] == self.generations))
        self.consumed_before.copy_(state.consumed_tokens[safe])
        self.generated_before.copy_(state.generated_tokens[safe])
        if self.history is not None:
            self.history_before.copy_(self.history[safe])

        # Every row commits from its slot's last selected row, which alone writes the slot's state.
        # Rows are evaluated together from the state before the round; each slot then takes its
        # owner's values in one update, so no slot is written twice whatever the package capacity.
        matches = selected[:, None] & selected[None, :] & (self.slots[:, None] == self.slots[None, :])
        count = matches.sum(1)
        row = torch.where(matches, self.rows[None, :], 0).amax(1)
        leader = selected & (row == self.rows)
        ids = acceptance.token_ids[row]
        inputs = self.inputs[row]
        proposed = acceptance.proposed_count[row].long()
        accepted = acceptance.accepted_drafts[row].long()
        offered = acceptance.offered_drafts[row].long()
        valid = acceptance.valid[row]
        increment = acceptance.rng_increment[row]
        status_before, error_before = state.status[safe], state.error_code[safe]
        generated_before, pending_before = state.generated_tokens[safe], state.pending_token[safe]
        max_new_tokens = state.max_new_tokens[safe]
        remaining = (max_new_tokens - generated_before).clamp_min(0)
        enabled = (count == 1) & (status_before == RUNNING) & (remaining > 0)
        proposal = columns[None, :] < proposed[:, None]
        shape_valid = ((proposed >= 1) & (proposed <= width) & (accepted == proposed - 1)
            & (offered >= accepted) & (offered <= width - 1))
        shape_valid &= (((ids >= 0) & (ids < state.vocab_size)) | ~proposal).all(-1)
        shape_valid &= (((inputs >= 0) & (inputs < state.vocab_size)) | ~proposal).all(-1)
        shape_valid &= (pending_before >= 0) & (inputs[:, 0] == pending_before)
        shape_valid &= ((ids[:, :-1] == inputs[:, 1:]) | ~(columns[None, :-1] < accepted[:, None])).all(-1)
        if self.rng is None:
            shape_valid &= increment == 0
        else:
            shape_valid &= increment >= 0
        valid = valid & shape_valid
        eos = ((ids[:, :, None] == state.eos_token_ids[safe][:, None, :])
            & (self.eos_columns[None, None, :] < state.eos_count[safe][:, None, None])).any(-1)
        eos_position = torch.where(eos & proposal, columns[None, :] + 1, width).amin(-1)
        emitted = torch.minimum(torch.minimum(proposed, remaining), eos_position)
        if self.token_limits is not None:
            emitted = torch.minimum(emitted, self.token_limits[row])
        emitted = torch.where(enabled & valid, emitted, 0)
        emitted_drafts = torch.minimum(accepted.clamp_min(0), emitted)
        take = columns[None, :] < emitted[:, None]
        pending = ids.gather(1, (emitted - 1).clamp(0, width - 1)[:, None])[:, 0]
        consumed = state.consumed_tokens[safe] + emitted
        generated = generated_before + emitted
        terminal = (generated >= max_new_tokens) | (eos & take).any(-1)
        failed = enabled & ~valid
        duplicate = count > 1
        status = torch.where(duplicate | failed, ERROR,
            torch.where((count == 1) & (status_before == RUNNING) & (terminal | (remaining == 0)), FINISHED, status_before))
        error = torch.where(duplicate, DUPLICATE_SLOT,
            torch.where(failed, torch.where(shape_valid, INVALID_DISTRIBUTION, INVALID_SPECULATION), error_before))
        generation = state.generation[safe]
        counter = state.rng_counter.view(torch.int64)
        counters = counter[safe] + emitted.view(-1, *([1] * (counter.dim() - 1)))
        if self.history is not None:
            history = self.history[safe]
            tail = torch.cat((history, inputs), 1).gather(1, emitted[:, None] + self.history_columns[None, :])
            history = torch.where((emitted > 0)[:, None], tail, history)
        if self.rng is not None:
            offset = self.rng.target_offset[safe]
            base = torch.where(self.rng.generation[safe] == generation, offset, 0)
            advance = (emitted > 0) & (increment > 0)
            rng_offset = torch.where(advance, base + increment, offset)
            rng_generation = torch.where(advance, generation, self.rng.generation[safe])
        live = enabled & valid
        record = torch.cat((torch.stack((generation, self.sequence.expand_as(generation), consumed, generated,
            torch.where(emitted > 0, pending, -1), status.long(), error.long(), torch.zeros_like(emitted), emitted,
            torch.where(live, offered, 0), emitted_drafts, torch.where(live, self.copy_offered_mask[row], 0),
            torch.full_like(emitted, -1),
            ((consumed - self.draft_context_lag).clamp_min(0) if self.draft_context_lag is not None
             else torch.full_like(emitted, -1)),
            torch.zeros_like(emitted), torch.full_like(emitted, width)), 1),
            torch.where(take, ids, -1), torch.full((emitted.numel(), 8 - width), -1, dtype=torch.int64, device=ids.device)), 1)

        # Records and per-row outputs precede request-state mutation: compiled graphs must not
        # re-derive enabled/valid fields from the status and token counts this round rewrites.
        self.emitted_ids.copy_(torch.where(selected[:, None] & take, ids, -1))
        self.committed_input_ids.copy_(torch.where(selected[:, None] & take, inputs, -1))
        self.emitted_count.copy_(torch.where(selected, emitted, 0))
        self.accepted_count.copy_(torch.where(selected, emitted, 0))
        self.accepted_drafts.copy_(torch.where(selected, emitted_drafts, 0))
        self.commit_column.copy_(torch.where(selected, emitted - 1, -1))
        self.owners.fill_(-1)
        self.owners.scatter_(0, torch.where(leader, safe, state.capacity), self.rows)
        owner = self.owners[:state.capacity]
        owned, source = owner >= 0, owner.clamp_min(0)
        if self._survival is not None:
            logical = torch.minimum(offered, (remaining - 1).clamp_min(0))
            if self.token_limits is not None:
                logical = torch.minimum(logical, (self.token_limits[row] - 1).clamp_min(0))
            extent = torch.minimum(logical, torch.where(emitted > emitted_drafts, logical, emitted_drafts))
            observed = owned[:, None] & live[source, None] & (self.survival_columns < extent[source, None])
            self._survival.copy_(torch.where(observed, acceptance.survival[row][source], -1.))
            self.survival.index_copy_(0, self.frame.long(), self._survival.unsqueeze(0))

        def updated(tensor, values):
            mask = owned.view(-1, *([1] * (tensor.dim() - 1)))
            return torch.where(mask, values[source].to(tensor.dtype), tensor)

        def commit(tensor, values):
            tensor.copy_(updated(tensor, values))

        # Consumed tokens and history may be views of the state pool: both are read before
        # either is written, so compiled commits never materialize an updated pool.
        self.consumed_after.copy_(updated(state.consumed_tokens, consumed))
        if self.history is not None:
            self.history_after.copy_(updated(self.history, history))
        self._records.copy_(torch.where(owned[:, None], record[source], self.idle_record))
        state.histogram.view(-1).scatter_add_(0, (safe[:, None] * state.vocab_size
            + ids.clamp(0, state.vocab_size - 1)).view(-1), (take & leader[:, None]).to(torch.int32).view(-1))
        commit(state.generated_tokens, generated)
        commit(state.pending_token, torch.where(emitted > 0, pending, pending_before))
        commit(counter, counters)
        commit(state.status, status)
        commit(state.error_code, error)
        if self.rng is not None:
            commit(self.rng.target_offset, rng_offset)
            commit(self.rng.generation, rng_generation)
        state.consumed_tokens.copy_(self.consumed_after)
        if self.history is not None:
            self.history.copy_(self.history_after)
        self.records.index_copy_(0, self.frame.long(), self._records.unsqueeze(0))
        self.logprobs()
        return self
