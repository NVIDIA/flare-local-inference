"""Copy proposal validation, conditional grammar, and captured-round composition."""

from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import RecordOnlyLogprobs, completion_outputs

from ignition.components.sampling import ERROR, RUNNING
from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (
    INVALID_SPECULATION,
    PreparedCopyProposals,
    PreparedSpeculativeDistributions,
)


def copy_fixture(width=4, batch=2, device="cpu", vocab=64):
    state = DeviceRequestState.allocate(device, vocab_size=vocab)
    state.admit(3, RequestAdmission(13, 2**64 - 1, temperature=1, max_new_tokens=100,
                eos_token_ids=()), [1, 2], consumed_tokens=2)
    state.pending_token[3] = 3
    slots = torch.tensor([3, 1][:batch], dtype=torch.int32, device=device)
    epochs = torch.tensor([13, 0][:batch], dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    args = dict(
        copy_ids=torch.arange(4, 11, device=device).expand(4, -1).clone(),
        copy_count=torch.full((4,), 7, dtype=torch.int32, device=device),
        copy_expected=torch.full((4,), 2, dtype=torch.int64, device=device),
        copy_generation=torch.tensor([0, 0, 0, 13], dtype=torch.int64, device=device),
        grammar_mask_count=torch.zeros(4, dtype=torch.int32, device=device),
        draft_ids=torch.zeros((batch, width - 1), dtype=torch.int64, device=device),
        draft_probabilities=torch.zeros((batch, width - 1, vocab), dtype=torch.float32, device=device),
        draft_valid=torch.zeros((batch, width - 1), dtype=torch.bool, device=device),
        copy_mask=torch.zeros(batch, dtype=torch.int64, device=device),
        previous_accepted=torch.full((4,), 8, dtype=torch.int32, device=device),
        accepted_generation=torch.zeros(4, dtype=torch.int64, device=device))
    op = PreparedCopyProposals(state, slots, epochs, live, **args)
    return state, slots, epochs, live, args, op


@pytest.mark.parametrize("width", range(1, 9))
def test_copy_one_hot_physical_slots_and_generation_reset(width):
    state, _, epochs, _, args, op = copy_fixture(width)
    original_rng = state.rng_counter.clone()
    op()
    assert op.active.tolist() == [True, False]
    assert args["draft_ids"][0].tolist() == list(range(4, 3 + width))
    assert args["draft_valid"][0].all() and not args["draft_valid"][1].any()
    assert args["draft_probabilities"].sum(-1).eq(1).all()
    assert args["draft_probabilities"][0].argmax(-1).tolist() == list(range(4, 3 + width))
    assert args["copy_mask"].tolist() == [(1 << (width - 1)) - 1, 0]
    assert args["previous_accepted"].tolist() == [8, 8, 8, 1]
    args["previous_accepted"][3] = 8
    op()
    assert args["previous_accepted"][3] == 8
    state.generation[3] = epochs[0] = args["copy_generation"][3] = 14
    op()
    assert args["previous_accepted"][3] == 1
    assert args["accepted_generation"].tolist() == [0, 0, 0, 14]
    assert torch.equal(state.rng_counter, original_rng)


@pytest.mark.parametrize("fault", ["generation", "position", "count", "negative_count", "id", "mask_short", "mask_large", "unconstrained_masks"])
def test_invalid_copy_suppresses_proposals_and_preserves_model_counters(fault):
    state, _, _, _, args, op = copy_fixture()
    if fault == "generation": args["copy_generation"][3] = 12
    elif fault == "position": args["copy_expected"][3] = 1
    elif fault == "count": args["copy_count"][3] = 8
    elif fault == "negative_count": args["copy_count"][3] = -1
    elif fault == "id": args["copy_ids"][3, 0] = 64
    elif fault in ("mask_short", "mask_large"):
        state.grammar_enabled[3] = 1
        args["grammar_mask_count"][3] = 3 if fault == "mask_short" else 9
    else: args["grammar_mask_count"][3] = 1
    before = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    op()
    assert state.status[3] == ERROR and state.error_code[3] == INVALID_SPECULATION
    assert not op.active.any() and not op.valid.any() and not op.copy_mask.any()
    assert args["previous_accepted"].eq(8).all() and not args["accepted_generation"].any()
    for name, value in before.items():
        if name not in ("status", "error_code"):
            assert torch.equal(getattr(state, name), value), name
    assert state.status[:3].eq(before["status"][:3]).all()


def test_stale_admission_epoch_is_ignored_and_width_one_accepts_no_drafts():
    state, _, epochs, _, args, op = copy_fixture(1)
    args["copy_count"].zero_()
    epochs[0] = 12
    op()
    assert state.status[3] == RUNNING and not op.active.any()
    epochs[0] = 13
    state.grammar_enabled[3] = 1
    args["grammar_mask_count"][3] = 1
    op()
    assert op.active.tolist() == [True, False]


def test_per_prefix_masks_use_physical_slot_and_include_bonus():
    state, slots, epochs, live, args, proposals = copy_fixture(4)
    state.grammar_enabled[3] = 1
    args["grammar_mask_count"][3] = 5  # A narrower graph may use a longer uploaded chain.
    masks = torch.zeros((4, 8, 2), dtype=torch.int32)
    for index, token in enumerate([4, 5, 6, 33, 7]):
        masks[3, index, token // 32] = 1 << (token % 32)
    logits = torch.ones((2, 4, 64))
    op = PreparedSpeculativeDistributions(logits, state, slots, epochs, live, args["draft_ids"],
        grammar_masks=masks, grammar_mask_count=args["grammar_mask_count"])
    proposals(); op.prepare_scores()
    assert op.valid.tolist() == [[True] * 4, [False] * 4]
    assert op.scores[0].argmax(-1).tolist() == [4, 5, 6, 33]
    assert torch.isfinite(op.scores[0]).sum(-1).tolist() == [1] * 4
    masks[3, 3].zero_()
    op.prepare_scores()
    assert op.valid[0].tolist() == [True, True, True, False]
    args["grammar_mask_count"][3] = 2
    op.prepare_scores()
    assert op.valid[0].tolist() == [True, True, False, False]


def test_copy_round_constructs_no_draft_module_and_commits_only_verified_prefix(monkeypatch):
    from ignition.recipes.qwen38 import speculative_model as module

    cuda_initialized = torch.cuda.is_initialized()
    batch, width, vocab = 2, 3, 64
    state, slots, epochs, live, _, _ = copy_fixture(width)
    regions = {"target_residual": torch.zeros((5, 10240), dtype=torch.bfloat16),
               "target_residual_position": torch.zeros((5, 3), dtype=torch.int64)}
    pool = SimpleNamespace(backing=torch.zeros(1, dtype=torch.uint8), region=regions.__getitem__,
                           rope_delta=torch.zeros((5, 1), dtype=torch.int64), gdn={})
    inputs = SimpleNamespace(row_slot_ids=slots, row_generations=epochs, live_rows=live,
        query_lengths=torch.full((batch,), width, dtype=torch.int32),
        request_state_rows=torch.arange(1, 5)[:, None].expand(4, 8).clone())
    scratch = SimpleNamespace(logits=torch.zeros((batch * width, vocab)),
                              sample_hidden=torch.zeros((batch * width, 2560), dtype=torch.bfloat16))
    observed = []

    class VerificationState:
        def __init__(self, inputs, requests, pool, candidates, previous, **kwargs):
            self.token_ids = torch.zeros(batch * width, dtype=torch.int64)
            self.candidates, self.previous = candidates, previous
        def _validate(self): pass
        def prepare_commit(self, count, before, residual, hidden):
            self._commit = (lambda: observed.append(("commit", count.clone())),)

    class Decoder:
        def __init__(self, weights, source, metadata, *args, **kwargs):
            self.scratch, self.state = source, metadata
        def __call__(self):
            observed.append(("decoder", state.status.clone(), self.state.previous.clone()))
            self.state.token_ids.copy_(self.state.candidates[:batch, :width].reshape(-1))

    def forbidden(*args, **kwargs):
        raise AssertionError("copy mode must not construct or execute MTP resources")

    monkeypatch.setattr(module, "PreparedVerificationState", VerificationState)
    monkeypatch.setattr(module, "PreparedDecoder", Decoder)
    monkeypatch.setattr(module, "DraftMetadata", forbidden)
    monkeypatch.setattr(module, "MTPScratch", SimpleNamespace(allocate=forbidden))

    class OneHotLibrary:
        # CPU wiring double; actual FlashInfer acceptance is covered on CUDA.
        top_k_mask_logits = staticmethod(lambda scores, k: scores)
        softmax = staticmethod(lambda scores: torch.softmax(scores, -1))
        top_p_renorm_probs = staticmethod(lambda probs, p, **kwargs: probs)

        @staticmethod
        def chain_speculative_sampling(draft_probs, draft_ids, target_probs, independent, emitted, **kwargs):
            assert ((target_probs == 0) | (target_probs == 1)).all()
            tokens = target_probs.argmax(-1)
            assert torch.equal(draft_ids.long(), tokens[:, :-1])
            independent.fill_(width - 1)
            emitted.fill_(width - 1)
            return tokens, independent, emitted

    def prepare_cpu(operator):
        operator._flashinfer = OneHotLibrary
        operator.library_scratch = ()
        operator.top_p_workspace = None
        for mask in getattr(operator, "probability_masks", ()):
            # This wiring double substitutes the prepared stores as well as
            # FlashInfer; CUDA tests exercise their compiled/native execution.
            def copy_probabilities(mask=mask):
                selected = mask.valid[:, None, None]
                if mask.limits is not None:
                    columns = torch.arange(mask.source.shape[1])[None, :, None]
                    selected = selected & (columns < mask.limits[:, None, None] - mask.limit_offset)
                source = mask.source if mask.snapshot is None else mask.snapshot
                mask.destination.copy_(torch.where(selected, source, mask.dummy))

            mask.kernel = copy_probabilities
            mask.prepared = True
        operator._prepared = True

    def normalize_one_hot(operator, probabilities, params):
        assert params[:, 2].eq(0).all()
        assert ((probabilities == 0) | (probabilities == 1)).all()
        assert probabilities.sum(-1).eq(1).all()
        operator.output.copy_(probabilities)
        return operator.output

    def filter_one_hot(scores, top_k, top_p, output, workspace):
        output.copy_(torch.softmax(scores, -1))

    monkeypatch.setattr("ignition.components.speculation.filter_probabilities_out", filter_one_hot)
    monkeypatch.setattr("ignition.components.probability_normalization.PreparedProbabilityNormalization.__call__",
                        normalize_one_hot)
    monkeypatch.setattr("ignition.components.speculation.verify_chain", OneHotLibrary.chain_speculative_sampling)
    monkeypatch.setattr(module.PreparedSpeculativeDistributions, "prepare", prepare_cpu)
    monkeypatch.setattr(module.PreparedStochasticVerification, "prepare", prepare_cpu)
    records = torch.full((2, 4, 24), -99, dtype=torch.int64)
    owner = module.SpeculativePreparation(weights=None, mtp_weights=None, pool=pool, requests=state,
        inputs=inputs, scratch=SimpleNamespace(rows=lambda *_: scratch), workspaces=None,
        rotary=None, cache={}, table=None, session=None, completion=completion_outputs(records),
        frame=torch.tensor([0], dtype=torch.int32), sequence=torch.tensor([19]),
        context=64, prefill_capacity=64, max_cache_pages=1, proposal_source="copy")
    assert owner.prefill(scratch, "test") is scratch and owner.resume(scratch) is scratch
    owner.rebind_pages_for_preparation({})
    assert "draft_page_tables" not in owner.buffers() and len(owner.buffers()) == 6
    assert owner.mtp_scratch is None and owner.draft_tables is None
    owner.copy_ids[3, :2] = torch.tensor([4, 5])
    owner.copy_count[3], owner.copy_generation[3], owner.copy_expected[3] = 2, 13, 2
    state.grammar_enabled[3] = 1
    owner.copy_grammar_mask_count[3] = 3
    owner.copy_grammar_mask[3, :3, 0] = torch.tensor([1 << 4, 1 << 5, 1 << 9], dtype=torch.int32)
    op = owner.round(batch, width, "copy")
    op.prepare()
    owner.previous_accepted.fill_(8)
    op()
    assert observed[0][0] == "decoder" and observed[0][2][3] == 1
    assert observed[1][1].tolist() == [3, 0]
    assert records[0, 3, 8:14].tolist() == [3, 2, 2, 3, -1, -1]
    assert records[0, 3, 16:24].tolist() == [4, 5, 9, -1, -1, -1, -1, -1]
    assert state.consumed_tokens[3] == 5 and state.pending_token[3] == 9
    assert not owner.draft_rng_offset.any() and not owner.draft_rng_generation.any()
    # Reusing stale consumed metadata fails before the decoder sees a live row.
    op()
    assert observed[-2][1][3] == ERROR and not observed[-1][1].any()
    assert state.consumed_tokens[3] == 5
    assert records[0, 3, 5:9].tolist() == [ERROR, INVALID_SPECULATION, 0, 0]
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.fixture(autouse=True)
def records_only_cpu_commit(monkeypatch):
    monkeypatch.setattr("ignition.components.speculation.PreparedLogprobs", RecordOnlyLogprobs)
