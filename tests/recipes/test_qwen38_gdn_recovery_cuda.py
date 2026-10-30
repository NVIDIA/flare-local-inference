"""Recorded Qwen GDN verification and commit against b12x's per-token checkpointing decode."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

KEY_HEADS, VALUE_HEADS = 16, 48


def _reference(tensors, columns):
    """b12x decode: per-token checkpoints in columns, gated output norm applied in place."""
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.sequence import gdn_decode

    device = tensors["mixed_qkv"].device
    caps = gdn_decode.Caps(device=device, max_tokens=tensors["mixed_qkv"].shape[0],
                           max_seqs=tensors["query_start_loc"].numel() - 1,
                           max_state_slots=tensors["recurrent_state"].shape[0], key_heads=KEY_HEADS,
                           value_heads=VALUE_HEADS, state_index_columns=columns, state_dtype=torch.float32,
                           gate_activation="sigmoid", null_state_index=0)
    plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **tensors))

    original_state, original_output = tensors["recurrent_state"].clone(), tensors["output"].clone()

    def restore():
        # Preparation primes the call; the comparison starts from the original state.
        tensors["recurrent_state"].copy_(original_state)
        tensors["output"].copy_(original_output)

    def prepare(state):
        if "scratch" not in tensors:
            (spec,) = state.layout.scratch_specs()
            tensors["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        bound = state.bind(**tensors)
        return PreparedCall(run=lambda: state.run(bound), restore=restore)

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name="gdn", prepare_call=prepare),))
    restore()
    gdn_decode.run(gdn_decode.bind(plan, **tensors), eps=1e-6)


def _backing(initial, prefix=4096, padding=8192):
    # Recurrent state as a pool would hold it: one region of a wider byte row per slot.
    slots = initial.shape[0]
    region = initial[0].numel() * 4
    backing = torch.zeros((slots, region + padding), dtype=torch.uint8, device=initial.device)
    state = backing[:, prefix:prefix + region].view(torch.float32).view_as(initial)
    state.copy_(initial)
    return backing, state


def _case(lengths):
    """Random packed verification inputs and b12x's checkpointed decode of them."""
    from ignition.components.recurrent import DeltaMetadata

    generator = torch.Generator(device="cuda").manual_seed(sum(lengths))
    rows, tokens, columns = len(lengths), sum(lengths), 8
    slots = 1 + rows * columns
    mixed = (torch.randn((tokens, 2 * KEY_HEADS * 128 + VALUE_HEADS * 128), generator=generator,
                         device="cuda") * 0.5).bfloat16()
    ba = torch.randn((tokens, 2 * VALUE_HEADS), generator=generator, device="cuda").bfloat16()
    a, b = ba[:, VALUE_HEADS:], ba[:, :VALUE_HEADS]
    z = torch.randn((tokens, VALUE_HEADS, 128), generator=generator, device="cuda").bfloat16()
    a_log = torch.rand(VALUE_HEADS, generator=generator, device="cuda") * 0.5
    dt_bias = torch.randn(VALUE_HEADS, generator=generator, device="cuda") * 0.1
    weight = (1 + 0.2 * torch.randn(128, generator=generator, device="cuda")).bfloat16()
    initial = torch.randn((slots, VALUE_HEADS, 128, 128), generator=generator, device="cuda") * 0.05
    initial[0].zero_()
    metadata = DeltaMetadata.allocate(rows, torch.device("cuda"), verification_columns=columns)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    metadata.query_start_loc.copy_(starts)
    model_slots = torch.tensor([1 + row * columns if lengths[row] else 0 for row in range(rows)], device="cuda")
    metadata.state_indices.copy_(torch.where(model_slots[:, None] > 0,
        model_slots[:, None] + torch.arange(columns, device="cuda")[None, :], 0))
    metadata.num_seqs.fill_(rows)
    metadata.num_tokens.fill_(tokens)

    expected = torch.full((tokens, VALUE_HEADS, 128), float("nan"), device="cuda").bfloat16()
    reference_state = initial.clone()
    _reference(dict(mixed_qkv=mixed, a=a, b=b, A_log=a_log, dt_bias=dt_bias, recurrent_state=reference_state,
                    z=z, norm_weight=weight, query_start_loc=metadata.query_start_loc,
                    num_accepted_tokens=metadata.accepted_tokens, state_indices=metadata.state_indices,
                    num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens, output=expected), columns)
    return dict(mixed=mixed, a=a, b=b, z=z, a_log=a_log, dt_bias=dt_bias, weight=weight, initial=initial,
                metadata=metadata, model_slots=model_slots, expected=expected, reference_state=reference_state,
                tokens=tokens, slots=slots)


def _recurrence(case, state, records):
    from ignition.components.gated_norm import PreparedGatedRMSNorm
    from ignition.components.gdn_recovery import PreparedRecoveredRecurrence

    tokens = case["tokens"]
    recurrence = torch.full((tokens, VALUE_HEADS * 128), float("nan"), device="cuda").bfloat16()
    gated = torch.full_like(recurrence, float("nan"))
    PreparedRecoveredRecurrence(mixed_qkv=case["mixed"], a=case["a"], b=case["b"], A_log=case["a_log"],
                                dt_bias=case["dt_bias"], state=state, metadata=case["metadata"], output=recurrence,
                                records=records, key_heads=KEY_HEADS, value_heads=VALUE_HEADS)()
    PreparedGatedRMSNorm(recurrence.view(tokens, VALUE_HEADS, 128), case["z"], case["weight"],
                         gated.view(tokens, VALUE_HEADS, 128), case["metadata"].num_tokens, eps=1e-6,
                         round_normalized=True)()
    return gated.view_as(case["expected"])


@pytest.mark.parametrize(("lengths", "accepted"), [
    ((4,), (3,)), ((6, 6, 0, 6), (1, 6, 0, 2)), ((5, 2), (5, 1)),
    *[((8,), (accepted,)) for accepted in range(1, 9)],
])
@pytest.mark.parametrize(("prefix", "padding", "alignment"), [(4096, 8192, 4), (4100, 8192, 1), (4096, 8196, 1)])
def test_recorded_verification_and_commit_match_checkpointing_decode(lengths, accepted, prefix, padding, alignment):
    from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit

    case = _case(lengths)
    initial, metadata, model_slots = case["initial"], case["metadata"], case["model_slots"]
    expected, reference_state, slots = case["expected"], case["reference_state"], case["slots"]
    backing, state = _backing(initial, prefix, padding)
    stacked = GDNLayerRecords.allocate((3,), slots, torch.device("cuda"), key_heads=KEY_HEADS, value_heads=VALUE_HEADS)
    gated = _recurrence(case, state, stacked[3])
    # Verification leaves committed state untouched.
    assert torch.equal(state, initial)
    counts = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    commit = PreparedGDNCommit(backing, {3: state}, stacked, metadata, counts, key_heads=KEY_HEADS,
                               value_heads=VALUE_HEADS)
    assert commit.options["STATE_ALIGNMENT"] == alignment
    commit()

    torch.testing.assert_close(gated, expected, rtol=2**-6, atol=2e-2)
    assert (gated == expected).float().mean() > 0.99
    for row, (length, count) in enumerate(zip(lengths, accepted)):
        slot = int(model_slots[row])
        if not length:
            continue
        committed = reference_state[slot + count - 1]
        torch.testing.assert_close(state[slot], committed, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lengths", [(1,), (1, 0, 1, 1)])
def test_single_token_rounds_advance_committed_state_in_place(lengths):
    """Without drafts the token is accepted when emitted, so verification commits it directly."""
    case = _case(lengths)
    initial, model_slots, reference_state = case["initial"], case["model_slots"], case["reference_state"]
    _, state = _backing(initial)
    gated = _recurrence(case, state, None)

    torch.testing.assert_close(gated, case["expected"], rtol=2**-6, atol=2e-2)
    assert (gated == case["expected"]).float().mean() > 0.99
    live = {int(slot) for slot in model_slots if slot}
    for slot in range(state.shape[0]):
        if slot in live:
            torch.testing.assert_close(state[slot], reference_state[slot], rtol=1e-5, atol=1e-6)
        else:
            assert torch.equal(state[slot], initial[slot])


def test_compilation_sees_every_recovery_store():
    """Forward compilation keeps a user kernel only for the operands its stores reach."""
    from torch._higher_order_ops.triton_kernel_wrap import get_mutated_tensors, kernel_side_table

    from ignition.components import gdn_recovery

    def mutated(kernel, names, options):
        constants = {name: value for name, value in options.items()}
        kwargs = {name: torch.empty(1, device="cuda") for name in names}
        return set(get_mutated_tensors(kernel_side_table.add_kernel(kernel),
                                       kernel_side_table.add_constant_args(constants), kwargs, {}))

    geometry = {"KEY_HEADS": KEY_HEADS, "VALUE_HEADS": VALUE_HEADS, "WINDOW": 8, "ROWS": 16}
    for in_place, stores in ((False, {"Output", "Delta", "Keys", "Decay"}), (True, {"Output", "State"})):
        verify = mutated(gdn_recovery._verify, ("Mixed", "A", "B", "ALog", "DtBias", "State", "Starts", "Slots",
            "NumSeqs", "Output", "Delta", "Keys", "Decay"), {**geometry, "SCALE": 0.1, "mixed_stride": 1,
            "a_stride": 1, "b_stride": 1, "state_stride": 1, "slot_stride": 1, "IN_PLACE": in_place})
        assert verify == stores
    commit = mutated(gdn_recovery._commit, ("State", "LayerOffsets", "Delta", "KeyRecords", "Decay", "Slots",
        "Counts", "NumSeqs"), {**geometry, "ROWS_CAPACITY": 3, "state_stride": 1, "slot_stride": 1, "STATE_ALIGNMENT": 1})
    # The analysis may conservatively report every operand; it must include the state backing.
    assert "State" in commit
