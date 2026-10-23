"""Plan sizing, GPU selection, and capacity errors from declared costs and injected free memory."""

import json
from dataclasses import replace
import re
from types import MappingProxyType

import pytest

from ignition import planning
from ignition.cli import display
from ignition.errors import CapacityError, UsageError
from ignition.recipes.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.recipes import recipes

GIB, MIB = 1 << 30, 1 << 20


DRAFTING = Speculation(draft_tokens=3)


def recipe(name, device_gib, ranks, *, speculation=None, mapped_gib=0, draft_gib=0):
    checkpoint = Checkpoint(name, f"lab/{name}", "ab12cd34" + "0" * 32, "DemoForCausalLM",
                            stored_bytes=(device_gib + mapped_gib) * GIB, device_bytes=device_gib * GIB,
                            mapped_host_bytes=mapped_gib * GIB, draft_bytes=draft_gib * GIB)
    speculating = speculation is not None
    return Recipe(name=name, served_model_name=name, checkpoints=(checkpoint,),
                  context=262144, page_tokens=64, concurrency=4,
                  sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
                  max_concurrency=64, max_draft_tokens=7 if speculating else 0, prefill_chunk=4096, prepare=None,
                  usage=None, mapped_host_name="lookup table", speculation=speculation,
                  ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB)
                                          for size in ranks}),
                  plain_ranks=MappingProxyType({size: RankCosts(80 * MIB // size, 3 * MIB // (2 * size), 3 * GIB)
                                                for size in ranks} if speculating else {}))


# While drafting, one rank holds 60 GiB of weights, 2 GiB of them the draft model, 4 GiB of graphs,
# 100 MiB per state row, and 2 MiB per page, a quarter of it the draft model's. At concurrency 4 it
# keeps 17 execution rows, and its minimum of 512 pages costs 1 GiB. Without drafting it holds 3 GiB of
# graphs, 80 MiB per state row, and 1.5 MiB per page.
SOLO = recipe("solo", 60, (1, 2, 4), speculation=DRAFTING, mapped_gib=4, draft_gib=2)
PAIR = recipe("pair", 160, (2, 4))


@pytest.fixture(autouse=True)
def headroom(monkeypatch):
    monkeypatch.setattr(planning, "HEADROOM_BYTES", 4 * GIB)


def gpu(index, free_gib, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, int(free_gib * GIB),
                        capability)


def spark(index, free_gib):
    return planning.Gpu(index, "NVIDIA GB10", f"GPU-{index:04d}", 121 * GIB, int(free_gib * GIB), (12, 1))


DEFAULT = object()


def derive(model=SOLO, gpus=(gpu(0, 95),), *, selected=None, concurrency=4, reserve=0, prefill_chunk=None,
           speculation=DEFAULT):
    return planning.plan(model, model.checkpoints[0], tuple(gpus), selected=selected, concurrency=concurrency,
                         reserve_bytes=int(reserve * GIB), prefill_chunk=prefill_chunk,
                         speculation=model.speculation if speculation is DEFAULT else speculation)


def test_pages_take_all_remaining_supply():
    limited = derive()
    assert limited.fits and limited.cache_states == limited.cache_target == 32
    assert limited.supply_bytes == 91 * GIB and limited.live_bytes == 17 * 100 * MIB
    # 91 GiB of supply less weights, graphs, 17 execution rows leaves 12,974 pages.
    assert limited.pages == (91 * GIB - 64 * GIB - 17 * 100 * MIB) // (2 * MIB) == 12974
    assert limited.demand_pages == 16384 and limited.minimum_pages == 512
    assert limited.context == 262144 and limited.guaranteed_tokens == 12974 // 4 * 64
    assert limited.snapshots == 32

    assert derive(gpus=(gpu(0, 95), gpu(1, 95))).world_size == 1

    # One request needs 4,096 pages at full context; the rest of the supply holds cached prefixes.
    whole = derive(concurrency=1)
    assert whole.demand_pages == 4096 and whole.pages == 13574 and whole.guaranteed_tokens == 262144
    assert whole.supply_bytes - whole.page_bytes - whole.fixed_bytes < 2 * MIB


@pytest.mark.parametrize("draft,markov", [(None, None), ("nvfp4", None), (None, "nvfp4"), ("nvfp4", "nvfp4")])
def test_proposal_precision_is_independent_plan_metadata(draft, markov, monkeypatch):
    from types import SimpleNamespace
    from ignition.cli import display
    from ignition.preparation import rebuild_manifest

    # Source identity needs the native export extension; the record's precision keys do not.
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")

    model = replace(SOLO, draft_head_quantizations=("nvfp4",), markov_head_quantizations=("nvfp4",))
    spec = replace(DRAFTING, draft_head_quantization=draft, markov_head_quantization=markov)
    result = derive(model=model, speculation=spec)
    assert result.to_dict()["speculation"]["draft_head_quantization"] == draft
    assert result.to_dict()["speculation"]["markov_head_quantization"] == markov
    summary = display.speculation(spec)
    assert ("draft head NVFP4" in summary) == bool(draft)
    assert ("markov head NVFP4" in summary) == bool(markov)
    # The rebuild record must distinguish the precision, or a rebuild would silently
    # produce checkpoint-default heads; an unset field stays absent so BF16 records
    # remain byte-identical. Encode through the schema the export actually uses, so a
    # key without a protobuf field fails here rather than at package export.
    from google.protobuf import json_format
    from ignition.preparation import runtime_package_pb2 as proto

    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), result, compiled_forwards=True)
    encoded = proto.RebuildManifest()
    json_format.ParseDict(manifest, encoded)
    assert encoded.speculation.HasField("draft_head_quantization") == bool(draft)
    assert encoded.speculation.HasField("markov_head_quantization") == bool(markov)
    assert encoded.speculation.draft_head_quantization == (draft or "")
    assert encoded.speculation.markov_head_quantization == (markov or "")
    if draft or markov:
        with pytest.raises(UsageError, match="head-quantization"):
            derive(speculation=spec)
        with pytest.raises(ValueError, match="positive learned"):
            replace(spec, draft_tokens=0)


def test_legacy_rebuild_speculation_without_head_precision_still_loads():
    from ignition.preparation import runtime_package_pb2 as proto

    # A record written before fields 8 and 9 existed must still load and report the
    # checkpoint-default precision, so packages prepared earlier keep their meaning.
    legacy = proto.RebuildSpeculation.FromString(b"\x08\x05\x10\x01\x1a\x03mtp\x30\x01")
    assert legacy.draft_tokens == 5 and legacy.packed_verification
    assert not legacy.HasField("draft_head_quantization")
    assert not legacy.HasField("markov_head_quantization")
    assert legacy.draft_head_quantization == "" and legacy.markov_head_quantization == ""


def test_mla_arithmetic_is_explicit_plan_metadata_without_changing_capacity():
    model = replace(SOLO, mla_compute_modes=("auto", "bf16"))
    options = dict(concurrency=4, reserve_bytes=0, speculation=model.speculation)
    auto = planning.fit(model, model.checkpoints[0], (gpu(0, 95),), **options)
    bf16 = planning.fit(model, model.checkpoints[0], (gpu(0, 95),), mla_compute_mode="bf16", **options)
    assert auto.mla_compute_mode == "auto" and bf16.mla_compute_mode == "bf16"
    assert bf16.to_dict()["mla_compute_mode"] == "bf16"
    assert (auto.pages, auto.cache_states, auto.fixed_bytes) == (bf16.pages, bf16.cache_states, bf16.fixed_bytes)
    with pytest.raises(UsageError, match="mla-compute-mode"):
        planning.fit(SOLO, SOLO.checkpoints[0], (gpu(0, 95),), mla_compute_mode="bf16", **options)


def test_image_concurrency_defaults_and_explicit_capacity():
    vision = replace(SOLO, max_images=4)
    settings = dict(concurrency=32, reserve_bytes=0, speculation=None, selected=None)
    def image_plan(value=None, model=vision, **overrides):
        return planning.plan(model, model.checkpoints[0], (gpu(0, 95),),
                             **(settings | overrides), image_concurrency=value)
    assert image_plan().image_concurrency == 4
    assert image_plan(concurrency=2).image_concurrency == 2
    assert image_plan(1).image_concurrency == 1
    assert image_plan(32).to_dict()["image_concurrency"] == 32
    assert image_plan(model=SOLO).image_concurrency == 0
    for invalid in (0, -1, 33, True):
        with pytest.raises(UsageError, match="image-concurrency"):
            image_plan(invalid)
    with pytest.raises(UsageError, match="vision model"):
        image_plan(1, model=SOLO)


def test_expert_scale_compression_changes_weight_budget_and_preserves_native_default():
    import pickle

    checkpoint = replace(SOLO.checkpoints[0], scale_compressed_device_bytes={"csf": 56 * GIB})
    assert pickle.loads(pickle.dumps(checkpoint)) == checkpoint
    settings = dict(concurrency=4, reserve_bytes=0, speculation=DRAFTING)
    native = planning.fit(SOLO, checkpoint, (gpu(0, 95),), **settings)
    compressed = planning.fit(SOLO, checkpoint, (gpu(0, 95),), expert_scale_compression="csf", **settings)
    assert native.weights_bytes == 60 * GIB and native.expert_scale_compression is None
    assert compressed.weights_bytes == 56 * GIB and compressed.pages > native.pages
    assert compressed.to_dict()["expert_scale_compression"] == "csf"
    plain = planning.fit(SOLO, checkpoint, (gpu(0, 95),), concurrency=4, reserve_bytes=0,
                          speculation=None, expert_scale_compression="csf")
    assert plain.weights_bytes == 54 * GIB
    with pytest.raises(UsageError, match="does not prepare this scale encoding"):
        planning.fit(SOLO, SOLO.checkpoints[0], (gpu(0, 95),), expert_scale_compression="csf", **settings)


def test_live_state_counts_only_null_live_and_speculative_rows():
    assert SOLO.live_rows(4, DRAFTING) == 1 + 4 * 4
    assert SOLO.live_rows(5, DRAFTING) == 1 + 8 * 4
    assert SOLO.live_rows(4, None) == PAIR.live_rows(4, None) == 1 + 4


def test_snapshot_slots_share_the_page_budget():
    assert derive(concurrency=1).cache_states == 8
    assert derive(concurrency=2).cache_states == 16
    capped = derive(reserve=12)
    assert capped.cache_states == 32
    assert capped.pages == (capped.supply_bytes - capped.fixed_bytes) // capped.costs.page_bytes
    devices = (gpu(0, 95),)
    common = dict(concurrency=4, reserve_bytes=0, speculation=DRAFTING)
    small = planning.fit(SOLO, SOLO.checkpoints[0], devices, cache_states=1, **common)
    large = planning.fit(SOLO, SOLO.checkpoints[0], devices, cache_states=64, **common)
    assert small.pages == large.pages and small.live_rows == large.live_rows
    assert (small.snapshots, large.snapshots) == (1, 64)


def test_speculation_settings_select_weights_rows_and_pages():
    drafting = derive()
    assert drafting.speculation == DRAFTING and drafting.live_rows == 17
    assert (drafting.weights_bytes, drafting.graph_bytes, drafting.costs.page_bytes) == (60 * GIB, 4 * GIB, 2 * MIB)
    # Without drafting the draft model is not loaded, rows shrink, and each request keeps no verification rows.
    plain = derive(speculation=None)
    assert (plain.weights_bytes, plain.graph_bytes, plain.live_rows) == (58 * GIB, 3 * GIB, 5)
    assert plain.live_bytes == 5 * 80 * MIB and plain.cache_states == 32
    assert plain.pages == (91 * GIB - 61 * GIB - 5 * 80 * MIB) * 2 // (3 * MIB) == 20213
    assert plain.to_dict()["speculation"] is None
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)), speculation=None)
    assert split.world_size == 2 and split.weights_bytes == 29 * GIB and split.costs.state_row_bytes == 40 * MIB


def test_speculation_settings_are_validated_and_described():
    assert DRAFTING.widths == (1, 2, 3, 4) and DRAFTING.fixed_width is None
    fixed = Speculation(2, adaptive=False)
    assert fixed.widths == (1, 2, 3) and fixed.fixed_width == 3
    assert Speculation(0).widths == (1,) and Speculation(0, adaptive=False).fixed_width == 1
    record = derive(speculation=Speculation(5)).to_dict()["speculation"]
    assert record == {"draft_tokens": 5, "adaptive": True,
                      "packed_verification": False, "draft_prefix_selection": "cpu",
                      "draft_head_quantization": None, "markov_head_quantization": None}
    assert derive(speculation=Speculation(5, packed_verification=True)).to_dict()["speculation"]["packed_verification"]
    assert display.speculation(None) == "off"
    assert display.speculation(DRAFTING) == "up to 3 draft tokens per round · MTP module"
    assert display.speculation(fixed) == "exactly 2 draft tokens per round · MTP module"
    assert display.speculation(None, brief=True) == "speculation off"
    assert display.speculation(DRAFTING, brief=True) == "up to 3 draft tokens"
    assert display.speculation(fixed, brief=True) == "exactly 2 draft tokens"


def test_headroom_grows_with_prefill_chunk_tokens_past_4096():
    # The fixture sets four GiB of headroom; each 4096 chunk tokens past 4096 add half a GiB.
    assert derive().headroom_bytes == derive(prefill_chunk=1024).headroom_bytes == 4 * GIB
    assert derive(prefill_chunk=8192).headroom_bytes == 4 * GIB + GIB // 2
    assert derive(prefill_chunk=8192).supply_bytes == derive().supply_bytes - GIB // 2
    assert derive(prefill_chunk=8192).to_dict()["headroom_bytes"] == 4 * GIB + GIB // 2


def test_prefill_chunk_defaults_to_the_recipe_and_accepts_an_override():
    assert derive().prefill_chunk == 4096
    assert derive(prefill_chunk=1024).prefill_chunk == 1024
    # A whole-context chunk's activations need about 32 GiB more headroom.
    assert derive(gpus=(gpu(0, 140),), prefill_chunk=262144).prefill_chunk == 262144
    for chunk in (0, 100, 262208):
        with pytest.raises(UsageError, match="multiple of 64 from 64 through 262,144"):
            derive(prefill_chunk=chunk)
    # 16 GiB of reserve and the chunk's 11.5 GiB headroom leave pages for 60,288 tokens, the longest
    # sequence this plan allows.
    with pytest.raises(UsageError, match="65,536 exceeds the 60,288-token context"):
        derive(reserve=16, prefill_chunk=65536)


@pytest.mark.parametrize("alias,world_size,packed", [
    (checkpoint.alias, 8 if recipe.name == "mimo_v2" else recipe.tensor_parallel[0],
     recipe.speculation.packed_verification)
    for recipe in recipes() if recipe.speculation is not None for checkpoint in recipe.checkpoints
])
@pytest.mark.parametrize("flags,configuration_text,expected", [
    ((), "", None),
    ((), '[defaults]\nmax_concurrency=16\nspeculative_token_capacity=96\n', (16, 4096, 96)),
    (("--max-concurrency", "8", "--prefill-chunk-size", "1024", "--speculative-token-capacity", "192"),
     '[models."{alias}"]\nmax_concurrency=16\nspeculative_token_capacity=96\n', (8, 1024, 192)),
])
def test_cli_plan_resolves_recipe_configuration_and_flag_defaults(tmp_path, alias, world_size, packed,
                                                                 flags, configuration_text, expected):
    from ignition.cli import _parser, preparation
    from ignition.cli.config import Configuration
    from ignition.recipes import resolve

    path = tmp_path / "flare.toml"
    path.write_text(configuration_text.format(alias=alias))
    configuration = Configuration.load(path)
    model = resolve(alias)
    if expected is None:
        expected = (4, 4096, 6) if model.recipe.name == "gemma4" else (32, 4096, 128)
    args = _parser().parse_args(["export", model.checkpoint.alias, *flags])
    selected = preparation.speculation(args, model, configuration)
    concurrency = configuration.max_concurrency(model.checkpoint.alias, model.recipe.concurrency,
                                                args.max_concurrency)
    devices = tuple(gpu(index, 95) for index in range(world_size))
    if model.recipe.name != "mimo_v2":
        devices = tuple(replace(device, total_bytes=256 * GIB, free_bytes=255 * GIB) for device in devices)
    host = preparation.Host(devices, None)
    plan = preparation.derive(configuration, model, host, gpus=args.gpus, concurrency=concurrency,
                              speculation=selected, prefill_chunk=args.prefill_chunk_size)
    assert plan.world_size == world_size
    assert (plan.concurrency, plan.prefill_chunk, plan.speculative_token_capacity) == expected
    assert plan.speculation.draft_tokens == model.recipe.speculation.draft_tokens and plan.speculation.adaptive
    assert plan.speculation.packed_verification is packed
    assert plan.speculation.draft_prefix_selection == ("gpu" if packed else "cpu")
    record = plan.to_dict()
    assert record["speculative_token_capacity"] == expected[2]
    assert record["speculation"]["packed_verification"] is packed
    assert record["speculation"]["draft_prefix_selection"] == ("gpu" if packed else "cpu")


def test_context_parallel_pages_hold_every_rank_share_unless_disabled():
    sharded = replace(PAIR, ranks=MappingProxyType({size: replace(costs, page_tokens=64 * size)
                                                    for size, costs in PAIR.ranks.items()}))
    pair = (gpu(0, 95), gpu(1, 95))
    shared = planning.plan(sharded, sharded.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                           speculation=None)
    whole = planning.plan(sharded, sharded.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                          speculation=None, context_parallel=False)
    assert (shared.context_parallel, shared.page_tokens) == (True, 128)
    assert (whole.context_parallel, whole.page_tokens) == (False, 64)
    assert shared.pages == whole.pages and shared.tokens == 2 * whole.tokens
    assert derive().context_parallel is False
    # Recipes that keep each page whole plan the same either way.
    plain = planning.plan(PAIR, PAIR.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                          speculation=None, context_parallel=False)
    assert plain == planning.plan(PAIR, PAIR.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                                  speculation=None)


def test_context_parallelism_follows_the_flag():
    from types import SimpleNamespace

    from ignition.cli.preparation import context_parallel

    sharded = replace(PAIR, ranks=MappingProxyType({size: replace(costs, page_tokens=64 * size)
                                                    for size, costs in PAIR.ranks.items()}))
    flag = SimpleNamespace(no_context_parallel=True)
    default = SimpleNamespace(no_context_parallel=False)
    model = SimpleNamespace(recipe=sharded, checkpoint=sharded.checkpoints[0])
    assert context_parallel(default, model) and not context_parallel(flag, model)
    whole = SimpleNamespace(recipe=PAIR, checkpoint=PAIR.checkpoints[0])
    assert context_parallel(default, whole)
    with pytest.raises(UsageError, match="does not shard attention context"):
        context_parallel(flag, whole)


def test_serialized_branches_are_planned_and_recorded_without_changing_memory():
    overlapping = replace(PAIR, stream_overlap=True)
    pair = (gpu(0, 95), gpu(1, 95))

    def planned(model, **settings):
        return planning.plan(model, model.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                             speculation=None, **settings)

    overlapped, serial = planned(overlapping), planned(overlapping, stream_overlap=False)
    assert overlapped.stream_overlap and not serial.stream_overlap
    assert overlapped.to_dict()["stream_overlap"] is True
    assert json.loads(json.dumps(serial.to_dict()))["stream_overlap"] is False
    # Serializing branches changes graph scheduling only; both plans size the same pages and state.
    assert serial == replace(overlapped, stream_overlap=False)
    # A recipe that runs no branches concurrently plans without overlap either way.
    assert not planned(PAIR).stream_overlap and planned(PAIR) == planned(PAIR, stream_overlap=False)


def test_stream_overlap_flag_applies_only_to_recipes_that_overlap_branches():
    from types import SimpleNamespace

    from ignition.cli.preparation import stream_overlap

    flag, default = SimpleNamespace(no_stream_overlap=True), SimpleNamespace(no_stream_overlap=False)
    overlapping = replace(PAIR, stream_overlap=True)
    model = SimpleNamespace(recipe=overlapping, checkpoint=overlapping.checkpoints[0])
    assert stream_overlap(default, model) and not stream_overlap(flag, model)
    serial = SimpleNamespace(recipe=PAIR, checkpoint=PAIR.checkpoints[0])
    assert stream_overlap(default, serial)
    with pytest.raises(UsageError, match="--no-stream-overlap: pair runs no branches concurrently"):
        stream_overlap(flag, serial)


def test_language_model_only_plans_no_images_for_recipes_that_declare_it():
    pair = (gpu(0, 95), gpu(1, 95))
    seeing = replace(PAIR, max_images=4, image_bytes=32 * MIB, language_model_only=True)

    def planned(model, **settings):
        return planning.plan(model, model.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                             speculation=None, **settings)

    text = planned(seeing, language_model_only=True)
    assert planned(seeing).max_images == 4 and text.max_images == 0
    # Image feature slots follow the vision encoder.
    assert planned(seeing).image_concurrency == 4 and text.image_concurrency == 0
    with pytest.raises(UsageError, match="--image-concurrency requires the vision encoder"):
        planned(seeing, language_model_only=True, image_concurrency=2)
    assert text.to_dict()["language_model_only"] is True and planned(seeing).to_dict()["language_model_only"] is False
    with pytest.raises(UsageError, match="--language-model-only: pair prepares no separable vision encoder"):
        planned(PAIR, language_model_only=True)


def test_first_fitting_gpus_and_smallest_supported_tensor_parallel_size():
    assert [g.index for g in derive(gpus=(gpu(0, 10), gpu(1, 95), gpu(2, 95))).gpus] == [1]
    assert [g.index for g in derive(gpus=(gpu(0, 95, (9, 0)), gpu(1, 95))).gpus] == [1]
    assert [g.index for g in derive(PAIR, (gpu(0, 95), gpu(1, 20), gpu(2, 95))).gpus] == [0, 2]
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)))
    assert split.world_size == 2 and split.weights_bytes == 30 * GIB
    assert derive(PAIR, tuple(gpu(index, 60) for index in range(4))).world_size == 4


def test_explicit_gpus_set_the_tensor_parallel_size():
    gpus = tuple(gpu(index, 95) for index in range(4))
    assert derive(gpus=gpus, selected=(2, 3)).world_size == 2
    with pytest.raises(CapacityError, match=r"pair needs 2 or 4 GPUs; 1 selected; use --gpus 0-1"):
        derive(PAIR, gpus, selected=(0,))
    with pytest.raises(CapacityError, match=r"solo needs 1 or 2 or 4 GPUs; 3 selected; use --gpus 0-3"):
        derive(gpus=gpus, selected=(0, 1, 2))
    with pytest.raises(UsageError, match="names a GPU this host does not have"):
        derive(gpus=gpus, selected=(4,))
    with pytest.raises(CapacityError, match="not an SM103, SM120, or SM121 device"):
        derive(gpus=(gpu(0, 95, (9, 0)),), selected=(0,))
    with pytest.raises(CapacityError, match="pair needs 2 or 4 SM103, SM120, or SM121 GPUs; this host has 1"):
        derive(PAIR, (gpu(0, 95),))


@pytest.mark.parametrize("limit", [0, 65])
def test_concurrency_outside_the_recipe_range_is_a_usage_error(limit):
    with pytest.raises(UsageError, match="must be 1 through 64"):
        derive(concurrency=limit)


def option_arguments(option, arguments):
    """Apply one printed option to the arguments of `derive`."""
    arguments = dict(arguments)
    if match := re.fullmatch(r'\[memory\] reserve = "(\d+)G?"', option):
        arguments["reserve"] = int(match[1])
    elif match := re.fullmatch(r"--max-concurrency (\d+)", option):
        arguments["concurrency"] = int(match[1])
    elif match := re.fullmatch(r"--gpus (\S+)", option):
        arguments["selected"] = planning.parse_gpus(match[1])
    elif option == "--no-mtp":
        arguments["speculation"] = None
    else:
        raise AssertionError(option)
    return arguments


@pytest.mark.parametrize("arguments, expected", [
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=25),
     ['[memory] reserve = "24G"', "--max-concurrency 2", "--no-mtp"]),
    (dict(gpus=(gpu(0, 95), gpu(1, 95)), selected=(0,), reserve=25),
     ['[memory] reserve = "24G"', "--max-concurrency 2", "--no-mtp", "--gpus 0-1"]),
    (dict(gpus=(gpu(0, 70.5),)), ["--max-concurrency 3", "--no-mtp"]),
    (dict(gpus=(gpu(0, 72),), concurrency=8), ["--max-concurrency 4", "--no-mtp"]),
    (dict(gpus=(gpu(0, 80), gpu(1, 80)), reserve=41),
     ['[memory] reserve = "40G"', "--max-concurrency 2", "--no-mtp"]),
    # Without speculation the plan has no drafting to give up.
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=30, speculation=None), ['[memory] reserve = "28G"']),
])
def test_every_capacity_error_option_fits_on_its_own(arguments, expected):
    with pytest.raises(CapacityError) as error:
        derive(**arguments)
    message = str(error.value)
    assert message.startswith("needs ") and "minimum pages" in message and "\nhas " in message
    options = re.search(r"Options: (.*)\.$", message)[1].split(", or ")
    assert options == expected
    for option in options:
        assert derive(**option_arguments(option, arguments)).fits, option


def test_capacity_error_states_requirement_and_supply():
    with pytest.raises(CapacityError) as error:
        derive(gpus=(gpu(0, 95),), selected=(0,), reserve=25)
    assert str(error.value).splitlines()[:3] == [
        "needs 60.0 GiB weights + 4.0 GiB graphs + 1.7 GiB live state",
        "+ 1.0 GiB minimum pages (4 × 8,192 tokens) = 66.7 GiB;",
        "has 95.0 GiB free − 4.0 GiB headroom − 25.0 GiB reserve = 66.0 GiB.",
    ]


def test_capacity_error_without_a_working_setting_says_so():
    with pytest.raises(CapacityError, match="No setting makes solo fit on this host"):
        derive(gpus=(gpu(0, 40),))
    with pytest.raises(CapacityError, match="No setting makes solo fit on GPU 0"):
        derive(gpus=(gpu(0, 40),), selected=(0,))


def test_automatic_selection_explains_the_closest_tensor_parallel_size():
    with pytest.raises(CapacityError) as error:
        derive(gpus=(gpu(0, 80), gpu(1, 80)), reserve=41)
    assert "= 35.3 GiB per GPU;" in str(error.value)


def test_estimate_record():
    plan = derive(reserve=2)
    record = json.loads(json.dumps(plan.to_dict()))
    assert record["pages"] == plan.pages == 11950 and record["snapshots"] == 32
    assert record["reserve_bytes"] == 2 * GIB and "measured" not in record


def test_gpu_lists_and_restrictions():
    assert planning.parse_gpus("0") == (0,)
    assert planning.parse_gpus("0,2") == (0, 2)
    assert planning.parse_gpus("0-3") == (0, 1, 2, 3)
    assert planning.format_gpus((0, 1, 2)) == "0-2" and planning.format_gpus((0, 2)) == "0,2"
    for text in ("", "a", "0-", "1,1", "3-1", "0,,1"):
        with pytest.raises(UsageError):
            planning.parse_gpus(text)
    gpus = tuple(gpu(index, 95) for index in range(3))
    restricted = planning.restrict_gpus(gpus, "2,GPU-0000")
    assert [(g.index, g.uuid) for g in restricted] == [(0, "GPU-0002"), (1, "GPU-0000")]
    assert planning.restrict_gpus(gpus, "") == ()
    for text in ("7", "GPU-", "1,1", "GPU-9"):
        with pytest.raises(UsageError):
            planning.restrict_gpus(gpus, text)


def test_unified_memory_gpus_plan_from_their_reported_free_memory():
    # The PLE table reaches the GPU through compact row delivery, so no mapped table is subtracted.
    discrete, unified = derive(gpus=(gpu(0, 95),)), derive(gpus=(spark(0, 95),))
    assert discrete.free_bytes == unified.free_bytes == 95 * GIB


def test_tuning_devices_keep_serving_capacity_and_tensor_parallel_size():
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    devices = tuple(gpu(index, 95) for index in range(6))
    host = SimpleNamespace(gpus=devices)
    for model, count in ((SOLO, 1), (PAIR, 2)):
        plan = derive(model, devices, selected=tuple(range(count)))
        assert tuning_devices(plan, host, None) is plan
        assert tuning_devices(plan, host, ','.join(str(i) for i in range(count))) == plan
        expanded = tuning_devices(plan, host, '0-5')
        assert expanded.world_size == count
        assert expanded.tuning_gpus == devices
        assert replace(expanded, tuning_gpus=()) == plan
        assert expanded.to_dict()['tuning_gpus'] == list(range(6))


@pytest.mark.parametrize('selection,message', [
    ('1,0', 'start with the serving GPUs'),
    ('0-2', 'complete TP replicas'),
    ('0,1,6,7', 'outside the visible device list'),
])
def test_tuning_devices_reject_invalid_replica_groups(selection, message):
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    devices = tuple(gpu(index, 95) for index in range(4))
    with pytest.raises(UsageError, match=message):
        tuning_devices(derive(PAIR, devices), SimpleNamespace(gpus=devices), selection)


@pytest.mark.parametrize('helper,message', [
    (gpu(1, 95, (12, 1)), 'model and architecture'),
    (replace(gpu(1, 95), name='another GPU'), 'model and architecture'),
    (gpu(1, 68), 'needs at least'),
])
def test_tuning_devices_require_matching_hardware_and_memory(helper, message):
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    plan = derive(reserve=4)
    with pytest.raises(UsageError, match=message):
        tuning_devices(plan, SimpleNamespace(gpus=(plan.gpus[0], helper)), '0,1')


@pytest.mark.parametrize('model,workers,selection,message', [
    (PAIR, 1, None, 'local serving rank'),
    (SOLO, 1, '0,1', 'local tuning rank'),
])
def test_compiler_budget_rejects_more_local_ranks_than_workers(model, workers, selection, message):
    from types import SimpleNamespace
    from ignition.cli.preparation import tuning_devices

    devices = (gpu(0, 95), gpu(1, 95))
    plan = replace(derive(model, devices), compile_workers=workers)
    with pytest.raises(UsageError, match=message):
        tuning_devices(plan, SimpleNamespace(gpus=devices), selection)


def test_recipe_b12x_settings_name_b12x_variables():
    from dataclasses import replace

    from ignition.recipes.kimi3.recipe import RECIPE

    with pytest.raises(ValueError, match="B12X_"):
        replace(RECIPE, b12x_environment={"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="B12X_"):
        replace(RECIPE, b12x_environment={"B12X_PCIE_DCP_THREADS": 512})
