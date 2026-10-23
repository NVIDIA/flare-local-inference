"""Declared recipe costs against the layouts and checkpoints they describe."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.recipes import resolve
from ignition.recipes.declaration import Speculation

HUB = Path.home() / ".cache/huggingface/hub"
QWEN = resolve("qwen3.8-flash")
GLM = resolve("glm-5.3-flash")
COPY = Speculation(3, proposal_source="copy")


@pytest.mark.parametrize("alias,drafts,graphs,options", [
    ("qwen3.8-flash", 5, 34, {"mtp": True}),
    ("glm-5.3-flash", 5, 34, {"mtp": True}),
    ("deepseek-v4-flash", 5, 34, {}),
    ("step5-preview-nvfp4", 3, 24, {}),
    ("mimo-v2.6-pro-mopd", 7, 44, {}),
    ("kimi-k3-mxfp4-csf", 7, 44, {"draft_checkpoint": Path("dspark")}),
    ("nemotron-bolt-24gb", 7, 44, {"draft_checkpoint": Path("lilicorr")}),
])
def test_rectangular_recipe_defaults_prepare_bounded_speculative_coverage(alias, drafts, graphs, options):
    from importlib import import_module

    recipe = resolve(alias).recipe
    selected = recipe.speculation
    config = import_module(f"ignition.recipes.{recipe.name}.bootstrap").PreparationConfig(
        max_active=recipe.concurrency, speculative_widths=selected.widths,
        speculative_token_capacity=selected.token_capacity, **options)
    assert config.max_active == 32 and config.speculative_token_capacity == 128
    assert selected.draft_tokens == drafts and not selected.packed_verification
    shapes = set(config.speculative_shapes)
    assert len(shapes) == graphs
    assert {(batch, 1) for batch in (1, 2, 4, 8, 16, 32)} <= shapes
    assert {(batch, drafts + 1) for batch in (1, 2, 4, 8, 16)} <= shapes
    assert {(32, width) for width in range(1, 5)} <= shapes
    assert all(batch * width <= 128 for batch, width in shapes)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_qwen_state_rows_and_pages_match_the_prepared_layouts(world_size):
    import torch
    from b12x.attention import qsa

    from ignition.recipes.qwen38.state import HybridStateLayout

    recipe = QWEN.recipe
    assert recipe.speculation == Speculation(draft_tokens=5, token_capacity=128) and recipe.max_draft_tokens == 7
    assert recipe.speculation.adaptive and recipe.history_copy

    def layer(speculative_tokens):
        geometry = qsa.cache_requirements(main_page_size=recipe.page_tokens, position_axes=3,
                                          kv_heads=max(1, 2 // world_size),
                                          max_speculative_tokens=speculative_tokens, kv_dtype=torch.float8_e4m3fn)
        return sum(math.prod(shape) * dtype.itemsize for shape, dtype in (
            (geometry.main_k_page_shape, geometry.kv_dtype), (geometry.main_v_page_shape, geometry.kv_dtype),
            (geometry.compressed_page_shape, torch.bfloat16)))

    # Layers 3, 7, ..., 47 each own one page pool; learned drafts add the draft layer's.
    target = len(range(3, 48, 4))
    for speculation, tokens, layers in ((recipe.speculation, 7, target + 1), (COPY, 7, target), (None, 0, target)):
        costs = recipe.costs(world_size, speculation)
        assert costs.state_row_bytes == HybridStateLayout.build(
            speculative_tokens=tokens, world_size=world_size).row_bytes
        assert costs.page_bytes == layers * layer(tokens)


@pytest.mark.parametrize("alias,world_size", [("qwen3.8-27b", 1), ("kimi-k3-mxfp4-csf", 16),
    ("mimo-v2.6-pro-mopd", 2), ("mimo-v2.6-pro-mopd", 4), ("mimo-v2.6-pro-mopd", 8), ("nemotron-bolt-24gb", 1)])
@pytest.mark.parametrize("drafting", [False, True])
def test_fp8_cache_views_and_planned_bytes(alias, world_size, drafting):
    from dataclasses import replace
    from importlib import import_module

    import torch
    from ignition.components.parallel import TensorParallelShard

    model = resolve(alias)
    if drafting and model.recipe.speculation is None:
        pytest.skip("recipe has no learned draft")
    snapshot = model.snapshot(HUB)
    if not (snapshot / "config.json").is_file():
        pytest.skip("checkpoint metadata is not installed")
    prefix = f"ignition.recipes.{model.recipe.name}"
    config = import_module(prefix + ".config").ModelConfig.load(snapshot)
    if drafting and model.recipe.name == "kimi3":
        from .test_kimi3 import DRAFT
        from ignition.recipes.kimi3.config import DSparkConfig

        if not (DRAFT / "config.json").is_file():
            pytest.skip("Red Hat DSpark checkpoint metadata is not installed")
        config = replace(config, draft=DSparkConfig.load(DRAFT, config))
    if drafting and model.recipe.name == "nemotron_h":
        draft = import_module(prefix + ".draft_config")
        draft_snapshot = HUB / f"models--{draft.REPOSITORY.replace('/', '--')}" / "snapshots" / draft.REVISION
        if not (draft_snapshot / "config.json").is_file():
            pytest.skip("draft checkpoint metadata is not installed")
        config = replace(config, draft=draft.DraftConfig.load(draft_snapshot, config))
    module = import_module(prefix + ".pool")
    if model.recipe.name == "mimo_v2":
        shard = TensorParallelShard(0, world_size)
        pool = module.ContinuationPool.allocate(config, shard, 2, 3, "meta", dflash=drafting)
        caches = [pool.cache(layer, kind) for layer in range(config.layers) for kind in ("key", "value")]
    elif model.recipe.name == "kimi3":
        # Decode context parallelism spreads each page's positions over every rank, as preparation lays it out.
        pool = module.ContinuationPool.for_model(config, world_size, 2, 3, "meta",
            speculative_tokens=model.recipe.speculation.draft_tokens if drafting else 0)
        caches = [pool.attention(layer) for layer, kind in enumerate(config.layers) if kind == "mla"]
    else:
        pool = module.ContinuationPool.for_model(config, 2, 3, "meta")
        caches = [cache for layer, kind in enumerate(config.layers) if kind in ("attention", "full_attention")
                  for cache in pool.attention(layer, config)]
    draft_caches = [pool.reserved_region(region.name) for region in pool.layout.regions
                    if region.name.startswith("draft") and region.name.endswith((".key", ".value"))]
    if model.recipe.name == "kimi3":
        assert len(draft_caches) == (2 * config.draft.layers if drafting else 0)
        assert all(cache.dtype == torch.bfloat16 for cache in draft_caches)
    else:
        caches.extend(draft_caches)
    assert caches and all(cache.dtype == torch.float8_e4m3fn for cache in caches)
    costs = model.recipe.costs(world_size, model.recipe.speculation if drafting else None)
    assert pool.layout.row_bytes == costs.state_row_bytes
    assert sum(page[0].nbytes for page in pool.pages.values()) == costs.page_bytes


@pytest.mark.parametrize("world_size", [2, 4])
def test_glm_state_rows_and_pages_match_the_continuation_pool(world_size):
    from ignition.recipes.glm53.pool import continuation_layout

    recipe = GLM.recipe
    assert recipe.speculation == Speculation(draft_tokens=5, token_capacity=128) and recipe.max_draft_tokens == 7
    assert not recipe.history_copy and recipe.page_tokens == 256
    layer_types = tuple("indexed_attention" if index % 4 == 3 else "linear_attention" for index in range(45))
    target = layer_types.count("indexed_attention")
    # Learned drafts add the draft layer's latent and index pages to the target's.
    for speculation, layers in ((recipe.speculation, target + 1), (None, target)):
        costs = recipe.costs(world_size, speculation)
        assert costs.state_row_bytes == continuation_layout(
            layer_types, world_size, mtp=speculation is not None).row_bytes
        assert costs.page_bytes == layers * (4 * 64 * 528 + 64 * 132)
        # Context parallelism places 256 records of each page on every rank.
        assert costs.page_tokens == 256 * world_size


@pytest.mark.parametrize("model", [QWEN, GLM])
def test_weights_exclude_the_draft_model_unless_drafts_are_learned(model):
    recipe, checkpoint = model.recipe, model.checkpoint
    world = recipe.tensor_parallel[0]
    learned = recipe.weights_bytes(checkpoint, world, recipe.speculation)
    loaded = checkpoint.device_bytes + checkpoint.expansion_bytes
    assert learned == -(-loaded // world) and 0 < checkpoint.draft_bytes < checkpoint.device_bytes
    for speculation in (None, COPY):
        assert recipe.weights_bytes(checkpoint, world, speculation) == -(
            -(loaded - checkpoint.draft_bytes) // world)
    # Both recipes commit verified tokens into one live recurrent row.
    assert recipe.live_rows(4, recipe.speculation) == 1 + 4
    assert recipe.live_rows(4, None) == 1 + 4


def test_glm_stored_bytes_match_the_cached_checkpoint():
    if not GLM.cached(HUB):
        pytest.skip("the pinned GLM revision is not in the hub cache")
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    tensors = inspect_indexed_checkpoint(GLM.snapshot(HUB)).tensors.values()
    checkpoint = GLM.checkpoint
    assert checkpoint.stored_bytes == sum(tensor.nbytes for tensor in tensors)
    assert checkpoint.device_bytes == sum(tensor.nbytes for tensor in tensors)
    assert checkpoint.mapped_host_bytes == 0
    assert checkpoint.draft_bytes == sum(
        tensor.nbytes for tensor in tensors if tensor.name.startswith("model.language_model.layers.45."))


def fitted(model, gpus, **settings):
    devices = tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index}", 96 << 30, 95 << 30, (12, 0))
                    for index in range(gpus))
    arguments = {"concurrency": 4, "reserve_bytes": 0, "speculation": model.recipe.speculation, **settings}
    return planning.fit(model.recipe, model.checkpoint, devices, **arguments)


@pytest.mark.parametrize("compile_forwards", [False, True])
@pytest.mark.parametrize("concurrency", [1, 5, 32, 64])
def test_qwen_adapter_configures_the_bootstrap_from_a_plan(monkeypatch, tmp_path, compile_forwards, concurrency):
    from ignition.recipes.qwen38.state import HybridStateLayout

    seen = {}

    def prepare_model(snapshot, config, *, progress, compile_decoder):
        seen.update(snapshot=snapshot, config=config, compile_decoder=compile_decoder)
        return "prepared"

    monkeypatch.setattr("ignition.recipes.qwen38.bootstrap.prepare_model", prepare_model)
    plan = fitted(QWEN, 1, concurrency=concurrency, reserve_bytes=2 << 30)
    assert QWEN.recipe.prepare(tmp_path, plan, compile_forwards=compile_forwards, kernel_cache=tmp_path / "kernels",
                               progress=None) == "prepared"
    config = seen["config"]
    assert seen["compile_decoder"] is compile_forwards and config.compile_decoder is compile_forwards
    assert config.devices == (0,) and config.max_active == concurrency and config.mtp
    assert (config.speculative_widths, config.speculative_fixed_width) == ((1, 2, 3, 4, 5, 6), None)
    assert config.speculative_token_capacity == 128
    assert (min(config.request_capacity, 16), 6) in config.speculative_shapes
    if concurrency >= 32:
        assert (32, 4) in config.speculative_shapes and (32, 5) not in config.speculative_shapes
    assert (config.speculative_proposal_source, config.history_drafts) == (
        "mtp", False)
    assert config.pages is None and config.context == plan.recipe.context
    assert config.preparation_pages == plan.prefill_chunk // 64 + 2 * config.request_capacity
    assert config.prefill_chunk_size == plan.prefill_chunk == 4096 and config.kernel_cache == tmp_path / "kernels"
    assert config.headroom_bytes == planning.HEADROOM_BYTES + (2 << 30)
    layout = HybridStateLayout.build(speculative_tokens=7, world_size=1)
    assert layout.snapshot_capacity(config.snapshot_budget_bytes) == plan.snapshots
    # The pool holds a null row, one live row per slot, and the snapshot rows.
    assert 1 + config.request_capacity + plan.snapshots == plan.live_rows + plan.cache_states


def test_qwen_adapter_passes_speculation_overrides(monkeypatch, tmp_path):
    from ignition.recipes.qwen38.state import HybridStateLayout

    seen = []
    monkeypatch.setattr("ignition.recipes.qwen38.bootstrap.prepare_model",
                        lambda snapshot, config, *, progress, compile_decoder: seen.append(config))

    def configured(speculation):
        plan = fitted(QWEN, 1, speculation=speculation)
        QWEN.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
        return plan, seen.pop()

    plan, config = configured(Speculation(2, adaptive=False, proposal_source="copy",
                                          history_drafts=True))
    assert config.mtp and (config.speculative_widths, config.speculative_fixed_width) == ((1, 2, 3), 3)
    assert (config.speculative_proposal_source, config.history_drafts) == (
        "copy", True)
    assert config.speculative_shapes == tuple((b, w) for b in (1, 2, 4) for w in (3, 2, 1) if w == 1 or b * w <= 4)

    plan, config = configured(None)
    assert not config.mtp
    layout = HybridStateLayout.build(speculative_tokens=0, world_size=1)
    assert layout.row_bytes == plan.costs.state_row_bytes
    assert layout.snapshot_capacity(config.snapshot_budget_bytes) == plan.snapshots
    # Without speculation the pool holds a null row, one live row per slot, and the snapshot rows.
    assert 1 + config.request_capacity + plan.snapshots == plan.live_rows + plan.cache_states


def test_tensor_parallel_adapters_prepare_a_rank_group(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, *, progress, package_storage=None, ranks=None, checkpoint=None:
                        seen.append((config, prepare, ranks, checkpoint)) or "group")
    from ignition.recipes.glm53 import bootstrap as glm
    from ignition.recipes.qwen38 import bootstrap as qwen

    plan = fitted(QWEN, 2)
    assert QWEN.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None) == "group"
    config, prepare, ranks, _ = seen.pop()
    assert prepare is qwen.prepare_rank and config.devices == (0, 1) and config.pages is None
    assert config.hosts is None and ranks is None

    # Across hosts, each rank prepares on its own host's GPU 0 through rank zero's rendezvous.
    rendezvous = SimpleNamespace(hosts=("boot-a", "boot-b"))
    QWEN.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None, ranks=rendezvous)
    config, prepare, ranks, checkpoint = seen.pop()
    assert config.devices == (0, 0) and config.hosts == ("boot-a", "boot-b")
    assert ranks is rendezvous and checkpoint == plan.checkpoint

    plan = fitted(GLM, 4, concurrency=2, prefill_chunk=2048)
    assert GLM.recipe.prepare(tmp_path, plan, compile_forwards=False, kernel_cache=tmp_path, progress=None) == "group"
    config, prepare, _, _ = seen.pop()
    assert prepare is glm.prepare_rank and config.devices == (0, 1, 2, 3)
    # Like Qwen, GLM pages fill the memory measured after graph capture, at least the plan's minimum.
    assert (config.max_active, config.pages, config.minimum_pages, config.context) == (
        2, None, plan.minimum_pages, plan.context)
    assert config.compile_decoder is False and config.kernel_cache == tmp_path and config.context_parallel
    assert config.page_tokens == plan.page_tokens == 1024 and config.context % config.page_tokens == 0
    assert config.page_capacity == plan.minimum_pages and config.prefill_chunk_size == 2048
    assert config.mtp and (config.speculative_widths, config.speculative_fixed_width) == ((1, 2, 3, 4, 5, 6), None)

    plan = fitted(GLM, 4, speculation=Speculation(2, adaptive=False, ))
    GLM.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    config, *_ = seen.pop()
    assert config.mtp and (config.speculative_widths, config.speculative_fixed_width) == ((1, 2, 3), 3)

    plan = fitted(GLM, 4, speculation=None)
    GLM.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    config, *_ = seen.pop()
    assert config.mtp is False and config.page_capacity == plan.minimum_pages


def test_recipes_declare_whether_they_overlap_branches():
    overlapping = {alias: resolve(alias).recipe.stream_overlap for alias in (
        "qwen3.8-flash", "glm-5.3-flash", "deepseek-v4-flash", "deepseek-v4.1-flash", "mimo-v2.6-pro-mopd")}
    # MiMo composes every branch on the main stream; the others overlap shared experts or projection chains.
    assert overlapping == {"qwen3.8-flash": True, "glm-5.3-flash": True, "deepseek-v4-flash": True,
                           "deepseek-v4.1-flash": True, "mimo-v2.6-pro-mopd": False}


@pytest.mark.parametrize("alias", ["qwen3.8-flash", "glm-5.3-flash", "deepseek-v4-flash", "deepseek-v4.1-flash"])
def test_adapters_carry_stream_overlap_into_the_preparation_config(monkeypatch, tmp_path, alias):
    seen = []
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **options: seen.append(config) or "group")
    model = resolve(alias)
    for overlap in (True, False):
        plan = fitted(model, 4, stream_overlap=overlap)
        assert plan.stream_overlap is overlap
        assert model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None) == "group"
        assert seen.pop().stream_overlap is overlap


@pytest.mark.parametrize("compression", [None, "csf"])
@pytest.mark.parametrize("drafting", [False, True])
def test_deepseek41_native_checkpoint_and_scale_compression(monkeypatch, tmp_path, compression, drafting):
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    model = resolve("deepseek-v4.1-flash")
    assert model.checkpoint.repository == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert resolve(model.checkpoint.repository).checkpoint is model.checkpoint
    assert model.recipe.sampling_defaults.as_dict() == {
        "temperature": 1., "top_p": 1., "top_k": -1, "seed": 0, "min_p": 0.,
        "repetition_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.}
    speculation = model.recipe.speculation if drafting else None
    native = fitted(model, 4, speculation=speculation)
    plan = fitted(model, 4, speculation=speculation, expert_scale_compression=compression)
    assert plan.max_images == 4 and plan.image_concurrency == min(4, plan.concurrency)
    assert plan.expert_scale_compression == compression
    assert (plan.weights_bytes < native.weights_bytes) == (compression == "csf")
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **options: config)
    config = model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert config.expert_scale_compression == compression and config.dspark == drafting
    assert config.image_concurrency == plan.image_concurrency
    with pytest.raises(ValueError, match="scale compression"):
        PreparationConfig(expert_scale_compression="zip")


@pytest.mark.parametrize("graph_budget", [64, 32])
def test_deepseek41_recipe_defaults_and_adapter_preserve_packed_gpu_planning(monkeypatch, tmp_path, graph_budget):
    from ignition.cli import _parser, preparation

    model = resolve("deepseek-v4.1-flash")
    assert (model.recipe.concurrency, model.recipe.max_concurrency, model.recipe.prefill_chunk) == (32, 64, 4096)
    assert model.recipe.speculation == Speculation(5, token_capacity=128, packed_verification=True,
                                                  draft_prefix_selection="gpu", graph_budget=64)
    flags = [] if graph_budget == 64 else ["--speculative-graph-budget", str(graph_budget)]
    args = _parser().parse_args(["export", model.checkpoint.alias, *flags])
    plan = fitted(model, 4, concurrency=model.recipe.concurrency,
                  speculation=preparation.speculation(args, model))
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group",
                        lambda snapshot, config, prepare, **options: config)
    config = model.recipe.prepare(tmp_path, plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert (config.max_active, config.prefill_chunk_size, config.speculative_token_capacity) == (32, 4096, 128)
    assert config.dspark and config.split_speculation
    assert config.packed_verification and config.draft_prefix_selection == "gpu"
    assert config.speculative_graph_budget == graph_budget
