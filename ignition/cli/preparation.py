"""Model resolution, memory planning, and downloads for preparation commands."""

import os
from dataclasses import dataclass, replace

from ignition import planning
from ignition.errors import FetchError, UsageError
from ignition.recipes import ResolvedModel, resolve
from ignition.recipes.declaration import Speculation

from .config import AUTOTUNE_EFFORTS, Configuration, Setting


@dataclass(frozen=True)
class Host:
    """GPUs as the command line indexes them, and the caller's restriction on them."""

    gpus: tuple[planning.Gpu, ...]
    restriction: str | None

    @classmethod
    def discover(cls) -> "Host":
        gpus = planning.discover_gpus()
        # The one environment variable Flare inspects: a caller's restriction
        # changes which GPU each index names, and the plan reports that.
        restriction = os.environ.get("CUDA_VISIBLE_DEVICES")
        if restriction is not None:
            gpus = planning.restrict_gpus(gpus, restriction)
        return cls(gpus, restriction)


def resolve_model(configuration: Configuration, name: str | None) -> ResolvedModel:
    name = configuration.model if name is None else name
    if name is None:
        raise UsageError("no MODEL given and defaults.model is not set in the configuration file")
    model = resolve(name, configuration.directory("hub_cache"))
    model.verify(configuration.directory("hub_cache"))
    return model


def derive(configuration: Configuration, model: ResolvedModel, host: Host, *, gpus: str | None,
           concurrency: Setting, speculation: Speculation | None, prefill_chunk: int | None = None,
           context_parallel: bool = True, stream_overlap: bool = True, roce_ring: bool = False, roce_chain: bool = False,
           mixed_prefill: bool = False, prompt_logprobs: bool = False, expert_scale_compression: str | None = None,
           online_mxfp8: bool = False, language_model_only: bool = False,
           compile_workers: int | None = None, tuning_gpus: str | None = None, autotune_effort: str | None = None, collective_tuning: str | None = None, collective_ladder=None,
           image_concurrency: int | None = None, mla_compute_mode: str | None = None,
           lock_gpu_clocks: bool = False, native_dense_activations: bool = False, packed_dense_w4a16: bool = False,
           joint_tuning: bool = False, shard_hyperconnections: bool = False) -> planning.Plan:
    """Select GPUs and estimate memory for one checkpoint from the configuration and command-line intent."""
    from ignition.preparation.collective_ladder import resolve as resolve_ladder

    mode = collective_tuning_mode(configuration, model, collective_tuning)
    ladder = resolve_ladder(mode, collective_ladder)
    workers = compiler_workers(configuration, compile_workers)
    result = planning.plan(
        model.recipe, model.checkpoint, host.gpus, selected=None if gpus is None else planning.parse_gpus(gpus),
        concurrency=concurrency.value, reserve_bytes=configuration.reserve_bytes, speculation=speculation,
        prefill_chunk=prefill_chunk, context_parallel=context_parallel, stream_overlap=stream_overlap, roce_ring=roce_ring, roce_chain=roce_chain,
        mixed_prefill=mixed_prefill, prompt_logprobs=prompt_logprobs, expert_scale_compression=expert_scale_compression,
        image_concurrency=image_capacity_setting(configuration, model, image_concurrency),
        online_mxfp8=online_mxfp8, language_model_only=language_model_only, mla_compute_mode=mla_compute_mode)
    result = replace(result, compile_workers=workers, hub_cache=configuration.directory("hub_cache"),
                     autotune_effort=tuning_effort(configuration, autotune_effort),
                     collective_tuning=mode, collective_ladder=ladder,
                     joint_tuning=joint_tuning, shard_hyperconnections=shard_hyperconnections,
                     lock_gpu_clocks=lock_gpu_clocks, native_dense_activations=native_dense_activations, packed_dense_w4a16=packed_dense_w4a16)
    return tuning_devices(result, host, tuning_gpus)


def image_capacity_setting(configuration, model, flag=None):
    if not model.recipe.max_images and flag is None:
        return None
    return configuration.model_setting(model.checkpoint.alias, "image_concurrency", None, flag).value


def tuning_effort(configuration, flag=None):
    value = configuration.get("preparation.autotune_effort", flag).value
    if value not in AUTOTUNE_EFFORTS:
        raise UsageError(f"--autotune-effort must be one of {', '.join(AUTOTUNE_EFFORTS)}")
    return value


def collective_tuning_mode(configuration, model, flag=None):
    value = configuration.get("preparation.collective_tuning", flag).value
    value = model.recipe.collective_tuning if value is None else value
    if value not in ("static", "runtime"):
        raise UsageError("--collective-tuning must be static or runtime")
    return value


def compiler_workers(configuration, flag=None):
    value = configuration.get("preparation.compile_workers", flag).value
    if type(value) is not int or value < 0 or flag == 0:
        raise UsageError("--compile-workers must be a positive integer")
    return value or None


def tuning_devices(plan, host, flag):
    if plan.compile_workers is not None and plan.compile_workers < plan.world_size:
        raise UsageError("--compile-workers must provide at least one process per local serving rank")
    if flag is None:
        return plan
    indices = planning.parse_gpus(flag)
    available = {gpu.index: gpu for gpu in host.gpus}
    if any(index not in available for index in indices):
        raise UsageError("--tuning-gpus names a GPU outside the visible device list")
    count = plan.world_size
    if indices[:count] != tuple(gpu.index for gpu in plan.gpus) or len(indices) % count:
        raise UsageError("--tuning-gpus must start with the serving GPUs and contain complete TP replicas")
    if plan.compile_workers is not None and plan.compile_workers < len(indices):
        raise UsageError("--compile-workers must provide at least one process per local tuning rank")
    selected = tuple(available[index] for index in indices)
    for position, gpu in enumerate(selected[count:], start=count):
        owner = plan.gpus[position % count]
        if (gpu.name, gpu.capability) != (owner.name, owner.capability):
            raise UsageError("each tuning replica must match its corresponding serving GPU model and architecture")
        minimum = (plan.fixed_bytes + plan.headroom_bytes + plan.reserve_bytes
                   + plan.minimum_pages * plan.costs.page_bytes)
        if gpu.free_bytes < minimum:
            raise UsageError(f"tuning GPU {gpu.index} needs at least {minimum >> 20} MiB free for a model replica")
    return replace(plan, tuning_gpus=selected if len(selected) > count else ())


def context_parallel(args, model: ResolvedModel) -> bool:
    """Whether a plan may shard attention context across its ranks, on one host or several.

    Planning shards only recipes that declare context-parallel page costs, and
    --no-context-parallel applies only to those.
    """
    if args.no_context_parallel and not any(costs.page_tokens for costs in model.recipe.ranks.values()):
        raise UsageError(f"--no-context-parallel: {model.checkpoint.alias} does not shard attention context")
    return not args.no_context_parallel


def stream_overlap(args, model: ResolvedModel) -> bool:
    """Whether a plan may run independent branches on an auxiliary stream.

    --no-stream-overlap applies only to recipes that declare stream overlap.
    """
    if args.no_stream_overlap and not model.recipe.stream_overlap:
        raise UsageError(f"--no-stream-overlap: {model.checkpoint.alias} runs no branches concurrently")
    return not args.no_stream_overlap


def speculation(args, model: ResolvedModel, configuration: Configuration | None = None) -> Speculation | None:
    """Apply the command line's speculation overrides to the recipe's settings; None disables speculation."""
    recipe, alias = model.recipe, model.checkpoint.alias
    default, limit = recipe.speculation, recipe.max_draft_tokens
    graph_budget = Setting(args.speculative_graph_budget, "flag")
    if configuration is not None:
        graph_budget = configuration.model_setting(alias, "speculative_graph_budget",
            None if default is None else default.graph_budget, args.speculative_graph_budget)
    elif graph_budget.value is None:
        graph_budget = Setting(None if default is None else default.graph_budget, "recipe")
    explicit_graph_budget = graph_budget.value is not None and graph_budget.source not in ("recipe", "[defaults]")
    given = [flag for flag, value in (
        ("--mtp-draft-tokens", args.mtp_draft_tokens),
        ("--speculative-token-capacity", args.speculative_token_capacity),
        ("--no-mtp-adaptive-verification", args.no_mtp_adaptive_verification or None),
        ("--draft-head-quantization", getattr(args, "draft_head_quantization", None)),
        ("--markov-head-quantization", getattr(args, "markov_head_quantization", None)),
        ("--vocabulary-parallel-sampling", getattr(args, "vocabulary_parallel_sampling", False) or None)) if value is not None]
    if explicit_graph_budget:
        given.append("--speculative-graph-budget")
    if default is None:
        if args.no_mtp or given:
            raise UsageError(f"{alias} has no multi-token speculation; {(given or ['--no-mtp'])[0]} does not apply")
        return None
    if args.no_mtp:
        if given:
            raise UsageError(f"--no-mtp disables speculation; {', '.join(given)} cannot accompany it")
        return None

    drafts = default.draft_tokens if args.mtp_draft_tokens is None else args.mtp_draft_tokens
    if not 0 <= drafts <= limit:
        raise UsageError(f"--mtp-draft-tokens must be 0 through {limit}; --no-mtp disables speculation")
    adaptive = default.adaptive and not args.no_mtp_adaptive_verification
    capacity = args.speculative_token_capacity
    if configuration is not None:
        capacity = configuration.model_setting(alias, "speculative_token_capacity", default.token_capacity, capacity).value
    elif capacity is None:
        capacity = default.token_capacity
    if capacity is not None and (type(capacity) is not int or capacity < drafts + 1):
        raise UsageError("--speculative-token-capacity must cover the maximum verification width")
    packed = default.packed_verification and drafts > 0
    prefix_selection = default.draft_prefix_selection if drafts else "cpu"
    if prefix_selection == "gpu" and not adaptive:
        raise UsageError(f"{alias} requires adaptive verification; --no-mtp-adaptive-verification does not apply")
    physical_catalog = packed
    if explicit_graph_budget and not (physical_catalog and drafts > 0):
        raise UsageError("--speculative-graph-budget requires a packed target catalog with positive draft depth")
    budget = graph_budget.value if physical_catalog and drafts > 0 else None
    if budget is not None and (type(budget) is not int or budget < 1):
        raise UsageError("--speculative-graph-budget must be a positive integer")
    draft_head = getattr(args, "draft_head_quantization", None)
    markov_head = getattr(args, "markov_head_quantization", None)
    if (draft_head or markov_head) and drafts < 1:
        raise UsageError("draft head quantization requires positive learned draft depth")
    return Speculation(drafts, adaptive, token_capacity=capacity, packed_verification=packed,
                       draft_prefix_selection=prefix_selection, graph_budget=budget,
                       draft_head_quantization=draft_head, markov_head_quantization=markov_head,
                       vocabulary_parallel_sampling=getattr(args, "vocabulary_parallel_sampling", False))


def fetch(model: ResolvedModel, hub_cache, *, progress_bars: bool = True) -> str:
    """Download the pinned revision's missing files and return the snapshot directory."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress_bars:
            disable_progress_bars()
        return snapshot_download(model.checkpoint.repository, revision=model.checkpoint.revision,
                                 cache_dir=hub_cache)
    except Exception as error:  # noqa: BLE001 - Network, authorization, and disk failures share one exit code.
        raise FetchError(f"cannot download {model.checkpoint.repository} at revision"
                         f" {model.checkpoint.revision[:8]}: {error}") from error
