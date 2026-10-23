"""Derive a serving plan from a recipe's cost model and the host's free GPU memory.

Weights, graphs, scratch and execution rows have dedicated storage. KV pages
take the remaining supply after headroom and the configured reserve. Immutable
prefix snapshots borrow KV backing while retained; an idle snapshot slot costs
no physical memory. Active requests need at least 8,192 tokens each.
"""

from dataclasses import dataclass, replace
import os
from pathlib import Path

from ignition.errors import CapacityError, UsageError
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.recipes.declaration import Checkpoint, Recipe, Speculation

# Free memory kept after the final KV allocation during preparation. Headroom
# covers the transient working memory of compiled warm-up and graph capture,
# measured at 0.83 GiB for one Qwen rank with 4096-token prefill chunks. Part of
# it holds chunk activations, so chunk tokens past 4096 add HEADROOM_CHUNK_TOKEN_BYTES
# each (8192-token chunks need about 1.3 GiB on GB300). Helios sizes its caches
# after graph upload with a separate driver margin. Graphs, scratch, and the CUDA
# context are planned separately.
HEADROOM_BYTES = 1 << 30
HEADROOM_CHUNK_TOKENS = 4096
HEADROOM_CHUNK_TOKEN_BYTES = 128 << 10
MINIMUM_REQUEST_TOKENS = 8192
# Each exchange can retain prompt and terminal checkpoints; snapshot slots have
# virtual capacity for about four recent exchanges per concurrent request.
CACHE_STATES_PER_REQUEST = 8
SUPPORTED_CAPABILITIES = ((10, 3), (12, 0), (12, 1))
SUPPORTED_ARCHITECTURES = "SM103, SM120, or SM121"
_GIB = 1 << 30


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    total_bytes: int
    free_bytes: int
    capability: tuple[int, int]

    @property
    def supported(self) -> bool:
        return self.capability in SUPPORTED_CAPABILITIES


def discover_gpus() -> tuple[Gpu, ...]:
    """Enumerate NVML devices and include the shared CUDA budget on Windows iGPUs."""
    try:
        from cuda.bindings import nvml

        nvml.init_v2()
    except Exception as error:  # noqa: BLE001 - Import and driver failures share one report.
        raise CapacityError(f"cannot enumerate GPUs through NVML: {error}") from error
    try:
        gpus = []
        for index in range(nvml.device_get_count_v2()):
            handle = nvml.device_get_handle_by_index_v2(index)
            try:
                capability = tuple(nvml.device_get_cuda_compute_capability(handle))
            except nvml.NotSupportedError:
                # NVML also enumerates non-CUDA devices, including Windows NPUs.
                continue
            try:
                memory = nvml.device_get_memory_info_v2(handle)
                total, free = memory.total, memory.free
                if os.name == "nt":
                    shared = _integrated_memory(nvml.device_get_pci_info_v3(handle).bus_id, required=False)
                    if shared is not None:
                        total, free = shared
            except nvml.NotSupportedError:
                # GB10 has no framebuffer for NVML to report; it allocates from system memory.
                total, free = _integrated_memory(nvml.device_get_pci_info_v3(handle).bus_id)
            gpus.append(Gpu(index, nvml.device_get_name(handle), nvml.device_get_uuid(handle), total,
                            free, capability))
        return tuple(gpus)
    finally:
        nvml.shutdown()


def _integrated_memory(bus_id: str, *, required=True) -> tuple[int, int] | None:
    """Query shared DRAM on Linux or the dedicated-plus-shared CUDA budget on Windows."""
    from cuda.bindings import driver

    def checked(call, *arguments):
        status, *values = call(*arguments)
        if status != driver.CUresult.CUDA_SUCCESS:
            raise CapacityError(f"cannot query integrated GPU memory: {call.__name__}: {status}")
        return values[0] if len(values) == 1 else tuple(values) if values else None

    checked(driver.cuInit, 0)
    device = checked(driver.cuDeviceGetByPCIBusId, bus_id.encode())
    if not checked(driver.cuDeviceGetAttribute, driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED, device):
        if not required:
            return None
        raise CapacityError(f"GPU {bus_id} does not report device memory and is not an integrated GPU")
    if os.name == "nt":
        context = checked(driver.cuDevicePrimaryCtxRetain, device)
        try:
            checked(driver.cuCtxPushCurrent, context)
            try:
                free, total = checked(driver.cuMemGetInfo)
                return total, free
            finally:
                checked(driver.cuCtxPopCurrent)
        finally:
            checked(driver.cuDevicePrimaryCtxRelease, device)
    total = checked(driver.cuDeviceTotalMem, device)
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return total, min(total, int(line.split()[1]) * 1024)
    raise CapacityError("Linux MemAvailable is required to size integrated GPU memory")


def restrict_gpus(gpus: tuple[Gpu, ...], visible: str) -> tuple[Gpu, ...]:
    """Renumber GPUs as a caller's CUDA_VISIBLE_DEVICES value enumerates them.

    Integer entries are NVML indices and other entries are GPU UUID prefixes.
    """
    restricted = []
    for entry in filter(None, (value.strip() for value in visible.split(","))):
        if entry.isdecimal():
            matches = [gpu for gpu in gpus if gpu.index == int(entry)]
        else:
            matches = [gpu for gpu in gpus if gpu.uuid.startswith(entry)]
        if len(matches) != 1 or matches[0] in restricted:
            raise UsageError(f"CUDA_VISIBLE_DEVICES entry {entry!r} does not name one distinct GPU")
        restricted.append(matches[0])
    return tuple(replace(gpu, index=index) for index, gpu in enumerate(restricted))


def parse_gpus(text: str) -> tuple[int, ...]:
    """Parse `0`, `0,1`, or `0-3` into distinct GPU indices."""
    indices = []
    for part in text.split(","):
        first, dash, last = part.strip().partition("-")
        if not first.isdecimal() or (dash and not last.isdecimal()):
            raise UsageError(f"--gpus expects indices such as 0, 0,1, or 0-3; got {text!r}")
        indices.extend(range(int(first), int(last if dash else first) + 1))
    if not indices or len(set(indices)) != len(indices):
        raise UsageError(f"--gpus expects distinct indices in increasing ranges; got {text!r}")
    return tuple(indices)


def format_gpus(indices) -> str:
    indices = tuple(indices)
    if len(indices) > 1 and indices == tuple(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}-{indices[-1]}"
    return ",".join(map(str, indices))


@dataclass(frozen=True)
class Plan:
    """Per-rank memory demand and supply for one checkpoint on selected GPUs."""

    recipe: Recipe
    checkpoint: Checkpoint
    gpus: tuple[Gpu, ...]
    concurrency: int
    prefill_chunk: int
    speculation: Speculation | None
    cache_target: int
    cache_states: int
    pages: int
    reserve_bytes: int
    weights_bytes: int
    graph_bytes: int
    # Ranks shard each page's context when the recipe declares context-parallel page costs.
    context_parallel: bool = True
    # Captured graphs run independent branches on an auxiliary stream when the recipe declares
    # stream overlap; otherwise every branch runs serially on the main stream.
    stream_overlap: bool = True
    mixed_prefill: bool = False
    prompt_logprobs: bool = False
    expert_scale_compression: str | None = None
    mla_compute_mode: str = "auto"
    roce_ring: bool = False
    roce_chain: bool = False
    compile_workers: int | None = None
    autotune_effort: str = "medium"
    hub_cache: Path | None = None
    tuning_gpus: tuple[Gpu, ...] = ()
    tuning_replica: TuningReplica | None = None
    image_concurrency: int = 0
    # The recipe's eligible BF16 projections load as MXFP8 when the plan asks for it.
    online_mxfp8: bool = False
    # The plan prepares no vision encoder and serves text requests only.
    language_model_only: bool = False

    @property
    def world_size(self) -> int:
        return len(self.gpus)

    @property
    def max_images(self) -> int:
        return 0 if self.language_model_only else self.recipe.max_images

    @property
    def speculative_token_capacity(self) -> int:
        from ignition.components.capacity import speculative_token_capacity

        return (0 if self.speculation is None else speculative_token_capacity(
            self.concurrency, self.speculation.draft_tokens + 1, self.speculation.token_capacity))

    @property
    def costs(self):
        return self.recipe.costs(self.world_size, self.speculation, self.checkpoint)

    @property
    def live_rows(self) -> int:
        return self.recipe.live_rows(self.concurrency, self.speculation)

    @property
    def live_bytes(self) -> int:
        return self.live_rows * self.costs.state_row_bytes

    @property
    def fixed_bytes(self) -> int:
        return self.weights_bytes + self.graph_bytes + self.live_bytes

    @property
    def demand_pages(self) -> int:
        return self.concurrency * (self.recipe.context // self.page_tokens)

    @property
    def page_tokens(self) -> int:
        return (self.costs.page_tokens if self.context_parallel else None) or self.recipe.page_tokens

    @property
    def minimum_pages(self) -> int:
        return self.concurrency * (MINIMUM_REQUEST_TOKENS // self.page_tokens)

    @property
    def free_bytes(self) -> int:
        return min(gpu.free_bytes for gpu in self.gpus)

    @property
    def headroom_bytes(self) -> int:
        tokens = max(self.prefill_chunk,
            max((p + (b - 1) * w for _, b, w, p in self.mixed_shapes), default=0))
        return HEADROOM_BYTES + max(0, tokens - HEADROOM_CHUNK_TOKENS) * HEADROOM_CHUNK_TOKEN_BYTES

    @property
    def mixed_shapes(self):
        return (self.recipe.mixed_shapes(self.concurrency, self.prefill_chunk, self.speculation)
                if self.mixed_prefill else ())

    @property
    def supply_bytes(self) -> int:
        return self.free_bytes - self.headroom_bytes - self.reserve_bytes

    @property
    def page_bytes(self) -> int:
        return self.pages * self.costs.page_bytes

    @property
    def tokens(self) -> int:
        return self.pages * self.page_tokens

    @property
    def context(self) -> int:
        """The longest sequence one request may reach."""
        return min(self.recipe.context, self.tokens)

    @property
    def guaranteed_tokens(self) -> int:
        """Tokens every request can hold while all admitted requests are active."""
        return min(self.recipe.context, self.pages // self.concurrency * self.page_tokens)

    @property
    def snapshots(self) -> int:
        return self.cache_states

    @property
    def fits(self) -> bool:
        return (self.pages >= self.minimum_pages
                and self.fixed_bytes + self.page_bytes <= self.supply_bytes)

    def to_dict(self) -> dict:
        return {
            "alias": self.checkpoint.alias, "repository": self.checkpoint.repository,
            "revision": self.checkpoint.revision, "architecture": self.checkpoint.architecture,
            "gpus": [{"index": gpu.index, "name": gpu.name, "uuid": gpu.uuid, "free_bytes": gpu.free_bytes,
                      "total_bytes": gpu.total_bytes} for gpu in self.gpus],
            "tensor_parallel": self.world_size, "max_concurrency": self.concurrency,
            "image_concurrency": self.image_concurrency,
            "prefill_chunk": self.prefill_chunk, "roce_ring": self.roce_ring, "roce_chain": self.roce_chain, "context_parallel": self.context_parallel,
            "stream_overlap": self.stream_overlap,
            "mixed_prefill": self.mixed_prefill,
            "prompt_logprobs": self.prompt_logprobs,
            "expert_scale_compression": self.expert_scale_compression,
            "mla_compute_mode": self.mla_compute_mode,
            "compile_workers": self.compile_workers,
            "autotune_effort": self.autotune_effort,
            "tuning_gpus": [gpu.index for gpu in self.tuning_gpus],
            "speculative_token_capacity": self.speculative_token_capacity,
            "online_mxfp8": self.online_mxfp8, "language_model_only": self.language_model_only,
            "speculation": None if self.speculation is None else {
                "draft_tokens": self.speculation.draft_tokens, "adaptive": self.speculation.adaptive,
                "packed_verification": self.speculation.packed_verification,
                "draft_prefix_selection": self.speculation.draft_prefix_selection,
                "draft_head_quantization": self.speculation.draft_head_quantization,
                "markov_head_quantization": self.speculation.markov_head_quantization,
                **({"graph_budget": self.speculation.graph_budget}
                   if self.speculation.graph_budget is not None else {})},
            "context": self.context, "checkpoint_context": self.recipe.context,
            "weights_bytes": self.weights_bytes, "mapped_host_bytes": self.checkpoint.mapped_host_bytes,
            "graph_bytes": self.graph_bytes,
            "mixed_shapes": self.mixed_shapes,
            "live_bytes": self.live_bytes, "prefix_cache_states": self.cache_states,
            "prefix_cache_target": self.cache_target,
            "snapshot_virtual_bytes": self.snapshots * self.costs.state_row_bytes,
            "snapshots": self.snapshots,
            "pages": self.pages, "demand_pages": self.demand_pages, "page_tokens": self.page_tokens,
            "page_bytes": self.page_bytes, "tokens": self.tokens, "guaranteed_tokens": self.guaranteed_tokens,
            "free_bytes": self.free_bytes, "headroom_bytes": self.headroom_bytes, "reserve_bytes": self.reserve_bytes,
            "supply_bytes": self.supply_bytes,
        }


def fit(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *, concurrency: int, reserve_bytes: int,
        speculation: Speculation | None, prefill_chunk: int | None = None,
        context_parallel: bool = True, stream_overlap: bool = True, cache_states: int | None = None,
        mixed_prefill: bool = False, prompt_logprobs: bool = False,
        expert_scale_compression: str | None = None, roce_ring: bool = False,
        roce_chain: bool = False,
        image_concurrency: int | None = None, online_mxfp8: bool = False,
        language_model_only: bool = False, mla_compute_mode: str | None = None) -> Plan:
    """Size pages and the prefix cache to the supply; the result may not fit.

    Weights come from the checkpoint's stored tensors and graphs and scratch from
    the recipe's estimate.
    """
    if mla_compute_mode is None:
        mla_compute_mode = recipe.mla_compute_mode
    if expert_scale_compression is None:
        expert_scale_compression = recipe.expert_scale_compression
    if expert_scale_compression == "native":
        expert_scale_compression = None
    if mla_compute_mode not in recipe.mla_compute_modes:
        raise UsageError(f"--mla-compute-mode {mla_compute_mode}: supported values for this recipe are "
                         + ", ".join(recipe.mla_compute_modes))
    if speculation is not None:
        for head in ("draft", "markov"):
            value = getattr(speculation, f"{head}_head_quantization")
            if value is not None and value not in getattr(recipe, f"{head}_head_quantizations"):
                raise UsageError(f"--{head}-head-quantization {value}: {checkpoint.alias}"
                                 " does not prepare this proposal head format")
    if expert_scale_compression is not None and expert_scale_compression not in checkpoint.scale_compressed_device_bytes:
        raise UsageError(f"--expert-scale-compression {expert_scale_compression}: {checkpoint.alias}"
                         " does not prepare this scale encoding")
    if (speculation is not None and speculation.packed_verification and speculation.adaptive
            and speculation.draft_tokens > 0
            and speculation.graph_budget is None and recipe.speculation is not None):
        speculation = replace(speculation, graph_budget=recipe.speculation.graph_budget)
    if speculation is not None and speculation.graph_budget is not None:
        from ignition.components.capacity import packed_speculative_shapes

        try:
            packed_speculative_shapes(concurrency, speculation.draft_tokens + 1,
                                      speculation.token_capacity, speculation.graph_budget)
        except ValueError as error:
            raise UsageError(f"--speculative-graph-budget: {error}") from error
    costs = recipe.costs(len(gpus), speculation, checkpoint)
    plan = Plan(recipe, checkpoint, gpus, concurrency, recipe.prefill_chunk if prefill_chunk is None else prefill_chunk,
                speculation, 0, 0, 0, reserve_bytes,
                recipe.weights_bytes(checkpoint, len(gpus), speculation, expert_scale_compression), costs.graph_bytes,
                context_parallel=context_parallel and costs.page_tokens is not None,
                stream_overlap=stream_overlap and recipe.stream_overlap, mixed_prefill=mixed_prefill, prompt_logprobs=prompt_logprobs,
                expert_scale_compression=expert_scale_compression, roce_ring=roce_ring, roce_chain=roce_chain,
                mla_compute_mode=mla_compute_mode,
                # Language-only plans prepare no vision encoder, so no image feature slots.
                image_concurrency=0 if language_model_only else (min(4, concurrency) if recipe.max_images else 0)
                    if image_concurrency is None else image_concurrency, online_mxfp8=online_mxfp8,
                language_model_only=language_model_only)
    available = plan.supply_bytes - plan.fixed_bytes
    target = min(CACHE_STATES_PER_REQUEST * concurrency, max(1, available // costs.state_row_bytes))
    if cache_states is not None:
        target = cache_states

    return replace(plan, cache_target=target, cache_states=target,
                   pages=max(available // costs.page_bytes, plan.minimum_pages))


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f}"


def _per_gpu(plan: Plan) -> str:
    return " per GPU" if plan.world_size > 1 else ""


def _shortfall(plan: Plan, options: list[str], scope: str) -> CapacityError:
    need = plan.fixed_bytes + plan.page_bytes
    lines = [
        f"needs {_gib(plan.weights_bytes)} GiB weights + {_gib(plan.graph_bytes)} GiB graphs"
        f" + {_gib(plan.live_bytes)} GiB live state",
        f"+ {_gib(plan.page_bytes)} GiB minimum pages ({plan.concurrency} × {MINIMUM_REQUEST_TOKENS:,} tokens)"
        f" = {_gib(need)} GiB{_per_gpu(plan)};",
        f"has {_gib(plan.free_bytes)} GiB free − {_gib(plan.headroom_bytes)} GiB headroom"
        f" − {_gib(plan.reserve_bytes)} GiB reserve = {_gib(plan.supply_bytes)} GiB.",
        f"Options: {', or '.join(options)}." if options
        else f"No setting makes {plan.checkpoint.alias} fit on {scope}.",
    ]
    return CapacityError("\n".join(lines))



def host_shape_error(roce_ring: bool, roce_chain: bool, hosts: int | None) -> str | None:
    """Why a ring or chain cannot be formed over ``hosts`` ranks on separate hosts, or None when it can.

    A ring or a chain needs 2 to 16 ranks in rank order on separate hosts; the two shapes exclude each
    other. ``hosts`` is the rank count, or None when ranks share one host."""
    if roce_ring and roce_chain:
        return "--roce-ring and --roce-chain exclude each other"
    if not (roce_ring or roce_chain):
        return None
    flag = "--roce-ring" if roce_ring else "--roce-chain"
    if hosts is None or not 2 <= hosts <= 16:
        return f"{flag} requires at least two hosts in rank order, at most 16, each rank on its own host"
    return None

def plan(recipe: Recipe, checkpoint: Checkpoint, gpus: tuple[Gpu, ...], *, selected: tuple[int, ...] | None,
         concurrency: int, reserve_bytes: int, speculation: Speculation | None,
         prefill_chunk: int | None = None, context_parallel: bool = True, stream_overlap: bool = True,
         cache_states: int | None = None, mixed_prefill: bool = False, prompt_logprobs: bool = False,
         expert_scale_compression: str | None = None, roce_ring: bool = False, roce_chain: bool = False,
         image_concurrency: int | None = None, online_mxfp8: bool = False,
         language_model_only: bool = False, mla_compute_mode: str | None = None) -> Plan:
    """Select GPUs and size the plan, or raise a capacity error that lists working settings.

    Without `selected`, the plan uses the smallest supported tensor-parallel size
    for which enough GPUs fit, taking the first such GPUs in index order. Without
    `prefill_chunk`, the recipe's chunk applies. `speculation` is the drafting the
    plan prepares, or None for none. `context_parallel` shards each page's context
    across the ranks of recipes that declare context-parallel page costs. Without
    `stream_overlap`, recipes that declare stream overlap run every branch serially.
    """
    if mixed_prefill and not recipe.mixed_prefill:
        raise UsageError(f"--mixed-prefill: {checkpoint.alias} does not capture mixed prefill/decode graphs")
    if online_mxfp8 and not recipe.online_mxfp8:
        raise UsageError(f"--online-mxfp8: {checkpoint.alias} names no projections to store as MXFP8")
    if language_model_only and not recipe.language_model_only:
        raise UsageError(f"--language-model-only: {checkpoint.alias} prepares no separable vision encoder")
    if not 1 <= concurrency <= recipe.max_concurrency:
        raise UsageError(f"--max-concurrency for {checkpoint.alias} must be 1 through {recipe.max_concurrency}")
    if image_concurrency is not None and (not recipe.max_images or type(image_concurrency) is not int
                                         or not 1 <= image_concurrency <= concurrency):
        raise UsageError("--image-concurrency requires a vision model and must be 1 through max concurrency")
    if language_model_only and image_concurrency is not None:
        raise UsageError("--image-concurrency requires the vision encoder that --language-model-only leaves unloaded")
    chunk = recipe.prefill_chunk if prefill_chunk is None else prefill_chunk
    if chunk % 64 or not 64 <= chunk <= recipe.context:
        raise UsageError(f"--prefill-chunk-size for {checkpoint.alias} must be a multiple of 64"
                         f" from 64 through {recipe.context:,}")
    sizes = " or ".join(map(str, recipe.tensor_parallel))
    supported = [gpu for gpu in gpus if gpu.supported]
    chosen = None
    if selected is not None:
        if any(index >= len(gpus) for index in selected):
            raise UsageError(f"--gpus {format_gpus(selected)} names a GPU this host does not have;"
                             f" it has {len(gpus)}")
        chosen = tuple(gpus[index] for index in selected)
        if len(chosen) not in recipe.tensor_parallel:
            size = min(recipe.tensor_parallel, key=lambda size: (size < len(chosen), size))
            first = max(0, min(selected[0], len(gpus) - size))
            example = format_gpus(range(first, first + size))
            raise CapacityError(f"{checkpoint.alias} needs {sizes} GPUs; {len(chosen)} selected; use --gpus {example}")
        for gpu in chosen:
            if not gpu.supported:
                raise CapacityError(f"GPU {gpu.index} ({gpu.name}) is not an {SUPPORTED_ARCHITECTURES} device")
    elif len(supported) < recipe.tensor_parallel[0]:
        raise CapacityError(f"{checkpoint.alias} needs {sizes} {SUPPORTED_ARCHITECTURES} GPUs; this host has {len(supported)}")

    def fitted(group, *, concurrency=concurrency, reserve_bytes=reserve_bytes, speculation=speculation):
        return fit(recipe, checkpoint, group, concurrency=concurrency, reserve_bytes=reserve_bytes,
                   speculation=speculation, prefill_chunk=chunk, context_parallel=context_parallel,
                   stream_overlap=stream_overlap, cache_states=cache_states,
                   mixed_prefill=mixed_prefill, prompt_logprobs=prompt_logprobs,
                   expert_scale_compression=expert_scale_compression, roce_ring=roce_ring, roce_chain=roce_chain,
                   image_concurrency=None if image_concurrency is None else min(image_concurrency, concurrency),
                   online_mxfp8=online_mxfp8, language_model_only=language_model_only,
                   mla_compute_mode=mla_compute_mode)

    def automatic(world_size, **settings):
        """Plan the first GPUs that each fit, else the GPUs with the most free memory."""
        fitting = [gpu for gpu in supported if fitted((gpu,) * world_size, **settings).fits]
        if len(fitting) < world_size:
            fitting = sorted(sorted(supported, key=lambda gpu: -gpu.free_bytes)[:world_size],
                             key=lambda gpu: gpu.index)
        return fitted(tuple(fitting[:world_size]), **settings)

    def candidates(**settings):
        if chosen is not None:
            return [fitted(chosen, **settings)]
        return [automatic(size, **settings) for size in recipe.tensor_parallel if size <= len(supported)]

    def attempt(**settings):
        return next((candidate for candidate in candidates(**settings) if candidate.fits), None)

    result = attempt()
    if result is not None:
        if chunk > result.context:
            raise UsageError(f"--prefill-chunk-size {chunk:,} exceeds the {result.context:,}-token context"
                             " this plan allows")
        return result

    closest = min(candidates(), key=lambda candidate: (
        candidate.fixed_bytes + candidate.page_bytes - candidate.supply_bytes))
    options = []
    if reserve_bytes > 0:
        for reserve in range((reserve_bytes - 1) // _GIB, -1, -1):
            if attempt(reserve_bytes=reserve * _GIB) is not None:
                options.append(f'[memory] reserve = "{reserve}G"' if reserve else '[memory] reserve = "0"')
                break
    for limit in range(concurrency - 1, 0, -1):
        if attempt(concurrency=limit) is not None:
            options.append(f"--max-concurrency {limit}")
            break
    if speculation is not None and attempt(speculation=None) is not None:
        options.append("--no-mtp")
    if chosen is not None:
        for size in recipe.tensor_parallel:
            if len(chosen) < size <= len(supported) and (larger := automatic(size)).fits:
                options.append(f"--gpus {format_gpus(gpu.index for gpu in larger.gpus)}")
                break
    scope = "this host" if chosen is None else f"GPU {format_gpus(gpu.index for gpu in chosen)}"
    raise _shortfall(closest, options, scope)
