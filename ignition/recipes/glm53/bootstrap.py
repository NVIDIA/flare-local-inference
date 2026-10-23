"""Prepare a GLM tensor-parallel rank before native serving takes ownership."""

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import speculative_token_capacity, batch_capacities, prefill_capacities, request_capacity
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from .contract import prepare as prepare_contract
from ignition.preparation.contract import history_region
from ignition.preparation import runtime_package_pb2 as contract_proto
from ignition.preparation.deferred import deferred_mode
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
    create_head_exchange,
    create_roce_runtime,
)
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import ScratchArena
from ignition.progress import phase
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica

from .recipe import RECIPE
from .batch import BatchInputs, PreparedBatch
from .generation import PreparedGeneration
from .maintenance import PreparedMaintenance
from .pool import ContinuationPool, continuation_layout
from .resume import PreparedResume
from .stack import DecoderScratch, PreparedDecoder
from .state import DecoderMetadata
from .weights import ModelWeights
from .vision import PreparedVision, VisionMetadata, VisionScratch
from .vision_state import FEATURES, PATCH_CAPACITIES, PreparedVisionGeneration, VisionInputs
from ignition.components.vision_state import PrefillVisionInputs, VisionState


_COMPILED_KINDS = ("prefill", "scored_prefill", "decode", "resume", "speculative", "vision")
# Prefill chunks of at least this many tokens all-reduce through the copy-engine ring.
_DMA_PREFILL_TOKENS = 1024
# Allocator granularity and fragmentation slack beyond the headroom kept after the final KV allocation.
_KV_ALLOCATION_MARGIN = 64 << 20


@dataclass(frozen=True)
class PreparationConfig:
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    autotune_effort: str = "medium"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = (0, 1)
    # Each rank's host when ranks run on separate hosts; devices are then host-local ordinals.
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    image_concurrency: int | None = None
    language_model_only: bool = False
    pages: int | None = None
    # Without fixed pages, KV pages fill the memory measured free after graph capture, at least this many.
    minimum_pages: int | None = None
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    mtp: bool = False
    # SM103 uses checkpoint-scaled A4; SM120/121 retain A16 with A4 prefill launches.
    expert_activation: str = "auto"
    # Shard latent records and C4 pools across the tensor-parallel ranks (decode context parallelism).
    context_parallel: bool = False
    # Expand routed experts' CSF block scales once at load instead of before every expert launch.
    expand_expert_scales: bool = False
    # Run shared experts, KDA gate projections, and decode query chains on an auxiliary stream;
    # without overlap every branch runs serially on the main stream.
    stream_overlap: bool = True
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None

    def __post_init__(self):
        if self.expert_activation not in ("auto", "a4", "a16"):
            raise ValueError("GLM expert activation must be auto, a4, or a16")
        from ignition.planning import host_shape_error

        error = host_shape_error(self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts))
        if error:
            raise ValueError(error)
        if type(self.mtp) is not bool:
            raise TypeError("mtp must be a boolean")
        self.speculative_shapes
        if self.speculative_fixed_width is not None and (
                type(self.speculative_fixed_width) is not int
                or self.speculative_fixed_width not in self.speculative_widths):
            raise ValueError("fixed speculative width must belong to the configured widths")
        if type(self.compile_decoder) is not bool or type(self.stream_overlap) is not bool:
            raise TypeError("compile_decoder and stream_overlap must be booleans")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (not isinstance(self.devices, tuple) or len(self.devices) not in (1, 2, 4)
                or len(set(placements)) != len(self.devices)
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("GLM requires one, two or four distinct CUDA devices")
        for name in ("context", "prefill_chunk_size", "max_active", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not self.page_tokens <= self.context <= 1048576 or self.context % self.page_tokens:
            raise ValueError(f"GLM context must be {self.page_tokens}-aligned in [{self.page_tokens},1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        if type(self.language_model_only) is not bool:
            raise TypeError("language_model_only must be a boolean")
        if not self.language_model_only:
            from ignition.components.capacity import image_capacity

            object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if self.pages is not None and (type(self.pages) is not int or not 0 < self.pages <= 2147483647):
            raise ValueError("page capacity must be a positive signed 32-bit integer")
        if self.minimum_pages is not None and (type(self.minimum_pages) is not int or self.minimum_pages <= 0
                                               or self.pages is not None and self.minimum_pages > self.pages):
            raise ValueError("minimum page capacity must be positive and at most the fixed page capacity")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)

    @property
    def token_capacity(self):
        return max(self.prefill_chunk_size, self.request_capacity,
                   max((b * w for b, w in self.speculative_shapes), default=0) if self.mtp else 0)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def page_capacity(self):
        """Fixed KV pages, or the context of every active request when memory does not size them."""
        if self.pages is not None:
            return self.pages
        return self.minimum_pages or (self.context // self.page_tokens + int(self.mtp)) * self.max_active

    @property
    def page_tokens(self):
        """Global positions per KV page: 256 local records on each rank that shares the context."""
        return 256 * (len(self.devices) if self.context_parallel else 1)

    @property
    def preparation_pages(self):
        """Pages held while tuning and measuring graphs; the final pools replace them before capture."""
        return min(self.page_capacity, 2 * self.request_capacity)


def _tensor_bytes(*owners):
    """Bytes of the distinct CUDA storages that owners and their nested fields reference."""
    storages, visited, pending = {}, set(), list(owners)
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        if isinstance(item, torch.Tensor):
            if item.device.type == "cuda":
                storage = item.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif hasattr(item, "__dataclass_fields__"):
            pending.extend(vars(item).values())
    return sum(storages.values())


def _memory_breakdown(device, *, weights, continuation, kv, operator_scratch, decoder_scratch, collective, headroom,
                      compiled_reservation):
    """Where one rank's memory goes once serving graphs are captured."""
    import resource

    from ignition.preparation.host_memory import device_memory_info

    allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    region = getattr(collective, "_region", None)
    named = weights + continuation + kv + operator_scratch + decoder_scratch
    return {"weights": weights, "continuation": continuation, "kv": kv, "operator_scratch": operator_scratch,
            "decoder_scratch": decoder_scratch, "other_allocated": allocated - named,
            "cached_unallocated": reserved - allocated,
            "collective_pinned_host": 0 if region is None else region.numel() * region.element_size(),
            "compiled_forward_reservation": compiled_reservation, "headroom": headroom,
            "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss << 10,
            "available": device_memory_info(device)[0], "cuda_free": torch.cuda.mem_get_info(device)[0]}


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot: Path, config: PreparationConfig, *, rank, group, progress=None, package_storage=None) -> PreparedModel:
    """Load, tune and capture vision and text variants on one supervised TP rank.

    The caller must terminate the complete rank group if any preparation fails.
    Successful cleanup requires every rank to close its IPC runtime together,
    after native replay drains. No eager execution path is exported. Package
    storage backs device memory, collective mappings and graphs with addresses
    that native restore reproduces.
    """
    def report(message):
        if progress is not None:
            progress(message)

    def memory(stage):
        free = torch.cuda.mem_get_info(device)[0]
        with open("/proc/self/statm") as statm:
            rss = int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        report(f"{stage}: free={free >> 20} MiB, allocated={torch.cuda.memory_allocated(device) >> 20} MiB, "
               f"reserved={torch.cuda.memory_reserved(device) >> 20} MiB, rss={rss >> 20} MiB")

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    if (shard.world_size > 1) != (group is not None):
        raise ValueError("a GLM rank group exists exactly when more than one rank prepares")
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    from ignition.components.sm103 import is_sm103

    expert_activation = config.expert_activation
    if expert_activation == "auto":
        expert_activation = "a4" if is_sm103(device) else "a16"
    if expert_activation == "a16":
        # b12x retains activation scales for A4 prefill launches only while preparing with this switch.
        os.environ["B12X_W4A16_A4_PREFILL"] = "1"
    # Loading, construction and tuning free most of what they allocate; a transient package pool returns
    # that memory to the device once tuning completes, before graph measurement sizes the KV pages.
    transient = contextlib.ExitStack()
    if package_storage is not None:
        transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, mtp=config.mtp,
                                expert_activation=expert_activation,
                                expand_expert_scales=config.expand_expert_scales, language_model_only=config.language_model_only)
    if torch.cuda.get_device_properties(device).is_integrated:
        from ignition.preparation.host_memory import release_page_cache

        release_page_cache(manifest.shard_paths)
    memory("after weights")
    with phase(progress, "Allocate model state and collectives") as stage:
        layer_types = weights.tree.config.text_config.layer_types
        layout = continuation_layout(layer_types, shard.world_size, mtp=config.mtp)
        # Native loading sizes a package's continuation rows and KV pages from the serving host's
        # memory, within their virtual extents. A package backs only what its tuning and capture use.
        snapshots = 1 if package_storage is not None else layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("GLM snapshot budget must hold at least one continuation row")
        pages, blocks = config.preparation_pages, config.context // config.page_tokens
        slots = config.request_capacity
        # Speculative verification records its tokens and commits into the live row: one row per request.
        state_rows = snapshots + slots
        page_bytes = (layer_types.count("indexed_attention") + int(config.mtp)) * (4 * 64 * 528 + 64 * 132)
        max_cache_pages = minimum_across_ranks(torch.cuda.mem_get_info(device)[1] // page_bytes, group)
        context_ranks = shard.world_size if config.context_parallel and group is not None else 1
        # Compiled forwards bind the page pools' shapes. Give in-process pools a package's
        # full virtual extent, backed by measured pages only after compilation. Context
        # ranks share their latent caches through CUDA IPC allocations instead.
        virtual_pages = None
        if package_storage is None and context_ranks == 1:
            from ignition.preparation.virtual_storage import VirtualStorage

            virtual_pages = VirtualStorage(device)
        pool = ContinuationPool.allocate(layer_types, shard.world_size,
            state_rows + 1, pages, device, mtp=config.mtp, storage=package_storage,
            maximum_pages=max_cache_pages, context_ranks=context_ranks, virtual=virtual_pages)
        requests = DeviceRequestState.allocate(device, vocab_size=154880, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=154880, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.mtp else 8), device=device)
        # One rank holds whole sums and vocabularies: it prepares no collectives.
        coordinator = CollectivePreparation(group) if group is not None else None
        ipc = None if package_storage is None else package_storage.peer_storage
        collective = prefill_collective = vocabulary_collective = None
        sample_rows = max(b * w for b, w in config.speculative_shapes) if config.mtp else config.request_capacity
        if coordinator is not None and coordinator.spans_hosts:
            # Ranks on separate Sparks reduce and gather through one RoCEnante runtime, which also
            # carries decode context parallelism's exchanges.
            # Decode rows reduce one-shot; prefill chunks through the pipelined two-shot path.
            decode_tokens = max(config.request_capacity, sample_rows)
            gather_bytes = sample_rows * (154880 // shard.world_size) * 2
            if config.context_parallel:
                from ignition.components.head_exchange import pack_width

                from .dcp import SLICE_ROWS

                # Per-rank contributions: query heads and the merge's head shards for one exchange
                # slice, and index candidates for one slice of rows.
                exchange = min(SLICE_ROWS, decode_tokens)
                gather_bytes = max(gather_bytes, exchange * (64 // shard.world_size) * 512 * 2,
                                   exchange * pack_width(shard.world_size, 64, 512) * 2,
                                   min(config.token_capacity, SLICE_ROWS) * 2048 * 2)
            varlen_bytes = 0
            if config.context_parallel:
                from .dcp import WINDOW_RECORDS

                # Prefill gathers each rank's latent records one window at a time.
                varlen_bytes = min(WINDOW_RECORDS, blocks * 256) * 528
            collective = prefill_collective = vocabulary_collective = create_roce_runtime(
                group, device, max_tokens=decode_tokens, pipe_tokens=config.token_capacity, hidden_size=4096,
                gather_bytes=gather_bytes, ring=config.roce_ring, ring_open=config.roce_chain, package_storage=package_storage, varlen_bytes=varlen_bytes)
        elif group is not None:
            collective = create_all_reduce_runtime(group, device, max_tokens=config.token_capacity,
                                                   hidden_size=4096, ipc=ipc)
            prefill_collective = create_dma_all_reduce_runtime(group, device, max_tokens=config.prefill_chunk_size,
                                                               hidden_size=4096, ipc=ipc)
            vocabulary_collective = create_all_gather_runtime(group, device,
                max_rows=sample_rows, columns=154880, ipc=ipc)
        identity = hashlib.sha256(json.dumps({"config": json.loads((snapshot / "config.json").read_text()),
            "headers": [s.header_sha256 for s in manifest.shards], "world_size": shard.world_size,
            "contract": "glm5next.graph-inputs.v3", "vision": not config.language_model_only}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity, workload_schema="glm-text-stack-tp-v1",
            numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.decoder.layers[0].recurrent,
            storage=pool.backing, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        forwards, signatures, decoders, scratch, metadata = {}, {}, {}, {}, {}
        buffers = dict(inputs.buffers(), **commands.buffers(), **{
            name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity, sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
        vision_state = vision_inputs = vision_prefill = vision_storage = None
        vision_owners = []
        if weights.vision is not None:
            vision_inputs = VisionInputs.allocate(device)
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(inputs, vision_prefill, requests, image_token=154854,
                features=FEATURES, hidden=4096, image_slots=config.image_concurrency)
            vision_storage = VisionScratch.allocate(max(PATCH_CAPACITIES), device)
            buffers.update(vision_inputs.buffers(), **vision_prefill.buffers())
        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        context = None
        if config.context_parallel and group is not None:
            from .dcp import SLICE_ROWS, ContextParallel, ContextShards

            owned = ContextShards.on(rank, shard.world_size, device)
            if coordinator.spans_hosts:
                # The ranks' one RoCEnante runtime carries every context-parallel exchange.
                context = ContextParallel(owned, heads=collective, candidates=collective, coordinator=coordinator,
                                          remote=True)
            else:
                # Only decode exchanges heads; prefill gathers context blocks instead.
                exchange_rows = min(SLICE_ROWS, max(config.request_capacity,
                    max((b * w for b, w in config.speculative_shapes), default=1) if config.mtp else 1))
                context = ContextParallel(owned,
                    heads=create_head_exchange(group, device, max_rows=exchange_rows, heads=64, head_dim=512,
                                               ipc=ipc),
                    candidates=create_all_gather_runtime(group, device,
                        max_rows=min(config.token_capacity, SLICE_ROWS), columns=2048 * shard.world_size, ipc=ipc),
                    coordinator=coordinator)
        shards = None if context is None else context.shards
        # Prefill chunks overlay their layer intermediates; decode rows keep small separate scratch.
        remote = context is not None and context.remote
        prefill_storage = DecoderScratch.allocate(config.prefill_chunk_size, blocks, "prefill", device,
            shard=shard, sample_rows=1, context_ranks=context_ranks, overlay_layers=True, remote=remote)
        decode_rows = max(config.request_capacity,
                          max((b * w for b, w in config.speculative_shapes), default=1) if config.mtp else 1)
        decoder_storage = DecoderScratch.allocate(decode_rows, blocks, "decode", device,
            shard=shard, sample_rows=sample_rows, context_ranks=context_ranks, remote=remote)

        if vision_storage is not None:
            from ignition.components.moe import RouterScratch
            from ignition.preparation.scratch import overlay
            from .indexer import IndexScratch

            # Vision and text graphs replay serially; index and router constants retain their storage.
            prefill_storage, vision_storage = overlay((prefill_storage, vision_storage),
                                                      keep=(IndexScratch, RouterScratch))
        speculation = None
        if config.mtp:
            from .speculative import SpeculativePreparation

            speculation = SpeculativePreparation(weights, pool, requests, inputs, decoder_storage, prefill_storage,
                session=session, workspace=arena, completion=completion, frame=frame,
                sequence=sequence, runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                max_cache_pages=max_cache_pages, prefill_tokens=config.prefill_chunk_size,
                decode_tokens=decode_rows, context=context)
            buffers["draft_page_tables"] = speculation.draft_tables
            forward_inputs += ("draft_page_tables",)

    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    serving_modes = (True, False) if config.mtp else (False,)
    graph_count = (len(serving_modes) * (prefill_graph_count(config.prefill_capacities, scoring)
                           + 4 * len(config.batch_capacities))
                   + len(config.batch_capacities) + (len(config.speculative_shapes) if config.mtp else 0))
    graph_count += len(PATCH_CAPACITIES) if vision_state is not None else 0
    memory(f"after model state (continuation {pool.backing.nbytes >> 20} MiB, {state_rows} rows)")
    with phase(progress, "Construct graph catalog", total=graph_count) as stage:
        def add(name, function, kind, rows, tokens, fields, *, enabled=None):
            if kind == "scored_prefill":
                function = PreparedScoredPrefill(function, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = function
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        if vision_state is not None:
            from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

            for capacity in reversed(PATCH_CAPACITIES):
                name = f"vision{capacity}"
                storage = vision_storage.rows(capacity)
                vision_metadata = VisionMetadata.allocate(capacity, 1, device)
                with torch.device("meta"):
                    module = Glm5NextVisionModel(weights.tree.config.vision_config)
                module.eval().requires_grad_(False)
                encoder = PreparedVision(module, storage, vision_metadata, weights.vision,
                    session=session, name=name, projection_workspace=arena, attention_workspace=arena)
                forward = PreparedVisionGeneration(encoder, vision_inputs, vision_state, completion, frame, sequence)
                add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
                vision_owners.extend((vision_metadata, encoder, forward))
        for enabled in serving_modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_inputs = tuple(field for field in forward_inputs if enabled or field != "draft_page_tables")
            for kind, mode, capacities in prefill_modes(
                    config.prefill_capacities, () if enabled else config.batch_capacities, scoring):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata[name] = DecoderMetadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    scratch[name] = storage.rows(capacity, mode, sample_rows=rows)
                    feedback = (torch.empty((capacity, 4096), dtype=torch.bfloat16, device=device)
                                if enabled and kind != "scored_prefill" else None)
                    decoders[name] = PreparedDecoder(weights, scratch[name], pool.decoder, metadata[name], mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        runtime=prefill_collective if mode == "prefill" and capacity >= _DMA_PREFILL_TOKENS else collective,
                        vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                        session=session, name=name, workspace=arena, feedback=feedback,
                        max_cache_pages=max_cache_pages, context=context, vision=vision_state)
                    batch = PreparedBatch(inputs, requests, metadata[name], mode=mode, shard=shard,
                                          pages=pool.memory_capacity[1:], shards=shards,
                        validate_prefill=vision_state.validate_prefill if vision_state is not None and mode == "prefill" else None)
                    forward = PreparedGeneration(decoders[name], batch, completion, frame, sequence,
                        pool=pool, runtime=collective, coordinator=coordinator, session=session, name=f"{name}.generation", replicated_sampling=enabled)
                    if enabled:
                        from .speculative import PreparedPrefill

                        forward = PreparedPrefill(speculation, forward, decoders[name].feedback, name)
                    fields = (*mode_inputs, "vision_token_indices", "vision_feature_slot") if mode == "prefill" and vision_state is not None else mode_inputs
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, fields, enabled=enabled)
        if speculation is not None:
            from .speculative import PreparedRound

            for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
                name = f"speculative{batch}x{width}"
                operation = PreparedRound(speculation, batch, width, name)
                operation.prepare()
                add(name, operation, "speculative", batch, width, forward_inputs, enabled=True)

        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in serving_modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                resume = PreparedResume(weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, replicated_sampling=enabled)
                add(name, resume, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    operation = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name,
                        capacity=capacity, admission_buffers=admission,
                        draft_tables=speculation.draft_tables if enabled else None,
                        page_tokens=config.page_tokens)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, operation, mode, capacity, 1, (*fields, "page_tables", *(("draft_page_tables",) if enabled else ()), *commands.buffers()), enabled=enabled)
    memory(f"after graph catalog (operator scratch {arena.nbytes >> 20} MiB)")
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    report(f"selected kernels: {tuning['selection_counts']}")
    report(f"after tuning: free={torch.cuda.mem_get_info(device)[0] >> 20} MiB, "
           f"allocated={torch.cuda.memory_allocated(device) >> 20} MiB, reserved={torch.cuda.memory_reserved(device) >> 20} MiB, "
           f"weights={weights.allocated_bytes >> 20} MiB, continuation={pool.backing.nbytes >> 20} MiB, "
           f"operator scratch={arena.nbytes >> 20} MiB")
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle() if package_storage is None else package_storage.graph_pool.id
    trials, _ = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
        compile_decoder=False, progress=progress, label="Measure graph memory", warmup_storage=package_storage)
    from ignition.preparation.host_memory import COMPILED_FORWARD_BYTES, device_memory_info

    # An integrated GPU's allocatable memory includes the reclaimable RAM CUDA's free memory excludes.
    integrated = torch.cuda.get_device_properties(device).is_integrated
    binds_pages = package_storage is None and virtual_pages is None
    if package_storage is not None:
        # Native loading checks the minimum KV capacity against the serving host's memory.
        pages = config.preparation_pages
        memory(f"after graph measurement (the package backs {pages} preparation KV pages)")
    else:
        free = device_memory_info(device)[0]
        if integrated and config.compile_decoder:
            # Restored compiled forwards stay in this process's memory, which KV pages share.
            free -= sum(signature.kind in _COMPILED_KINDS for signature in signatures.values()) * COMPILED_FORWARD_BYTES
        if virtual_pages is not None:
            # The preparation pages stay backed and become the first of the grown pools' pages;
            # growing rounds each pool up to the allocation granularity.
            free += virtual_pages.nbytes - len(virtual_pages.dynamic_regions) * virtual_pages.granularity
        available = minimum_across_ranks(free - config.headroom_bytes - _KV_ALLOCATION_MARGIN, group)
        pages = available // page_bytes if config.pages is None else config.pages
        if integrated and binds_pages and config.pages is None and pages > 0:
            # Shared system memory varies between launches; a coarser page count keeps compiled
            # forwards, which bind the page pool's shape, reusable across them.
            step = 1 << max(0, (pages // 8).bit_length() - 1)
            pages = pages // step * step
        memory(f"after graph measurement (KV may use {max(0, available) >> 20} MiB on every rank, {max(0, pages)} pages)")
        if pages < (config.minimum_pages or config.preparation_pages) or pages * page_bytes > available:
            raise MemoryError("prepared graph catalog leaves insufficient memory for requested KV pages and headroom")

    def rebind_pages():
        for forward in forwards.values():
            if isinstance(forward, PreparedScoredPrefill):
                forward = forward.forward
            if hasattr(forward, "rebind_pages"):
                forward.rebind_pages(pages)

    if config.compile_decoder and binds_pages:
        from b12x._lib.compile_plan import compile_only_launches

        with compile_only_launches():
            pool.replace_pages(pages, metadata_mode=deferred_mode(), ranks=context_ranks)
            for name, forward in forwards.items():
                if isinstance(forward, PreparedScoredPrefill):
                    forward = forward.forward
                if signatures[name].kind in _COMPILED_KINDS and hasattr(forward, "rebind_pages"):
                    forward.rebind_pages(pages)
    shared_pages = []

    def materialize_cache():
        with phase(progress, "Allocate KV cache") as stage:
            if package_storage is not None:
                package_storage.grow("KV_PAGES", pages)
                # Ranks on one host read each other's latent caches through CUDA IPC.
                if context_ranks > 1 and not remote:
                    pool.share_pages(package_storage, group)
            elif virtual_pages is not None:
                virtual_pages.grow("KV_PAGES", pages)
            else:
                from ignition.components.shared_pages import SharedPages

                shared_pages.extend(pool.replace_pages(pages, ranks=context_ranks,
                    share=None if remote else lambda nbytes: SharedPages(group, device, nbytes)))
                rebind_pages()
            pool.memory_capacity[1] = pages
            stage.update(f"{pages * page_bytes / (1 << 30):.2f} GiB per GPU · {pages * config.page_tokens:,} tokens")
            for graph in trials.values():
                graph.reset()
            trials.clear()
        memory("after KV allocation")

    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
        compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs, materialize=materialize_cache, warmup_storage=package_storage,
        compile_kinds=_COMPILED_KINDS)
    kv_bytes = pages * page_bytes
    if integrated:
        import resource

        memory(f"after capture (process peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10} MiB)")
    breakdown = _memory_breakdown(device, weights=weights.allocated_bytes, continuation=pool.backing.nbytes,
        kv=kv_bytes, operator_scratch=arena.nbytes, decoder_scratch=_tensor_bytes(prefill_storage, decoder_storage),
        collective=collective, headroom=config.headroom_bytes,
        compiled_reservation=(sum(signature.kind in _COMPILED_KINDS for signature in signatures.values())
                              * COMPILED_FORWARD_BYTES if integrated and config.compile_decoder and package_storage is None else 0))
    report("memory breakdown (MiB): " + ", ".join(f"{key}={value >> 20}" for key, value in breakdown.items()))
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    memory = {"rank": rank, "world_size": shard.world_size, "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes, "continuation_bytes": pool.backing.nbytes,
        "state_rows": state_rows, "pages": pages, "page_bytes": kv_bytes // pages, "kv_bytes": kv_bytes,
        "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": arena.nbytes, "graphs": len(graphs), "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free, "total_device_bytes": total, "headroom_bytes": config.headroom_bytes, "page_tokens": config.page_tokens,
        "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning, "compiled_decoder": config.compile_decoder, "breakdown": breakdown,
        "state_row_bytes": pool.backing.nbytes // max(1, state_rows + 1)}
    if scoring is not None:
        memory.update(scoring.memory_report())
    assets = {role: {"path": str(snapshot / filename)}
        for role, filename in (("tokenizer", "tokenizer.json"),)}
    assets["chat_template"] = annotated_template_asset(
        (snapshot / "chat_template.jinja").read_bytes(), Path(__file__).with_name("assets") / "chat_template.jinja")
    generation = json.loads((snapshot / "generation_config.json").read_text())
    from .mla import MLAState
    cache_regions = []
    for layer, value in enumerate((*pool.decoder.layers, *((pool.draft,) if pool.draft is not None else ()))):
        if isinstance(value, MLAState):
            cache_regions.append(history_region(buffers, f"cache.{layer}.latent", value.cache.view(-1, 4, 64, 528),
                contract_proto.PAGED_LATENT, tokens_per_record=config.page_tokens // 256))
            cache_regions.append(history_region(buffers, f"cache.{layer}.index", value.index.cache,
                contract_proto.PAGED_LATENT, tokens_per_record=config.page_tokens // 64))
    buffers["request_pending_token"] = requests.pending_token
    server_config = {"identity": identity, "model_id": snapshot.name, "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size, "pages": pages, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": config.page_tokens,
        "frontend": prepare_frontend(snapshot, eos_token_ids=generation["eos_token_id"],
            image_profile="glm" if vision_state is not None else None, preprocessor=snapshot / "processor_config.json"),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(), "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, page_tokens=config.page_tokens, draft_lag=1 if config.mtp else None, regions=cache_regions, image_slots=config.image_concurrency)}
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_shapes=config.speculative_shapes)
    resources = (scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, decoders,
        decoder_storage, prefill_storage, scratch, metadata, compiled, stream, graph_pool, session, collective,
        prefill_collective, vocabulary_collective, coordinator, speculation, context,
        vision_state, vision_inputs, vision_prefill, vision_storage, vision_owners)
    from .export import disposable_ranges

    return PreparedModel(rebuild_weights={"weights": weights}, device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, resources=resources, session=session,
        forwards={name: compiled.get(name, forward) for name, forward in forwards.items()},
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                            *disposable_ranges((decoder_storage, prefill_storage, vision_storage,
                                None if vision_state is None else vision_state.arena), arena, forwards), *completion.zero_ranges()),
        close_resources=(*((scoring,) if scoring is not None else ()),
                         *(() if context is None else (context.heads, context.candidates)), *shared_pages,
                         *(() if virtual_pages is None else (virtual_pages,)),
                         *(runtime for runtime in (prefill_collective, vocabulary_collective, collective)
                           if runtime is not None)),
        server_config=server_config, memory_report=memory)
