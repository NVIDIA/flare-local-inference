"""Prepare DeepSeek text weights, file-backed Engram tables, and native graph catalogs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
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
from .contract import prepare as prepare_contract, PATCH_CAPACITIES, INPUT_VOCABULARY
from ignition.components.deepseek_vision import (PreparedVision, VisionInputs, VisionScratch,
    PrefillVisionInputs, PreparedVisionGeneration, VisionState)
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
    create_all_reduce_runtime, create_dma_all_reduce_runtime, create_result_broadcast_runtime, create_roce_runtime)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import virtual_device_memory, release_unused
from ignition.preparation.frontend import PromptEncoder, prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from .recipe import RECIPE
from .attention import INDEXER_QUERY_ROWS
from .decoder import PreparedDecoder, Scratch
from ignition.components.dspark import PreparedContext, PreparedMarkov
from .dspark import PreparedDraft
from .engram import Tables
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from .pool import ContinuationPool, PAGE_TOKENS, continuation_layout, page_bytes
from ignition.components.paged_resume import PreparedResume
from .weights import ModelWeights


_CONTEXT = partial(PreparedContext, compact_prefill=True)


def _forward_factories(mla_compute_mode, *, window_staging, suffix_scratch, vision):
    """Bind one arithmetic contract to target, prefill, and draft attention."""
    return (partial(PreparedDecoder, window_staging=window_staging, suffix_scratch=suffix_scratch,
                    vision=vision, mla_compute_mode=mla_compute_mode),
            partial(PreparedDraft, mla_compute_mode=mla_compute_mode))


@dataclass(frozen=True)
class PreparationConfig:
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    autotune_effort: str = "medium"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = (0, 1, 2, 3)
    # Device ordinals are host-local when each rank runs on its own Spark.
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 32
    minimum_pages: int = 8
    # An explicit cache capacity makes memory comparisons independent of automatic sizing.
    kv_pages: int | None = None
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    dspark: bool = True
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = 128
    speculative_graph_budget: int = 64
    # Overlap shared and routed experts within each eligible text graph.
    stream_overlap: bool = True
    expert_scale_compression: str | None = None
    mla_compute_mode: str = "auto"
    split_speculation: bool | None = None
    packed_verification: bool | None = None
    draft_prefix_selection: str | None = None
    speculation_reference_graphs: bool = False

    def __post_init__(self):
        from ignition.components.capacity import image_capacity
        from ignition.components.compressed_mla import validate_mla_compute_mode

        validate_mla_compute_mode(self.mla_compute_mode)
        self.speculative_shapes
        self.packed_speculative_shapes
        drafting = self.dspark and self.draft_tokens > 0
        for name in ("split_speculation", "packed_verification"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, drafting)
            elif type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.draft_prefix_selection is None:
            object.__setattr__(self, "draft_prefix_selection", "gpu" if drafting else "cpu")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft prefix selection must be cpu or gpu")
        if drafting:
            if not self.split_speculation or not self.packed_verification or self.draft_prefix_selection != "gpu":
                raise ValueError("DeepSeek V4.1 DSpark requires packed GPU draft prefix selection")
            if self.speculative_fixed_width is not None or self.speculation_reference_graphs:
                raise ValueError("DeepSeek V4.1 DSpark requires adaptive packed GPU rounds")
        elif self.split_speculation or self.packed_verification or self.draft_prefix_selection != "cpu" or self.speculation_reference_graphs:
            raise ValueError("packed GPU speculation requires positive DSpark draft depth")
        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("DeepSeek V4.1 expert scale compression must be None or 'csf'")
        if type(self.stream_overlap) is not bool:
            raise TypeError("stream_overlap must be a boolean")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (not isinstance(self.devices, tuple) or len(self.devices) != 4 or len(set(placements)) != 4
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != 4)
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("DeepSeek V4.1 Flash preparation requires four distinct GPUs")
        from ignition.planning import host_shape_error

        error = host_shape_error(self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts))
        if error:
            raise ValueError(error)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("DeepSeek context must be 128-aligned in [128,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        self.speculative_shapes
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.minimum_pages <= 0 or self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("page and snapshot budgets must be positive and headroom nonnegative")
        if self.kv_pages is not None and (type(self.kv_pages) is not int or self.kv_pages < self.minimum_pages):
            raise ValueError("kv_pages must cover the minimum page capacity")

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return tuple((batch, width) for batch, width in
            speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
            if width == 1) if self.dspark else ()

    @property
    def packed_speculative_shapes(self):
        from ignition.components.capacity import packed_speculative_shapes

        return packed_speculative_shapes(self.max_active, max(self.speculative_widths),
            self.speculative_token_capacity, self.speculative_graph_budget) if self.draft_tokens else ()

    @property
    def decode_tokens(self):
        return max(self.request_capacity, max((tokens for _, tokens in self.packed_speculative_shapes), default=0))

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dspark else 0

    @property
    def all_reduce_tokens(self):
        return max(self.decode_tokens, self.request_capacity * self.draft_tokens,
                   *(capacity for capacity in self.prefill_capacities if capacity < 128))


def _prepare_speculation_catalog(config, speculation, *, add, buffers, graph_buffers,
                                 forward_inputs, host_allocate, device):
    if speculation is None:
        return None, None
    from ignition.components.parallel_draft import PreparedParallelDraftRound

    speculation.sampling_buffers = graph_buffers
    split = confidence = None
    if config.draft_tokens:
        from ignition.components.capacity import proposal_shapes
        from ignition.components.split_draft import (
            PreparedDraftProposal, PreparedDraftVerification, PreparedProposalStorage)
        from ignition.components.packed_speculation import PreparedPackedSpeculativeRound

        slots = config.request_capacity
        confidence = host_allocate("proposal_confidence", (slots, config.draft_tokens), torch.float32)
        buffers["proposal_confidence"] = confidence.device_view
        split = PreparedProposalStorage(speculation, config.draft_tokens)
        buffers.update(split.buffers())
        prefix_limits = torch.zeros(slots, dtype=torch.int32, device=device)
        buffers["speculative_prefix_limits"] = prefix_limits
        packed_inputs = tuple(field for field in forward_inputs if field != "query_lengths")
        packed_inputs += ("speculative_prefix_limits",)
        packed_batches = {batch for batch, _ in config.packed_speculative_shapes}
        proposals = {}
        for batch, depth in sorted(proposal_shapes(config.max_active, config.draft_tokens), reverse=True):
            if batch not in packed_batches:
                continue
            name = f"proposal{batch}x{depth + 1}"
            source = PreparedParallelDraftRound(speculation, batch, depth + 1, name,
                buffers=graph_buffers.cursor(), proposal_only=True)
            source.prepare()
            proposals[batch] = PreparedDraftProposal(source, confidence.device_view, split)
    for batch, _ in reversed(config.speculative_shapes):
        name = f"speculative{batch}x1"
        forward = PreparedParallelDraftRound(speculation, batch, 1, name,
            buffers=graph_buffers.cursor())
        forward.prepare()
        add(name, forward, "speculative", batch, 1, forward_inputs, enabled=True)
    for batch, packed_tokens in sorted(config.packed_speculative_shapes,
            key=lambda shape: (shape[1], shape[0]), reverse=True):
        logical_width = max(config.speculative_widths)
        target_name = f"verification{batch}x{logical_width}-tokens{packed_tokens}"
        target = PreparedParallelDraftRound(speculation, batch, logical_width, target_name,
            generate_proposals=False, buffers=graph_buffers.cursor(), target_token_capacity=packed_tokens)
        target.prepare()
        verification = PreparedDraftVerification(split, target)
        name = f"packed-speculative{batch}x{logical_width}-drafts{config.draft_tokens}-tokens{packed_tokens}"
        forward = PreparedPackedSpeculativeRound(proposals[batch], verification, prefix_limits,
            name=name, buffers=graph_buffers.cursor())
        add(name, forward, "packed_speculative", batch, logical_width, packed_inputs,
            enabled=True, packed_tokens=packed_tokens, draft_tokens=config.draft_tokens)
    return split, confidence


def _collective_runtimes(config, group, device, *, coordinator, package_storage, verification_samples):
    """Share one RoCE channel across ordered graph collectives on separate hosts."""
    tokens = max(config.prefill_chunk_size, config.decode_tokens, config.request_capacity * config.draft_tokens)
    sample_rows = max(verification_samples, config.request_capacity * config.draft_tokens)
    layouts = ()
    if config.dspark:
        from ignition.components.draft_sampling import PreparedDraftDistribution
        from ignition.components.speculation import SpeculativeAcceptance

        batch = max(config.batch_capacities)
        layouts = (SpeculativeAcceptance.broadcast_layout(batch, max(config.speculative_widths)),)
        if config.draft_tokens:
            layouts += (PreparedDraftDistribution.broadcast_layout(batch),)
    if coordinator.spans_hosts:
        from ignition.components.collectives import packed_broadcast_layout

        # Vocabulary partitions and packed sampling results use the same serial gather channel.
        result_bytes = max((packed_broadcast_layout(layout.values())[1] for layout in layouts), default=0)
        gather_bytes = max(sample_rows * (129280 // len(config.devices)) * 2,
                           (result_bytes + 15) // 16 * 16)
        decode_tokens = config.all_reduce_tokens
        runtime = create_roce_runtime(group, device, max_tokens=decode_tokens, pipe_tokens=tokens,
            hidden_size=5120, gather_bytes=gather_bytes, ring=config.roce_ring, ring_open=config.roce_chain, package_storage=package_storage)
        return runtime, runtime, runtime, runtime if layouts else None
    ipc = package_storage.peer_storage
    collective = create_all_reduce_runtime(group, device, max_tokens=config.all_reduce_tokens, hidden_size=5120, ipc=ipc)
    prefill = create_dma_all_reduce_runtime(group, device,
        max_tokens=config.prefill_chunk_size, hidden_size=5120, ipc=ipc)
    vocabulary = create_all_gather_runtime(group, device, max_rows=sample_rows, columns=129280, ipc=ipc)
    result = create_result_broadcast_runtime(group, device, layouts=layouts, ipc=ipc) if layouts else None
    return collective, prefill, vocabulary, result


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage=None):
    if package_storage is None:
        raise ValueError("DeepSeek preparation requires package storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    metadata_path = manifest.metadata_path
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, dspark=config.dspark,
                                confidence=config.split_speculation,
                                expert_scale_compression=config.expert_scale_compression)
    release_unused()
    if progress is not None:
        progress(f"Resident model tensors: {weights.allocated_bytes / (1 << 30):.3f} GiB per rank")
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    tables = Tables.prepare(manifest, weights.config, tokens=max(config.prefill_chunk_size, config.decode_tokens),
        device=device, sequence=sequence, frame=frame, host_allocator=package_storage.host_allocate, progress=progress, shard=shard)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, dspark=config.dspark)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete DeepSeek continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots * (max(config.speculative_widths) if config.dspark else 1)
        state_rows = snapshots + execution_rows
        pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        if config.kv_pages is not None:
            pages = config.kv_pages
        bytes_per_page = page_bytes(weights.config)
        total_memory = virtual_device_memory(device, group, tuning_replica=config.tuning_replica)
        pool = ContinuationPool.allocate(weights.config, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots), dspark=config.dspark)
        requests = DeviceRequestState.allocate(device, vocab_size=129280, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=129280, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.dspark else 8), device=device)
        coordinator = CollectivePreparation(group)
        verification_samples = (slots * max(config.speculative_widths)
                                if config.packed_verification else config.decode_tokens)
        collective, prefill_collective, vocabulary_collective, result_collective = _collective_runtimes(
            config, group, device, coordinator=coordinator, package_storage=package_storage,
            verification_samples=verification_samples)
        tables = replace(tables, readiness_runtime=collective)
        identity = hashlib.sha256(json.dumps({"config": json.loads((metadata_path / "config.json").read_text()),
            "headers": [s.header_sha256 for s in manifest.shards], "world_size": shard.world_size,
            "contract": "deepseek41.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity,
            workload_schema="deepseek41-text", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        activation_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        candidate_tokens = config.prefill_chunk_size
        if (weights.config.decoder_start is not None
                and weights.config.text["candidate_source_layer_id"] >= weights.config.decoder_start):
            candidate_tokens = min(candidate_tokens, 128)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, shard.world_size, device,
            buffers=activation_buffers.cursor(), candidate_tokens=candidate_tokens)
        scored_storage = (Scratch.allocate(config.prefill_chunk_size, 1, shard.world_size, device,
            buffers=activation_buffers.cursor(), candidate_tokens=config.prefill_chunk_size)
            if config.prompt_logprobs else None)
        # The suffix boundary gathers from the full prefill into disjoint destinations.
        suffix_storage = Scratch.allocate(128, 1, shard.world_size, device, buffers=prefill_storage.buffers)
        decoder_storage = Scratch.allocate(config.decode_tokens, verification_samples, shard.world_size, device,
            buffers=activation_buffers.cursor())
        window_staging = (decoder_storage.buffers.zeros(
            (len(weights.layers), slots, (128 + max(config.speculative_widths)) * 528), torch.uint8)
            if config.dspark else None)
        vision_state = vision_inputs = vision_prefill = vision_storage = None
        if weights.vision is not None:
            vision_inputs = VisionInputs.allocate(device, patches=max(PATCH_CAPACITIES))
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(weights.vision, inputs, vision_prefill, requests, features=4096,
                token_base=129280, sentinel_names=("image_start", "image_start", "image_newline", "image_end"),
                feature_type=1, image_slots=config.image_concurrency)
            vision_storage = {}
        decoder_factory, draft_factory = _forward_factories(config.mla_compute_mode,
            window_staging=window_staging, suffix_scratch=suffix_storage, vision=vision_state)
        buffers = dict(inputs.buffers(), **tables.buffers(), **commands.buffers(), **{
            name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
        if vision_state is not None:
            buffers.update(vision_inputs.buffers(), **vision_prefill.buffers())
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dspark:
            from ignition.components.dspark import DSparkForwards, SpeculativePreparation

            speculation = SpeculativePreparation(weights, tables, pool, requests, inputs, decoder_storage,
                draft_tokens=config.draft_tokens, completion=completion, frame=frame, sequence=sequence,
                session=session, workspace=arena,
                runtime=collective, vocabulary_runtime=vocabulary_collective,
                markov_runtime=None, coordinator=coordinator, result_runtime=result_collective,
                buffers=activation_buffers.cursor(),
                forwards=DSparkForwards(decoder_factory, Scratch, _CONTEXT,
                    draft_factory, PreparedMarkov))

    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    del manifest
    modes = (True, False) if config.dspark else (False,)
    count = (len(modes) * (prefill_graph_count(config.prefill_capacities, scoring)
                           + 4 * len(config.batch_capacities))
             + len(config.speculative_shapes) + len(config.packed_speculative_shapes)
             + len(config.batch_capacities))
    count += len(PATCH_CAPACITIES) if vision_state is not None else 0
    catalog_bytes = {}
    catalog_previous = torch.cuda.memory_allocated(device)
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
            nonlocal catalog_previous
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                                              packed_token_capacity=packed_tokens, draft_tokens=draft_tokens)
            allocated = torch.cuda.memory_allocated(device)
            catalog_bytes[kind] = catalog_bytes.get(kind, 0) + allocated - catalog_previous
            catalog_previous = allocated
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                    config.prefill_capacities, () if enabled else config.batch_capacities, scoring):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    cursor = graph_buffers.cursor()
                    metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=cursor)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    if kind == "scored_prefill":
                        storage = scored_storage
                    scratch[name] = storage.rows(capacity, rows)
                    decoder = decoder_factory(mode_weights, tables, scratch[name], metadata[name], pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 128 else collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator, session=session, name=name, workspace=arena)
                    batch = PreparedBatch(inputs, requests, metadata[name], mode=mode, shard=shard, pool=pool,
                        input_vocabulary=INPUT_VOCABULARY if vision_state is not None else None,
                        validate_prefill=vision_state.validate_prefill if vision_state is not None and mode == "prefill" else None,
                        buffers=cursor)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=PreparedContext if kind == "scored_prefill" else _CONTEXT, buffers=cursor)
                    fields = (*forward_inputs, "vision_token_indices", "vision_feature_slot") if mode == "prefill" and vision_state is not None else forward_inputs
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, fields, enabled=enabled)
        split, confidence = _prepare_speculation_catalog(config, speculation, add=add, buffers=buffers,
            graph_buffers=graph_buffers, forward_inputs=forward_inputs,
            host_allocate=package_storage.host_allocate, device=device)
        if vision_state is not None:
            for capacity in reversed(PATCH_CAPACITIES):
                name = f"vision{capacity}"
                cursor = activation_buffers.cursor()
                vision_storage[capacity] = VisionScratch.allocate(capacity, device, hidden=5120, buffers=cursor)
                encoder = PreparedVision(weights.vision, vision_storage[capacity], vision_inputs,
                    session=session, workspace=arena, name=name, buffers=cursor)
                add(name, PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=cursor),
                    "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, buffers=graph_buffers.cursor())
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name,
                        capacity=capacity, admission_buffers=admission, draft_state=enabled)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    if progress is not None:
        progress(f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
                 f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free")
        progress(f"Graph operand allocation by operation: {catalog_bytes}")
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
        progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=pool.clear_for_preparation, warmup_storage=package_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "packed_speculative", "vision"))
    session.freeze()
    pool.clear_for_preparation()
    if split is not None:
        for buffer in split.buffers().values():
            buffer.zero_()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)
    assets = {role: {"path": str(path)}
        for role, path in (("tokenizer", metadata_path / "tokenizer.json"),)}
    buffers["request_pending_token"] = requests.pending_token
    server_config = {"identity": identity, "model_id": snapshot.name,
        "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(metadata_path, eos_token_ids=[1], prompt_encoder=PromptEncoder.DEEPSEEK_V41,
            image_profile="deepseek_row_major" if vision_state is not None else None),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(), "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency,
            page_tokens=PAGE_TOKENS, draft_lag=0 if config.dspark else None, rows=None,
            embedding_tables=range(len(tables.sources)))}
    memory = {"rank": rank, "world_size": shard.world_size, "device": config.devices[rank],
        "catalog_operand_bytes": catalog_bytes,
        "weights_bytes": weights.allocated_bytes, "mapped_host_bytes": 0,
        "expert_scale_compression": config.expert_scale_compression,
        "mla_compute_mode": config.mla_compute_mode,
        "offloaded_bytes": sum(tables.partition(ordinal, shard.rank).rows * 264
                               for ordinal in range(len(tables.sources))),
        "continuation_bytes": pool.backing.nbytes, "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "pages": pages, "page_bytes": bytes_per_page, "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page, "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs), "prefill_capacities": config.prefill_capacities,
        "indexer_query_rows": INDEXER_QUERY_ROWS,
        "free_device_bytes": free, "total_device_bytes": total, "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning, "compiled_decoder": config.compile_decoder}
    if scoring is not None:
        memory.update(scoring.memory_report())
    memory["split_speculation"] = config.split_speculation
    memory["draft_prefix_selection"] = config.draft_prefix_selection
    if config.dspark:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_shapes=config.speculative_shapes,
            packed_speculative_shapes=config.packed_speculative_shapes,
            speculative_graph_budget=config.speculative_graph_budget,
            packed_speculative_graphs=len(config.packed_speculative_shapes))
    disposable = [activation_buffers, arena.tensor, metadata, tables.buffers(), graph_buffers]
    if vision_state is not None:
        disposable.extend((vision_storage, vision_state.arena))
    if speculation is not None:
        disposable.append(speculation.draft_scratch)
    if split is not None:
        disposable.append(split.buffers())
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
    model = PreparedModel(rebuild_weights={"weights": weights}, device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges(disposable), *completion.zero_ranges()),
        export_events=tables.events(),
        embedding_catalog=tables.resident_catalog(),
        package_data=tables if rank == 0 else None,
        resources=(scoring, scored_storage, weights, tables, pool, requests, inputs, admission, commands, arena, forwards,
            prefill_storage, decoder_storage, scratch, metadata, compiled, stream, graph_pool, session,
            collective, prefill_collective, vocabulary_collective, result_collective, coordinator, speculation,
            vision_state, vision_storage, vision_inputs, vision_prefill, graph_buffers, activation_buffers),
        close_resources=tuple(dict.fromkeys(c for c in
            (scoring, result_collective, prefill_collective, vocabulary_collective, collective) if c is not None)))
    if split is not None:
        model.proposal_confidence = confidence.host_view
        model.proposal_state = split
    return model
