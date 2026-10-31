# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Kimi K3 text and image graphs with tensor-parallel decoder storage."""

import contextlib
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity, speculative_token_capacity
from ignition.components.parallel import TensorParallelShard
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                              create_all_reduce_runtime, create_head_exchange,
                                              create_vocabulary_argmax_runtime)
from .routing import create_route_runtime
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES, release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena, scratch_views
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import DecoderStorage, PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .vision import PATCH_CAPACITIES, FEATURES, HIDDEN, PreparedVision, VisionInputs, VisionScratch
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState


# Most rows of one all-reduce; larger prefill reductions run in chunks of this many rows.
REDUCE_ROWS = 1024


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    compile_headroom_bytes: int = COMPILE_HEADROOM_BYTES
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = tuple(range(16))
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    # Red Hat DSpark checkpoint, either external or in the target's dspark directory.
    draft_checkpoint: Path | None = None
    # Verification widths count the pending input and up to seven drafts.
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_graph_budget: int = 64
    # Store eligible attention, shared-expert, latent MoE, and DSpark projections as MXFP8.
    online_mxfp8: bool = False
    # Shared experts overlap routing and routed experts on an auxiliary stream.
    stream_overlap: bool = True
    # Ranks share each page's context (decode context parallel over every rank, KV interleave 1).
    context_parallel: bool = True
    # The MoonViT encoder and its graphs prepare beside the text model unless the plan serves text only.
    vision: bool = True

    def __post_init__(self):
        from ignition.components.capacity import image_capacity

        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if len(self.devices) != 16 or len(set(self.devices)) != 16 or any(d < 0 for d in self.devices):
            raise ValueError("Kimi K3 preparation requires sixteen distinct GPUs")
        request_capacity(self.max_active)
        if self.drafting:
            if not self.speculative_widths or not set(self.speculative_widths) <= set(range(1, 9)):
                raise ValueError("Kimi K3 drafts verify one through eight rows per request")
            if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
                raise ValueError("fixed verification width must belong to the captured widths")
            _ = self.speculative_round_shapes
        if not self.page_tokens <= self.context <= 1048576 or self.context % self.page_tokens:
            raise ValueError(f"Kimi K3 context must be {self.page_tokens}-aligned in [{self.page_tokens},1048576]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Kimi K3 prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def drafting(self):
        return self.draft_checkpoint is not None

    @property
    def page_tokens(self):
        """Global positions per KV page: PAGE_TOKENS local records on each rank that shares the context."""
        return PAGE_TOKENS * (len(self.devices) if self.context_parallel else 1)

    @property
    def preparation_pages(self):
        """Back one prefill chunk and two boundary pages per request row."""
        return -(-self.prefill_chunk_size // self.page_tokens) + 2 * request_capacity(self.max_active)

    @property
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        return parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget,
            fixed_width=self.speculative_fixed_width) if self.drafting else ()

    @property
    def decode_tokens(self):
        return max((packed for _, _, packed in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def decode_samples(self):
        return max((batch * width for batch, width, _ in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.drafting else 0


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [shard.header_sha256 for shard in manifest.shards]
    with package_storage.transient():
        peers = None if group is None or package_storage.peer_storage is None else (group, package_storage.peer_storage)
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, online_mxfp8=config.online_mxfp8,
                                    draft_checkpoint=config.draft_checkpoint, peers=peers,
                                    vision=config.vision)
        del manifest
        release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard.world_size, speculative_tokens=config.draft_tokens)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Kimi K3 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // config.page_tokens
    drafting = weights.draft is not None
    max_tokens = max(config.prefill_chunk_size, config.decode_tokens)
    state_rows = slots + snapshots
    pages = config.preparation_pages
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, shard.world_size, state_rows + 1, pages, device,
            speculative_tokens=config.draft_tokens, storage=package_storage,
            maximum_pages=total_memory // page_bytes(weights.config),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max_tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24), device=device)
        coordinator = CollectivePreparation(group)
        # Prefill activations reduce in chunks of the rows decode and verification already need a channel for: a
        # channel sized for whole prefill chunks would stage 448 MiB per rank for 4096 rows.
        channel_rows = max(min(max_tokens, REDUCE_ROWS), config.decode_tokens)
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=channel_rows,
            hidden_size=weights.config.hidden, ipc=package_storage.peer_storage)
        vocabulary_collective = create_all_gather_runtime(group, device,
            max_rows=max(config.decode_samples, slots * (config.draft_tokens or 1)),
            columns=weights.config.vocabulary, ipc=package_storage.peer_storage)
        context_collective = draft_argmax = None
        if drafting:
            context_collective = create_all_gather_runtime(group, device, max_rows=channel_rows,
                columns=weights.config.hidden, ipc=package_storage.peer_storage)
            # DSpark drafts greedily: each position's head and Markov shards meet in one vocabulary-parallel
            # argmax without gathering both distributions.
            draft_argmax = create_vocabulary_argmax_runtime(group, device,
                local_vocab_size=weights.config.vocabulary // shard.world_size, max_rows=slots,
                ipc=package_storage.peer_storage)
        # Prefill rows beyond a channel's capacity gather in chunks, as they reduce.
        embedding_collective = create_all_gather_runtime(group, device, max_rows=channel_rows,
            columns=weights.config.hidden, ipc=package_storage.peer_storage)
        router_collective = create_all_gather_runtime(group, device, max_rows=channel_rows,
            columns=2 * shard.world_size * ((weights.config.text["num_experts"] // shard.world_size + 31) // 32 * 32),
            ipc=package_storage.peer_storage)
        route_collective = (create_route_runtime(group, device, ipc=package_storage.peer_storage)
                            if shard.world_size > 1 else None)
        context = None
        if config.context_parallel and shard.world_size > 1:
            from .dcp import RECORD_BYTES, ContextParallel, ContextShards

            # Decode and verification gather every rank's absorbed E4M3 query heads and combine latent outputs.
            heads = weights.config.text["num_attention_heads"]
            context = ContextParallel(ContextShards.on(rank, shard.world_size, device),
                heads=create_head_exchange(group, device, max_rows=config.decode_tokens, heads=heads, head_dim=512,
                                           query_head_dim=RECORD_BYTES, ipc=package_storage.peer_storage),
                coordinator=coordinator, peers=pool.share_pages(package_storage, group), slab=None)
        # Every MLA step gathers each rank's query and KV latent rows, padded to 16 bytes: one exchange for up to
        # eight rows, as vLLM's b12x latent transport carries, and one that prefill steps use in chunks.
        text = weights.config.text
        latent = (text["q_lora_rank"] + text["kv_lora_rank"] + text["qk_rope_head_dim"]) // shard.world_size
        latent_runtimes = tuple(create_head_exchange(group, device, max_rows=rows, heads=shard.world_size,
                                                     head_dim=-(-latent // 8) * 8, ipc=package_storage.peer_storage)
                                for rows in sorted({min(8, max_tokens), channel_rows}))
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": headers, "vision": weights.config.vision,
            "draft": None if not drafting else weights.config.draft.raw,
            "world_size": shard.world_size, "contract": "kimi3.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            compile_headroom_bytes=config.compile_headroom_bytes,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity,
            workload_schema="kimi3", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        with package_storage.scratch_pool() as scratch_pool:
            # Vision and text graphs execute serially and share disposable workspace.
            text_shape = (max(config.prefill_chunk_size, config.decode_tokens), weights.config.hidden)
            # KDA and MLA layers execute serially and share their attention output.
            text_shapes = (text_shape,) * 4
            if drafting:
                text_shapes += ((text_shape[0], len(weights.config.draft.auxiliary_layers) * text_shape[1]),)
            patches = PATCH_CAPACITIES if config.vision else ()
            cursor = graph_buffers.cursor()
            shared_scratch = cursor.zeros((max(max(patches, default=0) * 3 * weights.config.vision["qkv_hidden_size"],
                sum(rows * columns for rows, columns in text_shapes)),), torch.bfloat16)
            if patches:
                # Reserve the largest vision slabs before smaller text allocations share their chunks.
                VisionScratch.allocate(max(patches), device, work=shared_scratch, buffers=cursor)
                cursor = graph_buffers.cursor()
                shared_scratch = cursor.zeros(shared_scratch.shape, torch.bfloat16)
            activation, total, kda_output, moe_output, *auxiliary = scratch_views(shared_scratch, text_shapes)
            mla_output = kda_output
            prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, shard.world_size, device,
                                               activations=activation[:config.prefill_chunk_size], buffers=cursor)
            decode_storage = Scratch.allocate(config.decode_tokens, config.decode_samples, weights.config, shard.world_size,
                                              device, buffers=cursor)
            vision_inputs = VisionInputs.allocate(device) if patches else None
            vision_storage = {}
            if context is not None:
                from .dcp import CONTEXT_WINDOW, RECORD_BYTES

                # Serial prefill layers gather their context one window at a time into one slab beside the chunk's
                # own records; it is released while forwards compile.
                window = min(CONTEXT_WINDOW, config.context)
                context = replace(context, window=window, slab=torch.zeros(
                    (window + config.prefill_chunk_size, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device))
        vision_prefill = vision_state = None
        if patches:
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency,
                                       image_token=163605, features=FEATURES, hidden=HIDDEN // shard.world_size)
        vision_buffers = {} if not patches else {**vision_inputs.buffers(), **vision_prefill.buffers()}
        buffers = dict(inputs.buffers(), **vision_buffers, **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    forwards, signatures, metadata = {}, {}, {}
    decoder_buffers = DecoderStorage.allocate(weights.config, shard, device, allocator=cursor, scratch={
        "total": total, "kda": {"output": kda_output},
        "mla": {"output": mla_output}, "moe": {"output": moe_output},
        **({"auxiliary": auxiliary[0]} if drafting else {})})
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    collectives = dict(runtime=collective, embedding_runtime=embedding_collective, router_runtime=router_collective,
        route_runtime=route_collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
        vision_state=vision_state, context=context, latent_runtimes=latent_runtimes)
    speculation = None
    if drafting:
        from .dspark import SpeculativePreparation

        speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage, completion=completion,
            frame=frame, sequence=sequence, session=session, workspace=arena, collectives=collectives,
            decoder_buffers=decoder_buffers, draft_tokens=config.draft_tokens, context_runtime=context_collective,
            page_tokens=config.page_tokens, draft_argmax=draft_argmax)
    modes = (True, False) if drafting else (False,)
    count = (len(modes) * (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes)) + len(batch_sizes)
             + len(config.speculative_round_shapes) + len(patches))
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                packed_token_capacity=packed_tokens)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            # Speculative packages also hold plain graphs, selected when Flare loads without drafts.
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for kind, mode, sizes in prefill_modes(prefill_sizes, () if enabled else reversed(batch_sizes), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        session=session, name=name, workspace=arena, buffers=decoder_buffers, **collectives)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" and patches else None,
                        page_tokens=config.page_tokens)
                    # Plain graphs do not update draft context.
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=(None if speculation is None or not enabled
                                         else speculation.forwards.context))
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                        (*forward_inputs, *(vision_prefill.buffers() if mode == "prefill" and patches else ())),
                        enabled=enabled)
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            speculation.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            for batch, width, packed in sorted(config.speculative_round_shapes, key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                    buffers=speculation.sampling_buffers.cursor(), target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, (*inputs.buffers(), "sampling_grammar_mask"), enabled=True, packed_tokens=packed)

        for capacity in reversed(patches):
            name = f"vision{capacity}"
            with scratch_pool:
                cursor = graph_buffers.cursor()
                work = cursor.zeros(shared_scratch.shape, torch.bfloat16)
                vision_storage[capacity] = VisionScratch.allocate(capacity, device, work=work, buffers=cursor)
            encoder = PreparedVision(weights.vision, vision_storage[capacity], vision_inputs,
                session=session, workspace=arena, name=name, buffers=cursor)
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence,
                                                feature_range=shard.partition(HIDDEN), buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, activation_mode="auto")
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, page_tokens=config.page_tokens)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        if vision_inputs is not None:
            vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    # Large vision allocations must precede smaller text allocations in the shared graph pool.
    vision_names = tuple(f"vision{capacity}" for capacity in reversed(patches))
    capture_order = (*vision_names, *(name for name in forwards if name not in vision_names))
    capture_catalog = {name: forwards[name] for name in capture_order}
    try:
        graphs, compiled = capture_forwards(capture_catalog, signatures, session=session, device=device,
            stream=stream, pool=graph_pool, collective=collective, coordinator=coordinator,
            compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
            retain_graphs=config.retain_graphs, materialize=materialize,
            warmup_storage=package_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "vision", "speculative"))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    from .tokenizer import convert, runtime_assets
    assets = runtime_assets(convert(snapshot))
    eos = weights.config.text["eos_token_id"]
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), 163586]))
    frontend = prepare_frontend(snapshot, eos_token_ids=eos_ids, image_profile="moonvit" if patches else None)
    frontend["template_encoding"] = 1
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=config.page_tokens, frontend=frontend,
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency,
                                        vocabulary=vocabulary, page_tokens=config.page_tokens,
                                        draft_lag=0 if drafting else None, vision=bool(patches)))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank], weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config), page_tokens=config.page_tokens,
        kv_bytes=pages * page_bytes(weights.config), completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if scoring is not None:
        memory.update(scoring.memory_report())
    if drafting:
        server_config.update(speculative_widths=tuple(sorted({width for _, width, _ in config.speculative_round_shapes})),
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths),
                                                                  config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_round_shapes=config.speculative_round_shapes,
                      speculative_graph_budget=config.speculative_graph_budget)
    draft_disposable = [] if speculation is None else [speculation.draft_scratch.scratch_tensors(), speculation.context_buffers,
        speculation.kda_records.corrections, speculation.kda_records.keys_and_gates]
    return PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'output_res_score', 'vision', 'draft')), device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        close_resources=(() if scoring is None else (scoring,)),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges([prefill_storage, decode_storage, graph_buffers.chunks,
            None if speculation is None else speculation.sampling_buffers.chunks, arena.tensor, metadata,
                                           *(() if context is None else (context.slab,)),
                                           *(() if not patches else (vision_storage, vision_state.arena)),
                                           decoder_buffers.scratch_tensors(), *draft_disposable]), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage, context,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, collective,
                   vocabulary_collective, embedding_collective, router_collective, route_collective,
                   vision_inputs, vision_prefill, vision_state, vision_storage, scratch_pool, shared_scratch, graph_buffers,
                   speculation, context_collective, draft_argmax, latent_runtimes))
