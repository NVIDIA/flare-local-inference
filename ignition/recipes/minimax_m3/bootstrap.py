# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare TP4 MiniMax M3 target graphs with bounded history backing."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import (batch_capacities, prefill_capacities, request_capacity,
    image_capacity, speculative_shapes, speculative_token_capacity)
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.completion import CompletionOutputs
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
    create_all_reduce_runtime, create_dma_all_reduce_runtime, create_roce_runtime)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES, release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .vision import FEATURES, PATCH_CAPACITIES, PreparedVision, PreparedVisionGeneration, VisionInputs, VisionScratch


def _target_decoder(weights, tables, scratch, metadata, pool, **kwargs):
    return PreparedDecoder(weights, scratch, metadata, pool, **kwargs)


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    image_concurrency: int | None = None
    compile_workers: int | None = None
    compile_headroom_bytes: int = COMPILE_HEADROOM_BYTES
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = (0, 1, 2, 3)
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 32
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    stream_overlap: bool = True
    expert_scale_compression: str | None = None
    dspark: bool = True
    vision: bool = True
    diagnostic_graphs: tuple[str, ...] | None = None
    diagnostic_support_graphs: tuple[str, ...] = ()
    diagnostic_cache_pages: int | None = None
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = 128
    speculative_graph_budget: int = 64
    packed_verification: bool | None = None
    draft_prefix_selection: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if (not isinstance(self.devices, tuple) or len(self.devices) != 4
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != 4)
                or len(set(zip(self.hosts or ("",) * 4, self.devices))) != 4
                or any(type(device) is not int or device < 0 for device in self.devices)):
            raise ValueError("MiniMax M3 preparation requires four distinct host/device placements")
        if self.roce_ring and self.hosts is None:
            raise ValueError("RoCE ring requires ranks on separate hosts")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("MiniMax M3 context must be 128-aligned in [128,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("MiniMax M3 prefill chunk must be 64-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("MiniMax M3 expert scale compression must be None or 'csf'")
        drafting = self.dspark and max(self.speculative_widths, default=1) > 1
        packed = drafting and (self.diagnostic_graphs is None or self.packed_verification is True)
        if self.packed_verification is None:
            object.__setattr__(self, "packed_verification", packed)
        if self.draft_prefix_selection is None:
            object.__setattr__(self, "draft_prefix_selection", "gpu" if packed else "cpu")
        if type(self.packed_verification) is not bool or self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("DSpark requires a boolean packed mode and cpu/gpu prefix selection")
        if packed and (not self.packed_verification or self.draft_prefix_selection != "gpu"
                       or self.speculative_fixed_width is not None):
            raise ValueError("MiniMax M3 DSpark requires adaptive packed GPU rounds")
        if not drafting and (self.packed_verification or self.draft_prefix_selection != "cpu"):
            raise ValueError("packed GPU speculation requires positive DSpark draft depth")
        if self.dspark:
            if not self.speculative_widths or not set(self.speculative_widths) <= set(range(1, 9)):
                raise ValueError("MiniMax M3 DSpark verifies one through eight target positions")
            if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
                raise ValueError("fixed verification width must belong to the captured catalog")
            _ = self.speculative_shapes, self.packed_speculative_shapes

    @property
    def speculative_shapes(self):
        widths = (1,) if self.packed_verification else self.speculative_widths
        return speculative_shapes(self.max_active, widths, self.speculative_token_capacity) if self.dspark else ()

    @property
    def packed_speculative_shapes(self):
        from ignition.components.capacity import packed_speculative_shapes

        return packed_speculative_shapes(self.max_active, max(self.speculative_widths),
            self.speculative_token_capacity, self.speculative_graph_budget) if self.packed_verification else ()


def _verification_variants(shapes, selected):
    for batch, width in sorted(shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
        kinds = ("speculative",)
        for kind in kinds:
            name = f"{kind}{batch}x{width}"
            if selected is None or name in selected:
                yield name, kind, batch, width


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage):
    from .frontend import FRONTEND, prepare as prepare_frontend, runtime_assets
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise ValueError("MiniMax M3 target preparation requires SM120 or SM121 GPUs")
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [entry.header_sha256 for entry in manifest.shards]
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress,
        package_storage=package_storage, expert_scale_compression=config.expert_scale_compression,
        dspark=config.dspark, vision=config.vision)
    del manifest
    release_unused()
    if config.context > weights.config.text["max_position_embeddings"]:
        raise ValueError("prepared context exceeds the checkpoint's rotary position extent")
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete MiniMax M3 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    shapes = config.speculative_shapes
    packed_shapes = config.packed_speculative_shapes
    draft_depth = max(config.speculative_widths) - 1 if config.dspark else 0
    decode_tokens = max((slots, *(batch * width for batch, width in shapes),
                         *(tokens for _, tokens in packed_shapes)))
    decode_samples = max(decode_tokens, slots * max(config.speculative_widths) if config.dspark else slots)
    draft_tokens = slots * weights.config.draft.block_size if config.dspark else 0
    tokens = max(config.prefill_chunk_size, decode_tokens, draft_tokens)
    state_rows = slots + snapshots
    pages = -(-config.prefill_chunk_size // PAGE_TOKENS) + 2 * slots
    if config.diagnostic_cache_pages is not None:
        if config.diagnostic_graphs is None or config.diagnostic_cache_pages < pages:
            raise ValueError("diagnostic cache backing must cover preparation pages")
        pages = config.diagnostic_cache_pages
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, shard, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config, shard),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24), device=device)
        coordinator = CollectivePreparation(group)
        if coordinator.spans_hosts:
            sample_rows = max(decode_samples, draft_tokens)
            gather_bytes = max(sample_rows * weights.config.vocabulary_partition(shard)[2] * 2,
                               tokens * (weights.config.hidden // shard.world_size) * 2)
            collective = prefill_collective = vocabulary_collective = embedding_collective = create_roce_runtime(
                group, device, max_tokens=sample_rows,
                pipe_tokens=max(tokens, PATCH_CAPACITIES[-1] if config.vision else 0),
                hidden_size=weights.config.hidden, gather_bytes=gather_bytes,
                ring=config.roce_ring, package_storage=package_storage)
            vision_collective = collective if config.vision else None
        else:
            collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=max(decode_tokens, draft_tokens), hidden_size=weights.config.hidden,
                element_bytes=4, ipc=package_storage.peer_storage)
            prefill_collective = create_dma_all_reduce_runtime(group, device, max_tokens=tokens,
                hidden_size=weights.config.hidden, ipc=package_storage.peer_storage)
            vision_collective = (create_dma_all_reduce_runtime(group, device, max_tokens=PATCH_CAPACITIES[-1],
                hidden_size=weights.config.hidden, ipc=package_storage.peer_storage) if config.vision else None)
            vocabulary_collective = create_all_gather_runtime(group, device, max_rows=max(decode_samples, draft_tokens),
                columns=weights.config.vocabulary_partition(shard)[2] * shard.world_size,
                ipc=package_storage.peer_storage)
            embedding_collective = create_all_gather_runtime(group, device, max_rows=tokens,
                columns=weights.config.hidden, ipc=package_storage.peer_storage)
        identity = hashlib.sha256(json.dumps({"text": weights.config.text,
            "quantization": weights.config.quantization, "target_layers": weights.config.target_layers,
            "headers": headers, "world_size": shard.world_size,
            "draft": None if weights.config.draft is None else weights.config.draft.text,
            "vision": weights.config.vision if config.vision else None,
            "contract": "minimax-m3.paged-text.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(compile_workers=config.compile_workers, native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            compile_headroom_bytes=config.compile_headroom_bytes,
            tuning_replica=config.tuning_replica, device=device, checkpoint=identity,
            workload_schema="minimax-m3", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage,
            collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        private_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        with package_storage.scratch_pool() as scratch_pool:
            vision_inputs = VisionInputs.allocate(device) if config.vision else None
            vision_storage = (VisionScratch.allocate(PATCH_CAPACITIES[-1], shard, device,
                buffers=graph_buffers.cursor()) if config.vision else None)
            prefill_storage = Scratch.allocate(weights.config, shard, config.prefill_chunk_size, 1,
                                               device, buffers=graph_buffers.cursor())
            decode_storage = Scratch.allocate(weights.config, shard, decode_tokens, decode_samples,
                                              device, buffers=graph_buffers.cursor())
            draft_storage = None
            if config.dspark:
                from .draft import DraftScratch

                draft_storage = DraftScratch.allocate(weights.config, shard, draft_tokens, draft_tokens,
                    device, buffers=graph_buffers.cursor())
        vision_prefill = vision_state = None
        if config.vision:
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency,
                image_token=200025, features=FEATURES, hidden=weights.config.hidden // shard.world_size)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence, **completion.buffers())
        if config.vision:
            buffers.update(vision_inputs.buffers(), **vision_prefill.buffers())
        collectives = dict(runtime=collective, embedding_runtime=embedding_collective,
            vocabulary_runtime=vocabulary_collective, coordinator=coordinator)
        speculation = confidence = None
        if config.dspark:

            speculation = SimpleNamespace(weights=weights, tables=None, pool=pool, requests=requests,
                inputs=inputs, scratch=decode_storage, draft_scratch=draft_storage, completion=completion,
                frame=frame, sequence=sequence, session=session, workspace=arena, collectives=collectives,
                page_tokens=PAGE_TOKENS, block_size=weights.config.draft.block_size, logit_offset=0,
                markov_runtime=vocabulary_collective, rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device))
            if draft_depth:
                confidence = package_storage.host_allocate("proposal_confidence", (2, slots, draft_depth), torch.float32)
                buffers["proposal_confidence"] = confidence.device_view
                if config.packed_verification:
                    buffers["speculative_prefix_limits"] = torch.zeros(slots, dtype=torch.int32, device=device)
    forwards, signatures, metadata = {}, {}, {}
    prefills, batches = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    modes = (True, False) if config.dspark else (False,)
    patches = PATCH_CAPACITIES if config.vision else ()
    count = (len(modes) * (len(prefills) + 4 * len(batches)) + len(batches) + len(patches)
             + (len(shapes) + len(packed_shapes) + (2 * len(batches) if draft_depth and not packed_shapes else 0)
                if config.dspark else 0))
    selected = (None if config.diagnostic_graphs is None
                else set(config.diagnostic_graphs) | set(config.diagnostic_support_graphs))
    if selected is not None and not selected:
        raise ValueError("diagnostic graph selection must not be empty")
    with phase(progress, "Construct graph catalog", total=count if selected is None else len(selected)) as stage:
        def add(name, forward, kind, rows, tokens, fields, enabled=None, *, packed_tokens=0, draft_tokens=None):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                packed_token_capacity=packed_tokens, draft_tokens=draft_tokens,
                confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            for mode, sizes in (("prefill", reversed(prefills)), ("decode", () if enabled else reversed(batches))):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{mode}{capacity}" + (suffix if mode == "prefill" else "")
                    if selected is not None and name not in selected:
                        continue
                    private = private_buffers.cursor()
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=private)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode, vision=vision_state,
                        runtime=prefill_collective if mode == "prefill" else collective,
                        embedding_runtime=embedding_collective, vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator, session=session, name=name, workspace=arena,
                        overlap=config.stream_overlap, buffers=private)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" and config.vision else None,
                        buffers=private)
                    context = None
                    if enabled:
                        from .draft import PreparedContext

                        context = partial(PreparedContext, runtime=embedding_collective,
                            coordinator=coordinator, buffers=private)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=context, buffers=private)
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1,
                        (*forward_inputs, *(vision_prefill.buffers() if mode == "prefill" and config.vision else ())), enabled)
        if speculation is not None:
            from copy import copy
            from ignition.components.parallel_draft import PreparedParallelDraftRound
            from .draft import PreparedContext, PreparedDraft, PreparedMarkov

            def owner_for(private):
                owner = copy(speculation)
                owner.forwards = SimpleNamespace(
                    decoder=partial(_target_decoder, buffers=private, overlap=config.stream_overlap),
                    context=partial(PreparedContext, runtime=embedding_collective,
                        coordinator=coordinator, buffers=private),
                    draft=partial(PreparedDraft, block=weights.config.draft.block_size, buffers=private),
                    markov=partial(PreparedMarkov, buffers=private))
                return owner

            proposals = {}
            for batch in reversed(batches) if draft_depth else ():
                private = private_buffers.cursor()
                name = f"proposal{batch}x{draft_depth + 1}"
                if not packed_shapes and selected is not None and name not in selected:
                    continue
                source = PreparedParallelDraftRound(owner_for(private), batch, draft_depth + 1, name,
                    buffers=private, proposal_only=True)
                source.prepare()
                proposals[batch] = source
            for name, kind, batch, width in _verification_variants(shapes, selected):
                private = private_buffers.cursor()
                target = PreparedParallelDraftRound(owner_for(private), batch, width, name,
                    generate_proposals=kind == "speculative", buffers=private)
                target.prepare()
                forward = target
                add(name, forward, kind, batch, width, forward_inputs, True)
            if packed_shapes:
                from ignition.components.packed_speculation import PreparedPackedSpeculativeRound

                packed_inputs = tuple(field for field in forward_inputs if field != "query_lengths")
                packed_inputs += ("speculative_prefix_limits",)
                width = max(config.speculative_widths)
                for batch, packed_tokens in sorted(packed_shapes, key=lambda shape: (shape[1], shape[0]), reverse=True):
                    private = private_buffers.cursor()
                    private.join(proposals[batch].buffers)
                    name = f"packed-speculative{batch}x{width}-drafts{draft_depth}-tokens{packed_tokens}"
                    target = PreparedParallelDraftRound(owner_for(private), batch, width, name + ".target",
                        generate_proposals=False, buffers=private, target_token_capacity=packed_tokens)
                    target.prepare()
                    forward = PreparedPackedSpeculativeRound(proposals[batch], target,
                        buffers["speculative_prefix_limits"], confidence.device_view, name=name, buffers=private_buffers.cursor())
                    add(name, forward, "speculative", batch, width, packed_inputs, True,
                        packed_tokens=packed_tokens, draft_tokens=draft_depth)
        for capacity in reversed(patches):
            if selected is not None and f"vision{capacity}" not in selected:
                continue
            private = private_buffers.cursor()
            name = f"vision{capacity}"
            encoder = PreparedVision(weights.vision, vision_storage.rows(capacity), vision_inputs,
                runtime=vision_collective, coordinator=coordinator, session=session,
                workspace=arena, name=name, buffers=private)
            add(name, PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=private),
                "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            for capacity in reversed(batches):
                name = f"resume{capacity}" + suffix
                if selected is None or name in selected:
                    forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                        rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator, session=session, name=name, activation_mode="auto",
                        buffers=private_buffers.cursor())
                    add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    if selected is not None and name not in selected:
                        continue
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, buffers=private_buffers.cursor())
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled)
    if selected is not None and selected != set(forwards):
        raise ValueError(f"unknown diagnostic graphs: {sorted(selected - set(forwards))}")
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        if vision_inputs is not None:
            vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    vision_names = tuple(f"vision{capacity}" for capacity in reversed(patches) if f"vision{capacity}" in forwards)
    catalog = {name: forwards[name] for name in (*vision_names, *(name for name in forwards if name not in vision_names))}
    try:
        graphs, compiled = capture_forwards(catalog, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
            materialize=materialize, warmup_storage=package_storage, capture_names=config.diagnostic_graphs,
            compile_kinds=("prefill", "decode", "resume", "speculative", "vision"))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = runtime_assets(snapshot)
    frontend = prepare_frontend(snapshot)
    if not config.vision:
        frontend.pop("image", None)
    server_config = dict(identity=identity, model_id=RECIPE.served_model_name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=frontend, assets=assets,
        sampling_defaults=RECIPE.sampling_defaults.as_dict(),
        model_contract=(prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
            frontend=FRONTEND, image_slots=config.image_concurrency, dspark=config.dspark)
            if selected is None else None))
    if config.dspark:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths),
                config.speculative_token_capacity), packed_verification=config.packed_verification,
            draft_prefix_selection=config.draft_prefix_selection,
            packed_speculative_shapes=packed_shapes, speculative_graph_budget=config.speculative_graph_budget)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank],
        weights_bytes=weights.allocated_bytes, mapped_host_bytes=0, offloaded_bytes=0,
        continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config, shard), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config, shard), kv_dtype="float8_e4m3fn",
        operator_scratch_bytes=arena.nbytes, graph_buffer_bytes=graph_buffers.nbytes,
        private_buffer_bytes=private_buffers.nbytes, graphs=len(graphs), prefill_capacities=prefills,
        free_device_bytes=free, total_device_bytes=total, allocated_tensor_bytes=torch.cuda.memory_allocated(device),
        reserved_tensor_bytes=torch.cuda.memory_reserved(device), headroom_bytes=config.headroom_bytes,
        tuning=tuning, compiled_decoder=config.compile_decoder, speculative_shapes=shapes)
    channels = tuple(dict.fromkeys(channel for channel in (collective, prefill_collective, vision_collective,
        vocabulary_collective, embedding_collective) if channel is not None))
    disposable = [prefill_storage, decode_storage, draft_storage, vision_storage,
        metadata, arena.tensor, graph_buffers.chunks, private_buffers.chunks]
    if vision_state is not None:
        disposable.append(vision_state.arena)
    for forward in forwards.values():
        if hasattr(forward, "decoder"):
            disposable.append(forward.decoder.scratch_tensors())
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
        if getattr(forward, "context", None) is not None:
            disposable.append(forward.context.scratch_tensors())
        if hasattr(forward, "encoder"):
            disposable.extend((forward.errors, forward.encoder.scratch_tensors()))
    return PreparedModel(device=config.devices[rank], graphs=graphs, buffers=buffers, signatures={name: signatures[name] for name in graphs},
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        forwards={name: compiled.get(name, forwards[name]) for name in graphs},
        export_zero_ranges=(*storage_ranges(disposable), *completion.zero_ranges()),
        resources=(weights, pool, requests, inputs, admission, commands, completion, forwards, arena, prefill_storage,
            decode_storage, metadata, session, compiled, stream, graph_pool, coordinator, scratch_pool,
            graph_buffers, private_buffers, draft_storage, vision_inputs, vision_prefill, vision_state,
            vision_storage, speculation, confidence, *channels), close_resources=channels)
