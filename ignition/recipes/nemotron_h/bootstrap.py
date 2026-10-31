# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Nemotron-H graphs with explicit Mamba continuations and FP8 KV pages."""

import contextlib
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.parallel import TensorParallelShard
from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity, speculative_token_capacity
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES, release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights
from .vision import PATCH_CAPACITIES, FEATURES, HIDDEN, VisionInputs, VisionScratch, PreparedVision
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState


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
    device: int = 0
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    stream_overlap: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    draft_checkpoint: Path | None = None
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_graph_budget: int = 64

    @property
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        return parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget,
            fixed_width=self.speculative_fixed_width) if self.draft_checkpoint is not None else ()

    @property
    def decode_tokens(self):
        return max((packed for _, _, packed in self.speculative_round_shapes), default=request_capacity(self.max_active))

    @property
    def decode_samples(self):
        return max((batch * width for batch, width, _ in self.speculative_round_shapes), default=request_capacity(self.max_active))

    def __post_init__(self):
        from ignition.components.capacity import image_capacity

        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        request_capacity(self.max_active)
        if type(self.stream_overlap) is not bool:
            raise ValueError("stream_overlap must be a boolean")
        if not self.speculative_widths or max(self.speculative_widths) > 8:
            raise ValueError("LiLiCorr supports at most seven draft tokens")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the captured widths")
        _ = self.speculative_round_shapes
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Nemotron-H context must be 128-aligned in [128,262144]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Nemotron-H prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")


@torch.no_grad()
@with_autotune_effort
def prepare(snapshot, config, *, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=TensorParallelShard(0, 1), progress=progress,
                                draft_checkpoint=config.draft_checkpoint)
    drafting = weights.draft is not None
    release_unused()
    layout = continuation_layout(weights.config)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Nemotron-H continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = torch.cuda.mem_get_info(device)[1]
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, config.decode_tokens), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24), device=device)
        coordinator = None
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "draft": weights.config.draft.text if drafting else None,
            "headers": [s.header_sha256 for s in manifest.shards],
            "contract": "nemotron-h.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            compile_headroom_bytes=config.compile_headroom_bytes,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity,
            workload_schema="nemotron-h", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        scratch_pool = package_storage.scratch_pool()
        graph_buffers, layer_buffers = GraphBufferPool(device), GraphBufferPool(device)
        with scratch_pool:
            vision_inputs = VisionInputs.allocate(device)
        vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
        vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency, image_token=weights.config.image_token,
                                   features=FEATURES, hidden=HIDDEN)
        buffers = dict(inputs.buffers(), **vision_inputs.buffers(), **vision_prefill.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame)
    forwards, signatures, metadata = {}, {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    speculation = None
    if drafting:
        from .dflash import SpeculativePreparation
        speculation = SpeculativePreparation(weights, pool, requests, inputs, completion=completion, frame=frame,
            sequence=sequence, session=session, workspace=arena, coordinator=coordinator,
            buffers=graph_buffers, layer_buffers=layer_buffers)
    modes = (True, False) if drafting else (False,)
    count = (len(modes) * (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes))
             + len(batch_sizes) + len(config.speculative_round_shapes) + len(PATCH_CAPACITIES))
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                packed_token_capacity=packed_tokens)
            stage.update(advance=1)

        for capacity in reversed(PATCH_CAPACITIES):
            name = f"vision{capacity}"
            with scratch_pool:
                cursor = graph_buffers.cursor()
                scratch = VisionScratch.allocate(capacity, device, buffers=cursor)
            encoder = PreparedVision(weights.vision, scratch, vision_inputs,
                                     session=session, workspace=arena, name=name, buffers=cursor)
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for kind, mode, sizes in prefill_modes(prefill_sizes, () if enabled else reversed(batch_sizes), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    with scratch_pool:
                        cursor = graph_buffers.cursor()
                        m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=cursor)
                        scratch = Scratch.allocate(capacity, rows, weights.config, cursor)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None, buffers=cursor,
                        vocabulary_runtime=None, coordinator=coordinator, session=session, name=name, workspace=arena, layer_buffers=layer_buffers, vision_state=vision_state)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool, buffers=cursor,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" else None)
                    from .dflash import PreparedContext
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena, buffers=cursor,
                        context_factory=partial(PreparedContext, buffers=cursor) if enabled else None)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                        (*inputs.buffers(), "sampling_grammar_mask", *(vision_prefill.buffers() if mode == "prefill" else ())), enabled=enabled)
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            for batch, width, packed in sorted(config.speculative_round_shapes, key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                speculation.begin(batch, width, packed)
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                    buffers=speculation.cursor, target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, (*inputs.buffers(), "sampling_grammar_mask"),
                    enabled=True, packed_tokens=packed)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=None, vocabulary_runtime=None, coordinator=coordinator,
                    session=session, name=name, activation_mode="auto", buffers=graph_buffers.cursor())
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=None, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    try:
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
            stream=stream, pool=graph_pool, collective=None, coordinator=coordinator,
            compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
            retain_graphs=config.retain_graphs, materialize=materialize, warmup_storage=package_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "vision", "speculative"))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = {role: {"path": str(snapshot / file)}
              for role, file in (("tokenizer", "tokenizer.json"),)}
    assets["chat_template"] = annotated_template_asset(
        (snapshot / "chat_template.jinja").read_bytes(), Path(__file__).with_name("assets") / "chat_template.patch.json")
    eos = weights.config.eos_token
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    eos_text = tokenizer["eos_token"]
    if isinstance(eos_text, dict):
        eos_text = eos_text["content"]
    from tokenizers import Tokenizer
    tokenizer_eos = Tokenizer.from_file(str(snapshot / "tokenizer.json")).token_to_id(eos_text)
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), tokenizer_eos]))
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, eos_token_ids=eos_ids, image_profile="radio"),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency, vocabulary=vocabulary, draft_lag=0 if drafting else None))
    if drafting:
        server_config.update(speculative_widths=tuple(sorted({width for _, width, _ in config.speculative_round_shapes})),
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config), completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if scoring is not None:
        memory.update(scoring.memory_report())
    return PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'vision', 'draft')), device=config.device, graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        close_resources=(() if scoring is None else (scoring,)),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges([arena.tensor, metadata, vision_state.arena,
            *[chunk for p in (graph_buffers, layer_buffers) for chunks in p.chunks.values() for chunk in chunks]]), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, graph_buffers, layer_buffers,
                   metadata, compiled, stream, graph_pool, session, coordinator, scratch_pool, vision_state, vision_inputs, vision_prefill))
