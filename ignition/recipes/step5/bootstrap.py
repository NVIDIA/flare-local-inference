"""Prepare Step-5 text, MTP, and image graphs for the shared native runtime."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_modes,
)

from ignition.components.capacity import (
    batch_capacities,
    prefill_capacities,
    request_capacity,
    speculative_shapes,
    speculative_token_capacity,
)
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
)
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
from .frontend import prepare_frontend
from .mtp import PreparedFeedback
from .pool import ContinuationPool, continuation_layout, page_bytes
from .speculative import PreparedRound
from .vision import (
    FEATURES,
    PATCH_CAPACITIES,
    PreparedVision,
    VisionInputs,
    VisionScratch,
)
from .vision_state import PreparedVisionGeneration
from .weights import ModelWeights


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
    devices: tuple[int, ...] = tuple(range(8))
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    mtp: bool = True
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    stream_overlap: bool = True
    expert_scale_compression: str | None = None

    def __post_init__(self):
        from ignition.components.capacity import image_capacity

        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("Step-5 expert scale compression must be None or 'csf'")
        if len(self.devices) not in (4, 8, 16) or len(set(self.devices)) != len(self.devices) or any(d < 0 for d in self.devices):
            raise ValueError("Step-5 preparation requires four, eight, or sixteen distinct GPUs")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("Step-5 context must be 128-aligned in [128,1048576]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Step-5 prefill chunk must be 128-aligned and fit the context")
        speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
        if max(self.speculative_widths) > 4:
            raise ValueError("Step-5 MTP supports at most three draft tokens")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared catalog")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")


    @property
    def speculative_shapes(self):
        return (speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
                if self.mtp else ())


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
        headers = [s.header_sha256 for s in manifest.shards]
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, mtp=config.mtp,
        expert_scale_compression=config.expert_scale_compression, package_storage=package_storage)
    del manifest
    release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard, mtp=config.mtp)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold one complete Step-5 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    shapes = config.speculative_shapes
    decode_tokens = max((b * w for b, w in shapes), default=slots)
    draft_tokens = max((b * (weights.config.text["num_nextn_predict_layers"] + w) for b, w in shapes), default=slots)
    tokens = max(config.prefill_chunk_size, decode_tokens)
    state_rows, pages = slots + snapshots, config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, shard, state_rows + 1, pages, device, mtp=config.mtp,
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
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=max(decode_tokens, draft_tokens),
            hidden_size=weights.config.hidden, element_bytes=4, ipc=package_storage.peer_storage)
        prefill_collective = create_dma_all_reduce_runtime(group, device, max_tokens=tokens,
            hidden_size=weights.config.hidden, element_bytes=4, ipc=package_storage.peer_storage)
        vision_collective = create_dma_all_reduce_runtime(group, device, max_tokens=PATCH_CAPACITIES[-1],
            hidden_size=1536, ipc=package_storage.peer_storage)
        vocabulary_collective = create_all_gather_runtime(group, device, max_rows=decode_tokens,
            columns=weights.config.vocabulary_partition(shard)[2] * shard.world_size, ipc=package_storage.peer_storage)
        embedding_collective = create_all_gather_runtime(group, device, max_rows=tokens,
            columns=weights.config.hidden, ipc=package_storage.peer_storage)
        identity = hashlib.sha256(json.dumps({"config": weights.config.text, "vision": weights.config.vision,
            "quantization": weights.config.quantization, "headers": headers, "world_size": shard.world_size,
            "contract": "step5.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            compile_headroom_bytes=config.compile_headroom_bytes,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity, workload_schema="step5",
            numerical_policy="default", coordinator=coordinator, headroom_bytes=config.headroom_bytes,
            benchmark_storage=package_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        private_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        with package_storage.scratch_pool() as scratch_pool:
            vision_inputs = VisionInputs.allocate(device)
            vision_storage = VisionScratch.allocate(PATCH_CAPACITIES[-1], shard, device, buffers=graph_buffers.cursor())
            prefill_storage = Scratch.allocate(weights.config, shard, config.prefill_chunk_size, 1, config.context,
                                               device, buffers=graph_buffers.cursor())
            decode_storage = Scratch.allocate(weights.config, shard, decode_tokens, decode_tokens, config.context,
                                              device, buffers=graph_buffers.cursor())
            draft_storage = (Scratch.allocate(weights.config, shard, draft_tokens, draft_tokens, config.context,
                                             device, buffers=graph_buffers.cursor()) if config.mtp else None)
        vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
        vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency, image_token=128001,
                                   features=FEATURES, hidden=weights.config.hidden // shard.world_size)
        buffers = dict(inputs.buffers(), **vision_inputs.buffers(), **vision_prefill.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity, request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame, submission_sequence=sequence, **completion.buffers())
        collectives = {"runtime": collective, "embedding_runtime": embedding_collective,
                           "vocabulary_runtime": vocabulary_collective, "coordinator": coordinator}
        speculation = None
        if config.mtp:
            speculation = SimpleNamespace(weights=weights, pool=pool, inputs=inputs, requests=requests, scratch=decode_storage,
                draft_scratch=draft_storage,
                session=session, workspace=arena, collectives=collectives, overlap=config.stream_overlap,
                completion=completion, frame=frame, sequence=sequence, rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device),
                sampling_scratch=None)
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    forwards, signatures, metadata = {}, {}, {}
    prefills, batches = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    modes = (True, False) if config.mtp else (False,)
    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, tokens, fields, enabled=None):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, sizes in prefill_modes(prefills, () if enabled else reversed(batches), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    private = private_buffers.cursor()
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=private)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    execution_collectives = dict(collectives, runtime=prefill_collective if mode == "prefill" else collective)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None, vision=vision_state,
                        session=session, name=name, workspace=arena, overlap=config.stream_overlap, buffers=private, **execution_collectives)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" else None, buffers=private)
                    context = partial(PreparedFeedback, scratch=scratch, vision=vision_state, buffers=private, **execution_collectives)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=context if enabled else None, buffers=private)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                        (*forward_inputs, *(vision_prefill.buffers() if mode == "prefill" else ())), enabled)
        if speculation is not None:
            for batch, width in shapes:
                name = f"speculative{batch}x{width}"
                forward = PreparedRound(speculation, batch, width, name, buffers=private_buffers.cursor())
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, True)
        for capacity in reversed(PATCH_CAPACITIES):
            private = private_buffers.cursor()
            name = f"vision{capacity}"
            encoder = PreparedVision(weights.vision, vision_storage.rows(capacity), vision_inputs, runtime=vision_collective,
                coordinator=coordinator, session=session, workspace=arena, name=name, buffers=private)
            add(name, PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=private),
                "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in reversed(batches):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence, rows=capacity,
                    runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                    session=session, name=name, buffers=private_buffers.cursor())
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, buffers=private_buffers.cursor())
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    vision_names = tuple(f"vision{capacity}" for capacity in reversed(PATCH_CAPACITIES))
    catalog = {name: forwards[name] for name in (*vision_names, *(name for name in forwards if name not in vision_names))}
    try:
        graphs, compiled = capture_forwards(catalog, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs, materialize=materialize,
            warmup_storage=package_storage, compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "vision"))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    from .frontend import runtime_assets
    assets = runtime_assets(snapshot)
    frontend = prepare_frontend(snapshot, eos_token_ids=weights.config.text["eos_token_id"])
    frontend["image"] = {"step": {}}
    server_config = {"identity": identity, "model_id": snapshot.name, "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size, "pages": pages, "state_rows": state_rows, "snapshots": snapshots,
        "page_tokens": PAGE_TOKENS, "frontend": frontend, "sampling_defaults": RECIPE.sampling_defaults.as_dict(), "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency, vocabulary=vocabulary, mtp=config.mtp)}
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity))
    free, total = torch.cuda.mem_get_info(device)
    memory = {"rank": rank, "world_size": shard.world_size, "device": config.devices[rank], "weights_bytes": weights.allocated_bytes,
        "expert_scale_compression": config.expert_scale_compression,
        "mapped_host_bytes": 0, "offloaded_bytes": 0, "continuation_bytes": pool.backing.nbytes, "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "pages": pages, "page_bytes": page_bytes(weights.config, shard), "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * page_bytes(weights.config, shard), "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": arena.nbytes, "graphs": len(graphs),
        "graph_buffer_bytes": graph_buffers.nbytes, "private_buffer_bytes": private_buffers.nbytes,
        "kv_dtype": "float8_e4m3fn",
        "prefill_capacities": prefills, "free_device_bytes": free, "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes, "tuning": tuning, "compiled_decoder": config.compile_decoder,
        "speculative_shapes": shapes,
        "speculative_token_capacity": server_config.get("speculative_token_capacity", 0)}
    if scoring is not None:
        memory.update(scoring.memory_report())
    channels = (collective, prefill_collective, vision_collective, vocabulary_collective, embedding_collective)
    disposable = [prefill_storage, decode_storage, vision_storage, arena.tensor, metadata, vision_state.arena,
                  weights.scale_scratch, graph_buffers.chunks, private_buffers.chunks]
    if speculation is not None:
        disposable.append(speculation.sampling_scratch)
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "scratch_tensors"):
            disposable.append(forward.scratch_tensors())
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
        if hasattr(forward, "decoder"):
            disposable.append(forward.decoder.scratch_tensors())
        if getattr(forward, "context", None) is not None:
            disposable.append(forward.context.scratch_tensors())
        if hasattr(forward, "encoder"):
            disposable.extend((forward.errors, forward.encoder.scratch_tensors()))
    return PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'draft', 'vision')), device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges(disposable), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
                   decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, *channels,
                   vision_inputs, vision_prefill, vision_state, vision_storage, scratch_pool, graph_buffers, private_buffers, speculation),
        close_resources=(*channels, *((scoring,) if scoring is not None else ())))
