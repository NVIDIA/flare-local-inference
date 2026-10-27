"""Prepare V4 vision, C4/C128 attention, and integrated DSpark graph catalogs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import (
    speculative_token_capacity,
    batch_capacities,
    prefill_capacities,
    request_capacity,
)
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from .contract import prepare as prepare_contract
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES, virtual_device_memory, release_unused
from ignition.preparation.frontend import PromptEncoder, prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .recipe import RECIPE
from .decoder import PreparedDecoder, Scratch
from .dspark import FORWARDS
from .pool import PAGE_TOKENS, ContinuationPool, continuation_layout, page_bytes
from ignition.components.deepseek_vision import (PreparedVision, VisionInputs, VisionScratch,
    PrefillVisionInputs, PreparedVisionGeneration, VisionState)
from .weights import ModelWeights

PATCH_CAPACITIES = (576, 1152, 2304, 3456)


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
    devices: tuple[int, ...] = (0, 1, 2, 3)
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    dspark: bool = True
    draft_head_quantization: str | None = None
    markov_head_quantization: str | None = None
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_graph_budget: int = 64
    # Run single-row decode's shared expert on an auxiliary stream; without overlap it runs serially.
    stream_overlap: bool = True

    def __post_init__(self):
        from ignition.components.capacity import image_capacity
        from ignition.recipes.declaration import Speculation

        Speculation(self.draft_tokens, draft_head_quantization=self.draft_head_quantization,
                    markov_head_quantization=self.markov_head_quantization)
        if self.draft_head_quantization not in (None, "nvfp4"):
            raise ValueError("DeepSeek V4 draft head quantization must be omitted or nvfp4")

        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if type(self.stream_overlap) is not bool:
            raise TypeError("stream_overlap must be a boolean")
        if (
            len(self.devices) not in (2, 4)
            or len(set(self.devices)) != len(self.devices)
            or any(d < 0 for d in self.devices)
        ):
            raise ValueError("DeepSeek V4 Flash Vision preparation requires two or four distinct GPUs")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("DeepSeek context must be 256-aligned in [256,1048576]")
        if not 512 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [512,context] to hold an image span")
        request_capacity(self.max_active)
        _ = self.speculative_round_shapes
        if (
            self.speculative_fixed_width is not None
            and self.speculative_fixed_width not in self.speculative_widths
        ):
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

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
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        return parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget,
            fixed_width=self.speculative_fixed_width) if self.dspark else ()

    @property
    def decode_tokens(self):
        return max((packed for _, _, packed in self.speculative_round_shapes), default=self.request_capacity)

    @property
    def decode_samples(self):
        return max((batch * width for batch, width, _ in self.speculative_round_shapes), default=self.request_capacity)

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dspark else 0


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
                                draft_head_quantization=config.draft_head_quantization,
                                markov_head_quantization=config.markov_head_quantization)
    release_unused()
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, dspark=config.dspark)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete DeepSeek continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots * (max(config.speculative_widths) if config.dspark else 1)
        state_rows = snapshots + execution_rows
        initial_pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        bytes_per_page = page_bytes(weights.config)
        total_memory = virtual_device_memory(device, group, tuning_replica=config.tuning_replica)
        pool = ContinuationPool.allocate(
            weights.config,
            state_rows + 1,
            initial_pages,
            device,
            storage=package_storage,
            maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots),
            dspark=config.dspark,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=129280, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=129280, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24), device=device)
        coordinator = CollectivePreparation(group)
        ipc = package_storage.peer_storage
        tokens = max(config.prefill_chunk_size, config.decode_tokens, slots * config.draft_tokens)
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=tokens, hidden_size=4096, ipc=ipc)
        prefill_collective = create_dma_all_reduce_runtime(
            group, device, max_tokens=config.prefill_chunk_size, hidden_size=4096, ipc=ipc
        )
        vocabulary_collective = create_all_gather_runtime(
            group, device, max_rows=max(config.decode_samples, slots * config.draft_tokens), columns=129280, ipc=ipc
        )
        markov_collective = (
            create_all_gather_runtime(group, device, max_rows=slots, columns=2 * 129280, ipc=ipc)
            if config.dspark
            else None
        )
        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": json.loads((metadata_path / "config.json").read_text()),
                    "headers": [s.header_sha256 for s in manifest.shards],
                    "world_size": shard.world_size,
                    "contract": "deepseek4.graph-inputs.v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            compile_headroom_bytes=config.compile_headroom_bytes,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device,
            checkpoint=identity,
            workload_schema="deepseek4-vision",
            numerical_policy="default",
            coordinator=coordinator,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=package_storage,
        )
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, shard.world_size, device, blocks=blocks,
                                           buffers=graph_buffers.cursor())
        decoder_storage = Scratch.allocate(
            config.decode_tokens, config.decode_samples, shard.world_size, device, blocks=blocks,
            buffers=graph_buffers.cursor()
        )
        vision_inputs = VisionInputs.allocate(device, patches=max(PATCH_CAPACITIES))
        vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
        vision_state = VisionState(weights.vision, inputs, vision_prefill, requests, features=1536, token_base=129280,
            sentinel_names=("image_start", "image_pad", "image_pad", "image_newline", "image_end"), feature_type=2,
            image_slots=config.image_concurrency)
        vision_storage = {}
        buffers = dict(
            inputs.buffers(),
            **vision_inputs.buffers(),
            **vision_prefill.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
        )
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dspark:
            from ignition.components.dspark import SpeculativePreparation

            speculation = SpeculativePreparation(
                weights,
                vision_state,
                pool,
                requests,
                inputs,
                decoder_storage,
                draft_tokens=config.draft_tokens,
                completion=completion,
                frame=frame,
                sequence=sequence,
                session=session,
                workspace=arena,
                runtime=collective,
                vocabulary_runtime=vocabulary_collective,
                markov_runtime=markov_collective,
                coordinator=coordinator,
                forwards=FORWARDS,
                page_tokens=PAGE_TOKENS,
            )

    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    del manifest
    modes = (True, False) if config.dspark else (False,)
    count = (
        len(modes) * (prefill_graph_count(config.prefill_capacities, scoring) + 4 * len(config.batch_capacities))
        + len(config.speculative_round_shapes)
        + len(config.batch_capacities)
        + len(PATCH_CAPACITIES)
    )
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
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                    config.prefill_capacities, () if enabled else config.batch_capacities, scoring):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    scratch[name] = storage.rows(capacity, rows)
                    decoder = PreparedDecoder(
                        mode_weights,
                        vision_state,
                        scratch[name],
                        metadata[name],
                        pool,
                        mode=mode, scoring=scoring if kind == "scored_prefill" else None,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 1024 else collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        workspace=arena,
                    )
                    batch = PreparedBatch(
                        inputs,
                        requests,
                        metadata[name],
                        mode=mode,
                        shard=shard,
                        pool=pool,
                        page_tokens=PAGE_TOKENS,
                        input_vocabulary=129285,
                        validate_prefill=vision_state.validate_prefill if mode == "prefill" else None,
                    )
                    forward = PreparedGeneration(
                        decoder,
                        batch,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name + ".generation",
                        workspace=arena,
                        context_factory=FORWARDS.context,
                    )
                    add(
                        name,
                        forward,
                        mode,
                        rows,
                        capacity if mode == "prefill" else 1,
                        (
                            (*forward_inputs, *vision_prefill.buffers())
                            if mode == "prefill"
                            else forward_inputs
                        ),
                        enabled=enabled,
                    )
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            speculation.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            for batch, width, packed in sorted(config.speculative_round_shapes, key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                                                     buffers=speculation.sampling_buffers.cursor(), target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, enabled=True, packed_tokens=packed)
        for capacity in reversed(PATCH_CAPACITIES):
            name = f"vision{capacity}"
            cursor = graph_buffers.cursor()
            vision_storage[capacity] = VisionScratch.allocate(capacity, device, hidden=4096, buffers=cursor)
            encoder = PreparedVision(
                weights.vision,
                vision_storage[capacity],
                vision_inputs,
                session=session,
                workspace=arena,
                name=name,
                buffers=cursor,
            )
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(
                    mode_weights,
                    requests,
                    pool,
                    inputs,
                    completion,
                    frame,
                    sequence,
                    rows=capacity,
                    runtime=collective,
                    vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator,
                    session=session,
                    name=name,
                )
                add(
                    name,
                    forward,
                    "resume",
                    capacity,
                    1,
                    (*row_inputs, "sampling_grammar_mask"),
                    enabled=enabled,
                )
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(
                        mode,
                        requests,
                        pool,
                        commands,
                        inputs,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        capacity=capacity,
                        admission_buffers=admission,
                        draft_state=enabled,
                        page_tokens=PAGE_TOKENS,
                    )
                    fields = (
                        row_inputs
                        if mode == "snapshot"
                        else ("request_state_rows", *admission.__dataclass_fields__)
                    )
                    add(
                        name,
                        forward,
                        mode,
                        capacity,
                        1,
                        (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled,
                    )
    if progress is not None:
        progress(
            f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
            f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free"
        )
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    pages = initial_pages

    def materialize_cache():
        pool.clear_for_preparation()

    graphs, compiled = capture_forwards(
        forwards,
        signatures,
        session=session,
        device=device,
        stream=stream,
        pool=graph_pool,
        collective=collective,
        coordinator=coordinator,
        compile_decoder=config.compile_decoder,
        progress=progress,
        cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs,
        materialize=materialize_cache,
        warmup_storage=package_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "vision"),
    )
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)
    frontend = Path(__file__).parent / "assets"
    assets = {
        role: {"path": str(path)}
        for role, path in (
            ("tokenizer", metadata_path / "tokenizer.json"),
        )
    }
    buffers["request_pending_token"] = requests.pending_token
    server_config = {
        "identity": identity,
        "model_id": snapshot.name,
        "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages,
        "state_rows": state_rows,
        "snapshots": snapshots,
        "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(metadata_path, eos_token_ids=[1], prompt_encoder=PromptEncoder.DEEPSEEK_V4, image_profile="deepseek4",
            preprocessor=frontend / "preprocessor_config.json"),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency, page_tokens=PAGE_TOKENS, draft_lag=0 if config.dspark else None, rows=None),
    }
    memory = {
        "rank": rank,
        "world_size": shard.world_size,
        "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes,
        "draft_head_quantization": config.draft_head_quantization,
        "markov_head_quantization": config.markov_head_quantization,
        "mapped_host_bytes": 0,
        "offloaded_bytes": 0,
        "continuation_bytes": pool.backing.nbytes,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "state_rows": state_rows,
        "pages": pages,
        "page_bytes": bytes_per_page,
        "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page,
        "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs),
        "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning,
        "compiled_decoder": config.compile_decoder,
    }
    if scoring is not None:
        memory.update(scoring.memory_report())
    if config.dspark:
        server_config.update(
            speculative_widths=tuple(sorted({width for _, width, _ in config.speculative_round_shapes})),
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width,
        )
        memory.update(speculative_round_shapes=config.speculative_round_shapes,
                      speculative_graph_budget=config.speculative_graph_budget)
    disposable = [
        prefill_storage,
        decoder_storage,
        arena.tensor,
        metadata,
        vision_storage,
        vision_state.arena,
        graph_buffers.chunks,
    ]
    if speculation is not None:
        disposable.append(speculation.draft_scratch)
        disposable.append(speculation.sampling_buffers.chunks)
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
    return PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'head_mix', 'draft', 'vision')),
        device=config.devices[rank],
        graphs=graphs,
        buffers=buffers,
        signatures=signatures,
        max_active=config.max_active,
        session=session,
        server_config=server_config,
        memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges(disposable), *completion.zero_ranges()),
        resources=(scoring,
            weights,
            vision_state,
            vision_inputs,
            vision_prefill,
            vision_storage,
            graph_buffers,
            pool,
            requests,
            inputs,
            admission,
            commands,
            arena,
            forwards,
            prefill_storage,
            decoder_storage,
            scratch,
            metadata,
            compiled,
            stream,
            graph_pool,
            session,
            collective,
            prefill_collective,
            vocabulary_collective,
            markov_collective,
            coordinator,
            speculation,
        ),
        close_resources=tuple(
            c
            for c in (scoring, markov_collective, prefill_collective, vocabulary_collective, collective)
            if c is not None
        ),
    )
