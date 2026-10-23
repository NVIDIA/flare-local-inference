"""Prepare MiMo text, DFlash proposals and image encoder graphs for local TP."""

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
from .contract import prepare as prepare_contract
from ignition.preparation.contract import history_region
from ignition.preparation import runtime_package_pb2 as contract_proto
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import release_unused
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .recipe import RECIPE
from .decoder import PreparedDecoder, Scratch
from .dflash import SpeculativePreparation
from .model import DecoderEnvironment
from .pool import PAGE_TOKENS, ContinuationPool, continuation_layout, page_bytes
from .vision import FEATURES, HIDDEN, PATCH_CAPACITIES, PreparedVision, VisionInputs, VisionScratch
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig:
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    autotune_effort: str = "medium"
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
    dflash: bool = True
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None

    def __post_init__(self):
        from ignition.components.capacity import image_capacity

        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if (
            len(self.devices) not in (2, 4, 8)
            or len(set(self.devices)) != len(self.devices)
            or any(d < 0 for d in self.devices)
        ):
            raise ValueError("MiMo preparation requires distinct GPUs dividing its eight QKV checkpoint shards")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("MiMo context must be 128-aligned in [128,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        _ = self.speculative_shapes
        if not self.speculative_widths or max(self.speculative_widths) > 8:
            raise ValueError("MiMo DFlash verification supports at most seven drafts")
        if (
            self.speculative_fixed_width is not None
            and self.speculative_fixed_width not in self.speculative_widths
        ):
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
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
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity) if self.dflash else ()

    @property
    def decode_tokens(self):
        return max((b * w for b, w in self.speculative_shapes), default=self.request_capacity)

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dflash else 0


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage=None):
    if package_storage is None:
        raise ValueError("MiMo preparation requires package storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, dflash=config.dflash)
    release_unused()
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, shard, dflash=config.dflash)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete MiMo continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        state_rows = snapshots + slots
        initial_pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        bytes_per_page = page_bytes(weights.config, shard)
        total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
        pool = ContinuationPool.allocate(
            weights.config,
            shard,
            state_rows + 1,
            initial_pages,
            device,
            storage=package_storage,
            maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots),
            dflash=config.dflash,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=152576, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=152576, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.dflash else 8), device=device)
        coordinator = CollectivePreparation(group)
        ipc = package_storage.peer_storage
        draft_rows = slots * weights.config.draft.block_size if config.dflash else 0
        tokens = max(config.prefill_chunk_size, config.decode_tokens, draft_rows)
        collective = create_all_reduce_runtime(group, device, max_tokens=tokens, hidden_size=6144, ipc=ipc)
        prefill_collective = create_dma_all_reduce_runtime(
            group, device, max_tokens=config.prefill_chunk_size, hidden_size=6144, ipc=ipc
        )
        vocabulary_collective = create_all_gather_runtime(
            group, device, max_rows=max(config.decode_tokens, draft_rows), columns=152576, ipc=ipc
        )
        context_collective = (create_all_gather_runtime(group, device,
            max_rows=tokens, columns=6144, ipc=ipc) if config.dflash else None)
        vision_collective = create_dma_all_reduce_runtime(group, device,
            max_tokens=PATCH_CAPACITIES[-1] // 4, hidden_size=6144, ipc=ipc)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": json.loads((snapshot / "config.json").read_text()),
                    "draft": json.loads((snapshot / "dflash/config.json").read_text()),
                    "headers": [s.header_sha256 for s in manifest.shards],
                    "world_size": shard.world_size,
                    "contract": "mimo_v2.graph-inputs.v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            device=device,
            checkpoint=identity,
            workload_schema="mimo_v2-vision",
            numerical_policy="default",
            coordinator=coordinator,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=package_storage,
        )
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        # The recipe declares no stream overlap: every branch runs on the main stream.
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        prefill_storage = Scratch.allocate(weights.config, shard, config.prefill_chunk_size, 1, device,
                                           buffers=graph_buffers.cursor())
        decoder_storage = Scratch.allocate(
            weights.config, shard, config.decode_tokens, config.decode_tokens, device, buffers=graph_buffers.cursor()
        )
        vision_inputs = VisionInputs.allocate(device)
        vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
        vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency, image_token=weights.config.text["image_token_id"],
                                   features=FEATURES, hidden=HIDDEN)
        vision_storage = {}
        environment = DecoderEnvironment.load(snapshot, vision_state)
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
        if config.dflash:
            speculation = SpeculativePreparation(
                weights,
                environment,
                pool,
                requests,
                inputs,
                decoder_storage,
                completion=completion,
                frame=frame,
                sequence=sequence,
                session=session,
                workspace=arena,
                runtime=collective,
                vocabulary_runtime=vocabulary_collective,
                context_runtime=context_collective,
                coordinator=coordinator,
            )

    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    del manifest
    modes = (True, False) if config.dflash else (False,)
    count = (
        len(modes) * (prefill_graph_count(config.prefill_capacities, scoring) + 4 * len(config.batch_capacities))
        + len(config.speculative_shapes)
        + len(config.batch_capacities)
        + len(PATCH_CAPACITIES)
    )
    with phase(progress, "Construct graph catalog", total=count) as stage:

        def add(name, forward, kind, rows, tokens, fields, *, enabled=None):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dflash and not enabled else ""
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
                        environment,
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
                        context_factory=None if speculation is None else speculation.forwards.context,
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
            for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
                name = f"speculative{batch}x{width}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                                                     buffers=speculation.sampling_buffers.cursor())
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, enabled=True)
        for capacity in reversed(PATCH_CAPACITIES):
            name = f"vision{capacity}"
            cursor = graph_buffers.cursor()
            vision_storage[capacity] = VisionScratch.allocate(capacity, shard, device, buffers=cursor)
            encoder = PreparedVision(
                weights.vision,
                vision_storage[capacity],
                vision_inputs,
                runtime=vision_collective,
                coordinator=coordinator,
                session=session,
                workspace=arena,
                name=name,
                buffers=cursor,
            )
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dflash and not enabled else ""
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
    assets = {
        role: {"path": str(path)}
        for role, path in (
            ("tokenizer", snapshot / "tokenizer.json"),
        )
    }
    assets["chat_template"] = annotated_template_asset(
        (snapshot / "chat_template.jinja").read_bytes(),
        Path(__file__).with_name("assets") / "chat_template.jinja")
    cache_regions = [history_region(buffers, "cache." + name, tensor, contract_proto.PAGED_KEY_VALUE)
        for name, tensor in pool.pages.items()]
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
        "frontend": prepare_frontend(snapshot,
            eos_token_ids=json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"],
            image_profile="mimo"),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency, page_tokens=PAGE_TOKENS, draft_lag=0 if config.dflash else None, regions=cache_regions),
    }
    memory = {
        "rank": rank,
        "world_size": shard.world_size,
        "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes,
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
    if config.dflash:
        server_config.update(
            speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width,
        )
        memory.update(speculative_shapes=config.speculative_shapes)
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
        if hasattr(forward, "decoder"):
            disposable.append(forward.decoder.scratch_tensors())
        if getattr(forward, "context", None) is not None:
            disposable.append(forward.context.scratch_tensors())
    return PreparedModel(rebuild_weights={"weights": weights},
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
            context_collective,
            vision_collective,
            coordinator,
            speculation,
        ),
        close_resources=tuple(
            c
            for c in (scoring, context_collective, vision_collective, prefill_collective, vocabulary_collective, collective)
            if c is not None
        ),
    )
