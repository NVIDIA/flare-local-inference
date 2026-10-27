"""Prepare a single- or multi-rank Gemma 4 text and Assistant MTP graph catalog."""

import contextlib
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import (
    batch_capacities,
    prefill_capacities,
    request_capacity,
    speculative_shapes,
    speculative_token_capacity,
)
from ignition.components.completion import CompletionOutputs
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.parallel import TensorParallelShard
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill,
    PromptScoring,
    prefill_graph_count,
    prefill_modes,
)
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
    create_roce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES, release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import PAGE_TOKENS
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, Scratch, SoftcappedResume
from .pool import ContinuationPool, continuation_layout, page_bytes
from .speculative import PreparedContext, PreparedRound
from .weights import ModelWeights


def chat_template_asset(hidden_size: int):
    """Use Google's canonical template for the model size across quantization publishers."""
    if hidden_size in (1536, 2560):
        filename = "chat_template.google-e2b-e4b.jinja"
    elif hidden_size in (2816, 3840, 5376):
        filename = "chat_template.google.jinja"
    else:
        raise ValueError(f"unsupported Gemma 4 chat template hidden size: {hidden_size}")
    path = Path(__file__).with_name("assets").joinpath(filename).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    compile_headroom_bytes: int = COMPILE_HEADROOM_BYTES
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    tuning_replica: TuningReplica | None = None
    device: int = 0
    devices: tuple[int, ...] | None = None
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    minimum_pages: int = 8
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    draft_checkpoint: Path | None = None
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None

    @property
    def speculative_shapes(self):
        return (
            speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
            if self.draft_checkpoint is not None
            else ()
        )

    @property
    def decode_tokens(self):
        return max((b * w for b, w in self.speculative_shapes), default=request_capacity(self.max_active))

    def __post_init__(self):
        if self.roce_ring and (self.hosts is None or not 2 <= len(self.hosts) <= 16):
            raise ValueError("RoCE ring requires 2..16 ranks on separate hosts")
        if self.devices is not None:
            placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
            if (
                not isinstance(self.devices, tuple)
                or len(self.devices) not in (1, 2, 4)
                or len(set(placements)) != len(self.devices)
                or (
                    self.hosts is not None
                    and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                )
                or any(type(d) is not int or d < 0 for d in self.devices)
            ):
                raise ValueError("Gemma 4 requires one, two, or four distinct CUDA devices")
        elif self.hosts is not None:
            raise ValueError("ranks on separate hosts require their devices")
        request_capacity(self.max_active)
        if not self.speculative_widths or max(self.speculative_widths) > 8:
            raise ValueError("Gemma 4 Assistant supports at most seven draft tokens")
        if (
            self.speculative_fixed_width is not None
            and self.speculative_fixed_width not in self.speculative_widths
        ):
            raise ValueError("fixed verification width must belong to the captured widths")
        _ = self.speculative_shapes
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Gemma 4 context must be 128-aligned in [128,262144]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("Gemma 4 prefill chunk must be 128-aligned and fit the context")
        if type(self.minimum_pages) is not int or self.minimum_pages <= 0:
            raise ValueError("minimum page capacity must be a positive integer")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")


@torch.no_grad()
def prepare_rank(snapshot, config, *, rank=0, group=None, progress=None, package_storage=None):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, 1 if config.devices is None else len(config.devices))
    if (shard.world_size > 1) != (group is not None):
        raise ValueError("Gemma 4 tensor-parallel preparation requires an established control group")
    device_id = config.device if config.devices is None else config.devices[rank]
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    if package_storage is not None:
        transient.enter_context(package_storage.transient())
    draft_checkpoint = config.draft_checkpoint
    if draft_checkpoint is not None and not (Path(draft_checkpoint) / "config.json").is_file():
        from huggingface_hub import snapshot_download
        from .config import ASSISTANT_REPOSITORIES, ModelConfig

        repo, rev = ASSISTANT_REPOSITORIES[ModelConfig.load(snapshot).hidden]
        draft_checkpoint = snapshot_download(repo, revision=rev, allow_patterns=("config.json", "model.safetensors"))
    weights = ModelWeights.load(
        manifest,
        device,
        shard=shard,
        progress=progress,
        draft_checkpoint=draft_checkpoint,
    )
    if torch.cuda.get_device_properties(device).is_integrated:
        from ignition.preparation.host_memory import release_page_cache

        release_page_cache(manifest.shard_paths)
    drafting = weights.draft is not None
    release_unused()
    effective_context = min(config.context, weights.config.text["max_position_embeddings"])
    effective_context = (effective_context // PAGE_TOKENS) * PAGE_TOKENS
    if config.prefill_chunk_size > effective_context:
        raise ValueError("Gemma 4 prefill chunk size exceeds the checkpoint's max_position_embeddings")
    layout = continuation_layout(weights.config, shard)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if package_storage is None:
        snapshots = min(snapshots, 256)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Gemma 4 continuation")
    slots, blocks = request_capacity(config.max_active), effective_context // PAGE_TOKENS
    decode_tokens = config.decode_tokens
    tokens = max(config.prefill_chunk_size, decode_tokens)
    state_rows = slots + snapshots
    prep_pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    # Development serving keeps this bounded cache; exported packages size KV from the serving host's memory.
    pages = max(prep_pages, config.minimum_pages)
    initial_pages = prep_pages if package_storage is not None else pages
    bytes_per_page = page_bytes(weights.config, shard)
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(
            weights.config,
            state_rows + 1,
            initial_pages,
            device,
            shard=shard,
            storage=package_storage,
            maximum_pages=max(pages, total_memory // bytes_per_page),
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots) if package_storage is not None else None,
        )
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24), device=device)
        coordinator = CollectivePreparation(group) if group is not None else None
        ipc = None if package_storage is None else package_storage.peer_storage
        collective = prefill_collective = vocabulary_collective = None
        local_vocab = weights.config.vocabulary_partition(shard)[2]
        if coordinator is not None and coordinator.spans_hosts:
            gather_bytes = decode_tokens * local_vocab * 2
            collective = prefill_collective = vocabulary_collective = create_roce_runtime(
                group,
                device,
                max_tokens=decode_tokens,
                pipe_tokens=tokens,
                hidden_size=weights.config.hidden,
                gather_bytes=gather_bytes,
                ring=config.roce_ring,
                package_storage=package_storage,
            )
        elif group is not None:
            collective = create_all_reduce_runtime(
                group, device, collective_tuning=config.collective_tuning, max_tokens=tokens, hidden_size=weights.config.hidden, ipc=ipc
            )
            prefill_collective = create_dma_all_reduce_runtime(
                group,
                device,
                max_tokens=config.prefill_chunk_size,
                hidden_size=weights.config.hidden,
                ipc=ipc,
            )
            vocabulary_collective = create_all_gather_runtime(
                group,
                device,
                max_rows=decode_tokens,
                columns=local_vocab * shard.world_size,
                ipc=ipc,
            )
        collectives = {
            "runtime": collective,
            "vocabulary_runtime": vocabulary_collective,
            "coordinator": coordinator,
        }
        identity = hashlib.sha256(
            json.dumps(
                {
                    "architecture": weights.config.architecture,
                    "config": weights.config.text,
                    "draft": weights.config.draft.text if drafting else None,
                    "headers": [s.header_sha256 for s in manifest.shards],
                    "world_size": shard.world_size,
                    "contract": "gemma4.graph-inputs.v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        session = RecordedPreparationSession(
            collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder,
            compile_workers=config.compile_workers, native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            compile_headroom_bytes=config.compile_headroom_bytes,
            tuning_replica=config.tuning_replica,
            device=device,
            checkpoint=identity,
            workload_schema="gemma4",
            numerical_policy="default",
            coordinator=coordinator,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=package_storage,
        )
        session.__enter__()
        register_benchmark_state(
            session,
            pool.reserved_backing,
            storage=pool.reserved_backing,
            restore=pool.clear_for_preparation,
        )
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch_pool = contextlib.nullcontext() if package_storage is None else package_storage.scratch_pool()
        scratch_buffers, graph_buffers, layer_buffers = (
            GraphBufferPool(device),
            GraphBufferPool(device),
            GraphBufferPool(device),
        )
        buffers = dict(
            inputs.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
        )
        speculation = None
        if drafting:
            speculation = SimpleNamespace(
                weights=weights,
                pool=pool,
                inputs=inputs,
                requests=requests,
                session=session,
                workspace=arena,
                collectives=collectives,
                scratch_pool=scratch_pool,
                scratch_buffers=scratch_buffers,
                layer_buffers=layer_buffers,
                completion=completion,
                frame=frame,
                sequence=sequence,
                rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device),
            )
    scoring = PromptScoring.allocate(
        config,
        device,
        buffers,
        frame=frame,
        group=group,
        ipc=ipc,
        runtime=collective,
        coordinator=coordinator,
    )
    forwards, signatures, metadata = {}, {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    modes = (True, False) if drafting else (False,)
    count = (
        len(modes) * (prefill_graph_count(prefill_sizes, scoring) + 4 * len(batch_sizes))
        + len(batch_sizes)
        + len(config.speculative_shapes)
    )
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens_count, fields, *, enabled=False):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens_count, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for kind, mode, sizes in prefill_modes(
                prefill_sizes, () if enabled else reversed(batch_sizes), scoring
            ):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    cursor = graph_buffers.cursor()
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=cursor)
                    with scratch_pool:
                        scratch = Scratch.allocate(
                            capacity, rows, weights.config, scratch_buffers.cursor(), shard=shard
                        )
                    decoder = PreparedDecoder(
                        mode_weights,
                        scratch,
                        m,
                        pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        buffers=cursor,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 1024 else collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        workspace=arena,
                        layer_buffers=layer_buffers,
                    )
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool, buffers=cursor)
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
                        buffers=cursor,
                        context_factory=PreparedContext if enabled else None,
                    )
                    add(
                        name,
                        forward,
                        kind,
                        rows,
                        capacity if mode == "prefill" else 1,
                        forward_inputs,
                        enabled=enabled,
                    )
        if speculation is not None:
            for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
                name = f"speculative{batch}x{width}"
                forward = PreparedRound(speculation, batch, width, name, buffers=graph_buffers.cursor())
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, enabled=True)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, draft=None)
            suffix = "-no-mtp" if drafting and not enabled else ""
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = SoftcappedResume(
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
                    activation_mode="auto",
                    buffers=graph_buffers.cursor(),
                    softcap=weights.config.final_logit_softcapping,
                )
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
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
                        buffers=graph_buffers.cursor(),
                    )
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream = torch.cuda.Stream(device=device)
    graph_pool = torch.cuda.graph_pool_handle() if package_storage is None else package_storage.graph_pool.id
    if config.compile_decoder and package_storage is not None:
        scratch_pool.release_backing()

    def materialize():
        if package_storage is not None:
            scratch_pool.materialize()
            if pages > initial_pages:
                package_storage.grow("KV_PAGES", pages)
                pool.memory_capacity[1] = pages
        pool.clear_for_preparation()

    try:
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
            materialize=materialize,
            warmup_storage=package_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative"),
        )
    finally:
        if package_storage is not None:
            scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    assets = {
        "tokenizer": {
            "path": str(snapshot / "tokenizer.json"),
            "sha256": hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest(),
        },
        "chat_template": chat_template_asset(weights.config.hidden),
    }
    raw_eos = []
    gen_path = snapshot / "generation_config.json"
    if gen_path.is_file():
        gen_eos = json.loads(gen_path.read_text()).get("eos_token_id")
        if gen_eos is not None:
            raw_eos.extend(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    text_eos = weights.config.text.get("eos_token_id")
    if text_eos is not None:
        raw_eos.extend(text_eos if isinstance(text_eos, list) else [text_eos])
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    eos_text = tokenizer["eos_token"]
    if isinstance(eos_text, dict):
        eos_text = eos_text["content"]
    from tokenizers import Tokenizer

    tokenizer_eos = Tokenizer.from_file(str(snapshot / "tokenizer.json")).token_to_id(eos_text)
    if tokenizer_eos is not None:
        raw_eos.append(tokenizer_eos)
    eos_ids = list(dict.fromkeys(raw_eos))
    server_config = dict(
        identity=identity,
        model_id=snapshot.name,
        max_sequence_tokens=effective_context,
        prefill_chunk_size=config.prefill_chunk_size,
        pages=pages,
        state_rows=state_rows,
        snapshots=snapshots,
        page_tokens=PAGE_TOKENS,
        frontend=prepare_frontend(snapshot, eos_token_ids=eos_ids),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(),
        assets=assets,
        model_contract=prepare_contract(
            signatures, buffers, pool, vocabulary=vocabulary, world_size=shard.world_size, mtp=drafting
        ),
    )
    if drafting:
        server_config.update(
            speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity
            ),
            speculative_fixed_width=config.speculative_fixed_width,
        )
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(
        rank=rank,
        world_size=shard.world_size,
        device=device_id,
        weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0,
        offloaded_bytes=0,
        continuation_bytes=pool.backing.nbytes,
        state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages,
        page_bytes=bytes_per_page,
        page_tokens=PAGE_TOKENS,
        kv_bytes=pages * bytes_per_page,
        completion_output_bytes=completion.nbytes,
        operator_scratch_bytes=arena.nbytes,
        graphs=len(graphs),
        prefill_capacities=prefill_sizes,
        free_device_bytes=free,
        total_device_bytes=total,
        headroom_bytes=config.headroom_bytes,
        tuning=tuning,
        compiled_decoder=config.compile_decoder,
    )
    if drafting:
        memory.update(
            speculative_shapes=config.speculative_shapes,
            speculative_token_capacity=server_config["speculative_token_capacity"],
        )
    if scoring is not None:
        memory.update(scoring.memory_report())
    seen_ids = set()
    close_resources = []
    for res in (scoring, prefill_collective, vocabulary_collective, collective):
        if res is not None and id(res) not in seen_ids:
            seen_ids.add(id(res))
            close_resources.append(res)
    return PreparedModel(
        rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'ple', 'draft'),
            aliases=(('weights.head', 'weights.embedding'),) if weights.head is weights.embedding else ()),
        device=device_id,
        graphs=graphs,
        buffers=buffers,
        signatures=signatures,
        max_active=config.max_active,
        session=session,
        server_config=server_config,
        memory_report=memory,
        close_resources=tuple(close_resources),
        export_zero_ranges=(
            *(() if scoring is None else scoring.zero_ranges()),
            *storage_ranges([
                arena.tensor,
                metadata,
                *[
                    chunk
                    for p in (scratch_buffers, graph_buffers, layer_buffers)
                    for chunks in p.chunks.values()
                    for chunk in chunks
                ],
            ]),
            *completion.zero_ranges(),
        ),
        resources=(
            scoring,
            weights,
            pool,
            requests,
            inputs,
            admission,
            commands,
            arena,
            forwards,
            scratch_buffers,
            graph_buffers,
            layer_buffers,
            metadata,
            compiled,
            stream,
            graph_pool,
            session,
            coordinator,
            collective,
            prefill_collective,
            vocabulary_collective,
            speculation,
            scratch_pool,
        ),
    )


def prepare(snapshot, config, *, progress=None, package_storage):
    return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, package_storage=package_storage)
