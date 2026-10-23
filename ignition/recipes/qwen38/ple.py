"""Compact PLE row delivery and prepared n-gram residual contribution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields

import torch
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import ple_embedding, short_conv
from safetensors import safe_open

from ignition.components.linear import UnquantizedProjection
from ignition.preparation.checkpoint import CheckpointManifest
from ignition.progress import phase
from ignition.preparation.qwen38 import PLE_PREFIX, TEXT_PREFIX
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer

PLE_LAYER_PREFIX = f"{TEXT_PREFIX}.layers.1.ple"
TABLE_ROWS = 320001536
ROWS_PER_PLANE = 2500012
TABLE_BYTES = TABLE_ROWS * 90


def _read_small(manifest: CheckpointManifest, name: str) -> torch.Tensor:
    metadata = manifest.tensors[name]
    if metadata.nbytes > 1024:
        raise ValueError(f"{name} exceeds the PLE geometry/scalar read limit")
    from ignition.preparation.checkpoint_source import read_small_tensor, source_for

    if source_for(manifest.snapshot) is not None:
        return read_small_tensor(manifest, name, limit=1024)
    with safe_open(
        manifest.snapshot / metadata.shard, framework="pt", device="cpu"
    ) as source:
        view = source.get_slice(name)
        if (
            tuple(view.get_shape()) != metadata.shape
            or view.get_dtype() != metadata.dtype
        ):
            raise ValueError(f"{name} metadata changed after checkpoint inspection")
        return source.get_tensor(name).clone()


def _caps(config: dict, device, tokens: int, sequences: int, *, rank=0, world_size=1):
    eos = config["eos_token_id"]
    return ple_embedding.Caps(
        device=device,
        max_tokens=tokens,
        max_seqs=sequences,
        vocab_size=config["vocab_size"],
        eos_token_id=eos[0] if isinstance(eos, list) else eos,
        max_order=config["ngram_size"],
        heads_per_order=config["heads_per_ngram"],
        dense_layer_ordinal=0,
        base_table_size=config["ngram_vocab_size_base"],
        embedding_dim=config["ple_embed_dim"],
        tp_size=world_size,
        tp_rank=rank,
        table_alignment=config["make_ngram_vocab_size_divisible_by"],
        quant_mode="nvfp4_group16",
        table_memory="io_uring",
    )


def table_layout(manifest: CheckpointManifest, device, *, rank=0, world_size=1):
    """Validate exact checkpoint geometry without allocating its table."""
    config = json.loads((manifest.snapshot / "config.json").read_text())["text_config"]
    caps = _caps(config, torch.device(device), 1, 1, rank=rank, world_size=world_size)
    geometry = ple_embedding.compute_geometry(
        caps,
        prime_sizes=_read_small(manifest, f"{PLE_PREFIX}.ngram_heads_vocab_sizes"),
        table_offsets=_read_small(manifest, f"{PLE_PREFIX}.ngram_heads_offsets"),
        multipliers=_read_small(manifest, f"{PLE_PREFIX}.layer_multipliers"),
    )
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    if layout.weight_shape != (TABLE_ROWS // world_size, 80) or layout.weight_scale_shape != (
        TABLE_ROWS // world_size,
        10,
    ):
        raise ValueError(
            "PLE checkpoint requires 128 packed planes totaling 320001536 rows"
        )
    return config, layout


@dataclass
class PLETable:
    config: dict
    layout: ple_embedding.TableLayout
    geometry: ple_embedding.GeometryTensors
    scale: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor
    hashes: torch.Tensor
    mailbox: object
    ready: object
    sources: tuple
    global_scale: float
    sequence: torch.Tensor | None = None
    timings: object = None

    @classmethod
    def load(cls, manifest, device, *, tokens, progress=None, host_allocator=None):
        from b12x.sequence._shared.disk_table import MappedHostAllocation
        from ignition.preparation.runtime_package import FileSource

        device = torch.device(device)
        config, layout = table_layout(manifest, device)
        scale = _read_small(manifest, f"{PLE_PREFIX}.ngram_embedding.weight_scale_2")
        if scale.dtype != torch.float32 or scale.shape != (1,) or not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("PLE global scale must be a finite positive FP32 scalar")
        sources = []
        with phase(progress, "Prepare embedding tables") as stage:
            for suffix, columns, dtype in (("weight", 80, "U8"), ("weight_scale", 10, "F8_E4M3")):
                plane = []
                for shard in range(TABLE_ROWS // ROWS_PER_PLANE):
                    name = f"{PLE_PREFIX}.ngram_embedding.shard_{shard}.{suffix}"
                    metadata = manifest.tensors[name]
                    if metadata.shape != (ROWS_PER_PLANE, columns) or metadata.dtype != dtype:
                        raise ValueError(f"{name}: invalid PLE table geometry or dtype")
                    plane.append(FileSource(manifest.snapshot / metadata.shard, metadata.offset, metadata.nbytes))
                sources.append(tuple(plane))
            heads = layout.caps.head_count
            shape = (tokens * heads + 1,)
            mailbox = (MappedHostAllocation(shape, torch.int64, device) if host_allocator is None
                       else host_allocator("ple_requests", shape, torch.int64))
            try:
                mailbox.host_view.zero_()
                ready = torch.cuda.Event(external=True)
                ready.record(torch.cuda.current_stream(device))
                table = cls(config, layout, ple_embedding.allocate_geometry(layout.geometry, device=device),
                    scale.to(device), torch.zeros((tokens * heads, 80), dtype=torch.uint8, device=device),
                    torch.zeros((tokens * heads, 10), dtype=torch.uint8, device=device),
                    torch.full((tokens, heads), -1, dtype=torch.int64, device=device), mailbox, ready, tuple(sources),
                    float(scale.item()))
            except BaseException:
                mailbox.close()
                raise
            stage.update(f"{TABLE_BYTES / (1 << 30):.1f} GiB table")
        return table

    def buffers(self):
        return dict(ple_weights=self.weights, ple_scales=self.scales, ple_hashes=self.hashes,
                    ple_requests=self.mailbox.device_view, **self.timings.buffers())

    def events(self):
        return {"ple_ready": self.ready.cuda_event}

    def export_metadata(self):
        """Describe compact hash geometry and checkpoint spans for native publication."""
        hashes = dict(method="EOS_XOR_MODULO", orders=[order for order in range(2, self.config["ngram_size"] + 1)
                      for _ in range(self.config["heads_per_ngram"])])
        for name in ("multipliers", "prime_sizes", "table_offsets"):
            target = {"prime_sizes": "primes", "table_offsets": "offsets"}.get(name, name)
            hashes[target] = list(getattr(self.layout.geometry, name))
        table = dict(layer=1, rows=TABLE_ROWS, embedding_width=160, scale_block=16,
            encoding="NVFP4_E4M3_GLOBAL", weight_buffer="ple_weights", scale_buffer="ple_scales",
            hash_buffer="ple_hashes", ready_event="ple_ready", request_buffer="ple_requests",
            global_scale=self.global_scale, hash=hashes)
        for name, sources in zip(("weights", "scales"), self.sources, strict=True):
            table[name] = [dict(path=str(source.path), offset=source.offset, nbytes=source.nbytes)
                           for source in sources]
        return dict(pad_token_id=self.layout.caps.eos_token_id,
                    compressed_vocabulary=self.config["vocab_size"], tables=[table])

    def close(self):
        self.mailbox.close()


@dataclass(frozen=True)
class PLEMetadata:
    """Fixed device inputs for packed n-gram lookup and convolution history.

    ``committed_history`` contains each request's preceding two accepted token
    IDs, padded with the model's EOS ID. The complete model graph gathers it
    from persistent request history before lookup and updates that history after
    consuming tokens. PLE lookup itself only reads these packed rows.
    """

    token_ids: torch.Tensor
    query_start_loc: torch.Tensor
    committed_history: torch.Tensor
    convolution_slots: torch.Tensor
    fresh: torch.Tensor
    accepted_tokens: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls, tokens: int, sequences: int, device, *, eos_token_id: int = 248044
    ) -> PLEMetadata:
        if any(type(value) is not int or value <= 0 for value in (tokens, sequences)):
            raise ValueError("PLE token and sequence capacities must be positive")
        return cls(
            torch.full((tokens,), eos_token_id, dtype=torch.int64, device=device),
            torch.zeros((sequences + 1,), dtype=torch.int32, device=device),
            torch.full((sequences, 2), eos_token_id, dtype=torch.int64, device=device),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            torch.zeros((sequences,), dtype=torch.bool, device=device),
            torch.ones((sequences,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )


@dataclass(frozen=True)
class PLEScratch:
    embedding: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gated: torch.Tensor
    normalized: torch.Tensor
    convolved: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, buffers=None) -> PLEScratch:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("PLE token capacity must be positive")
        return cls(
            *(
                (torch.zeros((tokens, width), dtype=torch.bfloat16, device=device) if buffers is None
                 else buffers.zeros((tokens, width), torch.bfloat16))
                for width in (2560, 10240, 2560, 10240, 10240, 10240, 10240)
            )
        )


def allocate_convolution_state(
    slots: int, device, *, speculative_tokens: int = 0
) -> torch.Tensor:
    if (
        type(slots) is not int
        or slots <= 0
        or type(speculative_tokens) is not int
        or speculative_tokens < 0
    ):
        raise ValueError(
            "PLE needs positive state capacity and nonnegative speculative capacity"
        )
    return torch.zeros(
        (slots, 10240, 9 + speculative_tokens), dtype=torch.bfloat16, device=device
    )


def _validate_geometry(module: torch.nn.Module) -> None:
    if (module.hidden_size, module.hc_count, module.short_conv_state_len) != (
        2560,
        4,
        9,
    ):
        raise ValueError("PLE geometry differs from Qwen3.8 Flash Next")
    if (
        tuple(module.conv1d.weight.shape) != (10240, 1, 4)
        or module.conv1d.bias is not None
    ):
        raise ValueError("PLE requires unbiased width-10240 kernel-4 convolution")
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = getattr(module, name)
        if (
            norm.weight.shape != (10240,)
            or norm.weight.dtype != torch.float32
            or norm.group_size != 2560
        ):
            raise ValueError("PLE requires exact FP32 zero-centered group norm weights")


class PreparedPLE:
    """Recorded PLE lookup, BF16 projections, canonical gates, and convolution.

    Canonical normalization and gate tensor expressions preserve FP32 norm
    weights and the BF16 rounding of each gate operation. Their intermediates
    stay inside the complete forward graph. Exported contribution and history
    storage is caller-owned and remains valid across shared graph-pool replays.
    """

    def __init__(
        self,
        module,
        projections,
        source,
        scratch: PLEScratch,
        convolution_state,
        metadata: PLEMetadata,
        table: PLETable,
        *,
        mode: str,
        session: PreparationSession,
        name: str,
        workspace=None,
        partitions=None,
        runtime=None,
        coordinator=None,
    ):
        _validate_geometry(module)
        if mode not in ("decode", "prefill"):
            raise ValueError("PLE graph mode must be decode or prefill")
        if (
            source.ndim not in (2, 3)
            or math.prod(source.shape[1:]) != 10240
            or source.dtype != torch.bfloat16
            or not source.is_contiguous()
        ):
            raise ValueError(
                "PLE source must be contiguous BF16 [tokens,10240] or [tokens,4,2560]"
            )
        if source.ndim == 3 and tuple(source.shape[1:]) != (4, 2560):
            raise ValueError("PLE residual streams must have shape [tokens,4,2560]")
        self.module, self.table, self.scratch, self.metadata = (
            module,
            table,
            scratch,
            metadata,
        )
        self.mode, self.source, self.state = mode, source, convolution_state
        self.tokens, self.device = source.shape[0], source.device
        if self.device != table.layout.caps.device or self.device.type != "cuda":
            raise ValueError("PLE table and graph operands must share a CUDA device")
        for field, width in zip(
            fields(scratch), (2560, 10240, 2560, 10240, 10240, 10240, 10240)
        ):
            tensor = getattr(scratch, field.name)
            if (
                tuple(tensor.shape) != (self.tokens, width)
                or tensor.dtype != torch.bfloat16
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"PLE scratch {field.name} has incompatible geometry")
        if (
            convolution_state.ndim != 3
            or convolution_state.shape[1] != 10240
            or convolution_state.shape[2] < 9
        ):
            raise ValueError("PLE convolution history requires [slots,10240,9+drafts]")
        if metadata.num_seqs.item() or metadata.num_tokens.item():
            raise ValueError("PLE preparation requires zero live counts")
        if set(projections) != {"key_proj", "value_proj"}:
            raise ValueError("PLE requires key_proj and value_proj BF16 weights")
        for parameter in (
            module.conv1d.weight,
            module.norm_key.weight,
            module.norm_query.weight,
            module.norm_conv.weight,
        ):
            if parameter.device != self.device:
                raise ValueError("canonical PLE parameters must share the graph device")
        sequences = metadata.query_start_loc.numel() - 1
        caps = _caps(table.config, self.device, self.tokens, sequences,
                     rank=table.layout.caps.tp_rank, world_size=table.layout.caps.tp_size)
        from ignition.components.ngram_delivery import PreparedPLELookup

        self.embedding_lookup = PreparedPLELookup(table, metadata, scratch.embedding, caps,
            session=session, name=f"{name}.embedding", runtime=runtime, coordinator=coordinator)
        self.embedding_output = scratch.embedding
        self.key_projection = UnquantizedProjection(
            self.embedding_output,
            projections["key_proj"],
            scratch.key,
            session=session,
            name=f"{name}.key",
        )
        self.value_projection = UnquantizedProjection(
            self.embedding_output,
            projections["value_proj"],
            scratch.value,
            session=session,
            name=f"{name}.value",
        )
        self.partitions = partitions
        if partitions is None:
            args = {
                "x": scratch.normalized,
                "conv_weight": module.conv1d.weight.squeeze(1),
                "query_start_loc": metadata.query_start_loc,
                "state_slot_ids": metadata.convolution_slots,
                "state_is_fresh": metadata.fresh,
                "num_accepted_tokens": metadata.accepted_tokens,
                "num_seqs": metadata.num_seqs,
                "num_tokens": metadata.num_tokens,
                "conv_state": convolution_state,
                "out": scratch.convolved,
            }
            self.convolution_plan = short_conv.plan(
                short_conv.Caps(
                    device=self.device,
                    mode=mode,
                    channels=10240,
                    kernel_size=4,
                    dilation=3,
                    max_tokens=self.tokens,
                    max_seqs=sequences,
                    max_state_slots=convolution_state.shape[0],
                    max_speculative_tokens=convolution_state.shape[2] - 9,
                ),
                invocation=short_conv.invocation_from_tensors(**args),
            )
            self.convolution_binding = self._prepare_binding(
                self.convolution_plan,
                args,
                session,
                f"{name}.convolution",
                short_conv.bind,
                bind_plan=True,
                attribute="convolution_binding",
                workspace=workspace,
            )
        else:
            self.convolutions = tuple(PreparedPLEConvolution(module, part.scratch, convolution_state,
                part.metadata, mode=part.mode, session=session, name=f"{name}.{part.mode}", workspace=workspace)
                for part in partitions.parts)
        self.token_rows = torch.arange(
            self.tokens, dtype=torch.int32, device=self.device
        ).view(-1, 1)
        self.sequence_rows = torch.arange(
            sequences, dtype=torch.int32, device=self.device
        ).view(1, -1)

    def _prepare_binding(self, plan, args, session, name, public_bind, *,
                         attribute, bind_plan=False, workspace=None):

        def prepare_call(prepared):
            specs = plan.scratch_specs() if bind_plan else prepared.layout.scratch_specs()
            scratch = planned_scratch(specs, workspace)
            binding = prepared.bind(
                scratch=scratch, **args, **({"plan": plan} if bind_plan else {})
            )
            return PreparedCall(run=lambda: prepared.run(binding), owners=prime_owners(workspace, scratch, binding))

        declare(session, (plan.request(name=name, prepare_call=prepare_call),))

        def refresh():
            scratch = planned_scratch(plan.scratch_specs(), workspace)
            setattr(self, attribute, public_bind(plan, scratch=scratch, **args))
        setattr(self, attribute, None)

        register_finalizer(session, refresh, workspace=workspace)
        return getattr(self, attribute)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source is not self.source and (
            source.data_ptr() != self.source.data_ptr()
            or source.shape != self.source.shape
        ):
            raise ValueError("prepared PLE requires its fixed source storage")
        scratch, metadata = self.scratch, self.metadata
        self.embedding_lookup(source)
        self.key_projection(self.embedding_output)
        self.value_projection(self.embedding_output)
        key = self.module.norm_key(scratch.key).view(self.tokens, 4, 2560)
        query = self.module.norm_query(source.view(self.tokens, 10240)).view(
            self.tokens, 4, 2560
        )
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(2560)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        scratch.gated.copy_(
            (torch.sigmoid(gate) * scratch.value.unsqueeze(1)).flatten(1)
        )
        scratch.normalized.copy_(self.module.norm_conv(scratch.gated))
        live = (
            (self.token_rows >= metadata.query_start_loc[:-1].view(1, -1))
            & (self.token_rows < metadata.query_start_loc[1:].view(1, -1))
            & (self.sequence_rows < metadata.num_seqs)
            & (metadata.convolution_slots.view(1, -1) >= 0)
            & (self.token_rows < metadata.num_tokens)
        ).any(dim=1, keepdim=True)
        scratch.gated.masked_fill_(~live, 0)
        scratch.normalized.masked_fill_(~live, 0)
        if self.partitions is None:
            short_conv.run(self.convolution_binding)
        else:
            for part, convolution in zip(self.partitions.parts, self.convolutions):
                part.gather(scratch.normalized, part.scratch.normalized)
                convolution()
            self.partitions.combine(tuple(part.scratch.convolved for part in self.partitions.parts), scratch.convolved)
        torch.add(scratch.gated, scratch.convolved, out=scratch.output)
        return scratch.output


class PreparedPLEConvolution:
    """Phase-specific history update after the shared PLE projections and gates."""

    def __init__(self, module, scratch, state, metadata, *, mode, session, name, workspace):
        from ignition.components.recurrent import prepare_sequence_binding

        caps = short_conv.Caps(device=scratch.normalized.device, mode=mode, channels=10240,
            kernel_size=4, dilation=3, max_tokens=scratch.normalized.shape[0],
            max_seqs=metadata.query_start_loc.numel() - 1, max_state_slots=state.shape[0],
            max_speculative_tokens=state.shape[2] - 9)
        args = dict(x=scratch.normalized, conv_weight=module.conv1d.weight.squeeze(1),
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens, conv_state=state,
            out=scratch.convolved)
        plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**args))
        self.binding = prepare_sequence_binding(plan, args, session=session, name=name,
            public_bind=short_conv.bind, mode=mode, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "binding", binding))

    def __call__(self):
        short_conv.run(self.binding)


__all__ = [
    "PLETable",
    "PLEMetadata",
    "PLEScratch",
    "PreparedPLE",
    "allocate_convolution_state",
    "table_layout",
]
