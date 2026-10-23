"""Prepared GLM KDA projections, convolution, recurrence and gated output norm."""

from dataclasses import dataclass, fields

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv

from ignition.components import sm103
from ignition.components.kda_speculation import KDARecords, _record, register_recording
from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.linear import prepare_projection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.recurrent import (
    DeltaMetadata,
    DeltaState,
    prepare_sequence_binding,
)


@dataclass(frozen=True)
class KDAScratch:
    projected_input: torch.Tensor
    projected_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    forget_input: torch.Tensor
    gate_input: torch.Tensor
    raw_g: torch.Tensor
    raw_beta: torch.Tensor
    z: torch.Tensor
    recurrence: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, heads, device):
        if type(tokens) is not int or tokens <= 0 or heads not in (16, 32, 64):
            raise ValueError("GLM KDA scratch requires positive capacity and TP1/TP2/TP4 head counts")
        width = heads * 128
        def buffer(channels):
            return torch.zeros((tokens, channels), dtype=torch.bfloat16, device=device)
        projected = buffer(3 * width + heads + 128)
        qkv, beta, forget = projected.split((3 * width, heads, 128), dim=-1)
        if tokens > 1:
            forget = buffer(128)
        return cls(projected, qkv, buffer(3 * width), forget, buffer(128), buffer(width), beta,
                   buffer(width), buffer(width), buffer(width), buffer(4096))

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("KDA rows exceed scratch capacity")
        views = {field.name: getattr(self, field.name)[:tokens] for field in fields(self)}
        if tokens == 1:
            views["forget_input"] = views["projected_input"][:, -128:]
        return KDAScratch(**views)


class PreparedGlmKDA:
    """One rank's KDA branch before the row-parallel output reduction.

    Decode and prefill use b12x recurrence kernels. Prefill output normalization
    fuses the canonical sigmoid-gated FP32 RMSNorm and zeroes inactive rows. Device
    metadata selects live rows and persistent state slots; zero live counts are
    required while preparing and tuning kernels.
    """

    def __init__(self, module, weights, source, scratch: KDAScratch, state: DeltaState,
                 metadata: DeltaMetadata, *, mode, session, name, workspace, checkpoint_export=False,
                 records=None):
        if (source.ndim != 2 or source.shape[1] != 4096 or source.dtype != torch.bfloat16
                or source.device.type != "cuda"):
            raise ValueError("GLM KDA requires BF16 [token_capacity, 4096] CUDA input")
        if mode not in ("decode", "prefill"):
            raise ValueError("KDA mode must be decode or prefill")
        if (module.num_heads != 64 or module.head_dim != 128 or module.conv_kernel_size != 4
                or module.o_norm.activation != "sigmoid" or module.forget_gate.safe_gate_lower_bound != -5.0):
            raise ValueError("canonical KDA geometry differs from GLM 5.3 Flash")
        self.heads = state.recurrent.shape[1]
        self.tokens, self.device = source.shape[0], source.device
        self.width = self.heads * 128
        self.sequences = metadata.query_start_loc.numel() - 1
        self.columns = metadata.state_indices.shape[1]
        self.drafts = state.convolution.shape[-1] - 3
        if (self.heads not in (16, 32, 64) or state.recurrent.shape[2:] != (128, 128)
                or state.convolution.shape[:2] != (state.recurrent.shape[0], 3 * self.width)):
            raise ValueError("GLM KDA state must match one TP1/TP2/TP4 head partition")
        if metadata.num_tokens.item() or metadata.num_seqs.item():
            raise ValueError("KDA preparation requires zero live counts")
        if mode == "decode" and (self.tokens > self.sequences * self.columns or self.columns > self.drafts + 1):
            raise ValueError("KDA decode capacity exceeds metadata or convolution history")
        if records is not None and mode != "decode":
            raise ValueError("KDA verification records belong to decode rounds")
        self.module, self.weights, self.scratch = module, weights, scratch
        self.records = self.verification = None
        self.state, self.metadata, self.mode = state, metadata, mode
        module.o_norm.weight = torch.nn.Parameter(weights["o_norm.weight"], requires_grad=False)
        self.projections = {}
        for key, input, output in (
            ("input_proj", source, scratch.projected_input),
            ("f_b_proj", scratch.forget_input, scratch.raw_g),
            ("g_a_proj", source, scratch.gate_input),
            ("g_b_proj", scratch.gate_input, scratch.z),
            ("o_proj", scratch.gated, scratch.output),
        ):
            self.projections[key] = prepare_projection(input, weights[key], output,
                session=session, name=f"{name}.{key}", workspace=workspace, activation_mode="auto",
                # Convolution consumes the projected rows after this staging has been copied out.
                staging=scratch.convolved_qkv if key == "input_proj" else None)
        # The output-gate chain runs beside the input projections, or after them on the main stream.
        self.gate_overlap = None
        if workspace.overlap:
            self.gate_overlap = ConcurrentProjectionChain(
                (self.projections["g_a_proj"], self.projections["g_b_proj"]),
                parallel_plans=(*self.projections["input_proj"].plans, self.projections["f_b_proj"].plan)
                    if hasattr(self.projections["input_proj"], "plans") else
                    (self.projections["input_proj"].plan, self.projections["f_b_proj"].plan),
                session=session, workspace=workspace)
        conv_args = {
            "x": scratch.projected_qkv, "conv_weight": weights["conv1d.weight"],
            "query_start_loc": metadata.query_start_loc, "state_slot_ids": metadata.convolution_slots,
            "state_is_fresh": metadata.fresh, "num_accepted_tokens": metadata.accepted_tokens,
            "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            "conv_state": state.convolution, "out": scratch.convolved_qkv,
        }
        conv_caps = short_conv.Caps(device=self.device, mode=mode, channels=3 * self.width,
            kernel_size=4, max_tokens=self.tokens, max_seqs=self.sequences,
            max_state_slots=state.convolution.shape[0], max_speculative_tokens=self.drafts)
        self.conv_plan = short_conv.plan(conv_caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        self.conv_binding = prepare_sequence_binding(self.conv_plan, conv_args,
            session=session, name=f"{name}.short_conv", public_bind=short_conv.bind,
            mode=mode, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "conv_binding", binding))
        common = {
            "raw_g": scratch.raw_g.view(self.tokens, self.heads, 128), "raw_beta": scratch.raw_beta,
            "A_log": weights["A_log"], "dt_bias": weights["dt_bias"].view(self.heads, 128),
            "recurrent_state": state.recurrent, "num_seqs": metadata.num_seqs, "num_tokens": metadata.num_tokens,
            # Decode writes normalized output straight into the output projection's input.
            "output": (scratch.gated if mode == "decode" else scratch.recurrence).view(self.tokens, self.heads, 128),
        }
        self.options = {"lower_bound": -5.0}
        if mode == "decode" and records is not None and sm103.is_sm103(self.device):
            # On SM103 Triton records verification (b12x's CuTe recording runs 1.6x slower there); the gated
            # output norm follows as a separate launch.
            from .kda_recovery import PreparedKDAVerification

            self.verification = PreparedKDAVerification(mixed_qkv=scratch.convolved_qkv, raw_g=common["raw_g"],
                raw_beta=scratch.raw_beta, A_log=weights["A_log"], dt_bias=common["dt_bias"],
                state=state.recurrent, metadata=metadata, output=scratch.recurrence, records=records,
                heads=self.heads)
            self.output_norm = PreparedGatedRMSNorm(scratch.recurrence.view(self.tokens, self.heads, 128),
                scratch.z.view(self.tokens, self.heads, 128), module.o_norm.weight,
                scratch.gated.view(self.tokens, self.heads, 128), metadata.num_tokens,
                eps=module.layer_norm_epsilon)
            self.recurrent_plan = None
            return
        if mode == "decode":
            self.options["eps"] = module.layer_norm_epsilon
            # Speculative rounds record each verified token instead of persisting its FP32 state.
            caps = gdn_decode.Caps(device=self.device, max_tokens=self.tokens, max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0], key_heads=self.heads, value_heads=self.heads,
                state_index_columns=self.columns, gate_activation="sigmoid", null_state_index=0,
                recover_speculative_state=records is not None)
            args = dict(common, mixed_qkv=scratch.convolved_qkv, z=scratch.z.view(self.tokens, self.heads, 128),
                norm_weight=module.o_norm.weight, query_start_loc=metadata.query_start_loc,
                num_accepted_tokens=metadata.accepted_tokens, state_indices=metadata.state_indices)
            if records is not None:
                args.update(correction_cache=records[0], kg_cache=records[1])
                self.records = records
                self.handle = register_recording(self)
            self.recurrent_plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **args))
            public_bind, bind_method = gdn_decode.bind_kda, "bind_kda"
        else:
            caps = kda_prefill.Caps(device=self.device, max_tokens=self.tokens, max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0], heads=self.heads, null_state_index=0,
                checkpoint_export=checkpoint_export)
            q, k, v = (t.view(self.tokens, self.heads, 128) for t in scratch.convolved_qkv.split(self.width, dim=-1))
            args = dict(common, q=q, k=k, v=v, cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices, final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices, checkpoint_offsets=metadata.checkpoint_offsets)
            self.recurrent_plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            public_bind, bind_method = kda_prefill.bind, "bind"
        self.recurrence = prepare_sequence_binding(self.recurrent_plan, args,
            session=session, name=f"{name}.{mode}", public_bind=public_bind, bind_method=bind_method,
            mode=mode, workspace=workspace, run_options=self.options,
            set_binding=lambda binding: setattr(self, "recurrence", binding))
        if mode == "prefill":
            # The canonical sigmoid-gated FP32 RMSNorm, fused, writing the output projection's input.
            self.output_norm = PreparedGatedRMSNorm(scratch.recurrence.view(self.tokens, self.heads, 128),
                scratch.z.view(self.tokens, self.heads, 128), module.o_norm.weight,
                scratch.gated.view(self.tokens, self.heads, 128), metadata.num_tokens,
                eps=module.o_norm.variance_epsilon)

    def __call__(self, source):
        s, p = self.scratch, self.projections
        if self.gate_overlap is not None:
            self.gate_overlap.start(source)
        p["input_proj"](source)
        if self.tokens > 1:
            s.forget_input.copy_(s.projected_input[:, -128:])
        p["f_b_proj"](s.forget_input)
        if self.gate_overlap is None:
            p["g_b_proj"](p["g_a_proj"](source))
        else:
            self.gate_overlap.finish(source, s.raw_g)
        short_conv.run(self.conv_binding)
        if self.verification is not None:
            self.verification()
            self.output_norm()
        elif self.mode == "decode" and self.records is not None:
            m = self.metadata
            _record([s.convolved_qkv, s.raw_g, s.raw_beta, s.z, self.state.recurrent, m.query_start_loc,
                     m.state_indices, m.num_seqs, m.num_tokens], s.gated, *self.records, self.handle)
        elif self.mode == "decode":
            gdn_decode.run_kda(self.recurrence, **self.options)
        else:
            kda_prefill.run(self.recurrence, **self.options)
            self.output_norm()
        return p["o_proj"](s.gated)
