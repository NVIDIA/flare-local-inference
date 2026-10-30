"""Prepared b12x projections with caller-owned output and scratch storage."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch
from b12x.gemm import bf16_gemv, blockscaled
from b12x.preparation import PreparationSession, PreparedCall

from ignition.components import sm103_projection
from ignition.preparation.fixtures import bf16_gemv as bf16_fixtures
from ignition.preparation.fixtures import blockscaled_gemm as gemm_fixtures
from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.scratch import ScratchArena, prime_owners, scratch_storage
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare, register_context_plan, register_finalizer


def prepare_projection(source, weight, output, *, session, name, workspace, activation_mode="auto", staging=None):
    """Bind a projection without changing its checkpoint weight representation."""
    from .block_fp8 import BlockFP8Projection, BlockFP8Weight
    from .weights import GroupedLinearWeights

    if isinstance(weight, GroupedLinearWeights):
        return PreparedProjectionGroup(source, weight, output, session=session, name=name, workspace=workspace,
                                       activation_mode=activation_mode, staging=staging)
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    if isinstance(weight, BlockFP8Weight):
        return BlockFP8Projection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name,
                               workspace=workspace, activation_mode=activation_mode)


class PreparedProjectionGroup:
    """Project one source into ordered output slices with compatible weights fused."""

    def __init__(self, source, weight, output, *, session, name, workspace, activation_mode, staging):
        from .block_fp8 import BlockFP8Weight

        self.output = output
        self.projections, self.copies = [], []
        for index, (part, target) in enumerate(zip(weight.parts, output.split(weight.widths, dim=-1), strict=True)):
            destination = target
            if not isinstance(part, (torch.Tensor, BlockFP8Weight)) and not target.is_contiguous():
                if (staging is None or not staging.is_contiguous() or staging.numel() < target.numel()
                        or staging.dtype != target.dtype or staging.device != target.device):
                    raise ValueError("mixed packed projections require caller-owned contiguous staging")
                destination = staging.view(-1)[:target.numel()].view(target.shape)
            self.projections.append(prepare_projection(source, part, destination, session=session,
                name=f"{name}.part{index}", workspace=workspace, activation_mode=activation_mode))
            self.copies.append(None if destination is target else target)
        self.plans = tuple(projection.plan for projection in self.projections)

    def __call__(self, source):
        for projection, destination in zip(self.projections, self.copies, strict=True):
            value = projection(source)
            if destination is not None:
                destination.copy_(value)
        return self.output


class BatchedUnquantizedProjection:
    """Multiply BF16 matrix stacks into caller-owned, possibly strided outputs."""

    def __init__(self, source, weight, output, *, major, session, name):
        del session, name
        if major not in ("n", "k"):
            raise ValueError("batched projections take an n-major or k-major RHS")
        self.weight = weight if major == "n" else weight.transpose(1, 2)
        if (source.ndim != 3 or output.shape[:2] != source.shape[:2]
                or self.weight.shape != (source.shape[0], source.shape[2], output.shape[2])
                or any(value.dtype != torch.bfloat16 or value.device != source.device
                       for value in (source, weight, output))):
            raise ValueError("batched BF16 projections require matching matrix stacks and outputs")
        record_numeric_tensor(weight)
        self.output = output

    def __call__(self, source):
        _bmm_out(source, self.weight, self.output)
        return self.output


@torch.library.custom_op("flare::bmm_out", mutates_args=("out",))
def _bmm_out(source: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(source, weight, out=out)


@_bmm_out.register_fake
def _fake_bmm_out(source, weight, out):
    pass


class UnquantizedProjection:
    def __init__(
        self,
        source: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        *,
        session: PreparationSession,
        name: str,
        bias: torch.Tensor | None = None,
    ) -> None:
        if any(t.dtype not in (torch.bfloat16, torch.float32) for t in (source, weight, output)):
            raise TypeError("unquantized projections require BF16 or FP32 operands")
        for tensor in (weight, bias):
            if tensor is not None:
                record_numeric_tensor(tensor)
        # FlashInfer's GEMMs read and write contiguous rows; column slices of wider rows keep b12x's GEMV.
        rows = source.numel() // source.shape[-1]
        self.dense = (sm103_projection.is_sm103(source.device) and bias is None
                      and source.dtype == weight.dtype == torch.bfloat16
                      and source.is_contiguous() and output.is_contiguous()
                      and sm103_projection.dense_serves(rows, output.shape[-1]))
        if self.dense:
            # b12x's GEMV reaches about 1.7 TB/s on GB300; FlashInfer's tuned GEMMs approach its bandwidth.
            self.weight, self.bias, self.output = weight, None, output
            self.plan = sm103_projection.dense_plan(sm103_projection.DenseProjectionQuery(
                source.numel() // source.shape[-1], source.shape[-1], output.shape[-1],
                str(output.dtype).removeprefix("torch.")), source.device)
            declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(source, weight, out=output)),
                benchmark_call=partial(bf16_fixtures.benchmark_call,
                    inputs=input_factory(source, output, storage=getattr(session, "benchmark_storage", None)), weight=weight,
                                       bias=None)),))
            return
        query = bf16_gemv.query_from_call(source, weight, out=output, bias=bias)
        self.plan = bf16_gemv.plan(query)
        self.weight = weight
        self.bias = bias
        self.output = output
        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(source, weight, out=output, bias=bias)),
            benchmark_call=partial(bf16_fixtures.benchmark_call,
                inputs=input_factory(source, output, storage=getattr(session, "benchmark_storage", None)),
                                   weight=weight, bias=bias),
        ),))

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if self.dense:
            torch.ops.flare.sm103_dense_out(source, self.weight, self.output, self.plan.handle)
            return self.output
        return bf16_gemv.mm(source, self.weight, out=self.output, plan=self.plan, bias=self.bias)


class QuantizedProjection:
    def __init__(
        self,
        source: torch.Tensor,
        weight: blockscaled.Weight,
        output: torch.Tensor,
        *,
        session: PreparationSession,
        name: str,
        activation_mode: str = "quantized",
        source_activation: str = "identity",
        bias: torch.Tensor | None = None,
        workspace: ScratchArena | torch.Tensor | None = None,
        mxfp8_min_amax: float | None = None,
        encoded=None,
        rhs_prefetch_ready: bool = False,
        allow_bf16_atomic_reduction: bool = True,
        buffers=None,
    ) -> None:
        from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight, _overlap, _weight_parts
        from b12x.gemm.blockscaled._linear import MXFP8LinearWeight, _validate_bias

        # Address-dependent bias checks run before metadata-only forward compilation.
        _validate_bias(bias, out_features=weight.out_features, out_dtype=output.dtype, device=source.device)
        if bias is not None and _overlap(output, bias):
            raise ValueError("projection output must not overlap bias")

        self.weight = weight
        self.output = output
        self.activation_scale = getattr(weight, "activation_global_scale", None)
        if isinstance(weight, blockscaled.PackedNVFP4LinearWeight):
            if activation_mode == "quantized":
                raise ValueError("packed dense W4A16 requires BF16 activations")
            activation_mode = "a16"
        native = activation_mode == "auto" and getattr(session, "native_dense_activations", False)
        if native:
            activation_mode = "quantized"
            if isinstance(weight, NVFP4LinearWeight) and self.activation_scale is None:
                raise ValueError(f"{name}: native NVFP4 dense activations require an activation global scale")
        elif isinstance(weight, NVFP4LinearWeight) and self.activation_scale is None:
            activation_mode = "a16"
        self.activation_mode = activation_mode
        self.bias = bias
        self.workspace = workspace
        self.encoded = self.quantizer_plan = None
        self.sm103 = sm103_projection.is_sm103(source.device)
        if source_activation != "identity" and (self.sm103 or not isinstance(weight, NVFP4LinearWeight)):
            raise ValueError("rounded SwiGLU projection requires b12x NVFP4 execution")
        if mxfp8_min_amax is not None or encoded is not None:
            from b12x.quantization import mxfp8

            if (self.sm103 or not isinstance(weight, MXFP8LinearWeight)
                    or activation_mode not in ("auto", "quantized") or bias is not None
                    or workspace is None or source.ndim != 2 or source.dtype != torch.bfloat16
                    or not source.is_contiguous() or weight.padded_in_features != source.shape[1]
                    or output.shape != (source.shape[0], weight.out_features)
                    or output.dtype != torch.bfloat16 or not output.is_contiguous()
                    or (encoded is not None and (mxfp8_min_amax is not None
                                                 or not isinstance(encoded, mxfp8.MXFP8Rows)))):
                raise ValueError("prepared MXFP8 activation projection requires BF16 rows, packed MXFP8 weights,"
                                 " BF16 output, and caller-owned scratch")
            self.activation_mode = "quantized"
            rows, columns = source.shape
            def allocate(shape, dtype):
                return (torch.zeros(shape, dtype=dtype, device=source.device) if buffers is None
                        else buffers.zeros(shape, dtype))

            values_shape, row_shape, mma_shape = mxfp8.dense_rows_storage_shapes(rows, columns)
            if encoded is None:
                self.encoded_bases = (allocate(values_shape, torch.float8_e4m3fn),
                                      allocate(row_shape, torch.uint8), allocate(mma_shape, torch.uint8))
                self.encoded = mxfp8.dense_rows_from_bases(*self.encoded_bases)
                self.quantizer_plan = mxfp8.plan(mxfp8.query_from_call(
                    source, self.encoded.values, self.encoded.scale_rows, self.encoded.scale_mma,
                    min_amax=mxfp8_min_amax, release_dependents=rhs_prefetch_ready and rows <= 8))
            else:
                self.encoded_bases = (encoded.values, encoded.scale_rows.view(torch.uint8),
                                      mxfp8.dense_scale_storage(encoded.scale_mma))
                self.encoded = mxfp8.dense_rows_from_bases(*self.encoded_bases)
                if self.encoded.values.shape != source.shape or self.encoded.values.device != source.device:
                    raise ValueError("prepared MXFP8 activation rows must match their BF16 producer")
            encoded = self.encoded
            packed = weight.weight
            self.parts = packed.values, mxfp8.dense_scale_storage(packed.scale_mma), None
            for tensor in (packed.values, packed.scale_rows, packed.scale_mma):
                record_numeric_tensor(tensor)
            self.raw_source = encoded.values.view(rows, columns, 1), encoded.scale_mma
            self.raw_weight = packed.values.view(weight.out_features, columns, 1), packed.scale_mma
            self.raw_output = output.view(rows, weight.out_features, 1)
            query = blockscaled.query_from_call(self.raw_source, self.raw_weight, out=self.raw_output,
                ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", sf_vec_size=32,
                rhs_prefetch_ready=rhs_prefetch_ready,
                allow_bf16_atomic_reduction=allow_bf16_atomic_reduction)
            if isinstance(workspace, ScratchArena):
                query = replace(query, workspace_form="provided", workspace_nbytes=None)
            self.plan = blockscaled.plan(query)

            def selected_workspace(nbytes=None):
                if nbytes is None:
                    nbytes = self.plan.memory_requirements().scratch_nbytes
                return scratch_storage(workspace, nbytes=nbytes, device=source.device)

            def candidate_workspace():
                # RecordedPreparationSession reserves the complete candidate
                # envelope before tuning; the selected view is bound afterwards.
                return workspace.tensor if isinstance(workspace, ScratchArena) else workspace

            def prepare_quantizer(state):
                return PreparedCall(run=lambda: state.run(source, encoded.values, encoded.scale_rows,
                                                          encoded.scale_mma), owners=(encoded,))

            def prepare_projection(state):
                storage = candidate_workspace()
                return PreparedCall(run=lambda: state.run(*self.raw_source, *self.raw_weight, None,
                    self.raw_output, storage.view(torch.float32), torch.bfloat16, None),
                    owners=prime_owners(workspace, storage, encoded))

            def benchmark_projection(state):
                def make_inputs():
                    bases = (torch.empty(values_shape, dtype=torch.float8_e4m3fn, device=source.device),
                             torch.empty(row_shape, dtype=torch.uint8, device=source.device),
                             torch.empty(mma_shape, dtype=torch.uint8, device=source.device))
                    trial = mxfp8.dense_rows_from_bases(*bases)
                    seeds = (torch.ones_like(trial.values),
                             torch.full_like(trial.scale_rows.view(torch.uint8), 127),
                             torch.full_like(trial.scale_mma.view(torch.uint8), 127))
                    return trial, torch.empty_like(output), seeds

                trial, trial_output, seeds = benchmark_resource(session, ("mxfp8_projection", name), make_inputs)
                lhs = trial.values.view(rows, columns, 1), trial.scale_mma
                out = trial_output.view(rows, weight.out_features, 1)
                storage = candidate_workspace()
                def produce():
                    trial.values.copy_(seeds[0])
                    trial.scale_rows.view(torch.uint8).copy_(seeds[1])
                    trial.scale_mma.view(torch.uint8).copy_(seeds[2])

                return PreparedCall(run=lambda: state.run(*lhs, *self.raw_weight, None,
                    out, storage.view(torch.float32), torch.bfloat16, None),
                    produce=produce, output=trial_output,
                    owners=(trial, trial_output, seeds, *prime_owners(workspace, storage)))

            requests = []
            if self.quantizer_plan is not None:
                requests.append(self.quantizer_plan.request(name=name + ".activation",
                    prepare_call=prepare_quantizer))
            requests.append(self.plan.request(name=name,
                prepare_call=prepare_projection, benchmark_call=benchmark_projection))
            declare(session, tuple(requests))
            register_context_plan(session, name, self)
            register_finalizer(session, lambda: setattr(self, "workspace", selected_workspace()), workspace=workspace)
            return
        if self.sm103:
            # b12x targets SM120/SM121; SM103 runs FlashInfer's native GEMMs under the same preparation.
            from b12x.gemm.blockscaled._a16 import _weight_parts as parts

            query = sm103_projection.query_for(
                source, weight, output, workspace_form="provided")
            if native and query.mode != "quantized":
                raise ValueError(f"{name}: native dense activations are unsupported for this SM103 projection")
            self.plan = sm103_projection.plan(query,
                weight=parts(weight)[:2])
        else:
            query = blockscaled.query_from_call(source, weight, out=output, activation_mode=activation_mode,
                                               source_activation=source_activation,
                                               workspace=None if isinstance(workspace, ScratchArena) else workspace)
            if workspace is None or isinstance(workspace, ScratchArena):
                query = replace(query, workspace_form="provided", workspace_nbytes=None)
            self.plan = blockscaled.plan(query)
        values, scales, gain, _ = _weight_parts(weight)
        if self.sm103 and scales.dtype == torch.float8_e8m0fnu:
            # Operators receive UE8M0 scales as bytes, as b12x's own packed operators do.
            scales = scales.view(torch.uint8)
        self.parts = values, scales, gain
        # Packing can replace the checkpoint storage, including integer payloads
        # and byte-encoded scales. Their bits must not become export dependencies.
        planes = (*self.parts, self.activation_scale)
        if isinstance(weight, MXFP8LinearWeight):
            planes += (weight.weight.scale_rows, weight.weight.values_tiled)
        for tensor in planes:
            if tensor is not None:
                record_numeric_tensor(tensor)

        def selected_workspace(nbytes=None):
            if nbytes is None:
                nbytes = self.plan.memory_requirements().scratch_nbytes
            return scratch_storage(workspace, nbytes=nbytes, device=source.device)

        def prepare(state):
            storage = selected_workspace(state.required_workspace)
            return PreparedCall(run=lambda: state.run(source, values, scales, gain,
                                                       out=output, workspace=storage, activation_scale=self.activation_scale),
                                owners=prime_owners(workspace, storage) if storage is not None else ())

        declare(session, (self.plan.request(
            name=name,
            prepare_call=prepare,
            benchmark_call=partial(gemm_fixtures.benchmark_call,
                inputs=input_factory(source, output, storage=getattr(session, "benchmark_storage", None)),
                values=values, scales=scales, gain=gain, bias=bias, weight=weight,
                activation_scale=self.activation_scale,
                selected_workspace=selected_workspace),
        ),))
        register_finalizer(session, lambda: setattr(self, "workspace", selected_workspace()), workspace=workspace)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if self.encoded is not None:
            from b12x import gemm
            from b12x.quantization import mxfp8

            if self.quantizer_plan is not None:
                mxfp8.quantize_dense_rows(source, *self.encoded_bases, plan=self.quantizer_plan)
            gemm.mxfp8_mm_bases(self.encoded_bases[0], self.encoded_bases[2],
                self.parts[0], self.parts[1], self.output, self.workspace, plan=self.plan)
            return self.output
        if self.sm103:
            values, scales, gain = self.parts
            torch.ops.flare.sm103_packed_out(source, values, scales, gain, self.output, self.workspace,
                                             self.plan.handle)
        else:
            blockscaled.mm(source, self.weight, out=self.output, plan=self.plan, workspace=self.workspace)
        if self.bias is not None:
            self.output.add_(self.bias)
        return self.output


_EXPANDED_BMM_WEIGHTS = {}


def _expanded_bmm_weight(weight):
    """BF16 copy of one packed E4M3 matrix stack with per-32 E8M0 scales along its last axis.

    E4M3 values times power-of-two scales are exact in BF16. Layers that slice the
    same weights share one copy.
    """
    values, scales = weight
    key = (values.data_ptr(), tuple(values.shape), tuple(values.stride()))
    expanded = _EXPANDED_BMM_WEIGHTS.get(key)
    if expanded is None:
        exponents = scales.view(torch.uint8).to(torch.float32) - 127
        factors = torch.exp2(exponents).repeat_interleave(32, dim=-1)[..., :values.shape[-1]]
        expanded = (values.to(torch.float32) * factors).to(torch.bfloat16)
        _EXPANDED_BMM_WEIGHTS[key] = expanded
    return expanded


# Rows b12x's BF16-by-MXFP8 BMM covers in one launch.
_BMM_ROWS = 4096


class BatchedQuantizedProjection:
    """Prepared BF16-by-rowwise-MXFP8 BMM over caller-owned strided tensors.

    The RHS remains packed E4M3 values and per-32 E8M0 scales. The native b12x
    kernel has a fixed configuration and requires no scratch. The major axis
    describes the RHS layout and the axis compressed by its scales. On SM103,
    where b12x's kernel does not run, the RHS expands once to BF16 and cuBLAS
    multiplies it.
    """

    def __init__(self, source, weight, output, *, major, session, name):
        from b12x import gemm
        from b12x.gemm._bmm._tuning import BmmQuery

        if source.ndim != 3 or output.ndim != 3 or source.shape[:2] != output.shape[:2]:
            raise ValueError("batched projections require matching batch and row dimensions")
        self.weight, self.output, self.major = weight, output, major
        for tensor in weight:
            record_numeric_tensor(tensor)
        self.expanded = None
        if sm103_projection.is_sm103(source.device):
            if major not in ("n", "k"):
                raise ValueError("batched projections take an n-major or k-major RHS")
            expanded = _expanded_bmm_weight(weight)
            # n-major stacks are [in, out]; k-major stacks are [out, in].
            self.expanded = expanded if major == "n" else expanded.transpose(1, 2)
            if self.expanded.shape != (source.shape[0], source.shape[2], output.shape[2]):
                raise ValueError("batched projection weights must match the source and output widths")
            return
        # b12x's kernel covers at most _BMM_ROWS rows; longer prefill chunks run in row slices.
        rows = source.shape[1]
        self.slices = tuple((start, min(rows, start + _BMM_ROWS)) for start in range(0, rows, _BMM_ROWS))
        self.plans = []
        for start, end in self.slices:
            plan = gemm.plan_bmm(BmmQuery(batch=source.shape[0], max_rows=end - start,
                in_features=source.shape[2], out_features=output.shape[2], b_major=major, sf_axis=major))
            rows_in, rows_out = source[:, start:end], output[:, start:end]
            declare(session, (plan.request(name=name if len(self.slices) == 1 else f"{name}.rows{start}",
                prepare_call=lambda state, rows_in=rows_in, rows_out=rows_out: PreparedCall(
                    run=lambda: state.run(rows_in, weight, rows_out, b_major=major, sf_axis=major))),))
            self.plans.append(plan)

    def __call__(self, source):
        from b12x import gemm

        if self.expanded is not None:
            torch.ops.flare.sm103_bmm_out(source, self.expanded, self.output)
            return self.output
        for (start, end), plan in zip(self.slices, self.plans, strict=True):
            gemm.bmm(source[:, start:end], self.weight, self.output[:, start:end], plan=plan,
                a_dtype="bfloat16", b_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
                c_dtype="bfloat16", sf_vec_size=32, b_major=self.major, sf_axis=self.major)
        return self.output


# cuBLAS writes strided-batched results straight into the caller's per-head views; traced copies would
# instead stage a contiguous result (300 us per 4096-row GLM absorption on GB300).
@torch.library.custom_op("flare::sm103_bmm_out", mutates_args=("out",))
def _sm103_bmm_out(source: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    torch.bmm(source, weight, out=out)


@_sm103_bmm_out.register_fake
def _fake_sm103_bmm_out(source, weight, out):
    pass


@torch.library.custom_op("flare::parallel_mx_projections", mutates_args=("outputs", "workspaces"))
def _parallel_mx_projections(
    source: torch.Tensor, values: list[torch.Tensor], scales: list[torch.Tensor],
    outputs: list[torch.Tensor], workspaces: list[torch.Tensor],
    plans: list[int], auxiliary_stream: int, join: bool = True,
) -> None:
    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)

    def project(index):
        sm103_projection.packed_out_workspace(
            source, values[index], scales[index], None, None, outputs[index],
            workspaces[index], "none", None, plans[index], "any")

    with torch.cuda.stream(auxiliary):
        project(1)
    project(0)
    if join:
        main.wait_stream(auxiliary)


@_parallel_mx_projections.register_fake
def _fake_parallel_mx_projections(source, values, scales, outputs, workspaces, plans, auxiliary_stream, join=True):
    pass


class ParallelProjections:
    """Two prepared MX projections with explicit storage and a captured join."""

    def __init__(self, main: QuantizedProjection, auxiliary: QuantizedProjection, *, session,
                 workspace: ScratchArena):
        from b12x.gemm.blockscaled._a16 import _weight_parts

        self.projections = (main, auxiliary)
        self.arena = workspace
        self.values, self.scales, self.outputs, self.plans = [], [], [], []
        for projection in self.projections:
            values, scales, gain, fp4 = _weight_parts(projection.weight)
            if (values.dtype not in (torch.float8_e4m3fn, torch.uint8) or gain is not None or fp4
                    or projection.bias is not None):
                raise ValueError("parallel projections require unbiased MXFP8 or MXFP4 weights")
            self.values.append(values)
            self.scales.append(scales.view(torch.uint8))
            self.outputs.append(projection.output)
            self.plans.append(projection.plan.handle)
        workspace.register_parallel(*(projection.plan for projection in self.projections))
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        self.workspaces = list(self.arena.views(*(
            ((projection.plan.memory_requirements().scratch_nbytes,), torch.uint8)
            for projection in self.projections)))

    def __call__(self, source):
        _parallel_mx_projections(source, self.values, self.scales, self.outputs,
                                   self.workspaces, self.plans, self.stream_handle)
