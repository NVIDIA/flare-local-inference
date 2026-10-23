"""Prepared manifold hyperconnections with shared scratch and caller-owned outputs."""

from dataclasses import dataclass

import torch
from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import PreparedCall

from ignition.components import mhc_sm103, sm103
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer
from ignition.preparation.fixtures.resources import candidate_inputs


@dataclass(frozen=True)
class MHCWeights:
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    norm: torch.Tensor


@dataclass(frozen=True)
class MHCScratch:
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    normalized: torch.Tensor

    @classmethod
    def allocate(cls, tokens, hidden, device, *, buffers=None):
        if type(tokens) is not int or tokens <= 0 or hidden not in (4096, 5120, 7168):
            raise ValueError("mHC requires positive token capacity and a supported hidden size")
        def empty(shape, dtype):
            return (torch.empty(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        return cls(empty((tokens, 4, hidden), torch.bfloat16), empty((tokens, 4), torch.float32),
                   empty((tokens, 4, 4), torch.float32), empty((tokens, hidden), torch.bfloat16))

    def bind_args(self):
        return {"out": self.residual, "post": self.post, "comb": self.comb, "y": self.normalized}

    def rows(self, tokens, *, start=0):
        if (type(tokens) is not int or type(start) is not int or start < 0
                or not 0 < tokens <= self.residual.shape[0] - start):
            raise ValueError("mHC rows exceed scratch capacity")
        return MHCScratch(*(getattr(self, name)[start:start + tokens]
                            for name in self.__dataclass_fields__))


class _BenchmarkInputs:
    def __init__(self, residual, operation, lagged):
        tokens, _, hidden = residual.shape
        generator = torch.Generator(device=residual.device).manual_seed(3187)
        values = torch.randn(residual.shape, dtype=residual.dtype, device=residual.device,
                             generator=generator) * .1
        self.inputs = (values,)
        if operation == "post_pre":
            source = torch.randn((tokens, hidden), dtype=residual.dtype, device=residual.device,
                                 generator=generator) * .1
            post = torch.ones((tokens, 4), dtype=torch.float32, device=residual.device)
            comb = torch.eye(4, device=residual.device).expand(tokens, -1, -1).contiguous()
            self.inputs = (source, values, post, comb)
        self.seeds = self.inputs
        self.inputs = tuple(torch.empty_like(value) for value in self.seeds)
        self.scratch = MHCScratch.allocate(tokens, hidden, residual.device)
        self.pre_mix = self.pre_out = None
        if lagged:
            self.pre_mix = torch.zeros((tokens, 4), dtype=torch.float32, device=residual.device)
            self.pre_mix[:, 0] = 1
            self.pre_out = torch.empty_like(self.pre_mix)
        self.nbytes = sum(t.nbytes for t in (*self.seeds, *self.inputs, *self.scratch.bind_args().values()))
        if lagged:
            self.nbytes += self.pre_mix.nbytes + self.pre_out.nbytes

    def produce(self):
        for target, seed in zip(self.inputs, self.seeds, strict=True):
            target.copy_(seed)


class PreparedMHC:
    """mHC pre or fused post/pre, including the following weighted RMSNorm.

    Pre consumes a BF16 residual of shape [tokens, 4, hidden]. Post/pre also
    consumes the preceding branch output and its FP32 expansion/mixing weights.
    The output residual and normalized branch input retain BF16 storage; mixing
    coefficients and projection weights use FP32. Runtime calls use fixed shapes.
    ``max_rows_per_call`` bounds the row working set of each sequential call;
    b12x autotunes those shapes, and calls reuse the same temporary workspace.
    """

    def __init__(self, inputs, weights: MHCWeights, scratch: MHCScratch, *,
                 session, name, workspace, rms_eps, hc_eps, sinkhorn_iters, norm_eps,
                 pre_mix=None, pre_out=None, max_rows_per_call=None):
        if max_rows_per_call is not None and (type(max_rows_per_call) is not int
                                              or max_rows_per_call <= 0):
            raise ValueError("mHC row capacity per call must be a positive integer")
        self.operation = "pre" if len(inputs) == 1 else "post_pre" if len(inputs) == 4 else None
        if self.operation is None:
            raise ValueError("mHC requires residual or branch/residual/post/comb inputs")
        residual = inputs[0 if self.operation == "pre" else 1]
        if residual.ndim != 3 or residual.shape[1] != 4 or residual.dtype != torch.bfloat16:
            raise ValueError("mHC residual requires BF16 [tokens, 4, hidden]")
        tokens, _, hidden = residual.shape
        if (weights.fn.shape != (24, 4 * hidden) or weights.fn.dtype != torch.float32
                or weights.scale.shape != (3,) or weights.scale.dtype != torch.float32
                or weights.base.shape != (24,) or weights.base.dtype != torch.float32
                or weights.norm.shape != (hidden,) or weights.norm.dtype not in (torch.bfloat16, torch.float32)):
            raise ValueError("mHC weights require FP32 projection/mixes and a BF16 or FP32 RMSNorm vector")
        self.weights, self.scratch = weights, scratch
        self.options = {"rms_eps": rms_eps, "hc_eps": hc_eps, "sinkhorn_iters": sinkhorn_iters,
                            "norm_weight": weights.norm, "norm_eps": norm_eps}
        if (pre_mix is None) != (pre_out is None):
            raise ValueError("lagged mHC requires incoming and outgoing coefficient buffers")
        if pre_mix is not None:
            self.options.update(pre_mix=pre_mix)
        self.composed = self.row_calls = None
        if pre_mix is None and tokens > mhc_sm103.MIN_ROWS and sm103.is_sm103(residual.device):
            # SM103 prefill composes FlashInfer's mHC kernels; b12x's fused launches stay for decode.
            self.composed = mhc_sm103.ComposedMHC(self.operation, weights, scratch, rms_eps=rms_eps,
                hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters, norm_eps=norm_eps)
            return
        if max_rows_per_call is not None and tokens > max_rows_per_call:
            self.row_calls = []
            for start in range(0, tokens, max_rows_per_call):
                end = min(start + max_rows_per_call, tokens)
                call = PreparedMHC(tuple(value[start:end] for value in inputs), weights,
                    scratch.rows(end - start, start=start), session=session,
                    name=f"{name}.rows{start}_{end}", workspace=workspace,
                    rms_eps=rms_eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters,
                    norm_eps=norm_eps,
                    pre_mix=None if pre_mix is None else pre_mix[start:end],
                    pre_out=None if pre_out is None else pre_out[start:end])
                self.row_calls.append((start, end, call))
            return
        self.plan = mhc.plan(mhc.Caps(device=residual.device, max_tokens=tokens, hidden_size=hidden),
            invocation={"operation": self.operation, "output_mode": "provided", "has_norm_weight": True,
                "norm_weight_dtype": str(weights.norm.dtype).removeprefix("torch."),
                "expanded_residual": self.operation == "pre",
                "rms_eps": rms_eps, "hc_eps": hc_eps, "sinkhorn_iters": sinkhorn_iters, "norm_eps": norm_eps,
                "lagged_mix": pre_mix is not None})
        entry = getattr(_impl, f"_b12x_mhc_{self.operation}_impl")
        parameters = (weights.fn, weights.scale, weights.base)

        def call(state, values, buffers, fixture=None):
            storage = planned_scratch(state.scratch_specs(), workspace)
            target_pre = pre_out if fixture is None else fixture.pre_out
            binding = state.bind(scratch=storage, tokens=tokens, pre_out=target_pre, **buffers.bind_args())
            options = dict(self.options)
            if fixture is not None and pre_mix is not None:
                options.update(pre_mix=fixture.pre_mix)
            return PreparedCall(
                run=lambda: entry(*values, *parameters, **options, binding=binding, _state=state),
                produce=None if fixture is None else fixture.produce,
                output=tuple(buffers.bind_args().values()),
                owners=(*prime_owners(workspace, storage, binding), fixture))

        benchmark_inputs = candidate_inputs(lambda: _BenchmarkInputs(residual, self.operation, pre_mix is not None))

        def benchmark(state):
            fixture = benchmark_inputs()
            return call(state, fixture.inputs, fixture.scratch, fixture)

        declare(session, (self.plan.request(name=name, prepare_call=lambda state: call(state, inputs, scratch),
                                          benchmark_call=benchmark),))

        def finalize():
            self.binding = mhc.bind(self.plan, scratch=planned_scratch(self.plan.scratch_specs(), workspace),
                                    tokens=tokens, pre_out=pre_out, **scratch.bind_args())

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self, *inputs):
        if self.composed is not None:
            return self.composed(*inputs)
        if self.row_calls is not None:
            for start, end, call in self.row_calls:
                call(*(value[start:end] for value in inputs))
            return self.scratch.normalized
        entry = mhc.run_pre if self.operation == "pre" else mhc.run_post_pre
        entry(*inputs, self.weights.fn, self.weights.scale, self.weights.base,
              **self.options, binding=self.binding)
        return self.scratch.normalized


class PreparedMHCPost:
    def __init__(self, source, residual, post, comb, output, *, session, name):
        self.output = output
        self.plan = mhc.plan(mhc.Caps(device=residual.device, max_tokens=residual.shape[0],
                                     hidden_size=residual.shape[-1]),
                             invocation={"operation": "post", "output_mode": "provided"})
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: _impl._b12x_mhc_post_impl(source, residual, post, comb, out=output, _state=state))),))

    def __call__(self, source, residual, post, comb):
        return mhc.run_post(source, residual, post, comb, out=self.output, plan=self.plan)


class PreparedMHCHead:
    """Learn a per-token reduction from hyperconnection streams to head features."""

    def __init__(self, residual, weights, output, *, rms_eps, hc_eps, session, name):
        from ignition.components.linear import UnquantizedProjection

        self.weights, self.output = weights, output
        self.rms_eps, self.hc_eps = rms_eps, hc_eps
        self.mixes = torch.zeros(residual.shape[:2], dtype=torch.float32, device=residual.device)
        self.projection = UnquantizedProjection(residual.flatten(1), weights["fn"], self.mixes,
            session=session, name=name + ".projection")

    def __call__(self, residual):
        self.projection(residual.flatten(1))
        inverse = torch.rsqrt(residual.float().flatten(1).square().mean(-1,keepdim=True) + self.rms_eps)
        mixing = (self.mixes * inverse * self.weights["scale"] + self.weights["base"]).sigmoid() + self.hc_eps
        self.output.copy_((residual.float() * mixing[:,:,None]).sum(1))
        return self.output
