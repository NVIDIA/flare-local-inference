"""Prepared SwiGLU MLPs using b12x projections and caller-owned buffers."""

from dataclasses import dataclass

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.weights import MLPWeights
from ignition.preparation.tuning import declare


@dataclass(frozen=True)
class MLPScratch:
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, hidden, intermediate, device, *, buffers=None):
        if any(type(value) is not int or value <= 0 for value in (tokens, hidden, intermediate)):
            raise ValueError("MLP scratch dimensions must be positive integers")
        return cls(*((torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
                      if buffers is None else buffers.zeros((tokens, width), torch.bfloat16))
                     for width in (2 * intermediate, intermediate, hidden)))

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("MLP rows exceed scratch capacity")
        return MLPScratch(self.gate_up[:tokens], self.intermediate[:tokens], self.output[:tokens])


class PreparedMLP:
    """Dense or shared-expert MLP, including rank-local intermediate partitions.

    Gate/up projection rows are gate first, then up. The output is a partial
    hidden vector for sharded weights; the caller owns its collective reduction.
    All output and intermediate buffers must remain live through graph replay.
    """

    def __init__(self, source, weights: MLPWeights, gate_up, intermediate, output, *,
                 session, name, workspace, swiglu_limit=float("inf"), round_silu=True,
                 activation_mode="auto", gate_up_bias=None, down_bias=None, clamp_after_silu=False,
                 swiglu_alpha=1., swiglu_beta=0.):
        self.gate_up, self.intermediate, self.output = gate_up, intermediate, output
        self.swiglu_limit, self.round_silu = swiglu_limit, round_silu
        if source.ndim != 2 or intermediate.ndim != 2:
            raise ValueError("MLP input and intermediate must be matrices")
        tokens, hidden = source.shape
        width = intermediate.shape[1]
        for tensor, shape in ((source, (tokens, hidden)), (gate_up, (tokens, 2 * width)),
                              (intermediate, (tokens, width)), (output, (tokens, hidden))):
            if (tensor.dtype != torch.bfloat16 or tensor.shape != shape
                    or tensor.device != source.device or not tensor.is_contiguous()):
                raise ValueError("MLP buffers require contiguous BF16 storage with matching geometry and device")

        def projection(input, weight, out, suffix, bias):
            from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight

            if isinstance(weight, BlockFP8Weight):
                if bias is not None:
                    raise ValueError("block-FP8 MLP projections do not support bias")
                return BlockFP8Projection(input, weight, out, session=session, name=f"{name}.{suffix}")
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(input, weight, out, session=session, name=f"{name}.{suffix}", bias=bias)
            return QuantizedProjection(input, weight, out, session=session,
                                       name=f"{name}.{suffix}", workspace=workspace,
                                       activation_mode=activation_mode, bias=bias)

        self.up_projection = projection(source, weights.gate_up, gate_up, "gate_up", gate_up_bias)
        self.down_projection = projection(intermediate, weights.down, output, "down", down_bias)
        self.activation_plan = hyperconnection.plan(
            hyperconnection.Caps(device=source.device, max_tokens=tokens, hidden_size=width),
            invocation={"operation": "swiglu", "limit": swiglu_limit, "round_silu": round_silu,
                        "clamp_after_silu": clamp_after_silu,
                        "swiglu_alpha": swiglu_alpha, "swiglu_beta": swiglu_beta},
        )
        declare(session, (self.activation_plan.request(
            name=f"{name}.swiglu", prepare_call=lambda state: PreparedCall(
                run=lambda: state.launch(gate_up, gate_up, intermediate),
                owners=(gate_up, intermediate),
            ),
        ),))

    def __call__(self, source):
        self.up_projection(source)
        hyperconnection.run_swiglu(self.gate_up, limit=self.swiglu_limit,
                                  round_silu=self.round_silu, out=self.intermediate,
                                  plan=self.activation_plan)
        return self.down_projection(self.intermediate)


class PreparedUngatedMLP:
    """Two projections with a BF16 ReLU-squared or SiLU intermediate."""

    def __init__(self, source, up_weight, down_weight, projected, intermediate, output, *,
                 activation, session, name, workspace, up_bias=None, down_bias=None):
        if activation not in ("relu2", "silu"):
            raise ValueError("ungated MLP activation must be relu2 or silu")
        if (source.ndim != 2 or projected.ndim != 2 or projected.shape != intermediate.shape
                or projected.shape[0] != source.shape[0] or output.shape[0] != source.shape[0]
                or any(t.dtype != torch.bfloat16 or t.device != source.device or not t.is_contiguous()
                       for t in (source, projected, intermediate, output))):
            raise ValueError("ungated MLP buffers require matching contiguous BF16 rows")
        self.activation, self.intermediate = activation, intermediate

        def projection(x, weight, y, suffix, bias):
            kwargs = dict(session=session, name=name + suffix, bias=bias)
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(x, weight, y, **kwargs)
            return QuantizedProjection(x, weight, y, workspace=workspace, activation_mode="auto", **kwargs)

        self.up_projection = projection(source, up_weight, projected, ".up", up_bias)
        self.down_projection = projection(intermediate, down_weight, output, ".down", down_bias)
        if activation == "relu2":
            self.activation_plan = hyperconnection.plan(hyperconnection.Caps(device=source.device,
                max_tokens=source.shape[0], hidden_size=projected.shape[1]), invocation={"operation": "relu2"})
            declare(session, (self.activation_plan.request(name=name + ".relu2", prepare_call=lambda state:
                PreparedCall(run=lambda: state.launch(projected, projected, intermediate))),))

    def __call__(self, source):
        value = self.up_projection(source)
        self.intermediate.copy_(torch.relu(value).square() if self.activation == "relu2"
                                else torch.nn.functional.silu(value))
        return self.down_projection(self.intermediate)
