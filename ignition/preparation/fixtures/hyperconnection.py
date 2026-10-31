# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private operands for hyperconnection candidate measurements."""

import torch
from b12x.preparation import PreparedCall


class HyperconnectionBenchmarkInputs:
    """Private finite activations with the canonical RMS normalization gain."""

    def __init__(self, operation, arguments):
        weight_index = {"grouped_rmsnorm": 1, "combine_norm": 3}.get(operation)
        output_count = 2 if operation == "combine_norm" else 1
        self.arguments = tuple(
            value if index == weight_index else torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device)
            for index, value in enumerate(arguments)
        )
        generator = torch.Generator(device="cpu").manual_seed(65537)
        self.inputs = []
        for index, target in enumerate(self.arguments[:-output_count]):
            if index == weight_index:
                continue
            fixture = torch.randn(target.shape, generator=generator)
            self.inputs.append((target, fixture.to(device=target.device, dtype=target.dtype)))
        for output in self.arguments[-output_count:]:
            output.zero_()
        self.output = self.arguments[-output_count:] if output_count == 2 else self.arguments[-1]

    def produce(self):
        for target, fixture in self.inputs:
            target.copy_(fixture)


def benchmark_call(state, operation, arguments, kwargs):
    operands = HyperconnectionBenchmarkInputs(operation, arguments)
    return PreparedCall(
        run=lambda: state.launch(*operands.arguments, **kwargs),
        output=operands.output,
        produce=operands.produce,
        owners=(operands,),
    )


class SwigluBenchmarkInputs:
    """Private finite gate/up activations and output with the caller's strides."""

    def __init__(self, gate_up, output):
        self.gate_up = torch.empty_strided(
            gate_up.shape, gate_up.stride(), dtype=gate_up.dtype, device=gate_up.device)
        self.output = torch.empty_strided(
            output.shape, output.stride(), dtype=output.dtype, device=output.device)
        self.generator = torch.Generator(device=gate_up.device)
        self.produce()
        self.output.zero_()

    def produce(self):
        """Refill the activations with the same finite values."""
        self.generator.manual_seed(65537)
        self.gate_up.normal_(generator=self.generator)


def swiglu_benchmark_call(state, gate_up, output, *, session, name):
    """Race call over private operands; the SwiGLU launch reads one gate_up tensor."""
    from ignition.preparation.tuning import benchmark_resource

    operands = benchmark_resource(
        session, ("swiglu_benchmark", tuple(gate_up.shape), tuple(gate_up.stride()), name),
        lambda: SwigluBenchmarkInputs(gate_up, output))
    return PreparedCall(
        run=lambda: state.launch(operands.gate_up, operands.gate_up, operands.output),
        output=operands.output,
        produce=operands.produce,
        owners=(operands,),
    )
