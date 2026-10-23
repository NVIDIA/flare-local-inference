"""Final numeric weight storage is an export leaf across preparation contracts."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation.tensor_metadata import NumericStorage, subtract_ranges, tensor_ranges


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device")


def assert_numeric(numeric, planes, *, opaque=()):
    spans = numeric.ranges()
    for plane in planes:
        if plane is not None:
            assert not subtract_ranges(tensor_ranges(plane), spans)
    for plane in opaque:
        assert subtract_ranges(tensor_ranges(plane), spans) == tensor_ranges(plane)


def session(device):
    from ignition.preparation.tuning import RecordedPreparationSession

    return RecordedPreparationSession(device=device, checkpoint="synthetic",
        workload_schema="numeric-weight-storage", numerical_policy="exact")


@cuda
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "mxfp8", "mxfp8-online",
                                  "tensor_fp8", "block_fp8", "bf16", "batched", "grouped"])
def test_projection_weight_storage_is_numeric(fmt):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm import blockscaled, wo_projection
    from ignition.components.attention_output import PreparedGroupedOutputProjection
    from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight
    from ignition.components.linear import BatchedQuantizedProjection, QuantizedProjection, UnquantizedProjection
    from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
    from ignition.components.weights import quantize_bf16_linear_as_mxfp8

    device = torch.device("cuda:0")
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    source = zeros((1, 256), torch.bfloat16)
    options = dict(session=session(device), name="projection")
    with NumericStorage() as numeric:
        if fmt == "bf16":
            weight, bias = zeros((256, 256), torch.bfloat16), zeros(256, torch.bfloat16)
            UnquantizedProjection(source, weight, torch.empty_like(source), bias=bias, **options)
            planes = (weight, bias)
        elif fmt == "batched":
            source = source.unsqueeze(0).expand(2, -1, -1).contiguous()
            planes = (zeros((2, 256, 256), torch.float8_e4m3fn), zeros((2, 256, 8), torch.uint8))
            BatchedQuantizedProjection(source, planes, torch.empty_like(source), major="k", **options)
        elif fmt == "grouped":
            values = zeros((256, 256), torch.float8_e4m3fn)
            scales = zeros((8, 8), torch.uint8).fill_(127).view(torch.float8_e8m0fnu)
            weight = wo_projection.pack_weights(values, scales, values, scales,
                groups=2, group_width=256, rank=128, hidden=256, block_size=(32, 32))
            source = source[:, None].expand(-1, 2, -1).contiguous()
            operation = PreparedGroupedOutputProjection(source, weight, zeros((1, 256), torch.bfloat16),
                                                        workspace=None, **options)
            planes = operation.operands
        elif fmt == "block_fp8":
            weight = BlockFP8Weight(zeros((256, 256), torch.float8_e4m3fn), zeros((2, 2), torch.float32))
            BlockFP8Projection(source, weight, torch.empty_like(source), **options)
            planes = (weight.values, weight.scales)
        elif fmt == "tensor_fp8":
            packed = blockscaled.pack_weight(zeros((256, 256), torch.float8_e4m3fn),
                                             torch.ones(1, device=device), recipe=fmt)
            weight = TensorFP8Weight(packed, torch.ones(1, device=device))
            TensorFP8Projection(source, weight, torch.empty_like(source), **options)
            planes = (packed.values, packed.scale_mma, packed.block_scale, packed.output_scale, weight.input_scale)
        else:
            if fmt == "mxfp8-online":
                weight = quantize_bf16_linear_as_mxfp8(zeros((256, 256), torch.bfloat16))
            elif fmt == "mxfp8":
                weight = blockscaled.pack_weight(zeros((256, 256), torch.float8_e4m3fn),
                    zeros((256, 8), torch.uint8).fill_(127), recipe=fmt)
            else:
                nv = fmt == "nvfp4"
                scales = (torch.ones((256, 16), device=device).to(torch.float8_e4m3fn) if nv
                          else zeros((256, 8), torch.uint8).fill_(127))
                gains = dict(global_scale=torch.ones(1, device=device), global_scale_kind="multiplier",
                             activation_global_scale=torch.ones(1, device=device)) if nv else {}
                weight = blockscaled.pack_weight(zeros((256, 128), torch.uint8),
                    swizzle_block_scale(scales), recipe=fmt, **gains)
            QuantizedProjection(source, weight, torch.empty_like(source), activation_mode="auto", **options)
            if fmt.startswith("mxfp8"):
                planes = (weight.weight.values, weight.weight.scale_rows, weight.weight.scale_mma,
                          weight.weight.values_tiled)
            else:
                planes = (weight.values, weight.scale_mma, getattr(weight, "global_scale", None),
                          getattr(weight, "activation_global_scale", None))
    assert_numeric(numeric, planes)


@cuda
@pytest.mark.parametrize("fmt,mode", [("modelopt_nvfp4", "a4"), ("modelopt_nvfp4", "a16"),
                                       ("fp4_e8m0_k32", "a8"), ("fp4_e8m0_k32", "a16")])
@pytest.mark.parametrize("layout", ["w13", "w31"])
def test_expert_weight_storage_is_numeric(fmt, mode, layout):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.moe import fused_moe
    from ignition.components.moe import PreparedRoutedExperts

    device = torch.device("cuda:0")
    nv = fmt == "modelopt_nvfp4"
    group = 16 if nv else 32
    def scales(rows):
        values = torch.ones((2, rows, 256 // group), device=device)
        return swizzle_block_scale(values.to(torch.float8_e4m3fn)) if nv else values.mul(127).to(torch.uint8)
    one = torch.ones(2, device=device)
    plan = fused_moe.plan_weights(source=fused_moe.PackedSource(format=fmt, w13_layout=layout),
        activation=fused_moe.ActivationSpec(mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=2, hidden_size=256, intermediate_size=256))
    weight = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=torch.zeros((2, 512, 128), dtype=torch.uint8, device=device),
        w2=torch.zeros((2, 256, 128), dtype=torch.uint8, device=device),
        w13_block_scales=scales(512), w2_block_scales=scales(256),
        w13_global_scales=one, w2_global_scales=one,
        **(dict(input_scale=one, intermediate_scale=one, immutable_input_scales=True) if mode == "a4" else {})))
    source = torch.zeros((1, 256), dtype=torch.bfloat16, device=device)
    with NumericStorage() as numeric:
        PreparedRoutedExperts(source, weight, torch.zeros((1, 1), dtype=torch.int32, device=device),
            torch.ones((1, 1), device=device), torch.empty_like(source), session=session(device), name="experts")
    planes = [getattr(weight._impl, name) for name in
              ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale", "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale")]
    representation = weight._impl.representation
    opaque = []
    if representation is not None:
        # Inspect all actual tensor fields independently of the component's field list.
        for name, value in vars(representation.value).items():
            if isinstance(value, torch.Tensor):
                (opaque if name == "workspace" else planes).append(value)
    assert_numeric(numeric, planes, opaque=opaque)


def defer_planning(monkeypatch, module, planner):
    plan = SimpleNamespace(request=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(planner[0], planner[1], lambda *args, **kwargs: plan)
    return plan


@pytest.mark.parametrize("format", ["slab", "nvfp4", "mxfp4"])
def test_compressed_expert_records_are_numeric_but_program_storage_is_opaque(monkeypatch, format):
    from b12x._lib.quant.csf_blocks import CsfBlockPayload, CsfBlockScales, Nvfp4CompactBlockPlane
    from b12x._lib.quant.nvfp4_csf import Nvfp4CsfBatch, Nvfp4CsfDecoder
    from ignition.components import moe

    plan = defer_planning(monkeypatch, moe, (moe.fused_moe, "plan_execution"))
    monkeypatch.setattr(moe.fused_moe, "plan_scale_expansion", lambda *args, **kwargs: plan)
    def bytes():
        return torch.zeros(128, dtype=torch.uint8)
    opaque = bytes()
    batches = []
    for _ in range(2):
        batch = (Nvfp4CsfBatch(bytes(), bytes(), bytes().view(torch.int64), 128, 16, value_lut=bytes())
                 if format == "slab" else Nvfp4CompactBlockPlane(CsfBlockPayload(bytes(), 128, 16, 2, format)))
        batches.append(batch)
    indexed = (CsfBlockScales(bytes(), 128, 16, 2, "nvfp4"),) * 2
    decoder = Nvfp4CsfDecoder(*batches, (opaque,), (opaque,), bytes().view(torch.int32), indexed_scales=indexed)
    planes = {name: bytes() for name in
              ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale", "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale")}
    alternate = bytes()
    implementation = SimpleNamespace(**planes, nvfp4_csf=decoder,
        representation=SimpleNamespace(value=SimpleNamespace(micro_w13_scale=alternate, workspace=opaque)))
    experts = SimpleNamespace(_impl=implementation, has_csf_blocks=True)
    source = torch.zeros((1, 16), dtype=torch.bfloat16)
    with NumericStorage() as numeric:
        moe.PreparedRoutedExperts(source, experts, torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1, 1)), torch.empty_like(source), session=None, name="experts")
    expected = [*planes.values(), alternate, decoder.active, *(plane.storage for plane in indexed)]
    for batch in batches:
        expected.extend((batch.fixed, batch.exceptions, batch.task_offsets, batch.value_lut))
    assert_numeric(numeric, expected, opaque=(opaque,))


@pytest.mark.parametrize("kind", ["trtllm", "cute"])
def test_sm103_expert_storage_uses_the_same_numeric_contract(monkeypatch, kind):
    from ignition.components import moe, moe_sm103

    defer_planning(monkeypatch, moe_sm103, (moe_sm103, "plan_routed_experts"))
    w13, w2 = torch.zeros((2, 32, 8), dtype=torch.uint8), torch.zeros((2, 16, 8), dtype=torch.uint8)
    s13, s2 = torch.zeros_like(w13), torch.zeros_like(w2)
    one = torch.ones(2)
    if kind == "trtllm":
        experts = moe_sm103.TrtllmNvfp4Experts(w13, s13, w2, s2, one, one, one, one, one, one, one)
    else:
        experts = moe_sm103.CuteW4A16Experts(w13, s13, one, w2, s2, one, (s13, s2))
    source = torch.zeros((1, 16), dtype=torch.bfloat16)
    with NumericStorage() as numeric:
        operation = moe.PreparedRoutedExperts(source, experts, torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1, 1)), torch.empty_like(source), session=None, name="experts")
    assert isinstance(operation, moe_sm103.Sm103RoutedExperts)
    assert_numeric(numeric, (value for value in vars(experts).values() if isinstance(value, torch.Tensor)))
