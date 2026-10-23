"""Direct packed-weight loading and graph replay for ordinary FP4 storage."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def checkpoint(path, tensors, layers):
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}}))
    return inspect_indexed_checkpoint(path)


@pytest.mark.parametrize("local,tokens", [(96, 1), (192, 7), (384, 128), (864, 4), (1728, 1), (384, 4096)])
@torch.no_grad()
def test_mxfp8_gate_up_conversion_and_compiled_mlp(local, tokens):
    from torch.multiprocessing.reductions import StorageWeakRef
    from ignition.components.mlp import MLPScratch, PreparedMLP
    from ignition.components.weights import MLPWeights, quantize_bf16_linear_as_mxfp8
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.reference.qwen38_quant import decode_mxfp8, quantize_mxfp8_activation

    gen = torch.Generator().manual_seed(570)
    hidden, limit = 4096, .5
    original = (torch.randn(2 * local, hidden, generator=gen) * .03125).bfloat16()
    original[0].zero_()
    source = original.cuda()
    old_storage = StorageWeakRef(source.untyped_storage())
    packed = quantize_bf16_linear_as_mxfp8(source)
    del source
    assert old_storage.expired()
    decoded = decode_mxfp8(packed.weight.values.cpu(), packed.weight.scale_rows[0].view(torch.uint8).cpu())
    assert not decoded[0].any()
    assert (decoded - original.float()).norm() / original.float().norm() < .035
    down = (torch.randn(hidden, local, generator=gen) / local**.5).bfloat16()
    cpu = (torch.randn(tokens, hidden, generator=gen) * .5).bfloat16()
    x = cpu.cuda()
    scratch = MLPScratch.allocate(tokens, hidden, local, x.device)
    arena = ScratchArena(x.device)
    with RecordedPreparationSession(device=x.device, checkpoint="mxfp8-conversion",
            workload_schema="packed-swiglu", numerical_policy="auto") as session:
        session.register_scratch(arena)
        mlp = PreparedMLP(x, MLPWeights(packed, down.cuda()), scratch.gate_up, scratch.intermediate,
            scratch.output, session=session, name="mxfp8-mlp", workspace=arena,
            swiglu_limit=limit, round_silu=False, clamp_after_silu=True)
        session.tune(show_progress=False)
        compiled = compile_forward(lambda: mlp(x))
        compiled()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for factor in (1, -.5):
            current = (cpu * factor).bfloat16()
            x.copy_(current)
            operand = current.float()
            if mlp.up_projection.plan.selection.config.mode == "quantized":
                operand = quantize_mxfp8_activation(operand)
            gate, up = (operand @ decoded.T).bfloat16().float().chunk(2, -1)
            intermediate = (torch.nn.functional.silu(gate).clamp(max=limit)
                            * up.clamp(-limit, limit)).bfloat16()
            expected = intermediate.float() @ down.float().T
            scratch.gate_up.fill_(float("nan"))
            scratch.intermediate.fill_(float("nan"))
            scratch.output.fill_(float("nan"))
            arena.tensor.fill_(0x7f)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            actual = scratch.output.float().cpu()
            assert torch.isfinite(actual).all()
            assert (actual - expected).norm() / expected.norm() < .025
        graph.reset()


def test_mxfp8_conversion_padding_and_nonfinite_weights():
    from ignition.components.weights import quantize_bf16_linear_as_mxfp8
    from ignition.reference.qwen38_quant import decode_mxfp8

    weight = torch.ones((64, 288), device="cuda", dtype=torch.bfloat16)
    packed = quantize_bf16_linear_as_mxfp8(weight)
    assert (packed.in_features, packed.padded_in_features) == (288, 384)
    decoded = decode_mxfp8(packed.weight.values.cpu(), packed.weight.scale_rows[0].view(torch.uint8).cpu())
    torch.testing.assert_close(decoded[:, :288], weight.float().cpu(), atol=0, rtol=0)
    assert not decoded[:, 288:].any()
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite BF16 weights"):
        quantize_bf16_linear_as_mxfp8(weight)


def packed(rows, columns, fmt, generator):
    group = 32 if fmt == "mxfp4" else 16
    w = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator)
    if fmt == "mxfp4":
        s = torch.randint(121, 125, (rows, columns // group), dtype=torch.uint8, generator=generator)
    else:
        s = (torch.rand((rows, columns // group), generator=generator) * .02 + .01).to(torch.float8_e4m3fn)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    values = lut[torch.stack((w & 15, w >> 4), dim=-1).long()].flatten(-2)
    scales = torch.exp2(s.float() - 127) if fmt == "mxfp4" else s.float()
    tensors = {"weight": w, "weight_scale": s}
    if fmt != "mxfp4":
        tensors["weight_scale_2"] = torch.tensor(.5)
        scales = scales * .5
    return tensors, values * scales.repeat_interleave(group, dim=-1)


def mxfp4_activation(x):
    from ignition.reference.qwen38_quant import _FP4_VALUES

    groups = x.float().reshape(x.shape[0], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.where(amax > 0, amax / 6, 1).log2().ceil().clamp(-127, 127))
    v = groups / scale
    z = v.abs()
    code = ((z > .25).int() + (z >= .75).int() + (z > 1.25).int() + (z >= 1.75).int()
            + (z > 2.5).int() + (z >= 3.5).int() + (z > 5).int())
    code |= torch.where((v < 0) & (code != 0), 8, 0)
    return (torch.tensor(_FP4_VALUES)[code] * scale).reshape(x.shape)


@pytest.mark.parametrize("rows", [1, 17, 128])
@pytest.mark.parametrize("fmt", ["nvfp4", "nvfp4_a4", "mxfp4"])
def test_ordinary_fp4_linear_graph(tmp_path, rows, fmt):
    from b12x.preparation import PreparationSession
    from ignition.components.linear import QuantizedProjection
    from ignition.preparation.scratch import ScratchArena

    gen = torch.Generator().manual_seed(871)
    tensors, weight = packed(256, 512, fmt, gen)
    algo, group = ("MXFP4", 32) if fmt == "mxfp4" else ("W4A16_NVFP4", 16)
    if fmt == "nvfp4_a4":
        algo = "NVFP4"
        tensors["input_scale"] = torch.tensor(.03125)
    manifest = checkpoint(tmp_path, {"proj." + name: t for name, t in tensors.items()},
                          {"proj": {"quant_algo": algo, "group_size": group}})
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        w = loader.load_linear("proj", rows=(128, 256), columns=(32, 320))
        assert loader._session.torch_copy_bytes == 0
    cpu = torch.randn(rows, 288, generator=gen).to(torch.bfloat16)
    x, out = cpu.cuda(), torch.empty((rows, 128), device="cuda", dtype=torch.bfloat16)
    with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
        with pytest.raises(ValueError, match="output must not overlap bias"):
            QuantizedProjection(x, w, out, bias=out[0], session=session, name="aliased",
                                workspace=ScratchArena(x.device))
        projection = QuantizedProjection(x, w, out, session=session, name="linear", activation_mode="auto",
                                         workspace=ScratchArena(x.device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection(x)
        for factor in (1, -.5):
            x.copy_(cpu * factor)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            source = mxfp4_activation(cpu * factor) if fmt == "mxfp4" else (cpu * factor).float()
            expected = source @ weight[128:256, 32:320].T
            if fmt == "nvfp4_a4":
                from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_linear_reference

                stored = Nvfp4Projection(tensors["weight"][128:256, 16:160],
                    tensors["weight_scale"][128:256, 2:20], tensors["weight_scale_2"], tensors["input_scale"])
                expected = nvfp4_linear_reference(cpu * factor, stored)
            torch.testing.assert_close(out.float().cpu(), expected, rtol=.025, atol=.015)


@pytest.mark.parametrize("fmt,mode", [("nvfp4", "a16"), ("nvfp4", "a4"), ("mxfp4", "a8")])
@pytest.mark.parametrize("compression", [None, "csf"])
def test_ordinary_fp4_experts_graph(tmp_path, monkeypatch, fmt, mode, compression):
    from torch.multiprocessing.reductions import StorageWeakRef
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_expert_reference, quantize_mxfp8_activation

    gen = torch.Generator().manual_seed(192)
    tensors, decoded, serialized = {}, [], []
    for expert in range(2):
        weights, stored = [], []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            data, logical = packed(256, 256, fmt, gen)
            tensors.update({f"experts.{expert}.{projection}.{name}": t for name, t in data.items()})
            if mode == "a4":
                data["input_scale"] = torch.tensor(.03125)
                tensors[f"experts.{expert}.{projection}.input_scale"] = data["input_scale"]
            weights.append(logical)
            stored.append(data)
        decoded.append(weights)
        serialized.append(stored)
    manifest = checkpoint(tmp_path, tensors, {})
    scale_vectors = (torch.full((2,), .5), torch.full((2,), .5), torch.full((2,), 32.), torch.full((2,), 32.))
    buffers = None
    if compression is not None:
        shapes = ((2, 256, 16), (2, 256, 8)) if fmt == "nvfp4" else ((2, 8, 256), (2, 4, 256))
        dtype = torch.float8_e4m3fn if fmt == "nvfp4" else torch.uint8
        buffers = tuple(torch.empty(shape, dtype=dtype, device="cuda") for shape in shapes)
    native_scales = []
    prepare_weights = fused_moe.prepare_weights

    def inspect_scales(*, plan, weights, scale_scratch=None):
        if compression is not None:
            native_scales.extend(StorageWeakRef(t.untyped_storage())
                                 for t in (weights.w13_block_scales, weights.w2_block_scales))
        return prepare_weights(plan=plan, weights=weights, scale_scratch=scale_scratch)

    monkeypatch.setattr(fused_moe, "prepare_weights", inspect_scales)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        experts = loader.load_fp4_experts(("experts.0", "experts.1"), hidden_size=256, intermediate_size=256,
            intermediate_range=(128, 256), format=fmt, activation_mode=mode,
            scale_vectors=scale_vectors if mode == "a4" else None,
            scale_compression=compression, scale_scratch=buffers)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.scale_compression == compression
    assert all(reference.expired() for reference in native_scales)
    cpu = (torch.randn(2, 256, generator=gen) * .1).to(torch.bfloat16)
    x, out = cpu.cuda(), torch.empty_like(cpu, device="cuda")
    ids = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    probabilities = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
        operation = PreparedRoutedExperts(x, experts, ids, probabilities, out, session=session, name="moe")
        assert operation.plan.query.a4_prefill is False
        with pytest.raises(ValueError, match="a4_prefill differs from the prepared numerical contract"):
            fused_moe.bind(operation.plan, a=x, topk_ids=ids, topk_weights=probabilities,
                output=out, scratch=operation.scratch, a4_prefill=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        for order in ((0, 1), (1, 0)):
            ids.copy_(torch.tensor(order, dtype=torch.int32)[:, None])
            for buffer in buffers or ():
                buffer.view(torch.uint8).fill_(0x7f)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = []
            for row, expert in enumerate(order):
                gate, up, down = decoded[expert]
                source = cpu[row:row + 1].float()
                if mode == "a4":
                    projections = []
                    for index, data in enumerate(serialized[expert]):
                        w, sf = data["weight"], data["weight_scale"]
                        w, sf = (w[:, 64:128], sf[:, 8:16]) if index == 2 else (w[128:256], sf[128:256])
                        projections.append(Nvfp4Projection(w, sf, data["weight_scale_2"], data["input_scale"]))
                    result = nvfp4_expert_reference(source, *projections).down
                else:
                    if mode == "a8":
                        source = quantize_mxfp8_activation(source)
                    hidden = (torch.nn.functional.silu(source @ gate[128:256].T) * (source @ up[128:256].T)).to(torch.bfloat16).float()
                    if mode == "a8":
                        hidden = quantize_mxfp8_activation(hidden)
                    result = hidden @ down[:, 128:256].T
                expected.append(result)
            expected = torch.cat(expected)
            relative = torch.linalg.vector_norm(out.float().cpu() - expected) / torch.linalg.vector_norm(expected)
            assert relative < .035, float(relative)


@pytest.mark.parametrize("local", [96, 192, 384])
@pytest.mark.parametrize("tokens", [1, 7, 128])
@pytest.mark.parametrize("compression", [None, "csf"])
@torch.no_grad()
def test_bf16_experts_convert_to_nvfp4_w4a16(tmp_path, monkeypatch, local, tokens, compression):
    from torch.multiprocessing.reductions import StorageWeakRef
    from b12x.moe import fused_moe
    from b12x.moe._shared.kernels.w4a16.host import unswizzle_block_scale
    from b12x.preparation import PreparationSession
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    gen = torch.Generator().manual_seed(501)
    count, hidden, intermediate, topk, limit = 4, 256, 1536, 2, .5
    tensors = {}
    for name, scale in (("up", .125), ("gate", .5), ("down", .0625)):
        shape = (count, hidden, intermediate) if name == "down" else (count, intermediate, hidden)
        value = (torch.randn(shape, generator=gen) * scale).bfloat16()
        value[0].zero_()
        tensors[f"moe.{name}_proj.weight"] = value
    manifest = checkpoint(tmp_path, tensors, {})
    decoded = {}
    prepare_weights = fused_moe.prepare_weights
    native_scales = []
    padded = (local + 127) // 128 * 128
    buffers = None if compression is None else tuple(
        torch.empty(shape, dtype=torch.float8_e4m3fn, device="cuda")
        for shape in ((count, 2 * padded, hidden // 16), (count, hidden, padded // 16)))

    def inspect_quantized(*, plan, weights, scale_scratch=None):
        assert plan.activation.mode is fused_moe.ActivationMode.A16
        assert plan.scale_compression == compression
        assert weights.input_scale is None and weights.intermediate_scale is None
        if compression is not None:
            native_scales.extend(StorageWeakRef(t.untyped_storage())
                                 for t in (weights.w13_block_scales, weights.w2_block_scales))
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        for name, packed_weights, blocks, gains in (
            ("up_gate", weights.w13, weights.w13_block_scales, weights.w13_global_scales),
            ("down", weights.w2, weights.w2_block_scales, weights.w2_global_scales),
        ):
            packed_weights, blocks, gains = packed_weights.cpu(), blocks.cpu(), gains.cpu()
            assert (torch.isfinite(gains) & (gains > 0)).all()
            rows, columns = packed_weights.shape[1], packed_weights.shape[2] * 2
            scales = torch.stack([unswizzle_block_scale(blocks[e], rows, columns // 16) for e in range(count)])
            values = lut[torch.stack((packed_weights & 15, packed_weights >> 4), -1).long()].flatten(-2)
            decoded[name] = values * scales.repeat_interleave(16, -1) * gains[:, None, None]
            assert not decoded[name][0].any()
        return prepare_weights(plan=plan, weights=weights, scale_scratch=scale_scratch)

    monkeypatch.setattr(fused_moe, "prepare_weights", inspect_quantized)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        experts = loader.load_bf16_experts_as_nvfp4("moe", num_experts=count, hidden_size=hidden,
            intermediate_size=intermediate, intermediate_range=(local, 2 * local),
            swiglu_limit=limit, swiglu_clamp_after_silu=True,
            scale_compression=compression, scale_scratch=buffers)
        assert loader._session.torch_copy_bytes == 0
    assert all(reference.expired() for reference in native_scales)
    assert experts.intermediate_size == padded
    up, gate = decoded["up_gate"].chunk(2, 1)
    down = decoded["down"]
    assert not up[:, local:].any() and not gate[:, local:].any() and not down[:, :, local:].any()
    for name, quantized in (("up", up[:, :local]), ("gate", gate[:, :local]), ("down", down[:, :, :local])):
        original = tensors[f"moe.{name}_proj.weight"].float()
        original = original[:, :, local:2 * local] if name == "down" else original[:, local:2 * local]
        assert (quantized - original).norm() / original.norm() < .12

    cpu = (torch.randn(tokens, hidden, generator=gen) * .5).bfloat16()
    x, output = cpu.cuda(), torch.empty_like(cpu, device="cuda")
    ids = (torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk) % count).cuda()
    routes = torch.randn(tokens, topk, generator=gen).softmax(-1).cuda()
    arena = ScratchArena(x.device)
    with PreparationSession(device=x.device, autotune=True, compile_workers=0) as session:
        operation = PreparedRoutedExperts(x, experts, ids, routes, output, session=session,
                                          name="converted-experts", workspace=arena)
        session.freeze()
        arena.finalize()
        arena.freeze()
        compiled = compile_forward(operation)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        for shift in (0, 1):
            ids.add_(shift).remainder_(count)
            expected = torch.zeros_like(cpu, dtype=torch.float32)
            for expert in range(count):
                rows, choices = (ids.cpu() == expert).nonzero(as_tuple=True)
                if not rows.numel():
                    continue
                g = cpu[rows].float() @ gate[expert].T
                u = cpu[rows].float() @ up[expert].T
                activation = (torch.nn.functional.silu(g).clamp(max=limit) * u.clamp(-limit, limit)).bfloat16()
                expected[rows] += (activation.float() @ down[expert].T) * routes.cpu()[rows, choices, None]
            output.fill_(float("nan"))
            arena.tensor.fill_(0x7f)
            for buffer in buffers or ():
                buffer.view(torch.uint8).fill_(0x7f)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            actual = output.float().cpu()
            assert torch.isfinite(actual).all()
            assert (actual - expected).norm() / expected.norm() < .025
        graph.reset()


def ggml_blocks(rows, columns, codec):
    from b12x._lib.quant.block_codec import block_codec
    spec = block_codec(codec)
    raw = torch.randint(0, 256, (rows, columns // spec.block_weights, spec.block_bytes), dtype=torch.uint8)
    raw[..., :2] = torch.full((rows, columns // spec.block_weights, 1), .003, dtype=torch.float16).view(torch.uint8)
    entry = dict(quant_algo=codec.upper(), group_size=spec.block_weights,
                 block_payload_bytes=spec.block_bytes, packing="ggml")
    return raw, entry


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("rows", [1, 17])
def test_ggml_projection_fusion_and_sharding_graph(tmp_path, codec, rows):
    from importlib import import_module
    from b12x.preparation import PreparationSession
    from ignition.components.linear import QuantizedProjection
    from ignition.preparation.scratch import ScratchArena

    torch.manual_seed(72)
    tensors, entries, decoded = {}, {}, []
    for name in ("q", "k"):
        raw, entries[name] = ggml_blocks(256, 768, codec)
        tensors[name + ".weight"] = raw
        decoded.append(import_module(f"b12x.testing.{codec}_reference").dequantize_blocks(raw)[8:136, 256:512])
    manifest = checkpoint(tmp_path, tensors, entries)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        weight = loader.load_linear_group(("q", "k"), row_ranges=((8, 136), (8, 136)), column_range=(256, 512))
        assert loader._session.torch_copy_bytes == 0
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(rows, 256, device="cuda", dtype=torch.bfloat16)
    with PreparationSession(device=x.device, autotune=False, compile_workers=1) as session:
        projection = QuantizedProjection(x, weight, out, session=session, name="blocks",
                                         activation_mode="auto", workspace=ScratchArena(x.device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection(x)
        for _ in range(2):
            x.normal_()
            expected = x.float().cpu() @ torch.cat(decoded).bfloat16().float().T
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(out.float().cpu(), expected, rtol=.015, atol=.015)
        graph.reset()
