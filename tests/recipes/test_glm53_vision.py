"""GLM checkpoint vision numerics, padding isolation, and native graph replay."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.glm53.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights
from ignition.reference.qwen38_quant import (
    CheckpointTensors, decode_mxfp8, decode_nvfp4, quantize_mxfp8_activation,
    read_nvfp4_projection,
)

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")


def test_vision_geometry():
    metadata = VisionMetadata.allocate(256, 2, "cpu")
    scratch = VisionScratch.allocate(1024, "cpu").rows(256)
    assert scratch.patches.shape == (256, 1176)
    assert scratch.query.shape == (256, 16, 64)
    assert scratch.output.shape == (64, 4096)
    assert metadata.position_ids.shape == (256, 2)
    assert metadata.cu_seqlens.shape == (3,)
    with pytest.raises(ValueError):
        scratch.rows(255)


@pytest.fixture(scope="module")
def manifest():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    # Vision qualification needs only the indexed shards containing the tower.
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({shard for name, shard in index.items() if name.startswith("model.visual.")})
    tensors, shards = {}, []
    for filename in files:
        record, entries = _read_header(snapshot / filename, filename)
        assert all(index[name] == filename for name in entries)
        shards.append(record)
        tensors.update(entries)
    return IndexedCheckpoint(snapshot, tensors, tuple(shards))


def _module(manifest):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
    config = Glm5NextVisionConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text())["vision_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Glm5NextVisionModel(config)
    return module.eval().requires_grad_(False)


class _ReferenceProjection(torch.nn.Module):
    def __init__(self, weight, bias, *, quantized, scale=1.):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.quantized, self.scale = quantized, scale

    def forward(self, source):
        source = quantize_mxfp8_activation(source) if self.quantized else source.float()
        result = ((source @ self.weight.T) * self.scale).bfloat16()
        return result if self.bias is None else result + self.bias


def _reference(manifest, weights, device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionRotaryEmbedding
    module = _module(manifest)
    for path, value in weights.parameters.items():
        parent, _, attribute = path.rpartition(".")
        setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
    declarations = json.loads((manifest.snapshot / "config.json").read_text())["quantization_config"]["quantized_layers"]
    with CheckpointTensors(manifest, byte_budget=1 << 30) as reader:
        for path in weights.projections:
            prefix = f"model.visual.{path}"
            if declarations[prefix]["quant_algo"] == "MXFP8":
                decoded = decode_mxfp8(reader.read(prefix + ".weight"), reader.read(prefix + ".weight_scale"))
                scale, quantized = 1., True
            else:
                packed = read_nvfp4_projection(reader, prefix)
                decoded = decode_nvfp4(packed.packed, packed.scales, torch.ones_like(packed.weight_scale_2)).bfloat16().float()
                scale, quantized = packed.weight_scale_2.item(), False
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute,
                    _ReferenceProjection(decoded.to(device), weights.parameters.get(path + ".bias"),
                                         quantized=quantized, scale=scale))
    module.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(module.config).to(device)
    return module


def _input(sizes, variant):
    from PIL import Image
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
    images = []
    for width, height in sizes:
        pixels = bytes(component for y in range(height) for x in range(width)
                       for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
        images.append(Image.frombytes("RGB", (width, height), pixels))
    return Glm5NextImageProcessor()(images=images, return_tensors="pt")


def _check_stages(prepared, reference, pixels, metadata):
    """Compare each stage with identical input to isolate arithmetic and layout errors."""
    errors = []
    def compare(name, actual, expected):
        relative = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))
        errors.append((name, relative))
        assert relative < .03, errors
    prepared.scratch.patches.copy_(pixels)
    hidden = reference.patch_embed(pixels)
    compare("patch", prepared.patch_embed(prepared.scratch.patches), hidden)
    rope = reference.rotary_pos_emb(hidden, metadata.position_ids)
    for index, (actual, expected) in enumerate(zip(prepared.module.blocks, reference.blocks, strict=True)):
        output = expected(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rope)
        compare(f"block{index}", actual(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rope), output)
        hidden = output
    hidden = reference.post_layernorm(hidden)
    merged = reference.downsample(hidden.view(-1, 2, 2, 1024).permute(0, 3, 1, 2)).view(-1, 4096)
    compare("downsample", prepared.downsample(hidden.view(-1, 4096)), merged)
    compare("merger", prepared.module.merger(merged), reference.merger(merged))
    return max(relative for _, relative in errors)


@CUDA
@torch.no_grad()
@pytest.mark.parametrize("capacity", [256, 1024, 4096])
@pytest.mark.parametrize("compiler", ["none", "trace", "aot", "worker"])
def test_checkpoint_vision_native_replay(manifest, capacity, compiler, record_property, tmp_path):
    from b12x.preparation import PreparationSession
    from transformers.vision_utils import get_vision_position_ids
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    module = _module(manifest)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    reference = _reference(manifest, weights, device)
    scratch = VisionScratch.allocate(capacity, device)
    metadata = VisionMetadata.allocate(capacity, 2, device)
    pixels = torch.zeros((capacity, 1176), dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session)
        session.freeze()
        def forward():
            return prepared(pixels)
        if compiler == "trace":
            from ignition.preparation.compilation import compile_forward
            forward = compile_forward(forward)
        elif compiler == "aot":
            from ignition.preparation.aot import aot_forward
            owner = forward
            artifact = aot_forward(owner)
            forward = lambda: artifact(owner)
        elif compiler == "worker":
            from functools import partial
            from ignition.preparation.aot import ForwardCache
            from ignition.preparation.execution import GraphSignature
            cache = ForwardCache(namespace={"vision": "glm53"}, device=device, root=tmp_path)
            forward = cache.prepare_many({"vision": partial(prepared, pixels)},
                {"vision": GraphSignature("vision", 1, capacity)})["vision"]
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph},
            buffers={"pixels": pixels, "embeddings": scratch.output},
            resources=(prepared, weights, session, stream, forward)).start_worker()
        side = int(capacity ** .5) * 14
        errors = []
        try:
            for variant, sizes in enumerate(([(side, side)], [(112, 112), (112, 224)], [(112, 112)])):
                inputs = _input(sizes, variant)
                grid = inputs.image_grid_thw
                count = inputs.pixel_values.shape[0]
                metadata.position_ids.zero_()
                metadata.position_ids[:count].copy_(get_vision_position_ids(grid, 2))
                lengths = grid.prod(-1).cumsum(0).tolist()
                metadata.cu_seqlens.copy_(torch.tensor([0, *lengths, *([count] * (2 - len(lengths)))], dtype=torch.int32))
                metadata.num_patches.fill_(count)
                pixels.fill_(7.)
                pixels[:count].copy_(inputs.pixel_values)
                expected = reference(inputs.pixel_values.to(device), grid.to(device)).pooler_output.float()
                if variant == 0:
                    record_property("maximum_stage_relative_l2", _check_stages(prepared, reference, pixels, metadata))
                # GLM amplifies BF16 rounding differences through its tower.
                # Bound full-tower drift by the canonical eager/SDPA spread;
                # each isolated stage retains the fixed 3% numerical gate.
                reference.config._attn_implementation = "sdpa"
                sdpa = reference(inputs.pixel_values.to(device), grid.to(device)).pooler_output.float()
                reference.config._attn_implementation = "eager"
                spread = float((sdpa - expected).norm() / expected.norm().clamp_min(1e-12))
                replay_expected = forward().clone()
                scratch.output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                actual = scratch.output[:count // 4].float()
                relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
                errors.append({"variant": variant, "patches": count, "relative_l2": relative,
                               "canonical_backend_relative_l2": spread,
                               "max_absolute": float((actual - expected).abs().max())})
                assert relative < spread + .03, errors
                torch.testing.assert_close(scratch.output, replay_expected, rtol=0, atol=0)
                assert torch.count_nonzero(scratch.output[count // 4:]) == 0
        finally:
            worker.close()
        record_property("numerical_errors", json.dumps(errors))


@CUDA
@torch.no_grad()
def test_generation_publishes_features_and_rejects_stale_uploads(manifest):
    from b12x.preparation import PreparationSession
    from transformers.vision_utils import get_vision_position_ids
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import overlay
    from ignition.recipes.glm53.batch import BatchInputs
    from ignition.recipes.glm53.vision_state import PreparedVisionGeneration, VisionInputs

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    module = _module(manifest)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    # Poisoning the overlaid storage exercises the same scratch lifetime as text-to-vision transitions.
    scratch, = overlay((VisionScratch.allocate(256, device),))
    metadata = VisionMetadata.allocate(256, 1, device)
    inputs = VisionInputs.allocate(device)
    requests = DeviceRequestState.allocate(device, vocab_size=154880, capacity=2)
    batch = BatchInputs.allocate(3, 1, device, capacity=2)
    prefill = PrefillVisionInputs.allocate(3, device)
    state = VisionState(batch, prefill, requests, image_token=154854, features=128, hidden=4096, image_slots=1)
    completion = CompletionOutputs.allocate((2, 2, 8), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    requests.generation[1], requests.status[1] = 7, 1
    inputs.vision_slot.fill_(1)
    inputs.vision_generation.fill_(7)
    state.feature_slot.zero_()
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        encoder = PreparedVision(module, scratch, metadata, weights, session=session)
        session.freeze()
        operation = PreparedVisionGeneration(encoder, inputs, state, completion, frame, sequence)
        run = compile_forward(operation)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph}, buffers={},
            resources=(operation, run, weights, session, scratch)).start_worker()
        def replay():
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            ticket = worker.submit(0)
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations

        try:
            offset = 0
            for variant, size in enumerate((224, 112)):
                image = _input([(size, size)], variant)
                count = image.pixel_values.shape[0]
                inputs.vision_pixel_values.fill_(float("nan"))
                inputs.vision_pixel_values[:count].copy_(image.pixel_values)
                inputs.vision_position_ids[:count].copy_(get_vision_position_ids(image.image_grid_thw, 2))
                inputs.vision_num_patches.fill_(count)
                inputs.vision_cu_seqlens[1] = count
                inputs.vision_feature_offset.fill_(offset)
                requests.status[1] = 1
                operation()
                expected = scratch.output[:count // 4].clone()
                state.count[1] = offset
                state.arena[:, offset:].fill_(float("nan"))
                scratch.patches.untyped_storage().fill_(0xFF)
                replay()
                assert operation.errors.item() == 0
                assert state.generation.tolist() == [0, 7]
                assert state.count.tolist() == [0, offset + count // 4]
                torch.testing.assert_close(state.arena[0, offset:offset + count // 4], expected, rtol=0, atol=0)
                assert torch.isnan(state.arena[0, offset + count // 4:]).all()
                offset += count // 4
            batch.row_slot_ids[0], batch.row_generations[0] = 1, 7
            batch.token_ids.copy_(torch.tensor([7, 154854, 154854], device=device))
            prefill.vision_token_indices.copy_(torch.tensor([-1, 0, offset - 1], device=device))
            assert state.validate_prefill(batch.token_ids, torch.tensor([3], device=device))
            embedding = torch.ones((3, 4096), dtype=torch.bfloat16, device=device)
            state.inject(batch, embedding)
            assert (embedding[0] == 1).all()
            torch.testing.assert_close(embedding[1], state.arena[0, 0], rtol=0, atol=0)
            torch.testing.assert_close(embedding[2], state.arena[0, offset - 1], rtol=0, atol=0)
            before = state.arena.clone()
            requests.generation[1] = 8
            replay()
            assert operation.errors.item() != 0
            assert requests.status[1].item() == 1
            assert state.count.tolist() == [0, offset]
            torch.testing.assert_close(state.arena, before, rtol=0, atol=0, equal_nan=True)
        finally:
            worker.close()
