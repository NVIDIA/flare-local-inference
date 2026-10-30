"""Checkpoint MXFP8 DFlash2 body arithmetic and mutable graph replay."""

from dataclasses import replace
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.recipes.test_qwen38_dense_dflash_cuda import DRAFT, TARGET, relative_error, rms, rotary

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires assigned CUDA GPU and Qwen DFlash2 checkpoint")


def decoded(weight):
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
    packed = weight.weight
    return dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows)[:, :weight.in_features].bfloat16()


def projection_reference(projection, source):
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch, quantize_mxfp8_rows_torch
    from b12x.preparation.types import require_prepared
    state = require_prepared(projection.plan, "gemm.blockscaled_precision", source.device)
    if hasattr(state, "resolve"):
        state = state.resolve(source)
    mode = state.config.mode
    assert mode in ("a16", "quantized"), mode
    if mode == "quantized":
        encoded = quantize_mxfp8_rows_torch(source)
        source = dequantize_mxfp8_rows_torch(encoded.values, encoded.scale_rows).bfloat16()
    return F.linear(source.float(), decoded(projection.weight).float()).bfloat16()


def check(actual, expected):
    error = relative_error(actual, expected)
    assert error < .008, error
    return error


def replay(graph):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
        assert before[name] == after[name], name


@pytest.fixture(scope="module")
def weights():
    from ignition.components.dflash2_weights import DraftWeights
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.qwen38_dense.config import ModelConfig
    from ignition.recipes.qwen38_dense.draft_config import DraftConfig
    if not all((path / "config.json").is_file() for path in (TARGET, DRAFT)):
        pytest.skip("requires pinned Qwen3.8-27B target and DFlash2 checkpoints")
    target = ModelConfig.load(TARGET)
    config = replace(target, draft=DraftConfig.load(DRAFT, target))
    shard = TensorParallelShard(0, 1)
    return SimpleNamespace(config=config, shard=shard,
        draft=DraftWeights.load(DRAFT, config.draft, "cuda:0", shard, online_mxfp8=True))


@torch.no_grad()
def test_checkpoint_body_geometry_context_consistency_and_preserved_weights(weights):
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from safetensors import safe_open
    draft = weights.draft
    context = decoded(draft.context_kv)
    with safe_open(DRAFT / "model.safetensors", framework="pt", device="cpu") as source:
        def exact(actual, name):
            assert actual.dtype == torch.bfloat16
            assert torch.equal(actual.cpu(), source.get_tensor(name)), name

        def geometry(actual, rows, columns):
            assert isinstance(actual, MXFP8LinearWeight)
            assert (actual.out_features, actual.in_features) == (rows, columns)
            assert decoded(actual).shape == (rows, columns)

        geometry(draft.projection, 5120, 25600)
        geometry(draft.context_kv, 10240, 5120)
        exact(draft.hidden_norm, "hidden_norm.weight")
        exact(draft.norm, "norm.weight")
        for index, layer in enumerate(draft.layers):
            prefix = f"layers.{index}."
            for actual, rows, columns in ((layer.qkv, 6144, 5120), (layer.output, 5120, 4096),
                    (layer.mlp.gate_up, 34816, 5120), (layer.mlp.down, 5120, 17408)):
                geometry(actual, rows, columns)
            assert torch.equal(context[index * 2048:(index + 1) * 2048], decoded(layer.qkv)[4096:])
            for actual, suffix in ((layer.query_norm, "self_attn.q_norm"), (layer.key_norm, "self_attn.k_norm"),
                    (layer.input_norm, "input_layernorm"), (layer.post_attention_norm, "post_attention_layernorm")):
                exact(actual, prefix + suffix + ".weight")
            for name in ("attention_conv", "mlp_conv"):
                exact(getattr(layer, name)["base"], prefix + name + ".base_kernel")
                exact(getattr(layer, name)["projection"], prefix + name + ".kernel_projection.weight")
        for name, tensor in draft.selector.items():
            suffix = dict(projection="hidden_projection.weight", predecessor="predecessor_codebook",
                          successor="successor_codebook")[name]
            exact(tensor, "candidate_selector." + suffix)
        raw = source.get_tensor("fc.weight").cuda()
        print(dict(context_projection_weight_relative_l2=relative_error(decoded(draft.projection), raw)))


def metadata(rows=1):
    from ignition.components.paged_batch import Metadata
    return Metadata.allocate(8 * rows, rows, 1, "cuda", samples=8 * rows)


def set_metadata(batch, starts, lengths, states):
    rows = len(starts)
    batch.set(torch.tensor(starts, device="cuda"), torch.tensor(lengths, device="cuda", dtype=torch.int32),
        torch.tensor(states, device="cuda"), torch.zeros((rows, 1), device="cuda", dtype=torch.int32),
        torch.zeros((rows, 8), device="cuda", dtype=torch.int64),
        torch.tensor([rows], device="cuda", dtype=torch.int32))


def rings():
    return {f"draft{i}.{kind}": torch.zeros((3, 2048, 8, 128), device="cuda", dtype=torch.float8_e4m3fn)
            for i in range(5) for kind in ("key", "value")}


@torch.no_grad()
def test_context_compiled_mutable_replay_ring_wrap_and_inactive_rows(weights):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import PreparedContext
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    torch.manual_seed(5027)
    source = torch.randn((16, 25600), device="cuda", dtype=torch.bfloat16)
    batch, history = metadata(2), rings()
    set_metadata(batch, [2047, 0], [8, 0], [1, 2])
    counts = torch.tensor([8, 0], device="cuda", dtype=torch.int32)
    buffers, arena = GraphBufferPool("cuda"), ScratchArena("cuda")
    with PreparationSession(device="cuda:0", autotune=False) as session:
        context = PreparedContext(weights, source, batch, SimpleNamespace(reserved_region=history.__getitem__),
            session=session, name="mxfp8.context", workspace=arena, runtime=None, coordinator=None,
            buffers=buffers.cursor())
        arena.finalize()
        forward = compile_forward(lambda: context(source, counts))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for start, accepted in ((2047, 0), (2047, 1), (2047, 7), (4095, 8)):
            set_metadata(batch, [start, 0], [8, 0], [1, 2])
            source.normal_()
            counts[0] = accepted
            for tensor in history.values():
                tensor.fill_(7)
            replay(graph)
            projected = rms(projection_reference(context.project, source), weights.draft.hidden_norm)
            check(context.projected, projected)
            kv = projection_reference(context.project_kv, context.projected).view(16, 5, 2048)
            check(context.kv, kv.flatten(1))
            kv = context.kv.view(16, 5, 2048)
            for index, layer in enumerate(weights.draft.layers):
                key = rotary(rms(kv[:, index, :1024].reshape(16, 8, 128), layer.key_norm), batch.positions)
                value = kv[:, index, 1024:].reshape(16, 8, 128)
                for kind, expected in (("key", key), ("value", value)):
                    ring = history[f"draft{index}.{kind}"]
                    positions = (torch.arange(accepted, device="cuda") + start) % 2048
                    mask = torch.ones((3, 2048), device="cuda", dtype=torch.bool)
                    mask[1, positions] = False
                    assert ring.float()[mask].eq(7).all()
                    if accepted:
                        check(ring.float()[1, positions], expected[:accepted].clamp(-448, 448).to(torch.float8_e4m3fn))
        graph.reset()


@pytest.mark.parametrize("index", range(5))
@torch.no_grad()
def test_attention_mlp_compiled_mutable_replay(weights, index):
    from b12x.preparation import PreparationSession
    from ignition.components.dflash2 import DraftAttention, DraftScratch
    from ignition.components.mlp import PreparedMLP
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    torch.manual_seed(2718 + index)
    source = torch.randn((8, 5120), device="cuda", dtype=torch.bfloat16)
    batch, history = metadata(), rings()
    set_metadata(batch, [17], [8], [1])
    for tensor in history.values():
        tensor.copy_((torch.randn(tensor.shape, device="cuda") * .1).to(torch.float8_e4m3fn))
    saved = {name: value.view(torch.uint8).clone() for name, value in history.items()}
    buffers, private, arena = GraphBufferPool("cuda"), GraphBufferPool("cuda"), ScratchArena("cuda")
    layer = weights.draft.layers[index]
    with PreparationSession(device="cuda:0", autotune=False) as session:
        scratch = DraftScratch.allocate(weights.config.draft, weights.shard, 8, "cuda", buffers=buffers.cursor())
        attention = DraftAttention(index, layer, source, weights.config.draft, weights.shard, scratch, batch,
            SimpleNamespace(reserved_region=history.__getitem__), runtime=None, coordinator=None,
            session=session, name=f"mxfp8.attention{index}", workspace=arena, buffers=private.cursor())
        mlp = PreparedMLP(source, layer.mlp, scratch.mlp.gate_up, scratch.mlp.intermediate, scratch.mlp.output,
            session=session, name=f"mxfp8.mlp{index}", workspace=arena)
        arena.finalize()
        def body():
            attention(source)
            mlp(source)
        forward = compile_forward(body)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        for start in (17, 1, 23):
            set_metadata(batch, [start], [8], [1])
            source.normal_()
            scratch.output.fill_(float("nan"))
            scratch.mlp.output.fill_(float("nan"))
            replay(graph)
            qkv = projection_reference(attention.qkv, source)
            check(scratch.qkv, qkv)
            q, k, v = qkv.split((4096, 1024, 1024), -1)
            q = rotary(rms(q.reshape(8, 32, 128), layer.query_norm), batch.positions)
            k = rotary(rms(k.reshape(8, 8, 128), layer.key_norm), batch.positions)
            check(scratch.query, q)
            check(scratch.key, k)
            check(scratch.value, v.reshape(8, 8, 128))
            key = torch.cat((history[f"draft{index}.key"][1, :start].bfloat16(),
                scratch.key.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16())).repeat_interleave(4, 1)
            value = torch.cat((history[f"draft{index}.value"][1, :start].bfloat16(),
                scratch.value.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16())).repeat_interleave(4, 1)
            scores = torch.einsum("qhd,khd->hqk", scratch.query.float(), key.float()) * 128 ** -.5
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float()).bfloat16()
            check(scratch.attended, attended)
            check(scratch.output, projection_reference(attention.output, scratch.attended.flatten(1)))
            check(scratch.mlp.gate_up, projection_reference(mlp.up_projection, source))
            gate, up = scratch.mlp.gate_up.chunk(2, -1)
            check(scratch.mlp.intermediate, F.silu(gate) * up)
            check(scratch.mlp.output, projection_reference(mlp.down_projection, scratch.mlp.intermediate))
            for name, tensor in history.items():
                assert torch.equal(tensor.view(torch.uint8), saved[name]), name
        graph.reset()
