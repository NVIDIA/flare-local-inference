"""Real-checkpoint TP4 FFN oracle, independent of the prepared MoE implementation.

Run with torchrun on four assigned GPUs. Inputs are checkpoint embedding rows
normalized with the first routed layer's post-attention Gemma parameters.
This compares a decoder component, not complete-engine generations.
"""

import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from b12x.preparation import PreparationSession

from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.recipes.minimax_m3.config import TEXT_PREFIX
from ignition.recipes.minimax_m3.decoder import AttentionScratch, PreparedAttention
from ignition.recipes.minimax_m3.weights import MiniMaxWeightLoader

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or "FLARE_MINIMAX_CHECKPOINT" not in os.environ,
    reason="requires an assigned TP4 group and the local MiniMax checkpoint",
)


class LayerLoader(MiniMaxWeightLoader):
    def _selected_names(self):
        prefix = f"{TEXT_PREFIX}.layers.3."
        return {name for name in DirectWeightLoader._selected_names(self)
                if name.startswith(prefix) or name == TEXT_PREFIX + ".embed_tokens.weight"}


class RawWeights:
    def __init__(self, manifest, stack):
        self.manifest, self.stack, self.files = manifest, stack, {}

    def __call__(self, name, extent=None):
        from safetensors import safe_open

        shard = self.manifest.tensors[name].shard
        if shard not in self.files:
            self.files[shard] = self.stack.enter_context(safe_open(self.manifest.snapshot / shard, framework="pt", device="cpu"))
        source = self.files[shard]
        return source.get_tensor(name) if extent is None else source.get_slice(name)[extent]


def fp4_values(packed):
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                       device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    return lut[codes.long()]


def nvfp4_rows(values):
    from flashinfer.quantization.kernels.nvfp4_quantize import nvfp4_quantize_per_token_cute_dsl

    packed, sf, outer = nvfp4_quantize_per_token_cute_dsl(values, 1. / (448 * 6),
        sf_layout=2, enable_pdl=False, nvfp4_4over6=None)
    normalized = fp4_values(packed) * sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return normalized, outer


def mxfp8_rows(values):
    from flashinfer.quantization import mxfp8_quantize

    packed, sf = mxfp8_quantize(values, is_sf_swizzled_layout=False, enable_pdl=False, backend="cute-dsl")
    sf = sf.reshape(values.shape[0], -1)[:, :values.shape[1] // 32]
    gain = torch.exp2(sf.view(torch.uint8).float() - 127)
    return packed.float() * gain.repeat_interleave(32, -1)


def mxfp8_weight(loader, prefix, *, rows=None, columns=None):
    weight = loader.raw(prefix + ".weight")
    scales = loader.raw(prefix + ".weight_scale_inv")
    first, last = (0, weight.shape[1]) if columns is None else columns
    row_slice = slice(None) if rows is None else slice(*rows)
    values = weight[row_slice, first:last].to(loader.device).float()
    sf = scales[row_slice, first // 32:last // 32].to(loader.device)
    gain = sf.float() if sf.dtype == torch.float8_e8m0fnu else torch.exp2(sf.float() - 127)
    return values * gain.repeat_interleave(32, -1)


def routed_oracle(loader, values, ids, route_weights, shard):
    c = loader.config.text
    first, last = shard.partition(c["intermediate_size"], alignment=32)
    x, outer = nvfp4_rows(values)
    routes = torch.zeros((*ids.shape, c["hidden_size"]), device=values.device)
    for expert in ids.unique().tolist():
        token, branch = torch.where(ids == expert)
        prefix = f"{TEXT_PREFIX}.layers.3.block_sparse_moe.experts.{expert}"

        def matrix(role, down=False):
            p = prefix + "." + role
            source, sf = loader.raw(p + ".weight_packed"), loader.raw(p + ".weight_scale")
            if down:
                packed = source[:, first // 2:last // 2].to(values.device)
                scale = sf[:, first // 16:last // 16].to(values.device).float()
            else:
                packed = source[first:last].to(values.device)
                scale = sf[first:last].to(values.device).float()
            gain = torch.tensor(1. / float(loader.raw(p + ".weight_global_scale").item()),
                                dtype=torch.float32, device=values.device)
            return fp4_values(packed) * scale.repeat_interleave(16, -1), gain

        gate_weight, gate_gain = matrix("w1")
        up_weight, up_gain = matrix("w3")
        assert gate_gain == up_gain
        gain = (outer[token] * gate_gain)[:, None]
        gate = (x[token] @ gate_weight.T) * gain
        up = (x[token] @ up_weight.T) * gain
        gate, up = gate.clamp(max=c["swiglu_limit"]), up.clamp(-c["swiglu_limit"], c["swiglu_limit"])
        activated = (gate * torch.sigmoid(c["swiglu_alpha"] * gate) * (up + 1)).bfloat16()
        h, houter = nvfp4_rows(activated)
        down, down_gain = matrix("w2", down=True)
        result = ((h @ down.T) * (houter * down_gain)[:, None]).bfloat16().float()
        routes[token, branch] = result * route_weights[token, branch, None]
    return routes.sum(1).bfloat16()


def shared_oracle(loader, values, shard, *, quantized=True):
    c = loader.config.text
    interval = shard.partition(c["shared_intermediate_size"], alignment=32)
    p = f"{TEXT_PREFIX}.layers.3.block_sparse_moe.shared_experts"
    gate = mxfp8_weight(loader, p + ".gate_proj", rows=interval)
    up = mxfp8_weight(loader, p + ".up_proj", rows=interval)
    down = mxfp8_weight(loader, p + ".down_proj", columns=interval)
    x = mxfp8_rows(values) if quantized else values.float()
    gate, up = (x @ gate.T).bfloat16().float(), (x @ up.T).bfloat16().float()
    gate, up = gate.clamp(max=c["swiglu_limit"]), up.clamp(-c["swiglu_limit"], c["swiglu_limit"])
    activated = (gate * torch.sigmoid(c["swiglu_alpha"] * gate) * (up + 1)).bfloat16()
    hidden = mxfp8_rows(activated) if quantized else activated.float()
    return (hidden @ down.T).bfloat16()


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    delta = a - b
    return dict(max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt()),
        normalized_rmse=float(delta.norm() / b.norm().clamp_min(1e-30)),
        cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)))


def gemma(values, weight, epsilon):
    x = values.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * (1 + weight.float())).bfloat16()


def attention_products(query, key, value, positions, *, rounded_denominator=False, prescale_query=False):
    dim = query.shape[-1]
    query, key, value = query.float(), key.float(), value.float()
    if prescale_query:
        query = (query * dim ** -.5).bfloat16().float()
    scores = torch.einsum("thd,sd->hts", query, key)
    mask = positions[:, None] >= positions[None, :]
    scores = scores.masked_fill(~mask[None], -torch.inf)
    scale = 1.4426950408889634 * (1. if prescale_query else dim ** -.5)
    exponent = (scores - scores.amax(-1, keepdim=True)) * scale
    probabilities = torch.exp2(exponent)
    rounded = probabilities.bfloat16().float()
    denominator = (rounded if rounded_denominator else probabilities).sum(-1)
    numerator = torch.einsum("hts,sd->thd", rounded, value)
    output = (numerator / denominator.T[..., None]).bfloat16()
    return output, dict(scores=scores, exponent=exponent, probabilities=probabilities,
                        denominator=denominator, numerator=numerator)


def graph_replay_times(operation, session):
    graph = torch.cuda.CUDAGraph()
    with session.capture(), torch.cuda.graph(graph):
        for _ in range(100):
            operation()
    for _ in range(10):
        graph.replay()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(3):
        begin.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 2000)
    graph.reset()
    return samples


def attention_oracle(loader, source, shard, positions, *, input_mode, output_mode):
    c = loader.config.text
    dim, heads = c["head_dim"], c["num_attention_heads"] // 4
    qr = tuple(x * dim for x in shard.partition(c["num_attention_heads"]))
    kr = tuple(x * dim for x in shard.partition(c["num_key_value_heads"]))
    prefix = f"{TEXT_PREFIX}.layers.3.self_attn"
    x = source.float() if input_mode == "a16" else mxfp8_rows(source)
    q, k, v = tuple((x @ mxfp8_weight(loader, prefix + "." + role + "_proj", rows=rows).T).bfloat16()
                    for role, rows in (("q", qr), ("k", kr), ("v", kr)))
    inverse = loader.config.rotary_frequencies(source.device)
    angles = positions.float()[:, None] * inverse[None]
    cosine, sine = angles.cos().bfloat16().float()[:, None], angles.sin().bfloat16().float()[:, None]

    def rotate(values, norm):
        values = values.view(values.shape[0], -1, dim).float()
        weight = loader.raw(prefix + "." + norm + ".weight").to(source.device).float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + c["rms_norm_eps"]) * (1 + weight)
        half = inverse.numel()
        first, second = values[..., :half], values[..., half:2 * half]
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine,
                          values[..., 2 * half:]), -1)

    q = rotate(q, "q_norm").bfloat16().float()
    k = rotate(k, "k_norm").clamp(-448, 448).to(torch.float8_e4m3fn).float()[:, 0]
    v = v.clamp(-448, 448).to(torch.float8_e4m3fn).float().view(-1, dim)
    attended, _ = attention_products(q, k, v, positions)
    attended = attended.reshape(-1, heads * dim)
    x = attended.float() if output_mode == "a16" else mxfp8_rows(attended)
    output = mxfp8_weight(loader, prefix + ".o_proj", columns=qr)
    return (x @ output.T).bfloat16(), dict(query=q.bfloat16(), key=k, value=v, attended=attended)


@pytest.mark.parametrize("attention_mode", [None, "prefill", "verify"],
                         ids=("ffn", "full_layer", "verification_layer"))
@pytest.mark.parametrize("scale_compression", [None, "csf"], ids=("native_scales", "csf_scales"))
@torch.no_grad()
def test_real_checkpoint_first_routed_ffn_tp4(attention_mode, scale_compression):
    assert int(os.environ.get("WORLD_SIZE", "1")) == 4, "launch this oracle with torchrun --nproc-per-node=4"
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", device_id=device)
    shard = TensorParallelShard(rank, 4)
    manifest = inspect_indexed_checkpoint(Path(os.environ["FLARE_MINIMAX_CHECKPOINT"]))
    try:
        with ExitStack() as stack, LayerLoader(manifest, device, shard=shard, expert_scale_compression=scale_compression) as loader:
            loader.raw = RawWeights(manifest, stack)
            layer, c = loader.load_layer(3), loader.config.text
            for tokens in ((8,) if attention_mode == "verify" else (1, 17)):
                embedding = loader.raw(TEXT_PREFIX + ".embed_tokens.weight",
                                       (slice(1234, 1234 + tokens), slice(None))).to(device)
                source = embedding
                x = source.float() * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + c["rms_norm_eps"])
                source = (x * (1 + layer.post_attention_norm.float())).bfloat16()
                logits = torch.empty(tokens, c["num_local_experts"], dtype=torch.float32, device=device)
                router = RouterScratch.allocate(tokens, device, experts=c["num_local_experts"], top_k=c["num_experts_per_tok"])
                routed = torch.empty_like(source)
                scratch = MLPScratch.allocate(tokens, c["hidden_size"], c["shared_intermediate_size"] // 4, device)
                arena = ScratchArena(device, overlap=False)
                with PreparationSession(device=device, autotune=scale_compression is not None, compile_workers=0) as session:
                    attention_measured = None
                    if tokens > 1 and attention_mode is not None:
                        buffers = GraphBufferPool(device).cursor()
                        a = AttentionScratch.allocate(loader.config, shard, tokens, buffers)
                        normalized = gemma(embedding, layer.input_norm, c["rms_norm_eps"])
                        positions = torch.arange(tokens, device=device)
                        metadata = SimpleNamespace(lengths=torch.tensor([tokens], device=device, dtype=torch.int32),
                            starts=torch.zeros(1, device=device, dtype=torch.int32), positions=positions,
                            requests=torch.zeros(tokens, device=device, dtype=torch.int32),
                            query_start_loc=torch.tensor([0, tokens], device=device, dtype=torch.int32),
                            page_table=torch.ones((1, 1), device=device, dtype=torch.int32))
                        cache = {role: torch.zeros((2, 128, 128) if role == "index" else (2, 128, 1, 128),
                            dtype=torch.float8_e4m3fn, device=device) for role in ("key", "value", "index")}
                        pool = SimpleNamespace(cache=lambda index, role: cache[role])
                        attention = PreparedAttention(3, layer, loader.config, shard, normalized, a, metadata,
                            pool, mode=attention_mode, inverse_frequency=loader.config.rotary_frequencies(device),
                            session=session, name="attention", workspace=arena, cache_lengths=metadata.lengths)
                        attention(normalized)
                        input_mode = attention.input.plan.prepared.selection.config.mode
                        output_mode = attention.output.plan.prepared.selection.config.mode
                        expected_attention, diagnostic = attention_oracle(loader, normalized, shard, positions,
                            input_mode=input_mode, output_mode=output_mode)
                        attention_measured = metrics(a.output, expected_attention)
                        attention_measured.update(input_mode=input_mode, output_mode=output_mode)
                        attention_measured["query"] = metrics(a.query, diagnostic["query"])
                        attention_measured["key"] = metrics(cache["key"][1, :tokens, 0], diagnostic["key"])
                        attention_measured["value"] = metrics(cache["value"][1, :tokens, 0], diagnostic["value"])
                        attention_measured["attended"] = metrics(a.attended, diagnostic["attended"])
                        same_inputs = (a.query, cache["key"][1, :tokens, 0], cache["value"][1, :tokens, 0], positions)
                        private_output, private_products = attention_products(*same_inputs)
                        native_output, native_products = attention_products(*same_inputs, rounded_denominator=True)
                        prescaled_output, _ = attention_products(*same_inputs, prescale_query=True)
                        attention_measured["same_inputs_private"] = metrics(a.attended, private_output.flatten(1))
                        attention_measured["same_inputs_rounded_denominator"] = metrics(a.attended, native_output.flatten(1))
                        attention_measured["same_inputs_bf16_q_prescale"] = metrics(a.attended, prescaled_output.flatten(1))
                        attention_measured["rounded_denominator_delta"] = metrics(native_products["denominator"],
                                                                                  private_products["denominator"])
                        attention_measured["attention_plan"] = repr(attention.attention.plan.prepared.selection.config)
                        attention_measured["attention_replay_us"] = graph_replay_times(attention.attention, session)
                        dist.all_reduce(a.output)
                        dist.all_reduce(expected_attention)
                        residual = (embedding.float() + a.output.float()).bfloat16()
                        expected_residual = (embedding.float() + expected_attention.float()).bfloat16()
                        source.copy_(gemma(residual, layer.post_attention_norm, c["rms_norm_eps"]))
                        reference_source = gemma(expected_residual, layer.post_attention_norm, c["rms_norm_eps"])
                    project = UnquantizedProjection(source, layer.router, logits, session=session, name="router")
                    route = PreparedRouter(logits, router, session=session, name="route", score_func="sigmoid",
                        renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], correction_bias=layer.correction)
                    experts = PreparedRoutedExperts(source, layer.experts, router.ids, router.weights, routed,
                        session=session, name="experts", workspace=arena)
                    shared = PreparedMLP(source, layer.mlp, scratch.gate_up, scratch.intermediate, scratch.output,
                        session=session, name="shared", workspace=arena, swiglu_limit=c["swiglu_limit"],
                        swiglu_alpha=c["swiglu_alpha"], swiglu_beta=1., round_silu=False, activation_mode="auto")
                    route(project(source))
                    scores = torch.sigmoid(source.float() @ layer.router.T)
                    expected_ids = (scores + layer.correction).topk(c["num_experts_per_tok"], -1).indices
                    expected_weights = scores.gather(1, expected_ids)
                    expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True) * c["routed_scaling_factor"]
                    torch.testing.assert_close(router.ids.long(), expected_ids, atol=0, rtol=0)
                    torch.testing.assert_close(router.weights, expected_weights, atol=2e-6, rtol=2e-6)
                    experts()
                    shared(source)
                    expected_routed = routed_oracle(loader, source, expected_ids, expected_weights, shard)
                    expected_shared = shared_oracle(loader, source, shard)
                    expected_a16 = shared_oracle(loader, source, shard, quantized=False)
                    measured = {"routed": metrics(routed, expected_routed), "shared": metrics(scratch.output, expected_shared)}
                    if attention_measured is not None:
                        measured["attention"] = attention_measured
                    measured["shared_a16"] = metrics(scratch.output, expected_a16)
                    measured["shared_up_plan"] = repr(shared.up_projection.plan.prepared.selection.config)
                    measured["shared_down_plan"] = repr(shared.down_projection.plan.prepared.selection.config)
                    selected_up = shared.up_projection.plan.prepared.selection.config.mode
                    selected_down = shared.down_projection.plan.prepared.selection.config.mode
                    assert selected_up == selected_down, "the oracle must represent both selected arithmetic modes"
                    same_arithmetic = expected_a16 if selected_up == "a16" else expected_shared
                    gathered = [None] * 4
                    dist.all_gather_object(gathered, measured)
                    actual = (routed.float() + scratch.output.float()).bfloat16()
                    expected = (expected_routed.float() + same_arithmetic.float()).bfloat16()
                    private_expected = (expected_routed.float() + expected_shared.float()).bfloat16()
                    dist.all_reduce(actual)
                    dist.all_reduce(expected)
                    dist.all_reduce(private_expected)
                    if rank == 0:
                        print("TP4 real-checkpoint layer3 FFN", tokens, "ranks", gathered,
                              "reduced_same_arithmetic", metrics(actual, expected),
                              "reduced_private_mxfp8", metrics(actual, private_expected), flush=True)
                    assert measured["routed"]["normalized_rmse"] < .02
                    assert metrics(scratch.output, same_arithmetic)["normalized_rmse"] < .02
                    assert metrics(actual, expected)["normalized_rmse"] < .02
                    if attention_measured is not None:
                        scores = torch.sigmoid(reference_source.float() @ layer.router.T)
                        ids = (scores + layer.correction).topk(c["num_experts_per_tok"], -1).indices
                        unchanged_routes = (ids.sort(-1).values == expected_ids.sort(-1).values).all(-1)
                        route_agreement = unchanged_routes.float().mean()
                        weights = scores.gather(1, ids)
                        weights = weights / weights.sum(-1, keepdim=True) * c["routed_scaling_factor"]
                        reference_routed = routed_oracle(loader, reference_source, ids, weights, shard)
                        reference_shared = shared_oracle(loader, reference_source, shard, quantized=selected_up != "a16")
                        reference_ffn = (reference_routed.float() + reference_shared.float()).bfloat16()
                        dist.all_reduce(reference_ffn)
                        full_actual = (residual.float() + actual.float()).bfloat16()
                        full_expected = (expected_residual.float() + reference_ffn.float()).bfloat16()
                        full_metrics = metrics(full_actual, full_expected)
                        if rank == 0:
                            print("TP4 real-checkpoint layer3 full layer", full_metrics,
                                  "route_set_agreement", float(route_agreement),
                                  "unchanged_routes", metrics(full_actual[unchanged_routes], full_expected[unchanged_routes]),
                                  "ffn_input", metrics(source, reference_source), flush=True)
                        assert attention_measured["normalized_rmse"] < .02
                        assert full_metrics["normalized_rmse"] < .02
    finally:
        dist.destroy_process_group()
