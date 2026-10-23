"""MiniMax M3 DSpark context projection, noncausal blocks, and Markov logits."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLRMSNorm,
)

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.embedding import PreparedEmbedding, PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

from .config import ModelConfig
from .decoder import projection
from .weights import MiniMaxWeightLoader

PREFIX = "language_model.model.dspark"
FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class DraftConfig:
    text: dict
    quantization: dict
    auxiliary_layers: tuple[int, ...]

    @classmethod
    def load(cls, snapshot, target_config):
        path = Path(snapshot)
        if path.name != "dspark":
            path /= "dspark"
        c = json.loads((path / "config.json").read_text())
        t = dict(c["text_config"])
        if c.get("architectures") != ["DSparkMiniMaxDraftModel"]:
            raise ValueError("draft checkpoint must declare DSparkMiniMaxDraftModel")
        expected = {"hidden_size": target_config.hidden, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 128, "rotary_dim": 64,
            "hidden_act": "swigluoai", "use_gemma_norm": True, "use_qk_norm": True,
            "qk_norm_type": "per_head", "dspark_block_size": 7, "dspark_markov_rank": 256,
            "dspark_markov_head_type": "vanilla", "dspark_aux_gemm_precision": "bf16",
            "dspark_use_dense_ffn": True, "dspark_hybrid_context_fusion": False,
            "num_hidden_layers": 5, "sliding_window": 4096,
            "enable_confidence_head": True, "confidence_head_with_markov": True,
            "tie_word_embeddings": False, "attention_output_gate": False}
        if any(t.get(k) != v for k, v in expected.items()):
            raise ValueError("unsupported MiniMax M3 DSpark geometry or numerical contract")
        if t.get("moe_layer_freq") != [0] * t["num_hidden_layers"]:
            raise ValueError("MiniMax M3 DSpark layers must use dense FFNs")
        if t.get("layer_types") != ["sliding_attention"] * t["num_hidden_layers"]:
            raise ValueError("MiniMax M3 DSpark layers require sliding attention")
        d = t.get("dflash_config", {})
        if t.get("is_causal", False) or d.get("causal", False) or d.get("block_size") != 7:
            raise ValueError("MiniMax M3 DSpark requires a noncausal seven-position block")
        taps = tuple(t["dspark_target_layer_ids"])
        if (taps != target_config.target_layers or len(taps) != 5
                or any(not 0 <= layer < target_config.layers for layer in taps)):
            raise ValueError("DSpark target taps must match the target's retained layer outputs")
        if t["vocab_size"] != target_config.vocabulary:
            raise ValueError("DSpark and target must share their complete vocabulary")
        quantization = dict(c["quantization_config"])
        if (quantization.get("quant_method") != "mxfp8"
                or quantization.get("weight_block_size") != [1, 32]
                or quantization.get("activation_scheme") != "dynamic"
                or not isinstance(quantization.get("ignored_layers"), list)):
            raise ValueError("MiniMax M3 DSpark requires declared dynamic MXFP8 projection storage")
        return cls(t, quantization, taps)

    projection_format = ModelConfig.projection_format
    canonical_config = ModelConfig.canonical_config
    rotary_frequencies = ModelConfig.rotary_frequencies

    @property
    def hidden(self): return self.text["hidden_size"]
    @property
    def layers(self): return self.text["num_hidden_layers"]
    @property
    def heads(self): return self.text["num_attention_heads"]
    @property
    def kv_heads(self): return self.text["num_key_value_heads"]
    @property
    def head_dim(self): return self.text["head_dim"]
    @property
    def window(self): return self.text["sliding_window"]
    @property
    def block_size(self): return self.text["dspark_block_size"]
    @property
    def noise_token(self): return self.text["dspark_noise_token_id"]
    @property
    def markov_rank(self): return self.text["dspark_markov_rank"]

    def validate_shard(self, shard):
        if shard.world_size != 4:
            raise ValueError("MiniMax M3 DSpark requires tensor parallelism of four")
        shard.partition(self.text["dspark_ffn_hidden_size"], alignment=32)


class _DraftLoader(MiniMaxWeightLoader):
    __init__ = DirectWeightLoader.__init__
    _selected_names = DirectWeightLoader._selected_names


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: object
    output: object
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class DraftWeights:
    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: object
    norm: torch.Tensor
    embedding: torch.Tensor
    head: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    confidence_head: torch.Tensor
    confidence_bias: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, manifest, device, *, shard, target_embedding, target_head, config=None,
             progress=None, target_config=None):
        path = Path(manifest.snapshot if hasattr(manifest, "snapshot") else manifest)
        path = path if path.name == "dspark" else path / "dspark"
        manifest = inspect_indexed_checkpoint(path)
        d = config if isinstance(config, DraftConfig) else config.draft if config is not None else DraftConfig.load(path, target_config)
        d.validate_shard(shard)
        hidden, dim = d.hidden, d.head_dim
        qr = tuple(x * dim for x in shard.partition(d.heads))
        kr = tuple(x * dim for x in shard.partition(d.kv_heads))
        intermediate = shard.partition(d.text["dspark_ffn_hidden_size"], alignment=32)
        with _DraftLoader(manifest, device, prefixes=(PREFIX + ".",), progress=progress,
                          progress_label="Load draft weights") as loader:
            loader.config, loader.shard = d, shard
            projection = loader.load_linear(PREFIX + ".fc", rows=shard.partition(hidden, alignment=64))
            if not isinstance(projection, torch.Tensor) or projection.shape[1] != hidden * len(d.auxiliary_layers):
                raise ValueError("DSpark context combination must preserve its BF16 concatenated feature matrix")
            hidden_norm = loader.norm(PREFIX + ".hidden_norm.weight", hidden)
            norm = loader.norm(PREFIX + ".final_norm.weight", hidden)
            layers, context_prefixes, context_ranges = [], [], []
            for index in range(d.layers):
                p = f"{PREFIX}.layers.{index}.decoder_layer"
                a = p + ".self_attn"
                qkv = loader.load_linear_group(tuple(a + "." + role + "_proj" for role in ("q", "k", "v")),
                                               row_ranges=(qr, kr, kr))
                context_prefixes.extend((a + ".k_proj", a + ".v_proj"))
                context_ranges.extend((kr, kr))
                output = loader.load_linear(a + ".o_proj", columns=qr)
                mlp = MLPWeights(loader.load_linear_group((p + ".mlp.gate_proj", p + ".mlp.up_proj"),
                                                        row_ranges=(intermediate, intermediate)),
                                 loader.load_linear(p + ".mlp.down_proj", columns=intermediate))
                layers.append(DraftLayerWeights(qkv, output, loader.norm(a + ".q_norm.weight", dim),
                    loader.norm(a + ".k_norm.weight", dim), loader.norm(p + ".input_layernorm.weight", hidden),
                    loader.norm(p + ".post_attention_layernorm.weight", hidden), mlp))
            context = loader.load_linear_group(tuple(context_prefixes), row_ranges=tuple(context_ranges))
            markov_embedding = loader.load_tensor(PREFIX + ".markov_head.markov_w1.weight")
            source = loader._sources[PREFIX + ".markov_head.markov_w2.weight"]
            width = target_head.shape[0]
            first, last = shard.rank * width, min((shard.rank + 1) * width, d.text["vocab_size"])
            markov_head = torch.zeros((width, d.markov_rank), dtype=torch.bfloat16, device=device)
            loader._copy(markov_head[:last - first], source[first:last])
            confidence = loader.load_tensor(PREFIX + ".confidence_head.proj.weight")
            bias = loader.load_tensor(PREFIX + ".confidence_head.proj.bias")
            loader._session.flush()
            if (markov_embedding.shape != (d.text["vocab_size"], d.markov_rank)
                    or markov_embedding.dtype != torch.bfloat16 or source.dtype != torch.bfloat16
                    or confidence.shape != (1, hidden + d.markov_rank) or confidence.dtype not in (torch.bfloat16, torch.float32)
                    or bias.shape != (1,) or bias.dtype not in (torch.bfloat16, torch.float32)):
                raise ValueError("DSpark requires BF16 Markov weights and a biased confidence projection")
            bias = bias.float()
        return cls(projection, hidden_norm, context, norm, target_embedding, target_head,
                   markov_embedding, markov_head, confidence, bias, tuple(layers))


def continuation_specs(config, shard):
    d = config.draft if hasattr(config, "draft") else config
    for index in range(d.layers):
        for role in ("key", "value"):
            yield f"draft{index}.{role}", (d.window, d.kv_heads // shard.world_size, d.head_dim), FP8


def canonical_norm(weight, eps):
    with torch.device("meta"):
        norm = MiniMaxM3VLRMSNorm(weight.numel(), eps)
    norm.weight = torch.nn.Parameter(weight, requires_grad=False)
    return norm.eval()


def norm_rope(values, weight, positions, inverse, eps, output, *, fp8=False, round_normalized=False):
    x = values.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + weight.float())
    if round_normalized:
        x = x.to(torch.bfloat16).float()
    angle = positions.clamp_min(0).float()[:, None] * inverse[None]
    cos, sin = angle.cos().to(torch.bfloat16).float()[:, None], angle.sin().to(torch.bfloat16).float()[:, None]
    a, b = x[..., :32], x[..., 32:64]
    result = torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., 64:]), -1)
    if fp8:
        result = result.to(torch.bfloat16).clamp(-448, 448).to(FP8).to(torch.bfloat16)
    output.copy_(result)


class PreparedContext:
    """Write only accepted target positions into each draft layer's context ring."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 runtime, coordinator, buffers):
        self.metadata, self.pool = metadata, pool
        d, draft = weights.config.draft, weights.draft
        tokens, hidden = auxiliary.shape[0], weights.config.hidden
        self.window, self.eps = d.window, d.text["rms_norm_eps"]
        self.local = buffers.zeros((tokens, hidden // weights.shard.world_size), torch.bfloat16)
        self.projected = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.normalized = buffers.zeros((tokens, hidden), torch.bfloat16)
        self.kv = buffers.zeros((tokens, d.layers * 2 * (d.kv_heads // weights.shard.world_size) * d.head_dim), torch.bfloat16)
        self.key = buffers.zeros((tokens, d.kv_heads // weights.shard.world_size, d.head_dim), torch.bfloat16)
        self.value = buffers.zeros(self.key.shape, torch.bfloat16)
        self.slots = buffers.zeros((tokens,), torch.int64).fill_(-1)
        self.project = projection(auxiliary.flatten(1), draft.projection, self.local, session=session,
                                  name=name + ".combine", workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".combine_gather")
        self.hidden_norm = canonical_norm(draft.hidden_norm, self.eps)
        self.project_kv = projection(self.normalized, draft.context_kv, self.kv, session=session,
                                     name=name + ".all_layer_kv", workspace=workspace)
        self.inverse = d.rotary_frequencies(auxiliary.device)
        self.stages = tuple((layer.key_norm, PreparedPagedKVWriter(self.key, self.value,
            pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value"),
            self.slots, session=session, name=f"{name}.layer{index}.write")) for index, layer in enumerate(draft.layers))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1))
        self.gather(self.project(auxiliary.flatten(1)))
        self.normalized.copy_(self.hidden_norm(self.projected))
        self.project_kv(self.normalized)
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), 2, *self.key.shape[1:])
        for index, (weight, writer) in enumerate(self.stages):
            norm_rope(chunks[:, index, 0], weight, m.positions, self.inverse, self.eps, self.key,
                      round_normalized=True)
            self.value.copy_(chunks[:, index, 1])
            writer()

    def scratch_tensors(self):
        return (self.local, self.projected, self.normalized, self.kv, self.key, self.value, self.slots)


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device, *, buffers):
        d, hidden = config.draft, config.hidden
        if samples != tokens or tokens % d.block_size:
            raise ValueError("DSpark scratch requires one sample per complete block position")
        heads, kv = d.heads // shard.world_size, d.kv_heads // shard.world_size
        def z(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)
        pages = tokens // d.block_size * (-(-(d.window + d.block_size) // 128))
        return cls(z((tokens, hidden // shard.world_size)), z((tokens, hidden)),
            z((tokens, (heads + 2 * kv) * d.head_dim)), z((tokens, heads, d.head_dim)),
            z((tokens, kv, d.head_dim)), z((tokens, kv, d.head_dim)), z((tokens, heads, d.head_dim)),
            z((tokens, hidden)), z((tokens, hidden)), z((pages, 128, kv, d.head_dim), FP8),
            z((pages, 128, kv, d.head_dim), FP8), MLPScratch.allocate(tokens, hidden,
                d.text["dspark_ffn_hidden_size"] // shard.world_size, device, buffers=buffers),
            z((tokens, hidden)), z((tokens, config.vocabulary_partition(shard)[2])), z((tokens, config.vocabulary)))

    def rows(self, tokens, samples):
        if samples != tokens or not 0 < tokens <= self.embedding.shape[0]:
            raise ValueError("DSpark requires matched draft token and sample rows")
        return type(self)(*(getattr(self, name) if name.startswith("window_") else
            self.mlp.rows(tokens) if name == "mlp" else getattr(self, name)[:tokens]
            for name in self.__dataclass_fields__))

    def scratch_tensors(self):
        return (self,)


def stage_window(history, values, output, metadata, window):
    rows = metadata.lengths.numel()
    capacity = output.shape[0] * output.shape[1] // rows
    offsets = torch.arange(capacity, device=values.device)[None]
    prefix = metadata.starts.clamp(min=0, max=window)[:, None]
    positions = metadata.starts[:, None] - prefix + offsets
    previous = history[metadata.state_rows.clamp_min(0)[:, None], positions.remainder(window)]
    indices = (metadata.query_start_loc[:-1, None] + offsets - prefix).clamp(0, values.shape[0] - 1)
    current = values[indices]
    valid = (metadata.lengths[:, None] > 0) & (offsets < prefix + metadata.lengths[:, None])
    result = torch.where(offsets[:, :, None, None] < prefix[:, :, None, None], previous.float(), current.float())
    output.view(rows, capacity, *values.shape[1:]).copy_(torch.where(valid[:, :, None, None], result, 0).clamp(-448, 448).to(FP8))


class _Attention(torch.nn.Module):
    def __init__(self, index, weights, config, shard, scratch, metadata, pool, *, block,
                 runtime, coordinator, session, name, workspace, buffers):
        super().__init__()
        self.s, self.m, self.d = scratch, metadata, config.draft
        self.weights, self.eps = weights, self.d.text["rms_norm_eps"]
        self.input = projection(scratch.embedding, weights.qkv, scratch.qkv, session=session, name=name + ".qkv", workspace=workspace)
        self.inverse = self.d.rotary_frequencies(scratch.embedding.device)
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{role}") for role in ("key", "value"))
        rows, pages = metadata.lengths.numel(), -(-(self.d.window + block) // 128)
        self.key, self.value = scratch.window_key[:rows * pages], scratch.window_value[:rows * pages]
        self.pages = torch.arange(rows * pages, dtype=torch.int32,
                                  device=scratch.embedding.device).view(rows, pages)
        self.lengths = buffers.zeros(metadata.lengths.shape, torch.int32)
        self.attention = PreparedPagedAttention(scratch.query, self.key, self.value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, mode="verify", width=block, window=self.d.window - 1,
            causal=False, session=session, workspace=workspace, name=name + ".attention")
        self.output = projection(scratch.attended.flatten(1), weights.output, scratch.partial,
                                 session=session, name=name + ".output", workspace=workspace)
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce")

    def forward(self, hidden_states, **kwargs):
        s, m = self.s, self.m
        q, k, v = self.input(hidden_states).split((s.query.shape[1] * self.d.head_dim, s.key[0].numel(), s.value[0].numel()), -1)
        norm_rope(q.view_as(s.query), self.weights.query_norm, m.positions, self.inverse, self.eps, s.query, fp8=True)
        norm_rope(k.view_as(s.key), self.weights.key_norm, m.positions, self.inverse, self.eps, s.key)
        s.value.copy_(v.view_as(s.value))
        stage_window(self.key_cache, s.key, self.key, m, self.d.window)
        stage_window(self.value_cache, s.value, self.value, m, self.d.window)
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts.clamp(max=self.d.window).int() + m.lengths, 0))
        self.attention()
        return self.reduce(self.output(s.attended.flatten(1))), None


class _MLP(torch.nn.Module):
    def __init__(self, weights, config, scratch, *, runtime, coordinator, session, name, workspace):
        super().__init__()
        s, c = scratch.mlp, config.draft.text
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
            session=session, name=name, workspace=workspace, swiglu_limit=c["swiglu_limit"],
            swiglu_alpha=c["swiglu_alpha"], swiglu_beta=c.get("swiglu_beta", 1.), round_silu=False,
            activation_mode="auto")
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce")

    def forward(self, hidden_states):
        return self.reduce(self.mlp(hidden_states))


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, block, runtime, vocabulary_runtime,
                 embedding_runtime, coordinator, session, name, workspace, buffers):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        d, draft = weights.config.draft, weights.draft
        if block != d.block_size:
            raise ValueError("MiniMax M3 DSpark requires all seven query positions")
        self.embedding = PreparedShardedEmbedding(draft.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator, session=session,
            name=name + ".embedding", num_rows=metadata.live_counts[:1])
        self.layers = []
        for index, layer_weights in enumerate(draft.layers):
            layer = MiniMaxM3VLDecoderLayer.__new__(MiniMaxM3VLDecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.self_attn = _Attention(index, layer_weights, weights.config, weights.shard, scratch,
                metadata, pool, block=block, runtime=runtime, coordinator=coordinator, session=session,
                name=f"{name}.layer{index}.attention", workspace=workspace, buffers=buffers)
            layer.mlp = _MLP(layer_weights.mlp, weights.config, scratch, runtime=runtime, coordinator=coordinator,
                session=session, name=f"{name}.layer{index}.mlp", workspace=workspace)
            layer.input_layernorm = canonical_norm(layer_weights.input_norm, d.text["rms_norm_eps"])
            layer.post_attention_layernorm = canonical_norm(layer_weights.post_attention_norm, d.text["rms_norm_eps"])
            self.layers.append(layer.eval())
        self.norm = canonical_norm(draft.norm, d.text["rms_norm_eps"])
        self.head_hidden = scratch.hidden
        self.selector = round_combined_logits
        self.head = PreparedVocabularyProjection(scratch.hidden, draft.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", buffers=buffers)

    def __call__(self):
        hidden = self.embedding()
        for layer in self.layers:
            hidden = layer(hidden)
        self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        self.head(self.scratch.hidden)


def round_combined_logits(logits, preceding_tokens, position):
    logits.copy_(logits.to(torch.bfloat16))


class PreparedMarkov:
    """BF16 Markov projections and the reference's BF16 vocabulary bias."""

    def __init__(self, weights, tokens, *, runtime, coordinator, session, name, buffers):
        rows, d = tokens.numel(), weights.config.draft
        self.embedding = buffers.zeros((rows, d.markov_rank), torch.bfloat16)
        self.local = buffers.zeros((rows, weights.draft.markov_head.shape[0]), torch.bfloat16)
        padded = self.local.shape[1] * weights.shard.world_size
        self.gathered = buffers.zeros((rows, padded), torch.bfloat16)
        self.logits = buffers.zeros((rows, weights.config.vocabulary), torch.bfloat16)
        self.embed = PreparedEmbedding(weights.draft.markov_embedding, tokens, self.embedding,
                                      session=session, name=name + ".embed")
        self.projection = UnquantizedProjection(self.embedding, weights.draft.markov_head, self.local,
                                                session=session, name=name + ".head")
        self.gather = PreparedAllGather(self.local, self.gathered,
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".gather")

    def __call__(self):
        self.projection(self.embed())
        self.gather(self.local)
        self.logits.copy_(self.gathered[:, :self.logits.shape[1]])
        return self.logits
