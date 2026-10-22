"""Prepared Gemma 4 decoder layers, PLE, MoE, and softcapped vocabulary projection."""

from dataclasses import dataclass
import math
import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.paged_attention import PagedAttentionMetadata
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import PreparedAttention, attention_descales, prepare_projection, rms_norm
from .config import PAGE_TOKENS
from .moe import PreparedMoE


def softcap_logits(logits: torch.Tensor, softcap: float | None) -> None:
    if softcap is not None:
        logits.copy_((torch.tanh(logits.float() / softcap) * softcap).to(logits.dtype))


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(
        cls,
        tokens,
        rows,
        config,
        buffers,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
    ):
        def zeros(shape):
            return buffers.zeros(shape, torch.bfloat16)

        local_vocab = config.vocabulary_partition(shard)[2]
        return cls(
            activations=zeros((tokens, config.hidden)),
            normalized=zeros((tokens, config.hidden)),
            hidden=zeros((rows, config.hidden)),
            local_logits=zeros((rows, local_vocab)),
            logits=zeros((rows, config.vocabulary)),
        )

    def rows(self, tokens, rows):
        return Scratch(
            activations=self.activations[:tokens],
            normalized=self.normalized[:tokens],
            hidden=self.hidden[:rows],
            local_logits=self.local_logits[:rows],
            logits=self.logits[:rows],
        )


class PreparedLayer:
    def __init__(
        self,
        index: int,
        weights: dict,
        config,
        source: torch.Tensor,
        metadata,
        pool,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
        runtime=None,
        coordinator=None,
        mode: str,
        session,
        name: str,
        workspace,
        buffers,
        cache_metadata,
        unit_scale: torch.Tensor,
        k_descale: torch.Tensor,
    ):
        self.index = index
        self.config = config
        self.eps = config.rms_norm_eps
        self.is_sliding = config.is_sliding(index)
        self.input_layernorm = weights["input_layernorm"]
        self.post_attention_layernorm = weights["post_attention_layernorm"]
        self.pre_feedforward_layernorm = weights["pre_feedforward_layernorm"]
        self.post_feedforward_layernorm = weights["post_feedforward_layernorm"]
        self.layer_scalar = weights["layer_scalar"]

        tokens = source.shape[0]
        start, end = config.intermediate_range(index, shard)
        inter = end - start

        def zeros(shape, dtype=torch.bfloat16):
            return buffers.zeros(shape, dtype)

        self.normed = source
        self.self_attn = PreparedAttention(
            index,
            weights,
            config,
            self.normed,
            metadata,
            pool,
            shard=shard,
            runtime=runtime,
            coordinator=coordinator,
            mode=mode,
            session=session,
            name=name + ".self_attn",
            workspace=workspace,
            buffers=buffers,
            cache_metadata=cache_metadata,
            unit_scale=unit_scale,
            k_descale=k_descale,
        )

        self.gate_out = zeros((tokens, inter))
        self.up_out = zeros((tokens, inter))
        self.mlp_act = zeros((tokens, inter))
        self.mlp_partial = zeros((tokens, config.hidden))
        self.mlp_out = zeros((tokens, config.hidden))
        self.gate_proj = prepare_projection(
            self.normed, weights["gate_proj"], self.gate_out, session=session, name=name + ".gate_proj", workspace=workspace
        )
        self.up_proj = prepare_projection(
            self.normed, weights["up_proj"], self.up_out, session=session, name=name + ".up_proj", workspace=workspace
        )
        self.down_proj = prepare_projection(
            self.mlp_act,
            weights["down_proj"],
            self.mlp_partial,
            session=session,
            name=name + ".down_proj",
            workspace=workspace,
        )
        self.mlp_reduce = PreparedAllReduce(
            self.mlp_partial,
            self.mlp_out,
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".mlp_reduce",
        )

        self.moe = None
        if config.enable_moe:
            self.post_feedforward_layernorm_1 = weights["post_feedforward_layernorm_1"]
            self.moe = PreparedMoE(
                weights,
                config,
                self.normed,
                metadata.live_counts[:1],
                shard=shard,
                runtime=runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".moe",
                workspace=workspace,
                buffers=buffers,
            )

        self.ple_gate = None
        if config.ple_dim > 0:
            self.post_per_layer_input_norm = weights["post_per_layer_input_norm"]
            self.ple_gate_out = zeros((tokens, config.ple_dim))
            self.ple_act = zeros((tokens, config.ple_dim))
            self.ple_out = zeros((tokens, config.hidden))
            self.ple_gate = prepare_projection(
                self.normed,
                weights["per_layer_input_gate"],
                self.ple_gate_out,
                session=session,
                name=name + ".ple_gate",
                workspace=workspace,
            )
            self.ple_proj = prepare_projection(
                self.ple_act,
                weights["per_layer_projection"],
                self.ple_out,
                session=session,
                name=name + ".ple_proj",
                workspace=workspace,
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ple_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        self.normed.copy_(rms_norm(residual, self.input_layernorm, self.eps))
        attn_out = self.self_attn(self.normed, cos, sin)
        hidden_states = residual + rms_norm(attn_out, self.post_attention_layernorm, self.eps)

        residual = hidden_states
        self.normed.copy_(rms_norm(residual, self.pre_feedforward_layernorm, self.eps))
        self.gate_proj(self.normed)
        self.up_proj(self.normed)
        self.mlp_act.copy_(torch.nn.functional.gelu(self.gate_out, approximate="tanh") * self.up_out)
        mlp_out = self.mlp_reduce(self.down_proj(self.mlp_act))
        if self.moe is not None:
            h_1 = rms_norm(mlp_out, self.post_feedforward_layernorm_1, self.eps)
            h_2 = self.moe(residual)
            mlp_out = h_1 + h_2
        hidden_states = residual + rms_norm(mlp_out, self.post_feedforward_layernorm, self.eps)

        if self.ple_gate is not None and ple_input is not None:
            residual = hidden_states
            self.normed.copy_(residual)
            self.ple_gate(self.normed)
            self.ple_act.copy_(torch.nn.functional.gelu(self.ple_gate_out, approximate="tanh") * ple_input)
            ple_out = self.ple_proj(self.ple_act)
            hidden_states = residual + rms_norm(ple_out, self.post_per_layer_input_norm, self.eps)

        return hidden_states * self.layer_scalar.to(hidden_states.dtype)


class SoftcappedResume(PreparedResume):
    def __init__(self, *args, softcap: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.softcap = softcap

    def __call__(self):
        self.validation(self.state.validate())
        self.projection(self.state.gather())
        softcap_logits(self.logits, self.softcap)
        self.completion()


class PreparedDecoder:
    def __init__(
        self,
        weights,
        scratch: Scratch,
        metadata,
        pool,
        *,
        mode: str,
        session,
        name: str,
        workspace,
        layer_buffers,
        runtime=None,
        coordinator=None,
        vocabulary_runtime=None,
        buffers=None,
        scoring=None,
    ):
        self.weights = weights
        self.scratch = scratch
        self.metadata = metadata
        self.pool = pool
        self.auxiliary = None
        config = weights.config
        self.config = config
        self.eps = config.rms_norm_eps
        self.softcap = config.final_logit_softcapping
        device = scratch.activations.device
        tokens = scratch.activations.shape[0]

        self.cache_metadata = PagedAttentionMetadata(metadata, page_tokens=PAGE_TOKENS, buffers=buffers)
        self.embedding = PreparedEmbedding(
            weights.embedding,
            metadata.token_ids,
            scratch.activations,
            num_rows=metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        )
        self.embed_scale = torch.tensor(config.hidden**0.5, dtype=torch.bfloat16, device=device)

        self.ple_embedding = None
        if weights.ple is not None:
            total_ple = config.num_layers * config.ple_dim
            self.ple_token_ids = buffers.zeros(metadata.token_ids.shape, metadata.token_ids.dtype)
            self.ple_tok_raw = buffers.zeros((tokens, total_ple), torch.bfloat16)
            self.ple_proj_raw = buffers.zeros((tokens, total_ple), torch.bfloat16)
            self.ple_embedding = PreparedEmbedding(
                weights.ple.embedding,
                self.ple_token_ids,
                self.ple_tok_raw,
                num_rows=metadata.live_counts[:1],
                session=session,
                name=name + ".ple_embedding",
            )
            self.ple_model_proj = prepare_projection(
                scratch.activations,
                weights.ple.projection,
                self.ple_proj_raw,
                session=session,
                name=name + ".ple_model_proj",
                workspace=workspace,
            )
            self.ple_embed_scale = torch.tensor(config.ple_dim**0.5, dtype=torch.bfloat16, device=device)
            self.ple_input_scale = torch.tensor(config.hidden**-0.5, dtype=torch.bfloat16, device=device)
            self.ple_combine_scale = torch.tensor(math.sqrt(2.0) ** -1, dtype=torch.bfloat16, device=device)

        self.sliding_inv_freq = config.rotary_inverse("sliding_attention", device)
        self.full_inv_freq = config.rotary_inverse("full_attention", device)
        unit_scale, sliding_descale, full_descale = attention_descales(config, device)

        self.layers = []
        for index, layer_weights in enumerate(weights.layers):
            cursor = layer_buffers.cursor()
            descale = sliding_descale if config.is_sliding(index) else full_descale
            self.layers.append(
                PreparedLayer(
                    index,
                    layer_weights,
                    config,
                    scratch.normalized,
                    metadata,
                    pool,
                    shard=weights.shard,
                    runtime=runtime,
                    coordinator=coordinator,
                    mode=mode,
                    session=session,
                    name=f"{name}.layer{index}",
                    workspace=workspace,
                    buffers=cursor,
                    cache_metadata=self.cache_metadata,
                    unit_scale=unit_scale,
                    k_descale=descale,
                )
            )

        self.scoring = (
            None
            if scoring is None
            else scoring.prepare(
                scratch.normalized,
                weights.head,
                vocabulary=config.vocabulary,
                session=session,
                coordinator=coordinator,
                name=name + ".prompt",
            )
        )
        self.head = PreparedVocabularyProjection(
            scratch.hidden,
            weights.head,
            scratch.local_logits,
            scratch.logits,
            runtime=vocabulary_runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".head",
            activation_mode="auto",
            buffers=buffers,
        )

    def __call__(self):
        m = self.metadata
        self.cache_metadata()
        tokens = self.scratch.activations.shape[0]
        pos = m.positions.clamp_min(0).float()[:, None]

        s_freq = pos * self.sliding_inv_freq[None, :]
        s_emb = torch.cat((s_freq, s_freq), dim=-1)
        sliding_cos, sliding_sin = s_emb.cos().bfloat16(), s_emb.sin().bfloat16()

        f_freq = pos * self.full_inv_freq[None, :]
        f_emb = torch.cat((f_freq, f_freq), dim=-1)
        full_cos, full_sin = f_emb.cos().bfloat16(), f_emb.sin().bfloat16()

        hidden = self.embedding() * self.embed_scale
        self.scratch.activations.copy_(hidden)

        ple_inputs = None
        if self.ple_embedding is not None:
            valid_ple = (m.token_ids >= 0) & (m.token_ids < self.config.ple_vocabulary)
            self.ple_token_ids.copy_(torch.where(valid_ple, m.token_ids, 0))
            ple_tok = (self.ple_embedding() * self.ple_embed_scale).view(
                tokens, self.config.num_layers, self.config.ple_dim
            )
            ple_proj = (self.ple_model_proj(self.scratch.activations) * self.ple_input_scale).view(
                tokens, self.config.num_layers, self.config.ple_dim
            )
            ple_proj = rms_norm(ple_proj, self.weights.ple.norm, self.eps)
            ple_inputs = (ple_proj + ple_tok) * self.ple_combine_scale

        for index, layer in enumerate(self.layers):
            cos, sin = (sliding_cos, sliding_sin) if layer.is_sliding else (full_cos, full_sin)
            ple_layer = None if ple_inputs is None else ple_inputs[:, index, :]
            hidden = layer(hidden, cos, sin, ple_layer)

        normalized = rms_norm(hidden, self.weights.norm, self.eps)
        self.scratch.normalized.copy_(normalized)
        if self.scoring is not None:
            for source, projection, reduction in self.scoring.parts:
                logits = projection(source)
                softcap_logits(logits, self.softcap)
                reduction()
        self.scratch.hidden.copy_(self.scratch.normalized[m.sample_indices])
        self.head(self.scratch.hidden)
        softcap_logits(self.scratch.logits, self.softcap)
