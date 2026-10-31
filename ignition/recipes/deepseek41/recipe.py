# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 Flash checkpoint and tensor-parallel memory geometry."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity,
        "speculative_graph_budget": speculation.graph_budget,
        "speculative_fixed_width": speculation.fixed_width, "packed_verification": speculation.packed_verification,
        "draft_prefix_selection": speculation.draft_prefix_selection,
        "draft_head_quantization": speculation.draft_head_quantization,
        "markov_head_quantization": speculation.markov_head_quantization,
        "vocabulary_parallel_sampling": speculation.vocabulary_parallel_sampling}
    config = PreparationConfig(collective_tuning=plan.collective_tuning,
        collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        joint_tuning=plan.joint_tuning,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        image_concurrency=plan.image_concurrency,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, max_active=plan.concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        dspark=speculation is not None,
        stream_overlap=plan.stream_overlap, expert_scale_compression=plan.expert_scale_compression,
        mla_compute_mode=plan.mla_compute_mode, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="deepseek41", served_model_name="DeepSeek-V4.1-Flash", prepare=prepare, usage=usage,
    mla_compute_modes=("auto", "bf16"), mla_compute_mode="bf16", expert_scale_compression="csf",
    draft_head_quantizations=("nvfp4", "mxfp8"), markov_head_quantizations=("nvfp4", "fp8"),
    # Recommended sampling: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
    checkpoints=(Checkpoint(alias="deepseek-v4.1-flash", repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        revision="1bae0e1088993b9b82de08f59de5be5a34e42170", architecture="DeepseekV41ForCausalLM",
        stored_bytes=510_286_023_000, device_bytes=306_557_453_640, draft_bytes=7_932_874_632,
        # Estimate half of backbone scale bytes saved; retain the full draft budget.
        scale_compressed_device_bytes={"csf": 298_063_988_040}),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True,
                            draft_prefix_selection="gpu", graph_budget=64),
    max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    # Single-row decode runs the shared expert on an auxiliary stream.
    stream_overlap=True,
    ranks=MappingProxyType({world: RankCosts(3_043_840, 113_920, 8 << 30) for world in (3, 4)}),
    plain_ranks=MappingProxyType({world: RankCosts(2_841_088, 113_920, 8 << 30) for world in (3, 4)}),
)
