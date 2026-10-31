# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 MXFP4-CSF preparation capacities and serving defaults."""

from pathlib import Path
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank
    from .config import DRAFT_DIRECTORY, DSPARK_REPOSITORY, DSPARK_REVISION

    if ranks is not None:
        raise ValueError("Kimi K3 preparation requires GPUs on one host")
    drafting = {}
    if plan.speculation is not None:
        drafting = dict(speculative_widths=plan.speculation.widths, speculative_fixed_width=plan.speculation.fixed_width,
                        speculative_token_capacity=plan.speculative_token_capacity,
                        speculative_graph_budget=plan.speculation.graph_budget or 64)
        if (Path(snapshot) / DRAFT_DIRECTORY).is_dir():
            drafting["draft_checkpoint"] = str(Path(snapshot) / DRAFT_DIRECTORY)
        else:
            from huggingface_hub import snapshot_download

            drafting["draft_checkpoint"] = snapshot_download(DSPARK_REPOSITORY, revision=DSPARK_REVISION, cache_dir=plan.hub_cache,
                                                             allow_patterns=("config.json", "model.safetensors"))
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)), context=plan.context,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, online_mxfp8=plan.online_mxfp8,
        stream_overlap=plan.stream_overlap, context_parallel=plan.context_parallel,
        vision=not plan.language_model_only, **drafting)
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, package_storage=package_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="kimi3", served_model_name="Kimi-K3", prepare=prepare, usage=usage,
    # Agentic defaults; top-p spans single-step/agentic presets: https://huggingface.co/moonshotai/Kimi-K3#3-evaluation-results.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1,
        ranges={"top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(alias="kimi-k3-mxfp4-csf", repository="/data/models/Kimi-K3-MXFP4-CSF",
        revision="local", architecture="KimiK3ForConditionalGeneration",
        stored_bytes=1_491_026_420_236, device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000),
        # The native checkpoint's block scales are compressed while loading, as resident as the CSF copy's.
        Checkpoint(alias="kimi-k3", repository="moonshotai/Kimi-K3",
        revision="2496450e92e425c886db095102a52a6682ca3970", architecture="KimiK3ForConditionalGeneration",
        stored_bytes=1_560_860_324_864, device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000),
        # A local target checkpoint carrying Red Hat's DSpark weights in dspark/.
        Checkpoint(alias="kimi-k3-dspark", repository="/data/kimi-k3-dspark", revision="local",
        architecture="KimiK3ForConditionalGeneration", stored_bytes=1_500_516_228_062,
        device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000)),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    verification_rows=0,
    # KDA, MLA except batched key/value, shared experts, latent MoE projections, and the DSpark
    # draft's context projection, attention outputs and MLPs may load as MXFP8.
    online_mxfp8=True,
    # Decode and verification run each MoE layer's shared experts beside its routing and routed experts.
    stream_overlap=True,
    # Text-only plans, as vLLM serves Kimi K3 with --language-model-only, leave the MoonViT encoder unloaded.
    language_model_only=True,
    # vLLM's Kimi K3 launch settings for b12x's PCIe all-reduce (island reduce-scatter, deferred slab
    # consumption, 256 threads), its decode context exchange and its small-M W4A16 experts.
    b12x_environment=MappingProxyType({"B12X_PCIE_ALLREDUCE_ALGORITHM": "island_rs",
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION": "1", "B12X_PCIE_HIERARCHICAL_THREADS": "256",
        "B12X_PCIE_DCP_THREADS": "512", "B12X_PCIE_DCP_BLOCK_LIMIT": "8", "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET": "0"}),
    # Target MLA pages are context-parallel; DSpark keeps its sliding KV windows in continuation rows.
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=34_716_928, page_bytes=1_769_472,
                                        graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=29_869_824, page_bytes=1_769_472, graph_bytes=9 << 29,
                                              page_tokens=128 * 16)}),
)
