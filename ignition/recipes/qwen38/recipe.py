# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, serving range, and memory costs of the Qwen 3.8 recipe."""

from types import MappingProxyType

from ignition.preparation.qwen38 import ARCHITECTURE, MODEL_ID, REVISION
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation

# Twelve target sparse-attention layers each store one page; learned drafts add the draft layer's.
_TARGET_PAGE_LAYERS = 12
# Rough estimates of what a rank holds beyond stored weights, state rows, and
# pages: prepared weight layouts, replicated tensors, operator scratch, graphs, and
# the CUDA context. They serve every speculation setting.
_GRAPH_BYTES = {1: 7168 << 20, 2: 8000 << 20, 4: 8576 << 20}


def _costs(state_row_bytes, page_layer_bytes, graph_bytes, *, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    return RankCosts(state_row_bytes, layers * page_layer_bytes, graph_bytes)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, package_storage=None,
            ranks=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    Preparation backs a bounded KV working set; Helios sizes the serving cache.
    """
    from .bootstrap import PreparationConfig, prepare_model, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity,
        "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        shard_hyperconnections=plan.shard_hyperconnections,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain, stream_overlap=plan.stream_overlap,
        context=plan.recipe.context, prefill_chunk_size=plan.prefill_chunk, mixed_prefill=plan.mixed_prefill,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache,
        mtp=speculation is not None, **drafting)
    if plan.world_size == 1:
        storage = {} if package_storage is None else {"package_storage": package_storage}
        return prepare_model(snapshot, config, progress=progress, compile_decoder=compile_forwards,
                             **storage)
    from ignition.preparation.rank_group import prepare_group

    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="qwen38", served_model_name="Qwen3.8-Flash-Next", prepare=prepare, usage=usage,
    # Thinking defaults; bounds span thinking/instruct presets: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=20,
        ranges={"temperature": SamplingRange(.7, 1.), "top_p": SamplingRange(.8, .95),
                "presence_penalty": SamplingRange(0., 2.)}),
    checkpoints=(Checkpoint(
        alias="qwen3.8-flash", repository=MODEL_ID, revision=REVISION, architecture=ARCHITECTURE,
        stored_bytes=102_329_644_144,
        # Decoder, vision, and draft tensors; PLE rows are delivered through compact inputs.
        device_bytes=73_529_505_900,
        # The MTP module's tensors.
        draft_bytes=1_596_720_640,
        # Routed-expert block scales expand at load from 4,156,604,564 CSF bytes to
        # 48 layers x 512 experts x 307,200 bytes.
        expansion_bytes=3_393_142_636),),
    context=262144, page_tokens=64, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    # Rounds adapt among zero through five drafts, up to seven. Verification records
    # corrections and commits accepted tokens into the live state row, so it keeps
    # no extra rows.
    speculation=Speculation(draft_tokens=5, token_capacity=128), max_draft_tokens=7, verification_rows=0,
    # Shared experts, small-batch attention projections and selection, and GDN input projections run
    # on an auxiliary stream.
    stream_overlap=True,
    ranks=MappingProxyType({
        1: _costs(122_014_720, 69_632, _GRAPH_BYTES[1], speculating=True),
        2: _costs(61_246_464, 36_864, _GRAPH_BYTES[2], speculating=True),
        4: _costs(31_091_712, 36_864, _GRAPH_BYTES[4], speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(116_505_344, 69_632, _GRAPH_BYTES[1], speculating=False),
        2: _costs(58_383_104, 36_864, _GRAPH_BYTES[2], speculating=False),
        4: _costs(29_518_592, 36_864, _GRAPH_BYTES[4], speculating=False),
    }),
    max_images=4, image_bytes=32 << 20, mixed_prefill=True,
)
