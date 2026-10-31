# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H Bolt preparation capacities and serving defaults."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Nemotron-H preparation requires one local GPU")
    drafting = {}
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        draft = snapshot_download(REPOSITORY, revision=REVISION, cache_dir=plan.hub_cache,
                                  allow_patterns=("config.json", "model.safetensors"))
        drafting = dict(draft_checkpoint=draft, speculative_widths=plan.speculation.widths,
            speculative_fixed_width=plan.speculation.fixed_width, speculative_token_capacity=plan.speculative_token_capacity,
            speculative_graph_budget=plan.speculation.graph_budget or 64)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, context=plan.context,
        stream_overlap=plan.stream_overlap,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, **drafting)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(
    name="nemotron_h", prepare=prepare, usage=usage,
    served_model_name="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    # Text sampling: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1, min_p=0.,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.),
    checkpoints=(Checkpoint(alias="nemotron-bolt-24gb", repository="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
        revision="c6cc2e8ffa353403213d6b9b18aaf8eb684d705a", architecture="NemotronH_Omni_Reasoning_V3",
        stored_bytes=24_079_678_824, device_bytes=25_063_117_344, draft_bytes=983_438_520),),
    context=262144, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, verification_rows=0,
    max_images=4, image_bytes=32 << 20,
    stream_overlap=True,
    ranks=MappingProxyType({1: RankCosts(107_365_120, 524_288, 4 << 30)}),
    plain_ranks=MappingProxyType({1: RankCosts(86_876_416, 524_288, 4 << 30)}),
)
