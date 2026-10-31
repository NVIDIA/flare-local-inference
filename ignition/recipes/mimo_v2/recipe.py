# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""MiMo V2.6 Pro MOPD text, DFlash and image preparation."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, package_storage=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity, "speculative_graph_budget": speculation.graph_budget or 64, "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, max_active=plan.concurrency, image_concurrency=plan.image_concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        dflash=speculation is not None, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="mimo_v2",
    served_model_name="MiMo-V2.6-Pro-MOPD",
    # Recommended sampling: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD#4-deployment.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    prepare=prepare,
    usage=usage,
    checkpoints=(Checkpoint(alias="mimo-v2.6-pro-mopd", repository="XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
        revision="adea8e2c5373181e5a973fa1ecb343cb31af214b", architecture="MiMoV2ForCausalLM",
        stored_bytes=566_027_990_784, device_bytes=568_511_624_448, draft_bytes=5_536_632_576,
        expansion_bytes=14 << 30),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    verification_rows=0,
    ranks=MappingProxyType({world: RankCosts(33_816_576 // world, 3_276_800 // world, 8 << 30)
                            for world in (2, 4, 8)}),
    plain_ranks=MappingProxyType({world: RankCosts(23_199_744 // world, 3_276_800 // world, 8 << 30)
                                  for world in (2, 4, 8)}),
)
