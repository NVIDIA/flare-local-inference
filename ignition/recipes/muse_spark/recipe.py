"""Muse Spark Open text preparation, DFlash2 drafts, and FP8 attention caches."""

from pathlib import Path
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Muse Spark preparation requires eight GPUs on one host")
    drafting = {}
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        drafting = dict(draft_snapshot=Path(snapshot_download(REPOSITORY, revision=REVISION,
            cache_dir=plan.hub_cache, allow_patterns=("config.json", "model.safetensors"))),
            speculative_widths=plan.speculation.widths,
            speculative_fixed_width=plan.speculation.fixed_width,
            speculative_token_capacity=plan.speculative_token_capacity,
            speculative_graph_budget=plan.speculation.graph_budget or 64)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)), context=plan.context, max_active=plan.concurrency,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache,
        dflash=plan.speculation is not None, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"])
                 for rank in prepared.memory_report.get("ranks", (prepared.memory_report,)))


RECIPE = Recipe(name="muse_spark", served_model_name="Muse-Spark-Open-816B-A42B",
    # Recommended sampling: https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
    prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(alias="muse-spark-open-nvfp4", repository="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
        revision="4239e4db19376904f2e82c167e78f70abb5455fe", architecture="MuseSparkForConditionalGeneration",
        stored_bytes=533_010_718_720, device_bytes=545_401_038_848,
        draft_bytes=12_768_520_192, expansion_bytes=1 << 30),),
    context=262144, page_tokens=128, concurrency=4, max_concurrency=4, max_draft_tokens=7, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=32, packed_verification=True, graph_budget=64),
    verification_rows=0,
    # Five FP8 draft key/value rings: 2048 positions, two local KV heads, 64 elements per head.
    ranks=MappingProxyType({8: RankCosts(27_525_120, 524_288, 8 << 30)}),
    plain_ranks=MappingProxyType({8: RankCosts(24_903_680, 524_288, 8 << 30)}))
