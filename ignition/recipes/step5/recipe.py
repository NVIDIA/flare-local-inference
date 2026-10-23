"""Step-5 Preview NVFP4 text, causal MTP, and perception-encoder preparation."""

from types import MappingProxyType

from ignition.recipes.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Step-5 preparation requires GPUs on one host")
    speculation = plan.speculation
    config = PreparationConfig(prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)), context=plan.context,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, mtp=speculation is not None,
        speculative_widths=(1,) if speculation is None else speculation.widths,
        speculative_fixed_width=None if speculation is None else speculation.fixed_width,
        speculative_token_capacity=None if speculation is None else plan.speculative_token_capacity,
        stream_overlap=plan.stream_overlap, expert_scale_compression=plan.expert_scale_compression)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(name="step5", served_model_name="Step-5-Preview", prepare=prepare, usage=usage,
    # Recommended sampling: https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4#evaluation-notes.
    # The card's top_k=0 and Flare's top_k=-1 both disable the filter.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(Checkpoint(alias="step5-preview-nvfp4", repository="stepfun-ai/Step-5-Preview-NVFP4",
        revision="bc246cc2848e17b37f377eda8fd825c5b3018d1f", architecture="Step5ForConditionalGeneration",
        # W4A16 experts and MXFP8 gate/up include maximum scale padding across TP4/8/16.
        stored_bytes=394_164_831_232, device_bytes=367_482_051_584, draft_bytes=4_713_730_560,
        # Planning assumes 6% resident-weight savings; loading measures the encoded allocation.
        scale_compressed_device_bytes={"csf": 345_326_999_491},
        # Replicated MTP projections add up to fifteen copies of three BF16 4096x8192 matrices.
        expansion_bytes=(2 << 30) + 3 * (4096 * 8192 * 2) * 15),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=3, token_capacity=128), max_draft_tokens=3, max_images=4, image_bytes=32 << 20,
    verification_rows=0, stream_overlap=True,
    ranks=MappingProxyType({world: RankCosts(15_667_968, 1_224_704, 8 << 30) for world in (4, 8, 16)}),
    plain_ranks=MappingProxyType({world: RankCosts(14_987_520, 1_224_704, 8 << 30) for world in (4, 8, 16)}))
