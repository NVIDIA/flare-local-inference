"""MiniMax M3 mixed MXFP8/NVFP4 preparation on four SM120/SM121 GPUs."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else dict(speculative_widths=speculation.widths,
        speculative_fixed_width=speculation.fixed_width, speculative_token_capacity=plan.speculative_token_capacity,
        speculative_graph_budget=speculation.graph_budget, packed_verification=speculation.packed_verification,
        draft_prefix_selection=speculation.draft_prefix_selection)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        image_concurrency=plan.image_concurrency if not plan.language_model_only else None,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, roce_ring=plan.roce_ring,
        context=plan.context, max_active=plan.concurrency,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, stream_overlap=plan.stream_overlap,
        expert_scale_compression=plan.expert_scale_compression, dspark=speculation is not None,
        vision=not plan.language_model_only, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(name="minimax_m3", served_model_name="MiniMax-M3.1-Flash", prepare=prepare, usage=usage,
    # Recommended sampling: https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash#usage.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    checkpoints=(Checkpoint(alias="minimax-m3-1-flash", repository="MiniMaxAI/MiniMax-M3.1-Flash",
        revision="9e7ebacac350497ad4a1272a2e817639bf11561c", architecture="MiniMaxM3SparseForConditionalGeneration",
        stored_bytes=252_405_263_874, device_bytes=252_405_263_874, draft_bytes=2_301_852_674,
        expansion_bytes=2 << 30,
        scale_compressed_device_bytes={"csf": 252_405_263_874}),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64,
    max_draft_tokens=7, prefill_chunk=4096, verification_rows=0, stream_overlap=True,
    speculation=Speculation(draft_tokens=5, token_capacity=128, packed_verification=True,
                            draft_prefix_selection="gpu", graph_budget=64), max_images=4, image_bytes=32 << 20,
    language_model_only=True,
    ranks=MappingProxyType({4: RankCosts(8_265_984, 2_949_120, 8 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(3_023_104, 2_949_120, 8 << 30)}))
