"""Gemma 4 text preparation across dense, MoE, and Assistant MTP checkpoints."""

from pathlib import Path

from ignition.recipes.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)

_GRAPH_BYTES = 4 << 30


def _ranks(tp1: tuple[int, int], tp2: tuple[int, int], tp4: tuple[int, int]):
    return {
        1: RankCosts(*tp1, _GRAPH_BYTES),
        2: RankCosts(*tp2, _GRAPH_BYTES),
        4: RankCosts(*tp4, _GRAPH_BYTES),
    }


_E2B_RANKS = _ranks((1_182_976, 1_179_648), (1_182_976, 1_179_648), (1_182_976, 1_179_648))
_E4B_RANKS = _ranks((3_675_392, 3_670_016), (1_840_384, 1_835_008), (1_840_384, 1_835_008))
_12B_RANKS = _ranks((22_028_032, 22_020_096), (11_542_272, 11_534_336), (6_299_392, 6_291_456))
_26B_RANKS = _ranks((14_423_808, 14_417_920), (7_214_848, 7_208_960), (3_938_048, 3_932_160))
_31B_RANKS = _ranks((57_682_688, 57_671_680), (28_846_848, 28_835_840), (14_428_928, 14_417_920))


def _assistant_for(snapshot, checkpoint):
    from .config import ASSISTANT_REPOSITORIES, ModelConfig

    if (Path(snapshot) / "config.json").is_file():
        return ASSISTANT_REPOSITORIES[ModelConfig.load(snapshot).hidden]
    key = f"{checkpoint.alias} {checkpoint.repository}".casefold()
    for tag, hidden in (("31b", 5376), ("12b", 3840), ("e4b", 2560), ("e2b", 1536)):
        if tag in key:
            return ASSISTANT_REPOSITORIES[hidden]
    return ASSISTANT_REPOSITORIES[2816]


def prepare(
    snapshot,
    plan,
    *,
    compile_forwards,
    kernel_cache,
    progress,
    retain_graphs=False,
    package_storage=None,
    ranks=None,
):
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare as prepare_model, prepare_rank
    from .config import DRAFT_DIRECTORY

    drafting = {}
    if plan.speculation is not None:
        assistant = _assistant_for(snapshot, plan.checkpoint)
        drafting = dict(
            speculative_widths=plan.speculation.widths,
            speculative_fixed_width=plan.speculation.fixed_width,
            speculative_token_capacity=plan.speculative_token_capacity,
        )
        if (Path(snapshot) / DRAFT_DIRECTORY).is_dir():
            drafting["draft_checkpoint"] = str(Path(snapshot) / DRAFT_DIRECTORY)
        else:
            from huggingface_hub import snapshot_download

            drafting["draft_checkpoint"] = snapshot_download(
                assistant[0],
                revision=assistant[1],
                cache_dir=plan.hub_cache,
                allow_patterns=("config.json", "model.safetensors"),
            )
    config = PreparationConfig(
        prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers,
        tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        roce_ring=plan.roce_ring,
        max_active=plan.concurrency,
        minimum_pages=plan.minimum_pages,
        context=plan.recipe.context,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes,
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs,
        kernel_cache=kernel_cache,
        **drafting,
    )
    if plan.world_size == 1:
        return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)
    return prepare_group(
        snapshot,
        config,
        prepare_rank,
        progress=progress,
        package_storage=package_storage,
        ranks=ranks,
        checkpoint=plan.checkpoint,
    )


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="gemma4",
    served_model_name="Gemma-4",
    prepare=prepare,
    usage=usage,
    # Gemma 4 model card sampling recommendations: https://huggingface.co/google/gemma-4-26B-A4B-it.
    sampling_defaults=SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
    ),
    checkpoints=(
        Checkpoint(
            alias="gemma-4-e2b",
            repository="google/gemma-4-E2B-it",
            revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=10_246_621_918,
            device_bytes=9_414_698_574,
            draft_bytes=157_559_816,
            ranks=_E2B_RANKS,
            plain_ranks=_E2B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-e4b",
            repository="google/gemma-4-E4B-it",
            revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=15_992_595_884,
            device_bytes=15_085_159_516,
            draft_bytes=159_132_680,
            ranks=_E4B_RANKS,
            plain_ranks=_E4B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-12b",
            repository="google/gemma-4-12B-it",
            revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            architecture="Gemma4UnifiedForConditionalGeneration",
            stored_bytes=23_919_549_408,
            device_bytes=24_660_414_568,
            draft_bytes=845_713_928,
            ranks=_12B_RANKS,
            plain_ranks=_12B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-26b-a4b",
            repository="google/gemma-4-26B-A4B-it",
            revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=51_611_872_412,
            device_bytes=51_305_706_052,
            draft_bytes=839_422_472,
            ranks=_26B_RANKS,
            plain_ranks=_26B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-31b",
            repository="google/gemma-4-31B-it",
            revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=62_546_177_752,
            device_bytes=62_333_727_872,
            draft_bytes=939_037_192,
            ranks=_31B_RANKS,
            plain_ranks=_31B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-e4b-nvfp4",
            repository="unsloth/gemma-4-E4B-it-NVFP4",
            revision="1c363766fcfe575ac17a6d544963201a0d9b24c3",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=8_108_321_140,
            device_bytes=7_311_277_052,
            draft_bytes=159_132_680,
            ranks=_E4B_RANKS,
            plain_ranks=_E4B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-12b-nvfp4",
            repository="unsloth/gemma-4-12b-it-NVFP4",
            revision="b1f649734b34aa5575b03d186abd1b9be3d0d5c4",
            architecture="Gemma4UnifiedForConditionalGeneration",
            stored_bytes=9_304_786_336,
            device_bytes=10_116_093_864,
            draft_bytes=845_713_928,
            ranks=_12B_RANKS,
            plain_ranks=_12B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-26b-a4b-nvfp4",
            repository="nvidia/Gemma-4-26B-A4B-NVFP4",
            revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=18_782_360_732,
            device_bytes=18_476_148_292,
            draft_bytes=839_422_472,
            ranks=_26B_RANKS,
            plain_ranks=_26B_RANKS,
        ),
        Checkpoint(
            alias="gemma-4-31b-nvfp4",
            repository="nvidia/Gemma-4-31B-IT-NVFP4",
            revision="4135a98a9b728a548947683219633b25682223ac",
            architecture="Gemma4ForConditionalGeneration",
            stored_bytes=32_633_255_032,
            device_bytes=32_420_805_152,
            draft_bytes=939_037_192,
            ranks=_31B_RANKS,
            plain_ranks=_31B_RANKS,
        ),
    ),
    context=262144,
    page_tokens=128,
    concurrency=4,
    max_concurrency=64,
    prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5),
    max_draft_tokens=7,
    verification_rows=0,
    ranks=_26B_RANKS,
    plain_ranks=_26B_RANKS,
)
