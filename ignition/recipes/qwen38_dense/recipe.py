"""Single-GPU Qwen3.8-27B NVFP4 text preparation without host-mapped storage."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None or plan.speculation is not None:
        raise ValueError("Qwen dense preparation supports one local GPU and ordinary text generation")
    config = PreparationConfig(prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        max_active=plan.concurrency, context=plan.recipe.context,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(
    name="qwen38_dense", served_model_name="Qwen3.8-27B", prepare=prepare, usage=usage,
    # Thinking defaults; bounds span thinking/instruct presets: https://huggingface.co/Qwen/Qwen3.8-27B#best-practices.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=20, min_p=0.,
        presence_penalty=0., repetition_penalty=1., frequency_penalty=0.,
        ranges={"temperature": SamplingRange(.7, 1.), "top_p": SamplingRange(.8, .95),
                "presence_penalty": SamplingRange(0., 2.)}),
    checkpoints=(Checkpoint(alias="qwen3.8-27b", repository="nvidia/Qwen3.8-27B-NVFP4",
        revision="482ca0f3832238542f8f5295dde86b5f22711d80", architecture="Qwen3_5ForConditionalGeneration",
        stored_bytes=21_921_428_072, device_bytes=20_150_569_096, expansion_bytes=307_871_744),),
    context=262144, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    max_draft_tokens=0, ranks=MappingProxyType({1: RankCosts(158_148_864, 4_194_304, 4 << 30)}),
)
