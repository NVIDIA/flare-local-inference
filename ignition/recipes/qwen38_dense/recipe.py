"""Single-GPU Qwen3.8-27B NVFP4 text preparation without host-mapped storage."""

from types import MappingProxyType
from pathlib import Path

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Qwen dense preparation supports one local GPU")
    speculation = plan.speculation
    drafting = {}
    if speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        drafting = dict(draft_snapshot=Path(snapshot_download(REPOSITORY, revision=REVISION,
            cache_dir=plan.hub_cache, allow_patterns=("config.json", "model.safetensors"))),
            draft_head_quantization=speculation.draft_head_quantization,
            speculative_graph_budget=speculation.graph_budget or 64)
    selection = plan.benchmark_selection
    config = PreparationConfig(benchmark_selection=selection, collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        online_mxfp8=plan.online_mxfp8,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        max_active=plan.concurrency if selection is None else selection.max_batch, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, mtp=speculation is not None,
        speculative_widths=(1,) if speculation is None else speculation.widths,
        speculative_fixed_width=None if speculation is None else speculation.fixed_width,
        speculative_token_capacity=None if speculation is None else plan.speculative_token_capacity, **drafting)
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
        stored_bytes=21_921_428_072, device_bytes=24_104_235_656, draft_bytes=3_953_666_560,
        expansion_bytes=307_871_744),
        Checkpoint(alias="qwen3.8-27b-qad", repository="local-inference-lab/Qwen3.8-27B-NVFP4-QAD",
        revision="b9389787cbb5a09fbcf3517bdf673b5a56b16b02", architecture="Qwen3_5ForConditionalGeneration",
        stored_bytes=25_578_292_192, device_bytes=27_761_099_776, draft_bytes=3_953_666_560,
        expansion_bytes=1_619_001_344)),
    context=262144, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, verification_rows=0,
    online_mxfp8=True, draft_head_quantizations=("mxfp8", "nvfp4"),
    ranks=MappingProxyType({1: RankCosts(186_001_664, 4_194_304, 4 << 30)}),
    plain_ranks=MappingProxyType({1: RankCosts(158_148_864, 4_194_304, 4 << 30)}),
)
