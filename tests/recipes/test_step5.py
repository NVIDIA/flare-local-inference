"""Step-5 checkpoint declarations, quantization exclusions, and cache geometry."""

import copy
import json
import os
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes import resolve
from ignition.recipes.step5.config import ModelConfig
from ignition.recipes.step5.frontend import prepare_frontend
from ignition.recipes.step5.pool import continuation_layout, page_bytes


@pytest.fixture(scope="module")
def checkpoint_config():
    model = resolve("step5-preview-nvfp4")
    snapshot = Path(os.environ.get("FLARE_STEP5_SNAPSHOT", model.snapshot(Path.home() / ".cache/huggingface/hub")))
    if not (snapshot / "config.json").exists():
        pytest.skip("Step-5 checkpoint configuration is unavailable")
    return json.loads((snapshot / "config.json").read_text())


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("mtp", [False, True])
def test_declared_cache_geometry(checkpoint_config, world, mtp):
    config = ModelConfig.from_dict(checkpoint_config)
    shard = TensorParallelShard(0, world)
    config.validate_shard(shard)
    recipe = resolve("stepfun-ai/Step-5-Preview-NVFP4").recipe
    costs = recipe.costs(world, recipe.speculation if mtp else None)
    assert costs.state_row_bytes == continuation_layout(config, shard, mtp=mtp).row_bytes
    assert costs.page_bytes == page_bytes(config, shard)
    covered = []
    for rank in range(world):
        first, last, width = config.vocabulary_partition(TensorParallelShard(rank, world))
        assert width % 64 == 0
        covered.extend(range(first, last))
    assert covered == list(range(config.vocabulary))


def test_expert_precision_comes_from_metadata(checkpoint_config):
    config = ModelConfig.from_dict(checkpoint_config)
    assert config.expert_format("model.layers.3.moe") == "nvfp4"
    assert config.expert_format("model.layers.88.moe") == "bf16"
    changed = copy.deepcopy(checkpoint_config)
    changed["quantization_config"]["ignore"].append("*layers.3.moe.experts*")
    assert ModelConfig.from_dict(changed).expert_format("model.layers.3.moe") == "bf16"
    changed["quantization_config"]["ignore"].pop()
    changed["quantization_config"]["ignore"].append("*layers.3.moe.experts.0.gate_proj")
    with pytest.raises(ValueError, match="one declared weight format"):
        ModelConfig.from_dict(changed).expert_format("model.layers.3.moe")


def test_sampling_defaults():
    assert resolve("step5-preview-nvfp4").recipe.sampling_defaults.as_dict() == {
        "seed": 0, "temperature": 1., "top_p": .95, "top_k": -1, "min_p": 0., "repetition_penalty": 1.,
        "presence_penalty": 0., "frequency_penalty": 0.}


@pytest.mark.parametrize("world", [4, 8, 16])
def test_serial_expert_layers_share_scale_storage(checkpoint_config, world):
    import torch

    from ignition.recipes.step5.weights import Step5WeightLoader

    loader = Step5WeightLoader.__new__(Step5WeightLoader)
    loader.config = ModelConfig.from_dict(checkpoint_config)
    loader.shard = TensorParallelShard(0, world)
    loader.device = torch.device("meta")
    loader.expert_scale_compression = "csf"
    local = loader.config.text["moe_intermediate_size"] // world
    a4 = loader.expert_scale_scratch(local)
    a16 = loader.expert_scale_scratch((local + 127) // 128 * 128)
    for source, target, owner in zip(a4, a16, loader.scale_scratch, strict=True):
        assert source.is_contiguous() and target.is_contiguous()
        assert source.untyped_storage()._cdata == target.untyped_storage()._cdata == owner.untyped_storage()._cdata
        assert source.numel() <= target.numel() == owner.numel()


def test_generation_stops_at_chat_and_base_eos(tmp_path):
    from tokenizers import Tokenizer, models

    Tokenizer(models.WordLevel({"<base_eos>": 0, "<pad>": 1, "<|im_end|>": 2})).save(
        str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    assert prepare_frontend(tmp_path, eos_token_ids=[0, 1])["eos_token_ids"] == [0, 1, 2]
    assert prepare_frontend(tmp_path, eos_token_ids=[0, 2])["eos_token_ids"] == [0, 2]
    with pytest.raises(ValueError, match="EOS token IDs"):
        prepare_frontend(tmp_path, eos_token_ids=[0, 0])
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<missing>"}))
    with pytest.raises(ValueError, match="EOS marker is missing"):
        prepare_frontend(tmp_path, eos_token_ids=[0, 1])


def test_vision_text_and_draft_scratch_share_storage(checkpoint_config):
    from dataclasses import fields, is_dataclass
    import torch
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.step5.decoder import Scratch
    from ignition.recipes.step5.vision import VisionScratch

    config = ModelConfig.from_dict(checkpoint_config)
    shard = TensorParallelShard(0, 4)
    pool = GraphBufferPool('meta', chunk_bytes=2 << 20)
    vision = VisionScratch.allocate(65536, shard, 'meta', buffers=pool.cursor())
    text = Scratch.allocate(config, shard, 4096, 1, 1048576, 'meta', buffers=pool.cursor())
    decode = Scratch.allocate(config, shard, 16, 16, 1048576, 'meta', buffers=pool.cursor())
    draft = Scratch.allocate(config, shard, 64, 64, 1048576, 'meta', buffers=pool.cursor())

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from tensors(getattr(value, field.name))

    owners = {t.untyped_storage()._cdata for chunks in pool.chunks.values() for t in chunks}
    footprints = []
    for execution in (vision, text, decode, draft):
        values = list(tensors(execution))
        footprints.append(sum(t.nbytes for t in values))
        assert all(t.untyped_storage()._cdata in owners for t in values)
        if execution is vision:
            persistent = [vision.patches, vision.hidden, vision.normalized,
                          vision.projected, vision.reduced]
            stages = [persistent + [vision.qkv, vision.query, vision.key, vision.value, vision.attended],
                      persistent + [vision.mlp_up, vision.activated],
                      persistent + [vision.first_input, vision.first_output, vision.second_input,
                                    vision.second_output, vision.second_reduced, vision.output]]
        else:
            private = {"attention", "dense", "shared", "router_logits", "router", "experts", "partial"}
            persistent = [t for field in fields(execution) if field.name not in private
                          for t in tensors(getattr(execution, field.name))]
            stages = [persistent + list(tensors(execution.attention)),
                      persistent + list(tensors(execution.dense)),
                      persistent + [t for field in private - {"attention", "dense"}
                                    for t in tensors(getattr(execution, field))]]
        for stage in stages:
            intervals = {}
            for tensor in stage:
                start = tensor.storage_offset() * tensor.element_size()
                intervals.setdefault(tensor.untyped_storage()._cdata, []).append((start, start + tensor.nbytes))
            for ranges in intervals.values():
                ordered = sorted(ranges)
                assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    assert vision.patches.untyped_storage()._cdata == text.embedding.untyped_storage()._cdata
    assert (text.embedding.untyped_storage()._cdata == decode.embedding.untyped_storage()._cdata
            == draft.embedding.untyped_storage()._cdata)
    assert sum(footprints) - pool.nbytes > 400 << 20
    assert text.attention.key.dtype == text.attention.value.dtype == torch.bfloat16


def test_runtime_assets_preserve_validated_annotated_template_path(tmp_path):
    import hashlib
    from ignition.recipes.step5 import frontend

    template = Path(frontend.__file__).with_name("assets") / "chat_template.jinja"
    annotated = template.read_bytes()
    source = annotated.replace(b"{%- mark -%}", b"")
    opening = b"{%- if true %}"
    closing = b"{%- endif %}\n    {%- elif message.role == \"tool\" %}"
    assert source.count(opening) == source.count(closing) == 1
    source = source.replace(opening, b"{%- generation %}").replace(
        closing, b"{%- endgeneration %}\n    {%- elif message.role == \"tool\" %}")
    checkpoint_template = tmp_path / "chat_template.jinja"
    checkpoint_template.write_bytes(source)

    assets = frontend.runtime_assets(tmp_path)
    assert assets["chat_template"] == {
        "path": str(template.resolve()), "sha256": hashlib.sha256(annotated).hexdigest()}
    assert assets["tokenizer"] == {"path": str(tmp_path / "tokenizer.json")}
    assert checkpoint_template.read_bytes() == source
    assert template.read_bytes() == annotated
    assert [path.name for path in tmp_path.iterdir()] == ["chat_template.jinja"]

    checkpoint_template.write_bytes(source + b" ")
    with pytest.raises(ValueError, match="checkpoint chat template does not match"):
        frontend.runtime_assets(tmp_path)
