"""Native Step crop layout, RGB normalization, and placeholders against Transformers."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(os.environ.get("FLARE_STEP5_SNAPSHOT", Path.home() / ".cache/huggingface/hub/models--stepfun-ai--Step-5-Preview-NVFP4/snapshots/bc246cc2848e17b37f377eda8fd825c5b3018d1f"))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    if not (CHECKPOINT / "tokenizer.json").exists():
        pytest.skip("Step-5 tokenizer is unavailable")
    from tokenizers import Tokenizer
    from transformers.models.step3p7.image_processing_step3p7 import (
        Step3p7ImageProcessor,
    )

    from ignition.recipes.step5.frontend import inference_template
    assets = tmp_path_factory.mktemp("step5-assets")
    (assets / "chat_template.jinja").write_text(inference_template(CHECKPOINT))
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
        "--features", "server,test-tools", "--bin", "test_step_images"], check=True)
    return Step3p7ImageProcessor(), Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json")), assets


def test_native_chat_template(reference, tmp_path):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    _, tokenizer, assets = reference
    original = _compile_jinja_template((CHECKPOINT / "chat_template.jinja").read_text())
    config = json.loads((CHECKPOINT / "tokenizer_config.json").read_text())
    special = {key: config[key] for key in ("bos_token", "eos_token")}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (assets / name).symlink_to(CHECKPOINT / name)
    cases = [
        ("user", [{"role": "user", "content": "Hello"}], {}),
        ("history", [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": "Consider the question.", "content": "Hello."},
            {"role": "user", "content": "Continue."}], {}),
        ("effort", [{"role": "user", "content": "Think."}], {"reasoning_effort": "high"}),
        ("image", [{"role": "user", "content": "Describe <im_patch>"}], {}),
        ("tool", [{"role": "user", "content": "Find it."}, {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "中文", "count": 2}}}]},
            {"role": "tool", "content": "Found."}], {"tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]}),
    ]
    fixtures = {"cases": [], "encode_cases": [], "decode_cases": [], "assets": {}}
    for name, messages, options in cases:
        for preserve in (False, True):
            options = dict(options, add_generation_prompt=True, preserve_thinking=preserve)
            rendered = original.render(messages=messages, **special, **options)
            fixtures["cases"].append({"name": f"{name}-{preserve}", "messages": messages, "options": options,
                "rendered": rendered, "token_ids": tokenizer.encode(rendered, add_special_tokens=False).ids})
    for name, options in (("thinking_disabled", {"enable_thinking": False, "add_generation_prompt": True}),
                          ("thinking_enabled", {"enable_thinking": True, "add_generation_prompt": True}),
                          ("default", {"enable_thinking": True, "add_generation_prompt": False})):
        messages = [{"role": "user", "content": "Hello"}]
        rendered = original.render(messages=messages, **special, **options)
        fixtures["cases"].append({"name": name, "messages": messages, "options": options, "rendered": rendered,
            "token_ids": tokenizer.encode(rendered, add_special_tokens=False).ids})
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        fixtures["assets"][name] = hashlib.sha256((assets / name).read_bytes()).hexdigest()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
                    "--features", "frontend,test-tools", "--bin", "test_frontend"], check=True)
    result = subprocess.run([str(ROOT / "helios/target/debug/test_frontend"), str(assets), str(path)],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("sizes", [[(1, 1)], [(503, 701)], [(900, 1000)], [(700, 200), (32, 127)],
                                   [(16, 1000)], [(3400, 900)]])
def test_native_images(reference, tmp_path, sizes):
    processor, tokenizer, assets = reference
    images, paths = [], []
    rng = np.random.default_rng(51)
    for index, (width, height) in enumerate(sizes):
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        path = tmp_path / f"image{index}.png"
        image.save(path)
        images.append(image)
        paths.append({"path": str(path), "format": "png"})
    expected = processor(images, return_tensors="pt")
    prompt = "Look: " + " and ".join("<im_patch>" for _ in images) + "."
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"images": paths, "prompt": prompt}))
    subprocess.run([str(ROOT / "helios/target/debug/test_step_images"), str(CHECKPOINT), str(assets),
                     str(case), str(tmp_path)], check=True)
    actual = json.loads((tmp_path / "metadata.json").read_text())
    pixels = np.fromfile(tmp_path / "pixels.f32", dtype=np.float32).reshape(-1, 588)
    reference_patches, expansions, positions = [], [], []
    local_index = 0
    for index, count in enumerate(expected["num_local_patches"].tolist()):
        views = [*expected.get("pixel_values_local", [])[local_index:local_index + count], expected["pixel_values"][index]]
        local_index += count
        for view in views:
            side = view.shape[-1] // 14
            reference_patches.append(view.reshape(3, side, 14, side, 14).permute(1, 3, 0, 2, 4).reshape(-1, 588))
            positions.extend([[y, x] for y in range(side) for x in range(side)])
        mask = expected.get("patch_newline_masks", [[]] * len(images))[index]
        text = "".join("<patch_start>" + "<im_patch>" * 81 + "<patch_end>" + ("<patch_newline>" if mask[i] else "")
                       for i in range(count))
        expansions.append(text + "<im_start>" + "<im_patch>" * 169 + "<im_end>")
    np.testing.assert_allclose(pixels, torch.cat(reference_patches).numpy(), atol=1e-6, rtol=0)
    tokens = tokenizer.encode("Look: " + " and ".join(expansions) + ".", add_special_tokens=False).ids
    assert actual["tokens"] == tokens
    assert actual["positions"] == positions
    feature = tokenizer.token_to_id("<im_patch>")
    feature_index = 0
    for token, index in zip(tokens, actual["indices"], strict=True):
        if token == feature:
            assert index == feature_index
            feature_index += 1
        else:
            assert index == -1
    assert feature_index * 16 == len(pixels)
