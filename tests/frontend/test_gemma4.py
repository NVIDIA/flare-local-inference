"""Compare prepared Gemma prompts and token IDs with Google's pinned templates."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ignition.recipes.gemma4.bootstrap import chat_template_asset
from ignition.recipes.gemma4.config import ModelConfig
from ignition.recipes.gemma4.recipe import RECIPE


ROOT = Path(__file__).resolve().parents[2]


def snapshot(checkpoint):
    path = (Path.home() / ".cache/huggingface/hub"
            / ("models--" + checkpoint.repository.replace("/", "--")) / "snapshots" / checkpoint.revision)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (path / name).is_file():
            pytest.skip(f"{checkpoint.alias}: {name} is not cached")
    return path


@pytest.mark.parametrize("checkpoint", RECIPE.checkpoints, ids=lambda checkpoint: checkpoint.alias)
def test_native_gemma_prompts_match_google(checkpoint, tmp_path):
    from transformers import AutoTokenizer

    binary = ROOT / "helios/target/debug/test_frontend"
    if not binary.is_file():
        pytest.skip("build test_frontend with frontend,test-tools features")
    source = snapshot(checkpoint)
    small = checkpoint.alias in ("gemma-4-e2b", "gemma-4-e4b", "gemma-4-e4b-nvfp4")
    reference = next(c for c in RECIPE.checkpoints
                     if c.alias == ("gemma-4-e4b" if small else "gemma-4-26b-a4b"))
    canonical = (snapshot(reference) / "chat_template.jinja").read_text()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    asset = chat_template_asset(ModelConfig.load(source).hidden)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, tmp_path / name)
    shutil.copyfile(asset["path"], tmp_path / "chat_template.jinja")
    fixtures = dict(cases=[], encode_cases=[], decode_cases=[], assets={})
    for path in tmp_path.iterdir():
        fixtures["assets"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    user = {"role": "user", "content": "Hello café 世界"}
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Find a value",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    tool_history = [
        {"role": "system", "content": "Be precise"}, user,
        {"role": "assistant", "content": "", "reasoning_content": "Search first", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {
                "q": "café", "count": 2, "optional": None, "values": [True, 1e-6, -0.0, 1e16]}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Follow-up"},
    ]
    cases = [
        ("thinking_disabled", [user], {"enable_thinking": False}),
        ("thinking_enabled", [user], {"enable_thinking": True}),
        ("default", [user], {"enable_thinking": True, "add_generation_prompt": False}),
        ("reasoning", [user, {"role": "assistant", "reasoning_content": "Reason", "content": "A"}], {}),
        ("continuation", [user, {"role": "assistant", "content": "A"},
                          {"role": "assistant", "content": "B"}], {}),
        ("tool_response_parts", tool_history[:3] + [{"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "text", "text": "Result"}, {"type": "image_url", "image_url": {"url": "fixture://image"}}]}],
         {"tools": tools}),
    ]
    for thinking in (False, True):
        for preserve in (False, True):
            options = {"tools": tools, "enable_thinking": thinking, "preserve_thinking": preserve}
            cases.append((f"tools-{thinking}-{preserve}", tool_history, options))
            cases.append((f"pending-tool-{thinking}-{preserve}", tool_history[:4], options))
    for name, messages, options in cases:
        options = {"add_generation_prompt": True, **options}
        rendered = tokenizer.apply_chat_template(messages, chat_template=canonical, tokenize=False, **options)
        case = dict(name=name, messages=messages, options=options, rendered=rendered,
                    token_ids=tokenizer.encode(rendered, add_special_tokens=False))
        if name.startswith("tools-") and options["preserve_thinking"]:
            endpoints = []
            for count in (1, 2, 4, 5, 6):
                prefix = tokenizer.apply_chat_template(messages[:count], chat_template=canonical,
                    tokenize=False, **{**options, "add_generation_prompt": False})
                assert rendered.startswith(prefix)
                endpoints.append(len(tokenizer.encode(prefix, add_special_tokens=False)))
            case["message_boundaries"] = endpoints
        fixtures["cases"].append(case)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    result = subprocess.run([str(binary), str(tmp_path), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
