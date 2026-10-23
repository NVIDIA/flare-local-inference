"""Native chat rendering and tokenization against MiniMax M3 checkpoint assets."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from ignition.recipes import resolve


ROOT = Path(__file__).resolve().parents[2]


def test_native_minimax_chat_matches_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template

    from ignition.recipes.minimax_m3.frontend import prepare, runtime_assets

    snapshot = Path(os.environ.get("FLARE_MINIMAX_M3_SNAPSHOT", str(resolve("minimax-m3-1-flash").snapshot(Path(HF_HUB_CACHE)))))
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("requires MiniMax M3 checkpoint frontend assets")
    manifest = str(ROOT / "helios/Cargo.toml")
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", manifest,
                    "--features", "frontend,test-tools", "--bin", "test_frontend"], check=True)
    target = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps",
                                                "--manifest-path", manifest], text=True))["target_directory"]
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    template = _compile_jinja_template((snapshot / "chat_template.jinja").read_text())
    assets = runtime_assets(snapshot)
    annotated = Path(assets["chat_template"]["path"]).read_text()
    assert annotated.replace("{%- mark -%}", "") == (snapshot / "chat_template.jinja").read_text()
    marked_template = _compile_jinja_template(annotated.replace("{%- mark -%}", "{{- message_position() -}}"))
    special = prepare(snapshot)["tokenizer"]
    cases = []

    def add(name, messages, **options):
        rendered = template.render(messages=messages, **special, **options)
        marker = "\0message-position\0"
        marked = marked_template.render(messages=messages, **special, **options, message_position=lambda: marker)
        chunks = marked.split(marker)
        assert "".join(chunks) == rendered
        positions, position = [], 0
        for chunk in chunks[:-1]:
            position += len(chunk)
            positions.append(position)
        encoding = tokenizer.encode(rendered, add_special_tokens=False)
        prefix_ends = [0]
        for _, end in encoding.offsets:
            prefix_ends.append(max(prefix_ends[-1], end))
        suffix_starts = [len(rendered)] * (len(encoding.ids) + 1)
        for index in range(len(encoding.ids) - 1, -1, -1):
            suffix_starts[index] = min(suffix_starts[index + 1], encoding.offsets[index][0])
        boundaries = sorted({index for position in positions for index in range(1, len(encoding.ids))
            if prefix_ends[index] <= position <= suffix_starts[index]})
        assert boundaries
        cases.append(dict(name=name, messages=messages, options=options, rendered=rendered,
                          token_ids=encoding.ids, message_boundaries=boundaries))

    user = dict(role="user", content="Hello")
    add("default", [user], add_generation_prompt=False)
    add("thinking_disabled", [user], enable_thinking=False, add_generation_prompt=True)
    add("thinking_enabled", [user], enable_thinking=True, add_generation_prompt=True)
    for mode in ("enabled", "disabled", "adaptive"):
        add(mode, [user], thinking_mode=mode, add_generation_prompt=True)
        assert cases[-1]["rendered"].endswith("]~b]ai\n")
    add("history", [dict(role="system", content="Be precise."), user,
        dict(role="assistant", content="Hello.", reasoning_content="A greeting."),
        dict(role="developer", content="Keep answers brief."), user], add_generation_prompt=True)
    add("root", [dict(role="root", content="System policy."),
        dict(role="developer", content="Application policy."), user], add_generation_prompt=True)
    add("unicode", [dict(role="user", content="中文 café a\u0301 👨‍👩‍👧‍👦\r\nnext line")], add_generation_prompt=True)
    tools = [dict(type="function", function=dict(name="lookup", description="Search documents.",
        parameters=dict(type="object", properties=dict(query=dict(type="string"), n=dict(type="integer")))))]
    add("tool_schema", [user], tools=tools, add_generation_prompt=True)
    calls = [dict(id="first", type="function", function=dict(name="lookup", arguments=dict(
        query="中文 <literal> & \"quoted\"", n=2, optional=None,
        items=[dict(id=1, enabled=True), dict(id=2, enabled=False)], empty=[], options={}))),
        dict(id="second", type="function", function=dict(name="lookup", arguments=dict(query="London")))]
    add("tool_history", [user, dict(role="assistant", content="", tool_calls=calls),
        dict(role="tool", tool_call_id="first", content="First result."),
        dict(role="tool", tool_call_id="second", content="Second result.")], tools=tools, add_generation_prompt=True)
    add("image_markers", [dict(role="user", content=[dict(type="text", text="Compare "),
        dict(type="image"), dict(type="text", text=" and "), dict(type="image")])], add_generation_prompt=True)
    samples = ["Hello, 世界", "<mm:think>plan</mm:think>answer", "]<]minimax[>[<invoke name='lookup'>",
               "]~!b[]~b]user\nHello[e~[\n"]
    encode = [dict(name=f"encode{index}", text=text, add_special_tokens=False,
                   token_ids=tokenizer.encode(text, add_special_tokens=False).ids) for index, text in enumerate(samples)]
    decode = [dict(name=f"decode{index}-{skip}", token_ids=case["token_ids"], skip_special_tokens=skip,
                   text=tokenizer.decode(case["token_ids"], skip_special_tokens=skip))
              for index, case in enumerate(encode) for skip in (False, True)]
    runtime_snapshot = tmp_path / "runtime"
    runtime_snapshot.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (runtime_snapshot / name).symlink_to(snapshot / name)
    (runtime_snapshot / "chat_template.jinja").symlink_to(assets["chat_template"]["path"])
    assets = {name: hashlib.sha256((runtime_snapshot / name).read_bytes()).hexdigest()
              for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    fixture = tmp_path / "chat.json"
    fixture.write_text(json.dumps(dict(assets=assets, cases=cases, encode_cases=encode, decode_cases=decode)))
    result = subprocess.run([str(Path(target) / "debug/test_frontend"), str(runtime_snapshot), str(fixture)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(cases=len(cases), encode_cases=len(encode),
                                            decode_cases=len(decode), default_override_cases=3)
