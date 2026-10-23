"""Native Bolt chat rendering and tokenization against checkpoint assets."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_native_nemotron_chat_matches_checkpoint(tmp_path):
    from transformers import AutoTokenizer

    snapshot = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT",
        "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    if not (snapshot / "tokenizer_config.json").is_file():
        pytest.skip("requires the Nemotron Bolt checkpoint frontend")
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
                    "--features", "frontend,test-tools", "--bin", "test_frontend"], check=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    user = dict(role="user", content="What is 17 times 23?")
    cases = []

    def add(name, messages, **options):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        cases.append(dict(name=name, messages=messages, options=options, rendered=rendered,
                          token_ids=tokenizer.encode(rendered, add_special_tokens=False)))

    add("default", [user], add_generation_prompt=False)
    add("thinking_disabled", [user], add_generation_prompt=True, enable_thinking=False)
    add("thinking_enabled", [user], add_generation_prompt=True, enable_thinking=True)
    add("medium_effort", [user], add_generation_prompt=True, medium_effort=True)
    history = [dict(role="system", content="Be precise."), user,
               dict(role="assistant", content="391", reasoning_content="Multiply 17 by 23."),
               dict(role="user", content="Explain the calculation.")]
    for truncate in (False, True):
        add(f"reasoning_history_{truncate}", history, add_generation_prompt=True,
            truncate_history_thinking=truncate)
    add("assistant_null_fields", [user, dict(role="assistant", content=None, tool_calls=None), user],
        add_generation_prompt=True)
    add("unicode", [dict(role="user", content="中文 é a\u0301 👨‍👩‍👧‍👦\r\n  next line")],
        add_generation_prompt=True)

    tools = [dict(type="function", function=dict(name="lookup", description=" Search documents. ",
        parameters=dict(type="object", properties=dict(query=dict(type="string"), count=dict(type="integer"),
            options=dict(type="object", properties=dict(enabled=dict(type="boolean")))),
            required=["query"], additionalProperties=False)))]
    for name, value in (("absent_tools", None), ("empty_tools", []), ("tools", tools)):
        add(name, [user], add_generation_prompt=True, tools=value)
    calls = [dict(id="lookup-first", type="function", function=dict(name="lookup", arguments=dict(
        query="\nParis\nsecond line\n", count=2, options=dict(enabled=True, tags=["city", "法国"])))),
        dict(id="lookup-second", type="function", function=dict(name="lookup", arguments=dict(query="London")))]
    tool_history = [user, dict(role="assistant", content="", reasoning_content="Check both cities.", tool_calls=calls),
                   dict(role="tool", tool_call_id="lookup-first", content='{"temperature_c":21}'),
                   dict(role="tool", tool_call_id="lookup-second", content='{"temperature_c":15}')]
    add("tool_results", tool_history, tools=tools, add_generation_prompt=True)
    add("tool_history_before_user", [*tool_history, dict(role="assistant", content="Paris is warmer."), user],
        tools=tools, add_generation_prompt=True)
    for count in (1, 2, 4):
        content = [dict(type="text", text="Compare ")]
        for index in range(count):
            content.extend((dict(type="image"), dict(type="text", text=f" image {index + 1}. ")))
        add(f"images_{count}", [dict(role="user", content=content)], add_generation_prompt=True)
    add("explicit_image_placeholder", [dict(role="user", content=[dict(type="image"),
        dict(type="text", text="Describe <image>.")])], add_generation_prompt=True)

    samples = ["Hello, world!", "中文 é a\u0301 👨‍👩‍👧‍👦\r\n  next line", "<think>\nwork</think>answer",
               "<tool_call>\n<function=lookup>\n<parameter=query>\nParis\n</parameter>\n</function>\n</tool_call>"]
    encode_cases = [dict(name=f"encode_{index}", text=text, add_special_tokens=False,
                        token_ids=tokenizer.encode(text, add_special_tokens=False)) for index, text in enumerate(samples)]
    decode_cases = [dict(name=f"decode_{index}_{skip}", token_ids=case["token_ids"], skip_special_tokens=skip,
                        text=tokenizer.decode(case["token_ids"], skip_special_tokens=skip))
                    for index, case in enumerate(encode_cases) for skip in (False, True)]
    from ignition.preparation.frontend import annotated_template_asset
    annotated = annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
        ROOT / "ignition/recipes/nemotron_h/assets/chat_template.jinja")
    native_assets = tmp_path / "assets"
    native_assets.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (native_assets / name).symlink_to(snapshot / name)
    (native_assets / "chat_template.jinja").symlink_to(annotated["path"])
    assets = {name: hashlib.sha256((native_assets / name).read_bytes()).hexdigest()
              for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    fixture = tmp_path / "chat.json"
    fixture.write_text(json.dumps(dict(assets=assets, cases=cases,
                                      encode_cases=encode_cases, decode_cases=decode_cases)))
    result = subprocess.run([str(ROOT / "helios/target/debug/test_frontend"), str(native_assets), str(fixture)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(cases=len(cases), encode_cases=len(encode_cases),
                                          decode_cases=len(decode_cases), default_override_cases=3)
