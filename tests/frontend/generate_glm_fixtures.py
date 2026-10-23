"""Render GLM request cases with the checkpoint's canonical Transformers template."""

import argparse
import base64
import io
import copy
import json
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer
from PIL import Image


def generate(snapshot):
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "count": {"type": "integer"}}}}}
    system = {"role": "system", "content": "Answer accurately. α 世界"}
    user = {"role": "user", "content": "Find the weather in Paris."}
    cases = []

    def add(name, messages, options=None, *, wire_strings=False):
        options = dict(options or {})
        continuation = options.get("continue_final_message", False)
        effective = {"add_generation_prompt": not continuation, **options}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **effective)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **effective)
        source = (Path(__file__).resolve().parents[2] / "ignition/recipes/glm53/assets/chat_template.jinja").read_text()
        assert source.replace("{%- mark -%}", "") == tokenizer.get_chat_template()
        marker = "\0message-position\0"
        while marker in rendered:
            marker += "_"
        annotated = source.replace("{%- mark -%}", "{{- message_position() -}}")
        marked = tokenizer.apply_chat_template(messages, chat_template=annotated, tokenize=False,
            message_position=lambda: marker, **effective)
        chunks = marked.split(marker)
        assert "".join(chunks) == rendered
        positions, position = [], 0
        for chunk in chunks[:-1]:
            position += len(chunk)
            positions.append(position)
        encoding = tokenizer.backend_tokenizer.encode(rendered, add_special_tokens=False)
        assert encoding.ids == ids
        offsets = encoding.offsets
        prefix_ends = [0]
        for _, end in offsets:
            prefix_ends.append(max(prefix_ends[-1], end))
        suffix_starts = [len(rendered)] * (len(offsets) + 1)
        for index in range(len(offsets) - 1, -1, -1):
            suffix_starts[index] = min(suffix_starts[index + 1], offsets[index][0])
        boundaries = sorted({index for position in positions for index in range(1, len(offsets))
            if prefix_ends[index] <= position <= suffix_starts[index]})
        body = {"model": "glm53", "messages": copy.deepcopy(messages), "chat_template_kwargs": options}
        if wire_strings:
            for message in body["messages"]:
                for call in message.get("tool_calls", []):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
        suffix = rendered.rsplit("<|assistant|>", 1)[-1]
        reasoning = suffix.rfind("<think>") > suffix.rfind("</think>")
        cases.append({"name": name, "body": body, "rendered": rendered, "token_ids": ids,
                      "message_boundaries": boundaries, "reasoning": reasoning})

    add("default_reasoning", [user])
    add("system", [system, user])
    add("unicode_parts", [system, {"role": "user", "content": [{"type": "text", "text": "café 👋 世界"}]}])
    for effort in ("low", "high", "max"):
        add(f"effort_{effort}", [system, user], {"reasoning_effort": effort})
    add("tools", [system, user], {"tools": [tool]})
    calls = [{"id": "a", "type": "function", "function": {"name": "lookup", "arguments": {"city": "Paris α", "count": 2}}},
             {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": {"city": "London", "count": 1}}}]
    history = [system, user, {"role": "assistant", "content": "", "reasoning_content": "Search cities.", "tool_calls": calls},
               {"role": "tool", "tool_call_id": "b", "content": "London: cloudy"},
               {"role": "tool", "tool_call_id": "a", "content": "Paris: clear"}]
    add("parallel_tool_results", history, {"tools": [tool]}, wire_strings=True)
    for clear in (False, True):
        add(f"retained_reasoning_{clear}", [*history, {"role": "user", "content": "Compare."}],
            {"tools": [tool], "clear_thinking": clear})
    add("assistant_continuation", [system, user, {"role": "assistant", "content": "The weather is"}],
        {"continue_final_message": True})
    add("reasoning_continuation", [system, user, {"role": "assistant", "content": "", "reasoning_content": "Check cities"}],
        {"continue_final_message": "reasoning_content"})
    add("empty_assistant_continuation", [system, user, {"role": "assistant", "content": "", "reasoning_content": ""}],
        {"continue_final_message": True})
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    image_cases = []
    for count in (1, 2):
        images = [Image.new("RGB", (112 + 28 * i, 84), (31 * i, 100, 210)) for i in range(count)]
        content = [{"type": "text", "text": "Describe the images."}]
        canonical = [{"type": "text", "text": "Describe the images."}]
        for image in images:
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            content.append({"type": "image_url", "image_url": {"url":
                "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()}})
            canonical.append({"type": "image"})
        messages = [system, {"role": "user", "content": canonical}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        reference = processor(text=rendered, images=images, return_tensors="pt")
        image_cases.append({"body": {"model": "glm53", "messages": [system, {"role": "user", "content": content}]},
            "token_ids": reference.input_ids[0].tolist(),
            "grid_thw": reference.image_grid_thw.tolist()})
    return {"cases": cases, "image_cases": image_cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.snapshot), ensure_ascii=False))
