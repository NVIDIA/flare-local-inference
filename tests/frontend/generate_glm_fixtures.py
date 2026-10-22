"""Render GLM request cases with the checkpoint's canonical Transformers template."""

import argparse
import copy
import json
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template


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
        prefix_options = {**effective, "add_generation_prompt": False, "continue_final_message": False}
        boundary_messages = copy.deepcopy(messages)
        for index, message in enumerate(boundary_messages):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            ranks = {call["id"]: rank for rank, call in enumerate(message["tool_calls"])}
            end = index + 1
            while end < len(boundary_messages) and boundary_messages[end]["role"] == "tool":
                end += 1
            boundary_messages[index + 1:end] = sorted(boundary_messages[index + 1:end],
                key=lambda result: ranks[result["tool_call_id"]])
        boundaries = []
        length = len(boundary_messages)
        for count in sorted({0, min(1, length), max(0, length - 1), length}):
            if count:
                prefix = tokenizer.apply_chat_template(boundary_messages[:count], tokenize=True,
                    return_dict=False, **prefix_options)
            else:
                text = _compile_jinja_template(tokenizer.get_chat_template()).render(
                    messages=[], **tokenizer.special_tokens_map, **prefix_options)
                prefix = tokenizer.encode(text, add_special_tokens=False)
            if 0 < len(prefix) < len(ids) and ids[:len(prefix)] == prefix:
                boundaries.append(len(prefix))
        boundaries = sorted(set(boundaries))
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
    return {"cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.snapshot), ensure_ascii=False))
