"""SDK and HTTP contracts against an actual native Flare service.

Install the api-test extra and set FLARE_API_TEST_URL to an idle service.
FLARE_API_TEST_KEY optionally supplies that service's configured API key.
"""

import base64
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("FLARE_API_TEST_URL"), reason="requires a native Flare service"
)


@pytest.fixture(scope="module")
def clients():
    import httpx
    from anthropic import Anthropic
    from openai import OpenAI

    base = os.environ["FLARE_API_TEST_URL"].rstrip("/")
    key = os.environ.get("FLARE_API_TEST_KEY", "local")
    with (
        OpenAI(base_url=base + "/v1", api_key=key, timeout=120, max_retries=0) as openai,
        Anthropic(base_url=base, api_key=key, timeout=120, max_retries=0) as anthropic,
        httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=120) as http,
    ):
        yield openai, anthropic, http


EXTRA = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 1}
USER = [{"role": "user", "content": "Reply with exactly OK."}]
FUNCTION = {
    "name": "lookup",
    "description": "Look up a value in the local database.",
    "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
    },
}


def message_input_tokens(usage):
    return usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)


def test_model_discovery_and_lookup(clients):
    o, _, http = clients
    listed = http.get("/v1/models")
    listed.raise_for_status()
    assert listed.json()["object"] == "list"
    cards = listed.json()["data"]
    assert len(cards) == 1
    assert [model.id for model in o.models.list()] == [card["id"] for card in cards]
    for card in cards:
        assert card["object"] == "model"
        assert card["created"] > 0
        assert card["max_model_len"] > 0
        assert card["root"]
        assert card["parent"] is None
        assert card["flare"]["max_concurrency"] > 0
        assert card["flare"]["input_modalities"][0] == "text"
        assert card["flare"]["sampling_defaults"]["temperature"] > 0
        assert card["permission"][0]["allow_sampling"] is True
        assert card["permission"][0]["allow_logprobs"] is (card["flare"]["max_top_logprobs"] > 0)
        assert o.models.retrieve(card["id"]).model_dump() == card
        assert o.models.retrieve(card["root"]).model_dump() == card
        encoded = http.get("/v1/models/" + quote(card["id"], safe=""))
        encoded.raise_for_status()
        assert encoded.json() == card
    assert http.get("/v1/models").json()["data"] == cards
    missing = http.get("/v1/models/__flare_missing_model__")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
    if os.environ.get("FLARE_API_TEST_KEY"):
        wrong_key = os.environ["FLARE_API_TEST_KEY"] + "-invalid"
        for path in ["/v1/models", "/v1/models/" + quote(cards[0]["id"], safe="")]:
            denied = http.get(path, headers={"Authorization": f"Bearer {wrong_key}"})
            assert denied.status_code == 401


def test_sdk_buffered_and_streaming_text(clients):
    o, a, _ = clients
    chat = o.chat.completions.create(model="local", messages=USER, max_tokens=32, extra_body=EXTRA)
    assert chat.choices[0].message.content == "OK"
    chunks = list(o.chat.completions.create(
        model="local", messages=USER, max_tokens=32, stream=True,
        stream_options={"include_usage": True}, extra_body=EXTRA,
    ))
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "OK"
    assert chunks[-1].usage.total_tokens > 0
    with a.messages.stream(model="local", messages=USER, max_tokens=32, extra_body=EXTRA) as stream:
        assert "".join(stream.text_stream) == "OK"
        message = stream.get_final_message()
    assert message.stop_reason == "end_turn"
    buffered = a.messages.create(model="local", messages=USER, max_tokens=32, extra_body=EXTRA)
    assert [b.model_dump(exclude_none=True) for b in buffered.content] == [b.model_dump(exclude_none=True) for b in message.content]
    assert message_input_tokens(buffered.usage) == message_input_tokens(message.usage)
    with o.responses.stream(model="local", input=USER, max_output_tokens=32, extra_body=EXTRA) as stream:
        events = list(stream)
        response = stream.get_final_response()
    assert response.output_text == "OK"
    assert response.status == "completed"
    assert events[-1].type == "response.completed"
    buffered = o.responses.create(model="local", input=USER, max_output_tokens=32, extra_body=EXTRA)
    assert buffered.output_text == response.output_text


def test_raw_completion_batch_and_zero_tokens(clients):
    o, _, _ = clients
    zero = o.completions.create(model="local", prompt=["Hello", "World"], n=2, echo=True, max_tokens=0)
    assert [c.text for c in zero.choices] == ["Hello", "Hello", "World", "World"]
    assert zero.usage.completion_tokens == 0
    assert zero.usage.prompt_tokens == 2
    assert zero.usage.prompt_tokens_details.cached_tokens == 0
    result = o.completions.create(model="local", prompt=["The capital of France is", "One plus one equals"], n=2, max_tokens=4, temperature=1)
    assert [c.index for c in result.choices] == [0, 1, 2, 3]
    assert all(c.text for c in result.choices)
    chunks = list(o.completions.create(model="local", prompt=["Hello", "World"], n=2, max_tokens=2, stream=True, stream_options={"include_usage": True}, temperature=1))
    assert {c.index for chunk in chunks for c in chunk.choices} == {0, 1, 2, 3}
    assert chunks[-1].usage.completion_tokens == 8
    raw = o.completions.create(model="local", prompt=[1, 2, 3], max_tokens=0)
    assert raw.usage.prompt_tokens == 3


@pytest.mark.parametrize("hinted", [False, True])
def test_messages_block_hint_reuses_history_after_suffix_changes(clients, hinted):
    _, anthropic, _ = clients
    history = {"type": "text", "text": f"History {time.time_ns()}. "
               + "An earlier action read a file and checked its output.\n" * 100}
    if hinted:
        history["cache_control"] = {"type": "ephemeral"}
    blocks = [history, {"type": "text", "text": "First action: inspect the file. Reply OK."}]
    kwargs = {"model": "local", "max_tokens": 1, "extra_body": EXTRA}
    anthropic.messages.create(
        messages=[{"role": "user", "content": "Warm the shared message preamble."}], **kwargs,
    )
    first = anthropic.messages.create(
        messages=[{"role": "user", "content": blocks}], **kwargs,
    )
    blocks[1] = {"type": "text", "text": "Second action: run the tests. Reply OK."}
    history.pop("cache_control", None)
    second = anthropic.messages.create(
        messages=[{"role": "user", "content": blocks}], **kwargs,
    )
    reused = (second.usage.cache_read_input_tokens or 0) - (first.usage.cache_read_input_tokens or 0)
    if hinted:
        assert reused > 500
    else:
        assert reused == 0


@pytest.mark.parametrize("protocol", ["chat", "completions", "messages", "responses"])
def test_cached_usage_buffered_and_streaming(clients, protocol):
    o, a, _ = clients
    prompt = f"Usage test {time.time_ns()}. " + "A cached prefix avoids repeated computation. " * 40
    prompt += "Reply with OK."
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": "local", "extra_body": EXTRA}
    if protocol == "chat":
        create = o.chat.completions.create
        kwargs.update(messages=messages, max_tokens=8)
    elif protocol == "completions":
        create = o.completions.create
        kwargs.update(prompt=prompt, max_tokens=8)
    elif protocol == "messages":
        create = a.messages.create
        kwargs.update(messages=messages, max_tokens=8)
    else:
        create = o.responses.create
        kwargs.update(input=messages, max_output_tokens=8)

    first = create(**kwargs)
    buffered = create(**kwargs)
    if protocol in {"chat", "completions"}:
        chunks = list(create(**kwargs, stream=True, stream_options={"include_usage": True}))
        assert not chunks[-1].choices
        assert all(chunk.usage is None for chunk in chunks[:-1])
        usages = [buffered.usage, chunks[-1].usage]
        for usage in usages:
            assert 0 < usage.prompt_tokens_details.cached_tokens <= usage.prompt_tokens
            assert usage.prompt_tokens == first.usage.prompt_tokens
            assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
            assert usage.completion_tokens > 0
    elif protocol == "messages":
        with a.messages.stream(**kwargs) as stream:
            events = list(stream)
            final = stream.get_final_message()
        delta = next(event for event in events if event.type == "message_delta")
        for usage in [buffered.usage, final.usage, delta.usage]:
            assert 0 < usage.cache_read_input_tokens <= message_input_tokens(usage)
            assert message_input_tokens(usage) == message_input_tokens(first.usage)
            assert usage.output_tokens > 0
    else:
        events = list(create(**kwargs, stream=True))
        assert events[-1].type in {"response.completed", "response.incomplete"}
        for usage in [buffered.usage, events[-1].response.usage]:
            assert 0 < usage.input_tokens_details.cached_tokens <= usage.input_tokens
            assert usage.input_tokens == first.usage.input_tokens
            assert usage.total_tokens == usage.input_tokens + usage.output_tokens
            assert usage.output_tokens > 0


def test_cached_batch_usage_counts_each_input_once(clients):
    o, _, _ = clients
    prompts = [f"Batch usage {time.time_ns()} {i}. " + "Count this prefix once. " * 40 for i in range(2)]
    kwargs = {"model": "local", "prompt": prompts, "max_tokens": 2, "temperature": 1,
              "extra_body": {"ignore_eos": True}}
    first = o.completions.create(**kwargs)
    buffered = o.completions.create(**kwargs, n=2)
    chunks = list(o.completions.create(**kwargs, n=2, stream=True, stream_options={"include_usage": True}))
    for usage in [buffered.usage, chunks[-1].usage]:
        assert usage.prompt_tokens == first.usage.prompt_tokens
        assert 0 < usage.prompt_tokens_details.cached_tokens <= usage.prompt_tokens
        assert usage.completion_tokens == 8
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_token_count_and_compressed_requests(clients):
    _, a, http = clients
    counted = a.messages.count_tokens(model="local", messages=USER, extra_body=EXTRA)
    message = a.messages.create(model="local", messages=USER, max_tokens=16, extra_body=EXTRA)
    assert counted.input_tokens == message_input_tokens(message.usage)
    body = json.dumps({"input": "Reply with exactly OK.", "max_output_tokens": 16, **EXTRA}).encode()
    result = http.post("/v1/responses", content=gzip.compress(body), headers={"Content-Encoding": "gzip", "Content-Type": "application/json"})
    result.raise_for_status()
    assert result.json()["output"][0]["content"][0]["text"] == "OK"


def test_messages_tool_history(clients):
    _, a, _ = clients
    tools = [{"name": FUNCTION["name"], "description": FUNCTION["description"], "input_schema": FUNCTION["parameters"]}]
    history = [{"role": "user", "content": "Use lookup to look up the value for the query 'alpha'."}]
    with a.messages.stream(model="local", messages=history, tools=tools, max_tokens=256, extra_body=EXTRA) as stream:
        list(stream)
        result = stream.get_final_message()
    assert result.stop_reason == "tool_use"
    calls = [b for b in result.content if b.type == "tool_use"]
    assert len(calls) == 1 and calls[0].name == "lookup"
    history += [{"role": "assistant", "content": [b.model_dump() for b in result.content]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": calls[0].id, "content": "The result is 42."}]}]
    final = a.messages.create(model="local", messages=history, tools=tools, max_tokens=128, extra_body=EXTRA)
    assert "42" in "".join(b.text for b in final.content if b.type == "text")


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_structured_answer_with_reasoning_and_tools(clients, thinking, streaming):
    _, _, http = clients
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    }
    history = [{"role": "user", "content": (
        'Call lookup with exactly {"query":"alpha"} to retrieve the stored integer. '
        'Then return JSON with "value" set to that integer.'
    )}]
    for turn in range(3):
        response = http.post("/v1/chat/completions", json={
            "model": "local", "messages": history,
            "tools": [{"type": "function", "function": FUNCTION}],
            "tool_choice": "auto", "temperature": 1, "top_p": .95,
            "max_tokens": 2048, "stream": streaming,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answer", "strict": True, "schema": schema,
            }},
        })
        response.raise_for_status()
        if streaming:
            assert "data: [DONE]" in response.text, response.text
            message = {"role": "assistant", "content": "", "reasoning_content": ""}
            calls = {}
            for line in response.text.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                assert "error" not in chunk, chunk
                for choice in chunk.get("choices", []):
                    delta = choice["delta"]
                    for key in ("content", "reasoning_content"):
                        message[key] += delta.get(key) or ""
                    for call in delta.get("tool_calls", []):
                        dest = calls.setdefault(call["index"], {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if call.get("id"):
                            dest["id"] = call["id"]
                        for key in ("name", "arguments"):
                            dest["function"][key] += call.get("function", {}).get(key) or ""
            if calls:
                message["tool_calls"] = list(calls.values())
        else:
            message = response.json()["choices"][0]["message"]
        history.append(message)
        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                assert call["function"]["name"] == "lookup"
                assert json.loads(call["function"]["arguments"]) == {"query": "alpha"}
                history.append({"role": "tool", "tool_call_id": call["id"],
                                "content": '{"value":7391}'})
        else:
            assert turn > 0, message
            assert json.loads(message["content"]) == {"value": 7391}
            return
    pytest.fail("Model did not finish after receiving the lookup result")


@pytest.mark.parametrize("streaming", [False, True])
def test_required_tool_choice_sdk_contracts(clients, streaming):
    o, a, _ = clients
    history = [{"role": "user", "content": "Find the stored value for alpha."}]
    chat = o.chat.completions.create(
        model="local", messages=history, tools=[{"type": "function", "function": FUNCTION}],
        tool_choice="required", parallel_tool_calls=False,
        max_tokens=256, stream=streaming, extra_body=EXTRA,
    )
    if streaming:
        chunks = list(chat)
        assert any(c.choices[0].delta.tool_calls for c in chunks if c.choices)
        assert chunks[-1].choices[0].finish_reason == "tool_calls"
    else:
        assert chat.choices[0].message.tool_calls[0].function.name == "lookup"
        assert chat.choices[0].finish_reason == "tool_calls"
    kwargs = dict(
        model="local", messages=history,
        tools=[{"name": FUNCTION["name"], "description": FUNCTION["description"],
                "input_schema": FUNCTION["parameters"]}],
        tool_choice={"type": "any", "disable_parallel_tool_use": True},
        max_tokens=256, extra_body=EXTRA,
    )
    counted = a.messages.count_tokens(**{k: v for k, v in kwargs.items() if k != "max_tokens"})
    if streaming:
        with a.messages.stream(**kwargs) as stream:
            list(stream)
            message = stream.get_final_message()
    else:
        message = a.messages.create(**kwargs)
    assert message.stop_reason == "tool_use"
    assert any(b.type == "tool_use" and b.name == "lookup" for b in message.content)
    assert counted.input_tokens == message_input_tokens(message.usage)
    kwargs = dict(
        model="local", input=history, tools=[{"type": "function", **FUNCTION}],
        tool_choice="required", parallel_tool_calls=False,
        max_output_tokens=256, extra_body=EXTRA,
    )
    if streaming:
        with o.responses.stream(**kwargs) as stream:
            events = list(stream)
            response = stream.get_final_response()
        assert events[-1].type == "response.completed"
    else:
        response = o.responses.create(**kwargs)
    assert response.tool_choice == "required"
    assert response.parallel_tool_calls is False
    assert response.status == "completed"
    assert any(item.type == "function_call" and item.name == "lookup" for item in response.output)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"messages": USER, "max_tokens": 1, "tool_choice": "required",
                              "tools": [{"type": "function", "function": FUNCTION}]}),
    ("/v1/messages", {"messages": USER, "max_tokens": 1, "tool_choice": {"type": "any"},
                      "tools": [{"name": FUNCTION["name"], "input_schema": FUNCTION["parameters"]}]}),
    ("/v1/responses", {"input": USER, "max_output_tokens": 1, "tool_choice": "required",
                       "tools": [{"type": "function", **FUNCTION}]}),
])
def test_required_tool_choice_missing_call_is_a_terminal_error(clients, streaming, path, body):
    _, _, http = clients
    headers = {"anthropic-version": "2023-06-01"}
    response = http.post(path, json={**body, **EXTRA, "stream": streaming}, headers=headers)
    message = "required tool choice was not satisfied"
    if not streaming:
        assert response.status_code == 500
        assert message in response.json()["error"]["message"]
    else:
        response.raise_for_status()
        assert "[DONE]" not in response.text
        data = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        terminal = data[-1]
        if path == "/v1/responses":
            assert terminal["type"] == "response.failed"
            assert terminal["response"]["status"] == "failed"
            assert message in terminal["response"]["error"]["message"]
            assert not any(e["type"] in {"response.completed", "response.incomplete"} for e in data)
        else:
            assert message in terminal["error"]["message"]
            assert not any(e.get("type") == "message_stop" for e in data)
            assert not any(c.get("finish_reason") for e in data for c in e.get("choices", []))
    invalid = http.post(path, json={**body, **EXTRA, "tools": []}, headers=headers)
    assert invalid.status_code == 400
    assert "needs at least one tool" in invalid.json()["error"]["message"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol", ["chat/completions", "messages", "responses"])
def test_multiple_tool_calls_fail_when_parallel_calls_are_disabled(clients, protocol, streaming):
    _, _, http = clients
    history = [{"role": "user", "content": (
        "The queries alpha and beta are independent. Call lookup for BOTH queries in parallel "
        "in this single response. Do not wait for a result before issuing the second call."
    )}]
    body = {**EXTRA, "seed": 42, "max_tokens": 256, "messages": history,
            "tools": [{"type": "function", "function": FUNCTION}]}
    if protocol == "messages":
        body["tools"] = [{"name": FUNCTION["name"], "description": FUNCTION["description"],
                          "input_schema": FUNCTION["parameters"]}]
    elif protocol == "responses":
        body["input"] = body.pop("messages")
        body["max_output_tokens"] = body.pop("max_tokens")
        body["tools"] = [{"type": "function", **FUNCTION}]
    headers = {"anthropic-version": "2023-06-01"}
    allowed = http.post("/v1/" + protocol, json=body, headers=headers)
    allowed.raise_for_status()
    output = allowed.json()
    if protocol == "chat/completions":
        calls = output["choices"][0]["message"]["tool_calls"]
    elif protocol == "messages":
        calls = [b for b in output["content"] if b["type"] == "tool_use"]
    else:
        calls = [b for b in output["output"] if b["type"] == "function_call"]
    assert len(calls) == 2
    if protocol == "messages":
        body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    else:
        body["parallel_tool_calls"] = False
    response = http.post("/v1/" + protocol, json={**body, "stream": streaming}, headers=headers)
    if not streaming:
        assert response.status_code == 500
        assert "more than one tool call" in response.json()["error"]["message"]
    else:
        response.raise_for_status()
        assert "[DONE]" not in response.text
        data = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        terminal = data[-1]
        if protocol == "responses":
            assert terminal["type"] == "response.failed"
            error = terminal["response"]["error"]
            assert not any(e["type"] in {"response.completed", "response.incomplete"} for e in data)
            calls = [e for e in data if e["type"] == "response.output_item.added"
                     and e["item"]["type"] == "function_call"]
        elif protocol == "messages":
            error = terminal["error"]
            assert not any(e.get("type") == "message_stop" for e in data)
            calls = [e for e in data if e.get("type") == "content_block_start"
                     and e["content_block"]["type"] == "tool_use"]
        else:
            error = terminal["error"]
            assert not any(c.get("finish_reason") for e in data for c in e.get("choices", []))
            calls = [call for e in data for c in e.get("choices", [])
                     for call in c.get("delta", {}).get("tool_calls", []) if call.get("id")]
        assert "more than one tool call" in error["message"]
        assert len(calls) <= 1


def test_messages_inline_system_text_and_empty_controls(clients):
    _, a, _ = clients
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": [{"type": "text", "text": "Reply with exactly OK.",
         "cache_control": {"type": "ephemeral"}}], "output_config": {"effort": "xhigh"}},
    ]
    controls = {"role": "system", "content": [], "output_config": {"effort": "high"}}
    counted = a.messages.count_tokens(model="local", messages=history, extra_body=EXTRA)
    empty = a.messages.count_tokens(model="local", messages=history + [controls], extra_body=EXTRA)
    assert counted.input_tokens == empty.input_tokens
    with a.messages.stream(model="local", messages=history + [controls], max_tokens=32, extra_body=EXTRA) as stream:
        assert "".join(stream.text_stream) == "OK"
        result = stream.get_final_message()
    assert message_input_tokens(result.usage) == counted.input_tokens
    assert result.stop_reason == "end_turn"


def test_responses_tool_history(clients):
    o, _, _ = clients
    tools = [{"type": "function", **FUNCTION, "strict": False}]
    history = [{"role": "user", "content": "Use lookup to look up the value for the query 'alpha'."}]
    with o.responses.stream(model="local", input=history, tools=tools, max_output_tokens=256, extra_body=EXTRA) as stream:
        events = list(stream)
        result = stream.get_final_response()
    calls = [b for b in result.output if b.type == "function_call"]
    assert len(calls) == 1 and calls[0].name == "lookup"
    assert any(e.type == "response.function_call_arguments.delta" for e in events)
    history += [b.model_dump(exclude_none=True) for b in result.output]
    history += [{"type": "function_call_output", "call_id": calls[0].call_id, "output": "The result is 42."}]
    final = o.responses.create(model="local", input=history, tools=tools, max_output_tokens=128, extra_body=EXTRA)
    assert "42" in final.output_text


def test_responses_custom_tool_grammar(clients):
    o, _, http = clients
    tools = [{"type": "custom", "name": "record", "description": "Record the requested identifier as a plain string.",
              "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "ID-" /[0-9]+/'}}]
    history = [{"role": "user", "content": "Use record to record exactly ID-42."}]
    with o.responses.stream(model="local", input=history, tools=tools, max_output_tokens=256, extra_body=EXTRA) as stream:
        events = list(stream)
        result = stream.get_final_response()
    calls = [item for item in result.output if item.type == "custom_tool_call"]
    assert len(calls) == 1 and calls[0].input == "ID-42"
    assert any(e.type == "response.custom_tool_call_input.done" for e in events)
    history += [item.model_dump(exclude_none=True) for item in result.output]
    history += [{"type": "custom_tool_call_output", "call_id": calls[0].call_id, "output": "Recorded successfully."}]
    final = o.responses.create(model="local", input=history, tools=tools, max_output_tokens=64, extra_body=EXTRA)
    assert final.output_text
    tools[0]["format"]["definition"] = "start: missing_rule"
    invalid = http.post("/v1/responses", json={"input": "x", "tools": tools})
    assert invalid.status_code == 400 and "grammar" in invalid.json()["error"]["message"]


def test_reasoning_and_incomplete_output(clients):
    o, a, _ = clients
    with o.responses.stream(model="local", input="Compute 179 times 37 carefully.", max_output_tokens=16, extra_body={"enable_thinking": True}) as stream:
        events = list(stream)
        with pytest.raises(RuntimeError, match="response.completed"):
            stream.get_final_response()
        result = events[-1].response
    assert result.status == "incomplete"
    assert events[-1].type == "response.incomplete"
    assert any(b.type == "reasoning" for b in result.output)
    with a.messages.stream(model="local", messages=[{"role": "user", "content": "Compute 179 times 37 carefully."}], max_tokens=16, thinking={"type": "enabled", "budget_tokens": 1024}) as stream:
        list(stream)
        result = stream.get_final_message()
    assert result.stop_reason == "max_tokens"
    thinking = [b for b in result.content if b.type == "thinking"]
    assert thinking and thinking[0].signature.startswith("flare-thinking-v1:")


def test_stop_sequence_and_observations(clients):
    o, a, _ = clients
    result = a.messages.create(model="local", messages=USER, max_tokens=32, stop_sequences=["OK"], extra_body=EXTRA)
    assert result.stop_reason == "stop_sequence" and result.stop_sequence == "OK"
    extra = {**EXTRA, "return_token_ids": True, "return_token_timestamps": True}
    for _ in range(2):
        result = o.chat.completions.create(model="local", messages=USER, max_tokens=32, extra_body=extra).model_dump()
    assert result["reused_prompt_tokens"] > 0
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == result["reused_prompt_tokens"]
    assert len(result["choices"][0]["token_ids"]) == result["usage"]["completion_tokens"]
    assert len(result["choices"][0]["token_timestamps_ns"]) == result["usage"]["completion_tokens"]
    for include_usage in [False, True]:
        chunks = [c.model_dump() for c in o.chat.completions.create(
            model="local", messages=USER, max_tokens=32, extra_body=extra,
            stream=True, stream_options={"include_usage": include_usage},
        )]
        terminal = next(c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"])
        usage = chunks[-1]["usage"]
        assert usage["prompt_tokens_details"]["cached_tokens"] == terminal["reused_prompt_tokens"]
        assert terminal["reused_prompt_tokens"] > 0
        assert sum(len(c["choices"][0].get("token_ids", [])) for c in chunks if c["choices"]) == usage["completion_tokens"]


def test_unsupported_features_and_error_shapes(clients):
    _, _, http = clients
    for extra in [{"store": True}, {"previous_response_id": "r"}, {"tool_choice": {"type": "function", "name": "lookup"}}, {"tools": [{"type": "function", **FUNCTION, "strict": True}]}]:
        response = http.post("/v1/responses", json={"input": "x", **extra})
        assert response.status_code == 400 and response.json()["error"]["type"] == "invalid_request_error"
    response = http.post("/v1/messages", content=b"{", headers={"anthropic-version": "2023-06-01"})
    assert response.status_code == 400 and response.json()["type"] == "error"
    assert http.get("/v1/responses", headers={"Upgrade": "websocket", "Connection": "Upgrade"}).status_code == 400
    assert http.post("/v1/completions", json={"prompt": "x", "n": 100000}).status_code == 400


@pytest.mark.parametrize("temperature", [0., -0.])
@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"messages": USER, "max_tokens": 4}),
    ("/v1/completions", {"prompt": "Hello", "max_tokens": 0}),
    ("/v1/messages", {"messages": USER, "max_tokens": 4}),
    ("/v1/responses", {"input": USER, "max_output_tokens": 4}),
])
def test_zero_temperature_clamps_to_model_sampling_policy(clients, temperature, path, body):
    _, _, http = clients
    response = http.post(path, json={**body, "temperature": temperature},
                         headers={"anthropic-version": "2023-06-01"})
    assert response.status_code == 200, response.text
    assert "error" not in response.json() or response.json()["error"] is None


def test_mixed_endpoint_concurrency(clients):
    _, _, http = clients
    requests = [("/v1/chat/completions", {"messages": USER, "max_tokens": 16}), ("/v1/messages", {"messages": USER, "max_tokens": 16}), ("/v1/responses", {"input": USER, "max_output_tokens": 16}), ("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 4})]
    def call(pair):
        path, body = pair
        return http.post(path, json={**body, **EXTRA}, headers={"anthropic-version": "2023-06-01"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(call, requests))
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]


def test_batch_validation_and_disconnect_release_admission(clients):
    _, _, http = clients
    def metrics():
        response = http.get("/metrics")
        response.raise_for_status()
        return {key: float(value) for line in response.text.splitlines() if line and not line.startswith("#") for key, value in [line.rsplit(" ", 1)]}
    before = metrics()
    invalid = http.post("/v1/completions", json={"prompt": [[1], [4294967295]], "max_tokens": 8})
    assert invalid.status_code == 400
    assert metrics()["flare_requests_admitted_total"] == before["flare_requests_admitted_total"]
    with http.stream("POST", "/v1/completions", json={"prompt": ["Count the positive integers:"], "n": 4, "max_tokens": 4096, "ignore_eos": True, "stream": True, "temperature": 1}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                assert "error" not in json.loads(line[6:])
                break
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        after = metrics()
        if after["flare_active_requests"] == 0 and after["flare_http_inflight_requests"] == 0:
            break
        time.sleep(0.02)
    assert after["flare_active_requests"] == 0 and after["flare_http_inflight_requests"] == 0
    assert after["flare_requests_cancelled_total"] > before["flare_requests_cancelled_total"]
    followup = http.post("/v1/responses", json={"input": USER, "max_output_tokens": 16, **EXTRA})
    followup.raise_for_status()


def test_authenticated_protocol_headers(clients):
    if not os.environ.get("FLARE_API_TEST_KEY"):
        pytest.skip("requires a service configured with an API key")
    import httpx
    key = os.environ["FLARE_API_TEST_KEY"]
    with httpx.Client(base_url=os.environ["FLARE_API_TEST_URL"], timeout=30) as http:
        assert http.get("/health").status_code == 200
        for path in ["/v1/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens"]:
            result = http.post(path, json={})
            assert result.status_code == 401
            assert result.json()["error"]["type"] == "authentication_error"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {"messages": USER, **EXTRA}
        assert http.post("/v1/messages/count_tokens", json=body, headers=headers).status_code == 200
        headers["Authorization"] = "Bearer incorrect"
        assert http.post("/v1/messages/count_tokens", json=body, headers=headers).status_code == 401
        assert http.post("/v1/completions", json={"prompt": "x", "max_tokens": 0}, headers={"x-api-key": key}).status_code == 401


def test_image_input(clients):
    o, a, _ = clients
    image = Path(__file__).parents[1] / "frontend/qwen38_images-inputs/RGB.png"
    data = base64.b64encode(image.read_bytes()).decode()
    result = a.messages.create(model="local", max_tokens=48, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}, {"type": "text", "text": "Describe the image briefly."}]}], extra_body=EXTRA)
    assert any(b.type == "text" and b.text for b in result.content)
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        {"type": "text", "text": "Describe the image briefly."},
    ]}]
    extra = {**EXTRA, "return_token_ids": True}
    buffered = o.chat.completions.create(model="local", messages=messages, max_tokens=8, extra_body=extra)
    chunks = list(o.chat.completions.create(
        model="local", messages=messages, max_tokens=8, extra_body=extra,
        stream=True, stream_options={"include_usage": True},
    ))
    for usage in [buffered.usage, chunks[-1].usage]:
        assert 0 < usage.prompt_tokens_details.multimodal_tokens["image"] <= usage.prompt_tokens
        assert usage.prompt_tokens == message_input_tokens(result.usage)
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def _logprob_package(http):
    card = http.get('/v1/models').json()['data'][0]
    if not card['permission'][0]['allow_logprobs']:
        response = http.post('/v1/chat/completions', json={'messages': USER, 'max_tokens': 1, 'logprobs': True})
        assert response.status_code == 400
        assert 'not available in this model package' in response.text
        pytest.skip('package predates generation logprobs')
    assert card['flare']['max_top_logprobs'] == 20


def _check_logprob_entry(entry, top):
    assert entry['token'].startswith('token_id:')
    assert isinstance(entry['logprob'], (int, float)) and entry['logprob'] <= 0
    assert all(isinstance(b, int) and 0 <= b < 256 for b in entry['bytes'])
    alternatives = entry['top_logprobs']
    assert len(alternatives) == top
    keys = [(-a['logprob'], int(a['token'].split(':')[1])) for a in alternatives]
    assert keys == sorted(keys)


@pytest.mark.parametrize('streaming', [False, True])
def test_generation_logprobs_single_token_continuation(clients, streaming):
    _, _, http = clients
    _logprob_package(http)
    body = {'messages': [{'role': 'user', 'content': 'Choose A or B. Reply with only the letter.'},
                         {'role': 'assistant', 'content': 'Choice:'}],
            'max_tokens': 1, 'continue_final_message': True, 'add_generation_prompt': False,
            'logprobs': True, 'top_logprobs': 20, 'return_tokens_as_token_ids': True,
            'stream': streaming}
    if streaming:
        entries = []
        with http.stream('POST', '/v1/chat/completions', json=body) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    data = json.loads(line[6:])
                    for choice in data.get('choices', []):
                        entries.extend((choice.get('logprobs') or {}).get('content', []))
    else:
        response = http.post('/v1/chat/completions', json=body)
        assert response.status_code == 200, response.text
        entries = response.json()['choices'][0]['logprobs']['content']
    assert len(entries) == 1
    _check_logprob_entry(entries[0], 20)


@pytest.mark.parametrize('streaming', [False, True])
def test_completion_logprob_arrays_and_offsets(clients, streaming):
    _, _, http = clients
    _logprob_package(http)
    body = {'prompt': 'Once upon a time', 'max_tokens': 4, 'ignore_eos': True,
            'logprobs': 3, 'return_tokens_as_token_ids': True, 'stream': streaming}
    parts = []
    if streaming:
        with http.stream('POST', '/v1/completions', json=body) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    for choice in json.loads(line[6:]).get('choices', []):
                        if choice.get('logprobs') is not None:
                            parts.append(choice['logprobs'])
    else:
        response = http.post('/v1/completions', json=body)
        assert response.status_code == 200, response.text
        parts.append(response.json()['choices'][0]['logprobs'])
    tokens = [token for part in parts for token in part['tokens']]
    offsets = [offset for part in parts for offset in part['text_offset']]
    assert len(tokens) == len(offsets) == 4
    assert offsets[0] == 0 and offsets == sorted(offsets)
    for part in parts:
        assert len(part['tokens']) == len(part['token_logprobs']) == len(part['top_logprobs']) == len(part['text_offset'])
        assert all(token.startswith('token_id:') for token in part['tokens'])
        assert all(len(top) == 3 for top in part['top_logprobs'])


@pytest.mark.parametrize('streaming', [False, True])
def test_responses_logprobs_belong_to_output_text(clients, streaming):
    _, _, http = clients
    _logprob_package(http)
    body = {'input': USER, 'max_output_tokens': 16, 'include': ['message.output_text.logprobs'],
            'top_logprobs': 3, 'return_tokens_as_token_ids': True, 'stream': streaming, **EXTRA}
    deltas = []
    if streaming:
        final = None
        with http.stream('POST', '/v1/responses', json=body) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                event = json.loads(line[6:])
                if event['type'] == 'response.output_text.delta':
                    deltas.extend(event['logprobs'])
                if event['type'] == 'response.completed' or event['type'] == 'response.incomplete':
                    final = event['response']
        assert final is not None
    else:
        response = http.post('/v1/responses', json=body)
        assert response.status_code == 200, response.text
        final = response.json()
    entries = []
    for item in final['output']:
        for part in item.get('content', []):
            if part['type'] == 'output_text':
                entries.extend(part['logprobs'])
            else:
                assert 'logprobs' not in part
    for entry in entries:
        _check_logprob_entry(entry, 3)
    if streaming:
        assert entries == deltas


def test_logprob_validation_preserves_other_request_semantics(clients):
    _, _, http = clients
    for route, body in [
        ('/v1/chat/completions', {'messages': USER, 'top_logprobs': 1}),
        ('/v1/chat/completions', {'messages': USER, 'logprobs': True, 'top_logprobs': 21}),
        ('/v1/completions', {'prompt': 'x', 'logprobs': 0, 'echo': True}),
        ('/v1/responses', {'input': USER, 'top_logprobs': 1}),
        ('/v1/messages', {'messages': USER, 'max_tokens': 1, 'top_logprobs': 0}),
    ]:
        response = http.post(route, json=body, headers={'anthropic-version': '2023-06-01'})
        assert response.status_code == 400, response.text


def test_concurrent_speculative_logprobs_keep_request_lengths_and_flags_separate(clients):
    _, _, http = clients
    _logprob_package(http)
    for round in range(2):
        def generate(index):
            length = (1, 7, 23, 9)[index]
            requested = (index + round) % 2 == 0
            body = {'prompt': 'Count the integers starting at one:', 'max_tokens': length,
                    'ignore_eos': True, 'top_k': 1, 'temperature': 1, 'seed': 42}
            if requested:
                body.update(logprobs=5, return_tokens_as_token_ids=True)
            response = http.post('/v1/completions', json=body)
            assert response.status_code == 200, response.text
            choice = response.json()['choices'][0]
            if requested:
                values = choice['logprobs']
                assert len(values['tokens']) == len(values['token_logprobs']) == length
                assert len(values['top_logprobs']) == len(values['text_offset']) == length
                assert all(len(top) == 5 for top in values['top_logprobs'])
            else:
                assert choice['logprobs'] is None
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(generate, range(4)))


@pytest.mark.parametrize('protocol', ['chat', 'completions', 'responses'])
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('count', [0, 1, 20, 21, 127, 128])
def test_requested_logprob_ids_replace_natural_alternatives(clients, protocol, streaming, count):
    _, _, http = clients
    ids = list(reversed(range(count)))
    body = {'max_tokens': 1, 'return_tokens_as_token_ids': True,
            'logprob_token_ids': ids + (ids[:1] if count < 128 else []), 'stream': streaming,
            'ignore_eos': True, **EXTRA}
    if protocol == 'completions':
        path = '/v1/completions'
        body.update(prompt='Answer:', logprobs=1)
    elif protocol == 'chat':
        path = '/v1/chat/completions'
        body.update(messages=[{'role': 'user', 'content': 'Pick a letter.'},
                              {'role': 'assistant', 'content': 'Answer:'}],
                    continue_final_message=True, add_generation_prompt=False,
                    logprobs=True, top_logprobs=1)
    else:
        path = '/v1/responses'
        body.pop('max_tokens')
        body.update(input='Pick a letter.', max_output_tokens=1, top_logprobs=1,
                    include=['message.output_text.logprobs'])
    frames = []
    if streaming:
        with http.stream('POST', path, json=body) as response:
            assert response.status_code == 200, response.read()
            for line in response.iter_lines():
                if line.startswith('data: ') and line[6:] != '[DONE]':
                    frames.append(json.loads(line[6:]))
    else:
        response = http.post(path, json=body)
        assert response.status_code == 200, response.text
        frames = [response.json()]
    entries = []
    for frame in frames:
        if protocol == 'responses':
            if streaming and frame.get('type') == 'response.output_text.delta':
                entries.extend(frame['logprobs'])
            elif not streaming:
                entries.extend(entry for item in frame['output']
                               for part in item.get('content', [])
                               for entry in part.get('logprobs', []))
        else:
            for choice in frame.get('choices', []):
                values = choice.get('logprobs') or {}
                if protocol == 'chat':
                    entries.extend(values.get('content', []))
                else:
                    entries.extend({'token': token, 'top_logprobs': top}
                                   for token, top in zip(values.get('tokens', []),
                                                         values.get('top_logprobs', [])))
    if protocol != 'responses':
        assert len(entries) == 1
    for entry in entries:
        top = entry['top_logprobs']
        actual = list(top) if isinstance(top, dict) else [item['token'] for item in top]
        expected = [f'token_id:{id}' for id in ids]
        if ids:
            if entry['token'] not in expected:
                expected.append(entry['token'])
            assert actual == expected
        else:
            assert len(actual) == 1
