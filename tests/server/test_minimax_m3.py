"""Native MiniMax M3 text, tools, images, and complete continuation requests.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_MINIMAX_TEST_URL to an idle TP4 service
with concurrency four, a 4096-token prefill chunk, and image support. Run against
both DSpark and ordinary decoding when validating a prepared package.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_MINIMAX_TEST_URL"),
    reason="requires a native MiniMax M3 service",
)


@pytest.fixture(scope="module")
def service():
    url = os.environ["FLARE_MINIMAX_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    assert "minimax" in model.lower(), model

    def ask(content, **options):
        body = dict(model=model, messages=[dict(role="user", content=content)], temperature=1, top_k=1,
                    seed=1234, max_tokens=512, return_token_ids=True,
                    chat_template_kwargs=dict(thinking_mode="disabled"))
        body.update(options)
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if body.get("stream"):
                    events = []
                    for line in response:
                        if line.startswith(b"data: "):
                            payload = line.removeprefix(b"data: ").strip()
                            if payload == b"[DONE]":
                                return events
                            events.append(json.loads(payload))
                    pytest.fail("stream ended without [DONE]")
                result = json.load(response)
        except urllib.error.HTTPError as error:
            pytest.fail(f"HTTP {error.code}: {error.read().decode()}")
        assert result["choices"][0]["finish_reason"] != "length", result
        assert result["usage"]["completion_tokens"] == len(result["choices"][0]["token_ids"])
        return result

    return ask


def answer(result):
    return (result["choices"][0]["message"].get("content") or "").strip()


def image(color, label, size=(672, 672)):
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(picture)
    draw.rectangle((24, 200, 648, 320), fill="white")
    draw.text((40, 220), label, fill="black", font=ImageFont.load_default(size=64))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return dict(type="image_url", image_url=dict(url="data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode()))


def test_text_and_concurrent_requests(service):
    assert answer(service("What is 17 times 23? Reply with only the integer.")) == "391"
    prompts = [f"Repeat exactly this code and nothing else: FLARE{i}128" for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(service, prompts))
    for index, result in enumerate(results):
        assert answer(result) == f"FLARE{index}128"


@pytest.mark.parametrize("mode", ["enabled", "disabled", "adaptive"])
def test_reasoning_modes(service, mode):
    result = service("What is 17 times 23? Put only the integer in the final answer.",
                     chat_template_kwargs=dict(thinking_mode=mode), max_tokens=1024)
    message = result["choices"][0]["message"]
    assert answer(result) == "391"
    if mode == "enabled":
        assert message.get("reasoning_content", "").strip()
    if mode == "disabled":
        assert not message.get("reasoning_content")
    assert "<mm:think>" not in answer(result) and "</mm:think>" not in answer(result)


def test_streamed_reasoning_and_answer(service):
    events = service("What is 17 times 23? Put only the integer in the final answer.",
        chat_template_kwargs=dict(thinking_mode="enabled"), max_tokens=1024,
        stream=True, stream_options=dict(include_usage=True))
    choices = [choice for event in events for choice in event["choices"]]
    reasoning = "".join(choice.get("delta", {}).get("reasoning_content") or "" for choice in choices)
    content = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
    assert reasoning.strip() and content.strip() == "391"
    assert choices[-1]["finish_reason"] == "stop"
    assert any(event.get("usage", {}).get("completion_tokens", 0) > 0
               for event in events if event.get("usage"))


def test_xml_tool_call_and_result(service):
    tools = [dict(type="function", function=dict(name="lookup_weather", description="Get a city's weather.",
        parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))]
    prompt = "Use lookup_weather to get the weather in Paris."
    result = service(prompt, tools=tools, tool_choice="required")
    message = result["choices"][0]["message"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    call, = message["tool_calls"]
    assert call["function"]["name"] == "lookup_weather"
    assert json.loads(call["function"]["arguments"])["city"] == "Paris"
    result = service("", tools=tools, messages=[dict(role="user", content=prompt), message,
        dict(role="tool", tool_call_id=call["id"], content='{"temperature_c": 21, "condition": "sunny"}')])
    assert "21" in answer(result)


def test_structured_response(service):
    schema = dict(type="object", properties=dict(result=dict(type="integer")), required=["result"],
                  additionalProperties=False)
    result = service("Return the product of 17 and 23 as result.",
        response_format=dict(type="json_schema", json_schema=dict(name="product", strict=True, schema=schema)))
    assert json.loads(answer(result)) == {"result": 391}


def test_images_and_chunked_prefill(service):
    content = [image("red", "FIRST 128"), image("blue", "SECOND 256"),
        dict(type="text", text="Read both image labels in their original order. Only output the labels.")]
    for prefix in ([], [dict(type="text", text="filler " * 4100)]):
        response = service(prefix + content)
        assert response["usage"]["prompt_tokens_details"]["multimodal_tokens"]["image"] == 1152
        if prefix:
            assert response["usage"]["prompt_tokens"] > 4096
        text = answer(response)
        assert "FIRST 128" in text and "SECOND 256" in text
        assert text.index("FIRST 128") < text.index("SECOND 256")


def test_long_context_reused_prefix_and_parallel_branches(service):
    prefix = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(1200))
    question = "\nWhat code does item 211 record? Reply with only the integer."
    first, second = service(prefix + question), service(prefix + question)
    assert first["usage"]["prompt_tokens"] > 4096
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert first["choices"][0]["token_ids"] == second["choices"][0]["token_ids"]
    assert answer(first) == answer(second) == "596"
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(service, [prefix + question] * 4))
    assert all(answer(result) == "596" for result in results)


def test_decode_across_sparse_page_boundaries(service):
    result = service("Count from 1 through 160 inclusive, separated by commas. Output only the numbers, "
                     "with no omissions.", max_tokens=2048)
    assert result["usage"]["completion_tokens"] > 256
    assert [int(value) for value in re.findall(r"\d+", answer(result))] == list(range(1, 161))
