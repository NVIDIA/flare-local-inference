"""GLM text and image answers through native HTTP on one fully provisioned GPU."""

import base64
import io
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_FULL_MODEL_CUDA_TESTS") != "1",
    reason="requires an assigned GPU holding the complete GLM checkpoint and qualification heliopause",
)


def _image(color, side=112):
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (side, side), color).save(stream, format="PNG")
    data = base64.b64encode(stream.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def check_image_answers(address, model):
    """Exercise image identity, exact-prefix reuse, chunking, and request-slot reuse."""
    color = {"type": "text", "text": "What is the dominant color of the image? Answer with one color word."}
    cases = [
        ("text", [{"type": "text", "text": "What is 7 plus 5? Answer with just the number."}], ("12",)),
        ("red", [color, _image("red")], ("red",)),
        ("blue", [color, _image("blue")], ("blue",)),
        ("red-repeat", [color, _image("red")], ("red",)),
        ("two-images", [{"type": "text", "text": "Name the dominant colors of the first and second images, in that order."},
                        _image("red"), _image("blue")], ("red", "blue")),
        ("chunked-blue", [color, _image("blue", 784)], ("blue",)),
        ("text-after-images", [{"type": "text", "text": "What is 9 plus 6? Answer with just the number."}], ("15",)),
    ]
    answers = {}
    for name, content, expected in cases:
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "temperature": 1., "seed": 0, "max_tokens": 192, "chat_template_kwargs": {"reasoning_effort": "low"}}
        request = Request(f"http://{address}/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=240) as response:
            result = json.load(response)
        answer = result["choices"][0]["message"]["content"].strip().lower()
        assert result["choices"][0]["finish_reason"] == "stop", (name, result)
        assert all(re.search(r"\b" + re.escape(word) + r"\b", answer) for word in expected), (name, result)
        if len(expected) == 2:
            assert answer.index(expected[0]) < answer.index(expected[1]), result
        if name == "chunked-blue":
            assert result["usage"]["prompt_tokens"] > 512, result
        answers[name] = answer
    assert answers["red"] == answers["red-repeat"]
    return answers


@pytest.mark.parametrize("mtp", [False, True])
def test_native_glm_image_answers(mtp, record_property):
    import flare_heliopause
    from ignition.recipes.glm53.bootstrap import PreparationConfig, prepare_rank

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a GPU explicitly"
    assert hasattr(flare_heliopause, "Server"), "build heliopause with its qualification feature"
    snapshot = Path(os.environ["FLARE_GLM53_SNAPSHOT"])
    config = PreparationConfig(devices=(0,), context=8192, prefill_chunk_size=512, max_active=1,
                               pages=128, snapshot_budget_bytes=2 << 30, headroom_bytes=4 << 30,
                               compile_decoder=True, mtp=mtp)
    with prepare_rank(snapshot, config, rank=0, group=None) as prepared:
        server = prepared.start_server(bind="127.0.0.1:0", template_defaults={"reasoning_effort": "low"})
        answers = check_image_answers(server.address, prepared.server_config["model_id"])
        record_property("answers", json.dumps(answers))
