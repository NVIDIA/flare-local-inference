"""Namespace-delimited tools and explicit reasoning blocks for MiniMax M3."""

import json
from pathlib import Path

from tokenizers import Tokenizer

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend


FRONTEND = c.frontend(
    tools=p.MINIMAX_XML,
    assistant_marker="]~b]ai\n",
    message_end_markers=["[e~["],
    prefix_boundary=c.prefix_boundary(
        ["system", "developer"],
        append_role="user",
        minimum_messages=0,
        required_start="]~!b[",
        required_marker="[e~[",
    ),
)
FRONTEND["image_preparation"] = p.PATCH_MERGE_IMAGE_PREPARATION
FRONTEND["image_layout"] = p.BRACKETED_IMAGE_LAYOUT
FRONTEND["image_tokens"] = dict(feature_token=200025, span=None)
FRONTEND["rejected_tokens"] = [200026]
FRONTEND["reasoning"] = dict(
    kind=p.MM_THINK,
    open="<mm:think>",
    close="</mm:think>",
    wrapping_newlines=0,
    terminal_markers=[],
)


def prepare(snapshot):
    snapshot = Path(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    settings = json.loads((snapshot / "tokenizer_config.json").read_text())
    end = settings.get("eos_token")
    if isinstance(end, dict):
        end = end.get("content")
    if end != "[e~[":
        raise ValueError("MiniMax M3 requires the [e~[ end-of-message token")
    markers = (end, "]~b]", "]~!b[", "<mm:think>", "</mm:think>", "]<]minimax[>[")
    if any(tokenizer.token_to_id(marker) is None for marker in markers):
        raise ValueError("MiniMax M3 tokenizer is missing an envelope control token")
    settings = prepare_frontend(snapshot, eos_token_ids=[tokenizer.token_to_id(end)])
    image = json.loads((snapshot / "preprocessor_config.json").read_text())
    if (
        image.get("patch_size") != 14
        or image.get("process_image_mode") != "dynamic_res"
        or image.get("image_mean") != [0.48145466, 0.4578275, 0.40821073]
        or image.get("image_std") != [0.26862954, 0.26130258, 0.27577711]
        or image.get("size") != [672, 672]
        or image.get("add_start_end_special_tokens") is not True
    ):
        raise ValueError("unsupported MiniMax M3 image preprocessing configuration")
    compression = image["img_token_compression_config"]
    if any(
        compression.get(name) != value
        for name, value in {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }.items()
    ):
        raise ValueError(
            "MiniMax M3 images require temporal duplication and spatial patch merging"
        )
    for marker in ("]<]image[>[", "]<]start of image[>[", "]<]end of image[>["):
        if tokenizer.token_to_id(marker) is None:
            raise ValueError("MiniMax M3 tokenizer is missing an image control token")
    if (
        tokenizer.token_to_id("]<]image[>[") != 200025
        or tokenizer.token_to_id("]<]video[>[") != 200026
    ):
        raise ValueError(
            "MiniMax M3 image/video token IDs do not match the prepared layout"
        )
    settings["image"] = {
        "patch_merge": dict(
            patch_size=14,
            min_pixels=3136,
            max_pixels=451584,
            image_mean=image["image_mean"],
            image_std=image["image_std"],
            image_token="]<]image[>[",
            start_token="]<]start of image[>[",
            end_token="]<]end of image[>[",
        )
    }
    return settings


def runtime_assets(snapshot):
    snapshot = Path(snapshot)
    return {"tokenizer": {"path": str(snapshot / "tokenizer.json")},
            "chat_template": annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
                Path(__file__).with_name("assets") / "chat_template.jinja")}
