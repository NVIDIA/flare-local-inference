"""Generate GLM5Next image preprocessing fixtures from the canonical processor."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.vision_utils import get_vision_position_ids


def pattern(width, height, seed):
    # Vectorized channel ramps exercise bicubic interpolation and black padding.
    y, x = np.indices((height, width))
    return np.stack(((x * 7 + seed * 31) % 256, (y * 11) % 256, (x + y * 3) % 256), -1).astype(np.uint8)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def generate(snapshot, output):
    raw = json.loads((snapshot / "processor_config.json").read_text())
    model = json.loads((snapshot / "config.json").read_text())
    processor = Glm5NextImageProcessor(**raw["image_processor"])
    cases = []
    for seed, (width, height) in enumerate([(1, 1), (19, 37), (224, 224), (271, 283),
                                           (640, 360), (4000, 2400), (6000, 1)]):
        pixels = pattern(width, height, seed)
        result = processor(images=[Image.fromarray(pixels)], return_tensors="pt")
        resized = processor.resize(torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0),
            resample=processor.resample, factor=28, temporal_factor=2,
            min_image_tokens=16, max_image_tokens=8000)[0].permute(1, 2, 0)
        cases.append(dict(width=width, height=height, seed=seed,
            grid=result.image_grid_thw[0].tolist(), pixels=digest(result.pixel_values),
            resized=digest(resized), positions=digest(get_vision_position_ids(result.image_grid_thw, 2))))
    profile = {key: model[key] for key in ("vision_config", "model_type", "image_token_id", "video_token_id")}
    output.write_text(json.dumps(dict(processor=raw, model=profile, cases=cases), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.snapshot, args.output)
