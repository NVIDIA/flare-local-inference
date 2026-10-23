"""Native MiMo patches and prompt expansion match the Transformers image processor."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from ignition.recipes.mimo_v2.recipe import RECIPE

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = (Path.home() / '.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Pro-MOPD'
            / 'snapshots' / RECIPE.checkpoints[0].revision)


def test_native_mimo_chat_matches_transformers(tmp_path):
    if not (SNAPSHOT / 'tokenizer_config.json').is_file():
        pytest.skip('MiMo checkpoint frontend is not cached')
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'frontend,test-tools', '--bin', 'test_frontend'], check=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    user = dict(role='user', content='What is 17 times 23?')
    cases = []

    def add(name, messages, **options):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        cases.append(dict(name=name, messages=messages, options=options, rendered=rendered,
                          token_ids=tokenizer.encode(rendered, add_special_tokens=False)))

    add('default', [user], add_generation_prompt=False)
    add('thinking_disabled', [user], add_generation_prompt=True, enable_thinking=False)
    add('thinking_enabled', [user], add_generation_prompt=True, enable_thinking=True)
    for tools in (None, [], [dict(type='function', function=dict(name='lookup', parameters={}))]):
        add(f'tools_{tools}', [user], add_generation_prompt=True, tools=tools)
    add('assistant_null_fields', [user, dict(role='assistant', content=None, tool_calls=None), user],
        add_generation_prompt=True)
    add('images', [dict(role='user', content=[dict(type='image'), dict(type='text', text='Describe.')])],
        add_generation_prompt=True)
    add('tool_history', [user, dict(role='assistant', content='', tool_calls=[dict(type='function',
        function=dict(name='lookup', arguments=dict(city='Paris', count=2)))]),
        dict(role='tool', content='Sunny.')], add_generation_prompt=True)
    from ignition.preparation.frontend import annotated_template_asset
    annotated = annotated_template_asset((SNAPSHOT / 'chat_template.jinja').read_bytes(),
        ROOT / 'ignition/recipes/mimo_v2/assets/chat_template.jinja')
    native_assets = tmp_path / 'assets'
    native_assets.mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (native_assets / name).symlink_to(SNAPSHOT / name)
    (native_assets / 'chat_template.jinja').symlink_to(annotated['path'])
    assets = {name: hashlib.sha256((native_assets / name).read_bytes()).hexdigest()
              for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    fixture = tmp_path / 'chat.json'
    fixture.write_text(json.dumps(dict(assets=assets, cases=cases, encode_cases=[], decode_cases=[])))
    subprocess.run([str(ROOT / 'helios/target/debug/test_frontend'), str(native_assets), str(fixture)], check=True)


@pytest.fixture(scope='module')
def processor():
    if not (SNAPSHOT / 'preprocessor_config.json').is_file():
        pytest.skip('MiMo checkpoint frontend is not cached')
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'server,test-tools', '--bin', 'test_mimo_images'], check=True)
    return Qwen2VLImageProcessorPil.from_pretrained(SNAPSHOT, local_files_only=True)


@pytest.mark.parametrize('sizes', [[(1, 1)], [(64, 64)], [(701, 503)], [(3000, 47)],
                                  [(17, 3000)], [(503, 701), (149, 31)]])
def test_native_mimo_images_match_transformers(processor, tmp_path, sizes):
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / 'tokenizer.json'))
    prompt = 'Describe ' + '<|vision_start|><|image_pad|><|vision_end|>' * len(sizes)
    inputs, images = [], []
    for index, (width, height) in enumerate(sizes):
        rgb = np.random.default_rng(index + 51).integers(0, 256, (height, width, 3), dtype=np.uint8)
        path = tmp_path / f'image{index}.rgb'
        path.write_bytes(rgb.tobytes())
        inputs.append(dict(width=width, height=height, pixels=str(path)))
        images.append(Image.fromarray(rgb))
    expected = processor(images=images, return_tensors='pt')
    grids = expected['image_grid_thw'].tolist()
    counts = [h * w for _, h, w in grids]
    positions, cu = [], [0]
    for _, h, w in grids:
        positions.extend([[y + dy, x + dx] for y in range(0, h, 2) for x in range(0, w, 2)
                          for dy in range(2) for dx in range(2)])
        cu.append(cu[-1] + h * w)
    tokens, indices, offset, image = [], [], 0, 0
    for token in tokenizer.encode(prompt, add_special_tokens=False).ids:
        if token != 151655:
            tokens.append(token)
            indices.append(-1)
        else:
            count = counts[image] // 4
            tokens.extend([token] * count)
            indices.extend(range(offset, offset + count))
            offset += count
            image += 1
    case = tmp_path / 'case.json'
    case.write_text(json.dumps(dict(images=inputs, prompt=prompt)))
    subprocess.run([str(ROOT / 'helios/target/debug/test_mimo_images'), str(SNAPSHOT), str(ROOT / 'ignition/recipes/mimo_v2/assets'),
                    str(case), str(tmp_path)], check=True)
    actual = json.loads((tmp_path / 'metadata.json').read_text())
    assert actual == dict(tokens=tokens, indices=indices, grids=grids, patches=counts,
                          positions=positions, cu_seqlens=cu)
    pixels = torch.from_numpy(np.fromfile(tmp_path / 'pixels.f32', dtype=np.float32))
    torch.testing.assert_close(pixels, expected['pixel_values'].flatten(), rtol=0, atol=5e-7)
