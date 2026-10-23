"""Compare native segmented XTML rendering with the installed K3 checkpoint encoder."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

CHECKPOINT = Path('/data/models/Kimi-K3-MXFP4-CSF')
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def reference():
    if not CHECKPOINT.is_dir():
        pytest.skip('Kimi K3 checkpoint is not installed')
    import tiktoken
    from ignition.recipes.kimi3.tokenizer import PATTERN, _Converter, convert

    spec = importlib.util.spec_from_file_location('kimi3_reference_encoding', CHECKPOINT / 'encoding_k3.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = convert(CHECKPOINT)
    encoding = tiktoken.Encoding(name='kimi3', pat_str=PATTERN,
        mergeable_ranks=_Converter.load_tiktoken_bpe(CHECKPOINT / 'tiktoken.model'),
        special_tokens={tokenizer.id_to_token(i): i for i in range(163584, 163840)})
    return module, tokenizer, encoding


def test_tokenizer_matches_byte_ranked_checkpoint(reference):
    from tokenizers import Tokenizer
    module, tokenizer, encoding = reference
    raw = json.loads(tokenizer.to_str())
    raw['added_tokens'] = []
    ordinary = Tokenizer.from_str(json.dumps(raw))
    samples = ['Hello, world!', '你好世界，中文 English混合测试', 'éÉß ΑΒΓ δ 日本語 한국어',
        '<|open|>message role="user"<|sep|>[EOS]', '123456789\n  \n ', "don't I'm we've",
        'a\u0301 👨‍👩‍👧‍👦🦜\r\n', (CHECKPOINT / 'README.md').read_text()]
    for text in samples:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(text, allowed_special='all')
        assert ordinary.encode(text, add_special_tokens=False).ids == encoding.encode(text, disallowed_special=())


def test_native_xtml_matches_checkpoint(reference, tmp_path):
    module, tokenizer, encoding = reference
    binary = ROOT / 'helios/target/debug/test_frontend'
    if not binary.is_file():
        pytest.skip('build test_frontend with frontend,test-tools features')
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    shutil.copyfile(CHECKPOINT / 'tokenizer_config.json', tmp_path / 'tokenizer_config.json')
    shutil.copyfile(ROOT / 'ignition/recipes/kimi3/assets/chat_template.jinja', tmp_path / 'chat_template.jinja')
    fixtures = dict(template_encoding=1, cases=[], encode_cases=[], decode_cases=[], assets={})
    for path in tmp_path.iterdir():
        fixtures['assets'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Search documents',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]
    calls = [{'id': 'opaque-b', 'function': {'name': 'lookup', 'arguments': {
        'query': '中文 <|close|> "quoted"', 'count': 3, 'score': 1.5, 'enabled': True,
        'optional': None, 'list': [1, 'x'], 'object': {'z': False, 'a': []}}}},
        {'id': 'opaque-a', 'function': {'name': 'second', 'arguments': {}}}]
    history = 'Repeated "history"\\\n café 世界 😀 <|close|>'
    cases = [
        ('user', [{'role': 'user', 'content': 'Hello'}], {}),
        ('literal', [{'role': 'user', 'name': 'x"&<|open|>', 'content': '<|open|>think<|sep|>[EOS]'}], {}),
        ('history', [{'role': 'system', 'content': 'Be precise'}, {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'reasoning_content': 'Let me think', 'content': 'Hi'},
            {'role': 'user', 'content': 'Continue'}], {}),
        ('tools', [{'role': 'user', 'content': 'Search'}, {'role': 'assistant', 'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': 'opaque-a', 'name': 'stale', 'content': 'second result'},
            {'role': 'tool', 'tool_call_id': 'opaque-b', 'content': 'first result'}], {'tools': tools}),
        ('required', [{'role': 'user', 'content': 'Search'}], {'tools': tools, 'tool_choice': 'required'}),
        ('none', [{'role': 'user', 'content': 'Answer'}], {'tools': tools, 'tool_choice': 'none'}),
        ('thinking_disabled', [{'role': 'user', 'content': 'Hello'}], {'enable_thinking': False}),
        ('thinking_alias_disabled', [{'role': 'user', 'content': 'Hello'}], {'thinking': False}),
        ('thinking_disabled_history', [{'role': 'user', 'content': 'Search'},
            {'role': 'assistant', 'reasoning_content': 'Look up the documents.', 'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': 'opaque-b', 'content': 'first result'},
            {'role': 'tool', 'tool_call_id': 'opaque-a', 'content': 'second result'}],
            {'tools': tools, 'enable_thinking': False, 'thinking': False, 'reasoning_effort': 'low'}),
        ('thinking_enabled', [{'role': 'user', 'content': 'Hello'}], {'enable_thinking': True}),
        ('default', [{'role': 'user', 'content': 'Hello'}], {'enable_thinking': True, 'add_generation_prompt': False}),
        ('low', [{'role': 'user', 'content': 'Hello'}], {'reasoning_effort': 'low'}),
        ('json', [{'role': 'user', 'content': 'Return JSON'}], {'response_format': {'type': 'json_object'}}),
        ('schema', [{'role': 'user', 'content': 'Return JSON'}], {'response_format': {'type': 'json_schema',
            'json_schema': {'schema': {'type': 'object', 'properties': {'x': {'type': 'integer'}}}}}}),
        ('image-marker', [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe'},
            {'type': 'image_url', 'image_url': {'url': 'unused'}}]}], {}),
        ('block-hint', [{'role': 'user', '_checkpoint_hints': [2], 'content': [
            {'type': 'text', 'text': history}, {'type': 'text', 'text': history},
            {'type': 'text', 'text': 'Changing suffix'}]}], {}),
    ]
    for name, messages, options in cases:
        options = {'add_generation_prompt': True, **options}
        arguments = {k: v for k, v in options.items() if k not in ('enable_thinking', 'reasoning_effort')}
        arguments['thinking'] = True
        arguments['thinking_effort'] = options.get('reasoning_effort', 'max')
        if name == 'image-marker':
            arguments['image_prompts'] = ['<|media_pad|>']
        reference_messages = [{k: v for k, v in message.items() if k != '_checkpoint_hints'}
            for message in messages]
        segments = module.build_chat_segments(reference_messages, **arguments)
        ids, boundaries, history_count = [], [], 0
        for segment in segments:
            ids.extend(encoding.encode(segment.text,
                allowed_special='all' if segment.allow_special else set(), disallowed_special=()))
            if segment.text == history:
                history_count += 1
                if history_count == 2:
                    boundaries.append(len(ids))
            if segment.allow_special and segment.text == '<|end_of_msg|>':
                boundaries.append(len(ids))
        case = dict(name=name, messages=messages, options=options,
            rendered=''.join(s.text for s in segments), token_ids=ids)
        if name == 'block-hint':
            assert history_count == 2
            case['message_boundaries'] = boundaries
        fixtures['cases'].append(case)
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    result = subprocess.run([str(binary), str(tmp_path), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
