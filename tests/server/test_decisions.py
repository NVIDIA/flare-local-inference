"""Decision protocol checks for the hardware test round; requires an idle service.

FLARE_DECISION_TEST_CHECKPOINT enables matched tokenizer/raw-logprob checks.
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('FLARE_API_TEST_URL'), reason='requires a native Flare service'
)


@pytest.fixture(scope='module')
def service():
    import httpx
    with httpx.Client(base_url=os.environ['FLARE_API_TEST_URL'].rstrip('/'),
                      headers={'Authorization': 'Bearer ' + os.environ.get('FLARE_API_TEST_KEY', 'local')},
                      timeout=120) as http:
        card = http.get('/v1/models').json()['data'][0]
        yield http, card


def test_systemone_cold_warm_structured_questions_and_usage(service):
    http, card = service
    body = {'model': card['id'], 'state': {'message': 'Charged twice'}, 'questions': {
        'refund': {'type': 'noul', 'instructions': 'Is a refund requested?'},
        'team': {'type': 'choice', 'criteria': {'billing': None, 'support': {'help': ['Other help']}}},
        'urgency': {'type': 'score', 'criteria': ['Routine', 'Urgent']}}}
    for _ in range(2):
        response = http.post('/v1/systemone', json=body)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['model'] == card['root']
        assert list(result['answers']) == list(body['questions'])
        assert 0 <= result['answers']['refund']['noul'] <= 1
        for name in ['team', 'urgency']:
            answer = result['answers'][name]
            assert math.isclose(sum(answer['probabilities'].values()), 1, abs_tol=1e-14)
            assert 0 <= answer['confidence'] <= 1
        assert result['usage']['output_tokens'] == 3
        assert result['usage']['input_tokens'] > 3


def test_systemone_concurrent_chat_and_questions(service):
    http, card = service
    body = {'model': card['id'], 'state': 'Charged twice',
            'questions': {str(i): {'type': 'noul', 'instructions': f'Is this billing? {i}'} for i in range(4)}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        scoring = pool.submit(http.post, '/v1/systemone', json=body)
        chat = http.post('/v1/chat/completions', json={'model': card['id'],
            'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 8})
        assert chat.status_code == 200, chat.text
        response = scoring.result()
        assert response.status_code == 200, response.text
        assert response.json()['usage']['output_tokens'] == 4


def test_systemone_assets_use_existing_vision_and_multipart_paths(service):
    import base64
    import io
    from PIL import Image
    http, card = service
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(buffer, format='PNG')
    body = {'model': card['id'], 'state': {'photo': 'photo-ref'}, 'questions': {
        'red': {'type': 'noul', 'instructions': 'Is the image red?'}}, 'assets': {
        'photo-ref': {'type': 'image', 'format': 'png',
                      'data_base64': base64.b64encode(buffer.getvalue()).decode()}}}
    assert http.post('/v1/systemone', json=body).status_code == 400
    expected = 200 if 'image' in card['flare']['input_modalities'] else 400
    response = http.post('/v1/systemone', json=body, headers={'TypeSafe-Danger-Zone-Images': 'true'})
    assert response.status_code == expected, response.text
    body['assets']['photo-ref']['part'] = 'photo'
    body['assets']['photo-ref']['data_base64'] = 'part wins'
    response = http.post('/v1/systemone', headers={'TypeSafe-Danger-Zone-Images': 'true'}, files=[
        ('request', (None, json.dumps(body), 'application/json')),
        ('photo', ('photo.png', buffer.getvalue(), 'image/png'))])
    assert response.status_code == expected, response.text


def test_systemone_matches_requested_raw_label_reads(service):
    checkpoint = os.environ.get('FLARE_DECISION_TEST_CHECKPOINT')
    if not checkpoint:
        pytest.skip('requires the served package tokenizer checkpoint')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    http, card = service
    state = {'message': 'Charged twice'}
    instructions = 'Is a refund requested?'
    candidates = [' ' + chr(a) for a in range(65, 91)] + [
        ' ' + chr(a) + chr(b) for a in range(65, 91) for b in range(65, 91)]
    def serialized(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    def render(labels):
        options = [{'value': False, 'description': None}, {'value': True, 'description': None}]
        question = 'Question:\n' + serialized(instructions) + '\nOptions:\n'
        question += ''.join(label.lstrip() + ': ' + json.dumps(option, separators=(',', ':')) + '\n'
                            for label, option in zip(labels, options))
        messages = [
            {'role': 'system', 'content': 'Evaluate the state against the question. State, instructions and option descriptions below are JSON data. Select exactly one option by its label, with no explanation.'},
            {'role': 'user', 'content': 'State:\n' + serialized(state)},
            {'role': 'assistant', 'content': 'State received.'},
            {'role': 'user', 'content': question + 'Answer:'}]
        response = http.post('/v1/chat/completions', json={'model': card['id'],
            'messages': messages, 'max_tokens': 1, 'ignore_eos': True, 'return_token_ids': True,
            'chat_template_kwargs': {'enable_thinking': False, 'add_generation_prompt': True}})
        assert response.status_code == 200, response.text
        tokens = response.json()['prompt_token_ids']
        prompt = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert tokenizer.encode(prompt, add_special_tokens=False) == tokens
        return prompt
    prompt = render(candidates[:2])
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    labels, ids = [], []
    for label in candidates:
        extended = tokenizer.encode(prompt + label, add_special_tokens=False)
        if (len(extended) == len(tokens) + 1 and extended[:-1] == tokens
                and extended[-1] not in tokenizer.all_special_ids and extended[-1] not in ids
                and tokenizer.decode(extended) == tokenizer.decode(tokens) + label):
            labels.append(label)
            ids.append(extended[-1])
            if len(ids) == 2:
                break
    assert len(ids) == 2
    prompt = render(labels)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    for label, id in zip(labels, ids):
        assert tokenizer.encode(prompt + label, add_special_tokens=False) == tokens + [id]
    references = []
    for _ in range(5):
        response = http.post('/v1/completions', json={'model': card['id'], 'prompt': tokens,
            'max_tokens': 1, 'ignore_eos': True, 'logprobs': 0, 'logprob_token_ids': ids,
            'return_tokens_as_token_ids': True})
        assert response.status_code == 200, response.text
        values = response.json()['choices'][0]['logprobs']['top_logprobs'][0]
        scores = [values[f'token_id:{id}'] for id in ids]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        references.append(weights[1] / sum(weights))
    response = http.post('/v1/systemone', json={'model': card['id'], 'state': state,
        'questions': {'q': {'type': 'noul', 'instructions': instructions}}})
    assert response.status_code == 200, response.text
    actual = response.json()['answers']['q']['noul']
    spread = max(references) - min(references)
    assert min(references) - spread - 8 * math.ulp(1.0) <= actual <= max(references) + spread + 8 * math.ulp(1.0)
