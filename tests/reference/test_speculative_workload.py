# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for fixed stochastic inputs and paired scheduling semantics."""

from copy import deepcopy
import json

import pytest

from qualify import speculative_workload as sw


class TextTokenizer:
    """Byte encoding exercises corpus integrity without claiming model token parity."""

    def apply_chat_template(self, messages, *, tokenize, return_dict=None, **options):
        text = json.dumps([messages, options], ensure_ascii=False, separators=(",", ":"))
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())


@pytest.fixture(scope="module")
def workload():
    result = sw.build_workload(TextTokenizer())
    sw.validate_workload(result)
    return result


def test_contexts_cover_recorded_turns_code_constraints_and_real_eos(workload):
    assert len(workload["cases"]) == 24
    assert workload["eos_policy"] == {"ignore_eos": False, "token_ids": [248046, 248044]}
    for case in workload["cases"]:
        request = case["request"]
        assert all(request[k] == v for k, v in sw.SAMPLING.items())
        assert type(request["seed"]) is int
        assert "ignore_eos" not in request
        assert "stop_token_ids" not in request
    tools = [c for c in workload["cases"] if c["category"] == "tool_turn"]
    assert [len(c["recorded_tool_results"]) for c in tools] == [0, 0, 1, 1, 2, 2]
    assert sum("stop" in c["request"] for c in workload["cases"]) == 1
    assert sum("response_format" in c["request"] for c in workload["cases"]) == 4


def test_native_normalization_preserves_wire_json_strings_and_tool_choice(workload):
    case = next(c for c in workload["cases"] if c["id"] == "incident-turn-3")
    before = deepcopy(case["request"])
    messages, options = sw.template_input(case["request"])
    assert case["request"] == before
    assert isinstance(case["request"]["messages"][2]["tool_calls"][0]["function"]["arguments"], str)
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"incident_id": "INC-204"}
    assert options["tools"] == []
    assert case["request"]["tools"]


@pytest.mark.parametrize("system", [None, "Extract the requested record."])
def test_schema_instruction_is_rendered_without_changing_wire_request(system):
    request = deepcopy(next(c["request"] for c in sw.definitions() if c["id"] == "json-unicode"))
    if system is not None:
        request["messages"].insert(0, {"role": "system", "content": system})
    before = sw.json_bytes(request)
    messages, _ = sw.template_input(request)
    instruction = (
        "Return the final answer as a JSON value matching this schema. Do not wrap it in Markdown fences or add text outside the JSON value. Tool calls may precede the final answer.\n"
        "JSON schema:\n" + sw.json_bytes(request["response_format"]["json_schema"]["schema"]).decode()
    )
    assert messages[0] == {"role": "system", "content": (system + "\n\n" if system else "") + instruction}
    assert messages[1:] == request["messages"][1 if system is not None else 0:]
    assert sw.json_bytes(request) == before


@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_every_context_runs_once_and_tool_turns_and_prefix_branches_are_ordered(workload, concurrency):
    groups = sw.request_groups(workload, concurrency)
    assert [name for group in groups for name in group["case_ids"]] == [c["id"] for c in workload["cases"]]
    assert all(g["concurrency"] == len(g["case_ids"]) <= concurrency for g in groups)
    for conversation in ("incident", "release"):
        indices = [next(i for i, g in enumerate(groups) if f"{conversation}-turn-{turn}" in g["case_ids"])
                   for turn in (1, 2, 3)]
        assert indices == sorted(set(indices))
    assert [g["case_ids"] for g in groups[-4:]] == [[f"prefix-{suffix}"] for suffix in ("seed", "repeat", "branch-owner", "branch-retry")]


def test_paired_arms_preserve_request_bytes_and_alternate_order(workload):
    before = sw.json_bytes(workload)
    plan = sw.execution_plan(workload)
    assert len(plan["arms"]) == 11
    for concurrency in (1, 2, 4):
        first, second = [t for t in plan["trials"] if t["concurrency_ceiling"] == concurrency]
        assert first["arm_order"] == list(reversed(second["arm_order"]))
        assert first["groups"] == second["groups"]
    assert before == sw.json_bytes(workload)
    assert plan["arms"][0] == {"id": "no-mtp", "mtp": False}
    assert [a["speculative_fixed_width"] for a in plan["arms"][1:9]] == list(range(1, 9))


def test_identical_tail_has_identical_seed_payload_and_tokens_while_branches_diverge(workload):
    cases = [c for c in workload["cases"] if c["category"] == "prefix_branch"]
    assert sw.json_bytes(cases[0]["request"]) == sw.json_bytes(cases[1]["request"])
    assert cases[0]["prompt_tokens"] == cases[1]["prompt_tokens"]
    for case in cases[2:]:
        assert case["prefix_relation"]["aligned_common_tokens"] >= 64
        assert not case["prefix_relation"]["identical_request"]
        assert case["prompt_tokens"] != cases[0]["prompt_tokens"]


@pytest.mark.parametrize("change,error", [
    (lambda w: w["cases"].pop(), "24 ordered"),
    (lambda w: w["cases"].reverse(), "24 ordered"),
    (lambda w: w["cases"][0]["request"].update(temperature=0), "authored input"),
    (lambda w: w["cases"][0]["request"].update(top_p=1), "authored input"),
    (lambda w: w["cases"][0]["request"].update(seed=True), "authored input"),
    (lambda w: w["cases"][0]["prompt_tokens"].__setitem__(0, True), "invalid prompt"),
    (lambda w: w["cases"][0]["prompt_tokens"].__setitem__(0, 248320), "invalid prompt"),
    (lambda w: w["cases"][0].update(rendered_prompt="altered"), "fingerprint"),
    (lambda w: w["cases"][2]["recorded_tool_results"].clear(), "tool results"),
    (lambda w: w["cases"][0]["turn_boundaries"].clear(), "turn boundaries"),
    (lambda w: w["cases"][-1]["prefix_relation"].update(common_tokens=0), "prefix relation"),
])
def test_corrupt_or_reinterpreted_inputs_fail_validation(workload, change, error):
    value = deepcopy(workload)
    change(value)
    with pytest.raises(ValueError, match=error):
        sw.validate_workload(value)


@pytest.fixture
def artifact_files(workload, tmp_path):
    value = deepcopy(workload)
    for case in value["cases"]:
        folder = tmp_path / "cases" / case["id"]
        folder.mkdir(parents=True)
        (folder / "request.json").write_bytes(sw.json_bytes(case["request"]))
        (folder / "rendered.txt").write_text(case["rendered_prompt"])
    sw.write_json(tmp_path / "workload.json", value)
    sw.write_json(tmp_path / "execution-plan.json", sw.execution_plan(value))
    return tmp_path


def test_request_payloads_match_workload(artifact_files):
    assert len(sw.validate_artifacts(artifact_files)["cases"]) == 24
    request = artifact_files / "cases" / "incident-turn-1" / "request.json"
    request.write_bytes(request.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="published request/render bytes"):
        sw.validate_artifacts(artifact_files)


def test_rendered_text_must_encode_to_the_same_tokens():
    class BrokenTokenizer(TextTokenizer):
        def apply_chat_template(self, *args, tokenize, **kwargs):
            return [1] if tokenize else "not-token-one"
    with pytest.raises(ValueError, match="tokenization differ"):
        sw.build_workload(BrokenTokenizer())


@pytest.mark.parametrize("value", [0, 3, True])
def test_invalid_concurrency_cannot_silently_change_schedule(workload, value):
    with pytest.raises(ValueError, match="concurrency"):
        sw.request_groups(workload, value)
