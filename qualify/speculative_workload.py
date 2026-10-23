"""Freeze stochastic HTTP inputs and paired execution order for speculation.

The authored fixtures contain recorded conversations, never benchmark-generated
history. This module renders and validates inputs on CPU; it sends no requests.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess

from qualify.provenance import digest, write_json
from ignition.preparation.qwen38 import (
    CONFIG_SHA256, FRONTEND_SHA256, GENERATION_SHA256, INDEX_SHA256,
    MODEL_ID, REVISION, TRANSFORMERS_REVISION,
)


SAMPLING = dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
CATEGORIES = dict(tool_turn=6, code_edit=4, chat_reasoning=4,
                  structured_json=4, long_context=2, prefix_branch=4)
ASSETS = {**FRONTEND_SHA256, "config.json": CONFIG_SHA256,
          "generation_config.json": GENERATION_SHA256,
          "model.safetensors.index.json": INDEX_SHA256}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def value_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def token_sha(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def _message(role, content, **extra):
    return dict(role=role, content=content, **extra)


def _tool(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


def _recorded_call(name, arguments, result, call_id):
    return [_message("assistant", "", tool_calls=[{"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json_bytes(arguments).decode()}}]),
            _message("tool", json_bytes(result).decode(), tool_call_id=call_id)]


def _catalog(records):
    return "\n".join(
        f"service-{i:04d}: owner=team-{i % 17:02d}; region=zone-{i % 5}; "
        f"replicas={2 + i % 7}; timeout_ms={700 + i % 13 * 50}; "
        f"retry_limit={i % 4}; queue_limit={128 + i % 9 * 64}; "
        f"runbook=check queue depth, then dependency health, then error budget; "
        f"revision=r{1000 + i}."
        for i in range(records)
    )


def _code():
    return "\n\n".join(
        f"def validate_{name}(value):\n"
        f"    if not isinstance(value, int):\n        raise TypeError('{name} must be an integer')\n"
        f"    if value < {lower} or value > {upper}:\n"
        f"        raise ValueError('{name} out of range')\n    return value"
        for name, lower, upper in (
            ("workers", 1, 64), ("queue_limit", 1, 65536), ("timeout_ms", 1, 120000),
            ("retry_limit", 0, 8), ("batch_size", 1, 512), ("page_size", 1, 4096),
            ("port", 1, 65535), ("max_connections", 1, 4096),
        )
    ) + "\n"


def definitions():
    """Return ordered authored contexts and output semantics, without tokenization."""
    cases = []

    def add(identifier, category, messages, semantics, *, tools=(), budget=512,
            copy_opportunity="low", stage=None, **request_fields):
        request = dict(model=MODEL_ID, messages=deepcopy(messages), tools=deepcopy(list(tools)),
                       chat_template_kwargs={"enable_thinking": False, "add_generation_prompt": True},
                       **SAMPLING, seed=48100 + len(cases), max_tokens=budget,
                       stream=True, stream_options={"include_usage": True},
                       return_token_ids=True, return_token_timestamps=True)
        request.update(request_fields)
        cases.append(dict(id=identifier, category=category, request=request,
                          expected_output_semantics=semantics,
                          copy_opportunity=copy_opportunity,
                          schedule_stage=stage or identifier))

    tools = [_tool("read_incident", "Read a recorded incident, without changing it.",
                   {"incident_id": {"type": "string"}}),
             _tool("read_service", "Read a recorded service configuration.",
                   {"service": {"type": "string"}})]
    incident = [_message("system", "Investigate using the read-only tools. Do not invent tool results or execute changes."),
                _message("user", "Investigate INC-204: billing requests became slow at 09:12. First read the incident.")]
    release = [_message("system", "Review a release using read-only records. Recommend a decision; do not deploy anything."),
               _message("user", "Review INC-205 for the search release. Read the incident before recommending a rollout.")]
    conversations = [
        ("incident", incident, "INC-204", "billing", {"symptom": "p99 rose from 90ms to 850ms",
         "change": "retry_limit changed from 1 to 5", "time": "09:12", "queue_depth": 740},
         {"retry_limit": 5, "dependency_errors_percent": 7, "replicas": 8, "previous_retry_limit": 1}),
        ("release", release, "INC-205", "search", {"symptom": "canary p99 is 110ms; baseline 100ms",
         "change": "new ranking index", "time": "10:20", "error_rate_percent": 0.1},
         {"canary_fraction": 0.05, "abort_p99_ms": 140, "abort_error_percent": 0.5,
          "observation_minutes": 4, "required_observation_minutes": 20}),
    ]
    for turn in range(3):
        for name, initial, incident_id, service, incident_result, service_result in conversations:
            messages = deepcopy(initial)
            if turn >= 1:
                messages += _recorded_call("read_incident", {"incident_id": incident_id},
                                           incident_result, f"{name}-read-incident")
                messages += [_message("user", f"Read the {service} service configuration before giving your recommendation.")]
            if turn >= 2:
                messages += _recorded_call("read_service", {"service": service},
                                           service_result, f"{name}-read-service")
                messages += [_message("user", "Give a concise evidence-based recommendation and a reversible next step. No further tool calls.")]
            semantics = ({"kind": "tool_call", "allowed_function": "read_incident" if turn == 0 else "read_service",
                          "expected_arguments": {"incident_id": incident_id} if turn == 0 else {"service": service}}
                         if turn < 2 else {"kind": "text", "requirements": ["cite recorded values", "no claimed external actions"]})
            add(f"{name}-turn-{turn + 1}", "tool_turn", messages, semantics, tools=tools,
                copy_opportunity="recorded_tool_values", stage=f"tools-turn-{turn + 1}",
                tool_choice="auto" if turn < 2 else "none")

    source = _code()
    edits = [
        ("reject-bool", "Reject bool values in all validators while retaining integer behavior. Return the complete revised module.", {}),
        ("timeout-range", "Change only timeout_ms to accept zero and allow up to 300000. Return a unified diff.", {}),
        ("tests", "Write pytest parameterized tests for lower and upper boundaries and one out-of-range value of every validator.", {}),
        ("docstrings-stop", "Add one concise docstring to every validator. Return the complete module, followed by a line END_EDIT.", {"stop": ["\nEND_EDIT"]}),
    ]
    for name, instruction, extra in edits:
        add(f"code-{name}", "code_edit", [_message("system", "Make the requested Python edit. Preserve unrelated behavior."),
            _message("user", f"File validation.py:\n```python\n{source}```\n{instruction}")],
            {"kind": "code", "requirements": [instruction]}, budget=1024,
            copy_opportunity="supplied_source", **extra)

    for name, prompt in [
        ("capacity", "A queue receives 90 jobs/s. Four workers each handle 20 jobs/s. Explain the stability problem and two remedies; distinguish throughput from latency."),
        ("tradeoff", "Compare taking an overnight train with taking an early flight for a fictional 600 km trip. Give a balanced recommendation without inventing prices or schedules."),
        ("deduction", "Three sealed boxes are labeled red, blue, and mixed; every label is wrong. Explain how one draw identifies all three boxes."),
        ("review", "A colleague says a 99% cache hit rate proves a system is reliable. Give a respectful counterexample and useful measurements."),
    ]:
        add(f"chat-{name}", "chat_reasoning", [_message("user", prompt)],
            {"kind": "text", "requirements": ["answer the question with a justified conclusion"]}, budget=384)

    schemas = [
        ("incident", "Extract service=billing, severity=2, retry_limit=1 into JSON.",
         {"service": {"type": "string"}, "severity": {"type": "integer"}, "retry_limit": {"type": "integer"}}),
        ("plan", "Return a short plan for testing a configuration change, with two or three steps and approved=false.",
         {"steps": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3}, "approved": {"type": "boolean"}}),
        ("literal-tags", "Return text containing the literal strings <think> and <tool_call>, and count=2.",
         {"text": {"type": "string"}, "count": {"type": "integer"}}),
        ("unicode", "Return city=Montréal, note=雪, and temperatures=[-3,2,5].",
         {"city": {"type": "string"}, "note": {"type": "string"}, "temperatures": {"type": "array", "items": {"type": "integer"}}}),
    ]
    for name, prompt, properties in schemas:
        schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
        semantics = {"kind": "json_schema", "schema": schema, "expected_values": {
            "incident": {"service": "billing", "severity": 2, "retry_limit": 1},
            "plan": {"approved": False}, "literal-tags": {"count": 2},
            "unicode": {"city": "Montréal", "note": "雪", "temperatures": [-3, 2, 5]},
        }[name]}
        if name == "literal-tags":
            semantics["string_contains"] = {"text": ["<think>", "<tool_call>"]}
        add(f"json-{name}", "structured_json", [_message("user", prompt)],
            semantics, budget=256, copy_opportunity="supplied_values",
            response_format={"type": "json_schema", "json_schema": {"name": name.replace("-", "_"), "strict": True, "schema": schema}})

    for count in (192, 768):
        add(f"long-services-{count}", "long_context", [_message("system", "Use only the supplied service records."),
            _message("user", _catalog(count)),
            _message("user", f"Compare service-0007, service-{count // 2:04d}, and service-{count - 1:04d}. Quote each timeout, retry limit, and owner, then recommend which deserves the first retry-policy review.")],
            {"kind": "text", "requirements": ["extract all three records accurately", "justify prioritization"]},
            copy_opportunity="long_record_values")

    common = [_message("system", "Review these service records without inventing missing data."),
              _message("user", _catalog(96)),
              _message("assistant", "I have the service records. Specify the services and fields to compare.")]
    for name, tail in [("seed", "Compare service-0007 and service-0041: quote their complete records."),
                       ("repeat", "Compare service-0007 and service-0041: quote their complete records."),
                       ("branch-owner", "List the owners of service-0007, service-0041, and service-0089."),
                       ("branch-retry", "Compare retry limits of service-0007, service-0041, and service-0089; recommend one change.")]:
        add(f"prefix-{name}", "prefix_branch", common + [_message("user", tail)],
            {"kind": "text", "requirements": [tail]}, copy_opportunity="shared_records",
            seed=49100)
    return cases


def template_input(request):
    """Apply the native text-request normalization without changing wire data."""
    messages = deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            call["function"]["arguments"] = arguments
    response_format = request.get("response_format", {})
    if response_format.get("type") == "json_schema":
        schema = json_bytes(response_format["json_schema"]["schema"]).decode()
        instruction = (
            "Return the final answer as a JSON value matching this schema. Do not wrap it in Markdown fences or add text outside the JSON value. Tool calls may precede the final answer.\n"
            f"JSON schema:\n{schema}"
        )
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + instruction
        else:
            messages.insert(0, _message("system", instruction))
    options = deepcopy(request["chat_template_kwargs"])
    options["tools"] = [] if request.get("tool_choice") == "none" else deepcopy(request["tools"])
    return messages, options


def build_workload(tokenizer):
    cases = definitions()
    for case in cases:
        messages, options = template_input(case["request"])
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
        if tokenizer.encode(rendered, add_special_tokens=False) != tokens:
            raise ValueError(f"{case['id']}: rendered bytes and template tokenization differ")
        case.update(rendered_prompt=rendered, prompt_tokens=tokens, prompt_token_count=len(tokens),
                    rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    prompt_tokens_sha256=token_sha(tokens), request_sha256=value_sha(case["request"]),
                    turn_boundaries=[{"index": i, "role": m["role"], "message_sha256": value_sha(m)}
                                     for i, m in enumerate(case["request"]["messages"])],
                    recorded_tool_results=[{"index": i, "message": deepcopy(m), "sha256": value_sha(m)}
                                           for i, m in enumerate(case["request"]["messages"]) if m["role"] == "tool"])
    base = next(c for c in cases if c["id"] == "prefix-seed")["prompt_tokens"]
    for case in cases:
        if case["category"] == "prefix_branch":
            tokens = case["prompt_tokens"]
            shared = next((i for i, (a, b) in enumerate(zip(base, tokens)) if a != b), min(len(base), len(tokens)))
            case["prefix_relation"] = {"reference": "prefix-seed", "common_tokens": shared,
                                       "aligned_common_tokens": shared // 64 * 64,
                                       "identical_request": case["request_sha256"] == next(c for c in cases if c["id"] == "prefix-seed")["request_sha256"]}
    return {"artifact_type": "speculative_http_workload", "schema_version": 1,
            "model_id": MODEL_ID, "revision": REVISION, "cases": cases,
            "sampling": deepcopy(SAMPLING), "eos_policy": {"ignore_eos": False, "token_ids": [248046, 248044]},
            "construction": {"source": "Authored deterministic fixtures in qualify/speculative_workload.py",
                "history": "Recorded assistant calls and tool results are fixed inputs; benchmark outputs never enter later requests.",
                "external_tool_execution": False, "token_hash_encoding": "little-endian uint32",
                "completion": "Record actual EOS, stop, budget and error outcomes; no minimum output length or forced EOS suppression.",
                "copy_opportunity": "Fixture intent only; actual offered/accepted copy counts require runtime metrics."}}


def arms():
    return [{"id": "no-mtp", "mtp": False}] + [
        {"id": f"mtp-fixed-{width}", "mtp": True,
         "speculative_fixed_width": width, "history_drafts": False} for width in range(1, 9)
    ] + [{"id": f"mtp-adaptive-copy-{str(copy).lower()}", "mtp": True,
          "speculative_fixed_width": None,
          "history_drafts": copy} for copy in (False, True)]


def request_groups(workload, concurrency):
    if type(concurrency) is not int or concurrency not in (1, 2, 4):
        raise ValueError("concurrency must be 1, 2 or 4")
    stages = {}
    for case in workload["cases"]:
        stage = case["schedule_stage"] if case["category"] in ("tool_turn", "prefix_branch") else case["category"]
        stages.setdefault(stage, []).append(case["id"])
    return [{"stage": stage, "case_ids": ids[start:start + concurrency],
             "concurrency": len(ids[start:start + concurrency])}
            for stage, ids in stages.items() for start in range(0, len(ids), concurrency)]


def execution_plan(workload, repetitions=2):
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("at least two repetitions are required to alternate arm order")
    policies = arms()
    return {"arms": policies, "repetitions": repetitions,
            "cache_protocol": "Fresh empty prefix cache for each arm/repetition/concurrency trial; retain ordinary cache within the trial; await idle after every group. Record observed reuse rather than asserting cache residency.",
            "state_protocol": "Start each arm through its serving configuration; no runtime policy mutation. Graph width is bounded by the actual batch and available proposals.",
            "request_protocol": "Send cases/<id>/request.json bytes unchanged. Record raw token IDs/timestamps, usage, terminal reason, content/tool arguments and errors. Do not execute tool calls.",
            "comparison_protocol": "Check output semantics and completed work before latency comparisons; seed equality does not imply identical stochastic token streams across proposal algorithms.",
            "trials": [{"repetition": repetition, "concurrency_ceiling": concurrency,
                        "arm_order": [a["id"] for a in (policies if repetition % 2 == 0 else list(reversed(policies)))],
                        "groups": request_groups(workload, concurrency)}
                       for repetition in range(repetitions) for concurrency in (1, 2, 4)]}


def validate_workload(workload):
    if (workload.get("artifact_type") != "speculative_http_workload" or workload.get("schema_version") != 1
            or workload.get("model_id") != MODEL_ID or workload.get("revision") != REVISION
            or workload.get("sampling") != SAMPLING
            or workload.get("eos_policy") != {"ignore_eos": False, "token_ids": [248046, 248044]}):
        raise ValueError("workload identity or sampling/EOS contract differs")
    expected = definitions()
    cases = workload.get("cases", [])
    if [c.get("id") for c in cases] != [c["id"] for c in expected] or Counter(c.get("category") for c in cases) != CATEGORIES:
        raise ValueError("all 24 ordered workload cases are required")
    for case, definition in zip(cases, expected):
        name = case["id"]
        if any(json_bytes(case.get(k)) != json_bytes(v) for k, v in definition.items()):
            raise ValueError(f"{name}: authored input/semantics changed")
        tokens = case.get("prompt_tokens")
        if (not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < 248320 for t in tokens)
                or case.get("prompt_token_count") != len(tokens) or len(tokens) + case["request"]["max_tokens"] > 262144):
            raise ValueError(f"{name}: invalid prompt token IDs or context budget")
        if (case.get("prompt_tokens_sha256") != token_sha(tokens)
                or case.get("request_sha256") != value_sha(case["request"])
                or case.get("rendered_prompt_sha256") != hashlib.sha256(case["rendered_prompt"].encode()).hexdigest()):
            raise ValueError(f"{name}: payload fingerprint mismatch")
        messages = case["request"]["messages"]
        if case.get("turn_boundaries") != [{"index": i, "role": m["role"], "message_sha256": value_sha(m)} for i, m in enumerate(messages)]:
            raise ValueError(f"{name}: turn boundaries differ")
        if case.get("recorded_tool_results") != [{"index": i, "message": m, "sha256": value_sha(m)} for i, m in enumerate(messages) if m["role"] == "tool"]:
            raise ValueError(f"{name}: recorded tool results differ")
    prefix = [c for c in cases if c["category"] == "prefix_branch"]
    if prefix[0]["prompt_tokens"] != prefix[1]["prompt_tokens"] or prefix[0]["request_sha256"] != prefix[1]["request_sha256"]:
        raise ValueError("identical-tail replay must retain identical full request bytes")
    for case in prefix:
        base, tokens = prefix[0]["prompt_tokens"], case["prompt_tokens"]
        shared = next((i for i, (a, b) in enumerate(zip(base, tokens)) if a != b), min(len(base), len(tokens)))
        relation = {"reference": "prefix-seed", "common_tokens": shared, "aligned_common_tokens": shared // 64 * 64,
                    "identical_request": case["request_sha256"] == prefix[0]["request_sha256"]}
        if case.get("prefix_relation") != relation or shared < 64:
            raise ValueError("prefix relation does not match actual token IDs")


def load_workload(path):
    path = Path(path)
    workload = json.loads(path.read_text())
    validate_workload(workload)
    return workload


def validate_artifacts(folder):
    """Check that HTTP payloads and renders match the benchmark workload."""
    folder = Path(folder).resolve()
    workload = load_workload(folder / "workload.json")
    if json.loads((folder / "execution-plan.json").read_text()) != execution_plan(workload):
        raise ValueError("paired execution plan changed")
    for case in workload["cases"]:
        case_dir = folder / "cases" / case["id"]
        if ((case_dir / "request.json").read_bytes() != json_bytes(case["request"])
                or (case_dir / "rendered.txt").read_bytes() != case["rendered_prompt"].encode()):
            raise ValueError(f"{case['id']}: published request/render bytes differ")
    return workload


def generate(snapshot, output, native_frontend, baseline_fixture):
    from transformers import AutoTokenizer

    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    native_frontend, baseline_fixture = Path(native_frontend).resolve(), Path(baseline_fixture).resolve()
    distribution = importlib.metadata.distribution("transformers")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION:
        raise ValueError("pinned Transformers source is required")
    for name, expected_hash in ASSETS.items():
        if digest(snapshot / name) != expected_hash:
            raise ValueError(f"checkpoint asset fingerprint mismatch: {name}")
    baseline = json.loads(baseline_fixture.read_text())
    if baseline["assets"] != FRONTEND_SHA256 or baseline["model_id"] != MODEL_ID or baseline["revision"] != REVISION:
        raise ValueError("native frontend fixture identity differs")
    output.mkdir(parents=True, exist_ok=False)
    workload = build_workload(AutoTokenizer.from_pretrained(snapshot, local_files_only=True))
    native_fixture = deepcopy(baseline)
    for case in workload["cases"]:
        messages, options = template_input(case["request"])
        native_fixture["cases"].append({"name": "speculative-" + case["id"], "messages": messages,
                                       "options": options, "rendered": case["rendered_prompt"], "token_ids": case["prompt_tokens"]})
        folder = output / "cases" / case["id"]
        folder.mkdir(parents=True)
        (folder / "request.json").write_bytes(json_bytes(case["request"]))
        (folder / "rendered.txt").write_text(case["rendered_prompt"])
    fixture_path = output / "native-frontend-fixture.json"
    write_json(fixture_path, native_fixture)
    command = [str(native_frontend), str(snapshot), str(fixture_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (output / "native-frontend.stdout").write_text(result.stdout)
    (output / "native-frontend.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"native frontend validation failed; inspect {output}")
    counts = json.loads(result.stdout)
    if counts["cases"] != len(native_fixture["cases"]):
        raise ValueError("native frontend case count differs")
    validate_workload(workload)
    write_json(output / "workload.json", workload)
    write_json(output / "execution-plan.json", execution_plan(workload))
    validate_artifacts(output)
    return workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("generate")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--native-frontend", type=Path, default=Path("helios/target/debug/test_frontend"))
    build.add_argument("--frontend-fixture", type=Path, default=Path("tests/frontend/qwen38.json"))
    check = sub.add_parser("validate")
    check.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        workload = generate(args.snapshot, args.output, args.native_frontend, args.frontend_fixture)
    elif args.path.is_dir():
        workload = validate_artifacts(args.path)
    else:
        workload = load_workload(args.path)
    print(json.dumps({"passed": True, "cases": len(workload["cases"]), "categories": CATEGORIES}))


if __name__ == "__main__":
    main()
