use super::*;
use crate::{
    output::{Event, Termination},
    server::inference::{Input, Part, RawPrompt, ToolChoice, template_messages},
};
use serde_json::{Value, json};

fn request(protocol: Protocol, body: Value) -> inference::Request {
    decode(protocol, body, false).unwrap().request
}
fn tools() -> Vec<inference::Tool> {
    vec![inference::Tool {
        name: "lookup".into(),
        namespace: None,
        description: "Find a value".into(),
        parameters: json!({"type":"object"}),
        custom: false,
        grammar: None,
    }]
}
fn encoder(protocol: Protocol) -> Encoder {
    encoder_with_choice(protocol, ToolChoice::Auto)
}
fn encoder_with_choice(protocol: Protocol, tool_choice: ToolChoice) -> Encoder {
    encoder_with_parallel(protocol, tool_choice, true)
}
fn encoder_with_parallel(protocol: Protocol, tool_choice: ToolChoice, parallel: bool) -> Encoder {
    Encoder::new(
        protocol,
        "test".into(),
        "model".into(),
        1,
        1,
        vec![PromptUsage {
            tokens: 17,
            image_tokens: 0,
        }],
        tools(),
        tool_choice,
        parallel,
        true,
        json!({}),
    )
}
fn events() -> Vec<Event> {
    vec![
        Event::Reasoning("考え".into()),
        Event::Content("Answer ✓".into()),
        Event::ToolBegin {
            index: 0,
            name: "lookup".into(),
            id: Some("call_1".into()),
        },
        Event::ToolArguments {
            index: 0,
            text: "{\"query\":\"".into(),
        },
        Event::ToolArguments {
            index: 0,
            text: "café\"}".into(),
        },
        Event::ToolEnd { index: 0 },
    ]
}

#[test]
fn protocols_decode_to_equivalent_conversations() {
    let chat = request(
        Protocol::Chat,
        json!({"messages":[{"role":"system","content":"Instructions"},{"role":"user","content":"Hello"}],"max_tokens":123,"temperature":0.7}),
    );
    let messages = request(
        Protocol::Messages,
        json!({"system":"Instructions","messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],"max_tokens":123,"temperature":0.7}),
    );
    let responses = request(
        Protocol::Responses,
        json!({"instructions":"Instructions","input":"Hello","max_output_tokens":123,"temperature":0.7,"store":false}),
    );
    assert_eq!(chat, messages);
    assert_eq!(chat, responses);
}

#[test]
fn required_tool_choice_is_shared_and_requires_declared_tools() {
    let mut requests = vec![];
    for (protocol, mut body) in [
        (
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello"}],
            "tool_choice":"required","tools":[{"type":"function","function":{"name":"lookup"}}]}),
        ),
        (
            Protocol::Messages,
            json!({"messages":[{"role":"user","content":"Hello"}],
            "tool_choice":{"type":"any"},"tools":[{"name":"lookup"}]}),
        ),
        (
            Protocol::Responses,
            json!({"input":"Hello",
            "tool_choice":"required","tools":[{"type":"function","name":"lookup"}]}),
        ),
    ] {
        // Token counting and generation use the same tool-choice contract.
        for count in [false, true] {
            body[if protocol == Protocol::Responses {
                "max_output_tokens"
            } else {
                "max_tokens"
            }] = json!(128);
            let req = decode(protocol, body.clone(), count).unwrap().request;
            assert_eq!(req.tool_choice, ToolChoice::Required);
            requests.push(req);
        }
        body["tools"] = json!([]);
        let error = decode(protocol, body, false).err().unwrap();
        assert_eq!(
            error.to_string(),
            "required tool choice needs at least one tool"
        );
    }
    assert!(requests.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn named_tool_choices_decode_validate_and_preserve_messages() {
    let mut requests = vec![];
    for (protocol, mut body, choice) in [
        (Protocol::Chat, json!({"messages":[{"role":"user","content":"Hello"}],
            "tools":[{"type":"function","function":{"name":"lookup"}}]}),
            json!({"type":"function","function":{"name":"lookup"}})),
        (Protocol::Messages, json!({"messages":[{"role":"user","content":"Hello"}],
            "max_tokens":128,"tools":[{"name":"lookup"}]}),
            json!({"type":"tool","name":"lookup"})),
        (Protocol::Responses, json!({"input":"Hello",
            "tools":[{"type":"function","name":"lookup"}]}),
            json!({"type":"function","name":"lookup"})),
    ] {
        body[if protocol == Protocol::Responses { "max_output_tokens" } else { "max_tokens" }] = json!(128);
        let auto = request(protocol, body.clone());
        body["tool_choice"] = choice;
        for count in [false, true] {
            let named = decode(protocol, body.clone(), count).unwrap().request;
            assert_eq!(named.tool_choice, ToolChoice::Named("lookup".into()));
            assert_eq!(auto.input, named.input);
            assert_eq!(auto.options, named.options);
            let Input::Conversation(messages) = &named.input else { panic!() };
            assert_eq!(
                template_messages(messages, named.tool_choice.clone()).unwrap(),
                template_messages(messages, ToolChoice::Auto).unwrap(),
            );
            requests.push(named);
        }
        for name in [json!("missing"), json!(""), json!(null), json!(3)] {
            let mut invalid = body.clone();
            if protocol == Protocol::Chat {
                invalid["tool_choice"]["function"]["name"] = name;
            } else {
                invalid["tool_choice"]["name"] = name;
            }
            assert!(decode(protocol, invalid, false).is_err());
        }
        body["tools"] = json!([]);
        assert!(decode(protocol, body, false).err().unwrap().to_string().contains("not declared"));
    }
    assert!(requests.windows(2).all(|pair| pair[0] == pair[1]));
    for (kind, tool) in [
        ("custom", json!({"type":"custom","name":"lookup"})),
        ("function", json!({"type":"function","name":"lookup"})),
    ] {
        let body = json!({"input":"Hello","tools":[tool],
            "tool_choice":{"type":kind,"name":"lookup"}});
        assert!(decode(Protocol::Responses, body.clone(), false).is_ok());
        let mut wrong_kind = body;
        wrong_kind["tool_choice"]["type"] = json!(if kind == "custom" { "function" } else { "custom" });
        assert!(decode(Protocol::Responses, wrong_kind, false).is_err());
    }
    let named = request(Protocol::Responses, json!({"input":"Hello",
        "tools":[{"type":"namespace","name":"db","tools":[{"type":"function","name":"lookup"}]}],
        "tool_choice":{"type":"function","name":"db__lookup"}}));
    assert_eq!(named.tool_choice, ToolChoice::Named(named.tools[0].native_name()));
}

#[test]
fn named_calls_strip_answer_text_and_validate_before_emitting_other_tools() {
    use crate::server::output::Block;
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        for chunked in [false, true] {
            let mut enc = encoder_with_choice(protocol, ToolChoice::Named("lookup".into()));
            let mut generated = events();
            generated.insert(0, Event::ContentLogprobs(vec![]));
            generated.push(Event::Content("Done".into()));
            let mut frames = enc.start();
            for chunk in generated.chunks(if chunked { 1 } else { generated.len() }) {
                frames.extend(enc.apply(0, chunk.to_vec()).unwrap());
            }
            frames.extend(enc.finish(0, Termination::Eos, 19, 0, None).unwrap());
            assert!(enc.outputs[0].blocks.iter().all(|b| !matches!(b, Block::Text(_))));
            assert!(enc.outputs[0].blocks.iter().any(|b| matches!(b, Block::Reasoning(s) if s == "考え")));
            assert!(enc.outputs[0].blocks.iter().any(|b| matches!(b,
                Block::Tool { name, arguments, .. } if name == "lookup" && arguments == "{\"query\":\"café\"}")));
            for value in frames.iter().map(|f| &f.data).chain(std::iter::once(&enc.response())) {
                let text = value.to_string();
                assert!(!text.contains("Answer ✓") && !text.contains("Done"));
            }
            match protocol {
                Protocol::Chat => {
                    assert!(enc.response()["choices"][0]["message"]["content"].is_null());
                    assert_eq!(enc.response()["usage"]["completion_tokens"], 19);
                }
                Protocol::Messages => assert_eq!(enc.response()["usage"]["output_tokens"], 19),
                Protocol::Responses => {
                    assert_eq!(enc.response()["usage"]["output_tokens"], 19);
                    assert_eq!(enc.response()["tool_choice"], json!({"type":"function","name":"lookup"}));
                }
                _ => unreachable!(),
            }
        }
        for other in ["other_declared", "unknown"] {
            let mut declared = tools().remove(0);
            declared.name = "other_declared".into();
            let mut declared_tools = tools();
            declared_tools.push(declared);
            let mut enc = Encoder::new(protocol, "test".into(), "model".into(), 1, 1,
                vec![PromptUsage { tokens: 17, image_tokens: 0 }], declared_tools,
                ToolChoice::Named("lookup".into()), true, true, json!({}));
            let error = enc.apply(0, vec![Event::ToolBegin { index: 0, name: other.into(), id: None }]).unwrap_err();
            assert!(error.to_string().contains("tool_choice selects"));
            assert!(enc.outputs[0].blocks.is_empty());
            assert!(!enc.failure(&error).done);
        }
        for cause in [Termination::Eos, Termination::UserStop, Termination::OutputLimit] {
            for generated in [
                vec![Event::Content("Only prose".into())],
                events().into_iter().filter(|e| !matches!(e, Event::ToolEnd { .. })).collect(),
            ] {
                let mut enc = encoder_with_choice(protocol, ToolChoice::Named("lookup".into()));
                enc.apply(0, generated).unwrap();
                let error = enc.finish(0, cause, 19, 0, None).unwrap_err();
                let failure = enc.failure(&error);
                assert!(!failure.done);
                if protocol == Protocol::Responses {
                    assert_eq!(failure.data["type"], "response.failed");
                    assert_eq!(failure.data["response"]["tool_choice"], json!({"type":"function","name":"lookup"}));
                }
            }
        }
    }
}

#[test]
fn named_custom_tool_choice_preserves_input_and_response_metadata() {
    let mut declared = tools();
    declared[0].custom = true;
    let mut enc = Encoder::new(Protocol::Responses, "test".into(), "model".into(), 1, 1,
        vec![PromptUsage { tokens: 17, image_tokens: 0 }], declared,
        ToolChoice::Named("lookup".into()), true, true, json!({}));
    enc.start();
    let frames = enc.apply(0, vec![
        Event::Content("Explanation".into()),
        Event::ToolBegin { index: 0, name: "lookup".into(), id: Some("call_1".into()) },
        Event::ToolArguments { index: 0, text: "{\"input\":\"exact\\ntext\"}".into() },
        Event::ToolEnd { index: 0 },
    ]).unwrap();
    enc.finish(0, Termination::Eos, 12, 0, None).unwrap();
    assert!(frames.iter().any(|f| f.data["type"] == "response.custom_tool_call_input.done"
        && f.data["input"] == "exact\ntext"));
    let response = enc.response();
    assert_eq!(response["tool_choice"], json!({"type":"custom","name":"lookup"}));
    assert_eq!(response["output"].as_array().unwrap().len(), 1);
    assert_eq!(response["output"][0]["type"], "custom_tool_call");
    assert_eq!(response["output"][0]["input"], "exact\ntext");
}

#[test]
fn required_tool_instruction_preserves_history_and_sampling() {
    for messages in [
        json!([{"role":"user","content":"Hello"}]),
        json!([{"role":"system","content":"Policy"},{"role":"user","content":"Hello"}]),
        json!([{"role":"system","content":[{"type":"text","text":"Policy"},{"type":"text","text":"More policy"}]},
            {"role":"user","content":"Hello"}]),
        json!([{"role":"developer","content":"Policy"},{"role":"user","content":"Hello"},
            {"role":"assistant","content":"Let me"}]),
    ] {
        let body = json!({"messages":messages,"tools":[{"type":"function","function":{"name":"lookup"}}],
            "temperature":0.7,"top_p":0.95,"seed":123});
        let auto = request(Protocol::Chat, body.clone());
        let mut required_body = body;
        required_body["tool_choice"] = json!("required");
        let required = request(Protocol::Chat, required_body);
        assert_eq!(auto.input, required.input);
        assert_eq!(auto.options, required.options);
        let Input::Conversation(messages) = &required.input else {
            panic!()
        };
        let baseline = template_messages(messages, ToolChoice::Auto).unwrap();
        let native = template_messages(messages, ToolChoice::Required).unwrap();
        assert_eq!(native[0]["role"], "system");
        assert!(
            native[0]["content"]
                .to_string()
                .contains("must call at least one")
        );
        let offset = usize::from(baseline[0]["role"] == "system");
        assert_eq!(native[1..], baseline[offset..]);
        if offset == 1 {
            assert!(native[0]["content"].to_string().contains("Policy"));
        }
    }
}

#[test]
fn required_tool_choice_fails_without_a_completed_call_for_every_termination() {
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        for cause in [
            Termination::Eos,
            Termination::UserStop,
            Termination::OutputLimit,
        ] {
            for events in [
                vec![],
                vec![Event::Content("No tool needed".into())],
                vec![Event::Reasoning("Thinking".into())],
            ] {
                let mut enc = encoder_with_choice(protocol, ToolChoice::Required);
                enc.start();
                enc.apply(0, events).unwrap();
                let error = enc.finish(0, cause, 1, 0, None).unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("did not produce a completed tool call")
                );
                let failure = enc.failure(&error);
                assert!(!failure.done);
                match protocol {
                    Protocol::Chat => {
                        assert_eq!(failure.data["error"]["type"], "server_error");
                        assert!(enc.response()["choices"][0]["finish_reason"].is_null());
                    }
                    Protocol::Messages => {
                        assert_eq!(failure.event.as_deref(), Some("error"));
                        assert!(enc.response()["stop_reason"].is_null());
                    }
                    Protocol::Responses => {
                        assert_eq!(failure.event.as_deref(), Some("response.failed"));
                        assert_eq!(failure.data["response"]["status"], "failed");
                        assert_eq!(failure.data["response"]["tool_choice"], "required");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn required_tool_choice_accepts_complete_calls_and_rejects_unknown_or_unfinished_calls() {
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        let mut enc = encoder_with_choice(protocol, ToolChoice::Required);
        let mut frames = enc.start();
        frames.extend(enc.apply(0, events()).unwrap());
        frames.extend(enc.finish(0, Termination::Eos, 12, 0, None).unwrap());
        match protocol {
            Protocol::Chat => {
                assert_eq!(enc.response()["choices"][0]["finish_reason"], "tool_calls")
            }
            Protocol::Messages => assert_eq!(frames.last().unwrap().data["type"], "message_stop"),
            Protocol::Responses => {
                assert_eq!(frames.last().unwrap().data["type"], "response.completed");
                assert_eq!(enc.response()["tool_choice"], "required");
            }
            _ => unreachable!(),
        }
        let mut enc = encoder_with_choice(protocol, ToolChoice::Required);
        let mut unfinished = events();
        unfinished.pop();
        enc.apply(0, unfinished).unwrap();
        assert!(
            enc.finish(0, Termination::OutputLimit, 12, 0, None)
                .is_err()
        );
        let mut enc = encoder_with_choice(protocol, ToolChoice::Required);
        let error = enc
            .apply(
                0,
                vec![Event::ToolBegin {
                    index: 0,
                    name: "undeclared".into(),
                    id: None,
                }],
            )
            .unwrap_err();
        assert!(error.to_string().contains("undeclared tool"));
    }
}

#[test]
fn auto_tool_choice_returns_unknown_calls_across_parsers_and_protocols() {
    use crate::{
        model::contract::{ToolParser, ToolParserKind},
        output::{Input as OutputInput, OutputConfig, OutputPipeline, Piece, tagged},
    };

    let cases = [
        (
            ToolParserKind::QwenCoder,
            1,
            "",
            "<tool_call><function=missing__lookup><parameter=query>\ncafé\n</parameter></function></tool_call>",
        ),
        (
            ToolParserKind::QwenCoder,
            0,
            "",
            "<tool_call><function=missing__lookup><parameter=query>café</parameter></function></tool_call>",
        ),
        (
            ToolParserKind::GlmArgumentTags,
            0,
            "",
            "<tool_call>missing__lookup<arg_key>query</arg_key><arg_value>café</arg_value></tool_call>",
        ),
        (
            ToolParserKind::Dsml,
            0,
            "",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"missing__lookup\"><｜DSML｜parameter name=\"query\" string=\"true\">café</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        ),
        (
            ToolParserKind::Dsml,
            0,
            " ",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"missing__lookup\"><｜DSML｜ parameter name=\"query\" string=\"true\">café</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        ),
        (
            ToolParserKind::Xtml,
            0,
            "",
            concat!(
                "<|close|>response<|sep|><|open|>tools<|sep|>",
                "<|open|>call tool=\"missing__lookup\" index=\"1\"<|sep|>",
                "<|open|>argument key=\"query\" type=\"string\"<|sep|>café",
                "<|close|>argument<|sep|><|close|>call<|sep|>",
                "<|close|>tools<|sep|><|close|>message<|sep|>"
            ),
        ),
    ];
    let markers = [(1, "<|open|>"), (2, "<|close|>"), (3, "<|sep|>")];
    for (kind, wrapping_newlines, separator, text) in cases {
        let mut pieces = Vec::new();
        let mut rest = text;
        while let Some((at, id, marker)) = markers
            .iter()
            .filter_map(|(id, marker)| rest.find(marker).map(|at| (at, *id, *marker)))
            .min_by_key(|(at, _, _)| *at)
        {
            pieces.push(Piece::Text(rest[..at].into()));
            pieces.push(Piece::Token(id));
            pieces.push(Piece::Text(marker.into()));
            rest = &rest[at + marker.len()..];
        }
        pieces.push(Piece::Text(rest.into()));
        for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
            for split_characters in [false, true] {
                let mut syntax = crate::model::testing::metadata().frontend;
                syntax.tools = Some(ToolParser {
                    kind,
                    wrapping_newlines,
                    tag_separator: separator.into(),
                });
                let mut parser = OutputPipeline::new(OutputConfig {
                    syntax,
                    control_tokens: markers
                        .iter()
                        .map(|(id, text)| (*id, (*text).into()))
                        .collect(),
                    content: tagged::OutputConfig {
                        structured: false,
                        start: tagged::OutputStart::Content,
                        tools: vec![json!({"type":"function","function":{
                            "name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}
                        }})],
                    },
                });
                let mut enc = encoder(protocol);
                let mut frames = enc.start();
                let mut events = Vec::new();
                for piece in &pieces {
                    let inputs: Vec<_> = match piece {
                        Piece::Logprob(_) => {
                            unreachable!("tool parser fixture contains only tokens and text")
                        }
                        Piece::Token(id) => vec![OutputInput::Token(*id)],
                        Piece::Text(text) if split_characters => text
                            .char_indices()
                            .map(|(at, ch)| OutputInput::Text(&text[at..at + ch.len_utf8()]))
                            .collect(),
                        Piece::Text(text) => vec![OutputInput::Text(text)],
                    };
                    for input in inputs {
                        let parsed = parser.push(input).unwrap();
                        events.extend(parsed.clone());
                        frames.extend(enc.apply(0, parsed).unwrap());
                    }
                }
                let parsed = parser.finish(Termination::Eos).unwrap();
                events.extend(parsed.clone());
                frames.extend(enc.apply(0, parsed).unwrap());
                frames.extend(enc.finish(0, Termination::Eos, 12, 0, None).unwrap());
                let response = enc.response();
                let expected = json!({"query":"café"});
                match protocol {
                    Protocol::Chat => {
                        let call = &response["choices"][0]["message"]["tool_calls"][0];
                        assert_eq!(call["function"]["name"], "missing__lookup");
                        assert_eq!(
                            serde_json::from_str::<Value>(
                                call["function"]["arguments"].as_str().unwrap()
                            )
                            .unwrap(),
                            expected
                        );
                        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
                        assert!(frames.iter().any(|f| f.data["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "missing__lookup"));
                        assert!(frames.last().unwrap().done);
                    }
                    Protocol::Messages => {
                        assert_eq!(response["content"][0]["name"], "missing__lookup");
                        assert_eq!(response["content"][0]["input"], expected);
                        assert_eq!(response["stop_reason"], "tool_use");
                        assert!(
                            frames
                                .iter()
                                .any(|f| f.data["content_block"]["name"] == "missing__lookup")
                        );
                        assert_eq!(frames.last().unwrap().data["type"], "message_stop");
                    }
                    Protocol::Responses => {
                        let call = &response["output"][0];
                        assert_eq!(call["type"], "function_call");
                        assert_eq!(call["name"], "missing__lookup");
                        assert!(call.get("namespace").is_none());
                        assert_eq!(
                            serde_json::from_str::<Value>(call["arguments"].as_str().unwrap())
                                .unwrap(),
                            expected
                        );
                        assert_eq!(response["status"], "completed");
                        assert!(
                            frames
                                .iter()
                                .any(|f| f.data["item"]["name"] == "missing__lookup")
                        );
                        assert_eq!(frames.last().unwrap().data["type"], "response.completed");
                    }
                    _ => unreachable!(),
                }
                let streamed_arguments: String =
                    frames
                        .iter()
                        .filter_map(|frame| {
                            match protocol {
                                Protocol::Chat => frame.data["choices"][0]["delta"]["tool_calls"]
                                    [0]["function"]["arguments"]
                                    .as_str(),
                                Protocol::Messages => frame.data["delta"]["partial_json"].as_str(),
                                Protocol::Responses
                                    if frame.data["type"]
                                        == "response.function_call_arguments.delta" =>
                                {
                                    frame.data["delta"].as_str()
                                }
                                _ => None,
                            }
                        })
                        .collect();
                assert_eq!(
                    serde_json::from_str::<Value>(&streamed_arguments).unwrap(),
                    expected
                );
                let mut required = encoder_with_choice(protocol, ToolChoice::Required);
                assert!(
                    required
                        .apply(0, events)
                        .unwrap_err()
                        .to_string()
                        .contains("undeclared tool")
                );
            }
        }
    }
}

#[test]
fn parallel_tool_controls_decode_to_the_same_limit() {
    for (protocol, body) in [
        (
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello"}]}),
        ),
        (
            Protocol::Messages,
            json!({"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}),
        ),
        (Protocol::Responses, json!({"input":"Hello"})),
    ] {
        assert!(request(protocol, body.clone()).parallel_tool_calls);
        let mut disabled = body.clone();
        disabled["parallel_tool_calls"] = json!(false);
        assert!(!request(protocol, disabled).parallel_tool_calls);
        let mut enabled = body;
        enabled["parallel_tool_calls"] = json!(true);
        assert!(request(protocol, enabled).parallel_tool_calls);
    }
    for disabled in [false, true] {
        let req = request(
            Protocol::Messages,
            json!({"messages":[{"role":"user","content":"Hello"}],
            "max_tokens":32,"tool_choice":{"type":"auto","disable_parallel_tool_use":disabled}}),
        );
        assert_eq!(req.parallel_tool_calls, !disabled);
    }
    assert!(
        decode(
            Protocol::Messages,
            json!({"messages":[{"role":"user","content":"Hello"}],
        "max_tokens":32,"parallel_tool_calls":false,
        "tool_choice":{"type":"auto","disable_parallel_tool_use":"invalid"}}),
            false
        )
        .is_err()
    );
}

#[test]
fn disabled_parallel_calls_fail_before_the_second_call_is_emitted() {
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        for choice in [ToolChoice::Auto, ToolChoice::Required, ToolChoice::Named("lookup".into())] {
            for parallel in [false, true] {
                let mut enc = encoder_with_parallel(protocol, choice.clone(), parallel);
                enc.start();
                enc.apply(0, events()).unwrap();
                let first = enc.outputs[0].blocks.clone();
                let second = enc.apply(
                    0,
                    vec![
                        Event::ToolBegin {
                            index: 1,
                            name: "lookup".into(),
                            id: None,
                        },
                        Event::ToolArguments {
                            index: 1,
                            text: "{}".into(),
                        },
                        Event::ToolEnd { index: 1 },
                    ],
                );
                if parallel {
                    second.unwrap();
                    enc.finish(0, Termination::Eos, 12, 0, None).unwrap();
                } else {
                    let error = second.unwrap_err();
                    assert!(error.to_string().contains("more than one tool call"));
                    assert_eq!(enc.outputs[0].blocks, first);
                    let failed = enc.failure(&error);
                    assert!(!failed.done);
                    if protocol == Protocol::Responses {
                        assert_eq!(failed.data["type"], "response.failed");
                        assert_eq!(failed.data["response"]["parallel_tool_calls"], false);
                    } else {
                        assert!(failed.data["error"]["message"].is_string());
                    }
                }
            }
        }
    }
}

#[test]
fn disabled_parallel_calls_allow_zero_or_one_and_combine_with_required() {
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        for choice in [ToolChoice::Auto, ToolChoice::Required, ToolChoice::Named("lookup".into())] {
            for count in [0, 1] {
                let mut enc = encoder_with_parallel(protocol, choice.clone(), false);
                enc.apply(
                    0,
                    if count == 1 {
                        events()
                    } else {
                        vec![Event::Content("OK".into())]
                    },
                )
                .unwrap();
                let result = enc.finish(0, Termination::Eos, 12, 0, None);
                assert_eq!(result.is_ok(), count == 1 || choice == ToolChoice::Auto);
                if protocol == Protocol::Responses {
                    assert_eq!(enc.response()["parallel_tool_calls"], false);
                }
            }
        }
    }
}

#[test]
fn developer_and_system_instructions_lower_to_one_template_preamble() {
    let req = request(
        Protocol::Responses,
        json!({"instructions":"System policy", "input":[
            {"role":"developer","content":"Repository rules"},
            {"role":"user","content":"Do the task"},
            {"role":"developer","content":"Additional rules"}
        ]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages, ToolChoice::Auto).unwrap();
    assert_eq!(native.len(), 2);
    assert_eq!(
        native[0],
        json!({"role":"system","content":"System instructions:\nSystem policy\n\nDeveloper instructions:\nRepository rules\n\nDeveloper instructions:\nAdditional rules"})
    );
    assert_eq!(native[1]["role"], "user");
}
#[test]
fn response_history_retains_reasoning_calls_results_and_namespaces() {
    let req = request(
        Protocol::Responses,
        json!({"input":[{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"Think"}]},{"type":"function_call","call_id":"c","name":"run","namespace":"shell","arguments":"{\"x\":2}"},{"type":"function_call_output","call_id":"c","output":[{"type":"input_text","text":"ok"}]},{"role":"user","content":"Continue"}],"tools":[{"type":"namespace","name":"shell","tools":[{"type":"function","name":"run","parameters":{"type":"object"}}]}]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    assert_eq!(messages.len(), 3);
    assert!(
        matches!(&messages[0].parts[..],[Part::Reasoning(_),Part::ToolCall{name,..}] if name=="shell__run")
    );
    let native = template_messages(&messages, ToolChoice::Auto).unwrap();
    assert_eq!(native[0]["reasoning_content"], "Think");
    assert_eq!(native[0]["tool_calls"][0]["id"], "c");
    assert_eq!(native[1]["tool_call_id"], "c");
}
#[test]
fn messages_accept_webp_image_sources() {
    let req = request(
        Protocol::Messages,
        json!({"max_tokens":4,"messages":[{"role":"user",
        "content":[{"type":"image","source":{"type":"base64","media_type":"image/webp","data":"AA=="}}]}]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages, ToolChoice::Auto).unwrap();
    assert_eq!(
        native[0]["content"][0]["image_url"]["url"],
        "data:image/webp;base64,AA=="
    );
}

#[test]
fn messages_preserve_mixed_tool_results_and_images() {
    let req = request(
        Protocol::Messages,
        json!({"max_tokens":4,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"c","is_error":true,"content":[{"type":"text","text":"failed"}]},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}},{"type":"text","text":"after"}]}]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages, ToolChoice::Auto).unwrap();
    assert_eq!(native.len(), 3);
    assert_eq!(native[0]["content"], "before");
    assert_eq!(native[1]["is_error"], true);
    assert_eq!(
        native[2]["content"][0]["image_url"]["url"],
        "data:image/png;base64,AA=="
    );
}

#[test]
fn messages_preserve_inline_system_text_and_discard_message_controls() {
    let body = json!({"max_tokens":32,"system":"Initial instructions",
    "output_config":{"effort":"low"},"messages":[
        {"role":"user","content":"Hello"},
        {"role":"system","content":[{"type":"text","text":"Environment","cache_control":{"type":"ephemeral"}}],
            "output_config":{"effort":"xhigh"}},
        {"role":"assistant","content":"Ready"},
        {"role":"user","content":"Continue"},
        {"role":"system","content":[],"output_config":{"effort":"high"}}
    ]});
    for count in [false, true] {
        let req = decode(Protocol::Messages, body.clone(), count)
            .unwrap()
            .request;
        assert_eq!(req.options.effort, Some(inference::Effort::Low));
        let Input::Conversation(messages) = req.input else {
            panic!()
        };
        assert_eq!(
            template_messages(&messages, ToolChoice::Auto).unwrap(),
            vec![
                json!({"role":"system","content":"Initial instructions"}),
                json!({"role":"user","content":"Hello"}),
                json!({"role":"system","content":"Environment"}),
                json!({"role":"assistant","content":"Ready"}),
                json!({"role":"user","content":"Continue"}),
            ]
        );
    }
}

#[test]
fn messages_empty_system_controls_do_not_break_assistant_continuation() {
    let req = request(
        Protocol::Messages,
        json!({"max_tokens":32,"messages":[
            {"role":"user","content":"Finish the sentence"},
            {"role":"assistant","content":"The answer is"},
            {"role":"system","content":[],"output_config":{"effort":"xhigh"}},
            {"role":"system","content":"","thinking":{"type":"disabled"}}
        ]}),
    );
    assert_eq!(req.options.effort, None);
    assert_eq!(req.options.template.get("enable_thinking"), None);
    assert_eq!(req.options.template["continue_final_message"], true);
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    assert_eq!(messages.len(), 2);
}
#[test]
fn all_completion_prompt_forms_and_choice_counts() {
    for (value, prompts) in [
        (json!("x"), 1),
        (json!(["x", "y"]), 2),
        (json!([1, 2]), 1),
        (json!([[1, 2], [3]]), 2),
    ] {
        let req = request(
            Protocol::Completions,
            json!({"prompt":value,"n":3,"max_tokens":0,"echo":true}),
        );
        assert_eq!(req.generation_count().unwrap(), prompts * 3);
        assert_eq!(req.options.output_limit, Some(0));
    }
    let req = request(Protocol::Completions, json!({"prompt":[1,3,2]}));
    assert_eq!(
        req.input,
        Input::Raw(vec![RawPrompt::Tokens(vec![1, 3, 2])])
    );
    for value in [
        json!([]),
        json!([[]]),
        json!([-1]),
        json!([1, "x"]),
        json!(["x", [1]]),
    ] {
        assert!(decode(Protocol::Completions, json!({"prompt":value}), false).is_err());
    }
}
#[test]
fn unsupported_controls_are_rejected_before_generation() {
    for field in [
        json!({"store":true}),
        json!({"previous_response_id":"r"}),
        json!({"conversation":"c"}),
        json!({"background":true}),
        json!({"tool_choice":{"type":"function","name":"lookup"}}),
        json!({"truncation":"auto"}),
        json!({"tools":[{"type":"function","name":"f","strict":true}]}),
        json!({"tools":[{"type":"web_search"}]}),
        json!({"audio":{"voice":"alloy"}}),
    ] {
        let mut body = json!({"input":"x"});
        body.as_object_mut()
            .unwrap()
            .extend(field.as_object().unwrap().clone());
        assert!(decode(Protocol::Responses, body, false).is_err());
    }
    for field in [
        json!({"suffix":"tail"}),
        json!({"best_of":2}),
    ] {
        let mut body = json!({"prompt":"x"});
        body.as_object_mut()
            .unwrap()
            .extend(field.as_object().unwrap().clone());
        assert!(decode(Protocol::Completions, body, false).is_err());
    }
}
#[test]
fn reasoning_mapping_and_explicit_template_overrides() {
    for (budget, effort) in [
        (5000, inference::Effort::Low),
        (5001, inference::Effort::High),
        (20001, inference::Effort::Max),
    ] {
        let req = request(
            Protocol::Messages,
            json!({"max_tokens":30000,"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled","budget_tokens":budget}}),
        );
        assert_eq!(req.options.effort, Some(effort));
    }
    let req = request(
        Protocol::Responses,
        json!({"input":"x","reasoning":{"effort":"none"},"chat_template_kwargs":{"enable_thinking":true}}),
    );
    assert_eq!(req.options.template["enable_thinking"], true);
}

#[test]
fn messages_preserve_context_and_custom_function_schemas() {
    let body = json!({"max_tokens":32,"messages":[{"role":"user","content":"x"}],
        "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
        "tools":[{"type":"custom","name":"lookup","input_schema":{"type":"object","properties":{"query":{"type":"string"}}}}]});
    let req = request(Protocol::Messages, body.clone());
    assert!(!req.tools[0].custom);
    assert_eq!(
        req.tools[0].parameters["properties"]["query"]["type"],
        "string"
    );
    let mut edits = body;
    edits["context_management"]["edits"][0]["keep"] = json!({"type":"thinking_turns","value":1});
    assert!(decode(Protocol::Messages, edits, false).is_err());
}

#[test]
fn custom_tool_input_requires_a_string_before_completion() {
    for args in [r#"{"input":"hello"}"#, r#"{"input":3}"#, "{}"] {
        let mut tools = tools();
        tools[0].custom = true;
        let mut e = Encoder::new(
            Protocol::Responses,
            "test".into(),
            "model".into(),
            1,
            1,
            vec![PromptUsage {
                tokens: 1,
                image_tokens: 0,
            }],
            tools,
            ToolChoice::Auto,
            true,
            false,
            json!({}),
        );
        let result = e.apply(
            0,
            vec![
                Event::ToolBegin {
                    index: 0,
                    name: "lookup".into(),
                    id: None,
                },
                Event::ToolArguments {
                    index: 0,
                    text: args.into(),
                },
                Event::ToolEnd { index: 0 },
            ],
        );
        if args.contains("hello") {
            assert!(result.is_ok());
            e.finish(0, Termination::Eos, 5, 0, None).unwrap();
            assert_eq!(e.response()["output"][0]["input"], "hello");
        } else {
            assert!(result.is_err());
            assert_eq!(
                e.failure(&"invalid tool input".into()).data["type"],
                "response.failed"
            );
        }
    }
}
#[test]
fn tool_name_collisions_and_unknown_custom_formats_are_explicit_errors() {
    for tools in [
        json!([{"type":"function","name":"a__b"},{"type":"namespace","name":"a","tools":[{"type":"function","name":"b"}]}]),
        json!([{"type":"custom","name":"patch","format":{"type":"grammar","syntax":"unknown","definition":"start: /./"}}]),
    ] {
        assert!(
            decode(
                Protocol::Responses,
                json!({"input":"x","tools":tools}),
                false
            )
            .is_err()
        );
    }
}
#[test]
fn messages_stream_reconstructs_ordered_buffered_blocks() {
    let mut encoder = encoder(Protocol::Messages);
    let mut frames = encoder.start();
    frames.extend(encoder.apply(0, events()).unwrap());
    frames.extend(encoder.finish(0, Termination::Eos, 12, 3, None).unwrap());
    let expected = encoder.response();
    assert_eq!(expected["stop_reason"], "tool_use");
    assert_eq!(expected["usage"]["input_tokens"], 14);
    assert_eq!(expected["usage"]["cache_read_input_tokens"], 3);
    assert_eq!(expected["usage"]["output_tokens"], 12);
    let mut blocks = vec![];
    for frame in &frames {
        let i = frame.data["index"].as_u64().unwrap_or(0) as usize;
        match frame.data["type"].as_str() {
            Some("content_block_start") => blocks.push(frame.data["content_block"].clone()),
            Some("content_block_delta") => {
                let delta = &frame.data["delta"];
                match delta["type"].as_str().unwrap() {
                    "text_delta" => {
                        let s = blocks[i]["text"].as_str().unwrap().to_owned()
                            + delta["text"].as_str().unwrap();
                        blocks[i]["text"] = json!(s)
                    }
                    "thinking_delta" => {
                        let s = blocks[i]["thinking"].as_str().unwrap().to_owned()
                            + delta["thinking"].as_str().unwrap();
                        blocks[i]["thinking"] = json!(s)
                    }
                    "signature_delta" => blocks[i]["signature"] = delta["signature"].clone(),
                    "input_json_delta" => {
                        let s = blocks[i]
                            .get("partial")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned()
                            + delta["partial_json"].as_str().unwrap();
                        blocks[i]["partial"] = json!(s)
                    }
                    _ => panic!(),
                }
            }
            Some("content_block_stop") if blocks[i]["type"] == "tool_use" => {
                let partial = blocks[i]
                    .as_object_mut()
                    .unwrap()
                    .remove("partial")
                    .unwrap();
                blocks[i]["input"] = serde_json::from_str(partial.as_str().unwrap()).unwrap();
            }
            _ => {}
        }
    }
    assert_eq!(json!(blocks), expected["content"]);
    assert_eq!(frames.last().unwrap().data["type"], "message_stop");
}
#[test]
fn responses_stream_has_stable_ids_sequence_and_terminal_output() {
    let mut encoder = encoder(Protocol::Responses);
    let mut frames = encoder.start();
    frames.extend(encoder.apply(0, events()).unwrap());
    frames.extend(encoder.finish(0, Termination::Eos, 12, 3, None).unwrap());
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.data["sequence_number"], i as u64);
    }
    let done: Vec<_> = frames
        .iter()
        .filter(|f| f.data["type"] == "response.output_item.done")
        .map(|f| f.data["item"].clone())
        .collect();
    assert_eq!(json!(done), encoder.response()["output"]);
    assert_eq!(frames.last().unwrap().data["response"], encoder.response());
    assert_eq!(
        encoder.response()["usage"]["input_tokens_details"]["cached_tokens"],
        3
    );
    let mut history = encoder.response()["output"].as_array().unwrap().clone();
    history.push(json!({"type":"function_call_output","call_id":"call_1","output":"found"}));
    assert!(
        decode(
            Protocol::Responses,
            json!({"input":history,"tools":[{"type":"function","name":"lookup"}]}),
            false
        )
        .is_ok()
    );
}
#[test]
fn thinking_markers_round_trip_and_foreign_opaque_history_is_rejected() {
    let mut encoder = encoder(Protocol::Messages);
    encoder
        .apply(0, vec![Event::Reasoning("think".into())])
        .unwrap();
    encoder
        .finish(0, Termination::OutputLimit, 1, 0, None)
        .unwrap();
    let output = encoder.response();
    assert!(decode(Protocol::Messages,json!({"messages":[{"role":"assistant","content":output["content"]},{"role":"user","content":"next"}],"max_tokens":2}),false).is_ok());
    assert!(
        decode(
            Protocol::Responses,
            json!({"input":[{"type":"reasoning","summary":[],"encrypted_content":"opaque"}]}),
            false
        )
        .is_err()
    );
}
#[test]
fn usage_reports_uncached_partial_and_full_prefix_reuse() {
    for protocol in [
        Protocol::Chat,
        Protocol::Completions,
        Protocol::Messages,
        Protocol::Responses,
    ] {
        for (generated, cached) in [(0, 0), (4, 0), (4, 9), (4, 17)] {
            let mut enc = encoder(protocol);
            let start = enc.start();
            if protocol == Protocol::Messages {
                assert_eq!(start[0].data["message"]["usage"]["input_tokens"], 17);
                assert_eq!(
                    start[0].data["message"]["usage"]["cache_read_input_tokens"],
                    0
                );
            }
            let frames = enc
                .finish(0, Termination::OutputLimit, generated, cached, None)
                .unwrap();
            let response = enc.response();
            let usage = &response["usage"];
            match protocol {
                Protocol::Chat | Protocol::Completions => {
                    assert_eq!(
                        usage,
                        &json!({
                            "prompt_tokens":17,"completion_tokens":generated,
                            "total_tokens":17 + generated,
                            "prompt_tokens_details":{"cached_tokens":cached},
                        })
                    );
                    assert_eq!(frames[frames.len() - 2].data["usage"], *usage);
                    assert_eq!(frames[frames.len() - 2].data["choices"], json!([]));
                    assert!(frames.last().unwrap().done);
                    assert!(
                        frames[..frames.len() - 2]
                            .iter()
                            .all(|f| f.data["usage"].is_null())
                    );
                }
                Protocol::Messages => {
                    assert_eq!(
                        usage,
                        &json!({
                            "input_tokens":17-cached,"output_tokens":generated,
                            "cache_creation_input_tokens":0,"cache_read_input_tokens":cached,
                        })
                    );
                    let delta = frames
                        .iter()
                        .find(|f| f.data["type"] == "message_delta")
                        .unwrap();
                    assert_eq!(delta.data["usage"], *usage);
                }
                Protocol::Responses => {
                    assert_eq!(
                        usage,
                        &json!({
                            "input_tokens":17,"output_tokens":generated,
                            "total_tokens":17 + generated,
                            "input_tokens_details":{"cached_tokens":cached},
                        })
                    );
                    assert_eq!(frames.last().unwrap().data["response"]["usage"], *usage);
                }
            }
        }
    }
}

#[test]
fn image_usage_is_a_prompt_subset_and_stream_usage_remains_opt_in() {
    for include_usage in [false, true] {
        let mut enc = Encoder::new(
            Protocol::Chat,
            "c".into(),
            "m".into(),
            1,
            1,
            vec![PromptUsage {
                tokens: 17,
                image_tokens: 8,
            }],
            vec![],
            ToolChoice::Auto,
            true,
            include_usage,
            json!({}),
        );
        let mut frames = enc.start();
        frames.extend(enc.apply(0, vec![Event::Content("answer".into())]).unwrap());
        frames.extend(enc.finish(0, Termination::Eos, 4, 9, None).unwrap());
        let expected = json!({"prompt_tokens":17,"completion_tokens":4,"total_tokens":21,
            "prompt_tokens_details":{"cached_tokens":9,"multimodal_tokens":{"image":8}}});
        assert_eq!(enc.response()["usage"], expected);
        let usage_frames: Vec<_> = frames
            .iter()
            .filter(|f| f.data["usage"].is_object())
            .collect();
        if include_usage {
            assert_eq!(usage_frames.len(), 1);
            assert_eq!(usage_frames[0].data["usage"], expected);
        } else {
            assert!(frames.iter().all(|f| f.data.get("usage").is_none()));
        }
    }
}

#[test]
fn continuous_usage_counts_hidden_and_merged_tokens_without_extra_text_frames() {
    for protocol in [Protocol::Chat, Protocol::Completions] {
        let mut encoder = Encoder::new(protocol, "c".into(), "m".into(), 1, 1,
            vec![PromptUsage { tokens: 17, image_tokens: 0 }], vec![], ToolChoice::Auto,
            true, true, json!({}));
        encoder.set_continuous_usage(true);
        // Three generated byte tokens can decode to one text fragment.
        for _ in 0..3 { encoder.observe_generated_token(0); }
        let frames = encoder.apply(0, vec![Event::Content("€".into())]).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data["usage"]["completion_tokens"], 3);
        // Structural tokens are counted even when the decoder produces no text.
        encoder.observe_generated_token(0);
        encoder.observe_generated_token(0);
        assert!(encoder.apply(0, vec![]).unwrap().is_empty());
        let terminal = encoder.finish(0, Termination::Eos, 5, 0, None).unwrap();
        for frame in terminal.iter().filter(|frame| frame.data["usage"].is_object()) {
            assert_eq!(frame.data["usage"]["completion_tokens"], 5);
            assert_eq!(frame.data["usage"]["total_tokens"], 22);
        }
        assert_eq!(encoder.response()["usage"]["completion_tokens"], 5);
    }
}

#[test]
fn continuous_usage_sums_choices_without_counting_finished_choices_twice() {
    let mut encoder = Encoder::new(Protocol::Chat, "c".into(), "m".into(), 1, 2,
        vec![PromptUsage { tokens: 17, image_tokens: 0 }], vec![], ToolChoice::Auto,
        true, true, json!({}));
    encoder.set_continuous_usage(true);
    for _ in 0..3 { encoder.observe_generated_token(0); }
    for _ in 0..2 { encoder.observe_generated_token(1); }
    let frames = encoder.apply(1, vec![Event::Content("second".into())]).unwrap();
    assert_eq!(frames[0].data["choices"][0]["index"], 1);
    assert_eq!(frames[0].data["usage"]["completion_tokens"], 5);
    encoder.finish(0, Termination::Eos, 3, 9, None).unwrap();
    encoder.observe_generated_token(1);
    let frames = encoder.apply(1, vec![Event::Content(" choice".into())]).unwrap();
    assert_eq!(frames[0].data["usage"]["completion_tokens"], 6);
    encoder.finish(1, Termination::OutputLimit, 3, 9, None).unwrap();
    assert_eq!(encoder.usage_value()["completion_tokens"], 6);
    assert_eq!(encoder.usage_value()["prompt_tokens"], 17);
}

#[test]
fn batch_stream_indices_echo_and_totals() {
    let mut encoder = Encoder::new(
        Protocol::Completions,
        "c".into(),
        "m".into(),
        1,
        2,
        vec![
            PromptUsage {
                tokens: 3,
                image_tokens: 0,
            },
            PromptUsage {
                tokens: 4,
                image_tokens: 0,
            },
        ],
        vec![],
        ToolChoice::Auto,
        true,
        true,
        json!({}),
    );
    for i in [3, 0, 2, 1] {
        let frames = encoder
            .apply(i, vec![Event::Content(format!("choice{i}"))])
            .unwrap();
        assert_eq!(frames[0].data["choices"][0]["index"], i);
        let frames = encoder
            .finish(i, Termination::OutputLimit, 2, [1, 3, 2, 4][i], None)
            .unwrap();
        if i == 1 {
            assert!(frames.last().unwrap().done);
            assert_eq!(
                frames[frames.len() - 2].data["usage"],
                encoder.response()["usage"]
            );
        } else {
            assert!(!frames.last().unwrap().done);
        }
    }
    let response = encoder.response();
    assert_eq!(
        response["usage"],
        json!({"prompt_tokens":7,"completion_tokens":8,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":3}})
    );
    for i in 0..4 {
        assert_eq!(response["choices"][i]["text"], format!("choice{i}"));
    }
}
#[test]
fn failures_have_no_success_terminal_and_stops_remain_exact() {
    for protocol in [Protocol::Chat, Protocol::Messages, Protocol::Responses] {
        let mut enc = encoder(protocol);
        enc.apply(0, vec![Event::Content("partial".into())])
            .unwrap();
        let frame = enc.failure(&"broken".into());
        assert!(!frame.done);
        if protocol == Protocol::Responses {
            assert_eq!(frame.data["type"], "response.failed");
            assert_eq!(frame.data["response"]["status"], "failed");
        }
    }
    let mut enc = encoder(Protocol::Messages);
    enc.finish(0, Termination::UserStop, 3, 0, Some("世界".into()))
        .unwrap();
    assert_eq!(enc.response()["stop_sequence"], "世界");
    assert_eq!(enc.response()["stop_reason"], "stop_sequence");
}

#[test]
fn stream_errors_preserve_typed_codes_and_protocol_envelopes() {
    use crate::{AdmissionError, Error, ExecutionError};
    for (error, kind, code) in [
        (
            Error::admission(AdmissionError::Capacity, "full"),
            "server_error",
            "capacity_exhausted",
        ),
        (
            Error::admission(AdmissionError::Unavailable, "stopped"),
            "server_error",
            "server_unavailable",
        ),
        (
            Error::admission(AdmissionError::Context, "too long"),
            "invalid_request_error",
            "context_length_exceeded",
        ),
        (
            Error::execution(ExecutionError::Capacity, "slow consumer"),
            "server_error",
            "execution_capacity_exceeded",
        ),
        (
            Error::execution(ExecutionError::State, "unregistered tool"),
            "server_error",
            "server_error",
        ),
    ] {
        for protocol in [
            Protocol::Chat,
            Protocol::Completions,
            Protocol::Responses,
            Protocol::Messages,
        ] {
            let mut enc = encoder(protocol);
            let frame = enc.failure(&error);
            assert!(!frame.done);
            let body = if protocol == Protocol::Responses {
                assert_eq!(frame.event.as_deref(), Some("response.failed"));
                assert_eq!(frame.data["response"]["status"], "failed");
                &frame.data["response"]["error"]
            } else {
                &frame.data["error"]
            };
            assert_eq!(body["message"], error.to_string());
            if protocol == Protocol::Messages {
                assert_eq!(frame.event.as_deref(), Some("error"));
                assert_eq!(
                    body["type"],
                    if kind == "server_error" {
                        "api_error"
                    } else {
                        kind
                    }
                );
                assert!(body.get("code").is_none());
            } else {
                assert_eq!(body["code"], code);
                if protocol != Protocol::Responses {
                    assert_eq!(body["type"], kind);
                    assert_eq!(body.get("param"), Some(&Value::Null));
                }
            }
        }
    }
}

#[test]
fn output_limit_marks_only_the_open_response_item_incomplete() {
    let mut e = encoder(Protocol::Responses);
    let mut frames = e
        .apply(
            0,
            vec![
                Event::Reasoning("think".into()),
                Event::Content("partial".into()),
            ],
        )
        .unwrap();
    frames.extend(e.finish(0, Termination::OutputLimit, 4, 0, None).unwrap());
    let response = e.response();
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][1]["status"], "incomplete");
    let done: Vec<_> = frames
        .iter()
        .filter(|f| f.data["type"] == "response.output_item.done")
        .map(|f| f.data["item"].clone())
        .collect();
    assert_eq!(json!(done), response["output"]);
}

#[test]
fn chat_history_without_call_ids_retains_template_semantics() {
    let req = request(
        Protocol::Chat,
        json!({"messages":[
            {"role":"user","content":"Find it"},
            {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{}"}}]},
            {"role":"tool","content":"found"}
        ]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    let native = template_messages(&messages, ToolChoice::Auto).unwrap();
    assert_eq!(native.len(), 3);
    assert!(native[1]["tool_calls"][0].get("id").is_none());
    assert!(native[2].get("tool_call_id").is_none());
}

#[test]
fn empty_text_messages_preserve_conversation_roles() {
    let req = request(
        Protocol::Chat,
        json!({"messages":[
            {"role":"system","content":""}, {"role":"user","content":""},
            {"role":"assistant","content":null}
        ]}),
    );
    let Input::Conversation(messages) = req.input else {
        panic!()
    };
    assert_eq!(
        template_messages(&messages, ToolChoice::Auto).unwrap(),
        vec![
            json!({"role":"system","content":""}),
            json!({"role":"user","content":""}),
            json!({"role":"assistant","content":""})
        ]
    );
}

#[test]
fn generation_logprob_controls_have_endpoint_specific_types_and_limits() {
    for (protocol, base, accepted, rejected) in [
        (Protocol::Chat, json!({"messages":[{"role":"user","content":"x"}]}),
            vec![json!({"logprobs":true,"top_logprobs":20}), json!({"top_logprobs":0})],
            vec![json!({"top_logprobs":1}), json!({"logprobs":1}), json!({"logprobs":true,"top_logprobs":21})]),
        (Protocol::Completions, json!({"prompt":"x"}), vec![json!({"logprobs":0}), json!({"logprobs":20}), json!({"logprobs":1,"echo":true})],
            vec![json!({"logprobs":true}), json!({"logprobs":-1})]),
        (Protocol::Responses, json!({"input":"x"}),
            vec![json!({"top_logprobs":20,"include":["message.output_text.logprobs"]}), json!({"include":["unrelated"]})],
            vec![json!({"top_logprobs":1}), json!({"logprobs":true})]),
        (Protocol::Messages, json!({"messages":[{"role":"user","content":"x"}],"max_tokens":1}), vec![],
            vec![json!({"logprobs":false}), json!({"top_logprobs":0}), json!({"return_tokens_as_token_ids":true})]),
    ] {
        for (fields, valid) in accepted.into_iter().map(|v| (v,true)).chain(rejected.into_iter().map(|v| (v,false))) {
            let mut body = base.clone();
            body.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
            assert_eq!(decode(protocol, body.clone(), false).is_ok(), valid, "{protocol:?}: {body}");
        }
    }
    let body = json!({"messages":[{"role":"assistant","content":"Cletus:"}], "max_tokens":1,
        "continue_final_message":true,"add_generation_prompt":false,"logprobs":true,"top_logprobs":20});
    let request = decode(Protocol::Chat, body, false).unwrap().request;
    assert_eq!(request.options.logprobs.unwrap().top,20);
    assert!(request.options.sampling.temperature.is_none());
}

#[test]
fn prompt_logprob_request_types_limits_streaming_and_endpoint_placement() {
    for protocol in [Protocol::Completions, Protocol::Chat] {
        let base = if protocol == Protocol::Completions { json!({"prompt":"x"}) }
            else { json!({"messages":[{"role":"user","content":"x"}]}) };
        for top in [0, 1, 20] {
            let mut body = base.clone(); body["prompt_logprobs"] = json!(top);
            let decoded = super::decode::decode(protocol, body.clone(), false).unwrap();
            assert_eq!(decoded.request.options.prompt_logprobs, Some(top));
            body["stream"] = json!(true);
            assert!(super::decode::decode(protocol, body, false).is_err());
        }
        for bad in [json!(-1), json!(21), json!(true), json!(1.5), json!("1")] {
            let mut body = base.clone(); body["prompt_logprobs"] = bad;
            assert!(super::decode::decode(protocol, body, false).is_err());
        }
    }
    for protocol in [Protocol::Messages, Protocol::Responses] {
        assert!(super::decode::decode(protocol, json!({"prompt_logprobs":0}), false).is_err());
    }
    let decoded = super::decode::decode(Protocol::Completions,
        json!({"prompt":[1,2],"echo":true,"logprobs":2,"max_tokens":0,"stream":true}), false).unwrap();
    assert_eq!(decoded.request.options.prompt_logprobs, Some(2));
    assert_eq!(decoded.request.options.logprobs.unwrap().top, 2);
}

#[test]
fn requested_logprob_ids_require_enablement_and_validate_before_deduplication() {
    for (protocol, base, enable) in [
        (
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"x"}]}),
            json!({"logprobs":true}),
        ),
        (
            Protocol::Completions,
            json!({"prompt":"x"}),
            json!({"logprobs":0}),
        ),
        (
            Protocol::Responses,
            json!({"input":"x"}),
            json!({"include":["message.output_text.logprobs"]}),
        ),
    ] {
        for empty in [Value::Null, json!([])] {
            let mut body = base.clone();
            body["logprob_token_ids"] = empty;
            assert!(decode(protocol, body, false).is_ok());
        }
        let mut body = base.clone();
        body["logprob_token_ids"] = json!([9, 2, 9, 1]);
        assert!(decode(protocol, body.clone(), false).is_err());
        body.as_object_mut()
            .unwrap()
            .extend(enable.as_object().unwrap().clone());
        let options = decode(protocol, body.clone(), false)
            .unwrap()
            .request
            .options
            .logprobs
            .unwrap();
        assert_eq!(options.ids, [9, 2, 1]);
        for count in [1, 20, 21, 127, 128] {
            body["logprob_token_ids"] = json!((0..count).collect::<Vec<_>>());
            assert_eq!(
                decode(protocol, body.clone(), false)
                    .unwrap()
                    .request
                    .options
                    .logprobs
                    .unwrap()
                    .ids
                    .len(),
                count
            );
        }
        for invalid in [
            json!(true),
            json!(1),
            json!([true]),
            json!([1.0]),
            json!([1.5]),
            json!([-1]),
            json!([2147483648u64]),
            json!([u64::MAX]),
            json!(["1"]),
            json!(vec![1; 129]),
        ] {
            body["logprob_token_ids"] = invalid;
            assert!(decode(protocol, body.clone(), false).is_err(), "{body}");
        }
    }
    let echoed = decode(
        Protocol::Completions,
        json!({"prompt":[1,2],"echo":true,"logprobs":20,"logprob_token_ids":[3],"max_tokens":0}),
        false,
    )
    .unwrap();
    assert_eq!(echoed.request.options.prompt_logprobs, Some(20));
    assert_eq!(echoed.request.options.logprobs.unwrap().ids, [3]);
    for ids in [Value::Null, json!([]), json!([1])] {
        assert!(decode(Protocol::Messages,
            json!({"messages":[{"role":"user","content":"x"}],"max_tokens":1,"logprob_token_ids":ids}), false).is_err());
    }
}
