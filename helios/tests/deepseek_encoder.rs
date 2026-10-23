#![cfg(feature = "frontend")]

#[path = "support/native_prefix.rs"]
mod native_prefix;

use deepseek_recipe_core::{
    conversation::{Conversation, ReasoningEffort, ResponseFormat},
    messages::InputMessage,
    tools::{ToolChoice, ToolDefinition},
};
use deepseek_recipe_encoding::{
    EncodingError, PromptEncoding, dsv4::DeepseekV4Encoding, dsv41::DeepseekV41Encoding,
};
use flare_helios::frontend::{
    Asset, Frontend, FrontendAssets, FrontendSettings, PromptEncoder, TokenizerSettings,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokenizers::{AddedToken, Tokenizer};

fn frontend(encoder: PromptEncoder, defaults: Value) -> Frontend {
    configured_frontend(encoder, defaults, 0, Some(8)).unwrap()
}

fn configured_frontend(
    encoder: PromptEncoder,
    defaults: Value,
    template_encoding: i32,
    placeholder: Option<u32>,
) -> flare_helios::Result<Frontend> {
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab([("[UNK]".into(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer
        .add_special_tokens(
            [
                "<｜begin▁of▁sentence｜>",
                "<｜System｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<think>",
                "</think>",
                "<｜end▁of▁sentence｜>",
                "<｜deepseek_image｜>",
            ]
            .map(|s| AddedToken::from(s, true)),
        )
        .unwrap();
    let bytes = tokenizer.to_string(false).unwrap().into_bytes();
    let path = std::env::temp_dir().join(format!(
        "flare-encoder-{}-{:?}.json",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let settings: FrontendSettings = serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[],"prompt_encoder":i32::from(encoder),"image_placeholder_token_id":placeholder,"template_encoding":template_encoding})).unwrap();
    let template_path = path.with_extension("jinja");
    let template_bytes = b"{% mark %}{{ messages|tojson }}";
    let chat_template = if encoder == PromptEncoder::Jinja {
        std::fs::write(&template_path, template_bytes).unwrap();
        Some(Asset {
            path: template_path.clone(),
            sha256: format!("{:x}", Sha256::digest(template_bytes)),
            range: None,
        })
    } else {
        None
    };
    let frontend = Frontend::load_with_settings(
        FrontendAssets {
            tokenizer: Asset {
                path: path.clone(),
                sha256: format!("{:x}", Sha256::digest(&bytes)),
                range: None,
            },
            tokens: TokenizerSettings::default(),
            chat_template,
        },
        defaults.as_object().unwrap().clone(),
        &settings,
    );
    std::fs::remove_file(path).unwrap();
    if encoder == PromptEncoder::Jinja {
        std::fs::remove_file(template_path).unwrap();
    }
    frontend
}

fn official(encoder: PromptEncoder, conversation: &Conversation) -> String {
    match encoder {
        PromptEncoder::DeepseekV4 => {
            DeepseekV4Encoding::new()
                .render_conversation(conversation, &[])
                .prompt
        }
        PromptEncoder::DeepseekV41 => {
            DeepseekV41Encoding::new()
                .render_conversation(conversation, &[])
                .prompt
        }
        _ => unreachable!(),
    }
}

fn effort_conversation(thinking: bool, effort: ReasoningEffort) -> Conversation {
    Conversation {
        messages: vec![
            InputMessage::System { content: "Policy".into() },
            InputMessage::User { content: "Hello".into(), image_sources: vec![] },
        ],
        thinking_mode: thinking,
        reasoning_effort: Some(effort),
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        response_format: ResponseFormat::Text,
    }
}

#[test]
fn numeric_effort_scores_cover_the_interval_and_match_the_checked_encoder() {
    let frontend = frontend(PromptEncoder::DeepseekV41, json!({}));
    let messages = [json!({"role":"system","content":"Policy"}),
                    json!({"role":"user","content":"Hello"})];
    let conversation = effort_conversation(true, ReasoningEffort::High);
    for score in 1u8..=100 {
        let expected = DeepseekV41Encoding::new()
            .with_reasoning_effort_score(score).unwrap()
            .render_conversation(&conversation, &[]).prompt;
        let actual = frontend.render_chat(&messages,
            json!({"reasoning_effort":score}).as_object().unwrap()).unwrap();
        assert_eq!(actual.text, expected, "score {score}");
        assert_eq!(actual.token_ids, frontend.encode(&expected, false).unwrap(), "score {score}");
        assert_eq!(actual.text.matches("Reasoning Effort:").count(), 1);
        assert!(actual.text.contains(&format!("Reasoning Effort: {score} (range 1-100")));
        if score == 80 {
            assert!(actual.text.starts_with(concat!(
                "<｜begin▁of▁sentence｜><｜System｜>",
                "Reasoning Effort: 80 (range 1-100, the higher the value, the more thorough the reasoning)\n\nPolicy"
            )));
        }
    }
    for score in [0u8, 101, 255] {
        assert!(matches!(DeepseekV41Encoding::new().with_reasoning_effort_score(score),
            Err(EncodingError::InvalidReasoningEffortScore(value)) if value == score));
    }
}

#[test]
fn numeric_effort_preserves_content_block_checkpoint_hints() {
    let frontend = frontend(PromptEncoder::DeepseekV41, json!({}));
    for score in [1, 80, 100] {
        for thinking in [false, true] {
            let mut messages = [
                json!({"role":"system","content":"Policy"}),
                json!({
                    "role":"user","content":[
                        {"type":"text","text":"Stable earlier history"},
                        {"type":"text","text":"Changing suffix"}]
                }),
            ];
            let options = json!({"reasoning_effort":score,"enable_thinking":thinking});
            let plain = frontend
                .render_chat(&messages, options.as_object().unwrap())
                .unwrap();
            messages[1]["_checkpoint_hints"] = json!([1]);
            let marked = frontend
                .render_chat(&messages, options.as_object().unwrap())
                .unwrap();
            assert_eq!(marked.text, plain.text);
            assert_eq!(marked.token_ids, plain.token_ids);
            assert_eq!(
                marked
                    .text
                    .contains(&format!("Reasoning Effort: {score} (range 1-100")),
                thinking
            );
            let end = marked.text.find("Stable earlier history").unwrap()
                + "Stable earlier history".len();
            let encoding = frontend
                .tokenizer()
                .encode(marked.text.as_str(), false)
                .unwrap();
            let cut = encoding
                .get_offsets()
                .iter()
                .take_while(|&&(_, end_byte)| end_byte <= end)
                .count();
            assert!(!plain.message_boundaries.contains(&cut));
            assert!(marked.message_boundaries.contains(&cut));
            assert_eq!(
                marked.message_boundaries.len(),
                plain.message_boundaries.len() + 1
            );
        }
    }
}

#[test]
fn numeric_aliases_preserve_complete_named_prompts_and_tokens() {
    let frontend = frontend(PromptEncoder::DeepseekV41, json!({}));
    let messages = [json!({"role":"system","content":"Policy"}),
                    json!({"role":"user","content":"Hello"})];
    for thinking in [false, true] {
        for (name, score, effort) in [
            ("low", 50, ReasoningEffort::Low),
            ("high", 75, ReasoningEffort::High),
            ("xhigh", 75, ReasoningEffort::Xhigh),
            ("max", 100, ReasoningEffort::Max),
        ] {
            let named = frontend.render_chat(&messages,
                json!({"enable_thinking":thinking,"reasoning_effort":name}).as_object().unwrap()).unwrap();
            let numeric = frontend.render_chat(&messages,
                json!({"enable_thinking":thinking,"reasoning_effort":score}).as_object().unwrap()).unwrap();
            assert_eq!(named.text, official(PromptEncoder::DeepseekV41, &effort_conversation(thinking, effort)));
            assert_eq!(numeric.text, named.text, "{name}, thinking={thinking}");
            assert_eq!(numeric.token_ids, named.token_ids);
            if !thinking {
                assert!(!numeric.text.contains("Reasoning Effort:"));
                assert!(numeric.text.ends_with("</think>"));
            }
        }
    }
}

#[test]
fn numeric_effort_types_and_native_v4_restrictions_are_explicit() {
    let invalid = [json!(0), json!(101), json!(-1), json!(80.0), json!(80.5),
        json!(true), json!(false), json!("80"), json!(u64::MAX),
        serde_json::from_str::<Value>("18446744073709551616").unwrap()];
    let messages = [json!({"role":"user","content":"Hello"})];
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let frontend = frontend(encoder, json!({}));
        for value in &invalid {
            let options = json!({"reasoning_effort":value});
            assert!(frontend.render_chat(&messages, options.as_object().unwrap()).is_err(),
                "{encoder:?}: {value}");
            assert!(configured_frontend(encoder, options, 0, None).is_err(),
                "invalid default for {encoder:?}: {value}");
        }
    }
    let v4 = frontend(PromptEncoder::DeepseekV4, json!({}));
    for score in 1..=100 {
        assert!(v4.render_chat(&messages,
            json!({"reasoning_effort":score}).as_object().unwrap()).is_err());
    }
    assert!(configured_frontend(PromptEncoder::DeepseekV4,
        json!({"reasoning_effort":80}), 0, None).is_err());
}

#[test]
fn numeric_defaults_and_request_overrides_do_not_leak_between_renders() {
    let numeric = frontend(PromptEncoder::DeepseekV41, json!({"reasoning_effort":80}));
    let named = frontend(PromptEncoder::DeepseekV41, json!({"reasoning_effort":"low"}));
    let messages = [json!({"role":"user","content":"Hello"})];
    for (options, expected) in [
        (json!({}), 80), (json!({"reasoning_effort":1}), 1),
        (json!({"reasoning_effort":"max"}), 100),
        (json!({"reasoning_effort":null}), 75), (json!({}), 80),
    ] {
        let prompt = numeric.render_chat(&messages, options.as_object().unwrap()).unwrap();
        assert!(prompt.text.contains(&format!("Reasoning Effort: {expected} (range 1-100")));
        assert!(named.render_chat(&messages, &Map::new()).unwrap().text.contains("Reasoning Effort: 50 (range 1-100"));
    }
    let off = numeric.render_chat(&messages,
        json!({"enable_thinking":false,"reasoning_effort":80}).as_object().unwrap()).unwrap();
    assert!(!off.text.contains("Reasoning Effort:"));
    assert!(off.text.ends_with("</think>"));
}

#[test]
fn official_rendering_and_system_prefixes_follow_both_encoders() {
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let frontend = frontend(encoder, json!({}));
        for thinking in [false, true] {
            for with_tools in [false, true] {
                for with_json in [false, true] {
                    let messages = [
                        json!({"role":"system","content":""}),
                        json!({"role":"system","content":"Policy"}),
                        json!({"role":"user","content":"Hello"}),
                    ];
                    let tools = if with_tools {
                        vec![ToolDefinition {
                            name: "lookup".into(),
                            description: None,
                            parameters: json!({"type":"object"}),
                            strict: None,
                        }]
                    } else {
                        vec![]
                    };
                    let mut options = json!({"enable_thinking":thinking,"response_format":if with_json {"json_object"} else {"text"}}).as_object().unwrap().clone();
                    if with_tools {
                        options.insert("tools".into(), json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]));
                    }
                    let conversation = Conversation {
                        messages: vec![
                            InputMessage::System { content: "".into() },
                            InputMessage::System {
                                content: "Policy".into(),
                            },
                            InputMessage::User {
                                content: "Hello".into(),
                                image_sources: vec![],
                            },
                        ],
                        thinking_mode: thinking,
                        reasoning_effort: Some(ReasoningEffort::High),
                        tools,
                        tool_choice: ToolChoice::Auto,
                        response_format: if with_json {
                            ResponseFormat::JsonObject
                        } else {
                            ResponseFormat::Text
                        },
                    };
                    let expected = official(encoder, &conversation);
                    let actual = frontend.render_chat(&messages, &options).unwrap();
                    assert_eq!(actual.text, expected);
                    assert_eq!(actual.token_ids, frontend.encode(&expected, false).unwrap());
                    options.insert("add_generation_prompt".into(), false.into());
                    let prefix = native_prefix::without_generation(&frontend, &messages[..2], &options);
                    assert!(!prefix.token_ids.is_empty());
                    assert!(actual.token_ids.starts_with(&prefix.token_ids));
                    assert!(actual.text.starts_with(&prefix.text));
                    assert!(!prefix.text.ends_with("<｜Assistant｜>"));
                    assert!(!prefix.text.ends_with("<｜User｜>"));
                }
            }
        }
    }
}

#[test]
fn options_use_official_defaults_aliases_and_explicit_errors() {
    let frontend = frontend(
        PromptEncoder::DeepseekV41,
        json!({"thinking_mode":"thinking"}),
    );
    let messages = [json!({"role":"user","content":"Hello"})];
    assert!(
        frontend
            .render_chat(&messages, &Map::new())
            .unwrap()
            .text
            .ends_with("<think>")
    );
    let options = json!({"enable_thinking":false})
        .as_object()
        .unwrap()
        .clone();
    assert!(
        frontend
            .render_chat(&messages, &options)
            .unwrap()
            .text
            .ends_with("</think>")
    );
    for value in [
        json!({"add_generation_prompt":false}),
        json!({"continue_final_message":true}),
        json!({"continue_final_message":"reasoning_content"}),
        json!({"drop_thinking":false}),
        json!({"reasoning_effort":101}),
        json!({"tool_choice":false}),
        json!({"unrecognized":true}),
        json!({"bos_token":"arbitrary"}),
    ] {
        assert!(
            frontend
                .render_chat(&messages, value.as_object().unwrap())
                .is_err(),
            "{value}"
        );
    }
    let options = json!({"thinking_mode":"chat","tools":[{"function":{"name":"lookup"}}],"tool_choice":"required"});
    let prompt = frontend
        .render_chat(&messages, options.as_object().unwrap())
        .unwrap();
    assert!(
        prompt
            .text
            .ends_with("<｜Assistant｜></think>\n\n<｜DSML｜ calls>\n")
    );
    assert!(frontend.render_chat(&[], &Map::new()).is_err());
    assert!(
        frontend
            .render_chat(
                &[json!({"role":"assistant","content":"prefix","task":"action"})],
                &Map::new()
            )
            .is_err()
    );
    assert!(
        frontend
            .render_chat(
                &[json!({"role":"assistant","content":"prefix","prefix":true})],
                &Map::new()
            )
            .is_err()
    );
}

#[test]
fn canonical_tool_order_controls_image_sources_and_placeholder_ids() {
    let frontend = frontend(PromptEncoder::DeepseekV41, json!({}));
    let messages = [
        json!({"role":"user","content":[{"type":"text","text":"Look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,USER"}}]}),
        json!({"role":"assistant","content":"","tool_calls":[{"id":"a","function":{"name":"lookup","arguments":"{}"}},{"id":"b","function":{"name":"lookup","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"b","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,B"}}]}),
        json!({"role":"tool","tool_call_id":"a","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,A"}}]}),
    ];
    let prompt = frontend.render_chat(&messages, &Map::new()).unwrap();
    let mut object_arguments = messages.clone();
    for call in object_arguments[1]["tool_calls"].as_array_mut().unwrap() {
        call["function"]["arguments"] = json!({});
    }
    assert_eq!(
        frontend
            .render_chat(&object_arguments, &Map::new())
            .unwrap()
            .text,
        prompt.text
    );
    for invalid in [json!("[]"), json!("not JSON"), json!([])] {
        object_arguments[1]["tool_calls"][0]["function"]["arguments"] = invalid;
        assert!(
            frontend
                .render_chat(&object_arguments, &Map::new())
                .is_err()
        );
    }
    assert_eq!(
        prompt.image_sources,
        [
            "data:image/png;base64,USER",
            "data:image/png;base64,A",
            "data:image/png;base64,B"
        ]
    );
    assert_eq!(
        prompt
            .token_ids
            .iter()
            .filter(|&&id| id == frontend.image_placeholder_token_id().unwrap())
            .count(),
        3
    );
    assert_eq!(frontend.tokenizer().token_to_id("<｜image｜>"), None);
    let options = json!({"reasoning_effort":80}).as_object().unwrap().clone();
    let scored = frontend.render_chat(&messages, &options).unwrap();
    assert!(scored.text.contains("Reasoning Effort: 80 (range 1-100"));
    assert_eq!(scored.image_sources, prompt.image_sources);
    assert_eq!(scored.token_ids.iter().filter(|&&id|
        id == frontend.image_placeholder_token_id().unwrap()).count(), 3);
    for count in [1, messages.len()] {
        let prefix = native_prefix::without_generation(&frontend, &messages[..count], &options);
        assert!(scored.token_ids.starts_with(&prefix.token_ids));
        assert!(scored.message_boundaries.contains(&prefix.token_ids.len()));
    }
    for messages in [
        vec![json!({"role":"user","content":"<｜image｜>"})],
        vec![
            json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}),
        ],
    ] {
        assert!(frontend.render_chat(&messages, &Map::new()).is_err());
    }
}

#[test]
fn legacy_settings_keep_jinja() {
    let settings: FrontendSettings =
        serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[]})).unwrap();
    assert_eq!(settings.prompt_encoder, PromptEncoder::Jinja);
    assert_eq!(settings.image_placeholder_token_id, None);
    assert!(
        serde_json::from_value::<FrontendSettings>(
            json!({"tokenizer":{},"eos_token_ids":[],"prompt_encoder":3})
        )
        .is_err()
    );
}

#[test]
fn explicit_primary_thinking_option_overrides_configured_alias() {
    let frontend = frontend(PromptEncoder::DeepseekV41, json!({"enable_thinking":false}));
    let prompt = frontend
        .render_chat(
            &[json!({"role":"user","content":"Hello"})],
            json!({"thinking_mode":"thinking"}).as_object().unwrap(),
        )
        .unwrap();
    assert!(prompt.text.ends_with("<think>"));
}

#[test]
fn legacy_templates_receive_image_blocks_and_return_original_sources() {
    let frontend = frontend(PromptEncoder::Jinja, json!({}));
    let prompt = frontend.render_chat(&[json!({"role":"user","content":[{"type":"text","text":"Look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,A"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,B"}}]})], &Map::new()).unwrap();
    assert_eq!(
        prompt.image_sources,
        ["data:image/png;base64,A", "data:image/png;base64,B"]
    );
    let rendered: Value = serde_json::from_str(&prompt.text).unwrap();
    assert_eq!(rendered[0]["content"][1], json!({"type":"image"}));
    assert_eq!(rendered[0]["content"][2], json!({"type":"image"}));
}

#[test]
#[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to a DeepSeek tokenizer"]
fn checkpoint_tokenizer_preserves_canonical_image_order_and_prefixes() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_DEEPSEEK_TEST_SNAPSHOT").unwrap());
    for encoder in [PromptEncoder::DeepseekV4, PromptEncoder::DeepseekV41] {
        let path = root.join("tokenizer.json");
        let bytes = std::fs::read(&path).unwrap();
        let tokenizer = Tokenizer::from_bytes(&bytes).unwrap();
        let placeholder = tokenizer.token_to_id("<｜deepseek_image｜>").unwrap();
        let settings:FrontendSettings=serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[],"prompt_encoder":i32::from(encoder),"image_placeholder_token_id":placeholder})).unwrap();
        let frontend = Frontend::load_with_settings(
            FrontendAssets {
                tokenizer: Asset {
                    path,
                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                    range: None,
                },
                tokens: TokenizerSettings::default(),
                chat_template: None,
            },
            Map::new(),
            &settings,
        )
        .unwrap();
        assert_eq!(frontend.encode("<｜User｜>", false).unwrap().len(), 1);
        assert_ne!(
            frontend.encode("<｜User｜>", false).unwrap(),
            frontend.encode("<｜Assistant｜>", false).unwrap()
        );
        let systems = [
            json!({"role":"system","content":""}),
            json!({"role":"system","content":"Policy"}),
        ];
        for thinking in [false, true] {
            for tools in [false, true] {
                for json_output in [false, true] {
                    let mut options=json!({"enable_thinking":thinking,"response_format":if json_output {"json_object"} else {"text"}}).as_object().unwrap().clone();
                    if tools {
                        options.insert("tools".into(),json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]));
                    }
                    let mut messages = systems.to_vec();
                    messages.push(json!({"role":"user","content":"Hello"}));
                    let full = frontend.render_chat(&messages, &options).unwrap();
                    options.insert("add_generation_prompt".into(), false.into());
                    let prefix = native_prefix::without_generation(&frontend, &systems, &options);
                    assert!(full.token_ids.starts_with(&prefix.token_ids));
                    assert!(full.text.starts_with(&prefix.text));
                }
            }
        }
        let prompt=frontend.render_chat(&[json!({"role":"user","content":[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,A"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,B"}}]})],&Map::new()).unwrap();
        assert_eq!(
            prompt.image_sources,
            ["data:image/png;base64,A", "data:image/png;base64,B"]
        );
        assert_eq!(
            prompt
                .token_ids
                .iter()
                .filter(|&&id| id == placeholder)
                .count(),
            2
        );
        assert!(
            frontend
                .render_chat(
                    &[json!({"role":"user","content":"<｜deepseek_image｜>"})],
                    &Map::new()
                )
                .is_err()
        );
    }
}

#[test]
fn unsupported_native_defaults_fail_loading_and_text_only_needs_no_image_binding() {
    for defaults in [
        json!({"drop_thinking":true}),
        json!({"reasoning_effort":101}),
        json!({"unrecognized":true}),
        json!({"add_generation_prompt":false}),
        json!({"continue_final_message":true}),
    ] {
        assert!(configured_frontend(PromptEncoder::DeepseekV41, defaults, 0, None).is_err());
    }
    assert!(configured_frontend(PromptEncoder::DeepseekV41, json!({}), 1, None).is_err());
    let frontend = configured_frontend(PromptEncoder::DeepseekV41, json!({}), 0, None).unwrap();
    assert!(
        frontend
            .render_chat(&[json!({"role":"user","content":"Hello"})], &Map::new())
            .is_ok()
    );
    assert!(frontend.render_chat(&[json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,A"}}]})],&Map::new()).is_err());
}

#[test]
fn legacy_templates_reject_malformed_messages_and_text_blocks() {
    let frontend = frontend(PromptEncoder::Jinja, json!({}));
    for messages in [
        json!([null]),
        json!(["text"]),
        json!([{"role":"user","content":[{}]}]),
        json!([{"role":"user","content":[{"type":12}]}]),
        json!([{"role":"user","content":[{"type":"text"}]}]),
        json!([{"role":"user","content":[{"type":"text","text":12}]}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{}}]}]),
    ] {
        assert!(
            frontend
                .render_chat(messages.as_array().unwrap(), &Map::new())
                .is_err(),
            "{messages}"
        );
    }
    assert!(
        frontend
            .render_chat(
                &[json!({"role":"user","content":[{"type":"other"}]})],
                &Map::new()
            )
            .is_ok()
    );
}

#[test]
fn native_bindings_reject_template_assets_and_unknown_image_tokens() {
    assert!(configured_frontend(PromptEncoder::DeepseekV41, json!({}), 0, Some(70)).is_err());
    let settings: FrontendSettings =
        serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[],"prompt_encoder":2}))
            .unwrap();
    let absent_asset = || Asset {
        path: std::path::PathBuf::from("/unused/frontend-asset"),
        sha256: "0".repeat(64),
        range: None,
    };
    let result = Frontend::load_with_settings(
        FrontendAssets {
            tokenizer: absent_asset(),
            tokens: TokenizerSettings::default(),
            chat_template: Some(absent_asset()),
        },
        Map::new(),
        &settings,
    );
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("does not use a chat template asset")
    );
}
