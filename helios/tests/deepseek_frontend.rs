#![cfg(feature = "server")]
use flare_helios::{frontend, model, scheduler};
#[path = "support/frontend.rs"]
mod support;
use deepseek_recipe_core::{conversation::Conversation, messages::InputMessage};
use deepseek_recipe_encoding::{PromptEncoding, dsv41::DeepseekV41Encoding};
use flare_helios::frontend::{Asset, Frontend, FrontendAssets, FrontendSettings, PromptEncoder};
use serde_json::{Map, json};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to DeepSeek-V4.1-Flash-1017"]
fn numeric_effort_matches_python_reference_bytes_and_checkpoint_token_ids() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_DEEPSEEK_TEST_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| Asset {
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
        range: None,
    };
    let tokens = support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap();
    let frontend = Frontend::load_with_settings(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: tokens.clone(),
            chat_template: None,
        },
        Map::new(),
        &FrontendSettings {
            tokenizer: tokens,
            eos_token_ids: vec![],
            prompt_encoder: PromptEncoder::DeepseekV41,
            image_placeholder_token_id: None,
            template_encoding: Default::default(),
            image: None,
        },
    )
    .unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("support/deepseek41_numeric_effort.json")).unwrap();
    let prompt = frontend
        .render_chat(
            fixture["messages"].as_array().unwrap(),
            fixture["options"].as_object().unwrap(),
        )
        .unwrap();
    assert_eq!(prompt.text, fixture["rendered"].as_str().unwrap());
    assert_eq!(serde_json::to_value(prompt.token_ids).unwrap(), fixture["token_ids"]);
}

#[test]
#[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to the DeepSeek checkpoint"]
fn official_encoder_renders_and_initializes_native_output_state() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_DEEPSEEK_TEST_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| Asset {
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
        range: None,
    };
    let tokens = support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap();
    let settings = FrontendSettings {
        tokenizer: tokens.clone(),
        eos_token_ids: vec![],
        prompt_encoder: PromptEncoder::DeepseekV41,
        image_placeholder_token_id: None,
        template_encoding: Default::default(),
        image: None,
    };
    let frontend = Frontend::load_with_settings(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens,
            chat_template: None,
        },
        Map::new(),
        &settings,
    )
    .unwrap();
    let mut canonical = Conversation {
        messages: vec![InputMessage::User {
            content: "Hello".into(),
            image_sources: vec![],
        }],
        ..Default::default()
    };
    let messages = [json!({"role":"user", "content":"Hello"})];
    let mut options = Map::new();
    options.insert("add_generation_prompt".into(), true.into());
    let prompt = frontend.render_chat(&messages, &options).unwrap();
    assert_eq!(
        prompt.text,
        DeepseekV41Encoding::new()
            .render_conversation(&canonical, &[])
            .prompt
    );
    assert!(prompt.text.contains("Reasoning Effort: 75"));
    assert!(!prompt.token_ids.is_empty());
    let output = flare_helios::model::prepare_output(
        &support::model_config(support::contract("deepseek41")),
        &frontend,
        &prompt,
        &messages,
        &options,
        false,
    )
    .unwrap();
    assert!((output.content.start == flare_helios::output::tagged::OutputStart::Reasoning));
    options.insert("enable_thinking".into(), false.into());
    let prompt = frontend.render_chat(&messages, &options).unwrap();
    canonical.thinking_mode = false;
    assert_eq!(
        prompt.text,
        DeepseekV41Encoding::new()
            .render_conversation(&canonical, &[])
            .prompt
    );
    assert!(!prompt.text.contains("Reasoning Effort:"));
    let output = flare_helios::model::prepare_output(
        &support::model_config(support::contract("deepseek41")),
        &frontend,
        &prompt,
        &messages,
        &options,
        false,
    )
    .unwrap();
    assert!(!(output.content.start == flare_helios::output::tagged::OutputStart::Reasoning));
}
