#![cfg(feature = "server")]

use flare_helios::frontend;
use flare_helios::{model, scheduler};
#[path = "support/frontend.rs"]
mod support;

use flare_helios::{
    frontend::{
        Asset, Frontend, FrontendAssets,
        images::{ImageLimits, ImageProcessor},
    },
    grammar::GrammarCompiler,
    model::{ModelConfig, SamplingParameters},
    scheduler::SchedulerLimits,
    server::{PreparedRequest, prepare_request},
};
use serde_json::{Map, Value, json};
use std::{fs, path::PathBuf};

struct Fixture {
    frontend: Frontend,
    processor: ImageProcessor,
    compiler: GrammarCompiler,
    config: ModelConfig,
}

impl Fixture {
    fn load(defaults: Map<String, Value>) -> Self {
        let snapshot = PathBuf::from(
            std::env::var_os("FLARE_TEST_SNAPSHOT").expect("FLARE_TEST_SNAPSHOT is required"),
        );
        let fixture: Value = serde_json::from_slice(
            &fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../tests/frontend/qwen38_jpeg_exif.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let asset = |name: &str| Asset {
            range: None,
            path: snapshot.join(name),
            sha256: fixture["assets"][name].as_str().unwrap().into(),
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer.json"),
                tokens: support::tokens(&asset("tokenizer_config.json")).unwrap(),
                chat_template: Some(asset("chat_template.jinja")),
            },
            defaults,
        )
        .unwrap();
        let processor = ImageProcessor::load(
            support::image(&asset("preprocessor_config.json"), &asset("config.json")).unwrap(),
            ImageLimits::default(),
        )
        .unwrap();
        let config = ModelConfig {
            max_top_logprobs: 0,
            logprob_token_id_capacity: 0,
            prepared_sequence_tokens: 262144,
            contract: support::contract("qwen38"),
            model_id: "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD".into(),
            identity: [0; 32],
            limits: SchedulerLimits {
                request_slots: 4,
                max_active: 4,
                max_sequence_tokens: 262144,
                prefill_chunk_size: 4096,
                pages: 4096,
                snapshots: 37,
            },
            eos_token_ids: vec![248046, 248044],
            sampling_ranges: Default::default(),
            sampling_defaults: SamplingParameters::default(),
            max_new_tokens: 64,
            speculative_widths: vec![],
            speculative_fixed_width: None,
            speculative_token_capacity: 0,
            decode_weight_bytes: 0,
            history_drafts: false,
            speculative_proposal_source: Default::default(),
        };
        let compiler =
            GrammarCompiler::new(frontend.tokenizer(), 248320, &config.eos_token_ids).unwrap();
        Self {
            frontend,
            processor,
            compiler,
            config,
        }
    }

    fn prepare(&self, messages: Value, mut options: Value) -> PreparedRequest {
        let generation = !options.get("continue_final_message").is_some_and(|v| v != false);
        options.as_object_mut().unwrap().entry("add_generation_prompt").or_insert(generation.into());
        let expected = self
            .frontend
            .render_chat(messages.as_array().unwrap(), options.as_object().unwrap())
            .unwrap();
        let request = json!({"model":self.config.model_id,"messages":messages,"chat_template_kwargs":options});
        let prepared = prepare_request(
            &self.frontend,
            Some(&self.processor),
            &self.compiler,
            &self.config,
            request,
        )
        .unwrap();
        assert_eq!(prepared.rendered_prompt, expected.text);
        assert_eq!(prepared.prompt.tokens, expected.token_ids);
        flare_helios::model::admission::validate_prompt(&self.config, &prepared.prompt).unwrap();
        prepared
    }

    fn prefix(&self, prepared: &PreparedRequest) -> Vec<u32> {
        let boundary = *prepared.prompt.message_boundaries.first().expect("message boundary");
        assert!(boundary > 0 && boundary < prepared.prompt.tokens.len());
        prepared.prompt.tokens[..boundary].to_vec()
    }
}

fn tool(name: &str) -> Value {
    json!({"type":"function","function":{"name":name,"description":"Look up weather.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}})
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn inline_system_messages_render_in_place_without_changing_the_prefix() {
    let snapshot = PathBuf::from(std::env::var_os("FLARE_TEST_SNAPSHOT").unwrap());
    let fixture: Value = serde_json::from_slice(
        &fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/frontend/qwen38.json"))
            .unwrap(),
    )
    .unwrap();
    let asset = |name: &str| Asset {
        range: None,
        path: snapshot.join(name),
        sha256: fixture["assets"][name].as_str().unwrap().into(),
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: support::tokens(&asset("tokenizer_config.json")).unwrap(),
            chat_template: Some(asset("chat_template.jinja")),
        },
        Map::new(),
    )
    .unwrap();
    let prefix = json!([
        {"role":"system","content":"Be concise."},
        {"role":"user","content":"Hello"}
    ]);
    let options = json!({"enable_thinking":false,"add_generation_prompt":true});
    let mut messages = prefix.as_array().unwrap().clone();
    messages.push(json!({"role":"system","content":[
        {"type":"text","text":"Environment"},
        {"type":"text","text":"Use the repository."}
    ]}));
    frontend
        .normalize_inline_system_messages(&mut messages)
        .unwrap();
    let rendered = frontend
        .render_chat(&messages, options.as_object().unwrap())
        .unwrap();
    messages[2] =
        json!({"role":"user","content":"<system>Environment\n\nUse the repository.</system>"});
    let expected = frontend
        .render_chat(&messages, options.as_object().unwrap())
        .unwrap();
    assert_eq!(rendered.text, expected.text);
    assert_eq!(rendered.token_ids, expected.token_ids);
    let mut prefix_options = options.as_object().unwrap().clone();
    prefix_options.insert("add_generation_prompt".into(), false.into());
    let prefix_text = frontend
        .render_chat(prefix.as_array().unwrap(), &prefix_options)
        .unwrap();
    assert!(rendered.text.starts_with(&prefix_text.text));
    assert!(rendered.token_ids.starts_with(&prefix_text.token_ids));
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn omitted_output_limit_fills_context_after_chat_tokenization() {
    let mut fixture = Fixture::load(Map::new());
    fixture.config.max_new_tokens = fixture.config.limits.max_sequence_tokens as u32;
    let prepared = fixture.prepare(
        json!([{"role":"user","content":"Write a long explanation."}]),
        json!({"enable_thinking":false,"add_generation_prompt":true}),
    );
    assert_eq!(
        prepared.prompt.tokens.len() + prepared.prompt.max_new_tokens as usize,
        fixture.config.limits.max_sequence_tokens
    );
    assert!(prepared.prompt.max_new_tokens > 256);
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn exact_short_instruction_boundary_survives_changed_user_content() {
    let fixture = Fixture::load(Map::new());
    let options = json!({"enable_thinking":false,"add_generation_prompt":true});
    let messages =
        |text| json!([{"role":"system","content":"Be concise."},{"role":"user","content":text}]);
    let a = fixture.prepare(messages("What is two plus two?"), options.clone());
    let b = fixture.prepare(messages("Name a city in France."), options.clone());
    let prefix = fixture.prefix(&a);
    assert_eq!(prefix, fixture.prefix(&b));
    assert!(prefix.len() < 64);
    assert_ne!(prefix.len() % 64, 0);
    let decoded = fixture.frontend.decode(&prefix, false).unwrap();
    assert!(decoded.starts_with("<|im_start|>system\nBe concise.<|im_end|>\n"));
    assert!(!decoded.contains("two plus two"));
    assert!(!decoded.contains("<|im_start|>user"));
    let expected = fixture.frontend.encode("<|im_start|>system\nBe concise.<|im_end|>\n", false).unwrap();
    assert_eq!(prefix, expected);
    let mut malformed = a;
    for boundary in [0, malformed.prompt.tokens.len(), usize::MAX] {
        malformed.prompt.message_boundaries = vec![boundary];
        assert!(
            flare_helios::model::admission::validate_prompt(&fixture.config, &malformed.prompt)
                .is_err()
        );
    }
    println!("exact short boundary: {} tokens", prefix.len());
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn tool_instructions_are_rendered_without_explicit_system_and_disable_with_none() {
    let fixture = Fixture::load(Map::new());
    let messages = json!([{"role":"user","content":"Weather in Oslo?"}]);
    let options =
        |name| json!({"enable_thinking":false,"tools":[tool(name)],"add_generation_prompt":true});
    let a = fixture.prepare(messages.clone(), options("weather"));
    let b = fixture.prepare(
        json!([{"role":"user","content":"Weather in Paris?"}]),
        options("weather"),
    );
    let c = fixture.prepare(messages.clone(), options("other_weather"));
    assert_eq!(fixture.prefix(&a), fixture.prefix(&b));
    assert_ne!(fixture.prefix(&a), fixture.prefix(&c));
    for prepared in [&a, &b, &c] {
        assert_eq!(prepared.prompt.message_boundaries.len(), 2);
        let prefix = fixture.frontend.decode(&fixture.prefix(prepared), false).unwrap();
        assert!(!prefix.contains("Weather in"));
        assert!(prefix.ends_with("<|im_end|>\n"));
        let user_end = prepared.prompt.message_boundaries[1];
        let user_prefix = fixture.frontend.decode(&prepared.prompt.tokens[..user_end], false).unwrap();
        assert!(user_prefix.contains("Weather in"));
        assert!(user_prefix.ends_with("<|im_end|>\n"));
    }
    assert!(
        fixture
            .frontend
            .decode(&fixture.prefix(&a), false)
            .unwrap()
            .contains("\"name\": \"weather\"")
    );
    let request = json!({"model":fixture.config.model_id,"messages":messages,"tools":[tool("weather")],"tool_choice":"none","enable_thinking":false});
    let disabled = prepare_request(
        &fixture.frontend,
        Some(&fixture.processor),
        &fixture.compiler,
        &fixture.config,
        request,
    )
    .unwrap();
    assert!(disabled.prompt.message_boundaries.is_empty());
    assert!(!disabled.rendered_prompt.contains("# Tools"));
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn reasoning_defaults_and_explicit_overrides_follow_actual_system_rendering() {
    let messages = json!([{"role":"user","content":"Compute two plus two."}]);
    let default = Fixture::load(Map::new());
    let a = default.prepare(messages.clone(), json!({"add_generation_prompt":true}));
    let prefix = default.frontend.decode(&default.prefix(&a), false).unwrap();
    assert!(prefix.contains("Reasoning effort is set to xhigh."));
    let disabled = Fixture::load(
        json!({"enable_thinking":false})
            .as_object()
            .unwrap()
            .clone(),
    );
    assert!(!disabled.prepare(messages.clone(), json!({})).prompt.message_boundaries.is_empty());
    let overridden = disabled.prepare(
        messages.clone(),
        json!({"enable_thinking":true,"reasoning_effort":"low"}),
    );
    assert!(
        disabled
            .frontend
            .decode(&disabled.prefix(&overridden), false)
            .unwrap()
            .contains("Reasoning effort is set to low.")
    );
    assert!(!default.prepare(messages,
        json!({"enable_thinking":true,"reasoning_effort":"medium"})).prompt.message_boundaries.is_empty());
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn complete_user_and_empty_system_messages_have_checkpoints() {
    let fixture = Fixture::load(Map::new());
    for messages in [
        json!([{"role":"user","content":"Hello"}]),
        json!([{"role":"system","content":" "},{"role":"user","content":"Hello"}]),
    ] {
        assert!(!fixture.prepare(messages, json!({"enable_thinking":false}))
            .prompt.message_boundaries.is_empty());
    }
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn completed_prompt_and_invalid_messages_do_not_produce_reusable_final_cuts() {
    let fixture = Fixture::load(Map::new());
    let messages = json!([{"role":"system","content":"Be concise."},{"role":"user","content":"Explain the sky."}]);
    let rendered = fixture.frontend.render_chat(messages.as_array().unwrap(),
        json!({"enable_thinking":false,"add_generation_prompt":false}).as_object().unwrap()).unwrap();
    assert!(rendered.message_boundaries.iter().all(|&end| end > 0 && end < rendered.token_ids.len()));
    let malformed = json!([{"role":"system","content":{"not":"valid"}},{"role":"user","content":"Hello"}]);
    assert!(fixture.frontend.render_chat(malformed.as_array().unwrap(), &Map::new()).is_err());
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT pointing to the pinned Qwen checkpoint"]
fn tool_and_system_prefix_survives_assistant_continuation() {
    let fixture = Fixture::load(Map::new());
    let a = fixture.prepare(
        json!([{"role":"system","content":"Be concise."},{"role":"user","content":"Weather?"}]),
        json!({"tools":[tool("weather")],"enable_thinking":false}),
    );
    let b = fixture.prepare(json!([{"role":"system","content":"Be concise."},{"role":"user","content":"Weather?"},{"role":"assistant","content":"It is"}]), json!({"tools":[tool("weather")],"enable_thinking":false,"continue_final_message":true}));
    assert_eq!(fixture.prefix(&a), fixture.prefix(&b));
    assert!(b.rendered_prompt.ends_with("It is"));
}
