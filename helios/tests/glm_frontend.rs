#![cfg(feature = "server")]

use flare_helios::frontend;
use flare_helios::{model, scheduler};
#[path = "support/frontend.rs"]
mod support;

use flare_helios::{
    frontend::{Asset, Frontend, FrontendAssets},
    grammar::GrammarCompiler,
    model::{ModelConfig, SamplingParameters},
    output::{Event, Input, Termination},
    scheduler::SchedulerLimits,
    server::prepare_request,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[test]
#[ignore = "requires FLARE_GLM_TEST_SNAPSHOT and FLARE_GLM_FRONTEND_REFERENCE from generate_glm_fixtures.py"]
fn checkpoint_template_tokenization_requests_and_message_boundaries_match_transformers() {
    let snapshot = PathBuf::from(std::env::var_os("FLARE_GLM_TEST_SNAPSHOT").unwrap());
    let asset = |name: &str| {
        let path = snapshot.join(name);
        Asset {
            range: None,
            sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
            path,
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: support::tokens(&asset("tokenizer_config.json")).unwrap(),
            chat_template: Some({
                let path=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ignition/recipes/glm53/assets/chat_template.jinja");
                let contents=fs::read_to_string(&path).unwrap();
                assert_eq!(contents.replace("{%- mark -%}",""),fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap());
                Asset {range:None,sha256:format!("{:x}",Sha256::digest(contents.as_bytes())),path}
            }),
        },
        json!({"add_generation_prompt":true})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let config = ModelConfig {
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 1048576,
        contract: support::contract("glm53"),
        identity: [9; 32],
        model_id: "glm53".into(),
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 1048576,
            prefill_chunk_size: 4096,
            pages: 4096,
            snapshots: 16,
        },
        eos_token_ids: vec![154820, 154827, 154829],
        sampling_ranges: Default::default(),
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 128,
        speculative_widths: vec![],
        speculative_fixed_width: None,
        speculative_token_capacity: 0,
        decode_weight_bytes: 0,
        history_drafts: false,
        speculative_proposal_source: Default::default(),
    };
    let delimiters =
        model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend).unwrap();
    assert_eq!(
        delimiters,
        [
            (154841, "<think>".to_owned()),
            (154842, "</think>".to_owned())
        ]
    );
    let compiler = GrammarCompiler::new(
        frontend.tokenizer(),
        154880,
        &config.eos_token_ids,
        &delimiters,
    )
    .unwrap();
    let reference: Value = serde_json::from_slice(
        &fs::read(std::env::var_os("FLARE_GLM_FRONTEND_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    use flare_helios::frontend::images::{ImageLimits, ImageProcessor, ImageSettings};
    use flare_helios::model::contract::{ImageLayoutKind, ImagePreparationKind, ImageTokenLayout};
    let mut image_config = config.clone();
    image_config.contract.vision = serde_json::from_value(json!({
        "feature_slots": 1, "features_per_request": 8192, "max_images": 4,
        "patches_per_feature": 4, "patch_capacities": [64, 256, 1024, 4096, 16384, 32768]
    })).unwrap();
    image_config.contract.frontend.rejected_tokens = (154832..=154835).chain([154855]).collect();
    image_config.contract.frontend.image_layout = Some(ImageLayoutKind::Glm);
    image_config.contract.frontend.image_preparation = Some(ImagePreparationKind::Glm);
    image_config.contract.frontend.image_tokens = Some(ImageTokenLayout { feature_token: 154854, span: None });
    let processor = ImageProcessor::load(ImageSettings::Glm {}, ImageLimits {
        max_patches_per_image: 32768, max_total_patches: 128000, patches_per_feature: 4,
        ..ImageLimits::default()
    }).unwrap();
    for case in reference["image_cases"].as_array().unwrap() {
        let prepared = prepare_request(&frontend, Some(&processor), &compiler, &image_config, case["body"].clone()).unwrap();
        model::admission::validate_prompt(&image_config, &prepared.prompt).unwrap();
        assert_eq!(serde_json::to_value(&prepared.prompt.tokens).unwrap(), case["token_ids"]);
        let images = &prepared.prompt.conditioned_inputs;
        assert_eq!(serde_json::to_value(images.iter().map(|image| image.geometry.grid_thw).collect::<Vec<_>>()).unwrap(), case["grid_thw"]);
        assert_eq!(prepared.prompt.conditioning.len(), images.len());
        let indices = &prepared.prompt.data.vision_token_indices;
        let features = images.iter().map(|image| image.geometry.tokens).sum::<usize>();
        assert_eq!(indices.iter().copied().filter(|i| *i >= 0).collect::<Vec<_>>(), (0..features as i32).collect::<Vec<_>>());
        for (token, index) in prepared.prompt.tokens.iter().zip(indices) {
            assert_eq!(*token == 154854, *index >= 0);
        }
        for image in images {
            let pixels = image.materialize().unwrap();
            assert_eq!(pixels.pixel_values.len(), image.geometry.patches * 1176);
            assert!(pixels.interpolation_indices.is_empty());
        }
        let repeated = prepare_request(&frontend, Some(&processor), &compiler, &image_config, case["body"].clone()).unwrap();
        assert_eq!(prepared.prompt.conditioning, repeated.prompt.conditioning);
    }
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prepared = prepare_request(&frontend, None, &compiler, &config, case["body"].clone())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        flare_helios::model::admission::validate_prompt(&config, &prepared.prompt).unwrap();
        assert_eq!(
            prepared.rendered_prompt,
            case["rendered"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            serde_json::to_value(&prepared.prompt.tokens).unwrap(),
            case["token_ids"],
            "{name}"
        );
        assert_eq!(serde_json::to_value(&prepared.prompt.message_boundaries).unwrap(),
            case["message_boundaries"], "{name}");
        assert_eq!(
            prepared.output.content.start == flare_helios::output::tagged::OutputStart::Reasoning,
            case["reasoning"].as_bool().unwrap(),
            "{name}"
        );
        assert!(!prepared.output.content.structured, "{name}");
        assert!(prepared.prompt.grammar.is_none(), "{name}");
        let reasoning =
            prepared.output.content.start == flare_helios::output::tagged::OutputStart::Reasoning;
        let mut output = flare_helios::output::OutputPipeline::new(prepared.output);
        let mut events = Vec::new();
        if reasoning {
            events.extend(output.push(Input::Text("analysis")).unwrap());
            events.extend(output.push(Input::Token(154842)).unwrap());
            events.extend(output.push(Input::Text("</think>")).unwrap());
        }
        events.extend(output.push(Input::Text("result")).unwrap());
        events.extend(output.finish(Termination::Eos).unwrap());
        assert_eq!(
            events.last(),
            Some(&Event::Content("result".into())),
            "{name}"
        );
        assert_eq!(
            events.contains(&Event::Reasoning("analysis".into())),
            reasoning,
            "{name}"
        );
        if name == "parallel_tool_results" {
            let text = prepared
                .rendered_prompt
                .split_once("<|assistant|>")
                .unwrap()
                .1
                .split_once("<|observation|>")
                .unwrap()
                .0;
            let tools = case["body"]["chat_template_kwargs"]["tools"]
                .as_array()
                .unwrap()
                .clone();
            let mut output =
                flare_helios::output::OutputPipeline::new(flare_helios::output::OutputConfig {
                    syntax: support::contract("glm53").frontend,
                    control_tokens: vec![],
                    content: flare_helios::output::tagged::OutputConfig {
                        structured: false,
                        start: flare_helios::output::tagged::OutputStart::Content,
                        tools,
                    },
                });
            let mut events = Vec::new();
            for character in text.chars() {
                events.extend(output.push(Input::Text(&character.to_string())).unwrap());
            }
            events.extend(output.finish(Termination::Eos).unwrap());
            let arguments: Vec<Value> = events
                .iter()
                .filter_map(|event| match event {
                    Event::ToolArguments { text, .. } => Some(serde_json::from_str(text).unwrap()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                arguments,
                [
                    json!({"city":"Paris α", "count":2}),
                    json!({"city":"London", "count":1})
                ]
            );
        }
    }
    for request in [
        json!({"model":"glm53", "messages":[{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}]}]}),
        json!({"model":"glm53", "messages":[{"role":"user", "content":"<|begin_of_image|><|image|><|end_of_image|>"}]}),
    ] {
        assert!(prepare_request(&frontend, None, &compiler, &config, request).is_err());
    }
    let structured = prepare_request(&frontend, None, &compiler, &config,
        json!({"model":"glm53", "messages":[{"role":"user", "content":"JSON please"}], "response_format":{"type":"json_object"}})
    ).unwrap();
    assert!(
        structured.output.content.start == flare_helios::output::tagged::OutputStart::Reasoning
    );
    assert!(structured.output.content.structured);
    assert!(structured.prompt.grammar.is_some());
}
