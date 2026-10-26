use base64::Engine;
use flare_helios::{
    frontend::{
        Asset, Frontend, FrontendAssets,
        images::{ImageLimits, ImageProcessor},
    },
    model::{ModelConfig, SamplingParameters},
    scheduler::SchedulerLimits,
    server::prepare_request,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

fn main() -> Result<(), String> {
    run().map_err(|error| error.to_string())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err("usage: test_requests ABSOLUTE_SNAPSHOT TEXT_FIXTURES IMAGE_FIXTURES".into());
    }
    let snapshot = PathBuf::from(&args[1]);
    let text: Value = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let image_path = PathBuf::from(&args[3]);
    let image: Value = serde_json::from_slice(&fs::read(&image_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let asset = |name: &str| Asset {
        range: None,
        path: snapshot.join(name),
        sha256: image["assets"][name].as_str().unwrap().into(),
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: flare_helios::frontend::test_support::tokens(&asset("tokenizer_config.json"))
                .unwrap(),
            chat_template: Some(asset("chat_template.jinja")),
        },
        Map::new(),
    )?;
    let processor = ImageProcessor::load(
        flare_helios::frontend::test_support::image(
            &asset("preprocessor_config.json"),
            &asset("config.json"),
        )
        .unwrap(),
        ImageLimits::default(),
    )?;
    let config = ModelConfig {
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 262144,
        contract: flare_helios::frontend::test_support::contract("qwen38"),
        model_id: text["model_id"].as_str().unwrap().into(),
        identity: [0; 32],
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 262144,
            prefill_chunk_size: 512,
            pages: 4096,
            snapshots: 37,
        },
        eos_token_ids: vec![248046, 248044],
        sampling_ranges: Default::default(),
        sampling_defaults: SamplingParameters {
            temperature: 0.25,
            top_k: 50,
            ..Default::default()
        },
        max_new_tokens: 17,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        speculative_token_capacity: 0,
        decode_weight_bytes: 0,
        history_drafts: true,
        speculative_proposal_source: Default::default(),
    };
    let mut text_count = 0;
    let mut tool_string_count = 0;
    let compiler = flare_helios::grammar::GrammarCompiler::new(
        frontend.tokenizer(),
        248320,
        &config.eos_token_ids,
        &flare_helios::model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend)?,
    )?;
    for case in text["cases"].as_array().unwrap() {
        if case["messages"].to_string().contains("\"image") {
            continue;
        }
        let body = json!({"model":config.model_id,"messages":case["messages"],"chat_template_kwargs":case["options"]});
        let result = prepare_request(
            &frontend,
            Some(&processor),
            &compiler,
            &config,
            body.clone(),
        );
        if case["name"] == "invalid_role" && case["messages"][1]["role"] == "developer" {
            let prepared = result?;
            let expected = frontend.render_chat(
                &[
                    json!({"role":"system","content":"Developer instructions:\nInvalid role."}),
                    case["messages"][0].clone(),
                ],
                case["options"].as_object().unwrap(),
            )?;
            assert_eq!(prepared.prompt.tokens, expected.token_ids);
            text_count += 1;
            continue;
        }
        if let Some(expected) = case.get("error") {
            let expected = expected
                .as_str()
                .unwrap_or_else(|| expected["message"].as_str().unwrap());
            match result {
                Err(error)
                    if matches!(&error, flare_helios::Error::Admission(_))
                        && error.to_string() == expected => {}
                Err(error) => {
                    return Err((format!(
                        "{}: expected {expected:?}, received {error:?}",
                        case["name"]
                    ))
                    .into());
                }
                Ok(_) => {
                    return Err((format!("{}: expected template rejection", case["name"])).into());
                }
            }
        } else {
            let prepared = result.map_err(|error| format!("{}: {error}", case["name"]))?;
            if serde_json::to_value(&prepared.prompt.tokens).unwrap() != case["token_ids"] {
                return Err((format!("{}: request token IDs differ", case["name"])).into());
            }
            if prepared.prompt.sampling.temperature != 0.25
                || prepared.prompt.sampling.top_k != 50
                || prepared.prompt.max_new_tokens != 17
            {
                return Err("request sampling did not clamp to model defaults or output limit changed".into());
            }
            let mut wire = body;
            let mut changed = false;
            for message in wire["messages"].as_array_mut().unwrap() {
                if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    for call in calls {
                        let arguments = &mut call["function"]["arguments"];
                        if arguments.is_object() {
                            *arguments = Value::String(arguments.to_string());
                            changed = true;
                        }
                    }
                }
            }
            if changed {
                let roundtrip =
                    prepare_request(&frontend, Some(&processor), &compiler, &config, wire)?;
                if roundtrip.prompt.tokens != prepared.prompt.tokens {
                    return Err((format!(
                        "{}: JSON-string tool arguments changed rendered tokens",
                        case["name"]
                    ))
                    .into());
                }
                tool_string_count += 1;
            }
        }
        text_count += 1;
    }
    let mut string_stop_count = 0;
    for stop in [
        json!(null),
        json!("\nEND_EDIT"),
        json!(["STOP", "世界"]),
        json!([]),
    ] {
        let body = json!({"model":config.model_id,"messages":[{"role":"user","content":"Reply with text."}],"stop":stop});
        let mut plain = body.clone();
        plain.as_object_mut().unwrap().remove("stop");
        let expected = prepare_request(&frontend, Some(&processor), &compiler, &config, plain)?;
        let prepared = prepare_request(&frontend, Some(&processor), &compiler, &config, body)?;
        if prepared.prompt.tokens != expected.prompt.tokens
            || prepared.prompt.max_new_tokens != expected.prompt.max_new_tokens
        {
            return Err("string stop changed prompt IDs or output reservation".into());
        }
        let expected_stops = match stop {
            Value::Null => vec![],
            Value::String(value) => vec![value],
            Value::Array(values) => values
                .into_iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect(),
            _ => unreachable!(),
        };
        if prepared.prompt.stop_strings != expected_stops {
            return Err("string stop normalization differs".into());
        }
        string_stop_count += 1;
    }
    let mut image_count = 0;
    for case in image["cases"].as_array().unwrap() {
        if !case["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|input| input.get("png").is_some())
        {
            continue;
        }
        let mut messages = case["messages"].clone();
        let mut index = 0;
        for message in messages.as_array_mut().unwrap() {
            if let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) {
                for part in parts {
                    if part["type"] == "image" {
                        let bytes = fs::read(
                            image_path
                                .parent()
                                .unwrap()
                                .join(case["inputs"][index]["png"].as_str().unwrap()),
                        )
                        .map_err(|e| e.to_string())?;
                        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
                        *part = json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{encoded}")}});
                        index += 1;
                    }
                }
            }
        }
        let prepared = prepare_request(
            &frontend,
            Some(&processor),
            &compiler,
            &config,
            json!({"model":config.model_id,"messages":messages,"add_generation_prompt":true}),
        )?;
        if serde_json::to_value(&prepared.prompt.tokens).unwrap() != case["token_ids"] {
            return Err((format!("{}: image request token IDs differ", case["name"])).into());
        }
        let mut processed = flare_helios::frontend::images::ProcessedImages::default();
        for image in &prepared.prompt.conditioned_inputs { processed.append(image.materialize()?); }
        let bytes: Vec<_> = processed
            .pixel_values
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes)
            .collect();
        let hash = format!("{:x}", Sha256::digest(bytes));
        let expected = case["pixel_values_sha256"]
            .as_str()
            .or_else(|| case["hashes"]["pixel_values"].as_str())
            .ok_or("image fixture lacks pixel hash")?;
        if hash != expected {
            return Err((format!("{}: image request pixels differ", case["name"])).into());
        }
        image_count += 1;
    }
    for format in [
        json!({"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object","properties":{"code":{"enum":["A","B"]}},"required":["code"],"additionalProperties":false}}}),
        json!({"type":"json_object"}),
    ] {
        let body = json!({"model":config.model_id,"messages":[{"role":"user","content":"Return JSON."}],"response_format":format,"add_generation_prompt":true,"enable_thinking":false});
        let mut prepared = prepare_request(
            &frontend,
            Some(&processor),
            &compiler,
            &config,
            body.clone(),
        )?;
        if prepared.output.content.start == flare_helios::output::tagged::OutputStart::Reasoning
            || !prepared.output.content.structured
        {
            return Err("structured request entered reasoning mode".into());
        }
        let grammar = prepared
            .prompt
            .grammar
            .as_mut()
            .ok_or("structured request lacks grammar")?;
        for token in frontend.encode("{\"code\":\"A\"}", false)? {
            grammar.advance(token)?;
        }
        if !grammar.is_accepting() {
            return Err("structured request grammar did not accept canonical JSON".into());
        }
        let mut conflict = body.clone();
        conflict["stop_token_ids"] = json!([90]);
        if prepare_request(&frontend, Some(&processor), &compiler, &config, conflict).is_ok() {
            return Err("structured request accepted custom token stop".into());
        }
        let mut conflict = body;
        conflict["continue_final_message"] = json!(true);
        if prepare_request(&frontend, Some(&processor), &compiler, &config, conflict).is_ok() {
            return Err("structured request accepted assistant continuation".into());
        }
    }
    for continuation in [json!(true), json!("content")] {
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer.json"),
                tokens: flare_helios::frontend::test_support::tokens(&asset("tokenizer_config.json")).unwrap(),
                chat_template: Some(asset("chat_template.jinja")),
            },
            json!({"continue_final_message":continuation,"add_generation_prompt":false,"enable_thinking":false})
                .as_object()
                .unwrap()
                .clone(),
        )?;
        let body = json!({"model":config.model_id,"messages":[{"role":"user","content":"Return JSON."},{"role":"assistant","content":"{"}],"response_format":{"type":"json_object"}});
        match prepare_request(
            &frontend,
            Some(&processor),
            &compiler,
            &config,
            body.clone(),
        ) {
            Err(error)
                if error.to_string()
                    == "structured output cannot continue an existing assistant message" => {}
            Err(error) => {
                return Err((format!(
                    "configured continuation returned unexpected error: {error}"
                ))
                .into());
            }
            Ok(_) => {
                return Err("structured request accepted configured assistant continuation".into());
            }
        }
        for overrides in [
            json!({"continue_final_message":false}),
            json!({"continue_final_message":""}),
            json!({"continue_final_message":null}),
            json!({"chat_template_kwargs":{"continue_final_message":false}}),
        ] {
            let mut request = body.clone();
            request
                .as_object_mut()
                .unwrap()
                .extend(overrides.as_object().unwrap().clone());
            let mut prepared =
                prepare_request(&frontend, Some(&processor), &compiler, &config, request)?;
            let grammar = prepared
                .prompt
                .grammar
                .as_mut()
                .ok_or("structured request lacks grammar")?;
            for token in frontend.encode("{}", false)? {
                grammar.advance(token)?;
            }
            if !grammar.is_accepting() {
                return Err(
                    "explicit continuation override did not accept a complete JSON object".into(),
                );
            }
        }
    }
    println!(
        "{{\"string_stop_request_cases\":{string_stop_count},\"text_request_cases\":{text_count},\"tool_argument_roundtrips\":{tool_string_count},\"image_request_cases\":{image_count},\"structured_request_cases\":16}}"
    );
    Ok(())
}
