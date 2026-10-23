use std::fs;
use std::path::PathBuf;

use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use serde_json::{Map, Value};

fn main() -> Result<(), String> {
    run().map_err(|error| error.to_string())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments.len() != 3 {
        return Err("usage: test_frontend ABSOLUTE_SNAPSHOT FIXTURES_JSON".into());
    }
    let snapshot = PathBuf::from(&arguments[1]);
    let fixture: Value =
        serde_json::from_slice(&fs::read(&arguments[2]).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let asset = |name: &str| -> Result<Asset, String> {
        Ok(Asset {
            range: None,
            path: snapshot.join(name),
            sha256: fixture["assets"][name]
                .as_str()
                .ok_or_else(|| format!("missing expected asset fingerprint for {name}"))?
                .into(),
        })
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json")?,
            tokens: flare_helios::frontend::test_support::tokens(&asset("tokenizer_config.json")?)
                .unwrap(),
            chat_template: Some(asset("chat_template.jinja")?),
        },
        Map::new(),
    )?
    .with_template_encoding(serde_json::from_value(
        fixture
            .get("template_encoding")
            .cloned()
            .unwrap_or(Value::from(0)),
    )?);
    let mut counts = Map::new();
    for section in ["cases", "encode_cases", "decode_cases"] {
        let cases = fixture[section]
            .as_array()
            .ok_or_else(|| format!("missing {section}"))?;
        for case in cases {
            let name = case["name"].as_str().unwrap();
            match section {
                "cases" => {
                    let rendered = frontend.render_chat(
                        case["messages"].as_array().unwrap(),
                        case["options"].as_object().unwrap(),
                    );
                    if let Some(expected) = case.get("error") {
                        let expected = expected
                            .as_str()
                            .or_else(|| expected["message"].as_str())
                            .unwrap();
                        let error = match rendered {
                            Err(error) => error,
                            Ok(_) => {
                                return Err((format!(
                                    "{name}: expected error {expected:?}, received rendered prompt"
                                ))
                                .into());
                            }
                        };
                        if error.to_string() != expected {
                            return Err((format!(
                                "{name}: expected error {expected:?}, received {error:?}"
                            ))
                            .into());
                        }
                    } else {
                        let rendered = rendered.map_err(|error| format!("{name}: {error}"))?;
                        let expected = case["rendered"].as_str().unwrap();
                        if rendered.text != expected {
                            return Err((format!(
                                "{name}: rendered bytes differ\nexpected: {expected:?}\nreceived: {:?}",
                                rendered.text
                            )).into());
                        }
                        if serde_json::to_value(&rendered.token_ids).unwrap() != case["token_ids"] {
                            return Err((format!("{name}: token IDs differ")).into());
                        }
                        if let Some(expected) = case.get("message_boundaries") {
                            if serde_json::to_value(&rendered.message_boundaries).unwrap() != *expected {
                                return Err(format!("{name}: message boundaries differ").into());
                            }
                        }
                    }
                }
                "encode_cases" => {
                    let ids = frontend.encode(
                        case["text"].as_str().unwrap(),
                        case["add_special_tokens"].as_bool().unwrap(),
                    )?;
                    if serde_json::to_value(ids).unwrap() != case["token_ids"] {
                        return Err((format!("{name}: token IDs differ")).into());
                    }
                }
                "decode_cases" => {
                    let ids: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
                    let skip = case["skip_special_tokens"].as_bool().unwrap();
                    let expected = case["text"].as_str().unwrap();
                    if frontend.decode(&ids, skip)? != expected {
                        return Err((format!("{name}: decoded text differs")).into());
                    }
                    let mut decoder = frontend.incremental_decoder(skip);
                    let mut text = String::new();
                    for id in ids {
                        if let Some(chunk) = decoder.push(&[id])? {
                            text.push_str(&chunk);
                        }
                    }
                    text.push_str(&decoder.finish()?);
                    if text != expected {
                        return Err((format!(
                            "{name}: incremental text differs: {text:?} != {expected:?}"
                        ))
                        .into());
                    }
                }
                _ => unreachable!(),
            }
        }
        counts.insert(section.into(), Value::from(cases.len()));
    }
    let configured = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json")?,
            tokens: flare_helios::frontend::test_support::tokens(&asset("tokenizer_config.json")?)
                .unwrap(),
            chat_template: Some(asset("chat_template.jinja")?),
        },
        serde_json::json!({"enable_thinking": false, "add_generation_prompt": true})
            .as_object()
            .unwrap()
            .clone(),
    )?
    .with_template_encoding(serde_json::from_value(
        fixture
            .get("template_encoding")
            .cloned()
            .unwrap_or(Value::from(0)),
    )?);
    for (fixture_name, overrides) in [
        ("thinking_disabled", serde_json::json!({})),
        (
            "thinking_enabled",
            serde_json::json!({"enable_thinking": true}),
        ),
        (
            "default",
            serde_json::json!({"enable_thinking": true, "add_generation_prompt": false}),
        ),
    ] {
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == fixture_name)
            .unwrap();
        let prompt = configured.render_chat(
            case["messages"].as_array().unwrap(),
            overrides.as_object().unwrap(),
        )?;
        if prompt.text != case["rendered"].as_str().unwrap()
            || serde_json::to_value(prompt.token_ids).unwrap() != case["token_ids"]
        {
            return Err((format!(
                "configured defaults/explicit overrides diverge for {fixture_name}"
            ))
            .into());
        }
    }
    counts.insert("default_override_cases".into(), Value::from(3));
    println!("{}", Value::Object(counts));
    Ok(())
}
