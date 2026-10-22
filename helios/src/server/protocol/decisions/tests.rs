use super::*;

fn body(question: Value) -> Value {
    json!({"model":"local","state":{"message":"Charged twice"},"questions":{"q":question}})
}

#[test]
fn systemone_wire_fixtures_preserve_structured_values_and_false_true_order() {
    let input = body(
        json!({"type":"choice","instructions":{"ask":["which?",true]},
        "criteria":{"billing":null,"support":["Other help",{"x":1}]}}),
    );
    let (decoded, questions) = parse_systemone(input).unwrap();
    assert_eq!(decoded.questions.len(), 1);
    assert_eq!(questions[0].instructions, json!({"ask":["which?",true]}));
    assert_eq!(
        questions[0].options[0],
        json!({"value":"billing","description":null})
    );
    let (_, questions) = parse_systemone(body(json!({"type":"noul","criteria":{
        "true":{"meaning":"yes"},"false":["no"]}})))
    .unwrap();
    assert_eq!(
        questions[0].options,
        [
            json!({"value":false,"description":["no"]}),
            json!({"value":true,"description":{"meaning":"yes"}})
        ]
    );
}

#[test]
fn systemone_schema_rejections_have_jev_details() {
    for question in [
        json!(null),
        json!({"type":"box"}),
        json!({"type":"choice"}),
        json!({"type":"choice","criteria":{}}),
        json!({"type":"score","criteria":[null]}),
        json!({"type":"noul","instructions":true}),
        json!({"type":"noul","criteria":{"true":null}}),
        json!({"type":"choice","criteria":{"x":42}}),
        json!({"type":"score","criteria":[true]}),
    ] {
        let (status, error) = parse_systemone(body(question)).unwrap_err();
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error["detail"][0]["loc"], json!(["body", "questions", "q"]));
        assert!(error["detail"][0]["msg"].is_string());
    }
    for key in ["model", "state", "questions"] {
        let mut value = body(json!({"type":"noul"}));
        value.as_object_mut().unwrap().remove(key);
        assert!(parse_systemone(value).is_err());
    }
    for state in [Value::Null, json!(1), json!(false)] {
        let mut value = body(json!({"type":"noul"}));
        value["state"] = state;
        assert!(parse_systemone(value).is_err());
    }
}

#[test]
fn systemone_bounds_include_singletons_and_reject_excess_before_scoring() {
    for count in [1, 10, 128] {
        let criteria: Map<_, _> = (0..count).map(|i| (i.to_string(), Value::Null)).collect();
        assert!(parse_systemone(body(json!({"type":"choice","criteria":criteria}))).is_ok());
    }
    for count in [0, 129] {
        let criteria: Map<_, _> = (0..count).map(|i| (i.to_string(), Value::Null)).collect();
        assert!(parse_systemone(body(json!({"type":"choice","criteria":criteria}))).is_err());
    }
    for count in [0, 1, 10, 11] {
        let result = parse_systemone(body(
            json!({"type":"score","criteria":vec![json!("level");count]}),
        ));
        assert_eq!(result.is_ok(), (1..=10).contains(&count));
    }
    for count in [0, 1, 64, 65] {
        let questions: Map<_, _> = (0..count)
            .map(|i| (i.to_string(), json!({"type":"noul"})))
            .collect();
        assert_eq!(
            parse_systemone(json!({"model":"local","state":"x","questions":questions})).is_ok(),
            (1..=64).contains(&count)
        );
    }
}

#[test]
fn label_softmax_uses_raw_id_keyed_reads_and_preserves_mass() {
    let value = crate::execution::TokenLogprobs {
        token: 99,
        logprob: -0.1,
        top: vec![(7, -1001.0), (9, -1000.0), (99, -0.1)],
    };
    let (p, mass) = probabilities(&[9, 7], &value).unwrap();
    assert!((p[0] - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-15);
    assert_eq!(mass, 0.0);
    let value = crate::execution::TokenLogprobs {
        token: 9,
        logprob: -1.0,
        top: vec![(9, -1.0), (7, -2.0)],
    };
    let (p, mass) = probabilities(&[7, 9], &value).unwrap();
    assert!((mass - ((-1.0f64).exp() + (-2.0f64).exp())).abs() < 1e-15);
    assert!(p[1] > p[0]);
    for invalid in [
        vec![],
        vec![(7, f32::NAN)],
        vec![(7, f32::INFINITY)],
        vec![(7, 0.1)],
        vec![(7, -1.0), (7, -2.0)],
    ] {
        let value = crate::execution::TokenLogprobs {
            top: invalid,
            ..value.clone()
        };
        assert!(probabilities(&[7], &value).is_err());
    }
}

#[test]
fn jev_confidence_uses_first_mode_and_ordered_level_distance() {
    let choice = json!({"type":"choice","criteria":{"first":null,"second":null,"third":null}});
    let result = systemone_answer(&choice, &[0.6, 0.3, 0.1]);
    assert_eq!(result["choice"], "first");
    assert!((result["confidence"].as_f64().unwrap() - 0.4).abs() < 1e-15);
    assert_eq!(
        systemone_answer(&choice, &[1.0 / 3.0; 3])["choice"],
        "first"
    );
    let score = json!({"type":"score","criteria":["low",{"level":"mid"},["high"]]});
    let result = systemone_answer(&score, &[0.0, 0.5, 0.5]);
    assert_eq!(result["score"], 1.5);
    assert_eq!(result["confidence"], 0.25);
    assert_eq!(result["legend"]["1"], json!({"level":"mid"}));
    assert_eq!(
        systemone_answer(&score, &[0.5, 0.0, 0.5])["confidence"],
        0.0
    );
    for question in [
        json!({"type":"choice","criteria":{"only":null}}),
        json!({"type":"score","criteria":["only"]}),
    ] {
        assert_eq!(systemone_answer(&question, &[1.0])["confidence"], 1.0);
    }
    assert_eq!(
        systemone_answer(&json!({"type":"noul"}), &[0.25, 0.75]),
        json!({"type":"noul","noul":0.75})
    );
}

#[test]
fn structured_content_is_deterministic_and_never_substitutes_keys() {
    let assets = json!({"abc":"data:image/png;base64,AA==","abcdef":"data:image/png;base64,BB=="})
        .as_object()
        .unwrap()
        .clone();
    let mut used = HashSet::new();
    let mut parts = Vec::new();
    content(
        &mut parts,
        &json!({"abc":"abcdef","z":["xabcdefabc",true]}),
        &assets,
        &mut used,
        &mut 4096,
    )
    .unwrap();
    assert_eq!(parts[0]["text"], "{\"abc\":");
    assert_eq!(parts[1]["image_url"]["url"], assets["abcdef"]);
    let images: Vec<_> = parts
        .iter()
        .filter_map(|part| part.get("image_url"))
        .collect();
    assert_eq!(images.len(), 3);
    assert_eq!(images[1]["url"], assets["abc"]);
    assert_eq!(used.len(), 2);
    let mut plain = Vec::new();
    content(
        &mut plain,
        &json!({"z":1,"a":"x\ny"}),
        &Map::new(),
        &mut HashSet::new(),
        &mut 4096,
    )
    .unwrap();
    assert_eq!(plain[0]["text"], "{\"a\":\"x\\ny\",\"z\":1}");
}

#[test]
fn prompts_share_complete_state_messages_and_keep_option_names_literal() {
    let assets = json!({"photo":"data:image/png;base64,AA=="})
        .as_object()
        .unwrap()
        .clone();
    let mut used = HashSet::new();
    let one = Question {
        instructions: json!("first"),
        options: vec![json!({"value":"photo","description":null})],
    };
    let two = Question {
        instructions: json!({"second":["q"]}),
        options: vec![json!({"value":"x","description":"y"})],
    };
    let a = messages(
        &json!({"z":1}),
        &one,
        &[" A".into()],
        &assets,
        &mut used,
        4096,
    )
    .unwrap();
    let b = messages(
        &json!({"z":1}),
        &two,
        &[" A".into()],
        &assets,
        &mut used,
        4096,
    )
    .unwrap();
    assert_eq!(&a[..3], &b[..3]);
    assert_eq!(a.len(), 4);
    assert_eq!(a[3]["role"], "user");
    assert!(
        a[3]["content"][0]["text"]
            .as_str()
            .unwrap()
            .ends_with("Answer:")
    );
    assert!(used.is_empty());
    assert!(
        a[3]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("\"value\":\"photo\"")
    );
}

#[test]
fn assets_require_opt_in_unique_parts_valid_formats_and_complete_consumption() {
    let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\n");
    let assets = json!({"ref":{"type":"image","format":"png","data_base64":png}})
        .as_object()
        .unwrap()
        .clone();
    assert!(image_assets(&assets, Map::new(), false, 1024).is_err());
    assert!(image_assets(&assets, Map::new(), true, 4).is_err());
    assert_eq!(
        image_assets(&assets, Map::new(), true, 1024).unwrap()["ref"],
        json!(format!("data:image/png;base64,{png}"))
    );
    let mut multipart = assets.clone();
    multipart["ref"]["part"] = "file".into();
    multipart["ref"]["data_base64"] = "ignored".into();
    let parts = json!({"file":png}).as_object().unwrap().clone();
    assert!(image_assets(&multipart, parts.clone(), true, 1024).is_ok());
    multipart.insert("other".into(), multipart["ref"].clone());
    assert!(image_assets(&multipart, parts, true, 1024).is_err());
    assert!(
        image_assets(
            &Map::new(),
            json!({"unused":png}).as_object().unwrap().clone(),
            true,
            1024
        )
        .is_err()
    );
}

#[tokio::test]
async fn multipart_requires_json_first_unique_image_parts_and_valid_framing() {
    let permit = Arc::new(
        Arc::new(tokio::sync::Semaphore::new(1))
            .acquire_owned()
            .await
            .unwrap(),
    );
    let payload = "--test\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n{\"state\":\"x\"}\r\n--test\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\nraw\r\n--test--\r\n";
    let request = |payload: String| {
        Request::builder()
            .header("content-type", "multipart/form-data; boundary=test")
            .body(axum::body::Body::from(payload))
            .unwrap()
    };
    let (body, parts) = read_body(4096, request(payload.into()), permit.clone())
        .await
        .unwrap();
    assert_eq!(body["state"], "x");
    assert_eq!(parts["image"], "cmF3");
    for invalid in [
        payload.replace("name=\"request\"", "name=\"other\""),
        payload.replace("name=\"image\"", "name=\"request\""),
        payload.replace("image/png", "image/webp"),
        payload.replace("--test--", "--wrong--"),
    ] {
        assert!(
            read_body(4096, request(invalid), permit.clone())
                .await
                .is_err()
        );
    }
    assert_eq!(
        read_body(8, request(payload.into()), permit.clone())
            .await
            .unwrap_err()
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

#[tokio::test]
async fn compressed_multipart_is_rejected_with_a_clear_error() {
    let permit = Arc::new(
        Arc::new(tokio::sync::Semaphore::new(1))
            .acquire_owned()
            .await
            .unwrap(),
    );
    for encoding in ["gzip", "deflate", "br", "zstd", "unknown"] {
        let request = Request::builder()
            .header("content-type", "multipart/form-data; boundary=test")
            .header("content-encoding", encoding)
            .body(axum::body::Body::from("compressed body"))
            .unwrap();
        let response = read_body(4096, request, permit.clone()).await.unwrap_err();
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["error"]["message"],
            "multipart requests require identity Content-Encoding"
        );
    }
}

#[test]
fn contextual_labels_reject_prefix_retokenization_multi_token_and_special_ids() {
    use crate::frontend::{Asset, Frontend, FrontendAssets};
    use sha2::{Digest, Sha256};
    let root = std::env::temp_dir().join(format!("flare-decision-labels-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .map(|c| c.to_string())
        .collect();
    alphabet.sort_unstable();
    alphabet.extend(["ĠA".into(), "ĠB".into()]);
    let vocab: tokenizers::models::bpe::Vocab = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, text)| (text, id as u32))
        .collect();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(
            vocab,
            vec![("Ġ".into(), "A".into()), ("Ġ".into(), "B".into())],
        )
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .add_special_tokens([tokenizers::AddedToken::from("SPECIAL", true)])
        .unwrap();
    let bytes = tokenizer.to_string(false).unwrap().into_bytes();
    let path = root.join("tokenizer.json");
    std::fs::write(&path, &bytes).unwrap();
    let template = b"{% mark %}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
    let template_path = root.join("template.jinja");
    std::fs::write(&template_path, template).unwrap();
    let label = tokenizer.token_to_id("ĠA").unwrap();
    let vocabulary = tokenizer.get_vocab_size(true);
    for encoder in [
        crate::frontend::PromptEncoder::Jinja,
        crate::frontend::PromptEncoder::DeepseekV4,
        crate::frontend::PromptEncoder::DeepseekV41,
    ] {
        let settings = serde_json::from_value(json!({"tokenizer":{},"eos_token_ids":[],
            "prompt_encoder":i32::from(encoder)}))
        .unwrap();
        let frontend = Frontend::load_with_settings(
            FrontendAssets {
                tokenizer: Asset {
                    path: path.clone(),
                    range: None,
                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                },
                tokens: Default::default(),
                chat_template: (encoder == crate::frontend::PromptEncoder::Jinja).then(|| Asset {
                    path: template_path.clone(),
                    range: None,
                    sha256: format!("{:x}", Sha256::digest(template)),
                }),
            },
            json!({"enable_thinking":true}).as_object().unwrap().clone(),
            &settings,
        )
        .unwrap();
        let compiled = messages(
            &json!("State"),
            &Question {
                instructions: json!("Which?"),
                options: vec![json!({"value":true,"description":null})],
            },
            &[" A".into()],
            &Map::new(),
            &mut HashSet::new(),
            4096,
        )
        .unwrap();
        let options = json!({"enable_thinking":false,"add_generation_prompt":true});
        let rendered = frontend
            .render_chat(&compiled, options.as_object().unwrap())
            .unwrap();
        assert!(rendered.text.contains("Answer:"));
        assert!(!rendered.text.ends_with("Answer:"));
        assert_eq!(
            rendered.token_ids,
            frontend.encode(&rendered.text, false).unwrap()
        );
        assert_eq!(
            label_id(
                &frontend,
                &rendered.text,
                &rendered.token_ids,
                &frontend.decode(&rendered.token_ids, false).unwrap(),
                frontend.tokenizer().get_added_vocabulary(),
                " A",
                vocabulary
            )
            .unwrap(),
            Some(label),
            "{encoder:?}"
        );
        let tokens = frontend.encode(":", false).unwrap();
        let decoded = frontend.decode(&tokens, false).unwrap();
        let added_vocabulary = frontend.tokenizer().get_added_vocabulary();
        assert_eq!(
            label_id(
                &frontend,
                ":",
                &tokens,
                &decoded,
                added_vocabulary,
                " A",
                vocabulary
            )
            .unwrap(),
            Some(label)
        );
        assert_eq!(
            label_id(
                &frontend,
                ":",
                &tokens,
                &decoded,
                added_vocabulary,
                " C",
                vocabulary
            )
            .unwrap(),
            None
        );
        assert_eq!(
            label_id(
                &frontend,
                ":",
                &tokens,
                &decoded,
                added_vocabulary,
                "SPECIAL",
                vocabulary
            )
            .unwrap(),
            None
        );
        assert_eq!(
            label_id(
                &frontend,
                ":",
                &tokens,
                &decoded,
                added_vocabulary,
                " A",
                label as usize
            )
            .unwrap(),
            None
        );
        assert_eq!(
            label_id(
                &frontend,
                ": ",
                &frontend.encode(": ", false).unwrap(),
                &frontend
                    .decode(&frontend.encode(": ", false).unwrap(), false)
                    .unwrap(),
                added_vocabulary,
                "A",
                vocabulary
            )
            .unwrap(),
            None
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn repeated_image_references_are_bounded_before_cloning_payloads() {
    let assets = json!({"ref":"data:image/png;base64,AA=="})
        .as_object()
        .unwrap()
        .clone();
    let mut parts = Vec::new();
    assert!(
        content(
            &mut parts,
            &json!("refrefrefref"),
            &assets,
            &mut HashSet::new(),
            &mut 100
        )
        .is_err()
    );
    assert_eq!(
        parts
            .iter()
            .filter(|part| part["type"] == "image_url")
            .count(),
        1
    );
}

#[tokio::test]
async fn compressed_json_preserves_body_limits_and_releases_frontend_permits() {
    use std::io::Write;

    let slots = Arc::new(tokio::sync::Semaphore::new(1));
    for limit in [4096, 128] {
        let permit = Arc::new(slots.clone().acquire_owned().await.unwrap());
        let body = json!({"state":"x".repeat(1024)});
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(body.to_string().as_bytes()).unwrap();
        let request = Request::builder()
            .header("content-encoding", "gzip")
            .body(axum::body::Body::from(encoder.finish().unwrap()))
            .unwrap();
        let result = read_body(limit, request, permit.clone()).await;
        if limit == 4096 {
            let (decoded, parts) = result.unwrap();
            assert_eq!(decoded, body);
            assert!(parts.is_empty());
        } else {
            assert_eq!(result.unwrap_err().status(), StatusCode::PAYLOAD_TOO_LARGE);
        }
        assert_eq!(slots.available_permits(), 0);
        drop(permit);
        assert_eq!(slots.available_permits(), 1);
    }
}
