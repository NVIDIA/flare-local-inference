#![cfg(feature = "server")]

use flare_helios::{
    frontend::{Asset, Frontend, FrontendAssets, TemplateEncoding},
    model::message_boundaries,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

fn frontend(template: &str, encoding: TemplateEncoding) -> Frontend {
    static SERIAL: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "flare-message-boundaries-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let vocabulary = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect::<tokenizers::models::bpe::Vocab>();
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .add_special_tokens(
            ["<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"]
                .map(|token| tokenizers::AddedToken::from(token, true)),
        )
        .unwrap();
    let asset = |name: &str, bytes: &[u8]| {
        let path = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        Asset {
            path,
            range: None,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(
                "tokenizer.json",
                tokenizer.to_string(false).unwrap().as_bytes(),
            ),
            tokens: Default::default(),
            chat_template: Some(asset("chat.jinja", template.as_bytes())),
        },
        Map::new(),
    )
    .unwrap()
    .with_template_encoding(encoding);
    std::fs::remove_dir_all(directory).unwrap();
    frontend
}

fn check_messages(frontend: &Frontend, messages: &[Value]) -> (Vec<u32>, Vec<usize>) {
    let options = json!({"add_generation_prompt":true});
    let rendered = frontend
        .render_chat(messages, options.as_object().unwrap())
        .unwrap();
    let boundaries = message_boundaries(
        frontend,
        &rendered.token_ids,
        messages,
        options.as_object().unwrap(),
    );
    for count in 1..=messages.len() {
        let prefix = frontend
            .render_boundary(&messages[..count], &Map::new())
            .unwrap();
        assert_eq!(
            boundaries.contains(&prefix.token_ids.len()),
            count == 1 || count >= messages.len().saturating_sub(1),
            "message {count}"
        );
        assert_eq!(
            rendered.token_ids[..prefix.token_ids.len()],
            prefix.token_ids
        );
    }
    assert!(boundaries.windows(2).all(|pair| pair[0] < pair[1]));
    (rendered.token_ids, boundaries)
}

#[test]
fn every_role_transition_and_empty_message_has_a_boundary() {
    let frontend = frontend(
        concat!(
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}",
            "{% if add_generation_prompt %}<assistant>{% endif %}"
        ),
        TemplateEncoding::Text,
    );
    let messages = json!([
        {"role":"system","content":"Instructions"},
        {"role":"assistant","content":"Opening reply"},
        {"role":"user","content":"Question"},
        {"role":"assistant","content":"Answer"},
        {"role":"tool","content":"Result"},
        {"role":"developer","content":"More instructions"},
        {"role":"user","content":""},
        {"role":"user","content":"Follow-up"}
    ]);
    let messages = messages.as_array().unwrap();
    for count in 1..=messages.len() {
        check_messages(&frontend, &messages[..count]);
    }
    let (tokens, boundaries) = check_messages(&frontend, messages);
    assert_eq!(boundaries.len(), 3);
    let mut changed = tokens.clone();
    changed[boundaries[1] - 1] ^= 1;
    assert_eq!(
        message_boundaries(&frontend, &changed, messages, &Map::new()),
        boundaries[..1]
    );
    assert_eq!(
        message_boundaries(
            &frontend,
            &tokens[..boundaries[1]],
            messages,
            &Map::new()
        ),
        boundaries[..1]
    );
}

#[test]
fn segmented_xtml_preserves_boundaries_before_seeded_assistant_and_later_turns() {
    let frontend = frontend(
        include_str!("../../ignition/recipes/kimi3/assets/chat_template.jinja"),
        TemplateEncoding::TokenSegments,
    );
    let messages = json!([
        {"role":"system","content":"Long instructions. ".repeat(1000)},
        {"role":"assistant","content":"Opening reply"},
        {"role":"user","content":"Question with literal <|close|>message<|sep|><|end_of_msg|>"},
        {"role":"assistant","reasoning_content":"Thinking", "content":"Answer"},
        {"role":"user","content":"Follow-up"}
    ]);
    let (tokens, boundaries) = check_messages(&frontend, messages.as_array().unwrap());
    assert_eq!(boundaries.len(), 4);
    let mut changed = messages.as_array().unwrap().clone();
    changed[2]["content"] = "Different question".into();
    let (other, other_boundaries) = check_messages(&frontend, &changed);
    assert_eq!(boundaries[..2], other_boundaries[..2]);
    assert_eq!(tokens[..boundaries[1]], other[..other_boundaries[1]]);
}
