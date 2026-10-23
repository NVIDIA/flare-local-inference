#![cfg(feature = "server")]

use flare_helios::{
    frontend::{Asset, Frontend, FrontendAssets, TemplateEncoding},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};

fn frontend(template: &str, encoding: TemplateEncoding) -> Frontend {
    frontend_with_merges(template,encoding,vec![])
}

fn frontend_with_merges(template: &str, encoding: TemplateEncoding, merges: Vec<(String,String)>) -> Frontend {
    static SERIAL: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "flare-message-boundaries-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&directory).unwrap();
    let mut vocabulary = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect::<tokenizers::models::bpe::Vocab>();
    for (a,b) in &merges {vocabulary.insert(format!("{a}{b}"),vocabulary.len() as u32);}
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocabulary, merges)
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.with_post_processor(Some(tokenizers::processors::byte_level::ByteLevel::new(false,true,false)));
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

#[test]
fn bpe_crossings_and_unicode_offsets_use_the_complete_encoding() {
    for (source,merges,expected) in [
        ("a{% mark %}b{% mark %}c",vec![("a".into(),"b".into())],vec![1]),
        ("a {% mark %}b",vec![("Ġ".into(),"b".into())],vec![]),
        ("é{% mark %}x",vec![],vec![2]),
    ] {
        let frontend=frontend_with_merges(source,TemplateEncoding::Text,merges);
        let rendered=frontend.render_chat(&[json!({"role":"user","content":"unused"})],&Map::new()).unwrap();
        assert_eq!(rendered.message_boundaries,expected, "{source}");
        assert_eq!(rendered.token_ids,frontend.encode(&rendered.text,false).unwrap());
    }
}

#[test]
fn continuation_clips_marks_and_segments_before_mapping() {
    let frontend=frontend("{% for m in messages %}{{m.content}}{% mark %}|{% endfor %}",TemplateEncoding::Text);
    let rendered=frontend.render_chat(&[json!({"role":"user","content":"A"}),json!({"role":"assistant","content":"café  "})],
        json!({"continue_final_message":true}).as_object().unwrap()).unwrap();
    assert_eq!(rendered.text,"A|café  ");
    assert_eq!(rendered.message_boundaries,vec![1]);
    assert_eq!(rendered.token_ids,frontend.encode(&rendered.text,false).unwrap());
}

fn check_messages(frontend: &Frontend, messages: &[Value]) -> (Vec<u32>, Vec<usize>) {
    let options = json!({"add_generation_prompt":true});
    let rendered = frontend
        .render_chat(messages, options.as_object().unwrap())
        .unwrap();
    let boundaries = rendered.message_boundaries;
    for count in 1..=messages.len() {
        let prefix = frontend
            .render_chat(&messages[..count], &Map::new())
            .unwrap();
        assert!(
            boundaries.contains(&prefix.token_ids.len()),
            "message {count} has no checkpoint"
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
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% mark %}{% endfor %}",
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
    let (tokens, boundaries) = check_messages(&frontend, messages.as_array().unwrap());
    assert_eq!(boundaries.len(), 8);
    let mut changed = messages.as_array().unwrap().clone();
    changed[2]["content"] = "Different question".into();
    let (other, other_boundaries) = check_messages(&frontend, &changed);
    assert_eq!(&boundaries[..2], &other_boundaries[..2]);
    assert_eq!(&tokens[..boundaries[1]], &other[..other_boundaries[1]]);
    let complete = frontend.render_chat(messages.as_array().unwrap(), &Map::new()).unwrap();
    assert!(complete.message_boundaries.iter().all(|&end| end < complete.token_ids.len()));

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
    assert_eq!(boundaries.len(), 6);
    let mut changed = messages.as_array().unwrap().clone();
    changed[2]["content"] = "Different question".into();
    let (other, other_boundaries) = check_messages(&frontend, &changed);
    assert_eq!(boundaries[..3], other_boundaries[..3]);
    assert_eq!(tokens[..boundaries[2]], other[..other_boundaries[2]]);
}
