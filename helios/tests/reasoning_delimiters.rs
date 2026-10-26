#![cfg(feature = "server")]
//! Checkpoint tokenizers bound to the reasoning delimiters their recipes declare.

use flare_helios::{frontend, model, scheduler};
#[path = "support/frontend.rs"]
mod support;

use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use flare_helios::model::contract::{
    FrontendContract, ReasoningParser, ReasoningParserKind, ToolParserKind,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn delimiters(variable: &str, syntax: &FrontendContract) -> Vec<(u32, String)> {
    let root = PathBuf::from(std::env::var_os(variable).unwrap());
    let asset = |path: PathBuf| Asset {
        range: None,
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
    };
    let template = std::env::temp_dir().join(format!(
        "flare-reasoning-delimiters-{}-{variable}.jinja",
        std::process::id()
    ));
    std::fs::write(&template, "{% mark %}{{ messages[0].content }}").unwrap();
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
            chat_template: Some(asset(template.clone())),
        },
        Default::default(),
    )
    .unwrap();
    std::fs::remove_file(template).unwrap();
    model::reasoning_delimiter_tokens(&frontend, syntax).unwrap()
}

fn pairs(tokens: &[(u32, &str)]) -> Vec<(u32, String)> {
    tokens
        .iter()
        .map(|&(id, marker)| (id, marker.to_owned()))
        .collect()
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT with the Qwen3.8 tokenizer"]
fn qwen38_think_tags_are_single_tokens() {
    assert_eq!(
        delimiters("FLARE_TEST_SNAPSHOT", &support::contract("qwen38").frontend),
        pairs(&[(248068, "<think>"), (248069, "</think>")])
    );
}

#[test]
#[ignore = "requires FLARE_GLM_TEST_SNAPSHOT with the GLM-5.3 tokenizer"]
fn glm53_think_tags_are_single_tokens() {
    assert_eq!(
        delimiters(
            "FLARE_GLM_TEST_SNAPSHOT",
            &support::contract("glm53").frontend
        ),
        pairs(&[(154841, "<think>"), (154842, "</think>")])
    );
}

#[test]
#[ignore = "requires FLARE_MIMO_TEST_SNAPSHOT with the MiMo-V2 tokenizer"]
fn mimo_v2_think_tags_are_single_tokens() {
    assert_eq!(
        delimiters(
            "FLARE_MIMO_TEST_SNAPSHOT",
            &support::contract("mimo_v2").frontend
        ),
        pairs(&[(151667, "<think>"), (151668, "</think>")])
    );
}

#[test]
#[ignore = "requires FLARE_NEMOTRON_TEST_SNAPSHOT with the Nemotron-3.5 tokenizer"]
fn nemotron_h_think_tags_are_single_tokens() {
    // nemotron_h declares the same Qwen-coder tools and think tags as mimo_v2.
    assert_eq!(
        delimiters(
            "FLARE_NEMOTRON_TEST_SNAPSHOT",
            &support::contract("mimo_v2").frontend
        ),
        pairs(&[(12, "<think>"), (13, "</think>")])
    );
}

#[test]
#[ignore = "requires FLARE_STEP5_TEST_SNAPSHOT with the Step-5 tokenizer"]
fn step5_think_tags_are_single_tokens() {
    // step5 declares the same Qwen-coder tools and think tags as qwen38.
    assert_eq!(
        delimiters(
            "FLARE_STEP5_TEST_SNAPSHOT",
            &support::contract("qwen38").frontend
        ),
        pairs(&[(128798, "<think>"), (128799, "</think>")])
    );
}

#[test]
#[ignore = "requires FLARE_MINIMAX_TEST_SNAPSHOT with the MiniMax M3 tokenizer"]
fn minimax_m3_think_tags_are_single_tokens() {
    let mut syntax = support::contract("qwen38").frontend;
    syntax.tools.as_mut().unwrap().kind = ToolParserKind::MiniMaxXml;
    syntax.reasoning = Some(ReasoningParser {
        kind: ReasoningParserKind::MmThink,
        open: "<mm:think>".into(),
        close: "</mm:think>".into(),
        wrapping_newlines: 0,
        terminal_markers: vec![],
    });
    assert_eq!(
        delimiters("FLARE_MINIMAX_TEST_SNAPSHOT", &syntax),
        pairs(&[(200059, "<mm:think>"), (200060, "</mm:think>")])
    );
}

#[test]
#[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT pointing to the DeepSeek checkpoint"]
fn deepseek41_dsml_reasoning_stays_text_matched() {
    assert_eq!(
        delimiters(
            "FLARE_DEEPSEEK_TEST_SNAPSHOT",
            &support::contract("deepseek41").frontend
        ),
        []
    );
}

#[test]
#[ignore = "requires FLARE_GEMMA4_TEST_SNAPSHOT with the Gemma 4 tokenizer"]
fn gemma4_channel_close_is_a_single_token() {
    let mut syntax = support::contract("qwen38").frontend;
    syntax.tools.as_mut().unwrap().kind = ToolParserKind::GemmaCall;
    syntax.reasoning = Some(ReasoningParser {
        kind: ReasoningParserKind::ChannelThought,
        open: "<|channel>thought".into(),
        close: "<channel|>".into(),
        wrapping_newlines: 1,
        terminal_markers: vec!["<|tool_call>".into()],
    });
    assert_eq!(
        delimiters("FLARE_GEMMA4_TEST_SNAPSHOT", &syntax),
        pairs(&[(101, "<channel|>")])
    );
}
