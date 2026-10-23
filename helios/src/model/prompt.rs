//! Prepared frontend syntax and image layouts construct shared request data.

use crate::{
    Result,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, RgbImage},
    },
    model::{ModelConfig, PreparedInput, PromptData, contract::ImageLayoutKind},
};
use serde_json::{Map, Value};

pub fn prepare_input(
    config: &ModelConfig,
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: Vec<RgbImage>,
) -> Result<PreparedInput> {
    let images = if images.is_empty() { vec![] } else {
        let processor = processor.ok_or("image processing is unavailable")?;
        images.into_iter().map(|image| Ok(crate::frontend::images::ImageInput {
            description: processor.describe_image(&image)?,
            source: crate::frontend::images::ImageSource::Decoded(std::sync::Arc::new(image)),
        })).collect::<Result<Vec<_>>>()?
    };
    prepare_image_inputs(config, frontend, processor, rendered, images)
}

pub fn prepare_image_inputs(
    config: &ModelConfig,
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    inputs: Vec<crate::frontend::images::ImageInput>,
) -> Result<PreparedInput> {
    let images = inputs.iter().map(|i| i.description.clone()).collect::<Vec<_>>();
    if rendered
        .token_ids
        .iter()
        .any(|token| config.contract.frontend.rejected_tokens.contains(token))
    {
        return Err("prompt contains an input token outside the prepared frontend contract".into());
    }
    let mut prepared = match config.contract.frontend.image_layout {
        Some(ImageLayoutKind::Bracketed) => crate::frontend::images::layouts::bracketed::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::Glm) => crate::frontend::images::layouts::glm::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::Qwen) => crate::frontend::images::layouts::qwen::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::Moonvit) => crate::frontend::images::layouts::moonvit::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::Radio) => crate::frontend::images::layouts::radio::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::Step) => crate::frontend::images::layouts::step::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::Mimo) => crate::frontend::images::layouts::mimo::prepare_input(
            frontend, processor, rendered, &images,
        )?,
        Some(ImageLayoutKind::DeepseekRowMajor) => {
            crate::frontend::images::layouts::deepseek::prepare_row_major_input(
                config.contract.vocabulary,
                frontend,
                processor,
                rendered,
                &images,
            )?
        }
        Some(ImageLayoutKind::Deepseek) => {
            crate::frontend::images::layouts::deepseek::prepare_input(
                config.contract.vocabulary,
                frontend,
                processor,
                rendered,
                &images,
            )?
        }
        None => {
            if !images.is_empty() {
                return Err("prepared input contract has no image layout".into());
            }
            PreparedInput {
                tokens: rendered.token_ids.clone(),
                token_expansions: vec![],
                data: PromptData {
                    positions: if config.contract.frontend.position_dimensions == 3 {
                        (0..rendered.token_ids.len())
                            .map(|p| [p as i64; 3])
                            .collect()
                    } else {
                        vec![]
                    },
                    ..Default::default()
                },
                conditioning: vec![],
                conditioned_inputs: vec![],
            }
        }
    };
    if prepared.token_expansions.len() != inputs.len() {
        return Err("image expansions do not match input descriptions".into());
    }
    if !images.is_empty() {
        let processor = std::sync::Arc::new(processor.ok_or("image processing is unavailable")?.clone());
        let geometry = images.iter().map(|i| i.geometry).collect::<Vec<_>>();
        let mut feature_offset = 0;
        let mut expansion_offset = 0;
        for ((input, geometry), (original, additional)) in inputs.into_iter().zip(&geometry)
            .zip(&prepared.token_expansions) {
            let start = original + expansion_offset;
            prepared.conditioned_inputs.push(crate::frontend::images::PreparedImageInput::new(
                input, start..start + additional + 1,
                feature_offset..feature_offset + geometry.tokens, processor.clone(),
            ));
            feature_offset += geometry.tokens;
            expansion_offset += additional;
        }
    }
    super::validate_prompt_data(config, &prepared.tokens, &prepared.data)?;
    validate_conditioned_inputs(config, &prepared.tokens, &prepared.data, &prepared.conditioned_inputs)?;
    Ok(prepared)
}

pub(crate) fn validate_conditioned_inputs(
    config: &ModelConfig, tokens: &[u32], data: &PromptData,
    inputs: &[crate::frontend::images::PreparedImageInput],
) -> Result<()> {
    let mut token_end = 0;
    let mut feature_end = 0;
    let vision = &config.contract.vision;
    for input in inputs {
        if input.token_span.start < token_end || input.token_span.start >= input.token_span.end
            || input.token_span.end > tokens.len() || input.feature_range.start != feature_end
            || input.feature_range.end.checked_sub(feature_end) != Some(input.geometry.tokens)
            || input.geometry.tokens == 0 || input.geometry.tokens > vision.features_per_request
            || input.geometry.patches > vision.patch_capacities.last().copied().unwrap_or(0)
            || input.geometry.tokens.checked_mul(vision.patches_per_feature) != Some(input.geometry.patches)
            || input.feature_range.end > tokens.len() || input.feature_range.end > i32::MAX as usize
        {
            return Err("conditioned input spans or features violate prompt geometry".into());
        }
        let indices = data.vision_token_indices.get(input.token_span.clone())
            .ok_or("conditioned input has no prompt feature indices")?;
        let mut seen = vec![false; input.geometry.tokens];
        for index in indices.iter().copied().filter(|index| *index >= 0) {
            let local = (index as usize).checked_sub(feature_end)
                .filter(|local| *local < seen.len()).ok_or("conditioned feature belongs to another input")?;
            if std::mem::replace(&mut seen[local], true) {
                return Err("conditioned feature appears more than once in the prompt".into());
            }
        }
        if seen.iter().any(|seen| !seen) {
            return Err("conditioned input features are absent from its prompt span".into());
        }
        token_end = input.token_span.end;
        feature_end = input.feature_range.end;
    }
    if data.vision_token_indices.iter().filter(|index| **index >= 0).count() != feature_end {
        return Err("prompt features are not owned by conditioned input spans".into());
    }
    Ok(())
}

pub fn prepare_output(
    config: &ModelConfig,
    frontend: &Frontend,
    rendered: &RenderedPrompt,
    messages: &[Value],
    options: &Map<String, Value>,
    structured: bool,
) -> Result<crate::output::OutputConfig> {
    let syntax = &config.contract.frontend;
    let effective = |name: &str| {
        options
            .get(name)
            .or_else(|| frontend.template_default(name))
    };
    let continuation = effective("continue_final_message")
        .is_some_and(|v| v == &Value::Bool(true) || v.as_str().is_some_and(|s| !s.is_empty()));
    let open_assistant = effective("add_generation_prompt") == Some(&Value::Bool(true))
        || (continuation && messages.last().is_some_and(|m| m["role"] == "assistant"));
    let turn = open_assistant
        .then(|| rendered.text.rsplit_once(&syntax.assistant_marker))
        .flatten()
        .map(|(_, suffix)| suffix);
    let prefix = turn.filter(|suffix| {
        !syntax
            .message_end_markers
            .iter()
            .any(|end| suffix.contains(end))
    });
    // A template may resume the assistant turn after an embedded tool response.
    let resumed = turn.map(|suffix| {
        let end = syntax
            .message_end_markers
            .iter()
            .filter_map(|end| suffix.rfind(end.as_str()).map(|at| at + end.len()))
            .max();
        &suffix[end.unwrap_or(0)..]
    });
    let mut reasoning = syntax.reasoning.as_ref().is_some_and(|r| {
        resumed.is_some_and(|prefix| {
            prefix
                .rfind(&r.open)
                .is_some_and(|open| prefix.rfind(&r.close).is_none_or(|close| open > close))
        })
    });
    if let Some(r) = syntax
        .reasoning
        .as_ref()
        .filter(|r| r.kind == super::contract::ReasoningParserKind::XtmlThink)
    {
        // Ordinary XTML text can spell a delimiter without containing its control tokens.
        let last_marker = |marker: &str| -> Result<Option<usize>> {
            let ids = frontend.encode(marker, false)?;
            if ids.is_empty() {
                return Err("reasoning delimiters must encode to nonempty token sequences".into());
            }
            Ok(rendered
                .token_ids
                .windows(ids.len())
                .rposition(|tokens| tokens == ids))
        };
        let opened = last_marker(&r.open)?;
        let closed = last_marker(&r.close)?;
        reasoning =
            open_assistant && opened.is_some_and(|open| closed.is_none_or(|close| open > close));
    }
    let tools = options
        .get("tools")
        .or_else(|| frontend.template_default("tools"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !tools.is_empty() && syntax.tools.is_none() {
        return Err("tool parsing is not prepared".into());
    }
    let control_tokens = if syntax
        .tools
        .as_ref()
        .is_some_and(|tool| tool.kind == super::contract::ToolParserKind::Xtml)
    {
        ["<|open|>", "<|close|>", "<|sep|>"]
            .into_iter()
            .map(|marker| {
                let ids = frontend.encode(marker, false)?;
                if ids.len() != 1 {
                    return Err("XTML delimiters must each occupy one tokenizer token".into());
                }
                Ok((ids[0], marker.to_owned()))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![]
    };
    let start = if !tools.is_empty()
        && syntax.tools.as_ref().is_some_and(|tool| {
            tool.kind == super::contract::ToolParserKind::Dsml
                && prefix.is_some_and(|suffix| {
                    let open = if tool.tag_separator == " " {
                        "<｜DSML｜ calls>"
                    } else {
                        "<｜DSML｜tool_calls>"
                    };
                    suffix.trim_end().ends_with(open)
                })
        }) {
        crate::output::tagged::OutputStart::ToolBody
    } else if reasoning {
        crate::output::tagged::OutputStart::Reasoning
    } else {
        crate::output::tagged::OutputStart::Content
    };
    Ok(crate::output::OutputConfig {
        syntax: syntax.clone(),
        control_tokens,
        content: crate::output::tagged::OutputConfig {
            structured,
            start,
            tools,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn image_frontend() -> Frontend {
        use crate::frontend::{Asset, FrontendAssets};
        use sha2::{Digest, Sha256};
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!("flare-image-history-{}-{}", std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = (0..128)
            .map(|id| ((id as u8 as char).to_string(), id)).collect();
        let model = tokenizers::models::bpe::BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.add_special_tokens([
            "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
            "<｜deepseek_image｜>", "<|media_pad|>", "<|media_begin|>", "<|media_content|>",
            "<|media_end|>", "<im_patch>", "<patch_start>", "<patch_end>", "<patch_newline>",
            "<im_start>", "<im_end>", "<image>", "<img>", "</img>",
        ].map(|text| tokenizers::AddedToken::from(text, true))).unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name); std::fs::write(&path, bytes).unwrap();
            Asset { range: None, path, sha256: format!("{:x}", Sha256::digest(bytes)) }
        };
        let frontend = Frontend::load(FrontendAssets {
            tokenizer: asset("tokenizer.json", tokenizer.to_string(false).unwrap().as_bytes()),
            tokens: Default::default(), chat_template: Some(asset("chat_template.jinja", b"{% mark %}{{ messages[0].content }}")),
        }, Map::new()).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        frontend
    }

    #[test]
    fn every_image_layout_describes_history_larger_than_the_feature_window() {
        use crate::frontend::images::{DeepseekRowMajorImageSettings, ImageSettings,
            MimoImageSettings, QwenImageSettings};
        use crate::model::contract::{ImageSpan, ImageTokenLayout};
        let frontend = image_frontend();
        let token = |text: &str| frontend.encode(text, false).unwrap()[0];
        let profiles = [
            (ImageLayoutKind::Qwen, ImageSettings::Qwen(QwenImageSettings {
                min_pixels: 4096, max_pixels: 65536, position_side: 48,
                image_token_id: token("<|image_pad|>"), video_token_id: token("<|video_pad|>"),
            }), "<|image_pad|>", 4),
            (ImageLayoutKind::Mimo, ImageSettings::Mimo(MimoImageSettings {
                min_pixels: 4096, max_pixels: 65536, image_mean: [0.5; 3], image_std: [0.5; 3],
            }), "<|image_pad|>", 4),
            (ImageLayoutKind::Deepseek, ImageSettings::Deepseek4 {}, "<｜deepseek_image｜>", 9),
            (ImageLayoutKind::DeepseekRowMajor, ImageSettings::DeepseekRowMajor(
                DeepseekRowMajorImageSettings { min_pixels: 46656, max_tokens: 1024,
                    max_width_height_ratio: None }), "<｜deepseek_image｜>", 9),
            (ImageLayoutKind::Moonvit, ImageSettings::Moonvit {}, "<|media_pad|>", 4),
            (ImageLayoutKind::Step, ImageSettings::Step {}, "<im_patch>", 16),
            (ImageLayoutKind::Radio, ImageSettings::Radio {}, "<image>", 4),
        ];
        for (layout, settings, placeholder, merge) in profiles {
            let initial = ImageProcessor::load(settings.clone(), Default::default()).unwrap();
            let geometry = initial.geometry(64, 64).unwrap();
            let mut config = super::super::tests::config();
            config.limits.max_sequence_tokens = 8192;
            config.limits.prefill_chunk_size = 4096;
            config.contract.vocabulary = 1000;
            config.contract.input_vocabulary = 1005;
            config.contract.vision = crate::model::VisionCapabilities {
                feature_slots: 4, features_per_request: geometry.tokens,
                max_images: 1, patches_per_feature: merge, patch_capacities: vec![geometry.patches],
            };
            let syntax = &mut config.contract.frontend;
            syntax.image_layout = Some(layout);
            syntax.position_dimensions = if layout == ImageLayoutKind::Qwen { 3 } else { 0 };
            syntax.rejected_tokens.clear();
            syntax.image_tokens = Some(ImageTokenLayout {
                feature_token: match layout { ImageLayoutKind::Deepseek => 1002,
                    ImageLayoutKind::DeepseekRowMajor => 1001, _ => token(placeholder) },
                span: match layout {
                    ImageLayoutKind::Deepseek => Some(ImageSpan { start_token: 1000, end_token: 1004,
                        max_tokens: 4096, interior_tokens: vec![1003], unscoped_tokens: vec![1001] }),
                    ImageLayoutKind::DeepseekRowMajor => Some(ImageSpan { start_token: 1000, end_token: 1003,
                        max_tokens: 4096, interior_tokens: vec![1002], unscoped_tokens: vec![] }),
                    _ => None,
                },
            });
            let limits = crate::frontend::images::ImageLimits {
                max_images: config.limits.max_sequence_tokens,
                max_total_patches: config.limits.max_sequence_tokens * merge,
                max_patches_per_image: geometry.patches, patches_per_feature: merge,
                ..Default::default()
            };
            let processor = ImageProcessor::load(settings, limits).unwrap();
            let text = format!("a{placeholder}b{placeholder}c{placeholder}d");
            let rendered = RenderedPrompt { message_boundaries: Vec::new(), token_ids: frontend.encode(&text, false).unwrap(),
                text, image_sources: vec![] };
            let images = (0..3).map(|i| RgbImage::new(64, 64, vec![i; 64 * 64 * 3]).unwrap()).collect();
            let prepared = prepare_input(&config, &frontend, Some(&processor), &rendered, images).unwrap();
            assert_eq!(prepared.conditioned_inputs.len(), 3, "{layout:?}");
            for (index, input) in prepared.conditioned_inputs.iter().enumerate() {
                assert_eq!(input.feature_range, index * geometry.tokens..(index + 1) * geometry.tokens);
                assert_eq!(input.geometry, geometry);
            }
            assert_eq!(prepared.data.vision_token_indices.iter().copied().max(), Some((3 * geometry.tokens - 1) as i32));
            assert_eq!(processor.preparation_counts().prepared_images, 0);
            assert!(prepared.tokens.len() < config.limits.max_sequence_tokens);
            let mut malformed = prepared.data.clone();
            let first = malformed.vision_token_indices.iter_mut().find(|index| **index >= 0).unwrap();
            *first = geometry.tokens as i32;
            assert!(validate_conditioned_inputs(&config, &prepared.tokens, &malformed,
                &prepared.conditioned_inputs).is_err());
            let mut oversized = config.clone();
            oversized.contract.vision.features_per_request -= 1;
            assert!(validate_conditioned_inputs(&oversized, &prepared.tokens, &prepared.data,
                &prepared.conditioned_inputs).is_err());
        }
    }

    #[test]
    fn xtml_continuation_distinguishes_literal_channel_markers() {
        use crate::frontend::{Asset, FrontendAssets};
        use crate::model::contract::{ReasoningParserKind, ToolParserKind};
        use sha2::{Digest, Sha256};

        let root = std::env::temp_dir().join(format!("flare-xtml-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let vocab: tokenizers::models::bpe::Vocab = (0..128)
            .map(|id| ((id as u8 as char).to_string(), id))
            .collect();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer
            .add_special_tokens(
                ["<|open|>", "<|close|>", "<|sep|>"]
                    .map(|text| tokenizers::AddedToken::from(text, true)),
            )
            .unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = root.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
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
                chat_template: Some(asset("chat_template.jinja", b"{% mark %}{{ messages[0].content }}")),
            },
            Map::new(),
        )
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        let mut config = super::super::tests::config();
        let mut syntax = serde_json::from_str::<crate::model::PreparedContract>(include_str!(
            "../../tests/support/contracts/qwen38.json"
        ))
        .unwrap()
        .frontend;
        syntax.assistant_marker = "<|open|>message role=\"assistant\"<|sep|>".into();
        syntax.message_end_markers = vec!["<|close|>message<|sep|>".into()];
        syntax.tools.as_mut().unwrap().kind = ToolParserKind::Xtml;
        let reason = syntax.reasoning.as_mut().unwrap();
        reason.kind = ReasoningParserKind::XtmlThink;
        reason.open = "<|open|>think<|sep|>".into();
        reason.close = "<|close|>think<|sep|>".into();
        config.contract.frontend = syntax;
        for (protocol, literal, expected) in [
            (
                "<|open|>think<|sep|>",
                "literal <|close|>think<|sep|>",
                true,
            ),
            (
                "<|open|>think<|sep|>",
                "literal <|close|>message<|sep|>",
                true,
            ),
            (
                "<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>",
                "literal <|open|>think<|sep|>",
                false,
            ),
        ] {
            let prefix = config.contract.frontend.assistant_marker.clone() + protocol;
            let mut token_ids = frontend.encode(&prefix, false).unwrap();
            token_ids.extend(literal.bytes().map(u32::from));
            let rendered = RenderedPrompt { message_boundaries: Vec::new(),
                image_sources: vec![],
                text: prefix + literal,
                token_ids,
            };
            let output = prepare_output(
                &config,
                &frontend,
                &rendered,
                &[json!({"role":"assistant", "content":literal})],
                json!({"continue_final_message":true}).as_object().unwrap(),
                false,
            )
            .unwrap();
            assert_eq!(
                output.content.start == crate::output::tagged::OutputStart::Reasoning,
                expected,
                "{literal}"
            );
        }
    }

    #[test]
    fn initial_reasoning_uses_only_an_open_assistant_prefix() {
        let mut config = super::super::tests::config();
        let prepared: crate::model::PreparedContract =
            serde_json::from_str(include_str!("../../tests/support/contracts/qwen38.json"))
                .unwrap();
        config.contract.frontend = prepared.frontend;
        let frontend = crate::server::output::tests::frontend(&["x"]);
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/frontend/qwen38.json"
        )))
        .unwrap();
        for (name, reasoning) in [
            ("generation_prompt", true),
            ("thinking_disabled", false),
            ("assistant_continuation", false),
            ("reasoning_continuation", true),
            ("continuation_trailing_spaces", false),
            ("continuation_text_blocks", false),
        ] {
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == name)
                .unwrap();
            let rendered = RenderedPrompt { message_boundaries: Vec::new(),
                image_sources: vec![],
                text: case["rendered"].as_str().unwrap().into(),
                token_ids: vec![],
            };
            let output = prepare_output(
                &config,
                &frontend,
                &rendered,
                case["messages"].as_array().unwrap(),
                case["options"].as_object().unwrap(),
                false,
            )
            .unwrap();
            assert_eq!(
                output.content.start == crate::output::tagged::OutputStart::Reasoning,
                reasoning,
                "{name}"
            );
        }
        for role in ["user", "tool"] {
            let messages = vec![json!({"role":role,"content":"literal <think>"})];
            for (options, suffix, expected) in [
                (json!({}), "", false),
                (
                    json!({"add_generation_prompt":true}),
                    "<|im_start|>assistant\n",
                    false,
                ),
                (
                    json!({"add_generation_prompt":true}),
                    "<|im_start|>assistant\n<think>\n",
                    true,
                ),
            ] {
                let rendered = RenderedPrompt { message_boundaries: Vec::new(),
                    image_sources: vec![],
                    text: format!("<|im_start|>{role}\nliteral <think><|im_end|>\n{suffix}"),
                    token_ids: vec![],
                };
                assert_eq!(
                    prepare_output(
                        &config,
                        &frontend,
                        &rendered,
                        &messages,
                        options.as_object().unwrap(),
                        false
                    )
                    .unwrap()
                    .content
                    .start
                        == crate::output::tagged::OutputStart::Reasoning,
                    expected
                );
            }
        }
        let messages = vec![json!({"role":"user","content":"<think>"})];
        let rendered = RenderedPrompt { message_boundaries: Vec::new(),
            image_sources: vec![],
            text: "<|im_start|>user\n<think>".into(),
            token_ids: vec![],
        };
        assert!(
            prepare_output(
                &config,
                &frontend,
                &rendered,
                &messages,
                json!({"continue_final_message":true}).as_object().unwrap(),
                false
            )
            .unwrap()
            .content
            .start
                != crate::output::tagged::OutputStart::Reasoning
        );
    }

    #[test]
    fn output_start_follows_the_open_dsml_assistant_suffix() {
        use crate::output::tagged::OutputStart;
        let frontend = crate::server::output::tests::frontend(&["x"]);
        let mut config = super::super::tests::config();
        let prepared: crate::model::PreparedContract = serde_json::from_str(include_str!(
            "../../tests/support/contracts/deepseek41.json"
        ))
        .unwrap();
        config.contract.frontend = prepared.frontend;
        let tools = json!([{"type":"function","function":{"name":"lookup"}}]);
        for separator in ["", " "] {
            config
                .contract
                .frontend
                .tools
                .as_mut()
                .unwrap()
                .tag_separator = separator.into();
            let open = if separator.is_empty() {
                "<｜DSML｜tool_calls>"
            } else {
                "<｜DSML｜ calls>"
            };
            for (suffix, expected) in [
                (format!("</think>\n\n{open}\n"), OutputStart::ToolBody),
                ("<think>".into(), OutputStart::Reasoning),
                ("</think>".into(), OutputStart::Content),
            ] {
                let rendered = RenderedPrompt { message_boundaries: Vec::new(),
                    image_sources: vec![],
                    text: format!("<｜User｜>Hello<｜Assistant｜>{suffix}"),
                    token_ids: vec![],
                };
                let options = json!({"add_generation_prompt":true,"tools":tools});
                let output = prepare_output(
                    &config,
                    &frontend,
                    &rendered,
                    &[json!({"role":"user","content":"Hello"})],
                    options.as_object().unwrap(),
                    true,
                )
                .unwrap();
                assert_eq!(output.content.start, expected, "{suffix}");
                let envelope = crate::output::OutputPipeline::grammar_envelope(&output).unwrap();
                assert_eq!(envelope.tool_body_start, expected == OutputStart::ToolBody);
            }
        }
    }

    #[test]
    fn reasoning_resumes_after_an_embedded_tool_response() {
        use crate::model::contract::{ReasoningParser, ReasoningParserKind, ToolParserKind};
        use crate::output::tagged::OutputStart;
        let frontend = crate::server::output::tests::frontend(&["x"]);
        let mut config = super::super::tests::config();
        let mut syntax = serde_json::from_str::<crate::model::PreparedContract>(include_str!(
            "../../tests/support/contracts/qwen38.json"
        ))
        .unwrap()
        .frontend;
        syntax.assistant_marker = "<|turn>model\n".into();
        syntax.message_end_markers = vec!["<turn|>".into(), "<|tool_response>".into()];
        syntax.tools.as_mut().unwrap().kind = ToolParserKind::GemmaCall;
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ChannelThought,
            open: "<|channel>thought".into(),
            close: "<channel|>".into(),
            wrapping_newlines: 1,
            terminal_markers: vec![],
        });
        config.contract.frontend = syntax;
        let call = "<|turn>user\nHi<turn|>\n<|turn>model\n<|tool_call>call:f{}<tool_call|>\
                    <|tool_response>response:f{value:<|\"|>ok<|\"|>}<tool_response|>";
        for (text, expected) in [
            (format!("{call}<|channel>thought\n"), OutputStart::Reasoning),
            (call.into(), OutputStart::Content),
            (
                "<|turn>user\n<|channel>thought<turn|>\n<|turn>model\n".into(),
                OutputStart::Content,
            ),
            (
                "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>".into(),
                OutputStart::Content,
            ),
        ] {
            let rendered = RenderedPrompt {
                message_boundaries: Vec::new(),
                image_sources: vec![],
                text: text.clone(),
                token_ids: vec![],
            };
            let output = prepare_output(
                &config,
                &frontend,
                &rendered,
                &[json!({"role":"user","content":"Hi"})],
                json!({"add_generation_prompt":true}).as_object().unwrap(),
                false,
            )
            .unwrap();
            assert_eq!(output.content.start, expected, "{text}");
        }
    }
}
