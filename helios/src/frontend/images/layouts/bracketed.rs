//! Expand image envelopes and attach linear feature indices to image tokens.

use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, ImageDescription},
    },
    model::{PreparedInput, PromptData},
};

pub(crate) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
) -> Result<PreparedInput> {
    if images.is_empty() {
        if let Some(processor) = processor {
            let empty = crate::frontend::images::ProcessedImages {
                pixel_values: vec![],
                geometry: vec![],
                position_ids: vec![],
                interpolation_indices: vec![],
                interpolation_weights: vec![],
                cu_seqlens: vec![0],
            };
            processor.expand_prompt(frontend, &rendered.text, &empty)?;
        }
        return Ok(PreparedInput {
            tokens: rendered.token_ids.clone(),
            data: PromptData {
                vision_token_indices: vec![-1; rendered.token_ids.len()],
                ..Default::default()
            },
            conditioning: vec![],
            token_expansions: vec![],
            conditioned_inputs: vec![],
        });
    }
    let processor = processor.ok_or("image processing is unavailable")?;
    let processed = processor.describe_inputs(images)?;
    let expanded = processor.expand_prompt(frontend, &rendered.text, &processed)?;
    let mut conditioning = Vec::new();
    let mut image = 0;
    let mut feature = 0;
    let mut indices = Vec::with_capacity(expanded.token_ids.len());
    for (position, kind) in expanded.mm_token_type_ids.iter().enumerate() {
        if *kind == 0 {
            indices.push(-1);
            continue;
        }
        if position == 0 || expanded.mm_token_type_ids[position - 1] == 0 {
            let source = images.get(image).ok_or("unexpected image token span")?;
            conditioning.push(ConditioningEvent {
                position: position as u64,
                identity: source.identity,
            });
            image += 1;
        }
        indices.push(feature);
        feature += 1;
    }
    if image != images.len()
        || feature as usize != processed.geometry.iter().map(|g| g.tokens).sum::<usize>()
    {
        return Err("image token spans do not match processed images".into());
    }
    let image_token = expanded.token_ids[expanded.mm_token_type_ids.iter().position(|kind| *kind == 1)
        .ok_or("processed image has no prompt features")?];
    let token_expansions = rendered.token_ids.iter().enumerate()
        .filter(|(_, token)| **token == image_token)
        .zip(&processed.geometry)
        .map(|((index, _), geometry)| (index, geometry.tokens + 1))
        .collect();
    Ok(PreparedInput {
        token_expansions,
        tokens: expanded.token_ids,
        data: PromptData {
            vision_token_indices: indices,
            ..Default::default()
        },
        conditioning,
        conditioned_inputs: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{
        Asset, FrontendAssets,
        images::{ImageLimits, ImageSettings, PatchMergeImageSettings, RgbImage},
    };
    use sha2::{Digest, Sha256};
    use std::{fs, path::PathBuf};

    #[test]
    fn checkpoint_image_envelopes_preserve_feature_order() {
        let Some(snapshot) = std::env::var_os("FLARE_MINIMAX_M3_SNAPSHOT").map(PathBuf::from) else {
            return;
        };
        if !snapshot.join("tokenizer.json").is_file() {
            return;
        }
        let asset = |name: &str| {
            let path = snapshot.join(name);
            Asset {
                sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
                path,
                range: None,
            }
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset("tokenizer.json"),
                tokens: crate::frontend::test_support::tokens(&asset("tokenizer_config.json"))
                    .unwrap(),
                chat_template: Some({
                    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../ignition/recipes/minimax_m3/assets/chat_template.jinja");
                    Asset { sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())), path, range: None }
                }),
            },
            Default::default(),
        )
        .unwrap();
        let processor = ImageProcessor::load(
            ImageSettings::PatchMerge(PatchMergeImageSettings {
                patch_size: 14,
                min_pixels: 3136,
                max_pixels: 451584,
                image_mean: [0.48145466, 0.4578275, 0.40821073],
                image_std: [0.26862954, 0.26130258, 0.27577711],
                image_token: "]<]image[>[".into(),
                start_token: "]<]start of image[>[".into(),
                end_token: "]<]end of image[>[".into(),
            }),
            ImageLimits::default(),
        )
        .unwrap();
        let rendered = frontend.render_chat(&[serde_json::json!({"role":"user", "content":[
            {"type":"text", "text":"Compare "}, {"type":"image"}, {"type":"text", "text":" and "}, {"type":"image"} ]})],
            serde_json::json!({"add_generation_prompt":true}).as_object().unwrap()).unwrap();
        let images = vec![
            RgbImage::new(56, 56, vec![0; 56 * 56 * 3]).unwrap(),
            RgbImage::new(672, 672, vec![255; 672 * 672 * 3]).unwrap(),
        ];
        let images = images.iter().map(|image| processor.describe_image(image).unwrap()).collect::<Vec<_>>();
        let prepared = prepare_input(&frontend, Some(&processor), &rendered, &images).unwrap();
        assert_eq!(
            prepared.tokens.iter().filter(|id| **id == 200025).count(),
            580
        );
        assert_eq!(prepared.conditioning.len(), 2);
        assert_eq!(
            prepared
                .data
                .vision_token_indices
                .iter()
                .copied()
                .filter(|i| *i >= 0)
                .collect::<Vec<_>>(),
            (0..580).collect::<Vec<_>>()
        );
        let start = frontend.encode("]<]start of image[>[", false).unwrap()[0];
        let end = frontend.encode("]<]end of image[>[", false).unwrap()[0];
        for (event, count) in prepared.conditioning.iter().zip([4usize, 576]) {
            let p = event.position as usize;
            assert_eq!(prepared.tokens[p - 1], start);
            assert_eq!(prepared.tokens[p + count], end);
            assert!(prepared.tokens[p..p + count].iter().all(|id| *id == 200025));
        }
        assert!(prepare_input(&frontend, Some(&processor), &rendered, &[]).is_err());
    }
}
