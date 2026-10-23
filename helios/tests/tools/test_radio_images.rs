use flare_helios::{frontend, model, scheduler};
#[path = "../support/frontend.rs"]
mod support;
use flare_helios::frontend::{
    Asset, Frontend, FrontendAssets, RenderedPrompt,
    images::{ImageLimits, ImageProcessor},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
fn asset(path: PathBuf) -> Asset {
    Asset {
        sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
        path,
        range: None,
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 5 {
        return Err(
            "usage: test_radio_images SNAPSHOT ASSET_DIRECTORY CASE_JSON OUTPUT_DIRECTORY".into(),
        );
    }
    let snapshot = Path::new(&args[1]);
    let assets = Path::new(&args[2]);
    let case: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let output = Path::new(&args[4]);
    let processor = ImageProcessor::load(
        flare_helios::frontend::images::ImageSettings::Radio {},
        ImageLimits {
            max_patches_per_image: 13312,
            max_total_patches: 53248,
            ..ImageLimits::default()
        },
    )?;
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(snapshot.join("tokenizer.json")),
            tokens: flare_helios::frontend::test_support::tokens(&asset(
                snapshot.join("tokenizer_config.json"),
            ))
            .unwrap(),
            chat_template: Some(asset(assets.join("chat_template.jinja"))),
        },
        Map::new(),
    )?;
    let images: Vec<_> = case["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            let bytes = fs::read(i["path"].as_str().unwrap()).unwrap();
            if i["format"] == "jpeg" {
                processor.decode_jpeg(&bytes).unwrap()
            } else {
                processor.decode_png(&bytes).unwrap()
            }
        })
        .collect();
    let prompt = RenderedPrompt {
        message_boundaries: Vec::new(),
        image_sources: Vec::new(),
        image_checkpoint_hints: Vec::new(),
        text: case["prompt"].as_str().unwrap().into(),
        token_ids: frontend.encode(case["prompt"].as_str().unwrap(), false)?,
    };
    let mut contract = support::contract("mimo_v2");
    contract.vocabulary = 131072;
    contract.input_vocabulary = 131072;
    contract.frontend = serde_json::from_value(case["frontend"].clone())?;
    contract.vision = serde_json::from_value(case["vision"].clone())?;
    let input = flare_helios::model::prepare_input(
        &support::model_config(contract),
        &frontend,
        Some(&processor),
        &prompt,
        images,
    )?;
    let mut processed = flare_helios::frontend::images::ProcessedImages::default();
    for image in &input.conditioned_inputs { processed.append(image.materialize()?); }
    fs::write(
        output.join("pixels.f32"),
        processed
            .pixel_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    fs::write(
        output.join("metadata.json"),
        serde_json::to_vec(&json!({
            "tokens": input.tokens, "indices": input.data.vision_token_indices,
            "spans": input.data.spans.iter().map(|r| [r.start, r.end]).collect::<Vec<_>>(),
            "grids": processed.geometry.iter().map(|g| g.grid_thw).collect::<Vec<_>>(),
            "patches": processed.geometry.iter().map(|g| g.patches).collect::<Vec<_>>(),
            "interpolation_indices": processed.interpolation_indices, "interpolation_weights": processed.interpolation_weights,
            "positions": processed.position_ids, "cu_seqlens": processed.cu_seqlens,
        }))?,
    )?;
    Ok(())
}
