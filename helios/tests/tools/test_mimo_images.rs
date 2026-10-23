use flare_helios::{frontend, model, scheduler};
#[path = "../support/frontend.rs"]
mod support;
use flare_helios::frontend::{
    Asset, Frontend, FrontendAssets, RenderedPrompt,
    images::{ImageLimits, ImageProcessor, RgbImage},
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
            "usage: test_mimo_images SNAPSHOT ASSET_DIRECTORY CASE_JSON OUTPUT_DIRECTORY".into(),
        );
    }
    let snapshot = Path::new(&args[1]);
    let assets = Path::new(&args[2]);
    let case: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let output = Path::new(&args[4]);
    let processor = ImageProcessor::load(
        flare_helios::frontend::test_support::image(
            &asset(assets.join("preprocessor_config.json")),
            &asset(snapshot.join("config.json")),
        )
        .unwrap(),
        ImageLimits::default(),
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
            RgbImage::new(
                i["width"].as_u64().unwrap() as usize,
                i["height"].as_u64().unwrap() as usize,
                fs::read(i["pixels"].as_str().unwrap()).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let prompt = RenderedPrompt {
        message_boundaries: Vec::new(),
        image_sources: Vec::new(),
        text: case["prompt"].as_str().unwrap().into(),
        token_ids: frontend.encode(case["prompt"].as_str().unwrap(), false)?,
    };
    let input = flare_helios::model::prepare_input(
        &support::model_config(support::contract("mimo_v2")),
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
            "grids": processed.geometry.iter().map(|g| g.grid_thw).collect::<Vec<_>>(),
            "patches": processed.geometry.iter().map(|g| g.patches).collect::<Vec<_>>(),
            "positions": processed.position_ids, "cu_seqlens": processed.cu_seqlens,
        }))?,
    )?;
    Ok(())
}
