//! Convert reference checkpoint configuration for native frontend tests.

use crate::frontend::{Asset, TokenizerSettings};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn read(asset: &Asset) -> Result<Value, String> {
    if asset.range.is_some() {
        return Err("reference fixture requires a source configuration file".into());
    }
    let bytes = std::fs::read(&asset.path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != asset.sha256 {
        return Err("reference fixture asset SHA256 mismatch".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

pub fn tokens(asset: &Asset) -> Result<TokenizerSettings, String> {
    let source = read(asset)?;
    let content = |value: &Value| value.get("content").unwrap_or(value).clone();
    let mut settings = serde_json::Map::new();
    for name in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(value) = source.get(name).filter(|v| !v.is_null()) {
            settings.insert(name.into(), content(value));
        }
    }
    if let Some(values) = source
        .get("additional_special_tokens")
        .and_then(Value::as_array)
    {
        settings.insert(
            "additional_special_tokens".into(),
            json!({"values": values.iter().map(content).collect::<Vec<_>>()}),
        );
    }
    serde_json::from_value(settings.into()).map_err(|e| e.to_string())
}

#[cfg(feature = "images")]
pub fn image(
    preprocessor: &Asset,
    model: &Asset,
) -> Result<crate::frontend::images::ImageSettings, String> {
    let processor = read(preprocessor)?;
    let model = read(model)?;
    let settings = if processor["image_processor_type"] == "DeepSeekV4ImageProcessor" {
        json!({"deepseek4": {}})
    } else if model["model_type"] == "mimo_v2" {
        json!({"mimo": {
            "min_pixels": processor["min_pixels"], "max_pixels": processor["max_pixels"],
            "image_mean": processor["image_mean"], "image_std": processor["image_std"],
        }})
    } else {
        json!({"qwen": {
            "min_pixels": processor["size"]["shortest_edge"], "max_pixels": processor["size"]["longest_edge"],
            "position_side": 48, "image_token_id": model["image_token_id"], "video_token_id": model["video_token_id"],
        }})
    };
    serde_json::from_value(settings).map_err(|e| e.to_string())
}

pub fn contract(name: &str) -> crate::model::PreparedContract {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/support/contracts")
        .join(format!("{name}.json"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

pub fn model_config(contract: crate::model::PreparedContract) -> crate::model::ModelConfig {
    crate::model::ModelConfig {
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 1048576,
        contract,
        model_id: "frontend-fixture".into(),
        identity: [0; 32],
        limits: crate::scheduler::SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 1048576,
            prefill_chunk_size: 4096,
            pages: 8192,
            snapshots: 16,
        },
        eos_token_ids: vec![1],
        sampling_ranges: Default::default(),
        sampling_defaults: Default::default(),
        max_new_tokens: 2048,
        speculative_widths: vec![],
        speculative_fixed_width: None,
        speculative_token_capacity: 512,
        decode_weight_bytes: 0,
        history_drafts: false,
        speculative_proposal_source: Default::default(),
    }
}
