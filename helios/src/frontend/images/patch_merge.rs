//! Temporal RGB patch packing with configurable image envelope tokens.

use super::*;

#[derive(Clone)]
pub(super) struct Processor {
    pub(super) patches: patches::PatchProcessor,
    image_token: String,
    start_token: String,
    end_token: String,
}

impl Processor {
    pub(super) fn load(settings: PatchMergeImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.patch_size != 14
            || settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.image_mean.iter().any(|x| !x.is_finite())
            || settings
                .image_std
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.)
            || [
                &settings.image_token,
                &settings.start_token,
                &settings.end_token,
            ]
            .iter()
            .any(|s| s.is_empty())
            || settings.image_token == settings.start_token
            || settings.image_token == settings.end_token
            || settings.start_token == settings.end_token
        {
            return Err("invalid temporal patch-merge image settings".into());
        }
        Ok(Self {
            patches: patches::PatchProcessor {
                limits,
                patch_size: settings.patch_size,
                min_pixels: settings.min_pixels,
                max_pixels: settings.max_pixels,
                normalization: patches::Normalization::ScaledChannel {
                    mean: settings.image_mean,
                    std: settings.image_std,
                },
                resize: resize::resize,
            },
            image_token: settings.image_token,
            start_token: settings.start_token,
            end_token: settings.end_token,
        })
    }

    pub(super) fn expand_prompt(
        &self,
        frontend: &Frontend,
        text: &str,
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        let image_ids = frontend.encode(&self.image_token, false)?;
        let [image_id] = image_ids.as_slice() else {
            return Err("image placeholder must encode one token".into());
        };
        for marker in [&self.start_token, &self.end_token] {
            if frontend.encode(marker, false)?.len() != 1 {
                return Err("image envelope marker must encode one token".into());
            }
        }
        if text.matches(&self.image_token).count() != images.geometry.len() {
            return Err("image placeholder count does not match input images".into());
        }
        let mut parts = text.split(&self.image_token);
        let mut expanded = parts.next().unwrap_or_default().to_owned();
        for (part, geometry) in parts.zip(&images.geometry) {
            expanded.push_str(&self.start_token);
            for _ in 0..geometry.tokens {
                expanded.push_str(&self.image_token);
            }
            expanded.push_str(&self.end_token);
            expanded.push_str(part);
        }
        let token_ids = frontend.encode(&expanded, false)?;
        let mm_token_type_ids = token_ids
            .iter()
            .map(|id| i32::from(*id == *image_id))
            .collect();
        Ok(ImagePrompt {
            text: expanded,
            position_ids: std::array::from_fn(|_| (0..token_ids.len() as i64).collect()),
            token_ids,
            mm_token_type_ids,
            rope_delta: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::PatchMerge(PatchMergeImageSettings {
                patch_size: 14,
                min_pixels: 3136,
                max_pixels: 451584,
                image_mean: [0.48145466, 0.4578275, 0.40821073],
                image_std: [0.26862954, 0.26130258, 0.27577711],
                image_token: "image".into(),
                start_token: "start".into(),
                end_token: "end".into(),
            }),
            ImageLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn patch_geometry_and_duplicate_frames() {
        let processor = processor();
        assert_eq!(processor.geometry(672, 672).unwrap().patches, 2304);
        assert_eq!(processor.geometry(1, 1).unwrap().grid_thw, [1, 4, 4]);
        assert_eq!(processor.geometry(1008, 672).unwrap().grid_thw, [1, 38, 58]);
        let image =
            RgbImage::new(56, 56, (0..56 * 56 * 3).map(|x| (x % 256) as u8).collect()).unwrap();
        let images = processor.process(&[image]).unwrap();
        assert_eq!(images.pixel_values.len(), 16 * 1176);
        assert_eq!(&images.position_ids[..4], &[[0, 0], [0, 1], [1, 0], [1, 1]]);
        assert_eq!(images.cu_seqlens, [0, 16]);
        for patch in images.pixel_values.chunks_exact(1176) {
            for channel in patch.chunks_exact(392) {
                assert_eq!(&channel[..196], &channel[196..]);
            }
        }
        assert!((images.pixel_values[0] + 0.48145466 / 0.26862954).abs() < 1e-6);
    }
    #[test]
    fn native_pixels_match_canonical_torchvision_processor() {
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
        if !python.is_file() {
            return;
        }
        let script = r#"
import sys, struct, torch
from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor
image=(torch.arange(37*91*3,dtype=torch.int64)%256).to(torch.uint8).view(37,91,3).permute(2,0,1)
result=MiniMaxM3VLImageProcessor()(image,return_tensors='pt')
sys.stdout.buffer.write(struct.pack('<qqq',*result['image_grid_thw'][0].tolist()))
sys.stdout.buffer.write(result['pixel_values'].contiguous().numpy().tobytes())
"#;
        let reference = std::process::Command::new(python)
            .args(["-c", script])
            .output()
            .unwrap();
        assert!(
            reference.status.success(),
            "{}",
            String::from_utf8_lossy(&reference.stderr)
        );
        let image =
            RgbImage::new(91, 37, (0..37 * 91 * 3).map(|x| (x % 256) as u8).collect()).unwrap();
        let actual = processor().process(&[image]).unwrap();
        let grid: Vec<i64> = reference.stdout[..24]
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual.geometry[0].grid_thw.as_slice(), grid.as_slice());
        let pixels: Vec<f32> = reference.stdout[24..]
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual.pixel_values.len(), pixels.len());
        for (i, (a, b)) in actual.pixel_values.iter().zip(pixels).enumerate() {
            assert_eq!(*a, b, "pixel {i}");
        }
    }
}
