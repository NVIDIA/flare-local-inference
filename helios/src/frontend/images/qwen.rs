//! Qwen image profile, learned position interpolation and multimodal prompt positions.

use super::*;

#[derive(Clone)]
pub(super) struct QwenProcessor {
    pub(super) patches: patches::PatchProcessor,
    pub(super) position_side: usize,
    pub(super) image_token_id: u32,
    pub(super) video_token_id: u32,
}

impl QwenProcessor {
    pub(super) fn load(settings: &QwenImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.min_pixels == 0
            || settings.max_pixels < settings.min_pixels
            || settings.position_side != 48
            || settings.image_token_id == settings.video_token_id
        {
            return Err("invalid prepared Qwen image settings".into());
        }
        Ok(Self {
            patches: patches::PatchProcessor {
                limits,
                patch_size: PATCH,
                min_pixels: settings.min_pixels,
                max_pixels: settings.max_pixels,
                normalization: patches::Normalization::Symmetric,
                resize: resize::resize,
            },
            position_side: settings.position_side,
            image_token_id: settings.image_token_id,
            video_token_id: settings.video_token_id,
        })
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        let mut result = self.patches.process(images)?;
        let side = self.position_side;
        let mut start = 0;
        result
            .interpolation_indices
            .reserve_exact(result.position_ids.len());
        result
            .interpolation_weights
            .reserve_exact(result.position_ids.len());
        for geometry in &result.geometry {
            let grid_h = geometry.height / PATCH;
            let grid_w = geometry.width / PATCH;
            for &[row, col] in &result.position_ids[start..start + geometry.patches] {
                let (yi, yw) = interpolation_axis(row as usize, grid_h, side);
                let (xi, xw) = interpolation_axis(col as usize, grid_w, side);
                result.interpolation_indices.push([
                    (yi[0] * side + xi[0]) as i64,
                    (yi[0] * side + xi[1]) as i64,
                    (yi[1] * side + xi[0]) as i64,
                    (yi[1] * side + xi[1]) as i64,
                ]);
                result.interpolation_weights.push([
                    yw[0] * xw[0],
                    yw[0] * xw[1],
                    yw[1] * xw[0],
                    yw[1] * xw[1],
                ]);
            }
            start += geometry.patches;
        }
        Ok(result)
    }

    /// Expands the image placeholders in already rendered chat text, in image order.
    /// Inputs contain no padding; the scheduler adds padding outside logical lengths.
    pub fn expand_prompt(
        &self,
        frontend: &Frontend,
        text: &str,
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        if frontend.encode(IMAGE_TOKEN, false)? != [self.image_token_id] {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image token ID does not match tokenizer",
            ))));
        }
        if text.matches(IMAGE_TOKEN).count() != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image placeholder count does not match prepared images",
            ))));
        }
        let mut expanded = String::with_capacity(text.len());
        let mut parts = text.split(IMAGE_TOKEN);
        expanded.push_str(parts.next().unwrap_or_default());
        for (part, geometry) in parts.zip(&images.geometry) {
            for _ in 0..geometry.tokens {
                expanded.push_str(IMAGE_TOKEN);
            }
            expanded.push_str(part);
        }
        let token_ids = frontend.encode(&expanded, false)?;
        if token_ids.contains(&self.video_token_id) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "video inputs are not supported by the image processor",
            ))));
        }
        let mm_token_type_ids: Vec<i32> = token_ids
            .iter()
            .map(|id| i32::from(*id == self.image_token_id))
            .collect();
        let mut positions: [Vec<i64>; 4] =
            std::array::from_fn(|_| Vec::with_capacity(token_ids.len()));
        positions[0].extend((0..token_ids.len()).map(|index| index as i64));
        let (mut offset, mut current, mut image_index) = (0, 0i64, 0);
        while offset < token_ids.len() {
            let modality = mm_token_type_ids[offset];
            let end = offset
                + mm_token_type_ids[offset..]
                    .iter()
                    .take_while(|value| **value == modality)
                    .count();
            if modality == 0 {
                for row in &mut positions[1..] {
                    row.extend((0..end - offset).map(|index| current + index as i64));
                }
                current += (end - offset) as i64;
            } else {
                let geometry = images.geometry.get(image_index).ok_or_else(|| {
                    Error::frontend(FrontendError::Input(Failure::new(
                        "unexpected image token group",
                    )))
                })?;
                if end - offset != geometry.tokens {
                    return Err(Error::frontend(FrontendError::Input(Failure::new(
                        "image token group does not match image grid",
                    ))));
                }
                let (height, width) = (
                    geometry.grid_thw[1] / MERGE as i64,
                    geometry.grid_thw[2] / MERGE as i64,
                );
                for y in 0..height {
                    for x in 0..width {
                        positions[1].push(current);
                        positions[2].push(current + y);
                        positions[3].push(current + x);
                    }
                }
                current += height.max(width);
                image_index += 1;
            }
            offset = end;
        }
        if image_index != images.geometry.len() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "prepared image was not consumed by prompt",
            ))));
        }
        Ok(ImagePrompt {
            text: expanded,
            rope_delta: current - token_ids.len() as i64,
            token_ids,
            mm_token_type_ids,
            position_ids: positions,
        })
    }
}

fn interpolation_axis(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = index as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
    let floor = source.floor();
    let distance = source - floor;
    (
        [floor as usize, (floor as usize + 1).min(side - 1)],
        [1.0 - distance, 1.0 - (distance - 1.0).abs()],
    )
}
