//! Area-bounded resizing and spatially grouped RGB patch packing.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum Normalization {
    Symmetric,
    Channel { mean: [f32; 3], std: [f32; 3] },
    ScaledChannel { mean: [f32; 3], std: [f32; 3] },
}

#[derive(Clone)]
pub(super) struct PatchProcessor {
    pub(super) limits: ImageLimits,
    pub(super) patch_size: usize,
    pub(super) min_pixels: usize,
    pub(super) max_pixels: usize,
    pub(super) normalization: Normalization,
    pub(super) resize: fn(&RgbImage, usize, usize) -> Vec<u8>,
}

impl PatchProcessor {
    pub fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        if width.max(height) as f64 / width.min(height) as f64 > 200.0 {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "image aspect ratio exceeds 200",
            ))));
        }
        let factor = (self.patch_size * MERGE) as f64;
        let (h, w) = (height as f64, width as f64);
        let mut target_h = ((h / factor).round_ties_even() * factor).max(factor);
        let mut target_w = ((w / factor).round_ties_even() * factor).max(factor);
        if target_h * target_w > self.max_pixels as f64 {
            let beta = (h * w / self.max_pixels as f64).sqrt();
            target_h = ((h / beta / factor).floor() * factor).max(factor);
            target_w = ((w / beta / factor).floor() * factor).max(factor);
        } else if target_h * target_w < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (h * w)).sqrt();
            target_h = (h * beta / factor).ceil() * factor;
            target_w = (w * beta / factor).ceil() * factor;
        }
        let (height, width) = (target_h as usize, target_w as usize);
        let patches = height / self.patch_size * (width / self.patch_size);
        check_image_patches(self.limits, patches)?;
        Ok(ImageGeometry {
            height,
            width,
            grid_thw: [
                1,
                (height / self.patch_size) as i64,
                (width / self.patch_size) as i64,
            ],
            patches,
            tokens: patches / (MERGE * MERGE),
        })
    }

    pub fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        let geometry = self.geometry(image.width, image.height)?;
        Ok((
            geometry,
            (self.resize)(image, geometry.width, geometry.height),
        ))
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        self.limits.check_batch(images.len(), 0)?;
        let geometry: Vec<_> = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<_>>()?;
        let total: usize = geometry
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
            .ok_or_else(|| {
                Error::frontend(FrontendError::Input(Failure::new("patch count overflow")))
            })?;
        self.limits.check_batch(images.len(), total)?;
        let mut result = ProcessedImages {
            pixel_values: Vec::with_capacity(
                total
                    .checked_mul(3 * 2 * self.patch_size * self.patch_size)
                    .ok_or(Error::frontend(FrontendError::Input(Failure::new(
                        "patch buffer extent overflow",
                    ))))?,
            ),
            geometry,
            position_ids: Vec::with_capacity(total),
            interpolation_indices: Vec::new(),
            interpolation_weights: Vec::new(),
            cu_seqlens: vec![0],
        };
        result.reserve_outputs(total, FEATURES, false, images.len(), images.len() + 1)?;
        let normalized: [Vec<f32>; 3] = std::array::from_fn(|channel| {
            (0..256)
                .map(|value| match self.normalization {
                    Normalization::Symmetric => (value as f32 - 127.5) / 127.5,
                    Normalization::ScaledChannel { mean, std } => {
                        (value as f32 - mean[channel] * 255.) / (std[channel] * 255.)
                    }
                    Normalization::Channel { mean, std } => {
                        (value as f32 * (1.0 / 255.0) - mean[channel]) / std[channel]
                    }
                })
                .collect()
        });
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let pixels = (self.resize)(image, geometry.width, geometry.height);
            let grid_h = geometry.height / self.patch_size;
            let grid_w = geometry.width / self.patch_size;
            for by in 0..grid_h / MERGE {
                for bx in 0..grid_w / MERGE {
                    for dy in 0..MERGE {
                        for dx in 0..MERGE {
                            let (row, col) = (by * MERGE + dy, bx * MERGE + dx);
                            result.position_ids.push([row as i64, col as i64]);
                            for channel in 0..3 {
                                for _temporal in 0..2 {
                                    for y in 0..self.patch_size {
                                        for x in 0..self.patch_size {
                                            let byte = pixels[((row * self.patch_size + y)
                                                * geometry.width
                                                + col * self.patch_size
                                                + x)
                                                * 3
                                                + channel];
                                            result
                                                .pixel_values
                                                .push(normalized[channel][byte as usize]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            result
                .cu_seqlens
                .push(result.cu_seqlens.last().unwrap() + geometry.patches as i32);
        }
        Ok(result)
    }
}
