//! GLM5Next aligned canvas resize, black padding, and CLIP normalization profile.

use super::{ImageGeometry, ImageLimits, ProcessedImages, RgbImage, resize as bicubic};
use crate::Result;

pub(super) const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub(super) const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const FACTOR: usize = 28;
const MIN_PIXELS: usize = 16 * FACTOR * FACTOR;
const MAX_PIXELS: usize = 8000 * FACTOR * FACTOR;

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    super::check_input_extent(limits, width, height)?;
    let align = |value: usize| value.div_ceil(FACTOR) * FACTOR;
    let (mut h, mut w) = (align(height), align(width));
    if h * w < MIN_PIXELS {
        let scale = (MIN_PIXELS as f64 / (height * width) as f64).sqrt();
        h = align((height as f64 * scale).ceil().max(1.) as usize);
        w = align((width as f64 * scale).ceil().max(1.) as usize);
    }
    if h * w > MAX_PIXELS {
        let (mut low, mut high) = (1, height);
        (h, w) = (FACTOR, FACTOR);
        while low <= high {
            let content_h = (low + high) / 2;
            let content_w = (width as f64 * content_h as f64 / height as f64)
                .floor()
                .max(1.) as usize;
            let (candidate_h, candidate_w) = (align(content_h), align(content_w));
            if candidate_h * candidate_w <= MAX_PIXELS {
                (h, w) = (candidate_h, candidate_w);
                low = content_h + 1;
            } else {
                high = content_h - 1;
            }
        }
    }
    let patches = h / 14 * (w / 14);
    super::check_image_patches(limits, patches)?;
    Ok(ImageGeometry {
        height: h,
        width: w,
        grid_thw: [1, (h / 14) as i64, (w / 14) as i64],
        patches,
        tokens: patches / 4,
    })
}

pub(super) fn resize(image: &RgbImage, geometry: ImageGeometry) -> Vec<u8> {
    let mut scale = (geometry.height as f64 / image.height as f64)
        .min(geometry.width as f64 / image.width as f64);
    if image.height * image.width >= MIN_PIXELS {
        scale = scale.min(1.);
    }
    let height = (image.height as f64 * scale)
        .floor()
        .max(1.)
        .min(geometry.height as f64) as usize;
    let width = (image.width as f64 * scale)
        .floor()
        .max(1.)
        .min(geometry.width as f64) as usize;
    let pixels = bicubic::resize(image, width, height);
    let mut padded = vec![0; geometry.height * geometry.width * 3];
    for row in 0..height {
        padded[row * geometry.width * 3..(row * geometry.width + width) * 3]
            .copy_from_slice(&pixels[row * width * 3..(row + 1) * width * 3]);
    }
    padded
}

pub(super) fn resized_rgb(
    limits: ImageLimits,
    image: &RgbImage,
) -> Result<(ImageGeometry, Vec<u8>)> {
    let geometry = geometry(limits, image.width, image.height)?;
    Ok((geometry, resize(image, geometry)))
}

pub(super) fn process(limits: ImageLimits, images: &[RgbImage]) -> Result<ProcessedImages> {
    limits.check_batch(images.len(), 0)?;
    let geometry = images
        .iter()
        .map(|image| geometry(limits, image.width, image.height))
        .collect::<Result<Vec<_>>>()?;
    let patches = geometry
        .iter()
        .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
        .ok_or("image patch count overflows")?;
    limits.check_batch(images.len(), patches)?;
    let mut result = ProcessedImages {
        geometry,
        cu_seqlens: vec![0],
        ..Default::default()
    };
    result.reserve_outputs(patches, 1176, false, images.len(), images.len() + 1)?;
    let normalized: [[f32; 256]; 3] = std::array::from_fn(|channel| {
        std::array::from_fn(|value| (value as f32 - MEAN[channel] * 255.) / (STD[channel] * 255.))
    });
    for (image, geometry) in images.iter().zip(&result.geometry) {
        let pixels = resize(image, *geometry);
        for by in 0..geometry.height / 28 {
            for bx in 0..geometry.width / 28 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (row, col) = (by * 2 + dy, bx * 2 + dx);
                        result.position_ids.push([row as i64, col as i64]);
                        for channel in 0..3 {
                            for _temporal in 0..2 {
                                for y in 0..14 {
                                    for x in 0..14 {
                                        let byte = pixels[((row * 14 + y) * geometry.width
                                            + col * 14
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::images::{ImageProcessor, ImageSettings};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    #[test]
    fn canonical_processor_pixels_positions_and_padding() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../../../tests/frontend/glm53_images.json"))
                .unwrap();
        let processor = ImageProcessor::load(
            ImageSettings::Glm {},
            ImageLimits {
                max_patches_per_image: 32768,
                max_total_patches: 32768,
                patches_per_feature: 4,
                ..ImageLimits::default()
            },
        )
        .unwrap();
        let hash = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        for case in fixture["cases"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as usize;
            let height = case["height"].as_u64().unwrap() as usize;
            let seed = case["seed"].as_u64().unwrap() as usize;
            let pixels = (0..height)
                .flat_map(|y| {
                    (0..width).flat_map(move |x| {
                        [(x * 7 + seed * 31) as u8, (y * 11) as u8, (x + y * 3) as u8]
                    })
                })
                .collect();
            let image = RgbImage::new(width, height, pixels).unwrap();
            let (geometry, resized) = processor.resized_rgb(&image).unwrap();
            assert_eq!(json!(geometry.grid_thw), case["grid"], "{width}x{height}");
            assert_eq!(hash(&resized), case["resized"], "resize {width}x{height}");
            let processed = processor.process(&[image]).unwrap();
            let pixels: Vec<_> = processed
                .pixel_values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(hash(&pixels), case["pixels"], "pixels {width}x{height}");
            let positions: Vec<_> = processed
                .position_ids
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                hash(&positions),
                case["positions"],
                "positions {width}x{height}"
            );
            assert_eq!(processed.cu_seqlens, [0, geometry.patches as i32]);
            assert!(processed.interpolation_indices.is_empty());
        }
    }
}
