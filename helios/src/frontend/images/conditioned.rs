//! Per-image prompt descriptions and reconstructible CPU preparation inputs.

use super::{ImageGeometry, ImageProcessor, ProcessedImages, RgbImage};
use crate::Result;
use std::{ops::Range, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageDescription {
    pub identity: [u8; 32],
    pub width: usize,
    pub height: usize,
    pub geometry: ImageGeometry,
}

impl ImageDescription {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self).map_err(|e| e.to_string())?)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes).map_err(|e| e.to_string())?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageEncoding {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug)]
pub(super) struct DecodedBudget {
    pub(super) capacity: u64,
    pub(super) used: std::sync::atomic::AtomicU64,
}
impl DecodedBudget {
    pub(super) fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: std::sync::atomic::AtomicU64::new(0),
        }
    }
    fn reserve(self: &Arc<Self>, bytes: u64) -> Option<DecodedLease> {
        use std::sync::atomic::Ordering::Relaxed;
        self.used
            .fetch_update(Relaxed, Relaxed, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= self.capacity)
            })
            .ok()?;
        Some(DecodedLease {
            budget: self.clone(),
            bytes,
        })
    }
}
#[derive(Debug)]
struct DecodedLease {
    budget: Arc<DecodedBudget>,
    bytes: u64,
}
impl Drop for DecodedLease {
    fn drop(&mut self) {
        self.budget
            .used
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}
#[derive(Debug)]
pub struct RetainedDecoded {
    image: Arc<RgbImage>,
    _lease: Option<DecodedLease>,
}

#[derive(Clone, Debug)]
pub enum ImageSource {
    Decoded(Arc<RgbImage>),
    Encoded {
        encoding: ImageEncoding,
        bytes: Arc<[u8]>,
        decoded: Arc<std::sync::Mutex<Option<RetainedDecoded>>>,
    },
}

impl ImageSource {
    pub fn release_decoded(&self) {
        if let Self::Encoded { decoded, .. } = self {
            decoded.lock().unwrap().take();
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImageInput {
    pub description: ImageDescription,
    pub source: ImageSource,
}

#[derive(Clone)]
pub struct PreparedImageInput {
    pub identity: [u8; 32],
    pub token_span: Range<usize>,
    pub feature_range: Range<usize>,
    pub geometry: ImageGeometry,
    pub source: ImageSource,
    processor: Arc<ImageProcessor>,
}

impl std::fmt::Debug for PreparedImageInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedImageInput")
            .field("identity", &self.identity)
            .field("token_span", &self.token_span)
            .field("feature_range", &self.feature_range)
            .field("geometry", &self.geometry)
            .finish()
    }
}

impl PreparedImageInput {
    pub fn new(
        input: ImageInput,
        token_span: Range<usize>,
        feature_range: Range<usize>,
        processor: Arc<ImageProcessor>,
    ) -> Self {
        Self {
            identity: input.description.identity,
            geometry: input.description.geometry,
            token_span,
            feature_range,
            source: input.source,
            processor,
        }
    }

    #[cfg(feature = "server")]
    pub fn materialize_cached(
        &self,
        store: Option<&crate::offload::input::InputStore>,
    ) -> Result<ProcessedImages> {
        if let Some(record) = store.and_then(|s| s.get_blocking(self.identity, true)) {
            if let Some(bytes) = &record.payload {
                if let Ok(images) = ProcessedImages::decode(bytes) {
                    if self.valid_processed(&images) {
                        self.release_decoded();
                        return Ok(images);
                    }
                }
            }
            if let Some(store) = store {
                store.invalidate(self.identity);
            }
        }
        let images = self.materialize()?;
        if let Some(store) = store {
            let metadata = &b"flare-prepared-image-v1"[..];
            let bytes = images
                .encoded_len()?
                .checked_add(metadata.len())
                .ok_or("prepared image record extent overflows")?;
            store.insert_with(self.identity, bytes as u64, || {
                Ok(crate::offload::input::PreparedInputRecord {
                    metadata: Arc::from(metadata),
                    payload: Some(Arc::new(images.encode()?)),
                })
            })?;
        }
        Ok(images)
    }

    fn valid_processed(&self, images: &ProcessedImages) -> bool {
        use super::Profile;
        let pixel_width = match self.processor.profile {
            Profile::Qwen(_) | Profile::MiMo(_) => 1536,
            Profile::Radio => 768,
            Profile::Glm => 1176,
            _ => 588,
        };
        let patches = self.geometry.patches;
        let interpolation = matches!(
            self.processor.profile,
            Profile::Qwen(_) | Profile::Moonvit | Profile::Step | Profile::Radio
        );
        let sequences = if matches!(self.processor.profile, Profile::Step) {
            let local = self.geometry.grid_thw[1] * self.geometry.grid_thw[2];
            (0..=local)
                .map(|i| (i * 1296) as i32)
                .chain(std::iter::once(patches as i32))
                .collect::<Vec<_>>()
        } else if matches!(
            self.processor.profile,
            Profile::DeepSeek4 | Profile::DeepseekRowMajor(_)
        ) {
            vec![
                0,
                (self.geometry.grid_thw[1] * self.geometry.grid_thw[2]) as i32,
            ]
        } else {
            vec![0, patches as i32]
        };
        images.geometry.as_slice() == [self.geometry]
            && images.pixel_values.len() == patches * pixel_width
            && images.position_ids.len() == patches
            && images.interpolation_indices.len() == if interpolation { patches } else { 0 }
            && images.interpolation_weights.len() == if interpolation { patches } else { 0 }
            && images.cu_seqlens == sequences
    }

    pub fn release_decoded(&self) {
        self.source.release_decoded();
    }

    pub fn materialize(&self) -> Result<ProcessedImages> {
        let image = match &self.source {
            ImageSource::Decoded(image) => RetainedDecoded {
                image: image.clone(),
                _lease: None,
            },
            ImageSource::Encoded {
                encoding,
                bytes,
                decoded,
            } => {
                let retained = decoded.lock().unwrap().take();
                match retained {
                    Some(image) => image,
                    None => RetainedDecoded {
                        image: Arc::new(self.processor.decode_encoded(*encoding, bytes)?),
                        _lease: None,
                    },
                }
            }
        };
        if self.processor.semantic_identity(&image.image) != self.identity {
            return Err("image source differs from its prompt description".into());
        }
        let result = self.processor.process_one(&image.image)?;
        if !self.valid_processed(&result) {
            return Err("image preprocessing differs from its prompt geometry".into());
        }
        Ok(result)
    }
}

impl ImageProcessor {
    pub fn encoded_source(
        &self,
        encoding: ImageEncoding,
        bytes: Arc<[u8]>,
        decoded: Option<RgbImage>,
    ) -> ImageSource {
        let decoded = decoded.and_then(|image| {
            let bytes = image
                .pixels
                .capacity()
                .checked_add(image.alpha.as_ref().map_or(0, Vec::capacity))?
                .checked_add(image.cmyk.as_ref().map_or(0, Vec::capacity))?;
            let lease = self.decoded_budget.reserve(bytes as u64)?;
            Some(RetainedDecoded {
                image: Arc::new(image),
                _lease: Some(lease),
            })
        });
        ImageSource::Encoded {
            encoding,
            bytes,
            decoded: Arc::new(std::sync::Mutex::new(decoded)),
        }
    }

    pub fn describe_image(&self, image: &RgbImage) -> Result<ImageDescription> {
        Ok(ImageDescription {
            identity: self.semantic_identity(image),
            width: image.width(),
            height: image.height(),
            geometry: self.geometry(image.width(), image.height())?,
        })
    }

    pub fn describe_inputs(&self, inputs: &[ImageDescription]) -> Result<ProcessedImages> {
        self.check_batch(inputs.len(), 0)?;
        let geometry = inputs
            .iter()
            .map(|i| self.geometry(i.width, i.height))
            .collect::<Result<Vec<_>>>()?;
        if geometry.iter().zip(inputs).any(|(g, i)| *g != i.geometry) {
            return Err("image description geometry differs from the preparation profile".into());
        }
        self.check_batch(inputs.len(), geometry.iter().map(|g| g.patches).sum())?;
        Ok(ProcessedImages {
            geometry,
            ..Default::default()
        })
    }

    pub fn decode_encoded(&self, encoding: ImageEncoding, bytes: &[u8]) -> Result<RgbImage> {
        let image = match encoding {
            ImageEncoding::Png => self.decode_png(bytes),
            ImageEncoding::Jpeg => self.decode_jpeg(bytes),
            ImageEncoding::Webp => self.decode_webp(bytes),
        }?;
        self.counts
            .decoded
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::images::{
        DeepseekRowMajorImageSettings, ImageLimits, ImageSettings, MimoImageSettings,
        QwenImageSettings,
    };

    fn settings() -> Vec<ImageSettings> {
        vec![
            ImageSettings::Qwen(QwenImageSettings {
                min_pixels: 65536,
                max_pixels: 16777216,
                position_side: 48,
                image_token_id: 248056,
                video_token_id: 248057,
            }),
            ImageSettings::Mimo(MimoImageSettings {
                min_pixels: 65536,
                max_pixels: 16777216,
                image_mean: [0.5; 3],
                image_std: [0.5; 3],
            }),
            ImageSettings::Deepseek4 {},
            ImageSettings::DeepseekRowMajor(DeepseekRowMajorImageSettings {
                min_pixels: 46656,
                max_tokens: 1024,
                max_width_height_ratio: None,
            }),
            ImageSettings::Moonvit {},
            ImageSettings::Step {},
            ImageSettings::Radio {},
        ]
    }

    fn processor(settings: ImageSettings) -> Arc<ImageProcessor> {
        Arc::new(
            ImageProcessor::load(
                settings,
                ImageLimits {
                    max_patches_per_image: 65536,
                    max_total_patches: 65536,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
    }

    fn image(width: usize, height: usize, seed: usize) -> RgbImage {
        RgbImage::new(
            width,
            height,
            (0..width * height * 3)
                .map(|i| ((i * 13 + seed) % 256) as u8)
                .collect(),
        )
        .unwrap()
    }

    fn input(
        processor: Arc<ImageProcessor>,
        source: ImageSource,
        description: ImageDescription,
    ) -> PreparedImageInput {
        let tokens = description.geometry.tokens;
        PreparedImageInput::new(
            ImageInput {
                description,
                source,
            },
            3..tokens + 3,
            0..tokens,
            processor,
        )
    }

    fn png(image: &RgbImage) -> Arc<[u8]> {
        let mut bytes = vec![];
        {
            let mut encoder =
                png::Encoder::new(&mut bytes, image.width() as u32, image.height() as u32);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(image.pixels())
                .unwrap();
        }
        bytes.into()
    }

    #[test]
    fn independent_preparation_matches_batches_for_every_profile() {
        for settings in settings() {
            let processor = processor(settings.clone());
            let images = [image(192, 64, 7), image(37, 29, 51)];
            let descriptions = images
                .iter()
                .map(|i| processor.describe_image(i).unwrap())
                .collect::<Vec<_>>();
            let description = processor.describe_inputs(&descriptions).unwrap();
            assert!(description.pixel_values.is_empty(), "{settings:?}");
            assert!(description.position_ids.is_empty(), "{settings:?}");
            assert_eq!(processor.preparation_counts().prepared_images, 0);
            let expected = processor.process(&images).unwrap();
            let mut actual = ProcessedImages::default();
            for (source, description) in images.into_iter().zip(descriptions) {
                assert_eq!(
                    ImageDescription::decode(&description.encode().unwrap()).unwrap(),
                    description
                );
                let prepared = input(
                    processor.clone(),
                    ImageSource::Decoded(Arc::new(source)),
                    description,
                );
                let processed = prepared.materialize().unwrap();
                assert_eq!(
                    processed.pixel_values.capacity(),
                    processed.pixel_values.len()
                );
                assert_eq!(
                    processed.position_ids.capacity(),
                    processed.position_ids.len()
                );
                assert_eq!(
                    processed.interpolation_indices.capacity(),
                    processed.interpolation_indices.len()
                );
                assert_eq!(
                    processed.interpolation_weights.capacity(),
                    processed.interpolation_weights.len()
                );
                assert_eq!(processed.cu_seqlens.capacity(), processed.cu_seqlens.len());
                let encoded = processed.encode().unwrap();
                assert_eq!(encoded.len(), processed.encoded_len().unwrap());
                assert_eq!(encoded.capacity(), encoded.len());
                let decoded = ProcessedImages::decode(&encoded).unwrap();
                assert_eq!(decoded.geometry.capacity(), decoded.geometry.len());
                assert_eq!(decoded.pixel_values.capacity(), decoded.pixel_values.len());
                assert_eq!(decoded.position_ids.capacity(), decoded.position_ids.len());
                assert_eq!(
                    decoded.interpolation_indices.capacity(),
                    decoded.interpolation_indices.len()
                );
                assert_eq!(
                    decoded.interpolation_weights.capacity(),
                    decoded.interpolation_weights.len()
                );
                assert_eq!(decoded.cu_seqlens.capacity(), decoded.cu_seqlens.len());
                assert_eq!(decoded.encode().unwrap(), encoded, "{settings:?}");
                actual.append(processed);
            }
            assert_eq!(actual.geometry, expected.geometry, "{settings:?}");
            assert_eq!(actual.pixel_values, expected.pixel_values, "{settings:?}");
            assert_eq!(actual.position_ids, expected.position_ids, "{settings:?}");
            assert_eq!(
                actual.interpolation_indices, expected.interpolation_indices,
                "{settings:?}"
            );
            assert_eq!(
                actual.interpolation_weights, expected.interpolation_weights,
                "{settings:?}"
            );
            assert_eq!(actual.cu_seqlens, expected.cu_seqlens, "{settings:?}");
            assert_eq!(processor.preparation_counts().prepared_images, 2);
        }
    }

    #[test]
    fn semantic_identity_includes_profile_and_consumed_moonvit_channels() {
        let source = image(37, 29, 7);
        let ids = settings()
            .into_iter()
            .map(|s| processor(s).semantic_identity(&source))
            .collect::<Vec<_>>();
        for (index, id) in ids.iter().enumerate() {
            assert!(!ids[..index].contains(id));
        }
        let moonvit = processor(ImageSettings::Moonvit {});
        let original = moonvit.semantic_identity(&source);
        let mut altered = source.clone();
        altered.alpha = Some(vec![255; 37 * 29]);
        let alpha = moonvit.semantic_identity(&altered);
        assert_ne!(original, alpha);
        altered.alpha.as_mut().unwrap()[0] = 1;
        assert_ne!(alpha, moonvit.semantic_identity(&altered));
        altered = source.clone();
        altered.cmyk = Some(vec![127; 37 * 29 * 4]);
        let cmyk = moonvit.semantic_identity(&altered);
        assert_ne!(original, cmyk);
        altered.cmyk.as_mut().unwrap()[0] = 1;
        assert_ne!(cmyk, moonvit.semantic_identity(&altered));
        altered = source.clone();
        altered.nearest_resize = true;
        assert_ne!(original, moonvit.semantic_identity(&altered));
        let qwen = processor(settings().remove(0));
        assert_eq!(
            qwen.semantic_identity(&source),
            qwen.semantic_identity(&altered)
        );
        let mut config = settings().remove(1);
        let ImageSettings::Mimo(ref mut settings) = config else {
            unreachable!()
        };
        settings.image_mean[0] = 0.25;
        assert_ne!(ids[1], processor(config).semantic_identity(&source));
    }

    #[test]
    fn encoded_source_reconstructs_and_media_type_has_separate_lookup_identity() {
        let processor = processor(ImageSettings::Moonvit {});
        let source = image(37, 29, 7);
        let encoded = png(&source);
        let decoded = processor
            .decode_encoded(ImageEncoding::Png, &encoded)
            .unwrap();
        let description = processor.describe_image(&decoded).unwrap();
        let prepared = input(
            processor.clone(),
            processor.encoded_source(ImageEncoding::Png, encoded.clone(), None),
            description,
        );
        let expected = processor.process_one(&decoded).unwrap();
        assert_eq!(
            prepared.materialize().unwrap().encode().unwrap(),
            expected.encode().unwrap()
        );
        assert_eq!(processor.preparation_counts().decoded_images, 2);
        assert_ne!(
            processor.source_identity(ImageEncoding::Png, &encoded),
            processor.source_identity(ImageEncoding::Jpeg, &encoded)
        );
        assert_ne!(
            processor.source_identity(ImageEncoding::Png, &encoded),
            prepared.identity
        );
        assert!(
            processor
                .decode_encoded(ImageEncoding::Jpeg, &encoded)
                .is_err()
        );
    }

    #[test]
    fn initial_decoded_source_is_consumed_once_and_prefix_coverage_releases_it() {
        let processor = processor(ImageSettings::Moonvit {});
        let source = image(37, 29, 7);
        let bytes = png(&source);
        let description = processor.describe_image(&source).unwrap();
        let prepared = input(
            processor.clone(),
            processor.encoded_source(ImageEncoding::Png, bytes.clone(), Some(source)),
            description.clone(),
        );
        let first = prepared.materialize().unwrap();
        assert_eq!(processor.preparation_counts().decoded_images, 0);
        assert_eq!(
            prepared.materialize().unwrap().encode().unwrap(),
            first.encode().unwrap()
        );
        assert_eq!(processor.preparation_counts().decoded_images, 1);
        let source = image(37, 29, 7);
        let prepared = input(
            processor.clone(),
            processor.encoded_source(ImageEncoding::Png, bytes, Some(source)),
            description,
        );
        prepared.release_decoded();
        assert_eq!(
            prepared.materialize().unwrap().encode().unwrap(),
            first.encode().unwrap()
        );
        assert_eq!(processor.preparation_counts().decoded_images, 2);
    }

    #[test]
    fn decoded_source_retention_is_shared_bounded_and_tracks_live_processing_leases() {
        let processor = Arc::new(
            ImageProcessor::load(
                ImageSettings::Moonvit {},
                ImageLimits {
                    max_input_pixels: 16,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let clone = processor.clone();
        let mut sources = (0..8)
            .map(|seed| {
                let decoded = image(2, 2, seed);
                processor.encoded_source(ImageEncoding::Png, png(&decoded), Some(decoded))
            })
            .collect::<Vec<_>>();
        assert_eq!(clone.preparation_counts().decoded_source_capacity, 128);
        assert_eq!(clone.preparation_counts().decoded_source_bytes, 96);
        let mut oversized = Vec::with_capacity(128);
        oversized.extend_from_slice(&[7; 12]);
        let oversized = RgbImage::new(2, 2, oversized).unwrap();
        let skipped =
            processor.encoded_source(ImageEncoding::Png, png(&oversized), Some(oversized));
        let ImageSource::Encoded { decoded, .. } = &skipped else {
            unreachable!()
        };
        assert!(decoded.lock().unwrap().is_none());
        let selected = sources.pop().unwrap();
        let ImageSource::Encoded { decoded, .. } = &selected else {
            unreachable!()
        };
        let in_use = decoded.lock().unwrap().take().unwrap();
        selected.release_decoded();
        assert_eq!(clone.preparation_counts().decoded_source_bytes, 96);
        drop(sources);
        assert_eq!(clone.preparation_counts().decoded_source_bytes, 12);
        drop(in_use);
        assert_eq!(clone.preparation_counts().decoded_source_bytes, 0);
    }

    #[test]
    fn eight_small_sources_use_one_decode_each_within_shared_budget() {
        let processor = Arc::new(
            ImageProcessor::load(
                ImageSettings::Moonvit {},
                ImageLimits {
                    max_input_pixels: 16,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let inputs = (0..8)
            .map(|seed| {
                let bytes = png(&image(2, 2, seed));
                let decoded = processor
                    .decode_encoded(ImageEncoding::Png, &bytes)
                    .unwrap();
                let description = processor.describe_image(&decoded).unwrap();
                input(
                    processor.clone(),
                    processor.encoded_source(ImageEncoding::Png, bytes, Some(decoded)),
                    description,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(processor.preparation_counts().decoded_source_bytes, 96);
        for image in inputs {
            image.materialize().unwrap();
        }
        assert_eq!(processor.preparation_counts().decoded_images, 8);
        assert_eq!(processor.preparation_counts().prepared_images, 8);
        assert_eq!(processor.preparation_counts().decoded_source_bytes, 0);
    }

    #[cfg(feature = "server")]
    #[test]
    fn processed_cache_hit_skips_decode_and_patch_preparation() {
        let store = crate::offload::input::InputStore::start(
            64 << 20,
            None,
            crate::offload::input::DiskBudget::new(0, 0),
        )
        .unwrap();
        let processor = processor(ImageSettings::Step {});
        let source = image(192, 64, 7);
        let description = processor.describe_image(&source).unwrap();
        let prepared = input(
            processor.clone(),
            processor.encoded_source(ImageEncoding::Png, png(&source), None),
            description,
        );
        let before = processor.preparation_counts();
        let first = prepared.materialize_cached(Some(&store)).unwrap();
        let cached = processor.preparation_counts();
        assert_eq!(cached.decoded_images, before.decoded_images + 1);
        assert_eq!(cached.prepared_images, before.prepared_images + 1);
        let second = prepared.materialize_cached(Some(&store)).unwrap();
        assert_eq!(second.encode().unwrap(), first.encode().unwrap());
        assert_eq!(processor.preparation_counts(), cached);
        assert_eq!(store.stats().hits, 1);
    }

    #[cfg(feature = "server")]
    #[test]
    fn cached_tensor_extents_are_validated_before_use() {
        let store = crate::offload::input::InputStore::start(
            64 << 20,
            None,
            crate::offload::input::DiskBudget::new(0, 0),
        )
        .unwrap();
        let processor = processor(ImageSettings::Moonvit {});
        let source = image(37, 29, 7);
        let description = processor.describe_image(&source).unwrap();
        let prepared = input(
            processor.clone(),
            processor.encoded_source(ImageEncoding::Png, png(&source), None),
            description,
        );
        let mut malformed = processor.process(&[source]).unwrap();
        malformed.position_ids.pop();
        store.insert(
            prepared.identity,
            Arc::new(crate::offload::input::PreparedInputRecord {
                metadata: Arc::from(&b"flare-prepared-image-v1"[..]),
                payload: Some(Arc::new(malformed.encode().unwrap())),
            }),
        );
        let result = prepared.materialize_cached(Some(&store)).unwrap();
        assert_eq!(result.position_ids.len(), prepared.geometry.patches);
        assert_eq!(processor.preparation_counts().decoded_images, 1);
        assert_eq!(processor.preparation_counts().prepared_images, 1);
        let repaired = prepared.materialize_cached(Some(&store)).unwrap();
        assert_eq!(repaired.encode().unwrap(), result.encode().unwrap());
        assert_eq!(processor.preparation_counts().decoded_images, 1);
        assert_eq!(processor.preparation_counts().prepared_images, 1);
    }

    #[test]
    fn malformed_binary_records_cannot_allocate_from_unbounded_counts() {
        let mut bytes = b"FLAREIMG\x01".to_vec();
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(ProcessedImages::decode(&bytes).is_err());
        let valid = ProcessedImages::default().encode().unwrap();
        for end in 0..valid.len() {
            assert!(ProcessedImages::decode(&valid[..end]).is_err());
        }
        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(ProcessedImages::decode(&trailing).is_err());
    }
}
