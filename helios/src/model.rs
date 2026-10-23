//! Shared execution of prepared storage and graph-input contracts.

pub mod admission;
pub mod catalog;
pub mod contract;
pub(crate) mod inputs;
mod packing;
#[cfg(feature = "server")]
mod prompt;

use crate::{
    Result,
    execution::{CompletionRecord, GraphKind},
    scheduler::{SchedulerCapabilities, SchedulerLimits},
};
pub use admission::Admission;
pub use contract::PreparedContract;
pub use inputs::{InputStep, pack_inputs};
pub use packing::Inputs;
#[cfg(feature = "server")]
pub use prompt::{prepare_image_inputs, prepare_input, prepare_output};
#[cfg(feature = "server")]
pub(crate) use prompt::validate_conditioned_inputs;

#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionCapabilities {
    pub feature_slots: usize,
    pub features_per_request: usize,
    /// Serialized image-support declaration; image counts are bounded by prompt context.
    pub max_images: usize,
    pub patches_per_feature: usize,
    pub patch_capacities: Vec<usize>,
}

impl VisionCapabilities {
    pub fn disabled() -> Self {
        Self {
            feature_slots: 0,
            features_per_request: 0,
            max_images: 0,
            patches_per_feature: 0,
            patch_capacities: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_images > 0
    }

    pub fn validate(&self) -> Result<()> {
        if self == &Self::disabled() {
            return Ok(());
        }
        if self.feature_slots == 0
            || self.feature_slots > crate::execution::REQUEST_LANES
            || self.patches_per_feature == 0
            || self.max_images == 0
            || self.features_per_request == 0
            || self.patch_capacities.is_empty()
            || self
                .patch_capacities
                .iter()
                .any(|n| *n == 0 || !n.is_multiple_of(self.patches_per_feature))
            || self.patch_capacities.windows(2).any(|p| p[0] >= p[1])
        {
            return Err("invalid prepared vision capabilities".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub max_top_logprobs: usize,
    pub logprob_token_id_capacity: usize,
    /// Context extent of captured page tables and workspaces; serving may use a smaller limit.
    pub prepared_sequence_tokens: usize,
    pub contract: PreparedContract,
    pub identity: [u8; 32],
    pub model_id: String,
    pub limits: SchedulerLimits,
    pub eos_token_ids: Vec<u32>,
    pub sampling_defaults: SamplingParameters,
    pub sampling_ranges: SamplingRanges,
    pub max_new_tokens: u32,
    pub speculative_widths: Vec<usize>,
    pub speculative_fixed_width: Option<usize>,
    /// Maximum target-token rows in wider speculative graphs.
    pub speculative_token_capacity: usize,
    /// Bytes of weights one decode step reads for one request, as the package declares; zero
    /// when it declares none.
    pub decode_weight_bytes: u64,
    pub history_drafts: bool,
    pub speculative_proposal_source: crate::scheduler::speculation::ProposalSource,
}

#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "server", serde(deny_unknown_fields))]
#[derive(Clone, Debug)]
pub struct SamplingParameters {
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub top_k: i32,
}

impl Default for SamplingParameters {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 1.,
            top_p: 0.95,
            min_p: 0.,
            repetition_penalty: 1.,
            presence_penalty: 0.,
            frequency_penalty: 0.,
            top_k: 20,
        }
    }
}

#[cfg_attr(
    any(feature = "server", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "server", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingRange<T> {
    pub min: T,
    pub max: T,
}

impl<T: Copy> SamplingRange<T> {
    pub fn fixed(value: T) -> Self {
        Self {
            min: value,
            max: value,
        }
    }
}

#[cfg_attr(
    any(feature = "server", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "server", feature = "runtime-package"),
    serde(default, deny_unknown_fields)
)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingRanges {
    pub temperature: Option<SamplingRange<f32>>,
    pub top_k: Option<SamplingRange<i32>>,
    pub top_p: Option<SamplingRange<f32>>,
    pub min_p: Option<SamplingRange<f32>>,
    pub repetition_penalty: Option<SamplingRange<f32>>,
    pub presence_penalty: Option<SamplingRange<f32>>,
    pub frequency_penalty: Option<SamplingRange<f32>>,
}

impl SamplingRanges {
    pub fn validate(&self, defaults: &SamplingParameters) -> Result<()> {
        defaults.validate()?;
        let mut min = defaults.clone();
        let mut max = defaults.clone();
        macro_rules! bounds {
            ($($field:ident),*) => { $(
                if let Some(range) = self.$field {
                    if !(range.min <= defaults.$field && defaults.$field <= range.max) {
                        return Err(concat!("sampling range must contain the default: ", stringify!($field)).into());
                    }
                    min.$field = range.min;
                    max.$field = range.max;
                }
            )* };
        }
        bounds!(
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty
        );
        min.validate()?;
        max.validate()?;
        Ok(())
    }
}

impl SamplingParameters {
    pub(crate) fn values(&self) -> [f32; 6] {
        [
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        ]
    }
    pub fn validate(&self) -> Result<()> {
        if self.temperature == 0. {
            return Err(crate::Error::admission(
                crate::AdmissionError::Invalid,
                "temperature=0 is unsupported because greedy decoding incorrectly collapses the model’s probability distribution, and is not appropriate for most use cases",
            ));
        }
        if !self.temperature.is_finite() || self.temperature < f32::MIN_POSITIVE {
            return Err(crate::Error::admission(
                crate::AdmissionError::Invalid,
                "temperature must be a positive normal FP32 value",
            ));
        }
        if self.values().iter().any(|value| !value.is_finite())
            || !(0. < self.top_p && self.top_p <= 1.)
            || !(0. ..=1.).contains(&self.min_p)
            || self.repetition_penalty <= 0.
            || [self.temperature, self.top_p, self.repetition_penalty]
                .iter()
                .any(|value| *value > 0. && *value < f32::MIN_POSITIVE)
            || self.top_k < -1
        {
            return Err(crate::Error::admission(
                crate::AdmissionError::Invalid,
                "invalid sampling parameters",
            ));
        }
        Ok(())
    }
}

impl ModelConfig {
    pub fn fit_context(&mut self) {
        let capabilities = scheduler_capabilities(self);
        let draft_pages = if self.has_learned_drafts() {
            capabilities.draft_pages_per_request
        } else {
            0
        };
        self.limits.max_sequence_tokens = self.prepared_sequence_tokens.min(
            (self.limits.pages as usize)
                .saturating_sub(draft_pages)
                .saturating_mul(capabilities.page_tokens),
        );
    }

    pub fn has_learned_drafts(&self) -> bool {
        !self.speculative_widths.is_empty()
            && self.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Mtp
    }
}

#[derive(Clone, Debug, Default)]
pub struct PromptData {
    pub positions: Vec<[i64; 3]>,
    pub vision_token_indices: Vec<i32>,
    pub spans: Vec<std::ops::Range<usize>>,
}

impl PromptData {
    pub fn is_image_position(&self, position: usize) -> bool {
        self.vision_token_indices.get(position).is_some_and(|index| *index >= 0)
            || self.spans.iter().any(|span| span.contains(&position))
    }
}

#[cfg(feature = "server")]
pub struct PreparedInput {
    pub tokens: Vec<u32>,
    /// Original placeholder token index and the additional tokens in its prepared expansion.
    pub token_expansions: Vec<(usize, usize)>,
    pub data: PromptData,
    pub conditioning: Vec<crate::cache::ConditioningEvent>,
    pub conditioned_inputs: Vec<crate::frontend::images::PreparedImageInput>,
}

pub fn scheduler_capabilities(config: &ModelConfig) -> SchedulerCapabilities {
    use contract::{GraphOperation, ValueSource};
    SchedulerCapabilities {
        vocabulary: config.contract.vocabulary,
        page_tokens: config.contract.cache.page_tokens,
        vision: config.contract.vision.clone(),
        draft_pages_per_request: config
            .contract
            .cache
            .page_tables
            .iter()
            .filter(|view| {
                config
                    .contract
                    .graphs
                    .iter()
                    .any(|g| g.inputs.iter().any(|b| b.buffer == view.buffer))
            })
            .map(|view| view.page_offset)
            .max()
            .unwrap_or(0),
        completion_inputs: config
            .contract
            .graphs
            .iter()
            .filter(|g| g.kind == GraphOperation::Decode)
            .flat_map(|g| &g.inputs)
            .filter(|b| {
                matches!(
                    b.source,
                    ValueSource::PendingToken | ValueSource::PromptAndPendingToken
                )
            })
            .map(|b| b.buffer.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect(),
        state_rows_per_request: Some(
            config
                .contract
                .graphs
                .iter()
                .map(|g| g.state_rows_per_request)
                .max()
                .unwrap_or(1),
        ),
    }
}

pub fn validate_completion(
    config: &ModelConfig,
    contract: &contract::GraphContract,
    kind: GraphKind,
    record: &CompletionRecord,
) -> Result<()> {
    if ((record.status == 1 || record.status == 2)
        && record.token_id >= config.contract.vocabulary as i64)
        || record.vector.is_some_and(|v| {
            v.model_reserved != -1
                || v.tokens
                    .iter()
                    .any(|id| *id >= config.contract.vocabulary as i64)
                || (matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative)
                    && record.status != 3
                    && v.draft_cache_consumed
                        != contract.draft_cursor.as_ref().map_or(-1, |cursor| {
                            record
                                .consumed_tokens
                                .saturating_sub(cursor.lag_tokens as i64)
                                .max(0)
                        }))
        })
    {
        return Err(crate::Error::execution(
            crate::ExecutionError::Completion,
            "completion violates its prepared token or history contract",
        ));
    }
    Ok(())
}

pub fn validate_prompt_data(config: &ModelConfig, tokens: &[u32], data: &PromptData) -> Result<()> {
    let positions = config.contract.frontend.position_dimensions;
    if (positions == 0 && !data.positions.is_empty())
        || (positions == 3
            && (data.positions.len() != tokens.len()
                || data.positions.iter().flatten().any(|p| *p < 0)))
        || (!data.vision_token_indices.is_empty()
            && (data.vision_token_indices.len() != tokens.len()
                || data
                    .vision_token_indices
                    .iter()
                    .any(|i| *i < -1 || (*i >= 0 && (!config.contract.vision.enabled() || *i as usize >= tokens.len())))))
        || data.spans.windows(2).any(|p| p[0].end > p[1].start)
        || data.spans.iter().any(|s| {
            s.start >= s.end || s.end > tokens.len() || s.len() > config.limits.prefill_chunk_size
        })
    {
        return Err("prompt arrays violate prepared geometry".into());
    }
    if let Some(layout) = &config.contract.frontend.image_tokens {
        if data.vision_token_indices.len() != tokens.len()
            || tokens
                .iter()
                .zip(&data.vision_token_indices)
                .any(|(token, index)| (*token == layout.feature_token) != (*index >= 0))
        {
            return Err("image feature tokens do not match their feature indices".into());
        }
        if let Some(span) = &layout.span {
            if data.spans.iter().any(|s| {
                s.len() > span.max_tokens
                    || tokens[s.start] != span.start_token
                    || tokens[s.end - 1] != span.end_token
            }) || tokens.iter().enumerate().any(|(position, token)| {
                let scoped = *token == layout.feature_token
                    || *token == span.start_token
                    || *token == span.end_token
                    || span.interior_tokens.contains(token);
                scoped
                    && !span.unscoped_tokens.contains(token)
                    && !data.spans.iter().any(|s| s.contains(&position))
            }) {
                return Err("image token spans do not match their declared boundaries".into());
            }
        } else if !data.spans.is_empty() {
            return Err("image layout does not declare atomic spans".into());
        }
    }
    Ok(())
}

#[cfg(feature = "server")]
pub fn prompt_metadata(data: &mut PromptData) -> admission::PromptMetadataMut<'_> {
    admission::PromptMetadataMut {
        positions: (!data.positions.is_empty()).then_some(&mut data.positions),
        vision_token_indices: (!data.vision_token_indices.is_empty())
            .then_some(&mut data.vision_token_indices),
    }
}

#[cfg(test)]
pub(crate) mod testing;

#[cfg(test)]
mod tests;
