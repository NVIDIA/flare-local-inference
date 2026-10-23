//! Request-local speculative acceptance and copy proposal metadata.

use crate::{Result, execution::BATCH_CAPACITIES};

pub const WIDTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProposalSource {
    #[default]
    Mtp,
    Copy,
}

impl ProposalSource {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "mtp" => Ok(Self::Mtp),
            "copy" => Ok(Self::Copy),
            _ => Err("speculative_proposal_source must be mtp or copy".into()),
        }
    }
}

/// Host-validated continuation and masks for the corresponding committed prefix.
#[derive(Clone, Debug)]
pub struct CopyProposal {
    pub consumed: u64,
    pub tokens: [i64; 7],
    pub count: usize,
    pub grammar_masks: Vec<i32>,
}

pub fn graph_width_masks(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [u8; 7] {
    let mask = fixed.map_or_else(
        || {
            widths
                .iter()
                .fold(0, |mask, width| mask | (1 << (width - 1)))
        },
        |width| 1 << (width - 1),
    );
    BATCH_CAPACITIES.map(|batch| if batch <= capacity { mask } else { 0 })
}

pub fn graph_catalog(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [Vec<usize>; 7] {
    graph_width_masks(widths, fixed, capacity)
        .map(|mask| (1..=8).filter(|w| mask & (1 << (w - 1)) != 0).collect())
}

pub fn validate_widths(widths: &[usize]) -> Result<()> {
    if !widths.is_empty()
        && (widths[0] != 1
            || widths.windows(2).any(|w| w[0] >= w[1])
            || widths.iter().any(|w| !WIDTHS.contains(w)))
    {
        return Err(
            "speculative widths must be sorted unique integers in 1..=8 including 1".into(),
        );
    }
    Ok(())
}

/// A bounded observation mean whose information ages as the request advances.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefixEstimate {
    mean: f64,
    weight: f64,
    observed_at: u32,
}
impl PrefixEstimate {
    pub fn new(mean: f64, generated: u32) -> Self {
        Self { mean, weight: 2.0, observed_at: generated }
    }

    pub fn weight_at(&self, generated: u32) -> f64 {
        self.weight * (-f64::from(generated.saturating_sub(self.observed_at)) / 128.0).exp2()
    }

    pub fn mean(&self) -> f64 { self.mean }

    pub fn upper(&self, generated: u32) -> f64 {
        (self.mean + (20.0f64.ln() / (2.0 * self.weight_at(generated))).sqrt()).min(1.0)
    }

    pub fn observe(&mut self, value: f64, generated: u32) {
        debug_assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        self.weight = (self.weight_at(generated) + 1.0).min(128.0);
        self.mean += (value - self.mean) / self.weight;
        self.observed_at = generated;
    }
}

/// Prefix survival of the complete proposal program, including copy overrides.
#[derive(Clone, Copy)]
pub struct AcceptanceProfile {
    prefixes: [[PrefixEstimate; 7]; 2],
    rounds: u64,
}

impl Default for AcceptanceProfile {
    fn default() -> Self {
        Self { prefixes: [std::array::from_fn(|i| PrefixEstimate::new(0.5f64.powi(i as i32 + 1), 0)); 2], rounds: 0 }
    }
}

impl AcceptanceProfile {
    /// Depth zero is the first draft after the pending target token.
    pub fn prefix_probability(&self, source: ProposalSource, depth: usize) -> Option<f64> {
        self.prefixes[source as usize]
            .get(depth)
            .map(|_| self.curve(source, 0, false)[depth])
    }

    pub(super) fn curve(&self, source: ProposalSource, generated: u32, optimistic: bool) -> [f64; 7] {
        let mut bound = 1.0f64;
        self.prefixes[source as usize].map(|p| {
            bound = bound.min(if optimistic { p.upper(generated) } else { p.mean() });
            bound
        })
    }

    pub fn expected_tokens(&self, source: ProposalSource, width: usize) -> f64 {
        f64::from(width != 0) + self.curve(source, 0, false)[..width.saturating_sub(1).min(7)].iter().sum::<f64>()
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    pub(super) fn information(&self, source: ProposalSource, depth: usize, generated: u32) -> f64 {
        self.prefixes[source as usize][depth].weight_at(generated)
    }

    /// Rejection falsifies every longer offered prefix; termination censors its suffix.
    pub(super) fn observe(
        &mut self,
        source: ProposalSource,
        generated: u32,
        offered: usize,
        accepted: usize,
        rejected: bool,
        survival: Option<&[f32; 7]>,
    ) {
        debug_assert!(accepted <= offered && offered <= 7);
        if offered == 0 {
            return;
        }
        let observed = if rejected { offered } else { accepted };
        for depth in 0..observed {
            let value = survival.map_or(f64::from(depth < accepted), |p| f64::from(p[depth]));
            self.prefixes[source as usize][depth].observe(value, generated);
        }
        self.rounds = self.rounds.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_ages_information_without_changing_the_mean() {
        let mut estimate = PrefixEstimate::new(0.5, 0);
        for _ in 0..256 { estimate.observe(0.0, 0); }
        let mean = estimate.mean();
        let upper = estimate.upper(0);
        assert!(upper < 0.2);
        assert_eq!(estimate.mean(), mean);
        assert!(estimate.upper(128) > upper);
        assert_eq!(estimate.upper(u32::MAX), 1.0);
        estimate.observe(1.0, 4096);
        assert!(estimate.mean() > 0.99);
    }

    #[test]
    fn shallow_rounds_do_not_refresh_unobserved_suffixes() {
        let mut profile = AcceptanceProfile::default();
        for _ in 0..256 { profile.observe(ProposalSource::Mtp, 0, 7, 0, true, None); }
        let before = profile.curve(ProposalSource::Mtp, 0, true);
        for generated in 1..=1024 {
            profile.observe(ProposalSource::Mtp, generated, 1, 1, true, None);
        }
        let means = profile.curve(ProposalSource::Mtp, 1024, false);
        let upper = profile.curve(ProposalSource::Mtp, 1024, true);
        assert!(means[0] > 0.99);
        assert!(means[1] < 0.01);
        assert!(upper[1] > before[1] + 0.5);
        assert!(upper.windows(2).all(|p| p[0] >= p[1]));
    }

    #[test]
    fn probability_feedback_replaces_the_binary_sample_and_respects_censoring() {
        let mut profile = AcceptanceProfile::default();
        let survival = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01];
        profile.observe(ProposalSource::Mtp, 0, 7, 0, true, Some(&survival));
        for depth in 0..7 {
            let expected = (2.0 * 0.5f64.powi(depth as i32 + 1) + f64::from(survival[depth])) / 3.0;
            assert!((profile.prefix_probability(ProposalSource::Mtp, depth).unwrap() - expected).abs() < 1e-12);
        }
        let before = profile.curve(ProposalSource::Mtp, 0, false);
        profile.observe(ProposalSource::Mtp, 0, 7, 1, false, Some(&survival));
        assert_eq!(profile.curve(ProposalSource::Mtp, 0, false)[1..], before[1..]);
    }
}
