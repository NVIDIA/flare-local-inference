//! Retained duration estimates; zero denotes an unmeasured graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

pub(crate) fn observed_duration_ns(timing: ExecutionTiming) -> Option<u64> {
    timing
        .gpu_ns
        .into_iter()
        .flatten()
        .max()
        .filter(|ns| *ns > 0)
        .map(|ns| if timing.queued { ns } else { ns.max(timing.wall_ns) })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub drafts: usize,
    pub kind: GraphKind,
    pub batch: usize,
    pub tokens: usize,
    pub prefill: usize,
    pub packed: usize,
}

impl Key {
    pub fn from_step(step: &ScheduledStep) -> Self {
        Self { drafts: step.draft_tokens, kind: step.kind, batch: step.batch_capacity, tokens: step.token_capacity,
            prefill: step.prefill_capacity(), packed: step.packed_token_capacity }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PackedShape {
    pub batch: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

pub(super) struct PackedRoundShape {
    pub batch: usize,
    pub drafts: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

struct Estimate {
    nanos: u64,
    replays: ReplayTimes,
    handoff_ns: u64,
    observed_at_ns: u64,
    last_attempt_ns: u64,
}

#[derive(Default)]
struct ReplayTimes {
    samples: [u64; 9],
    len: usize,
    next: usize,
}

impl ReplayTimes {
    fn observe(&mut self, sample: u64, fresh: bool) -> u64 {
        if !fresh {
            *self = Self::default();
        }
        self.samples[self.next] = sample;
        self.next = (self.next + 1) % self.samples.len();
        self.len = (self.len + 1).min(self.samples.len());
        let mut ordered = self.samples;
        ordered[..self.len].sort_unstable();
        let middle = self.len / 2;
        if self.len % 2 == 0 {
            ((u128::from(ordered[middle - 1]) + u128::from(ordered[middle]) + 1) / 2) as u64
        } else {
            ordered[middle]
        }
    }
}

impl Estimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

#[derive(Default)]
pub(super) struct ActionCosts {
    entries: Vec<(Key, Estimate)>,
    index: std::collections::HashMap<Key, usize>,
    pub proposal_width: Option<usize>,
    packed_round_durations: [[Option<u64>; 9]; 7],
    packed_verification_durations: [[Option<u64>; 9]; 7],
    now_ns: u64,
    ttl_ns: u64,
}

impl ActionCosts {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn set_measured(&mut self, step: &ScheduledStep, nanos: u64) {
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        let estimate = &mut self.entries[self.index[&key]].1;
        estimate.nanos = nanos;
        estimate.replays.observe(nanos, false);
        estimate.handoff_ns = 0;
        estimate.observed_at_ns = self.now_ns;
        self.refresh_packed_durations();
    }

    pub fn prepared(graphs: &[GraphDescriptor]) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        let mut index = std::collections::HashMap::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                drafts: graph.draft_tokens,
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
                prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity,
            };
            if index.insert(key, entries.len()).is_some() {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
                    replays: ReplayTimes::default(),
                    handoff_ns: 0,
                    observed_at_ns: 0,
                    last_attempt_ns: 0,
                },
            ));
        }
        let proposal_width = entries.iter().filter_map(|(key, _)| match key.kind {
            GraphKind::Proposal => Some(key.tokens),
            GraphKind::PackedSpeculative => Some(key.drafts + 1),
            _ => None,
        }).max();
        // Only the full producer participates in scheduling, including timing and acceptance probes.
        entries.retain(|(key, _)| match key.kind {
            GraphKind::Proposal => Some(key.tokens) == proposal_width,
            GraphKind::PackedSpeculative => Some(key.drafts + 1) == proposal_width,
            _ => true,
        });
        let index = entries.iter().enumerate().map(|(index, (key, _))| (*key, index)).collect();
        let mut costs = Self { entries, index, proposal_width,
            ttl_ns: super::PolicyConfig::default().estimate_ttl_ns, ..Default::default() };
        costs.refresh_packed_durations();
        Ok(costs)
    }

    pub fn advance_to(&mut self, now_ns: u64, ttl_ns: u64) {
        self.now_ns = now_ns;
        self.ttl_ns = ttl_ns;
    }

    pub fn duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.shape_duration(kind, batch, tokens, 0).or_else(|| {
            (kind == GraphKind::Verification && batch.is_power_of_two() && batch <= super::REQUEST_LANES)
                .then(|| self.packed_verification_durations
                .get(batch.trailing_zeros() as usize)?.get(tokens).copied().flatten()).flatten()
        })
    }

    fn refresh_packed_durations(&mut self) {
        self.packed_round_durations = Default::default();
        self.packed_verification_durations = Default::default();
        for (key, estimate) in &self.entries {
            let durations = match key.kind {
                GraphKind::PackedSpeculative => &mut self.packed_round_durations,
                GraphKind::Verification if key.packed != 0 => &mut self.packed_verification_durations,
                _ => continue,
            };
            let maximum = if key.kind == GraphKind::PackedSpeculative {
                (key.drafts + 1).min(key.tokens)
            } else { key.tokens }.min(key.packed / key.batch);
            let nanos = estimate.nanos.saturating_add(estimate.handoff_ns);
            for duration in durations[key.batch.trailing_zeros() as usize].iter_mut().take(maximum + 1).skip(1) {
                *duration = Some(duration.map_or(nanos, |old| old.min(nanos)));
            }
        }
    }

    pub fn packed_round_duration(&self, batch: usize, width: usize) -> Option<u64> {
        if !batch.is_power_of_two() || batch > super::REQUEST_LANES { return None; }
        self.packed_round_durations.get(batch.trailing_zeros() as usize)?.get(width).copied().flatten()
    }

    pub fn packed_shapes(&self) -> impl Iterator<Item = PackedShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::Verification && key.packed != 0)
            .map(|(key, estimate)| PackedShape { batch: key.batch, width: key.tokens, tokens: key.packed,
                nanos: estimate.nanos.saturating_add(estimate.handoff_ns) })
    }

    pub fn packed_round_shapes(&self) -> impl Iterator<Item = PackedRoundShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::PackedSpeculative)
            .map(|(key, estimate)| PackedRoundShape {
                batch: key.batch, drafts: key.drafts, width: key.tokens, tokens: key.packed,
                nanos: estimate.nanos.saturating_add(estimate.handoff_ns),
            })
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.exact_duration(kind, batch, tokens, prefill, 0)
    }

    fn exact_duration(&self, kind: GraphKind, batch: usize, tokens: usize, prefill: usize, packed: usize) -> Option<u64> {
        self.entry(Key {
            drafts: 0,
            kind,
            batch,
            tokens,
            prefill,
            packed,
        })
        .map(|estimate| estimate.nanos.saturating_add(estimate.handoff_ns))
    }

    pub fn duration_key(&self, key: Key) -> Option<u64> {
        self.entry(key).map(|estimate| estimate.nanos.saturating_add(estimate.handoff_ns))
    }

    pub fn is_fresh(&self, key: Key) -> bool {
        self.entry(key).is_some_and(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
    }

    pub fn stale_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.entries.iter().filter(|(_, estimate)| estimate.nanos != 0
            && !estimate.fresh(self.now_ns, self.ttl_ns)).map(|(key, _)| *key)
    }

    pub fn refresh_order(&self, key: Key) -> u64 {
        self.entry(key).map_or(0, |estimate| estimate.observed_at_ns.max(estimate.last_attempt_ns))
    }

    pub fn record_attempt(&mut self, key: Key) {
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            estimate.last_attempt_ns = self.now_ns;
        }
    }

    pub fn observation(&self, key: Key) -> Option<(u64, Option<u64>, bool)> {
        self.entry(key).map(|estimate| (estimate.nanos.saturating_add(estimate.handoff_ns),
            (estimate.nanos != 0).then_some(estimate.observed_at_ns),
            estimate.fresh(self.now_ns, self.ttl_ns)))
    }

    fn entry(&self, key: Key) -> Option<&Estimate> {
        self.index.get(&key).map(|&index| &self.entries[index].1)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.duration_key(Key::from_step(step))
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(raw_sample) = timing.gpu_ns.into_iter().flatten().max().filter(|n| *n > 0) else {
            return;
        };
        let sample = raw_sample.saturating_sub(timing.readiness.map_or(0, |intervals| intervals.wait_ns()));
        if sample == 0 {
            return;
        }
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            let fresh = estimate.fresh(self.now_ns, self.ttl_ns);
            if !timing.queued {
                let handoff = timing.wall_ns.saturating_sub(raw_sample);
                estimate.handoff_ns = if fresh {
                    ((7 * u128::from(estimate.handoff_ns) + u128::from(handoff) + 4) / 8) as u64
                } else { handoff };
            } else if !fresh {
                estimate.handoff_ns = 0;
            }
            estimate.nanos = estimate.replays.observe(sample, fresh);
            estimate.observed_at_ns = self.now_ns;
            self.refresh_packed_durations();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReplayTimes;

    #[test]
    fn isolated_slow_replay_does_not_displace_the_steady_cost() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
        assert_eq!(replays.observe(16_000_000, true), 8_000_000);
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
    }

    #[test]
    fn sustained_cost_changes_replace_the_window_and_staleness_resets_it() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            replays.observe(8, true);
        }
        for _ in 0..4 {
            assert_eq!(replays.observe(16, true), 8);
        }
        assert_eq!(replays.observe(16, true), 16);
        assert_eq!(replays.observe(32, false), 32);
        assert_eq!(replays.observe(16, true), 24);
    }

    #[test]
    fn even_sample_median_does_not_overflow() {
        let mut replays = ReplayTimes::default();
        assert_eq!(replays.observe(u64::MAX, true), u64::MAX);
        assert_eq!(replays.observe(u64::MAX - 1, true), u64::MAX);
    }
}
