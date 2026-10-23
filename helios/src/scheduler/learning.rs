//! Cost-bounded acquisition of request-local speculative yield observations.

use super::{policy::{Candidate, bits}, utility::{LaneValue, ValueRate, best_cohort}, *};

impl RequestScheduler {
    pub(super) fn optimistic_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        f64::from(width != 0) + self.speculative_curve(request, true)[..width.saturating_sub(1)].iter().sum::<f64>()
    }

    pub(super) fn acceptance_probe(&self, ordinary: Candidate,
        values: &[LaneValue; REQUEST_LANES], age_price: f64) -> Option<Candidate> {
        let mut candidates = Vec::new();
        if self.split.is_some() {
            self.split_candidates(&mut candidates, values, age_price, true);
            candidates.retain(|candidate| matches!(candidate.kind, GraphKind::Proposal | GraphKind::PackedSpeculative));
        } else {
            self.mixed_candidates(&mut candidates, values, age_price, true);
            for batch in crate::execution::BATCH_CAPACITIES.into_iter()
                .take_while(|batch| *batch <= self.limits.max_active.next_power_of_two()) {
                for &width in &self.speculative_catalog[batch.ilog2() as usize] {
                    let Some(nanos) = self.costs.duration(GraphKind::Speculative, batch, width) else { continue; };
                    if width == 1 || nanos == 0 { continue; }
                    let offers: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
                        let request = entry.request.as_ref()?;
                        (self.eligible(request) && self.request_kind(request) == Some(GraphKind::Speculative)).then(|| {
                            let width = self.speculative_tokens(request, width);
                            (slot, values[slot].decode(self.optimistic_tokens(request, width), nanos as f64 / 1e6))
                        })
                    }).collect();
                    let mask = best_cohort::<REQUEST_LANES>(&offers, batch);
                    if mask == 0 { continue; }
                    let value = offers.iter().filter(|(slot, _)| mask & (1 << slot) != 0).map(|(_, value)| value).sum();
                    candidates.push(Candidate { draft_tokens: 0, kind: GraphKind::Speculative, mask, batch, capacity: width,
                        packed_token_capacity: 0, prefill_capacity: 0, prefill_slot: None, work: 0,
                        duration_ns: nanos, rate: ValueRate::new(nanos, value, age_price), feedback: true,
                        verification_budget: None });
                }
            }
        }
        let count = self.prune_padding(&mut candidates);
        let observed_width = |candidate: Candidate, slot: usize| {
            if candidate.mask & (1 << slot) == 0 || candidate.prefill_slot == Some(slot)
                || !matches!(candidate.kind, GraphKind::Speculative | GraphKind::Proposal | GraphKind::PackedSpeculative | GraphKind::Mixed) {
                1
            } else {
                let width = if candidate.kind == GraphKind::PackedSpeculative {
                    candidate.capacity.min(candidate.draft_tokens + 1)
                } else { candidate.capacity };
                self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(), width)
            }
        };
        candidates[..count].iter().copied().filter(|candidate| {
            candidate.rate.duration_ns != 0 && candidate.rate.compare(ordinary.rate).is_gt()
                && bits(candidate.mask).any(|slot| {
                    let request = self.slots[slot].request.as_ref().unwrap();
                    let width = observed_width(*candidate, slot);
                    width > observed_width(ordinary, slot)
                        && request.max_new_tokens - request.generated_tokens > width as u32
                        && self.limits.max_sequence_tokens - request.scheduled_consumed as usize > width
                })
        }).max_by(|a, b| {
            let gain = |candidate: &Candidate| (candidate.rate.score - ordinary.rate.score) / candidate.rate.duration_ns as f64;
            let information = |candidate: &Candidate| bits(candidate.mask).flat_map(|slot| {
                (observed_width(ordinary, slot) - 1..observed_width(*candidate, slot) - 1)
                    .map(move |depth| self.speculative_information(self.slots[slot].request.as_ref().unwrap(), depth))
            }).fold(f64::INFINITY, f64::min);
            gain(a).total_cmp(&gain(b))
                .then_with(|| information(b).total_cmp(&information(a)))
                .then(b.rate.duration_ns.cmp(&a.rate.duration_ns))
                .then(b.batch.cmp(&a.batch)).then(b.capacity.cmp(&a.capacity)).then(b.mask.cmp(&a.mask))
        })
    }
}
