//! Schedule one captured proposal and GPU-owned causal prefix allocation.

use super::*;

impl RequestScheduler {
    pub(in crate::scheduler) fn observe_packed_forecasts(&mut self, step: &ScheduledStep,
        records: &[CompletionRecord], confidence: &[[f32; 7]]) {
        let observations: Vec<_> = step.lanes.iter().enumerate().filter_map(|(row, lane)| {
            let record = &records[lane.request.slot as usize];
            let vector = record.vector?;
            (matches!(record.status, 1 | 2) && self.request(lane.request).is_ok()).then_some((
                lane.request, lane.tokens, (record.generated_tokens - vector.emitted_count) as u32,
                confidence[row]))
        }).collect();
        let split = self.split.as_mut().unwrap();
        for (request, limit, generated, confidence) in observations {
            let slot = request.slot as usize;
            let observation = survival(&confidence);
            let mut forecast = split.forecast[slot].filter(|f| f.request == request)
                .unwrap_or_else(|| Forecast::new(request, split.population, generated));
            for i in 0..limit.saturating_sub(1) {
                forecast.prefixes[i].observe(f64::from(observation[i]), generated);
                split.population[i] += POPULATION_WEIGHT * (observation[i] - split.population[i]);
            }
            split.forecast[slot] = Some(forecast);
        }
    }

    pub(super) fn packed_round_candidates(&self, output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES], age: f64, optimistic: bool) {
        let available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let request = entry.request.as_ref()?;
            (request.phase == RequestPhase::Decode && self.eligible(request)
                && !self.owns_proposal(request)).then_some(slot)
        }).collect();
        let mut forecasts = HashMap::new();
        for shape in self.costs.packed_round_shapes() {
            let scores = forecasts.entry(shape.width).or_insert_with(|| {
                let mut scores = Vec::new();
                for &slot in &available {
                    let request = self.slots[slot].request.as_ref().unwrap();
                    let limit = self.speculative_tokens(request, (shape.drafts + 1).min(shape.width));
                    let mut before = 1.0;
                    for position in 0..limit.saturating_sub(1) {
                        let total = self.proposal_forecast_tokens(request, position + 2, optimistic);
                        scores.push((total - before, position, slot));
                        before = total;
                    }
                }
                scores.sort_unstable_by(|a, b|
                    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                scores
            });
            let nanos = shape.nanos;
            let base: [_; REQUEST_LANES] = std::array::from_fn(|slot|
                values[slot].decode(1.0, nanos as f64 / 1e6));
            let gains: Vec<_> = scores.iter().map(|&(score, _, slot)|
                score * values[slot].output_token).collect();
            let mut ranks: [Vec<usize>; REQUEST_LANES] = std::array::from_fn(|_| Vec::new());
            for (rank, &(_, _, slot)) in scores.iter().enumerate() { ranks[slot].push(rank); }
            let mut best: Option<(u64, f64)> = None;
            // Alternate bounded orderings expose long useful prefixes, delivery
            // fairness, and short tails. Every cohort is priced by the GPU's
            // causal allocation and the full measured physical B/D/W/P round.
            for ordering in 0..3 {
                let priority = |slot: usize| {
                    let request = self.slots[slot].request.as_ref().unwrap();
                    let limit = self.speculative_tokens(request, (shape.drafts + 1).min(shape.width));
                    let full = base[slot] + ranks[slot].iter().take(shape.tokens - 1)
                        .map(|&rank| gains[rank]).sum::<f64>();
                    match ordering { 0 => full, 1 => base[slot], _ => full / limit as f64 }
                };
                let mut order: Vec<_> = available.iter().map(|&slot| (slot, priority(slot))).collect();
                order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                let mut frontier = std::collections::BinaryHeap::new();
                let (mut mask, mut pending_value, mut draft_value) = (0u64, 0., 0.);
                for (count, &(slot, _)) in order.iter().take(shape.batch.min(shape.tokens)).enumerate() {
                    mask |= 1 << slot;
                    pending_value += base[slot];
                    // The heap retains the best P-count speculative positions.
                    // Its capacity decreases as pending rows enter; discarded
                    // positions cannot re-enter at a subsequent cohort size.
                    for &rank in &ranks[slot] {
                        frontier.push(rank);
                        draft_value += gains[rank];
                    }
                    while frontier.len() > shape.tokens - count - 1 {
                        draft_value -= gains[frontier.pop().unwrap()];
                    }
                    let value = pending_value + draft_value;
                    if best.is_none_or(|(old_mask, old_value)|
                        value > old_value || (value == old_value &&
                            (mask.count_ones(), std::cmp::Reverse(mask)) >
                            (old_mask.count_ones(), std::cmp::Reverse(old_mask)))) {
                        best = Some((mask, value));
                    }
                }
            }
            if let Some((mask, value)) = best {
                output.push(Candidate { draft_tokens: shape.drafts, kind: GraphKind::PackedSpeculative,
                    mask, batch: shape.batch, capacity: shape.width, packed_token_capacity: shape.tokens,
                    prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: nanos,
                    rate: ValueRate::new(nanos, value, age), feedback: true,
                    verification_budget: None });
            }
        }
    }
}
