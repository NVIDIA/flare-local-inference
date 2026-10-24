//! Bounded retained proposals, causal verification widths, and shared execution costs.

use super::{policy::{Candidate, bits}, utility::{LaneValue, ValueRate}, *};

mod packed;
mod packed_round;

pub(super) struct SplitSpeculation {
    pub width: usize,
    pub retained: [Option<RetainedProposal>; REQUEST_LANES],
    // Forecasts average prefix-survival probabilities, preserving correlation
    // between the conditional probabilities in each observed draft block.
    forecast: [Option<Forecast>; REQUEST_LANES],
    population: [f32; 7],
    pub(super) pending_budget: Option<(Fence, usize)>,
    pub(super) pending_decision: Option<(Fence, Box<ProposalDecision>)>,
}

#[derive(Clone, Copy)]
struct Forecast {
    request: RequestHandle,
    prefixes: [speculation::PrefixEstimate; 7],
}

impl Forecast {
    fn new(request: RequestHandle, prior: [f32; 7], generated: u32) -> Self {
        Self { request, prefixes: prior.map(|p| speculation::PrefixEstimate::new(f64::from(p), generated)) }
    }

    fn estimate(&self, generated: u32, optimistic: bool) -> [f32; 7] {
        let mut bound = 1.0f64;
        self.prefixes.map(|p| {
            bound = bound.min(if optimistic { p.upper(generated) } else { p.mean() });
            bound as f32
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct RetainedProposal {
    pub request: RequestHandle,
    pub fence: Fence,
    pub consumed: u64,
    pub width: usize,
    limit: usize,
    confidence: [f32; 7],
}

#[derive(Clone, Copy, Default)]
struct PackedCall {
    mask: u64,
    batch: usize,
    width: usize,
    nanos: u64,
}

pub(super) struct ProposalDecision {
    values: [LaneValue; REQUEST_LANES],
    age_price: f64,
    shapes: [[Option<PackedCall>; 9]; REQUEST_LANES + 1],
    forecasts: [[f64; 7]; REQUEST_LANES],
    unknown_verifiers: [u8; 7],
}

struct Packing {
    nanos: u64,
    calls: Vec<PackedCall>,
}

#[derive(Default)]
pub(super) struct RefreshBudgetCache {
    scores: HashMap<(usize, usize, u64), Vec<(f64, usize, usize, f64)>>,
}

fn horizon(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a.saturating_add(b) }
}

const FORECAST_PRIOR: f32 = 0.5;
const POPULATION_WEIGHT: f32 = 1.0 / 64.0;

// Bias scheduling value without changing observed prefix probabilities.
fn prefix_value(probability: f64, alpha: f64) -> f64 {
    if alpha == 1.0 { probability } else { probability.powf(alpha) }
}

fn expected(confidence: &[f32; 7], width: usize, alpha: f64) -> f64 {
    let mut survival = 1.0;
    let mut tokens = 1.0;
    for &p in &confidence[..width.saturating_sub(1)] {
        survival *= f64::from(p);
        tokens += prefix_value(survival, alpha);
    }
    tokens
}

fn continuation_tokens(confidence: &[f32; 7], forecast: &[f64; 7], frontier: usize, width: usize, alpha: f64) -> f64 {
    if width <= frontier { return expected(confidence, width, alpha); }
    let mut total = expected(confidence, frontier, alpha);
    let mut prefix = confidence[..frontier].iter().map(|p| f64::from(*p)).product::<f64>();
    total += prefix_value(prefix, alpha);
    for depth in frontier..width - 1 {
        let conditional = if forecast[depth - 1] > 0.0 { forecast[depth] / forecast[depth - 1] } else { 0.5 };
        prefix *= conditional;
        total += prefix_value(prefix, alpha);
    }
    total
}

fn survival(confidence: &[f32; 7]) -> [f32; 7] {
    let mut prefix = 1.0;
    confidence.map(|p| { prefix *= p; prefix })
}

fn forecast_tokens(prefixes: &[f32; 7], width: usize, alpha: f64) -> f64 {
    let mut bound = 1.0f32;
    1.0 + prefixes[..width.saturating_sub(1)].iter().map(|&p| {
        bound = bound.min(p);
        prefix_value(f64::from(bound), alpha)
    }).sum::<f64>()
}

// Sorting widths permits optimal contiguous groups with O(requests squared) transitions.
fn pack(needs: &[(usize, usize)], shapes: &[[Option<PackedCall>; 9]; REQUEST_LANES + 1]) -> Option<Packing> {
    if needs.is_empty() { return Some(Packing { nanos: 0, calls: vec![] }); }
    let mut rows = needs.to_vec();
    rows.sort_unstable_by_key(|&(slot, width)| (std::cmp::Reverse(width), slot));
    let mut masks = [0; REQUEST_LANES + 1];
    for (index, &(slot, _)) in rows.iter().enumerate() { masks[index + 1] = masks[index] | (1 << slot); }
    let mut best = [None; REQUEST_LANES + 1];
    best[0] = Some((0u64, 0usize, 0usize, PackedCall::default(), false));
    for end in 1..=rows.len() {
        for start in 0..end {
            let Some((previous, count, _, _, unknown)) = best[start] else { continue; };
            let Some(mut call) = shapes[end - start][rows[start].1] else { continue; };
            call.mask = masks[end] ^ masks[start];
            let unknown = unknown || call.nanos == 0;
            let total = previous.saturating_add(call.nanos);
            let key = (!unknown, total, count + 1, call.batch, call.width);
            if best[end].is_none_or(|(old, n, _, c, u)| key < (!u, old, n, c.batch, c.width)) {
                best[end] = Some((total, count + 1, start, call, unknown));
            }
        }
    }
    let (total, _, _, _, unknown) = best[rows.len()]?;
    let mut calls = Vec::new();
    let mut end = rows.len();
    while end > 0 {
        let (_, _, start, call, _) = best[end]?;
        calls.push(call);
        end = start;
    }
    Some(Packing { nanos: if unknown { 0 } else { total }, calls })
}

impl RequestScheduler {
    pub fn split_speculation(&mut self, width: usize) -> Result<()> {
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || !(2..=8).contains(&width)
            || self.speculative_widths != (1..=width).collect::<Vec<_>>()
            || self.proposal_source != speculation::ProposalSource::Mtp
        {
            return Err("split speculation requires learned proposals and contiguous logical prefix widths before admission".into());
        }
        self.split = Some(SplitSpeculation {
            width,
            retained: [None; REQUEST_LANES],
            forecast: [None; REQUEST_LANES],
            population: survival(&[FORECAST_PRIOR; 7]),
            pending_budget: None,
            pending_decision: None,
        });
        Ok(())
    }

    pub(super) fn owns_proposal(&self, request: &LiveRequest) -> bool {
        self.split.as_ref().is_some_and(|s| s.retained[request.handle.slot as usize]
            .is_some_and(|p| p.request == request.handle))
    }

    pub(super) fn retained_proposal(&self, request: &LiveRequest) -> Option<RetainedProposal> {
        self.split.as_ref()?.retained[request.handle.slot as usize].filter(|p| {
            p.request == request.handle
                && p.fence <= self.completed
                && request.phase == RequestPhase::Decode
                && p.consumed == request.consumed_tokens
        })
    }

    pub(super) fn expected_speculative_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        if let Some(proposal) = self.retained_proposal(request) {
            return expected(&proposal.confidence, width.min(proposal.width), self.policy.adaptive_verification_alpha);
        }
        if self.split.is_some() {
            forecast_tokens(&self.speculative_curve(request, false).map(|p| p as f32), width, self.policy.adaptive_verification_alpha)
        } else {
            request.acceptance.expected_tokens(self.proposal_source, width)
        }
    }

    pub(super) fn speculative_curve(&self, request: &LiveRequest, optimistic: bool) -> [f64; 7] {
        if let Some(split) = &self.split {
            let forecast = split.forecast[request.handle.slot as usize]
                .filter(|f| f.request == request.handle)
                .unwrap_or_else(|| Forecast::new(request.handle, split.population, request.generated_tokens));
            forecast.estimate(request.generated_tokens, optimistic).map(f64::from)
        } else {
            request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
        }
    }

    pub(super) fn speculative_information(&self, request: &LiveRequest, depth: usize) -> f64 {
        if let Some(split) = &self.split {
            split.forecast[request.handle.slot as usize].filter(|f| f.request == request.handle)
                .map_or(2.0, |f| f.prefixes[depth].weight_at(request.generated_tokens))
        } else {
            request.acceptance.information(self.proposal_source, depth, request.generated_tokens)
        }
    }

    fn proposal_forecast_tokens(&self, request: &LiveRequest, width: usize, optimistic: bool) -> f64 {
        f64::from(width != 0) + self.speculative_curve(request, optimistic)[..width.saturating_sub(1)].iter().map(|&p| prefix_value(p, self.policy.adaptive_verification_alpha)).sum::<f64>()
    }

    fn proposal_cost(&self, batch: usize, width: usize) -> Option<u64> {
        let producer = self.costs.proposal_width?;
        (width <= producer).then(|| self.costs.duration(GraphKind::Proposal, batch, producer)).flatten()
    }

    pub(super) fn proposal_round_duration(&self, batch: usize, width: usize) -> Option<u64> {
        let joined = self.costs.duration(GraphKind::Speculative, batch, width);
        let split = (width > 1).then(|| Some(horizon(self.proposal_cost(batch, width)?,
            self.costs.duration(GraphKind::Verification, batch, width)?))).flatten();
        let packed = (width > 1).then(|| self.costs.packed_round_duration(batch, width)).flatten();
        joined.into_iter().chain(split).chain(packed).min()
    }

    pub(super) fn speculative_duration(&self, request: &LiveRequest, batch: usize, width: usize) -> Option<u64> {
        if let Some(p) = self.retained_proposal(request) {
            if width < p.width { return None; }
            self.costs.duration(GraphKind::Verification, batch, width)
        } else {
            self.proposal_round_duration(batch, width)
        }
    }

    fn packing_shapes(&self, kind: GraphKind) -> [[Option<PackedCall>; 9]; REQUEST_LANES + 1] {
        let mut choices = [[None; 9]; REQUEST_LANES + 1];
        let mut minimum_width = [[usize::MAX; 9]; REQUEST_LANES + 1];
        for batch in crate::execution::BATCH_CAPACITIES.into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let widths = if kind == GraphKind::Proposal { self.costs.proposal_width.as_slice() }
                else { &self.speculative_catalog[batch.ilog2() as usize][..] };
            for &width in widths {
                let Some(nanos) = self.costs.duration(kind, batch, width) else { continue; };
                let call = PackedCall { mask: 0, batch, width, nanos };
                for (row, bounds) in choices.iter_mut().zip(&mut minimum_width).take(batch + 1).skip(1) {
                    for (choice, bound) in row.iter_mut().zip(bounds).take(width + 1).skip(1) {
                        // A more expensive smaller geometry still excludes later padded shapes.
                        if width >= *bound { continue; }
                        *bound = width;
                        if choice.is_none_or(|c: PackedCall| (nanos, batch, width) < (c.nanos, c.batch, c.width)) {
                            *choice = Some(call);
                        }
                    }
                }
            }
        }
        choices
    }

    #[cfg(test)]
    fn pack(&self, kind: GraphKind, needs: &[(usize, usize)]) -> Option<Packing> {
        pack(needs, &self.packing_shapes(kind))
    }

    fn retained_needs(&self) -> Vec<(usize, usize)> {
        self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let r = entry.request.as_ref()?;
            self.retained_proposal(r).map(|p| (slot, p.width))
        }).collect()
    }

    pub(super) fn refresh_verification_candidates(&self, output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES], age: f64, required: u64, exact: Option<cost::Key>) {
        let retained = self.retained_needs();
        let mut shapes = Vec::new();
        if let Some(shape) = exact {
            shapes.push(shape);
        } else if self.costs.packed_shapes().next().is_some() {
            shapes.extend(self.costs.packed_shapes().map(|shape| cost::Key { drafts: 0,
                kind: GraphKind::Verification, batch: shape.batch, tokens: shape.width,
                prefill: 0, packed: shape.tokens,
            }));
        } else {
            for batch in crate::execution::BATCH_CAPACITIES.into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
                shapes.extend(self.speculative_catalog[batch.ilog2() as usize].iter().map(|&tokens|
                    cost::Key { drafts: 0, kind: GraphKind::Verification, batch, tokens, prefill: 0, packed: 0 }));
            }
        }
        for shape in shapes {
            let Some(nanos) = self.costs.duration_key(shape) else { continue; };
            let value = |slot: usize, width: usize| {
                let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                values[slot].decode(expected(&proposal.confidence, width, self.policy.adaptive_verification_alpha), nanos as f64 / 1e6)
            };
            let mut offers: Vec<_> = retained.iter().copied().filter(|(_, width)| *width <= shape.tokens).collect();
            offers.sort_unstable_by(|&(a, aw), &(b, bw)| {
                let divisor = |width| if shape.packed == 0 { 1.0 } else { width as f64 };
                (value(b, bw) / divisor(bw)).total_cmp(&(value(a, aw) / divisor(aw))).then(a.cmp(&b))
            });
            let protected = if required == 0 { 0 } else if exact.is_some() {
                required
            } else {
                offers.iter().find(|(slot, _)| required & (1 << slot) != 0)
                    .map_or(0, |(slot, _)| 1 << slot)
            };
            if required != 0 && protected == 0 { continue; }
            let mut mask = 0u64;
            let mut tokens = 0;
            let mut count = 0;
            for &(slot, width) in offers.iter().filter(|(slot, _)| protected & (1 << slot) != 0) {
                mask |= 1 << slot;
                tokens += width;
                count += 1;
            }
            if mask != protected || count > shape.batch || (shape.packed != 0 && tokens > shape.packed) {
                continue;
            }
            for &(slot, width) in &offers {
                if mask & (1 << slot) != 0 || count == shape.batch
                    || (shape.packed != 0 && tokens + width > shape.packed) { continue; }
                mask |= 1 << slot;
                tokens += width;
                count += 1;
            }
            if mask == 0 { continue; }
            let total = offers.iter().filter(|(slot, _)| mask & (1 << slot) != 0)
                .map(|&(slot, width)| value(slot, width)).sum();
            output.push(Candidate { draft_tokens: 0, kind: GraphKind::Verification, mask, batch: shape.batch,
                capacity: shape.tokens, packed_token_capacity: shape.packed, prefill_capacity: 0,
                prefill_slot: None, work: 0, duration_ns: nanos,
                rate: ValueRate::new(nanos, total, age), feedback: true, verification_budget: None });
        }
    }

    pub(super) fn split_refresh_candidates(&self, output: &mut Vec<Candidate>, target: cost::Key,
        values: &[LaneValue; REQUEST_LANES], age: f64, mut cache: Option<&mut RefreshBudgetCache>) {
        if !matches!(target.kind, GraphKind::Proposal | GraphKind::Verification) { return; }
        if target.kind == GraphKind::Verification {
            let before = output.len();
            self.refresh_verification_candidates(output, values, age, 0, Some(target));
            if output.len() != before { return; }
        }
        let split = self.split.as_ref().unwrap();
        let available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let request = entry.request.as_ref()?;
            (request.phase == RequestPhase::Decode && self.eligible(request) && !self.owns_proposal(request)
                && self.speculative_tokens(request, split.width) > 1).then_some(slot)
        }).collect();
        if available.is_empty() { return; }
        let packed = self.costs.packed_shapes().next().is_some();
        let retained = self.retained_needs();
        let verification_shapes = (!packed && target.kind == GraphKind::Proposal)
            .then(|| self.packing_shapes(GraphKind::Verification));
        let Some(depth) = self.costs.proposal_width else { return; };
        for batch in crate::execution::BATCH_CAPACITIES.into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let producer = cost::Key { drafts: 0, kind: GraphKind::Proposal, batch, tokens: depth, prefill: 0, packed: 0 };
            if target.kind == GraphKind::Proposal && producer != target { continue; }
            let Some(proposal_ns) = self.costs.duration_key(producer) else { continue; };
            let logical = self.speculative_fixed_width.unwrap_or(depth).min(depth);
            let mut offers: Vec<_> = available.iter().copied().filter(|&slot| {
                let request = self.slots[slot].request.as_ref().unwrap();
                self.speculative_fixed_width.is_none_or(|fixed| {
                    let width = self.speculative_tokens(request, fixed);
                    width <= depth && (target.kind != GraphKind::Verification || width <= target.tokens)
                })
            }).map(|slot| {
                let request = self.slots[slot].request.as_ref().unwrap();
                let limit = self.speculative_tokens(request, logical);
                let width = if target.kind == GraphKind::Verification { limit.min(target.tokens) } else { limit };
                let value = values[slot].decode(self.proposal_forecast_tokens(request, width, false), 0.0);
                (slot, limit, value)
            }).collect();
            offers.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
            let capacity = if target.kind == GraphKind::Verification { batch.min(target.batch) } else { batch };
            for count in 1..=offers.len().min(capacity) {
                let propose: Vec<_> = offers[..count].iter().map(|&(slot, limit, _)| (slot, limit)).collect();
                let (rate, budget) = if packed {
                    let Some((shape, rate)) = self.refresh_budget_choice(&propose, count.next_power_of_two(),
                        logical, values, age, producer,
                        (target.kind == GraphKind::Verification).then_some(target), cache.as_deref_mut(), false) else { continue; };
                    (rate, Some(shape.tokens))
                } else {
                    let mut verify = if target.kind == GraphKind::Verification { Vec::new() } else { retained.clone() };
                    verify.extend(propose.iter().map(|&(slot, width)| (slot,
                        if target.kind == GraphKind::Verification { width.min(target.tokens) } else { width })));
                    let verification_ns = if target.kind == GraphKind::Verification {
                        let Some(nanos) = self.costs.duration_key(target) else { continue; };
                        nanos
                    } else {
                        let Some(packing) = pack(&verify, verification_shapes.as_ref().unwrap()) else { continue; };
                        packing.nanos
                    };
                    let nanos = horizon(proposal_ns, verification_ns);
                    let value = verify.iter().map(|&(slot, width)| {
                        let tokens = split.retained[slot].map_or_else(|| self.proposal_forecast_tokens(
                            self.slots[slot].request.as_ref().unwrap(), width, false),
                            |proposal| expected(&proposal.confidence, proposal.width, self.policy.adaptive_verification_alpha));
                        values[slot].decode(tokens, nanos as f64 / 1e6)
                    }).sum();
                    (ValueRate::new(nanos, value, age), None)
                };
                output.push(Candidate { draft_tokens: 0, kind: GraphKind::Proposal,
                    mask: propose.iter().fold(0, |mask, (slot, _)| mask | (1 << slot)),
                    batch, capacity: depth, packed_token_capacity: 0, prefill_capacity: 0,
                    prefill_slot: None, work: 0, duration_ns: proposal_ns, rate,
                    feedback: true, verification_budget: budget });
            }
        }
    }

    pub(super) fn proposal_decision(&self) -> Box<ProposalDecision> {
        let (values, age_price, _) = self.lane_values(0);
        Box::new(ProposalDecision { values, age_price,
            shapes: self.packing_shapes(GraphKind::Verification),
            unknown_verifiers: crate::execution::BATCH_CAPACITIES.map(|batch|
                self.speculative_catalog[batch.ilog2() as usize].iter().filter(|&&width|
                    self.costs.duration(GraphKind::Verification, batch, width) == Some(0))
                    .fold(0, |mask, &width| mask | (1 << (width - 1)))),
            forecasts: std::array::from_fn(|slot| self.slots.get(slot).and_then(|entry| entry.request.as_ref())
                .map_or([0.0; 7], |request| self.speculative_curve(request, false))),
        })
    }

    fn finalize_proposals(&mut self, fresh: u64, refresh_trial: Option<usize>, decision: &ProposalDecision) -> Result<()> {
        let mut needs = self.retained_needs();
        let split = self.split.as_ref().unwrap();
        let maximum = needs.iter().filter(|(s, _)| fresh & (1 << s) != 0)
            .map(|(s, _)| split.retained[*s].unwrap().limit).max().unwrap_or(1);
        let trial = crate::execution::BATCH_CAPACITIES.into_iter()
            .filter(|b| *b <= needs.len().max(1).next_power_of_two())
            .rev().find_map(|batch| self.speculative_catalog[batch.ilog2() as usize].iter().copied()
                .find(|w| *w <= maximum && (decision.unknown_verifiers[batch.ilog2() as usize] & (1 << (*w - 1))) != 0));
        if let Some(width) = self.speculative_fixed_width.or(refresh_trial).or(trial) {
            for (slot, selected) in &mut needs {
                if fresh & (1 << *slot) != 0 { *selected = width.min(split.retained[*slot].unwrap().limit); }
            }
        } else {
            let score = |needs: &[(usize, usize)], frontiers: &[usize; REQUEST_LANES]| -> Result<ValueRate> {
                let packing = pack(needs, &decision.shapes).ok_or("no verification packing for retained proposals")?;
                let value = needs.iter().map(|&(slot, width)| {
                    let confidence = &split.retained[slot].unwrap().confidence;
                    let tokens = continuation_tokens(confidence, &decision.forecasts[slot], frontiers[slot], width, self.policy.adaptive_verification_alpha);
                    decision.values[slot].decode(tokens, packing.nanos as f64 / 1e6)
                }).sum();
                Ok(ValueRate::new(packing.nanos, value, decision.age_price))
            };
            loop {
                let mut frontiers = [0; REQUEST_LANES];
                for &(slot, width) in &needs { frontiers[slot] = width; }
                let rate = score(&needs, &frontiers)?;
                let mut best = None;
                for target in 2..=maximum {
                    // Equal-width cohorts and their union can share a graph-cost increase.
                    for cohort in 0..target {
                        let mut frontier: Vec<_> = needs.iter().enumerate().filter_map(|(index, &(slot, width))| {
                            let proposal = split.retained[slot].unwrap();
                            let end = target.min(proposal.limit);
                            (fresh & (1 << slot) != 0 && width < end && (cohort == 0 || cohort == width)).then(|| {
                                let gain = decision.values[slot].output_token *
                                    (continuation_tokens(&proposal.confidence, &decision.forecasts[slot], width, end, self.policy.adaptive_verification_alpha)
                                        - expected(&proposal.confidence, width, self.policy.adaptive_verification_alpha));
                                (index, end, gain)
                            })
                        }).collect();
                        frontier.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
                        let mut extended = 0u64;
                        for &(index, end, _) in &frontier {
                            needs[index].1 = end;
                            extended |= 1 << index;
                            let candidate = score(&needs, &frontiers)?;
                            if candidate.compare(rate).is_gt()
                                && best.is_none_or(|(_, old)| candidate.compare(old).is_gt()) {
                                best = Some((extended, candidate));
                            }
                        }
                        for &(index, _, _) in &frontier { needs[index].1 = frontiers[needs[index].0]; }
                    }
                }
                let Some((extended, _)) = best else { break; };
                // Commit only the immediate frontier before reading another confidence.
                for index in bits(extended) { needs[index].1 += 1; }
            }
        }
        self.finalize_selected(fresh, needs);
        Ok(())
    }

    fn finalize_selected(&mut self, fresh: u64, needs: Vec<(usize, usize)>) {
        let generated: Vec<_> = needs.iter()
            .map(|(slot, _)| self.slots[*slot].request.as_ref().map_or(0, |r| r.generated_tokens)).collect();
        let split = self.split.as_mut().unwrap();
        for ((slot, width), generated) in needs.into_iter().zip(generated) {
            let proposal = split.retained[slot].as_mut().unwrap();
            proposal.width = width;
            if fresh & (1 << slot) != 0 {
                let observation = survival(&proposal.confidence);
                let mut forecast = split.forecast[slot].filter(|f| f.request == proposal.request)
                    .unwrap_or_else(|| Forecast::new(proposal.request, split.population, generated));
                for i in 0..proposal.limit.saturating_sub(1) {
                    forecast.prefixes[i].observe(f64::from(observation[i]), generated);
                    split.population[i] += POPULATION_WEIGHT * (observation[i] - split.population[i]);
                }
                split.forecast[slot] = Some(forecast);
            }
        }
    }

    pub(super) fn split_candidates(&self, output: &mut Vec<Candidate>, values: &[LaneValue; REQUEST_LANES], age_price: f64, optimistic: bool) {
        if self.costs.packed_round_shapes().next().is_some() {
            self.packed_round_candidates(output, values, age_price, optimistic);
        }
        if self.costs.packed_shapes().next().is_some() {
            self.packed_candidates(output, values, age_price, optimistic);
            return;
        }
        let split = self.split.as_ref().unwrap();
        let retained = self.retained_needs();
        let proposal_shapes = self.packing_shapes(GraphKind::Proposal);
        let verification_shapes = self.packing_shapes(GraphKind::Verification);
        let make = |kind, call: PackedCall, rate| Candidate { draft_tokens: 0, packed_token_capacity: 0, verification_budget: None,
            kind, mask: call.mask, batch: call.batch, capacity: call.width,
            prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: call.nanos, rate, feedback: true,
        };
        if let Some(packing) = pack(&retained, &verification_shapes) {
            for call in packing.calls {
                let value = bits(call.mask).map(|s| {
                    let p = split.retained[s].unwrap();
                    values[s].decode(expected(&p.confidence, p.width, self.policy.adaptive_verification_alpha), call.nanos as f64 / 1e6)
                }).sum();
                output.push(make(GraphKind::Verification, call, ValueRate::new(call.nanos, value, age_price)));
            }
        }
        let available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let r = entry.request.as_ref()?;
            (r.phase == RequestPhase::Decode && self.eligible(r) && !self.owns_proposal(r)).then_some(slot)
        }).collect();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let Some(ns) = self.costs.duration(GraphKind::Speculative, batch, 1) else { continue; };
            let full = self.speculative_fixed_width.is_some_and(|w| self.speculative_catalog[batch.ilog2() as usize].contains(&w));
            let offers: Vec<_> = available.iter().filter(|&&s| !full
                || self.speculative_tokens(self.slots[s].request.as_ref().unwrap(), split.width) == 1)
                .map(|&s| (s, values[s].decode(1.0, ns as f64 / 1e6))).collect();
            let mask = utility::best_cohort::<REQUEST_LANES>(&offers, batch);
            if mask != 0 {
                let value = offers.iter().filter(|(s, _)| mask & (1 << s) != 0).map(|(_, v)| v).sum();
                output.push(make(GraphKind::Speculative, PackedCall { mask, batch, width: 1, nanos: ns }, ValueRate::new(ns, value, age_price)));
            }
        }
        for &width in self.speculative_widths.iter().filter(|w| **w > 1
            && self.speculative_fixed_width.is_none_or(|fixed| fixed == **w)) {
            let mut offers: Vec<_> = available.iter().copied().filter(|s| {
                self.speculative_tokens(self.slots[*s].request.as_ref().unwrap(), split.width) > 1
            }).collect();
            let value = |slot: usize| {
                let r = self.slots[slot].request.as_ref().unwrap();
                values[slot].decode(self.proposal_forecast_tokens(r, self.speculative_tokens(r, width), optimistic), 0.0)
            };
            offers.sort_by(|a, b| value(*b).total_cmp(&value(*a)).then(a.cmp(b)));
            for count in 1..=offers.len() {
                let propose: Vec<_> = offers[..count].iter().map(|&s|
                    (s, self.speculative_tokens(self.slots[s].request.as_ref().unwrap(), width))).collect();
                let Some(proposal) = pack(&propose, &proposal_shapes) else { continue; };
                let mut verify = retained.clone();
                verify.extend(offers[..count].iter().map(|&s| (s, self.speculative_tokens(self.slots[s].request.as_ref().unwrap(), width))));
                let Some(verification) = pack(&verify, &verification_shapes) else { continue; };
                let ns = horizon(proposal.nanos, verification.nanos);
                let expected = verify.iter().map(|&(s, w)| {
                    let tokens = split.retained[s].map_or_else(|| self.proposal_forecast_tokens(self.slots[s].request.as_ref().unwrap(), w, optimistic),
                        |p| expected(&p.confidence, p.width, self.policy.adaptive_verification_alpha));
                    values[s].decode(tokens, ns as f64 / 1e6)
                }).sum();
                if let Some(call) = proposal.calls.into_iter().max_by(|a, b| {
                    let value = |c: &PackedCall| bits(c.mask).map(value).sum::<f64>();
                    value(a).total_cmp(&value(b)).then(b.mask.cmp(&a.mask))
                }) {
                    output.push(make(GraphKind::Proposal, call, ValueRate::new(ns, expected, age_price)));
                }
            }
        }
    }

    pub fn complete_proposal(&mut self, fence: Fence, records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming, confidence: &[[f32; 7]]) -> Result<Vec<GeneratedToken>> {
        self.split.as_ref().ok_or("proposal completion without a split catalog")?;
        let step = self.submitted_step(fence)?.clone();
        if step.kind != GraphKind::Proposal || confidence.len() < step.batch_capacity
            || Some(step.token_capacity) != self.costs.proposal_width {
            return Err("proposal completion does not match its batch and confidence extent".into());
        }
        if self.split.as_ref().unwrap().pending_decision.as_ref().is_none_or(|(owner, _)| *owner != fence) {
            return Err("proposal decision belongs to another submission".into());
        }
        for row in confidence.iter().take(step.lanes.len()) {
            if row[..step.token_capacity - 1].iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err("proposal confidence must be finite and in [0, 1]".into());
            }
        }
        for (lane, &confidence) in step.lanes.iter().zip(confidence) {
            self.split.as_mut().unwrap().retained[lane.request.slot as usize] = Some(RetainedProposal {
                request: lane.request, fence, consumed: lane.start_position, width: 1, limit: lane.tokens, confidence,
            });
        }
        let generated = match self.complete(fence, records, timing) {
            Ok(generated) => generated,
            Err(error) => {
                for lane in &step.lanes { self.split.as_mut().unwrap().retained[lane.request.slot as usize] = None; }
                return Err(error);
            }
        };
        let mut fresh = 0;
        for lane in &step.lanes {
            if self.request(lane.request).is_ok_and(|r| r.phase == RequestPhase::Decode && r.error_code.is_none()) {
                fresh |= 1 << lane.request.slot;
            } else {
                self.split.as_mut().unwrap().retained[lane.request.slot as usize] = None;
            }
        }
        let decision = self.split.as_mut().unwrap().pending_decision.take().unwrap().1;
        let budget = self.split.as_mut().unwrap().pending_budget.take();
        let refresh_target = self.refresh_episode()
            .filter(|episode| episode.id == fence.0)
            .and_then(|episode| episode.target.filter(|target| target.kind == GraphKind::Verification));
        if let Some((owner, tokens)) = budget {
            if owner != fence { return Err("packed verification budget belongs to another proposal".into()); }
            if let Some(target) = refresh_target {
                self.finalize_packed_with_width(fresh, tokens, Some(target.tokens), &decision)?;
            } else {
                self.finalize_packed(fresh, tokens, &decision)?;
            }
        } else {
            self.finalize_proposals(fresh, refresh_target.map(|target| target.tokens), &decision)?;
        }
        self.reconcile_refresh();
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_budgets_adapt_verification_with_full_proposals() {
        for active in [1, 3, 8] {
            let mut s = engine(8);
            s.configure_policy(PolicyConfig::throughput()).unwrap();
            let mut graphs = super::super::tests::duration_graphs(8);
            graphs.retain(|g| g.kind != GraphKind::Speculative);
            for batch in [1, 2, 4, 8] {
                for (kind, width, budget) in [(GraphKind::Speculative, 1, 0)].into_iter()
                    .chain([(GraphKind::Proposal, 8, 0)])
                    .chain((1..=8).map(|width| (GraphKind::Verification, 8, batch * width)))
                {
                    graphs.push(GraphDescriptor { draft_tokens: 0, name: format!("{kind:?}-{batch}-{width}-{budget}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch, token_capacity: width,
                        packed_token_capacity: budget, prefill_capacity: 0, estimated_duration_ns: 1, input_bindings: vec![] });
                }
            }
            s.graph_durations(&graphs).unwrap();
            for graph in &graphs {
                let duration = match graph.kind {
                    GraphKind::Proposal => (graph.token_capacity as u64 - 1) * 500_000,
                    GraphKind::Verification => 8_000_000 + graph.packed_token_capacity as u64 * 375_000,
                    _ => 11_000_000,
                };
                s.costs.set_measured(&ScheduledStep { draft_tokens: 0, kind: graph.kind, batch_capacity: graph.batch_capacity,
                    token_capacity: graph.token_capacity, packed_token_capacity: graph.packed_token_capacity,
                    mixed: None, lanes: vec![], requires_feedback: true }, duration);
            }
            let handles: Vec<_> = (0..active).map(|_| ready(&mut s)).collect();
            for confidence in [[0.99; 7], [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]] {
                s.split.as_mut().unwrap().population = survival(&confidence);
                let plan = s.plan().unwrap();
                assert_eq!(plan.kind, GraphKind::Proposal);
                assert_eq!(plan.token_capacity, 8);
                assert_eq!(plan.lanes.len(), active);
                assert!(plan.lanes.iter().all(|lane| lane.tokens == 8));
            }
            let plan = s.plan().unwrap();
            let budget = plan.verification_budget.unwrap();
            s.commit(&plan, Fence(1)).unwrap();
            let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
            for &handle in &handles {
                records[handle.slot as usize] = CompletionRecord { generation: handle.generation as i64,
                    submission_sequence: 1, consumed_tokens: 65, generated_tokens: 1, token_id: -1,
                    status: 1, vector: Some(TokenVector::default()), ..Default::default() };
            }
            let mut confidence = vec![[f32::NAN; 7]; plan.batch_capacity];
            for row in confidence.iter_mut().take(active) { *row = [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]; }
            s.complete_proposal(Fence(1), &records, Default::default(), &confidence).unwrap();
            for (row, handle) in handles.into_iter().enumerate() {
                let proposal = s.retained_proposal(s.request(handle).unwrap()).unwrap();
                let width = budget / active + usize::from(row < budget % active);
                assert_eq!((proposal.width, proposal.limit), (width, 8));
                assert!(s.split.as_ref().unwrap().forecast[handle.slot as usize].unwrap().prefixes.iter().all(|p| p.mean().is_finite()));
            }
            let verify = s.plan().unwrap();
            assert_eq!(verify.kind, GraphKind::Verification);
            assert_eq!(verify.token_capacity, 8);
            assert_eq!(verify.lanes.iter().map(|lane| lane.tokens).sum::<usize>(), budget);
            s.commit(&verify, Fence(2)).unwrap();
        }
    }

    #[test]
    fn packed_budget_survives_cost_updates_before_commit_and_redistributes_across_requests() {
        let mut s = engine(8);
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(8);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        for batch in [1, 2, 4, 8] {
            for (kind,width,budget) in [(GraphKind::Speculative,1,0),(GraphKind::Proposal,8,0)].into_iter()
                .chain((1..=8).map(|width| (GraphKind::Verification,8,batch*width))) {
                graphs.push(GraphDescriptor { draft_tokens: 0, name: format!("{kind:?}-{batch}-{width}-{budget}"),kind,
                    sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch, token_capacity: width,
                    packed_token_capacity: budget, prefill_capacity: 0, estimated_duration_ns: 1, input_bindings: vec![] });
            }
        }
        s.graph_durations(&graphs).unwrap();
        let mut measurements = Vec::new();
        for graph in &graphs {
            let duration = match graph.kind {
                GraphKind::Proposal => 2_000_000,
                GraphKind::Verification => 8_000_000 + graph.packed_token_capacity as u64 * 375_000,
                _ => 11_000_000,
            };
            let step = ScheduledStep { draft_tokens: 0, kind: graph.kind, batch_capacity: graph.batch_capacity,
                token_capacity: graph.token_capacity, packed_token_capacity: graph.packed_token_capacity,
                mixed: None, lanes: vec![], requires_feedback: true };
            s.costs.set_measured(&step, duration);
            measurements.push((step, duration));
        }
        s.split.as_mut().unwrap().population = survival(&[0.7,0.5,0.1,0.1,0.1,0.1,0.1]);
        let handles: Vec<_> = (0..8).map(|_|ready(&mut s)).collect();
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind,GraphKind::Proposal);
        assert_eq!(plan.batch_capacity,8);
        assert_eq!(plan.verification_budget, Some(24));
        for (step, _) in &measurements {
            if step.kind == GraphKind::Verification {
                s.costs.set_measured(step, if step.packed_token_capacity == 8 { 1 } else { 100_000_000 });
            }
        }
        assert_eq!(s.plan().unwrap().verification_budget, Some(8));
        let fence = Fence(1);
        s.commit(&plan,fence).unwrap();
        let budget = s.split.as_ref().unwrap().pending_budget.unwrap().1;
        assert_eq!(budget,24);
        for (step, duration) in &measurements {
            s.costs.set_measured(step, *duration);
        }
        s.split.as_mut().unwrap().population = [1.;7];
        let mut records = vec![CompletionRecord::default();s.limits.request_capacity()];
        for &h in &handles {
            records[h.slot as usize] = CompletionRecord { generation: h.generation as i64,
                submission_sequence: fence.0 as i64, consumed_tokens: 65, generated_tokens: 1,
                token_id: -1, status: 1, vector: Some(TokenVector::default()), ..Default::default() };
        }
        let confidence: Vec<_> = (0..8).map(|row|if row<2 {[0.999;7]} else {[0.3;7]}).collect();
        assert!(s.complete_proposal(fence,&records,Default::default(),&confidence).unwrap().is_empty());
        let prefixes: Vec<_> = handles.iter().map(|&h| s.retained_proposal(s.request(h).unwrap()).unwrap().width).collect();
        assert_eq!(prefixes,[8,8,2,2,1,1,1,1]);
        assert_eq!(prefixes.iter().sum::<usize>(),budget);
        let verify = s.plan().unwrap();
        assert_eq!(verify.kind,GraphKind::Verification);
        assert_eq!(verify.batch_capacity,8);
        assert_eq!(verify.packed_token_capacity,24);
        assert_eq!(verify.token_capacity,8);
        s.commit(&verify,Fence(2)).unwrap();
    }
    use crate::execution::{GraphDescriptor, SamplingMode, TokenVector};

    #[test]
    fn profitable_proposals_compete_with_full_cohort_decode() {
        let mut s = measured_split_engine(8);
        let confidence = [0.73, 0.59, 0.49, 0.43, 0.38, 0.0, 0.0];
        for (batch, plain, draft) in [(1, 7.5, 1.215), (2, 8.5, 1.563), (4, 9.5, 1.935), (8, 10.99, 2.529)] {
            for (kind, width, ms) in [(GraphKind::Speculative, 1, plain), (GraphKind::Proposal, 6, draft)] {
                s.costs.set_measured(&ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind, batch_capacity: batch, token_capacity: width,
                    mixed: None, lanes: vec![], requires_feedback: true }, (ms * 1e6) as u64);
            }
        }
        costs(&mut s, 8, &[10.99, 14.609, 17.357, 19.252, 21.921, 24.046].map(|x| (x * 1e6) as u64));
        for _ in 0..8 {
            let h = ready(&mut s);
            let r = s.request_mut(h).unwrap();
            r.accounting.service.tokens = Some(utility::ServicePrice::initial(0.09));
            r.accounting.service.deliveries = Some(utility::ServicePrice::initial(0.09));
        }
        s.split.as_mut().unwrap().population = survival(&confidence);
        let (values, age, _) = s.lane_values(0);
        let mut candidates = vec![];
        s.split_candidates(&mut candidates, &values, age, false);
        candidates.sort_by(|a, b| b.rate.compare(a.rate));
        assert_eq!(candidates[0].kind, GraphKind::Proposal);
        assert_eq!(candidates[0].mask.count_ones(), 8);
    }

    #[test]
    fn throughput_preference_can_choose_more_tokens_in_a_longer_burst() {
        for (policy, wanted) in [(PolicyConfig::default(), 2), (PolicyConfig::throughput(), 3),
            (PolicyConfig { adaptive_verification_alpha: 0.5, ..Default::default() }, 3)] {
            let mut s = measured_split_engine(8);
            s.configure_policy(policy).unwrap();
            costs(&mut s, 8, &[10.8, 14.5, 16.6, 22.0, 25.0, 28.0].map(|x| (x * 1e6) as u64));
            let mut fresh = 0;
            for _ in 0..8 {
                let h = ready(&mut s);
                let request = s.request_mut(h).unwrap();
                request.accounting.service.tokens = Some(utility::ServicePrice::initial(0.09));
                request.accounting.service.deliveries = Some(utility::ServicePrice::initial(0.065));
                retain(&mut s, h, [0.67, 0.38, 0.01, 0.01, 0.01, 0.0, 0.0]);
                s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 6;
                fresh |= 1 << h.slot;
            }
            s.finalize_proposals(fresh, None, &s.proposal_decision()).unwrap();
            assert!(s.split.as_ref().unwrap().retained.iter().flatten().all(|p| p.width == wanted));
        }
    }

    fn engine(active: usize) -> RequestScheduler {
        let mut s = super::super::tests::scheduler(active, 512, 64);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(active);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        let mut shapes = Vec::new();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= active.next_power_of_two()) {
            for (kind, widths) in [
                (GraphKind::Proposal, vec![8]), (GraphKind::Speculative, vec![1]),
                (GraphKind::Verification, (1..=8).filter(|w| *w == 1 || batch * w <= 8).collect()),
            ] {
                for width in widths {
                    if kind == GraphKind::Verification { shapes.push((batch, width)); }
                    graphs.push(GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0, name: format!("{kind:?}-{batch}-{width}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: width, prefill_capacity: 0,
                        estimated_duration_ns: if kind == GraphKind::Speculative { 100_000 } else { 1000 },
                        input_bindings: vec![] });
                }
            }
        }
        s.speculative_graphs(&shapes).unwrap();
        s.measured_graph_durations(&graphs).unwrap();
        s.split_speculation(8).unwrap();
        s
    }

    fn ready(s: &mut RequestScheduler) -> RequestHandle {
        let handle = s.admit(super::super::tests::spec(65, 64)).unwrap();
        let r = s.request_mut(handle).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 65;
        r.scheduled_consumed = 65;
        r.generated_tokens = 1;
        handle
    }

    fn proposal(s: &mut RequestScheduler, handles: &[RequestHandle], confidence: &[[f32; 7]]) -> Fence {
        let fence = Fence(s.submitted.0 + 1);
        let step = ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: handles.len().next_power_of_two(),
            token_capacity: 8, mixed: None, requires_feedback: true,
            lanes: handles.iter().map(|&h| ScheduledLane { request: h, start_position: 65, tokens: 8 }).collect() };
        let plan = ScheduledPlan { origin: PlanOrigin::Ordinary, selected_cost: Default::default(),
            owner: s.plan_owner.clone(), epochs: handles.iter().map(|h| s.request(*h).unwrap().epoch).collect(),
            estimated_duration_ns: 1000, selection_value_rate: 0.0, selection_log_scale: 0.0, verification_budget: None, step };
        s.commit(&plan, fence).unwrap();
        let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
        for &h in handles { records[h.slot as usize] = CompletionRecord {
            generation: h.generation as i64, submission_sequence: fence.0 as i64,
            consumed_tokens: 65, generated_tokens: 1, token_id: -1, status: 1,
            vector: Some(TokenVector::default()), ..Default::default()
        }; }
        let output = s.complete_proposal(fence, &records, Default::default(), confidence).unwrap();
        assert!(output.is_empty());
        fence
    }

    fn retain(s: &mut RequestScheduler, handle: RequestHandle, confidence: [f32; 7]) {
        s.split.as_mut().unwrap().retained[handle.slot as usize] = Some(RetainedProposal {
            request: handle, fence: s.completed, consumed: 65, width: 1, limit: 8, confidence,
        });
    }

    fn costs(s: &mut RequestScheduler, batch: usize, times: &[u64]) {
        for (i, &ns) in times.iter().enumerate() {
            s.costs.set_measured(&ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Verification, batch_capacity: batch,
                token_capacity: i + 1, mixed: None, lanes: vec![], requires_feedback: true }, ns);
        }
    }

    #[test]
    fn retained_verification_uses_unpadded_work_despite_a_faster_padded_estimate() {
        let mut s = engine(4);
        costs(&mut s, 1, &[1000, 9000, 8000, 7000, 6000, 5000, 4000, 3000]);
        costs(&mut s, 2, &[1000, 2000, 1000, 1000]);
        costs(&mut s, 4, &[1000, 1000]);
        let h = ready(&mut s);
        retain(&mut s, h, [0.9; 7]);
        s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().width = 2;
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Verification);
        assert_eq!((plan.batch_capacity, plan.token_capacity, plan.lanes[0].tokens), (1, 2, 2));

        for _ in 0..2 {
            let h = ready(&mut s);
            retain(&mut s, h, [0.9; 7]);
            s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().width = 2;
        }
        let plan = s.plan().unwrap();
        assert_eq!((plan.batch_capacity, plan.lanes.len()), (4, 3));
    }

    #[test]
    fn an_unselected_packing_shape_still_excludes_larger_padding() {
        let mut s = measured_split_engine(4);
        s.speculative_graphs(&[(1, 6), (2, 4), (4, 4)]).unwrap();
        for (batch, width, ns) in [(1, 6, 1000), (2, 4, 2000), (4, 4, 500)] {
            s.costs.set_measured(&ScheduledStep { draft_tokens: 0, kind: GraphKind::Verification,
                batch_capacity: batch, token_capacity: width, packed_token_capacity: 0,
                mixed: None, lanes: vec![], requires_feedback: true }, ns);
        }
        let h = ready(&mut s);
        retain(&mut s, h, [0.9; 7]);
        s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().width = 4;
        let plan = s.plan().unwrap();
        assert_eq!((plan.kind, plan.batch_capacity, plan.token_capacity), (GraphKind::Verification, 1, 6));
    }

    fn measured_split_engine(active: usize) -> RequestScheduler {
        measured_split_engine_depths(active, &[6])
    }

    fn measured_split_engine_depths(active: usize, depths: &[usize]) -> RequestScheduler {
        let mut s = super::super::tests::scheduler(active, 512, 64);
        s.enable_speculation(&(1..=6).collect::<Vec<_>>()).unwrap();
        let mut graphs = super::super::tests::duration_graphs(active);
        graphs.retain(|g| g.kind != GraphKind::Speculative);
        let mut shapes = Vec::new();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= active.next_power_of_two()) {
            for (kind, widths) in [(GraphKind::Proposal, depths.to_vec()), (GraphKind::Speculative, vec![1]),
                                   (GraphKind::Verification, (1..=6).collect())] {
                for width in widths {
                    if kind == GraphKind::Verification { shapes.push((batch, width)); }
                    graphs.push(GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0, name: format!("{kind:?}-{batch}-{width}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: width, prefill_capacity: 0, estimated_duration_ns: 1000, input_bindings: vec![] });
                }
            }
        }
        s.speculative_graphs(&shapes).unwrap();
        s.measured_graph_durations(&graphs).unwrap();
        s.split_speculation(6).unwrap();
        let ms = |v: f64| (v * 1e6) as u64;
        for (batch, row) in [(1, [7.3, 7.8, 8.7, 9.0, 9.6, 11.0]), (2, [8.4, 9.6, 10.6, 11.6, 12.6, 13.8]),
                             (4, [9.8, 11.9, 13.4, 15.0, 16.6, 18.4]), (8, [11.6, 15.4, 17.5, 20.0, 22.6, 25.4])] {
            if batch > active.next_power_of_two() { continue; }
            costs(&mut s, batch, &row.map(ms));
        }
        s
    }

    #[test]
    fn shared_verification_calls_widen_together_at_measured_costs() {
        let confidence = [0.70, 0.62, 0.54, 0.52, 0.51, 0.5, 0.5];
        for active in [1, 8] {
            let mut s = measured_split_engine(active);
            let handles: Vec<_> = (0..active).map(|_| ready(&mut s)).collect();
            let mut fresh = 0;
            for &h in &handles {
                retain(&mut s, h, confidence);
                s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 6;
                fresh |= 1 << h.slot;
            }
            s.finalize_proposals(fresh, None, &s.proposal_decision()).unwrap();
            let widths: Vec<_> = handles.iter().map(|h| s.retained_proposal(s.request(*h).unwrap()).unwrap().width).collect();
            assert!(widths.iter().all(|w| *w == widths[0] && *w >= 3), "active {active}: {widths:?}");
        }
    }

    #[test]
    fn full_proposals_ignore_shorter_legacy_producers_and_prefix_forecasts() {
        let mut s = measured_split_engine_depths(8, &[2, 3, 4, 5, 6]);
        for batch in [1, 2, 4, 8] {
            for width in 2..=6 {
                let step = ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Proposal,
                    batch_capacity: batch, token_capacity: width, mixed: None, lanes: vec![], requires_feedback: true };
                s.costs.observe(&step, crate::execution::ExecutionTiming {
                    gpu_ns: [Some(if width == 6 { 2_500_000 } else { 1 }), None, None, None,
                        None, None, None, None, None, None, None, None, None, None, None, None],
                    ..Default::default()
                });
                assert_eq!(s.costs.get(&step).is_some(), width == 6);
            }
            s.costs.set_measured(&ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: batch,
                token_capacity: 1, mixed: None, lanes: vec![], requires_feedback: true }, 11_000_000);
        }
        for _ in 0..8 { ready(&mut s); }
        for confidence in [[0.9, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0], [0.99; 7]] {
            s.split.as_mut().unwrap().population = survival(&confidence);
            let plan = s.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Proposal);
            assert_eq!(plan.token_capacity, 6);
            assert_eq!(plan.lanes.len(), 8);
            assert!(plan.lanes.iter().all(|lane| lane.tokens == 6));
            for width in 2..6 {
                let short = ScheduledPlan { step: ScheduledStep { token_capacity: width, ..plan.step.clone() }, ..plan.clone() };
                assert!(s.commit(&short, Fence(1)).is_err());
                assert!(s.submissions.is_empty());
            }
        }
        s.advance_to(s.policy.estimate_ttl_ns).unwrap();
        assert!(s.costs.stale_keys().filter(|key| key.kind == GraphKind::Proposal).all(|key| key.tokens == 6));
        let (values, age, _) = s.lane_values(0);
        let mut learning = Vec::new();
        s.split_candidates(&mut learning, &values, age, true);
        assert!(learning.iter().any(|candidate| candidate.kind == GraphKind::Proposal));
        assert!(learning.iter().filter(|candidate| candidate.kind == GraphKind::Proposal).all(|candidate| candidate.capacity == 6));
    }

    #[test]
    fn output_tails_limit_observations_without_shortening_the_producer() {
        let mut s = measured_split_engine_depths(1, &[2, 6]);
        let h = ready(&mut s);
        s.request_mut(h).unwrap().max_new_tokens = 3;
        let fence = Fence(1);
        let step = ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: 1, token_capacity: 6,
            mixed: None, requires_feedback: true,
            lanes: vec![ScheduledLane { request: h, start_position: 65, tokens: 2 }] };
        let plan = ScheduledPlan { origin: PlanOrigin::Ordinary, selected_cost: Default::default(),
            owner: s.plan_owner.clone(), epochs: vec![s.request(h).unwrap().epoch],
            estimated_duration_ns: 1_000_000, selection_value_rate: 0.0, selection_log_scale: 0.0, verification_budget: None, step };
        s.commit(&plan, fence).unwrap();
        let record = CompletionRecord { generation: h.generation as i64, submission_sequence: 1,
            consumed_tokens: 65, generated_tokens: 1, token_id: -1, status: 1,
            vector: Some(TokenVector::default()), ..Default::default() };
        let before = s.split.as_ref().unwrap().population;
        s.complete_proposal(fence, &[record], Default::default(), &[[0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).unwrap();
        let f = s.split.as_ref().unwrap().forecast[h.slot as usize].unwrap();
        assert!(f.prefixes[0].mean() > f64::from(before[0]));
        assert_eq!(f.estimate(1, false)[1..], before[1..]);
        assert_eq!(&s.split.as_ref().unwrap().population[1..], &before[1..]);
        assert!(s.retained_proposal(s.request(h).unwrap()).unwrap().width <= 2);
        s.split.as_mut().unwrap().retained[h.slot as usize] = None;
        s.request_mut(h).unwrap().generated_tokens = 128 + 1;
        s.split.as_mut().unwrap().forecast[h.slot as usize].as_mut().unwrap().prefixes[0].observe(0.9, 129);
        assert!(s.proposal_forecast_tokens(s.request(h).unwrap(), 6, true)
            > s.expected_speculative_tokens(s.request(h).unwrap(), 6));
    }

    #[test]
    fn a_low_forecast_keeps_its_mean_and_requires_funded_exploration() {
        let ms = |v: f64| (v * 1e6) as u64;
        let mut s = measured_split_engine(1);
        let h = s.admit(super::super::tests::spec(65, 1024)).unwrap();
        let r = s.request_mut(h).unwrap();
        (r.phase, r.consumed_tokens, r.scheduled_consumed, r.generated_tokens) = (RequestPhase::Decode, 65, 65, 1);
        s.costs.set_measured(&ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: 1, token_capacity: 1,
            mixed: None, lanes: vec![], requires_feedback: true }, ms(7.3));
        s.costs.set_measured(&ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind: GraphKind::Proposal, batch_capacity: 1, token_capacity: 6,
            mixed: None, lanes: vec![], requires_feedback: true }, ms(1.34));
        let generated = s.request(h).unwrap().generated_tokens;
        s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast::new(h, [0.0; 7], generated));
        assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
        s.request_mut(h).unwrap().generated_tokens += 512;
        assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
        s.refresh.acceptance_credit_ns = refresh::CREDIT_CAP_NS;
        assert_eq!(s.plan().unwrap().origin, PlanOrigin::AcceptanceStart);
        assert_eq!(s.plan().unwrap().kind, GraphKind::Proposal);
    }

    #[test]
    fn forecasts_average_proposals_from_the_prior() {
        let mut s = engine(1);
        let h = ready(&mut s);
        for _ in 0..2 {
            retain(&mut s, h, [1.0; 7]);
            s.finalize_proposals(1 << h.slot, None, &s.proposal_decision()).unwrap();
        }
        let forecast = s.split.as_ref().unwrap().forecast[h.slot as usize].unwrap().estimate(s.request(h).unwrap().generated_tokens, false);
        assert!((forecast[0] - ((2.0 * 0.5 + 2.0) / 4.0)).abs() < 1e-6, "{forecast:?}");
    }

    #[test]
    fn verification_alpha_biases_scores_without_changing_observations() {
        for alpha in [0.5, 1.0, 2.0] {
            let mut s = engine(1);
            s.configure_policy(PolicyConfig { adaptive_verification_alpha: alpha, ..Default::default() }).unwrap();
            let h = ready(&mut s);
            let raw = s.speculative_curve(s.request(h).unwrap(), false);
            let score = 1.0 + 0.5f64.powf(alpha) + 0.25f64.powf(alpha);
            assert_eq!(s.proposal_forecast_tokens(s.request(h).unwrap(), 3, false), score);
            assert_eq!(s.expected_speculative_tokens(s.request(h).unwrap(), 3), score);
            retain(&mut s, h, [0.5; 7]);
            s.finalize_proposals(1 << h.slot, None, &s.proposal_decision()).unwrap();
            assert_eq!(s.speculative_curve(s.request(h).unwrap(), false), raw);
            assert_eq!(s.split.as_ref().unwrap().population.map(f64::from), raw);
            assert_eq!(s.retained_proposal(s.request(h).unwrap()).unwrap().confidence, [0.5; 7]);
            assert_eq!(expected(&[0.5; 7], 3, alpha), score);
            assert_eq!(continuation_tokens(&[0.5; 7], &raw, 1, 3, alpha), score);
            assert_eq!(prefix_value(0.0, alpha), 0.0);
            assert_eq!(prefix_value(1.0, alpha), 1.0);
        }
        for alpha in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(engine(1).configure_policy(PolicyConfig {
                adaptive_verification_alpha: alpha, ..Default::default()
            }).is_err());
        }
    }

    #[test]
    fn forecasting_averages_joint_prefixes_before_summing_token_value() {
        let mut s = engine(1);
        let h = ready(&mut s);
        let prior = survival(&[FORECAST_PRIOR; 7]);
        let mut reference = prior;
        for (index, confidence) in [[1.0; 7], [0.0; 7], [1.0; 7]].into_iter().enumerate() {
            retain(&mut s, h, confidence);
            s.finalize_proposals(1 << h.slot, None, &s.proposal_decision()).unwrap();
            let observation = survival(&confidence);
            reference = std::array::from_fn(|i| reference[i] + (observation[i] - reference[i]) / (3 + index) as f32);
        }
        s.split.as_mut().unwrap().retained[h.slot as usize] = None;
        for width in 1..=8 {
            assert!((s.expected_speculative_tokens(s.request(h).unwrap(), width)
                - forecast_tokens(&reference, width, 1.0)).abs() < 1e-6);
        }
        // Two perfectly correlated Bernoulli positions have E[p1*p2] = 0.5;
        // multiplying their separate means would incorrectly predict 0.25.
        assert_eq!(forecast_tokens(&[0.5; 7], 3, 1.0), 2.0);
        assert_eq!(expected(&[0.5; 7], 3, 1.0), 1.75);
    }

    #[test]
    fn an_unprofitable_population_does_not_lock_out_confidence_refresh() {
        let mut s = measured_split_engine(8);
        for batch in [1, 2, 4, 8] {
            for (kind, width, nanos) in [(GraphKind::Speculative, 1, 10_000_000),
                                        (GraphKind::Proposal, 6, 2_000_000)] {
                s.costs.set_measured(&ScheduledStep { draft_tokens: 0, packed_token_capacity: 0, kind, batch_capacity: batch,
                    token_capacity: width, mixed: None, lanes: vec![], requires_feedback: true }, nanos);
            }
        }
        let handles: Vec<_> = (0..8).map(|_| ready(&mut s)).collect();
        s.split.as_mut().unwrap().population = [0.0; 7];
        for &h in &handles {
            s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast::new(h, [0.0; 7], 1));
        }
        assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
        for &h in &handles {
            let request = s.request_mut(h).unwrap();
            request.max_new_tokens = 1024;
            request.generated_tokens += 128;
        }
        s.refresh.acceptance_credit_ns = refresh::CREDIT_CAP_NS;
        let refresh = s.plan().unwrap();
        assert_eq!(refresh.origin, PlanOrigin::AcceptanceStart);
        assert_eq!(refresh.kind, GraphKind::Proposal);
        assert_eq!(refresh.lanes.len(), 8);
        let mut fresh = 0;
        for h in handles {
            retain(&mut s, h, [0.0; 7]);
            s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 6;
            fresh |= 1 << h.slot;
        }
        s.finalize_proposals(fresh, None, &s.proposal_decision()).unwrap();
        assert!(s.split.as_ref().unwrap().retained.iter().flatten().all(|p| p.width == 1));
    }

    #[test]
    fn cold_requests_use_population_but_observed_requests_keep_their_mean() {
        let mut s = engine(2);
        let proposer = ready(&mut s);
        let quiet = ready(&mut s);
        for _ in 0..256 {
            retain(&mut s, proposer, [0.9; 7]);
            s.finalize_proposals(1 << proposer.slot, None, &s.proposal_decision()).unwrap();
        }
        let tokens = s.expected_speculative_tokens(s.request(quiet).unwrap(), 2);
        assert!((tokens - 1.9).abs() < 0.01, "{tokens}");
        let generated = s.request(quiet).unwrap().generated_tokens;
        s.split.as_mut().unwrap().forecast[quiet.slot as usize] = Some(Forecast::new(quiet, [0.0; 7], generated));
        s.request_mut(quiet).unwrap().generated_tokens += 8 * 128;
        let tokens = s.expected_speculative_tokens(s.request(quiet).unwrap(), 2);
        assert_eq!(tokens, 1.0);
        assert!(s.proposal_forecast_tokens(s.request(quiet).unwrap(), 2, true) > 1.85);
    }

    #[test]
    fn historical_continuations_cross_cost_cliffs_without_conditioning_on_the_admitted_token() {
        let mut zero_outputs = 0;
        for draft in 0..=1 {
            for bonus in 0..=1 {
                let mut s = engine(1);
                s.configure_policy(PolicyConfig::throughput()).unwrap();
                let h = ready(&mut s);
                costs(&mut s, 1, &[1_000_000, 1_700_000, 2_000_000, 99_000_000,
                    99_000_000, 99_000_000, 99_000_000, 99_000_000]);
                s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast::new(h,
                    [0.6, 0.54, 0.0, 0.0, 0.0, 0.0, 0.0], 1));
                let decision = s.proposal_decision();
                retain(&mut s, h, [0.6, if draft == 0 { 0.9 } else { 0.1 }, 0.0, 0.0, 0.0, 0.0, 0.0]);
                s.split.as_mut().unwrap().retained[h.slot as usize].as_mut().unwrap().limit = 3;
                s.finalize_proposals(1 << h.slot, None, &decision).unwrap();
                let width = s.retained_proposal(s.request(h).unwrap()).unwrap().width;
                assert!(width >= 2);
                let output = if width >= 2 { draft } else { bonus };
                zero_outputs += usize::from(output == 0);
            }
        }
        assert_eq!(zero_outputs, 2);
    }

    #[test]
    fn packed_frontiers_use_frozen_service_prices_and_cannot_skip_a_prefix() {
        let mut s = engine(2);
        let expensive = ready(&mut s);
        let confident = ready(&mut s);
        let mut decision = s.proposal_decision();
        decision.values[expensive.slot as usize].output_token = 4.0;
        decision.values[confident.slot as usize].output_token = 1.0;
        for tail in [0.0, 1.0] {
            retain(&mut s, expensive, [0.4, tail, tail, tail, tail, tail, tail]);
            retain(&mut s, confident, [0.9; 7]);
            s.finalize_packed((1 << expensive.slot) | (1 << confident.slot), 3, &decision).unwrap();
            assert_eq!(s.retained_proposal(s.request(expensive).unwrap()).unwrap().width, 2);
            assert_eq!(s.retained_proposal(s.request(confident).unwrap()).unwrap().width, 1);
        }
    }

    #[test]
    fn proposal_completion_uses_the_decision_frozen_before_dispatch() {
        let mut s = engine(1);
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        let h = ready(&mut s);
        costs(&mut s, 1, &[1_000_000, 1_700_000, 2_000_000, 99_000_000,
            99_000_000, 99_000_000, 99_000_000, 99_000_000]);
        s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast::new(h,
            [0.6, 0.54, 0.0, 0.0, 0.0, 0.0, 0.0], 1));
        measured_key(&mut s, cost::Key { drafts: 0, kind: GraphKind::Speculative, batch: 1,
            tokens: 1, prefill: 0, packed: 0 }, 2_000_000);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Proposal);
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        costs(&mut s, 1, &[1_000_000, 100_000_000, 100_000_000, 100_000_000,
            100_000_000, 100_000_000, 100_000_000, 100_000_000]);
        s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast::new(h, [0.0; 7], 1));
        let records = [CompletionRecord { generation: h.generation as i64,
            submission_sequence: fence.0 as i64, consumed_tokens: 65, generated_tokens: 1,
            token_id: -1, status: 1, vector: Some(TokenVector::default()), ..Default::default() }];
        s.complete_proposal(fence, &records, capture_time(1_000_000),
            &[[0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).unwrap();
        assert_eq!(s.retained_proposal(s.request(h).unwrap()).unwrap().width, 2);
        assert!(s.split.as_ref().unwrap().pending_decision.is_none());
    }

    #[test]
    fn delivery_references_do_not_batch_incompatible_retained_widths() {
        let mut s = engine(2);
        let narrow = ready(&mut s);
        let wide = ready(&mut s);
        for h in [narrow, wide] { retain(&mut s, h, [0.5; 7]); }
        s.split.as_mut().unwrap().retained[wide.slot as usize].as_mut().unwrap().width = 8;
        costs(&mut s, 1, &[100, 200, 300, 400, 500, 600, 700, 800]);
        costs(&mut s, 2, &[100, 200, 300, 400]);
        let mut rates = [None; REQUEST_LANES];
        rates[narrow.slot as usize] = s.service_rates(s.request(narrow).unwrap()).1;
        rates[wide.slot as usize] = s.service_rates(s.request(wide).unwrap()).1;
        let references = s.delivery_reference_intervals(&rates);
        assert_eq!(references[narrow.slot as usize], Some(0.0002));
        assert_eq!(references[wide.slot as usize], Some(0.0016));
    }

    #[test]
    fn causal_frontier_never_uses_a_suffix_to_cross_an_unprofitable_extension() {
        for tail in [0.0, 1.0] {
            let mut s = engine(1);
            let h = ready(&mut s);
            costs(&mut s, 1, &[100, 300, 210, 210, 210, 210, 210, 210]);
            s.split.as_mut().unwrap().forecast[h.slot as usize] = Some(Forecast::new(h,
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1));
            retain(&mut s, h, [1.0, tail, tail, tail, tail, tail, tail]);
            s.finalize_proposals(1 << h.slot, None, &s.proposal_decision()).unwrap();
            assert_eq!(s.retained_proposal(s.request(h).unwrap()).unwrap().width, 1);
        }
    }

    #[test]
    fn shared_verification_cost_can_make_a_cohort_extension_profitable() {
        let mut s = engine(2);
        let first = ready(&mut s);
        let second = ready(&mut s);
        for handle in [first, second] {
            retain(&mut s, handle, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        costs(&mut s, 1, &[10_000_000, 14_000_000, 100_000_000,
            100_000_000, 100_000_000, 100_000_000, 100_000_000, 100_000_000]);
        costs(&mut s, 2, &[10_000_000, 14_000_000, 100_000_000, 100_000_000]);
        let (values, age_price, _) = s.lane_values(0);
        let score = |widths: [usize; 2]| {
            let needs = [(first.slot as usize, widths[0]), (second.slot as usize, widths[1])];
            let packed = s.pack(GraphKind::Verification, &needs).unwrap();
            let value = needs.iter().map(|&(slot, width)| {
                values[slot].decode(expected(&[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], width, 1.0),
                    packed.nanos as f64 / 1e6)
            }).sum();
            ValueRate::new(packed.nanos, value, age_price)
        };
        assert!(score([2, 2]).compare(score([1, 1])).is_gt());
        assert!(score([2, 1]).compare(score([1, 1])).is_lt());
        s.finalize_proposals((1 << first.slot) | (1 << second.slot), None, &s.proposal_decision()).unwrap();
        let selected: Vec<_> = [first, second].into_iter().map(|h|
            s.retained_proposal(s.request(h).unwrap()).unwrap().width).collect();
        assert_eq!(selected, [2, 2]);
    }

    #[test]
    fn conditional_confidence_is_multiplied_and_replaces_acceptance_history() {
        let mut s = engine(1);
        let h = ready(&mut s);
        retain(&mut s, h, [0.8, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0]);
        s.finalize_proposals(1 << h.slot, None, &s.proposal_decision()).unwrap();
        assert_eq!(s.retained_proposal(s.request(h).unwrap()).unwrap().width, 3);
        assert!((s.expected_speculative_tokens(s.request(h).unwrap(), 8) - 2.2).abs() < 1e-6);
    }

    #[test]
    fn unmeasured_trial_width_does_not_depend_on_fresh_confidence() {
        let mut chosen = Vec::new();
        for p in [0.0, 1.0] {
            let mut s = engine(1);
            let h = ready(&mut s);
            costs(&mut s, 1, &[100, 100, 0, 100, 100, 100, 100, 100]);
            retain(&mut s, h, [p; 7]);
            s.finalize_proposals(1 << h.slot, None, &s.proposal_decision()).unwrap();
            chosen.push(s.retained_proposal(s.request(h).unwrap()).unwrap().width);
        }
        assert_eq!(chosen, [3, 3]);
    }

    #[test]
    fn multiple_proposals_regroup_without_changing_finalized_prefixes() {
        let mut s = engine(4);
        let handles: Vec<_> = (0..4).map(|_| ready(&mut s)).collect();
        proposal(&mut s, &handles[..2], &[[1.0; 7], [0.0; 7]]);
        assert_eq!(s.request(handles[0]).unwrap().consumed_tokens, 65);
        proposal(&mut s, &handles[2..], &[[0.0; 7], [1.0; 7]]);
        let needs = s.retained_needs();
        assert_eq!(needs.iter().map(|r| r.1).collect::<Vec<_>>(), [4, 1, 1, 4]);
        let packing = s.pack(GraphKind::Verification, &needs).unwrap();
        assert_eq!(packing.calls.len(), 2);
        assert!(packing.calls.iter().any(|c| c.mask == (1 << handles[1].slot) | (1 << handles[2].slot)));
        let mut seen = 0;
        for call in packing.calls {
            assert_eq!(seen & call.mask, 0);
            seen |= call.mask;
            for slot in bits(call.mask) { assert!(needs[slot].1 <= call.width); }
        }
        assert_eq!(seen, 15);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Verification);
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        assert!(s.commit(&plan, Fence(fence.0 + 1)).is_err());
        for lane in &plan.lanes { assert!(s.split.as_ref().unwrap().retained[lane.request.slot as usize].is_none()); }
    }

    #[test]
    fn proposal_is_a_prerequisite_with_no_progress_credit() {
        let mut s = engine(2);
        let h = ready(&mut s);
        let before = s.request(h).unwrap().accounting;
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Proposal);
        proposal(&mut s, &[h], &[[0.9; 7]]);
        let after = s.request(h).unwrap().accounting;
        assert_eq!(before.committed_output_tokens, after.committed_output_tokens);
        assert_eq!(before.published_deliveries, after.published_deliveries);
        assert!(s.plan().unwrap().kind == GraphKind::Verification);
    }

    #[test]
    fn proposal_storage_is_bounded_and_cancellation_does_not_follow_slot_reuse() {
        let mut s = engine(1);
        let h = ready(&mut s);
        proposal(&mut s, &[h], &[[1.0; 7]]);
        s.cancel(h).unwrap();
        let replacement = ready(&mut s);
        assert_eq!(replacement.slot, h.slot);
        assert_ne!(replacement.generation, h.generation);
        assert!(!s.owns_proposal(s.request(replacement).unwrap()));
        let prior = forecast_tokens(&s.split.as_ref().unwrap().population, 8, 1.0);
        assert!((s.expected_speculative_tokens(s.request(replacement).unwrap(), 8) - prior).abs() < 1e-10);
        assert!(prior < 2.1, "one proposal moves the population prior only slightly: {prior}");
    }

    #[test]
    fn retained_proposal_cannot_be_preempted_for_another_request() {
        let mut s = engine(2);
        let first = ready(&mut s);
        let last = ready(&mut s);
        s.request_mut(first).unwrap().growable_output = true;
        s.request_mut(last).unwrap().growable_output = true;
        proposal(&mut s, &[last], &[[1.0; 7]]);
        s.preempt_newest().unwrap();
        assert!(s.request(first).unwrap().preempted);
        assert!(!s.request(last).unwrap().preempted);
        assert!(s.owns_proposal(s.request(last).unwrap()));
    }

    fn measured_key(s: &mut RequestScheduler, key: cost::Key, nanos: u64) {
        s.costs.set_measured(&ScheduledStep { draft_tokens: 0, kind: key.kind, batch_capacity: key.batch,
            token_capacity: key.tokens, packed_token_capacity: key.packed, mixed: None,
            lanes: vec![], requires_feedback: true }, nanos);
    }

    fn stale_only(s: &mut RequestScheduler, target: cost::Key) {
        s.advance_to(s.policy.estimate_ttl_ns).unwrap();
        let other: Vec<_> = s.costs.stale_keys().filter(|key| *key != target)
            .map(|key| (key, s.costs.duration_key(key).unwrap())).collect();
        for (key, nanos) in other { measured_key(s, key, nanos); }
        s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    }

    fn capture_time(nanos: u64) -> crate::execution::ExecutionTiming {
        let mut gpu_ns = [None; 16];
        gpu_ns[0] = Some(nanos);
        crate::execution::ExecutionTiming { gpu_ns, wall_ns: nanos, ..Default::default() }
    }

    fn finish_refresh_proposal(s: &mut RequestScheduler, plan: &ScheduledPlan,
        confidence: &[[f32; 7]], nanos: u64) {
        let fence = Fence(s.submitted.0 + 1);
        s.commit(plan, fence).unwrap();
        let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
        for lane in &plan.lanes {
            let request = s.request(lane.request).unwrap();
            records[lane.request.slot as usize] = CompletionRecord {
                generation: lane.request.generation as i64, submission_sequence: fence.0 as i64,
                consumed_tokens: request.consumed_tokens as i64, generated_tokens: request.generated_tokens as i64,
                token_id: -1, status: 1, vector: Some(TokenVector::default()), ..Default::default()
            };
        }
        s.advance_to(s.now_ns + nanos).unwrap();
        s.complete_proposal(fence, &records, capture_time(nanos), confidence).unwrap();
    }

    fn finish_refresh_verification(s: &mut RequestScheduler, plan: &ScheduledPlan, nanos: u64) {
        let fence = Fence(s.submitted.0 + 1);
        s.commit(plan, fence).unwrap();
        let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
        for lane in &plan.lanes {
            let request = s.request(lane.request).unwrap();
            let mut vector = TokenVector { emitted_count: 1, offered_drafts: (lane.tokens - 1) as i64,
                verification_width: plan.token_capacity as i64, ..Default::default() };
            vector.tokens[0] = 1;
            records[lane.request.slot as usize] = CompletionRecord {
                generation: lane.request.generation as i64, submission_sequence: fence.0 as i64,
                consumed_tokens: lane.start_position as i64 + 1,
                generated_tokens: request.generated_tokens as i64 + 1,
                token_id: 1, status: 1, vector: Some(vector), ..Default::default()
            };
        }
        s.advance_to(s.now_ns + nanos).unwrap();
        s.complete(fence, &records, capture_time(nanos)).unwrap();
    }

    #[test]
    fn refresh_reaches_a_dominated_proposal_and_pays_for_each_verification_capture() {
        let mut s = engine(4);
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        let handles: Vec<_> = (0..4).map(|_| ready(&mut s)).collect();
        let target = cost::Key { drafts: 0, kind: GraphKind::Proposal, batch: 4, tokens: 8, prefill: 0, packed: 0 };
        measured_key(&mut s, target, 10_000_000);
        costs(&mut s, 1, &[10_000_000; 8]);
        costs(&mut s, 2, &[100, 100, 100, 100]);
        stale_only(&mut s, target);
        let plan = s.plan().unwrap();
        assert_eq!(plan.origin, PlanOrigin::RefreshStart(target));
        assert_eq!((plan.kind, plan.batch_capacity, plan.token_capacity), (GraphKind::Proposal, 4, 8));
        assert_eq!(plan.lanes.len(), 4);
        finish_refresh_proposal(&mut s, &plan, &[[1.0; 7]; 4], 10_000_000);
        assert_eq!(s.refresh.credit_ns, -5_000_000.0);
        let widths: Vec<_> = handles.iter().map(|&handle|
            s.retained_proposal(s.request(handle).unwrap()).unwrap().width).collect();
        assert_eq!(widths, [4; 4]);
        let episode = s.refresh_episode().unwrap().id;
        let mut seen = 0;
        for _ in 0..2 {
            let verify = s.plan().unwrap();
            assert_eq!(verify.origin, PlanOrigin::RefreshContinuation(episode));
            assert_eq!((verify.kind, verify.batch_capacity, verify.token_capacity), (GraphKind::Verification, 2, 4));
            assert_eq!(verify.lanes.len(), 2);
            for lane in &verify.lanes { assert_eq!(seen & (1 << lane.request.slot), 0); seen |= 1 << lane.request.slot; }
            finish_refresh_verification(&mut s, &verify, 2_000_000);
        }
        assert_eq!(seen, 15);
        assert!(s.refresh_episode().is_none());
        assert_eq!(s.refresh.credit_ns, -9_000_000.0);
        assert!(!matches!(s.plan().unwrap().origin, PlanOrigin::RefreshStart(_)));
    }

    #[test]
    fn verification_refresh_enables_only_fresh_trial_widths_and_survives_cancelled_rows() {
        let mut s = engine(4);
        let old = ready(&mut s);
        retain(&mut s, old, [1.0; 7]);
        s.split.as_mut().unwrap().retained[old.slot as usize].as_mut().unwrap().width = 8;
        let fresh: Vec<_> = (0..3).map(|_| ready(&mut s)).collect();
        let target = cost::Key { drafts: 0, kind: GraphKind::Verification, batch: 2, tokens: 4, prefill: 0, packed: 0 };
        measured_key(&mut s, target, 100_000_000);
        stale_only(&mut s, target);
        let plan = s.plan().unwrap();
        assert_eq!(plan.origin, PlanOrigin::RefreshStart(target));
        assert_eq!(plan.kind, GraphKind::Proposal);
        assert_eq!(plan.lanes.len(), 2);
        finish_refresh_proposal(&mut s, &plan, &[[0.0; 7]; 4], 8_000_000);
        assert_eq!(s.retained_proposal(s.request(old).unwrap()).unwrap().width, 8);
        assert!(plan.lanes.iter().all(|lane| s.retained_proposal(s.request(lane.request).unwrap()).unwrap().width == 4));
        let cancelled = plan.lanes[0].request;
        s.cancel(cancelled).unwrap();
        let verify = s.plan().unwrap();
        assert_eq!((verify.kind, verify.batch_capacity, verify.token_capacity), (GraphKind::Verification, 2, 4));
        assert_eq!(verify.lanes.len(), 1);
        assert_ne!(verify.lanes[0].request, cancelled);
        assert!(!verify.lanes.iter().any(|lane| lane.request == old));
        finish_refresh_verification(&mut s, &verify, 100_000_000);
        assert!(s.refresh_episode().is_none());
        assert_eq!(s.refresh.credit_ns, -103_000_000.0);
        assert!(fresh.iter().any(|&handle| s.request(handle).is_ok_and(|r| !s.owns_proposal(r))));
    }

    #[test]
    fn proposal_refresh_charges_a_shared_capture_without_changing_an_older_prefix() {
        let mut s = engine(2);
        let old = ready(&mut s);
        retain(&mut s, old, [1.0; 7]);
        s.split.as_mut().unwrap().retained[old.slot as usize].as_mut().unwrap().width = 4;
        let fresh = ready(&mut s);
        let target = cost::Key { drafts: 0, kind: GraphKind::Proposal, batch: 1, tokens: 8, prefill: 0, packed: 0 };
        measured_key(&mut s, target, 10_000_000);
        costs(&mut s, 2, &[100, 100, 100, 100]);
        stale_only(&mut s, target);
        let plan = s.plan().unwrap();
        assert_eq!(plan.origin, PlanOrigin::RefreshStart(target));
        assert_eq!(plan.lanes.iter().map(|lane| lane.request).collect::<Vec<_>>(), [fresh]);
        finish_refresh_proposal(&mut s, &plan, &[[1.0; 7]], 10_000_000);
        assert_eq!(s.retained_proposal(s.request(old).unwrap()).unwrap().width, 4);
        assert_eq!(s.retained_proposal(s.request(fresh).unwrap()).unwrap().width, 4);
        let verify = s.plan().unwrap();
        assert!(matches!(verify.origin, PlanOrigin::RefreshContinuation(_)));
        assert_eq!(verify.lanes.len(), 2);
        assert!(verify.lanes.iter().any(|lane| lane.request == old));
        finish_refresh_verification(&mut s, &verify, 3_000_000);
        assert_eq!(s.refresh.credit_ns, -8_000_000.0);
        assert!(s.refresh_episode().is_none());
    }

    #[test]
    fn packed_refresh_preserves_a_dominated_budget_and_physical_width_limit() {
        let mut s = engine(4);
        let mut graphs = super::super::tests::duration_graphs(4);
        graphs.retain(|graph| graph.kind != GraphKind::Speculative);
        for batch in [1, 2, 4] {
            for (kind, width, budget) in [(GraphKind::Speculative, 1, 0), (GraphKind::Proposal, 8, 0)]
                .into_iter().chain([2, 4, 8].map(|width| (GraphKind::Verification, width, batch * width))) {
                graphs.push(GraphDescriptor { draft_tokens: 0, name: format!("{kind:?}-{batch}-{width}-{budget}"), kind,
                    sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                    token_capacity: width, packed_token_capacity: budget, prefill_capacity: 0,
                    estimated_duration_ns: 1, input_bindings: vec![] });
            }
        }
        s.measured_graph_durations(&graphs).unwrap();
        for graph in &graphs {
            measured_key(&mut s, cost::Key { drafts: 0, kind: graph.kind, batch: graph.batch_capacity,
                tokens: graph.token_capacity, prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity }, graph.estimated_duration_ns);
        }
        let old = ready(&mut s);
        retain(&mut s, old, [1.0; 7]);
        s.split.as_mut().unwrap().retained[old.slot as usize].as_mut().unwrap().width = 8;
        for _ in 0..3 { ready(&mut s); }
        let target = cost::Key { drafts: 0, kind: GraphKind::Verification, batch: 4, tokens: 2, prefill: 0, packed: 8 };
        measured_key(&mut s, target, 100_000_000);
        stale_only(&mut s, target);
        let plan = s.plan().unwrap();
        assert_eq!(plan.origin, PlanOrigin::RefreshStart(target));
        assert_eq!((plan.kind, plan.token_capacity, plan.verification_budget), (GraphKind::Proposal, 8, Some(8)));
        finish_refresh_proposal(&mut s, &plan, &[[1.0; 7]; 4], 8_000_000);
        assert_eq!(s.retained_proposal(s.request(old).unwrap()).unwrap().width, 8);
        for lane in &plan.lanes {
            let proposal = s.retained_proposal(s.request(lane.request).unwrap()).unwrap();
            assert_eq!((proposal.width, proposal.limit), (2, 8));
            assert!(s.split.as_ref().unwrap().forecast[lane.request.slot as usize].unwrap().prefixes[6].mean() > 0.0);
        }
        let verify = s.plan().unwrap();
        assert_eq!((verify.kind, verify.batch_capacity, verify.token_capacity, verify.packed_token_capacity),
            (GraphKind::Verification, 4, 2, 8));
        assert_eq!(verify.lanes.len(), 3);
        finish_refresh_verification(&mut s, &verify, 100_000_000);
        assert!(s.refresh_episode().is_none());
        assert_eq!(s.refresh.credit_ns, -103_000_000.0);
    }

    #[test]
    fn legacy_fixed_producer_can_refresh_a_wider_rectangular_or_packed_verifier() {
        for packed in [0, 8] {
            let mut s = engine(1);
            let mut graphs = super::super::tests::duration_graphs(1);
            graphs.retain(|graph| graph.kind != GraphKind::Speculative);
            for (kind, width, budget) in [(GraphKind::Speculative, 1, 0),
                (GraphKind::Proposal, 2, 0), (GraphKind::Verification, 8, packed)] {
                graphs.push(GraphDescriptor { draft_tokens: 0, name: format!("{kind:?}-{width}-{budget}"), kind,
                    sampling_mode: SamplingMode::General, handle: 1, batch_capacity: 1,
                    token_capacity: width, packed_token_capacity: budget, prefill_capacity: 0,
                    estimated_duration_ns: 1, input_bindings: vec![] });
            }
            s.graph_durations(&graphs).unwrap();
            for graph in &graphs {
                measured_key(&mut s, cost::Key { drafts: 0, kind: graph.kind, batch: graph.batch_capacity,
                    tokens: graph.token_capacity, prefill: graph.prefill_capacity,
                    packed: graph.packed_token_capacity }, graph.estimated_duration_ns);
            }
            let handle = ready(&mut s);
            let target = cost::Key { drafts: 0, kind: GraphKind::Verification, batch: 1, tokens: 8, prefill: 0, packed };
            measured_key(&mut s, target, 100_000_000);
            stale_only(&mut s, target);
            let plan = s.plan().unwrap();
            assert_eq!(plan.origin, PlanOrigin::RefreshStart(target));
            assert_eq!((plan.kind, plan.token_capacity, plan.lanes[0].tokens), (GraphKind::Proposal, 2, 2));
            finish_refresh_proposal(&mut s, &plan, &[[1.0; 7]], 8_000_000);
            let proposal = s.retained_proposal(s.request(handle).unwrap()).unwrap();
            assert_eq!((proposal.width, proposal.limit), (2, 2));
            let verify = s.plan().unwrap();
            assert_eq!((verify.kind, verify.token_capacity, verify.packed_token_capacity, verify.lanes[0].tokens),
                (GraphKind::Verification, 8, packed, 2));
            finish_refresh_verification(&mut s, &verify, 100_000_000);
            assert_eq!(s.refresh.credit_ns, -103_000_000.0);
            assert!(s.refresh_episode().is_none());
        }
    }

    #[test]
    fn packed_frontier_reuse_preserves_heterogeneous_refresh_candidates_exactly() {
        for fixed in [None, Some(4)] {
            let mut s = engine(8);
            s.fixed_speculative_width(fixed).unwrap();
            let mut graphs = super::super::tests::duration_graphs(8);
            graphs.retain(|graph| graph.kind != GraphKind::Speculative);
            for batch in [1, 2, 4, 8] {
                for (kind, width, budget) in [(GraphKind::Speculative, 1, 0)].into_iter()
                    .chain([(GraphKind::Proposal, 8, 0)])
                    .chain([2, 4, 8].into_iter().flat_map(|width|
                        (1..=width).map(move |budget| (GraphKind::Verification, width, batch * budget)))) {
                    graphs.push(GraphDescriptor { draft_tokens: 0, name: format!("{kind:?}-{batch}-{width}-{budget}"), kind,
                        sampling_mode: SamplingMode::General, handle: 1, batch_capacity: batch,
                        token_capacity: width, packed_token_capacity: budget, prefill_capacity: 0,
                        estimated_duration_ns: (batch + width + budget) as u64 * 100_000,
                        input_bindings: vec![] });
                }
            }
            s.graph_durations(&graphs).unwrap();
            for graph in &graphs {
                measured_key(&mut s, cost::Key { drafts: 0, kind: graph.kind, batch: graph.batch_capacity,
                    tokens: graph.token_capacity, prefill: graph.prefill_capacity,
                    packed: graph.packed_token_capacity }, graph.estimated_duration_ns);
            }
            let mut handles = Vec::new();
            for (index, remaining) in [2, 3, 4, 5, 7, 9].into_iter().enumerate() {
                let handle = ready(&mut s);
                let request = s.request_mut(handle).unwrap();
                request.max_new_tokens = request.generated_tokens + remaining;
                let probability = [0.15, 0.91, 0.43, 0.77, 0.28, 0.64][index];
                s.split.as_mut().unwrap().forecast[handle.slot as usize] = Some(Forecast::new(handle, survival(&[probability; 7]), 1));
                handles.push(handle);
            }
            s.advance_to(s.policy.estimate_ttl_ns).unwrap();
            let (mut values, age, _) = s.lane_values(0);
            for (index, handle) in handles.iter().enumerate() {
                values[handle.slot as usize].output_token *= [4.0, 1.0, 7.0, 2.0, 5.0, 3.0][index];
                values[handle.slot as usize].delivery += index as f64;
            }
            let mut cache = RefreshBudgetCache::default();
            let mut total = 0;
            for target in s.costs.stale_keys().filter(|key|
                matches!(key.kind, GraphKind::Proposal | GraphKind::Verification)) {
                let mut reference = Vec::new();
                let mut reused = Vec::new();
                s.split_refresh_candidates(&mut reference, target, &values, age, None);
                s.split_refresh_candidates(&mut reused, target, &values, age, Some(&mut cache));
                assert_eq!(reference.len(), reused.len());
                for (a, b) in reference.iter().zip(&reused) {
                    let signature = |candidate: &Candidate| (candidate.cost_key(), candidate.mask,
                        candidate.duration_ns, candidate.verification_budget, candidate.rate.duration_ns,
                        candidate.rate.score.to_bits(), candidate.feedback);
                    assert_eq!(signature(a), signature(b));
                }
                total += reference.len();
            }
            assert!(total > 100 && cache.scores.len() > 10);
        }
    }
    include!("split/packed_round_tests.rs");

}
