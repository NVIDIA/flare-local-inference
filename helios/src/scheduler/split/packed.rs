//! Select a target-row budget before sampling, then allocate causal prefixes within it.

use super::*;
use crate::scheduler::cost::PackedShape;

impl RequestScheduler {
    fn budget_choice(&self, offers: &[(usize, usize)], batch: usize, width: usize,
        values: &[LaneValue; REQUEST_LANES], age_price: f64, optimistic: bool) -> Option<(PackedShape, ValueRate)> {
        self.refresh_budget_choice(offers, batch, width, values, age_price,
            cost::Key { drafts: 0, kind: GraphKind::Proposal, batch, tokens: width, prefill: 0, packed: 0 }, None, None, optimistic)
    }

    pub(super) fn refresh_budget_choice(&self, offers: &[(usize, usize)], batch: usize, width: usize,
        values: &[LaneValue; REQUEST_LANES], age_price: f64, proposal: cost::Key,
        target: Option<cost::Key>, cache: Option<&mut RefreshBudgetCache>, optimistic: bool) -> Option<(PackedShape, ValueRate)> {
        let proposal_ns = self.costs.duration_key(proposal)?;
        let frontier = || -> Option<Vec<(f64, usize, usize, f64)>> {
            let mut scores = Vec::new();
            for &(slot, limit) in offers {
                let request = self.slots[slot].request.as_ref()?;
                let mut before = 1.0;
                let limit = target.map_or(limit, |shape| limit.min(shape.tokens));
                for position in 0..limit.saturating_sub(1) {
                    let total = self.proposal_forecast_tokens(request, position + 2, optimistic);
                    scores.push((values[slot].output_token * (total - before), position, slot, total - before));
                    before = total;
                }
            }
            scores.sort_unstable_by(|a,b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            Some(scores)
        };
        let uncached;
        let scores = if let Some(cache) = cache {
            // Logical width fixes each row's limit within this immutable decision.
            let mask = offers.iter().fold(0, |mask, (slot, _)| mask | (1 << slot));
            let key = (target.map_or(0, |shape| shape.tokens), width, mask);
            match cache.scores.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry.insert(frontier()?),
            }
        } else {
            uncached = frontier()?;
            &uncached
        };
        let matches = |shape: &PackedShape| {
            let matching = target.map_or(shape.batch == batch && shape.width >= width, |target|
                shape.batch == target.batch && shape.width == target.tokens && shape.tokens == target.packed);
            matching && shape.batch >= offers.len() && shape.tokens >= offers.len()
        };
        let useful_tokens: usize = offers.iter().map(|(_, limit)| limit).sum();
        // A padded budget cannot be discovered when verification dispatch always chooses its smaller equivalent.
        let full_budget = target.is_none().then(|| self.costs.packed_shapes().filter(matches)
            .filter(|shape| shape.tokens >= useful_tokens).map(|shape| shape.tokens).min()).flatten();
        self.costs.packed_shapes().filter(matches)
            .filter(|shape| full_budget.is_none_or(|budget| shape.tokens <= budget)).map(|shape| {
            let nanos = horizon(proposal_ns, shape.nanos);
            let mut expected = [0.0; REQUEST_LANES];
            for &(slot, _) in offers { expected[slot] = 1.0; }
            for &(_, _, slot, probability) in scores.iter().take(shape.tokens - offers.len()) { expected[slot] += probability; }
            let value = offers.iter().map(|&(slot,_)| values[slot].decode(expected[slot], nanos as f64 / 1e6)).sum();
            (shape, ValueRate::new(nanos, value, age_price))
        }).max_by(|(a,ar),(b,br)| ar.compare(*br).then(b.tokens.cmp(&a.tokens)))
    }

    pub(super) fn finalize_packed(&mut self, fresh: u64, budget: usize, decision: &ProposalDecision) -> Result<()> {
        self.finalize_packed_with_width(fresh, budget, None, decision)
    }

    pub(super) fn finalize_packed_with_width(&mut self, fresh: u64, budget: usize,
        maximum: Option<usize>, decision: &ProposalDecision) -> Result<()> {
        let mut needs = self.retained_needs();
        let count = fresh.count_ones() as usize;
        if count > budget { return Err("proposal row budget cannot cover its pending inputs".into()); }
        let mut survival = [1.0f64; REQUEST_LANES];
        for (slot,width) in &mut needs {
            if fresh & (1 << *slot) != 0 { *width = 1; }
        }
        for _ in 0..budget - count {
            // Only a lane's immediate frontier is visible. A later confidence
            // cannot influence admission of the token on which it depends.
            let best = needs.iter().enumerate().filter_map(|(index,&(slot,width))| {
                let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                (fresh & (1 << slot) != 0 && width < maximum.map_or(proposal.limit,
                    |maximum| proposal.limit.min(maximum))).then(||
                    (index, decision.values[slot].output_token * survival[slot] * f64::from(proposal.confidence[width-1]), width, slot))
            }).max_by(|a,b| a.1.total_cmp(&b.1).then(b.2.cmp(&a.2)).then(b.3.cmp(&a.3)));
            let Some((index,_,width,slot)) = best else { break; };
            survival[slot] *= f64::from(self.split.as_ref().unwrap().retained[slot].unwrap().confidence[width - 1]);
            needs[index].1 += 1;
        }
        self.finalize_selected(fresh, needs);
        Ok(())
    }

    pub(super) fn packed_candidates(&self, output: &mut Vec<Candidate>, values: &[LaneValue; REQUEST_LANES], age: f64, optimistic: bool) {
        let retained = self.retained_needs();
        for shape in self.costs.packed_shapes() {
            let mut offers: Vec<_> = retained.iter().copied().filter(|(_,width)| *width <= shape.width).collect();
            offers.sort_unstable_by(|&(a,aw),&(b,bw)| {
                let value = |slot: usize, width: usize| {
                    let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                    values[slot].decode(expected(&proposal.confidence,width,self.policy.adaptive_verification_alpha), shape.nanos as f64 / 1e6)
                };
                (value(b,bw)/bw as f64).total_cmp(&(value(a,aw)/aw as f64)).then(a.cmp(&b))
            });
            let (mut mask,mut tokens,mut count,mut value) = (0u64,0usize,0usize,0.0);
            for (slot,width) in offers {
                if count == shape.batch || tokens + width > shape.tokens { continue; }
                mask |= 1 << slot;tokens += width;count += 1;
                let proposal = self.split.as_ref().unwrap().retained[slot].unwrap();
                value += values[slot].decode(expected(&proposal.confidence,width,self.policy.adaptive_verification_alpha), shape.nanos as f64 / 1e6);
            }
            if mask != 0 {
                output.push(Candidate { draft_tokens: 0, kind: GraphKind::Verification, mask, batch: shape.batch, verification_budget: None,
                    capacity: shape.width, packed_token_capacity: shape.tokens, prefill_capacity: 0, prefill_slot: None,
                    work: 0, duration_ns: shape.nanos, rate: ValueRate::new(shape.nanos,value,age), feedback: true });
            }
        }
        let mut available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot,entry)| {
            let request = entry.request.as_ref()?;
            (request.phase == RequestPhase::Decode && self.eligible(request) && !self.owns_proposal(request)).then_some(slot)
        }).collect();
        for batch in crate::execution::BATCH_CAPACITIES.into_iter().take_while(|b| *b <= self.limits.max_active.next_power_of_two()) {
            let Some(nanos) = self.costs.duration(GraphKind::Speculative,batch,1) else { continue; };
            let offers: Vec<_> = available.iter().map(|&slot| (slot,values[slot].decode(1.0,nanos as f64/1e6))).collect();
            let mask = utility::best_cohort::<REQUEST_LANES>(&offers,batch);
            if mask != 0 {
                let value = offers.iter().filter(|(slot,_)| mask & (1 << slot) != 0).map(|(_,v)| v).sum();
                output.push(Candidate { draft_tokens: 0, kind: GraphKind::Speculative, mask, batch, capacity: 1, packed_token_capacity: 0, verification_budget: None,
                    prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: nanos,
                    rate: ValueRate::new(nanos,value,age), feedback: true });
            }
        }
        let Some(width) = self.costs.proposal_width else { return; };
        available.retain(|&slot| self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(),width)>1);
        available.sort_unstable_by(|&a,&b| values[b].output_token.total_cmp(&values[a].output_token).then(a.cmp(&b)));
        for count in 1..=available.len() {
            let batch = count.next_power_of_two();
            let offers: Vec<_> = available[..count].iter().map(|&slot|
                (slot,self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(),width))).collect();
            if let Some((shape,rate)) = self.budget_choice(&offers,batch,width,values,age,optimistic) {
                output.push(Candidate { draft_tokens: 0, kind: GraphKind::Proposal, mask: offers.iter().fold(0,|mask,(slot,_)|mask|(1<<slot)),
                    batch, capacity: width, packed_token_capacity: 0, prefill_capacity: 0, prefill_slot: None,
                    work: 0, duration_ns: rate.duration_ns, rate, feedback: true, verification_budget: Some(shape.tokens) });
            }
        }
    }
}
