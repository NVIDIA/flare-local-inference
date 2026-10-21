//! Expected useful value per time over prepared physical execution shapes.

use super::{
    utility::{LaneValue, ServicePrice, ValueRate, best_cohort, best_two_stage_cohort},
    *,
};

#[derive(Clone, Copy, Debug)]
pub struct PolicyConfig {
    pub output_token_value: f64,
    pub token_fairness: f64,
    pub delivery_fairness: f64,
    pub decode_age_price: f64,
    pub adaptation_ms: f64,
    pub estimate_ttl_ns: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            output_token_value: 64.0,
            token_fairness: 16.0,
            delivery_fairness: 4.0,
            decode_age_price: 0.1,
            adaptation_ms: 100.0,
            estimate_ttl_ns: 30_000_000_000,
        }
    }
}

impl PolicyConfig {
    /// Prefer token throughput with token-service fairness, without pricing
    /// the frequency of output bursts or their age separately.
    pub fn throughput() -> Self {
        Self { delivery_fairness: 0.0, decode_age_price: 0.0, ..Self::default() }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RequestAccounting {
    pub arrival_ns: u64,
    pub last_progress_ns: u64,
    pub last_output_ns: Option<u64>,
    pub last_output_fence: Option<Fence>,
    pub published_deliveries: u64,
    pub committed_prefill_tokens: u64,
    pub committed_output_tokens: u64,
    pub(super) pending_publication_fence: Option<Fence>,
    pub(super) service: service::ServiceAccounting,
}

impl RequestAccounting {
    pub(super) fn credit(&mut self, prefill: u64, output: u32, now: u64) {
        self.committed_prefill_tokens += prefill;
        self.committed_output_tokens += u64::from(output);
        if prefill != 0 || output != 0 {
            self.last_progress_ns = now;
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Candidate {
    pub kind: GraphKind,
    pub mask: u64,
    pub batch: usize,
    pub capacity: usize,
    pub packed_token_capacity: usize,
    pub prefill_capacity: usize,
    pub prefill_slot: Option<usize>,
    pub work: usize,
    pub duration_ns: u64,
    pub rate: ValueRate,
    pub feedback: bool,
    pub verification_budget: Option<usize>,
}

impl Candidate {
    pub(super) fn cost_key(self) -> cost::Key {
        cost::Key {
            kind: self.kind,
            batch: self.batch,
            tokens: self.capacity,
            prefill: self.prefill_capacity,
            packed: self.packed_token_capacity,
        }
    }
}

impl RequestScheduler {
    pub(super) fn dominates_padding(&self, smaller: Candidate, larger: Candidate) -> bool {
        let rows = |candidate: Candidate| {
            if candidate.packed_token_capacity != 0 {
                candidate.packed_token_capacity
            } else if candidate.kind == GraphKind::Mixed {
                candidate.prefill_capacity + (candidate.batch - 1) * candidate.capacity
            } else {
                candidate.batch * candidate.capacity
            }
        };
        let physical = |candidate: Candidate| [candidate.batch, candidate.capacity,
            candidate.prefill_capacity, rows(candidate)];
        let left = physical(smaller);
        let right = physical(larger);
        if smaller.kind != larger.kind || smaller.mask != larger.mask
            || smaller.prefill_slot != larger.prefill_slot
            || smaller.verification_budget != larger.verification_budget
            || left == right || left.iter().zip(right).any(|(a, b)| *a > b) {
            return false;
        }
        if smaller.kind == GraphKind::Prefill {
            return smaller.work == larger.work;
        }
        if smaller.capacity == larger.capacity && smaller.prefill_capacity == larger.prefill_capacity {
            return true;
        }
        bits(smaller.mask).all(|slot| {
            let request = self.slots[slot].request.as_ref().unwrap();
            if smaller.prefill_slot == Some(slot) {
                self.prefill_work(request, smaller.prefill_capacity)
                    == self.prefill_work(request, larger.prefill_capacity)
            } else {
                let kind = if smaller.kind == GraphKind::Mixed {
                    if self.speculative_widths.is_empty() { GraphKind::Decode } else { GraphKind::Speculative }
                } else { smaller.kind };
                self.lane_tokens(request, kind, smaller.capacity)
                    == self.lane_tokens(request, kind, larger.capacity)
            }
        })
    }

    pub(super) fn prune_padding(&self, candidates: &mut [Candidate]) -> usize {
        let mut retained = 0;
        for index in 0..candidates.len() {
            let candidate = candidates[index];
            if !candidates.iter().copied().any(|other| self.dominates_padding(other, candidate)) {
                candidates.swap(index, retained);
                retained += 1;
            }
        }
        // Keep dominated candidates in the tail for funded cost-refresh probes.
        retained
    }

    pub fn configure_policy(&mut self, policy: PolicyConfig) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("scheduling policy must be configured before admission".into());
        }
        if !policy.output_token_value.is_finite()
            || policy.output_token_value <= 0.0
            || !policy.adaptation_ms.is_finite()
            || policy.adaptation_ms <= 0.0
            || policy.estimate_ttl_ns == 0
            || [
                policy.token_fairness,
                policy.delivery_fairness,
                policy.decode_age_price,
            ]
            .into_iter()
            .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err("invalid scheduler utility coefficients or estimation intervals".into());
        }
        self.policy = policy;
        self.costs.advance_to(self.now_ns, policy.estimate_ttl_ns);
        Ok(())
    }

    pub fn advance_to(&mut self, now_ns: u64) -> Result<()> {
        if now_ns < self.now_ns {
            return Err("scheduler clock must be monotonic".into());
        }
        self.now_ns = now_ns;
        self.costs.advance_to(now_ns, self.policy.estimate_ttl_ns);
        Ok(())
    }

    pub fn set_arrival(&mut self, handle: RequestHandle, arrival_ns: u64) -> Result<()> {
        let request = self.request(handle)?;
        if arrival_ns > self.now_ns || request.last_use != Fence(0) {
            return Err("arrival must precede admission and be assigned before dispatch".into());
        }
        let request = self.slots[handle.slot as usize].request.as_mut().unwrap();
        request.accounting.arrival_ns = arrival_ns;
        request.accounting.last_progress_ns = arrival_ns;
        Ok(())
    }

    pub fn published_output(&mut self, handle: RequestHandle, fence: Fence) -> Result<()> {
        if fence != self.completed {
            return Err("output publication must be reported before the next completion".into());
        }
        if self.request(handle)?.accounting.pending_publication_fence != Some(fence) {
            return Err(
                "output publication must name the request's latest output completion".into(),
            );
        }
        let accounting = &mut self.slots[handle.slot as usize]
            .request
            .as_mut()
            .unwrap()
            .accounting;
        if accounting.last_output_fence != Some(fence) {
            accounting.published_deliveries += 1;
            accounting.last_output_fence = Some(fence);
            accounting.service.publish(fence, self.policy.adaptation_ms);
        }
        accounting.last_output_ns = Some(self.now_ns);
        Ok(())
    }

    pub(super) fn accounting(&self) -> RequestAccounting {
        RequestAccounting {
            arrival_ns: self.now_ns,
            last_progress_ns: self.now_ns,
            ..Default::default()
        }
    }
    fn request_kind(&self, request: &LiveRequest) -> Option<GraphKind> {
        Some(match request.phase {
            RequestPhase::Initialize => GraphKind::Init,
            RequestPhase::Restore => GraphKind::Restore,
            RequestPhase::Vision => GraphKind::Vision,
            RequestPhase::Prefill => request.prefill_kind(),
            RequestPhase::Resume => GraphKind::Resume,
            RequestPhase::Decode if self.speculative_widths.is_empty() => GraphKind::Decode,
            RequestPhase::Decode if self.split.is_some() => {
                if self.retained_proposal(request).is_some() {
                    GraphKind::Verification
                } else {
                    GraphKind::Proposal
                }
            }
            RequestPhase::Decode => GraphKind::Speculative,
            RequestPhase::Snapshot => GraphKind::Snapshot,
            _ => return None,
        })
    }

    pub(super) fn eligible(&self, request: &LiveRequest) -> bool {
        if request.window_snapshot_ready
            || self.request_kind(request).is_none()
            || (request.phase == RequestPhase::Vision
                && (request.image_slot.is_none() || !request.conditioned_inputs_ready))
        {
            return false;
        }
        if request.pending > 0
            && !(request.phase == RequestPhase::Decode
                && self.speculative_widths.is_empty()
                && self.capabilities.completion_inputs.is_empty()
                && request.pending < 2
                && !request.grammar_constrained
                && !request.host_stops)
        {
            return false;
        }
        if request.phase == RequestPhase::Decode {
            request.scheduled_consumed < self.limits.max_sequence_tokens as u64
                && request.generated_tokens as usize + request.pending
                    < request.max_new_tokens as usize
                && self.fits_next_rounds(request)
        } else {
            true
        }
    }

    fn prefill_catalog(&self, kind: GraphKind) -> &[usize] {
        if kind == GraphKind::ScoredPrefill { &self.scored_prefill_catalog } else { &self.prefill_catalog }
    }

    pub(super) fn prefill_work(&self, request: &LiveRequest, capacity: usize) -> usize {
        let start = request.consumed_tokens as usize;
        let mut end = request
            .message_boundaries
            .iter()
            .copied()
            .find(|n| *n > start)
            .unwrap_or(request.prompt_tokens)
            .min(start.saturating_add(capacity))
            .min(request.prefill_limit());
        if let Some(span) = request
            .prefill_spans
            .iter()
            .find(|s| s.start < end && end < s.end)
        {
            end = span.start;
        }
        end.saturating_sub(start)
    }

    pub fn submission_window_available(&self, duration_ns: u64) -> bool {
        self.submissions.len() < 2
            && !self.submissions.values().any(|s| s.requires_feedback)
            && (self.submissions.is_empty() || duration_ns > 0)
            && self.window_remaining().is_some()
    }

    pub(super) fn window_remaining(&self) -> Option<u64> {
        let Some((_, first)) = self.submissions.first_key_value() else {
            return Some(0);
        };
        if self.now_ns >= first.expected_finish_ns {
            return None;
        }
        Some(
            self.submissions
                .last_key_value()
                .unwrap()
                .1
                .expected_finish_ns
                .saturating_sub(self.now_ns),
        )
    }

    pub(super) fn lane_values(&self, window_ns: u64) -> ([LaneValue; REQUEST_LANES], f64, f64) {
        let mut values = [LaneValue::default(); REQUEST_LANES];
        let count = self
            .slots
            .iter()
            .filter_map(|s| s.request.as_ref())
            .filter(|r| self.request_kind(r).is_some())
            .count()
            .max(1);
        let mut token_log = [f64::NEG_INFINITY; REQUEST_LANES];
        let mut delivery_log = token_log;
        let rates: [_; REQUEST_LANES] = std::array::from_fn(|slot| {
            self.slots
                .get(slot)
                .and_then(|s| s.request.as_ref())
                .filter(|r| self.request_kind(r).is_some())
                .map(|r| self.service_rates(r))
                .unwrap_or_default()
        });
        let references = self.delivery_reference_intervals(&rates.map(|(_, d)| d));
        let mut scale = 0.0f64
            .max(self.policy.output_token_value.ln())
            .max(self.policy.decode_age_price.ln());
        for (slot, entry) in self.slots.iter().enumerate() {
            let Some(r) = entry
                .request
                .as_ref()
                .filter(|r| self.request_kind(r).is_some())
            else {
                continue;
            };
            let (tokens, deliveries) = rates[slot];
            let service = self.projected_service(r);
            if let Some(rate) = tokens {
                token_log[slot] = service
                    .tokens
                    .unwrap_or_else(|| ServicePrice::initial(rate / count as f64))
                    .log_price(self.policy.token_fairness);
            }
            if let Some(reference) = references[slot].filter(|_| deliveries.is_some()) {
                delivery_log[slot] = service
                    .deliveries
                    .unwrap_or_else(|| ServicePrice::initial(1.0 / reference))
                    .log_price(self.policy.delivery_fairness);
                values[slot].reference_delivery_ms = reference;
                values[slot].age_price = self.policy.decode_age_price;
                let pending_delivery = self.submissions.values().any(|s| {
                    matches!(
                        s.kind,
                        GraphKind::Decode | GraphKind::Speculative | GraphKind::Resume
                    ) && s.lanes.iter().any(|l| l.request == r.handle)
                });
                values[slot].age_ms = if pending_delivery {
                    0.0
                } else {
                    self.now_ns.saturating_add(window_ns).saturating_sub(
                        r.accounting
                            .last_output_ns
                            .unwrap_or(r.accounting.last_progress_ns),
                    ) as f64
                        / 1e6
                };
            }
            scale = scale.max(token_log[slot]).max(delivery_log[slot]);
        }
        let unit = (-scale).exp();
        let mut age_price = 0.0;
        for (slot, value) in values.iter_mut().enumerate() {
            value.prefill_token = unit + (token_log[slot] - scale).exp();
            value.output_token =
                self.policy.output_token_value * unit + (token_log[slot] - scale).exp();
            value.delivery = (delivery_log[slot] - scale).exp();
            value.age_price *= unit;
            age_price += value.age_price;
        }
        (values, age_price, scale)
    }

    fn next_progress(&self, r: &LiveRequest, value: LaneValue) -> Option<(u64, f64)> {
        let mut best: Option<(ValueRate, f64)> = None;
        let mut consider = |ns, work| {
            let rate = ValueRate::new(ns, work, 0.0);
            if best.is_none_or(|(old, _)| rate.compare(old).is_gt()) {
                best = Some((rate, work));
            }
        };
        if r.consumed_tokens < r.prompt_tokens as u64 {
            let kind = r.prefill_kind();
            for &capacity in self.prefill_catalog(kind) {
                let work = self.prefill_work(r, capacity);
                if work == 0 {
                    continue;
                }
                if let Some(ns) = self.costs.duration(kind, 1, capacity) {
                    consider(
                        ns,
                        value.prefill(
                            work,
                            r.consumed_tokens + work as u64 == r.prompt_tokens as u64,
                        ),
                    );
                }
            }
        } else {
            for batch in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
            {
                if r.generated_tokens == 0 || self.speculative_widths.is_empty() {
                    let kind = if r.generated_tokens == 0 {
                        GraphKind::Resume
                    } else {
                        GraphKind::Decode
                    };
                    if let Some(ns) = self.costs.duration(kind, batch, 1) {
                        consider(ns, value.decode(1.0, ns as f64 / 1e6));
                    }
                } else {
                    for &width in &self.speculative_catalog[batch.ilog2() as usize] {
                        if let Some(ns) = self.speculative_duration(r, batch, width)
                        {
                            let expected = self.expected_speculative_tokens(r, self.speculative_tokens(r, width));
                            consider(ns, value.decode(expected, ns as f64 / 1e6));
                        }
                    }
                }
            }
        }
        for &(batch, width, prefill) in &self.mixed_catalog {
            let Some(ns) = self
                .costs
                .shape_duration(GraphKind::Mixed, batch, width, prefill)
            else {
                continue;
            };
            if r.consumed_tokens < r.prompt_tokens as u64 {
                let work = self.prefill_work(r, prefill);
                if work > 0 {
                    consider(
                        ns,
                        value.prefill(
                            work,
                            r.consumed_tokens + work as u64 == r.prompt_tokens as u64,
                        ),
                    );
                }
            } else if r.generated_tokens > 0 {
                let tokens = self.speculative_tokens(r, width);
                consider(
                    ns,
                    value.decode(
                        r.acceptance.expected_tokens(self.proposal_source, tokens),
                        ns as f64 / 1e6,
                    ),
                );
            }
        }
        best.map(|(rate, value)| (rate.duration_ns, value))
    }

    fn prerequisite_value(
        &self,
        r: &LiveRequest,
        kind: GraphKind,
        duration_ns: u64,
        value: LaneValue,
    ) -> f64 {
        if terminal_cleanup(r) {
            let milliseconds = duration_ns as f64 / 1e6;
            return value.finish(milliseconds, milliseconds);
        }
        let Some((mut following, work)) = self.next_progress(r, value) else {
            return 0.0;
        };
        if duration_ns == 0 || following == 0 {
            return work;
        }
        let next_window = r.following_conditioned_window();
        if next_window.is_some() {
            let restore = crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|batch| *batch <= self.limits.max_active.next_power_of_two())
                .filter_map(|batch| self.costs.duration(GraphKind::Restore, batch, 1))
                .min()
                .unwrap_or(0);
            if restore == 0 {
                return work;
            }
            following = following.saturating_add(restore);
        }
        let patches = next_window.map_or(&r.vision_patches, |window| &window.patches);
        let skip = usize::from(kind == GraphKind::Vision);
        for patches in patches.iter().skip(skip) {
            let cost = self
                .capabilities
                .vision
                .patch_capacities
                .iter()
                .filter(|n| *n >= patches)
                .filter_map(|n| self.costs.duration(GraphKind::Vision, 1, *n))
                .min()
                .unwrap_or(0);
            if cost == 0 {
                return work;
            }
            following = following.saturating_add(cost);
        }
        work * duration_ns as f64 / duration_ns.saturating_add(following) as f64
    }

    fn following_kind(&self, r: &LiveRequest, kind: GraphKind) -> Option<GraphKind> {
        let phase = match kind {
            GraphKind::Snapshot => r.boundary_capture?.resume,
            GraphKind::Init | GraphKind::Restore | GraphKind::Vision => {
                if r.vision_patches.len() > usize::from(kind == GraphKind::Vision) {
                    return None;
                }
                if r.consumed_tokens == r.prompt_tokens as u64 {
                    RequestPhase::Resume
                } else {
                    RequestPhase::Prefill
                }
            }
            _ => return None,
        };
        match phase {
            RequestPhase::Prefill => Some(r.prefill_kind()),
            RequestPhase::Resume => Some(GraphKind::Resume),
            RequestPhase::Decode if self.speculative_widths.is_empty() => Some(GraphKind::Decode),
            RequestPhase::Decode => Some(GraphKind::Speculative),
            _ => None,
        }
    }

    fn continuation_candidate(
        &self,
        kind: GraphKind,
        capacity: usize,
        duration_ns: u64,
        first_offers: &[(usize, f64)],
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) -> Option<(u64, ValueRate)> {
        if !matches!(
            kind,
            GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot | GraphKind::Vision
        ) {
            return None;
        }
        let mask = first_offers
            .iter()
            .fold(0, |mask, (slot, _)| mask | (1 << slot));
        let fill = |selected: u64| {
            let count = selected.count_ones() as usize;
            if count == capacity {
                return selected;
            }
            let mut remaining = [(0, 0.0); REQUEST_LANES];
            let mut n = 0;
            for &offer in first_offers {
                if selected & (1 << offer.0) == 0 {
                    remaining[n] = offer;
                    n += 1;
                }
            }
            selected | best_cohort::<REQUEST_LANES>(&remaining[..n], capacity - count)
        };
        let mut best: Option<(u64, ValueRate)> = None;
        let mut consider = |following, batch, width, prefill| {
            let ns = if following == GraphKind::Speculative {
                self.proposal_round_duration(batch, width)
            } else { self.costs.shape_duration(following, batch, width, prefill) };
            let Some(ns) = ns else {
                return;
            };
            let horizon = if ns == 0 || duration_ns == 0 {
                0
            } else {
                ns.saturating_add(duration_ns)
            };
            let total_ms = horizon as f64 / 1e6;
            let first_ms = duration_ns as f64 / 1e6;
            let mut preparing = [(0, 0.0); REQUEST_LANES];
            let mut ready = preparing;
            let mut finishing = preparing;
            let mut prompts: [Option<(usize, f64)>; 2] = [None, None];
            let (mut p, mut r, mut f) = (0, 0, 0);
            let mixed = following == GraphKind::Mixed;
            let decode = if self.speculative_widths.is_empty() {
                GraphKind::Decode
            } else {
                GraphKind::Speculative
            };
            for (slot, entry) in self.slots.iter().enumerate() {
                let Some(request) = entry.request.as_ref() else {
                    continue;
                };
                let offered = mask & (1 << slot) != 0;
                if offered && terminal_cleanup(request) {
                    finishing[f] = (slot, values[slot].finish(first_ms, total_ms));
                    f += 1;
                    continue;
                }
                let promoted_kind = offered
                    .then(|| self.following_kind(request, kind))
                    .flatten();
                let ready_kind = self
                    .eligible(request)
                    .then(|| self.request_kind(request))
                    .flatten()
                    .map(|k| if k == GraphKind::Proposal { GraphKind::Speculative } else { k });
                let accepts = |k: GraphKind| {
                    if mixed {
                        k == GraphKind::Prefill || k == decode
                    } else {
                        k == following
                    }
                };
                let (role, promoted) = if let Some(k) = promoted_kind.filter(|k| accepts(*k)) {
                    (k, true)
                } else if let Some(k) = ready_kind.filter(|k| accepts(*k)) {
                    (k, false)
                } else {
                    continue;
                };
                let extent = if mixed && role == GraphKind::Prefill {
                    prefill
                } else {
                    width
                };
                if role == GraphKind::Prefill && self.prefill_work(request, extent) == 0 {
                    continue;
                }
                let offer = (
                    slot,
                    self.lane_value(request, role, extent, horizon, values[slot]),
                );
                if mixed && role == GraphKind::Prefill {
                    let best = &mut prompts[usize::from(promoted)];
                    if best.is_none_or(|(_, value)| offer.1 > value) {
                        *best = Some(offer);
                    }
                    continue;
                }
                if promoted {
                    preparing[p] = offer;
                    p += 1;
                } else {
                    ready[r] = offer;
                    r += 1;
                }
            }
            for prompt in std::iter::once(None).chain(prompts.into_iter().enumerate().filter_map(
                |(promoted, offer)| offer.map(|(slot, value)| Some((slot, value, promoted == 1))),
            )) {
                let preparing_prompt = usize::from(prompt.is_some_and(|(_, _, promoted)| promoted));
                let Some((selected, _, mut value)) = best_two_stage_cohort::<REQUEST_LANES>(
                    &preparing[..p],
                    &ready[..r],
                    &finishing[..f],
                    capacity - preparing_prompt,
                    batch - usize::from(prompt.is_some()),
                    1 - preparing_prompt,
                ) else {
                    continue;
                };
                let mut selected = selected;
                if let Some((slot, work, promoted)) = prompt {
                    value += work;
                    if promoted {
                        selected |= 1 << slot;
                    }
                }
                let rate = ValueRate::new(horizon, value, age_price);
                if best.is_none_or(|(_, old)| rate.compare(old).is_gt()) {
                    best = Some((fill(selected), rate));
                }
            }
        };
        for &width in &self.prefill_catalog {
            consider(GraphKind::Prefill, 1, width, 0);
        }
        for &width in &self.scored_prefill_catalog {
            consider(GraphKind::ScoredPrefill, 1, width, 0);
        }
        for batch in crate::execution::BATCH_CAPACITIES
            .into_iter()
            .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
        {
            consider(GraphKind::Resume, batch, 1, 0);
            if self.speculative_widths.is_empty() {
                consider(GraphKind::Decode, batch, 1, 0);
            } else {
                for &width in &self.speculative_catalog[batch.ilog2() as usize] {
                    consider(GraphKind::Speculative, batch, width, 0);
                }
            }
        }
        for &(batch, width, prefill) in &self.mixed_catalog {
            consider(GraphKind::Mixed, batch, width, prefill);
        }
        let mut terminal = [(0, 0.0); REQUEST_LANES];
        let mut count = 0;
        for slot in
            bits(mask).filter(|s| terminal_cleanup(self.slots[*s].request.as_ref().unwrap()))
        {
            let milliseconds = duration_ns as f64 / 1e6;
            terminal[count] = (slot, values[slot].finish(milliseconds, milliseconds));
            count += 1;
        }
        if count > 0 {
            let selected = best_cohort::<REQUEST_LANES>(&terminal[..count], capacity);
            let immediate = terminal[..count]
                .iter()
                .filter(|(s, _)| selected & (1 << s) != 0)
                .map(|(_, v)| v)
                .sum();
            let rate = ValueRate::new(duration_ns, immediate, age_price);
            if best.is_none_or(|(_, old)| rate.compare(old).is_gt()) {
                best = Some((fill(selected), rate));
            }
        }
        best
    }

    fn lane_tokens(&self, r: &LiveRequest, kind: GraphKind, capacity: usize) -> usize {
        match kind {
            GraphKind::Prefill | GraphKind::ScoredPrefill => self.prefill_work(r, capacity),
            GraphKind::Decode => 1,
            GraphKind::Verification => self.retained_proposal(r)
                .map_or_else(|| self.speculative_tokens(r, capacity), |p| p.width),
            GraphKind::Speculative | GraphKind::Proposal => {
                self.speculative_tokens(r, capacity)
            }
            _ => 0,
        }
    }

    fn lane_value(
        &self,
        r: &LiveRequest,
        kind: GraphKind,
        capacity: usize,
        duration_ns: u64,
        value: LaneValue,
    ) -> f64 {
        let tokens = self.lane_tokens(r, kind, capacity);
        match kind {
            GraphKind::Prefill | GraphKind::ScoredPrefill => value.prefill(
                tokens,
                r.consumed_tokens + tokens as u64 == r.prompt_tokens as u64,
            ),
            GraphKind::Decode
            | GraphKind::Speculative
            | GraphKind::Proposal
            | GraphKind::Verification => value.decode(
                self.expected_speculative_tokens(r, tokens),
                duration_ns as f64 / 1e6,
            ),
            GraphKind::Resume => value.decode(1.0, duration_ns as f64 / 1e6),
            _ => self.prerequisite_value(r, kind, duration_ns, value),
        }
    }

    fn add_candidate(
        &self,
        output: &mut Vec<Candidate>,
        kind: GraphKind,
        batch: usize,
        capacity: usize,
        mask: u64,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) {
        let Some(duration_ns) = self.costs.duration(kind, batch, capacity) else {
            return;
        };
        let mut offers = [(0usize, 0.0); REQUEST_LANES];
        let mut count = 0;
        for slot in bits(mask) {
            let r = self.slots[slot].request.as_ref().unwrap();
            if kind.is_prefill() {
                let catalog = self.prefill_catalog(kind);
                let work = self.prefill_work(r, capacity);
                // Explore larger graphs only when they can consume additional tokens.
                if work == 0
                    || (duration_ns == 0 && catalog.iter()
                        .take_while(|&&smaller| smaller < capacity)
                        .any(|&smaller| self.prefill_work(r, smaller) == work))
                {
                    continue;
                }
            }
            if kind == GraphKind::Vision && capacity < r.vision_patches[0] {
                continue;
            }
            if kind == GraphKind::Verification
                && self.retained_proposal(r).is_none_or(|p| p.width > capacity)
            {
                continue;
            }
            offers[count] = (
                slot,
                self.lane_value(r, kind, capacity, duration_ns, values[slot]),
            );
            count += 1;
        }
        if count == 0 {
            return;
        }
        let selected = best_cohort::<REQUEST_LANES>(&offers[..count], batch);
        let expected = offers[..count]
            .iter()
            .filter(|(s, _)| selected & (1 << s) != 0)
            .map(|(_, v)| v)
            .sum();
        let feedback = !self.costs.is_fresh(cost::Key {
            kind, batch, tokens: capacity, prefill: 0, packed: 0,
        })
            || !matches!(kind, GraphKind::Decode | GraphKind::Prefill)
            || bits(selected).any(|slot| {
                let r = self.slots[slot].request.as_ref().unwrap();
                r.grammar_constrained
                    || r.host_stops
                    || (kind.is_prefill()
                        && (r.consumed_tokens + self.prefill_work(r, capacity) as u64
                            == r.prompt_tokens as u64
                            || r.message_boundaries.binary_search(
                                &(r.consumed_tokens as usize + self.prefill_work(r, capacity)),
                            ).is_ok()))
            });
        if feedback && !self.submissions.is_empty() {
            return;
        }
        let (selected, rate) = self
            .continuation_candidate(
                kind,
                batch,
                duration_ns,
                &offers[..count],
                values,
                age_price,
            )
            .unwrap_or_else(|| (selected, ValueRate::new(duration_ns, expected, age_price)));
        output.push(Candidate { packed_token_capacity: 0, verification_budget: None,
            kind,
            mask: selected,
            batch,
            capacity,
            prefill_capacity: 0,
            prefill_slot: None,
            work: if kind.is_prefill() {
                self.prefill_work(
                    self.slots[selected.trailing_zeros() as usize]
                        .request
                        .as_ref()
                        .unwrap(),
                    capacity,
                )
            } else {
                0
            },
            duration_ns,
            rate,
            feedback,
        });
    }

    pub(super) fn candidates(
        &self,
        output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) {
        output.clear();
        self.mixed_candidates(output, values, age_price);
        if self.split.is_some() {
            self.split_candidates(output, values, age_price);
        }
        for kind in [
            GraphKind::Prefill,
            GraphKind::ScoredPrefill,
            GraphKind::Vision,
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Resume,
            GraphKind::Decode,
            GraphKind::Speculative,
        ] {
            let mask = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, entry)| {
                    entry
                        .request
                        .as_ref()
                        .filter(|r| self.eligible(r) && self.request_kind(r) == Some(kind))
                        .map(|_| 1u64 << slot)
                })
                .fold(0, |a, b| a | b);
            if mask == 0 {
                continue;
            }
            if kind.is_prefill() {
                for &capacity in self.prefill_catalog(kind) {
                    self.add_candidate(output, kind, 1, capacity, mask, values, age_price);
                }
            } else if kind == GraphKind::Vision {
                for &capacity in &self.capabilities.vision.patch_capacities {
                    self.add_candidate(output, kind, 1, capacity, mask, values, age_price);
                }
            } else {
                for batch in crate::execution::BATCH_CAPACITIES
                    .into_iter()
                    .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
                {
                    if kind == GraphKind::Speculative {
                        for &capacity in &self.speculative_catalog[batch.ilog2() as usize] {
                            self.add_candidate(
                                output, kind, batch, capacity, mask, values, age_price,
                            );
                        }
                    } else {
                        self.add_candidate(output, kind, batch, 1, mask, values, age_price);
                    }
                }
            }
        }
    }

    fn mixed_candidates(
        &self,
        output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES],
        age_price: f64,
    ) {
        if !self.submissions.is_empty() {
            return;
        }
        let decode = if self.speculative_widths.is_empty() {
            GraphKind::Decode
        } else {
            GraphKind::Speculative
        };
        for &(batch, width, prefill) in &self.mixed_catalog {
            let Some(duration_ns) =
                self.costs
                    .shape_duration(GraphKind::Mixed, batch, width, prefill)
            else {
                continue;
            };
            let mut decoders = [(0usize, 0.0); REQUEST_LANES];
            let mut count = 0;
            let mut prompt: Option<(usize, f64)> = None;
            for (slot, entry) in self.slots.iter().enumerate() {
                let Some(r) = entry.request.as_ref().filter(|r| self.eligible(r)) else {
                    continue;
                };
                match self.request_kind(r) {
                    Some(GraphKind::Prefill) if self.prefill_work(r, prefill) > 0 => {
                        if duration_ns == 0 && self.mixed_catalog.iter().any(|&(b, w, smaller)| {
                            b == batch && w == width && smaller < prefill
                                && self.prefill_work(r, smaller) == self.prefill_work(r, prefill)
                        }) {
                            continue;
                        }
                        let value = self.lane_value(
                            r,
                            GraphKind::Prefill,
                            prefill,
                            duration_ns,
                            values[slot],
                        );
                        if prompt.is_none_or(|(_, best)| value > best) {
                            prompt = Some((slot, value));
                        }
                    }
                    Some(kind) if kind == decode => {
                        decoders[count] = (
                            slot,
                            self.lane_value(r, decode, width, duration_ns, values[slot]),
                        );
                        count += 1;
                    }
                    _ => {}
                }
            }
            for prompt in std::iter::once(None).chain(prompt.map(Some)) {
                if prompt.is_none() && duration_ns == 0
                    && self.mixed_catalog.iter().any(|&(b, w, smaller)| {
                        b == batch && w == width && smaller < prefill
                    })
                {
                    continue;
                }
                let mut mask = best_cohort::<REQUEST_LANES>(
                    &decoders[..count],
                    batch - usize::from(prompt.is_some()),
                );
                let mut expected = decoders[..count]
                    .iter()
                    .filter(|(s, _)| mask & (1 << s) != 0)
                    .map(|(_, v)| v)
                    .sum::<f64>();
                if let Some((slot, value)) = prompt {
                    mask |= 1 << slot;
                    expected += value;
                }
                if mask == 0 {
                    continue;
                }
                output.push(Candidate { packed_token_capacity: 0, verification_budget: None,
                    kind: GraphKind::Mixed,
                    mask,
                    batch,
                    capacity: width,
                    prefill_capacity: prefill,
                    prefill_slot: prompt.map(|(s, _)| s),
                    work: 0,
                    duration_ns,
                    rate: ValueRate::new(duration_ns, expected, age_price),
                    feedback: true,
                });
            }
        }
    }

    pub(super) fn select_plan(&self) -> Option<ScheduledPlan> {
        if self.submissions.len() >= 2 || self.submissions.values().any(|s| s.requires_feedback) {
            return None;
        }
        let window = self.window_remaining()?;
        let (values, age_price, log_scale) = self.lane_values(window);
        let mut candidates = self.candidates.borrow_mut();
        self.candidates(&mut candidates, &values, age_price);
        let ordinary_count = self.prune_padding(&mut candidates);
        let compare = |a: &Candidate, b: &Candidate| {
            let last = |c: &Candidate| {
                bits(c.mask)
                    .map(|s| self.slots[s].request.as_ref().unwrap().last_use.0 as u128)
                    .sum::<u128>()
            };
            a.rate
                .compare(b.rate)
                .then(a.mask.count_ones().cmp(&b.mask.count_ones()))
                .then(last(b).cmp(&last(a)))
                .then(b.batch.cmp(&a.batch))
                .then(b.capacity.cmp(&a.capacity))
                .then(b.prefill_capacity.cmp(&a.prefill_capacity))
                .then(b.mask.cmp(&a.mask))
        };
        let ordinary = *candidates[..ordinary_count].iter().max_by(|a, b| compare(a, b))?;
        if let Some(episode) = self.refresh_episode() {
            let mut continuations = Vec::new();
            self.refresh_verification_candidates(
                &mut continuations, &values, age_price, episode.owned_mask(),
                (episode.target.kind == GraphKind::Verification).then_some(episode.target),
            );
            let candidate = *continuations.iter().max_by(|a, b| compare(a, b))?;
            return Some(self.materialize(candidate, log_scale,
                PlanOrigin::RefreshContinuation(episode.id)));
        }
        if ordinary.rate.duration_ns == 0 {
            return Some(self.materialize(ordinary, log_scale, PlanOrigin::Discovery));
        }
        if self.refresh_ready() && self.submissions.is_empty() {
            let mut targets: Vec<_> = self.costs.stale_keys().filter(|key| *key != ordinary.cost_key())
                .map(|key| (self.costs.refresh_order(key), key)).collect();
            targets.sort_by_key(|(order, _)| *order);
            let mut best: Option<(u64, cost::Key, Candidate)> = None;
            let mut choices = Vec::new();
            let mut budget_cache = split::RefreshBudgetCache::default();
            for (order, target) in targets {
                if best.is_some_and(|(best_order, _, _)| best_order < order) { break; }
                choices.clear();
                choices.extend(candidates.iter().copied().filter(|candidate| candidate.cost_key() == target));
                if self.split.is_some() {
                    self.split_refresh_candidates(&mut choices, target, &values, age_price, Some(&mut budget_cache));
                }
                if let Some(candidate) = choices.iter().copied().max_by(compare) {
                    if best.is_none_or(|(_, _, previous)| compare(&candidate, &previous).is_gt()) {
                        best = Some((order, target, candidate));
                    }
                }
            }
            if let Some((_, target, candidate)) = best {
                return Some(self.materialize(candidate, log_scale, PlanOrigin::RefreshStart(target)));
            }
        }
        Some(self.materialize(ordinary, log_scale, PlanOrigin::Ordinary))
    }

    fn materialize(&self, candidate: Candidate, log_scale: f64, origin: PlanOrigin) -> ScheduledPlan {
        let mut lanes = Vec::with_capacity(candidate.mask.count_ones() as usize);
        let mut epochs = Vec::with_capacity(lanes.capacity());
        for slot in bits(candidate.mask) {
            let r = self.slots[slot].request.as_ref().unwrap();
            let kind = if candidate.kind != GraphKind::Mixed {
                candidate.kind
            } else if candidate.prefill_slot == Some(slot) {
                GraphKind::Prefill
            } else if self.speculative_widths.is_empty() {
                GraphKind::Decode
            } else {
                GraphKind::Speculative
            };
            epochs.push(r.epoch);
            lanes.push(ScheduledLane {
                request: r.handle,
                start_position: r.scheduled_consumed,
                tokens: if kind.is_prefill() {
                    if candidate.kind == GraphKind::Mixed {
                        self.prefill_work(r, candidate.prefill_capacity)
                    } else {
                        candidate.work
                    }
                } else {
                    self.lane_tokens(r, kind, candidate.capacity)
                },
            });
        }
        ScheduledPlan {
            origin,
            selected_cost: self.selected_cost(candidate.cost_key()),
            owner: self.plan_owner.clone(),
            epochs,
            estimated_duration_ns: candidate.duration_ns,
            selection_value_rate: candidate.rate.score,
            selection_log_scale: log_scale,
            verification_budget: candidate.verification_budget,
            step: ScheduledStep { packed_token_capacity: candidate.packed_token_capacity,
                kind: candidate.kind,
                batch_capacity: candidate.batch,
                token_capacity: candidate.capacity,
                mixed: (candidate.kind == GraphKind::Mixed).then(|| MixedStep {
                    prefill_capacity: candidate.prefill_capacity,
                    prefill_row: candidate
                        .prefill_slot
                        .map(|s| (candidate.mask & ((1u64 << s) - 1)).count_ones() as usize),
                    speculative: !self.speculative_widths.is_empty(),
                }),
                lanes,
                requires_feedback: candidate.feedback,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn phase_plan(
        &self,
        kind: GraphKind,
        request: Option<RequestHandle>,
    ) -> ScheduledPlan {
        let (values, age_price, scale) = self.lane_values(0);
        let mask = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, entry)| {
                entry
                    .request
                    .as_ref()
                    .filter(|r| {
                        self.eligible(r)
                            && self.request_kind(r) == Some(kind)
                            && request.is_none_or(|h| h == r.handle)
                    })
                    .map(|_| 1u64 << slot)
            })
            .fold(0, |a, b| a | b);
        let mut candidates = Vec::new();
        if kind.is_prefill() {
            for &capacity in self.prefill_catalog(kind) {
                self.add_candidate(&mut candidates, kind, 1, capacity, mask, &values, age_price);
            }
        } else {
            for batch in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
            {
                self.add_candidate(&mut candidates, kind, batch, 1, mask, &values, age_price);
            }
        }
        let candidate = *candidates
            .iter()
            .max_by(|a, b| {
                a.rate
                    .compare(b.rate)
                    .then(a.mask.count_ones().cmp(&b.mask.count_ones()))
                    .then(b.batch.cmp(&a.batch))
            })
            .expect("requested lifecycle operation must be feasible");
        let origin = if candidate.rate.duration_ns == 0 { PlanOrigin::Discovery } else { PlanOrigin::Ordinary };
        self.materialize(candidate, scale, origin)
    }
}

pub(super) fn terminal_cleanup(request: &LiveRequest) -> bool {
    request.phase == RequestPhase::Snapshot
        && request.boundary_capture.is_none_or(|capture| {
            matches!(
                capture.resume,
                RequestPhase::Snapshot | RequestPhase::Finishing
            )
        })
}

pub(super) fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let slot = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(slot)
    })
}

pub(super) fn scratch() -> std::cell::RefCell<Vec<Candidate>> {
    std::cell::RefCell::new(Vec::new())
}
