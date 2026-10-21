//! Useful-work rates, proportional-fair service prices, and decode-age value.

/// Marginal service price divided by its fairness coefficient, in log form.
#[derive(Clone, Copy, Debug)]
pub(super) struct ServicePrice {
    log_inverse_rate: f64,
}

impl ServicePrice {
    pub fn initial(rate: f64) -> Self {
        assert!(rate.is_finite() && rate > 0.0);
        Self {
            log_inverse_rate: -rate.ln(),
        }
    }

    pub fn log_price(&self, coefficient: f64) -> f64 {
        debug_assert!(coefficient >= 0.0);
        coefficient.ln() + self.log_inverse_rate
    }

    /// Integrate dp/dt = (f - p*y)/(m*tau) for physical service rate y.
    /// An unserved request's price grows linearly with occupied elapsed time.
    pub fn observe(
        &mut self,
        duration_ms: f64,
        delivered: f64,
        maximum_rate: f64,
        adaptation_ms: f64,
    ) {
        debug_assert!(duration_ms.is_finite() && duration_ms > 0.0);
        debug_assert!(delivered.is_finite() && delivered >= 0.0);
        debug_assert!(maximum_rate.is_finite() && maximum_rate > 0.0);
        debug_assert!(adaptation_ms.is_finite() && adaptation_ms > 0.0);
        let mut previous = self.log_inverse_rate;
        let incoming = if delivered > 0.0 {
            let normalized = delivered / maximum_rate;
            let decay = normalized / adaptation_ms;
            previous -= decay;
            (-(-decay).exp_m1()).ln() + duration_ms.ln() - delivered.ln()
        } else {
            duration_ms.ln() - adaptation_ms.ln() - maximum_rate.ln()
        };
        let maximum = previous.max(incoming);
        self.log_inverse_rate =
            maximum + ((previous - maximum).exp() + (incoming - maximum).exp()).ln();
    }
}

/// Values share one positive scaling factor across all candidates in a plan.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LaneValue {
    pub prefill_token: f64,
    pub output_token: f64,
    pub delivery: f64,
    pub age_price: f64,
    pub age_ms: f64,
    pub reference_delivery_ms: f64,
}

impl LaneValue {
    pub fn prefill(&self, consumed: usize, first_output: bool) -> f64 {
        self.prefill_token * consumed as f64 + self.output_token * f64::from(first_output)
    }

    pub fn decode(&self, expected_output: f64, duration_ms: f64) -> f64 {
        self.output_token * expected_output
            + self.delivery
            + self.age_price * self.reference_delivery_ms * (self.age_ms + duration_ms)
    }

    pub fn finish(&self, completion_ms: f64, horizon_ms: f64) -> f64 {
        debug_assert!(horizon_ms == 0.0 || completion_ms <= horizon_ms);
        // Completion finalizes the response and releases request-owned resources.
        // Give it one output-token-equivalent of useful value even when optional
        // delivery fairness and age pricing are zero; no token credit is added.
        let mut value = self.output_token + self.decode(0.0, horizon_ms);
        if horizon_ms > 0.0 {
            // A terminal delivery stops its age clock before the sequence ends.
            value += self.age_price
                * (horizon_ms - completion_ms)
                * (self.age_ms + (horizon_ms + completion_ms) / 2.0);
        }
        value
    }
}

/// Expected net value per time, up to terms common to every candidate.
/// Zero duration marks an optimistic exploration candidate, not a divisor.
#[derive(Clone, Copy, Debug)]
pub(super) struct ValueRate {
    pub duration_ns: u64,
    pub score: f64,
}

impl ValueRate {
    pub fn new(duration_ns: u64, expected_value: f64, decode_age_price: f64) -> Self {
        let duration_ms = duration_ns as f64 / 1e6;
        Self {
            duration_ns,
            score: if duration_ns == 0 {
                expected_value
            } else {
                expected_value / duration_ms - decode_age_price * duration_ms / 2.0
            },
        }
    }

    pub fn compare(self, other: Self) -> std::cmp::Ordering {
        (self.duration_ns == 0)
            .cmp(&(other.duration_ns == 0))
            .then_with(|| self.score.total_cmp(&other.score))
            .then_with(|| other.duration_ns.cmp(&self.duration_ns))
    }
}

/// Retain the largest nonnegative lane values for a fixed physical graph.
/// All excluded-lane waiting costs are common to cohorts of that graph.
pub(super) fn best_cohort<const N: usize>(offers: &[(usize, f64)], capacity: usize) -> u64 {
    debug_assert!(offers.len() <= N && capacity > 0);
    let mut selected = [(0usize, f64::NEG_INFINITY); N];
    let limit = capacity.min(N);
    let mut count = 0;
    for &(slot, value) in offers {
        debug_assert!(slot < 64 && value.is_finite() && value >= 0.0);
        let insertion = selected[..count]
            .iter()
            .position(|&(existing, score)| value > score || (value == score && slot < existing))
            .unwrap_or(count);
        if insertion < limit {
            let last = count.min(limit - 1);
            selected.copy_within(insertion..last, insertion + 1);
            selected[insertion] = (slot, value);
            count = (count + 1).min(limit);
        }
    }
    selected[..count]
        .iter()
        .fold(0, |mask, (slot, _)| mask | (1 << slot))
}

/// Maximize additive value across two fixed graph capacities. Preparing rows
/// deliver in the second graph; finishing rows deliver in the first graph.
pub(super) fn best_two_stage_cohort<const N: usize>(
    preparing: &[(usize, f64)],
    ready: &[(usize, f64)],
    finishing: &[(usize, f64)],
    preparation_capacity: usize,
    execution_capacity: usize,
    minimum_preparing: usize,
) -> Option<(u64, u64, f64)> {
    fn prefixes<const N: usize>(offers: &[(usize, f64)]) -> [(u64, f64); N] {
        debug_assert!(offers.len() <= N);
        let mut rows = [(0, 0.0); N];
        rows[..offers.len()].copy_from_slice(offers);
        rows[..offers.len()].sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut prefixes = [(0, 0.0); N];
        let mut prefix = (0, 0.0);
        for (i, &(slot, value)) in rows[..offers.len()].iter().enumerate() {
            debug_assert!(slot < 64 && value.is_finite() && value >= 0.0);
            prefix.0 |= 1 << slot;
            prefix.1 += value;
            prefixes[i] = prefix;
        }
        prefixes
    }
    let prepared = prefixes::<N>(preparing);
    let runnable = prefixes::<N>(ready);
    let terminal = prefixes::<N>(finishing);
    let prefix =
        |rows: &[(u64, f64); N], count: usize| count.checked_sub(1).map_or((0, 0.0), |i| rows[i]);
    let mut best: Option<(u64, u64, f64)> = None;
    for count in minimum_preparing
        ..=preparing
            .len()
            .min(preparation_capacity)
            .min(execution_capacity)
    {
        let (unlocked, a) = prefix(&prepared, count);
        let (ready, b) = prefix(&runnable, ready.len().min(execution_capacity - count));
        let (finished, c) = prefix(&terminal, finishing.len().min(preparation_capacity - count));
        let candidate = (unlocked | finished, unlocked | ready, a + b + c);
        if best.is_none_or(|old| {
            candidate.2.total_cmp(&old.2).is_gt()
                || (candidate.2 == old.2 && candidate.0.count_ones() > old.0.count_ones())
        }) {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_service_integrates_the_entropy_dual_flow() {
        let mut rate = ServicePrice::initial(5.0);
        rate.observe(20.0, 60.0, 10.0, 100.0);
        let expected = (-0.06f64).exp() / 5.0 + -(-0.06f64).exp_m1() / 3.0;
        assert!((rate.log_inverse_rate.exp() - expected).abs() < 1e-12);
        let before = rate.log_price(3.0).exp();
        rate.observe(200.0, 0.0, 10.0, 100.0);
        assert!((rate.log_price(3.0).exp() - before - 0.6).abs() < 1e-12);
    }

    #[test]
    fn equal_rate_intervals_compose_without_launch_count_bias() {
        let mut whole = ServicePrice::initial(10.0 / 3.0);
        let mut pieces = whole;
        whole.observe(60.0, 120.0, 10.0, 100.0);
        for dt in [3.0, 7.0, 19.0, 31.0] {
            pieces.observe(dt, 2.0 * dt, 10.0, 100.0);
        }
        assert!((whole.log_inverse_rate - pieces.log_inverse_rate).abs() < 1e-12);
    }

    #[test]
    fn long_stalls_do_not_underflow_the_rate_history() {
        let mut rate = ServicePrice::initial(5.0);
        rate.observe(100_000.0, 0.0, 10.0, 100.0);
        assert!(rate.log_price(4.0).is_finite());
        rate.observe(30.0, 8.0, 10.0, 100.0);
        assert!(rate.log_inverse_rate.is_finite());
        assert!(rate.log_price(4.0) < 10.0);
    }

    #[test]
    fn efficient_long_graphs_remain_candidates() {
        let narrow = ValueRate::new(17_300_000, 3.0 * 64.0, 0.0);
        let wide = ValueRate::new(30_000_000, 8.0 * 64.0, 0.0);
        assert!(wide.compare(narrow).is_gt());
    }

    #[test]
    fn unknown_costs_compete_by_useful_value() {
        let known = ValueRate::new(1, 100.0, 0.0);
        let unknown_small = ValueRate::new(0, 64.0, 0.0);
        let unknown_large = ValueRate::new(0, 4096.0, 0.0);
        assert!(unknown_small.compare(known).is_gt());
        assert!(unknown_large.compare(unknown_small).is_gt());
    }

    #[test]
    fn pure_prefill_and_decode_wait_use_the_same_value_rate() {
        let small = |price| ValueRate::new(8_000_000, 512.0, price);
        let large = |price| ValueRate::new(62_000_000, 4096.0, price);
        assert!(large(0.0).compare(small(0.0)).is_gt());
        assert!(small(4.0 * 0.04).compare(large(4.0 * 0.04)).is_gt());
    }

    #[test]
    fn age_value_counts_a_delivery_once_per_burst() {
        let value = LaneValue {
            output_token: 64.0,
            delivery: 2.0,
            age_price: 0.1,
            age_ms: 500.0,
            reference_delivery_ms: 30.0,
            ..Default::default()
        };
        assert_eq!(
            value.decode(8.0, 30.0) - value.decode(1.0, 30.0),
            7.0 * 64.0
        );
    }

    #[test]
    fn completion_has_base_value_without_delivery_or_age_prices() {
        let value = LaneValue { output_token: 64.0, ..Default::default() };
        assert_eq!(value.finish(1.0, 1.0), 64.0);
        assert_eq!(value.finish(1.0, 8.0), 64.0);
    }

    #[test]
    fn completion_inside_a_sequence_matches_integrated_age_and_continuation() {
        let terminal = LaneValue {
            delivery: 4.0,
            age_price: 0.2,
            age_ms: 13.0,
            reference_delivery_ms: 41.0,
            ..Default::default()
        };
        let continuing = LaneValue {
            output_token: 64.0,
            delivery: 6.0,
            age_price: 0.3,
            age_ms: 29.0,
            reference_delivery_ms: 27.0,
            ..Default::default()
        };
        let total = 6.0;
        for finish in [1.0, 3.0, 6.0] {
            let value = terminal.finish(finish, total) + continuing.decode(8.0, total);
            let selected = ValueRate::new(6_000_000, value, 0.5).score;
            let mut reward = 4.0 + 6.0 + 8.0 * 64.0;
            let mut potential = 0.0;
            let mut common = 0.0;
            for (lane, until) in [(terminal, finish), (continuing, total)] {
                reward -= lane.age_price * (lane.age_ms * until + until * until / 2.0);
                potential += lane.age_price * lane.reference_delivery_ms * lane.age_ms;
                common += lane.age_price * (lane.age_ms + lane.reference_delivery_ms);
            }
            assert!((selected - (reward + potential) / total - common).abs() < 1e-12);
        }
    }

    #[test]
    fn prefill_tail_counts_consumed_tokens_and_one_first_output() {
        let value = LaneValue {
            prefill_token: 1.0,
            output_token: 64.0,
            ..Default::default()
        };
        assert_eq!(value.prefill(64, true), 128.0);
        let small = ValueRate::new(3_000_000, value.prefill(64, true), 0.0);
        let padded = ValueRate::new(66_000_000, value.prefill(64, true), 0.0);
        assert!(small.compare(padded).is_gt());
    }

    #[test]
    fn cohort_matches_exhaustive_subsets_for_sixteen_requests() {
        let offers: [_; 16] = std::array::from_fn(|i| (i, ((i * 7 + 3) % 17) as f64));
        for capacity in [1, 2, 4, 8, 16] {
            let score = |mask: u64| {
                offers
                    .iter()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, v)| v)
                    .sum::<f64>()
            };
            let selected = best_cohort::<16>(&offers, capacity);
            let exhaustive = (1u64..1 << 16)
                .filter(|mask| mask.count_ones() as usize <= capacity)
                .map(score)
                .fold(0.0, f64::max);
            assert_eq!(score(selected), exhaustive);
        }
    }

    #[test]
    fn physical_padding_adds_neither_lanes_nor_work() {
        assert_eq!(
            best_cohort::<16>(&[(7, 9.0), (2, 1.0)], 16),
            (1 << 7) | (1 << 2)
        );
        assert_eq!(best_cohort::<16>(&[(7, 9.0), (2, 9.0)], 1), 1 << 2);
        assert_eq!(best_cohort::<16>(&[], 1), 0);
    }

    #[test]
    fn two_stage_cohorts_match_exhaustive_preparation_and_execution_subsets() {
        for minimum in [0, 1] {
            for seed in 0..16 {
                let offers: [_; 7] =
                    std::array::from_fn(|i| (i, ((i * 17 + seed * 11) % 23) as f64));
                for preparation_capacity in 1..=4 {
                    for execution_capacity in 1..=6 {
                        let (first, second, value) = best_two_stage_cohort::<16>(
                            &offers[..3],
                            &offers[3..6],
                            &offers[6..],
                            preparation_capacity,
                            execution_capacity,
                            minimum,
                        )
                        .unwrap();
                        assert!(first.count_ones() as usize <= preparation_capacity);
                        assert!(second.count_ones() as usize <= execution_capacity);
                        assert_eq!(second & 7 & !first, 0);
                        assert!((second & 7).count_ones() as usize >= minimum);
                        let mut exhaustive = 0.0f64;
                        for subset in 0u64..16 {
                            let prepared = subset & 7 | (subset & 8) << 3;
                            if prepared.count_ones() as usize > preparation_capacity {
                                continue;
                            }
                            for executed in 0u64..64 {
                                if executed.count_ones() as usize > execution_capacity
                                    || ((executed & 7).count_ones() as usize) < minimum
                                    || executed & 7 & !prepared != 0
                                {
                                    continue;
                                }
                                let selected = executed | (prepared & 64);
                                let score = offers
                                    .iter()
                                    .filter(|(s, _)| selected & (1 << s) != 0)
                                    .map(|(_, v)| v)
                                    .sum::<f64>();
                                exhaustive = exhaustive.max(score);
                            }
                        }
                        assert_eq!(value, exhaustive);
                    }
                }
            }
        }
        assert!(best_two_stage_cohort::<16>(&[], &[(0, 1.0)], &[], 1, 1, 1).is_none());
    }
}
