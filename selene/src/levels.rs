/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed copy priority and costs observed from completed restores.
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    ThisGpu,
    NeighbourGpu,
    HostDram,
    HostNvme,
    RemoteNvme,
}
impl Level {
    fn index(self) -> usize {
        match self {
            Self::ThisGpu => 0,
            Self::NeighbourGpu => 1,
            Self::HostDram => 2,
            Self::HostNvme => 3,
            Self::RemoteNvme => 4,
        }
    }
}

/// A pessimistic rate doubles back toward its seed every interval without a
/// new sample: a tier refused on its estimate would otherwise never be read
/// again, so one slow transfer would refuse its restores for good. A restore
/// in flight holds the estimate, so a slow device is probed once per interval.
/// Two slow samples in the median need two probe cycles to be outvoted.
const RECOVERY: Duration = Duration::from_secs(10);

/// Bytes per second, using the median of the last three completed transfers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rate {
    seed: f64,
    median: f64,
    samples: [f64; 3],
    count: usize,
    next: usize,
    last: Option<Instant>,
}
impl Rate {
    pub(crate) fn new(seed: f64) -> Self {
        Self {
            seed,
            median: seed,
            samples: [0.0; 3],
            count: 0,
            next: 0,
            last: None,
        }
    }
    pub(crate) fn observe(&mut self, bytes: u64, elapsed: Duration) {
        if bytes == 0 || elapsed.is_zero() {
            return;
        }
        self.samples[self.next] = bytes as f64 / elapsed.as_secs_f64();
        self.next = (self.next + 1) % 3;
        self.count = (self.count + 1).min(3);
        self.last = Some(Instant::now());
        let mut values = self.samples;
        values[..self.count].sort_by(f64::total_cmp);
        self.median = values[self.count / 2];
    }
    pub(crate) fn get(&self) -> f64 {
        match self.last {
            Some(last) if self.median < self.seed => (self.median
                * 2f64.powf(last.elapsed().as_secs_f64() / RECOVERY.as_secs_f64()))
            .min(self.seed),
            _ => self.median,
        }
    }
    /// A restore started: hold the estimate at the median until it reports.
    pub(crate) fn started(&mut self) {
        if self.median < self.seed {
            self.last = Some(Instant::now());
        }
    }
    #[cfg(test)]
    pub(crate) fn age(&mut self, by: Duration) {
        self.last = self.last.map(|last| last.checked_sub(by).expect("clock since boot"));
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ranking {
    rates: [Rate; 5],
}
impl Default for Ranking {
    fn default() -> Self {
        use crate::store::{INITIAL_MEMORY_RATE, INITIAL_STAGED_DISK_RATE};
        Self {
            rates: [
                Rate::new(INITIAL_MEMORY_RATE),
                Rate::new(INITIAL_MEMORY_RATE),
                Rate::new(INITIAL_MEMORY_RATE),
                Rate::new(INITIAL_STAGED_DISK_RATE),
                Rate::new(400.0 * 1024.0 * 1024.0),
            ],
        }
    }
}
impl Ranking {
    pub(crate) fn observe(&mut self, level: Level, bytes: u64, elapsed: Duration) {
        if level != Level::ThisGpu {
            self.rates[level.index()].observe(bytes, elapsed);
        }
    }
    pub(crate) fn started(&mut self, level: Level) {
        self.rates[level.index()].started();
    }
    pub fn part_seconds(&self, level: Level, bytes: u64) -> f64 {
        bytes as f64 / self.rates[level.index()].get()
    }
    #[cfg(test)]
    pub(crate) fn for_test(costs: &[(Level, f64)]) -> Self {
        let mut ranking = Self::default();
        for (level, us) in costs {
            ranking.rates[level.index()] = Rate::new(4096.0 * 1e6 / us);
        }
        ranking
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rates_use_recent_completed_transfers_and_ignore_empty_samples() {
        let mut rate = Rate::new(1000.0);
        rate.observe(0, Duration::from_secs(1));
        rate.observe(500, Duration::ZERO);
        assert_eq!(rate.get(), 1000.0);
        for bytes in [4000, 1000, 100000, 2000] {
            rate.observe(bytes, Duration::from_secs(1));
        }
        assert_eq!(rate.get(), 2000.0);
    }
    #[test]
    fn a_slow_rate_recovers_toward_its_seed_without_new_samples() {
        let mut rate = Rate::new(3000.0);
        rate.observe(100, Duration::from_secs(1));
        assert!((rate.get() - 100.0).abs() < 1.0);
        rate.age(RECOVERY * 2);
        assert!((rate.get() - 400.0).abs() < 1.0);
        rate.age(RECOVERY * 10);
        assert_eq!(rate.get(), 3000.0);
        let mut fast = Rate::new(3000.0);
        fast.observe(9000, Duration::from_secs(1));
        fast.age(RECOVERY * 10);
        assert_eq!(fast.get(), 9000.0);
    }
    #[test]
    fn a_restore_in_flight_holds_a_slow_estimate() {
        let mut rate = Rate::new(3000.0);
        rate.observe(100, Duration::from_secs(1));
        rate.age(RECOVERY * 10);
        assert_eq!(rate.get(), 3000.0);
        rate.started();
        assert!((rate.get() - 100.0).abs() < 1.0);
        let mut fast = Rate::new(3000.0);
        fast.observe(9000, Duration::from_secs(1));
        fast.started();
        assert_eq!(fast.get(), 9000.0);
    }
    #[test]
    fn two_slow_samples_need_two_fast_ones_to_be_outvoted() {
        let mut rate = Rate::new(3000.0);
        rate.observe(100, Duration::from_secs(1));
        rate.observe(100, Duration::from_secs(1));
        rate.observe(9000, Duration::from_secs(1));
        assert!((rate.get() - 100.0).abs() < 1.0);
        rate.observe(9000, Duration::from_secs(1));
        assert_eq!(rate.get(), 9000.0);
    }
    #[test]
    fn unmeasured_levels_use_the_seeded_bandwidth() {
        let rates = Ranking::default();
        assert_eq!(rates.part_seconds(Level::HostDram, 20_000), 1e-6);
        assert_eq!(
            rates.part_seconds(Level::RemoteNvme, 400 * 1024 * 1024),
            1.0
        );
    }
}
