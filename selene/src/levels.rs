//! Fixed copy priority and costs observed from completed restores.
use std::time::Duration;

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

/// Bytes per second, using the median of the last three completed transfers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rate(pub f64, [f64; 3], usize, usize);
impl Rate {
    pub(crate) fn new(seed: f64) -> Self {
        Self(seed, [0.0; 3], 0, 0)
    }
    pub(crate) fn observe(&mut self, bytes: u64, elapsed: Duration) {
        if bytes == 0 || elapsed.is_zero() {
            return;
        }
        self.1[self.3] = bytes as f64 / elapsed.as_secs_f64();
        self.3 = (self.3 + 1) % 3;
        self.2 = (self.2 + 1).min(3);
        let mut values = self.1;
        values[..self.2].sort_by(f64::total_cmp);
        self.0 = values[self.2 / 2];
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
    pub fn part_seconds(&self, level: Level, bytes: u64) -> f64 {
        bytes as f64 / self.rates[level.index()].0
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
        assert_eq!(rate.0, 1000.0);
        for bytes in [4000, 1000, 100000, 2000] {
            rate.observe(bytes, Duration::from_secs(1));
        }
        assert_eq!(rate.0, 2000.0);
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
