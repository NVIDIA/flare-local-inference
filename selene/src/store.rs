//! What a host tier is asked to do and what it reports back.

use crate::{CheckpointId, History, PageId, PrefixKey, Result, StateSlotId, index};

/// Identifies one transfer or disk job issued by a store. Memory and disk
/// stores own separate transfer queues, so their counters do not intersect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// Where a looked-up record is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Memory,
    Disk,
}

/// How record files move between disk and the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Through a registered staging pool and batched stream copies.
    Staged,
    /// Through GPUDirect Storage, with no host payload staging.
    Direct,
}

/// A GPU checkpoint the scheduler published, as the worker reports it to the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedCheckpoint {
    pub id: CheckpointId,
    pub key: PrefixKey,
    /// Key of the GPU parent checkpoint, or `None` at the root.
    pub parent: Option<PrefixKey>,
    /// Digest of the aligned hash-block boundary at or below the endpoint.
    pub bucket: [u8; 16],
    /// Every page of the checkpoint's chain from the root in ascending order.
    /// The last `tail_pages` entries belong to the checkpoint itself.
    pub chain_pages: Vec<PageId>,
    pub tail_pages: usize,
    /// State slot holding the checkpoint's row.
    pub slot: StateSlotId,
}

impl PublishedCheckpoint {
    pub fn tail(&self) -> &[PageId] {
        &self.chain_pages[self.chain_pages.len() - self.tail_pages..]
    }
}

/// A host record chain that extends a GPU checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct HostHit {
    pub key: PrefixKey,
    pub consumed_tokens: u64,
    /// Consumed count of the GPU checkpoint the chain extends.
    pub resident_tokens: u64,
    /// Pages the restore writes: complete pages at the endpoint minus complete
    /// pages at the resident checkpoint.
    pub pages: usize,
    pub bytes: u64,
    pub tier: Tier,
    pub estimated_seconds: f64,
    pub chain: index::Match,
}

/// Destination ids the scheduler reserved for one inbound transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboundTicket {
    pub key: PrefixKey,
    pub pages: Vec<PageId>,
    pub slot: StateSlotId,
}

/// Completion of work a store accepted earlier.
#[derive(Debug)]
pub enum StoreEvent {
    /// The copy-out of a published checkpoint finished; the worker may unpin it.
    Stored {
        checkpoint: CheckpointId,
        key: PrefixKey,
        result: Result<u64>,
    },
    /// The copy-in for an inbound ticket finished; the worker reports it to the scheduler.
    Fetched { key: PrefixKey, result: Result<u64> },
    /// No copy still reads the GPU checkpoint; retry its publication after progress.
    Deferred { checkpoint: CheckpointId },
}

/// Counters every store maintains; rates are bytes per second of completed transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StoreStats {
    pub lookups: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub stored: u64,
    pub row_refills: u64,
    pub progress: u64,
    pub store_skipped: u64,
    pub store_failed: u64,
    pub fetched: u64,
    pub fetch_failed: u64,
    pub bytes_out: u64,
    pub bytes_in: u64,
    pub outbound_in_flight: usize,
    pub inbound_in_flight: usize,
    pub memory_records: usize,
    pub memory_bytes: u64,
    pub disk_records: usize,
    pub disk_bytes: u64,
    pub outbound_rate: f64,
    pub inbound_rate: f64,
}

/// Immediate disposition of a publication; only Queued transfers keep a GPU pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOutcome {
    Queued,
    Present,
    Retry(StoreReason),
    Oversize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreReason {
    Inflight,
    Slabs,
    Queue,
    Budget,
}

/// A host tier the model worker drives without waiting on it.
///
/// `store` and `fetch` enqueue work and return. `poll` reports completions.
/// The worker pins queued copy-outs until Stored or Deferred, retaining only
/// checkpoint IDs for retries. Ticket IDs belong to the scheduler until Fetched.
///
/// A store whose transfers can no longer be trusted reports the failure from
/// `poll` and stops producing events. The worker then stops, so ids and pins
/// held by transfers it cannot account for are never reused or released.
pub trait HostStore {
    /// Deepest record chain extending `resident` and strictly below `before`.
    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit>;
    /// Only Queued promises an event and retains a GPU copy-out pin.
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome>;
    /// Copy a looked-up chain into the ticket's ids. An error means no event follows.
    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()>;
    /// Report completed work. `Err` means the store can no longer report
    /// completions: transfers may still be running, and the worker must stop
    /// rather than free the ids and pins they hold.
    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()>;
    fn stats(&self) -> StoreStats;
    /// Drain transfers, join threads, and finish pending disk writes.
    /// On failure the caller must retain the store: transfer jobs may still
    /// reference its registered host buffers and GPU destinations.
    fn close(&mut self) -> Result<()>;
}

/// Bytes per second assumed for a tier before its first transfers are measured.
pub const INITIAL_MEMORY_RATE: f64 = 20e9;
pub const INITIAL_STAGED_DISK_RATE: f64 = 3e9;
pub const INITIAL_DIRECT_DISK_RATE: f64 = 10e9;
/// Prefill tokens per second assumed before the worker supplies a measurement.
pub const INITIAL_PREFILL_RATE: f64 = 8000.0;

/// A fetch is worthwhile when the transfer takes less than half the prefill it replaces.
pub fn worth_fetching(hit: &HostHit, prefill_tokens_per_second: f64) -> bool {
    let gained = (hit.consumed_tokens - hit.resident_tokens) as f64;
    hit.estimated_seconds < 0.5 * gained / prefill_tokens_per_second.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_policy_compares_transfer_time_with_half_the_prefill_time() {
        let key = PrefixKey {
            consumed_tokens: 8000,
            digest: [0; 16],
        };
        let hit = HostHit {
            key,
            consumed_tokens: 8000,
            resident_tokens: 0,
            pages: 125,
            bytes: 1 << 30,
            tier: Tier::Memory,
            estimated_seconds: 0.4,
            chain: index::Match {
                key,
                consumed_tokens: 8000,
                resident_tokens: 0,
                nodes: Vec::new(),
                skip_pages: 0,
                pages: 125,
            },
        };
        assert!(worth_fetching(&hit, 8000.0));
        assert!(!worth_fetching(&hit, 16000.0));
        let mut slow = hit.clone();
        slow.estimated_seconds = 0.6;
        assert!(!worth_fetching(&slow, 8000.0));
    }
}
