//! What a host tier is asked to do and what it reports back.

use crate::shared::Tier;

use crate::{
    CheckpointId, History, PageId, PrefixKey, Result, StateSlotId, index,
    pool::{Announcer, HostPin, LookupTicket, Tier as PoolTier},
};

/// Identifies one transfer or disk job issued by a store. Memory and disk
/// stores own separate transfer queues, so their counters do not intersect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

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

/// Names one export a manager asked this instance to serve. Assigned by the
/// manager; unique within one manager incarnation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExportId(pub u64);

/// Why the worker did not pin a GPU checkpoint for an export. Nothing is
/// pinned after any of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportRefusal {
    /// The checkpoint is gone or was never ready.
    NotFound,
    /// The chain is not contiguous from the physical start, or lacks the wanted row.
    ChainGap,
    /// Pinning would leave too little room for running or queued requests.
    NoHeadroom,
    /// The chain is larger than the staging the manager reserved for it.
    NoWindows,
    /// This instance already serves as many exports as it may.
    Busy,
}

/// One record of an exported chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportNode {
    pub key: PrefixKey,
    /// The record's own pages in ascending order.
    pub pages: Vec<PageId>,
}

/// The pinned GPU chain an export reads, root first and trimmed to the
/// physical start: the first record carries all its pages and `skip_pages`
/// of them precede what the export takes. The last node is the endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportChain {
    pub nodes: Vec<ExportNode>,
    /// Pages of the first node the requester already holds.
    pub skip_pages: u32,
    /// The endpoint's state slot when the export carries its row.
    pub row: Option<StateSlotId>,
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
    /// No copy still reads the GPU checkpoint; retry its publication after progress.
    Deferred {
        checkpoint: CheckpointId,
        reason: StoreReason,
    },
    /// The copy-in for an inbound ticket finished; the worker reports it to the scheduler.
    Fetched { key: PrefixKey, result: Result<u64> },
    /// A manager asked this instance to export a GPU checkpoint. The worker
    /// answers with `export_pinned` or `export_refused` in the same iteration.
    ExportRequested {
        export: ExportId,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
        /// Bytes of staging the manager reserved for the whole export.
        reserved_bytes: u64,
    },
    /// The export finished or was cancelled and no copy still reads its pages:
    /// release its pin, once.
    ExportEnded { export: ExportId },
}

/// Counters every store maintains; rates are bytes per second of completed transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StoreStats {
    pub lookups: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub stored: u64,
    pub row_refills: u64,
    /// Completion and capacity-release events, including host pins and shared disk budget.
    pub progress: u64,
    pub store_skipped: u64,
    pub store_failed: u64,
    /// DRAM, NVMe holding events. Drops: each tier, then end_of_chain/oversize/io_error.
    pub held: [u64; 2],
    pub dropped: [u64; 6],
    pub expired: [u64; 2],
    pub fetched: u64,
    pub fetch_failed: u64,
    pub bytes_out: u64,
    pub bytes_in: u64,
    pub outbound_in_flight: usize,
    /// Records held in memory for a later write to disk: complete, no copy running.
    pub parked_records: usize,
    pub inbound_in_flight: usize,
    pub memory_records: usize,
    pub rows_used: usize,
    pub rows_total: usize,
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

/// A record a host tier would drop under pressure. The tier keeps it, and the
/// chain it needs, until [`TierStore::retired`] says what became of its move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Retiring {
    pub tier: PoolTier,
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub bucket: [u8; 16],
    /// Pages the record owns, counted from its parent.
    pub tail_pages: usize,
    pub row: bool,
}

/// How the move of a retiring record ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retirement {
    /// The next tier holds a committed copy: the tier drops its own.
    Moved,
    /// A terminal I/O failure permits discarding the source.
    Dropped,
    /// Keep the source and retry after progress.
    Failed,
}

pub(crate) fn headroom(capacity: u64, largest: u64) -> u64 {
    (capacity / 20)
        .max(largest.saturating_mul(2))
        .min(capacity.div_ceil(10))
}

pub(crate) fn retry_once(failures: &mut u8) -> Retirement {
    *failures += 1;
    if *failures == 1 {
        Retirement::Failed
    } else {
        Retirement::Dropped
    }
}

/// A host tier the model worker drives without waiting on it.
///
/// `store` and `fetch` enqueue work and return. `poll` reports completions.
/// The worker pins queued copy-outs until Stored or Deferred.
/// Ticket IDs belong to the scheduler until Fetched.
///
/// A store whose transfers can no longer be trusted reports the failure from
/// `poll` and stops producing events. The worker then stops, so ids and pins
/// held by transfers it cannot account for are never reused or released.
pub trait HostStore {
    /// Deepest record chain extending `resident` and strictly below `before`.
    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit>;
    /// Copy a published checkpoint. Only Queued promises an event and retains
    /// its GPU copy-out pin until Stored or Deferred.
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome>;
    /// A request reused this GPU checkpoint: refresh any retained host record.
    fn reused(&mut self, _key: PrefixKey) {}
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

    /// Whether request lookups should start while queued for this shared store.
    fn pooled(&self) -> bool {
        false
    }
    /// The worker pinned the checkpoint for `export` and describes its chain.
    /// An error means the store cannot serve the export; the worker unpins.
    fn export_pinned(&mut self, _export: ExportId, _chain: ExportChain) -> Result<()> {
        Ok(())
    }
    /// The worker did not pin the checkpoint for `export`.
    fn export_refused(&mut self, _export: ExportId, _reason: ExportRefusal) {}
    /// A published GPU checkpoint left the cache, including when compaction
    /// removed it as an endpoint; `heir` is the single child that took over its
    /// pages, with its updated publication.
    fn evicted(&mut self, _key: PrefixKey, _heir: Option<&PublishedCheckpoint>) {}
    /// Some eviction notices were dropped because they queued faster than the
    /// worker delivered them, so the announced holdings may be stale and must be
    /// announced again in full.
    fn evictions_lost(&mut self) {}
    /// Start looking for pool candidates for a request that just queued. `None`
    /// when no lookup is wanted or possible.
    fn begin_lookup(&mut self, _history: &dyn History) -> Option<LookupTicket> {
        None
    }
    /// [`HostStore::lookup`] for a request that holds `ticket`. Without a ticket,
    /// or while its lookup has no answer, this is the local lookup.
    fn lookup_for(
        &mut self,
        _ticket: Option<&LookupTicket>,
        history: &dyn History,
        resident: u64,
        before: u64,
    ) -> Option<HostHit> {
        self.lookup(history, resident, before)
    }
}

/// Internal cooperation between local tiers and the pool.
pub trait TierStore: HostStore {
    /// Persist optional host records before pool teardown, retaining transfer owners.
    fn drain_on_shutdown(&mut self) {}
    /// Announce what this store's tiers hold, now and as they change, through
    /// `announcer`, and serve them as exports. Nothing is recorded before this.
    fn announce_to(&mut self, _announcer: Announcer) {}
    /// Pin the chain a manager asked for from the DRAM or NVMe tier. The pin
    /// holds the records in place until it is dropped; the store releases them
    /// on its own thread. Refused when this store does not hold the chain.
    fn pin_export(
        &mut self,
        _tier: PoolTier,
        _key: PrefixKey,
        _begin_tokens: u64,
        _want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        Err(ExportRefusal::NotFound)
    }
    /// Whether the last tier may hand its records on to the pool's manager
    /// instead of dropping them. Set whenever the manager's availability changes.
    fn set_spill(&mut self, _enabled: bool) {}
    /// Records the last local tier wants to move to the manager, coldest first.
    /// Each stays in place until [`TierStore::retired`] reports its move.
    fn take_retiring(&mut self) -> Vec<Retiring> {
        Vec::new()
    }
    /// The move of a retiring record ended.
    fn retired(&mut self, _key: PrefixKey, _tier: PoolTier, _how: Retirement) {}
}

/// Bytes per second assumed for a tier before its first transfers are measured.
pub const INITIAL_MEMORY_RATE: f64 = 20e9;
pub const INITIAL_STAGED_DISK_RATE: f64 = 3e9;
/// Prefill tokens per second assumed before the worker supplies a measurement.
pub const INITIAL_PREFILL_RATE: f64 = 8000.0;

/// A fetch is worthwhile when the transfer takes less than half the prefill it replaces.
pub fn worth_fetching(hit: &HostHit, prefill_tokens_per_second: f64) -> bool {
    worth_transfer(
        hit.consumed_tokens - hit.resident_tokens,
        hit.estimated_seconds,
        prefill_tokens_per_second,
    )
}

/// The same rule for `gained_tokens` of prefill replaced by a transfer of `estimated_seconds`.
pub fn worth_transfer(
    gained_tokens: u64,
    estimated_seconds: f64,
    prefill_tokens_per_second: f64,
) -> bool {
    estimated_seconds < 0.5 * gained_tokens as f64 / prefill_tokens_per_second.max(1.0)
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
            tier: Tier::Dram,
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
