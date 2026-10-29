//! Local copies write through; pooled copies spill under headroom pressure.
//! A full disk keeps restorable DRAM copies and parent pins until reservation.
//! The disk index also holds uncharged parked metadata; DRAM links are flat.
//! Only committed files make DRAM copies evictable; disk LRU removes both.
//!
//! Restores pin slabs and files and use bounded staging for larger chains.
//! Filesystem pressure retries with backoff. Other I/O errors may discard a
//! record after the pooled retry. Storage outlives every transfer.

#[cfg(test)]
use crate::store::TierStore;

use std::{
    collections::{BinaryHeap, HashMap},
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::levels::Rate;
use crate::{
    CheckpointId, Error, ExportRefusal, History, PrefixKey, Result,
    pool::{Announcer, HostPin, Tier as PoolTier},
};

use super::{
    HostHit, HostStore, INITIAL_STAGED_DISK_RATE, InboundTicket, JobId, PublishedCheckpoint,
    Retirement, Retiring, StoreEvent, StoreOutcome, StoreReason, StoreStats, Tier,
    copier::CLOSE_TIMEOUT,
    disk_store::{DiskStore, Staged, TransportEvent},
    geometry::invalid,
    index::Match,
    memory::MemoryStore,
    record::Header,
};

/// A published checkpoint from its memory copy-out to the end of its disk write.
struct Outbound {
    /// The disk parent, pinned in the disk index until the record is reserved under it.
    parent: Option<PrefixKey>,
    memory_parent: Option<PrefixKey>,
    bucket: [u8; 16],
    tail_pages: usize,
    /// The parent's disk write failed while this record's memory copy was in flight.
    parent_failed: bool,
    /// The disk write in progress once the memory copy has landed.
    write: Option<(JobId, Instant)>,
}

enum NodeState {
    /// Pinned in the memory index; its slabs supply the upload.
    Resident,
    /// Allocated in the memory index and being filled from its file.
    Reading { started: Instant },
    /// Another fetch is filling its memory copy; this fetch attaches once the copy is complete.
    Waiting,
}

struct FetchNode {
    key: PrefixKey,
    tail_pages: usize,
    bucket: [u8; 16],
    /// The node is the chain's endpoint, whose row the ticket receives.
    row: bool,
    state: NodeState,
}

struct Inbound {
    ticket: InboundTicket,
    chain: Match,
    /// Tokens the restore saves over prefill.
    tokens: u64,
    nodes: Vec<FetchNode>,
    /// Chain nodes pinned in the disk index while their files may be read.
    disk_pinned: Vec<PrefixKey>,
    /// Read jobs whose replies are awaited.
    reads: usize,
    error: Option<Error>,
    /// The upload from memory slabs has been handed to the memory store.
    uploading: bool,
}

/// Disk writes started for parked records at once.
const MAX_SPILLS: usize = 8;

/// A complete DRAM copy awaiting its disk write.
struct Parked {
    failures: u8,
    writing: bool,
    reserved: bool,
}

/// The composite's jobs on the staged transport, by the ids it issued.
#[derive(Clone, Copy)]
enum DiskJob {
    Write(PrefixKey),
    Read { fetch: PrefixKey, node: PrefixKey },
}

pub struct CompositeStore {
    memory: MemoryStore,
    disk: DiskStore,
    stats: StoreStats,
    outbound: HashMap<PrefixKey, Outbound>,
    inbound: HashMap<PrefixKey, Inbound>,
    jobs: HashMap<JobId, DiskJob>,
    /// Memory copies of records that left the disk index while a job still
    /// pinned them; removed once their pins are released.
    stale: Vec<PrefixKey>,
    disk_outbound_rate: Rate,
    disk_inbound_rate: Rate,
    /// Joined a pool: records are written to disk when memory needs the room.
    pooled: bool,
    pub drain_on_shutdown: Duration,
    /// Records in a pool whose memory copy landed, by key.
    parked: HashMap<PrefixKey, Parked>,
    /// Scratch reused across polls.
    memory_events: Vec<StoreEvent>,
    transport_events: Vec<TransportEvent>,
    maintenance_progress: u64,
    retry_after: Option<Instant>,
}

/// The memory store holds the record's whole tail and, when `row`, its row,
/// and no removal is pending for it.
fn resident_copy(
    memory: &MemoryStore,
    stale: &[PrefixKey],
    key: &PrefixKey,
    tail_pages: usize,
    row: bool,
) -> bool {
    !stale.contains(key)
        && memory.record(key).is_some_and(|record| {
            record.pages.len() == tail_pages && (!row || record.row.is_some())
        })
}

impl CompositeStore {
    /// Compose a flat memory store with a staged disk store. Each tier owns
    /// its transfer queues; disk staging serves records that cannot stay in RAM.
    pub fn new(memory: MemoryStore, mut disk: DiskStore) -> Result<Self> {
        if memory.identity() != disk.identity() {
            return Err(invalid("composite store tiers belong to different models"));
        }
        if memory.layout() != disk.layout() {
            return Err(invalid(
                "composite store tiers describe different record layouts",
            ));
        }
        if !memory.flat() {
            return Err(invalid(
                "the memory store inside a composite must use flat parent links",
            ));
        }
        match disk.transport_mut().staged_mut() {
            None => {
                return Err(invalid(
                    "the composite store requires the staged disk transport",
                ));
            }
            Some(staged) if staged.pool().is_none() => {
                return Err(invalid(
                    "the composite disk transport requires a bounded staging pool",
                ));
            }
            Some(staged) if staged.shares_transfers(memory.transfers()) => {
                return Err(invalid("composite tiers require separate transfer queues"));
            }
            Some(_) => {}
        }
        let memory_stats = memory.stats();
        Ok(Self {
            memory,
            disk,
            stats: StoreStats {
                outbound_rate: memory_stats.outbound_rate,
                inbound_rate: memory_stats.inbound_rate,
                ..StoreStats::default()
            },
            outbound: HashMap::new(),
            inbound: HashMap::new(),
            jobs: HashMap::new(),
            stale: Vec::new(),
            disk_outbound_rate: Rate::new(INITIAL_STAGED_DISK_RATE),
            disk_inbound_rate: Rate::new(INITIAL_STAGED_DISK_RATE),
            pooled: false,
            drain_on_shutdown: Duration::ZERO,
            parked: HashMap::new(),
            memory_events: Vec::new(),
            transport_events: Vec::new(),
            maintenance_progress: 0,
            retry_after: None,
        })
    }

    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    pub fn disk(&self) -> &DiskStore {
        &self.disk
    }

    /// Records whose copy-out to disk or memory has not completed.
    pub fn outbound_records(&self) -> usize {
        self.outbound.len()
    }

    /// Whether the record's copy-out to disk or memory is still pending.
    pub fn copying_out(&self, key: &PrefixKey) -> bool {
        self.outbound.contains_key(key)
    }

    /// Records whose restore has not completed.
    pub fn inbound_records(&self) -> usize {
        self.inbound.len()
    }

    /// Disk jobs the composite has issued and not seen complete.
    pub fn disk_jobs(&self) -> usize {
        self.jobs.len()
    }

    fn resident(&self, key: &PrefixKey, tail_pages: usize, row: bool) -> bool {
        resident_copy(&self.memory, &self.stale, key, tail_pages, row)
    }

    fn staged(&mut self) -> Result<&mut Staged> {
        self.disk
            .transport_mut()
            .staged_mut()
            .ok_or_else(|| invalid("the composite store requires the staged disk transport"))
    }

    /// Host addresses of a ready memory record's page slabs and row slab.
    fn slab_addresses(&self, key: &PrefixKey) -> Result<(Vec<usize>, Option<usize>)> {
        let record = self
            .memory
            .record(key)
            .ok_or_else(|| invalid("the record has no complete memory copy"))?;
        let pool = self.memory.pool();
        Ok((
            record
                .pages
                .iter()
                .map(|slab| pool.page_address(*slab))
                .collect(),
            record.row.map(|slab| pool.row_address(slab)),
        ))
    }

    pub fn retains_chain(&self, chain: &Match) -> bool {
        let mut pages = 0;
        let mut rows = 1;
        for (position, key) in chain.nodes.iter().enumerate() {
            let Some(node) = self.disk.index().node(key) else {
                return false;
            };
            pages += node.tail_pages;
            let row = position + 1 == chain.nodes.len();
            if let Some(memory) = self.memory.index().node(key) {
                if memory.pinned()
                    && (memory.location.pages.len() != node.tail_pages
                        || (row && memory.location.row.is_none())
                        || self.stale.contains(key))
                {
                    return false;
                }
                if !row && memory.location.row.is_some() {
                    rows += 1;
                }
            }
        }
        for node in self.memory.index().nodes() {
            if !chain.nodes.contains(&node.key)
                && (node.pinned() || !node.ready || !node.location.durable)
            {
                pages += node.location.pages.len();
                rows += usize::from(node.location.row.is_some());
            }
        }
        pages <= self.memory.pool().page_count() && rows <= self.memory.pool().row_count()
    }

    fn fail_children(&mut self, key: PrefixKey) {
        for waiting in self.outbound.values_mut() {
            if waiting.parent == Some(key) {
                waiting.parent = None;
                waiting.parent_failed = true;
                let _ = self.disk.index_mut().unpin(&key);
                if waiting.memory_parent.take().is_some() {
                    let _ = self.memory.unpin_record(&key);
                }
            }
        }
    }

    fn streamed_event(&mut self, event: TransportEvent, events: &mut Vec<StoreEvent>) {
        if let TransportEvent::Stored {
            id, result: Err(_), ..
        }
        | TransportEvent::StoreInvalidated { id, .. } = &event
        {
            if let Some(key) = self.disk.pending_store_key(*id) {
                self.fail_children(key);
            }
        }
        let before = self.disk.stats();
        self.disk.apply(event, events);
        let after = self.disk.stats();
        self.stats.stored += after.stored - before.stored;
        self.stats.store_failed += after.store_failed - before.store_failed;
        self.stats.fetched += after.fetched - before.fetched;
        self.stats.fetch_failed += after.fetch_failed - before.fetch_failed;
        self.stats.reused_tokens += after.reused_tokens - before.reused_tokens;
        self.stats.bytes_out += after.bytes_out - before.bytes_out;
        self.stats.bytes_in += after.bytes_in - before.bytes_in;
        if after.stored > before.stored {
            self.stats.outbound_rate = after.outbound_rate;
        }
        if after.fetched > before.fetched {
            self.stats.inbound_rate = after.inbound_rate;
        }
    }

    /// A hit for `chain` priced at the disk rate for the records read from
    /// their files and the memory rate for the whole upload.
    fn hit(&self, chain: Match) -> HostHit {
        let layout = self.disk.layout();
        let last = chain.nodes.len().saturating_sub(1);
        let mut disk_bytes = 0u64;
        for (position, key) in chain.nodes.iter().enumerate() {
            let Some(node) = self.disk.index().node(key) else {
                continue;
            };
            let row = position == last;
            if !self.resident(key, node.tail_pages, row) {
                disk_bytes += node.tail_pages as u64 * layout.page_record_bytes
                    + if row { layout.row_record_bytes } else { 0 };
            }
        }
        let bytes = layout.record_bytes(chain.pages as u64);
        let memory_rate = self.memory.stats().inbound_rate.max(1.0);
        HostHit {
            key: chain.key,
            consumed_tokens: chain.consumed_tokens,
            resident_tokens: chain.resident_tokens,
            pages: chain.pages,
            bytes,
            tier: if disk_bytes == 0 {
                Tier::Dram
            } else {
                Tier::Nvme
            },
            estimated_seconds: disk_bytes as f64 / self.disk_inbound_rate.get().max(1.0)
                + bytes as f64 / memory_rate,
            chain,
        }
    }

    /// Remove a record's memory copy, or schedule the removal when a job still pins it.
    fn remove_copy(&mut self, key: PrefixKey) {
        if self.memory.remove_record(&key).is_err()
            && self.memory.index().contains(&key)
            && !self.stale.contains(&key)
        {
            self.stale.push(key);
        }
    }

    /// Records unlinked from disk lose their memory copies.
    fn drop_evicted_copies(&mut self) {
        for key in self.disk.take_evicted() {
            self.remove_copy(key);
        }
    }

    fn retry_stale(&mut self) {
        let memory = &mut self.memory;
        self.stale
            .retain(|key| memory.remove_record(key).is_err() && memory.index().contains(key));
    }

    fn release_parent(&mut self, outbound: &mut Outbound) {
        if let Some(parent) = outbound.parent.take() {
            let _ = self.disk.index_mut().unpin(&parent);
        }
        if let Some(parent) = outbound.memory_parent.take() {
            let _ = self.memory.unpin_record(&parent);
        }
    }

    /// The memory copy of a published checkpoint has landed or failed.
    fn memory_stored(
        &mut self,
        checkpoint: CheckpointId,
        key: PrefixKey,
        result: Result<u64>,
        events: &mut Vec<StoreEvent>,
    ) {
        let Some(mut outbound) = self.outbound.remove(&key) else {
            return;
        };
        let parent = outbound.parent;
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                self.release_parent(&mut outbound);
                // The memory store removed its record.
                self.stats.store_failed += 1;
                events.push(StoreEvent::Stored {
                    checkpoint,
                    key,
                    result: Err(error),
                });
                return;
            }
        };
        self.stats.bytes_out += bytes;
        self.stats.outbound_rate = self.memory.stats().outbound_rate;
        if outbound.parent_failed {
            self.stats.dropped[2] += 1;
            self.remove_copy(key);
            self.stats.store_failed += 1;
            events.push(StoreEvent::Stored {
                checkpoint,
                key,
                result: Err(invalid(
                    "the record's parent did not reach disk, and its tail pages alone cannot be restored",
                )),
            });
            return;
        }
        let reserved =
            self.disk
                .reserve_record(key, parent, outbound.bucket, outbound.tail_pages, true);
        self.drop_evicted_copies();
        let reserved_file = matches!(&reserved, Ok(Some(_)));
        let reserved = match reserved {
            Ok(None) => self
                .disk
                .reserve_record(key, parent, outbound.bucket, outbound.tail_pages, false)
                .map(|_| ()),
            Ok(Some(_)) => Ok(()),
            Err(error) => Err(error),
        };
        if let Err(error) = reserved {
            self.release_parent(&mut outbound);
            self.remove_copy(key);
            self.stats.store_failed += 1;
            events.push(StoreEvent::Stored {
                checkpoint,
                key,
                result: Err(error),
            });
            return;
        }
        self.stats.held[0] += u64::from(!reserved_file);
        if reserved_file {
            self.release_parent(&mut outbound);
        } else {
            outbound.memory_parent = parent.filter(|p| self.memory.pin_record(p).is_ok());
        }
        self.parked.insert(
            key,
            Parked {
                failures: 0,
                writing: false,
                reserved: reserved_file,
            },
        );
        self.outbound.insert(key, outbound);
        events.push(StoreEvent::Stored {
            checkpoint,
            key,
            result: Ok(bytes),
        });
        if !self.pooled && self.parked.values().filter(|p| p.writing).count() < MAX_SPILLS {
            self.spill(key);
        }
    }

    /// Write the memory record's slabs through to its reserved file.
    fn start_write(
        &mut self,
        key: PrefixKey,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
    ) -> Result<JobId> {
        let (pages, row) = self.slab_addresses(&key)?;
        let row = row.ok_or_else(|| invalid("the memory copy of the record has no row slab"))?;
        let id = self.disk.next_job();
        self.staged()?
            .write_from(id, header, temporary, path, pages, Some(row))?;
        self.jobs.insert(id, DiskJob::Write(key));
        Ok(id)
    }

    /// The disk write of a record whose memory copy is resident has finished.
    fn disk_stored(&mut self, id: JobId, result: Result<u64>) {
        let Some(DiskJob::Write(key)) = self.jobs.get(&id).copied() else {
            return;
        };
        self.jobs.remove(&id);
        let Some(outbound) = self.outbound.remove(&key) else {
            return;
        };
        let pressure = result.as_ref().err().is_some_and(Error::no_space);
        if pressure {
            self.stats.held[0] += 1;
            self.retry_after = Some(Instant::now() + Duration::from_millis(50));
        }
        let retry = pressure
            || self.pooled
                && result.is_err()
                && self.parked.get_mut(&key).is_some_and(|parked| {
                    crate::store::retry_once(&mut parked.failures) == crate::Retirement::Failed
                });
        if retry {
            self.parked
                .get_mut(&key)
                .expect("retrying parked write")
                .writing = false;
            self.outbound.insert(
                key,
                Outbound {
                    write: None,
                    ..outbound
                },
            );
            return;
        }
        self.parked.remove(&key);
        match result {
            Ok(bytes) => {
                if let Some((_, started)) = outbound.write {
                    self.disk_outbound_rate.observe(bytes, started.elapsed());
                }
                match self.disk.commit_record(key, Ok(bytes)) {
                    Ok(()) => {
                        let _ = self.memory.mark_durable(&key);
                        self.stats.stored += 1;
                    }
                    Err(_) => {
                        self.remove_copy(key);
                        self.stats.store_failed += 1;
                    }
                }
            }
            Err(error) => {
                // Children whose memory copies are still in flight hold only
                // their tail pages, so they cannot be written without this
                // record; their pins are released so it can leave the index.
                self.stats.dropped[2] += 1;
                self.fail_children(key);
                let _ = self.disk.commit_record(key, Err(error));
                self.remove_copy(key);
                self.stats.store_failed += 1;
            }
        }
    }

    /// Memory slabs the parked and writing records give back once their files
    /// commit.
    fn spillable_slabs(&self) -> (usize, usize) {
        self.parked
            .keys()
            .filter_map(|key| self.memory.record(key))
            .fold((0, 0), |(pages, rows), record| {
                (
                    pages + record.pages.len(),
                    rows + usize::from(record.row.is_some()),
                )
            })
    }

    /// Start the file write of a parked record.
    fn spill(&mut self, key: PrefixKey) {
        if self.retry_after.is_some_and(|at| Instant::now() < at) {
            return;
        }
        let Some(parked) = self.parked.get(&key) else {
            return;
        };
        if !parked.reserved {
            let outbound = &self.outbound[&key];
            if !matches!(
                self.disk.reserve_record(
                    key,
                    outbound.parent,
                    outbound.bucket,
                    outbound.tail_pages,
                    true
                ),
                Ok(Some(_))
            ) {
                return;
            }
            self.parked.get_mut(&key).unwrap().reserved = true;
            if let Some(mut outbound) = self.outbound.remove(&key) {
                self.release_parent(&mut outbound);
                self.outbound.insert(key, outbound);
            }
        }
        let record = self.disk.record(&key).expect("parked chain metadata");
        let header = record.header.clone();
        let path = record.path.clone();
        let temporary = self
            .disk
            .directory()
            .join(super::record::temporary_name(&key));
        match self.start_write(key, header, temporary, path) {
            Ok(job) => {
                if let Some(outbound) = self.outbound.get_mut(&key) {
                    outbound.write = Some((job, Instant::now()));
                }
                if let Some(parked) = self.parked.get_mut(&key) {
                    parked.writing = true;
                }
            }
            Err(_) => self.stats.held[0] += 1,
        }
    }

    fn waiting_spills(&self, reserve_only: bool) -> Vec<PrefixKey> {
        let mut waiting = BinaryHeap::with_capacity(MAX_SPILLS + 1);
        for (key, parked) in &self.parked {
            if !parked.writing
                && (!reserve_only || !parked.reserved)
                && let Some(node) = self.memory.index().node(key)
            {
                waiting.push((node.accessed(), *key));
                if waiting.len() > MAX_SPILLS {
                    waiting.pop();
                }
            }
        }
        waiting
            .into_sorted_vec()
            .into_iter()
            .map(|(_, key)| key)
            .collect()
    }

    /// Write cold parked records until their slabs can cover the demand.
    fn spill_parked(&mut self, need_pages: usize, need_rows: usize) {
        let waiting = self.waiting_spills(need_pages == 0 && need_rows == 0);
        let pages_of = |store: &Self, key: &PrefixKey| {
            store
                .memory
                .record(key)
                .map_or(0, |record| record.pages.len())
        };
        let mut writing = self.parked.values().filter(|parked| parked.writing).count();
        let mut covered: usize = self
            .parked
            .iter()
            .filter(|(_, parked)| parked.writing)
            .map(|(key, _)| pages_of(self, key))
            .sum();
        let mut rows = writing;
        for key in waiting {
            if covered >= need_pages && rows >= need_rows && self.parked[&key].reserved {
                continue;
            }
            let pages = pages_of(self, &key);
            if writing < MAX_SPILLS {
                self.spill(key);
                if self.parked.get(&key).is_some_and(|parked| parked.writing) {
                    writing += 1;
                    covered += pages;
                    rows += 1;
                }
            }
        }
        if writing == MAX_SPILLS && (covered < need_pages || rows < need_rows) {
            self.stats.held[0] += self.parked.len().saturating_sub(writing) as u64;
        }
    }

    /// Read a chain node's file into freshly allocated memory slabs.
    fn start_read(
        &mut self,
        fetch: PrefixKey,
        key: PrefixKey,
        tail_pages: usize,
        bucket: [u8; 16],
        row: bool,
    ) -> Result<()> {
        let source = self
            .disk
            .record(&key)
            .ok_or_else(|| invalid("host chain names a record that is not indexed on disk"))?;
        let path = source.path.clone();
        let header = source.header.clone();
        let record = self
            .memory
            .allocate_record(key, bucket, tail_pages, row)?
            .ok_or_else(|| invalid("no host memory slabs could be freed for the record"))?;
        let pool = self.memory.pool();
        let pages = record
            .pages
            .iter()
            .map(|slab| pool.page_address(*slab))
            .collect();
        let row = record.row.map(|slab| pool.row_address(slab));
        let id = self.disk.next_job();
        let submitted = self
            .staged()
            .and_then(|staged| staged.read_into(id, path, header, 0, pages, row));
        if let Err(error) = submitted {
            let _ = self.memory.complete_record(&key, false);
            return Err(error);
        }
        self.jobs.insert(id, DiskJob::Read { fetch, node: key });
        Ok(())
    }

    /// Attach waiting nodes to memory copies that became complete and start
    /// reads for those no job owns. A node whose read cannot start fails the fetch.
    fn advance(&mut self, inbound: &mut Inbound) {
        let fetch = inbound.ticket.key;
        for index in 0..inbound.nodes.len() {
            if !matches!(inbound.nodes[index].state, NodeState::Waiting) {
                continue;
            }
            let (key, tail_pages, bucket, row) = {
                let node = &inbound.nodes[index];
                (node.key, node.tail_pages, node.bucket, node.row)
            };
            if self.resident(&key, tail_pages, row) {
                match self.memory.pin_record(&key) {
                    Ok(()) => inbound.nodes[index].state = NodeState::Resident,
                    Err(error) => {
                        inbound.error.get_or_insert(error);
                    }
                }
                continue;
            }
            if let Some(node) = self.memory.index().node(&key) {
                if !node.ready || node.pinned() {
                    // Another job fills or reads this copy; attach once it is released.
                    continue;
                }
                // An incomplete copy nothing references, such as a row-less
                // record restored earlier as an ancestor, gives way to a fresh read.
                if let Err(error) = self.memory.remove_record(&key) {
                    inbound.error.get_or_insert(error);
                    continue;
                }
            }
            if tail_pages == 0 && !row {
                // An ancestor whose endpoint shares its parent's last page
                // contributes no bytes; an empty resident copy stands in for it.
                let placed = self
                    .memory
                    .allocate_record(key, bucket, 0, false)
                    .and_then(|record| {
                        record.ok_or_else(|| invalid("no host memory record could be indexed"))
                    })
                    .and_then(|_| self.memory.complete_record(&key, true))
                    .and_then(|()| self.memory.pin_record(&key));
                match placed {
                    Ok(()) => inbound.nodes[index].state = NodeState::Resident,
                    Err(error) => {
                        inbound.error.get_or_insert(error);
                    }
                }
                continue;
            }
            match self.start_read(fetch, key, tail_pages, bucket, row) {
                Ok(()) => {
                    inbound.nodes[index].state = NodeState::Reading {
                        started: Instant::now(),
                    };
                    inbound.reads += 1;
                    self.disk_inbound_rate.started();
                }
                Err(error) => {
                    inbound.error.get_or_insert(error);
                }
            }
        }
    }

    /// Release every pin and incomplete copy the fetch holds. Copies that
    /// were read completely stay resident.
    fn release(&mut self, inbound: &mut Inbound) {
        for node in inbound.nodes.drain(..) {
            match node.state {
                NodeState::Resident => {
                    let _ = self.memory.unpin_record(&node.key);
                }
                NodeState::Reading { .. } => {
                    let _ = self.memory.complete_record(&node.key, false);
                }
                NodeState::Waiting => {}
            }
        }
        for key in inbound.disk_pinned.drain(..) {
            let _ = self.disk.index_mut().unpin(&key);
        }
    }

    fn fail_inbound(&mut self, mut inbound: Inbound, events: &mut Vec<StoreEvent>) {
        let error = inbound
            .error
            .take()
            .unwrap_or_else(|| invalid("host restore failed"));
        self.release(&mut inbound);
        self.stats.fetch_failed += 1;
        events.push(StoreEvent::Fetched {
            key: inbound.ticket.key,
            result: Err(error),
        });
    }

    /// Advance a fetch: start reads, report a failure once every read has
    /// replied, and hand the chain to the memory store once every node is resident.
    fn settle(&mut self, fetch: PrefixKey, events: &mut Vec<StoreEvent>) {
        let Some(mut inbound) = self.inbound.remove(&fetch) else {
            return;
        };
        if inbound.uploading {
            self.inbound.insert(fetch, inbound);
            return;
        }
        if inbound.error.is_none() {
            self.advance(&mut inbound);
        }
        if inbound.error.is_some() {
            if inbound.reads == 0 {
                self.fail_inbound(inbound, events);
            } else {
                self.inbound.insert(fetch, inbound);
            }
            return;
        }
        if inbound
            .nodes
            .iter()
            .all(|node| matches!(node.state, NodeState::Resident))
        {
            match self
                .memory
                .fetch_chain(&inbound.chain, inbound.ticket.clone())
            {
                Ok(()) => inbound.uploading = true,
                Err(error) => {
                    inbound.error = Some(error);
                    self.fail_inbound(inbound, events);
                    return;
                }
            }
        }
        self.inbound.insert(fetch, inbound);
    }

    /// A read of one chain node's file into memory slabs has finished.
    fn disk_read(&mut self, id: JobId, result: Result<u64>, events: &mut Vec<StoreEvent>) {
        let Some(DiskJob::Read { fetch, node }) = self.jobs.get(&id).copied() else {
            return;
        };
        self.jobs.remove(&id);
        let Some(mut inbound) = self.inbound.remove(&fetch) else {
            // The fetch is gone: the slabs allocated for the read are released.
            let _ = self.memory.complete_record(&node, false);
            return;
        };
        inbound.reads = inbound.reads.saturating_sub(1);
        let position = inbound.nodes.iter().position(|entry| {
            entry.key == node && matches!(entry.state, NodeState::Reading { .. })
        });
        match (result, position) {
            (Ok(bytes), Some(index)) => {
                if let NodeState::Reading { started } = inbound.nodes[index].state {
                    self.disk_inbound_rate.observe(bytes, started.elapsed());
                }
                // The copy is complete: a durable resident, pinned for the upload.
                let pinned = self
                    .memory
                    .complete_record(&node, true)
                    .and_then(|()| self.memory.pin_record(&node));
                match pinned {
                    Ok(()) => inbound.nodes[index].state = NodeState::Resident,
                    Err(error) => {
                        inbound.nodes[index].state = NodeState::Waiting;
                        inbound.error.get_or_insert(error);
                    }
                }
            }
            (Ok(_), None) => {
                // No node of the fetch owns the copy: release it rather than
                // leave it pinned and incomplete.
                let _ = self.memory.complete_record(&node, false);
            }
            (Err(error), _) => {
                inbound.error.get_or_insert(error);
            }
        }
        self.inbound.insert(fetch, inbound);
        self.settle(fetch, events);
    }

    /// The memory store finished uploading a chain.
    fn memory_fetched(
        &mut self,
        key: PrefixKey,
        result: Result<u64>,
        events: &mut Vec<StoreEvent>,
    ) {
        let Some(mut inbound) = self.inbound.remove(&key) else {
            events.push(StoreEvent::Fetched { key, result });
            return;
        };
        self.release(&mut inbound);
        match &result {
            Ok(bytes) => {
                for node in &inbound.chain.nodes {
                    self.disk.index_mut().touch(node);
                }
                self.stats.fetched += 1;
                self.stats.bytes_in += bytes;
                self.stats.reused_tokens += inbound.tokens;
                self.stats.inbound_rate = self.memory.stats().inbound_rate;
            }
            Err(_) => {
                self.stats.fetch_failed += 1;
            }
        }
        events.push(StoreEvent::Fetched { key, result });
    }
}

impl HostStore for CompositeStore {
    fn reused(&mut self, key: PrefixKey) {
        self.memory.reused(key);
        self.disk.reused(key);
    }

    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
        self.stats.lookups += 1;
        let (memory, stale) = (&self.memory, &self.stale);
        let chain = self
            .disk
            .index_mut()
            .lookup_below(history, resident, before, |node| {
                node.ready || resident_copy(memory, stale, &node.key, node.tail_pages, true)
            })?;
        let hit = if self.retains_chain(&chain) {
            self.hit(chain)
        } else {
            let chain = self
                .disk
                .index_mut()
                .lookup_below(history, resident, before, |node| node.ready)?;
            self.disk.hit(chain)
        };
        self.stats.hits += 1;
        Some(hit)
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        if checkpoint.tail_pages > checkpoint.chain_pages.len() {
            self.stats.store_failed += 1;
            return Err(invalid(format!(
                "published checkpoint claims {} tail pages of a {}-page chain",
                checkpoint.tail_pages,
                checkpoint.chain_pages.len()
            )));
        }
        let key = checkpoint.key;
        if let Some(outbound) = self.outbound.get(&key) {
            self.stats.store_skipped += 1;
            // A parked record already has its complete memory copy.
            if self.parked.contains_key(&key) && self.resident(&key, outbound.tail_pages, true) {
                return Ok(StoreOutcome::Present);
            }
            return Ok(StoreOutcome::Retry(StoreReason::Inflight));
        }
        if let Some(outcome) = self.disk.store_status(checkpoint)? {
            self.stats.store_skipped += 1;
            self.stats.dropped[4] += u64::from(outcome == StoreOutcome::Oversize);
            return Ok(outcome);
        }
        if self.disk.index().contains(&key) {
            if self
                .memory
                .index()
                .node(&key)
                .is_some_and(|node| node.pinned())
            {
                self.stats.store_skipped += 1;
                return Ok(StoreOutcome::Retry(StoreReason::Inflight));
            }
            self.remove_copy(key);
            return self.disk.store(checkpoint);
        }
        if self.memory.index().contains(&key) {
            self.stats.store_skipped += 1;
            return Ok(StoreOutcome::Retry(StoreReason::Inflight));
        }
        let (parent, tail) = self.disk.plan_record(checkpoint);
        let (mut pages, mut rows) = self.memory.reclaimable_slabs();
        if !self.parked.is_empty() {
            // Parked records free their slabs once written.
            let (spillable_pages, spillable_rows) = self.spillable_slabs();
            pages += spillable_pages;
            rows += spillable_rows;
        }
        if tail.len() > pages || rows == 0 {
            let result = self.disk.store(checkpoint);
            match &result {
                Ok(outcome) if *outcome != StoreOutcome::Queued => self.stats.store_skipped += 1,
                Err(_) => self.stats.store_failed += 1,
                _ => {}
            }
            self.drop_evicted_copies();
            return result;
        }
        // The parent stays in the disk index while the memory copy is in
        // flight, so the record can be reserved under it once its tail lands.
        if let Some(parent) = &parent
            && let Err(error) = self.disk.index_mut().pin(parent)
        {
            self.stats.store_failed += 1;
            return Err(error);
        }
        match self.memory.store_record(checkpoint, parent, &tail, false) {
            Ok(true) => {}
            Ok(false) => {
                if let Some(parent) = &parent {
                    let _ = self.disk.index_mut().unpin(parent);
                }
                self.stats.store_skipped += 1;
                self.spill_parked(tail.len(), 1);
                return Ok(StoreOutcome::Retry(StoreReason::Slabs));
            }
            Err(error) => {
                if let Some(parent) = &parent {
                    let _ = self.disk.index_mut().unpin(parent);
                }
                self.stats.store_failed += 1;
                return Err(error);
            }
        }
        self.outbound.insert(
            key,
            Outbound {
                parent,
                memory_parent: None,
                bucket: checkpoint.bucket,
                tail_pages: tail.len(),
                parent_failed: false,
                write: None,
            },
        );
        Ok(StoreOutcome::Queued)
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        if !self.retains_chain(&hit.chain) {
            return self.disk.fetch(hit, ticket);
        }
        let chain = hit.chain;
        if ticket.key != chain.key {
            return Err(invalid(
                "inbound ticket and host chain name different endpoint keys",
            ));
        }
        if self.inbound.contains_key(&ticket.key) {
            return Err(invalid("an inbound transfer for this key is in flight"));
        }
        let Some(last) = chain.nodes.len().checked_sub(1) else {
            return Err(invalid("host chain names no records"));
        };
        let mut nodes = Vec::with_capacity(chain.nodes.len());
        let mut pages = 0usize;
        for (position, key) in chain.nodes.iter().enumerate() {
            let node =
                self.disk.index().node(key).ok_or_else(|| {
                    invalid("host chain names a record that is not indexed on disk")
                })?;
            let skip = if position == 0 { chain.skip_pages } else { 0 };
            pages += node.tail_pages.checked_sub(skip).ok_or_else(|| {
                invalid("host chain skips more pages than its first record holds")
            })?;
            let row = position == last;
            if row && !node.row {
                return Err(invalid("host chain ends at a record without a row"));
            }
            let resident = self.resident(key, node.tail_pages, row);
            if !resident && !node.ready {
                return Err(invalid(
                    "host chain names a record that is complete neither in memory nor on disk",
                ));
            }
            nodes.push(FetchNode {
                key: *key,
                tail_pages: node.tail_pages,
                bucket: node.bucket,
                row,
                state: if resident {
                    NodeState::Resident
                } else {
                    NodeState::Waiting
                },
            });
        }
        if pages != ticket.pages.len() || pages != chain.pages {
            return Err(invalid(format!(
                "host chain supplies {pages} pages for a ticket of {} and a match of {}",
                ticket.pages.len(),
                chain.pages
            )));
        }
        let mut inbound = Inbound {
            tokens: chain.consumed_tokens.saturating_sub(chain.resident_tokens),
            ticket,
            chain,
            nodes,
            disk_pinned: Vec::new(),
            reads: 0,
            error: None,
            uploading: false,
        };
        // Pin before any job is issued, so a refusal here leaves nothing behind.
        for index in 0..inbound.nodes.len() {
            let key = inbound.nodes[index].key;
            let pinned = if matches!(inbound.nodes[index].state, NodeState::Resident) {
                self.memory.pin_record(&key)
            } else {
                let pinned = self.disk.index_mut().pin(&key);
                if pinned.is_ok() {
                    inbound.disk_pinned.push(key);
                }
                pinned
            };
            if let Err(error) = pinned {
                inbound.nodes.truncate(index);
                self.release(&mut inbound);
                return Err(error);
            }
        }
        self.advance(&mut inbound);
        if let Some(error) = inbound.error.take() {
            if inbound.reads == 0 {
                self.release(&mut inbound);
                return Err(error);
            }
            // Reads already issued reply later; the failure follows them as an event.
            inbound.error = Some(error);
        } else if inbound
            .nodes
            .iter()
            .all(|node| matches!(node.state, NodeState::Resident))
        {
            if let Err(error) = self
                .memory
                .fetch_chain(&inbound.chain, inbound.ticket.clone())
            {
                self.release(&mut inbound);
                return Err(error);
            }
            inbound.uploading = true;
        }
        self.inbound.insert(inbound.ticket.key, inbound);
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        let mut memory_events = std::mem::take(&mut self.memory_events);
        self.memory.poll(&mut memory_events)?;
        for event in memory_events.drain(..) {
            match event {
                StoreEvent::Stored {
                    checkpoint,
                    key,
                    result,
                } => self.memory_stored(checkpoint, key, result, events),
                StoreEvent::Fetched { key, result } => self.memory_fetched(key, result, events),
                other => events.push(other),
            }
        }
        self.memory_events = memory_events;
        let mut transport_events = std::mem::take(&mut self.transport_events);
        let polled = self.disk.poll_transport(&mut transport_events);
        for event in transport_events.drain(..) {
            self.stats.progress += 1;
            match event {
                TransportEvent::Stored { id, result } if self.jobs.contains_key(&id) => {
                    self.disk_stored(id, result)
                }
                TransportEvent::Fetched { id, result } if self.jobs.contains_key(&id) => {
                    self.disk_read(id, result, events)
                }
                other => self.streamed_event(other, events),
            }
        }
        self.transport_events = transport_events;
        // Fetches attached to another fetch's reads advance as those copies complete.
        let waiting: Vec<PrefixKey> = self
            .inbound
            .iter()
            .filter(|(_, inbound)| {
                !inbound.uploading
                    && inbound.error.is_none()
                    && inbound
                        .nodes
                        .iter()
                        .any(|node| matches!(node.state, NodeState::Waiting))
            })
            .map(|(key, _)| *key)
            .collect();
        for key in waiting {
            self.settle(key, events);
        }
        self.drop_evicted_copies();
        self.retry_stale();
        // The flat memory records are announced under their parents in the disk index.
        // The disk tier announces first, so a record leaving memory for a file
        // is announced as one spill.
        self.disk.flush_announcements();
        let disk = &self.disk;
        self.memory.flush_announcements(
            &|key, _| disk.index().node(key).and_then(|node| node.parent),
            &|key| disk.announced_entry(key),
        );
        let progress = self.stats().progress;
        let retry = self.retry_after.is_some_and(|at| Instant::now() >= at);
        if !self.disk.preserve_committed && (self.maintenance_progress != progress || retry) {
            if retry {
                self.retry_after = None;
            }
            self.maintenance_progress = progress;
            if self.pooled && !retry {
                let (pages, rows) = self.memory.headroom();
                self.spill_parked(
                    pages.saturating_sub(self.memory.pool().free_pages()),
                    rows.saturating_sub(self.memory.pool().free_rows()),
                );
            } else {
                self.spill_parked(usize::MAX, usize::MAX);
            }
            self.disk.maintain();
        }
        polled
    }

    fn stats(&self) -> StoreStats {
        let memory = self.memory.stats();
        let disk = self.disk.stats();
        // A parked record has its outbound entry but no copy running.
        let parked = self
            .parked
            .values()
            .filter(|parked| !parked.writing)
            .count();
        StoreStats {
            outbound_in_flight: self.outbound.len().saturating_sub(parked)
                + disk.outbound_in_flight,
            parked_records: parked,
            inbound_in_flight: self.inbound.len() + disk.inbound_in_flight,
            memory_records: memory.memory_records,
            memory_bytes: memory.memory_bytes,
            disk_records: disk.disk_records,
            disk_bytes: disk.disk_bytes,
            row_refills: disk.row_refills,
            held: std::array::from_fn(|i| self.stats.held[i] + disk.held[i]),
            dropped: std::array::from_fn(|i| self.stats.dropped[i] + disk.dropped[i]),
            progress: disk.progress + memory.progress + self.stats.progress,
            ..self.stats
        }
    }

    /// Finish the copy-outs, disk writes, and restores in flight within
    /// [`CLOSE_TIMEOUT`], then drain the disk and memory tiers. An error leaves
    /// ownership intact until their transfers can be drained.
    fn close(&mut self) -> Result<()> {
        self.disk.preserve_committed = self.pooled;
        let deadline = Instant::now() + CLOSE_TIMEOUT;
        let mut events = Vec::new();
        while Instant::now() < deadline {
            if !self.pooled {
                self.spill_parked(usize::MAX, usize::MAX);
            }
            let stats = self.stats();
            if stats.outbound_in_flight == 0 && stats.inbound_in_flight == 0 {
                break;
            }
            let _ = self.poll(&mut events);
            events.clear();
            std::thread::sleep(Duration::from_millis(1));
        }
        self.disk.close()?;
        self.memory.close()?;
        self.outbound.clear();
        self.parked.clear();
        self.inbound.clear();
        self.jobs.clear();
        self.stale.clear();
        Ok(())
    }
}

impl crate::store::TierStore for CompositeStore {
    fn drain_on_shutdown(&mut self) {
        self.disk.preserve_committed = self.pooled;
        if !self.pooled || self.drain_on_shutdown.is_zero() {
            return;
        }
        let started = Instant::now();
        let mut remaining: std::collections::HashSet<_> = self.outbound.keys().copied().collect();
        let records = remaining.len();
        let layout = self.disk.layout();
        let pages: usize = self.outbound.values().map(|o| o.tail_pages).sum();
        let bytes =
            pages as u64 * layout.page_record_bytes + records as u64 * layout.row_record_bytes;
        eprintln!("prefix cache shutdown drain: records={records} bytes={bytes}");
        let before = self.stats;
        let (mut skipped, mut failed) = (0, 0);
        let mut events = Vec::new();
        while started.elapsed() < self.drain_on_shutdown {
            if self.poll(&mut events).is_err() {
                failed += remaining.len();
                break;
            }
            events.clear();
            if self.stats().outbound_in_flight != 0 {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let Some(mut key) = remaining
                .iter()
                .filter(|k| self.parked.contains_key(k))
                .max_by_key(|k| self.memory.index().node(k).map(|n| n.accessed()))
                .copied()
            else {
                break;
            };
            // Persist the dependencies of the newest endpoint first.
            let index = self.disk.index();
            while let Some(parent) = index
                .node(&key)
                .and_then(|n| n.parent)
                .filter(|p| remaining.contains(p) && self.parked.contains_key(p))
            {
                key = parent;
            }
            if started.elapsed() >= self.drain_on_shutdown {
                break;
            }
            remaining.remove(&key);
            self.spill(key);
            if let Some(p) = self.parked.get(&key).filter(|p| !p.writing) {
                let space = !p.reserved || self.retry_after.is_some();
                skipped += usize::from(space);
                failed += usize::from(!space);
            }
        }
        if self.retry_after.is_some() {
            skipped = self.parked.len();
        }
        let drained = self.stats.stored - before.stored;
        failed += (self.stats.store_failed - before.store_failed) as usize
            + self.parked.values().filter(|p| p.failures > 0).count();
        eprintln!(
            "prefix cache shutdown drain: drained={drained} skipped_for_space={skipped} failed={failed} elapsed={:?} limited={}",
            started.elapsed(),
            started.elapsed() >= self.drain_on_shutdown
        );
    }

    fn announce_to(&mut self, announcer: Announcer) {
        if let Err(error) = self.memory.announce_records(announcer.clone()) {
            eprintln!("warning: host memory records will not be shared: {error}");
        }
        self.disk.announce_records(announcer);
    }
    fn pin_export(
        &mut self,
        tier: PoolTier,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        if tier == PoolTier::Nvme {
            return self.disk.pin_export_files(key, begin_tokens, want_row);
        }
        // The memory records are flat: the disk index supplies the chain, and
        // each record on it must be resident with all its pages (and the row,
        // at the endpoint when wanted).
        let page_tokens = self.disk.layout().page_tokens;
        let start_page = begin_tokens / page_tokens;
        let mut chain = Vec::new();
        let mut cursor = key;
        loop {
            let node = self
                .disk
                .index()
                .node(&cursor)
                .ok_or(ExportRefusal::NotFound)?;
            let row = want_row && cursor == key;
            if !self.resident(&cursor, node.tail_pages, row) {
                return Err(ExportRefusal::ChainGap);
            }
            chain.push(cursor);
            let start =
                (cursor.consumed_tokens / page_tokens).saturating_sub(node.tail_pages as u64);
            if start <= start_page {
                break;
            }
            cursor = node.parent.ok_or(ExportRefusal::ChainGap)?;
        }
        chain.reverse();
        self.memory.pin_export_chain(&chain, begin_tokens, want_row)
    }
    fn cascade(&mut self) {
        self.pooled = true;
    }
    fn set_spill(&mut self, enabled: bool) {
        self.disk.set_spill(enabled);
    }
    fn take_retiring(&mut self) -> Vec<Retiring> {
        self.disk.take_retiring()
    }
    fn retired(&mut self, key: PrefixKey, tier: PoolTier, how: Retirement) {
        self.disk.retired(key, tier, how);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Ids, checkpoint, pattern, test_directory};
    use crate::testing::{StoppedCopier, TestHistory as PrefixHistory};
    use crate::{
        CacheGeometry, INITIAL_MEMORY_RATE, Layout, PageId, StateSlotId,
        disk::DiskThread,
        memory::MemoryRecord,
        record,
        slab::{PageSlab, RowSlab, SlabPool},
        testing::geometry as rank,
    };
    use std::{fs, path::Path};

    const IDENTITY: [u8; 32] = [0x3c; 32];
    const FINGERPRINT: [u8; 16] = [0xc3; 16];
    const PAGE_TOKENS: u64 = 64;
    /// Page record bytes of the test geometry: two 8 KiB planes.
    const PAGE_RECORD: u64 = 16384;
    /// Row record bytes of the test geometry: one 4 KiB row.
    const ROW_RECORD: u64 = 4096;

    fn geometry() -> Vec<CacheGeometry> {
        vec![rank(PAGE_TOKENS, &[("k", 8192), ("v", 8192)], 4096)]
    }

    fn history(tokens: u64) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        history
    }

    fn pages(range: std::ops::Range<u32>) -> Vec<PageId> {
        range.map(PageId).collect()
    }

    /// A composite over `pages` page slabs and `rows` row slabs of
    /// unregistered memory, transfer threads that accept nothing, and a disk
    /// store under `directory` with `limit` bytes.
    fn composite(directory: &Path, pages: usize, rows: usize, limit: u64) -> CompositeStore {
        composite_with_faults(
            directory,
            pages,
            rows,
            limit,
            crate::disk::Faults::default(),
        )
    }

    fn composite_with_faults(
        directory: &Path,
        pages: usize,
        rows: usize,
        limit: u64,
        faults: crate::disk::Faults,
    ) -> CompositeStore {
        let geometry = geometry();
        let transfers = StoppedCopier::shared(&geometry).unwrap();
        let pool = SlabPool::new(PAGE_RECORD, pages, ROW_RECORD, rows, true).unwrap();
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let directory_lock = record::DirectoryLock::acquire(directory).unwrap();
        let scanned = record::scan(&directory_lock, &IDENTITY, &FINGERPRINT, geometry[0].page_tokens, &record::rank_sizes(&geometry)).unwrap();
        let layout = Layout::new(&geometry).unwrap();
        let disk_transfers = StoppedCopier::shared(&geometry).unwrap();
        let disk = DiskStore::new(
            directory_lock,
            IDENTITY,
            FINGERPRINT,
            limit,
            geometry,
            scanned,
            Box::new(Staged::new(
                Some(SlabPool::staging(&layout).unwrap()),
                disk_transfers,
                DiskThread::start_with_faults("flare-offload-disk", faults).unwrap(),
            )),
        )
        .unwrap();
        CompositeStore::new(memory, disk).unwrap()
    }

    /// Fill a slab through its address. The pool's mapping is valid for its
    /// lifetime and the test touches a slab only while no job references it.
    fn fill_page(pool: &SlabPool, slab: PageSlab, seed: u64) {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(pool.page_address(slab) as *mut u8, pool.page_bytes())
        };
        pattern(bytes, seed);
    }

    fn fill_row(pool: &SlabPool, slab: RowSlab, seed: u64) {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(pool.row_address(slab) as *mut u8, pool.row_bytes())
        };
        pattern(bytes, seed);
    }

    /// A complete memory copy of `tail` pages and a row for `key`, as a landed
    /// copy-out leaves it, filled with patterns derived from `seed`.
    fn landed(
        store: &mut CompositeStore,
        key: PrefixKey,
        bucket: [u8; 16],
        tail: usize,
        seed: u64,
    ) -> MemoryRecord {
        let record = store
            .memory
            .allocate_record(key, bucket, tail, true)
            .unwrap()
            .unwrap();
        for (index, slab) in record.pages.iter().enumerate() {
            fill_page(store.memory.pool(), *slab, seed + index as u64);
        }
        fill_row(store.memory.pool(), record.row.unwrap(), seed + 1000);
        store.memory.complete_record(&key, true).unwrap();
        record
    }

    /// The bytes of a memory record's slabs, pages then row.
    fn record_bytes(store: &CompositeStore, record: &MemoryRecord) -> Vec<Vec<u8>> {
        let pool = store.memory.pool();
        let mut bytes: Vec<Vec<u8>> = record
            .pages
            .iter()
            .map(|slab| pool.page(*slab).to_vec())
            .collect();
        if let Some(row) = record.row {
            bytes.push(pool.row(row).to_vec());
        }
        bytes
    }

    /// The memory copy of `key` as a landed copy-out leaves it, with the
    /// outbound state `store` keeps for it under `parent`.
    fn copy_landed(
        store: &mut CompositeStore,
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        seed: u64,
    ) -> (MemoryRecord, u64) {
        let tail = checkpoint.tail_pages;
        let record = landed(store, checkpoint.key, checkpoint.bucket, tail, seed);
        store.outbound.insert(
            checkpoint.key,
            Outbound {
                parent,
                memory_parent: None,
                bucket: checkpoint.bucket,
                tail_pages: tail,
                parent_failed: false,
                write: None,
            },
        );
        (record, tail as u64 * PAGE_RECORD + ROW_RECORD)
    }

    fn poll_until(
        store: &mut CompositeStore,
        events: &mut Vec<StoreEvent>,
        mut done: impl FnMut(&CompositeStore) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done(store) {
            assert!(
                Instant::now() < deadline,
                "condition not reached within 60 s"
            );
            store.poll(events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Write `checkpoint` through to disk from a synthetic memory copy and
    /// wait for the file; returns the memory record and the copy-out bytes.
    fn written(
        store: &mut CompositeStore,
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        seed: u64,
    ) -> (MemoryRecord, u64) {
        let (record, bytes) = copy_landed(store, checkpoint, parent, seed);
        let mut events = Vec::new();
        store.memory_stored(checkpoint.id, checkpoint.key, Ok(bytes), &mut events);
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Stored { result: Ok(_), .. }]
        ));
        events.clear();
        let key = checkpoint.key;
        poll_until(store, &mut events, |store| {
            !store.outbound.contains_key(&key)
        });
        assert!(events.is_empty(), "{events:?}");
        let node = store.disk.index().node(&key).unwrap();
        assert!(node.ready && !node.pinned());
        assert!(store.memory.record(&key).unwrap().durable);
        (record, bytes)
    }

    fn stored(events: &[StoreEvent]) -> Vec<(PrefixKey, bool)> {
        events
            .iter()
            .map(|event| match event {
                StoreEvent::Stored { key, result, .. } => (*key, result.is_ok()),
                StoreEvent::Fetched { .. } => panic!("unexpected fetch event"),
                _ => panic!("unexpected event"),
            })
            .collect()
    }

    fn fetched(events: &[StoreEvent]) -> Vec<(PrefixKey, bool)> {
        events
            .iter()
            .map(|event| match event {
                StoreEvent::Fetched { key, result } => (*key, result.is_ok()),
                StoreEvent::Stored { .. } => panic!("unexpected store event"),
                _ => panic!("unexpected event"),
            })
            .collect()
    }

    #[test]
    fn oversized_chains_stream_without_evicting_retained_records() {
        let Some(directory) = test_directory("composite", "oversized-chain") else {
            return;
        };
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        let mut store = composite(&directory, 32, 4, 1 << 30);
        written(&mut store, &root, None, 1);
        written(&mut store, &child, Some(root.key), 2);
        store.close().unwrap();
        drop(store);

        // Both an oversized record and individually fitting records whose
        // combined chain exceeds retention must use the streaming path.
        for capacity in [4, 15] {
            let mut store = composite(&directory, capacity, 2, 1 << 30);
            let useful_record =
                checkpoint(&mut ids, &full, 100, None, pages(0..1), 1, StateSlotId(1));
            let useful = useful_record.key;
            // Replace the retained copy after reopening the directory.
            let retained = landed(&mut store, useful, useful_record.bucket, 1, 3);
            let bytes = record_bytes(&store, &retained);
            let hit = store.lookup(&full, 0, u64::MAX).unwrap();
            assert_eq!((hit.key, hit.tier, hit.pages), (child.key, Tier::Nvme, 20));
            assert!(!store.retains_chain(&hit.chain));
            store
                .fetch(
                    hit,
                    InboundTicket {
                        key: child.key,
                        pages: pages(30..50),
                        slot: StateSlotId(3),
                    },
                )
                .unwrap();
            assert!(store.inbound.is_empty());
            assert_eq!(store.disk.stats().inbound_in_flight, 1);
            assert_eq!(
                record_bytes(&store, store.memory.record(&useful).unwrap()),
                bytes
            );
            let mut events = Vec::new();
            poll_until(&mut store, &mut events, |store| {
                store.stats().inbound_in_flight == 0
            });
            // Stopped CUDA workers fail the upload after disk I/O; the record
            // remains retriable and no retained slab was borrowed or evicted.
            assert_eq!(fetched(&events), [(child.key, false)]);
            assert_eq!(store.stats().fetch_failed, 1);
            assert_eq!(
                record_bytes(&store, store.memory.record(&useful).unwrap()),
                bytes
            );
            assert!(store.disk.index().nodes().all(|node| !node.pinned()));
            assert!(store.lookup(&full, 0, u64::MAX).is_some());
            store.close().unwrap();
        }
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn streamed_store_failure_preserves_ram_and_releases_parent_pins() {
        let Some(directory) = test_directory("composite", "streamed-failure") else {
            return;
        };
        let mut store = composite(&directory, 4, 2, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            800,
            Some(700),
            pages(0..12),
            2,
            StateSlotId(2),
        );
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.disk.stats().outbound_in_flight, 1);
        assert!(store.outbound.is_empty());
        assert_eq!(store.memory.pool().free_pages(), 4);
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.outbound[&child.key].parent, Some(root.key));
        let parent_pins = store.disk.index().node(&root.key).unwrap().pins();
        assert!(parent_pins > 0);
        // Deliver the parent's failure first, while the child's copy is still
        // pending, and verify that its dependency cannot retain a dead parent.
        let mut events = Vec::new();
        let mut transport = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(60);
        while transport.is_empty() {
            assert!(Instant::now() < deadline);
            store.disk.poll_transport(&mut transport).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        for event in transport {
            store.streamed_event(event, &mut events);
        }
        assert_eq!(stored(&events), [(root.key, false)]);
        assert!(store.outbound[&child.key].parent_failed);
        assert_eq!(store.outbound[&child.key].parent, None);
        assert!(!store.disk.index().contains(&root.key));
        poll_until(&mut store, &mut events, |store| {
            store.stats().outbound_in_flight == 0
        });
        assert_eq!(stored(&events), [(root.key, false), (child.key, false)]);
        assert_eq!(store.stats().store_failed, 2);
        assert!(store.memory.index().is_empty());
        assert!(store.disk.index().is_empty());
        assert_eq!(
            (
                store.memory.pool().free_pages(),
                store.memory.pool().free_rows()
            ),
            (4, 2)
        );
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn retention_planning_accounts_for_pinned_and_incomplete_copies() {
        let Some(directory) = test_directory("composite", "retention-pressure") else {
            return;
        };
        let mut store = composite(&directory, 20, 3, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        written(&mut store, &root, None, 1);
        store.memory.remove_record(&root.key).unwrap();
        let other = full.key_at(1000).unwrap();
        landed(&mut store, other, full.bucket_at(1000).unwrap(), 15, 2);
        let chain = store
            .disk
            .index_mut()
            .lookup_below(&full, 0, u64::MAX, |node| node.ready)
            .unwrap();
        assert!(store.retains_chain(&chain));
        store.memory.pin_record(&other).unwrap();
        assert!(!store.retains_chain(&chain));
        store.memory.unpin_record(&other).unwrap();
        assert!(store.retains_chain(&chain));
        store.memory.remove_record(&other).unwrap();
        store
            .memory
            .allocate_record(other, full.bucket_at(1000).unwrap(), 15, true)
            .unwrap()
            .unwrap();
        assert!(!store.retains_chain(&chain));
        store.memory.complete_record(&other, true).unwrap();
        assert!(store.retains_chain(&chain));
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn new_requires_matching_flat_tiers_and_independent_staged_transfers() {
        let Some(directory) = test_directory("composite", "new") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let transfers = StoppedCopier::shared(&geometry).unwrap();
        let pool = || SlabPool::new(PAGE_RECORD, 4, ROW_RECORD, 2, true).unwrap();
        let staging = || Some(SlabPool::staging(&layout).unwrap());
        let disk = |identity: [u8; 32], staging: Option<SlabPool>, shared: bool| {
            let disk_transfers = if shared {
                transfers.clone()
            } else {
                StoppedCopier::shared(&geometry).unwrap()
            };
            DiskStore::new(
                record::DirectoryLock::acquire(&directory).unwrap(),
                identity,
                FINGERPRINT,
                1 << 30,
                geometry.clone(),
                Vec::new(),
                Box::new(Staged::new(
                    staging,
                    disk_transfers,
                    DiskThread::start("flare-offload-disk").unwrap(),
                )),
            )
            .unwrap()
        };
        let flat = || MemoryStore::new_flat(IDENTITY, pool(), transfers.clone()).unwrap();
        let tree = MemoryStore::new(IDENTITY, pool(), transfers.clone()).unwrap();
        assert!(CompositeStore::new(tree, disk(IDENTITY, staging(), false)).is_err());
        assert!(CompositeStore::new(flat(), disk([1; 32], staging(), false)).is_err());
        assert!(CompositeStore::new(flat(), disk(IDENTITY, None, false)).is_err());
        assert!(CompositeStore::new(flat(), disk(IDENTITY, staging(), true)).is_err());
        let mut store = CompositeStore::new(flat(), disk(IDENTITY, staging(), false)).unwrap();
        assert_eq!(
            store.stats(),
            StoreStats {
                outbound_rate: INITIAL_MEMORY_RATE,
                inbound_rate: INITIAL_MEMORY_RATE,
                ..StoreStats::default()
            }
        );
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn rejected_disk_endpoint_does_not_hide_a_warm_ancestor() {
        let Some(directory) = test_directory("composite", "cost-fallback") else {
            return;
        };
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        let mut store = composite(&directory, 32, 4, 1 << 30);
        written(&mut store, &root, None, 1);
        written(&mut store, &child, Some(root.key), 2);
        store.memory.remove_record(&child.key).unwrap();
        let deep = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(deep.key, child.key);
        assert!(!super::super::worth_fetching(&deep, 20e6));
        let warm = store.lookup(&full, 0, deep.consumed_tokens).unwrap();
        assert_eq!((warm.key, warm.tier), (root.key, Tier::Dram));
        assert!(super::super::worth_fetching(&warm, 20e6));
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn store_dedupes_pins_the_parent_and_reports_a_failed_memory_copy() {
        let Some(directory) = test_directory("composite", "store") else {
            return;
        };
        let mut store = composite(&directory, 48, 4, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        written(&mut store, &root, None, 1);
        assert_ne!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);

        // A child is shaped by the disk index and holds its parent in place
        // while its memory copy is in flight.
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_ne!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 2);
        assert_eq!(store.stats().outbound_in_flight, 1);
        assert_eq!(store.disk.index().node(&root.key).unwrap().pins(), 1);
        assert_eq!(store.outbound[&child.key].parent, Some(root.key));
        assert_eq!(store.outbound[&child.key].tail_pages, 10);
        assert!(!store.disk.index().contains(&child.key));

        // Without transfer threads the memory copy fails: the event reports
        // it, the parent is released, and nothing is left in either tier.
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert_eq!(stored(&events), [(child.key, false)]);
        assert!(!store.memory.index().contains(&child.key));
        assert!(!store.disk.index().contains(&child.key));
        assert_eq!(store.disk.index().node(&root.key).unwrap().pins(), 0);
        assert!(store.outbound.is_empty());
        let stats = store.stats();
        assert_eq!(
            (stats.store_failed, stats.stored, stats.outbound_in_flight),
            (1, 1, 0)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (1, 1));

        // A checkpoint whose parent is unknown is planned as a root record
        // holding its whole chain.
        let orphan = checkpoint(
            &mut ids,
            &full,
            1990,
            Some(1500),
            pages(0..31),
            8,
            StateSlotId(3),
        );
        assert_eq!(store.store(&orphan).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.outbound[&orphan.key].parent, None);
        assert_eq!(store.outbound[&orphan.key].tail_pages, 31);
        store.poll(&mut events).unwrap();

        let mut short = checkpoint(&mut ids, &full, 1000, None, pages(0..3), 3, StateSlotId(1));
        short.tail_pages = 15;
        assert!(store.store(&short).is_err());
        assert_eq!(store.stats().store_failed, 3);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn write_through_reserves_the_file_and_marks_the_memory_copy_durable() {
        let Some(directory) = test_directory("composite", "write-through") else {
            return;
        };
        let mut store = composite(&directory, 48, 4, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let (record, bytes) = copy_landed(&mut store, &root, None, 1);
        let mut events = Vec::new();
        store.memory_stored(root.id, root.key, Ok(bytes), &mut events);
        match &events[..] {
            [
                StoreEvent::Stored {
                    checkpoint,
                    key,
                    result,
                },
            ] => {
                assert_eq!((*checkpoint, *key), (root.id, root.key));
                assert_eq!(*result.as_ref().unwrap(), bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        events.clear();
        let node = store.disk.index().node(&root.key).unwrap();
        assert!(!node.ready && node.pinned() && node.row);
        assert_eq!(node.location.header.tail_pages, 10);
        assert!(store.outbound[&root.key].write.is_some());
        assert_eq!(store.jobs.len(), 1);

        // While the write is in flight the record matches lookups from memory.
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier, hit.pages), (root.key, Tier::Dram, 10));
        assert_eq!(hit.bytes, 10 * PAGE_RECORD + ROW_RECORD);
        assert!((hit.estimated_seconds - hit.bytes as f64 / INITIAL_MEMORY_RATE).abs() < 1e-12);
        assert!(store.lookup(&history(300), 0, u64::MAX).is_none());
        assert_eq!((store.stats().lookups, store.stats().hits), (2, 1));

        poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
        assert!(events.is_empty(), "{events:?}");
        let node = store.disk.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned());
        let header = node.location.header.clone();
        let path = directory.join(record::file_name(&root.key));
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());
        assert!(!directory.join(record::temporary_name(&root.key)).exists());
        assert!(store.memory.record(&root.key).unwrap().durable);
        assert!(store.jobs.is_empty());
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.bytes_out, stats.store_failed),
            (1, bytes, 0)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (1, 1));
        assert_eq!(stats.memory_bytes, 10 * PAGE_RECORD + ROW_RECORD);
        assert_eq!(stats.disk_bytes, header.length_with_row());
        assert_eq!(stats.outbound_in_flight, 0);

        // The file holds the slab bytes: every page record and the row.
        let file = fs::read(&path).unwrap();
        let slabs = record_bytes(&store, &record);
        for (index, page) in slabs[..10].iter().enumerate() {
            let start = (header.page_segment + index as u64 * PAGE_RECORD) as usize;
            assert!(
                file[start..start + PAGE_RECORD as usize] == page[..],
                "page {index}"
            );
        }
        let row = header.row_segment as usize;
        assert!(file[row..row + ROW_RECORD as usize] == slabs[10][..]);

        // An infeasible record is refused before copying.
        let small_directory = directory.join("small");
        fs::create_dir(&small_directory).unwrap();
        let mut small = composite(&small_directory, 32, 4, 1);
        let big = checkpoint(
            &mut ids,
            &full,
            1300,
            None,
            pages(0..20),
            20,
            StateSlotId(2),
        );
        assert_eq!(small.store(&big).unwrap(), StoreOutcome::Oversize);
        assert_eq!(small.stats().memory_records, 0);
        assert_eq!(small.stats().disk_records, 0);
        assert_eq!(small.stats().bytes_out, 0);
        small.close().unwrap();
        drop(small);

        // A parent that left the disk index fails the child, whose memory copy
        // holds only its tail pages.
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        let missing = full.key_at(1000).unwrap();
        let (_, bytes) = copy_landed(&mut store, &child, Some(missing), 3);
        store.memory_stored(child.id, child.key, Ok(bytes), &mut events);
        assert_eq!(stored(&events), [(child.key, false)]);
        events.clear();
        assert!(!store.memory.index().contains(&child.key));
        assert_eq!(store.stats().store_failed, 1);

        // A failed memory copy only counts; the memory store removed its record.
        store.outbound.insert(
            child.key,
            Outbound {
                parent: Some(root.key),
                memory_parent: None,
                bucket: child.bucket,
                tail_pages: 10,
                parent_failed: false,
                write: None,
            },
        );
        store.disk.index_mut().pin(&root.key).unwrap();
        store.memory_stored(
            child.id,
            child.key,
            Err(invalid("copy engine fault")),
            &mut events,
        );
        assert_eq!(stored(&events), [(child.key, false)]);
        events.clear();
        assert_eq!(store.disk.index().node(&root.key).unwrap().pins(), 0);
        assert_eq!(store.stats().store_failed, 2);

        // Under its parent the child is written as a child record.
        let (_, child_bytes) = written(&mut store, &child, Some(root.key), 4);
        let node = store.disk.index().node(&child.key).unwrap();
        assert_eq!(node.parent, Some(root.key));
        assert_eq!(node.location.header.parent, Some(root.key));
        assert_eq!(node.tail_pages, 10);
        assert_eq!(store.memory.index().node(&child.key).unwrap().parent, None);
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [root.key, child.key]);
        assert_eq!(hit.tier, Tier::Dram);
        let stats = store.stats();
        // Copy-out bytes count every landed memory copy: the root, the child
        // whose parent was missing, and the child written under its parent.
        assert_eq!(
            (stats.stored, stats.bytes_out),
            (2, bytes + bytes + child_bytes)
        );
        assert!(stats.outbound_rate > 0.0);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_failed_disk_write_removes_the_memory_copy_and_fails_waiting_children() {
        let Some(directory) = test_directory("composite", "write-failure") else {
            return;
        };
        let mut store = composite(&directory, 16, 4, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        let blocked = directory.join(record::temporary_name(&root.key));
        fs::create_dir(&blocked).unwrap();
        let (_, bytes) = copy_landed(&mut store, &root, None, 1);
        let mut events = Vec::new();
        store.memory_stored(root.id, root.key, Ok(bytes), &mut events);
        assert_eq!(stored(&events), [(root.key, true)]);
        events.clear();
        // The child's copy-out is in flight under the root: it holds the root pinned.
        store.outbound.insert(
            child.key,
            Outbound {
                parent: Some(root.key),
                memory_parent: None,
                bucket: child.bucket,
                tail_pages: 10,
                parent_failed: false,
                write: None,
            },
        );
        store.disk.index_mut().pin(&root.key).unwrap();
        assert_eq!(store.disk.index().node(&root.key).unwrap().pins(), 2);

        poll_until(&mut store, &mut events, |store| {
            !store.outbound.contains_key(&root.key)
        });
        assert!(events.is_empty(), "{events:?}");
        assert!(!store.memory.index().contains(&root.key));
        assert!(!store.disk.index().contains(&root.key));
        assert!(!directory.join(record::file_name(&root.key)).exists());
        assert!(blocked.is_dir());
        let stats = store.stats();
        assert_eq!(
            (stats.store_failed, stats.stored, stats.store_skipped),
            (1, 0, 0)
        );
        assert_eq!(
            (stats.memory_records, stats.disk_records, stats.disk_bytes),
            (0, 0, 0)
        );
        assert_eq!(stats.outbound_in_flight, 1);
        let waiting = &store.outbound[&child.key];
        assert!(waiting.parent_failed && waiting.parent.is_none());
        fs::remove_dir(&blocked).unwrap();

        // The child's copy lands without a parent to extend: it fails too.
        let record = landed(&mut store, child.key, child.bucket, 10, 2);
        assert_eq!(record.pages.len(), 10);
        store.memory_stored(child.id, child.key, Ok(bytes), &mut events);
        assert_eq!(stored(&events), [(child.key, false)]);
        assert!(!store.memory.index().contains(&child.key));
        assert!(store.outbound.is_empty());
        assert_eq!(store.stats().store_failed, 2);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn read_through_restores_a_file_into_memory_before_the_upload() {
        let Some(directory) = test_directory("composite", "read-through") else {
            return;
        };
        let mut store = composite(&directory, 48, 4, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let (record, _) = written(&mut store, &root, None, 1);
        let original = record_bytes(&store, &record);
        store.memory.remove_record(&root.key).unwrap();
        assert_eq!(store.stats().memory_records, 0);

        // The hit comes from disk and is priced for the file read plus the upload.
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier, hit.pages), (root.key, Tier::Nvme, 10));
        let bytes = 10 * PAGE_RECORD + ROW_RECORD;
        assert_eq!(hit.bytes, bytes);
        let expected = bytes as f64 / INITIAL_STAGED_DISK_RATE + bytes as f64 / INITIAL_MEMORY_RATE;
        assert!((hit.estimated_seconds - expected).abs() < 1e-12);
        let ticket = |pages: u32, slot: u32| InboundTicket {
            key: root.key,
            pages: (100..100 + pages).map(PageId).collect(),
            slot: StateSlotId(slot),
        };

        // Refusals leave no pins and no memory allocation behind.
        assert!(store.fetch(hit.clone(), ticket(9, 2)).is_err());
        let mut wrong = ticket(10, 2);
        wrong.key = full.key_at(300).unwrap();
        assert!(store.fetch(hit.clone(), wrong).is_err());
        let mut hollow = hit.clone();
        hollow.chain.nodes.clear();
        assert!(store.fetch(hollow, ticket(10, 2)).is_err());
        let mut stranger = hit.clone();
        stranger.chain.nodes[0] = full.key_at(300).unwrap();
        assert!(store.fetch(stranger, ticket(10, 2)).is_err());
        assert!(!store.disk.index().node(&root.key).unwrap().pinned());
        assert!(!store.memory.index().contains(&root.key));
        assert!(store.inbound.is_empty());

        // The fetch pins the file, allocates slabs, and reads the record.
        store.fetch(hit.clone(), ticket(10, 2)).unwrap();
        assert!(store.fetch(hit.clone(), ticket(10, 3)).is_err());
        assert_eq!(store.stats().inbound_in_flight, 1);
        assert_eq!(store.disk.index().node(&root.key).unwrap().pins(), 1);
        let node = store.memory.index().node(&root.key).unwrap();
        assert!(!node.ready && node.pinned());
        assert_eq!(node.location.pages.len(), 10);
        assert!(node.location.row.is_some());
        assert_eq!(store.jobs.len(), 1);
        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| {
            !store.inbound.is_empty() && store.jobs.is_empty()
        });
        // The copy is complete and resident; the upload fails without
        // transfer threads, which the event reports once.
        poll_until(&mut store, &mut events, |store| store.inbound.is_empty());
        assert_eq!(fetched(&events), [(root.key, false)]);
        let node = store.memory.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned() && node.location.durable);
        assert_eq!(record_bytes(&store, &node.location), original);
        assert!(!store.disk.index().node(&root.key).unwrap().pinned());
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.fetch_failed, stats.inbound_in_flight),
            (0, 1, 0)
        );
        assert!(stats.inbound_rate > 0.0);
        assert!(store.disk_inbound_rate.get() != INITIAL_STAGED_DISK_RATE);

        // Resident again, the record is a memory hit whose upload is handed
        // to the memory store at once; without transfer threads that upload
        // fails before `fetch` returns, leaving only the composite's pin.
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Dram);
        store.fetch(hit, ticket(10, 2)).unwrap();
        assert!(store.inbound[&root.key].uploading);
        assert_eq!(store.memory.index().node(&root.key).unwrap().pins(), 1);
        assert!(!store.disk.index().node(&root.key).unwrap().pinned());
        events.clear();
        poll_until(&mut store, &mut events, |store| store.inbound.is_empty());
        assert_eq!(fetched(&events), [(root.key, false)]);
        assert!(!store.memory.index().node(&root.key).unwrap().pinned());
        assert_eq!(store.stats().fetch_failed, 2);

        // A row-less ancestor copy gives way to a fresh read when the record
        // is the endpoint, and a chain mixes resident and read records.
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        written(&mut store, &child, Some(root.key), 2);
        store.memory.remove_record(&root.key).unwrap();
        let partial = store
            .memory
            .allocate_record(root.key, root.bucket, 10, false)
            .unwrap()
            .unwrap();
        assert!(partial.row.is_none());
        store.memory.complete_record(&root.key, true).unwrap();
        let hit = store.lookup(&history(700), 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier), (root.key, Tier::Nvme));
        store.fetch(hit, ticket(10, 2)).unwrap();
        let node = store.memory.index().node(&root.key).unwrap();
        assert!(!node.ready && node.location.row.is_some());
        events.clear();
        poll_until(&mut store, &mut events, |store| store.inbound.is_empty());
        assert_eq!(fetched(&events), [(root.key, false)]);
        assert_eq!(
            record_bytes(&store, store.memory.record(&root.key).unwrap()),
            original
        );

        let hit = store.lookup(&full, 500, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [root.key, child.key]);
        assert_eq!(
            (hit.chain.skip_pages, hit.pages, hit.tier),
            (7, 13, Tier::Dram)
        );
        store.memory.remove_record(&child.key).unwrap();
        let hit = store.lookup(&full, 500, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Nvme);
        let expected = (10 * PAGE_RECORD + ROW_RECORD) as f64 / store.disk_inbound_rate.get()
            + hit.bytes as f64 / store.memory.stats().inbound_rate;
        assert!((hit.estimated_seconds - expected).abs() < 1e-9);
        store
            .fetch(
                hit,
                InboundTicket {
                    key: child.key,
                    pages: (100..113).map(PageId).collect(),
                    slot: StateSlotId(3),
                },
            )
            .unwrap();
        assert_eq!(store.memory.index().node(&root.key).unwrap().pins(), 1);
        assert_eq!(store.disk.index().node(&child.key).unwrap().pins(), 1);
        assert!(!store.disk.index().node(&root.key).unwrap().pinned());
        events.clear();
        poll_until(&mut store, &mut events, |store| store.inbound.is_empty());
        assert_eq!(fetched(&events), [(child.key, false)]);
        assert!(store.memory.record(&child.key).unwrap().durable);
        assert!(store.memory.index().nodes().all(|node| !node.pinned()));
        assert!(store.disk.index().nodes().all(|node| !node.pinned()));
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_slow_disk_read_does_not_refuse_disk_restores_for_good() {
        let Some(directory) = test_directory("composite", "slow-read-recovers") else {
            return;
        };
        let mut store = composite(&directory, 32, 4, 1 << 30);
        let full = history(1280);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 640, None, pages(0..10), 10, StateSlotId(1));
        written(&mut store, &root, None, 1);
        store.memory.remove_record(&root.key).unwrap();
        let seeded = store.lookup(&full, 0, u64::MAX).unwrap().estimated_seconds;
        // The only disk read so far ran under contention.
        store.disk_inbound_rate.observe(1 << 20, Duration::from_secs(5));
        let slow = store.lookup(&full, 0, u64::MAX).unwrap().estimated_seconds;
        assert!(slow > 1000.0 * seeded, "{slow} vs {seeded}");
        // Refused restores read nothing; the estimate recovers on its own.
        store.disk_inbound_rate.age(Duration::from_secs(200));
        let later = store.lookup(&full, 0, u64::MAX).unwrap().estimated_seconds;
        assert!(later < 2.0 * seeded, "{later} vs {seeded}");
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_fetch_attaches_to_a_record_another_fetch_is_reading() {
        let Some(directory) = test_directory("composite", "attach") else {
            return;
        };
        let mut store = composite(&directory, 16, 4, 1 << 30);
        let full = history(400);
        let mut branch = history(133);
        branch.append(&[5; 100], &[]).unwrap();
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 133, None, pages(0..2), 2, StateSlotId(1));
        let left = checkpoint(
            &mut ids,
            &full,
            197,
            Some(133),
            pages(0..3),
            1,
            StateSlotId(2),
        );
        let right = checkpoint(
            &mut ids,
            &branch,
            197,
            Some(133),
            pages(0..3),
            1,
            StateSlotId(3),
        );
        written(&mut store, &root, None, 1);
        written(&mut store, &left, Some(root.key), 2);
        written(&mut store, &right, Some(root.key), 3);
        for key in [root.key, left.key, right.key] {
            store.memory.remove_record(&key).unwrap();
        }
        let ticket = |key, slot| InboundTicket {
            key,
            pages: (100..103).map(PageId).collect(),
            slot: StateSlotId(slot),
        };
        let first = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(first.chain.nodes, [root.key, left.key]);
        store.fetch(first, ticket(left.key, 1)).unwrap();
        let second = store.lookup(&branch, 0, u64::MAX).unwrap();
        assert_eq!(second.chain.nodes, [root.key, right.key]);
        store.fetch(second, ticket(right.key, 2)).unwrap();
        assert!(matches!(
            store.inbound[&right.key].nodes[0].state,
            NodeState::Waiting
        ));
        assert!(matches!(
            store.inbound[&right.key].nodes[1].state,
            NodeState::Reading { .. }
        ));
        assert_eq!(store.disk.index().node(&root.key).unwrap().pins(), 2);
        assert_eq!(store.memory.index().node(&root.key).unwrap().pins(), 1);

        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| store.inbound.is_empty());
        let mut outcomes = fetched(&events);
        outcomes.sort_by_key(|(key, _)| key.digest);
        let mut expected = vec![(left.key, false), (right.key, false)];
        expected.sort_by_key(|(key, _)| key.digest);
        assert_eq!(outcomes, expected);
        for key in [root.key, left.key, right.key] {
            let node = store.memory.index().node(&key).unwrap();
            assert!(node.ready && !node.pinned() && node.location.durable);
            assert!(!store.disk.index().node(&key).unwrap().pinned());
        }
        assert_eq!(store.stats().memory_records, 3);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn disk_eviction_removes_memory_copies_once_their_pins_are_released() {
        let Some(directory) = test_directory("composite", "eviction") else {
            return;
        };
        let full = history(4000);
        let mut ids = Ids::new();
        // Record files of 10, 20, 31, and 15 pages: a header block, a checksum
        // block, the page records, and the row record.
        let file_bytes = |pages: u64| 2 * 4096 + pages * PAGE_RECORD + ROW_RECORD;
        let limit = 900_000;
        assert!(file_bytes(10) + file_bytes(20) <= limit);
        assert!(file_bytes(10) + file_bytes(20) + file_bytes(31) > limit);
        assert!(file_bytes(20) + file_bytes(31) <= limit);
        assert!(file_bytes(20) + file_bytes(31) + file_bytes(15) > limit);
        assert!(file_bytes(31) + file_bytes(15) <= limit);
        let mut store = composite(&directory, 80, 8, limit);
        let first = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let second = checkpoint(
            &mut ids,
            &full,
            1300,
            None,
            pages(0..20),
            20,
            StateSlotId(2),
        );
        let third = checkpoint(
            &mut ids,
            &full,
            1990,
            None,
            pages(0..31),
            31,
            StateSlotId(3),
        );
        let fourth = checkpoint(
            &mut ids,
            &full,
            1000,
            None,
            pages(0..15),
            15,
            StateSlotId(4),
        );
        let mut events = Vec::new();

        // Both records fit; the third evicts the oldest, whose memory copy
        // goes with its file.
        written(&mut store, &first, None, 1);
        written(&mut store, &second, None, 2);
        assert_eq!(store.disk.used(), file_bytes(10) + file_bytes(20));
        let (_, bytes) = copy_landed(&mut store, &third, None, 3);
        store.memory_stored(third.id, third.key, Ok(bytes), &mut events);
        assert_eq!(stored(&events), [(third.key, true)]);
        events.clear();
        assert!(!store.disk.index().contains(&first.key));
        assert!(!store.memory.index().contains(&first.key));
        assert!(store.disk.index().contains(&second.key));
        assert!(store.memory.index().contains(&second.key));
        poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
        assert!(!directory.join(record::file_name(&first.key)).exists());
        assert!(store.memory.record(&third.key).unwrap().durable);

        // A pinned memory copy outlives its file until the pin is released.
        store.memory.pin_record(&second.key).unwrap();
        let (_, bytes) = copy_landed(&mut store, &fourth, None, 4);
        store.memory_stored(fourth.id, fourth.key, Ok(bytes), &mut events);
        assert_eq!(stored(&events), [(fourth.key, true)]);
        events.clear();
        assert!(!store.disk.index().contains(&second.key));
        assert!(store.memory.index().contains(&second.key));
        assert_eq!(store.stale, [second.key]);
        assert_eq!(
            store.lookup(&history(1300), 0, u64::MAX).unwrap().key,
            fourth.key
        );
        store.poll(&mut events).unwrap();
        assert!(store.memory.index().contains(&second.key));
        store.memory.unpin_record(&second.key).unwrap();
        poll_until(&mut store, &mut events, |store| store.stale.is_empty());
        assert!(!store.memory.index().contains(&second.key));
        poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
        assert!(events.is_empty(), "{events:?}");
        let stats = store.stats();
        assert_eq!(
            (stats.memory_records, stats.disk_records, stats.stored),
            (2, 2, 4)
        );
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn sigkill_during_restore_preserves_committed_records() {
        use std::io::{BufRead, Write};
        use std::os::unix::process::ExitStatusExt;
        use std::process::{Command, Stdio};

        const CHILD_DIR: &str = "FLARE_TEST_COMPOSITE_CRASH_DIRECTORY";
        if let Some(directory) = std::env::var_os(CHILD_DIR) {
            let directory = Path::new(&directory);
            let mut store = composite(directory, 48, 4, 1 << 30);
            let full = history(2000);
            let mut ids = Ids::new();
            let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
            let child = checkpoint(
                &mut ids,
                &full,
                1500,
                Some(700),
                pages(0..23),
                13,
                StateSlotId(2),
            );
            written(&mut store, &root, None, 1);
            written(&mut store, &child, Some(root.key), 2);
            store.memory.remove_record(&root.key).unwrap();
            store.memory.remove_record(&child.key).unwrap();
            // An interrupted write has not been renamed to a checkpoint file.
            let pending = full.key_at(1800).unwrap();
            fs::write(
                directory.join(record::temporary_name(&pending)),
                b"partial write",
            )
            .unwrap();
            let hit = store.lookup(&full, 0, u64::MAX).unwrap();
            assert_eq!(hit.key, child.key);
            store
                .fetch(
                    hit,
                    InboundTicket {
                        key: child.key,
                        pages: pages(100..123),
                        slot: StateSlotId(3),
                    },
                )
                .unwrap();
            assert_eq!(store.inbound.len(), 1);
            assert!(store.disk.index().node(&root.key).unwrap().pinned());
            assert!(store.disk.index().node(&child.key).unwrap().pinned());
            println!("restore-started");
            std::io::stdout().flush().unwrap();
            // The parent kills us with reads outstanding; no store close or Drop runs.
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).unwrap();
            panic!("expected SIGKILL");
        }

        let Some(directory) = test_directory("composite", "sigkill") else {
            return;
        };
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "composite::tests::sigkill_during_restore_preserves_committed_records",
                "--nocapture",
            ])
            .env(CHILD_DIR, &directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let ready = std::io::BufReader::new(stdout)
                .lines()
                .any(|line| line.unwrap().contains("restore-started"));
            let _ = send.send(ready);
        });
        let ready = receive.recv_timeout(Duration::from_secs(60));
        child.kill().unwrap();
        let status = child.wait().unwrap();
        reader.join().unwrap();
        assert!(ready.unwrap());
        assert_eq!(status.signal(), Some(libc::SIGKILL));

        let full = history(2000);
        let root = full.key_at(700).unwrap();
        let endpoint = full.key_at(1500).unwrap();
        let pending = directory.join(record::temporary_name(&full.key_at(1800).unwrap()));
        assert!(pending.exists());
        let before: Vec<_> = [root, endpoint]
            .map(|key| {
                let path = directory.join(record::file_name(&key));
                (path.clone(), fs::read(path).unwrap())
            })
            .into();
        // The killed process's directory lock and read pins do not survive it.
        let mut store = composite(&directory, 48, 4, 1 << 30);
        assert!(!pending.exists());
        assert_eq!(store.stats().disk_records, 2);
        for (path, bytes) in &before {
            assert_eq!(&fs::read(path).unwrap(), bytes);
        }
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier, hit.pages), (endpoint, Tier::Nvme, 23));
        store
            .fetch(
                hit,
                InboundTicket {
                    key: endpoint,
                    pages: pages(100..123),
                    slot: StateSlotId(3),
                },
            )
            .unwrap();
        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| store.inbound.is_empty());
        // The CPU fixture stops the GPU upload; the verified disk reads still fill RAM.
        assert_eq!(fetched(&events), [(endpoint, false)]);
        for key in [root, endpoint] {
            let node = store.disk.index().node(&key).unwrap();
            let file = fs::read(&node.location.path).unwrap();
            let header = &node.location.header;
            let memory = record_bytes(&store, store.memory.record(&key).unwrap());
            for (index, page) in memory[..header.tail_pages as usize].iter().enumerate() {
                let start = header.page_segment as usize + index * PAGE_RECORD as usize;
                assert_eq!(page, &file[start..start + PAGE_RECORD as usize]);
            }
            let start = header.row_segment as usize;
            let record = store.memory.record(&key).unwrap();
            if record.row.is_some() {
                assert_eq!(
                    memory.last().unwrap(),
                    &file[start..start + ROW_RECORD as usize]
                );
            }
        }
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn close_finishes_a_pending_disk_write() {
        let Some(directory) = test_directory("composite", "close") else {
            return;
        };
        let mut store = composite(&directory, 16, 4, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let (_, bytes) = copy_landed(&mut store, &root, None, 1);
        let mut events = Vec::new();
        store.memory_stored(root.id, root.key, Ok(bytes), &mut events);
        assert!(!store.outbound.is_empty());
        store.close().unwrap();
        assert!(store.outbound.is_empty() && store.jobs.is_empty());
        let node = store.disk.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned());
        assert_eq!(
            fs::metadata(directory.join(record::file_name(&root.key)))
                .unwrap()
                .len(),
            node.location.header.length_with_row()
        );
        assert!(store.memory.record(&root.key).unwrap().durable);
        assert_eq!(store.stats().stored, 1);
        // The closed store still answers: the key is indexed, so it is skipped.
        assert_ne!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn memory_and_disk_records_are_announced_under_the_disk_parents_and_exported_per_tier() {
        use crate::pool::{Announcer, HostWindow, Notice, Tier as PoolTier};
        let Some(directory) = test_directory("composite", "announce") else {
            return;
        };
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        let mut store = composite(&directory, 32, 4, 1 << 30);
        let announcer = Announcer::new(std::sync::Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer.clone());
        let (_, root_bytes) = written(&mut store, &root, None, 1);
        written(&mut store, &child, Some(root.key), 2);
        // Memory records are hashed before they are announced.
        let mut notices = Vec::new();
        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| {
            let _ = store;
            if let Some(batch) = announcer.take_batch(100) {
                notices.extend(batch);
            }
            notices
                .iter()
                .filter(|n| matches!(n, Notice::Saved(e) if e.tier == PoolTier::Dram))
                .count()
                == 2
        });
        let entry = |key: PrefixKey, tier: PoolTier| {
            notices
                .iter()
                .find_map(|n| match n {
                    Notice::Saved(e) if e.key == key && e.tier == tier => Some(e.clone()),
                    _ => None,
                })
                .unwrap()
        };
        for tier in [PoolTier::Dram, PoolTier::Nvme] {
            assert_eq!(entry(child.key, tier).parent, Some(root.key));
            assert_eq!(entry(root.key, tier).parent, None);
        }
        // DRAM serves the resident copy, NVMe the file; both with committed checksums.
        let mut pins = Vec::new();
        for tier in [PoolTier::Dram, PoolTier::Nvme] {
            let mut pin = store.pin_export(tier, child.key, 0, true).unwrap();
            assert_eq!(pin.chain().total_pages, 20);
            let mut dst = vec![0u8; (PAGE_RECORD + ROW_RECORD) as usize];
            let window = HostWindow {
                record: 1,
                first_page: 9,
                page_count: 1,
                row: true,
            };
            let sums = pin.read(&window, &mut dst).unwrap();
            assert_eq!(
                sums.pages[0],
                crate::record::checksum(&dst[..PAGE_RECORD as usize])
            );
            assert_eq!(
                sums.row,
                Some(crate::record::checksum(&dst[PAGE_RECORD as usize..]))
            );
            pins.push((tier, dst));
        }
        assert_eq!(pins[0].1, pins[1].1, "both tiers serve the same bytes");
        let _ = root_bytes;
        // Releasing happens when the pins go.
        drop(pins);
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert!(store.memory.index().nodes().all(|n| !n.pinned()));
        assert!(store.disk.index().nodes().all(|n| !n.pinned()));
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    /// A record of `tokens` tokens in its own history (a root), landed in
    /// memory and parked behind its disk reservation.
    fn parked_root(
        store: &mut CompositeStore,
        ids: &mut Ids,
        salt: u32,
        tokens: u64,
    ) -> PublishedCheckpoint {
        let mut history = PrefixHistory::new(IDENTITY);
        let list: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(salt))
            .collect();
        history.append(&list, &[]).unwrap();
        let pages = (tokens / PAGE_TOKENS) as u32;
        let root = checkpoint(
            ids,
            &history,
            tokens,
            None,
            self::pages(0..pages),
            pages as usize,
            StateSlotId(salt),
        );
        let (_, bytes) = copy_landed(store, &root, None, u64::from(salt));
        // A copy-out is not durable until its file commits.
        store.memory.mark_volatile(&root.key);
        let mut events = Vec::new();
        store.memory_stored(root.id, root.key, Ok(bytes), &mut events);
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Stored { result: Ok(_), .. }]
        ));
        root
    }

    #[test]
    fn filesystem_full_keeps_the_record_until_space_returns() {
        for (pooled, pages) in [(false, 12), (true, 12), (true, 64)] {
            let Some(directory) = test_directory("composite", "filesystem-full") else {
                return;
            };
            let flag = directory.join("full");
            let faults = crate::disk::Faults {
                disk_full: Some(flag.clone()),
                ..Default::default()
            };
            let mut store = composite_with_faults(&directory, pages, 4, 1 << 30, faults);
            if pooled {
                store.cascade();
            }
            fs::write(&flag, []).unwrap();
            let root = parked_root(&mut store, &mut Ids::new(), 1, 640);
            store.spill_parked(10, 1);
            poll_until(&mut store, &mut Vec::new(), |s| s.retry_after.is_some());
            let deadline = Instant::now() + Duration::from_millis(120);
            while Instant::now() < deadline {
                store.poll(&mut Vec::new()).unwrap();
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(store.memory.record(&root.key).is_some());
            assert!(!store.memory.record(&root.key).unwrap().durable);
            assert_eq!(store.parked[&root.key].failures, 0);
            assert_eq!(store.stats().store_failed, 0);
            assert_eq!(store.stats().dropped, [0; 6]);
            assert!(store.stats().held[0] > 0);
            fs::remove_file(flag).unwrap();
            poll_until(&mut store, &mut Vec::new(), |s| s.outbound.is_empty());
            assert!(store.memory.record(&root.key).unwrap().durable);
            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn an_infeasible_chain_is_oversize_before_copying() {
        for durable in [false, true] {
            let Some(directory) = test_directory("composite", "infeasible-chain") else {
                return;
            };
            let mut store = composite(&directory, 32, 4, 1 << 30);
            if !durable {
                store.cascade();
            }
            let mut ids = Ids::new();
            let full = history(1280);
            let root = checkpoint(&mut ids, &full, 640, None, pages(0..10), 10, StateSlotId(1));
            let (_, bytes) = copy_landed(&mut store, &root, None, 1);
            store.memory.mark_volatile(&root.key);
            store.memory_stored(root.id, root.key, Ok(bytes), &mut Vec::new());
            if durable {
                poll_until(&mut store, &mut Vec::new(), |s| s.outbound.is_empty());
            }
            let required = store
                .disk
                .record(&root.key)
                .unwrap()
                .header
                .length_with_row();
            store.disk.set_limit(required * 2 - 1);
            let child = checkpoint(
                &mut ids,
                &full,
                1280,
                Some(640),
                pages(0..20),
                10,
                StateSlotId(2),
            );
            let before = store.stats().bytes_out;
            assert_eq!(store.store(&child).unwrap(), StoreOutcome::Oversize);
            assert_eq!(store.stats().dropped[4], 1);
            assert_eq!(store.stats().bytes_out, before);
            assert!(!store.memory.index().contains(&child.key));
            assert!(!store.disk.index().contains(&child.key));
            store.disk.set_limit(required * 2);
            assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn a_durable_parent_can_release_its_dram_slabs_for_a_child_copy() {
        let Some(directory) = test_directory("composite", "parent-slabs") else {
            return;
        };
        let mut store = composite(&directory, 12, 4, 1 << 30);
        let mut ids = Ids::new();
        let full = history(1280);
        let root = checkpoint(&mut ids, &full, 640, None, pages(0..10), 10, StateSlotId(1));
        written(&mut store, &root, None, 1);
        let child = checkpoint(
            &mut ids,
            &full,
            1280,
            Some(640),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert!(store.memory.record(&root.key).is_none());
        assert!(store.disk.index().node(&root.key).unwrap().ready);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_burst_waits_for_slow_handovers_without_losing_records() {
        let Some(directory) = test_directory("composite", "slow-handovers") else {
            return;
        };
        let mut store = composite(&directory, 64, 16, 1 << 30);
        let mut ids = Ids::new();
        let mut events = Vec::new();
        let mut old = Vec::new();
        for salt in 1..=3 {
            old.push(parked_root(&mut store, &mut ids, salt, 640));
            poll_until(&mut store, &mut events, |s| s.outbound.is_empty());
        }
        store.disk.set_limit(store.disk.used());
        store.cascade();
        let announcer =
            crate::pool::Announcer::new(std::sync::Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer);
        store.disk.flush_announcements();
        store.set_spill(true);
        let burst: Vec<_> = (4..=6)
            .map(|salt| parked_root(&mut store, &mut ids, salt, 640))
            .collect();
        for _ in 0..10 {
            store.poll(&mut events).unwrap();
        }
        assert!(old.iter().all(|r| store.disk.index().contains(&r.key)));
        assert!(burst.iter().all(|r| store.memory.record(&r.key).is_some()));
        assert_eq!(store.parked.len(), burst.len());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !burst
            .iter()
            .all(|r| store.disk.index().node(&r.key).is_some_and(|n| n.ready))
        {
            assert!(
                Instant::now() < deadline,
                "parked {:?}, disk {:?}",
                store
                    .parked
                    .iter()
                    .map(|(k, p)| (k.consumed_tokens, p.reserved, p.writing))
                    .collect::<Vec<_>>(),
                store
                    .disk
                    .index()
                    .nodes()
                    .map(|n| (n.key.digest, n.ready, n.pins()))
                    .collect::<Vec<_>>()
            );
            for r in store.take_retiring() {
                let how = if old.iter().any(|o| o.key == r.key) {
                    Retirement::Moved
                } else {
                    Retirement::Failed
                };
                store.retired(r.key, r.tier, how);
            }
            if old.iter().all(|r| !store.disk.index().contains(&r.key)) {
                store.set_spill(false);
            }
            store.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(store.stats().stored, 6);
        assert_eq!(store.stats().store_failed, 0);
        assert_eq!(store.stats().dropped, [0; 6]);
        assert!(store.stats().held[0] > 0);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_completed_spill_advances_disk_retirement_without_another_store() {
        let Some(directory) = test_directory("composite", "disk-progress") else {
            return;
        };
        let mut store = composite(&directory, 32, 4, 1 << 30);
        store.cascade();
        let announcer =
            crate::pool::Announcer::new(std::sync::Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer);
        let root = parked_root(&mut store, &mut Ids::new(), 1, 640);
        store.disk.set_limit(store.disk.used());
        store.set_spill(true);
        store.disk.maintain();
        assert!(store.take_retiring().is_empty());
        store.spill_parked(10, 1);
        poll_until(&mut store, &mut Vec::new(), |s| {
            s.disk
                .index()
                .node(&root.key)
                .is_some_and(|n| n.ready && n.pinned())
        });
        let retiring = store.take_retiring();
        assert_eq!(retiring.len(), 1);
        assert_eq!(retiring[0].key, root.key);
        store.retired(root.key, PoolTier::Nvme, Retirement::Failed);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn progress_spills_parked_records_at_the_headroom_watermark() {
        let Some(directory) = test_directory("composite", "headroom") else {
            return;
        };
        let mut store = composite(&directory, 11, 4, 1 << 30);
        store.cascade();
        let mut ids = Ids::new();
        let root = parked_root(&mut store, &mut ids, 1, 640);
        assert!(store.jobs.is_empty());
        store.poll(&mut Vec::new()).unwrap();
        assert!(store.parked[&root.key].writing);
        poll_until(&mut store, &mut Vec::new(), |s| s.outbound.is_empty());
        assert!(store.memory.record(&root.key).unwrap().durable);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn disk_pressure_parks_complete_dram_records_in_both_modes() {
        for pooled in [false, true] {
            let Some(directory) = test_directory("composite", "disk-pressure-parked") else {
                return;
            };
            let mut store = composite(&directory, 80, 16, 1 << 30);
            if pooled {
                store.cascade();
            }
            let mut ids = Ids::new();
            let root = parked_root(&mut store, &mut ids, 1, 640);
            store.spill_parked(10, 0);
            let mut events = Vec::new();
            poll_until(&mut store, &mut events, |s| s.outbound.is_empty());
            store.disk.set_limit(store.disk.used());
            store.disk.index_mut().pin(&root.key).unwrap();
            let held = parked_root(&mut store, &mut ids, 2, 640);
            assert!(!store.parked[&held.key].reserved);
            assert!(store.memory.record(&held.key).is_some());
            assert!(!store.memory.record(&held.key).unwrap().durable);
            let mut history = PrefixHistory::new(IDENTITY);
            history
                .append(
                    &(0..640u32)
                        .map(|t| t.wrapping_mul(2654435761).wrapping_add(2))
                        .collect::<Vec<_>>(),
                    &[],
                )
                .unwrap();
            assert_eq!(store.lookup(&history, 0, u64::MAX).unwrap().key, held.key);
            assert_eq!(store.store(&held).unwrap(), StoreOutcome::Present);
            store.disk.index_mut().unpin(&root.key).unwrap();
            store.spill_parked(10, 0);
            poll_until(&mut store, &mut events, |s| s.outbound.is_empty());
            assert!(store.disk.index().node(&held.key).unwrap().ready);
            assert!(store.memory.record(&held.key).unwrap().durable);
            let disk = store.disk.record(&held.key).unwrap();
            let mut expected = disk.header.clone();
            let memory = store.memory.record(&held.key).unwrap();
            expected.row_checksum = record::checksum(store.memory.pool().row(memory.row.unwrap()));
            assert_eq!(
                Header::decode(&fs::read(&disk.path).unwrap()).unwrap(),
                expected
            );
            assert_eq!(store.stats.store_failed, 0);
            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn unreserved_child_keeps_its_parent_in_memory() {
        let Some(directory) = test_directory("composite", "parked-parent") else {
            return;
        };
        let mut store = composite(&directory, 80, 16, 1 << 30);
        let mut ids = Ids::new();
        let full = history(1280);
        let root = checkpoint(&mut ids, &full, 640, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1280,
            Some(640),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        written(&mut store, &root, None, 1);
        store.cascade();
        store.disk.set_limit(store.disk.used());
        store.disk.index_mut().pin(&root.key).unwrap();
        let (_, bytes) = copy_landed(&mut store, &child, Some(root.key), 2);
        store.memory.mark_volatile(&child.key);
        store.memory_stored(child.id, child.key, Ok(bytes), &mut Vec::new());
        assert!(!store.parked[&child.key].reserved);
        assert!(store.memory.remove_record(&root.key).is_err());
        assert_eq!(store.lookup(&full, 0, u64::MAX).unwrap().key, child.key);
        store.disk.set_limit(1 << 30);
        store.spill_parked(10, 0);
        assert!(!store.memory.index().node(&root.key).unwrap().pinned());
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn sigkill_after_pooled_publication_recovers_only_committed_files() {
        use std::io::{BufRead, Write};
        use std::os::unix::process::ExitStatusExt;
        use std::process::{Command, Stdio};

        const CHILD_DIR: &str = "FLARE_TEST_POOLED_CRASH_DIRECTORY";
        if let Some(directory) = std::env::var_os(CHILD_DIR) {
            let directory = Path::new(&directory);
            let mut store = composite(directory, 24, 4, 1 << 30);
            store.cascade();
            let mut ids = Ids::new();
            let committed = parked_root(&mut store, &mut ids, 1, 640);
            store.spill_parked(10, 0);
            let mut events = Vec::new();
            poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
            assert!(directory.join(record::file_name(&committed.key)).exists());
            let parked = parked_root(&mut store, &mut ids, 2, 640);
            assert!(!directory.join(record::file_name(&parked.key)).exists());
            let stats = store.stats();
            assert_eq!(
                (
                    stats.memory_records,
                    stats.disk_records,
                    stats.parked_records
                ),
                (2, 2, 1)
            );
            println!("published");
            std::io::stdout().flush().unwrap();
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).unwrap();
            panic!("expected SIGKILL");
        }

        let Some(directory) = test_directory("composite", "pooled-sigkill") else {
            return;
        };
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "composite::tests::sigkill_after_pooled_publication_recovers_only_committed_files",
                "--nocapture",
            ])
            .env(CHILD_DIR, &directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            let ready = std::io::BufReader::new(stdout)
                .lines()
                .any(|line| line.unwrap() == "published");
            let _ = send.send(ready);
        });
        let ready = receive.recv_timeout(Duration::from_secs(60));
        child.kill().unwrap();
        let status = child.wait().unwrap();
        reader.join().unwrap();
        assert!(ready.unwrap());
        assert_eq!(status.signal(), Some(libc::SIGKILL));

        let mut store = composite(&directory, 24, 4, 1 << 30);
        store.cascade();
        assert_eq!(store.stats().disk_records, 1);
        for salt in 1..=2u32 {
            let mut history = PrefixHistory::new(IDENTITY);
            let tokens: Vec<u32> = (0..640u32)
                .map(|t| t.wrapping_mul(2654435761).wrapping_add(salt))
                .collect();
            history.append(&tokens, &[]).unwrap();
            let hit = store.lookup(&history, 0, u64::MAX);
            if salt == 1 {
                assert_eq!(hit.unwrap().tier, Tier::Nvme);
            } else {
                assert!(hit.is_none());
            }
        }
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn shutdown_drain_sigterm_recovery() {
        use std::io::{BufRead, Write};
        use std::process::{Command, Stdio};
        use std::sync::atomic::{AtomicBool, Ordering};
        static STOP: AtomicBool = AtomicBool::new(false);
        extern "C" fn stop(_: libc::c_int) {
            STOP.store(true, Ordering::SeqCst);
        }
        const CHILD: &str = "FLARE_TEST_SHUTDOWN_DIRECTORY";
        const MODE: &str = "FLARE_TEST_SHUTDOWN_MODE";
        if let Some(directory) = std::env::var_os(CHILD) {
            unsafe {
                libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t);
            }
            let directory = Path::new(&directory);
            let mode = std::env::var(MODE).unwrap();
            let mut store = composite_with_faults(
                directory,
                80,
                8,
                1 << 30,
                crate::disk::Faults {
                    delay_ms: if mode == "limit" { 100 } else { 0 },
                    ..Default::default()
                },
            );
            store.cascade();
            let mut ids = Ids::new();
            let committed = parked_root(&mut store, &mut ids, 1, 640);
            store.spill(committed.key);
            poll_until(&mut store, &mut Vec::new(), |s| s.outbound.is_empty());
            let path = directory.join(record::file_name(&committed.key));
            fs::write(directory.join("committed-before"), fs::read(&path).unwrap()).unwrap();
            let modified = fs::metadata(&path).unwrap().modified().unwrap();
            for salt in 2..=4 {
                parked_root(&mut store, &mut ids, salt, 640);
            }
            // Access the oldest publication last, so MRU differs from publication order.
            let key = store
                .memory
                .index()
                .nodes()
                .find(|n| n.key == shutdown_key(2))
                .unwrap()
                .key;
            store.reused(key);
            store.drain_on_shutdown = match mode.as_str() {
                "on" => Duration::MAX,
                "limit" => Duration::from_millis(30),
                _ => Duration::ZERO,
            };
            println!("ready");
            std::io::stdout().flush().unwrap();
            while !STOP.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            let started = Instant::now();
            store.drain_on_shutdown();
            store.close().unwrap();
            let bound = if mode == "off" {
                Duration::from_millis(200)
            } else {
                Duration::from_secs(5)
            };
            assert!(started.elapsed() < bound);
            assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), modified);
            assert_eq!(
                fs::read(&path).unwrap(),
                fs::read(directory.join("committed-before")).unwrap()
            );
            return;
        }
        for mode in ["off", "on", "limit"] {
            let Some(directory) = test_directory("composite", "shutdown-sigterm") else {
                return;
            };
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "composite::tests::shutdown_drain_sigterm_recovery",
                    "--nocapture",
                ])
                .env(CHILD, &directory)
                .env(MODE, mode)
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            let stdout = child.stdout.take().unwrap();
            let (send, receive) = std::sync::mpsc::channel();
            let reader = std::thread::spawn(move || {
                for line in std::io::BufReader::new(stdout).lines() {
                    if line.unwrap() == "ready" {
                        let _ = send.send(true);
                    }
                }
            });
            let ready = receive.recv_timeout(Duration::from_secs(60));
            if ready != Ok(true) {
                child.kill().unwrap();
            } else {
                assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
            }
            let status = child.wait().unwrap();
            reader.join().unwrap();
            assert_eq!(ready, Ok(true));
            assert!(status.success(), "{mode}: {status}");
            let mut store = composite(&directory, 80, 8, 1 << 30);
            for salt in 1..=4 {
                let key = shutdown_key(salt);
                let expected = salt == 1 || mode == "on" || (mode == "limit" && salt == 2);
                assert_eq!(
                    store.disk.index().contains(&key),
                    expected,
                    "{mode}: {salt}"
                );
                if expected {
                    let node = store.disk.record(&key).unwrap();
                    assert_eq!(node.header.key, key);
                    let bytes = fs::read(&node.path).unwrap();
                    let header = &node.header;
                    for page in 0..10 {
                        let mut expected = vec![0; PAGE_RECORD as usize];
                        pattern(&mut expected, u64::from(salt) + page);
                        let start = (header.page_segment + page * PAGE_RECORD) as usize;
                        assert_eq!(&bytes[start..start + expected.len()], expected);
                    }
                    let mut expected = vec![0; ROW_RECORD as usize];
                    pattern(&mut expected, u64::from(salt) + 1000);
                    let start = header.row_segment as usize;
                    assert_eq!(&bytes[start..start + expected.len()], expected);
                }
            }
            store.close().unwrap();
            drop(store);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    fn shutdown_key(salt: u32) -> PrefixKey {
        let mut history = PrefixHistory::new(IDENTITY);
        history
            .append(
                &(0..640u32)
                    .map(|t| t.wrapping_mul(2654435761).wrapping_add(salt))
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap();
        history.key_at(640).unwrap()
    }

    #[test]
    fn shutdown_drain_keeps_committed_files_when_disk_is_full() {
        let Some(directory) = test_directory("composite", "shutdown-full") else {
            return;
        };
        let mut store = composite(&directory, 80, 8, 1 << 30);
        store.cascade();
        let mut ids = Ids::new();
        let committed = parked_root(&mut store, &mut ids, 1, 640);
        store.spill(committed.key);
        poll_until(&mut store, &mut Vec::new(), |s| s.outbound.is_empty());
        let path = directory.join(record::file_name(&committed.key));
        let before = fs::read(&path).unwrap();
        store.disk.set_limit(store.disk.used());
        store.disk.index_mut().pin(&committed.key).unwrap();
        let parked = parked_root(&mut store, &mut ids, 2, 640);
        store.disk.index_mut().unpin(&committed.key).unwrap();
        assert!(!store.parked[&parked.key].reserved);
        store.drain_on_shutdown = Duration::MAX;
        store.drain_on_shutdown();
        assert!(!path.with_file_name(record::file_name(&parked.key)).exists());
        assert_eq!(fs::read(&path).unwrap(), before);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn shutdown_drain_recovers_a_complete_parent_chain() {
        let Some(directory) = test_directory("composite", "shutdown-chain") else {
            return;
        };
        let mut store = composite(&directory, 80, 8, 1 << 30);
        store.cascade();
        let full = history(1280);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 640, None, pages(0..10), 10, StateSlotId(1));
        let child = checkpoint(
            &mut ids,
            &full,
            1280,
            Some(640),
            pages(10..20),
            10,
            StateSlotId(2),
        );
        for (record, parent) in [(&root, None), (&child, Some(root.key))] {
            if let Some(parent) = parent {
                store.disk.index_mut().pin(&parent).unwrap();
            }
            let (_, bytes) = copy_landed(&mut store, record, parent, 1);
            store.memory.mark_volatile(&record.key);
            store.memory_stored(record.id, record.key, Ok(bytes), &mut Vec::new());
        }
        assert!(store.parked.len() == 2);
        store.drain_on_shutdown = Duration::MAX;
        store.drain_on_shutdown();
        assert_eq!(store.stats.stored, 2);
        store.close().unwrap();
        drop(store);
        let mut store = composite(&directory, 80, 8, 1 << 30);
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, child.key);
        assert_eq!(hit.chain.nodes, [root.key, child.key]);
        assert_eq!(hit.chain.pages, 20);
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn shutdown_drain_write_failure_does_not_fail_close() {
        let Some(directory) = test_directory("composite", "shutdown-failure") else {
            return;
        };
        let (gate, ready, release) = crate::disk::IoGate::new();
        drop((ready, release));
        let mut store = composite_with_faults(
            &directory,
            80,
            8,
            1 << 30,
            crate::disk::Faults {
                io_gate: Some(gate),
                ..Default::default()
            },
        );
        store.cascade();
        parked_root(&mut store, &mut Ids::new(), 1, 640);
        store.drain_on_shutdown = Duration::MAX;
        let started = Instant::now();
        store.drain_on_shutdown();
        assert!(store.parked.values().any(|p| p.failures == 1));
        store.close().unwrap();
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn a_pooled_record_waits_in_memory_and_is_written_when_memory_needs_the_room() {
        let Some(directory) = test_directory("composite", "pooled-parked") else {
            return;
        };
        let mut store = composite(&directory, 16, 4, 1 << 30);
        store.cascade();
        let mut ids = Ids::new();
        let root = parked_root(&mut store, &mut ids, 1, 640);

        // The copy landed and the file is reserved, but nothing is written, and
        // nothing is in flight: the record is parked, not copying.
        let stats = store.stats();
        assert_eq!((stats.outbound_in_flight, stats.parked_records), (0, 1));
        let node = store.disk.index().node(&root.key).unwrap();
        assert!(!node.ready && node.pinned() && node.row);
        assert!(store.outbound[&root.key].write.is_none());
        assert!(store.jobs.is_empty());
        assert!(!store.memory.record(&root.key).unwrap().durable);
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert!(!directory.join(record::file_name(&root.key)).exists());
        // It serves lookups from memory, and a republication finds it present.
        let mut history = PrefixHistory::new(IDENTITY);
        let list: Vec<u32> = (0..640u32)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(1))
            .collect();
        history.append(&list, &[]).unwrap();
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier), (root.key, Tier::Dram));
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Present);

        // A record that cannot fit beside it makes memory write it.
        let mut other_history = PrefixHistory::new(IDENTITY);
        let list: Vec<u32> = (0..640u32)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(2))
            .collect();
        other_history.append(&list, &[]).unwrap();
        let other = checkpoint(
            &mut ids,
            &other_history,
            640,
            None,
            pages(0..10),
            10,
            StateSlotId(2),
        );
        assert_eq!(
            store.store(&other).unwrap(),
            StoreOutcome::Retry(StoreReason::Slabs)
        );
        assert!(store.parked[&root.key].writing);
        assert_eq!(store.jobs.len(), 1);
        // Once its write runs it is in flight and no longer parked.
        let stats = store.stats();
        assert_eq!((stats.outbound_in_flight, stats.parked_records), (1, 0));
        poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
        assert_eq!(store.stats().parked_records, 0);
        let node = store.disk.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned());
        assert!(store.memory.record(&root.key).unwrap().durable);
        assert!(store.parked.is_empty());
        assert_eq!(store.stats().stored, 1);

        // Written once, its memory copy now gives way to the new record.
        assert_eq!(store.store(&other).unwrap(), StoreOutcome::Queued);
        assert!(store.memory.record(&root.key).is_none());
        assert!(store.disk.index().node(&root.key).unwrap().ready);
        assert_eq!(store.stats().stored, 1);
        store.close().unwrap();
    }

    #[test]
    fn a_failed_write_is_tried_once_more_and_then_the_record_is_dropped() {
        let Some(directory) = test_directory("composite", "pooled-failed-write") else {
            return;
        };
        for recovers in [true, false] {
            let mut store = composite(&directory, 12, 4, 1 << 30);
            store.cascade();
            let mut ids = Ids::new();
            let root = parked_root(&mut store, &mut ids, 1, 640);
            let mut events = Vec::new();

            // The directory disappears under the first write.
            fs::remove_dir_all(&directory).unwrap();
            store.spill_parked(10, 0);
            let deadline = Instant::now() + Duration::from_secs(10);
            while store.parked[&root.key].writing {
                assert!(Instant::now() < deadline);
                let mut completed = Vec::new();
                store.disk.poll_transport(&mut completed).unwrap();
                for event in completed {
                    if let TransportEvent::Stored { id, result } = event {
                        store.disk_stored(id, result);
                    }
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            // The record stays, reserved and in memory, for the next pass.
            assert_eq!(store.parked[&root.key].failures, 1);
            assert!(store.outbound[&root.key].write.is_none());
            assert!(store.memory.record(&root.key).is_some());
            let node = store.disk.index().node(&root.key).unwrap();
            assert!(!node.ready && node.pinned());
            assert_eq!(store.stats().store_failed, 0);

            if recovers {
                fs::create_dir_all(&directory).unwrap();
            }
            store.spill_parked(10, 0);
            poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
            if recovers {
                assert!(store.disk.index().node(&root.key).unwrap().ready);
                assert!(store.memory.record(&root.key).unwrap().durable);
                assert_eq!(store.stats().store_failed, 0);
            } else {
                // The second failure drops it from memory and the disk index.
                assert!(store.memory.record(&root.key).is_none());
                assert!(!store.disk.index().contains(&root.key));
                assert!(store.parked.is_empty());
                assert_eq!(store.stats().store_failed, 1);
                assert_eq!(store.stats().dropped[2], 1);
                fs::create_dir_all(&directory).unwrap();
            }
            drop(store);
            let _ = fs::remove_dir_all(&directory);
            fs::create_dir_all(&directory).unwrap();
        }
    }

    #[test]
    fn a_full_write_queue_keeps_the_coldest_parked_leaves() {
        let Some(directory) = test_directory("composite", "pooled-queue-full") else {
            return;
        };
        let mut store = composite(&directory, 16, 16, 1 << 30);
        store.cascade();
        let mut ids = Ids::new();
        let roots: Vec<_> = (1..=MAX_SPILLS as u32 + 2)
            .map(|salt| parked_root(&mut store, &mut ids, salt, 64))
            .collect();
        // Warm the most recent two so the coldest are the first ten.
        store.spill_parked(roots.len(), 0);
        let writing = store.parked.values().filter(|p| p.writing).count();
        assert_eq!(writing, MAX_SPILLS);
        assert!(
            roots
                .iter()
                .all(|root| store.memory.record(&root.key).is_some())
        );
        assert_eq!(store.parked.len(), roots.len());
        assert_eq!(store.stats().dropped, [0; 6]);
        assert!(store.stats().held[0] > 0);
        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| store.jobs.is_empty());
        assert_eq!(
            store.stats().stored + store.stats().parked_records as u64,
            roots.len() as u64
        );
        let committed = store.stats().stored;
        store.close().unwrap();
        assert_eq!(
            roots
                .iter()
                .filter(|r| store.disk.index().node(&r.key).is_some_and(|n| n.ready))
                .count() as u64,
            committed
        );
    }

    #[test]
    fn a_full_disk_holds_its_coldest_file_for_the_manager_and_drops_it_once_moved() {
        let Some(directory) = test_directory("composite", "pooled-disk-full") else {
            return;
        };
        let record_bytes = PAGE_RECORD + ROW_RECORD;
        // Room for one one-page record's file beside the header overhead.
        let limit = 2 * (record_bytes + 4096 * 3);
        let mut store = composite(&directory, 16, 16, limit);
        let (copier, _) = crate::testing::SimulatedGpu::shared(&geometry()).unwrap();
        store.memory = MemoryStore::new_flat(
            IDENTITY,
            SlabPool::new(PAGE_RECORD, 16, ROW_RECORD, 16, true).unwrap(),
            copier,
        )
        .unwrap();
        store.cascade();
        let announcer =
            crate::pool::Announcer::new(std::sync::Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer);
        let mut ids = Ids::new();
        let mut events = Vec::new();
        let mut roots = Vec::new();
        for salt in 1..=2u32 {
            roots.push(parked_root(&mut store, &mut ids, salt, 64));
            store.spill_parked(1, 0);
            poll_until(&mut store, &mut events, |store| store.outbound.is_empty());
        }
        store.set_spill(true);
        assert!(store.disk.used() * 2 > limit, "the disk is nearly full");

        // The next record needs file room: the coldest file is held for the
        // manager, and the new publication is skipped after one copy.
        let mut history = PrefixHistory::new(IDENTITY);
        let list: Vec<u32> = (0..64u32)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(9))
            .collect();
        history.append(&list, &[]).unwrap();
        let wanted = checkpoint(&mut ids, &history, 64, None, pages(0..1), 1, StateSlotId(9));
        let bytes_before = store.stats().bytes_out;
        let stored_before = store.stats().stored;
        assert_eq!(store.store(&wanted).unwrap(), StoreOutcome::Queued);
        let mut landed_events = Vec::new();
        store.poll(&mut landed_events).unwrap();
        assert!(matches!(
            landed_events.as_slice(),
            [StoreEvent::Stored { checkpoint, result: Ok(_), .. }] if *checkpoint == wanted.id
        ));
        // Follow the worker's retry rule through unrelated progress while the
        // full disk's retiring record remains held by the manager.
        for _ in 0..128 {
            for event in landed_events.drain(..) {
                if matches!(event, StoreEvent::Deferred { .. }) {
                    assert_eq!(store.store(&wanted).unwrap(), StoreOutcome::Queued);
                }
            }
            store.poll(&mut landed_events).unwrap();
        }
        assert_eq!(store.stats().stored, stored_before);
        assert_eq!(store.stats().bytes_out - bytes_before, record_bytes);
        assert!(store.stats().bytes_out <= (store.stats().stored + 1) * record_bytes);
        store.poll(&mut events).unwrap();
        let retiring = store.take_retiring();
        assert_eq!(retiring.len(), 1);
        assert_eq!(retiring[0].key, roots[0].key);
        assert_eq!(retiring[0].tier, PoolTier::Nvme);
        assert!(store.disk.index().contains(&roots[0].key));

        // When the manager holds it, the file and its memory copy go.
        store.retired(roots[0].key, PoolTier::Nvme, Retirement::Moved);
        store.poll(&mut events).unwrap();
        assert!(!store.disk.index().contains(&roots[0].key));
        poll_until(&mut store, &mut events, |store| {
            store.memory.record(&roots[0].key).is_none()
        });
        assert!(store.disk.index().contains(&roots[1].key));
        store.close().unwrap();
    }

    #[test]
    fn a_record_leaving_memory_for_its_file_is_announced_as_one_spill() {
        use crate::pool::{Announcer, Notice, Tier as Pool};
        let Some(directory) = test_directory("composite", "pooled-spill-notice") else {
            return;
        };
        let mut store = composite(&directory, 16, 4, 1 << 30);
        store.cascade();
        let announcer = Announcer::new(std::sync::Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer.clone());
        let mut ids = Ids::new();
        let root = parked_root(&mut store, &mut ids, 1, 640);
        let mut notices = Vec::new();
        let collect = |store: &mut CompositeStore,
                       notices: &mut Vec<Notice>,
                       done: &dyn Fn(&[Notice]) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(20);
            while !done(notices) {
                assert!(Instant::now() < deadline, "notices: {notices:?}");
                store.poll(&mut Vec::new()).unwrap();
                if let Some(batch) = announcer.take_batch(100) {
                    notices.extend(batch);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        // Hashed, the memory copy is announced; the file does not exist yet.
        collect(&mut store, &mut notices, &|notices| {
            notices
                .iter()
                .any(|n| matches!(n, Notice::Saved(e) if e.tier == Pool::Dram && e.key == root.key))
        });
        assert!(
            !notices
                .iter()
                .any(|n| matches!(n, Notice::Saved(e) if e.tier == Pool::Nvme))
        );
        // Memory needs the room: the file is written and announced...
        let mut other_history = PrefixHistory::new(IDENTITY);
        let list: Vec<u32> = (0..640u32)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(2))
            .collect();
        other_history.append(&list, &[]).unwrap();
        let other = checkpoint(
            &mut ids,
            &other_history,
            640,
            None,
            pages(0..10),
            10,
            StateSlotId(2),
        );
        assert_eq!(
            store.store(&other).unwrap(),
            StoreOutcome::Retry(StoreReason::Slabs)
        );
        collect(&mut store, &mut notices, &|notices| {
            notices
                .iter()
                .any(|n| matches!(n, Notice::Saved(e) if e.tier == Pool::Nvme && e.key == root.key))
        });
        // ...and when the memory copy gives way, the move is one notice.
        assert_eq!(store.store(&other).unwrap(), StoreOutcome::Queued);
        collect(&mut store, &mut notices, &|notices| {
            notices.iter().any(|n| matches!(n, Notice::Spilled { .. }))
        });
        let spills: Vec<_> = notices
            .iter()
            .filter_map(|n| match n {
                Notice::Spilled { to, from_tier } => Some((to.key, to.tier, *from_tier)),
                _ => None,
            })
            .collect();
        assert_eq!(spills, [(root.key, Pool::Nvme, Pool::Dram)]);
        assert!(!notices.iter().any(|n| matches!(
            n,
            Notice::Evicted { key, tier, .. } if *key == root.key && *tier == Pool::Dram
        )));
        store.close().unwrap();
    }

    #[test]
    fn a_publication_writes_through_in_local_mode() {
        let Some(directory) = test_directory("composite", "publish-local") else {
            return;
        };
        let mut store = composite(&directory, 12, 4, 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        // Local mode keeps write-through: the copy lands in memory and a file
        // is reserved and written at once.
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert!(store.outbound.contains_key(&root.key));
        assert!(store.parked.is_empty());
        store.close().unwrap();
    }
}
