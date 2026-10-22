//! Memory and disk stores composed with write-through and read-through.
//!
//! Checkpoints that fit available retention capacity are copied out of the GPU
//! into the memory store first. The composite then reserves a disk record and
//! writes the memory slabs through to its file; when the write completes, the
//! memory record becomes durable. Lookups run on the disk index, a superset of
//! the memory index that includes records whose writes are in flight: a disk
//! node is usable when its file is complete or when the memory store holds its
//! complete copy. A fetch uploads memory-resident records from their slabs and
//! first reads every other record of the chain from its file into freshly
//! allocated memory slabs, so a restored record becomes a durable resident.
//! Memory eviction takes only durable records, and a record unlinked from disk
//! loses its memory copy, so memory contents remain a subset of disk contents.
//!
//! The memory store is flat: its index holds no parent links, and the disk
//! index owns the tree. Records and restore chains that cannot fit available
//! memory stream through the disk transport's bounded staging pool without
//! evicting retained copies. Each tier owns independent transfer queues on the
//! existing rank contexts, so completions remain associated with their owning tier.
//!
//! Pins follow the bytes a job references. A fetch pins the memory copies it
//! uploads from in the memory index and the files it reads in the disk index,
//! so a record whose disk write fails while a memory fetch reads it can still
//! leave the disk index. A disk parent stays pinned while its child's memory
//! copy is in flight, so the child is reserved under it once its tail pages
//! have landed; when the parent's own write fails, those children fail too,
//! because their memory copies hold only their tail pages.

use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    Error, Result,
    cache::{CheckpointId, PrefixHistory, PrefixKey},
};

use super::{
    HostHit, INITIAL_STAGED_DISK_RATE, InboundTicket, JobId, PrefixStore, PublishedCheckpoint,
    StoreEvent, StoreOutcome, StoreReason, StoreStats, Tier,
    disk_store::{DiskStore, Staged, TransportEvent},
    geometry::invalid,
    index::Match,
    memory::MemoryStore,
    record::Header,
    transfer::CLOSE_TIMEOUT,
};

/// Weight of the newest observation in the disk rate estimates.
const RATE_WEIGHT: f64 = 0.2;

/// Exponentially weighted bytes per second of completed disk jobs in one direction.
#[derive(Clone, Copy)]
struct Rate(f64);

impl Rate {
    fn observe(&mut self, bytes: u64, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        if bytes == 0 || seconds <= 0.0 {
            return;
        }
        self.0 = (1.0 - RATE_WEIGHT) * self.0 + RATE_WEIGHT * (bytes as f64 / seconds);
    }
}

/// A published checkpoint from its memory copy-out to the end of its disk write.
struct Outbound {
    /// The disk parent, pinned in the disk index until the record is reserved under it.
    parent: Option<PrefixKey>,
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
    /// Scratch reused across polls.
    memory_events: Vec<StoreEvent>,
    transport_events: Vec<TransportEvent>,
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
            disk_outbound_rate: Rate(INITIAL_STAGED_DISK_RATE),
            disk_inbound_rate: Rate(INITIAL_STAGED_DISK_RATE),
            memory_events: Vec::new(),
            transport_events: Vec::new(),
        })
    }

    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }

    pub fn disk(&self) -> &DiskStore {
        &self.disk
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

    fn retains_chain(&self, chain: &Match) -> bool {
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
            }
        }
    }

    fn streamed_event(&mut self, event: TransportEvent, events: &mut Vec<StoreEvent>) {
        if let TransportEvent::Stored {
            id, result: Err(_), ..
        } = &event
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
                Tier::Memory
            } else {
                Tier::Disk
            },
            estimated_seconds: disk_bytes as f64 / self.disk_inbound_rate.0.max(1.0)
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
        let parent = outbound.parent.take();
        if let Some(parent) = &parent {
            let _ = self.disk.index_mut().unpin(parent);
        }
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
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
        let reserved = self
            .disk
            .reserve_record(key, parent, outbound.bucket, outbound.tail_pages);
        self.drop_evicted_copies();
        let (header, temporary, path) = match reserved {
            Ok(Some(reserved)) => reserved,
            Ok(None) => {
                self.remove_copy(key);
                self.stats.store_skipped += 1;
                events.push(StoreEvent::Deferred { checkpoint });
                return;
            }
            Err(error) => {
                self.remove_copy(key);
                self.stats.store_failed += 1;
                events.push(StoreEvent::Stored {
                    checkpoint,
                    key,
                    result: Err(error),
                });
                return;
            }
        };
        match self.start_write(key, header, temporary, path) {
            Ok(job) => {
                outbound.write = Some((job, Instant::now()));
                self.outbound.insert(key, outbound);
                // The GPU checkpoint is no longer read; the disk write reads the memory slabs.
                events.push(StoreEvent::Stored {
                    checkpoint,
                    key,
                    result: Ok(bytes),
                });
            }
            Err(error) => {
                let _ = self.disk.commit_record(key, Err(error.clone()));
                self.remove_copy(key);
                self.stats.store_failed += 1;
                events.push(StoreEvent::Stored {
                    checkpoint,
                    key,
                    result: Err(error),
                });
            }
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
                self.fail_children(key);
                let _ = self.disk.commit_record(key, Err(error));
                self.remove_copy(key);
                self.stats.store_failed += 1;
            }
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

impl PrefixStore for CompositeStore {
    fn lookup(&mut self, history: &PrefixHistory, resident: u64, before: u64) -> Option<HostHit> {
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
        if self.outbound.contains_key(&key) {
            self.stats.store_skipped += 1;
            return Ok(StoreOutcome::Retry(StoreReason::Inflight));
        }
        if let Some(outcome) = self.disk.store_status(checkpoint)? {
            self.stats.store_skipped += 1;
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
        let (pages, rows) = self.memory.reclaimable_slabs();
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
                StoreEvent::Deferred { checkpoint } => {
                    events.push(StoreEvent::Deferred { checkpoint })
                }
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
        polled
    }

    fn stats(&self) -> StoreStats {
        let memory = self.memory.stats();
        let disk = self.disk.stats();
        StoreStats {
            outbound_in_flight: self.outbound.len() + disk.outbound_in_flight,
            inbound_in_flight: self.inbound.len() + disk.inbound_in_flight,
            memory_records: memory.memory_records,
            memory_bytes: memory.memory_bytes,
            disk_records: disk.disk_records,
            disk_bytes: disk.disk_bytes,
            row_refills: disk.row_refills,
            progress: disk.progress + memory.progress + self.stats.progress,
            ..self.stats
        }
    }

    /// Finish the copy-outs, disk writes, and restores in flight within
    /// [`CLOSE_TIMEOUT`], then drain the disk and memory tiers. An error leaves
    /// ownership intact until their transfers can be drained.
    fn close(&mut self) -> Result<()> {
        let deadline = Instant::now() + CLOSE_TIMEOUT;
        let mut events = Vec::new();
        while Instant::now() < deadline {
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
        self.inbound.clear();
        self.jobs.clear();
        self.stale.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cache::{Fence, PageId, PrefixCheckpointCache, Publication, SnapshotId, StateSnapshot},
        offload::{
            CacheGeometry, INITIAL_MEMORY_RATE, Layout,
            disk::DiskThread,
            geometry::tests::geometry as rank,
            memory::MemoryRecord,
            record,
            slab::{PageSlab, RowSlab, SlabPool},
            transfer::Transfers,
        },
        scheduler::StateSlotId,
    };
    use std::{
        fs,
        path::Path,
        sync::{Arc, Mutex},
    };

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

    /// Mints checkpoint ids from a GPU prefix cache, whose constructor is the
    /// only source of them.
    struct Ids {
        cache: PrefixCheckpointCache,
        history: PrefixHistory,
        previous: Option<CheckpointId>,
        count: u32,
    }

    impl Ids {
        fn new() -> Self {
            Self {
                cache: PrefixCheckpointCache::new(IDENTITY, 4096, 64, 64).unwrap(),
                history: history(64 * 32),
                previous: None,
                count: 0,
            }
        }

        fn next(&mut self) -> CheckpointId {
            self.count += 1;
            let consumed = 64 * u64::from(self.count);
            let id = self
                .cache
                .publish(
                    &self.history,
                    &Publication {
                        parent: self.previous,
                        consumed_tokens: consumed,
                        tail_pages: vec![PageId(self.count)],
                        snapshot: StateSnapshot {
                            id: SnapshotId(self.count),
                            consumed_tokens: consumed,
                        },
                        ready_after: Fence(0),
                    },
                )
                .unwrap()
                .id();
            self.previous = Some(id);
            id
        }
    }

    fn checkpoint(
        ids: &mut Ids,
        history: &PrefixHistory,
        consumed: u64,
        parent: Option<u64>,
        chain_pages: Vec<PageId>,
        tail_pages: usize,
        slot: StateSlotId,
    ) -> PublishedCheckpoint {
        PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(consumed).unwrap(),
            parent: parent.map(|parent| history.key_at(parent).unwrap()),
            bucket: history.bucket_at(consumed).unwrap(),
            chain_pages,
            tail_pages,
            slot,
        }
    }

    fn pages(range: std::ops::Range<u32>) -> Vec<PageId> {
        range.map(PageId).collect()
    }

    /// A fresh directory for one test under `FLARE_OFFLOAD_TEST_DIR` or the
    /// system temporary directory, or `None` when it refuses direct I/O.
    fn test_directory(test: &str) -> Option<PathBuf> {
        let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = base.join(format!(
            "flare-offload-composite-{test}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let probe = directory.join(".probe");
        match crate::os::direct_file(&probe, true) {
            Ok(_) => {
                fs::remove_file(&probe).unwrap();
                Some(directory)
            }
            Err(error) if cfg!(unix) && error.raw_os_error() == Some(libc::EINVAL) => {
                eprintln!(
                    "skipping {test}: {} refuses direct I/O",
                    directory.display()
                );
                fs::remove_dir_all(&directory).unwrap();
                None
            }
            Err(error) => panic!("probe {}: {error}", probe.display()),
        }
    }

    #[cfg(feature = "runtime-package")]
    fn gpu_transport(context: usize, device: i32, geometry: &[CacheGeometry]) -> Staged {
        let mut pool = SlabPool::staging(&Layout::new(geometry).unwrap()).unwrap();
        pool.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.to_vec(), &pool).unwrap(),
        ));
        Staged::new(
            Some(pool),
            transfers,
            DiskThread::start("flare-offload-disk").unwrap(),
        )
    }

    fn pattern(bytes: &mut [u8], seed: u64) {
        for (index, byte) in bytes.iter_mut().enumerate() {
            let index = index as u64;
            *byte = (index.wrapping_mul(31) ^ seed.wrapping_mul(7919) ^ (index >> 8)) as u8;
        }
    }

    /// A composite over `pages` page slabs and `rows` row slabs of
    /// unregistered memory, transfer threads that accept nothing, and a disk
    /// store under `directory` with `limit` bytes.
    fn composite(directory: &Path, pages: usize, rows: usize, limit: u64) -> CompositeStore {
        let geometry = geometry();
        let transfers = Arc::new(Mutex::new(Transfers::stopped(&geometry).unwrap()));
        let pool = SlabPool::new(PAGE_RECORD, pages, ROW_RECORD, rows, true).unwrap();
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let directory_lock = record::DirectoryLock::acquire(directory).unwrap();
        let scanned = record::scan(&directory_lock, &IDENTITY, &FINGERPRINT, &geometry).unwrap();
        let layout = Layout::new(&geometry).unwrap();
        let disk_transfers = Arc::new(Mutex::new(Transfers::stopped(&geometry).unwrap()));
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
                DiskThread::start("flare-offload-disk").unwrap(),
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
                StoreEvent::Deferred { .. } => panic!("unexpected deferred event"),
            })
            .collect()
    }

    fn fetched(events: &[StoreEvent]) -> Vec<(PrefixKey, bool)> {
        events
            .iter()
            .map(|event| match event {
                StoreEvent::Fetched { key, result } => (*key, result.is_ok()),
                StoreEvent::Stored { .. } => panic!("unexpected store event"),
                StoreEvent::Deferred { .. } => panic!("unexpected deferred event"),
            })
            .collect()
    }

    #[test]
    fn oversized_chains_stream_without_evicting_retained_records() {
        let Some(directory) = test_directory("oversized-chain") else {
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
            assert_eq!((hit.key, hit.tier, hit.pages), (child.key, Tier::Disk, 20));
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
        let Some(directory) = test_directory("streamed-failure") else {
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
        let Some(directory) = test_directory("retention-pressure") else {
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
        let Some(directory) = test_directory("new") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let transfers = Arc::new(Mutex::new(Transfers::stopped(&geometry).unwrap()));
        let pool = || SlabPool::new(PAGE_RECORD, 4, ROW_RECORD, 2, true).unwrap();
        let staging = || Some(SlabPool::staging(&layout).unwrap());
        let disk = |identity: [u8; 32], staging: Option<SlabPool>, shared: bool| {
            let disk_transfers = if shared {
                transfers.clone()
            } else {
                Arc::new(Mutex::new(Transfers::stopped(&geometry).unwrap()))
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
        let Some(directory) = test_directory("cost-fallback") else {
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
        assert_eq!((warm.key, warm.tier), (root.key, Tier::Memory));
        assert!(super::super::worth_fetching(&warm, 20e6));
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn store_dedupes_pins_the_parent_and_reports_a_failed_memory_copy() {
        let Some(directory) = test_directory("store") else {
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
        let Some(directory) = test_directory("write-through") else {
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
        assert_eq!((hit.key, hit.tier, hit.pages), (root.key, Tier::Memory, 10));
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

        // Asynchronous reservation refusal releases the GPU pin for a later retry.
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
        let (_, big_bytes) = copy_landed(&mut small, &big, None, 2);
        small.memory_stored(big.id, big.key, Ok(big_bytes), &mut events);
        assert!(
            matches!(&events[..], [StoreEvent::Deferred { checkpoint }] if *checkpoint == big.id)
        );
        events.clear();
        assert!(!small.memory.index().contains(&big.key));
        assert!(!small.disk.index().contains(&big.key));
        let stats = small.stats();
        assert_eq!(
            (stats.store_skipped, stats.store_failed, stats.bytes_out),
            (1, 0, big_bytes)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (0, 0));
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
        assert_eq!(hit.tier, Tier::Memory);
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
        let Some(directory) = test_directory("write-failure") else {
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
        let Some(directory) = test_directory("read-through") else {
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
        assert_eq!((hit.key, hit.tier, hit.pages), (root.key, Tier::Disk, 10));
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
        assert!(store.disk_inbound_rate.0 != INITIAL_STAGED_DISK_RATE);

        // Resident again, the record is a memory hit whose upload is handed
        // to the memory store at once; without transfer threads that upload
        // fails before `fetch` returns, leaving only the composite's pin.
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Memory);
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
        assert_eq!((hit.key, hit.tier), (root.key, Tier::Disk));
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
            (7, 13, Tier::Memory)
        );
        store.memory.remove_record(&child.key).unwrap();
        let hit = store.lookup(&full, 500, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Disk);
        let expected = (10 * PAGE_RECORD + ROW_RECORD) as f64 / store.disk_inbound_rate.0
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
    fn a_fetch_attaches_to_a_record_another_fetch_is_reading() {
        let Some(directory) = test_directory("attach") else {
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
        let Some(directory) = test_directory("eviction") else {
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

    #[test]
    fn close_finishes_a_pending_disk_write() {
        let Some(directory) = test_directory("close") else {
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

    #[cfg(feature = "runtime-package")]
    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn composite_writes_a_child_published_while_its_parent_is_in_flight() {
        use crate::{
            cuda::Stream,
            offload::geometry::fingerprint,
            runtime_package::{LoadOptions, NativeModel},
            startup::StartupProgress,
        };
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let Some(directory) = test_directory("gpu-child") else {
            return;
        };
        let model = unsafe {
            NativeModel::load(
                crate::runtime_package::test_support::authorize(Path::new(&package)),
                &[0],
                &LoadOptions::default(),
                &StartupProgress::silent(),
            )
        }
        .unwrap();
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank = geometry[0].clone();
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let page_tokens = rank.page_tokens;
        let stream = Stream::open(context, device).unwrap();
        let mut staging = stream.host_buffer(rank.row.unit_bytes as usize).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            for (index, byte) in staging.bytes_mut()[..bytes].iter_mut().enumerate() {
                *byte = (index as u64).wrapping_mul(31).wrapping_add(seed) as u8;
            }
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in 1..=8u64 {
                write_unit(
                    geometry.unit_address(id),
                    geometry.unit_bytes as usize,
                    plane as u64 * 100 + id,
                );
            }
        }
        for slot in 1..=4u64 {
            write_unit(
                rank.row.unit_address(slot),
                rank.row.unit_bytes as usize,
                900 + slot,
            );
        }
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            16,
            layout.row_record_bytes,
            4,
            true,
        )
        .unwrap();
        pool.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let disk = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            fingerprint,
            1 << 40,
            geometry.clone(),
            Vec::new(),
            Box::new(gpu_transport(context, device, &geometry)),
        )
        .unwrap();
        let mut store = CompositeStore::new(memory, disk).unwrap();
        let mut ids = Ids::new();
        let drain = |store: &mut CompositeStore,
                     events: &mut Vec<StoreEvent>,
                     until: &dyn Fn(&CompositeStore) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(60);
            while !until(store) {
                assert!(
                    Instant::now() < deadline,
                    "composite did not settle: outbound {:?} jobs {:?} stats {:?} events {events:?}",
                    store.outbound.keys().collect::<Vec<_>>(),
                    store.jobs.keys().collect::<Vec<_>>(),
                    store.stats()
                );
                store.poll(events).unwrap();
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        // Two root and child pairs: the child of the first pair is published
        // before the root's memory copy lands, the child of the second after
        // the root's memory copy landed but while its disk write may still run.
        for (pair, wait_for_root_copy) in [(0u32, false), (1u32, true)] {
            let root_consumed = 3 * page_tokens;
            let child_consumed = 4 * page_tokens + 5;
            let mut history = PrefixHistory::new(IDENTITY);
            let tokens: Vec<u32> = (0..child_consumed as u32 + 8)
                .map(|t| t.wrapping_mul(2654435761).wrapping_add(pair))
                .collect();
            history.append(&tokens, &[]).unwrap();
            let base = 1 + pair * 4;
            let root_pages: Vec<PageId> = (base..base + 3).map(PageId).collect();
            let child_page = PageId(base + 3);
            let root = PublishedCheckpoint {
                id: ids.next(),
                key: history.key_at(root_consumed).unwrap(),
                parent: None,
                bucket: history.bucket_at(root_consumed).unwrap(),
                chain_pages: root_pages.clone(),
                tail_pages: 3,
                slot: StateSlotId(1 + pair * 2),
            };
            let child = PublishedCheckpoint {
                id: ids.next(),
                key: history.key_at(child_consumed).unwrap(),
                parent: Some(root.key),
                bucket: history.bucket_at(child_consumed).unwrap(),
                chain_pages: root_pages.iter().copied().chain([child_page]).collect(),
                tail_pages: 1,
                slot: StateSlotId(2 + pair * 2),
            };
            let mut events = Vec::new();
            assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
            if wait_for_root_copy {
                drain(&mut store, &mut events, &|store| {
                    store.stats().bytes_out > 0 && store.memory.record(&root.key).is_some()
                });
            }
            assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
            drain(&mut store, &mut events, &|store| store.outbound.is_empty());
            let stored: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    StoreEvent::Stored { key, result, .. } => Some((*key, result.is_ok())),
                    _ => None,
                })
                .collect();
            assert_eq!(stored.len(), 2, "pair {pair}: {events:?}");
            assert!(stored.iter().all(|(_, ok)| *ok), "pair {pair}: {events:?}");
            let stats = store.stats();
            assert_eq!(
                (stats.stored, stats.store_failed, stats.store_skipped),
                (2 * (u64::from(pair) + 1), 0, 0),
                "pair {pair}"
            );
            for key in [root.key, child.key] {
                let node = store.disk.index().node(&key).unwrap();
                assert!(
                    node.ready && !node.pinned(),
                    "pair {pair} disk node {key:?}"
                );
                assert_eq!(
                    fs::metadata(&node.location.path).unwrap().len(),
                    node.location.header.length_with_row()
                );
                assert!(
                    store.memory.record(&key).unwrap().durable,
                    "pair {pair} memory {key:?}"
                );
            }
            assert!(
                store.jobs.is_empty(),
                "pair {pair}: {:?}",
                store.jobs.keys().collect::<Vec<_>>()
            );
        }
        // A grandchild under a page-less child: the child's endpoint shares its
        // parent's last page, so its record holds only a row. After a restart the
        // chain restores through disk with the child contributing no bytes.
        let (pair, base) = (2u32, 9u32);
        let root_consumed = 3 * page_tokens;
        let child_consumed = 3 * page_tokens + 5;
        let grandchild_consumed = 4 * page_tokens + 7;
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..grandchild_consumed as u32 + 8)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(pair))
            .collect();
        history.append(&tokens, &[]).unwrap();
        let root_pages: Vec<PageId> = (base..base + 3).map(PageId).collect();
        let grandchild_page = PageId(base + 3);
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in base..=base + 3 {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    plane as u64 * 100 + u64::from(id),
                );
            }
        }
        for slot in 5..=6u64 {
            write_unit(
                rank.row.unit_address(slot),
                rank.row.unit_bytes as usize,
                900 + slot,
            );
        }
        let root = PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(root_consumed).unwrap(),
            parent: None,
            bucket: history.bucket_at(root_consumed).unwrap(),
            chain_pages: root_pages.clone(),
            tail_pages: 3,
            slot: StateSlotId(5),
        };
        let child = PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(child_consumed).unwrap(),
            parent: Some(root.key),
            bucket: history.bucket_at(child_consumed).unwrap(),
            chain_pages: root_pages.clone(),
            tail_pages: 0,
            slot: StateSlotId(6),
        };
        let grandchild = PublishedCheckpoint {
            id: ids.next(),
            key: history.key_at(grandchild_consumed).unwrap(),
            parent: Some(child.key),
            bucket: history.bucket_at(grandchild_consumed).unwrap(),
            chain_pages: root_pages
                .iter()
                .copied()
                .chain([grandchild_page])
                .collect(),
            tail_pages: 1,
            slot: StateSlotId(5),
        };
        let mut events = Vec::new();
        for checkpoint in [&root, &child, &grandchild] {
            assert_eq!(store.store(checkpoint).unwrap(), StoreOutcome::Queued);
            drain(&mut store, &mut events, &|store| store.outbound.is_empty());
        }
        assert_eq!(store.stats().stored, 7);
        store.close().unwrap();
        drop(store);

        // Restart: a fresh composite over the scanned directory and an empty memory pool.
        let directory_lock = record::DirectoryLock::acquire(&directory).unwrap();
        let scanned =
            crate::offload::record::scan(&directory_lock, &IDENTITY, &fingerprint, &geometry)
                .unwrap();
        assert_eq!(scanned.len(), 7);
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            16,
            layout.row_record_bytes,
            4,
            true,
        )
        .unwrap();
        pool.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let disk = DiskStore::new(
            directory_lock,
            IDENTITY,
            fingerprint,
            1 << 40,
            geometry.clone(),
            scanned,
            Box::new(gpu_transport(context, device, &geometry)),
        )
        .unwrap();
        let mut store = CompositeStore::new(memory, disk).unwrap();
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!(
            (hit.key, hit.pages, hit.tier),
            (grandchild.key, 4, Tier::Disk)
        );
        let targets: Vec<PageId> = (base + 10..base + 14).map(PageId).collect();
        let ticket = InboundTicket {
            key: grandchild.key,
            pages: targets.clone(),
            slot: StateSlotId(7),
        };
        store.fetch(hit, ticket).unwrap();
        let mut events = Vec::new();
        drain(&mut store, &mut events, &|store| store.inbound.is_empty());
        assert!(
            matches!(
                events.as_slice(),
                [StoreEvent::Fetched { result: Ok(_), .. }]
            ),
            "{events:?}"
        );
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        let pattern = |seed: u64, bytes: usize| -> Vec<u8> {
            (0..bytes)
                .map(|index| (index as u64).wrapping_mul(31).wrapping_add(seed) as u8)
                .collect()
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for (source, target) in (base..=base + 3).zip(&targets) {
                let bytes = geometry.unit_bytes as usize;
                assert!(
                    read_unit(geometry.unit_address(u64::from(target.0)), bytes)
                        == pattern(plane as u64 * 100 + u64::from(source), bytes),
                    "plane {} page {} differs from page {source}",
                    geometry.name,
                    target.0
                );
            }
        }
        let bytes = rank.row.unit_bytes as usize;
        assert!(read_unit(rank.row.unit_address(7), bytes) == pattern(905, bytes));
        assert_eq!(store.stats().fetched, 1);
        assert!(store.memory.record(&child.key).is_some());
        store.close().unwrap();
        drop(store);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn composite_round_trips_through_memory_and_disk_on_a_loaded_package() {
        use crate::{
            cuda::Stream,
            offload::geometry::fingerprint,
            runtime_package::{LoadOptions, NativeModel},
            startup::StartupProgress,
        };
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let Some(directory) = test_directory("gpu") else {
            return;
        };
        let model = unsafe {
            NativeModel::load(
                crate::runtime_package::test_support::authorize(Path::new(&package)),
                &[0],
                &LoadOptions::default(),
                &StartupProgress::silent(),
            )
        }
        .unwrap();
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank = geometry[0].clone();
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let page_tokens = rank.page_tokens;
        assert!(rank.page_capacity >= 16 && rank.row_capacity >= 4);
        let capacity = rank.page_capacity as u32;
        let sources = pages_of([1, 7, capacity / 2]);
        let targets = pages_of([2, 9, capacity / 2 + 1]);
        let (source_slot, target_slot) = (StateSlotId(1), StateSlotId(2));
        fn pages_of(ids: [u32; 3]) -> Vec<PageId> {
            ids.into_iter().map(PageId).collect()
        }

        fn fill(bytes: &mut [u8], seed: u64) {
            for (index, byte) in bytes.iter_mut().enumerate() {
                let index = index as u64;
                *byte = (index.wrapping_mul(31) ^ seed.wrapping_mul(7919) ^ (index >> 8)) as u8;
            }
        }
        let seed = |plane: usize, unit: u32| (plane as u64 + 1) * 1_000_003 + u64::from(unit);
        let row_plane = rank.planes.len();
        let stream = Stream::open(context, device).unwrap();
        let largest = rank
            .planes
            .iter()
            .chain([&rank.row])
            .map(|plane| plane.unit_bytes as usize)
            .max()
            .unwrap();
        let mut staging = stream.host_buffer(largest).unwrap();
        let mut write_unit = |address: u64, bytes: usize, seed: u64| {
            fill(&mut staging.bytes_mut()[..bytes], seed);
            unsafe { stream.upload(address as usize, &staging, 0, bytes) }.unwrap();
            stream.synchronize().unwrap();
        };
        let write_pages = |write_unit: &mut dyn FnMut(u64, usize, u64),
                           ids: &[PageId],
                           pattern: &dyn Fn(usize, u32) -> u64| {
            for (plane, geometry) in rank.planes.iter().enumerate() {
                for id in ids {
                    write_unit(
                        geometry.unit_address(u64::from(id.0)),
                        geometry.unit_bytes as usize,
                        pattern(plane, id.0),
                    );
                }
            }
        };
        let write_row =
            |write_unit: &mut dyn FnMut(u64, usize, u64), slot: StateSlotId, seed: u64| {
                write_unit(
                    rank.row.unit_address(u64::from(slot.0)),
                    rank.row.unit_bytes as usize,
                    seed,
                );
            };
        write_pages(&mut write_unit, &sources, &seed);
        write_row(&mut write_unit, source_slot, seed(row_plane, source_slot.0));
        write_pages(&mut write_unit, &targets, &|_, _| 0xDD);
        write_row(&mut write_unit, target_slot, 0xDD);

        // Seven page slabs and two row slabs hold two three-page records with
        // one page to spare; a third record evicts the least recently used.
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            7,
            layout.row_record_bytes,
            2,
            true,
        )
        .unwrap();
        pool.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let memory = MemoryStore::new_flat(IDENTITY, pool, transfers.clone()).unwrap();
        let disk = DiskStore::new(
            record::DirectoryLock::acquire(&directory).unwrap(),
            IDENTITY,
            fingerprint,
            1 << 40,
            geometry.clone(),
            Vec::new(),
            Box::new(gpu_transport(context, device, &geometry)),
        )
        .unwrap();
        let mut store = CompositeStore::new(memory, disk).unwrap();

        let consumed = 3 * page_tokens + 5;
        let base: Vec<u32> = (0..consumed as u32 + 40)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        let histories: Vec<PrefixHistory> = (0..4u32)
            .map(|salt| {
                let mut history = PrefixHistory::new(IDENTITY);
                let tokens: Vec<u32> = base.iter().map(|t| t.wrapping_add(salt)).collect();
                history.append(&tokens, &[]).unwrap();
                history
            })
            .collect();
        let mut ids = Ids::new();
        let checkpoints: Vec<PublishedCheckpoint> = histories
            .iter()
            .enumerate()
            .map(|(index, history)| PublishedCheckpoint {
                id: ids.next(),
                key: history.key_at(consumed).unwrap(),
                parent: None,
                bucket: history.bucket_at(consumed).unwrap(),
                chain_pages: if index == 0 {
                    sources.clone()
                } else {
                    targets.clone()
                },
                tail_pages: 3,
                slot: source_slot,
            })
            .collect();
        let gpu_bytes = 3 * rank.page_bytes() + rank.row.unit_bytes;
        let record_bytes = layout.record_bytes(3);

        fn wait_event(store: &mut CompositeStore) -> StoreEvent {
            let deadline = Instant::now() + Duration::from_secs(120);
            let mut events = Vec::new();
            while events.is_empty() {
                assert!(Instant::now() < deadline, "no store event within 120 s");
                store.poll(&mut events).unwrap();
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(events.len(), 1, "{events:?}");
            events.pop().unwrap()
        }
        fn wait_until(store: &mut CompositeStore, mut done: impl FnMut(&CompositeStore) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(120);
            let mut events = Vec::new();
            while !done(store) {
                assert!(
                    Instant::now() < deadline,
                    "condition not reached within 120 s"
                );
                store.poll(&mut events).unwrap();
                assert!(events.is_empty(), "{events:?}");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let store_and_land = |store: &mut CompositeStore, checkpoint: &PublishedCheckpoint| {
            assert_eq!(store.store(checkpoint).unwrap(), StoreOutcome::Queued);
            match wait_event(store) {
                StoreEvent::Stored {
                    checkpoint: id,
                    key,
                    result,
                } => {
                    assert_eq!((id, key), (checkpoint.id, checkpoint.key));
                    assert_eq!(result.unwrap(), gpu_bytes);
                }
                other => panic!("unexpected event {other:?}"),
            }
            let key = checkpoint.key;
            wait_until(store, |store| !store.outbound.contains_key(&key));
            let node = store.disk.index().node(&key).unwrap();
            assert!(node.ready && !node.pinned());
            assert_eq!(
                fs::metadata(&node.location.path).unwrap().len(),
                node.location.header.length_with_row()
            );
            assert!(store.memory.record(&key).unwrap().durable);
        };

        // The first checkpoint lands in memory, is written through, and is
        // served from memory.
        let first = &checkpoints[0];
        store_and_land(&mut store, first);
        let stats = store.stats();
        assert_eq!(
            (
                stats.stored,
                stats.bytes_out,
                stats.memory_records,
                stats.disk_records
            ),
            (1, gpu_bytes, 1, 1)
        );
        assert_eq!(stats.memory_bytes, record_bytes);
        write_pages(&mut write_unit, &sources, &|plane, _| 0xEE + plane as u64);
        write_row(&mut write_unit, source_slot, 0xEE);
        let hit = store.lookup(&histories[0], 0, u64::MAX).unwrap();
        assert_eq!(
            (hit.key, hit.pages, hit.tier, hit.bytes),
            (first.key, 3, Tier::Memory, record_bytes)
        );
        let ticket = || InboundTicket {
            key: first.key,
            pages: targets.clone(),
            slot: target_slot,
        };
        store.fetch(hit, ticket()).unwrap();
        assert_eq!(store.stats().inbound_in_flight, 1);
        match wait_event(&mut store) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, first.key);
                assert_eq!(result.unwrap(), gpu_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        let readback = stream.host_buffer(largest).unwrap();
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&readback, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            readback.bytes()[..bytes].to_vec()
        };
        let check = |read_unit: &dyn Fn(u64, usize) -> Vec<u8>| {
            for (plane, geometry) in rank.planes.iter().enumerate() {
                for (source, target) in sources.iter().zip(&targets) {
                    let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                    fill(&mut wanted, seed(plane, source.0));
                    let actual = read_unit(
                        geometry.unit_address(u64::from(target.0)),
                        geometry.unit_bytes as usize,
                    );
                    assert!(
                        actual == wanted,
                        "plane {} ({plane}) page {} differs from page {}",
                        geometry.name,
                        target.0,
                        source.0
                    );
                }
            }
            let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
            fill(&mut wanted, seed(row_plane, source_slot.0));
            let actual = read_unit(
                rank.row.unit_address(u64::from(target_slot.0)),
                rank.row.unit_bytes as usize,
            );
            assert!(
                actual == wanted,
                "row {} differs from row {}",
                target_slot.0,
                source_slot.0
            );
        };
        check(&read_unit);
        let stats = store.stats();
        assert_eq!(
            (
                stats.fetched,
                stats.hits,
                stats.reused_tokens,
                stats.bytes_in
            ),
            (1, 1, consumed, gpu_bytes)
        );

        // Two more records fill the memory pool; the third evicts the first,
        // which stays on disk.
        store_and_land(&mut store, &checkpoints[1]);
        assert!(store.memory.index().contains(&first.key));
        store_and_land(&mut store, &checkpoints[2]);
        assert!(!store.memory.index().contains(&first.key));
        assert!(store.disk.index().node(&first.key).unwrap().ready);
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.memory_records, stats.disk_records),
            (3, 2, 3)
        );

        // The first record now comes from its file through memory slabs, and
        // is resident again afterwards.
        write_pages(&mut write_unit, &targets, &|_, _| 0xCC);
        write_row(&mut write_unit, target_slot, 0xCC);
        let hit = store.lookup(&histories[0], 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.tier), (first.key, Tier::Disk));
        assert!(hit.estimated_seconds > record_bytes as f64 / INITIAL_MEMORY_RATE);
        store.fetch(hit, ticket()).unwrap();
        match wait_event(&mut store) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, first.key);
                assert_eq!(result.unwrap(), gpu_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        check(&read_unit);
        let node = store.memory.index().node(&first.key).unwrap();
        assert!(node.ready && !node.pinned() && node.location.durable);
        assert_eq!(node.location.pages.len(), 3);
        assert!(!store.memory.index().contains(&checkpoints[1].key));
        assert!(store.memory.index().nodes().all(|node| !node.pinned()));
        assert!(store.disk.index().nodes().all(|node| !node.pinned()));
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.fetch_failed, stats.bytes_in),
            (2, 0, 2 * gpu_bytes)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (2, 3));

        // A record whose file cannot be created is dropped from memory and counted.
        let fourth = &checkpoints[3];
        let blocked = directory.join(record::temporary_name(&fourth.key));
        fs::create_dir(&blocked).unwrap();
        assert_eq!(store.store(fourth).unwrap(), StoreOutcome::Queued);
        match wait_event(&mut store) {
            StoreEvent::Stored { key, result, .. } => {
                assert_eq!(key, fourth.key);
                assert_eq!(result.unwrap(), gpu_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        wait_until(&mut store, |store| store.outbound.is_empty());
        fs::remove_dir(&blocked).unwrap();
        assert!(!store.memory.index().contains(&fourth.key));
        assert!(!store.disk.index().contains(&fourth.key));
        assert!(store.memory.index().contains(&first.key));
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.store_failed, stats.store_skipped),
            (3, 1, 0)
        );
        assert_eq!((stats.memory_records, stats.disk_records), (1, 3));
        assert_eq!(
            record::scan(
                &store.disk.directory_lock,
                &IDENTITY,
                &fingerprint,
                &geometry
            )
            .unwrap()
            .len(),
            3
        );
        store.close().unwrap();
        drop(store);
        // Retention is smaller than a record and staging holds one record's
        // pages. Exercise simultaneous memory copy-out, streamed copy-out,
        // and streamed restore, then reopen the files with an empty RAM tier.
        let open_small = || {
            let mut pool = SlabPool::new(
                layout.page_record_bytes,
                1,
                layout.row_record_bytes,
                1,
                true,
            )
            .unwrap();
            pool.register(context).unwrap();
            let transfers = Arc::new(Mutex::new(
                Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
            ));
            let memory = MemoryStore::new_flat(IDENTITY, pool, transfers).unwrap();
            let mut staging_pool = SlabPool::new(
                layout.page_record_bytes,
                3,
                layout.row_record_bytes,
                2,
                true,
            )
            .unwrap();
            staging_pool.register(context).unwrap();
            let transfers = Arc::new(Mutex::new(
                Transfers::start(&[(context, device)], geometry.clone(), &staging_pool).unwrap(),
            ));
            let lock = record::DirectoryLock::acquire(&directory).unwrap();
            let scanned = record::scan(&lock, &IDENTITY, &fingerprint, &geometry).unwrap();
            let disk = DiskStore::new(
                lock,
                IDENTITY,
                fingerprint,
                1 << 40,
                geometry.clone(),
                scanned,
                Box::new(Staged::new(
                    Some(staging_pool),
                    transfers,
                    DiskThread::start("flare-offload-disk").unwrap(),
                )),
            )
            .unwrap();
            CompositeStore::new(memory, disk).unwrap()
        };
        let mut store = open_small();
        write_pages(&mut write_unit, &sources, &seed);
        write_row(&mut write_unit, source_slot, seed(row_plane, source_slot.0));
        let mut fresh = PrefixHistory::new(IDENTITY);
        fresh
            .append(
                &base
                    .iter()
                    .map(|token| token.wrapping_add(77))
                    .collect::<Vec<_>>(),
                &[],
            )
            .unwrap();
        let short = checkpoint(
            &mut ids,
            &fresh,
            page_tokens + 5,
            None,
            sources[..1].to_vec(),
            1,
            source_slot,
        );
        let large = checkpoint(
            &mut ids,
            &fresh,
            consumed,
            None,
            sources.clone(),
            3,
            source_slot,
        );
        assert_eq!(store.store(&short).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.store(&large).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.outbound.len(), 1);
        assert_eq!(store.disk.stats().outbound_in_flight, 1);
        assert_eq!(store.memory.pool().free_pages(), 0);
        let hit = store.lookup(&histories[0], 0, u64::MAX).unwrap();
        assert!(!store.retains_chain(&hit.chain));
        store.fetch(hit, ticket()).unwrap();
        let mut events = Vec::new();
        poll_until(&mut store, &mut events, |store| {
            let stats = store.stats();
            stats.outbound_in_flight == 0 && stats.inbound_in_flight == 0
        });
        assert_eq!(events.len(), 3, "{events:?}");
        let mut stored_keys = Vec::new();
        for event in events {
            match event {
                StoreEvent::Stored { key, result, .. } => {
                    result.unwrap();
                    stored_keys.push(key);
                }
                StoreEvent::Fetched { key, result } => {
                    assert_eq!(key, first.key);
                    assert_eq!(result.unwrap(), record_bytes);
                }
                StoreEvent::Deferred { .. } => panic!("unexpected deferred event"),
            }
        }
        assert!(stored_keys.contains(&short.key) && stored_keys.contains(&large.key));
        check(&read_unit);
        assert!(store.memory.record(&short.key).unwrap().durable);
        assert!(store.memory.record(&large.key).is_none());
        assert!(store.memory.index().nodes().all(|node| !node.pinned()));
        assert!(store.disk.index().nodes().all(|node| !node.pinned()));
        assert_eq!(
            (
                store.stats().stored,
                store.stats().fetched,
                store.stats().store_failed,
                store.stats().fetch_failed
            ),
            (2, 1, 0, 0)
        );
        assert_eq!(store.stats().inbound_rate, store.disk.stats().inbound_rate);
        store.close().unwrap();
        drop(store);

        let mut store = open_small();
        write_pages(&mut write_unit, &targets, &|_, _| 0xA5);
        write_row(&mut write_unit, target_slot, 0xA5);
        let hit = store.lookup(&fresh, 0, u64::MAX).unwrap();
        assert_eq!((hit.key, hit.pages, hit.tier), (large.key, 3, Tier::Disk));
        store
            .fetch(
                hit,
                InboundTicket {
                    key: large.key,
                    pages: targets.clone(),
                    slot: target_slot,
                },
            )
            .unwrap();
        match wait_event(&mut store) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, large.key);
                assert_eq!(result.unwrap(), record_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        check(&read_unit);
        assert_eq!(store.stats().memory_records, 0);
        assert!(store.disk.index().nodes().all(|node| !node.pinned()));
        store.close().unwrap();
        drop(store);
        drop(readback);
        drop(staging);
        drop(stream);
        drop(model);
        fs::remove_dir_all(&directory).unwrap();
    }
}
