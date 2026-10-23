//! Records retained in registered host memory.
//!
//! A record is the page slabs and row slab of one checkpoint inside a retaining
//! [`SlabPool`]. `store` allocates slabs, evicting under the tree policy of the
//! [`Index`] when the pool is full, and copies the published GPU checkpoint out
//! through the shared [`Transfers`] threads in windows of at most
//! [`WINDOW_PAGES`] pages; the record stays resident until evicted. `fetch`
//! copies a looked-up chain of resident records back into the ids a ticket
//! names. At most [`IN_FLIGHT_WINDOWS`] windows of one record are submitted at
//! a time; the rest wait in the store and follow as replies arrive.
//!
//! A checkpoint whose parent's copy-out is still in flight is held back until
//! that copy-out finishes, then stored as the parent's child, or as a root
//! record holding its whole chain when the parent's copy-out failed. A
//! held-back checkpoint that cannot start fails with an event and releases
//! the checkpoints held back behind it the same way. Index edges therefore
//! only ever point at ready records, so a failed copy-out can always remove
//! its node, and a parent is pinned while its child's slabs are allocated so
//! eviction cannot remove the record the child extends.
//!
//! Restores are submitted before copy-outs whenever the queue has room, and
//! `close` waits up to [`CLOSE_TIMEOUT`] for transfers in progress before
//! stopping the threads.
//!
//! Inside a composite store the memory index holds no tree: every node is a
//! root whose slabs hold only the checkpoint's tail pages, the disk index
//! supplies chains, and `fetch_chain` reads page counts from the slabs. Such a
//! store answers no lookups of its own.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

use crate::{
    Error, Result,
    cache::{CheckpointId, PageId, PrefixHistory, PrefixKey},
    scheduler::StateSlotId,
};

use super::{
    HostHit, INITIAL_MEMORY_RATE, InboundTicket, JobId, Layout, PrefixStore, PublishedCheckpoint,
    StoreEvent, StoreOutcome, StoreReason, StoreStats, Tier,
    geometry::invalid,
    index::{Eviction, Index, Match},
    slab::{PageSlab, RowSlab, SlabPool, WINDOW_PAGES},
    transfer::{CLOSE_TIMEOUT, Direction, Job, Reply, Transfers},
};

/// Windows of one record submitted to the transfer threads at a time.
pub const IN_FLIGHT_WINDOWS: usize = 2;

/// Weight of the newest observation in the transfer rate estimates.
const RATE_WEIGHT: f64 = 0.2;

/// Slabs holding one record. `durable` is set by the composite once the disk
/// copy is complete; a standalone memory store treats every record as durable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRecord {
    pub pages: Vec<PageSlab>,
    pub row: Option<RowSlab>,
    pub durable: bool,
}

/// Windows of one record: those not yet submitted and the count in flight.
/// After a failed window the remaining windows are dropped, so the record
/// completes once its submitted windows have replied.
#[derive(Debug)]
struct Pending {
    queued: VecDeque<Job>,
    in_flight: usize,
    bytes: u64,
    error: Option<Error>,
}

impl Pending {
    fn new(queued: VecDeque<Job>) -> Self {
        Self {
            queued,
            in_flight: 0,
            bytes: 0,
            error: None,
        }
    }

    /// Submit queued windows until [`IN_FLIGHT_WINDOWS`] are in flight. A
    /// refused window stays at the front of the queue and its error is
    /// returned; the windows accepted before it count as in flight.
    fn pump(&mut self, submit: &mut impl FnMut(Job) -> Result<()>) -> Option<Error> {
        while self.error.is_none() && self.in_flight < IN_FLIGHT_WINDOWS {
            let Some(job) = self.queued.front() else {
                break;
            };
            if let Err(error) = submit(job.clone()) {
                return Some(error);
            }
            self.queued.pop_front();
            self.in_flight += 1;
        }
        None
    }

    fn absorb(&mut self, reply: Reply) {
        self.in_flight = self.in_flight.saturating_sub(1);
        self.bytes += reply.bytes;
        if let Err(error) = reply.result {
            self.fail(error);
        }
    }

    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
        self.queued.clear();
    }

    fn done(&self) -> bool {
        self.in_flight == 0 && self.queued.is_empty()
    }
}

#[derive(Debug)]
struct Outbound {
    checkpoint: CheckpointId,
    key: PrefixKey,
    pending: Pending,
}

#[derive(Debug)]
struct Inbound {
    key: PrefixKey,
    /// Chain nodes pinned for the transfer, ascending.
    nodes: Vec<PrefixKey>,
    /// Tokens the restore saves over recomputation.
    tokens: u64,
    pending: Pending,
}

/// Exponentially weighted bytes per second of completed windows.
#[derive(Clone, Copy, Debug)]
struct Rate(f64);

impl Rate {
    fn observe(&mut self, bytes: u64, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        if bytes == 0 || seconds <= 0.0 {
            return;
        }
        let observed = bytes as f64 / seconds;
        self.0 = (1.0 - RATE_WEIGHT) * self.0 + RATE_WEIGHT * observed;
    }
}

/// The store's bookkeeping: the index, the pool, the per-record transfer
/// state, and the counters. It submits windows through a caller-supplied
/// function, so it is exercised without transfer threads.
struct Core {
    index: Index<MemoryRecord>,
    pool: SlabPool,
    layout: Layout,
    identity: [u8; 32],
    /// Every node is a root; the composite's disk index owns the tree.
    flat: bool,
    outbound: BTreeMap<JobId, Outbound>,
    inbound: BTreeMap<JobId, Inbound>,
    /// Publications waiting for the copy-out of the named parent key.
    deferred: Vec<(PrefixKey, PublishedCheckpoint)>,
    stats: StoreStats,
    outbound_rate: Rate,
    inbound_rate: Rate,
    next_job: u64,
}

impl Core {
    fn new(identity: [u8; 32], pool: SlabPool, layout: Layout, flat: bool) -> Self {
        Self {
            index: Index::new(layout.page_tokens),
            pool,
            layout,
            identity,
            flat,
            outbound: BTreeMap::new(),
            inbound: BTreeMap::new(),
            deferred: Vec::new(),
            stats: StoreStats::default(),
            outbound_rate: Rate(INITIAL_MEMORY_RATE),
            inbound_rate: Rate(INITIAL_MEMORY_RATE),
            next_job: 1,
        }
    }

    fn next_job(&mut self) -> JobId {
        let id = JobId(self.next_job);
        self.next_job += 1;
        id
    }

    /// Windows accepted by the transfer threads whose replies are outstanding.
    fn in_flight_jobs(&self) -> usize {
        self.outbound
            .values()
            .map(|outbound| outbound.pending.in_flight)
            .chain(
                self.inbound
                    .values()
                    .map(|inbound| inbound.pending.in_flight),
            )
            .sum()
    }

    fn deferred_key(&self, key: &PrefixKey) -> bool {
        self.deferred
            .iter()
            .any(|(_, checkpoint)| checkpoint.key == *key)
    }

    /// The key is indexed, or waits for its parent, so its bytes exist or will.
    fn known(&self, key: &PrefixKey) -> bool {
        self.index.contains(key) || self.deferred_key(key)
    }

    /// The key's copy-out has been accepted and has not completed.
    fn copying_out(&self, key: &PrefixKey) -> bool {
        self.index.node(key).is_some_and(|node| !node.ready) || self.deferred_key(key)
    }

    /// The record shape for a standalone store: a child of a ready parent, or
    /// a root holding the whole chain.
    fn shape<'a>(&self, checkpoint: &'a PublishedCheckpoint) -> (Option<PrefixKey>, &'a [PageId]) {
        match checkpoint
            .parent
            .filter(|parent| self.index.node(parent).is_some_and(|node| node.ready))
        {
            Some(parent) => (Some(parent), checkpoint.tail()),
            None => (None, &checkpoint.chain_pages),
        }
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<bool> {
        if checkpoint.tail_pages > checkpoint.chain_pages.len() {
            self.stats.store_failed += 1;
            return Err(invalid(format!(
                "published checkpoint claims {} tail pages of a {}-page chain",
                checkpoint.tail_pages,
                checkpoint.chain_pages.len()
            )));
        }
        if self.known(&checkpoint.key) {
            self.stats.store_skipped += 1;
            return Ok(false);
        }
        if !self.flat
            && let Some(parent) = checkpoint.parent
            && self.copying_out(&parent)
        {
            self.deferred.push((parent, checkpoint.clone()));
            return Ok(true);
        }
        let (parent, tail) = self.shape(checkpoint);
        self.store_record(checkpoint, parent, tail, true)
    }

    /// Start a copy-out, counting a skip or a failure to start.
    fn store_record(
        &mut self,
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        tail: &[PageId],
        durable: bool,
    ) -> Result<bool> {
        let result = self.start_record(checkpoint, parent, tail, durable);
        match &result {
            Ok(true) => {}
            Ok(false) => self.stats.store_skipped += 1,
            Err(_) => self.stats.store_failed += 1,
        }
        result
    }

    /// Allocate and index the record and queue its windows. `Ok(false)` when
    /// the key is known or no slabs could be freed.
    fn start_record(
        &mut self,
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        tail: &[PageId],
        durable: bool,
    ) -> Result<bool> {
        if self.known(&checkpoint.key) {
            return Ok(false);
        }
        let parent = if self.flat { None } else { parent };
        // Eviction during allocation must not remove the record being extended.
        if let Some(parent) = &parent
            && self.index.pin(parent).is_err()
        {
            return Err(invalid("host record parent is not indexed"));
        }
        let allocated = self.allocate(
            checkpoint.key,
            parent,
            checkpoint.bucket,
            tail.len(),
            true,
            durable,
        );
        if let Some(parent) = &parent {
            let _ = self.index.unpin(parent);
        }
        let Some(record) = allocated? else {
            return Ok(false);
        };
        let Some(row) = record.row else {
            return Err(invalid("host record was allocated without a row slab"));
        };
        let id = self.next_job();
        let pages = tail
            .iter()
            .zip(&record.pages)
            .map(|(page, slab)| (*page, self.pool.page_address(*slab)));
        let row = Some((checkpoint.slot, self.pool.row_address(row)));
        let queued = windows(id, Direction::Out, pages, row);
        self.outbound.insert(
            id,
            Outbound {
                checkpoint: checkpoint.id,
                key: checkpoint.key,
                pending: Pending::new(queued),
            },
        );
        Ok(true)
    }

    /// Allocate slabs for a record, evicting durable unpinned records until
    /// they fit, and index it unready and pinned. `Ok(None)` when the pool
    /// cannot hold the record or nothing evictable remains.
    fn allocate(
        &mut self,
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: usize,
        row: bool,
        durable: bool,
    ) -> Result<Option<MemoryRecord>> {
        if tail_pages > self.pool.page_count() || (row && self.pool.row_count() == 0) {
            return Ok(None);
        }
        while self.pool.free_pages() < tail_pages || (row && self.pool.free_rows() == 0) {
            let need_row =
                row && self.pool.free_rows() == 0 && self.pool.free_pages() >= tail_pages;
            match self.index.evict(need_row, |node| node.location.durable) {
                Some(Eviction::Leaves(nodes)) => {
                    for node in &nodes {
                        self.release(&node.location);
                    }
                }
                Some(Eviction::Compacted(compacted)) => {
                    let slab = self
                        .index
                        .node_mut(&compacted)
                        .and_then(|node| node.location.row.take());
                    if let Some(slab) = slab {
                        self.pool.release_row(slab);
                    }
                }
                None => return Ok(None),
            }
        }
        let mut record = MemoryRecord {
            pages: Vec::with_capacity(tail_pages),
            row: None,
            durable,
        };
        for _ in 0..tail_pages {
            match self.pool.allocate_page() {
                Some(slab) => record.pages.push(slab),
                None => {
                    self.release(&record);
                    return Ok(None);
                }
            }
        }
        if row {
            match self.pool.allocate_row() {
                Some(slab) => record.row = Some(slab),
                None => {
                    self.release(&record);
                    return Ok(None);
                }
            }
        }
        // The index requires a root's tail to cover its whole chain; a flat
        // store's roots hold only their checkpoint's tail, so their count is
        // the chain's and `fetch_chain` reads page counts from the slabs.
        let index_pages = if self.flat {
            (key.consumed_tokens / self.layout.page_tokens) as usize
        } else {
            tail_pages
        };
        if let Err(error) =
            self.index
                .insert(key, parent, bucket, index_pages, row, false, record.clone())
        {
            self.release(&record);
            return Err(error);
        }
        self.index.pin(&key)?;
        Ok(Some(record))
    }

    fn release(&mut self, record: &MemoryRecord) {
        for slab in &record.pages {
            self.pool.release_page(*slab);
        }
        if let Some(slab) = record.row {
            self.pool.release_row(slab);
        }
    }

    fn allocate_record(
        &mut self,
        key: PrefixKey,
        bucket: [u8; 16],
        tail_pages: usize,
        row: bool,
    ) -> Result<Option<MemoryRecord>> {
        self.allocate(key, None, bucket, tail_pages, row, true)
    }

    fn complete_record(&mut self, key: &PrefixKey, success: bool) -> Result<()> {
        self.index.unpin(key)?;
        if success {
            if let Some(node) = self.index.node_mut(key) {
                node.ready = true;
            }
            return Ok(());
        }
        for node in self.index.remove(key)? {
            self.release(&node.location);
        }
        Ok(())
    }

    fn mark_durable(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .index
            .node_mut(key)
            .ok_or_else(|| invalid("cannot mark an unindexed host record durable"))?;
        node.location.durable = true;
        Ok(())
    }

    fn record(&self, key: &PrefixKey) -> Option<&MemoryRecord> {
        self.index
            .node(key)
            .filter(|node| node.ready)
            .map(|node| &node.location)
    }

    fn remove_record(&mut self, key: &PrefixKey) -> Result<()> {
        for node in self.index.remove(key)? {
            self.release(&node.location);
        }
        Ok(())
    }

    fn lookup(&mut self, history: &PrefixHistory, resident: u64, before: u64) -> Option<HostHit> {
        if self.flat {
            return None;
        }
        self.stats.lookups += 1;
        let chain = self
            .index
            .lookup_below(history, resident, before, |node| node.ready)?;
        self.stats.hits += 1;
        let bytes = self.layout.record_bytes(chain.pages as u64);
        Some(HostHit {
            key: chain.key,
            consumed_tokens: chain.consumed_tokens,
            resident_tokens: chain.resident_tokens,
            pages: chain.pages,
            bytes,
            tier: Tier::Memory,
            estimated_seconds: bytes as f64 / self.inbound_rate.0.max(1.0),
            chain,
        })
    }

    fn fetch_chain(&mut self, chain: &Match, ticket: InboundTicket) -> Result<()> {
        if ticket.key != chain.key {
            return Err(invalid(
                "inbound ticket and host chain name different endpoint keys",
            ));
        }
        let Some((last, _)) = chain.nodes.split_last() else {
            return Err(invalid("host chain names no records"));
        };
        if self
            .inbound
            .values()
            .any(|inbound| inbound.key == ticket.key)
        {
            return Err(invalid("an inbound transfer for this key is in flight"));
        }
        let mut slabs = Vec::with_capacity(chain.pages);
        let mut row = None;
        for (position, key) in chain.nodes.iter().enumerate() {
            let node = self
                .index
                .node(key)
                .ok_or_else(|| invalid("host chain names a record that is not resident"))?;
            if !node.ready {
                return Err(invalid(
                    "host chain names a record whose copy is incomplete",
                ));
            }
            let record = &node.location;
            let skip = if position == 0 { chain.skip_pages } else { 0 };
            if skip > record.pages.len() {
                return Err(invalid(format!(
                    "host chain skips {skip} pages of a {}-page record",
                    record.pages.len()
                )));
            }
            slabs.extend(
                record.pages[skip..]
                    .iter()
                    .map(|slab| self.pool.page_address(*slab)),
            );
            if key == last {
                let slab = record
                    .row
                    .ok_or_else(|| invalid("host chain endpoint has no row"))?;
                row = Some((ticket.slot, self.pool.row_address(slab)));
            }
        }
        if slabs.len() != ticket.pages.len() || slabs.len() != chain.pages {
            return Err(invalid(format!(
                "host chain supplies {} pages for a ticket of {} and a match of {}",
                slabs.len(),
                ticket.pages.len(),
                chain.pages
            )));
        }
        for (pinned, key) in chain.nodes.iter().enumerate() {
            if let Err(error) = self.index.pin(key) {
                for key in &chain.nodes[..pinned] {
                    let _ = self.index.unpin(key);
                }
                return Err(error);
            }
        }
        let id = self.next_job();
        let pages = ticket.pages.iter().copied().zip(slabs);
        let queued = windows(id, Direction::In, pages, row);
        self.inbound.insert(
            id,
            Inbound {
                key: ticket.key,
                nodes: chain.nodes.clone(),
                tokens: chain.consumed_tokens.saturating_sub(chain.resident_tokens),
                pending: Pending::new(queued),
            },
        );
        Ok(())
    }

    /// Account for one completed window; a reply for a job this store did not
    /// submit is ignored.
    fn absorb(&mut self, reply: Reply) {
        if let Some(outbound) = self.outbound.get_mut(&reply.id) {
            self.outbound_rate.observe(reply.bytes, reply.elapsed);
            outbound.pending.absorb(reply);
        } else if let Some(inbound) = self.inbound.get_mut(&reply.id) {
            self.inbound_rate.observe(reply.bytes, reply.elapsed);
            inbound.pending.absorb(reply);
        }
    }

    /// Submit queued windows, restores before copy-outs. A refused window
    /// waits for a reply to free the queue, unless nothing is in flight to
    /// produce one: then its record fails.
    fn pump(&mut self, submit: &mut impl FnMut(Job) -> Result<()>) {
        let ids: Vec<JobId> = self.inbound.keys().copied().collect();
        for id in ids {
            let refused = self
                .inbound
                .get_mut(&id)
                .and_then(|inbound| inbound.pending.pump(submit));
            if let Some(error) = refused
                && self.in_flight_jobs() == 0
                && let Some(inbound) = self.inbound.get_mut(&id)
            {
                inbound.pending.fail(error);
            }
        }
        let ids: Vec<JobId> = self.outbound.keys().copied().collect();
        for id in ids {
            let refused = self
                .outbound
                .get_mut(&id)
                .and_then(|outbound| outbound.pending.pump(submit));
            if let Some(error) = refused
                && self.in_flight_jobs() == 0
                && let Some(outbound) = self.outbound.get_mut(&id)
            {
                outbound.pending.fail(error);
            }
        }
    }

    /// Finish every record whose windows have all replied, then start the
    /// publications that waited for a finished copy-out.
    fn settle(&mut self, events: &mut Vec<StoreEvent>) {
        let done: Vec<JobId> = self
            .outbound
            .iter()
            .filter(|(_, outbound)| outbound.pending.done())
            .map(|(id, _)| *id)
            .collect();
        for id in done {
            if let Some(outbound) = self.outbound.remove(&id) {
                self.finish_outbound(outbound, events);
            }
        }
        let done: Vec<JobId> = self
            .inbound
            .iter()
            .filter(|(_, inbound)| inbound.pending.done())
            .map(|(id, _)| *id)
            .collect();
        for id in done {
            if let Some(inbound) = self.inbound.remove(&id) {
                self.finish_inbound(inbound, events);
            }
        }
    }

    fn finish_outbound(&mut self, outbound: Outbound, events: &mut Vec<StoreEvent>) {
        self.stats.progress += 1;
        let Outbound {
            checkpoint,
            key,
            pending,
        } = outbound;
        let result = match pending.error {
            None => match self.complete_record(&key, true) {
                Ok(()) => {
                    self.stats.stored += 1;
                    self.stats.bytes_out += pending.bytes;
                    Ok(pending.bytes)
                }
                Err(error) => Err(error),
            },
            Some(error) => {
                self.stats.store_failed += 1;
                if let Err(removal) = self.complete_record(&key, false) {
                    Err(invalid(format!("{error}; {removal}")))
                } else {
                    Err(error)
                }
            }
        };
        events.push(StoreEvent::Stored {
            checkpoint,
            key,
            result,
        });
        self.start_deferred(&key, events);
    }

    /// Start the publications that waited for `finished`. One that cannot
    /// start fails with an event, and the publications waiting for it follow
    /// in turn, so nothing waits for a key whose copy-out will never finish.
    fn start_deferred(&mut self, finished: &PrefixKey, events: &mut Vec<StoreEvent>) {
        let mut finished = vec![*finished];
        while let Some(parent) = finished.pop() {
            let mut waiting = Vec::new();
            self.deferred.retain(|(waited, checkpoint)| {
                if *waited == parent {
                    waiting.push(checkpoint.clone());
                    false
                } else {
                    true
                }
            });
            for checkpoint in waiting {
                let (parent, tail) = self.shape(&checkpoint);
                let result = match self.start_record(&checkpoint, parent, tail, true) {
                    Ok(true) => continue,
                    Ok(false) => {
                        events.push(StoreEvent::Deferred {
                            checkpoint: checkpoint.id,
                        });
                        finished.push(checkpoint.key);
                        continue;
                    }
                    Err(error) => Err(error),
                };
                self.stats.store_failed += 1;
                events.push(StoreEvent::Stored {
                    checkpoint: checkpoint.id,
                    key: checkpoint.key,
                    result,
                });
                finished.push(checkpoint.key);
            }
        }
    }

    fn finish_inbound(&mut self, inbound: Inbound, events: &mut Vec<StoreEvent>) {
        self.stats.progress += 1;
        let Inbound {
            key,
            nodes,
            tokens,
            pending,
        } = inbound;
        for node in &nodes {
            let _ = self.index.unpin(node);
        }
        let result = match pending.error {
            None => {
                for node in &nodes {
                    self.index.touch(node);
                }
                self.stats.fetched += 1;
                self.stats.bytes_in += pending.bytes;
                self.stats.reused_tokens += tokens;
                Ok(pending.bytes)
            }
            Some(error) => {
                self.stats.fetch_failed += 1;
                Err(error)
            }
        };
        events.push(StoreEvent::Fetched { key, result });
    }

    /// Drop every transfer in progress: unready records leave the index and
    /// pinned chains are released. No event follows.
    fn abandon(&mut self) {
        let outbound = std::mem::take(&mut self.outbound);
        for (_, outbound) in outbound {
            let _ = self.complete_record(&outbound.key, false);
        }
        let inbound = std::mem::take(&mut self.inbound);
        for (_, inbound) in inbound {
            for node in &inbound.nodes {
                let _ = self.index.unpin(node);
            }
        }
        self.deferred.clear();
    }

    fn resident_bytes(&self) -> u64 {
        let pages = (self.pool.page_count() - self.pool.free_pages()) as u64;
        let rows = (self.pool.row_count() - self.pool.free_rows()) as u64;
        pages * self.pool.page_bytes() as u64 + rows * self.pool.row_bytes() as u64
    }

    fn stats(&self) -> StoreStats {
        StoreStats {
            outbound_in_flight: self.outbound.len() + self.deferred.len(),
            inbound_in_flight: self.inbound.len(),
            memory_records: self.index.len(),
            memory_bytes: self.resident_bytes(),
            outbound_rate: self.outbound_rate.0,
            inbound_rate: self.inbound_rate.0,
            ..self.stats
        }
    }
}

/// Split a record into transfer windows of at most [`WINDOW_PAGES`] pages,
/// attaching the row to the last one. A record without pages is one window
/// holding only the row.
fn windows(
    id: JobId,
    direction: Direction,
    pages: impl IntoIterator<Item = (PageId, usize)>,
    row: Option<(StateSlotId, usize)>,
) -> VecDeque<Job> {
    let pages: Vec<(PageId, usize)> = pages.into_iter().collect();
    let mut queued = VecDeque::with_capacity(pages.len().div_ceil(WINDOW_PAGES).max(1));
    let mut chunks = pages.chunks(WINDOW_PAGES).peekable();
    while let Some(chunk) = chunks.next() {
        let last = chunks.peek().is_none();
        queued.push_back(Job {
            id,
            window: queued.len() as u32,
            direction,
            pages: chunk.to_vec(),
            row: if last { row } else { None },
        });
    }
    if queued.is_empty() && row.is_some() {
        queued.push_back(Job {
            id,
            window: 0,
            direction,
            pages: Vec::new(),
            row,
        });
    }
    queued
}

pub struct MemoryStore {
    core: Core,
    transfers: Arc<Mutex<Transfers>>,
    /// Reply scratch reused across polls.
    replies: Vec<Reply>,
    /// Completions produced outside `poll`, delivered by the next `poll`.
    events: Vec<StoreEvent>,
}

impl MemoryStore {
    /// A standalone store whose index mirrors the GPU checkpoint tree.
    pub fn new(
        identity: [u8; 32],
        pool: SlabPool,
        transfers: Arc<Mutex<Transfers>>,
    ) -> Result<Self> {
        Self::open(identity, pool, transfers, false)
    }

    /// A store inside a composite: every node is a root and the composite's
    /// disk index owns parent links.
    pub fn new_flat(
        identity: [u8; 32],
        pool: SlabPool,
        transfers: Arc<Mutex<Transfers>>,
    ) -> Result<Self> {
        Self::open(identity, pool, transfers, true)
    }

    fn open(
        identity: [u8; 32],
        pool: SlabPool,
        transfers: Arc<Mutex<Transfers>>,
        flat: bool,
    ) -> Result<Self> {
        let layout = transfers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .layout()
            .clone();
        if pool.page_bytes() as u64 != layout.page_record_bytes
            || pool.row_bytes() as u64 != layout.row_record_bytes
        {
            return Err(invalid(format!(
                "slab pool holds {}-byte page and {}-byte row slabs; the record layout needs {} and {}",
                pool.page_bytes(),
                pool.row_bytes(),
                layout.page_record_bytes,
                layout.row_record_bytes
            )));
        }
        Ok(Self {
            core: Core::new(identity, pool, layout, flat),
            transfers,
            replies: Vec::new(),
            events: Vec::new(),
        })
    }

    /// Switch an empty store to flat parent links.
    pub fn into_flat(mut self) -> Result<Self> {
        if !self.core.index.is_empty() || !self.core.deferred.is_empty() {
            return Err(invalid(
                "a memory store with records cannot switch to flat parent links",
            ));
        }
        self.core.flat = true;
        Ok(self)
    }

    pub fn flat(&self) -> bool {
        self.core.flat
    }

    pub fn index(&self) -> &Index<MemoryRecord> {
        &self.core.index
    }

    pub(crate) fn transfers(&self) -> &Arc<Mutex<Transfers>> {
        &self.transfers
    }

    pub(crate) fn reclaimable_slabs(&self) -> (usize, usize) {
        self.core
            .index
            .nodes()
            .filter(|node| node.ready && !node.pinned() && node.location.durable)
            .fold(
                (self.core.pool.free_pages(), self.core.pool.free_rows()),
                |(pages, rows), node| {
                    (
                        pages + node.location.pages.len(),
                        rows + usize::from(node.location.row.is_some()),
                    )
                },
            )
    }

    pub fn pool(&self) -> &SlabPool {
        &self.core.pool
    }

    pub fn identity(&self) -> [u8; 32] {
        self.core.identity
    }

    pub fn layout(&self) -> &Layout {
        &self.core.layout
    }

    /// Store the checkpoint's `tail` pages and row as a record with the given
    /// parent link; the composite decides the shape from its disk index. Under
    /// flat links the node is inserted as a root whatever `parent` says.
    pub(crate) fn store_record(
        &mut self,
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        tail: &[PageId],
        durable: bool,
    ) -> Result<bool> {
        let accepted = self.core.store_record(checkpoint, parent, tail, durable)?;
        if accepted {
            self.pump();
        }
        Ok(accepted)
    }

    /// Allocate slabs for a record read in from elsewhere, evicting under the
    /// policy, and index it unready and pinned; `None` when no slabs can be
    /// freed. The record is durable: it exists outside this store already.
    pub(crate) fn allocate_record(
        &mut self,
        key: PrefixKey,
        bucket: [u8; 16],
        tail_pages: usize,
        row: bool,
    ) -> Result<Option<MemoryRecord>> {
        self.core.allocate_record(key, bucket, tail_pages, row)
    }

    /// Finish a record from `allocate_record`: on success it becomes ready and
    /// unpinned; on failure it is unpinned, removed, and its slabs released.
    pub(crate) fn complete_record(&mut self, key: &PrefixKey, success: bool) -> Result<()> {
        self.core.complete_record(key, success)
    }

    pub(crate) fn mark_durable(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.mark_durable(key)
    }

    /// The slabs of a ready record.
    pub(crate) fn record(&self, key: &PrefixKey) -> Option<&MemoryRecord> {
        self.core.record(key)
    }

    /// Remove an unpinned childless record and release its slabs.
    pub(crate) fn remove_record(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.remove_record(key)
    }

    /// Hold a record in place while a job outside this store references its
    /// slabs: eviction and removal refuse pinned records.
    pub(crate) fn pin_record(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.index.pin(key)
    }

    pub(crate) fn unpin_record(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.index.unpin(key)
    }

    /// Copy a chain of resident records into the ticket's ids. An error means
    /// nothing was submitted and no event follows.
    pub(crate) fn fetch_chain(&mut self, chain: &Match, ticket: InboundTicket) -> Result<()> {
        self.core.fetch_chain(chain, ticket)?;
        self.pump();
        Ok(())
    }

    fn pump(&mut self) {
        let mut transfers = self
            .transfers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.core.pump(&mut |job| transfers.submit(job));
        drop(transfers);
        self.core.settle(&mut self.events);
    }
}

impl PrefixStore for MemoryStore {
    fn lookup(&mut self, history: &PrefixHistory, resident: u64, before: u64) -> Option<HostHit> {
        self.core.lookup(history, resident, before)
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        if self.core.copying_out(&checkpoint.key) {
            self.core.stats.store_skipped += 1;
            return Ok(StoreOutcome::Retry(StoreReason::Inflight));
        }
        if let Some(node) = self.core.index.node(&checkpoint.key) {
            if node.ready && node.row {
                self.core.stats.store_skipped += 1;
                return Ok(StoreOutcome::Present);
            }
            if node.pinned() {
                self.core.stats.store_skipped += 1;
                return Ok(StoreOutcome::Retry(StoreReason::Inflight));
            }
            // Memory-only rows can be republished as complete roots after compaction.
            // A node with descendants must retain its page chain until they are evicted.
            if node.children() != 0 {
                self.core.stats.store_skipped += 1;
                return Ok(StoreOutcome::Retry(StoreReason::Slabs));
            }
            self.core.remove_record(&checkpoint.key)?;
        }
        let (_, tail) = self.core.shape(checkpoint);
        if tail.len() > self.core.pool.page_count() || self.core.pool.row_count() == 0 {
            self.core.stats.store_skipped += 1;
            return Ok(StoreOutcome::Oversize);
        }
        let accepted = self.core.store(checkpoint)?;
        if accepted {
            self.pump();
        }
        Ok(if accepted {
            StoreOutcome::Queued
        } else {
            StoreOutcome::Retry(StoreReason::Slabs)
        })
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        self.fetch_chain(&hit.chain, ticket)
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        events.append(&mut self.events);
        self.transfers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .poll(&mut self.replies);
        for reply in self.replies.drain(..) {
            self.core.absorb(reply);
        }
        self.core.settle(events);
        self.pump();
        events.append(&mut self.events);
        Ok(())
    }

    fn stats(&self) -> StoreStats {
        self.core.stats()
    }

    /// Wait up to [`CLOSE_TIMEOUT`] for records in transfer to finish, so
    /// copy-outs that complete become resident, then stop the transfer
    /// threads and abandon whatever remains after a confirmed drain. Events
    /// raised meanwhile are dropped: nothing polls a closed store.
    fn close(&mut self) -> Result<()> {
        let deadline = Instant::now() + CLOSE_TIMEOUT;
        let mut events = Vec::new();
        while !(self.core.outbound.is_empty() && self.core.inbound.is_empty())
            && Instant::now() < deadline
        {
            let _ = self.poll(&mut events);
            events.clear();
            if !(self.core.outbound.is_empty() && self.core.inbound.is_empty()) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        self.transfers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .close()?;
        self.core.abandon();
        self.events.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::geometry::tests::geometry;

    const IDENTITY: [u8; 32] = [7; 32];
    const PAGE: u64 = 8192;
    const ROW: u64 = 4096;
    const PT: u64 = 64;

    /// One rank with two 4 KiB planes and a 4 KiB row: 8 KiB page records, 4 KiB row records.
    fn layout() -> Layout {
        Layout::new(&[geometry(PT, &[("k", 4096), ("v", 4096)], 4096)]).unwrap()
    }

    fn store_core(pages: usize, rows: usize, flat: bool) -> Core {
        let pool = SlabPool::new(PAGE, pages, ROW, rows, true).unwrap();
        Core::new(IDENTITY, pool, layout(), flat)
    }

    fn history(tokens: u64, salt: u32) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761).wrapping_add(salt))
            .collect();
        history.append(&tokens, &[]).unwrap();
        history
    }

    /// The `n`th checkpoint id a GPU cache mints, from a chain of one-token pages.
    fn checkpoint_id(n: u32) -> CheckpointId {
        use crate::cache::{Fence, PrefixCheckpointCache, Publication, SnapshotId, StateSnapshot};
        let mut cache = PrefixCheckpointCache::new(IDENTITY, n + 1, n + 1, 1).unwrap();
        let mut history = PrefixHistory::new(IDENTITY);
        history.append(&vec![1; n as usize], &[]).unwrap();
        let mut parent = None;
        for consumed in 1..=n {
            let published = cache
                .publish(
                    &history,
                    &Publication {
                        parent,
                        consumed_tokens: u64::from(consumed),
                        tail_pages: vec![PageId(consumed - 1)],
                        snapshot: StateSnapshot {
                            id: SnapshotId(consumed - 1),
                            consumed_tokens: u64::from(consumed),
                        },
                        ready_after: Fence(0),
                    },
                )
                .unwrap();
            parent = Some(published.id());
        }
        parent.unwrap()
    }

    /// A publication at `consumed` tokens whose chain pages are numbered from
    /// `first_page`, with the last `consumed / PT - parent / PT` belonging to it.
    fn published(
        history: &PrefixHistory,
        consumed: u64,
        parent: Option<u64>,
        first_page: u32,
        slot: u32,
    ) -> PublishedCheckpoint {
        let chain = (consumed / PT) as u32;
        let tail = chain - parent.map_or(0, |p| (p / PT) as u32);
        PublishedCheckpoint {
            id: checkpoint_id(slot),
            key: history.key_at(consumed).unwrap(),
            parent: parent.map(|p| history.key_at(p).unwrap()),
            bucket: history.bucket_at(consumed).unwrap(),
            chain_pages: (first_page..first_page + chain).map(PageId).collect(),
            tail_pages: tail as usize,
            slot: StateSlotId(slot),
        }
    }

    fn reply(job: &Job, result: Result<()>) -> Reply {
        let bytes = if result.is_ok() {
            job.pages.len() as u64 * PAGE + if job.row.is_some() { ROW } else { 0 }
        } else {
            0
        };
        Reply {
            id: job.id,
            window: job.window,
            direction: job.direction,
            result,
            bytes,
            elapsed: Duration::from_millis(1),
        }
    }

    /// Pumps `core` through a submit stub that accepts every window.
    fn pump_into(core: &mut Core, submitted: &mut Vec<Job>) {
        core.pump(&mut |job| {
            submitted.push(job);
            Ok(())
        });
    }

    /// Completes a store accepted by `core` through a submit stub, returning its events.
    fn finish(core: &mut Core) -> Vec<StoreEvent> {
        let mut events = Vec::new();
        loop {
            let mut submitted = Vec::new();
            core.pump(&mut |job| {
                submitted.push(job);
                Ok(())
            });
            if submitted.is_empty() {
                core.settle(&mut events);
                if core.outbound.is_empty() && core.inbound.is_empty() {
                    return events;
                }
                panic!("records remain with nothing submitted");
            }
            for job in &submitted {
                core.absorb(reply(job, Ok(())));
            }
            core.settle(&mut events);
        }
    }

    fn stored(events: &[StoreEvent]) -> Vec<(PrefixKey, Option<u64>)> {
        events
            .iter()
            .map(|event| match event {
                StoreEvent::Stored { key, result, .. } => (*key, result.as_ref().ok().copied()),
                StoreEvent::Fetched { .. } => panic!("unexpected fetch event"),
                StoreEvent::Deferred { .. } => panic!("unexpected deferred event"),
            })
            .collect()
    }

    #[test]
    fn store_dedupes_shapes_and_defers_children_of_records_in_flight() {
        let mut core = store_core(32, 4, false);
        let full = history(1200, 1);
        let root = published(&full, 200, None, 10, 1);
        assert!(core.store(&root).unwrap());
        assert!(!core.store(&root).unwrap());
        assert_eq!(core.stats().store_skipped, 1);
        assert_eq!(core.outbound.len(), 1);
        let node = core.index.node(&root.key).unwrap();
        assert!(!node.ready);
        assert!(node.pinned());
        assert_eq!(node.location.pages.len(), 3);
        assert!(node.location.row.is_some());
        assert!(node.location.durable);
        assert_eq!(core.pool.free_pages(), 29);
        assert_eq!(core.pool.free_rows(), 3);

        // The child arrives while the root's copy-out is in flight: it waits.
        let child = published(&full, 400, Some(200), 10, 2);
        assert!(core.store(&child).unwrap());
        assert!(!core.store(&child).unwrap());
        assert!(!core.index.contains(&child.key));
        assert_eq!(core.deferred.len(), 1);
        assert_eq!(core.stats().outbound_in_flight, 2);

        let mut submitted = Vec::new();
        core.pump(&mut |job| {
            submitted.push(job);
            Ok(())
        });
        assert_eq!(submitted.len(), 1);
        let job = &submitted[0];
        assert_eq!(job.direction, Direction::Out);
        assert_eq!(
            job.pages.iter().map(|(page, _)| *page).collect::<Vec<_>>(),
            [PageId(10), PageId(11), PageId(12)]
        );
        assert_eq!(job.row.map(|(slot, _)| slot), Some(StateSlotId(1)));
        let slabs = &core.index.node(&root.key).unwrap().location;
        for ((_, address), slab) in job.pages.iter().zip(&slabs.pages) {
            assert_eq!(*address, core.pool.page_address(*slab));
        }
        assert_eq!(
            job.row.unwrap().1,
            core.pool.row_address(slabs.row.unwrap())
        );

        let mut events = Vec::new();
        core.absorb(reply(job, Ok(())));
        core.settle(&mut events);
        assert_eq!(stored(&events), [(root.key, Some(3 * PAGE + ROW))]);
        let node = core.index.node(&root.key).unwrap();
        assert!(node.ready);
        assert!(!node.pinned());
        assert_eq!(core.stats().stored, 1);
        assert_eq!(core.stats().bytes_out, 3 * PAGE + ROW);

        // The deferred child starts as the root's child with only its tail pages.
        assert!(core.deferred.is_empty());
        assert_eq!(core.outbound.len(), 1);
        let node = core.index.node(&child.key).unwrap();
        assert_eq!(node.parent, Some(root.key));
        assert_eq!(node.tail_pages, 3);
        assert_eq!(node.location.pages.len(), 3);
        assert_eq!(core.index.node(&root.key).unwrap().children(), 1);
        let job = core
            .outbound
            .values()
            .next()
            .unwrap()
            .pending
            .queued
            .front()
            .unwrap()
            .clone();
        assert_eq!(
            job.pages.iter().map(|(page, _)| *page).collect::<Vec<_>>(),
            [PageId(13), PageId(14), PageId(15)]
        );
        assert_eq!(
            stored(&finish(&mut core)),
            [(child.key, Some(3 * PAGE + ROW))]
        );

        // A checkpoint whose parent is absent becomes a root holding its whole chain.
        let orphan = published(&full, 1000, Some(900), 40, 3);
        assert!(core.store(&orphan).unwrap());
        let node = core.index.node(&orphan.key).unwrap();
        assert_eq!(node.parent, None);
        assert_eq!(node.tail_pages, 15);
        assert_eq!(node.location.pages.len(), 15);
        let job = core
            .outbound
            .values()
            .next()
            .unwrap()
            .pending
            .queued
            .front()
            .unwrap()
            .clone();
        assert_eq!(job.pages.len(), 15);
        assert_eq!(job.pages[0].0, PageId(40));
        assert_eq!(
            stored(&finish(&mut core)),
            [(orphan.key, Some(15 * PAGE + ROW))]
        );
        assert_eq!(core.index.len(), 3);
        assert_eq!(core.stats().memory_records, 3);
        assert_eq!(core.stats().memory_bytes, 21 * PAGE + 3 * ROW);

        let mut bad = published(&full, 600, Some(400), 50, 4);
        bad.tail_pages = bad.chain_pages.len() + 1;
        assert!(core.store(&bad).is_err());
        assert!(!core.index.contains(&bad.key));
    }

    #[test]
    fn a_failed_copy_out_removes_the_record_and_restarts_children_as_roots() {
        let mut core = store_core(32, 4, false);
        let full = history(1200, 2);
        let root = published(&full, 200, None, 0, 1);
        let child = published(&full, 400, Some(200), 0, 2);
        assert!(core.store(&root).unwrap());
        assert!(core.store(&child).unwrap());
        let mut submitted = Vec::new();
        core.pump(&mut |job| {
            submitted.push(job);
            Ok(())
        });
        assert_eq!(submitted.len(), 1);
        let mut events = Vec::new();
        core.absorb(reply(&submitted[0], Err(invalid("copy engine fault"))));
        core.settle(&mut events);
        assert_eq!(stored(&events), [(root.key, None)]);
        assert!(!core.index.contains(&root.key));
        assert_eq!(core.stats().store_failed, 1);
        // The child no longer has a parent in the index: it stores its whole chain.
        let node = core.index.node(&child.key).unwrap();
        assert_eq!(node.parent, None);
        assert_eq!(node.location.pages.len(), 6);
        assert_eq!(core.pool.free_pages(), 26);
        assert_eq!(core.pool.free_rows(), 3);
        assert_eq!(
            stored(&finish(&mut core)),
            [(child.key, Some(6 * PAGE + ROW))]
        );
        assert_eq!(core.pool.free_pages(), 26);
        assert_eq!(core.stats().stored, 1);
    }

    #[test]
    fn windows_hold_sixty_four_pages_and_two_are_in_flight_at_a_time() {
        let pages: Vec<(PageId, usize)> = (0..200u32).map(|id| (PageId(id), 0x1000)).collect();
        let row = Some((StateSlotId(5), 0x9000));
        let queued = windows(JobId(3), Direction::Out, pages.clone(), row);
        assert_eq!(queued.len(), 4);
        for (window, job) in queued.iter().enumerate() {
            assert_eq!(job.id, JobId(3));
            assert_eq!(job.window as usize, window);
            assert_eq!(job.direction, Direction::Out);
            let expected = if window == 3 { 8 } else { WINDOW_PAGES };
            assert_eq!(job.pages.len(), expected);
            assert_eq!(job.pages[0].0, PageId(window as u32 * WINDOW_PAGES as u32));
            assert_eq!(job.row, if window == 3 { row } else { None });
        }
        let exact = windows(JobId(4), Direction::In, pages[..128].to_vec(), row);
        assert_eq!(exact.len(), 2);
        assert_eq!(exact[1].row, row);
        let only_row = windows(JobId(5), Direction::In, Vec::new(), row);
        assert_eq!(only_row.len(), 1);
        assert!(only_row[0].pages.is_empty());
        assert_eq!(only_row[0].row, row);
        assert!(windows(JobId(6), Direction::In, Vec::new(), None).is_empty());

        let mut core = store_core(256, 2, false);
        let long = history(200 * PT + 1, 3);
        let checkpoint = published(&long, 200 * PT + 1, None, 0, 1);
        assert!(core.store(&checkpoint).unwrap());
        let mut submitted = Vec::new();
        pump_into(&mut core, &mut submitted);
        assert_eq!(submitted.len(), 2);
        assert_eq!(core.in_flight_jobs(), 2);
        {
            let pending = &core.outbound.values().next().unwrap().pending;
            assert_eq!(pending.in_flight, 2);
            assert_eq!(pending.queued.len(), 2);
            assert_eq!(pending.queued[0].window, 2);
        }
        pump_into(&mut core, &mut submitted);
        assert_eq!(submitted.len(), 2);

        let mut events = Vec::new();
        core.absorb(reply(&submitted[0], Ok(())));
        core.settle(&mut events);
        assert!(events.is_empty());
        pump_into(&mut core, &mut submitted);
        assert_eq!(submitted.len(), 3);
        assert_eq!(submitted[2].window, 2);
        assert!(submitted[2].row.is_none());
        core.absorb(reply(&submitted[1], Ok(())));
        core.absorb(reply(&submitted[2], Ok(())));
        core.settle(&mut events);
        assert!(events.is_empty());
        pump_into(&mut core, &mut submitted);
        assert_eq!(submitted.len(), 4);
        assert_eq!(submitted[3].window, 3);
        assert_eq!(submitted[3].pages.len(), 8);
        assert!(submitted[3].row.is_some());
        core.absorb(reply(&submitted[3], Ok(())));
        core.settle(&mut events);
        assert_eq!(stored(&events), [(checkpoint.key, Some(200 * PAGE + ROW))]);
        assert_eq!(core.in_flight_jobs(), 0);
        assert!(core.outbound.is_empty());
        assert!(core.index.node(&checkpoint.key).unwrap().ready);
    }

    #[test]
    fn a_refused_window_waits_while_replies_are_due_and_fails_otherwise() {
        let mut core = store_core(32, 4, false);
        let full = history(1200, 4);
        let first = published(&full, 200, None, 0, 1);
        let second = published(&full, 300, None, 10, 2);
        assert!(core.store(&first).unwrap());
        let mut accepted = Vec::new();
        core.pump(&mut |job| {
            accepted.push(job);
            Ok(())
        });
        assert_eq!(accepted.len(), 1);
        // The queue refuses the second record while the first is in flight: it waits.
        assert!(core.store(&second).unwrap());
        core.pump(&mut |_| Err(invalid("offload queue for rank 0 holds 16 jobs")));
        let waiting = core
            .outbound
            .values()
            .find(|outbound| outbound.key == second.key)
            .unwrap();
        assert_eq!(waiting.pending.in_flight, 0);
        assert_eq!(waiting.pending.queued.len(), 1);
        assert!(waiting.pending.error.is_none());
        let mut events = Vec::new();
        core.absorb(reply(&accepted[0], Ok(())));
        core.settle(&mut events);
        assert_eq!(stored(&events), [(first.key, Some(3 * PAGE + ROW))]);
        events.clear();
        core.pump(&mut |job| {
            accepted.push(job);
            Ok(())
        });
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[1].id, JobId(2));
        core.absorb(reply(&accepted[1], Ok(())));
        core.settle(&mut events);
        assert_eq!(stored(&events), [(second.key, Some(4 * PAGE + ROW))]);

        // With nothing in flight a refusal fails the record and frees its slabs.
        let third = published(&full, 500, None, 20, 3);
        let free = (core.pool.free_pages(), core.pool.free_rows());
        assert!(core.store(&third).unwrap());
        events.clear();
        core.pump(&mut |_| Err(invalid("offload thread for rank 0 has stopped")));
        core.settle(&mut events);
        assert_eq!(stored(&events), [(third.key, None)]);
        assert!(!core.index.contains(&third.key));
        assert_eq!((core.pool.free_pages(), core.pool.free_rows()), free);
        assert_eq!(core.stats().store_failed, 1);
        assert_eq!(core.stats().stored, 2);
    }

    #[test]
    fn eviction_releases_the_slabs_of_durable_leaves_and_compacts_rows() {
        let mut core = store_core(9, 2, false);
        let full = history(1200, 5);
        let other = history(300, 6);
        let root = published(&full, 200, None, 0, 1);
        let child = published(&full, 400, Some(200), 0, 2);
        assert!(core.store(&root).unwrap());
        finish(&mut core);
        assert!(core.store(&child).unwrap());
        assert_eq!(core.pool.free_pages(), 3);
        assert_eq!(core.pool.free_rows(), 0);

        // The child is in flight and the root has that child: no leaf
        // qualifies, so the root's row is compacted away for the stranger.
        let stranger = published(&other, 250, None, 20, 3);
        assert!(core.store(&stranger).unwrap());
        let root_node = core.index.node(&root.key).unwrap();
        assert!(!root_node.row);
        assert_eq!(root_node.location.row, None);
        assert_eq!(root_node.location.pages.len(), 3);
        assert!(core.index.contains(&child.key));
        assert_eq!(core.pool.free_pages(), 0);
        assert_eq!(core.pool.free_rows(), 0);

        // Nothing is evictable for the pages a further record needs: it is skipped.
        let late = published(&other, 64, None, 30, 4);
        assert!(!core.store(&late).unwrap());
        assert_eq!(core.stats().store_skipped, 1);
        assert!(!core.index.contains(&late.key));
        assert_eq!(stored(&finish(&mut core)).len(), 2);

        // The compacted root no longer matches but still supplies its pages.
        assert!(core.lookup(&history(250, 5), 0, u64::MAX).is_none());
        let found = core.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(found.chain.nodes, [root.key, child.key]);

        // With every record ready, the least recently used leaf goes first;
        // the lookup touched the child, so the stranger is older.
        assert!(core.store(&late).unwrap());
        assert!(!core.index.contains(&stranger.key));
        assert!(core.index.contains(&child.key));
        assert!(core.index.contains(&root.key));
        assert_eq!(core.pool.free_pages(), 2);
        assert_eq!(core.pool.free_rows(), 0);
        finish(&mut core);

        // Removing the child leaves the page-only root childless: both go.
        core.index.touch(&late.key);
        let wide = published(&other, 300, None, 40, 5);
        assert!(core.store(&wide).unwrap());
        assert!(!core.index.contains(&child.key));
        assert!(!core.index.contains(&root.key));
        assert!(core.index.contains(&late.key));
        assert_eq!(core.pool.free_pages(), 4);
        assert_eq!(core.pool.free_rows(), 0);
        finish(&mut core);

        // Records that are not durable are never evicted.
        let mut core = store_core(3, 1, false);
        let fresh = published(&full, 200, None, 0, 1);
        assert!(
            core.store_record(&fresh, None, &fresh.chain_pages, false)
                .unwrap()
        );
        finish(&mut core);
        assert!(!core.store(&stranger).unwrap());
        assert!(core.index.contains(&fresh.key));
        core.mark_durable(&fresh.key).unwrap();
        assert!(core.store(&stranger).unwrap());
        assert!(!core.index.contains(&fresh.key));
        assert!(core.index.contains(&stranger.key));
        finish(&mut core);

        // A record larger than the pool is refused without evicting anything.
        let huge = published(&full, 1200, None, 0, 2);
        assert!(!core.store(&huge).unwrap());
        assert!(core.index.contains(&stranger.key));
        assert_eq!(core.pool.free_pages(), 0);
    }

    #[test]
    fn composite_records_are_allocated_completed_and_removed_through_the_flat_index() {
        let mut core = store_core(16, 4, true);
        let full = history(1200, 7);
        let key = full.key_at(400).unwrap();
        let bucket = full.bucket_at(400).unwrap();
        let record = core.allocate_record(key, bucket, 2, true).unwrap().unwrap();
        assert_eq!(record.pages.len(), 2);
        assert!(record.row.is_some());
        assert!(record.durable);
        let node = core.index.node(&key).unwrap();
        assert!(!node.ready);
        assert!(node.pinned());
        assert_eq!(node.parent, None);
        assert_eq!(node.location, record);
        assert!(core.record(&key).is_none());
        assert!(core.allocate_record(key, bucket, 2, true).is_err());
        core.complete_record(&key, true).unwrap();
        let node = core.index.node(&key).unwrap();
        assert!(node.ready);
        assert!(!node.pinned());
        assert_eq!(core.record(&key), Some(&record));
        core.remove_record(&key).unwrap();
        assert!(core.remove_record(&key).is_err());
        assert_eq!(core.pool.free_pages(), 16);
        assert_eq!(core.pool.free_rows(), 4);

        let paged = core
            .allocate_record(key, bucket, 3, false)
            .unwrap()
            .unwrap();
        assert_eq!(paged.row, None);
        assert!(!core.index.node(&key).unwrap().row);
        core.complete_record(&key, false).unwrap();
        assert!(!core.index.contains(&key));
        assert_eq!(core.pool.free_pages(), 16);
        assert!(core.complete_record(&key, false).is_err());
        assert!(core.mark_durable(&key).is_err());

        // A child record under flat links is a root holding only its tail.
        let child = published(&full, 400, Some(200), 10, 2);
        assert!(
            core.store_record(&child, child.parent, child.tail(), false)
                .unwrap()
        );
        assert!(
            !core
                .store_record(&child, child.parent, child.tail(), false)
                .unwrap()
        );
        let node = core.index.node(&child.key).unwrap();
        assert_eq!(node.parent, None);
        assert_eq!(node.tail_pages, 6);
        assert_eq!(node.location.pages.len(), 3);
        assert!(!node.location.durable);
        let job = core
            .outbound
            .values()
            .next()
            .unwrap()
            .pending
            .queued
            .front()
            .unwrap()
            .clone();
        assert_eq!(
            job.pages.iter().map(|(page, _)| *page).collect::<Vec<_>>(),
            [PageId(13), PageId(14), PageId(15)]
        );
        finish(&mut core);
        core.mark_durable(&child.key).unwrap();
        assert!(core.record(&child.key).unwrap().durable);
        assert!(core.lookup(&full, 0, u64::MAX).is_none());
        assert_eq!(core.stats().lookups, 0);
    }

    #[test]
    fn fetch_chain_validates_the_ticket_pins_the_chain_and_reports_completion() {
        let mut core = store_core(32, 4, false);
        let full = history(1200, 8);
        let root = published(&full, 200, None, 0, 1);
        let child = published(&full, 400, Some(200), 0, 2);
        assert!(core.store(&root).unwrap());
        finish(&mut core);
        assert!(core.store(&child).unwrap());
        finish(&mut core);
        let root_slabs = core.record(&root.key).unwrap().clone();
        let child_slabs = core.record(&child.key).unwrap().clone();

        // A resident checkpoint inside the root's tail skips the page it holds.
        let hit = core.lookup(&full, 100, u64::MAX).unwrap();
        assert_eq!(hit.key, child.key);
        assert_eq!(hit.tier, Tier::Memory);
        assert_eq!(hit.pages, 5);
        assert_eq!(hit.bytes, 5 * PAGE + ROW);
        assert_eq!(hit.chain.skip_pages, 1);
        assert!((hit.estimated_seconds - hit.bytes as f64 / INITIAL_MEMORY_RATE).abs() < 1e-12);
        assert_eq!(core.stats().lookups, 1);
        assert_eq!(core.stats().hits, 1);
        assert!(core.lookup(&history(100, 9), 0, u64::MAX).is_none());
        assert_eq!(core.stats().lookups, 2);
        assert_eq!(core.stats().hits, 1);

        let ticket = |pages: u32, slot: u32| InboundTicket {
            key: child.key,
            pages: (100..100 + pages).map(PageId).collect(),
            slot: StateSlotId(slot),
        };
        assert!(core.fetch_chain(&hit.chain, ticket(4, 7)).is_err());
        let mut wrong = ticket(5, 7);
        wrong.key = root.key;
        assert!(core.fetch_chain(&hit.chain, wrong).is_err());
        let mut hollow = hit.chain.clone();
        hollow.nodes.clear();
        assert!(core.fetch_chain(&hollow, ticket(5, 7)).is_err());
        let mut stranger = hit.chain.clone();
        stranger.nodes[0] = full.key_at(100).unwrap();
        assert!(core.fetch_chain(&stranger, ticket(5, 7)).is_err());
        assert!(!core.index.node(&root.key).unwrap().pinned());
        assert!(!core.index.node(&child.key).unwrap().pinned());
        assert!(core.inbound.is_empty());

        core.fetch_chain(&hit.chain, ticket(5, 7)).unwrap();
        assert!(core.fetch_chain(&hit.chain, ticket(5, 8)).is_err());
        assert!(core.index.node(&root.key).unwrap().pinned());
        assert!(core.index.node(&child.key).unwrap().pinned());
        assert_eq!(core.stats().inbound_in_flight, 1);
        let mut submitted = Vec::new();
        core.pump(&mut |job| {
            submitted.push(job);
            Ok(())
        });
        assert_eq!(submitted.len(), 1);
        let job = &submitted[0];
        assert_eq!(job.direction, Direction::In);
        let expected: Vec<(PageId, usize)> = (100..105)
            .map(PageId)
            .zip(
                root_slabs.pages[1..]
                    .iter()
                    .chain(&child_slabs.pages)
                    .map(|slab| core.pool.page_address(*slab)),
            )
            .collect();
        assert_eq!(job.pages, expected);
        assert_eq!(
            job.row,
            Some((
                StateSlotId(7),
                core.pool.row_address(child_slabs.row.unwrap())
            ))
        );
        // While the chain is pinned, eviction cannot take it.
        assert!(core.index.evict(true, |_| true).is_none());
        let mut events = Vec::new();
        core.absorb(reply(job, Ok(())));
        core.settle(&mut events);
        match events.as_slice() {
            [StoreEvent::Fetched { key, result }] => {
                assert_eq!(*key, child.key);
                assert_eq!(*result.as_ref().unwrap(), 5 * PAGE + ROW);
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!core.index.node(&root.key).unwrap().pinned());
        assert!(!core.index.node(&child.key).unwrap().pinned());
        let stats = core.stats();
        assert_eq!(stats.fetched, 1);
        assert_eq!(stats.bytes_in, 5 * PAGE + ROW);
        assert_eq!(stats.reused_tokens, 300);
        assert_eq!(stats.inbound_in_flight, 0);
        assert!(stats.inbound_rate != INITIAL_MEMORY_RATE);
        assert!(stats.outbound_rate != INITIAL_MEMORY_RATE);

        // A failed window reports the failure once every window has replied.
        let hit = core.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.pages, 6);
        core.fetch_chain(&hit.chain, ticket(6, 9)).unwrap();
        submitted.clear();
        core.pump(&mut |job| {
            submitted.push(job);
            Ok(())
        });
        events.clear();
        core.absorb(reply(&submitted[0], Err(invalid("copy engine fault"))));
        core.settle(&mut events);
        match events.as_slice() {
            [StoreEvent::Fetched { key, result }] => {
                assert_eq!(*key, child.key);
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!core.index.node(&root.key).unwrap().pinned());
        assert_eq!(core.stats().fetch_failed, 1);
        assert_eq!(core.stats().fetched, 1);

        // Abandoning drops transfers in progress without events.
        core.fetch_chain(&hit.chain, ticket(6, 9)).unwrap();
        let late = published(&full, 700, Some(400), 0, 3);
        assert!(core.store(&late).unwrap());
        core.pump(&mut |_| Ok(()));
        core.abandon();
        assert!(core.inbound.is_empty());
        assert!(core.outbound.is_empty());
        assert!(!core.index.contains(&late.key));
        assert!(!core.index.node(&child.key).unwrap().pinned());
        assert_eq!(core.pool.free_pages(), 26);
    }

    #[test]
    fn a_deferred_publication_that_cannot_start_releases_those_waiting_for_it() {
        // Five pages and one row: the non-durable root holds four pages and
        // the row, so nothing can be allocated while it stays resident.
        let mut core = store_core(5, 1, false);
        let full = history(1200, 10);
        let root = published(&full, 256, None, 0, 1);
        let child = published(&full, 320, Some(256), 0, 2);
        let grandchild = published(&full, 384, Some(320), 0, 3);
        assert!(
            core.store_record(&root, None, &root.chain_pages, false)
                .unwrap()
        );
        assert!(core.store(&child).unwrap());
        assert!(core.store(&grandchild).unwrap());
        assert_eq!(core.deferred.len(), 2);
        assert_eq!(core.stats().outbound_in_flight, 3);

        // Both blocked publications release their GPU pins and become retries.
        let events = finish(&mut core);
        assert!(
            matches!(&events[0], StoreEvent::Stored { key, result: Ok(_), .. } if *key == root.key)
        );
        assert!(
            matches!(&events[1], StoreEvent::Deferred { checkpoint } if *checkpoint == child.id)
        );
        assert!(
            matches!(&events[2], StoreEvent::Deferred { checkpoint } if *checkpoint == grandchild.id)
        );
        assert!(core.deferred.is_empty());
        assert!(core.outbound.is_empty());
        assert!(!core.known(&child.key));
        assert!(!core.known(&grandchild.key));
        let stats = core.stats();
        assert_eq!(stats.outbound_in_flight, 0);
        assert_eq!(stats.stored, 1);
        assert_eq!(stats.store_failed, 0);
        assert_eq!(stats.store_skipped, 0);
        assert_eq!(core.pool.free_pages(), 1);
        assert_eq!(core.pool.free_rows(), 0);
        let node = core.index.node(&root.key).unwrap();
        assert!(node.ready);
        assert!(!node.pinned());
        assert_eq!(node.children(), 0);

        // Neither failed key is deduped afterwards: once the root is gone the
        // child stores as a root holding its whole chain.
        core.remove_record(&root.key).unwrap();
        assert!(core.store(&child).unwrap());
        let node = core.index.node(&child.key).unwrap();
        assert_eq!(node.parent, None);
        assert_eq!(node.location.pages.len(), 5);
        assert_eq!(
            stored(&finish(&mut core)),
            [(child.key, Some(5 * PAGE + ROW))]
        );
    }

    #[test]
    fn allocating_a_child_never_evicts_the_parent_it_extends() {
        let mut core = store_core(9, 3, false);
        let full = history(1200, 11);
        let other = history(300, 12);
        let root = published(&full, 200, None, 0, 1);
        let child = published(&full, 400, Some(200), 0, 2);
        let stranger = published(&other, 200, None, 20, 3);
        for checkpoint in [&root, &child, &stranger] {
            assert!(core.store(checkpoint).unwrap());
            finish(&mut core);
        }
        assert_eq!(core.pool.free_pages(), 0);
        assert_eq!(core.pool.free_rows(), 0);

        // The child is the least recently used leaf, but it is the parent the
        // grandchild extends: the stranger is evicted instead.
        let grandchild = published(&full, 600, Some(400), 0, 4);
        assert!(core.store(&grandchild).unwrap());
        assert!(core.index.contains(&root.key));
        assert!(core.index.contains(&child.key));
        assert!(!core.index.contains(&stranger.key));
        assert_eq!(
            core.index.node(&grandchild.key).unwrap().parent,
            Some(child.key)
        );
        let parent = core.index.node(&child.key).unwrap();
        assert!(!parent.pinned());
        assert_eq!(parent.children(), 1);
        assert_eq!(
            stored(&finish(&mut core)),
            [(grandchild.key, Some(3 * PAGE + ROW))]
        );

        // With nothing else evictable the publication is skipped rather than
        // stored at its parent's expense, and the parent's pin is released.
        let great = published(&full, 800, Some(600), 0, 5);
        assert!(!core.store(&great).unwrap());
        assert_eq!(core.stats().store_skipped, 1);
        assert_eq!(core.stats().store_failed, 0);
        assert!(!core.index.contains(&great.key));
        assert!(core.index.contains(&grandchild.key));
        assert!(!core.index.node(&grandchild.key).unwrap().pinned());
        assert_eq!(core.index.len(), 3);
        assert_eq!(core.pool.free_pages(), 0);
    }

    #[test]
    fn a_refusal_after_accepted_windows_keeps_them_counted_in_flight() {
        let mut core = store_core(256, 2, false);
        let long = history(200 * PT + 1, 13);
        let first = published(&long, 200 * PT + 1, None, 0, 1);
        assert!(core.store(&first).unwrap());
        let mut accepted = Vec::new();
        core.pump(&mut |job| {
            if accepted.is_empty() {
                accepted.push(job);
                Ok(())
            } else {
                Err(invalid("offload queue for rank 0 holds 16 jobs"))
            }
        });
        assert_eq!(accepted.len(), 1);
        assert_eq!(core.in_flight_jobs(), 1);
        {
            let pending = &core.outbound.values().next().unwrap().pending;
            assert_eq!(pending.in_flight, 1);
            assert_eq!(pending.queued.len(), 3);
            assert_eq!(pending.queued[0].window, 1);
            assert!(pending.error.is_none());
        }

        // A record refused while that window is outstanding waits for its reply.
        let short = history(200, 14);
        let second = published(&short, 200, None, 220, 2);
        assert!(core.store(&second).unwrap());
        core.pump(&mut |_| Err(invalid("offload queue for rank 0 holds 16 jobs")));
        assert!(
            core.outbound
                .values()
                .all(|outbound| outbound.pending.error.is_none())
        );
        assert_eq!(core.stats().store_failed, 0);

        let mut events = Vec::new();
        core.absorb(reply(&accepted[0], Ok(())));
        core.settle(&mut events);
        assert!(events.is_empty());
        assert_eq!(core.in_flight_jobs(), 0);
        let mut submitted = Vec::new();
        pump_into(&mut core, &mut submitted);
        assert_eq!(
            submitted
                .iter()
                .map(|job| (job.id, job.window))
                .collect::<Vec<_>>(),
            [(JobId(1), 1), (JobId(1), 2), (JobId(2), 0)]
        );
        assert_eq!(core.in_flight_jobs(), 3);
        for job in &submitted {
            core.absorb(reply(job, Ok(())));
        }
        core.settle(&mut events);
        assert_eq!(stored(&events), [(second.key, Some(3 * PAGE + ROW))]);
        assert_eq!(
            stored(&finish(&mut core)),
            [(first.key, Some(200 * PAGE + ROW))]
        );
        assert_eq!(core.stats().stored, 2);
    }

    #[test]
    fn restores_are_submitted_before_copy_outs() {
        let mut core = store_core(32, 4, false);
        let full = history(1200, 15);
        let root = published(&full, 200, None, 0, 1);
        assert!(core.store(&root).unwrap());
        finish(&mut core);
        let child = published(&full, 400, Some(200), 0, 2);
        assert!(core.store(&child).unwrap());
        let hit = core.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, root.key);
        let ticket = InboundTicket {
            key: root.key,
            pages: (100..103).map(PageId).collect(),
            slot: StateSlotId(7),
        };
        core.fetch_chain(&hit.chain, ticket).unwrap();
        let mut submitted = Vec::new();
        pump_into(&mut core, &mut submitted);
        assert_eq!(
            submitted
                .iter()
                .map(|job| job.direction)
                .collect::<Vec<_>>(),
            [Direction::In, Direction::Out]
        );
    }

    #[test]
    fn rates_weight_recent_windows_and_ignore_empty_ones() {
        let mut rate = Rate(1000.0);
        rate.observe(0, Duration::from_secs(1));
        rate.observe(500, Duration::ZERO);
        assert_eq!(rate.0, 1000.0);
        rate.observe(2000, Duration::from_secs(1));
        assert!((rate.0 - 1200.0).abs() < 1e-9);
        rate.observe(2000, Duration::from_secs(1));
        assert!((rate.0 - 1360.0).abs() < 1e-9);
    }

    #[cfg(feature = "runtime-package")]
    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn memory_store_round_trips_a_multi_window_checkpoint() {
        use std::time::Instant;

        use crate::{
            cuda::Stream,
            offload::CacheGeometry,
            runtime_package::{LoadOptions, NativeModel},
            startup::StartupProgress,
        };

        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let model = unsafe {
            NativeModel::load(
                crate::runtime_package::test_support::authorize(std::path::Path::new(&package)),
                &[0],
                &LoadOptions::default(),
                &StartupProgress::silent(),
            )
        }
        .unwrap();
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank: &CacheGeometry = &geometry[0];
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let pt = layout.page_tokens;
        let pages = 99u32;
        assert!(rank.page_capacity >= 2 * u64::from(pages) + 400 && rank.row_capacity >= 4);
        let sources: Vec<u32> = (1..=pages).collect();
        let targets: Vec<u32> = (300..300 + pages).collect();
        let (source_slot, target_slot) = (1u32, 2u32);

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
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for &id in &sources {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    seed(plane, id),
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            seed(row_plane, source_slot),
        );

        let mut pool = SlabPool::sized(&layout, SlabPool::minimum_bytes(&layout)).unwrap();
        pool.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let mut store = MemoryStore::new(IDENTITY, pool, transfers.clone()).unwrap();
        let consumed = u64::from(pages) * pt;
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..consumed as u32 + 40)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        let checkpoint = PublishedCheckpoint {
            id: checkpoint_id(1),
            key: history.key_at(consumed).unwrap(),
            parent: None,
            bucket: history.bucket_at(consumed).unwrap(),
            chain_pages: sources.iter().map(|id| PageId(*id)).collect(),
            tail_pages: pages as usize,
            slot: StateSlotId(source_slot),
        };
        let wait = |store: &mut MemoryStore| -> StoreEvent {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut events = Vec::new();
            while events.is_empty() {
                assert!(Instant::now() < deadline, "no store event within 60 s");
                store.poll(&mut events).unwrap();
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(events.len(), 1, "{events:?}");
            events.pop().unwrap()
        };
        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        match wait(&mut store) {
            StoreEvent::Stored { result, .. } => {
                assert_eq!(
                    result.unwrap(),
                    u64::from(pages) * rank.page_bytes() + rank.row.unit_bytes
                );
            }
            other => panic!("unexpected event {other:?}"),
        }
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!(hit.pages, pages as usize);
        let ticket = InboundTicket {
            key: checkpoint.key,
            pages: targets.iter().map(|id| PageId(*id)).collect(),
            slot: StateSlotId(target_slot),
        };
        store.fetch(hit, ticket).unwrap();
        match wait(&mut store) {
            StoreEvent::Fetched { result, .. } => {
                assert_eq!(
                    result.unwrap(),
                    u64::from(pages) * rank.page_bytes() + rank.row.unit_bytes
                );
            }
            other => panic!("unexpected event {other:?}"),
        }
        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        let mut mismatches = Vec::new();
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for (source, target) in sources.iter().zip(&targets) {
                let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                fill(&mut wanted, seed(plane, *source));
                let actual = read_unit(
                    geometry.unit_address(u64::from(*target)),
                    geometry.unit_bytes as usize,
                );
                if actual != wanted {
                    mismatches.push((geometry.name.clone(), *source, *target));
                }
            }
        }
        let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
        fill(&mut wanted, seed(row_plane, source_slot));
        let actual = read_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
        );
        assert!(
            actual == wanted,
            "row {target_slot} differs from row {source_slot}"
        );
        assert!(
            mismatches.is_empty(),
            "mismatched planes and pages: {mismatches:?}"
        );
        store.close().unwrap();
    }

    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn memory_store_round_trips_a_checkpoint_through_a_loaded_package() {
        use std::time::Instant;

        use crate::{
            cuda::Stream,
            offload::CacheGeometry,
            runtime_package::{LoadOptions, NativeModel},
            startup::StartupProgress,
        };

        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            eprintln!("skipping: FLARE_OFFLOAD_TEST_PACKAGE is unset");
            return;
        };
        let model = unsafe {
            NativeModel::load(
                crate::runtime_package::test_support::authorize(std::path::Path::new(&package)),
                &[0],
                &LoadOptions::default(),
                &StartupProgress::silent(),
            )
        }
        .unwrap();
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = vec![manifest.cache.clone().unwrap()];
        let rank: &CacheGeometry = &geometry[0];
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        let pt = layout.page_tokens;
        assert!(rank.page_capacity >= 16 && rank.row_capacity >= 4);
        let capacity = rank.page_capacity as u32;
        let sources = [1u32, 7, capacity / 2];
        let targets = [2u32, 9, capacity / 2 + 1];
        let (source_slot, target_slot) = (1u32, 2u32);

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
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in sources {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    seed(plane, id),
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            seed(row_plane, source_slot),
        );
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in targets {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    0xDD + plane as u64,
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
            0xDD,
        );

        let mut pool = SlabPool::sized(&layout, SlabPool::minimum_bytes(&layout)).unwrap();
        pool.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap(),
        ));
        let mut store = MemoryStore::new(IDENTITY, pool, transfers.clone()).unwrap();

        let consumed = 3 * pt + 5;
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..consumed as u32 + 40)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        let checkpoint = PublishedCheckpoint {
            id: checkpoint_id(1),
            key: history.key_at(consumed).unwrap(),
            parent: None,
            bucket: history.bucket_at(consumed).unwrap(),
            chain_pages: sources.iter().map(|id| PageId(*id)).collect(),
            tail_pages: 3,
            slot: StateSlotId(source_slot),
        };

        let wait = |store: &mut MemoryStore| -> StoreEvent {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut events = Vec::new();
            while events.is_empty() {
                assert!(Instant::now() < deadline, "no store event within 60 s");
                store.poll(&mut events).unwrap();
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(events.len(), 1, "{events:?}");
            events.pop().unwrap()
        };

        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().outbound_in_flight, 1);
        match wait(&mut store) {
            StoreEvent::Stored {
                checkpoint: id,
                key,
                result,
            } => {
                assert_eq!(id, checkpoint.id);
                assert_eq!(key, checkpoint.key);
                assert_eq!(result.unwrap(), 3 * rank.page_bytes() + rank.row.unit_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        let stats = store.stats();
        assert_eq!(stats.stored, 1);
        assert_eq!(stats.outbound_in_flight, 0);
        assert_eq!(stats.memory_records, 1);
        assert_eq!(stats.memory_bytes, layout.record_bytes(3));
        assert!(stats.outbound_rate > 0.0);

        // The record's slabs hold the source pages' and row's patterns under the layout.
        let check_record = |store: &MemoryStore, key: &PrefixKey, label: &str| {
            let record = store.record(key).unwrap().clone();
            assert_eq!(record.pages.len(), 3, "{label}");
            let pool = store.pool();
            for (slab, id) in record.pages.iter().zip(sources) {
                let bytes = pool.page(*slab);
                for (plane, geometry) in rank.planes.iter().enumerate() {
                    let start = layout.ranks[0].plane_offsets[plane] as usize;
                    let mut expected = vec![0u8; geometry.unit_bytes as usize];
                    fill(&mut expected, seed(plane, id));
                    assert!(
                        bytes[start..start + expected.len()] == expected[..],
                        "{label}: slab bytes of plane {} ({plane}) page {id} differ from the device",
                        geometry.name
                    );
                }
            }
            let row_start = layout.ranks[0].row_offset as usize;
            let mut expected = vec![0u8; rank.row.unit_bytes as usize];
            fill(&mut expected, seed(row_plane, source_slot));
            assert!(
                pool.row(record.row.unwrap())[row_start..row_start + expected.len()]
                    == expected[..],
                "{label}: row slab bytes differ from the device"
            );
        };
        check_record(&store, &checkpoint.key, "first record");

        // Poison the device copies, then restore the chain into other ids.
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for id in sources {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    0xEE + plane as u64,
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            0xEE,
        );
        let hit = store.lookup(&history, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, checkpoint.key);
        assert_eq!(hit.pages, 3);
        assert_eq!(hit.bytes, layout.record_bytes(3));
        assert_eq!(hit.tier, Tier::Memory);
        assert!(hit.estimated_seconds > 0.0);
        let ticket = InboundTicket {
            key: checkpoint.key,
            pages: targets.iter().map(|id| PageId(*id)).collect(),
            slot: StateSlotId(target_slot),
        };
        store.fetch(hit, ticket).unwrap();
        assert_eq!(store.stats().inbound_in_flight, 1);
        match wait(&mut store) {
            StoreEvent::Fetched { key, result } => {
                assert_eq!(key, checkpoint.key);
                assert_eq!(result.unwrap(), 3 * rank.page_bytes() + rank.row.unit_bytes);
            }
            other => panic!("unexpected event {other:?}"),
        }
        let stats = store.stats();
        assert_eq!(stats.fetched, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.reused_tokens, consumed);
        assert_eq!(stats.inbound_in_flight, 0);
        assert!(!store.index().node(&checkpoint.key).unwrap().pinned());

        let read_unit = |address: u64, bytes: usize| -> Vec<u8> {
            unsafe { stream.download(&staging, address as usize, bytes) }.unwrap();
            stream.synchronize().unwrap();
            staging.bytes()[..bytes].to_vec()
        };
        for (plane, geometry) in rank.planes.iter().enumerate() {
            for (source, target) in sources.iter().zip(targets) {
                let mut wanted = vec![0u8; geometry.unit_bytes as usize];
                fill(&mut wanted, seed(plane, *source));
                let actual = read_unit(
                    geometry.unit_address(u64::from(target)),
                    geometry.unit_bytes as usize,
                );
                assert!(
                    actual == wanted[..],
                    "plane {} ({plane}) page {target} differs from page {source}",
                    geometry.name
                );
            }
        }
        let mut wanted = vec![0u8; rank.row.unit_bytes as usize];
        fill(&mut wanted, seed(row_plane, source_slot));
        let actual = read_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
        );
        assert!(
            actual == wanted[..],
            "row {target_slot} differs from row {source_slot}"
        );

        // The same key is already resident: no second copy-out.
        assert_ne!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        store.close().unwrap();
        assert_eq!(store.stats().memory_records, 1);

        // A pool holding one record evicts it when a second key is stored.
        let mut small = SlabPool::new(
            layout.page_record_bytes,
            3,
            layout.row_record_bytes,
            1,
            true,
        )
        .unwrap();
        small.register(context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(context, device)], geometry.clone(), &small).unwrap(),
        ));
        let mut store = MemoryStore::new(IDENTITY, small, transfers).unwrap();
        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        assert!(matches!(
            wait(&mut store),
            StoreEvent::Stored { result: Ok(_), .. }
        ));
        assert_eq!(store.pool().free_pages(), 0);
        assert_eq!(store.pool().free_rows(), 0);
        let mut other = PrefixHistory::new(IDENTITY);
        other.append(&tokens[..consumed as usize], &[]).unwrap();
        other.append(&[11, 12, 13, 14, 15], &[]).unwrap();
        let second_consumed = consumed + 5;
        let second = PublishedCheckpoint {
            id: checkpoint_id(2),
            key: other.key_at(second_consumed).unwrap(),
            parent: None,
            bucket: other.bucket_at(second_consumed).unwrap(),
            chain_pages: targets.iter().map(|id| PageId(*id)).collect(),
            tail_pages: 3,
            slot: StateSlotId(target_slot),
        };
        assert_ne!(second.key, checkpoint.key);
        assert_eq!(store.store(&second).unwrap(), StoreOutcome::Queued);
        assert!(!store.index().contains(&checkpoint.key));
        assert!(store.index().contains(&second.key));
        assert_eq!(store.pool().free_pages(), 0);
        match wait(&mut store) {
            StoreEvent::Stored { key, result, .. } => {
                assert_eq!(key, second.key);
                result.unwrap();
            }
            other => panic!("unexpected event {other:?}"),
        }
        assert_eq!(store.stats().memory_records, 1);
        assert!(store.lookup(&history, 0, u64::MAX).is_none());
        assert_eq!(store.lookup(&other, 0, u64::MAX).unwrap().key, second.key);
        check_record(&store, &second.key, "second record");

        // Closing waits for a copy-out in progress: the record is resident afterwards.
        let mut another = PrefixHistory::new(IDENTITY);
        another.append(&tokens[..consumed as usize], &[]).unwrap();
        another.append(&[21, 22, 23], &[]).unwrap();
        let third_consumed = consumed + 3;
        let third = PublishedCheckpoint {
            id: checkpoint_id(3),
            key: another.key_at(third_consumed).unwrap(),
            parent: None,
            bucket: another.bucket_at(third_consumed).unwrap(),
            chain_pages: targets.iter().map(|id| PageId(*id)).collect(),
            tail_pages: 3,
            slot: StateSlotId(target_slot),
        };
        assert_eq!(store.store(&third).unwrap(), StoreOutcome::Queued);
        assert!(!store.index().contains(&second.key));
        assert_eq!(store.stats().outbound_in_flight, 1);
        store.close().unwrap();
        let stats = store.stats();
        assert_eq!(stats.stored, 3);
        assert_eq!(stats.store_failed, 0);
        assert_eq!(stats.outbound_in_flight, 0);
        assert_eq!(stats.memory_records, 1);
        assert!(store.index().node(&third.key).unwrap().ready);
        check_record(&store, &third.key, "record stored across close");
        drop(store);
        drop(staging);
        drop(stream);
        drop(model);
    }
}
