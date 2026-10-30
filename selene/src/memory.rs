//! Records retained in registered host memory.
//!
//! A record is the page slabs and row slab of one checkpoint inside a retaining
//! [`SlabPool`]. `store` allocates slabs, evicting under the tree policy of the
//! [`Index`] when the pool is full, and copies the published GPU checkpoint out
//! through the shared [`PageCopier`](super::PageCopier) in windows of at most
//! [`WINDOW_PAGES`] pages; the record stays resident until evicted. `fetch`
//! copies a looked-up chain of resident records back into the ids a ticket
//! names. At most [`IN_FLIGHT_WINDOWS`] windows of one record are submitted at
//! a time; the rest wait in the store and follow as replies arrive.
//!
//! A checkpoint whose parent's copy-out is still in flight is held back until
//! that copy-out finishes, then stored as the parent's child, or as a root
//! record holding its whole chain when the parent's copy-out failed. A
//! held-back checkpoint that cannot start reports Deferred and releases
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
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{Arc, PoisonError},
    time::{Duration, Instant},
};

use crate::levels::Rate;
use crate::{
    CheckpointId, Error, ExportRefusal, History, PageId, PrefixKey, Result, StateSlotId,
    index::{NodeEvent, NodeNotice},
    pool::{
        Announcer, HostPin, IndexEntry, MemoryPin, Releases, Tier as PoolTier, TierView,
        engine::hostfill::HostFiller,
        pins::{MemorySpan, host_chain_kept},
        tiers::{HashJob, RecordSums},
    },
};

use super::{
    HostHit, HostStore, INITIAL_MEMORY_RATE, InboundTicket, JobId, Layout, PublishedCheckpoint,
    Retirement, Retiring, StoreEvent, StoreOutcome, StoreReason, StoreStats, Tier,
    copier::{CLOSE_TIMEOUT, Direction, Job, Reply, SharedCopier},
    geometry::invalid,
    index::{Eviction, Index, Match},
    slab::{PageSlab, RowSlab, SlabPool, WINDOW_PAGES},
};

/// Windows of one record submitted to the transfer threads at a time.
pub const IN_FLIGHT_WINDOWS: usize = crate::shared::WINDOW_CREDITS as usize;

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
    /// Eviction hands records to the pool's manager instead of dropping them.
    spill: bool,
    /// Records chosen for the manager and not yet taken by the pool store.
    retiring: Vec<Retiring>,
    /// Records held in place until their move ends: their pages and whether
    /// they hold a row slab.
    held: HashMap<PrefixKey, (usize, bool)>,
    /// Records whose moves ended and that leave once nothing reads them.
    finishing: Vec<PrefixKey>,
    largest_pages: usize,
    /// Checksum-backed holdings announced to the pool.
    announced: std::collections::HashSet<PrefixKey>,
}

impl Core {
    fn reused(&mut self, key: &PrefixKey) {
        self.index.touch(key);
    }

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
            outbound_rate: Rate::new(INITIAL_MEMORY_RATE),
            inbound_rate: Rate::new(INITIAL_MEMORY_RATE),
            next_job: 1,
            spill: false,
            retiring: Vec::new(),
            held: HashMap::new(),
            finishing: Vec::new(),
            largest_pages: 0,
            announced: Default::default(),
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
        if self.spill && self.retire_for(tail_pages, usize::from(row)) {
            return Ok(None);
        }
        while self.pool.free_pages() < tail_pages || (row && self.pool.free_rows() == 0) {
            let need_row =
                row && self.pool.free_rows() == 0 && self.pool.free_pages() >= tail_pages;
            match self.index.evict(need_row, |node| node.location.durable) {
                Some(Eviction::Leaves(nodes)) => {
                    self.stats.dropped[0] += if self.flat { 0 } else { nodes.len() as u64 };
                    for node in &nodes {
                        self.release(&node.location);
                    }
                }
                Some(Eviction::Compacted(compacted)) => {
                    self.stats.dropped[0] += u64::from(!self.flat);
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
        self.largest_pages = self.largest_pages.max(tail_pages);
        Ok(Some(record))
    }

    /// Hold bounded cold leaves until they cover the missing page and row slabs.
    /// A full tier waits while any such records remain held.
    fn retire_for(&mut self, tail_pages: usize, row_count: usize) -> bool {
        let want_pages = tail_pages.saturating_sub(self.pool.free_pages());
        let want_rows = row_count.saturating_sub(self.pool.free_rows());
        if want_pages == 0 && want_rows == 0 {
            return false;
        }
        let mut pages: usize = self.held.values().map(|(pages, _)| pages).sum();
        let mut rows: usize = self.held.values().map(|(_, row)| usize::from(*row)).sum();
        for _ in 0..crate::pool::cascade::MOVES_MAX {
            if (pages >= want_pages && rows >= want_rows)
                || self.held.len() >= crate::pool::cascade::MOVES_MAX
            {
                break;
            }
            let announced = &self.announced;
            let Some(key) = self.index.candidate(false, |node| {
                node.location.durable && announced.contains(&node.key)
            }) else {
                break;
            };
            let Some(node) = self.index.node(&key) else {
                break;
            };
            let (held_pages, held_row) = (node.location.pages.len(), node.location.row.is_some());
            let retiring = Retiring {
                tier: PoolTier::Dram,
                key,
                parent: node.parent,
                bucket: node.bucket,
                tail_pages: node.tail_pages,
                row: node.row,
            };
            if self.index.pin(&key).is_err() {
                break;
            }
            pages += held_pages;
            rows += usize::from(held_row);
            self.held.insert(key, (held_pages, held_row));
            self.retiring.push(retiring);
        }
        !self.held.is_empty()
    }

    fn headroom(&self) -> (usize, usize) {
        (
            crate::store::headroom(self.pool.page_count() as u64, self.largest_pages as u64)
                as usize,
            crate::store::headroom(self.pool.row_count() as u64, 1) as usize,
        )
    }

    /// The move of a held record ended: drop it, or let it be offered again.
    fn retired(&mut self, key: &PrefixKey, how: Retirement) {
        if self.held.remove(key).is_none() {
            return;
        }
        let _ = self.index.unpin(key);
        self.stats.held[0] += u64::from(how == Retirement::Failed);
        self.stats.progress += 1;
        if how != Retirement::Failed {
            self.finishing.push(*key);
            self.finish_retired();
        }
    }

    /// Remove the records whose moves ended, once no export pin still reads
    /// them. A record that gained a child meanwhile stays.
    fn finish_retired(&mut self) {
        let mut finishing = std::mem::take(&mut self.finishing);
        finishing.retain(|key| {
            if self.index.node(key).is_some_and(|node| node.pinned()) {
                return true;
            }
            if self.remove_record(key).is_ok() {
                self.stats.progress += 1;
            }
            false
        });
        self.finishing = finishing;
    }

    fn release(&mut self, record: &MemoryRecord) {
        self.stats.progress += 1;
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
        self.stats.progress += 1;
        if success {
            let _ = self.index.set_ready(key, true);
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
        self.stats.progress += 1;
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

    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
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
            tier: Tier::Dram,
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
                            reason: StoreReason::Slabs,
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
                    self.index.access(node);
                }
                if let Some(key) = nodes.last() {
                    self.index.touch_ancestors(key);
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
            rows_used: self.pool.row_count() - self.pool.free_rows(),
            rows_total: self.pool.row_count(),
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

/// What a memory store keeps to announce its records and serve them as exports.
struct Announce {
    view: TierView,
    hasher: HostFiller<PrefixKey>,
    /// Ready records being hashed, pinned until the result is collected.
    hashing: HashMap<PrefixKey, NodeNotice>,
    /// The checksums committed with each announced record.
    sums: HashMap<PrefixKey, Arc<RecordSums>>,
    releases: Releases,
}

impl Drop for Announce {
    fn drop(&mut self) {
        self.hasher.join();
    }
}

pub struct MemoryStore {
    // Dropped first: the checksum thread reads slabs the core frees.
    announce: Option<Announce>,
    core: Core,
    transfers: SharedCopier,
    /// Reply scratch reused across polls.
    replies: Vec<Reply>,
    /// Completions produced outside `poll`, delivered by the next `poll`.
    events: Vec<StoreEvent>,
}

impl MemoryStore {
    /// A standalone store whose index mirrors the GPU checkpoint tree.
    pub fn new(identity: [u8; 32], pool: SlabPool, transfers: SharedCopier) -> Result<Self> {
        Self::open(identity, pool, transfers, false)
    }

    /// A store inside a composite: every node is a root and the composite's
    /// disk index owns parent links.
    pub fn new_flat(identity: [u8; 32], pool: SlabPool, transfers: SharedCopier) -> Result<Self> {
        Self::open(identity, pool, transfers, true)
    }

    fn open(
        identity: [u8; 32],
        pool: SlabPool,
        transfers: SharedCopier,
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
            announce: None,
            transfers,
            replies: Vec::new(),
            events: Vec::new(),
        })
    }

    pub fn flat(&self) -> bool {
        self.core.flat
    }

    pub fn index_mut(&mut self) -> &mut Index<MemoryRecord> {
        &mut self.core.index
    }

    pub fn index(&self) -> &Index<MemoryRecord> {
        &self.core.index
    }

    pub fn transfers(&self) -> &SharedCopier {
        &self.transfers
    }

    pub fn reclaimable_slabs(&self) -> (usize, usize) {
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

    pub(crate) fn headroom(&self) -> (usize, usize) {
        let (pages, rows) = self.core.headroom();
        (pages.max(self.largest_pages()), rows)
    }

    pub(crate) fn largest_pages(&self) -> usize {
        self.core.largest_pages
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
    pub fn store_record(
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
    pub fn allocate_record(
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
    pub fn complete_record(&mut self, key: &PrefixKey, success: bool) -> Result<()> {
        self.core.complete_record(key, success)
    }

    pub fn mark_durable(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.mark_durable(key)
    }

    /// A copy-out that landed in a composite is not durable until its file
    /// commits; tests that allocate records directly say so with this.
    #[cfg(test)]
    pub(crate) fn mark_volatile(&mut self, key: &PrefixKey) {
        if let Some(node) = self.core.index.node_mut(key) {
            node.location.durable = false;
        }
    }

    /// The checksums committed with a ready, announced record. A removed
    /// record's sums are dropped by the next flush, which every poll runs
    /// before it can spill, so a spill never sees another record's sums.
    pub fn announced_sums(&self, key: &PrefixKey) -> Option<Arc<RecordSums>> {
        self.announce.as_ref()?.sums.get(key).cloned()
    }

    /// The slabs of a ready record.
    pub fn record(&self, key: &PrefixKey) -> Option<&MemoryRecord> {
        self.core.record(key)
    }

    /// Remove an unpinned childless record and release its slabs.
    pub fn remove_record(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.remove_record(key)
    }

    /// Hold a record in place while a job outside this store references its
    /// slabs: eviction and removal refuse pinned records.
    pub fn pin_record(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.index.pin(key)
    }

    pub fn unpin_record(&mut self, key: &PrefixKey) -> Result<()> {
        self.core.index.unpin(key)?;
        self.core.stats.progress += 1;
        Ok(())
    }

    /// Copy a chain of resident records into the ticket's ids. An error means
    /// nothing was submitted and no event follows.
    pub fn fetch_chain(&mut self, chain: &Match, ticket: InboundTicket) -> Result<()> {
        self.core.fetch_chain(chain, ticket)?;
        self.pump();
        Ok(())
    }

    /// Announce this store's records through `announcer` and serve them as
    /// exports. Records get their checksums on a thread of their own before they
    /// are announced. A pool split by rank is not readable by the CPU and
    /// announces nothing.
    pub fn announce_records(&mut self, announcer: Announcer) -> Result<()> {
        if self.announce.is_some() || self.core.pool.rank_parts().is_some() {
            return Ok(());
        }
        let wake = announcer.wake.clone();
        let view = TierView::new(announcer, PoolTier::Dram, &self.core.layout);
        self.core.index.observe(view.sink());
        // What is resident already counts as newly ready.
        view.sink()
            .lock()
            .unwrap()
            .extend(self.core.index.ready_notices());
        self.announce = Some(Announce {
            view,
            hasher: HostFiller::start(wake)
                .map_err(|e| invalid(format!("cannot start the checksum worker: {e}")))?,
            hashing: HashMap::new(),
            sums: HashMap::new(),
            releases: Releases::default(),
        });
        Ok(())
    }

    /// Release finished exports' pins, collect checksums and announce what
    /// changed. `parent_of` gives the parent to announce a record under, from
    /// the record's own and its key; a flat store's records have none of their
    /// own.
    pub fn flush_announcements(
        &mut self,
        parent_of: &dyn Fn(&PrefixKey, Option<PrefixKey>) -> Option<PrefixKey>,
        spilled_to: &dyn Fn(&PrefixKey) -> Option<IndexEntry>,
    ) {
        let Some(a) = self.announce.as_mut() else {
            return;
        };
        for key in a.releases.drain() {
            let _ = self.core.index.unpin(&key);
            self.core.stats.progress += 1;
        }
        let mut done = Vec::new();
        a.hasher.collect(&mut done);
        for done in done {
            let Some(notice) = a.hashing.remove(&done.owner) else {
                continue;
            };
            let _ = self.core.index.unpin(&done.owner);
            self.core.stats.progress += 1;
            if self.core.index.node(&done.owner).is_some_and(|n| n.ready) {
                a.sums.insert(
                    done.owner,
                    Arc::new(done.result.expect("hashing pinned slabs cannot fail")),
                );
                a.view
                    .announce(&notice, parent_of(&notice.key, notice.parent));
                self.core.announced.insert(done.owner);
            }
        }
        for notice in a.view.drain() {
            let key = notice.key;
            match notice.event {
                NodeEvent::Ready => {
                    if a.hashing.contains_key(&key) {
                        continue;
                    }
                    let Some(node) = self.core.index.node(&key).filter(|n| n.ready) else {
                        continue;
                    };
                    let pool = &self.core.pool;
                    let job = HashJob {
                        key,
                        pages: node
                            .location
                            .pages
                            .iter()
                            .map(|slab| pool.page_address(*slab))
                            .collect(),
                        page_bytes: pool.page_bytes(),
                        row: node.location.row.map(|slab| pool.row_address(slab)),
                        row_bytes: pool.row_bytes(),
                    };
                    if self.core.index.pin(&key).is_ok() {
                        if a.hasher.submit_hash(job) {
                            a.hashing.insert(key, notice);
                        } else {
                            let _ = self.core.index.unpin(&key);
                        }
                    }
                }
                NodeEvent::Gone => {
                    a.hashing.remove(&key);
                    a.sums.remove(&key);
                    self.core.announced.remove(&key);
                    match spilled_to(&key) {
                        Some(to) => a.view.spill(&key, to),
                        None => a.view.retire(&key),
                    }
                }
                NodeEvent::RowDropped => {
                    if a.sums.contains_key(&key) {
                        a.view.announce(&notice, parent_of(&key, notice.parent));
                    }
                }
            }
        }
        self.core.finish_retired();
    }

    /// Pin the resident records `keys` (ascending, the last the endpoint) for an
    /// export from `begin_tokens`, serving the checksums committed with them.
    pub fn pin_export_chain(
        &mut self,
        keys: &[PrefixKey],
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        let (Some(a), Some(endpoint)) = (self.announce.as_ref(), keys.last()) else {
            return Err(ExportRefusal::NotFound);
        };
        if !a.view.contains(endpoint) {
            return Err(ExportRefusal::NotFound);
        }
        let pool = &self.core.pool;
        let mut nodes = Vec::with_capacity(keys.len());
        let mut spans = Vec::with_capacity(keys.len());
        for (position, key) in keys.iter().enumerate() {
            let node = self
                .core
                .index
                .node(key)
                .filter(|node| node.ready)
                .ok_or(ExportRefusal::ChainGap)?;
            let Some(sums) = a.sums.get(key).filter(|_| a.view.contains(key)) else {
                return Err(ExportRefusal::ChainGap);
            };
            if want_row && position + 1 == keys.len() && node.location.row.is_none() {
                return Err(ExportRefusal::ChainGap);
            }
            nodes.push((*key, node.location.pages.len()));
            spans.push(MemorySpan {
                pages: node
                    .location
                    .pages
                    .iter()
                    .map(|slab| pool.page_address(*slab))
                    .collect(),
                row: node.location.row.map(|slab| pool.row_address(slab)),
                sums: sums.clone(),
            });
        }
        let page_tokens = self.core.layout.page_tokens;
        let (chain, kept) = host_chain_kept(page_tokens, begin_tokens, want_row, &nodes);
        if chain.total_pages
            != (endpoint.consumed_tokens / page_tokens).saturating_sub(begin_tokens / page_tokens)
        {
            return Err(ExportRefusal::ChainGap);
        }
        // Windows name records by their place in the chain, which leaves out nodes with no page.
        let mut spans: Vec<Option<MemorySpan>> = spans.into_iter().map(Some).collect();
        let spans: Vec<MemorySpan> = kept.iter().filter_map(|&i| spans[i].take()).collect();
        for (position, key) in keys.iter().enumerate() {
            if self.core.index.pin(key).is_err() {
                for pinned in &keys[..position] {
                    let _ = self.core.index.unpin(pinned);
                }
                return Err(ExportRefusal::NotFound);
            }
        }
        Ok(Box::new(MemoryPin {
            chain,
            spans,
            page_bytes: pool.page_bytes(),
            row_bytes: pool.row_bytes(),
            keys: keys.to_vec(),
            releases: a.releases.clone(),
        }))
    }

    /// The keys from the record holding the page that contains `begin_tokens` up
    /// to `endpoint`, ascending, following this store's own parent links.
    fn export_chain(
        &self,
        endpoint: &PrefixKey,
        begin_tokens: u64,
    ) -> std::result::Result<Vec<PrefixKey>, ExportRefusal> {
        let page_tokens = self.core.layout.page_tokens;
        let start_page = begin_tokens / page_tokens;
        let mut chain = Vec::new();
        let mut key = *endpoint;
        loop {
            let node = self.core.index.node(&key).ok_or(ExportRefusal::NotFound)?;
            chain.push(key);
            let start = (key.consumed_tokens / page_tokens).saturating_sub(node.tail_pages as u64);
            if start <= start_page {
                break;
            }
            key = node.parent.ok_or(ExportRefusal::ChainGap)?;
        }
        chain.reverse();
        Ok(chain)
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

impl HostStore for MemoryStore {
    fn reused(&mut self, key: PrefixKey) {
        self.core.reused(&key);
    }

    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
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
                self.core.index.touch_ancestors(&checkpoint.key);
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
            self.core.stats.dropped[1] += 1;
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
        // Inside a composite the disk index supplies parents, so it flushes.
        if !self.core.flat {
            for key in self.core.index.expired(|node| !node.pinned()) {
                if self.core.remove_record(&key).is_ok() {
                    self.core.stats.expired[0] += 1;
                }
            }
            self.flush_announcements(&|_, parent| parent, &|_| None);
            if self.core.spill {
                let (pages, rows) = self.headroom();
                self.core.retire_for(pages, rows);
            }
        }
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

impl crate::store::TierStore for MemoryStore {
    fn announce_to(&mut self, announcer: Announcer) {
        if let Err(error) = self.announce_records(announcer) {
            eprintln!("warning: host memory records will not be shared: {error}");
        }
    }
    fn pin_export(
        &mut self,
        tier: PoolTier,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        if tier != PoolTier::Dram || self.core.flat {
            return Err(ExportRefusal::NotFound);
        }
        let chain = self.export_chain(&key, begin_tokens)?;
        self.pin_export_chain(&chain, begin_tokens, want_row)
    }
    fn set_spill(&mut self, enabled: bool) {
        let enabled = enabled && !self.core.flat && self.announce.is_some();
        self.core.stats.progress += u64::from(self.core.spill != enabled);
        self.core.spill = enabled;
    }
    fn take_retiring(&mut self) -> Vec<Retiring> {
        std::mem::take(&mut self.core.retiring)
    }
    fn retired(&mut self, key: PrefixKey, tier: PoolTier, how: Retirement) {
        if tier == PoolTier::Dram {
            self.core.retired(&key, how);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestHistory as PrefixHistory;
    use crate::testing::geometry;

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

    /// The `n`th checkpoint id a GPU cache mints.
    fn checkpoint_id(n: u32) -> CheckpointId {
        CheckpointId::new(u64::from(n))
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
                _ => panic!("unexpected event"),
            })
            .collect()
    }

    #[test]
    fn releasing_a_host_pin_notifies_capacity_waiters() {
        let geometry = [crate::testing::geometry(64, &[("kv", PAGE)], ROW)];
        let (copier, _) = crate::testing::SimulatedGpu::shared(&geometry).unwrap();
        let pool = SlabPool::new(PAGE, 1, ROW, 1, true).unwrap();
        let mut store = MemoryStore::new(IDENTITY, pool, copier).unwrap();
        let checkpoint = published(&history(64, 1), 64, None, 0, 1);
        assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        store.pin_record(&checkpoint.key).unwrap();
        let progress = store.stats().progress;
        assert_eq!(store.reclaimable_slabs(), (0, 0));
        store.unpin_record(&checkpoint.key).unwrap();
        assert!(store.stats().progress > progress);
        assert_eq!(store.reclaimable_slabs(), (1, 1));
        store.close().unwrap();
    }

    #[test]
    fn published_prompt_copy_survives_reply_and_gpu_storage_reuse() {
        let geometry = [crate::testing::geometry(64, &[("kv", PAGE)], ROW)];
        let (copier, gpu) = crate::testing::SimulatedGpu::shared(&geometry).unwrap();
        let pool = SlabPool::new(PAGE, 2, ROW, 2, true).unwrap();
        let mut store = MemoryStore::new(IDENTITY, pool, copier).unwrap();
        let full = history(128, 1);
        let prompt = published(&full, 64, None, 0, 1);
        let reply = published(&full, 128, Some(64), 0, 2);
        gpu.pages.lock().unwrap().insert(0, vec![3; PAGE as usize]);
        gpu.rows.lock().unwrap().insert(1, vec![17; ROW as usize]);
        assert_eq!(store.store(&prompt).unwrap(), StoreOutcome::Queued);
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Stored { result: Ok(_), .. }]
        ));
        gpu.pages.lock().unwrap().insert(1, vec![4; PAGE as usize]);
        gpu.rows.lock().unwrap().insert(2, vec![18; ROW as usize]);
        assert_eq!(store.store(&reply).unwrap(), StoreOutcome::Queued);
        events.clear();
        store.poll(&mut events).unwrap();
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Stored { result: Ok(_), .. }]
        ));
        gpu.pages.lock().unwrap().insert(0, vec![99; PAGE as usize]);
        gpu.rows.lock().unwrap().insert(1, vec![99; ROW as usize]);
        let hit = store.lookup(&full, 0, 128).unwrap();
        assert_eq!(hit.key, prompt.key);
        store
            .fetch(
                hit,
                InboundTicket {
                    key: prompt.key,
                    pages: vec![PageId(10)],
                    slot: StateSlotId(3),
                },
            )
            .unwrap();
        events.clear();
        store.poll(&mut events).unwrap();
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Fetched { result: Ok(_), .. }]
        ));
        assert_eq!(gpu.pages.lock().unwrap()[&10], vec![3; PAGE as usize]);
        assert_eq!(gpu.rows.lock().unwrap()[&3], vec![17; ROW as usize]);
        store.close().unwrap();
    }

    #[test]
    fn gpu_reuse_refreshes_the_host_record_before_eviction() {
        let mut core = store_core(2, 2, false);
        let a = published(&history(64, 1), 64, None, 0, 1);
        let b = published(&history(64, 2), 64, None, 1, 2);
        let c = published(&history(64, 3), 64, None, 2, 3);
        for record in [&a, &b] {
            assert!(core.store(record).unwrap());
            finish(&mut core);
        }
        core.reused(&a.key);
        assert!(core.store(&c).unwrap());
        assert!(core.index.contains(&a.key));
        assert!(!core.index.contains(&b.key));
        finish(&mut core);
    }

    #[test]
    fn gpu_reuse_cannot_prevent_row_compaction_or_recreate_a_host_copy() {
        let mut core = store_core(3, 2, false);
        let full = history(128, 1);
        let root = published(&full, 64, None, 0, 1);
        let child = published(&full, 128, Some(64), 0, 2);
        let stranger = published(&history(64, 2), 64, None, 2, 3);
        assert!(core.store(&root).unwrap());
        finish(&mut core);
        assert!(core.store(&child).unwrap());
        finish(&mut core);
        core.index.pin(&child.key).unwrap();
        core.reused(&root.key);
        assert!(core.store(&stranger).unwrap());
        finish(&mut core);
        assert!(!core.index.node(&root.key).unwrap().row);
        core.reused(&root.key);
        assert!(core.lookup(&full, 0, 128).is_none());
        core.index.unpin(&child.key).unwrap();
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
        assert_eq!(hit.tier, Tier::Dram);
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
            matches!(&events[1], StoreEvent::Deferred { checkpoint, reason: StoreReason::Slabs } if *checkpoint == child.id)
        );
        assert!(
            matches!(&events[2], StoreEvent::Deferred { checkpoint, reason: StoreReason::Slabs } if *checkpoint == grandchild.id)
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

    /// Marks every resident record as announced, as the hasher would have.
    fn announce_all(core: &mut Core) {
        let keys: Vec<_> = core.index.nodes().map(|node| node.key).collect();
        for key in keys {
            core.announced.insert(key);
        }
    }

    #[test]
    fn headroom_is_clamped_separately_for_pages_and_rows() {
        let mut core = store_core(100, 4, false);
        for salt in 1..=3 {
            let h = history(600, salt);
            assert!(core.store(&published(&h, 128, None, 0, salt)).unwrap());
            finish(&mut core);
        }
        assert_eq!(core.headroom(), (5, 1));
        announce_all(&mut core);
        core.spill = true;
        assert!(core.retire_for(5, 2));
        assert_eq!(core.held.len(), 1);
        let h = history(600, 4);
        assert!(
            core.store(&published(&h, 128, None, 0, 4)).unwrap(),
            "free slabs remain usable during retirement"
        );
        core.largest_pages = 90;
        assert_eq!(core.headroom(), (10, 1));
    }

    #[test]
    fn progress_retires_before_a_record_larger_than_the_watermark_arrives() {
        use crate::store::TierStore;

        let geometry = [geometry(PT, &[("kv", PAGE)], ROW)];
        let (copier, _) = crate::testing::SimulatedGpu::shared(&geometry).unwrap();
        let pool = SlabPool::new(PAGE, 100, ROW, 16, true).unwrap();
        let mut store = MemoryStore::new(IDENTITY, pool, copier).unwrap();
        store.announce_to(Announcer::new(Arc::new(tokio::sync::Notify::new())));
        for salt in 1..=4 {
            let h = history(22 * PT, salt);
            let checkpoint = published(&h, 22 * PT, None, 0, salt);
            assert_eq!(store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while store.core.announced.len() != 4 {
            assert!(Instant::now() < deadline, "records were not announced");
            store.poll(&mut Vec::new()).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(store.core.headroom().0, 10);
        assert_eq!(store.pool().free_pages(), 12);
        store.set_spill(true);
        store.poll(&mut Vec::new()).unwrap();
        let retiring = store.take_retiring();
        assert_eq!(retiring.len(), 1);
        assert_eq!(retiring[0].tail_pages, 22);
        store.retired(retiring[0].key, PoolTier::Dram, Retirement::Moved);
        let h = history(22 * PT, 5);
        let next = published(&h, 22 * PT, None, 0, 5);
        assert_eq!(store.store(&next).unwrap(), StoreOutcome::Queued);
        store.close().unwrap();
    }

    #[test]
    fn a_spilling_store_holds_the_coldest_announced_leaf_until_its_move_ends() {
        let mut core = store_core(4, 4, false);
        let (one, two, three) = (history(600, 1), history(600, 2), history(600, 3));
        let cold = published(&one, 128, None, 0, 1);
        let warm = published(&two, 128, None, 10, 2);
        for checkpoint in [&cold, &warm] {
            assert!(core.store(checkpoint).unwrap());
            finish(&mut core);
        }
        announce_all(&mut core);
        core.spill = true;
        let wanted = published(&three, 128, None, 20, 3);

        // The pool is full: the coldest leaf is held, and nothing is evicted.
        assert!(!core.store(&wanted).unwrap());
        assert_eq!(core.held.len(), 1);
        let retiring = std::mem::take(&mut core.retiring);
        assert_eq!(retiring.len(), 1);
        assert_eq!(
            (retiring[0].key, retiring[0].tier, retiring[0].tail_pages),
            (cold.key, PoolTier::Dram, 2)
        );
        assert!(core.index.contains(&cold.key));

        // A second attempt finds the held leaf covers it: nothing more is chosen.
        assert!(!core.store(&wanted).unwrap());
        assert!(core.retiring.is_empty());

        // A failed move releases the leaf, which is offered again.
        core.retired(&cold.key, Retirement::Failed);
        assert!(core.index.contains(&cold.key) && core.held.is_empty());
        assert!(!core.store(&wanted).unwrap());
        assert_eq!(core.retiring.len(), 1);
        core.retiring.clear();

        // Once moved, its slabs are free for the new record.
        core.retired(&cold.key, Retirement::Moved);
        assert!(!core.index.contains(&cold.key));
        assert!(core.index.contains(&warm.key));
        assert!(core.store(&wanted).unwrap());
    }

    #[test]
    fn unavailable_manager_uses_durable_dram_lru() {
        let mut core = store_core(4, 4, false);
        let (one, two, three) = (history(600, 1), history(600, 2), history(600, 3));
        let cold = published(&one, 128, None, 0, 1);
        let warm = published(&two, 128, None, 10, 2);
        for r in [&cold, &warm] {
            assert!(core.store(r).unwrap());
            finish(&mut core);
        }
        announce_all(&mut core);
        core.spill = true;
        assert!(core.retire_for(2, 1));
        for r in std::mem::take(&mut core.retiring) {
            core.retired(&r.key, Retirement::Failed);
        }
        core.spill = false;
        assert!(core.index.contains(&cold.key));
        assert!(core.store(&published(&three, 128, None, 20, 3)).unwrap());
        assert!(!core.index.contains(&cold.key));
        assert!(core.index.contains(&warm.key));
        assert_eq!(core.stats().dropped[0], 1);
        assert_eq!(core.stats().held[0], 1);
    }

    #[test]
    fn records_not_yet_announced_are_evicted_in_place_as_before() {
        let mut core = store_core(4, 4, false);
        let (one, two, three) = (history(600, 1), history(600, 2), history(600, 3));
        let cold = published(&one, 128, None, 0, 1);
        let warm = published(&two, 128, None, 10, 2);
        for checkpoint in [&cold, &warm] {
            assert!(core.store(checkpoint).unwrap());
            finish(&mut core);
        }
        core.spill = true;
        // Nothing was announced, so nothing can be handed on.
        assert!(core.store(&published(&three, 128, None, 20, 3)).unwrap());
        assert!(core.retiring.is_empty() && core.held.is_empty());
        assert!(!core.index.contains(&cold.key));
    }

    #[test]
    fn a_moved_record_stays_until_its_export_pin_is_released() {
        let mut core = store_core(4, 4, false);
        let (one, two, three) = (history(600, 1), history(600, 2), history(600, 3));
        let cold = published(&one, 128, None, 0, 1);
        let warm = published(&two, 128, None, 10, 2);
        for checkpoint in [&cold, &warm] {
            assert!(core.store(checkpoint).unwrap());
            finish(&mut core);
        }
        announce_all(&mut core);
        core.spill = true;
        assert!(!core.store(&published(&three, 128, None, 20, 3)).unwrap());
        assert!(core.held.contains_key(&cold.key));
        // An export reads the slabs.
        core.index.pin(&cold.key).unwrap();
        core.retired(&cold.key, Retirement::Moved);
        assert!(core.index.contains(&cold.key));
        core.index.unpin(&cold.key).unwrap();
        core.finish_retired();
        assert!(!core.index.contains(&cold.key));
    }
    #[test]
    fn expired_memory_records_release_their_slabs() {
        let geometry = [crate::testing::geometry(64, &[("kv", PAGE)], ROW)];
        let (copier, _) = crate::testing::SimulatedGpu::shared(&geometry).unwrap();
        let mut store = MemoryStore::new(
            IDENTITY,
            SlabPool::new(PAGE, 2, ROW, 2, true).unwrap(),
            copier,
        )
        .unwrap();
        store.index_mut().max_age = std::time::Duration::from_secs(7200);
        let checkpoint = published(&history(64, 1), 64, None, 0, 1);
        store.store(&checkpoint).unwrap();
        assert_eq!((store.stats().rows_used, store.stats().rows_total), (1, 2));
        store
            .index_mut()
            .node_mut(&checkpoint.key)
            .unwrap()
            .last_used = std::time::SystemTime::UNIX_EPOCH;
        store.poll(&mut Vec::new()).unwrap();
        assert_eq!(store.stats().expired, [1, 0]);
        assert_eq!((store.stats().rows_used, store.stats().rows_total), (0, 2));
        assert_eq!(
            (store.pool().free_pages(), store.pool().free_rows()),
            (2, 2)
        );
        store.close().unwrap();
    }
}
