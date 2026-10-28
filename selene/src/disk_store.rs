//! Records retained as files, moved through registered host staging.
//!
//! [`DiskStore`] indexes the record files under one directory, accounts their
//! bytes against a limit, and evicts under the tree policy of the index: the
//! least recently used unpinned leaf is unlinked, and when every leaf is pinned
//! the least recently used single-child ancestor is compacted by truncating its
//! file at the row segment. Every byte movement goes through a [`DiskTransport`].
//! [`Staged`] streams records between GPU ids and files through a registered
//! staging pool, the shared transfer threads, and the disk thread, in windows of
//! at most [`WINDOW_PAGES`] pages with at most [`WINDOWS_IN_FLIGHT`] windows of
//! one record staged at a time. It also moves records between files and
//! caller-owned registered buffers retained by the composite's memory tier.
//! Nothing here waits: `store`, `fetch`, `truncate`, and `unlink`
//! enqueue, and `poll` advances every pipeline as far as free slabs and queue
//! space allow before reporting completions.

#[cfg(test)]
use crate::store::TierStore;

use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant, SystemTime},
};

use crate::levels::Rate;
use crate::{
    CheckpointId, Error, ExportRefusal, History, PageId, PrefixKey, Result, StateSlotId,
    index::NodeEvent,
    pool::{
        Announcer, FilePin, HostPin, IndexEntry, Releases, Tier as PoolTier, TierView,
        pins::{FileSpan, host_chain_kept},
    },
    shared::SliceSums,
};

use super::{
    CacheGeometry, HostHit, HostStore, INITIAL_STAGED_DISK_RATE, InboundTicket, JobId, Layout,
    PublishedCheckpoint, Retirement, Retiring, StoreEvent, StoreOutcome, StoreReason, StoreStats,
    Tier,
    copier::{self, Direction, PageCopier, SharedCopier},
    disk::{self, DiskThread, Operation},
    geometry::invalid,
    index::{Eviction, Index, Match, Node},
    record::{self, Header, Scanned},
    slab::{PageSlab, RowSlab, SlabPool, WINDOW_PAGES},
};

/// Windows of one record the staged transport keeps in staging at a time.
pub const WINDOWS_IN_FLIGHT: usize = crate::shared::WINDOW_CREDITS as usize;
/// Records per direction a disk store keeps in flight. Further `store` calls
/// are skipped and further `fetch` calls refused until one completes, so GPU
/// pins and ticket ids held for the store stay bounded.
pub const RECORDS_IN_FLIGHT: usize = 8;

/// A record file as the disk index sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskRecord {
    pub path: PathBuf,
    pub header: Header,
    /// File length: the row is present when it covers the row segment. Zero
    /// for the remnant of a failed write, which owns no file.
    pub bytes: u64,
}

/// Completion the transport reports to the disk store.
#[derive(Debug)]
pub enum TransportEvent {
    Stored {
        id: super::JobId,
        result: Result<u64>,
    },
    /// A failed store changed the destination on some ranks; its prior file is unusable.
    StoreInvalidated { id: super::JobId, error: Error },
    Fetched {
        id: super::JobId,
        result: Result<u64>,
    },
    Truncated {
        id: super::JobId,
        result: Result<()>,
    },
    Unlinked {
        id: super::JobId,
        result: Result<()>,
    },
}

/// Moves whole records between GPU ids and record files.
pub trait DiskTransport {
    /// Stream a published GPU checkpoint into a new record file at `temporary`,
    /// renamed to `path` on completion. The record's pages are the last
    /// `header.tail_pages` entries of the checkpoint's chain.
    fn store(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
    ) -> Result<()>;
    /// Stream the chain into the ticket's ids: for each record, the page
    /// records `first..first + count` and, for the last record, the row.
    fn fetch(
        &mut self,
        id: super::JobId,
        plan: Vec<FetchSegment>,
        ticket: &InboundTicket,
    ) -> Result<()>;
    /// Drop the row segment of a committed record.
    fn truncate(&mut self, id: super::JobId, path: PathBuf, length: u64) -> Result<()>;
    fn unlink(&mut self, id: super::JobId, path: PathBuf) -> Result<()>;
    /// Report completed jobs. `Err` means this transport can no longer report
    /// them: a job it accepted may still run, so the caller must stop rather
    /// than reuse the ids that job holds.
    fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    /// The staged transport behind this handle, for the composite store's
    /// transfers between files and its memory slabs.
    fn staged_mut(&mut self) -> Option<&mut Staged> {
        None
    }
}

/// One record's contribution to a fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchSegment {
    pub path: PathBuf,
    pub header: Header,
    pub first_page: u32,
    pub pages: u32,
    pub row: bool,
}

fn lock<'a>(
    transfers: &'a Mutex<dyn PageCopier + 'static>,
) -> MutexGuard<'a, dyn PageCopier + 'static> {
    transfers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Host addresses of one window's page and row records, with the staging
/// slabs behind them when the transport allocated them.
struct Buffers {
    pages: Vec<usize>,
    row: Option<usize>,
    slabs: Option<(Vec<PageSlab>, Option<RowSlab>)>,
}

/// The threads and pool every pipeline draws on, with the counters that keep
/// submissions within the queues' depths so a refusal is a failure, not backpressure.
struct Resources {
    /// Non-retaining buffers for device transfers; host-buffer-only users may omit them.
    pool: Option<SlabPool>,
    transfers: SharedCopier,
    disk: DiskThread,
    /// Disk jobs submitted whose replies have not been polled.
    disk_outstanding: usize,
    next_disk: u64,
}

impl Resources {
    fn disk_id(&mut self) -> JobId {
        let id = JobId(self.next_disk);
        self.next_disk += 1;
        id
    }

    fn disk_space(&self) -> bool {
        self.disk_outstanding < disk::QUEUE_DEPTH
    }

    fn submit_disk(&mut self, job: disk::Job) -> Result<()> {
        self.disk.submit(job)?;
        self.disk_outstanding += 1;
        Ok(())
    }

    fn transfer_space(&self) -> bool {
        lock(&self.transfers).in_flight() < copier::QUEUE_DEPTH
    }

    fn submit_transfer(&mut self, job: copier::Job) -> Result<()> {
        lock(&self.transfers).submit(job)
    }

    /// Staging slabs for `pages` page records and, when `row`, one row record;
    /// `None` leaves the pool unchanged when it cannot supply all of them.
    fn allocate(&mut self, pages: usize, row: bool) -> Option<Buffers> {
        let pool = self.pool.as_mut()?;
        let mut slabs = Vec::with_capacity(pages);
        for _ in 0..pages {
            match pool.allocate_page() {
                Some(slab) => slabs.push(slab),
                None => {
                    for slab in slabs {
                        pool.release_page(slab);
                    }
                    return None;
                }
            }
        }
        let row_slab = if row {
            match pool.allocate_row() {
                Some(slab) => Some(slab),
                None => {
                    for slab in slabs {
                        pool.release_page(slab);
                    }
                    return None;
                }
            }
        } else {
            None
        };
        Some(Buffers {
            pages: slabs.iter().map(|slab| pool.page_address(*slab)).collect(),
            row: row_slab.map(|slab| pool.row_address(slab)),
            slabs: Some((slabs, row_slab)),
        })
    }

    fn release(&mut self, buffers: &mut Buffers) {
        if let (Some(pool), Some((pages, row))) = (self.pool.as_mut(), buffers.slabs.take()) {
            for slab in pages {
                pool.release_page(slab);
            }
            if let Some(slab) = row {
                pool.release_row(slab);
            }
        }
    }
}

/// The store job a disk job belongs to.
#[derive(Clone, Copy)]
enum Owner {
    Outbound(JobId),
    Inbound(JobId, usize),
    Truncate(JobId),
    Unlink(JobId),
}

/// Where an outbound record's bytes come from.
enum Source {
    /// GPU ids and the row's state slot, staged through the pool.
    Device {
        pages: Vec<PageId>,
        slot: StateSlotId,
    },
    /// Caller-owned registered buffers already holding every record, and their checksums if known.
    Host {
        pages: Vec<usize>,
        row: usize,
        sums: Option<SliceSums>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TemporaryFile {
    Unsubmitted,
    Creating,
    Open,
    /// Never created, removed by a failed commit, or removed by an abort.
    Absent,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutPhase {
    Transferring,
    Transferred,
    Writing,
}

struct OutWindow {
    index: u32,
    first: u32,
    buffers: Buffers,
    phase: OutPhase,
    pages_submitted: bool,
    row_submitted: bool,
    /// Jobs that may still touch the buffers.
    referenced: usize,
}

struct OutboundRecord {
    disk: JobId,
    header: Header,
    temporary: PathBuf,
    path: PathBuf,
    source: Source,
    windows: u32,
    next_window: u32,
    active: Vec<OutWindow>,
    written: u32,
    file: TemporaryFile,
    commit_submitted: bool,
    abort_submitted: bool,
    /// Jobs submitted for this record whose replies are awaited.
    outstanding: usize,
    committed: Option<u64>,
    error: Option<Error>,
}

impl OutboundRecord {
    fn new(disk: JobId, header: Header, temporary: PathBuf, path: PathBuf, source: Source) -> Self {
        let windows = (header.tail_pages as usize).div_ceil(WINDOW_PAGES).max(1) as u32;
        Self {
            disk,
            header,
            temporary,
            path,
            source,
            windows,
            next_window: 0,
            active: Vec::with_capacity(WINDOWS_IN_FLIGHT),
            written: 0,
            file: TemporaryFile::Unsubmitted,
            commit_submitted: false,
            abort_submitted: false,
            outstanding: 0,
            committed: None,
            error: None,
        }
    }

    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    fn apply_disk(&mut self, reply: disk::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        match reply.operation {
            Operation::Create => match reply.result {
                Ok(()) => self.file = TemporaryFile::Open,
                Err(error) => {
                    self.file = TemporaryFile::Absent;
                    self.fail(error);
                }
            },
            Operation::WritePages | Operation::WriteRow => {
                let window = self.active.iter_mut().find(|window| {
                    if reply.operation == Operation::WritePages {
                        window.index == reply.window
                    } else {
                        window.buffers.row.is_some()
                    }
                });
                if let Some(window) = window {
                    window.referenced = window.referenced.saturating_sub(1);
                }
                if let Err(error) = reply.result {
                    self.fail(error);
                }
            }
            Operation::Commit => match reply.result {
                Ok(()) => self.committed = Some(reply.bytes),
                Err(error) => {
                    self.file = TemporaryFile::Absent;
                    self.fail(error);
                }
            },
            Operation::Abort => self.file = TemporaryFile::Absent,
            _ => {}
        }
    }

    fn apply_transfer(&mut self, reply: copier::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        let Some(window) = self
            .active
            .iter_mut()
            .find(|window| window.index == reply.window)
        else {
            return;
        };
        window.referenced = window.referenced.saturating_sub(1);
        match reply.result {
            Ok(()) => window.phase = OutPhase::Transferred,
            Err(error) => self.fail(error),
        }
    }

    /// Advance as far as slabs and queue space allow; `Some` when the record
    /// is finished and nothing references its buffers or its temporary file.
    fn pump(&mut self, id: JobId, resources: &mut Resources) -> Option<Result<u64>> {
        if let Some(bytes) = self.committed {
            return Some(Ok(bytes));
        }
        if self.file == TemporaryFile::Unsubmitted {
            if !resources.disk_space() {
                return None;
            }
            let job = disk::Job::Create {
                id: self.disk,
                temporary: self.temporary.clone(),
                header: self.header.clone(),
                sums: match &self.source {
                    Source::Host { sums, .. } => sums.clone(),
                    Source::Device { .. } => None,
                },
            };
            match resources.submit_disk(job) {
                Ok(()) => {
                    self.file = TemporaryFile::Creating;
                    self.outstanding += 1;
                }
                Err(error) => {
                    self.file = TemporaryFile::Absent;
                    self.fail(error);
                }
            }
        }
        if self.error.is_some() {
            return self.unwind(resources);
        }
        let mut index = 0;
        while index < self.active.len() {
            let window = &self.active[index];
            if window.phase == OutPhase::Writing && window.referenced == 0 {
                let mut window = self.active.remove(index);
                resources.release(&mut window.buffers);
                self.written += 1;
            } else {
                index += 1;
            }
        }
        while self.active.len() < WINDOWS_IN_FLIGHT && self.next_window < self.windows {
            match self.start_window(id, resources) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    self.fail(error);
                    return self.unwind(resources);
                }
            }
        }
        for index in 0..self.active.len() {
            if self.active[index].phase == OutPhase::Transferred
                && let Err(error) = self.submit_writes(index, resources)
            {
                self.fail(error);
                return self.unwind(resources);
            }
        }
        if self.written == self.windows
            && !self.commit_submitted
            && self.file == TemporaryFile::Open
            && resources.disk_space()
        {
            let job = disk::Job::Commit {
                id: self.disk,
                path: self.path.clone(),
                sync: false,
            };
            match resources.submit_disk(job) {
                Ok(()) => {
                    self.commit_submitted = true;
                    self.outstanding += 1;
                }
                Err(error) => {
                    self.fail(error);
                    return self.unwind(resources);
                }
            }
        }
        None
    }

    /// Stage the next window: allocate its slabs and copy it out of the GPU,
    /// or take the caller's buffers. `Ok(false)` holds the window until slabs
    /// or queue space free up.
    fn start_window(&mut self, id: JobId, resources: &mut Resources) -> Result<bool> {
        let index = self.next_window;
        let first = index as usize * WINDOW_PAGES;
        let count = (self.header.tail_pages as usize - first).min(WINDOW_PAGES);
        let last = index + 1 == self.windows;
        let (buffers, phase, referenced) = match &self.source {
            Source::Device { pages, slot } => {
                if !resources.transfer_space() {
                    return Ok(false);
                }
                let Some(mut buffers) = resources.allocate(count, last) else {
                    return Ok(false);
                };
                let job = copier::Job {
                    id,
                    window: index,
                    direction: Direction::Out,
                    pages: pages[first..first + count]
                        .iter()
                        .copied()
                        .zip(buffers.pages.iter().copied())
                        .collect(),
                    row: buffers.row.map(|address| (*slot, address)),
                };
                if let Err(error) = resources.submit_transfer(job) {
                    resources.release(&mut buffers);
                    return Err(error);
                }
                self.outstanding += 1;
                (buffers, OutPhase::Transferring, 1)
            }
            Source::Host { pages, row, .. } => (
                Buffers {
                    pages: pages[first..first + count].to_vec(),
                    row: last.then_some(*row),
                    slabs: None,
                },
                OutPhase::Transferred,
                0,
            ),
        };
        self.active.push(OutWindow {
            index,
            first: first as u32,
            buffers,
            phase,
            pages_submitted: false,
            row_submitted: false,
            referenced,
        });
        self.next_window += 1;
        Ok(true)
    }

    fn submit_writes(&mut self, index: usize, resources: &mut Resources) -> Result<()> {
        let window = &mut self.active[index];
        if !window.buffers.pages.is_empty() && !window.pages_submitted {
            if !resources.disk_space() {
                return Ok(());
            }
            resources.submit_disk(disk::Job::WritePages {
                id: self.disk,
                window: window.index,
                first: window.first,
                buffers: window.buffers.pages.clone(),
            })?;
            window.pages_submitted = true;
            window.referenced += 1;
            self.outstanding += 1;
        }
        if let Some(buffer) = window.buffers.row
            && !window.row_submitted
        {
            if !resources.disk_space() {
                return Ok(());
            }
            resources.submit_disk(disk::Job::WriteRow {
                id: self.disk,
                buffer,
            })?;
            window.row_submitted = true;
            window.referenced += 1;
            self.outstanding += 1;
        }
        if (window.buffers.pages.is_empty() || window.pages_submitted)
            && (window.buffers.row.is_none() || window.row_submitted)
        {
            window.phase = OutPhase::Writing;
        }
        Ok(())
    }

    /// Release every window no job references; once nothing is outstanding,
    /// abort the temporary file and report the failure after the abort replies.
    fn unwind(&mut self, resources: &mut Resources) -> Option<Result<u64>> {
        self.active.retain_mut(|window| {
            if window.referenced == 0 {
                resources.release(&mut window.buffers);
                false
            } else {
                true
            }
        });
        if self.outstanding > 0 || !self.active.is_empty() {
            return None;
        }
        match self.file {
            TemporaryFile::Open if !self.abort_submitted => {
                if !resources.disk_space() {
                    return None;
                }
                match resources.submit_disk(disk::Job::Abort { id: self.disk }) {
                    Ok(()) => {
                        self.abort_submitted = true;
                        self.outstanding += 1;
                        None
                    }
                    Err(_) => Some(Err(self.error.clone()?)),
                }
            }
            TemporaryFile::Absent => Some(Err(self.error.clone()?)),
            _ => None,
        }
    }
}

/// Where an inbound record's bytes go.
enum Destination {
    /// The ticket's GPU ids and state slot, through the pool.
    Device {
        pages: Vec<PageId>,
        slot: StateSlotId,
    },
    /// Caller-owned registered buffers.
    Host {
        pages: Vec<usize>,
        row: Option<usize>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FileState {
    Unopened,
    Opened,
    Closing,
    Closed,
}

struct Segment {
    disk: JobId,
    path: PathBuf,
    /// The indexed header the disk thread validates before reading payloads.
    expected: Header,
    state: FileState,
    /// Windows of this segment whose reads have not completed.
    reads_left: usize,
}

impl Segment {
    fn settled(&self) -> bool {
        matches!(self.state, FileState::Unopened | FileState::Closed)
    }
}

/// One window of a fetch: `count` page records from `first` of one segment,
/// landing at `offset` in the destination pages, plus the row on the last.
#[derive(Clone, Copy)]
struct WindowPlan {
    segment: usize,
    index: u32,
    first: u32,
    count: usize,
    offset: usize,
    row: bool,
}

/// Split segments of `(first page, page count, row)` into windows.
fn plan_windows(segments: impl IntoIterator<Item = (u32, usize, bool)>) -> Vec<WindowPlan> {
    let mut plans = Vec::new();
    let mut offset = 0;
    for (segment, (first, pages, row)) in segments.into_iter().enumerate() {
        let windows = if pages == 0 {
            usize::from(row)
        } else {
            pages.div_ceil(WINDOW_PAGES)
        };
        for window in 0..windows {
            let start = window * WINDOW_PAGES;
            plans.push(WindowPlan {
                segment,
                index: plans.len() as u32,
                first: first + start as u32,
                count: (pages - start).min(WINDOW_PAGES),
                offset: offset + start,
                row: row && window + 1 == windows,
            });
        }
        offset += pages;
    }
    plans
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InPhase {
    Reading,
    Read,
    Uploading,
}

struct InWindow {
    plan: WindowPlan,
    buffers: Buffers,
    phase: InPhase,
    pages_submitted: bool,
    row_submitted: bool,
    referenced: usize,
}

struct InboundRecord {
    segments: Vec<Segment>,
    destination: Destination,
    plans: Vec<WindowPlan>,
    next_window: usize,
    active: Vec<InWindow>,
    done: usize,
    outstanding: usize,
    bytes: u64,
    error: Option<Error>,
}

impl InboundRecord {
    fn new(segments: Vec<Segment>, destination: Destination, plans: Vec<WindowPlan>) -> Self {
        Self {
            segments,
            destination,
            plans,
            next_window: 0,
            active: Vec::with_capacity(WINDOWS_IN_FLIGHT),
            done: 0,
            outstanding: 0,
            bytes: 0,
            error: None,
        }
    }

    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    fn apply_disk(&mut self, segment: usize, reply: disk::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        let Some(file) = self.segments.get_mut(segment) else {
            return;
        };
        match reply.operation {
            Operation::Open => match reply.result {
                Ok(()) => {}
                Err(error) => {
                    file.state = FileState::Closed;
                    self.fail(error);
                }
            },
            Operation::ReadPages | Operation::ReadRow => {
                let window = self.active.iter_mut().find(|window| {
                    window.plan.segment == segment
                        && if reply.operation == Operation::ReadPages {
                            window.plan.index == reply.window
                        } else {
                            window.plan.row
                        }
                });
                let mut finished = false;
                if let Some(window) = window {
                    window.referenced = window.referenced.saturating_sub(1);
                    if window.phase == InPhase::Reading
                        && window.pages_submitted
                        && window.row_submitted
                        && window.referenced == 0
                    {
                        window.phase = InPhase::Read;
                        finished = true;
                    }
                }
                if finished {
                    file.reads_left = file.reads_left.saturating_sub(1);
                }
                match reply.result {
                    Ok(()) => self.bytes += reply.bytes,
                    Err(error) => self.fail(error),
                }
            }
            Operation::Close => file.state = FileState::Closed,
            _ => {}
        }
    }

    fn apply_transfer(&mut self, reply: copier::Reply) {
        self.outstanding = self.outstanding.saturating_sub(1);
        let Some(window) = self
            .active
            .iter_mut()
            .find(|window| window.plan.index == reply.window)
        else {
            return;
        };
        window.referenced = window.referenced.saturating_sub(1);
        if let Err(error) = reply.result {
            self.fail(error);
        }
    }

    fn pump(&mut self, id: JobId, resources: &mut Resources) -> Option<Result<u64>> {
        if self.error.is_some() {
            return self.unwind(resources);
        }
        let mut index = 0;
        while index < self.active.len() {
            let window = &self.active[index];
            let complete = match (&self.destination, window.phase) {
                (Destination::Device { .. }, InPhase::Uploading) => window.referenced == 0,
                (Destination::Host { .. }, InPhase::Read) => true,
                _ => false,
            };
            if complete {
                let mut window = self.active.remove(index);
                resources.release(&mut window.buffers);
                self.done += 1;
            } else {
                index += 1;
            }
        }
        while self.active.len() < WINDOWS_IN_FLIGHT && self.next_window < self.plans.len() {
            match self.start_window(resources) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    self.fail(error);
                    return self.unwind(resources);
                }
            }
        }
        for index in 0..self.active.len() {
            let outcome = match self.active[index].phase {
                InPhase::Reading => self.submit_reads(index, resources),
                InPhase::Read => self.submit_upload(id, index, resources),
                InPhase::Uploading => Ok(()),
            };
            if let Err(error) = outcome {
                self.fail(error);
                return self.unwind(resources);
            }
        }
        for segment in &mut self.segments {
            if segment.state == FileState::Opened
                && segment.reads_left == 0
                && resources.disk_space()
            {
                match resources.submit_disk(disk::Job::Close { id: segment.disk }) {
                    Ok(()) => {
                        segment.state = FileState::Closing;
                        self.outstanding += 1;
                    }
                    Err(error) => {
                        self.fail(error);
                        return self.unwind(resources);
                    }
                }
            }
        }
        if self.done == self.plans.len() && self.segments.iter().all(Segment::settled) {
            return Some(Ok(self.bytes));
        }
        None
    }

    fn start_window(&mut self, resources: &mut Resources) -> Result<bool> {
        let plan = self.plans[self.next_window];
        let segment = &mut self.segments[plan.segment];
        if segment.state == FileState::Unopened {
            if !resources.disk_space() {
                return Ok(false);
            }
            resources.submit_disk(disk::Job::Open {
                id: segment.disk,
                path: segment.path.clone(),
                expected: segment.expected.clone(),
            })?;
            segment.state = FileState::Opened;
            self.outstanding += 1;
        }
        let buffers = match &self.destination {
            Destination::Device { .. } => match resources.allocate(plan.count, plan.row) {
                Some(buffers) => buffers,
                None => return Ok(false),
            },
            Destination::Host { pages, row } => Buffers {
                pages: pages[plan.offset..plan.offset + plan.count].to_vec(),
                row: if plan.row { *row } else { None },
                slabs: None,
            },
        };
        self.active.push(InWindow {
            plan,
            buffers,
            phase: InPhase::Reading,
            pages_submitted: false,
            row_submitted: false,
            referenced: 0,
        });
        self.next_window += 1;
        Ok(true)
    }

    fn submit_reads(&mut self, index: usize, resources: &mut Resources) -> Result<()> {
        let window = &mut self.active[index];
        let disk = self.segments[window.plan.segment].disk;
        if !window.pages_submitted {
            if window.buffers.pages.is_empty() {
                window.pages_submitted = true;
            } else {
                if !resources.disk_space() {
                    return Ok(());
                }
                resources.submit_disk(disk::Job::ReadPages {
                    id: disk,
                    window: window.plan.index,
                    first: window.plan.first,
                    buffers: window.buffers.pages.clone(),
                })?;
                window.pages_submitted = true;
                window.referenced += 1;
                self.outstanding += 1;
            }
        }
        if !window.row_submitted {
            match window.buffers.row {
                None => window.row_submitted = true,
                Some(buffer) => {
                    if !resources.disk_space() {
                        return Ok(());
                    }
                    resources.submit_disk(disk::Job::ReadRow { id: disk, buffer })?;
                    window.row_submitted = true;
                    window.referenced += 1;
                    self.outstanding += 1;
                }
            }
        }
        Ok(())
    }

    fn submit_upload(&mut self, id: JobId, index: usize, resources: &mut Resources) -> Result<()> {
        let Destination::Device { pages, slot } = &self.destination else {
            return Ok(());
        };
        let window = &mut self.active[index];
        if !resources.transfer_space() {
            return Ok(());
        }
        let plan = window.plan;
        resources.submit_transfer(copier::Job {
            id,
            window: plan.index,
            direction: Direction::In,
            pages: pages[plan.offset..plan.offset + plan.count]
                .iter()
                .copied()
                .zip(window.buffers.pages.iter().copied())
                .collect(),
            row: window.buffers.row.map(|address| (*slot, address)),
        })?;
        window.phase = InPhase::Uploading;
        window.referenced += 1;
        self.outstanding += 1;
        Ok(())
    }

    /// Release every window no job references, close every opened file, and
    /// report the failure once every reply has arrived.
    fn unwind(&mut self, resources: &mut Resources) -> Option<Result<u64>> {
        self.active.retain_mut(|window| {
            if window.referenced == 0 {
                resources.release(&mut window.buffers);
                false
            } else {
                true
            }
        });
        for segment in &mut self.segments {
            if segment.state == FileState::Opened && resources.disk_space() {
                match resources.submit_disk(disk::Job::Close { id: segment.disk }) {
                    Ok(()) => {
                        segment.state = FileState::Closing;
                        self.outstanding += 1;
                    }
                    Err(_) => segment.state = FileState::Closed,
                }
            }
        }
        if self.outstanding == 0
            && self.active.is_empty()
            && self.segments.iter().all(Segment::settled)
        {
            return Some(Err(self.error.clone()?));
        }
        None
    }
}

/// The staged transport: record files move through a non-retaining staging
/// pool and the shared transfer threads. It also moves records between files
/// and caller-owned registered buffers for the composite store.
///
/// Without a pool the transport never submits transfer jobs and never polls
/// the shared transfer threads, whose replies then belong to the memory store.
pub struct Staged {
    resources: Resources,
    outbound: HashMap<JobId, OutboundRecord>,
    inbound: HashMap<JobId, InboundRecord>,
    /// Disk job ids to the store job each belongs to.
    owners: HashMap<JobId, Owner>,
    /// Truncate and unlink jobs awaiting disk queue space.
    maintenance: VecDeque<(JobId, disk::Job)>,
    events: Vec<TransportEvent>,
    closed: bool,
}

impl Staged {
    pub fn new(pool: Option<SlabPool>, transfers: SharedCopier, disk: DiskThread) -> Self {
        Self {
            resources: Resources {
                pool,
                transfers,
                disk,
                disk_outstanding: 0,
                next_disk: 1,
            },
            outbound: HashMap::new(),
            inbound: HashMap::new(),
            owners: HashMap::new(),
            maintenance: VecDeque::new(),
            events: Vec::new(),
            closed: false,
        }
    }

    pub fn shares_transfers(&self, transfers: &SharedCopier) -> bool {
        Arc::ptr_eq(&self.resources.transfers, transfers)
    }

    pub fn pool(&self) -> Option<&SlabPool> {
        self.resources.pool.as_ref()
    }

    /// Records of each direction whose pipelines have not finished.
    pub fn in_flight(&self) -> (usize, usize) {
        (self.outbound.len(), self.inbound.len())
    }

    /// Write a record whose page and row bytes already sit in caller-owned
    /// registered buffers (host addresses in record order).
    #[allow(clippy::too_many_arguments)]
    pub fn write_from(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        pages: Vec<usize>,
        row: Option<usize>,
        sums: Option<SliceSums>,
    ) -> Result<()> {
        self.accept(id)?;
        if pages.len() != header.tail_pages as usize {
            return Err(invalid(format!(
                "record {} has {} tail pages; {} page buffers were supplied",
                record::file_name(&header.key),
                header.tail_pages,
                pages.len()
            )));
        }
        let Some(row) = row else {
            return Err(invalid("a record cannot be written without its row"));
        };
        let source = Source::Host { pages, row, sums };
        self.start_outbound(id, header, temporary, path, source);
        Ok(())
    }

    /// Read a record's page records `first..first + buffers.len()` and row into
    /// caller-owned registered buffers.
    pub fn read_into(
        &mut self,
        id: super::JobId,
        path: PathBuf,
        expected: Header,
        first_page: u32,
        pages: Vec<usize>,
        row: Option<usize>,
    ) -> Result<()> {
        self.accept(id)?;
        if pages.is_empty() && row.is_none() {
            return Err(invalid(
                "a record read must name page buffers or the row buffer",
            ));
        }
        let plans = plan_windows([(first_page, pages.len(), row.is_some())]);
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Inbound(id, 0));
        let segment = Segment {
            disk,
            path,
            expected,
            state: FileState::Unopened,
            reads_left: plans.len(),
        };
        self.inbound.insert(
            id,
            InboundRecord::new(vec![segment], Destination::Host { pages, row }, plans),
        );
        self.pump_inbound(id);
        Ok(())
    }

    /// Stream the GPU ids `pages` and the row in `slot` into a new record file,
    /// as `store` does for the record's pages of a published checkpoint.
    pub fn store_pages(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        pages: Vec<PageId>,
        slot: StateSlotId,
    ) -> Result<()> {
        self.accept(id)?;
        if self.resources.pool.is_none() {
            return Err(invalid(
                "the staged transport has no staging pool; records move only through caller-owned buffers",
            ));
        }
        if pages.len() != header.tail_pages as usize {
            return Err(invalid(format!(
                "record {} holds {} tail pages; {} GPU pages were supplied",
                record::file_name(&header.key),
                header.tail_pages,
                pages.len()
            )));
        }
        self.start_outbound(id, header, temporary, path, Source::Device { pages, slot });
        Ok(())
    }

    fn accept(&self, id: JobId) -> Result<()> {
        if self.closed {
            return Err(invalid("the staged transport is closed"));
        }
        if self.outbound.contains_key(&id) || self.inbound.contains_key(&id) {
            return Err(invalid(format!(
                "disk transport job {} is already in flight",
                id.0
            )));
        }
        Ok(())
    }

    fn start_outbound(
        &mut self,
        id: JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        source: Source,
    ) {
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Outbound(id));
        self.outbound.insert(
            id,
            OutboundRecord::new(disk, header, temporary, path, source),
        );
        self.pump_outbound(id);
    }

    fn pump_outbound(&mut self, id: JobId) {
        let Some(record) = self.outbound.get_mut(&id) else {
            return;
        };
        if let Some(result) = record.pump(id, &mut self.resources) {
            let record = self.outbound.remove(&id).unwrap();
            self.owners.remove(&record.disk);
            self.events.push(TransportEvent::Stored { id, result });
        }
    }

    fn pump_inbound(&mut self, id: JobId) {
        let Some(record) = self.inbound.get_mut(&id) else {
            return;
        };
        if let Some(result) = record.pump(id, &mut self.resources) {
            let record = self.inbound.remove(&id).unwrap();
            for segment in &record.segments {
                self.owners.remove(&segment.disk);
            }
            self.events.push(TransportEvent::Fetched { id, result });
        }
    }

    fn pump_all(&mut self) {
        while let Some((disk, _)) = self.maintenance.front()
            && self.resources.disk_space()
        {
            let disk = *disk;
            let (_, job) = self.maintenance.pop_front().unwrap();
            if let Err(error) = self.resources.submit_disk(job) {
                self.finish_maintenance(disk, Err(error));
            }
        }
        let mut ids: Vec<JobId> = self.outbound.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            self.pump_outbound(id);
        }
        let mut ids: Vec<JobId> = self.inbound.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            self.pump_inbound(id);
        }
    }

    fn finish_maintenance(&mut self, disk: JobId, result: Result<()>) {
        match self.owners.remove(&disk) {
            Some(Owner::Truncate(id)) => self.events.push(TransportEvent::Truncated { id, result }),
            Some(Owner::Unlink(id)) => self.events.push(TransportEvent::Unlinked { id, result }),
            _ => {}
        }
    }

    /// Absorb every reply and advance every pipeline.
    fn step(&mut self) {
        let mut replies = Vec::new();
        self.resources.disk.poll(&mut replies);
        for reply in replies {
            self.resources.disk_outstanding = self.resources.disk_outstanding.saturating_sub(1);
            match self.owners.get(&reply.id).copied() {
                Some(Owner::Outbound(id)) => {
                    if let Some(record) = self.outbound.get_mut(&id) {
                        record.apply_disk(reply);
                    }
                }
                Some(Owner::Inbound(id, segment)) => {
                    if let Some(record) = self.inbound.get_mut(&id) {
                        record.apply_disk(segment, reply);
                    }
                }
                Some(Owner::Truncate(_) | Owner::Unlink(_)) => {
                    self.finish_maintenance(reply.id, reply.result);
                }
                None => {}
            }
        }
        if self.resources.pool.is_some() {
            let mut replies = Vec::new();
            lock(&self.resources.transfers).poll(&mut replies);
            for reply in replies {
                match reply.direction {
                    Direction::Out => {
                        if let Some(record) = self.outbound.get_mut(&reply.id) {
                            record.apply_transfer(reply);
                        }
                    }
                    Direction::In => {
                        if let Some(record) = self.inbound.get_mut(&reply.id) {
                            record.apply_transfer(reply);
                        }
                    }
                }
            }
        }
        self.pump_all();
    }

    fn busy(&self) -> bool {
        !self.outbound.is_empty() || !self.inbound.is_empty() || !self.owners.is_empty()
    }
}

impl DiskTransport for Staged {
    fn store(
        &mut self,
        id: super::JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
    ) -> Result<()> {
        let tail = header.tail_pages as usize;
        let chain = checkpoint.chain_pages.len();
        if tail > chain {
            return Err(invalid(format!(
                "record {} holds {tail} tail pages but the published chain has only {chain}",
                record::file_name(&header.key)
            )));
        }
        self.store_pages(
            id,
            header,
            temporary,
            path,
            checkpoint.chain_pages[chain - tail..].to_vec(),
            checkpoint.slot,
        )
    }

    fn fetch(
        &mut self,
        id: super::JobId,
        plan: Vec<FetchSegment>,
        ticket: &InboundTicket,
    ) -> Result<()> {
        self.accept(id)?;
        if self.resources.pool.is_none() {
            return Err(invalid(
                "the staged transport has no staging pool; records move only through caller-owned buffers",
            ));
        }
        let pages: usize = plan.iter().map(|segment| segment.pages as usize).sum();
        if pages != ticket.pages.len() {
            return Err(invalid(format!(
                "fetch plan covers {pages} pages; the ticket reserved {}",
                ticket.pages.len()
            )));
        }
        let rows = plan.iter().filter(|segment| segment.row).count();
        if rows != 1 || !plan.last().is_some_and(|segment| segment.row) {
            return Err(invalid(
                "fetch plan must take the row from its last record only",
            ));
        }
        let plans = plan_windows(
            plan.iter()
                .map(|segment| (segment.first_page, segment.pages as usize, segment.row)),
        );
        let mut segments = Vec::with_capacity(plan.len());
        for (index, segment) in plan.into_iter().enumerate() {
            let disk = self.resources.disk_id();
            self.owners.insert(disk, Owner::Inbound(id, index));
            segments.push(Segment {
                disk,
                path: segment.path,
                expected: segment.header,
                state: FileState::Unopened,
                reads_left: plans.iter().filter(|plan| plan.segment == index).count(),
            });
        }
        let destination = Destination::Device {
            pages: ticket.pages.clone(),
            slot: ticket.slot,
        };
        self.inbound
            .insert(id, InboundRecord::new(segments, destination, plans));
        self.pump_inbound(id);
        Ok(())
    }

    fn truncate(&mut self, id: super::JobId, path: PathBuf, length: u64) -> Result<()> {
        if self.closed {
            return Err(invalid("the staged transport is closed"));
        }
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Truncate(id));
        self.maintenance.push_back((
            disk,
            disk::Job::Truncate {
                id: disk,
                path,
                length,
            },
        ));
        Ok(())
    }

    fn unlink(&mut self, id: super::JobId, path: PathBuf) -> Result<()> {
        if self.closed {
            return Err(invalid("the staged transport is closed"));
        }
        let disk = self.resources.disk_id();
        self.owners.insert(disk, Owner::Unlink(id));
        self.maintenance
            .push_back((disk, disk::Job::Unlink { id: disk, path }));
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
        if !self.closed {
            self.step();
        }
        events.append(&mut self.events);
        Ok(())
    }

    /// Finish outstanding pipelines within [`copier::CLOSE_TIMEOUT`], then
    /// stop the disk thread and, when this transport owns the staging pool,
    /// the transfer threads. Completions reached before the deadline remain
    /// available through `poll`.
    fn close(&mut self) -> Result<()> {
        let deadline = Instant::now() + copier::CLOSE_TIMEOUT;
        loop {
            self.step();
            if !self.busy() || Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.closed = true;
        let abandoned = self.owners.len();
        let disk = self.resources.disk.close();
        let transfers = match self.resources.pool {
            Some(_) => lock(&self.resources.transfers).close(),
            None => Ok(()),
        };
        disk?;
        transfers?;
        if abandoned > 0 {
            return Err(invalid(format!(
                "{abandoned} disk transport jobs did not complete within {:?}",
                copier::CLOSE_TIMEOUT
            )));
        }
        Ok(())
    }

    fn staged_mut(&mut self) -> Option<&mut Staged> {
        Some(self)
    }
}

struct Outbound {
    checkpoint: CheckpointId,
    key: PrefixKey,
    started: Instant,
}

struct Inbound {
    key: PrefixKey,
    nodes: Vec<PrefixKey>,
    /// Tokens the restore saves over prefill.
    tokens: u64,
    started: Instant,
}

/// Bytes charged until the unlink or truncate succeeds.
struct Reclaim {
    /// The compacted record, for a truncate; `None` for an unlink.
    key: Option<PrefixKey>,
    bytes: u64,
}

pub struct DiskStore {
    index: Index<DiskRecord>,
    identity: [u8; 32],
    fingerprint: [u8; 16],
    limit: u64,
    budget: Option<Arc<super::input::DiskBudget>>,
    used: u64,
    transport: Box<dyn DiskTransport>,
    stats: StoreStats,
    next_job: u64,
    geometry: Vec<CacheGeometry>,
    layout: Layout,
    outbound: HashMap<JobId, Outbound>,
    inbound: HashMap<JobId, Inbound>,
    /// Unlink and truncate jobs whose replies are awaited.
    maintenance: HashMap<JobId, Reclaim>,
    replacements: HashMap<PrefixKey, DiskRecord>,
    /// Keys whose files were unlinked since the last `take_evicted`.
    evicted: Vec<PrefixKey>,
    outbound_rate: Rate,
    inbound_rate: Rate,
    announce: Option<Announce>,
    /// Eviction hands records to the pool's manager instead of unlinking them.
    spill: bool,
    pub(crate) preserve_committed: bool,
    /// Records chosen for the manager and not yet taken by the pool store.
    retiring: Vec<Retiring>,
    /// Records held in place until their move ends, with the bytes freed when
    /// they go.
    held: HashMap<PrefixKey, u64>,
    /// Records whose moves ended and that leave once nothing reads them.
    finishing: Vec<PrefixKey>,
    largest: u64,
    maintenance_progress: u64,
    pub directory_lock: record::DirectoryLock,
}

/// What a disk store keeps to announce its records and serve them as exports.
struct Announce {
    view: TierView,
    releases: Releases,
}

impl DiskStore {
    /// Open a store over the locked directory, already scanned by
    /// [`super::record::scan`], whose records are inserted parents first and
    /// take their access order from file age, oldest first.
    pub fn new(
        directory_lock: record::DirectoryLock,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        limit: u64,
        geometry: Vec<CacheGeometry>,
        scanned: Vec<Scanned>,
        transport: Box<dyn DiskTransport>,
    ) -> Result<Self> {
        let layout = Layout::new(&geometry)?;
        let mut index = Index::new(layout.page_tokens);
        let mut used = 0u64;
        for record in &scanned {
            index.insert(
                record.header.key,
                record.header.parent,
                record.header.bucket,
                record.header.tail_pages as usize,
                record.row,
                true,
                DiskRecord {
                    path: record.path.clone(),
                    header: record.header.clone(),
                    bytes: record.bytes,
                },
            )?;
            used += record.bytes;
        }
        let mut by_age: Vec<&Scanned> = scanned.iter().collect();
        by_age.sort_by_key(|record| (record.modified, record.header.key.consumed_tokens));
        for (tick, record) in by_age.into_iter().enumerate() {
            index.access_at(&record.header.key, tick as u64 + 1);
            let node = index.node_mut(&record.header.key).unwrap();
            node.last_used = record.modified;
        }
        Ok(Self {
            index,
            identity,
            fingerprint,
            limit,
            budget: None,
            used,
            transport,
            stats: StoreStats::default(),
            next_job: 1,
            geometry,
            layout,
            outbound: HashMap::new(),
            inbound: HashMap::new(),
            maintenance: HashMap::new(),
            replacements: HashMap::new(),
            evicted: Vec::new(),
            outbound_rate: Rate::new(INITIAL_STAGED_DISK_RATE),
            inbound_rate: Rate::new(INITIAL_STAGED_DISK_RATE),
            announce: None,
            spill: false,
            preserve_committed: false,
            retiring: Vec::new(),
            held: HashMap::new(),
            finishing: Vec::new(),
            largest: scanned.iter().map(|r| r.bytes).max().unwrap_or(0),
            maintenance_progress: 0,
            directory_lock,
        })
    }

    /// Announce this store's records through `announcer` and serve them as
    /// exports. Recovered holdings are announced before this returns.
    pub fn announce_records(&mut self, announcer: Announcer) {
        if self.announce.is_some() {
            return;
        }
        let view = TierView::new(announcer, PoolTier::Nvme, &self.layout);
        self.index.observe(view.sink());
        view.sink()
            .lock()
            .unwrap()
            .extend(self.index.ready_notices());
        self.announce = Some(Announce {
            view,
            releases: Releases::default(),
        });
        self.flush_announcements();
    }

    /// What this tier announced for a complete record, as the next tier below
    /// memory: the entry a record spilled here is announced as.
    pub fn announced_entry(&self, key: &PrefixKey) -> Option<IndexEntry> {
        let node = self.index.node(key).filter(|node| node.ready)?;
        if !self.announce.as_ref()?.view.contains(key) {
            return None;
        }
        Some(IndexEntry {
            key: *key,
            parent: node.parent,
            bucket: node.bucket,
            tail_pages: node.tail_pages as u32,
            row: node.row,
            tier: PoolTier::Nvme,
            bytes: node.tail_pages as u64 * self.layout.page_record_bytes
                + if node.row {
                    self.layout.row_record_bytes
                } else {
                    0
                },
        })
    }

    /// Release finished exports' pins and announce what changed.
    pub fn flush_announcements(&mut self) {
        let Some(a) = self.announce.as_mut() else {
            return;
        };
        for key in a.releases.drain() {
            let _ = self.index.unpin(&key);
            self.stats.progress += 1;
        }
        for notice in a.view.drain() {
            match notice.event {
                NodeEvent::Ready | NodeEvent::RowDropped => a.view.announce(&notice, notice.parent),
                NodeEvent::Gone => a.view.retire(&notice.key),
            }
        }
        let _ = self.finish_retired();
    }

    /// Pin the chain of complete record files ending at `key` for an export
    /// from `begin_tokens`. The files' own checksums are what is served.
    pub fn pin_export_files(
        &mut self,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        let Some(a) = self.announce.as_ref() else {
            return Err(ExportRefusal::NotFound);
        };
        if !a.view.contains(&key) {
            return Err(ExportRefusal::NotFound);
        }
        let page_tokens = self.layout.page_tokens;
        let start_page = begin_tokens / page_tokens;
        let mut keys = Vec::new();
        let mut nodes = Vec::new();
        let mut spans = Vec::new();
        let mut cursor = key;
        loop {
            let node = self
                .index
                .node(&cursor)
                .filter(|node| node.ready)
                .ok_or(ExportRefusal::ChainGap)?;
            if !a.view.contains(&cursor) {
                return Err(ExportRefusal::ChainGap);
            }
            if want_row && cursor == key && !node.row {
                return Err(ExportRefusal::ChainGap);
            }
            keys.push(cursor);
            nodes.push((cursor, node.tail_pages));
            spans.push(FileSpan::new(
                node.location.path.clone(),
                node.location.header.clone(),
            ));
            let start =
                (cursor.consumed_tokens / page_tokens).saturating_sub(node.tail_pages as u64);
            if start <= start_page {
                break;
            }
            cursor = node.parent.ok_or(ExportRefusal::ChainGap)?;
        }
        keys.reverse();
        nodes.reverse();
        spans.reverse();
        let (chain, kept) = host_chain_kept(page_tokens, begin_tokens, want_row, &nodes);
        if chain.total_pages != (key.consumed_tokens / page_tokens).saturating_sub(start_page) {
            return Err(ExportRefusal::ChainGap);
        }
        // Windows name records by their place in the chain, which leaves out nodes with no page.
        let mut spans: Vec<Option<FileSpan>> = spans.into_iter().map(Some).collect();
        let spans: Vec<FileSpan> = kept.iter().filter_map(|&i| spans[i].take()).collect();
        for (position, key) in keys.iter().enumerate() {
            if self.index.pin(key).is_err() {
                for pinned in &keys[..position] {
                    let _ = self.index.unpin(pinned);
                }
                return Err(ExportRefusal::NotFound);
            }
        }
        Ok(Box::new(FilePin {
            chain,
            spans,
            page_bytes: self.layout.page_record_bytes,
            row_bytes: self.layout.row_record_bytes,
            keys,
            releases: a.releases.clone(),
        }))
    }

    pub fn share_budget(&mut self, budget: Arc<super::input::DiskBudget>) {
        budget.set_checkpoint(self.used);
        self.budget = Some(budget);
    }

    fn reclaim_inputs(&mut self) {
        let pressure = self.budget.as_ref().map_or(0, |b| b.checkpoint_pressure());
        if pressure != 0 {
            let _ = self.make_room(pressure);
        }
    }

    fn sync_budget(&self) {
        if let Some(budget) = &self.budget {
            budget.set_checkpoint(self.used.saturating_add(self.pending_bytes()));
        }
    }

    #[cfg(test)]
    pub(crate) fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    pub fn index(&self) -> &Index<DiskRecord> {
        &self.index
    }

    pub fn index_mut(&mut self) -> &mut Index<DiskRecord> {
        &mut self.index
    }

    pub fn directory(&self) -> &Path {
        self.directory_lock.path()
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn fingerprint(&self) -> [u8; 16] {
        self.fingerprint
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn geometry(&self) -> &[CacheGeometry] {
        &self.geometry
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn transport_mut(&mut self) -> &mut dyn DiskTransport {
        self.transport.as_mut()
    }

    pub fn pending_store_key(&self, id: JobId) -> Option<PrefixKey> {
        self.outbound.get(&id).map(|job| job.key)
    }

    pub fn next_job(&mut self) -> super::JobId {
        let id = super::JobId(self.next_job);
        self.next_job += 1;
        id
    }

    pub fn record(&self, key: &PrefixKey) -> Option<&DiskRecord> {
        self.index.node(key).map(|node| &node.location)
    }

    /// Keys whose files were unlinked since the previous call.
    pub fn take_evicted(&mut self) -> Vec<PrefixKey> {
        std::mem::take(&mut self.evicted)
    }

    /// Drain the transport's completions without applying them, for a
    /// composite store that drives the transport directly.
    pub fn poll_transport(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
        self.reclaim_inputs();
        self.transport.poll(events)
    }

    /// The record shape for a published checkpoint: a child of its parent when
    /// the parent is indexed, otherwise a root record holding the whole chain.
    /// The remnant of a failed write, which has neither a row nor readable
    /// pages, counts as absent.
    pub fn plan_record(
        &self,
        checkpoint: &PublishedCheckpoint,
    ) -> (Option<PrefixKey>, Vec<PageId>) {
        if let Some(node) = self.index.node(&checkpoint.key) {
            let parent = node.parent;
            let first = parent.map_or(0, |p| p.consumed_tokens / self.layout.page_tokens) as usize;
            return (parent, checkpoint.chain_pages[first..].to_vec());
        }
        let parent = checkpoint.parent.filter(|parent| {
            self.index
                .node(parent)
                .is_some_and(|node| node.ready || node.row)
        });
        let tail = match parent {
            Some(_) => checkpoint.tail().to_vec(),
            None => checkpoint.chain_pages.clone(),
        };
        (parent, tail)
    }

    pub fn store_status(&self, checkpoint: &PublishedCheckpoint) -> Result<Option<StoreOutcome>> {
        if checkpoint.tail_pages > checkpoint.chain_pages.len() {
            return Err(invalid("published checkpoint tail exceeds its chain"));
        }
        if let Some(node) = self.index.node(&checkpoint.key) {
            if node.ready && node.row && self.index.complete(&checkpoint.key) {
                return Ok(Some(StoreOutcome::Present));
            }
            if !node.ready
                || node.row
                || node.pinned()
                || self
                    .maintenance
                    .values()
                    .any(|job| job.key == Some(checkpoint.key))
            {
                return Ok(Some(StoreOutcome::Retry(StoreReason::Inflight)));
            }
        }
        if self.outbound.len() >= RECORDS_IN_FLIGHT {
            return Ok(Some(StoreOutcome::Retry(StoreReason::Queue)));
        }
        let (parent, tail) = self.plan_record(checkpoint);
        let header = Header::new(
            &self.geometry,
            self.identity,
            self.fingerprint,
            checkpoint.key,
            parent,
            checkpoint.bucket,
            tail.len() as u32,
            SystemTime::now(),
        )?;
        let mut footprint = header.length_with_row();
        let mut cursor = parent;
        while let Some(key) = cursor {
            let node = &self.index[&key];
            footprint = footprint.saturating_add(if Some(key) != parent {
                node.location.header.row_segment
            } else if node.location.bytes == 0 {
                node.location.header.length_with_row()
            } else {
                node.location.bytes
            });
            cursor = node.parent;
        }
        Ok((footprint > self.limit).then_some(StoreOutcome::Oversize))
    }

    /// With `reserve=false`, index only uncharged metadata for a DRAM copy.
    /// Make room for a record of `tail_pages`, index it as writing, pin it,
    /// and return its header and file names; `None` when evicting every
    /// unpinned record would still not bring it under the limit. The key must
    /// be absent and the parent, when named, indexed with a row or readable
    /// pages. The parent is pinned while room is made, so eviction neither
    /// unlinks nor compacts it, and removing its other children never cascades
    /// into it, before the record names it.
    pub fn reserve_record(
        &mut self,
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: usize,
        reserve: bool,
    ) -> Result<Option<(Header, PathBuf, PathBuf)>> {
        let parked = self.index.node(&key).is_some_and(|n| n.location.bytes == 0);
        let replacement = if let Some(node) = self.index.node(&key).filter(|_| !parked) {
            if node.row {
                return Err(invalid("host record key is already indexed"));
            }
            if !node.ready
                || node.pinned()
                || self.maintenance.values().any(|job| job.key == Some(key))
            {
                return Ok(None);
            }
            if node.parent != parent || node.tail_pages != tail_pages {
                return Err(invalid(
                    "replacement must preserve the stored parent and page tail",
                ));
            }
            Some(node.location.clone())
        } else {
            None
        };
        if let Some(parent) = &parent
            && !self
                .index
                .node(parent)
                .is_some_and(|node| node.ready || node.row)
        {
            return Err(invalid(
                "host record parent is not indexed or is the remnant of a failed write",
            ));
        }
        let tail = u32::try_from(tail_pages)
            .map_err(|_| invalid(format!("a record cannot hold {tail_pages} tail pages")))?;
        let header = Header::new(
            &self.geometry,
            self.identity,
            self.fingerprint,
            key,
            parent,
            bucket,
            tail,
            SystemTime::now(),
        )?;
        let bytes = header.length_with_row();
        if let Some(parent) = &parent {
            self.index.pin(parent)?;
        }
        if replacement.is_some() {
            self.index.pin(&key)?;
        }
        let room = if reserve {
            self.make_room(bytes)
        } else {
            Ok(true)
        };
        if replacement.is_some() {
            self.index.unpin(&key)?;
        }
        if let Some(parent) = &parent {
            self.index.unpin(parent)?;
        }
        if !room? {
            return Ok(None);
        }
        if reserve
            && let Some(budget) = &self.budget
            && !budget.reserve_checkpoint(bytes)
        {
            return Ok(None);
        }
        let bytes = if reserve { bytes } else { 0 };
        let path = self.directory().join(record::file_name(&key));
        let temporary = self.directory().join(record::temporary_name(&key));
        let location = DiskRecord {
            path: path.clone(),
            header: header.clone(),
            bytes,
        };
        if parked {
            self.index.node_mut(&key).unwrap().location = location;
        } else if let Some(old) = replacement {
            self.replacements.insert(key, old);
            self.index.node_mut(&key).unwrap().location = location;
            self.index.set_ready(&key, false)?;
        } else {
            self.index
                .insert(key, parent, bucket, tail_pages, true, false, location)?;
        }
        if !parked {
            self.index.pin(&key)?;
        }
        self.used += bytes;
        self.largest = self.largest.max(bytes);
        self.stats.progress += 1;
        Ok(Some((header, temporary, path)))
    }

    /// Finish a reserved record: a successful write makes it ready; a failed
    /// one returns its bytes to the budget and drops it from the index.
    pub fn commit_record(&mut self, key: PrefixKey, result: Result<u64>) -> Result<()> {
        self.finish_record(key, result, true)
    }

    fn finish_record(
        &mut self,
        key: PrefixKey,
        result: Result<u64>,
        prior_readable: bool,
    ) -> Result<()> {
        self.stats.progress += 1;
        if let Some(old) = self.replacements.remove(&key) {
            let node = self.index.node_mut(&key).unwrap();
            let ready = if result.is_ok() {
                self.used -= old.bytes;
                node.row = true;
                self.stats.row_refills += 1;
                self.stats.stored += 1;
                self.stats.bytes_out += result.as_ref().copied().unwrap();
                self.evicted.push(key);
                true
            } else {
                self.stats.store_failed += 1;
                if prior_readable {
                    self.used -= node.location.bytes;
                    node.location = old;
                    true
                } else {
                    // A failed cleanup can leave a complete replacement on one rank.
                    self.used -= old.bytes;
                    node.row = false;
                    false
                }
            };
            self.index.set_ready(&key, ready)?;
            self.index.unpin(&key)?;
            if !prior_readable {
                self.evicted.push(key);
                if self.index.node(&key).unwrap().children() == 0 {
                    for removed in self.index.remove(&key)? {
                        self.unlink(removed)?;
                    }
                }
            }
            self.sync_budget();
            return Ok(());
        }
        match result {
            Ok(bytes) => {
                if !self.index.contains(&key) {
                    return Err(invalid("cannot commit an unindexed host record"));
                }
                self.index.set_ready(&key, true)?;
                self.index.unpin(&key)?;
                self.stats.stored += 1;
                self.stats.bytes_out += bytes;
            }
            Err(_) => {
                self.stats.store_failed += 1;
                self.discard(&key)?;
            }
        }
        Ok(())
    }

    /// Evict until `bytes` more fit under the limit; false when nothing
    /// evictable remains, and false before evicting anything when `bytes`
    /// alone exceed the limit.
    fn make_room(&mut self, bytes: u64) -> Result<bool> {
        if bytes > self.limit {
            return Ok(false);
        }
        while self.used.saturating_add(bytes)
            > self
                .budget
                .as_ref()
                .map_or(self.limit, |b| b.checkpoint_limit())
                .min(self.limit)
        {
            if self.preserve_committed {
                return Ok(false);
            }
            if self.spill && self.retire_for(bytes) {
                return Ok(false);
            }
            match self.index.evict(true, |_| true) {
                Some(Eviction::Leaves(nodes)) => {
                    self.stats.dropped[3] += nodes.len() as u64;
                    for node in nodes {
                        self.unlink(node)?;
                    }
                }
                Some(Eviction::Compacted(key)) => {
                    self.stats.dropped[3] += 1;
                    self.compact(&key)?;
                }
                None => {
                    if let Some(budget) = &self.budget {
                        budget.request_input_reclaim(bytes);
                    }
                    return Ok(false);
                }
            }
        }
        Ok(self.budget.is_none()
            || self
                .used
                .saturating_add(self.pending_bytes())
                .saturating_add(bytes)
                <= self
                    .budget
                    .as_ref()
                    .unwrap()
                    .checkpoint_limit()
                    .min(self.limit))
    }

    fn pending_bytes(&self) -> u64 {
        self.maintenance.values().map(|reclaim| reclaim.bytes).sum()
    }

    pub(crate) fn maintain(&mut self) {
        let progress = self.stats().progress;
        if !self.preserve_committed && self.spill && self.maintenance_progress != progress {
            self.maintenance_progress = progress;
            self.retire_for(
                crate::store::headroom(self.limit, self.largest)
                    .saturating_add(self.pending_bytes()),
            );
        }
    }

    /// Hold bounded cold leaves for handover until they cover the demand.
    /// A full tier waits while any such records remain held.
    fn retire_for(&mut self, bytes: u64) -> bool {
        let limit = self
            .budget
            .as_ref()
            .map_or(self.limit, |b| b.checkpoint_limit())
            .min(self.limit);
        let mut freed: u64 = self.held.values().sum();
        for _ in 0..crate::pool::cascade::MOVES_MAX {
            if self.used.saturating_sub(freed).saturating_add(bytes) <= limit
                || self.held.len() >= crate::pool::cascade::MOVES_MAX
            {
                break;
            }
            let announce = self.announce.as_ref();
            let Some(key) = self.index.candidate(false, |node| {
                announce.is_some_and(|a| a.view.contains(&node.key))
            }) else {
                break;
            };
            let Some(node) = self.index.node(&key) else {
                break;
            };
            let retiring = Retiring {
                tier: PoolTier::Nvme,
                key,
                parent: node.parent,
                bucket: node.bucket,
                tail_pages: node.tail_pages,
                row: node.row,
            };
            let node_bytes = node.location.bytes;
            if self.index.pin(&key).is_err() {
                break;
            }
            freed += node_bytes;
            self.held.insert(key, node_bytes);
            self.retiring.push(retiring);
        }
        !self.held.is_empty()
    }

    /// The move of a held record ended: unlink it, or let it be offered again.
    fn retired(&mut self, key: &PrefixKey, how: Retirement) -> Result<()> {
        if self.held.remove(key).is_none() {
            return Ok(());
        }
        self.index.unpin(key)?;
        self.stats.held[1] += u64::from(how == Retirement::Failed);
        self.stats.progress += 1;
        if how != Retirement::Failed {
            self.finishing.push(*key);
            self.finish_retired()?;
        }
        Ok(())
    }

    /// Unlink the records whose moves ended, once no export pin still reads
    /// them. A record that gained a child meanwhile stays.
    fn finish_retired(&mut self) -> Result<()> {
        let mut finishing = std::mem::take(&mut self.finishing);
        let mut result = Ok(());
        finishing.retain(|key| {
            if self.index.node(key).is_some_and(|node| node.pinned()) {
                return true;
            }
            if let Ok(removed) = self.index.remove(key) {
                for node in removed {
                    if let Err(error) = self.unlink(node) {
                        result = Err(error);
                    }
                }
                self.stats.progress += 1;
            }
            false
        });
        self.finishing = finishing;
        result
    }

    /// Return a removed node's bytes to the budget and unlink its file.
    fn unlink(&mut self, node: Node<DiskRecord>) -> Result<()> {
        self.used = self.used.saturating_sub(node.location.bytes);
        if node.location.bytes == 0 {
            return Ok(());
        }
        let id = self.next_job();
        self.maintenance.insert(
            id,
            Reclaim {
                key: None,
                bytes: node.location.bytes,
            },
        );
        self.sync_budget();
        if let Err(error) = self.transport.unlink(id, node.location.path) {
            self.apply(
                TransportEvent::Unlinked {
                    id,
                    result: Err(error.clone()),
                },
                &mut Vec::new(),
            );
            return Err(error);
        }
        self.evicted.push(node.key);
        Ok(())
    }

    pub(crate) fn expire(&mut self, key: PrefixKey) -> Result<()> {
        for node in self.index.remove(&key)? {
            if node.location.bytes == 0 {
                self.evicted.push(node.key);
            }
            // Unready records are parked memory copies in a composite.
            self.stats.expired[usize::from(node.ready)] += 1;
            self.unlink(node)?;
        }
        Ok(())
    }

    /// Drop a compacted node's row segment from its file and the budget.
    fn compact(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .index
            .node_mut(key)
            .ok_or_else(|| invalid("cannot compact an unindexed host record"))?;
        let length = node.location.header.row_segment;
        let freed = node.location.bytes.saturating_sub(length);
        node.location.bytes = length;
        let path = node.location.path.clone();
        self.used = self.used.saturating_sub(freed);
        let id = self.next_job();
        self.maintenance.insert(
            id,
            Reclaim {
                key: Some(*key),
                bytes: freed,
            },
        );
        self.sync_budget();
        if let Err(error) = self.transport.truncate(id, path, length) {
            self.apply(
                TransportEvent::Truncated {
                    id,
                    result: Err(error.clone()),
                },
                &mut Vec::new(),
            );
            return Err(error);
        }
        Ok(())
    }

    /// Drop a record whose write failed. It leaves the index unless children
    /// already name it as their parent; then it stays as a remnant without a
    /// row or a file until the last child is removed.
    fn discard(&mut self, key: &PrefixKey) -> Result<()> {
        self.index.unpin(key)?;
        let node = self
            .index
            .node_mut(key)
            .ok_or_else(|| invalid("cannot discard an unindexed host record"))?;
        let bytes = std::mem::take(&mut node.location.bytes);
        self.used = self.used.saturating_sub(bytes);
        if node.children() == 0 {
            for removed in self.index.remove(key)? {
                self.unlink(removed)?;
            }
        } else {
            self.index.drop_row(key);
        }
        self.sync_budget();
        Ok(())
    }

    /// Plan the segments that restore a looked-up chain: the first record
    /// contributes its tail after the pages the GPU holds, every other record
    /// its whole tail, and the last record the row.
    pub fn plan_fetch(&self, chain: &Match) -> Result<Vec<FetchSegment>> {
        let last = chain
            .nodes
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid("host record chain is empty"))?;
        let mut plan = Vec::with_capacity(chain.nodes.len());
        for (position, key) in chain.nodes.iter().enumerate() {
            let node = self
                .index
                .node(key)
                .ok_or_else(|| invalid("host record chain names an unindexed record"))?;
            if !node.ready {
                return Err(invalid(
                    "host record chain names a record whose write has not completed",
                ));
            }
            let skip = if position == 0 { chain.skip_pages } else { 0 };
            let pages = node.tail_pages.checked_sub(skip).ok_or_else(|| {
                invalid("host record chain skips more pages than its first record holds")
            })?;
            let row = position == last;
            if row && !node.row {
                return Err(invalid("host record chain ends at a record without a row"));
            }
            plan.push(FetchSegment {
                path: node.location.path.clone(),
                header: node.location.header.clone(),
                first_page: skip as u32,
                pages: pages as u32,
                row,
            });
        }
        Ok(plan)
    }

    /// A hit for `chain` priced at this store's measured inbound rate.
    pub fn hit(&self, chain: Match) -> HostHit {
        let bytes = self.layout.record_bytes(chain.pages as u64);
        HostHit {
            key: chain.key,
            consumed_tokens: chain.consumed_tokens,
            resident_tokens: chain.resident_tokens,
            pages: chain.pages,
            bytes,
            tier: Tier::Nvme,
            estimated_seconds: bytes as f64 / self.inbound_rate.0.max(1.0),
            chain,
        }
    }

    /// Apply one transport completion to the index, counters, and rates.
    pub fn apply(&mut self, event: TransportEvent, events: &mut Vec<StoreEvent>) {
        self.stats.progress += 1;
        match event {
            TransportEvent::Stored { id, result } => {
                self.finish_outbound(id, result, true, events);
            }
            TransportEvent::StoreInvalidated { id, error } => {
                self.finish_outbound(id, Err(error), false, events);
            }
            TransportEvent::Fetched { id, result } => {
                let Some(job) = self.inbound.remove(&id) else {
                    return;
                };
                for key in &job.nodes {
                    let _ = self.index.unpin(key);
                }
                match &result {
                    Ok(bytes) => {
                        for key in job.nodes.iter().rev().skip(1).rev() {
                            self.index.access(key);
                        }
                        if let Some(key) = job.nodes.last() {
                            self.reused(*key);
                        }
                        self.stats.fetched += 1;
                        self.stats.bytes_in += bytes;
                        self.stats.reused_tokens += job.tokens;
                        self.inbound_rate.observe(*bytes, job.started.elapsed());
                    }
                    Err(_) => {
                        self.stats.fetch_failed += 1;
                        // A record that could not be restored is not offered
                        // again. Its pages stay in the index for descendants
                        // that hold it as an ancestor; only its row, which
                        // makes it a lookup match, is dropped.
                        if let Some(endpoint) = job.nodes.last() {
                            self.index.drop_row(endpoint);
                        }
                    }
                }
                events.push(StoreEvent::Fetched {
                    key: job.key,
                    result,
                });
            }
            TransportEvent::Truncated { id, result } | TransportEvent::Unlinked { id, result } => {
                let Some(reclaim) = self.maintenance.remove(&id) else {
                    return;
                };
                let Reclaim { key, bytes } = reclaim;
                if result.is_ok() {
                    self.sync_budget();
                    return;
                }
                // The file keeps the bytes, so they stay charged against the
                // limit until the next startup scan reconciles the directory.
                // A compacted record regains its full length; its row stays
                // unmatchable. An unlinked record is already out of the index.
                if let Some(node) = key.and_then(|key| self.index.node_mut(&key))
                    && node.location.bytes == node.location.header.row_segment
                {
                    node.location.bytes += bytes;
                }
                self.used = self.used.saturating_add(bytes);
                self.sync_budget();
            }
        }
    }

    fn finish_outbound(
        &mut self,
        id: JobId,
        result: Result<u64>,
        prior_readable: bool,
        events: &mut Vec<StoreEvent>,
    ) {
        let Some(job) = self.outbound.remove(&id) else {
            return;
        };
        if let Ok(bytes) = &result {
            self.outbound_rate.observe(*bytes, job.started.elapsed());
        }
        let pressure = result.as_ref().err().is_some_and(Error::no_space);
        let result = match self.finish_record(job.key, result.clone(), prior_readable) {
            Ok(()) => result,
            Err(error) => Err(error),
        };
        self.stats.dropped[5] += u64::from(result.is_err() && !pressure);
        self.stats.held[1] += u64::from(pressure);
        events.push(if pressure {
            StoreEvent::Deferred {
                checkpoint: job.checkpoint,
                reason: StoreReason::Budget,
            }
        } else {
            StoreEvent::Stored {
                checkpoint: job.checkpoint,
                key: job.key,
                result,
            }
        });
    }
}

impl HostStore for DiskStore {
    fn reused(&mut self, key: PrefixKey) {
        self.index.touch(&key);
    }

    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
        self.stats.lookups += 1;
        let chain = self
            .index
            .lookup_below(history, resident, before, |node| node.ready)?;
        self.stats.hits += 1;
        Some(self.hit(chain))
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        if let Some(outcome) = self.store_status(checkpoint)? {
            self.stats.store_skipped += 1;
            if outcome == StoreOutcome::Present {
                self.index.touch_ancestors(&checkpoint.key);
            }
            self.stats.dropped[4] += u64::from(outcome == StoreOutcome::Oversize);
            return Ok(outcome);
        }
        let (parent, tail) = self.plan_record(checkpoint);
        let Some((header, temporary, path)) =
            self.reserve_record(checkpoint.key, parent, checkpoint.bucket, tail.len(), true)?
        else {
            self.stats.store_skipped += 1;
            return Ok(StoreOutcome::Retry(StoreReason::Budget));
        };
        let id = self.next_job();
        if let Err(error) = self
            .transport
            .store(id, header, temporary, path, checkpoint)
        {
            self.commit_record(checkpoint.key, Err(error.clone()))?;
            return Err(error);
        }
        self.outbound.insert(
            id,
            Outbound {
                checkpoint: checkpoint.id,
                key: checkpoint.key,
                started: Instant::now(),
            },
        );
        Ok(StoreOutcome::Queued)
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        if hit.key != ticket.key {
            return Err(invalid(
                "inbound ticket names a different key than the host hit",
            ));
        }
        if self.inbound.len() >= RECORDS_IN_FLIGHT {
            return Err(invalid(format!(
                "disk store already has {RECORDS_IN_FLIGHT} inbound records in flight"
            )));
        }
        let chain = hit.chain;
        let plan = self.plan_fetch(&chain)?;
        let pages: usize = plan.iter().map(|segment| segment.pages as usize).sum();
        if pages != ticket.pages.len() || pages != chain.pages {
            return Err(invalid(format!(
                "host record chain supplies {pages} pages; the hit covers {} and the ticket reserved {}",
                chain.pages,
                ticket.pages.len()
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
        if let Err(error) = self.transport.fetch(id, plan, &ticket) {
            for key in &chain.nodes {
                let _ = self.index.unpin(key);
            }
            return Err(error);
        }
        self.inbound.insert(
            id,
            Inbound {
                key: hit.key,
                nodes: chain.nodes,
                tokens: chain.consumed_tokens.saturating_sub(chain.resident_tokens),
                started: Instant::now(),
            },
        );
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        for key in self.index.expired(|node| !node.pinned()) {
            self.expire(key)?;
        }
        self.reclaim_inputs();
        let mut completed = Vec::new();
        let polled = self.transport.poll(&mut completed);
        for event in completed {
            self.apply(event, events);
        }
        self.flush_announcements();
        self.maintain();
        polled
    }

    fn stats(&self) -> StoreStats {
        StoreStats {
            outbound_in_flight: self.outbound.len(),
            inbound_in_flight: self.inbound.len(),
            disk_records: self.index.len(),
            disk_bytes: self.used,
            progress: self.stats.progress
                + self.budget.as_ref().map_or(0, |budget| budget.progress()),
            outbound_rate: self.outbound_rate.0,
            inbound_rate: self.inbound_rate.0,
            ..self.stats
        }
    }

    fn close(&mut self) -> Result<()> {
        let closed = self.transport.close();
        let mut completed = Vec::new();
        let _ = self.transport.poll(&mut completed);
        let mut events = Vec::new();
        for event in completed {
            self.apply(event, &mut events);
        }
        closed
    }
}

impl crate::store::TierStore for DiskStore {
    fn announce_to(&mut self, announcer: Announcer) {
        self.announce_records(announcer);
    }
    fn pin_export(
        &mut self,
        tier: PoolTier,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        if tier != PoolTier::Nvme {
            return Err(ExportRefusal::NotFound);
        }
        self.pin_export_files(key, begin_tokens, want_row)
    }
    fn set_spill(&mut self, enabled: bool) {
        let enabled = enabled && self.announce.is_some();
        self.stats.progress += u64::from(self.spill != enabled);
        self.spill = enabled;
    }
    fn take_retiring(&mut self) -> Vec<Retiring> {
        std::mem::take(&mut self.retiring)
    }
    fn retired(&mut self, key: PrefixKey, tier: PoolTier, how: Retirement) {
        if tier == PoolTier::Nvme {
            let _ = DiskStore::retired(self, &key, how);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Ids, checkpoint, pattern, test_directory};
    use crate::testing::{StoppedCopier, TestHistory as PrefixHistory};
    use crate::{
        slab::{STAGING_PAGE_SLABS, STAGING_ROW_SLABS},
        testing::geometry as rank,
    };
    use std::{
        cell::RefCell,
        fs,
        io::{Read as _, Seek, SeekFrom, Write as _},
        path::Path,
        rc::Rc,
        time::UNIX_EPOCH,
    };

    const IDENTITY: [u8; 32] = [0x5a; 32];
    const FINGERPRINT: [u8; 16] = [0xa5; 16];
    const PAGE_TOKENS: u64 = 64;
    /// Page record bytes of the test geometry: two 8 KiB planes.
    const PAGE_RECORD: u64 = 16384;
    /// Row record bytes of the test geometry: one 4 KiB row.
    const ROW_RECORD: u64 = 4096;
    const CREATED: Duration = Duration::from_secs(1_700_000_000);

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

    fn header(
        geometry: &[CacheGeometry],
        history: &PrefixHistory,
        consumed: u64,
        parent: Option<u64>,
    ) -> Header {
        let tail = consumed / PAGE_TOKENS - parent.map_or(0, |parent| parent / PAGE_TOKENS);
        Header::new(
            geometry,
            IDENTITY,
            FINGERPRINT,
            history.key_at(consumed).unwrap(),
            parent.map(|parent| history.key_at(parent).unwrap()),
            history.bucket_at(consumed).unwrap(),
            tail as u32,
            UNIX_EPOCH + CREATED,
        )
        .unwrap()
    }

    fn scanned(directory: &Path, header: Header, age: u64, row: bool) -> Scanned {
        let bytes = if row {
            header.length_with_row()
        } else {
            header.row_segment
        };
        Scanned {
            path: directory.join(record::file_name(&header.key)),
            header,
            row,
            bytes,
            modified: SystemTime::now() - Duration::from_secs(50u64.saturating_sub(age)),
        }
    }

    fn pages(range: std::ops::Range<u32>) -> Vec<PageId> {
        range.map(PageId).collect()
    }

    #[derive(Debug)]
    enum Call {
        Store {
            id: JobId,
            header: Header,
            temporary: PathBuf,
            path: PathBuf,
            pages: Vec<PageId>,
            slot: StateSlotId,
        },
        Fetch {
            id: JobId,
            plan: Vec<FetchSegment>,
            ticket: InboundTicket,
        },
        Truncate {
            id: JobId,
            path: PathBuf,
            length: u64,
        },
        Unlink {
            id: JobId,
            path: PathBuf,
        },
    }

    #[derive(Default)]
    struct Log {
        calls: Vec<Call>,
        events: Vec<TransportEvent>,
        closed: bool,
    }

    /// A transport that records what the store asks of it and replies with
    /// whatever events the test queues.
    struct Recorder(Rc<RefCell<Log>>);

    impl Drop for Recorder {
        fn drop(&mut self) {
            let _ = fs::remove_dir(directory());
        }
    }

    impl DiskTransport for Recorder {
        fn store(
            &mut self,
            id: JobId,
            header: Header,
            temporary: PathBuf,
            path: PathBuf,
            checkpoint: &PublishedCheckpoint,
        ) -> Result<()> {
            let tail = header.tail_pages as usize;
            let pages = checkpoint.chain_pages[checkpoint.chain_pages.len() - tail..].to_vec();
            self.0.borrow_mut().calls.push(Call::Store {
                id,
                header,
                temporary,
                path,
                pages,
                slot: checkpoint.slot,
            });
            Ok(())
        }

        fn fetch(
            &mut self,
            id: JobId,
            plan: Vec<FetchSegment>,
            ticket: &InboundTicket,
        ) -> Result<()> {
            self.0.borrow_mut().calls.push(Call::Fetch {
                id,
                plan,
                ticket: ticket.clone(),
            });
            Ok(())
        }

        fn truncate(&mut self, id: JobId, path: PathBuf, length: u64) -> Result<()> {
            self.0
                .borrow_mut()
                .calls
                .push(Call::Truncate { id, path, length });
            Ok(())
        }

        fn unlink(&mut self, id: JobId, path: PathBuf) -> Result<()> {
            self.0.borrow_mut().calls.push(Call::Unlink { id, path });
            Ok(())
        }

        fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
            events.append(&mut self.0.borrow_mut().events);
            Ok(())
        }

        fn close(&mut self) -> Result<()> {
            self.0.borrow_mut().closed = true;
            Ok(())
        }
    }

    fn directory() -> PathBuf {
        std::env::temp_dir().join(format!(
            "flare-disk-store-mock-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn store_with(scanned: Vec<Scanned>, limit: u64) -> (DiskStore, Rc<RefCell<Log>>) {
        let log = Rc::new(RefCell::new(Log::default()));
        let store = DiskStore::new(
            crate::testing::directory_lock(&directory()).unwrap(),
            IDENTITY,
            FINGERPRINT,
            limit,
            geometry(),
            scanned,
            Box::new(Recorder(log.clone())),
        )
        .unwrap();
        (store, log)
    }

    #[test]
    fn partial_row_replacement_failure_invalidates_ancestry_until_reclaimed() {
        for has_child in [false, true] {
            let full = history(1000);
            let root = header(&geometry(), &full, 128, None);
            let child = header(&geometry(), &full, 256, Some(128));
            let mut records = vec![scanned(&directory(), root.clone(), 0, false)];
            if has_child {
                records.push(scanned(&directory(), child.clone(), 1, true));
            }
            let (mut store, log) = store_with(records, 1 << 30);
            let original = store.used();
            let mut ids = Ids::new();
            let publication =
                checkpoint(&mut ids, &full, 128, None, pages(0..2), 2, StateSlotId(1));
            assert_eq!(store.store(&publication).unwrap(), StoreOutcome::Queued);
            let events = complete(
                &log,
                TransportEvent::StoreInvalidated {
                    id: JobId(1),
                    error: invalid("rank 1: disk full"),
                },
                &mut store,
            );
            assert!(matches!(
                &events[..],
                [StoreEvent::Stored { result: Err(_), .. }]
            ));
            assert!(store.lookup(&full, 0, u64::MAX).is_none());
            assert_eq!(store.stats().store_failed, 1);
            assert_eq!(store.stats().row_refills, 0);
            assert_eq!(store.stats().outbound_in_flight, 0);
            if has_child {
                let node = store.index.node(&root.key).unwrap();
                assert!(!node.ready && !node.row && !node.pinned());
                assert_eq!(node.children(), 1);
                assert_eq!(store.used(), original + ROW_RECORD);
                assert_eq!(store.index.node(&child.key).unwrap().parent, Some(root.key));
                // A GPU-resident ancestor still supplies a valid prefix for the child's tail.
                assert_eq!(store.lookup(&full, 128, u64::MAX).unwrap().key, child.key);
                assert!(store.make_room(store.limit()).unwrap());
            }
            assert!(store.index.is_empty());
            assert_eq!(store.used(), 0);
            let unlinks: Vec<_> = log
                .borrow()
                .calls
                .iter()
                .filter_map(|call| match call {
                    Call::Unlink { id, path } => Some((*id, path.clone())),
                    _ => None,
                })
                .collect();
            assert_eq!(unlinks.len(), if has_child { 2 } else { 1 });
            for (id, _) in unlinks {
                complete(
                    &log,
                    TransportEvent::Unlinked { id, result: Ok(()) },
                    &mut store,
                );
            }
            assert!(store.maintenance.is_empty());
            assert_eq!(store.used(), 0);
        }
    }

    #[test]
    fn compacted_row_replacement_preserves_links_and_rolls_back_failure() {
        for success in [false, true] {
            let full = history(1000);
            let root = header(&geometry(), &full, 128, None);
            let compacted = header(&geometry(), &full, 512, Some(128));
            let child = header(&geometry(), &full, 768, Some(512));
            let records = vec![
                scanned(&directory(), root.clone(), 0, true),
                scanned(&directory(), compacted.clone(), 1, false),
                scanned(&directory(), child.clone(), 2, true),
            ];
            let (mut store, log) = store_with(records, 1 << 30);
            let original = store.used();
            let mut ids = Ids::new();
            // The GPU has a different parent: the stored parent determines the rewrite tail.
            let publication =
                checkpoint(&mut ids, &full, 512, None, pages(0..8), 8, StateSlotId(3));
            assert_eq!(store.store(&publication).unwrap(), StoreOutcome::Queued);
            let node = store.index.node(&compacted.key).unwrap();
            assert!(!node.ready && !node.row && node.pinned());
            assert_eq!(node.parent, Some(root.key));
            assert_eq!(node.children(), 1);
            assert_eq!(store.used(), original + compacted.length_with_row());
            match &log.borrow().calls[0] {
                Call::Store {
                    header,
                    pages: copied,
                    ..
                } => {
                    assert_eq!(header.parent, Some(root.key));
                    assert_eq!(copied, &pages(2..8));
                }
                other => panic!("unexpected call {other:?}"),
            }
            store.apply(
                TransportEvent::Stored {
                    id: JobId(1),
                    result: if success {
                        Ok(compacted.length_with_row())
                    } else {
                        Err(invalid("write failed"))
                    },
                },
                &mut Vec::new(),
            );
            let node = store.index.node(&compacted.key).unwrap();
            assert!(node.ready && !node.pinned());
            assert_eq!(node.row, success);
            assert_eq!(node.children(), 1);
            assert_eq!(
                store.index.node(&child.key).unwrap().parent,
                Some(compacted.key)
            );
            assert_eq!(
                store.used(),
                original + if success { ROW_RECORD } else { 0 }
            );
            assert_eq!(store.stats().row_refills, u64::from(success));
            assert!(store.lookup(&full, 0, u64::MAX).is_some());
        }
    }

    #[test]
    fn disk_store_keeps_exclusive_directory_ownership_until_drop() {
        let (mut store, _) = store_with(Vec::new(), 1 << 30);
        assert!(crate::testing::directory_lock(store.directory()).is_err());
        store.close().unwrap();
        assert!(crate::testing::directory_lock(store.directory()).is_err());
        let path = store.directory().to_owned();
        drop(store);
        let reopened = crate::testing::directory_lock(&path).unwrap();
        drop(reopened);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn feasible_ancestor_row_compaction_is_not_oversize() {
        let full = history(192);
        let root = header(&geometry(), &full, 64, None);
        let parent = header(&geometry(), &full, 128, Some(64));
        let leaf = header(&geometry(), &full, 192, Some(128));
        let limit = root.row_segment + parent.length_with_row() + leaf.length_with_row();
        let (mut store, _) = store_with(
            vec![
                scanned(&directory(), root.clone(), 0, true),
                scanned(&directory(), parent.clone(), 1, true),
            ],
            limit,
        );
        let record = checkpoint(
            &mut Ids::new(),
            &full,
            192,
            Some(128),
            pages(0..3),
            1,
            StateSlotId(0),
        );
        assert_eq!(store.store(&record).unwrap(), StoreOutcome::Queued);
        assert!(!store.index.node(&root.key).unwrap().row);
        assert!(store.index.node(&parent.key).unwrap().row);
    }

    #[test]
    fn filesystem_full_defers_a_disk_only_copy_after_drain() {
        let full = history(64);
        let record = checkpoint(
            &mut Ids::new(),
            &full,
            64,
            None,
            vec![PageId(0)],
            1,
            StateSlotId(0),
        );
        let (mut store, _) = store_with(Vec::new(), 1 << 30);
        assert_eq!(store.store(&record).unwrap(), StoreOutcome::Queued);
        let id = *store.outbound.keys().next().unwrap();
        let mut events = Vec::new();
        store.apply(
            TransportEvent::Stored {
                id,
                result: Err(Error::caused_by(
                    "disk full",
                    std::io::Error::from(std::io::ErrorKind::StorageFull),
                )),
            },
            &mut events,
        );
        assert!(
            matches!(&events[..], [StoreEvent::Deferred { checkpoint, reason: StoreReason::Budget }] if *checkpoint == record.id)
        );
        assert!(store.outbound.is_empty());
        assert_eq!(store.stats().dropped, [0; 6]);
        assert_eq!(store.stats().held[1], 1);
        assert!(!store.index.contains(&record.key));
    }

    #[test]
    fn a_full_local_tier_keeps_every_record_until_one_is_needed() {
        let full = history(1000);
        let headers: Vec<_> = (1..=3)
            .map(|n| header(&geometry(), &full, n * 64, None))
            .collect();
        let limit: u64 = headers.iter().map(Header::length_with_row).sum();
        let records = headers
            .iter()
            .enumerate()
            .map(|(i, h)| scanned(&directory(), h.clone(), i as u64, true))
            .collect();
        let (mut store, _log) = store_with(records, limit);
        let _announcer = announced(&mut store);
        // Local mode never sets spill: no proactive demotion, however full the tier.
        store.maintain();
        assert!(store.take_retiring().is_empty());
        assert!(store.maintenance.is_empty());
        assert_eq!(store.stats().disk_records, 3);
        // With the pool ready the same tier does demote ahead of need.
        store.set_spill(true);
        store.maintain();
        assert!(!store.take_retiring().is_empty());
    }

    #[test]
    fn pending_unlinks_are_included_in_proactive_headroom() {
        let full = history(1000);
        let headers: Vec<_> = (1..=3)
            .map(|n| header(&geometry(), &full, n * 64, None))
            .collect();
        let limit: u64 = headers.iter().map(Header::length_with_row).sum();
        let records = headers
            .iter()
            .enumerate()
            .map(|(i, h)| scanned(&directory(), h.clone(), i as u64, true))
            .collect();
        let (mut store, log) = store_with(records, limit);
        let budget = super::super::input::DiskBudget::new(limit, limit);
        store.share_budget(budget);
        let _announcer = announced(&mut store);
        store.set_spill(true);
        store.maintain();
        let first = store.take_retiring();
        assert!(!first.is_empty());
        for r in first {
            store.retired(&r.key, Retirement::Moved).unwrap();
        }
        let pending = store.pending_bytes();
        assert!(pending > 0);
        store.maintain();
        let next = store.take_retiring();
        assert!(!next.is_empty(), "pending maintenance advances retirement");
        for r in next {
            store.retired(&r.key, Retirement::Moved).unwrap();
        }
        let id = *store.maintenance.keys().next().unwrap();
        complete(
            &log,
            TransportEvent::Unlinked { id, result: Ok(()) },
            &mut store,
        );
        let candidate = checkpoint(
            &mut Ids::new(),
            &full,
            64,
            None,
            vec![PageId(0)],
            1,
            StateSlotId(0),
        );
        let key = PrefixKey {
            digest: [93; 16],
            ..candidate.key
        };
        assert!(store.pending_bytes() > 0);
        assert!(
            store
                .reserve_record(key, None, candidate.bucket, 1, true)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn shared_disk_budget_keeps_maintenance_bytes_until_success() {
        let history = history(700);
        let header = header(&geometry(), &history, 700, None);
        let bytes = header.length_with_row();
        let (mut store, log) =
            store_with(vec![scanned(&directory(), header.clone(), 0, true)], bytes);
        let budget = super::super::input::DiskBudget::new(bytes, bytes);
        store.share_budget(budget.clone());
        let nodes = store.index.remove(&header.key).unwrap();
        for node in nodes {
            store.unlink(node).unwrap();
        }
        assert_eq!(store.used(), 0);
        assert!(!budget.reserve_checkpoint(1));
        let id = *store.maintenance.keys().next().unwrap();
        complete(
            &log,
            TransportEvent::Unlinked {
                id,
                result: Err(invalid("unlink refused")),
            },
            &mut store,
        );
        assert_eq!(store.used(), bytes);
        assert!(!budget.reserve_checkpoint(1));
        drop(store);

        let (mut store, log) =
            store_with(vec![scanned(&directory(), header.clone(), 0, true)], bytes);
        let budget = super::super::input::DiskBudget::new(bytes, bytes);
        store.share_budget(budget.clone());
        let nodes = store.index.remove(&header.key).unwrap();
        for node in nodes {
            store.unlink(node).unwrap();
        }
        let id = *store.maintenance.keys().next().unwrap();
        complete(
            &log,
            TransportEvent::Unlinked { id, result: Ok(()) },
            &mut store,
        );
        assert!(budget.reserve_checkpoint(bytes));
    }

    fn complete(
        log: &Rc<RefCell<Log>>,
        event: TransportEvent,
        store: &mut DiskStore,
    ) -> Vec<StoreEvent> {
        log.borrow_mut().events.push(event);
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        events
    }

    fn unlinked(log: &Rc<RefCell<Log>>) -> Vec<PathBuf> {
        log.borrow()
            .calls
            .iter()
            .filter_map(|call| match call {
                Call::Unlink { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn startup_inventory_applies_recency_to_leaves_and_compactable_ancestors() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let child = header(&geometry, &full, 1300, Some(700));
        let leaf = header(&geometry, &full, 1990, Some(1300));
        let mut branch = history(1300);
        branch.append(&[5; 100], &[]).unwrap();
        let sibling = header(&geometry, &branch, 1400, Some(1300));
        assert_eq!(
            (
                root.tail_pages,
                child.tail_pages,
                sibling.tail_pages,
                leaf.tail_pages
            ),
            (10, 10, 1, 11)
        );
        let records = vec![
            scanned(&directory, root.clone(), 30, true),
            scanned(&directory, child.clone(), 10, true),
            scanned(&directory, sibling.clone(), 0, true),
            scanned(&directory, leaf.clone(), 20, false),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);
        assert_eq!(store.used(), total);
        assert_eq!(store.index().len(), 4);
        assert!(store.index().node(&root.key).unwrap().ready);
        assert!(!store.index().node(&leaf.key).unwrap().row);
        assert_eq!(store.record(&leaf.key).unwrap().bytes, leaf.row_segment);
        assert_eq!(
            store.record(&leaf.key).unwrap().path,
            directory.join(record::file_name(&leaf.key))
        );
        let stats = store.stats();
        assert_eq!((stats.disk_records, stats.disk_bytes), (4, total));
        assert_eq!(stats.inbound_rate, INITIAL_STAGED_DISK_RATE);
        assert_eq!(stats.outbound_rate, INITIAL_STAGED_DISK_RATE);
        assert_eq!(store.limit(), total);

        // After the oldest sibling leaves, the older child compacts before
        // the page-only leaf is removed; that removal also reclaims the child.
        let key = full.key_at(300).unwrap();
        let (reserved, temporary, path) = store
            .reserve_record(key, None, full.bucket_at(300).unwrap(), 4, true)
            .unwrap()
            .unwrap();
        assert_eq!(reserved.tail_pages, 4);
        assert_eq!(reserved.key, key);
        assert_eq!(path, directory.join(record::file_name(&key)));
        assert_eq!(temporary, directory.join(record::temporary_name(&key)));
        let bytes = reserved.length_with_row();
        assert_eq!(bytes, 2 * 4096 + 4 * PAGE_RECORD + ROW_RECORD);
        let sibling_bytes = sibling.length_with_row();
        assert_eq!(sibling_bytes, 2 * 4096 + PAGE_RECORD + ROW_RECORD);
        assert_eq!(
            unlinked(&log),
            [
                directory.join(record::file_name(&sibling.key)),
                directory.join(record::file_name(&leaf.key)),
                directory.join(record::file_name(&child.key))
            ]
        );
        assert_eq!(log.borrow().calls.len(), 4);
        assert_eq!(
            store.used(),
            total - sibling_bytes - leaf.row_segment - child.length_with_row() + bytes
        );
        assert_eq!(store.take_evicted(), [sibling.key, leaf.key, child.key]);
        assert!(store.take_evicted().is_empty());
        assert_eq!(store.index().len(), 2);
        let node = store.index().node(&key).unwrap();
        assert!(node.pinned() && !node.ready && node.row);
        assert_eq!(node.location.bytes, bytes);
        assert!(store.index().node(&child.key).is_none());
        assert_eq!(store.index().node(&root.key).unwrap().children(), 0);
    }

    #[test]
    fn store_dedupes_shapes_records_and_applies_completions() {
        let (mut store, log) = store_with(Vec::new(), 1 << 30);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        match &log.borrow().calls[..] {
            [
                Call::Store {
                    id,
                    header,
                    temporary,
                    path,
                    pages,
                    slot,
                },
            ] => {
                assert_eq!(*id, JobId(1));
                assert_eq!(header.parent, None);
                assert_eq!(header.tail_pages, 10);
                assert_eq!(header.key, root.key);
                assert_eq!(header.bucket, root.bucket);
                assert_eq!(pages, &root.chain_pages);
                assert_eq!(*slot, StateSlotId(1));
                assert_eq!(*path, store.directory().join(record::file_name(&root.key)));
                assert_eq!(
                    *temporary,
                    store.directory().join(record::temporary_name(&root.key))
                );
            }
            other => panic!("unexpected calls {other:?}"),
        }
        let node = store.index().node(&root.key).unwrap();
        assert!(node.pinned() && !node.ready && node.row);
        let root_bytes = node.location.bytes;
        assert_eq!(root_bytes, 2 * 4096 + 10 * PAGE_RECORD + ROW_RECORD);
        assert_eq!(store.used(), root_bytes);

        // The same key again is skipped without a second call, and a record
        // being written never matches a lookup.
        assert_ne!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        assert_eq!(store.stats().outbound_in_flight, 1);
        assert_eq!(log.borrow().calls.len(), 1);
        assert!(store.lookup(&full, 0, u64::MAX).is_none());

        // A child of a record being written is shaped as its child.
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
        match &log.borrow().calls[1] {
            Call::Store {
                id, header, pages, ..
            } => {
                assert_eq!(*id, JobId(2));
                assert_eq!(header.parent, Some(root.key));
                assert_eq!(header.tail_pages, 10);
                assert_eq!(pages, &child.chain_pages[10..]);
            }
            other => panic!("unexpected call {other:?}"),
        }
        assert_eq!(store.index().node(&root.key).unwrap().children(), 1);

        // A checkpoint whose parent the store never saw becomes a root record
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
        assert_eq!(
            store.plan_record(&orphan),
            (None, orphan.chain_pages.clone())
        );
        assert_eq!(store.store(&orphan).unwrap(), StoreOutcome::Queued);
        match &log.borrow().calls[2] {
            Call::Store {
                id, header, pages, ..
            } => {
                assert_eq!(*id, JobId(3));
                assert_eq!(header.parent, None);
                assert_eq!(header.tail_pages, 31);
                assert_eq!(pages, &orphan.chain_pages);
            }
            other => panic!("unexpected call {other:?}"),
        }
        assert_eq!(store.stats().outbound_in_flight, 3);
        let orphan_bytes = store.record(&orphan.key).unwrap().bytes;

        // The root's completion makes it ready, unpinned, and matchable.
        let events = complete(
            &log,
            TransportEvent::Stored {
                id: JobId(1),
                result: Ok(root_bytes),
            },
            &mut store,
        );
        match &events[..] {
            [
                StoreEvent::Stored {
                    checkpoint,
                    key,
                    result,
                },
            ] => {
                assert_eq!(*checkpoint, root.id);
                assert_eq!(*key, root.key);
                assert_eq!(*result.as_ref().unwrap(), root_bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        let node = store.index().node(&root.key).unwrap();
        assert!(!node.pinned() && node.ready);
        let stats = store.stats();
        assert_eq!(
            (stats.stored, stats.bytes_out, stats.outbound_in_flight),
            (1, root_bytes, 2)
        );
        assert!(stats.outbound_rate > 0.0 && stats.outbound_rate.is_finite());
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, root.key);
        assert_eq!(
            (hit.consumed_tokens, hit.resident_tokens, hit.pages),
            (700, 0, 10)
        );
        assert_eq!(hit.bytes, 10 * PAGE_RECORD + ROW_RECORD);
        assert_eq!(hit.tier, Tier::Nvme);
        assert_eq!(
            hit.estimated_seconds,
            hit.bytes as f64 / INITIAL_STAGED_DISK_RATE
        );
        assert_eq!(hit.chain.nodes, [root.key]);
        assert_eq!((store.stats().lookups, store.stats().hits), (2, 1));

        // The deeper root record wins once it completes.
        complete(
            &log,
            TransportEvent::Stored {
                id: JobId(3),
                result: Ok(orphan_bytes),
            },
            &mut store,
        );
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.consumed_tokens, 1990);
        assert_eq!(hit.chain.nodes, [orphan.key]);
        assert_eq!(hit.pages, 31);

        // A failed write leaves no record and returns its bytes.
        let child_bytes = store.record(&child.key).unwrap().bytes;
        let used = store.used();
        let events = complete(
            &log,
            TransportEvent::Stored {
                id: JobId(2),
                result: Err("disk full".into()),
            },
            &mut store,
        );
        match &events[..] {
            [
                StoreEvent::Stored {
                    checkpoint,
                    key,
                    result,
                },
            ] => {
                assert_eq!(*checkpoint, child.id);
                assert_eq!(*key, child.key);
                assert_eq!(result.as_ref().unwrap_err().to_string(), "disk full");
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(store.record(&child.key).is_none());
        assert_eq!(store.used(), used - child_bytes);
        assert_eq!(store.index().node(&root.key).unwrap().children(), 0);
        let stats = store.stats();
        assert_eq!(
            (
                stats.store_failed,
                stats.outbound_in_flight,
                stats.disk_records
            ),
            (1, 0, 2)
        );
        assert_eq!(stats.disk_bytes, root_bytes + orphan_bytes);
        assert!(store.take_evicted().is_empty());
        assert_eq!(log.borrow().calls.len(), 3);

        // Completions for unknown jobs are ignored.
        let events = complete(
            &log,
            TransportEvent::Stored {
                id: JobId(9),
                result: Ok(1),
            },
            &mut store,
        );
        assert!(events.is_empty());

        // The store refuses a checkpoint whose chain cannot hold its tail.
        let mut short = checkpoint(&mut ids, &full, 1000, None, pages(0..3), 15, StateSlotId(1));
        short.tail_pages = 15;
        assert!(store.store(&short).is_err());
        assert!(store.record(&short.key).is_none());

        store.close().unwrap();
        assert!(log.borrow().closed);
    }

    #[test]
    fn a_failed_parent_write_leaves_a_remnant_until_its_children_go() {
        let root_bytes = 2 * 4096 + 10 * PAGE_RECORD + ROW_RECORD;
        let (mut store, log) = store_with(Vec::new(), 2 * root_bytes);
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
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.used(), 2 * root_bytes);

        let events = complete(
            &log,
            TransportEvent::Stored {
                id: JobId(1),
                result: Err("torn".into()),
            },
            &mut store,
        );
        assert_eq!(events.len(), 1);
        let remnant = store.index().node(&root.key).unwrap();
        assert!(!remnant.ready && !remnant.row && !remnant.pinned());
        assert_eq!((remnant.children(), remnant.location.bytes), (1, 0));
        assert_eq!(store.used(), root_bytes);
        assert_eq!(store.stats().store_failed, 1);

        // The remnant is no parent for later records.
        let grandchild = checkpoint(
            &mut ids,
            &full,
            1990,
            Some(700),
            pages(0..31),
            21,
            StateSlotId(3),
        );
        assert_eq!(
            store.plan_record(&grandchild),
            (None, grandchild.chain_pages.clone())
        );
        assert!(
            store
                .reserve_record(grandchild.key, Some(root.key), grandchild.bucket, 21, true)
                .is_err()
        );
        assert!(store.record(&grandchild.key).is_none());

        // The child becomes ready, but no chain passes through the remnant.
        complete(
            &log,
            TransportEvent::Stored {
                id: JobId(2),
                result: Ok(root_bytes),
            },
            &mut store,
        );
        assert!(store.index().node(&child.key).unwrap().ready);
        assert!(store.lookup(&full, 0, u64::MAX).is_none());

        // Evicting the child takes the remnant with it; only the child's file is unlinked.
        let key = full.key_at(704).unwrap();
        let (reserved, _, _) = store
            .reserve_record(key, None, full.bucket_at(704).unwrap(), 11, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            unlinked(&log),
            [store.directory().join(record::file_name(&child.key))]
        );
        assert_eq!(log.borrow().calls.len(), 3);
        assert_eq!(store.take_evicted(), [child.key]);
        assert!(store.index().node(&root.key).is_none());
        assert!(store.index().node(&child.key).is_none());
        assert_eq!(store.used(), reserved.length_with_row());
        assert_eq!(store.index().len(), 1);
    }

    #[test]
    fn a_failed_restore_is_not_offered_again() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let leaf = header(&geometry, &full, 1990, Some(700));
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, 2 * total);

        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.key, leaf.key);
        let ticket = InboundTicket {
            key: hit.key,
            pages: (100..100 + hit.pages as u32).map(PageId).collect(),
            slot: StateSlotId(2),
        };
        store.fetch(hit, ticket).unwrap();
        log.borrow_mut().events.push(TransportEvent::Fetched {
            id: JobId(1),
            result: Err(invalid("page checksum mismatch")),
        });
        let mut events = Vec::new();
        store.poll(&mut events).unwrap();
        assert!(matches!(
            events.as_slice(),
            [StoreEvent::Fetched { result: Err(_), .. }]
        ));

        // A record whose bytes could not be restored is not offered again, so
        // every later admission does not repeat the same failing transfer. Its
        // ancestor stays available, because only the failed endpoint's row was
        // dropped.
        assert_eq!(store.lookup(&full, 0, u64::MAX).unwrap().key, root.key);
        assert!(store.lookup(&full, 700, u64::MAX).is_none());
    }

    #[test]
    fn fetch_plans_segments_after_skipped_pages_and_pins_the_chain() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let child = header(&geometry, &full, 1300, Some(700));
        let leaf = header(&geometry, &full, 1990, Some(1300));
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, child.clone(), 1, true),
            scanned(&directory, leaf.clone(), 2, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, 2 * total);

        let hit = store.lookup(&full, 1000, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [child.key, leaf.key]);
        assert_eq!(
            (hit.chain.skip_pages, hit.pages, hit.resident_tokens),
            (5, 16, 1000)
        );
        let bytes = 16 * PAGE_RECORD + ROW_RECORD;
        assert_eq!(hit.bytes, bytes);
        let ticket = |pages: u32| InboundTicket {
            key: hit.key,
            pages: (100..100 + pages).map(PageId).collect(),
            slot: StateSlotId(2),
        };

        // A ticket of the wrong size or key is refused before anything is pinned.
        assert!(store.fetch(hit.clone(), ticket(15)).is_err());
        let mut wrong = ticket(16);
        wrong.key = root.key;
        assert!(store.fetch(hit.clone(), wrong).is_err());
        assert!(store.index().nodes().all(|node| !node.pinned()));
        assert!(log.borrow().calls.is_empty());

        store.fetch(hit.clone(), ticket(16)).unwrap();
        match &log.borrow().calls[..] {
            [Call::Fetch { id, plan, ticket }] => {
                assert_eq!(*id, JobId(1));
                assert_eq!(ticket.pages.len(), 16);
                assert_eq!(
                    *plan,
                    [
                        FetchSegment {
                            path: directory.join(record::file_name(&child.key)),
                            header: child.clone(),
                            first_page: 5,
                            pages: 5,
                            row: false,
                        },
                        FetchSegment {
                            path: directory.join(record::file_name(&leaf.key)),
                            header: leaf.clone(),
                            first_page: 0,
                            pages: 11,
                            row: true,
                        }
                    ]
                );
            }
            other => panic!("unexpected calls {other:?}"),
        }
        let pins = |store: &DiskStore| {
            [&root.key, &child.key, &leaf.key].map(|key| store.index().node(key).unwrap().pins())
        };
        assert_eq!(pins(&store), [0, 1, 1]);
        assert_eq!(store.stats().inbound_in_flight, 1);

        // A second restore of the same chain is an independent job.
        let again = store.lookup(&full, 1000, u64::MAX).unwrap();
        store.fetch(again, ticket(16)).unwrap();
        assert_eq!(pins(&store), [0, 2, 2]);

        // Pinned records survive eviction; the unpinned root compacts instead.
        let key = full.key_at(300).unwrap();
        assert!(
            store
                .reserve_record(key, None, full.bucket_at(300).unwrap(), 4, true)
                .unwrap()
                .is_some()
        );
        assert!(store.index().node(&root.key).unwrap().row);
        store.limit = store.used();
        assert!(
            store
                .reserve_record(
                    full.key_at(400).unwrap(),
                    None,
                    full.bucket_at(400).unwrap(),
                    6,
                    true
                )
                .unwrap()
                .is_none()
        );
        match &log.borrow().calls[2..] {
            [Call::Truncate { id, path, length }] => {
                assert_eq!(*id, JobId(3));
                assert_eq!(*path, directory.join(record::file_name(&root.key)));
                assert_eq!(*length, root.row_segment);
            }
            other => panic!("unexpected calls {other:?}"),
        }
        assert!(!store.index().node(&root.key).unwrap().row);
        assert_eq!(store.record(&root.key).unwrap().bytes, root.row_segment);
        assert_eq!(pins(&store), [0, 2, 2]);

        let events = complete(
            &log,
            TransportEvent::Fetched {
                id: JobId(1),
                result: Ok(bytes),
            },
            &mut store,
        );
        match &events[..] {
            [StoreEvent::Fetched { key, result }] => {
                assert_eq!(*key, leaf.key);
                assert_eq!(*result.as_ref().unwrap(), bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(pins(&store), [0, 1, 1]);
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.bytes_in, stats.reused_tokens),
            (1, bytes, 990)
        );
        assert_eq!(stats.inbound_in_flight, 1);
        assert!(stats.inbound_rate > 0.0 && stats.inbound_rate.is_finite());

        let events = complete(
            &log,
            TransportEvent::Fetched {
                id: JobId(2),
                result: Err("checksum".into()),
            },
            &mut store,
        );
        match &events[..] {
            [StoreEvent::Fetched { key, result }] => {
                assert_eq!(*key, leaf.key);
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(pins(&store), [0, 0, 0]);
        let stats = store.stats();
        assert_eq!(
            (stats.fetched, stats.fetch_failed, stats.inbound_in_flight),
            (1, 1, 0)
        );
        assert_eq!(stats.reused_tokens, 990);

        // A failed truncate leaves the row bytes in the file: the record and
        // the budget regain them without a store event, and the row stays
        // unmatchable.
        let used = store.used();
        let events = complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(3),
                result: Err("busy".into()),
            },
            &mut store,
        );
        assert!(events.is_empty());
        assert_eq!(
            store.record(&root.key).unwrap().bytes,
            root.length_with_row()
        );
        assert!(!store.index().node(&root.key).unwrap().row);
        assert_eq!(store.used(), used + ROW_RECORD);
        assert!(store.maintenance.is_empty());
    }

    #[test]
    fn reserving_a_child_never_evicts_its_parent_and_an_oversized_record_evicts_nothing() {
        let root_bytes = 2 * 4096 + 10 * PAGE_RECORD + ROW_RECORD;
        let (mut store, log) = store_with(Vec::new(), root_bytes + 4096);
        let full = history(2000);
        let mut ids = Ids::new();
        let root = checkpoint(&mut ids, &full, 700, None, pages(0..10), 10, StateSlotId(1));
        assert_eq!(store.store(&root).unwrap(), StoreOutcome::Queued);
        complete(
            &log,
            TransportEvent::Stored {
                id: JobId(1),
                result: Ok(root_bytes),
            },
            &mut store,
        );

        // With only the parent to evict, the child is skipped and the parent stays.
        let child = checkpoint(
            &mut ids,
            &full,
            1300,
            Some(700),
            pages(0..20),
            10,
            StateSlotId(2),
        );
        assert_ne!(store.store(&child).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 1);
        assert!(unlinked(&log).is_empty());
        assert_eq!(log.borrow().calls.len(), 1);
        let node = store.index().node(&root.key).unwrap();
        assert!(node.ready && !node.pinned());
        assert_eq!(node.children(), 0);
        assert_eq!((store.used(), store.index().len()), (root_bytes, 1));
        assert!(store.take_evicted().is_empty());

        // Beside its parent the child fits, and the parent ends unpinned.
        store.limit = 2 * root_bytes;
        assert_eq!(store.store(&child).unwrap(), StoreOutcome::Queued);
        let node = store.index().node(&root.key).unwrap();
        assert_eq!((node.children(), node.pins()), (1, 0));
        complete(
            &log,
            TransportEvent::Stored {
                id: JobId(2),
                result: Ok(root_bytes),
            },
            &mut store,
        );
        let mut branch = history(700);
        branch.append(&[5; 700], &[]).unwrap();
        let sibling = checkpoint(
            &mut ids,
            &branch,
            1400,
            Some(700),
            pages(0..21),
            11,
            StateSlotId(3),
        );
        let sibling_bytes = 2 * 4096 + 11 * PAGE_RECORD + ROW_RECORD;
        store.limit = 2 * root_bytes + sibling_bytes;
        assert_eq!(store.store(&sibling).unwrap(), StoreOutcome::Queued);
        complete(
            &log,
            TransportEvent::Stored {
                id: JobId(3),
                result: Ok(sibling_bytes),
            },
            &mut store,
        );
        assert_eq!(store.index().node(&root.key).unwrap().children(), 2);

        // A grandchild's parent is the least recently used leaf, so the
        // younger sibling is the one evicted.
        let grandchild = checkpoint(
            &mut ids,
            &full,
            1990,
            Some(1300),
            pages(0..31),
            11,
            StateSlotId(4),
        );
        assert_eq!(store.store(&grandchild).unwrap(), StoreOutcome::Queued);
        assert_eq!(
            unlinked(&log),
            [store.directory().join(record::file_name(&sibling.key))]
        );
        assert_eq!(store.take_evicted(), [sibling.key]);
        let parent = store.index().node(&child.key).unwrap();
        assert!(parent.ready && !parent.pinned());
        assert_eq!(parent.children(), 1);
        assert_eq!(store.index().node(&root.key).unwrap().children(), 1);
        assert_eq!(
            store.index().node(&grandchild.key).unwrap().parent,
            Some(child.key)
        );
        assert_eq!(store.used(), 2 * root_bytes + sibling_bytes);
        // The sibling's unlink took job 4; the grandchild's write is job 5.
        complete(
            &log,
            TransportEvent::Stored {
                id: JobId(5),
                result: Ok(sibling_bytes),
            },
            &mut store,
        );

        // A record larger than the whole limit is skipped before anything is evicted.
        store.limit = store.used();
        let long = history(4000);
        let huge = checkpoint(
            &mut ids,
            &long,
            3010,
            None,
            pages(0..47),
            47,
            StateSlotId(5),
        );
        assert!(2 * 4096 + 47 * PAGE_RECORD + ROW_RECORD > store.limit());
        assert_ne!(store.store(&huge).unwrap(), StoreOutcome::Queued);
        assert_eq!(store.stats().store_skipped, 2);
        assert_eq!(log.borrow().calls.len(), 5);
        assert_eq!(store.index().len(), 3);
        assert_eq!(store.used(), store.limit());

        // Reservation refuses an indexed key and an unknown parent before
        // evicting anything.
        assert!(
            store
                .reserve_record(root.key, None, root.bucket, 10, true)
                .is_err()
        );
        let unknown = full.key_at(1000).unwrap();
        assert!(
            store
                .reserve_record(
                    full.key_at(1100).unwrap(),
                    Some(unknown),
                    full.bucket_at(1100).unwrap(),
                    1,
                    true
                )
                .is_err()
        );
        assert_eq!(log.borrow().calls.len(), 5);
        assert_eq!(store.index().len(), 3);
        assert!(store.index().nodes().all(|node| !node.pinned()));
    }

    #[test]
    fn reserving_under_a_page_only_parent_keeps_it_through_the_cascade() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let leaf = header(&geometry, &full, 1300, Some(700));
        let records = vec![
            scanned(&directory, root.clone(), 0, false),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);

        // A second child of the page-only root needs the leaf's bytes. Removing
        // the leaf would cascade into the childless root, which is the parent
        // being named, so the root survives and the child is indexed under it.
        let mut branch = history(700);
        branch.append(&[5; 700], &[]).unwrap();
        let key = branch.key_at(1340).unwrap();
        let (reserved, _, _) = store
            .reserve_record(
                key,
                Some(root.key),
                branch.bucket_at(1340).unwrap(),
                10,
                true,
            )
            .unwrap()
            .unwrap();
        assert_eq!(reserved.length_with_row(), leaf.length_with_row());
        assert_eq!(
            unlinked(&log),
            [directory.join(record::file_name(&leaf.key))]
        );
        assert_eq!(store.take_evicted(), [leaf.key]);
        let parent = store.index().node(&root.key).unwrap();
        assert!(parent.ready && !parent.row && !parent.pinned());
        assert_eq!(parent.children(), 1);
        assert_eq!(store.index().node(&key).unwrap().parent, Some(root.key));
        assert_eq!(store.used(), root.row_segment + reserved.length_with_row());
        assert_eq!(store.index().len(), 2);

        // Once the child is ready, the chain through the page-only root matches.
        store
            .commit_record(key, Ok(reserved.length_with_row()))
            .unwrap();
        let hit = store.lookup(&branch, 0, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [root.key, key]);
        assert_eq!(hit.pages, 20);
    }

    #[test]
    fn failed_unlinks_and_truncates_keep_their_bytes_against_the_limit() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let child = header(&geometry, &full, 1300, Some(700));
        let leaf = header(&geometry, &full, 1990, Some(1300));
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, child.clone(), 10, true),
            scanned(&directory, leaf.clone(), 20, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);
        store.index_mut().pin(&leaf.key).unwrap();
        let key = full.key_at(300).unwrap();
        let bucket = full.bucket_at(300).unwrap();
        assert!(
            store
                .reserve_record(key, None, bucket, 4, true)
                .unwrap()
                .is_none()
        );
        match &log.borrow().calls[..] {
            [
                Call::Truncate { id: first, .. },
                Call::Truncate { id: second, .. },
            ] => assert_eq!((*first, *second), (JobId(1), JobId(2))),
            other => panic!("unexpected calls {other:?}"),
        }
        assert_eq!(store.used(), total - 2 * ROW_RECORD);

        // The root's truncate fails: its file keeps the row bytes, so the
        // record regains its length and the budget its bytes; the child's
        // succeeds and changes nothing further.
        let events = complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(1),
                result: Err("busy".into()),
            },
            &mut store,
        );
        assert!(events.is_empty());
        assert_eq!(
            store.record(&root.key).unwrap().bytes,
            root.length_with_row()
        );
        assert!(!store.index().node(&root.key).unwrap().row);
        assert_eq!(store.used(), total - ROW_RECORD);
        complete(
            &log,
            TransportEvent::Truncated {
                id: JobId(2),
                result: Ok(()),
            },
            &mut store,
        );
        assert_eq!(store.record(&child.key).unwrap().bytes, child.row_segment);
        assert_eq!(store.stats().disk_bytes, total - ROW_RECORD);

        // Releasing the leaf lets the whole chain go. The child's unlink fails,
        // so its bytes stay charged although the record left the index.
        store.index_mut().unpin(&leaf.key).unwrap();
        let (reserved, _, _) = store
            .reserve_record(key, None, bucket, 4, true)
            .unwrap()
            .unwrap();
        assert_eq!(store.take_evicted(), [leaf.key, child.key, root.key]);
        assert_eq!(store.used(), reserved.length_with_row());
        match &log.borrow().calls[2..] {
            [
                Call::Unlink { id: first, .. },
                Call::Unlink { id: second, .. },
                Call::Unlink { id: third, .. },
            ] => assert_eq!((*first, *second, *third), (JobId(3), JobId(4), JobId(5))),
            other => panic!("unexpected calls {other:?}"),
        }
        for (id, result) in [
            (JobId(3), Ok(())),
            (JobId(4), Err("permission denied".into())),
            (JobId(5), Ok(())),
            (JobId(9), Err("unknown".into())),
        ] {
            let events = complete(&log, TransportEvent::Unlinked { id, result }, &mut store);
            assert!(events.is_empty());
        }
        assert_eq!(store.used(), reserved.length_with_row() + child.row_segment);
        let stats = store.stats();
        assert_eq!(
            (stats.disk_records, stats.disk_bytes),
            (1, reserved.length_with_row() + child.row_segment)
        );
        assert!(store.maintenance.is_empty());
    }

    #[test]
    fn pinned_leaves_compact_ancestors_and_release_cascades_to_page_only_parents() {
        let geometry = geometry();
        let directory = directory();
        let full = history(2000);
        let root = header(&geometry, &full, 700, None);
        let child = header(&geometry, &full, 1300, Some(700));
        let leaf = header(&geometry, &full, 1990, Some(1300));
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, child.clone(), 10, true),
            scanned(&directory, leaf.clone(), 20, true),
        ];
        let total: u64 = records.iter().map(|record| record.bytes).sum();
        let (mut store, log) = store_with(records, total);
        store.index_mut().pin(&leaf.key).unwrap();

        // With the only leaf pinned, the oldest single-child ancestors lose
        // their rows, oldest first, and the record still does not fit.
        let key = full.key_at(300).unwrap();
        let bucket = full.bucket_at(300).unwrap();
        assert!(
            store
                .reserve_record(key, None, bucket, 4, true)
                .unwrap()
                .is_none()
        );
        match &log.borrow().calls[..] {
            [
                Call::Truncate {
                    path: first,
                    length: first_length,
                    ..
                },
                Call::Truncate {
                    path: second,
                    length: second_length,
                    ..
                },
            ] => {
                assert_eq!(*first, directory.join(record::file_name(&root.key)));
                assert_eq!(*first_length, root.row_segment);
                assert_eq!(*second, directory.join(record::file_name(&child.key)));
                assert_eq!(*second_length, child.row_segment);
            }
            other => panic!("unexpected calls {other:?}"),
        }
        assert_eq!(store.used(), total - 2 * ROW_RECORD);
        assert!(!store.index().node(&root.key).unwrap().row);
        assert!(!store.index().node(&child.key).unwrap().row);
        assert_eq!(store.record(&child.key).unwrap().bytes, child.row_segment);
        assert!(store.take_evicted().is_empty());
        assert!(store.index().node(&key).is_none());

        // Page-only ancestors still supply pages to the leaf's chain but no
        // longer match on their own.
        let hit = store.lookup(&full, 0, u64::MAX).unwrap();
        assert_eq!(hit.chain.nodes, [root.key, child.key, leaf.key]);
        assert_eq!(hit.pages, 31);
        assert!(store.lookup(&history(1300), 0, u64::MAX).is_none());

        // Releasing the pin lets the leaf go, and its page-only ancestors follow.
        store.index_mut().unpin(&leaf.key).unwrap();
        let (reserved, _, _) = store
            .reserve_record(key, None, bucket, 4, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            unlinked(&log),
            [&leaf, &child, &root].map(|header| directory.join(record::file_name(&header.key)))
        );
        assert_eq!(store.take_evicted(), [leaf.key, child.key, root.key]);
        assert_eq!(store.used(), reserved.length_with_row());
        assert_eq!(store.index().len(), 1);
        assert!(store.index().node(&key).unwrap().pinned());

        // Committing the reserved record directly, as the composite does.
        store
            .commit_record(key, Ok(reserved.length_with_row()))
            .unwrap();
        let node = store.index().node(&key).unwrap();
        assert!(node.ready && !node.pinned());
        assert_eq!(store.stats().stored, 1);
        assert!(store.commit_record(root.key, Ok(0)).is_err());
    }

    #[test]
    fn windows_split_segments_at_the_window_size_and_carry_the_row_last() {
        let plans = plan_windows([(5u32, 3usize, false), (0, 0, false), (2, 130, true)]);
        let summary: Vec<_> = plans
            .iter()
            .map(|plan| {
                (
                    plan.segment,
                    plan.index,
                    plan.first,
                    plan.count,
                    plan.offset,
                    plan.row,
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (0, 0, 5, 3, 0, false),
                (2, 1, 2, 64, 3, false),
                (2, 2, 66, 64, 67, false),
                (2, 3, 130, 2, 131, true),
            ]
        );
        let row_only = plan_windows([(0, 0, true)]);
        assert_eq!(row_only.len(), 1);
        assert_eq!((row_only[0].count, row_only[0].row), (0, true));
        let exact = plan_windows([(0, 64, true)]);
        assert_eq!(exact.len(), 1);
        assert_eq!((exact[0].count, exact[0].row), (64, true));
        assert!(plan_windows([]).is_empty());
        assert!(plan_windows([(0, 0, false)]).is_empty());
    }

    fn stopped_transfers() -> SharedCopier {
        StoppedCopier::shared(&geometry()).unwrap()
    }

    /// Caller-owned aligned buffers: the slabs of an unregistered pool.
    fn host_pool() -> SlabPool {
        SlabPool::staging(&Layout::new(&geometry()).unwrap()).unwrap()
    }

    fn page_slabs(pool: &mut SlabPool, count: usize, seed: u64) -> Vec<PageSlab> {
        (0..count as u64)
            .map(|index| {
                let slab = pool.allocate_page().unwrap();
                pattern(pool.page_mut(slab), seed + index);
                slab
            })
            .collect()
    }

    fn row_slab(pool: &mut SlabPool, seed: u64) -> RowSlab {
        let slab = pool.allocate_row().unwrap();
        pattern(pool.row_mut(slab), seed);
        slab
    }

    fn addresses(pool: &SlabPool, slabs: &[PageSlab]) -> Vec<usize> {
        slabs.iter().map(|slab| pool.page_address(*slab)).collect()
    }

    fn transport_events(staged: &mut Staged, count: usize) -> Vec<TransportEvent> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut events = Vec::new();
        while events.len() < count {
            assert!(
                Instant::now() < deadline,
                "{} of {count} transport events within 60 s",
                events.len()
            );
            staged.poll(&mut events).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(events.len(), count);
        events
    }

    /// Nothing in flight, no disk job owned, and no disk reply awaited.
    fn settled(staged: &Staged) -> bool {
        staged.in_flight() == (0, 0)
            && staged.owners.is_empty()
            && staged.resources.disk_outstanding == 0
            && staged.maintenance.is_empty()
    }

    fn corrupt(path: &Path, offset: u64) {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte[0] ^ 0x5a]).unwrap();
    }

    #[test]
    fn staged_reads_reject_changed_headers_before_writing_host_or_staging_slabs() {
        let Some(directory) = test_directory("disk-store", "changed-header") else {
            return;
        };
        let geometry = geometry();
        let expected = header(&geometry, &history(200), 200, None);
        let mut variants = Vec::new();
        let mut changed = expected.clone();
        changed.key.digest[0] ^= 1;
        variants.push(("key", changed));
        let mut changed = expected.clone();
        changed.ranks[0].planes[0] -= 1;
        variants.push(("geometry", changed));
        let mut changed = expected.clone();
        changed.ranks[0].planes[0] += 4096;
        changed.row_segment += u64::from(changed.tail_pages) * 4096;
        variants.push(("page-size", changed));
        let mut changed = expected.clone();
        changed.ranks[0].row_bytes += 4096;
        variants.push(("row-size", changed));
        for (name, changed) in variants {
            let path = directory.join(format!("{name}.ckpt"));
            let mut bytes = vec![0x5a; changed.length_with_row() as usize];
            bytes[..record::HEADER_BYTES].copy_from_slice(&changed.encode().unwrap());
            fs::write(&path, bytes).unwrap();
            for device in [false, true] {
                let mut pool = SlabPool::new(PAGE_RECORD, 4, ROW_RECORD, 2, true).unwrap();
                for index in 0..4 {
                    pool.page_mut(PageSlab(index)).fill(0xee);
                }
                for index in 0..2 {
                    pool.row_mut(RowSlab(index)).fill(0xee);
                }
                let page = pool.page_address(PageSlab(0));
                let row = pool.row_address(RowSlab(0));
                let (host, staging) = if device {
                    (None, Some(pool))
                } else {
                    (Some(pool), None)
                };
                let mut staged = Staged::new(
                    staging,
                    stopped_transfers(),
                    DiskThread::start("flare-offload-disk").unwrap(),
                );
                if device {
                    staged
                        .fetch(
                            JobId(1),
                            vec![FetchSegment {
                                path: path.clone(),
                                header: expected.clone(),
                                first_page: 0,
                                pages: 1,
                                row: true,
                            }],
                            &InboundTicket {
                                key: expected.key,
                                pages: pages(0..1),
                                slot: StateSlotId(0),
                            },
                        )
                        .unwrap();
                } else {
                    staged
                        .read_into(
                            JobId(1),
                            path.clone(),
                            expected.clone(),
                            0,
                            vec![page],
                            Some(row),
                        )
                        .unwrap();
                }
                match &transport_events(&mut staged, 1)[..] {
                    [
                        TransportEvent::Fetched {
                            result: Err(error), ..
                        },
                    ] => assert!(
                        error.to_string().contains("different header"),
                        "{name}, device={device}: {error}"
                    ),
                    other => panic!("{name}, device={device}: {other:?}"),
                }
                assert!(settled(&staged));
                let pool = host.as_ref().or_else(|| staged.pool()).unwrap();
                for index in 0..4 {
                    assert!(
                        pool.page(PageSlab(index)).iter().all(|byte| *byte == 0xee),
                        "{name} changed page {index}, device={device}"
                    );
                }
                for index in 0..2 {
                    assert!(
                        pool.row(RowSlab(index)).iter().all(|byte| *byte == 0xee),
                        "{name} changed row {index}, device={device}"
                    );
                }
                staged.close().unwrap();
            }
        }
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compacted_record_is_atomically_rewritten_and_survives_rescan() {
        let Some(directory) = test_directory("disk-store", "row-replacement") else {
            return;
        };
        let full = history(500);
        let root = header(&geometry(), &full, 128, None);
        let child = header(&geometry(), &full, 256, Some(128));
        let mut pool = host_pool();
        let source = page_slabs(&mut pool, 2, 17);
        let row = row_slab(&mut pool, 19);
        let mut staged = Staged::new(
            None,
            stopped_transfers(),
            DiskThread::start("row-refill").unwrap(),
        );
        for (id, header) in [(1, &root), (2, &child)] {
            staged
                .write_from(
                    JobId(id),
                    header.clone(),
                    directory.join(record::temporary_name(&header.key)),
                    directory.join(record::file_name(&header.key)),
                    addresses(&pool, &source),
                    Some(pool.row_address(row)),
                    None,
                )
                .unwrap();
            assert!(matches!(
                &transport_events(&mut staged, 1)[0],
                TransportEvent::Stored { result: Ok(_), .. }
            ));
        }
        staged
            .truncate(
                JobId(3),
                directory.join(record::file_name(&root.key)),
                root.row_segment,
            )
            .unwrap();
        assert!(matches!(
            &transport_events(&mut staged, 1)[0],
            TransportEvent::Truncated { result: Ok(_), .. }
        ));
        let guard = crate::testing::directory_lock(&directory).unwrap();
        let scanned = record::scan(&guard, &IDENTITY, &FINGERPRINT, geometry()[0].page_tokens, &record::rank_sizes(&geometry())).unwrap();
        let mut store = DiskStore::new(
            guard,
            IDENTITY,
            FINGERPRINT,
            1 << 30,
            geometry(),
            scanned,
            Box::new(staged),
        )
        .unwrap();
        let (replacement, temporary, path) = store
            .reserve_record(root.key, None, root.bucket, 2, true)
            .unwrap()
            .unwrap();
        assert!(!store.index.complete(&child.key));
        for slab in &source {
            pool.page_mut(*slab).fill(0x87);
        }
        pool.row_mut(row).fill(0x91);
        store
            .transport
            .staged_mut()
            .unwrap()
            .write_from(
                JobId(4),
                replacement.clone(),
                temporary,
                path.clone(),
                addresses(&pool, &source),
                Some(pool.row_address(row)),
                None,
            )
            .unwrap();
        assert!(matches!(
            &transport_events(store.transport.staged_mut().unwrap(), 1)[0],
            TransportEvent::Stored { result: Ok(_), .. }
        ));
        store
            .commit_record(root.key, Ok(replacement.length_with_row()))
            .unwrap();
        assert!(store.index.complete(&child.key));
        let bytes = fs::read(&path).unwrap();
        let layout = Layout::new(&geometry()).unwrap();
        for page in 0..2 {
            for plane in 0..2 {
                let offset = (replacement.page_segment
                    + page * layout.page_record_bytes
                    + layout.ranks[0].plane_offsets[plane]) as usize;
                assert!(
                    bytes[offset..offset + 8192]
                        .iter()
                        .all(|byte| *byte == 0x87)
                );
            }
        }
        assert!(
            bytes[replacement.row_segment as usize..]
                .iter()
                .all(|byte| *byte == 0x91)
        );
        store.close().unwrap();
        drop(store);
        let guard = crate::testing::directory_lock(&directory).unwrap();
        let scanned = record::scan(&guard, &IDENTITY, &FINGERPRINT, geometry()[0].page_tokens, &record::rank_sizes(&geometry())).unwrap();
        assert_eq!(scanned.len(), 2);
        assert!(scanned.iter().all(|record| record.row));
        drop(guard);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn staged_moves_records_between_files_and_caller_buffers_without_a_pool() {
        let Some(directory) = test_directory("disk-store", "host-buffers") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let mut pool = host_pool();
        let mut staged = Staged::new(
            None,
            stopped_transfers(),
            DiskThread::start("flare-offload-disk").unwrap(),
        );
        const PAGES: usize = WINDOW_PAGES + 6;
        let full = history(5000);
        let consumed = PAGES as u64 * PAGE_TOKENS + 3;
        let header = header(&geometry, &full, consumed, None);
        assert_eq!(header.tail_pages as usize, PAGES);
        let sources = page_slabs(&mut pool, PAGES, 1);
        let source_row = row_slab(&mut pool, 99);
        let source_addresses = addresses(&pool, &sources);
        let path = directory.join(record::file_name(&header.key));
        let temporary = directory.join(record::temporary_name(&header.key));

        // Without a pool, device pipelines are refused; host pipelines check
        // their arguments before taking the job.
        let mut ids = Ids::new();
        let checkpoint = checkpoint(
            &mut ids,
            &full,
            consumed,
            None,
            pages(0..PAGES as u32),
            PAGES,
            StateSlotId(1),
        );
        assert!(
            staged
                .store(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    &checkpoint
                )
                .is_err()
        );
        let segment = FetchSegment {
            path: path.clone(),
            header: header.clone(),
            first_page: 0,
            pages: PAGES as u32,
            row: true,
        };
        let ticket = InboundTicket {
            key: header.key,
            pages: pages(0..PAGES as u32),
            slot: StateSlotId(1),
        };
        assert!(staged.fetch(JobId(1), vec![segment], &ticket).is_err());
        assert!(
            staged
                .write_from(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses[..PAGES - 1].to_vec(),
                    Some(pool.row_address(source_row)),
                    None,
                )
                .is_err()
        );
        assert!(
            staged
                .write_from(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses.clone(),
                    None,
                    None,
                )
                .is_err()
        );
        assert!(
            staged
                .read_into(JobId(1), path.clone(), header.clone(), 0, Vec::new(), None)
                .is_err()
        );
        assert!(settled(&staged));

        // Two windows and the row stream into the file; a second job under
        // the same id is refused while the first is in flight.
        staged
            .write_from(
                JobId(1),
                header.clone(),
                temporary.clone(),
                path.clone(),
                source_addresses.clone(),
                Some(pool.row_address(source_row)),
                None,
            )
            .unwrap();
        assert!(
            staged
                .write_from(
                    JobId(1),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses.clone(),
                    Some(pool.row_address(source_row)),
                    None,
                )
                .is_err()
        );
        assert_eq!(staged.in_flight(), (1, 0));
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result }] => {
                assert_eq!(*id, JobId(1));
                assert_eq!(*result.as_ref().unwrap(), header.length_with_row());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());
        assert!(!temporary.exists());
        let scanned = record::scan(
            &crate::testing::directory_lock(&directory).unwrap(),
            &IDENTITY,
            &FINGERPRINT,
            geometry[0].page_tokens,
            &record::rank_sizes(&geometry),
        )
        .unwrap();
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].header.key, header.key);
        assert!(scanned[0].row);
        assert!(settled(&staged));

        // A suffix of the pages and the row land in caller buffers byte-exact,
        // and pages can be read without the row.
        let targets = page_slabs(&mut pool, PAGES, 1000);
        let target_row = row_slab(&mut pool, 1999);
        const SKIP: usize = 5;
        staged
            .read_into(
                JobId(2),
                path.clone(),
                header.clone(),
                SKIP as u32,
                addresses(&pool, &targets[..PAGES - SKIP]),
                Some(pool.row_address(target_row)),
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(2));
                assert_eq!(
                    *result.as_ref().unwrap(),
                    layout.record_bytes((PAGES - SKIP) as u64)
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        for (target, source) in targets.iter().zip(&sources[SKIP..]) {
            assert!(pool.page(*target) == pool.page(*source));
        }
        assert!(pool.row(target_row) == pool.row(source_row));
        staged
            .read_into(
                JobId(3),
                path.clone(),
                header.clone(),
                PAGES as u32 - 2,
                addresses(&pool, &targets[PAGES - 2..]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(3));
                assert_eq!(*result.as_ref().unwrap(), 2 * layout.page_record_bytes);
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(pool.page(targets[PAGES - 2]) == pool.page(sources[PAGES - 2]));
        assert!(pool.page(targets[PAGES - 1]) == pool.page(sources[PAGES - 1]));
        assert!(settled(&staged));

        // A missing file fails at open and a corrupted page fails its
        // checksum; both leave nothing in flight.
        staged
            .read_into(
                JobId(4),
                directory.join("missing.ckpt"),
                header.clone(),
                0,
                addresses(&pool, &targets[..1]),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(4));
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        corrupt(
            &path,
            header.page_segment + 10 * layout.page_record_bytes + 100,
        );
        staged
            .read_into(
                JobId(5),
                path.clone(),
                header.clone(),
                0,
                addresses(&pool, &targets),
                Some(pool.row_address(target_row)),
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(5));
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("checksum")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));

        // Truncation drops the row, after which reading it fails; unlinking
        // removes the file, and unlinking again fails.
        staged
            .truncate(JobId(6), path.clone(), header.row_segment)
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Truncated { id, result }] => {
                assert_eq!(*id, JobId(6));
                result.as_ref().unwrap();
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert_eq!(fs::metadata(&path).unwrap().len(), header.row_segment);
        staged
            .read_into(
                JobId(7),
                path.clone(),
                header.clone(),
                WINDOW_PAGES as u32,
                addresses(&pool, &targets[..PAGES - WINDOW_PAGES]),
                Some(pool.row_address(target_row)),
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(7));
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        staged.unlink(JobId(8), path.clone()).unwrap();
        staged.unlink(JobId(9), path.clone()).unwrap();
        match &transport_events(&mut staged, 2)[..] {
            [
                TransportEvent::Unlinked {
                    id: first,
                    result: Ok(()),
                },
                TransportEvent::Unlinked {
                    id: second,
                    result: Ok(()),
                },
            ] => assert_eq!((*first, *second), (JobId(8), JobId(9))),
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!path.exists());
        assert!(settled(&staged));

        staged.close().unwrap();
        assert!(
            staged
                .write_from(
                    JobId(10),
                    header.clone(),
                    temporary.clone(),
                    path.clone(),
                    source_addresses.clone(),
                    Some(pool.row_address(source_row)),
                    None,
                )
                .is_err()
        );
        assert!(staged.truncate(JobId(10), path.clone(), 0).is_err());
        assert!(staged.unlink(JobId(10), path.clone()).is_err());
        let mut events = Vec::new();
        staged.poll(&mut events).unwrap();
        assert!(events.is_empty());
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn staged_failures_release_staging_slabs_and_abort_temporary_files() {
        let Some(directory) = test_directory("disk-store", "failures") else {
            return;
        };
        let geometry = geometry();
        let layout = Layout::new(&geometry).unwrap();
        let mut host = host_pool();
        let mut staged = Staged::new(
            Some(SlabPool::staging(&layout).unwrap()),
            stopped_transfers(),
            DiskThread::start("flare-offload-disk").unwrap(),
        );
        let full = history(2000);
        let mut ids = Ids::new();
        let free = |staged: &Staged| {
            let pool = staged.pool().unwrap();
            (pool.free_pages(), pool.free_rows())
        };
        let all = (STAGING_PAGE_SLABS, STAGING_ROW_SLABS);
        assert_eq!(free(&staged), all);

        // A record written from caller buffers gives the fetches below a file.
        let stored = header(&geometry, &full, 3 * PAGE_TOKENS + 1, None);
        let sources = page_slabs(&mut host, 3, 1);
        let source_row = row_slab(&mut host, 9);
        let source_addresses = addresses(&host, &sources);
        let path = directory.join(record::file_name(&stored.key));
        staged
            .write_from(
                JobId(1),
                stored.clone(),
                directory.join(record::temporary_name(&stored.key)),
                path.clone(),
                source_addresses.clone(),
                Some(host.row_address(source_row)),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result }] => {
                assert_eq!(*id, JobId(1));
                result.as_ref().unwrap();
            }
            other => panic!("unexpected events {other:?}"),
        }

        // Without transfer threads a store fails at its first window: the
        // slabs return to the pool at once, the temporary file is aborted once
        // its create reply arrives, and the failure is reported once.
        let device = header(&geometry, &full, 3 * PAGE_TOKENS + 2, None);
        let checkpoint = checkpoint(
            &mut ids,
            &full,
            3 * PAGE_TOKENS + 2,
            None,
            pages(0..3),
            3,
            StateSlotId(1),
        );
        let device_temporary = directory.join(record::temporary_name(&device.key));
        let device_path = directory.join(record::file_name(&device.key));
        staged
            .store(
                JobId(2),
                device.clone(),
                device_temporary.clone(),
                device_path.clone(),
                &checkpoint,
            )
            .unwrap();
        assert_eq!(staged.in_flight(), (1, 0));
        assert_eq!(free(&staged), all);
        assert!(staged.outbound[&JobId(2)].file == TemporaryFile::Creating);
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result }] => {
                assert_eq!(*id, JobId(2));
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("no offload threads")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!device_temporary.exists() && !device_path.exists());
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);

        // A fetch reads its window from the file into staging and fails at
        // the upload: the slabs are released after the read replies and the
        // file is closed.
        let ticket = InboundTicket {
            key: stored.key,
            pages: pages(10..13),
            slot: StateSlotId(2),
        };
        let segment = FetchSegment {
            path: path.clone(),
            header: stored.clone(),
            first_page: 0,
            pages: 3,
            row: true,
        };
        staged
            .fetch(JobId(3), vec![segment.clone()], &ticket)
            .unwrap();
        assert_eq!(
            free(&staged),
            (STAGING_PAGE_SLABS - 3, STAGING_ROW_SLABS - 1)
        );
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(3));
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("no offload threads")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);

        // A file holding a different record than the index expects fails the
        // fetch at open.
        let mut other = segment.clone();
        other.header = device.clone();
        staged.fetch(JobId(4), vec![other], &ticket).unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { id, result }] => {
                assert_eq!(*id, JobId(4));
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("different header")
                );
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);

        // A plan must take the row from its last record and cover the ticket.
        let mut no_row = segment.clone();
        no_row.row = false;
        assert!(staged.fetch(JobId(5), vec![no_row], &ticket).is_err());
        let short = InboundTicket {
            pages: pages(10..12),
            ..ticket.clone()
        };
        assert!(
            staged
                .fetch(JobId(5), vec![segment.clone()], &short)
                .is_err()
        );
        assert!(settled(&staged));

        // A commit that cannot rename into place removes the temporary file;
        // a create that cannot open its temporary file fails without an abort.
        let lost = header(&geometry, &full, 3 * PAGE_TOKENS + 4, None);
        let lost_temporary = directory.join(record::temporary_name(&lost.key));
        let missing = directory.join("missing");
        staged
            .write_from(
                JobId(6),
                lost.clone(),
                lost_temporary.clone(),
                missing.join(record::file_name(&lost.key)),
                source_addresses.clone(),
                Some(host.row_address(source_row)),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Fetched { .. }] => panic!("unexpected fetch event"),
            [TransportEvent::Stored { id, result }] => {
                assert_eq!(*id, JobId(6));
                assert!(result.as_ref().unwrap_err().to_string().contains("rename"));
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(!lost_temporary.exists());
        staged
            .write_from(
                JobId(7),
                lost.clone(),
                missing.join(record::temporary_name(&lost.key)),
                directory.join(record::file_name(&lost.key)),
                source_addresses.clone(),
                Some(host.row_address(source_row)),
                None,
            )
            .unwrap();
        match &transport_events(&mut staged, 1)[..] {
            [TransportEvent::Stored { id, result }] => {
                assert_eq!(*id, JobId(7));
                assert!(result.is_err());
            }
            other => panic!("unexpected events {other:?}"),
        }
        assert!(settled(&staged));
        assert_eq!(free(&staged), all);
        assert_eq!(
            record::scan(
                &crate::testing::directory_lock(&directory).unwrap(),
                &IDENTITY,
                &FINGERPRINT,
                geometry[0].page_tokens,
                &record::rank_sizes(&geometry)
            )
            .unwrap()
            .len(),
            1
        );
        staged.close().unwrap();
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn staged_holds_submissions_at_the_disk_queue_depth_and_finishes_every_record() {
        let Some(directory) = test_directory("disk-store", "queue-depth") else {
            return;
        };
        let geometry = geometry();
        let mut host = host_pool();
        let mut staged = Staged::new(
            None,
            stopped_transfers(),
            DiskThread::start("flare-offload-disk").unwrap(),
        );
        // Three windows, two staged at a time: a record first submits its
        // create and two page writes, then the third window with the row,
        // then the commit.
        const PAGES: usize = 2 * WINDOW_PAGES + 1;
        const RECORDS: u64 = 24;
        let sources = page_slabs(&mut host, PAGES, 1);
        let source_row = row_slab(&mut host, 7);
        let source_addresses = addresses(&host, &sources);
        let full = history(PAGES as u64 * PAGE_TOKENS + 100);
        let headers: Vec<Header> = (0..RECORDS)
            .map(|offset| header(&geometry, &full, PAGES as u64 * PAGE_TOKENS + offset, None))
            .collect();
        for (index, header) in headers.iter().enumerate() {
            staged
                .write_from(
                    JobId(index as u64 + 1),
                    header.clone(),
                    directory.join(record::temporary_name(&header.key)),
                    directory.join(record::file_name(&header.key)),
                    source_addresses.clone(),
                    Some(host.row_address(source_row)),
                    None,
                )
                .unwrap();
        }
        // Twenty-one records submit their create and both page writes, the
        // next record's create fills the queue and its page writes hold, and
        // the last two hold before their create.
        assert_eq!(staged.resources.disk_outstanding, disk::QUEUE_DEPTH);
        let (held, partial) = staged
            .outbound
            .values()
            .fold((0, 0), |(held, partial), record| {
                if record.file == TemporaryFile::Unsubmitted {
                    (held + 1, partial)
                } else if record.active.iter().any(|window| !window.pages_submitted) {
                    (held, partial + 1)
                } else {
                    (held, partial)
                }
            });
        assert_eq!((held, partial), (2, 1));
        assert_eq!(staged.in_flight(), (RECORDS as usize, 0));

        let deadline = Instant::now() + Duration::from_secs(120);
        let mut events = Vec::new();
        while events.len() < RECORDS as usize {
            assert!(
                Instant::now() < deadline,
                "{} of {RECORDS} records within 120 s",
                events.len()
            );
            staged.poll(&mut events).unwrap();
            assert!(staged.resources.disk_outstanding <= disk::QUEUE_DEPTH);
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut done: Vec<u64> = events
            .iter()
            .map(|event| match event {
                TransportEvent::Stored { id, result } => {
                    result.as_ref().unwrap();
                    id.0
                }
                other => panic!("unexpected event {other:?}"),
            })
            .collect();
        done.sort_unstable();
        assert_eq!(done, (1..=RECORDS).collect::<Vec<_>>());
        for header in &headers {
            assert_eq!(
                fs::metadata(directory.join(record::file_name(&header.key)))
                    .unwrap()
                    .len(),
                header.length_with_row()
            );
        }
        assert!(settled(&staged));
        assert_eq!(
            record::scan(
                &crate::testing::directory_lock(&directory).unwrap(),
                &IDENTITY,
                &FINGERPRINT,
                geometry[0].page_tokens,
                &record::rank_sizes(&geometry)
            )
            .unwrap()
            .len(),
            RECORDS as usize
        );
        staged.close().unwrap();
        fs::remove_dir_all(&directory).unwrap();
    }

    /// Write the record file of `header` with known pages, their checksums and a
    /// row; returns the pages and the row.
    fn write_record(directory: &Path, header: &mut Header, seed: u8) -> (Vec<Vec<u8>>, Vec<u8>) {
        let tail = header.tail_pages as usize;
        let pages: Vec<Vec<u8>> = (0..tail)
            .map(|p| {
                (0..PAGE_RECORD as usize)
                    .map(|i| (i * 3 + p + seed as usize) as u8)
                    .collect()
            })
            .collect();
        let row: Vec<u8> = (0..ROW_RECORD as usize)
            .map(|i| (i + seed as usize) as u8)
            .collect();
        header.row_checksum = record::checksum(&row);
        let mut file = vec![0u8; header.length_with_row() as usize];
        file[..record::HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
        for (index, page) in pages.iter().enumerate() {
            let at = header.page_segment as usize + index * PAGE_RECORD as usize;
            file[at..at + page.len()].copy_from_slice(page);
            let sum = header.checksum_segment() as usize + index * 8;
            file[sum..sum + 8].copy_from_slice(&record::checksum(page).to_le_bytes());
        }
        let at = header.row_segment as usize;
        file[at..at + row.len()].copy_from_slice(&row);
        fs::write(directory.join(record::file_name(&header.key)), file).unwrap();
        (pages, row)
    }

    #[test]
    fn records_on_disk_are_announced_and_exported_with_the_checksums_their_files_persist() {
        use crate::pool::{Announcer, HostWindow, Notice, Tier as PoolTier};
        let geometry = geometry();
        let directory = directory();
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let full = history(2000);
        let mut root = header(&geometry, &full, 700, None);
        let mut leaf = header(&geometry, &full, 1300, Some(700));
        let (root_pages, _) = write_record(&directory, &mut root, 1);
        let (leaf_pages, leaf_row) = write_record(&directory, &mut leaf, 2);
        let records = vec![
            scanned(&directory, root.clone(), 0, true),
            scanned(&directory, leaf.clone(), 1, true),
        ];
        let (mut store, _log) = store_with(records, 1 << 30);
        let announcer = Announcer::new(Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer.clone());
        store.flush_announcements();
        let notices = announcer.take_batch(10).unwrap();
        assert_eq!(notices.len(), 2, "both files are announced");
        assert!(
            notices
                .iter()
                .all(|n| matches!(n, Notice::Saved(e) if e.tier == PoolTier::Nvme))
        );
        assert_eq!(
            store.pin_export(PoolTier::Dram, leaf.key, 0, true).err(),
            Some(ExportRefusal::NotFound)
        );
        let mut pin = store.pin_export(PoolTier::Nvme, leaf.key, 0, true).unwrap();
        let chain = pin.chain().clone();
        assert_eq!(chain.records.len(), 2);
        assert_eq!(chain.total_pages, 20);
        assert_eq!(chain.records[0].key, root.key);
        assert!(store.index.nodes().all(|node| node.pinned()));
        // Pages and checksums come from the files.
        let window = HostWindow {
            record: 0,
            first_page: 2,
            page_count: 3,
            row: false,
        };
        let mut dst = vec![0u8; 3 * PAGE_RECORD as usize];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_eq!(dst[..PAGE_RECORD as usize], root_pages[2][..]);
        assert_eq!(sums.pages[0], record::checksum(&root_pages[2]));
        let window = HostWindow {
            record: 1,
            first_page: 9,
            page_count: 1,
            row: true,
        };
        let mut dst = vec![0u8; (PAGE_RECORD + ROW_RECORD) as usize];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_eq!(dst[PAGE_RECORD as usize..], leaf_row[..]);
        assert_eq!(sums.row, Some(record::checksum(&leaf_row)));
        // A page that rotted on disk is served as it is, with the checksum it
        // was written with, so the requester's check fails.
        let path = directory.join(record::file_name(&leaf.key));
        let mut file = fs::read(&path).unwrap();
        file[leaf.page_segment as usize + 5] ^= 1;
        fs::write(&path, file).unwrap();
        let window = HostWindow {
            record: 1,
            first_page: 0,
            page_count: 1,
            row: false,
        };
        let mut dst = vec![0u8; PAGE_RECORD as usize];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_ne!(record::checksum(&dst), sums.pages[0]);
        assert_eq!(sums.pages[0], record::checksum(&leaf_pages[0]));
        // Starting inside the leaf needs only the leaf.
        let suffix = store
            .pin_export(PoolTier::Nvme, leaf.key, 700 + 640, false)
            .unwrap();
        assert_eq!(suffix.chain().records.len(), 1);
        drop(suffix);
        // Dropping the pins releases the records at the next flush.
        drop(pin);
        store.flush_announcements();
        assert!(store.index.nodes().all(|node| !node.pinned()));
        // Nothing further changed, so nothing further is announced.
        store.flush_announcements();
        assert!(announcer.take_batch(10).is_none());
    }

    fn announced(store: &mut DiskStore) -> crate::pool::Announcer {
        let announcer = crate::pool::Announcer::new(Arc::new(tokio::sync::Notify::new()));
        announcer.begin_session();
        store.announce_to(announcer.clone());
        store.flush_announcements();
        announcer
    }

    #[test]
    fn a_spilling_store_holds_the_coldest_leaf_until_its_move_ends() {
        let full = history(1000);
        let cold = header(&geometry(), &full, 128, None);
        let warm = header(&geometry(), &full, 256, None);
        let limit = cold.length_with_row() + warm.length_with_row();
        let records = vec![
            scanned(&directory(), cold.clone(), 0, true),
            scanned(&directory(), warm.clone(), 1, true),
        ];
        let (mut store, log) = store_with(records, limit);
        let _announcer = announced(&mut store);
        store.set_spill(true);

        // Room for one more page: the coldest file is held, nothing is unlinked.
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        let retiring = store.take_retiring();
        assert_eq!(retiring.len(), 1);
        assert_eq!(
            (retiring[0].key, retiring[0].tier, retiring[0].row),
            (cold.key, PoolTier::Nvme, true)
        );
        assert!(unlinked(&log).is_empty());
        assert!(store.index.node(&cold.key).unwrap().pinned());

        // The held file already covers the request: nothing more is chosen.
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        assert!(store.take_retiring().is_empty());

        // A failed move keeps the file, which is offered again.
        crate::store::TierStore::retired(&mut store, cold.key, PoolTier::Nvme, Retirement::Failed);
        assert!(unlinked(&log).is_empty());
        assert!(!store.index.node(&cold.key).unwrap().pinned());
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        assert_eq!(store.take_retiring()[0].key, cold.key);

        // Once moved, the file is unlinked and its bytes are free.
        crate::store::TierStore::retired(&mut store, cold.key, PoolTier::Nvme, Retirement::Moved);
        assert_eq!(unlinked(&log).len(), 1);
        assert!(!store.index.contains(&cold.key));
        assert_eq!(store.take_evicted(), [cold.key]);
        assert!(store.make_room(PAGE_RECORD).unwrap());
    }

    #[test]
    fn a_busy_or_unavailable_manager_leaves_local_lru_as_the_end() {
        let full = history(1000);
        let cold = header(&geometry(), &full, 128, None);
        let warm = header(&geometry(), &full, 256, None);
        let limit = cold.length_with_row() + warm.length_with_row();
        let (mut store, log) = store_with(
            vec![
                scanned(&directory(), cold.clone(), 0, true),
                scanned(&directory(), warm.clone(), 1, true),
            ],
            limit,
        );
        let _announcer = announced(&mut store);
        store.set_spill(true);
        store.maintain();
        let selected = store.take_retiring();
        assert!(!selected.is_empty());
        for r in selected {
            store.retired(&r.key, Retirement::Failed).unwrap();
        }
        store.set_spill(false);
        assert!(store.index.contains(&cold.key));
        assert!(unlinked(&log).is_empty());
        assert!(store.make_room(PAGE_RECORD).unwrap());
        assert!(!store.index.contains(&cold.key));
        assert!(store.index.contains(&warm.key));
        assert_eq!(unlinked(&log).len(), 1);
        assert_eq!(store.stats().dropped[3], 1);
        assert!(store.stats().held[1] > 0);
    }

    #[test]
    fn a_store_that_is_not_spilling_unlinks_the_coldest_leaf_as_before() {
        let full = history(1000);
        let cold = header(&geometry(), &full, 128, None);
        let warm = header(&geometry(), &full, 256, None);
        let limit = cold.length_with_row() + warm.length_with_row();
        let records = vec![
            scanned(&directory(), cold.clone(), 0, true),
            scanned(&directory(), warm.clone(), 1, true),
        ];
        let (mut store, log) = store_with(records, limit);
        let _announcer = announced(&mut store);
        // Spilling was on and the manager went away.
        store.set_spill(true);
        store.set_spill(false);
        assert!(store.make_room(PAGE_RECORD).unwrap());
        assert!(store.take_retiring().is_empty());
        assert_eq!(unlinked(&log).len(), 1);
        assert!(!store.index.contains(&cold.key));
    }

    #[test]
    fn a_moved_record_stays_until_the_pin_that_reads_it_is_released() {
        use crate::pool::Tier as PoolTier;
        let geometry = geometry();
        let directory = directory();
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let full = history(2000);
        let mut leaf = header(&geometry, &full, 700, None);
        write_record(&directory, &mut leaf, 1);
        let (mut store, log) = store_with(
            vec![scanned(&directory, leaf.clone(), 0, true)],
            leaf.length_with_row(),
        );
        let _announcer = announced(&mut store);
        store.set_spill(true);
        assert!(!store.make_room(PAGE_RECORD).unwrap());
        assert_eq!(store.take_retiring().remove(0).key, leaf.key);
        let pin = store.pin_export(PoolTier::Nvme, leaf.key, 0, true).unwrap();

        crate::store::TierStore::retired(&mut store, leaf.key, PoolTier::Nvme, Retirement::Moved);
        assert!(unlinked(&log).is_empty(), "a pin still reads the file");
        drop(pin);
        store.flush_announcements();
        assert_eq!(unlinked(&log).len(), 1);
        assert!(!store.index.contains(&leaf.key));
    }

    #[test]
    fn a_publication_starts_the_disk_copy() {
        let (mut store, log) = store_with(Vec::new(), 1 << 30);
        let full = history(1000);
        let mut ids = Ids::new();
        let publication = checkpoint(&mut ids, &full, 128, None, pages(0..2), 2, StateSlotId(1));
        assert_eq!(store.store(&publication).unwrap(), StoreOutcome::Queued);
        assert_eq!(log.borrow().calls.len(), 1);
    }
}
