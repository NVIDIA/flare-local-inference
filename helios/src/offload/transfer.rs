//! Per-rank threads that move checkpoint bytes between GPU ids and host slabs.
//!
//! Each thread makes its rank's context current, owns one stream at the lowest
//! priority and one blocking event, and receives jobs naming page ids, a state
//! slot, and slab addresses. The thread expands ids into device and host
//! addresses from its rank's geometry and the cross-rank [`Layout`], or from
//! its [`RankPart`] for the pool's device addresses, issues
//! batched copies of at most [`BATCH_BYTES`] and [`BATCH_ENTRIES`] per call,
//! records its event, sleeps until the event completes, and replies. A failure
//! is reported only after the stream has drained, so destination ids are
//! quiescent when the scheduler frees them. The model stream never waits on
//! these streams: host-observed completion orders later model work after the
//! copies.
//!
//! Copies within one batch execute in no defined order, so a job may not name
//! a page id or a slab address twice: either would be two writes to one range
//! in one direction.
//!
//! Descriptor arrays are sized at startup for one window and one batch and
//! reused for every job; steady-state jobs allocate nothing on the GPU side
//! and only clone the job message per rank on the submitting thread.

use std::{
    collections::HashMap,
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError},
    thread::JoinHandle,
    time::Instant,
};

use flare_selene::{
    CLOSE_TIMEOUT, CacheGeometry, Direction, Error, Job, JobId, Layout, PageCopier, QUEUE_DEPTH,
    Reply, Result,
    geometry::{RankLayout, invalid},
    slab::{RankPart, SlabPool, WINDOW_PAGES},
};

use crate::cuda::{Event, Stream};

pub use crate::cuda::{BATCH_BYTES, BATCH_ENTRIES};

pub const NICE: i32 = 10;

pub struct Transfers {
    layout: Layout,
    ranks: Vec<RankThread>,
    receiver: Receiver<RankReply>,
    pending: HashMap<(JobId, u32), Pending>,
}

struct RankThread {
    sender: Option<SyncSender<Job>>,
    thread: Option<JoinHandle<()>>,
    /// Jobs sent to this rank whose replies have not been received.
    outstanding: usize,
}

struct RankReply {
    id: JobId,
    window: u32,
    rank: usize,
    result: Result<()>,
    bytes: u64,
}

struct Pending {
    direction: Direction,
    remaining: usize,
    bytes: u64,
    error: Option<Error>,
    started: Instant,
}

/// One contiguous copy of at most [`BATCH_BYTES`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Descriptor {
    pub destination: u64,
    pub source: u64,
    pub bytes: usize,
}

/// Expand a job's ids into copy descriptors for one rank: pages in job order
/// with planes in geometry order, then the row. A unit larger than
/// [`BATCH_BYTES`] becomes several descriptors. Ids outside the rank's
/// capacities, a null slab address, a page id or slab address named twice, or
/// more than [`WINDOW_PAGES`] pages are rejected without touching the driver.
///
/// `part` resolves a slab identity to this rank's GPU address, with the planes
/// at their offsets relative to the rank's first plane. Without a part, the
/// addresses describe an already GPU-addressable cross-rank record.
pub(crate) fn expand(
    geometry: &CacheGeometry,
    rank: &RankLayout,
    part: Option<&RankPart>,
    job: &Job,
    descriptors: &mut Vec<Descriptor>,
) -> Result<()> {
    descriptors.clear();
    if job.pages.len() > WINDOW_PAGES {
        return Err(invalid(format!(
            "transfer job {} window {} names {} pages; a window holds at most {WINDOW_PAGES}",
            job.id.0,
            job.window,
            job.pages.len()
        )));
    }
    if rank.plane_offsets.len() != geometry.planes.len() {
        return Err(invalid(
            "record layout and cache geometry disagree on the plane count",
        ));
    }
    reject_repeats(job)?;
    for (page, slab) in &job.pages {
        let id = u64::from(page.0);
        if id >= geometry.page_capacity {
            return Err(invalid(format!(
                "page id {id} is outside the cache's {} pages",
                geometry.page_capacity
            )));
        }
        let record = match part {
            Some(part) => part
                .page(*slab)?
                .ok_or_else(|| invalid("page slab is outside the transfer pool"))?,
            None => slab_address(*slab)? + rank.page_offset,
        };
        for (plane, offset) in geometry.planes.iter().zip(&rank.plane_offsets) {
            push_unit(
                job.direction,
                plane.unit_address(id),
                record + offset,
                plane.unit_bytes,
                descriptors,
            );
        }
    }
    if let Some((slot, slab)) = job.row {
        let index = u64::from(slot.0);
        if index >= geometry.row_capacity {
            return Err(invalid(format!(
                "state slot {index} is outside the cache's {} rows",
                geometry.row_capacity
            )));
        }
        let row = match part {
            Some(part) => part
                .row(slab)?
                .ok_or_else(|| invalid("row slab is outside the transfer pool"))?,
            None => slab_address(slab)? + rank.row_offset,
        };
        push_unit(
            job.direction,
            geometry.row.unit_address(index),
            row,
            geometry.row.unit_bytes,
            descriptors,
        );
    }
    Ok(())
}

/// A page id or a slab address named twice would place two copies of one
/// range in a batch whose entries execute in no defined order.
fn reject_repeats(job: &Job) -> Result<()> {
    for (index, (page, slab)) in job.pages.iter().enumerate() {
        if job.pages[..index]
            .iter()
            .any(|(earlier, _)| earlier == page)
        {
            return Err(invalid(format!(
                "transfer job {} window {} names page id {} twice",
                job.id.0, job.window, page.0
            )));
        }
        if job.pages[..index]
            .iter()
            .any(|(_, earlier)| earlier == slab)
        {
            return Err(invalid(format!(
                "transfer job {} window {} names slab {slab:#x} twice",
                job.id.0, job.window
            )));
        }
    }
    match job.row {
        Some((_, row_slab)) if job.pages.iter().any(|(_, slab)| *slab == row_slab) => {
            Err(invalid(format!(
                "transfer job {} window {} names slab {row_slab:#x} for both a page and the row",
                job.id.0, job.window
            )))
        }
        _ => Ok(()),
    }
}

fn slab_address(slab: usize) -> Result<u64> {
    if slab == 0 {
        return Err(invalid("transfer job names a null slab address"));
    }
    Ok(slab as u64)
}

fn push_unit(
    direction: Direction,
    device: u64,
    host: u64,
    bytes: u64,
    descriptors: &mut Vec<Descriptor>,
) {
    let mut offset = 0;
    while offset < bytes {
        let chunk = (bytes - offset).min(BATCH_BYTES as u64);
        let (destination, source) = match direction {
            Direction::Out => (host + offset, device + offset),
            Direction::In => (device + offset, host + offset),
        };
        descriptors.push(Descriptor {
            destination,
            source,
            bytes: chunk as usize,
        });
        offset += chunk;
    }
}

/// End of the longest run of descriptors from `start` that one batched copy may carry.
pub(crate) fn batch_end(descriptors: &[Descriptor], start: usize) -> usize {
    let mut end = start;
    let mut bytes = 0;
    while end < descriptors.len()
        && end - start < BATCH_ENTRIES
        && bytes + descriptors[end].bytes <= BATCH_BYTES
    {
        bytes += descriptors[end].bytes;
        end += 1;
    }
    end
}

/// Lower scheduling priority where the host permits it.
fn lower_priority() {
    crate::os::lower_priority(NICE);
}

struct RankWorker {
    rank: usize,
    geometry: CacheGeometry,
    layout: RankLayout,
    /// Resolves slab identities to GPU addresses; without it host addresses are
    /// already GPU-addressable, as for memory registered with identical addresses.
    part: Option<RankPart>,
    stream: Stream,
    event: Event,
    descriptors: Vec<Descriptor>,
    destinations: Vec<u64>,
    sources: Vec<u64>,
    sizes: Vec<usize>,
}

impl RankWorker {
    /// Bind the rank's context to the calling thread and size the descriptor
    /// arrays for one window and one batch.
    fn open(
        rank: usize,
        context: usize,
        device: i32,
        geometry: CacheGeometry,
        layout: RankLayout,
        part: Option<RankPart>,
    ) -> Result<Self> {
        let stream = Stream::open_lowest_priority(context, device)?;
        let event = stream.blocking_event()?;
        let chunks = |bytes: u64| bytes.div_ceil(BATCH_BYTES as u64) as usize;
        let window = geometry
            .planes
            .iter()
            .map(|plane| chunks(plane.unit_bytes))
            .sum::<usize>()
            * WINDOW_PAGES
            + chunks(geometry.row.unit_bytes);
        Ok(Self {
            rank,
            geometry,
            layout,
            part,
            stream,
            event,
            descriptors: Vec::with_capacity(window),
            destinations: Vec::with_capacity(BATCH_ENTRIES),
            sources: Vec::with_capacity(BATCH_ENTRIES),
            sizes: Vec::with_capacity(BATCH_ENTRIES),
        })
    }

    /// Issue every batch of the job, then wait for the last one; returns the bytes copied.
    fn run(&mut self, job: &Job) -> Result<u64> {
        expand(
            &self.geometry,
            &self.layout,
            self.part.as_ref(),
            job,
            &mut self.descriptors,
        )?;
        let mut start = 0;
        let mut bytes = 0u64;
        while start < self.descriptors.len() {
            let end = batch_end(&self.descriptors, start);
            if end == start {
                return Err(invalid("copy descriptor exceeds the batch byte limit"));
            }
            self.destinations.clear();
            self.sources.clear();
            self.sizes.clear();
            for descriptor in &self.descriptors[start..end] {
                self.destinations.push(descriptor.destination);
                self.sources.push(descriptor.source);
                self.sizes.push(descriptor.bytes);
                bytes += descriptor.bytes as u64;
            }
            // SAFETY: Device ranges lie inside the rank's cache allocations by
            // the id checks in `expand`, and the submitting store keeps every
            // slab mapped, registered, and unchanged until this job's reply.
            unsafe {
                self.stream
                    .copy_batch(&self.destinations, &self.sources, &self.sizes)?;
            }
            start = end;
        }
        self.stream.record(&self.event)?;
        self.event.synchronize()?;
        Ok(bytes)
    }

    /// Wait for every copy issued so far, sleeping on the blocking event where
    /// it can be recorded and spinning on the stream otherwise.
    fn drain(&self) {
        if self.stream.record(&self.event).is_err() || self.event.synchronize().is_err() {
            let _ = self.stream.synchronize();
        }
    }

    fn serve(mut self, jobs: Receiver<Job>, replies: Sender<RankReply>) {
        for job in jobs {
            let (result, bytes) = match self.run(&job) {
                Ok(bytes) => (Ok(()), bytes),
                Err(error) => {
                    // Drain the stream before reporting, so destination ids
                    // are quiescent when the caller frees them.
                    self.drain();
                    (Err(error), 0)
                }
            };
            let reply = RankReply {
                id: job.id,
                window: job.window,
                rank: self.rank,
                result,
                bytes,
            };
            if replies.send(reply).is_err() {
                break;
            }
        }
        self.drain();
    }
}

impl Transfers {
    /// Start one thread per rank. `contexts` pairs each rank's CUDA context with
    /// its device ordinal, in rank order matching `geometry`. The pool supplies
    /// device addresses for registered slabs or storage split by rank, and must
    /// remain alive until the transfer threads have drained.
    pub fn start(
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        pool: &SlabPool,
    ) -> Result<Self> {
        if contexts.is_empty() || contexts.len() != geometry.len() {
            return Err(invalid(format!(
                "transfer threads need one context per rank: {} contexts for {} ranks",
                contexts.len(),
                geometry.len()
            )));
        }
        let layout = Layout::new(&geometry)?;
        let parts = pool.device_parts(&layout)?.into_iter().map(Some).collect();
        Self::start_parts(contexts, geometry, layout, parts)
    }

    /// Start threads that copy to and from host memory registered with identical
    /// host and device addresses, which a job names directly. The shared pool
    /// regions map and register during startup, so no slab pool describes them.
    pub fn start_unmapped(contexts: &[(usize, i32)], geometry: Vec<CacheGeometry>) -> Result<Self> {
        if contexts.is_empty() || contexts.len() != geometry.len() {
            return Err(invalid(format!(
                "transfer threads need one context per rank: {} contexts for {} ranks",
                contexts.len(),
                geometry.len()
            )));
        }
        let layout = Layout::new(&geometry)?;
        let parts = vec![None; contexts.len()];
        Self::start_parts(contexts, geometry, layout, parts)
    }

    fn start_parts(
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        layout: Layout,
        parts: Vec<Option<RankPart>>,
    ) -> Result<Self> {
        let (reply_sender, receiver) = mpsc::channel();
        let mut transfers = Self {
            layout,
            ranks: Vec::with_capacity(contexts.len()),
            receiver,
            pending: HashMap::new(),
        };
        for (rank, ((&(context, device), geometry), part)) in
            contexts.iter().zip(geometry).zip(parts).enumerate()
        {
            let layout = transfers.layout.ranks[rank].clone();
            let (sender, jobs) = mpsc::sync_channel(QUEUE_DEPTH);
            let (ready_sender, ready) = mpsc::channel();
            let replies = reply_sender.clone();
            let thread = std::thread::Builder::new()
                .name(format!("flare-offload-{rank}"))
                .spawn(move || {
                    let worker =
                        match RankWorker::open(rank, context, device, geometry, layout, part) {
                            Ok(worker) => worker,
                            Err(error) => {
                                let _ = ready_sender.send(Err(error));
                                return;
                            }
                        };
                    lower_priority();
                    if ready_sender.send(Ok(())).is_err() {
                        return;
                    }
                    worker.serve(jobs, replies);
                })
                .map_err(|error| {
                    Error::caused_by(
                        format!("cannot spawn the offload thread for rank {rank}"),
                        error,
                    )
                })?;
            transfers.ranks.push(RankThread {
                sender: Some(sender),
                thread: Some(thread),
                outstanding: 0,
            });
            match ready.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(error),
                Err(_) => {
                    return Err(invalid(format!(
                        "offload thread for rank {rank} stopped during startup"
                    )));
                }
            }
        }
        Ok(transfers)
    }
}

impl PageCopier for Transfers {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Queue the job on every rank without waiting.
    ///
    /// `Err` means no rank accepted the job and no reply follows: a rank's
    /// queue already holds [`QUEUE_DEPTH`] jobs, a thread has stopped, `close`
    /// ran, or the (id, window) pair is already in flight. After `Ok`, exactly
    /// one [`Reply`] follows from `poll`. A thread that stops between that
    /// check and its send counts as a failed rank reply: the ranks that did
    /// accept the job complete their copies and the reply carries the error.
    fn submit(&mut self, job: Job) -> Result<()> {
        let key = (job.id, job.window);
        if self.pending.contains_key(&key) {
            return Err(invalid(format!(
                "transfer job {} window {} is already in flight",
                job.id.0, job.window
            )));
        }
        for (rank, thread) in self.ranks.iter().enumerate() {
            if thread.sender.is_none() || thread.thread.as_ref().is_none_or(JoinHandle::is_finished)
            {
                return Err(invalid(format!(
                    "offload thread for rank {rank} has stopped"
                )));
            }
            if thread.outstanding >= QUEUE_DEPTH {
                return Err(invalid(format!(
                    "offload queue for rank {rank} holds {QUEUE_DEPTH} jobs"
                )));
            }
        }
        let mut pending = Pending {
            direction: job.direction,
            remaining: self.ranks.len(),
            bytes: 0,
            error: None,
            started: Instant::now(),
        };
        let mut accepted = 0;
        if let Some((last, rest)) = self.ranks.split_last_mut() {
            for (rank, thread) in rest.iter_mut().enumerate() {
                accepted += usize::from(thread.offer(rank, job.clone(), &mut pending));
            }
            accepted += usize::from(last.offer(rest.len(), job, &mut pending));
        }
        if accepted == 0 {
            return Err(pending
                .error
                .unwrap_or_else(|| invalid("no offload threads are running")));
        }
        self.pending.insert(key, pending);
        Ok(())
    }

    /// Collect replies whose every rank has completed.
    fn poll(&mut self, replies: &mut Vec<Reply>) {
        while let Ok(reply) = self.receiver.try_recv() {
            self.absorb(reply, replies);
        }
    }

    fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Wait up to [`CLOSE_TIMEOUT`] for outstanding jobs, discarding their
    /// replies, then stop and join every thread. Jobs still outstanding at
    /// the timeout leave the threads running and are reported as an error;
    /// dropping `Transfers` joins the threads once those jobs finish.
    fn close(&mut self) -> Result<()> {
        let deadline = Instant::now() + CLOSE_TIMEOUT;
        let mut discarded = Vec::new();
        while !self.pending.is_empty() {
            let wait = deadline.saturating_duration_since(Instant::now());
            match self.receiver.recv_timeout(wait) {
                Ok(reply) => self.absorb(reply, &mut discarded),
                Err(RecvTimeoutError::Timeout) => {
                    return Err(invalid(format!(
                        "{} transfer jobs did not complete within {:?}",
                        self.pending.len(),
                        CLOSE_TIMEOUT
                    )));
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        for rank in &mut self.ranks {
            rank.sender.take();
        }
        let mut stopped = Vec::new();
        for (rank, thread) in self.ranks.iter_mut().enumerate() {
            if thread
                .thread
                .take()
                .is_some_and(|thread| thread.join().is_err())
            {
                stopped.push(rank);
            }
        }
        if !stopped.is_empty() {
            return Err(invalid(format!(
                "offload threads for ranks {stopped:?} panicked"
            )));
        }
        if !self.pending.is_empty() {
            return Err(invalid(format!(
                "offload threads stopped with {} transfer jobs outstanding",
                self.pending.len()
            )));
        }
        Ok(())
    }
}

impl Transfers {
    fn absorb(&mut self, reply: RankReply, replies: &mut Vec<Reply>) {
        if let Some(thread) = self.ranks.get_mut(reply.rank) {
            thread.outstanding = thread.outstanding.saturating_sub(1);
        }
        let key = (reply.id, reply.window);
        let Some(pending) = self.pending.get_mut(&key) else {
            return;
        };
        pending.bytes += reply.bytes;
        if let Err(error) = reply.result {
            pending.error.get_or_insert(error);
        }
        pending.remaining = pending.remaining.saturating_sub(1);
        if pending.remaining > 0 {
            return;
        }
        if let Some(pending) = self.pending.remove(&key) {
            replies.push(Reply {
                id: reply.id,
                window: reply.window,
                direction: pending.direction,
                result: pending.error.map_or(Ok(()), Err),
                bytes: pending.bytes,
                elapsed: pending.started.elapsed(),
            });
        }
    }
}

impl RankThread {
    /// Send without blocking; a refusal is recorded on `pending` as this rank's reply.
    fn offer(&mut self, rank: usize, job: Job, pending: &mut Pending) -> bool {
        let refusal = match self.sender.as_ref().map(|sender| sender.try_send(job)) {
            Some(Ok(())) => {
                self.outstanding += 1;
                return true;
            }
            Some(Err(TrySendError::Full(_))) => "its queue is full",
            Some(Err(TrySendError::Disconnected(_))) | None => "its thread has stopped",
        };
        pending.remaining = pending.remaining.saturating_sub(1);
        pending.error.get_or_insert_with(|| {
            invalid(format!(
                "rank {rank} did not accept the transfer job: {refusal}"
            ))
        });
        false
    }
}

impl Drop for Transfers {
    fn drop(&mut self) {
        for rank in &mut self.ranks {
            rank.sender.take();
        }
        for rank in &mut self.ranks {
            if let Some(thread) = rank.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use flare_selene::{
        PageId, StateSlotId,
        testing::{geometry, pattern as fill, rank_part},
    };

    fn job(direction: Direction) -> Job {
        Job {
            id: JobId(9),
            window: 2,
            direction,
            pages: vec![(PageId(5), 0x10000), (PageId(7), 0x20000)],
            row: Some((StateSlotId(3), 0x30000)),
        }
    }

    #[test]
    fn registered_parts_use_device_aliases_with_cross_rank_record_strides() {
        let ranks = [
            geometry(64, &[("kv", 8192), ("selector", 100)], 10_000),
            geometry(64, &[("kv", 4096), ("selector", 200)], 5000),
        ];
        let layout = Layout::new(&ranks).unwrap();
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            3,
            layout.row_record_bytes,
            2,
            false,
        )
        .unwrap();
        assert!(pool.device_parts(&layout).is_err());
        let device = 0x2_0420_0000;
        let parts = pool.registered_parts(&layout, device);
        let pages: Vec<_> = (0..3).map(|_| pool.allocate_page().unwrap()).collect();
        let rows: Vec<_> = (0..2).map(|_| pool.allocate_row().unwrap()).collect();
        for (rank, part) in parts.iter().enumerate() {
            let mut job = Job {
                id: JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: vec![(PageId(5), pool.page_address(pages[2]))],
                row: Some((StateSlotId(3), pool.row_address(rows[1]))),
            };
            let mut outbound = Vec::new();
            expand(
                &ranks[rank],
                &layout.ranks[rank],
                Some(part),
                &job,
                &mut outbound,
            )
            .unwrap();
            let page = device + 2 * layout.page_record_bytes + layout.ranks[rank].page_offset;
            let row = device
                + 3 * layout.page_record_bytes
                + layout.row_record_bytes
                + layout.ranks[rank].row_offset;
            assert_eq!(
                outbound.iter().map(|d| d.destination).collect::<Vec<_>>(),
                vec![page, page + ranks[rank].planes[0].padded_bytes(), row]
            );
            job.direction = Direction::In;
            let mut inbound = Vec::new();
            expand(
                &ranks[rank],
                &layout.ranks[rank],
                Some(part),
                &job,
                &mut inbound,
            )
            .unwrap();
            for (out, incoming) in outbound.iter().zip(&inbound) {
                assert_eq!(
                    (incoming.source, incoming.destination, incoming.bytes),
                    (out.destination, out.source, out.bytes)
                );
            }
        }
    }

    #[test]
    fn expansion_addresses_every_plane_and_the_row() {
        let ranks = [
            geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
            geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
        ];
        let layout = Layout::new(&ranks).unwrap();
        let mut descriptors = Vec::new();
        expand(
            &ranks[1],
            &layout.ranks[1],
            None,
            &job(Direction::Out),
            &mut descriptors,
        )
        .unwrap();
        assert_eq!(descriptors.len(), 7);
        let rank = &layout.ranks[1];
        for (page, (id, slab)) in [(5u64, 0x10000u64), (7, 0x20000)].into_iter().enumerate() {
            for (plane, geometry) in ranks[1].planes.iter().enumerate() {
                assert_eq!(
                    descriptors[page * 3 + plane],
                    Descriptor {
                        destination: slab + rank.page_offset + rank.plane_offsets[plane],
                        source: geometry.unit_address(id),
                        bytes: geometry.unit_bytes as usize,
                    }
                );
            }
        }
        assert_eq!(
            descriptors[6],
            Descriptor {
                destination: 0x30000 + rank.row_offset,
                source: ranks[1].row.unit_address(3),
                bytes: 10_000,
            }
        );
        let outbound = descriptors.clone();
        expand(
            &ranks[1],
            &layout.ranks[1],
            None,
            &job(Direction::In),
            &mut descriptors,
        )
        .unwrap();
        for (inbound, outbound) in descriptors.iter().zip(&outbound) {
            assert_eq!(inbound.destination, outbound.source);
            assert_eq!(inbound.source, outbound.destination);
            assert_eq!(inbound.bytes, outbound.bytes);
        }
        let total: usize = descriptors.iter().map(|d| d.bytes).sum();
        assert_eq!(total as u64, 2 * ranks[1].page_bytes() + 10_000);
    }

    #[test]
    fn expansion_places_split_slabs_in_the_rank_part_and_rejects_other_pools() {
        let ranks = [
            geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
            geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
        ];
        let layout = Layout::new(&ranks).unwrap();
        let rank = &layout.ranks[1];
        let (page_part, row_part) = (20480usize, 12288usize);
        let (page_record, row_record) = (
            layout.page_record_bytes as usize,
            layout.row_record_bytes as usize,
        );
        let start = 0x7e00_0000_0000usize;
        let (page_base, row_base) = (0x7e00_0100_0000usize, 0x7e00_0200_0000usize);
        let part = rank_part(
            start..start + (8 << 20),
            (page_record, 8, page_base, page_part),
            (row_record, 2, row_base, row_part),
        );
        let rows = start + 8 * page_record;
        let job = Job {
            id: JobId(4),
            window: 0,
            direction: Direction::Out,
            pages: vec![
                (PageId(5), start + 3 * page_record),
                (PageId(7), start + 4 * page_record),
            ],
            row: Some((StateSlotId(3), rows + row_record)),
        };
        let mut descriptors = Vec::new();
        expand(&ranks[1], rank, Some(&part), &job, &mut descriptors).unwrap();
        assert_eq!(descriptors.len(), 7);
        let records = [
            (5u64, (page_base + 3 * page_part) as u64),
            (7, (page_base + 4 * page_part) as u64),
        ];
        for (page, (id, record)) in records.into_iter().enumerate() {
            for (plane, geometry) in ranks[1].planes.iter().enumerate() {
                assert_eq!(
                    descriptors[page * 3 + plane],
                    Descriptor {
                        destination: record + rank.plane_offsets[plane],
                        source: geometry.unit_address(id),
                        bytes: geometry.unit_bytes as usize,
                    }
                );
            }
        }
        assert_eq!(
            descriptors[6],
            Descriptor {
                destination: (row_base + row_part) as u64,
                source: ranks[1].row.unit_address(3),
                bytes: 10_000,
            }
        );
        let mut inside = job.clone();
        inside.pages[1].1 = start + 4096;
        let error = expand(&ranks[1], rank, Some(&part), &inside, &mut descriptors)
            .unwrap_err()
            .to_string();
        assert!(error.contains("page slabs"), "{error}");
        let mut crossed = job.clone();
        crossed.row = Some((StateSlotId(3), start));
        let error = expand(&ranks[1], rank, Some(&part), &crossed, &mut descriptors)
            .unwrap_err()
            .to_string();
        assert!(error.contains("row slabs"), "{error}");
        let mut outside = job.clone();
        outside.pages[1].1 = 0x10000;
        assert!(expand(&ranks[1], rank, Some(&part), &outside, &mut descriptors).is_err());
        outside = job;
        outside.row = Some((StateSlotId(3), 0x10000));
        assert!(expand(&ranks[1], rank, Some(&part), &outside, &mut descriptors).is_err());
    }

    #[test]
    fn expansion_rejects_bad_ids_null_slabs_and_oversize_windows() {
        let ranks = [geometry(64, &[("k", 8192)], 4096)];
        let layout = Layout::new(&ranks).unwrap();
        let mut descriptors = Vec::new();
        let mut bad_page = job(Direction::Out);
        bad_page.pages[1].0 = PageId(ranks[0].page_capacity as u32);
        assert!(
            expand(
                &ranks[0],
                &layout.ranks[0],
                None,
                &bad_page,
                &mut descriptors
            )
            .is_err()
        );
        let mut bad_slot = job(Direction::Out);
        bad_slot.row = Some((StateSlotId(ranks[0].row_capacity as u32), 0x30000));
        assert!(
            expand(
                &ranks[0],
                &layout.ranks[0],
                None,
                &bad_slot,
                &mut descriptors
            )
            .is_err()
        );
        let mut null_slab = job(Direction::In);
        null_slab.pages[0].1 = 0;
        assert!(
            expand(
                &ranks[0],
                &layout.ranks[0],
                None,
                &null_slab,
                &mut descriptors
            )
            .is_err()
        );
        let mut wide = job(Direction::Out);
        wide.pages = (0..=WINDOW_PAGES as u32)
            .map(|id| (PageId(id), 0x100000 + id as usize * 0x10000))
            .collect();
        assert!(expand(&ranks[0], &layout.ranks[0], None, &wide, &mut descriptors).is_err());
        wide.pages.pop();
        expand(&ranks[0], &layout.ranks[0], None, &wide, &mut descriptors).unwrap();
        assert_eq!(descriptors.len(), WINDOW_PAGES + 1);
        let mut empty = job(Direction::Out);
        empty.pages.clear();
        empty.row = None;
        expand(&ranks[0], &layout.ranks[0], None, &empty, &mut descriptors).unwrap();
        assert!(descriptors.is_empty());
    }

    #[test]
    fn expansion_rejects_repeated_page_ids_and_slabs() {
        let ranks = [geometry(64, &[("k", 8192)], 4096)];
        let layout = Layout::new(&ranks).unwrap();
        let mut descriptors = Vec::new();
        let mut check = |job: &Job| {
            expand(&ranks[0], &layout.ranks[0], None, job, &mut descriptors)
                .map_err(|error| error.to_string())
        };
        for direction in [Direction::Out, Direction::In] {
            let mut repeated_id = job(direction);
            repeated_id.pages[1].0 = repeated_id.pages[0].0;
            let error = check(&repeated_id).unwrap_err();
            assert!(error.contains("page id 5 twice"), "{error}");
            let mut repeated_slab = job(direction);
            repeated_slab.pages[1].1 = repeated_slab.pages[0].1;
            let error = check(&repeated_slab).unwrap_err();
            assert!(error.contains("slab 0x10000 twice"), "{error}");
            let mut row_on_page_slab = job(direction);
            row_on_page_slab.row = Some((StateSlotId(3), 0x20000));
            let error = check(&row_on_page_slab).unwrap_err();
            assert!(error.contains("both a page and the row"), "{error}");
            check(&job(direction)).unwrap();
        }
    }

    #[test]
    fn units_larger_than_a_batch_split_and_batches_respect_both_bounds() {
        let row = 2 * BATCH_BYTES as u64 + 4096;
        let ranks = [geometry(64, &[("k", 8192)], row)];
        let layout = Layout::new(&ranks).unwrap();
        let mut descriptors = Vec::new();
        let mut only_row = job(Direction::Out);
        only_row.pages.clear();
        expand(
            &ranks[0],
            &layout.ranks[0],
            None,
            &only_row,
            &mut descriptors,
        )
        .unwrap();
        let base = ranks[0].row.unit_address(3);
        assert_eq!(
            descriptors
                .iter()
                .map(|d| (d.source - base, d.destination - 0x30000, d.bytes))
                .collect::<Vec<_>>(),
            [
                (0, 0, BATCH_BYTES),
                (BATCH_BYTES as u64, BATCH_BYTES as u64, BATCH_BYTES),
                (2 * BATCH_BYTES as u64, 2 * BATCH_BYTES as u64, 4096),
            ]
        );
        assert_eq!(batch_end(&descriptors, 0), 1);
        assert_eq!(batch_end(&descriptors, 1), 2);
        assert_eq!(batch_end(&descriptors, 2), 3);
        assert_eq!(batch_end(&descriptors, 3), 3);
        let unit = |bytes| Descriptor {
            destination: 0,
            source: 0,
            bytes,
        };
        let halves = [
            unit(BATCH_BYTES / 2),
            unit(BATCH_BYTES / 2),
            unit(1),
            unit(BATCH_BYTES),
        ];
        assert_eq!(batch_end(&halves, 0), 2);
        assert_eq!(batch_end(&halves, 2), 3);
        assert_eq!(batch_end(&halves, 3), 4);
        let many = vec![unit(1); BATCH_ENTRIES + 5];
        assert_eq!(batch_end(&many, 0), BATCH_ENTRIES);
        assert_eq!(batch_end(&many, BATCH_ENTRIES), BATCH_ENTRIES + 5);
    }

    fn detached(ranks: usize) -> (Transfers, Sender<RankReply>) {
        let (sender, receiver) = mpsc::channel();
        let layout = Layout::new(&[geometry(64, &[("k", 8192)], 4096)]).unwrap();
        let transfers = Transfers {
            layout,
            ranks: (0..ranks)
                .map(|_| RankThread {
                    sender: None,
                    thread: None,
                    outstanding: 1,
                })
                .collect(),
            receiver,
            pending: HashMap::new(),
        };
        (transfers, sender)
    }

    #[test]
    fn poll_merges_rank_replies_and_keeps_the_first_error() {
        let (mut transfers, sender) = detached(2);
        let started = Instant::now();
        for (id, window) in [(JobId(1), 0), (JobId(1), 1)] {
            transfers.pending.insert(
                (id, window),
                Pending {
                    direction: Direction::Out,
                    remaining: 2,
                    bytes: 0,
                    error: None,
                    started,
                },
            );
        }
        let reply = |id, window, rank, result, bytes| RankReply {
            id,
            window,
            rank,
            result,
            bytes,
        };
        let mut replies = Vec::new();
        sender.send(reply(JobId(1), 0, 1, Ok(()), 100)).unwrap();
        sender.send(reply(JobId(1), 1, 0, Ok(()), 7)).unwrap();
        transfers.poll(&mut replies);
        assert!(replies.is_empty());
        assert_eq!(transfers.in_flight(), 2);
        assert_eq!(transfers.ranks[1].outstanding, 0);
        assert_eq!(transfers.ranks[0].outstanding, 0);
        sender
            .send(reply(JobId(1), 0, 0, Err(invalid("first")), 0))
            .unwrap();
        sender.send(reply(JobId(1), 1, 1, Ok(()), 8)).unwrap();
        transfers.poll(&mut replies);
        assert_eq!(replies.len(), 2);
        let failed = replies.iter().find(|reply| reply.window == 0).unwrap();
        assert_eq!(failed.id, JobId(1));
        assert_eq!(failed.direction, Direction::Out);
        assert_eq!(failed.bytes, 100);
        assert_eq!(failed.result.as_ref().unwrap_err().to_string(), "first");
        let finished = replies.iter().find(|reply| reply.window == 1).unwrap();
        assert!(finished.result.is_ok());
        assert_eq!(finished.bytes, 15);
        assert!(finished.elapsed >= Duration::ZERO);
        assert_eq!(transfers.in_flight(), 0);
        sender.send(reply(JobId(42), 0, 0, Ok(()), 1)).unwrap();
        transfers.poll(&mut replies);
        assert_eq!(replies.len(), 2);
    }

    #[test]
    fn submit_refuses_stopped_threads_full_queues_and_duplicates() {
        let (mut transfers, sender) = detached(1);
        let error = transfers.submit(job(Direction::Out)).unwrap_err();
        assert!(error.to_string().contains("stopped"), "{error}");
        assert_eq!(transfers.in_flight(), 0);
        transfers.pending.insert(
            (JobId(9), 2),
            Pending {
                direction: Direction::In,
                remaining: 1,
                bytes: 0,
                error: None,
                started: Instant::now(),
            },
        );
        let error = transfers.submit(job(Direction::Out)).unwrap_err();
        assert!(error.to_string().contains("already in flight"), "{error}");
        // With every reply sender gone, close reports the outstanding job
        // instead of waiting for CLOSE_TIMEOUT.
        drop(sender);
        let error = transfers.close().unwrap_err();
        assert!(
            error.to_string().contains("1 transfer jobs outstanding"),
            "{error}"
        );
        assert_eq!(transfers.in_flight(), 1);
    }

    /// A rank whose send fails after the pre-check counts as a failed rank
    /// reply: the ranks that accepted the job run it and the merged reply
    /// carries the refusal.
    #[test]
    fn submit_keeps_accepted_ranks_when_a_later_send_fails() {
        let (reply_sender, receiver) = mpsc::channel();
        let geometry = geometry(64, &[("k", 8192)], 4096);
        let layout = Layout::new(&[geometry.clone(), geometry.clone(), geometry]).unwrap();
        // Threads parked on a channel stand in for running workers.
        let mut parks = Vec::new();
        let mut spawn = || {
            let (park, parked) = mpsc::channel::<()>();
            parks.push(park);
            std::thread::spawn(move || {
                let _ = parked.recv();
            })
        };
        let (accepting, accepted) = mpsc::sync_channel(QUEUE_DEPTH);
        let (full, _full_receiver) = mpsc::sync_channel(1);
        full.try_send(job(Direction::In)).unwrap();
        let (disconnected, gone) = mpsc::sync_channel(1);
        drop(gone);
        let mut rank = |sender| RankThread {
            sender: Some(sender),
            thread: Some(spawn()),
            outstanding: 0,
        };
        let mut transfers = Transfers {
            layout,
            ranks: vec![rank(accepting), rank(full), rank(disconnected)],
            receiver,
            pending: HashMap::new(),
        };
        transfers.submit(job(Direction::Out)).unwrap();
        assert_eq!(transfers.in_flight(), 1);
        assert_eq!(accepted.try_recv().unwrap(), job(Direction::Out));
        assert_eq!(transfers.ranks[0].outstanding, 1);
        assert_eq!(transfers.ranks[1].outstanding, 0);
        let pending = &transfers.pending[&(JobId(9), 2)];
        assert_eq!(pending.remaining, 1);
        assert!(
            pending
                .error
                .as_ref()
                .is_some_and(|error| error.to_string().contains("rank 1")
                    && error.to_string().contains("queue is full")),
            "{:?}",
            pending.error
        );
        let mut replies = Vec::new();
        transfers.poll(&mut replies);
        assert!(replies.is_empty());
        reply_sender
            .send(RankReply {
                id: JobId(9),
                window: 2,
                rank: 0,
                result: Ok(()),
                bytes: 40,
            })
            .unwrap();
        transfers.poll(&mut replies);
        assert_eq!(replies.len(), 1);
        let reply = &replies[0];
        assert_eq!((reply.id, reply.window), (JobId(9), 2));
        assert_eq!(reply.bytes, 40);
        let error = reply.result.as_ref().unwrap_err().to_string();
        assert!(
            error.contains("rank 1") && error.contains("queue is full"),
            "{error}"
        );
        assert_eq!(transfers.in_flight(), 0);
        assert_eq!(transfers.ranks[0].outstanding, 0);
        // Every rank refusing leaves nothing in flight and reports the refusal.
        transfers.ranks[0].sender = Some(disconnected_sender());
        transfers.ranks[1].sender = Some(disconnected_sender());
        transfers.ranks[2].sender = Some(disconnected_sender());
        let error = transfers.submit(job(Direction::Out)).unwrap_err();
        assert!(error.to_string().contains("thread has stopped"), "{error}");
        assert_eq!(transfers.in_flight(), 0);
        drop(parks);
        drop(transfers);
    }

    fn disconnected_sender() -> SyncSender<Job> {
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        sender
    }

    #[cfg(feature = "runtime-package")]
    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn transfers_round_trip_pages_and_a_row_through_a_loaded_package() {
        use crate::{
            cuda::Stream,
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
        let rank = &geometry[0];
        let (context, device) = (manifest.context, manifest.device);
        let layout = Layout::new(&geometry).unwrap();
        assert!(rank.page_capacity >= 8 && rank.row_capacity >= 4);
        let capacity = rank.page_capacity as u32;
        let sources = [1, 7, capacity / 2, capacity - 1];
        let targets = [2, 9, capacity / 2 + 1, capacity - 2];
        let (source_slot, target_slot) = (1u32, 2u32);

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
            for id in targets {
                write_unit(
                    geometry.unit_address(u64::from(id)),
                    geometry.unit_bytes as usize,
                    0xDD,
                );
            }
        }
        write_unit(
            rank.row.unit_address(u64::from(source_slot)),
            rank.row.unit_bytes as usize,
            seed(row_plane, source_slot),
        );
        write_unit(
            rank.row.unit_address(u64::from(target_slot)),
            rank.row.unit_bytes as usize,
            0xDD,
        );

        let pages = sources.len() as u64;
        let bytes = (pages * layout.page_record_bytes + layout.row_record_bytes) as usize;
        let mut pool = SlabPool::new(
            layout.page_record_bytes,
            pages as usize,
            layout.row_record_bytes,
            1,
            false,
        )
        .unwrap();
        pool.register(&crate::offload::registrar::CudaRegistrar, context)
            .unwrap();
        let page_slabs: Vec<usize> = (0..pages)
            .map(|_| {
                let slab = pool.allocate_page().unwrap();
                pool.page_address(slab)
            })
            .collect();
        let row = pool.allocate_row().unwrap();
        let row_slab = pool.row_address(row);
        let host = page_slabs[0];

        let mut transfers =
            Transfers::start(&[(context, device)], geometry.clone(), &pool).unwrap();
        let wait = |transfers: &mut Transfers| -> Reply {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut replies = Vec::new();
            while replies.is_empty() {
                assert!(Instant::now() < deadline, "no transfer reply within 60 s");
                transfers.poll(&mut replies);
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(replies.len(), 1);
            replies.pop().unwrap()
        };
        transfers
            .submit(Job {
                id: JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: sources
                    .iter()
                    .zip(&page_slabs)
                    .map(|(id, slab)| (PageId(*id), *slab))
                    .collect(),
                row: Some((StateSlotId(source_slot), row_slab)),
            })
            .unwrap();
        assert_eq!(transfers.in_flight(), 1);
        let reply = wait(&mut transfers);
        assert!(reply.result.is_ok(), "{:?}", reply.result);
        assert_eq!(reply.id, JobId(1));
        assert_eq!(reply.direction, Direction::Out);
        assert_eq!(reply.bytes, pages * rank.page_bytes() + rank.row.unit_bytes);
        assert_eq!(transfers.in_flight(), 0);

        // SAFETY: No copy is in flight and the mapping stays valid.
        let image = unsafe { std::slice::from_raw_parts(host as *const u8, bytes) };
        let mut expected = vec![0u8; bytes];
        for (index, id) in sources.iter().enumerate() {
            let record = page_slabs[index] - host;
            for (plane, geometry) in rank.planes.iter().enumerate() {
                let start = record + layout.ranks[0].plane_offsets[plane] as usize;
                fill(
                    &mut expected[start..start + geometry.unit_bytes as usize],
                    seed(plane, *id),
                );
            }
        }
        let row_start = row_slab - host + layout.ranks[0].row_offset as usize;
        fill(
            &mut expected[row_start..row_start + rank.row.unit_bytes as usize],
            seed(row_plane, source_slot),
        );
        assert!(
            image == &expected[..],
            "host records differ from the device units"
        );

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

        transfers
            .submit(Job {
                id: JobId(2),
                window: 0,
                direction: Direction::In,
                pages: targets
                    .iter()
                    .zip(&page_slabs)
                    .map(|(id, slab)| (PageId(*id), *slab))
                    .collect(),
                row: Some((StateSlotId(target_slot), row_slab)),
            })
            .unwrap();
        let reply = wait(&mut transfers);
        assert!(reply.result.is_ok(), "{:?}", reply.result);
        assert_eq!(reply.direction, Direction::In);
        assert_eq!(reply.bytes, pages * rank.page_bytes() + rank.row.unit_bytes);

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
                    actual == wanted,
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
            actual == wanted,
            "row {target_slot} differs from row {source_slot}"
        );

        for (id, slot) in [(capacity, 0), (0, rank.row_capacity as u32)] {
            transfers
                .submit(Job {
                    id: JobId(3 + u64::from(slot != 0)),
                    window: 0,
                    direction: Direction::Out,
                    pages: vec![(PageId(id), page_slabs[0])],
                    row: Some((StateSlotId(slot), row_slab)),
                })
                .unwrap();
            let reply = wait(&mut transfers);
            assert!(reply.result.is_err());
            assert_eq!(reply.bytes, 0);
        }
        assert_eq!(transfers.in_flight(), 0);
        transfers.close().unwrap();
        assert!(transfers.submit(job(Direction::Out)).is_err());
        drop(transfers);
        drop(pool);
        drop(staging);
        drop(stream);
        drop(model);
    }

    /// Two contexts on the first two devices, each with `k` and `v` page planes
    /// and a row plane; a split pool must cost each device only its part and
    /// carry each rank's units in that rank's part.
    #[cfg(feature = "runtime-package")]
    #[test]
    #[ignore = "requires two CUDA devices"]
    fn split_pools_cost_each_device_its_part_and_round_trip_units() {
        use flare_selene::{Plane, slab::SlabPool};
        use crate::runtime_package::driver::Driver as Loader;
        const PAGES: u64 = 80;
        const ROWS: u64 = 4;
        const ROW_BYTES: u64 = 10_000;
        const POOL_BYTES: u64 = 8 << 30;
        let loader = Loader::load().unwrap();
        let mut contexts = Vec::new();
        let mut geometry = Vec::new();
        let mut allocations = Vec::new();
        // SAFETY: Driver signatures match cuda.h and every output pointer is valid.
        unsafe {
            loader.check("cuInit", (loader.init)(0)).unwrap();
            for ordinal in 0..2 {
                let mut device = 0;
                loader
                    .check("cuDeviceGet", (loader.device)(&mut device, ordinal))
                    .unwrap();
                let mut context = std::ptr::null_mut();
                loader
                    .check(
                        "cuCtxCreate_v2",
                        (loader.context_create)(&mut context, 0, device),
                    )
                    .unwrap();
                let mut allocate = |bytes: u64| {
                    let mut address = 0;
                    loader
                        .check(
                            "cuMemAlloc_v2",
                            (loader.allocate)(&mut address, bytes as usize),
                        )
                        .unwrap();
                    allocations.push((context, address));
                    address
                };
                let planes = [("k", 8192u64), ("v", 4196)]
                    .into_iter()
                    .map(|(name, unit_bytes)| Plane {
                        name: name.into(),
                        address: allocate(unit_bytes * PAGES),
                        unit_bytes,
                    })
                    .collect();
                geometry.push(CacheGeometry {
                    snapshot_pages: Vec::new(),
                    page_tokens: 64,
                    page_capacity: PAGES,
                    row_capacity: ROWS,
                    row: Plane {
                        name: "state".into(),
                        address: allocate(ROW_BYTES * ROWS),
                        unit_bytes: ROW_BYTES,
                    },
                    planes,
                });
                contexts.push((context as usize, device));
            }
        }
        let free = || -> Vec<i64> {
            contexts
                .iter()
                .map(|&(context, _)| {
                    let (mut free, mut total) = (0, 0);
                    // SAFETY: The context outlives the call and both outputs are valid.
                    unsafe {
                        loader
                            .check("cuCtxSetCurrent", (loader.context_set)(context as _))
                            .unwrap();
                        loader
                            .check(
                                "cuMemGetInfo_v2",
                                (loader.memory_info)(&mut free, &mut total),
                            )
                            .unwrap();
                    }
                    free as i64
                })
                .collect()
        };
        let layout = Layout::new(&geometry).unwrap();
        let before = free();
        let mut registered = SlabPool::sized(&layout, POOL_BYTES).unwrap();
        registered
            .register(&crate::offload::registrar::CudaRegistrar, contexts[0].0)
            .unwrap();
        let with_registered = free();
        drop(registered);
        let between = free();
        let pool = SlabPool::sized_by_rank(
            &layout,
            POOL_BYTES,
            &contexts,
            &crate::offload::registrar::CudaRegistrar,
        )
        .unwrap();
        let with_split = free();
        let mib = |bytes: i64| bytes as f64 / (1 << 20) as f64;
        for rank in 0..2 {
            let registered = before[rank] - with_registered[rank];
            let split = between[rank] - with_split[rank];
            eprintln!(
                "rank {rank}: registering {:.1} GiB costs {:.1} MiB, its split part {:.1} MiB",
                POOL_BYTES as f64 / (1u64 << 30) as f64,
                mib(registered),
                mib(split)
            );
            assert!(registered > 0, "registration cost rank {rank} nothing");
            assert!(
                split * 4 <= registered * 3,
                "rank {rank}'s split part costs {split} bytes against {registered} registered"
            );
        }
        assert!(pool.registered());
        assert_eq!(pool.page_bytes() as u64, layout.page_record_bytes);
        let parts = pool.rank_parts().unwrap();
        assert_eq!(parts.len(), 2);

        fn fill(bytes: &mut [u8], seed: u64) {
            for (index, byte) in bytes.iter_mut().enumerate() {
                let index = index as u64;
                *byte = (index.wrapping_mul(29) ^ seed.wrapping_mul(7907) ^ (index >> 7)) as u8;
            }
        }
        let seed = |rank: usize, plane: usize, unit: u32| {
            (rank as u64 + 1) * 1_000_000_007 + (plane as u64 + 1) * 1_000_003 + u64::from(unit)
        };
        let write = |context: usize, address: u64, bytes: &[u8]| unsafe {
            loader
                .check("cuCtxSetCurrent", (loader.context_set)(context as _))
                .unwrap();
            loader
                .check(
                    "cuMemcpyHtoD_v2",
                    (loader.upload)(address, bytes.as_ptr().cast(), bytes.len()),
                )
                .unwrap();
        };
        let read = |context: usize, address: u64, length: usize| -> Vec<u8> {
            let mut bytes = vec![0u8; length];
            unsafe {
                loader
                    .check("cuCtxSetCurrent", (loader.context_set)(context as _))
                    .unwrap();
                loader
                    .check(
                        "cuMemcpyDtoH_v2",
                        (loader.download)(bytes.as_mut_ptr().cast(), address, length),
                    )
                    .unwrap();
            }
            bytes
        };
        let sources: Vec<u32> = (1..=WINDOW_PAGES as u32 / 2).collect();
        let targets: Vec<u32> = sources
            .iter()
            .map(|id| id + WINDOW_PAGES as u32 / 2)
            .collect();
        let row_plane = 2;
        for (rank, (rank_geometry, &(context, _))) in geometry.iter().zip(&contexts).enumerate() {
            for (plane, unit) in rank_geometry.planes.iter().enumerate() {
                for &id in &sources {
                    let mut bytes = vec![0u8; unit.unit_bytes as usize];
                    fill(&mut bytes, seed(rank, plane, id));
                    write(context, unit.unit_address(u64::from(id)), &bytes);
                }
            }
            let mut bytes = vec![0u8; ROW_BYTES as usize];
            fill(&mut bytes, seed(rank, row_plane, 1));
            write(context, rank_geometry.row.unit_address(1), &bytes);
        }
        let mut pool = pool;
        let slabs: Vec<usize> = sources
            .iter()
            .map(|_| {
                let slab = pool.allocate_page().unwrap();
                pool.page_address(slab)
            })
            .collect();
        let row = pool.allocate_row().unwrap();
        let row_slab = pool.row_address(row);
        let mut transfers = Transfers::start(&contexts, geometry.clone(), &pool).unwrap();
        let wait = |transfers: &mut Transfers| -> Reply {
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut replies = Vec::new();
            while replies.is_empty() {
                assert!(Instant::now() < deadline, "no transfer reply within 60 s");
                transfers.poll(&mut replies);
                std::thread::sleep(Duration::from_millis(1));
            }
            replies.pop().unwrap()
        };
        let unit_bytes: u64 = geometry
            .iter()
            .map(|rank| sources.len() as u64 * rank.page_bytes() + ROW_BYTES)
            .sum();
        transfers
            .submit(Job {
                id: JobId(1),
                window: 0,
                direction: Direction::Out,
                pages: sources
                    .iter()
                    .zip(&slabs)
                    .map(|(id, slab)| (PageId(*id), *slab))
                    .collect(),
                row: Some((StateSlotId(1), row_slab)),
            })
            .unwrap();
        let reply = wait(&mut transfers);
        assert!(reply.result.is_ok(), "{:?}", reply.result);
        assert_eq!(reply.bytes, unit_bytes);

        // Each rank's units must sit in its own part, which the host can read.
        for (rank, part) in parts.iter().enumerate() {
            let offsets = &layout.ranks[rank].plane_offsets;
            for (index, &id) in sources.iter().enumerate() {
                let record = part.page(slabs[index]).unwrap().unwrap() as usize;
                for (plane, unit) in geometry[rank].planes.iter().enumerate() {
                    let mut wanted = vec![0u8; unit.unit_bytes as usize];
                    fill(&mut wanted, seed(rank, plane, id));
                    // SAFETY: The part is mapped for the host and no copy is in flight.
                    let actual = unsafe {
                        std::slice::from_raw_parts(
                            (record + offsets[plane] as usize) as *const u8,
                            unit.unit_bytes as usize,
                        )
                    };
                    assert!(
                        actual == &wanted[..],
                        "rank {rank} plane {plane} page {id} differs in its host part"
                    );
                }
            }
            let row = part.row(row_slab).unwrap().unwrap() as usize;
            let mut wanted = vec![0u8; ROW_BYTES as usize];
            fill(&mut wanted, seed(rank, row_plane, 1));
            // SAFETY: As above.
            let actual =
                unsafe { std::slice::from_raw_parts(row as *const u8, ROW_BYTES as usize) };
            assert!(
                actual == &wanted[..],
                "rank {rank} row differs in its host part"
            );
        }

        transfers
            .submit(Job {
                id: JobId(2),
                window: 0,
                direction: Direction::In,
                pages: targets
                    .iter()
                    .zip(&slabs)
                    .map(|(id, slab)| (PageId(*id), *slab))
                    .collect(),
                row: Some((StateSlotId(2), row_slab)),
            })
            .unwrap();
        let reply = wait(&mut transfers);
        assert!(reply.result.is_ok(), "{:?}", reply.result);
        assert_eq!(reply.bytes, unit_bytes);
        for (rank, (rank_geometry, &(context, _))) in geometry.iter().zip(&contexts).enumerate() {
            for (plane, unit) in rank_geometry.planes.iter().enumerate() {
                for (source, target) in sources.iter().zip(&targets) {
                    let mut wanted = vec![0u8; unit.unit_bytes as usize];
                    fill(&mut wanted, seed(rank, plane, *source));
                    let actual = read(
                        context,
                        unit.unit_address(u64::from(*target)),
                        unit.unit_bytes as usize,
                    );
                    assert!(
                        actual == wanted,
                        "rank {rank} plane {plane} page {target} differs from page {source}"
                    );
                }
            }
            let mut wanted = vec![0u8; ROW_BYTES as usize];
            fill(&mut wanted, seed(rank, row_plane, 1));
            let actual = read(
                context,
                rank_geometry.row.unit_address(2),
                ROW_BYTES as usize,
            );
            assert!(actual == wanted, "rank {rank} row 2 differs from row 1");
        }
        transfers.close().unwrap();
        drop(transfers);
        drop(pool);
        let released = free();
        for rank in 0..2 {
            assert!(
                released[rank] + (4 << 20) >= between[rank],
                "rank {rank} kept {} MiB after the split pool was dropped",
                mib(between[rank] - released[rank])
            );
        }
        // SAFETY: Every allocation belongs to its context and nothing uses it.
        unsafe {
            for (context, address) in allocations {
                loader
                    .check("cuCtxSetCurrent", (loader.context_set)(context))
                    .unwrap();
                loader
                    .check("cuMemFree_v2", (loader.free)(address))
                    .unwrap();
            }
            for (context, _) in contexts {
                loader
                    .check("cuCtxDestroy_v2", (loader.context_destroy)(context as _))
                    .unwrap();
            }
        }
    }
}

pub(crate) struct Copiers;
impl flare_selene::offload::CopierFactory for Copiers {
    fn mapped(&self, contexts: &[(usize, i32)], geometry: Vec<flare_selene::CacheGeometry>, slabs: &flare_selene::slab::SlabPool) -> flare_selene::Result<flare_selene::SharedCopier> {
        Ok(std::sync::Arc::new(std::sync::Mutex::new(Transfers::start(contexts, geometry, slabs)?)))
    }
    fn unmapped(&self, contexts: &[(usize, i32)], geometry: Vec<flare_selene::CacheGeometry>) -> flare_selene::Result<Box<dyn flare_selene::PageCopier>> {
        Ok(Box::new(Transfers::start_unmapped(contexts, geometry)?))
    }
}
