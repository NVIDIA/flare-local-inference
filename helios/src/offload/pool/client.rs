//! The `selene-io` thread: one current-thread tokio runtime, one HTTP/2
//! connection over the Unix socket, and the staged CPU/CUDA transfer pipeline.
//!
//! The thread never calls CUDA directly. `Core::pump` hands jobs to the
//! existing [`Transfers`] queues, and replies return through `poll` on the
//! same thread. No blocking call or sleep runs on the runtime.

use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime},
};

use tokio::sync::mpsc;
use tonic::{
    Request,
    codegen::tokio_stream::wrappers::ReceiverStream,
    metadata::MetadataValue,
    transport::{Channel, Endpoint},
};

use super::super::{
    CacheGeometry, HostHit, InboundTicket, JobId, Layout, PublishedCheckpoint, StoreEvent,
    StoreStats, Tier,
    index::Index,
    record::{self, HEADER_BYTES, Header},
    slab::{PageSlab, RowSlab, SlabPool, WINDOW_PAGES},
    transfer::{Direction, Job, Reply, Transfers},
};
use crate::{Result, cache::PrefixKey};

use super::{
    Command, Completion, Incoming, Shared,
    mirror::{Entry, PoolRecord, Retired, Update, decode_key, gauge},
    proto,
    window::{Grant, SharedRegion},
};

/// Control messages are limited to 16 MiB by the protocol; keep a margin for
/// framing so a full record header plus checksum segment is always accepted.
const MAX_CONTROL: usize = 16 << 20;
/// Metadata fragmentation bound. The protocol allows larger fragments, but a
/// 4 MiB fragment keeps every control message well below the 16 MiB limit.
const MAX_METADATA: usize = 4 << 20;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const BACKOFF_START: Duration = Duration::from_millis(100);
const BACKOFF_MAX: Duration = Duration::from_secs(5);
const RESUBSCRIBE_DELAY: Duration = Duration::from_millis(100);

#[cfg(test)]
mod test_support {
    use super::{Job, Reply};
    use crate::Result;

    pub(in crate::offload::pool) trait TransferBackend: Send {
        fn submit(&mut self, job: Job) -> Result<()>;
        fn poll(&mut self, replies: &mut Vec<Reply>);
        fn close(&mut self) -> Result<()>;
    }
}

#[cfg(test)]
pub(super) use test_support::TransferBackend;

/// Production uses the shared transfer pipeline directly; unit tests replace
/// it with a deterministic fake.
pub(super) enum TransferHandle {
    Real(Arc<Mutex<Transfers>>),
    #[cfg(test)]
    Fake(Box<dyn TransferBackend + Send>),
}

impl TransferHandle {
    fn submit(&mut self, job: Job) -> Result<()> {
        match self {
            Self::Real(transfers) => transfers.lock().unwrap().submit(job),
            #[cfg(test)]
            Self::Fake(fake) => fake.submit(job),
        }
    }

    fn poll(&mut self, replies: &mut Vec<Reply>) {
        match self {
            Self::Real(transfers) => transfers.lock().unwrap().poll(replies),
            #[cfg(test)]
            Self::Fake(fake) => fake.poll(replies),
        }
    }

    fn close(&mut self) -> Result<()> {
        match self {
            Self::Real(transfers) => transfers.lock().unwrap().close(),
            #[cfg(test)]
            Self::Fake(fake) => fake.close(),
        }
    }
}

/// A live manager session: one Hello generation, both mappings, and the
/// negotiated limits.
pub(super) struct Session {
    pub generation: u64,
    pub arena: SharedRegion,
    pub staging: SharedRegion,
    pub max_window_bytes: u64,
    pub max_open_puts: usize,
    pub max_open_gets: usize,
    pub window_credits: usize,
    pub channel: Channel,
}

pub(super) enum Event {
    Incoming(Incoming),
    Stream {
        id: u64,
        event: StreamEvent<StreamReply>,
    },
    Resubscribe {
        generation: u64,
        epoch: u64,
    },
}

pub(super) enum StreamEvent<T> {
    Reply(T),
    Error(tonic::Status),
    End,
}

impl<T> StreamEvent<T> {
    fn map<U>(self, map: impl FnOnce(T) -> U) -> StreamEvent<U> {
        match self {
            Self::Reply(reply) => StreamEvent::Reply(map(reply)),
            Self::Error(status) => StreamEvent::Error(status),
            Self::End => StreamEvent::End,
        }
    }
}

pub(super) enum StreamReply {
    Put(proto::PutReply),
    Get(proto::GetReply),
}

struct QueuedPut {
    generation: u64,
    checkpoint: PublishedCheckpoint,
}

enum Operation {
    Put(PutState),
    Get(GetState),
}

enum Finish {
    Stored(Result<u64>),
    Deferred,
    Fetched(Result<u64>),
}

struct Open {
    generation: u64,
    operation: Operation,
    finish: Option<Finish>,
    budget: Option<u64>,
    wake: u64,
    started: Instant,
    progress: Instant,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl Open {
    fn abort(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        match &mut self.operation {
            Operation::Put(put) => {
                put.tx.take();
            }
            Operation::Get(get) => {
                get.tx.take();
            }
        }
    }

    fn is_put(&self) -> bool {
        matches!(self.operation, Operation::Put(_))
    }

    fn is_get(&self) -> bool {
        matches!(self.operation, Operation::Get(_))
    }
}

struct PutState {
    checkpoint: PublishedCheckpoint,
    header: Option<Header>,
    /// Absolute page index of the selected parent inside the full chain.
    first: usize,
    transfer: Option<u64>,
    grants: VecDeque<Grant>,
    grant_index: u32,
    next_page: usize,
    filled_pages: usize,
    row_filled: bool,
    hashes: Vec<u64>,
    committed: bool,
    tx: Option<mpsc::Sender<proto::PutMessage>>,
}

struct GetRecord {
    metadata: Vec<u8>,
    header: Option<Header>,
}

struct GetState {
    hit: HostHit,
    ticket: InboundTicket,
    parents: Vec<Option<PrefixKey>>,
    /// Full tail page count of each chain record.
    tails: Vec<usize>,
    /// Chain-absolute first page of each chain record.
    origins: Vec<usize>,
    /// Destination page base for each chain record.
    bases: Vec<usize>,
    /// Chain-absolute page cursor in the `WindowGrant.page_first`
    /// convention; record zero starts after its skipped pages.
    cursors: Vec<usize>,
    remote: bool,
    records: Vec<GetRecord>,
    record_bytes: Vec<u64>,
    transfer: Option<u64>,
    grants: VecDeque<Grant>,
    grant_index: u32,
    pages_submitted: usize,
    row_submitted: bool,
    tx: Option<mpsc::Sender<proto::GetMessage>>,
}

#[derive(Clone)]
enum WindowKind {
    /// D2H landed in staging; copy into the manager's window and fill it.
    PutFill(Grant),
    /// Staging was filled from a Get window and submitted for H2D.
    GetCopy,
}

struct Window {
    op: u64,
    direction: Direction,
    pages: Vec<PageSlab>,
    row: Option<RowSlab>,
    kind: WindowKind,
}

pub(super) struct ModelConfig {
    pub namespace: [u8; 32],
    pub identity: [u8; 32],
    pub fingerprint: [u8; 16],
    pub geometry: Vec<CacheGeometry>,
    pub layout: Layout,
    pub descriptor: proto::Descriptor,
}

pub(super) struct Core {
    path: PathBuf,
    replica: String,
    namespace: [u8; 32],
    identity: [u8; 32],
    fingerprint: [u8; 16],
    geometry: Vec<CacheGeometry>,
    layout: Layout,
    descriptor: proto::Descriptor,
    pool: std::mem::ManuallyDrop<SlabPool>,
    transfers: TransferHandle,
    completions: std::sync::mpsc::Sender<Result<Completion>>,
    shared: Arc<Shared>,
    events: mpsc::Sender<Event>,
    generation: u64,
    session: bool,
    channel: Option<Channel>,
    arena: Option<SharedRegion>,
    staging: Option<SharedRegion>,
    max_window_bytes: u64,
    max_open_puts: usize,
    max_open_gets: usize,
    window_credits: usize,
    connecting: bool,
    pending_resubscribe: bool,
    next_attempt: Instant,
    next_subscribe: Instant,
    backoff: Duration,
    subscription_epoch: Arc<AtomicU64>,
    subscription: Option<tokio::task::JoinHandle<()>>,
    ops: BTreeMap<u64, Open>,
    queued: BTreeMap<u64, QueuedPut>,
    windows: BTreeMap<JobId, Window>,
    next_job: u64,
    fatal: Option<crate::Error>,
    failed: bool,
    retain: bool,
    cancelled: bool,
    #[cfg(test)]
    paused: Vec<Job>,
}

impl Core {
    pub(super) fn new(
        path: PathBuf,
        model: ModelConfig,
        pool: SlabPool,
        transfers: TransferHandle,
        completions: std::sync::mpsc::Sender<Result<Completion>>,
        shared: Arc<Shared>,
        events: mpsc::Sender<Event>,
    ) -> Self {
        let ModelConfig {
            namespace,
            identity,
            fingerprint,
            geometry,
            layout,
            descriptor,
        } = model;
        Self {
            path,
            replica: format!("flare-{}", std::process::id()),
            namespace,
            identity,
            fingerprint,
            geometry,
            layout,
            descriptor,
            pool: std::mem::ManuallyDrop::new(pool),
            transfers,
            completions,
            shared,
            events,
            generation: 0,
            session: false,
            channel: None,
            arena: None,
            staging: None,
            max_window_bytes: 0,
            max_open_puts: 1,
            max_open_gets: 1,
            window_credits: 1,
            connecting: false,
            pending_resubscribe: false,
            next_attempt: Instant::now(),
            next_subscribe: Instant::now(),
            backoff: BACKOFF_START,
            subscription_epoch: Arc::new(AtomicU64::new(0)),
            subscription: None,
            ops: BTreeMap::new(),
            queued: BTreeMap::new(),
            windows: BTreeMap::new(),
            next_job: 1,
            fatal: None,
            failed: false,
            retain: false,
            cancelled: false,
            #[cfg(test)]
            paused: Vec::new(),
        }
    }

    fn submit(&mut self, job: Job) -> Result<()> {
        #[cfg(test)]
        if (job.direction == Direction::In && self.shared.test.pause_in.load(Ordering::Acquire))
            || (job.direction == Direction::Out
                && self.shared.test.pause_out.load(Ordering::Acquire))
        {
            self.paused.push(job);
            self.shared.test.paused.fetch_add(1, Ordering::AcqRel);
            return Ok(());
        }
        self.transfers.submit(job)
    }

    #[cfg(test)]
    fn abort_paused(&mut self) {
        let paused = std::mem::take(&mut self.paused);
        self.shared
            .test
            .paused
            .fetch_sub(paused.len(), Ordering::AcqRel);
        for job in paused {
            if let Some(window) = self.windows.remove(&job.id) {
                release(&mut self.pool, window);
            }
        }
    }

    #[cfg(test)]
    fn flush_paused(&mut self) {
        if self.paused.is_empty()
            || self.shared.test.pause_in.load(Ordering::Acquire)
            || self.shared.test.pause_out.load(Ordering::Acquire)
        {
            return;
        }
        let paused = std::mem::take(&mut self.paused);
        self.shared
            .test
            .paused
            .fetch_sub(paused.len(), Ordering::AcqRel);
        for job in paused {
            let id = job.id;
            if let Err(error) = self.transfers.submit(job) {
                if let Some(window) = self.windows.remove(&id) {
                    release(&mut self.pool, window);
                }
                self.fail_session(&error.to_string());
                break;
            }
        }
    }

    async fn drive(
        &mut self,
        commands: &mut mpsc::Receiver<Command>,
        events: &mut mpsc::Receiver<Event>,
    ) {
        let mut tick = tokio::time::interval(Duration::from_millis(10));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let stopping = self.shared.stop.load(Ordering::Acquire);
            if stopping {
                self.cancel_all();
            }
            if stopping && self.windows.is_empty() {
                return;
            }
            tokio::select! {
                biased;
                command = commands.recv() => {
                    if let Some(command) = command {
                        self.command(command);
                        // Consume the burst the worker queued with try_send before
                        // starting publications, so coalescing sees the whole set.
                        while let Ok(command) = commands.try_recv() {
                            self.command(command);
                        }
                    }
                }
                event = events.recv() => {
                    if let Some(event) = event {
                        self.event(event);
                    }
                }
                _ = tick.tick() => {}
            }
            self.pump();
            if (self.fatal.is_some() || self.failed) && self.windows.is_empty() {
                return;
            }
        }
    }

    fn command(&mut self, command: Command) {
        let stopping = self.shared.stop.load(Ordering::Acquire);
        match command {
            Command::Put {
                generation,
                id,
                checkpoint,
            } => {
                if stopping || !self.session || generation != self.generation {
                    self.emit_deferred(id, checkpoint.id, false);
                    return;
                }
                self.queued.insert(
                    id,
                    QueuedPut {
                        generation,
                        checkpoint,
                    },
                );
            }
            Command::Get {
                generation,
                id,
                hit,
                ticket,
                parents,
                tails,
                remote,
            } => {
                if stopping || !self.session || generation != self.generation {
                    self.emit_fetched_error(id, ticket.key);
                    return;
                }
                self.start_get(id, hit, ticket, parents, tails, remote);
            }
        }
    }

    fn event(&mut self, event: Event) {
        match event {
            Event::Incoming(incoming) => self.incoming(incoming),
            Event::Stream { id, event } => self.stream_event(id, event),
            Event::Resubscribe { generation, epoch } => {
                if generation == self.generation
                    && epoch == self.subscription_epoch.load(Ordering::Acquire)
                {
                    self.resubscribe(generation);
                }
            }
        }
    }

    fn incoming(&mut self, incoming: Incoming) {
        match incoming {
            Incoming::Connected(session) => {
                if self.shared.stop.load(Ordering::Acquire) {
                    return;
                }
                self.connecting = false;
                self.backoff = BACKOFF_START;
                self.pending_resubscribe = false;
                self.generation = session.generation;
                self.channel = Some(session.channel.clone());
                self.arena = Some(session.arena);
                self.staging = Some(session.staging);
                self.max_window_bytes = session.max_window_bytes;
                self.max_open_puts = session.max_open_puts;
                self.max_open_gets = session.max_open_gets;
                self.window_credits = session.window_credits;
                self.session = true;
                self.shared
                    .generation
                    .store(session.generation, Ordering::Release);
                self.shared.connected.store(true, Ordering::Release);
                self.subscribe(session.channel, session.generation);
            }
            Incoming::Catalog {
                generation,
                epoch,
                updates,
            } => {
                if generation != self.generation
                    || epoch != self.subscription_epoch.load(Ordering::Acquire)
                {
                    return;
                }
                for update in &updates {
                    if matches!(update, Update::Snapshot(..)) {
                        self.shared.wake.fetch_add(1, Ordering::Release);
                    }
                }
                let mut inbox = self.shared.mirror.lock().unwrap();
                for update in updates {
                    if !inbox.push(update) {
                        self.shared.resync.store(true, Ordering::Release);
                        break;
                    }
                }
            }
            Incoming::Room {
                generation,
                epoch,
                bytes,
            } => {
                if generation != self.generation
                    || epoch != self.subscription_epoch.load(Ordering::Acquire)
                {
                    return;
                }
                let _ = bytes;
                self.shared.wake.fetch_add(1, Ordering::Release);
                self.shared.room.fetch_add(1, Ordering::Release);
            }
            Incoming::Disconnected { generation, epoch } => {
                if self.session
                    && generation == self.generation
                    && epoch == self.subscription_epoch.load(Ordering::Acquire)
                {
                    self.fail_session("cache manager connection closed");
                }
            }
            Incoming::Failed(reason) => {
                self.connecting = false;
                eprintln!("prefix cache manager: {reason}");
                self.schedule_reconnect();
            }
        }
    }

    fn stream_event(&mut self, id: u64, event: StreamEvent<StreamReply>) {
        let Some(open) = self.ops.get_mut(&id) else {
            return;
        };
        if open.finish.is_some() {
            return;
        }
        if open.generation != self.generation {
            return;
        }
        open.progress = Instant::now();
        let result = match event {
            StreamEvent::Reply(StreamReply::Put(reply)) => self.put_reply(id, reply),
            StreamEvent::Reply(StreamReply::Get(reply)) => self.get_reply(id, reply),
            StreamEvent::Error(status) => {
                Err(format!("cache manager stream failed: {}", status.message()).into())
            }
            StreamEvent::End => Err("cache manager stream ended".into()),
        };
        if let Err(error) = result {
            self.fail_session(&error.to_string());
        }
    }

    fn put_reply(&mut self, id: u64, reply: proto::PutReply) -> Result<()> {
        let Some(open) = self.ops.get_mut(&id) else {
            return Ok(());
        };
        let Operation::Put(put) = &mut open.operation else {
            return Err("cache manager sent a Put reply to a Get".into());
        };
        match reply.r {
            Some(proto::put_reply::R::Accept(accept)) => {
                let first = usize::try_from(accept.first_page)
                    .map_err(|_| "cache manager accepted an overflowing first page")?;
                let parent = if accept.has_parent {
                    Some(decode_key(
                        accept
                            .parent
                            .as_ref()
                            .ok_or("cache manager accepted a missing parent")?,
                    )?)
                } else {
                    None
                };
                let page_tokens = self.layout.page_tokens;
                if first > put.checkpoint.chain_pages.len()
                    || parent.map_or(0, |key| key.consumed_tokens / page_tokens) != first as u64
                {
                    return Err("cache manager accepted an impossible record shape".into());
                }
                let tail = put.checkpoint.chain_pages.len() - first;
                let header = Header::new(
                    &self.geometry,
                    self.identity,
                    self.fingerprint,
                    put.checkpoint.key,
                    parent,
                    put.checkpoint.bucket,
                    tail as u32,
                    SystemTime::now(),
                )?;
                if accept.record_bytes != 0 && header.length_with_row() != accept.record_bytes {
                    return Err("cache manager accepted a record of the wrong length".into());
                }
                put.header = Some(header);
                put.first = first;
                put.transfer = Some(accept.transfer);
                put.hashes = vec![0; tail];
            }
            Some(proto::put_reply::R::Grant(grant)) => {
                let grant = Grant::decode(grant)?;
                if put.transfer != Some(grant.transfer) {
                    return Err("cache manager granted a foreign transfer".into());
                }
                put.grants.push_back(grant);
            }
            Some(proto::put_reply::R::Stored(stored)) => {
                open.finish = Some(Finish::Stored(Ok(stored.bytes)));
            }
            Some(proto::put_reply::R::Refused(refused)) => {
                let reason = proto::Reason::try_from(refused.reason)
                    .map_err(|_| "cache manager refused a Put with an unknown reason")?;
                let detail = detail(&refused.detail);
                open.finish = Some(match reason {
                    proto::Reason::Present => Finish::Stored(Ok(0)),
                    proto::Reason::Inflight
                    | proto::Reason::Budget
                    | proto::Reason::Quota
                    | proto::Reason::Queue => {
                        if reason == proto::Reason::Budget {
                            open.budget = Some(open.wake);
                        }
                        Finish::Deferred
                    }
                    proto::Reason::Oversize => Finish::Stored(Err(format!(
                        "checkpoint exceeds cache manager capacity{detail}"
                    )
                    .into())),
                    proto::Reason::Invalid | proto::Reason::Io => {
                        Finish::Stored(Err(format!("cache manager refused Put{detail}").into()))
                    }
                    proto::Reason::Unspecified => {
                        return Err("cache manager refused a Put without a reason".into());
                    }
                });
            }
            None => return Err("cache manager sent an empty Put reply".into()),
        }
        Ok(())
    }

    fn get_reply(&mut self, id: u64, reply: proto::GetReply) -> Result<()> {
        let Some(open) = self.ops.get_mut(&id) else {
            return Ok(());
        };
        let Operation::Get(get) = &mut open.operation else {
            return Err("cache manager sent a Get reply to a Put".into());
        };
        match reply.r {
            Some(proto::get_reply::R::Leased(leased)) => {
                if get.transfer.is_some() || leased.record_bytes.len() != get.hit.chain.nodes.len()
                {
                    return Err("cache manager leased an impossible restore".into());
                }
                get.record_bytes = leased.record_bytes;
                get.transfer = Some(leased.transfer);
            }
            Some(proto::get_reply::R::Meta(meta)) => {
                if get.transfer.is_none() {
                    return Err("cache manager sent metadata before the lease".into());
                }
                let record = usize::try_from(meta.record)
                    .map_err(|_| "cache manager restored an overflowing record index")?;
                let source = proto::Tier::try_from(meta.source)
                    .map_err(|_| "cache manager restored from an unknown tier")?;
                if source == proto::Tier::Unspecified {
                    return Err("cache manager restored from an unspecified tier".into());
                }
                let Some(entry) = get.records.get_mut(record) else {
                    return Err("cache manager restored an absent chain record".into());
                };
                if meta.metadata_offset != entry.metadata.len() as u64 {
                    return Err("cache manager metadata fragments are not contiguous".into());
                }
                entry.metadata.extend_from_slice(&meta.metadata);
                if entry.header.is_none() && entry.metadata.len() >= HEADER_BYTES {
                    let header = Header::decode(&entry.metadata)?;
                    let endpoint = record + 1 == get.hit.chain.nodes.len();
                    let stored = *get
                        .record_bytes
                        .get(record)
                        .ok_or("cache manager restored metadata without a lease record")?;
                    if header.identity != self.identity
                        || header.fingerprint != self.fingerprint
                        || !header.matches(&self.geometry)
                        || header.flags != 0
                        || header.key != get.hit.chain.nodes[record]
                        || header.parent != get.parents[record]
                        || if endpoint {
                            stored != header.length_with_row()
                        } else {
                            stored != header.row_segment && stored != header.length_with_row()
                        }
                        || header.tail_pages as usize != get.tails[record]
                    {
                        return Err("cache manager record metadata mismatch".into());
                    }
                    entry.header = Some(header);
                }
                if let Some(header) = &entry.header
                    && entry.metadata.len() as u64
                        > header.checksum_segment() + header.checksum_bytes()
                {
                    return Err("cache manager restored trailing metadata".into());
                }
            }
            Some(proto::get_reply::R::Ready(ready)) => {
                let grant = Grant::decode(ready)?;
                if get.transfer != Some(grant.transfer) {
                    return Err("cache manager granted a foreign restore".into());
                }
                get.grants.push_back(grant);
            }
            Some(proto::get_reply::R::Done(_)) => {
                if get.transfer.is_none()
                    || get.pages_submitted != get.hit.pages
                    || !get.row_submitted
                    || get
                        .cursors
                        .iter()
                        .zip(&get.origins)
                        .zip(&get.tails)
                        .any(|((cursor, origin), tail)| *cursor != origin + tail)
                    || get.records.iter().any(|record| {
                        record.header.as_ref().is_none_or(|header| {
                            record.metadata.len() as u64
                                != header.checksum_segment() + header.checksum_bytes()
                        })
                    })
                {
                    return Err("cache manager restored a short record".into());
                }
                open.finish = Some(Finish::Fetched(Ok(get.hit.bytes)));
            }
            Some(proto::get_reply::R::Refused(refused)) => {
                let detail = detail(&refused.detail);
                open.finish = Some(Finish::Fetched(Err(format!(
                    "cache manager refused restore{detail}"
                )
                .into())));
            }
            None => return Err("cache manager sent an empty Get reply".into()),
        }
        Ok(())
    }

    fn start_get(
        &mut self,
        id: u64,
        hit: HostHit,
        ticket: InboundTicket,
        parents: Vec<Option<PrefixKey>>,
        tails: Vec<usize>,
        remote: bool,
    ) {
        if self.open_ops(false) >= self.max_open_gets
            || tails.len() != hit.chain.nodes.len()
            || parents.len() != hit.chain.nodes.len()
        {
            self.emit_fetched_error(id, ticket.key);
            return;
        }
        let page_tokens = self.layout.page_tokens;
        let mut origins = Vec::with_capacity(tails.len());
        let mut cursors = Vec::with_capacity(tails.len());
        let mut bases = Vec::with_capacity(tails.len());
        let mut base = 0;
        for (index, (key, tail)) in hit.chain.nodes.iter().zip(&tails).enumerate() {
            let Ok(pages) = usize::try_from(key.consumed_tokens / page_tokens) else {
                self.emit_fetched_error(id, ticket.key);
                return;
            };
            let Some(origin) = pages.checked_sub(*tail) else {
                self.emit_fetched_error(id, ticket.key);
                return;
            };
            let skip = if index == 0 { hit.chain.skip_pages } else { 0 };
            if skip > *tail {
                self.emit_fetched_error(id, ticket.key);
                return;
            }
            origins.push(origin);
            cursors.push(origin + skip);
            bases.push(base);
            base += tail - skip;
        }
        if base != hit.pages {
            self.emit_fetched_error(id, ticket.key);
            return;
        }
        let Some(channel) = self.channel.clone() else {
            self.emit_fetched_error(id, ticket.key);
            return;
        };
        let chain = hit
            .chain
            .nodes
            .iter()
            .map(|key| proto::Key {
                consumed_tokens: key.consumed_tokens,
                digest: key.digest.to_vec(),
            })
            .collect();
        let begin = proto::GetMessage {
            m: Some(proto::get_message::M::Begin(proto::GetBegin {
                namespace_id: self.namespace.to_vec(),
                chain,
                skip_pages: hit.chain.skip_pages as u64,
                expected_pages: hit.pages as u64,
                row: true,
                destinations: Vec::new(),
                transfer: 0,
            })),
        };
        let (tx, rx) = mpsc::channel(8);
        if tx.try_send(begin).is_err() {
            self.emit_fetched_error(id, ticket.key);
            return;
        }
        let generation = self.generation;
        let events = self.events.clone();
        let task = tokio::spawn(get_call(channel, generation, rx, events, id));
        let wake = self.shared.wake.load(Ordering::Acquire);
        let now = Instant::now();
        self.ops.insert(
            id,
            Open {
                generation,
                operation: Operation::Get(GetState {
                    records: (0..hit.chain.nodes.len())
                        .map(|_| GetRecord {
                            metadata: Vec::new(),
                            header: None,
                        })
                        .collect(),
                    hit,
                    ticket,
                    parents,
                    tails,
                    origins,
                    bases,
                    cursors,
                    remote,
                    record_bytes: Vec::new(),
                    transfer: None,
                    grants: VecDeque::new(),
                    grant_index: 0,
                    pages_submitted: 0,
                    row_submitted: false,
                    tx: Some(tx),
                }),
                finish: None,
                budget: None,
                wake,
                started: now,
                progress: now,
                task: Some(task),
            },
        );
    }

    fn open_ops(&self, put: bool) -> usize {
        self.ops
            .values()
            .filter(|open| open.finish.is_none() && if put { open.is_put() } else { open.is_get() })
            .count()
    }

    fn pump(&mut self) {
        #[cfg(test)]
        self.flush_paused();
        self.drain_replies();
        if let Some(error) = self.fatal.take() {
            if !self.failed {
                let _ = self.completions.send(Err(error));
            }
            self.failed = true;
            self.retain = true;
            self.fail_session("cache manager transfer worker failed");
        }
        if !self.shared.stop.load(Ordering::Acquire) {
            self.expire();
        }
        self.settle();
        if !self.shared.stop.load(Ordering::Acquire) && !self.failed {
            self.start_queued();
            self.process_grants();
            if self.shared.resync.swap(false, Ordering::AcqRel) {
                self.resubscribe(self.generation);
            }
            if self.session && self.take_due_resubscribe(Instant::now()) {
                self.resubscribe_now(self.generation);
            }
        }
        self.maybe_connect();
        self.drain_retired();
    }

    fn drain_replies(&mut self) {
        let mut replies = Vec::new();
        self.transfers.poll(&mut replies);
        let now = Instant::now();
        for reply in replies {
            let Some(window) = self.windows.remove(&reply.id) else {
                continue;
            };
            let op = window.op;
            if let Some(open) = self.ops.get_mut(&op) {
                open.progress = now;
            }
            if let Err(error) = reply.result {
                release(&mut self.pool, window);
                self.fatal = Some(error);
                continue;
            }
            match window.kind.clone() {
                WindowKind::PutFill(grant) => {
                    if let Some(error) = self.finish_put_fill(window, grant) {
                        self.fail_session(&error.to_string());
                    }
                }
                WindowKind::GetCopy => {
                    release(&mut self.pool, window);
                }
            }
        }
    }

    fn finish_put_fill(&mut self, window: Window, grant: Grant) -> Option<crate::Error> {
        let Some(open) = self.ops.get_mut(&window.op) else {
            release(&mut self.pool, window);
            return None;
        };
        if open.finish.is_some() {
            release(&mut self.pool, window);
            return None;
        }
        let Operation::Put(put) = &mut open.operation else {
            release(&mut self.pool, window);
            return None;
        };
        let page_bytes = self.pool.page_bytes();
        let row_bytes = self.pool.row_bytes();
        let page_hashes: Vec<u64> = window
            .pages
            .iter()
            .map(|slab| record::checksum(self.pool.page(*slab)))
            .collect();
        let row_hash = window.row.map(|slab| record::checksum(self.pool.row(slab)));
        let region = if grant.arena {
            self.arena.as_ref()
        } else {
            self.staging.as_ref()
        };
        let Some(region) = region else {
            release(&mut self.pool, window);
            return Some("cache manager filled a window without mapped regions".into());
        };
        let mut result = Ok(());
        for (unit, slab) in window.pages.iter().enumerate() {
            if let Err(error) = region.write(
                grant.offset + (unit * page_bytes) as u64,
                page_bytes as u64,
                self.pool.page(*slab),
            ) {
                result = Err(error);
                break;
            }
        }
        if result.is_ok()
            && let Some(slab) = window.row
        {
            result = region.write(
                grant.offset + (window.pages.len() * page_bytes) as u64,
                row_bytes as u64,
                self.pool.row(slab),
            );
        }
        release(&mut self.pool, window);
        if let Err(error) = result {
            return Some(error);
        }
        if grant.page_count > 0 {
            let Some(base) = grant.page_first.checked_sub(put.first) else {
                return Some("cache manager filled a window before the record".into());
            };
            for (unit, hash) in page_hashes.iter().enumerate() {
                put.hashes[base + unit] = *hash;
            }
        }
        put.filled_pages += grant.page_count;
        put.row_filled |= grant.row;
        if let (Some(header), Some(hash)) = (put.header.as_mut(), row_hash) {
            header.row_checksum = hash;
        }
        let filled = proto::PutMessage {
            m: Some(proto::put_message::M::Filled(proto::WindowFilled {
                transfer: grant.transfer,
                index: grant.index,
                page_xxh3: page_hashes,
                row_xxh3: row_hash.unwrap_or(0),
                has_row: grant.row,
            })),
        };
        let sent = put
            .tx
            .as_ref()
            .is_some_and(|tx| tx.try_send(filled).is_ok());
        if !sent {
            open.finish = Some(Finish::Deferred);
            return None;
        }
        open.progress = Instant::now();
        if put.filled_pages == put.hashes.len() && put.row_filled && !put.committed {
            commit_put(open);
        }
        None
    }

    fn start_queued(&mut self) {
        if !self.session {
            return;
        }
        #[cfg(test)]
        if self.shared.test.hold_puts.load(Ordering::Acquire) {
            return;
        }
        loop {
            if self.open_ops(true) >= self.max_open_puts {
                return;
            }
            let Some(id) = self.next_startable() else {
                return;
            };
            let Some(queued) = self.queued.remove(&id) else {
                return;
            };
            self.start_put(id, queued);
        }
    }

    fn next_startable(&mut self) -> Option<u64> {
        // FIFO between unrelated chains: the oldest queued publication is
        // chosen, except that any queued ancestor of it starts first. A newer,
        // shorter, unrelated publication can therefore never overtake an older
        // one, while a retried ancestor still starts before its descendants.
        loop {
            let (&oldest, queued) = self.queued.iter().next()?;
            let stale = queued.generation != self.generation;
            let oldest_checkpoint = queued.checkpoint.id;
            if stale {
                self.queued.remove(&oldest);
                self.emit_deferred(oldest, oldest_checkpoint, false);
                continue;
            }
            let oldest_checkpoint = &self.queued[&oldest].checkpoint;
            let (candidate, checkpoint) = self
                .queued
                .iter()
                .filter(|(other, candidate)| {
                    **other == oldest || is_descendant(&candidate.checkpoint, oldest_checkpoint)
                })
                .min_by_key(|(other, candidate)| (candidate.checkpoint.chain_pages.len(), **other))
                .map(|(id, queued)| (*id, queued.checkpoint.id))?;
            let (newer, needed) = {
                let queued = &self.queued[&candidate];
                let newer = self.queued.iter().any(|(other, other_candidate)| {
                    *other != candidate
                        && is_descendant(&queued.checkpoint, &other_candidate.checkpoint)
                });
                // A queued or in-flight child still needs this record as its
                // parent.
                let needed = self
                    .queued
                    .values()
                    .map(|other| &other.checkpoint)
                    .chain(self.ops.values().filter_map(|open| match &open.operation {
                        Operation::Put(put) => Some(&put.checkpoint),
                        Operation::Get(_) => None,
                    }))
                    .any(|other| other.parent == Some(queued.checkpoint.key));
                (newer, needed)
            };
            if newer && !needed {
                self.queued.remove(&candidate);
                self.emit_deferred(candidate, checkpoint, true);
                continue;
            }
            return Some(candidate);
        }
    }

    fn start_put(&mut self, id: u64, queued: QueuedPut) {
        let Some(channel) = self.channel.clone() else {
            self.emit_deferred(id, queued.checkpoint.id, false);
            return;
        };
        let checkpoint = queued.checkpoint;
        let begin = proto::PutMessage {
            m: Some(proto::put_message::M::Begin(proto::PutBegin {
                namespace_id: self.namespace.to_vec(),
                key: Some(proto::Key {
                    consumed_tokens: checkpoint.key.consumed_tokens,
                    digest: checkpoint.key.digest.to_vec(),
                }),
                chain_pages: checkpoint.chain_pages.len() as u64,
                row: true,
                published_parent: checkpoint.parent.map(|key| proto::Key {
                    consumed_tokens: key.consumed_tokens,
                    digest: key.digest.to_vec(),
                }),
                has_published_parent: checkpoint.parent.is_some(),
                bucket: checkpoint.bucket.to_vec(),
                published_tail_pages: checkpoint.tail_pages as u64,
                descriptor: Some(self.descriptor.clone()),
            })),
        };
        let (tx, rx) = mpsc::channel(8);
        if tx.try_send(begin).is_err() {
            self.emit_deferred(id, checkpoint.id, false);
            return;
        }
        let generation = self.generation;
        let events = self.events.clone();
        let task = tokio::spawn(put_call(channel, generation, rx, events, id));
        let wake = self.shared.wake.load(Ordering::Acquire);
        let now = Instant::now();
        self.ops.insert(
            id,
            Open {
                generation,
                operation: Operation::Put(PutState {
                    checkpoint,
                    header: None,
                    first: 0,
                    transfer: None,
                    grants: VecDeque::new(),
                    grant_index: 0,
                    next_page: 0,
                    filled_pages: 0,
                    row_filled: false,
                    hashes: Vec::new(),
                    committed: false,
                    tx: Some(tx),
                }),
                finish: None,
                budget: None,
                wake,
                started: now,
                progress: now,
                task: Some(task),
            },
        );
    }

    fn take_grant(&mut self, id: u64) -> Option<Grant> {
        match &mut self.ops.get_mut(&id)?.operation {
            Operation::Put(put) => put.grants.pop_front(),
            Operation::Get(get) => get.grants.pop_front(),
        }
    }

    fn return_grant(&mut self, id: u64, grant: Grant) {
        if let Some(open) = self.ops.get_mut(&id) {
            match &mut open.operation {
                Operation::Put(put) => put.grants.push_front(grant),
                Operation::Get(get) => get.grants.push_front(grant),
            }
        }
    }

    // The loop body mutates `self.ops`, so the entry borrow is re-taken each
    // iteration rather than held by a `while let` binding.
    #[allow(clippy::while_let_loop)]
    fn process_grants(&mut self) {
        let ids: Vec<u64> = self.ops.keys().copied().collect();
        'ops: for id in ids {
            loop {
                let (is_put, has_grant, finished) = match self.ops.get(&id) {
                    Some(open) => (
                        open.is_put(),
                        match &open.operation {
                            Operation::Put(put) => !put.grants.is_empty(),
                            Operation::Get(get) => !get.grants.is_empty(),
                        },
                        open.finish.is_some(),
                    ),
                    None => break,
                };
                if finished || !has_grant {
                    break;
                }
                let full = if is_put {
                    self.direction_windows(Direction::Out) >= self.window_credits
                } else {
                    self.direction_windows(Direction::In) >= self.window_credits
                };
                if full {
                    break;
                }
                let Some(grant) = self.take_grant(id) else {
                    break;
                };
                let result = if is_put {
                    self.allocate_put_job(id, grant)
                } else {
                    self.allocate_get_job(id, grant)
                };
                match result {
                    Ok(()) => {}
                    Err(Allocate::Retry(grant)) => {
                        self.return_grant(id, grant);
                        break;
                    }
                    Err(Allocate::Failed(error)) => {
                        self.fail_session(&error.to_string());
                        break 'ops;
                    }
                }
            }
        }
    }

    fn direction_windows(&self, direction: Direction) -> usize {
        self.windows
            .values()
            .filter(|window| window.direction == direction)
            .count()
    }

    fn allocate_put_job(&mut self, op: u64, grant: Grant) -> std::result::Result<(), Allocate> {
        let (pages, slot) = {
            let Some(Open {
                operation: Operation::Put(put),
                ..
            }) = self.ops.get(&op)
            else {
                return Err(Allocate::Failed("cache manager lost a Put".into()));
            };
            validate_put_grant(
                &grant,
                put,
                self.layout.page_record_bytes,
                self.layout.row_record_bytes,
                self.max_window_bytes,
            )?;
            (put.checkpoint.chain_pages.clone(), put.checkpoint.slot)
        };
        let region = if grant.arena {
            self.arena.as_ref()
        } else {
            self.staging.as_ref()
        };
        let Some(region) = region else {
            return Err(Allocate::Failed(
                "cache manager granted an unmapped window".into(),
            ));
        };
        if let Err(error) = region.check(grant.offset, grant.length) {
            return Err(Allocate::Failed(error));
        }
        let Some((page_slabs, row_slab)) = take_slabs(&mut self.pool, grant.page_count, grant.row)
        else {
            return Err(Allocate::Retry(grant));
        };
        let job = Job {
            id: JobId(self.next_job),
            window: 0,
            direction: Direction::Out,
            pages: page_slabs
                .iter()
                .enumerate()
                .map(|(index, slab)| {
                    (
                        pages[grant.page_first + index],
                        self.pool.page_address(*slab),
                    )
                })
                .collect(),
            row: row_slab.map(|slab| (slot, self.pool.row_address(slab))),
        };
        self.next_job += 1;
        if let Err(error) = self.submit(job) {
            drop_slabs(&mut self.pool, &page_slabs, row_slab);
            return Err(Allocate::Failed(error));
        }
        let granted_pages = grant.page_count;
        let job_id = JobId(self.next_job - 1);
        self.windows.insert(
            job_id,
            Window {
                op,
                direction: Direction::Out,
                pages: page_slabs,
                row: row_slab,
                kind: WindowKind::PutFill(grant),
            },
        );
        if let Some(Open {
            operation: Operation::Put(put),
            ..
        }) = self.ops.get_mut(&op)
        {
            put.grant_index += 1;
            put.next_page += granted_pages;
        }
        if let Some(open) = self.ops.get_mut(&op) {
            open.progress = Instant::now();
        }
        Ok(())
    }

    fn allocate_get_job(&mut self, op: u64, grant: Grant) -> std::result::Result<(), Allocate> {
        let (page_slabs, row_slab) = {
            let Some(Open {
                operation: Operation::Get(get),
                ..
            }) = self.ops.get(&op)
            else {
                return Err(Allocate::Failed("cache manager lost a Get".into()));
            };
            let record = grant.record as usize;
            if record >= get.records.len() {
                return Err(Allocate::Failed(
                    "cache manager restored an absent chain record".into(),
                ));
            }
            // WindowGrant.page_first is chain-absolute.
            let origin = get.origins[record];
            let end = origin + get.tails[record];
            if (grant.page_count > 0 && grant.page_first != get.cursors[record])
                || grant.page_first.saturating_add(grant.page_count) > end
                || (grant.row && record + 1 != get.records.len())
                || (grant.page_count == 0 && get.cursors[record] != end)
            {
                return Err(Allocate::Failed(
                    "cache manager restore window is out of order".into(),
                ));
            }
            let entry = &get.records[record];
            let Some(header) = entry.header.as_ref() else {
                return Err(Allocate::Failed(
                    "cache manager restore window precedes its metadata".into(),
                ));
            };
            if entry.metadata.len() as u64 != header.checksum_segment() + header.checksum_bytes() {
                return Err(Allocate::Failed(
                    "cache manager restore metadata is incomplete".into(),
                ));
            }
            let page_bytes = self.pool.page_bytes() as u64;
            let row_bytes = self.pool.row_bytes() as u64;
            let expected = grant.page_count as u64 * page_bytes + u64::from(grant.row) * row_bytes;
            if grant.length != expected
                || grant.page_count > WINDOW_PAGES
                || grant.index != get.grant_index
                || grant.length > self.max_window_bytes.max(page_bytes.max(row_bytes))
            {
                return Err(Allocate::Failed(
                    "cache manager restore window has the wrong length".into(),
                ));
            }
            if !grant.record_offset.is_multiple_of(4096) {
                return Err(Allocate::Failed(
                    "cache manager restore window has an unaligned offset".into(),
                ));
            }
            let Some((pages, row)) = take_slabs(&mut self.pool, grant.page_count, grant.row) else {
                return Err(Allocate::Retry(grant));
            };
            let region = if grant.arena {
                self.arena.as_ref()
            } else {
                self.staging.as_ref()
            };
            let Some(region) = region else {
                drop_slabs(&mut self.pool, &pages, row);
                return Err(Allocate::Failed(
                    "cache manager restored without mapped regions".into(),
                ));
            };
            let mut result = Ok(());
            for (index, slab) in pages.iter().enumerate() {
                let destination = self.pool.page_mut(*slab);
                if let Err(error) = region.read(
                    grant.offset + index as u64 * page_bytes,
                    page_bytes,
                    destination,
                ) {
                    result = Err(error);
                    break;
                }
                let expected =
                    checksum_at(&entry.metadata, header, grant.page_first - origin + index);
                if expected != record::checksum(destination) {
                    result = Err("cache manager payload checksum mismatch".into());
                    break;
                }
            }
            if result.is_ok()
                && let Some(slab) = row
            {
                let destination = self.pool.row_mut(slab);
                if let Err(error) = region.read(
                    grant.offset + pages.len() as u64 * page_bytes,
                    row_bytes,
                    destination,
                ) {
                    result = Err(error);
                } else if header.row_checksum != record::checksum(destination) {
                    result = Err("cache manager row checksum mismatch".into());
                }
            }
            if let Err(error) = result {
                drop_slabs(&mut self.pool, &pages, row);
                return Err(Allocate::Failed(error));
            }
            (pages, row)
        };
        if let Some(Open {
            operation: Operation::Get(get),
            ..
        }) = self.ops.get_mut(&op)
        {
            // Release the manager's window before submitting the GPU copy;
            // the staging slabs stay allocated until the CUDA reply arrives.
            let released = proto::GetMessage {
                m: Some(proto::get_message::M::Released(proto::WindowReleased {
                    transfer: grant.transfer,
                    index: grant.index,
                })),
            };
            let sent = get
                .tx
                .as_ref()
                .is_some_and(|tx| tx.try_send(released).is_ok());
            if !sent {
                drop_slabs(&mut self.pool, &page_slabs, row_slab);
                return Err(Allocate::Failed(
                    "cache manager restore stream closed".into(),
                ));
            }
        }
        let (ticket_pages, ticket_slot, base, origin, skip) = match self.ops.get(&op) {
            Some(Open {
                operation: Operation::Get(get),
                ..
            }) => (
                get.ticket.pages.clone(),
                get.ticket.slot,
                get.bases[grant.record as usize],
                get.origins[grant.record as usize],
                if grant.record == 0 {
                    get.hit.chain.skip_pages
                } else {
                    0
                },
            ),
            _ => {
                return Err(Allocate::Failed("cache manager lost a Get".into()));
            }
        };
        let destination_start = if grant.page_count == 0 {
            base
        } else {
            // Map the chain-absolute page into the record's ticket range.
            base + (grant.page_first - origin - skip)
        };
        let job = Job {
            id: JobId(self.next_job),
            window: 0,
            direction: Direction::In,
            pages: page_slabs
                .iter()
                .enumerate()
                .map(|(index, slab)| {
                    (
                        ticket_pages[destination_start + index],
                        self.pool.page_address(*slab),
                    )
                })
                .collect(),
            row: row_slab.map(|slab| (ticket_slot, self.pool.row_address(slab))),
        };
        self.next_job += 1;
        if let Err(error) = self.submit(job) {
            drop_slabs(&mut self.pool, &page_slabs, row_slab);
            return Err(Allocate::Failed(error));
        }
        let job_id = JobId(self.next_job - 1);
        self.windows.insert(
            job_id,
            Window {
                op,
                direction: Direction::In,
                pages: page_slabs,
                row: row_slab,
                kind: WindowKind::GetCopy,
            },
        );
        if let Some(Open {
            operation: Operation::Get(get),
            ..
        }) = self.ops.get_mut(&op)
        {
            get.grant_index += 1;
            get.cursors[grant.record as usize] += grant.page_count;
            get.pages_submitted += grant.page_count;
            if grant.row {
                get.row_submitted = true;
            }
        }
        if let Some(open) = self.ops.get_mut(&op) {
            open.progress = Instant::now();
        }
        Ok(())
    }

    fn expire(&mut self) {
        let deadline = self.shared.deadline();
        let now = Instant::now();
        if self
            .ops
            .values()
            .any(|open| open.finish.is_none() && now.duration_since(open.progress) >= deadline)
        {
            // A timer is a session failure: rebuild the connection and let
            // every operation drain before it reports.
            self.fail_session("cache manager no-progress deadline");
        }
    }

    fn settle(&mut self) {
        let ready: Vec<u64> = self
            .ops
            .iter()
            .filter_map(|(id, open)| {
                (open.finish.is_some() && !self.windows.values().any(|window| window.op == *id))
                    .then_some(*id)
            })
            .collect();
        for id in ready {
            let Some(mut open) = self.ops.remove(&id) else {
                continue;
            };
            open.abort();
            let finish = open.finish.take().unwrap();
            let (event, tier, remote, tokens) = match open.operation {
                Operation::Put(put) => match finish {
                    Finish::Deferred => (
                        StoreEvent::Deferred {
                            checkpoint: put.checkpoint.id,
                        },
                        Tier::Memory,
                        false,
                        0,
                    ),
                    Finish::Stored(result) => (
                        StoreEvent::Stored {
                            checkpoint: put.checkpoint.id,
                            key: put.checkpoint.key,
                            result,
                        },
                        Tier::Memory,
                        false,
                        0,
                    ),
                    Finish::Fetched(_) => unreachable!(),
                },
                Operation::Get(get) => match finish {
                    Finish::Fetched(result) => (
                        StoreEvent::Fetched {
                            key: get.ticket.key,
                            result,
                        },
                        get.hit.tier,
                        get.remote,
                        get.hit.consumed_tokens - get.hit.resident_tokens,
                    ),
                    Finish::Deferred | Finish::Stored(_) => unreachable!(),
                },
            };
            let _ = self.completions.send(Ok(Completion {
                id,
                event,
                elapsed: open.started.elapsed(),
                tier,
                remote,
                tokens,
                budget: open.budget,
                superseded: false,
            }));
        }
    }

    fn emit_deferred(&mut self, id: u64, checkpoint: crate::cache::CheckpointId, superseded: bool) {
        let _ = self.completions.send(Ok(Completion {
            id,
            event: StoreEvent::Deferred { checkpoint },
            elapsed: Duration::ZERO,
            tier: Tier::Memory,
            remote: false,
            tokens: 0,
            budget: None,
            superseded,
        }));
    }

    fn emit_fetched_error(&mut self, id: u64, key: PrefixKey) {
        let _ = self.completions.send(Ok(Completion {
            id,
            event: StoreEvent::Fetched {
                key,
                result: Err("cache manager fetch unavailable".into()),
            },
            elapsed: Duration::ZERO,
            tier: Tier::Disk,
            remote: false,
            tokens: 0,
            budget: None,
            superseded: false,
        }));
    }

    fn cancel_all(&mut self) {
        if self.cancelled {
            return;
        }
        self.cancelled = true;
        #[cfg(test)]
        self.abort_paused();
        self.pending_resubscribe = false;
        self.subscription_epoch.fetch_add(1, Ordering::AcqRel);
        if let Some(task) = self.subscription.take() {
            task.abort();
        }
        self.finish_ops("cache manager client closed");
    }

    fn fail_session(&mut self, reason: &str) {
        if !self.session {
            return;
        }
        self.session = false;
        self.channel = None;
        self.arena = None;
        self.staging = None;
        self.shared.disconnect();
        self.pending_resubscribe = false;
        self.subscription_epoch.fetch_add(1, Ordering::AcqRel);
        if let Some(task) = self.subscription.take() {
            task.abort();
        }
        self.finish_ops(&format!("cache manager session failed: {reason}"));
        self.schedule_reconnect();
    }

    fn finish_ops(&mut self, get_error: &str) {
        let ids: Vec<u64> = self.ops.keys().copied().collect();
        for id in ids {
            let Some(open) = self.ops.get_mut(&id) else {
                continue;
            };
            if open.finish.is_none() {
                open.finish = Some(match &open.operation {
                    Operation::Put(_) => Finish::Deferred,
                    Operation::Get(_) => Finish::Fetched(Err(get_error.into())),
                });
            }
            open.abort();
        }
        let queued: Vec<(u64, QueuedPut)> = std::mem::take(&mut self.queued).into_iter().collect();
        for (id, queued) in queued {
            self.emit_deferred(id, queued.checkpoint.id, false);
        }
    }

    fn schedule_reconnect(&mut self) {
        self.next_attempt = Instant::now() + self.backoff;
        self.backoff = (self.backoff * 2).min(BACKOFF_MAX);
    }

    fn maybe_connect(&mut self) {
        if self.session
            || self.connecting
            || self.shared.stop.load(Ordering::Acquire)
            || Instant::now() < self.next_attempt
        {
            return;
        }
        self.connecting = true;
        let path = self.path.clone();
        let replica = self.replica.clone();
        let descriptor = self.descriptor.clone();
        let namespace = self.namespace;
        let events = self.events.clone();
        let shared = self.shared.clone();
        tokio::spawn(async move {
            match connect(path, replica, descriptor, namespace, shared).await {
                Ok(session) => {
                    let _ = events
                        .send(Event::Incoming(Incoming::Connected(session)))
                        .await;
                }
                Err(error) => {
                    let _ = events
                        .send(Event::Incoming(Incoming::Failed(error.to_string())))
                        .await;
                }
            }
        });
    }

    fn subscribe(&mut self, channel: Channel, generation: u64) {
        let epoch = self.subscription_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let namespace = self.namespace;
        let descriptor = self.descriptor.clone();
        let page_tokens = self.layout.page_tokens;
        let deadline = self.shared.deadline();
        let events = self.events.clone();
        let current = self.subscription_epoch.clone();
        if let Some(task) = self.subscription.take() {
            task.abort();
        }
        self.subscription = Some(tokio::spawn(subscription(Subscription {
            channel,
            generation,
            namespace,
            descriptor,
            page_tokens,
            epoch,
            current,
            deadline,
            events,
        })));
    }

    fn resubscribe(&mut self, generation: u64) {
        if !self.session || generation != self.generation {
            return;
        }
        if Instant::now() < self.next_subscribe {
            self.pending_resubscribe = true;
            return;
        }
        self.resubscribe_now(generation);
    }

    fn resubscribe_now(&mut self, generation: u64) {
        self.pending_resubscribe = false;
        self.next_subscribe = Instant::now() + RESUBSCRIBE_DELAY;
        self.shared.ready.store(false, Ordering::Release);
        self.shared.mirror.lock().unwrap().clear();
        let Some(channel) = self.channel.clone() else {
            self.fail_session("cache manager subscribe channel unavailable");
            return;
        };
        self.subscribe(channel, generation);
    }

    fn take_due_resubscribe(&mut self, now: Instant) -> bool {
        let due = self.pending_resubscribe && now >= self.next_subscribe;
        if due {
            self.pending_resubscribe = false;
        }
        due
    }

    fn drain_retired(&mut self) {
        let retired = self.shared.mirror.lock().unwrap().retired.take();
        if let Some(Retired { index, updates }) = retired {
            drop((index, updates));
        }
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        if !self.retain && self.windows.is_empty() {
            // SAFETY: Every CUDA reply arrived before releasing registered slabs.
            unsafe {
                std::mem::ManuallyDrop::drop(&mut self.pool);
            }
        }
    }
}

enum Allocate {
    Retry(Grant),
    Failed(crate::Error),
}

fn take_slabs(
    pool: &mut SlabPool,
    pages: usize,
    row: bool,
) -> Option<(Vec<PageSlab>, Option<RowSlab>)> {
    let pages: Vec<PageSlab> = (0..pages)
        .map(|_| pool.allocate_page())
        .collect::<Option<_>>()?;
    let row = if row {
        match pool.allocate_row() {
            Some(row) => Some(row),
            None => {
                for page in &pages {
                    pool.release_page(*page);
                }
                return None;
            }
        }
    } else {
        None
    };
    Some((pages, row))
}

fn drop_slabs(pool: &mut SlabPool, pages: &[PageSlab], row: Option<RowSlab>) {
    for page in pages {
        pool.release_page(*page);
    }
    if let Some(row) = row {
        pool.release_row(row);
    }
}

fn release(pool: &mut SlabPool, window: Window) {
    for page in window.pages {
        pool.release_page(page);
    }
    if let Some(row) = window.row {
        pool.release_row(row);
    }
}

fn commit_put(open: &mut Open) {
    let Operation::Put(put) = &mut open.operation else {
        return;
    };
    let Some(header) = put.header.as_ref() else {
        open.finish = Some(Finish::Stored(Err(
            "cache manager grant arrived before Accept".into(),
        )));
        return;
    };
    let header_bytes = match header.encode() {
        Ok(bytes) => bytes,
        Err(error) => {
            open.finish = Some(Finish::Stored(Err(error)));
            return;
        }
    };
    let total = (header.checksum_segment() + header.checksum_bytes()) as usize;
    let mut metadata = vec![0u8; total];
    metadata[..HEADER_BYTES].copy_from_slice(&header_bytes);
    for (index, hash) in put.hashes.iter().enumerate() {
        let offset = header.checksum_segment() as usize + index * size_of::<u64>();
        metadata[offset..offset + size_of::<u64>()].copy_from_slice(&hash.to_le_bytes());
    }
    let Some(transfer) = put.transfer else {
        open.finish = Some(Finish::Stored(Err(
            "cache manager accepted a record without a transfer id".into(),
        )));
        return;
    };
    let Some(tx) = &put.tx else {
        open.finish = Some(Finish::Deferred);
        return;
    };
    for (fragment, bytes) in metadata.chunks(MAX_METADATA).enumerate() {
        let message = proto::PutMessage {
            m: Some(proto::put_message::M::Commit(proto::PutCommit {
                transfer,
                metadata_offset: (fragment * MAX_METADATA) as u64,
                metadata: bytes.to_vec(),
            })),
        };
        if tx.try_send(message).is_err() {
            open.finish = Some(Finish::Deferred);
            return;
        }
    }
    put.committed = true;
    open.progress = Instant::now();
}

fn checksum_at(metadata: &[u8], header: &Header, page: usize) -> u64 {
    let offset = header.checksum_segment() as usize + page * size_of::<u64>();
    u64::from_le_bytes(
        metadata[offset..offset + size_of::<u64>()]
            .try_into()
            .unwrap(),
    )
}

fn validate_put_grant(
    grant: &Grant,
    put: &PutState,
    page_bytes: u64,
    row_bytes: u64,
    max_window: u64,
) -> std::result::Result<(), Allocate> {
    if grant.record != 0 || put.header.is_none() {
        return Err(Allocate::Failed(
            "cache manager granted an invalid Put window".into(),
        ));
    }
    let tail = put.hashes.len();
    let pages = grant.page_count > 0;
    if grant.index != put.grant_index
        || grant.page_count > WINDOW_PAGES
        || (pages
            && (grant.page_first != put.first + put.next_page
                || grant.page_first.saturating_add(grant.page_count) > put.first + tail))
        || (!pages && put.next_page != tail)
        || grant.length > max_window.max(page_bytes.max(row_bytes))
    {
        return Err(Allocate::Failed(
            "cache manager granted an invalid Put window".into(),
        ));
    }
    let expected = grant.page_count as u64 * page_bytes + u64::from(grant.row) * row_bytes;
    if grant.length != expected || !grant.length.is_multiple_of(4096) {
        return Err(Allocate::Failed(
            "cache manager granted a window of the wrong length".into(),
        ));
    }
    if (grant.page_count == 0 && !grant.row) || grant.length == 0 {
        return Err(Allocate::Failed(
            "cache manager granted an empty window".into(),
        ));
    }
    if !grant.record_offset.is_multiple_of(4096) {
        return Err(Allocate::Failed(
            "cache manager granted an unaligned record offset".into(),
        ));
    }
    Ok(())
}

fn detail(detail: &str) -> String {
    if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    }
}

fn is_descendant(older: &PublishedCheckpoint, newer: &PublishedCheckpoint) -> bool {
    newer.chain_pages.starts_with(&older.chain_pages)
        && older.key.consumed_tokens < newer.key.consumed_tokens
}

fn generation_request<T>(message: T, generation: u64) -> Request<T> {
    let mut request = Request::new(message);
    let value = MetadataValue::try_from(generation.to_string())
        .expect("a decimal generation fits a metadata value");
    request.metadata_mut().insert("x-selene-generation", value);
    request
}

fn local_client(channel: Channel) -> proto::local_cache_client::LocalCacheClient<Channel> {
    proto::local_cache_client::LocalCacheClient::new(channel)
        .max_decoding_message_size(MAX_CONTROL)
        .max_encoding_message_size(MAX_CONTROL)
}

async fn connect(
    path: PathBuf,
    replica: String,
    descriptor: proto::Descriptor,
    namespace: [u8; 32],
    shared: Arc<Shared>,
) -> Result<Session> {
    let endpoint = Endpoint::from_shared(format!("unix:{}", path.display()))
        .map_err(|error| format!("cannot parse cache manager socket: {error}"))?
        .connect_timeout(CONNECT_TIMEOUT)
        .http2_keep_alive_interval(Duration::from_secs(10))
        .keep_alive_timeout(Duration::from_secs(20));
    let channel = endpoint
        .connect()
        .await
        .map_err(|error| format!("cannot connect to cache manager: {error}"))?;
    let mut client = local_client(channel.clone());
    let reply = tokio::time::timeout(
        shared.deadline(),
        client.hello(Request::new(proto::HelloRequest {
            protocol: 1,
            replica,
            descriptor: Some(descriptor),
        })),
    )
    .await
    .map_err(|_| "cache manager Hello timed out")?
    .map_err(|status| format!("cache manager rejected Hello: {}", status.message()))?
    .into_inner();
    if reply.namespace_id.len() != 32 || reply.namespace_id.as_slice() != namespace {
        return Err("cache manager namespace id disagrees with the descriptor".into());
    }
    if reply.arena_path.is_empty() || reply.staging_path.is_empty() {
        return Err("cache manager Hello has no region paths".into());
    }
    if reply.max_window_bytes == 0
        || !(1..=4).contains(&reply.max_open_puts)
        || !(1..=4).contains(&reply.max_open_gets)
        || !(1..=2).contains(&reply.window_credits)
    {
        return Err("cache manager negotiated invalid limits".into());
    }
    let arena_path = PathBuf::from(reply.arena_path);
    let arena_bytes = reply.arena_bytes;
    let staging_path = PathBuf::from(reply.staging_path);
    let staging_bytes = reply.staging_bytes;
    let (arena, staging) =
        tokio::task::spawn_blocking(move || -> Result<(SharedRegion, SharedRegion)> {
            let arena = SharedRegion::map(&arena_path, arena_bytes)?;
            let staging = SharedRegion::map(&staging_path, staging_bytes)?;
            Ok((arena, staging))
        })
        .await
        .map_err(|error| format!("cache manager region mapping task failed: {error}"))??;
    if shared.stop.load(Ordering::Acquire) {
        return Err("cache manager client stopped".into());
    }
    Ok(Session {
        generation: reply.generation,
        arena,
        staging,
        max_window_bytes: reply.max_window_bytes,
        max_open_puts: reply.max_open_puts as usize,
        max_open_gets: reply.max_open_gets as usize,
        window_credits: reply.window_credits as usize,
        channel,
    })
}

async fn put_call(
    channel: Channel,
    generation: u64,
    rx: mpsc::Receiver<proto::PutMessage>,
    events: mpsc::Sender<Event>,
    id: u64,
) {
    let mut client = local_client(channel);
    let outbound = ReceiverStream::new(rx);
    let response = client.put(generation_request(outbound, generation)).await;
    match response {
        Ok(response) => {
            pump_replies(response.into_inner(), events, id, |id, event| {
                Event::Stream {
                    id,
                    event: event.map(StreamReply::Put),
                }
            })
            .await;
        }
        Err(status) => {
            let _ = events
                .send(Event::Stream {
                    id,
                    event: StreamEvent::Error(status),
                })
                .await;
        }
    }
}

async fn get_call(
    channel: Channel,
    generation: u64,
    rx: mpsc::Receiver<proto::GetMessage>,
    events: mpsc::Sender<Event>,
    id: u64,
) {
    let mut client = local_client(channel);
    let outbound = ReceiverStream::new(rx);
    let response = client.get(generation_request(outbound, generation)).await;
    match response {
        Ok(response) => {
            pump_replies(response.into_inner(), events, id, |id, event| {
                Event::Stream {
                    id,
                    event: event.map(StreamReply::Get),
                }
            })
            .await;
        }
        Err(status) => {
            let _ = events
                .send(Event::Stream {
                    id,
                    event: StreamEvent::Error(status),
                })
                .await;
        }
    }
}

async fn pump_replies<R>(
    mut stream: tonic::codec::Streaming<R>,
    events: mpsc::Sender<Event>,
    id: u64,
    make: fn(u64, StreamEvent<R>) -> Event,
) {
    loop {
        let event = match stream.message().await {
            Ok(Some(reply)) => StreamEvent::Reply(reply),
            Ok(None) => StreamEvent::End,
            Err(status) => StreamEvent::Error(status),
        };
        let end = matches!(event, StreamEvent::End | StreamEvent::Error(_));
        if events.send(make(id, event)).await.is_err() || end {
            return;
        }
    }
}

async fn request_resubscribe(events: &mpsc::Sender<Event>, generation: u64, epoch: u64) {
    let _ = events.send(Event::Resubscribe { generation, epoch }).await;
}

async fn signal_disconnected(
    events: &mpsc::Sender<Event>,
    generation: u64,
    epoch: u64,
    current: &AtomicU64,
) {
    if epoch == current.load(Ordering::Acquire) {
        let _ = events
            .send(Event::Incoming(Incoming::Disconnected {
                generation,
                epoch,
            }))
            .await;
    }
}

pub(super) struct Subscription {
    channel: Channel,
    generation: u64,
    namespace: [u8; 32],
    descriptor: proto::Descriptor,
    page_tokens: u64,
    epoch: u64,
    current: Arc<AtomicU64>,
    deadline: Duration,
    events: mpsc::Sender<Event>,
}

async fn subscription(s: Subscription) {
    let Subscription {
        channel,
        generation,
        namespace,
        descriptor,
        page_tokens,
        epoch,
        current,
        deadline,
        events,
    } = s;
    if epoch != current.load(Ordering::Acquire) {
        return;
    }
    let mut client = local_client(channel);
    let request = generation_request(
        proto::SubscribeRequest {
            namespace_id: namespace.to_vec(),
            generation,
            descriptor: Some(descriptor),
        },
        generation,
    );
    let mut stream = match tokio::time::timeout(deadline, client.subscribe(request)).await {
        Ok(Ok(response)) => response.into_inner(),
        _ => {
            signal_disconnected(&events, generation, epoch, &current).await;
            return;
        }
    };
    let mut expected: Option<u64> = None;
    let mut snapshot: Option<Index<PoolRecord>> = None;
    loop {
        if epoch != current.load(Ordering::Acquire) {
            return;
        }
        let event = match stream.message().await {
            Ok(Some(event)) => event,
            Ok(None) => {
                signal_disconnected(&events, generation, epoch, &current).await;
                return;
            }
            Err(_) => {
                signal_disconnected(&events, generation, epoch, &current).await;
                return;
            }
        };
        if epoch != current.load(Ordering::Acquire) {
            return;
        }
        let sequence = event.sequence;
        let Some(oneof) = event.event else {
            request_resubscribe(&events, generation, epoch).await;
            return;
        };
        let first = expected.is_none();
        let valid = match expected {
            None => matches!(oneof, proto::catalog_event::Event::SnapshotBegin(_)),
            Some(previous) => sequence == previous.wrapping_add(1),
        };
        if !valid {
            request_resubscribe(&events, generation, epoch).await;
            return;
        }
        if let proto::catalog_event::Event::SnapshotBegin(begin) = &oneof
            && first
            && sequence != begin.seq0.wrapping_add(1)
        {
            request_resubscribe(&events, generation, epoch).await;
            return;
        }
        expected = Some(sequence);
        let changed = matches!(&oneof, proto::catalog_event::Event::Changed(_));
        match oneof {
            proto::catalog_event::Event::SnapshotBegin(_) => {
                snapshot = Some(Index::new(page_tokens));
            }
            proto::catalog_event::Event::Entry(entry) => {
                let Some(snapshot) = snapshot.as_mut() else {
                    request_resubscribe(&events, generation, epoch).await;
                    return;
                };
                if Entry::decode(entry)
                    .and_then(|entry| entry.insert(snapshot))
                    .is_err()
                {
                    request_resubscribe(&events, generation, epoch).await;
                    return;
                }
            }
            proto::catalog_event::Event::SnapshotEnd(_) => {
                let Some(index) = snapshot.take() else {
                    request_resubscribe(&events, generation, epoch).await;
                    return;
                };
                let mut gauges = StoreStats::default();
                for node in index.nodes() {
                    gauge(&mut gauges, &node.location, true);
                }
                if events
                    .send(Event::Incoming(Incoming::Catalog {
                        generation,
                        epoch,
                        updates: vec![Update::Snapshot(index, gauges)],
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            proto::catalog_event::Event::Added(entry)
            | proto::catalog_event::Event::Changed(entry) => {
                if snapshot.is_some() {
                    request_resubscribe(&events, generation, epoch).await;
                    return;
                }
                let update = match Entry::decode(entry) {
                    Ok(entry) if changed => Update::Changed(entry),
                    Ok(entry) => Update::Added(entry),
                    Err(_) => {
                        request_resubscribe(&events, generation, epoch).await;
                        return;
                    }
                };
                if events
                    .send(Event::Incoming(Incoming::Catalog {
                        generation,
                        epoch,
                        updates: vec![update],
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            proto::catalog_event::Event::Removed(key) => {
                let update = match decode_key(&key) {
                    Ok(key) => Update::Removed(key),
                    Err(_) => {
                        request_resubscribe(&events, generation, epoch).await;
                        return;
                    }
                };
                if events
                    .send(Event::Incoming(Incoming::Catalog {
                        generation,
                        epoch,
                        updates: vec![update],
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            proto::catalog_event::Event::Room(room) => {
                if events
                    .send(Event::Incoming(Incoming::Room {
                        generation,
                        epoch,
                        bytes: room.bytes,
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
    }
}

/// Start the io thread's runtime, drive the client, then drain transfers off
/// the runtime.
pub(super) fn run(
    mut core: Core,
    mut commands: mpsc::Receiver<Command>,
    mut events: mpsc::Receiver<Event>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    match runtime {
        Ok(runtime) => {
            runtime.block_on(core.drive(&mut commands, &mut events));
            drop(runtime);
        }
        Err(error) => {
            let _ = core.completions.send(Err(
                format!("cannot start the selene runtime: {error}").into()
            ));
            core.retain = true;
        }
    }
    let drained = core.transfers.close();
    if let Err(error) = drained {
        // A failed drain retains the registered pool: jobs may still reference it.
        let _ = core.completions.send(Err(error));
        core.retain = true;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, VecDeque},
        fs::{self, File, OpenOptions},
        io::{Read, Seek, SeekFrom, Write},
        os::unix::net::UnixListener as StdUnixListener,
        path::{Path, PathBuf},
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        },
        thread::{self, JoinHandle},
        time::{Duration, Instant, SystemTime},
    };

    use tokio::sync::{broadcast, mpsc, oneshot};
    use tonic::codegen::tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
    use tonic::{Request, Response, Status, Streaming};

    use super::super::{PoolStore, TRACKED_AGE, TRACKED_PUBLICATIONS, Tracked, namespace_id};
    use super::*;
    use crate::{
        cache::{
            CheckpointId, Fence, PageId, PrefixCheckpointCache, PrefixHistory, Publication,
            SnapshotId, StateSnapshot,
        },
        offload::{
            CacheGeometry, Layout, PrefixCacheOffload, PrefixStore, PublishedCheckpoint,
            StoreEvent, StoreOutcome, StoreReason, Tier,
            geometry::{self, tests::geometry},
            record::{self, HEADER_BYTES, Header},
            slab::SlabPool,
        },
        scheduler::StateSlotId,
    };

    const IDENTITY: [u8; 32] = [7; 32];
    const TENANT: &str = "test";
    const PAGE_TOKENS: u64 = 64;
    const PAGE_BYTES: usize = 4096;
    const ROW_BYTES: usize = 4096;
    const ARENA_BYTES: u64 = 1 << 20;
    const STAGING_BYTES: u64 = 1 << 20;
    const GENERATION: u64 = 19;

    type KeyId = (u64, [u8; 16]);

    fn geometries() -> Vec<CacheGeometry> {
        vec![geometry(
            PAGE_TOKENS,
            &[("kv", PAGE_BYTES as u64)],
            ROW_BYTES as u64,
        )]
    }

    fn fingerprint() -> [u8; 16] {
        geometry::fingerprint(&geometries())
    }

    fn key_id(key: &PrefixKey) -> KeyId {
        (key.consumed_tokens, key.digest)
    }

    fn proto_key(key: PrefixKey) -> proto::Key {
        proto::Key {
            consumed_tokens: key.consumed_tokens,
            digest: key.digest.to_vec(),
        }
    }

    fn decode_proto_key(key: &proto::Key) -> PrefixKey {
        PrefixKey {
            consumed_tokens: key.consumed_tokens,
            digest: key.digest.as_slice().try_into().unwrap(),
        }
    }

    fn checksum_bytes(tail: usize) -> u64 {
        ((tail as u64 * 8).div_ceil(4096)) * 4096
    }

    fn page_segment(tail: usize) -> u64 {
        HEADER_BYTES as u64 + checksum_bytes(tail)
    }

    fn record_bytes(tail: usize) -> u64 {
        page_segment(tail) + tail as u64 * PAGE_BYTES as u64 + ROW_BYTES as u64
    }

    // ---- transfer backend -------------------------------------------------

    struct FakeInner {
        pending: VecDeque<Job>,
        replies: VecDeque<Reply>,
        hold: bool,
        submitted_out: u64,
        submitted_in: u64,
        completed_out: u64,
        completed_in: u64,
        h2d: HashMap<u32, Vec<u8>>,
        h2d_row: Option<Vec<u8>>,
        max_pending_out: usize,
        max_pending_in: usize,
    }

    impl FakeInner {
        fn new() -> Self {
            Self {
                pending: VecDeque::new(),
                replies: VecDeque::new(),
                hold: false,
                submitted_out: 0,
                submitted_in: 0,
                completed_out: 0,
                completed_in: 0,
                h2d: HashMap::new(),
                h2d_row: None,
                max_pending_out: 0,
                max_pending_in: 0,
            }
        }

        fn complete(&mut self, job: Job) {
            match job.direction {
                Direction::Out => {
                    for (page, address) in &job.pages {
                        // SAFETY: The fake owns this D2H job until returning its reply.
                        unsafe {
                            std::slice::from_raw_parts_mut(*address as *mut u8, PAGE_BYTES)
                                .fill(page.0 as u8);
                        }
                    }
                    if let Some((_, address)) = job.row {
                        // SAFETY: The same fake D2H ownership covers the row.
                        unsafe {
                            std::slice::from_raw_parts_mut(address as *mut u8, ROW_BYTES)
                                .fill(0xA5);
                        }
                    }
                    self.completed_out += 1;
                }
                Direction::In => {
                    for (page, address) in &job.pages {
                        // SAFETY: The fake owns this H2D job while reading the staging slab.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(*address as *const u8, PAGE_BYTES)
                        };
                        self.h2d.insert(page.0, bytes.to_vec());
                    }
                    if let Some((_, address)) = job.row {
                        // SAFETY: The same fake H2D ownership covers the row.
                        self.h2d_row = Some(unsafe {
                            std::slice::from_raw_parts(address as *const u8, ROW_BYTES).to_vec()
                        });
                    }
                    self.completed_in += 1;
                }
            }
            self.replies.push_back(Reply {
                id: job.id,
                window: job.window,
                direction: job.direction,
                result: Ok(()),
                bytes: (job.pages.len() * PAGE_BYTES + usize::from(job.row.is_some()) * ROW_BYTES)
                    as u64,
                elapsed: Duration::from_millis(1),
            });
        }

        fn complete_next(&mut self) -> bool {
            let Some(job) = self.pending.pop_front() else {
                return false;
            };
            self.complete(job);
            true
        }
    }

    struct FakeTransfers {
        inner: Arc<Mutex<FakeInner>>,
    }

    impl TransferBackend for FakeTransfers {
        fn submit(&mut self, job: Job) -> Result<()> {
            let mut inner = self.inner.lock().unwrap();
            match job.direction {
                Direction::Out => inner.submitted_out += 1,
                Direction::In => inner.submitted_in += 1,
            }
            if !inner.hold {
                inner.complete(job);
            } else {
                inner.pending.push_back(job);
                inner.max_pending_out = inner.max_pending_out.max(
                    inner
                        .pending
                        .iter()
                        .filter(|j| j.direction == Direction::Out)
                        .count(),
                );
                inner.max_pending_in = inner.max_pending_in.max(
                    inner
                        .pending
                        .iter()
                        .filter(|j| j.direction == Direction::In)
                        .count(),
                );
            }
            Ok(())
        }

        fn poll(&mut self, replies: &mut Vec<Reply>) {
            let mut inner = self.inner.lock().unwrap();
            replies.extend(inner.replies.drain(..));
        }

        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    // ---- fake manager -----------------------------------------------------

    struct FakeRecord {
        metadata: Vec<u8>,
        pages: Vec<Vec<u8>>,
        row: Vec<u8>,
    }

    impl FakeRecord {
        fn record_bytes(&self) -> u64 {
            self.metadata.len() as u64
                + self.pages.len() as u64 * PAGE_BYTES as u64
                + self.row.len() as u64
        }
    }

    struct FakeState {
        arena_path: PathBuf,
        arena_len: u64,
        staging_path: PathBuf,
        staging_len: u64,
        namespace: [u8; 32],
        generation: AtomicU64,
        entries: Mutex<Vec<proto::CatalogEntry>>,
        records: Mutex<HashMap<KeyId, FakeRecord>>,
        events: broadcast::Sender<proto::catalog_event::Event>,
        stop: broadcast::Sender<()>,
        put_refusals: Mutex<VecDeque<i32>>,
        get_refusals: Mutex<VecDeque<i32>>,
        observed_puts: Mutex<Vec<u64>>,
        observed_gets: Mutex<Vec<u64>>,
        subscribe_count: AtomicU64,
        arena_alloc: AtomicU64,
        staging_alloc: AtomicU64,
        close_after_grants: AtomicBool,
        hold_begin: AtomicBool,
        corrupt_get: AtomicBool,
        gap: AtomicBool,
        hello_count: AtomicU64,
        hold_hello: AtomicBool,
        bad_put_tail: AtomicBool,
        bad_get_row: AtomicBool,
        delay_refusal_ms: AtomicU64,
        meta_before_leased: AtomicBool,
        duplicate_leased: AtomicBool,
        bad_put_range: AtomicBool,
        grant_pages: AtomicUsize,
    }

    impl FakeState {
        fn allocate(&self, arena: bool, length: u64) -> u64 {
            let allocator = if arena {
                &self.arena_alloc
            } else {
                &self.staging_alloc
            };
            let start = allocator.fetch_add(length, Ordering::SeqCst);
            let limit = if arena {
                self.arena_len
            } else {
                self.staging_len
            };
            assert!(start + length <= limit, "fake manager region exhausted");
            start
        }

        fn write(&self, arena: bool, offset: u64, bytes: &[u8]) {
            let path = if arena {
                &self.arena_path
            } else {
                &self.staging_path
            };
            let mut file = OpenOptions::new().write(true).open(path).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(bytes).unwrap();
        }

        fn read(&self, arena: bool, offset: u64, length: usize) -> Vec<u8> {
            let path = if arena {
                &self.arena_path
            } else {
                &self.staging_path
            };
            let mut bytes = vec![0; length];
            let mut file = File::open(path).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.read_exact(&mut bytes).unwrap();
            bytes
        }

        fn publish_live(&self, event: proto::catalog_event::Event) {
            let _ = self.events.send(event);
        }
    }

    #[derive(Clone)]
    struct FakeService {
        state: Arc<FakeState>,
    }

    fn catalog_entry(
        checkpoint: &PublishedCheckpoint,
        tail: usize,
        tier: proto::Tier,
        durable: bool,
        bytes: u64,
        unreadable: bool,
    ) -> proto::CatalogEntry {
        catalog_entry_with_row(checkpoint, tail, tier, durable, bytes, unreadable, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn catalog_entry_with_row(
        checkpoint: &PublishedCheckpoint,
        tail: usize,
        tier: proto::Tier,
        durable: bool,
        bytes: u64,
        unreadable: bool,
        row: bool,
    ) -> proto::CatalogEntry {
        proto::CatalogEntry {
            key: Some(proto_key(checkpoint.key)),
            parent: checkpoint.parent.map(proto_key),
            has_parent: checkpoint.parent.is_some(),
            bucket: checkpoint.bucket.to_vec(),
            tail_pages: tail as u64,
            row,
            tier: tier as i32,
            durable,
            bytes,
            unreadable,
        }
    }

    fn live_entry(
        checkpoint: &PublishedCheckpoint,
        tail: usize,
        tier: proto::Tier,
        durable: bool,
        bytes: u64,
        unreadable: bool,
    ) -> proto::catalog_event::Event {
        proto::catalog_event::Event::Added(catalog_entry(
            checkpoint, tail, tier, durable, bytes, unreadable,
        ))
    }

    #[tonic::async_trait]
    impl proto::local_cache_server::LocalCache for FakeService {
        async fn hello(
            &self,
            request: Request<proto::HelloRequest>,
        ) -> std::result::Result<Response<proto::HelloReply>, Status> {
            self.state.hello_count.fetch_add(1, Ordering::SeqCst);
            if self.state.hold_hello.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
            let request = request.into_inner();
            if request.protocol != 1 {
                return Err(Status::failed_precondition("protocol"));
            }
            let descriptor = request
                .descriptor
                .ok_or_else(|| Status::invalid_argument("descriptor"))?;
            if descriptor.identity.len() != 32
                || descriptor.fingerprint.len() != 16
                || descriptor.page_tokens != PAGE_TOKENS
                || descriptor.page_record_bytes != PAGE_BYTES as u64
                || descriptor.row_record_bytes != ROW_BYTES as u64
            {
                return Err(Status::failed_precondition("descriptor"));
            }
            let generation = self.state.generation.fetch_add(1, Ordering::SeqCst);
            Ok(Response::new(proto::HelloReply {
                namespace_id: self.state.namespace.to_vec(),
                generation,
                arena_path: self.state.arena_path.display().to_string(),
                arena_bytes: self.state.arena_len,
                staging_path: self.state.staging_path.display().to_string(),
                staging_bytes: self.state.staging_len,
                max_open_puts: 4,
                max_open_gets: 4,
                max_window_bytes: 4 << 20,
                window_credits: 2,
            }))
        }

        type SubscribeStream = ReceiverStream<std::result::Result<proto::CatalogEvent, Status>>;

        async fn subscribe(
            &self,
            request: Request<proto::SubscribeRequest>,
        ) -> std::result::Result<Response<Self::SubscribeStream>, Status> {
            let request = request.into_inner();
            if request.namespace_id.as_slice() != self.state.namespace {
                return Err(Status::failed_precondition("namespace"));
            }
            self.state.subscribe_count.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = mpsc::channel(32);
            let state = self.state.clone();
            let entries = state.entries.lock().unwrap().clone();
            let mut live = state.events.subscribe();
            tokio::spawn(async move {
                let seq0 = 0u64;
                let mut sequence = seq0 + 1;
                if tx
                    .send(Ok(proto::CatalogEvent {
                        sequence,
                        event: Some(proto::catalog_event::Event::SnapshotBegin(
                            proto::SnapshotBegin { seq0 },
                        )),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
                for entry in entries {
                    sequence += 1;
                    if tx
                        .send(Ok(proto::CatalogEvent {
                            sequence,
                            event: Some(proto::catalog_event::Event::Entry(entry)),
                        }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                sequence += 1;
                if tx
                    .send(Ok(proto::CatalogEvent {
                        sequence,
                        event: Some(proto::catalog_event::Event::SnapshotEnd(
                            proto::SnapshotEnd {},
                        )),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }
                let mut stop = state.stop.subscribe();
                loop {
                    let event = tokio::select! {
                        biased;
                        _ = stop.recv() => return,
                        event = live.recv() => match event {
                        Ok(event) => event,
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            let _ = tx
                                .send(Err(Status::unavailable("catalog backlog overflow")))
                                .await;
                            return;
                        }
                        Err(broadcast::error::RecvError::Closed) => return,
                        },
                    };
                    if state.gap.swap(false, Ordering::SeqCst) {
                        sequence += 1;
                    }
                    sequence += 1;
                    if tx
                        .send(Ok(proto::CatalogEvent {
                            sequence,
                            event: Some(event),
                        }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            });
            Ok(Response::new(ReceiverStream::new(rx)))
        }

        type PutStream = ReceiverStream<std::result::Result<proto::PutReply, Status>>;

        async fn put(
            &self,
            request: Request<Streaming<proto::PutMessage>>,
        ) -> std::result::Result<Response<Self::PutStream>, Status> {
            let (tx, rx) = mpsc::channel(32);
            let state = self.state.clone();
            let mut input = request.into_inner();
            tokio::spawn(async move {
                handle_put(state, &mut input, tx).await;
            });
            Ok(Response::new(ReceiverStream::new(rx)))
        }

        type GetStream = ReceiverStream<std::result::Result<proto::GetReply, Status>>;

        async fn get(
            &self,
            request: Request<Streaming<proto::GetMessage>>,
        ) -> std::result::Result<Response<Self::GetStream>, Status> {
            let (tx, rx) = mpsc::channel(32);
            let state = self.state.clone();
            let mut input = request.into_inner();
            tokio::spawn(async move {
                handle_get(state, &mut input, tx).await;
            });
            Ok(Response::new(ReceiverStream::new(rx)))
        }

        async fn locate(
            &self,
            _request: Request<proto::LocateRequest>,
        ) -> std::result::Result<Response<proto::LocateReply>, Status> {
            Ok(Response::new(proto::LocateReply::default()))
        }
    }

    struct PutWindow {
        index: u32,
        offset: u64,
        page_count: usize,
        row: bool,
    }

    async fn handle_put(
        state: Arc<FakeState>,
        input: &mut Streaming<proto::PutMessage>,
        tx: mpsc::Sender<std::result::Result<proto::PutReply, Status>>,
    ) {
        let mut key = None;
        let mut page_segment_len = 0u64;
        let mut record_len = 0u64;
        let mut grant_index = 0u32;
        let mut windows: Vec<PutWindow> = Vec::new();
        let mut metadata = vec![0u8; 0];
        let mut stop = state.stop.subscribe();
        loop {
            let message = tokio::select! {
                biased;
                _ = stop.recv() => return,
                message = input.message() => match message {
                    Ok(Some(message)) => message,
                    _ => return,
                },
            };
            match message.m {
                Some(proto::put_message::M::Begin(begin)) => {
                    let begin_key = match begin.key.as_ref() {
                        Some(key) => decode_proto_key(key),
                        None => {
                            let _ = tx.send(Ok(refused(proto::Reason::Invalid))).await;
                            return;
                        }
                    };
                    state
                        .observed_puts
                        .lock()
                        .unwrap()
                        .push(begin_key.consumed_tokens);
                    let refusal = state.put_refusals.lock().unwrap().pop_front();
                    if let Some(reason) = refusal {
                        let delay = state.delay_refusal_ms.load(Ordering::SeqCst);
                        if delay != 0 {
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        }
                        let _ = tx.send(Ok(refused_i32(reason))).await;
                        return;
                    }
                    if state.hold_begin.load(Ordering::SeqCst) {
                        continue;
                    }
                    key = Some(begin_key);
                    let parent = if begin.has_published_parent {
                        begin.published_parent.as_ref().map(decode_proto_key)
                    } else {
                        None
                    };
                    let first =
                        parent.map_or(0, |parent| (parent.consumed_tokens / PAGE_TOKENS) as usize);
                    let tail = begin.chain_pages as usize - first;
                    page_segment_len = page_segment(tail);
                    record_len = record_bytes(tail);
                    let transfer = 1u64;
                    let _ = tx
                        .send(Ok(proto::PutReply {
                            r: Some(proto::put_reply::R::Accept(proto::PutAccept {
                                transfer,
                                first_page: first as u64,
                                parent: parent.map(proto_key),
                                has_parent: parent.is_some(),
                                record_bytes: record_len,
                            })),
                        }))
                        .await;
                    if state.bad_put_range.load(Ordering::SeqCst) {
                        let length = PAGE_BYTES as u64;
                        let offset = state.staging_len;
                        let _ = tx
                            .send(Ok(proto::PutReply {
                                r: Some(proto::put_reply::R::Grant(proto::WindowGrant {
                                    transfer,
                                    index: 0,
                                    record: 0,
                                    record_offset: page_segment_len,
                                    length,
                                    memory: Some(proto::WindowRef {
                                        region: proto::Region::Staging as i32,
                                        offset,
                                        length,
                                    }),
                                    page_first: 0,
                                    page_count: 1,
                                    row: false,
                                })),
                            }))
                            .await;
                        return;
                    }
                    if state.bad_put_tail.load(Ordering::SeqCst) {
                        let length = PAGE_BYTES as u64;
                        let offset = state.allocate(false, length);
                        let _ = tx
                            .send(Ok(proto::PutReply {
                                r: Some(proto::put_reply::R::Grant(proto::WindowGrant {
                                    transfer,
                                    index: 0,
                                    record: 0,
                                    record_offset: page_segment_len
                                        + tail as u64 * PAGE_BYTES as u64,
                                    length,
                                    memory: Some(proto::WindowRef {
                                        region: proto::Region::Staging as i32,
                                        offset,
                                        length,
                                    }),
                                    page_first: (first + tail) as u64,
                                    page_count: 1,
                                    row: false,
                                })),
                            }))
                            .await;
                        return;
                    }
                    let mut page = 0usize;
                    loop {
                        let count = (tail - page).min(state.grant_pages.load(Ordering::SeqCst));
                        let row = page + count == tail;
                        if count == 0 && !row {
                            break;
                        }
                        let length =
                            count as u64 * PAGE_BYTES as u64 + u64::from(row) * ROW_BYTES as u64;
                        let offset = state.allocate(false, length);
                        let record_offset = page_segment_len + page as u64 * PAGE_BYTES as u64;
                        let _ = tx
                            .send(Ok(proto::PutReply {
                                r: Some(proto::put_reply::R::Grant(proto::WindowGrant {
                                    transfer,
                                    index: grant_index,
                                    record: 0,
                                    record_offset,
                                    length,
                                    memory: Some(proto::WindowRef {
                                        region: proto::Region::Staging as i32,
                                        offset,
                                        length,
                                    }),
                                    page_first: if count == 0 { 0 } else { (first + page) as u64 },
                                    page_count: count as u32,
                                    row,
                                })),
                            }))
                            .await;
                        windows.push(PutWindow {
                            index: grant_index,
                            offset,
                            page_count: count,
                            row,
                        });
                        grant_index += 1;
                        page += count;
                        if page >= tail {
                            break;
                        }
                    }
                    if state.close_after_grants.load(Ordering::SeqCst) {
                        return;
                    }
                }
                Some(proto::put_message::M::Filled(filled)) => {
                    if let Some(window) = windows.iter().find(|window| window.index == filled.index)
                    {
                        for (index, expected) in filled.page_xxh3.iter().enumerate() {
                            let bytes = state.read(
                                false,
                                window.offset + index as u64 * PAGE_BYTES as u64,
                                PAGE_BYTES,
                            );
                            if record::checksum(&bytes) != *expected {
                                let _ = tx.send(Ok(refused(proto::Reason::Invalid))).await;
                                return;
                            }
                        }
                        if filled.has_row {
                            let bytes = state.read(
                                false,
                                window.offset + window.page_count as u64 * PAGE_BYTES as u64,
                                ROW_BYTES,
                            );
                            if record::checksum(&bytes) != filled.row_xxh3 {
                                let _ = tx.send(Ok(refused(proto::Reason::Invalid))).await;
                                return;
                            }
                        }
                    } else {
                        let _ = tx.send(Ok(refused(proto::Reason::Invalid))).await;
                        return;
                    }
                }
                Some(proto::put_message::M::Commit(commit)) => {
                    let end = commit.metadata_offset as usize + commit.metadata.len();
                    if metadata.len() < end {
                        metadata.resize(end, 0);
                    }
                    metadata[commit.metadata_offset as usize..end]
                        .copy_from_slice(&commit.metadata);
                    if metadata.len() >= page_segment_len as usize {
                        let mut pages = Vec::new();
                        let mut row = Vec::new();
                        for window in &windows {
                            for index in 0..window.page_count {
                                pages.push(state.read(
                                    false,
                                    window.offset + index as u64 * PAGE_BYTES as u64,
                                    PAGE_BYTES,
                                ));
                            }
                            if window.row {
                                row = state.read(
                                    false,
                                    window.offset + window.page_count as u64 * PAGE_BYTES as u64,
                                    ROW_BYTES,
                                );
                            }
                        }
                        if let Some(key) = key {
                            state.records.lock().unwrap().insert(
                                key_id(&key),
                                FakeRecord {
                                    metadata: metadata[..page_segment_len as usize].to_vec(),
                                    pages,
                                    row,
                                },
                            );
                        }
                        let _ = tx
                            .send(Ok(proto::PutReply {
                                r: Some(proto::put_reply::R::Stored(proto::Stored {
                                    bytes: record_len,
                                })),
                            }))
                            .await;
                        return;
                    }
                }
                Some(proto::put_message::M::Cancel(_)) | None => return,
            }
        }
    }

    type GetWindow = u32;

    async fn handle_get(
        state: Arc<FakeState>,
        input: &mut Streaming<proto::GetMessage>,
        tx: mpsc::Sender<std::result::Result<proto::GetReply, Status>>,
    ) {
        let mut windows: Vec<GetWindow> = Vec::new();
        let mut release = vec![false; 0];
        let mut begin_seen = false;
        let mut stop = state.stop.subscribe();
        loop {
            let message = tokio::select! {
                biased;
                _ = stop.recv() => return,
                message = input.message() => match message {
                    Ok(Some(message)) => message,
                    _ => return,
                },
            };
            match message.m {
                Some(proto::get_message::M::Begin(begin)) => {
                    if begin_seen {
                        let _ = tx.send(Ok(refused_get(proto::Reason::Invalid))).await;
                        return;
                    }
                    begin_seen = true;
                    let refusal = state.get_refusals.lock().unwrap().pop_front();
                    if let Some(reason) = refusal {
                        let _ = tx.send(Ok(refused_get_i32(reason))).await;
                        return;
                    }
                    if state.hold_begin.load(Ordering::SeqCst) {
                        continue;
                    }
                    let keys: Vec<PrefixKey> = begin.chain.iter().map(decode_proto_key).collect();
                    if let Some(first) = keys.first() {
                        state
                            .observed_gets
                            .lock()
                            .unwrap()
                            .push(first.consumed_tokens);
                    }
                    let chain = {
                        let records = state.records.lock().unwrap();
                        let mut chain = Vec::new();
                        let mut missing = false;
                        for key in &keys {
                            let Some(record) = records.get(&key_id(key)) else {
                                missing = true;
                                break;
                            };
                            chain.push(FakeRecord {
                                metadata: record.metadata.clone(),
                                pages: record.pages.clone(),
                                row: record.row.clone(),
                            });
                        }
                        if missing { None } else { Some(chain) }
                    };
                    let Some(chain) = chain else {
                        let _ = tx.send(Ok(refused_get(proto::Reason::Invalid))).await;
                        return;
                    };
                    let transfer = 7;
                    if state.meta_before_leased.load(Ordering::SeqCst) {
                        if let Some(record) = chain.first() {
                            let _ = tx
                                .send(Ok(proto::GetReply {
                                    r: Some(proto::get_reply::R::Meta(proto::RecordMeta {
                                        record: 0,
                                        metadata_offset: 0,
                                        metadata: record.metadata[..HEADER_BYTES].to_vec(),
                                        source: proto::Tier::Dram as i32,
                                    })),
                                }))
                                .await;
                        }
                        return;
                    }
                    let leased = proto::GetReply {
                        r: Some(proto::get_reply::R::Leased(proto::GetLeased {
                            transfer,
                            record_bytes: chain.iter().map(FakeRecord::record_bytes).collect(),
                        })),
                    };
                    let _ = tx.send(Ok(leased.clone())).await;
                    if state.duplicate_leased.load(Ordering::SeqCst) {
                        let _ = tx.send(Ok(leased)).await;
                        return;
                    }
                    let mut index = 0u32;
                    for (record, data) in chain.iter().enumerate() {
                        for (offset, fragment) in data.metadata.chunks(4096).enumerate() {
                            let _ = tx
                                .send(Ok(proto::GetReply {
                                    r: Some(proto::get_reply::R::Meta(proto::RecordMeta {
                                        record: record as u32,
                                        metadata_offset: (offset * 4096) as u64,
                                        metadata: fragment.to_vec(),
                                        source: proto::Tier::Dram as i32,
                                    })),
                                }))
                                .await;
                        }
                        if state.bad_get_row.load(Ordering::SeqCst) && record + 1 == chain.len() {
                            let length = ROW_BYTES as u64;
                            let offset = state.allocate(true, length);
                            state.write(true, offset, &data.row);
                            let _ = tx
                                .send(Ok(proto::GetReply {
                                    r: Some(proto::get_reply::R::Ready(proto::WindowGrant {
                                        transfer,
                                        index,
                                        record: record as u32,
                                        record_offset: page_segment(data.pages.len()),
                                        length,
                                        memory: Some(proto::WindowRef {
                                            region: proto::Region::Arena as i32,
                                            offset,
                                            length,
                                        }),
                                        page_first: 0,
                                        page_count: 0,
                                        row: true,
                                    })),
                                }))
                                .await;
                            return;
                        }
                        let skip = if record == 0 {
                            begin.skip_pages as usize
                        } else {
                            0
                        };
                        let tail = data.pages.len();
                        let origin = (keys[record].consumed_tokens / PAGE_TOKENS) as usize - tail;
                        let mut page = skip;
                        while page < tail {
                            let count = (tail - page).min(state.grant_pages.load(Ordering::SeqCst));
                            let row = record + 1 == chain.len() && page + count == tail;
                            let length = count as u64 * PAGE_BYTES as u64
                                + u64::from(row) * ROW_BYTES as u64;
                            let offset = state.allocate(true, length);
                            let mut bytes = Vec::with_capacity(length as usize);
                            for unit in &data.pages[page..page + count] {
                                bytes.extend_from_slice(unit);
                            }
                            if row {
                                bytes.extend_from_slice(&data.row);
                            }
                            if state.corrupt_get.load(Ordering::SeqCst) && !bytes.is_empty() {
                                bytes[0] ^= 0x80;
                            }
                            state.write(true, offset, &bytes);
                            let _ = tx
                                .send(Ok(proto::GetReply {
                                    r: Some(proto::get_reply::R::Ready(proto::WindowGrant {
                                        transfer,
                                        index,
                                        record: record as u32,
                                        record_offset: page_segment(tail)
                                            + page as u64 * PAGE_BYTES as u64,
                                        length,
                                        memory: Some(proto::WindowRef {
                                            region: proto::Region::Arena as i32,
                                            offset,
                                            length,
                                        }),
                                        page_first: (origin + page) as u64,
                                        page_count: count as u32,
                                        row,
                                    })),
                                }))
                                .await;
                            windows.push(index);
                            index += 1;
                            page += count;
                        }
                        if skip == tail && record + 1 == chain.len() {
                            let offset = state.allocate(true, ROW_BYTES as u64);
                            state.write(true, offset, &data.row);
                            let mut row = data.row.clone();
                            if state.corrupt_get.load(Ordering::SeqCst) && !row.is_empty() {
                                row[0] ^= 0x80;
                                state.write(true, offset, &row);
                            }
                            let _ = tx
                                .send(Ok(proto::GetReply {
                                    r: Some(proto::get_reply::R::Ready(proto::WindowGrant {
                                        transfer,
                                        index,
                                        record: record as u32,
                                        record_offset: page_segment(tail)
                                            + tail as u64 * PAGE_BYTES as u64,
                                        length: ROW_BYTES as u64,
                                        memory: Some(proto::WindowRef {
                                            region: proto::Region::Arena as i32,
                                            offset,
                                            length: ROW_BYTES as u64,
                                        }),
                                        page_first: 0,
                                        page_count: 0,
                                        row: true,
                                    })),
                                }))
                                .await;
                            windows.push(index);
                            index += 1;
                        }
                    }
                    release = vec![false; windows.len()];
                }
                Some(proto::get_message::M::Released(released)) => {
                    if let Some(slot) = windows.iter().position(|index| *index == released.index) {
                        release[slot] = true;
                    }
                    if release.iter().all(|released| *released) {
                        let _ = tx
                            .send(Ok(proto::GetReply {
                                r: Some(proto::get_reply::R::Done(proto::GetDone {})),
                            }))
                            .await;
                        return;
                    }
                }
                Some(proto::get_message::M::Cancel(_)) | None => return,
            }
        }
    }

    fn refused(reason: proto::Reason) -> proto::PutReply {
        proto::PutReply {
            r: Some(proto::put_reply::R::Refused(proto::Refused {
                reason: reason as i32,
                detail: String::new(),
            })),
        }
    }

    fn refused_i32(reason: i32) -> proto::PutReply {
        proto::PutReply {
            r: Some(proto::put_reply::R::Refused(proto::Refused {
                reason,
                detail: String::new(),
            })),
        }
    }

    fn refused_get(reason: proto::Reason) -> proto::GetReply {
        proto::GetReply {
            r: Some(proto::get_reply::R::Refused(proto::Refused {
                reason: reason as i32,
                detail: String::new(),
            })),
        }
    }

    fn refused_get_i32(reason: i32) -> proto::GetReply {
        proto::GetReply {
            r: Some(proto::get_reply::R::Refused(proto::Refused {
                reason,
                detail: String::new(),
            })),
        }
    }

    struct FakeManager {
        dir: PathBuf,
        socket: PathBuf,
        state: Arc<FakeState>,
        shutdown: Option<oneshot::Sender<()>>,
        thread: Option<JoinHandle<()>>,
    }

    impl FakeManager {
        fn start(dir: &Path, namespace: [u8; 32]) -> Self {
            Self::start_with(dir, namespace, GENERATION)
        }

        fn state(dir: &Path, namespace: [u8; 32], generation: u64) -> Arc<FakeState> {
            let arena_path = dir.join("arena");
            let staging_path = dir.join("staging");
            for path in [&arena_path, &staging_path] {
                let file = File::create(path).unwrap();
                file.set_len(ARENA_BYTES.max(STAGING_BYTES)).unwrap();
            }
            let (events, _) = broadcast::channel(64);
            let (stop, _) = broadcast::channel(1);
            Arc::new(FakeState {
                arena_path,
                arena_len: ARENA_BYTES,
                staging_path,
                staging_len: STAGING_BYTES,
                namespace,
                generation: AtomicU64::new(generation),
                entries: Mutex::new(Vec::new()),
                records: Mutex::new(HashMap::new()),
                events,
                stop,
                put_refusals: Mutex::new(VecDeque::new()),
                get_refusals: Mutex::new(VecDeque::new()),
                observed_puts: Mutex::new(Vec::new()),
                observed_gets: Mutex::new(Vec::new()),
                subscribe_count: AtomicU64::new(0),
                arena_alloc: AtomicU64::new(0),
                staging_alloc: AtomicU64::new(0),
                close_after_grants: AtomicBool::new(false),
                hold_begin: AtomicBool::new(false),
                corrupt_get: AtomicBool::new(false),
                gap: AtomicBool::new(false),
                hello_count: AtomicU64::new(0),
                hold_hello: AtomicBool::new(false),
                bad_put_tail: AtomicBool::new(false),
                bad_get_row: AtomicBool::new(false),
                delay_refusal_ms: AtomicU64::new(0),
                meta_before_leased: AtomicBool::new(false),
                duplicate_leased: AtomicBool::new(false),
                bad_put_range: AtomicBool::new(false),
                grant_pages: AtomicUsize::new(WINDOW_PAGES),
            })
        }

        fn start_state(dir: &Path, state: Arc<FakeState>) -> Self {
            let socket = dir.join("manager.sock");
            let _ = fs::remove_file(&socket);
            let listener = StdUnixListener::bind(&socket).unwrap();
            listener.set_nonblocking(true).unwrap();
            let (shutdown, signal) = oneshot::channel();
            let service = FakeService {
                state: state.clone(),
            };
            let thread = thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                runtime.block_on(async move {
                    let listener = tokio::net::UnixListener::from_std(listener).unwrap();
                    let incoming = UnixListenerStream::new(listener);
                    let _ = tonic::transport::Server::builder()
                        .add_service(proto::local_cache_server::LocalCacheServer::new(service))
                        .serve_with_incoming_shutdown(incoming, async {
                            let _ = signal.await;
                        })
                        .await;
                });
            });
            Self {
                dir: dir.to_owned(),
                socket,
                state,
                shutdown: Some(shutdown),
                thread: Some(thread),
            }
        }

        fn start_with(dir: &Path, namespace: [u8; 32], generation: u64) -> Self {
            let state = Self::state(dir, namespace, generation);
            Self::start_state(dir, state)
        }

        fn stop(&mut self) {
            let _ = self.state.stop.send(());
            if let Some(shutdown) = self.shutdown.take() {
                let _ = shutdown.send(());
            }
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
            let _ = fs::remove_file(&self.socket);
        }

        fn restart(&mut self, generation: u64) {
            self.stop();
            self.state.generation.store(generation, Ordering::SeqCst);
            let state = self.state.clone();
            let mut restarted = Self::start_state(&self.dir, state);
            self.shutdown = restarted.shutdown.take();
            self.thread = restarted.thread.take();
            // The restarted handle's only remaining effect would be removing
            // the new socket in Drop.
            std::mem::forget(restarted);
        }

        fn insert_record(&self, key: &PrefixKey, record: FakeRecord) {
            self.state
                .records
                .lock()
                .unwrap()
                .insert(key_id(key), record);
        }

        fn observe_entry(&self, entry: proto::CatalogEntry) {
            self.state.entries.lock().unwrap().push(entry);
        }

        fn subscribe_count(&self) -> u64 {
            self.state.subscribe_count.load(Ordering::SeqCst)
        }

        fn puts(&self) -> Vec<u64> {
            self.state.observed_puts.lock().unwrap().clone()
        }
    }

    impl Drop for FakeManager {
        fn drop(&mut self) {
            self.stop();
        }
    }

    // ---- fixtures ---------------------------------------------------------

    struct Fixtures {
        cache: PrefixCheckpointCache,
        history: PrefixHistory,
        next_page: u32,
        next_snapshot: u32,
    }

    impl Fixtures {
        fn new() -> Self {
            let mut history = PrefixHistory::new(IDENTITY);
            history
                .append(&(0..20_000).collect::<Vec<_>>(), &[])
                .unwrap();
            Self {
                cache: PrefixCheckpointCache::new(IDENTITY, 4096, 4096, PAGE_TOKENS as usize)
                    .unwrap(),
                history,
                next_page: 0,
                next_snapshot: 0,
            }
        }

        fn published(
            &mut self,
            tokens: usize,
            _parent: Option<CheckpointId>,
        ) -> PublishedCheckpoint {
            let tail = tokens / PAGE_TOKENS as usize;
            let start = self.next_page;
            self.next_page += tail as u32 + 1;
            let snapshot = self.next_snapshot;
            self.next_snapshot += 1;
            let pages: Vec<PageId> = (0..tail).map(|i| PageId(start + i as u32)).collect();
            let id = self
                .cache
                .publish(
                    &self.history,
                    &Publication {
                        parent: None,
                        consumed_tokens: tokens as u64,
                        tail_pages: pages.clone(),
                        snapshot: StateSnapshot {
                            id: SnapshotId(snapshot),
                            consumed_tokens: tokens as u64,
                        },
                        ready_after: Fence(0),
                    },
                )
                .unwrap()
                .id();
            PublishedCheckpoint {
                id,
                key: self.history.key_at(tokens as u64).unwrap(),
                parent: None,
                bucket: self.history.bucket_at(tokens as u64).unwrap(),
                chain_pages: pages,
                tail_pages: tail,
                slot: StateSlotId(1),
            }
        }

        fn child(&mut self, tokens: usize, parent: &PublishedCheckpoint) -> PublishedCheckpoint {
            let mut child = self.published(tokens, Some(parent.id));
            let tail = child.chain_pages.len() - parent.chain_pages.len();
            let own = child.chain_pages[child.chain_pages.len() - tail..].to_vec();
            child.chain_pages = parent.chain_pages.iter().copied().chain(own).collect();
            child.tail_pages = tail;
            child.parent = Some(parent.key);
            child
        }
    }

    fn fake_record(
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        first: usize,
    ) -> FakeRecord {
        fake_record_with_row(checkpoint, parent, first, true)
    }

    fn fake_record_with_row(
        checkpoint: &PublishedCheckpoint,
        parent: Option<PrefixKey>,
        first: usize,
        with_row: bool,
    ) -> FakeRecord {
        let tail = checkpoint.chain_pages.len() - first;
        let mut header = Header::new(
            &geometries(),
            IDENTITY,
            fingerprint(),
            checkpoint.key,
            parent,
            checkpoint.bucket,
            tail as u32,
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        let pages: Vec<Vec<u8>> = (0..tail)
            .map(|index| vec![index as u8; PAGE_BYTES])
            .collect();
        let row = if with_row {
            vec![0xA5; ROW_BYTES]
        } else {
            Vec::new()
        };
        if with_row {
            header.row_checksum = record::checksum(&row);
        }
        let mut metadata = header.encode().unwrap().to_vec();
        metadata.resize(header.page_segment as usize, 0);
        for (index, page) in pages.iter().enumerate() {
            let offset = header.checksum_segment() as usize + index * 8;
            metadata[offset..offset + 8].copy_from_slice(&record::checksum(page).to_le_bytes());
        }
        FakeRecord {
            metadata,
            pages,
            row,
        }
    }

    // ---- harness ----------------------------------------------------------

    struct Harness {
        store: PoolStore,
        fake: FakeManager,
        transfers: Arc<Mutex<FakeInner>>,
        fixtures: Fixtures,
        events: VecDeque<StoreEvent>,
        dir: PathBuf,
    }

    impl Harness {
        fn new() -> Self {
            let dir = temp_dir();
            fs::create_dir_all(&dir).unwrap();
            let namespace = namespace_id(&IDENTITY, &fingerprint(), TENANT).unwrap();
            let fake = FakeManager::start(&dir, namespace);
            let layout = Layout::new(&geometries()).unwrap();
            let pool = SlabPool::staging(&layout).unwrap();
            let transfers = Arc::new(Mutex::new(FakeInner::new()));
            let store = PoolStore::with_backend(
                fake.socket.clone(),
                TENANT.into(),
                IDENTITY,
                fingerprint(),
                geometries(),
                pool,
                Box::new(FakeTransfers {
                    inner: transfers.clone(),
                }),
            )
            .unwrap();
            let mut harness = Self {
                store,
                fake,
                transfers,
                fixtures: Fixtures::new(),
                events: VecDeque::new(),
                dir,
            };
            harness.wait_ready();
            harness
        }

        fn poll(&mut self) -> Vec<StoreEvent> {
            let mut events = Vec::new();
            self.store.poll(&mut events).unwrap();
            events
        }

        fn wait_until(&mut self, mut predicate: impl FnMut(&mut Self) -> bool) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !predicate(self) {
                assert!(Instant::now() < deadline, "cache manager test timed out");
                thread::sleep(Duration::from_millis(1));
            }
        }

        fn wait_ready(&mut self) {
            self.wait_until(|harness| {
                let _ = harness.poll();
                harness.store.shared.connected.load(Ordering::Acquire)
                    && harness.store.shared.ready.load(Ordering::Acquire)
            });
        }

        fn event(&mut self) -> StoreEvent {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(event) = self.events.pop_front() {
                    return event;
                }
                let events = self.poll();
                self.events.extend(events);
                assert!(Instant::now() < deadline, "cache manager test timed out");
                thread::sleep(Duration::from_millis(1));
            }
        }

        fn hold_transfers(&self, hold: bool) {
            self.transfers.lock().unwrap().hold = hold;
        }

        fn complete_next(&self) -> bool {
            self.transfers.lock().unwrap().complete_next()
        }

        fn pending(&self, direction: Direction) -> usize {
            self.transfers
                .lock()
                .unwrap()
                .pending
                .iter()
                .filter(|job| job.direction == direction)
                .count()
        }

        fn h2d(&self, page: u32) -> Option<Vec<u8>> {
            self.transfers.lock().unwrap().h2d.get(&page).cloned()
        }

        fn h2d_row(&self) -> Option<Vec<u8>> {
            self.transfers.lock().unwrap().h2d_row.clone()
        }

        fn insert_entry(&mut self, checkpoint: &PublishedCheckpoint, tier: proto::Tier) {
            self.fake.observe_entry(catalog_entry(
                checkpoint,
                checkpoint.tail_pages,
                tier,
                true,
                checkpoint.tail_pages as u64 * 8192 + 4096,
                false,
            ));
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = self.store.close();
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn temp_dir() -> PathBuf {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "selene-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ))
    }

    // ---- tests ------------------------------------------------------------

    #[test]
    fn proto_digest_and_namespace_vectors() {
        let digest = sha256(include_bytes!("../../../proto/selene.proto"));
        assert_eq!(
            hex(&digest),
            "88c0ec67f6e904c35f822810435f4428374efe2d4a62a55b538fac7278215e47"
        );
        let identity = [0x63; 32];
        let fingerprint = [0x1b; 16];
        assert_eq!(
            hex(&namespace_id(&identity, &fingerprint, "test").unwrap()),
            "83bbb4befe798c5d717456d55695acc1d166d45c214e54e253f249cd025d4040"
        );
        assert_eq!(
            hex(&namespace_id(&identity, &fingerprint, "").unwrap()),
            "6fbf3a101213ca13f48c6ab6ae91e7e1e35fd409de023b30696e32b18c97ed74"
        );
    }

    #[test]
    fn tenant_validation_matches_the_wire_rule() {
        for valid in ["a", "A-z_0.9", &"x".repeat(64)] {
            assert!(crate::offload::validate_tenant(Some(valid)).is_ok());
        }
        assert!(crate::offload::validate_tenant(None).unwrap().is_empty());
        for invalid in ["", " ", "a/b", &"x".repeat(65)] {
            assert!(crate::offload::validate_tenant(Some(invalid)).is_err());
        }
    }

    #[test]
    fn manager_off_and_startup_refusals() {
        let prepared = crate::offload::prepare(
            &PrefixCacheOffload::default(),
            Path::new("missing"),
            [0; 32],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert!(matches!(
            prepared,
            crate::offload::Preparation::None(reason) if reason == "disabled"
        ));
        let base = PrefixCacheOffload {
            manager: Some("/run/selene/default.sock".into()),
            ..Default::default()
        };
        for (options, expected) in [
            (
                PrefixCacheOffload {
                    memory_bytes: Some(1),
                    ..base.clone()
                },
                "the cache manager replaces the local host tiers",
            ),
            (
                PrefixCacheOffload {
                    disk: Some(temp_dir()),
                    disk_limit: Some(0),
                    ..base.clone()
                },
                "the cache manager replaces the local host tiers",
            ),
        ] {
            let error = crate::offload::prepare(
                &options,
                Path::new("missing"),
                [0; 32],
                Vec::new(),
                Vec::new(),
            )
            .err()
            .unwrap();
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn outcomes_map_every_manager_refusal() {
        for (reason, expected) in [
            (proto::Reason::Present as i32, "present"),
            (proto::Reason::Inflight as i32, "deferred"),
            (proto::Reason::Budget as i32, "deferred"),
            (proto::Reason::Quota as i32, "deferred"),
            (proto::Reason::Queue as i32, "deferred"),
            (proto::Reason::Oversize as i32, "error"),
            (proto::Reason::Invalid as i32, "error"),
            (proto::Reason::Io as i32, "error"),
        ] {
            let mut harness = Harness::new();
            harness
                .fake
                .state
                .put_refusals
                .lock()
                .unwrap()
                .push_back(reason);
            let checkpoint = harness.fixtures.published(130, None);
            assert_eq!(
                harness.store.store(&checkpoint).unwrap(),
                StoreOutcome::Queued
            );
            match (expected, harness.event()) {
                ("present", StoreEvent::Stored { result: Ok(0), .. }) => {}
                ("deferred", StoreEvent::Deferred { .. }) => {}
                ("error", StoreEvent::Stored { result: Err(_), .. }) => {}
                (_, other) => panic!("{reason}: {other:?}"),
            }
        }
    }

    #[test]
    fn store_prechecks_inflight_queue_and_budget() {
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        let checkpoint = harness.fixtures.published(130, None);
        assert_eq!(
            harness.store.store(&checkpoint).unwrap(),
            StoreOutcome::Queued
        );
        assert_eq!(
            harness.store.store(&checkpoint).unwrap(),
            StoreOutcome::Retry(StoreReason::Inflight)
        );
        for tokens in [194, 258, 322] {
            let checkpoint = harness.fixtures.published(tokens, None);
            assert_eq!(
                harness.store.store(&checkpoint).unwrap(),
                StoreOutcome::Queued
            );
        }
        let extra = harness.fixtures.published(386, None);
        assert_eq!(
            harness.store.store(&extra).unwrap(),
            StoreOutcome::Retry(StoreReason::Queue)
        );
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(false, Ordering::Release);
    }

    #[test]
    fn put_accepts_exact_tail_row_and_decodable_commit() {
        let mut harness = Harness::new();
        let root = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &root);
        harness.store.store(&root).unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Stored { result: Ok(_), .. }
        ));
        harness.store.store(&child).unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Stored { result: Ok(_), .. }
        ));
        let records = harness.fake.state.records.lock().unwrap();
        let record = records.get(&key_id(&child.key)).unwrap();
        let header = Header::decode(&record.metadata).unwrap();
        assert_eq!(header.parent, Some(root.key));
        assert_eq!(header.key, child.key);
        assert_eq!(header.tail_pages as usize, child.tail_pages);
        assert_eq!(record.pages.len(), child.tail_pages);
        for (index, page) in record.pages.iter().enumerate() {
            let expected = child.chain_pages[child.chain_pages.len() - child.tail_pages + index];
            assert!(page.iter().all(|byte| *byte == expected.0 as u8));
        }
        assert_eq!(record.row, vec![0xA5; ROW_BYTES]);
        assert_eq!(header.tail_pages as usize, record.pages.len());
    }

    #[test]
    fn put_grants_use_absolute_page_indices() {
        let mut fixtures = Fixtures::new();
        let parent = fixtures.published(70 * 64, None);
        let child = fixtures.child(140 * 64, &parent);
        let header = Header::new(
            &geometries(),
            IDENTITY,
            fingerprint(),
            child.key,
            Some(parent.key),
            child.bucket,
            child.tail_pages as u32,
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        let mut put = PutState {
            checkpoint: child,
            header: Some(header),
            first: 70,
            transfer: Some(1),
            grants: VecDeque::new(),
            grant_index: 0,
            next_page: 0,
            filled_pages: 0,
            row_filled: false,
            hashes: vec![0; 70],
            committed: false,
            tx: None,
        };
        let grant = |index, page_first, page_count, row| Grant {
            transfer: 1,
            index,
            record: 0,
            record_offset: 0,
            length: page_count as u64 * 77824 + u64::from(row) * 1_003_520,
            arena: false,
            offset: 0,
            page_first,
            page_count,
            row,
        };
        let validate = |grant: &Grant, put: &PutState| {
            validate_put_grant(grant, put, 77824, 1_003_520, 4_194_304)
        };
        assert!(validate(&grant(0, 70, 53, false), &put).is_ok());
        // A tail-relative page index is not the protocol's absolute index.
        assert!(validate(&grant(0, 0, 53, false), &put).is_err());
        put.grant_index = 1;
        put.next_page = 53;
        assert!(validate(&grant(1, 123, 17, true), &put).is_ok());
        assert!(validate(&grant(1, 53, 17, true), &put).is_err());
        put.next_page = 70;
        put.grant_index = 2;
        assert!(validate(&grant(2, 0, 0, true), &put).is_ok());
    }

    #[test]
    fn put_child_uses_absolute_page_first_across_windows() {
        let mut harness = Harness::new();
        harness.fake.state.grant_pages.store(53, Ordering::SeqCst);
        let parent = harness.fixtures.published(70 * 64, None);
        let child = harness.fixtures.child(140 * 64, &parent);
        assert_eq!(child.chain_pages.len(), 140);
        assert_eq!(child.tail_pages, 70);
        harness.store.store(&child).unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Stored { result: Ok(_), .. }
        ));
        let records = harness.fake.state.records.lock().unwrap();
        let record = records.get(&key_id(&child.key)).unwrap();
        assert_eq!(record.pages.len(), 70);
        for (index, page) in record.pages.iter().enumerate() {
            let expected = child.chain_pages[70 + index];
            assert!(
                page.iter().all(|byte| *byte == expected.0 as u8),
                "page {index}"
            );
        }
        assert_eq!(record.row, vec![0xA5; ROW_BYTES]);
    }

    #[test]
    fn put_child_row_only_grant_completes() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(131, &parent);
        assert_eq!(child.tail_pages, 0);
        assert_eq!(child.chain_pages, parent.chain_pages);
        harness.store.store(&child).unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Stored { result: Ok(_), .. }
        ));
        let records = harness.fake.state.records.lock().unwrap();
        let record = records.get(&key_id(&child.key)).unwrap();
        assert!(record.pages.is_empty());
        assert_eq!(record.row, vec![0xA5; ROW_BYTES]);
    }

    #[test]
    fn restore_child_chain_with_a_non_zero_first_record() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(256, None);
        let child = harness.fixtures.child(384, &parent);
        harness
            .fake
            .insert_record(&parent.key, fake_record(&parent, None, 0));
        harness.fake.insert_record(
            &child.key,
            fake_record(&child, Some(parent.key), parent.chain_pages.len()),
        );
        harness.insert_entry(&parent, proto::Tier::Dram);
        harness.insert_entry(&child, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(384), 129, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(384), 129, u64::MAX).unwrap();
        assert_eq!(hit.chain.skip_pages, 2);
        assert_eq!(hit.pages, 4);
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: child.key,
                    pages: (10..14).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Ok(_), .. }
        ));
        assert_eq!(harness.h2d(10).unwrap(), vec![2u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(11).unwrap(), vec![3u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(12).unwrap(), vec![0u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(13).unwrap(), vec![1u8; PAGE_BYTES]);
        assert_eq!(harness.h2d_row().unwrap(), vec![0xA5; ROW_BYTES]);
    }

    #[test]
    fn restore_child_chain_uses_chain_absolute_page_indices() {
        let mut harness = Harness::new();
        harness.fake.state.grant_pages.store(53, Ordering::SeqCst);
        let parent = harness.fixtures.published(70 * 64, None);
        let child = harness.fixtures.child(140 * 64, &parent);
        harness
            .fake
            .insert_record(&parent.key, fake_record(&parent, None, 0));
        harness.fake.insert_record(
            &child.key,
            fake_record(&child, Some(parent.key), parent.chain_pages.len()),
        );
        harness.insert_entry(&parent, proto::Tier::Dram);
        harness.insert_entry(&child, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness
                .store
                .lookup(&history(140 * 64), 0, u64::MAX)
                .is_some()
        });
        let hit = harness
            .store
            .lookup(&history(140 * 64), 0, u64::MAX)
            .unwrap();
        assert_eq!(hit.pages, 140);
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: child.key,
                    pages: (300..440).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Ok(_), .. }
        ));
        // Record 0 owns chain pages 0..69; record 1 owns 70..139.
        assert_eq!(harness.h2d(300).unwrap(), vec![0u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(369).unwrap(), vec![69u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(370).unwrap(), vec![0u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(439).unwrap(), vec![69u8; PAGE_BYTES]);
        assert_eq!(harness.h2d_row().unwrap(), vec![0xA5; ROW_BYTES]);
    }

    #[test]
    fn transient_failures_defer_only_after_the_d2h_drain() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .close_after_grants
            .store(true, Ordering::SeqCst);
        harness.hold_transfers(true);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        harness.wait_until(|harness| {
            harness.poll();
            harness.pending(Direction::Out) == 1
        });
        // The manager has gone away, but the D2H reply is still outstanding.
        harness.wait_until(|harness| {
            harness.poll();
            !harness.store.shared.connected.load(Ordering::Acquire)
        });
        assert!(harness.poll().is_empty());
        assert!(harness.complete_next());
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        assert_eq!(harness.store.stats().outbound_in_flight, 0);
    }

    #[test]
    fn progress_counts_snapshots_rooms_and_slot_release() {
        let mut harness = Harness::new();
        let root = harness.fixtures.published(130, None);
        harness.fake.observe_entry(catalog_entry(
            &root,
            2,
            proto::Tier::Dram,
            false,
            20_000,
            false,
        ));
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        let progress = harness.store.stats().progress;
        assert!(progress >= 1);
        harness
            .fake
            .state
            .publish_live(proto::catalog_event::Event::Room(proto::Room {
                bytes: 4096,
            }));
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.stats().progress > progress
        });
    }

    #[test]
    fn window_credits_bound_two_windows_in_each_direction() {
        let mut harness = Harness::new();
        harness.hold_transfers(true);
        let checkpoint = harness.fixtures.published(8320, None);
        assert_eq!(
            harness.store.store(&checkpoint).unwrap(),
            StoreOutcome::Queued
        );
        harness.wait_until(|harness| {
            harness.poll();
            harness.pending(Direction::Out) == 2
        });
        assert!(harness.pending(Direction::Out) <= 2);
        let mut stored = false;
        harness.wait_until(|harness| {
            while harness.complete_next() {
                harness.poll();
            }
            for event in harness.poll() {
                if matches!(event, StoreEvent::Stored { result: Ok(_), .. }) {
                    stored = true;
                }
            }
            stored
        });
    }

    #[test]
    fn fetch_round_trips_and_detects_get_time_corruption() {
        let mut harness = Harness::new();
        let root = harness.fixtures.published(130, None);
        harness
            .fake
            .insert_record(&root.key, fake_record(&root, None, 0));
        harness.insert_entry(&root, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(130), 0, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(130), 0, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Memory);
        let ticket = InboundTicket {
            key: hit.key,
            pages: (0..hit.pages as u32).map(PageId).collect(),
            slot: StateSlotId(2),
        };
        harness.store.fetch(hit, ticket).unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Ok(_), .. }
        ));
        assert_eq!(harness.h2d(0).unwrap(), vec![0u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(1).unwrap(), vec![1u8; PAGE_BYTES]);

        let child = harness.fixtures.published(258, None);
        harness
            .fake
            .insert_record(&child.key, fake_record(&child, None, 0));
        harness.insert_entry(&child, proto::Tier::Nvme);
        harness.fake.restart(GENERATION + 2);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(258), 0, u64::MAX).is_some()
        });
        harness.fake.state.corrupt_get.store(true, Ordering::SeqCst);
        let hit = harness.store.lookup(&history(258), 0, u64::MAX).unwrap();
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: child.key,
                    pages: (0..4).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Err(_), .. }
        ));
        assert_eq!(harness.store.stats().fetch_failed, 1);
    }

    #[test]
    fn deadline_cancels_both_directions_after_the_drain() {
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .deadline_ms
            .store(50, Ordering::Release);
        harness.fake.state.hold_begin.store(true, Ordering::SeqCst);
        let put = harness.fixtures.published(130, None);
        harness.store.store(&put).unwrap();
        let mut deferred = false;
        harness.wait_until(|harness| {
            for event in harness.poll() {
                if matches!(event, StoreEvent::Deferred { .. }) {
                    deferred = true;
                }
            }
            deferred
        });
        let root = harness.fixtures.published(194, None);
        harness
            .fake
            .insert_record(&root.key, fake_record(&root, None, 0));
        harness.insert_entry(&root, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(194), 0, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(194), 0, u64::MAX).unwrap();
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: root.key,
                    pages: (0..3).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        let mut failed = false;
        harness.wait_until(|harness| {
            for event in harness.poll() {
                if matches!(event, StoreEvent::Fetched { result: Err(_), .. }) {
                    failed = true;
                }
            }
            failed
        });
    }

    #[test]
    fn stale_generation_never_drops_a_cuda_reply() {
        let mut harness = Harness::new();
        harness.hold_transfers(true);
        harness
            .fake
            .state
            .close_after_grants
            .store(true, Ordering::SeqCst);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        harness.wait_until(|harness| {
            harness.poll();
            harness.pending(Direction::Out) == 1
        });
        harness.wait_until(|harness| {
            harness.poll();
            !harness.store.shared.connected.load(Ordering::Acquire)
        });
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.shared.connected.load(Ordering::Acquire)
                && harness.store.shared.generation.load(Ordering::Acquire) != GENERATION
        });
        assert!(harness.poll().is_empty());
        assert!(harness.complete_next());
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
    }

    #[test]
    fn close_waits_for_the_transfer_drain() {
        let mut harness = Harness::new();
        harness.hold_transfers(true);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        harness.wait_until(|harness| {
            harness.poll();
            harness.pending(Direction::Out) == 1
        });
        let transfers = harness.transfers.clone();
        let release = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            transfers.lock().unwrap().complete_next();
        });
        let start = Instant::now();
        harness.store.close().unwrap();
        assert!(start.elapsed() >= Duration::from_millis(20));
        assert!(start.elapsed() < crate::offload::transfer::CLOSE_TIMEOUT);
        release.join().unwrap();
    }

    #[test]
    fn mirror_orders_snapshots_bounds_application_and_resubscribes_on_gaps() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        harness.fake.observe_entry(catalog_entry(
            &parent,
            2,
            proto::Tier::Dram,
            true,
            20_000,
            false,
        ));
        harness.fake.observe_entry(catalog_entry(
            &child,
            2,
            proto::Tier::Nvme,
            true,
            30_000,
            false,
        ));
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        assert!(harness.store.lookup(&history(258), 0, u64::MAX).is_some());
        harness
            .fake
            .state
            .publish_live(proto::catalog_event::Event::Removed(proto_key(child.key)));
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.mirror.len() == 1
        });
        let before = harness.fake.subscribe_count();
        harness.fake.state.gap.store(true, Ordering::SeqCst);
        harness.fake.state.publish_live(live_entry(
            &parent,
            2,
            proto::Tier::Dram,
            true,
            20_000,
            false,
        ));
        harness.wait_until(|harness| {
            harness.poll();
            harness.fake.subscribe_count() > before
        });
        harness.wait_ready();
    }

    #[test]
    fn unreadable_ancestry_anchor_is_a_miss() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        harness.fake.observe_entry(catalog_entry(
            &parent,
            2,
            proto::Tier::Nvme,
            true,
            20_000,
            true,
        ));
        harness.fake.observe_entry(catalog_entry(
            &child,
            2,
            proto::Tier::Nvme,
            true,
            30_000,
            false,
        ));
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        assert!(harness.store.lookup(&history(258), 0, u64::MAX).is_none());
    }

    #[test]
    fn mirror_retires_replaced_indexes_off_the_serving_thread() {
        let mut harness = Harness::new();
        let (notice, dropped) = std::sync::mpsc::channel();
        let drop_notice = Arc::new(super::super::mirror::DropNotice(notice));
        {
            let mirror = &mut harness.store.mirror;
            let checkpoint = harness.fixtures.published(130, None);
            let entry = super::super::mirror::Entry {
                key: checkpoint.key,
                parent: None,
                bucket: checkpoint.bucket,
                tail: 2,
                row: true,
                location: super::super::mirror::PoolRecord {
                    tier: super::super::mirror::LocationTier::Dram,
                    durable: true,
                    bytes: 4096,
                    unreadable: false,
                    test: super::super::mirror::TestHooks {
                        drop_notice: Some(drop_notice.clone()),
                    },
                },
            };
            entry.insert(mirror).unwrap();
        }
        drop(drop_notice);
        {
            let mut inbox = harness.store.shared.mirror.lock().unwrap();
            assert!(inbox.push(Update::Snapshot(
                crate::offload::index::Index::new(PAGE_TOKENS),
                crate::offload::StoreStats::default(),
            )));
        }
        harness.wait_until(|harness| {
            harness.poll();
            harness
                .store
                .shared
                .mirror
                .lock()
                .unwrap()
                .retired
                .is_none()
        });
        let thread = dropped.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_ne!(thread, thread::current().id());
    }

    #[test]
    fn mirror_applies_at_most_4096_updates_per_poll() {
        let mut harness = Harness::new();
        harness.wait_until(|harness| {
            harness.poll();
            harness
                .store
                .shared
                .mirror
                .lock()
                .unwrap()
                .retired
                .is_none()
        });
        let mut parent = None;
        let mut updates = VecDeque::new();
        for index in 1..=4097u64 {
            let key = PrefixKey {
                consumed_tokens: index * PAGE_TOKENS,
                digest: (index as u128).to_le_bytes(),
            };
            updates.push_back(Update::Added(super::super::mirror::Entry {
                key,
                parent,
                bucket: [0; 16],
                tail: 1,
                row: true,
                location: super::super::mirror::PoolRecord {
                    tier: super::super::mirror::LocationTier::Dram,
                    durable: false,
                    bytes: 4096,
                    unreadable: false,
                    test: super::super::mirror::TestHooks::default(),
                },
            }));
            parent = Some(key);
        }
        {
            let mut inbox = harness.store.shared.mirror.lock().unwrap();
            for update in updates {
                assert!(inbox.push(update));
            }
        }
        harness.poll();
        assert_eq!(harness.store.mirror.len(), 4096);
        harness.poll();
        assert_eq!(harness.store.mirror.len(), 4097);
    }

    #[test]
    fn credits_bound_four_operations_in_each_direction() {
        let mut harness = Harness::new();
        harness.fake.state.hold_begin.store(true, Ordering::SeqCst);
        let root = harness.fixtures.published(130, None);
        super::super::mirror::Entry {
            key: root.key,
            parent: None,
            bucket: root.bucket,
            tail: root.tail_pages,
            row: true,
            location: super::super::mirror::PoolRecord {
                tier: super::super::mirror::LocationTier::Dram,
                durable: false,
                bytes: 20_000,
                unreadable: false,
                test: super::super::mirror::TestHooks::default(),
            },
        }
        .insert(&mut harness.store.mirror)
        .unwrap();
        let hit = harness.store.lookup(&history(130), 0, u64::MAX).unwrap();
        for _ in 0..4 {
            harness
                .store
                .fetch(
                    hit.clone(),
                    InboundTicket {
                        key: hit.key,
                        pages: (0..hit.pages as u32).map(PageId).collect(),
                        slot: StateSlotId(2),
                    },
                )
                .unwrap();
        }
        assert_eq!(harness.store.stats().inbound_in_flight, 4);
        assert!(
            harness
                .store
                .fetch(
                    hit.clone(),
                    InboundTicket {
                        key: hit.key,
                        pages: (0..hit.pages as u32).map(PageId).collect(),
                        slot: StateSlotId(2),
                    },
                )
                .is_err()
        );
    }

    #[test]
    fn stats_use_mirror_tiers_and_chain_tier() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        harness.fake.observe_entry(catalog_entry(
            &parent,
            2,
            proto::Tier::Nvme,
            true,
            20_000,
            false,
        ));
        harness.fake.observe_entry(catalog_entry(
            &child,
            2,
            proto::Tier::Dram,
            true,
            30_000,
            false,
        ));
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        let stats = harness.store.stats();
        assert_eq!((stats.memory_records, stats.disk_records), (1, 2));
        let hit = harness.store.lookup(&history(258), 0, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Disk);
        let hit = harness.store.lookup(&history(258), 130, u64::MAX).unwrap();
        assert_eq!(hit.tier, Tier::Memory);
        harness
            .fake
            .state
            .publish_live(proto::catalog_event::Event::Removed(proto_key(child.key)));
        harness.wait_until(|harness| {
            harness.poll();
            !harness.store.mirror.contains(&child.key)
        });
        assert_eq!(
            harness.store.mirror.node(&parent.key).unwrap().children(),
            0
        );
    }

    #[test]
    fn coalescing_drops_a_queued_ancestor() {
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        let ancestor = harness.fixtures.published(130, None);
        let mut descendant = harness.fixtures.published(258, None);
        descendant.chain_pages = ancestor
            .chain_pages
            .iter()
            .copied()
            .chain([PageId(999)])
            .chain([PageId(998)])
            .collect();
        descendant.tail_pages = descendant.chain_pages.len();
        harness.store.store(&ancestor).unwrap();
        harness.store.store(&descendant).unwrap();
        let before = harness.fake.puts();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(false, Ordering::Release);
        let mut first = harness.event();
        match &first {
            StoreEvent::Deferred { checkpoint } if *checkpoint == ancestor.id => {}
            other => panic!("{other:?}, puts {:?}", harness.fake.puts()),
        }
        assert_eq!(harness.store.superseded_count, 1);
        let stored = harness.event();
        assert!(matches!(
            &stored,
            StoreEvent::Stored { checkpoint, .. } if *checkpoint == descendant.id
        ));
        let sent = harness.fake.puts();
        assert_eq!(sent[before.len()..].to_vec(), vec![258]);
        first = stored;
        let _ = first;
    }

    #[test]
    fn coalescing_keeps_started_copies_needed_parents_and_siblings() {
        // A copy already started is never dropped.
        let mut harness = Harness::new();
        harness.hold_transfers(true);
        let ancestor = harness.fixtures.published(130, None);
        harness.store.store(&ancestor).unwrap();
        harness.wait_until(|harness| {
            harness.poll();
            harness.pending(Direction::Out) == 1
        });
        let mut descendant = harness.fixtures.published(258, None);
        descendant.chain_pages = ancestor
            .chain_pages
            .iter()
            .copied()
            .chain([PageId(999), PageId(998)])
            .collect();
        descendant.tail_pages = descendant.chain_pages.len();
        harness.store.store(&descendant).unwrap();
        harness.complete_next();
        assert!(matches!(
            harness.event(),
            StoreEvent::Stored { checkpoint, .. } if checkpoint == ancestor.id
        ));
        harness.complete_next();
        assert!(matches!(harness.event(), StoreEvent::Stored { .. }));

        // A published parent keeps its queued ancestor.
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        harness.store.store(&parent).unwrap();
        harness.store.store(&child).unwrap();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(false, Ordering::Release);
        let mut events = Vec::new();
        harness.wait_until(|harness| {
            events.extend(harness.poll());
            events
                .iter()
                .filter(|event| matches!(event, StoreEvent::Stored { .. }))
                .count()
                == 2
        });
        assert!(harness.fake.puts().contains(&130));
        assert!(harness.fake.puts().contains(&258));

        // Sibling branches are both kept.
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        let mut left = harness.fixtures.published(130, None);
        left.chain_pages = vec![PageId(10), PageId(11)];
        let mut right = harness.fixtures.published(194, None);
        right.chain_pages = vec![PageId(20), PageId(21), PageId(22)];
        harness.store.store(&left).unwrap();
        harness.store.store(&right).unwrap();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(false, Ordering::Release);
        let mut stored = 0;
        harness.wait_until(|harness| {
            for event in harness.poll() {
                if matches!(event, StoreEvent::Stored { .. }) {
                    stored += 1;
                }
            }
            stored == 2
        });
    }

    #[test]
    fn superseded_retry_returns_present_with_no_event_and_no_pin() {
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        let ancestor = harness.fixtures.published(130, None);
        let mut descendant = harness.fixtures.published(258, None);
        descendant.chain_pages = ancestor
            .chain_pages
            .iter()
            .copied()
            .chain([PageId(999), PageId(998)])
            .collect();
        descendant.tail_pages = descendant.chain_pages.len();
        harness.store.store(&ancestor).unwrap();
        harness.store.store(&descendant).unwrap();
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(false, Ordering::Release);
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        assert!(matches!(harness.event(), StoreEvent::Stored { .. }));
        assert_eq!(
            harness.store.store(&ancestor).unwrap(),
            StoreOutcome::Present
        );
        assert!(harness.poll().is_empty());
        assert_eq!(harness.store.superseded_count, 1);
        assert_eq!(harness.store.stats().outbound_in_flight, 0);
    }

    #[test]
    fn restores_are_not_blocked_by_a_put_budget_refusal() {
        let mut harness = Harness::new();
        let root = harness.fixtures.published(130, None);
        harness
            .fake
            .insert_record(&root.key, fake_record(&root, None, 0));
        harness.insert_entry(&root, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness
            .fake
            .state
            .put_refusals
            .lock()
            .unwrap()
            .push_back(proto::Reason::Budget as i32);
        let put = harness.fixtures.published(194, None);
        harness.store.store(&put).unwrap();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(130), 0, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(130), 0, u64::MAX).unwrap();
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: root.key,
                    pages: (0..2).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        let mut deferred = false;
        let mut fetched = false;
        harness.wait_until(|harness| {
            for event in harness.poll() {
                match event {
                    StoreEvent::Deferred { .. } => deferred = true,
                    StoreEvent::Fetched { result: Ok(_), .. } => fetched = true,
                    _ => {}
                }
            }
            deferred && fetched
        });
    }

    #[test]
    fn restore_accepts_a_page_only_ancestor() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        harness
            .fake
            .insert_record(&parent.key, fake_record_with_row(&parent, None, 0, false));
        harness.fake.insert_record(
            &child.key,
            fake_record_with_row(&child, Some(parent.key), parent.chain_pages.len(), true),
        );
        harness.fake.observe_entry(catalog_entry_with_row(
            &parent,
            parent.tail_pages,
            proto::Tier::Nvme,
            true,
            20_000,
            false,
            false,
        ));
        harness.fake.observe_entry(catalog_entry(
            &child,
            child.tail_pages,
            proto::Tier::Nvme,
            true,
            30_000,
            false,
        ));
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(258), 0, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(258), 0, u64::MAX).unwrap();
        assert_eq!(hit.pages, 4);
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: child.key,
                    pages: (0..4).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Ok(_), .. }
        ));
        assert_eq!(harness.h2d(0).unwrap(), vec![0u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(1).unwrap(), vec![1u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(2).unwrap(), vec![0u8; PAGE_BYTES]);
        assert_eq!(harness.h2d(3).unwrap(), vec![1u8; PAGE_BYTES]);
    }

    #[test]
    fn restore_accepts_a_row_only_window() {
        let mut harness = Harness::new();
        let root = harness.fixtures.published(130, None);
        harness
            .fake
            .insert_record(&root.key, fake_record(&root, None, 0));
        harness.insert_entry(&root, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(130), 129, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(130), 129, u64::MAX).unwrap();
        assert_eq!(hit.pages, 0);
        assert_eq!(hit.chain.skip_pages, 2);
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: root.key,
                    pages: Vec::new(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Ok(_), .. }
        ));
        assert_eq!(harness.h2d_row().unwrap(), vec![0xA5; ROW_BYTES]);
    }

    #[test]
    fn malformed_put_grant_is_a_session_failure() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .bad_put_tail
            .store(true, Ordering::SeqCst);
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        harness.store.store(&child).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        assert_eq!(harness.store.stats().outbound_in_flight, 0);
        harness
            .fake
            .state
            .bad_put_tail
            .store(false, Ordering::SeqCst);
        harness.wait_ready();
    }

    #[test]
    fn malformed_get_window_is_a_session_failure() {
        let mut harness = Harness::new();
        harness.fake.state.bad_get_row.store(true, Ordering::SeqCst);
        let root = harness.fixtures.published(130, None);
        harness
            .fake
            .insert_record(&root.key, fake_record(&root, None, 0));
        harness.insert_entry(&root, proto::Tier::Dram);
        harness.fake.restart(GENERATION + 1);
        harness.wait_ready();
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.lookup(&history(130), 0, u64::MAX).is_some()
        });
        let hit = harness.store.lookup(&history(130), 0, u64::MAX).unwrap();
        harness
            .store
            .fetch(
                hit,
                InboundTicket {
                    key: root.key,
                    pages: (0..2).map(PageId).collect(),
                    slot: StateSlotId(2),
                },
            )
            .unwrap();
        assert!(matches!(
            harness.event(),
            StoreEvent::Fetched { result: Err(_), .. }
        ));
        harness
            .fake
            .state
            .bad_get_row
            .store(false, Ordering::SeqCst);
        harness.wait_ready();
    }

    #[test]
    fn reconnect_times_out_hello_and_recovers() {
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .deadline_ms
            .store(50, Ordering::Release);
        let before = harness.fake.state.hello_count.load(Ordering::SeqCst);
        harness.fake.state.hold_hello.store(true, Ordering::SeqCst);
        harness
            .fake
            .state
            .bad_put_tail
            .store(true, Ordering::SeqCst);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        harness
            .fake
            .state
            .bad_put_tail
            .store(false, Ordering::SeqCst);
        // The Hello RPC itself must time out, so the client keeps retrying.
        harness.wait_until(|harness| {
            harness.poll();
            harness.fake.state.hello_count.load(Ordering::SeqCst) >= before + 2
        });
        harness.fake.state.hold_hello.store(false, Ordering::SeqCst);
        harness.wait_ready();
    }

    #[test]
    fn expiry_rebuilds_the_session() {
        let mut harness = Harness::new();
        harness
            .store
            .shared
            .test
            .deadline_ms
            .store(50, Ordering::Release);
        let generation = harness.store.shared.generation.load(Ordering::Acquire);
        harness.fake.state.hold_begin.store(true, Ordering::SeqCst);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.shared.connected.load(Ordering::Acquire)
                && harness.store.shared.generation.load(Ordering::Acquire) != generation
        });
        harness.fake.state.hold_begin.store(false, Ordering::SeqCst);
        harness.wait_ready();
    }

    #[test]
    fn room_before_a_budget_refusal_allows_the_retry() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .delay_refusal_ms
            .store(200, Ordering::SeqCst);
        harness
            .fake
            .state
            .put_refusals
            .lock()
            .unwrap()
            .push_back(proto::Reason::Budget as i32);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        harness.wait_until(|harness| {
            harness.poll();
            harness.fake.puts().contains(&130)
        });
        let available = harness.store.shared.wake.load(Ordering::Acquire);
        harness
            .fake
            .state
            .publish_live(proto::catalog_event::Event::Room(proto::Room {
                bytes: 4096,
            }));
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.shared.wake.load(Ordering::Acquire) > available
        });
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        assert_eq!(
            harness.store.store(&checkpoint).unwrap(),
            StoreOutcome::Queued
        );
    }

    #[test]
    fn deferred_publications_coalesce_before_budget_rejection() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .put_refusals
            .lock()
            .unwrap()
            .push_back(proto::Reason::Budget as i32);
        let ancestor = harness.fixtures.published(130, None);
        harness.store.store(&ancestor).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        let available = harness.store.shared.wake.load(Ordering::Acquire);
        harness
            .fake
            .state
            .publish_live(proto::catalog_event::Event::Room(proto::Room {
                bytes: 4096,
            }));
        harness.wait_until(|harness| {
            harness.poll();
            harness.store.shared.wake.load(Ordering::Acquire) > available
        });
        let mut descendant = harness.fixtures.published(258, None);
        descendant.chain_pages = ancestor
            .chain_pages
            .iter()
            .copied()
            .chain([PageId(999), PageId(998)])
            .collect();
        descendant.tail_pages = descendant.chain_pages.len();
        assert_eq!(
            harness.store.store(&descendant).unwrap(),
            StoreOutcome::Queued
        );
        // Simulate the unchanged budget counter the v1 rule compares.
        harness.store.budget = Some(harness.store.shared.wake.load(Ordering::Acquire));
        assert_eq!(
            harness.store.store(&ancestor).unwrap(),
            StoreOutcome::Present
        );
        assert_eq!(harness.store.superseded_count, 1);
    }

    #[test]
    fn subscription_fences_generation_and_epoch() {
        let (mut core, shared) = bare_core();
        core.generation = 5;
        core.session = true;
        core.subscription_epoch.store(2, Ordering::Release);
        let checkpoint = Fixtures::new().published(64, None);
        let update = || {
            Update::Added(super::super::mirror::Entry {
                key: checkpoint.key,
                parent: None,
                bucket: checkpoint.bucket,
                tail: 1,
                row: true,
                location: super::super::mirror::PoolRecord {
                    tier: super::super::mirror::LocationTier::Dram,
                    durable: false,
                    bytes: 4096,
                    unreadable: false,
                    test: super::super::mirror::TestHooks::default(),
                },
            })
        };
        core.incoming(Incoming::Catalog {
            generation: 4,
            epoch: 2,
            updates: vec![update()],
        });
        assert!(shared.mirror.lock().unwrap().updates.is_empty());
        core.incoming(Incoming::Catalog {
            generation: 5,
            epoch: 1,
            updates: vec![update()],
        });
        assert!(shared.mirror.lock().unwrap().updates.is_empty());
        core.incoming(Incoming::Catalog {
            generation: 5,
            epoch: 2,
            updates: vec![update()],
        });
        assert_eq!(shared.mirror.lock().unwrap().updates.len(), 1);

        shared.mirror.lock().unwrap().updates.clear();
        core.incoming(Incoming::Room {
            generation: 4,
            epoch: 2,
            bytes: 1,
        });
        assert_eq!(shared.wake.load(Ordering::Acquire), 0);
        core.incoming(Incoming::Room {
            generation: 5,
            epoch: 1,
            bytes: 1,
        });
        assert_eq!(shared.wake.load(Ordering::Acquire), 0);
        core.incoming(Incoming::Room {
            generation: 5,
            epoch: 2,
            bytes: 1,
        });
        assert_eq!(shared.wake.load(Ordering::Acquire), 1);

        let epoch = core.subscription_epoch.load(Ordering::Acquire);
        core.fail_session("test");
        assert_ne!(core.subscription_epoch.load(Ordering::Acquire), epoch);
    }

    fn bare_core() -> (Core, Arc<Shared>) {
        let geometry = geometries();
        let layout = Layout::new(&geometry).unwrap();
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let shared = Arc::new(Shared::new());
        let (events, _events) = tokio::sync::mpsc::channel(4);
        let (completions, _completions) = std::sync::mpsc::channel();
        let core = Core::new(
            root,
            ModelConfig {
                namespace: namespace_id(&IDENTITY, &fingerprint(), TENANT).unwrap(),
                identity: IDENTITY,
                fingerprint: fingerprint(),
                geometry,
                layout,
                descriptor: proto::Descriptor {
                    identity: IDENTITY.to_vec(),
                    fingerprint: fingerprint().to_vec(),
                    page_tokens: PAGE_TOKENS,
                    ranks: Vec::new(),
                    page_record_bytes: PAGE_BYTES as u64,
                    row_record_bytes: ROW_BYTES as u64,
                    tenant: TENANT.into(),
                },
            },
            SlabPool::staging(&Layout::new(&geometries()).unwrap()).unwrap(),
            TransferHandle::Fake(Box::new(FakeTransfers {
                inner: Arc::new(Mutex::new(FakeInner::new())),
            })),
            completions,
            shared.clone(),
            events,
        );
        (core, shared)
    }

    #[test]
    fn metadata_and_duplicate_leases_are_session_failures() {
        for metadata_first in [true, false] {
            let mut harness = Harness::new();
            if metadata_first {
                harness
                    .fake
                    .state
                    .meta_before_leased
                    .store(true, Ordering::SeqCst);
            } else {
                harness
                    .fake
                    .state
                    .duplicate_leased
                    .store(true, Ordering::SeqCst);
            }
            let root = harness.fixtures.published(130, None);
            harness
                .fake
                .insert_record(&root.key, fake_record(&root, None, 0));
            harness.insert_entry(&root, proto::Tier::Dram);
            harness.fake.restart(GENERATION + 1);
            harness.wait_ready();
            harness.wait_until(|harness| {
                harness.poll();
                harness.store.lookup(&history(130), 0, u64::MAX).is_some()
            });
            let hit = harness.store.lookup(&history(130), 0, u64::MAX).unwrap();
            harness
                .store
                .fetch(
                    hit,
                    InboundTicket {
                        key: root.key,
                        pages: (0..2).map(PageId).collect(),
                        slot: StateSlotId(2),
                    },
                )
                .unwrap();
            assert!(matches!(
                harness.event(),
                StoreEvent::Fetched { result: Err(_), .. }
            ));
            harness
                .fake
                .state
                .meta_before_leased
                .store(false, Ordering::SeqCst);
            harness
                .fake
                .state
                .duplicate_leased
                .store(false, Ordering::SeqCst);
            harness.wait_ready();
        }
    }

    #[test]
    fn invalid_mapped_put_window_is_a_session_failure() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .bad_put_range
            .store(true, Ordering::SeqCst);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        assert_eq!(harness.store.stats().outbound_in_flight, 0);
        harness
            .fake
            .state
            .bad_put_range
            .store(false, Ordering::SeqCst);
        harness.wait_ready();
    }

    #[tokio::test]
    async fn teardown_aborts_the_subscription_and_retires_inbox_work() {
        let (mut core, shared) = bare_core();
        core.session = true;
        core.subscription = Some(tokio::spawn(std::future::pending()));
        core.fail_session("test");
        assert!(core.subscription.is_none());

        let checkpoint = Fixtures::new().published(64, None);
        shared
            .mirror
            .lock()
            .unwrap()
            .push(Update::Added(super::super::mirror::Entry {
                key: checkpoint.key,
                parent: None,
                bucket: checkpoint.bucket,
                tail: 1,
                row: true,
                location: super::super::mirror::PoolRecord {
                    tier: super::super::mirror::LocationTier::Dram,
                    durable: false,
                    bytes: 4096,
                    unreadable: false,
                    test: super::super::mirror::TestHooks::default(),
                },
            }));
        shared.disconnect();
        let inbox = shared.mirror.lock().unwrap();
        assert!(inbox.updates.is_empty());
        assert!(inbox.invalid);
    }

    #[test]
    fn resubscribe_retry_survives_the_throttle() {
        let (mut core, _) = bare_core();
        core.session = true;
        core.next_subscribe = Instant::now() + Duration::from_secs(60);
        core.resubscribe(core.generation);
        assert!(core.pending_resubscribe);
        assert!(!core.take_due_resubscribe(Instant::now()));
        assert!(core.pending_resubscribe);
        assert!(core.take_due_resubscribe(core.next_subscribe));
        assert!(!core.pending_resubscribe);
    }

    #[test]
    fn deferred_tracking_retires_and_rejects_recycled_pages() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .put_refusals
            .lock()
            .unwrap()
            .push_back(proto::Reason::Budget as i32);
        let ancestor = harness.fixtures.published(130, None);
        harness.store.store(&ancestor).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        let tracked = harness.store.publications.get_mut(&ancestor.id).unwrap();
        assert!(tracked.deferred);
        tracked.age = Instant::now() - TRACKED_AGE - Duration::from_secs(1);
        harness.store.retire_tracked();
        assert!(!harness.store.publications.contains_key(&ancestor.id));

        let mut reused = harness.fixtures.published(258, None);
        reused.chain_pages = ancestor
            .chain_pages
            .iter()
            .copied()
            .chain([PageId(999), PageId(998)])
            .collect();
        reused.tail_pages = reused.chain_pages.len();
        harness.store.budget = None;
        assert_eq!(harness.store.store(&reused).unwrap(), StoreOutcome::Queued);
    }

    #[test]
    fn deferred_tracking_is_bounded() {
        let mut harness = Harness::new();
        for _ in 0..TRACKED_PUBLICATIONS + 10 {
            let checkpoint = harness.fixtures.published(64, None);
            harness.store.publications.insert(
                checkpoint.id,
                Tracked {
                    key: checkpoint.key,
                    chain_pages: checkpoint.chain_pages.clone(),
                    parent: None,
                    deferred: true,
                    age: Instant::now(),
                },
            );
        }
        harness.store.retire_tracked();
        assert!(harness.store.publications.len() <= TRACKED_PUBLICATIONS);
    }

    #[test]
    fn deferred_pages_are_not_ancestry_evidence_before_expiration() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .put_refusals
            .lock()
            .unwrap()
            .push_back(proto::Reason::Budget as i32);
        let ancestor = harness.fixtures.published(130, None);
        harness.store.store(&ancestor).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        let mut reused = harness.fixtures.published(64, None);
        reused.chain_pages = vec![ancestor.chain_pages[0]];
        reused.tail_pages = 1;
        harness.store.budget = None;
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        assert!(!harness.store.newer_descendant(&reused));
        assert_eq!(harness.store.store(&reused).unwrap(), StoreOutcome::Queued);
    }

    #[test]
    fn abandoned_superseded_ids_retire() {
        let mut harness = Harness::new();
        let checkpoint = harness.fixtures.published(64, None);
        harness.store.superseded.insert(
            checkpoint.id,
            Instant::now() - TRACKED_AGE - Duration::from_secs(1),
        );
        harness.store.retire_tracked();
        assert!(!harness.store.superseded.contains_key(&checkpoint.id));
    }

    #[test]
    fn a_started_child_keeps_its_late_parent_from_being_superseded() {
        let mut harness = Harness::new();
        let parent = harness.fixtures.published(130, None);
        let child = harness.fixtures.child(258, &parent);
        let grandchild = harness.fixtures.child(386, &child);

        // Start the child and hold its Accept so it stays in flight.
        harness.fake.state.hold_begin.store(true, Ordering::SeqCst);
        assert_eq!(harness.store.store(&child).unwrap(), StoreOutcome::Queued);
        harness.wait_until(|harness| {
            harness
                .fake
                .state
                .observed_puts
                .lock()
                .unwrap()
                .contains(&child.key.consumed_tokens)
        });

        // The parent arrives late, as a retried publication does, while the
        // grandchild queues behind it. The parent must still start: its child
        // is in flight and needs it as the chain link.
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(true, Ordering::Release);
        assert_eq!(harness.store.store(&parent).unwrap(), StoreOutcome::Queued);
        assert_eq!(
            harness.store.store(&grandchild).unwrap(),
            StoreOutcome::Queued
        );
        harness
            .store
            .shared
            .test
            .hold_puts
            .store(false, Ordering::Release);
        harness.fake.state.hold_begin.store(false, Ordering::SeqCst);

        let mut events = Vec::new();
        harness.wait_until(|harness| {
            events.extend(harness.poll());
            events.iter().any(|event| {
                matches!(event, StoreEvent::Stored { checkpoint, .. } if *checkpoint == parent.id)
            })
        });
        assert!(
            !events.iter().any(|event| {
                matches!(event, StoreEvent::Deferred { checkpoint } if *checkpoint == parent.id)
            }),
            "the parent was superseded while its child was in flight"
        );
    }

    #[test]
    fn queued_puts_start_in_order_and_ancestors_first() {
        // FIFO between unrelated chains: newer, shorter publications must not
        // overtake an older long one no matter how many arrive.
        let (mut core, _shared) = bare_core();
        let fixtures = &mut Fixtures::new();
        let long = fixtures.published(384, None);
        core.queued.insert(
            0,
            QueuedPut {
                generation: core.generation,
                checkpoint: long,
            },
        );
        for index in 1..9 {
            let short = fixtures.published(64, None);
            core.queued.insert(
                index,
                QueuedPut {
                    generation: core.generation,
                    checkpoint: short,
                },
            );
        }
        assert_eq!(
            core.next_startable(),
            Some(0),
            "a newer unrelated publication overtook the oldest"
        );

        // A queued ancestor of the oldest record starts before it.
        let (mut core, _shared) = bare_core();
        let fixtures = &mut Fixtures::new();
        let parent = fixtures.published(128, None);
        let child = fixtures.child(192, &parent);
        core.queued.insert(
            0,
            QueuedPut {
                generation: core.generation,
                checkpoint: child,
            },
        );
        core.queued.insert(
            1,
            QueuedPut {
                generation: core.generation,
                checkpoint: parent,
            },
        );
        assert_eq!(
            core.next_startable(),
            Some(1),
            "the queued ancestor must start before its descendant"
        );
    }

    #[test]
    fn deferred_tracking_survives_iterations_but_retires_by_age() {
        let mut harness = Harness::new();
        harness
            .fake
            .state
            .put_refusals
            .lock()
            .unwrap()
            .push_back(proto::Reason::Budget as i32);
        let checkpoint = harness.fixtures.published(130, None);
        harness.store.store(&checkpoint).unwrap();
        assert!(matches!(harness.event(), StoreEvent::Deferred { .. }));
        assert!(
            harness
                .store
                .publications
                .get(&checkpoint.id)
                .unwrap()
                .deferred
        );
        // Worker iterations no longer age a deferred publication out.
        for _ in 0..10_000 {
            harness.store.retire_tracked();
        }
        assert!(harness.store.publications.contains_key(&checkpoint.id));
        // The wall-clock bound still does.
        harness
            .store
            .publications
            .get_mut(&checkpoint.id)
            .unwrap()
            .age = Instant::now() - TRACKED_AGE - Duration::from_secs(1);
        harness.store.retire_tracked();
        assert!(!harness.store.publications.contains_key(&checkpoint.id));
    }

    fn history(tokens: usize) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        history
            .append(&(0..tokens as u32).collect::<Vec<_>>(), &[])
            .unwrap();
        history
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut state = [
            0x6a09e667u32,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];
        let bit_len = (bytes.len() as u64) * 8;
        let mut data = bytes.to_vec();
        data.push(0x80);
        while data.len() % 64 != 56 {
            data.push(0);
        }
        data.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in data.chunks(64) {
            let mut w = [0u32; 64];
            for index in 0..16 {
                w[index] = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
            }
            for index in 16..64 {
                let s0 = w[index - 15].rotate_right(7)
                    ^ w[index - 15].rotate_right(18)
                    ^ (w[index - 15] >> 3);
                let s1 = w[index - 2].rotate_right(17)
                    ^ w[index - 2].rotate_right(19)
                    ^ (w[index - 2] >> 10);
                w[index] = w[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[index - 7])
                    .wrapping_add(s1);
            }
            let mut v = state;
            for index in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[index])
                    .wrapping_add(w[index]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v[7] = v[6];
                v[6] = v[5];
                v[5] = v[4];
                v[4] = v[3].wrapping_add(t1);
                v[3] = v[2];
                v[2] = v[1];
                v[1] = v[0];
                v[0] = t1.wrapping_add(t2);
            }
            for index in 0..8 {
                state[index] = state[index].wrapping_add(v[index]);
            }
        }
        let mut out = [0u8; 32];
        for (index, word) in state.iter().enumerate() {
            out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}
