//! Client of the shared cache manager.
//!
//! Lookup uses a local, sequenced mirror; a single `selene-io` thread owns
//! the gRPC connection, both shared mappings, and the staged transfer
//! pipeline. The model worker never waits for IPC or CUDA: publications and
//! restores are queued with `try_send`, and completions return through the
//! channel `PrefixStore::poll` drains.

mod client;
mod mirror;
mod window;

#[allow(clippy::all)]
pub(super) mod proto {
    tonic::include_proto!("selene.v1");
}

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::sync::atomic::AtomicUsize;

use blake2::{
    Blake2bVar,
    digest::{Update as BlakeUpdate, VariableOutput},
};

use super::{
    CacheGeometry, HostHit, InboundTicket, Layout, PrefixStore, PublishedCheckpoint, StoreEvent,
    StoreOutcome, StoreReason, StoreStats, Tier,
    index::Index,
    slab::SlabPool,
    transfer::{self, Transfers},
};
use crate::{
    Result,
    cache::{CheckpointId, PageId, PrefixHistory, PrefixKey},
};
use mirror::{DELTA_LIMIT, Inbox, LocationTier, PoolRecord, Retired, Update, gauge};

const OPEN_LIMIT: usize = 4;
/// Retained publication metadata bound. Deferred publications the runtime
/// abandons are retired after this wall-clock age so recycled page ids cannot
/// create false ancestry matches; a long prefill can defer a publication for
/// minutes, so the bound cannot be worker iterations.
const TRACKED_PUBLICATIONS: usize = 1024;
const TRACKED_AGE: Duration = Duration::from_secs(120);

/// One queued publication or restore handed to the io thread.
pub(super) enum Command {
    Put {
        generation: u64,
        id: u64,
        checkpoint: PublishedCheckpoint,
    },
    Get {
        generation: u64,
        id: u64,
        hit: HostHit,
        ticket: InboundTicket,
        parents: Vec<Option<PrefixKey>>,
        /// Full tail page count of each chain node, in ascending order.
        tails: Vec<usize>,
        remote: bool,
    },
}

/// Events the io thread reports to the serving thread.
enum Incoming {
    Connected(client::Session),
    Catalog {
        generation: u64,
        epoch: u64,
        updates: Vec<Update>,
    },
    Room {
        generation: u64,
        epoch: u64,
        bytes: u64,
    },
    Disconnected {
        generation: u64,
        epoch: u64,
    },
    Failed(String),
}

/// Bound on queued io-thread events; a full channel backpressures the
/// subscription and operation tasks instead of growing without limit.
const EVENT_CAPACITY: usize = 256;

/// One completion produced by the io thread.
pub(super) struct Completion {
    pub id: u64,
    pub event: StoreEvent,
    pub elapsed: Duration,
    pub tier: Tier,
    pub remote: bool,
    pub tokens: u64,
    pub budget: Option<u64>,
    /// The publication was dropped because a newer checkpoint covers it.
    pub superseded: bool,
}

/// State shared between the serving thread and the io thread.
struct Shared {
    stop: AtomicBool,
    connected: AtomicBool,
    ready: AtomicBool,
    generation: AtomicU64,
    room: AtomicU64,
    wake: AtomicU64,
    resync: AtomicBool,
    mirror: Mutex<Inbox>,
    #[cfg(test)]
    test: TestHooks,
}

#[cfg(test)]
struct TestHooks {
    deadline_ms: AtomicU64,
    hold_puts: AtomicBool,
    pause_in: AtomicBool,
    pause_out: AtomicBool,
    paused: AtomicUsize,
}

#[cfg(test)]
impl TestHooks {
    fn new() -> Self {
        Self {
            deadline_ms: AtomicU64::new(3000),
            hold_puts: AtomicBool::new(false),
            pause_in: AtomicBool::new(false),
            pause_out: AtomicBool::new(false),
            paused: AtomicUsize::new(0),
        }
    }
}

/// Releases a test pause even when an assertion unwinds.
#[cfg(test)]
pub(super) struct PauseGuard {
    shared: Arc<Shared>,
}

#[cfg(test)]
impl PauseGuard {
    pub(super) fn release(mut self) {
        self.clear();
    }

    fn clear(&mut self) {
        self.shared.test.pause_in.store(false, Ordering::Release);
        self.shared.test.pause_out.store(false, Ordering::Release);
    }
}

#[cfg(test)]
impl Drop for PauseGuard {
    fn drop(&mut self) {
        self.clear();
    }
}

impl Shared {
    fn new() -> Self {
        Self {
            stop: AtomicBool::new(false),
            connected: AtomicBool::new(false),
            ready: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            room: AtomicU64::new(0),
            wake: AtomicU64::new(0),
            resync: AtomicBool::new(false),
            mirror: Mutex::new(Inbox::default()),
            #[cfg(test)]
            test: TestHooks::new(),
        }
    }

    #[cfg(not(test))]
    pub(super) fn deadline(&self) -> Duration {
        Duration::from_secs(3)
    }

    #[cfg(test)]
    pub(super) fn deadline(&self) -> Duration {
        Duration::from_millis(self.test.deadline_ms.load(Ordering::Acquire))
    }

    pub(super) fn disconnect(&self) {
        self.connected.store(false, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        self.mirror.lock().unwrap().clear();
    }

    fn shutdown(&self) {
        self.stop.store(true, Ordering::Release);
        self.disconnect();
    }
}

/// A publication the worker accepted and has not yet settled.
struct Tracked {
    key: PrefixKey,
    chain_pages: Vec<PageId>,
    parent: Option<PrefixKey>,
    deferred: bool,
    age: Instant,
}

/// The worker-facing handle. No method on the serving path waits for IPC or CUDA.
pub struct PoolStore {
    commands: tokio::sync::mpsc::Sender<Command>,
    completed: Receiver<Result<Completion>>,
    shared: Arc<Shared>,
    io: Option<JoinHandle<()>>,
    mirror: Index<PoolRecord>,
    pending: HashMap<u64, Option<PrefixKey>>,
    next_id: u64,
    stats: StoreStats,
    gauges: StoreStats,
    rates: [f64; 2],
    remote_rate: f64,
    room: u64,
    closed: bool,
    page_bytes: u64,
    row_bytes: u64,
    failure: Option<crate::Error>,
    budget: Option<u64>,
    publications: HashMap<CheckpointId, Tracked>,
    superseded: HashMap<CheckpointId, Instant>,
    superseded_count: u64,
}

impl PoolStore {
    pub fn new(
        socket: PathBuf,
        tenant: String,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        geometry: Vec<CacheGeometry>,
        pool: SlabPool,
        transfers: Arc<Mutex<Transfers>>,
    ) -> Result<Self> {
        Self::start(
            socket,
            tenant,
            identity,
            fingerprint,
            geometry,
            pool,
            client::TransferHandle::Real(transfers),
        )
    }

    #[cfg(test)]
    fn with_backend(
        socket: PathBuf,
        tenant: String,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        geometry: Vec<CacheGeometry>,
        pool: SlabPool,
        transfers: Box<dyn client::TransferBackend + Send>,
    ) -> Result<Self> {
        Self::start(
            socket,
            tenant,
            identity,
            fingerprint,
            geometry,
            pool,
            client::TransferHandle::Fake(transfers),
        )
    }

    fn start(
        socket: PathBuf,
        tenant: String,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        geometry: Vec<CacheGeometry>,
        pool: SlabPool,
        transfers: client::TransferHandle,
    ) -> Result<Self> {
        if !tenant.is_empty()
            && (tenant.len() > 64
                || !tenant
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')))
        {
            return Err("prefix cache tenant must be 1..=64 bytes of [A-Za-z0-9._-]".into());
        }
        let layout = Layout::new(&geometry)?;
        let namespace = namespace_id(&identity, &fingerprint, &tenant)?;
        let descriptor = descriptor(&identity, &fingerprint, &geometry, &layout, &tenant);
        let shared = Arc::new(Shared::new());
        let (commands, requests) = tokio::sync::mpsc::channel(8);
        let (events, incoming) = tokio::sync::mpsc::channel(EVENT_CAPACITY);
        let (completions, completed) = mpsc::channel();
        let core = client::Core::new(
            socket,
            client::ModelConfig {
                namespace,
                identity,
                fingerprint,
                geometry,
                layout: layout.clone(),
                descriptor,
            },
            pool,
            transfers,
            completions.clone(),
            shared.clone(),
            events,
        );
        let io = thread::Builder::new()
            .name("selene-io".into())
            .spawn(move || client::run(core, requests, incoming))
            .map_err(|error| format!("cannot start the selene io thread: {error}"))?;
        Ok(Self {
            commands,
            completed,
            shared,
            io: Some(io),
            mirror: Index::new(layout.page_tokens),
            pending: HashMap::new(),
            next_id: 1,
            stats: StoreStats {
                outbound_rate: super::INITIAL_MEMORY_RATE,
                inbound_rate: super::INITIAL_MEMORY_RATE,
                ..Default::default()
            },
            gauges: StoreStats::default(),
            rates: [super::INITIAL_MEMORY_RATE, super::INITIAL_STAGED_DISK_RATE],
            remote_rate: super::INITIAL_STAGED_DISK_RATE,
            room: 0,
            closed: false,
            page_bytes: layout.page_record_bytes,
            row_bytes: layout.row_record_bytes,
            failure: None,
            budget: None,
            publications: HashMap::new(),
            superseded: HashMap::new(),
            superseded_count: 0,
        })
    }

    /// Hold inbound transfer jobs so a live manager window owner can be paused.
    #[cfg(test)]
    pub(super) fn pause_inbound_guard(&self) -> PauseGuard {
        self.shared.test.pause_in.store(true, Ordering::Release);
        PauseGuard {
            shared: self.shared.clone(),
        }
    }

    /// Hold outbound transfer jobs so a live manager window owner can be paused.
    #[cfg(test)]
    pub(super) fn pause_outbound_guard(&self) -> PauseGuard {
        self.shared.test.pause_out.store(true, Ordering::Release);
        PauseGuard {
            shared: self.shared.clone(),
        }
    }

    #[cfg(test)]
    pub(super) fn paused_transfers(&self) -> usize {
        self.shared.test.paused.load(Ordering::Acquire)
    }

    fn apply_mirror(&mut self) {
        let shared = self.shared.clone();
        let Ok(mut inbox) = shared.mirror.try_lock() else {
            return;
        };
        let page_tokens = self.mirror.page_tokens();
        if inbox.retired.is_some() {
            return;
        }
        if inbox.invalid {
            self.invalidate_mirror(&mut inbox, page_tokens);
            return;
        }
        for _ in 0..DELTA_LIMIT {
            if !self.shared.ready.load(Ordering::Acquire)
                && !matches!(inbox.updates.front(), Some(Update::Snapshot(..)))
            {
                break;
            }
            let Some(update) = inbox.updates.pop_front() else {
                break;
            };
            let result = match update {
                Update::Snapshot(index, gauges) => {
                    let index = std::mem::replace(&mut self.mirror, index);
                    inbox.retired = Some(Retired {
                        index,
                        updates: std::collections::VecDeque::new(),
                    });
                    self.gauges = gauges;
                    self.shared.ready.store(true, Ordering::Release);
                    self.stats.progress += 1;
                    break;
                }
                Update::Added(entry) => {
                    let location = entry.location.clone();
                    entry
                        .insert(&mut self.mirror)
                        .map(|_| gauge(&mut self.gauges, &location, true))
                }
                Update::Changed(entry) => {
                    if let Some(node) = self.mirror.node_mut(&entry.key) {
                        if node.parent != entry.parent
                            || node.tail_pages != entry.tail
                            || node.bucket != entry.bucket
                        {
                            Err("cache manager changed record geometry".into())
                        } else {
                            gauge(&mut self.gauges, &node.location, false);
                            gauge(&mut self.gauges, &entry.location, true);
                            node.row = entry.row;
                            node.location = entry.location;
                            Ok(())
                        }
                    } else {
                        Err("cache manager changed an absent record".into())
                    }
                }
                Update::Removed(key) => self
                    .mirror
                    .remove_one(&key)
                    .map(|node| gauge(&mut self.gauges, &node.location, false)),
            };
            if result.is_err() {
                self.invalidate_mirror(&mut inbox, page_tokens);
                break;
            }
        }
    }

    fn invalidate_mirror(&mut self, inbox: &mut Inbox, page_tokens: u64) {
        let index = std::mem::replace(&mut self.mirror, Index::new(page_tokens));
        let updates = std::mem::take(&mut inbox.updates);
        inbox.retired = Some(Retired { index, updates });
        inbox.invalid = false;
        self.shared.ready.store(false, Ordering::Release);
        self.gauges = StoreStats::default();
        self.shared.resync.store(true, Ordering::Release);
    }

    fn needed_parent(&self, key: PrefixKey) -> bool {
        self.publications
            .values()
            .any(|publication| publication.parent == Some(key))
    }

    fn retire_tracked(&mut self) {
        self.publications
            .retain(|_, tracked| !tracked.deferred || tracked.age.elapsed() <= TRACKED_AGE);
        self.superseded
            .retain(|_, age| age.elapsed() <= TRACKED_AGE);
        while self.publications.len() > TRACKED_PUBLICATIONS {
            let oldest = self
                .publications
                .iter()
                .filter(|(_, tracked)| tracked.deferred)
                .min_by_key(|(_, tracked)| tracked.age)
                .map(|(id, _)| *id)
                .or_else(|| {
                    self.publications
                        .iter()
                        .min_by_key(|(_, tracked)| tracked.age)
                        .map(|(id, _)| *id)
                });
            let Some(id) = oldest else {
                break;
            };
            self.publications.remove(&id);
        }
        while self.superseded.len() > TRACKED_PUBLICATIONS {
            let Some(oldest) = self
                .superseded
                .iter()
                .min_by_key(|(_, age)| **age)
                .map(|(id, _)| *id)
            else {
                break;
            };
            self.superseded.remove(&oldest);
        }
    }

    fn newer_descendant(&self, checkpoint: &PublishedCheckpoint) -> bool {
        // A deferred publication has no stable page ownership: the runtime may
        // abandon it and its PageIds may already belong to another record.
        self.publications.values().any(|publication| {
            !publication.deferred
                && publication.chain_pages.starts_with(&checkpoint.chain_pages)
                && checkpoint.key.consumed_tokens < publication.key.consumed_tokens
        })
    }
}

impl PrefixStore for PoolStore {
    fn lookup(&mut self, history: &PrefixHistory, resident: u64, before: u64) -> Option<HostHit> {
        self.stats.lookups += 1;
        if !self.shared.connected.load(Ordering::Acquire)
            || !self.shared.ready.load(Ordering::Acquire)
        {
            return None;
        }
        let chain = self
            .mirror
            .lookup_below(history, resident, before, |node| {
                node.ready && !node.location.unreadable
            })?;
        let mut memory = true;
        let mut remote = false;
        for key in &chain.nodes {
            let location = &self.mirror.node(key).unwrap().location;
            memory &= location.tier == LocationTier::Dram;
            remote |= location.tier == LocationTier::Remote;
        }
        let tier = if memory { Tier::Memory } else { Tier::Disk };
        let bytes = chain.pages as u64 * self.page_bytes + self.row_bytes;
        let rate = if memory {
            self.rates[0]
        } else if remote {
            self.remote_rate
        } else {
            self.rates[1]
        };
        self.stats.hits += 1;
        Some(HostHit {
            key: chain.key,
            consumed_tokens: chain.consumed_tokens,
            resident_tokens: chain.resident_tokens,
            pages: chain.pages,
            bytes,
            tier,
            estimated_seconds: bytes as f64 / rate,
            chain,
        })
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        self.retire_tracked();
        if self.superseded.remove(&checkpoint.id).is_some() {
            self.publications.remove(&checkpoint.id);
            return Ok(StoreOutcome::Present);
        }
        if self
            .pending
            .values()
            .any(|key| *key == Some(checkpoint.key))
        {
            return Ok(StoreOutcome::Retry(StoreReason::Inflight));
        }
        // A newer descendant already queued, in flight, or deferred covers this
        // prefix; keep the older record only when a tracked publication needs
        // it as its published parent. This is decided before queue and budget
        // backpressure so those rejections cannot hide supersession.
        if self.newer_descendant(checkpoint) && !self.needed_parent(checkpoint.key) {
            self.publications.remove(&checkpoint.id);
            self.superseded_count += 1;
            return Ok(StoreOutcome::Present);
        }
        if self.closed
            || !self.shared.connected.load(Ordering::Acquire)
            || self.stats.outbound_in_flight >= OPEN_LIMIT
        {
            return Ok(StoreOutcome::Retry(StoreReason::Queue));
        }
        if self.budget == Some(self.shared.wake.load(Ordering::Acquire)) {
            return Ok(StoreOutcome::Retry(StoreReason::Budget));
        }
        let id = self.next_id;
        let next = id
            .checked_add(1)
            .expect("cache manager operation IDs exhausted");
        let command = Command::Put {
            generation: self.shared.generation.load(Ordering::Acquire),
            id,
            checkpoint: checkpoint.clone(),
        };
        match self.commands.try_send(command) {
            Ok(()) => {}
            Err(_) => return Ok(StoreOutcome::Retry(StoreReason::Queue)),
        }
        self.next_id = next;
        self.pending.insert(id, Some(checkpoint.key));
        self.stats.outbound_in_flight += 1;
        self.publications.insert(
            checkpoint.id,
            Tracked {
                key: checkpoint.key,
                chain_pages: checkpoint.chain_pages.clone(),
                parent: checkpoint.parent,
                deferred: false,
                age: Instant::now(),
            },
        );
        Ok(StoreOutcome::Queued)
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        if self.closed
            || !self.shared.connected.load(Ordering::Acquire)
            || self.stats.inbound_in_flight >= OPEN_LIMIT
            || hit.chain.nodes.len() > 65536
            || ticket.pages.len() != hit.pages
        {
            return Err("cache manager fetch queue unavailable".into());
        }
        let parents = hit
            .chain
            .nodes
            .iter()
            .map(|key| {
                self.mirror
                    .node(key)
                    .map(|node| node.parent)
                    .ok_or_else(|| "cache manager mirror changed before fetch".into())
            })
            .collect::<Result<Vec<_>>>()?;
        let tails = hit
            .chain
            .nodes
            .iter()
            .map(|key| {
                self.mirror
                    .node(key)
                    .map(|node| node.tail_pages)
                    .ok_or_else(|| "cache manager mirror changed before fetch".into())
            })
            .collect::<Result<Vec<_>>>()?;
        let remote = hit.chain.nodes.iter().any(|key| {
            self.mirror
                .node(key)
                .is_some_and(|node| node.location.tier == LocationTier::Remote)
        });
        let id = self.next_id;
        let next = id
            .checked_add(1)
            .expect("cache manager operation IDs exhausted");
        let command = Command::Get {
            generation: self.shared.generation.load(Ordering::Acquire),
            id,
            hit,
            ticket,
            parents,
            tails,
            remote,
        };
        self.commands
            .try_send(command)
            .map_err(|_| "cache manager fetch queue full")?;
        self.next_id = next;
        self.pending.insert(id, None);
        self.stats.inbound_in_flight += 1;
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        self.retire_tracked();
        self.apply_mirror();
        let room = self.shared.room.load(Ordering::Acquire);
        self.stats.progress += room.wrapping_sub(self.room);
        self.room = room;
        while let Ok(completion) = self.completed.try_recv() {
            let completion = match completion {
                Ok(completion) => completion,
                Err(error) => {
                    self.failure = Some(error.clone());
                    return Err(error);
                }
            };
            let Some(outbound) = self.pending.remove(&completion.id) else {
                continue;
            };
            // A stale Budget completion must not consume an earlier Room wakeup.
            self.budget = self.budget.max(completion.budget);
            if completion.budget.is_none()
                || self.budget != Some(self.shared.wake.load(Ordering::Acquire))
            {
                self.stats.progress += 1;
            }
            if outbound.is_some() {
                self.stats.outbound_in_flight -= 1;
            } else {
                self.stats.inbound_in_flight -= 1;
            }
            match &completion.event {
                StoreEvent::Stored {
                    checkpoint,
                    result: Ok(bytes),
                    ..
                } => {
                    self.stats.stored += 1;
                    self.stats.bytes_out += bytes;
                    observe_rate(&mut self.stats.outbound_rate, *bytes, completion.elapsed);
                    self.publications.remove(checkpoint);
                }
                StoreEvent::Stored {
                    checkpoint,
                    result: Err(_),
                    ..
                } => {
                    self.stats.store_failed += 1;
                    self.publications.remove(checkpoint);
                }
                StoreEvent::Fetched {
                    result: Ok(bytes), ..
                } => {
                    self.stats.fetched += 1;
                    self.stats.bytes_in += bytes;
                    self.stats.reused_tokens += completion.tokens;
                    let rate = if completion.tier == Tier::Memory {
                        &mut self.rates[0]
                    } else if completion.remote {
                        &mut self.remote_rate
                    } else {
                        &mut self.rates[1]
                    };
                    observe_rate(rate, *bytes, completion.elapsed);
                    self.stats.inbound_rate = *rate;
                }
                StoreEvent::Fetched { result: Err(_), .. } => self.stats.fetch_failed += 1,
                StoreEvent::Deferred { checkpoint } => {
                    if completion.superseded {
                        self.publications.remove(checkpoint);
                        self.superseded.insert(*checkpoint, Instant::now());
                        self.superseded_count += 1;
                    } else if let Some(tracked) = self.publications.get_mut(checkpoint) {
                        tracked.deferred = true;
                        tracked.age = Instant::now();
                    }
                }
            }
            events.push(completion.event);
        }
        if self.io.as_ref().is_some_and(JoinHandle::is_finished) && !self.closed {
            return Err("cache manager io worker stopped".into());
        }
        Ok(())
    }

    fn stats(&self) -> StoreStats {
        StoreStats {
            memory_records: self.gauges.memory_records,
            memory_bytes: self.gauges.memory_bytes,
            disk_records: self.gauges.disk_records,
            disk_bytes: self.gauges.disk_bytes,
            ..self.stats
        }
    }

    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.shared.shutdown();
        if self.superseded_count != 0 {
            eprintln!(
                "flare prefix cache pool: {} superseded publications",
                self.superseded_count
            );
        }
        let deadline = Instant::now() + transfer::CLOSE_TIMEOUT;
        while self.io.as_ref().is_some_and(|thread| !thread.is_finished()) {
            if Instant::now() >= deadline {
                return Err("cache manager client did not drain before close timeout".into());
            }
            thread::sleep(Duration::from_millis(1));
        }
        if self.io.take().is_some_and(|thread| thread.join().is_err()) {
            return Err("cache manager io worker panicked".into());
        }
        while let Ok(completion) = self.completed.try_recv() {
            if let Err(error) = completion {
                self.failure = Some(error);
            }
        }
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        Ok(())
    }
}

impl Drop for PoolStore {
    fn drop(&mut self) {
        self.shared.shutdown();
        if let Some(thread) = self.io.take() {
            let _ = thread.join();
        }
    }
}

fn observe_rate(rate: &mut f64, bytes: u64, elapsed: Duration) {
    if bytes != 0 && !elapsed.is_zero() {
        *rate = 0.8 * *rate + 0.2 * bytes as f64 / elapsed.as_secs_f64();
    }
}

/// Namespace id: BLAKE2b-256 over the versioned domain, identity, layout
/// fingerprint, and length-prefixed tenant.
pub(super) fn namespace_id(
    identity: &[u8; 32],
    fingerprint: &[u8; 16],
    tenant: &str,
) -> Result<[u8; 32]> {
    let len = u8::try_from(tenant.len()).map_err(|_| "prefix cache tenant exceeds 255 bytes")?;
    let mut hash =
        Blake2bVar::new(32).map_err(|_| "cannot initialize the selene namespace hash")?;
    hash.update(b"selene-ns-v1");
    hash.update(identity);
    hash.update(fingerprint);
    hash.update(&[len]);
    hash.update(tenant.as_bytes());
    let mut output = [0u8; 32];
    hash.finalize_variable(&mut output)
        .map_err(|_| "cannot finalize the selene namespace hash")?;
    Ok(output)
}

fn descriptor(
    identity: &[u8; 32],
    fingerprint: &[u8; 16],
    geometry: &[CacheGeometry],
    layout: &Layout,
    tenant: &str,
) -> proto::Descriptor {
    proto::Descriptor {
        identity: identity.to_vec(),
        fingerprint: fingerprint.to_vec(),
        page_tokens: layout.page_tokens,
        ranks: geometry
            .iter()
            .map(|rank| proto::RankPlanes {
                plane_bytes: rank.planes.iter().map(|plane| plane.unit_bytes).collect(),
                row_bytes: rank.row.unit_bytes,
            })
            .collect(),
        page_record_bytes: layout.page_record_bytes,
        row_record_bytes: layout.row_record_bytes,
        tenant: tenant.to_owned(),
    }
}

#[cfg(all(test, feature = "runtime-package"))]
#[path = "../pool_cuda_tests.rs"]
mod cuda_tests;
