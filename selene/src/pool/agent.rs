//! The rank agent: the part of a multi-process instance that runs in each peer rank.
//!
//! Rank 0 of such an instance decides everything. A peer rank runs a [`RankAgent`], which
//! decides nothing: it is told, over the rank link, which of this rank's GPU pages to copy
//! to or from which window of the manager, does that for its own slice of each window
//! (shared memory in place, or its own data channel), and says when it has drained.
//!
//! Every transfer is bound to the manager incarnation and attachment generation this agent
//! reported to rank 0, and keeps the mapping it was opened against until it ends, so a
//! delayed command can never resolve against a newer mapping. A transfer ends with exactly
//! one `Done`, sent after exact coverage and after every accepted copy drained (and, for a
//! channel, the socket closed). What this agent may still be touching is its own
//! obligation to the manager: it attests it with a `RankFence` of its own.

use rustls::pki_types::ServerName;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio_rustls::TlsConnector;

use tokio::sync::{Notify, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    CLOSE_TIMEOUT, Direction, Error, HostRegistrar, Job, JobId, PageId, PrefixKey, Result,
    SharedCopier, StateSlotId, slab::WINDOW_PAGES,
};

use super::conn::{Attempt, BUSY_TICK, FIRST_BACKOFF, IDLE_TICK, MAX_BACKOFF};
use super::{
    Checksums,
    client::PoolAddress,
    codec, conn,
    engine::{
        self,
        coverage::Coverage,
        hostfill::{FillJob, FillOwner, HostFiller},
        requester::ExpectedRecord,
    },
    identity::Scope,
    net::{self, Channel, Dial, Event, Id, Staging},
    rank::{self as rank, Incoming, Msg, RankLink},
    shared::PoolStatus,
    shm::Regions,
    slice::{SliceLayout, SliceSums},
    startup::{Approve, RegisteredRegion, Startup},
    wire::{self, rank_client_message as rc, rank_server_message as rs},
};

const ATTACH_TIMEOUT: Duration = Duration::from_secs(5);
/// This rank's record files, for the host-tier exports and puts of a multi-process instance
/// whose local tier is sharded: every rank keeps the slices of every record in its own files,
/// and rank 0's index holds them for all ranks.
pub trait RankRecords: Send + 'static {
    /// This rank's slices of pages `first_page..first_page + pages` of the host record
    /// `key` (record-relative pages), then its slice of the row when `row`,
    /// packed into `dst`, with the persisted slice checksums (never hashes of what was just
    /// read). Blocks on the device: it runs on the agent's one reader thread, so at most one
    /// read is in flight.
    fn read(
        &mut self,
        key: PrefixKey,
        first_page: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums>;

    /// The same for a put: this rank's slices of pages `first_page..` of the chain being
    /// published, `key` naming the record the put commits (`first_page` counts from the
    /// chain's start, as a single process's `PutBytes::fill` does).
    fn fill(
        &mut self,
        key: PrefixKey,
        first_page: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums>;
}

/// What a peer rank's agent is built from. The pool address and CA are this
/// process's own flags; `scope` is the instance's, identical to rank 0's.
pub struct AgentConfig {
    pub address: PoolAddress,
    pub ca: Option<PathBuf>,
    pub scope: Scope,
    pub rank: usize,
    pub world: usize,
    /// The link to rank 0.
    pub link: RankLink,
    /// This rank's page copier, built for its own one-rank geometry.
    pub copier: SharedCopier,
    pub registrar: Arc<dyn HostRegistrar + Send + Sync>,
    /// The device context made current when registering memory.
    pub registration_context: usize,
    pub gpu_uuid: [u8; 16],
    /// This rank's record files, when its local tier is sharded.
    pub records: Option<Box<dyn RankRecords>>,
    /// Called once at startup with everything this agent is about to register (the manager's
    /// shared files and/or its own data-channel staging), after the manager's first Hello and
    /// before any registration. `Err(reason)` registers nothing: the agent is disabled with one
    /// warning, and rank 0 turns the pool off for every rank. `None` approves.
    pub approve: Option<Approve>,
}

/// A running agent.
pub struct RankAgent {
    stop: Arc<AtomicBool>,
    wake: Arc<Notify>,
    status: Arc<Mutex<PoolStatus>>,
    thread: Option<JoinHandle<Result<()>>>,
    registered: Vec<RegisteredRegion>,
}

impl RankAgent {
    /// Start the agent's thread and return at once.
    pub fn start(config: AgentConfig) -> Result<RankAgent> {
        Self::start_with(config, super::net::Roots::System)
    }

    pub(crate) fn start_with(config: AgentConfig, roots: net::Roots) -> Result<RankAgent> {
        let layout = Arc::new(SliceLayout::new(&config.scope.geometry)?);
        if config.world != layout.world() || config.rank == 0 || config.rank >= config.world {
            return Err(Error::new("the rank does not fit the scope's rank count"));
        }
        {
            let copier = config.copier.lock().unwrap();
            let mine = layout.rank(config.rank);
            if copier.layout().page_record_bytes != mine.page_len
                || copier.layout().row_record_bytes != mine.row_len
            {
                return Err(Error::new(
                    "the pool scope does not match the copier's layout for this rank",
                ));
            }
        }
        let wake = Arc::new(Notify::new());
        let stop = Arc::new(AtomicBool::new(false));
        let startup = Arc::new(Startup::new());
        let status = Arc::new(Mutex::new(PoolStatus::Connecting));
        let end = config.link;
        let agent = Agent {
            address: config.address,
            ca: config.ca,
            roots,
            fingerprint: config.scope.layout_fingerprint,
            layout,
            rank: config.rank,
            world: config.world,
            gpu_uuid: config.gpu_uuid,
            context: config.registration_context,
            copier: config.copier,
            registrar: config.registrar,
            startup: startup.clone(),
            approve: config.approve,
            shared: None,
            wake: wake.clone(),
            stop: stop.clone(),
            status: status.clone(),
            end,
            agent_nonce: super::identity::process_nonce()?,
            bound: None,
            att: None,
            staging: None,
            net_events: net::Events::new(wake.clone()),
            dial: None,
            xfers: HashMap::new(),
            high: 0,
            jobs: HashMap::new(),
            pending: VecDeque::new(),
            next_job: 0,
            owed: Vec::new(),
            quarantined: false,
            closing: false,
            records: config.records.map(|records| Arc::new(Mutex::new(records))),
            worker: None,
        };
        let thread = std::thread::Builder::new()
            .name("selene-rank-agent".into())
            .spawn(move || run(agent))
            .map_err(|e| Error::caused_by("cannot start the rank agent thread", e))?;
        let registered = match startup.wait(super::client::STARTUP_WAIT) {
            Ok(registered) => registered,
            Err(timed_out) => {
                if timed_out {
                    let reason = format!(
                        "the pool manager did not answer within {:?} of startup; this rank \
                         will not attach to the pool later (start the manager before the instance)",
                        super::client::STARTUP_WAIT
                    );
                    eprintln!("warning: rank {}: {reason}", config.rank);
                    *status.lock().unwrap() = PoolStatus::Disabled(reason);
                }
                Vec::new()
            }
        };
        Ok(RankAgent {
            stop,
            wake,
            status,
            thread: Some(thread),
            registered,
        })
    }

    pub fn status(&self) -> PoolStatus {
        self.status.lock().unwrap().clone()
    }

    /// The ranges registered at startup. Empty when the agent did not attach.
    pub fn registered_regions(&self) -> &[RegisteredRegion] {
        &self.registered
    }

    /// Stop, drain every transfer, attest it to the manager and end. `Err` means copies
    /// may still be running: the caller must keep the memory they touch.
    pub fn close(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::SeqCst);
        self.wake.notify_one();
        match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| Error::new("the rank agent thread panicked"))?,
            None => Ok(()),
        }
    }
}

impl Drop for RankAgent {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// ----------------------------------------------------------------------------------

/// The manager attachment of the running session.
struct Att {
    incarnation: u64,
    generation: u64,
    mode: wire::RankMode,
    regions: Option<Arc<Regions>>,
    tx: mpsc::Sender<wire::RankClientMessage>,
}

struct Bound {
    rank0_nonce: Vec<u8>,
    secret: Vec<u8>,
    instance_id: Vec<u8>,
}

/// One window of a transfer.
struct Win {
    jobs_left: u32,
    spec: wire::WindowSpec,
    /// The staging slot a channel window lives in.
    slot: Option<u32>,
    /// The persisted checksums of a host window's slices, once its read is in.
    persisted: Option<Checksums>,
}

struct Xfer {
    /// The bytes are this rank's record files', not GPU pages.
    host: bool,
    /// The records the transfer covers, for the reader.
    records: Vec<wire::Record>,
    /// A put's first page in its chain.
    chain_first: u64,
    out: bool,
    kind: i32,
    fence_id: u64,
    coverage: Coverage,
    pages: Vec<PageId>,
    slot: Option<StateSlotId>,
    shared: bool,
    channel: Option<Channel>,
    channel_open: bool,
    ended: bool,
    windows: BTreeMap<u32, Win>,
    jobs_outstanding: u32,
    failed: Option<String>,
    cancelled: bool,
    page_bytes: u64,
    row_bytes: u64,
}

struct Agent {
    address: PoolAddress,
    ca: Option<PathBuf>,
    roots: net::Roots,
    fingerprint: [u8; 16],
    layout: Arc<SliceLayout>,
    rank: usize,
    world: usize,
    gpu_uuid: [u8; 16],
    context: usize,
    copier: SharedCopier,
    registrar: Arc<dyn HostRegistrar + Send + Sync>,
    startup: Arc<Startup>,
    approve: Option<Approve>,
    /// The manager's shared files as they were when mapped and registered at startup.
    shared: Option<(wire::SharedMemory, Arc<Regions>)>,
    wake: Arc<Notify>,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<PoolStatus>>,
    end: RankLink,
    agent_nonce: [u8; 16],
    bound: Option<Bound>,
    att: Option<Att>,
    staging: Option<Arc<Staging>>,
    net_events: Arc<net::Events>,
    dial: Option<Arc<dyn Dial>>,
    xfers: HashMap<u64, Xfer>,
    /// Transfers at or below this have been opened; a lower id is a replay.
    high: u64,
    jobs: HashMap<JobId, (u64, u32)>,
    pending: VecDeque<(Job, (u64, u32))>,
    next_job: u64,
    /// Fences for the manager not yet delivered: (kind, id, drained).
    owed: Vec<(i32, u64, bool)>,
    quarantined: bool,
    closing: bool,
    /// The rank's record files, retained by queued host reads.
    records: Option<Arc<Mutex<Box<dyn RankRecords>>>>,
    worker: Option<HostFiller>,
}

fn run(mut agent: Agent) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::caused_by("cannot start the rank agent runtime", e))?;
    runtime.block_on(async {
        let result = agent.serve().await;
        agent.finish().await;
        let status = agent.status.clone();
        let mut status = status.lock().unwrap();
        if !matches!(*status, PoolStatus::Disabled(_)) {
            *status = PoolStatus::Disabled("the agent stopped".into());
        }
        result
    })
}

impl Agent {
    fn set_status(&mut self, status: PoolStatus) {
        if let PoolStatus::Disabled(reason) = &status {
            // All ranks use the pool or none do: rank 0 hears it.
            self.end.send_rank(Msg::Disabled {
                reason: reason.clone(),
            });
        }
        *self.status.lock().unwrap() = status;
    }

    fn stopping(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    async fn serve(&mut self) -> Result<()> {
        if !self.prepare().await {
            return Ok(());
        }
        if !self.greet().await {
            return Ok(());
        }
        let mut backoff = FIRST_BACKOFF;
        let mut warned = false;
        loop {
            if self.stopping() || self.end.failed() {
                return Ok(());
            }
            let started = Instant::now();
            match self.attempt().await {
                Attempt::Disabled(why) => {
                    eprintln!("warning: rank {}: {why}", self.rank);
                    self.set_status(PoolStatus::Disabled(why));
                    return Ok(());
                }
                Attempt::Retry(why) => {
                    if !warned {
                        warned = true;
                        eprintln!(
                            "warning: rank {}: cannot attach to the pool manager ({why}); retrying the existing attachment",
                            self.rank
                        );
                    }
                }
                Attempt::Ended => {
                    warned = false;
                    if started.elapsed() > MAX_BACKOFF {
                        backoff = FIRST_BACKOFF;
                    }
                }
            }
            // Whatever the attachment was doing is cancelled and drained before the next one.
            self.cancel_all();
            if !self.drain().await {
                return Err(Error::new("the rank agent could not drain its transfers"));
            }
            if self.att.take().is_some() {
                self.end.send_rank(Msg::Detached {
                    reason: "the manager attachment ended".into(),
                });
            }
            if self.stopping() || self.end.failed() {
                return Ok(());
            }
            self.set_status(PoolStatus::Local);
            let until = Instant::now() + backoff;
            while Instant::now() < until && !self.stopping() && !self.end.failed() {
                self.events();
                let _ = tokio::time::timeout(IDLE_TICK, self.wake.notified()).await;
            }
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }

    /// The last act: the link says goodbye.
    async fn finish(&mut self) {
        self.end.send_rank(Msg::Close);
        tokio::time::sleep(Duration::from_millis(20)).await;
        let _ = self.end.drain_outgoing(CLOSE_TIMEOUT);
        self.end.fail(Error::new("rank agent closed"));
    }

    /// Stop work: cancel, drain, attest to the manager what was touched, and, if
    /// `goodbye`, say whether everything stopped. The stream must still be open.
    async fn wind_down(&mut self, goodbye: bool) {
        self.cancel_all();
        let drained = self.drain().await;
        self.deliver_owed();
        if goodbye && let Some(att) = &self.att {
            let quiesced = drained && self.xfers.is_empty() && !self.quarantined;
            let _ = att
                .tx
                .send(wire::RankClientMessage {
                    m: Some(rc::M::Goodbye(wire::RankGoodbye { quiesced })),
                })
                .await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// The link handshake with rank 0: our hello, theirs, then the secret.
    async fn greet(&mut self) -> bool {
        self.end.send_rank(Msg::AgentHello {
            rank: self.rank as u32,
            world: self.world as u32,
            agent_nonce: self.agent_nonce.to_vec(),
            layout: self.fingerprint.to_vec(),
            gpu_uuid: self.gpu_uuid.to_vec(),
        });
        let mut hello = None;
        while !self.stopping() {
            let mut incoming = Vec::new();
            self.end.poll_rank(&mut incoming);
            for item in incoming {
                match item {
                    Incoming::Closed(why) => {
                        self.set_status(PoolStatus::Disabled(format!(
                            "the rank link failed: {why}"
                        )));
                        return false;
                    }
                    Incoming::Message(Msg::Hello {
                        rank,
                        world,
                        rank0_nonce,
                        layout,
                    }) if hello.is_none() => hello = Some((rank, world, rank0_nonce, layout)),
                    Incoming::Message(Msg::Bind {
                        rank_secret,
                        instance_id,
                    }) => {
                        let Some((rank, world, rank0_nonce, layout)) = hello.take() else {
                            self.set_status(PoolStatus::Disabled("bind before hello".into()));
                            return false;
                        };
                        let fine = rank as usize == self.rank
                            && world as usize == self.world
                            && layout == self.fingerprint
                            && rank_secret.len() == 32;
                        if !fine {
                            self.set_status(PoolStatus::Disabled(
                                "rank 0 disagrees about the instance".into(),
                            ));
                            return false;
                        }
                        self.bound = Some(Bound {
                            rank0_nonce,
                            secret: rank_secret,
                            instance_id,
                        });
                        return true;
                    }
                    Incoming::Message(Msg::Close) => {
                        *self.status.lock().unwrap() =
                            PoolStatus::Disabled("rank 0 is not using the pool".into());
                        return false;
                    }
                    Incoming::Message(_) => {}
                }
            }
            let _ = tokio::time::timeout(IDLE_TICK, self.wake.notified()).await;
        }
        false
    }

    fn link(&self) -> conn::Link {
        conn::Link {
            address: self.address.clone(),
            ca: self.ca.clone(),
            roots: self.roots.clone(),
            wake: self.wake.clone(),
            stop: self.stop.clone(),
            startup: self.startup.clone(),
            approve: self.approve.clone(),
        }
    }

    /// Startup: the one time this agent registers memory. On the first Hello it maps and
    /// registers the manager's shared files, when it offers them, and its own data-channel
    /// staging, when the manager has a data listener; which of the two a rank then uses is
    /// the manager's decision at attach. The registration window closes behind it. `false`
    /// when the agent is not going to attach at all.
    async fn prepare(&mut self) -> bool {
        let link = self.link();
        let mut backoff = FIRST_BACKOFF;
        loop {
            if self.stopping() || self.end.failed() {
                return false;
            }
            if self.startup.is_closed() {
                self.set_status(PoolStatus::Disabled(
                    "the pool manager did not answer at startup".into(),
                ));
                return false;
            }
            match conn::reach(&link, ATTACH_TIMEOUT).await {
                Ok((_client, hello, _tls)) => {
                    if !self.startup.begin() {
                        return false;
                    }
                    let staging = (!hello.data_transports.is_empty())
                        .then(|| (self.staging_slots(&hello), self.slot_bytes(&hello)));
                    let prepared = conn::register_memory(
                        hello.shared_memory.as_ref(),
                        staging,
                        &*self.registrar,
                        self.context,
                        self.approve.as_ref(),
                    );
                    let (regions, staging, registered) = match prepared {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            self.registrar = Arc::new(super::startup::ClosedRegistrar);
                            self.startup.done(Vec::new());
                            eprintln!("warning: rank {}: {error}", self.rank);
                            self.set_status(PoolStatus::Disabled(error.to_string()));
                            return false;
                        }
                    };
                    self.shared = regions
                        .map(|regions| (hello.shared_memory.clone().expect("mapped"), regions));
                    self.staging = staging;
                    self.registrar = Arc::new(super::startup::ClosedRegistrar);
                    self.startup.done(registered);
                    return true;
                }
                Err(Attempt::Disabled(why)) => {
                    self.startup.close();
                    eprintln!("warning: rank {}: {why}", self.rank);
                    self.set_status(PoolStatus::Disabled(why));
                    return false;
                }
                Err(_) => {
                    let until = Instant::now() + backoff;
                    while Instant::now() < until && !self.stopping() {
                        let _ = tokio::time::timeout(IDLE_TICK, self.wake.notified()).await;
                    }
                    backoff = (backoff * 2).min(MAX_BACKOFF);
                }
            }
        }
    }

    /// Reach the manager, attach, and serve until the attachment ends.
    async fn attempt(&mut self) -> Attempt {
        let link = self.link();
        let (mut client, hello, tls) = match conn::reach(&link, ATTACH_TIMEOUT).await {
            Ok(reached) => reached,
            Err(attempt) => return attempt,
        };
        if let Some((registered, _)) = &self.shared
            && hello.shared_memory.as_ref() != Some(registered)
        {
            return Attempt::Disabled(conn::regions_changed(
                registered,
                hello.shared_memory.as_ref(),
            ));
        }
        let Some(bound) = self.bound.as_ref() else {
            return Attempt::Disabled("not bound to rank 0".into());
        };
        let (tx, rx) = mpsc::channel::<wire::RankClientMessage>(256);
        let attach = wire::RankAttach {
            manager_incarnation: hello.manager_incarnation,
            instance_id: bound.instance_id.clone(),
            rank0_nonce: bound.rank0_nonce.clone(),
            rank: self.rank as u32,
            agent_nonce: self.agent_nonce.to_vec(),
            rank_secret: bound.secret.clone(),
        };
        let _ = tx
            .send(wire::RankClientMessage {
                m: Some(rc::M::Attach(attach)),
            })
            .await;
        let response =
            tokio::time::timeout(ATTACH_TIMEOUT, client.rank(ReceiverStream::new(rx))).await;
        let mut stream = match response {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(status)) => return Attempt::Retry(format!("attach failed: {status}")),
            Err(_) => return Attempt::Retry("attaching timed out".into()),
        };
        let attached = match tokio::time::timeout(ATTACH_TIMEOUT, stream.message()).await {
            Ok(Ok(Some(wire::RankServerMessage {
                m: Some(rs::M::Attached(attached)),
            }))) => attached,
            Ok(Ok(other)) => return Attempt::Retry(format!("unexpected answer {other:?}")),
            Ok(Err(status)) => return Attempt::Retry(format!("attach failed: {status}")),
            Err(_) => return Attempt::Retry("the manager did not answer the attach".into()),
        };
        use wire::RankAttachStatus as S;
        match S::try_from(attached.status) {
            Ok(S::RankOk) => {}
            Ok(S::RankNoSession | S::RankStaleManager) => {
                return Attempt::Retry("rank 0 has no session yet".into());
            }
            Ok(S::RankNoDataListener) => {
                return Attempt::Disabled(
                    "the manager has no data listener this rank can reach".into(),
                );
            }
            other => {
                return Attempt::Disabled(format!("the manager refused the attach ({other:?})"));
            }
        }
        let mode = wire::RankMode::try_from(attached.mode).unwrap_or(wire::RankMode::Unspecified);
        let regions = match (mode, &self.shared) {
            (wire::RankMode::RankShared, Some((_, regions))) => Some(regions.clone()),
            (wire::RankMode::RankShared, None) => {
                return Attempt::Disabled(
                    "the manager gave shared mode but no shared memory was registered at startup"
                        .into(),
                );
            }
            _ => None,
        };
        if mode == wire::RankMode::RankChannel
            && let Err(error) = self.make_dial(tls, &hello)
        {
            return Attempt::Disabled(format!("cannot use the pool's data channel: {error}"));
        }
        self.att = Some(Att {
            incarnation: hello.manager_incarnation,
            generation: attached.generation,
            mode,
            regions,
            tx,
        });
        self.set_status(PoolStatus::Ready);
        self.deliver_owed();
        self.end.send_rank(Msg::Attached {
            manager_incarnation: hello.manager_incarnation,
            generation: attached.generation,
            mode: mode as i32,
        });
        self.attached_loop(&mut stream).await;
        Attempt::Ended
    }

    /// Slots for claims, puts and exports, each with the stated window credits.
    fn staging_slots(&self, hello: &wire::HelloReply) -> u32 {
        let limits = conn::checked_limits(hello.limits);
        super::limits::staging_slots(
            &limits,
            limits.max_claims_in_flight,
            limits.max_export_slots,
        )
    }

    /// The size of one slot of the data-channel staging.
    fn slot_bytes(&self, hello: &wire::HelloReply) -> u64 {
        let limits = conn::checked_limits(hello.limits);
        let mine = self.layout.rank(self.rank);
        limits.max_window_bytes.max(mine.page_len).max(mine.row_len)
    }

    fn make_dial(
        &mut self,
        tls: Option<(TlsConnector, ServerName<'static>)>,
        hello: &wire::HelloReply,
    ) -> Result<()> {
        let limits = conn::checked_limits(hello.limits);
        let address = match &self.address {
            PoolAddress::Tcp(address) | PoolAddress::TcpPlaintext(address) => address.as_str(),
            PoolAddress::Unix(_) => "127.0.0.1",
        };
        let Some(staging) = self.staging.as_ref() else {
            return Err(Error::new(
                "no data-channel staging was registered at startup",
            ));
        };
        if staging.slot_bytes() < limits.max_window_bytes {
            return Err(Error::new(
                "the manager now allows larger windows than the staging registered at startup",
            ));
        }
        self.dial = Some(conn::dial(
            tls,
            address,
            staging.clone(),
            self.net_events.clone(),
            Arc::default(),
            hello
                .data_transports
                .contains(&(wire::DataTransport::Rdma as i32)),
        ));
        Ok(())
    }

    /// Serve the link, the channel tasks and the copier while the attachment lasts.
    async fn attached_loop(&mut self, stream: &mut tonic::Streaming<wire::RankServerMessage>) {
        loop {
            self.events();
            if self.stopping() || self.end.failed() {
                self.wind_down(true).await;
                return;
            }
            let tick = if self.busy() { BUSY_TICK } else { IDLE_TICK };
            tokio::select! {
                message = stream.message() => match message {
                    Ok(Some(wire::RankServerMessage { m: Some(rs::M::Closing(closing)) })) => {
                        eprintln!(
                            "warning: rank {}: the manager closed the attachment: {}",
                            self.rank, closing.detail
                        );
                        self.wind_down(true).await;
                        return;
                    }
                    Ok(Some(_)) => {}
                    Ok(None) | Err(_) => return,
                },
                _ = self.wake.notified() => {}
                _ = tokio::time::sleep(tick) => {}
            }
        }
    }

    fn busy(&self) -> bool {
        !self.jobs.is_empty() || !self.pending.is_empty() || !self.xfers.is_empty()
    }

    /// One non-blocking pass: the link, the channel tasks, the copier.
    fn events(&mut self) {
        let mut incoming = Vec::new();
        self.end.poll_rank(&mut incoming);
        for item in incoming {
            match item {
                Incoming::Closed(why) => {
                    eprintln!("warning: rank {}: the rank link failed ({why})", self.rank);
                    self.closing = true;
                    self.cancel_all();
                }
                Incoming::Message(message) => self.on_link(message),
            }
        }
        for event in self.net_events.take() {
            self.on_net(event);
        }
        self.poll_copier();
        self.poll_records();
        self.submit_pending();
        let ids: Vec<u64> = self.xfers.keys().copied().collect();
        for xfer in ids {
            self.drive(xfer);
        }
    }

    // ---- link ---------------------------------------------------------------------

    fn on_link(&mut self, message: Msg) {
        match message {
            Msg::Open(open) => self.on_open(open),
            Msg::Window(window) => self.on_window(window),
            Msg::Cancel { xfer } => self.cancel(xfer),
            Msg::Close => {
                self.closing = true;
                self.cancel_all();
            }
            _ => {}
        }
    }

    fn reject(&mut self, xfer: u64, why: &str) {
        self.end.send_rank(Msg::Done {
            xfer,
            ok: false,
            error: why.into(),
            quarantined: false,
        });
    }

    fn on_open(&mut self, open: rank::Open) {
        let xfer = open.xfer;
        let bound = self.att.as_ref().is_some_and(|att| {
            att.incarnation == open.manager_incarnation && att.generation == open.generation
        });
        if !bound {
            return self.reject(xfer, "the transfer is bound to another attachment");
        }
        if xfer <= self.high || self.xfers.contains_key(&xfer) {
            return self.reject(xfer, "the transfer id was already used");
        }
        if self.closing || self.quarantined {
            return self.reject(xfer, "the agent is closing");
        }
        self.high = xfer;
        let (shared, out) = (open.channel.is_none(), open.out);
        let att = self.att.as_ref().expect("bound");
        if shared != (att.mode == wire::RankMode::RankShared) {
            return self.reject(xfer, "the transfer's mode is not this attachment's");
        }
        let mut expected = Vec::new();
        for record in &open.records {
            let Some(key) = record.key.as_ref().and_then(codec::key_from) else {
                return self.reject(xfer, "a record names no key");
            };
            expected.push(ExpectedRecord {
                key,
                first_page: record.first_page,
                pages: record.pages,
                row: record.row,
            });
        }
        let total: u64 = expected.iter().map(|r| u64::from(r.pages)).sum();
        if open.host {
            if !out || self.records.is_none() {
                return self.reject(xfer, "this agent has no record files to serve");
            }
            if self.worker.is_none() {
                match HostFiller::start(self.wake.clone()) {
                    Ok(worker) => self.worker = Some(worker),
                    Err(error) => return self.reject(xfer, &error.to_string()),
                }
            }
            if expected.is_empty() || !open.pages.is_empty() {
                return self.reject(xfer, "a host transfer names records, not pages");
            }
        } else if expected.is_empty() || total != open.pages.len() as u64 {
            return self.reject(xfer, "the transfer's pages do not match its records");
        }
        if open.page_bytes != self.layout.page_record_bytes()
            || open.row_bytes != self.layout.row_record_bytes()
        {
            return self.reject(xfer, "the transfer's record sizes are not this instance's");
        }
        let mut run = Xfer {
            host: open.host,
            records: open.records.clone(),
            chain_first: open.chain_first,
            out,
            kind: open.kind,
            fence_id: open.fence_id,
            coverage: Coverage::new(expected),
            pages: open.pages.iter().map(|id| PageId(*id)).collect(),
            slot: open.has_slot.then_some(StateSlotId(open.slot)),
            shared,
            channel: None,
            channel_open: false,
            ended: false,
            windows: BTreeMap::new(),
            jobs_outstanding: 0,
            failed: None,
            cancelled: false,
            page_bytes: open.page_bytes,
            row_bytes: open.row_bytes,
        };
        if let Some(reference) = &open.channel {
            let id = if out {
                Id::Export(xfer)
            } else {
                Id::Claim(xfer)
            };
            match self.open_channel(id, reference) {
                Ok(channel) => {
                    run.channel = Some(channel);
                    run.channel_open = true;
                }
                Err(error) => return self.reject(xfer, &error.to_string()),
            }
        }
        self.xfers.insert(xfer, run);
    }

    fn open_channel(&self, id: Id, reference: &wire::DataChannelRef) -> Result<Channel> {
        let dial = self
            .dial
            .as_ref()
            .ok_or_else(|| Error::new("this agent has no data channel"))?;
        dial.open(net::Open {
            id,
            reference: reference.clone(),
        })
        .map_err(|e| Error::caused_by("cannot open the data channel", e))
    }

    /// A shared window of a transfer.
    fn on_window(&mut self, window: rank::Window) {
        let xfer = window.xfer;
        let Some(spec) = window.spec else { return };
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return;
        };
        if !run.shared || run.cancelled || run.failed.is_some() {
            return;
        }
        let dest = match run.coverage.accept(&spec, run.page_bytes, run.row_bytes) {
            Ok(dest) => dest as usize,
            Err(why) => return self.fail(xfer, why),
        };
        let Some(region) = self
            .shared
            .as_ref()
            .and_then(|(_, regions)| regions.get(spec.region()))
        else {
            return self.fail(xfer, "window in an unknown region".into());
        };
        let Some(base) = region.address(spec.offset, spec.length) else {
            return self.fail(xfer, "window outside its region".into());
        };
        let (count, row) = (spec.page_count as usize, spec.row);
        let rank = self.rank;
        if !run.out {
            // Verify this rank's slices against the transfer checksums before copying.
            let Some(sums) = window.sums else {
                return self.fail(xfer, "a window arrived without checksums".into());
            };
            let bytes = region
                .bytes(spec.offset, spec.length)
                .expect("checked above");
            if sums.pages.len() != count || sums.row.is_some() != row {
                return self.fail(xfer, "window checksums do not match its pages".into());
            }
            let mine = match self.layout.slice_sums(rank, bytes, count as u32, row) {
                Ok(mine) => mine,
                Err(why) => return self.fail(xfer, why.to_string()),
            };
            let bad = mine.pages != sums.pages || mine.row != sums.row;
            if bad {
                return self.fail(xfer, "a page or row failed its checksum".into());
            }
        }
        if run.host {
            let record = spec.record as usize;
            return self.read_window(xfer, spec, record, base, None);
        }
        let layout = self.layout.clone();
        let pages = run.pages.clone();
        let slot = run.slot;
        let direction = if run.out {
            Direction::Out
        } else {
            Direction::In
        };
        run.windows.insert(
            spec.index,
            Win {
                jobs_left: 0,
                spec,
                slot: None,
                persisted: None,
            },
        );
        let jobs = jobs_of(
            direction,
            spec.index,
            count,
            row,
            |i| pages[dest + i],
            slot,
            |i| base + layout.page_at(rank, i as u32) as usize,
            base + layout.row_at(rank, count as u32) as usize,
        );
        self.queue(xfer, spec.index, jobs);
    }

    /// Hand one window of a host transfer to the reader thread.
    fn read_window(
        &mut self,
        xfer: u64,
        spec: wire::WindowSpec,
        record: usize,
        address: usize,
        slot: Option<u32>,
    ) {
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return;
        };
        let Some(wanted) = run.records.get(record) else {
            return self.fail(xfer, "a window names no record".into());
        };
        let Some(key) = wanted.key.as_ref().and_then(codec::key_from) else {
            return self.fail(xfer, "a record names no key".into());
        };
        let put = run.kind == wire::rank_fence::Kind::Put as i32;
        let first = if put {
            (run.chain_first + u64::from(spec.first_page)) as u32
        } else {
            spec.first_page
        };
        let records = self
            .records
            .as_ref()
            .expect("host transfer has records")
            .clone();
        let scatter = slot.is_none().then(|| (self.layout.clone(), self.rank));
        let job = FillJob {
            owner: FillOwner::Export(xfer),
            window: spec.index,
            work: Box::new(move || {
                // The transfer retains its window or staging slot until this read completes.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(address as *mut u8, spec.length as usize)
                };
                let mut records = records.lock().unwrap();
                let mut read = |bytes: &mut [u8]| {
                    if put {
                        records.fill(key, first, spec.page_count, spec.row, bytes)
                    } else {
                        records.read(key, first, spec.page_count, spec.row, bytes)
                    }
                };
                match scatter {
                    Some((layout, rank)) => {
                        let mut packed =
                            vec![0; layout.slice_bytes(rank, spec.page_count, spec.row) as usize];
                        let sums = read(&mut packed)?;
                        layout.scatter(rank, dst, spec.page_count, spec.row, &packed)?;
                        Ok(sums)
                    }
                    None => read(dst),
                }
            }),
        };
        run.jobs_outstanding += 1;
        run.windows.insert(
            spec.index,
            Win {
                jobs_left: 1,
                spec,
                slot,
                persisted: None,
            },
        );
        let sent = self
            .worker
            .as_ref()
            .is_some_and(|worker| worker.submit(job));
        if !sent {
            self.job_done((xfer, spec.index), Err("the record reader is gone".into()));
        }
    }

    /// Collect the reads the reader thread finished.
    fn poll_records(&mut self) {
        let mut done = Vec::new();
        if let Some(worker) = &self.worker {
            worker.collect(&mut done);
        }
        for engine::hostfill::FillDone {
            owner,
            window: index,
            result,
        } in done
        {
            let FillOwner::Export(xfer) = owner else {
                unreachable!()
            };
            let owner = (xfer, index);
            match result {
                Ok(sums) => {
                    if let Some(win) = self
                        .xfers
                        .get_mut(&xfer)
                        .and_then(|run| run.windows.get_mut(&index))
                    {
                        win.persisted = Some(sums);
                    }
                    self.job_done(owner, Ok(()));
                }
                Err(error) => self.job_done(owner, Err(error.to_string())),
            }
        }
    }

    fn queue(&mut self, xfer: u64, window: u32, jobs: Vec<Job>) {
        if let Some(run) = self.xfers.get_mut(&xfer) {
            run.jobs_outstanding += jobs.len() as u32;
            if let Some(win) = run.windows.get_mut(&window) {
                win.jobs_left = jobs.len() as u32;
            }
        }
        for job in jobs {
            engine::queue_job(&mut self.next_job, &mut self.pending, job, (xfer, window));
        }
    }

    // ---- channel ------------------------------------------------------------------

    fn on_net(&mut self, event: Event) {
        match event {
            Event::Window {
                claim,
                window,
                slot,
            } => self.on_net_window(claim, window, slot),
            Event::End { claim } => {
                if let Some(run) = self.xfers.get_mut(&claim) {
                    run.ended = true;
                    if !run.coverage.complete() && run.failed.is_none() && !run.cancelled {
                        self.fail(
                            claim,
                            "the data channel ended before every window arrived".into(),
                        );
                    }
                }
            }
            Event::Granted { id, shape, slot } => match id {
                Id::Export(xfer) => self.on_net_grant(xfer, shape, slot),
                _ => self.release_slot(slot),
            },
            Event::Sent { id, index, slot } => {
                self.release_slot(slot);
                if let Id::Export(xfer) = id
                    && let Some(run) = self.xfers.get_mut(&xfer)
                {
                    run.windows.remove(&index);
                }
            }
            Event::Failed { id, error } => {
                let (Id::Claim(xfer) | Id::Export(xfer) | Id::Put(xfer)) = id;
                if self.xfers.contains_key(&xfer) {
                    self.fail(xfer, error);
                }
            }
            Event::Closed { id } => {
                let (Id::Claim(xfer) | Id::Export(xfer) | Id::Put(xfer)) = id;
                if let Some(run) = self.xfers.get_mut(&xfer) {
                    run.channel = None;
                    run.channel_open = false;
                    let finished = run.coverage.complete()
                        && (!run.out || run.windows.is_empty())
                        && (run.ended || run.out);
                    if !finished && run.failed.is_none() && !run.cancelled {
                        self.fail(
                            xfer,
                            "the data channel closed before the transfer was complete".into(),
                        );
                    }
                }
            }
        }
    }

    fn release_slot(&self, slot: u32) {
        if let Some(staging) = &self.staging {
            staging.release(slot);
        }
    }

    /// A window of an incoming claim arrived whole in a slot of this agent's staging.
    fn on_net_window(&mut self, xfer: u64, window: super::frame::Window, slot: u32) {
        let Some(staging) = self.staging.clone() else {
            return;
        };
        let rank = self.rank;
        let mine = *self.layout.rank(rank);
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return self.release_slot(slot);
        };
        if run.shared || run.out || run.cancelled || run.failed.is_some() {
            return self.release_slot(slot);
        }
        let shape = window.shape;
        let spec = wire::WindowSpec {
            index: shape.index,
            region: wire::Region::Staging as i32,
            offset: staging.offset(slot),
            length: shape.length,
            record: shape.record,
            first_page: shape.first_page,
            page_count: shape.page_count,
            row: shape.row,
        };
        let dest = match run.coverage.accept(&spec, mine.page_len, mine.row_len) {
            Ok(dest) => dest as usize,
            Err(why) => {
                self.release_slot(slot);
                return self.fail(xfer, why);
            }
        };
        let (count, row) = (spec.page_count as usize, spec.row);
        let bytes = staging
            .region()
            .bytes(spec.offset, spec.length)
            .expect("a slot of the staging");
        let page_len = mine.page_len as usize;
        let sums = self.layout.packed_sums(rank, bytes, count as u32, row);
        let bad = sums.map_or(true, |sums| {
            sums.pages != window.page_xxh3 || sums.row != row.then_some(window.row_xxh3)
        });
        if bad {
            self.release_slot(slot);
            return self.fail(xfer, "a page or row failed its checksum".into());
        }
        let base = staging
            .region()
            .address(spec.offset, spec.length)
            .expect("a slot of the staging");
        let pages = run.pages.clone();
        let rowslot = run.slot;
        run.windows.insert(
            spec.index,
            Win {
                jobs_left: 0,
                spec,
                slot: Some(slot),
                persisted: None,
            },
        );
        let jobs = jobs_of(
            Direction::In,
            spec.index,
            count,
            row,
            |i| pages[dest + i],
            rowslot,
            |i| base + i * page_len,
            base + count * page_len,
        );
        self.queue(xfer, spec.index, jobs);
    }

    /// The manager granted a window of an outgoing transfer; its slice is copied into a slot.
    fn on_net_grant(&mut self, xfer: u64, shape: super::frame::Shape, slot: u32) {
        let Some(staging) = self.staging.clone() else {
            return;
        };
        let mine = *self.layout.rank(self.rank);
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return self.release_slot(slot);
        };
        if run.shared || !run.out || run.cancelled || run.failed.is_some() {
            return self.release_slot(slot);
        }
        let spec = wire::WindowSpec {
            index: shape.index,
            region: wire::Region::Staging as i32,
            offset: staging.offset(slot),
            length: shape.length,
            record: shape.record,
            first_page: shape.first_page,
            page_count: shape.page_count,
            row: shape.row,
        };
        let dest = match run.coverage.accept(&spec, mine.page_len, mine.row_len) {
            Ok(dest) => dest as usize,
            Err(why) => {
                self.release_slot(slot);
                return self.fail(xfer, why);
            }
        };
        let (count, row) = (spec.page_count as usize, spec.row);
        let base = staging
            .region()
            .address(spec.offset, spec.length)
            .expect("a slot of the staging");
        if run.host {
            let record = spec.record as usize;
            return self.read_window(xfer, spec, record, base, Some(slot));
        }
        let page_len = mine.page_len as usize;
        let pages = run.pages.clone();
        let rowslot = run.slot;
        run.windows.insert(
            spec.index,
            Win {
                jobs_left: 0,
                spec,
                slot: Some(slot),
                persisted: None,
            },
        );
        let jobs = jobs_of(
            Direction::Out,
            spec.index,
            count,
            row,
            |i| pages[dest + i],
            rowslot,
            |i| base + i * page_len,
            base + count * page_len,
        );
        self.queue(xfer, spec.index, jobs);
    }

    // ---- copier -------------------------------------------------------------------

    fn submit_pending(&mut self) {
        for (owner, result) in
            engine::submit_pending(&self.copier, &mut self.pending, &mut self.jobs)
        {
            self.job_done(owner, result.map_err(|e| e.to_string()));
        }
    }

    fn poll_copier(&mut self) {
        for (owner, result) in engine::poll_jobs(&self.copier, &mut self.jobs) {
            self.job_done(owner, result.map_err(|e| e.to_string()));
        }
    }

    fn job_done(&mut self, (xfer, window): (u64, u32), result: std::result::Result<(), String>) {
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return;
        };
        run.jobs_outstanding = run.jobs_outstanding.saturating_sub(1);
        let Some(win) = run.windows.get_mut(&window) else {
            return;
        };
        win.jobs_left = win.jobs_left.saturating_sub(1);
        let finished = win.jobs_left == 0;
        if let Err(why) = result {
            return self.fail(xfer, why);
        }
        if !finished {
            return;
        }
        self.window_drained(xfer, window);
    }

    /// Every copy of one window drained.
    fn window_drained(&mut self, xfer: u64, index: u32) {
        let rank = self.rank;
        let Some(run) = self.xfers.get(&xfer) else {
            return;
        };
        let Some(win) = run.windows.get(&index) else {
            return;
        };
        let (spec, slot) = (win.spec, win.slot);
        let persisted = win.persisted.clone();
        let (count, row) = (spec.page_count as usize, spec.row);
        let over = run.failed.is_some() || run.cancelled;
        let (shared, out) = (run.shared, run.out);
        if over {
            // Nothing is reported for a window of a transfer that is already over; the
            // slot, if any, goes back.
            if let Some(slot) = slot {
                self.release_slot(slot);
            }
            if let Some(run) = self.xfers.get_mut(&xfer) {
                run.windows.remove(&index);
            }
            return;
        }
        match (shared, out) {
            (true, false) => {
                self.end.send_rank(Msg::WindowDone {
                    xfer,
                    index,
                    ok: true,
                    error: String::new(),
                    sums: None,
                });
                self.forget_window(xfer, index);
            }
            (true, true) => {
                // The slice this rank wrote into the window, hashed after the drain.
                let region = self
                    .shared
                    .as_ref()
                    .and_then(|(_, regions)| regions.get(spec.region()));
                let Some(bytes) = region.and_then(|r| r.bytes(spec.offset, spec.length)) else {
                    return self.fail(xfer, "window outside its region".into());
                };
                let sums = if let Some(persisted) = persisted {
                    SliceSums {
                        pages: persisted.pages,
                        row: persisted.row,
                    }
                } else {
                    match self.layout.slice_sums(rank, bytes, count as u32, row) {
                        Ok(sums) => sums,
                        Err(why) => return self.fail(xfer, why.to_string()),
                    }
                };
                self.end.send_rank(Msg::WindowDone {
                    xfer,
                    index,
                    ok: true,
                    error: String::new(),
                    sums: Some(SliceSums {
                        pages: sums.pages,
                        row: sums.row,
                    }),
                });
                self.forget_window(xfer, index);
            }
            (false, false) => {
                // A channel window leaves its slot once the copy drained.
                if let Some(slot) = slot {
                    self.release_slot(slot);
                }
                self.forget_window(xfer, index);
            }
            (false, true) => {
                // The filled slot goes to the manager; `Sent` returns it.
                let Some(slot) = slot else { return };
                let Some(staging) = self.staging.clone() else {
                    return;
                };
                let bytes = staging
                    .region()
                    .bytes(spec.offset, spec.length)
                    .expect("a slot of the staging");
                let (page_xxh3, row_xxh3) = match persisted {
                    Some(sums) => (sums.pages, sums.row),
                    None => match self.layout.packed_sums(rank, bytes, count as u32, row) {
                        Ok(sums) => (sums.pages, sums.row),
                        Err(why) => return self.fail(xfer, why.to_string()),
                    },
                };
                let sent = run.channel.as_ref().is_some_and(|channel| {
                    channel.send(net::Filled {
                        index,
                        row_xxh3,
                        page_xxh3,
                        slot,
                        length: spec.length,
                    })
                });
                if !sent {
                    self.release_slot(slot);
                    self.fail(xfer, "the data channel is gone".into());
                }
            }
        }
    }

    fn forget_window(&mut self, xfer: u64, index: u32) {
        if let Some(run) = self.xfers.get_mut(&xfer) {
            run.windows.remove(&index);
        }
    }

    // ---- ending a transfer -----------------------------------------------------------

    fn fail(&mut self, xfer: u64, why: String) {
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return;
        };
        if run.failed.is_none() {
            eprintln!("warning: rank {}: transfer {xfer} failed: {why}", self.rank);
            run.failed = Some(why);
        }
        if let Some(channel) = &run.channel {
            channel.stop();
        }
    }

    fn cancel(&mut self, xfer: u64) {
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return;
        };
        run.cancelled = true;
        if let Some(channel) = &run.channel {
            channel.stop();
        }
    }

    fn cancel_all(&mut self) {
        let ids: Vec<u64> = self.xfers.keys().copied().collect();
        for xfer in ids {
            self.cancel(xfer);
        }
    }

    /// Send `Done` for a transfer when it is over and drained.
    fn drive(&mut self, xfer: u64) {
        let Some(run) = self.xfers.get_mut(&xfer) else {
            return;
        };
        // An outgoing channel transfer whose last window was sent ends its task.
        if run.out
            && !run.shared
            && run.coverage.complete()
            && run.windows.is_empty()
            && let Some(channel) = &run.channel
        {
            channel.stop();
        }
        if run.jobs_outstanding > 0 || run.channel_open {
            return;
        }
        let over = run.failed.is_some()
            || run.cancelled
            || (run.coverage.complete()
                && run.windows.is_empty()
                && (run.shared || run.out || run.ended));
        if !over {
            return;
        }
        // A cancelled or failed transfer waits for its slots and queued jobs as well.
        if self.pending.iter().any(|(_, (x, _))| *x == xfer) {
            return;
        }
        let run = self.xfers.remove(&xfer).expect("present");
        for win in run.windows.values() {
            if let Some(slot) = win.slot {
                self.release_slot(slot);
            }
        }
        let ok = run.failed.is_none() && !run.cancelled && run.coverage.complete();
        let error = match (&run.failed, run.cancelled) {
            (Some(why), _) => why.clone(),
            (None, true) => "cancelled".into(),
            (None, false) if !ok => "the transfer did not cover its records".into(),
            _ => String::new(),
        };
        self.end.send_rank(Msg::Done {
            xfer,
            ok,
            error,
            quarantined: false,
        });
        self.attest(run.kind, run.fence_id, true);
    }

    /// Tell the manager this agent stopped touching a transfer's memory.
    fn attest(&mut self, kind: i32, id: u64, drained: bool) {
        self.owed.push((kind, id, drained));
        self.deliver_owed();
    }

    fn deliver_owed(&mut self) {
        let Some(att) = &self.att else { return };
        let mut kept = Vec::new();
        for (kind, id, drained) in std::mem::take(&mut self.owed) {
            let fence = wire::RankFence {
                kind,
                id,
                outcome: if drained {
                    wire::FenceOutcome::Drained
                } else {
                    wire::FenceOutcome::Quarantined
                } as i32,
            };
            if att
                .tx
                .try_send(wire::RankClientMessage {
                    m: Some(rc::M::Fence(fence)),
                })
                .is_err()
            {
                kept.push((kind, id, drained));
            }
        }
        self.owed = kept;
    }

    /// Wait for every transfer to end. `false` when stop was requested or the time ran
    /// out; in the latter case the agent cannot show its copies stopped.
    async fn drain(&mut self) -> bool {
        let started = Instant::now();
        loop {
            self.events();
            if self.xfers.is_empty() && self.jobs.is_empty() && self.pending.is_empty() {
                return true;
            }
            if started.elapsed() > CLOSE_TIMEOUT {
                self.quarantine();
                return false;
            }
            let _ = tokio::time::timeout(BUSY_TICK, self.wake.notified()).await;
        }
    }

    /// Drain cannot be shown: keep every mapping and registration for the life of the
    /// process and say so.
    fn quarantine(&mut self) {
        if self.quarantined {
            return;
        }
        self.quarantined = true;
        let ids: Vec<u64> = self.xfers.keys().copied().collect();
        for xfer in ids {
            if let Some(run) = self.xfers.remove(&xfer) {
                self.end.send_rank(Msg::Done {
                    xfer,
                    ok: false,
                    error: "the agent could not drain".into(),
                    quarantined: true,
                });
                self.owed.push((run.kind, run.fence_id, false));
                std::mem::forget(run);
            }
        }
        self.deliver_owed();
        if let Some(att) = &self.att {
            std::mem::forget(att.regions.clone());
        }
        if let Some(staging) = self.staging.take() {
            std::mem::forget(staging);
        }
        self.set_status(PoolStatus::Disabled(
            "a copy could not be drained; memory is quarantined".into(),
        ));
    }
}

/// Copier jobs for one window: pages and then the row, at most [`WINDOW_PAGES`] pages each.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jobs_of(
    direction: Direction,
    window: u32,
    count: usize,
    row: bool,
    page_id: impl Fn(usize) -> PageId,
    slot: Option<StateSlotId>,
    page_address: impl Fn(usize) -> usize,
    row_address: usize,
) -> Vec<Job> {
    let mut jobs = Vec::new();
    let mut first = 0;
    while first < count || (count == 0 && jobs.is_empty()) {
        let end = (first + WINDOW_PAGES).min(count);
        let pages = (first..end)
            .map(|i| (page_id(i), page_address(i)))
            .collect();
        let last = end == count;
        let row =
            (last && row).then(|| (slot.expect("a window with a row has a slot"), row_address));
        jobs.push(Job {
            id: JobId(0),
            window,
            direction,
            pages,
            row,
        });
        first = end;
        if count == 0 {
            break;
        }
    }
    jobs
}
