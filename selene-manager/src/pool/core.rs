//! The pool actor: sessions, readiness and the clock.

use flare_selene::shared::Limits;

use super::{
    Violation,
    claims::{Claim, ClaimKey},
    index::{InstanceId, ScopeIndex, SourceId},
    ranks::{Agent, Ranks},
    store_bridge::{Link, PutKey, PutRec, StoreHandle},
    validate_scope,
};
use crate::{
    arena::Arena,
    data::channel::{self, Hub},
    local::{pidfd_exited, watch_pid},
    proto::v2::{self, server_message::M},
};
use std::{
    collections::{HashMap, HashSet},
    os::fd::OwnedFd,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};

/// What a data-channel transfer belongs to.
#[derive(Clone, Copy, Debug)]
pub(super) enum ChanOwner {
    /// A peer rank's leg of a claim; the index is into the claim's `legs`.
    ClaimRank(ClaimKey, usize),
    Claim(ClaimKey),
    /// The source leg of a claim relayed through the manager.
    Source(ClaimKey),
    Put(PutKey),
    /// A peer rank's leg of a put; the index is into the put's `legs`.
    PutRank(PutKey, usize),
}

pub type Nonce = [u8; 16];
pub type ProcKey = (InstanceId, Nonce);
pub type ScopeId = [u8; 32];

/// The context an export is sized against when the staging region is checked: Flare's
/// usual per-request limit. The manager does not know a model's real maximum.
const REFERENCE_CONTEXT_TOKENS: u64 = 262_144;

/// Tokens of context one export can hold in `staging` bytes: whole pages and the row.
fn export_tokens(staging: u64, page_bytes: u64, row_bytes: u64, page_tokens: u64) -> u64 {
    staging.saturating_sub(row_bytes) / page_bytes.max(1) * page_tokens
}

#[cfg(test)]
mod staging_tests {
    use super::*;

    #[test]
    fn staging_holds_whole_pages_and_the_row() {
        // 4 GiB, 2 MiB pages of 64 tokens, a 112 MiB row.
        assert_eq!(
            export_tokens(4 << 30, 2 << 20, 112 << 20, 64),
            ((4 << 30) - (112 << 20)) / (2 << 20) * 64
        );
        assert_eq!(export_tokens(1 << 20, 2 << 20, 0, 64), 0);
        assert_eq!(export_tokens(10, 0, 0, 64), 10 * 64);
    }
}

pub struct PoolConfig {
    pub name: String,
    pub limits: Limits,
    pub encryption: v2::Encryption,
    pub transports: Vec<v2::DataTransport>,
    /// The data channel instances without shared memory copy through, when the
    /// manager listens for other hosts.
    pub channel: Option<Hub>,
    /// The manager's shared regions, for the window assembly pass of multi-process instances.
    pub regions: Option<Arc<crate::regions::LocalRegions>>,
    /// True while the shared regions are held for the previous run's processes; claims are then
    /// refused `REGIONS_HELD`.
    pub held: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            limits: Limits::default(),
            encryption: v2::Encryption::Plaintext,
            transports: vec![v2::DataTransport::Tcp],
            channel: None,
            regions: None,
            held: Arc::default(),
        }
    }
}

/// What `Hello` reads without asking the actor.
pub struct Shared {
    pub name: String,
    pub incarnation: u64,
    pub limits: Limits,
    pub readiness: Mutex<v2::ReadinessInfo>,
    /// Prometheus text for the pool's own state, refreshed on the manager's clock.
    pub metrics: Mutex<String>,
    pub shm: v2::SharedMemory,
    pub encryption: v2::Encryption,
    pub transports: Vec<v2::DataTransport>,
    pub uid: u32,
    pub gid: u32,
}

/// What the connection said about its peer.
#[derive(Clone, Copy, Debug)]
pub struct OpenPeer {
    /// A Unix-domain connection (the only kind whose credentials are meaningful).
    pub unix: bool,
    pub pid: u32,
    pub uid: u32,
    pub gid: u32,
}

pub enum Event {
    Open {
        peer: OpenPeer,
        open: Box<v2::OpenSession>,
        tx: mpsc::Sender<v2::ServerMessage>,
        reply: oneshot::Sender<Option<u64>>,
    },
    Message {
        sid: u64,
        message: Box<v2::ClientMessage>,
    },
    Closed {
        sid: u64,
    },
    /// A peer rank's agent attaches.
    RankOpen {
        peer: OpenPeer,
        attach: Box<v2::RankAttach>,
        tx: mpsc::Sender<v2::RankServerMessage>,
        reply: oneshot::Sender<Option<u64>>,
    },
    RankMessage {
        aid: u64,
        message: Box<v2::RankClientMessage>,
    },
    RankClosed {
        aid: u64,
    },
    PutAnswer {
        process: ProcKey,
        put_id: u64,
        answer: std::io::Result<crate::store::PutAnswer>,
    },
    GetAnswer {
        key: super::claims::ClaimKey,
        answer: std::io::Result<crate::store::GetAnswer>,
    },
    Catalog {
        scope: ScopeId,
        event: crate::proto::CatalogEvent,
    },
    /// A data-channel handler reported.
    Channel(channel::Event),
    /// A window assembly pass of a put finished.
    PutPass {
        key: super::store_bridge::PutKey,
        index: u32,
        result: Result<super::pass::PassOut, super::pass::PassError>,
    },
    /// A window assembly pass finished.
    Pass {
        key: super::claims::ClaimKey,
        index: u32,
        result: Result<super::pass::PassOut, super::pass::PassError>,
    },
    Shutdown {
        done: oneshot::Sender<()>,
    },
}

#[derive(Clone)]
pub struct PoolHandle {
    pub events: mpsc::Sender<Event>,
    pub shared: Arc<Shared>,
}

pub(super) struct Session {
    pub sid: u64,
    pub instance: InstanceId,
    pub nonce: Nonce,
    pub scope: ScopeId,
    pub tx: mpsc::Sender<v2::ServerMessage>,
    pub shm: bool,
    pub pid: u32,
    pub export_slots: u32,
    pub claim_slots: u32,
    pub need_resync: bool,
    pub syncing: bool,
    pub lookups: HashSet<u64>,
    /// `Goodbye` with this value was received.
    pub goodbye: Option<bool>,
    /// The outbound queue overflowed; the session is closed at the next turn.
    pub overflowed: bool,
    /// Set for rank 0 of a multi-process instance.
    pub ranks: Option<Ranks>,
}

pub(super) struct Instance {
    pub current: Option<u64>,
    pub scope: ScopeId,
}

pub(super) struct Proc {
    pub pid: Option<u32>,
    pub pidfd: Option<OwnedFd>,
    pub exited: bool,
    pub claim_hw: u64,
    pub put_hw: u64,
    /// When the process last had no session; a process nobody can watch is forgotten after
    /// an hour of that.
    pub idle_since: Option<Instant>,
    /// The process's connection to the manager's store, opened on first use.
    pub link: Option<Link>,
}

impl Proc {
    pub(super) fn watch(&mut self, pid: Option<u32>) {
        if self.pid.is_none()
            && let Some(pid) = pid
        {
            self.pid = Some(pid);
            let (pidfd, exited) = watch_pid(pid);
            self.pidfd = pidfd;
            self.exited |= exited;
        }
    }
}

pub(super) struct ScopeState {
    pub valid: super::ValidScope,
    pub index: ScopeIndex,
}

pub struct Core {
    pub(super) cfg: PoolConfig,
    pub(super) store: Option<StoreHandle>,
    pub(super) self_tx: mpsc::Sender<Event>,
    pub(super) puts: HashMap<PutKey, PutRec>,
    pub(super) present_puts: u64,
    /// Open data-channel transfers by id.
    pub(super) channels: HashMap<u64, ChanOwner>,
    pub(super) catalogs: HashSet<ScopeId>,
    pub(super) shared: Arc<Shared>,
    pub(super) incarnation: u64,
    pub(super) staging: Arena,
    pub(super) sessions: HashMap<u64, Session>,
    pub(super) next_session: u64,
    pub(super) instances: HashMap<InstanceId, Instance>,
    pub(super) procs: HashMap<ProcKey, Proc>,
    /// Attached peer ranks of multi-process instances, by attachment id.
    pub(super) agents: HashMap<u64, Agent>,
    /// A session of this process is being replaced by another of the same process.
    pub(super) reconnecting: Option<ProcKey>,
    pub(super) scopes: HashMap<ScopeId, ScopeState>,
    pub(super) claims: HashMap<super::claims::ClaimKey, Claim>,
    pub(super) exports: HashMap<u64, super::claims::ClaimKey>,
    pub(super) next_export: u64,
    /// Counters by metric name and label, for the metrics text.
    pub(super) counters: std::collections::BTreeMap<(&'static str, String), u64>,
    pub(super) lookup_queue: super::lookup::LookupQueue,
    pub(super) state: v2::ReadinessState,
    pub(super) shutting_down: Option<oneshot::Sender<()>>,
    pub(super) euid: u32,
    pub(super) egid: u32,
}

pub fn random_incarnation() -> u64 {
    use std::io::Read;
    let mut bytes = [0u8; 8];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .expect("read /dev/urandom");
    u64::from_le_bytes(bytes) | 1
}

impl Core {
    /// Builds the actor and its handle. `shm` describes the region files the
    /// store opened; `staging` is a handle on the staging region the claims
    /// carve windows from.
    pub fn new(
        cfg: PoolConfig,
        shm: v2::SharedMemory,
        staging: Arena,
        store: Option<mpsc::Sender<crate::store::StoreCommand>>,
    ) -> (Self, PoolHandle, mpsc::Receiver<Event>) {
        let incarnation = random_incarnation();
        let euid = unsafe { libc::geteuid() };
        let egid = unsafe { libc::getegid() };
        let shared = Arc::new(Shared {
            name: cfg.name.clone(),
            incarnation,
            limits: cfg.limits.clone(),
            readiness: Mutex::new(v2::ReadinessInfo::default()),
            metrics: Mutex::new(String::new()),
            shm,
            encryption: cfg.encryption,
            transports: cfg.transports.clone(),
            uid: euid,
            gid: egid,
        });
        let (events, receiver) = mpsc::channel(4096);
        let mut core = Self {
            cfg,
            store: store.map(|commands| StoreHandle {
                commands,
                next_connection: 1 << 40,
            }),
            self_tx: events.clone(),
            puts: HashMap::new(),
            present_puts: 0,
            channels: HashMap::new(),
            catalogs: HashSet::new(),
            shared: shared.clone(),
            incarnation,
            staging,
            sessions: HashMap::new(),
            next_session: 0,
            instances: HashMap::new(),
            procs: HashMap::new(),
            agents: HashMap::new(),
            reconnecting: None,
            scopes: HashMap::new(),
            claims: HashMap::new(),
            exports: HashMap::new(),
            next_export: 0,
            counters: Default::default(),
            lookup_queue: Default::default(),
            state: v2::ReadinessState::Ready,
            shutting_down: None,
            euid,
            egid,
        };
        core.update_readiness();
        (core, PoolHandle { events, shared }, receiver)
    }

    /// Says once per scope, when its geometry first arrives, whether the staging region
    /// can hold one export of a long context. A GPU export reserves staging for the
    /// whole export, so a longer context than this is refused and recomputed.
    fn check_staging(&self, scope: &super::ValidScope) {
        let geometry = &scope.descriptor;
        let staging = self.staging.capacity() as u64;
        let tokens = export_tokens(
            staging,
            geometry.page_bytes(),
            geometry.row_bytes(),
            geometry.page_tokens,
        );
        if tokens < REFERENCE_CONTEXT_TOKENS {
            eprintln!(
                "selene: warning: --staging {} holds one export of about {tokens} tokens for the \
                 scope that just opened (a page is {} bytes, the row {}); longer contexts are \
                 refused as NO_WINDOWS and recomputed. Use at least {} for {REFERENCE_CONTEXT_TOKENS} tokens.",
                staging,
                geometry.page_bytes(),
                geometry.row_bytes(),
                REFERENCE_CONTEXT_TOKENS.div_ceil(geometry.page_tokens.max(1))
                    * geometry.page_bytes()
                    + geometry.row_bytes(),
            );
        }
    }

    /// Runs until `Shutdown` has drained the claims in flight.
    pub async fn run(mut self, mut events: mpsc::Receiver<Event>) {
        let mut tick = tokio::time::interval(Duration::from_millis(10));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                event = events.recv() => {
                    let Some(event) = event else { break };
                    self.handle(event);
                    // Answer lookups as soon as no more work is waiting, so a lone lookup
                    // does not wait for the clock; a burst still queues and meets the
                    // concurrency bound before the first of it is answered.
                    while events.is_empty() && !self.lookup_queue.is_empty() {
                        self.lookup_pass();
                    }
                }
                _ = tick.tick() => self.tick(),
            }
            if self.shutting_down.is_some() && self.drained() {
                break;
            }
        }
        self.finish_shutdown();
    }

    pub(super) fn handle(&mut self, event: Event) {
        match event {
            Event::Open {
                peer,
                open,
                tx,
                reply,
            } => {
                let sid = self.open(peer, *open, tx);
                let _ = reply.send(sid);
            }
            Event::Message { sid, message } => self.message(sid, *message),
            Event::Closed { sid } => self.session_lost(sid),
            Event::RankOpen {
                peer,
                attach,
                tx,
                reply,
            } => {
                let aid = self.rank_open(peer, *attach, tx);
                let _ = reply.send(aid);
            }
            Event::RankMessage { aid, message } => self.rank_message(aid, *message),
            Event::RankClosed { aid } => self.rank_closed(aid),
            Event::PutAnswer {
                process,
                put_id,
                answer,
            } => self.put_answer(process, put_id, answer),
            Event::GetAnswer { key, answer } => self.get_answer(key, answer),
            Event::Catalog { scope, event } => self.catalog_event(scope, event),
            Event::Channel(event) => self.channel_event(event),
            Event::Pass { key, index, result } => self.pass_done(key, index, result),
            Event::PutPass { key, index, result } => self.put_pass_done(key, index, result),
            Event::Shutdown { done } => self.begin_shutdown(done),
        }
        self.flush_overflowed();
    }

    // ------------------------------------------------------------------ sending

    pub(super) fn send(&mut self, sid: u64, m: M) {
        let incarnation = self.incarnation;
        let Some(session) = self.sessions.get_mut(&sid) else {
            return;
        };
        let message = v2::ServerMessage {
            header: Some(v2::Header {
                manager_incarnation: incarnation,
                session_incarnation: sid,
            }),
            m: Some(m),
        };
        match session.tx.try_send(message) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => session.overflowed = true,
            // The client has gone; its reader reports the end of the stream.
            Err(mpsc::error::TrySendError::Closed(_)) => {}
        }
    }

    /// A session that cannot keep up with its queue is closed rather than allowed to
    /// grow the manager.
    fn flush_overflowed(&mut self) {
        let stuck: Vec<u64> = self
            .sessions
            .values()
            .filter(|session| session.overflowed)
            .map(|session| session.sid)
            .collect();
        for sid in stuck {
            eprintln!("selene: closing session {sid}: its outbound queue is full");
            self.session_lost(sid);
        }
    }

    pub(super) fn close_session(&mut self, sid: u64, reason: v2::ClosingReason, detail: &str) {
        self.send(
            sid,
            M::Closing(v2::SessionClosing {
                reason: reason as i32,
                detail: detail.to_string(),
            }),
        );
        self.session_lost(sid);
    }

    pub(super) fn violation(&mut self, sid: u64, violation: Violation) {
        eprintln!(
            "selene: protocol violation on session {sid}: {}",
            violation.0
        );
        self.close_session(
            sid,
            v2::ClosingReason::ClosingProtocolViolation,
            &violation.0,
        );
    }

    // ------------------------------------------------------------------ open

    pub(super) fn open(
        &mut self,
        peer: OpenPeer,
        open: v2::OpenSession,
        tx: mpsc::Sender<v2::ServerMessage>,
    ) -> Option<u64> {
        let refuse = |status: v2::OpenStatus, core: &Core| {
            let _ = tx.try_send(v2::ServerMessage {
                header: Some(v2::Header {
                    manager_incarnation: core.incarnation,
                    session_incarnation: 0,
                }),
                m: Some(M::Opened(v2::SessionOpened {
                    status: status as i32,
                    ..Default::default()
                })),
            });
            None
        };
        if self.shutting_down.is_some() || self.state == v2::ReadinessState::Draining {
            return refuse(v2::OpenStatus::ShuttingDown, self);
        }
        if open.manager_incarnation != self.incarnation {
            return refuse(v2::OpenStatus::StaleManager, self);
        }
        let Some(scope) = validate_scope(open.scope.as_ref(), &self.cfg.name) else {
            return refuse(v2::OpenStatus::InvalidScope, self);
        };
        let (Ok(instance), Ok(nonce)) = (
            InstanceId::try_from(open.instance_id.as_slice()),
            Nonce::try_from(open.process_nonce.as_slice()),
        ) else {
            return refuse(v2::OpenStatus::InvalidScope, self);
        };
        let ranks = if open.rank_secret.is_empty() {
            None
        } else {
            match <[u8; 32]>::try_from(open.rank_secret.as_slice()) {
                Ok(secret) if scope.descriptor.ranks.len() >= 2 => Some(Ranks {
                    world: scope.descriptor.ranks.len() as u32,
                    secret,
                }),
                _ => return refuse(v2::OpenStatus::InvalidScope, self),
            }
        };
        let limits = &self.cfg.limits;
        let shm = peer.unix && peer.uid == self.euid && peer.gid == self.egid;
        // A session without shared memory serves exports over the data channel.
        let export_slots = if shm || self.cfg.channel.is_some() {
            open.export_slots.min(limits.max_export_slots)
        } else {
            0
        };
        let claim_slots = open.claim_slots.min(limits.max_claims_in_flight);

        if let Some(old) = self.instances.get(&instance).and_then(|i| i.current) {
            // The same process reconnecting keeps its agents; another process does not.
            self.reconnecting = Some((instance, nonce));
            self.close_session(
                old,
                v2::ClosingReason::ClosingSuperseded,
                "a newer session opened",
            );
            self.reconnecting = None;
        }

        self.next_session += 1;
        let sid = self.next_session;
        let proc_key = (instance, nonce);
        let started_pid = peer.unix.then_some(peer.pid).filter(|pid| *pid != 0);
        let entry = self.procs.entry(proc_key).or_insert_with(|| Proc {
            pid: None,
            pidfd: None,
            exited: false,
            claim_hw: 0,
            put_hw: 0,
            idle_since: None,
            link: None,
        });
        entry.idle_since = None;
        entry.watch(started_pid);
        let fresh = !self.scopes.contains_key(&scope.id);
        self.scopes.entry(scope.id).or_insert_with(|| ScopeState {
            valid: scope.clone(),
            index: ScopeIndex::default(),
        });
        if fresh {
            self.watch_catalog(scope.id);
            self.check_staging(&scope);
        }
        // Opening the session is the process telling the manager it is back: the store's
        // connection for it is made now, not at its first window, because a restarted manager
        // holds its shared regions until every process of the previous run has made one (and a
        // process that is refused claims for the hold would never get to its first window).
        if let Some(pid) = started_pid {
            self.link_for(proc_key, scope.id, pid);
        }
        self.instances.insert(
            instance,
            Instance {
                current: Some(sid),
                scope: scope.id,
            },
        );
        self.sessions.insert(
            sid,
            Session {
                sid,
                instance,
                nonce,
                scope: scope.id,
                tx,
                shm,
                pid: peer.pid,
                export_slots,
                claim_slots,
                need_resync: true,
                syncing: false,
                lookups: HashSet::new(),
                goodbye: None,
                overflowed: false,
                ranks,
            },
        );
        let readiness = self.readiness_info();
        let scope_id = scope.id.to_vec();
        self.send(
            sid,
            M::Opened(v2::SessionOpened {
                status: v2::OpenStatus::OpenOk as i32,
                session_incarnation: sid,
                scope_id,
                export_slots,
                readiness: Some(readiness),
            }),
        );
        self.update_readiness();
        Some(sid)
    }

    // ------------------------------------------------------------------ lost

    /// The session's stream ended or it was closed. Everything it owned is lost.
    pub(super) fn session_lost(&mut self, sid: u64) {
        let Some(session) = self.sessions.remove(&sid) else {
            return;
        };
        let proc_key = (session.instance, session.nonce);
        if !self
            .sessions
            .values()
            .any(|other| (other.instance, other.nonce) == proc_key)
        {
            if let Some(proc) = self.procs.get_mut(&proc_key) {
                proc.idle_since = Some(Instant::now());
            }
        }
        if session.goodbye == Some(true) {
            self.party_quiesced(proc_key);
        }
        let is_current = self
            .instances
            .get(&session.instance)
            .is_some_and(|instance| instance.current == Some(sid));
        if is_current {
            if let Some(instance) = self.instances.get_mut(&session.instance) {
                instance.current = None;
            }
            if let Some(scope) = self.scopes.get_mut(&session.scope) {
                scope.index.clear(SourceId::Instance(session.instance));
            }
        }
        self.session_claims_lost(sid, &session);
        self.puts_lost(&session);
        if session.ranks.is_some() && is_current && self.reconnecting != Some(proc_key) {
            self.rank0_lost(session.instance, session.nonce);
        }
        self.update_readiness();
    }

    // ------------------------------------------------------------------ messages

    fn message(&mut self, sid: u64, message: v2::ClientMessage) {
        let Some(header) = message.header else {
            return self.violation(sid, Violation::new("message without a header"));
        };
        if header.manager_incarnation != self.incarnation {
            // A message from before a manager restart: dropped.
            return;
        }
        let Some(body) = message.m else {
            return self.violation(sid, Violation::new("empty message"));
        };
        let fence_class = matches!(
            body,
            v2::client_message::M::FenceAck(_)
                | v2::client_message::M::ClaimRelease(_)
                | v2::client_message::M::WindowReleased(_)
                | v2::client_message::M::ExportEnd(_)
        );
        let current = self.sessions.get(&sid).is_some_and(|session| {
            self.instances
                .get(&session.instance)
                .is_some_and(|instance| instance.current == Some(sid))
        });
        if !self.sessions.contains_key(&sid) {
            return;
        }
        if !current && !fence_class {
            return self.close_session(
                sid,
                v2::ClosingReason::ClosingStaleSession,
                "superseded session",
            );
        }
        if header.session_incarnation != sid && !fence_class {
            return self.close_session(
                sid,
                v2::ClosingReason::ClosingStaleSession,
                "wrong session incarnation",
            );
        }
        use v2::client_message::M as C;
        let result = match body {
            C::Open(_) => Err(Violation::new("a second open on a session")),
            C::Goodbye(goodbye) => {
                if let Some(session) = self.sessions.get_mut(&sid) {
                    session.goodbye = Some(goodbye.quiesced);
                }
                Ok(())
            }
            C::ResyncBegin(m) => self.resync_begin(sid, m),
            C::ResyncChunk(m) => self.resync_chunk(sid, m),
            C::ResyncEnd(m) => self.resync_end(sid, m),
            C::Notices(m) => self.notices(sid, m),
            C::Lookup(m) => self.lookup(sid, m),
            C::LookupCancel(m) => {
                self.lookup_cancel(sid, m.lookup_id);
                Ok(())
            }
            C::Claim(m) => self.claim(sid, m),
            C::ClaimRelease(m) => self.claim_release(sid, m.claim_id),
            C::WindowReleased(m) => self.window_released(sid, m),
            C::CopyResult(m) => self.copy_result(sid, m),
            C::FenceAck(m) => self.fence_ack(sid, m),
            C::PinAck(m) => self.pin_ack(sid, m),
            C::WindowFilled(m) => self.window_filled(sid, m),
            C::ExportEnd(m) => self.export_end(sid, m),
            C::PutBegin(m) => self.put_begin(sid, m),
            C::PutFilled(m) => self.put_filled(sid, m),
            C::PutCommit(m) => self.put_commit(sid, m),
            C::PutCancel(m) => self.put_cancel(sid, m),
        };
        if let Err(violation) = result {
            self.violation(sid, violation);
        }
    }

    // ------------------------------------------------------------------ readiness

    pub(super) fn readiness_info(&self) -> v2::ReadinessInfo {
        v2::ReadinessInfo {
            state: self.state as i32,
        }
    }

    pub(super) fn ready(&self) -> bool {
        self.state == v2::ReadinessState::Ready
    }

    /// Recomputes the readiness state and pushes it when it changed.
    pub(super) fn update_readiness(&mut self) {
        let next = if self.shutting_down.is_some() {
            v2::ReadinessState::Draining
        } else {
            v2::ReadinessState::Ready
        };
        let changed = next != self.state;
        self.state = next;
        let info = self.readiness_info();
        *self.shared.readiness.lock().unwrap() = info;
        if changed {
            let sids: Vec<u64> = self.sessions.keys().copied().collect();
            for sid in sids {
                self.send(sid, M::Readiness(info));
            }
        }
    }

    // ------------------------------------------------------------------ clock

    pub(super) fn tick(&mut self) {
        let now = Instant::now();
        self.poll_exits();
        self.lookup_pass();
        self.claim_timers(now);
        self.pump_store_windows();
        self.prune_procs();
        self.update_readiness();
        self.refresh_metrics();
        self.flush_overflowed();
    }

    /// Forgets processes that are gone and that nothing refers to any more.
    fn prune_procs(&mut self) {
        let referenced: std::collections::HashSet<ProcKey> = self
            .sessions
            .values()
            .map(|session| (session.instance, session.nonce))
            .chain(self.claims.keys().map(|key| (key.0, key.1)))
            .chain(self.claims.values().filter_map(|claim| claim.source_proc))
            .chain(self.puts.keys().map(|key| (key.0, key.1)))
            .chain(self.agents.values().map(|agent| agent.proc()))
            .chain(
                self.claims
                    .values()
                    .flat_map(|c| c.parties.iter())
                    .chain(self.puts.values().flat_map(|r| r.parties.iter()))
                    .filter_map(|p| match p.owner {
                        super::claims::Owner::Proc(proc) => Some(proc),
                        _ => None,
                    }),
            )
            .collect();
        let now = Instant::now();
        let mut forget = Vec::new();
        let mut unlink = Vec::new();
        for (key, proc) in &self.procs {
            if referenced.contains(key) {
                continue;
            }
            let watched = proc.pidfd.is_some() || proc.pid.is_some();
            if !watched && proc.link.is_some() {
                // A remote process's store connection has nothing left on it.
                unlink.push(*key);
            }
            let keep = if watched {
                !proc.exited
            } else {
                // A remote process cannot be watched; forget it after an hour of silence.
                proc.idle_since
                    .is_none_or(|since| now.duration_since(since) < Duration::from_secs(3600))
            };
            if !keep {
                forget.push(*key);
            }
        }
        for key in unlink {
            self.drop_link(key);
        }
        for key in forget {
            self.drop_link(key);
            self.procs.remove(&key);
        }
    }

    fn poll_exits(&mut self) {
        let mut exited = Vec::new();
        for (key, proc) in &mut self.procs {
            if !proc.exited {
                if let Some(fd) = &proc.pidfd {
                    if pidfd_exited(fd) {
                        proc.exited = true;
                        exited.push(*key);
                    }
                }
            }
        }
        for key in exited {
            self.proc_exited(key);
        }
    }

    // ------------------------------------------------------------------ metrics

    pub(super) fn count(&mut self, name: &'static str, label: impl Into<String>) {
        *self.counters.entry((name, label.into())).or_insert(0) += 1;
    }

    fn refresh_metrics(&mut self) {
        let info = self.readiness_info();
        let in_flight = self
            .claims
            .values()
            .filter(|claim| !claim.is_terminal())
            .count();
        let entries: usize = self.scopes.values().map(|scope| scope.index.len()).sum();
        let mut text = String::new();
        let mut gauge =
            |name: &str, value: u64| text.push_str(&format!("selene_pool_{name} {value}\n"));
        gauge("state", info.state as u64);
        gauge("ready", u64::from(self.ready()));
        gauge(
            "regions_held",
            u64::from(self.cfg.held.load(std::sync::atomic::Ordering::Acquire)),
        );
        gauge("sessions", self.sessions.len() as u64);
        gauge("index_entries", entries as u64);
        gauge("claims_in_flight", in_flight as u64);
        gauge("claim_records", self.claims.len() as u64);
        gauge("puts_open", self.puts.len() as u64);
        gauge("lookups_queued", self.lookup_queue.len() as u64);
        gauge("quarantined_bytes", self.quarantined_bytes());
        for ((name, label), value) in &self.counters {
            text.push_str(&format!("selene_pool_{name}{{{label}}} {value}\n"));
        }
        if let Some(hub) = &self.cfg.channel {
            let moved = hub.moved();
            for (index, carrier) in ["stream", "rdma"].into_iter().enumerate() {
                text.push_str(&format!(
                    "selene_pool_channel_windows_total{{carrier=\"{carrier}\"}} {}\nselene_pool_channel_bytes_total{{carrier=\"{carrier}\"}} {}\n",
                    moved.windows[index], moved.bytes[index]
                ));
            }
        }
        *self.shared.metrics.lock().unwrap() = text;
    }

    // ------------------------------------------------------------------ shutdown

    fn begin_shutdown(&mut self, done: oneshot::Sender<()>) {
        self.shutting_down = Some(done);
        self.update_readiness();
    }

    fn drained(&self) -> bool {
        self.claims.values().all(|claim| claim.is_terminal()) && self.channels.is_empty()
    }

    fn finish_shutdown(&mut self) {
        if let Some(hub) = &self.cfg.channel {
            hub.shutdown();
        }
        let sids: Vec<u64> = self.sessions.keys().copied().collect();
        for sid in sids {
            self.close_session(
                sid,
                v2::ClosingReason::ClosingShutdown,
                "the manager is shutting down",
            );
        }
        if let Some(done) = self.shutting_down.take() {
            let _ = done.send(());
        }
    }
}
