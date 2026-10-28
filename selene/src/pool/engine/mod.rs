//! The session engine: every protocol decision of the pool client, with no I/O.
//!
//! The connection task feeds it server messages, worker commands and the
//! passage of time, and sends what it queues. It owns the claim and export
//! state machines, the resync and notice flow, and the drain-before-release
//! rules, so each can be tested by scripting the manager's messages.

pub(crate) mod coverage;
pub(crate) mod hostfill;
#[cfg(test)]
mod net_tests;
mod netio;
mod outbound;
pub(crate) mod ranks;
pub(crate) mod requester;
mod source;
#[cfg(test)]
mod tests;
mod writer;

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{CLOSE_TIMEOUT, HostRegistrar, Job, JobId, SharedCopier};

use super::{
    announce::Announcer,
    codec::{self, Sources},
    identity::Scope,
    net::{self, Dial, Staging},
    shared::{Command, PoolEvent, PoolStatus, Shared},
    shm::Regions,
    wire::{self, client_message::M as Out, server_message::M as In},
};

pub(crate) use ranks::Ranks;

/// The protocol version this client speaks.
pub const PROTOCOL: u32 = 2;
/// How long a lookup may wait for its reply.
pub const LOOKUP_TIMEOUT: Duration = Duration::from_millis(250);
/// Setup allowance asked of the manager; it adds time for the claimed bytes.
pub const CLAIM_DEADLINE_MS: u32 = 10_000;
/// Restore setup allowance, in addition to time for its bytes.
pub const RESTORE_DEADLINE: Duration = Duration::from_secs(12);

/// What identifies and bounds this process's sessions.
#[derive(Clone)]
pub(crate) struct Settings {
    pub pool_name: Option<String>,
    pub gpu_uuids: Vec<[u8; 16]>,
    pub instance_id: [u8; 16],
    pub nonce: [u8; 16],
    pub scope: Scope,
    pub scope_id: [u8; 32],
    pub export_slots: u32,
    pub claim_slots: u32,
    /// The context made current when registering the shared regions.
    pub context: usize,
    /// Set for rank 0 of a multi-process instance: the secret its peers attach with.
    pub rank_secret: Option<[u8; 32]>,
}

impl Settings {
    pub(crate) fn bind_pool(&mut self, name: &str) -> crate::Result<()> {
        if let Some(bound) = &self.pool_name {
            if bound != name {
                return Err(crate::Error::new(
                    "the manager's pool name changed; restart this instance",
                ));
            }
            return Ok(());
        }
        self.scope_id = super::identity::scope_id(
            name,
            &self.scope.model_identity,
            &self.scope.layout_fingerprint,
            &self.scope.geometry,
        )?;
        self.instance_id = super::identity::instance_id(
            name,
            &self.scope.model_identity,
            &self.scope.layout_fingerprint,
            &self.gpu_uuids,
        );
        self.pool_name = Some(name.into());
        Ok(())
    }

    pub(crate) fn page_record_bytes(&self) -> u64 {
        self.scope.geometry.page_record_bytes
    }
    pub(crate) fn row_record_bytes(&self) -> u64 {
        self.scope.geometry.row_record_bytes
    }
}

struct Resync {
    chunks: VecDeque<Vec<wire::IndexEntry>>,
    begun: bool,
    ended: bool,
}

pub(crate) struct Session {
    header: wire::Header,
    pub(crate) limits: wire::Limits,
    pub(crate) regions: Option<Arc<Regions>>,
    installed: bool,
    ready: bool,
    resync: Option<Resync>,
    pub(crate) export_slots: u32,
}

/// Who is waiting for a copier job.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Owner {
    Claim { claim: u64, window: u32 },
    Export { export: u64, window: u32 },
}

/// Why the engine stopped serving a session.
#[derive(Debug)]
pub(crate) enum Ended {
    Closing(String),
    Violation(String),
}

pub(crate) struct Engine {
    pub(crate) st: Arc<Settings>,
    pub(crate) shared: Arc<Shared>,
    announce: Announcer,
    copier: SharedCopier,
    registrar: Arc<dyn HostRegistrar + Send + Sync>,
    sources: Sources,
    /// What the first connect attached to; `None` until then.
    pub(crate) attachment: Option<super::shm::Attachment>,
    pub(crate) sess: Option<Session>,
    /// Regions of a lost session, kept mapped until its copies have drained.
    draining: Vec<Arc<Regions>>,
    out: Vec<wire::ClientMessage>,
    /// Fence-class messages owed from before a reconnect, sent on the next session.
    owed: Vec<Out>,
    next_job: u64,
    jobs: HashMap<JobId, Owner>,
    pending: VecDeque<(Job, Owner)>,
    lookups: HashMap<u64, Instant>,
    pub(crate) requester: requester::State,
    pub(crate) source: source::State,
    pub(crate) writer: writer::State,
    quarantined: bool,
    drain_started: Option<Instant>,
    pub(crate) closing: bool,
    filler: Option<hostfill::HostFiller>,
    /// The registered slots windows of a session without shared memory pass through.
    pub(crate) net: Option<Arc<Staging>>,
    pub(crate) net_events: Arc<net::Events>,
    /// An RDMA transfer failed: the rest of this process's transfers use the stream.
    pub(crate) rdma_broken: Arc<std::sync::atomic::AtomicBool>,
    /// How the open session opens data-channel transfers, when it has no shared memory.
    pub(crate) dial: Option<Arc<dyn Dial>>,
    /// The peer ranks of a multi-process instance.
    pub(crate) ranks: Option<ranks::Ranks>,
    /// Why the pool is off because of a peer rank; the status stays Disabled.
    ranks_off: Option<String>,
}

impl Engine {
    pub(crate) fn new(
        st: Arc<Settings>,
        shared: Arc<Shared>,
        announce: Announcer,
        copier: SharedCopier,
        registrar: Arc<dyn HostRegistrar + Send + Sync>,
    ) -> Self {
        let net_events = net::Events::new(shared.wake.clone());
        Self {
            st,
            shared,
            announce,
            copier,
            registrar,
            sources: Sources::default(),
            sess: None,
            draining: Vec::new(),
            out: Vec::new(),
            owed: Vec::new(),
            next_job: 0,
            jobs: HashMap::new(),
            pending: VecDeque::new(),
            lookups: HashMap::new(),
            requester: Default::default(),
            source: Default::default(),
            writer: Default::default(),
            quarantined: false,
            drain_started: None,
            closing: false,
            filler: None,
            net: None,
            attachment: None,
            net_events,
            rdma_broken: Arc::default(),
            dial: None,
            ranks: None,
            ranks_off: None,
        }
    }

    /// Make this the rank 0 of a multi-process instance.
    pub(crate) fn set_ranks(&mut self, ranks: ranks::Ranks) {
        self.ranks = Some(ranks);
    }

    pub(crate) fn registrar(&self) -> &dyn HostRegistrar {
        &*self.registrar
    }

    /// Registration is over: from here every attempt to register host memory fails.
    pub(crate) fn close_registration(&mut self) {
        self.registrar = Arc::new(super::startup::ClosedRegistrar);
    }

    /// Copies are queued or running, so the copier needs polling soon.
    pub(crate) fn busy(&self) -> bool {
        !self.jobs.is_empty()
            || !self.pending.is_empty()
            || self.source.busy()
            || self.writer.busy()
    }

    pub(crate) fn quarantined(&self) -> bool {
        self.quarantined
    }

    /// Queue a message for the open session. Fence-class messages sent with no
    /// session are owed to the next one; anything else is dropped.
    pub(crate) fn emit(&mut self, message: Out) {
        match &self.sess {
            Some(sess) => self.out.push(wire::ClientMessage {
                header: Some(sess.header),
                m: Some(message),
            }),
            None => {
                if matches!(
                    message,
                    Out::FenceAck(_) | Out::ExportEnd(_) | Out::WindowReleased(_)
                ) {
                    self.owed.push(message);
                }
            }
        }
    }

    /// Everything queued for the wire since the last call.
    pub(crate) fn take_out(&mut self) -> Vec<wire::ClientMessage> {
        std::mem::take(&mut self.out)
    }

    // ---- session ----------------------------------------------------------

    /// The manager opened a session. `regions` are the mapped and registered
    /// shared-memory regions when the connection is same-host.
    pub(crate) fn session_opened(
        &mut self,
        manager_incarnation: u64,
        opened: &wire::SessionOpened,
        limits: wire::Limits,
        regions: Option<Arc<Regions>>,
    ) {
        let export_slots = if regions.is_some() || self.dial.is_some() {
            opened.export_slots.min(self.st.export_slots)
        } else {
            0
        };
        self.announce
            .set_max_entries(limits.max_resync_entries as usize);
        self.shared.set_max_lookups(limits.max_concurrent_lookups);
        self.sess = Some(Session {
            header: wire::Header {
                manager_incarnation,
                session_incarnation: opened.session_incarnation,
            },
            limits,
            regions,
            installed: false,
            ready: opened
                .readiness
                .as_ref()
                .is_some_and(|r| r.state == wire::ReadinessState::Ready as i32),
            resync: None,
            export_slots,
        });
        self.shared.set_status(PoolStatus::Connecting);
        for message in std::mem::take(&mut self.owed) {
            self.emit(message);
        }
        let snapshot = self.announce.begin_session();
        self.begin_resync(snapshot);
    }

    fn begin_resync(&mut self, snapshot: super::announce::Snapshot) {
        let Some(sess) = self.sess.as_mut() else {
            return;
        };
        let per_chunk = (sess.limits.max_resync_chunk_entries as usize).max(1);
        let entries: Vec<wire::IndexEntry> = snapshot.entries.iter().map(codec::entry_to).collect();
        sess.resync = Some(Resync {
            chunks: entries.chunks(per_chunk).map(|c| c.to_vec()).collect(),
            begun: false,
            ended: false,
        });
        sess.installed = false;
        self.update_status();
    }

    /// The stream ended or the manager closed it: stop submitting, fail what
    /// depends on the session, and let the copies drain before anything is
    /// released. Callers keep ticking until [`Engine::quiet`].
    pub(crate) fn session_lost(&mut self, now: Instant) {
        let Some(sess) = self.sess.take() else {
            return;
        };
        self.dial = None;
        self.announce.end_session();
        self.out.clear();
        if let Some(regions) = sess.regions {
            self.draining.push(regions);
        }
        for (_, _) in self.lookups.drain() {}
        self.shared.fail_lookups();
        self.requester_lost();
        self.source_lost();
        self.writer_lost();
        self.drain_started.get_or_insert(now);
        self.shared.set_status(PoolStatus::Local);
        self.pump(now);
    }

    fn update_status(&self) {
        if let Some(why) = &self.ranks_off {
            self.shared.set_status(PoolStatus::Disabled(why.clone()));
            return;
        }
        let ranks_attached = self.ranks_attached();
        let status = match &self.sess {
            None => return,
            Some(s) if !s.installed || !ranks_attached => PoolStatus::Connecting,
            Some(s) if s.ready => PoolStatus::Ready,
            Some(_) => PoolStatus::NotReady,
        };
        self.shared.set_status(status);
    }

    /// No copy, claim, export or queued job remains: memory of a lost session
    /// may be released.
    pub(crate) fn quiet(&self) -> bool {
        self.jobs.is_empty()
            && self.pending.is_empty()
            && self.ranks.as_ref().is_none_or(|r| r.idle())
            && self.requester.is_empty()
            && self.source.is_empty()
            && self.writer.is_empty()
    }

    // ---- server messages --------------------------------------------------

    pub(crate) fn handle(
        &mut self,
        now: Instant,
        message: wire::ServerMessage,
    ) -> std::result::Result<(), Ended> {
        let Some(sess) = &self.sess else {
            return Ok(());
        };
        if let Some(header) = &message.header
            && (header.manager_incarnation != sess.header.manager_incarnation
                || header.session_incarnation != sess.header.session_incarnation)
        {
            return Ok(());
        }
        let Some(m) = message.m else {
            return Ok(());
        };
        match m {
            In::Opened(_) => return Err(Ended::Violation("second SessionOpened".into())),
            In::Closing(closing) => {
                return Err(Ended::Closing(format!(
                    "manager closed the session ({:?}): {}",
                    wire::ClosingReason::try_from(closing.reason),
                    closing.detail
                )));
            }
            In::Readiness(info) => {
                if let Some(sess) = self.sess.as_mut() {
                    sess.ready = info.state == wire::ReadinessState::Ready as i32;
                }
                self.update_status();
            }
            In::ResyncRequired(_) => {
                let snapshot = self.announce.snapshot();
                self.begin_resync(snapshot);
            }
            In::ResyncAck(_) => {
                if let Some(sess) = self.sess.as_mut()
                    && sess.resync.as_ref().is_some_and(|r| r.ended)
                {
                    sess.resync = None;
                    sess.installed = true;
                }
                self.update_status();
            }
            In::LookupReply(reply) => self.on_lookup_reply(now, reply),
            In::ClaimGranted(granted) => self.on_claim_granted(granted),
            In::WindowReady(ready) => self.on_window_ready(ready),
            In::ClaimTerminal(terminal) => self.on_claim_terminal(terminal),
            In::ExportRequest(request) => self.on_export_request(now, request),
            In::ExportFill(fill) => self.on_export_fill(fill),
            In::ExportCancel(cancel) => self.on_export_cancel(cancel),
            In::PutAccept(accept) => self.on_put_accept(accept),
            In::PutGrant(grant) => self.on_put_grant(grant),
            In::Stored(stored) => self.on_stored(stored),
            In::PutRefused(refused) => self.on_put_refused(refused),
        }
        self.pump(now);
        Ok(())
    }

    fn on_lookup_reply(&mut self, now: Instant, reply: wire::LookupReply) {
        if self.lookups.remove(&reply.lookup_id).is_none() {
            return;
        }
        let ok = reply.status == wire::LookupStatus::LookupOk as i32;
        if !ok {
            let retry = Duration::from_millis(u64::from(reply.retry_after_ms).max(100));
            self.shared.pause_lookups(now + retry);
            self.shared.complete_lookup(reply.lookup_id, None);
            return;
        }
        let max = self
            .sess
            .as_ref()
            .map_or(8, |s| s.limits.max_candidates as usize);
        self.sources.begin_reply();
        let candidates = reply
            .candidates
            .iter()
            .take(max)
            .filter_map(|c| self.sources.candidate(c))
            .collect();
        self.shared
            .complete_lookup(reply.lookup_id, Some(candidates));
    }

    // ---- worker commands --------------------------------------------------

    pub(crate) fn command(&mut self, now: Instant, command: Command) {
        match command {
            Command::Lookup(lookup) => self.send_lookup(now, lookup),
            Command::CancelLookup(id) => {
                if self.lookups.remove(&id).is_some() {
                    self.emit(Out::LookupCancel(wire::LookupCancel { lookup_id: id }));
                }
            }
            Command::Fetch(fetch) => self.start_fetch(now, fetch),
            Command::ExportPinned { export, chain } => self.on_export_pinned(export, chain),
            Command::HostPinned { export, pin } => self.on_host_pinned(export, pin),
            Command::Put { id, request } => self.start_put(now, id, request),
            Command::PutCancel(id) => self.cancel_put(id),
            Command::ExportRefused { export, reason } => self.on_export_refused(export, reason),
        }
    }

    fn send_lookup(&mut self, now: Instant, lookup: super::shared::LookupSend) {
        let ready = self.sess.as_ref().is_some_and(|s| s.installed && s.ready);
        if !ready {
            self.shared.complete_lookup(lookup.id, None);
            return;
        }
        let limits = &self.sess.as_ref().unwrap().limits;
        let mut boundaries = lookup.boundaries;
        let cap = limits.max_boundaries as usize;
        if cap > 0 && boundaries.len() > cap {
            // The root and the deepest boundaries.
            let root = boundaries[0];
            let tail = boundaries.split_off(boundaries.len() - (cap - 1));
            boundaries = std::iter::once(root).chain(tail).collect();
        }
        let request = wire::LookupRequest {
            lookup_id: lookup.id,
            boundaries: boundaries.into_iter().map(codec::key_to).collect(),
            before_tokens: lookup.before,
            resident_tokens: lookup.resident,
        };
        self.lookups.insert(lookup.id, now + LOOKUP_TIMEOUT);
        self.emit(Out::Lookup(request));
    }

    // ---- time and pumping -------------------------------------------------

    /// Advance timers and the copier, then flush what is queued.
    pub(crate) fn tick(&mut self, now: Instant) {
        let expired: Vec<u64> = self
            .lookups
            .iter()
            .filter(|(_, deadline)| now >= **deadline)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            self.lookups.remove(&id);
            self.emit(Out::LookupCancel(wire::LookupCancel { lookup_id: id }));
            self.shared.pause_lookups(now + Duration::from_millis(500));
            self.shared.complete_lookup(id, None);
        }
        self.requester_tick(now);
        self.source_tick(now);
        self.writer_tick(now);
        self.pump(now);
        if self.quiet() {
            self.drain_started = None;
            if self.sess.is_none() {
                self.draining.clear();
            }
        } else if self.sess.is_none()
            && self
                .drain_started
                .is_some_and(|start| now.duration_since(start) > CLOSE_TIMEOUT)
        {
            self.quarantine();
        }
    }

    /// Drain cannot be established: keep every pin, mapping and registration
    /// for the life of the process and report it.
    fn quarantine(&mut self) {
        if self.quarantined {
            return;
        }
        self.quarantined = true;
        self.shared.set_quarantined();
        for regions in self.draining.drain(..) {
            std::mem::forget(regions);
        }
        if let Some(staging) = self.net.take() {
            std::mem::forget(staging);
        }
        self.source_quarantine();
        self.requester_quarantine();
        self.writer_quarantine();
        self.shared.set_status(PoolStatus::Disabled(
            "a copy could not be drained; memory is quarantined".into(),
        ));
    }

    /// Poll the copier, submit queued jobs, and send resync chunks and notices.
    /// Returns whether another call has more to do right away.
    pub(crate) fn pump(&mut self, now: Instant) -> bool {
        self.collect_net();
        self.collect_ranks();
        for (owner, result) in poll_jobs(&self.copier, &mut self.jobs) {
            self.copy_done(owner, result);
        }
        self.submit_pending();
        let more = self.flow_index();
        let _ = now;
        more
    }

    fn submit_pending(&mut self) {
        for (owner, result) in submit_pending(&self.copier, &mut self.pending, &mut self.jobs) {
            self.copy_done(owner, result);
        }
    }

    fn copy_done(&mut self, owner: Owner, result: crate::Result<()>) {
        match owner {
            Owner::Claim { claim, window } => self.on_claim_job(claim, window, result),
            Owner::Export { export, window } => self.on_export_job(export, window, result),
        }
    }

    pub(crate) fn queue_job(&mut self, job: Job, owner: Owner) -> JobId {
        queue_job(&mut self.next_job, &mut self.pending, job, owner)
    }

    /// Resync chunks (one per call) and unsent notices.
    fn flow_index(&mut self) -> bool {
        if self.sess.is_none() {
            return false;
        }
        if self.announce.overflowed() {
            self.begin_resync(self.announce.snapshot());
        }
        let mut more = false;
        let mut messages = Vec::new();
        {
            let sess = self.sess.as_mut().unwrap();
            if let Some(resync) = sess.resync.as_mut() {
                if !resync.begun {
                    resync.begun = true;
                    messages.push(Out::ResyncBegin(wire::ResyncBegin {}));
                } else if let Some(entries) = resync.chunks.pop_front() {
                    messages.push(Out::ResyncChunk(wire::ResyncChunk { entries }));
                } else if !resync.ended {
                    resync.ended = true;
                    messages.push(Out::ResyncEnd(wire::ResyncEnd {}));
                }
                more = !resync.ended;
            }
        }
        let snapshot_sent = self
            .sess
            .as_ref()
            .is_some_and(|s| s.resync.as_ref().is_none_or(|r| r.ended));
        for message in messages {
            self.emit(message);
        }
        if snapshot_sent {
            let max = self
                .sess
                .as_ref()
                .map_or(1, |s| s.limits.max_notices_per_batch.max(1) as usize);
            if let Some(notices) = self.announce.take_batch(max) {
                self.emit(Out::Notices(wire::NoticeBatch {
                    notices: notices.iter().map(codec::notice_to).collect(),
                }));
                more = true;
            }
        }
        more
    }

    /// Whether the claim limit allows another restore: the pool must be ready.
    pub(crate) fn ready(&self) -> bool {
        self.sess
            .as_ref()
            .is_some_and(|s| s.installed && s.ready && self.ranks_attached())
    }

    /// Every peer rank is attached to the manager this session is with.
    fn ranks_attached(&self) -> bool {
        match (&self.ranks, &self.sess) {
            (None, _) => true,
            (Some(ranks), Some(sess)) => ranks.attached(sess.header.manager_incarnation),
            (Some(_), None) => false,
        }
    }

    /// Apply what the peer ranks said since the last call.
    fn collect_ranks(&mut self) {
        use ranks::{RankEvent, XferOwner};
        let Some(ranks) = self.ranks.as_mut() else {
            return;
        };
        let events = ranks.poll();
        let any = !events.is_empty();
        for event in events {
            match event {
                RankEvent::WindowDone {
                    rank,
                    xfer,
                    index,
                    result,
                    sums,
                } => {
                    let owner = self.ranks.as_ref().and_then(|r| r.owner(xfer));
                    match owner {
                        Some((XferOwner::Claim(claim), _)) => {
                            self.on_peer_window_done(claim, rank, index, result)
                        }
                        Some((owner, _)) => {
                            let result = self.outbound(owner).and_then(|out| {
                                out.peers.window_done(index, rank, result, sums)
                            });
                            if let Some(result) = result {
                                if let Err(why) = result {
                                    self.fail_outbound(owner, crate::Error::new(format!("rank {rank}: {why}")));
                                }
                                self.drive_outbound(owner, Some(index));
                            }
                        }
                        None => {}
                    }
                }
                RankEvent::Done {
                    rank,
                    xfer,
                    result,
                    quarantined,
                } => {
                    let owner = self.ranks.as_mut().and_then(|r| {
                        let owner = r.owner(xfer);
                        r.forget(xfer);
                        owner
                    });
                    match owner {
                        Some((XferOwner::Claim(claim), _)) => {
                            self.on_peer_done(claim, rank, result, quarantined)
                        }
                        Some((owner, _)) => {
                            if let Some(out) = self.outbound(owner) {
                                out.peers.done(rank, &result, quarantined);
                            }
                            if quarantined {
                                self.quarantine();
                            } else {
                                if let Err(why) = result {
                                    self.fail_outbound(owner, crate::Error::new(format!("rank {rank}: {why}")));
                                }
                                self.drive_outbound(owner, None);
                            }
                        }
                        None => {}
                    }
                }
                RankEvent::Attached | RankEvent::Detached => {}
                RankEvent::Lost { rank, reason } => self.rank_lost(
                    rank,
                    &format!("the link to rank {rank} failed ({reason}); the pool is off"),
                ),
                RankEvent::Disabled { rank, reason } => self.rank_lost(
                    rank,
                    &format!("rank {rank} will not use the pool ({reason}); the pool is off for every rank"),
                ),
            }
        }
        if any {
            self.update_status();
        }
    }

    /// Rank 0 is not using the pool: the peers are told, so none of them does.
    pub(crate) fn close_ranks(&mut self) {
        if let Some(ranks) = &mut self.ranks {
            ranks.close_all();
        }
    }

    /// A peer rank's link failed: whatever it was doing, nothing can show it stopped.
    fn rank_lost(&mut self, rank: usize, message: &str) {
        eprintln!("warning: {message}");
        let open = self
            .ranks
            .as_ref()
            .map(|ranks| ranks.open_on(rank))
            .unwrap_or_default();
        if open.is_empty() {
            self.ranks_off = Some(message.to_string());
            self.shared
                .set_status(PoolStatus::Disabled(message.to_string()));
            self.shared.fail_lookups();
            return;
        }
        self.quarantine();
    }

    // ---- shutdown -----------------------------------------------------------

    /// Stop submitting new work and release what has drained. Returns the
    /// `quiesced` flag for Goodbye: true only if nothing could still be
    /// touched by this process.
    pub(crate) fn begin_close(&mut self, now: Instant) {
        self.closing = true;
        self.lookups.clear();
        self.shared.fail_lookups();
        self.requester_close();
        self.source_close();
        self.writer_close();
        self.pump(now);
    }

    /// Run a window fill on the host copy thread, starting it on first use.
    pub(crate) fn submit_fill(&mut self, job: hostfill::FillJob) -> bool {
        if self.filler.is_none() {
            match hostfill::HostFiller::start(self.shared.wake.clone()) {
                Ok(filler) => self.filler = Some(filler),
                Err(_) => return false,
            }
        }
        self.filler
            .as_ref()
            .is_some_and(|filler| filler.submit(job))
    }

    pub(crate) fn collect_fills(&self) -> Vec<hostfill::FillDone> {
        let mut done = Vec::new();
        if let Some(filler) = &self.filler {
            filler.collect(&mut done);
        }
        done
    }

    pub(crate) fn goodbye(&mut self) {
        let quiesced = self.quiet() && !self.quarantined;
        self.emit(Out::Goodbye(wire::Goodbye { quiesced }));
    }

    pub(crate) fn event(&self, event: PoolEvent) {
        self.shared.push_event(event);
    }
}

pub(crate) fn queue_job<O>(
    next: &mut u64,
    pending: &mut VecDeque<(Job, O)>,
    mut job: Job,
    owner: O,
) -> JobId {
    *next += 1;
    job.id = JobId(*next);
    let id = job.id;
    pending.push_back((job, owner));
    id
}

pub(crate) fn submit_pending<O>(
    copier: &SharedCopier,
    pending: &mut VecDeque<(Job, O)>,
    jobs: &mut HashMap<JobId, O>,
) -> Vec<(O, crate::Result<()>)> {
    let mut failed = Vec::new();
    while let Some((job, owner)) = pending.pop_front() {
        let result = copier.lock().unwrap().submit(job.clone());
        match result {
            Ok(()) => {
                jobs.insert(job.id, owner);
            }
            Err(error) => {
                if copier.lock().unwrap().in_flight() > 0 {
                    pending.push_front((job, owner));
                    break;
                }
                failed.push((owner, Err(error)));
            }
        }
    }
    failed
}

pub(crate) fn poll_jobs<O>(
    copier: &SharedCopier,
    jobs: &mut HashMap<JobId, O>,
) -> Vec<(O, crate::Result<()>)> {
    let mut replies = Vec::new();
    copier.lock().unwrap().poll(&mut replies);
    replies
        .into_iter()
        .filter_map(|reply| jobs.remove(&reply.id).map(|owner| (owner, reply.result)))
        .collect()
}
