//! The client state machine, driven by scripted manager messages, a fake page
//! copier that executes its jobs when the test says so, and heap regions
//! standing in for the shared memory.

use crate::shared::Tier;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use tokio::sync::Notify;

use crate::{
    Direction, ExportChain, ExportId, ExportNode, Job, Layout, PageCopier, PageId, PrefixKey,
    Registration, Reply, Result, SharedCopier, StateSlotId, levels::Level, plan, record::checksum,
    testing::geometry,
};

use super::{
    super::{
        announce::{Announcer, IndexEntry},
        identity::{Geometry, RankPlanes, Scope},
        shared::{Command, FetchPart, PoolEvent, PoolFetch, PoolStatus, Shared},
        shm::{Region, Regions},
        types::{PoolCandidate, SourceRef},
        wire::{self, client_message::M as Out, server_message::M as In},
    },
    Engine, Settings,
};

pub(super) const PAGE: u64 = 8192;
pub(super) const ROW: u64 = 4096;
const MANAGER: u64 = 11;
const SESSION: u64 = 7;

// ---- fakes ------------------------------------------------------------------

#[derive(Default)]
pub(super) struct FakeCopier {
    layout: Option<Layout>,
    /// Device pages and rows by id.
    pub gpu_pages: HashMap<u32, Vec<u8>>,
    pub gpu_rows: HashMap<u32, Vec<u8>>,
    queued: VecDeque<Job>,
    ready: Vec<Reply>,
    pub submitted: Vec<Job>,
    pub fail_next: bool,
    pub reject: bool,
    pub stall: bool,
}

impl FakeCopier {
    fn new() -> Self {
        let g = geometry(64, &[("k", 4096), ("v", 4096)], 4096);
        Self {
            layout: Some(Layout::new(&[g]).unwrap()),
            ..Self::default()
        }
    }

    /// Run every queued job against memory and make its reply available.
    pub fn complete_all(&mut self) {
        while let Some(job) = self.queued.pop_front() {
            let mut result = Ok(());
            if std::mem::take(&mut self.fail_next) {
                result = Err("injected failure".into());
            } else {
                for (page, address) in &job.pages {
                    let host = unsafe {
                        std::slice::from_raw_parts_mut(*address as *mut u8, PAGE as usize)
                    };
                    match job.direction {
                        Direction::In => {
                            self.gpu_pages.insert(page.0, host.to_vec());
                        }
                        Direction::Out => {
                            if let Some(bytes) = self.gpu_pages.get(&page.0) {
                                host.copy_from_slice(bytes);
                            }
                        }
                    }
                }
                if let Some((slot, address)) = job.row {
                    let host =
                        unsafe { std::slice::from_raw_parts_mut(address as *mut u8, ROW as usize) };
                    match job.direction {
                        Direction::In => {
                            self.gpu_rows.insert(slot.0, host.to_vec());
                        }
                        Direction::Out => {
                            if let Some(bytes) = self.gpu_rows.get(&slot.0) {
                                host.copy_from_slice(bytes);
                            }
                        }
                    }
                }
            }
            self.ready.push(Reply {
                id: job.id,
                window: job.window,
                direction: job.direction,
                result,
                bytes: 0,
                elapsed: Duration::ZERO,
            });
        }
    }

    pub fn queued(&self) -> usize {
        self.queued.len()
    }
}

impl PageCopier for FakeCopier {
    fn layout(&self) -> &Layout {
        self.layout.as_ref().unwrap()
    }
    fn submit(&mut self, job: Job) -> Result<()> {
        if self.reject {
            return Err("rejected".into());
        }
        self.submitted.push(job.clone());
        self.queued.push_back(job);
        Ok(())
    }
    fn poll(&mut self, replies: &mut Vec<Reply>) {
        if !self.stall {
            replies.append(&mut self.ready);
        }
    }
    fn in_flight(&self) -> usize {
        self.queued.len() + self.ready.len()
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct NoRegistrar;
impl crate::HostRegistrar for NoRegistrar {
    fn register(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Registration>> {
        Err("not used".into())
    }
    fn map_split(&self, _: &[(usize, i32, usize)]) -> Result<Box<dyn crate::SplitMapping>> {
        Err("not used".into())
    }
}

// ---- the rig ----------------------------------------------------------------

pub(super) struct Rig {
    pub engine: Engine,
    pub copier: Arc<Mutex<FakeCopier>>,
    pub shared: Arc<Shared>,
    pub announce: Announcer,
    pub regions: Arc<Regions>,
    pub now: Instant,
    pub limits: wire::Limits,
}

pub(super) fn key(tokens: u64, salt: u8) -> PrefixKey {
    PrefixKey {
        consumed_tokens: tokens,
        digest: [salt; 16],
    }
}

fn settings(export_slots: u32) -> Arc<Settings> {
    Arc::new(Settings {
        pool_name: None,
        gpu_uuids: vec![[7; 16]],
        instance_id: [9; 16],
        nonce: [5; 16],
        scope: Scope {
            model_identity: [1; 32],
            layout_fingerprint: [2; 16],
            geometry: Geometry {
                page_tokens: 64,
                ranks: vec![RankPlanes {
                    plane_bytes: vec![4096, 4096],
                    row_bytes: 4096,
                }],
                page_record_bytes: PAGE,
                row_record_bytes: ROW,
            },
        },
        scope_id: [3; 32],
        export_slots,
        claim_slots: 2,
        context: 0,
        rank_secret: None,
    })
}

impl Rig {
    pub fn new() -> Self {
        let copier = Arc::new(Mutex::new(FakeCopier::new()));
        let shared_copier: SharedCopier = copier.clone();
        let wake = Arc::new(Notify::new());
        let shared = Shared::new(wake.clone());
        let announce = Announcer::new(wake);
        let engine = Engine::new(
            settings(2),
            shared.clone(),
            announce.clone(),
            shared_copier,
            Arc::new(NoRegistrar),
        );
        Rig {
            engine,
            copier,
            shared,
            announce,
            regions: Arc::new(Regions {
                arena: Region::heap(1 << 20),
                staging: Region::heap(1 << 20),
            }),
            now: Instant::now(),
            limits: super::super::limits::wire_defaults(),
        }
    }

    /// Open a session and complete the first snapshot: the pool is Ready.
    pub fn open(&mut self) -> Vec<wire::ClientMessage> {
        self.open_session(Some(self.regions.clone()))
    }

    pub fn open_session(&mut self, regions: Option<Arc<Regions>>) -> Vec<wire::ClientMessage> {
        let opened = wire::SessionOpened {
            status: wire::OpenStatus::OpenOk as i32,
            session_incarnation: SESSION,
            scope_id: vec![3; 32],
            export_slots: 2,
            readiness: Some(wire::ReadinessInfo {
                state: wire::ReadinessState::Ready as i32,
            }),
        };
        self.engine
            .session_opened(MANAGER, &opened, self.limits, regions);
        let mut out = Vec::new();
        // Begin, every chunk and End.
        for _ in 0..8 {
            self.engine.pump(self.now);
            out.extend(self.engine.take_out());
        }
        assert!(out.iter().any(|m| matches!(m.m, Some(Out::ResyncBegin(_)))));
        self.send(In::ResyncAck(wire::ResyncAck {}));
        self.events();
        out
    }

    pub fn send(&mut self, m: In) -> Vec<wire::ClientMessage> {
        let message = wire::ServerMessage {
            header: Some(wire::Header {
                manager_incarnation: MANAGER,
                session_incarnation: SESSION,
            }),
            m: Some(m),
        };
        let _ = self.engine.handle(self.now, message);
        self.engine.take_out()
    }

    pub fn tick(&mut self, after: Duration) -> Vec<wire::ClientMessage> {
        self.now += after;
        self.engine.tick(self.now);
        self.engine.take_out()
    }

    /// Let the copier finish its jobs and the engine collect the replies.
    pub fn drain_copies(&mut self) -> Vec<wire::ClientMessage> {
        self.copier.lock().unwrap().complete_all();
        self.engine.pump(self.now);
        self.engine.take_out()
    }

    pub fn events(&self) -> Vec<PoolEvent> {
        let mut events = Vec::new();
        self.shared.drain_events(&mut events);
        events
    }

    pub fn command(&mut self, command: Command) -> Vec<wire::ClientMessage> {
        self.engine.command(self.now, command);
        self.engine.pump(self.now);
        self.engine.take_out()
    }

    pub fn staging(&self, offset: u64, bytes: &[u8]) {
        self.regions
            .staging
            .bytes_mut(offset, bytes.len() as u64)
            .unwrap()
            .copy_from_slice(bytes);
    }
}

macro_rules! pick {
    ($out:expr, $variant:ident) => {
        $out.iter()
            .filter_map(|m| match &m.m {
                Some(Out::$variant(x)) => Some(x.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
}

pub(super) fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add(i as u8).wrapping_mul(7))
        .collect()
}

// ---- restore fixtures -------------------------------------------------------

pub(super) fn candidate(
    level: Level,
    source: u64,
    end: PrefixKey,
    nodes: &[(PrefixKey, usize)],
) -> PoolCandidate {
    PoolCandidate {
        plan: plan::Candidate {
            level,
            source: plan::SourceId(source),
            key: end,
            begin_tokens: 0,
            skip_pages: 0,
            nodes: nodes
                .iter()
                .map(|(key, tail_pages)| plan::ChainNode {
                    key: *key,
                    tail_pages: *tail_pages,
                })
                .collect(),
        },
        source: SourceRef {
            instance: vec![source as u8],
            session_incarnation: 1,
        },
        tier: Tier::Dram,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn part(
    candidate: PoolCandidate,
    begin: u64,
    pages: usize,
    dest_page: usize,
    row: bool,
    chain: &[PrefixKey],
    first_page: usize,
    dest: &[u32],
) -> FetchPart {
    FetchPart {
        part: plan::Part {
            level: candidate.plan.level,
            source: candidate.plan.source,
            begin_tokens: begin,
            end: candidate.plan.key,
            first_page,
            dest_page,
            pages,
            row,
            chain: chain.to_vec(),
        },
        slot: row.then_some(StateSlotId(9)),
        pages: dest.iter().map(|p| PageId(*p)).collect(),
        candidate,
    }
}

/// A one-part restore of three pages and the row: records of two and one pages.
pub(super) fn single_part() -> PoolFetch {
    let (n1, n2) = (key(128, 1), key(192, 2));
    let c = candidate(Level::HostDram, 1, n2, &[(n1, 2), (n2, 1)]);
    PoolFetch {
        id: 1,
        page_tokens: 64,
        part: part(c, 0, 3, 0, true, &[n1, n2], 0, &[10, 11, 12]),
    }
}

pub(super) fn records(fetch: &PoolFetch) -> Vec<wire::Record> {
    let part = &fetch.part;
    let nodes = &part.candidate.plan.nodes;
    part.part
        .chain
        .iter()
        .enumerate()
        .map(|(i, k)| wire::Record {
            key: Some(super::super::codec::key_to(*k)),
            first_page: 0,
            pages: nodes[i].tail_pages as u32,
            row: i + 1 == nodes.len(),
        })
        .collect()
}

pub(super) fn window(
    index: u32,
    record: u32,
    first_page: u32,
    count: u32,
    row: bool,
    offset: u64,
) -> wire::WindowSpec {
    wire::WindowSpec {
        index,
        region: wire::Region::Staging as i32,
        offset,
        length: u64::from(count) * PAGE + if row { ROW } else { 0 },
        record,
        first_page,
        page_count: count,
        row,
    }
}

/// Fill the staging region for a window and return the manager's WindowReady.
fn ready(rig: &Rig, claim: u64, spec: wire::WindowSpec, seed: u8) -> wire::WindowReady {
    let bytes = pattern(seed, spec.length as usize);
    rig.staging(spec.offset, &bytes);
    let pages = (0..spec.page_count as usize)
        .map(|i| checksum(&bytes[i * PAGE as usize..(i + 1) * PAGE as usize]))
        .collect();
    let row = spec
        .row
        .then(|| checksum(&bytes[spec.page_count as usize * PAGE as usize..]));
    wire::WindowReady {
        claim_id: claim,
        window: Some(spec),
        page_xxh3: pages,
        row_xxh3: row.unwrap_or(0),
        has_row: row.is_some(),
        ..Default::default()
    }
}

fn granted(claim: u64, fetch: &PoolFetch) -> In {
    In::ClaimGranted(wire::ClaimGranted {
        claim_id: claim,
        records: records(fetch),
        total_pages: fetch.part.part.pages as u64,
        row: fetch.part.part.row,
        channel: None,
        window_credits: 2,
        retained: true,
        ..Default::default()
    })
}

pub(super) fn terminal(
    claim: u64,
    state: wire::ClaimState,
    reason: wire::ClaimReason,
    fence: bool,
) -> In {
    In::ClaimTerminal(wire::ClaimTerminal {
        claim_id: claim,
        state: state as i32,
        reason: reason as i32,
        fence_required: fence,
    })
}

pub(super) fn fetched(events: &[PoolEvent]) -> Vec<(u64, std::result::Result<u64, String>)> {
    events
        .iter()
        .filter_map(|e| match e {
            PoolEvent::Fetched { id, result } => Some((
                *id,
                result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()),
            )),
            _ => None,
        })
        .collect()
}

// ---- index ------------------------------------------------------------------

#[test]
fn the_session_opens_with_a_snapshot_and_becomes_ready_when_it_installs() {
    let mut rig = Rig::new();
    rig.announce.saved(IndexEntry {
        key: key(64, 1),
        parent: None,
        bucket: [0; 16],
        tail_pages: 1,
        row: true,
        tier: Tier::Dram,
        bytes: PAGE,
    });
    let out = rig.open_session(Some(rig.regions.clone()));
    let begin = pick!(out, ResyncBegin);
    let chunks = pick!(out, ResyncChunk);
    let end = pick!(out, ResyncEnd);
    assert_eq!((begin.len(), chunks.len(), end.len()), (1, 1, 1));
    assert_eq!(chunks[0].entries.len(), 1);
    assert_eq!(rig.shared.status(), PoolStatus::Ready);
}

#[test]
fn notices_follow_the_snapshot_in_order_and_the_manager_not_ready_stops_lookups_only() {
    let mut rig = Rig::new();
    rig.open();
    for n in 1..=3u64 {
        rig.announce.saved(IndexEntry {
            key: key(n * 64, n as u8),
            parent: None,
            bucket: [0; 16],
            tail_pages: 1,
            row: true,
            tier: Tier::Gpu,
            bytes: PAGE,
        });
    }
    rig.engine.pump(rig.now);
    let out = rig.engine.take_out();
    let batches = pick!(out, Notices);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].notices.len(), 3);
    rig.send(In::Readiness(wire::ReadinessInfo {
        state: wire::ReadinessState::Draining as i32,
    }));
    assert_eq!(rig.shared.status(), PoolStatus::NotReady);
    rig.announce.saved(IndexEntry {
        key: key(999, 9),
        parent: None,
        bucket: [0; 16],
        tail_pages: 1,
        row: true,
        tier: Tier::Gpu,
        bytes: PAGE,
    });
    rig.engine.pump(rig.now);
    assert_eq!(
        pick!(rig.engine.take_out(), Notices)[0].notices.len(),
        1,
        "notices still flow"
    );
}

#[test]
fn a_resync_request_from_the_manager_takes_a_new_snapshot() {
    let mut rig = Rig::new();
    rig.open();
    rig.announce.saved(IndexEntry {
        key: key(64, 1),
        parent: None,
        bucket: [0; 16],
        tail_pages: 1,
        row: true,
        tier: Tier::Gpu,
        bytes: PAGE,
    });
    rig.engine.pump(rig.now);
    rig.engine.take_out();
    let mut out = rig.send(In::ResyncRequired(wire::ResyncRequired {
        reason: wire::ResyncReason::ResyncStateLost as i32,
    }));
    for _ in 0..4 {
        rig.engine.pump(rig.now);
        out.extend(rig.engine.take_out());
    }
    let begin = pick!(out, ResyncBegin);
    assert_eq!(begin.len(), 1);
    assert_eq!(pick!(out, ResyncChunk)[0].entries.len(), 1);
    assert_eq!(rig.shared.status(), PoolStatus::Connecting);
    rig.send(In::ResyncAck(wire::ResyncAck {}));
    assert_eq!(rig.shared.status(), PoolStatus::Ready);
}

// ---- lookup -----------------------------------------------------------------

#[test]
fn a_lookup_that_times_out_is_cancelled_ends_without_candidates_and_pauses_new_ones() {
    let mut rig = Rig::new();
    rig.open();
    let ticket = rig
        .shared
        .begin_lookup(rig.now, vec![key(0, 0), key(64, 1)], 128, 0);
    let commands = rig.shared.take_commands();
    let out = rig.command(commands.into_iter().next().unwrap());
    assert_eq!(pick!(out, Lookup).len(), 1);
    let out = rig.tick(super::LOOKUP_TIMEOUT + Duration::from_millis(1));
    assert_eq!(pick!(out, LookupCancel).len(), 1);
    assert!(matches!(ticket.state(), crate::pool::LookupState::None));
    let again = rig
        .shared
        .begin_lookup(rig.now, vec![key(0, 0), key(64, 2)], 128, 0);
    assert!(
        matches!(again.state(), crate::pool::LookupState::None),
        "no retry storm"
    );
}

#[test]
fn lookup_replies_reach_only_their_ticket_and_refusals_pause_new_lookups() {
    let mut rig = Rig::new();
    rig.open();
    let t1 = rig
        .shared
        .begin_lookup(rig.now, vec![key(0, 0), key(64, 1)], 128, 0);
    let t2 = rig
        .shared
        .begin_lookup(rig.now, vec![key(0, 0), key(64, 1)], 128, 0);
    let mut commands = rig.shared.take_commands();
    assert_eq!(commands.len(), 2);
    let Command::Lookup(sent) = commands.remove(0) else {
        panic!()
    };
    let id = sent.id;
    rig.command(Command::Lookup(sent));
    rig.command(commands.remove(0));
    let candidate = wire::Candidate {
        level: wire::Level::HostDram as i32,
        tier: wire::Tier::Dram as i32,
        source: Some(wire::SourceRef {
            instance_id: vec![1],
            session_incarnation: 3,
        }),
        key: Some(super::super::codec::key_to(key(64, 1))),
        begin_tokens: 0,
        pages: 1,
        bytes: PAGE,
        row: true,
        nodes: vec![wire::ChainNode {
            key: Some(super::super::codec::key_to(key(64, 1))),
            tail_pages: 1,
            row: true,
        }],
        skip_pages: 0,
    };
    let mut malformed = candidate.clone();
    malformed.key = Some(wire::PrefixKey {
        consumed_tokens: 64,
        digest: vec![1, 2],
    });
    rig.send(In::LookupReply(wire::LookupReply {
        lookup_id: id,
        status: wire::LookupStatus::LookupOk as i32,
        candidates: vec![malformed, candidate],
        retry_after_ms: 0,
    }));
    match t1.state() {
        crate::pool::LookupState::Ready(answer) => assert_eq!(answer.candidates.len(), 1),
        other => panic!("{other:?}"),
    }
    assert!(matches!(t2.state(), crate::pool::LookupState::Pending));
    drop(t2);
    for command in rig.shared.take_commands() {
        rig.command(command);
    }
    let t3 = rig
        .shared
        .begin_lookup(rig.now, vec![key(0, 0), key(128, 4)], 256, 0);
    let Command::Lookup(sent) = rig.shared.take_commands().remove(0) else {
        panic!()
    };
    let id = sent.id;
    rig.command(Command::Lookup(sent));
    rig.send(In::LookupReply(wire::LookupReply {
        lookup_id: id,
        status: wire::LookupStatus::LookupNotReady as i32,
        candidates: Vec::new(),
        retry_after_ms: 500,
    }));
    assert!(matches!(t3.state(), crate::pool::LookupState::None));
    let paused = rig
        .shared
        .begin_lookup(rig.now, vec![key(0, 0), key(192, 5)], 256, 0);
    assert!(matches!(paused.state(), crate::pool::LookupState::None));
}

// ---- claims -----------------------------------------------------------------

/// Start the single-part restore and grant it; returns the claim id.
fn granted_single(rig: &mut Rig) -> (PoolFetch, u64) {
    let fetch = single_part();
    let out = rig.command(Command::Fetch(fetch.clone()));
    let claims = pick!(out, Claim);
    assert_eq!(claims.len(), 1);
    let claim = claims[0].claim_id;
    assert_eq!(claims[0].begin_tokens, 0);
    assert!(claims[0].want_row);
    assert_eq!(claims[0].mode, wire::CopyMode::SharedWindows as i32);
    rig.send(granted(claim, &fetch));
    (fetch, claim)
}

fn windows_for(rig: &mut Rig, claim: u64) -> Vec<wire::ClientMessage> {
    let w0 = ready(rig, claim, window(0, 0, 0, 2, false, 0), 1);
    let w1 = ready(rig, claim, window(1, 1, 0, 1, true, 2 * PAGE), 2);
    let mut out = rig.send(In::WindowReady(w0));
    out.extend(rig.send(In::WindowReady(w1)));
    out
}

#[test]
fn a_restore_copies_every_window_releases_after_the_drain_and_reports_after_the_terminal() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    let out = windows_for(&mut rig, claim);
    assert!(
        pick!(out, WindowReleased).is_empty(),
        "nothing is returned before its copy drained"
    );
    assert!(pick!(out, CopyResult).is_empty());
    let jobs = rig.copier.lock().unwrap().submitted.clone();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|j| j.direction == Direction::In));
    assert_eq!(
        jobs[0].pages.iter().map(|(p, _)| p.0).collect::<Vec<_>>(),
        [10, 11]
    );
    assert_eq!(
        jobs[1].pages.iter().map(|(p, _)| p.0).collect::<Vec<_>>(),
        [12]
    );
    assert_eq!(jobs[1].row.map(|(slot, _)| slot), Some(StateSlotId(9)));
    let out = rig.drain_copies();
    let released = pick!(out, WindowReleased);
    assert_eq!(released.iter().map(|r| r.index).collect::<Vec<_>>(), [0, 1]);
    let verdict = pick!(out, CopyResult);
    assert_eq!(verdict.len(), 1);
    assert_eq!(verdict[0].outcome, wire::CopyOutcome::CopyOk as i32);
    assert!(
        fetched(&rig.events()).is_empty(),
        "COPIED is the manager's word"
    );
    {
        let copier = rig.copier.lock().unwrap();
        let sent = pattern(1, 2 * PAGE as usize);
        assert_eq!(copier.gpu_pages[&10], sent[..PAGE as usize]);
        assert_eq!(copier.gpu_pages[&11], sent[PAGE as usize..]);
        assert_eq!(
            copier.gpu_pages[&12],
            pattern(2, (PAGE + ROW) as usize)[..PAGE as usize]
        );
        assert_eq!(
            copier.gpu_rows[&9],
            pattern(2, (PAGE + ROW) as usize)[PAGE as usize..]
        );
    }
    rig.send(terminal(
        claim,
        wire::ClaimState::Copied,
        wire::ClaimReason::ReasonNone,
        true,
    ));
    let events = rig.events();
    assert_eq!(fetched(&events), [(1, Ok(3 * PAGE + ROW))]);
}

#[test]
fn records_that_do_not_match_the_plan_are_abandoned_after_nothing_was_copied() {
    let mut rig = Rig::new();
    rig.open();
    let fetch = single_part();
    let claim = pick!(rig.command(Command::Fetch(fetch.clone())), Claim)[0].claim_id;
    let mut wrong = records(&fetch);
    wrong[0].key = Some(super::super::codec::key_to(key(128, 99)));
    let out = rig.send(In::ClaimGranted(wire::ClaimGranted {
        claim_id: claim,
        records: wrong,
        total_pages: 3,
        row: true,
        channel: None,
        window_credits: 2,
        retained: true,
        ..Default::default()
    }));
    let verdict = pick!(out, CopyResult);
    assert_eq!(verdict[0].outcome, wire::CopyOutcome::CopyAbandoned as i32);
    assert!(rig.copier.lock().unwrap().submitted.is_empty());
    let events = rig.events();
    assert_eq!(fetched(&events).len(), 1);
    assert!(fetched(&events)[0].1.is_err());
}

#[test]
fn a_corrupt_window_is_never_copied_and_the_restore_fails_with_a_checksum_verdict() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    let mut w0 = ready(&rig, claim, window(0, 0, 0, 2, false, 0), 1);
    // Corrupt the persisted bytes after the source computed its checksums.
    rig.regions.staging.bytes_mut(5, 1).unwrap()[0] ^= 0xff;
    let out = rig.send(In::WindowReady(std::mem::take(&mut w0)));
    assert_eq!(
        pick!(out, WindowReleased).len(),
        1,
        "an uncopied window goes straight back"
    );
    let verdict = pick!(out, CopyResult);
    assert_eq!(verdict[0].outcome, wire::CopyOutcome::CopyChecksum as i32);
    assert!(rig.copier.lock().unwrap().submitted.is_empty());
    let events = rig.events();
    assert!(fetched(&events)[0].1.is_err());
    // The manager's terminal needs a fence only; nothing was in flight.
    let out = rig.send(terminal(
        claim,
        wire::ClaimState::Failed,
        wire::ClaimReason::Checksum,
        true,
    ));
    assert_eq!(
        pick!(out, FenceAck)[0].outcome,
        wire::FenceOutcome::Drained as i32
    );
}

#[test]
fn a_failure_with_a_copy_in_flight_reports_and_fences_only_after_it_drains() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    windows_for(&mut rig, claim);
    assert_eq!(rig.copier.lock().unwrap().queued(), 2);
    // The manager gives up while both copies run.
    let out = rig.send(terminal(
        claim,
        wire::ClaimState::Failed,
        wire::ClaimReason::Deadline,
        true,
    ));
    assert!(
        pick!(out, FenceAck).is_empty(),
        "no fence while copies read the windows"
    );
    assert!(pick!(out, CopyResult).is_empty());
    assert!(
        fetched(&rig.events()).is_empty(),
        "the destination pages are still being written"
    );
    let out = rig.drain_copies();
    assert_eq!(
        pick!(out, FenceAck)[0].outcome,
        wire::FenceOutcome::Drained as i32
    );
    assert_eq!(pick!(out, WindowReleased).len(), 2);
    let events = rig.events();
    assert!(fetched(&events)[0].1.is_err());
}

#[test]
fn a_late_grant_is_never_used_and_its_terminal_is_fenced_at_once() {
    let mut rig = Rig::new();
    rig.open();
    let fetch = single_part();
    // A claim this client does not hold (never heard of, or restarted).
    let out = rig.send(granted(77, &fetch));
    assert!(out.is_empty());
    let w = ready(&rig, 77, window(0, 0, 0, 2, false, 0), 1);
    assert!(rig.send(In::WindowReady(w)).is_empty());
    assert!(rig.copier.lock().unwrap().submitted.is_empty());
    let out = rig.send(terminal(
        77,
        wire::ClaimState::Failed,
        wire::ClaimReason::Cancelled,
        true,
    ));
    assert_eq!(pick!(out, FenceAck).len(), 1);
    // Released before the grant arrived.
    let claim = pick!(rig.command(Command::Fetch(fetch.clone())), Claim)[0].claim_id;
    let out = rig.tick(super::RESTORE_DEADLINE + Duration::from_secs(1));
    assert_eq!(pick!(out, ClaimRelease)[0].claim_id, claim);
    assert!(
        rig.send(granted(claim, &fetch)).is_empty(),
        "the late grant is ignored"
    );
    let out = rig.send(terminal(
        claim,
        wire::ClaimState::Failed,
        wire::ClaimReason::Cancelled,
        true,
    ));
    assert_eq!(pick!(out, FenceAck).len(), 1);
    assert!(rig.copier.lock().unwrap().submitted.is_empty());
    assert!(fetched(&rig.events())[0].1.is_err());
}

#[test]
fn coverage_must_be_exact_before_a_copy_ok_is_sent() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    // Only the first window, released: not complete.
    let w0 = ready(&rig, claim, window(0, 0, 0, 2, false, 0), 1);
    rig.send(In::WindowReady(w0));
    let out = rig.drain_copies();
    assert_eq!(pick!(out, WindowReleased).len(), 1);
    assert!(
        pick!(out, CopyResult).is_empty(),
        "later windows are still to come"
    );
    // A duplicate ordinal is a protocol error: COPY_ERROR, not COPIED.
    let dup = ready(&rig, claim, window(0, 1, 0, 1, true, 2 * PAGE), 2);
    let out = rig.send(In::WindowReady(dup));
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyErrorOutcome as i32
    );
}

#[test]
fn a_claim_whose_row_has_not_arrived_is_not_covered_until_its_window_does() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    let w0 = ready(&rig, claim, window(0, 0, 0, 2, false, 0), 1);
    rig.send(In::WindowReady(w0));
    // The pages are all in, without the row: the claim is not covered yet.
    let w1 = ready(&rig, claim, window(1, 1, 0, 1, false, 2 * PAGE), 2);
    rig.send(In::WindowReady(w1));
    let out = rig.drain_copies();
    assert!(
        pick!(out, CopyResult).is_empty(),
        "the row is still to come, so the claim is not covered"
    );
    // The row may follow in a window of its own, which completes the record.
    let w2 = ready(&rig, claim, window(2, 1, 1, 0, true, 3 * PAGE), 3);
    rig.send(In::WindowReady(w2));
    let out = rig.drain_copies();
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyOk as i32
    );
}

#[test]
fn a_window_that_repeats_a_page_is_not_a_row_of_its_own() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    for spec in [
        window(0, 0, 0, 2, false, 0),
        window(1, 1, 0, 1, false, 2 * PAGE),
    ] {
        let ready = ready(&rig, claim, spec, 1);
        rig.send(In::WindowReady(ready));
    }
    let wrong = ready(&rig, claim, window(2, 1, 0, 1, true, 3 * PAGE), 3);
    rig.send(In::WindowReady(wrong));
    let out = rig.drain_copies();
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyErrorOutcome as i32
    );
}

#[test]
fn goodbye_is_quiesced_only_when_nothing_is_left_to_drain() {
    let mut rig = Rig::new();
    rig.open();
    rig.engine.goodbye();
    let out = rig.engine.take_out();
    assert!(pick!(out, Goodbye)[0].quiesced);
    let (_, claim) = granted_single(&mut rig);
    windows_for(&mut rig, claim);
    rig.copier.lock().unwrap().stall = true;
    rig.engine.goodbye();
    let out = rig.engine.take_out();
    assert!(
        !pick!(out, Goodbye)[0].quiesced,
        "copies still read the windows"
    );
}

// ---- loss -------------------------------------------------------------------

#[test]
fn losing_the_session_mid_copy_drains_before_failing_and_owes_the_fence_to_the_next_session() {
    let mut rig = Rig::new();
    rig.open();
    let (_, claim) = granted_single(&mut rig);
    windows_for(&mut rig, claim);
    rig.engine.session_lost(rig.now);
    assert!(!rig.engine.quiet(), "copies still read the regions");
    assert!(fetched(&rig.events()).is_empty());
    assert_eq!(rig.shared.status(), PoolStatus::Local);
    rig.copier.lock().unwrap().complete_all();
    rig.engine.tick(rig.now);
    assert!(rig.engine.quiet());
    let events = rig.events();
    assert!(fetched(&events)[0].1.is_err());
    // The next session receives the acknowledgment the old one could not.
    let out = rig.open_session(Some(rig.regions.clone()));
    let acks = pick!(out, FenceAck);
    assert_eq!(acks.len(), 1);
    assert_eq!(acks[0].outcome, wire::FenceOutcome::Drained as i32);
}

#[test]
fn a_fetch_is_refused_with_one_failed_event_when_the_pool_is_not_ready() {
    let mut rig = Rig::new();
    let out = rig.command(Command::Fetch(single_part()));
    assert!(out.is_empty());
    assert_eq!(fetched(&rig.events()).len(), 1);
}

#[test]
fn restores_beyond_the_claim_slots_wait_for_a_slot_instead_of_being_refused() {
    let mut rig = Rig::new();
    rig.open();
    let mut started = Vec::new();
    for id in 1..=3 {
        let mut fetch = single_part();
        fetch.id = id;
        let out = rig.command(Command::Fetch(fetch.clone()));
        started.push((fetch, pick!(out, Claim)));
    }
    assert_eq!(started[0].1.len(), 1);
    assert_eq!(started[1].1.len(), 1);
    assert!(
        started[2].1.is_empty(),
        "two slots are taken: the third waits"
    );
    assert!(fetched(&rig.events()).is_empty(), "waiting is not failing");
    // The first restore ends and the third takes its slot.
    let claim = started[0].1[0].claim_id;
    rig.send(granted(claim, &started[0].0));
    windows_for(&mut rig, claim);
    rig.drain_copies();
    let out = rig.send(terminal(
        claim,
        wire::ClaimState::Copied,
        wire::ClaimReason::ReasonNone,
        true,
    ));
    assert_eq!(
        pick!(out, Claim).len(),
        1,
        "the waiting restore claims the freed slot"
    );
}

#[test]
fn a_large_slow_restore_and_its_queued_successor_outlive_the_setup_allowance() {
    let mut rig = Rig::new();
    rig.open();
    let pages = 8192u32;
    let end = key(u64::from(pages) * 64, 3);
    let source = candidate(Level::HostNvme, 1, end, &[(end, pages as usize)]);
    let dest: Vec<_> = (0..pages).collect();
    let mut fetch = PoolFetch {
        id: 1,
        page_tokens: 64,
        part: part(source, 0, pages as usize, 0, true, &[end], 0, &dest),
    };
    let out = rig.command(Command::Fetch(fetch.clone()));
    let claim = pick!(out, Claim)[0].claim_id;
    rig.send(granted(claim, &fetch));
    for id in 2..=3 {
        fetch.id = id;
        let out = rig.command(Command::Fetch(fetch.clone()));
        assert_eq!(pick!(out, Claim).len(), usize::from(id == 2));
    }
    for first in (0..pages).step_by(64) {
        rig.tick(Duration::from_millis(120));
        let spec = window(first / 64, 0, first, 64, first + 64 == pages, 0);
        let message = ready(&rig, claim, spec, 17);
        rig.send(In::WindowReady(message));
        rig.drain_copies();
        assert!(fetched(&rig.events()).is_empty());
    }
    let out = rig.send(terminal(
        claim,
        wire::ClaimState::Copied,
        wire::ClaimReason::ReasonNone,
        true,
    ));
    assert_eq!(
        pick!(out, Claim).len(),
        1,
        "the queued restore still claims"
    );
    let events = rig.events();
    let results = fetched(&events);
    assert_eq!(results.len(), 1);
    assert!(results[0].1.is_ok());
    let copier = rig.copier.lock().unwrap();
    assert_eq!(copier.gpu_pages.len(), pages as usize);
    assert_eq!(copier.gpu_pages[&0], pattern(17, PAGE as usize));
    assert_eq!(copier.gpu_pages[&(pages - 1)], pattern(17, PAGE as usize));
    assert_eq!(copier.gpu_rows[&9], pattern(17, ROW as usize));
}

#[test]
fn a_claim_refused_because_the_regions_are_held_is_retried_not_failed() {
    let mut rig = Rig::new();
    rig.open();
    let fetch = single_part();
    let out = rig.command(Command::Fetch(fetch.clone()));
    let first = pick!(out, Claim)[0].claim_id;
    let out = rig.send(terminal(
        first,
        wire::ClaimState::Failed,
        wire::ClaimReason::RegionsHeld,
        false,
    ));
    assert!(pick!(out, Claim).is_empty(), "not at once");
    assert!(
        fetched(&rig.events()).is_empty(),
        "a held refusal does not fail the restore"
    );
    // The first tick starts the wait; the claim is made again when it is over.
    assert!(pick!(rig.tick(Duration::from_millis(1)), Claim).is_empty());
    assert!(pick!(rig.tick(Duration::from_millis(100)), Claim).is_empty());
    let again = pick!(rig.tick(Duration::from_millis(300)), Claim);
    assert_eq!(again.len(), 1, "it claims again inside its deadline");
    assert_ne!(again[0].claim_id, first);
    // Refused again and again, it gives up when the restore's own deadline passes.
    let second = again[0].claim_id;
    rig.send(terminal(
        second,
        wire::ClaimState::Failed,
        wire::ClaimReason::RegionsHeld,
        false,
    ));
    rig.tick(super::RESTORE_DEADLINE + Duration::from_secs(1));
    let events = rig.events();
    let results = fetched(&events);
    assert_eq!(results.len(), 1);
    assert!(results[0].1.is_err());
}

#[test]
fn a_busy_source_retries_until_the_scaled_restore_deadline() {
    let mut rig = Rig::new();
    rig.open();
    let fetch = single_part();
    let bytes = fetch.part.part.pages as u64 * PAGE + ROW;
    let deadline = super::super::transfer_timeout(super::RESTORE_DEADLINE, bytes);
    let out = rig.command(Command::Fetch(fetch));
    let mut claim = pick!(out, Claim)[0].claim_id;
    // Repeated refusals never reset the restore's original deadline.
    let started = rig.now;
    while rig.now - started < deadline - Duration::from_millis(500) {
        let out = rig.send(terminal(
            claim,
            wire::ClaimState::Failed,
            wire::ClaimReason::SourceLimit,
            false,
        ));
        assert!(pick!(out, Claim).is_empty());
        assert!(fetched(&rig.events()).is_empty());
        assert!(pick!(rig.tick(Duration::from_millis(1)), Claim).is_empty());
        let again = pick!(rig.tick(Duration::from_millis(300)), Claim);
        assert_eq!(again.len(), 1);
        assert_ne!(again[0].claim_id, claim);
        claim = again[0].claim_id;
    }
    rig.send(terminal(
        claim,
        wire::ClaimState::Failed,
        wire::ClaimReason::SourceLimit,
        false,
    ));
    let remaining = deadline - (rig.now - started);
    assert!(pick!(rig.tick(remaining - Duration::from_millis(1)), Claim).is_empty());
    assert!(fetched(&rig.events()).is_empty());
    assert!(pick!(rig.tick(Duration::from_millis(1)), Claim).is_empty());
    let events = rig.events();
    let results = fetched(&events);
    assert_eq!(results.len(), 1);
    assert!(results[0].1.is_err());
    assert!(rig.engine.requester.is_empty());
}

// ---- exports ----------------------------------------------------------------

pub(super) fn export_request(id: u64, reserved: u64) -> In {
    In::ExportRequest(wire::ExportRequest {
        export_id: id,
        claim_id: 1,
        tier: wire::Tier::Gpu as i32,
        key: Some(super::super::codec::key_to(key(192, 2))),
        begin_tokens: 0,
        want_row: true,
        pin_deadline_ms: 250,
        session_incarnation: SESSION,
        reserved_bytes: reserved,
        channel: None,
        ..Default::default()
    })
}

pub(super) fn chain() -> ExportChain {
    ExportChain {
        nodes: vec![
            ExportNode {
                key: key(128, 1),
                pages: vec![PageId(20), PageId(21)],
            },
            ExportNode {
                key: key(192, 2),
                pages: vec![PageId(22)],
            },
        ],
        skip_pages: 0,
        row: Some(StateSlotId(5)),
    }
}

pub(super) fn gpu_with_chain(rig: &Rig) {
    let mut copier = rig.copier.lock().unwrap();
    for p in 20..23u32 {
        copier.gpu_pages.insert(p, pattern(p as u8, PAGE as usize));
    }
    copier.gpu_rows.insert(5, pattern(77, ROW as usize));
}

fn pinned_export(rig: &mut Rig) -> u64 {
    gpu_with_chain(rig);
    rig.send(export_request(40, 3 * PAGE + ROW));
    let events = rig.events();
    assert!(
        matches!(&events[..], [PoolEvent::Export(r)] if r.export == ExportId(40) && r.reserved_bytes == 3 * PAGE + ROW)
    );
    let out = rig.command(Command::ExportPinned {
        export: ExportId(40),
        chain: chain(),
    });
    let ack = pick!(out, PinAck);
    assert_eq!(ack[0].result, wire::PinResult::Pinned as i32);
    assert_eq!(
        (ack[0].total_pages, ack[0].row, ack[0].records.len()),
        (3, true, 2)
    );
    assert_eq!(
        ack[0].records[1].key,
        Some(super::super::codec::key_to(key(192, 2)))
    );
    40
}

fn fill_all(id: u64) -> In {
    In::ExportFill(wire::ExportFill {
        export_id: id,
        windows: vec![
            window(0, 0, 0, 2, false, 0),
            window(1, 1, 0, 1, true, 2 * PAGE),
        ],
    })
}

#[test]
fn a_gpu_export_fills_windows_by_d2h_reports_hashes_after_the_drain_and_releases_the_pin_once() {
    let mut rig = Rig::new();
    rig.open();
    let id = pinned_export(&mut rig);
    let out = rig.send(fill_all(id));
    assert!(
        pick!(out, WindowFilled).is_empty(),
        "nothing is reported before the copy drained"
    );
    assert!(
        rig.copier
            .lock()
            .unwrap()
            .submitted
            .iter()
            .all(|j| j.direction == Direction::Out)
    );
    let out = rig.drain_copies();
    let filled = pick!(out, WindowFilled);
    assert_eq!(filled.len(), 2);
    assert!(
        filled
            .iter()
            .all(|f| f.status == wire::FillStatus::FillOk as i32)
    );
    let bytes = rig.regions.staging.bytes(0, 2 * PAGE).unwrap();
    assert_eq!(filled[0].page_xxh3[0], checksum(&bytes[..PAGE as usize]));
    assert_eq!(bytes[..PAGE as usize], pattern(20, PAGE as usize)[..]);
    assert!(filled[1].has_row);
    assert_eq!(filled[1].row_xxh3, checksum(&pattern(77, ROW as usize)));
    let ends = pick!(out, ExportEnd);
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].status, wire::ExportEndStatus::ExportComplete as i32);
    assert_eq!(ends[0].fence, wire::FenceOutcome::Drained as i32);
    let released: Vec<_> = rig
        .events()
        .into_iter()
        .filter(|e| matches!(e, PoolEvent::ExportEnded { .. }))
        .collect();
    assert_eq!(released.len(), 1);
    // A duplicate cancel afterwards is answered, and releases nothing more.
    let out = rig.send(In::ExportCancel(wire::ExportCancel {
        export_id: id,
        reason: 0,
    }));
    assert_eq!(
        pick!(out, ExportEnd)[0].fence,
        wire::FenceOutcome::Drained as i32
    );
    assert!(rig.events().is_empty());
}

#[test]
fn a_cancel_with_the_copy_in_flight_waits_for_the_drain_before_the_pin_is_released() {
    let mut rig = Rig::new();
    rig.open();
    let id = pinned_export(&mut rig);
    rig.send(fill_all(id));
    let out = rig.send(In::ExportCancel(wire::ExportCancel {
        export_id: id,
        reason: 0,
    }));
    assert!(pick!(out, ExportEnd).is_empty());
    assert!(rig.events().is_empty(), "a started DMA is never revoked");
    let out = rig.drain_copies();
    assert_eq!(
        pick!(out, ExportEnd)[0].status,
        wire::ExportEndStatus::ExportReleased as i32
    );
    assert_eq!(rig.events().len(), 1);
}

#[test]
fn an_answer_after_the_deadline_is_discarded_and_its_pin_released() {
    let mut rig = Rig::new();
    rig.open();
    gpu_with_chain(&rig);
    rig.send(export_request(41, 3 * PAGE + ROW));
    rig.events();
    let out = rig.tick(Duration::from_millis(251));
    let ack = pick!(out, PinAck);
    assert_eq!(ack[0].result, wire::PinResult::Refused as i32);
    let out = rig.command(Command::ExportPinned {
        export: ExportId(41),
        chain: chain(),
    });
    assert!(
        pick!(out, PinAck).is_empty(),
        "exactly one PinAck per request"
    );
    assert!(
        matches!(&rig.events()[..], [PoolEvent::ExportEnded { export }] if *export == ExportId(41))
    );
}

#[test]
fn a_chain_larger_than_the_reserved_staging_is_refused_and_unpinned() {
    let mut rig = Rig::new();
    rig.open();
    rig.send(export_request(42, 2 * PAGE));
    rig.events();
    let out = rig.command(Command::ExportPinned {
        export: ExportId(42),
        chain: chain(),
    });
    let ack = pick!(out, PinAck);
    assert_eq!(ack[0].refusal, wire::ClaimReason::NoWindows as i32);
    assert_eq!(rig.events().len(), 1, "the worker is told to unpin");
}

#[test]
fn a_worker_refusal_is_relayed_and_a_stale_session_or_third_export_is_refused() {
    let mut rig = Rig::new();
    rig.open();
    rig.send(export_request(43, 1 << 20));
    rig.events();
    let out = rig.command(Command::ExportRefused {
        export: ExportId(43),
        reason: crate::ExportRefusal::NoHeadroom,
    });
    assert_eq!(
        pick!(out, PinAck)[0].refusal,
        wire::ClaimReason::NoHeadroom as i32
    );
    let mut stale = export_request(44, 1 << 20);
    if let In::ExportRequest(r) = &mut stale {
        r.session_incarnation = 999;
    }
    assert_eq!(
        pick!(rig.send(stale), PinAck)[0].refusal,
        wire::ClaimReason::StaleSource as i32
    );
    rig.send(export_request(45, 1 << 20));
    rig.send(export_request(46, 1 << 20));
    let out = rig.send(export_request(47, 1 << 20));
    assert_eq!(
        pick!(out, PinAck)[0].refusal,
        wire::ClaimReason::SourceBusy as i32
    );
}

#[test]
fn a_cancel_before_the_worker_took_the_request_withdraws_it() {
    let mut rig = Rig::new();
    rig.open();
    rig.send(export_request(48, 1 << 20));
    let out = rig.send(In::ExportCancel(wire::ExportCancel {
        export_id: 48,
        reason: 0,
    }));
    assert_eq!(
        pick!(out, PinAck)[0].refusal,
        wire::ClaimReason::Cancelled as i32
    );
    assert!(rig.events().is_empty(), "the worker never sees it");
}

#[test]
fn losing_the_session_with_an_export_running_releases_the_pin_only_after_the_drain() {
    let mut rig = Rig::new();
    rig.open();
    let id = pinned_export(&mut rig);
    rig.send(fill_all(id));
    rig.engine.session_lost(rig.now);
    assert!(
        rig.events()
            .iter()
            .all(|e| !matches!(e, PoolEvent::ExportEnded { .. }))
    );
    rig.copier.lock().unwrap().complete_all();
    rig.engine.tick(rig.now);
    assert_eq!(
        rig.events()
            .iter()
            .filter(|e| matches!(e, PoolEvent::ExportEnded { .. }))
            .count(),
        1
    );
    let out = rig.open_session(Some(rig.regions.clone()));
    let ends = pick!(out, ExportEnd);
    assert_eq!(ends.len(), 1, "the end is reported on the next session");
    assert_eq!(ends[0].fence, wire::FenceOutcome::Drained as i32);
}

#[test]
fn a_copy_that_never_drains_quarantines_the_pin_and_the_regions() {
    let mut rig = Rig::new();
    rig.open();
    let id = pinned_export(&mut rig);
    rig.send(fill_all(id));
    rig.copier.lock().unwrap().stall = true;
    rig.engine.session_lost(rig.now);
    rig.tick(crate::CLOSE_TIMEOUT + Duration::from_secs(1));
    assert!(rig.engine.quarantined());
    assert!(rig.shared.quarantined());
    assert!(
        rig.events()
            .iter()
            .all(|e| !matches!(e, PoolEvent::ExportEnded { .. })),
        "a pin whose copy cannot be drained is never released"
    );
    assert!(matches!(rig.shared.status(), PoolStatus::Disabled(_)));
}

#[test]
fn a_failed_d2h_fails_the_fill_and_ends_the_export_after_the_drain() {
    let mut rig = Rig::new();
    rig.open();
    let id = pinned_export(&mut rig);
    rig.copier.lock().unwrap().fail_next = true;
    rig.send(fill_all(id));
    let out = rig.drain_copies();
    assert!(
        pick!(out, WindowFilled)
            .iter()
            .any(|f| f.status == wire::FillStatus::FillFailed as i32)
    );
    assert_eq!(
        pick!(out, ExportEnd)[0].status,
        wire::ExportEndStatus::ExportFailed as i32
    );
    assert_eq!(rig.events().len(), 1);
}

// ---- host-tier exports ------------------------------------------------------

mod host {
    use super::*;
    use crate::pool::{Checksums, HostChain, HostPin, HostRecord, HostWindow};

    pub struct Pin {
        chain: HostChain,
        /// Checksums the record was committed with.
        persisted: Vec<u64>,
        data: Vec<u8>,
        pub dropped: Arc<Mutex<bool>>,
    }

    impl Drop for Pin {
        fn drop(&mut self) {
            *self.dropped.lock().unwrap() = true;
        }
    }

    impl HostPin for Pin {
        fn chain(&self) -> &HostChain {
            &self.chain
        }
        fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums> {
            let from = window.first_page as usize * PAGE as usize;
            dst[..window.page_count as usize * PAGE as usize].copy_from_slice(
                &self.data[from..from + window.page_count as usize * PAGE as usize],
            );
            Ok(Checksums {
                pages: self.persisted[window.first_page as usize..][..window.page_count as usize]
                    .to_vec(),
                row: None,
            })
        }
    }

    /// A pinned two-page chain whose bytes may have rotted since its checksums
    /// were committed.
    pub fn pin(corrupt: bool, dropped: Arc<Mutex<bool>>) -> Box<dyn HostPin> {
        let data = pattern(3, 2 * PAGE as usize);
        let persisted = (0..2)
            .map(|i| checksum(&data[i * PAGE as usize..(i + 1) * PAGE as usize]))
            .collect();
        let mut stored = data;
        if corrupt {
            stored[10] ^= 1;
        }
        Box::new(Pin {
            chain: HostChain {
                records: vec![HostRecord {
                    key: key(192, 2),
                    first_page: 0,
                    pages: 2,
                    row: false,
                }],
                total_pages: 2,
                row: false,
            },
            persisted,
            data: stored,
            dropped,
        })
    }
}

#[test]
fn a_host_export_is_answered_by_the_store_and_serves_the_persisted_checksums() {
    for corrupt in [false, true] {
        let mut rig = Rig::new();
        rig.open();
        let mut request = export_request(50, 0);
        if let In::ExportRequest(r) = &mut request {
            r.tier = wire::Tier::Dram as i32;
            r.want_row = false;
        }
        assert!(
            pick!(rig.send(request), PinAck).is_empty(),
            "the store answers"
        );
        let events = rig.events();
        assert!(matches!(&events[..], [PoolEvent::Export(r)] if r.tier == Tier::Dram));
        let dropped = Arc::new(Mutex::new(false));
        let out = rig.command(Command::HostPinned {
            export: ExportId(50),
            pin: host::pin(corrupt, dropped.clone()),
        });
        let ack = pick!(out, PinAck);
        assert_eq!(ack[0].result, wire::PinResult::Pinned as i32);
        assert_eq!(ack[0].total_pages, 2);
        rig.send(In::ExportFill(wire::ExportFill {
            export_id: 50,
            windows: vec![window(0, 0, 0, 2, false, 0)],
        }));
        let mut filled = Vec::new();
        let mut ended = Vec::new();
        for _ in 0..200 {
            std::thread::sleep(Duration::from_millis(5));
            let out = rig.tick(Duration::ZERO);
            filled.extend(pick!(out, WindowFilled));
            ended.extend(pick!(out, ExportEnd));
            if !ended.is_empty() {
                break;
            }
        }
        assert_eq!(filled.len(), 1, "the copy thread finished");
        let served = pattern(3, 2 * PAGE as usize);
        // The checksums are those committed with the record, whatever the bytes are now.
        assert_eq!(filled[0].page_xxh3[0], checksum(&served[..PAGE as usize]));
        let in_window = rig.regions.staging.bytes(0, PAGE).unwrap();
        assert_eq!(checksum(in_window) == filled[0].page_xxh3[0], !corrupt);
        assert_eq!(ended[0].fence, wire::FenceOutcome::Drained as i32);
        assert!(
            *dropped.lock().unwrap(),
            "the pin is released with the export"
        );
        // A host export never produces a worker event.
        assert!(
            rig.events()
                .iter()
                .all(|e| !matches!(e, PoolEvent::ExportEnded { .. }))
        );
    }
}

#[test]
fn a_host_pin_the_export_cannot_use_is_dropped() {
    let mut rig = Rig::new();
    rig.open();
    let mut request = export_request(51, 0);
    if let In::ExportRequest(r) = &mut request {
        r.tier = wire::Tier::Nvme as i32;
        r.want_row = false;
        r.reserved_bytes = PAGE;
    }
    rig.send(request);
    rig.events();
    let dropped = Arc::new(Mutex::new(false));
    let out = rig.command(Command::HostPinned {
        export: ExportId(51),
        pin: host::pin(false, dropped.clone()),
    });
    assert_eq!(
        pick!(out, PinAck)[0].refusal,
        wire::ClaimReason::NoWindows as i32
    );
    assert!(*dropped.lock().unwrap());
    // A pin for an export nobody waits for is dropped too.
    let dropped = Arc::new(Mutex::new(false));
    rig.command(Command::HostPinned {
        export: ExportId(99),
        pin: host::pin(false, dropped.clone()),
    });
    assert!(*dropped.lock().unwrap());
}

// ---- puts ---------------------------------------------------------------------

pub(super) use crate::testing::RecordBytes as Bytes;

pub(super) fn host_put(parent: Option<PrefixKey>) -> crate::pool::PutRequest {
    let mut data: Vec<_> = (20..23)
        .flat_map(|seed| pattern(seed, PAGE as usize))
        .collect();
    data.extend(pattern(77, ROW as usize));
    let sums = data[..3 * PAGE as usize]
        .chunks_exact(PAGE as usize)
        .map(checksum)
        .collect();
    let row = checksum(&data[3 * PAGE as usize..]);
    crate::pool::PutRequest {
        key: key(192, 2),
        parent,
        bucket: [4; 16],
        chain_pages: 3,
        tail_pages: 3,
        row: true,
        source: crate::pool::PutSource::Host(Box::new(Bytes {
            data,
            sums,
            row,
            page_bytes: PAGE as usize,
            row_bytes: ROW as usize,
        })),
    }
}

/// The record length the manager expects for `pages` pages and a row.
pub(super) fn record_len(pages: u64) -> u64 {
    4096 + 4096 + pages * PAGE + ROW
}

fn accept(put: u64, first_page: u64, record_bytes: u64) -> In {
    In::PutAccept(wire::PutAccept {
        put_id: put,
        first_page,
        has_parent: false,
        parent: None,
        record_bytes,
        channel: None,
        ..Default::default()
    })
}

fn grant_put(put: u64, spec: wire::WindowSpec) -> In {
    In::PutGrant(wire::PutGrant {
        put_id: put,
        window: Some(spec),
    })
}

pub(super) fn puts(events: &[PoolEvent]) -> Vec<(u64, String)> {
    events
        .iter()
        .filter_map(|e| match e {
            PoolEvent::Put { id, outcome } => Some((*id, format!("{outcome:?}"))),
            _ => None,
        })
        .collect()
}

#[test]
fn a_refusal_ends_the_put_and_present_is_a_success() {
    let mut rig = Rig::new();
    rig.open();
    rig.command(Command::Put {
        id: 1,
        request: host_put(None),
    });
    rig.send(In::PutRefused(wire::PutRefused {
        put_id: 1,
        reason: wire::Reason::Present as i32,
        detail: String::new(),
    }));
    assert_eq!(puts(&rig.events()), [(1, "Present".to_string())]);
}

#[test]
fn bytes_a_store_supplies_are_committed_with_the_checksums_it_persists() {
    let mut rig = Rig::new();
    rig.open();
    let data = pattern(9, 3 * PAGE as usize + ROW as usize);
    // The persisted sums differ from the bytes' own: the writer passes them on.
    let source = Bytes {
        data,
        sums: vec![101, 102, 103],
        row: 104,
        page_bytes: PAGE as usize,
        row_bytes: ROW as usize,
    };
    let mut request = host_put(None);
    request.source = crate::pool::PutSource::Host(Box::new(source));
    rig.command(Command::Put { id: 4, request });
    rig.send(accept(4, 0, record_len(3)));
    rig.send(grant_put(4, window(0, 0, 0, 3, true, 0)));
    let mut out = Vec::new();
    for _ in 0..200 {
        std::thread::sleep(Duration::from_millis(5));
        out.extend(rig.tick(Duration::ZERO));
        if !pick!(out, PutCommit).is_empty() {
            break;
        }
    }
    let filled = pick!(out, PutFilled);
    assert_eq!(filled[0].page_xxh3, [101, 102, 103]);
    assert_eq!(filled[0].row_xxh3, 104);
    let commit = pick!(out, PutCommit);
    let header = crate::record::Header::decode(&commit[0].metadata[..4096]).unwrap();
    assert_eq!(header.row_checksum, 104);
    assert_eq!(
        u64::from_le_bytes(commit[0].metadata[4096 + 8..4096 + 16].try_into().unwrap()),
        102
    );
}

#[test]
fn a_record_the_manager_sizes_differently_or_a_window_out_of_order_fails_the_put() {
    let mut rig = Rig::new();
    rig.open();
    rig.command(Command::Put {
        id: 5,
        request: host_put(None),
    });
    let out = rig.send(accept(5, 0, record_len(3) + 4096));
    assert_eq!(pick!(out, PutCancel).len(), 1);
    let ended = puts(&rig.events());
    assert!(ended[0].1.starts_with("Failed"), "{ended:?}");
    rig.command(Command::Put {
        id: 6,
        request: host_put(None),
    });
    rig.send(accept(6, 0, record_len(3)));
    let out = rig.send(grant_put(6, window(1, 0, 0, 3, true, 0)));
    assert_eq!(pick!(out, PutCancel).len(), 1);
    assert!(rig.copier.lock().unwrap().submitted.is_empty());
    assert!(puts(&rig.events())[0].1.starts_with("Failed"));
}

// ---- expected records -------------------------------------------------------

#[test]
fn expected_records_follow_flooring_for_unaligned_splits_and_row_only_tails() {
    use super::requester::{ExpectedRecord, expected_records};
    let (n1, n2, n3) = (key(130, 1), key(250, 2), key(251, 3));
    // B starts at 130 tokens (page 2, inside n1's last page) and ends at 250.
    let nodes = [
        plan::ChainNode {
            key: n1,
            tail_pages: 2,
        },
        plan::ChainNode {
            key: n2,
            tail_pages: 2,
        },
    ];
    let part = |begin, end, chain: &[PrefixKey], pages, row| plan::Part {
        level: Level::HostDram,
        source: plan::SourceId(1),
        begin_tokens: begin,
        end,
        first_page: 0,
        dest_page: 0,
        pages,
        row,
        chain: chain.to_vec(),
    };
    // Records start mid-way: n1 holds pages 0..2, n2 pages 2..3 (250/64 = 3, tail 2 -> start 1).
    let got = expected_records(64, &part(130, n2, &[n1, n2], 1, true), &nodes).unwrap();
    assert_eq!(
        got,
        [ExpectedRecord {
            key: n2,
            first_page: 1,
            pages: 1,
            row: true
        }],
        "n1 ends at page 2 = the start page, so it supplies nothing"
    );
    // A row-only tail: A ends mid-page and the endpoint lies in the same page.
    let tail = [plan::ChainNode {
        key: n3,
        tail_pages: 0,
    }];
    let got = expected_records(64, &part(250, n3, &[n3], 0, true), &tail).unwrap();
    assert_eq!(
        got,
        [ExpectedRecord {
            key: n3,
            first_page: 0,
            pages: 0,
            row: true
        }]
    );
    // The pages must add up to what the plan says.
    assert!(expected_records(64, &part(0, n2, &[n1, n2], 99, true), &nodes).is_none());
}

#[test]
fn mutations_after_a_snapshot_cut_follow_its_last_chunk() {
    let mut rig = Rig::new();
    rig.open();
    let entry = IndexEntry {
        key: key(64, 1),
        parent: None,
        bucket: [0; 16],
        tail_pages: 1,
        row: true,
        tier: Tier::Gpu,
        bytes: PAGE,
    };
    rig.announce.saved(entry.clone());
    let mut out = rig.send(In::ResyncRequired(wire::ResyncRequired {
        reason: wire::ResyncReason::ResyncStateLost as i32,
    }));
    rig.announce.evicted(entry.key, entry.tier, None);
    for _ in 0..3 {
        rig.engine.pump(rig.now);
        out.extend(rig.engine.take_out());
    }
    let end = out
        .iter()
        .position(|m| matches!(m.m, Some(Out::ResyncEnd(_))))
        .unwrap();
    let notice = out
        .iter()
        .position(|m| matches!(m.m, Some(Out::Notices(_))))
        .unwrap();
    assert!(end < notice);
    assert_eq!(pick!(out, ResyncChunk)[0].entries.len(), 1);
}

#[test]
fn pool_names_salt_scope_and_instance_ids_and_cannot_change_on_reconnect() {
    let mut a = (*settings(2)).clone();
    let mut b = a.clone();
    a.bind_pool("one").unwrap();
    b.bind_pool("two").unwrap();
    assert_ne!(a.scope_id, b.scope_id);
    assert_ne!(a.instance_id, b.instance_id);
    let bound = (a.scope_id, a.instance_id);
    a.bind_pool("one").unwrap();
    assert_eq!((a.scope_id, a.instance_id), bound);
    assert!(a.bind_pool("two").is_err());
    assert_eq!((a.scope_id, a.instance_id), bound);
}
