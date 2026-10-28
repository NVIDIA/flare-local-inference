//! Claims, exports, windows and fences (PROTOCOL.md sections 5 to 8).
//!
//! Every claim is one record, changed only by the actor, so "one authoritative
//! transition" is a plain state check. Windows are carved from the staging region
//! and are reclaimed only when every party that could have touched them has
//! drained, or is confirmed gone.

use super::{
    Checked, Violation,
    core::{ChanOwner, Core, Nonce, ProcKey, Session},
    index::{InstanceId, SourceId, walk_chain},
    key_from, key_to,
    mp::{Leg, MpSide, MpState, join_legs, root, stop_legs},
    tier_from, tier_to,
    windows::{self, PlannedWindow, RecordShape},
};
use crate::{
    Key,
    arena::Extent,
    data::channel,
    proto::v2::{self, server_message::M},
    store::GetAnswer,
};
use flare_selene::shared::Tier;
use flare_selene::shared::{SliceSums, frame::Shape, transfer_timeout};
use std::time::{Duration, Instant};

pub type ClaimKey = (InstanceId, Nonce, u64);

/// One party's drain obligation for a claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Party {
    /// It may still be accessing memory and has not said otherwise.
    Pending,
    /// No window is held; a put may receive another grant.
    Idle,
    Drained,
    Quarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Proc(ProcKey),
    Leg(u64),
    Store,
}

/// A drain obligation; channel sources also owe an export release without accessing memory.
pub struct Participant {
    pub owner: Owner,
    pub source: bool,
    pub memory: bool,
    pub state: Party,
}

pub fn party_state(parties: &[Participant], owner: Owner) -> Party {
    parties
        .iter()
        .find(|p| p.owner == owner)
        .map_or(Party::Drained, |p| p.state)
}

pub fn attest(parties: &mut [Participant], owner: Owner, state: Party, exited: bool) {
    for p in parties.iter_mut().filter(|p| p.owner == owner) {
        if matches!(p.state, Party::Pending | Party::Idle) || exited {
            p.state = state;
        }
    }
}

pub fn activate(parties: &mut [Participant], source: Option<bool>) {
    for p in parties
        .iter_mut()
        .filter(|p| p.memory && source.is_none_or(|s| s == p.source))
    {
        if matches!(p.state, Party::Drained | Party::Idle) && matches!(p.owner, Owner::Proc(_)) {
            p.state = Party::Pending;
        }
    }
}

pub fn participants(
    requester: Option<ProcKey>,
    source: Option<ProcKey>,
    legs: &[Leg],
    sides: impl IntoIterator<Item = (bool, Vec<ProcKey>)>,
) -> Vec<Participant> {
    let mut parties = Vec::new();
    for (source, proc) in [(false, requester), (true, source)] {
        if let Some(proc) = proc {
            let memory = root(legs, source).is_none();
            if source || memory {
                parties.push(Participant {
                    owner: Owner::Proc(proc),
                    source,
                    memory,
                    state: if source {
                        Party::Pending
                    } else {
                        Party::Drained
                    },
                });
            }
        }
    }
    for (source, peers) in sides {
        parties.extend(peers.into_iter().map(|proc| Participant {
            owner: Owner::Proc(proc),
            source,
            memory: true,
            state: Party::Drained,
        }));
    }
    parties.extend(legs.iter().map(|leg| Participant {
        owner: Owner::Leg(leg.ch.transfer),
        source: leg.source,
        memory: true,
        state: Party::Pending,
    }));
    parties
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Planned, not yet handed to anyone.
    Unissued,
    /// The source may be writing it.
    SourceWrite,
    /// The requester may be reading it.
    RequesterRead,
    Free,
}

pub struct Win {
    pub plan: PlannedWindow,
    pub region: i32,
    pub offset: u64,
    pub phase: Phase,
    /// A window allocated on its own (streaming claims).
    pub extent: Option<std::sync::Arc<Extent>>,
    /// Readers that must finish with the window before it is free, one bit per accessor:
    /// bit 0 is the requester's session (or its own leg), bit `r` the leg of rank `r`.
    pub readers: u64,
    /// Writers that must finish with the window before it is whole, one bit per accessor like
    /// `readers`: bit 0 is the source's session (rank 0 and its shared peers) or its own leg.
    pub writers: u64,
    /// The slice checksums a multi-process source's ranks stated, by rank.
    pub stated: Vec<Option<SliceSums>>,
    /// The window's contents have been handed to the pass (or delivered): a repeat is ignored.
    pub arrived: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportState {
    /// The claim has no source export (a refusal before any request).
    None,
    Requested,
    Pinned,
    Ended,
}

pub struct Claim {
    pub key: ClaimKey,
    pub requester_sid: u64,
    pub source_proc: Option<ProcKey>,
    pub source_sid: u64,
    pub endpoint: Key,
    pub want_row: bool,
    pub state: v2::ClaimState,
    pub reason: v2::ClaimReason,
    pub deadline: Instant,
    pub last_progress: Instant,
    pub pin_deadline: Instant,
    pub export_id: u64,
    pub export: ExportState,
    pub records: Vec<v2::Record>,
    pub total_pages: u64,
    pub row: bool,
    pub retained: bool,
    pub reserved_bytes: u64,
    pub block: Option<Extent>,
    pub windows: Vec<Win>,
    pub parties: Vec<Participant>,
    /// The requester reported exact coverage; completion still waits for every drain.
    pub verdict_latched: bool,
    pub granted_at: Option<Instant>,
    pub page_bytes: u64,
    pub row_bytes: u64,
    /// Set when the manager's own store is the source.
    pub manager: Option<ManagerGet>,
    /// Every requester/source channel, including rank zero and its peers.
    pub legs: Vec<Leg>,
    /// What multi-process instances add: peer participants, their legs, the assembly pass.
    pub mp: MpState,
}

/// A claim's data-channel transfer. The manager's handler is the only party that
/// reads the claim's windows, so its exit is what lets them go back to the store.
pub struct Channel {
    pub transfer: u64,
    /// What the requester presents to be served this transfer.
    pub token: [u8; flare_selene::shared::frame::TOKEN_BYTES],
}

/// What a claim on the manager's store tracks beyond the common state.
pub struct ManagerGet {
    /// Persisted page and row checksums by record, from the store's metadata.
    pub checks: std::collections::HashMap<u32, (Vec<u64>, u64)>,
    pub remaining: usize,
    pub pending: usize,
    /// The chain page at which each record's tail starts.
    pub starts: Vec<u64>,
}

impl Claim {
    fn new(
        key: ClaimKey,
        sid: u64,
        m: &v2::Claim,
        mp: MpState,
        limits: &flare_selene::shared::Limits,
        total: u64,
    ) -> Self {
        let now = Instant::now();
        let deadline_ms = if m.deadline_ms == 0 {
            limits.max_claim_deadline_ms
        } else {
            m.deadline_ms.min(limits.max_claim_deadline_ms)
        };
        Claim {
            key,
            requester_sid: sid,
            source_proc: None,
            source_sid: 0,
            endpoint: key_from(m.key.as_ref()).unwrap_or_default(),
            want_row: m.want_row,
            state: v2::ClaimState::Requested,
            reason: v2::ClaimReason::Unspecified,
            deadline: now + transfer_timeout(Duration::from_millis(u64::from(deadline_ms)), total),
            last_progress: now,
            pin_deadline: now + limits.pin_deadline(),
            export_id: 0,
            export: ExportState::None,
            records: Vec::new(),
            total_pages: 0,
            row: false,
            retained: false,
            reserved_bytes: 0,
            block: None,
            windows: Vec::new(),
            parties: Vec::new(),
            verdict_latched: false,
            granted_at: None,
            page_bytes: mp.req.layout.page_record_bytes(),
            row_bytes: mp.req.layout.row_record_bytes(),
            manager: None,
            legs: Vec::new(),
            mp,
        }
    }

    pub(super) fn channel(&self) -> Option<&Channel> {
        root(&self.legs, false)
    }
    fn source_channel(&self) -> Option<&Channel> {
        root(&self.legs, true)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.state, v2::ClaimState::Copied | v2::ClaimState::Failed)
    }

    fn holds_export_slot(&self, source_exited: bool) -> bool {
        matches!(self.export, ExportState::Requested | ExportState::Pinned)
            || (!source_exited
                && self.source_proc.is_some_and(|proc| {
                    party_state(&self.parties, Owner::Proc(proc)) == Party::Quarantined
                }))
    }

    fn unreleased(&self) -> usize {
        self.windows
            .iter()
            .filter(|window| matches!(window.phase, Phase::SourceWrite | Phase::RequesterRead))
            .count()
    }

    /// Some peer rank's leg has not finished reading a window the requester has.
    pub(super) fn legs_pending(&self) -> bool {
        self.windows
            .iter()
            .any(|w| w.phase == Phase::RequesterRead && w.readers & !1 != 0)
    }

    fn all_windows_done(&self) -> bool {
        self.windows
            .iter()
            .all(|window| window.phase == Phase::Free)
    }

    pub(super) fn spec(&self, index: usize) -> v2::WindowSpec {
        let window = &self.windows[index];
        v2::WindowSpec {
            index: window.plan.index,
            region: window.region,
            offset: window.offset,
            length: window.plan.length,
            record: window.plan.record,
            first_page: window.plan.first_page,
            page_count: window.plan.page_count,
            row: window.plan.row,
        }
    }
}

fn refusal_reason(value: i32) -> v2::ClaimReason {
    use v2::ClaimReason as R;
    match R::try_from(value) {
        Ok(
            reason @ (R::NotFound
            | R::ChainGap
            | R::NoHeadroom
            | R::SourceBusy
            | R::InvalidClaim
            | R::Cancelled
            | R::StaleSource
            | R::NoWindows),
        ) => reason,
        _ => R::SourceBusy,
    }
}

impl Core {
    // ------------------------------------------------------------------ helpers

    fn session_key(&self, sid: u64) -> Option<(InstanceId, Nonce)> {
        self.sessions
            .get(&sid)
            .map(|session| (session.instance, session.nonce))
    }

    fn proc_exited_flag(&self, key: &ProcKey) -> bool {
        self.procs.get(key).is_none_or(|proc| proc.exited)
    }

    fn exports_outstanding(&self, instance: InstanceId) -> usize {
        self.claims
            .values()
            .filter(|claim| {
                claim
                    .source_proc
                    .is_some_and(|(source, _)| source == instance)
                    && claim.holds_export_slot(
                        claim
                            .source_proc
                            .as_ref()
                            .is_some_and(|proc| self.proc_exited_flag(proc)),
                    )
            })
            .count()
    }

    fn claims_in_flight(&self, requester: (InstanceId, Nonce)) -> usize {
        self.claims
            .values()
            .filter(|claim| (claim.key.0, claim.key.1) == requester && !claim.is_terminal())
            .count()
    }

    /// Bytes of windows no one may reuse yet because a party that held them has
    /// reported it cannot prove it drained, or has gone without saying.
    pub(super) fn quarantined_bytes(&self) -> u64 {
        self.claims
            .values()
            .filter(|c| c.is_terminal())
            .map(|claim| {
                claim
                    .windows
                    .iter()
                    .filter(|w| {
                        let source = match w.phase {
                            Phase::SourceWrite => true,
                            Phase::RequesterRead => false,
                            _ => return false,
                        };
                        self.holders(&claim.parties, Some(source), claim.mp.passes)
                            == Party::Quarantined
                    })
                    .map(|w| w.plan.length)
                    .sum::<u64>()
            })
            .sum()
    }

    /// One answer for reclaim, forgetting and quarantine accounting.
    pub(super) fn holders(
        &self,
        parties: &[Participant],
        source: Option<bool>,
        passes: u32,
    ) -> Party {
        let mut result = if passes == 0 {
            Party::Drained
        } else {
            Party::Pending
        };
        for p in parties
            .iter()
            .filter(|p| source.is_none_or(|s| p.memory && s == p.source))
        {
            let state = match p.owner {
                Owner::Proc(proc) if self.proc_exited_flag(&proc) => Party::Drained,
                Owner::Proc(proc)
                    if p.state == Party::Pending
                        && !self
                            .sessions
                            .values()
                            .any(|s| (s.instance, s.nonce) == proc)
                        && !self.agents.values().any(|a| a.proc() == proc) =>
                {
                    Party::Quarantined
                }
                Owner::Leg(_) if p.state == Party::Pending => Party::Quarantined,
                _ if p.state == Party::Idle => Party::Drained,
                _ => p.state,
            };
            if state == Party::Quarantined {
                return state;
            }
            if state == Party::Pending {
                result = state;
            }
        }
        result
    }

    fn terminal_message(&self, claim: &Claim) -> v2::ClaimTerminal {
        v2::ClaimTerminal {
            claim_id: claim.key.2,
            state: claim.state as i32,
            reason: claim.reason as i32,
            fence_required: claim.granted_at.is_some(),
        }
    }

    // ------------------------------------------------------------------ claim

    pub(super) fn claim(&mut self, sid: u64, m: v2::Claim) -> Checked<()> {
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        let (instance, nonce) = (session.instance, session.nonce);
        let (scope_id, shm, claim_slots) = (session.scope, session.shm, session.claim_slots);
        let key: ClaimKey = (instance, nonce, m.claim_id);
        let proc = self
            .procs
            .get_mut(&(instance, nonce))
            .ok_or_else(|| Violation::new("claim from an unknown process"))?;
        if m.claim_id == 0 || m.claim_id <= proc.claim_hw {
            // A replay or a reused id: refused without a record.
            self.send(
                sid,
                M::ClaimTerminal(v2::ClaimTerminal {
                    claim_id: m.claim_id,
                    state: v2::ClaimState::Failed as i32,
                    reason: v2::ClaimReason::InvalidClaim as i32,
                    fence_required: false,
                }),
            );
            return Ok(());
        }
        proc.claim_hw = m.claim_id;

        let refuse = |core: &mut Core, reason: v2::ClaimReason| {
            core.refused_claim(sid, key, &m, reason);
            Ok(())
        };
        if !self.ready() {
            return refuse(self, v2::ClaimReason::NotReady);
        }
        if self.cfg.held.load(std::sync::atomic::Ordering::Acquire) {
            return refuse(self, v2::ClaimReason::RegionsHeld);
        }
        if !self.ranks_ready(sid) {
            return refuse(self, v2::ClaimReason::RanksDetached);
        }
        if claim_slots == 0 || self.claims_in_flight((instance, nonce)) >= claim_slots as usize {
            return refuse(self, v2::ClaimReason::ClaimLimit);
        }
        let Ok(endpoint) = key_from(m.key.as_ref()) else {
            return refuse(self, v2::ClaimReason::InvalidClaim);
        };
        let tier = match tier_from(m.tier) {
            Ok(tier) => tier,
            Err(_) => return refuse(self, v2::ClaimReason::InvalidClaim),
        };
        let mode = v2::CopyMode::try_from(m.mode).unwrap_or(v2::CopyMode::Unspecified);
        if endpoint.consumed_tokens == 0
            || m.begin_tokens >= endpoint.consumed_tokens + u64::from(m.want_row)
            || mode == v2::CopyMode::Unspecified
            || (mode == v2::CopyMode::SharedWindows && !shm)
        {
            return refuse(self, v2::ClaimReason::InvalidClaim);
        }
        if mode == v2::CopyMode::DataChannel && self.cfg.channel.is_none() {
            // Claims carry no detail text; the manager's log says why.
            eprintln!(
                "selene: claim {} refused: this manager serves no data channel (it has no network listener)",
                m.claim_id
            );
            return refuse(self, v2::ClaimReason::InvalidClaim);
        }
        if m.begin_tokens > endpoint.consumed_tokens {
            return refuse(self, v2::ClaimReason::InvalidClaim);
        }

        // Resolve the source.
        let source_ref = m.source.clone().unwrap_or_default();
        let (source, source_sid, source_proc) = if source_ref.instance_id.is_empty() {
            (SourceId::Manager, 0, None)
        } else {
            let Ok(id) = InstanceId::try_from(source_ref.instance_id.as_slice()) else {
                return refuse(self, v2::ClaimReason::InvalidClaim);
            };
            if id == instance {
                return refuse(self, v2::ClaimReason::InvalidClaim);
            }
            let Some(entry) = self
                .instances
                .get(&id)
                .filter(|entry| entry.scope == scope_id)
            else {
                return refuse(self, v2::ClaimReason::UnknownSource);
            };
            let Some(current) = entry.current else {
                return refuse(self, v2::ClaimReason::UnknownSource);
            };
            if current != source_ref.session_incarnation {
                return refuse(self, v2::ClaimReason::StaleSource);
            }
            // A source or a requester without shared memory is relayed through the data
            // channel, which needs a manager that serves one.
            if !self.ranks_ready(current) {
                return refuse(self, v2::ClaimReason::RanksDetached);
            }
            let source_session = &self.sessions[&current];
            if (mode == v2::CopyMode::DataChannel || !source_session.shm)
                && self.cfg.channel.is_none()
            {
                return refuse(self, v2::ClaimReason::InvalidClaim);
            }
            (
                SourceId::Instance(id),
                current,
                Some((id, source_session.nonce)),
            )
        };
        let Some(state) = self.scopes.get(&scope_id) else {
            return refuse(self, v2::ClaimReason::UnknownSource);
        };
        let Some(entry) = state.index.get(source, endpoint, tier).cloned() else {
            return refuse(self, v2::ClaimReason::UnknownSource);
        };
        if m.want_row && !entry.row {
            return refuse(self, v2::ClaimReason::InvalidClaim);
        }
        let geometry = state.valid.descriptor.clone();
        if let SourceId::Instance(id) = source {
            let slots = self.sessions[&source_sid].export_slots as usize;
            if self.exports_outstanding(id) >= slots {
                return refuse(self, v2::ClaimReason::SourceLimit);
            }
        }
        // The windows the claim will need, from the index's view of the chain.
        let start_page = m.begin_tokens / geometry.page_tokens;
        let Some(walk) = walk_chain(
            &state.index,
            source,
            &entry,
            geometry.page_tokens,
            start_page,
            self.cfg.limits.max_chain_nodes as usize,
        ) else {
            return refuse(self, v2::ClaimReason::ChainGap);
        };
        let shapes = shapes_of(&walk, m.want_row);
        let Some(plan) = windows::plan(
            &shapes,
            geometry.page_bytes(),
            geometry.row_bytes(),
            self.cfg.limits.max_window_bytes,
        ) else {
            return refuse(self, v2::ClaimReason::InvalidClaim);
        };
        let total = windows::total_bytes(&plan);
        if source == SourceId::Manager {
            return self.manager_claim(sid, key, &m, walk, &geometry, mode);
        }

        // Retained staging: required for a GPU export, preferred for host tiers.
        let mut block = None;
        let mut retained = false;
        match usize::try_from(total)
            .ok()
            .and_then(|total| self.staging.allocate(total))
        {
            Some(extent) => {
                block = Some(extent);
                retained = true;
            }
            None if tier == Tier::Gpu => return refuse(self, v2::ClaimReason::NoWindows),
            None => {}
        }

        // The legs that cross the network: to a requester without shared windows, and
        // from a source without shared memory.
        let requester_slices = self.requester_slices(sid);
        let source_slices = self.requester_slices(source_sid);
        let requester_leg = if mode == v2::CopyMode::DataChannel {
            match self.open_leg(channel::Kind::Get, requester_slices, 0) {
                Ok(leg) => Some(leg),
                Err(_) => return refuse(self, v2::ClaimReason::NoWindows),
            }
        } else {
            None
        };
        let source_leg = if self.sessions[&source_sid].shm {
            None
        } else {
            match self.open_leg(channel::Kind::Put, source_slices, 0) {
                Ok(leg) => Some(leg),
                Err(_) => {
                    self.discard_legs(requester_leg);
                    return refuse(self, v2::ClaimReason::NoWindows);
                }
            }
        };
        if let Some(leg) = &requester_leg {
            self.channels.insert(leg.transfer, ChanOwner::Claim(key));
        }
        if let Some(leg) = &source_leg {
            self.channels.insert(leg.transfer, ChanOwner::Source(key));
        }
        // The requester's peer ranks, when it is a multi-process instance.
        let mut mp_legs = Vec::new();
        let sides = self
            .mp_side(
                sid,
                &|i| ChanOwner::ClaimRank(key, i),
                channel::Kind::Get,
                &mut mp_legs,
            )
            .and_then(|req| {
                self.mp_side(
                    source_sid,
                    &|i| ChanOwner::ClaimRank(key, i),
                    channel::Kind::Put,
                    &mut mp_legs,
                )
                .map(|src| (req, src))
            });
        let (req_mp, src_mp) = match sides {
            Ok(sides) => sides,
            Err(reason) => {
                self.discard_legs(
                    requester_leg
                        .into_iter()
                        .chain(source_leg)
                        .chain(mp_legs.drain(..).map(|l| l.ch)),
                );
                return refuse(self, reason);
            }
        };
        let source_channel_ref = source_leg
            .as_ref()
            .zip(self.cfg.channel.as_ref())
            .map(|(leg, hub)| super::channel_ref(hub, &leg.token));

        self.next_export += 1;
        let export_id = self.next_export;
        let mut claim = Claim::new(
            key,
            sid,
            &m,
            MpState::new(req_mp, src_mp),
            &self.cfg.limits,
            total,
        );
        claim.source_proc = source_proc;
        claim.source_sid = source_sid;
        claim.export_id = export_id;
        claim.export = ExportState::Requested;
        claim.retained = retained;
        claim.reserved_bytes = if retained { total } else { 0 };
        claim.block = block;
        claim.legs = join_legs(requester_leg, source_leg, mp_legs);
        self.bind_claim_legs(&mut claim);
        let reserved = claim.reserved_bytes;
        self.claims.insert(key, claim);
        self.exports.insert(export_id, key);
        let source_rank_channels = self.rank_channels_of(key, true);
        self.send(
            source_sid,
            M::ExportRequest(v2::ExportRequest {
                export_id,
                claim_id: key.2,
                tier: tier_to(tier) as i32,
                key: Some(key_to(endpoint)),
                begin_tokens: m.begin_tokens,
                want_row: m.want_row,
                pin_deadline_ms: self.cfg.limits.pin_deadline_ms,
                session_incarnation: source_sid,
                reserved_bytes: reserved,
                channel: source_channel_ref,
                rank_channels: source_rank_channels,
            }),
        );
        Ok(())
    }

    /// Open one endpoint's rank leg.
    pub(super) fn open_leg(
        &self,
        kind: channel::Kind,
        layout: std::sync::Arc<flare_selene::shared::SliceLayout>,
        rank: usize,
    ) -> std::io::Result<Channel> {
        let hub = self
            .cfg
            .channel
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no data channel"))?;
        let (transfer, token) = hub.open_rank(kind, layout, rank)?;
        Ok(Channel { transfer, token })
    }

    /// The layout of this process's rank slices, including a whole-record rank of one.
    pub(super) fn requester_slices(
        &self,
        sid: u64,
    ) -> std::sync::Arc<flare_selene::shared::SliceLayout> {
        let session = &self.sessions[&sid];
        let layout = &self.scopes[&session.scope].valid.slices;
        if session.ranks.is_some() {
            layout.clone()
        } else {
            std::sync::Arc::new(layout.whole())
        }
    }

    /// A claim whose source is the manager's own store: the store pins the chain
    /// itself and hands out the windows.
    fn manager_claim(
        &mut self,
        sid: u64,
        key: ClaimKey,
        m: &v2::Claim,
        walk: super::index::ChainWalk,
        geometry: &crate::record::Descriptor,
        mode: v2::CopyMode,
    ) -> Checked<()> {
        let refuse = |core: &mut Core, reason: v2::ClaimReason| {
            core.refused_claim(sid, key, m, reason);
            Ok(())
        };
        let (scope, pid) = {
            let session = &self.sessions[&sid];
            (session.scope, session.pid)
        };
        if self.link_for((key.0, key.1), scope, pid).is_none() {
            return refuse(self, v2::ClaimReason::SourceBusy);
        }
        let slices = self.requester_slices(sid);
        let channel = if mode == v2::CopyMode::DataChannel {
            match self.open_leg(channel::Kind::Get, slices, 0) {
                Ok(ch) => Some(ch),
                Err(_) => return refuse(self, v2::ClaimReason::NoWindows),
            }
        } else {
            None
        };
        let mut mp_legs = Vec::new();
        let req_mp = match self.mp_side(
            sid,
            &|i| ChanOwner::ClaimRank(key, i),
            channel::Kind::Get,
            &mut mp_legs,
        ) {
            Ok(side) => side,
            Err(reason) => {
                self.discard_legs(channel);
                return refuse(self, reason);
            }
        };
        let last = walk.nodes.len() - 1;
        let records: Vec<v2::Record> = walk
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let first_page = if index == 0 { walk.skip_pages } else { 0 };
                v2::Record {
                    key: Some(key_to(node.key)),
                    first_page,
                    pages: node.tail_pages - first_page,
                    row: index == last && m.want_row,
                }
            })
            .collect();
        let mp = MpState::new(req_mp, MpSide::whole(&self.scopes[&scope].valid.slices));
        let total =
            walk.pages * geometry.page_bytes() + if m.want_row { geometry.row_bytes() } else { 0 };
        let mut claim = Claim::new(key, sid, m, mp, &self.cfg.limits, total);
        claim.records = records;
        claim.total_pages = walk.pages;
        claim.row = m.want_row;
        claim.manager = Some(ManagerGet {
            checks: Default::default(),
            remaining: 0,
            pending: 0,
            starts: walk
                .nodes
                .iter()
                .map(|node| {
                    (node.key.consumed_tokens / geometry.page_tokens)
                        .saturating_sub(u64::from(node.tail_pages))
                })
                .collect(),
        });
        claim.legs = join_legs(channel, None, mp_legs);
        self.bind_claim_legs(&mut claim);
        self.claims.insert(key, claim);
        let chain: Vec<Key> = walk.nodes.iter().map(|node| node.key).collect();
        if let Err(reason) = self.manager_get(key, chain, walk.skip_pages, walk.pages, m.want_row) {
            self.finish(key, v2::ClaimState::Failed, reason);
        }
        Ok(())
    }

    /// The store's answers for a claim on the manager's own records.
    pub(super) fn get_answer(&mut self, key: ClaimKey, answer: std::io::Result<GetAnswer>) {
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        if claim.manager.is_none() {
            return;
        }
        let terminal = claim.is_terminal();
        match answer {
            Ok(GetAnswer::Leased { windows, .. }) => {
                claim.manager.as_mut().unwrap().remaining = windows;
                if terminal || claim.state != v2::ClaimState::Requested {
                    return;
                }
                let now = Instant::now();
                claim.state = v2::ClaimState::Granted;
                claim.granted_at = Some(now);
                claim.last_progress = now;
                activate(&mut claim.parties, Some(false));
                let channel = claim
                    .channel()
                    .zip(self.cfg.channel.as_ref())
                    .map(|(open, hub)| super::channel_ref(hub, &open.token));
                let mut granted = v2::ClaimGranted {
                    claim_id: key.2,
                    records: claim.records.clone(),
                    total_pages: claim.total_pages,
                    row: claim.row,
                    channel,
                    window_credits: self.cfg.limits.window_credits,
                    retained: false,
                    rank_channels: Vec::new(),
                };
                let sid = claim.requester_sid;
                granted.rank_channels = self.rank_channels(key);
                self.send(sid, M::ClaimGranted(granted));
            }
            Ok(GetAnswer::Meta {
                record, pages, row, ..
            }) => {
                claim
                    .manager
                    .as_mut()
                    .unwrap()
                    .checks
                    .insert(record, (pages, row));
            }
            Ok(GetAnswer::Ready(grant)) => {
                if terminal {
                    return;
                }
                // The store numbers pages from the start of the chain; the protocol
                // numbers them within a record.
                let start = claim
                    .manager
                    .as_ref()
                    .and_then(|manager| manager.starts.get(grant.record as usize))
                    .copied()
                    .unwrap_or(0);
                let first = grant.page_first.saturating_sub(start) as usize;
                let count = grant.page_count as usize;
                let sums = claim
                    .manager
                    .as_ref()
                    .and_then(|manager| manager.checks.get(&grant.record))
                    .and_then(|(pages, row_sum)| {
                        pages
                            .get(first..first + count)
                            .map(|pages| (pages.to_vec(), *row_sum))
                    });
                let Some((page_xxh3, row_xxh3)) = sums else {
                    self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
                    return;
                };
                let plan = PlannedWindow {
                    index: grant.index,
                    record: grant.record,
                    first_page: first as u32,
                    page_count: grant.page_count,
                    row: grant.row,
                    length: grant.length,
                };
                let window = Win {
                    plan,
                    region: grant.region,
                    offset: grant.offset,
                    phase: Phase::RequesterRead,
                    extent: Some(grant.hold),
                    readers: 1,
                    writers: 0,
                    stated: Vec::new(),
                    arrived: true,
                };
                let index = grant.index as usize;
                while claim.windows.len() <= index {
                    // A window index the store skipped is not in anyone's hands.
                    let mut filler = Win {
                        plan,
                        region: grant.region,
                        offset: 0,
                        phase: Phase::Free,
                        extent: None,
                        readers: 0,
                        writers: 0,
                        stated: Vec::new(),
                        arrived: true,
                    };
                    filler.plan.index = claim.windows.len() as u32;
                    claim.windows.push(filler);
                }
                claim.windows[index] = window;
                let window = &mut claim.windows[index];
                window.phase = Phase::SourceWrite;
                window.readers = 0;
                self.window_arrived(
                    key,
                    index,
                    vec![SliceSums {
                        pages: page_xxh3,
                        row: grant.row.then_some(row_xxh3),
                    }],
                );
            }
            Ok(GetAnswer::Done) => {
                attest(&mut claim.parties, Owner::Store, Party::Drained, false);
                // Every window is queued: the handler ends the transfer after them.
                if let Some(hub) = self.cfg.channel.as_ref() {
                    for leg in claim.legs.iter().filter(|leg| !leg.source) {
                        hub.finish(leg.ch.transfer);
                    }
                }
                self.complete_if_latched(key);
            }
            Ok(GetAnswer::Refused(refused)) => {
                let reason = match refused.reason {
                    // BUDGET, OVERSIZE: nothing to read into.
                    3 | 4 => v2::ClaimReason::NoWindows,
                    // IO: the record could not be read intact.
                    8 => v2::ClaimReason::CopyError,
                    _ => v2::ClaimReason::NotFound,
                };
                self.finish(key, v2::ClaimState::Failed, reason);
            }
            Err(_) => self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError),
        }
    }

    /// Records a claim refused before any export and tells the requester.
    fn refused_claim(&mut self, sid: u64, key: ClaimKey, m: &v2::Claim, reason: v2::ClaimReason) {
        let layout = self.requester_slices(sid);
        let mp = MpState::new(MpSide::whole(&layout), MpSide::whole(&layout));
        let mut claim = Claim::new(key, sid, m, mp, &self.cfg.limits, 0);
        claim.state = v2::ClaimState::Failed;
        claim.reason = reason;
        let message = self.terminal_message(&claim);
        self.claims.insert(key, claim);
        self.count("claims_refused_total", format!("reason=\"{reason:?}\""));
        self.send(sid, M::ClaimTerminal(message));
    }

    // ------------------------------------------------------------------ pin

    pub(super) fn pin_ack(&mut self, sid: u64, ack: v2::PinAck) -> Checked<()> {
        let Some(&key) = self.exports.get(&ack.export_id) else {
            return Ok(());
        };
        let Some(sender) = self.session_key(sid) else {
            return Ok(());
        };
        let Some(claim) = self.claims.get_mut(&key) else {
            return Ok(());
        };
        if claim.source_proc != Some(sender) || claim.export != ExportState::Requested {
            // Not this source's export, or an answer already counted.
            return Ok(());
        }
        let result = v2::PinResult::try_from(ack.result).unwrap_or(v2::PinResult::Unspecified);
        match result {
            v2::PinResult::Refused => {
                claim.export = ExportState::None;
                attest(
                    &mut claim.parties,
                    Owner::Proc(sender),
                    Party::Drained,
                    false,
                );
                let reason = refusal_reason(ack.refusal);
                self.finish(key, v2::ClaimState::Failed, reason);
            }
            v2::PinResult::Pinned => {
                claim.export = ExportState::Pinned;
                if claim.is_terminal() {
                    // A late pin: release it.
                    self.cancel_export(key, v2::ClaimReason::Cancelled);
                    return Ok(());
                }
                self.grant(key, ack);
            }
            v2::PinResult::Unspecified => {
                return Err(Violation::new("PinAck without a result"));
            }
        }
        Ok(())
    }

    /// The source pinned the chain: validate what it says, carve windows, grant.
    fn grant(&mut self, key: ClaimKey, ack: v2::PinAck) {
        let limits = self.cfg.limits.clone();
        let claim = self.claims.get(&key).expect("claim exists");
        let bad = |core: &mut Core, reason: v2::ClaimReason| {
            core.finish(key, v2::ClaimState::Failed, reason);
        };
        if ack.records.is_empty() || ack.records.len() > limits.max_chain_nodes as usize {
            return bad(self, v2::ClaimReason::CopyError);
        }
        let mut shapes = Vec::with_capacity(ack.records.len());
        let mut pages = 0u64;
        for (index, record) in ack.records.iter().enumerate() {
            let last = index + 1 == ack.records.len();
            if record.row && !last || record.row != (last && claim.want_row) && last {
                return bad(self, v2::ClaimReason::CopyError);
            }
            if key_from(record.key.as_ref()).is_err() {
                return bad(self, v2::ClaimReason::CopyError);
            }
            pages += u64::from(record.pages);
            shapes.push(RecordShape {
                first_page: record.first_page,
                pages: record.pages,
                row: record.row,
            });
        }
        let last_key = key_from(ack.records.last().and_then(|record| record.key.as_ref()));
        if last_key.ok() != Some(claim.endpoint)
            || pages != ack.total_pages
            || ack.row != claim.want_row
        {
            return bad(self, v2::ClaimReason::CopyError);
        }
        let Some(plan) = windows::plan(
            &shapes,
            claim.page_bytes,
            claim.row_bytes,
            limits.max_window_bytes,
        ) else {
            return bad(self, v2::ClaimReason::CopyError);
        };
        let total = windows::total_bytes(&plan);
        let claim = self.claims.get_mut(&key).expect("claim exists");
        if claim.retained {
            let capacity = claim
                .block
                .as_ref()
                .map_or(0, |block| block.capacity() as u64);
            if total > claim.reserved_bytes || total > capacity {
                return bad(self, v2::ClaimReason::NoWindows);
            }
        }
        let base = claim
            .block
            .as_ref()
            .map_or(0, |block| block.offset() as u64);
        let mut running = 0u64;
        claim.windows = plan
            .into_iter()
            .map(|plan| {
                let offset = base + running;
                if claim.retained {
                    running += plan.length;
                }
                Win {
                    plan,
                    region: v2::Region::Staging as i32,
                    offset,
                    phase: Phase::Unissued,
                    extent: None,
                    readers: 0,
                    writers: 0,
                    stated: Vec::new(),
                    arrived: false,
                }
            })
            .collect();
        claim.records = ack.records.clone();
        claim.total_pages = ack.total_pages;
        claim.row = ack.row;
        let now = Instant::now();
        claim.granted_at = Some(now);
        claim.last_progress = now;
        activate(&mut claim.parties, Some(false));
        let requester_sid = claim.requester_sid;
        let channel = claim
            .channel()
            .zip(self.cfg.channel.as_ref())
            .map(|(open, hub)| super::channel_ref(hub, &open.token));
        let mut granted = v2::ClaimGranted {
            claim_id: key.2,
            records: ack.records,
            total_pages: ack.total_pages,
            row: ack.row,
            channel,
            window_credits: limits.window_credits,
            retained: claim.retained,
            rank_channels: Vec::new(),
        };
        claim.state = v2::ClaimState::Granted;
        activate(&mut claim.parties, None);
        granted.rank_channels = self.rank_channels(key);
        self.send(requester_sid, M::ClaimGranted(granted));
        self.issue_windows(key);
    }

    /// Hands out windows to the source: all of them for a retained claim, as many as
    /// the credits allow otherwise.
    fn issue_windows(&mut self, key: ClaimKey) {
        let credits = self.cfg.limits.window_credits as usize;
        let staging = self.staging.clone();
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        if claim.is_terminal() {
            return;
        }
        let mut specs = Vec::new();
        let mut starved = false;
        for index in 0..claim.windows.len() {
            if claim.windows[index].phase != Phase::Unissued {
                continue;
            }
            if !claim.retained {
                if claim.unreleased() >= credits {
                    break;
                }
                let length = claim.windows[index].plan.length as usize;
                match staging.allocate(length) {
                    Some(extent) => {
                        claim.windows[index].offset = extent.offset() as u64;
                        claim.windows[index].extent = Some(std::sync::Arc::new(extent));
                    }
                    None => {
                        starved = true;
                        break;
                    }
                }
            }
            claim.windows[index].phase = Phase::SourceWrite;
            claim.windows[index].writers = claim
                .legs
                .iter()
                .filter(|leg| leg.source)
                .fold(1u64, |mask, leg| mask | 1 << leg.rank);
            claim.windows[index].stated = vec![None; claim.mp.src.layout.world()];
            specs.push(claim.spec(index));
        }
        let outstanding = claim.unreleased();
        let (export_id, source_sid) = (claim.export_id, claim.source_sid);
        let legs = claim
            .legs
            .iter()
            .filter(|leg| leg.source)
            .map(|leg| (leg.ch.transfer, leg.rank))
            .collect();
        let layout = claim.mp.src.layout.clone();
        if specs.is_empty() && starved && outstanding == 0 {
            self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::NoWindows);
            return;
        }
        let message = M::ExportFill(v2::ExportFill {
            export_id,
            windows: specs.clone(),
        });
        self.grant_windows(source_sid, message, legs, &layout, &specs);
    }

    pub(super) fn grant_windows(
        &mut self,
        sid: u64,
        message: M,
        legs: Vec<(u64, u32)>,
        layout: &flare_selene::shared::SliceLayout,
        specs: &[v2::WindowSpec],
    ) {
        if let Some(hub) = &self.cfg.channel {
            for (transfer, rank) in &legs {
                for spec in specs {
                    let Some(region) = u8::try_from(spec.region)
                        .ok()
                        .and_then(flare_selene::shared::region::Region::from_u8)
                    else {
                        continue;
                    };
                    hub.push_grant(
                        *transfer,
                        channel::GrantOut {
                            shape: shape_for(spec, layout, *rank as usize),
                            region,
                            offset: spec.offset,
                        },
                    );
                }
            }
        }
        if !specs.is_empty() && !legs.iter().any(|(_, rank)| *rank == 0) {
            self.send(sid, message);
        }
    }

    // ------------------------------------------------------------------ windows

    pub(super) fn window_filled(&mut self, sid: u64, m: v2::WindowFilled) -> Checked<()> {
        let Some(&key) = self.exports.get(&m.export_id) else {
            return Ok(());
        };
        let Some(sender) = self.session_key(sid) else {
            return Ok(());
        };
        let Some(claim) = self.claims.get_mut(&key) else {
            return Ok(());
        };
        if claim.source_proc != Some(sender) || claim.source_channel().is_some() {
            // A relayed source states its windows' checksums on the data channel.
            return Ok(());
        }
        let Some(window) = claim.windows.get_mut(m.index as usize) else {
            return Err(Violation::new(
                "WindowFilled names a window that was never issued",
            ));
        };
        match window.phase {
            Phase::SourceWrite => {}
            Phase::Unissued => {
                return Err(Violation::new("WindowFilled before the window was issued"));
            }
            // A duplicate, or a window already released.
            Phase::RequesterRead | Phase::Free => return Ok(()),
        }
        let status = v2::FillStatus::try_from(m.status).unwrap_or(v2::FillStatus::Unspecified);
        if status == v2::FillStatus::FillFailed {
            self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
            return Ok(());
        }
        if status != v2::FillStatus::FillOk {
            return Err(Violation::new("WindowFilled without a status"));
        }
        if claim.mp.src.layout.world() > 1 && (!m.page_xxh3.is_empty() || m.rank_sums.is_empty()) {
            self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
            return Ok(());
        }
        let rank_sums = if claim.mp.src.layout.world() > 1 {
            m.rank_sums
        } else {
            Vec::new()
        };
        let parts = if rank_sums.is_empty() {
            vec![(
                0,
                SliceSums {
                    pages: m.page_xxh3,
                    row: m.has_row.then_some(m.row_xxh3),
                },
            )]
        } else {
            super::mp::rank_sums(rank_sums)
        };
        self.window_parts(ChanOwner::Source(key), m.index, parts, 0);
        Ok(())
    }

    pub(super) fn window_released(&mut self, sid: u64, m: v2::WindowReleased) -> Checked<()> {
        let Some((instance, nonce)) = self.session_key(sid) else {
            return Ok(());
        };
        let key = (instance, nonce, m.claim_id);
        let Some(claim) = self.claims.get_mut(&key) else {
            return Ok(());
        };
        let Some(window) = claim.windows.get_mut(m.index as usize) else {
            return Err(Violation::new(
                "WindowReleased names a window that was never issued",
            ));
        };
        match window.phase {
            Phase::RequesterRead => {}
            Phase::Free => return Ok(()),
            Phase::SourceWrite | Phase::Unissued => {
                return Err(Violation::new(
                    "WindowReleased before the window was filled",
                ));
            }
        }
        self.reader_done(key, m.index, 0);
        Ok(())
    }

    pub(super) fn copy_result(&mut self, sid: u64, m: v2::CopyResult) -> Checked<()> {
        let Some((instance, nonce)) = self.session_key(sid) else {
            return Ok(());
        };
        let key = (instance, nonce, m.claim_id);
        let Some(claim) = self.claims.get(&key) else {
            return Ok(());
        };
        if claim.is_terminal() {
            return Ok(());
        }
        let outcome = v2::CopyOutcome::try_from(m.outcome).unwrap_or(v2::CopyOutcome::Unspecified);
        match outcome {
            v2::CopyOutcome::CopyOk => {
                // Exact coverage: every window issued, filled and released.
                let covered = claim.state == v2::ClaimState::Granted;
                let pages: u64 = claim
                    .windows
                    .iter()
                    .map(|window| u64::from(window.plan.page_count))
                    .sum();
                let exact = pages == claim.total_pages
                    && (!claim.want_row || claim.windows.iter().any(|window| window.plan.row));
                // On the data channel the claim's windows are freed as the handler reports
                // them sent, on another path than this message. The requester's verdict
                // proves it read them all, so windows still marked sent-or-sending count.
                let windows_done =
                    if claim.channel().is_some() || claim.legs.iter().any(|leg| leg.rank != 0) {
                        claim.windows.iter().all(|window| {
                            matches!(window.phase, Phase::RequesterRead | Phase::Free)
                        })
                    } else {
                        claim.all_windows_done()
                    };
                let complete = covered && exact && windows_done && !claim.windows.is_empty();
                if complete {
                    self.claims.get_mut(&key).unwrap().verdict_latched = true;
                    self.complete_if_latched(key);
                } else {
                    self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
                }
            }
            v2::CopyOutcome::CopyChecksum => {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Checksum);
            }
            v2::CopyOutcome::CopyErrorOutcome => {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
            }
            v2::CopyOutcome::CopyAbandoned => {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Cancelled);
            }
            v2::CopyOutcome::Unspecified => {
                return Err(Violation::new("CopyResult without an outcome"));
            }
        }
        Ok(())
    }

    pub(super) fn claim_release(&mut self, sid: u64, claim_id: u64) -> Checked<()> {
        let Some((instance, nonce)) = self.session_key(sid) else {
            return Ok(());
        };
        let key = (instance, nonce, claim_id);
        match self.claims.get(&key) {
            None => {
                self.send(
                    sid,
                    M::ClaimTerminal(v2::ClaimTerminal {
                        claim_id,
                        state: v2::ClaimState::Failed as i32,
                        reason: v2::ClaimReason::UnknownClaim as i32,
                        fence_required: false,
                    }),
                );
            }
            Some(claim) if claim.is_terminal() => {}
            Some(_) => {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Cancelled);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------ fences

    pub(super) fn fence_ack(&mut self, sid: u64, m: v2::FenceAck) -> Checked<()> {
        let outcome =
            v2::FenceOutcome::try_from(m.outcome).unwrap_or(v2::FenceOutcome::Unspecified);
        let party = match outcome {
            v2::FenceOutcome::Drained => Party::Drained,
            v2::FenceOutcome::Quarantined => Party::Quarantined,
            v2::FenceOutcome::Unspecified => {
                return Err(Violation::new("FenceAck without an outcome"));
            }
        };
        let Some((instance, nonce)) = self.session_key(sid) else {
            return Ok(());
        };
        match m.subject {
            Some(v2::fence_ack::Subject::ClaimId(claim_id)) => {
                let key = (instance, nonce, claim_id);
                if let Some(claim) = self.claims.get_mut(&key) {
                    attest(
                        &mut claim.parties,
                        Owner::Proc((instance, nonce)),
                        party,
                        false,
                    );
                    self.reclaim(key);
                }
                Ok(())
            }
            Some(v2::fence_ack::Subject::PutId(put_id)) => {
                self.put_fence(instance, nonce, put_id, party);
                Ok(())
            }
            None => Err(Violation::new("FenceAck without a subject")),
        }
    }

    pub(super) fn export_end(&mut self, sid: u64, m: v2::ExportEnd) -> Checked<()> {
        let Some(&key) = self.exports.get(&m.export_id) else {
            return Ok(());
        };
        let Some(sender) = self.session_key(sid) else {
            return Ok(());
        };
        let Some(claim) = self.claims.get_mut(&key) else {
            return Ok(());
        };
        if claim.source_proc != Some(sender) {
            return Ok(());
        }
        let fence = v2::FenceOutcome::try_from(m.fence).unwrap_or(v2::FenceOutcome::Unspecified);
        let party = match fence {
            v2::FenceOutcome::Drained => Party::Drained,
            v2::FenceOutcome::Quarantined => Party::Quarantined,
            v2::FenceOutcome::Unspecified => {
                return Err(Violation::new("ExportEnd without a fence"));
            }
        };
        match claim.export {
            ExportState::Requested | ExportState::Pinned => {}
            _ => return Ok(()),
        }
        claim.export = ExportState::Ended;
        attest(&mut claim.parties, Owner::Proc(sender), party, false);
        if !claim.is_terminal() {
            let status =
                v2::ExportEndStatus::try_from(m.status).unwrap_or(v2::ExportEndStatus::Unspecified);
            // A relayed source's last windows may still be on the channel when its
            // ExportEnd arrives; the channel reports any that never come. So may the last
            // windows of a multi-process source's channel ranks: rank 0 ends the export once
            // every rank has finished, and a leg's bytes can still be in the manager's
            // handler. The leg's close fails the claim if any window never arrived.
            let finished = status == v2::ExportEndStatus::ExportComplete
                && (claim.source_channel().is_some()
                    || claim.legs.iter().filter(|leg| leg.rank != 0).any(|leg| {
                        leg.source
                            && party_state(&claim.parties, Owner::Leg(leg.ch.transfer))
                                != Party::Drained
                    })
                    || claim.windows.iter().all(|window| {
                        // A window in the assembly pass has been filled, too.
                        window.arrived || matches!(window.phase, Phase::RequesterRead | Phase::Free)
                    }));
            if !finished {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
                return Ok(());
            }
        }
        self.reclaim(key);
        Ok(())
    }

    // ------------------------------------------------------------------ terminal

    /// The single authoritative transition. The first call for a claim wins.
    pub(super) fn finish(&mut self, key: ClaimKey, state: v2::ClaimState, reason: v2::ClaimReason) {
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        if claim.is_terminal() {
            return;
        }
        claim.state = state;
        claim.reason = reason;
        self.count(
            "claims_total",
            format!("state=\"{state:?}\",reason=\"{reason:?}\""),
        );
        let claim = self.claims.get_mut(&key).expect("claim exists");
        // The store retains its own I/O holds after cancellation.
        attest(&mut claim.parties, Owner::Store, Party::Drained, false);
        let message = self.terminal_message(self.claims.get(&key).expect("claim exists"));
        let requester_sid = self.claims[&key].requester_sid;
        self.send(requester_sid, M::ClaimTerminal(message));
        self.cancel_export(key, reason);
        if self.claims[&key].manager.is_some() {
            self.store_cancel_get(key);
        }
        self.stop_channel(key);
        self.reclaim(key);
    }

    /// The claim is over: end its data-channel transfers. The claim's windows go back
    /// only when the handlers have exited, which `reclaim` checks.
    fn stop_channel(&mut self, key: ClaimKey) {
        if let Some(claim) = self.claims.get_mut(&key) {
            stop_legs(
                &claim.legs,
                &mut claim.parties,
                self.cfg.channel.as_ref(),
                &mut self.channels,
            );
        }
    }
    fn bind_claim_legs(&mut self, claim: &mut Claim) {
        claim.parties = participants(
            Some((claim.key.0, claim.key.1)),
            claim.source_proc,
            &claim.legs,
            [(false, &claim.mp.req), (true, &claim.mp.src)]
                .into_iter()
                .map(|(source, side)| {
                    (
                        source,
                        side.peers
                            .iter()
                            .filter(|p| p.shared)
                            .map(|p| p.proc)
                            .collect(),
                    )
                }),
        );
        if claim.manager.is_some() {
            claim.parties.push(Participant {
                owner: Owner::Store,
                source: true,
                memory: false,
                state: Party::Pending,
            });
        }
        for (index, leg) in claim.legs.iter().enumerate() {
            let owner = match (leg.rank, leg.source) {
                (0, false) => ChanOwner::Claim(claim.key),
                (0, true) => ChanOwner::Source(claim.key),
                _ => ChanOwner::ClaimRank(claim.key, index),
            };
            self.channels.insert(leg.ch.transfer, owner);
        }
    }

    // ------------------------------------------------------------------ data channel

    /// What a data-channel handler reported.
    pub(super) fn channel_event(&mut self, event: channel::Event) {
        use channel::Event as E;
        match event {
            E::Sent { transfer, index } => match self.channels.get(&transfer).copied() {
                Some(ChanOwner::Claim(key)) => self.reader_done(key, index, 0),
                Some(ChanOwner::ClaimRank(key, leg)) => {
                    let rank = self
                        .claims
                        .get(&key)
                        .and_then(|claim| claim.legs.get(leg))
                        .map(|leg| leg.rank);
                    if let Some(rank) = rank {
                        self.reader_done(key, index, rank);
                    }
                }
                _ => {}
            },
            E::Filled {
                transfer,
                index,
                row_xxh3,
                page_xxh3,
            } => {
                let endpoint = match self.channels.get(&transfer).copied() {
                    Some(ChanOwner::ClaimRank(key, leg)) => self
                        .claims
                        .get(&key)
                        .and_then(|c| c.legs.get(leg))
                        .filter(|l| l.source)
                        .map(|l| (ChanOwner::Source(key), l.rank)),
                    Some(ChanOwner::PutRank(key, leg)) => self
                        .puts
                        .get(&key)
                        .and_then(|r| r.legs.get(leg))
                        .map(|l| (ChanOwner::Put(key), l.rank)),
                    Some(owner @ (ChanOwner::Put(_) | ChanOwner::Source(_))) => Some((owner, 0)),
                    _ => None,
                };
                if let Some((owner, rank)) = endpoint {
                    self.window_parts(
                        owner,
                        index,
                        vec![(
                            rank as usize,
                            SliceSums {
                                pages: page_xxh3,
                                row: row_xxh3,
                            },
                        )],
                        rank,
                    );
                }
            }
            E::Failed { transfer, reason } => match self.channels.get(&transfer).copied() {
                Some(ChanOwner::ClaimRank(key, _)) => {
                    eprintln!(
                        "selene: claim {}: a peer rank's data channel failed: {reason}",
                        key.2
                    );
                    self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
                }
                Some(ChanOwner::Claim(key)) => {
                    eprintln!(
                        "selene: claim {} on the data channel failed: {reason}",
                        key.2
                    );
                    self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
                }
                Some(ChanOwner::Source(key)) => {
                    eprintln!(
                        "selene: claim {}: its source's data channel failed: {reason}",
                        key.2
                    );
                    self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::SourceLost);
                }
                Some(ChanOwner::Put(key)) => self.put_channel_failed(key, &reason),
                Some(ChanOwner::PutRank(key, _)) => self.put_channel_failed(key, &reason),
                None => {}
            },
            E::Closed { transfer } => {
                if let Some(hub) = &self.cfg.channel {
                    hub.forget(transfer);
                }
                let owner = self.channels.remove(&transfer);
                let parties = match owner {
                    Some(
                        ChanOwner::Claim(key)
                        | ChanOwner::Source(key)
                        | ChanOwner::ClaimRank(key, _),
                    ) => self.claims.get_mut(&key).map(|c| &mut c.parties),
                    Some(ChanOwner::Put(key) | ChanOwner::PutRank(key, _)) => {
                        self.puts.get_mut(&key).map(|r| &mut r.parties)
                    }
                    None => None,
                };
                if let Some(parties) = parties {
                    attest(parties, Owner::Leg(transfer), Party::Drained, false);
                }
                match owner {
                    Some(ChanOwner::ClaimRank(key, leg)) => {
                        // A leg that ends before the claim is over, with windows it never
                        // read or wrote, fails the claim; one that finished is simply done.
                        let owed = self.claims.get(&key).is_some_and(|claim| {
                            !claim.is_terminal()
                                && claim.legs.get(leg).is_some_and(|open| {
                                    if open.source {
                                        claim
                                            .windows
                                            .iter()
                                            .any(|w| !w.arrived && w.phase != Phase::Free)
                                    } else {
                                        claim.windows.iter().any(|w| {
                                            w.phase == Phase::RequesterRead
                                                && w.readers & (1 << open.rank) != 0
                                        })
                                    }
                                })
                        });
                        if owed {
                            self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
                        }
                        self.reclaim(key);
                    }
                    Some(ChanOwner::Claim(key)) => {
                        self.reclaim(key);
                    }
                    Some(ChanOwner::Source(key)) => {
                        // A source that closes its channel before the claim is over has
                        // given up the windows it owed.
                        let owed = self.claims.get(&key).is_some_and(|claim| {
                            !claim.is_terminal()
                                && claim.windows.iter().any(|w| {
                                    matches!(w.phase, Phase::Unissued | Phase::SourceWrite)
                                })
                        });
                        if owed {
                            self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::SourceLost);
                        }
                        self.reclaim(key);
                    }
                    Some(ChanOwner::Put(key)) => self.puts_settle(key),
                    Some(ChanOwner::PutRank(key, _)) => {
                        self.puts_settle(key);
                    }
                    None => {}
                }
            }
        }
    }

    /// One accessor of a window is done reading it: slot 0 is the requester's session (or its
    /// own leg), slot `r` the leg of peer rank `r`. The window is free when every accessor
    /// has finished; an acknowledgment counts once per accessor, whatever its repeats.
    pub(super) fn reader_done(&mut self, key: ClaimKey, index: u32, slot: u32) {
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        let Some(window) = claim.windows.get_mut(index as usize) else {
            return;
        };
        if window.phase != Phase::RequesterRead || window.readers & (1 << slot) == 0 {
            return;
        }
        window.readers &= !(1 << slot);
        if window.readers != 0 {
            return self.complete_if_latched(key);
        }
        window.phase = Phase::Free;
        window.extent = None;
        claim.last_progress = Instant::now();
        let terminal = claim.is_terminal();
        let from_store = claim.manager.is_some();
        if from_store {
            self.store_release_window(key, index);
        }
        if terminal {
            self.reclaim(key);
        } else if !from_store {
            // A relayed claim's next window can take the freed room.
            self.issue_windows(key);
        }
        self.complete_if_latched(key);
    }

    /// The requester's COPY_OK was exact but waited for a leg or the store; finish the claim
    /// now if nothing is outstanding.
    fn complete_if_latched(&mut self, key: ClaimKey) {
        let Some(claim) = self.claims.get(&key) else {
            return;
        };
        if claim.is_terminal() || !claim.verdict_latched || claim.legs_pending() {
            return;
        }
        if party_state(&claim.parties, Owner::Store) == Party::Pending {
            return;
        }
        self.finish(key, v2::ClaimState::Copied, v2::ClaimReason::ReasonNone);
    }

    /// Tells the source to stop and release, if its export is still open.
    pub(super) fn cancel_export(&mut self, key: ClaimKey, reason: v2::ClaimReason) {
        let Some(claim) = self.claims.get(&key) else {
            return;
        };
        if !matches!(claim.export, ExportState::Requested | ExportState::Pinned) {
            return;
        }
        let export_id = claim.export_id;
        let Some((instance, _)) = claim.source_proc else {
            return;
        };
        let Some(sid) = self
            .instances
            .get(&instance)
            .and_then(|entry| entry.current)
        else {
            return;
        };
        // Only a session of the same process can act on the export.
        let same_process = self
            .sessions
            .get(&sid)
            .is_some_and(|session| Some((session.instance, session.nonce)) == claim.source_proc);
        if same_process {
            self.send(
                sid,
                M::ExportCancel(v2::ExportCancel {
                    export_id,
                    reason: reason as i32,
                }),
            );
        }
    }

    /// Frees every window whose last toucher has drained, and drops the claim record
    /// once the reply is no longer needed.
    pub(super) fn reclaim(&mut self, key: ClaimKey) {
        let Some(claim) = self.claims.get(&key) else {
            return;
        };
        if !claim.is_terminal() {
            return;
        }
        let source_ok = self.holders(&claim.parties, Some(true), claim.mp.passes) == Party::Drained;
        let requester_ok =
            self.holders(&claim.parties, Some(false), claim.mp.passes) == Party::Drained;
        let claim = self.claims.get_mut(&key).expect("claim exists");
        for window in &mut claim.windows {
            match window.phase {
                Phase::Unissued | Phase::Free => {
                    window.phase = Phase::Free;
                    window.extent = None;
                }
                Phase::SourceWrite if source_ok => {
                    window.phase = Phase::Free;
                    window.extent = None;
                }
                Phase::RequesterRead if requester_ok => {
                    window.phase = Phase::Free;
                    window.extent = None;
                }
                _ => {}
            }
        }
        if claim
            .windows
            .iter()
            .all(|window| window.phase == Phase::Free)
        {
            claim.block = None;
        }
        if requester_ok && claim.manager.is_some() {
            self.store_release_transfer(key);
        }
    }

    pub(super) fn claim_timers(&mut self, now: Instant) {
        let progress = self.cfg.limits.progress_timeout();
        let mut expire = Vec::new();
        let mut forget = Vec::new();
        for (key, claim) in &self.claims {
            if claim.is_terminal() {
                if claim.block.is_none()
                    && claim.windows.iter().all(|w| w.phase == Phase::Free)
                    && self.holders(&claim.parties, None, claim.mp.passes) == Party::Drained
                {
                    forget.push(*key);
                }
                continue;
            }
            if now >= claim.deadline {
                expire.push((*key, v2::ClaimReason::Deadline));
            } else if claim.state == v2::ClaimState::Requested && now >= claim.pin_deadline {
                expire.push((*key, v2::ClaimReason::SourceTimeout));
            } else if matches!(claim.state, v2::ClaimState::Granted)
                && now.duration_since(claim.last_progress) >= progress
            {
                expire.push((*key, v2::ClaimReason::NoProgress));
            }
        }
        for (key, reason) in expire {
            self.finish(key, v2::ClaimState::Failed, reason);
        }
        for key in forget {
            if let Some(claim) = self.claims.remove(&key) {
                self.exports.remove(&claim.export_id);
            }
        }
    }

    // ------------------------------------------------------------------ rank agents

    /// An agent attested that it drained one transfer.
    pub(super) fn agent_fence(&mut self, agent: ProcKey, kind: i32, id: u64, party: Party) {
        use v2::rank_fence::Kind;
        let key = match Kind::try_from(kind) {
            Ok(Kind::Claim) => {
                // The claim is named by rank 0's id; the agent's instance is its rank 0's.
                let rank0 = self
                    .agents
                    .values()
                    .find(|a| a.proc() == agent)
                    .map(|a| a.rank0);
                rank0.map(|nonce| (agent.0, nonce, id))
            }
            Ok(Kind::Export) => self.exports.get(&id).copied(),
            Ok(Kind::Put) => {
                let rank0 = self
                    .agents
                    .values()
                    .find(|a| a.proc() == agent)
                    .map(|a| a.rank0);
                if let Some(nonce) = rank0 {
                    self.put_agent_fence((agent.0, nonce, id), agent, party);
                }
                return;
            }
            _ => None,
        };
        let Some(key) = key else { return };
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        attest(&mut claim.parties, Owner::Proc(agent), party, false);
        self.reclaim(key);
    }

    /// An attached agent is gone or replaced: the transfers it takes part in fail. What it
    /// may still be touching stays its own obligation.
    pub(super) fn agent_gone(&mut self, agent: ProcKey) {
        self.puts_peer_gone(agent);
        let keys: Vec<ClaimKey> = self
            .claims
            .iter()
            .filter(|(_, claim)| !claim.is_terminal())
            .filter(|(_, claim)| {
                claim.parties.iter().any(|p| {
                    p.owner == Owner::Proc(agent)
                        && p.memory
                        && agent != (claim.key.0, claim.key.1)
                        && Some(agent) != claim.source_proc
                })
            })
            .map(|(key, _)| *key)
            .collect();
        for key in keys {
            self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::RankLost);
        }
    }

    // ------------------------------------------------------------------ processes

    /// A process is confirmed gone: nothing it did can touch memory any more.
    pub(super) fn proc_exited(&mut self, process: ProcKey) {
        self.on_proc(process, true);
    }

    /// `Goodbye{quiesced}` cannot clear a previously quarantined obligation.
    pub(super) fn party_quiesced(&mut self, process: ProcKey) {
        self.on_proc(process, false);
    }

    fn on_proc(&mut self, process: ProcKey, exited: bool) {
        let keys: Vec<_> = self.claims.keys().copied().collect();
        for key in keys {
            let claim = self.claims.get_mut(&key).unwrap();
            attest(
                &mut claim.parties,
                Owner::Proc(process),
                Party::Drained,
                exited,
            );
            if claim.source_proc == Some(process)
                && matches!(claim.export, ExportState::Requested | ExportState::Pinned)
            {
                claim.export = ExportState::Ended;
            }
            self.reclaim(key);
        }
        let keys: Vec<_> = self.puts.keys().copied().collect();
        for key in keys {
            let rec = self.puts.get_mut(&key).unwrap();
            attest(
                &mut rec.parties,
                Owner::Proc(process),
                Party::Drained,
                exited,
            );
            if exited && (key.0, key.1) == process {
                rec.phase = super::store_bridge::PutPhase::Finished;
                rec.refused = true;
                rec.pending_windows = 0;
                let connection = rec.link.connection;
                self.store_send(crate::store::StoreCommand::Cancel {
                    connection,
                    transfer: key.2,
                });
            }
            self.puts_settle(key);
        }
        if exited && !self.puts.keys().any(|k| (k.0, k.1) == process) {
            self.drop_link(process);
        }
    }

    /// A session ended: its claims, as requester and as source, fail.
    pub(super) fn session_claims_lost(&mut self, sid: u64, session: &Session) {
        let keys: Vec<(ClaimKey, bool)> = self
            .claims
            .iter()
            .filter(|(_, claim)| !claim.is_terminal())
            .filter_map(|(key, claim)| {
                if claim.requester_sid == sid {
                    Some((*key, true))
                } else if claim.source_sid == sid && claim.source_sid != 0 {
                    Some((*key, false))
                } else {
                    None
                }
            })
            .collect();
        for (key, as_requester) in keys {
            if !as_requester {
                // A source without shared memory reaches the claim's windows only through
                // the manager's handler, whose exit is the fence. Its session ending
                // clears what it owed for the export itself.
                if let Some(claim) = self.claims.get_mut(&key) {
                    if claim.source_channel().is_some() {
                        claim.export = ExportState::Ended;
                        if let Some(proc) = claim.source_proc {
                            attest(&mut claim.parties, Owner::Proc(proc), Party::Drained, false);
                        }
                    }
                }
            }
            if as_requester {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::RequesterLost);
            } else {
                self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::SourceLost);
            }
        }
        // The process may already be gone.
        let process = (session.instance, session.nonce);
        if self.proc_exited_flag(&process) {
            self.proc_exited(process);
        }
    }
}

/// The shape a window has on one rank's leg: its slice's length when the instance is a
/// multi-process one, else the whole window's.
pub(super) fn shape_for(
    spec: &v2::WindowSpec,
    layout: &flare_selene::shared::SliceLayout,
    rank: usize,
) -> Shape {
    Shape {
        index: spec.index,
        record: spec.record,
        first_page: spec.first_page,
        page_count: spec.page_count,
        row: spec.row,
        length: layout.slice_bytes(rank, spec.page_count, spec.row),
    }
}

/// Record shapes for the index's view of a chain.
fn shapes_of(walk: &super::index::ChainWalk, want_row: bool) -> Vec<RecordShape> {
    let last = walk.nodes.len() - 1;
    walk.nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let first_page = if index == 0 { walk.skip_pages } else { 0 };
            RecordShape {
                first_page,
                pages: node.tail_pages - first_page,
                row: want_row && index == last,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arena::Arena,
        pool::core::{Core, PoolConfig, Session},
    };
    use std::collections::HashSet;
    use tokio::sync::mpsc;

    const INSTANCE: InstanceId = [1; 16];
    const NONCE: Nonce = [2; 16];

    fn core() -> (Core, mpsc::Receiver<v2::ServerMessage>) {
        core_with(PoolConfig::default())
    }

    fn core_with(config: PoolConfig) -> (Core, mpsc::Receiver<v2::ServerMessage>) {
        let (mut core, _handle, _events) = Core::new(
            config,
            v2::SharedMemory::default(),
            Arena::new(1 << 20).unwrap(),
            None,
        );
        let (tx, rx) = mpsc::channel(64);
        core.sessions.insert(
            1,
            Session {
                sid: 1,
                instance: INSTANCE,
                nonce: NONCE,
                scope: [3; 32],
                tx,
                shm: true,
                pid: 1,
                export_slots: 2,
                claim_slots: 2,
                need_resync: false,
                syncing: false,
                lookups: HashSet::new(),
                goodbye: None,
                overflowed: false,
                ranks: None,
            },
        );
        (core, rx)
    }

    /// A claim on the manager's store whose five pages and row were all delivered and
    /// released.
    fn delivered(core: &mut Core) -> ClaimKey {
        let key: ClaimKey = (INSTANCE, NONCE, 1);
        let plan = PlannedWindow {
            index: 0,
            record: 0,
            first_page: 0,
            page_count: 5,
            row: true,
            length: 6 * 4096,
        };
        let now = Instant::now();
        let mp = MpState::new(
            MpSide::whole(&rank_layout(1)),
            MpSide::whole(&rank_layout(1)),
        );
        let mut claim = Claim::new(
            key,
            1,
            &v2::Claim {
                want_row: true,
                ..Default::default()
            },
            mp,
            &core.cfg.limits,
            6 * 4096,
        );
        claim.state = v2::ClaimState::Granted;
        claim.total_pages = 5;
        claim.row = true;
        claim.granted_at = Some(now);
        claim.windows = vec![Win {
            plan,
            region: v2::Region::Staging as i32,
            offset: 0,
            phase: Phase::Free,
            extent: None,
            readers: 0,
            writers: 0,
            stated: Vec::new(),
            arrived: false,
        }];
        claim.parties = vec![
            Participant {
                owner: Owner::Proc((INSTANCE, NONCE)),
                source: false,
                memory: true,
                state: Party::Pending,
            },
            Participant {
                owner: Owner::Store,
                source: true,
                memory: false,
                state: Party::Pending,
            },
        ];
        claim.manager = Some(ManagerGet {
            checks: Default::default(),
            remaining: 0,
            pending: 0,
            starts: vec![0],
        });
        core.claims.insert(key, claim);
        key
    }

    fn copy_ok() -> v2::CopyResult {
        v2::CopyResult {
            claim_id: 1,
            outcome: v2::CopyOutcome::CopyOk as i32,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn copy_ok_before_the_stores_drain_completes_when_it_arrives() {
        let (mut core, _out) = core();
        let key = delivered(&mut core);
        core.copy_result(1, copy_ok()).unwrap();
        assert!(
            !core.claims[&key].is_terminal(),
            "waits for the store's drain"
        );
        core.get_answer(key, Ok(GetAnswer::Done));
        assert_eq!(core.claims[&key].state, v2::ClaimState::Copied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn copy_ok_after_the_stores_drain_completes_at_once() {
        let (mut core, _out) = core();
        let key = delivered(&mut core);
        core.get_answer(key, Ok(GetAnswer::Done));
        assert!(!core.claims[&key].is_terminal());
        core.copy_result(1, copy_ok()).unwrap();
        assert_eq!(core.claims[&key].state, v2::ClaimState::Copied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_store_error_after_an_early_copy_ok_fails_the_claim() {
        let (mut core, _out) = core();
        let key = delivered(&mut core);
        core.copy_result(1, copy_ok()).unwrap();
        core.get_answer(
            key,
            Ok(GetAnswer::Refused(crate::store::Refused {
                reason: 8,
                detail: "disk read failed".into(),
            })),
        );
        assert_eq!(core.claims[&key].state, v2::ClaimState::Failed);
        assert_eq!(core.claims[&key].reason, v2::ClaimReason::CopyError);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_inexact_copy_ok_still_fails_whatever_the_store_says() {
        let (mut core, _out) = core();
        let key = delivered(&mut core);
        core.claims.get_mut(&key).unwrap().total_pages = 6;
        core.get_answer(key, Ok(GetAnswer::Done));
        core.copy_result(1, copy_ok()).unwrap();
        assert_eq!(core.claims[&key].state, v2::ClaimState::Failed);
    }

    /// A handler whose exit is never seen: the claim's windows stay out of service, are
    /// counted as quarantined while the handler remains live, and return only when the exit is seen.
    #[tokio::test(flavor = "multi_thread")]
    async fn windows_of_a_handler_that_cannot_be_seen_to_exit_are_quarantined() {
        use crate::{arena::Arena, data::channel::Hub, regions::LocalRegions};
        use flare_selene::shared::frame::{Carrier, ToManager};
        use tokio::io::AsyncWriteExt;
        let regions = std::sync::Arc::new(LocalRegions::new(
            Arena::new(1 << 20).unwrap(),
            Arena::new(1 << 20).unwrap(),
        ));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let (hub, _events) = Hub::start(
            &tokio::runtime::Handle::current(),
            listener,
            regions,
            None,
            64,
            None,
        )
        .unwrap();
        hub.delay_closed(Duration::from_secs(60));
        let config = PoolConfig {
            channel: Some(hub.clone()),
            ..PoolConfig::default()
        };
        let (mut core, _out) = core_with(config);
        let key = delivered(&mut core);
        let (transfer, token) = hub.open(channel::Kind::Get).unwrap();
        // The instance is connected and quiet: a handler is attached.
        let mut socket = tokio::net::TcpStream::connect(hub.address()).await.unwrap();
        socket.write_all(&token).await.unwrap();
        ToManager::Hello(Carrier::Stream)
            .write(&mut socket)
            .await
            .unwrap();
        while hub.handlers() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let claim = core.claims.get_mut(&key).unwrap();
        claim.windows[0].phase = Phase::RequesterRead;
        claim.legs.insert(
            0,
            Leg {
                rank: 0,
                source: false,
                ch: Channel { transfer, token },
            },
        );
        claim
            .parties
            .retain(|p| p.owner != Owner::Proc((INSTANCE, NONCE)));
        claim.parties.push(Participant {
            owner: Owner::Leg(transfer),
            source: claim.legs[0].source,
            memory: true,
            state: Party::Pending,
        });
        core.channels.insert(transfer, ChanOwner::Claim(key));
        core.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Cancelled);
        let claim = &core.claims[&key];
        assert_eq!(
            party_state(&claim.parties, Owner::Leg(transfer)),
            Party::Pending
        );
        assert_eq!(
            claim.windows[0].phase,
            Phase::RequesterRead,
            "the handler may still read the window"
        );
        core.claim_timers(Instant::now());
        assert_eq!(core.quarantined_bytes(), 6 * 4096);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::RequesterRead);
        // Seeing the exit is the proof that nothing reads the window any more.
        core.channel_event(channel::Event::Closed { transfer });
        assert_eq!(core.quarantined_bytes(), 0);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::Free);
        assert!(core.channels.is_empty());
    }

    /// The same for the leg a source without shared memory writes its windows on.
    #[tokio::test(flavor = "multi_thread")]
    async fn windows_of_a_source_leg_that_cannot_be_seen_to_exit_are_quarantined() {
        use crate::{arena::Arena, data::channel::Hub, regions::LocalRegions};
        use flare_selene::shared::frame::{Carrier, ToManager};
        use tokio::io::AsyncWriteExt;
        let regions = std::sync::Arc::new(LocalRegions::new(
            Arena::new(1 << 20).unwrap(),
            Arena::new(1 << 20).unwrap(),
        ));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let (hub, _events) = Hub::start(
            &tokio::runtime::Handle::current(),
            listener,
            regions,
            None,
            64,
            None,
        )
        .unwrap();
        hub.delay_closed(Duration::from_secs(60));
        let config = PoolConfig {
            channel: Some(hub.clone()),
            ..PoolConfig::default()
        };
        let (mut core, _out) = core_with(config);
        let key = delivered(&mut core);
        let (transfer, token) = hub.open(channel::Kind::Put).unwrap();
        let mut socket = tokio::net::TcpStream::connect(hub.address()).await.unwrap();
        socket.write_all(&token).await.unwrap();
        ToManager::Hello(Carrier::Stream)
            .write(&mut socket)
            .await
            .unwrap();
        while hub.handlers() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let claim = core.claims.get_mut(&key).unwrap();
        claim.windows[0].phase = Phase::SourceWrite;
        claim.legs.insert(
            0,
            Leg {
                rank: 0,
                source: true,
                ch: Channel { transfer, token },
            },
        );
        claim
            .parties
            .retain(|p| p.owner != Owner::Proc((INSTANCE, NONCE)));
        claim.parties.push(Participant {
            owner: Owner::Leg(transfer),
            source: claim.legs[0].source,
            memory: true,
            state: Party::Pending,
        });
        core.channels.insert(transfer, ChanOwner::Source(key));
        core.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Cancelled);
        assert_eq!(
            core.claims[&key].windows[0].phase,
            Phase::SourceWrite,
            "the handler may still write the window"
        );
        core.claim_timers(Instant::now());
        assert_eq!(core.quarantined_bytes(), 6 * 4096);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::SourceWrite);
        core.channel_event(channel::Event::Closed { transfer });
        assert_eq!(core.quarantined_bytes(), 0);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::Free);
        assert!(core.channels.is_empty());
    }

    // ---- multi-process requesters -------------------------------------------------

    use super::super::mp::{Leg, MpPeer, MpSide};

    fn rank_layout(world: usize) -> std::sync::Arc<flare_selene::shared::SliceLayout> {
        use flare_selene::shared::{Geometry, RankPlanes};
        let rank = RankPlanes {
            plane_bytes: vec![4096],
            row_bytes: 4096,
        };
        std::sync::Arc::new(
            flare_selene::shared::SliceLayout::new(&Geometry {
                page_tokens: 64,
                ranks: vec![rank; world],
                page_record_bytes: world as u64 * 4096,
                row_record_bytes: world as u64 * 4096,
            })
            .unwrap(),
        )
    }

    const AGENT_1: ProcKey = ([1; 16], [11; 16]);
    const AGENT_2: ProcKey = ([1; 16], [12; 16]);

    /// A claim from a three-rank requester with one window read by rank 0's session, the
    /// shared peer rank 1 (a participant) and the leg of rank 2.
    fn ranked(core: &mut Core, window_phase: Phase) -> ClaimKey {
        let key = delivered(core);
        for agent in [AGENT_1, AGENT_2] {
            core.procs.insert(
                agent,
                crate::pool::core::Proc {
                    pid: None,
                    pidfd: None,
                    exited: false,
                    claim_hw: 0,
                    put_hw: 0,
                    idle_since: None,
                    link: None,
                },
            );
        }
        let claim = core.claims.get_mut(&key).unwrap();
        claim.windows[0].phase = window_phase;
        claim.windows[0].readers = 0b101;
        claim.mp.req = MpSide {
            layout: rank_layout(3),
            peers: vec![
                MpPeer {
                    rank: 1,
                    proc: AGENT_1,
                    shared: true,
                },
                MpPeer {
                    rank: 2,
                    proc: AGENT_2,
                    shared: false,
                },
            ],
        };
        claim.legs = vec![Leg {
            rank: 2,
            source: false,
            ch: Channel {
                transfer: 99,
                token: [0; flare_selene::shared::frame::TOKEN_BYTES],
            },
        }];
        attest(&mut claim.parties, Owner::Store, Party::Drained, false);
        claim.parties.push(Participant {
            owner: Owner::Proc(AGENT_1),
            source: false,
            memory: true,
            state: Party::Pending,
        });
        claim.parties.push(Participant {
            owner: Owner::Leg(99),
            source: false,
            memory: true,
            state: Party::Pending,
        });
        key
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_window_is_free_only_when_every_accessor_finished_and_repeats_are_not_votes() {
        let (mut core, _out) = core();
        let key = ranked(&mut core, Phase::RequesterRead);
        core.reader_done(key, 0, 0);
        core.reader_done(key, 0, 0);
        core.reader_done(key, 0, 0);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::RequesterRead);
        assert_eq!(core.claims[&key].windows[0].readers, 0b100);
        // A slot that was never required is no vote either.
        core.reader_done(key, 0, 1);
        core.reader_done(key, 0, 5);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::RequesterRead);
        core.reader_done(key, 0, 2);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::Free);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_exact_copy_ok_waits_for_the_legs_and_completes_when_the_last_one_reports() {
        let (mut core, _out) = core();
        let key = ranked(&mut core, Phase::RequesterRead);
        core.reader_done(key, 0, 0);
        core.copy_result(1, copy_ok()).unwrap();
        assert!(
            !core.claims[&key].is_terminal(),
            "rank 2's leg has not finished reading"
        );
        assert!(core.claims[&key].verdict_latched);
        core.reader_done(key, 0, 2);
        assert_eq!(core.claims[&key].state, v2::ClaimState::Copied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_shared_peers_window_is_released_by_its_own_fence_and_not_by_anyone_elses() {
        let (mut core, _out) = core();
        let key = ranked(&mut core, Phase::RequesterRead);
        core.claims.get_mut(&key).unwrap().windows[0].readers = 0;
        core.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Cancelled);
        // The requester's session fenced itself (the fixture's party is Pending).
        attest(
            &mut core.claims.get_mut(&key).unwrap().parties,
            Owner::Proc((INSTANCE, NONCE)),
            Party::Drained,
            false,
        );
        attest(
            &mut core.claims.get_mut(&key).unwrap().parties,
            Owner::Leg(99),
            Party::Drained,
            false,
        );
        core.reclaim(key);
        assert_eq!(
            core.claims[&key].windows[0].phase,
            Phase::RequesterRead,
            "peer 1 may still be reading"
        );
        // A different agent (a replacement) attesting does nothing for peer 1.
        core.agent_fence(
            AGENT_2,
            v2::rank_fence::Kind::Claim as i32,
            1,
            Party::Drained,
        );
        core.reclaim(key);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::RequesterRead);
        // Peer 1's own process exiting is its proof.
        core.proc_exited(AGENT_1);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::Free);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_that_attests_quarantined_keeps_the_window_until_its_process_exits() {
        let (mut core, _out) = core();
        let key = ranked(&mut core, Phase::RequesterRead);
        core.claims.get_mut(&key).unwrap().windows[0].readers = 0;
        core.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Cancelled);
        let claim = core.claims.get_mut(&key).unwrap();
        attest(
            &mut claim.parties,
            Owner::Proc((INSTANCE, NONCE)),
            Party::Drained,
            false,
        );
        attest(&mut claim.parties, Owner::Leg(99), Party::Drained, false);
        attest(
            &mut claim.parties,
            Owner::Proc(AGENT_1),
            Party::Quarantined,
            false,
        );
        core.reclaim(key);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::RequesterRead);
        assert_eq!(core.quarantined_bytes(), 6 * 4096);
        core.agent_fence(
            AGENT_1,
            v2::rank_fence::Kind::Claim as i32,
            1,
            Party::Drained,
        );
        assert_eq!(core.claims[&key].windows[0].phase, Phase::RequesterRead);
        core.proc_exited(AGENT_1);
        assert_eq!(core.claims[&key].windows[0].phase, Phase::Free);
        assert_eq!(core.quarantined_bytes(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn losing_an_attached_peer_fails_the_claims_it_takes_part_in() {
        let (mut core, _out) = core();
        let key = ranked(&mut core, Phase::RequesterRead);
        core.agent_gone(AGENT_2);
        assert!(
            !core.claims[&key].is_terminal(),
            "rank 2 is on a leg, not a participant"
        );
        core.agent_gone(AGENT_1);
        assert_eq!(core.claims[&key].state, v2::ClaimState::Failed);
        assert_eq!(core.claims[&key].reason, v2::ClaimReason::RankLost);
    }

    fn ending_source(core: &mut Core, leg_exited: bool) -> ClaimKey {
        let key = ranked(core, Phase::SourceWrite);
        let claim = core.claims.get_mut(&key).unwrap();
        claim.source_proc = Some((INSTANCE, NONCE));
        claim.export_id = 7;
        claim.export = ExportState::Pinned;
        claim.windows[0].readers = 0;
        // Rank 2's slices are still to come from its data channel.
        claim.windows[0].writers = 0b100;
        claim.legs[0].source = true;
        for p in claim
            .parties
            .iter_mut()
            .filter(|p| p.owner == Owner::Leg(99))
        {
            p.source = true;
            p.state = if leg_exited {
                Party::Drained
            } else {
                Party::Pending
            };
        }
        core.exports.insert(7, key);
        key
    }

    fn complete_export() -> v2::ExportEnd {
        v2::ExportEnd {
            export_id: 7,
            status: v2::ExportEndStatus::ExportComplete as i32,
            fence: v2::FenceOutcome::Drained as i32,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_export_that_ends_before_a_rank_legs_last_bytes_are_handled_does_not_fail_the_claim()
    {
        let (mut core, _out) = core();
        let key = ending_source(&mut core, false);
        core.export_end(1, complete_export()).unwrap();
        assert!(
            !core.claims[&key].is_terminal(),
            "the leg's channel reports any window that never comes"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_export_that_ends_with_a_window_missing_and_no_leg_left_to_deliver_it_fails_the_claim()
     {
        let (mut core, _out) = core();
        let key = ending_source(&mut core, true);
        core.export_end(1, complete_export()).unwrap();
        assert_eq!(core.claims[&key].reason, v2::ClaimReason::CopyError);
    }
}
