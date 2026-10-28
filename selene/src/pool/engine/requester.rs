//! The requester role: restores from other instances and the manager's store.
//!
//! A restore has one pool source. Its claim is a transfer that the manager
//! brokers: the source pins first, the manager grants windows, this client
//! checks every page and the row against the checksums the source supplied,
//! copies the window into the destination pages through the page copier, and
//! returns the window only after that copy has drained. The restore reports
//! one result after every destination write has drained, whatever happened.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    time::{Duration, Instant},
};

use crate::{Direction, Error, PageId, PrefixKey, StateSlotId, plan};

use super::{
    super::{
        codec, frame,
        net::{Channel, Id},
        shared::{FetchFailure, FetchPart, PoolEvent, PoolFetch},
        transfer_timeout,
        wire::{self, client_message::M as Out},
    },
    CLAIM_DEADLINE_MS, Engine, Owner, RESTORE_DEADLINE,
    coverage::Coverage,
    ranks::{PeerPart, XferOwner},
};

/// A record the claim is expected to be granted, derived from the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpectedRecord {
    pub key: PrefixKey,
    pub first_page: u32,
    pub pages: u32,
    pub row: bool,
}

/// The records a part must be granted: those of its chain from the page that
/// holds its first token to its end, with the first record entered at the
/// record-relative page where the part starts. A part that ends in the page it
/// starts in has no pages, and its last record is a row-only tail.
pub(crate) fn expected_records(
    page_tokens: u64,
    part: &plan::Part,
    nodes: &[plan::ChainNode],
) -> Option<Vec<ExpectedRecord>> {
    let start_page = part.begin_tokens / page_tokens;
    let mut records = Vec::new();
    let mut total = 0usize;
    for (index, key) in part.chain.iter().enumerate() {
        let node = nodes.iter().find(|node| node.key == *key)?;
        let end = key.consumed_tokens / page_tokens;
        let start = end.checked_sub(node.tail_pages as u64)?;
        let from = start.max(start_page);
        let pages = end.saturating_sub(from);
        let last = index + 1 == part.chain.len();
        if pages == 0 && !last {
            continue;
        }
        total += pages as usize;
        records.push(ExpectedRecord {
            key: *key,
            first_page: (from - start) as u32,
            pages: pages as u32,
            row: last && part.row,
        });
    }
    (total == part.pages && !records.is_empty()).then_some(records)
}

#[derive(Default)]
pub(crate) struct State {
    next_claim: u64,
    restores: HashMap<u64, Restore>,
    claims: HashMap<u64, Claim>,
    /// Restores that wait for one of the manager's claim slots, oldest first.
    waiting: VecDeque<(PoolFetch, Instant)>,
}

impl State {
    pub(crate) fn is_empty(&self) -> bool {
        self.restores.is_empty() && self.claims.is_empty() && self.waiting.is_empty()
    }
}

/// Delay before retrying a claim refused for temporary capacity.
const CLAIM_RETRY: Duration = Duration::from_millis(250);

struct Restore {
    fetch: FetchPart,
    expected: Vec<ExpectedRecord>,
    claim: Option<u64>,
    deadline: Instant,
    failure: Option<FetchFailure>,
    bytes: u64,
    /// A claim refused for temporary capacity, and when to retry (set
    /// by the next tick, which knows the time).
    retry: Option<Option<Instant>>,
}

struct Window {
    jobs_left: u32,
    /// Shared-memory peer ranks that have not yet drained their copy of this window.
    peers_left: BTreeSet<usize>,
    released: bool,
    /// The staging slot holding the window, when it came over the data channel.
    slot: Option<u32>,
}

struct Claim {
    restore: u64,
    peers: Vec<PeerPart>,
    total_pages: u64,
    wants_row: bool,
    pages: Vec<PageId>,
    slot: Option<StateSlotId>,
    granted: bool,
    released: bool,
    terminal: Option<wire::ClaimTerminal>,
    coverage: Coverage,
    windows: BTreeMap<u32, Window>,
    jobs_outstanding: u32,
    bad: Option<wire::CopyOutcome>,
    verdict_sent: bool,
    fence_sent: bool,
    /// The claim's bytes come over the data channel (this session has no shared memory).
    via_channel: bool,
    /// The channel's task, until it has been stopped and has reported that it is gone.
    channel: Option<Channel>,
    channel_open: bool,
    /// The manager said the transfer is complete.
    ended: bool,
}

impl Engine {
    // ---- starting a restore ---------------------------------------------------

    pub(crate) fn start_fetch(&mut self, now: Instant, fetch: PoolFetch) {
        let id = fetch.id;
        match self.build_restore(now, &fetch) {
            Ok(restore) if !self.claim_room() => {
                // The manager grants a requester only so many claims at once and refuses the
                // rest; a restore waits for a slot instead of being refused.
                let ahead = self
                    .requester
                    .waiting
                    .back()
                    .map(|(_, deadline)| *deadline)
                    .or_else(|| self.requester.restores.values().map(|r| r.deadline).max())
                    .unwrap_or(now)
                    .max(now);
                let deadline = ahead + transfer_timeout(RESTORE_DEADLINE, restore.bytes);
                self.requester.waiting.push_back((fetch, deadline));
            }
            Ok(restore) => {
                self.requester.restores.insert(id, restore);
                self.claim_part(id);
            }
            Err(error) => self.event(PoolEvent::Fetched {
                id,
                result: Err(FetchFailure {
                    error,
                    source: None,
                    no_headroom: false,
                }),
            }),
        }
    }

    /// Keep each restore within the session's granted claim slots.
    fn claim_room(&self) -> bool {
        let granted = self
            .sess
            .as_ref()
            .map_or(0, |s| s.limits.max_claims_in_flight);
        let slots = match (self.st.claim_slots, granted) {
            (0, _) => return true,
            (own, 0) => own,
            (own, granted) => own.min(granted),
        } as usize;
        self.requester.restores.len() < slots
    }

    /// Start the waiting restores that now fit, in order, and give up on those that waited
    /// longer than a restore may take.
    pub(crate) fn admit_waiting(&mut self, now: Instant) {
        while let Some((_, deadline)) = self.requester.waiting.front() {
            let expired = now >= *deadline;
            if !expired && !self.claim_room() {
                break;
            }
            let (fetch, _) = self.requester.waiting.pop_front().expect("front");
            let id = fetch.id;
            let built = if expired {
                Err(Error::new("the restore waited too long for a claim slot"))
            } else {
                self.build_restore(now, &fetch)
            };
            match built {
                Ok(restore) => {
                    self.requester.restores.insert(id, restore);
                    self.claim_part(id);
                }
                Err(error) => self.event(PoolEvent::Fetched {
                    id,
                    result: Err(FetchFailure {
                        error,
                        source: None,
                        no_headroom: false,
                    }),
                }),
            }
        }
    }

    fn build_restore(&self, now: Instant, fetch: &PoolFetch) -> Result<Restore, Error> {
        if !self.ready() || self.closing || self.quarantined() {
            return Err(Error::new("the pool is not ready"));
        }
        if self.memory().is_none() {
            return Err(Error::new(
                "this session has neither shared memory nor a data channel to copy through",
            ));
        }
        let part = &fetch.part;
        let expected = expected_records(fetch.page_tokens, &part.part, &part.candidate.plan.nodes)
            .ok_or_else(|| Error::new("a part's chain does not match its pages"))?;
        if part.pages.len() != part.part.pages || part.part.row != part.slot.is_some() {
            return Err(Error::new("a part's destination does not match its plan"));
        }
        let bytes = part.part.pages as u64 * self.st.page_record_bytes()
            + if part.part.row {
                self.st.row_record_bytes()
            } else {
                0
            };
        Ok(Restore {
            fetch: part.clone(),
            expected,
            claim: None,
            deadline: now + transfer_timeout(RESTORE_DEADLINE, bytes),
            failure: None,
            bytes,
            retry: None,
        })
    }

    fn claim_part(&mut self, restore: u64) {
        let Some(part) = self.requester.restores.get_mut(&restore) else {
            return;
        };
        self.requester.next_claim += 1;
        let id = self.requester.next_claim;
        part.claim = Some(id);
        let fetch = &part.fetch;
        let p = &fetch.part;
        let claim = wire::Claim {
            claim_id: id,
            source: Some(wire::SourceRef {
                instance_id: fetch.candidate.source.instance.clone(),
                session_incarnation: fetch.candidate.source.session_incarnation,
            }),
            tier: codec::tier_to(fetch.candidate.tier) as i32,
            key: Some(codec::key_to(p.end)),
            begin_tokens: p.begin_tokens,
            want_row: p.row,
            mode: if self.dial.is_some() {
                wire::CopyMode::DataChannel
            } else {
                wire::CopyMode::SharedWindows
            } as i32,
            deadline_ms: CLAIM_DEADLINE_MS,
        };
        let expected = part.expected.clone();
        let total_pages = expected.iter().map(|r| u64::from(r.pages)).sum::<u64>();
        self.requester.claims.insert(
            id,
            Claim {
                restore,
                peers: Vec::new(),
                coverage: Coverage::new(expected),
                total_pages,
                wants_row: p.row,
                pages: fetch.pages.clone(),
                slot: fetch.slot,
                granted: false,
                released: false,
                terminal: None,
                windows: BTreeMap::new(),
                jobs_outstanding: 0,
                bad: None,
                verdict_sent: false,
                fence_sent: false,
                via_channel: self.dial.is_some(),
                channel: None,
                channel_open: false,
                ended: false,
            },
        );
        self.emit(Out::Claim(claim));
    }

    // ---- grants and windows ---------------------------------------------------

    pub(crate) fn on_claim_granted(&mut self, granted: wire::ClaimGranted) {
        let id = granted.claim_id;
        // A grant for a claim this client no longer holds is never used; its
        // terminal message is answered with an immediate fence.
        let Some(claim) = self.requester.claims.get_mut(&id) else {
            return;
        };
        if claim.granted || claim.released || claim.terminal.is_some() || claim.bad.is_some() {
            return;
        }
        let via_channel = claim.via_channel;
        let verdict = if granted.channel.is_some() != via_channel {
            Err(if via_channel {
                "the grant names no data channel, which this session needs"
            } else {
                "the grant names a data channel, which this session does not use"
            })
        } else if !records_match(&claim.coverage.expected, &granted.records)
            || granted.total_pages != claim.total_pages
            || granted.row != claim.wants_row
        {
            Err("the granted records do not match the plan")
        } else {
            Ok(())
        };
        match verdict {
            Err(why) => {
                // The grant is held, so release it with a verdict, not a cancel.
                claim.granted = true;
                let restore = claim.restore;
                self.fail_restore(
                    restore,
                    Some(id),
                    Error::new(why),
                    wire::CopyOutcome::CopyAbandoned,
                );
            }
            Ok(()) => {
                claim.granted = true;
                let restore = claim.restore;
                if let Some(reference) = &granted.channel {
                    match self.open_channel(Id::Claim(id), reference) {
                        Ok(channel) => {
                            let claim = self.requester.claims.get_mut(&id).expect("present");
                            claim.channel = Some(channel);
                            claim.channel_open = true;
                        }
                        Err(error) => {
                            return self.fail_restore(
                                restore,
                                Some(id),
                                error,
                                wire::CopyOutcome::CopyErrorOutcome,
                            );
                        }
                    }
                }
                if let Err(error) = self.open_peer_parts(id, &granted) {
                    return self.fail_restore(
                        restore,
                        Some(id),
                        error,
                        wire::CopyOutcome::CopyErrorOutcome,
                    );
                }
            }
        }
    }

    pub(crate) fn on_window_ready(&mut self, ready: wire::WindowReady) {
        self.window_ready(ready, None);
    }

    /// A window of a claim arrived over the data channel into `slot`: it is a
    /// shared window that lies in the instance's own staging.
    pub(crate) fn on_net_window(&mut self, claim: u64, window: frame::Window, slot: u32) {
        let Some(staging) = self.net.clone() else {
            return;
        };
        let shape = window.shape;
        let ready = wire::WindowReady {
            claim_id: claim,
            window: Some(wire::WindowSpec {
                index: shape.index,
                region: wire::Region::Staging as i32,
                offset: staging.offset(slot),
                length: shape.length,
                record: shape.record,
                first_page: shape.first_page,
                page_count: shape.page_count,
                row: shape.row,
            }),
            page_xxh3: window.page_xxh3,
            row_xxh3: window.row_xxh3,
            has_row: shape.row,
            rank_sums: Vec::new(),
        };
        self.window_ready(ready, Some(slot));
    }

    /// Take up a window. A slot the window did not keep is freed.
    fn window_ready(&mut self, mut ready: wire::WindowReady, slot: Option<u32>) {
        // A multi-process instance's shared window comes with every shared rank's slice
        // sums; this rank checks its own and passes the others on to their agents.
        let others = if self.ranks.is_some() && slot.is_none() {
            let mut all = std::mem::take(&mut ready.rank_sums);
            match all.iter().position(|sums| sums.rank == 0) {
                Some(position) => {
                    let mine = all.remove(position);
                    ready.page_xxh3 = mine.page_xxh3;
                    ready.row_xxh3 = mine.row_xxh3;
                    ready.has_row = mine.has_row;
                    all
                }
                None => {
                    ready.page_xxh3.clear();
                    all
                }
            }
        } else {
            Vec::new()
        };
        let taken = self.take_window(ready, slot, others);
        if let (false, Some(slot)) = (taken, slot) {
            self.free_slot(slot);
        }
    }

    /// `false` when the window was not taken up and its slot, if any, is still the
    /// caller's. Once a window is recorded its slot belongs to it.
    fn take_window(
        &mut self,
        ready: wire::WindowReady,
        slot: Option<u32>,
        others: Vec<wire::RankSums>,
    ) -> bool {
        let id = ready.claim_id;
        let Some(spec) = ready.window else {
            return false;
        };
        let memory = self.memory();
        let (page_bytes, row_bytes) = self.window_shape(slot.is_some());
        let stride = if slot.is_some() {
            page_bytes
        } else {
            self.st.page_record_bytes()
        } as usize;
        let Some(claim) = self.requester.claims.get_mut(&id) else {
            return false;
        };
        if !claim.granted || claim.released || claim.terminal.is_some() || claim.bad.is_some() {
            return false;
        }
        let restore = claim.restore;
        let dest = match claim.coverage.accept(&spec, page_bytes, row_bytes) {
            Ok(dest) => dest,
            Err(why) => {
                self.fail_restore(
                    restore,
                    Some(id),
                    Error::new(why),
                    wire::CopyOutcome::CopyErrorOutcome,
                );
                return false;
            }
        };
        claim.windows.insert(
            spec.index,
            Window {
                jobs_left: 0,
                peers_left: BTreeSet::new(),
                released: false,
                slot,
            },
        );
        let Some(region) = memory.as_ref().and_then(|m| m.region(spec.region())) else {
            self.fail_restore(
                restore,
                Some(id),
                Error::new("window in an unknown region"),
                wire::CopyOutcome::CopyErrorOutcome,
            );
            return true;
        };
        let Some(_) = region.bytes(spec.offset, spec.length) else {
            self.fail_restore(
                restore,
                Some(id),
                Error::new("window outside its region"),
                wire::CopyOutcome::CopyErrorOutcome,
            );
            return true;
        };
        let count = spec.page_count as usize;
        if ready.page_xxh3.len() != count || ready.has_row != spec.row {
            self.fail_restore(
                restore,
                Some(id),
                Error::new("window checksums do not match its pages"),
                wire::CopyOutcome::CopyErrorOutcome,
            );
            return true;
        }
        let sums = self.window_sums(spec, slot.is_some());
        let mismatch = sums.map_or(true, |sums| {
            sums.pages != ready.page_xxh3 || sums.row != spec.row.then_some(ready.row_xxh3)
        });
        if mismatch {
            // Nothing was copied, so the window is free to go back at once.
            self.release_window(id, spec.index);
            self.fail_restore(
                restore,
                Some(id),
                Error::new("a page or row failed its checksum"),
                wire::CopyOutcome::CopyChecksum,
            );
            return true;
        }
        let base = region
            .address(spec.offset, spec.length)
            .expect("checked above");
        let claim = self.requester.claims.get_mut(&id).expect("present");
        let dest = dest as usize;
        let jobs = super::super::agent::jobs_of(
            Direction::In,
            spec.index,
            count,
            spec.row,
            |i| claim.pages[dest + i],
            claim.slot,
            |i| base + i * stride,
            base + count * stride,
        );
        // The shared-memory peers copy their own slices of this window in step.
        let mut forward = Vec::new();
        let mut missing = None;
        for peer in claim.peers.iter().filter(|p| p.shared && p.done.is_none()) {
            match others.iter().find(|sums| sums.rank as usize == peer.rank) {
                Some(sums) => forward.push((peer.rank, peer.xfer, sums.clone())),
                None => missing = Some(peer.rank),
            }
        }
        let window = claim.windows.get_mut(&spec.index).expect("inserted");
        window.jobs_left = jobs.len() as u32;
        window.peers_left = forward.iter().map(|(rank, _, _)| *rank).collect();
        claim.jobs_outstanding += jobs.len() as u32;
        for job in jobs {
            self.queue_job(
                job,
                Owner::Claim {
                    claim: id,
                    window: spec.index,
                },
            );
        }
        if let Some(rank) = missing {
            self.fail_restore(
                restore,
                Some(id),
                Error::new(format!("the manager gave no checksums for rank {rank}")),
                wire::CopyOutcome::CopyErrorOutcome,
            );
            return true;
        }
        for (rank, xfer, sums) in forward {
            let sent = self.ranks.as_mut().is_some_and(|ranks| {
                ranks.send_window(
                    rank,
                    super::super::rank::Window {
                        xfer,
                        spec: Some(spec),
                        sums: Some(crate::shared::SliceSums {
                            pages: sums.page_xxh3,
                            row: sums.has_row.then_some(sums.row_xxh3),
                        }),
                    },
                )
            });
            if !sent {
                self.fail_restore(
                    restore,
                    Some(id),
                    Error::new(format!("rank {rank}'s link is down")),
                    wire::CopyOutcome::CopyErrorOutcome,
                );
                return true;
            }
        }
        true
    }

    /// Start every peer rank's part of a granted claim.
    fn open_peer_parts(
        &mut self,
        id: u64,
        granted: &wire::ClaimGranted,
    ) -> std::result::Result<(), Error> {
        use super::super::rank;
        let Some(ranks) = self.ranks.as_mut() else {
            return Ok(());
        };
        let Some(claim) = self.requester.claims.get_mut(&id) else {
            return Ok(());
        };
        let (page_bytes, row_bytes) = (self.st.page_record_bytes(), self.st.row_record_bytes());
        let open = rank::Open {
            out: false,
            kind: wire::rank_fence::Kind::Claim as i32,
            fence_id: id,
            records: granted.records.clone(),
            pages: claim.pages.iter().map(|page| page.0).collect(),
            has_slot: claim.slot.is_some(),
            slot: claim.slot.map_or(0, |slot| slot.0),
            page_bytes,
            row_bytes,
            ..Default::default()
        };
        ranks
            .open_peers(
                XferOwner::Claim(id),
                &granted.rank_channels,
                open,
                &mut claim.peers,
            )
            .map_err(Error::new)
    }

    /// A shared-memory peer drained its copy of one window.
    pub(crate) fn on_peer_window_done(
        &mut self,
        claim_id: u64,
        rank: usize,
        index: u32,
        result: std::result::Result<(), String>,
    ) {
        let Some(claim) = self.requester.claims.get_mut(&claim_id) else {
            return;
        };
        let restore = claim.restore;
        if let Some(window) = claim.windows.get_mut(&index) {
            window.peers_left.remove(&rank);
        }
        if let Err(error) = result {
            self.fail_restore(
                restore,
                Some(claim_id),
                Error::new(format!("rank {rank}: {error}")),
                wire::CopyOutcome::CopyErrorOutcome,
            );
        }
        let done = self
            .requester
            .claims
            .get(&claim_id)
            .and_then(|c| c.windows.get(&index))
            .is_some_and(|w| w.jobs_left == 0 && w.peers_left.is_empty() && !w.released);
        if done {
            self.release_window(claim_id, index);
        }
        self.drive_claim(claim_id);
    }

    /// A peer rank ended its part of a claim: it has drained, or says it cannot prove it.
    pub(crate) fn on_peer_done(
        &mut self,
        claim_id: u64,
        rank: usize,
        result: std::result::Result<(), String>,
        quarantined: bool,
    ) {
        let Some(claim) = self.requester.claims.get_mut(&claim_id) else {
            return;
        };
        let restore = claim.restore;
        let Some(peer) = claim
            .peers
            .iter_mut()
            .find(|p| p.rank == rank && p.done.is_none())
        else {
            return;
        };
        peer.done = Some(result.clone());
        peer.quarantined = quarantined;
        if quarantined {
            // The rank cannot show its copies stopped: nothing it may write is released.
            return self.quarantine();
        }
        // A shared peer that is done owes no window any more.
        for window in claim.windows.values_mut() {
            window.peers_left.remove(&rank);
        }
        if let Err(error) = result {
            self.fail_restore(
                restore,
                Some(claim_id),
                Error::new(format!("rank {rank}: {error}")),
                wire::CopyOutcome::CopyErrorOutcome,
            );
        }
        let ready: Vec<u32> = self
            .requester
            .claims
            .get(&claim_id)
            .map(|c| {
                c.windows
                    .iter()
                    .filter(|(_, w)| w.jobs_left == 0 && w.peers_left.is_empty() && !w.released)
                    .map(|(index, _)| *index)
                    .collect()
            })
            .unwrap_or_default();
        for index in ready {
            self.release_window(claim_id, index);
        }
        self.drive_claim(claim_id);
    }

    /// The manager ended a claim's transfer cleanly.
    pub(crate) fn on_net_end(&mut self, claim: u64) {
        let Some(open) = self.requester.claims.get_mut(&claim) else {
            return;
        };
        open.ended = true;
        let restore = open.restore;
        let stray = open.terminal.is_none() && open.bad.is_none() && !open.coverage.complete();
        if stray {
            self.fail_restore(
                restore,
                Some(claim),
                Error::new("the data channel ended before every window arrived"),
                wire::CopyOutcome::CopyErrorOutcome,
            );
        }
        self.drive_claim(claim);
    }

    /// A claim's transfer failed.
    fn net_claim_failed(&mut self, claim: u64, error: String) {
        let Some(open) = self.requester.claims.get(&claim) else {
            return;
        };
        if open.terminal.is_some() || open.bad.is_some() {
            return;
        }
        let restore = open.restore;
        self.fail_restore(
            restore,
            Some(claim),
            Error::new(error),
            wire::CopyOutcome::CopyErrorOutcome,
        );
    }

    /// A claim's task is gone and its connection closed: nothing writes to its slots.
    fn net_claim_closed(&mut self, claim: u64) {
        let Some(open) = self.requester.claims.get_mut(&claim) else {
            return;
        };
        open.channel = None;
        open.channel_open = false;
        let early = open.terminal.is_none()
            && open.bad.is_none()
            && !(open.ended && open.coverage.complete());
        let restore = open.restore;
        if early {
            self.fail_restore(
                restore,
                Some(claim),
                Error::new("the data channel closed before the claim was complete"),
                wire::CopyOutcome::CopyErrorOutcome,
            );
        }
        self.drive_claim(claim);
    }

    pub(crate) fn on_net_failed(&mut self, id: Id, error: String) {
        match id {
            Id::Claim(claim) => self.net_claim_failed(claim, error),
            Id::Put(put) => self.net_put_failed(put, error),
            Id::Export(export) => self.net_export_failed(export, error),
        }
    }

    pub(crate) fn on_net_closed(&mut self, id: Id) {
        match id {
            Id::Claim(claim) => self.net_claim_closed(claim),
            Id::Put(put) => self.net_put_closed(put),
            Id::Export(export) => self.net_export_closed(export),
        }
    }

    pub(crate) fn on_claim_job(&mut self, id: u64, window: u32, result: crate::Result<()>) {
        let Some(claim) = self.requester.claims.get_mut(&id) else {
            return;
        };
        claim.jobs_outstanding = claim.jobs_outstanding.saturating_sub(1);
        if let Some(w) = claim.windows.get_mut(&window) {
            w.jobs_left = w.jobs_left.saturating_sub(1);
        }
        let failed = result.err();
        let restore = claim.restore;
        if let Some(error) = failed {
            self.fail_restore(
                restore,
                Some(id),
                error,
                wire::CopyOutcome::CopyErrorOutcome,
            );
        }
        let done = self
            .requester
            .claims
            .get(&id)
            .and_then(|c| c.windows.get(&window))
            .is_some_and(|w| w.jobs_left == 0 && w.peers_left.is_empty() && !w.released);
        if done {
            self.release_window(id, window);
        }
        self.drive_claim(id);
    }

    fn release_window(&mut self, claim: u64, window: u32) {
        let Some(w) = self
            .requester
            .claims
            .get_mut(&claim)
            .and_then(|c| c.windows.get_mut(&window))
        else {
            return;
        };
        if w.released {
            return;
        }
        w.released = true;
        match w.slot.take() {
            // A window that came over the data channel is released by freeing its slot.
            Some(slot) => self.free_slot(slot),
            None => self.emit(Out::WindowReleased(wire::WindowReleased {
                claim_id: claim,
                index: window,
            })),
        }
    }

    pub(crate) fn on_claim_terminal(&mut self, terminal: wire::ClaimTerminal) {
        let id = terminal.claim_id;
        let Some(claim) = self.requester.claims.get_mut(&id) else {
            // Unknown here (released, restarted, never heard of): there is
            // nothing of ours to drain.
            if terminal.fence_required {
                self.emit(Out::FenceAck(wire::FenceAck {
                    subject: Some(wire::fence_ack::Subject::ClaimId(id)),
                    outcome: wire::FenceOutcome::Drained as i32,
                }));
            }
            return;
        };
        if claim.terminal.is_some() {
            return;
        }
        let state = terminal.state;
        claim.terminal = Some(terminal);
        if let Some(channel) = &claim.channel {
            channel.stop();
        }
        let (restore, verdict_ok) = (claim.restore, claim.verdict_sent && claim.bad.is_none());
        let retry = !claim.granted
            && matches!(
                wire::ClaimReason::try_from(terminal.reason),
                Ok(wire::ClaimReason::RegionsHeld | wire::ClaimReason::SourceLimit)
            );
        if retry && let Some(run) = self.requester.restores.get_mut(&restore) {
            // Wait for capacity without extending this restore's deadline.
            run.claim = None;
            run.retry = Some(None);
        } else if state != wire::ClaimState::Copied as i32 || !verdict_ok {
            let error = Error::new(format!(
                "claim ended {:?}: {:?}",
                wire::ClaimState::try_from(state),
                wire::ClaimReason::try_from(terminal.reason)
            ));
            self.fail_restore(restore, Some(id), error, wire::CopyOutcome::CopyAbandoned);
            if terminal.reason == wire::ClaimReason::NoHeadroom as i32
                && let Some(failure) = self
                    .requester
                    .restores
                    .get_mut(&restore)
                    .and_then(|r| r.failure.as_mut())
            {
                failure.no_headroom = true;
            }
        }
        self.drive_claim(id);
    }

    // ---- driving a claim to its verdict, fence and removal ----------------------

    /// Once no copy of the claim is outstanding: send its verdict or fence
    /// acknowledgment, forget it when nothing more is owed, and check whether
    /// its restore is over.
    fn drive_claim(&mut self, id: u64) {
        let Some(claim) = self.requester.claims.get_mut(&id) else {
            return;
        };
        // Nothing is acknowledged while a copy reads a window or a task may still write one.
        if claim.jobs_outstanding > 0 || claim.channel_open {
            return;
        }
        let mut send = Vec::new();
        let restore = claim.restore;
        if let Some(terminal) = &claim.terminal {
            if terminal.fence_required
                && !claim.fence_sent
                && claim.peers.iter().all(|peer| peer.done.is_some())
            {
                claim.fence_sent = true;
                send.push(Out::FenceAck(wire::FenceAck {
                    subject: Some(wire::fence_ack::Subject::ClaimId(id)),
                    outcome: wire::FenceOutcome::Drained as i32,
                }));
            }
        } else if !claim.verdict_sent && claim.granted && !claim.released {
            let outcome = if let Some(bad) = claim.bad {
                Some(bad)
            } else if claim.coverage.complete()
                && claim.windows.values().all(|w| w.released)
                && (claim.ended || !claim.via_channel)
                && claim.peers.iter().all(|p| matches!(p.done, Some(Ok(()))))
            {
                Some(wire::CopyOutcome::CopyOk)
            } else {
                None
            };
            if let Some(outcome) = outcome {
                claim.verdict_sent = true;
                send.push(Out::CopyResult(wire::CopyResult {
                    claim_id: id,
                    outcome: outcome as i32,
                }));
            }
        }
        let gone = claim
            .terminal
            .as_ref()
            .is_some_and(|t| !t.fence_required || claim.fence_sent)
            && claim.peers.iter().all(|p| p.done.is_some());
        for message in send {
            self.emit(message);
        }
        self.check_restore(restore);
        if gone {
            // Slots of windows that never reached a release go back with the claim.
            if let Some(mut claim) = self.requester.claims.remove(&id) {
                for window in claim.windows.values_mut() {
                    if let Some(slot) = window.slot.take() {
                        self.free_slot(slot);
                    }
                }
            }
        }
    }

    fn check_restore(&mut self, id: u64) {
        let Some(restore) = self.requester.restores.get(&id) else {
            return;
        };
        let drained = !self.requester.claims.values().any(|c| {
            c.restore == id
                && (c.jobs_outstanding > 0
                    || c.channel_open
                    || c.peers.iter().any(|p| p.done.is_none()))
        });
        let result = if restore.failure.is_some() {
            if !drained {
                return;
            }
            Err(restore.failure.clone().expect("checked"))
        } else if drained
            && restore
                .claim
                .and_then(|id| self.requester.claims.get(&id))
                .is_some_and(|claim| {
                    claim.verdict_sent
                        && claim.bad.is_none()
                        && claim
                            .terminal
                            .as_ref()
                            .is_some_and(|t| t.state == wire::ClaimState::Copied as i32)
                })
        {
            Ok(restore.bytes)
        } else {
            return;
        };
        self.requester.restores.remove(&id);
        self.event(PoolEvent::Fetched { id, result });
        self.admit_waiting(Instant::now());
    }

    /// The restore cannot succeed. Granted claims get a verdict once their
    /// copies drain (`culprit` carries `outcome`, the others are abandoned);
    /// claims without a grant are released; queued copies are withdrawn.
    fn fail_restore(
        &mut self,
        id: u64,
        culprit: Option<u64>,
        error: Error,
        outcome: wire::CopyOutcome,
    ) {
        let Some(restore) = self.requester.restores.get_mut(&id) else {
            return;
        };
        let source = culprit.map(|_| {
            (
                restore.fetch.candidate.source.clone(),
                restore.fetch.candidate.tier,
            )
        });
        eprintln!("warning: restore {id} failed: {error}");
        restore.failure.get_or_insert(FetchFailure {
            error,
            source,
            no_headroom: false,
        });
        let claims: Vec<u64> = self
            .requester
            .claims
            .iter()
            .filter(|(_, c)| c.restore == id)
            .map(|(claim, _)| *claim)
            .collect();
        for claim_id in claims {
            self.withdraw_queued(claim_id);
            let claim = self.requester.claims.get_mut(&claim_id).expect("listed");
            // The peer ranks stop too; their `Done` come back after they have drained.
            let cancels: Vec<u64> = claim
                .peers
                .iter_mut()
                .filter(|p| p.done.is_none() && !p.cancelled)
                .map(|p| {
                    p.cancelled = true;
                    p.xfer
                })
                .collect();
            if let Some(ranks) = &mut self.ranks {
                for xfer in cancels {
                    ranks.cancel(xfer);
                }
            }
            let claim = self.requester.claims.get_mut(&claim_id).expect("listed");
            // No more windows are wanted; the task goes and reports `Closed`.
            if let Some(channel) = &claim.channel {
                channel.stop();
            }
            if claim.terminal.is_none() {
                if !claim.granted {
                    if !claim.released {
                        claim.released = true;
                        self.emit(Out::ClaimRelease(wire::ClaimRelease { claim_id }));
                    }
                } else {
                    let bad = if Some(claim_id) == culprit {
                        outcome
                    } else {
                        wire::CopyOutcome::CopyAbandoned
                    };
                    claim.bad.get_or_insert(bad);
                }
            }
            self.drive_claim(claim_id);
        }
        self.check_restore(id);
    }

    /// Withdraw the claim's copies that were queued but never submitted.
    fn withdraw_queued(&mut self, claim_id: u64) {
        let mut kept = std::collections::VecDeque::new();
        let mut released = Vec::new();
        for (job, owner) in std::mem::take(&mut self.pending) {
            match owner {
                Owner::Claim { claim, window } if claim == claim_id => {
                    if let Some(c) = self.requester.claims.get_mut(&claim) {
                        c.jobs_outstanding = c.jobs_outstanding.saturating_sub(1);
                        if let Some(w) = c.windows.get_mut(&window) {
                            w.jobs_left = w.jobs_left.saturating_sub(1);
                            if w.jobs_left == 0 {
                                released.push(window);
                            }
                        }
                    }
                }
                _ => kept.push_back((job, owner)),
            }
        }
        self.pending = kept;
        for window in released {
            self.release_window(claim_id, window);
        }
    }

    // ---- time, loss, close --------------------------------------------------------

    pub(crate) fn requester_tick(&mut self, now: Instant) {
        self.admit_waiting(now);
        let mut due = Vec::new();
        for (id, run) in &mut self.requester.restores {
            match run.retry {
                Some(None) => run.retry = Some(Some(now + CLAIM_RETRY)),
                Some(Some(at)) if run.failure.is_none() && now >= at && now < run.deadline => {
                    run.retry = None;
                    due.push(*id);
                }
                _ => {}
            }
        }
        for id in due {
            self.claim_part(id);
        }
        let late: Vec<u64> = self
            .requester
            .restores
            .iter()
            .filter(|(_, r)| r.failure.is_none() && now >= r.deadline)
            .map(|(id, _)| *id)
            .collect();
        for id in late {
            self.fail_restore(
                id,
                None,
                Error::new("the restore ran out of time"),
                wire::CopyOutcome::CopyAbandoned,
            );
        }
    }

    /// The session is gone: every restore fails; claims are treated as ended by
    /// the requester's loss, and each is fenced once its copies drain (the
    /// acknowledgment is owed to the next session).
    pub(crate) fn requester_lost(&mut self) {
        for (fetch, _) in std::mem::take(&mut self.requester.waiting) {
            self.event(PoolEvent::Fetched {
                id: fetch.id,
                result: Err(FetchFailure {
                    error: Error::new("the pool session was lost"),
                    source: None,
                    no_headroom: false,
                }),
            });
        }
        let ids: Vec<u64> = self.requester.restores.keys().copied().collect();
        let granted: Vec<u64> = self.requester.claims.keys().copied().collect();
        for id in granted {
            if let Some(claim) = self.requester.claims.get_mut(&id)
                && claim.terminal.is_none()
            {
                claim.terminal = Some(wire::ClaimTerminal {
                    claim_id: id,
                    state: wire::ClaimState::Failed as i32,
                    reason: wire::ClaimReason::RequesterLost as i32,
                    fence_required: claim.granted,
                });
            }
        }
        for id in ids {
            self.fail_restore(
                id,
                None,
                Error::new("the pool session was lost"),
                wire::CopyOutcome::CopyAbandoned,
            );
        }
        let all: Vec<u64> = self.requester.claims.keys().copied().collect();
        for id in all {
            self.withdraw_queued(id);
            self.drive_claim(id);
        }
    }

    pub(crate) fn requester_close(&mut self) {
        let ids: Vec<u64> = self.requester.restores.keys().copied().collect();
        for id in ids {
            self.fail_restore(
                id,
                None,
                Error::new("the pool client is closing"),
                wire::CopyOutcome::CopyAbandoned,
            );
        }
    }

    pub(crate) fn requester_quarantine(&mut self) {
        let ids: Vec<u64> = self.requester.claims.keys().copied().collect();
        for id in ids {
            self.emit(Out::FenceAck(wire::FenceAck {
                subject: Some(wire::fence_ack::Subject::ClaimId(id)),
                outcome: wire::FenceOutcome::Quarantined as i32,
            }));
        }
        // Their copies may still run: nothing is freed, nothing is reported.
        std::mem::forget(std::mem::take(&mut self.requester.claims));
        std::mem::forget(std::mem::take(&mut self.requester.restores));
    }
}

fn records_match(expected: &[ExpectedRecord], granted: &[wire::Record]) -> bool {
    expected.len() == granted.len()
        && expected.iter().zip(granted).all(|(want, got)| {
            got.key.as_ref().and_then(codec::key_from) == Some(want.key)
                && got.first_page == want.first_page
                && got.pages == want.pages
                && got.row == want.row
        })
}

#[cfg(test)]
mod tests {
    use super::super::super::wire::server_message::M as In;
    use super::super::tests::{Rig, fetched, records, single_part, terminal};
    use super::*;

    #[test]
    fn a_terminal_before_peer_done_waits_for_its_drain_or_quarantine() {
        for shared in [false, true] {
            for quarantined in [false, true] {
                let mut rig = Rig::new();
                rig.open();
                let fetch = single_part();
                let out = rig.command(super::super::super::shared::Command::Fetch(fetch.clone()));
                let claim = out
                    .iter()
                    .find_map(|message| match &message.m {
                        Some(Out::Claim(claim)) => Some(claim.claim_id),
                        _ => None,
                    })
                    .unwrap();
                rig.send(In::ClaimGranted(wire::ClaimGranted {
                    claim_id: claim,
                    records: records(&fetch),
                    total_pages: fetch.part.part.pages as u64,
                    row: fetch.part.part.row,
                    retained: true,
                    window_credits: 2,
                    ..Default::default()
                }));
                rig.engine
                    .requester
                    .claims
                    .get_mut(&claim)
                    .unwrap()
                    .peers
                    .push(PeerPart {
                        rank: 1,
                        xfer: 1,
                        shared,
                        done: None,
                        quarantined: false,
                        cancelled: false,
                    });
                let out = rig.send(terminal(
                    claim,
                    wire::ClaimState::Failed,
                    wire::ClaimReason::RankLost,
                    true,
                ));
                assert!(
                    !out.iter()
                        .any(|message| matches!(message.m, Some(Out::FenceAck(_))))
                );
                assert!(fetched(&rig.events()).is_empty());
                rig.engine
                    .on_peer_done(claim, 1, Err("the rank stopped".into()), quarantined);
                let out = rig.engine.take_out();
                let fences: Vec<_> = out
                    .iter()
                    .filter_map(|message| match &message.m {
                        Some(Out::FenceAck(fence)) => Some(fence),
                        _ => None,
                    })
                    .collect();
                assert_eq!(fences.len(), 1);
                assert_eq!(
                    fences[0].outcome,
                    if quarantined {
                        wire::FenceOutcome::Quarantined
                    } else {
                        wire::FenceOutcome::Drained
                    } as i32
                );
                if quarantined {
                    assert!(fetched(&rig.events()).is_empty());
                } else {
                    assert!(fetched(&rig.events())[0].1.is_err());
                }
            }
        }
    }
}
