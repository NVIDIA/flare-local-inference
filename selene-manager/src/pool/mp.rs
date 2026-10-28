//! Multi-process instances in claims: peer participants, their data-channel legs, the
//! assembly pass and the in-order delivery of windows.
//!
//! The bookkeeping that is the same for every claim stays in `claims.rs`; what a rank adds
//! lives here. A claim whose requester is a multi-process instance has one *leg* per peer
//! rank in channel mode, which carries only that rank's slices, and one *participant* per
//! peer rank in shared mode, which reads its slices out of the manager's windows in place
//! and answers for itself under its own process identity.

use super::{
    claims::{Channel, ClaimKey, Owner, Participant, Party, Phase, attest},
    core::{ChanOwner, Core, Event, ProcKey},
    pass::{PassError, PassJob, PassOut},
};
use crate::{
    data::channel,
    proto::v2::{self, server_message::M},
};
use flare_selene::shared::region::Region;
use flare_selene::shared::{SliceLayout, SliceSums};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

/// A peer rank that reads or writes the manager's windows in place.
pub struct MpPeer {
    pub rank: u32,
    /// The agent's own identity: what it may still be touching is its own obligation.
    pub proc: ProcKey,
    pub shared: bool,
}

/// One side (requester or source) of a claim that is a multi-process instance.
pub struct MpSide {
    pub layout: Arc<SliceLayout>,
    /// Ranks `1..world`, in order.
    pub peers: Vec<MpPeer>,
}

pub fn rank_sums(sums: Vec<v2::RankSums>) -> Vec<(usize, SliceSums)> {
    sums.into_iter()
        .map(|s| {
            (
                s.rank as usize,
                SliceSums {
                    pages: s.page_xxh3,
                    row: s.has_row.then_some(s.row_xxh3),
                },
            )
        })
        .collect()
}

pub fn accept_parts(
    stated: &mut Vec<Option<SliceSums>>,
    writers: &mut u64,
    parts: Vec<(usize, SliceSums)>,
    slot: u32,
    pages: u32,
    row: bool,
) -> Result<Option<Vec<SliceSums>>, ()> {
    let bit = 1u64.checked_shl(slot).ok_or(())?;
    if *writers & bit == 0 {
        return Ok(None);
    }
    for (rank, sums) in parts {
        if sums.pages.len() != pages as usize || sums.row.is_some() != row {
            return Err(());
        }
        match stated.get_mut(rank) {
            Some(entry @ None) => *entry = Some(sums),
            _ => return Err(()),
        }
    }
    *writers &= !bit;
    if *writers != 0 {
        return Ok(None);
    }
    stated
        .drain(..)
        .collect::<Option<Vec<_>>>()
        .map(Some)
        .ok_or(())
}

/// One root or peer data-channel leg, on the source or requester side.
pub struct Leg {
    pub rank: u32,
    /// The leg of the claim's source (a put-kind transfer) rather than its requester.
    pub source: bool,
    pub ch: Channel,
}

/// Windows are delivered in index order whatever order their passes finish in.
#[derive(Default)]
pub struct Delivery {
    pub next: u32,
    pub ready: BTreeMap<u32, PassOut>,
}

/// What a claim tracks for multi-process instances.
pub struct MpState {
    pub req: MpSide,
    pub src: MpSide,
    pub deliver: Delivery,
    /// Passes running on the claim's windows: they read manager memory, so the windows are
    /// not reclaimed until they are over.
    pub passes: u32,
}

impl MpSide {
    pub fn whole(layout: &SliceLayout) -> Self {
        Self {
            layout: Arc::new(layout.whole()),
            peers: Vec::new(),
        }
    }
}

impl MpState {
    pub fn new(req: MpSide, src: MpSide) -> Self {
        Self {
            req,
            src,
            deliver: Delivery::default(),
            passes: 0,
        }
    }
}

impl Core {
    /// The multi-process side of a claim for the session `sid`: one participant per peer
    /// rank, and a leg opened for each that is in channel mode.
    pub(super) fn mp_side(
        &mut self,
        sid: u64,
        owner: &dyn Fn(usize) -> ChanOwner,
        kind: channel::Kind,
        legs: &mut Vec<Leg>,
    ) -> Result<MpSide, v2::ClaimReason> {
        let Some(session) = self.sessions.get(&sid) else {
            return Err(v2::ClaimReason::RanksDetached);
        };
        let world = session.ranks.as_ref().map_or(1, |r| r.world);
        let (instance, rank0) = (session.instance, session.nonce);
        let layout = self.requester_slices(sid);
        let mut peers = Vec::new();
        for rank in 1..world {
            let Some(agent) = self.agents.values().find(|a| {
                a.instance == instance && a.rank0 == rank0 && a.rank == rank && !a.drain_only
            }) else {
                self.discard_legs(legs.drain(..).map(|l| l.ch));
                return Err(v2::ClaimReason::RanksDetached);
            };
            let shared = agent.mode == v2::RankMode::RankShared;
            let proc = agent.proc();
            if !shared {
                match self.open_leg(kind, layout.clone(), rank as usize) {
                    Ok(ch) => {
                        self.channels.insert(ch.transfer, owner(legs.len()));
                        legs.push(Leg {
                            rank,
                            source: kind == channel::Kind::Put,
                            ch,
                        });
                    }
                    Err(_) => {
                        self.discard_legs(legs.drain(..).map(|l| l.ch));
                        return Err(v2::ClaimReason::NoWindows);
                    }
                }
            }
            peers.push(MpPeer { rank, proc, shared });
        }
        Ok(MpSide { layout, peers })
    }

    /// Drop legs that no one has connected to.
    pub(super) fn discard_legs(&mut self, legs: impl IntoIterator<Item = Channel>) {
        for leg in legs {
            self.channels.remove(&leg.transfer);
            if let Some(hub) = &self.cfg.channel {
                hub.stop(leg.transfer);
                hub.forget(leg.transfer);
            }
        }
    }

    /// The channels the requester's peers in channel mode dial, for `ClaimGranted`.
    pub(super) fn rank_channels(&self, key: ClaimKey) -> Vec<v2::RankChannel> {
        self.rank_channels_of(key, false)
    }

    /// The channels the peers of the requester (`source` false) or of the source (true) dial.
    pub(super) fn rank_channels_of(&self, key: ClaimKey, source: bool) -> Vec<v2::RankChannel> {
        let (Some(claim), Some(hub)) = (self.claims.get(&key), self.cfg.channel.as_ref()) else {
            return Vec::new();
        };
        claim
            .legs
            .iter()
            .filter(|leg| leg.rank != 0)
            .filter(|leg| leg.source == source)
            .map(|leg| v2::RankChannel {
                rank: leg.rank,
                channel: Some(super::channel_ref(hub, &leg.ch.token)),
            })
            .collect()
    }

    /// A window is whole in the manager's memory and its source has stated its checksums.
    /// An ordinary claim hands it on at once; one with a multi-process side goes through
    /// the assembly pass first.
    pub(super) fn window_arrived(&mut self, key: ClaimKey, index: usize, stated: Vec<SliceSums>) {
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        if claim.mp.req.layout.world() == 1 && claim.mp.src.layout.world() == 1 {
            return self.deliver(key, index, stated);
        }
        let Some(regions) = self.cfg.regions.clone() else {
            eprintln!("selene: a multi-process claim needs the manager's regions");
            return self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
        };
        let window = &claim.windows[index];
        let Some(region) = u8::try_from(window.region).ok().and_then(Region::from_u8) else {
            return self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
        };
        let layout = if claim.mp.req.layout.world() > 1 {
            &claim.mp.req.layout
        } else {
            &claim.mp.src.layout
        }
        .clone();
        let (pages, row) = (window.plan.page_count, window.plan.row);
        let (stated_slices, stated_whole) = if claim.mp.src.layout.world() > 1 {
            (Some(stated), None)
        } else {
            (None, stated.into_iter().next())
        };
        let job = PassJob {
            regions,
            region,
            offset: window.offset,
            pages,
            row,
            layout,
            stated_slices,
            stated_whole,
        };
        claim.mp.passes += 1;
        let events = self.self_tx.clone();
        let index = index as u32;
        tokio::task::spawn_blocking(move || {
            let result = super::pass::run(job);
            let _ = events.blocking_send(Event::Pass { key, index, result });
        });
    }

    /// Each accessor states its rank slices once, for both exports and puts.
    pub(super) fn window_parts(
        &mut self,
        owner: ChanOwner,
        index: u32,
        parts: Vec<(usize, SliceSums)>,
        slot: u32,
    ) {
        let fields = match owner {
            ChanOwner::Source(key) => self.claims.get_mut(&key).and_then(|c| {
                let w = c.windows.get_mut(index as usize)?;
                (w.phase == Phase::SourceWrite).then_some((
                    &mut w.stated,
                    &mut w.writers,
                    &mut w.arrived,
                    w.plan.page_count,
                    w.plan.row,
                ))
            }),
            ChanOwner::Put(key) => self.puts.get_mut(&key).and_then(|r| {
                if r.phase != super::store_bridge::PutPhase::Open {
                    return None;
                }
                let w = r.mp.windows.get_mut(&index)?;
                Some((
                    &mut w.stated,
                    &mut w.writers,
                    &mut w.arrived,
                    w.pages,
                    w.row,
                ))
            }),
            _ => None,
        };
        let Some((stated, writers, arrived, pages, row)) = fields else {
            return;
        };
        if *arrived {
            return;
        }
        let sums = match accept_parts(stated, writers, parts, slot, pages, row) {
            Ok(Some(sums)) => sums,
            Ok(None) => return,
            Err(()) => {
                return match owner {
                    ChanOwner::Source(key) => {
                        self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError)
                    }
                    ChanOwner::Put(key) => self.put_fail_by_key(
                        key,
                        v2::Reason::Invalid,
                        "a rank's slice checksums do not fit its window",
                    ),
                    _ => unreachable!(),
                };
            }
        };
        *arrived = true;
        match owner {
            ChanOwner::Source(key) => {
                let claim = self.claims.get_mut(&key).unwrap();
                if claim.is_terminal() {
                    claim.windows[index as usize].phase = Phase::Free;
                    claim.windows[index as usize].extent = None;
                    self.reclaim(key);
                } else {
                    self.window_arrived(key, index as usize, sums);
                }
            }
            ChanOwner::Put(key) => self.put_window_arrived(key, index, sums),
            _ => unreachable!(),
        }
    }

    /// A pass finished.
    pub(super) fn pass_done(
        &mut self,
        key: ClaimKey,
        index: u32,
        result: Result<PassOut, PassError>,
    ) {
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        claim.mp.passes = claim.mp.passes.saturating_sub(1);
        if claim.is_terminal() {
            // Nobody wants the window now; its memory is free once the pass is over.
            return self.reclaim(key);
        }
        match result {
            Ok(out) => {
                claim.mp.deliver.ready.insert(index, out);
            }
            Err(PassError::Mismatch(why)) => {
                eprintln!("selene: claim {}: {why}", key.2);
                return self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::Checksum);
            }
            Err(PassError::Io(why)) => {
                eprintln!("selene: claim {}: the assembly pass failed: {why}", key.2);
                return self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
            }
        }
        loop {
            let Some(claim) = self.claims.get_mut(&key) else {
                return;
            };
            if claim.is_terminal() {
                return;
            }
            let next = claim.mp.deliver.next;
            let Some(ready) = claim.mp.deliver.ready.remove(&next) else {
                return;
            };
            claim.mp.deliver.next += 1;
            let sums = if self.claims[&key].mp.req.layout.world() == 1 {
                vec![ready.whole]
            } else {
                ready.slices
            };
            self.deliver(key, next as usize, sums);
        }
    }

    /// Deliver a filled window through the requester's session and rank channels.
    pub(super) fn deliver(&mut self, key: ClaimKey, index: usize, sums: Vec<SliceSums>) {
        let Some(claim) = self.claims.get_mut(&key) else {
            return;
        };
        let layout = claim.mp.req.layout.clone();
        let shared_ranks: Vec<_> = std::iter::once(0)
            .chain(
                claim
                    .mp
                    .req
                    .peers
                    .iter()
                    .filter(|p| p.shared)
                    .map(|p| p.rank),
            )
            .collect();
        let spec = claim.spec(index);
        let Some(region) = u8::try_from(spec.region).ok().and_then(Region::from_u8) else {
            return self.finish(key, v2::ClaimState::Failed, v2::ClaimReason::CopyError);
        };
        let window = &mut claim.windows[index];
        window.phase = Phase::RequesterRead;
        window.readers = 1 | leg_readers(&claim.legs);
        window.arrived = true;
        claim.state = v2::ClaimState::Granted;
        claim.last_progress = Instant::now();
        let sid = claim.requester_sid;
        let channel = claim.channel().is_some();
        if let Some(hub) = &self.cfg.channel {
            for leg in claim.legs.iter().filter(|leg| !leg.source) {
                let rank = leg.rank as usize;
                hub.push_window(
                    leg.ch.transfer,
                    channel::WindowOut {
                        shape: super::claims::shape_for(&spec, &layout, rank),
                        page_xxh3: sums[rank].pages.clone(),
                        row_xxh3: sums[rank].row.unwrap_or(0),
                        region,
                        offset: spec.offset,
                    },
                );
            }
        }
        if !channel {
            let mut message = v2::WindowReady {
                claim_id: key.2,
                window: Some(spec),
                ..Default::default()
            };
            if layout.world() > 1 {
                message.rank_sums = shared_ranks
                    .into_iter()
                    .map(|rank| v2::RankSums {
                        rank,
                        page_xxh3: sums[rank as usize].pages.clone(),
                        row_xxh3: sums[rank as usize].row.unwrap_or(0),
                        has_row: sums[rank as usize].row.is_some(),
                    })
                    .collect();
            } else {
                let whole = sums.into_iter().next().unwrap();
                message.page_xxh3 = whole.pages;
                message.row_xxh3 = whole.row.unwrap_or(0);
                message.has_row = whole.row.is_some();
            }
            self.send(sid, M::WindowReady(message));
        }
        self.finish_req_legs_if_delivered(key);
    }

    /// Once every window is on its way, each leg ends its transfer after the ones queued.
    pub(super) fn finish_req_legs_if_delivered(&mut self, key: ClaimKey) {
        let Some(claim) = self.claims.get(&key) else {
            return;
        };
        let all = claim
            .windows
            .iter()
            .all(|w| !matches!(w.phase, Phase::Unissued | Phase::SourceWrite));
        // A claim on the store's records ends its transfers when the store is done.
        if !all || claim.windows.is_empty() || claim.manager.is_some() {
            return;
        }
        if let Some(hub) = &self.cfg.channel {
            for leg in claim.legs.iter().filter(|leg| !leg.source) {
                hub.finish(leg.ch.transfer);
            }
        }
    }
}

/// A window of a put, while its ranks fill it.
pub struct PutWin {
    pub region: Region,
    /// The record-relative page this window's pages start at.
    pub first_page: u32,
    pub offset: u64,
    pub pages: u32,
    pub row: bool,
    /// Writers still to report, one bit per accessor like a claim window's.
    pub writers: u64,
    pub stated: Vec<Option<SliceSums>>,
    pub arrived: bool,
}

/// What a put adds when its writer is a multi-process instance.
pub struct PutMp {
    pub layout: Arc<SliceLayout>,
    pub peers: Vec<MpPeer>,
    pub windows: BTreeMap<u32, PutWin>,
    /// Passes running on the put's windows.
    pub passes: u32,
    /// The whole-page checksums the passes computed, by record page, and the row's: what
    /// the record keeps, whatever the ranks' own checksums were.
    pub whole: Vec<Option<u64>>,
    pub row: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_rank_statements_cannot_complete_a_window() {
        let sums = SliceSums {
            pages: vec![7],
            row: None,
        };
        let mut stated = vec![None];
        let mut writers = 1;
        assert!(
            accept_parts(
                &mut stated,
                &mut writers,
                vec![(0, sums.clone()), (0, sums)],
                0,
                1,
                false
            )
            .is_err()
        );
        assert_eq!(writers, 1, "a rejected batch must not discharge its writer");
    }
}

/// Rank zero's channels precede the peer legs; peer indices are bound after assembly.
pub fn join_legs(requester: Option<Channel>, source: Option<Channel>, peers: Vec<Leg>) -> Vec<Leg> {
    [
        requester.map(|ch| Leg {
            rank: 0,
            source: false,
            ch,
        }),
        source.map(|ch| Leg {
            rank: 0,
            source: true,
            ch,
        }),
    ]
    .into_iter()
    .flatten()
    .chain(peers)
    .collect()
}
pub fn root(legs: &[Leg], source: bool) -> Option<&Channel> {
    legs.iter()
        .take(2)
        .find(|leg| leg.rank == 0 && leg.source == source)
        .map(|leg| &leg.ch)
}
pub fn leg_readers(legs: &[Leg]) -> u64 {
    legs.iter()
        .filter(|leg| !leg.source)
        .fold(0, |mask, leg| mask | 1 << leg.rank)
}
/// Stop every channel through the same handler-exit fence.
pub fn stop_legs(
    legs: &[Leg],
    parties: &mut [Participant],
    hub: Option<&channel::Hub>,
    owners: &mut std::collections::HashMap<u64, ChanOwner>,
) {
    for leg in legs {
        if super::claims::party_state(parties, Owner::Leg(leg.ch.transfer)) != Party::Drained
            && hub.is_none_or(|hub| hub.stop(leg.ch.transfer))
        {
            attest(parties, Owner::Leg(leg.ch.transfer), Party::Drained, false);
            owners.remove(&leg.ch.transfer);
        }
    }
}
