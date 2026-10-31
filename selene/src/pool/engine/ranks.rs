/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rank 0's side of a multi-process instance: the links to the peer ranks' agents and the
//! transfers it fans out over them.
//!
//! Rank 0 decides; an agent only moves its own slice. Every transfer is bound to the
//! manager incarnation and attachment generation the agent last reported, and an agent
//! ends each transfer with exactly one `Done` after it has drained. Nothing is released on
//! rank 0 (destination pages, pins) until every rank's `Done` is in: a rank whose drain can
//! never be established is quarantined and stops the pool instead.

use std::{collections::HashMap, sync::Arc};

use crate::{Error, Result};

use super::super::{
    rank::{self as rank, Incoming, Msg, RankLink, RankSet},
    slice::{SliceLayout, SliceSums},
    wire,
};

/// What the agents say that the engine acts on.
#[derive(Debug)]
pub(crate) enum RankEvent {
    WindowDone {
        rank: usize,
        xfer: u64,
        index: u32,
        result: std::result::Result<(), String>,
        sums: Option<SliceSums>,
    },
    Done {
        rank: usize,
        xfer: u64,
        result: std::result::Result<(), String>,
        quarantined: bool,
    },
    /// The rank's agent will not use the pool (for instance its registration was not
    /// approved).
    Disabled { rank: usize, reason: String },
    /// The rank attached to the manager.
    Attached,
    /// The attachment of a rank changed or ended; its open transfers are cancelled by the
    /// agent and their `Done` follow.
    Detached,
    /// The link failed: nothing more will come from this rank, drained or not.
    Lost { rank: usize, reason: String },
}

/// A peer rank's part of a claim.
pub(crate) struct PeerPart {
    pub(crate) rank: usize,
    pub(crate) xfer: u64,
    pub(crate) shared: bool,
    /// The agent's `Done`: the rank has drained and covered its slice (or failed).
    pub(crate) done: Option<std::result::Result<(), String>>,
    /// The agent could not prove it drained.
    pub(crate) quarantined: bool,
    pub(crate) cancelled: bool,
}

/// A window shared peers have not all drained, and the slice checksums of those that did.
pub(crate) struct PeerWindow {
    pub(crate) left: std::collections::BTreeSet<usize>,
    pub(crate) sums: Vec<(usize, crate::shared::SliceSums)>,
}

#[derive(Default)]
pub(super) struct PeerSet {
    pub parts: Vec<PeerPart>,
    pub windows: HashMap<u32, PeerWindow>,
}

impl PeerSet {
    pub fn take_uncancelled(&mut self) -> Vec<u64> {
        self.parts
            .iter_mut()
            .filter(|p| p.done.is_none() && !p.cancelled)
            .map(|p| {
                p.cancelled = true;
                p.xfer
            })
            .collect()
    }

    pub fn window_done(
        &mut self,
        index: u32,
        rank: usize,
        result: std::result::Result<(), String>,
        sums: Option<SliceSums>,
    ) -> Option<std::result::Result<(), String>> {
        let window = self.windows.get_mut(&index)?;
        if !window.left.remove(&rank) {
            return None;
        }
        Some(match (result, sums) {
            (Ok(()), Some(sums)) => {
                window.sums.push((rank, sums));
                Ok(())
            }
            (Err(why), _) => Err(why),
            (Ok(()), None) => Err("the rank stated no checksums".into()),
        })
    }

    pub fn done(
        &mut self,
        rank: usize,
        result: &std::result::Result<(), String>,
        quarantined: bool,
    ) {
        if let Some(peer) = self
            .parts
            .iter_mut()
            .find(|p| p.rank == rank && p.done.is_none())
        {
            peer.done = Some(result.clone());
            peer.quarantined = quarantined;
        }
    }
}

/// Who a transfer belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum XferOwner {
    Claim(u64),
    Export(u64),
    Put(u64),
}

#[derive(Debug)]
enum State {
    /// Hello sent, waiting for the agent's.
    Greeting,
    /// Bound; waiting for the agent's attachment to the manager.
    Bound,
    Attached {
        incarnation: u64,
        generation: u64,
        mode: wire::RankMode,
    },
    Detached,
    Lost,
}

struct Peer {
    end: RankLink,
    state: State,
}

/// What rank 0 tells its peers about the instance.
pub(crate) struct Identity {
    pub secret: [u8; 32],
    pub nonce: [u8; 16],
    pub fingerprint: [u8; 16],
    /// The GPUs of the instance; an agent's must be one of them.
    pub gpus: Vec<[u8; 16]>,
}

pub(crate) struct Ranks {
    pub layout: Arc<SliceLayout>,
    secret: [u8; 32],
    instance_id: Option<[u8; 16]>,
    fingerprint: [u8; 16],
    /// The GPUs of the instance; an agent's must be one of them.
    gpus: Vec<[u8; 16]>,
    peers: Vec<Peer>,
    next_xfer: u64,
    /// Open transfers: owner and rank.
    xfers: HashMap<u64, (XferOwner, usize)>,
}

impl Ranks {
    pub(super) fn open_peers(
        &mut self,
        owner: XferOwner,
        channels: &[wire::RankChannel],
        template: rank::Open,
        opened: &mut Vec<PeerPart>,
    ) -> std::result::Result<(), String> {
        for rank in 1..self.world() {
            let channel = channels
                .iter()
                .find(|c| c.rank as usize == rank)
                .and_then(|c| c.channel.clone());
            let shared = channel.is_none();
            let expected = if shared {
                wire::RankMode::RankShared
            } else {
                wire::RankMode::RankChannel
            };
            let mut open = template.clone();
            open.channel = channel;
            let xfer = (self.mode(rank) == Some(expected))
                .then(|| self.open(rank, owner, open))
                .flatten();
            let Some(xfer) = xfer else {
                for peer in opened.iter_mut() {
                    self.cancel(peer.xfer);
                    peer.cancelled = true;
                }
                return Err(format!("rank {rank} is not attached in the granted mode"));
            };
            opened.push(PeerPart {
                rank,
                xfer,
                shared,
                done: None,
                quarantined: false,
                cancelled: false,
            });
        }
        Ok(())
    }

    /// Start the links. `set[i]` reaches rank `i + 1`; Hello goes out at once.
    pub(crate) fn new(set: RankSet, layout: Arc<SliceLayout>, who: Identity) -> Result<Self> {
        let Identity {
            secret,
            nonce,
            fingerprint,
            gpus,
        } = who;
        let world = set.len() + 1;
        if layout.world() != world {
            return Err(Error::new(
                "the scope's rank count does not match the rank links",
            ));
        }
        let mut peers = Vec::with_capacity(set.len());
        for (index, link) in set.into_iter().enumerate() {
            let mut end = link;
            end.send_rank(Msg::Hello {
                rank: (index + 1) as u32,
                world: world as u32,
                rank0_nonce: nonce.to_vec(),
                layout: fingerprint.to_vec(),
            });
            peers.push(Peer {
                end,
                state: State::Greeting,
            });
        }
        Ok(Self {
            layout,
            secret,
            instance_id: None,
            fingerprint,
            gpus,
            peers,
            next_xfer: 0,
            xfers: HashMap::new(),
        })
    }

    pub(crate) fn bind(&mut self, instance_id: [u8; 16]) {
        if self.instance_id.replace(instance_id).is_none() {
            for peer in &mut self.peers {
                if matches!(peer.state, State::Bound) {
                    peer.end.send_rank(Msg::Bind {
                        rank_secret: self.secret.to_vec(),
                        instance_id: instance_id.to_vec(),
                    });
                }
            }
        }
    }

    pub(crate) fn world(&self) -> usize {
        self.peers.len() + 1
    }

    /// Every peer is attached to the manager incarnation `incarnation`.
    pub(crate) fn attached(&self, incarnation: u64) -> bool {
        self.peers.iter().all(
            |peer| matches!(peer.state, State::Attached { incarnation: i, .. } if i == incarnation),
        )
    }

    /// The mode the manager gave `rank`'s agent, once attached.
    pub(crate) fn mode(&self, rank: usize) -> Option<wire::RankMode> {
        match self.peers.get(rank.checked_sub(1)?)?.state {
            State::Attached { mode, .. } => Some(mode),
            _ => None,
        }
    }

    /// Start a transfer on `rank`: the next id, bound to the agent's current attachment.
    /// `None` when the rank is not attached.
    pub(crate) fn open(
        &mut self,
        rank: usize,
        owner: XferOwner,
        mut open: rank::Open,
    ) -> Option<u64> {
        let peer = self.peers.get(rank.checked_sub(1)?)?;
        let State::Attached {
            incarnation,
            generation,
            ..
        } = peer.state
        else {
            return None;
        };
        self.next_xfer += 1;
        let xfer = self.next_xfer;
        open.xfer = xfer;
        open.manager_incarnation = incarnation;
        open.generation = generation;
        if !self.peers[rank - 1].end.send_rank(Msg::Open(open)) {
            return None;
        }
        self.xfers.insert(xfer, (owner, rank));
        Some(xfer)
    }

    pub(crate) fn send_window(&mut self, rank: usize, window: rank::Window) -> bool {
        self.peers
            .get_mut(rank - 1)
            .is_some_and(|peer| peer.end.send_rank(Msg::Window(window)))
    }

    pub(crate) fn cancel(&mut self, xfer: u64) {
        if let Some((_, rank)) = self.xfers.get(&xfer) {
            self.peers[rank - 1].end.send_rank(Msg::Cancel { xfer });
        }
    }

    /// The transfer is over (its `Done` arrived or its rank is lost).
    pub(crate) fn forget(&mut self, xfer: u64) {
        self.xfers.remove(&xfer);
    }

    pub(crate) fn owner(&self, xfer: u64) -> Option<(XferOwner, usize)> {
        self.xfers.get(&xfer).copied()
    }

    /// Open transfers of `rank`.
    pub(crate) fn open_on(&self, rank: usize) -> Vec<u64> {
        self.xfers
            .iter()
            .filter(|(_, (_, r))| *r == rank)
            .map(|(x, _)| *x)
            .collect()
    }

    /// Every open transfer, on any rank.
    /// Tell every agent the pool is off for this instance.
    pub(crate) fn close_all(&mut self) {
        for peer in &mut self.peers {
            peer.end.send_rank(Msg::Close);
        }
    }

    pub(crate) fn idle(&self) -> bool {
        self.xfers.is_empty()
    }

    /// Collect what the links delivered since the last call.
    pub(crate) fn poll(&mut self) -> Vec<RankEvent> {
        let mut events = Vec::new();
        for index in 0..self.peers.len() {
            let rank = index + 1;
            let mut incoming = Vec::new();
            self.peers[index].end.poll_rank(&mut incoming);
            for item in incoming {
                match item {
                    Incoming::Closed(reason) => {
                        if !matches!(self.peers[index].state, State::Lost) {
                            self.peers[index].state = State::Lost;
                            events.push(RankEvent::Lost { rank, reason });
                        }
                    }
                    Incoming::Message(message) => self.message(index, message, &mut events),
                }
            }
        }
        events
    }

    fn lose(&mut self, index: usize, reason: String, events: &mut Vec<RankEvent>) {
        self.peers[index].state = State::Lost;
        events.push(RankEvent::Lost {
            rank: index + 1,
            reason,
        });
    }

    fn message(&mut self, index: usize, message: Msg, events: &mut Vec<RankEvent>) {
        let rank = index + 1;
        match message {
            Msg::AgentHello {
                rank: offered_rank,
                world,
                layout,
                gpu_uuid,
                agent_nonce,
            } => {
                let fine = matches!(self.peers[index].state, State::Greeting)
                    && offered_rank as usize == rank
                    && world as usize == self.world()
                    && layout == self.fingerprint
                    && agent_nonce.len() == 16
                    && self
                        .gpus
                        .iter()
                        .any(|gpu| gpu.as_slice() == gpu_uuid.as_slice());
                if !fine {
                    return self.lose(
                        index,
                        format!("rank {rank}'s agent disagrees about the instance"),
                        events,
                    );
                }
                self.peers[index].state = State::Bound;
                if let Some(instance_id) = self.instance_id {
                    self.peers[index].end.send_rank(Msg::Bind {
                        rank_secret: self.secret.to_vec(),
                        instance_id: instance_id.to_vec(),
                    });
                }
            }
            Msg::Attached {
                manager_incarnation,
                generation,
                mode,
            } => {
                if self.instance_id.is_none()
                    || matches!(self.peers[index].state, State::Greeting | State::Lost)
                {
                    return self.lose(
                        index,
                        format!("rank {rank} attached before its hello"),
                        events,
                    );
                }
                let mode = wire::RankMode::try_from(mode).unwrap_or(wire::RankMode::Unspecified);
                self.peers[index].state = State::Attached {
                    incarnation: manager_incarnation,
                    generation,
                    mode,
                };
                events.push(RankEvent::Attached);
            }
            Msg::Disabled { reason } => {
                self.peers[index].state = State::Lost;
                events.push(RankEvent::Disabled { rank, reason });
            }
            Msg::Detached { .. } => {
                if !matches!(self.peers[index].state, State::Lost) {
                    self.peers[index].state = State::Detached;
                    events.push(RankEvent::Detached);
                }
            }
            // A peer's close is final: the link ends right after it. Record
            // the rank as lost without the warning a lost link would raise.
            Msg::Close => self.peers[index].state = State::Lost,
            Msg::WindowDone {
                xfer,
                index,
                ok,
                error,
                sums,
            } => {
                let result = if ok { Ok(()) } else { Err(error) };
                events.push(RankEvent::WindowDone {
                    rank,
                    xfer,
                    index,
                    result,
                    sums,
                });
            }
            Msg::Done {
                xfer,
                ok,
                error,
                quarantined,
            } => {
                events.push(RankEvent::Done {
                    rank,
                    xfer,
                    result: if ok { Ok(()) } else { Err(error) },
                    quarantined,
                });
            }
            _ => self.lose(
                index,
                format!("rank {rank} sent a message only rank 0 may"),
                events,
            ),
        }
    }
}

impl super::Engine {
    /// Start every peer rank's part of an export about to be pinned. The pages are the
    /// chain's, in the order the records are covered.
    pub(super) fn open_export_peers(
        &mut self,
        id: u64,
        records: &[wire::Record],
        chain: Option<&crate::ExportChain>,
        rank_channels: &[wire::RankChannel],
    ) -> std::result::Result<(), String> {
        let Some(ranks) = self.ranks.as_mut() else {
            return Ok(());
        };
        // `None`: a host-tier export, whose peers read their own record files.
        let mut pages = Vec::new();
        for (index, node) in chain.iter().flat_map(|c| c.nodes.iter().enumerate()) {
            let chain = chain.expect("a chain has nodes");
            let first = if index == 0 { chain.skip_pages } else { 0 } as usize;
            let count = records.get(index).map_or(0, |record| record.pages as usize);
            let Some(slice) = node.pages.get(first..first + count) else {
                return Err("a chain node has fewer pages than its record".into());
            };
            pages.extend(slice.iter().map(|page| page.0));
        }
        let (page_bytes, row_bytes) = (self.st.page_record_bytes(), self.st.row_record_bytes());
        let mut opened: Vec<PeerPart> = Vec::new();
        let open = rank::Open {
            out: true,
            kind: wire::rank_fence::Kind::Export as i32,
            fence_id: id,
            records: records.to_vec(),
            pages: pages.clone(),
            has_slot: chain.is_some_and(|c| c.row.is_some()),
            slot: chain.and_then(|c| c.row).map_or(0, |slot| slot.0),
            page_bytes,
            row_bytes,
            host: chain.is_none(),
            ..Default::default()
        };
        let result = ranks.open_peers(XferOwner::Export(id), rank_channels, open, &mut opened);
        if let Some(out) = self.source.outbound(id) {
            out.peers.parts = opened;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pool::rank::memory_links,
        shared::{Geometry, RankPlanes},
    };
    use std::time::Duration;

    #[test]
    fn an_agent_is_bound_only_after_hello_names_the_pool() {
        let (zero, mut agent) = memory_links();
        let layout = Arc::new(
            SliceLayout::new(&Geometry {
                page_tokens: 64,
                ranks: vec![
                    RankPlanes {
                        plane_bytes: vec![4096],
                        row_bytes: 4096
                    };
                    2
                ],
                page_record_bytes: 8192,
                row_record_bytes: 8192,
            })
            .unwrap(),
        );
        let mut ranks = Ranks::new(
            vec![zero],
            layout,
            Identity {
                secret: [1; 32],
                nonce: [2; 16],
                fingerprint: [3; 16],
                gpus: vec![[4; 16]],
            },
        )
        .unwrap();
        agent.expect(12, Duration::from_secs(2)).unwrap(); // Rank-zero Hello.
        let mut events = Vec::new();
        ranks.message(
            0,
            Msg::AgentHello {
                rank: 1,
                world: 2,
                layout: vec![3; 16],
                gpu_uuid: vec![4; 16],
                agent_nonce: vec![5; 16],
            },
            &mut events,
        );
        assert!(events.is_empty());
        assert!(agent.receive().unwrap().is_none());
        ranks.bind([6; 16]);
        let mut incoming = Vec::new();
        for _ in 0..200 {
            agent.poll_rank(&mut incoming);
            if !incoming.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let Incoming::Message(bound) = incoming.remove(0) else {
            panic!("link failed")
        };
        assert!(matches!(bound, Msg::Bind { instance_id, .. } if instance_id == vec![6; 16]));
    }
}
