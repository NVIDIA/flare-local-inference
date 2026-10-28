//! Rank agents: the attachments of the peer ranks of a multi-process instance.
//!
//! Rank 0 opens the instance's one session (with a `rank_secret`) and decides everything.
//! Each other rank attaches here with `Rank` to move its own slice of the windows. An
//! agent is its own party: what it may still be touching is tracked under its own process
//! key `(instance_id, agent_nonce)`, never under "rank r", so a replacement attachment
//! neither inherits nor clears its predecessor's obligations.

use super::{
    claims::Party,
    core::{Core, Nonce, OpenPeer, ProcKey},
    index::InstanceId,
};
use crate::proto::v2;
use std::time::Instant;
use tokio::sync::mpsc;

/// What a rank-0 session says about its instance.
#[derive(Clone, Debug)]
pub(super) struct Ranks {
    pub world: u32,
    pub secret: [u8; 32],
}

/// One attached peer rank.
pub(super) struct Agent {
    pub aid: u64,
    pub instance: InstanceId,
    /// The rank-0 process this agent serves.
    pub rank0: Nonce,
    pub rank: u32,
    /// The agent's own process identity.
    pub nonce: Nonce,
    pub mode: v2::RankMode,
    /// The session that was rank 0's is gone or superseded: no transfer is granted, and the
    /// agent is here only to discharge what it may still be touching.
    pub drain_only: bool,
    pub tx: mpsc::Sender<v2::RankServerMessage>,
    pub goodbye: Option<bool>,
}

impl Agent {
    pub(super) fn proc(&self) -> ProcKey {
        (self.instance, self.nonce)
    }
}

/// Constant-time equality of two secrets.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl Core {
    fn rank_refuse(
        &self,
        tx: &mpsc::Sender<v2::RankServerMessage>,
        status: v2::RankAttachStatus,
    ) -> Option<u64> {
        let _ = tx.try_send(v2::RankServerMessage {
            m: Some(v2::rank_server_message::M::Attached(v2::RankAttached {
                status: status as i32,
                generation: 0,
                mode: v2::RankMode::Unspecified as i32,
            })),
        });
        None
    }

    /// The live session of `instance` run by process `rank0`, if it is a multi-process one.
    fn rank0_session(&self, instance: InstanceId, rank0: Nonce) -> Option<u64> {
        let sid = self.instances.get(&instance)?.current?;
        let session = self.sessions.get(&sid)?;
        (session.nonce == rank0 && session.ranks.is_some()).then_some(sid)
    }

    /// An agent opens its stream. Returns its id when attached.
    pub(super) fn rank_open(
        &mut self,
        peer: OpenPeer,
        attach: v2::RankAttach,
        tx: mpsc::Sender<v2::RankServerMessage>,
    ) -> Option<u64> {
        use v2::RankAttachStatus as S;
        if self.shutting_down.is_some() || self.state == v2::ReadinessState::Draining {
            return self.rank_refuse(&tx, S::RankNoSession);
        }
        if attach.manager_incarnation != self.incarnation {
            return self.rank_refuse(&tx, S::RankStaleManager);
        }
        let (Ok(instance), Ok(rank0), Ok(nonce)) = (
            InstanceId::try_from(attach.instance_id.as_slice()),
            Nonce::try_from(attach.rank0_nonce.as_slice()),
            Nonce::try_from(attach.agent_nonce.as_slice()),
        ) else {
            return self.rank_refuse(&tx, S::RankBadRank);
        };
        let Some(sid) = self.rank0_session(instance, rank0) else {
            return self.rank_refuse(&tx, S::RankNoSession);
        };
        let ranks = self.sessions[&sid].ranks.as_ref().expect("checked");
        let (secret, world) = (ranks.secret, ranks.world);
        if !same(&secret, &attach.rank_secret) {
            return self.rank_refuse(&tx, S::RankBadSecret);
        }
        if attach.rank == 0 || attach.rank >= world {
            return self.rank_refuse(&tx, S::RankBadRank);
        }
        let shm_capable = peer.unix && peer.uid == self.euid && peer.gid == self.egid;
        let rank0_shm = self.sessions[&sid].shm;
        let mode = if shm_capable && rank0_shm {
            v2::RankMode::RankShared
        } else {
            v2::RankMode::RankChannel
        };
        if mode == v2::RankMode::RankChannel && self.cfg.channel.is_none() {
            return self.rank_refuse(&tx, S::RankNoDataListener);
        }

        // A new attachment of this rank supersedes the live one: that one stays here as a
        // drain-only party, so what it may still touch is not forgotten.
        let held: Vec<u64> = self
            .agents
            .values()
            .filter(|a| {
                a.instance == instance && a.rank0 == rank0 && a.rank == attach.rank && !a.drain_only
            })
            .map(|a| a.aid)
            .collect();
        for aid in held {
            self.rank_detach(
                aid,
                v2::ClosingReason::ClosingSuperseded,
                "a newer agent attached",
            );
        }

        self.next_session += 1;
        let aid = self.next_session;
        let key = (instance, nonce);
        let started_pid = peer.unix.then_some(peer.pid).filter(|pid| *pid != 0);
        let entry = self.procs.entry(key).or_insert_with(|| super::core::Proc {
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
        self.agents.insert(
            aid,
            Agent {
                aid,
                instance,
                rank0,
                rank: attach.rank,
                nonce,
                mode,
                drain_only: false,
                tx: tx.clone(),
                goodbye: None,
            },
        );
        let _ = tx.try_send(v2::RankServerMessage {
            m: Some(v2::rank_server_message::M::Attached(v2::RankAttached {
                status: S::RankOk as i32,
                generation: aid,
                mode: mode as i32,
            })),
        });
        Some(aid)
    }

    /// A message of an attached agent.
    pub(super) fn rank_message(&mut self, aid: u64, message: v2::RankClientMessage) {
        use v2::rank_client_message::M;
        let Some(agent) = self.agents.get_mut(&aid) else {
            return;
        };
        let proc = agent.proc();
        match message.m {
            Some(M::Attach(_)) | None => {
                // A second attach, or an empty message, is a protocol violation.
                self.rank_detach(
                    aid,
                    v2::ClosingReason::ClosingProtocolViolation,
                    "bad message",
                );
            }
            Some(M::Goodbye(goodbye)) => {
                agent.goodbye = Some(goodbye.quiesced);
                if goodbye.quiesced {
                    self.party_quiesced(proc);
                }
            }
            Some(M::Fence(fence)) => {
                let party = match v2::FenceOutcome::try_from(fence.outcome) {
                    Ok(v2::FenceOutcome::Drained) => Party::Drained,
                    Ok(v2::FenceOutcome::Quarantined) => Party::Quarantined,
                    _ => {
                        self.rank_detach(
                            aid,
                            v2::ClosingReason::ClosingProtocolViolation,
                            "fence without an outcome",
                        );
                        return;
                    }
                };
                self.agent_fence(proc, fence.kind, fence.id, party);
            }
        }
    }

    /// The agent's stream ended.
    pub(super) fn rank_closed(&mut self, aid: u64) {
        let Some(agent) = self.agents.remove(&aid) else {
            return;
        };
        let proc = agent.proc();
        if agent.goodbye == Some(true) {
            self.party_quiesced(proc);
        }
        if !agent.drain_only {
            self.agent_gone(proc);
        }
        if let Some(entry) = self.procs.get_mut(&proc) {
            entry.idle_since = Some(Instant::now());
        }
    }

    /// Stop granting this agent anything and ask it to drain. It stays as a party until it
    /// attests, ends, or its process is confirmed gone.
    pub(super) fn rank_detach(&mut self, aid: u64, reason: v2::ClosingReason, detail: &str) {
        let Some(agent) = self.agents.get_mut(&aid) else {
            return;
        };
        let was_live = !agent.drain_only;
        let proc = agent.proc();
        agent.drain_only = true;
        let _ = agent.tx.try_send(v2::RankServerMessage {
            m: Some(v2::rank_server_message::M::Closing(v2::RankClosing {
                reason: reason as i32,
                detail: detail.to_string(),
            })),
        });
        if was_live {
            self.agent_gone(proc);
        }
    }

    /// Rank 0's session ended: its agents drain and nothing new is granted to them.
    pub(super) fn rank0_lost(&mut self, instance: InstanceId, rank0: Nonce) {
        let live: Vec<u64> = self
            .agents
            .values()
            .filter(|a| a.instance == instance && a.rank0 == rank0 && !a.drain_only)
            .map(|a| a.aid)
            .collect();
        for aid in live {
            self.rank_detach(
                aid,
                v2::ClosingReason::ClosingStaleSession,
                "rank 0 is gone",
            );
        }
    }

    /// Whether every peer rank of the multi-process session `sid` is attached and live.
    pub(super) fn ranks_ready(&self, sid: u64) -> bool {
        let Some(session) = self.sessions.get(&sid) else {
            return false;
        };
        let Some(ranks) = &session.ranks else {
            return true;
        };
        (1..ranks.world).all(|rank| {
            self.agents.values().any(|a| {
                a.instance == session.instance
                    && a.rank0 == session.nonce
                    && a.rank == rank
                    && !a.drain_only
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arena::Arena, pool::core::PoolConfig};
    use v2::{RankAttachStatus as S, rank_server_message::M};

    const INSTANCE: InstanceId = [7; 16];
    const RANK0: Nonce = [1; 16];
    const SECRET: [u8; 32] = [9; 32];

    fn core() -> Core {
        let (core, _handle, _events) = Core::new(
            PoolConfig::default(),
            v2::SharedMemory::default(),
            Arena::new(1 << 20).unwrap(),
            None,
        );
        core
    }

    fn scope(world: usize) -> v2::Scope {
        v2::Scope {
            model_identity: vec![1; 32],
            layout_fingerprint: vec![2; 16],
            geometry: Some(v2::Geometry {
                page_tokens: 64,
                ranks: vec![
                    v2::RankPlanes {
                        plane_bytes: vec![4096],
                        row_bytes: 4096,
                    };
                    world
                ],
                page_record_bytes: world as u64 * 4096,
                row_record_bytes: world as u64 * 4096,
            }),
        }
    }

    fn unix(core: &Core) -> OpenPeer {
        OpenPeer {
            unix: true,
            pid: std::process::id(),
            uid: core.euid,
            gid: core.egid,
        }
    }

    fn tcp() -> OpenPeer {
        OpenPeer {
            unix: false,
            pid: 0,
            uid: 0,
            gid: 0,
        }
    }

    /// Opens rank 0 of a `world`-rank instance as process `nonce`.
    fn open_rank0(core: &mut Core, world: usize, nonce: Nonce, peer: OpenPeer) -> u64 {
        open_rank0_with(core, world, nonce, peer).0
    }

    fn open_rank0_with(
        core: &mut Core,
        world: usize,
        nonce: Nonce,
        peer: OpenPeer,
    ) -> (u64, mpsc::Receiver<v2::ServerMessage>) {
        let (tx, rx) = mpsc::channel(64);
        let sid = core
            .open(
                peer,
                v2::OpenSession {
                    manager_incarnation: core.incarnation,
                    instance_id: INSTANCE.to_vec(),
                    scope: Some(scope(world)),
                    process_nonce: nonce.to_vec(),
                    export_slots: 2,
                    claim_slots: 2,
                    rank_secret: SECRET.to_vec(),
                },
                tx,
            )
            .expect("rank 0 opens");
        (sid, rx)
    }

    fn attach(rank: u32, nonce: u8) -> v2::RankAttach {
        v2::RankAttach {
            manager_incarnation: 0,
            instance_id: INSTANCE.to_vec(),
            rank0_nonce: RANK0.to_vec(),
            rank,
            agent_nonce: vec![nonce; 16],
            rank_secret: SECRET.to_vec(),
        }
    }

    fn join(core: &mut Core, mut a: v2::RankAttach, peer: OpenPeer) -> (Option<u64>, Vec<M>) {
        a.manager_incarnation = core.incarnation;
        let (tx, mut rx) = mpsc::channel(16);
        let aid = core.rank_open(peer, a, tx);
        let mut got = Vec::new();
        while let Ok(message) = rx.try_recv() {
            got.extend(message.m);
        }
        std::mem::forget(rx);
        (aid, got)
    }

    fn status(messages: &[M]) -> S {
        match messages.first() {
            Some(M::Attached(a)) => S::try_from(a.status).unwrap(),
            other => panic!("expected an Attached, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_attach_is_checked_against_the_session_secret_and_rank() {
        let mut core = core();
        let peer = unix(&core);
        open_rank0(&mut core, 3, RANK0, peer);
        let me = unix(&core);

        let mut bad = attach(1, 1);
        bad.rank_secret = vec![0; 32];
        assert_eq!(status(&join(&mut core, bad, me).1), S::RankBadSecret);
        assert_eq!(status(&join(&mut core, attach(0, 1), me).1), S::RankBadRank);
        assert_eq!(status(&join(&mut core, attach(3, 1), me).1), S::RankBadRank);
        let mut stale = attach(1, 1);
        stale.rank0_nonce = vec![5; 16];
        assert_eq!(status(&join(&mut core, stale, me).1), S::RankNoSession);
        let mut other = attach(1, 1);
        other.instance_id = vec![8; 16];
        assert_eq!(status(&join(&mut core, other, me).1), S::RankNoSession);

        let (aid, got) = join(&mut core, attach(1, 1), me);
        assert!(aid.is_some());
        assert_eq!(status(&got), S::RankOk);
        let M::Attached(a) = &got[0] else { panic!() };
        assert_eq!(a.mode, v2::RankMode::RankShared as i32);
        assert_eq!(a.generation, aid.unwrap());

        // A stale manager incarnation is told so.
        let (tx, mut rx) = mpsc::channel(4);
        let mut old = attach(2, 2);
        old.manager_incarnation = core.incarnation ^ 2;
        assert!(core.rank_open(me, old, tx).is_none());
        assert_eq!(
            status(&[rx.try_recv().unwrap().m.unwrap()]),
            S::RankStaleManager
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn claims_wait_for_every_peer_rank() {
        let mut core = core();
        let me = unix(&core);
        let sid = open_rank0(&mut core, 3, RANK0, me);
        assert!(!core.ranks_ready(sid));
        let (one, _) = join(&mut core, attach(1, 1), me);
        assert!(!core.ranks_ready(sid));
        let (two, _) = join(&mut core, attach(2, 2), me);
        assert!(core.ranks_ready(sid));
        core.rank_closed(two.unwrap());
        assert!(
            !core.ranks_ready(sid),
            "a rank that detached gates claims again"
        );
        let (two, _) = join(&mut core, attach(2, 3), me);
        assert!(core.ranks_ready(sid));
        core.rank_closed(one.unwrap());
        core.rank_closed(two.unwrap());
        assert!(!core.ranks_ready(sid));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_claim_while_the_regions_are_held_is_refused_distinctly_and_then_served() {
        let mut core = core();
        let me = unix(&core);
        let (sid, mut rx) = open_rank0_with(&mut core, 2, RANK0, me);
        core.state = v2::ReadinessState::Ready;
        core.cfg
            .held
            .store(true, std::sync::atomic::Ordering::Release);
        while rx.try_recv().is_ok() {}
        let claim = |id| v2::Claim {
            claim_id: id,
            tier: v2::Tier::Nvme as i32,
            key: Some(v2::PrefixKey {
                consumed_tokens: 64,
                digest: vec![1; 16],
            }),
            mode: v2::CopyMode::SharedWindows as i32,
            ..Default::default()
        };
        core.claim(sid, claim(1)).unwrap();
        let Some(v2::server_message::M::ClaimTerminal(t)) = rx.try_recv().unwrap().m else {
            panic!("no terminal");
        };
        assert_eq!(t.reason, v2::ClaimReason::RegionsHeld as i32);
        assert!(!t.fence_required);
        // Once the hold ends the same claim gets past the gate (and is refused for what it names).
        core.cfg
            .held
            .store(false, std::sync::atomic::Ordering::Release);
        core.claim(sid, claim(2)).unwrap();
        let Some(v2::server_message::M::ClaimTerminal(t)) = rx.try_recv().unwrap().m else {
            panic!("no terminal");
        };
        assert_ne!(t.reason, v2::ClaimReason::RegionsHeld as i32);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_claim_or_put_of_an_instance_with_a_rank_missing_is_refused_at_once() {
        let mut core = core();
        let me = unix(&core);
        let (sid, mut rx) = open_rank0_with(&mut core, 2, RANK0, me);
        core.state = v2::ReadinessState::Ready;
        while rx.try_recv().is_ok() {}
        core.claim(
            sid,
            v2::Claim {
                claim_id: 1,
                tier: v2::Tier::Nvme as i32,
                key: Some(v2::PrefixKey {
                    consumed_tokens: 64,
                    digest: vec![1; 16],
                }),
                mode: v2::CopyMode::SharedWindows as i32,
                ..Default::default()
            },
        )
        .unwrap();
        let Some(v2::server_message::M::ClaimTerminal(t)) = rx.try_recv().unwrap().m else {
            panic!("no terminal");
        };
        assert_eq!(t.state, v2::ClaimState::Failed as i32);
        assert_eq!(t.reason, v2::ClaimReason::RanksDetached as i32);
        assert!(!t.fence_required);
        core.put_begin(
            sid,
            v2::PutBegin {
                put_id: 1,
                mode: v2::CopyMode::SharedWindows as i32,
                ..Default::default()
            },
        )
        .unwrap();
        let Some(v2::server_message::M::PutRefused(r)) = rx.try_recv().unwrap().m else {
            panic!("no refusal");
        };
        assert_eq!(r.reason, v2::Reason::PutNotReady as i32);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_replacement_agent_does_not_replace_the_obligations_of_the_old_one() {
        let mut core = core();
        let me = unix(&core);
        let sid = open_rank0(&mut core, 2, RANK0, me);
        let (old, _) = join(&mut core, attach(1, 1), me);
        let (new, got) = join(&mut core, attach(1, 2), me);
        assert_eq!(status(&got), S::RankOk);
        let (old, new) = (old.unwrap(), new.unwrap());
        assert!(
            core.agents[&old].drain_only,
            "the predecessor only drains now"
        );
        assert!(!core.agents[&new].drain_only);
        assert!(core.ranks_ready(sid));
        // Two distinct party identities: the replacement never stands in for the old one.
        assert_ne!(core.agents[&old].proc(), core.agents[&new].proc());
        assert!(core.procs.contains_key(&core.agents[&old].proc()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn when_rank_zero_ends_existing_agents_only_drain_and_new_attachments_are_refused() {
        let mut core = core();
        let me = unix(&core);
        let sid = open_rank0(&mut core, 2, RANK0, me);
        let (aid, _) = join(&mut core, attach(1, 1), me);
        let aid = aid.unwrap();
        core.session_lost(sid);
        assert!(core.agents[&aid].drain_only);
        // The existing stream may drain, but a new attachment needs a live session.
        assert_eq!(
            status(&join(&mut core, attach(1, 4), me).1),
            S::RankNoSession
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_same_process_reconnecting_keeps_its_agents_and_another_process_does_not() {
        let mut core = core();
        let me = unix(&core);
        open_rank0(&mut core, 2, RANK0, me);
        let (aid, _) = join(&mut core, attach(1, 1), me);
        let aid = aid.unwrap();
        // The same process opens a second session.
        let second = open_rank0(&mut core, 2, RANK0, me);
        assert!(!core.agents[&aid].drain_only, "a reconnect keeps the agent");
        assert!(core.ranks_ready(second));
        // Another process takes the instance over.
        let other = [2u8; 16];
        let third = open_rank0(&mut core, 2, other, me);
        assert!(
            core.agents[&aid].drain_only,
            "the old process's agent only drains"
        );
        assert!(!core.ranks_ready(third));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_remote_agent_needs_the_data_listener_and_a_single_process_open_is_refused_malformed()
    {
        let mut core = core();
        let me = unix(&core);
        open_rank0(&mut core, 2, RANK0, me);
        // No data listener: a network agent cannot be served.
        assert_eq!(
            status(&join(&mut core, attach(1, 1), tcp()).1),
            S::RankNoDataListener
        );
        // A short secret, or one rank only, is an invalid scope.
        let (tx, mut rx) = mpsc::channel(4);
        let sid = core.open(
            me,
            v2::OpenSession {
                manager_incarnation: core.incarnation,
                instance_id: [5; 16].to_vec(),
                scope: Some(scope(1)),
                process_nonce: [3; 16].to_vec(),
                rank_secret: SECRET.to_vec(),
                ..Default::default()
            },
            tx,
        );
        assert!(sid.is_none());
        let Some(v2::server_message::M::Opened(opened)) = rx.try_recv().unwrap().m else {
            panic!()
        };
        assert_eq!(opened.status, v2::OpenStatus::InvalidScope as i32);
    }
}
