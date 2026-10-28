//! The manager's own store as a participant in the pool protocol.
//!
//! The pool owns window credits and buffer holds. The store owner thread keeps
//! records and performs the requested I/O through typed commands.

use super::{
    Checked, Violation,
    claims::{Channel, ClaimKey, Owner, Participant, Party, activate, attest, participants},
    core::{ChanOwner, Core, Event, Nonce, ProcKey, ScopeId},
    index::{Entry, InstanceId, SourceId},
    key_from, key_to,
    mp::{PutMp, PutWin},
};
use crate::{
    Key,
    data::channel,
    proto::{self, v2, v2::server_message::M},
    record::{Descriptor, NamespaceId},
    store::{GetAnswer, PutAnswer, Reply, StoreCommand},
};
use flare_selene::shared::Tier;
use flare_selene::shared::record::{HEADER_BYTES, Header};
use std::io;
use tokio::sync::{mpsc, oneshot};

/// A process's connection to the store.
#[derive(Clone, Copy, Debug)]
pub(super) struct Link {
    pub connection: u64,
    pub namespace: NamespaceId,
}

pub(super) struct StoreHandle {
    pub commands: mpsc::Sender<StoreCommand>,
    pub next_connection: u64,
}

pub(super) type PutKey = (InstanceId, Nonce, u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PutPhase {
    Open,
    Committed,
    Finished,
}

pub(super) struct PutRec {
    pub phase: PutPhase,
    pub link: Link,
    pub pending_windows: u32,
    pub remaining: usize,
    pub windows: std::collections::HashMap<u32, std::sync::Arc<crate::arena::Extent>>,
    pub parties: Vec<Participant>,
    pub refused: bool,
    pub key: Key,
    pub chain_pages: u64,
    /// The record carries the state row.
    pub row: bool,
    /// What the store accepted: the parent it will link to and the first page it needs.
    pub accepted: Option<(Option<Key>, u32)>,
    /// Set when the writer's bytes arrive on the data channel.
    pub legs: Vec<super::mp::Leg>,
    /// A commit that arrived before the data channel delivered every window: the
    /// channel and the session are separate streams, so it waits for the last fill.
    pub deferred_commit: Option<(u64, v2::PutCommit)>,
    /// What a multi-process writer adds: its peers, their legs, the windows being assembled.
    pub mp: PutMp,
}

impl PutRec {
    pub(super) fn channel(&self) -> Option<&Channel> {
        super::mp::root(&self.legs, true)
    }
}

impl Core {
    /// Asks the store and waits for its answer. The store thread never waits on the
    /// pool, so this returns promptly; the multi-thread runtime keeps serving others.
    fn store_call<T: Send>(
        &self,
        make: impl FnOnce(Reply<io::Result<T>>) -> StoreCommand,
    ) -> io::Result<T> {
        let Some(store) = &self.store else {
            return Err(io::Error::other("the store is not available"));
        };
        let (reply, answer) = oneshot::channel();
        let command = make(reply);
        tokio::task::block_in_place(|| {
            store
                .commands
                .blocking_send(command)
                .map_err(|_| io::Error::other("the store stopped"))?;
            answer
                .blocking_recv()
                .map_err(|_| io::Error::other("the store dropped the request"))?
        })
    }

    pub(super) fn store_send(&self, command: StoreCommand) {
        if let Some(store) = &self.store {
            // A full queue means the store is busy; the pool retries nothing here because
            // every use is a release the store also performs when the connection ends.
            let _ = store.commands.try_send(command);
        }
    }

    fn new_store_connection(&mut self) -> Option<u64> {
        let store = self.store.as_mut()?;
        store.next_connection += 1;
        Some(store.next_connection)
    }

    // ------------------------------------------------------------------ catalog

    /// Subscribes to the store's catalog for a scope so its records appear in the
    /// index as the manager's entries.
    pub(super) fn watch_catalog(&mut self, scope: ScopeId) {
        if self.store.is_none() || self.catalogs.contains(&scope) {
            return;
        }
        let Some(descriptor) = self.scopes.get(&scope).map(|s| s.valid.descriptor.clone()) else {
            return;
        };
        let (events, mut feed) = mpsc::unbounded_channel();
        let snapshot = self.store_call(|reply| StoreCommand::Catalog {
            descriptor,
            events,
            reply,
        });
        let snapshot = match snapshot {
            Ok(snapshot) => snapshot,
            Err(error) => {
                eprintln!("selene: warning: the store would not open its catalog: {error}");
                return;
            }
        };
        self.catalogs.insert(scope);
        self.scopes
            .get_mut(&scope)
            .unwrap()
            .index
            .replace(SourceId::Manager, Vec::new());
        for entry in snapshot {
            self.catalog_event(scope, proto::CatalogEvent::Added(entry));
        }
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            while let Some(event) = feed.recv().await {
                if tx.send(Event::Catalog { scope, event }).await.is_err() {
                    return;
                }
            }
        });
    }

    pub(super) fn catalog_event(&mut self, scope: ScopeId, event: proto::CatalogEvent) {
        use proto::CatalogEvent as E;
        let to_entry = |entry: &crate::store::EntryData| -> Entry {
            let tier = match entry.tier {
                0 => Tier::Dram,
                _ => Tier::Nvme,
            };
            Entry {
                key: entry.key,
                parent: entry.parent,
                bucket: entry.bucket,
                tail_pages: entry.tail_pages,
                row: entry.row,
                tier,
                bytes: entry.bytes,
            }
        };
        let Some(state) = self.scopes.get_mut(&scope) else {
            return;
        };
        let manager = SourceId::Manager;
        match event {
            E::Added(entry) | E::Changed(entry) => {
                let entry = to_entry(&entry);
                let other = match entry.tier {
                    Tier::Dram => Tier::Nvme,
                    _ => Tier::Dram,
                };
                state.index.remove(manager, entry.key, other);
                state.index.upsert(manager, entry);
            }
            E::Removed(key) => {
                state.index.remove(manager, key, Tier::Dram);
                state.index.remove(manager, key, Tier::Nvme);
            }
        }
    }

    // ------------------------------------------------------------------ links

    /// The process's store connection, opened on first use.
    pub(super) fn link_for(&mut self, process: ProcKey, scope: ScopeId, pid: u32) -> Option<Link> {
        if let Some(link) = self.procs.get(&process).and_then(|p| p.link) {
            return Some(link);
        }
        let descriptor: Descriptor = self.scopes.get(&scope)?.valid.descriptor.clone();
        let connection = self.new_store_connection()?;
        let namespace = descriptor.namespace_id();
        self.store_call(|reply| StoreCommand::Hello {
            connection,
            pid,
            descriptor,
            reply,
        })
        .ok()?;
        let link = Link {
            connection,
            namespace,
        };
        if let Some(proc) = self.procs.get_mut(&process) {
            proc.link = Some(link);
        }
        Some(link)
    }

    /// The process is gone: its store connection and everything on it goes too.
    pub(super) fn drop_link(&mut self, process: ProcKey) {
        if let Some(link) = self.procs.get_mut(&process).and_then(|p| p.link.take()) {
            self.store_send(StoreCommand::ConnectionClosed {
                connection: link.connection,
            });
        }
    }

    fn proc_session(&self, process: ProcKey) -> Option<u64> {
        let sid = self.instances.get(&process.0)?.current?;
        let session = self.sessions.get(&sid)?;
        (session.nonce == process.1).then_some(sid)
    }

    // ------------------------------------------------------------------ put

    fn put_refusal_log(&mut self, put_id: u64, reason: v2::Reason, detail: &str) -> Option<String> {
        if reason == v2::Reason::Present {
            self.present_puts = self.present_puts.saturating_add(1);
            return self.present_puts.is_power_of_two().then(|| {
                format!(
                    "selene: {} duplicate puts refused (Present) so far",
                    self.present_puts
                )
            });
        }
        Some(format!(
            "selene: put {put_id} refused ({reason:?}): {detail}"
        ))
    }

    fn put_refusal(&mut self, sid: u64, put_id: u64, reason: v2::Reason, detail: &str) {
        if let Some(message) = self.put_refusal_log(put_id, reason, detail) {
            eprintln!("{message}");
        }
        self.send(
            sid,
            M::PutRefused(v2::PutRefused {
                put_id,
                reason: reason as i32,
                detail: detail.to_string(),
            }),
        );
    }

    pub(super) fn put_begin(&mut self, sid: u64, m: v2::PutBegin) -> Checked<()> {
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        let process = (session.instance, session.nonce);
        let (scope, shm) = (session.scope, session.shm);
        let put_id = m.put_id;
        let key = (process.0, process.1, put_id);
        let Some(proc) = self.procs.get_mut(&process) else {
            return Ok(());
        };
        if put_id == 0 || put_id <= proc.put_hw {
            self.put_refusal(sid, put_id, v2::Reason::Invalid, "put ids must increase");
            return Ok(());
        }
        proc.put_hw = put_id;
        if !self.ready() || self.store.is_none() {
            self.put_refusal(
                sid,
                put_id,
                v2::Reason::PutNotReady,
                "the manager is not ready",
            );
            return Ok(());
        }
        if !self.ranks_ready(sid) {
            self.put_refusal(
                sid,
                put_id,
                v2::Reason::PutNotReady,
                "a peer rank of this instance is not attached",
            );
            return Ok(());
        }
        let open = self.puts.keys().filter(|k| (k.0, k.1) == process).count();
        if open >= self.cfg.limits.max_open_puts as usize {
            self.put_refusal(sid, put_id, v2::Reason::Queue, "too many open puts");
            return Ok(());
        }
        let mode = v2::CopyMode::try_from(m.mode).unwrap_or(v2::CopyMode::Unspecified);
        match mode {
            v2::CopyMode::SharedWindows if shm => {}
            v2::CopyMode::SharedWindows => {
                self.put_refusal(
                    sid,
                    put_id,
                    v2::Reason::Invalid,
                    "shared windows need a Unix-socket session of the manager's own user",
                );
                return Ok(());
            }
            v2::CopyMode::DataChannel if self.cfg.channel.is_some() => {}
            v2::CopyMode::DataChannel => {
                self.put_refusal(
                    sid,
                    put_id,
                    v2::Reason::Invalid,
                    "this manager serves no data channel (it has no network listener)",
                );
                return Ok(());
            }
            _ => {
                self.put_refusal(sid, put_id, v2::Reason::Invalid, "bad mode");
                return Ok(());
            }
        }
        let Ok(record_key) = key_from(m.key.as_ref()) else {
            self.put_refusal(sid, put_id, v2::Reason::Invalid, "bad key");
            return Ok(());
        };
        let parent = if m.has_published_parent {
            match key_from(m.published_parent.as_ref()) {
                Ok(parent) => Some(parent),
                Err(_) => {
                    self.put_refusal(sid, put_id, v2::Reason::Invalid, "bad parent");
                    return Ok(());
                }
            }
        } else {
            None
        };
        let Ok(bucket) = <[u8; 16]>::try_from(m.bucket.as_slice()) else {
            self.put_refusal(sid, put_id, v2::Reason::Invalid, "bucket must be 16 bytes");
            return Ok(());
        };
        let (Ok(chain_pages), Ok(tail_pages)) = (
            u32::try_from(m.chain_pages),
            u32::try_from(m.published_tail_pages),
        ) else {
            self.put_refusal(sid, put_id, v2::Reason::Invalid, "page counts");
            return Ok(());
        };
        let pid = self.sessions[&sid].pid;
        let Some(link) = self.link_for(process, scope, pid) else {
            let detail = if pid == 0 {
                "the peer's process id is not visible; share a PID namespace with the manager"
            } else {
                "no connection to the manager's store"
            };
            self.put_refusal(sid, put_id, v2::Reason::Io, detail);
            return Ok(());
        };
        let descriptor = self.scopes[&scope].valid.descriptor.clone();
        let (events, mut answers) = mpsc::channel::<io::Result<PutAnswer>>(32);
        let begun = self.store_call(|reply| StoreCommand::PutBegin {
            connection: link.connection,
            transfer: put_id,
            namespace: link.namespace,
            descriptor,
            key: record_key,
            published_parent: parent,
            bucket,
            chain_pages,
            published_tail_pages: tail_pages,
            row: m.row,
            events,
            reply,
        });
        if let Err(error) = begun {
            self.put_refusal(sid, put_id, v2::Reason::Invalid, &error.to_string());
            return Ok(());
        }
        let slices = self.requester_slices(sid);
        let channel = if mode == v2::CopyMode::DataChannel {
            match self.open_leg(channel::Kind::Put, slices, 0) {
                Ok(ch) => Some(ch),
                Err(error) => {
                    self.store_send(StoreCommand::Cancel {
                        connection: link.connection,
                        transfer: put_id,
                    });
                    self.put_refusal(sid, put_id, v2::Reason::Io, &error.to_string());
                    return Ok(());
                }
            }
        } else {
            None
        };
        // The peer ranks of a multi-process writer.
        let mut legs = Vec::new();
        let mp = match self.mp_side(
            sid,
            &|i| ChanOwner::PutRank(key, i),
            channel::Kind::Put,
            &mut legs,
        ) {
            Ok(side) => PutMp {
                layout: side.layout,
                peers: side.peers,
                windows: Default::default(),
                passes: 0,
                whole: Vec::new(),
                row: None,
            },
            Err(_) => {
                self.discard_legs(channel);
                self.store_send(StoreCommand::Cancel {
                    connection: link.connection,
                    transfer: put_id,
                });
                self.put_refusal(
                    sid,
                    put_id,
                    v2::Reason::PutNotReady,
                    "a peer rank of this instance is not attached",
                );
                return Ok(());
            }
        };
        let legs = super::mp::join_legs(None, channel, legs);
        for (index, leg) in legs.iter().enumerate() {
            self.channels.insert(
                leg.ch.transfer,
                if leg.rank == 0 {
                    ChanOwner::Put(key)
                } else {
                    ChanOwner::PutRank(key, index)
                },
            );
        }
        let mut parties = participants(
            None,
            Some(process),
            &legs,
            [(
                true,
                mp.peers
                    .iter()
                    .filter(|p| p.shared)
                    .map(|p| p.proc)
                    .collect(),
            )],
        );
        parties.retain(|p| p.memory);
        for p in parties
            .iter_mut()
            .filter(|p| p.owner == Owner::Proc(process))
        {
            p.state = Party::Idle;
        }
        self.puts.insert(
            key,
            PutRec {
                phase: PutPhase::Open,
                link,
                pending_windows: 0,
                remaining: 0,
                windows: Default::default(),
                parties,
                refused: false,
                key: record_key,
                chain_pages: m.chain_pages,
                row: m.row,
                accepted: None,
                legs,
                deferred_commit: None,
                mp,
            },
        );
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            while let Some(answer) = answers.recv().await {
                let event = Event::PutAnswer {
                    process,
                    put_id,
                    answer,
                };
                if tx.send(event).await.is_err() {
                    return;
                }
            }
        });
        Ok(())
    }

    pub(super) fn put_answer(
        &mut self,
        process: ProcKey,
        put_id: u64,
        answer: io::Result<PutAnswer>,
    ) {
        let key = (process.0, process.1, put_id);
        let Some(sid) = self.proc_session(process) else {
            // The writer is gone; the store finishes or abandons the operation itself.
            if matches!(
                answer,
                Ok(PutAnswer::Stored { .. }) | Ok(PutAnswer::Refused(_)) | Err(_)
            ) {
                self.puts_finish(key);
            }
            return;
        };
        let Some(rec) = self.puts.get_mut(&key) else {
            return;
        };
        match answer {
            Ok(PutAnswer::Accept(accept)) => {
                rec.accepted = Some((accept.parent, accept.first_page));
                rec.remaining = accept.windows;
                let channel = rec
                    .channel()
                    .zip(self.cfg.channel.as_ref())
                    .map(|(open, hub)| super::channel_ref(hub, &open.token));
                let rank_channels = self.cfg.channel.as_ref().map_or_else(Vec::new, |hub| {
                    rec.legs
                        .iter()
                        .filter(|leg| leg.rank != 0)
                        .map(|leg| v2::RankChannel {
                            rank: leg.rank,
                            channel: Some(super::channel_ref(hub, &leg.ch.token)),
                        })
                        .collect()
                });
                self.send(
                    sid,
                    M::PutAccept(v2::PutAccept {
                        put_id,
                        first_page: u64::from(accept.first_page),
                        has_parent: accept.parent.is_some(),
                        parent: accept.parent.map(key_to),
                        record_bytes: accept.record_bytes,
                        channel,
                        rank_channels,
                    }),
                );
            }
            Ok(PutAnswer::Grant(grant)) => {
                activate(&mut rec.parties, None);
                rec.pending_windows = rec.pending_windows.saturating_sub(1);
                rec.windows.insert(grant.index, grant.hold);
                let region = u8::try_from(grant.region)
                    .ok()
                    .and_then(flare_selene::shared::region::Region::from_u8);
                let layout = rec.mp.layout.clone();
                // A multi-process writer's window is written by every rank: record who
                // must report, and grant each peer on a leg its own slice.
                {
                    let mp = &mut rec.mp;
                    let Some(region) = region else {
                        self.put_fail_by_key(key, v2::Reason::Io, "a window in an unknown region");
                        return;
                    };
                    let writers = rec
                        .legs
                        .iter()
                        .filter(|leg| leg.rank != 0)
                        .fold(1u64, |mask, leg| mask | 1 << leg.rank);
                    mp.windows.insert(
                        grant.index,
                        PutWin {
                            region,
                            first_page: grant.page_first as u32,
                            offset: grant.offset,
                            pages: grant.page_count,
                            row: grant.row,
                            writers,
                            stated: vec![None; mp.layout.world()],
                            arrived: false,
                        },
                    );
                }
                let legs = rec.legs.iter().map(|l| (l.ch.transfer, l.rank)).collect();
                let spec = v2::WindowSpec {
                    index: grant.index,
                    region: grant.region,
                    offset: grant.offset,
                    length: grant.length,
                    record: grant.record,
                    first_page: grant.page_first as u32,
                    page_count: grant.page_count,
                    row: grant.row,
                };
                let message = M::PutGrant(v2::PutGrant {
                    put_id,
                    window: Some(spec),
                });
                self.grant_windows(sid, message, legs, &layout, &[spec]);
            }
            Ok(PutAnswer::Stored { bytes }) => {
                rec.phase = PutPhase::Finished;
                rec.pending_windows = 0;
                self.send(sid, M::Stored(v2::Stored { put_id, bytes }));
                self.puts_settle(key);
            }
            Ok(PutAnswer::Refused(refused)) => {
                rec.phase = PutPhase::Finished;
                rec.pending_windows = 0;
                let already = rec.refused;
                rec.refused = true;
                if !already {
                    let reason = v2::Reason::try_from(i32::from(refused.reason))
                        .unwrap_or(v2::Reason::Invalid);
                    self.put_refusal(sid, put_id, reason, &refused.detail);
                }
                self.puts_settle(key);
            }
            Err(error) => {
                rec.phase = PutPhase::Finished;
                rec.pending_windows = 0;
                let already = rec.refused;
                rec.refused = true;
                if !already {
                    self.put_refusal(sid, put_id, v2::Reason::Io, &error.to_string());
                }
                self.puts_settle(key);
            }
        }
    }

    /// A finished put with no window the writer may still hold needs no record.
    /// One on the data channel is held until its handler has exited.
    pub(super) fn puts_settle(&mut self, key: PutKey) {
        let Some(rec) = self.puts.get(&key) else {
            return;
        };
        if rec.phase != PutPhase::Finished {
            return;
        }
        self.put_stop_legs(key);
        let Some(rec) = self.puts.get(&key) else {
            return;
        };
        if self.holders(&rec.parties, None, rec.mp.passes) == Party::Drained {
            if let Some(store) = &self.store {
                let command = StoreCommand::Cancel {
                    connection: rec.link.connection,
                    transfer: key.2,
                };
                tokio::task::block_in_place(|| store.commands.blocking_send(command)).ok();
            }
            self.puts.remove(&key);
        }
    }

    /// The put's windows that have been filled and written are accounted for; a
    /// handler that reports an error ends the put.
    pub(super) fn put_channel_failed(&mut self, key: PutKey, reason: &str) {
        eprintln!("selene: put {} on the data channel failed: {reason}", key.2);
        self.put_fail_by_key(key, v2::Reason::Io, reason);
    }

    /// Fails a put, telling its writer when its session is still there.
    pub(super) fn put_fail_by_key(&mut self, key: PutKey, reason: v2::Reason, detail: &str) {
        match self.proc_session((key.0, key.1)) {
            Some(sid) => self.put_fail(sid, key, reason, detail),
            None => {
                if let Some(rec) = self.puts.get_mut(&key) {
                    let link = rec.link;
                    rec.refused = true;
                    rec.phase = PutPhase::Finished;
                    rec.pending_windows = 0;
                    self.store_send(StoreCommand::Cancel {
                        connection: link.connection,
                        transfer: key.2,
                    });
                }
                self.puts_settle(key);
            }
        }
    }

    fn puts_finish(&mut self, key: PutKey) {
        if let Some(rec) = self.puts.get_mut(&key) {
            rec.phase = PutPhase::Finished;
        }
        self.puts_settle(key);
    }

    pub(super) fn put_filled(&mut self, sid: u64, m: v2::PutFilled) -> Checked<()> {
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        let key = (session.instance, session.nonce, m.put_id);
        // Windows of a put on the data channel are filled there, never reported here.
        if self
            .puts
            .get(&key)
            .is_some_and(|rec| rec.channel().is_some())
        {
            return Err(Violation::new("PutFilled for a put on the data channel"));
        }
        if self
            .puts
            .get(&key)
            .is_some_and(|rec| rec.mp.layout.world() > 1)
        {
            // A multi-process writer states its ranks' slice checksums, not whole ones.
            if !m.page_xxh3.is_empty() || m.rank_sums.is_empty() {
                return Err(Violation::new(
                    "a multi-process writer states slice checksums",
                ));
            }
            let parts = super::mp::rank_sums(m.rank_sums);
            self.window_parts(ChanOwner::Put(key), m.index, parts, 0);
            return Ok(());
        }
        self.window_parts(
            ChanOwner::Put(key),
            m.index,
            vec![(
                0,
                flare_selene::shared::SliceSums {
                    pages: m.page_xxh3,
                    row: m.has_row.then_some(m.row_xxh3),
                },
            )],
            0,
        );
        Ok(())
    }

    pub(super) fn put_window_arrived(
        &mut self,
        key: PutKey,
        index: u32,
        stated: Vec<flare_selene::shared::SliceSums>,
    ) {
        let Some(rec) = self.puts.get_mut(&key) else {
            return;
        };
        let mp = &mut rec.mp;
        if mp.layout.world() == 1 {
            let whole = stated.into_iter().next().unwrap();
            return self.put_window_filled(key, index, whole.pages, whole.row);
        }
        let window = &mp.windows[&index];
        let Some(regions) = self.cfg.regions.clone() else {
            return self.put_fail_by_key(key, v2::Reason::Io, "the manager has no regions");
        };
        let job = super::pass::PassJob {
            regions,
            region: window.region,
            offset: window.offset,
            pages: window.pages,
            row: window.row,
            layout: mp.layout.clone(),
            stated_slices: Some(stated),
            stated_whole: None,
        };
        mp.passes += 1;
        let events = self.self_tx.clone();
        tokio::task::spawn_blocking(move || {
            let result = super::pass::run(job);
            let _ = events.blocking_send(Event::PutPass { key, index, result });
        });
    }

    /// A put's assembly pass finished: the verified whole checksums go to the store.
    pub(super) fn put_pass_done(
        &mut self,
        key: PutKey,
        index: u32,
        result: Result<super::pass::PassOut, super::pass::PassError>,
    ) {
        let Some(rec) = self.puts.get_mut(&key) else {
            return;
        };
        rec.mp.passes = rec.mp.passes.saturating_sub(1);
        if rec.phase != PutPhase::Open {
            return self.puts_settle(key);
        }
        match result {
            Ok(out) => {
                let whole = out.whole;
                {
                    let mp = &mut rec.mp;
                    let first = mp.windows.get(&index).map_or(0, |w| w.first_page as usize);
                    if mp.whole.len() < first + whole.pages.len() {
                        mp.whole.resize(first + whole.pages.len(), None);
                    }
                    for (i, sum) in whole.pages.iter().enumerate() {
                        mp.whole[first + i] = Some(*sum);
                    }
                    if whole.row.is_some() {
                        mp.row = whole.row;
                    }
                }
                self.put_window_filled(key, index, whole.pages, whole.row);
            }
            Err(super::pass::PassError::Mismatch(why)) => {
                self.put_fail_by_key(key, v2::Reason::Invalid, &why);
            }
            Err(super::pass::PassError::Io(why)) => {
                self.put_fail_by_key(key, v2::Reason::Io, &why);
            }
        }
    }

    /// Stop every leg of a put that is over.
    fn put_stop_legs(&mut self, key: PutKey) {
        if let Some(rec) = self.puts.get_mut(&key) {
            super::mp::stop_legs(
                &rec.legs,
                &mut rec.parties,
                self.cfg.channel.as_ref(),
                &mut self.channels,
            );
        }
    }

    pub(super) fn put_agent_fence(&mut self, key: PutKey, agent: ProcKey, state: Party) {
        if let Some(rec) = self.puts.get_mut(&key) {
            attest(&mut rec.parties, Owner::Proc(agent), state, false);
        }
        self.puts_settle(key);
    }

    /// A peer rank of some puts is gone: those puts cannot continue.
    pub(super) fn puts_peer_gone(&mut self, process: ProcKey) {
        let keys: Vec<PutKey> = self
            .puts
            .iter()
            .filter(|(_, rec)| {
                rec.phase == PutPhase::Open
                    && rec.mp.peers.iter().any(|p| p.shared && p.proc == process)
            })
            .map(|(key, _)| *key)
            .collect();
        for key in keys {
            self.put_fail_by_key(key, v2::Reason::Io, "a peer rank of the writer was lost");
        }
    }

    /// The writer filled window `index`: the store checks and consumes it.
    pub(super) fn put_window_filled(
        &mut self,
        key: PutKey,
        index: u32,
        page_xxh3: Vec<u64>,
        row_xxh3: Option<u64>,
    ) {
        let Some(rec) = self.puts.get(&key) else {
            return;
        };
        if rec.phase != PutPhase::Open {
            return;
        }
        let link = rec.link;
        let result = self.store_call(|reply| StoreCommand::WindowFilled {
            connection: link.connection,
            transfer: key.2,
            index,
            page_xxh3,
            row_xxh3,
            reply,
        });
        match result {
            Ok(()) => {
                let mut deferred = None;
                if let Some(rec) = self.puts.get_mut(&key) {
                    rec.windows.remove(&index);
                    if rec.windows.is_empty() {
                        attest(
                            &mut rec.parties,
                            Owner::Proc((key.0, key.1)),
                            Party::Idle,
                            false,
                        );
                    }
                    if rec.windows.is_empty() {
                        deferred = rec.deferred_commit.take();
                    }
                }
                if let Some((sid, commit)) = deferred {
                    if let Err(violation) = self.put_commit(sid, commit) {
                        self.violation(sid, violation);
                    }
                }
            }
            Err(error) => {
                self.put_fail_by_key(key, v2::Reason::Invalid, &error.to_string());
            }
        }
    }

    fn put_fail(&mut self, sid: u64, key: PutKey, reason: v2::Reason, detail: &str) {
        let Some(rec) = self.puts.get_mut(&key) else {
            return;
        };
        let link = rec.link;
        let already = rec.refused;
        rec.refused = true;
        rec.phase = PutPhase::Finished;
        rec.pending_windows = 0;
        self.store_send(StoreCommand::Cancel {
            connection: link.connection,
            transfer: key.2,
        });
        if !already {
            self.put_refusal(sid, key.2, reason, detail);
        }
        self.puts_settle(key);
    }

    pub(super) fn put_commit(&mut self, sid: u64, m: v2::PutCommit) -> Checked<()> {
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        let process = (session.instance, session.nonce);
        let key = (process.0, process.1, m.put_id);
        let Some(rec) = self.puts.get_mut(&key) else {
            return Ok(());
        };
        if rec.phase != PutPhase::Open {
            return Ok(());
        }
        if (rec.channel().is_some() || rec.mp.layout.world() > 1) && !rec.windows.is_empty() {
            rec.deferred_commit = Some((sid, m));
            return Ok(());
        }
        // The metadata must describe exactly the record that was accepted, in this
        // scope: its identity, geometry, key, parent and length.
        let descriptor = &self.scopes[&session.scope].valid.descriptor;
        let valid = m.metadata.len() >= HEADER_BYTES
            && Header::decode(&m.metadata).is_ok_and(|header| {
                let (parent, first_page) = rec.accepted.unwrap_or((None, 0));
                header.identity == descriptor.identity
                    && header.fingerprint == descriptor.fingerprint
                    && descriptor.matches(&header)
                    && header.key == rec.key
                    && header.parent == parent
                    && u64::from(header.tail_pages) + u64::from(first_page) == rec.chain_pages
            });
        if !valid {
            self.put_fail(
                sid,
                key,
                v2::Reason::Invalid,
                "record metadata does not match the put",
            );
            return Ok(());
        }
        // A multi-process writer's ranks cannot state whole-page checksums; the record keeps
        // the ones the manager computed over the assembled windows.
        let mut metadata = m.metadata.clone();
        if rec.mp.layout.world() > 1 {
            let mp = &rec.mp;
            let header = Header::decode(&metadata).expect("validated above");
            let pages: Option<Vec<u64>> = (0..header.tail_pages as usize)
                .map(|page| mp.whole.get(page).copied().flatten())
                .collect();
            let row = mp.row.filter(|_| rec.row);
            match pages {
                Some(pages) if set_checksums(&mut metadata, &header, &pages, row) => {}
                _ => {
                    self.put_fail(
                        sid,
                        key,
                        v2::Reason::Invalid,
                        "the record's pages were not all written by every rank",
                    );
                    return Ok(());
                }
            }
        }
        let Some(rec) = self.puts.get_mut(&key) else {
            return Ok(());
        };
        rec.phase = PutPhase::Committed;
        let link = rec.link;
        let result = self.store_call(|reply| StoreCommand::PutCommit {
            connection: link.connection,
            transfer: m.put_id,
            metadata_offset: m.metadata_offset,
            metadata,
            reply,
        });
        if let Err(error) = result {
            self.put_fail(sid, key, v2::Reason::Invalid, &error.to_string());
        }
        Ok(())
    }

    pub(super) fn put_cancel(&mut self, sid: u64, m: v2::PutCancel) -> Checked<()> {
        let Some(session) = self.sessions.get(&sid) else {
            return Ok(());
        };
        let key = (session.instance, session.nonce, m.put_id);
        let Some(rec) = self.puts.get_mut(&key) else {
            return Ok(());
        };
        if rec.phase != PutPhase::Open {
            return Ok(());
        }
        rec.phase = PutPhase::Finished;
        rec.refused = true;
        let link = rec.link;
        self.store_send(StoreCommand::Cancel {
            connection: link.connection,
            transfer: m.put_id,
        });
        self.puts_settle(key);
        Ok(())
    }

    /// The writer's fence for a put: its writes to the leased windows have stopped.
    pub(super) fn put_fence(
        &mut self,
        instance: InstanceId,
        nonce: Nonce,
        put_id: u64,
        party: Party,
    ) {
        let key = (instance, nonce, put_id);
        let Some(rec) = self.puts.get_mut(&key) else {
            return;
        };
        if rec.channel().is_some() {
            // The writer touches none of the manager's windows; its handler's exit
            // is what returns them.
            return;
        }
        attest(
            &mut rec.parties,
            Owner::Proc((instance, nonce)),
            party,
            false,
        );
        if party == Party::Drained {
            rec.phase = PutPhase::Finished;
        }
        self.puts_settle(key);
    }

    /// The stream ended: puts that have not committed cannot continue; their leased
    /// windows wait for the writer's fence or its exit.
    pub(super) fn puts_lost(&mut self, session: &super::core::Session) {
        let process = (session.instance, session.nonce);
        let keys: Vec<PutKey> = self
            .puts
            .iter()
            .filter(|(k, rec)| (k.0, k.1) == process && rec.phase == PutPhase::Open)
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            if let Some(rec) = self.puts.get_mut(&key) {
                rec.phase = PutPhase::Finished;
                rec.pending_windows = 0;
                rec.refused = true;
                let link = rec.link;
                self.store_send(StoreCommand::Cancel {
                    connection: link.connection,
                    transfer: key.2,
                });
            }
            self.puts_settle(key);
        }
    }

    // ------------------------------------------------------------------ get (manager claims)

    /// Starts the store's read of a chain for a claim whose source is the manager.
    pub(super) fn manager_get(
        &mut self,
        key: ClaimKey,
        chain: Vec<Key>,
        skip_pages: u32,
        pages: u64,
        row: bool,
    ) -> Result<(), v2::ClaimReason> {
        let process = (key.0, key.1);
        if !self.claims.contains_key(&key) {
            return Err(v2::ClaimReason::UnknownClaim);
        }
        let Some(link) = self.procs.get(&process).and_then(|p| p.link) else {
            return Err(v2::ClaimReason::SourceBusy);
        };
        let Ok(expected) = u32::try_from(pages) else {
            return Err(v2::ClaimReason::InvalidClaim);
        };
        let (events, mut answers) = mpsc::channel::<io::Result<GetAnswer>>(64);
        let begun = self.store_call(|reply| StoreCommand::GetBegin {
            connection: link.connection,
            transfer: key.2,
            namespace: link.namespace,
            chain,
            skip_pages,
            expected_pages: expected,
            row,
            events,
            reply,
        });
        if begun.is_err() {
            return Err(v2::ClaimReason::SourceBusy);
        }
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            while let Some(answer) = answers.recv().await {
                if tx.send(Event::GetAnswer { key, answer }).await.is_err() {
                    return;
                }
            }
        });
        Ok(())
    }

    pub(super) fn store_cancel_get(&mut self, key: ClaimKey) {
        if let Some(link) = self.procs.get(&(key.0, key.1)).and_then(|p| p.link) {
            self.store_send(StoreCommand::Cancel {
                connection: link.connection,
                transfer: key.2,
            });
        }
    }

    pub(super) fn store_release_window(&mut self, key: ClaimKey, _index: u32) {
        if let Some(manager) = self
            .claims
            .get_mut(&key)
            .and_then(|claim| claim.manager.as_mut())
        {
            manager.pending = manager.pending.saturating_sub(1);
        }
    }

    pub(super) fn store_release_transfer(&mut self, key: ClaimKey) {
        if let Some(claim) = self.claims.get_mut(&key) {
            for window in &mut claim.windows {
                window.extent = None;
            }
            if let Some(manager) = claim.manager.as_mut() {
                manager.pending = 0;
            }
        }
    }

    pub(super) fn pump_store_windows(&mut self) {
        let mut processes: Vec<_> = self
            .procs
            .iter()
            .filter_map(|(key, process)| process.link.map(|link| (*key, link)))
            .collect();
        processes.sort_unstable_by_key(|(key, _)| *key);
        for (process, link) in processes {
            let mut gets: Vec<_> = self
                .claims
                .iter()
                .filter(|(key, claim)| (key.0, key.1) == process && claim.manager.is_some())
                .map(|(key, _)| *key)
                .collect();
            gets.sort_unstable();
            let mut used: usize = gets
                .iter()
                .map(|key| self.claims[key].manager.as_ref().unwrap().pending)
                .sum();
            for key in gets {
                while used < self.cfg.limits.window_credits as usize
                    && !self.claims[&key].is_terminal()
                    && self.claims[&key].manager.as_ref().unwrap().remaining > 0
                {
                    if self
                        .store_call(|reply| StoreCommand::RequestWindow {
                            connection: link.connection,
                            transfer: key.2,
                            reply,
                        })
                        .is_err()
                    {
                        break;
                    }
                    let manager = self.claims.get_mut(&key).unwrap().manager.as_mut().unwrap();
                    manager.remaining -= 1;
                    manager.pending += 1;
                    used += 1;
                }
            }
            let mut puts: Vec<_> = self
                .puts
                .keys()
                .filter(|key| (key.0, key.1) == process)
                .copied()
                .collect();
            puts.sort_unstable();
            let mut used: usize = puts
                .iter()
                .map(|key| self.puts[key].windows.len() + self.puts[key].pending_windows as usize)
                .sum();
            for key in puts {
                while used < self.cfg.limits.window_credits as usize
                    && self.puts[&key].phase == PutPhase::Open
                    && self.puts[&key].remaining > 0
                    && !self.puts[&key].refused
                {
                    if self
                        .store_call(|reply| StoreCommand::RequestWindow {
                            connection: link.connection,
                            transfer: key.2,
                            reply,
                        })
                        .is_err()
                    {
                        break;
                    }
                    let rec = self.puts.get_mut(&key).unwrap();
                    rec.remaining -= 1;
                    rec.pending_windows += 1;
                    used += 1;
                }
            }
        }
    }
}

/// Replace the checksums of a record's metadata (its header block and checksum table) with
/// `pages` and, when the record has a row, `row`.
fn set_checksums(metadata: &mut [u8], header: &Header, pages: &[u64], row: Option<u64>) -> bool {
    let table = HEADER_BYTES..HEADER_BYTES + pages.len() * 8;
    if pages.len() != header.tail_pages as usize || metadata.len() < table.end {
        return false;
    }
    for (chunk, sum) in metadata[table].chunks_exact_mut(8).zip(pages) {
        chunk.copy_from_slice(&sum.to_le_bytes());
    }
    if let Some(row) = row {
        let mut header = header.clone();
        header.row_checksum = row;
        let Ok(encoded) = header.encode() else {
            return false;
        };
        metadata[..HEADER_BYTES].copy_from_slice(&encoded);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_put_logs_are_aggregated_at_doubling_totals() {
        let (mut core, _handle, _events) = Core::new(
            Default::default(),
            Default::default(),
            crate::arena::Arena::new(4096).unwrap(),
            None,
        );
        let mut logged = Vec::new();
        for id in 1..=4096 {
            if let Some(message) = core.put_refusal_log(id, v2::Reason::Present, "already stored") {
                assert!(message.contains(&format!("{id} duplicate puts")));
                logged.push(id);
            }
        }
        assert_eq!(
            logged,
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        );
        assert_eq!(core.present_puts, 4096);
        for id in 4097..4100 {
            assert!(
                core.put_refusal_log(id, v2::Reason::Budget, "full")
                    .is_some()
            );
        }
        assert_eq!(core.present_puts, 4096);
    }

    #[test]
    fn a_quarantined_put_peer_is_retained_until_its_process_exits() {
        use super::super::{core::Proc, mp::MpPeer};
        use flare_selene::shared::{Geometry, RankPlanes, SliceLayout};
        use std::{collections::BTreeMap, sync::Arc};
        let (mut core, _handle, _events) = Core::new(
            Default::default(),
            Default::default(),
            crate::arena::Arena::new(4096).unwrap(),
            None,
        );
        let process = ([1; 16], [2; 16]);
        core.procs.insert(
            process,
            Proc {
                pid: None,
                pidfd: None,
                exited: false,
                claim_hw: 0,
                put_hw: 0,
                idle_since: None,
                link: None,
            },
        );
        let key = ([3; 16], [4; 16], 1);
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
        core.puts.insert(
            key,
            PutRec {
                phase: PutPhase::Finished,
                link: Link {
                    connection: 1,
                    namespace: [0; 32],
                },
                pending_windows: 0,
                remaining: 0,
                windows: Default::default(),
                parties: vec![Participant {
                    owner: Owner::Proc(process),
                    source: true,
                    memory: true,
                    state: Party::Quarantined,
                }],
                refused: false,
                key: Key::default(),
                chain_pages: 0,
                row: true,
                accepted: None,
                legs: Vec::new(),
                deferred_commit: None,
                mp: PutMp {
                    layout,
                    peers: vec![MpPeer {
                        rank: 1,
                        proc: process,
                        shared: true,
                    }],
                    windows: BTreeMap::new(),
                    passes: 0,
                    whole: Vec::new(),
                    row: None,
                },
            },
        );
        core.party_quiesced(process);
        assert_eq!(
            super::super::claims::party_state(&core.puts[&key].parties, Owner::Proc(process)),
            Party::Quarantined
        );
        core.proc_exited(process);
        assert!(!core.puts.contains_key(&key));
    }

    #[test]
    fn the_row_checksum_sits_where_the_header_keeps_it_and_set_checksums_rewrites_the_table() {
        // A header the client encodes, read by the manager's decoder: patching the offset
        // this module assumes changes exactly the row checksum it reports.
        use flare_selene::{CacheGeometry, Plane};
        use flare_selene::{PrefixKey, geometry::fingerprint, shared::record::Header as Client};
        let rank = CacheGeometry {
            page_tokens: 64,
            page_capacity: 16,
            row_capacity: 4,
            row: Plane {
                name: "state".into(),
                address: 0x1000_0000,
                unit_bytes: 4096,
            },
            planes: ["k", "v"]
                .iter()
                .enumerate()
                .map(|(i, name)| Plane {
                    name: (*name).into(),
                    address: 0x2000_0000 + i as u64 * 0x100_0000,
                    unit_bytes: 4096,
                })
                .collect(),
            snapshot_pages: Vec::new(),
        };
        let geometry = vec![rank; 2];
        let key = PrefixKey {
            consumed_tokens: 192,
            digest: [3; 16],
        };
        let mut header = Client::new(
            &geometry,
            [1; 32],
            fingerprint(&geometry),
            key,
            None,
            [4; 16],
            3,
            std::time::SystemTime::now(),
        )
        .unwrap();
        header.row_checksum = 0x1111;
        let mut metadata = header.encode().unwrap().to_vec();
        metadata.extend(vec![0u8; header.checksum_bytes() as usize]);
        let decoded = Header::decode(&metadata).unwrap();
        assert_eq!(decoded.row_checksum, 0x1111);
        assert!(set_checksums(
            &mut metadata,
            &decoded,
            &[7, 8, 9],
            Some(0x2222)
        ));
        let patched = Header::decode(&metadata).unwrap();
        assert_eq!(patched.row_checksum, 0x2222);
        assert_eq!(
            metadata[HEADER_BYTES..HEADER_BYTES + 24]
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![7, 8, 9]
        );
        // Without a row the row checksum is left as it was.
        assert!(set_checksums(&mut metadata, &patched, &[1, 2, 3], None));
        assert_eq!(Header::decode(&metadata).unwrap().row_checksum, 0x2222);
        // The wrong number of pages, or a short block, is refused.
        assert!(!set_checksums(&mut metadata, &patched, &[1, 2], None));
        assert!(!set_checksums(
            &mut metadata[..HEADER_BYTES],
            &patched,
            &[1, 2, 3],
            None
        ));
    }
}
