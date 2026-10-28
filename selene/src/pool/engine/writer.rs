//! The writer role: spilling a record into the manager's store.
//!
//! A put asks the manager to accept a record, fills the windows the manager
//! grants from bytes a store supplies on the host copy thread, reports each
//! window's checksums after the copy drained, and commits the record's header
//! and checksum segment. The
//! writer's own copy may be dropped only when the put ends `Stored` or
//! `Present`. A put ends, and reports once, only after every copy that read
//! its source has drained; windows left unfilled stay leased to it until the
//! fence acknowledgment that follows a refusal, a cancel or a failure.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use crate::{Error, PrefixKey, record::Header};

use super::{
    super::{
        codec,
        exporter::{Checksums, PutBytes},
        frame,
        net::Id,
        shared::{PoolEvent, PutOutcome, PutRefusal, PutRequest, PutSource},
        wire::{self, client_message::M as Out},
    },
    Engine,
    coverage::Coverage,
    hostfill::FillOwner,
    outbound::Outbound,
    requester::ExpectedRecord,
};

/// A put that has not ended by then is abandoned.
pub const PUT_DEADLINE: Duration = Duration::from_secs(30);

#[derive(PartialEq, Eq)]
enum Phase {
    Begun,
    Accepted,
    Committed,
}

struct Put {
    key: PrefixKey,
    bucket: [u8; 16],
    chain_pages: u64,
    row: bool,
    source: Arc<Mutex<Box<dyn PutBytes>>>,
    phase: Phase,
    first_page: u64,
    parent: Option<PrefixKey>,
    coverage: Option<Coverage>,
    /// Checksums of the accepted record's pages, in order, and of its row.
    sums: Vec<u64>,
    row_sum: Option<u64>,
    /// Windows granted and not yet reported filled, with the claim-order index
    /// of their first page.
    windows: HashMap<u32, (wire::WindowSpec, u64)>,
    failed: Option<Error>,
    cancelled: bool,
    cancel_sent: bool,
    refused: Option<PutOutcome>,
    stored: Option<u64>,
    fence_sent: bool,
    deadline: Instant,
    /// Windows go to the manager over the data channel (no shared memory).
    via_channel: bool,
    out: Outbound,
}

#[derive(Default)]
pub(crate) struct State {
    puts: HashMap<u64, Put>,
}

impl State {
    pub(super) fn outbound(&mut self, id: u64) -> Option<&mut Outbound> {
        self.puts.get_mut(&id).map(|put| &mut put.out)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.puts.is_empty()
    }

    pub(crate) fn busy(&self) -> bool {
        self.puts.values().any(|p| p.out.jobs_outstanding > 0)
    }
}

fn refusal(reason: i32) -> PutOutcome {
    use wire::Reason as R;
    match R::try_from(reason) {
        Ok(R::Present) => PutOutcome::Present,
        Ok(R::Inflight) => PutOutcome::Refused(PutRefusal::Inflight),
        Ok(R::Budget) => PutOutcome::Refused(PutRefusal::Budget),
        Ok(R::Oversize) => PutOutcome::Refused(PutRefusal::Oversize),
        Ok(R::Invalid) => PutOutcome::Refused(PutRefusal::Invalid),
        Ok(R::Queue) => PutOutcome::Refused(PutRefusal::Queue),
        Ok(R::Io) => PutOutcome::Refused(PutRefusal::Io),
        Ok(R::PutNotReady) => PutOutcome::Refused(PutRefusal::NotReady),
        _ => PutOutcome::Failed(Error::new("the manager refused the put without a reason")),
    }
}

impl Engine {
    /// Start every peer rank's part of an accepted put. The record's pages are the writer's
    /// chain pages from the first the manager needs.
    pub(super) fn open_put_peers(
        &mut self,
        id: u64,
        pages: u32,
        rank_channels: &[wire::RankChannel],
    ) -> std::result::Result<(), String> {
        let Some(put) = self.writer.puts.get(&id) else {
            return Err("the put's source cannot be set up for the peer ranks".into());
        };
        let Some(ranks) = self.ranks.as_mut() else {
            return Ok(());
        };
        let (page_bytes, row_bytes) = (self.st.page_record_bytes(), self.st.row_record_bytes());
        let mut opened = Vec::new();
        let open = super::super::rank::Open {
            out: true,
            kind: wire::rank_fence::Kind::Put as i32,
            fence_id: id,
            records: vec![wire::Record {
                key: Some(codec::key_to(put.key)),
                first_page: 0,
                pages,
                row: put.row,
            }],
            page_bytes,
            row_bytes,
            host: true,
            chain_first: put.first_page,
            ..Default::default()
        };
        let result = ranks.open_peers(
            super::ranks::XferOwner::Put(id),
            rank_channels,
            open,
            &mut opened,
        );
        if let Some(out) = self.writer.outbound(id) {
            out.peers.parts = opened;
        }
        result
    }
    pub(crate) fn start_put(&mut self, now: Instant, id: u64, request: PutRequest) {
        let fail = |engine: &Engine, why: &str| {
            engine.event(PoolEvent::Put {
                id,
                outcome: PutOutcome::Failed(Error::new(why)),
            })
        };
        if !self.ready() || self.closing || self.quarantined() {
            return fail(self, "the pool is not ready");
        }
        if self.memory().is_none() {
            return fail(
                self,
                "this session has neither shared memory nor a data channel to write through",
            );
        }
        if request.tail_pages > request.chain_pages {
            return fail(self, "the put does not describe a consistent record");
        }
        self.emit(Out::PutBegin(wire::PutBegin {
            put_id: id,
            key: Some(codec::key_to(request.key)),
            chain_pages: request.chain_pages,
            row: request.row,
            has_published_parent: request.parent.is_some(),
            published_parent: request.parent.map(codec::key_to),
            bucket: request.bucket.to_vec(),
            published_tail_pages: request.tail_pages,
            mode: if self.dial.is_some() {
                wire::CopyMode::DataChannel
            } else {
                wire::CopyMode::SharedWindows
            } as i32,
        }));
        self.writer.puts.insert(
            id,
            Put {
                key: request.key,
                bucket: request.bucket,
                chain_pages: request.chain_pages,
                row: request.row,
                source: match request.source {
                    PutSource::Host(bytes) => Arc::new(Mutex::new(bytes)),
                },
                phase: Phase::Begun,
                first_page: 0,
                parent: None,
                coverage: None,
                sums: Vec::new(),
                row_sum: None,
                windows: HashMap::new(),
                failed: None,
                cancelled: false,
                cancel_sent: false,
                refused: None,
                stored: None,
                fence_sent: false,
                deadline: now + PUT_DEADLINE,
                via_channel: self.dial.is_some(),
                out: Outbound::default(),
            },
        );
    }

    /// Stop a put that has not been committed. It ends `Cancelled` once its
    /// copies drained; one already committed runs to the manager's answer.
    pub(crate) fn cancel_put(&mut self, id: u64) {
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return;
        };
        put.cancelled = true;
        self.send_cancel(id);
        self.drive_put(id);
    }

    fn send_cancel(&mut self, id: u64) {
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return;
        };
        if put.phase != Phase::Committed && !put.cancel_sent {
            put.cancel_sent = true;
            self.emit(Out::PutCancel(wire::PutCancel { put_id: id }));
        }
    }

    pub(super) fn fail_put(&mut self, id: u64, error: Error) {
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return;
        };
        put.failed.get_or_insert(error);
        self.send_cancel(id);
        self.drive_put(id);
    }

    /// The record the manager accepted, as a header and a checksum segment.
    fn put_header(&self, put: &Put) -> crate::Result<Header> {
        let scope = &self.st.scope;
        Header::new(
            &scope.record_geometry(),
            scope.model_identity,
            scope.layout_fingerprint,
            put.key,
            put.parent,
            put.bucket,
            (put.chain_pages - put.first_page) as u32,
            SystemTime::now(),
        )
    }

    pub(crate) fn on_put_accept(&mut self, accept: wire::PutAccept) {
        let id = accept.put_id;
        let Some(put) = self.writer.puts.get(&id) else {
            return;
        };
        if put.phase != Phase::Begun || put.cancelled || put.failed.is_some() {
            return;
        }
        let pages = put.chain_pages.checked_sub(accept.first_page);
        let parent = accept
            .parent
            .as_ref()
            .and_then(codec::key_from)
            .filter(|_| accept.has_parent);
        let via_channel = put.via_channel;
        let Some(pages) = pages.filter(|_| accept.channel.is_some() == via_channel) else {
            return self.fail_put(
                id,
                Error::new("the manager accepted a put this client cannot carry out"),
            );
        };
        if pages == 0 && !put.row {
            return self.fail_put(
                id,
                Error::new("the manager accepted a put with nothing to write"),
            );
        }
        let put = self.writer.puts.get_mut(&id).expect("present");
        put.first_page = accept.first_page;
        put.parent = parent;
        let expected = match self.put_header(self.writer.puts.get(&id).expect("present")) {
            Ok(header) => {
                if self.writer.puts[&id].row {
                    header.length_with_row()
                } else {
                    header.row_segment
                }
            }
            Err(error) => return self.fail_put(id, error),
        };
        if accept.record_bytes != expected {
            return self.fail_put(
                id,
                Error::new(format!(
                    "the manager expects a {}-byte record, this client writes {expected}",
                    accept.record_bytes
                )),
            );
        }
        let put = self.writer.puts.get_mut(&id).expect("present");
        put.coverage = Some(Coverage::new(vec![ExpectedRecord {
            key: put.key,
            first_page: 0,
            pages: pages as u32,
            row: put.row,
        }]));
        put.sums = vec![0; pages as usize];
        put.phase = Phase::Accepted;
        if self.ranks.is_some()
            && let Err(why) = self.open_put_peers(id, pages as u32, &accept.rank_channels)
        {
            return self.fail_put(id, Error::new(why));
        }
        if let Some(reference) = &accept.channel {
            match self.open_channel(Id::Put(id), reference) {
                Ok(channel) => {
                    let put = self.writer.puts.get_mut(&id).expect("present");
                    put.out.channel = Some(channel);
                    put.out.channel_open = true;
                }
                Err(error) => self.fail_put(id, error),
            }
        }
    }

    pub(crate) fn on_put_grant(&mut self, grant: wire::PutGrant) {
        let Some(spec) = grant.window else {
            return;
        };
        self.put_window(grant.put_id, spec, None);
    }

    /// The manager granted a window over the data channel; its bytes go in `slot`.
    pub(crate) fn on_net_grant(&mut self, put: u64, shape: frame::Shape, slot: u32) {
        let Some(staging) = self.net.clone() else {
            return;
        };
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
        if !self.put_window(put, spec, Some(slot)) {
            self.free_slot(slot);
        }
    }

    /// A window of a put to fill. `false` when it was not taken up and its slot,
    /// if any, is still the caller's.
    fn put_window(&mut self, id: u64, spec: wire::WindowSpec, slot: Option<u32>) -> bool {
        let (page_bytes, row_bytes) = self.window_shape(slot.is_some());
        let memory = self.memory();
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return false;
        };
        if put.phase != Phase::Accepted || put.cancelled || put.failed.is_some() {
            return false;
        }
        let dest = match put
            .coverage
            .as_mut()
            .expect("accepted")
            .accept(&spec, page_bytes, row_bytes)
        {
            Ok(dest) => dest,
            Err(why) => {
                self.fail_put(id, Error::new(why));
                return false;
            }
        };
        let address = memory
            .as_ref()
            .and_then(|m| m.region(spec.region()))
            .and_then(|region| region.address(spec.offset, spec.length));
        let put = self.writer.puts.get_mut(&id).expect("present");
        if let Some(slot) = slot {
            assert!(put.out.slots.insert(spec.index, slot).is_none());
        }
        let Some(address) = address else {
            self.fail_put(id, Error::new("the granted window lies outside its region"));
            return true;
        };
        let first = put.first_page + dest;
        put.windows.insert(spec.index, (spec, dest));
        let bytes = put.source.clone();
        let job = put.out.fill_host(
            FillOwner::Put(id),
            spec,
            address,
            slot,
            self.ranks.as_ref(),
            Box::new(move |dst| {
                bytes
                    .lock()
                    .unwrap()
                    .fill(first, spec.page_count, spec.row, dst)
            }),
        );
        let fan_out = put.out.fan_out(self.ranks.as_mut(), spec);
        if !self.submit_fill(job) {
            let put = self.writer.puts.get_mut(&id).expect("present");
            put.out.jobs_outstanding -= 1;
            put.out.host_filling.remove(&spec.index);
            self.fail_put(id, Error::new("the host copy thread cannot start"));
            return true;
        }
        if let Err(error) = fan_out {
            self.fail_put(id, error);
        }
        true
    }

    /// After a shared peer reported a window: finish the window if its other copies are in.
    pub(super) fn put_after_peer_window(&mut self, id: u64, window: u32) {
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return;
        };
        if put.failed.is_some() {
            self.send_cancel(id);
        } else if !put.out.running(window) && put.windows.contains_key(&window) {
            let sums = put.out.host_sums.remove(&window);
            self.window_filled(id, window, sums);
        }
        self.drive_put(id);
    }

    /// The window's bytes are in place: report its persisted checksums.
    fn window_filled(&mut self, id: u64, window: u32, supplied: Option<Checksums>) {
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return;
        };
        if put.failed.is_some() || put.cancelled || put.refused.is_some() || put.stored.is_some() {
            return;
        }
        let Some((spec, dest)) = put.windows.remove(&window) else {
            return;
        };
        let Some(sums) = supplied.filter(|s| s.pages.len() == spec.page_count as usize) else {
            return self.fail_put(id, Error::new("a window's checksums are missing"));
        };
        // A multi-process writer cannot state whole-page checksums; the manager computes the
        // ones the record keeps.
        if self.ranks.is_none() {
            for (i, sum) in sums.pages.iter().enumerate() {
                put.sums[dest as usize + i] = *sum;
            }
            if spec.row {
                put.row_sum = sums.row;
            }
        }
        let result = put
            .out
            .filled(spec, sums, put.via_channel, self.ranks.is_some());
        match result {
            Ok(Some((sums, rank_sums))) => self.emit(Out::PutFilled(wire::PutFilled {
                put_id: id,
                index: window,
                page_xxh3: sums.pages,
                row_xxh3: sums.row.unwrap_or(0),
                has_row: sums.row.is_some(),
                rank_sums,
            })),
            Ok(None) => {}
            Err(error) => self.fail_put(id, error),
        }
    }

    /// A put's transfer failed.
    pub(crate) fn net_put_failed(&mut self, put: u64, error: String) {
        if self.writer.puts.contains_key(&put) {
            self.fail_put(put, Error::new(error));
        }
    }

    /// A put's task is gone and its connection closed.
    pub(crate) fn net_put_closed(&mut self, put: u64) {
        let Some(open) = self.writer.puts.get_mut(&put) else {
            return;
        };
        open.out.channel = None;
        open.out.channel_open = false;
        // Nothing sends from a slot any more.
        open.out.sends_outstanding = 0;
        let early = open.stored.is_none()
            && open.refused.is_none()
            && open.failed.is_none()
            && !open.cancelled
            && open.phase != Phase::Committed;
        if early {
            self.fail_put(
                put,
                Error::new("the data channel closed before the put was committed"),
            );
        }
        self.drive_put(put);
    }

    pub(crate) fn on_stored(&mut self, stored: wire::Stored) {
        if let Some(put) = self.writer.puts.get_mut(&stored.put_id) {
            put.stored = Some(stored.bytes);
            self.drive_put(stored.put_id);
        }
    }

    pub(crate) fn on_put_refused(&mut self, refused: wire::PutRefused) {
        if let Some(put) = self.writer.puts.get_mut(&refused.put_id) {
            put.refused.get_or_insert(refusal(refused.reason));
            self.drive_put(refused.put_id);
        }
    }

    /// Commit when every window is filled; end the put when it is over and its
    /// source is no longer read.
    pub(super) fn drive_put(&mut self, id: u64) {
        let Some(put) = self.writer.puts.get_mut(&id) else {
            return;
        };
        put.out.release_slots(self.net.as_deref());
        let over =
            put.stored.is_some() || put.refused.is_some() || put.failed.is_some() || put.cancelled;
        if !over {
            let ready = put.phase == Phase::Accepted
                && put.coverage.as_ref().is_some_and(|c| c.complete())
                && put.windows.is_empty()
                && put.out.peers.windows.is_empty()
                && put.out.jobs_outstanding == 0
                && put.out.sends_outstanding == 0;
            if ready {
                return self.commit_put(id);
            }
            return;
        }
        // The put is over: the task stops, and nothing is acknowledged or freed
        // while a copy reads a window or the task may still use a slot.
        if let Some(channel) = &put.out.channel {
            channel.stop();
        }
        if put.out.jobs_outstanding > 0 || put.out.channel_open {
            return;
        }
        // The peer ranks end their parts too: after their copies drained.
        self.cancel_outbound_peers(super::ranks::XferOwner::Put(id));
        let put = self.writer.puts.get_mut(&id).expect("present");
        if put.out.peers.parts.iter().any(|peer| peer.done.is_none()) {
            return;
        }
        let fenced = put.stored.is_some() || put.phase == Phase::Begun || put.fence_sent;
        if !fenced {
            put.fence_sent = true;
            self.emit(Out::FenceAck(wire::FenceAck {
                subject: Some(wire::fence_ack::Subject::PutId(id)),
                outcome: wire::FenceOutcome::Drained as i32,
            }));
        }
        let put = self.writer.puts.remove(&id).expect("present");
        let outcome = if let Some(bytes) = put.stored {
            PutOutcome::Stored { bytes }
        } else if let Some(outcome) = put.refused {
            outcome
        } else if put.cancelled {
            PutOutcome::Cancelled
        } else {
            PutOutcome::Failed(put.failed.unwrap_or_else(|| Error::new("the put failed")))
        };
        self.event(PoolEvent::Put { id, outcome });
    }

    fn commit_put(&mut self, id: u64) {
        let put = &self.writer.puts[&id];
        let metadata = self.put_header(put).and_then(|mut header| {
            header.row_checksum = put.row_sum.unwrap_or(0);
            let mut bytes = header.encode()?.to_vec();
            let mut sums = vec![0u8; header.checksum_bytes() as usize];
            for (i, sum) in put.sums.iter().enumerate() {
                sums[i * 8..i * 8 + 8].copy_from_slice(&sum.to_le_bytes());
            }
            bytes.extend(sums);
            Ok(bytes)
        });
        match metadata {
            Err(error) => self.fail_put(id, error),
            Ok(metadata) => {
                self.writer.puts.get_mut(&id).expect("present").phase = Phase::Committed;
                self.emit(Out::PutCommit(wire::PutCommit {
                    put_id: id,
                    metadata_offset: 0,
                    metadata,
                }));
            }
        }
    }

    pub(crate) fn writer_tick(&mut self, now: Instant) {
        let late: Vec<u64> = self
            .writer
            .puts
            .iter()
            .filter(|(_, p)| {
                now >= p.deadline && p.failed.is_none() && p.stored.is_none() && p.refused.is_none()
            })
            .map(|(id, _)| *id)
            .collect();
        for id in late {
            self.fail_put(id, Error::new("the put ran out of time"));
        }
    }

    /// The session is gone: every put fails once its copies drained, and the
    /// fence for its windows is owed to the next session.
    pub(crate) fn writer_lost(&mut self) {
        let ids: Vec<u64> = self.writer.puts.keys().copied().collect();
        for id in ids {
            self.fail_put(id, Error::new("the pool session was lost"));
        }
    }

    pub(crate) fn writer_close(&mut self) {
        let ids: Vec<u64> = self.writer.puts.keys().copied().collect();
        for id in ids {
            self.cancel_put(id);
        }
    }

    pub(crate) fn writer_quarantine(&mut self) {
        let ids: Vec<u64> = self.writer.puts.keys().copied().collect();
        for id in ids {
            self.emit(Out::FenceAck(wire::FenceAck {
                subject: Some(wire::fence_ack::Subject::PutId(id)),
                outcome: wire::FenceOutcome::Quarantined as i32,
            }));
        }
        std::mem::forget(std::mem::take(&mut self.writer.puts));
    }
}
