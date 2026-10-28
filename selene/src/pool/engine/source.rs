//! The source role: serving exports of this instance's holdings.
//!
//! A GPU export is answered by the model worker, which pins the checkpoint;
//! this module then fills the manager's windows by D2H copies through the page
//! copier. A host-tier export is pinned by the store on the worker thread and
//! fills windows on a copy thread, serving the checksums persisted with the
//! records. In both, the pin is released only after every copy that reads it
//! has drained, and the export ends with exactly one `ExportEnd`.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{Direction, ExportChain, ExportId, ExportRefusal};

use super::{
    super::{
        codec,
        exporter::{Checksums, HostPin, HostWindow},
        net::Id,
        shared::{ExportRequest, PoolEvent},
        wire::{self, client_message::M as Out},
    },
    Engine, Owner,
    coverage::Coverage,
    hostfill::FillOwner,
    outbound::Outbound,
    requester::ExpectedRecord,
};

enum Phase {
    /// Delivered to the model worker, which has not answered.
    AtWorker,
    Pinned,
}

enum Pin {
    Gpu(ExportChain),
    Host(Arc<Mutex<Box<dyn HostPin>>>),
}

struct Export {
    phase: Phase,
    pin: Option<Pin>,
    want_row: bool,
    reserved: u64,
    deadline: Instant,
    /// The PinAck went out already (a refusal at the deadline).
    answered: bool,
    cancelled: bool,
    failed: bool,
    lost: bool,
    coverage: Option<Coverage>,
    /// Windows the copies of which are still running, by index.
    windows: HashMap<u32, wire::WindowSpec>,
    /// The windows go to the manager on the data channel (no shared memory), which the
    /// request named.
    reference: Option<wire::DataChannelRef>,
    out: Outbound,
    /// The channels the request named for the peer ranks of a multi-process instance.
    rank_channels: Vec<wire::RankChannel>,
}

#[derive(Default)]
pub(crate) struct State {
    exports: HashMap<u64, Export>,
}

impl State {
    pub(super) fn outbound(&mut self, id: u64) -> Option<&mut Outbound> {
        self.exports.get_mut(&id).map(|run| &mut run.out)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.exports.is_empty()
    }

    pub(crate) fn busy(&self) -> bool {
        self.exports.values().any(|e| e.out.jobs_outstanding > 0)
    }
}

fn reason(refusal: ExportRefusal) -> wire::ClaimReason {
    match refusal {
        ExportRefusal::NotFound => wire::ClaimReason::NotFound,
        ExportRefusal::ChainGap => wire::ClaimReason::ChainGap,
        ExportRefusal::NoHeadroom => wire::ClaimReason::NoHeadroom,
        ExportRefusal::NoWindows => wire::ClaimReason::NoWindows,
        ExportRefusal::Busy => wire::ClaimReason::SourceBusy,
    }
}

impl Engine {
    fn pin_refused(&mut self, export: u64, why: wire::ClaimReason) {
        self.emit(Out::PinAck(wire::PinAck {
            export_id: export,
            result: wire::PinResult::Refused as i32,
            refusal: why as i32,
            records: Vec::new(),
            total_pages: 0,
            row: false,
        }));
    }

    pub(crate) fn on_export_request(&mut self, now: Instant, request: wire::ExportRequest) {
        let id = request.export_id;
        if self.source.exports.contains_key(&id) {
            return;
        }
        let Some(sess) = &self.sess else {
            return;
        };
        if request.session_incarnation != sess.header.session_incarnation {
            return self.pin_refused(id, wire::ClaimReason::StaleSource);
        }
        let reachable = if request.channel.is_some() {
            self.dial.is_some()
        } else {
            sess.regions.is_some()
        };
        if !reachable
            || sess.export_slots == 0
            || self.source.exports.len() >= sess.export_slots as usize
            || self.closing
        {
            return self.pin_refused(id, wire::ClaimReason::SourceBusy);
        }
        let deadline = now + Duration::from_millis(u64::from(sess.limits.pin_deadline_ms));
        let (Some(key), Some(tier)) = (
            request.key.as_ref().and_then(codec::key_from),
            codec::tier_from(request.tier),
        ) else {
            return self.pin_refused(id, wire::ClaimReason::InvalidClaim);
        };
        let export = Export {
            phase: Phase::AtWorker,
            pin: None,
            want_row: request.want_row,
            reserved: request.reserved_bytes,
            deadline,
            answered: false,
            cancelled: false,
            failed: false,
            lost: false,
            coverage: None,
            windows: HashMap::new(),
            reference: request.channel.clone(),
            out: Outbound::default(),
            rank_channels: request.rank_channels.clone(),
        };
        self.source.exports.insert(id, export);
        self.event(PoolEvent::Export(ExportRequest {
            export: ExportId(id),
            tier,
            key,
            begin_tokens: request.begin_tokens,
            want_row: request.want_row,
            deadline,
            reserved_bytes: request.reserved_bytes,
        }));
    }

    /// The worker pinned the checkpoint and describes its chain.
    pub(crate) fn on_export_pinned(&mut self, export: ExportId, chain: ExportChain) {
        let last = chain.nodes.len().saturating_sub(1);
        let records: Vec<ExpectedRecord> = chain
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let first = if index == 0 { chain.skip_pages } else { 0 };
                ExpectedRecord {
                    key: node.key,
                    first_page: first,
                    pages: (node.pages.len() as u32).saturating_sub(first),
                    row: index == last && chain.row.is_some(),
                }
            })
            .collect();
        let row = chain.row.is_some();
        self.pinned(export, records, row, Pin::Gpu(chain));
    }

    /// The store pinned a host-tier chain for the export.
    pub(crate) fn on_host_pinned(&mut self, export: ExportId, pin: Box<dyn HostPin>) {
        let chain = pin.chain().clone();
        let records = chain
            .records
            .iter()
            .map(|r| ExpectedRecord {
                key: r.key,
                first_page: r.first_page,
                pages: r.pages,
                row: r.row,
            })
            .collect();
        self.pinned(
            export,
            records,
            chain.row,
            Pin::Host(Arc::new(Mutex::new(pin))),
        );
    }

    /// Answer an export request with the chain that was pinned for it, or
    /// release the pin if the export can no longer use it.
    fn pinned(&mut self, export: ExportId, expected: Vec<ExpectedRecord>, row: bool, pin: Pin) {
        let id = export.0;
        let release = |engine: &Engine, pin: Pin| {
            // A GPU pin is the worker's to release; a host pin releases on drop.
            if matches!(pin, Pin::Gpu(_)) {
                engine.event(PoolEvent::ExportEnded { export });
            }
        };
        let Some(run) = self.source.exports.get_mut(&id) else {
            // Nothing waits for this pin any more.
            return release(self, pin);
        };
        if !matches!(run.phase, Phase::AtWorker) {
            return;
        }
        let (page_bytes, row_bytes) = (self.st.page_record_bytes(), self.st.row_record_bytes());
        let total: u64 = expected.iter().map(|r| u64::from(r.pages)).sum();
        let bytes = total * page_bytes + if row { row_bytes } else { 0 };
        let valid = !expected.is_empty() && row == run.want_row;
        // A GPU export always has its staging reserved; a host export with none
        // reserved streams.
        let over = if matches!(pin, Pin::Gpu(_)) {
            bytes > run.reserved
        } else {
            run.reserved != 0 && bytes > run.reserved
        };
        if run.cancelled || run.lost || run.answered || !valid || over {
            let (answered, cancelled) = (run.answered, run.cancelled);
            self.source.exports.remove(&id);
            if !answered {
                let why = if cancelled {
                    wire::ClaimReason::Cancelled
                } else if valid {
                    wire::ClaimReason::NoWindows
                } else {
                    wire::ClaimReason::ChainGap
                };
                self.pin_refused(id, why);
            }
            return release(self, pin);
        }
        let records: Vec<wire::Record> = expected
            .iter()
            .map(|r| wire::Record {
                key: Some(codec::key_to(r.key)),
                first_page: r.first_page,
                pages: r.pages,
                row: r.row,
            })
            .collect();
        let rank_channels = run.rank_channels.clone();
        // A multi-process instance's peers start before the manager can send a window.
        if self.ranks.is_some() {
            let opened = match &pin {
                Pin::Gpu(chain) => {
                    self.open_export_peers(id, &records, Some(chain), &rank_channels)
                }
                Pin::Host(_) => self.open_export_peers(id, &records, None, &rank_channels),
            };
            if let Err(why) = opened {
                eprintln!("warning: export {id} refused: {why}");
                self.cancel_outbound_peers(super::ranks::XferOwner::Export(id));
                self.source.exports.remove(&id);
                self.pin_refused(id, wire::ClaimReason::SourceBusy);
                return release(self, pin);
            }
        }
        let Some(run) = self.source.exports.get_mut(&id) else {
            return release(self, pin);
        };
        run.phase = Phase::Pinned;
        run.coverage = Some(Coverage::new(expected));
        run.pin = Some(pin);
        let reference = run.reference.clone();
        self.emit(Out::PinAck(wire::PinAck {
            export_id: id,
            result: wire::PinResult::Pinned as i32,
            refusal: 0,
            records,
            total_pages: total,
            row,
        }));
        if let Some(reference) = reference {
            // The manager grants this export's windows on the channel from now on.
            match self.open_channel(Id::Export(id), &reference) {
                Ok(channel) => {
                    let run = self.source.exports.get_mut(&id).expect("present");
                    run.out.channel = Some(channel);
                    run.out.channel_open = true;
                }
                Err(_) => {
                    if let Some(run) = self.source.exports.get_mut(&id) {
                        run.failed = true;
                    }
                    self.drive_export(id);
                }
            }
        }
    }

    /// The worker did not pin the checkpoint.
    pub(crate) fn on_export_refused(&mut self, export: ExportId, why: ExportRefusal) {
        let id = export.0;
        let Some(run) = self.source.exports.get(&id) else {
            return;
        };
        if !matches!(run.phase, Phase::AtWorker) {
            return;
        }
        let (answered, cancelled) = (run.answered, run.cancelled);
        self.source.exports.remove(&id);
        if !answered {
            self.pin_refused(
                id,
                if cancelled {
                    wire::ClaimReason::Cancelled
                } else {
                    reason(why)
                },
            );
        }
    }

    pub(crate) fn on_export_cancel(&mut self, cancel: wire::ExportCancel) {
        let id = cancel.export_id;
        let Some(run) = self.source.exports.get_mut(&id) else {
            // Unknown: the idempotent answer.
            self.emit(Out::ExportEnd(wire::ExportEnd {
                export_id: id,
                status: wire::ExportEndStatus::ExportReleased as i32,
                fence: wire::FenceOutcome::Drained as i32,
            }));
            return;
        };
        run.cancelled = true;
        match run.phase {
            Phase::AtWorker => {
                // A request the worker never took is withdrawn at once.
                if self.shared.withdraw_export(ExportId(id)) {
                    self.source.exports.remove(&id);
                    self.pin_refused(id, wire::ClaimReason::Cancelled);
                }
            }
            Phase::Pinned => self.drive_export(id),
        }
    }

    pub(crate) fn on_export_fill(&mut self, fill: wire::ExportFill) {
        let id = fill.export_id;
        let Some(run) = self.source.exports.get_mut(&id) else {
            return;
        };
        if !matches!(run.phase, Phase::Pinned) || run.cancelled || run.failed {
            return;
        }
        if run.reference.is_some() {
            // This export's windows are granted on its channel.
            return;
        }
        for spec in fill.windows {
            if !self.export_window(id, spec, None) {
                break;
            }
        }
        self.drive_export(id);
    }

    /// The manager granted a window of an export over the data channel; its bytes go
    /// in `slot`.
    pub(crate) fn on_export_grant(
        &mut self,
        export: u64,
        shape: super::super::frame::Shape,
        slot: u32,
    ) {
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
        let taken = self.export_window(export, spec, Some(slot));
        if !taken {
            self.free_slot(slot);
        }
        self.drive_export(export);
    }

    /// Start filling one window. `false` when it was not taken up: the export is over, or
    /// the window is wrong and the export failed. A slot, if any, then is still the
    /// caller's.
    fn export_window(&mut self, id: u64, spec: wire::WindowSpec, slot: Option<u32>) -> bool {
        let (page_bytes, row_bytes) = self.window_shape(slot.is_some());
        let memory = self.memory();
        let Some(run) = self.source.exports.get_mut(&id) else {
            return false;
        };
        if !matches!(run.phase, Phase::Pinned) || run.cancelled || run.failed {
            return false;
        }
        let coverage = run.coverage.as_mut().expect("pinned has coverage");
        let ok = coverage.accept(&spec, page_bytes, row_bytes);
        let address = memory
            .as_ref()
            .and_then(|m| m.region(spec.region()))
            .and_then(|region| region.address(spec.offset, spec.length));
        let (Ok(_), Some(address)) = (ok, address) else {
            run.failed = true;
            self.fill_failed(id, spec.index);
            return false;
        };
        if let Some(slot) = slot {
            assert!(run.out.slots.insert(spec.index, slot).is_none());
        }
        let record = spec.record as usize;
        let first = spec.first_page as usize;
        let count = spec.page_count as usize;
        match run.pin.as_ref().expect("pinned") {
            Pin::Gpu(chain) => {
                let node = &chain.nodes[record];
                let jobs = super::super::agent::jobs_of(
                    Direction::Out,
                    spec.index,
                    count,
                    spec.row,
                    |i| node.pages[first + i],
                    chain.row,
                    |i| address + i * page_bytes as usize,
                    address + count * page_bytes as usize,
                );
                run.out.jobs_outstanding += jobs.len() as u32;
                run.windows.insert(spec.index, spec);
                let sent = run.out.fan_out(self.ranks.as_mut(), spec);
                run.failed |= sent.is_err();
                for job in jobs {
                    self.queue_job(
                        job,
                        Owner::Export {
                            export: id,
                            window: spec.index,
                        },
                    );
                }
            }
            Pin::Host(pin) => {
                let pin = pin.clone();
                run.windows.insert(spec.index, spec);
                let span = HostWindow {
                    record: spec.record,
                    first_page: spec.first_page,
                    page_count: spec.page_count,
                    row: spec.row,
                };
                let job = run.out.fill_host(
                    FillOwner::Export(id),
                    spec,
                    address,
                    slot,
                    self.ranks.as_ref(),
                    Box::new(move |dst| pin.lock().unwrap().read(&span, dst)),
                );
                run.failed |= run.out.fan_out(self.ranks.as_mut(), spec).is_err();
                if !self.submit_fill(job) {
                    let run = self.source.exports.get_mut(&id).expect("present");
                    run.out.jobs_outstanding -= 1;
                    run.out.host_filling.remove(&spec.index);
                    run.failed = true;
                    self.fill_failed(id, spec.index);
                    return true;
                }
            }
        }
        true
    }

    pub(super) fn fill_failed(&mut self, export: u64, window: u32) {
        if self
            .source
            .exports
            .get(&export)
            .is_some_and(|run| run.reference.is_some())
        {
            // The channel carries the windows: dropping it tells the manager.
            return;
        }
        self.emit(Out::WindowFilled(wire::WindowFilled {
            export_id: export,
            index: window,
            status: wire::FillStatus::FillFailed as i32,
            page_xxh3: Vec::new(),
            row_xxh3: 0,
            has_row: false,
            rank_sums: Vec::new(),
        }));
    }

    /// A D2H job of an export drained.
    pub(crate) fn on_export_job(&mut self, export: u64, window: u32, result: crate::Result<()>) {
        let Some(run) = self.source.exports.get_mut(&export) else {
            return;
        };
        run.out.jobs_outstanding = run.out.jobs_outstanding.saturating_sub(1);
        let spec = run.windows.get(&window).cloned();
        let failed_before = run.failed;
        match (result, spec) {
            (Ok(()), Some(spec)) => {
                if self.window_still_running(export, window) {
                    return;
                }
                if !failed_before {
                    self.fill_done(export, spec, None);
                }
            }
            (Err(_), _) | (_, None) => {
                if !failed_before {
                    if let Some(run) = self.source.exports.get_mut(&export) {
                        run.failed = true;
                    }
                    self.fill_failed(export, window);
                }
            }
        }
        self.drive_export(export);
    }

    pub(super) fn source_failed(&mut self, id: u64, error: crate::Error) {
        if let Some(run) = self.source.exports.get_mut(&id) {
            run.failed = true;
            eprintln!("warning: export {id}: {error}");
        }
    }

    pub(super) fn export_after_peer_window(&mut self, export: u64, index: u32) {
        let Some(run) = self.source.exports.get_mut(&export) else {
            return;
        };
        if run.failed {
            self.fill_failed(export, index);
        } else if let Some(spec) = run.windows.get(&index).copied() {
            if !self.window_still_running(export, index) {
                let sums = self
                    .source
                    .exports
                    .get_mut(&export)
                    .and_then(|run| run.out.host_sums.remove(&index));
                self.fill_done(export, spec, sums);
            }
        } else {
            run.failed = true;
            self.fill_failed(export, index);
        }
    }

    /// Whether another job of the same window is still queued or running.
    fn window_still_running(&self, export: u64, window: u32) -> bool {
        let owner_is = |o: &Owner| matches!(o, Owner::Export { export: e, window: w } if *e == export && *w == window);
        self.jobs.values().any(owner_is)
            || self.pending.iter().any(|(_, o)| owner_is(o))
            || self
                .source
                .exports
                .get(&export)
                .is_some_and(|run| run.out.running(window))
    }

    /// The window's bytes are in place: tell the manager, with the persisted
    /// checksums for a host source and fresh ones for a GPU source.
    fn fill_done(&mut self, export: u64, spec: wire::WindowSpec, persisted: Option<Checksums>) {
        let on_channel = self
            .source
            .exports
            .get(&export)
            .is_some_and(|run| run.reference.is_some());
        let sums = persisted
            .map(Ok)
            .unwrap_or_else(|| self.window_sums(spec, on_channel));
        let result = sums.and_then(|sums| {
            let run = self.source.exports.get_mut(&export).expect("present");
            run.windows.remove(&spec.index);
            run.out.filled(spec, sums, on_channel, self.ranks.is_some())
        });
        match result {
            Ok(Some((sums, rank_sums))) => self.emit(Out::WindowFilled(wire::WindowFilled {
                export_id: export,
                index: spec.index,
                status: wire::FillStatus::FillOk as i32,
                page_xxh3: sums.pages,
                row_xxh3: sums.row.unwrap_or(0),
                has_row: sums.row.is_some(),
                rank_sums,
            })),
            Ok(None) => {}
            Err(_) => {
                if let Some(run) = self.source.exports.get_mut(&export) {
                    run.failed = true;
                }
                self.fill_failed(export, spec.index);
            }
        }
    }

    /// An export's transfer failed.
    pub(crate) fn net_export_failed(&mut self, export: u64, _error: String) {
        if let Some(run) = self.source.exports.get_mut(&export) {
            run.failed = true;
            self.drive_export(export);
        }
    }

    /// An export's task is gone and its connection closed.
    pub(crate) fn net_export_closed(&mut self, export: u64) {
        let Some(run) = self.source.exports.get_mut(&export) else {
            return;
        };
        run.out.channel = None;
        run.out.channel_open = false;
        // Nothing sends from a slot any more.
        run.out.sends_outstanding = 0;
        let complete = run.coverage.as_ref().is_some_and(|c| c.complete());
        if !complete && !run.cancelled && !run.lost {
            run.failed = true;
        }
        self.drive_export(export);
    }

    /// End the export if it is over and nothing reads its pages any more.
    pub(super) fn drive_export(&mut self, id: u64) {
        let Some(run) = self.source.exports.get_mut(&id) else {
            return;
        };
        run.out.release_slots(self.net.as_deref());
        if run.out.jobs_outstanding > 0 || !matches!(run.phase, Phase::Pinned) {
            return;
        }
        let complete = run.coverage.as_ref().is_some_and(|c| c.complete());
        if !(complete || run.cancelled || run.failed || run.lost) {
            return;
        }
        // Windows still on their way to the manager are let out first, unless the export
        // is being abandoned. The task is then stopped, and nothing is ended while it
        // may still use a slot.
        let abandoned = run.cancelled || run.failed || run.lost;
        if run.out.sends_outstanding > 0 && !abandoned {
            return;
        }
        if let Some(channel) = &run.out.channel {
            channel.stop();
        }
        if run.out.channel_open {
            return;
        }
        // The peer ranks end their parts too: after their copies drained.
        if abandoned {
            self.cancel_outbound_peers(super::ranks::XferOwner::Export(id));
        }
        let run = self.source.exports.get_mut(&id).expect("present");
        if run.out.peers.parts.iter().any(|peer| peer.done.is_none()) {
            return;
        }
        let status = if run.failed {
            wire::ExportEndStatus::ExportFailed
        } else if complete && !run.cancelled && !run.lost {
            wire::ExportEndStatus::ExportComplete
        } else {
            wire::ExportEndStatus::ExportReleased
        };
        let run = self.source.exports.remove(&id).expect("present");
        if matches!(run.pin, Some(Pin::Gpu(_))) {
            self.event(PoolEvent::ExportEnded {
                export: ExportId(id),
            });
        }
        drop(run);
        self.emit(Out::ExportEnd(wire::ExportEnd {
            export_id: id,
            status: status as i32,
            fence: wire::FenceOutcome::Drained as i32,
        }));
    }

    /// Collect finished host fills and answer export requests the worker
    /// missed the deadline on.
    pub(crate) fn source_tick(&mut self, now: Instant) {
        self.collect_outbound_fills();
        let late: Vec<u64> = self
            .source
            .exports
            .iter()
            .filter(|(_, r)| matches!(r.phase, Phase::AtWorker) && !r.answered && now >= r.deadline)
            .map(|(id, _)| *id)
            .collect();
        for id in late {
            // The manager has given up; the worker's late answer is discarded
            // and its pin, if any, released.
            if let Some(run) = self.source.exports.get_mut(&id) {
                run.answered = true;
            }
            self.pin_refused(id, wire::ClaimReason::SourceBusy);
        }
    }

    /// The session is gone: pins are released after the copies drain, and the
    /// `ExportEnd` is owed to the next session. Requests the worker has not
    /// answered are dropped; a late pin is released when the worker answers.
    pub(crate) fn source_lost(&mut self) {
        let ids: Vec<u64> = self.source.exports.keys().copied().collect();
        for id in ids {
            let Some(run) = self.source.exports.get_mut(&id) else {
                continue;
            };
            run.lost = true;
            if matches!(run.phase, Phase::AtWorker) {
                self.shared.withdraw_export(ExportId(id));
                self.source.exports.remove(&id);
            } else {
                self.drive_export(id);
            }
        }
    }

    pub(crate) fn source_close(&mut self) {
        let ids: Vec<u64> = self.source.exports.keys().copied().collect();
        for id in ids {
            if let Some(run) = self.source.exports.get_mut(&id) {
                run.cancelled = true;
            }
            self.on_export_cancel(wire::ExportCancel {
                export_id: id,
                reason: 0,
            });
        }
    }

    pub(crate) fn source_quarantine(&mut self) {
        let ids: Vec<u64> = self.source.exports.keys().copied().collect();
        for id in ids {
            self.emit(Out::ExportEnd(wire::ExportEnd {
                export_id: id,
                status: wire::ExportEndStatus::ExportFailed as i32,
                fence: wire::FenceOutcome::Quarantined as i32,
            }));
        }
        // The pins stay: their copies may still be reading the pages.
        std::mem::forget(std::mem::take(&mut self.source.exports));
    }
}
