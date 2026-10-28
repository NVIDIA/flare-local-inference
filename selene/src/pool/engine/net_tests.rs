//! The engine on a session without shared memory: windows pass through staging
//! slots and the data-channel tasks, played here by the test, report what they did.

use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
};

use tokio::sync::mpsc;

use crate::{
    Registration, Result,
    pool::{
        frame::{NO_SLOT, Shape, Window},
        net::{Channel, Dial, Event, Filled, Id, Open, Staging},
        shared::{Command, PutSource},
        wire::{self, client_message::M as Out, server_message::M as In},
    },
    record::checksum,
};

use super::tests::{
    PAGE, ROW, Rig, chain, export_request, fetched, gpu_with_chain, host_put, pattern, puts,
    record_len, records, single_part, terminal,
};

struct Reg;
impl Registration for Reg {
    fn device_address(&self) -> u64 {
        0
    }
}
struct OkRegistrar;
impl crate::HostRegistrar for OkRegistrar {
    fn register(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Registration>> {
        Ok(Box::new(Reg))
    }
    fn map_split(&self, _: &[(usize, i32, usize)]) -> Result<Box<dyn crate::SplitMapping>> {
        Err("not used".into())
    }
}

/// Opens detached channels and remembers what it was asked for.
#[derive(Default)]
struct FakeDial {
    opened: Mutex<Vec<(Id, wire::DataChannelRef)>>,
    filled: Mutex<HashMap<u64, mpsc::UnboundedReceiver<Filled>>>,
    refuse: Mutex<bool>,
}

impl Dial for FakeDial {
    fn open(&self, request: Open) -> io::Result<Channel> {
        if *self.refuse.lock().unwrap() {
            return Err(io::Error::other("refused"));
        }
        self.opened
            .lock()
            .unwrap()
            .push((request.id, request.reference.clone()));
        let fills = matches!(request.id, Id::Put(_) | Id::Export(_));
        let (channel, receiver) = Channel::detached(fills);
        if let (Id::Put(id) | Id::Export(id), Some(receiver)) = (request.id, receiver) {
            self.filled.lock().unwrap().insert(id, receiver);
        }
        Ok(channel)
    }
}

struct NetRig {
    rig: Rig,
    dial: Arc<FakeDial>,
    staging: Arc<Staging>,
}

fn net_rig() -> NetRig {
    let mut rig = Rig::new();
    let staging = Arc::new(Staging::new(&OkRegistrar, 0, 4, 64 * 1024).unwrap());
    rig.engine.net = Some(staging.clone());
    let dial = Arc::new(FakeDial::default());
    rig.engine.set_dial(dial.clone());
    rig.open_session(None);
    NetRig { rig, dial, staging }
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

impl NetRig {
    /// Report events as the tasks would, and let the engine act on them.
    fn report(&mut self, events: Vec<Event>) -> Vec<wire::ClientMessage> {
        for event in events {
            self.rig.engine.net_events.push(event);
        }
        self.rig.engine.pump(self.rig.now);
        self.rig.engine.take_out()
    }

    /// Put `bytes` in a free slot, as a task that read a window would.
    fn slot_with(&self, bytes: &[u8]) -> u32 {
        let slot = self.staging.take_now().expect("a free slot");
        self.staging
            .region()
            .bytes_mut(self.staging.offset(slot), bytes.len() as u64)
            .unwrap()
            .copy_from_slice(bytes);
        slot
    }

    /// A claim's window the manager would send: its frame and the bytes in a slot.
    fn window(&self, claim: u64, shape: Shape, seed: u8, corrupt: bool) -> Event {
        let mut bytes = pattern(seed, shape.length as usize);
        let sums: Vec<u64> = (0..shape.page_count as usize)
            .map(|i| checksum(&bytes[i * PAGE as usize..(i + 1) * PAGE as usize]))
            .collect();
        let row = if shape.row {
            checksum(&bytes[shape.page_count as usize * PAGE as usize..])
        } else {
            0
        };
        if corrupt {
            bytes[3] ^= 1;
        }
        let slot = self.slot_with(&bytes);
        Event::Window {
            claim,
            window: Window {
                shape,
                page_xxh3: sums,
                row_xxh3: row,
                slot: NO_SLOT,
            },
            slot,
        }
    }
}

fn shape(index: u32, record: u32, first_page: u32, count: u32, row: bool) -> Shape {
    Shape {
        index,
        record,
        first_page,
        page_count: count,
        row,
        length: u64::from(count) * PAGE + if row { ROW } else { 0 },
    }
}

fn channel_ref(token: u8) -> wire::DataChannelRef {
    wire::DataChannelRef {
        transport: wire::DataTransport::Tcp as i32,
        address: "manager:9201".into(),
        token: vec![token; 32],
    }
}

fn granted(
    claim: u64,
    fetch: &crate::pool::PoolFetch,
    channel: Option<wire::DataChannelRef>,
) -> In {
    In::ClaimGranted(wire::ClaimGranted {
        claim_id: claim,
        records: records(fetch),
        total_pages: fetch.part.part.pages as u64,
        row: fetch.part.part.row,
        channel,
        window_credits: 2,
        retained: false,
        ..Default::default()
    })
}

/// A started restore whose claim was granted a channel: the claim is id 1.
fn granted_restore(n: &mut NetRig) -> crate::pool::PoolFetch {
    let fetch = single_part();
    let out = n.rig.command(Command::Fetch(fetch.clone()));
    let claim = pick!(out, Claim);
    assert_eq!(claim.len(), 1);
    assert_eq!(
        claim[0].mode,
        wire::CopyMode::DataChannel as i32,
        "a session without shared memory asks for the data channel"
    );
    n.rig.send(granted(1, &fetch, Some(channel_ref(7))));
    fetch
}

#[test]
fn a_claim_over_the_data_channel_copies_each_window_out_of_its_slot_and_returns_it() {
    let mut n = net_rig();
    granted_restore(&mut n);
    assert_eq!(
        n.dial.opened.lock().unwrap().as_slice(),
        [(Id::Claim(1), channel_ref(7))],
        "the grant's channel is opened"
    );
    let free = n.staging.free_slots();
    let first = n.window(1, shape(0, 0, 0, 2, false), 1, false);
    let second = n.window(1, shape(1, 1, 0, 1, true), 2, false);
    assert_eq!(n.staging.free_slots(), free - 2);
    let out = n.report(vec![first, second]);
    assert!(
        pick!(out, WindowReleased).is_empty(),
        "no shared window is released; the slot is"
    );
    // The copies read the slots: they are not free until the copies drained.
    let submitted = n.rig.copier.lock().unwrap().submitted.len();
    assert!(submitted >= 2);
    assert_eq!(n.staging.free_slots(), free - 2);
    let out = n.rig.drain_copies();
    assert_eq!(n.staging.free_slots(), free, "both slots are back");
    assert!(
        pick!(out, CopyResult).is_empty(),
        "the verdict waits for the manager to end the transfer"
    );
    // The manager ends the transfer and the task exits.
    let out = n.report(vec![Event::End { claim: 1 }]);
    assert!(pick!(out, CopyResult).is_empty(), "the task is still there");
    let out = n.report(vec![Event::Closed { id: Id::Claim(1) }]);
    let results = pick!(out, CopyResult);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].outcome, wire::CopyOutcome::CopyOk as i32);
    let out = n.rig.send(terminal(
        1,
        wire::ClaimState::Copied,
        wire::ClaimReason::ReasonNone,
        true,
    ));
    assert_eq!(pick!(out, FenceAck).len(), 1);
    assert_eq!(fetched(&n.rig.events()).len(), 1);
    // The pages landed where the plan said: the first window's bytes are page 10.
    let copier = n.rig.copier.lock().unwrap();
    assert_eq!(
        copier.gpu_pages[&10],
        pattern(1, PAGE as usize)[..PAGE as usize]
    );
    assert_eq!(
        copier.gpu_rows[&9],
        pattern(2, (PAGE + ROW) as usize)[PAGE as usize..]
    );
}

#[test]
fn a_transfer_that_ends_short_fails_the_claim_once_its_task_is_gone() {
    let mut n = net_rig();
    granted_restore(&mut n);
    let first = n.window(1, shape(0, 0, 0, 2, false), 1, false);
    n.report(vec![first]);
    n.rig.drain_copies();
    let out = n.report(vec![Event::End { claim: 1 }]);
    assert!(
        pick!(out, CopyResult).is_empty(),
        "nothing is reported while the task may still write"
    );
    let out = n.report(vec![Event::Closed { id: Id::Claim(1) }]);
    let results = pick!(out, CopyResult);
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].outcome,
        wire::CopyOutcome::CopyErrorOutcome as i32
    );
    n.rig.send(terminal(
        1,
        wire::ClaimState::Failed,
        wire::ClaimReason::CopyError,
        true,
    ));
    let events = fetched(&n.rig.events());
    assert!(
        events[0]
            .1
            .as_ref()
            .is_err_and(|e| e.contains("ended before"))
    );
}

#[test]
fn a_page_that_fails_its_checksum_in_a_slot_is_never_copied_and_frees_the_slot() {
    let mut n = net_rig();
    granted_restore(&mut n);
    let free = n.staging.free_slots();
    let bad = n.window(1, shape(0, 0, 0, 2, false), 1, true);
    let out = n.report(vec![bad]);
    assert!(n.rig.copier.lock().unwrap().submitted.is_empty());
    assert_eq!(n.staging.free_slots(), free, "the slot is back at once");
    assert!(pick!(out, WindowReleased).is_empty());
    let out = n.report(vec![Event::Closed { id: Id::Claim(1) }]);
    let results = pick!(out, CopyResult);
    assert_eq!(results[0].outcome, wire::CopyOutcome::CopyChecksum as i32);
}

#[test]
fn a_failed_channel_fails_the_restore_and_the_fence_waits_for_the_task() {
    let mut n = net_rig();
    granted_restore(&mut n);
    let first = n.window(1, shape(0, 0, 0, 2, false), 1, false);
    let out = n.report(vec![
        first,
        Event::Failed {
            id: Id::Claim(1),
            error: "reset".into(),
        },
    ]);
    // The copy of the first window is still running: nothing is said yet.
    assert!(pick!(out, CopyResult).is_empty());
    let out = n.rig.drain_copies();
    assert!(
        pick!(out, CopyResult).is_empty(),
        "the task has not reported that it is gone"
    );
    let out = n.report(vec![Event::Closed { id: Id::Claim(1) }]);
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyErrorOutcome as i32
    );
    let out = n.rig.send(terminal(
        1,
        wire::ClaimState::Failed,
        wire::ClaimReason::CopyError,
        true,
    ));
    assert_eq!(pick!(out, FenceAck).len(), 1);
    let events = fetched(&n.rig.events());
    assert!(events[0].1.as_ref().is_err_and(|e| e.contains("reset")));
    assert_eq!(n.staging.free_slots(), 4);
}

#[test]
fn a_window_for_a_claim_that_is_gone_frees_its_slot() {
    let mut n = net_rig();
    let stray = n.window(9, shape(0, 0, 0, 1, false), 1, false);
    assert_eq!(n.staging.free_slots(), 3);
    n.report(vec![stray]);
    assert_eq!(n.staging.free_slots(), 4);
}

#[test]
fn a_grant_must_agree_with_the_session_about_the_data_channel() {
    let mut n = net_rig();
    let fetch = single_part();
    n.rig.command(Command::Fetch(fetch.clone()));
    // A channel-less grant on a session that needs one is abandoned.
    let out = n.rig.send(granted(1, &fetch, None));
    assert!(n.dial.opened.lock().unwrap().is_empty());
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyAbandoned as i32
    );
    // A shared-memory session refuses a grant that names a channel.
    let mut shared = Rig::new();
    shared.open();
    shared.command(Command::Fetch(fetch.clone()));
    let out = shared.send(granted(1, &fetch, Some(channel_ref(1))));
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyAbandoned as i32
    );
}

#[test]
fn a_dial_that_cannot_open_the_channel_fails_the_claim_with_its_grant_held() {
    let mut n = net_rig();
    *n.dial.refuse.lock().unwrap() = true;
    let fetch = single_part();
    n.rig.command(Command::Fetch(fetch.clone()));
    let out = n.rig.send(granted(1, &fetch, Some(channel_ref(1))));
    assert_eq!(
        pick!(out, CopyResult)[0].outcome,
        wire::CopyOutcome::CopyErrorOutcome as i32
    );
}

#[test]
fn a_restore_needs_shared_memory_or_a_data_channel() {
    let mut rig = Rig::new();
    rig.open_session(None);
    let out = rig.command(Command::Fetch(single_part()));
    assert!(pick!(out, Claim).is_empty());
    let events = fetched(&rig.events());
    assert!(
        events[0]
            .1
            .as_ref()
            .is_err_and(|e| e.contains("neither shared memory nor a data channel"))
    );
}

#[test]
fn a_lost_session_waits_for_the_task_before_it_owes_the_fence() {
    let mut n = net_rig();
    granted_restore(&mut n);
    n.rig.engine.session_lost(n.rig.now);
    // The fence is owed to the next session, and only once the task is gone.
    assert!(!n.rig.engine.quiet());
    n.report(vec![Event::Closed { id: Id::Claim(1) }]);
    assert!(n.rig.engine.quiet());
    let out = n.rig.open_session(None);
    assert_eq!(pick!(out, FenceAck).len(), 1, "owed from before the loss");
}

// ---- puts ---------------------------------------------------------------------

fn put_accept(put: u64, record_bytes: u64, channel: Option<wire::DataChannelRef>) -> In {
    In::PutAccept(wire::PutAccept {
        put_id: put,
        first_page: 0,
        has_parent: false,
        parent: None,
        record_bytes,
        channel,
        ..Default::default()
    })
}

#[test]
fn a_put_over_the_data_channel_fills_granted_slots_and_commits_after_the_last_send() {
    let mut n = net_rig();
    let out = n.rig.command(Command::Put {
        id: 7,
        request: host_put(None),
    });
    let begin = pick!(out, PutBegin);
    assert_eq!(begin[0].mode, wire::CopyMode::DataChannel as i32);
    // A shared-window accept is not for this session; a channel accept is.
    n.rig
        .send(put_accept(7, record_len(3), Some(channel_ref(3))));
    assert_eq!(n.dial.opened.lock().unwrap().len(), 1);
    assert_eq!(n.dial.opened.lock().unwrap()[0].0, Id::Put(7));
    // The task read a grant into a slot and reports it.
    let slot = n.staging.take_now().unwrap();
    let out = n.report(vec![Event::Granted {
        id: Id::Put(7),
        shape: shape(0, 0, 0, 3, true),
        slot,
    }]);
    assert!(
        pick!(out, PutFilled).is_empty(),
        "the message path is not used"
    );
    for _ in 0..200 {
        n.rig.tick(std::time::Duration::ZERO);
        if !n.dial.filled.lock().unwrap().get(&7).unwrap().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut receivers = n.dial.filled.lock().unwrap();
    let filled = receivers
        .get_mut(&7)
        .unwrap()
        .try_recv()
        .expect("a filled window");
    assert_eq!(
        (filled.index, filled.slot, filled.length),
        (0, slot, 3 * PAGE + ROW)
    );
    assert_eq!(filled.page_xxh3.len(), 3);
    let bytes = n
        .staging
        .region()
        .bytes(n.staging.offset(slot), 3 * PAGE + ROW)
        .unwrap();
    assert_eq!(bytes[..PAGE as usize], pattern(20, PAGE as usize)[..]);
    assert_eq!(
        filled.page_xxh3[1],
        checksum(&bytes[PAGE as usize..2 * PAGE as usize])
    );
    assert_eq!(filled.row_xxh3, Some(checksum(&bytes[3 * PAGE as usize..])));
    drop(receivers);
    // Not committed until the window has left the slot.
    let out = n.report(vec![]);
    assert!(pick!(out, PutCommit).is_empty());
    assert_eq!(n.staging.free_slots(), 3);
    let out = n.report(vec![Event::Sent {
        id: Id::Put(7),
        index: 0,
        slot,
    }]);
    assert_eq!(n.staging.free_slots(), 4);
    assert_eq!(
        pick!(out, PutCommit).len(),
        1,
        "committed after the last send"
    );
    let out = n.rig.send(In::Stored(wire::Stored {
        put_id: 7,
        bytes: record_len(3),
    }));
    assert!(
        pick!(out, FenceAck).is_empty(),
        "a stored put leaves nothing leased"
    );
    // The put is over once its task is gone.
    assert!(puts(&n.rig.events()).is_empty());
    n.report(vec![Event::Closed { id: Id::Put(7) }]);
    let ended = puts(&n.rig.events());
    assert_eq!(ended.len(), 1);
    assert!(ended[0].1.starts_with("Stored"));
}

#[test]
fn a_put_that_fails_on_the_channel_cancels_frees_its_slots_and_fences_after_the_task() {
    let mut n = net_rig();
    n.rig.command(Command::Put {
        id: 8,
        request: host_put(None),
    });
    n.rig
        .send(put_accept(8, record_len(3), Some(channel_ref(3))));
    let slot = n.staging.take_now().unwrap();
    n.report(vec![Event::Granted {
        id: Id::Put(8),
        shape: shape(0, 0, 0, 3, true),
        slot,
    }]);
    for _ in 0..200 {
        n.rig.tick(std::time::Duration::ZERO);
        if !n.dial.filled.lock().unwrap().get(&8).unwrap().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_eq!(n.staging.free_slots(), 3);
    let out = n.report(vec![Event::Failed {
        id: Id::Put(8),
        error: "reset".into(),
    }]);
    assert_eq!(pick!(out, PutCancel).len(), 1);
    assert!(
        pick!(out, FenceAck).is_empty(),
        "the task has not reported that it is gone"
    );
    let out = n.report(vec![Event::Closed { id: Id::Put(8) }]);
    assert_eq!(pick!(out, FenceAck).len(), 1);
    assert_eq!(
        n.staging.free_slots(),
        4,
        "the slot of the unsent window is back"
    );
    let ended = puts(&n.rig.events());
    assert!(ended[0].1.starts_with("Failed"), "{ended:?}");
}

#[test]
fn an_accept_that_disagrees_with_the_session_about_the_channel_fails_the_put() {
    let mut n = net_rig();
    n.rig.command(Command::Put {
        id: 9,
        request: host_put(None),
    });
    // No channel named, on a session that needs one.
    let out = n.rig.send(put_accept(9, record_len(3), None));
    assert!(n.dial.opened.lock().unwrap().is_empty());
    assert_eq!(pick!(out, PutCancel).len(), 1);
    assert!(puts(&n.rig.events())[0].1.starts_with("Failed"));
}

#[test]
fn a_put_on_a_session_with_nothing_to_copy_through_fails_at_once() {
    let mut rig = Rig::new();
    rig.open_session(None);
    rig.command(Command::Put {
        id: 1,
        request: host_put(None),
    });
    let ended = puts(&rig.events());
    assert!(
        ended[0]
            .1
            .contains("neither shared memory nor a data channel")
    );
}

// ---- exports through the manager --------------------------------------------------

/// An export request of a session without shared memory, naming the channel its windows go to.
fn channel_export(id: u64) -> In {
    let In::ExportRequest(mut request) = export_request(id, 3 * PAGE + ROW) else {
        unreachable!();
    };
    request.channel = Some(channel_ref(5));
    In::ExportRequest(request)
}

/// The export is pinned and its channel dialed; returns its id.
fn channel_export_pinned(n: &mut NetRig) -> u64 {
    gpu_with_chain(&n.rig);
    n.rig.send(channel_export(40));
    n.rig.events();
    let out = n.rig.command(Command::ExportPinned {
        export: crate::ExportId(40),
        chain: chain(),
    });
    assert_eq!(pick!(out, PinAck)[0].result, wire::PinResult::Pinned as i32);
    assert_eq!(n.dial.opened.lock().unwrap().len(), 1);
    assert_eq!(n.dial.opened.lock().unwrap()[0].0, Id::Export(40));
    assert_eq!(n.dial.opened.lock().unwrap()[0].1, channel_ref(5));
    40
}

#[test]
fn an_export_over_the_data_channel_fills_granted_slots_and_ends_after_the_last_send() {
    let mut n = net_rig();
    let id = channel_export_pinned(&mut n);
    // The manager grants the windows on the channel, not in an ExportFill.
    let first = n.staging.take_now().unwrap();
    let second = n.staging.take_now().unwrap();
    let out = n.report(vec![
        Event::Granted {
            id: Id::Export(id),
            shape: shape(0, 0, 0, 2, false),
            slot: first,
        },
        Event::Granted {
            id: Id::Export(id),
            shape: shape(1, 1, 0, 1, true),
            slot: second,
        },
    ]);
    assert!(pick!(out, WindowFilled).is_empty());
    n.rig.drain_copies();
    {
        let mut receivers = n.dial.filled.lock().unwrap();
        let receiver = receivers.get_mut(&id).unwrap();
        let one = receiver.try_recv().expect("the first window");
        assert_eq!((one.index, one.slot, one.length), (0, first, 2 * PAGE));
        let bytes = n
            .staging
            .region()
            .bytes(n.staging.offset(first), 2 * PAGE)
            .unwrap();
        assert_eq!(bytes[..PAGE as usize], pattern(20, PAGE as usize)[..]);
        assert_eq!(one.page_xxh3[1], checksum(&bytes[PAGE as usize..]));
        let two = receiver.try_recv().expect("the second window");
        assert_eq!((two.index, two.slot, two.length), (1, second, PAGE + ROW));
        assert!(two.row_xxh3.is_some());
    }
    // Nothing ends while a window is still in a slot.
    let out = n.report(vec![Event::Sent {
        id: Id::Export(id),
        index: 0,
        slot: first,
    }]);
    assert!(pick!(out, ExportEnd).is_empty());
    let out = n.report(vec![Event::Sent {
        id: Id::Export(id),
        index: 1,
        slot: second,
    }]);
    assert_eq!(n.staging.free_slots(), 4);
    // The task is stopped and the export ends when it is gone.
    assert!(pick!(out, ExportEnd).is_empty());
    let out = n.report(vec![Event::Closed { id: Id::Export(id) }]);
    let end = pick!(out, ExportEnd);
    assert_eq!(end.len(), 1);
    assert_eq!(end[0].status, wire::ExportEndStatus::ExportComplete as i32);
    assert_eq!(end[0].fence, wire::FenceOutcome::Drained as i32);
    assert!(
        n.rig
            .events()
            .iter()
            .any(|e| matches!(e, crate::pool::PoolEvent::ExportEnded { export } if *export == crate::ExportId(id))),
        "the checkpoint's pin is released"
    );
}

#[test]
fn an_export_cancelled_mid_relay_stops_its_task_frees_its_slots_and_ends_released() {
    let mut n = net_rig();
    let id = channel_export_pinned(&mut n);
    let slot = n.staging.take_now().unwrap();
    n.report(vec![Event::Granted {
        id: Id::Export(id),
        shape: shape(0, 0, 0, 2, false),
        slot,
    }]);
    n.rig.drain_copies();
    assert_eq!(n.staging.free_slots(), 3);
    let out = n.rig.send(In::ExportCancel(wire::ExportCancel {
        export_id: id,
        reason: wire::ClaimReason::Cancelled as i32,
    }));
    assert!(
        pick!(out, ExportEnd).is_empty(),
        "the task may still use the slot"
    );
    let out = n.report(vec![Event::Closed { id: Id::Export(id) }]);
    let end = pick!(out, ExportEnd);
    assert_eq!(end.len(), 1);
    assert_eq!(end[0].status, wire::ExportEndStatus::ExportReleased as i32);
    assert_eq!(
        n.staging.free_slots(),
        4,
        "the unsent window's slot is back"
    );
}

#[test]
fn a_window_out_of_order_on_the_channel_fails_the_export() {
    let mut n = net_rig();
    let id = channel_export_pinned(&mut n);
    let slot = n.staging.take_now().unwrap();
    let out = n.report(vec![Event::Granted {
        id: Id::Export(id),
        shape: shape(1, 1, 0, 1, true),
        slot,
    }]);
    assert!(pick!(out, ExportEnd).is_empty(), "waits for the task");
    assert_eq!(
        n.staging.free_slots(),
        4,
        "the refused window's slot is back"
    );
    let out = n.report(vec![Event::Closed { id: Id::Export(id) }]);
    let end = pick!(out, ExportEnd);
    assert_eq!(end[0].status, wire::ExportEndStatus::ExportFailed as i32);
}

#[test]
fn an_export_whose_task_fails_is_ended_failed_once_the_task_is_gone() {
    let mut n = net_rig();
    let id = channel_export_pinned(&mut n);
    let out = n.report(vec![Event::Failed {
        id: Id::Export(id),
        error: "reset".into(),
    }]);
    assert!(pick!(out, ExportEnd).is_empty());
    let out = n.report(vec![Event::Closed { id: Id::Export(id) }]);
    assert_eq!(
        pick!(out, ExportEnd)[0].status,
        wire::ExportEndStatus::ExportFailed as i32
    );
}

#[test]
fn an_export_naming_a_channel_is_refused_by_a_session_that_cannot_dial() {
    let mut rig = Rig::new();
    rig.open_session(None);
    gpu_with_chain(&rig);
    let out = rig.send(channel_export(41));
    let ack = pick!(out, PinAck);
    assert_eq!(ack[0].result, wire::PinResult::Refused as i32);
}

#[test]
fn disconnect_keeps_export_staging_until_delayed_d2h_drains() {
    let mut n = net_rig();
    let id = channel_export_pinned(&mut n);
    let slot = n.staging.take_now().unwrap();
    n.report(vec![Event::Granted {
        id: Id::Export(id),
        shape: shape(0, 0, 0, 2, false),
        slot,
    }]);
    assert!(n.rig.copier.lock().unwrap().queued() > 0);
    let out = n.report(vec![Event::Closed { id: Id::Export(id) }]);
    assert!(pick!(out, ExportEnd).is_empty());
    assert_eq!(n.staging.free_slots(), 3);
    n.rig.drain_copies();
    assert_eq!(n.staging.free_slots(), 4);
    n.report(vec![Event::Closed { id: Id::Export(id) }]);
    assert_eq!(n.staging.free_slots(), 4);
}

#[test]
fn disconnect_keeps_put_staging_until_blocked_host_fill_finishes() {
    use crate::pool::{Checksums, PutBytes};
    let (started, start) = std::sync::mpsc::channel();
    let (release, wait) = std::sync::mpsc::channel();
    struct Blocked {
        started: std::sync::mpsc::Sender<()>,
        wait: std::sync::mpsc::Receiver<()>,
    }
    impl PutBytes for Blocked {
        fn fill(&mut self, _: u64, pages: u32, row: bool, dst: &mut [u8]) -> Result<Checksums> {
            self.started.send(()).unwrap();
            self.wait.recv().unwrap();
            dst.fill(7);
            Ok(Checksums { pages: vec![0; pages as usize], row: row.then_some(0) })
        }
    }
    let mut n = net_rig();
    let mut request = host_put(None);
    request.source = PutSource::Host(Box::new(Blocked { started, wait }));
    n.rig.command(Command::Put { id: 8, request });
    n.rig.send(put_accept(8, record_len(3), Some(channel_ref(3))));
    let slot = n.staging.take_now().unwrap();
    n.report(vec![Event::Granted {
        id: Id::Put(8), shape: shape(0, 0, 0, 3, true), slot,
    }]);
    start.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
    let out = n.report(vec![Event::Closed { id: Id::Put(8) }]);
    assert!(pick!(out, FenceAck).is_empty());
    assert_eq!(n.staging.free_slots(), 3);
    let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
    rt.block_on(async {
        tokio::time::timeout(std::time::Duration::from_secs(2), n.rig.shared.wake.notified()).await.unwrap();
    });
    release.send(()).unwrap();
    rt.block_on(async {
        tokio::time::timeout(std::time::Duration::from_secs(2), n.rig.shared.wake.notified()).await.unwrap();
    });
    n.rig.tick(std::time::Duration::ZERO);
    assert_eq!(n.staging.free_slots(), 4);
}
