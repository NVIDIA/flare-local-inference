/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Framed byte links shared by the rank record relay and pool protocol.
use crate::{
    Error, InboundTicket, JobId, PageId, PrefixKey, PublishedCheckpoint, Result, SharedCopier,
    StateSlotId,
    disk::DiskThread,
    disk_store::{DiskTransport, FetchSegment, Staged, TransportEvent},
    geometry::invalid,
    record::{self, HEADER_BYTES, Header, Scanned},
};
use std::{
    collections::HashMap,
    io::{ErrorKind, Read, Write},
    net::{Shutdown, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
pub const IDLE: Duration = Duration::from_millis(1);
pub const BUSY: Duration = Duration::from_micros(100);
pub const BACKLOG: usize = 64;
const FRAME_LIMIT: usize = 64 << 20;
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);
// ---------------------------------------------------------------------------------------------
// Framing

pub struct Writer(Vec<u8>);

impl Writer {
    pub fn new(tag: u8) -> Self {
        Self(vec![tag, 0, 0, 0, 0])
    }

    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.0.push(value);
        self
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.extend(value);
        self
    }

    pub fn blob(&mut self, value: &[u8]) -> &mut Self {
        self.u32(value.len() as u32).bytes(value)
    }

    pub fn text(&mut self, value: &str) -> &mut Self {
        self.u32(value.len() as u32).bytes(value.as_bytes())
    }

    pub fn key(&mut self, key: &PrefixKey) -> &mut Self {
        self.u64(key.consumed_tokens).bytes(&key.digest)
    }

    pub fn header(&mut self, header: &Header) -> Result<&mut Self> {
        Ok(self.bytes(&header.encode()?))
    }

    pub fn finish(mut self) -> Vec<u8> {
        let length = (self.0.len() - 5) as u32;
        self.0[1..5].copy_from_slice(&length.to_le_bytes());
        self.0
    }
}

pub struct Reader<'a>(pub &'a [u8]);

impl<'a> Reader<'a> {
    pub fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.0.len() < count {
            return Err(invalid("truncated prefix record message"));
        }
        let (head, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(head)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn boolean(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid rank message boolean")),
        }
    }

    pub fn blob(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    pub fn text(&mut self) -> Result<&'a str> {
        let length = self.u32()? as usize;
        std::str::from_utf8(self.take(length)?)
            .map_err(|_| invalid("prefix record message text is not UTF-8"))
    }

    pub fn key(&mut self) -> Result<PrefixKey> {
        Ok(PrefixKey {
            consumed_tokens: self.u64()?,
            digest: self.take(16)?.try_into().unwrap(),
        })
    }

    pub fn header(&mut self) -> Result<Header> {
        Header::decode(self.take(HEADER_BYTES)?)
    }

    /// A count of items that each occupy at least `item` bytes of what remains.
    pub fn count(&mut self, item: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        if count.saturating_mul(item) > self.0.len() {
            return Err(invalid("prefix record message count exceeds its payload"));
        }
        Ok(count)
    }

    pub fn end(&self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid("prefix record message has trailing bytes"))
        }
    }
}

/// Writes queued frames on one connection and records how many are still queued.
#[derive(Default)]
struct Outgoing {
    state: Mutex<OutgoingState>,
    changed: Condvar,
}

#[derive(Default)]
struct OutgoingState {
    queued: usize,
    failure: Option<String>,
}

impl Outgoing {
    fn lock(&self) -> std::sync::MutexGuard<'_, OutgoingState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// One end of a record connection.
///
/// Reads never block. Writes go through a writer thread, so the model worker
/// thread only enqueues a frame; a peer that stops reading fills the queue and
/// fails the job instead of stalling the worker.
pub struct Link {
    pub(crate) stream: TcpStream,
    outgoing: Option<SyncSender<Vec<u8>>>,
    writer: Option<JoinHandle<()>>,
    queued: Arc<Outgoing>,
    pub(crate) received: Vec<u8>,
    /// Set once the connection failed or the other end closed it.
    broken: Option<Error>,
}

impl Link {
    pub fn new(stream: TcpStream) -> Result<Self> {
        stream
            .set_nodelay(true)
            .and_then(|()| stream.set_nonblocking(true))
            .map_err(|error| record::io_failure("configure a prefix record connection", error))?;
        let writer_stream = stream
            .try_clone()
            .map_err(|error| record::io_failure("duplicate a prefix record connection", error))?;
        let (sender, jobs) = sync_channel(BACKLOG);
        let queued = Arc::new(Outgoing::default());
        let progress = queued.clone();
        let writer = std::thread::Builder::new()
            .name("flare-offload-link".into())
            .spawn(move || write_frames(writer_stream, jobs, progress))
            .map_err(|error| invalid(format!("cannot start a prefix record writer: {error}")))?;
        Ok(Self {
            stream,
            outgoing: Some(sender),
            writer: Some(writer),
            queued,
            received: Vec::new(),
            broken: None,
        })
    }

    pub(crate) fn failed(&self) -> bool {
        self.broken.is_some() || self.queued.lock().failure.is_some()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn shutdown_handle(&self) -> Result<TcpStream> {
        self.stream
            .try_clone()
            .map_err(|e| record::io_failure("clone a rank connection", e))
    }

    pub fn fail(&mut self, error: Error) -> Error {
        let error = self.broken.get_or_insert(error).clone();
        // The peer must learn that this rank stopped driving it, so its own
        // transfers drain instead of running on with no one to report to.
        let _ = self.stream.shutdown(Shutdown::Both);
        error
    }

    /// Queue a frame without waiting. Fails when the peer is not keeping up.
    pub fn enqueue(&mut self, frame: &[u8]) -> Result<()> {
        if let Some(error) = &self.broken {
            return Err(error.clone());
        }
        if frame.len() > FRAME_LIMIT {
            return Err(self.fail(invalid("prefix record frame exceeds its limit")));
        }
        let mut state = self.queued.lock();
        if let Some(failure) = &state.failure {
            let error = invalid(failure.clone());
            drop(state);
            return Err(self.fail(error));
        }
        state.queued += 1;
        drop(state);
        let sender = self
            .outgoing
            .as_ref()
            .ok_or_else(|| invalid("prefix record connection is closed"))?;
        match sender.try_send(frame.to_vec()) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                self.queued.lock().queued -= 1;
                Err(self.fail(invalid(
                    "prefix record connection backlog is full; the peer is not keeping up",
                )))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.queued.lock().queued -= 1;
                Err(self.fail(invalid("prefix record connection closed")))
            }
        }
    }

    /// Queue a frame and wait until the writer has sent every queued frame.
    pub fn send(&mut self, frame: &[u8]) -> Result<()> {
        self.enqueue(frame)?;
        self.drain_outgoing(REPLY_TIMEOUT)
    }

    /// Wait until nothing is queued, the writer failed, or `timeout` passes.
    pub fn drain_outgoing(&mut self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let mut state = self.queued.lock();
        loop {
            if let Some(failure) = &state.failure {
                let error = invalid(failure.clone());
                drop(state);
                return Err(self.fail(error));
            }
            if state.queued == 0 {
                return Ok(());
            }
            let now = Instant::now();
            if now >= deadline {
                drop(state);
                return Err(self.fail(invalid("prefix record connection stalled")));
            }
            state = self
                .queued
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Both);
        self.outgoing.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

/// Write queued frames in order. A failure is recorded for the owner and the
/// connection is shut down so the peer stops.
fn write_frames(mut stream: TcpStream, jobs: Receiver<Vec<u8>>, progress: Arc<Outgoing>) {
    for frame in jobs {
        let deadline = Instant::now() + REPLY_TIMEOUT;
        let mut sent = 0;
        let mut failure = None;
        while sent < frame.len() {
            match stream.write(&frame[sent..]) {
                Ok(0) => {
                    failure = Some("prefix record connection closed".to_owned());
                    break;
                }
                Ok(count) => sent += count,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        failure = Some("prefix record connection stalled".to_owned());
                        break;
                    }
                    std::thread::sleep(BUSY);
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => {
                    failure = Some(format!("send on a prefix record connection: {error}"));
                    break;
                }
            }
        }
        let mut state = progress.lock();
        state.queued -= 1;
        if failure.is_some() {
            state.failure = failure;
        }
        let failed = state.failure.is_some();
        drop(state);
        progress.changed.notify_all();
        if failed {
            let _ = stream.shutdown(Shutdown::Both);
            return;
        }
    }
}

impl Link {
    /// The next complete frame, if one has arrived.
    pub fn receive(&mut self) -> Result<Option<(u8, Vec<u8>)>> {
        if let Some(error) = &self.broken {
            return Err(error.clone());
        }
        receive_frame(&mut self.stream, &mut self.received).map_err(|error| self.fail(error))
    }

    /// Wait up to `timeout` for a frame tagged `tag`.
    pub fn expect(&mut self, tag: u8, timeout: Duration) -> Result<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some((received, payload)) = self.receive()? {
                if received != tag {
                    return Err(self.fail(invalid("unexpected prefix record message")));
                }
                return Ok(payload);
            }
            if Instant::now() >= deadline {
                return Err(self.fail(invalid("prefix record peer did not answer in time")));
            }
            std::thread::sleep(IDLE);
        }
    }
}

pub(crate) fn receive_frame(
    stream: &mut TcpStream,
    received: &mut Vec<u8>,
) -> Result<Option<(u8, Vec<u8>)>> {
    let mut buffer = [0u8; 64 << 10];
    loop {
        if received.len() >= 5 {
            let length = u32::from_le_bytes(received[1..5].try_into().unwrap()) as usize;
            if length > FRAME_LIMIT {
                return Err(invalid("prefix record frame exceeds its limit"));
            }
            if received.len() >= 5 + length {
                let tag = received[0];
                let payload = received[5..5 + length].to_vec();
                received.drain(..5 + length);
                return Ok(Some((tag, payload)));
            }
        }
        match stream.read(&mut buffer) {
            Ok(0) => return Err(invalid("prefix record connection closed")),
            Ok(count) => received.extend_from_slice(&buffer[..count]),
            Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(record::io_failure(
                    "receive on a prefix record connection",
                    error,
                ));
            }
        }
    }
}

use crate::CLOSE_TIMEOUT;
use blake2::{Blake2b, Digest, digest::consts::U16};
/// Jobs ids at or above this belong to [`Sharded`] itself, not to the disk store.
const INTERNAL_JOBS: u64 = 1 << 63;
/// A peer answers its record handshake within this time.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// A job every rank accepted completes within this time, or its link is unusable.
const JOB_TIMEOUT: Duration = Duration::from_secs(120);

pub const HELLO: u8 = 1;
pub const SUMMARY: u8 = 2;
pub const AGREED: u8 = 3;
const STORE: u8 = 4;
const FETCH: u8 = 5;
const TRUNCATE: u8 = 6;
const UNLINK: u8 = 7;
const CLOSE: u8 = 8;
const EVENT: u8 = 9;
const CLOSED: u8 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Stored,
    Fetched,
    Truncated,
    Unlinked,
    StoreInvalidated,
}

impl Kind {
    fn code(self) -> u8 {
        self as u8
    }

    fn decode(code: u8) -> Result<Self> {
        Ok(match code {
            0 => Self::Stored,
            1 => Self::Fetched,
            2 => Self::Truncated,
            3 => Self::Unlinked,
            4 => Self::StoreInvalidated,
            _ => return Err(invalid("unknown prefix record event kind")),
        })
    }

    fn of(event: &TransportEvent) -> (Self, JobId, Result<u64>) {
        match event {
            TransportEvent::Stored { id, result } => (Self::Stored, *id, copy(result)),
            TransportEvent::StoreInvalidated { id, error } => {
                (Self::StoreInvalidated, *id, Err(error.clone()))
            }
            TransportEvent::Fetched { id, result } => (Self::Fetched, *id, copy(result)),
            TransportEvent::Truncated { id, result } => {
                (Self::Truncated, *id, copy(result).map(|()| 0))
            }
            TransportEvent::Unlinked { id, result } => {
                (Self::Unlinked, *id, copy(result).map(|()| 0))
            }
        }
    }

    fn event(self, id: JobId, result: Result<u64>) -> TransportEvent {
        match self {
            Self::Stored => TransportEvent::Stored { id, result },
            Self::StoreInvalidated => TransportEvent::StoreInvalidated {
                id,
                error: result
                    .err()
                    .unwrap_or_else(|| invalid("invalidated store lacks a failure")),
            },
            Self::Fetched => TransportEvent::Fetched { id, result },
            Self::Truncated => TransportEvent::Truncated {
                id,
                result: result.map(drop),
            },
            Self::Unlinked => TransportEvent::Unlinked {
                id,
                result: result.map(drop),
            },
        }
    }
}

fn copy<T: Copy>(result: &Result<T>) -> Result<T> {
    match result {
        Ok(value) => Ok(*value),
        Err(error) => Err(error.clone()),
    }
}

/// A record file name as rank zero sends it: a plain record or temporary name, never a path.
fn file(directory: &Path, name: &str) -> Result<PathBuf> {
    let plain = name
        .strip_prefix('.')
        .and_then(|n| n.strip_suffix(".tmp"))
        .unwrap_or(name);
    if record::parse_file_name(plain).is_none() {
        return Err(invalid(
            "prefix record message names an invalid record file",
        ));
    }
    Ok(directory.join(name))
}

fn name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("prefix record path has no file name"))
}

// ---------------------------------------------------------------------------------------------
// Rank zero's transport

/// One job's completion on every rank: index zero is rank zero.
struct Pending {
    kind: Kind,
    results: Vec<Option<Result<u64>>>,
    /// A store's record file, removed from the ranks that wrote it when another rank failed.
    path: Option<PathBuf>,
    /// The store a cleanup unlink belongs to, with the failure it reports.
    cleanup: Option<(JobId, Error)>,
    started: Instant,
}

/// Rank zero's transport for a model whose ranks run as separate processes: rank zero's own
/// staged transport plus one connection per peer rank, ranks 1 onward.
pub struct Sharded {
    local: Staged,
    peers: Vec<Link>,
    pending: HashMap<(Kind, JobId), Pending>,
    events: Vec<TransportEvent>,
    next_internal: u64,
    closed: bool,
    /// A job this transport accepted can no longer be accounted for. No event
    /// follows: the ids and pins it holds must not be reused or released.
    poison: Option<Error>,
}

impl Sharded {
    pub fn new(local: Staged, peers: Vec<Link>) -> Result<Self> {
        Ok(Self {
            local,
            peers,
            pending: HashMap::new(),
            events: Vec::new(),
            next_internal: INTERNAL_JOBS,
            closed: false,
            poison: None,
        })
    }

    fn ranks(&self) -> usize {
        self.peers.len() + 1
    }

    /// Begin tracking a job rank zero accepted, and queue it for every peer.
    ///
    /// Queueing never waits for a peer: a peer that is not keeping up fails the
    /// job, and a failed send stops the transport because that peer's state is
    /// no longer known.
    fn begin(&mut self, kind: Kind, id: JobId, path: Option<PathBuf>, frame: &[u8]) {
        let mut results = vec![None; self.ranks()];
        let mut broken = None;
        for (index, peer) in self.peers.iter_mut().enumerate() {
            if let Err(error) = peer.enqueue(frame) {
                results[index + 1] = Some(Err(error.clone()));
                broken.get_or_insert((index + 1, error));
            }
        }
        if let Some((rank, error)) = broken {
            self.abandon(rank, error);
        }
        self.pending.insert(
            (kind, id),
            Pending {
                kind,
                results,
                path,
                cleanup: None,
                started: Instant::now(),
            },
        );
    }

    fn record(&mut self, kind: Kind, id: JobId, rank: usize, result: Result<u64>) {
        if self.poison.is_some() {
            return;
        }
        let Some(pending) = self.pending.get_mut(&(kind, id)) else {
            return;
        };
        if pending.results[rank].is_none() {
            pending.results[rank] = Some(result);
        }
        if pending.results.iter().all(Option::is_some) {
            let pending = self.pending.remove(&(kind, id)).unwrap();
            self.finish(id, pending);
        }
    }

    fn finish(&mut self, id: JobId, pending: Pending) {
        let failure = pending
            .results
            .iter()
            .enumerate()
            .find_map(|(rank, result)| match result {
                Some(Err(error)) => Some(Error::caused_by(
                    format!("rank {rank}: {error}"),
                    error.clone(),
                )),
                _ => None,
            });
        if let Some((store, error)) = pending.cleanup {
            // Preserve cleanup failures alongside the write failure.
            let leftover =
                pending
                    .results
                    .iter()
                    .enumerate()
                    .find_map(|(rank, result)| match result {
                        Some(Err(failure)) => Some(format!("rank {rank}: {failure}")),
                        _ => None,
                    });
            let error = match leftover {
                Some(failure) => invalid(format!("{error}; cleanup unlink failed on {failure}")),
                None => error,
            };
            self.events
                .push(TransportEvent::StoreInvalidated { id: store, error });
            return;
        }
        let Some(error) = failure else {
            let result = pending.results[0].as_ref().map_or(Ok(0), copy);
            self.events.push(pending.kind.event(id, result));
            return;
        };
        let wrote: Vec<usize> = pending
            .results
            .iter()
            .enumerate()
            .filter(|(_, result)| matches!(result, Some(Ok(_))))
            .map(|(rank, _)| rank)
            .collect();
        match (pending.kind, pending.path) {
            (Kind::Stored, Some(path)) if !wrote.is_empty() => self.remove(&wrote, path, id, error),
            (kind, _) => self.events.push(kind.event(id, Err(error))),
        }
    }

    /// Unlink a failed store's file on the ranks that wrote it, then report the failure.
    fn remove(&mut self, ranks: &[usize], path: PathBuf, store: JobId, error: Error) {
        let frame = match name(&path) {
            Ok(name) => {
                let mut writer = Writer::new(UNLINK);
                writer.u64(self.next_internal).text(name);
                writer.finish()
            }
            Err(_) => {
                self.events
                    .push(TransportEvent::StoreInvalidated { id: store, error });
                return;
            }
        };
        let id = JobId(self.next_internal);
        self.next_internal += 1;
        let mut results: Vec<Option<Result<u64>>> = vec![Some(Ok(0)); self.ranks()];
        for &rank in ranks {
            let sent = if rank == 0 {
                self.local.unlink(id, path.clone())
            } else {
                self.peers[rank - 1].enqueue(&frame)
            };
            results[rank] = sent.err().map(Err);
        }
        let pending = Pending {
            kind: Kind::Unlinked,
            results,
            path: None,
            cleanup: Some((store, error)),
            started: Instant::now(),
        };
        if pending.results.iter().all(Option::is_some) {
            self.finish(id, pending);
        } else {
            self.pending.insert((Kind::Unlinked, id), pending);
        }
    }

    fn absorb_local(&mut self) {
        let mut events = Vec::new();
        let _ = self.local.poll(&mut events);
        for event in events {
            let (kind, id, result) = Kind::of(&event);
            self.record(kind, id, 0, result);
        }
    }

    fn absorb_peers(&mut self) {
        for index in 0..self.peers.len() {
            let rank = index + 1;
            loop {
                let received = self.peers[index].receive();
                match received {
                    Ok(None) => break,
                    Ok(Some((EVENT, payload))) => match decode_event(&payload) {
                        Ok((kind, id, result)) => self.record(kind, id, rank, result),
                        Err(error) => {
                            self.peers[index].fail(error);
                        }
                    },
                    Ok(Some(_)) => {
                        self.peers[index].fail(invalid("unexpected prefix record message"));
                    }
                    Err(error) => {
                        self.abandon(rank, error);
                        break;
                    }
                }
            }
        }
    }

    /// Stop reporting completions when a rank's connection breaks.
    ///
    /// The peer may still be copying into the ids of the jobs it already
    /// received. Reporting those jobs as failed would return their pages and
    /// state slot to the scheduler's free lists while the peer's transfer
    /// thread is still writing them, so the transport stops instead and the
    /// worker stops with it.
    fn abandon(&mut self, rank: usize, error: Error) {
        if self.poison.is_none() {
            self.poison = Some(invalid(format!(
                "rank {rank}'s prefix record connection failed while jobs were outstanding: {error}"
            )));
        }
    }

    fn accept(&self) -> Result<()> {
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        if self.closed {
            return Err(invalid("the sharded record transport is closed"));
        }
        Ok(())
    }

    /// Stop every link once a job outlives [`JOB_TIMEOUT`].
    fn expire(&mut self) {
        let Some((kind, id)) = self
            .pending
            .iter()
            .find(|(_, pending)| pending.started.elapsed() >= JOB_TIMEOUT)
            .map(|(key, _)| *key)
        else {
            return;
        };
        if self.poison.is_none() {
            self.poison = Some(invalid(format!(
                "a prefix record job ({kind:?} {}) did not complete within {JOB_TIMEOUT:?}",
                id.0
            )));
        }
    }
}

fn encode_event(kind: Kind, id: JobId, result: &Result<u64>) -> Vec<u8> {
    let mut writer = Writer::new(EVENT);
    writer.u8(kind.code()).u64(id.0);
    match result {
        Ok(value) => writer.u8(1).u64(*value),
        Err(error) => writer
            .u8(if error.no_space() { 2 } else { 0 })
            .text(&error.to_string()),
    };
    writer.finish()
}

fn decode_event(payload: &[u8]) -> Result<(Kind, JobId, Result<u64>)> {
    let mut reader = Reader(payload);
    let kind = Kind::decode(reader.u8()?)?;
    let id = JobId(reader.u64()?);
    let result = match reader.u8()? {
        1 => Ok(reader.u64()?),
        0 => Err(invalid(reader.text()?.to_owned())),
        2 => Err(Error::caused_by(
            reader.text()?.to_owned(),
            std::io::Error::from(ErrorKind::StorageFull),
        )),
        _ => return Err(invalid("invalid prefix record event result")),
    };
    reader.end()?;
    Ok((kind, id, result))
}

impl DiskTransport for Sharded {
    fn store(
        &mut self,
        id: JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
    ) -> Result<()> {
        self.accept()?;
        let tail = header.tail_pages as usize;
        let chain = checkpoint.chain_pages.len();
        if tail > chain {
            return Err(invalid(format!(
                "record {} holds {tail} tail pages but the published chain has only {chain}",
                record::file_name(&header.key)
            )));
        }
        let pages = &checkpoint.chain_pages[chain - tail..];
        let mut writer = Writer::new(STORE);
        writer
            .u64(id.0)
            .header(&header)?
            .text(name(&temporary)?)
            .text(name(&path)?)
            .u32(checkpoint.slot.0)
            .u32(pages.len() as u32);
        for page in pages {
            writer.u32(page.0);
        }
        let frame = writer.finish();
        self.local.store_pages(
            id,
            header,
            temporary,
            path.clone(),
            pages.to_vec(),
            checkpoint.slot,
        )?;
        self.begin(Kind::Stored, id, Some(path), &frame);
        Ok(())
    }

    fn fetch(&mut self, id: JobId, plan: Vec<FetchSegment>, ticket: &InboundTicket) -> Result<()> {
        self.accept()?;
        let mut writer = Writer::new(FETCH);
        writer
            .u64(id.0)
            .u32(ticket.slot.0)
            .u32(ticket.pages.len() as u32);
        for page in &ticket.pages {
            writer.u32(page.0);
        }
        writer.u32(plan.len() as u32);
        for segment in &plan {
            writer
                .text(name(&segment.path)?)
                .header(&segment.header)?
                .u32(segment.first_page)
                .u32(segment.pages)
                .u8(segment.row as u8);
        }
        let frame = writer.finish();
        self.local.fetch(id, plan, ticket)?;
        self.begin(Kind::Fetched, id, None, &frame);
        Ok(())
    }

    fn truncate(&mut self, id: JobId, path: PathBuf, length: u64) -> Result<()> {
        self.accept()?;
        let mut writer = Writer::new(TRUNCATE);
        writer.u64(id.0).text(name(&path)?).u64(length);
        let frame = writer.finish();
        self.local.truncate(id, path, length)?;
        self.begin(Kind::Truncated, id, None, &frame);
        Ok(())
    }

    fn unlink(&mut self, id: JobId, path: PathBuf) -> Result<()> {
        self.accept()?;
        let mut writer = Writer::new(UNLINK);
        writer.u64(id.0).text(name(&path)?);
        let frame = writer.finish();
        self.local.unlink(id, path)?;
        self.begin(Kind::Unlinked, id, None, &frame);
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<TransportEvent>) -> Result<()> {
        if let Some(error) = &self.poison {
            events.clear();
            self.events.clear();
            return Err(error.clone());
        }
        if !self.closed {
            self.absorb_local();
            self.absorb_peers();
            self.expire();
        }
        if let Some(error) = &self.poison {
            // Nothing completed here can be trusted to describe the peer's state.
            events.clear();
            self.events.clear();
            return Err(error.clone());
        }
        events.append(&mut self.events);
        Ok(())
    }

    /// Close every peer's transport and rank zero's, collecting the completions each reaches.
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let close = Writer::new(CLOSE).finish();
        for peer in &mut self.peers {
            let _ = peer.send(&close);
        }
        let local = self.local.close();
        self.absorb_local();
        let deadline = Instant::now() + CLOSE_TIMEOUT + REPLY_TIMEOUT;
        let mut failure = local.err();
        for index in 0..self.peers.len() {
            let rank = index + 1;
            let result = loop {
                match self.peers[index].receive() {
                    Ok(Some((EVENT, payload))) => match decode_event(&payload) {
                        Ok((kind, id, result)) => self.record(kind, id, rank, result),
                        Err(error) => break Err(self.peers[index].fail(error)),
                    },
                    Ok(Some((CLOSED, payload))) => break decode_closed(&payload),
                    Ok(Some(_)) => {
                        break Err(
                            self.peers[index].fail(invalid("unexpected prefix record message"))
                        );
                    }
                    Ok(None) if Instant::now() >= deadline => {
                        break Err(self.peers[index]
                            .fail(invalid("prefix record peer did not close in time")));
                    }
                    Ok(None) => std::thread::sleep(IDLE),
                    Err(error) => break Err(error),
                }
            };
            if let Err(error) = result {
                self.abandon(rank, error.clone());
                failure.get_or_insert(invalid(format!("rank {rank}: {error}")));
            }
        }
        self.closed = true;
        // A poisoned transport never confirms its drain: the caller retains it,
        // and with it the registered staging its transfers may still use.
        if let Some(error) = &self.poison {
            return Err(error.clone());
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn decode_closed(payload: &[u8]) -> Result<()> {
    let mut reader = Reader(payload);
    let result = match reader.u8()? {
        1 => Ok(()),
        0 => Err(invalid(reader.text()?.to_owned())),
        _ => return Err(invalid("invalid prefix record close result")),
    };
    reader.end()?;
    result
}

/// Rank zero's disk transport over its prepared staging resources and its peer connections.
pub fn transport(
    staging: Option<crate::slab::SlabPool>,
    transfers: SharedCopier,
    peers: Vec<Link>,
) -> Result<Box<dyn DiskTransport>> {
    let local = Staged::new(staging, transfers, DiskThread::start("flare-offload-disk")?);
    Ok(Box::new(Sharded::new(local, peers)?))
}

pub struct Peer {
    pub staged: Staged,
    pub directory: PathBuf,
    pub fingerprint: [u8; 16],
    /// This rank's own cache geometry: every header rank zero sends must
    /// describe the same plane and row sizes before any I/O starts.
    pub geometry: Vec<crate::CacheGeometry>,
    pub link: Link,
    /// Retains the record directory through every job.
    pub _lock: record::DirectoryLock,
}

impl Peer {
    /// Rank zero's header with this rank's own layout fingerprint, after
    /// checking that its sizes are this rank's sizes.
    fn own(&self, mut header: Header) -> Result<Header> {
        if !header.matches(&self.geometry) {
            return Err(invalid(format!(
                "record {} was described with a layout this rank does not have",
                record::file_name(&header.key)
            )));
        }
        header.fingerprint = self.fingerprint;
        Ok(header)
    }

    fn dispatch(&mut self, tag: u8, payload: &[u8]) -> Result<Option<(Kind, JobId, Error)>> {
        let mut reader = Reader(payload);
        let id = JobId(reader.u64()?);
        if tag == STORE {
            let header = self.own(reader.header()?)?;
            let temporary = file(&self.directory, reader.text()?)?;
            let path = file(&self.directory, reader.text()?)?;
            let slot = StateSlotId(reader.u32()?);
            let count = reader.count(4)?;
            let pages = (0..count)
                .map(|_| reader.u32().map(PageId))
                .collect::<Result<Vec<_>>>()?;
            reader.end()?;
            let result = self
                .staged
                .store_pages(id, header, temporary, path, pages, slot);
            return Ok(result.err().map(|error| (Kind::Stored, id, error)));
        }
        if tag == FETCH {
            let slot = StateSlotId(reader.u32()?);
            let count = reader.count(4)?;
            let pages = (0..count)
                .map(|_| reader.u32().map(PageId))
                .collect::<Result<Vec<_>>>()?;
            let segments = reader.count(HEADER_BYTES)?;
            let mut plan = Vec::with_capacity(segments);
            for _ in 0..segments {
                let path = file(&self.directory, reader.text()?)?;
                let header = self.own(reader.header()?)?;
                plan.push(FetchSegment {
                    path,
                    header,
                    first_page: reader.u32()?,
                    pages: reader.u32()?,
                    row: reader.u8()? != 0,
                });
            }
            reader.end()?;
            // The key names no record on this side; the transport uses the ids and the slot.
            let ticket = InboundTicket {
                key: plan
                    .last()
                    .map(|segment| segment.header.key)
                    .ok_or_else(|| invalid("a prefix record fetch names no record"))?,
                pages,
                slot,
            };
            let result = self.staged.fetch(id, plan, &ticket);
            return Ok(result.err().map(|error| (Kind::Fetched, id, error)));
        }
        if tag == TRUNCATE {
            let path = file(&self.directory, reader.text()?)?;
            let length = reader.u64()?;
            reader.end()?;
            let result = self.staged.truncate(id, path, length);
            return Ok(result.err().map(|error| (Kind::Truncated, id, error)));
        }
        if tag == UNLINK {
            let path = file(&self.directory, reader.text()?)?;
            reader.end()?;
            let result = self.staged.unlink(id, path);
            return Ok(result.err().map(|error| (Kind::Unlinked, id, error)));
        }
        Err(invalid("unexpected prefix record message"))
    }

    fn forward(&mut self) -> Result<()> {
        let mut events = Vec::new();
        self.staged.poll(&mut events)?;
        for event in &events {
            let (kind, id, result) = Kind::of(event);
            self.link.send(&encode_event(kind, id, &result))?;
        }
        Ok(())
    }

    /// Drain the transport and report the outcome to rank zero.
    ///
    /// `Err` means the drain is unconfirmed: the caller must retain this peer,
    /// and with it the registered staging its copy and I/O threads may use.
    fn shutdown(&mut self) -> Result<()> {
        let closed = self.staged.close();
        let forwarded = self.forward();
        let mut writer = Writer::new(CLOSED);
        match &closed {
            Ok(()) => writer.u8(1),
            Err(error) => writer.u8(0).text(&error.to_string()),
        };
        let sent = self.link.send(&writer.finish());
        closed.and(forwarded).and(sent)
    }

    /// Serve rank zero until it closes the connection or the link fails.
    ///
    /// Never closes the transport: `serve` owns that decision, so a transport
    /// whose close did not confirm its drain is retained rather than dropped
    /// underneath its copy and I/O threads.
    fn run(&mut self) -> Result<()> {
        loop {
            let mut progressed = false;
            loop {
                match self.link.receive() {
                    Ok(None) => break,
                    Ok(Some((CLOSE, _))) => return Ok(()),
                    Ok(Some((tag, payload))) => {
                        progressed = true;
                        match self.dispatch(tag, &payload) {
                            Ok(Some((kind, id, error))) => {
                                self.link.send(&encode_event(kind, id, &Err(error)))?
                            }
                            Ok(None) => {}
                            Err(error) => return Err(error),
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
            self.forward()?;
            let (outbound, inbound) = self.staged.in_flight();
            std::thread::sleep(if progressed || outbound + inbound > 0 {
                BUSY
            } else {
                IDLE
            });
        }
    }
}

pub fn serve(mut peer: Peer) -> Result<()> {
    let outcome = peer.run();
    // Rank zero's CLOSE and a failed link both end the service, and both need
    // the transport drained before its staging can be released.
    let shutdown = peer.shutdown();
    if shutdown.is_err() {
        // The drain is unconfirmed: keep the transport, its transfer threads,
        // its disk thread, and its registered staging alive together.
        std::mem::forget(peer);
    }
    match outcome {
        Err(error) => Err(error),
        Ok(()) => shutdown,
    }
}

// ---------------------------------------------------------------------------------------------
// Joining the ranks before serving

/// What a rank's scan holds, as the agreement compares it.
pub fn summary(scanned: &[Scanned]) -> Result<Vec<u8>> {
    let mut writer = Writer::new(SUMMARY);
    writer.u32(scanned.len() as u32);
    for record in scanned {
        writer
            .key(&record.header.key)
            .bytes(&shape(&record.header)?)
            .u8(record.row as u8);
    }
    Ok(writer.finish())
}

/// Digest of a header without the fields that differ between ranks' files of one record.
fn shape(header: &Header) -> Result<[u8; 16]> {
    let mut header = header.clone();
    header.fingerprint = [0; 16];
    header.row_checksum = 0;
    let mut hash = Blake2b::<U16>::new();
    hash.update(header.encode()?);
    Ok(hash.finalize().into())
}

pub fn decode_summary(payload: &[u8]) -> Result<HashMap<PrefixKey, ([u8; 16], bool)>> {
    let mut reader = Reader(payload);
    let count = reader.count(41)?;
    let mut records = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = reader.key()?;
        let shape: [u8; 16] = reader.take(16)?.try_into().unwrap();
        records.insert(key, (shape, reader.u8()? != 0));
    }
    reader.end()?;
    Ok(records)
}

/// The records every rank holds with equal headers, each with its row when every rank has it,
/// closed under parents: a record whose parent is not agreed is dropped too.
pub fn agree(
    scanned: &[Scanned],
    peers: &[HashMap<PrefixKey, ([u8; 16], bool)>],
) -> Result<HashMap<PrefixKey, bool>> {
    let mut agreed = HashMap::new();
    for record in scanned {
        let shape = shape(&record.header)?;
        let mut row = record.row;
        let everywhere = peers.iter().all(|peer| match peer.get(&record.header.key) {
            Some((peer_shape, peer_row)) if *peer_shape == shape => {
                row &= peer_row;
                true
            }
            _ => false,
        });
        if everywhere {
            agreed.insert(record.header.key, row);
        }
    }
    let parents: HashMap<PrefixKey, Option<PrefixKey>> = scanned
        .iter()
        .map(|record| (record.header.key, record.header.parent))
        .collect();
    loop {
        let orphans: Vec<PrefixKey> = agreed
            .keys()
            .filter(|key| parents[key].is_some_and(|parent| !agreed.contains_key(&parent)))
            .copied()
            .collect();
        if orphans.is_empty() {
            return Ok(agreed);
        }
        for key in orphans {
            agreed.remove(&key);
        }
    }
}

pub fn encode_agreed(agreed: &HashMap<PrefixKey, bool>) -> Vec<u8> {
    let mut writer = Writer::new(AGREED);
    writer.u32(agreed.len() as u32);
    let mut keys: Vec<_> = agreed.iter().collect();
    keys.sort_by_key(|(key, _)| (key.consumed_tokens, key.digest));
    for (key, row) in keys {
        writer.key(key).u8(*row as u8);
    }
    writer.finish()
}

pub fn decode_agreed(payload: &[u8]) -> Result<HashMap<PrefixKey, bool>> {
    let mut reader = Reader(payload);
    let count = reader.count(25)?;
    let mut agreed = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = reader.key()?;
        agreed.insert(key, reader.u8()? != 0);
    }
    reader.end()?;
    Ok(agreed)
}

/// A fresh token that only the ranks of this launch know.
///
/// Failing here stops startup: a predictable token would let any host on the
/// network claim a peer's record connection.
#[cfg(unix)]
pub fn nonce() -> Result<[u8; 16]> {
    let mut random = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut random))
        .map_err(|error| record::io_failure("read a prefix record token", error))?;
    Ok(random)
}

/// Windows has no `/dev/urandom`; the system-preferred CSPRNG fills the same token.
#[cfg(windows)]
pub fn nonce() -> Result<[u8; 16]> {
    use windows_sys::Win32::Security::Cryptography::{
        BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom,
    };
    let mut random = [0u8; 16];
    let status = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            random.as_mut_ptr(),
            random.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if status < 0 {
        return Err(record::io_failure(
            "read a prefix record token",
            std::io::Error::other(format!("BCryptGenRandom failed with NTSTATUS {status:#x}")),
        ));
    }
    Ok(random)
}

#[cfg(test)]
mod tests;
