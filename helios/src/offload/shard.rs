//! Disk records of a model whose ranks run as separate processes on separate hosts.
//!
//! Each host keeps the record files of its own rank in its own record directory. A file holds
//! one rank's planes and row in the ordinary record format, so every host's files have the same
//! names and lengths. Rank zero's [`DiskStore`](super::disk_store::DiskStore) indexes, sizes,
//! evicts, and plans every record; [`Sharded`] runs each of its transport jobs on rank zero's own
//! [`Staged`] transport and on every peer's, through a dedicated connection to each peer, and
//! reports the job only after every rank finished it. A fetch therefore completes only once every
//! rank's GPU holds its pages and row. A store that some rank failed removes the file from the
//! ranks that wrote it before the failure is reported, so no host keeps a record the index lacks.
//!
//! Before serving, the ranks agree on the records they scanned: rank zero keeps a record only
//! when every rank holds a file with the same header and keeps its row only when every rank
//! holds the row. Every rank then deletes or truncates its files to match, so the index rank zero
//! builds describes every host's directory.

use std::{
    collections::HashMap,
    io::{ErrorKind, Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use blake2::{Blake2b, Digest, digest::consts::U16};

use crate::{
    Error, Result,
    cache::{PageId, PrefixKey},
    execution::RankStream,
    scheduler::StateSlotId,
};

use super::{
    InboundTicket, JobId, Prepared, PublishedCheckpoint, Shared, Transport,
    disk::DiskThread,
    disk_store::{DiskTransport, FetchSegment, Staged, TransportEvent},
    geometry::invalid,
    record::{self, HEADER_BYTES, Header, Scanned},
    transfer::{self, Transfers},
};

/// Peers connect to rank zero's record listener within this time.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
/// The scan agreement and the final close are answered within this time.
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);
/// Jobs ids at or above this belong to [`Sharded`] itself, not to the disk store.
const INTERNAL_JOBS: u64 = 1 << 63;
/// A frame larger than this is a protocol error; a fetch of a long chain stays far below it.
const FRAME_LIMIT: usize = 64 << 20;
const IDLE: Duration = Duration::from_millis(1);
const BUSY: Duration = Duration::from_micros(100);
/// Frames a connection may queue before a job fails instead of waiting.
const BACKLOG: usize = 64;
/// A peer answers its record handshake within this time.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// A job every rank accepted completes within this time, or its link is unusable.
const JOB_TIMEOUT: Duration = Duration::from_secs(120);

const HELLO: u8 = 1;
const SUMMARY: u8 = 2;
const AGREED: u8 = 3;
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

// ---------------------------------------------------------------------------------------------
// Framing

struct Writer(Vec<u8>);

impl Writer {
    fn new(tag: u8) -> Self {
        Self(vec![tag, 0, 0, 0, 0])
    }

    fn u8(&mut self, value: u8) -> &mut Self {
        self.0.push(value);
        self
    }

    fn u32(&mut self, value: u32) -> &mut Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    fn u64(&mut self, value: u64) -> &mut Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.extend(value);
        self
    }

    fn text(&mut self, value: &str) -> &mut Self {
        self.u32(value.len() as u32).bytes(value.as_bytes())
    }

    fn key(&mut self, key: &PrefixKey) -> &mut Self {
        self.u64(key.consumed_tokens).bytes(&key.digest)
    }

    fn header(&mut self, header: &Header) -> Result<&mut Self> {
        Ok(self.bytes(&header.encode()?))
    }

    fn finish(mut self) -> Vec<u8> {
        let length = (self.0.len() - 5) as u32;
        self.0[1..5].copy_from_slice(&length.to_le_bytes());
        self.0
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.0.len() < count {
            return Err(invalid("truncated prefix record message"));
        }
        let (head, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn text(&mut self) -> Result<&'a str> {
        let length = self.u32()? as usize;
        std::str::from_utf8(self.take(length)?)
            .map_err(|_| invalid("prefix record message text is not UTF-8"))
    }

    fn key(&mut self) -> Result<PrefixKey> {
        Ok(PrefixKey {
            consumed_tokens: self.u64()?,
            digest: self.take(16)?.try_into().unwrap(),
        })
    }

    fn header(&mut self) -> Result<Header> {
        Header::decode(self.take(HEADER_BYTES)?)
    }

    /// A count of items that each occupy at least `item` bytes of what remains.
    fn count(&mut self, item: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        if count.saturating_mul(item) > self.0.len() {
            return Err(invalid("prefix record message count exceeds its payload"));
        }
        Ok(count)
    }

    fn end(&self) -> Result<()> {
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
pub(crate) struct Link {
    stream: TcpStream,
    outgoing: Option<SyncSender<Vec<u8>>>,
    writer: Option<JoinHandle<()>>,
    queued: Arc<Outgoing>,
    received: Vec<u8>,
    /// Set once the connection failed or the other end closed it.
    broken: Option<Error>,
}

impl Link {
    fn new(stream: TcpStream) -> Result<Self> {
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

    fn fail(&mut self, error: Error) -> Error {
        let error = self.broken.get_or_insert(error).clone();
        // The peer must learn that this rank stopped driving it, so its own
        // transfers drain instead of running on with no one to report to.
        let _ = self.stream.shutdown(Shutdown::Both);
        error
    }

    /// Queue a frame without waiting. Fails when the peer is not keeping up.
    fn enqueue(&mut self, frame: &[u8]) -> Result<()> {
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
    fn send(&mut self, frame: &[u8]) -> Result<()> {
        self.enqueue(frame)?;
        self.drain_outgoing(REPLY_TIMEOUT)
    }

    /// Wait until nothing is queued, the writer failed, or `timeout` passes.
    fn drain_outgoing(&mut self, timeout: Duration) -> Result<()> {
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
    fn receive(&mut self) -> Result<Option<(u8, Vec<u8>)>> {
        if let Some(error) = &self.broken {
            return Err(error.clone());
        }
        let mut buffer = [0u8; 64 << 10];
        loop {
            if self.received.len() >= 5 {
                let length = u32::from_le_bytes(self.received[1..5].try_into().unwrap()) as usize;
                if length > FRAME_LIMIT {
                    return Err(self.fail(invalid("prefix record frame exceeds its limit")));
                }
                if self.received.len() >= 5 + length {
                    let tag = self.received[0];
                    let payload = self.received[5..5 + length].to_vec();
                    self.received.drain(..5 + length);
                    return Ok(Some((tag, payload)));
                }
            }
            match self.stream.read(&mut buffer) {
                Ok(0) => return Err(self.fail(invalid("prefix record connection closed"))),
                Ok(count) => self.received.extend_from_slice(&buffer[..count]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => {
                    return Err(self.fail(record::io_failure(
                        "receive on a prefix record connection",
                        error,
                    )));
                }
            }
        }
    }

    /// Wait up to `timeout` for a frame tagged `tag`.
    fn expect(&mut self, tag: u8, timeout: Duration) -> Result<Vec<u8>> {
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
    fn new(local: Staged, peers: Vec<Link>) -> Result<Self> {
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
                Some(Err(error)) => Some(invalid(format!("rank {rank}: {error}"))),
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
        Err(error) => writer.u8(0).text(&error.to_string()),
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
        _ => return Err(invalid("invalid prefix record event result")),
    };
    reader.end()?;
    Ok((kind, id, result))
}

impl DiskTransport for Sharded {
    fn kind(&self) -> Transport {
        Transport::Staged
    }

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
        let deadline = Instant::now() + transfer::CLOSE_TIMEOUT + REPLY_TIMEOUT;
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
pub(super) fn transport(
    staging: Option<super::slab::SlabPool>,
    transfers: Arc<Mutex<Transfers>>,
    peers: Vec<Link>,
) -> Result<Box<dyn DiskTransport>> {
    let local = Staged::new(staging, transfers, DiskThread::start("flare-offload-disk")?);
    Ok(Box::new(Sharded::new(local, peers)?))
}

// ---------------------------------------------------------------------------------------------
// A peer's record service

/// A peer rank's record service: its staged transport, driven by rank zero over one connection.
pub struct Service {
    thread: Option<JoinHandle<Result<()>>>,
    finished: Receiver<Result<()>>,
    /// What this rank joined with, for the startup report.
    pub summary: String,
}

impl Service {
    /// Wait for rank zero to close the service, or for its connection to end.
    ///
    /// A rank zero that stays alive without closing cannot hold this rank's
    /// shutdown open: after [`JOB_TIMEOUT`] the service is reported as not
    /// drained, and its resources are retained by the caller.
    pub fn join(mut self) -> Result<()> {
        let result = match self.finished.recv_timeout(JOB_TIMEOUT) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(invalid(
                "rank zero did not close the prefix record service in time",
            )),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err(invalid("the prefix record service ended without a result"))
            }
        };
        if let Some(thread) = self.thread.take() {
            // The service reports its result before it returns.
            let _ = thread.join();
        }
        result
    }
}

struct Peer {
    staged: Staged,
    directory: PathBuf,
    fingerprint: [u8; 16],
    /// This rank's own cache geometry: every header rank zero sends must
    /// describe the same plane and row sizes before any I/O starts.
    geometry: Vec<super::geometry::CacheGeometry>,
    link: Link,
    /// Retains the record directory through every job.
    _lock: record::DirectoryLock,
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

fn serve(mut peer: Peer) -> Result<()> {
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
fn summary(scanned: &[Scanned]) -> Result<Vec<u8>> {
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

fn decode_summary(payload: &[u8]) -> Result<HashMap<PrefixKey, ([u8; 16], bool)>> {
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
fn agree(
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

fn encode_agreed(agreed: &HashMap<PrefixKey, bool>) -> Vec<u8> {
    let mut writer = Writer::new(AGREED);
    writer.u32(agreed.len() as u32);
    let mut keys: Vec<_> = agreed.iter().collect();
    keys.sort_by_key(|(key, _)| (key.consumed_tokens, key.digest));
    for (key, row) in keys {
        writer.key(key).u8(*row as u8);
    }
    writer.finish()
}

fn decode_agreed(payload: &[u8]) -> Result<HashMap<PrefixKey, bool>> {
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

/// Delete the files of records outside the agreement and truncate rows it does not keep.
fn retain(prepared: &mut Prepared, agreed: &HashMap<PrefixKey, bool>) -> Result<()> {
    let mut kept = Vec::with_capacity(agreed.len());
    for mut record in std::mem::take(&mut prepared.scanned) {
        match agreed.get(&record.header.key) {
            None => match std::fs::remove_file(&record.path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(record::io_failure(
                        format!("delete record file {}", record.path.display()),
                        error,
                    ));
                }
            },
            Some(&row) => {
                if record.row && !row {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&record.path)
                        .and_then(|file| file.set_len(record.header.row_segment))
                        .map_err(|error| {
                            record::io_failure(
                                format!("truncate record file {}", record.path.display()),
                                error,
                            )
                        })?;
                    record.row = false;
                    record.bytes = record.header.row_segment;
                }
                kept.push(record);
            }
        }
    }
    prepared
        .disk_budget
        .set_checkpoint(kept.iter().map(|record| record.bytes).sum());
    prepared.scanned = kept;
    Ok(())
}

/// Whether this rank serves a disk store its peers can mirror; a memory tier keeps records in
/// one process's memory, which other hosts cannot reach.
fn mirrored(prepared: &Prepared) -> Result<bool> {
    if prepared.memory_pool.is_some() {
        return Err(invalid(
            "a multi-host package keeps host prefix records on disk only; set --prefix-cache-memory 0",
        ));
    }
    Ok(prepared.directory_lock.is_some())
}

/// A fresh token that only the ranks of this launch know.
///
/// Failing here stops startup: a predictable token would let any host on the
/// network claim a peer's record connection.
fn nonce() -> Result<[u8; 16]> {
    let mut random = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut random))
        .map_err(|error| record::io_failure("read a prefix record token", error))?;
    Ok(random)
}

/// Rank zero, after its peers joined and before serving: agree with every peer on whether a disk
/// store exists, connect a record connection to each, and agree on the scanned records. Returns
/// a summary for the startup report when the ranks share a store.
pub fn join_rank_zero(store: Option<&Shared>, peers: &mut [RankStream]) -> Result<Option<String>> {
    use crate::execution::rendezvous;

    let mut prepared = store.map(Shared::lock);
    let prepared = prepared.as_mut().and_then(|guard| guard.as_mut());
    let enabled = match &prepared {
        Some(prepared) => mirrored(prepared)?,
        None => false,
    };
    // Each peer's own view of this rank: the local address of the rendezvous
    // connection that peer opened, so a multi-homed rank zero is reachable
    // from every peer on the interface that peer already uses.
    let addresses = peers
        .iter()
        .map(|peer| match peer {
            RankStream::Tcp(stream) => {
                let remote = stream.peer_addr().map_err(|error| {
                    record::io_failure("inspect a rendezvous connection", error)
                })?;
                let local = stream.local_addr().map_err(|error| {
                    record::io_failure("inspect a rendezvous connection", error)
                })?;
                Ok((remote, local))
            }
            _ => Err(invalid(
                "a multi-host prefix store requires TCP rank connections",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let remote: Vec<SocketAddr> = addresses.iter().map(|(remote, _)| *remote).collect();
    let local: Vec<SocketAddr> = addresses.iter().map(|(_, local)| *local).collect();
    let listener = if enabled {
        let wildcard = match local.first().map(|local| local.is_ipv4()) {
            Some(true) => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            Some(false) => SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0),
            None => return Err(invalid("a multi-host prefix store needs a peer rank")),
        };
        Some(
            TcpListener::bind(wildcard)
                .map_err(|error| record::io_failure("listen for prefix record peers", error))?,
        )
    } else {
        None
    };
    let token = nonce()?;
    let world = peers.len() + 1;
    let layout = prepared
        .as_ref()
        .map(|prepared| super::geometry::fingerprint(&prepared.geometry));
    let limit = prepared
        .as_ref()
        .map(|prepared| prepared.resolved.disk_limit);
    let mut offers = vec![Vec::new()];
    for peer_local in &local {
        let mut offer = vec![enabled as u8];
        if let Some(listener) = &listener {
            let port = listener
                .local_addr()
                .map_err(|error| record::io_failure("inspect the prefix record listener", error))?
                .port();
            let address = SocketAddr::new(peer_local.ip(), port);
            offer.extend(token);
            offer.extend(layout.unwrap_or_default());
            offer.extend(limit.unwrap_or_default().to_le_bytes());
            offer.extend((address.to_string().len() as u32).to_le_bytes());
            offer.extend(address.to_string().as_bytes());
        }
        offers.push(offer);
    }
    let replies = rendezvous::share_records_each(peers, offers)?;
    let mut peer_enabled = Vec::with_capacity(peers.len());
    let mut limits = Vec::with_capacity(peers.len());
    for (index, reply) in replies[1..].iter().enumerate() {
        let enabled = reply.first() == Some(&1);
        peer_enabled.push(enabled);
        if enabled {
            let mut reader = Reader(&reply[1..]);
            let peer_layout = reader.take(16)?;
            let peer_limit = reader.u64()?;
            reader.end()?;
            if let Some(layout) = layout
                && peer_layout != layout
            {
                return Err(invalid(format!(
                    "rank {} describes a different cache layout than rank zero; every rank of a multi-host package must have the same plane and row sizes",
                    index + 1
                )));
            }
            limits.push(peer_limit);
        }
    }
    if let Some(rank) = peer_enabled.iter().position(|&peer| peer != enabled) {
        return Err(invalid(format!(
            "rank {} {} a disk prefix store and rank zero {}; give every rank the same --prefix-cache-disk settings",
            rank + 1,
            if enabled { "lacks" } else { "has" },
            if enabled { "has one" } else { "does not" },
        )));
    }
    let (Some(listener), Some(prepared)) = (listener, prepared) else {
        return Ok(None);
    };
    // The limit applies to every host's directory, so the smallest one bounds
    // the records every host can hold.
    if let Some(minimum) = limits.iter().copied().min() {
        prepared.resolved.disk_limit = prepared.resolved.disk_limit.min(minimum);
        prepared.disk_budget.set_limit(prepared.resolved.disk_limit);
    }
    let mut links: Vec<Option<Link>> = (1..world).map(|_| None).collect();
    listener
        .set_nonblocking(true)
        .map_err(|error| record::io_failure("configure the prefix record listener", error))?;
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    while links.iter().any(Option::is_none) {
        if Instant::now() >= deadline {
            return Err(invalid(
                "peer ranks did not connect their prefix record connections",
            ));
        }
        let stream = match listener.accept() {
            Ok((stream, address)) => (stream, address),
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(IDLE);
                continue;
            }
            Err(error) => {
                return Err(record::io_failure("accept a prefix record peer", error));
            }
        };
        let (stream, address) = stream;
        let mut link = Link::new(stream)?;
        let Ok(hello) = link.expect(HELLO, HANDSHAKE_TIMEOUT) else {
            continue;
        };
        let mut reader = Reader(&hello);
        let (Ok(offered), Ok(rank), Ok(peer_layout)) =
            (reader.take(16), reader.u32(), reader.take(16))
        else {
            continue;
        };
        let rank = rank as usize;
        if offered != token
            || !(1..world).contains(&rank)
            || links[rank - 1].is_some()
            || address.ip() != remote[rank - 1].ip()
            || Some(peer_layout) != layout.as_ref().map(|layout| layout.as_slice())
        {
            continue;
        }
        links[rank - 1] = Some(link);
    }
    let mut links: Vec<Link> = links.into_iter().map(Option::unwrap).collect();
    let summaries = links
        .iter_mut()
        .map(|link| decode_summary(&link.expect(SUMMARY, REPLY_TIMEOUT)?))
        .collect::<Result<Vec<_>>>()?;
    let agreed = agree(&prepared.scanned, &summaries)?;
    let frame = encode_agreed(&agreed);
    for link in &mut links {
        link.send(&frame)?;
    }
    let scanned = prepared.scanned.len();
    retain(prepared, &agreed)?;
    prepared.shards = links;
    Ok(Some(format!(
        "{} of {scanned} records on every rank, limit {}",
        prepared.scanned.len(),
        super::gibibytes(prepared.resolved.disk_limit)
    )))
}

/// A peer, at the point where rank zero runs [`join_rank_zero`]: connect this rank's record
/// connection, apply the scan agreement, and start the service rank zero drives.
pub fn join_peer(
    store: Option<&Shared>,
    stream: &mut RankStream,
    rank: usize,
    world: usize,
) -> Result<Option<Service>> {
    use crate::execution::rendezvous;

    // Read what the offer needs, then release the guard: taking the prepared
    // resources below locks the same shared cell.
    let (enabled, layout, limit) = match store.map(Shared::lock).as_ref().and_then(|g| g.as_ref()) {
        Some(prepared) => (
            mirrored(prepared)?,
            super::geometry::fingerprint(&prepared.geometry),
            prepared.resolved.disk_limit,
        ),
        None => (false, [0; 16], 0),
    };
    let mut record = vec![enabled as u8];
    if enabled {
        record.extend(layout);
        record.extend(limit.to_le_bytes());
    }
    let records = rendezvous::exchange_record(stream, &record, world)?;
    let offer = &records[0];
    let zero = offer.first() == Some(&1);
    if zero != enabled {
        return Err(invalid(format!(
            "rank zero {} a disk prefix store and rank {rank} {}; give every rank the same --prefix-cache-disk settings",
            if zero { "has" } else { "lacks" },
            if enabled { "has one" } else { "does not" },
        )));
    }
    if !enabled {
        return Ok(None);
    }
    if records[1..].iter().any(|record| record.first() != Some(&1)) {
        return Err(invalid(
            "another peer rank lacks a disk prefix store; give every rank the same --prefix-cache-disk settings",
        ));
    }
    let mut reader = Reader(&offer[1..]);
    let token = reader.take(16)?;
    let zero_layout = reader.take(16)?;
    let zero_limit = reader.u64()?;
    let length = reader.u32()? as usize;
    let address = std::str::from_utf8(reader.take(length)?)
        .map_err(|_| invalid("rank zero's prefix record address is malformed"))?;
    reader.end()?;
    if zero_layout != layout {
        return Err(invalid(format!(
            "rank {rank} describes a different cache layout than rank zero; every rank of a multi-host package must have the same plane and row sizes"
        )));
    }
    let limit = zero_limit.min(limit);
    let connected = TcpStream::connect_timeout(
        &address
            .parse()
            .map_err(|_| invalid("rank zero's prefix record address is malformed"))?,
        CONNECT_TIMEOUT,
    )
    .map_err(|error| {
        record::io_failure(
            format!("connect to rank zero's prefix records at {address}"),
            error,
        )
    })?;
    let mut link = Link::new(connected)?;
    let mut hello = Writer::new(HELLO);
    hello.bytes(token).u32(rank as u32).bytes(&layout);
    link.send(&hello.finish())?;
    let shared = store.ok_or_else(|| invalid("a peer's prefix store vanished"))?;
    let mut prepared = shared
        .take()
        .ok_or_else(|| invalid("a peer's prefix store was already opened"))?;
    // The smallest limit among the ranks bounds what every host can hold.
    if limit < prepared.resolved.disk_limit {
        prepared.resolved.disk_limit = limit;
        prepared.disk_budget.set_limit(limit);
    }
    // Start this rank's record transport before reporting the scan. A failure
    // to open the transfer threads, the disk thread, or the record directory
    // then fails the join instead of interrupting serving, when rank zero has
    // already treated this rank as joined.
    let pool = prepared
        .staging_pool
        .as_ref()
        .ok_or_else(|| invalid("a peer's prefix record service needs a staging pool"))?;
    let transfers = Arc::new(Mutex::new(Transfers::start(
        &prepared.contexts,
        prepared.geometry.clone(),
        pool,
    )?));
    let staged = Staged::new(
        prepared.staging_pool.take(),
        transfers,
        DiskThread::start("flare-offload-disk")?,
    );
    link.send(&summary(&prepared.scanned)?)?;
    let agreed = decode_agreed(&link.expect(AGREED, REPLY_TIMEOUT)?)?;
    retain(&mut prepared, &agreed)?;
    let summary = format!(
        "serving rank zero, limit {}",
        super::gibibytes(prepared.resolved.disk_limit)
    );
    let directory = prepared
        .resolved
        .directory
        .clone()
        .ok_or_else(|| invalid("a peer's prefix record service needs a record directory"))?;
    let lock = prepared
        .directory_lock
        .take()
        .ok_or_else(|| invalid("a peer's prefix record service needs its directory lock"))?;
    let peer = Peer {
        staged,
        directory,
        fingerprint: prepared.fingerprint,
        geometry: prepared.geometry,
        link,
        _lock: lock,
    };
    let (finished, outcome) = std::sync::mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("flare-offload-peer".into())
        .spawn(move || {
            let result = serve(peer);
            let _ = finished.send(result.clone());
            result
        })
        .map_err(|error| invalid(format!("cannot start the prefix record service: {error}")))?;
    Ok(Some(Service {
        thread: Some(thread),
        finished: outcome,
        summary,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::geometry::tests::geometry;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn directory(test: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("flare-shard-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// A staged transport without a staging pool: its disk thread runs truncates and unlinks.
    fn staged() -> Staged {
        let transfers = Transfers::stopped(&[geometry(64, &[("kv", 4096)], 4096)]).unwrap();
        Staged::new(
            None,
            Arc::new(Mutex::new(transfers)),
            DiskThread::start("flare-shard-test").unwrap(),
        )
    }

    fn connected() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (client, listener.accept().unwrap().0)
    }

    fn record_file(directory: &Path, tokens: u64) -> PathBuf {
        let path = directory.join(record::file_name(&key(tokens, 0x5a)));
        std::fs::write(&path, vec![1u8; 8192]).unwrap();
        path
    }

    /// Poll until `count` events arrived, failing after a few seconds.
    fn events(transport: &mut Sharded, count: usize) -> Vec<TransportEvent> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut events = Vec::new();
        while events.len() < count {
            assert!(Instant::now() < deadline, "only {events:?} arrived");
            transport.poll(&mut events);
            std::thread::sleep(IDLE);
        }
        events
    }

    /// Poll for a while and require that nothing was reported.
    fn quiet(transport: &mut Sharded) {
        let mut events = Vec::new();
        for _ in 0..50 {
            transport.poll(&mut events);
            std::thread::sleep(IDLE);
        }
        assert!(events.is_empty(), "{events:?}");
    }

    fn reply(peer: &mut Link) -> (u8, Vec<u8>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(frame) = peer.receive().unwrap() {
                return frame;
            }
            assert!(Instant::now() < deadline, "no frame arrived");
            std::thread::sleep(IDLE);
        }
    }

    #[test]
    fn rank_zero_reports_a_job_after_every_rank_and_unlinks_partial_stores() {
        let directory = directory("zero");
        let (zero, peer) = connected();
        let mut peer = Link::new(peer).unwrap();
        let mut transport = Sharded::new(staged(), vec![Link::new(zero).unwrap()]).unwrap();

        // Rank zero's unlink finishes first; the job waits for the peer.
        let first = record_file(&directory, 128);
        transport.unlink(JobId(1), first.clone()).unwrap();
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        assert_eq!((tag, reader.u64().unwrap()), (UNLINK, 1));
        assert_eq!(reader.text().unwrap(), name(&first).unwrap());
        quiet(&mut transport);
        assert!(!first.exists());
        peer.send(&encode_event(Kind::Unlinked, JobId(1), &Ok(0)))
            .unwrap();
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Unlinked {
                id: JobId(1),
                result: Ok(())
            }]
        ));

        // A peer's failure fails the job and names the rank.
        let second = record_file(&directory, 256);
        transport.truncate(JobId(2), second.clone(), 4096).unwrap();
        assert_eq!(reply(&mut peer).0, TRUNCATE);
        peer.send(&encode_event(
            Kind::Truncated,
            JobId(2),
            &Err(invalid("disk full")),
        ))
        .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::Truncated {
                    id: JobId(2),
                    result: Err(error),
                },
            ] => {
                assert!(error.to_string().contains("rank 1: "), "{error}");
                assert!(error.to_string().contains("disk full"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(std::fs::metadata(&second).unwrap().len(), 4096);

        // Rank zero wrote a record the peer failed to write: rank zero removes its file and
        // only then reports the store as failed. The peer, which wrote nothing, is not asked.
        let third = record_file(&directory, 384);
        transport.pending.insert(
            (Kind::Stored, JobId(3)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Ok(8192)), None],
                path: Some(third.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(3), 1, Err(invalid("no space")));
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::StoreInvalidated {
                    id: JobId(3),
                    error,
                },
            ] => {
                assert!(error.to_string().contains("no space"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert!(!third.exists());

        // The peer wrote a record rank zero failed to write: the peer is asked to remove it,
        // with a job id outside the disk store's, and the failure waits for its answer.
        let fourth = record_file(&directory, 512);
        transport.pending.insert(
            (Kind::Stored, JobId(4)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("short write"))), None],
                path: Some(fourth.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(4), 1, Ok(8192));
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        let internal = reader.u64().unwrap();
        assert_eq!(tag, UNLINK);
        assert!(internal >= INTERNAL_JOBS);
        assert_eq!(reader.text().unwrap(), name(&fourth).unwrap());
        quiet(&mut transport);
        peer.send(&encode_event(Kind::Unlinked, JobId(internal), &Ok(0)))
            .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::StoreInvalidated {
                    id: JobId(4),
                    error,
                },
            ] => {
                assert!(error.to_string().contains("short write"), "{error}");
            }
            other => panic!("{other:?}"),
        }

        // A cleanup unlink that itself fails leaves a file the index does not
        // charge for, so the failure names the rank that kept it.
        let sixth = record_file(&directory, 576);
        transport.pending.insert(
            (Kind::Stored, JobId(6)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("no space"))), None],
                path: Some(sixth.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(6), 1, Ok(8192));
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        let internal = reader.u64().unwrap();
        assert_eq!(tag, UNLINK);
        peer.send(&encode_event(
            Kind::Unlinked,
            JobId(internal),
            &Err(invalid("directory is read-only")),
        ))
        .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::StoreInvalidated {
                    id: JobId(6),
                    error,
                },
            ] => {
                let error = error.to_string();
                assert!(error.contains("no space"), "{error}");
                assert!(error.contains("cleanup unlink failed"), "{error}");
                assert!(error.contains("rank 1"), "{error}");
            }
            other => panic!("{other:?}"),
        }

        // When every write fails, the previous files remain usable.
        let seventh = record_file(&directory, 608);
        transport.pending.insert(
            (Kind::Stored, JobId(7)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("write failed"))), None],
                path: Some(seventh.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(7), 1, Err(invalid("disk full")));
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Stored {
                id: JobId(7),
                result: Err(_)
            }]
        ));
        assert!(seventh.exists());

        // A broken connection fails every job still waiting on that rank.
        let fifth = record_file(&directory, 640);
        transport.unlink(JobId(5), fifth).unwrap();
        assert_eq!(reply(&mut peer).0, UNLINK);
        drop(peer);
        // The peer may still be copying into the ids of the jobs it already
        // received, so the transport stops instead of reporting them failed.
        let mut reported = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        let error = loop {
            assert!(
                Instant::now() < deadline,
                "the transport never noticed the broken link"
            );
            match transport.poll(&mut reported) {
                Ok(()) => std::thread::sleep(IDLE),
                Err(error) => break error,
            }
        };
        assert!(error.to_string().contains("rank 1"), "{error}");
        assert!(reported.is_empty(), "{reported:?}");
        assert!(
            transport
                .unlink(JobId(6), record_file(&directory, 768))
                .is_err()
        );
        // A poisoned transport never confirms its drain, so its staging and
        // transfer threads stay alive with it.
        assert!(transport.close().is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn peer_service_mirrors_jobs_into_its_own_directory_until_closed() {
        let zero_directory = directory("mirror-zero");
        let peer_directory = directory("mirror-peer");
        let (zero, peer) = connected();
        let lock = record::DirectoryLock::acquire(&peer_directory).unwrap();
        let service_directory = peer_directory.clone();
        let service = std::thread::spawn(move || {
            let mut peer = Peer {
                staged: staged(),
                directory: service_directory,
                fingerprint: [3; 16],
                geometry: vec![geometry(64, &[("kv", 4096)], 4096)],
                link: Link::new(peer).unwrap(),
                _lock: lock,
            };
            let outcome = peer.run();
            let shutdown = peer.shutdown();
            if shutdown.is_err() {
                std::mem::forget(peer);
            }
            outcome.and(shutdown)
        });
        let mut transport = Sharded::new(staged(), vec![Link::new(zero).unwrap()]).unwrap();
        let (ours, theirs) = (
            record_file(&zero_directory, 128),
            record_file(&peer_directory, 128),
        );
        let (kept_ours, kept_theirs) = (
            record_file(&zero_directory, 256),
            record_file(&peer_directory, 256),
        );
        transport.unlink(JobId(1), ours.clone()).unwrap();
        transport
            .truncate(JobId(2), kept_ours.clone(), 4096)
            .unwrap();
        let mut events = events(&mut transport, 2);
        events.sort_by_key(|event| Kind::of(event).1);
        assert!(matches!(
            events.as_slice(),
            [
                TransportEvent::Unlinked {
                    id: JobId(1),
                    result: Ok(())
                },
                TransportEvent::Truncated {
                    id: JobId(2),
                    result: Ok(())
                },
            ]
        ));
        assert!(!ours.exists() && !theirs.exists());
        assert_eq!(std::fs::metadata(&kept_ours).unwrap().len(), 4096);
        assert_eq!(std::fs::metadata(&kept_theirs).unwrap().len(), 4096);

        transport.close().unwrap();
        service.join().unwrap().unwrap();
        std::fs::remove_dir_all(zero_directory).unwrap();
        std::fs::remove_dir_all(peer_directory).unwrap();
    }

    fn key(tokens: u64, seed: u8) -> PrefixKey {
        PrefixKey {
            consumed_tokens: tokens,
            digest: [seed; 16],
        }
    }

    fn scanned(key: PrefixKey, parent: Option<PrefixKey>, row: bool, fingerprint: u8) -> Scanned {
        let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
        let mut header = Header::new(
            &geometry,
            [7; 32],
            [fingerprint; 16],
            key,
            parent,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        header.row_checksum = fingerprint as u64;
        Scanned {
            bytes: if row {
                header.length_with_row()
            } else {
                header.row_segment
            },
            path: PathBuf::from(record::file_name(&key)),
            header,
            row,
            modified: SystemTime::now(),
        }
    }

    #[test]
    fn an_unconfirmed_drain_is_reported_so_the_peer_keeps_its_staging() {
        let directory = directory("drain");
        let lock = record::DirectoryLock::acquire(&directory).unwrap();
        let (zero, peer) = connected();
        let mut peer = Peer {
            staged: staged(),
            directory,
            fingerprint: [3; 16],
            geometry: vec![geometry(64, &[("kv", 4096)], 4096)],
            link: Link::new(peer).unwrap(),
            _lock: lock,
        };
        // The link failed, so the peer cannot report that it drained. The
        // caller must retain the transport rather than release its staging
        // under copies that may still run.
        peer.link.fail(invalid("rank zero closed the connection"));
        assert!(peer.shutdown().is_err());
        drop(zero);
    }

    #[test]
    fn ranks_without_a_store_exchange_no_record_settings() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let mut zero_stream = RankStream::Tcp(server);
        let mut peer_stream = RankStream::Tcp(client);
        let zero = std::thread::spawn(move || {
            let mut peers = vec![zero_stream];
            join_rank_zero(None, &mut peers)
        });
        assert!(join_peer(None, &mut peer_stream, 1, 2).unwrap().is_none());
        assert!(zero.join().unwrap().unwrap().is_none());
    }

    #[test]
    fn a_job_past_its_deadline_stops_the_transport() {
        let (zero, _peer) = connected();
        let mut transport = Sharded::new(staged(), vec![Link::new(zero).unwrap()]).unwrap();
        transport.pending.insert(
            (Kind::Unlinked, JobId(7)),
            Pending {
                kind: Kind::Unlinked,
                results: vec![Some(Ok(0)), None],
                path: None,
                cleanup: None,
                started: Instant::now() - JOB_TIMEOUT - Duration::from_secs(1),
            },
        );
        transport.expire();
        let mut events = Vec::new();
        let error = transport.poll(&mut events).unwrap_err();
        assert!(error.to_string().contains("did not complete"), "{error}");
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn a_peer_rejects_a_header_from_another_layout() {
        let directory = directory("layout");
        let lock = record::DirectoryLock::acquire(&directory).unwrap();
        let (zero, peer) = connected();
        let mut peer = Peer {
            staged: staged(),
            directory,
            fingerprint: [3; 16],
            geometry: vec![geometry(64, &[("kv", 4096)], 4096)],
            link: Link::new(peer).unwrap(),
            _lock: lock,
        };
        let own = Header::new(
            &peer.geometry,
            [7; 32],
            [1; 16],
            key(128, 0x5a),
            None,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        assert!(peer.own(own).is_ok());
        let wider = vec![geometry(64, &[("kv", 8192)], 8192)];
        let foreign = Header::new(
            &wider,
            [7; 32],
            [1; 16],
            key(128, 0x5a),
            None,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        assert!(peer.own(foreign).is_err());
        drop(zero);
    }

    #[test]
    fn a_peer_that_stops_reading_fails_the_job_instead_of_blocking() {
        let (zero, peer) = connected();
        let mut link = Link::new(zero).unwrap();
        // The peer never reads, so the socket buffer and then the queue fill.
        let frame = vec![0u8; 1 << 20];
        let started = Instant::now();
        let mut failure = None;
        for _ in 0..(BACKLOG * 4) {
            if let Err(error) = link.enqueue(&frame) {
                failure = Some(error);
                break;
            }
        }
        assert!(started.elapsed() < Duration::from_secs(5));
        let failure = failure.expect("the queue accepted more than its backlog");
        assert!(failure.to_string().contains("backlog"), "{failure}");
        drop(peer);
    }

    #[test]
    fn agreement_keeps_records_every_rank_holds_and_drops_their_orphans() {
        let (a, b, c, d) = (key(128, 1), key(256, 2), key(384, 3), key(128, 4));
        let zero = vec![
            scanned(a, None, true, 0),
            scanned(b, Some(a), true, 0),
            scanned(c, Some(b), true, 0),
            scanned(d, None, true, 0),
        ];
        // Rank one lacks b's row and holds no d; rank two lacks d's row too.
        let one = vec![
            scanned(a, None, true, 1),
            scanned(b, Some(a), false, 1),
            scanned(c, Some(b), true, 1),
        ];
        let two = vec![
            scanned(a, None, true, 2),
            scanned(b, Some(a), true, 2),
            scanned(c, Some(b), true, 2),
            scanned(d, None, false, 2),
        ];
        let peers = [one, two]
            .iter()
            .map(|records| decode_summary(&summary(records).unwrap()[5..]).unwrap())
            .collect::<Vec<_>>();
        let agreed = agree(&zero, &peers).unwrap();
        assert_eq!(agreed, HashMap::from([(a, true), (b, false), (c, true)]));
        let decoded = decode_agreed(&encode_agreed(&agreed)[5..]).unwrap();
        assert_eq!(decoded, agreed);

        // A peer whose header for `a` differs drops `a` and, with it, every descendant.
        let mut other = scanned(a, None, true, 1);
        other.header.bucket = [9; 16];
        let peers = [decode_summary(&summary(&[other]).unwrap()[5..]).unwrap()];
        assert!(agree(&zero, &peers).unwrap().is_empty());
    }

    #[test]
    fn events_and_peer_file_names_round_trip_and_reject_paths() {
        let event = TransportEvent::StoreInvalidated {
            id: JobId(11),
            error: invalid("partial store"),
        };
        let (kind, id, result) = Kind::of(&event);
        let frame = encode_event(kind, id, &result);
        let (kind, id, result) = decode_event(&frame[5..]).unwrap();
        assert!(matches!(
            kind.event(id, result),
            TransportEvent::StoreInvalidated { id: JobId(11), .. }
        ));
        for result in [Ok(42u64), Err(invalid("disk full"))] {
            let frame = encode_event(Kind::Fetched, JobId(9), &result);
            let (kind, id, decoded) = decode_event(&frame[5..]).unwrap();
            assert_eq!((kind, id), (Kind::Fetched, JobId(9)));
            match (result, decoded) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(a), Err(b)) => assert!(b.to_string().contains(&a.to_string())),
                other => panic!("{other:?}"),
            }
        }
        let directory = Path::new("/records");
        let key = key(4096, 0xab);
        let plain = record::file_name(&key);
        assert_eq!(file(directory, &plain).unwrap(), directory.join(&plain));
        let temporary = record::temporary_name(&key);
        assert_eq!(
            file(directory, &temporary).unwrap(),
            directory.join(&temporary)
        );
        for name in [
            "../etc/passwd",
            "/abs.ckpt",
            "x.ckpt",
            ".tmp",
            "",
            &format!("a/{plain}"),
        ] {
            assert!(file(directory, name).is_err(), "{name:?}");
        }
    }
}
