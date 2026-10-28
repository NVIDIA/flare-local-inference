//! A thread that moves record segments between files and caller-owned buffers
//! with direct I/O.
//!
//! The thread owns an io_uring ring on Linux or a completion port on Windows.
//! Files use unbuffered I/O, so
//! every buffer, offset, and length is a multiple of [`BLOCK_BYTES`], which the
//! record layout guarantees. Transfers are chunked at [`CHUNK_BYTES`]. Writes
//! hash every page record and the row as they pass through; reads verify them
//! against the record's checksum segment and header. The thread runs under
//! reduced scheduling priority so it yields to serving threads.
//!
//! Jobs execute one at a time in submission order, so a record's `Create`,
//! page and row writes, and `Commit` or `Abort` form a sequence under one id,
//! as do `Open`, reads, and `Close`. Every job produces exactly one [`Reply`].
//! A job that fails leaves no partial record behind: a page or row write that
//! fails leaves its pages or row unwritten, `Commit` refuses a record with an
//! unwritten page or row, and a failed `Commit` closes and unlinks the
//! temporary file, as does thread exit for every record still being written.
//! The thread allocates its ring and its aligned scratch block once at start;
//! the scratch holds the header block and a checksum segment of up to
//! [`MAX_TAIL_PAGES`] pages, which bounds the records the thread accepts.

use std::{
    cell::Cell,
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
};

#[cfg(unix)]
use io_uring::{EnterFlags, IoUring, opcode, types};
#[cfg(unix)]
use std::os::fd::AsRawFd;

use crate::{Error, Result};

use super::{
    JobId,
    geometry::{BLOCK_BYTES, invalid},
    record::{HEADER_BYTES, Header, checksum, io_failure},
};

#[cfg(any(test, feature = "test-support"))]
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
pub struct IoGate {
    armed: AtomicBool,
    entered: SyncSender<()>,
    release: std::sync::Mutex<Receiver<()>>,
}
#[cfg(any(test, feature = "test-support"))]
impl IoGate {
    pub fn new() -> (Arc<Self>, Receiver<()>, SyncSender<()>) {
        let (entered, ready) = mpsc::sync_channel(1);
        let (release, proceed) = mpsc::sync_channel(1);
        (
            Arc::new(Self {
                armed: AtomicBool::new(true),
                entered,
                release: std::sync::Mutex::new(proceed),
            }),
            ready,
            release,
        )
    }
    pub fn arm(&self) {
        assert!(
            !self.armed.swap(true, Ordering::AcqRel),
            "disk gate is already armed"
        );
    }
    fn wait(&self) -> io::Result<()> {
        if self.armed.swap(false, Ordering::AcqRel) {
            self.entered
                .send(())
                .map_err(|_| io::Error::other("disk gate observer dropped"))?;
            self.release
                .lock()
                .unwrap()
                .recv()
                .map_err(|_| io::Error::other("disk gate released without a signal"))?;
        }
        Ok(())
    }
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, Default)]
pub struct Faults {
    pub io_gate: Option<Arc<IoGate>>,
    pub read_gate: Option<Arc<IoGate>>,
    pub pause_at: Option<String>,
    pub signal: Option<PathBuf>,
    pub disk_full: Option<PathBuf>,
    pub delay_ms: u64,
    pub trace: Option<Arc<std::sync::Mutex<Vec<u8>>>>,
}
#[cfg(any(test, feature = "test-support"))]
impl Faults {
    pub fn checkpoint(&self, point: &str) {
        if self.pause_at.as_deref() == Some(point) {
            if let Some(path) = &self.signal {
                fs::write(path, point).unwrap();
                while path.exists() {
                    thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }
    fn trace(&self, kind: u8) {
        if let Some(trace) = &self.trace {
            trace.lock().unwrap().push(kind);
        }
    }
    fn before_io(&self) -> io::Result<()> {
        if let Some(gate) = &self.io_gate {
            gate.wait()?;
        }
        if self.delay_ms > 0 {
            thread::sleep(Duration::from_millis(self.delay_ms));
        }
        if self.disk_full.as_ref().is_some_and(|path| path.exists()) {
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }
        Ok(())
    }
}

pub const CHUNK_BYTES: usize = 8 << 20;
/// Jobs the submitting store may have outstanding; `submit` fails when full.
pub const QUEUE_DEPTH: usize = 64;
pub const NICE: i32 = 10;
/// Chunks the ring keeps in flight for one transfer.
const RING_DEPTH: usize = 64;
/// Bytes of the thread's aligned checksum scratch.
const CHECKSUM_SCRATCH_BYTES: usize = 1 << 20;
/// Tail pages per record the thread can write or open, bounded by the scratch
/// that holds a record's checksum segment during `Commit` and `Open`.
pub const MAX_TAIL_PAGES: u32 = (CHECKSUM_SCRATCH_BYTES / 8) as u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    ReadRecord,
    WriteRecord,
    Create,
    WritePages,
    WriteRow,
    Commit,
    Abort,
    Open,
    ReadPages,
    ReadRow,
    Close,
    Truncate,
    Unlink,
}

/// Work for the disk thread. Buffers are host addresses of page or row slabs.
/// `Create` through `Commit` or `Abort` share one id and run in submission
/// order; `Open` through `Close` likewise. A buffer must stay valid and
/// untouched until the job's reply arrives.
#[derive(Debug)]
pub enum Job {
    /// Read an encoded record span. The caller verifies its page and row checksums.
    #[cfg(unix)]
    ReadRecord {
        id: JobId,
        file: Arc<File>,
        span: Span,
    },
    /// Write a prevalidated, checksummed record span, optionally syncing and publishing it.
    /// The caller retains the buffer and temporary path until the reply.
    #[cfg(unix)]
    WriteRecord {
        id: JobId,
        temporary: PathBuf,
        span: Span,
        publish: Option<(PathBuf, u64)>,
    },
    /// Create the temporary file for a record and keep it open under `id`.
    Create {
        id: JobId,
        temporary: PathBuf,
        header: Header,
    },
    /// Write page records `first..first + buffers.len()` from `buffers`.
    WritePages {
        id: JobId,
        window: u32,
        first: u32,
        buffers: Vec<usize>,
    },
    WriteRow {
        id: JobId,
        buffer: usize,
    },
    /// Write the checksum segment and header, optionally `fdatasync`, rename to
    /// `path`, and close. Every page record and the row must have been written.
    Commit {
        id: JobId,
        path: PathBuf,
        sync: bool,
    },
    /// Close and unlink the temporary file.
    Abort {
        id: JobId,
    },
    /// Validate the file's header before loading checksums or accepting payload reads.
    Open {
        id: JobId,
        path: PathBuf,
        expected: Header,
    },
    /// Read page records `first..first + buffers.len()` into `buffers`, verifying checksums.
    ReadPages {
        id: JobId,
        window: u32,
        first: u32,
        buffers: Vec<usize>,
    },
    ReadRow {
        id: JobId,
        buffer: usize,
    },
    Close {
        id: JobId,
    },
    /// Drop the row segment of a committed record.
    Truncate {
        id: JobId,
        path: PathBuf,
        length: u64,
    },
    Unlink {
        id: JobId,
        path: PathBuf,
    },
}

impl Job {
    /// The caller retains `data` until the reply, as for page and row jobs.
    #[cfg(unix)]
    pub fn write_record(
        id: JobId,
        temporary: PathBuf,
        offset: u64,
        data: &[u8],
        publish: Option<(PathBuf, u64)>,
    ) -> Self {
        Self::WriteRecord {
            id,
            temporary,
            span: Span {
                address: data.as_ptr() as usize,
                offset,
                length: data.len(),
            },
            publish,
        }
    }
}

#[derive(Debug)]
pub struct Reply {
    pub id: JobId,
    pub operation: Operation,
    pub window: u32,
    pub result: Result<()>,
    /// Bytes moved by this job; for `Commit`, every byte written for the record.
    pub bytes: u64,
    pub elapsed: std::time::Duration,
    /// The header read by `Open`.
    pub header: Option<Header>,
}

pub struct DiskThread {
    sender: Option<SyncSender<Job>>,
    receiver: Receiver<Reply>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Jobs submitted whose replies have not been polled.
    outstanding: Cell<usize>,
}

impl DiskThread {
    /// Spawn the thread under `name` with its ring and scratch allocated.
    pub fn start(name: &str) -> Result<Self> {
        Self::start_inner(
            name,
            #[cfg(any(test, feature = "test-support"))]
            Faults::default(),
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn start_with_faults(name: &str, faults: Faults) -> Result<Self> {
        Self::start_inner(name, faults)
    }

    fn start_inner(
        name: &str,
        #[cfg(any(test, feature = "test-support"))] faults: Faults,
    ) -> Result<Self> {
        let ring = Ring::new()?;
        let scratch = AlignedBuffer::new(HEADER_BYTES + CHECKSUM_SCRATCH_BYTES)?;
        let (sender, jobs) = mpsc::sync_channel(QUEUE_DEPTH);
        let (replies, receiver) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                run(
                    ring,
                    scratch,
                    jobs,
                    replies,
                    #[cfg(any(test, feature = "test-support"))]
                    faults,
                )
            })
            .map_err(|error| io_failure(format!("spawn disk thread {name}"), error))?;
        Ok(Self {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
            outstanding: Cell::new(0),
        })
    }

    /// Enqueue without waiting; an error means the queue is full or the thread stopped.
    pub fn submit(&self, job: Job) -> Result<()> {
        match self.try_submit(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(invalid(format!(
                "disk thread queue is full with {QUEUE_DEPTH} jobs outstanding"
            ))),
            Err(TrySendError::Disconnected(_)) => Err(invalid("disk thread stopped")),
        }
    }

    /// Return an unsubmitted job so the caller can retain its buffers and retry.
    pub fn try_submit(&self, job: Job) -> std::result::Result<(), TrySendError<Job>> {
        let Some(sender) = &self.sender else {
            return Err(TrySendError::Disconnected(job));
        };
        sender.try_send(job)?;
        self.outstanding.set(self.outstanding.get() + 1);
        Ok(())
    }

    pub fn poll(&self, replies: &mut Vec<Reply>) {
        while let Ok(reply) = self.receiver.try_recv() {
            self.outstanding
                .set(self.outstanding.get().saturating_sub(1));
            replies.push(reply);
        }
    }

    /// Finish outstanding jobs and join the thread. The thread answers every
    /// queued job before it exits, and those replies remain available through
    /// `poll` after this returns. Records still being written when the thread
    /// exits are closed and their temporary files unlinked; records open for
    /// reading are closed. A timeout retains the worker and its resources;
    /// callers must retain their buffers until a later successful drain.
    pub fn close(&mut self) -> Result<()> {
        self.close_until(std::time::Instant::now() + super::copier::CLOSE_TIMEOUT)
    }

    fn close_until(&mut self, deadline: std::time::Instant) -> Result<()> {
        self.sender.take();
        while self
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
        {
            if std::time::Instant::now() >= deadline {
                return Err(invalid(format!(
                    "disk thread did not drain before shutdown deadline; {} jobs await replies",
                    self.outstanding.get()
                )));
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        thread.join().map_err(|_| {
            invalid(format!(
                "disk thread stopped abnormally; {} submitted jobs have no polled reply",
                self.outstanding.get()
            ))
        })
    }
}

impl Drop for DiskThread {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Every direct-I/O span must be a multiple of the block size.
pub fn aligned(value: u64) -> bool {
    value.is_multiple_of(BLOCK_BYTES)
}

fn run(
    ring: Ring,
    scratch: AlignedBuffer,
    jobs: Receiver<Job>,
    replies: mpsc::Sender<Reply>,
    #[cfg(any(test, feature = "test-support"))] faults: Faults,
) {
    lower_priority();
    let mut worker = Worker {
        ring,
        scratch,
        records: HashMap::new(),
        #[cfg(any(test, feature = "test-support"))]
        faults,
    };
    while let Ok(job) = jobs.recv() {
        let started = std::time::Instant::now();
        let mut reply = worker.execute(job);
        reply.elapsed = started.elapsed();
        if replies.send(reply).is_err() {
            break;
        }
    }
}

/// Lower scheduling priority where the host permits it.
fn lower_priority() {
    flare_eos::lower_priority(NICE);
}

/// A record being written: its temporary file, header, and the checksums of
/// the page records written so far.
struct Writing {
    file: File,
    temporary: PathBuf,
    header: Header,
    checksums: Vec<u64>,
    written: Vec<bool>,
    row_written: bool,
    bytes: u64,
}

/// A committed record open for reading with its checksum segment loaded.
struct Reading {
    file: File,
    path: PathBuf,
    header: Header,
    checksums: Vec<u64>,
}

enum Record {
    Writing(Writing),
    Reading(Reading),
}

struct Worker {
    ring: Ring,
    scratch: AlignedBuffer,
    records: HashMap<JobId, Record>,
    #[cfg(any(test, feature = "test-support"))]
    faults: Faults,
}

impl Drop for Worker {
    fn drop(&mut self) {
        for record in self.records.values() {
            if let Record::Writing(writing) = record {
                let _ = fs::remove_file(&writing.temporary);
            }
        }
    }
}

impl Worker {
    fn execute(&mut self, job: Job) -> Reply {
        match job {
            #[cfg(unix)]
            Job::ReadRecord { id, file, span } => {
                let result = (|| {
                    #[cfg(any(test, feature = "test-support"))]
                    if let Some(gate) = &self.faults.read_gate {
                        gate.wait().map_err(Error::carrying)?;
                    }
                    self.ring.transfer(&file, &[span], false)
                })();
                #[cfg(any(test, feature = "test-support"))]
                self.faults.trace(b'R');
                reply(id, Operation::ReadRecord, 0, result)
            }
            #[cfg(unix)]
            Job::WriteRecord {
                id,
                temporary,
                span,
                publish,
            } => {
                let result = (|| {
                    #[cfg(any(test, feature = "test-support"))]
                    {
                        self.faults.before_io().map_err(Error::carrying)?;
                        if publish
                            .as_ref()
                            .is_some_and(|(_, size)| *size == span.length as u64)
                        {
                            self.faults.checkpoint("parent_write");
                        }
                    }
                    let file = flare_eos::direct_update_file(&temporary, true)
                        .map_err(|e| io_failure("open record writer", e))?;
                    let bytes = self.ring.transfer(&file, &[span], true)?;
                    if let Some((path, final_length)) = publish {
                        file.set_len(final_length)
                            .map_err(|e| io_failure("size record", e))?;
                        self.publish(&file, &temporary, &path, true)?;
                    }
                    Ok(bytes)
                })();
                #[cfg(any(test, feature = "test-support"))]
                self.faults.trace(b'W');
                reply(id, Operation::WriteRecord, 0, result)
            }
            Job::Create {
                id,
                temporary,
                header,
            } => reply(id, Operation::Create, 0, self.create(id, temporary, header)),
            Job::WritePages {
                id,
                window,
                first,
                buffers,
            } => reply(
                id,
                Operation::WritePages,
                window,
                self.write_pages(id, first, &buffers),
            ),
            Job::WriteRow { id, buffer } => {
                reply(id, Operation::WriteRow, 0, self.write_row(id, buffer))
            }
            Job::Commit { id, path, sync } => {
                reply(id, Operation::Commit, 0, self.commit(id, &path, sync))
            }
            Job::Abort { id } => reply(id, Operation::Abort, 0, self.abort(id)),
            Job::Open { id, path, expected } => match self.open(id, path, expected) {
                Ok((bytes, header)) => Reply {
                    id,
                    operation: Operation::Open,
                    window: 0,
                    result: Ok(()),
                    bytes,
                    elapsed: std::time::Duration::ZERO,
                    header: Some(header),
                },
                Err(error) => reply(id, Operation::Open, 0, Err(error)),
            },
            Job::ReadPages {
                id,
                window,
                first,
                buffers,
            } => reply(
                id,
                Operation::ReadPages,
                window,
                self.read_pages(id, first, &buffers),
            ),
            Job::ReadRow { id, buffer } => {
                reply(id, Operation::ReadRow, 0, self.read_row(id, buffer))
            }
            Job::Close { id } => reply(id, Operation::Close, 0, self.close(id)),
            Job::Truncate { id, path, length } => {
                let result = (|| {
                    #[cfg(any(test, feature = "test-support"))]
                    self.faults.before_io().map_err(Error::carrying)?;
                    truncate(&path, length)
                })();
                #[cfg(any(test, feature = "test-support"))]
                self.faults.trace(b'W');
                reply(id, Operation::Truncate, 0, result)
            }
            Job::Unlink { id, path } => reply(id, Operation::Unlink, 0, unlink(&path)),
        }
    }

    fn create(&mut self, id: JobId, temporary: PathBuf, header: Header) -> Result<u64> {
        if self.records.contains_key(&id) {
            return Err(invalid(format!("disk job {} is already open", id.0)));
        }
        check_tail_pages(&header)?;
        let file = flare_eos::direct_file(&temporary, true)
            .map_err(|error| direct_open_failure("create record file", &temporary, error))?;
        #[cfg(windows)]
        self.ring
            .queue
            .attach(&file)
            .map_err(|e| io_failure("attach record writer", e))?;
        let tail = header.tail_pages as usize;
        self.records.insert(
            id,
            Record::Writing(Writing {
                file,
                temporary,
                header,
                checksums: vec![0; tail],
                written: vec![false; tail],
                row_written: false,
                bytes: 0,
            }),
        );
        Ok(0)
    }

    /// Pages count as written, and their checksums are recorded, only after
    /// the transfer completed, so a failed write keeps `Commit` from renaming
    /// a record with an unwritten page into place.
    fn write_pages(&mut self, id: JobId, first: u32, buffers: &[usize]) -> Result<u64> {
        #[cfg(any(test, feature = "test-support"))]
        self.faults.before_io().map_err(Error::carrying)?;
        let writing = writing(&mut self.records, id)?;
        let page_bytes = writing.header.page_record_bytes();
        let pages = page_range(&writing.header, first, buffers.len())?;
        let spans: Vec<Span> = pages
            .clone()
            .zip(buffers)
            .map(|(page, address)| Span {
                address: *address,
                offset: writing.header.page_segment + page as u64 * page_bytes,
                length: page_bytes as usize,
            })
            .collect();
        let moved = self.ring.transfer(&writing.file, &spans, true)?;
        for (page, address) in pages.zip(buffers) {
            // SAFETY: The transfer checked the span's alignment and completed;
            // the caller keeps the slab at `address` valid and unchanged for
            // `page_bytes` until this job's reply.
            let bytes =
                unsafe { std::slice::from_raw_parts(*address as *const u8, page_bytes as usize) };
            writing.checksums[page] = checksum(bytes);
            writing.written[page] = true;
        }
        writing.bytes += moved;
        Ok(moved)
    }

    fn write_row(&mut self, id: JobId, buffer: usize) -> Result<u64> {
        let writing = writing(&mut self.records, id)?;
        let row_bytes = writing.header.row_record_bytes();
        let span = Span {
            address: buffer,
            offset: writing.header.row_segment,
            length: row_bytes as usize,
        };
        let moved = self.ring.transfer(&writing.file, &[span], true)?;
        // SAFETY: As for page buffers.
        let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, row_bytes as usize) };
        writing.header.row_checksum = checksum(bytes);
        writing.row_written = true;
        writing.bytes += moved;
        Ok(moved)
    }

    /// Remove the record being written under `id`.
    fn take_writing(&mut self, id: JobId) -> Result<Writing> {
        writing(&mut self.records, id)?;
        match self.records.remove(&id) {
            Some(Record::Writing(writing)) => Ok(writing),
            Some(other) => {
                self.records.insert(id, other);
                Err(invalid(format!(
                    "disk job {} is not a record being written",
                    id.0
                )))
            }
            None => Err(invalid(format!("disk job {} names no open record", id.0))),
        }
    }

    fn commit(&mut self, id: JobId, path: &Path, sync: bool) -> Result<u64> {
        let mut writing = self.take_writing(id)?;
        let outcome = self.finish(&mut writing, path, sync);
        if outcome.is_err() {
            drop(writing.file);
            let _ = fs::remove_file(&writing.temporary);
        }
        outcome
    }

    fn finish(&mut self, writing: &mut Writing, path: &Path, sync: bool) -> Result<u64> {
        let missing = writing.written.iter().filter(|written| !**written).count();
        if missing > 0 || !writing.row_written {
            return Err(invalid(format!(
                "record {} cannot be committed: {missing} of {} page records and {} row are unwritten",
                writing.temporary.display(),
                writing.written.len(),
                if writing.row_written { "no" } else { "its" }
            )));
        }
        let checksum_bytes = writing.header.checksum_bytes() as usize;
        let encoded = writing.header.encode()?;
        let scratch = self.scratch.as_mut_slice();
        scratch[..HEADER_BYTES].copy_from_slice(&encoded);
        let segment = &mut scratch[HEADER_BYTES..HEADER_BYTES + checksum_bytes];
        for (slot, sum) in segment.chunks_exact_mut(8).zip(&writing.checksums) {
            slot.copy_from_slice(&sum.to_le_bytes());
        }
        segment[writing.checksums.len() * 8..].fill(0);
        let mut spans = vec![Span {
            address: self.scratch.address(),
            offset: 0,
            length: HEADER_BYTES,
        }];
        if checksum_bytes > 0 {
            spans.push(Span {
                address: self.scratch.address() + HEADER_BYTES,
                offset: HEADER_BYTES as u64,
                length: checksum_bytes,
            });
        }
        writing.bytes += self.ring.transfer(&writing.file, &spans, true)?;
        self.publish(&writing.file, &writing.temporary, path, sync)?;
        Ok(writing.bytes)
    }

    fn publish(&self, file: &File, temporary: &Path, path: &Path, sync: bool) -> Result<()> {
        if sync {
            file.sync_data().map_err(|e| io_failure("sync record", e))?;
        }
        #[cfg(any(test, feature = "test-support"))]
        self.faults.checkpoint("before_rename");
        fs::rename(temporary, path)
            .map_err(|e| io_failure(format!("rename record {}", path.display()), e))?;
        if sync {
            File::open(path.parent().unwrap())
                .and_then(|f| f.sync_all())
                .map_err(|e| io_failure("sync record directory", e))?;
        }
        #[cfg(any(test, feature = "test-support"))]
        self.faults.checkpoint("after_rename");
        Ok(())
    }

    fn abort(&mut self, id: JobId) -> Result<u64> {
        let writing = self.take_writing(id)?;
        drop(writing.file);
        match fs::remove_file(&writing.temporary) {
            Ok(()) => Ok(0),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(io_failure(
                format!(
                    "unlink temporary record file {}",
                    writing.temporary.display()
                ),
                error,
            )),
        }
    }

    fn open(&mut self, id: JobId, path: PathBuf, mut expected: Header) -> Result<(u64, Header)> {
        if self.records.contains_key(&id) {
            return Err(invalid(format!("disk job {} is already open", id.0)));
        }
        let file = flare_eos::direct_file(&path, false)
            .map_err(|error| direct_open_failure("open record file", &path, error))?;
        #[cfg(windows)]
        self.ring
            .queue
            .attach(&file)
            .map_err(|e| io_failure("attach record reader", e))?;
        let header_span = Span {
            address: self.scratch.address(),
            offset: 0,
            length: HEADER_BYTES,
        };
        let mut bytes = self.ring.transfer(&file, &[header_span], false)?;
        let header = Header::decode(&self.scratch.as_slice()[..HEADER_BYTES])
            .map_err(|error| invalid(format!("record file {}: {error}", path.display())))?;
        check_tail_pages(&header)?;
        // The writer fills this checksum after reserving the indexed header.
        expected.row_checksum = header.row_checksum;
        if header != expected {
            return Err(invalid(format!(
                "record file {} has a different header than the indexed record",
                path.display()
            )));
        }
        let checksum_bytes = header.checksum_bytes() as usize;
        if checksum_bytes > 0 {
            let span = Span {
                address: self.scratch.address() + HEADER_BYTES,
                offset: HEADER_BYTES as u64,
                length: checksum_bytes,
            };
            bytes += self.ring.transfer(&file, &[span], false)?;
        }
        let segment = &self.scratch.as_slice()[HEADER_BYTES..HEADER_BYTES + checksum_bytes];
        let checksums = segment
            .chunks_exact(8)
            .take(header.tail_pages as usize)
            .map(|slot| u64::from_le_bytes(slot.try_into().unwrap()))
            .collect();
        self.records.insert(
            id,
            Record::Reading(Reading {
                file,
                path,
                header: header.clone(),
                checksums,
            }),
        );
        Ok((bytes, header))
    }

    fn read_pages(&mut self, id: JobId, first: u32, buffers: &[usize]) -> Result<u64> {
        let reading = reading(&mut self.records, id)?;
        let page_bytes = reading.header.page_record_bytes();
        let pages = page_range(&reading.header, first, buffers.len())?;
        let spans: Vec<Span> = pages
            .clone()
            .zip(buffers)
            .map(|(page, address)| Span {
                address: *address,
                offset: reading.header.page_segment + page as u64 * page_bytes,
                length: page_bytes as usize,
            })
            .collect();
        let moved = self.ring.transfer(&reading.file, &spans, false)?;
        for (page, address) in pages.zip(buffers) {
            // SAFETY: The caller keeps the slab at `address` valid for
            // `page_bytes` until this job's reply, and the read into it completed.
            let bytes =
                unsafe { std::slice::from_raw_parts(*address as *const u8, page_bytes as usize) };
            if checksum(bytes) != reading.checksums[page] {
                return Err(invalid(format!(
                    "page record {page} of {} failed checksum verification",
                    reading.path.display()
                )));
            }
        }
        Ok(moved)
    }

    fn read_row(&mut self, id: JobId, buffer: usize) -> Result<u64> {
        let reading = reading(&mut self.records, id)?;
        let row_bytes = reading.header.row_record_bytes();
        let span = Span {
            address: buffer,
            offset: reading.header.row_segment,
            length: row_bytes as usize,
        };
        let moved = self.ring.transfer(&reading.file, &[span], false)?;
        // SAFETY: As for page buffers.
        let bytes = unsafe { std::slice::from_raw_parts(buffer as *const u8, row_bytes as usize) };
        if checksum(bytes) != reading.header.row_checksum {
            return Err(invalid(format!(
                "row record of {} failed checksum verification",
                reading.path.display()
            )));
        }
        Ok(moved)
    }

    fn close(&mut self, id: JobId) -> Result<u64> {
        reading(&mut self.records, id)?;
        self.records.remove(&id);
        Ok(0)
    }
}

fn reply(id: JobId, operation: Operation, window: u32, outcome: Result<u64>) -> Reply {
    let (result, bytes) = match outcome {
        Ok(bytes) => (Ok(()), bytes),
        Err(error) => (Err(error), 0),
    };
    Reply {
        id,
        operation,
        window,
        result,
        bytes,
        elapsed: std::time::Duration::ZERO,
        header: None,
    }
}

fn writing(open: &mut HashMap<JobId, Record>, id: JobId) -> Result<&mut Writing> {
    match open.get_mut(&id) {
        Some(Record::Writing(writing)) => Ok(writing),
        Some(Record::Reading(_)) => Err(invalid(format!(
            "disk job {} is a record open for reading, not one being written",
            id.0
        ))),
        None => Err(invalid(format!("disk job {} names no open record", id.0))),
    }
}

fn reading(open: &mut HashMap<JobId, Record>, id: JobId) -> Result<&mut Reading> {
    match open.get_mut(&id) {
        Some(Record::Reading(reading)) => Ok(reading),
        Some(Record::Writing(_)) => Err(invalid(format!(
            "disk job {} is a record being written; commit or abort it instead",
            id.0
        ))),
        None => Err(invalid(format!("disk job {} names no open record", id.0))),
    }
}

fn check_tail_pages(header: &Header) -> Result<()> {
    if header.tail_pages > MAX_TAIL_PAGES {
        return Err(invalid(format!(
            "record has {} tail pages; the disk thread handles at most {MAX_TAIL_PAGES}",
            header.tail_pages
        )));
    }
    Ok(())
}

fn page_range(header: &Header, first: u32, count: usize) -> Result<std::ops::Range<usize>> {
    let end = (first as usize)
        .checked_add(count)
        .filter(|end| *end <= header.tail_pages as usize)
        .ok_or_else(|| {
            invalid(format!(
                "page records {first}..{} lie outside the record's {} tail pages",
                first as u64 + count as u64,
                header.tail_pages
            ))
        })?;
    Ok(first as usize..end)
}

fn direct_open_failure(action: &str, path: &Path, error: io::Error) -> Error {
    if error.raw_os_error() == Some(libc::EINVAL) {
        return invalid(format!(
            "{action} {}: the filesystem refuses direct I/O (O_DIRECT)",
            path.display()
        ));
    }
    io_failure(format!("{action} {}", path.display()), error)
}

fn truncate(path: &Path, length: u64) -> Result<u64> {
    OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.set_len(length))
        .map(|()| 0)
        .map_err(|error| {
            io_failure(
                format!("truncate record file {} to {length} bytes", path.display()),
                error,
            )
        })
}

fn unlink(path: &Path) -> Result<u64> {
    match fs::remove_file(path) {
        Ok(()) => Ok(0),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(io_failure(
            format!("unlink record file {}", path.display()),
            error,
        )),
    }
}

/// One contiguous direct-I/O transfer between a buffer and a file offset.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub address: usize,
    pub offset: u64,
    pub length: usize,
}

/// A span piece of at most [`CHUNK_BYTES`], the unit the ring carries.
#[derive(Clone, Copy, Debug, Default)]
struct Chunk {
    address: usize,
    offset: u64,
    length: u32,
}

struct Chunks<'a> {
    spans: &'a [Span],
    index: usize,
    done: usize,
}

impl Iterator for Chunks<'_> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Chunk> {
        loop {
            let span = self.spans.get(self.index)?;
            if self.done >= span.length {
                self.index += 1;
                self.done = 0;
                continue;
            }
            let length = (span.length - self.done).min(CHUNK_BYTES);
            let chunk = Chunk {
                address: span.address + self.done,
                offset: span.offset + self.done as u64,
                length: length as u32,
            };
            self.done += length;
            return Some(chunk);
        }
    }
}

/// The ring and the per-slot bookkeeping of chunks in flight.
#[cfg(unix)]
struct Ring {
    ring: IoUring,
    slots: Vec<Chunk>,
    free: Vec<usize>,
    /// Set when `io_uring_enter` failed outright. The ring is not used again:
    /// entries it never consumed stay in the submission queue and their slots
    /// stay reserved until the ring is dropped.
    broken: Option<String>,
}

#[cfg(unix)]
impl Ring {
    fn new() -> Result<Self> {
        let ring = IoUring::new(RING_DEPTH as u32)
            .map_err(|error| io_failure("create the disk thread's io_uring", error))?;
        Ok(Self {
            ring,
            slots: vec![Chunk::default(); RING_DEPTH],
            free: (0..RING_DEPTH).collect(),
            broken: None,
        })
    }

    /// Move every span, keeping up to [`RING_DEPTH`] chunks in flight, and
    /// return the bytes moved. Returns after every chunk the kernel holds has
    /// completed, so the caller's buffers are quiescent whatever the outcome.
    /// An undrainable ring retains the call and its buffers through shutdown.
    fn transfer(&mut self, file: &File, spans: &[Span], write: bool) -> Result<u64> {
        if let Some(reason) = &self.broken {
            return Err(invalid(format!(
                "disk thread ring failed earlier: {reason}"
            )));
        }
        let verb = if write { "write" } else { "read" };
        for span in spans {
            if !aligned(span.address as u64)
                || !aligned(span.offset)
                || !aligned(span.length as u64)
            {
                return Err(invalid(format!(
                    "direct {verb} of {} bytes at file offset {} from address {:#x} is not \
                     {BLOCK_BYTES}-byte aligned",
                    span.length, span.offset, span.address
                )));
            }
        }
        let fd = types::Fd(file.as_raw_fd());
        let mut chunks = Chunks {
            spans,
            index: 0,
            done: 0,
        };
        let mut pending = chunks.next();
        let mut moved = 0u64;
        let mut failure: Option<Error> = None;
        loop {
            while failure.is_none() {
                let Some(chunk) = pending else { break };
                let Some(slot) = self.free.pop() else { break };
                let entry = if write {
                    opcode::Write::new(fd, chunk.address as *const u8, chunk.length)
                        .offset(chunk.offset)
                        .build()
                } else {
                    opcode::Read::new(fd, chunk.address as *mut u8, chunk.length)
                        .offset(chunk.offset)
                        .build()
                }
                .user_data(slot as u64);
                // SAFETY: The buffer stays valid until this function returns,
                // which happens only after the chunk's completion is consumed
                // or `abandon` established that the kernel never saw the entry.
                if unsafe { self.ring.submission().push(&entry) }.is_err() {
                    self.free.push(slot);
                    // The queue holds one entry per slot, so a free slot
                    // always has room; a refused push is a bookkeeping fault.
                    failure.get_or_insert_with(|| {
                        invalid(format!(
                            "disk thread ring refused a submission entry with {} free slots",
                            self.free.len()
                        ))
                    });
                    break;
                }
                self.slots[slot] = chunk;
                pending = chunks.next();
            }
            if self.free.len() == RING_DEPTH && (pending.is_none() || failure.is_some()) {
                break;
            }
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(error) if retryable(&error) => std::thread::yield_now(),
                Err(error) => return Err(self.abandon(verb, error)),
            }
            for completion in self.ring.completion() {
                let slot = completion.user_data() as usize;
                let chunk = self.slots[slot];
                self.free.push(slot);
                let result = completion.result();
                if result < 0 {
                    failure.get_or_insert_with(|| {
                        io_failure(
                            format!(
                                "direct {verb} of {} bytes at file offset {}",
                                chunk.length, chunk.offset
                            ),
                            io::Error::from_raw_os_error(-result),
                        )
                    });
                } else if result as u32 != chunk.length {
                    failure.get_or_insert_with(|| {
                        invalid(format!(
                            "short direct {verb} of {result} bytes at file offset {}; {} were requested",
                            chunk.offset, chunk.length
                        ))
                    });
                } else {
                    moved += chunk.length as u64;
                }
            }
        }
        match failure {
            None => Ok(moved),
            Some(error) => Err(error),
        }
    }

    /// `io_uring_enter` failed outright, so it consumed no entries: the ones
    /// still in the submission queue were never seen by the kernel, while the
    /// others may still be in flight. Wait for every consumed entry even if
    /// waiting fails, so a reply always means the caller's buffers are quiescent.
    /// Shutdown retains this worker and its buffers when completion cannot be confirmed.
    fn abandon(&mut self, verb: &str, error: io::Error) -> Error {
        let reason = format!("submit direct {verb}: {error}");
        self.broken = Some(reason.clone());
        let unconsumed = self.ring.submission().len();
        let held = (RING_DEPTH - self.free.len()).saturating_sub(unconsumed);
        drain_completions(held, |pending| {
            let completed = self.ring.completion().count();
            if completed >= pending {
                return (completed, Ok(()));
            }
            // SAFETY: Nothing is submitted and no argument is passed; the call
            // only waits for one completion.
            let waited = unsafe {
                self.ring.submitter().enter::<libc::sigset_t>(
                    0,
                    1,
                    EnterFlags::GETEVENTS.bits(),
                    None,
                )
            };
            (completed, waited.map(|_| ()))
        });
        invalid(reason)
    }
}

#[cfg(windows)]
struct Ring {
    queue: flare_eos::completion::Queue,
    slots: Vec<Chunk>,
    free: Vec<usize>,
}

#[cfg(windows)]
impl Ring {
    fn new() -> Result<Self> {
        Ok(Self {
            queue: flare_eos::completion::Queue::new(RING_DEPTH)
                .map_err(|e| io_failure("create disk completion port", e))?,
            slots: vec![Chunk::default(); RING_DEPTH],
            free: (0..RING_DEPTH).collect(),
        })
    }

    fn transfer(&mut self, file: &File, spans: &[Span], write: bool) -> Result<u64> {
        for span in spans {
            if !aligned(span.address as u64)
                || !aligned(span.offset)
                || !aligned(span.length as u64)
            {
                return Err(invalid("direct I/O span must be 4096-byte aligned"));
            }
        }
        let mut chunks = Chunks {
            spans,
            index: 0,
            done: 0,
        };
        let mut pending = chunks.next();
        let mut moved = 0;
        let mut failure = None;
        loop {
            while failure.is_none() {
                let Some(chunk) = pending else { break };
                let Some(slot) = self.free.pop() else { break };
                match unsafe {
                    self.queue
                        .submit(slot, file, chunk.address, chunk.length, chunk.offset, write)
                } {
                    Ok(()) => {
                        self.slots[slot] = chunk;
                        pending = chunks.next();
                    }
                    Err(error) => {
                        self.free.push(slot);
                        failure = Some(Self::transfer_error(
                            "submit direct I/O",
                            error,
                            chunk,
                            write,
                        ));
                        self.queue.cancel();
                    }
                }
            }
            if self.queue.pending() == 0 {
                break;
            }
            match self.queue.wait() {
                Ok((slot, result)) => {
                    self.free.push(slot);
                    let chunk = self.slots[slot];
                    match result {
                        Ok(n) if n == chunk.length as usize => moved += n as u64,
                        Ok(n) => {
                            failure.get_or_insert_with(|| {
                                let verb = if write { "write" } else { "read" };
                                invalid(format!(
                                    "short direct {verb} of {n} bytes at file offset {}; {} were requested",
                                    chunk.offset, chunk.length
                                ))
                            });
                        }
                        Err(error) => {
                            failure.get_or_insert_with(|| {
                                Self::transfer_error("complete direct I/O", error, chunk, write)
                            });
                        }
                    }
                    if failure.is_some() {
                        self.queue.cancel();
                    }
                }
                Err(error) => {
                    failure.get_or_insert_with(|| io_failure("wait for direct I/O", error));
                    self.queue.cancel();
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(moved),
        }
    }

    fn transfer_error(action: &str, error: io::Error, chunk: Chunk, write: bool) -> crate::Error {
        if !write
            && error.raw_os_error() == Some(windows_sys::Win32::Foundation::ERROR_HANDLE_EOF as i32)
        {
            invalid(format!(
                "short direct read of 0 bytes at file offset {}; {} were requested",
                chunk.offset, chunk.length
            ))
        } else {
            io_failure(action, error)
        }
    }
}

#[cfg(any(unix, test))]
fn drain_completions(mut pending: usize, mut wait: impl FnMut(usize) -> (usize, io::Result<()>)) {
    while pending > 0 {
        let (completed, waited) = wait(pending);
        pending = pending.saturating_sub(completed);
        if let Err(error) = waited {
            if retryable(&error) {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

/// `io_uring_enter` errors after which the call is repeated: an interrupting
/// signal, or the kernel asking for a retry because a request could not be
/// allocated or the completion queue has overflowed.
#[cfg(any(unix, test))]
fn retryable(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINTR | libc::EAGAIN | libc::EBUSY)
    )
}

/// Heap memory aligned to [`BLOCK_BYTES`], zero-filled, usable as a direct-I/O buffer.
struct AlignedBuffer {
    address: usize,
    length: usize,
}

// SAFETY: The buffer is plain process memory owned by whoever holds the value.
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(length: usize) -> Result<Self> {
        let pointer = flare_eos::allocate_aligned(length)
            .map_err(|e| io_failure(format!("allocate {length} aligned bytes"), e))?
            as *mut u8;
        // SAFETY: The allocation holds `length` bytes.
        unsafe { std::ptr::write_bytes(pointer as *mut u8, 0, length) };
        Ok(Self {
            address: pointer as usize,
            length,
        })
    }

    fn address(&self) -> usize {
        self.address
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: The allocation lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.address as *const u8, self.length) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: As above, with exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.address as *mut u8, self.length) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: The pointer came from `allocate_aligned` and is freed once.
        unsafe { flare_eos::free_aligned(self.address, self.length) };
    }
}

/// Bytes of one block a [`ReadAhead`] reads at a time.
const AHEAD_BLOCK_BYTES: usize = 4 << 20;
/// Blocks one ring transfer carries, and so the reads the device has in flight.
const AHEAD_BATCH: usize = 4;
/// Blocks a [`ReadAhead`] owns: the batch being read and the ones waiting to be consumed.
const AHEAD_BLOCKS: usize = 3 * AHEAD_BATCH;

/// A block read ahead: where in the file it came from and how much of the buffer is valid.
struct AheadBlock {
    buffer: AlignedBuffer,
    offset: u64,
    length: usize,
}

/// Sequential reads of one file range past the page cache, with the device kept busy: a
/// thread reads the range ahead of the consumer in [`AHEAD_BLOCK_BYTES`] blocks, several in
/// flight through the same ring the disk thread moves records with, while the consumer copies
/// what was read out. A consumer that reads the range window by window, each read waiting for
/// the device, leaves the device idle between windows and keeps one request in flight; this
/// keeps [`AHEAD_BATCH`] and is not slowed by the consumer's own work.
pub struct ReadAhead {
    blocks: Receiver<Result<AheadBlock>>,
    free: Option<mpsc::Sender<AlignedBuffer>>,
    thread: Option<std::thread::JoinHandle<()>>,
    current: Option<AheadBlock>,
    /// The file offset the next byte handed out comes from.
    next: u64,
    end: u64,
}

impl ReadAhead {
    /// Read `start..end` of `file`, which must have been opened for direct I/O. `start` and
    /// `end` are multiples of [`BLOCK_BYTES`].
    pub fn start(file: File, start: u64, end: u64) -> Result<Self> {
        if !aligned(start) || !aligned(end) || end < start {
            return Err(invalid("read-ahead range is not block aligned"));
        }
        let ring = Ring::new()?;
        let (free, recycled) = mpsc::channel::<AlignedBuffer>();
        let (sender, blocks) = mpsc::channel::<Result<AheadBlock>>();
        let count = AHEAD_BLOCKS.min(((end - start) as usize).div_ceil(AHEAD_BLOCK_BYTES).max(1));
        for _ in 0..count {
            let _ = free.send(AlignedBuffer::new(AHEAD_BLOCK_BYTES)?);
        }
        let thread = std::thread::Builder::new()
            .name("selene-read-ahead".into())
            .spawn(move || read_ahead(ring, file, start, end, recycled, sender))
            .map_err(|error| io_failure("spawn the read-ahead thread", error))?;
        Ok(Self {
            blocks,
            free: Some(free),
            thread: Some(thread),
            current: None,
            next: start,
            end,
        })
    }

    /// Whether the next byte this hands out is the one at `offset`.
    pub fn at(&self, offset: u64) -> bool {
        self.next == offset
    }

    /// Copy the next `dst.len()` bytes of the range into `dst`.
    pub fn read(&mut self, dst: &mut [u8]) -> Result<()> {
        if self.next + dst.len() as u64 > self.end {
            return Err(invalid("read-ahead read runs past its range"));
        }
        let mut done = 0;
        while done < dst.len() {
            if self.current.is_none() {
                self.current = Some(
                    self.blocks
                        .recv()
                        .map_err(|_| invalid("the read-ahead thread stopped"))??,
                );
            }
            let block = self.current.as_ref().expect("a block");
            let at = (self.next - block.offset) as usize;
            let take = (block.length - at).min(dst.len() - done);
            dst[done..done + take].copy_from_slice(&block.buffer.as_slice()[at..at + take]);
            done += take;
            self.next += take as u64;
            if at + take == block.length {
                let block = self.current.take().expect("a block");
                if let Some(free) = &self.free {
                    let _ = free.send(block.buffer);
                }
            }
        }
        Ok(())
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        // The thread waits for buffers; with the sender gone it ends after the transfer it
        // may be in the middle of, which is the last thing to touch its buffers.
        self.free.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn read_ahead(
    mut ring: Ring,
    file: File,
    start: u64,
    end: u64,
    recycled: Receiver<AlignedBuffer>,
    blocks: mpsc::Sender<Result<AheadBlock>>,
) {
    let mut offset = start;
    while offset < end {
        // Take what buffers are free, at least one, up to a batch.
        let Ok(first) = recycled.recv() else { return };
        let mut batch = vec![first];
        while batch.len() < AHEAD_BATCH {
            match recycled.try_recv() {
                Ok(buffer) => batch.push(buffer),
                Err(_) => break,
            }
        }
        let mut spans = Vec::with_capacity(batch.len());
        let mut lengths = Vec::with_capacity(batch.len());
        let mut at = offset;
        for buffer in &batch {
            let length = (end - at).min(AHEAD_BLOCK_BYTES as u64) as usize;
            if length == 0 {
                break;
            }
            spans.push(Span {
                address: buffer.address(),
                offset: at,
                length,
            });
            lengths.push((at, length));
            at += length as u64;
        }
        batch.truncate(spans.len());
        let result = ring.transfer(&file, &spans, false);
        match result {
            Ok(_) => {
                for (buffer, (block_offset, length)) in batch.into_iter().zip(lengths) {
                    let block = AheadBlock {
                        buffer,
                        offset: block_offset,
                        length,
                    };
                    if blocks.send(Ok(block)).is_err() {
                        return;
                    }
                }
            }
            Err(error) => {
                let _ = blocks.send(Err(error));
                return;
            }
        }
        offset = at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PrefixKey,
        record::{file_name, temporary_name},
        testing::geometry,
    };
    use std::{
        io::{Seek, SeekFrom, Write as _},
        time::{Duration, Instant, SystemTime},
    };

    #[cfg(unix)]
    #[test]
    fn record_reads_finish_while_writes_are_blocked() {
        let Some(directory) = test_directory("record-priority") else {
            return;
        };
        let (gate, entered, release) = IoGate::new();
        let writes = DiskThread::start_with_faults(
            "record-write",
            Faults {
                io_gate: Some(gate),
                ..Faults::default()
            },
        )
        .unwrap();
        let reads = DiskThread::start("record-read").unwrap();
        let mut buffer = AlignedBuffer::new(4096).unwrap();
        buffer.as_mut_slice().fill(7);
        for id in 0..8 {
            writes
                .submit(Job::write_record(
                    JobId(id),
                    directory.join(format!("{id}.tmp")),
                    0,
                    buffer.as_slice(),
                    None,
                ))
                .unwrap();
            if id == 0 {
                entered.recv_timeout(Duration::from_secs(5)).unwrap();
            }
        }
        let source = directory.join("source");
        fs::write(&source, vec![9; 4096]).unwrap();
        let mut output = AlignedBuffer::new(4096).unwrap();
        reads
            .submit(Job::ReadRecord {
                id: JobId(99),
                file: Arc::new(flare_eos::direct_file(&source, false).unwrap()),
                span: Span {
                    address: output.address(),
                    offset: 0,
                    length: 4096,
                },
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut replies = Vec::new();
        while replies.is_empty() && Instant::now() < deadline {
            reads.poll(&mut replies);
            std::thread::sleep(Duration::from_millis(1));
        }
        release.send(()).unwrap();
        assert_eq!(replies.len(), 1, "read waited for persistence");
        replies.pop().unwrap().result.unwrap();
        assert_eq!(output.as_mut_slice(), &[9; 4096]);
        for reply in wait(&writes, 8) {
            reply.result.unwrap();
        }
        drop((writes, reads));
        fs::remove_dir_all(directory).unwrap();
    }

    /// A fresh directory for one test under `FLARE_OFFLOAD_TEST_DIR` or the
    /// system temporary directory, or `None` when it refuses direct I/O.
    fn test_directory(test: &str) -> Option<PathBuf> {
        let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = base.join(format!("flare-offload-disk-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let probe = directory.join(".probe");
        match flare_eos::direct_file(&probe, true) {
            Ok(_) => {
                fs::remove_file(&probe).unwrap();
                Some(directory)
            }
            Err(error) if cfg!(unix) && error.raw_os_error() == Some(libc::EINVAL) => {
                eprintln!(
                    "skipping {test}: {} refuses direct I/O",
                    directory.display()
                );
                fs::remove_dir_all(&directory).unwrap();
                None
            }
            Err(error) => panic!("probe {}: {error}", probe.display()),
        }
    }

    fn wait(thread: &DiskThread, count: usize) -> Vec<Reply> {
        let started = Instant::now();
        let mut replies = Vec::new();
        while replies.len() < count {
            thread.poll(&mut replies);
            if replies.len() >= count {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "{} of {count} replies after 60 s",
                replies.len()
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(replies.len(), count);
        replies
    }

    fn succeeded(reply: &Reply) -> u64 {
        if let Err(error) = &reply.result {
            panic!("{:?} failed: {error}", reply.operation);
        }
        reply.bytes
    }

    fn failed(reply: &Reply) -> String {
        match &reply.result {
            Ok(()) => panic!("{:?} succeeded unexpectedly", reply.operation),
            Err(error) => error.to_string(),
        }
    }

    fn buffer(length: usize, seed: u8) -> AlignedBuffer {
        let mut buffer = AlignedBuffer::new(length).unwrap();
        for (index, byte) in buffer.as_mut_slice().iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(31).wrapping_add(seed);
        }
        buffer
    }

    fn corrupt(path: &Path, offset: u64) {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut byte = [0u8; 1];
        std::io::Read::read_exact(&mut file, &mut byte).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte[0] ^ 0x5a]).unwrap();
    }

    fn header(key: PrefixKey, tail_pages: u32) -> Header {
        let geometry = [geometry(
            64,
            &[("k", 8192), ("v", 8192), ("sel", 100)],
            10_000,
        )];
        Header::new(
            &geometry,
            [1; 32],
            [2; 16],
            key,
            None,
            [3; 16],
            tail_pages,
            SystemTime::now(),
        )
        .unwrap()
    }

    #[test]
    fn record_round_trips_through_direct_io() {
        let Some(directory) = test_directory("round-trip") else {
            return;
        };
        let key = PrefixKey {
            consumed_tokens: 200,
            digest: [7; 16],
        };
        let header = header(key, 3);
        let page_bytes = header.page_record_bytes() as usize;
        let row_bytes = header.row_record_bytes() as usize;
        assert_eq!(page_bytes, 8192 + 8192 + 4096);
        assert_eq!(row_bytes, 12288);
        let pages: Vec<AlignedBuffer> = (0..3u8)
            .map(|index| buffer(page_bytes, index * 17 + 1))
            .collect();
        let row = buffer(row_bytes, 99);
        let temporary = directory.join(temporary_name(&key));
        let path = directory.join(file_name(&key));

        let mut thread = DiskThread::start("flare-offload-disk").unwrap();
        let writer = JobId(1);
        thread
            .submit(Job::Create {
                id: writer,
                temporary: temporary.clone(),
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id: writer,
                window: 0,
                first: 0,
                buffers: vec![pages[0].address(), pages[1].address()],
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id: writer,
                window: 1,
                first: 2,
                buffers: vec![pages[2].address()],
            })
            .unwrap();
        thread
            .submit(Job::WriteRow {
                id: writer,
                buffer: row.address(),
            })
            .unwrap();
        thread
            .submit(Job::Commit {
                id: writer,
                path: path.clone(),
                sync: true,
            })
            .unwrap();
        let replies = wait(&thread, 5);
        let operations: Vec<_> = replies.iter().map(|reply| reply.operation).collect();
        assert_eq!(
            operations,
            [
                Operation::Create,
                Operation::WritePages,
                Operation::WritePages,
                Operation::WriteRow,
                Operation::Commit
            ]
        );
        assert!(replies.iter().all(|reply| reply.id == writer));
        assert_eq!(succeeded(&replies[0]), 0);
        assert_eq!(succeeded(&replies[1]), 2 * page_bytes as u64);
        assert_eq!(replies[1].window, 0);
        assert_eq!(succeeded(&replies[2]), page_bytes as u64);
        assert_eq!(replies[2].window, 1);
        assert_eq!(succeeded(&replies[3]), row_bytes as u64);
        assert_eq!(
            succeeded(&replies[4]),
            3 * page_bytes as u64 + row_bytes as u64 + 2 * HEADER_BYTES as u64
        );
        assert!(!temporary.exists());
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());

        let contents = fs::read(&path).unwrap();
        let mut expected = header.clone();
        expected.row_checksum = checksum(row.as_slice());
        assert_eq!(Header::decode(&contents[..HEADER_BYTES]).unwrap(), expected);
        let page_segment = header.page_segment as usize;
        for (index, page) in pages.iter().enumerate() {
            let slot = HEADER_BYTES + index * 8;
            assert_eq!(
                u64::from_le_bytes(contents[slot..slot + 8].try_into().unwrap()),
                checksum(page.as_slice())
            );
            let start = page_segment + index * page_bytes;
            assert_eq!(&contents[start..start + page_bytes], page.as_slice());
        }
        let row_segment = header.row_segment as usize;
        assert_eq!(
            &contents[row_segment..row_segment + row_bytes],
            row.as_slice()
        );
        assert!(
            contents[HEADER_BYTES + 24..page_segment]
                .iter()
                .all(|b| *b == 0)
        );

        let reader = JobId(2);
        let fresh: Vec<AlignedBuffer> = (0..2).map(|_| buffer(page_bytes, 0xee)).collect();
        let fresh_row = buffer(row_bytes, 0xee);
        thread
            .submit(Job::Open {
                id: reader,
                path: path.clone(),
                expected: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 3,
                first: 1,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        thread
            .submit(Job::ReadRow {
                id: reader,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 4,
                first: 2,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        let replies = wait(&thread, 4);
        assert_eq!(replies[0].operation, Operation::Open);
        assert_eq!(succeeded(&replies[0]), 2 * HEADER_BYTES as u64);
        assert_eq!(replies[0].header, Some(expected.clone()));
        assert_eq!(succeeded(&replies[1]), 2 * page_bytes as u64);
        assert_eq!(replies[1].window, 3);
        assert_eq!(fresh[0].as_slice(), pages[1].as_slice());
        assert_eq!(fresh[1].as_slice(), pages[2].as_slice());
        assert_eq!(succeeded(&replies[2]), row_bytes as u64);
        assert_eq!(fresh_row.as_slice(), row.as_slice());
        assert!(failed(&replies[3]).contains("outside the record's 3 tail pages"));
        assert_eq!(replies[3].window, 4);

        corrupt(&path, header.page_segment + 2 * page_bytes as u64 + 17);
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 0,
                first: 1,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 0,
                first: 0,
                buffers: vec![fresh[0].address(), fresh[1].address()],
            })
            .unwrap();
        let replies = wait(&thread, 2);
        assert!(failed(&replies[0]).contains("page record 2"));
        succeeded(&replies[1]);
        assert_eq!(fresh[0].as_slice(), pages[0].as_slice());
        assert_eq!(fresh[1].as_slice(), pages[1].as_slice());

        corrupt(&path, header.row_segment + 4096 + 1);
        thread
            .submit(Job::ReadRow {
                id: reader,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread.submit(Job::Close { id: reader }).unwrap();
        thread.submit(Job::Close { id: reader }).unwrap();
        let replies = wait(&thread, 3);
        assert!(failed(&replies[0]).contains("row record"));
        succeeded(&replies[1]);
        assert!(failed(&replies[2]).contains("names no open record"));

        thread
            .submit(Job::Truncate {
                id: JobId(3),
                path: path.clone(),
                length: header.row_segment,
            })
            .unwrap();
        let replies = wait(&thread, 1);
        assert_eq!(replies[0].operation, Operation::Truncate);
        succeeded(&replies[0]);
        let length = fs::metadata(&path).unwrap().len();
        assert_eq!(length, header.row_segment);
        assert!(!header.row_present(length));

        let reopened = JobId(4);
        thread
            .submit(Job::Open {
                id: reopened,
                path: path.clone(),
                expected: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reopened,
                window: 0,
                first: 0,
                buffers: vec![fresh[0].address()],
            })
            .unwrap();
        thread
            .submit(Job::ReadRow {
                id: reopened,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread.submit(Job::Close { id: reopened }).unwrap();
        thread
            .submit(Job::Unlink {
                id: JobId(5),
                path: path.clone(),
            })
            .unwrap();
        let replies = wait(&thread, 5);
        assert_eq!(replies[0].header, Some(expected));
        succeeded(&replies[1]);
        assert_eq!(fresh[0].as_slice(), pages[0].as_slice());
        let message = failed(&replies[2]);
        assert!(message.contains("short direct read"), "{message}");
        succeeded(&replies[3]);
        assert_eq!(replies[4].operation, Operation::Unlink);
        succeeded(&replies[4]);
        assert!(!path.exists());

        thread
            .submit(Job::Unlink {
                id: JobId(6),
                path: path.clone(),
            })
            .unwrap();
        thread.close().unwrap();
        assert!(thread.submit(Job::Close { id: JobId(6) }).is_err());
        let replies = wait(&thread, 1);
        assert_eq!(replies[0].operation, Operation::Unlink);
        succeeded(&replies[0]);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn failed_writes_abort_and_exit_leave_no_record_behind() {
        let Some(directory) = test_directory("abort") else {
            return;
        };
        let key = PrefixKey {
            consumed_tokens: 192,
            digest: [8; 16],
        };
        let header = header(key, 3);
        let page = buffer(header.page_record_bytes() as usize, 5);
        let row = buffer(header.row_record_bytes() as usize, 6);
        let temporary = directory.join(temporary_name(&key));
        let path = directory.join(file_name(&key));
        let thread = DiskThread::start("flare-offload-disk").unwrap();

        // A failed page write leaves its page unwritten, so a commit queued
        // behind it is refused and removes the temporary file.
        let id = JobId(1);
        thread
            .submit(Job::Create {
                id,
                temporary: temporary.clone(),
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 1,
                first: 1,
                buffers: vec![page.address() + 1],
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 2,
                first: 2,
                buffers: vec![page.address()],
            })
            .unwrap();
        thread
            .submit(Job::WriteRow {
                id,
                buffer: row.address(),
            })
            .unwrap();
        thread
            .submit(Job::Commit {
                id,
                path: path.clone(),
                sync: false,
            })
            .unwrap();
        let replies = wait(&thread, 6);
        succeeded(&replies[0]);
        succeeded(&replies[1]);
        assert!(failed(&replies[2]).contains("aligned"));
        assert_eq!(replies[2].window, 1);
        succeeded(&replies[3]);
        succeeded(&replies[4]);
        let message = failed(&replies[5]);
        assert!(message.contains("1 of 3 page records"), "{message}");
        assert!(!temporary.exists());
        assert!(!path.exists());

        thread
            .submit(Job::Create {
                id,
                temporary: temporary.clone(),
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        let replies = wait(&thread, 2);
        succeeded(&replies[0]);
        succeeded(&replies[1]);
        assert!(temporary.exists());
        thread
            .submit(Job::WritePages {
                id: JobId(9),
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        thread.submit(Job::Close { id }).unwrap();
        thread.submit(Job::Abort { id }).unwrap();
        thread.submit(Job::Abort { id }).unwrap();
        let replies = wait(&thread, 4);
        assert!(failed(&replies[0]).contains("names no open record"));
        assert!(failed(&replies[1]).contains("being written"));
        succeeded(&replies[2]);
        assert!(!temporary.exists());
        assert!(failed(&replies[3]).contains("names no open record"));

        thread
            .submit(Job::Create {
                id,
                temporary: temporary.clone(),
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id,
                window: 0,
                first: 0,
                buffers: vec![page.address()],
            })
            .unwrap();
        thread
            .submit(Job::Commit {
                id,
                path: path.clone(),
                sync: false,
            })
            .unwrap();
        let replies = wait(&thread, 3);
        succeeded(&replies[0]);
        succeeded(&replies[1]);
        let message = failed(&replies[2]);
        assert!(message.contains("2 of 3 page records"), "{message}");
        assert!(message.contains("its row"), "{message}");
        assert!(!temporary.exists());
        assert!(!path.exists());

        // Dropping the thread with a record still being written unlinks its
        // temporary file.
        thread
            .submit(Job::Create {
                id,
                temporary: temporary.clone(),
                header,
            })
            .unwrap();
        succeeded(&wait(&thread, 1)[0]);
        assert!(temporary.exists());
        drop(thread);
        assert!(!temporary.exists());
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn transfers_cross_the_chunk_size_and_the_ring_depth() {
        let Some(directory) = test_directory("chunked") else {
            return;
        };
        // One page record per ring slot plus seven, so one write exhausts the
        // ring's slots, and a row of three chunks, two of them full.
        let tail = RING_DEPTH as u32 + 7;
        let geometry = [geometry(64, &[("k", 4096)], 2 * CHUNK_BYTES as u64 + 4096)];
        let key = PrefixKey {
            consumed_tokens: 64 * tail as u64,
            digest: [9; 16],
        };
        let header = Header::new(
            &geometry,
            [1; 32],
            [2; 16],
            key,
            None,
            [3; 16],
            tail,
            SystemTime::now(),
        )
        .unwrap();
        let page_bytes = header.page_record_bytes() as usize;
        let row_bytes = header.row_record_bytes() as usize;
        assert_eq!(page_bytes, 4096);
        assert_eq!(row_bytes, 2 * CHUNK_BYTES + 4096);
        let pages: Vec<AlignedBuffer> = (0..tail)
            .map(|index| buffer(page_bytes, index as u8))
            .collect();
        let row = buffer(row_bytes, 77);
        let temporary = directory.join(temporary_name(&key));
        let path = directory.join(file_name(&key));

        let mut thread = DiskThread::start("flare-offload-disk").unwrap();
        let writer = JobId(1);
        thread
            .submit(Job::Create {
                id: writer,
                temporary,
                header: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::WritePages {
                id: writer,
                window: 0,
                first: 0,
                buffers: pages.iter().map(AlignedBuffer::address).collect(),
            })
            .unwrap();
        thread
            .submit(Job::WriteRow {
                id: writer,
                buffer: row.address(),
            })
            .unwrap();
        thread
            .submit(Job::Commit {
                id: writer,
                path: path.clone(),
                sync: false,
            })
            .unwrap();
        let replies = wait(&thread, 4);
        succeeded(&replies[0]);
        assert_eq!(succeeded(&replies[1]), tail as u64 * page_bytes as u64);
        assert_eq!(succeeded(&replies[2]), row_bytes as u64);
        assert_eq!(
            succeeded(&replies[3]),
            tail as u64 * page_bytes as u64 + row_bytes as u64 + 2 * HEADER_BYTES as u64
        );
        assert_eq!(fs::metadata(&path).unwrap().len(), header.length_with_row());

        let reader = JobId(2);
        let fresh: Vec<AlignedBuffer> = (0..tail).map(|_| buffer(page_bytes, 0xee)).collect();
        let fresh_row = buffer(row_bytes, 0xee);
        thread
            .submit(Job::Open {
                id: reader,
                path: path.clone(),
                expected: header.clone(),
            })
            .unwrap();
        thread
            .submit(Job::ReadPages {
                id: reader,
                window: 0,
                first: 0,
                buffers: fresh.iter().map(AlignedBuffer::address).collect(),
            })
            .unwrap();
        thread
            .submit(Job::ReadRow {
                id: reader,
                buffer: fresh_row.address(),
            })
            .unwrap();
        thread.submit(Job::Close { id: reader }).unwrap();
        thread
            .submit(Job::Unlink {
                id: JobId(3),
                path: path.clone(),
            })
            .unwrap();
        let replies = wait(&thread, 5);
        assert_eq!(succeeded(&replies[0]), 2 * HEADER_BYTES as u64);
        assert_eq!(replies[0].header.as_ref().unwrap().tail_pages, tail);
        assert_eq!(succeeded(&replies[1]), tail as u64 * page_bytes as u64);
        assert_eq!(succeeded(&replies[2]), row_bytes as u64);
        succeeded(&replies[3]);
        succeeded(&replies[4]);
        for (copy, page) in fresh.iter().zip(&pages) {
            assert_eq!(copy.as_slice(), page.as_slice());
        }
        assert_eq!(fresh_row.as_slice(), row.as_slice());
        assert!(!path.exists());
        thread.close().unwrap();
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn mismatched_headers_reject_queued_reads_without_touching_destinations() {
        let Some(directory) = test_directory("header-contract") else {
            return;
        };
        let expected = header(
            PrefixKey {
                consumed_tokens: 200,
                digest: [7; 16],
            },
            3,
        );
        let mut variants = Vec::new();
        let mut changed = expected.clone();
        changed.key.digest[0] ^= 1;
        variants.push(("key", changed));
        let mut changed = expected.clone();
        changed.identity[0] ^= 1;
        variants.push(("identity", changed));
        let mut changed = expected.clone();
        changed.fingerprint[0] ^= 1;
        variants.push(("fingerprint", changed));
        let mut changed = expected.clone();
        changed.bucket[0] ^= 1;
        variants.push(("bucket", changed));
        let mut changed = expected.clone();
        changed.created += 1;
        variants.push(("created", changed));
        let mut changed = expected.clone();
        changed.ranks[0].planes[2] -= 1;
        variants.push(("geometry-with-the-same-padded-size", changed));
        let mut changed = expected.clone();
        changed.ranks[0].planes[0] += BLOCK_BYTES;
        changed.row_segment += u64::from(changed.tail_pages) * BLOCK_BYTES;
        variants.push(("larger-page", changed));
        let mut changed = expected.clone();
        changed.ranks[0].row_bytes += BLOCK_BYTES;
        variants.push(("larger-row", changed));

        let mut thread = DiskThread::start("flare-offload-disk").unwrap();
        for (index, (name, changed)) in variants.into_iter().enumerate() {
            let path = directory.join(format!("{name}.ckpt"));
            let mut contents = vec![0x5a; changed.length_with_row() as usize];
            contents[..HEADER_BYTES].copy_from_slice(&changed.encode().unwrap());
            fs::write(&path, contents).unwrap();
            // Extra mapped bytes make an oversized read observable without crossing the allocation.
            let page = buffer(changed.page_record_bytes() as usize + HEADER_BYTES, 0xee);
            let row = buffer(changed.row_record_bytes() as usize + HEADER_BYTES, 0xee);
            let page_before = page.as_slice().to_vec();
            let row_before = row.as_slice().to_vec();
            let id = JobId(index as u64);
            thread
                .submit(Job::Open {
                    id,
                    path,
                    expected: expected.clone(),
                })
                .unwrap();
            thread
                .submit(Job::ReadPages {
                    id,
                    window: 0,
                    first: 0,
                    buffers: vec![page.address()],
                })
                .unwrap();
            thread
                .submit(Job::ReadRow {
                    id,
                    buffer: row.address(),
                })
                .unwrap();
            let replies = wait(&thread, 3);
            assert!(failed(&replies[0]).contains("different header"), "{name}");
            for reply in &replies[1..] {
                assert!(failed(reply).contains("names no open record"), "{name}");
            }
            assert_eq!(
                page.as_slice(),
                page_before,
                "{name} changed page destination"
            );
            assert_eq!(row.as_slice(), row_before, "{name} changed row destination");
        }
        thread.close().unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn open_and_create_reject_files_and_headers_outside_the_contract() {
        let Some(directory) = test_directory("reject") else {
            return;
        };
        let short = directory.join("short.ckpt");
        fs::write(&short, [7u8; 100]).unwrap();
        let zeroed = directory.join("zeroed.ckpt");
        fs::write(&zeroed, [0u8; HEADER_BYTES]).unwrap();
        let wide_key = PrefixKey {
            consumed_tokens: 64 * (MAX_TAIL_PAGES as u64 + 1),
            digest: [4; 16],
        };
        let wide = header(wide_key, MAX_TAIL_PAGES + 1);
        let wide_path = directory.join(file_name(&wide_key));
        fs::write(&wide_path, wide.encode().unwrap()).unwrap();
        let wide_temporary = directory.join(temporary_name(&wide_key));
        let page = buffer(wide.page_record_bytes() as usize, 1);

        let mut thread = DiskThread::start("flare-offload-disk").unwrap();
        thread
            .submit(Job::Open {
                id: JobId(1),
                path: short,
                expected: wide.clone(),
            })
            .unwrap();
        thread
            .submit(Job::Open {
                id: JobId(2),
                path: zeroed,
                expected: wide.clone(),
            })
            .unwrap();
        thread
            .submit(Job::Open {
                id: JobId(3),
                path: wide_path,
                expected: wide.clone(),
            })
            .unwrap();
        thread
            .submit(Job::Open {
                id: JobId(4),
                path: directory.join("missing.ckpt"),
                expected: wide.clone(),
            })
            .unwrap();
        thread
            .submit(Job::Create {
                id: JobId(5),
                temporary: wide_temporary.clone(),
                header: wide,
            })
            .unwrap();
        for id in 1..=5 {
            thread
                .submit(Job::ReadPages {
                    id: JobId(id),
                    window: 0,
                    first: 0,
                    buffers: vec![page.address()],
                })
                .unwrap();
        }
        let replies = wait(&thread, 10);
        let message = failed(&replies[0]);
        assert!(
            message.contains("short direct read of 100 bytes"),
            "{message}"
        );
        let message = failed(&replies[1]);
        assert!(message.contains("magic"), "{message}");
        let message = failed(&replies[2]);
        assert!(
            message.contains(&format!("at most {MAX_TAIL_PAGES}")),
            "{message}"
        );
        failed(&replies[3]);
        let message = failed(&replies[4]);
        assert!(
            message.contains(&format!("at most {MAX_TAIL_PAGES}")),
            "{message}"
        );
        assert!(!wide_temporary.exists());
        for reply in &replies[5..] {
            assert!(reply.header.is_none());
            assert!(failed(reply).contains("names no open record"));
        }
        thread.close().unwrap();
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn close_timeout_keeps_the_worker_until_its_buffer_references_are_released() {
        let (release, resume) = mpsc::channel();
        let (sender, jobs) = mpsc::sync_channel(1);
        let (reply, receiver) = mpsc::channel();
        let released = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_released = released.clone();
        let thread = std::thread::spawn(move || {
            assert!(jobs.recv().is_err());
            resume.recv().unwrap();
            worker_released.store(true, std::sync::atomic::Ordering::Release);
            drop(reply);
        });
        let mut disk = DiskThread {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
            outstanding: Cell::new(1),
        };
        let error = disk.close_until(Instant::now()).unwrap_err();
        assert!(error.to_string().contains("did not drain"));
        assert!(disk.thread.is_some());
        assert!(!released.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(disk.outstanding.get(), 1);
        release.send(()).unwrap();
        disk.close_until(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert!(disk.thread.is_none());
        assert!(released.load(std::sync::atomic::Ordering::Acquire));
        disk.close_until(Instant::now()).unwrap();
    }

    #[test]
    fn draining_keeps_ownership_across_wait_errors_until_every_completion_arrives() {
        let (waiting, entered) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let (finished, done) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut calls = 0;
            drain_completions(3, |pending| {
                calls += 1;
                match calls {
                    1 => {
                        assert_eq!(pending, 3);
                        waiting.send(()).unwrap();
                        (1, Err(io::Error::from_raw_os_error(libc::EIO)))
                    }
                    2 => {
                        assert_eq!(pending, 2);
                        resume.recv().unwrap();
                        (0, Err(io::Error::from_raw_os_error(libc::EBADF)))
                    }
                    3 => {
                        assert_eq!(pending, 2);
                        (2, Ok(()))
                    }
                    _ => panic!("draining continued after all completions arrived"),
                }
            });
            assert_eq!(calls, 3);
            finished.send(()).unwrap();
        });
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(matches!(done.try_recv(), Err(mpsc::TryRecvError::Empty)));
        release.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
        drain_completions(0, |_| panic!("an empty drain must not wait"));
    }

    #[test]
    fn direct_open_failure_names_a_filesystem_refusing_direct_io() {
        let path = Path::new("/records/example.ckpt");
        let refused = direct_open_failure(
            "open record file",
            path,
            io::Error::from_raw_os_error(libc::EINVAL),
        )
        .to_string();
        assert!(refused.contains("refuses direct I/O"), "{refused}");
        assert!(refused.contains("/records/example.ckpt"), "{refused}");
        let missing = direct_open_failure(
            "open record file",
            path,
            io::Error::from_raw_os_error(libc::ENOENT),
        )
        .to_string();
        assert!(!missing.contains("refuses direct I/O"), "{missing}");
        assert!(missing.contains("/records/example.ckpt"), "{missing}");
    }

    #[test]
    fn full_queue_fails_submit_without_blocking() {
        let missing = PathBuf::from("/nonexistent/flare-offload/record.ckpt");
        let (sender, jobs) = mpsc::sync_channel(QUEUE_DEPTH);
        let (_replies, receiver) = mpsc::channel();
        let (release, parked) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _jobs = jobs;
            let _ = parked.recv();
        });
        let stalled = DiskThread {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
            outstanding: Cell::new(0),
        };
        for index in 0..QUEUE_DEPTH {
            stalled
                .submit(Job::Unlink {
                    id: JobId(index as u64),
                    path: missing.clone(),
                })
                .unwrap();
        }
        let error = stalled
            .submit(Job::Unlink {
                id: JobId(QUEUE_DEPTH as u64),
                path: missing.clone(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("full"), "{error}");
        assert_eq!(stalled.outstanding.get(), QUEUE_DEPTH);
        drop(release);
        drop(stalled);

        let mut live = DiskThread::start("flare-offload-disk").unwrap();
        let (mut accepted, mut refused) = (0, 0);
        for index in 0..4 * QUEUE_DEPTH {
            match live.submit(Job::Unlink {
                id: JobId(index as u64),
                path: missing.clone(),
            }) {
                Ok(()) => accepted += 1,
                Err(error) => {
                    assert!(error.to_string().contains("full"), "{error}");
                    refused += 1;
                }
            }
        }
        assert_eq!(accepted + refused, 4 * QUEUE_DEPTH);
        live.close().unwrap();
        let mut replies = Vec::new();
        live.poll(&mut replies);
        assert_eq!(replies.len(), accepted);
        for reply in &replies {
            assert_eq!(reply.operation, Operation::Unlink);
            succeeded(reply);
        }
        assert!(live.submit(Job::Close { id: JobId(0) }).is_err());
    }

    #[test]
    fn chunks_split_spans_at_the_chunk_size() {
        let spans = [
            Span {
                address: 0x1000,
                offset: 4096,
                length: 2 * CHUNK_BYTES + 4096,
            },
            Span {
                address: 0x9000,
                offset: 0,
                length: 0,
            },
            Span {
                address: 0xa000,
                offset: 1 << 30,
                length: 4096,
            },
        ];
        let chunks: Vec<(usize, u64, u32)> = Chunks {
            spans: &spans,
            index: 0,
            done: 0,
        }
        .map(|chunk| (chunk.address, chunk.offset, chunk.length))
        .collect();
        assert_eq!(
            chunks,
            [
                (0x1000, 4096, CHUNK_BYTES as u32),
                (
                    0x1000 + CHUNK_BYTES,
                    4096 + CHUNK_BYTES as u64,
                    CHUNK_BYTES as u32
                ),
                (
                    0x1000 + 2 * CHUNK_BYTES,
                    4096 + 2 * CHUNK_BYTES as u64,
                    4096
                ),
                (0xa000, 1 << 30, 4096),
            ]
        );
        assert!(aligned(0));
        assert!(aligned(8192));
        assert!(!aligned(8193));
    }

    #[test]
    fn a_read_ahead_hands_out_a_file_range_in_any_window_sizes_and_can_be_dropped_early() {
        let Some(directory) = test_directory("read-ahead") else {
            return;
        };
        let path = directory.join("range");
        let length = 3 * AHEAD_BLOCK_BYTES + 5 * 4096;
        let bytes: Vec<u8> = (0..length).map(|i| (i * 7 + i / 4096) as u8).collect();
        fs::write(&path, &bytes).unwrap();
        let start = 4096u64;
        let end = (length - 4096) as u64;
        let mut ahead =
            ReadAhead::start(flare_eos::direct_file(&path, false).unwrap(), start, end).unwrap();
        assert!(ahead.at(start) && !ahead.at(0));
        let mut at = start as usize;
        for size in [4096, 1 << 20, 3 * 4096, AHEAD_BLOCK_BYTES + 4096, 8192] {
            let mut got = vec![0u8; size];
            ahead.read(&mut got).unwrap();
            assert_eq!(got, bytes[at..at + size]);
            at += size;
            assert!(ahead.at(at as u64));
        }
        // The rest of the range, then nothing more.
        let mut rest = vec![0u8; end as usize - at];
        ahead.read(&mut rest).unwrap();
        assert_eq!(rest, bytes[at..end as usize]);
        assert!(ahead.read(&mut [0u8; 4096]).is_err());
        // A reader dropped with most of its range unread stops cleanly.
        let mut early = ReadAhead::start(
            flare_eos::direct_file(&path, false).unwrap(),
            0,
            length as u64,
        )
        .unwrap();
        let mut first = vec![0u8; 4096];
        early.read(&mut first).unwrap();
        assert_eq!(first, bytes[..4096]);
        drop(early);
        // A range of one block.
        let mut small =
            ReadAhead::start(flare_eos::direct_file(&path, false).unwrap(), 8192, 16384).unwrap();
        let mut got = vec![0u8; 8192];
        small.read(&mut got).unwrap();
        assert_eq!(got, bytes[8192..16384]);
        fs::remove_dir_all(&directory).unwrap();
    }
}
