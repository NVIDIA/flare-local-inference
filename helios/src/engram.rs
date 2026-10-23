//! Host- or graph-produced n-gram requests and asynchronous compact-row delivery.

mod cache;
#[cfg(unix)]
mod prepared;
mod reader;
use crate::{
    Error, Result,
    cuda::{Event, PinnedHost, Stream},
    execution::{ExecutionManifest, GraphKind, Upload},
};
#[cfg(unix)]
pub use prepared::{PlaneSource, PreparedTables, SourceSegment, TableSource};
pub(crate) use reader::Plane;
use std::{
    fmt::Write,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

#[derive(Default)]
pub(crate) struct Stats {
    gathers: AtomicU64,
    gather_ns: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    resident_rows: AtomicU64,
    capacity_rows: AtomicU64,
    cache_bytes: AtomicU64,
    resident_scale_bytes: AtomicU64,
    disk_reads: AtomicU64,
    disk_bytes: AtomicU64,
    disk_plan_ns: AtomicU64,
    disk_submit_ns: AtomicU64,
    disk_wait_ns: AtomicU64,
    disk_complete_ns: AtomicU64,
}

impl Stats {
    pub(crate) fn render(&self, text: &mut String) {
        for (name, kind, help, counter) in [
            (
                "gathers_total",
                "counter",
                "Embedding row gathers, excluding the wait for GPU-produced IDs.",
                &self.gathers,
            ),
            (
                "gather_nanoseconds_total",
                "counter",
                "Wall time inside embedding row gathers.",
                &self.gather_ns,
            ),
            (
                "row_cache_hits_total",
                "counter",
                "Cached row hits; duplicate IDs within one gather count once.",
                &self.hits,
            ),
            (
                "row_cache_misses_total",
                "counter",
                "Cached row misses; duplicate IDs within one gather count once.",
                &self.misses,
            ),
            (
                "row_cache_evictions_total",
                "counter",
                "Resident rows replaced by other row IDs.",
                &self.evictions,
            ),
            (
                "row_cache_resident_rows",
                "gauge",
                "Rows held in the embedding row cache.",
                &self.resident_rows,
            ),
            (
                "row_cache_capacity_rows",
                "gauge",
                "Maximum rows held in the embedding row cache.",
                &self.capacity_rows,
            ),
            (
                "row_cache_bytes",
                "gauge",
                "Allocated embedding row payload, usage counters, index and bounded miss workspace bytes.",
                &self.cache_bytes,
            ),
            (
                "resident_scale_bytes",
                "gauge",
                "Resident scale-plane allocation bytes for disk-backed embedding tables, separate from the row cache.",
                &self.resident_scale_bytes,
            ),
            (
                "disk_reads_total",
                "counter",
                "Direct-I/O embedding read requests queued.",
                &self.disk_reads,
            ),
            (
                "disk_bytes_total",
                "counter",
                "Bytes requested by direct-I/O embedding reads.",
                &self.disk_bytes,
            ),
            (
                "disk_plan_nanoseconds_total",
                "counter",
                "Wall time building, sorting and coalescing disk row reads.",
                &self.disk_plan_ns,
            ),
            (
                "disk_submit_nanoseconds_total",
                "counter",
                "Cumulative worker wall time preparing and issuing disk read batches; Linux kernel submission is included in wait time.",
                &self.disk_submit_ns,
            ),
            (
                "disk_wait_nanoseconds_total",
                "counter",
                "Cumulative worker wall time in disk completion waits, including Linux io_uring submission.",
                &self.disk_wait_ns,
            ),
            (
                "disk_complete_nanoseconds_total",
                "counter",
                "Cumulative worker wall time processing disk completions and copying rows, excluding completion waits.",
                &self.disk_complete_ns,
            ),
        ] {
            let _ = writeln!(
                text,
                "# HELP flare_engram_{name} {help}\n# TYPE flare_engram_{name} {kind}\nflare_engram_{name} {}",
                counter.load(Ordering::Relaxed)
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Storage {
    #[default]
    Disk,
    Ram,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Table {
    pub(crate) planes: [Plane; 2],
    pub(crate) heads: usize,
    pub(crate) weights: String,
    pub(crate) scales: String,
    pub(crate) hashes: String,
    pub(crate) event: String,
    pub(crate) request: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Index {
    pub(crate) tables: Vec<Table>,
}

struct Staging {
    data: PinnedHost,
    ready: PinnedHost,
    error: Mutex<Option<Error>>,
}
impl Staging {
    fn signal(&self, generation: u32) {
        unsafe { &*(self.ready.address() as *const AtomicU32) }
            .store(generation, Ordering::Release);
    }
}
enum Requests {
    Host(Vec<i64>),
    Device {
        address: usize,
        flag: usize,
        sequence: u64,
        rows: usize,
    },
}
impl Requests {
    fn rows(&self) -> usize {
        match self {
            Self::Host(ids) => ids.len(),
            Self::Device { rows, .. } => *rows,
        }
    }
    fn read(self, cancelled: &AtomicBool) -> Result<Vec<i64>> {
        match self {
            Self::Host(ids) => Ok(ids),
            Self::Device {
                address,
                flag,
                sequence,
                rows,
            } => {
                let ready = unsafe { &*(flag as *const AtomicU64) };
                while ready.load(Ordering::Acquire) != sequence {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("n-gram request cancelled".into());
                    }
                    std::hint::spin_loop();
                }
                Ok(unsafe { std::slice::from_raw_parts(address as *const i64, rows) }.to_vec())
            }
        }
    }
}
struct Job {
    generation: u32,
    requests: Requests,
}
struct State {
    sequence: u64,
    generation: u32,
    submitted: usize,
    completed: usize,
}
struct Layer {
    staging: Arc<Staging>,
    sender: Option<mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    state: Mutex<State>,
    cancelled: Arc<AtomicBool>,
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) struct Shared {
    pub index: Arc<Index>,
    layers: Vec<Layer>,
    ranks: usize,
    max_tokens: usize,
    stats: Arc<Stats>,
}
impl Shared {
    #[cfg(test)]
    pub(crate) fn layout_fixture(index: Index, ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            index: Arc::new(index), layers: Vec::new(), ranks, max_tokens: 4096,
            stats: Arc::new(Stats::default()),
        })
    }

    pub fn new(
        index: Index,
        files: &[Arc<std::fs::File>],
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        storage: Storage,
        cache_bytes: u64,
        workers: usize,
        resident_scales: bool,
    ) -> Result<Arc<Self>> {
        if storage == Storage::Ram && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        let index = Arc::new(index);
        let stats = Arc::new(Stats::default());
        let table_bytes: u128 = index
            .tables
            .iter()
            .flat_map(|table| &table.planes)
            .map(|plane| plane.nbytes as u128)
            .sum();
        let readers = index
            .tables
            .iter()
            .map(|table| {
                let bytes: u128 = table.planes.iter().map(|plane| plane.nbytes as u128).sum();
                let budget = usize::try_from(cache_bytes as u128 * bytes / table_bytes)
                    .map_err(|_| "embedding row cache exceeds host address range")?;
                let rows = max_tokens
                    .checked_mul(table.heads)
                    .ok_or("embedding row capacity exceeds host range")?;
                reader::Reader::from_files(
                    files,
                    &table.planes,
                    storage == Storage::Ram,
                    budget,
                    rows,
                    stats.clone(),
                    workers,
                    resident_scales,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_readers(index, readers, ranks, max_tokens, context, device, stats)
    }

    fn with_readers(
        index: Arc<Index>,
        readers: Vec<reader::Reader>,
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        stats: Arc<Stats>,
    ) -> Result<Arc<Self>> {
        if ranks == 0 || ranks > 16 || max_tokens == 0 || readers.len() != index.tables.len() {
            return Err("invalid embedding delivery geometry".into());
        }
        let stream = Stream::open(context, device)?;
        let mut layers = Vec::new();
        for (table, mut reader) in index.tables.iter().zip(readers) {
            let staging_bytes = table
                .planes
                .iter()
                .try_fold(0usize, |total, plane| total.checked_add(plane.row_bytes))
                .and_then(|n| n.checked_mul(table.heads))
                .and_then(|n| n.checked_mul(max_tokens))
                .ok_or("embedding staging exceeds host range")?;
            let staging = Arc::new(Staging {
                data: stream.mapped_host_buffer(staging_bytes)?,
                ready: stream.mapped_host_buffer(4)?,
                error: Mutex::new(None),
            });
            let (sender, receiver) = mpsc::sync_channel::<Job>(1);
            let buffers = staging.clone();
            let planes = table.planes;
            let row_bytes = planes.iter().map(|p| p.row_bytes).sum::<usize>();
            let cancelled = Arc::new(AtomicBool::new(false));
            let stop = cancelled.clone();
            let worker = thread::Builder::new()
                .name(format!("flare-{}", table.event))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        // A single reader owns these bytes. Every rank completes the previous
                        // consumer graph before another job can reuse them; flag publication
                        // precedes GPU DMA through the queued stream memory wait.
                        let output = unsafe {
                            std::slice::from_raw_parts_mut(
                                buffers.data.address() as *mut u8,
                                job.requests.rows() * row_bytes,
                            )
                        };
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let ids = job.requests.read(&stop)?;
                            reader.gather(&planes, &ids, output)
                        }))
                        .unwrap_or_else(|_| Err("Engram reader panicked".into()));
                        if let Err(error) = result {
                            output.fill(0);
                            *buffers.error.lock().unwrap() = Some(error);
                        }
                        // Release failed reads too, so graph shutdown can drain before reporting the error.
                        buffers.signal(job.generation);
                    }
                })
                .map_err(|e| format!("start Engram reader: {e}"))?;
            layers.push(Layer {
                staging,
                cancelled,
                sender: Some(sender),
                worker: Some(worker),
                state: Mutex::new(State {
                    sequence: 0,
                    generation: 0,
                    submitted: 0,
                    completed: 0,
                }),
            });
        }
        Ok(Arc::new(Self {
            index,
            layers,
            ranks,
            max_tokens,
            stats,
        }))
    }

    pub(crate) fn stats(&self) -> Arc<Stats> {
        self.stats.clone()
    }

    pub(crate) fn cache_bytes(&self) -> u64 {
        self.stats.cache_bytes.load(Ordering::Relaxed)
    }
    fn start(
        &self,
        table: usize,
        rank: usize,
        sequence: u64,
        requests: Requests,
    ) -> Result<(usize, u32)> {
        let rows = requests.rows();
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let mut state = layer.state.lock().unwrap();
        if let Some(error) = layer.staging.error.lock().unwrap().as_ref() {
            return Err(error.clone());
        }
        if state.sequence != sequence {
            if state.sequence > sequence
                || (state.sequence != 0 && state.completed != (1 << self.ranks) - 1)
            {
                return Err("Engram staging reused before all ranks completed".into());
            }
            let generation = state.generation.wrapping_add(1);
            layer
                .sender
                .as_ref()
                .unwrap()
                .send(Job {
                    generation,
                    requests,
                })
                .map_err(|_| "Engram reader disconnected")?;
            *state = State {
                sequence,
                generation,
                submitted: 0,
                completed: 0,
            };
        }
        if state.submitted & (1 << rank) != 0 {
            return Err("rank submitted Engram generation twice".into());
        }
        state.submitted |= 1 << rank;
        Ok((rows, state.generation))
    }
    fn finish(&self, rank: usize, sequence: u64) -> Result<()> {
        let mut error = None;
        for layer in &self.layers {
            let mut state = layer.state.lock().unwrap();
            if state.sequence != sequence || state.submitted & (1 << rank) == 0 {
                return Err("Engram completion does not match its submission".into());
            }
            state.completed |= 1 << rank;
            error = error.or_else(|| layer.staging.error.lock().unwrap().clone());
        }
        error.map_or(Ok(()), Err)
    }
}

#[derive(Clone)]
pub struct Binding {
    pub(crate) shared: Arc<Shared>,
    pub(crate) rank: usize,
    pub(crate) events: Vec<usize>,
}
impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramBinding")
            .field("rank", &self.rank)
            .finish()
    }
}
struct Transfer {
    stream: Stream,
    flag: usize,
    weights: usize,
    scales: usize,
    event: usize,
    request: Option<(usize, usize)>,
}
pub(crate) struct Transfers {
    binding: Binding,
    layers: Vec<Transfer>,
}
impl Transfers {
    pub fn new(binding: Binding, manifest: &ExecutionManifest) -> Result<Self> {
        if let Some(buffer) = manifest.buffers.iter().find(|buffer| buffer.name == "readiness_timestamps") {
            if buffer.shape[1] / 3 != binding.shared.index.tables.len() {
                return Err("readiness interval count differs from the embedding table catalog".into());
            }
        }
        let mut layers = Vec::new();
        for (i, table) in binding.shared.index.tables.iter().enumerate() {
            let stream = Stream::open(manifest.context, manifest.device)?;
            let flag = stream.mapped_address(&binding.shared.layers[i].staging.ready)?;
            let request = table
                .request
                .as_ref()
                .map(|name| {
                    let buffer = manifest.buffer(name)?;
                    let address = stream.host_address(buffer.address, buffer.nbytes)?;
                    Ok::<_, Error>((address, address + buffer.nbytes - 8))
                })
                .transpose()?;
            layers.push(Transfer {
                stream,
                flag,
                weights: manifest.buffer(&table.weights)?.address,
                scales: manifest.buffer(&table.scales)?.address,
                event: binding.events[i],
                request,
            });
        }
        Ok(Self { binding, layers })
    }
    pub fn enqueue(
        &self,
        sequence: u64,
        tokens: usize,
        uploads: &[Upload<'_>],
        inputs_ready: &Event,
    ) -> Result<()> {
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let heads = table.heads;
            let requests = if let Some((address, flag)) = transfer.request {
                Requests::Device {
                    address,
                    flag,
                    sequence,
                    rows: tokens * heads,
                }
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                Requests::Host(
                    upload
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            };
            let (rows, generation) =
                self.binding
                    .shared
                    .start(i, self.binding.rank, sequence, requests)?;
            let staging = &self.binding.shared.layers[i].staging;
            // The host reader may wait for IDs produced by the model graph. Express its input
            // upload prerequisite to CUDA before an otherwise opaque stream-memory wait.
            transfer.stream.wait(inputs_ready)?;
            unsafe {
                transfer.stream.wait_value(transfer.flag, generation)?;
                transfer.stream.upload(
                    transfer.weights,
                    &staging.data,
                    0,
                    rows * table.planes[0].row_bytes,
                )?;
                transfer.stream.upload(
                    transfer.scales,
                    &staging.data,
                    rows * table.planes[0].row_bytes,
                    rows * table.planes[1].row_bytes,
                )?;
                transfer.stream.record_borrowed(transfer.event)?;
            }
        }
        Ok(())
    }
    pub fn finish(&self, sequence: u64) -> Result<()> {
        self.binding.shared.finish(self.binding.rank, sequence)
    }
    pub fn cancel(&self) {
        for layer in &self.binding.shared.layers {
            layer.cancelled.store(true, Ordering::Release);
        }
    }
    pub fn drain(&self) -> Result<()> {
        let mut error = None;
        for transfer in &self.layers {
            error = error.or(transfer.stream.synchronize().err());
        }
        error.map_or(Ok(()), Err)
    }
}

pub(crate) fn consumes(kind: GraphKind) -> bool {
    matches!(
        kind,
        GraphKind::Prefill
            | GraphKind::ScoredPrefill
            | GraphKind::Decode
            | GraphKind::Speculative
            | GraphKind::Verification
            | GraphKind::PackedSpeculative
            | GraphKind::Mixed
    )
}
