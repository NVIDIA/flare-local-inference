use crate::{
    Key, Result,
    arena::{Arena, Extent, ExtentSlice},
    disk::{self, DiskExtent},
    invalid,
    local::Local,
    proto,
    record::{self, Descriptor, NamespaceId},
};
use flare_selene::shared::index::{Eviction, Index as Tree};
use flare_selene::shared::region::Region;
use flare_selene::{
    JobId,
    shared::{
        disk::{DiskThread, Job as DiskJob, Span},
        record::Header,
    },
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

pub const OPEN_GETS: usize = 4;
pub const MAX_CHAIN: usize = flare_selene::shared::MAX_CHAIN_NODES as usize;
pub use flare_selene::shared::PROGRESS_TIMEOUT as NO_PROGRESS;

/// Whether an operation that has been idle for `idle` has stalled. A put whose client has
/// delivered everything (`delivered`) waits on this manager's own disk, whose jobs queue behind
/// each other, so it has stalled only if the disk made no progress either (`disk_idle`, `None`
/// when no job is outstanding). Otherwise the idle time alone decides.
fn stalled(idle: Duration, delivered: bool, disk_idle: Option<Duration>) -> bool {
    idle >= NO_PROGRESS && !(delivered && disk_idle.is_some_and(|disk| disk < NO_PROGRESS))
}
pub use flare_selene::shared::MAX_WINDOW_BYTES as DEFAULT_MAX_WINDOW;

/// The staging of a store built in code (an embedding or a test). The command line always sets it
/// from `--staging`, whose default is larger.
pub const EMBEDDED_STAGING: usize = 16 * 1024 * 1024;

pub use crate::config::Config;

/// Physical charge: the arena allocator size class or the padded disk record.
pub fn record_charge(header: &Header, in_memory: bool) -> u64 {
    let length = header.length_with_row();
    if in_memory {
        length.max(4096).next_power_of_two()
    } else {
        length
    }
}

#[derive(Clone)]
pub struct Location {
    pub header: Header,
    pub memory: Option<Arc<Extent>>,
    pub disk: Option<Arc<DiskExtent>>,
    pub durable: bool,
    pub writing: bool,
    /// The disk job currently transforming these bytes, so a stale completion is ignored.
    pub job: Option<u64>,
    pub retry: Instant,
    pub bytes: u64,
}

/// What a finished Put should do to the tree entry for its key.
enum PublishTarget {
    /// No entry exists; index a new record.
    Insert,
    /// A page-only record is present; take its place.
    Replace,
}

impl Location {
    pub fn reading(&self) -> bool {
        self.memory
            .as_ref()
            .is_some_and(|extent| Arc::strong_count(extent) > 1)
    }
}

pub struct Namespace {
    pub directory: Option<PathBuf>,
    pub descriptor: Descriptor,
    pub tree: Tree<Location>,
    pub disk: Option<DiskThread>,
    reads: Option<DiskThread>,
    pub inflight: HashSet<Key>,
}

impl Namespace {
    fn path(&self, key: Key) -> Option<PathBuf> {
        self.directory
            .as_ref()
            .map(|directory| directory.join(record::file_name(&key)))
    }
    fn temporary(&self, id: u64) -> Option<Arc<DiskExtent>> {
        let directory = self.directory.as_ref()?;
        let extent = Arc::new(DiskExtent::new(directory.join(format!(".write-{id}.tmp"))));
        extent.retire();
        Some(extent)
    }
    fn parent_durable(&self, parent: Option<Key>) -> bool {
        parent.is_none_or(|key| self.tree[&key].location.durable)
    }
}

#[derive(Clone, Copy)]
struct PlannedGrant {
    record: u32,
    record_offset: u64,
    length: u64,
    pages: u32,
    row: bool,
    page_first: u64,
}

impl PlannedGrant {
    fn window(
        self,
        transfer: u64,
        index: u32,
        region: Region,
        offset: u64,
        hold: Arc<Extent>,
    ) -> WindowGrant {
        WindowGrant {
            transfer,
            index,
            record: self.record,
            record_offset: self.record_offset,
            length: self.length,
            region: region_wire(region),
            offset,
            hold,
            page_first: self.page_first,
            page_count: self.pages,
            row: self.row,
        }
    }
}

struct FilledWindows {
    granted: HashMap<u32, PlannedGrant>,
    next: usize,
    pages: Vec<Option<u64>>,
    row: Option<u64>,
    pages_filled: u32,
    row_filled: bool,
}

/// Physical resources a Put occupies, for the shape (arena or streamed) it
/// will actually get. The arena shape is one buddy-rounded record extent;
/// the streaming shape is the disk record plus one fixed
/// staging hold of two of its largest windows. `unit` keeps the largest
/// indivisible window separate so an oversized window is never mistaken for a
/// whole-record requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PutRequirement {
    streaming: bool,
    /// Charge against the overall budget (`charge_total`).
    charge: u64,
    /// Portion retained until the Put finishes (arena extent or disk record).
    reservation: u64,
    /// Region bytes the shape needs in total.
    allocation: u64,
    /// Largest indivisible region allocation (one window, or the extent).
    unit: u64,
}

impl PutRequirement {
    fn arena(extent: u64) -> Self {
        Self {
            streaming: false,
            charge: extent,
            reservation: extent,
            allocation: extent,
            unit: extent,
        }
    }

    /// A streamed Put reserves one fixed hold of two of its largest windows.
    /// Live windows are slices of that hold, so the staging headroom the
    /// transfer will need cannot be taken by a concurrent operation.
    fn streaming(disk: u64, unit: u64) -> Self {
        let allocation = unit.saturating_mul(2);
        Self {
            streaming: true,
            charge: disk.saturating_add(allocation),
            reservation: disk,
            allocation,
            unit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct HoldSlot {
    index: u32,
    /// Chunk write still reading this slot; the slot is reusable once it
    /// completes.
    job: Option<u64>,
}

fn set_slot_job(put: &mut Put, index: u32, job: u64) {
    for slot in put.hold_slots.iter_mut().flatten() {
        if slot.index == index {
            slot.job = Some(job);
        }
    }
}

fn release_slot_job(put: &mut Put, job: u64) {
    for slot in &mut put.hold_slots {
        if slot.is_some_and(|slot| slot.job == Some(job)) {
            *slot = None;
        }
    }
}

enum RootChoice {
    Store(Extent),
    Wait,
}

pub struct Put {
    namespace: NamespaceId,
    header: Header,
    first_page: u32,
    grants: Vec<PlannedGrant>,
    requested: usize,
    windows: FilledWindows,
    memory: Option<Arc<Extent>>,
    temporary: Option<Arc<DiskExtent>>,
    streaming: bool,
    commit_bytes: Vec<u8>,
    metadata: Option<Arc<Extent>>,
    committed: bool,
    finalizing: bool,
    cancelled: bool,
    terminal: bool,
    finished: bool,
    pending: usize,
    blocked: VecDeque<(u64, u64, ExtentSlice)>,
    pins: VecDeque<Key>,
    reservation: u64,
    /// Fixed staging reservation for a streamed Put. Live windows are
    /// slices of it, so concurrent operations cannot consume the headroom
    /// the transfer still needs.
    staging_hold: Option<Arc<Extent>>,
    /// Slot size inside `staging_hold`.
    hold_unit: usize,
    /// Window occupying each hold slot, and the write draining it.
    hold_slots: [Option<HoldSlot>; 2],
    events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
    outbox: VecDeque<Result<PutAnswer>>,
}

impl Put {
    fn owns_payload(&self) -> bool {
        self.windows.pages_filled as usize == self.header.tail_pages as usize
            && (self.windows.row_filled || self.header.row_record_bytes() == 0)
    }
}

struct GetRecord {
    header: Header,
    row: bool,
    memory: Option<Arc<Extent>>,
    file: Option<Arc<File>>,
    meta: Option<Vec<u8>>,
    meta_sent: bool,
}

pub struct Get {
    namespace: NamespaceId,
    pins: VecDeque<Key>,
    records: Vec<GetRecord>,
    grants: Vec<PlannedGrant>,
    requested: usize,
    next_grant: usize,
    next_meta: usize,
    window_reads: usize,
    pending_reads: usize,
    cancelled: bool,
    terminal: bool,
    finished: bool,
    events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
    outbox: VecDeque<Result<GetAnswer>>,
}

pub enum Operation {
    Put(Box<Put>),
    Get(Box<Get>),
}

pub struct Open {
    pub operation: Operation,
    pub progress: Instant,
}

impl Open {
    /// Already failed or cancelled: failing it again would only count and refuse it again.
    fn is_over(&self) -> bool {
        match &self.operation {
            Operation::Put(put) => put.terminal || put.cancelled,
            Operation::Get(get) => get.terminal || get.cancelled,
        }
    }

    /// A put that owes nothing more to its client and waits on this manager's own disk: it is
    /// committed, or it has asked for windows, holds none, and both staging slots are still
    /// draining or chunks are blocked on a full disk queue.
    fn waits_on_disk(&self) -> bool {
        matches!(&self.operation, Operation::Put(put) if put.committed
            || (put.requested > 0
                && put.windows.granted.is_empty()
                && (put.hold_slots.iter().all(Option::is_some) || !put.blocked.is_empty())))
    }

    /// A put whose next window the pool has not yet asked for, with no granted window for the
    /// client to fill, is queued behind other transfers' window credits. That wait is not a
    /// client stall: `progress` counts from the grant.
    fn waits_for_grant(&self) -> bool {
        matches!(&self.operation, Operation::Put(put) if !put.committed
            && put.requested == 0
            && put.windows.granted.is_empty()
            && put.windows.next < put.grants.len())
    }
}

pub struct Session {
    pub namespace: NamespaceId,
    pub ops: HashMap<u64, Open>,
    pub pid: u32,
    pub closed: bool,
}

#[derive(Clone, Copy)]
enum CatalogKind {
    Added,
    Changed,
    Removed,
}

enum ReadKind {
    Meta { length: usize },
    Fill,
    Window { grant: u32 },
}

enum Job {
    Write {
        namespace: NamespaceId,
        key: Key,
        _memory: Arc<Extent>,
        _temporary: Arc<DiskExtent>,
    },
    Chunk {
        connection: u64,
        transfer: u64,
        _data: ExtentSlice,
        _temporary: Arc<DiskExtent>,
    },
    Finish {
        connection: u64,
        transfer: u64,
        final_extent: Arc<DiskExtent>,
        _metadata: Arc<Extent>,
        _temporary: Arc<DiskExtent>,
    },
    Compact {
        namespace: NamespaceId,
        key: Key,
        bytes: u64,
    },
    GetRead {
        connection: u64,
        transfer: u64,
        record: u32,
        kind: ReadKind,
        buffer: Extent,
    },
}

fn finish_record(
    put: &Put,
    job: u64,
    connection: u64,
    transfer: u64,
    path: PathBuf,
) -> Option<(DiskJob, Job)> {
    let metadata = put.metadata.clone()?;
    let temporary = put.temporary.clone()?;
    let final_extent = Arc::new(DiskExtent::new(path));
    let task = DiskJob::write_record(
        JobId(job),
        temporary.path.clone(),
        0,
        metadata.bytes(),
        Some((final_extent.path.clone(), put.header.length(true))),
    );
    Some((
        task,
        Job::Finish {
            connection,
            transfer,
            final_extent,
            _metadata: metadata,
            _temporary: temporary,
        },
    ))
}

#[derive(Default)]
pub struct Metrics {
    pub dropped: u64,
    pub cancels: u64,
    pub puts_ok: u64,
    pub puts_error: u64,
    pub present: u64,
    pub retry_queue: u64,
    pub retry_budget: u64,
    pub retry_inflight: u64,
    pub oversize: u64,
    pub gets_ok: u64,
    pub gets_error: u64,
    pub evictions: u64,
    pub expired: [u64; 2],
    pub compactions: u64,
    pub nvme_bytes: u64,
    pub nvme_seconds: f64,
    pub nvme_errors: u64,
    pub hits: HashMap<(String, &'static str), u64>,
}

#[derive(Clone, Debug)]
pub struct EntryData {
    pub key: Key,
    pub parent: Option<Key>,
    pub bucket: [u8; 16],
    pub tail_pages: u32,
    pub row: bool,
    /// 0 = DRAM, 1 = NVMe.
    pub tier: u8,
    pub durable: bool,
    pub bytes: u64,
}

struct PendingPut {
    connection: u64,
    transfer: u64,
    request: PutRequest,
    events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
}

struct PendingGet {
    connection: u64,
    transfer: u64,
    request: GetRequest,
    events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
}

pub struct PutRequest {
    pub namespace: NamespaceId,
    pub key: Key,
    pub published_parent: Option<Key>,
    pub bucket: [u8; 16],
    pub chain_pages: u32,
    pub published_tail_pages: u32,
    pub row: bool,
}

pub struct GetRequest {
    pub namespace: NamespaceId,
    pub keys: Vec<Key>,
    pub skip_pages: u32,
    pub expected_pages: u32,
    pub row: bool,
}

pub struct Store {
    config: Config,
    local: Local,
    memory: Arena,
    staging: Arena,
    // Engines drain before the root lock and job buffers are released.
    namespaces: HashMap<NamespaceId, Namespace>,
    _lock: Option<disk::RootLock>,
    sessions: HashMap<u64, Session>,
    jobs: HashMap<u64, Job>,
    job_counter: u64,
    /// The last time any disk job finished, or the last tick with none outstanding.
    disk_progress: Instant,
    clock: u64,
    metrics: Metrics,
    charged: u64,
    /// The highest `charge_total` seen at an admission or a tick.
    reserved_peak: u64,
    retired_records: Vec<(NamespaceId, Key, Location)>,
    pending_puts: HashMap<u64, VecDeque<PendingPut>>,
    pending_gets: HashMap<u64, VecDeque<PendingGet>>,
    put_order: VecDeque<u64>,
    get_order: VecDeque<u64>,
    catalogs: HashMap<NamespaceId, tokio::sync::mpsc::UnboundedSender<proto::CatalogEvent>>,
    read_backlog: VecDeque<(NamespaceId, u64, DiskJob)>,
    _name_lock: File,
}

/// Names the startup step and path an error came from.
fn step_error(step: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{step} at {}: {error}", path.display()),
    )
}

fn put_mut(
    sessions: &mut HashMap<u64, Session>,
    connection: u64,
    transfer: u64,
) -> Option<(&mut Put, &mut Instant)> {
    let open = sessions.get_mut(&connection)?.ops.get_mut(&transfer)?;
    match &mut open.operation {
        Operation::Put(put) => Some((put, &mut open.progress)),
        _ => None,
    }
}

fn get_mut(
    sessions: &mut HashMap<u64, Session>,
    connection: u64,
    transfer: u64,
) -> Option<(&mut Get, &mut Instant)> {
    let open = sessions.get_mut(&connection)?.ops.get_mut(&transfer)?;
    match &mut open.operation {
        Operation::Get(get) => Some((get, &mut open.progress)),
        _ => None,
    }
}

fn get_ref(sessions: &HashMap<u64, Session>, connection: u64, transfer: u64) -> Option<&Get> {
    match &sessions.get(&connection)?.ops.get(&transfer)?.operation {
        Operation::Get(get) => Some(get),
        _ => None,
    }
}

fn put_ref(sessions: &HashMap<u64, Session>, connection: u64, transfer: u64) -> Option<&Put> {
    match &sessions.get(&connection)?.ops.get(&transfer)?.operation {
        Operation::Put(put) => Some(put),
        _ => None,
    }
}

impl Store {
    pub fn open(config: Config) -> Result<Self> {
        config.validate()?;
        let memory = usize::try_from(config.memory)
            .map_err(|_| invalid("--memory is too large for this host"))?;
        let staging = usize::try_from(config.staging)
            .map_err(|_| invalid("--staging is too large for this host"))?;
        let name_lock = crate::local::lock_pool(&config.name, &config.socket)?;
        if config.root.is_some() {
            // Every namespace needs disk threads, and each needs an io_uring ring. Find out now:
            // a store that cannot write would otherwise run and refuse every put.
            DiskThread::start("manager-probe").map_err(|error| {
                invalid(format!(
                    "the store cannot start its disk threads: {error}. The manager needs \
                     io_uring: allow io_uring_setup, io_uring_enter and io_uring_register in \
                     the container's seccomp profile"
                ))
            })?;
        }
        let lock = config
            .root
            .as_ref()
            .map(|root| {
                disk::RootLock::acquire(root)
                    .map_err(|error| step_error("lock the root", root, invalid(error.to_string())))
            })
            .transpose()?;
        let recovered = match (&config.root, &lock) {
            (Some(root), Some(_)) => disk::recover(root, &config.name)
                .map_err(|error| step_error("recover", root, error))?,
            _ => Vec::new(),
        };
        let local = Local::open(&config.region_name()?, memory, staging).map_err(|error| {
            step_error(
                "open shared memory in",
                Path::new(crate::local::SHM_ROOT),
                error,
            )
        })?;
        let memory = local.arena();
        let staging = local.staging();
        let mut store = Self {
            _name_lock: name_lock,
            config: config.clone(),
            local,
            memory,
            staging,
            _lock: lock,
            namespaces: HashMap::new(),
            sessions: HashMap::new(),
            jobs: HashMap::new(),
            job_counter: 0,
            disk_progress: Instant::now(),
            clock: 0,
            metrics: Metrics::default(),
            charged: 0,
            reserved_peak: 0,
            retired_records: Vec::new(),
            pending_puts: HashMap::new(),
            pending_gets: HashMap::new(),
            put_order: VecDeque::new(),
            get_order: VecDeque::new(),
            catalogs: HashMap::new(),
            read_backlog: VecDeque::new(),
        };
        let mut recovered_age = Vec::new();
        for recovered in recovered {
            let mut namespace = store.new_namespace(recovered.descriptor)?;
            let id = namespace.descriptor.namespace_id();
            for (header, row, extent) in recovered.records {
                let modified = fs::metadata(&extent.path)?
                    .modified()
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                recovered_age.push((modified, id, header.key));
                let bytes = header.length(row);
                namespace.tree.insert(
                    header.key,
                    header.parent,
                    header.bucket,
                    header.tail_pages as usize,
                    row,
                    true,
                    Location {
                        header,
                        memory: None,
                        disk: Some(extent),
                        durable: false,
                        writing: false,
                        job: None,
                        retry: Instant::now(),
                        bytes,
                    },
                )
                .map_err(|e| invalid(e.to_string()))?;
                store.charged += bytes;
            }
            store.namespaces.insert(id, namespace);
            store.refresh_durability(id);
        }
        recovered_age.sort_unstable();
        for (modified, id, key) in recovered_age {
            store.clock += 1;
            let tree = &mut store.namespaces.get_mut(&id).unwrap().tree;
            tree.access_at(&key, store.clock);
            let node = tree.node_mut(&key).unwrap();
            node.last_used = modified;
        }
        Ok(store)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    /// What the pool service needs from the store: the shared region files, a handle on the
    /// staging region.
    pub fn pool_view(&self) -> (crate::proto::v2::SharedMemory, crate::arena::Arena) {
        (
            crate::proto::v2::SharedMemory {
                arena_path: self.local.arena_path().display().to_string(),
                arena_bytes: self.local.arena_size(),
                staging_path: self.local.staging_path().display().to_string(),
                staging_bytes: self.local.staging_size(),
                regions_generation: self.local.regions_generation(),
            },
            self.local.staging(),
        )
    }

    /// True while the shared regions are held for the previous run's processes.
    pub fn held_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.local.held_flag()
    }

    pub(crate) fn check_regions(&self) -> Result<()> {
        self.local.check_regions()
    }

    pub fn shm_directory(&self) -> PathBuf {
        self.local.directory().to_path_buf()
    }
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn new_namespace(&self, descriptor: Descriptor) -> Result<Namespace> {
        let directory = self
            .config
            .root
            .as_ref()
            .map(|root| root.join(descriptor.name()));
        let start = |name| -> Result<_> {
            #[cfg(any(test, feature = "test-support"))]
            let worker = DiskThread::start_with_faults(name, self.config.faults.clone());
            #[cfg(not(any(test, feature = "test-support")))]
            let worker = DiskThread::start(name);
            worker.map_err(|e| invalid(e.to_string()))
        };
        let (disk, reads) = if directory.is_some() {
            (Some(start("manager-write")?), Some(start("manager-read")?))
        } else {
            (None, None)
        };
        let mut tree = Tree::new(descriptor.page_tokens);
        tree.max_age = self.config.max_age;
        Ok(Namespace {
            directory,
            tree,
            descriptor,
            disk,
            reads,
            inflight: HashSet::new(),
        })
    }

    pub fn register_namespace(&mut self, descriptor: Descriptor) -> Result<NamespaceId> {
        descriptor.validate()?;
        let id = descriptor.namespace_id();
        if let Some(stored) = self.namespaces.get(&id) {
            if stored.descriptor != descriptor {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "namespace descriptor mismatch",
                ));
            }
            return Ok(id);
        }
        if let Some(root) = &self.config.root {
            fs::create_dir_all(root.join(descriptor.name()))?;
        }
        let namespace = self.new_namespace(descriptor)?;
        self.namespaces.insert(id, namespace);
        Ok(id)
    }

    pub fn namespace_id(&self) -> Option<NamespaceId> {
        self.namespaces.keys().next().copied()
    }

    pub fn session(&self, connection: u64) -> Option<&Session> {
        self.sessions.get(&connection)
    }
    pub fn hello(&mut self, connection: u64, pid: u32, descriptor: Descriptor) -> Result<()> {
        let id = self.register_namespace(descriptor)?;
        self.local.accept(connection, pid)?;
        self.sessions.insert(
            connection,
            Session {
                namespace: id,
                ops: HashMap::new(),
                pid,
                closed: false,
            },
        );
        Ok(())
    }

    /// The arena and staging mappings, for the data channels.
    pub fn regions(&self) -> crate::regions::LocalRegions {
        crate::regions::LocalRegions::new(self.memory.clone(), self.staging.clone())
    }

    /// Remove leaves through the shared tree, accounting bytes and files.
    fn evict_accounted(&mut self, id: NamespaceId, key: Key) -> u64 {
        let eligible = self
            .namespaces
            .get(&id)
            .and_then(|namespace| namespace.tree.node(&key))
            .is_some_and(|node| node.ready && node.children() == 0 && !node.pinned());
        if !eligible {
            return 0;
        }
        let removed = self
            .namespaces
            .get_mut(&id)
            .and_then(|namespace| namespace.tree.remove(&key).ok());
        let Some(nodes) = removed else {
            return 0;
        };
        let count = nodes.len() as u64;
        for node in nodes {
            if node.location.memory.is_some() || node.location.disk.is_some() {
                self.charged = self.charged.saturating_sub(node.location.bytes);
            }
            if let Some(disk) = &node.location.disk {
                disk.retire();
            }
            self.delta(id, CatalogKind::Removed, node.key);
        }
        count
    }

    fn reused(&mut self, id: NamespaceId, key: Key) {
        self.clock += 1;
        let namespace = self.namespaces.get_mut(&id).unwrap();
        namespace.tree.access_at(&key, self.clock);
        namespace.tree.touch_ancestors(&key);
    }

    pub fn snapshot(&self, id: NamespaceId) -> Option<Vec<EntryData>> {
        let namespace = self.namespaces.get(&id)?;
        let mut keys: Vec<Key> = namespace.tree.nodes().map(|node| node.key).collect();
        keys.sort_unstable();
        let entries = keys
            .into_iter()
            .map(|key| Self::entry(namespace, key))
            .collect();
        Some(entries)
    }

    fn entry(namespace: &Namespace, key: Key) -> EntryData {
        let node = &namespace.tree[&key];
        EntryData {
            key,
            parent: node.parent,
            bucket: node.bucket,
            tail_pages: node.tail_pages as u32,
            row: node.row,
            tier: u8::from(node.location.memory.is_none()),
            durable: node.location.durable,
            bytes: node.location.bytes,
        }
    }

    fn delta(&mut self, id: NamespaceId, kind: CatalogKind, key: Key) {
        let Some(sender) = self.catalogs.get(&id) else {
            return;
        };
        let namespace = &self.namespaces[&id];
        let event = match kind {
            CatalogKind::Added => proto::CatalogEvent::Added(Self::entry(namespace, key)),
            CatalogKind::Changed => proto::CatalogEvent::Changed(Self::entry(namespace, key)),
            CatalogKind::Removed => proto::CatalogEvent::Removed(key),
        };
        if sender.send(event).is_err() {
            self.catalogs.remove(&id);
        }
    }

    /// Recomputes chain-closed durability parents-first, emitting `changed` on flips.
    fn refresh_durability(&mut self, id: NamespaceId) {
        let namespace = &self.namespaces[&id];
        let mut keys: Vec<Key> = namespace.tree.nodes().map(|node| node.key).collect();
        keys.sort_unstable_by_key(|key| key.consumed_tokens);
        let mut flipped = Vec::new();
        for key in keys {
            let parent_ok = self.namespaces[&id].tree[&key]
                .parent
                .is_none_or(|parent| self.namespaces[&id].tree[&parent].location.durable);
            let node = &mut self.namespaces.get_mut(&id).unwrap().tree[&key];
            let durable = node.location.disk.is_some() && parent_ok;
            if node.location.durable != durable {
                node.location.durable = durable;
                flipped.push((key, durable));
            }
        }
        for (key, _) in flipped {
            self.delta(id, CatalogKind::Changed, key);
        }
    }

    /// The NVMe limit covers retained records and arena extents, unfinished disk
    /// records, and streamed puts' staging holds. Restore and export windows are
    /// bounded separately by `--staging`.
    fn charge_total(&self) -> u64 {
        self.charged.saturating_add(self.put_staging_held())
    }

    /// Staging held until each streamed Put and its disk writes finish.
    fn put_staging_held(&self) -> u64 {
        fn held(hold: &Option<Arc<Extent>>) -> u64 {
            hold.as_ref().map_or(0, |hold| hold.capacity() as u64)
        }
        let mut total = 0u64;
        for session in self.sessions.values() {
            for open in session.ops.values() {
                if let Operation::Put(put) = &open.operation {
                    total = total.saturating_add(held(&put.staging_hold));
                }
            }
        }
        total
    }

    /// Remember the most the limit's measure has been.
    fn note_reserved(&mut self) {
        let now = self.charge_total();
        self.reserved_peak = self.reserved_peak.max(now);
    }

    pub fn submit_put(
        &mut self,
        connection: u64,
        transfer: u64,
        request: PutRequest,
        events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
    ) {
        let queue = self.pending_puts.entry(connection).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(PendingPut {
            connection,
            transfer,
            request,
            events,
        });
        if was_empty {
            self.put_order.push_back(connection);
        }
    }

    pub fn submit_get(
        &mut self,
        connection: u64,
        transfer: u64,
        request: GetRequest,
        events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
    ) {
        let open = self
            .sessions
            .get(&connection)
            .map(Self::get_count)
            .unwrap_or(0);
        let queued = self
            .pending_gets
            .get(&connection)
            .map_or(0, |queue| queue.len());
        if open + queued >= OPEN_GETS {
            self.metrics.gets_error += 1;
            let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                reason: proto::Reason::Invalid as u8,
                detail: "open Get limit".into(),
            })));
            return;
        }
        let queue = self.pending_gets.entry(connection).or_default();
        let was_empty = queue.is_empty();
        queue.push_back(PendingGet {
            connection,
            transfer,
            request,
            events,
        });
        if was_empty {
            self.get_order.push_back(connection);
        }
    }

    fn get_count(session: &Session) -> usize {
        session
            .ops
            .values()
            .filter(|open| matches!(open.operation, Operation::Get(_)))
            .count()
    }

    fn pop_get(&mut self) -> Option<PendingGet> {
        while let Some(connection) = self.get_order.pop_front() {
            let Some(queue) = self.pending_gets.get_mut(&connection) else {
                continue;
            };
            let Some(pending) = queue.pop_front() else {
                self.pending_gets.remove(&connection);
                continue;
            };
            if queue.is_empty() {
                self.pending_gets.remove(&connection);
            } else {
                self.get_order.push_back(connection);
            }
            return Some(pending);
        }
        None
    }

    fn pop_put(&mut self) -> Option<PendingPut> {
        while let Some(connection) = self.put_order.pop_front() {
            let Some(queue) = self.pending_puts.get_mut(&connection) else {
                continue;
            };
            let Some(pending) = queue.pop_front() else {
                self.pending_puts.remove(&connection);
                continue;
            };
            if queue.is_empty() {
                self.pending_puts.remove(&connection);
            } else {
                self.put_order.push_back(connection);
            }
            return Some(pending);
        }
        None
    }

    #[cfg(test)]
    fn dispatch_trace(&mut self) -> Vec<&'static str> {
        let mut trace = Vec::new();
        let mut gets = 0;
        for _ in 0..4 {
            let Some(pending) = self.pop_get() else {
                break;
            };
            gets += 1;
            self.begin_get(pending);
        }
        trace.extend(std::iter::repeat_n("get", gets));
        if let Some(pending) = self.pop_put() {
            trace.push("put");
            self.begin_put(pending);
        }
        trace
    }

    fn dispatch(&mut self) {
        for _ in 0..4 {
            let Some(pending) = self.pop_get() else {
                break;
            };
            self.begin_get(pending);
        }
        if let Some(pending) = self.pop_put() {
            self.begin_put(pending);
        }
    }

    // ---------------------------------------------------------------- Put begin

    fn reject_put(&mut self, events: &tokio::sync::mpsc::Sender<Result<PutAnswer>>, reason: u8) {
        self.reject_put_detail(events, reason, &format!("reason {reason}"));
    }

    fn reject_put_detail(
        &mut self,
        events: &tokio::sync::mpsc::Sender<Result<PutAnswer>>,
        reason: u8,
        detail: &str,
    ) {
        let _ = events.try_send(Ok(PutAnswer::Refused(Refused {
            reason,
            detail: detail.to_string(),
        })));
    }

    /// A transfer id is unusable while any operation, queued request or drain with that
    /// (connection, transfer) is still alive, in either direction.
    fn transfer_in_use(&self, connection: u64, transfer: u64) -> bool {
        if self
            .sessions
            .get(&connection)
            .is_some_and(|session| session.ops.contains_key(&transfer))
        {
            return true;
        }
        if self
            .pending_puts
            .get(&connection)
            .is_some_and(|queue| queue.iter().any(|pending| pending.transfer == transfer))
            || self
                .pending_gets
                .get(&connection)
                .is_some_and(|queue| queue.iter().any(|pending| pending.transfer == transfer))
        {
            return true;
        }
        false
    }

    fn begin_put(&mut self, pending: PendingPut) {
        let PendingPut {
            connection,
            transfer,
            request,
            events,
        } = pending;
        let Some(id) = self.sessions.get(&connection).map(|_| request.namespace) else {
            return;
        };
        let Some(descriptor) = self
            .namespaces
            .get(&id)
            .map(|namespace| namespace.descriptor.clone())
        else {
            self.reject_put(&events, proto::Reason::Invalid as u8);
            return;
        };
        let invalid_geometry = request.key.consumed_tokens == 0
            || request.key.consumed_tokens / descriptor.page_tokens
                != u64::from(request.chain_pages)
            || request.published_tail_pages > request.chain_pages
            || (request.published_parent.is_none()
                && request.published_tail_pages != request.chain_pages)
            || request.published_parent.is_some_and(|parent| {
                parent.consumed_tokens >= request.key.consumed_tokens
                    || request.key.consumed_tokens / descriptor.page_tokens
                        - parent.consumed_tokens / descriptor.page_tokens
                        != u64::from(request.published_tail_pages)
            });
        if invalid_geometry {
            self.metrics.puts_error += 1;
            self.reject_put(&events, proto::Reason::Invalid as u8);
            return;
        }
        let initial_old = self.namespaces[&id].tree.node(&request.key).cloned();
        if self.namespaces[&id].inflight.contains(&request.key)
            || initial_old
                .as_ref()
                .is_some_and(|node| !node.row && node.pinned())
        {
            self.metrics.retry_inflight += 1;
            self.reject_put(&events, proto::Reason::Inflight as u8);
            return;
        }
        if initial_old.as_ref().is_some_and(|node| {
            node.row && (node.location.memory.is_some() || node.location.disk.is_some())
        }) {
            self.namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .touch_ancestors(&request.key);
            self.metrics.present += 1;
            self.reject_put(&events, proto::Reason::Present as u8);
            return;
        }
        if self.sessions[&connection].pid == 0
            && request.published_parent.is_some_and(|parent| {
                !self.namespaces[&id]
                    .tree
                    .node(&parent)
                    .is_some_and(|node| node.row)
            })
        {
            // A remote session names the exact ancestor its record continues, and the writer
            // stores that ancestor first. A local Put only hints at its parent and falls back
            // to a root below when the hint is unusable.
            self.metrics.puts_error += 1;
            self.reject_put_detail(
                &events,
                proto::Reason::Invalid as u8,
                "published parent is not stored",
            );
            return;
        }
        let initial_bucket = match &initial_old {
            Some(node) => {
                if node.bucket != request.bucket {
                    self.reject_put(&events, proto::Reason::Invalid as u8);
                    return;
                }
                node.bucket
            }
            None => request.bucket,
        };
        let initial_parent = match &initial_old {
            Some(node) => node.parent,
            None => request.published_parent.filter(|parent| {
                self.namespaces[&id]
                    .tree
                    .node(parent)
                    .is_some_and(|node| node.row)
            }),
        };
        let initial_first =
            initial_parent.map_or(0, |parent| parent.consumed_tokens / descriptor.page_tokens);
        let Some(initial_tail) = request
            .chain_pages
            .checked_sub(u32::try_from(initial_first).unwrap_or(u32::MAX))
        else {
            self.reject_put(&events, proto::Reason::Invalid as u8);
            return;
        };
        let child_header = match crate::record::new_header(
            &descriptor,
            request.key,
            initial_parent,
            initial_bucket,
            initial_tail,
        ) {
            Ok(header) => header,
            Err(_) => {
                self.reject_put(&events, proto::Reason::Invalid as u8);
                return;
            }
        };
        let root_header = match crate::record::new_header(
            &descriptor,
            request.key,
            None,
            initial_bucket,
            request.chain_pages,
        ) {
            Ok(header) => header,
            Err(_) => {
                self.reject_put(&events, proto::Reason::Invalid as u8);
                return;
            }
        };
        let rootless = self.config.root.is_none();
        let replacement = initial_old.is_some();
        let arena_bytes = self.memory.capacity() as u64;
        // The arena rounds every record extent up to a power of two. Without a
        // root there is no disk to stream to, so a record whose own extent
        // cannot fit the arena can never be stored. Whether it fits beside
        // what else must stay resident is a capacity question.
        if rootless && child_header.length_with_row().max(4096).next_power_of_two() > arena_bytes {
            self.metrics.oversize += 1;
            self.reject_put(&events, proto::Reason::Oversize as u8);
            return;
        }
        let selected_pins: Vec<Key> = initial_parent
            .into_iter()
            .chain(replacement.then_some(request.key))
            .collect();
        // Admission charges the durable budget with the shape's full
        // reservation: the rounded extent for a record the arena keeps, or
        // for a streamed record its file and
        // its mandatory two-window staging hold. Only the record's own charge
        // can make it OVERSIZE; a shortage caused by what else must stay
        // resident is a Budget refusal.
        let extent = child_header.length_with_row().max(4096).next_power_of_two();
        let charge = if extent <= arena_bytes {
            extent
        } else {
            self.put_requirement(&child_header, 0).charge
        };
        if !rootless && charge > self.config.nvme {
            self.metrics.oversize += 1;
            self.reject_put(&events, proto::Reason::Oversize as u8);
            return;
        }
        // Pick the shape by asking the arena allocator. A failed allocation
        // falls back to streaming; a rootless manager instead tries to evict
        // the parent and store the root. `memory` holds the extent the shape
        // was admitted with.
        let selected = self.put_requirement(&child_header, arena_bytes);
        let mut memory: Option<Arc<Extent>> = None;
        let (header, parent, pins, requirement): (Header, Option<Key>, Vec<Key>, PutRequirement) =
            if !selected.streaming {
                match self.memory_room_excluding(
                    child_header.length_with_row() as usize,
                    !rootless,
                    &selected_pins,
                ) {
                    Some(extent) => {
                        let requirement = PutRequirement::arena(extent.capacity() as u64);
                        memory = Some(Arc::new(extent));
                        (child_header, initial_parent, selected_pins, requirement)
                    }
                    None if rootless => {
                        let root_pins: Vec<Key> =
                            replacement.then_some(request.key).into_iter().collect();
                        match self.rootless_fallback(&root_header, root_pins, arena_bytes) {
                            RootChoice::Store(extent) => {
                                let requirement = PutRequirement::arena(extent.capacity() as u64);
                                memory = Some(Arc::new(extent));
                                let root_pins: Vec<Key> =
                                    replacement.then_some(request.key).into_iter().collect();
                                (root_header, None, root_pins, requirement)
                            }
                            RootChoice::Wait => {
                                self.metrics.retry_budget += 1;
                                self.reject_put(&events, proto::Reason::Budget as u8);
                                return;
                            }
                        }
                    }
                    None => {
                        let requirement = self.put_requirement(&child_header, 0);
                        (child_header, initial_parent, selected_pins, requirement)
                    }
                }
            } else {
                let requirement = self.put_requirement(&child_header, 0);
                (child_header, initial_parent, selected_pins, requirement)
            };
        let length = header.length_with_row();
        // A streamed record reserves two windows in staging. When that hold
        // cannot fit the whole region it can never be admitted: if the record
        // has no arena shape either, refuse it as OVERSIZE. A record that still
        // fits an arena extent can be tried again after space is freed.
        if requirement.streaming && requirement.allocation > self.staging.capacity() as u64 {
            if selected.streaming {
                self.metrics.oversize += 1;
                self.reject_put(&events, proto::Reason::Oversize as u8);
                return;
            }
            self.metrics.retry_budget += 1;
            self.reject_put(&events, proto::Reason::Budget as u8);
            return;
        }
        for key in &pins {
            if self.namespaces.get_mut(&id).unwrap().tree.pin(key).is_err() {
                for key in &pins {
                    let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
                }
                self.reject_put(&events, proto::Reason::Invalid as u8);
                return;
            }
        }
        if !self.make_room_with(Some(id), requirement.charge, &pins) {
            drop(memory);
            for key in &pins {
                let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
            }
            self.metrics.retry_budget += 1;
            self.reject_put(&events, proto::Reason::Budget as u8);
            return;
        }
        let staging_hold = if requirement.streaming {
            match self.staging.allocate(requirement.allocation as usize) {
                Some(extent) => Some(Arc::new(extent)),
                None => {
                    drop(memory);
                    for key in &pins {
                        let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
                    }
                    self.metrics.retry_budget += 1;
                    self.reject_put(&events, proto::Reason::Budget as u8);
                    return;
                }
            }
        } else {
            None
        };
        let temporary = if requirement.streaming {
            self.job_counter += 1;
            let job = self.job_counter;
            match self.namespaces[&id].temporary(job) {
                Some(extent) => Some(extent),
                None => {
                    drop(memory);
                    drop(staging_hold);
                    for key in &pins {
                        let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(key);
                    }
                    self.reject_put(&events, proto::Reason::Io as u8);
                    return;
                }
            }
        } else {
            None
        };
        self.charged += requirement.reservation;
        // The hold is allocated, and counts from here.
        let held_now = staging_hold
            .as_ref()
            .map_or(0, |hold| hold.capacity() as u64);
        self.reserved_peak = self
            .reserved_peak
            .max(self.charge_total().saturating_add(held_now));
        let first =
            parent.map_or(0, |parent| parent.consumed_tokens / descriptor.page_tokens) as u32;
        let grants = record_grants(&header, first, 0, true, DEFAULT_MAX_WINDOW, 0);
        let pages = vec![None; header.tail_pages as usize];
        let mut put = Put {
            namespace: id,
            header: header.clone(),
            first_page: first,
            grants,
            requested: 0,
            windows: FilledWindows {
                granted: HashMap::new(),
                next: 0,
                pages,
                row: None,
                pages_filled: 0,
                row_filled: header.row_record_bytes() == 0,
            },
            memory,
            temporary,
            streaming: requirement.streaming,
            commit_bytes: Vec::new(),
            metadata: None,
            committed: false,
            finalizing: false,
            cancelled: false,
            terminal: false,
            finished: false,
            pending: 0,
            blocked: VecDeque::new(),
            pins: pins.into(),
            reservation: requirement.reservation,
            staging_hold,
            hold_unit: requirement.unit as usize,
            hold_slots: [None, None],
            events,
            outbox: VecDeque::new(),
        };
        let namespace = self.namespaces.get_mut(&id).unwrap();
        namespace.inflight.insert(request.key);
        put.outbox.push_back(Ok(PutAnswer::Accept(PutAccept {
            transfer,
            first_page: first,
            parent,
            record_bytes: length,
            windows: put.grants.len(),
        })));
        self.sessions.get_mut(&connection).unwrap().ops.insert(
            transfer,
            Open {
                progress: Instant::now(),
                operation: Operation::Put(Box::new(put)),
            },
        );
    }

    fn oldest(
        &self,
        eligible: impl Fn(&flare_selene::shared::index::Node<Location>) -> bool,
    ) -> Option<(NamespaceId, Key)> {
        self.namespaces
            .iter()
            .filter_map(|(id, namespace)| {
                namespace
                    .tree
                    .ordered()
                    .find(|node| {
                        !node.pinned()
                            && !node.location.writing
                            && !node.location.reading()
                            && eligible(node)
                    })
                    .map(move |node| (node.accessed(), node.key, *id))
            })
            .min_by_key(|(accessed, key, _)| (*accessed, key.consumed_tokens))
            .map(|(_, key, id)| (id, key))
    }

    fn memory_room(&mut self, bytes: usize, require_durable: bool) -> Option<Extent> {
        self.memory_room_excluding(bytes, require_durable, &[])
    }

    /// Attempt the arena allocation, evicting eligible leaves one at a time
    /// until it succeeds. Asking the allocator rather than summing free bytes
    /// handles buddy fragmentation and chains whose parent only becomes a
    /// leaf after its child is evicted. `excluded` keys must survive.
    fn memory_room_excluding(
        &mut self,
        bytes: usize,
        require_durable: bool,
        excluded: &[Key],
    ) -> Option<Extent> {
        loop {
            if let Some(extent) = self.memory.allocate(bytes) {
                return Some(extent);
            }
            let (id, key) = self.oldest(|node| {
                Self::memory_evictable(node, require_durable) && !excluded.contains(&node.key)
            })?;
            if !self.evict_memory(id, key) {
                return None;
            }
        }
    }

    /// Shared eligibility for reclaiming an arena extent. A rooted manager may
    /// only drop records that have a disk copy; a rootless one may drop true
    /// leaves outright.
    fn memory_evictable(
        node: &flare_selene::shared::index::Node<Location>,
        require_durable: bool,
    ) -> bool {
        node.location.memory.is_some()
            && (!require_durable || node.location.durable)
            && (if require_durable {
                node.children() == 0 || (node.row && node.children() == 1)
            } else {
                node.children() == 0
            })
    }

    /// Try the whole chain when the arena cannot hold the child beside its parent.
    fn rootless_fallback(
        &mut self,
        root: &Header,
        root_pins: Vec<Key>,
        arena_bytes: u64,
    ) -> RootChoice {
        let requirement = self.put_requirement(root, arena_bytes);
        if requirement.streaming {
            return RootChoice::Wait;
        }
        match self.memory_room_excluding(root.length_with_row() as usize, false, &root_pins) {
            Some(extent) => RootChoice::Store(extent),
            None => RootChoice::Wait,
        }
    }

    /// Returns whether the candidate was actually evicted. A node rejected by
    /// `Tree::remove` (because it gained children) is left untouched.
    fn evict_memory(&mut self, id: NamespaceId, key: Key) -> bool {
        let (disk, old) = {
            let node = &mut self.namespaces.get_mut(&id).unwrap().tree[&key];
            (node.location.disk.is_some(), node.location.bytes)
        };
        if disk {
            let node = &mut self.namespaces.get_mut(&id).unwrap().tree[&key];
            node.location.memory = None;
            node.location.bytes = record_charge(&node.location.header, false);
            self.charged = self.charged - old + node.location.bytes;
            self.delta(id, CatalogKind::Changed, key);
            return true;
        }
        let removed = self.evict_accounted(id, key);
        self.metrics.evictions += removed;
        removed != 0
    }

    /// The physical requirement for this Put shape. A record whose
    /// buddy-rounded extent fits `arena_bytes` is stored in one arena extent;
    /// otherwise it streams to disk behind a fixed staging hold of two of its
    /// largest windows. `arena_bytes` is the arena capacity for the shape
    /// decision; admission confirms it by allocating.
    fn put_requirement(&self, header: &Header, arena_bytes: u64) -> PutRequirement {
        let length = header.length_with_row();
        let extent = length.max(4096).next_power_of_two();
        if extent <= arena_bytes {
            return PutRequirement::arena(extent);
        }
        let first = header
            .parent
            .map_or(0, |parent| parent.consumed_tokens / header.page_tokens)
            as u32;
        let unit = record_grants(header, first, 0, true, DEFAULT_MAX_WINDOW, 0)
            .iter()
            .map(|grant| grant.length.max(4096).next_power_of_two())
            .max()
            .unwrap_or(4096);
        PutRequirement::streaming(length, unit)
    }

    /// Reserve budget space while the shape's protected keys stay put, so an
    /// eviction cannot invalidate the shape the reservation was computed for.
    fn make_room_with(
        &mut self,
        namespace: Option<NamespaceId>,
        bytes: u64,
        protected: &[Key],
    ) -> bool {
        if self.config.root.is_none() {
            return true;
        }
        let mut pinned = Vec::new();
        if let Some(id) = namespace {
            for key in protected {
                if let Some(namespace) = self.namespaces.get_mut(&id) {
                    if namespace.tree.pin(key).is_ok() {
                        pinned.push(*key);
                    }
                }
            }
        }
        let result = self.make_room_evict(bytes);
        if let Some(id) = namespace {
            for key in pinned {
                let _ = self.namespaces.get_mut(&id).unwrap().tree.unpin(&key);
            }
        }
        result
    }

    fn make_room_evict(&mut self, bytes: u64) -> bool {
        if self.config.root.is_none() {
            return true;
        }
        let mut compact = true;
        while self.charge_total().saturating_add(bytes) > self.config.nvme {
            let candidate = self.oldest(|node| {
                node.location.durable
                    && (node.children() == 0 || (compact && node.row && node.children() == 1))
            });
            let Some((id, key)) = candidate else {
                return false;
            };
            let namespace = self.namespaces.get_mut(&id).unwrap();
            match namespace.tree.evict(true, |node| node.key == key) {
                Some(Eviction::Leaves(nodes)) => {
                    for node in nodes {
                        self.metrics.evictions += 1;
                        if let Some(extent) = node.location.disk {
                            extent.retire();
                        }
                        self.charged -= node.location.bytes;
                        self.delta(id, CatalogKind::Removed, node.key);
                    }
                }
                Some(Eviction::Compacted(key)) => {
                    let node = &self.namespaces[&id].tree[&key];
                    let Some(source) = node.location.disk.clone() else {
                        return false;
                    };
                    let header = node.location.header.clone();
                    let freed = node.location.bytes.saturating_sub(header.row_segment);
                    self.job_counter += 1;
                    let job = self.job_counter;
                    let task = DiskJob::Truncate {
                        id: JobId(job),
                        path: source.path.clone(),
                        length: header.row_segment,
                    };
                    let Some(worker) = &self.namespaces[&id].disk else {
                        return false;
                    };
                    match worker.submit(task) {
                        Ok(()) => {
                            self.jobs.insert(
                                job,
                                Job::Compact {
                                    namespace: id,
                                    key,
                                    bytes: freed,
                                },
                            );
                            let node = &mut self.namespaces.get_mut(&id).unwrap().tree[&key];
                            node.location.writing = true;
                            node.location.job = Some(job);
                            self.namespaces
                                .get_mut(&id)
                                .unwrap()
                                .tree
                                .pin(&key)
                                .unwrap();
                            self.metrics.compactions += 1;
                            self.delta(id, CatalogKind::Changed, key);
                        }
                        Err(_) => {
                            self.namespaces.get_mut(&id).unwrap().tree[&key].row = true;
                            compact = false;
                        }
                    }
                    // Compaction frees bytes asynchronously. Keep looking for
                    // leaves that can make room for this admission now.
                }
                None => return false,
            }
        }
        true
    }

    // ----------------------------------------------------------- Window grants

    fn pump_grants(&mut self) {
        let mut ops: Vec<_> = self
            .sessions
            .iter()
            .flat_map(|(connection, session)| {
                session.ops.iter().map(move |(transfer, open)| {
                    (
                        matches!(open.operation, Operation::Put(_)),
                        *connection,
                        *transfer,
                    )
                })
            })
            .collect();
        ops.sort_unstable();
        for (put, connection, transfer) in ops {
            if put {
                while self.send_put_grant(connection, transfer) {}
            } else {
                self.send_get_meta(connection, transfer);
                while self.send_get_grant(connection, transfer) {}
            }
        }
    }

    fn send_put_grant(&mut self, connection: u64, transfer: u64) -> bool {
        let prepared = {
            let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
                return false;
            };
            if put.cancelled
                || put.terminal
                || put.requested == 0
                || put.windows.next >= put.grants.len()
            {
                return false;
            }
            let index = put.windows.next as u32;
            let plan = put.grants[put.windows.next];
            let memory = put.memory.clone();
            let hold = put.staging_hold.clone();
            let hold_unit = put.hold_unit;
            let slot = if memory.is_some() {
                None
            } else {
                put.hold_slots.iter().position(Option::is_none)
            };
            (index, plan, memory, hold, hold_unit, slot)
        };
        let (index, plan, memory, hold, hold_unit, slot) = prepared;
        if memory.is_none() && slot.is_none() {
            // Both hold slots are still live; retry once one is released.
            return false;
        }
        let (region, extent, within) = match &memory {
            Some(memory) => (Region::Arena, memory.clone(), plan.record_offset as usize),
            None => {
                let Some(hold) = hold else {
                    self.fail_put(
                        connection,
                        transfer,
                        proto::Reason::Invalid as u8,
                        invalid("missing staging hold"),
                    );
                    return false;
                };
                (Region::Staging, hold, slot.unwrap() * hold_unit)
            }
        };
        let result = self.local.offset(region, &extent, within, plan.length);
        match result {
            Ok(offset) => {
                if let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer) {
                    put.windows.next += 1;
                    put.requested -= 1;
                    if let Some(slot) = slot {
                        put.hold_slots[slot] = Some(HoldSlot { index, job: None });
                    }
                    put.windows.granted.insert(index, plan);
                    put.outbox.push_back(Ok(PutAnswer::Grant(
                        plan.window(transfer, index, region, offset, extent),
                    )));
                    *progress = Instant::now();
                }
                true
            }
            Err(error) => {
                self.fail_put(connection, transfer, proto::Reason::Invalid as u8, error);
                false
            }
        }
    }

    fn fail_put(&mut self, connection: u64, transfer: u64, reason: u8, error: io::Error) {
        self.metrics.puts_error += 1;
        if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
            put.terminal = true;
            // Ownership is ending; never submit unsent chunks afterwards.
            put.blocked.clear();
            put.outbox.push_back(Ok(PutAnswer::Refused(Refused {
                reason,
                detail: error.to_string(),
            })));
            if !put.finished && put.pending == 0 {
                put.finished = true;
                let id = put.namespace;
                let pins = std::mem::take(&mut put.pins);
                let key = put.header.key;
                let reservation = put.reservation;
                let namespace = self.namespaces.get_mut(&id).unwrap();
                for key in &pins {
                    let _ = namespace.tree.unpin(key);
                }
                namespace.inflight.remove(&key);
                self.charged = self.charged.saturating_sub(reservation);
            }
        }
    }

    fn send_get_meta(&mut self, connection: u64, transfer: u64) -> bool {
        let mut progressed = false;
        loop {
            let Some((get, progress)) = get_mut(&mut self.sessions, connection, transfer) else {
                return progressed;
            };
            if get.cancelled || get.terminal || get.next_meta >= get.records.len() {
                return progressed;
            }
            let record = get.next_meta as u32;
            if get.records[get.next_meta].meta_sent {
                get.next_meta += 1;
                continue;
            }
            let Some(metadata) = get.records[get.next_meta].meta.clone() else {
                return progressed;
            };
            let source = if get.records[get.next_meta].memory.is_some() {
                flare_selene::shared::Tier::Dram as i32
            } else {
                flare_selene::shared::Tier::Nvme as i32
            };
            let header = match Header::decode(&metadata) {
                Ok(header) => header,
                Err(error) => {
                    self.fail_get(
                        connection,
                        transfer,
                        proto::Reason::Io as u8,
                        invalid(error.to_string()),
                    );
                    return progressed;
                }
            };
            let Some(table) = metadata.get(4096..4096 + header.tail_pages as usize * 8) else {
                self.fail_get(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("incomplete checksum table"),
                );
                return progressed;
            };
            let pages = table
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            get.outbox.push_back(Ok(GetAnswer::Meta {
                record,
                pages,
                row: header.row_checksum,
                source,
            }));
            get.records[get.next_meta].meta_sent = true;
            get.next_meta += 1;
            *progress = Instant::now();
            progressed = true;
        }
    }

    fn send_get_grant(&mut self, connection: u64, transfer: u64) -> bool {
        let Some(ready) = get_ref(&self.sessions, connection, transfer).map(|get| {
            (
                get.namespace,
                get.next_grant,
                get.grants.get(get.next_grant).copied(),
            )
        }) else {
            return false;
        };
        let (namespace, position, Some(plan)) = ready else {
            return false;
        };
        enum Next {
            Stop,
            Fail(proto::Reason, io::Error),
            Lease { memory: Arc<Extent> },
            Read { file: Arc<File> },
        }
        let action = {
            let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                return false;
            };
            if get.cancelled
                || get.terminal
                || get.requested == 0
                || !get.records[plan.record as usize].meta_sent
            {
                Next::Stop
            } else {
                let record = &get.records[plan.record as usize];
                if let Some(memory) = &record.memory {
                    Next::Lease {
                        memory: memory.clone(),
                    }
                } else if let Some(file) = &record.file {
                    Next::Read { file: file.clone() }
                } else {
                    Next::Fail(proto::Reason::Io, invalid("record file closed"))
                }
            }
        };
        let schedule = match action {
            Next::Stop => return false,
            Next::Fail(reason, error) => {
                self.fail_get(connection, transfer, reason as u8, error);
                return false;
            }
            Next::Lease { memory } => {
                let offset = match self.local.offset(
                    Region::Arena,
                    &memory,
                    plan.record_offset as usize,
                    plan.length,
                ) {
                    Ok(offset) => offset,
                    Err(error) => {
                        self.fail_get(connection, transfer, proto::Reason::Invalid as u8, error);
                        return false;
                    }
                };
                let Some((get, progress)) = get_mut(&mut self.sessions, connection, transfer)
                else {
                    return false;
                };
                get.next_grant += 1;
                get.requested -= 1;
                get.outbox.push_back(Ok(GetAnswer::Ready(plan.window(
                    transfer,
                    position as u32,
                    Region::Arena,
                    offset,
                    memory,
                ))));
                *progress = Instant::now();
                return true;
            }
            Next::Read { file } => {
                let length = usize::try_from(plan.length).unwrap_or(usize::MAX);
                // Restore windows are transient staging traffic: reads do not
                // grow the durable set, so they are bounded by the staging
                // region alone and never by the NVMe budget.
                let Some(extent) = self.staging.allocate(length) else {
                    self.fail_get(
                        connection,
                        transfer,
                        proto::Reason::Budget as u8,
                        invalid("staging budget"),
                    );
                    return false;
                };
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.next_grant += 1;
                    get.requested -= 1;
                    get.window_reads += 1;
                    get.pending_reads += 1;
                }
                Some((namespace, file, extent))
            }
        };
        let Some((namespace, file, extent)) = schedule else {
            self.fail_get(
                connection,
                transfer,
                proto::Reason::Io as u8,
                invalid("record file closed"),
            );
            return false;
        };
        self.job_counter += 1;
        let job = self.job_counter;
        let task = DiskJob::ReadRecord {
            id: JobId(job),
            file,
            span: Span {
                address: extent.bytes().as_ptr() as usize,
                offset: plan.record_offset,
                length: extent.bytes().len(),
            },
        };
        self.jobs.insert(
            job,
            Job::GetRead {
                connection,
                transfer,
                record: plan.record,
                kind: ReadKind::Window {
                    grant: position as u32,
                },
                buffer: extent,
            },
        );
        self.submit_read(namespace, job, task);
        true
    }

    fn fail_get(&mut self, connection: u64, transfer: u64, reason: u8, error: io::Error) {
        self.metrics.gets_error += 1;
        if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
            get.terminal = true;
            get.cancelled = true;
            get.outbox.push_back(Ok(GetAnswer::Refused(Refused {
                reason,
                detail: error.to_string(),
            })));
        }
    }

    fn submit_read(&mut self, namespace: NamespaceId, job: u64, task: DiskJob) {
        use std::sync::mpsc::TrySendError::{Disconnected, Full};
        let result = match &self.namespaces[&namespace].reads {
            Some(worker) => worker.try_submit(task),
            None => Err(Disconnected(task)),
        };
        match result {
            Ok(()) => {}
            Err(Full(task)) => self.read_backlog.push_back((namespace, job, task)),
            Err(Disconnected(_)) => {
                if let Some(Job::GetRead { connection, transfer, .. }) = self.jobs.remove(&job) {
                    self.read_failed(connection, transfer);
                }
            }
        }
    }

    // --------------------------------------------------------------- Put frames

    fn window_filled(
        &mut self,
        connection: u64,
        transfer: u64,
        index: u32,
        page_xxh3: Vec<u64>,
        row_xxh3: Option<u64>,
    ) -> Result<()> {
        struct Prepared {
            plan: PlannedGrant,
            memory: Option<Arc<Extent>>,
            hold: Option<Arc<Extent>>,
            within: usize,
            header: Header,
            first_page: u32,
            temporary: Option<Arc<DiskExtent>>,
            streaming: bool,
        }
        let prepared = {
            let Some(open) = self
                .sessions
                .get(&connection)
                .and_then(|session| session.ops.get(&transfer))
            else {
                return Err(invalid("unknown Put transfer"));
            };
            let Operation::Put(put) = &open.operation else {
                return Err(invalid("transfer is not a Put"));
            };
            if put.cancelled || put.terminal {
                return Ok(());
            }
            let slot = put
                .hold_slots
                .iter()
                .position(|slot| slot.is_some_and(|slot| slot.index == index));
            put.windows
                .granted
                .get(&index)
                .copied()
                .map(|plan| Prepared {
                    plan,
                    memory: put.memory.clone(),
                    hold: put.staging_hold.clone(),
                    within: slot.map_or(0, |slot| slot * put.hold_unit),
                    header: put.header.clone(),
                    first_page: put.first_page,
                    temporary: put.temporary.clone(),
                    streaming: put.streaming,
                })
        };
        let Some(prepared) = prepared else {
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("filled for an ungranted window"),
            );
            return Ok(());
        };
        let plan = prepared.plan;
        if page_xxh3.len() != plan.pages as usize
            || (plan.row && row_xxh3.is_none())
            || (!plan.row && row_xxh3.is_some())
            || plan.pages > 64
        {
            self.abort_put_window(connection, transfer, index);
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("window hash shape"),
            );
            return Ok(());
        }
        // The window bytes stay in the arena extent or staging hold; a chunk
        // keeps this slice alive until its write completes, so no copy is
        // buffered in the manager.
        let data = if let Some(memory) = &prepared.memory {
            ExtentSlice::new(
                memory.clone(),
                plan.record_offset as usize,
                plan.length as usize,
            )
        } else if let Some(hold) = &prepared.hold {
            ExtentSlice::new(hold.clone(), prepared.within, plan.length as usize)
        } else {
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("window vanished"),
            );
            return Ok(());
        };
        let bytes = data.bytes();
        let page_bytes = prepared.header.page_record_bytes() as usize;
        let mut verified = Vec::with_capacity(page_xxh3.len());
        for (position, hash) in page_xxh3.iter().enumerate() {
            let start = position * page_bytes;
            if flare_selene::shared::record::checksum(&bytes[start..start + page_bytes]) != *hash {
                self.abort_put_window(connection, transfer, index);
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("page checksum"),
                );
                return Ok(());
            }
            verified.push(*hash);
        }
        if plan.row {
            let row_bytes = prepared.header.row_record_bytes() as usize;
            let start = bytes.len() - row_bytes;
            if flare_selene::shared::record::checksum(&bytes[start..]) != row_xxh3.unwrap() {
                self.abort_put_window(connection, transfer, index);
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("row checksum"),
                );
                return Ok(());
            }
        }
        {
            let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer) else {
                return Ok(());
            };
            put.windows.granted.remove(&index);
            // The slot stays occupied until the chunk write drains.
            for (position, hash) in verified.iter().enumerate() {
                let relative =
                    (plan.page_first - u64::from(prepared.first_page)) as usize + position;
                put.windows.pages[relative] = Some(*hash);
                put.windows.pages_filled += 1;
            }
            if plan.row {
                put.windows.row = row_xxh3;
                put.windows.row_filled = true;
            }
            *progress = Instant::now();
        }
        if prepared.streaming {
            let Some(temporary) = prepared.temporary else {
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("missing temporary file"),
                );
                return Ok(());
            };
            let Some(namespace) =
                put_ref(&self.sessions, connection, transfer).map(|put| put.namespace)
            else {
                return Ok(());
            };
            self.job_counter += 1;
            let job = self.job_counter;
            if self.namespaces[&namespace].disk.is_none() {
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("no disk"),
                );
                return Ok(());
            }
            match self.write_chunk(
                namespace,
                job,
                connection,
                transfer,
                temporary,
                plan.record_offset,
                data,
            ) {
                Ok(()) => {
                    if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
                        put.pending += 1;
                        set_slot_job(put, index, job);
                    }
                    #[cfg(any(test, feature = "test-support"))]
                    self.config.faults.checkpoint("data");
                }
                Err(data) => {
                    if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
                        put.blocked.push_back((job, plan.record_offset, data));
                        set_slot_job(put, index, job);
                    }
                }
            }
        }
        Ok(())
    }

    fn abort_put_window(&mut self, connection: u64, transfer: u64, index: u32) {
        if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) {
            if let Some(slot) = put
                .hold_slots
                .iter_mut()
                .find(|slot| slot.is_some_and(|slot| slot.index == index))
            {
                *slot = None;
            }
        }
    }

    fn put_commit(
        &mut self,
        connection: u64,
        transfer: u64,
        metadata_offset: u64,
        metadata: Vec<u8>,
    ) -> Result<()> {
        enum Action {
            None,
            Fail(proto::Reason, io::Error),
            Publish,
        }
        let namespace = put_ref(&self.sessions, connection, transfer).map(|put| put.namespace);
        let Some(namespace) = namespace else {
            return Err(invalid("unknown Put transfer"));
        };
        let descriptor = self.namespaces[&namespace].descriptor.clone();
        let action = {
            let Some(open) = self
                .sessions
                .get_mut(&connection)
                .and_then(|session| session.ops.get_mut(&transfer))
            else {
                return Err(invalid("unknown Put transfer"));
            };
            let Operation::Put(put) = &mut open.operation else {
                return Err(invalid("transfer is not a Put"));
            };
            if put.cancelled || put.terminal || put.committed {
                return Ok(());
            }
            if !put.owns_payload() {
                Action::Fail(
                    proto::Reason::Invalid,
                    invalid("commit before every payload range"),
                )
            } else {
                let header_page_segment = put.header.page_segment as usize;
                if metadata_offset as usize != put.commit_bytes.len()
                    || metadata.is_empty()
                    || !metadata.len().is_multiple_of(4096)
                    || put.commit_bytes.len() + metadata.len() > header_page_segment
                    || !(metadata_offset as usize).is_multiple_of(4096)
                {
                    Action::Fail(proto::Reason::Invalid, invalid("commit offset or length"))
                } else {
                    put.commit_bytes.extend_from_slice(&metadata);
                    if put.commit_bytes.len() < header_page_segment {
                        Action::None
                    } else if let Err(error) = validate_commit(put, &descriptor) {
                        Action::Fail(proto::Reason::Invalid, error)
                    } else {
                        if let Some(memory) = &mut put.memory {
                            let checksum_segment =
                                put.commit_bytes[4096..header_page_segment].to_vec();
                            let bytes = Arc::get_mut(memory).unwrap().bytes_mut();
                            bytes[..4096].copy_from_slice(&put.commit_bytes[..4096]);
                            bytes[4096..header_page_segment].copy_from_slice(&checksum_segment);
                            put.committed = true;
                            Action::Publish
                        } else {
                            match Arena::new(header_page_segment.next_power_of_two()) {
                                Ok(arena) => {
                                    let mut metadata = arena.allocate(header_page_segment).unwrap();
                                    metadata.bytes_mut().copy_from_slice(&put.commit_bytes);
                                    put.metadata = Some(Arc::new(metadata));
                                    put.commit_bytes = Vec::new();
                                    put.committed = true;
                                    Action::None
                                }
                                Err(error) => Action::Fail(proto::Reason::Io, error),
                            }
                        }
                    }
                }
            }
        };
        match action {
            Action::None => {}
            Action::Fail(reason, error) => {
                self.fail_put(connection, transfer, reason as u8, error);
            }
            Action::Publish => {
                self.publish_put(connection, transfer, None, false);
            }
        }
        if let Some(open) = self
            .sessions
            .get_mut(&connection)
            .and_then(|session| session.ops.get_mut(&transfer))
        {
            open.progress = Instant::now();
        }
        Ok(())
    }

    fn publish_target(&self, id: NamespaceId, header: &Header) -> PublishTarget {
        let namespace = &self.namespaces[&id];
        let Some(_) = namespace.tree.node(&header.key) else {
            return PublishTarget::Insert;
        };
        PublishTarget::Replace
    }

    fn publish_put(
        &mut self,
        connection: u64,
        transfer: u64,
        disk: Option<Arc<DiskExtent>>,
        durable: bool,
    ) {
        let Some((id, header)) = put_ref(&self.sessions, connection, transfer)
            .filter(|put| !put.finished)
            .map(|put| (put.namespace, put.header.clone()))
        else {
            return;
        };
        let target = self.publish_target(id, &header);
        let prepared = {
            let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
                return;
            };
            if put.finished {
                return;
            }
            let header = put.header.clone();
            let memory = put.memory.clone();
            let bytes = record_charge(&header, memory.is_some());
            let reservation = put.reservation;
            let pins = std::mem::take(&mut put.pins);
            let record_bytes = header.length_with_row();
            let key = header.key;
            let already_terminal = put.terminal;
            put.terminal = true;
            (
                header,
                memory,
                bytes,
                reservation,
                pins,
                key,
                already_terminal,
                record_bytes,
            )
        };
        let (header, memory, bytes, reservation, pins, key, already_terminal, record_bytes) =
            prepared;
        let replace = matches!(target, PublishTarget::Replace);
        let location = Location {
            header: header.clone(),
            memory,
            disk,
            durable: false,
            writing: false,
            job: None,
            retry: Instant::now(),
            bytes,
        };
        let inserted = {
            let namespace = self.namespaces.get_mut(&id).unwrap();
            let inserted = if replace {
                let node = namespace.tree.node_mut(&key).unwrap();
                let old = std::mem::replace(&mut node.location, location);
                node.row = true;
                self.retired_records.push((id, key, old));
                true
            } else {
                namespace
                    .tree
                    .insert(
                        key,
                        header.parent,
                        header.bucket,
                        header.tail_pages as usize,
                        true,
                        true,
                        location,
                    )
                    .is_ok()
            };
            for key in &pins {
                let _ = namespace.tree.unpin(key);
            }
            namespace.inflight.remove(&key);
            inserted
        };
        if !inserted {
            self.fail_put(
                connection,
                transfer,
                proto::Reason::Invalid as u8,
                invalid("publish insert"),
            );
            return;
        }
        if let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer)
            && !put.finished
        {
            put.finished = true;
            put.pending = 0;
            if !already_terminal {
                put.outbox.push_back(Ok(PutAnswer::Stored {
                    bytes: record_bytes,
                }));
            }
        }
        self.reused(id, key);
        self.charged = self.charged.saturating_sub(reservation) + bytes;
        if durable {
            if let Some(node) = self.namespaces.get_mut(&id).unwrap().tree.node_mut(&key) {
                node.location.durable = true;
            }
        }
        self.refresh_durability(id);
        self.metrics.puts_ok += 1;
        self.delta(
            id,
            if replace {
                CatalogKind::Changed
            } else {
                CatalogKind::Added
            },
            key,
        );
    }

    fn begin_get(&mut self, pending: PendingGet) {
        let PendingGet {
            connection,
            transfer,
            request,
            events,
        } = pending;
        let Some(id) = self.sessions.get(&connection).map(|_| request.namespace) else {
            return;
        };
        let parsed = self.plan_get(id, &request);
        let (records, grants) = match parsed {
            Ok(value) => value,
            Err(error) => {
                self.metrics.gets_error += 1;
                let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                    reason: proto::Reason::Invalid as u8,
                    detail: error.to_string(),
                })));
                return;
            }
        };
        let pages: Vec<Key> = request.keys.clone();
        {
            let namespace = self.namespaces.get_mut(&id).unwrap();
            for key in &pages {
                if namespace.tree.pin(key).is_err() {
                    let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                        reason: proto::Reason::Invalid as u8,
                        detail: "pin".into(),
                    })));
                    return;
                }
            }
        }
        for key in pages.iter().rev().skip(1).rev() {
            self.clock += 1;
            self.namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .access_at(key, self.clock);
        }
        if let Some(key) = pages.last() {
            self.reused(id, *key);
        }
        let get = Get {
            namespace: id,
            pins: pages.into(),
            records,
            grants,
            requested: 0,
            next_grant: 0,
            next_meta: 0,
            window_reads: 0,
            pending_reads: 0,
            cancelled: false,
            terminal: false,
            finished: false,
            events,
            outbox: VecDeque::new(),
        };
        let mut get = get;
        get.outbox.push_back(Ok(GetAnswer::Leased {
            transfer,
            windows: get.grants.len(),
        }));
        self.sessions.get_mut(&connection).unwrap().ops.insert(
            transfer,
            Open {
                progress: Instant::now(),
                operation: Operation::Get(Box::new(get)),
            },
        );
        self.prepare_get_records(connection, transfer, id);
    }

    fn plan_get(
        &mut self,
        id: NamespaceId,
        request: &GetRequest,
    ) -> Result<(Vec<GetRecord>, Vec<PlannedGrant>)> {
        if request.keys.is_empty() || request.keys.len() > MAX_CHAIN || !request.row {
            return Err(invalid("Get node count or row"));
        }
        let namespace = self
            .namespaces
            .get(&id)
            .ok_or_else(|| invalid("Get namespace is not registered"))?;
        let mut total = 0u64;
        let mut records = Vec::new();
        let mut grants = Vec::new();
        for (index, key) in request.keys.iter().enumerate() {
            let node = namespace
                .tree
                .node(key)
                .ok_or_else(|| invalid("Get key missing"))?;
            if (index > 0 && node.parent != Some(request.keys[index - 1]))
                || (index + 1 == request.keys.len() && !node.row)
            {
                return Err(invalid("Get chain"));
            }
            if index == 0 && u64::from(request.skip_pages) > node.tail_pages as u64 {
                return Err(invalid("Get skip"));
            }
            total += node.tail_pages as u64;
            let header = node.location.header.clone();
            let skip = if index == 0 { request.skip_pages } else { 0 };
            let first_page = node.parent.map_or(0, |parent| {
                parent.consumed_tokens / namespace.descriptor.page_tokens
            }) as u32;
            grants.extend(record_grants(
                &header,
                first_page,
                skip,
                index + 1 == request.keys.len(),
                DEFAULT_MAX_WINDOW,
                index as u32,
            ));
            records.push(GetRecord {
                memory: node.location.memory.clone(),
                file: node
                    .location
                    .disk
                    .as_ref()
                    .map(|extent| extent.reader())
                    .transpose()?,
                header,
                row: node.row,
                meta: None,
                meta_sent: false,
            });
        }
        if total - u64::from(request.skip_pages) != u64::from(request.expected_pages) {
            return Err(invalid("Get page count"));
        }
        for record in &records {
            *self
                .metrics
                .hits
                .entry((
                    flare_selene::shared::record::hex(&id),
                    if record.memory.is_some() {
                        "memory"
                    } else {
                        "disk"
                    },
                ))
                .or_default() += 1;
        }
        Ok((records, grants))
    }

    fn prepare_get_records(&mut self, connection: u64, transfer: u64, id: NamespaceId) {
        let records = {
            let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                return;
            };
            get.records.len()
        };
        for index in 0..records {
            let (memory, file, header, length) = {
                let Some(get) = get_ref(&self.sessions, connection, transfer) else {
                    return;
                };
                let record = &get.records[index];
                (
                    record.memory.clone(),
                    record.file.clone(),
                    record.header.clone(),
                    record.header.length(record.row),
                )
            };
            if let Some(memory) = memory {
                let meta = memory.bytes()[..header.page_segment as usize].to_vec();
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.records[index].meta = Some(meta);
                }
                continue;
            }
            let Some(file) = file else {
                self.fail_get(
                    connection,
                    transfer,
                    proto::Reason::Io as u8,
                    invalid("missing file"),
                );
                return;
            };
            let meta_length = header.page_segment as usize;
            let fill = self.memory_room(length as usize, self.config.root.is_some());
            let (buffer, kind) = if let Some(extent) = fill {
                (extent, ReadKind::Fill)
            } else {
                // Metadata windows are staging traffic too; a full durable
                // budget must not refuse a restore.
                let Some(extent) = self.staging.allocate(meta_length) else {
                    self.fail_get(
                        connection,
                        transfer,
                        proto::Reason::Budget as u8,
                        invalid("staging budget"),
                    );
                    return;
                };
                (
                    extent,
                    ReadKind::Meta {
                        length: meta_length,
                    },
                )
            };
            self.job_counter += 1;
            let job = self.job_counter;
            let task = DiskJob::ReadRecord {
                id: JobId(job),
                file,
                span: Span {
                    address: buffer.bytes().as_ptr() as usize,
                    offset: 0,
                    length: buffer.bytes().len(),
                },
            };
            if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                get.pending_reads += 1;
            }
            self.jobs.insert(
                job,
                Job::GetRead {
                    connection,
                    transfer,
                    record: index as u32,
                    kind,
                    buffer,
                },
            );
            self.submit_read(id, job, task);
        }
    }

    fn get_read_done(
        &mut self,
        connection: u64,
        transfer: u64,
        record: u32,
        kind: ReadKind,
        data: Option<Extent>,
    ) {
        let terminal = self
            .sessions
            .get(&connection)
            .and_then(|session| session.ops.get(&transfer))
            .is_none_or(|open| {
                matches!(&open.operation, Operation::Get(get) if get.terminal || get.finished)
            });
        if terminal {
            if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                get.pending_reads = get.pending_reads.saturating_sub(1);
                if matches!(kind, ReadKind::Window { .. }) {
                    get.window_reads = get.window_reads.saturating_sub(1);
                }
            }
            return;
        }
        let Some(extent) = data else {
            self.read_failed(connection, transfer);
            return;
        };
        let state = get_ref(&self.sessions, connection, transfer).map(|get| {
            (
                get.namespace,
                usize::try_from(record).is_ok_and(|index| index < get.records.len()),
            )
        });
        let Some((id, in_range)) = state else {
            return;
        };
        if !in_range {
            self.read_failed(connection, transfer);
            return;
        }
        match kind {
            ReadKind::Fill => {
                let (header, row) = {
                    let Some(get) = get_ref(&self.sessions, connection, transfer) else {
                        return;
                    };
                    let record = &get.records[record as usize];
                    (record.header.clone(), record.row)
                };
                if record::verify_record(extent.bytes(), &header, row).is_err() {
                    self.read_failed(connection, transfer);
                    return;
                }
                let meta = extent.bytes()[..header.page_segment as usize].to_vec();
                let extent = Arc::new(extent);
                self.get_filled(id, &header, extent.clone());
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.records[record as usize].memory = Some(extent);
                    get.records[record as usize].meta = Some(meta);
                    get.pending_reads -= 1;
                }
            }
            ReadKind::Meta { length } => {
                let meta = extent.bytes()[..length].to_vec();
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.records[record as usize].meta = Some(meta);
                    get.pending_reads -= 1;
                }
            }
            ReadKind::Window { grant } => {
                let plan = {
                    let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                        return;
                    };
                    get.pending_reads = get.pending_reads.saturating_sub(1);
                    get.window_reads = get.window_reads.saturating_sub(1);
                    let Some(plan) = get.grants.get(grant as usize) else {
                        return;
                    };
                    *plan
                };
                let extent = Arc::new(extent);
                let offset = match self.local.offset(Region::Staging, &extent, 0, plan.length) {
                    Ok(offset) => offset,
                    Err(error) => {
                        self.fail_get(connection, transfer, proto::Reason::Invalid as u8, error);
                        return;
                    }
                };
                if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                    get.outbox.push_back(Ok(GetAnswer::Ready(plan.window(
                        transfer,
                        grant,
                        Region::Staging,
                        offset,
                        extent,
                    ))));
                }
            }
        }
    }

    fn read_failed(&mut self, connection: u64, transfer: u64) {
        let terminal = self
            .sessions
            .get(&connection)
            .and_then(|session| session.ops.get(&transfer))
            .is_none_or(|open| {
                matches!(&open.operation, Operation::Get(get) if get.terminal || get.finished)
            });
        if terminal {
            if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
                get.pending_reads = get.pending_reads.saturating_sub(1);
                get.window_reads = get.window_reads.saturating_sub(1);
            }
            return;
        }
        if let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) {
            get.pending_reads = get.pending_reads.saturating_sub(1);
            get.window_reads = get.window_reads.saturating_sub(1);
        }
        self.fail_get(
            connection,
            transfer,
            proto::Reason::Io as u8,
            invalid("disk read failed"),
        );
    }

    fn get_filled(&mut self, id: NamespaceId, header: &Header, extent: Arc<Extent>) {
        let Some(node) = self
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .node_mut(&header.key)
        else {
            return;
        };
        if node.location.memory.is_some() || node.location.header != *header {
            return;
        }
        let old = node.location.bytes;
        node.location.memory = Some(extent);
        node.location.bytes = record_charge(header, true);
        let new = node.location.bytes;
        self.charged = self.charged - old + new;
        self.delta(id, CatalogKind::Changed, header.key);
    }

    fn maybe_finish_get(&mut self, connection: u64, transfer: u64) {
        let done = {
            let Some(get) = get_ref(&self.sessions, connection, transfer) else {
                return;
            };
            !get.terminal
                && !get.cancelled
                && get.next_grant >= get.grants.len()
                && get.window_reads == 0
                && get.pending_reads == 0
                && get.next_meta >= get.records.len()
        };
        if done {
            let (namespace, pins) = {
                let Some((get, _)) = get_mut(&mut self.sessions, connection, transfer) else {
                    return;
                };
                get.terminal = true;
                get.finished = true;
                get.outbox.push_back(Ok(GetAnswer::Done));
                (get.namespace, std::mem::take(&mut get.pins))
            };
            self.metrics.gets_ok += 1;
            self.release_pins(namespace, pins);
        }
    }

    fn release_pins(&mut self, id: NamespaceId, pins: VecDeque<Key>) {
        let namespace = self.namespaces.get_mut(&id).unwrap();
        for key in pins {
            let _ = namespace.tree.unpin(&key);
        }
    }

    // ----------------------------------------------------------- Cancel, close

    fn cancel(&mut self, connection: u64, transfer: u64) {
        self.metrics.cancels += 1;
        // A Begin that has not reached dispatch yet must be withdrawn too.
        let mut removed_pending = false;
        if let Some(queue) = self.pending_puts.get_mut(&connection) {
            if let Some(index) = queue.iter().position(|put| put.transfer == transfer) {
                queue.remove(index);
                removed_pending = true;
            }
            if queue.is_empty() {
                self.pending_puts.remove(&connection);
                self.put_order.retain(|id| *id != connection);
            }
        }
        if let Some(queue) = self.pending_gets.get_mut(&connection) {
            if let Some(index) = queue.iter().position(|get| get.transfer == transfer) {
                queue.remove(index);
                removed_pending = true;
            }
            if queue.is_empty() {
                self.pending_gets.remove(&connection);
                self.get_order.retain(|id| *id != connection);
            }
        }
        if removed_pending {
            return;
        }
        let Some(open) = self
            .sessions
            .get_mut(&connection)
            .and_then(|session| session.ops.get_mut(&transfer))
        else {
            return;
        };
        match &mut open.operation {
            Operation::Put(put) => {
                if put.committed {
                    return;
                }
                put.cancelled = true;
                put.terminal = true;
                put.blocked.clear();
                put.outbox.clear();
            }
            Operation::Get(get) => {
                get.cancelled = true;
                get.terminal = true;
                get.outbox.clear();
            }
        }
    }

    fn connection_closed(&mut self, connection: u64) {
        self.local.close(connection);
        self.pending_gets.remove(&connection);
        self.pending_puts.remove(&connection);
        self.get_order.retain(|id| *id != connection);
        self.put_order.retain(|id| *id != connection);
        let Some(session) = self.sessions.get_mut(&connection) else {
            return;
        };
        session.closed = true;
        let transfers: Vec<_> = session.ops.keys().copied().collect();
        for transfer in transfers {
            self.cancel(connection, transfer);
        }
    }

    // ------------------------------------------------------------- Maintenance

    fn flush_outboxes(&mut self) {
        let connections: Vec<u64> = self.sessions.keys().copied().collect();
        let mut remove = Vec::new();
        for connection in connections {
            let Some(session) = self.sessions.get_mut(&connection) else {
                continue;
            };
            for (transfer, open) in &mut session.ops {
                match &mut open.operation {
                    Operation::Put(put) => {
                        if session.closed {
                            put.outbox.clear();
                        }
                        while let Some(answer) = put.outbox.pop_front() {
                            match put.events.try_send(answer) {
                                Ok(()) => {}
                                Err(tokio::sync::mpsc::error::TrySendError::Full(answer)) => {
                                    put.outbox.push_front(answer);
                                    break;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
                            }
                        }
                        // Keep failures until the pool fences the writer and cancels the put.
                        if put.terminal
                            && (put.cancelled || put.committed || session.closed)
                            && put.outbox.is_empty()
                            && put.pending == 0
                        {
                            remove.push((connection, *transfer));
                        }
                    }
                    Operation::Get(get) => {
                        if session.closed {
                            get.outbox.clear();
                        }
                        while let Some(answer) = get.outbox.pop_front() {
                            match get.events.try_send(answer) {
                                Ok(()) => {}
                                Err(tokio::sync::mpsc::error::TrySendError::Full(answer)) => {
                                    get.outbox.push_front(answer);
                                    break;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
                            }
                        }
                        if get.terminal
                            && get.finished
                            && get.pending_reads == 0
                            && get.outbox.is_empty()
                        {
                            remove.push((connection, *transfer));
                        }
                    }
                }
            }
        }
        for (connection, transfer) in remove {
            if let Some(session) = self.sessions.get_mut(&connection) {
                session.ops.remove(&transfer);
            }
        }
        self.sessions
            .retain(|_, session| !session.closed || !session.ops.is_empty());
    }

    fn settle_terminal_puts(&mut self) {
        let mut ops = Vec::new();
        for (connection, session) in &self.sessions {
            for (transfer, open) in &session.ops {
                if matches!(&open.operation, Operation::Put(put) if put.terminal && !put.finished && put.pending == 0)
                {
                    ops.push((*connection, *transfer));
                }
            }
        }
        for (connection, transfer) in ops {
            self.settle_put(connection, transfer);
        }
    }

    fn cleanup_turn(&mut self) {
        let mut pins = Vec::new();
        for session in self.sessions.values_mut() {
            for open in session.ops.values_mut() {
                if let Operation::Get(get) = &mut open.operation {
                    if get.terminal && !get.finished && get.pending_reads == 0 {
                        get.finished = true;
                        pins.push((get.namespace, std::mem::take(&mut get.pins)));
                    }
                }
            }
        }
        for (namespace, pins) in pins {
            self.release_pins(namespace, pins);
        }
        self.retired_records.retain(|(id, key, location)| {
            let replaced = self.namespaces[id]
                .tree
                .node(key)
                .is_none_or(|node| node.location.durable);
            if !replaced || location.reading() {
                true
            } else {
                self.charged = self.charged.saturating_sub(location.bytes);
                false
            }
        });
    }

    fn disk_events(&mut self) {
        let mut completions = Vec::new();
        for namespace in self.namespaces.values() {
            for worker in [&namespace.reads, &namespace.disk].into_iter().flatten() {
                worker.poll(&mut completions);
            }
        }
        if !completions.is_empty() {
            self.disk_progress = Instant::now();
        }
        for completion in completions {
            let failed = completion.result.is_err();
            if failed {
                self.metrics.nvme_errors += 1;
            }
            let Some(job) = self.jobs.remove(&completion.id.0) else {
                continue;
            };
            self.metrics.nvme_bytes += completion.bytes;
            self.metrics.nvme_seconds += completion.elapsed.as_secs_f64();
            match job {
                Job::Write { namespace, key, .. } => {
                    let path = self.namespaces[&namespace].path(key);
                    let node = self
                        .namespaces
                        .get_mut(&namespace)
                        .unwrap()
                        .tree
                        .node_mut(&key)
                        .unwrap();
                    if node.location.job != Some(completion.id.0) {
                        // The record this write belonged to was replaced. The replacement owns
                        // the key's file now, so the stale completion must not touch either the
                        // node or that path.
                        continue;
                    }
                    node.location.job = None;
                    node.location.writing = false;
                    if !failed {
                        if let Some(path) = path {
                            node.location.disk = Some(Arc::new(DiskExtent::new(path)));
                        }
                        node.location.retry = Instant::now();
                    } else {
                        node.location.retry = Instant::now() + Duration::from_millis(100);
                    }
                    self.refresh_durability(namespace);
                }
                Job::Chunk {
                    connection,
                    transfer,
                    ..
                } => {
                    if let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer)
                    {
                        put.pending = put.pending.saturating_sub(1);
                        release_slot_job(put, completion.id.0);
                        *progress = Instant::now();
                    }
                    if failed {
                        self.fail_put(
                            connection,
                            transfer,
                            proto::Reason::Io as u8,
                            invalid("streaming disk write failed"),
                        );
                    }
                    self.settle_put(connection, transfer);
                }
                Job::Finish {
                    connection,
                    transfer,
                    final_extent,
                    ..
                } => {
                    if let Some((put, progress)) = put_mut(&mut self.sessions, connection, transfer)
                    {
                        put.pending = put.pending.saturating_sub(1);
                        *progress = Instant::now();
                    }
                    if failed {
                        self.fail_put(
                            connection,
                            transfer,
                            proto::Reason::Io as u8,
                            invalid("streaming disk write failed"),
                        );
                        self.settle_put(connection, transfer);
                    } else {
                        final_extent.keep();
                        self.publish_put(connection, transfer, Some(final_extent), true);
                    }
                }
                Job::Compact {
                    namespace,
                    key,
                    bytes,
                } => {
                    let _ = self
                        .namespaces
                        .get_mut(&namespace)
                        .unwrap()
                        .tree
                        .unpin(&key);
                    let Some(node) = self
                        .namespaces
                        .get_mut(&namespace)
                        .and_then(|namespace| namespace.tree.node_mut(&key))
                    else {
                        continue;
                    };
                    if node.location.job != Some(completion.id.0) {
                        // The record was replaced; its successor owns this path.
                        continue;
                    }
                    node.location.job = None;
                    node.location.writing = false;
                    if !failed {
                        node.location.bytes -= bytes;
                        self.charged -= bytes;
                        node.location.memory = None;
                        self.delta(namespace, CatalogKind::Changed, key);
                    } else {
                        node.row = true;
                        self.delta(namespace, CatalogKind::Changed, key);
                    }
                    self.refresh_durability(namespace);
                }
                Job::GetRead {
                    connection,
                    transfer,
                    record,
                    kind,
                    buffer,
                } => {
                    if failed {
                        self.read_failed(connection, transfer);
                    } else {
                        self.get_read_done(connection, transfer, record, kind, Some(buffer));
                    }
                }
            }
        }
    }

    fn settle_put(&mut self, connection: u64, transfer: u64) {
        let released = {
            let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
                return;
            };
            if !put.terminal || put.finished || put.pending > 0 {
                return;
            }
            put.finished = true;
            put.blocked.clear();
            Some((
                put.namespace,
                std::mem::take(&mut put.pins),
                put.header.key,
                put.reservation,
            ))
        };
        let Some((id, pins, key, reservation)) = released else {
            return;
        };
        let namespace = self.namespaces.get_mut(&id).unwrap();
        for key in &pins {
            let _ = namespace.tree.unpin(key);
        }
        namespace.inflight.remove(&key);
        self.charged = self.charged.saturating_sub(reservation);
    }

    fn pump_reads(&mut self) {
        for (namespace, job, task) in std::mem::take(&mut self.read_backlog) {
            self.submit_read(namespace, job, task);
        }
    }

    fn write_chunk(
        &mut self,
        namespace: NamespaceId,
        job: u64,
        connection: u64,
        transfer: u64,
        temporary: Arc<DiskExtent>,
        offset: u64,
        data: ExtentSlice,
    ) -> std::result::Result<(), ExtentSlice> {
        let task = DiskJob::write_record(
            JobId(job),
            temporary.path.clone(),
            offset,
            data.bytes(),
            None,
        );
        let worker = self.namespaces[&namespace].disk.as_ref().unwrap();
        if let Err(error) = worker.try_submit(task) {
            if matches!(error, std::sync::mpsc::TrySendError::Disconnected(_)) {
                let error = invalid("disk worker stopped");
                self.fail_put(connection, transfer, proto::Reason::Io as u8, error);
            }
            return Err(data);
        }
        self.jobs.insert(
            job,
            Job::Chunk {
                connection,
                transfer,
                _data: data,
                _temporary: temporary,
            },
        );
        Ok(())
    }

    fn pump_chunks(&mut self) {
        let connections: Vec<u64> = self.sessions.keys().copied().collect();
        for connection in connections {
            let transfers: Vec<u64> = self.sessions[&connection]
                .ops
                .iter()
                .filter(|(_, open)| matches!(open.operation, Operation::Put(_)))
                .map(|(transfer, _)| *transfer)
                .collect();
            for transfer in transfers {
                loop {
                    let next = {
                        let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer)
                        else {
                            break;
                        };
                        if put.cancelled || put.terminal {
                            break;
                        }
                        let Some((job, offset, data)) = put.blocked.pop_front() else {
                            break;
                        };
                        put.temporary
                            .clone()
                            .map(|temporary| (job, offset, data, temporary))
                    };
                    let Some((job, offset, data, temporary)) = next else {
                        break;
                    };
                    let namespace =
                        put_ref(&self.sessions, connection, transfer).map(|put| put.namespace);
                    let Some(namespace) = namespace else {
                        break;
                    };
                    if self.namespaces[&namespace].disk.is_none() {
                        self.fail_put(
                            connection,
                            transfer,
                            proto::Reason::Io as u8,
                            invalid("no disk"),
                        );
                        break;
                    }
                    match self.write_chunk(
                        namespace, job, connection, transfer, temporary, offset, data,
                    ) {
                        Ok(()) => {
                            if let Some((put, _)) =
                                put_mut(&mut self.sessions, connection, transfer)
                            {
                                put.pending += 1;
                            }
                        }
                        Err(data) => {
                            if let Some((put, _)) =
                                put_mut(&mut self.sessions, connection, transfer)
                            {
                                put.blocked.push_front((job, offset, data));
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    fn schedule(&mut self) {
        let ids: Vec<NamespaceId> = self.namespaces.keys().copied().collect();
        for id in ids {
            if self.namespaces[&id].disk.is_none() {
                continue;
            }
            let namespace = &self.namespaces[&id];
            let mut keys: Vec<Key> = namespace
                .tree
                .nodes()
                .filter(|node| {
                    !node.location.durable
                        && !node.location.writing
                        && node.location.memory.is_some()
                        && node.location.retry <= Instant::now()
                        && namespace.parent_durable(node.parent)
                })
                .map(|node| node.key)
                .collect();
            keys.sort_unstable();
            for key in keys {
                self.job_counter += 1;
                let job = self.job_counter;
                let namespace = &self.namespaces[&id];
                let node = &namespace.tree[&key];
                let Some(memory) = node.location.memory.clone() else {
                    continue;
                };
                let Some(temporary) = namespace.temporary(job) else {
                    continue;
                };
                let Some(path) = namespace.path(key) else {
                    continue;
                };
                let task = DiskJob::write_record(
                    JobId(job),
                    temporary.path.clone(),
                    0,
                    &memory.bytes()[..node.location.header.length(node.row) as usize],
                    Some((path, node.location.header.length(node.row))),
                );
                let Some(worker) = &namespace.disk else {
                    continue;
                };
                if worker.submit(task).is_ok() {
                    self.jobs.insert(
                        job,
                        Job::Write {
                            namespace: id,
                            key,
                            _memory: memory,
                            _temporary: temporary,
                        },
                    );
                    let location = &mut self.namespaces.get_mut(&id).unwrap().tree[&key].location;
                    location.writing = true;
                    location.job = Some(job);
                } else {
                    break;
                }
            }
        }
        let connections: Vec<u64> = self.sessions.keys().copied().collect();
        for connection in connections {
            let ops: Vec<(u64, NamespaceId)> = self.sessions[&connection]
                .ops
                .iter()
                .filter_map(|(transfer, open)| match &open.operation {
                    Operation::Put(put)
                        if put.committed
                            && put.metadata.is_some()
                            && put.pending == 0
                            && put.blocked.is_empty()
                            && put.temporary.is_some()
                            && !put.terminal
                            && !put.finalizing =>
                    {
                        Some((*transfer, put.namespace))
                    }
                    _ => None,
                })
                .collect();
            for (transfer, id) in ops {
                self.schedule_finish(connection, transfer, id);
            }
        }
    }

    fn schedule_finish(&mut self, connection: u64, transfer: u64, id: NamespaceId) {
        let Some((parent, terminal)) = put_ref(&self.sessions, connection, transfer)
            .map(|put| (put.header.parent, put.terminal))
        else {
            return;
        };
        if terminal {
            return;
        }
        if !self.namespaces[&id].parent_durable(parent) {
            return;
        }
        let Some((put, _)) = put_mut(&mut self.sessions, connection, transfer) else {
            return;
        };
        self.job_counter += 1;
        let job = self.job_counter;
        let Some(path) = self.namespaces[&id].path(put.header.key) else {
            return;
        };
        let Some((task, completion)) = finish_record(put, job, connection, transfer, path) else {
            return;
        };
        let Some(worker) = &self.namespaces[&id].disk else {
            return;
        };
        if worker.submit(task).is_ok() {
            put.finalizing = true;
            put.pending += 1;
            self.jobs.insert(job, completion);
        }
    }

    fn tick(&mut self) {
        self.note_reserved();
        let now = Instant::now();
        if self.jobs.is_empty() {
            self.disk_progress = now;
        }
        self.timeouts();
        if self.config.max_age.is_zero() {
            return;
        }
        let ids: Vec<_> = self.namespaces.keys().copied().collect();
        for id in ids {
            let keys = self.namespaces.get_mut(&id).unwrap().tree.expired(|node| {
                !node.pinned() && !node.location.writing && !node.location.reading()
            });
            for key in keys {
                let node = &self.namespaces[&id].tree[&key];
                let tier = usize::from(node.location.disk.is_some());
                let removed = self.evict_accounted(id, key);
                self.metrics.expired[tier] += removed;
            }
        }
    }

    fn timeouts(&mut self) {
        let mut expired = Vec::new();
        let disk_idle = (!self.jobs.is_empty()).then(|| self.disk_progress.elapsed());
        for (connection, session) in &mut self.sessions {
            if session.closed {
                continue;
            }
            for (transfer, open) in &mut session.ops {
                if open.is_over() {
                    continue;
                }
                if open.waits_for_grant() {
                    open.progress = Instant::now();
                    continue;
                }
                let delivered = open.waits_on_disk();
                if stalled(open.progress.elapsed(), delivered, disk_idle) {
                    expired.push((*connection, *transfer));
                }
            }
        }
        for (connection, transfer) in expired {
            let is_put = self
                .sessions
                .get(&connection)
                .and_then(|session| session.ops.get(&transfer))
                .is_some_and(|open| matches!(open.operation, Operation::Put(_)));
            if is_put {
                self.fail_put(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("no progress"),
                );
            } else {
                self.fail_get(
                    connection,
                    transfer,
                    proto::Reason::Invalid as u8,
                    invalid("no progress"),
                );
            }
        }
    }

    pub fn pump(&mut self) {
        self.pump_reads();
        self.pump_chunks();
        self.disk_events();
        self.dispatch();
        self.pump_grants();
        self.cleanup_turn();
        self.settle_terminal_puts();
        self.schedule();
        self.tick();
        self.maybe_finish_gets();
        self.flush_outboxes();
        self.local.poll_prior();
    }

    fn maybe_finish_gets(&mut self) {
        let mut ops = Vec::new();
        for (connection, session) in &self.sessions {
            for (transfer, open) in &session.ops {
                if matches!(open.operation, Operation::Get(_)) {
                    ops.push((*connection, *transfer));
                }
            }
        }
        for (connection, transfer) in ops {
            self.maybe_finish_get(connection, transfer);
        }
    }

    pub fn catalog(
        &mut self,
        descriptor: Descriptor,
        sender: tokio::sync::mpsc::UnboundedSender<proto::CatalogEvent>,
    ) -> Result<Vec<EntryData>> {
        let namespace = self.register_namespace(descriptor)?;
        let entries = self
            .snapshot(namespace)
            .ok_or_else(|| invalid("namespace"))?;
        self.catalogs.insert(namespace, sender);
        Ok(entries)
    }

    pub fn metrics_text(&self) -> String {
        let mut text = String::new();
        for (i, tier) in ["dram", "nvme"].into_iter().enumerate() {
            text.push_str(&format!(
                "selene_expired_total{{tier=\"{tier}\"}} {}\n",
                self.metrics.expired[i]
            ));
        }
        for (name, value) in [
            ("unknown_frames_total", self.metrics.dropped),
            ("cancels_total", self.metrics.cancels),
            ("evictions_total", self.metrics.evictions),
            ("compactions_total", self.metrics.compactions),
            ("nvme_bytes_total", self.metrics.nvme_bytes),
            ("nvme_errors_total", self.metrics.nvme_errors),
            ("connections", self.sessions.len() as u64),
            (
                "leases",
                self.sessions
                    .values()
                    .flat_map(|session| session.ops.values())
                    .filter(|open| matches!(open.operation, Operation::Get(_)))
                    .count() as u64,
            ),
            (
                "open_ops",
                self.sessions
                    .values()
                    .map(|session| session.ops.len() as u64)
                    .sum(),
            ),
            ("memory_allocated_bytes", self.local.arena_used() as u64),
            ("staging_allocated_bytes", self.local.staging_used() as u64),
            ("nvme_reserved_bytes", self.charge_total()),
            (
                "nvme_reserved_peak_bytes",
                self.reserved_peak.max(self.charge_total()),
            ),
            (
                "cleanup_records",
                self.sessions
                    .values()
                    .flat_map(|session| session.ops.values())
                    .map(|open| match &open.operation {
                        Operation::Put(put) if put.terminal && put.pending > 0 => {
                            put.pins.len() as u64
                        }
                        Operation::Get(get) if get.terminal && get.pending_reads > 0 => {
                            get.pins.len() as u64
                        }
                        _ => 0,
                    })
                    .sum::<u64>(),
            ),
            (
                "pending_puts",
                self.pending_puts.values().map(|q| q.len() as u64).sum(),
            ),
            (
                "pending_gets",
                self.pending_gets.values().map(|q| q.len() as u64).sum(),
            ),
        ] {
            text.push_str(&format!("selene_{name} {value}\n"));
        }
        text.push_str(&format!(
            "selene_nvme_seconds_total {}\n",
            self.metrics.nvme_seconds
        ));
        for (outcome, count) in [
            ("ok", self.metrics.puts_ok),
            ("error", self.metrics.puts_error),
            ("present", self.metrics.present),
            ("queue", self.metrics.retry_queue),
            ("budget", self.metrics.retry_budget),
            ("inflight", self.metrics.retry_inflight),
            ("oversize", self.metrics.oversize),
        ] {
            text.push_str(&format!(
                "selene_puts_total{{outcome=\"{outcome}\"}} {count}\n"
            ));
        }
        for (outcome, count) in [
            ("ok", self.metrics.gets_ok),
            ("error", self.metrics.gets_error),
        ] {
            text.push_str(&format!(
                "selene_gets_total{{outcome=\"{outcome}\"}} {count}\n"
            ));
        }
        let mut ids: Vec<&NamespaceId> = self.namespaces.keys().collect();
        ids.sort_unstable();
        for id in ids {
            let name = flare_selene::shared::record::hex(id);
            let namespace = &self.namespaces[id];
            for tier in ["memory", "disk"] {
                let mut count = 0;
                let mut bytes = 0;
                for node in namespace.tree.nodes() {
                    let retained = if tier == "memory" {
                        node.location.memory.is_some()
                    } else {
                        node.location.durable
                    };
                    if retained {
                        count += 1;
                        bytes += node.location.header.length(node.row);
                    }
                }
                text.push_str(&format!("selene_records{{namespace=\"{name}\",tier=\"{tier}\"}} {count}\nselene_bytes{{namespace=\"{name}\",tier=\"{tier}\"}} {bytes}\n"));
                let hits = self
                    .metrics
                    .hits
                    .get(&(name.clone(), tier))
                    .copied()
                    .unwrap_or(0);
                text.push_str(&format!(
                    "selene_hits_total{{namespace=\"{name}\",tier=\"{tier}\"}} {hits}\n"
                ));
            }
        }
        text
    }
}

pub type Reply<T> = tokio::sync::oneshot::Sender<T>;

pub struct PutAccept {
    pub transfer: u64,
    pub first_page: u32,
    pub parent: Option<Key>,
    pub record_bytes: u64,
    pub windows: usize,
}

pub struct WindowGrant {
    pub hold: Arc<Extent>,
    pub transfer: u64,
    pub index: u32,
    pub record: u32,
    pub record_offset: u64,
    pub length: u64,
    pub region: i32,
    pub offset: u64,
    pub page_first: u64,
    pub page_count: u32,
    pub row: bool,
}

pub struct Refused {
    pub reason: u8,
    pub detail: String,
}

pub enum PutAnswer {
    Accept(PutAccept),
    Grant(WindowGrant),
    Stored { bytes: u64 },
    Refused(Refused),
}

pub enum GetAnswer {
    Leased {
        transfer: u64,
        windows: usize,
    },
    Meta {
        record: u32,
        pages: Vec<u64>,
        row: u64,
        source: i32,
    },
    Ready(WindowGrant),
    Done,
    Refused(Refused),
}

pub enum StoreCommand {
    Hello {
        connection: u64,
        pid: u32,
        descriptor: Descriptor,
        reply: Reply<Result<()>>,
    },
    Catalog {
        descriptor: Descriptor,
        events: tokio::sync::mpsc::UnboundedSender<proto::CatalogEvent>,
        reply: Reply<Result<Vec<EntryData>>>,
    },
    PutBegin {
        connection: u64,
        transfer: u64,
        namespace: NamespaceId,
        descriptor: Descriptor,
        key: Key,
        published_parent: Option<Key>,
        bucket: [u8; 16],
        chain_pages: u32,
        published_tail_pages: u32,
        row: bool,
        events: tokio::sync::mpsc::Sender<Result<PutAnswer>>,
        reply: Reply<Result<()>>,
    },
    WindowFilled {
        connection: u64,
        transfer: u64,
        index: u32,
        page_xxh3: Vec<u64>,
        row_xxh3: Option<u64>,
        reply: Reply<Result<()>>,
    },
    PutCommit {
        connection: u64,
        transfer: u64,
        metadata_offset: u64,
        metadata: Vec<u8>,
        reply: Reply<Result<()>>,
    },
    GetBegin {
        connection: u64,
        transfer: u64,
        namespace: NamespaceId,
        chain: Vec<Key>,
        skip_pages: u32,
        expected_pages: u32,
        row: bool,
        events: tokio::sync::mpsc::Sender<Result<GetAnswer>>,
        reply: Reply<Result<()>>,
    },
    RequestWindow {
        connection: u64,
        transfer: u64,
        reply: Reply<Result<()>>,
    },
    Cancel {
        connection: u64,
        transfer: u64,
    },
    ConnectionClosed {
        connection: u64,
    },
    Tick,
    Shutdown,
}

fn region_wire(region: Region) -> i32 {
    match region {
        Region::Arena => proto::Region::Arena as i32,
        Region::Staging => proto::Region::Staging as i32,
    }
}

fn record_grants(
    header: &Header,
    first_page: u32,
    skip: u32,
    include_row: bool,
    max_window: u64,
    record: u32,
) -> Vec<PlannedGrant> {
    use crate::pool::windows::{self, RecordShape};
    let shape = RecordShape {
        first_page: skip,
        pages: header.tail_pages.saturating_sub(skip),
        row: include_row,
    };
    if shape.pages == 0 && !shape.row {
        return Vec::new();
    }
    windows::plan(
        &[shape],
        header.page_record_bytes(),
        header.row_record_bytes(),
        max_window,
    )
    .expect("stored records have validated geometry")
    .into_iter()
    .map(|w| PlannedGrant {
        record,
        record_offset: if w.page_count == 0 {
            header.row_segment
        } else {
            header.page_segment + u64::from(w.first_page) * header.page_record_bytes()
        },
        length: w.length,
        pages: w.page_count,
        row: w.row,
        page_first: if w.page_count == 0 {
            0
        } else {
            u64::from(first_page) + u64::from(w.first_page)
        },
    })
    .collect()
}

impl Store {
    pub fn handle(&mut self, command: StoreCommand) {
        match command {
            StoreCommand::Hello {
                connection,
                pid,
                descriptor,
                reply,
            } => {
                let result = self.hello(connection, pid, descriptor);
                let _ = reply.send(result);
            }
            StoreCommand::Catalog {
                descriptor,
                events,
                reply,
            } => {
                let _ = reply.send(self.catalog(descriptor, events));
            }
            StoreCommand::PutBegin {
                connection,
                transfer,
                namespace,
                descriptor,
                key,
                published_parent,
                bucket,
                chain_pages,
                published_tail_pages,
                row,
                events,
                reply,
            } => {
                let session_ok = self
                    .sessions
                    .get(&connection)
                    .is_some_and(|session| !session.closed && session.namespace == namespace);
                let valid = session_ok
                    && descriptor.namespace_id() == namespace
                    && self.register_namespace(descriptor).is_ok();
                if !valid {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "PutBegin namespace or descriptor",
                    )));
                    return;
                }
                if self.transfer_in_use(connection, transfer) {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "Put transfer id is already open on this connection",
                    )));
                    return;
                }
                self.submit_put(
                    connection,
                    transfer,
                    PutRequest {
                        namespace,
                        key,
                        published_parent,
                        bucket,
                        chain_pages,
                        published_tail_pages,
                        row,
                    },
                    events,
                );
                let _ = reply.send(Ok(()));
            }
            StoreCommand::WindowFilled {
                connection,
                transfer,
                index,
                page_xxh3,
                row_xxh3,
                reply,
            } => {
                let result = self.window_filled(connection, transfer, index, page_xxh3, row_xxh3);
                let _ = reply.send(result);
            }
            StoreCommand::PutCommit {
                connection,
                transfer,
                metadata_offset,
                metadata,
                reply,
            } => {
                let result = self.put_commit(connection, transfer, metadata_offset, metadata);
                let _ = reply.send(result);
            }
            StoreCommand::GetBegin {
                connection,
                transfer,
                namespace,
                chain,
                skip_pages,
                expected_pages,
                row,
                events,
                reply,
            } => {
                let allowed = self
                    .sessions
                    .get(&connection)
                    .is_some_and(|session| !session.closed && session.namespace == namespace);
                if !allowed {
                    // a Get for a namespace the session has not registered is a
                    // terminal operation refusal.
                    let _ = reply.send(Ok(()));
                    let _ = events.try_send(Ok(GetAnswer::Refused(Refused {
                        reason: proto::Reason::Invalid as u8,
                        detail: "Get namespace is not registered".into(),
                    })));
                    return;
                }
                if self.transfer_in_use(connection, transfer) {
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "Get transfer id is already open on this connection",
                    )));
                    return;
                }
                self.submit_get(
                    connection,
                    transfer,
                    GetRequest {
                        namespace,
                        keys: chain,
                        skip_pages,
                        expected_pages,
                        row,
                    },
                    events,
                );
                let _ = reply.send(Ok(()));
            }
            StoreCommand::RequestWindow {
                connection,
                transfer,
                reply,
            } => {
                let result = self
                    .sessions
                    .get_mut(&connection)
                    .and_then(|session| session.ops.get_mut(&transfer))
                    .ok_or_else(|| invalid("unknown operation"))
                    .map(|open| match &mut open.operation {
                        Operation::Put(put) => put.requested += 1,
                        Operation::Get(get) => get.requested += 1,
                    });
                let _ = reply.send(result);
            }
            StoreCommand::Cancel {
                connection,
                transfer,
            } => {
                self.cancel(connection, transfer);
            }
            StoreCommand::ConnectionClosed { connection } => self.connection_closed(connection),
            StoreCommand::Tick => {}
            StoreCommand::Shutdown => {}
        }
    }
}

fn validate_commit(put: &mut Put, descriptor: &Descriptor) -> Result<()> {
    let header = Header::decode(&put.commit_bytes[..4096]).map_err(|e| invalid(e.to_string()))?;
    let planned = &put.header;
    let row_ok = match put.windows.row {
        Some(hash) => header.row_checksum == hash,
        None => header.row_checksum == planned.row_checksum,
    };
    if header.key != planned.key
        || header.parent != planned.parent
        || header.bucket != planned.bucket
        || header.tail_pages != planned.tail_pages
        || header.page_segment != planned.page_segment
        || header.row_segment != planned.row_segment
        || header.flags != 0
        || !row_ok
        || !descriptor.matches(&header)
    {
        return Err(invalid("commit header disagrees with accept"));
    }
    let page_segment = planned.page_segment as usize;
    let checksum_segment = &put.commit_bytes[4096..page_segment];
    for (index, hash) in put.windows.pages.iter().enumerate() {
        let Some(hash) = hash else {
            return Err(invalid("missing page hash"));
        };
        if checksum_segment[index * 8..index * 8 + 8] != hash.to_le_bytes() {
            return Err(invalid("commit checksum table"));
        }
    }
    put.header = header;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_delivered_put_waits_while_the_disk_makes_progress() {
        use super::{NO_PROGRESS, stalled};
        use std::time::Duration;
        let long = NO_PROGRESS + Duration::from_secs(1);
        let short = Duration::from_millis(10);
        // Waiting on the client for too long is a stall.
        assert!(stalled(long, false, Some(short)));
        assert!(!stalled(short, false, None));
        // Everything delivered and the disk busy but moving: not a stall, however long.
        assert!(!stalled(long, true, Some(short)));
        // The disk has made no progress either, or nothing is queued for it.
        assert!(stalled(long, true, Some(long)));
        assert!(stalled(long, true, None));
    }

    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn descriptor() -> Descriptor {
        Descriptor {
            identity: [7; 32],
            fingerprint: [3; 16],
            page_tokens: 64,
            ranks: vec![flare_selene::shared::record::RankSizes {
                planes: vec![4000],
                row_bytes: 1000,
            }],
            pool_name: String::new(),
        }
    }

    fn store(label: &str) -> (Store, PathBuf) {
        dram_store(label, 1024 * 1024)
    }

    fn dram_store(label: &str, memory: usize) -> (Store, PathBuf) {
        let name = format!(
            "selene-unit-{}-{label}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let mut config =
            Config::dram_only(std::env::temp_dir().join(format!("{name}.sock")), memory);
        config.name = name.clone();
        config.staging = 8 * 1024 * 1024;
        let store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        (store, directory)
    }

    fn store_with_root(label: &str, memory: usize) -> (Store, PathBuf, PathBuf) {
        let name = format!(
            "selene-unit-{}-{label}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            memory,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 * 1024 * 1024;
        let store = Store::open(config).unwrap();
        let directory = store.shm_directory();
        (store, base, directory)
    }

    #[test]
    fn admission_classifies_shapes_and_retained_capacity() {
        use proto::Reason::{Budget, Oversize};
        const K: u64 = 1024;
        const M: u64 = 1024 * K;
        // label, arena, disk budget, staging, planes, row, ancestors, held arena,
        // requested pages, expected refusal (None means accepted).
        #[rustfmt::skip]
        let cases = [
            ("small-arena", 16 * K, Some(24 * K), 8 * M, vec![4000], 1000, vec![], 0, 1, None),
            ("stream-charge", 4096, Some(24 * K), 8 * M, vec![4000], 1000, vec![], 0, 2, Some(Oversize)),
            ("row-only", 4096, Some(24 * K), 8 * M, vec![4000], 1000, vec![], 0, 0, None),
            ("two-windows-tight", 4096, Some(1100 * K), 8 * M, vec![4000], 1000, vec![], 0, 128, Some(Oversize)),
            ("two-windows-fit", 4096, Some(2 * M), 8 * M, vec![4000], 1000, vec![], 0, 128, None),
            ("trailing-window-tight", 4096, Some(1400 * K), 8 * M, vec![4000], 1000, vec![], 0, 192, Some(Oversize)),
            ("trailing-window-fit", 4096, Some(2 * M), 8 * M, vec![4000], 1000, vec![], 0, 192, None),
            ("oversized-page", 16 * M, Some(64 * M), 8 * M, vec![16 * M], 1000, vec![], 0, 1, Some(Oversize)),
            ("retained-parent", 32 * K, None, 8 * M, vec![4000], 1000, vec![2], 0, 6, Some(Budget)),
            ("exact-arena", 16 * M, None, 8 * M, vec![65_539, 4100], 1_000_003, vec![], 0, 140, None),
            ("over-arena", 16 * M, None, 8 * M, vec![65_539, 4100], 1_000_003, vec![], 0, 220, Some(Oversize)),
            ("rounded-over-arena", 12 * M, None, 8 * M, vec![4000], 1000, vec![], 0, 2048, Some(Oversize)),
            ("rounded-within-arena", 12 * M, None, 8 * M, vec![4000], 1000, vec![], 0, 1024, None),
            ("staging-hold-tight", M, Some(64 * M), 8 * M, vec![4000], 8 * M, vec![], 0, 1, Some(Oversize)),
            ("staging-hold-fit", M, Some(64 * M), 16 * M, vec![4000], 8 * M, vec![], 0, 1, None),
            ("temporary-pressure", 48 * K, None, 8 * M, vec![4000], 1000, vec![1], 32 * K, 6, Some(Budget)),
            ("pressure-released", 48 * K, None, 8 * M, vec![4000], 1000, vec![1], 0, 6, None),
            ("rounded-charge-tight", 2 * M, Some(1_500_000), 8 * M, vec![4000], 1000, vec![], 0, 300, Some(Oversize)),
            ("rounded-charge-fit", 2 * M, Some(2_300_000), 8 * M, vec![4000], 1000, vec![], 0, 300, None),
            ("retained-chain", 64 * K, None, 8 * M, vec![4000], 1000, vec![5, 10], 0, 14, Some(Budget)),
            ("parent-disk-charge", 64 * K, Some(56 * K), 8 * M, vec![4000], 1000, vec![5], 0, 10, Some(Budget)),
            ("four-open-children", 8 * M, Some(24 * M), 8 * M, vec![4000], 4096 * 1023, vec![1], 0, 2, None),
        ];
        for (label, arena, disk, staging, planes, row, ancestors, held, pages, refusal) in cases {
            let name = format!(
                "selene-unit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let base = std::env::temp_dir().join(&name);
            let mut config = Config::dram_only(base.join("manager.sock"), arena as usize);
            config.name = name;
            config.root = disk.map(|_| base.join("records"));
            config.nvme = disk.unwrap_or(0);
            config.staging = staging;
            let mut store = Store::open(config).unwrap();
            let mut value = descriptor();
            value.ranks[0].planes = planes;
            value.ranks[0].row_bytes = row;
            store.hello(1, std::process::id(), value.clone()).unwrap();
            let id = value.namespace_id();
            let key = |pages: u32| Key {
                consumed_tokens: u64::from(pages) * 64 + 3,
                digest: (pages as u128).to_le_bytes(),
            };
            let mut parent = None;
            let mut first = 0;
            for ancestor in ancestors {
                let header = crate::record::new_header(
                    &value,
                    key(ancestor),
                    parent,
                    [0; 16],
                    ancestor - first,
                )
                .unwrap();
                let extent = store
                    .memory
                    .allocate(header.length_with_row() as usize)
                    .unwrap();
                insert_memory_node(
                    &mut store,
                    id,
                    &value,
                    key(ancestor),
                    parent,
                    ancestor - first,
                    extent,
                );
                parent = Some(key(ancestor));
                first = ancestor;
            }
            let pressure = (held > 0).then(|| store.memory.allocate(held as usize).unwrap());
            let (events, mut answers) = tokio::sync::mpsc::channel(16);
            let count = if label == "four-open-children" { 4 } else { 1 };
            let mut accepted = 0;
            for index in 0..count {
                store.begin_put(PendingPut {
                    connection: 1,
                    transfer: index + 1,
                    request: PutRequest {
                        namespace: id,
                        key: key(pages + index as u32),
                        published_parent: parent,
                        bucket: [0; 16],
                        chain_pages: pages + index as u32,
                        published_tail_pages: pages + index as u32 - first,
                        row: true,
                    },
                    events: events.clone(),
                });
                store.flush_outboxes();
                let answer = answers.try_recv().unwrap().unwrap();
                match answer {
                    PutAnswer::Accept(_) => {
                        assert!(refusal.is_none(), "{label}: accepted");
                        accepted += 1;
                    }
                    PutAnswer::Refused(reason) => {
                        let expected = if count > 1 {
                            Budget
                        } else {
                            refusal.expect(label)
                        };
                        assert_eq!(reason.reason, expected as u8, "{label}");
                    }
                    _ => panic!("{label}: expected an admission verdict"),
                }
            }
            if refusal.is_none() {
                assert!(accepted > 0, "{label}: no put admitted");
            }
            drop(pressure);
            let directory = store.shm_directory();
            drop(store);
            let _ = fs::remove_dir_all(base);
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn recovery_counts_compacted_files_without_their_removed_rows() {
        let (store, base, directory) = store_with_root("recovery-bytes", 64 * 1024);
        let config = store.config.clone();
        drop(store);
        let value = descriptor();
        let records = config.root.as_ref().unwrap().join(value.name());
        fs::create_dir_all(&records).unwrap();
        let mut parent = None;
        let mut expected = 0;
        for (pages, row) in [(1u32, false), (2, true)] {
            let key = Key {
                consumed_tokens: u64::from(pages) * 64,
                digest: [pages as u8; 16],
            };
            let mut header = crate::record::new_header(&value, key, parent, [0; 16], 1).unwrap();
            let mut bytes = vec![0; header.length(row) as usize];
            let page_hash = flare_selene::shared::record::checksum(
                &bytes[header.page_segment as usize..header.row_segment as usize],
            );
            if row {
                header.row_checksum =
                    flare_selene::shared::record::checksum(&bytes[header.row_segment as usize..]);
            }
            bytes[..4096].copy_from_slice(&header.encode().unwrap());
            bytes[4096..4104].copy_from_slice(&page_hash.to_le_bytes());
            expected += bytes.len() as u64;
            fs::write(records.join(record::file_name(&key)), bytes).unwrap();
            parent = Some(key);
        }
        let mut store = Store::open(config.clone()).unwrap();
        assert_eq!(store.charge_total(), expected);
        let id = store.namespace_id().unwrap();
        assert_eq!(store.namespaces[&id].tree.nodes().count(), 2);
        let name = flare_selene::shared::record::hex(&id);
        assert!(store.metrics_text().contains(&format!(
            "selene_bytes{{namespace=\"{name}\",tier=\"disk\"}} {expected}\n"
        )));
        let old = SystemTime::now() - Duration::from_secs(10800);
        let key = parent.unwrap();
        let path = records.join(record::file_name(&key));
        let bytes = fs::read(&path).unwrap();
        let last_used = store.namespaces[&id].tree[&key].last_used;
        store.reused(id, key);
        assert!(store.jobs.is_empty());
        assert!(!store.namespaces[&id].tree[&key].pinned());
        assert_eq!(store.namespaces[&id].tree[&key].last_used, last_used);
        store.config.max_age = Duration::from_secs(7200);
        store.namespaces.get_mut(&id).unwrap().tree.max_age = store.config.max_age;
        store.reused(id, key);
        assert!(store.namespaces[&id].tree[&key].last_used > last_used);
        store.reused(id, key);
        assert!(store.jobs.is_empty());
        assert!(!store.namespaces[&id].tree[&key].pinned());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), last_used);
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(old)
            .unwrap();
        drop(store);
        let mut store = Store::open(config).unwrap();
        assert_eq!(store.namespaces[&id].tree[&key].last_used, old);
        store.config.max_age = Duration::from_secs(7200);
        store.namespaces.get_mut(&id).unwrap().tree.max_age = store.config.max_age;
        store.tick();
        assert!(!store.namespaces[&id].tree.contains(&key));
        assert_eq!(store.metrics.expired, [0, 2]);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn put_requirement_matches_the_shape_the_put_gets() {
        let (store, directory) = store("requirement");
        let value = descriptor();
        let key = |tokens| Key {
            consumed_tokens: tokens,
            digest: [0; 16],
        };
        let one_page = crate::record::new_header(&value, key(64), None, [0; 16], 1).unwrap();
        // The arena shape charges its 16 KiB extent.
        let arena = store.put_requirement(&one_page, 16 * 1024);
        assert!(!arena.streaming);
        assert_eq!(arena.charge, 16 * 1024);
        assert_eq!(arena.reservation, 16 * 1024);
        assert_eq!(arena.allocation, 16 * 1024);
        assert_eq!(arena.unit, 16 * 1024);
        // 128 pages and a 4 KiB row: 524 KiB of disk record plus a fixed
        // hold of two 512 KiB windows.
        let many = crate::record::new_header(&value, key(128 * 64), None, [0; 16], 128).unwrap();
        let stream = store.put_requirement(&many, 0);
        assert!(stream.streaming);
        assert_eq!(stream.reservation, 524 * 1024);
        assert_eq!(stream.allocation, 2 * 512 * 1024);
        assert_eq!(stream.unit, 512 * 1024);
        assert_eq!(stream.charge, (524 + 2 * 512) * 1024);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    fn insert_memory_node(
        store: &mut Store,
        id: NamespaceId,
        value: &Descriptor,
        key: Key,
        parent: Option<Key>,
        tail: u32,
        extent: Extent,
    ) {
        let header = crate::record::new_header(value, key, parent, [0; 16], tail).unwrap();
        let bytes = record_charge(&header, true);
        store.charged += bytes;
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                parent,
                [0; 16],
                tail as usize,
                true,
                true,
                Location {
                    header,
                    memory: Some(Arc::new(extent)),
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    bytes,
                },
            )
            .unwrap();
    }

    #[test]
    fn closed_get_keeps_one_operation_until_its_reads_drain() {
        let (mut store, directory) = store("get-drain");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), value)
            .unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        let get = Get {
            namespace: id,
            pins: VecDeque::new(),
            records: Vec::new(),
            grants: Vec::new(),
            requested: 0,
            next_grant: 0,
            next_meta: 0,
            window_reads: 0,
            pending_reads: 2,
            cancelled: false,
            terminal: true,
            finished: false,
            events,
            outbox: VecDeque::new(),
        };
        store.sessions.get_mut(&1).unwrap().ops.insert(
            1,
            Open {
                operation: Operation::Get(Box::new(get)),
                progress: Instant::now(),
            },
        );
        store.connection_closed(1);
        assert!(store.sessions[&1].closed);
        for expected in [1, 0] {
            let extent = store.staging.allocate(4096).unwrap();
            store.get_read_done(1, 1, 0, ReadKind::Meta { length: 0 }, Some(extent));
            store.flush_outboxes();
            let Operation::Get(get) = &store.sessions[&1].ops[&1].operation else {
                panic!("expected Get");
            };
            assert_eq!(get.pending_reads, expected);
        }
        store.cleanup_turn();
        store.flush_outboxes();
        assert!(!store.sessions.contains_key(&1));
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn namespace_registration_is_first_sight_and_rejects_mismatch() {
        let (mut store, directory) = store("register");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        assert_eq!(id, value.namespace_id());
        assert!(store.namespaces.contains_key(&id));
        assert_eq!(store.register_namespace(value.clone()).unwrap(), id);
        let mut mismatch = value;
        mismatch.ranks[0].row_bytes += 1;
        assert_eq!(mismatch.namespace_id(), id);
        let error = store.register_namespace(mismatch).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    fn insert_node(
        store: &mut Store,
        id: NamespaceId,
        value: &Descriptor,
        key: Key,
        parent: Option<Key>,
        tail: u32,
    ) {
        let header = crate::record::new_header(value, key, parent, [0; 16], tail).unwrap();
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                parent,
                [0; 16],
                tail as usize,
                true,
                true,
                Location {
                    header,
                    memory: None,
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    bytes: 0,
                },
            )
            .unwrap();
    }

    #[test]
    fn catalog_snapshot_precedes_ordered_unbounded_deltas() {
        let (mut store, directory) = store("catalog");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        insert_node(&mut store, id, &value, key, None, 1);
        store.delta(id, CatalogKind::Added, key);
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let snapshot = store.catalog(value, sender).unwrap();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].key, key);
        assert!(receiver.try_recv().is_err());
        for _ in 0..70_000 {
            store.delta(id, CatalogKind::Changed, key);
        }
        store.delta(id, CatalogKind::Removed, key);
        for _ in 0..70_000 {
            assert!(
                matches!(receiver.try_recv().unwrap(), proto::CatalogEvent::Changed(entry) if entry.key == key)
            );
        }
        assert!(
            matches!(receiver.try_recv().unwrap(), proto::CatalogEvent::Removed(removed) if removed == key)
        );
        assert!(receiver.try_recv().is_err());
        drop(receiver);
        store.delta(id, CatalogKind::Removed, key);
        assert!(store.catalogs.is_empty());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_put_naming_an_absent_parent_asks_for_ancestry_repair() {
        let (mut store, directory) = store("parent");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, value.clone()).unwrap();
        let parent = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let key = Key {
            consumed_tokens: 128,
            digest: [9; 16],
        };
        let (events, mut receiver) = tokio::sync::mpsc::channel(4);
        let (reply, _answer) = tokio::sync::oneshot::channel();
        store.handle(StoreCommand::PutBegin {
            connection: 1,
            transfer: 1,
            namespace: id,
            descriptor: value.clone(),
            key,
            published_parent: Some(parent),
            bucket: [0; 16],
            chain_pages: 2,
            published_tail_pages: 1,
            row: true,
            events,
            reply,
        });
        store.pump();
        match receiver.try_recv().unwrap().unwrap() {
            PutAnswer::Refused(refused) => assert!(
                refused.detail.contains("published parent"),
                "unexpected detail: {}",
                refused.detail
            ),
            _ => panic!("expected a refusal"),
        }
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_local_put_with_an_unusable_parent_falls_back_to_a_root() {
        let (mut store, directory) = store("local-parent");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), value.clone())
            .unwrap();
        let parent = Key {
            consumed_tokens: 64,
            digest: [1; 16],
        };
        let key = Key {
            consumed_tokens: 128,
            digest: [9; 16],
        };
        let (events, mut receiver) = tokio::sync::mpsc::channel(4);
        let (reply, _answer) = tokio::sync::oneshot::channel();
        store.handle(StoreCommand::PutBegin {
            connection: 1,
            transfer: 1,
            namespace: id,
            descriptor: value.clone(),
            key,
            published_parent: Some(parent),
            bucket: [0; 16],
            chain_pages: 2,
            published_tail_pages: 1,
            row: true,
            events,
            reply,
        });
        store.pump();
        match receiver.try_recv().unwrap().unwrap() {
            PutAnswer::Accept(accept) => assert!(
                accept.parent.is_none(),
                "a local Put whose parent hint is unusable must be accepted as a root"
            ),
            PutAnswer::Refused(refused) => panic!("expected an accept: {}", refused.detail),
            _ => panic!("expected an accept"),
        }
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_page_only_record_is_replaced_by_a_complete_put() {
        let (mut store, directory) = store("publish-target");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [6; 16],
        };
        let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
        let extent = Arc::new(
            store
                .memory
                .allocate(header.length_with_row() as usize)
                .unwrap(),
        );
        let bytes = record_charge(&header, true);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                None,
                [0; 16],
                1,
                true,
                true,
                Location {
                    header: header.clone(),
                    memory: Some(extent),
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    bytes,
                },
            )
            .unwrap();
        // Compaction drops the row, keeping the pages as a replacement target.
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .node_mut(&key)
            .unwrap()
            .row = false;
        assert!(
            matches!(store.publish_target(id, &header), PublishTarget::Replace),
            "a page-only record must be replaced, not treated as present"
        );
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_get_without_a_row_is_refused() {
        let (mut store, directory) = store("get-row");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [7; 16],
        };
        insert_node(&mut store, id, &value, key, None, 1);
        let refused = store.plan_get(
            id,
            &GetRequest {
                namespace: id,
                keys: vec![key],
                skip_pages: 0,
                expected_pages: 1,
                row: false,
            },
        );
        assert!(
            refused.is_err(),
            "the Get contract requires the endpoint row"
        );
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_stale_disk_write_does_not_attach_to_a_replacement() {
        let (mut store, directory, _root) = store_with_root("stale-write", 1 << 20);
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [12; 16],
        };
        let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
        let extent = Arc::new(
            store
                .memory
                .allocate(header.length_with_row() as usize)
                .unwrap(),
        );
        let bytes = record_charge(&header, true);
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .insert(
                key,
                None,
                [0; 16],
                1,
                true,
                true,
                Location {
                    header,
                    memory: Some(extent),
                    disk: None,
                    durable: false,
                    writing: false,
                    job: None,
                    retry: Instant::now(),
                    bytes,
                },
            )
            .unwrap();
        store.pump();
        let job = store.namespaces[&id]
            .tree
            .node(&key)
            .unwrap()
            .location
            .job
            .expect("the write-back was submitted");
        // Pretend the record these bytes belong to was replaced while the write drained.
        let replacement = job + 1000;
        {
            let location = &mut store.namespaces.get_mut(&id).unwrap().tree[&key].location;
            location.job = Some(replacement);
            location.writing = true;
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while store.jobs.contains_key(&job) {
            assert!(Instant::now() < deadline, "the disk write never completed");
            store.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        let location = &store.namespaces[&id].tree[&key].location;
        assert!(
            location.disk.is_none(),
            "a stale write attached its file to the replacement record"
        );
        assert_eq!(location.job, Some(replacement));
        assert!(location.writing, "a stale completion cleared the new write");
        drop(store);
        let _ = fs::remove_dir_all(directory);
        // `_root` is the instance's shared-memory directory.
        let _ = fs::remove_dir_all(_root);
    }

    #[test]
    fn a_draining_or_opposite_transfer_id_is_fenced() {
        let (mut store, directory) = store("fence");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [6; 16],
        };
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        let get = |store: &mut Store, transfer: u64| {
            let (reply, answer) = tokio::sync::oneshot::channel();
            store.handle(StoreCommand::GetBegin {
                connection: 1,
                transfer,
                namespace: id,
                chain: vec![key],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
                events: get_events.clone(),
                reply,
            });
            answer.blocking_recv().unwrap()
        };
        let put = |store: &mut Store, transfer: u64| {
            let (reply, answer) = tokio::sync::oneshot::channel();
            store.handle(StoreCommand::PutBegin {
                connection: 1,
                transfer,
                namespace: id,
                descriptor: value.clone(),
                key,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 1,
                published_tail_pages: 1,
                row: true,
                events: put_events.clone(),
                reply,
            });
            answer.blocking_recv().unwrap()
        };
        assert!(get(&mut store, 7).is_ok());
        // The opposite direction is fenced while the Get is pending.
        assert!(put(&mut store, 7).is_err());
        // A draining Get keeps its id reserved until the disk copy drains.
        store.sessions.get_mut(&1).unwrap().ops.insert(
            8,
            Open {
                operation: Operation::Get(Box::new(Get {
                    namespace: id,
                    pins: VecDeque::new(),
                    records: Vec::new(),
                    grants: Vec::new(),
                    requested: 0,
                    next_grant: 0,
                    next_meta: 0,
                    window_reads: 0,
                    pending_reads: 1,
                    cancelled: true,
                    terminal: true,
                    finished: false,
                    events: get_events.clone(),
                    outbox: VecDeque::new(),
                })),
                progress: Instant::now(),
            },
        );
        assert!(get(&mut store, 8).is_err());
        assert!(put(&mut store, 8).is_err());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn catalog_feeds_keep_namespaces_separate() {
        let (mut store, directory) = store("multi-sub");
        let first = descriptor();
        let mut second = descriptor();
        second.pool_name = "tenant-b".to_string();
        let (sender_a, mut receiver_a) = tokio::sync::mpsc::unbounded_channel();
        let (sender_b, mut receiver_b) = tokio::sync::mpsc::unbounded_channel();
        assert!(store.catalog(first.clone(), sender_a).unwrap().is_empty());
        assert!(store.catalog(second, sender_b).unwrap().is_empty());
        let key = Key {
            consumed_tokens: 64,
            digest: [5; 16],
        };
        insert_node(&mut store, first.namespace_id(), &first, key, None, 1);
        store.delta(first.namespace_id(), CatalogKind::Added, key);
        assert!(receiver_a.try_recv().is_ok());
        assert!(receiver_b.try_recv().is_err());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_reused_get_transfer_id_is_refused() {
        let (mut store, directory) = store("dup");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store.hello(1, 0, value.clone()).unwrap();
        let key = Key {
            consumed_tokens: 64,
            digest: [4; 16],
        };
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        let begin = |store: &mut Store| {
            let (reply, answer) = tokio::sync::oneshot::channel();
            store.handle(StoreCommand::GetBegin {
                connection: 1,
                transfer: 7,
                namespace: id,
                chain: vec![key],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
                events: events.clone(),
                reply,
            });
            answer.blocking_recv().unwrap()
        };
        assert!(begin(&mut store).is_ok());
        assert!(begin(&mut store).is_err());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn rootless_eviction_only_removes_leaves_and_accounts_them() {
        let (mut store, directory) = store("evict");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        let root_key = Key {
            consumed_tokens: 64,
            digest: [2; 16],
        };
        let child_key = Key {
            consumed_tokens: 128,
            digest: [3; 16],
        };
        let root_header = crate::record::new_header(&value, root_key, None, [0; 16], 1).unwrap();
        let child_header = crate::record::new_header(&value, child_key, Some(root_key), [0; 16], 1).unwrap();
        for header in [&root_header, &child_header] {
            let extent = Arc::new(store.memory.allocate(4096).unwrap());
            let bytes = record_charge(header, true);
            store.charged += bytes;
            store
                .namespaces
                .get_mut(&id)
                .unwrap()
                .tree
                .insert(
                    header.key,
                    header.parent,
                    [0; 16],
                    1,
                    true,
                    true,
                    Location {
                        header: header.clone(),
                        memory: Some(extent),
                        disk: None,
                        durable: false,
                        writing: false,
                        job: None,
                        retry: Instant::now(),
                        bytes,
                    },
                )
                .unwrap();
        }
        let charged = store.charged;
        assert!(!store.evict_memory(id, root_key));
        assert_eq!(store.charged, charged);
        assert!(store.namespaces[&id].tree.node(&root_key).is_some());
        assert!(store.evict_memory(id, child_key));
        assert!(store.namespaces[&id].tree.node(&child_key).is_none());
        assert!(store.evict_memory(id, root_key));
        assert_eq!(store.charged, 0);
        assert!(store.namespaces[&id].tree.node(&root_key).is_none());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_chain_admits_the_root_by_successive_leaf_eviction() {
        // A full 64 KiB rootless arena: 32 -> 16 -> 16 KiB. The new record
        // needs the arena only after three successive leaf evictions.
        let (mut store, directory) = dram_store("chain", 64 * 1024);
        let value = descriptor();
        store.hello(1, std::process::id(), value.clone()).unwrap();
        let id = store.sessions[&1].namespace;
        let root = store.memory.allocate(32 * 1024).unwrap();
        let child = store.memory.allocate(16 * 1024).unwrap();
        let leaf = store.memory.allocate(16 * 1024).unwrap();
        let (r, c, l) = (
            Key {
                consumed_tokens: 128,
                digest: [1; 16],
            },
            Key {
                consumed_tokens: 192,
                digest: [2; 16],
            },
            Key {
                consumed_tokens: 256,
                digest: [3; 16],
            },
        );
        insert_memory_node(&mut store, id, &value, r, None, 2, root);
        insert_memory_node(&mut store, id, &value, c, Some(r), 1, child);
        insert_memory_node(&mut store, id, &value, l, Some(c), 1, leaf);
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: id,
                key: Key {
                    consumed_tokens: 384,
                    digest: [4; 16],
                },
                published_parent: Some(l),
                bucket: [0; 16],
                chain_pages: 6,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        let open = &store.sessions[&1].ops[&1];
        let Operation::Put(put) = &open.operation else {
            panic!("expected Put");
        };
        assert!(!put.streaming);
        assert!(put.memory.is_some());
        assert_eq!(put.header.parent, None);
        assert_eq!(put.header.tail_pages, 6);
        let tree = &store.namespaces[&id].tree;
        assert!(tree.node(&r).is_none());
        assert!(tree.node(&c).is_none());
        assert!(tree.node(&l).is_none());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_put_queued_for_credits_is_not_stalled_but_an_idle_granted_one_is() {
        let (mut store, directory) = store("queued-put");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store.hello(1, std::process::id(), value).unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(8);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        let stale = Instant::now() - NO_PROGRESS * 2;
        let live = |store: &Store| match store.sessions[&1].ops.get(&1) {
            Some(Open {
                operation: Operation::Put(put),
                ..
            }) => !put.terminal,
            _ => false,
        };
        // Waiting for the pool to ask for a window, however long: not a stall.
        store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap().progress =
            stale;
        store.tick();
        assert!(live(&store));
        // After the grant the client owes the data; idle for too long, the put is aborted.
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Put(put) = &mut open.operation else {
            panic!("expected Put");
        };
        put.requested = 1;
        assert!(store.send_put_grant(1, 1));
        store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap().progress =
            stale;
        store.tick();
        assert!(!live(&store));
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    fn idle_put(label: &str) -> (Store, PathBuf) {
        let (mut store, directory) = store(label);
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store.hello(1, std::process::id(), value).unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(8);
        let namespace = store.namespace_id().unwrap();
        let key = Key {
            consumed_tokens: 128,
            digest: [8; 16],
        };
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace,
                key,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        (store, directory)
    }

    fn put_of(store: &mut Store) -> &mut Put {
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Put(put) = &mut open.operation else {
            panic!("expected Put");
        };
        put
    }

    #[test]
    fn a_failed_put_is_failed_once_however_many_ticks_pass() {
        let (mut store, directory) = idle_put("fail-once");
        put_of(&mut store).requested = 1;
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        open.progress = Instant::now() - NO_PROGRESS * 2;
        for _ in 0..20 {
            store.tick();
        }
        assert_eq!(store.metrics.puts_error, 1);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn a_put_waiting_on_our_own_disk_is_not_a_client_stall() {
        let (mut store, directory) = idle_put("waits-on-disk");
        let put = put_of(&mut store);
        put.requested = 1;
        put.windows.granted.clear();
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        let slot = |index, job| Some(HoldSlot { index, job: Some(job) });
        put_of(&mut store).hold_slots = [slot(0, 1), slot(1, 2)];
        assert!(store.sessions[&1].ops[&1].waits_on_disk());
        put_of(&mut store).hold_slots = [None, None];
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        let arena = Arena::new(4096).unwrap();
        let extent = Arc::new(arena.allocate(64).unwrap());
        let chunk = ExtentSlice::new(extent, 0, 64);
        put_of(&mut store).blocked.push_back((1, 0, chunk));
        assert!(store.sessions[&1].ops[&1].waits_on_disk());
        // A granted window outstanding is the client's to fill.
        let grant = PlannedGrant {
            record: 0,
            record_offset: 0,
            length: 64,
            pages: 1,
            row: false,
            page_first: 0,
        };
        put_of(&mut store).windows.granted.insert(0, grant);
        assert!(!store.sessions[&1].ops[&1].waits_on_disk());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn cancel_removes_a_queued_begin() {
        let (mut store, directory) = store("cancel-queued");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), value)
            .unwrap();
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            put_events,
        );
        store.cancel(1, 1);
        assert!(store.dispatch_trace().is_empty());
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_get(
            1,
            2,
            GetRequest {
                namespace: store.namespace_id().unwrap(),
                keys: vec![Key {
                    consumed_tokens: 64,
                    digest: [4; 16],
                }],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            get_events,
        );
        store.cancel(1, 2);
        assert!(store.dispatch_trace().is_empty());
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn committed_streamed_put_finishes_on_its_closed_connection() {
        let (mut store, base, directory) = store_with_root("closed-commit", 4096);
        let value = descriptor();
        store.hello(1, std::process::id(), value.clone()).unwrap();
        let id = value.namespace_id();
        let key = Key {
            consumed_tokens: 128,
            digest: [11; 16],
        };
        let (events, mut answers) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: id,
                key,
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch();
        let (reply, answer) = tokio::sync::oneshot::channel();
        store.handle(StoreCommand::RequestWindow {
            connection: 1,
            transfer: 1,
            reply,
        });
        answer.blocking_recv().unwrap().unwrap();
        store.pump();
        assert!(matches!(
            answers.try_recv().unwrap().unwrap(),
            PutAnswer::Accept(_)
        ));
        let PutAnswer::Grant(grant) = answers.try_recv().unwrap().unwrap() else {
            panic!("expected grant")
        };
        let payload = vec![0x5a; grant.length as usize];
        // The test writer exclusively owns this granted window until WindowFilled.
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                grant
                    .hold
                    .bytes()
                    .as_ptr()
                    .cast_mut()
                    .add((grant.offset as usize) - grant.hold.offset()),
                payload.len(),
            );
        }
        let checksum = flare_selene::shared::record::checksum(&payload[..4096]);
        store
            .window_filled(1, 1, grant.index, vec![checksum; 2], Some(checksum))
            .unwrap();
        drop(grant);
        let Operation::Put(put) = &store.sessions[&1].ops[&1].operation else {
            panic!("expected Put")
        };
        assert!(put.pending > 0);
        let mut header = put.header.clone();
        header.row_checksum = checksum;
        let mut metadata = vec![0; header.page_segment as usize];
        metadata[..4096].copy_from_slice(&header.encode().unwrap());
        for page in 0..2 {
            metadata[4096 + page * 8..4104 + page * 8].copy_from_slice(&checksum.to_le_bytes());
        }
        store.put_commit(1, 1, 0, metadata.clone()).unwrap();
        store.cancel(1, 1);
        store.connection_closed(1);
        assert!(store.sessions[&1].closed);
        let deadline = Instant::now() + Duration::from_secs(5);
        while store.sessions.contains_key(&1) {
            store.pump();
            assert!(Instant::now() < deadline, "committed Put did not drain");
            std::thread::sleep(Duration::from_millis(1));
        }
        let node = &store.namespaces[&id].tree[&key];
        assert!(node.location.durable);
        let bytes = fs::read(&node.location.disk.as_ref().unwrap().path).unwrap();
        metadata.extend_from_slice(&payload);
        assert_eq!(bytes, metadata);
        record::verify_record(&bytes, &header, true).unwrap();
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn cancelled_closed_puts_do_not_submit_blocked_chunks() {
        let (mut store, base, directory) = store_with_root("detached-blocked", 4096);
        let value = descriptor();
        store.hello(1, std::process::id(), value).unwrap();
        let id = store.sessions[&1].namespace;
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: id,
                key: Key {
                    consumed_tokens: 128,
                    digest: [10; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        let open = store.sessions.get_mut(&1).unwrap().ops.get_mut(&1).unwrap();
        let Operation::Put(put) = &mut open.operation else {
            panic!("expected Put")
        };
        let slice = ExtentSlice::new(Arc::new(store.staging.allocate(4096).unwrap()), 0, 4096);
        put.blocked.push_back((777, 0, slice));
        store.connection_closed(1);
        store.pump_chunks();
        assert!(!store.jobs.contains_key(&777));
        let Operation::Put(put) = &store.sessions[&1].ops[&1].operation else {
            panic!("expected Put")
        };
        assert!(put.blocked.is_empty());
        assert!(put.cancelled);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn dispatch_prioritises_gets() {
        let (mut store, directory) = store("dispatch");
        let value = descriptor();
        let id = store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), value)
            .unwrap();
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        for index in 0..4 {
            store.submit_get(
                1,
                100 + index,
                GetRequest {
                    namespace: id,
                    keys: vec![Key {
                        consumed_tokens: 64,
                        digest: [4; 16],
                    }],
                    skip_pages: 0,
                    expected_pages: 1,
                    row: true,
                },
                get_events.clone(),
            );
        }
        for index in 0..4 {
            store.submit_put(
                1,
                200 + index,
                PutRequest {
                    namespace: id,
                    key: Key {
                        consumed_tokens: 128 + index * 64,
                        digest: [5; 16],
                    },
                    published_parent: None,
                    bucket: [0; 16],
                    chain_pages: (128 + index * 64) as u32 / 64,
                    published_tail_pages: (128 + index * 64) as u32 / 64,
                    row: true,
                },
                put_events.clone(),
            );
        }
        assert_eq!(
            store.dispatch_trace(),
            vec!["get", "get", "get", "get", "put"]
        );
        // One more round: the fifth get still precedes the queued Put.
        store.submit_get(
            1,
            300,
            GetRequest {
                namespace: id,
                keys: vec![Key {
                    consumed_tokens: 64,
                    digest: [4; 16],
                }],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            get_events,
        );
        assert_eq!(store.dispatch_trace(), vec!["get", "put"]);
        let _ = id;
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn failed_puts_drop_unsent_chunks() {
        let (mut store, base, directory) = store_with_root("failed-chunks", 4096);
        let value = descriptor();
        store
            .hello(1, std::process::id(), value)
            .unwrap();
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [9; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        {
            let operation = &mut store
                .sessions
                .get_mut(&1)
                .unwrap()
                .ops
                .get_mut(&1)
                .unwrap()
                .operation;
            let Operation::Put(put) = operation else {
                panic!("expected Put");
            };
            let slice = ExtentSlice::new(Arc::new(store.staging.allocate(4096).unwrap()), 0, 4096);
            put.blocked.push_back((999, 0, slice));
        }
        store.fail_put(1, 1, proto::Reason::Io as u8, invalid("test"));
        let operation = &store.sessions[&1].ops[&1].operation;
        let Operation::Put(put) = operation else {
            panic!("expected Put");
        };
        assert!(put.blocked.is_empty());
        store.pump_chunks();
        assert!(store.jobs.keys().all(|job| *job != 999));
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn fragmented_arena_holes_fall_back_to_streaming() {
        // 64 KiB arena with 16 KiB live at 0 and 32 KiB; the freed buddies
        // leave 32 KiB free as two separated 16 KiB holes.
        let (mut store, base, directory) = store_with_root("fragmented", 64 * 1024);
        let value = descriptor();
        store.hello(1, std::process::id(), value.clone()).unwrap();
        let id = store.sessions[&1].namespace;
        let first = store.memory.allocate(16 * 1024).unwrap();
        let second = store.memory.allocate(16 * 1024).unwrap();
        let third = store.memory.allocate(16 * 1024).unwrap();
        let fourth = store.memory.allocate(16 * 1024).unwrap();
        assert_eq!(first.offset(), 0);
        assert_eq!(second.offset(), 16 * 1024);
        assert_eq!(third.offset(), 32 * 1024);
        assert_eq!(fourth.offset(), 48 * 1024);
        drop(second);
        drop(fourth);
        insert_memory_node(
            &mut store,
            id,
            &value,
            Key {
                consumed_tokens: 64,
                digest: [1; 16],
            },
            None,
            1,
            first,
        );
        insert_memory_node(
            &mut store,
            id,
            &value,
            Key {
                consumed_tokens: 128,
                digest: [2; 16],
            },
            Some(Key {
                consumed_tokens: 64,
                digest: [1; 16],
            }),
            1,
            third,
        );
        // A 32 KiB record cannot use the holes. With no evictable leaf (the
        // occupants are not durable) the Put streams instead of refusing.
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: id,
                key: Key {
                    consumed_tokens: 192,
                    digest: [3; 16],
                },
                published_parent: Some(Key {
                    consumed_tokens: 64,
                    digest: [1; 16],
                }),
                bucket: [0; 16],
                chain_pages: 3,
                published_tail_pages: 2,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        let open = &store.sessions[&1].ops[&1];
        let Operation::Put(put) = &open.operation else {
            panic!("expected Put");
        };
        assert!(put.streaming);
        assert!(put.memory.is_none());
        assert_eq!(put.staging_hold.as_ref().unwrap().capacity(), 2 * 16 * 1024);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn queued_get_overtakes_an_earlier_put() {
        let (mut store, directory) = store("batch");
        let value = descriptor();
        store.register_namespace(value.clone()).unwrap();
        store
            .hello(1, std::process::id(), value)
            .unwrap();
        let (get_events, _get_receiver) = tokio::sync::mpsc::channel(4);
        let (put_events, _put_receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [5; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            put_events,
        );
        store.submit_get(
            1,
            2,
            GetRequest {
                namespace: store.namespace_id().unwrap(),
                keys: vec![Key {
                    consumed_tokens: 64,
                    digest: [4; 16],
                }],
                skip_pages: 0,
                expected_pages: 1,
                row: true,
            },
            get_events,
        );
        assert_eq!(store.dispatch_trace(), vec!["get", "put"]);
        drop(store);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn the_nvme_measure_counts_put_holds_and_reservations_not_windows_leased_for_claims() {
        let (store, base, directory) = store_with_root("nvme-measure", 4096);
        assert_eq!(store.charge_total(), 0);
        // Windows the pool leases from the staging region for claims and exports are not
        // records and never reach the disk: they do not use the limit.
        let leased = store
            .local
            .staging()
            .allocate(1 << 20)
            .expect("a window of staging");
        assert_eq!(store.local.staging_used(), 1 << 20);
        assert_eq!(store.charge_total(), 0);
        assert!(
            store
                .metrics_text()
                .contains("selene_nvme_reserved_bytes 0\n")
        );
        drop(leased);
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn streaming_admission_uses_the_hold() {
        let name = format!(
            "selene-unit-{}-budget-staging-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let base = std::env::temp_dir().join(&name);
        let mut config = Config::new(
            base.join("records"),
            base.join("manager.sock"),
            4096,
            64 << 20,
        );
        config.name = name.clone();
        config.staging = 8 * 1024 * 1024;
        let mut store = Store::open(config).unwrap();
        let value = descriptor();
        store.hello(1, std::process::id(), value).unwrap();
        let id = store.sessions[&1].namespace;
        let (events, _receiver) = tokio::sync::mpsc::channel(4);
        store.submit_put(
            1,
            1,
            PutRequest {
                namespace: store.namespace_id().unwrap(),
                key: Key {
                    consumed_tokens: 128,
                    digest: [7; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 2,
                published_tail_pages: 2,
                row: true,
            },
            events.clone(),
        );
        store.dispatch_trace();
        // The hold is two 16 KiB windows (32 KiB), not two negotiated 4 MiB
        // maxima, so both Puts stay in budget.
        assert!(store.sessions[&1].ops.contains_key(&1));
        store.submit_put(
            1,
            2,
            PutRequest {
                namespace: id,
                key: Key {
                    consumed_tokens: 192,
                    digest: [8; 16],
                },
                published_parent: None,
                bucket: [0; 16],
                chain_pages: 3,
                published_tail_pages: 3,
                row: true,
            },
            events,
        );
        store.dispatch_trace();
        assert!(store.sessions[&1].ops.contains_key(&2));
        assert_eq!(store.charge_total(), (20480 + 32768) + (24576 + 32768));
        store.cancel(1, 1);
        store.cancel(1, 2);
        store.pump();
        assert_eq!(store.charged, 0);
        assert_eq!(store.charge_total(), 0);
        let directory = store.shm_directory();
        drop(store);
        let _ = fs::remove_dir_all(base);
        let _ = fs::remove_dir_all(directory);
    }
    #[test]
    fn oldest_uses_each_namespaces_first_eligible_record() {
        let (mut store, directory) = store("oldest");
        let mut value = descriptor();
        let mut entries = Vec::new();
        for n in 1..=2 {
            value.identity = [n; 32];
            let id = store.register_namespace(value.clone()).unwrap();
            for pages in 1..=3 {
                let key = Key {
                    consumed_tokens: pages * 64,
                    digest: [n; 16],
                };
                let header =
                    crate::record::new_header(&value, key, None, [0; 16], pages as u32).unwrap();
                let extent = store
                    .memory
                    .allocate(record_charge(&header, true) as usize)
                    .unwrap();
                insert_memory_node(&mut store, id, &value, key, None, pages as u32, extent);
                store.reused(id, key);
                entries.push((id, key));
            }
        }
        let (id, key) = entries[0];
        store
            .namespaces
            .get_mut(&id)
            .unwrap()
            .tree
            .pin(&key)
            .unwrap();
        let (id, key) = entries[1];
        store.namespaces.get_mut(&id).unwrap().tree[&key]
            .location
            .writing = true;
        assert_eq!(store.oldest(|_| true), Some(entries[2]));
        let (id, key) = entries[2];
        store.reused(id, key);
        assert_eq!(store.oldest(|_| true), Some(entries[3]));
        assert_eq!(
            store.oldest(|node| node.key.consumed_tokens == 128),
            Some(entries[4])
        );
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn expiry_is_bounded_and_releases_manager_memory() {
        let (mut store, directory) = dram_store("expiry", 8 << 20);
        let mut value = descriptor();
        for identity in [1, 2] {
            value.identity = [identity; 32];
            let id = store.register_namespace(value.clone()).unwrap();
            store.config.max_age = Duration::from_secs(7200);
            store.namespaces.get_mut(&id).unwrap().tree.max_age = store.config.max_age;
            for n in 1..=70u64 {
                let key = Key {
                    consumed_tokens: 64,
                    digest: (n as u128).to_le_bytes(),
                };
                let header = crate::record::new_header(&value, key, None, [0; 16], 1).unwrap();
                let extent = store
                    .memory
                    .allocate(record_charge(&header, true) as usize)
                    .unwrap();
                insert_memory_node(&mut store, id, &value, key, None, 1, extent);
                store
                    .namespaces
                    .get_mut(&id)
                    .unwrap()
                    .tree
                    .node_mut(&key)
                    .unwrap()
                    .last_used = SystemTime::UNIX_EPOCH;
            }
        }
        let before = store.charged;
        let evictions = store.metrics.evictions;
        store.tick();
        assert!(store.namespaces.values().all(|ns| ns.tree.len() == 6));
        assert_eq!(store.metrics.expired, [128, 0]);
        assert_eq!(store.metrics.evictions, evictions);
        assert!(
            store
                .metrics_text()
                .contains("selene_expired_total{tier=\"dram\"} 128\n")
        );
        assert!(store.charged < before);
        store.tick();
        assert!(store.namespaces.values().all(|ns| ns.tree.len() == 6));
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }
}
