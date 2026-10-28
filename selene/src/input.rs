//! CPU prepared-input records retained beside GPU checkpoint records.

use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak, mpsc},
    time::Duration,
};

use super::record::checksum;

pub type InputKey = [u8; 32];
const HEADER: usize = 88;
const MAX_METADATA: u64 = 4 << 20;
const MAX_PAYLOAD: u64 = 1 << 30;
const QUEUE: usize = 8;
const ENTRY_BYTES: u64 = 256;
const MAX_DISK_RECORDS: usize = 4096;
pub const INDEX_HEADROOM_BYTES: u64 = MAX_DISK_RECORDS as u64 * ENTRY_BYTES;

#[derive(Clone, Debug)]
pub struct PreparedInputRecord {
    pub metadata: Arc<[u8]>,
    pub payload: Option<Arc<Vec<u8>>>,
}
impl PreparedInputRecord {
    fn bytes(&self) -> u64 {
        ENTRY_BYTES
            + self.metadata.len() as u64
            + self.payload.as_ref().map_or(0, |p| p.capacity() as u64)
    }
    fn valid(&self) -> bool {
        self.metadata.len() as u64 <= MAX_METADATA
            && self
                .payload
                .as_ref()
                .is_none_or(|p| p.capacity() as u64 <= MAX_PAYLOAD)
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct InputStats {
    pub hits: u64,
    pub misses: u64,
    pub failures: u64,
    pub stored: u64,
    pub metadata_hits: u64,
    pub metadata_misses: u64,
    pub payload_hits: u64,
    pub payload_misses: u64,
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub skipped: u64,
    pub queued: usize,
    pub pending_writes: usize,
    pub staging_bytes: u64,
    pub staging_limit: u64,
    pub index_bytes: u64,
    pub memory_limit: u64,
    pub memory_records: usize,
    pub memory_bytes: u64,
    pub disk_records: usize,
    pub disk_bytes: u64,
}

#[derive(Default)]
struct Usage {
    checkpoint: u64,
    input: u64,
    checkpoint_pressure: u64,
    input_pressure: u64,
    progress: u64,
}
/// Shared reservations for both concrete record kinds, including pending writes.
pub struct DiskBudget {
    limit: std::sync::atomic::AtomicU64,
    usage: Mutex<Usage>,
}
impl DiskBudget {
    pub fn new(limit: u64, checkpoint: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: std::sync::atomic::AtomicU64::new(limit),
            usage: Mutex::new(Usage {
                checkpoint,
                ..Default::default()
            }),
        })
    }
    fn limit(&self) -> u64 {
        self.limit.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn checkpoint_limit(&self) -> u64 {
        self.limit()
            .saturating_sub(self.usage.lock().unwrap().input)
    }
    pub fn set_checkpoint(&self, bytes: u64) {
        self.usage.lock().unwrap().checkpoint = bytes;
    }
    /// Lower the limit before any record is indexed, when every host's
    /// directory must hold the same records and the smallest host bounds them.
    pub fn set_limit(&self, limit: u64) {
        let mut usage = self.usage.lock().unwrap();
        self.limit
            .store(limit, std::sync::atomic::Ordering::Relaxed);
        usage.checkpoint = usage.checkpoint.min(limit);
    }
    pub fn reserve_checkpoint(&self, bytes: u64) -> bool {
        let mut u = self.usage.lock().unwrap();
        if u.checkpoint.saturating_add(u.input).saturating_add(bytes) > self.limit() {
            u.input_pressure = u.input_pressure.max(bytes);
            false
        } else {
            u.checkpoint += bytes;
            true
        }
    }
    pub fn request_input_reclaim(&self, bytes: u64) {
        let mut u = self.usage.lock().unwrap();
        u.input_pressure = u.input_pressure.max(bytes);
    }
    pub fn checkpoint_pressure(&self) -> u64 {
        self.usage.lock().unwrap().checkpoint_pressure
    }
    fn reserve_input(&self, bytes: u64) -> bool {
        let mut u = self.usage.lock().unwrap();
        if u.checkpoint.saturating_add(u.input).saturating_add(bytes) > self.limit() {
            u.checkpoint_pressure = u.checkpoint_pressure.max(bytes);
            false
        } else {
            u.input += bytes;
            u.checkpoint_pressure = 0;
            true
        }
    }
    fn clear_checkpoint_pressure(&self) {
        self.usage.lock().unwrap().checkpoint_pressure = 0;
    }
    pub fn progress(&self) -> u64 {
        self.usage.lock().unwrap().progress
    }
    fn release_input(&self, bytes: u64) {
        let mut u = self.usage.lock().unwrap();
        u.input = u.input.saturating_sub(bytes);
        u.progress += 1;
    }
    fn input_pressure(&self) -> u64 {
        self.usage.lock().unwrap().input_pressure
    }
    fn room_for_checkpoint(&self, bytes: u64) -> bool {
        let u = self.usage.lock().unwrap();
        u.checkpoint.saturating_add(u.input).saturating_add(bytes) <= self.limit()
    }
    fn clear_input_pressure(&self) {
        self.usage.lock().unwrap().input_pressure = 0;
    }
}

struct Staging {
    limit: u64,
    queued: u64,
    leases: Vec<(Weak<PreparedInputRecord>, u64, bool)>,
}
impl Staging {
    fn used(&mut self) -> u64 {
        self.leases
            .retain(|(record, _, retained)| record.strong_count() > usize::from(*retained));
        self.queued + self.leases.iter().map(|(_, bytes, _)| bytes).sum::<u64>()
    }
}
struct Reservation {
    staging: Arc<Mutex<Staging>>,
    bytes: u64,
}
impl Reservation {
    fn new(staging: &Arc<Mutex<Staging>>, bytes: u64) -> Option<Self> {
        let mut state = staging.lock().unwrap();
        if state.used().saturating_add(bytes) > state.limit {
            return None;
        }
        state.queued += bytes;
        Some(Self {
            staging: staging.clone(),
            bytes,
        })
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut s = self.staging.lock().unwrap();
        s.queued = s.queued.saturating_sub(self.bytes);
    }
}

type Waiters =
    Arc<Mutex<HashMap<(InputKey, bool), Vec<mpsc::SyncSender<Option<Arc<PreparedInputRecord>>>>>>>;
enum Job {
    Invalidate(InputKey),
    Get(InputKey, bool),
    Put(InputKey, Arc<PreparedInputRecord>, Reservation),
}
struct InputLookup {
    receiver: mpsc::Receiver<Option<Arc<PreparedInputRecord>>>,
}
struct Owner {
    source_stripes: [Mutex<()>; 64],
    staging: Arc<Mutex<Staging>>,
    waiters: Waiters,
    sender: Mutex<Option<mpsc::SyncSender<Job>>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    stats: Arc<Mutex<InputStats>>,
    directory_guard: Mutex<Option<super::record::DirectoryLock>>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.sender.get_mut().unwrap().take();
        if let Some(t) = self.thread.get_mut().unwrap().take() {
            let _ = t.join();
        }
    }
}
#[derive(Clone)]
pub struct InputStore {
    owner: Arc<Owner>,
}
impl InputStore {
    #[cfg(any(test, feature = "test-support"))]
    pub fn start(
        memory: u64,
        directory: Option<PathBuf>,
        budget: Arc<DiskBudget>,
    ) -> crate::Result<Self> {
        Self::start_bounded(
            memory,
            directory,
            budget,
            MAX_METADATA + MAX_PAYLOAD + ENTRY_BYTES,
        )
    }
    pub fn start_bounded(
        memory: u64,
        directory: Option<PathBuf>,
        budget: Arc<DiskBudget>,
        staging_capacity: u64,
    ) -> crate::Result<Self> {
        let staging = Arc::new(Mutex::new(Staging {
            limit: staging_capacity,
            queued: 0,
            leases: Vec::new(),
        }));
        let worker_staging = staging.clone();
        let (sender, receiver) = mpsc::sync_channel(QUEUE);
        let stats = Arc::new(Mutex::new(InputStats {
            memory_limit: memory,
            ..InputStats::default()
        }));
        let shared = stats.clone();
        let waiters: Waiters = Arc::new(Mutex::new(HashMap::new()));
        let pending = waiters.clone();
        let (ready, startup) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("flare-input-records".into())
            .spawn(move || {
                flare_eos::lower_priority(super::disk::NICE);
                match Worker::new(memory, directory, budget, shared, worker_staging) {
                    Ok(worker) => {
                        if ready.send(Ok(())).is_ok() {
                            worker.run(receiver, pending);
                        }
                    }
                    Err(error) => {
                        let _ = ready.send(Err(error));
                    }
                }
            })
            .map_err(|error| super::record::io_failure("spawn prepared-input worker", error))?;
        match startup.recv() {
            Ok(Ok(())) => (),
            result => {
                drop(sender);
                let _ = thread.join();
                return Err(match result {
                    Ok(Err(error)) => error,
                    Err(error) => {
                        format!("prepared-input worker failed during startup: {error}").into()
                    }
                    Ok(Ok(())) => unreachable!(),
                });
            }
        }
        Ok(Self {
            owner: Arc::new(Owner {
                source_stripes: std::array::from_fn(|_| Mutex::new(())),
                staging,
                waiters,
                sender: Mutex::new(Some(sender)),
                thread: Mutex::new(Some(thread)),
                stats,
                directory_guard: Mutex::new(None),
            }),
        })
    }
    pub fn source_guard(&self, key: InputKey) -> std::sync::MutexGuard<'_, ()> {
        let index = u64::from_le_bytes(key[..8].try_into().unwrap()) as usize
            % self.owner.source_stripes.len();
        self.owner.source_stripes[index].lock().unwrap()
    }
    pub fn retain_directory(&self, guard: super::record::DirectoryLock) {
        *self.owner.directory_guard.lock().unwrap() = Some(guard);
    }

    /// Queue saturation and I/O failures are cache misses; callers retain original inputs.
    fn lookup(&self, key: InputKey, payload: bool) -> Option<InputLookup> {
        let (reply, receiver) = mpsc::sync_channel(1);
        let mut waiters = self.owner.waiters.lock().unwrap();
        if let Some(existing) = waiters.get_mut(&(key, payload)) {
            if existing.len() >= QUEUE {
                self.owner.stats.lock().unwrap().skipped += 1;
                return None;
            }
            existing.push(reply);
            return Some(InputLookup { receiver });
        }
        waiters.insert((key, payload), vec![reply]);
        self.owner.stats.lock().unwrap().queued += 1;
        let submitted = self
            .owner
            .sender
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|sender| sender.try_send(Job::Get(key, payload)).is_ok());
        if !submitted {
            waiters.remove(&(key, payload));
            let mut stats = self.owner.stats.lock().unwrap();
            stats.skipped += 1;
            stats.queued = stats.queued.saturating_sub(1);
            return None;
        }
        Some(InputLookup { receiver })
    }
    pub fn get_blocking(&self, key: InputKey, payload: bool) -> Option<Arc<PreparedInputRecord>> {
        self.lookup(key, payload)?.receiver.recv().ok().flatten()
    }
    /// Drop a record rejected by the consumer's typed metadata or payload validation.
    pub fn invalidate(&self, key: InputKey) {
        if let Some(sender) = self.owner.sender.lock().unwrap().as_ref() {
            self.owner.stats.lock().unwrap().queued += 1;
            if sender.try_send(Job::Invalidate(key)).is_err() {
                let mut stats = self.owner.stats.lock().unwrap();
                stats.queued = stats.queued.saturating_sub(1);
                stats.skipped += 1;
            }
        }
    }
    pub fn insert(&self, key: InputKey, record: Arc<PreparedInputRecord>) {
        if !record.valid() {
            self.owner.stats.lock().unwrap().skipped += 1;
            return;
        }
        let Some(reservation) = Reservation::new(&self.owner.staging, record.bytes()) else {
            self.owner.stats.lock().unwrap().skipped += 1;
            return;
        };
        if let Some(sender) = self.owner.sender.lock().unwrap().as_ref() {
            self.owner.stats.lock().unwrap().queued += 1;
            if sender.try_send(Job::Put(key, record, reservation)).is_err() {
                let mut stats = self.owner.stats.lock().unwrap();
                stats.queued = stats.queued.saturating_sub(1);
                stats.skipped += 1;
            }
        }
    }
    /// Reserve serialized storage before allocating an encoded CPU input record.
    pub fn insert_with(
        &self,
        key: InputKey,
        expected_serialized_bytes: u64,
        encode: impl FnOnce() -> crate::Result<PreparedInputRecord>,
    ) -> crate::Result<bool> {
        let Some(bytes) = expected_serialized_bytes.checked_add(ENTRY_BYTES) else {
            return Ok(false);
        };
        let Some(reservation) = Reservation::new(&self.owner.staging, bytes) else {
            self.owner.stats.lock().unwrap().skipped += 1;
            return Ok(false);
        };
        let record = match encode() {
            Ok(record) if record.valid() && record.bytes() <= bytes => Arc::new(record),
            Ok(_) => {
                self.owner.stats.lock().unwrap().skipped += 1;
                return Ok(false);
            }
            Err(error) => {
                self.owner.stats.lock().unwrap().failures += 1;
                return Err(error);
            }
        };
        let sender = self.owner.sender.lock().unwrap();
        let Some(sender) = sender.as_ref() else {
            return Ok(false);
        };
        self.owner.stats.lock().unwrap().queued += 1;
        if sender.try_send(Job::Put(key, record, reservation)).is_err() {
            let mut stats = self.owner.stats.lock().unwrap();
            stats.queued = stats.queued.saturating_sub(1);
            stats.skipped += 1;
            return Ok(false);
        }
        Ok(true)
    }
    pub fn stats(&self) -> InputStats {
        let mut snapshot = *self.owner.stats.lock().unwrap();
        let mut staging = self.owner.staging.lock().unwrap();
        snapshot.staging_bytes = staging.used();
        snapshot.staging_limit = staging.limit;
        snapshot
    }
}
struct MemoryEntry {
    record: Arc<PreparedInputRecord>,
    touched: u64,
}
struct DiskEntry {
    bytes: u64,
    touched: u64,
    has_payload: bool,
}
struct Worker {
    staging: Arc<Mutex<Staging>>,
    limit: u64,
    directory: Option<PathBuf>,
    budget: Arc<DiskBudget>,
    stats: Arc<Mutex<InputStats>>,
    memory: HashMap<InputKey, MemoryEntry>,
    leased: Vec<(Weak<PreparedInputRecord>, u64)>,
    disk: HashMap<InputKey, DiskEntry>,
    clock: u64,
    orphan_bytes: u64,
    disk_writes_blocked: bool,
}
impl Worker {
    fn new(
        limit: u64,
        directory: Option<PathBuf>,
        budget: Arc<DiskBudget>,
        stats: Arc<Mutex<InputStats>>,
        staging: Arc<Mutex<Staging>>,
    ) -> crate::Result<Self> {
        let mut w = Self {
            staging,
            limit,
            directory,
            budget,
            stats,
            memory: HashMap::new(),
            leased: Vec::new(),
            disk: HashMap::new(),
            clock: 0,
            orphan_bytes: 0,
            disk_writes_blocked: false,
        };
        if let Err(error) = w.scan() {
            w.budget
                .release_input(w.disk.values().map(|entry| entry.bytes).sum());
            return Err(error);
        }
        w.refresh();
        Ok(w)
    }
    fn scan(&mut self) -> crate::Result<()> {
        let Some(directory) = &self.directory else {
            return Ok(());
        };
        let failure = |action: &str, path: &Path, error| {
            super::record::io_failure(format!("{action} {}", path.display()), error)
        };
        for entry in fs::read_dir(directory)
            .map_err(|error| failure("scan prepared-input directory", directory, error))?
        {
            let entry = entry.map_err(|error| {
                failure("read prepared-input directory entry", directory, error)
            })?;
            let path = entry.path();
            let kind = entry
                .file_type()
                .map_err(|error| failure("inspect prepared-input directory entry", &path, error))?;
            if !kind.is_file() {
                continue;
            }
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_temporary_name)
            {
                fs::remove_file(&path).map_err(|error| {
                    failure("remove incomplete prepared-input record", &path, error)
                })?;
                continue;
            }
            if path
                .extension()
                .is_none_or(|extension| extension != "input")
            {
                continue;
            }
            match inspect(&path) {
                Ok((key, bytes, has_payload))
                    if self.disk.len() < MAX_DISK_RECORDS && self.budget.reserve_input(bytes) =>
                {
                    self.disk.insert(
                        key,
                        DiskEntry {
                            bytes,
                            touched: 0,
                            has_payload,
                        },
                    );
                }
                Err(error) if error.kind() != std::io::ErrorKind::InvalidData => {
                    return Err(failure("read prepared-input record", &path, error));
                }
                _ => fs::remove_file(&path)
                    .map_err(|error| failure("remove prepared-input record", &path, error))?,
            }
        }
        Ok(())
    }
    fn completed(&self) {
        let mut s = self.stats.lock().unwrap();
        s.queued = s.queued.saturating_sub(1);
    }
    fn run(mut self, receiver: mpsc::Receiver<Job>, waiters: Waiters) {
        flare_eos::lower_priority(super::disk::NICE);
        let mut pending: std::collections::VecDeque<(
            InputKey,
            Arc<PreparedInputRecord>,
            std::time::Instant,
            Reservation,
        )> = std::collections::VecDeque::new();
        let mut disconnected = false;
        loop {
            self.reclaim();
            self.budget.clear_checkpoint_pressure();
            let waiting = pending.len();
            for _ in 0..waiting {
                let (key, record, until, reservation) = pending.pop_front().unwrap();
                if !self.write(key, record.clone()) {
                    if std::time::Instant::now() < until {
                        pending.push_back((key, record, until, reservation));
                    } else {
                        self.stats.lock().unwrap().skipped += 1;
                        self.completed();
                    }
                } else {
                    self.completed();
                }
            }
            if disconnected && pending.is_empty() {
                break;
            }
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(Job::Invalidate(key)) => {
                    let before = pending.len();
                    pending.retain(|(pending_key, _, _, _)| *pending_key != key);
                    for _ in pending.len()..before {
                        self.completed();
                    }
                    if let Some(entry) = self.memory.remove(&key) {
                        self.leased
                            .push((Arc::downgrade(&entry.record), entry.record.bytes()));
                    }
                    self.remove_disk(&key);
                    self.completed();
                }
                Ok(Job::Get(key, payload)) => {
                    self.clock += 1;
                    let value = self.get(key, payload);
                    let mut s = self.stats.lock().unwrap();
                    s.queued = s.queued.saturating_sub(1);
                    if value.is_some() {
                        s.hits += 1;
                        if payload {
                            s.payload_hits += 1;
                        } else {
                            s.metadata_hits += 1;
                        }
                    } else {
                        s.misses += 1;
                        if payload {
                            s.payload_misses += 1;
                        } else {
                            s.metadata_misses += 1;
                        }
                    }
                    drop(s);
                    if let Some(replies) = waiters.lock().unwrap().remove(&(key, payload)) {
                        for reply in replies {
                            let _ = reply.try_send(value.clone());
                        }
                    }
                }
                Ok(Job::Put(key, record, reservation)) => {
                    self.clock += 1;
                    self.retain(key, record.clone());
                    if !self.write(key, record.clone()) {
                        if pending.len() < QUEUE {
                            pending.push_back((
                                key,
                                record,
                                std::time::Instant::now() + Duration::from_secs(1),
                                reservation,
                            ));
                        } else {
                            self.stats.lock().unwrap().skipped += 1;
                            self.completed();
                        }
                    } else {
                        self.completed();
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    disconnected = true;
                    if !pending.is_empty() {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
            self.stats.lock().unwrap().pending_writes = pending.len();
            self.refresh();
        }
    }
    fn get(&mut self, key: InputKey, payload: bool) -> Option<Arc<PreparedInputRecord>> {
        if let Some(entry) = self.memory.get_mut(&key) {
            if !payload || entry.record.payload.is_some() {
                entry.touched = self.clock;
                self.stats.lock().unwrap().memory_hits += 1;
                return Some(entry.record.clone());
            }
        }
        let entry = self.disk.get_mut(&key)?;
        if payload && !entry.has_payload {
            return None;
        }
        entry.touched = self.clock;
        let path = self.directory.as_ref()?.join(file_name(&key));
        let metadata_bytes = match inspect_metadata_bytes(&path) {
            Ok(bytes) => bytes,
            Err(_) => {
                self.stats.lock().unwrap().failures += 1;
                self.remove_disk(&key);
                return None;
            }
        };
        let bytes = if payload {
            entry
                .bytes
                .saturating_add(metadata_bytes)
                .saturating_add(ENTRY_BYTES)
        } else {
            metadata_bytes
                .saturating_mul(2)
                .saturating_add(ENTRY_BYTES + HEADER as u64)
        };
        let reservation = Reservation::new(&self.staging, bytes)?;
        match read_limited(&path, key, payload, bytes) {
            Ok(record) => {
                self.stats.lock().unwrap().disk_hits += 1;
                let record = Arc::new(record);
                self.retain(key, record.clone());
                let retained = self
                    .memory
                    .get(&key)
                    .is_some_and(|e| Arc::ptr_eq(&e.record, &record));
                self.staging.lock().unwrap().leases.push((
                    Arc::downgrade(&record),
                    bytes,
                    retained,
                ));
                drop(reservation);
                Some(record)
            }
            Err(_) => {
                self.stats.lock().unwrap().failures += 1;
                self.remove_disk(&key);
                None
            }
        }
    }
    fn memory_bytes(&mut self) -> u64 {
        self.leased.retain(|(weak, _)| weak.strong_count() != 0);
        self.memory.values().map(|e| e.record.bytes()).sum::<u64>()
            + self.leased.iter().map(|(_, bytes)| bytes).sum::<u64>()
    }
    fn retain(&mut self, key: InputKey, record: Arc<PreparedInputRecord>) {
        if self.limit == 0 || record.bytes() > self.limit {
            return;
        }
        if let Some(entry) = self.memory.get(&key) {
            if entry.record.payload.is_some() || record.payload.is_none() {
                return;
            }
        }
        if let Some(entry) = self.memory.remove(&key) {
            self.leased
                .push((Arc::downgrade(&entry.record), entry.record.bytes()));
        }
        while self.memory_bytes().saturating_add(record.bytes()) > self.limit {
            let Some(key) = self
                .memory
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.record) == 1)
                .min_by_key(|(_, e)| e.touched)
                .map(|(k, _)| *k)
            else {
                return;
            };
            self.memory.remove(&key);
        }
        self.memory.insert(
            key,
            MemoryEntry {
                record,
                touched: self.clock,
            },
        );
    }
    fn remove_disk(&mut self, key: &InputKey) -> bool {
        if !self.disk.contains_key(key) {
            return false;
        }
        let Some(directory) = &self.directory else {
            return false;
        };
        match fs::remove_file(directory.join(file_name(key))) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(_) => return false,
        }
        let entry = self.disk.remove(key).unwrap();
        self.budget.release_input(entry.bytes);
        true
    }
    fn evict_disk(&mut self) -> bool {
        let Some(key) = self
            .disk
            .iter()
            .min_by_key(|(_, e)| e.touched)
            .map(|(k, _)| *k)
        else {
            return false;
        };
        self.remove_disk(&key)
    }
    fn reclaim(&mut self) {
        let pressure = self.budget.input_pressure();
        if pressure != 0 {
            while !self.budget.room_for_checkpoint(pressure) && self.evict_disk() {}
            self.budget.clear_input_pressure();
        }
    }
    fn write(&mut self, key: InputKey, record: Arc<PreparedInputRecord>) -> bool {
        let Some(dir) = self.directory.clone() else {
            return true;
        };
        if self.disk_writes_blocked {
            self.stats.lock().unwrap().skipped += 1;
            return true;
        }
        if self
            .disk
            .get(&key)
            .is_some_and(|entry| entry.has_payload || record.payload.is_none())
        {
            return true;
        }
        while !self.disk.contains_key(&key) && self.disk.len() >= MAX_DISK_RECORDS {
            if !self.evict_disk() {
                return true;
            }
        }
        let bytes = HEADER as u64
            + record.metadata.len() as u64
            + record.payload.as_ref().map_or(0, |p| p.len() as u64);
        if bytes > self.budget.limit() {
            self.stats.lock().unwrap().skipped += 1;
            return true;
        }
        while !self.budget.reserve_input(bytes) {
            if !self.evict_disk() {
                return false;
            }
        }
        let path = dir.join(file_name(&key));
        match write(&path, key, &record) {
            Ok(()) => {
                if let Some(previous) = self.disk.insert(
                    key,
                    DiskEntry {
                        bytes,
                        touched: self.clock,
                        has_payload: record.payload.is_some(),
                    },
                ) {
                    self.budget.release_input(previous.bytes);
                }
                self.stats.lock().unwrap().stored += 1;
            }
            Err(WriteFailure {
                temporary_retained,
                error: _error,
            }) => {
                if temporary_retained {
                    self.orphan_bytes += bytes;
                    self.disk_writes_blocked = true;
                } else {
                    self.budget.release_input(bytes);
                }
                self.stats.lock().unwrap().failures += 1;
            }
        }
        true
    }
    fn refresh(&mut self) {
        let bytes = self.memory_bytes();
        let mut s = self.stats.lock().unwrap();
        s.memory_records = self.memory.len();
        s.memory_bytes = bytes;
        s.disk_records = self.disk.len();
        s.index_bytes = self.disk.len() as u64 * ENTRY_BYTES;
        s.memory_limit = self.limit;
        s.disk_bytes = self.disk.values().map(|e| e.bytes).sum::<u64>() + self.orphan_bytes;
    }
}
fn is_temporary_name(name: &str) -> bool {
    name.strip_suffix(".input.pending").is_some_and(|key| {
        key.len() == 64
            && key
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
fn file_name(key: &InputKey) -> String {
    let mut name = String::with_capacity(70);
    for byte in key {
        use std::fmt::Write;
        write!(name, "{byte:02x}").unwrap();
    }
    name.push_str(".input");
    name
}
fn header(key: InputKey, record: &PreparedInputRecord) -> [u8; HEADER] {
    let mut h = [0; HEADER];
    h[..8].copy_from_slice(b"FLAREINP");
    h[8..12].copy_from_slice(&1u32.to_le_bytes());
    h[12..16].copy_from_slice(&u32::from(record.payload.is_some()).to_le_bytes());
    h[16..48].copy_from_slice(&key);
    h[48..56].copy_from_slice(&(record.metadata.len() as u64).to_le_bytes());
    h[56..64]
        .copy_from_slice(&(record.payload.as_ref().map_or(0, |p| p.len()) as u64).to_le_bytes());
    h[64..72].copy_from_slice(&checksum(&record.metadata).to_le_bytes());
    h[72..80].copy_from_slice(
        &record
            .payload
            .as_ref()
            .map_or(0, |p| checksum(p))
            .to_le_bytes(),
    );
    let sum = checksum(&h[..80]);
    h[80..88].copy_from_slice(&sum.to_le_bytes());
    h
}
fn parse(h: &[u8; HEADER], length: u64) -> std::io::Result<(InputKey, usize, usize, bool)> {
    let value = |offset| u64::from_le_bytes(h[offset..offset + 8].try_into().unwrap());
    let metadata = value(48);
    let payload = value(56);
    if checksum(&h[..80]) != value(80)
        || &h[..8] != b"FLAREINP"
        || h[8..12] != 1u32.to_le_bytes()
        || h[12..16] != [0, 0, 0, 0] && h[12..16] != [1, 0, 0, 0]
        || metadata > MAX_METADATA
        || payload > MAX_PAYLOAD
        || HEADER as u64 + metadata + payload != length
        || h[12] == 0 && payload != 0
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid prepared-input header",
        ));
    }
    Ok((
        h[16..48].try_into().unwrap(),
        metadata as usize,
        payload as usize,
        h[12] == 1,
    ))
}
fn inspect_metadata_bytes(path: &Path) -> std::io::Result<u64> {
    let mut f = fs::File::open(path)?;
    let length = f.metadata()?.len();
    let mut h = [0; HEADER];
    f.read_exact(&mut h)?;
    let (_, metadata, _, _) = parse(&h, length)?;
    Ok(metadata as u64)
}
fn inspect(path: &Path) -> std::io::Result<(InputKey, u64, bool)> {
    let mut f = fs::File::open(path)?;
    let length = f.metadata()?.len();
    if length < HEADER as u64 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated prepared-input header",
        ));
    }
    let mut h = [0; HEADER];
    f.read_exact(&mut h)?;
    let (key, _, _, has_payload) = parse(&h, length)?;
    if path
        .file_name()
        .is_none_or(|n| n != file_name(&key).as_str())
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "prepared-input key differs from file name",
        ));
    }
    Ok((key, length, has_payload))
}
#[cfg(test)]
fn read(path: &Path, expected: InputKey, payload: bool) -> std::io::Result<PreparedInputRecord> {
    read_limited(path, expected, payload, u64::MAX)
}
fn read_limited(
    path: &Path,
    expected: InputKey,
    payload: bool,
    limit: u64,
) -> std::io::Result<PreparedInputRecord> {
    let mut f = fs::File::open(path)?;
    let length = f.metadata()?.len();
    let mut h = [0; HEADER];
    f.read_exact(&mut h)?;
    let (key, m, p, present) = parse(&h, length)?;
    if key != expected {
        return Err(std::io::Error::other("prepared-input identity mismatch"));
    }
    let bytes = HEADER as u64 + 2 * m as u64 + if payload { p as u64 } else { 0 } + ENTRY_BYTES;
    if bytes > limit {
        return Err(std::io::Error::other(
            "prepared input exceeds reserved read window",
        ));
    }
    let mut metadata = vec![0; m];
    f.read_exact(&mut metadata)?;
    if checksum(&metadata) != u64::from_le_bytes(h[64..72].try_into().unwrap()) {
        return Err(std::io::Error::other(
            "prepared-input metadata checksum mismatch",
        ));
    }
    let data = if payload && present {
        let mut data = vec![0; p];
        f.read_exact(&mut data)?;
        if checksum(&data) != u64::from_le_bytes(h[72..80].try_into().unwrap()) {
            return Err(std::io::Error::other(
                "prepared-input payload checksum mismatch",
            ));
        }
        Some(Arc::new(data))
    } else {
        None
    };
    if payload && !present {
        return Err(std::io::Error::other("prepared-input payload absent"));
    }
    Ok(PreparedInputRecord {
        metadata: metadata.into(),
        payload: data,
    })
}
#[derive(Debug)]
struct WriteFailure {
    temporary_retained: bool,
    error: std::io::Error,
}
fn write(
    path: &Path,
    key: InputKey,
    record: &PreparedInputRecord,
) -> std::result::Result<(), WriteFailure> {
    let temporary = path.with_extension("input.pending");
    let result: std::io::Result<()> = (|| {
        let mut f = fs::File::create(&temporary)?;
        f.write_all(&header(key, record))?;
        f.write_all(&record.metadata)?;
        if let Some(payload) = &record.payload {
            f.write_all(payload)?;
        }
        f.sync_data()?;
        drop(f);
        fs::rename(&temporary, path)
    })();
    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            let temporary_retained = match fs::remove_file(&temporary) {
                Ok(()) => false,
                Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => false,
                Err(_) => true,
            };
            Err(WriteFailure {
                temporary_retained,
                error,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "flare-input-test-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn record(payload: bool) -> Arc<PreparedInputRecord> {
        Arc::new(PreparedInputRecord {
            metadata: Arc::from(&b"geometry"[..]),
            payload: payload.then(|| Arc::new(b"tensor bytes".to_vec())),
        })
    }
    fn wait(store: &InputStore, predicate: impl Fn(InputStats) -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !predicate(store.stats()) {
            assert!(
                std::time::Instant::now() < deadline,
                "stats {:?}",
                store.stats()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    #[test]
    fn codec_metadata_is_lazy_and_corruption_is_rejected() {
        let dir = Directory::new();
        let key = [7; 32];
        let path = dir.0.join(file_name(&key));
        write(&path, key, &record(true)).unwrap();
        assert!(read(&path, key, false).unwrap().payload.is_none());
        assert_eq!(
            &*read(&path, key, true).unwrap().payload.unwrap(),
            b"tensor bytes"
        );
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(read(&path, key, false).is_ok());
        assert!(read(&path, key, true).is_err());
        bytes[48] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(read(&path, key, false).is_err());
    }
    #[test]
    fn codec_bounds_lengths_before_allocating_and_rejects_wrong_keys() {
        let dir = Directory::new();
        let key = [1; 32];
        let path = dir.0.join(file_name(&key));
        write(&path, key, &record(false)).unwrap();
        assert!(read(&path, [2; 32], false).is_err());
        let mut h = header(key, &record(false));
        h[48..56].copy_from_slice(&(MAX_METADATA + 1).to_le_bytes());
        let sum = checksum(&h[..80]);
        h[80..88].copy_from_slice(&sum.to_le_bytes());
        assert!(parse(&h, HEADER as u64 + MAX_METADATA + 1).is_err());
    }
    #[test]
    fn memory_only_reuses_payload_and_zero_memory_retains_nothing() {
        for limit in [0, 4096] {
            let store = InputStore::start(limit, None, DiskBudget::new(0, 0)).unwrap();
            store.insert([1; 32], record(true));
            let found = store.get_blocking([1; 32], true);
            assert_eq!(found.is_some(), limit != 0);
            wait(&store, |s| s.queued == 0);
            assert_eq!(store.stats().memory_records, usize::from(limit != 0));
        }
    }
    #[test]
    fn disk_only_restarts_and_metadata_can_be_upgraded() {
        let dir = Directory::new();
        let key = [3; 32];
        {
            let store =
                InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
            store.insert(key, record(false));
            wait(&store, |s| s.stored == 1);
            assert!(store.get_blocking(key, false).is_some());
            assert!(store.get_blocking(key, true).is_none());
            store.insert(key, record(true));
            wait(&store, |s| s.stored == 2);
            assert_eq!(store.stats().memory_bytes, 0);
        }
        let store = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
        assert!(store.get_blocking(key, true).is_some());
        assert_eq!(store.stats().disk_records, 1);
    }
    #[test]
    fn live_leases_prevent_memory_eviction_and_release_allows_replacement() {
        let store = InputStore::start(record(true).bytes(), None, DiskBudget::new(0, 0)).unwrap();
        store.insert([1; 32], record(true));
        let lease = store.get_blocking([1; 32], true).unwrap();
        store.insert([2; 32], record(true));
        assert!(store.get_blocking([2; 32], true).is_none());
        assert_eq!(&*lease.metadata, b"geometry");
        drop(lease);
        store.insert([2; 32], record(true));
        assert!(store.get_blocking([2; 32], true).is_some());
        assert!(store.get_blocking([1; 32], true).is_none());
    }
    #[test]
    fn shared_budget_reclaims_inputs_and_checkpoint_pressure_allows_pending_writes() {
        let dir = Directory::new();
        let size = HEADER as u64
            + record(true).metadata.len() as u64
            + record(true).payload.as_ref().unwrap().len() as u64;
        let budget = DiskBudget::new(size, 0);
        let store = InputStore::start(0, Some(dir.0.clone()), budget.clone()).unwrap();
        store.insert([1; 32], record(true));
        wait(&store, |s| s.stored == 1);
        assert!(!budget.reserve_checkpoint(size));
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !budget.reserve_checkpoint(size) {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        store.insert([2; 32], record(true));
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while budget.checkpoint_pressure() == 0 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        budget.set_checkpoint(0);
        wait(&store, |s| s.stored == 2);
        assert!(store.get_blocking([2; 32], true).is_some());
        assert!(store.stats().disk_bytes <= size);
    }
    #[test]
    fn canceled_lookup_does_not_prevent_other_waiters() {
        let store = InputStore::start(4096, None, DiskBudget::new(0, 0)).unwrap();
        store.insert([1; 32], record(true));
        let canceled = store.lookup([1; 32], true);
        let second = store.lookup([1; 32], true).unwrap();
        drop(canceled);
        assert!(second.receiver.recv().unwrap().is_some());
    }
    #[test]
    fn memory_and_disk_read_through_after_memory_eviction() {
        let dir = Directory::new();
        let store = InputStore::start(
            record(true).bytes(),
            Some(dir.0.clone()),
            DiskBudget::new(4096, 0),
        )
        .unwrap();
        store.insert([1; 32], record(true));
        wait(&store, |s| s.stored == 1);
        store.insert([2; 32], record(true));
        wait(&store, |s| s.stored == 2);
        let first = store.get_blocking([1; 32], true).unwrap();
        assert_eq!(&*first.metadata, b"geometry");
        assert_eq!(store.stats().disk_hits, 1);
        assert!(store.stats().memory_bytes <= record(true).bytes());
    }

    #[test]
    fn disk_limit_evicts_records_and_skips_oversized_writes() {
        let dir = Directory::new();
        let bytes = HEADER as u64
            + record(true).metadata.len() as u64
            + record(true).payload.as_ref().unwrap().len() as u64;
        let store = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(bytes, 0)).unwrap();
        store.insert([1; 32], record(true));
        wait(&store, |s| s.stored == 1);
        store.insert([2; 32], record(true));
        wait(&store, |s| s.stored == 2);
        assert!(store.get_blocking([1; 32], true).is_none());
        assert!(store.get_blocking([2; 32], true).is_some());
        store.insert(
            [3; 32],
            Arc::new(PreparedInputRecord {
                metadata: Arc::from(&b"geometry"[..]),
                payload: Some(Arc::new(vec![7; 4096])),
            }),
        );
        assert!(store.get_blocking([3; 32], true).is_none());
        assert!(store.get_blocking([2; 32], true).is_some());
        assert_eq!(store.stats().disk_records, 1);
    }

    #[test]
    fn disk_read_leases_and_write_queue_share_the_staging_limit() {
        let dir = Directory::new();
        let capacity = record(true).bytes() + HEADER as u64 + record(true).metadata.len() as u64;
        let store =
            InputStore::start_bounded(0, Some(dir.0.clone()), DiskBudget::new(4096, 0), capacity)
                .unwrap();
        store.insert([1; 32], record(true));
        wait(&store, |s| s.stored == 1);
        store.insert([2; 32], record(true));
        wait(&store, |s| s.stored == 2);
        let lease = store.get_blocking([1; 32], true).unwrap();
        assert!(store.get_blocking([2; 32], true).is_none());
        store.insert([3; 32], record(true));
        assert!(store.stats().skipped > 0);
        assert!(store.stats().staging_bytes <= capacity);
        drop(lease);
        assert!(store.get_blocking([2; 32], true).is_some());
        assert_eq!(store.stats().memory_bytes, 0);
    }

    #[test]
    fn restart_bounds_the_disk_index_and_preserves_unrelated_pending_files() {
        let dir = Directory::new();
        let empty = PreparedInputRecord {
            metadata: Arc::from([]),
            payload: None,
        };
        for index in 0..=MAX_DISK_RECORDS {
            let mut key = [0; 32];
            key[..8].copy_from_slice(&(index as u64).to_le_bytes());
            fs::write(dir.0.join(file_name(&key)), header(key, &empty)).unwrap();
        }
        let unrelated = dir.0.join("user.pending");
        fs::write(&unrelated, b"preserve").unwrap();
        let temporary = dir.0.join(format!("{}.pending", file_name(&[255; 32])));
        fs::write(&temporary, b"interrupted").unwrap();
        let store =
            InputStore::start_bounded(0, Some(dir.0.clone()), DiskBudget::new(1 << 30, 0), 4096)
                .unwrap();
        let _ = store.get_blocking([255; 32], false);
        assert_eq!(store.stats().disk_records, MAX_DISK_RECORDS);
        assert_eq!(store.stats().index_bytes, INDEX_HEADROOM_BYTES);
        assert!(unrelated.exists());
        assert!(!temporary.exists());
    }

    #[test]
    fn semantic_invalidation_replaces_records_without_damaging_live_leases() {
        let dir = Directory::new();
        let store = InputStore::start(4096, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
        store.insert([1; 32], record(true));
        wait(&store, |s| s.stored == 1);
        let lease = store.get_blocking([1; 32], true).unwrap();
        store.invalidate([1; 32]);
        store.insert(
            [1; 32],
            Arc::new(PreparedInputRecord {
                metadata: Arc::from(&b"corrected"[..]),
                payload: Some(Arc::new(b"correct tensor".to_vec())),
            }),
        );
        let repaired = store.get_blocking([1; 32], true).unwrap();
        assert_eq!(&*repaired.metadata, b"corrected");
        assert_eq!(&*lease.metadata, b"geometry");
        wait(&store, |s| s.stored == 2);
        drop(store);
        let reopened = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
        assert_eq!(
            &*reopened.get_blocking([1; 32], true).unwrap().metadata,
            b"corrected"
        );
    }

    #[test]
    fn invalidation_cancels_pressured_writes_before_replacement() {
        let dir = Directory::new();
        let budget = DiskBudget::new(4096, 4096);
        let store = InputStore::start(0, Some(dir.0.clone()), budget.clone()).unwrap();
        store.insert([1; 32], record(true));
        wait(&store, |s| s.pending_writes == 1);
        store.invalidate([1; 32]);
        store.insert(
            [1; 32],
            Arc::new(PreparedInputRecord {
                metadata: Arc::from(&b"corrected"[..]),
                payload: Some(Arc::new(b"valid".to_vec())),
            }),
        );
        let _ = store.get_blocking([255; 32], false);
        budget.set_checkpoint(0);
        wait(&store, |s| s.stored == 1 && s.queued == 0);
        assert_eq!(
            &*store.get_blocking([1; 32], true).unwrap().metadata,
            b"corrected"
        );
    }

    #[test]
    fn corrupted_disk_metadata_is_counted_and_can_be_recomputed() {
        let dir = Directory::new();
        let key = [1; 32];
        let store = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
        store.insert(key, record(false));
        wait(&store, |s| s.stored == 1);
        let path = dir.0.join(file_name(&key));
        let mut bytes = fs::read(&path).unwrap();
        bytes[48] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(store.get_blocking(key, false).is_none());
        assert_eq!(store.stats().failures, 1);
        assert!(!path.exists());
        store.insert(key, record(false));
        wait(&store, |s| s.stored == 2);
        assert!(store.get_blocking(key, false).is_some());
    }

    #[test]
    fn serialized_storage_is_reserved_before_running_the_encoder() {
        let store = InputStore::start_bounded(4096, None, DiskBudget::new(0, 0), 512).unwrap();
        let called = std::cell::Cell::new(false);
        assert!(
            !store
                .insert_with([1; 32], 4096, || {
                    called.set(true);
                    Ok((*record(true)).clone())
                })
                .unwrap()
        );
        assert!(!called.get());
        assert!(
            store
                .insert_with([1; 32], 32, || {
                    called.set(true);
                    Ok((*record(true)).clone())
                })
                .unwrap()
        );
        assert!(called.get());
        assert!(store.get_blocking([1; 32], true).is_some());
    }

    #[test]
    fn startup_rejects_a_directory_it_cannot_scan() {
        let dir = Directory::new();
        let file = dir.0.join("file");
        fs::write(&file, b"not a directory").unwrap();
        for path in [file, dir.0.join("missing")] {
            let error = InputStore::start(0, Some(path), DiskBudget::new(4096, 0))
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("scan prepared-input directory"), "{error}");
        }
    }

    #[test]
    fn startup_removes_malformed_headers_before_accepting_replacement_records() {
        let dir = Directory::new();
        let key = [1; 32];
        let path = dir.0.join(file_name(&key));
        fs::write(&path, b"truncated header").unwrap();
        let store = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
        assert!(!path.exists());
        store.insert(key, record(true));
        wait(&store, |stats| stats.stored == 1);
        assert!(store.get_blocking(key, true).is_some());
    }

    #[cfg(unix)]
    struct ModeGuard {
        path: PathBuf,
        original: fs::Permissions,
    }
    #[cfg(unix)]
    impl ModeGuard {
        fn set(path: &Path, mode: u32) -> Self {
            use std::os::unix::fs::PermissionsExt;
            let original = fs::metadata(path).unwrap().permissions();
            fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
            Self {
                path: path.to_owned(),
                original,
            }
        }
    }
    #[cfg(unix)]
    impl Drop for ModeGuard {
        fn drop(&mut self) {
            fs::set_permissions(&self.path, self.original.clone()).unwrap();
        }
    }
    #[cfg(unix)]
    fn modes_restrict_user() -> bool {
        // SAFETY: geteuid has no preconditions.
        if unsafe { libc::geteuid() } == 0 {
            eprintln!("skipping: file modes do not restrict the superuser");
            false
        } else {
            true
        }
    }

    #[test]
    #[cfg(unix)]
    fn startup_preserves_unreadable_records_and_rejects_failed_cleanup() {
        if !modes_restrict_user() {
            return;
        }
        let dir = Directory::new();
        let key = [1; 32];
        let path = dir.0.join(file_name(&key));
        write(&path, key, &record(true)).unwrap();
        {
            let _mode = ModeGuard::set(&dir.0, 0o000);
            let error = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0))
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("scan prepared-input directory"), "{error}");
        }
        {
            let _mode = ModeGuard::set(&path, 0o000);
            let error = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0))
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("read prepared-input record"), "{error}");
            assert!(path.exists());
        }
        for corrupt in [false, true] {
            if corrupt {
                fs::write(&path, b"invalid").unwrap();
            }
            let _mode = ModeGuard::set(&dir.0, 0o555);
            let budget = DiskBudget::new(0, 0);
            let error = InputStore::start(0, Some(dir.0.clone()), budget)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("remove prepared-input record"), "{error}");
            assert!(path.exists());
        }
        fs::remove_file(&path).unwrap();
        let temporary = path.with_extension("input.pending");
        fs::write(&temporary, b"partial").unwrap();
        let _mode = ModeGuard::set(&dir.0, 0o555);
        let error = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0))
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("remove incomplete prepared-input record"),
            "{error}"
        );
        assert!(temporary.exists());
    }

    #[test]
    #[cfg(unix)]
    fn failed_write_cleanup_keeps_physical_bytes_charged_until_restart() {
        if !modes_restrict_user() {
            return;
        }
        let dir = Directory::new();
        let key = [1; 32];
        let bytes = HEADER as u64
            + record(true).metadata.len() as u64
            + record(true).payload.as_ref().unwrap().len() as u64;
        let budget = DiskBudget::new(2 * bytes, 0);
        let store = InputStore::start(0, Some(dir.0.clone()), budget.clone()).unwrap();
        let temporary = dir.0.join(file_name(&key)).with_extension("input.pending");
        fs::write(&temporary, []).unwrap();
        {
            let _mode = ModeGuard::set(&dir.0, 0o555);
            store.insert(key, record(true));
            wait(&store, |stats| {
                stats.failures == 1 && stats.queued == 0 && stats.disk_bytes == bytes
            });
            assert_eq!(fs::metadata(&temporary).unwrap().len(), bytes);
            assert!(!budget.reserve_checkpoint(bytes + 1));
            store.insert(key, record(true));
            let _ = store.get_blocking([255; 32], false);
            assert_eq!(store.stats().disk_bytes, bytes);
            assert!(store.stats().skipped > 0);
            assert!(!budget.reserve_checkpoint(bytes + 1));
        }
        drop(store);
        let reopened =
            InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(2 * bytes, 0)).unwrap();
        assert!(!temporary.exists());
        assert_eq!(reopened.stats().disk_bytes, 0);
        reopened.insert(key, record(true));
        wait(&reopened, |stats| stats.stored == 1);
    }

    #[test]
    fn directory_lock_lives_through_last_input_handle() {
        let dir = Directory::new();
        let guard = crate::testing::directory_lock(&dir.0).unwrap();
        let store = InputStore::start(0, Some(dir.0.clone()), DiskBudget::new(4096, 0)).unwrap();
        store.retain_directory(guard.clone());
        drop(guard);
        assert!(crate::testing::directory_lock(&dir.0).is_err());
        drop(store);
        assert!(crate::testing::directory_lock(&dir.0).is_ok());
    }
}
