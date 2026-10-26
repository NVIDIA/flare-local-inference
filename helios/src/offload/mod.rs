//! Host retention of published prefix checkpoints in registered memory and on disk.
//!
//! A GPU prefix checkpoint owns its tail pages, the complete cache pages since its
//! parent, and one continuation-state row. The host store copies those bytes out
//! when the scheduler publishes a checkpoint and copies them back into free GPU
//! ids when a later request matches a checkpoint the GPU no longer holds. The
//! restored bytes are published as an ordinary GPU checkpoint, so the model
//! worker, the scheduler, and the recipes see no difference between a checkpoint
//! produced by prefill and one restored from the host.
//!
//! Three stores implement [`PrefixStore`]: [`memory::MemoryStore`] retains records
//! in registered host memory, [`disk_store::DiskStore`] retains record files, and
//! [`composite::CompositeStore`] composes the two with write-through and
//! read-through. Every store is driven from the model worker thread and never
//! blocks it: `store` and `fetch` enqueue work, `poll` drains completions.

mod adapter;
#[cfg(all(test, feature = "runtime-package"))]
mod gpu_tests;
mod registrar;
pub mod shard;
pub mod transfer;

pub use flare_selene::{
    CacheGeometry, HostHit, INITIAL_PREFILL_RATE, InboundTicket, Layout, Plane, PublishedCheckpoint,
    StoreOutcome, StoreReason, StoreStats, Tier, Transport, disk, disk_store, geometry, index,
    input, record, slab, worth_fetching,
};
use flare_selene::{HostStore, composite, memory};

use std::path::PathBuf;

use crate::{
    Result,
    cache::{CheckpointId, PrefixHistory, PrefixKey},
};

/// Offload settings as the command line and configuration file supply them.
/// `None` selects the default: no memory store and no disk store until the
/// isolation measurements in `docs/prefix-cache-offload.md` enable derived
/// sizes, one tenth of the directory's free space as the disk limit, and the
/// staged transport.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheOffload {
    /// Host retention budget shared by registered checkpoints and ordinary CPU inputs.
    /// Zero selects no memory retention.
    pub memory_bytes: Option<u64>,
    /// Directory under which the model's record directory is created.
    pub disk: Option<PathBuf>,
    /// Bytes of record files kept on disk; zero selects no disk store.
    pub disk_limit: Option<u64>,
    pub transport: Option<Transport>,
}

impl PrefixCacheOffload {
    pub fn enabled(&self) -> bool {
        self.memory_bytes.unwrap_or(0) != 0 || (self.disk.is_some() && self.disk_limit != Some(0))
    }

    /// Future device and host bytes, sized from the actual record layout. A registered host
    /// pool costs each device about 2 MiB per GiB, plus transfer streams and events. Without
    /// a disk tier the retaining pool is split by rank; disk staging is shared by all ranks.
    pub fn headroom_bytes(&self, ranks: usize, layout: &Layout) -> Result<(u64, u64)> {
        self.headroom_with_inputs(ranks, layout, false)
    }

    pub fn headroom_with_inputs(
        &self,
        ranks: usize,
        layout: &Layout,
        inputs: bool,
    ) -> Result<(u64, u64)> {
        if !self.enabled() {
            return Ok((0, 0));
        }
        let memory = self.memory_bytes.unwrap_or(0);
        let disk = self.disk.is_some() && self.disk_limit != Some(0);
        let staging = if disk {
            layout
                .page_record_bytes
                .checked_mul(slab::STAGING_PAGE_SLABS as u64)
                .and_then(|n| {
                    layout
                        .row_record_bytes
                        .checked_mul(slab::STAGING_ROW_SLABS as u64)
                        .and_then(|rows| n.checked_add(rows))
                })
                .ok_or("prefix cache staging memory overflows the device budget")?
        } else {
            0
        };
        let host = memory
            .checked_add(staging)
            .ok_or("prefix cache host memory overflows the device budget")?;
        let ordinary = input_memory_bytes(memory, slab::SlabPool::minimum_bytes(layout), inputs);
        let pinned = memory - ordinary;
        let mapped = if disk {
            pinned + staging
        } else {
            pinned.div_ceil(ranks.max(1) as u64)
        };
        Ok((mapped.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES, host))
    }
}

/// Transfer streams, events, and their driver allocations on each GPU; eight streams
/// take 4 MiB.
const STORE_DEVICE_MARGIN_BYTES: u64 = 16 << 20;

/// Parse a byte count with an optional binary K, M, G, or T suffix, with or
/// without a trailing `B` or `iB`, as the Ignition configuration does.
pub fn parse_size(text: &str) -> std::result::Result<u64, String> {
    let text = text.trim();
    let digits = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, suffix) = text.split_at(digits);
    let suffix = suffix.trim().to_ascii_uppercase();
    let shift = match suffix.as_str() {
        "" => 0,
        "K" | "KB" | "KIB" => 10,
        "M" | "MB" | "MIB" => 20,
        "G" | "GB" | "GIB" => 30,
        "T" | "TB" | "TIB" => 40,
        _ => return Err(invalid_size(text)),
    };
    if number.is_empty() || number.chars().filter(|c| *c == '.').count() > 1 {
        return Err(invalid_size(text));
    }
    let value: f64 = number.parse().map_err(|_| invalid_size(text))?;
    let bytes = value * (1u64 << shift) as f64;
    if !bytes.is_finite() || bytes < 0.0 || bytes >= u64::MAX as f64 {
        return Err(invalid_size(text));
    }
    Ok(bytes as u64)
}

fn invalid_size(text: &str) -> String {
    format!("size {text:?} must be a number with an optional K, M, G, or T suffix, such as \"16G\"")
}

/// Settings after defaults are applied and the record directory is resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub memory_bytes: u64,
    /// The model's record directory, or `None` for no disk store.
    pub directory: Option<PathBuf>,
    pub disk_limit: u64,
    pub transport: Transport,
}

/// Host resources the loader prepares once per process and the model worker
/// turns into a store: registered pools, the scanned record directory, and
/// every rank's geometry and CUDA context.
pub struct Prepared {
    pub inputs: Option<input::InputStore>,
    pub disk_budget: std::sync::Arc<input::DiskBudget>,
    pub resolved: Resolved,
    pub identity: [u8; 32],
    pub fingerprint: [u8; 16],
    pub geometry: Vec<CacheGeometry>,
    pub layout: Layout,
    /// Each rank's CUDA context handle and local device ordinal.
    pub contexts: Vec<(usize, i32)>,
    pub memory_pool: Option<slab::SlabPool>,
    pub staging_pool: Option<slab::SlabPool>,
    pub scanned: Vec<record::Scanned>,
    pub directory_lock: Option<record::DirectoryLock>,
    /// Rank zero's record connection to each peer rank, ranks 1 onward, when the ranks run as
    /// separate processes; see [`shard`].
    pub(crate) shards: Vec<shard::Link>,
}

/// The loader's result: either prepared resources or the reason no store exists.
pub enum Preparation {
    Store(Box<Prepared>),
    None(String),
}

impl Preparation {
    pub fn summary(&self) -> String {
        match self {
            Self::Store(prepared) => prepared.summary(),
            Self::None(reason) => reason.clone(),
        }
    }

    pub fn into_shared(self) -> Option<std::sync::Arc<Shared>> {
        match self {
            Self::Store(prepared) => Some(std::sync::Arc::new(Shared {
                summary: prepared.summary(),
                inputs: prepared.inputs.clone(),
                prepared: std::sync::Mutex::new(Some(*prepared)),
            })),
            Self::None(_) => None,
        }
    }
}

impl Prepared {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(pool) = &self.memory_pool {
            parts.push(format!(
                "memory {} ({} pages + {} rows{})",
                gibibytes(pool.bytes()),
                pool.page_count(),
                pool.row_count(),
                if pool.rank_parts().is_some() {
                    ", split by rank"
                } else {
                    ""
                }
            ));
        }
        if let Some(inputs) = &self.inputs {
            parts.push(format!(
                "CPU inputs {} limit, {} staging",
                gibibytes(inputs.stats().memory_limit),
                gibibytes(inputs.stats().staging_limit)
            ));
        }
        if let Some(pool) = &self.staging_pool {
            parts.push(format!("staging {}", gibibytes(pool.bytes())));
        }
        if let Some(directory) = &self.resolved.directory {
            parts.push(format!(
                "disk {} (limit {}, {} records, {})",
                directory.display(),
                gibibytes(self.resolved.disk_limit),
                self.scanned.len(),
                match self.resolved.transport {
                    Transport::Staged => "staged",
                    Transport::Direct => "direct",
                }
            ));
        }
        parts.join(" · ")
    }
}

fn gibibytes(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// Shared ownership of prepared resources across rank manifests; the model
/// worker takes them once when it opens the store.
pub struct Shared {
    inputs: Option<input::InputStore>,
    prepared: std::sync::Mutex<Option<Prepared>>,
    pub summary: String,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Prepared>> {
        self.prepared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn take(&self) -> Option<Prepared> {
        self.lock().take()
    }
}

/// A rank's handle on the shared prepared resources.
#[derive(Clone)]
pub struct Binding {
    shared: std::sync::Arc<Shared>,
    rank: usize,
}

impl Binding {
    pub fn new(shared: std::sync::Arc<Shared>, rank: usize) -> Self {
        Self { shared, rank }
    }

    pub fn inputs(&self) -> Option<input::InputStore> {
        self.shared.inputs.clone()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn summary(&self) -> &str {
        &self.shared.summary
    }

    /// Both bindings name the same prepared resources.
    pub fn shares(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.shared, &other.shared)
    }

    /// The prepared resources, once; later calls return `None`.
    pub fn take(&self) -> Option<Prepared> {
        self.shared.take()
    }

    /// The store's records are mirrored onto peer rank processes; see [`shard`].
    pub fn sharded(&self) -> bool {
        self.shared
            .lock()
            .as_ref()
            .is_some_and(|prepared| !prepared.shards.is_empty())
    }
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Binding")
            .field("rank", &self.rank)
            .field("summary", &self.shared.summary)
            .finish()
    }
}

/// Resolve the settings against the package and host, map and register the
/// pools with rank zero's context current, and scan the record directory.
/// Without a disk tier the CPU never reads a record, so the memory pool is
/// split by rank and each device maps only its part.
///
/// `package` is the package file, beside which the default record directory
/// would be created once derived defaults are enabled.
pub fn prepare(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
) -> Result<Preparation> {
    prepare_with_inputs(options, package, identity, geometry, contexts, 0, None)
}

pub fn prepare_with_inputs(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
    input_staging: u64,
    shard: Option<(usize, usize)>,
) -> Result<Preparation> {
    if !options.enabled() {
        return Ok(Preparation::None("disabled".to_owned()));
    }
    let _ = package;
    let layout = Layout::new(&geometry)?;
    let fingerprint = match (shard, geometry.as_slice()) {
        (None, _) => geometry::fingerprint(&geometry),
        (Some((rank, world)), [local]) => geometry::shard_fingerprint(local, rank, world),
        (Some(_), _) => {
            return Err("a rank process of a multi-host package holds exactly one rank".into());
        }
    };
    let memory_bytes = options.memory_bytes.unwrap_or(0);
    let minimum = slab::SlabPool::minimum_bytes(&layout);
    if memory_bytes != 0 && memory_bytes < minimum {
        return Err(format!(
            "host prefix cache memory {memory_bytes} is below the {minimum} bytes this package needs for one outbound and one inbound window"
        )
        .into());
    }
    let transport = options.transport.unwrap_or(Transport::Staged);
    if transport == Transport::Direct {
        return Err("the direct prefix cache transport is not available".into());
    }
    let mut notes = Vec::new();
    let directory = match options
        .disk
        .as_ref()
        .filter(|_| options.disk_limit != Some(0))
    {
        None => None,
        Some(base) => {
            let directory = base.join(record::directory_name(&identity, &fingerprint));
            match std::fs::create_dir_all(&directory) {
                Ok(()) => Some(directory),
                Err(error) => {
                    notes.push(format!(
                        "no disk store: cannot create {}: {error}",
                        directory.display()
                    ));
                    None
                }
            }
        }
    };
    let disk_limit = match &directory {
        None => 0,
        Some(directory) => match options.disk_limit {
            Some(limit) => limit,
            None => free_space(directory)? / 10,
        },
    };
    let directory = directory.filter(|_| disk_limit != 0);
    if memory_bytes == 0 && directory.is_none() {
        notes.insert(0, "disabled".to_owned());
        return Ok(Preparation::None(notes.join("; ")));
    }
    let directory_lock = directory
        .as_deref()
        .map(record::DirectoryLock::acquire)
        .transpose()?;
    let context = contexts
        .first()
        .ok_or("the host prefix store requires at least one rank")?
        .0;
    let input_memory = input_memory_bytes(memory_bytes, minimum, input_staging != 0);
    let checkpoint_memory = memory_bytes - input_memory;
    let memory_pool = (checkpoint_memory != 0)
        .then(|| {
            if directory.is_none() {
                return Ok(slab::SlabPool::sized_by_rank(
                    &layout,
                    checkpoint_memory,
                    &contexts,
                    &registrar::CudaRegistrar,
                )?);
            }
            let mut pool = slab::SlabPool::sized(&layout, checkpoint_memory)?;
            pool.register(&registrar::CudaRegistrar, context)?;
            Ok::<_, crate::Error>(pool)
        })
        .transpose()?;
    let staging_pool = directory
        .is_some()
        .then(|| {
            let mut pool = slab::SlabPool::staging(&layout)?;
            pool.register(&registrar::CudaRegistrar, context)?;
            Ok::<_, crate::Error>(pool)
        })
        .transpose()?;
    let scanned = match &directory_lock {
        Some(directory) => record::scan(directory, &identity, &fingerprint, &geometry)?,
        None => Vec::new(),
    };
    let disk_budget = input::DiskBudget::new(disk_limit, scanned.iter().map(|r| r.bytes).sum());
    let inputs = (input_staging != 0)
        .then(|| {
            input::InputStore::start_bounded(
                input_memory,
                directory.clone(),
                disk_budget.clone(),
                input_staging,
            )
        })
        .transpose()?;
    if let (Some(inputs), Some(guard)) = (&inputs, &directory_lock) {
        inputs.retain_directory(guard.clone());
    }
    Ok(Preparation::Store(Box::new(Prepared {
        inputs,
        disk_budget,
        resolved: Resolved {
            memory_bytes,
            directory,
            disk_limit,
            transport,
        },
        identity,
        fingerprint,
        geometry,
        layout,
        contexts,
        memory_pool,
        staging_pool,
        scanned,
        directory_lock,
        shards: Vec::new(),
    })))
}

/// One bounded ordinary-memory window for serialized prepared inputs and disk-read leases.
pub fn input_staging_bytes(
    contract: &crate::model::PreparedContract,
    buffer_bytes: impl Fn(&str) -> Option<u64>,
) -> Result<u64> {
    use crate::model::contract::ValueSource;
    let mut maximum = 0;
    for graph in &contract.graphs {
        let mut bytes = 0u64;
        for binding in graph
            .inputs
            .iter()
            .filter(|binding| binding.source == ValueSource::ImageTensor)
        {
            let input =
                buffer_bytes(&binding.buffer).ok_or("prepared image input buffer is absent")?;
            bytes = bytes
                .checked_add(input)
                .ok_or("prepared image input extent overflows")?;
        }
        maximum = maximum.max(bytes);
    }
    if maximum == 0 {
        Ok(0)
    } else {
        maximum
            .checked_add(4096)
            .ok_or_else(|| "prepared input staging extent overflows".into())
    }
}

/// Ordinary CPU retention leaves enough registered memory for checkpoint transfers.
pub fn input_memory_bytes(memory: u64, checkpoint_minimum: u64, enabled: bool) -> u64 {
    if !enabled {
        return 0;
    }
    (memory / 8).min(memory.saturating_sub(checkpoint_minimum)) / 4096 * 4096
}

/// Free bytes on the filesystem holding `path`.
#[cfg(unix)]
fn free_space(path: &std::path::Path) -> Result<u64> {
    crate::os::free_space(path).map_err(|e| e.to_string().into())
}

/// Open the configured store over the prepared resources, starting its
/// transfer threads on every rank's context.
pub fn open(prepared: Prepared) -> Result<Box<dyn PrefixStore>> {
    let Prepared {
        inputs: _,
        disk_budget,
        resolved,
        identity,
        fingerprint,
        geometry,
        layout: _,
        contexts,
        memory_pool,
        staging_pool,
        scanned,
        directory_lock,
        shards,
    } = prepared;
    let transfers: flare_selene::SharedCopier =
        std::sync::Arc::new(std::sync::Mutex::new(transfer::Transfers::start(
            &contexts,
            geometry.clone(),
            memory_pool
                .as_ref()
                .or(staging_pool.as_ref())
                .ok_or("prefix transfers require a slab pool")?,
        )?));
    let disk = directory_lock
        .map(|directory| {
            let disk_transfers: flare_selene::SharedCopier = if memory_pool.is_some() {
                std::sync::Arc::new(std::sync::Mutex::new(transfer::Transfers::start(
                    &contexts,
                    geometry.clone(),
                    staging_pool
                        .as_ref()
                        .ok_or("disk transfers require a staging pool")?,
                )?))
            } else {
                transfers.clone()
            };
            let transport: Box<dyn disk_store::DiskTransport> = match resolved.transport {
                Transport::Staged if !shards.is_empty() => {
                    if memory_pool.is_some() {
                        return Err(
                            "a multi-host package keeps host prefix records on disk only".into(),
                        );
                    }
                    shard::transport(staging_pool, disk_transfers, shards)?
                }
                Transport::Staged => Box::new(disk_store::Staged::new(
                    staging_pool,
                    disk_transfers,
                    disk::DiskThread::start("flare-offload-disk")?,
                )),
                Transport::Direct => {
                    return Err("the direct prefix cache transport is not available".into());
                }
            };
            let mut store = disk_store::DiskStore::new(
                directory,
                identity,
                fingerprint,
                resolved.disk_limit,
                geometry,
                scanned,
                transport,
            )?;
            store.share_budget(disk_budget.clone());
            Ok::<_, crate::Error>(store)
        })
        .transpose()?;
    // Inside a composite the disk index owns the tree; memory nodes are flat.
    let memory = memory_pool
        .map(|pool| {
            if disk.is_some() {
                memory::MemoryStore::new_flat(identity, pool, transfers.clone())
            } else {
                memory::MemoryStore::new(identity, pool, transfers.clone())
            }
        })
        .transpose()?;
    let store: Box<dyn HostStore> = match (memory, disk) {
        (Some(memory), Some(disk)) => Box::new(composite::CompositeStore::new(memory, disk)?),
        (Some(memory), None) => Box::new(memory),
        (None, Some(disk)) => Box::new(disk),
        (None, None) => return Err("no prefix store tier is configured".into()),
    };
    Ok(Box::new(adapter::Host::new(store)))
}

/// Completion of work a store accepted earlier.
#[derive(Debug)]
pub enum StoreEvent {
    /// The copy-out of a published checkpoint finished; the worker may unpin it.
    Stored {
        checkpoint: CheckpointId,
        key: PrefixKey,
        result: Result<u64>,
    },
    /// The copy-in for an inbound ticket finished; the worker reports it to the scheduler.
    Fetched { key: PrefixKey, result: Result<u64> },
    /// No copy still reads the GPU checkpoint; retry its publication after progress.
    Deferred { checkpoint: CheckpointId },
}

#[derive(Clone, Copy)]
pub enum LookupOutcome {
    NoMatch,
    CostRejected,
    CapacityRejected,
    EnqueueRejected,
    Admitted,
}

/// A host tier the model worker drives without waiting on it.
///
/// `store` and `fetch` enqueue work and return. `poll` reports completions.
/// The worker pins queued copy-outs until Stored or Deferred, retaining only
/// checkpoint IDs for retries. Ticket IDs belong to the scheduler until Fetched.
///
/// A store whose transfers can no longer be trusted reports the failure from
/// `poll` and stops producing events. The worker then stops, so ids and pins
/// held by transfers it cannot account for are never reused or released.
pub trait PrefixStore {
    /// Deepest record chain extending `resident` and strictly below `before`.
    fn lookup(&mut self, history: &PrefixHistory, resident: u64, before: u64) -> Option<HostHit>;
    /// Only Queued promises an event and retains a GPU copy-out pin.
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome>;
    /// Copy a looked-up chain into the ticket's ids. An error means no event follows.
    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()>;
    /// Report completed work. `Err` means the store can no longer report
    /// completions: transfers may still be running, and the worker must stop
    /// rather than free the ids and pins they hold.
    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()>;
    fn stats(&self) -> StoreStats;
    /// Drain transfers, join threads, and finish pending disk writes.
    /// On failure the caller must retain the store: transfer jobs may still
    /// reference its registered host buffers and GPU destinations.
    fn close(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_directory() -> PathBuf {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "flare-offload-disabled-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[test]
    fn disabled_preparation_requires_no_geometry_context_or_worker_binding() {
        for options in [
            PrefixCacheOffload::default(),
            PrefixCacheOffload {
                memory_bytes: Some(0),
                disk_limit: Some(0),
                ..Default::default()
            },
            PrefixCacheOffload {
                memory_bytes: Some(0),
                transport: Some(Transport::Direct),
                ..Default::default()
            },
        ] {
            let prepared = prepare(
                &options,
                std::path::Path::new("/missing/model.safetensors"),
                [0; 32],
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            assert!(matches!(&prepared, Preparation::None(reason) if reason == "disabled"));
            assert!(prepared.into_shared().is_none());
        }
    }

    #[test]
    fn zero_disk_limit_does_not_create_a_record_directory() {
        let directory = test_directory();
        let prepared = prepare(
            &PrefixCacheOffload {
                disk: Some(directory.clone()),
                disk_limit: Some(0),
                ..Default::default()
            },
            std::path::Path::new("/missing/model.safetensors"),
            [0; 32],
            vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
            Vec::new(),
        )
        .unwrap();
        let created = directory.exists();
        if created {
            std::fs::remove_dir_all(&directory).unwrap();
        }
        assert!(
            !created,
            "disabled disk store created {}",
            directory.display()
        );
        assert!(prepared.into_shared().is_none());
    }

    #[test]
    fn zero_disk_limit_avoids_disk_setup_when_memory_is_enabled() {
        let directory = test_directory();
        let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
        let memory_bytes = slab::SlabPool::minimum_bytes(&Layout::new(&geometry).unwrap());
        let error = prepare(
            &PrefixCacheOffload {
                memory_bytes: Some(memory_bytes),
                disk: Some(directory.clone()),
                disk_limit: Some(0),
                ..Default::default()
            },
            std::path::Path::new("/missing/model.safetensors"),
            [0; 32],
            geometry,
            Vec::new(),
        )
        .err()
        .expect("a memory store requires a CUDA context");
        let created = directory.exists();
        if created {
            std::fs::remove_dir_all(&directory).unwrap();
        }
        assert_eq!(
            error.to_string(),
            "the host prefix store requires at least one rank"
        );
        assert!(
            !created,
            "disabled disk store created {}",
            directory.display()
        );
    }

    #[test]
    fn zero_disk_limit_does_not_lock_or_scan_existing_records() {
        let base = test_directory();
        let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
        let fingerprint = geometry::fingerprint(&geometry);
        let directory = base.join(record::directory_name(&[0; 32], &fingerprint));
        std::fs::create_dir_all(&directory).unwrap();
        let lock = record::DirectoryLock::acquire(&directory).unwrap();
        let active_write = directory.join("active-record.tmp");
        std::fs::write(&active_write, b"active writer payload").unwrap();
        let prepared = prepare(
            &PrefixCacheOffload {
                disk: Some(base.clone()),
                disk_limit: Some(0),
                ..Default::default()
            },
            std::path::Path::new("/missing/model.safetensors"),
            [0; 32],
            geometry,
            Vec::new(),
        )
        .unwrap();
        assert!(prepared.into_shared().is_none());
        assert_eq!(
            std::fs::read(active_write).unwrap(),
            b"active writer payload"
        );
        drop(lock);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn sizes_parse_binary_suffixes_like_the_ignition_configuration() {
        assert_eq!(parse_size("16G").unwrap(), 16 << 30);
        assert_eq!(parse_size("1.5 GiB").unwrap(), 3 << 29);
        assert_eq!(parse_size("512m").unwrap(), 512 << 20);
        assert_eq!(parse_size("2TB").unwrap(), 2 << 40);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("0").unwrap(), 0);
        for text in ["", "G", "16X", "1.2.3G", "-1G", "16 GB extra"] {
            assert!(parse_size(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn memory_headroom_accounts_for_the_actual_host_staging_geometry() {
        let layout = Layout {
            page_tokens: 64,
            ranks: Vec::new(),
            page_record_bytes: 32 << 20,
            row_record_bytes: 8 << 20,
        };
        let mut options = PrefixCacheOffload::default();
        assert_eq!(options.headroom_bytes(2, &layout).unwrap(), (0, 0));
        options.memory_bytes = Some(16 << 30);
        assert_eq!(
            options.headroom_bytes(2, &layout).unwrap(),
            ((8u64 << 30) / 512 + STORE_DEVICE_MARGIN_BYTES, 16 << 30)
        );
        options.disk = Some("records".into());
        let staging = 256 * (32u64 << 20) + 4 * (8u64 << 20);
        let host = (16u64 << 30) + staging;
        assert_eq!(
            options.headroom_bytes(2, &layout).unwrap(),
            (host.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES, host)
        );
        options.disk_limit = Some(0);
        assert_eq!(options.headroom_bytes(2, &layout).unwrap().1, 16 << 30);
    }

    #[test]
    fn ordinary_input_share_preserves_checkpoint_minimum_and_mapping() {
        assert_eq!(input_memory_bytes(0, 4096, true), 0);
        assert_eq!(input_memory_bytes(8192, 4096, true), 0);
        assert_eq!(input_memory_bytes(1 << 30, 128 << 20, true), 128 << 20);
        assert_eq!(input_memory_bytes(1 << 30, 1000 << 20, true), 24 << 20);
        assert_eq!(input_memory_bytes(1 << 30, 128 << 20, false), 0);
        let layout = Layout {
            page_tokens: 64,
            ranks: Vec::new(),
            page_record_bytes: 4096,
            row_record_bytes: 4096,
        };
        let options = PrefixCacheOffload {
            memory_bytes: Some(1 << 30),
            ..Default::default()
        };
        let ordinary = input_memory_bytes(1 << 30, slab::SlabPool::minimum_bytes(&layout), true);
        let (device, host) = options.headroom_with_inputs(2, &layout, true).unwrap();
        assert_eq!(host, 1 << 30);
        assert_eq!(
            device,
            ((1u64 << 30) - ordinary).div_ceil(2).div_ceil(512) + STORE_DEVICE_MARGIN_BYTES
        );
    }
}

#[cfg(windows)]
fn free_space(path: &std::path::Path) -> Result<u64> {
    crate::os::free_space(path).map_err(|e| format!("inspect disk capacity: {e}").into())
}
