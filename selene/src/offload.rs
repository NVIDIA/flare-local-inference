//! Preparation and assembly of local and pooled checkpoint stores.
pub mod sharded;
use crate::{
    CacheGeometry, HostStore, Layout, Result, composite, disk, disk_store, geometry, input, memory,
    pool, pool::PoolOptions, record, shard, slab,
};
use std::{path::PathBuf, time::Duration};

/// The host supplies CUDA copy engines after the final cache geometry is known.
pub trait CopierFactory {
    fn mapped(
        &self,
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        slabs: &slab::SlabPool,
    ) -> Result<crate::SharedCopier>;
    fn unmapped(
        &self,
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
    ) -> Result<Box<dyn crate::PageCopier>>;
}

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
    /// Zero disables shutdown persistence; MAX drains without a time limit.
    pub drain_on_shutdown: Duration,
    /// Pool manager through which this instance shares prefix checkpoints.
    pub pool: Option<PoolOptions>,
}

impl PrefixCacheOffload {
    pub fn enabled(&self) -> bool {
        self.memory_bytes.unwrap_or(0) != 0
            || (self.disk.is_some() && self.disk_limit != Some(0))
            || self.pool.is_some()
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
        } else if self.pool.is_some() {
            // Records a pool shares must be readable by the CPU, so the pool is not split.
            pinned
        } else {
            pinned.div_ceil(ranks.max(1) as u64)
        };
        // Pool copy streams and events are fixed headroom. Hello's shared regions or
        // remote data staging are budgeted separately before sizing GPU caches.
        let tiers = if memory != 0 || disk {
            mapped.div_ceil(512) + STORE_DEVICE_MARGIN_BYTES
        } else {
            0
        };
        let pool = if self.pool.is_some() {
            STORE_DEVICE_MARGIN_BYTES
        } else {
            0
        };
        Ok((tiers + pool, host))
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

/// Parse on, off, or a positive shutdown persistence duration.
pub fn parse_drain_on_shutdown(text: &str) -> std::result::Result<Duration, String> {
    let text = text.trim();
    match text {
        "on" => return Ok(Duration::MAX),
        "off" | "0" => return Ok(Duration::ZERO),
        _ => {}
    }
    let (number, scale) = [("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)]
        .into_iter()
        .find_map(|(suffix, scale)| text.strip_suffix(suffix).map(|n| (n, scale)))
        .unwrap_or((text, 1.0));
    let seconds = number.parse::<f64>().unwrap_or(f64::NAN) * scale;
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|d| !d.is_zero() && d.as_secs() < (1 << 63))
        .ok_or_else(|| "shutdown drain must be on, off, or a positive duration such as 90s".into())
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
    pub shards: Vec<crate::shard::Link>,
    pub pool: Option<pool::PreparedPool>,
    pub drain_on_shutdown: Duration,
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
    pub fn pool_row_warning(&self, max_concurrency: usize) -> Option<String> {
        let rows = self.memory_pool.as_ref()?.row_count();
        let recommended = max_concurrency.saturating_mul(2);
        (rows < recommended).then(|| {
            format!(
                "pooled host memory has {rows} rows, fewer than twice max concurrency ({recommended}); \
             publications may be skipped while older records wait for pool persistence; \
             increase --prefix-cache-memory or reduce --max-concurrency"
            )
        })
    }

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
                "disk {} (limit {}, {} records, staged)",
                directory.display(),
                gibibytes(self.resolved.disk_limit),
                self.scanned.len(),
            ));
        }
        parts.join(" · ")
    }
}

pub fn gibibytes(bytes: u64) -> String {
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
    pub fn lock(&self) -> std::sync::MutexGuard<'_, Option<Prepared>> {
        self.prepared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn take(&self) -> Option<Prepared> {
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
    registrar: &dyn crate::HostRegistrar,
) -> Result<Preparation> {
    prepare_with_inputs(
        options, package, identity, geometry, contexts, 0, None, registrar,
    )
}

pub fn prepare_with_inputs(
    options: &PrefixCacheOffload,
    package: &std::path::Path,
    identity: [u8; 32],
    geometry: Vec<CacheGeometry>,
    contexts: Vec<(usize, i32)>,
    input_staging: u64,
    shard: Option<(usize, usize)>,
    registrar: &dyn crate::HostRegistrar,
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
    if shard.is_some() && memory_bytes != 0 {
        return Err("a multi-host package keeps host prefix records on disk only; set --prefix-cache-memory 0".into());
    }
    let minimum = slab::SlabPool::minimum_bytes(&layout);
    if memory_bytes != 0 && memory_bytes < minimum {
        return Err(format!(
            "host prefix cache memory {memory_bytes} is below the {minimum} bytes this package needs for one outbound and one inbound window"
        )
        .into());
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
    if memory_bytes == 0 && directory.is_none() && options.pool.is_none() {
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
            // A pool shares what the CPU can read, so its memory is never split by rank.
            if directory.is_none() && options.pool.is_none() {
                return Ok(slab::SlabPool::sized_by_rank(
                    &layout,
                    checkpoint_memory,
                    &contexts,
                    registrar,
                )?);
            }
            let mut pool = slab::SlabPool::sized(&layout, checkpoint_memory)?;
            pool.register(registrar, context)?;
            Ok::<_, crate::Error>(pool)
        })
        .transpose()?;
    let staging_pool = directory
        .is_some()
        .then(|| {
            let mut pool = slab::SlabPool::staging(&layout)?;
            pool.register(registrar, context)?;
            Ok::<_, crate::Error>(pool)
        })
        .transpose()?;
    let scanned = match &directory_lock {
        Some(directory) => record::scan(directory, &identity, &fingerprint, geometry[0].page_tokens, &record::rank_sizes(&geometry))?,
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
        drain_on_shutdown: options.drain_on_shutdown,
        disk_budget,
        resolved: Resolved {
            memory_bytes,
            directory,
            disk_limit,
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
        pool: None,
    })))
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
    flare_eos::free_space(path).map_err(|e| e.to_string().into())
}

/// Open the configured store over the prepared resources, starting its
/// transfer threads on every rank's context.
pub fn open(prepared: Prepared, factory: &dyn CopierFactory) -> Result<Box<dyn HostStore>> {
    let Prepared {
        inputs: _,
        drain_on_shutdown,
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
        pool,
    } = prepared;
    if let Some(pool) = &pool {
        pool.start_copies(&contexts, geometry.clone(), factory)?;
    }
    let transfers: Option<crate::SharedCopier> = memory_pool
        .as_ref()
        .or(staging_pool.as_ref())
        .map(|slabs| factory.mapped(&contexts, geometry.clone(), slabs))
        .transpose()?;
    let tier_transfers = || {
        transfers
            .clone()
            .ok_or_else(|| crate::Error::new("prefix transfers require a slab pool"))
    };
    let disk = directory_lock
        .map(|directory| {
            let disk_transfers: crate::SharedCopier = if memory_pool.is_some() {
                factory.mapped(
                    &contexts,
                    geometry.clone(),
                    staging_pool
                        .as_ref()
                        .ok_or("disk transfers require a staging pool")?,
                )?
            } else {
                tier_transfers()?
            };
            let transport: Box<dyn disk_store::DiskTransport> = if !shards.is_empty() {
                if memory_pool.is_some() {
                    return Err(
                        "a multi-host package keeps host prefix records on disk only".into(),
                    );
                }
                shard::transport(staging_pool, disk_transfers, shards)?
            } else {
                Box::new(disk_store::Staged::new(
                    staging_pool,
                    disk_transfers,
                    disk::DiskThread::start("flare-offload-disk")?,
                ))
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
                memory::MemoryStore::new_flat(identity, pool, tier_transfers()?)
            } else {
                memory::MemoryStore::new(identity, pool, tier_transfers()?)
            }
        })
        .transpose()?;
    let local: Option<Box<dyn crate::store::TierStore>> = match (memory, disk) {
        (Some(memory), Some(disk)) => {
            let mut store = composite::CompositeStore::new(memory, disk)?;
            store.drain_on_shutdown = drain_on_shutdown;
            Some(Box::new(store))
        }
        (Some(memory), None) => Some(Box::new(memory)),
        (None, Some(disk)) => Some(Box::new(disk)),
        (None, None) => None,
    };
    match pool {
        Some(pool) => pool.attach(local),
        None => {
            let store = local.ok_or("no prefix store tier is configured")?;
            Ok(store)
        }
    }
}

#[cfg(windows)]
fn free_space(path: &std::path::Path) -> Result<u64> {
    flare_eos::free_space(path).map_err(|e| format!("inspect disk capacity: {e}").into())
}

#[cfg(test)]
mod tests;

pub const HOST_SERVICE_HEADROOM_BYTES: u64 = 2 << 30;
pub const HOST_SYSTEM_SHARE: u64 = 20;
#[cfg(any(unix, test))]
pub fn available_memory(meminfo: &str, total: usize) -> Result<usize> {
    let available = meminfo.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next()? != "MemAvailable:" {
            return None;
        }
        let kib = fields.next()?.parse::<usize>().ok()?;
        (fields.next()? == "kB").then_some(kib)?.checked_mul(1024)
    });
    available.map(|n| n.min(total)).ok_or_else(|| {
        crate::Error::new("Linux MemAvailable is required to size integrated GPU memory")
    })
}
/// Keep the same server allowance and system share as integrated GPU planning.
pub fn host_pool_budget(future: u64) -> Result<u64> {
    #[cfg(unix)]
    let (available, total) = {
        let info = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
        let total = info
            .lines()
            .find_map(|line| {
                let mut fields = line.split_whitespace();
                (fields.next()? == "MemTotal:").then_some(())?;
                let kib = fields.next()?.parse::<u64>().ok()?;
                (fields.next()? == "kB").then_some(kib)?.checked_mul(1024)
            })
            .ok_or_else(|| {
                crate::Error::new("Linux MemTotal is required to budget pool host memory")
            })?;
        (available_memory(&info, usize::MAX)? as u64, total)
    };
    #[cfg(windows)]
    let (available, total) = unsafe {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut memory: MEMORYSTATUSEX = std::mem::zeroed();
        memory.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut memory) == 0 {
            return Err(std::io::Error::last_os_error().to_string().into());
        }
        (memory.ullAvailPhys, memory.ullTotalPhys)
    };
    Ok(available
        .saturating_sub(future)
        .saturating_sub(HOST_SERVICE_HEADROOM_BYTES + total / HOST_SYSTEM_SHARE))
}

/// Resources retained from pre-sizing registration through the worker's store attachment.
#[derive(Default)]
pub struct Startup {
    pool: Option<pool::PreparedPool>,
    shared: Option<std::sync::Arc<Shared>>,
}
impl Startup {
    pub fn start_pool(
        &mut self,
        builder: pool::PoolBuilder<'_>,
        registrar: std::sync::Arc<dyn crate::HostRegistrar + Send + Sync>,
        approve: impl FnOnce(bool) -> Result<bool>,
    ) -> Result<(String, Vec<u64>)> {
        let (mut pool, mut charges) = builder.start(registrar)?;
        let summary = if !approve(pool.approved())? {
            pool.close()?;
            charges.fill(0);
            "local tiers only".into()
        } else {
            pool.summary()?
        };
        self.pool = Some(pool);
        Ok((summary, charges))
    }
    pub fn attach(
        &mut self,
        mut preparation: Preparation,
        factory: &dyn CopierFactory,
    ) -> Result<String> {
        if let Preparation::Store(store) = &mut preparation
            && let Some(mut pool) = self.pool.take()
        {
            pool.prepare_peer(store, factory)?;
            store.pool = Some(pool);
        }
        let summary = preparation.summary();
        self.shared = preparation.into_shared();
        Ok(summary)
    }
    pub fn binding(&self, rank: usize) -> Option<Binding> {
        self.shared
            .as_ref()
            .map(|shared| Binding::new(shared.clone(), rank))
    }
    pub fn shared(&self) -> Option<&Shared> {
        self.shared.as_deref()
    }
    pub fn reset(&mut self) {
        self.shared.take();
        self.pool.take();
    }
}
