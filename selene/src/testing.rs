//! Fixtures for tests of this crate and of the crates that implement its seams.

pub use crate::pool::{engine::RESTORE_DEADLINE, net::{Roots, trusted}, options::pool_address};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use blake2::{Blake2b, Digest, digest::consts::U16};

use crate::{
    CacheGeometry, CheckpointId, HASH_BLOCK_TOKENS, History, Job, Layout, PageCopier, PageId,
    Plane, PrefixKey, PublishedCheckpoint, Reply, Result, StateSlotId,
};

pub use crate::slab::rank_part;

/// Retry locks briefly inherited by parallel subprocess tests before exec.
pub fn directory_lock(directory: &Path) -> Result<crate::record::DirectoryLock> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match crate::record::DirectoryLock::acquire(directory) {
            Ok(guard) => return Ok(guard),
            Err(error) => {
                let busy = error.cause().is_some_and(|cause| {
                    cause
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|error| error.kind() == std::io::ErrorKind::WouldBlock)
                });
                if !busy || Instant::now() >= deadline {
                    return Err(error);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// A token history with the key structure the index reads: a key at every
/// hash-block boundary and at any endpoint between them. Digests chain, so two
/// histories share a key exactly while their tokens agree.
#[derive(Clone, Debug)]
pub struct TestHistory {
    identity: [u8; 32],
    boundaries: Vec<PrefixKey>,
    tokens: Vec<u32>,
}

impl TestHistory {
    pub fn new(identity: [u8; 32]) -> Self {
        let mut hash = Blake2b::<U16>::new();
        hash.update(identity);
        Self {
            identity,
            boundaries: vec![PrefixKey {
                consumed_tokens: 0,
                digest: hash.finalize().into(),
            }],
            tokens: Vec::new(),
        }
    }

    pub fn append(&mut self, tokens: &[u32], _events: &[()]) -> Result<()> {
        for token in tokens {
            self.tokens.push(*token);
            if self.tokens.len().is_multiple_of(HASH_BLOCK_TOKENS) {
                let end = self.tokens.len();
                let key = self.digest(end, self.boundaries.last().unwrap().digest);
                self.boundaries.push(key);
            }
        }
        Ok(())
    }

    fn digest(&self, end: usize, parent: [u8; 16]) -> PrefixKey {
        let start = end - end % HASH_BLOCK_TOKENS;
        let start = if end.is_multiple_of(HASH_BLOCK_TOKENS) {
            end - HASH_BLOCK_TOKENS
        } else {
            start
        };
        let mut hash = Blake2b::<U16>::new();
        hash.update(self.identity);
        hash.update(parent);
        hash.update((end as u64).to_le_bytes());
        for token in &self.tokens[start..end] {
            hash.update(token.to_le_bytes());
        }
        PrefixKey {
            consumed_tokens: end as u64,
            digest: hash.finalize().into(),
        }
    }

    /// Digest of the aligned boundary at or below `consumed_tokens`.
    pub fn bucket_at(&self, consumed_tokens: u64) -> Option<[u8; 16]> {
        let aligned = consumed_tokens - consumed_tokens % HASH_BLOCK_TOKENS as u64;
        self.key_at(aligned).map(|key| key.digest)
    }
}

impl History for TestHistory {
    fn token_count(&self) -> u64 {
        self.tokens.len() as u64
    }

    fn key_at(&self, consumed_tokens: u64) -> Option<PrefixKey> {
        let end = usize::try_from(consumed_tokens).ok()?;
        if end > self.tokens.len() {
            return None;
        }
        let boundary = self.boundaries[end / HASH_BLOCK_TOKENS];
        if end.is_multiple_of(HASH_BLOCK_TOKENS) {
            return Some(boundary);
        }
        Some(self.digest(end, boundary.digest))
    }

    fn aligned_boundaries(&self) -> &[PrefixKey] {
        &self.boundaries[1..]
    }
}

/// A cache geometry of `planes` page planes of the given unit sizes and a state row, at
/// fixed addresses.
pub fn geometry(page_tokens: u64, planes: &[(&str, u64)], row_bytes: u64) -> CacheGeometry {
    let mut address = 0x7000_0000_0000u64;
    let planes = planes
        .iter()
        .map(|(name, unit_bytes)| {
            let plane = Plane {
                name: (*name).to_owned(),
                address,
                unit_bytes: *unit_bytes,
            };
            address += unit_bytes * 4096;
            plane
        })
        .collect();
    CacheGeometry {
        snapshot_pages: Vec::new(),
        page_tokens,
        page_capacity: 4096,
        row_capacity: 64,
        row: Plane {
            name: "state".into(),
            address,
            unit_bytes: row_bytes,
        },
        planes,
    }
}

/// A copier with no rank threads: every `submit` fails and `poll` returns
/// nothing, for tests of stores that never submit a transfer or expect every
/// submission to fail.
pub struct StoppedCopier {
    layout: Layout,
}

impl StoppedCopier {
    pub fn new(geometry: &[CacheGeometry]) -> Result<Self> {
        Ok(Self {
            layout: Layout::new(geometry)?,
        })
    }

    pub fn shared(geometry: &[CacheGeometry]) -> Result<crate::SharedCopier> {
        Ok(std::sync::Arc::new(std::sync::Mutex::new(Self::new(
            geometry,
        )?)))
    }
}

impl PageCopier for StoppedCopier {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn submit(&mut self, _: Job) -> Result<()> {
        Err("no offload threads are running".into())
    }

    fn poll(&mut self, _: &mut Vec<Reply>) {}

    fn in_flight(&self) -> usize {
        0
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The device pages and state rows a [`SimulatedGpu`] copies to and from, by id.
#[derive(Clone, Default)]
pub struct GpuMemory {
    pub pages: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u32, Vec<u8>>>>,
    pub rows: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u32, Vec<u8>>>>,
}

/// A copier that moves bytes between host slabs and an in-memory stand-in for
/// the GPU as soon as a job is submitted, and reports the job on the next poll.
pub struct SimulatedGpu {
    layout: Layout,
    memory: GpuMemory,
    replies: Vec<Reply>,
}

impl SimulatedGpu {
    pub fn new(geometry: &[CacheGeometry]) -> Result<(Self, GpuMemory)> {
        let memory = GpuMemory::default();
        Ok((
            Self {
                layout: Layout::new(geometry)?,
                memory: memory.clone(),
                replies: Vec::new(),
            },
            memory,
        ))
    }

    pub fn shared(geometry: &[CacheGeometry]) -> Result<(crate::SharedCopier, GpuMemory)> {
        let (copier, memory) = Self::new(geometry)?;
        Ok((std::sync::Arc::new(std::sync::Mutex::new(copier)), memory))
    }
}

impl PageCopier for SimulatedGpu {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn submit(&mut self, job: Job) -> Result<()> {
        let (page, row) = (
            self.layout.page_record_bytes as usize,
            self.layout.row_record_bytes as usize,
        );
        let (mut pages, mut rows) = (
            self.memory.pages.lock().unwrap(),
            self.memory.rows.lock().unwrap(),
        );
        for (id, address) in &job.pages {
            // The host address names a slab the caller keeps mapped for the job.
            let host = unsafe { std::slice::from_raw_parts_mut(*address as *mut u8, page) };
            match job.direction {
                crate::Direction::Out => {
                    host.copy_from_slice(pages.get(&id.0).map_or(&vec![0; page], |p| p));
                }
                crate::Direction::In => {
                    pages.insert(id.0, host.to_vec());
                }
            }
        }
        if let Some((slot, address)) = job.row {
            let host = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, row) };
            match job.direction {
                crate::Direction::Out => {
                    host.copy_from_slice(rows.get(&slot.0).map_or(&vec![0; row], |p| p));
                }
                crate::Direction::In => {
                    rows.insert(slot.0, host.to_vec());
                }
            }
        }
        self.replies.push(Reply {
            id: job.id,
            window: job.window,
            direction: job.direction,
            result: Ok(()),
            bytes: job.pages.len() as u64 * self.layout.page_record_bytes
                + u64::from(job.row.is_some()) * self.layout.row_record_bytes,
            elapsed: std::time::Duration::from_micros(1),
        });
        Ok(())
    }

    fn poll(&mut self, replies: &mut Vec<Reply>) {
        replies.append(&mut self.replies);
    }

    fn in_flight(&self) -> usize {
        self.replies.len()
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct Ids {
    count: u32,
}
impl Ids {
    pub fn new() -> Self {
        Self { count: 0 }
    }
    pub fn next(&mut self) -> CheckpointId {
        self.count += 1;
        CheckpointId::new(u64::from(self.count))
    }
}

pub fn checkpoint(
    ids: &mut Ids,
    history: &dyn History,
    consumed: u64,
    parent: Option<u64>,
    chain_pages: Vec<PageId>,
    tail_pages: usize,
    slot: StateSlotId,
) -> PublishedCheckpoint {
    PublishedCheckpoint {
        id: ids.next(),
        key: history.key_at(consumed).unwrap(),
        parent: parent.map(|parent| history.key_at(parent).unwrap()),
        bucket: history
            .key_at(consumed - consumed % HASH_BLOCK_TOKENS as u64)
            .unwrap()
            .digest,
        chain_pages,
        tail_pages,
        slot,
    }
}

pub fn test_directory(tier: &str, test: &str) -> Option<PathBuf> {
    let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let directory = base.join(format!(
        "flare-offload-{tier}-{test}-{}",
        std::process::id()
    ));
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

pub fn pattern(bytes: &mut [u8], seed: u64) {
    for (index, byte) in bytes.iter_mut().enumerate() {
        let index = index as u64;
        *byte = (index.wrapping_mul(31) ^ seed.wrapping_mul(7919) ^ (index >> 8)) as u8;
    }
}

/// A host registration whose device address is the supplied host address.
pub struct IdentityRegistrar;
struct IdentityRegistration(usize);
impl crate::Registration for IdentityRegistration {
    fn device_address(&self) -> u64 {
        self.0 as u64
    }
}
impl crate::HostRegistrar for IdentityRegistrar {
    fn register(&self, _: usize, address: usize, _: usize) -> Result<Box<dyn crate::Registration>> {
        Ok(Box::new(IdentityRegistration(address)))
    }
    fn map_split(&self, _: &[(usize, i32, usize)]) -> Result<Box<dyn crate::SplitMapping>> {
        Err("split mapping is not provided by this fixture".into())
    }
}

#[derive(Default)]
pub struct TrackingRegistrar {
    pub calls: std::sync::Mutex<Vec<(usize, usize)>>,
    pub drops: std::sync::Arc<std::sync::Mutex<usize>>,
}
struct Registration {
    address: usize,
    pub drops: std::sync::Arc<std::sync::Mutex<usize>>,
}
impl crate::Registration for Registration {
    fn device_address(&self) -> u64 {
        self.address as u64
    }
}
impl Drop for Registration {
    fn drop(&mut self) {
        *self.drops.lock().unwrap() += 1;
    }
}
impl crate::HostRegistrar for TrackingRegistrar {
    fn register(
        &self,
        _: usize,
        address: usize,
        bytes: usize,
    ) -> crate::Result<Box<dyn crate::Registration>> {
        self.calls.lock().unwrap().push((address, bytes));
        Ok(Box::new(Registration {
            address,
            drops: self.drops.clone(),
        }))
    }
    fn map_split(&self, _: &[(usize, i32, usize)]) -> crate::Result<Box<dyn crate::SplitMapping>> {
        Err("not used".into())
    }
}

/// Whole-record bytes with the checksums supplied by a test.
pub struct RecordBytes {
    pub data: Vec<u8>,
    pub sums: Vec<u64>,
    pub row: u64,
    pub page_bytes: usize,
    pub row_bytes: usize,
}
impl crate::pool::PutBytes for RecordBytes {
    fn fill(
        &mut self,
        first: u64,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<crate::pool::Checksums> {
        let from = first as usize * self.page_bytes;
        let n = pages as usize * self.page_bytes;
        dst[..n].copy_from_slice(&self.data[from..from + n]);
        if row {
            dst[n..].copy_from_slice(&self.data[self.data.len() - self.row_bytes..]);
        }
        Ok(crate::pool::Checksums {
            pages: self.sums[first as usize..][..pages as usize].to_vec(),
            row: row.then_some(self.row),
        })
    }
}
pub struct BytesPin {
    pub chain: crate::pool::HostChain,
    pub bytes: RecordBytes,
}
impl crate::pool::HostPin for BytesPin {
    fn chain(&self) -> &crate::pool::HostChain {
        &self.chain
    }
    fn read(
        &mut self,
        window: &crate::pool::HostWindow,
        dst: &mut [u8],
    ) -> Result<crate::pool::Checksums> {
        crate::pool::PutBytes::fill(
            &mut self.bytes,
            u64::from(window.first_page),
            window.page_count,
            window.row,
            dst,
        )
    }
}
