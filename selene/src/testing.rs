//! Fixtures for tests of this crate and of the crates that implement its seams.

use std::{fs, path::PathBuf};

use blake2::{Blake2b, Digest, digest::consts::U16};

use crate::{
    CacheGeometry, CheckpointId, HASH_BLOCK_TOKENS, History, Job, Layout, PageCopier, PageId,
    Plane, PrefixKey, PublishedCheckpoint, Reply, Result, StateSlotId,
};

pub use crate::slab::rank_part;

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
