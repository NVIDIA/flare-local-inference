//! Sequenced catalog mirror fed by the manager's `Subscribe` stream.
//!
//! The model worker owns the live [`Index`] and applies updates in bounded
//! batches; the io thread builds snapshots and decodes events off the serving
//! thread. Replaced indexes move through a single retirement slot so a large
//! snapshot is dropped off the serving thread.

use std::collections::VecDeque;
#[cfg(test)]
use std::sync::Arc;

use super::super::{StoreStats, index::Index, record::HEADER_BYTES};
use crate::{Result, cache::PrefixKey};

use super::proto;

/// Applied updates per `PrefixStore::poll`; a longer queue is drained on
/// subsequent polls rather than delaying the worker.
pub(super) const DELTA_LIMIT: usize = 4096;
/// Queued catalog updates before the mirror is invalidated and rebuilt.
pub(super) const MIRROR_LIMIT: usize = 64 * 1024;

/// Where the manager keeps a record. `REMOTE` entries are served by an
/// upstream manager and count as disk-tier bytes locally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LocationTier {
    Dram,
    Nvme,
    Remote,
}

#[derive(Clone)]
pub(super) struct PoolRecord {
    pub tier: LocationTier,
    pub durable: bool,
    /// Envelope-inclusive reservation, as the manager reports it.
    pub bytes: u64,
    pub unreadable: bool,
    #[cfg(test)]
    #[allow(dead_code)] // Only its Drop observes mirror retirement in tests.
    pub test: TestHooks,
}

#[cfg(test)]
#[derive(Clone, Default)]
#[allow(dead_code)]
pub(super) struct TestHooks {
    pub drop_notice: Option<Arc<DropNotice>>,
}

impl PoolRecord {
    fn new(tier: LocationTier, durable: bool, bytes: u64, unreadable: bool) -> Self {
        Self {
            tier,
            durable,
            bytes,
            unreadable,
            #[cfg(test)]
            test: TestHooks::default(),
        }
    }
}

#[cfg(test)]
pub(super) struct DropNotice(pub std::sync::mpsc::Sender<std::thread::ThreadId>);
#[cfg(test)]
impl Drop for DropNotice {
    fn drop(&mut self) {
        let _ = self.0.send(std::thread::current().id());
    }
}

#[derive(Clone)]
pub(super) struct Entry {
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub bucket: [u8; 16],
    pub tail: usize,
    pub row: bool,
    pub location: PoolRecord,
}

pub(super) fn decode_key(key: &proto::Key) -> Result<PrefixKey> {
    if key.digest.len() != 16 || key.consumed_tokens == 0 {
        return Err("cache manager key has an invalid digest or consumed count".into());
    }
    Ok(PrefixKey {
        consumed_tokens: key.consumed_tokens,
        digest: key.digest.as_slice().try_into().unwrap(),
    })
}

fn optional_key(key: Option<&proto::Key>, has: bool) -> Result<Option<PrefixKey>> {
    match (has, key) {
        (false, None) => Ok(None),
        (false, Some(key)) if key.consumed_tokens == 0 && key.digest.iter().all(|b| *b == 0) => {
            Ok(None)
        }
        (true, Some(key)) => Ok(Some(decode_key(key)?)),
        (true, None) => {
            Err("cache manager catalog entry has a present parent flag but no parent".into())
        }
        (false, Some(_)) => Err("cache manager catalog entry has a parent without the flag".into()),
    }
}

impl Entry {
    pub(super) fn decode(entry: proto::CatalogEntry) -> Result<Self> {
        let key = entry
            .key
            .as_ref()
            .ok_or("cache manager catalog entry has no key")?;
        let bucket: [u8; 16] = entry
            .bucket
            .as_slice()
            .try_into()
            .map_err(|_| "cache manager catalog entry bucket is not 16 bytes")?;
        let tier = proto::Tier::try_from(entry.tier)
            .map_err(|_| "cache manager catalog entry has an unknown tier")?;
        let tier = match tier {
            proto::Tier::Dram => LocationTier::Dram,
            proto::Tier::Nvme => LocationTier::Nvme,
            proto::Tier::Remote => LocationTier::Remote,
            proto::Tier::Unspecified => {
                return Err("cache manager catalog entry has no tier".into());
            }
        };
        let tail = usize::try_from(entry.tail_pages)
            .map_err(|_| "cache manager catalog entry tail page count overflows")?;
        Ok(Self {
            key: decode_key(key)?,
            parent: optional_key(entry.parent.as_ref(), entry.has_parent)?,
            bucket,
            tail,
            row: entry.row,
            location: PoolRecord::new(tier, entry.durable, entry.bytes, entry.unreadable),
        })
    }

    pub(super) fn insert(self, index: &mut Index<PoolRecord>) -> Result<()> {
        index.insert(
            self.key,
            self.parent,
            self.bucket,
            self.tail,
            self.row,
            true,
            self.location,
        )
    }
}

/// One completed catalog transition.
pub(super) enum Update {
    Snapshot(Index<PoolRecord>, StoreStats),
    Added(Entry),
    Changed(Entry),
    Removed(PrefixKey),
}

pub(super) struct Retired {
    pub index: Index<PoolRecord>,
    pub updates: VecDeque<Update>,
}

#[derive(Default)]
pub(super) struct Inbox {
    pub retired: Option<Retired>,
    pub updates: VecDeque<Update>,
    pub invalid: bool,
}

impl Inbox {
    /// Queue an update; false when the bound is exceeded and the mirror must
    /// be rebuilt.
    pub(super) fn push(&mut self, update: Update) -> bool {
        if self.updates.len() >= MIRROR_LIMIT {
            self.updates.clear();
            self.invalid = true;
            false
        } else {
            self.updates.push_back(update);
            true
        }
    }

    pub(super) fn clear(&mut self) {
        self.updates.clear();
        self.invalid = true;
    }
}

pub(super) fn gauge(stats: &mut StoreStats, location: &PoolRecord, add: bool) {
    if location.tier == LocationTier::Dram {
        if add {
            stats.memory_records += 1;
            stats.memory_bytes += location.bytes.saturating_sub(HEADER_BYTES as u64);
        } else {
            stats.memory_records -= 1;
            stats.memory_bytes -= location.bytes.saturating_sub(HEADER_BYTES as u64);
        }
    }
    if location.durable {
        if add {
            stats.disk_records += 1;
            stats.disk_bytes += location.bytes;
        } else {
            stats.disk_records -= 1;
            stats.disk_bytes -= location.bytes;
        }
    }
}
