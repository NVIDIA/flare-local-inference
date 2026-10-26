//! Host tiers for prefix checkpoints: DRAM slabs and local record files.
//!
//! Selene holds the host-tier index, the eviction policy, the record format and
//! its recovery. It has no CUDA dependency: the GPU tier's owner moves bytes
//! between devices and host slabs through the [`PageCopier`] and
//! [`HostRegistrar`] seams it implements.
//!
//! Three stores implement [`HostStore`]: [`memory::MemoryStore`] retains records
//! in registered host memory, [`disk_store::DiskStore`] retains record files, and
//! [`composite::CompositeStore`] composes the two with write-through and
//! read-through. Every store is driven from the model worker thread and never
//! blocks it: `store` and `fetch` enqueue work, `poll` drains completions.

pub mod composite;
pub mod copier;
pub mod disk;
pub mod disk_store;
mod error;
pub mod geometry;
pub mod index;
pub mod input;
pub mod memory;
pub mod record;
pub mod registrar;
pub mod slab;
pub mod store;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
mod vocab;

pub use copier::{CLOSE_TIMEOUT, Direction, Job, PageCopier, QUEUE_DEPTH, Reply, SharedCopier};
pub use error::{Error, Result};
pub use geometry::{CacheGeometry, Layout, Plane};
pub use registrar::{HostRegistrar, Registration, SplitMapping};
pub use store::{
    HostHit, HostStore, INITIAL_DIRECT_DISK_RATE, INITIAL_MEMORY_RATE, INITIAL_PREFILL_RATE,
    INITIAL_STAGED_DISK_RATE, InboundTicket, JobId, PublishedCheckpoint, StoreEvent, StoreOutcome,
    StoreReason, StoreStats, Tier, Transport, worth_fetching,
};
pub use vocab::{CheckpointId, HASH_BLOCK_TOKENS, History, PageId, PrefixKey, StateSlotId};
