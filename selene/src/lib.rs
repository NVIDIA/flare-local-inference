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
//! blocks it: `store` copies each publication into the first host tier, `fetch`
//! queues restores, and `poll` drains completions. Pooled stores announce GPU
//! holdings at publication and move retained records down the eviction cascade.

pub mod levels;
pub mod composite;
pub mod copier;
pub mod disk;
pub mod disk_store;
mod error;
pub mod geometry;
pub use shared::index;
pub mod input;
pub mod memory;
pub mod offload;
pub mod plan;
pub mod pool;
pub use shared::record;
pub mod registrar;
pub mod slab;
pub mod shard;
pub mod shared;
pub mod store;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
mod vocab;

pub use copier::{CLOSE_TIMEOUT, Direction, Job, PageCopier, QUEUE_DEPTH, Reply, SharedCopier};
pub use error::{Error, Result};
pub use geometry::{CacheGeometry, Layout, Plane};
pub use registrar::{HostRegistrar, Registration, SplitMapping};
pub use store::{
    ExportChain, ExportId, ExportNode, ExportRefusal, HostHit, HostStore,
    INITIAL_MEMORY_RATE, INITIAL_PREFILL_RATE, INITIAL_STAGED_DISK_RATE, InboundTicket, JobId,
    PublishedCheckpoint, Retirement, Retiring, StoreEvent, StoreOutcome, StoreReason,
    StoreStats, worth_fetching, worth_transfer,
};
pub use vocab::{CheckpointId, HASH_BLOCK_TOKENS, History, PageId, PrefixKey, StateSlotId};

pub use shared::Tier;
