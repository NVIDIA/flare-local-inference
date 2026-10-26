//! Host tiers for prefix checkpoints: DRAM slabs and local record files.
//!
//! Selene holds the host-tier index, the eviction policy, the record format and
//! its recovery. It has no CUDA dependency: the GPU tier's owner moves bytes
//! between devices and host slabs through the [`PageCopier`] and
//! [`HostRegistrar`] seams it implements.

mod error;
mod vocab;

pub use error::{Error, Result};
pub use vocab::{CheckpointId, HASH_BLOCK_TOKENS, History, PageId, PrefixKey, StateSlotId};
