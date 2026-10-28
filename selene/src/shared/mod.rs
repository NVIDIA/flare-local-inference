//! Concepts shared by instances and the manager: protocol messages and framing, record
//! layouts and checksums, the index tree, identities, rank slices, tiers and limits.

pub mod frame;
pub mod identity;
pub mod index;
pub mod limits;
pub mod rdma;
pub mod record;
pub mod region;
pub mod slice;
#[allow(clippy::all, dead_code)]
pub mod wire {
    tonic::include_proto!("selene.v2");
}

pub use identity::{Geometry, RankPlanes, Scope, instance_id, process_nonce, scope_id};
pub use limits::{
    Limits, MAX_CHAIN_NODES, MAX_OPEN_PUTS, MAX_WINDOW_BYTES, PROGRESS_TIMEOUT, WINDOW_CREDITS,
    transfer_timeout,
};
pub use slice::{RankSlice, SliceLayout, SliceSums};

/// Pages per slab transfer window.
pub const WINDOW_PAGES: usize = 64;
pub const MAX_WINDOW_PAGES: u32 = WINDOW_PAGES as u32;

/// A holding's tier, ordered from the closest to the farthest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    Gpu = 0,
    Dram = 1,
    Nvme = 2,
}

pub fn tier_to(tier: Tier) -> wire::Tier {
    match tier {
        Tier::Gpu => wire::Tier::Gpu,
        Tier::Dram => wire::Tier::Dram,
        Tier::Nvme => wire::Tier::Nvme,
    }
}

pub fn tier_from(tier: i32) -> Option<Tier> {
    match wire::Tier::try_from(tier).ok()? {
        wire::Tier::Gpu => Some(Tier::Gpu),
        wire::Tier::Dram => Some(Tier::Dram),
        wire::Tier::Nvme => Some(Tier::Nvme),
        wire::Tier::Unspecified => None,
    }
}

pub use crate::disk;
