//! The in-process pool client: the instance side of the pool protocol.
//!
//! Nothing in this module runs unless a pool address is configured. Without
//! one no pool type is constructed, so there is no thread, connection, mapping
//! or allocation.

use crate::shared::{frame, identity, limits, slice, transfer_timeout, wire};

mod agent;
mod announce;
pub(crate) mod cascade;
mod budget;
mod builder;
mod client;
mod codec;
mod conn;
pub(crate) mod engine;
mod exporter;
#[cfg(test)]
mod host_tests;
mod local;
mod lookup;
pub(crate) mod net;
pub(crate) mod pins;
mod peers;
pub(crate) mod options;
mod records;
pub mod rank;
pub mod rdma;
mod shared;
mod shm;
mod startup;
mod store;
pub(crate) mod tiers;
mod types;

pub use budget::Budget;
pub use builder::{PoolBuilder, PreparedPool};
pub use options::{PoolOptions, parse_pool_address};
use options::{pool_address, pool_scope};
pub use records::DeferredRecords;
pub use agent::{AgentConfig, RankAgent, RankRecords};
pub use announce::{Announcer, IndexEntry, Notice, Snapshot};
pub use client::{PoolAddress, PoolClient, PoolConfig, STARTUP_WAIT};
pub use exporter::{Checksums, HostChain, HostPin, HostRecord, HostWindow, PutBytes};
pub use lookup::{Answer, LookupState, LookupTicket};
pub use net::Roots;
pub use peers::{PoolOffer, PoolPeers, pool_rank_zero, pool_peer};
pub use pins::{FilePin, MemoryPin};
pub use rank::{RankLink, RankSet};
pub use shared::{
    ExportRequest, FetchFailure, FetchPart, PoolEvent, PoolFetch, PoolStatus, PutOutcome,
    PutRefusal, PutRequest, PutSource,
};
pub use startup::{Approve, PlannedRegion, RegisteredRegion};
pub use store::PoolPort;
pub use store::PooledStore;
pub use tiers::{Releases, TierView};
pub use types::{PoolCandidate, SourceRef};
pub use crate::shared::Tier;
