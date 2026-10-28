//! The pool's vocabulary in selene's own types. The wire messages stay inside
//! the codec; nothing here depends on them.

use crate::{plan, shared::Tier};

/// An instance, or the manager's own store when `instance` is empty.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceRef {
    pub instance: Vec<u8>,
    pub session_incarnation: u64,
}

/// A candidate the manager offered, in the planner's terms plus what a claim
/// on it needs. The planner's `source` names `(source, tier)` through the
/// client's source table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolCandidate {
    pub plan: plan::Candidate,
    pub source: SourceRef,
    pub tier: Tier,
}
