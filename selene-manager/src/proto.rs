#![allow(clippy::large_enum_variant)]

//! The store's internal vocabulary lives at the top of this module; the pool protocol, the
//! only protocol the manager speaks, is [`v2`].

/// The pool protocol, `selene.v2`.
pub use flare_selene::shared::wire as v2;

pub use v2::{Reason, Region};

#[derive(Clone, Debug)]
pub enum CatalogEvent {
    Added(crate::store::EntryData),
    Changed(crate::store::EntryData),
    Removed(crate::Key),
}
