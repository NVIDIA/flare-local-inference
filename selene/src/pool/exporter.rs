//! What a host tier pins for an export: a chain of records that fills windows.

use crate::{PrefixKey, Result};

/// One record of a pinned host chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostRecord {
    pub key: PrefixKey,
    /// Record-relative page where this export's pages start.
    pub first_page: u32,
    pub pages: u32,
    /// The record carries the endpoint row this export includes.
    pub row: bool,
}

/// What a host pin holds, in restore order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostChain {
    pub records: Vec<HostRecord>,
    pub total_pages: u64,
    pub row: bool,
}

/// A span of one record to copy into a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostWindow {
    pub record: u32,
    pub first_page: u32,
    pub page_count: u32,
    pub row: bool,
}

/// The checksums persisted with the records read, never hashes of the bytes
/// just read: a corrupted record must reach the requester as a mismatch.
pub use crate::shared::SliceSums as Checksums;

/// A host record chain held against eviction. Dropping it releases the hold.
pub trait HostPin: Send {
    fn chain(&self) -> &HostChain;
    /// Copy the span into `dst` (page records, then the row) and return its
    /// persisted checksums. May block on a device: it runs on a copy thread.
    fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums>;
}

/// Bytes a store or a probe supplies for a put, window by window.
pub trait PutBytes: Send {
    /// Fill `dst` with `page_count` page records starting at page `first_page`
    /// of the chain, then the row record when `row`, and return the checksums to
    /// commit with them: the persisted ones when the source has them, never
    /// hashes of what was just read.
    fn fill(
        &mut self,
        first_page: u64,
        page_count: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums>;
}
