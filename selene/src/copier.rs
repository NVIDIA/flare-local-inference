//! The seam through which the GPU tier copies pages and rows to and from host slabs.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{JobId, Layout, PageId, Result, StateSlotId};

/// Jobs a store may have outstanding per rank; `submit` fails when full.
pub const QUEUE_DEPTH: usize = 16;
/// Longest `close` waits for outstanding jobs before reporting them.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// GPU ids to host slabs.
    Out,
    /// Host slabs to GPU ids.
    In,
}

/// One window of a record: up to [`crate::slab::WINDOW_PAGES`] pages and
/// optionally the row. `pages` pairs a GPU page id with the host address of
/// the page slab holding that page's record across ranks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub id: JobId,
    pub window: u32,
    pub direction: Direction,
    pub pages: Vec<(PageId, usize)>,
    pub row: Option<(StateSlotId, usize)>,
}

#[derive(Debug)]
pub struct Reply {
    pub id: JobId,
    pub window: u32,
    pub direction: Direction,
    pub result: Result<()>,
    /// Bytes copied across every rank.
    pub bytes: u64,
    /// Wall time from submission to the last rank's completion.
    pub elapsed: Duration,
}

/// Copies the pages and row a job names between GPU ids and host slabs.
///
/// A job may not name a page id or a slab address twice. Memory and disk keep
/// independent copiers, so one tier's queue never delays the other's.
pub trait PageCopier: Send {
    /// The cross-rank record layout the copier moves.
    fn layout(&self) -> &Layout;

    /// Queue the job on every rank without waiting.
    ///
    /// `Err` means the job was rejected and no reply follows. After `Ok`,
    /// exactly one [`Reply`] follows from `poll`, once every participating rank
    /// has drained, including when submission failed on some ranks: the ranks
    /// that did accept the job complete their copies and the reply carries the
    /// error. A failure is reported only after the destination ids are
    /// quiescent.
    fn submit(&mut self, job: Job) -> Result<()>;

    /// Collect replies whose every rank has completed.
    fn poll(&mut self, replies: &mut Vec<Reply>);

    /// Accepted jobs whose replies `poll` has not yet returned.
    fn in_flight(&self) -> usize;

    /// Wait up to [`CLOSE_TIMEOUT`] for outstanding jobs, discarding their
    /// replies, then stop the copier. `Err` means copies may still be running:
    /// the caller must keep every slab and GPU id they reference alive.
    fn close(&mut self) -> Result<()>;
}

/// A copier shared between the stores of one tier.
pub type SharedCopier = Arc<Mutex<dyn PageCopier>>;
