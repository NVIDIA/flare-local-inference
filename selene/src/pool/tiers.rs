//! What a host tier holds, as the pool hears of it, and the pins and checksum
//! thread an export from that tier needs.

use crate::shared::Tier;
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{
    Layout, PrefixKey,
    index::{NodeEvent, NodeNotice, NodeNotices},
    record::checksum,
};

use super::announce::{Announcer, IndexEntry};

/// Checksums of one record: one per tail page, and the row's.
pub use crate::shared::SliceSums as RecordSums;

/// Keys pinned for exports that have ended, for the store to unpin on its own
/// thread. A pin is released by dropping it, from any thread.
#[derive(Clone, Default)]
pub struct Releases(Arc<Mutex<Vec<PrefixKey>>>);

impl Releases {
    pub fn push(&self, keys: &[PrefixKey]) {
        self.0.lock().unwrap().extend_from_slice(keys);
    }

    pub fn drain(&self) -> Vec<PrefixKey> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// Announces one tier's holdings from the events of its index.
pub struct TierView {
    announcer: Announcer,
    tier: Tier,
    sink: NodeNotices,
    announced: HashSet<PrefixKey>,
    page_record_bytes: u64,
    row_record_bytes: u64,
}

impl TierView {
    pub fn new(announcer: Announcer, tier: Tier, layout: &Layout) -> Self {
        Self {
            announcer,
            tier,
            sink: Default::default(),
            announced: HashSet::new(),
            page_record_bytes: layout.page_record_bytes,
            row_record_bytes: layout.row_record_bytes,
        }
    }

    /// Where the tier's index records its events.
    pub fn sink(&self) -> NodeNotices {
        self.sink.clone()
    }

    /// The events recorded since the last call.
    pub fn drain(&self) -> Vec<NodeNotice> {
        std::mem::take(&mut *self.sink.lock().unwrap())
    }

    pub fn contains(&self, key: &PrefixKey) -> bool {
        self.announced.contains(key)
    }

    /// Announce a node that became ready, or whose row was dropped, with the
    /// parent the manager should chain it under.
    pub fn announce(&mut self, notice: &NodeNotice, parent: Option<PrefixKey>) {
        match notice.event {
            NodeEvent::Ready => {
                self.announced.insert(notice.key);
            }
            NodeEvent::RowDropped if self.announced.contains(&notice.key) => {}
            _ => return,
        }
        self.announcer.saved(IndexEntry {
            key: notice.key,
            parent,
            bucket: notice.bucket,
            tail_pages: notice.tail_pages as u32,
            row: notice.row,
            tier: self.tier,
            bytes: notice.tail_pages as u64 * self.page_record_bytes
                + if notice.row { self.row_record_bytes } else { 0 },
        });
    }

    /// The node left the tier because the next tier holds it, which `to` describes:
    /// one notice, so a lookup never sees neither.
    pub fn spill(&mut self, key: &PrefixKey, to: IndexEntry) {
        if self.announced.remove(key) {
            self.announcer.spilled(to, self.tier);
        }
    }

    /// The node left the tier.
    pub fn retire(&mut self, key: &PrefixKey) {
        if self.announced.remove(key) {
            self.announcer.evicted(*key, self.tier, None);
        }
    }
}

/// A record in a window of memory the hasher may read: pinned in its index
/// until the result is collected.
pub(crate) struct HashJob {
    pub key: PrefixKey,
    pub pages: Vec<usize>,
    pub page_bytes: usize,
    pub row: Option<usize>,
    pub row_bytes: usize,
}

impl super::engine::hostfill::HostFiller<PrefixKey> {
    pub(crate) fn submit_hash(&self, job: HashJob) -> bool {
        self.submit(super::engine::hostfill::FillJob {
            owner: job.key,
            window: 0,
            work: Box::new(move || {
                // The owner pins these slabs until it collects the result or joins the worker.
                let read = |address: usize, len: usize| unsafe {
                    std::slice::from_raw_parts(address as *const u8, len)
                };
                Ok(RecordSums {
                    pages: job
                        .pages
                        .iter()
                        .map(|address| checksum(read(*address, job.page_bytes)))
                        .collect(),
                    row: job
                        .row
                        .map(|address| checksum(read(address, job.row_bytes))),
                })
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pool::announce::Notice, testing::geometry};
    use tokio::sync::Notify;

    fn key(n: u64) -> PrefixKey {
        PrefixKey {
            consumed_tokens: n * 64,
            digest: [n as u8; 16],
        }
    }

    fn notice(event: NodeEvent, n: u64, row: bool) -> NodeNotice {
        NodeNotice {
            event,
            key: key(n),
            parent: None,
            bucket: [0; 16],
            tail_pages: 2,
            row,
        }
    }

    #[test]
    fn a_tier_view_announces_ready_nodes_updates_rows_and_retires_once() {
        let announcer = Announcer::new(Arc::new(Notify::new()));
        announcer.begin_session();
        let layout = Layout::new(&[geometry(64, &[("k", 4096)], 4096)]).unwrap();
        let mut view = TierView::new(announcer.clone(), Tier::Dram, &layout);
        view.announce(&notice(NodeEvent::Ready, 1, true), Some(key(0)));
        view.announce(&notice(NodeEvent::Ready, 2, true), None);
        assert!(view.contains(&key(1)) && view.contains(&key(2)));
        // A dropped row updates the same content entry.
        view.announce(&notice(NodeEvent::RowDropped, 1, false), Some(key(0)));
        assert!(view.contains(&key(1)));
        view.retire(&key(1));
        view.retire(&key(1));
        assert!(!view.contains(&key(1)));
        // A node never announced is not retired.
        view.retire(&key(9));
        let notices = announcer.take_batch(100).unwrap();
        let kinds: Vec<_> = notices
            .iter()
            .map(|n| match n {
                Notice::Saved(e) => ("saved", e.row, e.bytes),
                Notice::Evicted { .. } => ("evicted", false, 0),
                Notice::Spilled { .. } => ("spilled", false, 0),
            })
            .collect();
        assert_eq!(kinds[0], ("saved", true, 2 * 4096 + 4096));
        assert_eq!(kinds[2], ("saved", false, 2 * 4096));
        assert_eq!(kinds[3], ("evicted", false, 0));
        assert_eq!(kinds.len(), 4);
    }

    #[test]
    fn the_hasher_hashes_each_page_and_the_row_off_the_calling_thread() {
        let mut hasher =
            super::super::engine::hostfill::HostFiller::start(Arc::new(Notify::new())).unwrap();
        let data: Vec<u8> = (0..3 * 64).map(|i| i as u8).collect();
        let base = data.as_ptr() as usize;
        assert!(hasher.submit_hash(HashJob {
            key: key(1),
            pages: vec![base, base + 64],
            page_bytes: 64,
            row: Some(base + 128),
            row_bytes: 64,
        }));
        let done = loop {
            let mut done = Vec::new();
            hasher.collect(&mut done);
            if let Some(done) = done.into_iter().next() {
                break done;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(done.owner, key(1));
        let sums = done.result.unwrap();
        assert_eq!(
            sums.pages,
            [checksum(&data[..64]), checksum(&data[64..128])]
        );
        assert_eq!(sums.row, Some(checksum(&data[128..])));
        hasher.join();
    }

    #[test]
    fn releases_collect_keys_from_any_thread_and_drain_once() {
        let releases = Releases::default();
        let clone = releases.clone();
        std::thread::spawn(move || clone.push(&[key(1), key(2)]))
            .join()
            .unwrap();
        assert_eq!(releases.drain(), [key(1), key(2)]);
        assert!(releases.drain().is_empty());
    }
}
