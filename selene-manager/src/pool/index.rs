//! The announcement index: what each source says it holds, per scope.
//!
//! Entries are hints. They are keyed by `(key, tier)` inside one source, and a
//! scope also keeps a bucket map so a lookup probes by digest instead of
//! scanning. Nothing here knows about sessions: the caller decides which source
//! an entry belongs to and when a source's entries go away.

use crate::Key;
use flare_selene::shared::Tier;
use std::collections::{BTreeSet, HashMap};

pub type InstanceId = [u8; 16];
pub type Bucket = [u8; 16];

/// Whose holdings an entry set describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SourceId {
    Instance(InstanceId),
    /// The manager's own store.
    Manager,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub key: Key,
    pub parent: Option<Key>,
    pub bucket: Bucket,
    pub tail_pages: u32,
    pub row: bool,
    pub tier: Tier,
    pub bytes: u64,
}

/// Where an entry lives, as a lookup finds it.
pub type EntryRef = (SourceId, Key, Tier);

#[derive(Default)]
pub struct ScopeIndex {
    sources: HashMap<SourceId, HashMap<(Key, Tier), Entry>>,
    buckets: HashMap<Bucket, BTreeSet<EntryRef>>,
    entries: usize,
}

impl ScopeIndex {
    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn source_len(&self, source: SourceId) -> usize {
        self.sources.get(&source).map_or(0, HashMap::len)
    }

    pub fn get(&self, source: SourceId, key: Key, tier: Tier) -> Option<&Entry> {
        self.sources.get(&source)?.get(&(key, tier))
    }

    /// Upsert one content key and tier.
    pub fn upsert(&mut self, source: SourceId, entry: Entry) {
        let set = self.sources.entry(source).or_default();
        let slot = (entry.key, entry.tier);
        if let Some(existing) = set.get(&slot) {
            if existing.bucket != entry.bucket {
                if let Some(bucket) = self.buckets.get_mut(&existing.bucket) {
                    bucket.remove(&(source, slot.0, slot.1));
                    if bucket.is_empty() {
                        self.buckets.remove(&existing.bucket);
                    }
                }
            }
        } else {
            self.entries += 1;
        }
        self.buckets
            .entry(entry.bucket)
            .or_default()
            .insert((source, slot.0, slot.1));
        set.insert(slot, entry);
    }

    /// Remove one content key and tier.
    pub fn remove(&mut self, source: SourceId, key: Key, tier: Tier) -> bool {
        let Some(set) = self.sources.get_mut(&source) else {
            return false;
        };
        let Some(removed) = set.remove(&(key, tier)) else {
            return false;
        };
        if set.is_empty() {
            self.sources.remove(&source);
        }
        self.drop_bucket(source, &removed);
        self.entries -= 1;
        true
    }

    /// Removes every entry of one source.
    pub fn clear(&mut self, source: SourceId) -> usize {
        let Some(set) = self.sources.remove(&source) else {
            return 0;
        };
        for entry in set.values() {
            self.drop_bucket(source, entry);
        }
        self.entries -= set.len();
        set.len()
    }

    /// Replaces every entry of one source with `entries`, as one step.
    pub fn replace(&mut self, source: SourceId, entries: impl IntoIterator<Item = Entry>) {
        self.clear(source);
        for entry in entries {
            self.upsert(source, entry);
        }
    }

    /// The entries whose bucket is `bucket`, in a stable order.
    pub fn in_bucket(&self, bucket: &Bucket) -> impl Iterator<Item = EntryRef> + '_ {
        self.buckets.get(bucket).into_iter().flatten().copied()
    }

    fn drop_bucket(&mut self, source: SourceId, entry: &Entry) {
        if let Some(bucket) = self.buckets.get_mut(&entry.bucket) {
            bucket.remove(&(source, entry.key, entry.tier));
            if bucket.is_empty() {
                self.buckets.remove(&entry.bucket);
            }
        }
    }
}

/// One node of a chain a lookup walked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainStep {
    pub key: Key,
    pub tail_pages: u32,
    pub row: bool,
}

/// The pages a chain supplies from a physical start page to its endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainWalk {
    pub nodes: Vec<ChainStep>,
    pub skip_pages: u32,
    pub pages: u64,
}

/// Walks parents from `endpoint` (an entry of `source` at `tier`) back to the
/// node containing page `start_page`. Returns `None` when the chain has a hole,
/// is inconsistent, or exceeds `max_nodes`.
///
/// A node's pages are `[end - tail_pages, end)` where `end` is its consumed
/// tokens floored to pages: the flooring Flare's `Match` uses. The start of a
/// node is the end of its parent; a root starts at page 0.
pub fn walk_chain(
    index: &ScopeIndex,
    source: SourceId,
    endpoint: &Entry,
    page_tokens: u64,
    start_page: u64,
    max_nodes: usize,
) -> Option<ChainWalk> {
    if page_tokens == 0 {
        return None;
    }
    let end_of = |key: Key| key.consumed_tokens / page_tokens;
    let mut nodes = vec![ChainStep {
        key: endpoint.key,
        tail_pages: endpoint.tail_pages,
        row: endpoint.row,
    }];
    let endpoint_end = end_of(endpoint.key);
    let mut current = endpoint.clone();
    loop {
        let end = end_of(current.key);
        let first = end.checked_sub(u64::from(current.tail_pages))?;
        if first <= start_page {
            break;
        }
        let parent_key = current.parent?;
        let parent = index.get(source, parent_key, endpoint.tier)?;
        // The parent must end exactly where this node starts.
        if end_of(parent.key) != first || nodes.len() >= max_nodes {
            return None;
        }
        // A checkpoint that ends in the page its parent ended in holds no page of its
        // own, and a record with no page and no row is not part of a claim.
        if parent.tail_pages > 0 {
            nodes.push(ChainStep {
                key: parent.key,
                tail_pages: parent.tail_pages,
                row: parent.row,
            });
        }
        current = parent.clone();
    }
    nodes.reverse();
    let first_node_start = {
        let node = &nodes[0];
        end_of(node.key).checked_sub(u64::from(node.tail_pages))?
    };
    // The physical start cannot precede the chain's own beginning.
    let start = start_page.max(first_node_start);
    if start > endpoint_end {
        return None;
    }
    Some(ChainWalk {
        nodes,
        skip_pages: u32::try_from(start - first_node_start).ok()?,
        pages: endpoint_end - start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(tokens: u64, tag: u8) -> Key {
        Key {
            consumed_tokens: tokens,
            digest: [tag; 16],
        }
    }

    fn entry(tokens: u64, tag: u8, parent: Option<(u64, u8)>, tail: u32, tier: Tier) -> Entry {
        Entry {
            key: key(tokens, tag),
            parent: parent.map(|(tokens, tag)| key(tokens, tag)),
            bucket: [9; 16],
            tail_pages: tail,
            row: true,
            tier,
            bytes: 10,
        }
    }

    const SOURCE: SourceId = SourceId::Instance([1; 16]);

    #[test]
    fn content_entries_update_in_stream_order_and_eviction_cleans_buckets() {
        let mut index = ScopeIndex::default();
        let entry = entry(128, 1, None, 2, Tier::Dram);
        index.upsert(SOURCE, entry.clone());
        let mut updated = entry.clone();
        updated.bytes = 77;
        index.upsert(SOURCE, updated);
        assert_eq!(index.get(SOURCE, entry.key, Tier::Dram).unwrap().bytes, 77);
        assert!(index.remove(SOURCE, entry.key, Tier::Dram));
        assert!(!index.remove(SOURCE, entry.key, Tier::Dram));
        assert!(index.in_bucket(&[9; 16]).next().is_none());
    }

    #[test]
    fn tiers_are_independent() {
        let mut index = ScopeIndex::default();
        let gpu = entry(128, 1, None, 2, Tier::Gpu);
        let dram = entry(128, 1, None, 2, Tier::Dram);
        index.upsert(SOURCE, gpu.clone());
        index.upsert(SOURCE, dram.clone());
        assert_eq!(index.len(), 2);
        assert!(index.remove(SOURCE, dram.key, Tier::Dram));
        assert!(index.get(SOURCE, gpu.key, Tier::Gpu).is_some());
    }

    #[test]
    fn replace_is_one_step_and_clear_cleans_buckets() {
        let mut index = ScopeIndex::default();
        index.upsert(SOURCE, entry(128, 1, None, 2, Tier::Dram));
        index.upsert(SOURCE, entry(256, 2, Some((128, 1)), 2, Tier::Dram));
        index.replace(SOURCE, [entry(64, 7, None, 1, Tier::Dram)]);
        assert_eq!(index.len(), 1);
        assert_eq!(index.in_bucket(&[9; 16]).count(), 1);
        assert_eq!(index.clear(SOURCE), 1);
        assert_eq!(index.len(), 0);
        assert!(index.sources.is_empty());
    }

    #[test]
    fn walk_covers_the_chain_from_a_physical_start() {
        let mut index = ScopeIndex::default();
        // Root 128 tokens = 2 pages, then 256 = pages [2, 4), then 384 = [4, 6).
        index.upsert(SOURCE, entry(128, 1, None, 2, Tier::Dram));
        index.upsert(SOURCE, entry(256, 2, Some((128, 1)), 2, Tier::Dram));
        let endpoint = entry(384, 3, Some((256, 2)), 2, Tier::Dram);
        index.upsert(SOURCE, endpoint.clone());
        let walk = walk_chain(&index, SOURCE, &endpoint, 64, 0, 100).unwrap();
        assert_eq!(walk.nodes.len(), 3);
        assert_eq!((walk.skip_pages, walk.pages), (0, 6));
        // Starting mid-record: page 3 lies in the second node.
        let walk = walk_chain(&index, SOURCE, &endpoint, 64, 3, 100).unwrap();
        assert_eq!(walk.nodes.len(), 2);
        assert_eq!(walk.nodes[0].key, key(256, 2));
        assert_eq!((walk.skip_pages, walk.pages), (1, 3));
        // Starting at the endpoint's own page: a row-only tail has zero pages.
        let walk = walk_chain(&index, SOURCE, &endpoint, 64, 6, 100).unwrap();
        assert_eq!(walk.nodes.len(), 1);
        assert_eq!((walk.skip_pages, walk.pages), (2, 0));
    }

    #[test]
    fn a_checkpoint_that_holds_no_page_is_not_a_record_of_the_walk() {
        let mut index = ScopeIndex::default();
        // Root = pages [0, 2); a checkpoint at 150 tokens ends in the same page and
        // holds none; the endpoint at 256 tokens holds [2, 4).
        index.upsert(SOURCE, entry(128, 1, None, 2, Tier::Dram));
        index.upsert(SOURCE, entry(150, 2, Some((128, 1)), 0, Tier::Dram));
        let endpoint = entry(256, 3, Some((150, 2)), 2, Tier::Dram);
        index.upsert(SOURCE, endpoint.clone());
        let walk = walk_chain(&index, SOURCE, &endpoint, 64, 0, 100).unwrap();
        let keys: Vec<_> = walk.nodes.iter().map(|node| node.key).collect();
        assert_eq!(keys, [key(128, 1), key(256, 3)]);
        assert_eq!((walk.skip_pages, walk.pages), (0, 4));
        // Starting where the empty checkpoint ends needs only the endpoint.
        let walk = walk_chain(&index, SOURCE, &endpoint, 64, 2, 100).unwrap();
        assert_eq!(walk.nodes.len(), 1);
        assert_eq!((walk.skip_pages, walk.pages), (0, 2));
        // The endpoint stays even when it holds no page: its row is the claim.
        let tail = entry(160, 4, Some((150, 2)), 0, Tier::Dram);
        index.upsert(SOURCE, tail.clone());
        let walk = walk_chain(&index, SOURCE, &tail, 64, 2, 100).unwrap();
        assert_eq!(walk.nodes.len(), 1);
        assert_eq!(walk.pages, 0);
    }

    #[test]
    fn a_hole_or_a_long_chain_gives_no_walk() {
        let mut index = ScopeIndex::default();
        index.upsert(SOURCE, entry(128, 1, None, 2, Tier::Dram));
        // The middle node is missing.
        let endpoint = entry(384, 3, Some((256, 2)), 2, Tier::Dram);
        index.upsert(SOURCE, endpoint.clone());
        assert!(walk_chain(&index, SOURCE, &endpoint, 64, 0, 100).is_none());
        // A walk that stops inside the endpoint's own node needs no ancestor.
        assert!(walk_chain(&index, SOURCE, &endpoint, 64, 5, 100).is_some());
        index.upsert(SOURCE, entry(256, 2, Some((128, 1)), 2, Tier::Dram));
        assert!(walk_chain(&index, SOURCE, &endpoint, 64, 0, 2).is_none());
        assert!(walk_chain(&index, SOURCE, &endpoint, 64, 0, 3).is_some());
    }

    #[test]
    fn inconsistent_page_arithmetic_is_not_a_chain() {
        let mut index = ScopeIndex::default();
        index.upsert(SOURCE, entry(128, 1, None, 2, Tier::Dram));
        // Claims 3 tail pages but the parent ends at page 2 and it ends at page 6.
        let endpoint = entry(384, 3, Some((128, 1)), 3, Tier::Dram);
        index.upsert(SOURCE, endpoint.clone());
        assert!(walk_chain(&index, SOURCE, &endpoint, 64, 0, 100).is_none());
    }
}
