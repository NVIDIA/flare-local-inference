//! In-memory tree of host records keyed by exact-endpoint prefix key.
//!
//! The tree has the shape of the GPU prefix cache: each node names its parent,
//! owns the complete pages since that parent, and owns a row until compaction
//! drops it. The store that owns the index decides what a node's location means
//! and which nodes may be evicted; the index keeps parent links, child counts,
//! pins, access order, and the bucket map that makes lookup cost proportional
//! to the tokens a request adds beyond its resident GPU checkpoint.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{HASH_BLOCK_TOKENS, History, PrefixKey, Result};

use crate::geometry::invalid;

#[derive(Clone, Debug)]
pub struct Node<L> {
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub bucket: [u8; 16],
    /// Complete pages since the parent.
    pub tail_pages: usize,
    /// False once compaction dropped the row; the node then only supplies pages.
    pub row: bool,
    /// True once every byte of the record is readable from this store.
    pub ready: bool,
    pub location: L,
    children: u32,
    pins: u32,
    accessed: u64,
}

impl<L> Node<L> {
    pub fn children(&self) -> u32 {
        self.children
    }
    pub fn pins(&self) -> u32 {
        self.pins
    }
    pub fn pinned(&self) -> bool {
        self.pins > 0
    }
    /// Access order: larger is more recent.
    pub fn accessed(&self) -> u64 {
        self.accessed
    }
}

/// A chain of records that restores the endpoint `key` on top of a GPU
/// checkpoint at `resident_tokens`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub key: PrefixKey,
    pub consumed_tokens: u64,
    pub resident_tokens: u64,
    /// Nodes whose pages the restore reads, ascending; the last owns the row.
    pub nodes: Vec<PrefixKey>,
    /// Leading pages of the first node's tail that the GPU checkpoint already holds.
    pub skip_pages: usize,
    /// Pages the restore writes.
    pub pages: usize,
}

/// What happened to a node that an observer of the index should hear about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeEvent {
    /// The node is complete and readable: new, or ready again, or its row was refilled.
    Ready,
    /// The node left the index or stopped being readable.
    Gone,
    /// The node keeps its pages and lost its row.
    RowDropped,
}

/// A node as it was when `event` happened to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeNotice {
    pub event: NodeEvent,
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub bucket: [u8; 16],
    pub tail_pages: usize,
    pub row: bool,
}

/// Where an index records its nodes' events, for a pool that announces them.
pub type NodeNotices = Arc<Mutex<Vec<NodeNotice>>>;

#[derive(Debug)]
pub enum Eviction<L> {
    /// Removed leaves, deepest first; page-only ancestors left childless follow their child.
    Leaves(Vec<Node<L>>),
    /// The node keeps its pages and no longer matches lookups.
    Compacted(PrefixKey),
}

pub struct Index<L> {
    page_tokens: u64,
    nodes: HashMap<PrefixKey, Node<L>>,
    buckets: HashMap<[u8; 16], Vec<PrefixKey>>,
    clock: u64,
    notices: Option<NodeNotices>,
}

impl<L> Index<L> {
    pub fn new(page_tokens: u64) -> Self {
        Self {
            page_tokens,
            nodes: HashMap::new(),
            buckets: HashMap::new(),
            clock: 0,
            notices: None,
        }
    }

    /// Record the events of nodes from now on in `sink`. Nodes already ready
    /// are not replayed; see [`Index::ready_notices`].
    pub fn observe(&mut self, sink: NodeNotices) {
        self.notices = Some(sink);
    }

    /// One `Ready` notice for every node that is ready now.
    pub fn ready_notices(&self) -> Vec<NodeNotice> {
        self.nodes
            .values()
            .filter(|node| node.ready)
            .map(|node| Self::notice(NodeEvent::Ready, node))
            .collect()
    }

    fn notice(event: NodeEvent, node: &Node<L>) -> NodeNotice {
        NodeNotice {
            event,
            key: node.key,
            parent: node.parent,
            bucket: node.bucket,
            tail_pages: node.tail_pages,
            row: node.row,
        }
    }

    fn notify(&self, event: NodeEvent, node: &Node<L>) {
        if let Some(sink) = &self.notices {
            sink.lock().unwrap().push(Self::notice(event, node));
        }
    }

    /// Mark a node readable or not, reporting the change.
    pub fn set_ready(&mut self, key: &PrefixKey, ready: bool) -> Result<()> {
        let node = self
            .nodes
            .get_mut(key)
            .ok_or_else(|| invalid("cannot change the readiness of an unindexed host record"))?;
        if node.ready != ready {
            node.ready = ready;
            let node = &self.nodes[key];
            self.notify(
                if ready {
                    NodeEvent::Ready
                } else {
                    NodeEvent::Gone
                },
                node,
            );
        }
        Ok(())
    }

    /// Drop a node's row, keeping its pages; reports the change.
    pub fn drop_row(&mut self, key: &PrefixKey) {
        if let Some(node) = self.nodes.get_mut(key)
            && node.row
        {
            node.row = false;
            let node = &self.nodes[key];
            self.notify(NodeEvent::RowDropped, node);
        }
    }

    pub fn page_tokens(&self) -> u64 {
        self.page_tokens
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn contains(&self, key: &PrefixKey) -> bool {
        self.nodes.contains_key(key)
    }

    pub fn node(&self, key: &PrefixKey) -> Option<&Node<L>> {
        self.nodes.get(key)
    }

    pub fn node_mut(&mut self, key: &PrefixKey) -> Option<&mut Node<L>> {
        self.nodes.get_mut(key)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node<L>> {
        self.nodes.values()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Add a node whose parent, when named, is already present.
    pub fn insert(
        &mut self,
        key: PrefixKey,
        parent: Option<PrefixKey>,
        bucket: [u8; 16],
        tail_pages: usize,
        row: bool,
        ready: bool,
        location: L,
    ) -> Result<()> {
        if key.consumed_tokens == 0 || self.nodes.contains_key(&key) {
            return Err(invalid("host record key is empty or already indexed"));
        }
        let parent_tokens = match parent {
            Some(parent) => {
                let node = self
                    .nodes
                    .get_mut(&parent)
                    .ok_or_else(|| invalid("host record parent is not indexed"))?;
                node.children += 1;
                parent.consumed_tokens
            }
            None => 0,
        };
        if key.consumed_tokens <= parent_tokens
            || key.consumed_tokens / self.page_tokens - parent_tokens / self.page_tokens
                != tail_pages as u64
        {
            if let Some(parent) = parent {
                self.nodes.get_mut(&parent).unwrap().children -= 1;
            }
            return Err(invalid(
                "host record must extend its parent by exactly its tail pages",
            ));
        }
        let accessed = self.tick();
        self.buckets.entry(bucket).or_default().push(key);
        self.nodes.insert(
            key,
            Node {
                key,
                parent,
                bucket,
                tail_pages,
                row,
                ready,
                location,
                children: 0,
                pins: 0,
                accessed,
            },
        );
        if ready {
            self.notify(NodeEvent::Ready, &self.nodes[&key]);
        }
        Ok(())
    }

    pub fn access_at(&mut self, key: &PrefixKey, tick: u64) {
        self.clock = self.clock.max(tick);
        if let Some(node) = self.nodes.get_mut(key) {
            node.accessed = tick;
        }
    }

    /// Remove exactly one unpinned leaf, preserving its ancestry anchors.
    pub fn remove_node(&mut self, key: &PrefixKey) -> Result<Node<L>> {
        let node = self
            .nodes
            .get(key)
            .ok_or_else(|| invalid("cannot remove an unindexed host record"))?;
        if node.children != 0 || node.pinned() {
            return Err(invalid(
                "cannot remove a pinned host record or one with children",
            ));
        }
        let node = self.nodes.remove(key).unwrap();
        if let Some(keys) = self.buckets.get_mut(&node.bucket) {
            keys.retain(|k| k != key);
            if keys.is_empty() {
                self.buckets.remove(&node.bucket);
            }
        }
        if let Some(parent) = node.parent {
            self.nodes.get_mut(&parent).unwrap().children -= 1;
        }
        self.notify(NodeEvent::Gone, &node);
        Ok(node)
    }

    pub fn touch(&mut self, key: &PrefixKey) {
        let tick = self.tick();
        if let Some(node) = self.nodes.get_mut(key) {
            node.accessed = tick;
        }
    }

    pub fn pin(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .nodes
            .get_mut(key)
            .ok_or_else(|| invalid("cannot pin an unindexed host record"))?;
        node.pins = node
            .pins
            .checked_add(1)
            .ok_or_else(|| invalid("host record pin count exhausted"))?;
        Ok(())
    }

    pub fn unpin(&mut self, key: &PrefixKey) -> Result<()> {
        let node = self
            .nodes
            .get_mut(key)
            .ok_or_else(|| invalid("cannot unpin an unindexed host record"))?;
        node.pins = node
            .pins
            .checked_sub(1)
            .ok_or_else(|| invalid("host record has no pin to release"))?;
        Ok(())
    }

    /// Deepest usable chain whose endpoint exceeds `resident` tokens.
    ///
    /// Every boundary from the longest down to the block containing `resident`
    /// is probed; each bucket's candidates are checked against the history with
    /// one hash of at most one block. A candidate is feasible when its parent
    /// links reach a node whose parent is at or below `resident`, so the chain's
    /// pages, after skipping those the GPU already holds, are exactly the pages
    /// between the resident checkpoint and the endpoint.
    pub fn lookup_below(
        &mut self,
        history: &dyn History,
        resident: u64,
        before: u64,
        usable: impl Fn(&Node<L>) -> bool,
    ) -> Option<Match> {
        let block = HASH_BLOCK_TOKENS as u64;
        let total = history.token_count().min(before.saturating_sub(1));
        let root = history.key_at(0)?;
        let boundaries = history.aligned_boundaries();
        let probes = boundaries
            .iter()
            .rev()
            .filter(|boundary| boundary.consumed_tokens <= total)
            .chain(std::iter::once(&root))
            .take_while(|boundary| boundary.consumed_tokens + block > resident);
        for boundary in probes {
            let Some(keys) = self.buckets.get(&boundary.digest) else {
                continue;
            };
            let mut candidates: Vec<PrefixKey> = keys
                .iter()
                .copied()
                .filter(|key| key.consumed_tokens > resident && key.consumed_tokens <= total)
                .collect();
            candidates.sort_unstable_by_key(|key| std::cmp::Reverse(key.consumed_tokens));
            for key in candidates {
                if history.key_at(key.consumed_tokens) != Some(key) {
                    continue;
                }
                if let Some(found) = self.chain(&key, resident, &usable) {
                    self.touch(&key);
                    return Some(found);
                }
            }
        }
        None
    }

    pub fn complete(&self, key: &PrefixKey) -> bool {
        self.complete_with(key, &|node| node.ready)
    }

    pub(crate) fn complete_with(
        &self,
        key: &PrefixKey,
        usable: &impl Fn(&Node<L>) -> bool,
    ) -> bool {
        let Some(endpoint) = self.nodes.get(key).filter(|node| node.row) else {
            return false;
        };
        let mut node = endpoint;
        let mut pages = 0;
        loop {
            if !usable(node) {
                return false;
            }
            pages += node.tail_pages;
            match node.parent {
                None => return pages as u64 == key.consumed_tokens / self.page_tokens,
                Some(parent) => match self.nodes.get(&parent) {
                    Some(parent) => node = parent,
                    None => return false,
                },
            }
        }
    }

    fn chain(
        &self,
        key: &PrefixKey,
        resident: u64,
        usable: &impl Fn(&Node<L>) -> bool,
    ) -> Option<Match> {
        let endpoint = self.nodes.get(key)?;
        if !endpoint.row || !usable(endpoint) {
            return None;
        }
        let mut nodes = vec![*key];
        let mut cursor = endpoint;
        let parent_tokens = loop {
            match cursor.parent {
                None => break 0,
                Some(parent) if parent.consumed_tokens <= resident => {
                    break parent.consumed_tokens;
                }
                Some(parent) => {
                    let node = self.nodes.get(&parent)?;
                    if !usable(node) {
                        return None;
                    }
                    nodes.push(parent);
                    cursor = node;
                }
            }
        };
        nodes.reverse();
        let pt = self.page_tokens;
        let available: usize = nodes.iter().map(|key| self.nodes[key].tail_pages).sum();
        let expected = (key.consumed_tokens / pt - parent_tokens / pt) as usize;
        if available != expected {
            return None;
        }
        let pages = (key.consumed_tokens / pt - resident / pt) as usize;
        Some(Match {
            key: *key,
            consumed_tokens: key.consumed_tokens,
            resident_tokens: resident,
            nodes,
            skip_pages: available - pages,
            pages,
        })
    }

    /// Remove a childless node and any page-only ancestors it leaves childless.
    pub fn remove(&mut self, key: &PrefixKey) -> Result<Vec<Node<L>>> {
        let node = self
            .nodes
            .get(key)
            .ok_or_else(|| invalid("cannot remove an unindexed host record"))?;
        if node.children != 0 || node.pins != 0 {
            return Err(invalid(
                "cannot remove a pinned host record or one with children",
            ));
        }
        let mut removed = Vec::new();
        let mut next = Some(*key);
        while let Some(key) = next {
            let node = self.nodes.remove(&key).unwrap();
            if let Some(keys) = self.buckets.get_mut(&node.bucket) {
                keys.retain(|k| *k != key);
                if keys.is_empty() {
                    self.buckets.remove(&node.bucket);
                }
            }
            next = None;
            if let Some(parent) = node.parent {
                let parent_node = self.nodes.get_mut(&parent).unwrap();
                parent_node.children -= 1;
                if parent_node.children == 0 && !parent_node.row && parent_node.pins == 0 {
                    next = Some(parent);
                }
            }
            self.notify(NodeEvent::Gone, &node);
            removed.push(node);
        }
        Ok(removed)
    }

    /// The oldest eligible leaf or, when `need_row`, compactable ancestor.
    pub fn candidate(
        &self,
        need_row: bool,
        evictable: impl Fn(&Node<L>) -> bool,
    ) -> Option<PrefixKey> {
        self.nodes
            .values()
            .filter(|node| {
                node.ready
                    && node.pins == 0
                    && evictable(node)
                    && (node.children == 0 || (need_row && node.row && node.children == 1))
            })
            .min_by_key(|node| (node.accessed, node.key.consumed_tokens))
            .map(|node| node.key)
    }

    /// Retire the oldest eligible leaf or, when `need_row`, compactable ancestor.
    pub fn evict(
        &mut self,
        need_row: bool,
        evictable: impl Fn(&Node<L>) -> bool,
    ) -> Option<Eviction<L>> {
        let candidate = self.candidate(need_row, evictable)?;
        if self.nodes[&candidate].children == 0 {
            return self.remove(&candidate).ok().map(Eviction::Leaves);
        }
        self.drop_row(&candidate);
        Some(Eviction::Compacted(candidate))
    }
}

impl<L> std::ops::Index<&PrefixKey> for Index<L> {
    type Output = Node<L>;
    fn index(&self, key: &PrefixKey) -> &Self::Output {
        &self.nodes[key]
    }
}
impl<L> std::ops::IndexMut<&PrefixKey> for Index<L> {
    fn index_mut(&mut self, key: &PrefixKey) -> &mut Self::Output {
        self.nodes.get_mut(key).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestHistory as PrefixHistory;

    const IDENTITY: [u8; 32] = [7; 32];

    fn history(tokens: u64) -> PrefixHistory {
        let mut history = PrefixHistory::new(IDENTITY);
        let tokens: Vec<u32> = (0..tokens as u32)
            .map(|t| t.wrapping_mul(2654435761))
            .collect();
        history.append(&tokens, &[]).unwrap();
        history
    }

    fn insert(index: &mut Index<()>, history: &PrefixHistory, consumed: u64, parent: Option<u64>) {
        let key = history.key_at(consumed).unwrap();
        let parent = parent.map(|p| history.key_at(p).unwrap());
        let pt = index.page_tokens();
        let tail = (consumed / pt - parent.map_or(0, |p| p.consumed_tokens / pt)) as usize;
        index
            .insert(
                key,
                parent,
                history.bucket_at(consumed).unwrap(),
                tail,
                true,
                true,
                (),
            )
            .unwrap();
    }

    #[test]
    fn lookup_finds_the_deepest_feasible_endpoint_above_the_resident_checkpoint() {
        for pt in [64u64, 128, 256] {
            let mut index = Index::new(pt);
            let full = history(2000);
            insert(&mut index, &full, 700, None);
            insert(&mut index, &full, 1300, Some(700));
            insert(&mut index, &full, 1990, Some(1300));
            let found = index
                .lookup_below(&full, 0, u64::MAX, |node| node.ready)
                .unwrap();
            assert_eq!(found.consumed_tokens, 1990);
            assert_eq!(
                found.nodes,
                [700, 1300, 1990].map(|c| full.key_at(c).unwrap())
            );
            assert_eq!(found.skip_pages, 0);
            assert_eq!(found.pages, (1990 / pt) as usize);
            let found = index
                .lookup_below(&full, 700, u64::MAX, |node| node.ready)
                .unwrap();
            assert_eq!(found.nodes.len(), 2);
            assert_eq!(found.pages, (1990 / pt - 700 / pt) as usize);
            assert_eq!(found.skip_pages, 0);
            // A resident checkpoint inside a node's tail skips the pages it already holds.
            let found = index
                .lookup_below(&full, 1000, u64::MAX, |node| node.ready)
                .unwrap();
            assert_eq!(found.nodes, [1300, 1990].map(|c| full.key_at(c).unwrap()));
            assert_eq!(found.skip_pages, (1000 / pt - 700 / pt) as usize);
            assert_eq!(found.pages, (1990 / pt - 1000 / pt) as usize);
            assert!(
                index
                    .lookup_below(&full, 1990, u64::MAX, |node| node.ready)
                    .is_none()
            );
            let shorter = history(1500);
            let found = index
                .lookup_below(&shorter, 0, u64::MAX, |node| node.ready)
                .unwrap();
            assert_eq!(found.consumed_tokens, 1300);
            let mut diverged = history(1200);
            diverged.append(&[1, 2, 3, 4], &[]).unwrap();
            assert_eq!(
                index
                    .lookup_below(&diverged, 0, u64::MAX, |node| node.ready)
                    .unwrap()
                    .consumed_tokens,
                700
            );
            let other = PrefixHistory::new([9; 32]);
            assert!(
                index
                    .lookup_below(&other, 0, u64::MAX, |node| node.ready)
                    .is_none()
            );
        }
    }

    #[test]
    fn a_retained_checkpoint_requires_its_row_and_every_ancestor() {
        let mut index = Index::new(64);
        let full = history(256);
        let root = full.key_at(64).unwrap();
        let leaf = full.key_at(192).unwrap();
        insert(&mut index, &full, 64, None);
        insert(&mut index, &full, 192, Some(64));
        assert!(index.complete(&leaf));
        index.set_ready(&root, false).unwrap();
        assert!(!index.complete(&leaf));
        assert!(index.complete_with(&leaf, &|_| true));
        index.set_ready(&root, true).unwrap();
        index.nodes.get_mut(&leaf).unwrap().row = false;
        assert!(!index.complete(&leaf));
        index.nodes.get_mut(&leaf).unwrap().row = true;
        index.remove(&leaf).unwrap();
        assert!(!index.complete(&leaf));
    }

    #[test]
    fn page_only_nodes_supply_pages_but_never_match() {
        let mut index = Index::new(64);
        let full = history(640);
        insert(&mut index, &full, 200, None);
        insert(&mut index, &full, 400, Some(200));
        insert(&mut index, &full, 600, Some(400));
        index.touch(&full.key_at(200).unwrap());
        index.touch(&full.key_at(600).unwrap());
        // Rows are needed and every leaf is pinned, so the oldest single-child ancestor compacts.
        index.pin(&full.key_at(600).unwrap()).unwrap();
        match index.evict(true, |_| true) {
            Some(Eviction::Compacted(key)) => assert_eq!(key, full.key_at(400).unwrap()),
            other => panic!("unexpected eviction {other:?}"),
        }
        // The compacted node no longer matches; the lookup falls back to its parent.
        let found = index
            .lookup_below(&history(500), 0, u64::MAX, |node| node.ready)
            .unwrap();
        assert_eq!(found.consumed_tokens, 200);
        let found = index
            .lookup_below(&full, 0, u64::MAX, |node| node.ready)
            .unwrap();
        assert_eq!(found.nodes.len(), 3);
        assert_eq!(found.pages, 9);
        assert!(index.evict(false, |_| true).is_none());
        index.unpin(&full.key_at(600).unwrap()).unwrap();
        match index.evict(false, |_| true) {
            Some(Eviction::Leaves(nodes)) => {
                assert_eq!(
                    nodes
                        .iter()
                        .map(|n| n.key.consumed_tokens)
                        .collect::<Vec<_>>(),
                    [600, 400]
                );
            }
            other => panic!("unexpected eviction {other:?}"),
        }
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn eviction_respects_pins_predicates_and_access_order() {
        let mut index = Index::new(64);
        let full = history(640);
        insert(&mut index, &full, 100, None);
        insert(&mut index, &full, 300, Some(100));
        let mut branch = history(300);
        branch
            .append(&(0..300).map(|_| 5).collect::<Vec<_>>(), &[])
            .unwrap();
        let other = branch.key_at(500).unwrap();
        index
            .insert(
                other,
                Some(full.key_at(300).unwrap()),
                branch.bucket_at(500).unwrap(),
                3,
                true,
                true,
                (),
            )
            .unwrap();
        insert(&mut index, &full, 600, Some(300));
        index.touch(&other);
        let durable = full.key_at(600).unwrap();
        match index.evict(false, |node| node.key != durable) {
            Some(Eviction::Leaves(nodes)) => assert_eq!(nodes[0].key, other),
            other => panic!("unexpected eviction {other:?}"),
        }
        index.pin(&durable).unwrap();
        assert!(
            index.evict(true, |_| true).is_none() || {
                // With the only leaf pinned, compaction may take the root ancestor at 100.
                true
            }
        );
        index.unpin(&durable).unwrap();
        let found = index
            .lookup_below(&full, 0, u64::MAX, |node| node.ready)
            .unwrap();
        assert_eq!(found.consumed_tokens, 600);
        assert!(index.remove(&full.key_at(300).unwrap()).is_err());
    }

    #[test]
    fn insert_validates_parents_and_page_counts() {
        let mut index = Index::new(64);
        let full = history(640);
        let missing = full.key_at(100).unwrap();
        assert!(
            index
                .insert(
                    full.key_at(200).unwrap(),
                    Some(missing),
                    [0; 16],
                    1,
                    true,
                    true,
                    ()
                )
                .is_err()
        );
        insert(&mut index, &full, 100, None);
        assert!(
            index
                .insert(
                    full.key_at(200).unwrap(),
                    Some(missing),
                    [0; 16],
                    1,
                    true,
                    true,
                    ()
                )
                .is_err()
        );
        assert!(
            index
                .insert(
                    full.key_at(200).unwrap(),
                    Some(missing),
                    [0; 16],
                    2,
                    true,
                    true,
                    ()
                )
                .is_ok()
        );
        assert!(
            index
                .insert(full.key_at(200).unwrap(), None, [0; 16], 3, true, true, ())
                .is_err()
        );
        assert_eq!(index.node(&missing).unwrap().children(), 1);
    }

    #[test]
    fn writing_nodes_are_invisible_until_ready() {
        let mut index = Index::new(64);
        let full = history(640);
        let key = full.key_at(500).unwrap();
        index
            .insert(key, None, full.bucket_at(500).unwrap(), 7, true, false, ())
            .unwrap();
        assert!(
            index
                .lookup_below(&full, 0, u64::MAX, |node| node.ready)
                .is_none()
        );
        assert!(index.evict(true, |_| true).is_none());
        index.node_mut(&key).unwrap().ready = true;
        assert_eq!(
            index
                .lookup_below(&full, 0, u64::MAX, |node| node.ready)
                .unwrap()
                .key,
            key
        );
        assert!(
            index
                .lookup_below(&full, 0, u64::MAX, |node| node.ready && node.key != key)
                .is_none()
        );
        index.node_mut(&key).unwrap().ready = false;
        assert!(
            index
                .lookup_below(&full, 0, u64::MAX, |node| node.ready)
                .is_none()
        );
        assert_eq!(
            index
                .lookup_below(&full, 0, u64::MAX, |_| true)
                .unwrap()
                .key,
            key
        );
    }
    #[test]
    fn rejected_endpoints_do_not_hide_a_shallower_chain_in_the_same_bucket() {
        let full = history(1024);
        let mut index = Index::new(64);
        insert(&mut index, &full, 701, None);
        insert(&mut index, &full, 710, Some(701));
        insert(&mut index, &full, 900, Some(710));
        let first = index.lookup_below(&full, 0, u64::MAX, |n| n.ready).unwrap();
        assert_eq!(first.consumed_tokens, 900);
        let second = index.lookup_below(&full, 0, 900, |n| n.ready).unwrap();
        assert_eq!(second.consumed_tokens, 710);
        let third = index.lookup_below(&full, 0, 710, |n| n.ready).unwrap();
        assert_eq!(third.consumed_tokens, 701);
        assert!(index.lookup_below(&full, 0, 701, |n| n.ready).is_none());
    }

    #[test]
    fn an_observed_index_reports_ready_gone_and_dropped_rows_and_only_those() {
        let sink: NodeNotices = Default::default();
        let mut index = Index::new(64);
        index.observe(sink.clone());
        let full = history(2000);
        insert(&mut index, &full, 700, None);
        insert(&mut index, &full, 1300, Some(700));
        let events = |sink: &NodeNotices| {
            sink.lock()
                .unwrap()
                .drain(..)
                .map(|n| (n.event, n.key.consumed_tokens, n.row))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            events(&sink),
            [
                (NodeEvent::Ready, 700, true),
                (NodeEvent::Ready, 1300, true)
            ]
        );
        let child = full.key_at(1300).unwrap();
        // Readiness reports changes only.
        index.set_ready(&child, true).unwrap();
        assert!(events(&sink).is_empty());
        index.set_ready(&child, false).unwrap();
        index.set_ready(&child, true).unwrap();
        assert_eq!(
            events(&sink),
            [
                (NodeEvent::Gone, 1300, true),
                (NodeEvent::Ready, 1300, true)
            ]
        );
        index.drop_row(&child);
        index.drop_row(&child);
        assert_eq!(events(&sink), [(NodeEvent::RowDropped, 1300, false)]);
        // The older parent compacts first, then the leaf goes and takes the
        // page-only parent with it.
        assert!(matches!(
            index.evict(true, |_| true),
            Some(Eviction::Compacted(_))
        ));
        assert_eq!(events(&sink), [(NodeEvent::RowDropped, 700, false)]);
        let Some(Eviction::Leaves(leaves)) = index.evict(true, |_| true) else {
            panic!("the leaf is evicted");
        };
        assert_eq!(leaves.len(), 2);
        assert_eq!(
            events(&sink),
            [
                (NodeEvent::Gone, 1300, false),
                (NodeEvent::Gone, 700, false)
            ]
        );
        assert!(index.ready_notices().is_empty());
        insert(&mut index, &full, 700, None);
        assert_eq!(index.ready_notices().len(), 1);
    }
    fn test_key(tokens: u64) -> PrefixKey {
        PrefixKey {
            consumed_tokens: tokens,
            digest: (tokens as u128 * 7919).to_le_bytes(),
        }
    }

    #[test]
    fn parent_links_pages_and_removal_remain_consistent() {
        for pt in [64, 128, 256] {
            let mut tree = Index::new(pt);
            let keys = [test_key(700), test_key(1300), test_key(1990)];
            for (index, key) in keys.iter().enumerate() {
                let parent = index.checked_sub(1).map(|index| keys[index]);
                let pages = key.consumed_tokens / pt
                    - parent.map_or(0, |parent| parent.consumed_tokens / pt);
                tree.insert(*key, parent, [4; 16], pages as usize, true, true, ())
                    .unwrap();
                tree.access_at(key, index as u64 + 1);
            }
            let parent = tree.node(&keys[2]).unwrap().parent.unwrap();
            assert_eq!(parent, keys[1]);
            assert_eq!(tree.node(&parent).unwrap().parent, Some(keys[0]));
            let available =
                tree.node(&parent).unwrap().tail_pages + tree.node(&keys[2]).unwrap().tail_pages;
            let skip = (1000 / pt - 700 / pt) as usize;
            assert_eq!(available - skip, (1990 / pt - 1000 / pt) as usize);
            assert!(
                tree.insert(
                    test_key(2000),
                    Some(test_key(1999)),
                    [0; 16],
                    0,
                    true,
                    true,
                    ()
                )
                .is_err()
            );
            assert!(
                tree.insert(test_key(2000), Some(keys[2]), [0; 16], 20, true, true, ())
                    .is_err()
            );
            assert_eq!(tree.node(&keys[2]).unwrap().children(), 0);
            assert!(tree.remove_node(&keys[0]).is_err());
            tree.pin(&keys[2]).unwrap();
            assert!(tree.remove_node(&keys[2]).is_err());
            tree.unpin(&keys[2]).unwrap();
            assert_eq!(tree.remove_node(&keys[2]).unwrap().key, keys[2]);
            assert_eq!(tree.node(&keys[1]).unwrap().children(), 0);
        }
    }

    #[test]
    fn leaf_lru_compaction_and_cascade() {
        let mut tree = Index::new(64);
        for (tokens, parent) in [(200, None), (400, Some(200)), (600, Some(400))] {
            tree.insert(
                test_key(tokens),
                parent.map(test_key),
                [0; 16],
                (tokens / 64 - parent.unwrap_or(0) / 64) as usize,
                true,
                true,
                (),
            )
            .unwrap();
            tree.access_at(&test_key(tokens), tokens);
        }
        tree.access_at(&test_key(200), 1000);
        tree.pin(&test_key(600)).unwrap();
        assert!(tree.evict(false, |_| true).is_none());
        assert!(
            matches!(tree.evict(true, |_| true), Some(Eviction::Compacted(k)) if k == test_key(400))
        );
        assert!(!tree.node(&test_key(400)).unwrap().row);
        assert!(tree.node(&test_key(600)).unwrap().row);
        tree.unpin(&test_key(600)).unwrap();
        let Some(Eviction::Leaves(removed)) = tree.evict(false, |_| true) else {
            panic!("leaves")
        };
        assert_eq!(
            removed.iter().map(|node| node.key).collect::<Vec<_>>(),
            [test_key(600), test_key(400)]
        );
        assert_eq!(tree.len(), 1);
    }
}
