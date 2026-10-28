//! Immutable terminal prefix metadata with explicit GPU readiness and retirement.

mod hash;

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::Result;
pub use flare_selene::{CheckpointId, PageId};
pub use hash::{ConditioningEvent, HASH_BLOCK_TOKENS, PrefixHistory, PrefixKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SnapshotId(pub u32);

/// Completion on one ordered timeline covering every relevant GPU stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fence(pub u64);

#[derive(Clone, Copy, Debug)]
pub struct StateSnapshot {
    pub id: SnapshotId,
    pub consumed_tokens: u64,
}

#[derive(Clone, Debug)]
pub struct Publication {
    pub parent: Option<CheckpointId>,
    pub consumed_tokens: u64,
    pub tail_pages: Vec<PageId>,
    pub snapshot: StateSnapshot,
    pub ready_after: Fence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishResult {
    Created { id: CheckpointId, ready: bool },
    Existing { id: CheckpointId, ready: bool },
}

impl PublishResult {
    pub fn id(self) -> CheckpointId {
        match self {
            Self::Created { id, .. } | Self::Existing { id, .. } => id,
        }
    }
}

#[derive(Debug)]
pub struct Checkpoint {
    id: CheckpointId,
    parent: Option<CheckpointId>,
    prefix: PrefixKey,
    bucket: [u8; 16],
    tail_pages: Vec<PageId>,
    snapshot: SnapshotId,
    ready_after: Fence,
    ready: bool,
    last_use: Fence,
    child_count: usize,
    direct_pins: u64,
    pins: u64,
    accessed: u64,
}

impl Checkpoint {
    pub fn id(&self) -> CheckpointId {
        self.id
    }
    pub fn parent(&self) -> Option<CheckpointId> {
        self.parent
    }
    pub fn prefix(&self) -> PrefixKey {
        self.prefix
    }
    /// Digest of the aligned hash-block boundary at or below the endpoint, which
    /// keys host-store buckets.
    pub fn bucket(&self) -> [u8; 16] {
        self.bucket
    }
    pub fn tail_pages(&self) -> &[PageId] {
        &self.tail_pages
    }
    pub fn snapshot(&self) -> SnapshotId {
        self.snapshot
    }
    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn pin_count(&self) -> u64 {
        self.pins
    }
    pub fn child_count(&self) -> usize {
        self.child_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup {
    pub checkpoint: Option<CheckpointId>,
    pub consumed_tokens: u64,
    pub pages: Vec<PageId>,
    pub snapshot: Option<SnapshotId>,
}

/// Logically evicted resources, which are not yet permission to reuse storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetiredResources {
    pub checkpoint: CheckpointId,
    pub key: PrefixKey,
    /// The single child that took this checkpoint's pages when it was compacted.
    pub heir: Option<CheckpointId>,
    pub pages: Vec<PageId>,
    pub snapshot: SnapshotId,
    pub retire_after: Fence,
}

/// Storage that eviction could retire, as projected by `reclaimable_if_pinned`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reclaimable {
    pub pages: Vec<PageId>,
    pub snapshots: Vec<SnapshotId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReclaimedResources {
    pub pages: Vec<PageId>,
    pub snapshots: Vec<SnapshotId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompletionUpdate {
    pub published: Vec<CheckpointId>,
    pub reusable: ReclaimedResources,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvictionTarget {
    pub pages: usize,
    pub snapshots: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictionReport {
    pub retired: Vec<RetiredResources>,
    pub pages_scheduled: usize,
    pub snapshots_scheduled: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheStats {
    pub published_checkpoints: usize,
    pub pending_checkpoints: usize,
    pub cached_pages: usize,
    pub retired_pages: usize,
    pub retired_snapshots: usize,
    pub owned_pages: usize,
    pub owned_snapshots: usize,
}

/// CPU cache policy over externally allocated GPU storage.
///
/// A Created result transfers logical ownership of the proposal's private pages
/// and snapshot to this cache. Existing or error retains them with the caller;
/// pending writes to unused duplicate resources still require caller retirement.
/// Snapshot IDs cover the recipe's complete continuation state and saved hidden
/// vector. Page IDs cover every layer's corresponding immutable cache storage.
pub struct PrefixCheckpointCache {
    identity: [u8; 32],
    page_tokens: usize,
    page_capacity: u32,
    snapshot_capacity: u32,
    nodes: BTreeMap<CheckpointId, Checkpoint>,
    prefixes: HashMap<PrefixKey, CheckpointId>,
    owned_pages: HashSet<PageId>,
    owned_snapshots: HashSet<SnapshotId>,
    retirement: Vec<RetiredResources>,
    completed: Fence,
    next_id: u64,
    clock: u64,
}

impl PrefixCheckpointCache {
    pub fn new(
        identity: [u8; 32],
        page_capacity: u32,
        snapshot_capacity: u32,
        page_tokens: usize,
    ) -> Result<Self> {
        if page_tokens == 0 {
            return Err("physical cache pages require a positive token capacity".into());
        }
        Ok(Self {
            identity,
            page_tokens,
            page_capacity,
            snapshot_capacity,
            nodes: BTreeMap::new(),
            prefixes: HashMap::new(),
            owned_pages: HashSet::new(),
            owned_snapshots: HashSet::new(),
            retirement: Vec::new(),
            completed: Fence(0),
            next_id: 1,
            clock: 0,
        })
    }

    pub fn completed(&self) -> Fence {
        self.completed
    }

    pub fn checkpoint(&self, id: CheckpointId) -> Result<&Checkpoint> {
        self.nodes.get(&id).ok_or_else(|| {
            crate::Error::execution(
                crate::ExecutionError::State,
                format!("unknown or retired checkpoint {}", id.value()),
            )
        })
    }

    /// The GPU-ready checkpoint published under `prefix`, if any.
    pub fn ready_by_prefix(&self, prefix: &PrefixKey) -> Option<CheckpointId> {
        let id = *self.prefixes.get(prefix)?;
        self.nodes[&id].ready.then_some(id)
    }

    /// Checkpoints from the root to `id`, each with its parent.
    pub(crate) fn chain(&self, id: CheckpointId) -> Result<Vec<CheckpointId>> {
        let mut chain = Vec::new();
        let mut next = Some(id);
        while let Some(id) = next {
            let checkpoint = self.checkpoint(id)?;
            chain.push(id);
            next = checkpoint.parent;
        }
        Ok(chain)
    }

    fn history_matches(&self, history: &PrefixHistory) -> Result<()> {
        if history.identity() != self.identity {
            return Err("prefix history belongs to a different model/recipe identity".into());
        }
        Ok(())
    }

    pub fn publish(
        &mut self,
        history: &PrefixHistory,
        proposal: &Publication,
    ) -> Result<PublishResult> {
        self.history_matches(history)?;
        let prefix = history.key_at(proposal.consumed_tokens).ok_or(
            "checkpoint endpoint must be an existing nonempty prefix in the supplied history",
        )?;
        let bucket = history.bucket_at(proposal.consumed_tokens).ok_or(
            "checkpoint endpoint has no aligned hash-block boundary in the supplied history",
        )?;
        if proposal.consumed_tokens == 0
            || proposal.snapshot.consumed_tokens != proposal.consumed_tokens
        {
            return Err(
                "snapshot consumed-token length must exactly equal the checkpoint endpoint".into(),
            );
        }
        let parent_chain = match proposal.parent {
            Some(id) => self.chain(id)?,
            None => Vec::new(),
        };
        let parent_len = if let Some(id) = proposal.parent {
            let parent = self.checkpoint(id)?;
            if !parent.ready {
                return Err("checkpoint parent is not GPU-ready".into());
            }
            if history.key_at(parent.prefix.consumed_tokens) != Some(parent.prefix) {
                return Err("checkpoint parent does not match the supplied prefix history".into());
            }
            parent.prefix.consumed_tokens
        } else {
            0
        };
        let tail_tokens = proposal
            .consumed_tokens
            .checked_sub(parent_len)
            .ok_or("checkpoint endpoint precedes its parent")?;
        if tail_tokens == 0
            || proposal.consumed_tokens / self.page_tokens as u64
                - parent_len / self.page_tokens as u64
                != proposal.tail_pages.len() as u64
        {
            return Err(
                "checkpoint must extend its parent and transfer exactly the additional complete pages"
                    .into(),
            );
        }
        let mut candidate_pages = HashSet::new();
        for page in &proposal.tail_pages {
            if page.0 >= self.page_capacity
                || self.owned_pages.contains(page)
                || !candidate_pages.insert(*page)
            {
                return Err(
                    "checkpoint tail pages are out of capacity, duplicated, or already owned"
                        .into(),
                );
            }
        }
        if proposal.snapshot.id.0 >= self.snapshot_capacity
            || self.owned_snapshots.contains(&proposal.snapshot.id)
        {
            return Err("checkpoint snapshot is out of capacity or already owned".into());
        }
        let tick = self
            .clock
            .checked_add(1)
            .ok_or("cache access clock exhausted")?;
        if let Some(id) = self.prefixes.get(&prefix).copied() {
            self.clock = tick;
            let existing = self.nodes.get_mut(&id).unwrap();
            existing.accessed = tick;
            let ready = existing.ready;
            for ancestor in parent_chain {
                let node = self.nodes.get_mut(&ancestor).unwrap();
                node.last_use = node.last_use.max(proposal.ready_after);
            }
            return Ok(PublishResult::Existing { id, ready });
        }
        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or("checkpoint ID counter exhausted")?;
        let id = CheckpointId::new(self.next_id);
        let ready = proposal.ready_after <= self.completed;
        let node = Checkpoint {
            id,
            parent: proposal.parent,
            prefix,
            bucket,
            tail_pages: proposal.tail_pages.clone(),
            snapshot: proposal.snapshot.id,
            ready_after: proposal.ready_after,
            ready,
            last_use: proposal.ready_after,
            child_count: 0,
            direct_pins: 0,
            pins: 0,
            accessed: tick,
        };
        self.next_id = next_id;
        self.clock = tick;
        self.owned_pages.extend(candidate_pages);
        self.owned_snapshots.insert(proposal.snapshot.id);
        if let Some(parent) = proposal.parent {
            self.nodes.get_mut(&parent).unwrap().child_count += 1;
        }
        for ancestor in parent_chain {
            let node = self.nodes.get_mut(&ancestor).unwrap();
            node.last_use = node.last_use.max(proposal.ready_after);
        }
        self.prefixes.insert(prefix, id);
        self.nodes.insert(id, node);
        Ok(PublishResult::Created { id, ready })
    }

    pub fn lookup(&mut self, history: &PrefixHistory) -> Result<Lookup> {
        self.history_matches(history)?;
        let selected = self
            .nodes
            .values()
            .filter(|node| {
                node.ready && history.key_at(node.prefix.consumed_tokens) == Some(node.prefix)
            })
            .max_by_key(|node| node.prefix.consumed_tokens)
            .map(|node| node.id);
        if let Some(id) = selected {
            self.clock = self
                .clock
                .checked_add(1)
                .ok_or("cache access clock exhausted")?;
            self.nodes.get_mut(&id).unwrap().accessed = self.clock;
        }
        match selected {
            None => Ok(Lookup {
                checkpoint: None,
                consumed_tokens: 0,
                pages: Vec::new(),
                snapshot: None,
            }),
            Some(id) => {
                let node = &self.nodes[&id];
                Ok(Lookup {
                    checkpoint: Some(id),
                    consumed_tokens: node.prefix.consumed_tokens,
                    pages: self.page_chain(id)?,
                    snapshot: Some(node.snapshot),
                })
            }
        }
    }

    pub fn page_chain(&self, id: CheckpointId) -> Result<Vec<PageId>> {
        let mut pages = Vec::new();
        for ancestor in self.chain(id)?.into_iter().rev() {
            pages.extend_from_slice(&self.nodes[&ancestor].tail_pages);
        }
        Ok(pages)
    }

    pub fn pin(&mut self, id: CheckpointId) -> Result<()> {
        if !self.checkpoint(id)?.ready {
            return Err("cannot pin a checkpoint before GPU readiness".into());
        }
        let chain = self.chain(id)?;
        if self.nodes[&id].direct_pins == u64::MAX
            || chain.iter().any(|id| self.nodes[id].pins == u64::MAX)
        {
            return Err("checkpoint reference count exhausted".into());
        }
        self.nodes.get_mut(&id).unwrap().direct_pins += 1;
        for ancestor in chain {
            self.nodes.get_mut(&ancestor).unwrap().pins += 1;
        }
        Ok(())
    }

    pub fn unpin(&mut self, id: CheckpointId) -> Result<()> {
        if self.checkpoint(id)?.direct_pins == 0 {
            return Err("checkpoint has no direct pin to release".into());
        }
        let chain = self.chain(id)?;
        self.nodes.get_mut(&id).unwrap().direct_pins -= 1;
        for ancestor in chain {
            self.nodes.get_mut(&ancestor).unwrap().pins -= 1;
        }
        Ok(())
    }

    /// Record GPU reads before submission; pinning alone does not track GPU use.
    pub fn mark_used(&mut self, id: CheckpointId, through: Fence) -> Result<()> {
        if !self.checkpoint(id)?.ready {
            return Err("cannot use a checkpoint before GPU readiness".into());
        }
        for ancestor in self.chain(id)? {
            let node = self.nodes.get_mut(&ancestor).unwrap();
            node.last_use = node.last_use.max(through);
        }
        Ok(())
    }

    fn retire_leaf(&mut self, id: CheckpointId) -> RetiredResources {
        let node = self.nodes.remove(&id).unwrap();
        self.prefixes.remove(&node.prefix);
        if let Some(parent) = node.parent {
            self.nodes.get_mut(&parent).unwrap().child_count -= 1;
        }
        let retired = RetiredResources {
            checkpoint: id,
            key: node.prefix,
            heir: None,
            pages: node.tail_pages,
            snapshot: node.snapshot,
            retire_after: node.last_use.max(node.ready_after),
        };
        self.retirement.push(retired.clone());
        retired
    }

    fn compact_ancestor(&mut self, id: CheckpointId) -> RetiredResources {
        let node = self.nodes.remove(&id).unwrap();
        let child = self
            .nodes
            .values_mut()
            .find(|child| child.parent == Some(id))
            .unwrap();
        debug_assert_eq!(node.direct_pins, 0);
        debug_assert_eq!(node.child_count, 1);
        debug_assert_eq!(node.pins, child.pins);
        let heir = child.id;
        let mut pages = node.tail_pages;
        pages.append(&mut child.tail_pages);
        child.tail_pages = pages;
        child.parent = node.parent;
        // Transferred pages retain outstanding reads from the removed ancestor.
        child.last_use = child.last_use.max(node.last_use).max(node.ready_after);
        self.prefixes.remove(&node.prefix);
        let retired = RetiredResources {
            checkpoint: id,
            key: node.prefix,
            heir: Some(heir),
            pages: Vec::new(),
            snapshot: node.snapshot,
            retire_after: node.last_use.max(node.ready_after),
        };
        self.retirement.push(retired.clone());
        retired
    }

    pub fn cancel_publication(&mut self, id: CheckpointId) -> Result<RetiredResources> {
        let node = self.checkpoint(id)?;
        if node.ready || node.pins != 0 || node.child_count != 0 {
            return Err(
                "only an unpublished, unreferenced leaf publication can be cancelled".into(),
            );
        }
        Ok(self.retire_leaf(id))
    }

    fn eviction_candidates(&self) -> impl Iterator<Item = &Checkpoint> {
        self.nodes.values().filter(|node| {
            node.ready
                && ((node.pins == 0 && node.child_count == 0)
                    || (node.direct_pins == 0 && node.child_count == 1))
        })
    }

    /// Thin dense checkpoint chains before retiring endpoints under snapshot pressure.
    /// Only advance_completed permits reuse of the returned storage.
    pub fn evict(&mut self, target: EvictionTarget) -> EvictionReport {
        let mut report = EvictionReport::default();
        while report.pages_scheduled < target.pages || report.snapshots_scheduled < target.snapshots
        {
            let interior = (target.pages == 0)
                .then(|| {
                    self.nodes
                        .values()
                        .filter_map(|child| {
                            let parent = self.nodes.get(&child.parent?)?;
                            (parent.ready && parent.direct_pins == 0 && parent.child_count == 1)
                                .then_some((parent, child))
                        })
                        .min_by_key(|(node, child)| {
                            let start = node
                                .parent
                                .map_or(0, |id| self.nodes[&id].prefix.consumed_tokens);
                            (
                                node.parent.is_none(),
                                child.prefix.consumed_tokens - start,
                                node.accessed,
                                node.id,
                            )
                        })
                        .map(|(node, _)| node.id)
                })
                .flatten();
            let candidate = interior.or_else(|| {
                self.eviction_candidates()
                    .min_by_key(|node| (node.accessed, node.id))
                    .map(|node| node.id)
            });
            let Some(id) = candidate else {
                break;
            };
            let retired = if self.nodes[&id].child_count == 0 {
                self.retire_leaf(id)
            } else {
                self.compact_ancestor(id)
            };
            report.pages_scheduled += retired.pages.len();
            report.snapshots_scheduled += 1;
            report.retired.push(retired);
            // Snapshot aliases can release KV pages; let the scheduler recheck actual backing.
            if target.pages != 0 && report.snapshots_scheduled >= target.snapshots {
                break;
            }
        }
        report
    }

    /// What `evict` could retire if `pin(id)` had happened: both of its rules
    /// run to a fixpoint on the pin-adjusted counts, read-only. A node's
    /// unready children count as live; compaction keeps a node's pages (they
    /// move to its child and are reclaimed with it) and reclaims its snapshot.
    pub fn reclaimable_if_pinned(&self, id: CheckpointId) -> Result<Reclaimable> {
        Ok(self.reclaimable(&self.chain(id)?))
    }

    fn reclaimable(&self, chain: &[CheckpointId]) -> Reclaimable {
        let pinned = |node: &Checkpoint| node.pins + u64::from(chain.contains(&node.id));
        let direct =
            |node: &Checkpoint| node.direct_pins + u64::from(chain.first() == Some(&node.id));
        let mut live: HashMap<CheckpointId, usize> = HashMap::new();
        let mut kids: HashMap<CheckpointId, Vec<CheckpointId>> = HashMap::new();
        let mut parent: HashMap<CheckpointId, Option<CheckpointId>> = HashMap::new();
        for node in self.nodes.values() {
            live.insert(node.id, node.child_count);
            parent.insert(node.id, node.parent);
            if let Some(up) = node.parent {
                kids.entry(up).or_default().push(node.id);
            }
        }
        // Nodes whose pages a node now holds because they were compacted into it.
        let mut absorbed: HashMap<CheckpointId, Vec<CheckpointId>> = HashMap::new();
        let mut removed = HashSet::new();
        let mut out = Reclaimable::default();
        let mut work: Vec<_> = self.nodes.keys().copied().collect();
        while let Some(next) = work.pop() {
            if removed.contains(&next) {
                continue;
            }
            let node = &self.nodes[&next];
            if !node.ready {
                continue;
            }
            let children = live[&next];
            if pinned(node) == 0 && children == 0 {
                removed.insert(next);
                out.snapshots.push(node.snapshot);
                out.pages.extend_from_slice(&node.tail_pages);
                for from in absorbed.remove(&next).unwrap_or_default() {
                    out.pages.extend_from_slice(&self.nodes[&from].tail_pages);
                }
                if let Some(up) = parent[&next] {
                    *live.get_mut(&up).unwrap() -= 1;
                    work.push(up);
                }
            } else if direct(node) == 0 && children == 1 {
                let heir = kids[&next]
                    .iter()
                    .copied()
                    .find(|child| !removed.contains(child) && parent[child] == Some(next))
                    .unwrap();
                removed.insert(next);
                out.snapshots.push(node.snapshot);
                let mut moved = absorbed.remove(&next).unwrap_or_default();
                moved.push(next);
                absorbed.entry(heir).or_default().extend(moved);
                let up = parent[&next];
                parent.insert(heir, up);
                if let Some(up) = up {
                    kids.entry(up).or_default().push(heir);
                }
            }
        }
        out
    }

    pub(crate) fn pending_retirement(&self) -> &[RetiredResources] {
        &self.retirement
    }

    /// Publish completed snapshots and release retired IDs on an ordered fence.
    /// The caller must cover model work and asynchronous copies with this fence.
    pub fn advance_completed(&mut self, completed: Fence) -> Result<CompletionUpdate> {
        if completed < self.completed {
            return Err("completed GPU fence must not move backwards".into());
        }
        self.completed = completed;
        let mut update = CompletionUpdate::default();
        for node in self.nodes.values_mut() {
            if !node.ready && node.ready_after <= completed {
                node.ready = true;
                update.published.push(node.id);
            }
        }
        self.retirement.retain(|record| {
            if record.retire_after > completed {
                return true;
            }
            for page in &record.pages {
                self.owned_pages.remove(page);
            }
            self.owned_snapshots.remove(&record.snapshot);
            update.reusable.pages.extend_from_slice(&record.pages);
            update.reusable.snapshots.push(record.snapshot);
            false
        });
        Ok(update)
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            published_checkpoints: self.nodes.values().filter(|node| node.ready).count(),
            pending_checkpoints: self.nodes.values().filter(|node| !node.ready).count(),
            cached_pages: self.nodes.values().map(|node| node.tail_pages.len()).sum(),
            retired_pages: self
                .retirement
                .iter()
                .map(|record| record.pages.len())
                .sum(),
            retired_snapshots: self.retirement.len(),
            owned_pages: self.owned_pages.len(),
            owned_snapshots: self.owned_snapshots.len(),
        }
    }
}

#[cfg(test)]
mod tests;
