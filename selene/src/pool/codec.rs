//! Conversions between the pool's own types and the wire messages.

use crate::shared::Tier;
use std::collections::HashMap;

use crate::{PrefixKey, levels::Level, plan};

use super::{
    announce::{IndexEntry, Notice},
    identity::Scope,
    types::{PoolCandidate, SourceRef},
    wire,
};

pub(crate) fn key_to(key: PrefixKey) -> wire::PrefixKey {
    wire::PrefixKey {
        consumed_tokens: key.consumed_tokens,
        digest: key.digest.to_vec(),
    }
}

pub(crate) fn key_from(key: &wire::PrefixKey) -> Option<PrefixKey> {
    Some(PrefixKey {
        consumed_tokens: key.consumed_tokens,
        digest: key.digest.as_slice().try_into().ok()?,
    })
}

pub(crate) use crate::shared::{tier_from, tier_to};

pub(crate) fn level_from(level: i32) -> Option<Level> {
    match wire::Level::try_from(level).ok()? {
        wire::Level::ThisGpu => Some(Level::ThisGpu),
        wire::Level::NeighbourGpu => Some(Level::NeighbourGpu),
        wire::Level::HostDram => Some(Level::HostDram),
        wire::Level::HostNvme => Some(Level::HostNvme),
        wire::Level::RemoteNvme => Some(Level::RemoteNvme),
        wire::Level::Unspecified => None,
    }
}

pub(crate) fn entry_to(entry: &IndexEntry) -> wire::IndexEntry {
    wire::IndexEntry {
        key: Some(key_to(entry.key)),
        has_parent: entry.parent.is_some(),
        parent: entry.parent.map(key_to),
        bucket: entry.bucket.to_vec(),
        tail_pages: entry.tail_pages,
        row: entry.row,
        tier: tier_to(entry.tier) as i32,
        bytes: entry.bytes,
    }
}

pub(crate) fn notice_to(notice: &Notice) -> wire::Notice {
    use wire::notice::Kind;
    wire::Notice {
        kind: Some(match notice {
            Notice::Saved(entry) => Kind::Saved(wire::Saved {
                entry: Some(entry_to(entry)),
            }),
            Notice::Spilled { to, from_tier } => Kind::Spilled(wire::Spilled {
                to: Some(entry_to(to)),
                from_tier: tier_to(*from_tier) as i32,
            }),
            Notice::Evicted { key, tier, heir } => Kind::Evicted(wire::Evicted {
                key: Some(key_to(*key)),
                tier: tier_to(*tier) as i32,
                kind: if heir.is_some() {
                    wire::EvictKind::Compacted
                } else {
                    wire::EvictKind::Removed
                } as i32,
                heir: heir.as_ref().map(entry_to),
            }),
        }),
    }
}

pub(crate) fn scope_to(scope: &Scope) -> wire::Scope {
    wire::Scope {
        model_identity: scope.model_identity.to_vec(),
        layout_fingerprint: scope.layout_fingerprint.to_vec(),
        geometry: Some(wire::Geometry {
            page_tokens: scope.geometry.page_tokens,
            ranks: scope
                .geometry
                .ranks
                .iter()
                .map(|rank| wire::RankPlanes {
                    plane_bytes: rank.plane_bytes.clone(),
                    row_bytes: rank.row_bytes,
                })
                .collect(),
            page_record_bytes: scope.geometry.page_record_bytes,
            row_record_bytes: scope.geometry.row_record_bytes,
        }),
    }
}

/// Interns sources within one bounded reply. IDs are never reused: older plans
/// and tickets retain their original IDs without retaining this table.
#[derive(Default)]
pub(crate) struct Sources {
    ids: HashMap<(SourceRef, Tier), u64>,
    next: u64,
}

impl Sources {
    pub(crate) fn begin_reply(&mut self) {
        self.ids.clear();
    }

    fn id(&mut self, source: &SourceRef, tier: Tier) -> Option<plan::SourceId> {
        if let Some(id) = self.ids.get(&(source.clone(), tier)) {
            return Some(plan::SourceId(*id));
        }
        self.next = self.next.checked_add(1)?;
        self.ids.insert((source.clone(), tier), self.next);
        Some(plan::SourceId(self.next))
    }

    /// A reply candidate in the planner's terms, or `None` if it is malformed.
    pub(crate) fn candidate(&mut self, wire: &wire::Candidate) -> Option<PoolCandidate> {
        let level = level_from(wire.level).filter(|level| *level != Level::ThisGpu)?;
        let tier = tier_from(wire.tier)?;
        let source = wire.source.as_ref()?;
        let source = SourceRef {
            instance: source.instance_id.clone(),
            session_incarnation: source.session_incarnation,
        };
        let nodes = wire
            .nodes
            .iter()
            .map(|node| {
                Some(plan::ChainNode {
                    key: key_from(node.key.as_ref()?)?,
                    tail_pages: node.tail_pages as usize,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(PoolCandidate {
            plan: plan::Candidate {
                level,
                source: self.id(&source, tier)?,
                key: key_from(wire.key.as_ref()?)?,
                begin_tokens: wire.begin_tokens,
                skip_pages: wire.skip_pages as usize,
                nodes,
            },
            source,
            tier,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_bookkeeping_is_bounded_without_reusing_outstanding_ids() {
        let mut sources = Sources::default();
        let first = SourceRef { instance: vec![1], session_incarnation: 1 };
        let retained = sources.id(&first, Tier::Dram).unwrap();
        assert_eq!(sources.id(&first, Tier::Dram), Some(retained));
        for session in 2..1024 {
            sources.begin_reply();
            let source = SourceRef { instance: vec![1], session_incarnation: session };
            let id = sources.id(&source, Tier::Dram).unwrap();
            assert_ne!(id, retained);
            assert_eq!(sources.id(&source, Tier::Dram), Some(id));
            assert_eq!(sources.ids.len(), 1);
        }
        sources.begin_reply();
        assert_ne!(sources.id(&first, Tier::Dram), Some(retained));
        assert_eq!(retained, plan::SourceId(1));
    }
}
