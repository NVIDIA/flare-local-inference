use super::*;

fn history(blocks: u32, offset: u32) -> PrefixHistory {
    let mut history = PrefixHistory::new([7; 32]);
    history
        .append(
            &(0..blocks * 64)
                .map(|token| token + offset)
                .collect::<Vec<_>>(),
            &[],
        )
        .unwrap();
    history
}

fn cache() -> PrefixCheckpointCache {
    PrefixCheckpointCache::new([7; 32], 32, 32, 64).unwrap()
}

fn proposal(
    parent: Option<CheckpointId>,
    consumed_tokens: u64,
    pages: &[u32],
    snapshot: u32,
    fence: u64,
) -> Publication {
    Publication {
        parent,
        consumed_tokens,
        tail_pages: pages.iter().copied().map(PageId).collect(),
        snapshot: StateSnapshot {
            id: SnapshotId(snapshot),
            consumed_tokens,
        },
        ready_after: Fence(fence),
    }
}

fn created(
    cache: &mut PrefixCheckpointCache,
    history: &PrefixHistory,
    proposal: &Publication,
) -> CheckpointId {
    match cache.publish(history, proposal).unwrap() {
        PublishResult::Created { id, .. } => id,
        result => panic!("expected a created checkpoint, received {result:?}"),
    }
}

fn hexadecimal(digest: [u8; 16]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn physical_page_ownership_is_independent_of_prefix_hash_blocks() {
    let history = history(9, 0);
    let mut cache = PrefixCheckpointCache::new([7; 32], 8, 8, 256).unwrap();
    assert!(PrefixCheckpointCache::new([7; 32], 8, 8, 0).is_err());
    assert!(
        cache
            .publish(&history, &proposal(None, 65, &[0], 0, 0))
            .is_err()
    );
    let system = created(&mut cache, &history, &proposal(None, 65, &[], 0, 0));
    let aligned = created(
        &mut cache,
        &history,
        &proposal(Some(system), 256, &[0], 1, 0),
    );
    let prompt = created(
        &mut cache,
        &history,
        &proposal(Some(aligned), 300, &[], 2, 0),
    );
    let end = created(
        &mut cache,
        &history,
        &proposal(Some(prompt), 513, &[1], 3, 0),
    );
    assert_eq!(cache.page_chain(system).unwrap(), vec![]);
    assert_eq!(cache.page_chain(prompt).unwrap(), vec![PageId(0)]);
    assert_eq!(cache.page_chain(end).unwrap(), vec![PageId(0), PageId(1)]);
    let hit = cache.lookup(&history).unwrap();
    assert_eq!(hit.consumed_tokens, 513);
    assert_eq!(hit.pages, vec![PageId(0), PageId(1)]);
    assert_eq!(history.aligned_boundaries().len(), 9);
}

#[test]
fn hash_matches_independent_blake2b128_vectors() {
    let mut history = PrefixHistory::new([7; 32]);
    assert_eq!(
        hexadecimal(history.key_at(0).unwrap().digest),
        "6f0178338c96c0355272862173689b48"
    );
    history
        .append(
            &(0..128).collect::<Vec<_>>(),
            &[
                ConditioningEvent {
                    position: 3,
                    identity: [1; 32],
                },
                ConditioningEvent {
                    position: 64,
                    identity: [2; 32],
                },
            ],
        )
        .unwrap();
    assert_eq!(
        hexadecimal(history.key_at(64).unwrap().digest),
        "2b20a8f1daa365699d9a90bb10600d30"
    );
    assert_eq!(
        hexadecimal(history.key_at(128).unwrap().digest),
        "df75df435a8efc99d5e398479fdf2519"
    );
}

#[test]
fn chunking_retains_partial_tokens_and_conditioning_until_complete() {
    let mut whole = PrefixHistory::new([7; 32]);
    let events = [
        ConditioningEvent {
            position: 3,
            identity: [1; 32],
        },
        ConditioningEvent {
            position: 64,
            identity: [2; 32],
        },
    ];
    whole
        .append(&(0..128).collect::<Vec<_>>(), &events)
        .unwrap();
    let mut chunks = PrefixHistory::new([7; 32]);
    chunks
        .append(&(0..13).collect::<Vec<_>>(), &events[..1])
        .unwrap();
    assert_eq!(chunks.token_count(), 13);
    assert!(chunks.aligned_boundaries().is_empty());
    chunks
        .append(&(13..83).collect::<Vec<_>>(), &events[1..])
        .unwrap();
    assert_eq!(chunks.aligned_boundaries().len(), 1);
    assert_eq!(chunks.key_at(64), whole.key_at(64));
    assert_eq!(chunks.key_at(83), whole.key_at(83));
    chunks.append(&(83..128).collect::<Vec<_>>(), &[]).unwrap();
    assert_eq!(chunks.aligned_boundaries(), whole.aligned_boundaries());
}

#[test]
fn conditioning_and_model_identity_separate_keys_without_changing_earlier_prefixes() {
    let original = history(1, 0);
    let mut left = original.clone();
    let mut right = original.clone();
    left.append(
        &[11; 64],
        &[ConditioningEvent {
            position: 64,
            identity: [1; 32],
        }],
    )
    .unwrap();
    right
        .append(
            &[11; 64],
            &[ConditioningEvent {
                position: 64,
                identity: [2; 32],
            }],
        )
        .unwrap();
    assert_eq!(left.key_at(64), original.key_at(64));
    assert_eq!(right.key_at(64), original.key_at(64));
    assert_ne!(left.key_at(128), right.key_at(128));
    let mut cache = cache();
    let parent = created(&mut cache, &original, &proposal(None, 64, &[0], 0, 0));
    let image_checkpoint = created(&mut cache, &left, &proposal(Some(parent), 128, &[1], 1, 0));
    assert_eq!(
        cache.lookup(&left).unwrap().checkpoint,
        Some(image_checkpoint)
    );
    assert_eq!(cache.lookup(&right).unwrap().checkpoint, Some(parent));
    let mut other_model = PrefixHistory::new([8; 32]);
    other_model
        .append(&(0..64).collect::<Vec<_>>(), &[])
        .unwrap();
    assert_ne!(other_model.key_at(64), original.key_at(64));
    assert!(cache.lookup(&other_model).is_err());
}

#[test]
fn conditioning_order_is_significant_and_invalid_append_is_atomic() {
    let events = [
        ConditioningEvent {
            position: 5,
            identity: [1; 32],
        },
        ConditioningEvent {
            position: 5,
            identity: [2; 32],
        },
    ];
    let mut first = PrefixHistory::new([7; 32]);
    let mut second = first.clone();
    first.append(&[1; 64], &events).unwrap();
    second.append(&[1; 64], &[events[1], events[0]]).unwrap();
    assert_ne!(first.key_at(64), second.key_at(64));
    let before = first.aligned_boundaries().to_vec();
    assert!(first.append(&[2; 64], &events).is_err());
    assert_eq!(first.token_count(), 64);
    assert_eq!(first.aligned_boundaries(), before);
    assert!(
        first
            .append(
                &[],
                &[ConditioningEvent {
                    position: 64,
                    identity: [0; 32]
                }]
            )
            .is_err()
    );
}

#[test]
fn lookup_probes_absent_boundaries_and_returns_full_prompt_hit() {
    let mut cache = cache();
    let history = history(3, 0);
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, None);
    let id = created(&mut cache, &history, &proposal(None, 192, &[0, 1, 2], 0, 0));
    let hit = cache.lookup(&history).unwrap();
    assert_eq!(hit.checkpoint, Some(id));
    assert_eq!(hit.consumed_tokens, history.token_count());
    assert_eq!(hit.pages, [PageId(0), PageId(1), PageId(2)]);
    assert_eq!(hit.snapshot, Some(SnapshotId(0)));
}

#[test]
fn unaligned_suffix_is_not_part_of_checkpoint() {
    let mut cache = cache();
    let mut history = history(1, 0);
    let id = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 0));
    history.append(&[100; 17], &[]).unwrap();
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, Some(id));
    assert_eq!(cache.lookup(&history).unwrap().consumed_tokens, 64);
    let mut invalid = proposal(None, 64, &[1], 1, 0);
    invalid.snapshot.consumed_tokens = 81;
    let error = cache.publish(&history, &invalid).unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Execution(crate::ExecutionError::State(_))
    ));
    assert_eq!(
        error.to_string(),
        "snapshot consumed-token length must exactly equal the checkpoint endpoint"
    );
}

#[test]
fn snapshots_stay_invisible_until_their_completion_fence() {
    let mut cache = cache();
    let history = history(2, 0);
    let id = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 5));
    assert_eq!(cache.stats().pending_checkpoints, 1);
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, None);
    assert!(cache.pin(id).is_err());
    assert!(cache.mark_used(id, Fence(6)).is_err());
    assert!(
        cache
            .publish(&history, &proposal(Some(id), 128, &[1], 1, 6))
            .is_err()
    );
    assert!(
        cache
            .advance_completed(Fence(4))
            .unwrap()
            .published
            .is_empty()
    );
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, None);
    assert_eq!(cache.advance_completed(Fence(5)).unwrap().published, [id]);
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, Some(id));
    let before = cache.stats();
    assert!(cache.advance_completed(Fence(4)).is_err());
    assert_eq!(cache.completed(), Fence(5));
    assert_eq!(cache.stats(), before);
}

#[test]
fn duplicate_publication_retains_unused_resources_with_caller() {
    let mut cache = cache();
    let history = history(1, 0);
    let id = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 4));
    let duplicate = proposal(None, 64, &[1], 1, 8);
    assert_eq!(
        cache.publish(&history, &duplicate).unwrap(),
        PublishResult::Existing { id, ready: false }
    );
    assert_eq!(cache.stats().owned_pages, 1);
    assert_eq!(cache.stats().owned_snapshots, 1);
    cache.advance_completed(Fence(4)).unwrap();
    assert_eq!(
        cache.publish(&history, &duplicate).unwrap(),
        PublishResult::Existing { id, ready: true }
    );
    let alternate = super::tests::history(1, 900);
    created(&mut cache, &alternate, &duplicate);
    assert_eq!(cache.stats().owned_pages, 2);
}

#[test]
fn branching_pins_ancestors_and_prevents_direct_ancestor_unpin() {
    let mut cache = cache();
    let left = history(2, 0);
    let mut right = history(1, 0);
    right.append(&[999; 64], &[]).unwrap();
    let parent = created(&mut cache, &left, &proposal(None, 64, &[0], 0, 0));
    let left_id = created(&mut cache, &left, &proposal(Some(parent), 128, &[1], 1, 0));
    let right_id = created(&mut cache, &right, &proposal(Some(parent), 128, &[2], 2, 0));
    assert_eq!(cache.page_chain(left_id).unwrap(), [PageId(0), PageId(1)]);
    assert_eq!(cache.page_chain(right_id).unwrap(), [PageId(0), PageId(2)]);
    cache.pin(left_id).unwrap();
    cache.pin(right_id).unwrap();
    assert_eq!(cache.checkpoint(parent).unwrap().pin_count(), 2);
    assert!(cache.unpin(parent).is_err());
    assert_eq!(cache.checkpoint(parent).unwrap().pin_count(), 2);
    cache.unpin(left_id).unwrap();
    assert_eq!(cache.checkpoint(parent).unwrap().pin_count(), 1);
    let evicted = cache.evict(EvictionTarget {
        pages: 99,
        snapshots: 99,
    });
    assert_eq!(evicted.retired.len(), 2);
    assert_eq!(evicted.retired[0].checkpoint, left_id);
    assert_eq!(evicted.retired[1].checkpoint, parent);
    assert!(evicted.retired[1].pages.is_empty());
    assert_eq!(cache.checkpoint(right_id).unwrap().parent(), None);
    assert_eq!(cache.page_chain(right_id).unwrap(), [PageId(0), PageId(2)]);
    assert_eq!(cache.lookup(&right).unwrap().checkpoint, Some(right_id));
    cache.unpin(right_id).unwrap();
    let evicted = cache.evict(EvictionTarget {
        pages: 2,
        snapshots: 2,
    });
    assert_eq!(
        evicted
            .retired
            .iter()
            .map(|record| record.checkpoint)
            .collect::<Vec<_>>(),
        [right_id]
    );
}

#[test]
fn ancestor_compaction_preserves_pins_page_order_and_pending_gpu_reads() {
    let mut cache = cache();
    let history = history(3, 0);
    let root = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 0));
    let middle = created(&mut cache, &history, &proposal(Some(root), 128, &[1], 1, 0));
    let leaf = created(
        &mut cache,
        &history,
        &proposal(Some(middle), 192, &[2], 2, 0),
    );
    cache.pin(root).unwrap();
    cache.pin(middle).unwrap();
    cache.pin(leaf).unwrap();
    cache.mark_used(middle, Fence(20)).unwrap();
    assert!(
        cache
            .evict(EvictionTarget {
                pages: 0,
                snapshots: 1
            })
            .retired
            .is_empty()
    );
    cache.unpin(middle).unwrap();
    let retired = cache.evict(EvictionTarget {
        pages: 1,
        snapshots: 0,
    });
    assert_eq!(retired.retired[0].checkpoint, middle);
    assert_eq!(retired.pages_scheduled, 0);
    assert_eq!(retired.snapshots_scheduled, 1);
    assert_eq!(cache.checkpoint(root).unwrap().child_count(), 1);
    assert_eq!(cache.checkpoint(root).unwrap().pin_count(), 2);
    assert_eq!(cache.checkpoint(leaf).unwrap().parent(), Some(root));
    assert_eq!(cache.checkpoint(leaf).unwrap().pin_count(), 1);
    assert_eq!(
        cache.page_chain(leaf).unwrap(),
        [PageId(0), PageId(1), PageId(2)]
    );
    assert_eq!(
        cache.lookup(&history).unwrap().snapshot,
        Some(SnapshotId(2))
    );
    assert!(
        cache
            .lookup(&super::tests::history(2, 0))
            .unwrap()
            .consumed_tokens
            == 64
    );
    assert_eq!(cache.stats().owned_pages, 3);
    let duplicate = proposal(Some(root), 192, &[3, 4], 3, 0);
    assert_eq!(
        cache.publish(&history, &duplicate).unwrap(),
        PublishResult::Existing {
            id: leaf,
            ready: true
        }
    );
    assert_eq!(cache.stats().owned_pages, 3);
    assert_eq!(cache.stats().owned_snapshots, 3);
    assert!(
        cache
            .advance_completed(Fence(19))
            .unwrap()
            .reusable
            .snapshots
            .is_empty()
    );
    cache.unpin(leaf).unwrap();
    cache.unpin(root).unwrap();
    let retired = cache.evict(EvictionTarget {
        pages: 3,
        snapshots: 2,
    });
    assert!(
        retired
            .retired
            .iter()
            .all(|record| record.retire_after == Fence(20))
    );
    assert!(
        cache
            .advance_completed(Fence(19))
            .unwrap()
            .reusable
            .pages
            .is_empty()
    );
    let reclaimed = cache.advance_completed(Fence(20)).unwrap().reusable;
    assert_eq!(reclaimed.pages, [PageId(0), PageId(1), PageId(2)]);
    assert_eq!(
        reclaimed.snapshots,
        [SnapshotId(1), SnapshotId(0), SnapshotId(2)]
    );
    assert_eq!(cache.stats().owned_pages, 0);
    assert_eq!(cache.stats().owned_snapshots, 0);
}

#[test]
fn compacted_pending_child_retains_parent_pages_until_cancellation_completes() {
    let mut cache = cache();
    let history = history(2, 0);
    let parent = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 0));
    let child = created(
        &mut cache,
        &history,
        &proposal(Some(parent), 128, &[1], 1, 9),
    );
    let retired = cache.evict(EvictionTarget {
        pages: 0,
        snapshots: 1,
    });
    assert_eq!(retired.retired[0].checkpoint, parent);
    assert!(retired.retired[0].pages.is_empty());
    assert_eq!(retired.retired[0].retire_after, Fence(9));
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, None);
    assert_eq!(cache.page_chain(child).unwrap(), [PageId(0), PageId(1)]);
    assert_eq!(
        cache.cancel_publication(child).unwrap().pages,
        [PageId(0), PageId(1)]
    );
    assert_eq!(
        cache.advance_completed(Fence(8)).unwrap().reusable,
        ReclaimedResources::default()
    );
    let reclaimed = cache.advance_completed(Fence(9)).unwrap().reusable;
    assert_eq!(reclaimed.pages, [PageId(0), PageId(1)]);
    assert_eq!(reclaimed.snapshots, [SnapshotId(0), SnapshotId(1)]);
}

#[test]
fn eviction_uses_leaf_lru_and_honors_separate_snapshot_pressure() {
    let mut cache = cache();
    let first_history = history(1, 0);
    let first = created(&mut cache, &first_history, &proposal(None, 64, &[0], 0, 0));
    let second = created(
        &mut cache,
        &history(1, 1000),
        &proposal(None, 64, &[1], 1, 0),
    );
    let third = created(
        &mut cache,
        &history(1, 2000),
        &proposal(None, 64, &[2], 2, 0),
    );
    cache.lookup(&first_history).unwrap();
    let report = cache.evict(EvictionTarget {
        pages: 1,
        snapshots: 0,
    });
    assert_eq!(report.retired[0].checkpoint, second);
    let report = cache.evict(EvictionTarget {
        pages: 0,
        snapshots: 1,
    });
    assert_eq!(report.retired[0].checkpoint, third);
    assert_eq!(
        cache.lookup(&first_history).unwrap().checkpoint,
        Some(first)
    );
    assert_eq!(cache.stats().retired_pages, 2);
    assert_eq!(cache.stats().owned_pages, 3);
}

#[test]
fn both_eviction_targets_must_be_met() {
    let mut cache = cache();
    created(
        &mut cache,
        &history(3, 0),
        &proposal(None, 192, &[0, 1, 2], 0, 0),
    );
    created(
        &mut cache,
        &history(1, 1000),
        &proposal(None, 64, &[3], 1, 0),
    );
    let report = cache.evict(EvictionTarget {
        pages: 1,
        snapshots: 2,
    });
    assert_eq!(report.pages_scheduled, 4);
    assert_eq!(report.snapshots_scheduled, 2);
    assert_eq!(cache.stats().cached_pages, 0);
}

#[test]
fn cancelled_request_unpins_but_shared_pages_wait_for_gpu_retirement() {
    let mut cache = cache();
    let history = history(2, 0);
    let parent = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 0));
    let child = created(
        &mut cache,
        &history,
        &proposal(Some(parent), 128, &[1], 1, 0),
    );
    cache.pin(child).unwrap();
    cache.mark_used(child, Fence(20)).unwrap();
    cache.unpin(child).unwrap();
    let retired = cache
        .evict(EvictionTarget {
            pages: 2,
            snapshots: 2,
        })
        .retired;
    assert!(
        retired
            .iter()
            .all(|record| record.retire_after == Fence(20))
    );
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, None);
    assert!(
        cache
            .advance_completed(Fence(19))
            .unwrap()
            .reusable
            .pages
            .is_empty()
    );
    assert!(
        cache
            .publish(&history, &proposal(None, 64, &[0], 0, 0))
            .is_err()
    );
    assert!(
        cache
            .publish(&history, &proposal(None, 64, &[2], 0, 0))
            .is_err()
    );
    assert_eq!(cache.stats().owned_snapshots, 2);
    let reusable = cache.advance_completed(Fence(20)).unwrap().reusable;
    assert_eq!(reusable.pages, [PageId(0), PageId(1)]);
    assert_eq!(reusable.snapshots, [SnapshotId(0), SnapshotId(1)]);
    assert_eq!(cache.stats().owned_pages, 0);
    assert!(
        cache
            .advance_completed(Fence(20))
            .unwrap()
            .reusable
            .pages
            .is_empty()
    );
}

#[test]
fn cancelling_pending_child_protects_parent_until_snapshot_work_finishes() {
    let mut cache = cache();
    let history = history(2, 0);
    let parent = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 0));
    let child = created(
        &mut cache,
        &history,
        &proposal(Some(parent), 128, &[1], 1, 9),
    );
    assert_eq!(cache.checkpoint(parent).unwrap().child_count(), 1);
    assert_eq!(
        cache.cancel_publication(child).unwrap().retire_after,
        Fence(9)
    );
    let report = cache.evict(EvictionTarget {
        pages: 1,
        snapshots: 1,
    });
    assert_eq!(report.retired[0].retire_after, Fence(9));
    assert!(
        cache
            .advance_completed(Fence(8))
            .unwrap()
            .reusable
            .pages
            .is_empty()
    );
    assert_eq!(
        cache
            .advance_completed(Fence(9))
            .unwrap()
            .reusable
            .pages
            .len(),
        2
    );
    assert!(cache.checkpoint(parent).is_err());
    assert!(cache.checkpoint(child).is_err());
}

#[test]
fn stale_ids_cannot_alias_reused_pages_or_snapshot_slots() {
    let mut cache = cache();
    let history = history(1, 0);
    let pending = proposal(None, 64, &[0], 0, 5);
    let stale = created(&mut cache, &history, &pending);
    cache.cancel_publication(stale).unwrap();
    assert!(cache.publish(&history, &pending).is_err());
    cache.advance_completed(Fence(5)).unwrap();
    let replacement = created(&mut cache, &history, &pending);
    assert!(replacement.value() > stale.value());
    assert!(cache.pin(stale).is_err());
    assert!(cache.unpin(stale).is_err());
    assert!(cache.mark_used(stale, Fence(10)).is_err());
    assert_eq!(
        cache.lookup(&history).unwrap().checkpoint,
        Some(replacement)
    );
}

#[test]
fn invalid_publications_leave_resources_and_ids_unchanged() {
    let mut cache = cache();
    let history = history(2, 0);
    let parent = created(&mut cache, &history, &proposal(None, 64, &[0], 0, 0));
    let baseline = cache.stats();
    let mut candidates = vec![
        proposal(None, 65, &[], 1, 0),
        proposal(None, 128, &[1], 1, 0),
        proposal(None, 128, &[1, 1], 1, 0),
        proposal(None, 64, &[0], 1, 0),
        proposal(None, 64, &[1], 0, 0),
        proposal(None, 64, &[32], 1, 0),
        proposal(None, 64, &[1], 32, 0),
        proposal(Some(parent), 64, &[], 1, 0),
    ];
    let mut wrong_snapshot = proposal(None, 128, &[1, 2], 1, 0);
    wrong_snapshot.snapshot.consumed_tokens = 129;
    candidates.push(wrong_snapshot);
    for invalid in candidates {
        assert!(cache.publish(&history, &invalid).is_err(), "{invalid:?}");
        assert_eq!(cache.stats(), baseline);
    }
    let valid_child = proposal(Some(parent), 128, &[1], 1, 0);
    let other_history = super::tests::history(2, 3000);
    assert!(cache.publish(&other_history, &valid_child).is_err());
    assert_eq!(cache.stats(), baseline);
    let child = created(&mut cache, &history, &valid_child);
    assert_eq!(child.value(), parent.value() + 1);
}

#[test]
fn exact_tail_keys_preserve_conditioning_order_and_append_boundaries() {
    let tokens: Vec<_> = (0..130).collect();
    let events = [
        ConditioningEvent {
            position: 2,
            identity: [1; 32],
        },
        ConditioningEvent {
            position: 63,
            identity: [2; 32],
        },
        ConditioningEvent {
            position: 64,
            identity: [3; 32],
        },
    ];
    let mut whole = PrefixHistory::new([7; 32]);
    whole.append(&tokens, &events).unwrap();
    for end in [1, 2, 3, 10, 15, 63, 64, 65, 66, 67, 127, 128, 129, 130] {
        let mut chunks = PrefixHistory::new([7; 32]);
        for start in 0..end {
            let local: Vec<_> = events
                .iter()
                .filter(|e| e.position == start as u64)
                .copied()
                .collect();
            chunks.append(&tokens[start..start + 1], &local).unwrap();
        }
        let key = chunks.key_at(end as u64).unwrap();
        assert_eq!(Some(key), whole.key_at(end as u64));
        chunks
            .append(
                &[999],
                &[ConditioningEvent {
                    position: end as u64,
                    identity: [9; 32],
                }],
            )
            .unwrap();
        assert_eq!(chunks.key_at(end as u64), Some(key));
    }
    let mut changed = events;
    changed[0].identity = [5; 32];
    let mut other = PrefixHistory::new([7; 32]);
    other.append(&tokens, &changed).unwrap();
    assert_eq!(whole.key_at(2), other.key_at(2));
    assert_ne!(whole.key_at(3), other.key_at(3));
    assert_ne!(whole.key_at(65), other.key_at(65));
}

#[test]
fn exact_zero_page_children_compact_without_losing_pages_or_read_fences() {
    let history = history(3, 0);
    let mut cache = cache();
    let first = created(&mut cache, &history, &proposal(None, 3, &[], 0, 0));
    let second = created(&mut cache, &history, &proposal(Some(first), 63, &[], 1, 0));
    let third = created(
        &mut cache,
        &history,
        &proposal(Some(second), 65, &[4], 2, 0),
    );
    cache.pin(third).unwrap();
    cache.mark_used(third, Fence(7)).unwrap();
    let evicted = cache.evict(EvictionTarget {
        pages: 0,
        snapshots: 2,
    });
    assert_eq!(evicted.snapshots_scheduled, 2);
    assert_eq!(evicted.pages_scheduled, 0);
    assert_eq!(cache.page_chain(third).unwrap(), vec![PageId(4)]);
    assert!(cache.checkpoint(first).is_err());
    assert!(cache.checkpoint(second).is_err());
    assert!(
        cache
            .advance_completed(Fence(6))
            .unwrap()
            .reusable
            .snapshots
            .is_empty()
    );
    let retired = cache.advance_completed(Fence(7)).unwrap().reusable;
    assert_eq!(retired.snapshots.len(), 2);
    assert!(retired.pages.is_empty());
    cache.unpin(third).unwrap();
    let evicted = cache.evict(EvictionTarget {
        pages: 1,
        snapshots: 1,
    });
    assert_eq!(evicted.pages_scheduled, 1);
    assert_eq!(
        cache.advance_completed(Fence(7)).unwrap().reusable.pages,
        vec![PageId(4)]
    );
}

#[test]
fn exact_pending_and_duplicate_checkpoints_keep_external_tail_ownership() {
    let history = history(2, 0);
    let mut cache = cache();
    let parent = created(&mut cache, &history, &proposal(None, 63, &[], 0, 0));
    let child = created(
        &mut cache,
        &history,
        &proposal(Some(parent), 65, &[4], 1, 7),
    );
    assert_eq!(cache.lookup(&history).unwrap().checkpoint, Some(parent));
    let duplicate = cache
        .publish(&history, &proposal(Some(parent), 65, &[5], 2, 9))
        .unwrap();
    assert_eq!(
        duplicate,
        PublishResult::Existing {
            id: child,
            ready: false
        }
    );
    assert!(!cache.owned_pages.contains(&PageId(5)));
    assert!(!cache.owned_snapshots.contains(&SnapshotId(2)));
    cache.cancel_publication(child).unwrap();
    assert!(
        cache
            .advance_completed(Fence(6))
            .unwrap()
            .reusable
            .pages
            .is_empty()
    );
    assert_eq!(
        cache.advance_completed(Fence(7)).unwrap().reusable.pages,
        vec![PageId(4)]
    );
    let retired = cache.evict(EvictionTarget {
        pages: 0,
        snapshots: 1,
    });
    assert_eq!(retired.retired[0].retire_after, Fence(9));
    assert!(
        cache
            .advance_completed(Fence(8))
            .unwrap()
            .reusable
            .snapshots
            .is_empty()
    );
    assert_eq!(
        cache
            .advance_completed(Fence(9))
            .unwrap()
            .reusable
            .snapshots,
        vec![SnapshotId(0)]
    );
}

#[test]
fn bucket_digest_names_the_aligned_boundary_at_or_below_the_endpoint() {
    let mut cache = cache();
    let history = history(3, 0);
    let short = created(&mut cache, &history, &proposal(None, 3, &[], 0, 0));
    let aligned = created(
        &mut cache,
        &history,
        &proposal(Some(short), 128, &[0, 1], 1, 0),
    );
    let unaligned = created(
        &mut cache,
        &history,
        &proposal(Some(aligned), 150, &[], 2, 0),
    );
    let root = history.key_at(0).unwrap().digest;
    let block = history.key_at(128).unwrap().digest;
    assert_eq!(cache.checkpoint(short).unwrap().bucket(), root);
    assert_eq!(cache.checkpoint(aligned).unwrap().bucket(), block);
    assert_eq!(cache.checkpoint(unaligned).unwrap().bucket(), block);
    assert_eq!(history.bucket_at(150), Some(block));
    assert_ne!(root, block);
    assert_ne!(cache.checkpoint(unaligned).unwrap().prefix().digest, block);
}

#[test]
fn oldest_ancestor_compacts_before_a_newer_unpinned_leaf() {
    for target in [
        EvictionTarget {
            pages: 0,
            snapshots: 1,
        },
        EvictionTarget {
            pages: 1,
            snapshots: 0,
        },
    ] {
        let full = history(4, 0);
        let mut cache = cache();
        let root = created(&mut cache, &full, &proposal(None, 64, &[0], 0, 0));
        let child = created(&mut cache, &full, &proposal(Some(root), 128, &[1], 1, 0));
        let other = history(4, 9);
        let leaf = created(&mut cache, &other, &proposal(None, 64, &[2], 2, 0));
        let report = cache.evict(target);
        assert_eq!(report.retired[0].checkpoint, root);
        assert!(cache.checkpoint(leaf).is_ok());
        assert_eq!(cache.page_chain(child).unwrap(), [PageId(0), PageId(1)]);
    }
}
