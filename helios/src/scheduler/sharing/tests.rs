use super::super::tests::{scheduler, spec};
use super::*;
use crate::cache::{Publication, StateSnapshot};

fn history(blocks: u32) -> PrefixHistory {
    let mut history = PrefixHistory::new([9; 32]);
    history
        .append(&(0..blocks * 64).collect::<Vec<_>>(), &[])
        .unwrap();
    history
}

/// Publishes a checkpoint ending at `end` tokens with the given tail pages.
fn publish(
    s: &mut RequestScheduler,
    history: &PrefixHistory,
    parent: Option<CheckpointId>,
    end: u64,
    pages: &[u32],
    ready_after: u64,
) -> CheckpointId {
    let snapshot = s.reserve_snapshot(0).unwrap();
    for page in pages {
        assert!(s.free_pages.remove(&PageId(*page)));
    }
    s.cache
        .publish(
            history,
            &Publication {
                parent,
                consumed_tokens: end,
                tail_pages: pages.iter().copied().map(PageId).collect(),
                snapshot: StateSnapshot {
                    id: snapshot,
                    consumed_tokens: end,
                },
                ready_after: Fence(ready_after),
            },
        )
        .unwrap()
        .id()
}

#[test]
fn compaction_and_leaf_eviction_are_reported_with_the_heir() {
    let mut s = scheduler(2, 16, 4);
    s.enable_evictions();
    let h = history(3);
    let a = publish(&mut s, &h, None, 64, &[0], 0);
    let b = publish(&mut s, &h, Some(a), 128, &[1], 0);
    let c = publish(&mut s, &h, Some(b), 192, &[2], 0);
    let keys: Vec<_> = [a, b, c]
        .into_iter()
        .map(|id| s.cache.checkpoint(id).unwrap().prefix())
        .collect();
    s.evict(EvictionTarget {
        pages: 0,
        snapshots: 1,
    });
    let heir = s.publication(c).unwrap();
    assert_eq!(heir.chain_pages, [PageId(0), PageId(1), PageId(2)]);
    assert_eq!(heir.parent, Some(keys[0]));
    assert_eq!(heir.tail_pages, 2);
    assert_eq!(
        s.take_evictions(),
        [EvictedCheckpoint {
            key: keys[1],
            heir: Some(heir),
        }]
    );
    assert!(s.take_evictions().is_empty());
    for _ in 0..2 {
        s.evict(EvictionTarget {
            pages: 0,
            snapshots: 1,
        });
    }
    let notices = s.take_evictions();
    assert_eq!(notices.len(), 2);
    assert_eq!(notices[0].key, keys[0]);
    assert_eq!(notices[0].heir.as_ref().unwrap().key, keys[2]);
    assert_eq!(notices[1].key, keys[2]);
    assert_eq!(notices[1].heir, None);
}

#[test]
fn an_heir_evicted_by_the_same_call_is_not_named() {
    let mut s = scheduler(2, 16, 4);
    s.enable_evictions();
    let h = history(2);
    let a = publish(&mut s, &h, None, 64, &[0], 0);
    publish(&mut s, &h, Some(a), 128, &[1], 0);
    s.evict(EvictionTarget {
        pages: 0,
        snapshots: 2,
    });
    let notices = s.take_evictions();
    assert_eq!(
        notices
            .iter()
            .map(|n| n.key.consumed_tokens)
            .collect::<Vec<_>>(),
        [64, 128]
    );
    assert!(notices.iter().all(|n| n.heir.is_none()));
}

#[test]
fn an_unready_heir_is_not_announced_and_untracked_evictions_are_silent() {
    let h = history(2);
    let mut s = scheduler(2, 16, 4);
    s.enable_evictions();
    let a = publish(&mut s, &h, None, 64, &[0], 0);
    let b = publish(&mut s, &h, Some(a), 128, &[1], 9);
    s.evict(EvictionTarget {
        pages: 0,
        snapshots: 1,
    });
    let notices = s.take_evictions();
    assert_eq!(notices[0].key, h.key_at(64).unwrap());
    assert_eq!(notices[0].heir, None);
    assert!(s.cache.checkpoint(b).is_ok());

    let mut quiet = scheduler(2, 16, 4);
    let a = publish(&mut quiet, &h, None, 64, &[0], 0);
    assert!(quiet.cache.checkpoint(a).is_ok());
    quiet.evict(EvictionTarget {
        pages: 0,
        snapshots: 1,
    });
    assert!(quiet.take_evictions().is_empty());
}

#[test]
fn queued_evictions_are_bounded_and_drop_the_oldest() {
    let mut s = scheduler(2, 16, 2);
    s.enable_evictions();
    let mut keys = Vec::new();
    for n in 0..6 {
        let mut h = PrefixHistory::new([9; 32]);
        h.append(&vec![n; 64], &[]).unwrap();
        let a = publish(&mut s, &h, None, 64, &[0], 0);
        keys.push(s.cache.checkpoint(a).unwrap().prefix());
        s.evict(EvictionTarget {
            pages: 0,
            snapshots: 1,
        });
        let update = s.cache.advance_completed(Fence(0)).unwrap();
        s.apply_completion(update);
    }
    let kept: Vec<_> = s.take_evictions().iter().map(|e| e.key).collect();
    assert_eq!(kept, keys[2..]);
    assert!(s.take_evictions_lost());
    assert!(!s.take_evictions_lost());
}

#[test]
fn lookups_by_key_only_return_ready_content() {
    let mut s = scheduler(2, 16, 4);
    let h = history(2);
    let a = publish(&mut s, &h, None, 64, &[0], 0);
    let pending = publish(&mut s, &h, Some(a), 128, &[1], 9);
    let key = s.cache.checkpoint(a).unwrap().prefix();
    let pending_key = s.cache.checkpoint(pending).unwrap().prefix();
    assert_eq!(s.checkpoint_by_key(&key), Ok(a));
    assert_eq!(
        s.checkpoint_by_key(&pending_key),
        Err(ExportRefusal::NotFound)
    );
}

#[test]
fn the_export_chain_is_trimmed_to_the_physical_start() {
    let mut s = scheduler(2, 16, 4);
    let h = history(4);
    let a = publish(&mut s, &h, None, 64, &[0], 0);
    let b = publish(&mut s, &h, Some(a), 192, &[1, 2], 0);
    let c = publish(&mut s, &h, Some(b), 256, &[3], 0);
    let keys: Vec<_> = [a, b, c]
        .into_iter()
        .map(|id| s.cache.checkpoint(id).unwrap().prefix())
        .collect();
    let shape = |chain: &ExportChain| {
        (
            chain.nodes.iter().map(|n| n.key).collect::<Vec<_>>(),
            chain.skip_pages,
        )
    };
    let all = s.export_chain(c, 0, true).unwrap();
    assert_eq!(shape(&all), (keys.clone(), 0));
    assert_eq!(all.nodes[1].key, keys[1]);
    assert_eq!(all.nodes[1].pages, [PageId(1), PageId(2)]);
    let slot = s.snapshot_slot(s.cache.checkpoint(c).unwrap().snapshot());
    assert_eq!(all.row, Some(slot));
    // Page 2 starts in the middle of b's record, and the row is optional.
    let middle = s.export_chain(c, 130, false).unwrap();
    assert_eq!(shape(&middle), (keys[1..].to_vec(), 1));
    assert_eq!(middle.row, None);
    assert_eq!(
        shape(&s.export_chain(c, 192, true).unwrap()),
        (keys[2..].to_vec(), 0)
    );
    // Starting at the endpoint leaves only its row, which must then be wanted.
    let row_only = s.export_chain(c, 256, true).unwrap();
    assert_eq!((row_only.nodes.len(), row_only.skip_pages), (1, 1));
    assert!(s.export_chain(c, 256, false).is_none());
    assert!(s.export_chain(c, 320, true).is_none());
}

#[test]
fn a_checkpoint_that_holds_no_page_is_left_out_of_the_export_chain() {
    let mut s = scheduler(2, 16, 4);
    let h = history(4);
    let a = publish(&mut s, &h, None, 64, &[0], 0);
    // Ends in the page `a` ended in, so it holds none.
    let mut inside = PrefixHistory::new([9; 32]);
    inside.append(&(0..70).collect::<Vec<_>>(), &[]).unwrap();
    let hollow = publish(&mut s, &inside, Some(a), 70, &[], 0);
    let c = publish(&mut s, &h, Some(hollow), 192, &[1, 2], 0);
    let chain = s.export_chain(c, 0, true).unwrap();
    assert_eq!(
        chain.nodes.iter().map(|n| n.key).collect::<Vec<_>>(),
        [h.key_at(64).unwrap(), h.key_at(192).unwrap()]
    );
    // The endpoint is kept even with no page of its own: its row is the claim.
    let row_only = s.export_chain(hollow, 64, true).unwrap();
    assert_eq!(row_only.nodes.len(), 1);
    assert_eq!(row_only.nodes[0].key, inside.key_at(70).unwrap());
}

/// Builds the scarce-page tree used by the headroom tests: a chain `a` -> `b`
/// and a sibling `c` of `b`, on a 6-page server with the rest of the pages free.
fn branched() -> (RequestScheduler, [CheckpointId; 3]) {
    let mut s = scheduler(2, 6, 4);
    let mut long = PrefixHistory::new([9; 32]);
    long.append(&(0..192).collect::<Vec<_>>(), &[]).unwrap();
    let mut other = PrefixHistory::new([9; 32]);
    other
        .append(&(0..64).chain(500..628).collect::<Vec<_>>(), &[])
        .unwrap();
    let a = publish(&mut s, &long, None, 64, &[0], 0);
    let b = publish(&mut s, &long, Some(a), 128, &[1], 0);
    let c = publish(&mut s, &other, Some(a), 128, &[2], 0);
    (s, [a, b, c])
}

fn next(pages: usize) -> NextAdmission {
    // A non-growable request reserves ceil((prompt + new) / 64) pages.
    NextAdmission {
        prompt_tokens: pages * 64 - 1,
        max_new_tokens: 1,
        growable: false,
    }
}

/// The oracle: really pin, then ask admission's own `ensure_resources`.
fn admission_agrees(build: impl Fn() -> (RequestScheduler, CheckpointId), pages: usize) {
    let (s, id) = build();
    let predicted = s.export_fits(id, Some(next(pages)));
    let (mut s, id) = build();
    s.cache.pin(id).unwrap();
    let actual = s.ensure_resources(pages, 1).is_ok();
    assert_eq!(predicted, actual, "{pages} pages");
}

#[test]
fn headroom_counts_siblings_that_stay_reclaimable_and_agrees_with_admission() {
    // Three pages are free; the sibling's page is reclaimable, the chain's are not.
    let build = || {
        let (s, [_, b, _]) = branched();
        (s, b)
    };
    let (s, b) = build();
    assert!(s.export_fits(b, Some(next(4))));
    assert!(!s.export_fits(b, Some(next(5))));
    assert!(s.export_fits(b, None));
    for pages in 1..=6 {
        admission_agrees(build, pages);
    }
}

#[test]
fn headroom_gives_back_compactable_snapshots_but_not_pinned_ones() {
    // The fourth snapshot id is taken too, so a restore ticket needs a reclaimed one.
    let build = |pin_sibling: bool| {
        move || {
            let (mut s, [_, b, c]) = branched();
            s.reserve_snapshot(0).unwrap();
            if pin_sibling {
                s.cache.pin(c).unwrap();
            }
            (s, b)
        }
    };
    let (s, b) = build(false)();
    assert_eq!(s.free_snapshot_count(), 0);
    // The sibling leaf is reclaimed, which lets the root compact.
    assert!(s.export_fits(b, Some(next(1))));
    admission_agrees(build(false), 1);
    // A running request on the sibling protects it and the root: no snapshot is left.
    let (s, b) = build(true)();
    assert!(!s.export_fits(b, Some(next(1))));
    assert!(s.export_fits(b, None));
    admission_agrees(build(true), 1);
}

#[test]
fn headroom_honours_snapshot_aliases_and_agrees_with_admission() {
    // Snapshot 0 and 1 overlap on pages 10..12, so releasing 0 frees only 8..10.
    let build = || {
        let mut s = scheduler(2, 16, 2);
        s.share_snapshot_pages(&[8..12, 10..14]).unwrap();
        let h = history(2);
        let a = publish(&mut s, &h, None, 64, &[0], 0);
        let b = publish(&mut s, &h, Some(a), 128, &[1], 0);
        (s, b)
    };
    let (s, b) = build();
    assert_eq!(s.free_page_count(), 8);
    for pages in 6..=10 {
        admission_agrees(build, pages);
    }
    assert!(s.export_fits(b, Some(next(8))));
    assert!(!s.export_fits(b, Some(next(9))));
}

#[test]
fn headroom_reserves_the_next_decode_step_of_running_requests() {
    let build = |free: usize| {
        let mut s = scheduler(2, 8, 4);
        let mut spec = spec(60, 400);
        spec.growable_output = true;
        let handle = s.admit(spec).unwrap();
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.scheduled_consumed = 127;
        let h = history(1);
        let a = publish(&mut s, &h, None, 64, &[7], 0);
        // The request reserved two pages; one more is needed for its next rounds.
        let spare: Vec<_> = s.free_pages.iter().copied().collect();
        for page in spare.into_iter().skip(free) {
            s.free_pages.remove(&page);
        }
        (s, a)
    };
    let (s, a) = build(1);
    assert!(s.export_fits(a, None));
    let (s, a) = build(0);
    assert!(!s.export_fits(a, None));
}

#[test]
fn a_key_republished_under_a_new_checkpoint_id_can_be_pinned() {
    let mut s = scheduler(2, 16, 4);
    let h = history(1);
    let old = publish(&mut s, &h, None, 64, &[0], 0);
    let key = s.cache.checkpoint(old).unwrap().prefix();
    s.evict(EvictionTarget {
        pages: 1,
        snapshots: 1,
    });
    let update = s.cache.advance_completed(Fence(0)).unwrap();
    s.apply_completion(update);
    let current = publish(&mut s, &h, None, 64, &[0], 0);
    assert_ne!(old, current);
    assert_eq!(s.checkpoint_by_key(&key), Ok(current));
    s.pin_checkpoint(current).unwrap();
    assert!(s.export_chain(current, 0, true).is_some());
    s.unpin_checkpoint(current).unwrap();
}
