//! Mutually exclusive ownership of snapshot aliases and KV page ids.

use super::*;

impl RequestScheduler {
    pub fn share_snapshot_pages(&mut self, ranges: &[std::ops::Range<u32>]) -> Result<()> {
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || ranges.len() != self.limits.snapshots as usize
            || ranges
                .iter()
                .any(|r| r.start > r.end || r.end > self.limits.pages)
        {
            return Err(
                "snapshot aliases must cover valid KV ids and be configured before admission"
                    .into(),
            );
        }
        self.snapshot_pages = ranges.to_vec();
        Ok(())
    }

    pub(super) fn available_snapshot(&self, keep_pages: usize) -> Option<SnapshotId> {
        self.free_snapshots.iter().copied().find(|id| {
            let Some(range) = self.snapshot_pages.get(id.0 as usize) else {
                return true;
            };
            let mut cost = 0;
            for page in range.clone() {
                if self.page_snapshot_refs[page as usize] == 0 {
                    if !self.free_pages.contains(&PageId(page)) {
                        return false;
                    }
                    cost += 1;
                }
            }
            self.free_pages.len().saturating_sub(cost) >= keep_pages
        })
    }

    pub(super) fn reserve_snapshot(&mut self, keep_pages: usize) -> Option<SnapshotId> {
        let id = self.available_snapshot(keep_pages)?;
        self.free_snapshots.remove(&id);
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                self.free_pages.remove(&PageId(page));
                if self.page_snapshot_refs[page as usize] == 0 {
                    self.stats.snapshot_backing_pages += 1;
                }
                self.page_snapshot_refs[page as usize] += 1;
            }
        }
        Some(id)
    }

    pub(super) fn retain_snapshot(&mut self) -> Result<Option<SnapshotId>> {
        loop {
            if let Some(id) = self.reserve_snapshot(0) {
                return Ok(Some(id));
            }
            let evicted = self.evict(EvictionTarget {
                pages: 0,
                snapshots: 1,
            });
            self.stats.snapshot_evictions += evicted.retired.len() as u64;
            let update = self.cache.advance_completed(self.completed)?;
            self.apply_completion(update);
            if evicted.retired.is_empty() {
                return Ok(None);
            }
        }
    }

    /// Backing that is free or becomes free once every retired owner and `extra` reaches
    /// its fence, with the snapshots those owners release.
    pub(super) fn pending_backing(&self, extra: &Reclaimable) -> (Vec<bool>, Vec<SnapshotId>) {
        let mut available = vec![false; self.limits.pages as usize];
        let mut retiring_refs = vec![0; available.len()];
        let mut snapshots = extra.snapshots.clone();
        for page in self.free_pages.iter().chain(&extra.pages) {
            available[page.0 as usize] = true;
        }
        for retirement in self.cache.pending_retirement() {
            for page in &retirement.pages {
                available[page.0 as usize] = true;
            }
            snapshots.push(retirement.snapshot);
        }
        for retirement in &self.retired {
            for page in &retirement.pages {
                available[page.0 as usize] = true;
            }
            snapshots.extend(retirement.snapshots.iter().flatten().copied());
        }
        for snapshot in &snapshots {
            if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                for page in range.clone() {
                    retiring_refs[page as usize] += 1;
                }
            }
        }
        for (page, refs) in retiring_refs.into_iter().enumerate() {
            if refs > 0 && refs == self.page_snapshot_refs[page] {
                available[page] = true;
            }
        }
        (available, snapshots)
    }

    /// Project free backing after every already-retired owner reaches its fence.
    pub(super) fn pending_page_count(&self) -> usize {
        let (available, _) = self.pending_backing(&Reclaimable::default());
        available.into_iter().filter(|free| *free).count()
    }

    pub(super) fn pending_resources(&self, pages: usize, snapshots: usize) -> bool {
        if self.cache.pending_retirement().is_empty() && self.retired.is_empty() {
            return false;
        }
        let (backing, retiring) = self.pending_backing(&Reclaimable::default());
        self.projected_fits(&backing, &retiring, pages, snapshots)
    }

    /// Whether `pages` pages and `snapshots` snapshot ids are obtainable once `backing` is
    /// free and the `retiring` snapshots are released, honouring snapshot aliases.
    pub(super) fn projected_fits(
        &self,
        backing: &[bool],
        retiring: &[SnapshotId],
        pages: usize,
        snapshots: usize,
    ) -> bool {
        let mut available = 0;
        for free in backing {
            available += usize::from(*free);
        }
        if available < pages {
            return false;
        }
        if snapshots == 0 {
            return true;
        }
        let mut eligible = 0;
        for snapshot in self.free_snapshots.iter().chain(retiring) {
            let mut cost = 0;
            let mut usable = true;
            if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                for page in range.clone() {
                    if backing[page as usize] {
                        cost += 1;
                    } else if self.page_snapshot_refs[page as usize] == 0 {
                        usable = false;
                        break;
                    }
                }
            }
            if usable && available.saturating_sub(cost) >= pages {
                eligible += 1;
                if eligible == snapshots {
                    return true;
                }
            }
        }
        false
    }

    pub(super) fn release_snapshot(&mut self, id: SnapshotId) {
        assert!(self.free_snapshots.insert(id), "snapshot released twice");
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                let count = &mut self.page_snapshot_refs[page as usize];
                *count = count.checked_sub(1).expect("snapshot alias ownership");
                if *count == 0 {
                    self.stats.snapshot_backing_pages -= 1;
                    assert!(self.free_pages.insert(PageId(page)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_in_execution_alignment_padding_do_not_exclude_pages() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[0..0]).unwrap();
        let snapshot = s.reserve_snapshot(16).unwrap();
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        s.release_snapshot(snapshot);
        assert_eq!(s.free_page_count(), 16);
    }

    #[test]
    fn overlapping_snapshot_aliases_release_pages_only_after_the_last_owner() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[8..12, 10..14]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        assert_eq!(s.free_page_count(), 10);
        assert_eq!(s.stats().snapshot_backing_pages, 6);
        s.release_snapshot(a);
        assert_eq!(s.free_page_count(), 12);
        assert_eq!(s.stats().snapshot_backing_pages, 4);
        assert!(!s.free_pages.contains(&PageId(10)));
        s.release_snapshot(b);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
    }

    #[test]
    fn request_page_reservations_exclude_snapshot_aliases() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[0..8, 8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64 * 12, 1)).unwrap();
        assert_eq!(s.free_snapshot_count(), 2);
        assert!(s.reserve_snapshot(0).is_none());
        s.cancel(handle).unwrap();
        assert!(s.reserve_snapshot(8).is_some());
        assert!(s.reserve_snapshot(1).is_none());
    }

    #[test]
    fn cancellation_keeps_snapshot_backing_until_the_last_gpu_fence() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64, 1)).unwrap();
        let snapshot = s.reserve_snapshot(0).unwrap();
        s.request_mut(handle).unwrap().terminal_snapshot = Some(snapshot);
        let step = s.plan().unwrap();
        s.commit(&step, Fence(1)).unwrap();
        s.cancel(handle).unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 8);
        assert!(s.reserve_snapshot(0).is_none());
        s.complete(
            Fence(1),
            &[CompletionRecord::default(); REQUEST_LANES],
            Default::default(),
        )
        .unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.free_snapshot_count(), 1);
    }
    #[test]
    fn page_pressure_compacts_aliases_with_free_snapshot_ids_and_preserves_fences() {
        for delayed in [false, true] {
            let mut s = super::super::tests::scheduler(2, 16, 4);
            s.share_snapshot_pages(&[8..12, 10..14, 0..0, 0..0])
                .unwrap();
            let mut history = PrefixHistory::new([9; 32]);
            history.append(&(0..128).collect::<Vec<_>>(), &[]).unwrap();
            let a = s.reserve_snapshot(0).unwrap();
            let b = s.reserve_snapshot(0).unwrap();
            let mut parent = None;
            for (endpoint, page, snapshot) in [(64, 0, a), (128, 1, b)] {
                s.free_pages.remove(&PageId(page));
                parent = Some(
                    s.cache
                        .publish(
                            &history,
                            &Publication {
                                parent,
                                consumed_tokens: endpoint,
                                tail_pages: vec![PageId(page)],
                                snapshot: StateSnapshot {
                                    id: snapshot,
                                    consumed_tokens: endpoint,
                                },
                                ready_after: Fence(0),
                            },
                        )
                        .unwrap()
                        .id(),
                );
            }
            let child = parent.unwrap();
            s.cache.pin(child).unwrap();
            if delayed {
                s.cache.mark_used(child, Fence(7)).unwrap();
            }
            let held = std::mem::take(&mut s.free_pages);
            assert_eq!(s.free_snapshot_count(), 2);
            let mut admission = super::super::tests::spec(64, 1);
            admission.prompt.fill(99);
            let result = s.admit(admission);
            if delayed {
                assert!(result.is_err());
                assert_eq!(s.free_page_count(), 0);
                assert_eq!(s.pending_page_count(), 2);
                assert_eq!(s.stats().snapshot_backing_pages, 6);
                let update = s.cache.advance_completed(Fence(7)).unwrap();
                s.apply_completion(update);
            } else {
                s.cancel(result.unwrap()).unwrap();
            }
            assert_eq!(s.free_page_count(), 2);
            assert_eq!(s.stats().snapshot_backing_pages, 4);
            assert_eq!(s.page_snapshot_refs[10], 1);
            assert_eq!(s.cache.page_chain(child).unwrap(), [PageId(0), PageId(1)]);
            s.free_pages.extend(held);
        }
    }

    #[test]
    fn decoder_growth_waits_for_pending_alias_reclamation() {
        let mut s = super::super::tests::scheduler(1, 16, 4);
        s.share_snapshot_pages(&[8..12, 10..14, 0..0, 0..0])
            .unwrap();
        let mut spec = super::super::tests::spec(60, 400);
        spec.growable_output = true;
        let handle = s.admit(spec).unwrap();
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.scheduled_consumed = 127;
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&vec![7; 128], &[]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        let mut parent = None;
        for (endpoint, page, snapshot) in [(64, 2, a), (128, 3, b)] {
            s.free_pages.remove(&PageId(page));
            parent = Some(
                s.cache
                    .publish(
                        &history,
                        &Publication {
                            parent,
                            consumed_tokens: endpoint,
                            tail_pages: vec![PageId(page)],
                            snapshot: StateSnapshot {
                                id: snapshot,
                                consumed_tokens: endpoint,
                            },
                            ready_after: Fence(0),
                        },
                    )
                    .unwrap()
                    .id(),
            );
        }
        let child = parent.unwrap();
        s.cache.pin(child).unwrap();
        s.cache.mark_used(child, Fence(7)).unwrap();
        let held = std::mem::take(&mut s.free_pages);
        s.refill_reservations().unwrap();
        assert_eq!(s.stats().preemptions, 0);
        assert!(!s.fits_next_rounds(s.request(handle).unwrap()));
        assert_eq!(s.free_page_count(), 0);
        assert_eq!(s.pending_page_count(), 2);
        let update = s.cache.advance_completed(Fence(7)).unwrap();
        s.apply_completion(update);
        s.refill_reservations().unwrap();
        assert!(s.fits_next_rounds(s.request(handle).unwrap()));
        assert_eq!(s.stats().preemptions, 0);
        s.free_pages.extend(held);
    }
}
