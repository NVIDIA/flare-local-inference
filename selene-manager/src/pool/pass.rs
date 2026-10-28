//! The window assembly pass: the one place the manager hashes payload for multi-process
//! instances.
//!
//! A multi-process instance's ranks move only their own slices of a window, each with a
//! transfer checksum; records at rest keep one checksum per whole page and one for the
//! row. When a window is assembled from rank slices, or is handed on to rank slices, the
//! manager verifies what the sender stated and computes what the receiver needs in one
//! pass over the window, page by page, on several cores, off the pool actor.

use crate::regions::LocalRegions;
use flare_selene::shared::region::Region;
use flare_selene::shared::{SliceLayout, SliceSums};
use std::{sync::Arc, thread};

/// Windows smaller than this are hashed on the calling thread.
const PARALLEL_BYTES: usize = 512 * 1024;
const MAX_THREADS: usize = 8;

/// What to check and what to produce for one window.
pub struct PassJob {
    pub regions: Arc<LocalRegions>,
    pub region: Region,
    pub offset: u64,
    pub pages: u32,
    pub row: bool,
    pub layout: Arc<SliceLayout>,
    /// The slice transfer checksums the source's ranks stated, by rank (a multi-process
    /// source). Every rank must have stated them.
    pub stated_slices: Option<Vec<SliceSums>>,
    /// The whole-page and row checksums a single-process source or the store stated; checked.
    pub stated_whole: Option<SliceSums>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct PassOut {
    pub whole: SliceSums,
    /// By rank.
    pub slices: Vec<SliceSums>,
}

/// Why a pass failed. A mismatch is the sender's bytes disagreeing with its checksums.
#[derive(Debug, PartialEq, Eq)]
pub enum PassError {
    Mismatch(String),
    Io(String),
}

fn compute(job: &PassJob, window: &[u8]) -> Result<PassOut, PassError> {
    let layout = &*job.layout;
    let pages = job.pages as usize;
    let world = layout.world();
    if let Some(stated) = &job.stated_slices {
        let ok = stated.len() == world
            && stated
                .iter()
                .all(|s| s.pages.len() == pages && s.row.is_some() == job.row);
        if !ok {
            return Err(PassError::Mismatch(
                "the source's slice checksums do not fit the window".into(),
            ));
        }
    }
    if let Some(stated) = &job.stated_whole
        && (stated.pages.len() != pages || stated.row.is_some() != job.row)
    {
        return Err(PassError::Mismatch(
            "the stated checksums do not fit the window".into(),
        ));
    }
    let threads = if window.len() < PARALLEL_BYTES || pages < 2 {
        1
    } else {
        thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(MAX_THREADS)
            .min(pages)
    };
    // Per page: (whole, per-rank slice sums).
    let per_page = |page: u32| -> (u64, Vec<u64>) {
        let whole = {
            let record = layout.page_record_bytes() as usize;
            flare_selene::shared::record::checksum(
                &window[page as usize * record..(page as usize + 1) * record],
            )
        };
        let slices = (0..world)
            .map(|r| layout.page_sum(r, window, page))
            .collect();
        (whole, slices)
    };
    let mut results: Vec<(u64, Vec<u64>)> = Vec::with_capacity(pages);
    if threads <= 1 {
        results.extend((0..job.pages).map(per_page));
    } else {
        let chunk = pages.div_ceil(threads);
        let mut parts: Vec<Vec<(u64, Vec<u64>)>> = Vec::new();
        thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (from, to) = (t * chunk, ((t + 1) * chunk).min(pages));
                    let per_page = &per_page;
                    scope.spawn(move || (from as u32..to as u32).map(per_page).collect::<Vec<_>>())
                })
                .collect();
            for handle in handles {
                parts.push(handle.join().expect("a hashing thread panicked"));
            }
        });
        results.extend(parts.into_iter().flatten());
    }
    let row_whole = job.row.then(|| {
        flare_selene::shared::record::checksum(
            &window[pages * layout.page_record_bytes() as usize..],
        )
    });
    let row_slices: Vec<u64> = if job.row {
        (0..world)
            .map(|r| layout.row_sum(r, window, job.pages))
            .collect()
    } else {
        Vec::new()
    };

    if let Some(stated) = &job.stated_slices {
        for (rank, said) in stated.iter().enumerate() {
            for (page, sum) in said.pages.iter().enumerate() {
                if results[page].1[rank] != *sum {
                    return Err(PassError::Mismatch(format!(
                        "rank {rank}'s slice of page {page} failed its checksum"
                    )));
                }
            }
            if job.row && said.row != Some(row_slices[rank]) {
                return Err(PassError::Mismatch(format!(
                    "rank {rank}'s slice of the row failed its checksum"
                )));
            }
        }
    }
    if let Some(stated) = &job.stated_whole {
        for (page, sum) in stated.pages.iter().enumerate() {
            if results[page].0 != *sum {
                return Err(PassError::Mismatch(format!(
                    "page {page} failed its checksum"
                )));
            }
        }
        if job.row && stated.row != row_whole {
            return Err(PassError::Mismatch("the row failed its checksum".into()));
        }
    }
    let whole = SliceSums {
        pages: results.iter().map(|r| r.0).collect(),
        row: row_whole,
    };
    let slices = (0..world)
        .map(|rank| SliceSums {
            pages: results.iter().map(|r| r.1[rank]).collect(),
            row: job.row.then(|| row_slices[rank]),
        })
        .collect();
    Ok(PassOut { whole, slices })
}

/// Run the pass. Blocks: call it from a blocking thread.
pub fn run(job: PassJob) -> Result<PassOut, PassError> {
    let length = job.layout.window_bytes(job.pages, job.row);
    // SAFETY: the pool retains this window unchanged until the pass completes.
    let window = unsafe { job.regions.bytes(job.region, job.offset, length) }
        .map_err(|error| PassError::Io(error.to_string()))?;
    compute(&job, window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use flare_selene::shared::{Geometry, RankPlanes};

    fn layout(world: usize) -> Arc<SliceLayout> {
        let rank = RankPlanes {
            plane_bytes: vec![4096, 8192],
            row_bytes: 4096,
        };
        Arc::new(
            SliceLayout::new(&Geometry {
                page_tokens: 64,
                ranks: vec![rank; world],
                page_record_bytes: world as u64 * 12288,
                row_record_bytes: world as u64 * 4096,
            })
            .unwrap(),
        )
    }

    struct Fixture {
        regions: Arc<LocalRegions>,
        layout: Arc<SliceLayout>,
        pages: u32,
        row: bool,
    }

    impl Fixture {
        fn new(world: usize, pages: u32, row: bool) -> Self {
            let regions = Arc::new(LocalRegions::new(
                Arena::new(1 << 20).unwrap(),
                Arena::new(8 << 20).unwrap(),
            ));
            let layout = layout(world);
            let length = layout.window_bytes(pages, row) as usize;
            let bytes: Vec<u8> = (0..length)
                .map(|i| (i.wrapping_mul(2654435761) >> 7) as u8)
                .collect();
            // SAFETY: the test owns this window for the borrow.
            unsafe {
                let w = regions.bytes(Region::Staging, 4096, length as u64).unwrap();
                w.copy_from_slice(&bytes);
            }
            Self {
                regions,
                layout,
                pages,
                row,
            }
        }

        fn job(&self) -> PassJob {
            PassJob {
                regions: self.regions.clone(),
                region: Region::Staging,
                offset: 4096,
                pages: self.pages,
                row: self.row,
                layout: self.layout.clone(),
                stated_slices: None,
                stated_whole: None,
            }
        }

        fn window(&self) -> Vec<u8> {
            // SAFETY: the test owns this window for the borrow.
            unsafe {
                self.regions
                    .bytes(
                        Region::Staging,
                        4096,
                        self.layout.window_bytes(self.pages, self.row),
                    )
                    .unwrap()
                    .to_vec()
            }
        }

        fn flip(&self, at: usize) {
            // SAFETY: the test owns this window for the borrow.
            unsafe {
                let w = self
                    .regions
                    .bytes(Region::Staging, 4096 + at as u64, 1)
                    .unwrap();
                w[0] ^= 0xFF;
            }
        }
    }

    #[test]
    fn a_single_process_source_into_a_multi_process_requester_verifies_whole_and_produces_slices() {
        for (world, pages, row) in [(2, 3, true), (4, 1, false), (3, 0, true), (2, 70, true)] {
            let f = Fixture::new(world, pages, row);
            let window = f.window();
            let whole = f.layout.whole_sums(&window, pages, row).unwrap();
            let mut job = f.job();
            job.stated_whole = Some(whole);
            let out = run(job).unwrap();
            let slices = out.slices;
            assert_eq!(out.whole, f.layout.whole_sums(&window, pages, row).unwrap());
            for (rank, got) in slices.iter().enumerate() {
                assert_eq!(
                    *got,
                    f.layout.slice_sums(rank, &window, pages, row).unwrap(),
                    "world {world} pages {pages}"
                );
            }
        }
    }

    #[test]
    fn a_multi_process_source_into_a_single_process_requester_verifies_slices_and_produces_whole() {
        let f = Fixture::new(3, 70, true);
        let window = f.window();
        let stated: Vec<_> = (0..3)
            .map(|r| f.layout.slice_sums(r, &window, 70, true).unwrap())
            .collect();
        let mut job = f.job();
        job.stated_slices = Some(stated);
        let out = run(job).unwrap();
        assert_eq!(out.whole, f.layout.whole_sums(&window, 70, true).unwrap());
        assert_eq!(out.slices.len(), 3);
    }

    #[test]
    fn multi_to_multi_verifies_slices_and_produces_both_checksum_forms() {
        let f = Fixture::new(2, 5, false);
        let window = f.window();
        let stated: Vec<_> = (0..2)
            .map(|r| f.layout.slice_sums(r, &window, 5, false).unwrap())
            .collect();
        let mut job = f.job();
        job.stated_slices = Some(stated.clone());
        let out = run(job).unwrap();
        assert_eq!(out.slices, stated);
        assert_eq!(out.whole, f.layout.whole_sums(&window, 5, false).unwrap());
    }

    #[test]
    fn one_flipped_byte_is_a_mismatch_naming_what_failed() {
        let f = Fixture::new(2, 4, true);
        let window = f.window();
        let whole = f.layout.whole_sums(&window, 4, true).unwrap();
        let stated: Vec<_> = (0..2)
            .map(|r| f.layout.slice_sums(r, &window, 4, true).unwrap())
            .collect();
        // Rank 1's slice of page 2.
        let at = f.layout.page_at(1, 2) as usize + 17;
        f.flip(at);
        let mut job = f.job();
        job.stated_slices = Some(stated);
        let Err(PassError::Mismatch(why)) = run(job) else {
            panic!("expected a mismatch");
        };
        assert!(why.contains("rank 1") && why.contains("page 2"), "{why}");
        let mut job = f.job();
        job.stated_whole = Some(whole.clone());
        let Err(PassError::Mismatch(why)) = run(job) else {
            panic!("expected a mismatch");
        };
        assert!(why.contains("page 2"), "{why}");
        // The row.
        f.flip(at);
        let row_at = f.layout.row_at(0, 4) as usize + 3;
        f.flip(row_at);
        let mut job = f.job();
        job.stated_whole = Some(whole);
        let Err(PassError::Mismatch(why)) = run(job) else {
            panic!("expected a mismatch");
        };
        assert!(why.contains("row"), "{why}");
    }

    #[test]
    fn sums_that_do_not_fit_the_window_are_refused_and_a_bad_range_is_an_io_error() {
        let f = Fixture::new(2, 2, true);
        let mut job = f.job();
        job.stated_whole = Some(SliceSums {
            pages: vec![1],
            row: Some(2),
        });
        assert!(matches!(run(job), Err(PassError::Mismatch(_))));
        let mut job = f.job();
        job.stated_slices = Some(vec![SliceSums::default()]);
        assert!(matches!(run(job), Err(PassError::Mismatch(_))));
        let mut job = f.job();
        job.offset = 64 << 20;
        assert!(matches!(run(job), Err(PassError::Io(_))));
    }

    #[test]
    fn the_parallel_result_equals_the_serial_one() {
        // Large enough to be split across threads.
        let f = Fixture::new(2, 64, true);
        let job = f.job();
        let out = run(job).unwrap();
        let window = f.window();
        assert_eq!(out.whole, f.layout.whole_sums(&window, 64, true).unwrap());
        let slices = out.slices;
        for (rank, got) in slices.iter().enumerate() {
            assert_eq!(*got, f.layout.slice_sums(rank, &window, 64, true).unwrap());
        }
    }
}
