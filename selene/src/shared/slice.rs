//! Rank slices of windows.
//!
//! A window holds whole page records, then optionally the row record. Within a page record
//! the ranks follow each other, each rank's planes padded to a block (the record layout of
//! [`crate::geometry::Layout`]), so one rank's slice of one page is contiguous, and the
//! slices of one rank in a window are `pages` strided pieces plus one row piece. When the
//! ranks of an instance are separate processes, each reads and writes only its own slices of
//! the manager's windows; the records at rest are the ordinary composite records.
//!
//! [`SliceLayout`] is the one place that arithmetic lives, for the manager and the instance.

use super::identity::Geometry;
use crate::{Error, Result, geometry::BLOCK_BYTES, record::checksum};

/// Where one rank's bytes lie in a page record and in the row record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankSlice {
    pub page_offset: u64,
    pub page_len: u64,
    pub row_offset: u64,
    pub row_len: u64,
}

/// Transfer checksums of one rank's slices of a window: one xxh3 per page and one for the
/// row when the window has it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SliceSums {
    pub pages: Vec<u64>,
    pub row: Option<u64>,
}

/// The slices of every rank of a geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceLayout {
    ranks: Vec<RankSlice>,
    page_record: u64,
    row_record: u64,
}

fn padded(bytes: u64) -> u64 {
    bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES
}

impl SliceLayout {
    pub fn new(geometry: &Geometry) -> Result<Self> {
        if geometry.ranks.is_empty() {
            return Err(Error::new("a geometry has at least one rank"));
        }
        let (mut page, mut row) = (0u64, 0u64);
        let mut ranks = Vec::with_capacity(geometry.ranks.len());
        for rank in &geometry.ranks {
            let page_len: u64 = rank.plane_bytes.iter().map(|b| padded(*b)).sum();
            let row_len = padded(rank.row_bytes);
            if page_len == 0 || row_len == 0 {
                return Err(Error::new("a rank has an empty plane or row"));
            }
            ranks.push(RankSlice {
                page_offset: page,
                page_len,
                row_offset: row,
                row_len,
            });
            page += page_len;
            row += row_len;
        }
        if page != geometry.page_record_bytes || row != geometry.row_record_bytes {
            return Err(Error::new(
                "the geometry's record sizes disagree with its ranks",
            ));
        }
        Ok(Self {
            ranks,
            page_record: page,
            row_record: row,
        })
    }

    /// The same record transferred by one process as one rank.
    pub fn whole(&self) -> Self {
        Self {
            ranks: vec![RankSlice {
                page_offset: 0,
                page_len: self.page_record,
                row_offset: 0,
                row_len: self.row_record,
            }],
            page_record: self.page_record,
            row_record: self.row_record,
        }
    }

    /// Ranks of the instance.
    pub fn world(&self) -> usize {
        self.ranks.len()
    }

    pub fn rank(&self, rank: usize) -> &RankSlice {
        &self.ranks[rank]
    }

    pub fn page_record_bytes(&self) -> u64 {
        self.page_record
    }

    pub fn row_record_bytes(&self) -> u64 {
        self.row_record
    }

    /// Bytes of a whole window of `pages` page records and the row if `row`.
    pub fn window_bytes(&self, pages: u32, row: bool) -> u64 {
        u64::from(pages) * self.page_record + if row { self.row_record } else { 0 }
    }

    /// Bytes of one rank's slices of such a window, packed back to back (its pages in order,
    /// then its row): the length of that rank's data-channel shape.
    pub fn slice_bytes(&self, rank: usize, pages: u32, row: bool) -> u64 {
        let r = &self.ranks[rank];
        u64::from(pages) * r.page_len + if row { r.row_len } else { 0 }
    }

    /// Offset inside the window of rank `rank`'s slice of page `page`.
    pub fn page_at(&self, rank: usize, page: u32) -> u64 {
        u64::from(page) * self.page_record + self.ranks[rank].page_offset
    }

    /// Offset inside a window of `pages` pages of rank `rank`'s slice of the row.
    pub fn row_at(&self, rank: usize, pages: u32) -> u64 {
        u64::from(pages) * self.page_record + self.ranks[rank].row_offset
    }

    fn check(&self, rank: usize, window: usize, pages: u32, row: bool) -> Result<()> {
        if rank >= self.ranks.len() {
            return Err(Error::new("no such rank"));
        }
        if window as u64 != self.window_bytes(pages, row) {
            return Err(Error::new("a window's length does not fit its pages"));
        }
        Ok(())
    }

    /// Copy rank `rank`'s slices out of `window` into `out`, packed: pages in order, then
    /// the row.
    pub fn gather(
        &self,
        rank: usize,
        window: &[u8],
        pages: u32,
        row: bool,
        out: &mut [u8],
    ) -> Result<()> {
        self.check(rank, window.len(), pages, row)?;
        if out.len() as u64 != self.slice_bytes(rank, pages, row) {
            return Err(Error::new("a packed slice's length does not fit its pages"));
        }
        let r = &self.ranks[rank];
        let mut at = 0usize;
        for page in 0..pages {
            let from = self.page_at(rank, page) as usize;
            out[at..at + r.page_len as usize]
                .copy_from_slice(&window[from..from + r.page_len as usize]);
            at += r.page_len as usize;
        }
        if row {
            let from = self.row_at(rank, pages) as usize;
            out[at..].copy_from_slice(&window[from..from + r.row_len as usize]);
        }
        Ok(())
    }

    /// The inverse of [`gather`](Self::gather): write packed bytes into rank `rank`'s slices.
    pub fn scatter(
        &self,
        rank: usize,
        window: &mut [u8],
        pages: u32,
        row: bool,
        packed: &[u8],
    ) -> Result<()> {
        self.check(rank, window.len(), pages, row)?;
        if packed.len() as u64 != self.slice_bytes(rank, pages, row) {
            return Err(Error::new("a packed slice's length does not fit its pages"));
        }
        let r = &self.ranks[rank];
        let mut at = 0usize;
        for page in 0..pages {
            let to = self.page_at(rank, page) as usize;
            window[to..to + r.page_len as usize]
                .copy_from_slice(&packed[at..at + r.page_len as usize]);
            at += r.page_len as usize;
        }
        if row {
            let to = self.row_at(rank, pages) as usize;
            window[to..to + r.row_len as usize].copy_from_slice(&packed[at..]);
        }
        Ok(())
    }

    /// Transfer checksums of rank `rank`'s slice of one page of `window`.
    pub fn page_sum(&self, rank: usize, window: &[u8], page: u32) -> u64 {
        let from = self.page_at(rank, page) as usize;
        checksum(&window[from..from + self.ranks[rank].page_len as usize])
    }

    /// Transfer checksum of rank `rank`'s slice of the row of a window of `pages` pages.
    pub fn row_sum(&self, rank: usize, window: &[u8], pages: u32) -> u64 {
        let from = self.row_at(rank, pages) as usize;
        checksum(&window[from..from + self.ranks[rank].row_len as usize])
    }

    /// All of rank `rank`'s transfer checksums for `window`.
    pub fn slice_sums(
        &self,
        rank: usize,
        window: &[u8],
        pages: u32,
        row: bool,
    ) -> Result<SliceSums> {
        self.check(rank, window.len(), pages, row)?;
        Ok(SliceSums {
            pages: (0..pages).map(|p| self.page_sum(rank, window, p)).collect(),
            row: row.then(|| self.row_sum(rank, window, pages)),
        })
    }

    /// Checksums of a channel window holding one rank's packed records.
    pub fn packed_sums(
        &self,
        rank: usize,
        window: &[u8],
        pages: u32,
        row: bool,
    ) -> Result<SliceSums> {
        let mine = self.rank(rank);
        if window.len() as u64 != self.slice_bytes(rank, pages, row) {
            return Err(Error::new("a window's length does not fit its slices"));
        }
        let stride = mine.page_len as usize;
        Ok(SliceSums {
            pages: (0..pages as usize)
                .map(|p| checksum(&window[p * stride..(p + 1) * stride]))
                .collect(),
            row: row.then(|| checksum(&window[pages as usize * stride..])),
        })
    }

    /// The stored checksums of `window`: one xxh3 per whole page record, and one over the
    /// whole row record.
    pub fn whole_sums(&self, window: &[u8], pages: u32, row: bool) -> Result<SliceSums> {
        if window.len() as u64 != self.window_bytes(pages, row) {
            return Err(Error::new("a window's length does not fit its pages"));
        }
        let page = self.page_record as usize;
        Ok(SliceSums {
            pages: (0..pages as usize)
                .map(|p| checksum(&window[p * page..(p + 1) * page]))
                .collect(),
            row: row.then(|| checksum(&window[pages as usize * page..])),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::identity::RankPlanes;

    fn geometry(world: usize) -> Geometry {
        let rank = RankPlanes {
            plane_bytes: vec![4096, 5000],
            row_bytes: 4096,
        };
        Geometry {
            page_tokens: 64,
            ranks: vec![rank; world],
            page_record_bytes: world as u64 * 12288,
            row_record_bytes: world as u64 * 4096,
        }
    }

    fn window(layout: &SliceLayout, pages: u32, row: bool) -> Vec<u8> {
        // Every byte says which rank, page and offset it belongs to.
        let mut bytes = vec![0u8; layout.window_bytes(pages, row) as usize];
        for rank in 0..layout.world() {
            for page in 0..pages {
                let at = layout.page_at(rank, page) as usize;
                for i in 0..layout.rank(rank).page_len as usize {
                    bytes[at + i] = (rank * 31 + page as usize * 7 + i) as u8;
                }
            }
            if row {
                let at = layout.row_at(rank, pages) as usize;
                for i in 0..layout.rank(rank).row_len as usize {
                    bytes[at + i] = (rank * 13 + 200 + i) as u8;
                }
            }
        }
        bytes
    }

    #[test]
    fn slices_tile_the_record_and_pad_to_blocks() {
        for world in 1..=8 {
            let layout = SliceLayout::new(&geometry(world)).unwrap();
            assert_eq!(layout.world(), world);
            let mut next = 0;
            for rank in 0..world {
                let r = layout.rank(rank);
                assert_eq!(r.page_offset, next);
                assert_eq!(r.page_len, 12288);
                assert_eq!(r.row_len, 4096);
                next += r.page_len;
            }
            assert_eq!(next, layout.page_record_bytes());
            assert_eq!(layout.slice_bytes(0, 3, true), 3 * 12288 + 4096);
            assert_eq!(layout.slice_bytes(0, 0, true), 4096);
        }
    }

    #[test]
    fn mismatched_sizes_are_refused() {
        let mut g = geometry(2);
        g.page_record_bytes += 4096;
        assert!(SliceLayout::new(&g).is_err());
        let mut g = geometry(2);
        g.ranks.clear();
        assert!(SliceLayout::new(&g).is_err());
    }

    #[test]
    fn gather_and_scatter_round_trip_every_rank_without_touching_the_others() {
        for world in [1usize, 2, 3, 4, 8] {
            for (pages, row) in [(0u32, true), (1, false), (3, true), (5, false)] {
                let layout = SliceLayout::new(&geometry(world)).unwrap();
                let source = window(&layout, pages, row);
                let mut rebuilt = vec![0xEEu8; source.len()];
                for rank in 0..world {
                    let mut packed = vec![0u8; layout.slice_bytes(rank, pages, row) as usize];
                    layout
                        .gather(rank, &source, pages, row, &mut packed)
                        .unwrap();
                    // Scatter into a window where every other rank's bytes are a sentinel.
                    let mut one = vec![0xEEu8; source.len()];
                    layout.scatter(rank, &mut one, pages, row, &packed).unwrap();
                    for other in 0..world {
                        for page in 0..pages {
                            let at = layout.page_at(other, page) as usize;
                            let len = layout.rank(other).page_len as usize;
                            let expect = if other == rank { &source } else { &one };
                            if other == rank {
                                assert_eq!(&one[at..at + len], &expect[at..at + len]);
                            } else {
                                assert!(one[at..at + len].iter().all(|b| *b == 0xEE));
                            }
                        }
                    }
                    layout
                        .scatter(rank, &mut rebuilt, pages, row, &packed)
                        .unwrap();
                }
                assert_eq!(rebuilt, source, "world {world} pages {pages} row {row}");
            }
        }
    }

    #[test]
    fn slice_and_whole_sums_cover_what_they_name() {
        let layout = SliceLayout::new(&geometry(3)).unwrap();
        let mut bytes = window(&layout, 2, true);
        let whole = layout.whole_sums(&bytes, 2, true).unwrap();
        let slices: Vec<_> = (0..3)
            .map(|r| layout.slice_sums(r, &bytes, 2, true).unwrap())
            .collect();
        // One flipped byte in rank 1's slice of page 1 changes that slice's sum, the
        // page's whole sum, and nothing else.
        bytes[layout.page_at(1, 1) as usize + 5] ^= 1;
        let whole2 = layout.whole_sums(&bytes, 2, true).unwrap();
        assert_eq!(whole.pages[0], whole2.pages[0]);
        assert_ne!(whole.pages[1], whole2.pages[1]);
        assert_eq!(whole.row, whole2.row);
        for (r, before) in slices.iter().enumerate() {
            let now = layout.slice_sums(r, &bytes, 2, true).unwrap();
            if r == 1 {
                assert_eq!(before.pages[0], now.pages[0]);
                assert_ne!(before.pages[1], now.pages[1]);
            } else {
                assert_eq!(*before, now);
            }
        }
        assert!(layout.slice_sums(3, &bytes, 2, true).is_err());
        assert!(layout.slice_sums(0, &bytes, 1, true).is_err());
    }
}
