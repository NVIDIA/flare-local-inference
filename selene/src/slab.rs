//! Registered host memory split into fixed-size page and row slabs.
//!
//! One anonymous mapping, 4 KiB aligned and advised for transparent huge pages,
//! holds every slab. A page slab holds one page record across all ranks and a
//! row slab holds one row record across all ranks, in the cross-rank
//! [`Layout`](super::Layout). Registration with the CUDA driver is a separate
//! step so CPU tests run without a driver. Allocation never blocks: when no
//! slab is free the caller evicts or fails its job.
//!
//! A pool split by rank instead keeps each rank's part of every slab in pinned
//! memory that only that rank's device maps, for stores whose records the CPU
//! never reads. Its slab addresses are identities, and [`RankPart`] locates
//! each rank's part of a slab for the transfer threads.
//!
//! Each sub-pool hands out slabs from a FIFO free list that starts in index
//! order, and tracks which slabs are allocated so that releasing a slab that is
//! free, released twice, or outside the pool is ignored instead of duplicating
//! a free-list entry. The free lists are sized at construction and never grow.

use std::collections::VecDeque;

use crate::{HostRegistrar, Registration, Result, SplitMapping};

use super::geometry::{BLOCK_BYTES, Layout, invalid};

/// Page slabs in the smallest retaining pool: one outbound and one inbound window.
pub const MINIMUM_PAGE_SLABS: usize = 128;
/// Row slabs in the smallest retaining pool.
pub const MINIMUM_ROW_SLABS: usize = 2;
/// Page slabs in a staging pool: two outbound and two inbound windows.
pub const STAGING_PAGE_SLABS: usize = 256;
/// Row slabs in a staging pool.
pub const STAGING_ROW_SLABS: usize = 4;
/// Pages per transfer window.
pub use crate::shared::WINDOW_PAGES;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageSlab(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RowSlab(pub u32);

struct SubPool {
    base: usize,
    slab_bytes: usize,
    count: usize,
    free: VecDeque<u32>,
    allocated: Vec<bool>,
}

impl SubPool {
    fn new(base: usize, slab_bytes: usize, count: usize) -> Self {
        let mut free = VecDeque::with_capacity(count);
        free.extend(0..count as u32);
        Self {
            base,
            slab_bytes,
            count,
            free,
            allocated: vec![false; count],
        }
    }

    /// Slab indices are produced only by `allocate`, so an index outside the
    /// pool is a caller defect; the panic keeps `page` and `row` inside the mapping.
    fn address(&self, index: u32) -> usize {
        assert!(
            (index as usize) < self.count,
            "slab index {index} outside a pool of {} slabs",
            self.count
        );
        self.base + index as usize * self.slab_bytes
    }

    fn allocate(&mut self) -> Option<u32> {
        let index = self.free.pop_front()?;
        self.allocated[index as usize] = true;
        Some(index)
    }

    /// Return `index` to the back of the free list. A slab that is not
    /// allocated, including one released twice or an index outside the pool,
    /// is ignored so the free list never holds a slab twice.
    fn release(&mut self, index: u32) {
        if let Some(allocated) = self.allocated.get_mut(index as usize)
            && *allocated
        {
            *allocated = false;
            self.free.push_back(index);
        }
    }
}

/// Bytes a region of `count` slabs of `slab_bytes` occupies. A zero count
/// occupies nothing and ignores the slab size; otherwise the size must be a
/// nonzero multiple of [`BLOCK_BYTES`] and the region must fit the address space.
fn region_bytes(kind: &str, slab_bytes: u64, count: usize) -> Result<usize> {
    if count == 0 {
        return Ok(0);
    }
    if slab_bytes == 0 || !slab_bytes.is_multiple_of(BLOCK_BYTES) {
        return Err(invalid(format!(
            "{kind} slab size of {slab_bytes} bytes is not a nonzero multiple of {BLOCK_BYTES}"
        )));
    }
    u32::try_from(count)
        .ok()
        .and_then(|_| usize::try_from(slab_bytes).ok()?.checked_mul(count))
        .ok_or_else(|| {
            invalid(format!(
                "{count} {kind} slabs of {slab_bytes} bytes exceed the addressable pool size"
            ))
        })
}

/// Device addresses of one rank's part of every slab.
///
/// A split pool keeps each rank's part of every page slab, then of every row
/// slab, contiguously in that rank's memory. Slab addresses remain identities:
/// where each slab would begin if the pool held whole cross-rank records. They
/// lie inside the pool's reserved range, so they never name another pool's slab.
/// A registered pool instead uses CPU slab addresses as identities and the
/// registration's device alias for copies, keeping cross-rank record strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPart {
    /// The pool's reserved range.
    range: std::ops::Range<usize>,
    pages: PartRegion,
    rows: PartRegion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PartRegion {
    /// Identity of slab 0; identities advance by one cross-rank record.
    identity: usize,
    record_bytes: usize,
    count: usize,
    /// This rank's part of slab 0; parts advance by the rank's share of a record.
    base: usize,
    part_bytes: usize,
}

impl PartRegion {
    fn locate(&self, slab: usize) -> Option<u64> {
        let offset = slab.checked_sub(self.identity)?;
        let index = offset / self.record_bytes;
        (offset.is_multiple_of(self.record_bytes) && index < self.count)
            .then(|| (self.base + index * self.part_bytes) as u64)
    }
}

impl RankPart {
    /// This rank's part of the page slab at `slab`, or `None` when `slab` lies
    /// outside the pool. An address inside the pool that is not one of its page
    /// slabs is rejected.
    pub fn page(&self, slab: usize) -> Result<Option<u64>> {
        self.locate(&self.pages, slab, "page")
    }

    /// This rank's part of the row slab at `slab`, as [`RankPart::page`] does for pages.
    pub fn row(&self, slab: usize) -> Result<Option<u64>> {
        self.locate(&self.rows, slab, "row")
    }

    fn locate(&self, region: &PartRegion, slab: usize, kind: &str) -> Result<Option<u64>> {
        if !self.range.contains(&slab) {
            return Ok(None);
        }
        region.locate(slab).map(Some).ok_or_else(|| {
            invalid(format!(
                "address {slab:#x} inside a slab pool is not one of its {kind} slabs"
            ))
        })
    }
}

/// Each rank's parts of a pool split by rank and the memory holding them.
struct Split {
    parts: Vec<RankPart>,
    _memory: Box<dyn SplitMapping>,
}

/// Fixed pools of page and row slabs inside one registered mapping, or split by rank.
pub struct SlabPool {
    mapping: usize,
    mapping_bytes: usize,
    pages: SubPool,
    rows: SubPool,
    retain: bool,
    registration: Option<Box<dyn Registration>>,
    split: Option<Split>,
}

// SAFETY: The mapping is plain process memory; the pool is used from one thread
// at a time and slab addresses handed to other threads outlive their transfers.
unsafe impl Send for SlabPool {}

impl SlabPool {
    /// Map `page_count` page slabs of `page_bytes` and `row_count` row slabs of
    /// `row_bytes`, each a multiple of [`BLOCK_BYTES`], populated and advised for
    /// transparent huge pages. Page slabs precede row slabs in the mapping.
    pub fn new(
        page_bytes: u64,
        page_count: usize,
        row_bytes: u64,
        row_count: usize,
        retain: bool,
    ) -> Result<Self> {
        let page_region = region_bytes("page", page_bytes, page_count)?;
        let row_region = region_bytes("row", row_bytes, row_count)?;
        let mapping_bytes = page_region.checked_add(row_region).ok_or_else(|| {
            invalid("page and row slabs together exceed the addressable pool size")
        })?;
        if mapping_bytes == 0 {
            return Err(invalid(
                "a slab pool requires at least one page or row slab",
            ));
        }
        let base = flare_eos::map_host(mapping_bytes)
            .map_err(|error| invalid(format!("allocate host slabs: {error}")))?;
        Ok(Self {
            mapping: base,
            mapping_bytes,
            pages: SubPool::new(base, page_bytes as usize, page_count),
            rows: SubPool::new(base + page_region, row_bytes as usize, row_count),
            retain,
            registration: None,
            split: None,
        })
    }

    /// A retaining pool of at most `bytes`: rows take one quarter and pages the
    /// rest, each at least its minimum window count, with rows yielding to the
    /// page minimum so the pool never exceeds `bytes`. A `bytes` below
    /// [`SlabPool::minimum_bytes`] fails with that minimum in the message.
    pub fn sized(layout: &Layout, bytes: u64) -> Result<Self> {
        let (page_count, row_count) = Self::sized_counts(layout, bytes)?;
        Self::new(
            layout.page_record_bytes,
            page_count,
            layout.row_record_bytes,
            row_count,
            true,
        )
    }

    /// A retaining pool with the slab counts of [`SlabPool::sized`], split by
    /// rank: each rank's part of every slab is pinned host memory that only the
    /// device of `contexts[rank]` and the host access. Registering one mapping
    /// maps it into every device at about 2 MiB of device memory per GiB; a
    /// split pool costs each device that for its own part only. The CPU cannot
    /// address a slab's record, so the pool suits stores that move records only
    /// between devices and slabs; [`SlabPool::rank_parts`] locates the parts.
    pub fn sized_by_rank(
        layout: &Layout,
        bytes: u64,
        contexts: &[(usize, i32)],
        registrar: &dyn HostRegistrar,
    ) -> Result<Self> {
        if contexts.len() != layout.ranks.len() {
            return Err(invalid(format!(
                "a pool split by rank needs one context per rank: {} contexts for {} ranks",
                contexts.len(),
                layout.ranks.len()
            )));
        }
        let (page_count, row_count) = Self::sized_counts(layout, bytes)?;
        let page_region = region_bytes("page", layout.page_record_bytes, page_count)?;
        let row_region = region_bytes("row", layout.row_record_bytes, row_count)?;
        let shares: Vec<(usize, usize)> = (0..layout.ranks.len())
            .map(|rank| {
                let next = layout.ranks.get(rank + 1);
                let this = &layout.ranks[rank];
                (
                    (next.map_or(layout.page_record_bytes, |next| next.page_offset)
                        - this.page_offset) as usize,
                    (next.map_or(layout.row_record_bytes, |next| next.row_offset) - this.row_offset)
                        as usize,
                )
            })
            .collect();
        let requests: Vec<_> = contexts
            .iter()
            .zip(&shares)
            .map(|(&(context, device), (page, row))| {
                (context, device, page * page_count + row * row_count)
            })
            .collect();
        let memory = registrar.map_split(&requests)?;
        let base = memory.address();
        if memory.nbytes() < page_region + row_region {
            return Err(invalid(
                "the reserved range of a pool split by rank is smaller than its records",
            ));
        }
        let range = base..base + memory.nbytes();
        let parts = shares
            .iter()
            .enumerate()
            .map(|(rank, &(page, row))| RankPart {
                range: range.clone(),
                pages: PartRegion {
                    identity: base,
                    record_bytes: layout.page_record_bytes as usize,
                    count: page_count,
                    base: memory.part_address(rank),
                    part_bytes: page,
                },
                rows: PartRegion {
                    identity: base + page_region,
                    record_bytes: layout.row_record_bytes as usize,
                    count: row_count,
                    base: memory.part_address(rank) + page * page_count,
                    part_bytes: row,
                },
            })
            .collect();
        Ok(Self {
            mapping: base,
            mapping_bytes: memory.nbytes(),
            pages: SubPool::new(base, layout.page_record_bytes as usize, page_count),
            rows: SubPool::new(
                base + page_region,
                layout.row_record_bytes as usize,
                row_count,
            ),
            retain: true,
            registration: None,
            split: Some(Split {
                parts,
                _memory: memory,
            }),
        })
    }

    /// Page and row slab counts of a retaining pool of at most `bytes`.
    fn sized_counts(layout: &Layout, bytes: u64) -> Result<(usize, usize)> {
        let page_record = layout.page_record_bytes;
        let row_record = layout.row_record_bytes;
        if page_record == 0 || row_record == 0 {
            return Err(invalid(
                "record layout declares zero-byte page or row records",
            ));
        }
        let minimum = Self::minimum_bytes(layout);
        if bytes < minimum {
            return Err(invalid(format!(
                "prefix cache memory of {bytes} bytes is below the minimum of {minimum} bytes \
                 for this layout: {MINIMUM_PAGE_SLABS} page records of {page_record} bytes \
                 and {MINIMUM_ROW_SLABS} row records of {row_record} bytes"
            )));
        }
        let quarter_rows = bytes / 4 / row_record;
        let most_rows = (bytes - MINIMUM_PAGE_SLABS as u64 * page_record) / row_record;
        let row_count = quarter_rows.max(MINIMUM_ROW_SLABS as u64).min(most_rows);
        let page_count = (bytes - row_count * row_record) / page_record;
        Ok((page_count as usize, row_count as usize))
    }

    /// Each rank's part of the slabs, in rank order, for a pool split by rank.
    pub fn rank_parts(&self) -> Option<Vec<RankPart>> {
        self.split.as_ref().map(|split| split.parts.clone())
    }

    /// Resolve GPU copy addresses once at startup. Disk I/O and CPU slices
    /// continue to use the host addresses returned by `page_address`/`row_address`.
    pub fn device_parts(&self, layout: &Layout) -> Result<Vec<RankPart>> {
        if layout.page_record_bytes != self.page_bytes() as u64
            || layout.row_record_bytes != self.row_bytes() as u64
        {
            return Err(invalid(
                "slab sizes disagree with the transfer record layout",
            ));
        }
        if let Some(parts) = self.rank_parts() {
            if parts.len() != layout.ranks.len() {
                return Err(invalid("a split pool needs one part per transfer rank"));
            }
            return Ok(parts);
        }
        let registration = self
            .registration
            .as_ref()
            .ok_or_else(|| invalid("transfer slabs must be registered with CUDA"))?;
        Ok(self.registered_parts(layout, registration.device_address()))
    }

    pub fn registered_parts(&self, layout: &Layout, device: u64) -> Vec<RankPart> {
        layout
            .ranks
            .iter()
            .map(|rank| RankPart {
                range: self.mapping..self.mapping + self.mapping_bytes,
                pages: PartRegion {
                    identity: self.pages.base,
                    record_bytes: self.page_bytes(),
                    count: self.page_count(),
                    base: (device + rank.page_offset) as usize,
                    part_bytes: self.page_bytes(),
                },
                rows: PartRegion {
                    identity: self.rows.base,
                    record_bytes: self.row_bytes(),
                    count: self.row_count(),
                    base: (device + (self.rows.base - self.mapping) as u64 + rank.row_offset)
                        as usize,
                    part_bytes: self.row_bytes(),
                },
            })
            .collect()
    }

    /// A non-retaining staging pool of [`STAGING_PAGE_SLABS`] and [`STAGING_ROW_SLABS`].
    pub fn staging(layout: &Layout) -> Result<Self> {
        Self::new(
            layout.page_record_bytes,
            STAGING_PAGE_SLABS,
            layout.row_record_bytes,
            STAGING_ROW_SLABS,
            false,
        )
    }

    /// Bytes of the smallest retaining pool for a layout.
    pub fn minimum_bytes(layout: &Layout) -> u64 {
        MINIMUM_PAGE_SLABS as u64 * layout.page_record_bytes
            + MINIMUM_ROW_SLABS as u64 * layout.row_record_bytes
    }

    /// Register the mapping through `registrar` as portable so every rank's context
    /// can address it, with `context` made current on the calling thread.
    /// A pool that is already registered is not registered again.
    pub fn register(&mut self, registrar: &dyn HostRegistrar, context: usize) -> Result<()> {
        if self.registered() {
            return Err(invalid(
                "slab pool is already registered with the CUDA driver",
            ));
        }
        // The mapping is page aligned and stays mapped at this size for the
        // pool's lifetime; `Drop` releases the registration before unmapping.
        let registration = registrar.register(context, self.mapping, self.mapping_bytes)?;
        self.registration = Some(registration);
        Ok(())
    }

    /// Release the driver registration; a pool that is not registered is unchanged.
    pub fn unregister(&mut self) -> Result<()> {
        self.registration.take();
        Ok(())
    }

    /// Whether the driver can copy the slabs: after `register`, and always for a pool split by rank.
    pub fn registered(&self) -> bool {
        self.registration.is_some() || self.split.is_some()
    }

    pub fn retain(&self) -> bool {
        self.retain
    }

    pub fn page_bytes(&self) -> usize {
        self.pages.slab_bytes
    }
    pub fn row_bytes(&self) -> usize {
        self.rows.slab_bytes
    }
    pub fn page_count(&self) -> usize {
        self.pages.count
    }
    pub fn row_count(&self) -> usize {
        self.rows.count
    }
    pub fn free_pages(&self) -> usize {
        self.pages.free.len()
    }
    pub fn free_rows(&self) -> usize {
        self.rows.free.len()
    }
    pub fn bytes(&self) -> u64 {
        self.mapping_bytes as u64
    }

    pub fn allocate_page(&mut self) -> Option<PageSlab> {
        self.pages.allocate().map(PageSlab)
    }
    pub fn allocate_row(&mut self) -> Option<RowSlab> {
        self.rows.allocate().map(RowSlab)
    }
    /// Returns `slab` to the free list; a slab that is not allocated is ignored.
    pub fn release_page(&mut self, slab: PageSlab) {
        self.pages.release(slab.0);
    }
    /// Returns `slab` to the free list; a slab that is not allocated is ignored.
    pub fn release_row(&mut self, slab: RowSlab) {
        self.rows.release(slab.0);
    }

    pub fn page_address(&self, slab: PageSlab) -> usize {
        self.pages.address(slab.0)
    }
    pub fn row_address(&self, slab: RowSlab) -> usize {
        self.rows.address(slab.0)
    }

    /// The record bytes of a pool that is not split by rank; a split pool has
    /// no contiguous record, so asking for one is a caller defect.
    pub fn page(&self, slab: PageSlab) -> &[u8] {
        self.assert_contiguous();
        // SAFETY: The mapping outlives the pool and the slab lies inside it.
        unsafe {
            std::slice::from_raw_parts(self.page_address(slab) as *const u8, self.pages.slab_bytes)
        }
    }
    pub fn page_mut(&mut self, slab: PageSlab) -> &mut [u8] {
        self.assert_contiguous();
        // SAFETY: As above, with exclusive access through `&mut self`.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.page_address(slab) as *mut u8,
                self.pages.slab_bytes,
            )
        }
    }
    pub fn row(&self, slab: RowSlab) -> &[u8] {
        self.assert_contiguous();
        // SAFETY: As for `page`.
        unsafe {
            std::slice::from_raw_parts(self.row_address(slab) as *const u8, self.rows.slab_bytes)
        }
    }
    pub fn row_mut(&mut self, slab: RowSlab) -> &mut [u8] {
        self.assert_contiguous();
        // SAFETY: As for `page_mut`.
        unsafe {
            std::slice::from_raw_parts_mut(self.row_address(slab) as *mut u8, self.rows.slab_bytes)
        }
    }

    fn assert_contiguous(&self) {
        assert!(
            self.split.is_none(),
            "the slabs of a pool split by rank hold no contiguous record"
        );
    }
}

impl Drop for SlabPool {
    fn drop(&mut self) {
        self.registration.take();
        // A split pool's memory releases its own mappings and reservation.
        if self.split.take().is_none() && self.mapping != 0 {
            // SAFETY: The mapping was created by this pool and is no longer registered.
            unsafe {
                flare_eos::unmap_host(self.mapping, self.mapping_bytes);
            }
        }
    }
}

/// Fixture for tests of code that consumes a pool's parts.
#[cfg(any(test, feature = "test-support"))]
/// A rank's part of a pool split by rank reserved at `range`, whose page and
/// row identities begin at `range.start` and right after the page records.
/// `pages` and `rows` give this rank's part of slab 0 and its share of a record.
pub fn rank_part(
    range: std::ops::Range<usize>,
    (page_record, page_count, page_base, page_part): (usize, usize, usize, usize),
    (row_record, row_count, row_base, row_part): (usize, usize, usize, usize),
) -> RankPart {
    let rows = range.start + page_record * page_count;
    RankPart {
        range: range.clone(),
        pages: PartRegion {
            identity: range.start,
            record_bytes: page_record,
            count: page_count,
            base: page_base,
            part_bytes: page_part,
        },
        rows: PartRegion {
            identity: rows,
            record_bytes: row_record,
            count: row_count,
            base: row_base,
            part_bytes: row_part,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::geometry;

    const PAGE: u64 = 8192;
    const ROW: u64 = 4096;

    #[test]
    fn rank_parts_locate_slabs_and_reject_other_addresses_inside_the_pool() {
        // Two ranks' 8 KiB + 4 KiB page records and 4 KiB + 4 KiB row records:
        // rank 1's part of each page slab is 4 KiB and of each row slab 4 KiB.
        let start = 0x7f00_0000_0000usize;
        let range = start..start + (1 << 20);
        let part = rank_part(
            range.clone(),
            (12288, 10, 0x7f00_0008_0000, 4096),
            (8192, 4, 0x7f00_000a_8000, 4096),
        );
        for index in 0..10 {
            assert_eq!(
                part.page(start + index * 12288).unwrap(),
                Some(0x7f00_0008_0000 + index as u64 * 4096)
            );
        }
        let rows = start + 10 * 12288;
        for index in 0..4 {
            assert_eq!(
                part.row(rows + index * 8192).unwrap(),
                Some(0x7f00_000a_8000 + index as u64 * 4096)
            );
        }
        assert_eq!(part.page(start - 4096).unwrap(), None);
        assert_eq!(part.page(range.end).unwrap(), None);
        assert_eq!(part.row(0x1000).unwrap(), None);
        for (address, kind) in [
            (start + 4096, "page"),
            (start + 10 * 12288, "page"),
            (rows + 4096, "row"),
            (start, "row"),
            (rows + 4 * 8192, "row"),
        ] {
            let located = if kind == "page" {
                part.page(address)
            } else {
                part.row(address)
            };
            let error = located.unwrap_err().to_string();
            assert!(error.contains(&format!("{kind} slabs")), "{error}");
        }
    }

    /// One rank with two 4 KiB planes and a 4 KiB row: 8 KiB page records, 4 KiB row records.
    fn layout() -> Layout {
        Layout::new(&[geometry(64, &[("k", 4096), ("v", 4096)], 4096)]).unwrap()
    }

    #[test]
    fn slabs_are_block_aligned_and_disjoint() {
        let mut pool = SlabPool::new(PAGE, 6, ROW, 3, true).unwrap();
        assert_eq!(pool.bytes(), 6 * PAGE + 3 * ROW);
        let pages: Vec<_> = (0..6).map(|_| pool.allocate_page().unwrap()).collect();
        let rows: Vec<_> = (0..3).map(|_| pool.allocate_row().unwrap()).collect();
        let mut ranges = Vec::new();
        for &slab in &pages {
            let address = pool.page_address(slab);
            assert_eq!(address % BLOCK_BYTES as usize, 0);
            ranges.push((address, address + PAGE as usize));
        }
        for &slab in &rows {
            let address = pool.row_address(slab);
            assert_eq!(address % BLOCK_BYTES as usize, 0);
            ranges.push((address, address + ROW as usize));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "slabs overlap: {pair:?}");
        }
        let end = pool.mapping + pool.mapping_bytes;
        assert!(
            ranges
                .iter()
                .all(|(start, stop)| *start >= pool.mapping && *stop <= end)
        );

        for (fill, &slab) in (1u8..).zip(&pages) {
            pool.page_mut(slab).fill(fill);
        }
        for (fill, &slab) in (101u8..).zip(&rows) {
            pool.row_mut(slab).fill(fill);
        }
        for (fill, &slab) in (1u8..).zip(&pages) {
            assert!(pool.page(slab).iter().all(|&byte| byte == fill));
        }
        for (fill, &slab) in (101u8..).zip(&rows) {
            assert!(pool.row(slab).iter().all(|&byte| byte == fill));
        }
    }

    #[test]
    fn exhaustion_returns_none_then_reuses_released_slabs_in_release_order() {
        let mut pool = SlabPool::new(PAGE, 3, ROW, 2, true).unwrap();
        assert_eq!(pool.free_pages(), 3);
        assert_eq!(pool.free_rows(), 2);
        let pages: Vec<_> = (0..3).map(|_| pool.allocate_page().unwrap()).collect();
        assert_eq!(pages, [PageSlab(0), PageSlab(1), PageSlab(2)]);
        assert_eq!(pool.allocate_page(), None);
        assert_eq!(pool.free_pages(), 0);
        pool.release_page(PageSlab(1));
        pool.release_page(PageSlab(0));
        assert_eq!(pool.free_pages(), 2);
        assert_eq!(pool.allocate_page(), Some(PageSlab(1)));
        assert_eq!(pool.allocate_page(), Some(PageSlab(0)));
        assert_eq!(pool.allocate_page(), None);

        let rows: Vec<_> = (0..2).map(|_| pool.allocate_row().unwrap()).collect();
        assert_eq!(rows, [RowSlab(0), RowSlab(1)]);
        assert_eq!(pool.allocate_row(), None);
        pool.release_row(RowSlab(1));
        assert_eq!(pool.allocate_row(), Some(RowSlab(1)));
        assert_eq!(pool.allocate_row(), None);
        pool.release_row(RowSlab(0));
        pool.release_row(RowSlab(1));
        assert_eq!(pool.free_rows(), 2);
    }

    #[test]
    fn double_release_does_not_duplicate_a_slab() {
        let mut pool = SlabPool::new(PAGE, 2, ROW, 2, true).unwrap();
        let page = pool.allocate_page().unwrap();
        pool.release_page(page);
        pool.release_page(page);
        assert_eq!(pool.free_pages(), 2);
        pool.release_page(PageSlab(1));
        pool.release_page(PageSlab(7));
        assert_eq!(pool.free_pages(), 2);
        let mut pages = [pool.allocate_page().unwrap(), pool.allocate_page().unwrap()];
        pages.sort_unstable();
        assert_eq!(pages, [PageSlab(0), PageSlab(1)]);
        assert_eq!(pool.allocate_page(), None);

        let row = pool.allocate_row().unwrap();
        pool.release_row(row);
        pool.release_row(row);
        pool.release_row(RowSlab(9));
        assert_eq!(pool.free_rows(), 2);
        let mut rows = [pool.allocate_row().unwrap(), pool.allocate_row().unwrap()];
        rows.sort_unstable();
        assert_eq!(rows, [RowSlab(0), RowSlab(1)]);
        assert_eq!(pool.allocate_row(), None);
    }

    #[test]
    fn sized_gives_rows_a_quarter_and_pages_the_rest() {
        let layout = layout();
        let bytes = 16u64 << 20;
        let pool = SlabPool::sized(&layout, bytes).unwrap();
        assert!(pool.retain());
        assert_eq!(pool.page_bytes(), PAGE as usize);
        assert_eq!(pool.row_bytes(), ROW as usize);
        assert_eq!(pool.row_count(), (bytes / 4 / ROW) as usize);
        assert_eq!(pool.page_count(), ((bytes - bytes / 4) / PAGE) as usize);
        assert_eq!(pool.bytes(), bytes);
        assert_eq!(pool.free_pages(), pool.page_count());
        assert_eq!(pool.free_rows(), pool.row_count());
    }

    #[test]
    fn sized_never_goes_below_the_minimum_counts_or_above_the_budget() {
        let layout = layout();
        let minimum = SlabPool::minimum_bytes(&layout);
        let pool = SlabPool::sized(&layout, minimum).unwrap();
        assert_eq!(pool.page_count(), MINIMUM_PAGE_SLABS);
        assert_eq!(pool.row_count(), MINIMUM_ROW_SLABS);
        assert_eq!(pool.bytes(), minimum);

        let budget = minimum + 3 * ROW;
        let pool = SlabPool::sized(&layout, budget).unwrap();
        assert_eq!(pool.page_count(), MINIMUM_PAGE_SLABS);
        assert_eq!(pool.row_count(), MINIMUM_ROW_SLABS + 3);
        assert_eq!(pool.bytes(), budget);

        let budget = minimum + 7 * PAGE + 1;
        let pool = SlabPool::sized(&layout, budget).unwrap();
        assert!(pool.page_count() >= MINIMUM_PAGE_SLABS);
        assert!(pool.row_count() >= MINIMUM_ROW_SLABS);
        assert!(pool.bytes() <= budget);
    }

    #[test]
    fn sized_below_the_minimum_reports_the_minimum() {
        let layout = layout();
        let minimum = SlabPool::minimum_bytes(&layout);
        let Err(error) = SlabPool::sized(&layout, minimum - 1) else {
            panic!("a budget below the minimum must fail");
        };
        let message = error.to_string();
        assert!(message.contains(&minimum.to_string()), "{message}");
        assert!(SlabPool::sized(&layout, 0).is_err());
    }

    #[test]
    fn minimum_bytes_is_one_outbound_and_one_inbound_window() {
        let layout = layout();
        assert_eq!(layout.page_record_bytes, PAGE);
        assert_eq!(layout.row_record_bytes, ROW);
        assert_eq!(SlabPool::minimum_bytes(&layout), 128 * PAGE + 2 * ROW);
    }

    #[test]
    fn staging_pool_has_fixed_counts_and_no_retention() {
        let pool = SlabPool::staging(&layout()).unwrap();
        assert!(!pool.retain());
        assert_eq!(pool.page_count(), STAGING_PAGE_SLABS);
        assert_eq!(pool.row_count(), STAGING_ROW_SLABS);
        assert_eq!(pool.free_pages(), STAGING_PAGE_SLABS);
        assert_eq!(pool.free_rows(), STAGING_ROW_SLABS);
        assert_eq!(
            pool.bytes(),
            STAGING_PAGE_SLABS as u64 * PAGE + STAGING_ROW_SLABS as u64 * ROW
        );
    }

    #[test]
    fn writes_through_mutable_slices_are_visible_through_shared_slices() {
        let mut pool = SlabPool::new(PAGE, 2, ROW, 2, false).unwrap();
        let page = pool.allocate_page().unwrap();
        let row = pool.allocate_row().unwrap();
        assert_eq!(pool.page(page).len(), PAGE as usize);
        assert_eq!(pool.row(row).len(), ROW as usize);
        assert!(pool.page(page).iter().all(|&byte| byte == 0));
        for (index, byte) in pool.page_mut(page).iter_mut().enumerate() {
            *byte = index as u8;
        }
        pool.row_mut(row).copy_from_slice(&[0xA5; 4096]);
        assert!(
            pool.page(page)
                .iter()
                .enumerate()
                .all(|(index, &byte)| byte == index as u8)
        );
        assert!(pool.row(row).iter().all(|&byte| byte == 0xA5));
        assert_eq!(pool.page(page).as_ptr() as usize, pool.page_address(page));
        assert_eq!(pool.row(row).as_ptr() as usize, pool.row_address(row));
    }

    #[test]
    fn rejects_unaligned_empty_or_oversized_slabs() {
        assert!(SlabPool::new(4000, 1, ROW, 1, true).is_err());
        assert!(SlabPool::new(0, 1, ROW, 1, true).is_err());
        assert!(SlabPool::new(PAGE, 1, 4097, 1, true).is_err());
        assert!(SlabPool::new(PAGE, 0, ROW, 0, true).is_err());
        assert!(SlabPool::new(PAGE, usize::MAX, ROW, 1, true).is_err());
        assert!(SlabPool::new(PAGE, 1, ROW, u32::MAX as usize + 1, true).is_err());

        let mut pool = SlabPool::new(0, 0, ROW, 1, true).unwrap();
        assert_eq!(pool.page_count(), 0);
        assert_eq!(pool.allocate_page(), None);
        assert_eq!(pool.allocate_row(), Some(RowSlab(0)));
        assert_eq!(pool.bytes(), ROW);
    }

    #[test]
    fn unregister_without_registration_is_ok() {
        let mut pool = SlabPool::new(PAGE, 1, ROW, 1, true).unwrap();
        assert!(!pool.registered());
        pool.unregister().unwrap();
        pool.unregister().unwrap();
        assert!(!pool.registered());
    }
}
