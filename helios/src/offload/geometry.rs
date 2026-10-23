//! Per-rank cache geometry, the cross-rank record layout, and the layout fingerprint.
//!
//! Every runtime-sized `KV_PAGES` allocation on a rank is indexed by the same
//! physical page id, so one logical page occupies `unit_bytes` at
//! `address + id × unit_bytes` in each of them. Those allocations are the page
//! planes. The single `STATE_ROWS` allocation holds one continuation row per
//! state slot. A record stores every plane of every page and the row of every
//! rank with each plane padded to [`BLOCK_BYTES`], so direct I/O can address any
//! plane at an aligned offset and the same bytes serve staged and direct transports.

use blake2::{Blake2b, Digest, digest::consts::U16};

use crate::{Error, ExecutionError, Result};

/// Alignment of every plane inside a record and of every direct-I/O transfer.
pub const BLOCK_BYTES: u64 = 4096;

/// One runtime-sized allocation addressed by unit index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    pub name: String,
    pub address: u64,
    pub unit_bytes: u64,
}

impl Plane {
    /// Bytes one unit occupies inside a record.
    pub fn padded_bytes(&self) -> u64 {
        self.unit_bytes.div_ceil(BLOCK_BYTES) * BLOCK_BYTES
    }

    pub fn unit_address(&self, index: u64) -> u64 {
        self.address + index * self.unit_bytes
    }
}

/// One rank's checkpoint storage: the row plane and the page planes in name order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheGeometry {
    pub page_tokens: u64,
    pub page_capacity: u64,
    pub row_capacity: u64,
    pub row: Plane,
    pub planes: Vec<Plane>,
    /// KV ids sharing physical bytes with each snapshot row, excluding execution rows.
    /// Ownership of a snapshot excludes every page in its range through GPU retirement.
    pub snapshot_pages: Vec<std::ops::Range<u32>>,
}

impl CacheGeometry {
    /// Bytes a page occupies on the GPU across every plane.
    pub fn page_bytes(&self) -> u64 {
        self.planes.iter().map(|plane| plane.unit_bytes).sum()
    }

    /// Bytes a page occupies inside a record.
    pub fn page_record_bytes(&self) -> u64 {
        self.planes.iter().map(Plane::padded_bytes).sum()
    }

    pub fn row_record_bytes(&self) -> u64 {
        self.row.padded_bytes()
    }

    /// Equal plane names, unit sizes, and page tokens; addresses and capacities may differ.
    pub fn same_layout(&self, other: &Self) -> bool {
        self.page_tokens == other.page_tokens
            && self.row.name == other.row.name
            && self.row.unit_bytes == other.row.unit_bytes
            && self.planes.len() == other.planes.len()
            && self
                .planes
                .iter()
                .zip(&other.planes)
                .all(|(a, b)| a.name == b.name && a.unit_bytes == b.unit_bytes)
    }

    pub fn validate(&self) -> Result<()> {
        if self.page_tokens == 0
            || self.page_capacity == 0
            || self.row_capacity == 0
            || self.row.unit_bytes == 0
            || self.planes.is_empty()
            || self.planes.iter().any(|plane| plane.unit_bytes == 0)
        {
            return Err(invalid(
                "cache geometry requires nonempty planes and capacities",
            ));
        }
        Ok(())
    }
}

/// Byte offsets of one rank's planes and row inside a record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankLayout {
    /// Offset of this rank's first plane inside a page record.
    pub page_offset: u64,
    /// Offset of each plane relative to `page_offset`.
    pub plane_offsets: Vec<u64>,
    /// Offset of this rank's row inside a row record.
    pub row_offset: u64,
}

/// Cross-rank record layout: a page record holds one page of every rank, a row
/// record holds the row of every rank, each plane padded to [`BLOCK_BYTES`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub page_tokens: u64,
    pub ranks: Vec<RankLayout>,
    pub page_record_bytes: u64,
    pub row_record_bytes: u64,
}

impl Layout {
    pub fn new(geometry: &[CacheGeometry]) -> Result<Self> {
        let first = geometry
            .first()
            .ok_or_else(|| invalid("record layout requires at least one rank"))?;
        let mut ranks = Vec::with_capacity(geometry.len());
        let (mut page_record_bytes, mut row_record_bytes) = (0u64, 0u64);
        for rank in geometry {
            rank.validate()?;
            if rank.page_tokens != first.page_tokens {
                return Err(invalid("ranks disagree on page token count"));
            }
            let mut plane_offsets = Vec::with_capacity(rank.planes.len());
            let mut offset = 0;
            for plane in &rank.planes {
                plane_offsets.push(offset);
                offset += plane.padded_bytes();
            }
            ranks.push(RankLayout {
                page_offset: page_record_bytes,
                plane_offsets,
                row_offset: row_record_bytes,
            });
            page_record_bytes += offset;
            row_record_bytes += rank.row_record_bytes();
        }
        Ok(Self {
            page_tokens: first.page_tokens,
            ranks,
            page_record_bytes,
            row_record_bytes,
        })
    }

    /// Bytes a record with `pages` tail pages and a row occupies.
    pub fn record_bytes(&self, pages: u64) -> u64 {
        pages * self.page_record_bytes + self.row_record_bytes
    }
}

/// 16-byte BLAKE2b digest over the rank count, page token count, and every
/// rank's ordered plane names and unit sizes and row name and unit size.
/// Packages sharing a model identity can differ here, for example when they
/// were prepared with different draft-token counts.
pub fn fingerprint(geometry: &[CacheGeometry]) -> [u8; 16] {
    let mut hash = Blake2b::<U16>::new();
    hash.update(b"flare-cache-layout-v1\0");
    hash.update((geometry.len() as u64).to_le_bytes());
    hash.update(
        geometry
            .first()
            .map_or(0, |rank| rank.page_tokens)
            .to_le_bytes(),
    );
    for rank in geometry {
        hash.update((rank.planes.len() as u64).to_le_bytes());
        for plane in &rank.planes {
            hash.update((plane.name.len() as u64).to_le_bytes());
            hash.update(plane.name.as_bytes());
            hash.update(plane.unit_bytes.to_le_bytes());
        }
        hash.update((rank.row.name.len() as u64).to_le_bytes());
        hash.update(rank.row.name.as_bytes());
        hash.update(rank.row.unit_bytes.to_le_bytes());
    }
    hash.finalize().into()
}

/// The fingerprint of one rank's records when every rank runs in its own process and keeps
/// its own record files: [`fingerprint`] of that rank's geometry, bound to the rank and the
/// world size, so a host never restores records another rank wrote.
pub fn shard_fingerprint(geometry: &CacheGeometry, rank: usize, world: usize) -> [u8; 16] {
    let mut hash = Blake2b::<U16>::new();
    hash.update(b"flare-cache-shard-v1\0");
    hash.update((rank as u64).to_le_bytes());
    hash.update((world as u64).to_le_bytes());
    hash.update(fingerprint(std::slice::from_ref(geometry)));
    hash.finalize().into()
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::State, message)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn geometry(
        page_tokens: u64,
        planes: &[(&str, u64)],
        row_bytes: u64,
    ) -> CacheGeometry {
        let mut address = 0x7000_0000_0000u64;
        let planes = planes
            .iter()
            .map(|(name, unit_bytes)| {
                let plane = Plane {
                    name: (*name).to_owned(),
                    address,
                    unit_bytes: *unit_bytes,
                };
                address += unit_bytes * 4096;
                plane
            })
            .collect();
        CacheGeometry {
            snapshot_pages: Vec::new(),
            page_tokens,
            page_capacity: 4096,
            row_capacity: 64,
            row: Plane {
                name: "state".into(),
                address,
                unit_bytes: row_bytes,
            },
            planes,
        }
    }

    #[test]
    fn layout_pads_every_plane_and_concatenates_ranks() {
        let ranks = [
            geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
            geometry(64, &[("k", 8192), ("v", 8192), ("sel", 100)], 10_000),
        ];
        let layout = Layout::new(&ranks).unwrap();
        assert_eq!(layout.ranks[0].plane_offsets, [0, 8192, 16384]);
        assert_eq!(layout.ranks[0].page_offset, 0);
        assert_eq!(layout.ranks[1].page_offset, 16384 + 4096);
        assert_eq!(layout.page_record_bytes, 2 * (16384 + 4096));
        assert_eq!(layout.ranks[1].row_offset, 12288);
        assert_eq!(layout.row_record_bytes, 2 * 12288);
        assert_eq!(
            layout.record_bytes(3),
            3 * layout.page_record_bytes + 2 * 12288
        );
        assert_eq!(ranks[0].page_bytes(), 16484);
    }

    #[test]
    fn fingerprint_ignores_addresses_and_capacities_but_not_layout() {
        let base = geometry(64, &[("k", 8192), ("v", 8192)], 10_000);
        let mut moved = base.clone();
        moved.page_capacity *= 2;
        moved.row_capacity += 7;
        for plane in &mut moved.planes {
            plane.address += 0x1000_0000;
        }
        assert_eq!(fingerprint(&[base.clone()]), fingerprint(&[moved.clone()]));
        assert!(base.same_layout(&moved));
        let mut drafted = base.clone();
        drafted.planes.push(Plane {
            name: "draft".into(),
            address: 0,
            unit_bytes: 64,
        });
        assert_ne!(
            fingerprint(&[base.clone()]),
            fingerprint(&[drafted.clone()])
        );
        assert!(!base.same_layout(&drafted));
        let mut wider = base.clone();
        wider.row.unit_bytes += 1;
        assert_ne!(fingerprint(&[base.clone()]), fingerprint(&[wider]));
        assert_ne!(
            fingerprint(&[base.clone()]),
            fingerprint(&[base.clone(), base.clone()])
        );
        let mut paged = base.clone();
        paged.page_tokens = 128;
        assert_ne!(fingerprint(&[base]), fingerprint(&[paged]));
    }

    #[test]
    fn layout_rejects_empty_or_inconsistent_ranks() {
        assert!(Layout::new(&[]).is_err());
        let a = geometry(64, &[("k", 8192)], 4096);
        let b = geometry(128, &[("k", 8192)], 4096);
        assert!(Layout::new(&[a.clone(), b]).is_err());
        let mut empty = a.clone();
        empty.planes.clear();
        assert!(Layout::new(&[empty]).is_err());
        let mut zero = a;
        zero.row.unit_bytes = 0;
        assert!(Layout::new(&[zero]).is_err());
    }
}

#[cfg(all(test, feature = "runtime-package"))]
mod package_tests {
    #[test]
    #[ignore = "requires FLARE_OFFLOAD_TEST_PACKAGE naming an exported package and a free GPU 0"]
    fn package_geometry_lists_every_plane() {
        let Some(package) = std::env::var_os("FLARE_OFFLOAD_TEST_PACKAGE") else {
            return;
        };
        let model = unsafe {
            crate::runtime_package::NativeModel::load(
                crate::runtime_package::test_support::authorize(std::path::Path::new(&package)),
                &[0],
                &crate::runtime_package::LoadOptions::default(),
                &crate::startup::StartupProgress::silent(),
            )
        }
        .unwrap();
        let manifest = model.manifests().unwrap().remove(0);
        let geometry = manifest.cache.unwrap();
        eprintln!(
            "page_tokens {} pages {} rows {} row {} {} bytes",
            geometry.page_tokens,
            geometry.page_capacity,
            geometry.row_capacity,
            geometry.row.name,
            geometry.row.unit_bytes
        );
        for plane in &geometry.planes {
            eprintln!(
                "plane {} {} bytes at {:#x}",
                plane.name, plane.unit_bytes, plane.address
            );
        }
        eprintln!(
            "page bytes {} record {}",
            geometry.page_bytes(),
            geometry.page_record_bytes()
        );
    }
}
