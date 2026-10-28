//! Fixed virtual aliases let immutable continuation rows borrow KV storage.

use std::ops::Range;

use super::{
    NativeResources, Storage,
    driver::{Access, AllocationProperties, Location},
    format::{Memory, Package, invalid},
};
use crate::Result;
use flare_selene::CacheGeometry;

#[derive(Debug)]
struct Chunk {
    plane: usize,
    offset: u64,
    bytes: u64,
    row_offset: u64,
    row_bytes: u64,
}

#[derive(Debug)]
pub(super) struct Layout {
    fixed_bytes: u64,
    chunks: Vec<Chunk>,
    pub aliases: Vec<Range<u32>>,
}

pub(super) fn alias_plane(unit: u64, largest: u64, row: u64, page: u64, alignment: u64) -> bool {
    // A tiny plane can exclude hundreds of pages for one mapping granule. Borrow
    // from planes whose granules span at most a row's page cost, including the densest.
    unit == largest || alignment.div_ceil(unit) <= row.div_ceil(page)
}

impl Layout {
    pub fn new(
        geometry: &CacheGeometry,
        execution: u64,
        alignments: &[u64],
        row_alignment: u64,
    ) -> Result<Self> {
        let row_bytes = geometry.row.unit_bytes;
        let fixed_bytes = (execution * row_bytes).div_ceil(row_alignment) * row_alignment;
        let row_end = (geometry.row_capacity * row_bytes).div_ceil(row_alignment) * row_alignment;
        let snapshots = geometry.row_capacity - execution;
        let mut aliases = vec![u32::MAX..0; snapshots as usize];
        let mut cursor = fixed_bytes;
        let mut chunks = Vec::new();
        let largest = geometry.planes.iter().map(|p| p.unit_bytes).max().unwrap();
        let eligible: Vec<_> = geometry
            .planes
            .iter()
            .zip(alignments)
            .map(|(p, &alignment)| {
                alias_plane(
                    p.unit_bytes,
                    largest,
                    row_bytes,
                    geometry.page_bytes(),
                    alignment,
                )
            })
            .collect();
        for (plane, (p, &alignment)) in geometry.planes.iter().zip(alignments).enumerate() {
            if alignment != row_alignment {
                return Err(invalid(
                    "cache aliases require matching CUDA allocation granularity",
                ));
            }
            if !eligible[plane] {
                chunks.push(Chunk {
                    plane,
                    offset: 0,
                    bytes: (geometry.page_capacity * p.unit_bytes).div_ceil(alignment) * alignment,
                    row_offset: cursor,
                    row_bytes: 0,
                });
            }
        }
        // Interleave the planes in small page runs so a row borrows approximately its own
        // byte cost in logical pages. Begin at the high end; requests allocate low ids first.
        let group = if snapshots == 0 {
            geometry.page_capacity
        } else {
            (row_bytes / geometry.page_bytes() / 8).clamp(1, 32)
        };
        for group_id in (0..geometry.page_capacity.div_ceil(group)).rev() {
            let first = group_id * group;
            let last = (first + group).min(geometry.page_capacity);
            for (plane, (p, &alignment)) in geometry.planes.iter().zip(alignments).enumerate() {
                if !eligible[plane] {
                    continue;
                }
                let offset = first * p.unit_bytes / alignment * alignment;
                let end = if last == geometry.page_capacity {
                    (last * p.unit_bytes).div_ceil(alignment) * alignment
                } else {
                    last * p.unit_bytes / alignment * alignment
                };
                if end == offset {
                    continue;
                }
                let bytes = end - offset;
                let mapped = bytes.min(row_end.saturating_sub(cursor));
                if mapped > 0 {
                    let first_row = (cursor / row_bytes).max(execution);
                    let last_row = (cursor + mapped)
                        .div_ceil(row_bytes)
                        .min(geometry.row_capacity);
                    for row in first_row..last_row {
                        let start = offset + (row * row_bytes).saturating_sub(cursor);
                        let end = offset + ((row + 1) * row_bytes).min(cursor + mapped) - cursor;
                        let range = &mut aliases[(row - execution) as usize];
                        range.start = range.start.min((start / p.unit_bytes) as u32);
                        range.end = range
                            .end
                            .max(end.div_ceil(p.unit_bytes).min(geometry.page_capacity) as u32);
                    }
                }
                chunks.push(Chunk {
                    plane,
                    offset,
                    bytes,
                    row_offset: cursor,
                    row_bytes: mapped,
                });
                cursor += mapped;
            }
            if cursor == row_end {
                for (plane, p) in geometry.planes.iter().enumerate() {
                    if !eligible[plane] {
                        continue;
                    }
                    let bytes = first * p.unit_bytes / row_alignment * row_alignment;
                    if bytes != 0 {
                        chunks.push(Chunk {
                            plane,
                            offset: 0,
                            bytes,
                            row_offset: cursor,
                            row_bytes: 0,
                        });
                    }
                }
                break;
            }
        }
        if cursor < row_end {
            return Err(invalid("KV backing cannot cover continuation aliases"));
        }
        for (snapshot, range) in aliases.iter_mut().enumerate() {
            if (execution + snapshot as u64 + 1) * row_bytes <= fixed_bytes {
                *range = 0..0;
            } else if range.start >= range.end {
                return Err(invalid("continuation alias lacks a KV conflict range"));
            }
        }
        Ok(Self {
            fixed_bytes,
            chunks,
            aliases,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(row_bytes: u64, units: &[u64], pages: u64, rows: u64) -> CacheGeometry {
        CacheGeometry {
            page_tokens: 64,
            page_capacity: pages,
            row_capacity: rows,
            row: flare_selene::Plane {
                name: "rows".into(),
                address: 0,
                unit_bytes: row_bytes,
            },
            planes: units
                .iter()
                .enumerate()
                .map(|(i, &unit)| flare_selene::Plane {
                    name: format!("plane-{i}"),
                    address: 0,
                    unit_bytes: unit,
                })
                .collect(),
            snapshot_pages: Vec::new(),
        }
    }

    #[test]
    fn alignment_padding_can_back_complete_snapshots_without_kv_aliases() {
        for snapshots in [0, 1, 7, 8, 16, 32] {
            let geometry = geometry(64 << 10, &[64 << 10, 4 << 10], 1024, 9 + snapshots);
            let layout = Layout::new(&geometry, 9, &[2 << 20; 2], 2 << 20).unwrap();
            assert_eq!(layout.aliases.len(), snapshots as usize);
            for (row, range) in layout.aliases.iter().enumerate() {
                assert_eq!(range.is_empty(), row < 23);
            }
        }
    }

    #[test]
    #[ignore = "requires an assigned CUDA_VISIBLE_DEVICES GPU"]
    fn cuda_aliases_preserve_rows_while_unowned_pages_are_rewritten() {
        use super::super::{
            driver::Driver,
            format::{Allocation, Context},
        };
        let driver = Driver::load().unwrap();
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            let mut device = 0;
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            let mut architecture = [0; 2];
            for (value, attribute) in architecture.iter_mut().zip([75, 76]) {
                driver
                    .check(
                        "cuDeviceGetAttribute",
                        (driver.device_attribute)(value, attribute, device),
                    )
                    .unwrap();
            }
            let mut package = Package {
                resources: Default::default(),
                events: vec![],
                architecture,
                modules: vec![],
                functions: vec![],
                graphs: vec![],
                reservations: vec![],
                execution: None,
                context: Context::default(),
                allocations: vec![],
            };
            let mut owner = NativeResources::initialize(&package, 0).unwrap();
            let properties = AllocationProperties {
                kind: 1,
                handles: 0,
                location: Location { kind: 1, device },
                windows: std::ptr::null_mut(),
                flags: [0; 8],
            };
            let mut alignment = 0;
            driver
                .check(
                    "cuMemGetAllocationGranularity",
                    (driver.granularity)(&mut alignment, &properties, 0),
                )
                .unwrap();
            let alignment = alignment as u64;
            let mut geometry = geometry(
                3 * alignment / 2,
                &[alignment / 8, alignment / 4, alignment / 64],
                128,
                7,
            );
            for plane in std::iter::once(&mut geometry.row).chain(&mut geometry.planes) {
                let units = if plane.name == "rows" { 7 } else { 128 };
                let bytes = (units * plane.unit_bytes).div_ceil(alignment) * alignment;
                driver
                    .check(
                        "cuMemAddressReserve",
                        (driver.reserve)(&mut plane.address, bytes as usize, 0, 0, 0),
                    )
                    .unwrap();
                owner.reservations.push((plane.address, bytes as usize));
                package.allocations.push(Allocation {
                    name: plane.name.clone(),
                    nbytes: bytes as usize,
                    alignment: alignment as usize,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(plane.address),
                    memory: Memory::Device,
                    sizing: None,
                });
            }
            let layout = Layout::new(&geometry, 3, &[alignment; 3], alignment).unwrap();
            owner
                .allocate_shared_cache(&package, &geometry, &layout)
                .unwrap();
            for row in 0..7 {
                driver
                    .check(
                        "cuMemsetD8_v2",
                        (driver.zero)(
                            geometry.row.address + row * geometry.row.unit_bytes,
                            row as u8 + 1,
                            geometry.row.unit_bytes as usize,
                        ),
                    )
                    .unwrap();
            }
            // Release snapshots one at a time, overwriting every page outside the remaining exclusions.
            for released in 0..=4 {
                for page in 0..geometry.page_capacity {
                    if layout.aliases[released..]
                        .iter()
                        .any(|r| r.contains(&(page as u32)))
                    {
                        continue;
                    }
                    for plane in &geometry.planes {
                        driver
                            .check(
                                "cuMemsetD8_v2",
                                (driver.zero)(
                                    plane.address + page * plane.unit_bytes,
                                    0xA5,
                                    plane.unit_bytes as usize,
                                ),
                            )
                            .unwrap();
                    }
                }
                driver
                    .check("cuCtxSynchronize", (driver.context_sync)())
                    .unwrap();
                for row in (0..3).chain(3 + released as u64..7) {
                    let mut contents = vec![0; geometry.row.unit_bytes as usize];
                    driver
                        .check(
                            "cuMemcpyDtoH_v2",
                            (driver.download)(
                                contents.as_mut_ptr().cast(),
                                geometry.row.address + row * geometry.row.unit_bytes,
                                contents.len(),
                            ),
                        )
                        .unwrap();
                    assert!(
                        contents.iter().all(|&b| b == row as u8 + 1),
                        "released {released}, row {row}"
                    );
                }
            }
        }
    }

    #[test]
    fn unaligned_rows_and_multiplane_pages_have_complete_conflict_ranges() {
        for (row_bytes, units, pages, alignment) in [
            (81_464_576, vec![135_168, 8_448].repeat(12), 3844, 65536),
            (81_464_576, vec![135_168, 8_448].repeat(12), 3844, 2097152),
            (
                122_932_224,
                vec![65_536, 65_536, 4_096].repeat(13),
                9056,
                2097152,
            ),
        ] {
            let planes: Vec<_> = units
                .iter()
                .enumerate()
                .map(|(i, &unit)| flare_selene::Plane {
                    name: format!("plane-{i}"),
                    address: 0,
                    unit_bytes: unit,
                })
                .collect();
            let geometry = CacheGeometry {
                page_tokens: 64,
                page_capacity: pages,
                row_capacity: 41,
                row: flare_selene::Plane {
                    name: "rows".into(),
                    address: 0,
                    unit_bytes: row_bytes,
                },
                planes,
                snapshot_pages: Vec::new(),
            };
            let layout =
                Layout::new(&geometry, 9, &vec![alignment; units.len()], alignment).unwrap();
            assert_eq!(layout.aliases.len(), 32);
            let physical = layout.fixed_bytes + layout.chunks.iter().map(|c| c.bytes).sum::<u64>();
            let expected = layout.fixed_bytes
                + units
                    .iter()
                    .map(|u| (u * pages).div_ceil(alignment) * alignment)
                    .sum::<u64>();
            assert_eq!(physical, expected);
            for (plane, &unit) in units.iter().enumerate() {
                let mut pieces: Vec<_> =
                    layout.chunks.iter().filter(|c| c.plane == plane).collect();
                pieces.sort_by_key(|c| c.offset);
                let mut end = 0;
                for chunk in pieces {
                    assert_eq!(chunk.offset, end);
                    end += chunk.bytes;
                }
                assert_eq!(end, (unit * pages).div_ceil(alignment) * alignment);
            }
            for (snapshot, range) in layout.aliases.iter().enumerate() {
                let start = (snapshot as u64 + 9) * row_bytes;
                let end = start + row_bytes;
                for chunk in &layout.chunks {
                    let lo = start.max(chunk.row_offset);
                    let hi = end.min(chunk.row_offset + chunk.row_bytes);
                    if lo >= hi {
                        continue;
                    }
                    let unit = units[chunk.plane];
                    let first = (chunk.offset + lo - chunk.row_offset) / unit;
                    let last = (chunk.offset + hi - chunk.row_offset)
                        .div_ceil(unit)
                        .min(pages);
                    assert!(range.start as u64 <= first && range.end as u64 >= last);
                }
                // Interleaving prevents one row from excluding whole per-layer page pools.
                assert!(
                    (range.end - range.start) as u64
                        <= 2 * row_bytes.div_ceil(geometry.page_bytes()) + 16
                );
            }
        }
    }
}

impl NativeResources {
    pub(super) unsafe fn allocate_shared_cache(
        &mut self,
        package: &Package,
        geometry: &CacheGeometry,
        layout: &Layout,
    ) -> Result<()> {
        unsafe {
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            for plane in std::iter::once(&geometry.row).chain(&geometry.planes) {
                let allocation = package
                    .allocations
                    .iter()
                    .find(|a| a.name == plane.name)
                    .unwrap();
                self.allocations.insert(
                    plane.name.clone(),
                    Storage {
                        base: 0,
                        address: plane.address,
                        nbytes: allocation.nbytes,
                        memory: Memory::Device,
                        fixed: true,
                        mappings: Vec::new(),
                    },
                );
            }
            self.map_cache_chunk(
                layout.fixed_bytes,
                &[(
                    geometry.row.name.as_str(),
                    geometry.row.address,
                    layout.fixed_bytes,
                )],
            )?;
            for chunk in &layout.chunks {
                let plane = &geometry.planes[chunk.plane];
                let mut mappings = vec![(
                    plane.name.as_str(),
                    plane.address + chunk.offset,
                    chunk.bytes,
                )];
                if chunk.row_bytes != 0 {
                    mappings.push((
                        geometry.row.name.as_str(),
                        geometry.row.address + chunk.row_offset,
                        chunk.row_bytes,
                    ));
                }
                self.map_cache_chunk(chunk.bytes, &mappings)?;
            }
            // Zero each physical byte once, before any graph or transfer can read an alias.
            self.driver.check(
                "cuMemsetD8_v2",
                (self.driver.zero)(geometry.row.address, 0, layout.fixed_bytes as usize),
            )?;
            for plane in &geometry.planes {
                let bytes = self.allocations[&plane.name].nbytes;
                self.driver
                    .check("cuMemsetD8_v2", (self.driver.zero)(plane.address, 0, bytes))?;
            }
            self.driver
                .check("cuCtxSynchronize", (self.driver.context_sync)())?;
        }
        Ok(())
    }

    unsafe fn map_cache_chunk(&mut self, bytes: u64, mappings: &[(&str, u64, u64)]) -> Result<()> {
        let properties = AllocationProperties {
            kind: 1,
            handles: 0,
            location: Location {
                kind: 1,
                device: self.device,
            },
            windows: std::ptr::null_mut(),
            flags: [0; 8],
        };
        let access = Access {
            location: Location {
                kind: 1,
                device: self.device,
            },
            flags: 3,
        };
        let mut physical = 0;
        unsafe {
            self.driver.check(
                "cuMemCreate",
                (self.driver.physical_create)(&mut physical, bytes as usize, &properties, 0),
            )?;
            let result = (|| {
                for &(name, address, length) in mappings {
                    self.driver.check(
                        "cuMemMap",
                        (self.driver.map)(address, length as usize, 0, physical, 0),
                    )?;
                    self.allocations
                        .get_mut(name)
                        .unwrap()
                        .mappings
                        .push((address, length as usize));
                    self.driver.check(
                        "cuMemSetAccess",
                        (self.driver.access)(address, length as usize, &access, 1),
                    )?;
                }
                Ok(())
            })();
            (self.driver.physical_release)(physical);
            result
        }
    }
}
