/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conservative retention of graph-reachable allocations with typed numeric leaves.

use std::collections::{HashMap, HashSet, VecDeque};

use super::{Allocation, ExportResult, Kernel, invalid, wire};

pub(crate) fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|(start, end)| start < end);
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

pub(crate) fn subtract_ranges(start: u64, end: u64, excluded: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut cursor = start;
    let mut spans = Vec::new();
    let first = excluded.partition_point(|&(_, stop)| stop <= start);
    for &(begin, stop) in &excluded[first..] {
        if begin >= end {
            break;
        }
        if cursor < begin {
            spans.push((cursor, begin.min(end)));
        }
        cursor = cursor.max(stop.min(end));
    }
    if cursor < end {
        spans.push((cursor, end));
    }
    spans
}

pub(crate) struct Reachability {
    allocations: HashMap<String, Allocation>,
    intervals: Vec<(u64, u64, String)>,
    numeric: Vec<(u64, u64)>,
    inactive: Vec<(u64, u64)>,
    pub(crate) reached: HashSet<String>,
    pending: VecDeque<String>,
    blobs: HashSet<(Vec<u8>, Vec<(usize, usize)>)>,
    tail: Option<(String, u64, Vec<u8>)>,
}

impl Reachability {
    pub(crate) fn new(
        allocations: &[Allocation],
        numeric: &[(u64, u64)],
        inactive: &[(u64, u64)],
        roots: &[String],
    ) -> ExportResult<Self> {
        let mut this = Self {
            allocations: allocations
                .iter()
                .cloned()
                .map(|a| (a.name.clone(), a))
                .collect(),
            intervals: allocations
                .iter()
                .map(|a| (a.address, a.address + a.nbytes, a.name.clone()))
                .collect(),
            numeric: merge_ranges(numeric.to_vec()),
            inactive: merge_ranges(inactive.to_vec()),
            reached: HashSet::new(),
            pending: VecDeque::new(),
            blobs: HashSet::new(),
            tail: None,
        };
        this.intervals.sort_unstable();
        for name in roots {
            this.reference(name)?;
        }
        Ok(this)
    }

    pub(crate) fn reference(&mut self, name: &str) -> ExportResult<()> {
        if !self.allocations.contains_key(name) {
            return Err(invalid(format!("unregistered graph allocation: {name}")));
        }
        if self.reached.insert(name.to_owned()) {
            self.pending.push_back(name.to_owned());
        }
        Ok(())
    }

    /// Drop an allocation from further consideration.
    ///
    /// A recorder-derived record can outlive the memory it describes. Forgetting it keeps a released
    /// range from being retained and stops later pointer hits inside that range from re-adding it.
    pub(crate) fn forget(&mut self, name: &str) {
        self.allocations.remove(name);
        self.intervals.retain(|(_, _, existing)| existing != name);
        self.reached.remove(name);
        self.pending.retain(|existing| existing != name);
    }

    pub(crate) fn pointer(&mut self, value: u64) -> ExportResult<()> {
        let inactive = self.inactive.partition_point(|&(start, _)| start <= value);
        if inactive > 0 && value < self.inactive[inactive - 1].1 {
            return Ok(());
        }
        let index = self
            .intervals
            .partition_point(|&(start, _, _)| start <= value);
        if index > 0 && value < self.intervals[index - 1].1 {
            let name = self.intervals[index - 1].2.clone();
            self.reference(&name)?;
        }
        Ok(())
    }

    pub(crate) fn blob(&mut self, data: &[u8], numeric: &[(usize, usize)]) -> ExportResult<()> {
        if !self.blobs.insert((data.to_vec(), numeric.to_vec())) {
            return Ok(());
        }
        let excluded = merge_ranges(
            numeric
                .iter()
                .map(|&(start, end)| (start as u64, end as u64))
                .collect(),
        );
        for (start, end) in subtract_ranges(0, data.len() as u64, &excluded) {
            self.scan(&data[start as usize..end as usize])?;
        }
        Ok(())
    }

    fn scan(&mut self, data: &[u8]) -> ExportResult<()> {
        if self.intervals.is_empty() {
            return Ok(());
        }
        let floor = self.intervals[0].0;
        let ceiling = self.intervals.last().unwrap().1;
        for bytes in data.windows(8) {
            let pointer = u64::from_le_bytes(bytes.try_into().unwrap());
            if pointer >= floor && pointer < ceiling {
                self.pointer(pointer)?;
            }
        }
        Ok(())
    }

    pub(crate) fn graph(&mut self, nodes: &[wire::Node], kernels: &[Kernel]) -> ExportResult<()> {
        use wire::node::Operation;
        for node in nodes {
            match node
                .operation
                .as_ref()
                .ok_or_else(|| invalid("graph node lacks operation"))?
            {
                Operation::Kernel(node) => {
                    let kernel = kernels
                        .get(node.function as usize)
                        .ok_or_else(|| invalid("graph function index is out of bounds"))?;
                    if node.arguments.len() != kernel.arguments.len() {
                        return Err(invalid("kernel argument count differs from its function"));
                    }
                    for (blob, argument) in node.arguments.iter().zip(&kernel.arguments) {
                        for relocation in &blob.relocations {
                            self.reference(
                                &relocation
                                    .target
                                    .as_ref()
                                    .ok_or_else(|| invalid("relocation lacks target"))?
                                    .allocation,
                            )?;
                        }
                        self.blob(&blob.data, &argument.numeric)?;
                    }
                    for attribute in &node.attributes {
                        if let Some(value) = &attribute.value {
                            for relocation in &value.relocations {
                                self.reference(
                                    &relocation
                                        .target
                                        .as_ref()
                                        .ok_or_else(|| {
                                            invalid("attribute relocation lacks target")
                                        })?
                                        .allocation,
                                )?;
                            }
                        }
                    }
                }
                Operation::Copy(copy) => {
                    self.reference(
                        &copy
                            .source
                            .as_ref()
                            .ok_or_else(|| invalid("copy lacks source"))?
                            .allocation,
                    )?;
                    self.reference(
                        &copy
                            .destination
                            .as_ref()
                            .ok_or_else(|| invalid("copy lacks destination"))?
                            .allocation,
                    )?;
                }
                Operation::Memset(memset) => self.reference(
                    &memset
                        .destination
                        .as_ref()
                        .ok_or_else(|| invalid("memset lacks destination"))?
                        .allocation,
                )?,
                Operation::Empty(_) | Operation::EventWait(_) => {}
            }
        }
        Ok(())
    }

    fn scan_spans(&mut self, address: u64, data: &[u8], numeric: bool) -> ExportResult<()> {
        if numeric {
            let end = address
                .checked_add(data.len() as u64)
                .ok_or_else(|| invalid("allocation scan extent overflow"))?;
            for (start, stop) in subtract_ranges(address, end, &self.numeric) {
                self.scan(&data[(start - address) as usize..(stop - address) as usize])?;
            }
            Ok(())
        } else {
            self.scan(data)
        }
    }

    fn scan_contents(
        &mut self,
        name: &str,
        address: u64,
        data: &[u8],
        numeric: bool,
    ) -> ExportResult<()> {
        let mut boundary = [0u8; 14];
        let mut previous = 0;
        if let Some((previous_name, previous_end, tail)) = self.tail.take() {
            if previous_name == name && previous_end == address {
                previous = tail.len();
                boundary[..previous].copy_from_slice(&tail);
            }
        }
        let prefix = data.len().min(7);
        boundary[previous..previous + prefix].copy_from_slice(&data[..prefix]);
        if previous > 0 {
            let start = address
                .checked_sub(previous as u64)
                .ok_or_else(|| invalid("allocation scan boundary underflow"))?;
            self.scan_spans(start, &boundary[..previous + prefix], numeric)?;
        }
        self.scan_spans(address, data, numeric)?;
        let tail = if data.len() >= 7 {
            &data[data.len() - 7..]
        } else {
            &boundary[(previous + prefix).saturating_sub(7)..previous + prefix]
        };
        self.tail = Some((
            name.to_owned(),
            address
                .checked_add(data.len() as u64)
                .ok_or_else(|| invalid("allocation scan extent overflow"))?,
            tail.to_vec(),
        ));
        Ok(())
    }

    pub(crate) fn contents(
        &mut self,
        allocation: &Allocation,
        offset: u64,
        data: &[u8],
    ) -> ExportResult<()> {
        let address = allocation
            .address
            .checked_add(offset)
            .ok_or_else(|| invalid("allocation scan address overflow"))?;
        self.scan_contents(&allocation.name, address, data, true)
    }

    pub(crate) fn opaque_contents(
        &mut self,
        name: &str,
        offset: u64,
        data: &[u8],
    ) -> ExportResult<()> {
        self.scan_contents(name, offset, data, false)
    }

    pub(crate) fn take(&mut self) -> ExportResult<Option<Allocation>> {
        let Some(name) = self.pending.pop_front() else {
            return Ok(None);
        };
        let allocation = self.allocations[&name].clone();
        for relocation in &allocation.relocations {
            self.reference(&relocation.target.allocation)?;
        }
        Ok(Some(allocation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn allocation(name: &str, address: u64) -> Allocation {
        Allocation {
            name: name.into(),
            address,
            nbytes: 256,
            alignment: 256,
            initialize: "snapshot".into(),
            relocations: vec![],
            preserve_address: true,
            memory: "device".into(),
            snapshot_spans: None,
            sizing: None,
        }
    }
    #[test]
    fn unaligned_pointer_across_chunks_retains_target() {
        let allocations = [allocation("source", 0x1000), allocation("target", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[], &[], &["source".into()]).unwrap();
        let mut bytes = vec![0; 19];
        bytes[5..13].copy_from_slice(&0x2040u64.to_le_bytes());
        reach.contents(&allocations[0], 0, &bytes[..9]).unwrap();
        reach.contents(&allocations[0], 9, &bytes[9..]).unwrap();
        assert!(reach.reached.contains("target"));
    }
    #[test]
    fn forgotten_allocation_is_not_referenced_by_a_later_pointer_hit() {
        let allocations = [allocation("source", 0x1000), allocation("released", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[], &[], &[]).unwrap();
        reach.pointer(0x2040).unwrap();
        assert!(reach.reached.contains("released"));
        reach.forget("released");
        assert!(!reach.reached.contains("released"));
        assert!(reach.take().unwrap().is_none(), "a forgotten allocation is never scanned");
        reach.pointer(0x2040).unwrap();
        assert!(
            !reach.reached.contains("released"),
            "a forgotten range must not be re-referenced"
        );
    }

    #[test]
    fn arbitrary_chunk_widths_preserve_pointer_hits_across_numeric_ranges_and_gaps() {
        let mut source = allocation("source", 0x0800_0000_0000);
        source.nbytes = 256;
        let mut allocations = vec![source.clone()];
        for index in 0..6 {
            allocations.push(allocation(
                &format!("target-{index}"),
                0x0800_1000_0000 + index * 0x1000,
            ));
        }
        let mut state = 17u32;
        let mut data = (0..256)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 24) as u8
            })
            .collect::<Vec<_>>();
        for (index, offset) in [5, 19, 61, 111, 178, 226].into_iter().enumerate() {
            data[offset..offset + 8]
                .copy_from_slice(&(allocations[index + 1].address + 13).to_le_bytes());
        }
        let numeric = [
            (source.address, source.address + 5),
            (source.address + 16, source.address + 28),
            (source.address + 107, source.address + 114),
        ];
        let expected: HashSet<String> = ["target-0", "target-4", "target-5"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for width in 1..=31 {
            let mut reach = Reachability::new(&allocations, &numeric, &[], &[]).unwrap();
            for (start, end) in [(0, 59), (65, 170), (176, 256)] {
                for offset in (start..end).step_by(width) {
                    reach
                        .contents(
                            &source,
                            offset as u64,
                            &data[offset..(offset + width).min(end)],
                        )
                        .unwrap();
                }
            }
            assert_eq!(reach.reached, expected, "chunk width {width}");
        }
    }

    #[test]
    fn numeric_leaves_and_inactive_candidates_do_not_retain() {
        let allocations = [
            allocation("source", 0x1000),
            allocation("numeric", 0x2000),
            allocation("inactive", 0x3000),
        ];
        let mut reach =
            Reachability::new(&allocations, &[(0x1000, 0x1008)], &[(0x3000, 0x3100)], &[]).unwrap();
        let bytes = [0x2000u64.to_le_bytes(), 0x3000u64.to_le_bytes()].concat();
        reach.contents(&allocations[0], 0, &bytes).unwrap();
        assert!(reach.reached.is_empty());
        reach.reference("inactive").unwrap();
        assert!(reach.reached.contains("inactive"));
    }
    #[test]
    fn opaque_globals_scan_across_chunks_without_numeric_tensor_classification() {
        let allocations = [allocation("target", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[(0, 32)], &[], &[]).unwrap();
        let mut bytes = vec![0; 15];
        bytes[3..11].copy_from_slice(&0x2020u64.to_le_bytes());
        reach
            .opaque_contents("module-global", 0, &bytes[..7])
            .unwrap();
        reach
            .opaque_contents("module-global", 7, &bytes[7..])
            .unwrap();
        assert!(reach.reached.contains("target"));
    }

    #[test]
    fn numeric_argument_holes_are_conservative() {
        let allocations = [allocation("target", 0x2000)];
        let mut reach = Reachability::new(&allocations, &[], &[], &[]).unwrap();
        let bytes = [0x2000u64.to_le_bytes(), 0x2000u64.to_le_bytes()].concat();
        reach.blob(&bytes, &[(0, 8)]).unwrap();
        assert!(reach.reached.contains("target"));
    }

    #[test]
    fn numeric_scale_and_adjacent_bytes_never_form_a_pointer() {
        let source = allocation("scale", 0x200a295e5000);
        let target = allocation("unused", 0x200a3f800000);
        let mut reach = Reachability::new(
            &[source.clone(), target.clone()],
            &[(source.address, source.address + 4)],
            &[],
            &[source.name.clone()],
        )
        .unwrap();
        let bytes = target.address.to_le_bytes();
        assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), 1.0);
        reach.contents(&source, 0, &bytes).unwrap();
        assert_eq!(reach.reached, HashSet::from([source.name]));
    }

    #[test]
    fn pointer_table_beside_numeric_values_survives_chunk_boundaries() {
        let source = allocation("mixed", 0x200000000000);
        let live = allocation("live", source.address + 256);
        let dead = allocation("dead", source.address + 512);
        let mut reach = Reachability::new(
            &[source.clone(), live.clone(), dead.clone()],
            &[(source.address, source.address + 16)],
            &[],
            &[source.name.clone()],
        )
        .unwrap();
        let bytes = [
            dead.address.to_le_bytes().as_slice(),
            dead.address.to_le_bytes().as_slice(),
            b"padding",
            (live.address + 1).to_le_bytes().as_slice(),
        ]
        .concat();
        reach.contents(&source, 0, &bytes[..26]).unwrap();
        reach.contents(&source, 26, &bytes[26..]).unwrap();
        assert_eq!(reach.reached, HashSet::from([source.name, live.name]));
    }

    #[test]
    fn opaque_arguments_and_disjoint_snapshot_spans_do_not_bridge() {
        let source = allocation("source", 0x200000000000);
        let target = allocation("target", source.address + 256);
        let bytes = target.address.to_le_bytes();
        let mut reach =
            Reachability::new(&[source.clone(), target.clone()], &[], &[], &[]).unwrap();
        reach.blob(&bytes[..4], &[]).unwrap();
        reach.blob(&bytes[4..], &[]).unwrap();
        reach.contents(&source, 0, &bytes[..4]).unwrap();
        reach.contents(&source, 128, &bytes[4..]).unwrap();
        assert!(reach.reached.is_empty());
        reach
            .blob(
                &[b"abc".as_slice(), bytes.as_slice(), b"xyz".as_slice()].concat(),
                &[],
            )
            .unwrap();
        assert_eq!(reach.reached, HashSet::from([target.name]));
    }

    #[test]
    fn typed_scalar_and_pointer_parameters_keep_independent_roles() {
        let scalar = allocation("scalar_bits", 0x200000000000);
        let input = allocation("input", scalar.address + 256);
        let kernel = Kernel {
            function: 1,
            module: "module".into(),
            arguments: vec![
                super::super::Argument {
                    size: 8,
                    relocations: vec![],
                    numeric: vec![(0, 8)],
                },
                super::super::Argument {
                    size: 8,
                    relocations: vec![],
                    numeric: vec![],
                },
            ],
        };
        let mut node = wire::Kernel {
            function: 0,
            arguments: vec![
                wire::Blob {
                    data: scalar.address.to_le_bytes().to_vec(),
                    relocations: vec![],
                },
                wire::Blob {
                    data: input.address.to_le_bytes().to_vec(),
                    relocations: vec![],
                },
            ],
            ..Default::default()
        };
        let mut reach = Reachability::new(&[scalar.clone(), input.clone()], &[], &[], &[]).unwrap();
        reach
            .graph(
                &[wire::Node {
                    operation: Some(wire::node::Operation::Kernel(node.clone())),
                }],
                &[kernel.clone()],
            )
            .unwrap();
        assert_eq!(reach.reached, HashSet::from([input.name.clone()]));
        node.arguments[1].data = node.arguments[0].data.clone();
        reach
            .graph(
                &[wire::Node {
                    operation: Some(wire::node::Operation::Kernel(node)),
                }],
                &[kernel],
            )
            .unwrap();
        assert_eq!(reach.reached, HashSet::from([input.name, scalar.name]));
    }

    #[test]
    fn freed_interval_boundaries_exclude_candidates_but_not_explicit_bindings() {
        let mut storage = allocation("segment", 0x200000000000);
        storage.nbytes = 4096;
        for (offset, retained) in [
            (0, true),
            (1023, true),
            (1024, false),
            (2048, false),
            (3071, false),
            (3072, true),
        ] {
            let mut reach = Reachability::new(
                &[storage.clone()],
                &[],
                &[(storage.address + 1024, storage.address + 3072)],
                &[],
            )
            .unwrap();
            reach
                .blob(
                    &[
                        b"unaligned".as_slice(),
                        (storage.address + offset).to_le_bytes().as_slice(),
                    ]
                    .concat(),
                    &[],
                )
                .unwrap();
            assert_eq!(!reach.reached.is_empty(), retained);
            reach.reference(&storage.name).unwrap();
            assert!(reach.reached.contains(&storage.name));
        }
    }

    #[test]
    fn unaligned_numeric_token_ids_do_not_retain_address_shaped_bits() {
        let mut source = allocation("indices", 0x200000000000);
        source.nbytes = 256 * 8;
        let target = allocation("unrelated", 50 << 16);
        let bytes = (0..256u64).flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        let mut raw = Reachability::new(&[source.clone(), target.clone()], &[], &[], &[]).unwrap();
        raw.contents(&source, 0, &bytes).unwrap();
        assert!(raw.reached.contains(&target.name));
        let mut typed = Reachability::new(
            &[source.clone(), target],
            &[(source.address, source.address + source.nbytes)],
            &[],
            &[],
        )
        .unwrap();
        typed.contents(&source, 0, &bytes).unwrap();
        assert!(typed.reached.is_empty());
    }
}
