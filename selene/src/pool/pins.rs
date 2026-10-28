//! Exports of host-tier holdings: pinned record chains that fill windows from
//! slabs in memory or from record files.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use flare_eos::FileReadAt;

use crate::{Error, PrefixKey, Result, disk::ReadAhead, geometry::BLOCK_BYTES, record::Header};

use super::{
    exporter::{Checksums, HostChain, HostPin, HostRecord, HostWindow},
    tiers::{RecordSums, Releases},
};

/// The records of an export: the nodes from the one holding the page that
/// contains `begin_tokens` to the endpoint, each with the pages and first page
/// it supplies. `nodes` are ascending `(key, tail pages)`; the
/// endpoint carries the row when `want_row`.
#[cfg(test)]
pub(crate) fn host_chain(
    page_tokens: u64,
    begin_tokens: u64,
    want_row: bool,
    nodes: &[(PrefixKey, usize)],
) -> HostChain {
    host_chain_kept(page_tokens, begin_tokens, want_row, nodes).0
}

/// Build the export chain and the positions of the records it kept, in order. A
/// node that supplies no page is not a record unless it is the last, so whatever
/// is held per node must be cut to these before windows name records by index.
pub(crate) fn host_chain_kept(
    page_tokens: u64,
    begin_tokens: u64,
    want_row: bool,
    nodes: &[(PrefixKey, usize)],
) -> (HostChain, Vec<usize>) {
    let start_page = begin_tokens / page_tokens;
    let mut records = Vec::new();
    let mut kept = Vec::new();
    for (index, (key, tail)) in nodes.iter().enumerate() {
        let end = key.consumed_tokens / page_tokens;
        let start = end.saturating_sub(*tail as u64);
        let from = start.max(start_page);
        let pages = end.saturating_sub(from);
        let last = index + 1 == nodes.len();
        if pages == 0 && !last {
            continue;
        }
        kept.push(index);
        records.push(HostRecord {
            key: *key,
            first_page: (from - start) as u32,
            pages: pages as u32,
            row: last && want_row,
        });
    }
    let chain = HostChain {
        total_pages: records.iter().map(|r| u64::from(r.pages)).sum(),
        row: want_row,
        records,
    };
    (chain, kept)
}

fn check_window(
    chain: &HostChain,
    window: &HostWindow,
    dst: &[u8],
    page: u64,
    row: u64,
) -> Result<()> {
    let record = chain
        .records
        .get(window.record as usize)
        .ok_or_else(|| Error::new("window names no record"))?;
    let end = record.first_page + record.pages;
    if window.first_page < record.first_page
        || window.first_page + window.page_count > end
        || (window.row && !record.row)
        || dst.len() as u64
            != u64::from(window.page_count) * page + if window.row { row } else { 0 }
    {
        return Err(Error::new("window does not fit the pinned record"));
    }
    Ok(())
}

/// One pinned record in memory.
pub(crate) struct MemorySpan {
    /// Addresses of every tail page of the record.
    pub pages: Vec<usize>,
    pub row: Option<usize>,
    pub sums: Arc<RecordSums>,
}

/// A chain of records in registered host memory, held in place by pins that
/// the store releases once this is dropped.
pub struct MemoryPin {
    pub(crate) chain: HostChain,
    pub(crate) spans: Vec<MemorySpan>,
    pub(crate) page_bytes: usize,
    pub(crate) row_bytes: usize,
    pub(crate) keys: Vec<PrefixKey>,
    pub(crate) releases: Releases,
}

impl HostPin for MemoryPin {
    fn chain(&self) -> &HostChain {
        &self.chain
    }

    fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums> {
        check_window(
            &self.chain,
            window,
            dst,
            self.page_bytes as u64,
            self.row_bytes as u64,
        )?;
        let span = &self.spans[window.record as usize];
        let mut offset = 0;
        let mut pages = Vec::with_capacity(window.page_count as usize);
        for page in window.first_page..window.first_page + window.page_count {
            let source = span.pages[page as usize];
            // The record is pinned, so its slab is mapped and unchanged.
            let bytes = unsafe { std::slice::from_raw_parts(source as *const u8, self.page_bytes) };
            dst[offset..offset + self.page_bytes].copy_from_slice(bytes);
            offset += self.page_bytes;
            pages.push(span.sums.pages[page as usize]);
        }
        let row = if window.row {
            let source = span
                .row
                .ok_or_else(|| Error::new("the record has no row"))?;
            let bytes = unsafe { std::slice::from_raw_parts(source as *const u8, self.row_bytes) };
            dst[offset..offset + self.row_bytes].copy_from_slice(bytes);
            span.sums.row
        } else {
            None
        };
        Ok(Checksums { pages, row })
    }
}

impl Drop for MemoryPin {
    fn drop(&mut self) {
        self.releases.push(&self.keys);
    }
}

/// Records of fewer bytes than this are read through the page cache.
const MIN_AHEAD_BYTES: u64 = 2 * 4 << 20;

/// One pinned record file.
pub(crate) struct FileSpan {
    pub path: PathBuf,
    pub header: Header,
    open: Option<OpenSpan>,
}

impl FileSpan {
    pub(crate) fn new(path: PathBuf, header: Header) -> Self {
        Self {
            path,
            header,
            open: None,
        }
    }
}

/// A span's file as the export reads it.
struct OpenSpan {
    /// Page-cache reads, for the checksums and whatever cannot be read directly.
    file: File,
    /// The pages' read-ahead, then the row's, from the file past the page cache, if the
    /// filesystem allows it and the range is aligned for it.
    ahead: Option<ReadAhead>,
}

/// A chain of record files, held in place by pins that the store releases once
/// this is dropped. Pages are read from the files and the checksums served are
/// the ones the files persist.
///
/// A cold record read through the page cache a window at a time runs at a fraction of the
/// device's rate: readahead is a few requests deep, and each window waits for the device
/// before the next is asked for. The pages of a record are therefore read ahead directly
/// from the device, the same way the disk tier restores them, with several blocks in flight
/// while earlier windows are copied out. The row, last in the record, is read the same way.
pub struct FilePin {
    pub(crate) chain: HostChain,
    pub(crate) spans: Vec<FileSpan>,
    pub(crate) page_bytes: u64,
    pub(crate) row_bytes: u64,
    pub(crate) keys: Vec<PrefixKey>,
    pub(crate) releases: Releases,
}

impl HostPin for FilePin {
    fn chain(&self) -> &HostChain {
        &self.chain
    }

    fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums> {
        check_window(&self.chain, window, dst, self.page_bytes, self.row_bytes)?;
        let (record_first, record_pages) = {
            let record = &self.chain.records[window.record as usize];
            (record.first_page, record.pages)
        };
        let span = &mut self.spans[window.record as usize];
        let path = span.path.clone();
        let io = |what: &str, e: std::io::Error| {
            Error::caused_by(format!("{what} {}", path.display()), e)
        };
        if span.open.is_none() {
            let file = File::open(&span.path).map_err(|e| io("cannot open", e))?;
            span.open = Some(OpenSpan { file, ahead: None });
        }
        let page_bytes = self.page_bytes;
        let count = window.page_count as usize;
        let pages_bytes = count * page_bytes as usize;
        let first = u64::from(window.first_page);
        let pages_at = span.header.page_segment + first * page_bytes;
        let header = span.header.clone();
        let open = span.open.as_mut().expect("opened");
        if count > 0 {
            let end = header.page_segment + u64::from(record_first + record_pages) * page_bytes;
            // Reading ahead starts at the first window read of a record and runs to the end
            // of the pages this export takes. It pays for the thread and ring only on a
            // record of several blocks; a window that does not follow the last, a record
            // too small to gain from it and a filesystem that refuses direct I/O are read
            // through the cache.
            read_range(&span.path, open, &mut dst[..pages_bytes], pages_at, end)
                .map_err(|e| io("cannot read pages of", e))?;
        }
        let mut stored = vec![0u8; count * 8];
        open.file
            .read_exact_at(&mut stored, header.checksum_segment() + first * 8)
            .map_err(|e| io("cannot read checksums of", e))?;
        let pages = stored
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("eight bytes")))
            .collect();
        let row = if window.row {
            // The row, often 100 MiB or more, is read ahead like the pages: drop the pages'
            // read-ahead, which ends at their last page, so the row gets its own.
            let at = header.row_segment;
            open.ahead = None;
            let dst = &mut dst[pages_bytes..pages_bytes + self.row_bytes as usize];
            read_range(&span.path, open, dst, at, at + self.row_bytes)
                .map_err(|e| io("cannot read the row of", e))?;
            // The persisted row checksum is in the file's own header.
            let mut block = vec![0u8; crate::record::HEADER_BYTES];
            open.file
                .read_exact_at(&mut block, 0)
                .map_err(|e| io("cannot read the header of", e))?;
            Some(Header::decode(&block)?.row_checksum)
        } else {
            None
        };
        // The record's pages are all served: its file, thread and ring are not kept for the
        // rest of the export. A chain of many records otherwise holds a descriptor or two and
        // a thread for each until the export ends.
        if window.first_page + window.page_count >= record_first + record_pages {
            self.spans[window.record as usize].open = None;
        }
        Ok(Checksums { pages, row })
    }
}

/// Reads `dst` from `at` in the open file, ahead of the consumer past the page cache when the
/// range `at..end` is block aligned and long enough, and through the cache otherwise.
fn read_range(
    path: &Path,
    open: &mut OpenSpan,
    dst: &mut [u8],
    at: u64,
    end: u64,
) -> std::io::Result<()> {
    let direct = at.is_multiple_of(BLOCK_BYTES)
        && end.is_multiple_of(BLOCK_BYTES)
        && end - at >= MIN_AHEAD_BYTES;
    if open.ahead.is_none() && direct {
        open.ahead = flare_eos::direct_file(path, false)
            .ok()
            .and_then(|direct| ReadAhead::start(direct, at, end).ok());
    }
    match open.ahead.as_mut().filter(|ahead| ahead.at(at)) {
        Some(ahead) => ahead
            .read(dst)
            .map_err(|e| std::io::Error::other(e.to_string())),
        None => {
            open.ahead = None;
            open.file.read_exact_at(dst, at)
        }
    }
}

impl Drop for FilePin {
    fn drop(&mut self) {
        self.releases.push(&self.keys);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(tokens: u64, salt: u8) -> PrefixKey {
        PrefixKey {
            consumed_tokens: tokens,
            digest: [salt; 16],
        }
    }

    #[test]
    fn a_node_with_no_page_is_not_a_record_and_the_kept_positions_say_which_nodes_are() {
        // b ends in the page a ended in, so it holds none; c holds pages 50..100.
        let nodes = [(key(800, 1), 50), (key(805, 2), 0), (key(1600, 3), 50)];
        let (chain, kept) = host_chain_kept(16, 0, true, &nodes);
        assert_eq!(kept, [0, 2]);
        assert_eq!(
            chain.records.iter().map(|r| r.key).collect::<Vec<_>>(),
            [key(800, 1), key(1600, 3)]
        );
        // The last node stays whatever it supplies: its row is the claim.
        let tail = [(key(800, 1), 50), (key(805, 2), 0)];
        let (chain, kept) = host_chain_kept(16, 0, true, &tail);
        assert_eq!(kept, [0, 1]);
        assert!(chain.records[1].row && chain.records[1].pages == 0);
    }

    #[test]
    fn records_start_where_the_split_falls_and_a_split_in_the_last_page_leaves_a_row_only_tail() {
        // Pages of 16 tokens: a covers pages 0..50, b pages 50..100.
        let nodes = [(key(800, 1), 50), (key(1600, 2), 50)];
        let chain = host_chain(16, 0, true, &nodes);
        assert_eq!(chain.total_pages, 100);
        assert_eq!(chain.records[0].first_page, 0);
        assert!(!chain.records[0].row && chain.records[1].row);
        // Starting inside b skips a entirely.
        let chain = host_chain(16, 1000, true, &nodes);
        assert_eq!(chain.records.len(), 1);
        assert_eq!(chain.records[0].key, key(1600, 2));
        assert_eq!(
            (chain.records[0].first_page, chain.records[0].pages),
            (12, 38)
        );
        // Starting at the page the endpoint ends in: its row alone.
        let chain = host_chain(16, 1600, true, &nodes);
        assert_eq!(chain.total_pages, 0);
        assert_eq!(
            (chain.records[0].first_page, chain.records[0].pages),
            (50, 0)
        );
        assert!(chain.records[0].row);
        // Without the row, no record carries one.
        assert!(
            host_chain(16, 0, false, &nodes)
                .records
                .iter()
                .all(|r| !r.row)
        );
    }

    #[test]
    fn a_memory_pin_copies_pages_and_serves_the_checksums_it_was_given_and_releases_on_drop() {
        let page = 64usize;
        let data: Vec<u8> = (0..3 * page + 32).map(|i| (i * 7) as u8).collect();
        let base = data.as_ptr() as usize;
        let releases = Releases::default();
        let sums = Arc::new(RecordSums {
            pages: vec![11, 22, 33],
            row: Some(44),
        });
        let nodes = [(key(48, 1), 3)];
        let mut pin = MemoryPin {
            chain: host_chain(16, 16, true, &nodes),
            spans: vec![MemorySpan {
                pages: vec![base, base + page, base + 2 * page],
                row: Some(base + 3 * page),
                sums,
            }],
            page_bytes: page,
            row_bytes: 32,
            keys: vec![key(48, 1)],
            releases: releases.clone(),
        };
        assert_eq!(
            (
                pin.chain().records[0].first_page,
                pin.chain().records[0].pages
            ),
            (1, 2)
        );
        let window = HostWindow {
            record: 0,
            first_page: 1,
            page_count: 2,
            row: true,
        };
        let mut dst = vec![0u8; 2 * page + 32];
        let sums = pin.read(&window, &mut dst).unwrap();
        assert_eq!(dst, data[page..]);
        assert_eq!((sums.pages, sums.row), (vec![22, 33], Some(44)));
        // A window outside what was pinned is refused.
        let outside = HostWindow {
            record: 0,
            first_page: 0,
            page_count: 1,
            row: false,
        };
        assert!(pin.read(&outside, &mut vec![0u8; page]).is_err());
        assert!(pin.read(&window, &mut [0u8; 5]).is_err());
        drop(pin);
        assert_eq!(releases.drain(), [key(48, 1)]);
    }

    /// Writes a record file: the header, `pages` pages of `pattern` with their checksums, the row.
    fn write_record(
        directory: &Path,
        header: &Header,
        page: usize,
        pages: u32,
        pattern: &dyn Fn(u32) -> Vec<u8>,
        row: &[u8],
    ) -> PathBuf {
        let mut file = vec![0u8; header.length_with_row() as usize];
        file[..crate::record::HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
        for p in 0..pages {
            let at = header.page_segment as usize + p as usize * page;
            file[at..at + page].copy_from_slice(&pattern(p));
            let sum = header.checksum_segment() as usize + p as usize * 8;
            file[sum..sum + 8].copy_from_slice(&crate::record::checksum(&pattern(p)).to_le_bytes());
        }
        let at = header.row_segment as usize;
        file[at..at + row.len()].copy_from_slice(row);
        let path = directory.join("record");
        std::fs::write(&path, &file).unwrap();
        path
    }

    #[test]
    fn a_file_pin_serves_windows_in_order_and_out_of_order_with_the_persisted_checksums() {
        use crate::testing::geometry;
        let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = base.join(format!("flare-file-pin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let geometry = [geometry(16, &[("k", 4096), ("v", 4096)], 4096)];
        let pages = 40u32;
        let mut header = Header::new(
            &geometry,
            [1; 32],
            [2; 16],
            key(16 * u64::from(pages), 9),
            None,
            [3; 16],
            pages,
            std::time::SystemTime::now(),
        )
        .unwrap();
        let page = 8192usize;
        let pattern =
            |p: u32| -> Vec<u8> { (0..page).map(|i| (i as u32 * 3 + p * 11) as u8).collect() };
        let stored_row = vec![5u8; 4096];
        header.row_checksum = crate::record::checksum(&stored_row);
        let path = write_record(&directory, &header, page, pages, &pattern, &stored_row);
        // The export starts at page 4 and takes the row.
        let nodes = [(header.key, pages as usize)];
        let mut pin = FilePin {
            chain: host_chain(16, 4 * 16, true, &nodes),
            spans: vec![FileSpan::new(path, header.clone())],
            page_bytes: page as u64,
            row_bytes: 4096,
            keys: vec![header.key],
            releases: Releases::default(),
        };
        // Windows of 5 pages in order, a window skipped (read later), then the tail.
        let mut check = |first: u32, count: u32, row: bool| {
            let mut dst = vec![0u8; count as usize * page + if row { 4096 } else { 0 }];
            let window = HostWindow {
                record: 0,
                first_page: first,
                page_count: count,
                row,
            };
            let sums = pin.read(&window, &mut dst).unwrap();
            for i in 0..count {
                let at = i as usize * page;
                assert_eq!(
                    dst[at..at + page],
                    pattern(first + i)[..],
                    "page {}",
                    first + i
                );
                assert_eq!(
                    sums.pages[i as usize],
                    crate::record::checksum(&pattern(first + i))
                );
            }
            if row {
                assert_eq!(dst[count as usize * page..], stored_row[..]);
                assert_eq!(sums.row, Some(crate::record::checksum(&stored_row)));
            }
        };
        check(4, 5, false);
        check(9, 5, false);
        check(19, 5, false);
        check(14, 5, false);
        check(24, 16, true);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_row_of_several_blocks_read_ahead_directly_follows_pages_read_ahead_in_the_same_window() {
        let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = base.join(format!("flare-file-pin-row-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        // 1024 pages of 8 KiB are MIN_AHEAD_BYTES: they are read ahead, then the row is.
        let (page, pages) = (8192usize, 1024u32);
        let row_bytes = 3 * MIN_AHEAD_BYTES as usize;
        let geometry = [crate::testing::geometry(
            16,
            &[("k", 4096), ("v", 4096)],
            row_bytes as u64,
        )];
        let mut header = Header::new(
            &geometry,
            [1; 32],
            [2; 16],
            key(16 * u64::from(pages), 9),
            None,
            [3; 16],
            pages,
            std::time::SystemTime::now(),
        )
        .unwrap();
        let pattern =
            |p: u32| -> Vec<u8> { (0..page).map(|i| (i as u32 * 3 + p * 11) as u8).collect() };
        let stored_row: Vec<u8> = (0..row_bytes).map(|i| (i * 7 + i / 4093) as u8).collect();
        header.row_checksum = crate::record::checksum(&stored_row);
        let path = write_record(&directory, &header, page, pages, &pattern, &stored_row);
        let nodes = [(header.key, pages as usize)];
        let mut pin = FilePin {
            chain: host_chain(16, 0, true, &nodes),
            spans: vec![FileSpan::new(path, header.clone())],
            page_bytes: page as u64,
            row_bytes: row_bytes as u64,
            keys: vec![header.key],
            releases: Releases::default(),
        };
        let mut dst = vec![0u8; pages as usize * page + row_bytes];
        let window = HostWindow {
            record: 0,
            first_page: 0,
            page_count: pages,
            row: true,
        };
        let sums = pin.read(&window, &mut dst).unwrap();
        for p in 0..pages {
            let at = p as usize * page;
            assert!(dst[at..at + page] == pattern(p)[..], "page {p}");
            assert_eq!(sums.pages[p as usize], crate::record::checksum(&pattern(p)));
        }
        assert!(dst[pages as usize * page..] == stored_row[..]);
        assert_eq!(sums.row, Some(crate::record::checksum(&stored_row)));
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_pin_over_many_records_keeps_open_only_the_record_it_is_reading() {
        let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = base.join(format!("flare-file-pin-many-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let geometry = [crate::testing::geometry(
            16,
            &[("k", 4096), ("v", 4096)],
            4096,
        )];
        let page = 8192u64;
        // One record file stands for every record of a long chain; a record of 1500 pages
        // (12 MiB) is read ahead, the small ones are not.
        let shape = |pages: u32| {
            let mut header = Header::new(
                &geometry,
                [1; 32],
                [2; 16],
                key(16 * u64::from(pages), 9),
                None,
                [3; 16],
                pages,
                std::time::SystemTime::now(),
            )
            .unwrap();
            header.row_checksum = 0;
            let mut file = vec![0u8; header.length_with_row() as usize];
            file[..crate::record::HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
            (header, file)
        };
        let (small, small_file) = shape(4);
        let (large, large_file) = shape(1500);
        let small_path = directory.join("small");
        let large_path = directory.join("large");
        std::fs::write(&small_path, small_file).unwrap();
        std::fs::write(&large_path, large_file).unwrap();
        let mut records = Vec::new();
        let mut spans = Vec::new();
        for index in 0..400u32 {
            let (path, header, pages) = if index % 100 == 99 {
                (&large_path, &large, 1500)
            } else {
                (&small_path, &small, 4)
            };
            records.push(HostRecord {
                key: key(1000 + u64::from(index), 1),
                first_page: 0,
                pages,
                row: false,
            });
            spans.push(FileSpan::new(path.clone(), header.clone()));
        }
        let mut pin = FilePin {
            chain: HostChain {
                total_pages: records.iter().map(|r| u64::from(r.pages)).sum(),
                row: false,
                records,
            },
            spans,
            page_bytes: page,
            row_bytes: 4096,
            keys: Vec::new(),
            releases: Releases::default(),
        };
        let mut dst = vec![0u8; 500 * page as usize];
        for record in 0..400u32 {
            let pages = pin.chain.records[record as usize].pages;
            let mut first = 0;
            while first < pages {
                let count = (pages - first).min(500);
                let window = HostWindow {
                    record,
                    first_page: first,
                    page_count: count,
                    row: false,
                };
                pin.read(&window, &mut dst[..count as usize * page as usize])
                    .unwrap();
                first += count;
                let open = pin.spans.iter().filter(|span| span.open.is_some()).count();
                assert!(open <= 1, "{open} records held open at record {record}");
            }
            assert!(
                pin.spans.iter().all(|span| span.open.is_none()),
                "record {record} was left open once its pages were read"
            );
        }
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
