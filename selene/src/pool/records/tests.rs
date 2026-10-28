use super::*;
use crate::{pool::RankRecords, testing::geometry};
use std::{path::Path, time::SystemTime};

fn key(tokens: u64) -> PrefixKey {
    PrefixKey {
        consumed_tokens: tokens,
        digest: [tokens as u8; 16],
    }
}

fn write(
    directory: &Path,
    geometry: &[CacheGeometry],
    tokens: u64,
    parent: Option<PrefixKey>,
    values: &[u8],
) -> Header {
    let mut header = Header::new(
        geometry,
        [1; 32],
        [2; 16],
        key(tokens),
        parent,
        [0; 16],
        values.len() as u32,
        SystemTime::now(),
    )
    .unwrap();
    header.row_checksum = record::checksum(&vec![tokens as u8; 4096]);
    let mut bytes = vec![0; header.length_with_row() as usize];
    bytes[..HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
    for (page, &value) in values.iter().enumerate() {
        let at = header.page_segment as usize + page * 4096;
        bytes[at..at + 4096].fill(value);
        let sum = record::checksum(&bytes[at..at + 4096]);
        let at = header.checksum_segment() as usize + page * 8;
        bytes[at..at + 8].copy_from_slice(&sum.to_le_bytes());
    }
    bytes[header.row_segment as usize..].fill(tokens as u8);
    std::fs::write(directory.join(record::file_name(&header.key)), bytes).unwrap();
    header
}

#[test]
fn rank_records_read_slices_rows_and_chain_spills_with_persisted_sums() {
    let directory =
        std::env::temp_dir().join(format!("selene-pool-records-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
    let root = write(&directory, &geometry, 128, None, &[11, 22]);
    let tail = write(&directory, &geometry, 256, Some(root.key), &[33, 44]);
    let lock = crate::testing::directory_lock(&directory).unwrap();
    let mut records = DeferredRecords(Arc::new(Mutex::new(Some(Records {
        directory: directory.clone(),
        identity: [1; 32],
        fingerprint: [2; 16],
        layout: Layout::new(&geometry).unwrap(),
        geometry,

        _lock: lock,
    }))));
    let mut storage = vec![0; 4 * 4096];
    let offset = storage.as_ptr().align_offset(4096);
    let dst = &mut storage[offset..offset + 3 * 4096];
    let sums = records
        .read(root.key, 1, 1, true, &mut dst[..8192])
        .unwrap();
    assert_eq!(&dst[..4096], vec![22; 4096]);
    assert_eq!(&dst[4096..8192], vec![128; 4096]);
    assert_eq!(sums.pages, vec![record::checksum(&vec![22; 4096])]);
    assert_eq!(sums.row, Some(root.row_checksum));
    let sums = records.fill(tail.key, 1, 2, true, dst).unwrap();
    assert_eq!(&dst[..4096], vec![22; 4096]);
    assert_eq!(&dst[4096..8192], vec![33; 4096]);
    assert_eq!(&dst[8192..], vec![0; 4096]);
    assert_eq!(sums.pages.len(), 2);
    assert_eq!(sums.row, Some(tail.row_checksum));
    assert!(
        records
            .read(root.key, 2, 1, false, &mut dst[..4096])
            .is_err()
    );
    assert!(
        records
            .read(root.key, 0, 1, false, &mut dst[..12])
            .is_err()
    );
    // A damaged slice still serves the persisted checksum for the manager to verify.
    let path = directory.join(record::file_name(&root.key));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[root.page_segment as usize] ^= 1;
    std::fs::write(path, bytes).unwrap();
    let sums = records
        .read(root.key, 0, 1, false, &mut dst[..4096])
        .unwrap();
    assert_eq!(sums.pages, vec![record::checksum(&vec![11; 4096])]);
    assert_ne!(sums.pages[0], record::checksum(&dst[..4096]));
    drop(records);
    std::fs::remove_dir_all(directory).unwrap();
}
