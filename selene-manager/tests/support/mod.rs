#![allow(dead_code, unused_imports)]
use flare_selene::shared::record::{Header, RankSizes};
use selene_manager::{
    Key, control,
    record::{self, Descriptor},
    store::Config,
};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

pub mod channel;
pub mod net;
pub mod pool;
mod put;
#[cfg(feature = "rdma")]
pub mod rdma;
pub mod store_client;

pub use store_client::*;

static SERIAL: AtomicU64 = AtomicU64::new(0);
pub fn key(tokens: u64) -> Key {
    Key {
        consumed_tokens: tokens,
        digest: (tokens as u128 * 7919).to_le_bytes(),
    }
}

pub fn descriptor() -> Descriptor {
    Descriptor {
        identity: [7; 32],
        fingerprint: [3; 16],
        page_tokens: 64,
        ranks: vec![RankSizes {
            planes: vec![4000],
            row_bytes: 1000,
        }],
        pool_name: "tenant".into(),
    }
}

pub fn base(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "selene-{}-{}-{label}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Staging for a test manager: small, since many run at once in a small `/dev/shm`.
pub const TEST_STAGING: usize = 16 * 1024 * 1024;

/// Removes a test manager's region files after it stops, keeping the lock inode.
pub struct ShmCleanup(PathBuf);

impl ShmCleanup {
    pub fn new(name: &str, socket: &Path) -> Self {
        let mut config = Config::dram_only(socket.to_path_buf(), 4096);
        config.name = name.to_owned();
        Self(config.shm_directory().unwrap())
    }
}

impl Drop for ShmCleanup {
    fn drop(&mut self) {
        selene_manager::testing::remove_regions(&self.0);
    }
}

/// Removes the shared-memory directories of a test child that was killed, which cannot clean up
/// after itself. They are named by the child's pid, which is gone.
pub fn remove_children_shm(pid: u32) {
    let prefix = format!("selene-{pid}-");
    if let Ok(entries) = std::fs::read_dir("/dev/shm/selene") {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                selene_manager::testing::remove_regions(&entry.path());
            }
        }
    }
}

pub fn instance_name(label: &str) -> String {
    format!(
        "selene-{}-{label}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    )
}

pub use selene_manager::testing::Manager as Host;
pub fn wait(condition: impl FnMut() -> bool) {
    selene_manager::testing::wait("manager test", condition);
}

pub trait HostClient {
    async fn client(&self) -> Client;
    async fn client_with(&self, descriptor: Descriptor) -> std::result::Result<Client, String>;
    fn durable_records(&self) -> usize;
    fn durable(&self, count: usize);
}
impl HostClient for Host {
    async fn client(&self) -> Client {
        Client::connect(&self.config, descriptor())
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "a client could not connect to the manager at {}: {error}",
                    self.config.socket.display()
                )
            })
    }

    async fn client_with(&self, descriptor: Descriptor) -> std::result::Result<Client, String> {
        Client::connect(&self.config, descriptor).await
    }

    fn durable_records(&self) -> usize {
        self.metrics()
            .lines()
            .filter(|line| line.starts_with("selene_records{") && line.contains("tier=\"disk\""))
            .filter_map(|line| line.rsplit(' ').next().unwrap().parse::<usize>().ok())
            .sum()
    }

    fn durable(&self, count: usize) {
        wait(|| {
            self.metrics()
                .lines()
                .filter(|line| {
                    line.starts_with("selene_records{") && line.contains("tier=\"disk\"")
                })
                .filter_map(|line| line.rsplit(' ').next().unwrap().parse::<usize>().ok())
                .sum::<usize>()
                == count
        });
    }
}

pub fn encode_header(header: &Header) -> [u8; 4096] {
    header.encode().unwrap()
}

pub fn content_for(
    descriptor: &Descriptor,
    key: Key,
    parent: Option<Key>,
    first: u32,
) -> (Header, Vec<u8>, Vec<u64>) {
    let pages = key.consumed_tokens / descriptor.page_tokens - u64::from(first);
    let mut header = selene_manager::record::new_header(descriptor, key, parent, [4; 16], pages as u32).unwrap();
    let mut bytes = vec![0; header.length_with_row() as usize];
    let page_bytes = header.page_record_bytes() as usize;
    let mut hashes = Vec::new();
    for page in 0..pages as usize {
        let offset = header.page_segment as usize + page * page_bytes;
        let data = &mut bytes[offset..offset + page_bytes];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = ((page + first as usize) * 31 + index * 7) as u8;
        }
        hashes.push(flare_selene::shared::record::checksum(data));
    }
    for (index, byte) in bytes[header.row_segment as usize..].iter_mut().enumerate() {
        *byte = (key.consumed_tokens as usize + index * 13) as u8;
    }
    header.row_checksum =
        flare_selene::shared::record::checksum(&bytes[header.row_segment as usize..]);
    bytes[..4096].copy_from_slice(&encode_header(&header));
    for (index, hash) in hashes.iter().enumerate() {
        bytes[4096 + index * 8..4096 + index * 8 + 8].copy_from_slice(&hash.to_le_bytes());
    }
    (header, bytes, hashes)
}

pub fn recovered_keys(root: &Path) -> Vec<Key> {
    selene_manager::disk::recover(root, "")
        .unwrap()
        .iter()
        .flat_map(|namespace| namespace.records.iter().map(|(header, _, _)| header.key))
        .collect()
}
