use super::super::Plane;
use super::*;
use crate::{
    cache::{Fence, PageId, PrefixCheckpointCache, Publication, SnapshotId, StateSnapshot},
    runtime_package::driver::Driver,
    scheduler::StateSlotId,
};
use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Child, Command},
    time::SystemTime,
};

const IDENTITY: [u8; 32] = [0x63; 32];

struct Manager {
    process: Child,
}

impl Manager {
    fn start(base: &Path, memory: &str) -> Self {
        Self::launch(base, memory, "16M", true, None)
    }

    fn start_dram(base: &Path, memory: &str) -> Self {
        Self::launch(base, memory, "32M", false, None)
    }

    fn start_budget(base: &Path, memory: &str, budget: &str) -> Self {
        Self::launch(base, memory, "32M", false, Some(budget))
    }

    fn launch(base: &Path, memory: &str, staging: &str, root: bool, budget: Option<&str>) -> Self {
        fs::create_dir_all(base).unwrap();
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(base.join("manager.log"))
            .unwrap();
        let mut command = Command::new(std::env::var_os("FLARE_POOL_TEST_MANAGER").unwrap());
        command
            .args(["--name"])
            .arg(format!("pool-test-{}", std::process::id()))
            .args(["--socket"])
            .arg(base.join("manager.sock"));
        if root {
            command
                .args(["--root"])
                .arg(base.join("records"))
                .args(["--nvme", "1G"]);
        }
        if let Some(budget) = budget {
            command.args(["--write-budget", budget]);
        }
        let mut process = command
            .args(["--memory", memory, "--staging", staging])
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(process.try_wait().unwrap().is_none(), "manager exited");
            if UnixStream::connect(base.join("manager.sock")).is_ok() {
                break;
            }
            assert!(Instant::now() < deadline, "manager startup timeout");
            thread::sleep(Duration::from_millis(10));
        }
        Self { process }
    }

    fn kill(mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

struct Gpu {
    driver: Driver,
    context: usize,
    geometry: Vec<CacheGeometry>,
}

impl Gpu {
    fn new() -> Self {
        let driver = Driver::load().unwrap();
        let mut context = std::ptr::null_mut();
        let mut device = 0;
        // SAFETY: Valid CUDA driver arguments; Gpu owns the context and allocations.
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            driver
                .check(
                    "cuCtxCreate_v2",
                    (driver.context_create)(&mut context, 0, device),
                )
                .unwrap();
        }
        let allocate = |bytes| {
            let mut address = 0;
            unsafe {
                driver
                    .check("cuMemAlloc_v2", (driver.allocate)(&mut address, bytes))
                    .unwrap();
            }
            address
        };
        let planes = [("k", 65_539u64), ("v", 4_100)]
            .into_iter()
            .map(|(name, unit_bytes)| Plane {
                name: name.into(),
                address: allocate(unit_bytes as usize * 600),
                unit_bytes,
            })
            .collect();
        let geometry = vec![CacheGeometry {
            page_tokens: 64,
            page_capacity: 600,
            row_capacity: 8,
            row: Plane {
                name: "state".into(),
                address: allocate(1_000_003 * 8),
                unit_bytes: 1_000_003,
            },
            planes,
            snapshot_pages: Vec::new(),
        }];
        Self {
            driver,
            context: context as usize,
            geometry,
        }
    }

    fn bytes(plane: usize, page: u32, length: usize) -> Vec<u8> {
        (0..length)
            .map(|i| {
                ((i as u64 * 31) ^ (i as u64 >> 8) ^ ((plane as u64 + 1) * 7919 + u64::from(page)))
                    as u8
            })
            .collect()
    }

    fn write(&self, address: u64, bytes: &[u8]) {
        unsafe {
            self.driver
                .check(
                    "cuMemcpyHtoD_v2",
                    (self.driver.upload)(address, bytes.as_ptr().cast(), bytes.len()),
                )
                .unwrap();
        }
    }

    fn fill(&self, pages: usize, slot: u32) {
        for (p, plane) in self.geometry[0].planes.iter().enumerate() {
            for page in 0..pages as u32 {
                self.write(
                    plane.unit_address(u64::from(page)),
                    &Self::bytes(p, page, plane.unit_bytes as usize),
                );
            }
        }
        let row = &self.geometry[0].row;
        self.write(
            row.unit_address(u64::from(slot)),
            &Self::bytes(2, slot, row.unit_bytes as usize),
        );
    }

    fn check(&self, pages: usize, source_slot: u32) {
        for (p, plane) in self.geometry[0]
            .planes
            .iter()
            .chain([&self.geometry[0].row])
            .enumerate()
        {
            let count = if p == 2 { 1 } else { pages };
            for page in 0..count as u32 {
                let source = if p == 2 { source_slot } else { page };
                let target = if p == 2 { 7 } else { 300 + page };
                let mut actual = vec![0; plane.unit_bytes as usize];
                unsafe {
                    self.driver
                        .check(
                            "cuMemcpyDtoH_v2",
                            (self.driver.download)(
                                actual.as_mut_ptr().cast(),
                                plane.unit_address(u64::from(target)),
                                actual.len(),
                            ),
                        )
                        .unwrap();
                }
                assert_eq!(
                    actual,
                    Self::bytes(p, source, actual.len()),
                    "plane {p}, page {page}"
                );
                self.write(
                    plane.unit_address(u64::from(target)),
                    &vec![0xDD; actual.len()],
                );
            }
        }
    }

    fn open(&self, socket: PathBuf, tenant: &str) -> PoolStore {
        let layout = Layout::new(&self.geometry).unwrap();
        let mut pool = SlabPool::staging(&layout).unwrap();
        pool.register(self.context).unwrap();
        let transfers = Arc::new(Mutex::new(
            Transfers::start(&[(self.context, 0)], self.geometry.clone(), &pool).unwrap(),
        ));
        PoolStore::new(
            socket,
            tenant.to_owned(),
            IDENTITY,
            super::super::geometry::fingerprint(&self.geometry),
            self.geometry.clone(),
            pool,
            transfers,
        )
        .unwrap()
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            (self.driver.context_set)(self.context as _);
            for plane in self.geometry[0]
                .planes
                .iter()
                .chain([&self.geometry[0].row])
            {
                (self.driver.free)(plane.address);
            }
            (self.driver.context_destroy)(self.context as _);
        }
    }
}

fn until(mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !check() {
        assert!(Instant::now() < deadline, "L1 operation timeout");
        thread::sleep(Duration::from_millis(1));
    }
}

fn ready(store: &mut PoolStore) {
    until(|| {
        store.poll(&mut Vec::new()).unwrap();
        store.shared.connected.load(Ordering::Acquire) && store.shared.ready.load(Ordering::Acquire)
    });
}

fn event(store: &mut PoolStore) -> StoreEvent {
    let mut events = Vec::new();
    until(|| {
        store.poll(&mut events).unwrap();
        !events.is_empty()
    });
    assert_eq!(events.len(), 1);
    assert_eq!(
        (
            store.stats().outbound_in_flight,
            store.stats().inbound_in_flight
        ),
        (0, 0)
    );
    assert!(store.pending.is_empty());
    events.pop().unwrap()
}

fn history() -> PrefixHistory {
    let mut history = PrefixHistory::new(IDENTITY);
    history
        .append(&(0..20_000).collect::<Vec<_>>(), &[])
        .unwrap();
    history
}

fn checkpoint(tokens: usize, parent: Option<usize>, slot: u32) -> PublishedCheckpoint {
    let history = history();
    let mut cache = PrefixCheckpointCache::new(IDENTITY, 600, 8, 64).unwrap();
    let id = cache
        .publish(
            &history,
            &Publication {
                parent: None,
                consumed_tokens: tokens as u64,
                tail_pages: (0..tokens as u32 / 64).map(PageId).collect(),
                snapshot: StateSnapshot {
                    id: SnapshotId(slot),
                    consumed_tokens: tokens as u64,
                },
                ready_after: Fence(0),
            },
        )
        .unwrap()
        .id();
    PublishedCheckpoint {
        id,
        key: history.key_at(tokens as u64).unwrap(),
        parent: parent.map(|p| history.key_at(p as u64).unwrap()),
        bucket: history.bucket_at(tokens as u64).unwrap(),
        chain_pages: (0..tokens as u32 / 64).map(PageId).collect(),
        tail_pages: tokens / 64 - parent.unwrap_or(0) / 64,
        slot: StateSlotId(slot),
    }
}

fn put(store: &mut PoolStore, checkpoint: &PublishedCheckpoint) {
    ready(store);
    assert_eq!(store.store(checkpoint).unwrap(), StoreOutcome::Queued);
    match event(store) {
        StoreEvent::Stored { result, .. } => {
            result.unwrap();
        }
        other => panic!("{other:?}"),
    }
}

fn hit(store: &mut PoolStore, checkpoint: &PublishedCheckpoint, tier: Tier) -> HostHit {
    let mut found = None;
    until(|| {
        store.poll(&mut Vec::new()).unwrap();
        found = store.lookup(&history(), 0, checkpoint.key.consumed_tokens + 1);
        found
            .as_ref()
            .is_some_and(|hit| hit.key == checkpoint.key && hit.tier == tier)
    });
    found.unwrap()
}

fn fetch(store: &mut PoolStore, checkpoint: &PublishedCheckpoint, tier: Tier, success: bool) {
    let hit = hit(store, checkpoint, tier);
    let pages = checkpoint.chain_pages.len();
    store
        .fetch(
            hit,
            InboundTicket {
                key: checkpoint.key,
                pages: (300..300 + pages as u32).map(PageId).collect(),
                slot: StateSlotId(7),
            },
        )
        .unwrap();
    match event(store) {
        StoreEvent::Fetched { result, .. } => assert_eq!(result.is_ok(), success, "{result:?}"),
        other => panic!("{other:?}"),
    }
}

fn put_later(store: &mut PoolStore, checkpoint: &PublishedCheckpoint) {
    until(|| match store.store(checkpoint).unwrap() {
        StoreOutcome::Queued => true,
        StoreOutcome::Retry(StoreReason::Budget) => {
            store.poll(&mut Vec::new()).unwrap();
            thread::sleep(Duration::from_millis(20));
            false
        }
        other => panic!("{other:?}"),
    });
    match event(store) {
        StoreEvent::Stored { result, .. } => {
            result.unwrap();
        }
        other => panic!("{other:?}"),
    }
}

fn record_files(base: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for directory in fs::read_dir(base.join("records")).unwrap().flatten() {
        if !directory.file_type().unwrap().is_dir() {
            continue;
        }
        for entry in fs::read_dir(directory.path()).unwrap().flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|extension| extension == "rec") {
                files.push(path);
            }
        }
    }
    files
}

fn corrupt(path: &Path) {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let end = file.metadata().unwrap().len() - 1;
    file.seek(SeekFrom::Start(end)).unwrap();
    let mut byte = [0u8];
    file.read_exact(&mut byte).unwrap();
    file.seek(SeekFrom::Start(end)).unwrap();
    file.write_all(&[byte[0] ^ 0x80]).unwrap();
    file.sync_all().unwrap();
}

fn base(tag: &str) -> PathBuf {
    let base = PathBuf::from(std::env::var_os("FLARE_POOL_TEST_DIR").unwrap()).join(format!(
        "l1-{tag}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&base).unwrap();
    base
}

#[test]
#[ignore = "requires GPU 0, FLARE_POOL_TEST_MANAGER and FLARE_POOL_TEST_DIR on local NVMe"]
fn real_manager_rejects_corruption_and_stays_usable() {
    let base = base("corrupt");
    let gpu = Gpu::new();
    let first = checkpoint(8 * 64 + 3, None, 1);
    let second = checkpoint(16 * 64 + 5, None, 2);
    gpu.fill(16, 1);
    gpu.fill(16, 2);
    // A tiny DRAM tier keeps both records on NVMe so a separate handle can
    // damage one after its Put.
    let manager = Manager::start(&base, "4K");
    let mut store = gpu.open(base.join("manager.sock"), "");
    put(&mut store, &first);
    let files = record_files(&base);
    assert_eq!(files.len(), 1);
    corrupt(&files[0]);
    fetch(&mut store, &first, Tier::Disk, false);
    // The damaged record fails the operation, not the manager: a later
    // publication and restore on the reconnected session succeed.
    put(&mut store, &second);
    fetch(&mut store, &second, Tier::Disk, true);
    gpu.check(16, 2);
    store.close().unwrap();
    drop(store);
    drop(manager);
}

#[test]
#[ignore = "requires GPU 0, FLARE_POOL_TEST_MANAGER and FLARE_POOL_TEST_DIR on local NVMe"]
fn real_manager_interrupted_transfers_recover_on_a_new_generation() {
    let base = base("interrupt");
    let gpu = Gpu::new();
    let root = checkpoint(70 * 64 + 3, None, 1);
    let child = checkpoint(140 * 64 + 5, Some(70 * 64 + 3), 2);
    gpu.fill(140, 1);
    gpu.fill(140, 2);
    let manager = Manager::start(&base, "128M");
    let mut store = gpu.open(base.join("manager.sock"), "");
    put(&mut store, &root);
    put(&mut store, &child);

    let hit = hit(&mut store, &child, Tier::Memory);
    let pages = hit.pages;
    let pause = store.pause_inbound_guard();
    store
        .fetch(
            hit,
            InboundTicket {
                key: child.key,
                pages: (300..300 + pages as u32).map(PageId).collect(),
                slot: StateSlotId(7),
            },
        )
        .unwrap();
    // Two restored windows are held in flight; a further grant keeps the
    // manager from finishing the chain while it is killed.
    until(|| {
        store.poll(&mut Vec::new()).unwrap();
        store.paused_transfers() >= 2
    });
    manager.kill();
    pause.release();
    assert!(matches!(
        event(&mut store),
        StoreEvent::Fetched { result: Err(_), .. }
    ));

    let manager = Manager::start(&base, "4K");
    ready(&mut store);
    fetch(&mut store, &root, Tier::Disk, true);
    gpu.check(70, 1);

    let newer = checkpoint(210 * 64 + 7, None, 3);
    gpu.fill(210, 3);
    let pause = store.pause_outbound_guard();
    assert_eq!(store.store(&newer).unwrap(), StoreOutcome::Queued);
    // Kill with the first two write windows already in flight.
    until(|| {
        store.poll(&mut Vec::new()).unwrap();
        store.paused_transfers() >= 2
    });
    manager.kill();
    pause.release();
    assert!(matches!(event(&mut store), StoreEvent::Deferred { .. }));

    let manager = Manager::start(&base, "4K");
    put(&mut store, &newer);
    fetch(&mut store, &newer, Tier::Disk, true);
    gpu.check(210, 3);
    store.close().unwrap();
    drop(store);
    drop(manager);
}

#[test]
#[ignore = "requires GPU 0, FLARE_POOL_TEST_MANAGER and FLARE_POOL_TEST_DIR on local NVMe"]
fn real_manager_pauses_a_window_owner_while_another_client_progresses() {
    let base = base("paused");
    let gpu = Gpu::new();
    let owner = checkpoint(140 * 64 + 3, None, 1);
    let waiting = checkpoint(140 * 64 + 5, None, 2);
    gpu.fill(140, 1);
    gpu.fill(140, 2);
    // One record fits exactly in the DRAM-only manager; the pinned owner
    // holds that space, so the second equal record must wait.
    let manager = Manager::start_dram(&base, "16M");
    let mut a = gpu.open(base.join("manager.sock"), "");
    let mut b = gpu.open(base.join("manager.sock"), "");
    put(&mut a, &owner);

    let pause = a.pause_inbound_guard();
    let hit = hit(&mut a, &owner, Tier::Memory);
    let pages = hit.pages;
    a.fetch(
        hit,
        InboundTicket {
            key: owner.key,
            pages: (300..300 + pages as u32).map(PageId).collect(),
            slot: StateSlotId(7),
        },
    )
    .unwrap();
    until(|| {
        a.poll(&mut Vec::new()).unwrap();
        a.paused_transfers() >= 2
    });
    // The paused restore keeps the owner's record pinned, so the other
    // client's equal record cannot be admitted yet.
    let outcome = b.store(&waiting).unwrap();
    let refused = event(&mut b);
    pause.release();
    assert_eq!(outcome, StoreOutcome::Queued);
    assert!(
        matches!(refused, StoreEvent::Deferred { .. }),
        "{refused:?}"
    );

    // Releasing the window owner lets the other client progress.
    assert!(matches!(
        event(&mut a),
        StoreEvent::Fetched { result: Ok(_), .. }
    ));
    put_later(&mut b, &waiting);
    a.close().unwrap();
    b.close().unwrap();
    drop(a);
    drop(b);
    drop(manager);
}

#[test]
#[ignore = "requires GPU 0, FLARE_POOL_TEST_MANAGER and FLARE_POOL_TEST_DIR on local NVMe"]
fn real_manager_restores_multi_record_chain_with_non_aligned_boundaries() {
    let base = base("chain");
    let gpu = Gpu::new();
    // Every boundary is mid-page: floors are 2, 2, 3, 4. The second record is
    // row-only in the middle; the tails are 2, 0, 1, 1 pages.
    let first = checkpoint(130, None, 1);
    let middle = checkpoint(131, Some(130), 2);
    let third = checkpoint(195, Some(131), 3);
    let last = checkpoint(259, Some(195), 4);
    assert_eq!(middle.tail_pages, 0);
    gpu.fill(4, 1);
    gpu.fill(4, 2);
    gpu.fill(4, 3);
    gpu.fill(4, 4);
    let manager = Manager::start(&base, "128M");
    let mut store = gpu.open(base.join("manager.sock"), "");
    put(&mut store, &first);
    put(&mut store, &middle);
    put(&mut store, &third);
    put(&mut store, &last);

    // Full chain: pages 0..3 in order, state from the final record.
    fetch(&mut store, &last, Tier::Memory, true);
    gpu.check(4, 4);
    // Prefix ending at the third record: pages 0..2, state from that record.
    fetch(&mut store, &third, Tier::Memory, true);
    gpu.check(3, 3);
    // Prefix ending at the row-only middle record: pages 0..1, its own row.
    fetch(&mut store, &middle, Tier::Memory, true);
    gpu.check(2, 2);
    store.close().unwrap();
    drop(store);
    drop(manager);
}

#[test]
#[ignore = "requires GPU 0, FLARE_POOL_TEST_MANAGER and FLARE_POOL_TEST_DIR on local NVMe"]
fn real_manager_retries_a_budget_refusal_after_room() {
    let base = base("room");
    let gpu = Gpu::new();
    let first = checkpoint(64 * 64 + 3, None, 1);
    let second = checkpoint(16 * 64 + 5, None, 2);
    gpu.fill(64, 1);
    gpu.fill(16, 2);
    // One MiB/s admits the 64-page first record (going about 3.5 MiB into
    // debt), so the second Put is refused long after the first record's
    // charge and the Room that follows is bounded by the refill, not raced
    // by it.
    let manager = Manager::start_budget(&base, "16M", "1048576");
    let mut store = gpu.open(base.join("manager.sock"), "");
    put(&mut store, &first);
    ready(&mut store);
    let generation = store.shared.generation.load(Ordering::Acquire);
    let rooms = store.shared.room.load(Ordering::Acquire);
    assert_eq!(store.store(&second).unwrap(), StoreOutcome::Queued);
    let refused = event(&mut store);
    assert!(
        matches!(refused, StoreEvent::Deferred { .. }),
        "{refused:?}"
    );
    // The deferral above was the budget: while the connection is armed, a
    // retry reports BUDGET rather than a transient QUEUE or INFLIGHT.
    assert_eq!(
        store.store(&second).unwrap(),
        StoreOutcome::Retry(StoreReason::Budget)
    );
    // The same connection must accept the retry once Room arrives, and Room
    // is a live event on the unchanged subscription: no reconnect or epoch.
    put_later(&mut store, &second);
    assert_eq!(store.shared.generation.load(Ordering::Acquire), generation);
    assert!(store.shared.room.load(Ordering::Acquire) > rooms);
    store.close().unwrap();
    drop(store);
    drop(manager);
}

#[test]
#[ignore = "requires GPU 0, FLARE_POOL_TEST_MANAGER and FLARE_POOL_TEST_DIR on local NVMe"]
fn real_manager_cuda_round_trips_and_recovery() {
    let base = PathBuf::from(std::env::var_os("FLARE_POOL_TEST_DIR").unwrap()).join(format!(
        "l1-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&base).unwrap();
    let gpu = Gpu::new();
    let root = checkpoint(70 * 64 + 3, None, 1);
    let child = checkpoint(140 * 64 + 5, Some(70 * 64 + 3), 2);
    gpu.fill(140, 1);
    gpu.fill(140, 2);
    let manager = Manager::start(&base, "128M");
    let mut store = gpu.open(base.join("manager.sock"), "");
    put(&mut store, &root);
    fetch(&mut store, &root, Tier::Memory, true);
    gpu.check(70, 1);
    put(&mut store, &child);
    fetch(&mut store, &child, Tier::Memory, true);
    gpu.check(140, 2);
    eprintln!("L1 DRAM root/child, partial tails, padded planes/row: exact bytes pass");

    // Tenant isolation: another tenant shares the manager but not records.
    let mut other = gpu.open(base.join("manager.sock"), "other");
    assert!(
        other
            .lookup(&history(), 0, child.key.consumed_tokens + 1)
            .is_none()
    );
    other.close().unwrap();
    drop(other);

    store.close().unwrap();
    drop(store);
    drop(manager);
    let manager = Manager::start(&base, "4K");
    let mut store = gpu.open(base.join("manager.sock"), "");
    fetch(&mut store, &child, Tier::Disk, true);
    gpu.check(140, 2);
    eprintln!("L1 NVMe recovered chain longer than DRAM: exact bytes pass");
    store.close().unwrap();
    drop(store);
    drop(manager);

    let streaming = base.join("streaming");
    let manager = Manager::start(&streaming, "2M");
    let mut store = gpu.open(streaming.join("manager.sock"), "");
    put(&mut store, &root);
    put(&mut store, &child);
    fetch(&mut store, &child, Tier::Disk, true);
    gpu.check(140, 2);
    eprintln!("L1 streaming root/child individually exceed DRAM: exact bytes pass");
    store.close().unwrap();
    drop(store);
    drop(manager);
    eprintln!(
        "L1 gRPC manager round trips and recovery pass; raw files at {}",
        base.display()
    );
}
