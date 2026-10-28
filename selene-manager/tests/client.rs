//! The client against the real manager, in process over a Unix socket: two
//! instances sharing records through shared memory, a put into the manager's
//! store, characterization, and a manager that restarts.

use flare_selene::shared::{Scope, Tier};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use flare_selene::{
    History, PageId, PrefixKey, Result, StateSlotId, StoreEvent, StoreOutcome,
    geometry::fingerprint,
    plan::{self, Identity, Request},
    pool::{
        Checksums, FetchPart, HostChain, HostPin, HostRecord, HostWindow, IndexEntry, LookupState,
        PoolAddress, PoolClient, PoolConfig, PoolEvent, PoolFetch, PoolStatus, PooledStore,
        PutRequest, PutSource, STARTUP_WAIT,
    },
    record::checksum,
    testing::{GpuMemory, SimulatedGpu, TestHistory, geometry},
};

use flare_selene::{
    CacheGeometry, CheckpointId, HostStore, Layout, PublishedCheckpoint,
    composite::CompositeStore,
    disk::DiskThread,
    disk_store::{DiskStore, Staged},
    memory::MemoryStore,
    record,
    slab::SlabPool,
};

const PT: u64 = 16;
const PAGE: usize = 8192;
const ROW: usize = 4096;

use flare_selene::testing::IdentityRegistrar as OkRegistrar;
use flare_selene::testing::TrackingRegistrar as CountingRegistrar;
use selene_manager::testing::free_address;
use selene_manager::testing::{Manager, wait};

fn network_manager(tls: bool) -> Manager {
    network_manager_with(tls, false)
}

fn network_manager_with(tls: bool, rdma: bool) -> Manager {
    let mut manager = Manager::start();
    manager.stop();
    let _ = std::fs::remove_file(&manager.config.socket);
    let free = free_address;
    let net = &mut manager.config;
    net.listen = Some(free());
    net.listen_data = Some(free());
    net.metrics = Some(free());
    net.rdma = rdma;
    if tls {
        let generated = selene_manager::testing::certificate();
        let (cert, key) = (
            manager.base.join("server.pem"),
            manager.base.join("server-key.pem"),
        );
        std::fs::write(&cert, generated.cert.pem()).unwrap();
        std::fs::write(&key, generated.signing_key.serialize_pem()).unwrap();
        net.tls_cert = Some(cert.clone());
        net.tls_key = Some(key);
    }
    manager.run();
    manager
}

fn moved(manager: &Manager, carrier: &str) -> (u64, u64) {
    let text = manager.metrics();
    let value = |name: &str| {
        text.lines()
            .find_map(|line| {
                line.strip_prefix(&format!(
                    "selene_pool_channel_{name}_total{{carrier=\"{carrier}\"}} "
                ))
            })
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(0)
    };
    (value("windows"), value("bytes"))
}

fn remote(manager: &Manager) -> Option<(PoolAddress, Option<PathBuf>)> {
    let port = manager.config.listen?.port();
    let ca = manager.config.tls_cert.clone();
    let host = if ca.is_some() {
        "localhost"
    } else {
        "127.0.0.1"
    };
    let address = format!("{host}:{port}");
    Some((
        if ca.is_some() {
            PoolAddress::Tcp(address)
        } else {
            PoolAddress::TcpPlaintext(address)
        },
        ca,
    ))
}

fn geo() -> Vec<flare_selene::CacheGeometry> {
    vec![geometry(PT, &[("k", 4096), ("v", 4096)], 4096)]
}

fn config(manager: &Manager, gpu: u8) -> PoolConfig {
    config_on(manager, gpu, false)
}

/// As [`config`]; `local` reaches a network manager over its socket, as an instance on the
/// manager's own host does.
fn config_on(manager: &Manager, gpu: u8, local: bool) -> PoolConfig {
    let geometry = geo();
    let (address, ca) = remote(manager)
        .filter(|_| !local)
        .unwrap_or_else(|| (PoolAddress::Unix(manager.config.socket.clone()), None));
    PoolConfig {
        address,
        ca,
        scope: Scope::new([7; 32], fingerprint(&geometry), &geometry).unwrap(),
        gpu_uuids: vec![[gpu; 16]],
        export_slots: 2,
        claim_slots: 2,

        registration_context: 0,
        approve: None,
    }
}

struct Instance {
    client: PoolClient,
    gpu: GpuMemory,
    events: Vec<PoolEvent>,
}

fn instance(manager: &Manager, gpu: u8) -> Instance {
    instance_on(manager, gpu, false)
}

fn instance_on(manager: &Manager, gpu: u8, local: bool) -> Instance {
    let (copier, memory) = SimulatedGpu::shared(&geo()).unwrap();
    let client = PoolClient::start(
        config_on(manager, gpu, local),
        None,
        copier,
        Arc::new(OkRegistrar),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    Instance {
        client,
        gpu: memory,
        events: Vec::new(),
    }
}

impl Instance {
    fn ready(&self) {
        wait("Ready", || self.client.status() == PoolStatus::Ready);
    }

    fn poll(&mut self) {
        let mut events = Vec::new();
        self.client.poll(&mut events).unwrap();
        self.events.extend(events);
    }

    /// Poll until `take` finds an event, removing it.
    fn event<T>(&mut self, what: &str, mut take: impl FnMut(&PoolEvent) -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            self.poll();
            if let Some(position) = self.events.iter().position(|e| take(e).is_some()) {
                let event = self.events.remove(position);
                return take(&event).unwrap();
            }
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(2));
        }
    }
}

fn history() -> TestHistory {
    history_of(4000)
}

fn history_of(tokens: u32) -> TestHistory {
    let mut history = TestHistory::new([7; 32]);
    history
        .append(&(0..tokens).collect::<Vec<u32>>(), &[])
        .unwrap();
    history
}

fn identity() -> Identity {
    Identity {
        model: [7; 32],
        layout: fingerprint(&geo()),
    }
}

use flare_selene::testing::{BytesPin as TestPin, RecordBytes as Bytes};

fn pin_for(key: PrefixKey, pages: usize) -> Box<dyn HostPin> {
    let data: Vec<u8> = (0..pages * PAGE + ROW)
        .map(|i| (i * 13 + 5) as u8)
        .collect();
    let sums = (0..pages)
        .map(|p| checksum(&data[p * PAGE..(p + 1) * PAGE]))
        .collect();
    let row = checksum(&data[pages * PAGE..]);
    Box::new(TestPin {
        chain: HostChain {
            records: vec![HostRecord {
                key,
                first_page: 0,
                pages: pages as u32,
                row: true,
            }],
            total_pages: pages as u64,
            row: true,
        },
        bytes: Bytes { data, sums, row, page_bytes: PAGE, row_bytes: ROW },
    })
}

fn expected_page(page: usize) -> Vec<u8> {
    (page * PAGE..(page + 1) * PAGE)
        .map(|i| (i * 13 + 5) as u8)
        .collect()
}

/// Plan a restore of `history` from what the lookup answered, as the store does.
fn fetch_for(
    instance: &mut Instance,
    history: &TestHistory,
    pages: usize,
    scratch: &[u32],
) -> PoolFetch {
    let ticket = instance.client.begin_lookup(history, 0);
    wait("the lookup", || {
        !matches!(ticket.state(), LookupState::Pending)
    });
    let LookupState::Ready(answer) = ticket.state() else {
        panic!("the lookup has no answer");
    };
    assert!(
        !answer.candidates.is_empty(),
        "the manager offered a candidate"
    );
    let request = Request {
        history,
        identity: &identity(),
        page_tokens: PT,
        resident: 0,
        before: u64::MAX,
    };
    let plan = plan::plan(&request, &answer.plan_candidates()).expect("a plan");
    assert_eq!(plan.pages(), pages);
    let part = plan.part.clone();
    let candidate = answer
        .candidates
        .iter()
        .find(|c| c.plan.source == part.source)
        .unwrap()
        .clone();
    PoolFetch {
        id: 0,
        page_tokens: PT,
        part: FetchPart {
            candidate,
            part,
            pages: scratch.iter().map(|p| PageId(*p)).collect(),
            slot: Some(StateSlotId(9)),
        },
    }
}

#[test]
fn a_client_lookup_includes_an_endpoint_exactly_at_the_prompt_length() {
    let manager = Manager::start();
    let source = instance(&manager, 1);
    let requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history_of(65);
    let key = history.key_at(history.token_count()).unwrap();
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: 4,
        row: true,
        tier: Tier::Dram,
        bytes: (4 * PAGE + ROW) as u64,
    });
    wait("the exact prompt endpoint", || {
        let ticket = requester.client.begin_lookup(&history, 0);
        wait("the lookup reply", || {
            !matches!(ticket.state(), LookupState::Pending)
        });
        matches!(ticket.state(), LookupState::Ready(answer) if answer.candidates.iter().any(|c| c.plan.key == key))
    });
}

#[test]
fn an_instance_restores_a_neighbours_host_record_through_the_real_manager() {
    let manager = Manager::start();
    let mut source = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history();
    let pages = 4usize;
    let key = history.key_at(pages as u64 * PT).unwrap();
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Dram,
        bytes: (pages * PAGE + ROW) as u64,
    });
    // The manager learns of it; the requester's lookup then finds it.
    wait("the manager to offer the record", || {
        let ticket = requester.client.begin_lookup(&history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    });
    let scratch = [100, 101, 102, 103];
    // Built before the claim starts: the manager's pin deadline runs from the request, and
    // making the data in an unoptimised build on a loaded machine is not what is under test.
    let pin = pin_for(key, pages);
    let fetch = fetch_for(&mut requester, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    // The source is asked to export; its store (here, the test) pins the chain.
    let export = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => Some((request.export, request.tier)),
        _ => None,
    });
    assert_eq!(export.1, Tier::Dram);
    source.client.host_pinned(export.0, pin);
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * PAGE + ROW) as u64));
    let gpu = requester.gpu.pages.lock().unwrap();
    for page in 0..pages {
        assert_eq!(
            gpu[&(100 + page as u32)],
            expected_page(page),
            "page {page}"
        );
    }
    assert_eq!(
        requester.gpu.rows.lock().unwrap()[&9],
        (pages * PAGE..pages * PAGE + ROW)
            .map(|i| (i * 13 + 5) as u8)
            .collect::<Vec<u8>>()
    );
    drop(gpu);
}

fn put_request(history: &TestHistory, pages: usize) -> (PutRequest, PrefixKey) {
    let key = history.key_at(pages as u64 * PT).unwrap();
    let data: Vec<u8> = (0..pages * PAGE + ROW)
        .map(|i| (i * 13 + 5) as u8)
        .collect();
    let sums = (0..pages)
        .map(|p| checksum(&data[p * PAGE..(p + 1) * PAGE]))
        .collect();
    let row = checksum(&data[pages * PAGE..]);
    (
        PutRequest {
            key,
            parent: None,
            bucket: history.bucket_at(key.consumed_tokens).unwrap(),
            chain_pages: pages as u64,
            tail_pages: pages as u64,
            row: true,
            source: PutSource::Host(Box::new(Bytes { data, sums, row, page_bytes: PAGE, row_bytes: ROW })),
        },
        key,
    )
}

fn wait_store_index(requester: &Instance, history: &TestHistory) {
    // Put completion can precede the asynchronous catalog update of the pool index.
    wait("the manager's store record to be indexed", || {
        offered(requester, history)
    });
}

fn restore(requester: &mut Instance, history: &TestHistory, pages: usize, scratch: &[u32]) {
    let fetch = fetch_for(requester, history, pages, scratch);
    requester.client.fetch(fetch).unwrap();
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * PAGE + ROW) as u64));
    let gpu = requester.gpu.pages.lock().unwrap();
    for page in 0..pages {
        assert_eq!(gpu[&scratch[page]], expected_page(page), "page {page}");
    }
}

fn busy_source_restores(manager: &Manager) {
    let mut source = instance(manager, 1);
    let mut requesters: Vec<_> = (2..=5).map(|id| instance(manager, id)).collect();
    source.ready();
    let histories: Vec<_> = (1..=4).map(salted).collect();
    let pages = 4;
    let scratch = [100, 101, 102, 103];
    let keys: Vec<_> = requesters
        .iter()
        .zip(&histories)
        .map(|(requester, history)| {
            requester.ready();
            offer_host_record(&source, requester, history, pages)
        })
        .collect();
    let started = Instant::now();
    let deadline = started
        + flare_selene::shared::transfer_timeout(
            flare_selene::testing::RESTORE_DEADLINE,
            (pages * PAGE + ROW) as u64,
        );
    // Occupy both export slots before the other lookups and claims arrive.
    let mut held = Vec::new();
    for i in 0..2 {
        let fetch = fetch_for(&mut requesters[i], &histories[i], pages, &scratch);
        requesters[i].client.fetch(fetch).unwrap();
        held.push(
            source.event("an occupied export slot", |event| match event {
                PoolEvent::Export(request) => Some(request.export),
                _ => None,
            }),
        );
    }
    for i in 2..4 {
        let fetch = fetch_for(&mut requesters[i], &histories[i], pages, &scratch);
        requesters[i].client.fetch(fetch).unwrap();
    }
    wait("both excess claims to encounter the busy source", || {
        manager.metrics().lines().any(|line| {
            line.starts_with("selene_pool_claims_refused_total{reason=\"SourceLimit\"} ")
                && line
                    .split_whitespace()
                    .last()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
                    >= 2
        })
    });
    for (export, key) in held.into_iter().zip(&keys) {
        source.client.host_pinned(export, pin_for(*key, pages));
    }
    let mut restored = 0;
    while restored < requesters.len() {
        assert!(
            Instant::now() < deadline,
            "restores exceeded their deadline"
        );
        source.poll();
        for event in std::mem::take(&mut source.events) {
            if let PoolEvent::Export(request) = event {
                assert!(keys.contains(&request.key));
                source
                    .client
                    .host_pinned(request.export, pin_for(request.key, pages));
            }
        }
        for requester in &mut requesters {
            requester.poll();
            for event in std::mem::take(&mut requester.events) {
                if let PoolEvent::Fetched { result, .. } = event {
                    assert_eq!(result.unwrap(), (pages * PAGE + ROW) as u64);
                    check_restored(requester, &scratch, pages);
                    restored += 1;
                }
            }
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(restored, 4, "every request restored without recomputing");
    for requester in &mut requesters {
        requester.client.close().unwrap();
    }
    source.client.close().unwrap();
}

#[test]
fn a_busy_source_restores_every_request_through_shared_memory() {
    let mut manager = Manager::start();
    manager.stop();
    manager.config.limits.pin_deadline_ms = 5_000;
    manager.config.metrics = Some(free_address());
    manager.run();
    busy_source_restores(&manager);
}

#[test]
fn a_busy_source_restores_every_request_over_tcp() {
    let mut manager = network_manager(false);
    manager.stop();
    manager.config.limits.pin_deadline_ms = 5_000;
    manager.run();
    busy_source_restores(&manager);
}

#[test]
fn a_record_put_into_the_manager_store_is_restored_by_another_instance() {
    let manager = Manager::start();
    let mut writer = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    writer.ready();
    requester.ready();
    let history = history();
    let (request, _) = put_request(&history, 5);
    writer.client.put(request).unwrap();
    let outcome = writer.event("the put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert!(outcome.starts_with("Stored"), "{outcome}");
    wait_store_index(&requester, &history);
    restore(&mut requester, &history, 5, &[200, 201, 202, 203, 204]);
    // The same record put again is a successful deduplication.
    let (request, _) = put_request(&history, 5);
    writer.client.put(request).unwrap();
    let outcome = writer.event("the second put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert_eq!(outcome, "Present");
}

#[test]
fn a_gpu_checkpoint_is_exported_by_device_copies_and_its_pin_released() {
    let manager = Manager::start();
    let mut source = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history();
    let pages = 3usize;
    let key = history.key_at(pages as u64 * PT).unwrap();
    // The checkpoint's pages on the source's device.
    for page in 0..pages {
        source
            .gpu
            .pages
            .lock()
            .unwrap()
            .insert(20 + page as u32, expected_page(page));
    }
    let row: Vec<u8> = (pages * PAGE..pages * PAGE + ROW)
        .map(|i| (i * 13 + 5) as u8)
        .collect();
    source.gpu.rows.lock().unwrap().insert(5, row.clone());
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Gpu,
        bytes: (pages * PAGE + ROW) as u64,
    });
    wait("the manager to offer it", || {
        let ticket = requester.client.begin_lookup(&history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    });
    let fetch = fetch_for(&mut requester, &history, pages, &[300, 301, 302]);
    assert_eq!(
        fetch.part.candidate.plan.level,
        flare_selene::levels::Level::NeighbourGpu
    );
    requester.client.fetch(fetch).unwrap();
    // The worker's side: pin the chain and answer, then release at the end.
    let (export, reserved) = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => {
            assert_eq!((request.tier, request.key), (Tier::Gpu, key));
            Some((request.export, request.reserved_bytes))
        }
        _ => None,
    });
    assert!(reserved >= (pages * PAGE + ROW) as u64);
    source.client.export_pinned(
        export,
        flare_selene::ExportChain {
            nodes: vec![flare_selene::ExportNode {
                key,
                pages: (20..20 + pages as u32).map(PageId).collect(),
            }],
            skip_pages: 0,
            row: Some(StateSlotId(5)),
        },
    );
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * PAGE + ROW) as u64));
    for page in 0..pages {
        assert_eq!(
            requester.gpu.pages.lock().unwrap()[&(300 + page as u32)],
            expected_page(page)
        );
    }
    assert_eq!(requester.gpu.rows.lock().unwrap()[&9], row);
    source.event("the pin to be released", |e| match e {
        PoolEvent::ExportEnded { export: ended } if *ended == export => Some(()),
        _ => None,
    });
}

#[test]
fn announcements_survive_a_manager_restart_and_the_instances_reconnect() {
    let mut manager = Manager::start();
    let mut source = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history();
    let key = history.key_at(4 * PT).unwrap();
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: 4,
        row: true,
        tier: Tier::Dram,
        bytes: (4 * PAGE + ROW) as u64,
    });
    let offered = |requester: &Instance| {
        let ticket = requester.client.begin_lookup(&history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    };
    wait("the record to be offered", || offered(&requester));
    manager.restart();
    // Both instances fall back to their local tiers and come back.
    wait("the instances to reconnect", || {
        source.client.status() == PoolStatus::Ready
            && requester.client.status() == PoolStatus::Ready
    });
    // The source announced from its own index again under the same id.
    wait("the record to be offered again", || offered(&requester));
    source.client.close().unwrap();
    requester.client.close().unwrap();
}

#[test]
fn a_restore_through_the_real_manager_still_works_after_the_manager_restarts() {
    let mut manager = Manager::start();
    let mut source = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history();
    let pages = 4usize;
    let key = history.key_at(pages as u64 * PT).unwrap();
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Dram,
        bytes: (pages * PAGE + ROW) as u64,
    });
    let offered = |requester: &Instance| {
        let ticket = requester.client.begin_lookup(&history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    };
    wait("the record to be offered", || offered(&requester));
    manager.restart();
    let restarted = Instant::now();
    wait("the instances to reconnect", || {
        source.client.status() == PoolStatus::Ready
            && requester.client.status() == PoolStatus::Ready
    });
    assert!(
        restarted.elapsed() < Duration::from_secs(4),
        "reconnected {:?} after the restart",
        restarted.elapsed()
    );
    wait("the record to be offered again", || offered(&requester));
    // Round after round, the restore copies through the regions the restart reused.
    for round in 0..3u32 {
        let scratch = [100 + round * 10, 101 + round * 10, 102 + round * 10, 103 + round * 10];
        let pin = pin_for(key, pages);
        let fetch = fetch_for(&mut requester, &history, pages, &scratch);
        requester.client.fetch(fetch).unwrap();
        let export = source.event("an export request", |e| match e {
            PoolEvent::Export(request) => Some(request.export),
            _ => None,
        });
        source.client.host_pinned(export, pin);
        let result = requester.event("the restore", |e| match e {
            PoolEvent::Fetched { result, .. } => {
                Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
            }
            _ => None,
        });
        assert_eq!(result, Ok((pages * PAGE + ROW) as u64), "round {round}");
    }
}

fn counted(manager: &Manager, registrar: &Arc<CountingRegistrar>, wait: Duration) -> PoolClient {
    counted_approving(manager, registrar, wait, None)
}

fn counted_approving(
    manager: &Manager,
    registrar: &Arc<CountingRegistrar>,
    wait: Duration,
    approve: Option<flare_selene::pool::Approve>,
) -> PoolClient {
    let (copier, _gpu) = SimulatedGpu::shared(&geo()).unwrap();
    let mut config = config(manager, 1);
    config.approve = approve;
    PoolClient::start(
        config,
        None,
        copier,
        registrar.clone(),
        flare_selene::testing::Roots::System,
        wait,
    )
    .unwrap()
}

fn inode(manager: &Manager, file: &str) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(manager.config.shm_directory().unwrap().join(file))
        .unwrap()
        .ino()
}

#[test]
fn a_manager_restart_reuses_the_regions_and_the_client_registers_nothing_again() {
    let mut manager = Manager::start();
    let registrar = Arc::new(CountingRegistrar::default());
    let client = counted(&manager, &registrar, STARTUP_WAIT);
    let regions = client.registered_regions().to_vec();
    assert_eq!(
        regions.iter().map(|r| r.name).collect::<Vec<_>>(),
        ["arena", "staging"],
        "the startup registered the manager's two regions"
    );
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    wait("Ready", || client.status() == PoolStatus::Ready);
    let files = (inode(&manager, "arena"), inode(&manager, "staging"));

    manager.stop();
    wait("the old session to disconnect", || {
        client.status() != PoolStatus::Ready
    });
    manager.run();
    wait("a session with the restarted manager", || {
        client.status() == PoolStatus::Ready
    });
    assert_eq!(
        (inode(&manager, "arena"), inode(&manager, "staging")),
        files,
        "the manager reopened the same files"
    );
    assert_eq!(
        registrar.calls.lock().unwrap().len(),
        2,
        "the reconnect registered nothing"
    );
    assert_eq!(client.registered_regions(), regions);
}

/// An approval that records each plan with the registrations made before it, and says `verdict`.
fn recording(
    registrar: &Arc<CountingRegistrar>,
    verdict: std::result::Result<(), String>,
) -> (
    flare_selene::pool::Approve,
    Arc<std::sync::Mutex<Vec<(Vec<flare_selene::pool::PlannedRegion>, usize)>>>,
) {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (log, registrar) = (seen.clone(), registrar.clone());
    let approve: flare_selene::pool::Approve = Arc::new(move |plan| {
        let before = registrar.calls.lock().unwrap().len();
        log.lock().unwrap().push((plan.to_vec(), before));
        verdict.clone()
    });
    (approve, seen)
}

#[test]
fn the_plan_arrives_whole_before_any_registration_and_a_reconnect_asks_nothing() {
    let mut manager = Manager::start();
    let registrar = Arc::new(CountingRegistrar::default());
    let (approve, seen) = recording(&registrar, Ok(()));
    let client = counted_approving(&manager, &registrar, STARTUP_WAIT, Some(approve));
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let (plan, registered_before) = &seen[0];
        assert_eq!(*registered_before, 0, "approved before the first register");
        assert_eq!(
            plan.iter().map(|r| r.name).collect::<Vec<_>>(),
            ["arena", "staging"]
        );
        let sizes: Vec<_> = client
            .registered_regions()
            .iter()
            .map(|r| r.bytes as u64)
            .collect();
        assert_eq!(plan.iter().map(|r| r.bytes).collect::<Vec<_>>(), sizes);
    }
    wait("Ready", || client.status() == PoolStatus::Ready);
    manager.stop();
    wait("the old session to disconnect", || {
        client.status() != PoolStatus::Ready
    });
    manager.run();
    wait("a session with the restarted manager", || {
        client.status() == PoolStatus::Ready
    });
    assert_eq!(seen.lock().unwrap().len(), 1, "the reconnect asked nothing");
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
}

#[test]
fn a_rejected_plan_registers_nothing_and_turns_the_pool_off() {
    let manager = Manager::start();
    let registrar = Arc::new(CountingRegistrar::default());
    let (approve, seen) = recording(&registrar, Err("over the host budget".into()));
    let mut client = counted_approving(&manager, &registrar, STARTUP_WAIT, Some(approve));
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(registrar.calls.lock().unwrap().is_empty());
    assert!(client.registered_regions().is_empty());
    // The thread answers startup first and sets the status just after.
    wait("the pool to be disabled", || {
        matches!(client.status(), PoolStatus::Disabled(_))
    });
    let PoolStatus::Disabled(reason) = client.status() else {
        panic!("{:?}", client.status());
    };
    assert!(
        reason.contains("over the host budget")
            && reason.contains("arena")
            && reason.contains("staging"),
        "{reason}"
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(registrar.calls.lock().unwrap().is_empty());
    client.close().unwrap();
}

#[test]
fn a_changed_staging_size_turns_the_pool_off_for_a_running_instance() {
    let mut manager = Manager::start();
    let registrar = Arc::new(CountingRegistrar::default());
    let mut client = counted(&manager, &registrar, STARTUP_WAIT);
    wait("Ready", || client.status() == PoolStatus::Ready);

    manager.stop();
    manager.config.staging *= 2;
    let _ = std::fs::remove_file(&manager.config.socket);
    manager.run();
    wait("the instance to notice", || {
        matches!(client.status(), PoolStatus::Disabled(_))
    });
    let PoolStatus::Disabled(reason) = client.status() else {
        unreachable!()
    };
    assert!(
        reason.contains("recreated its shared memory") && reason.contains("restarted"),
        "{reason}"
    );
    assert!(matches!(client.status(), PoolStatus::Disabled(_)));
    assert_eq!(
        registrar.calls.lock().unwrap().len(),
        2,
        "nothing was registered late"
    );
    client.close().unwrap();
}

#[test]
fn a_manager_that_is_not_there_at_startup_is_never_attached_to_later() {
    const CHILD: &str = "FLARE_TEST_POOL_STARTUP_WARNING";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "a_manager_that_is_not_there_at_startup_is_never_attached_to_later",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{stderr}");
        assert_eq!(stderr.matches("warning:").count(), 1, "{stderr}");
        assert!(
            stderr.contains("did not answer within 400ms of startup"),
            "{stderr}"
        );
        assert!(stderr.contains("local tiers only"), "{stderr}");
        assert!(stderr.contains("will not attach to the pool later"), "{stderr}");
        return;
    }
    let mut manager = Manager::start();
    manager.stop();
    let _ = std::fs::remove_file(&manager.config.socket);
    let registrar = Arc::new(CountingRegistrar::default());
    let started = Instant::now();
    let mut client = counted(&manager, &registrar, Duration::from_millis(400));
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(client.registered_regions().is_empty());
    let PoolStatus::Disabled(reason) = client.status() else {
        panic!("{:?}", client.status());
    };
    assert!(reason.contains("local tiers only"), "{reason}");

    // The manager comes up after the instance gave up on it.
    manager.run();
    assert!(matches!(client.status(), PoolStatus::Disabled(_)));
    client.close().unwrap();
    assert!(registrar.calls.lock().unwrap().is_empty());
}

// ---- the eviction cascade against the real manager ----

/// A history of its own: every record published from one is a root.
fn salted(salt: u32) -> TestHistory {
    let mut history = TestHistory::new([7; 32]);
    let tokens: Vec<u32> = (0..4000u32)
        .map(|t| t.wrapping_mul(2654435761).wrapping_add(salt))
        .collect();
    history.append(&tokens, &[]).unwrap();
    history
}

fn salted_page(salt: u32, page: usize) -> Vec<u8> {
    (0..PAGE)
        .map(|i| ((i * 13 + page * 7 + salt as usize * 31 + 5) % 251) as u8)
        .collect()
}

fn salted_row(salt: u32) -> Vec<u8> {
    (0..ROW)
        .map(|i| ((i * 11 + salt as usize * 17 + 3) % 251) as u8)
        .collect()
}

/// The checkpoint of `pages` pages that record `salt` publishes, with its
/// bytes placed in the simulated device first.
fn publication(gpu: &GpuMemory, salt: u32, pages: usize) -> (PublishedCheckpoint, TestHistory) {
    let history = salted(salt);
    let base = salt * 100;
    for page in 0..pages {
        gpu.pages
            .lock()
            .unwrap()
            .insert(base + page as u32, salted_page(salt, page));
    }
    gpu.rows.lock().unwrap().insert(salt, salted_row(salt));
    let key = history.key_at(pages as u64 * PT).unwrap();
    let checkpoint = PublishedCheckpoint {
        id: CheckpointId::new(u64::from(salt)),
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        chain_pages: (base..base + pages as u32).map(PageId).collect(),
        tail_pages: pages,
        slot: StateSlotId(salt),
    };
    (checkpoint, history)
}

/// Publish as the model worker does: start the copy and poll until it is
/// reported, and try again after a refusal or a deferral. The device cache
/// then evicts the checkpoint, as it does under pressure.
fn publish<S: HostStore>(store: &mut S, checkpoint: &PublishedCheckpoint) {
    store_and_wait(store, checkpoint);
    store.evicted(checkpoint.key, None);
}

fn store_and_wait<S: HostStore>(store: &mut S, checkpoint: &PublishedCheckpoint) {
    let deadline = Instant::now() + Duration::from_secs(20);
    'attempt: loop {
        let outcome = store.store(checkpoint).unwrap();
        if outcome == StoreOutcome::Present {
            return;
        }
        assert_ne!(outcome, StoreOutcome::Oversize);
        let mut events = Vec::new();
        while Instant::now() < deadline {
            store.poll(&mut events).unwrap();
            let mut retry = outcome != StoreOutcome::Queued;
            for event in events.drain(..) {
                match event {
                    StoreEvent::Stored { key, result, .. } if key == checkpoint.key => {
                        if let Err(error) = &result {
                            assert_eq!(error.to_string(), "no room for the record on disk");
                            retry = true;
                        } else if outcome == StoreOutcome::Queued {
                            return;
                        }
                    }
                    StoreEvent::Deferred { checkpoint: id, .. } if id == checkpoint.id => {
                        retry = true
                    }
                    _ => {}
                }
            }
            thread::sleep(Duration::from_millis(2));
            if retry {
                continue 'attempt;
            }
        }
        panic!("timed out publishing {:?}", checkpoint.key);
    }
}

/// Poll `store` until `done`.
fn drive<S: HostStore>(store: &mut S, what: &str, mut done: impl FnMut(&mut S) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut events = Vec::new();
    while !done(store) {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        store.poll(&mut events).unwrap();
        events.clear();
        thread::sleep(Duration::from_millis(2));
    }
}

fn offered(requester: &Instance, history: &TestHistory) -> bool {
    let ticket = requester.client.begin_lookup(history, 0);
    let deadline = Instant::now() + Duration::from_secs(2);
    while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(2));
    }
    matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
}

/// Restore the record `salt` of `pages` pages from the manager's store into
/// the requester and check every byte.
fn restore_salted(
    requester: &mut Instance,
    history: &TestHistory,
    salt: u32,
    pages: usize,
    evictor: &mut impl HostStore,
) {
    // The evictor's own notices must reach the manager first; it is polled
    // as the worker would.
    wait("the manager's store to be the only source", || {
        evictor.poll(&mut Vec::new()).unwrap();
        let ticket = requester.client.begin_lookup(history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        let key = history.key_at(pages as u64 * PT).unwrap();
        matches!(ticket.state(), LookupState::Ready(a)
            if a.candidates.iter().any(|c| c.plan.key == key && c.source.instance.is_empty())
                && a.candidates.iter().all(|c| c.plan.key != key || c.source.instance.is_empty()))
    });
    let scratch: Vec<u32> = (0..pages as u32).map(|p| 700 + salt * 10 + p).collect();
    let fetch = fetch_for(requester, history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * PAGE + ROW) as u64));
    let gpu = requester.gpu.pages.lock().unwrap();
    for (page, id) in scratch.iter().enumerate() {
        assert_eq!(
            gpu[id],
            salted_page(salt, page),
            "record {salt} page {page}"
        );
    }
    drop(gpu);
    assert_eq!(requester.gpu.rows.lock().unwrap()[&9], salted_row(salt));
}

/// A pooled store over a standalone memory store of `pages` pages, with its
/// own simulated device.
fn memory_instance(manager: &Manager, pages: usize) -> (PooledStore<MemoryStore>, GpuMemory) {
    let (transfers, gpu) = SimulatedGpu::shared(&geo()).unwrap();
    let pool = SlabPool::new(PAGE as u64, pages, ROW as u64, pages, true).unwrap();
    let memory = MemoryStore::new([7; 32], pool, transfers).unwrap();
    let (copier, _unused) = SimulatedGpu::shared(&geo()).unwrap();
    let store =
        PooledStore::connect(memory, config(manager, 1), copier, Arc::new(OkRegistrar)).unwrap();
    wait("Ready", || store.pool().status() == PoolStatus::Ready);
    (store, gpu)
}

#[test]
fn an_evicted_dram_record_reaches_the_manager_store_and_is_restored_by_another_instance() {
    let manager = Manager::start();
    let mut requester = instance(&manager, 2);
    requester.ready();
    let (mut store, gpu) = memory_instance(&manager, 8);
    let (first, first_history) = publication(&gpu, 1, 4);
    let (second, second_history) = publication(&gpu, 2, 4);
    let (third, third_history) = publication(&gpu, 3, 4);
    publish(&mut store, &first);
    publish(&mut store, &second);
    // The manager hears of both once their checksums are done.
    drive(&mut store, "the records to be announced", |_| {
        offered(&requester, &first_history) && offered(&requester, &second_history)
    });

    // The third needs the room: the coldest record moves to the manager first.
    publish(&mut store, &third);
    drive(&mut store, "the first record to leave", |store| {
        store.lookup(&first_history, 0, u64::MAX).is_none()
    });
    assert!(store.lookup(&second_history, 0, u64::MAX).is_some());
    assert!(store.lookup(&third_history, 0, u64::MAX).is_some());
    // Another instance restores it from the manager's store.
    restore_salted(&mut requester, &first_history, 1, 4, &mut store);
    flare_selene::HostStore::close(&mut store).unwrap();
}

/// A directory for the disk tier that takes direct I/O, or `None`.
fn offload_directory(test: &str) -> Option<PathBuf> {
    let base = std::env::var_os("FLARE_OFFLOAD_TEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let directory = base.join(format!(
        "flare-offload-cascade-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let probe = directory.join(".probe");
    match flare_eos::direct_file(&probe, true) {
        Ok(_) => {
            std::fs::remove_file(&probe).unwrap();
            Some(directory)
        }
        Err(error) if cfg!(unix) && error.raw_os_error() == Some(libc::EINVAL) => {
            eprintln!(
                "skipping {test}: {} refuses direct I/O",
                directory.display()
            );
            std::fs::remove_dir_all(&directory).unwrap();
            None
        }
        Err(error) => panic!("probe {}: {error}", probe.display()),
    }
}

#[test]
fn a_record_cascades_from_memory_to_the_disk_tier_to_the_manager_store() {
    let Some(directory) = offload_directory("composite") else {
        return;
    };
    let manager = Manager::start();
    let mut requester = instance(&manager, 2);
    requester.ready();
    let geometry: Vec<CacheGeometry> = geo();
    let layout = Layout::new(&geometry).unwrap();
    let fingerprint = fingerprint(&geometry);
    // Memory holds two records of two pages; the disk tier, two files.
    let (memory_transfers, gpu) = SimulatedGpu::shared(&geometry).unwrap();
    let pool = SlabPool::new(PAGE as u64, 4, ROW as u64, 4, true).unwrap();
    let memory = MemoryStore::new_flat([7; 32], pool, memory_transfers).unwrap();
    let lock = record::DirectoryLock::acquire(&directory).unwrap();
    let scanned = record::scan(&lock, &[7; 32], &fingerprint, geometry[0].page_tokens, &record::rank_sizes(&geometry)).unwrap();
    let file_bytes = record::Header::new(
        &geometry,
        [7; 32],
        fingerprint,
        salted(1).key_at(2 * PT).unwrap(),
        None,
        [0; 16],
        2,
        std::time::SystemTime::now(),
    )
    .unwrap()
    .length_with_row();
    let (disk_transfers, _) = SimulatedGpu::shared(&geometry).unwrap();
    let disk = DiskStore::new(
        lock,
        [7; 32],
        fingerprint,
        2 * file_bytes + file_bytes / 2,
        geometry,
        scanned,
        Box::new(Staged::new(
            Some(SlabPool::staging(&layout).unwrap()),
            disk_transfers,
            DiskThread::start("cascade-test-disk").unwrap(),
        )),
    )
    .unwrap();
    let composite = CompositeStore::new(memory, disk).unwrap();
    let (copier, _unused) = SimulatedGpu::shared(&geo()).unwrap();
    let mut store = PooledStore::connect(
        composite,
        config(&manager, 1),
        copier,
        Arc::new(OkRegistrar),
    )
    .unwrap();
    wait("Ready", || store.pool().status() == PoolStatus::Ready);

    let (first, first_history) = publication(&gpu, 1, 2);
    let (second, _) = publication(&gpu, 2, 2);
    let (third, _) = publication(&gpu, 3, 2);
    let (fourth, _) = publication(&gpu, 4, 2);
    // Memory and the disk tier fill: nothing is written until memory needs room.
    publish(&mut store, &first);
    publish(&mut store, &second);
    assert!(!directory.join(record::file_name(&first.key)).exists());
    // The third pushes the first to disk; the fourth finds the disk tier full
    // and pushes the coldest file on to the manager.
    publish(&mut store, &third);
    publish(&mut store, &fourth);
    drive(&mut store, "the first record's file to go", |_| {
        !directory.join(record::file_name(&first.key)).exists()
    });
    restore_salted(&mut requester, &first_history, 1, 2, &mut store);
    flare_selene::HostStore::close(&mut store).unwrap();
    drop(store);
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn drained_records_are_announced_and_claimable_after_restart() {
    let Some(directory) = offload_directory("drained-restart") else {
        return;
    };
    let manager = Manager::start();
    let mut requester = instance(&manager, 2);
    requester.ready();
    let open = || {
        let geometry = geo();
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let lock = record::DirectoryLock::acquire(&directory).unwrap();
        let scanned = record::scan(&lock, &[7; 32], &fingerprint, geometry[0].page_tokens, &record::rank_sizes(&geometry)).unwrap();
        let (memory_copier, gpu) = SimulatedGpu::shared(&geometry).unwrap();
        let memory = MemoryStore::new_flat(
            [7; 32],
            SlabPool::new(PAGE as u64, 16, ROW as u64, 4, true).unwrap(),
            memory_copier,
        )
        .unwrap();
        let (disk_copier, _) = SimulatedGpu::shared(&geometry).unwrap();
        let disk = DiskStore::new(
            lock,
            [7; 32],
            fingerprint,
            1 << 20,
            geometry,
            scanned,
            Box::new(Staged::new(
                Some(SlabPool::staging(&layout).unwrap()),
                disk_copier,
                DiskThread::start("drained-restart").unwrap(),
            )),
        )
        .unwrap();
        let mut composite = CompositeStore::new(memory, disk).unwrap();
        composite.drain_on_shutdown = Duration::MAX;
        let (copier, _) = SimulatedGpu::shared(&geo()).unwrap();
        let store = PooledStore::connect(
            composite,
            config(&manager, 1),
            copier,
            Arc::new(OkRegistrar),
        )
        .unwrap();
        (store, gpu)
    };
    let (mut source, gpu) = open();
    let records: Vec<_> = (1..=3).map(|salt| publication(&gpu, salt, 8)).collect();
    for (checkpoint, _) in &records {
        publish(&mut source, checkpoint);
    }
    drive(&mut source, "records offered before shutdown", |_| {
        records
            .iter()
            .all(|(_, history)| offered(&requester, history))
    });
    assert!(
        directory
            .join(record::file_name(&records[0].0.key))
            .exists()
    );
    let last_file = directory.join(record::file_name(&records[2].0.key));
    assert!(!last_file.exists(), "the last record is DRAM-only");
    source.close().unwrap();
    drop(source);
    assert!(last_file.exists(), "shutdown drained the last record");

    let (mut source, _) = open();
    assert_eq!(source.stats().disk_records, records.len());
    assert_eq!(source.stats().memory_records, 0);
    assert_eq!(source.pool().announcer().len(), records.len());
    wait("restarted source ready", || {
        source.pool().status() == PoolStatus::Ready
    });
    for (_, history) in &records {
        wait(
            "recovered record offered before the first source poll",
            || offered(&requester, history),
        );
    }
    for (index, (_, history)) in records.iter().enumerate() {
        let scratch: Vec<_> = (700..708).collect();
        let fetch = fetch_for(&mut requester, history, 8, &scratch);
        assert_eq!(fetch.part.candidate.tier, Tier::Nvme);
        requester.client.fetch(fetch).unwrap();
        let mut result = None;
        wait("recovered record claimed and copied", || {
            source.poll(&mut Vec::new()).unwrap();
            requester.poll();
            for event in requester.events.drain(..) {
                if let PoolEvent::Fetched { result: done, .. } = event {
                    result = Some(done.map_err(|failure| failure.error.to_string()));
                }
            }
            result.is_some()
        });
        assert_eq!(result.unwrap(), Ok((8 * PAGE + ROW) as u64));
        let salt = index as u32 + 1;
        for (page, id) in scratch.iter().enumerate() {
            assert_eq!(
                requester.gpu.pages.lock().unwrap()[id],
                salted_page(salt, page)
            );
        }
        assert_eq!(requester.gpu.rows.lock().unwrap()[&9], salted_row(salt));
    }
    source.close().unwrap();
    drop(source);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn large_recovered_nvme_records_restore_consecutively() {
    let Some(directory) = offload_directory("recovered-pair") else {
        return;
    };
    let mut manager = Manager::new();
    manager.config.staging = 128 << 20;
    manager.run();
    let mut requester = instance(&manager, 2);
    requester.ready();
    let open = || {
        let geometry = geo();
        let layout = Layout::new(&geometry).unwrap();
        let fingerprint = fingerprint(&geometry);
        let lock = record::DirectoryLock::acquire(&directory).unwrap();
        let scanned = record::scan(&lock, &[7; 32], &fingerprint, geometry[0].page_tokens, &record::rank_sizes(&geometry)).unwrap();
        let (copier, gpu) = SimulatedGpu::shared(&geometry).unwrap();
        let store = DiskStore::new(
            lock,
            [7; 32],
            fingerprint,
            256 << 20,
            geometry,
            scanned,
            Box::new(Staged::new(
                Some(SlabPool::staging(&layout).unwrap()),
                copier,
                DiskThread::start("recovered-pair").unwrap(),
            )),
        )
        .unwrap();
        (store, gpu)
    };
    let (mut disk, gpu) = open();
    let pages = 110_000 / PT as usize;
    let mut histories = Vec::new();
    for salt in 1..=2u32 {
        let mut history = TestHistory::new([7; 32]);
        history.append(&vec![salt; 110_000], &[]).unwrap();
        for page in 0..pages {
            gpu.pages
                .lock()
                .unwrap()
                .insert(page as u32, salted_page(salt, page));
        }
        gpu.rows.lock().unwrap().insert(salt, salted_row(salt));
        let checkpoint = PublishedCheckpoint {
            id: CheckpointId::new(u64::from(salt)),
            key: history.key_at(110_000).unwrap(),
            parent: None,
            bucket: history.bucket_at(110_000).unwrap(),
            chain_pages: (0..pages as u32).map(PageId).collect(),
            tail_pages: pages,
            slot: StateSlotId(salt),
        };
        store_and_wait(&mut disk, &checkpoint);
        histories.push(history);
    }
    disk.close().unwrap();
    drop(disk);
    let (disk, _) = open();
    let (copier, _) = SimulatedGpu::shared(&geo()).unwrap();
    let mut source =
        PooledStore::connect(disk, config(&manager, 1), copier, Arc::new(OkRegistrar)).unwrap();
    wait("recovered source ready", || {
        source.pool().status() == PoolStatus::Ready
    });
    source.poll(&mut Vec::new()).unwrap();
    for (index, history) in histories.iter().enumerate() {
        wait("recovered record offered", || {
            source.poll(&mut Vec::new()).unwrap();
            offered(&requester, history)
        });
        let scratch: Vec<_> = (0..pages as u32).collect();
        let fetch = fetch_for(&mut requester, history, pages, &scratch);
        assert_eq!(fetch.part.candidate.tier, Tier::Nvme);
        requester.client.fetch(fetch).unwrap();
        let mut result = None;
        wait("recovered restore", || {
            source.poll(&mut Vec::new()).unwrap();
            requester.poll();
            for event in requester.events.drain(..) {
                if let PoolEvent::Fetched { result: done, .. } = event {
                    result = Some(done.map_err(|f| f.error.to_string()));
                }
            }
            result.is_some()
        });
        assert_eq!(result.unwrap(), Ok((pages * PAGE + ROW) as u64));
        let salt = index as u32 + 1;
        let gpu = requester.gpu.pages.lock().unwrap();
        for page in 0..pages {
            assert_eq!(gpu[&(page as u32)], salted_page(salt, page));
        }
        assert_eq!(requester.gpu.rows.lock().unwrap()[&9], salted_row(salt));
    }
    source.close().unwrap();
}

// ---- instances on other hosts: the data channel ---------------------------------

/// Put a record through `writer`, restore it through `requester`, and check the level
/// the lookup gave it.
fn put_then_restore(manager: &Manager, carrier: &str) {
    let mut writer = instance(manager, 1);
    let mut requester = instance(manager, 2);
    writer.ready();
    requester.ready();
    // Neither instance has shared memory with the manager: no neighbour is offered.
    let history = history();
    let (request, _) = put_request(&history, 5);
    writer.client.put(request).unwrap();
    let outcome = writer.event("the put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert!(outcome.starts_with("Stored"), "{outcome}");
    wait_store_index(&requester, &history);
    let scratch = [200, 201, 202, 203, 204];
    let fetch = fetch_for(&mut requester, &history, 5, &scratch);
    assert_eq!(
        fetch.part.candidate.plan.level,
        flare_selene::levels::Level::RemoteNvme,
        "a manager on another host is the remote level"
    );
    requester.client.fetch(fetch).unwrap();
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((5 * PAGE + ROW) as u64));
    let gpu = requester.gpu.pages.lock().unwrap();
    for page in 0..5 {
        assert_eq!(gpu[&scratch[page]], expected_page(page), "page {page}");
    }
    assert_eq!(
        requester.gpu.rows.lock().unwrap()[&9],
        (5 * PAGE..5 * PAGE + ROW)
            .map(|i| (i * 13 + 5) as u8)
            .collect::<Vec<u8>>()
    );
    drop(gpu);
    // The same record again is a successful deduplication, and a second restore works.
    let (request, _) = put_request(&history, 5);
    writer.client.put(request).unwrap();
    let outcome = writer.event("the second put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert_eq!(outcome, "Present");
    restore(&mut requester, &history, 5, &[300, 301, 302, 303, 304]);
    writer.client.close().unwrap();
    requester.client.close().unwrap();
    // RDMA writes restore windows; uploads use the stream.
    let copies = if carrier == "rdma" { 2 } else { 3 };
    let (windows, bytes) = moved(manager, carrier);
    assert!(windows >= copies, "{carrier}: {windows} windows");
    assert_eq!(bytes, copies * (5 * PAGE + ROW) as u64, "{carrier}");
    if carrier == "rdma" {
        let (windows, bytes) = moved(manager, "stream");
        assert!(windows >= 1, "stream: {windows} windows");
        assert_eq!(bytes, (5 * PAGE + ROW) as u64);
    } else {
        assert_eq!(moved(manager, "rdma"), (0, 0), "nothing went by RDMA");
    }
}

#[test]
fn a_record_put_into_a_manager_on_another_host_is_restored_over_tcp() {
    let manager = network_manager(false);
    put_then_restore(&manager, "stream");
}

#[test]
fn a_record_put_into_a_manager_on_another_host_is_restored_over_tls() {
    let manager = network_manager(true);
    put_then_restore(&manager, "stream");
}

#[cfg(feature = "rdma")]
#[test]
fn a_record_put_into_a_manager_on_another_host_is_restored_over_rdma() {
    let manager = network_manager_with(false, true);
    put_then_restore(&manager, "rdma");
}

#[test]
fn instances_on_other_hosts_come_back_and_copy_after_the_manager_restarts() {
    let mut manager = network_manager(false);
    let mut writer = instance(&manager, 1);
    writer.ready();
    let history = history();
    let (request, _) = put_request(&history, 3);
    writer.client.put(request).unwrap();
    writer.event("the put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    manager.restart();
    wait("the instance to reconnect", || {
        writer.client.status() == PoolStatus::Ready
    });
    // The restarted manager starts empty: the put works again, and a second instance restores it.
    let (request, _) = put_request(&history, 3);
    writer.client.put(request).unwrap();
    let outcome = writer.event("the put after the restart", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert!(outcome.starts_with("Stored"), "{outcome}");
    let mut requester = instance(&manager, 2);
    requester.ready();
    wait_store_index(&requester, &history);
    restore(&mut requester, &history, 3, &[400, 401, 402]);
    writer.client.close().unwrap();
    requester.client.close().unwrap();
}

/// Pages that fill the manager's window exactly leave the row to a window of its own.
fn a_window_of_pages_with_the_row_alone(manager: &Manager) {
    let mut writer = instance(manager, 1);
    let mut requester = instance(manager, 2);
    writer.ready();
    requester.ready();
    // The manager's windows hold 4 MiB: 512 pages of 8 KiB fill one, and the row follows alone.
    let pages = 512usize;
    let history = history_of(pages as u32 * PT as u32 + 64);
    let (request, _) = put_request(&history, pages);
    writer.client.put(request).unwrap();
    let outcome = writer.event("the put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert!(outcome.starts_with("Stored"), "{outcome}");
    wait_store_index(&requester, &history);
    let scratch: Vec<u32> = (1000..1000 + pages as u32).collect();
    restore(&mut requester, &history, pages, &scratch);
    writer.client.close().unwrap();
    requester.client.close().unwrap();
}

#[test]
fn a_row_in_a_window_of_its_own_is_put_and_restored_over_tcp() {
    a_window_of_pages_with_the_row_alone(&network_manager(false));
}

#[test]
fn a_row_in_a_window_of_its_own_is_put_and_restored_through_shared_memory() {
    a_window_of_pages_with_the_row_alone(&Manager::start());
}

// ---- copies between instances through the manager ---------------------------------

/// A host pin that holds back every window after the first until the gate opens, and says
/// when it was dropped.
struct StallPin {
    inner: TestPin,
    gate: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

impl HostPin for StallPin {
    fn chain(&self) -> &HostChain {
        self.inner.chain()
    }
    fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums> {
        let deadline = Instant::now() + Duration::from_secs(20);
        while window.first_page > 0 && !self.gate.load(Ordering::Acquire) {
            if Instant::now() > deadline {
                return Err("the gate never opened".into());
            }
            thread::sleep(Duration::from_millis(5));
        }
        self.inner.read(window, dst)
    }
}

impl Drop for StallPin {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}

/// `source` saves a host record of `pages` pages; the manager offers it to `requester`.
fn offer_host_record(
    source: &Instance,
    requester: &Instance,
    history: &TestHistory,
    pages: usize,
) -> PrefixKey {
    let key = history.key_at(pages as u64 * PT).unwrap();
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Dram,
        bytes: (pages * PAGE + ROW) as u64,
    });
    wait("the manager to offer the record", || {
        let ticket = requester.client.begin_lookup(history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    });
    key
}

fn finish_restore(requester: &mut Instance) -> std::result::Result<u64, String> {
    requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    })
}

/// The pages and row a restore of `pages` pages of `expected_page` put on `requester`'s device.
fn check_restored(requester: &Instance, scratch: &[u32], pages: usize) {
    let gpu = requester.gpu.pages.lock().unwrap();
    for page in 0..pages {
        assert_eq!(gpu[&scratch[page]], expected_page(page), "page {page}");
    }
    assert_eq!(
        requester.gpu.rows.lock().unwrap()[&9],
        (pages * PAGE..pages * PAGE + ROW)
            .map(|i| (i * 13 + 5) as u8)
            .collect::<Vec<u8>>()
    );
}

/// A host record of `pages` pages moves from `source` to `requester` through the manager,
/// whichever of them are on the manager's host.
fn relay_host_record(manager: &Manager, source_local: bool, requester_local: bool, pages: usize) {
    let mut source = instance_on(manager, 1, source_local);
    let mut requester = instance_on(manager, 2, requester_local);
    source.ready();
    requester.ready();
    let history = history_of(pages as u32 * PT as u32 + 64);
    let key = offer_host_record(&source, &requester, &history, pages);
    let scratch: Vec<u32> = (0..pages as u32).map(|p| 5000 + p).collect();
    let pin = pin_for(key, pages);
    let fetch = fetch_for(&mut requester, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    let export = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => Some(request.export),
        _ => None,
    });
    source.client.host_pinned(export, pin);
    assert_eq!(
        finish_restore(&mut requester),
        Ok((pages * PAGE + ROW) as u64)
    );
    check_restored(&requester, &scratch, pages);
    source.client.close().unwrap();
    requester.client.close().unwrap();
}

#[test]
fn a_host_record_is_relayed_between_two_instances_on_other_hosts_over_tcp() {
    let manager = network_manager(false);
    // Two windows of a 4 MiB manager window each way.
    relay_host_record(&manager, false, false, 700);
    let (windows, bytes) = moved(&manager, "stream");
    assert!(windows >= 4, "{windows} windows crossed the manager twice");
    assert!(bytes >= 2 * (700 * PAGE + ROW) as u64, "{bytes} bytes");
}

#[cfg(feature = "rdma")]
#[test]
fn a_host_record_is_relayed_between_two_instances_on_other_hosts_over_rdma() {
    let manager = network_manager_with(false, true);
    relay_host_record(&manager, false, false, 700);
    let (windows, bytes) = moved(&manager, "rdma");
    assert!(windows >= 4, "{windows} windows crossed the manager twice");
    assert_eq!(bytes, (700 * PAGE + ROW) as u64);
    let (windows, bytes) = moved(&manager, "stream");
    assert!(windows >= 4, "{windows} upload windows crossed the manager");
    assert_eq!(bytes, (700 * PAGE + ROW) as u64);
}

#[test]
fn a_host_record_is_relayed_between_two_instances_on_other_hosts_over_tls() {
    let manager = network_manager(true);
    relay_host_record(&manager, false, false, 5);
}

#[test]
fn a_tls_manager_relays_between_unix_and_tls_clients() {
    let manager = network_manager(true);
    relay_host_record(&manager, false, true, 5);
    relay_host_record(&manager, true, false, 5);
}

#[test]
fn a_remote_instance_serves_an_instance_on_the_managers_host() {
    let manager = network_manager(false);
    relay_host_record(&manager, false, true, 700);
    let (windows, _) = moved(&manager, "stream");
    assert!(windows >= 2, "{windows} windows came in from the source");
}

#[test]
fn an_instance_on_the_managers_host_serves_a_remote_instance() {
    let manager = network_manager(false);
    relay_host_record(&manager, true, false, 700);
    let (windows, _) = moved(&manager, "stream");
    assert!(windows >= 2, "{windows} windows went out to the requester");
}

#[test]
fn a_gpu_checkpoint_on_another_host_is_relayed_through_reserved_staging() {
    let manager = network_manager(false);
    let mut source = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history();
    let pages = 3usize;
    let key = history.key_at(pages as u64 * PT).unwrap();
    for page in 0..pages {
        source
            .gpu
            .pages
            .lock()
            .unwrap()
            .insert(20 + page as u32, expected_page(page));
    }
    let row: Vec<u8> = (pages * PAGE..pages * PAGE + ROW)
        .map(|i| (i * 13 + 5) as u8)
        .collect();
    source.gpu.rows.lock().unwrap().insert(5, row);
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Gpu,
        bytes: (pages * PAGE + ROW) as u64,
    });
    wait("the manager to offer it", || {
        let ticket = requester.client.begin_lookup(&history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    });
    let scratch = [300, 301, 302];
    let fetch = fetch_for(&mut requester, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    let (export, reserved) = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => Some((request.export, request.reserved_bytes)),
        _ => None,
    });
    assert!(reserved >= (pages * PAGE + ROW) as u64);
    source.client.export_pinned(
        export,
        flare_selene::ExportChain {
            nodes: vec![flare_selene::ExportNode {
                key,
                pages: (20..20 + pages as u32).map(PageId).collect(),
            }],
            skip_pages: 0,
            row: Some(StateSlotId(5)),
        },
    );
    assert_eq!(
        finish_restore(&mut requester),
        Ok((pages * PAGE + ROW) as u64)
    );
    check_restored(&requester, &scratch, pages);
    source.event("the pin to be released", |e| match e {
        PoolEvent::ExportEnded { export: ended } if *ended == export => Some(()),
        _ => None,
    });
}

#[test]
fn a_stalled_source_fails_the_relay_and_the_same_record_relays_after() {
    let manager = network_manager(false);
    let pages = 700usize;
    let mut source = instance(&manager, 1);
    let mut requester = instance(&manager, 2);
    source.ready();
    requester.ready();
    let history = history_of(pages as u32 * PT as u32 + 64);
    let key = offer_host_record(&source, &requester, &history, pages);
    let scratch: Vec<u32> = (0..pages as u32).map(|p| 5000 + p).collect();
    let fetch = fetch_for(&mut requester, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    let export = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => Some(request.export),
        _ => None,
    });
    let gate = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let TestPinParts(inner) = TestPinParts::new(key, pages);
    source.client.host_pinned(
        export,
        Box::new(StallPin {
            inner,
            gate: gate.clone(),
            dropped: dropped.clone(),
        }),
    );
    // The second window never comes: the claim runs out of progress and fails the restore.
    let result = finish_restore(&mut requester);
    assert!(result.is_err(), "{result:?}");
    // The source lets go once its copy can finish, and releases the pin.
    gate.store(true, Ordering::Release);
    wait("the source to release its pin", || {
        source.poll();
        dropped.load(Ordering::Acquire)
    });
    // Nothing was left behind: the same record relays again.
    let pin = pin_for(key, pages);
    let fetch = fetch_for(&mut requester, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    let export = source.event("a second export request", |e| match e {
        PoolEvent::Export(request) => Some(request.export),
        _ => None,
    });
    source.client.host_pinned(export, pin);
    assert_eq!(
        finish_restore(&mut requester),
        Ok((pages * PAGE + ROW) as u64)
    );
    check_restored(&requester, &scratch, pages);
}

/// A `TestPin` built the way `pin_for` builds one.
struct TestPinParts(TestPin);

impl TestPinParts {
    fn new(key: PrefixKey, pages: usize) -> Self {
        let data: Vec<u8> = (0..pages * PAGE + ROW)
            .map(|i| (i * 13 + 5) as u8)
            .collect();
        let sums = (0..pages)
            .map(|p| checksum(&data[p * PAGE..(p + 1) * PAGE]))
            .collect();
        let row = checksum(&data[pages * PAGE..]);
        Self(TestPin {
            chain: HostChain {
                records: vec![HostRecord {
                    key,
                    first_page: 0,
                    pages: pages as u32,
                    row: true,
                }],
                total_pages: pages as u64,
                row: true,
            },
            bytes: Bytes { data, sums, row, page_bytes: PAGE, row_bytes: ROW },
        })
    }
}

#[path = "client/ranks.rs"]
mod ranks;
#[path = "client/tls.rs"]
mod tls;
