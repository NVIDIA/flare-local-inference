//! A multi-process instance against the real manager, in process: rank 0 and the peer
//! ranks' agents, each with its own simulated GPU, restore records a single-process
//! instance of the same shape holds, slice by slice.

use super::{Manager, OkRegistrar, PT, history_of, network_manager, remote, wait};
use flare_selene::shared::Scope;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use flare_selene::{
    CacheGeometry, History, PageId, PrefixKey, Result, StateSlotId,
    geometry::fingerprint,
    plan::{self, Identity, Request},
    pool::{
        AgentConfig, Checksums, FetchPart, HostChain, HostPin, HostRecord, HostWindow, IndexEntry,
        LookupState, PoolAddress, PoolClient, PoolConfig, PoolEvent, PoolFetch, PoolStatus,
        PutBytes, RankAgent, RankRecords, Tier, rank::memory_links,
    },
    record::checksum,
    testing::{GpuMemory, SimulatedGpu, TestHistory, geometry},
};

/// One rank's pages: two planes of 4 KiB, and a 4 KiB row.
fn local() -> CacheGeometry {
    geometry(PT, &[("k", 4096), ("v", 4096)], 4096)
}

fn ranks(world: usize) -> Vec<CacheGeometry> {
    vec![local(); world]
}

/// Bytes of a whole page record of a `world`-rank instance, and of its row.
fn whole(world: usize) -> (usize, usize) {
    (world * 8192, world * 4096)
}

fn scope(world: usize) -> Scope {
    let geometry = ranks(world);
    Scope::new([7; 32], fingerprint(&geometry), &geometry).unwrap()
}

fn config(manager: &Manager, world: usize, gpus: Vec<[u8; 16]>) -> PoolConfig {
    config_at(
        PoolAddress::Unix(manager.config.socket.clone()),
        world,
        gpus,
    )
}

fn config_at(address: PoolAddress, world: usize, gpus: Vec<[u8; 16]>) -> PoolConfig {
    PoolConfig {
        address,
        ca: None,
        scope: scope(world),
        gpu_uuids: gpus,
        export_slots: 2,
        claim_slots: 2,

        registration_context: 0,
        approve: None,
    }
}

fn bytes_of(world: usize, pages: usize) -> Vec<u8> {
    let (page, row) = whole(world);
    (0..pages * page + row)
        .map(|i| (i.wrapping_mul(31) + i / 4096 * 7 + 5) as u8)
        .collect()
}

use flare_selene::testing::{BytesPin as Pin, RecordBytes};

fn pin(world: usize, key: PrefixKey, pages: usize) -> Box<dyn HostPin> {
    let (page, row) = whole(world);
    let data = bytes_of(world, pages);
    let sums = (0..pages)
        .map(|p| checksum(&data[p * page..(p + 1) * page]))
        .collect();
    let row_sum = checksum(&data[pages * page..]);
    Box::new(Pin {
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
        bytes: RecordBytes { data, sums, row: row_sum, page_bytes: page, row_bytes: row },
    })
}

struct Source {
    client: PoolClient,
    _gpu: GpuMemory,
}

fn source(manager: &Manager, world: usize) -> Source {
    let (copier, gpu) = SimulatedGpu::shared(&ranks(world)).unwrap();
    let client = PoolClient::start(
        config(manager, world, vec![[1; 16]]),
        None,
        copier,
        Arc::new(OkRegistrar),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    Source { client, _gpu: gpu }
}

/// A multi-process requester: rank 0's client and the agents of ranks `1..world`.
struct Instance {
    rank0: PoolClient,
    memory: Vec<GpuMemory>,
    agents: Vec<RankAgent>,
    events: Vec<PoolEvent>,
}

/// How each rank of an instance reaches the manager: over its Unix socket (shared memory)
/// or over the network (its own data channel).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    Local,
    Network,
}

fn instance(manager: &Manager, world: usize) -> Instance {
    instance_reaching(manager, world, &vec![Reach::Local; world])
}

fn address_for(manager: &Manager, reach: Reach) -> PoolAddress {
    match reach {
        Reach::Local => PoolAddress::Unix(manager.config.socket.clone()),
        Reach::Network => remote(&manager).expect("a network manager").0,
    }
}

fn instance_reaching(manager: &Manager, world: usize, reach: &[Reach]) -> Instance {
    instance_seeded(manager, world, reach, 2)
}

/// As [`instance_reaching`]; the GPUs, and so the instance, differ with `seed`.
fn instance_seeded(manager: &Manager, world: usize, reach: &[Reach], seed: u8) -> Instance {
    instance_with(manager, world, reach, seed, &Faults::default())
}

/// What a test does to a rank of an instance.
#[derive(Default)]
struct Faults {
    /// Rank r's copier holds its replies while the gate is shut, and fails them when asked.
    gates: Vec<(usize, Arc<Gate>)>,
    /// Rank r's link to rank 0 can be cut.
    cuts: Vec<(usize, Arc<Mutex<Option<std::net::TcpStream>>>)>,
    /// The peers keep a host-tier record of this many pages in their own files; the rank
    /// listed has a wrong persisted checksum in its files.
    host: Option<(usize, Option<usize>)>,
    /// What the peers' agents register through.
    registrar: Option<Arc<super::CountingRegistrar>>,
    /// The peers' agents ask this before registering.
    approve: Option<flare_selene::pool::Approve>,
}

struct Gate {
    held: AtomicUsize,
    open: AtomicBool,
    fail: AtomicBool,
}

impl Gate {
    fn shut() -> Arc<Self> {
        Arc::new(Self {
            held: AtomicUsize::new(0),
            open: AtomicBool::new(false),
            fail: AtomicBool::new(false),
        })
    }
}

/// A simulated GPU whose completions wait for a gate.
struct GatedGpu {
    inner: SimulatedGpu,
    gate: Arc<Gate>,
    held: Vec<flare_selene::Reply>,
}

impl flare_selene::PageCopier for GatedGpu {
    fn layout(&self) -> &flare_selene::Layout {
        self.inner.layout()
    }
    fn submit(&mut self, job: flare_selene::Job) -> Result<()> {
        self.inner.submit(job)
    }
    fn poll(&mut self, replies: &mut Vec<flare_selene::Reply>) {
        self.inner.poll(&mut self.held);
        self.gate.held.store(self.held.len(), Ordering::SeqCst);
        if self.gate.open.load(Ordering::SeqCst) {
            for mut reply in self.held.drain(..) {
                if self.gate.fail.load(Ordering::SeqCst) {
                    reply.result = Err("injected copy failure".into());
                }
                replies.push(reply);
            }
        }
    }
    fn in_flight(&self) -> usize {
        self.inner.in_flight() + self.held.len()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

fn instance_with(
    manager: &Manager,
    world: usize,
    reach: &[Reach],
    seed: u8,
    faults: &Faults,
) -> Instance {
    let mut links = Vec::new();
    let mut peer_ends = Vec::new();
    for rank in 1..world {
        let (zero, peer) = memory_links();
        match faults.cuts.iter().find(|(r, _)| *r == rank) {
            Some((_, cut)) => {
                *cut.lock().unwrap() = Some(zero.shutdown_handle().unwrap());
                links.push(zero);
                peer_ends.push(peer);
            }
            None => {
                links.push(zero);
                peer_ends.push(peer);
            }
        }
    }
    let (copier, gpu0) = SimulatedGpu::shared(&[local()]).unwrap();
    let gpus: Vec<[u8; 16]> = (0..world).map(|r| [seed + r as u8; 16]).collect();
    let rank0 = PoolClient::start(
        config_at(address_for(manager, reach[0]), world, gpus.clone()),
        Some(links),
        copier,
        Arc::new(OkRegistrar),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    let mut memory = vec![gpu0];
    let mut agents = Vec::new();
    for (index, link) in peer_ends.into_iter().enumerate() {
        let rank = index + 1;
        let (copier, gpu) = match faults.gates.iter().find(|(r, _)| *r == rank) {
            Some((_, gate)) => {
                let (inner, gpu) = SimulatedGpu::new(&[local()]).unwrap();
                let copier: flare_selene::SharedCopier = Arc::new(std::sync::Mutex::new(GatedGpu {
                    inner,
                    gate: gate.clone(),
                    held: Vec::new(),
                }));
                (copier, gpu)
            }
            None => SimulatedGpu::shared(&[local()]).unwrap(),
        };
        memory.push(gpu);
        agents.push(
            RankAgent::start(AgentConfig {
                address: address_for(manager, reach[rank]),
                ca: None,
                scope: scope(world),
                rank,
                world,
                link,
                copier,
                registrar: match &faults.registrar {
                    Some(counting) => counting.clone(),
                    None => Arc::new(OkRegistrar),
                },
                registration_context: 0,
                gpu_uuid: gpus[rank],
                approve: faults.approve.clone(),
                records: faults.host.map(|(pages, bad)| -> Box<dyn RankRecords> {
                    Box::new(RankFiles {
                        world,
                        rank,
                        pages,
                        bad: bad == Some(rank),
                    })
                }),
            })
            .unwrap(),
        );
    }
    Instance {
        rank0,
        memory,
        agents,
        events: Vec::new(),
    }
}

impl Instance {
    fn ready(&self) {
        wait("rank 0 to be Ready", || {
            self.rank0.status() == PoolStatus::Ready
        });
    }

    fn poll(&mut self) {
        let mut events = Vec::new();
        self.rank0.poll(&mut events).unwrap();
        self.events.extend(events);
    }

    fn event<T>(&mut self, what: &str, mut take: impl FnMut(&PoolEvent) -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(20);
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

    fn close(mut self) {
        let _ = self.rank0.close();
        for agent in &mut self.agents { let _ = agent.close(); }
    }
}

fn identity(world: usize) -> Identity {
    Identity {
        model: [7; 32],
        layout: fingerprint(&ranks(world)),
    }
}

fn fetch_for(
    client: &PoolClient,
    world: usize,
    history: &TestHistory,
    pages: usize,
    scratch: &[u32],
) -> PoolFetch {
    let ticket = client.begin_lookup(history, 0);
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
        identity: &identity(world),
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

/// Every rank's destination pages and row hold exactly its slice of the source record.
fn assert_slices(instance: &Instance, world: usize, pages: usize, scratch: &[u32]) {
    let data = bytes_of(world, pages);
    let (page, row) = whole(world);
    for (rank, memory) in instance.memory.iter().enumerate() {
        let landed = memory.pages.lock().unwrap();
        for (i, id) in scratch.iter().enumerate() {
            let from = i * page + rank * 8192;
            assert_eq!(
                landed.get(id).map(Vec::as_slice),
                Some(&data[from..from + 8192]),
                "rank {rank} page {i}"
            );
        }
        let rows = memory.rows.lock().unwrap();
        let from = pages * page + rank * 4096;
        assert_eq!(
            rows.get(&9).map(Vec::as_slice),
            Some(&data[from..from + 4096]),
            "rank {rank}'s row (of {row})"
        );
    }
}

fn offer(source: &Source, history: &TestHistory, world: usize, pages: usize) -> PrefixKey {
    let key = history.key_at(pages as u64 * PT).unwrap();
    let (page, row) = whole(world);
    source.client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Dram,
        bytes: (pages * page + row) as u64,
    });
    key
}

#[test]
fn a_two_rank_instance_restores_a_single_process_record_slice_by_slice() {
    restore_through(&Manager::start(), 2, &[Reach::Local, Reach::Local]);
}

#[test]
fn every_rank_on_its_own_data_channel_restores_a_record() {
    let manager = network_manager(false);
    restore_through(&manager, 2, &[Reach::Network, Reach::Network]);
}

#[test]
fn three_ranks_mix_shared_memory_and_channels() {
    let manager = network_manager(false);
    restore_through(&manager, 3, &[Reach::Local, Reach::Network, Reach::Local]);
}

fn restore_through(manager: &Manager, world: usize, reach: &[Reach]) {
    let mut source = source(manager, world);
    let mut requester = instance_reaching(manager, world, reach);
    wait("the source", || source.client.status() == PoolStatus::Ready);
    requester.ready();
    let history = history_of(4000);
    let pages = 4;
    let key = offer(&source, &history, world, pages);
    let scratch = [100, 101, 102, 103];
    wait("the offer to be indexed", || {
        let ticket = requester.rank0.begin_lookup(&history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    });
    let fetch = fetch_for(&requester.rank0, world, &history, pages, &scratch);
    requester.rank0.fetch(fetch).unwrap();
    let export = loop {
        let mut events = Vec::new();
        source.client.poll(&mut events).unwrap();
        if let Some(request) = events.into_iter().find_map(|e| match e {
            PoolEvent::Export(request) => Some(request),
            _ => None,
        }) {
            break request;
        }
        thread::sleep(Duration::from_millis(2));
    };
    source
        .client
        .host_pinned(export.export, pin(world, key, pages));
    let (page, row) = whole(world);
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * page + row) as u64));
    assert_slices(&requester, world, pages, &scratch);
    requester.close();
    let _ = source.client.close();
}

// ---- exports from a multi-process instance ---------------------------------------

use flare_selene::{ExportChain, ExportNode};

/// Put a GPU checkpoint of `pages` pages and its row on every rank of `source`, in the
/// pages `first..` and slot 5, and announce it. Returns its key.
fn publish(source: &Instance, history: &TestHistory, world: usize, pages: usize) -> PrefixKey {
    let data = bytes_of(world, pages);
    let (page, _) = whole(world);
    for (rank, memory) in source.memory.iter().enumerate() {
        for p in 0..pages {
            let from = p * page + rank * 8192;
            memory
                .pages
                .lock()
                .unwrap()
                .insert(20 + p as u32, data[from..from + 8192].to_vec());
        }
        let from = pages * page + rank * 4096;
        memory
            .rows
            .lock()
            .unwrap()
            .insert(5, data[from..from + 4096].to_vec());
    }
    let key = history.key_at(pages as u64 * PT).unwrap();
    let (page, row) = whole(world);
    source.rank0.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Gpu,
        bytes: (pages * page + row) as u64,
    });
    key
}

fn wait_offered(client: &PoolClient, history: &TestHistory) {
    wait("the offer to be indexed", || {
        let ticket = client.begin_lookup(history, 0);
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(ticket.state(), LookupState::Pending) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(2));
        }
        matches!(ticket.state(), LookupState::Ready(a) if !a.candidates.is_empty())
    });
}

/// The worker's side of a GPU export: pin the chain, answer, and wait for the pin to end.
fn serve_export(source: &mut Instance, key: PrefixKey, pages: usize) {
    let export = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => {
            assert_eq!(request.tier, Tier::Gpu);
            Some(request.export)
        }
        _ => None,
    });
    source.rank0.export_pinned(
        export,
        ExportChain {
            nodes: vec![ExportNode {
                key,
                pages: (0..pages).map(|p| PageId(20 + p as u32)).collect(),
            }],
            skip_pages: 0,
            row: Some(StateSlotId(5)),
        },
    );
    source.event("the pin to be released", |e| match e {
        PoolEvent::ExportEnded { export: ended } if *ended == export => Some(()),
        _ => None,
    });
}

/// A plain single-process instance of the same shape, as a requester.
struct Plain {
    client: PoolClient,
    memory: GpuMemory,
}

fn plain(manager: &Manager, world: usize, reach: Reach) -> Plain {
    let (copier, memory) = SimulatedGpu::shared(&ranks(world)).unwrap();
    let client = PoolClient::start(
        config_at(address_for(manager, reach), world, vec![[9; 16]]),
        None,
        copier,
        Arc::new(OkRegistrar),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    Plain { client, memory }
}

fn expect_fetched(client: &mut PoolClient, what: &str, want: u64) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let mut events = Vec::new();
        client.poll(&mut events).unwrap();
        for event in events {
            if let PoolEvent::Fetched { result, .. } = event {
                assert_eq!(
                    result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()),
                    Ok(want),
                    "{what}"
                );
                return;
            }
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(2));
    }
}

fn export_to_plain(manager: &Manager, world: usize, source_reach: &[Reach], plain_reach: Reach) {
    let mut source = instance_reaching(manager, world, source_reach);
    let mut requester = plain(manager, world, plain_reach);
    source.ready();
    wait("the requester", || {
        requester.client.status() == PoolStatus::Ready
    });
    let history = history_of(4000);
    let pages = 4;
    let key = publish(&source, &history, world, pages);
    wait_offered(&requester.client, &history);
    let scratch = [300, 301, 302, 303];
    let fetch = fetch_for(&requester.client, world, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    serve_export(&mut source, key, pages);
    let (page, row) = whole(world);
    expect_fetched(
        &mut requester.client,
        "the restore",
        (pages * page + row) as u64,
    );
    let data = bytes_of(world, pages);
    let landed = requester.memory.pages.lock().unwrap();
    for (i, id) in scratch.iter().enumerate() {
        assert_eq!(
            landed.get(id).map(Vec::as_slice),
            Some(&data[i * page..(i + 1) * page]),
            "page {i} of the whole record"
        );
    }
    assert_eq!(
        requester
            .memory
            .rows
            .lock()
            .unwrap()
            .get(&9)
            .map(Vec::as_slice),
        Some(&data[pages * page..]),
        "the whole row"
    );
    drop(landed);
    source.close();
    let _ = requester.client.close();
}

#[test]
fn a_two_rank_instance_exports_a_gpu_checkpoint_to_a_single_process_instance() {
    export_to_plain(
        &Manager::start(),
        2,
        &[Reach::Local, Reach::Local],
        Reach::Local,
    );
}

#[test]
fn a_whole_instance_on_channels_exports_to_a_remote_single_process_instance() {
    let manager = network_manager(false);
    export_to_plain(&manager, 3, &[Reach::Network; 3], Reach::Network);
}

#[test]
fn three_ranks_mixing_shared_memory_and_channels_export_to_a_multi_process_instance() {
    let manager = network_manager(false);
    let world = 3;
    let mut source = instance_reaching(
        &manager,
        world,
        &[Reach::Local, Reach::Network, Reach::Local],
    );
    let mut requester = instance_seeded(
        &manager,
        world,
        &[Reach::Local, Reach::Local, Reach::Network],
        40,
    );
    source.ready();
    requester.ready();
    let history = history_of(4000);
    let pages = 4;
    let key = publish(&source, &history, world, pages);
    wait_offered(&requester.rank0, &history);
    let scratch = [300, 301, 302, 303];
    let fetch = fetch_for(&requester.rank0, world, &history, pages, &scratch);
    requester.rank0.fetch(fetch).unwrap();
    serve_export(&mut source, key, pages);
    let (page, row) = whole(world);
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * page + row) as u64));
    assert_slices(&requester, world, pages, &scratch);
    source.close();
    requester.close();
}

// ---- puts from a multi-process instance -------------------------------------------

use flare_selene::pool::{PutRequest, PutSource};

// ---- faults ------------------------------------------------------------------------

/// A single-process source offering a four-page host record, a multi-process requester
/// with `faults`, and a fetch of the record that the source's pin has not yet served.
struct Scenario {
    source: Source,
    requester: Instance,
    world: usize,
    pages: usize,
    key: PrefixKey,
    scratch: [u32; 4],
}

fn scenario(manager: &Manager, world: usize, reach: &[Reach], faults: &Faults) -> Scenario {
    let source = source(manager, world);
    let mut requester = instance_with(manager, world, reach, 2, faults);
    wait("the source", || source.client.status() == PoolStatus::Ready);
    requester.ready();
    let history = history_of(4000);
    let pages = 4;
    let key = offer(&source, &history, world, pages);
    wait_offered(&requester.rank0, &history);
    let scratch = [100, 101, 102, 103];
    let fetch = fetch_for(&requester.rank0, world, &history, pages, &scratch);
    requester.rank0.fetch(fetch).unwrap();
    Scenario {
        source,
        requester,
        world,
        pages,
        key,
        scratch,
    }
}

impl Scenario {
    /// The source pins and serves the record, with `pin` unless a test supplies its own.
    fn serve(&mut self, pin_for: Option<Box<dyn HostPin>>) {
        let export = loop {
            let mut events = Vec::new();
            self.source.client.poll(&mut events).unwrap();
            if let Some(request) = events.into_iter().find_map(|e| match e {
                PoolEvent::Export(request) => Some(request),
                _ => None,
            }) {
                break request;
            }
            thread::sleep(Duration::from_millis(2));
        };
        let pin = pin_for.unwrap_or_else(|| pin(self.world, self.key, self.pages));
        self.source.client.host_pinned(export.export, pin);
    }

    fn fetched(&mut self) -> std::result::Result<u64, String> {
        self.requester.event("the restore", |e| match e {
            PoolEvent::Fetched { result, .. } => {
                Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
            }
            _ => None,
        })
    }

    /// No further `Fetched` is ever reported for this restore.
    fn assert_no_more_events(&mut self) {
        let history = history_of(4000);
        let ticket = self.requester.rank0.begin_lookup(&history, 0);
        wait("an ordered lookup after the verdict", || {
            !matches!(ticket.state(), LookupState::Pending)
        });
        self.requester.poll();
        assert!(
            !self
                .requester
                .events
                .iter()
                .any(|e| matches!(e, PoolEvent::Fetched { .. })),
            "a second result"
        );
    }
}

#[test]
fn a_restore_is_reported_only_after_every_rank_drained_and_no_rank_is_read_early() {
    let manager = Manager::start();
    let gate = Gate::shut();
    let faults = Faults {
        gates: vec![(1, gate.clone())],
        cuts: Vec::new(),
        host: None,
        registrar: None,
        approve: None,
    };
    let mut scenario = scenario(&manager, 2, &[Reach::Local, Reach::Local], &faults);
    scenario.serve(None);
    // Rank 0 finishes its own copies; rank 1's replies are held by its gate.
    wait("rank 0 to have its pages", || {
        scenario.requester.memory[0].pages.lock().unwrap().len() == scenario.pages
    });
    wait("rank 1 to have started its copies", || {
        !scenario.requester.memory[1]
            .pages
            .lock()
            .unwrap()
            .is_empty()
    });
    wait("rank 1 to hold its completions", || {
        gate.held.load(Ordering::SeqCst) > 0
    });
    scenario.requester.poll();
    assert!(
        !scenario
            .requester
            .events
            .iter()
            .any(|e| matches!(e, PoolEvent::Fetched { .. })),
        "nothing is reported while a rank has not drained: {:?}",
        scenario.requester.events
    );
    gate.open.store(true, Ordering::SeqCst);
    let (page, row) = whole(2);
    assert_eq!(scenario.fetched(), Ok((scenario.pages * page + row) as u64));
    assert_slices(&scenario.requester, 2, scenario.pages, &scenario.scratch);
    scenario.assert_no_more_events();
    scenario.requester.close();
}

#[test]
fn a_rank_whose_copy_fails_fails_the_whole_restore_once_and_the_next_restore_works() {
    let manager = Manager::start();
    let gate = Gate::shut();
    gate.fail.store(true, Ordering::SeqCst);
    gate.open.store(true, Ordering::SeqCst);
    let faults = Faults {
        gates: vec![(1, gate.clone())],
        cuts: Vec::new(),
        host: None,
        registrar: None,
        approve: None,
    };
    let mut scenario = scenario(&manager, 2, &[Reach::Local, Reach::Local], &faults);
    scenario.serve(None);
    let failure = scenario.fetched().expect_err("a rank failed");
    assert!(failure.contains("rank 1"), "{failure}");
    scenario.assert_no_more_events();
    // Nothing is left leased: the same restore, with the rank healthy again, succeeds.
    gate.fail.store(false, Ordering::SeqCst);
    let history = history_of(4000);
    let scratch = [200, 201, 202, 203];
    let fetch = fetch_for(
        &scenario.requester.rank0,
        2,
        &history,
        scenario.pages,
        &scratch,
    );
    scenario.requester.rank0.fetch(fetch).unwrap();
    scenario.serve(None);
    let (page, row) = whole(2);
    assert_eq!(scenario.fetched(), Ok((scenario.pages * page + row) as u64));
    assert_slices(&scenario.requester, 2, scenario.pages, &scratch);
    scenario.requester.close();
}

#[test]
fn a_cut_link_with_a_rank_still_copying_stops_the_pool_and_reports_nothing() {
    let manager = Manager::start();
    let gate = Gate::shut();
    let cut = Arc::new(Mutex::new(None));
    let faults = Faults {
        gates: vec![(1, gate.clone())],
        cuts: vec![(1, cut.clone())],
        host: None,
        registrar: None,
        approve: None,
    };
    let mut scenario = scenario(&manager, 2, &[Reach::Local, Reach::Local], &faults);
    scenario.serve(None);
    wait("rank 0 to have its pages", || {
        scenario.requester.memory[0].pages.lock().unwrap().len() == scenario.pages
    });
    wait("rank 1 to have started its copies", || {
        !scenario.requester.memory[1]
            .pages
            .lock()
            .unwrap()
            .is_empty()
    });
    cut.lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .shutdown(std::net::Shutdown::Both)
        .unwrap();
    // The pool can no longer account for rank 1's copies: polling fails, as a copy that
    // cannot be drained does, and no result is ever given for pages it may still write.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let mut events = Vec::new();
        match scenario.requester.rank0.poll(&mut events) {
            Err(_) => break,
            Ok(()) => assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, PoolEvent::Fetched { .. })),
                "a result was given for pages a rank may still write"
            ),
        }
        assert!(Instant::now() < deadline, "the pool was not stopped");
        thread::sleep(Duration::from_millis(5));
    }
    // Let the stuck rank finish so the test can clean up.
    gate.open.store(true, Ordering::SeqCst);
    let _ = scenario.requester.rank0.close();
}

#[test]
fn a_stored_checksum_that_is_wrong_fails_the_restore_before_any_rank_copies() {
    let manager = Manager::start();
    let mut scenario = scenario(
        &manager,
        2,
        &[Reach::Local, Reach::Local],
        &Faults::default(),
    );
    // The source's pin states a wrong persisted checksum for page 2.
    let mut bad = pin(2, scenario.key, scenario.pages);
    // SAFETY of the downcast: `pin` builds a `Pin`; flip a sum through its serving side by
    // reading through a wrapper instead.
    struct Corrupt(Box<dyn HostPin>);
    impl HostPin for Corrupt {
        fn chain(&self) -> &HostChain {
            self.0.chain()
        }
        fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums> {
            let mut sums = self.0.read(window, dst)?;
            if let Some(sum) = sums
                .pages
                .get_mut(2usize.saturating_sub(window.first_page as usize))
            {
                *sum ^= 1;
            }
            Ok(sums)
        }
    }
    bad = Box::new(Corrupt(bad));
    scenario.serve(Some(bad));
    let failure = scenario.fetched().expect_err("the checksum is wrong");
    assert!(
        failure.contains("Checksum") || failure.contains("Failed"),
        "{failure}"
    );
    for (rank, memory) in scenario.requester.memory.iter().enumerate() {
        assert!(
            memory.pages.lock().unwrap().is_empty(),
            "rank {rank} copied a page of a window that failed its checksum"
        );
    }
    scenario.assert_no_more_events();
    scenario.requester.close();
}

#[test]
fn a_manager_that_restarts_is_reattached_by_every_rank_and_restores_again() {
    let mut manager = Manager::start();
    let mut requester = instance(&manager, 2);
    requester.ready();
    manager.restart();
    // Rank 0 reconnects and every agent attaches again, then a restore from a fresh source
    // works.
    let mut scenario_source = source(&manager, 2);
    wait("the source", || {
        scenario_source.client.status() == PoolStatus::Ready
    });
    wait("rank 0 to be Ready again", || {
        requester.rank0.status() == PoolStatus::Ready
    });
    let history = history_of(4000);
    let pages = 4;
    let key = offer(&scenario_source, &history, 2, pages);
    wait_offered(&requester.rank0, &history);
    let scratch = [100, 101, 102, 103];
    let fetch = fetch_for(&requester.rank0, 2, &history, pages, &scratch);
    requester.rank0.fetch(fetch).unwrap();
    let export = loop {
        let mut events = Vec::new();
        scenario_source.client.poll(&mut events).unwrap();
        if let Some(request) = events.into_iter().find_map(|e| match e {
            PoolEvent::Export(request) => Some(request),
            _ => None,
        }) {
            break request;
        }
        thread::sleep(Duration::from_millis(2));
    };
    scenario_source
        .client
        .host_pinned(export.export, pin(2, key, pages));
    let (page, row) = whole(2);
    let result = requester.event("the restore", |e| match e {
        PoolEvent::Fetched { result, .. } => {
            Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
        }
        _ => None,
    });
    assert_eq!(result, Ok((pages * page + row) as u64));
    assert_slices(&requester, 2, pages, &scratch);
    requester.close();
}

// ---- the host tier ------------------------------------------------------------------

/// One rank's record files: its slices of the checkpoint `bytes_of` makes, with the slice
/// checksums persisted beside them (one of which is wrong when `bad`).
struct RankFiles {
    world: usize,
    rank: usize,
    pages: usize,
    bad: bool,
}

impl RankFiles {
    fn serve(&self, first_page: u32, pages: u32, row: bool, dst: &mut [u8]) -> Result<Checksums> {
        let data = bytes_of(self.world, self.pages);
        let (page, _) = whole(self.world);
        let (mut at, mut sums) = (0, Vec::new());
        for p in first_page as usize..(first_page + pages) as usize {
            let from = p * page + self.rank * 8192;
            dst[at..at + 8192].copy_from_slice(&data[from..from + 8192]);
            sums.push(checksum(&data[from..from + 8192]));
            at += 8192;
        }
        let row_sum = row.then(|| {
            let from = self.pages * page + self.rank * 4096;
            dst[at..at + 4096].copy_from_slice(&data[from..from + 4096]);
            checksum(&data[from..from + 4096])
        });
        if self.bad {
            sums[0] ^= 1;
        }
        Ok(Checksums {
            pages: sums,
            row: row_sum,
        })
    }
}

impl RankRecords for RankFiles {
    fn read(
        &mut self,
        _key: PrefixKey,
        first_page: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        self.serve(first_page, pages, row, dst)
    }
    fn fill(
        &mut self,
        _key: PrefixKey,
        first_page: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        self.serve(first_page, pages, row, dst)
    }
}

/// Rank 0's pin over its own files, as a store hands the engine: its slices, packed.
struct ZeroPin {
    chain: HostChain,
    files: RankFiles,
}

impl HostPin for ZeroPin {
    fn chain(&self) -> &HostChain {
        &self.chain
    }
    fn read(&mut self, window: &HostWindow, dst: &mut [u8]) -> Result<Checksums> {
        self.files
            .serve(window.first_page, window.page_count, window.row, dst)
    }
}

fn zero_pin(world: usize, key: PrefixKey, pages: usize) -> Box<dyn HostPin> {
    Box::new(ZeroPin {
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
        files: RankFiles {
            world,
            rank: 0,
            pages,
            bad: false,
        },
    })
}

/// Rank 0's bytes of a host put.
struct ZeroBytes(RankFiles);

impl PutBytes for ZeroBytes {
    fn fill(
        &mut self,
        first_page: u64,
        page_count: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        self.0.serve(first_page as u32, page_count, row, dst)
    }
}

/// A multi-process source whose records are in its ranks' files exports one to a plain
/// instance; `bad` names a rank whose persisted checksum is wrong. Returns the fetch result.
fn host_export(
    manager: &Manager,
    world: usize,
    source_reach: &[Reach],
    bad: Option<usize>,
) -> std::result::Result<u64, String> {
    let pages = 4;
    let faults = Faults {
        host: Some((pages, bad)),
        ..Default::default()
    };
    let mut source = instance_with(manager, world, source_reach, 2, &faults);
    let mut requester = plain(manager, world, Reach::Local);
    source.ready();
    wait("the requester", || {
        requester.client.status() == PoolStatus::Ready
    });
    let history = history_of(4000);
    let key = history.key_at(pages as u64 * PT).unwrap();
    let (page, row) = whole(world);
    source.rank0.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: pages as u32,
        row: true,
        tier: Tier::Dram,
        bytes: (pages * page + row) as u64,
    });
    wait_offered(&requester.client, &history);
    let scratch = [300, 301, 302, 303];
    let fetch = fetch_for(&requester.client, world, &history, pages, &scratch);
    requester.client.fetch(fetch).unwrap();
    let export = source.event("an export request", |e| match e {
        PoolEvent::Export(request) => Some(request.export),
        _ => None,
    });
    source
        .rank0
        .host_pinned(export, zero_pin(world, key, pages));
    let deadline = Instant::now() + Duration::from_secs(20);
    let result = loop {
        let mut events = Vec::new();
        requester.client.poll(&mut events).unwrap();
        source.poll();
        if let Some(result) = events.into_iter().find_map(|e| match e {
            PoolEvent::Fetched { result, .. } => {
                Some(result.as_ref().map(|b| *b).map_err(|f| f.error.to_string()))
            }
            _ => None,
        }) {
            break result;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the restore"
        );
        thread::sleep(Duration::from_millis(2));
    };
    if result.is_ok() {
        let data = bytes_of(world, pages);
        let landed = requester.memory.pages.lock().unwrap();
        for (i, id) in scratch.iter().enumerate() {
            assert_eq!(
                landed.get(id).map(Vec::as_slice),
                Some(&data[i * page..(i + 1) * page]),
                "page {i} of the whole record"
            );
        }
        assert_eq!(
            requester
                .memory
                .rows
                .lock()
                .unwrap()
                .get(&9)
                .map(Vec::as_slice),
            Some(&data[pages * page..]),
            "the whole row"
        );
    }
    source.close();
    let _ = requester.client.close();
    result
}

#[test]
fn a_two_rank_instance_exports_a_record_from_its_ranks_files() {
    let (page, row) = whole(2);
    assert_eq!(
        host_export(&Manager::start(), 2, &[Reach::Local, Reach::Local], None),
        Ok((4 * page + row) as u64)
    );
}

#[test]
fn three_ranks_mixing_shared_memory_and_channels_export_from_their_files() {
    let manager = network_manager(false);
    let (page, row) = whole(3);
    assert_eq!(
        host_export(
            &manager,
            3,
            &[Reach::Network, Reach::Local, Reach::Network],
            None
        ),
        Ok((4 * page + row) as u64)
    );
}

#[test]
fn a_peers_wrong_persisted_checksum_fails_the_export_from_its_files() {
    for reach in [Reach::Local, Reach::Network] {
        let manager = network_manager(false);
        let result = host_export(&manager, 2, &[Reach::Local, reach], Some(1));
        assert!(result.is_err(), "a corrupted file reached the requester");
    }
}

/// A multi-process writer puts a record from its ranks' files; a plain instance restores it.
fn host_put(manager: &Manager, world: usize, reach: &[Reach]) {
    let pages = 4;
    let faults = Faults {
        host: Some((pages, None)),
        ..Default::default()
    };
    let mut writer = instance_with(manager, world, reach, 2, &faults);
    writer.ready();
    let history = history_of(4000);
    let key = history.key_at(pages as u64 * PT).unwrap();
    writer
        .rank0
        .put(PutRequest {
            key,
            parent: None,
            bucket: history.bucket_at(key.consumed_tokens).unwrap(),
            chain_pages: pages as u64,
            tail_pages: pages as u64,
            row: true,
            source: PutSource::Host(Box::new(ZeroBytes(RankFiles {
                world,
                rank: 0,
                pages,
                bad: false,
            }))),
        })
        .unwrap();
    let outcome = writer.event("the put", |e| match e {
        PoolEvent::Put { outcome, .. } => Some(format!("{outcome:?}")),
        _ => None,
    });
    assert!(outcome.starts_with("Stored"), "{outcome}");
    let (page, row) = whole(world);
    let mut plain = plain(manager, world, Reach::Local);
    wait("the reader", || plain.client.status() == PoolStatus::Ready);
    wait_offered(&plain.client, &history);
    let scratch = [300, 301, 302, 303];
    let fetch = fetch_for(&plain.client, world, &history, pages, &scratch);
    plain.client.fetch(fetch).unwrap();
    expect_fetched(
        &mut plain.client,
        "the restore",
        (pages * page + row) as u64,
    );
    let data = bytes_of(world, pages);
    let landed = plain.memory.pages.lock().unwrap();
    for (i, id) in scratch.iter().enumerate() {
        assert_eq!(
            landed.get(id).map(Vec::as_slice),
            Some(&data[i * page..(i + 1) * page]),
            "page {i}"
        );
    }
    assert_eq!(
        plain.memory.rows.lock().unwrap().get(&9).map(Vec::as_slice),
        Some(&data[pages * page..])
    );
    drop(landed);
    let _ = plain.client.close();
    writer.close();
}

#[test]
fn a_two_rank_instance_spills_a_record_from_its_ranks_files() {
    host_put(&Manager::start(), 2, &[Reach::Local, Reach::Local]);
}

#[test]
fn three_ranks_mixing_shared_memory_and_channels_spill_their_files_slices() {
    let manager = network_manager(false);
    host_put(&manager, 3, &[Reach::Local, Reach::Network, Reach::Local]);
}

#[test]
fn an_agent_registers_at_startup_and_never_again_across_a_manager_restart() {
    let mut manager = Manager::start();
    let counting = Arc::new(super::CountingRegistrar::default());
    let faults = Faults {
        registrar: Some(counting.clone()),
        ..Default::default()
    };
    let requester = instance_with(&manager, 2, &[Reach::Local, Reach::Local], 2, &faults);
    requester.ready();
    assert_eq!(
        requester.agents[0]
            .registered_regions()
            .iter()
            .map(|r| r.name)
            .collect::<Vec<_>>(),
        ["arena", "staging"],
        "the agent registered the manager's two regions at startup"
    );
    assert_eq!(counting.calls.lock().unwrap().len(), 2);
    manager.stop();
    wait("rank 0 and its agent to disconnect", || {
        requester.rank0.status() != PoolStatus::Ready && requester.agents[0].status() != PoolStatus::Ready
    });
    manager.run();
    wait("rank 0 and its agent to attach again", || {
        requester.rank0.status() == PoolStatus::Ready
            && requester.agents[0].status() == PoolStatus::Ready
    });
    assert_eq!(
        counting.calls.lock().unwrap().len(),
        2,
        "nothing was registered after startup"
    );
    requester.close();
}

#[test]
fn an_agent_is_asked_about_its_whole_plan_before_it_registers_anything() {
    let manager = Manager::start();
    let counting = Arc::new(super::CountingRegistrar::default());
    let asked = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (seen, registered_when_asked) = (asked.clone(), counting.clone());
    let faults = Faults {
        registrar: Some(counting.clone()),
        approve: Some(Arc::new(move |plan: &[flare_selene::pool::PlannedRegion]| {
            seen.lock().unwrap().push((
                plan.iter().map(|r| r.name).collect::<Vec<_>>(),
                registered_when_asked.calls.lock().unwrap().len(),
            ));
            Ok(())
        })),
        ..Default::default()
    };
    let requester = instance_with(&manager, 2, &[Reach::Local, Reach::Local], 2, &faults);
    requester.ready();
    assert_eq!(*asked.lock().unwrap(), vec![(vec!["arena", "staging"], 0)]);
    assert_eq!(counting.calls.lock().unwrap().len(), 2);
    requester.close();
}

#[test]
fn an_agent_whose_plan_is_refused_registers_nothing_and_turns_the_pool_off_on_every_rank() {
    let manager = Manager::start();
    let counting = Arc::new(super::CountingRegistrar::default());
    let faults = Faults {
        registrar: Some(counting.clone()),
        approve: Some(Arc::new(|_: &[flare_selene::pool::PlannedRegion]| {
            Err("over the memory budget".to_string())
        })),
        ..Default::default()
    };
    let requester = instance_with(&manager, 2, &[Reach::Local, Reach::Local], 2, &faults);
    wait("the agent to be disabled", || {
        matches!(requester.agents[0].status(), PoolStatus::Disabled(_))
    });
    assert!(counting.calls.lock().unwrap().is_empty());
    assert!(requester.agents[0].registered_regions().is_empty());
    wait("rank 0 to turn the pool off", || {
        matches!(requester.rank0.status(), PoolStatus::Disabled(_))
    });
    requester.close();
}

#[test]
fn when_rank_zero_turns_the_pool_off_its_agents_do_too() {
    let manager = Manager::start();
    let (copier, _gpu) = SimulatedGpu::shared(&[local()]).unwrap();
    let (zero, peer) = memory_links();
    let mut config = config(&manager, 2, vec![[2; 16], [3; 16]]);
    config.approve = Some(Arc::new(|_: &[flare_selene::pool::PlannedRegion]| {
        Err("over the memory budget".to_string())
    }));
    let rank0 = PoolClient::start(
        config,
        Some(vec![zero]),
        copier,
        Arc::new(OkRegistrar),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    let (copier, _gpu1) = SimulatedGpu::shared(&[local()]).unwrap();
    let agent = RankAgent::start(AgentConfig {
        address: PoolAddress::Unix(manager.config.socket.clone()),
        ca: None,
        scope: scope(2),
        rank: 1,
        world: 2,
        link: peer,
        copier,
        registrar: Arc::new(OkRegistrar),
        registration_context: 0,
        gpu_uuid: [3; 16],
        records: None,
        approve: None,
    })
    .unwrap();
    wait("rank 0 to be disabled", || {
        matches!(rank0.status(), PoolStatus::Disabled(_))
    });
    wait("the agent to follow", || {
        matches!(agent.status(), PoolStatus::Disabled(_))
    });
}
