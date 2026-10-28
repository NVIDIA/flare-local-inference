//! One process of a multi-process pool test: a single-process source, rank 0 of a
//! multi-process instance, or the agent of one of its peer ranks. Its harness
//! (`tests/rank_processes.rs`) starts several, talks to them over stdin and stdout, and kills
//! them; the GPUs are simulated.
//!
//! ```text
//! rank_node source --socket PATH --world W
//! rank_node rank0  --socket PATH --world W
//! rank_node agent  --socket PATH --world W --rank R --link HOST:PORT [--hold-copy true]
//! ```
//!
//! Rank 0 listens for its agents' links on a loopback port and prints `link-port N`. Each node
//! prints `ready` once it can serve and then answers its commands, one per line: `restore`
//! (rank 0: restore the source's record), `verify` (check this rank's slice of what was
//! restored), `poll` (rank 0: report transfer health), `release` (agent: release held copies),
//! and `quit`. Results are `fetched ok N`, `fetched err WHY`, `verify ok`, `poll ok`, and `poll err`.

use std::{
    io::{BufRead, Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use flare_selene::{
    CacheGeometry, History, Job, Layout, PageCopier, PageId, Reply, Result, StateSlotId,
    geometry::fingerprint,
    plan::{self, Identity, Request},
    pool::{
        AgentConfig, FetchPart, HostChain, HostPin, HostRecord, IndexEntry, LookupState,
        PoolAddress, PoolClient, PoolConfig, PoolEvent, PoolFetch, PoolStatus, RankAgent, RankLink,
    },
    record::checksum,
    shared::{Scope, Tier},
    testing::{GpuMemory, SimulatedGpu, TestHistory, geometry},
};

const PT: u64 = 16;
const PAGES: usize = 4;
const SCRATCH: [u32; 4] = [100, 101, 102, 103];

use flare_selene::testing::IdentityRegistrar as NoRegistration;

/// Hold copy completions until the harness releases them.
struct Held {
    inner: SimulatedGpu,
    release: Arc<std::sync::atomic::AtomicBool>,
    held: Vec<Reply>,
}

impl PageCopier for Held {
    fn layout(&self) -> &Layout {
        self.inner.layout()
    }
    fn submit(&mut self, job: Job) -> Result<()> {
        self.inner.submit(job)?;
        say("copy pending");
        Ok(())
    }
    fn poll(&mut self, replies: &mut Vec<Reply>) {
        self.inner.poll(&mut self.held);
        if self.release.load(std::sync::atomic::Ordering::Acquire) {
            replies.append(&mut self.held);
        }
    }
    fn in_flight(&self) -> usize {
        self.inner.in_flight() + self.held.len()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

fn link(stream: TcpStream) -> RankLink {
    RankLink::new(stream).expect("rank connection")
}

// ---- shapes -------------------------------------------------------------------------

fn local() -> CacheGeometry {
    geometry(PT, &[("k", 4096), ("v", 4096)], 4096)
}

fn ranks(world: usize) -> Vec<CacheGeometry> {
    vec![local(); world]
}

fn scope(world: usize) -> Scope {
    let g = ranks(world);
    Scope::new([7; 32], fingerprint(&g), &g).unwrap()
}

fn whole(world: usize) -> (usize, usize) {
    (world * 8192, world * 4096)
}

fn bytes_of(world: usize) -> Vec<u8> {
    let (page, row) = whole(world);
    (0..PAGES * page + row)
        .map(|i| (i.wrapping_mul(31) + i / 4096 * 7 + 5) as u8)
        .collect()
}

fn history() -> TestHistory {
    let mut history = TestHistory::new([7; 32]);
    history
        .append(&(0..4000).collect::<Vec<u32>>(), &[])
        .unwrap();
    history
}

use flare_selene::testing::{BytesPin, RecordBytes};

fn pin(world: usize, key: flare_selene::PrefixKey) -> Box<dyn HostPin> {
    let (page, row) = whole(world);
    Box::new(BytesPin {
        chain: HostChain {
            records: vec![HostRecord {
                key,
                first_page: 0,
                pages: PAGES as u32,
                row: true,
            }],
            total_pages: PAGES as u64,
            row: true,
        },
        bytes: {
            let data = bytes_of(world);
            let sums = data[..PAGES * page]
                .chunks_exact(page)
                .map(checksum)
                .collect();
            let row_sum = checksum(&data[PAGES * page..]);
            RecordBytes {
                data,
                sums,
                row: row_sum,
                page_bytes: page,
                row_bytes: row,
            }
        },
    })
}

// ---- arguments ------------------------------------------------------------------------

struct Args {
    role: String,
    socket: PathBuf,
    world: usize,
    rank: usize,
    link: String,
    hold: bool,
}

fn args() -> Args {
    let mut it = std::env::args().skip(1);
    let role = it.next().expect("a role");
    let mut a = Args {
        role,
        socket: PathBuf::new(),
        world: 2,
        rank: 1,
        link: String::new(),
        hold: false,
    };
    while let Some(flag) = it.next() {
        let value = it.next().expect("a value");
        match flag.as_str() {
            "--socket" => a.socket = value.into(),
            "--world" => a.world = value.parse().unwrap(),
            "--rank" => a.rank = value.parse().unwrap(),
            "--link" => a.link = value,
            "--hold-copy" => a.hold = value.parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn say(line: &str) {
    println!("{line}");
    let _ = std::io::stdout().flush();
}

fn commands() -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(|l| l.ok()) {
            if tx.send(line).is_err() {
                return;
            }
        }
        let _ = tx.send("quit".into());
    });
    rx
}

fn verify(memory: &GpuMemory, world: usize, rank: usize) -> bool {
    let data = bytes_of(world);
    let (page, _) = whole(world);
    let landed = memory.pages.lock().unwrap();
    let pages = SCRATCH.iter().enumerate().all(|(i, id)| {
        let from = i * page + rank * 8192;
        landed.get(id).map(Vec::as_slice) == Some(&data[from..from + 8192])
    });
    let rows = memory.rows.lock().unwrap();
    let from = PAGES * page + rank * 4096;
    pages && rows.get(&9).map(Vec::as_slice) == Some(&data[from..from + 4096])
}

fn wait_until(mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn gpus(world: usize) -> Vec<[u8; 16]> {
    (0..world).map(|r| [50 + r as u8; 16]).collect()
}

fn config(args: &Args, name: &str, gpus: Vec<[u8; 16]>) -> PoolConfig {
    PoolConfig {
        address: PoolAddress::Unix(args.socket.clone()),
        ca: None,
        scope: scope(args.world),
        gpu_uuids: gpus,
        export_slots: 2,
        claim_slots: 2,

        registration_context: 0,
        approve: None,
    }
}

fn main() {
    let args = args();
    match args.role.as_str() {
        "source" => source(&args),
        "rank0" => rank0(&args),
        "agent" => agent(&args),
        other => panic!("unknown role {other}"),
    }
}

fn source(args: &Args) {
    let (copier, _gpu) = SimulatedGpu::shared(&ranks(args.world)).unwrap();
    let mut client = PoolClient::start(
        config(args, "source", vec![[1; 16]]),
        None,
        copier,
        Arc::new(NoRegistration),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    wait_until(|| client.status() == PoolStatus::Ready);
    let history = history();
    let key = history.key_at(PAGES as u64 * PT).unwrap();
    let (page, row) = whole(args.world);
    client.saved(IndexEntry {
        key,
        parent: None,
        bucket: history.bucket_at(key.consumed_tokens).unwrap(),
        tail_pages: PAGES as u32,
        row: true,
        tier: Tier::Dram,
        bytes: (PAGES * page + row) as u64,
    });
    say("ready");
    let commands = commands();
    loop {
        let mut events = Vec::new();
        let _ = client.poll(&mut events);
        for event in events {
            if let PoolEvent::Export(request) = event {
                client.host_pinned(request.export, pin(args.world, key));
                say("served");
            }
        }
        if let Ok(line) = commands.try_recv()
            && line == "quit"
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let _ = client.close();
}

fn rank0(args: &Args) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    say(&format!(
        "link-port {}",
        listener.local_addr().unwrap().port()
    ));
    let mut links: Vec<Option<RankLink>> = (1..args.world).map(|_| None).collect();
    while links.iter().any(Option::is_none) {
        let (mut stream, _) = listener.accept().unwrap();
        let mut rank = [0u8; 4];
        stream.read_exact(&mut rank).unwrap();
        let rank = u32::from_le_bytes(rank) as usize;
        links[rank - 1] = Some(link(stream));
    }
    let (copier, gpu) = SimulatedGpu::shared(&[local()]).unwrap();
    let mut client = PoolClient::start(
        config(args, "requester", gpus(args.world)),
        Some(links.into_iter().map(Option::unwrap).collect()),
        copier,
        Arc::new(NoRegistration),
        flare_selene::pool::Roots::System,
        flare_selene::pool::STARTUP_WAIT,
    )
    .unwrap();
    wait_until(|| client.status() == PoolStatus::Ready);
    say("ready");
    let commands = commands();
    let history = history();
    loop {
        match commands.recv_timeout(Duration::from_millis(20)) {
            Ok(line) => match line.as_str() {
                "restore" => restore(&mut client, args.world, &history),
                "verify" => say(if verify(&gpu, args.world, 0) {
                    "verify ok"
                } else {
                    "verify bad"
                }),
                "poll" => {
                    let mut events = Vec::new();
                    say(if client.poll(&mut events).is_ok() {
                        "poll ok"
                    } else {
                        "poll err"
                    });
                }
                "quit" => break,
                _ => {}
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Keep the pool serviced, and say once if it can no longer account for copies.
                let mut events = Vec::new();
                if client.poll(&mut events).is_err() {
                    say("poll err");
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
            Err(_) => break,
        }
    }
    let _ = client.close();
}

fn restore(client: &mut PoolClient, world: usize, history: &TestHistory) {
    let ticket = client.begin_lookup(history, 0);
    wait_until(|| !matches!(ticket.state(), LookupState::Pending));
    let LookupState::Ready(answer) = ticket.state() else {
        return say("fetched err no lookup answer");
    };
    if answer.candidates.is_empty() {
        return say("fetched err no candidate");
    }
    let g = ranks(world);
    let identity = Identity {
        model: [7; 32],
        layout: fingerprint(&g),
    };
    let request = Request {
        history,
        identity: &identity,
        page_tokens: PT,
        resident: 0,
        before: u64::MAX,
    };
    let Some(plan) = plan::plan(&request, &answer.plan_candidates()) else {
        return say("fetched err no plan");
    };
    let part = plan.part.clone();
    let candidate = answer
        .candidates
        .iter()
        .find(|c| c.plan.source == part.source && c.plan.key == part.end)
        .unwrap()
        .clone();
    let fetch = PoolFetch {
        id: 0,
        page_tokens: PT,
        part: FetchPart {
            candidate,
            part,
            pages: SCRATCH.iter().map(|p| PageId(*p)).collect(),
            slot: Some(StateSlotId(9)),
        },
    };
    if let Err(error) = client.fetch(fetch) {
        return say(&format!("fetched err {error}"));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        let mut events = Vec::new();
        if client.poll(&mut events).is_err() {
            return say("poll err");
        }
        for event in events {
            if let PoolEvent::Fetched { result, .. } = event {
                return match result {
                    Ok(bytes) => say(&format!("fetched ok {bytes}")),
                    Err(failure) => say(&format!("fetched err {}", failure.error)),
                };
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    say("fetched err timeout");
}

fn agent(args: &Args) {
    let mut stream = TcpStream::connect(&args.link).unwrap();
    stream.write_all(&(args.rank as u32).to_le_bytes()).unwrap();
    let (inner, gpu) = SimulatedGpu::new(&[local()]).unwrap();
    let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let copier: flare_selene::SharedCopier = if !args.hold {
        Arc::new(Mutex::new(inner))
    } else {
        Arc::new(Mutex::new(Held {
            inner,
            release: release.clone(),
            held: Vec::new(),
        }))
    };
    let mut agent = RankAgent::start(AgentConfig {
        address: PoolAddress::Unix(args.socket.clone()),
        ca: None,
        scope: scope(args.world),
        rank: args.rank,
        world: args.world,
        link: link(stream),
        copier,
        registrar: Arc::new(NoRegistration),
        registration_context: 0,
        gpu_uuid: gpus(args.world)[args.rank],
        records: None,
        approve: None,
    })
    .unwrap();
    wait_until(|| agent.status() == PoolStatus::Ready);
    say("ready");
    let commands = commands();
    loop {
        match commands.recv_timeout(Duration::from_millis(50)) {
            Ok(line) if line == "verify" => say(if verify(&gpu, args.world, args.rank) {
                "verify ok"
            } else {
                "verify bad"
            }),
            Ok(line) if line == "release" => {
                release.store(true, std::sync::atomic::Ordering::Release);
            }
            Ok(line) if line == "quit" => break,
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if matches!(agent.status(), PoolStatus::Disabled(_)) {
                    say("agent disabled");
                    break;
                }
            }
            Err(_) => break,
        }
    }
    say(if agent.close().is_ok() {
        "closed ok"
    } else {
        "closed err"
    });
}
