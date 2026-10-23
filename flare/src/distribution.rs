//! Serving bootstrap: exchange rank metadata, distribute verified sparse packages over CX7,
//! then hand the management sockets to the native collective rendezvous.

use crate::{
    cli::Run,
    cluster::Membership,
    fabric::{self, Failure, Topology},
    fabric_transport::{Endpoint, Relay},
    package_cache::{self as cache, Lease},
    trust,
};
use flare_helios::{
    runtime_package::{self, AuthorizedPackage, Publisher, VerifiedPackage},
    startup::StartupProgress,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const VERSION: u32 = 1;
const TIMEOUT: Duration = Duration::from_secs(1800);
const IO_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_FRAME: usize = 16 << 20;

#[derive(Serialize, Deserialize)]
struct Join {
    version: u32,
    rank: usize,
    endpoint: Endpoint,
}
#[derive(Clone, Serialize, Deserialize)]
struct FileSpec {
    name: String,
    size: u64,
    ranges: Vec<(u64, u64)>,
}
#[derive(Serialize, Deserialize)]
struct Plan {
    version: u32,
    rank: usize,
    export: String,
    primary: String,
    directory: String,
    files: Vec<FileSpec>,
    publisher: Publisher,
    topology: Topology,
    streams: usize,
    stage_only: bool,
}
#[derive(Serialize, Deserialize)]
struct Request {
    file: usize,
    ranges: Vec<(u64, u64)>,
}

fn write<T: Serialize>(socket: &mut TcpStream, value: &T) -> Result<(), Failure> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > MAX_FRAME {
        return Err("serving bootstrap frame exceeds 16 MiB".into());
    }
    socket.write_all(&(bytes.len() as u32).to_le_bytes())?;
    socket.write_all(&bytes)?;
    Ok(())
}
fn read<T: DeserializeOwned>(socket: &mut TcpStream) -> Result<T, Failure> {
    let mut length = [0; 4];
    socket.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > MAX_FRAME {
        return Err("serving bootstrap frame exceeds 16 MiB".into());
    }
    let mut bytes = vec![0; length];
    socket.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}
fn configure(socket: &TcpStream, timeout: Duration) -> Result<(), Failure> {
    socket.set_nodelay(true)?;
    socket.set_read_timeout(Some(timeout))?;
    socket.set_write_timeout(Some(timeout))?;
    Ok(())
}
fn membership(socket: TcpStream) -> Result<flare_helios::execution::RankStream, Failure> {
    socket.set_read_timeout(None)?;
    socket.set_write_timeout(None)?;
    Ok(socket.into())
}
fn endpoint() -> Result<(TcpListener, Endpoint), Failure> {
    let listener = TcpListener::bind("0.0.0.0:0")?;
    let endpoint = Endpoint {
        record: fabric::discover()?,
        port: listener.local_addr()?.port(),
    };
    Ok((listener, endpoint))
}

/// Held through serving so agent eviction cannot remove an active rank's package.
pub struct Received {
    pub authorized: AuthorizedPackage,
    pub membership: Membership,
    pub lease: Option<Lease>,
}

/// Rank zero already owns the trusted complete package. Peers receive only their required ranges.
pub fn root(
    package: &AuthorizedPackage,
    publisher: &Publisher,
    primary: &Path,
    args: &Run,
    progress: &StartupProgress,
) -> Result<Membership, Failure> {
    let (world, export) = runtime_package::package_cluster(package)
        .ok_or("distribution requires a multi-host package")?;
    if !(2..=16).contains(&world) {
        return Err("distribution requires 2..16 ranks".into());
    }
    let address = args
        .rendezvous
        .as_deref()
        .ok_or("multi-host serving requires --rendezvous HOST:PORT")?;
    let listener = TcpListener::bind(address)?;
    listener.set_nonblocking(true)?;
    let (network, local) = endpoint()?;
    let primary = primary
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("package filename must be UTF-8")?
        .to_owned();
    let streams = args.transfer_streams.unwrap_or(8) as usize;
    let mut files = Vec::new();
    let mut specifications = Vec::new();
    for rank in 0..world {
        let ranges = runtime_package::rank_ranges(package, rank).map_err(|e| e.to_string())?;
        let mut specs = Vec::new();
        let mut handles = Vec::new();
        for file in ranges {
            specs.push(FileSpec {
                name: if file.filename.is_empty() {
                    primary.clone()
                } else {
                    file.filename
                },
                size: file.file.metadata()?.len(),
                ranges: file.ranges,
            });
            handles.push(file.file);
        }
        specifications.push(specs);
        files.push(handles);
    }
    let source = Source::start(files, specifications.clone(), streams)?;
    let phase = progress.phase("Wait for tensor-parallel hosts", None);
    let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
    let mut endpoints = vec![local; world];
    let deadline = Instant::now() + TIMEOUT;
    let mut joined = 1;
    while joined < world {
        if crate::STOP.load(Ordering::Acquire) {
            return Err("serving launch cancelled".into());
        }
        if Instant::now() >= deadline {
            return Err("timed out waiting for serving ranks".into());
        }
        match listener.accept() {
            Ok((mut socket, _)) => {
                configure(&socket, IO_TIMEOUT)?;
                let join: Join = read(&mut socket)?;
                if join.version != VERSION
                    || join.rank == 0
                    || join.rank >= world
                    || peers[join.rank].is_some()
                {
                    return Err("serving rank has an incompatible protocol, duplicate rank, or invalid rank".into());
                }
                endpoints[join.rank] = join.endpoint;
                configure(&socket, TIMEOUT)?;
                peers[join.rank] = Some(socket);
                joined += 1;
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20))
            }
            Err(e) => return Err(e.into()),
        }
    }
    phase.finish(format!("{} peer ranks", world - 1));
    let topology = Topology::plan_shape(
        &endpoints
            .iter()
            .map(|e| e.record.clone())
            .collect::<Vec<_>>(),
        &endpoints.iter().map(|e| e.port).collect::<Vec<_>>(),
        args.roce_ring,
        args.roce_chain,
    )?;
    let relay = Relay::start(0, streams, topology.clone(), network, source.port)?;
    for rank in 1..world {
        write(
            peers[rank].as_mut().unwrap(),
            &Plan {
                version: VERSION,
                rank,
                export: export.clone(),
                primary: primary.clone(),
                directory: format!("{export}-{}-rank{rank}", specifications[rank][0].size),
                files: specifications[rank].clone(),
                publisher: publisher.clone(),
                topology: topology.clone(),
                streams,
                stage_only: args.stage_only,
            },
        )?;
    }
    // Every listener exists before any receiver connects; all relays stay alive through this barrier.
    for socket in peers.iter_mut().skip(1).flatten() {
        expect(socket, "ready")?;
    }
    for socket in peers.iter_mut().skip(1).flatten() {
        write(socket, &"transfer")?;
    }
    let phase = progress.phase("Distribute rank packages over CX7", None);
    let cancellation = Channels::default();
    for socket in peers.iter().skip(1).flatten() {
        cancellation.add(socket)?;
    }
    thread::scope(|scope| -> Result<(), Failure> {
        let workers = peers
            .iter_mut()
            .skip(1)
            .flatten()
            .map(|s| {
                let cancellation = &cancellation;
                scope.spawn(move || {
                    let result = expect(s, "verified");
                    if result.is_err() {
                        cancellation.stop();
                    }
                    result
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().map_err(|_| "rank receiver panicked")??;
        }
        Ok(())
    })?;
    // Receivers closed their data sockets before acknowledging verification.
    source.finish()?;
    relay.finish()?;
    for socket in peers.iter_mut().skip(1).flatten() {
        write(socket, &"serve")?;
    }
    phase.finish("all rank packages verified");
    cancellation.state.lock().unwrap().1.clear();
    Ok(Membership::Zero(
        peers
            .into_iter()
            .skip(1)
            .flatten()
            .map(membership)
            .collect::<Result<_, _>>()?,
    ))
}
fn expect(socket: &mut TcpStream, expected: &str) -> Result<(), Failure> {
    let message: String = read(socket)?;
    if message != expected {
        return Err(format!("serving bootstrap expected {expected}, received {message}").into());
    }
    Ok(())
}

/// A manually launched peer needs only its rank, rendezvous address, and publisher acceptance.
pub fn peer(args: &mut Run, progress: &mut StartupProgress) -> Result<Received, Failure> {
    let rank = args.rank.ok_or("a peer requires --rank")?;
    if rank == 0 {
        return Err("rank zero requires a package".into());
    }
    let address = args
        .rendezvous
        .as_deref()
        .ok_or("a peer requires --rendezvous HOST:PORT")?;
    let addresses = address.to_socket_addrs()?.collect::<Vec<_>>();
    let (network, endpoint) = endpoint()?;
    let deadline = Instant::now() + TIMEOUT;
    let mut control = loop {
        let socket = addresses
            .iter()
            .find_map(|a| TcpStream::connect_timeout(a, Duration::from_secs(2)).ok());
        if let Some(socket) = socket {
            break socket;
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out joining rank zero at {address}").into());
        }
        thread::sleep(Duration::from_millis(100));
    };
    configure(&control, TIMEOUT)?;
    write(
        &mut control,
        &Join {
            version: VERSION,
            rank,
            endpoint,
        },
    )?;
    let plan: Plan = read(&mut control)?;
    plan.topology.validate()?;
    plan.publisher.validate().map_err(|e| e.to_string())?;
    if plan.version != VERSION
        || plan.rank != rank
        || rank >= plan.topology.links.len()
        || !(1..=64).contains(&plan.streams)
        || plan.files.is_empty()
        || plan.primary != plan.files[0].name
    {
        return Err("invalid serving distribution plan".into());
    }
    if args.roce_ring && !plan.topology.ring {
        return Err("rank zero did not select --roce-ring".into());
    }
    if args.roce_chain && !plan.topology.chain {
        return Err("rank zero did not select --roce-chain".into());
    }
    if args.stage_only && !plan.stage_only {
        return Err("rank zero did not select --stage-only".into());
    }
    args.roce_ring = plan.topology.ring;
    args.roce_chain = plan.topology.chain;
    args.stage_only = plan.stage_only;
    let relay = Relay::start(rank, plan.streams, plan.topology.clone(), network, 0)?;
    write(&mut control, &"ready")?;
    expect(&mut control, "transfer")?;
    if let Some(path) = args.package.as_ref().filter(|path| path.is_file()) {
        let cache_root = args
            .package_cache
            .clone()
            .unwrap_or_else(cache::default_root)
            .join("packages");
        let lease = if path
            .canonicalize()?
            .starts_with(cache_root.canonicalize().unwrap_or(cache_root))
        {
            Some(Lease::shared(
                path.parent().ok_or("package path lacks a parent")?,
            )?)
        } else {
            None
        };
        let (authorized, _) = trust::load(path, args, progress, None).map_err(|e| e.to_string())?;
        if runtime_package::package_cluster(&authorized)
            != Some((plan.topology.links.len(), plan.export.clone()))
        {
            return Err("local package differs from rank zero's export".into());
        }
        write(&mut control, &"verified")?;
        expect(&mut control, "serve")?;
        relay.finish()?;
        return Ok(Received {
            authorized,
            membership: Membership::Peer(membership(control)?),
            lease,
        });
    }
    let cache = args
        .package_cache
        .clone()
        .unwrap_or_else(cache::default_root);
    for file in &plan.files {
        cache::entry(
            &cache,
            "packages",
            &format!("{}/{}", plan.directory, file.name),
        )?;
        cache::validate_ranges(file.size, &file.ranges)?;
    }
    let path = cache::entry(
        &cache,
        "packages",
        &format!("{}/{}", plan.directory, plan.primary),
    )?;
    let lease = Lease::shared(path.parent().unwrap())?;
    let marker = path.parent().unwrap().join(".verified");
    let marker_text = format!("{} {}\n", plan.primary, plan.files[0].size);
    let complete = || -> Result<bool, Failure> {
        if fs::read_to_string(&marker).ok().as_deref() != Some(&marker_text) {
            return Ok(false);
        }
        for file in &plan.files {
            if cache::have(
                &cache,
                "package",
                &format!("{}/{}", plan.directory, file.name),
                file.size,
            )?
            .is_none()
            {
                return Ok(false);
            }
        }
        Ok(true)
    };
    if !complete()? {
        lease.exclusive()?;
        if !complete()? {
            let phase = progress.phase("Receive rank package over CX7", None);
            let result = receive_files(&cache, &plan, &relay).and_then(|()| {
                VerifiedPackage::open(&path, std::slice::from_ref(&plan.publisher), Some(rank))
                    .and_then(|p| p.verify_stored_data())
                    .map_err(|e| e.to_string())?;
                fs::write(&marker, &marker_text)?;
                Ok(())
            });
            if let Err(e) = result {
                // A rejected sparse package must not look complete on a subsequent attempt.
                for file in &plan.files {
                    let _ = fs::remove_file(path.with_file_name(&file.name));
                }
                let _ = fs::remove_file(&marker);
                return Err(e);
            }
            phase.finish(format!("rank {rank} verified"));
        }
        lease.downgrade()?;
    }
    let profiles = cache.join("publishers");
    fs::create_dir_all(&profiles)?;
    let profile = profiles.join(format!("{}.json", plan.publisher.fingerprint()));
    fs::write(&profile, serde_json::to_vec(&vec![&plan.publisher])?)?;
    args.serving.publisher_keys.push(profile);
    let (authorized, _) = trust::load(&path, args, progress, None).map_err(|e| e.to_string())?;
    if runtime_package::package_cluster(&authorized)
        != Some((plan.topology.links.len(), plan.export))
    {
        return Err("received package differs from rank zero's export".into());
    }
    if let Some(expected) = &args.package {
        if expected.file_name() != path.file_name() {
            return Err("received package differs from the requested filename".into());
        }
    }
    write(&mut control, &"verified")?;
    expect(&mut control, "serve")?;
    relay.finish()?;
    Ok(Received {
        authorized,
        membership: Membership::Peer(membership(control)?),
        lease: Some(lease),
    })
}

fn receive_files(cache: &Path, plan: &Plan, relay: &Relay) -> Result<(), Failure> {
    if plan
        .files
        .iter()
        .map(|file| {
            cache::have(
                cache,
                "package",
                &format!("{}/{}", plan.directory, file.name),
                file.size,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .all(Option::is_some)
    {
        return Ok(());
    }
    let channels = Channels::default();
    let sockets = (0..plan.streams)
        .map(|i| {
            let socket = relay.connect(i)?;
            configure(&socket, IO_TIMEOUT)?;
            channels.add(&socket)?;
            Ok(Mutex::new(socket))
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    let result = (|| -> Result<(), Failure> {
        for (index, file) in plan.files.iter().enumerate() {
            let name = format!("{}/{}", plan.directory, file.name);
            if cache::have(&cache, "package", &name, file.size)?.is_some() {
                continue;
            }
            cache::store_parallel(
                &cache,
                &name,
                file.size,
                &file.ranges,
                plan.streams,
                |stream, ranges| {
                    let result = (|| -> Result<Payload<'_>, Failure> {
                        let mut socket = sockets[stream].lock().unwrap();
                        write(
                            &mut socket,
                            &Request {
                                file: index,
                                ranges: ranges.to_vec(),
                            },
                        )?;
                        let mut bytes = [0; 8];
                        socket.read_exact(&mut bytes)?;
                        let remaining = u64::from_le_bytes(bytes);
                        if remaining != cache::range_bytes(ranges) {
                            return Err("package range length differs".into());
                        }
                        Ok(Payload {
                            socket: socket.try_clone()?,
                            remaining,
                            channels: &channels,
                        })
                    })();
                    if result.is_err() {
                        channels.stop();
                    }
                    result
                },
            )?;
        }
        Ok(())
    })();
    channels.stop();
    result
}

#[derive(Default)]
struct Channels {
    state: Mutex<(bool, Vec<TcpStream>)>,
}
impl Channels {
    fn add(&self, socket: &TcpStream) -> Result<(), Failure> {
        let mut state = self.state.lock().unwrap();
        if state.0 {
            let _ = socket.shutdown(Shutdown::Both);
            return Err("package transfer stopped".into());
        }
        state.1.push(socket.try_clone()?);
        Ok(())
    }
    fn stop(&self) {
        let mut state = self.state.lock().unwrap();
        state.0 = true;
        for socket in &state.1 {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
}
impl Drop for Channels {
    fn drop(&mut self) {
        self.stop();
    }
}
struct Payload<'a> {
    socket: TcpStream,
    remaining: u64,
    channels: &'a Channels,
}
impl Read for Payload<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let want = self.remaining.min(bytes.len() as u64) as usize;
        match self.socket.read(&mut bytes[..want]) {
            Ok(0) => {
                self.channels.stop();
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "package channel ended inside a range",
                ))
            }
            Ok(n) => {
                self.remaining -= n as u64;
                Ok(n)
            }
            Err(e) => {
                self.channels.stop();
                Err(e)
            }
        }
    }
}
impl Drop for Payload<'_> {
    fn drop(&mut self) {
        if self.remaining != 0 {
            self.channels.stop();
        }
    }
}

struct Source {
    port: u16,
    stopping: Arc<AtomicBool>,
    channels: Arc<Channels>,
    worker: Option<thread::JoinHandle<Result<(), Failure>>>,
}
impl Source {
    fn start(
        files: Vec<Vec<Arc<File>>>,
        specifications: Vec<Vec<FileSpec>>,
        streams: usize,
    ) -> Result<Self, Failure> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        listener.set_nonblocking(true)?;
        let stopping = Arc::new(AtomicBool::new(false));
        let channels = Arc::new(Channels::default());
        let (stop, held) = (stopping.clone(), channels.clone());
        let worker = thread::spawn(move || {
            let slots = Mutex::new(std::collections::HashSet::new());
            thread::scope(|scope| -> Result<(), Failure> {
                let mut workers = Vec::new();
                let (sender, receiver) = std::sync::mpsc::channel();
                let mut error = None;
                while !stop.load(Ordering::Acquire) {
                    if let Ok(Err(e)) = receiver.try_recv() {
                        error = Some(e);
                        break;
                    }
                    match listener.accept() {
                        Ok((socket, _)) => {
                            if workers.len() >= (files.len() - 1) * streams {
                                error = Some("package source channel capacity exceeded".into());
                                break;
                            }
                            held.add(&socket)?;
                            let (files, specs, slots, sender) =
                                (&files, &specifications, &slots, sender.clone());
                            workers.push(scope.spawn(move || {
                                let _ =
                                    sender.send(serve_source(socket, files, specs, streams, slots));
                            }));
                        }
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(10))
                        }
                        Err(e) => {
                            error = Some(e.into());
                            break;
                        }
                    }
                }
                held.stop();
                for worker in workers {
                    worker.join().map_err(|_| "package source panicked")?;
                }
                for result in receiver.try_iter() {
                    if let Err(e) = result {
                        error.get_or_insert(e);
                    }
                }
                error.map_or(Ok(()), Err)
            })
        });
        Ok(Self {
            port,
            stopping,
            channels,
            worker: Some(worker),
        })
    }
    fn finish(mut self) -> Result<(), Failure> {
        self.stopping.store(true, Ordering::Release);
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| "package source panicked")?
    }
}
impl Drop for Source {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.channels.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn serve_source(
    mut socket: TcpStream,
    files: &[Vec<Arc<File>>],
    specs: &[Vec<FileSpec>],
    streams: usize,
    slots: &Mutex<std::collections::HashSet<(usize, usize)>>,
) -> Result<(), Failure> {
    configure(&socket, IO_TIMEOUT)?;
    let mut bytes = [0; 8];
    socket.read_exact(&mut bytes)?;
    let rank = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let stream = u32::from_le_bytes(bytes[4..].try_into().unwrap()) as usize;
    if rank == 0
        || rank >= files.len()
        || stream >= streams
        || !slots.lock().unwrap().insert((rank, stream))
    {
        return Err("invalid or duplicate package source channel".into());
    }
    loop {
        // EOF between requests is normal: a receiver closes its channels before the final barrier.
        if socket.peek(&mut [0])? == 0 {
            return Ok(());
        }
        let request: Request = read(&mut socket)?;
        let spec = specs[rank]
            .get(request.file)
            .ok_or("package file index is out of bounds")?;
        cache::validate_ranges(spec.size, &request.ranges)?;
        if request.ranges.iter().any(|&(offset, size)| {
            !spec
                .ranges
                .iter()
                .any(|&(start, count)| offset >= start && offset + size <= start + count)
        }) {
            return Err("requested bytes are outside the rank's package ranges".into());
        }
        socket.write_all(&cache::range_bytes(&request.ranges).to_le_bytes())?;
        io::copy(
            &mut cache::RangeReader::new(&files[rank][request.file], &request.ranges),
            &mut socket,
        )?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_rank_distribution_reuses_channels_across_files_and_cached_launches() {
        for (world, ring) in [(2, false), (4, false), (4, true)] {
            let temporary = tempfile::tempdir().unwrap();
            let bytes = (0..3)
                .map(|file| {
                    (0..(2 << 20))
                        .map(|i| ((i + file * 23) % 251) as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut files = Vec::new();
            for (index, bytes) in bytes.iter().enumerate() {
                let path = temporary.path().join(format!("source{index}"));
                fs::write(&path, bytes).unwrap();
                files.push(Arc::new(File::open(path).unwrap()));
            }
            let specs = (0..world)
                .map(|rank| {
                    (0..3)
                        .map(|file| FileSpec {
                            name: format!("file{file}.safetensors"),
                            size: bytes[file].len() as u64,
                            ranges: vec![(0, 255), (4096 + rank as u64 * 1024, 1024 * 1024)],
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let source = Source::start(vec![files; world], specs.clone(), 8).unwrap();
            let network = (0..world)
                .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
                .collect::<Vec<_>>();
            let ports = network
                .iter()
                .map(|s| s.local_addr().unwrap().port())
                .collect::<Vec<_>>();
            let topology =
                Topology::plan(&vec![b"a 127.0.0.1/8\n".to_vec(); world], &ports, ring).unwrap();
            let relays = network
                .into_iter()
                .enumerate()
                .map(|(rank, socket)| {
                    Relay::start(
                        rank,
                        8,
                        topology.clone(),
                        socket,
                        if rank == 0 { source.port } else { 0 },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            thread::scope(|scope| {
                for rank in 1..world {
                    let (relays, specs, topology, temporary, bytes) =
                        (&relays, &specs, &topology, &temporary, &bytes);
                    scope.spawn(move || {
                        let plan = Plan {
                            version: VERSION,
                            rank,
                            export: "export".into(),
                            primary: specs[rank][0].name.clone(),
                            directory: format!("export-rank{rank}"),
                            files: specs[rank].clone(),
                            publisher: Publisher {
                                name: "Fixture".into(),
                                namespace: "local-development".into(),
                                development: true,
                                public_key: [0; 32],
                            },
                            topology: topology.clone(),
                            streams: 8,
                            stage_only: true,
                        };
                        receive_files(temporary.path(), &plan, &relays[rank]).unwrap();
                        receive_files(temporary.path(), &plan, &relays[rank]).unwrap();
                        for (index, file) in plan.files.iter().enumerate() {
                            let stored = fs::read(
                                temporary
                                    .path()
                                    .join("packages")
                                    .join(&plan.directory)
                                    .join(&file.name),
                            )
                            .unwrap();
                            for &(offset, length) in &file.ranges {
                                let range = offset as usize..(offset + length) as usize;
                                assert_eq!(stored[range.clone()], bytes[index][range]);
                            }
                            assert!(stored[255..4096].iter().all(|&b| b == 0));
                        }
                    });
                }
            });
            source.finish().unwrap();
            for relay in relays {
                relay.finish().unwrap();
            }
        }
    }

    #[test]
    fn source_rejects_ranges_outside_the_destination_rank() {
        let temporary = tempfile::tempfile().unwrap();
        temporary.set_len(1024).unwrap();
        let files = vec![vec![Arc::new(temporary)]; 2];
        let specs = vec![
            vec![FileSpec {
                name: "file".into(),
                size: 1024,
                ranges: vec![(0, 64)]
            }];
            2
        ];
        let source = Source::start(files, specs, 1).unwrap();
        let mut socket = TcpStream::connect(("127.0.0.1", source.port)).unwrap();
        configure(&socket, Duration::from_secs(2)).unwrap();
        socket.write_all(&[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        write(
            &mut socket,
            &Request {
                file: 0,
                ranges: vec![(64, 1)],
            },
        )
        .unwrap();
        assert!(socket.read_exact(&mut [0; 8]).is_err());
        assert!(source.finish().is_err());
    }
}
