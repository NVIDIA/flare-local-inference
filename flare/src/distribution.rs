//! Serving bootstrap: exchange rank metadata, distribute verified sparse packages over CX7,
//! then hand the management sockets to the native collective rendezvous.

use crate::{
    cli::Run,
    cluster::Membership,
    fabric::{self, Failure, Topology},
    fabric_transport::{Endpoint, Relay},
    package_cache as cache, trust,
};
use flare_helios::package::cache::{self as model_cache, Entry, Root};
use flare_helios::{
    runtime_package::{
        self, AuthorizedPackage, NativeResources, PackageFormat, Publisher, VerifiedPackage,
    },
    startup::{Phase, StartupProgress},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const VERSION: u32 = 3;
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
#[derive(Clone, Serialize, Deserialize)]
struct Plan {
    version: u32,
    rank: usize,
    export: String,
    primary: String,
    directory: String,
    files: Vec<FileSpec>,
    /// Verified identity of rank zero's selected deployment (model/deployment, not a hash).
    #[serde(default)]
    model: String,
    /// Physical device platform name from hardware detection (gb202/gb10).
    #[serde(default)]
    platform: String,
    #[serde(default)]
    tensor_parallel: u32,
    /// SHA-256 of rank zero's exact authenticated signed envelope.
    expected_envelope: [u8; 32],
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
    pub lease: Option<Entry>,
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
    let signed = package.identity().map_err(|e| e.to_string())?;
    let expected_envelope = package.envelope_digest();
    let platform = if flare_helios::package::naming::PLATFORMS.contains(&signed.hardware.as_str()) {
        signed.hardware.clone()
    } else {
        String::new()
    };
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
                model: signed.model.clone(),
                platform: platform.clone(),
                tensor_parallel: signed.tensor_parallel,
                expected_envelope,
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
fn model_cache_root(args: &Run) -> Option<&std::path::Path> {
    args.hub
        .model_cache
        .as_deref()
        .or(args.package_cache.as_deref())
}

/// Fill the peer's selection from the received distribution plan when the user omitted it.
/// An explicit conflicting selection is rejected rather than overwritten.
fn select_plan_tensor_parallel(args: &mut Run, tensor_parallel: u32) -> Result<(), Failure> {
    if tensor_parallel == 0 {
        return Err("serving distribution plan has no tensor-parallel size".into());
    }
    match args.serving.hub.tensor_parallel.as_slice() {
        [] => {
            args.serving.hub.tensor_parallel.push(tensor_parallel);
            Ok(())
        }
        [selected] if *selected == tensor_parallel => Ok(()),
        _ => Err(format!(
            "requested tensor-parallel size {:?} differs from rank zero's plan {tensor_parallel}",
            args.serving.hub.tensor_parallel
        )
        .into()),
    }
}

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
    select_plan_tensor_parallel(args, plan.tensor_parallel)?;
    let relay = Relay::start(rank, plan.streams, plan.topology.clone(), network, 0)?;
    write(&mut control, &"ready")?;
    expect(&mut control, "transfer")?;
    if let Some(path) = args.package.as_ref().filter(|path| path.is_file()) {
        let cache_root = cache::model_root(model_cache_root(args))?;
        // A path inside the managed model cache is leased before it is read, so collection and
        // eviction cannot reclaim the entry while this rank serves from it. Caller-owned packages
        // outside the cache remain their own responsibility.
        let lease = Root::open(&cache_root)
            .map_err(|e| e.to_string())?
            .entry_for_path(path)
            .map_err(|e| e.to_string())?;
        let (authorized, _) = trust::load(path, args, progress, None).map_err(|e| e.to_string())?;
        verify_local_package(&authorized, &plan)?;
        write(&mut control, &"verified")?;
        expect(&mut control, "serve")?;
        relay.finish()?;
        return Ok(Received {
            authorized,
            membership: Membership::Peer(membership(control)?),
            lease,
        });
    }
    let model_root = cache::model_root(model_cache_root(args))?;
    let entry = receive_rank(&model_root, &plan, &relay, progress, rank)?;
    let path = entry.directory().join("model.safetensors");
    let profiles = model_root.join("publishers");
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
        lease: Some(entry),
    })
}

/// Compare a locally supplied authorized package with rank zero's plan.
fn verify_local_package(package: &AuthorizedPackage, plan: &Plan) -> Result<(), Failure> {
    if package.envelope_digest() != plan.expected_envelope {
        return Err("local package signed envelope differs from rank zero's package".into());
    }
    let identity = package.identity().map_err(|error| error.to_string())?;
    let cluster_ok = identity.cluster.as_ref().is_some_and(|cluster| {
        cluster.world as usize == plan.topology.links.len() && cluster.export == plan.export
    });
    if identity.model != plan.model
        || identity.tensor_parallel != plan.tensor_parallel
        || !cluster_ok
    {
        return Err("local package differs from the planned verified identity".into());
    }
    Ok(())
}

/// Verify one received rank against the plan using signed metadata only, then hash its stored
/// bytes. Returns the verified package for the caller to retain or discard.
fn verify_plan_metadata(
    path: &Path,
    plan: &Plan,
    rank: usize,
) -> Result<runtime_package::VerifiedPackage, Failure> {
    // Open exactly the deployment the sender selected. Payload packages can hold several
    // deployments, so they are opened by topology; legacy packages are one prepared size and use
    // the compatibility constructor. Both keep their normal signature/rank verification.
    let rank_id = u32::try_from(rank).map_err(|_| "rank id exceeds the package rank domain")?;
    let package = match NativeResources::package_format(path).map_err(|e| e.to_string())? {
        PackageFormat::Payload => runtime_package::VerifiedPackage::open_tensor_parallel(
            path,
            std::slice::from_ref(&plan.publisher),
            plan.tensor_parallel,
            Some(rank_id),
        ),
        PackageFormat::Legacy => runtime_package::VerifiedPackage::open(
            path,
            std::slice::from_ref(&plan.publisher),
            Some(rank),
        ),
    }
    .map_err(|e| e.to_string())?;
    if package.envelope_digest() != plan.expected_envelope {
        return Err("received rank signed envelope differs from rank zero's package".into());
    }
    let identity = package.identity().map_err(|e| e.to_string())?;
    // Distribution carries one rank of a multi-host package. The signed cluster identity must
    // match the planned world and export exactly; an absent cluster identity is rejected.
    let cluster_ok = identity.cluster.as_ref().is_some_and(|cluster| {
        cluster.world as usize == plan.topology.links.len() && cluster.export == plan.export
    });
    if identity.model != plan.model
        || identity.tensor_parallel != plan.tensor_parallel
        || !cluster_ok
    {
        return Err("received rank differs from the planned verified identity".into());
    }
    Ok(package)
}

/// Re-validate a freshly received rank, including its stored payload bytes.
fn verify_plan(
    path: &Path,
    plan: &Plan,
    rank: usize,
    phase: Option<&Phase>,
) -> Result<runtime_package::VerifiedPackage, Failure> {
    let package = verify_plan_metadata(path, plan, rank)?;
    package
        .verify_stored_data_with_progress(phase)
        .map_err(|e| e.to_string())?;
    Ok(package)
}

/// Bytes a full verification reads for this rank, for progress reporting.
fn plan_bytes(plan: &Plan) -> u64 {
    plan.files
        .iter()
        .map(|file| file.size)
        .fold(0u64, u64::saturating_add)
}

const VERIFIED_MARKER: &str = ".flare-verified.json";

/// What a completed full verification covered, so a warm launch can check that the immutable
/// entry has not deviated instead of hashing every payload again.
#[derive(Serialize, Deserialize)]
struct VerifiedMarker {
    version: u32,
    envelope: String,
    export: String,
    rank: usize,
    primary_sha256: String,
    files: Vec<VerifiedFile>,
}

#[derive(Serialize, Deserialize)]
struct VerifiedFile {
    name: String,
    size: u64,
    modified_ns: u128,
    changed_ns: u128,
    inode: u64,
    device: u64,
}

fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn modified_ns(metadata: &fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos())
}

/// Status-change time. Unlike mtime it cannot be reset by ordinary tools, so a rewritten file
/// that copied its timestamps still fails the marker check.
#[cfg(unix)]
fn changed_ns(metadata: &fs::Metadata) -> u128 {
    use std::os::unix::fs::MetadataExt;
    let seconds = metadata.ctime();
    let nanos = metadata.ctime_nsec();
    if seconds < 0 {
        return 0;
    }
    seconds as u128 * 1_000_000_000 + nanos.max(0) as u128
}

#[cfg(not(unix))]
fn changed_ns(_: &fs::Metadata) -> u128 {
    0
}

#[cfg(unix)]
fn inode_device(metadata: &fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (metadata.ino(), metadata.dev())
}

#[cfg(not(unix))]
fn inode_device(_: &fs::Metadata) -> (u64, u64) {
    (0, 0)
}

/// Hash one file in bounded chunks; the primary of a monolithic package can be tens of GiB.
fn sha256_file(path: &Path) -> Result<[u8; 32], Failure> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buffer = vec![0u8; 4 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    Ok(context
        .finish()
        .as_ref()
        .try_into()
        .expect("SHA-256 digest is 32 bytes"))
}

/// The entry root names the primary `model.safetensors`; every other file keeps its signed name.
fn entry_file_name(plan: &Plan, file: &FileSpec) -> String {
    if file.name == plan.primary {
        "model.safetensors".to_owned()
    } else {
        file.name.clone()
    }
}

/// Record the state a successful full verification covered. The marker is written beside the
/// immutable entry and binds the signed envelope, the primary digest, and every file's identity.
fn write_verified_marker(root: &Path, plan: &Plan, rank: usize) -> Result<(), Failure> {
    let primary_sha256 = hex_bytes(&sha256_file(&root.join("model.safetensors"))?);
    let mut files = Vec::with_capacity(plan.files.len());
    for file in &plan.files {
        let name = entry_file_name(plan, file);
        let metadata = fs::metadata(root.join(&name)).map_err(|e| e.to_string())?;
        let (inode, device) = inode_device(&metadata);
        files.push(VerifiedFile {
            name,
            size: metadata.len(),
            modified_ns: modified_ns(&metadata),
            changed_ns: changed_ns(&metadata),
            inode,
            device,
        });
    }
    let marker = VerifiedMarker {
        version: 2,
        envelope: hex_bytes(&plan.expected_envelope),
        export: plan.export.clone(),
        rank,
        primary_sha256,
        files,
    };
    let temporary = root.join(format!("{VERIFIED_MARKER}.tmp"));
    fs::write(
        &temporary,
        serde_json::to_vec(&marker).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let marker_path = root.join(VERIFIED_MARKER);
    fs::rename(&temporary, &marker_path).map_err(|e| e.to_string())?;
    // Published entries are sealed read-only; the marker matches that immutability. Replacing
    // it later only needs write permission on the directory, which stays writable.
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(&marker_path)
            .map_err(|e| e.to_string())?
            .permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&marker_path, permissions).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// True when the entry still matches the state its last full verification covered.
fn verified_marker_matches(root: &Path, plan: &Plan, rank: usize) -> bool {
    let Ok(bytes) = fs::read(root.join(VERIFIED_MARKER)) else {
        return false;
    };
    let Ok(marker) = serde_json::from_slice::<VerifiedMarker>(&bytes) else {
        return false;
    };
    if marker.version != 2
        || marker.envelope != hex_bytes(&plan.expected_envelope)
        || marker.export != plan.export
        || marker.rank != rank
        || marker.files.len() != plan.files.len()
    {
        return false;
    }
    // Cheap identity checks first: a file that was replaced, truncated, or rewritten fails
    // without reading its contents.
    for (file, expected) in plan.files.iter().zip(&marker.files) {
        if entry_file_name(plan, file) != expected.name {
            return false;
        }
        let Ok(metadata) = fs::metadata(root.join(&expected.name)) else {
            return false;
        };
        let (inode, device) = inode_device(&metadata);
        if metadata.len() != expected.size
            || modified_ns(&metadata) != expected.modified_ns
            || changed_ns(&metadata) != expected.changed_ns
            || inode != expected.inode
            || device != expected.device
        {
            return false;
        }
    }
    // The entry primary anchors the package: hash it even when metadata is unchanged.
    match sha256_file(&root.join("model.safetensors")) {
        Ok(primary) => marker.primary_sha256 == hex_bytes(&primary),
        Err(_) => false,
    }
}

fn receiver_candidate(platform: &str, tensor_parallel: u32) -> Result<String, Failure> {
    use flare_helios::package::naming;
    let now = std::time::SystemTime::now();
    if naming::PLATFORMS.contains(&platform) {
        return naming::candidate(platform, tensor_parallel, now).ok_or(
            "receiver naming requires a supported platform and tensor-parallel size".into(),
        );
    }
    naming::unknown_device_candidate(&[tensor_parallel], now)
        .ok_or("receiver naming requires a tensor-parallel size".into())
}

/// Receive, verify, and publish one sparse rank package inside the model cache transaction.
///
/// Warm reuse resolves `<export>-rank<rank>` and re-validates the selected rank before use.
/// A miss stages an owned tree, receives the sparse ranges, flattens them to the canonical entry
/// root, verifies the selected rank, commits the entry, records the source reference, and selects
/// the entry only after publication. The returned `Entry` lease must be retained through serving.
pub(crate) fn receive_rank(
    model_root: &Path,
    plan: &Plan,
    relay: &Relay,
    progress: &StartupProgress,
    rank: usize,
) -> Result<Entry, Failure> {
    let root = Root::open(model_root).map_err(|e| e.to_string())?;
    // Warm reuse keys on the distributed source identity and this rank, never on a claim of full
    // package readiness: a sparse rank slice is not a complete package.
    let selector = format!("{}-rank{rank}", plan.export);
    let (entry, path) = match root
        .resolve_source_ref(&selector)
        .map_err(|e| e.to_string())?
    {
        Some((_revision, entry)) => {
            let path = entry.directory().join("model.safetensors");
            let phase = progress.phase("Verify cached rank package", Some(plan_bytes(plan)));
            let root = entry.directory();
            // The signed envelope is re-authenticated on every launch; it is cheap metadata.
            // A verified marker records the exact state a full payload verification covered, so
            // only an absent or stale marker hashes every payload again.
            let cached = verified_marker_matches(root, plan, rank);
            let valid = path.is_file()
                && match verify_plan_metadata(&path, plan, rank) {
                    Ok(package) => {
                        if cached {
                            true
                        } else {
                            let verified = package
                                .verify_stored_data_with_progress(Some(&phase))
                                .is_ok();
                            // A read-only or full cache still serves a verified rank; the
                            // marker only decides whether the next launch skips hashing.
                            if verified {
                                let _ = write_verified_marker(root, plan, rank);
                            }
                            verified
                        }
                    }
                    Err(_) => false,
                };
            phase.finish(if valid {
                if cached {
                    format!("rank {rank} cached")
                } else {
                    format!("rank {rank}")
                }
            } else {
                "mismatch".to_owned()
            });
            if valid {
                (Some(entry), path)
            } else {
                (None, PathBuf::new())
            }
        }
        None => (None, PathBuf::new()),
    };
    if let Some(entry) = entry {
        // Warm reuse is a successful acquisition: it moves the selected default like a fresh
        // download, under the same atomic last-writer-wins ref semantics.
        root.publish_entry(&entry).map_err(|e| e.to_string())?;
        return Ok(entry);
    }
    {
        for file in &plan.files {
            cache::validate_ranges(file.size, &file.ranges)?;
        }
        if plan.model.is_empty() || plan.tensor_parallel == 0 {
            return Err("rank plan lacks verified model/deployment identity".into());
        }
        let model = model_cache::model_key(&plan.model).map_err(|e| e.to_string())?;
        let candidate = receiver_candidate(&plan.platform, plan.tensor_parallel)?;
        let staging = root
            .stage_named(&model, &candidate)
            .map_err(|e| e.to_string())?;
        {
            let phase = progress.phase("Receive rank package over CX7", None);
            let staging_dir = staging.directory().to_path_buf();
            let received = receive_files(&staging_dir, &plan, &relay).and_then(|()| {
                // Cache readers resolve model.safetensors and companions at the entry root, not
                // under a transport directory; flatten the received sparse files in place.
                let nested = staging_dir.join("packages").join(&plan.directory);
                for file in &plan.files {
                    let source = nested.join(&file.name);
                    let destination = if file.name == plan.primary {
                        staging_dir.join("model.safetensors")
                    } else {
                        staging_dir.join(&file.name)
                    };
                    if let Some(parent) = destination.parent() {
                        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    fs::rename(&source, &destination).map_err(|e| e.to_string())?;
                }
                let _ = fs::remove_dir_all(staging_dir.join("packages"));
                let path = staging_dir.join("model.safetensors");
                let verify = progress.phase("Verify rank package", Some(plan_bytes(plan)));
                verify_plan(&path, plan, rank, Some(&verify))?;
                verify.finish(format!("rank {rank} verified"));
                Ok(())
            });
            if let Err(error) = received {
                staging.abort();
                return Err(error);
            }
            phase.finish(format!("rank {rank} verified"));
        }
        let entry = staging.commit().map_err(|e| e.to_string())?;
        // Publication seals every file read-only, which changes its status time, so the marker
        // is recorded against the committed entry. It only enables the next launch's fast path;
        // never fail a verified rank because its cache cannot record one.
        let _ = write_verified_marker(entry.directory(), plan, rank);
        if let Err(error) =
            root.publish_source_ref(&selector, &plan.export, &entry, plan.expected_envelope)
        {
            // The entry is not yet selected; report the original failure and any cleanup conflict.
            let (model_key, entry_id) = (entry.model().to_owned(), entry.id().to_owned());
            drop(entry);
            return match root.remove(&model_key, &entry_id, false) {
                Ok(()) => Err(error.to_string().into()),
                Err(cleanup) => Err(format!(
                    "{error}; unselected cache entry {model_key}/{entry_id} could not be removed: {cleanup}"
                )
                .into()),
            };
        }
        // A verified rank package is complete for this host; select it only after publication.
        root.publish_entry(&entry).map_err(|e| e.to_string())?;
        Ok(entry)
    }
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
    // Receivers hash and publish each file before reusing channels for the next request.
    socket.set_read_timeout(Some(TIMEOUT))?;
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
    fn peer_cache_root_prefers_model_cache_then_package_cache() {
        use clap::Parser as _;
        let run = |arguments: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(arguments).unwrap();
            match cli.command {
                crate::cli::Command::Run(run) => *run,
                _ => panic!("expected run command"),
            }
        };
        let both = run(&[
            "flare",
            "run",
            "/tmp/package.safetensors",
            "--package-cache",
            "/tmp/package-cache",
            "--model-cache",
            "/tmp/model-cache",
        ]);
        assert_eq!(
            model_cache_root(&both),
            Some(std::path::Path::new("/tmp/model-cache"))
        );
        let package_only = run(&[
            "flare",
            "run",
            "/tmp/package.safetensors",
            "--package-cache",
            "/tmp/package-cache",
        ]);
        assert_eq!(
            model_cache_root(&package_only),
            Some(std::path::Path::new("/tmp/package-cache"))
        );
        let neither = run(&["flare", "run", "/tmp/package.safetensors"]);
        assert_eq!(model_cache_root(&neither), None);
    }

    #[test]
    fn plan_tensor_parallel_fills_missing_and_rejects_conflicts() {
        use clap::Parser as _;
        let run = |tensor_parallel: Option<&str>| {
            let mut arguments = vec![
                "flare".to_owned(),
                "run".to_owned(),
                "/tmp/package.safetensors".to_owned(),
            ];
            if let Some(value) = tensor_parallel {
                arguments.extend(["--tensor-parallel".to_owned(), value.to_owned()]);
            }
            let cli = crate::cli::Cli::try_parse_from(arguments).unwrap();
            match cli.command {
                crate::cli::Command::Run(run) => *run,
                _ => panic!("expected run command"),
            }
        };

        let mut omitted = run(None);
        select_plan_tensor_parallel(&mut omitted, 2).unwrap();
        assert_eq!(omitted.serving.hub.tensor_parallel, vec![2]);

        let mut matching = run(Some("2"));
        select_plan_tensor_parallel(&mut matching, 2).unwrap();
        assert_eq!(matching.serving.hub.tensor_parallel, vec![2]);

        let mut conflict = run(Some("4"));
        let error = select_plan_tensor_parallel(&mut conflict, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("differs from rank zero's plan"), "{error}");
        assert_eq!(conflict.serving.hub.tensor_parallel, vec![4]);
    }

    #[test]
    fn selected_plan_tp_verifies_existing_multi_deployment_package() {
        use clap::Parser as _;
        let temp = tempfile::tempdir().unwrap();
        let (files, _, plan) = cache_tests::fixture(temp.path(), "selected-plan-tp");
        let package = files
            .iter()
            .find(|path| path.ends_with("model.safetensors"))
            .unwrap()
            .parent()
            .unwrap();
        let arguments = vec![
            "flare".to_owned(),
            "run".to_owned(),
            package.join("model.safetensors").display().to_string(),
        ];
        let cli = crate::cli::Cli::try_parse_from(&arguments).unwrap();
        let mut args = match cli.command {
            crate::cli::Command::Run(run) => *run,
            _ => panic!("expected run command"),
        };
        select_plan_tensor_parallel(&mut args, plan.tensor_parallel).unwrap();
        let selected = args.serving.hub.tensor_parallel[0];
        let verified = runtime_package::VerifiedPackage::open_tensor_parallel(
            package,
            std::slice::from_ref(&plan.publisher),
            selected,
            Some(1),
        )
        .unwrap();
        let identity = verified.identity().unwrap();
        assert_eq!(identity.tensor_parallel, plan.tensor_parallel);
    }

    #[test]
    #[ignore = "requires FLARE_LEGACY_FIXTURE with a signed multi-host legacy package and publisher.json"]
    fn receive_rank_accepts_old_metadata_with_empty_platform() {
        use flare_helios::runtime_package::{VerifiedPackage, rank_ranges};
        let fixture = PathBuf::from(
            std::env::var_os("FLARE_LEGACY_FIXTURE")
                .expect("FLARE_LEGACY_FIXTURE must name the legacy fixture"),
        );
        let primary = fixture.join("model.safetensors");
        let profile = fixture.join("publisher.json");
        let publishers: Vec<runtime_package::Publisher> =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        let verified =
            VerifiedPackage::open(&primary, std::slice::from_ref(&publishers[0]), None).unwrap();
        let authorized = verified.authorize(&publishers[0].namespace, false).unwrap();
        let identity = authorized.identity().unwrap();
        let cluster = identity
            .cluster
            .clone()
            .expect("fixture must be multi-host");
        let rank = 1usize;
        let ranges = rank_ranges(&authorized, rank).unwrap();
        let primary_name = primary.file_name().unwrap().to_string_lossy().into_owned();
        let mut handles = Vec::new();
        let mut specs = Vec::new();
        for range in ranges {
            let name = if range.filename.is_empty() {
                primary_name.clone()
            } else {
                range.filename
            };
            specs.push(FileSpec {
                name,
                size: range.file.metadata().unwrap().len(),
                ranges: range.ranges,
            });
            handles.push(range.file);
        }
        let source = Source::start(
            vec![Vec::new(), handles],
            vec![Vec::new(), specs.clone()],
            1,
        )
        .unwrap();
        let network0 = TcpListener::bind("127.0.0.1:0").unwrap();
        let network1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let ports = vec![
            network0.local_addr().unwrap().port(),
            network1.local_addr().unwrap().port(),
        ];
        let topology = Topology::plan(
            &vec![
                b"a 127.0.0.1/8
"
                .to_vec();
                2
            ],
            &ports,
            false,
        )
        .unwrap();
        let _relay0 = Relay::start(0, 1, topology.clone(), network0, source.port).unwrap();
        let relay1 = Relay::start(1, 1, topology.clone(), network1, 0).unwrap();
        let plan = Plan {
            version: VERSION,
            rank,
            export: cluster.export,
            primary: primary_name,
            directory: format!("legacy-export-rank{rank}"),
            files: specs.clone(),
            model: identity.model,
            platform: String::new(),
            tensor_parallel: identity.tensor_parallel,
            expected_envelope: authorized.envelope_digest(),
            publisher: publishers[0].clone(),
            topology,
            streams: 1,
            stage_only: true,
        };
        let model_root = tempfile::tempdir().unwrap();
        let entry = receive_rank(
            model_root.path(),
            &plan,
            &relay1,
            &StartupProgress::silent(),
            rank,
        )
        .unwrap();
        assert!(
            entry.id().starts_with("unknown-device-tp2-"),
            "empty platform did not produce an explicit unknown-device name: {}",
            entry.id()
        );
    }

    #[test]
    fn receiver_name_uses_package_platform_or_explicit_unknown_label() {
        let gb202 = receiver_candidate("gb202", 2).unwrap();
        assert!(gb202.starts_with("gb202-tp2-"), "{gb202}");
        let unknown = receiver_candidate("", 2).unwrap();
        assert!(unknown.starts_with("unknown-device-tp2-"), "{unknown}");
        let sm = receiver_candidate("sm120", 2).unwrap();
        assert!(sm.starts_with("unknown-device-tp2-"), "{sm}");
        assert!(!sm.contains("sm120"), "{sm}");
    }

    #[test]
    fn source_waits_for_receiver_file_verification() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"payload").unwrap();
        let files = vec![vec![Arc::new(file)]; 2];
        let specs = vec![
            vec![FileSpec {
                name: "file".into(),
                size: 7,
                ranges: vec![(0, 7)],
            }];
            2
        ];
        let source = Source::start(files, specs, 1).unwrap();
        let mut socket = TcpStream::connect(("127.0.0.1", source.port)).unwrap();
        configure(&socket, IO_TIMEOUT).unwrap();
        socket.write_all(&[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        for request in 0..2 {
            if request == 1 {
                thread::sleep(IO_TIMEOUT + Duration::from_secs(10));
            }
            write(
                &mut socket,
                &Request {
                    file: 0,
                    ranges: vec![(0, 7)],
                },
            )
            .unwrap();
            let mut bytes = [0; 15];
            socket.read_exact(&mut bytes).unwrap();
            assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()), 7);
            assert_eq!(&bytes[8..], b"payload");
        }
        socket.shutdown(Shutdown::Both).unwrap();
        source.finish().unwrap();
    }

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
                            model: "fixture".into(),
                            platform: "gb202".into(),
                            tensor_parallel: world as u32,
                            expected_envelope: [0; 32],
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

#[cfg(test)]
mod cache_tests;
