//! The resident agent: one per host, started by systemd as `flare agent`.
//!
//! The agent owns its node key, its authorized controller set, and a cache of executables and
//! package files. Over an authenticated session it pairs with a controller, receives files,
//! fetches files from other nodes under a grant, runs this host's rank of a group under a grant,
//! and streams the rank's output to whoever attaches. It supervises the rank for the process's
//! life: a controller going away neither stops the rank nor loses its output, and a STOP from an
//! authorized controller sends SIGTERM.
//!
//! Request bodies are JSON except PUT, whose body is a little-endian `u32` header length, the
//! JSON header, and the transferred bytes in range order.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use flare_helios::runtime_package::{Publisher, VerifiedPackage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::cache::{self, Pinned, Pins};
use super::gpu::{self, Gpu};
use super::grant::{Grant, Signed};
use super::identity::{Key, PublicKey};
use super::session::{FRAME_LIMIT, Role, Session};
use super::store::AgentDir;
use super::{Failure, PROTOCOL_VERSION, frame, hex, hmac_sha256, now};

/// The first frame each side sends: who it is and what it speaks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hello {
    pub version: u32,
    /// `controller` or `node`.
    pub kind: String,
    pub name: String,
    pub public_key: String,
    /// The controller-set revision the sender holds; 0 when unpaired or for a controller.
    pub controllers_revision: u64,
    #[serde(default)]
    pub free_bytes: u64,
    /// The node's installed Flare, used when RUN omits a cached binary name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executable: Option<PathBuf>,
}

impl Hello {
    pub fn installed_executable(&self) -> Result<&Path, Failure> {
        self.executable.as_deref().ok_or_else(|| {
            "agent does not advertise an installed Flare executable; update it with `flare agent install`, or supply a node-compatible build with --executable PATH"
                .into()
        })
    }
}

/// A repository download on the node, using a controller-resolved revision and access token.
#[derive(Serialize, Deserialize)]
pub struct DownloadRequest {
    pub source: String,
    pub options: crate::cli::HubOptions,
    pub access_token: Option<String>,
    /// Public publisher profiles supplied by the controller. Never private
    /// material or controller-local paths.
    #[serde(default)]
    pub profiles: Vec<flare_helios::runtime_package::Publisher>,
}

fn download(
    request: DownloadRequest,
) -> Result<(Vec<u8>, flare_helios::package::cache::Entry), Failure> {
    if request.options.revision.is_none() || request.options.token_file.is_some() {
        return Err(
            "DOWNLOAD requires an explicit revision; token-file paths stay on the controller"
                .into(),
        );
    }
    let path = crate::hub::download_with_token(
        &request.source,
        &request.options,
        0,
        request.access_token,
        &[],
        None,
        &request.profiles,
    )
    .map_err(|e| e.to_string())?;
    // The response names a cache entry the controller will transfer from, so the
    // lease must outlive this request rather than dropping with the Entry.
    let directory = path.directory().to_path_buf();
    Ok((serde_json::to_vec(&json!({"path": directory}))?, path))
}

/// Exchange HELLO frames: the initiator sends first. The peer's HELLO must name the key that
/// authenticated the session and speak the same protocol.
pub fn exchange_hello(session: &mut Session, mine: &Hello) -> Result<Hello, Failure> {
    let read = |session: &mut Session| -> Result<Hello, Failure> {
        let body = session.expect(frame::HELLO)?;
        let hello: Hello = serde_json::from_slice(&body)?;
        if hello.version != PROTOCOL_VERSION {
            return Err(format!(
                "peer speaks agent protocol {}; this flare speaks {PROTOCOL_VERSION}",
                hello.version
            )
            .into());
        }
        if PublicKey::from_hex(&hello.public_key)? != session.remote() {
            return Err("peer's HELLO names a key other than the one it authenticated".into());
        }
        Ok(hello)
    };
    let body = serde_json::to_vec(mine)?;
    match session.role() {
        Role::Initiator => {
            session.write_frame(frame::HELLO, &body)?;
            read(session)
        }
        Role::Responder => {
            let theirs = read(session)?;
            session.write_frame(frame::HELLO, &body)?;
            Ok(theirs)
        }
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The URL of rank zero's metrics endpoint for its last `--host` and `--port` options;
/// a wildcard bind is reached through the loopback address.
pub fn metrics_url(rank_zero: &[String]) -> String {
    let option = |name: &str| {
        rank_zero
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, argument)| {
                if argument == name {
                    rank_zero.get(i + 1).cloned()
                } else {
                    argument
                        .strip_prefix(name)?
                        .strip_prefix('=')
                        .map(str::to_owned)
                }
            })
    };
    let host = match option("--host").as_deref() {
        None | Some("0.0.0.0") | Some("::") => "127.0.0.1".to_owned(),
        Some(host) if host.contains(':') => format!("[{host}]"),
        Some(host) => host.to_owned(),
    };
    let port = option("--port").unwrap_or_else(|| "8000".into());
    format!("http://{host}:{port}/metrics?format=json")
}

/// Free bytes on the filesystem holding `path`.
pub fn free_bytes(path: &Path) -> u64 {
    let Ok(path) = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()) else {
        return 0;
    };
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: path is a valid C string and stat is a valid out-pointer.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return 0;
    }
    stat.f_bavail as u64 * stat.f_frsize as u64
}

/// The JSON body of a frame, read within [`FRAME_LIMIT`].
fn json_body(session: &mut Session, length: u64) -> Result<Value, Failure> {
    Ok(serde_json::from_slice(
        &session.read_body(length, FRAME_LIMIT)?,
    )?)
}

fn field<'a>(request: &'a Value, name: &str) -> Result<&'a str, Failure> {
    request[name]
        .as_str()
        .ok_or_else(|| format!("request lacks {name}").into())
}

// ---- groups ------------------------------------------------------------------------------------

enum Event {
    Line(String),
    Exit(i32),
}

struct Log {
    lines: VecDeque<String>,
    bytes: usize,
    exit: Option<(i32, Instant)>,
    subscribers: Vec<mpsc::Sender<Event>>,
    /// Cache pins held while the rank runs.
    pins: Option<Pinned>,
}

const RETAINED_BYTES: usize = 64 << 10;

/// A rank process this agent started and supervises.
pub struct Group {
    pub id: String,
    pub name: String,
    pub rank: usize,
    pub package: String,
    /// UUIDs of the GPUs the rank holds while it runs.
    pub gpus: Vec<String>,
    pub pid: i32,
    /// Rank zero's metrics URL, derived from its HTTP bind options; None for a peer rank.
    pub metrics_url: Option<String>,
    started: Instant,
    log: Mutex<Log>,
    /// The latest metrics document scraped from the rank, with the Unix millisecond it was read.
    metrics: Mutex<Option<(u64, Value)>>,
}

impl Group {
    fn push(&self, line: String) {
        let mut log = self.log.lock().unwrap();
        log.bytes += line.len();
        log.lines.push_back(line.clone());
        while log.bytes > RETAINED_BYTES && log.lines.len() > 1 {
            let dropped = log.lines.pop_front().unwrap();
            log.bytes -= dropped.len();
        }
        log.subscribers
            .retain(|s| s.send(Event::Line(line.clone())).is_ok());
    }

    fn finish(&self, code: i32) {
        let mut log = self.log.lock().unwrap();
        log.exit = Some((code, Instant::now()));
        log.subscribers
            .retain(|s| s.send(Event::Exit(code)).is_ok());
        log.subscribers.clear();
        log.pins = None;
    }

    pub fn running(&self) -> bool {
        self.log.lock().unwrap().exit.is_none()
    }

    pub fn exit(&self) -> Option<i32> {
        self.log.lock().unwrap().exit.map(|(code, _)| code)
    }

    fn retained_at(&self, now: Instant) -> bool {
        let log = self.log.lock().unwrap();
        match log.exit {
            None => true,
            Some((0, finished)) => now.saturating_duration_since(finished) < SUCCESS_RETENTION,
            Some(_) => true,
        }
    }

    /// Read the rank's metrics endpoint and keep the document.
    fn scrape(&self) {
        let Some(url) = &self.metrics_url else { return };
        static CLIENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
        let client = CLIENT.get_or_init(|| {
            ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(2)))
                .build()
                .into()
        });
        let document = client
            .get(url)
            .call()
            .ok()
            .and_then(|mut response| response.body_mut().read_to_string().ok())
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        if let Some(document) = document {
            *self.metrics.lock().unwrap() = Some((now_millis(), document));
        }
    }

    /// The retained output, the exit status if finished, and a receiver of later events.
    fn subscribe(&self) -> (Vec<String>, Option<i32>, mpsc::Receiver<Event>) {
        let mut log = self.log.lock().unwrap();
        let (sender, receiver) = mpsc::channel();
        if log.exit.is_none() {
            log.subscribers.push(sender);
        }
        (
            log.lines.iter().cloned().collect(),
            log.exit.map(|(code, _)| code),
            receiver,
        )
    }
}

// ---- state -------------------------------------------------------------------------------------

/// Verifies a package's stored data at `primary` for `rank` (None for a complete copy) against
/// the publisher's signing key.
pub type Verifier =
    Box<dyn Fn(&Path, Option<usize>, &Publisher) -> Result<(), Failure> + Send + Sync>;

/// The marker a package cache directory holds once its stored data verified.
const VERIFIED: &str = ".verified";
/// Successful runs retained for STATUS and ATTACH; failures require explicit removal.
const RETAINED_GROUPS: usize = 16;
const SUCCESS_RETENTION: Duration = Duration::from_secs(5 * 60);

/// Bound successful history by time since exit and count. Preserve running and failed ranks.
fn prune_groups(groups: &mut Vec<Arc<Group>>, now: Instant) {
    groups.retain(|group| group.retained_at(now));
    let mut excess = groups
        .iter()
        .filter(|group| group.exit() == Some(0))
        .count()
        .saturating_sub(RETAINED_GROUPS);
    groups.retain(|group| {
        if excess > 0 && group.exit() == Some(0) {
            excess -= 1;
            false
        } else {
            true
        }
    });
}

/// Enumerates the host's GPUs.
pub type Gpus = Box<dyn Fn() -> Result<Vec<Gpu>, Failure> + Send + Sync>;

fn verify_package(
    primary: &Path,
    rank: Option<usize>,
    publisher: &Publisher,
) -> Result<(), Failure> {
    VerifiedPackage::open(primary, std::slice::from_ref(publisher), rank)
        .and_then(|package| package.verify_stored_data())
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub struct Agent {
    pub dir: AgentDir,
    pub key: Key,
    pub cache: PathBuf,
    pub name: String,
    /// The executable running this agent, also used for serving unless RUN overrides it.
    executable: PathBuf,
    pins: Pins,
    /// Groups started by this agent, oldest first; finished ones are retained up to a limit.
    groups: Mutex<Vec<Arc<Group>>>,
    verifier: Verifier,
    gpus: Gpus,
    /// The GPU inventory the sampler last read, with the Unix millisecond it was read.
    sampled_gpus: Mutex<Option<(u64, Vec<Gpu>)>>,
    #[cfg(test)]
    fabric_endpoints: Option<Vec<u8>>,
}

/// How often the sampler reads GPUs and scrapes running rank-zero groups.
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);

impl Agent {
    pub fn new(dir: AgentDir, cache: PathBuf) -> Result<Self, Failure> {
        Self::with_fakes(
            dir,
            cache,
            Box::new(verify_package),
            Box::new(gpu::enumerate),
        )
    }

    /// An agent verifying fetched packages with `verifier` and enumerating GPUs with `gpus`
    /// instead of Helios and NVML.
    pub fn with_fakes(
        dir: AgentDir,
        cache: PathBuf,
        verifier: Verifier,
        gpus: Gpus,
    ) -> Result<Self, Failure> {
        let key = dir.key()?;
        for directory in ["bin", "packages", "publishers"] {
            fs::create_dir_all(cache.join(directory))?;
        }
        Ok(Self {
            dir,
            key,
            cache,
            name: super::hostname(),
            executable: std::env::current_exe()?,
            pins: Pins::default(),
            groups: Mutex::new(Vec::new()),
            verifier,
            gpus,
            sampled_gpus: Mutex::new(None),
            #[cfg(test)]
            fabric_endpoints: None,
        })
    }

    /// Read the GPUs and every running rank zero's metrics once.
    pub fn sample(&self) {
        if let Ok(gpus) = (self.gpus)() {
            *self.sampled_gpus.lock().unwrap() = Some((now_millis(), gpus));
        }
        for group in self.groups().iter().filter(|g| g.running()) {
            group.scrape();
        }
    }

    /// The METRICS reply: the sampled GPUs with their holders, and every group with its latest
    /// metrics document.
    fn metrics(&self) -> Value {
        let groups = self.groups();
        let (gpus_at, gpus) = self
            .sampled_gpus
            .lock()
            .unwrap()
            .clone()
            .unwrap_or((0, Vec::new()));
        json!({
            "at": now_millis(),
            "gpus_at": gpus_at,
            "gpus": gpus
                .iter()
                .map(|gpu| {
                    let holder = groups
                        .iter()
                        .find(|g| g.running() && g.gpus.contains(&gpu.uuid))
                        .map(|g| g.id.clone());
                    let mut value = serde_json::to_value(gpu).unwrap_or(Value::Null);
                    value["group"] = holder.into();
                    value
                })
                .collect::<Vec<_>>(),
            "groups": groups
                .iter()
                .map(|g| {
                    let (metrics_at, metrics) = g
                        .metrics
                        .lock()
                        .unwrap()
                        .clone()
                        .map_or((Value::Null, Value::Null), |(at, m)| (at.into(), m));
                    json!({
                        "id": g.id,
                        "name": g.name,
                        "rank": g.rank,
                        "gpus": g.gpus,
                        "running": g.running(),
                        "exit": g.exit(),
                        "uptime_seconds": g.started.elapsed().as_secs(),
                        "metrics_at": metrics_at,
                        "metrics": metrics,
                    })
                })
                .collect::<Vec<_>>(),
        })
    }

    pub fn hello(&self) -> Result<Hello, Failure> {
        Ok(Hello {
            version: PROTOCOL_VERSION,
            kind: "node".into(),
            name: self.name.clone(),
            public_key: self.key.public().to_hex(),
            controllers_revision: self.dir.controllers()?.map_or(0, |(_, set)| set.revision),
            free_bytes: free_bytes(&self.cache),
            executable: Some(self.executable.clone()),
        })
    }

    fn is_controller(&self, key: &PublicKey) -> Result<bool, Failure> {
        Ok(self
            .dir
            .controllers()?
            .is_some_and(|(_, set)| set.contains(key)))
    }

    /// Verify a grant against this node's controllers.
    fn grant(&self, signed: &Value) -> Result<Grant, Failure> {
        let signed: Signed = serde_json::from_value(signed.clone())?;
        let Some((_, set)) = self.dir.controllers()? else {
            return Err("this node is not paired with any controller".into());
        };
        let grant: Grant = signed.verify(|k| set.contains(k))?;
        if grant.members.is_empty() || grant.package.files.is_empty() {
            return Err("grant names no members or files".into());
        }
        Ok(grant)
    }

    fn groups(&self) -> Vec<Arc<Group>> {
        let mut groups = self.groups.lock().unwrap();
        // The sampler calls this even when no controller is connected or launching work.
        prune_groups(&mut groups, Instant::now());
        groups.clone()
    }

    /// The group a request names, or the only group when it names none.
    fn find_group(&self, id: Option<&str>) -> Result<Arc<Group>, Failure> {
        let groups = self.groups();
        match id {
            Some(id) => groups
                .iter()
                .find(|g| g.id == id)
                .cloned()
                .ok_or_else(|| format!("group {id} is not on this node").into()),
            None => match groups.as_slice() {
                [] => Err("no group has run on this node".into()),
                [only] => Ok(only.clone()),
                several => Err(format!(
                    "several groups are on this node; name one: {}",
                    several
                        .iter()
                        .map(|g| g.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .into()),
            },
        }
    }

    /// Package cache directories with their files and verification state.
    fn packages(&self) -> Vec<Value> {
        use std::os::unix::fs::MetadataExt;
        let mut packages = Vec::new();
        let Ok(entries) = fs::read_dir(self.cache.join("packages")) else {
            return packages;
        };
        let mut directories: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_dir()
                    && !p
                        .file_name()
                        .is_none_or(|n| n.to_string_lossy().starts_with('.'))
            })
            .collect();
        directories.sort();
        for directory in directories {
            let mut files = Vec::new();
            let mut bytes = 0u64;
            if let Ok(entries) = fs::read_dir(&directory) {
                let mut names: Vec<PathBuf> =
                    entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
                names.sort();
                for path in names {
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    let Ok(metadata) = fs::metadata(&path) else {
                        continue;
                    };
                    bytes += metadata.blocks() * 512;
                    if !name.starts_with('.') {
                        files.push(json!({"name": name, "size": metadata.len()}));
                    }
                }
            }
            packages.push(json!({
                "name": directory.file_name().unwrap().to_string_lossy(),
                "files": files,
                "verified": directory.join(VERIFIED).is_file(),
                "bytes": bytes,
            }));
        }
        packages
    }
}

/// Serve sessions on `listen` until killed.
pub fn serve(listen: &str, dir: AgentDir, cache: PathBuf) -> Result<(), Failure> {
    let listener =
        TcpListener::bind(listen).map_err(|e| format!("cannot listen on {listen}: {e}"))?;
    let agent = Arc::new(Agent::new(dir, cache)?);
    eprintln!(
        "flare agent {} ({}) listening on {}; cache {}; {}",
        agent.name,
        agent.key.public().fingerprint(),
        listener.local_addr()?,
        agent.cache.display(),
        match agent.dir.controllers()? {
            Some((_, set)) => format!("controller set revision {}", set.revision),
            None => "unpaired".to_owned(),
        }
    );
    serve_on(listener, agent)
}

pub fn serve_on(listener: TcpListener, agent: Arc<Agent>) -> Result<(), Failure> {
    {
        let agent = agent.clone();
        thread::spawn(move || {
            loop {
                agent.sample();
                thread::sleep(SAMPLE_PERIOD);
            }
        });
    }
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let agent = agent.clone();
        thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map(|a| a.to_string())
                .unwrap_or_default();
            if let Err(error) = session(&agent, stream) {
                eprintln!("{peer}: {error}");
            }
        });
    }
    Ok(())
}

fn session(agent: &Arc<Agent>, stream: TcpStream) -> Result<(), Failure> {
    let mut session = Session::accept(stream, &agent.key)?;
    let hello = exchange_hello(&mut session, &agent.hello()?)?;
    let mut pinned = Pinned::new(&agent.pins);
    // Downloads stay leased for this controller session, which is the window in
    // which its transfers read those entries, and drop when the session ends.
    let mut downloads: Vec<flare_helios::package::cache::Entry> = Vec::new();
    loop {
        let (tag, length) = match session.read_header() {
            Ok(header) => header,
            Err(_) => return Ok(()),
        };
        let authorized = agent.is_controller(&session.remote())?;
        let result = match tag {
            frame::PAIR => {
                let request = json_body(&mut session, length)?;
                pair(agent, &mut session, &hello, &request)
            }
            frame::FETCH => {
                let request = json_body(&mut session, length)?;
                match fetch(agent, &mut session, &mut pinned, &request) {
                    Ok(()) => continue,
                    Err(error) => Err(error),
                }
            }
            frame::FETCH_FABRIC => {
                let request = json_body(&mut session, length)?;
                match fetch_fabric(agent, &mut session, &mut pinned, &request) {
                    Ok(()) => continue,
                    Err(error) => Err(error),
                }
            }
            _ if !authorized => Err(format!(
                "{} is not an authorized controller of this node",
                session.remote().fingerprint()
            )
            .into()),
            frame::CONTROLLERS => {
                let record: Signed = serde_json::from_value(json_body(&mut session, length)?)?;
                agent
                    .dir
                    .apply_controllers(&record)
                    .map(|set| json!({"revision": set.revision}).to_string().into_bytes())
            }
            frame::HAVE => {
                let request = json_body(&mut session, length)?;
                have(agent, &mut pinned, &request)
            }
            frame::PUT => put(agent, &mut session, &mut pinned, length),
            frame::TRANSFER => {
                let request = json_body(&mut session, length)?;
                transfer(agent, &mut pinned, &request)
            }
            frame::RUN | frame::ATTACH => {
                let request = json_body(&mut session, length)?;
                let started = if tag == frame::RUN {
                    run(agent, &request)
                } else {
                    attach(agent, &request)
                };
                match started {
                    Ok(group) => {
                        session.ok(json!({"pid": group.pid, "group": group.id})
                            .to_string()
                            .as_bytes())?;
                        return stream_group(&mut session, &group);
                    }
                    Err(error) => Err(error),
                }
            }
            frame::STOP => {
                let request = json_body(&mut session, length)?;
                stop(agent, &request)
            }
            frame::REMOVE => {
                let request = json_body(&mut session, length)?;
                remove(agent, &request)
            }
            frame::STATUS => {
                let request = json_body(&mut session, length)?;
                status(agent, &request)
            }
            frame::FABRIC => {
                session.read_body(length, FRAME_LIMIT)?;
                #[cfg(test)]
                if let Some(endpoints) = &agent.fabric_endpoints {
                    session.ok(endpoints)?;
                    continue;
                }
                super::fabric::discover()
            }
            frame::METRICS => {
                session.read_body(length, FRAME_LIMIT)?;
                Ok(agent.metrics().to_string().into_bytes())
            }
            frame::GET => {
                let request = json_body(&mut session, length)?;
                match get(&mut session, &request) {
                    Ok(()) => continue,
                    Err(error) => Err(error),
                }
            }
            frame::STAT => {
                let request = json_body(&mut session, length)?;
                stat(&request)
            }
            frame::DOWNLOAD => {
                let request = serde_json::from_value(json_body(&mut session, length)?)?;
                match download(request) {
                    Ok((reply, entry)) => {
                        downloads.push(entry);
                        Ok(reply)
                    }
                    Err(error) => Err(error),
                }
            }
            _ => Err(format!("unknown request {tag}").into()),
        };
        match result {
            Ok(reply) => session.ok(&reply)?,
            Err(error) => {
                session.error(&error.to_string())?;
                return Err(error);
            }
        }
    }
}

fn constant_time_equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

fn pair(
    agent: &Agent,
    session: &mut Session,
    hello: &Hello,
    request: &Value,
) -> Result<Vec<u8>, Failure> {
    if agent.dir.controllers()?.is_some() {
        return Err("this node is paired; add controllers with `flare orbit controllers add` or run `flare agent reset` on it".into());
    }
    let record: Signed = serde_json::from_value(request["controllers"].clone())
        .map_err(|_| "pairing needs the controller-set record")?;
    if record.signer()? != session.remote() {
        return Err("the controller-set record must be signed by the pairing controller".into());
    }
    let mut reply = json!({
        "name": agent.name,
        "public_key": agent.key.public().to_hex(),
        "fingerprint": agent.key.public().fingerprint(),
    });
    let hash = *session.handshake_hash();
    match (agent.dir.join_code()?, request["proof"].as_str()) {
        (Some(code), Some(proof)) => {
            let expected = hmac_sha256(&code.key, &[b"controller".as_slice(), &hash].concat());
            if !constant_time_equal(&super::unhex(proof)?, &expected) {
                return Err("join code proof rejected".into());
            }
            let set = agent.dir.apply_controllers(&record)?;
            agent.dir.consume_join_code()?;
            agent.dir.clear_accept_first();
            reply["proof"] = hex(&hmac_sha256(
                &code.key,
                &[b"node".as_slice(), &hash].concat(),
            ))
            .into();
            reply["revision"] = set.revision.into();
        }
        _ if agent.dir.accept_first() => {
            let set = agent.dir.apply_controllers(&record)?;
            agent.dir.clear_accept_first();
            agent.dir.consume_join_code()?;
            reply["revision"] = set.revision.into();
        }
        (Some(_), None) => return Err("this node requires its join code".into()),
        (None, _) => {
            return Err(
                "this node accepts no pairing: install it with a join code or --accept-first"
                    .into(),
            );
        }
    }
    eprintln!(
        "paired with controller {} ({})",
        hello.name,
        session.remote().fingerprint()
    );
    Ok(reply.to_string().into_bytes())
}

fn have(agent: &Agent, pinned: &mut Pinned, request: &Value) -> Result<Vec<u8>, Failure> {
    let size = request["size"].as_u64().ok_or("request lacks size")?;
    let held = cache::have(
        &agent.cache,
        field(request, "kind")?,
        field(request, "name")?,
        size,
    )?;
    if let Some(path) = &held {
        pinned.pin(cache::unit(&agent.cache, path));
    }
    Ok(vec![held.is_some() as u8])
}

fn put(
    agent: &Agent,
    session: &mut Session,
    pinned: &mut Pinned,
    length: u64,
) -> Result<Vec<u8>, Failure> {
    let mut body = session.body();
    let mut prefix = [0u8; 4];
    body.read_exact(&mut prefix)?;
    let header_length = u32::from_le_bytes(prefix) as usize;
    if header_length as u64 + 4 > length || header_length > FRAME_LIMIT as usize {
        return Err("invalid PUT header".into());
    }
    let mut header = vec![0u8; header_length];
    body.read_exact(&mut header)?;
    let request: Value = serde_json::from_slice(&header)?;
    let kind = field(&request, "kind")?;
    let name = field(&request, "name")?;
    let size = request["size"].as_u64().ok_or("request lacks size")?;
    let ranges = cache::ranges_from_json(size, request.get("ranges"))?;
    if cache::range_bytes(&ranges) + 4 + header_length as u64 != length {
        return Err("PUT length does not match its ranges".into());
    }
    let entry = cache::entry(&agent.cache, cache::kind_directory(kind)?, name)?;
    let lease = cache::Lease::write(&cache::unit(&agent.cache, &entry))?;
    let path = cache::store(&agent.cache, kind, name, size, &ranges, &mut body)?;
    let unit = cache::unit(&agent.cache, &path);
    if kind == "package" {
        let _ = fs::remove_file(unit.join(VERIFIED));
    }
    pinned.pin(unit.clone());
    for removed in cache::evict(unit.parent().unwrap(), &agent.pins)? {
        eprintln!("removed superseded {}", removed.display());
    }
    Ok(Vec::new())
}

/// Fetch the files of a package from another node under a grant into the cache.
fn transfer(agent: &Agent, pinned: &mut Pinned, request: &Value) -> Result<Vec<u8>, Failure> {
    let grant = agent.grant(&request["grant"])?;
    let rank = grant
        .member(&agent.key.public())
        .map(|m| m.rank)
        .ok_or("this node is not a member of the grant")?;
    let publisher: Publisher = serde_json::from_value(request["publisher"].clone())
        .map_err(|_| "TRANSFER lacks the publisher profile")?;
    publisher.validate().map_err(|e| e.to_string())?;
    let source_key = PublicKey::from_hex(field(&request["source"], "public_key")?)?;
    let address = field(&request["source"], "address")?;
    let directory = field(request, "directory")?;
    let files = request["files"].as_array().ok_or("request lacks files")?;
    let primary = cache::entry(
        &agent.cache,
        "packages",
        &format!("{directory}/{}", grant.primary()),
    )?;
    let lease = cache::Lease::write(primary.parent().unwrap())?;
    let marker = primary.with_file_name(VERIFIED);
    let fabric_source = if grant.transfer_fabric.is_some() {
        super::fabric::topology(&grant)?;
        if let Some(member) = grant.member(&source_key) {
            if member.rank != super::fabric::source_rank(&grant) {
                return Err("transfer source differs from the fabric package source".into());
            }
            Some(member.rank)
        } else if rank == 0
            && grant
                .package
                .source
                .as_ref()
                .is_some_and(|s| s.public_key == source_key.to_hex())
        {
            None // The one ingress transfer from a source outside the fabric membership.
        } else {
            return Err("fabric transfer source must be a member, or ingress to rank zero".into());
        }
    } else {
        None
    };
    let streams = match request.get("streams") {
        None => 1,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=64).contains(n))
            .ok_or("transfer streams must be in 1..=64")? as usize,
    };
    if streams > 1 && fabric_source.is_none() {
        return Err("parallel package transfers require fabric endpoints".into());
    }
    let mut received = 0u64;
    for (file_index, file) in files.iter().enumerate() {
        let name = field(file, "name")?;
        let size = file["size"].as_u64().ok_or("file lacks size")?;
        let ranges = cache::ranges_from_json(size, file.get("ranges"))?;
        let entry = format!("{directory}/{name}");
        if let Some(path) = cache::have(&agent.cache, "package", &entry, size)? {
            pinned.pin(cache::unit(&agent.cache, &path));
            continue;
        }
        // Bytes are arriving, so whatever the directory held is no longer verified.
        let _ = fs::remove_file(&marker);
        let started = Instant::now();
        let path = if let Some(source) = fabric_source {
            let route = super::fabric::topology(&grant)?.path(source, rank)?;
            eprintln!(
                "{name}: fabric TCP {} ({streams} streams)",
                route
                    .iter()
                    .map(|r| grant.members[*r].name.as_str())
                    .collect::<Vec<_>>()
                    .join(" -> ")
            );
            cache::store_parallel(
                &agent.cache,
                &entry,
                size,
                &ranges,
                streams,
                |stream, part| {
                    let upstream = fabric_upstream(
                        agent,
                        &grant,
                        &json!({
                            "grant": request["grant"], "source_rank": source, "destination_rank": rank,
                            "rail": (file_index + stream) % 2,
                            "name": name, "size": size, "ranges": part,
                        }),
                        rank,
                    )?;
                    Ok(upstream.into_body())
                },
            )?
        } else {
            let mut session = Session::connect(address, &agent.key)?;
            authenticate_source(agent, &mut session, source_key)?;
            session.write_frame(
                frame::FETCH,
                &serde_json::to_vec(&json!({
                    "grant": request["grant"], "name": name, "ranges": ranges,
                }))?,
            )?;
            expect_bytes(&mut session, cache::range_bytes(&ranges))?;
            cache::store(
                &agent.cache,
                "package",
                &entry,
                size,
                &ranges,
                &mut session.body(),
            )?
        };
        let length = cache::range_bytes(&ranges);
        pinned.pin(cache::unit(&agent.cache, &path));
        received += length;
        eprintln!(
            "{name}: received {length} bytes in {:.2}s ({:.1} MiB/s)",
            started.elapsed().as_secs_f64(),
            length as f64 / (1 << 20) as f64 / started.elapsed().as_secs_f64()
        );
    }
    for removed in cache::evict(&agent.cache.join("packages"), &agent.pins)? {
        eprintln!("removed superseded {}", removed.display());
    }
    // Fetched bytes are verified once, here, and the directory remembers it; a rank started from
    // a marked directory loads without re-reading its stored data.
    let verified = !marker.is_file();
    if verified {
        let complete = rank == 0;
        if let Err(error) = (agent.verifier)(&primary, (!complete).then_some(rank), &publisher) {
            let _ = fs::remove_dir_all(primary.parent().unwrap());
            return Err(format!(
                "{directory}: stored data fetched from {address} failed verification and was removed: {error}"
            )
            .into());
        }
        fs::write(
            &marker,
            format!("{} {}\n", grant.primary(), grant.package.length),
        )?;
        eprintln!("{directory}: stored data verified");
    }
    Ok(json!({"received": received, "verified": verified})
        .to_string()
        .into_bytes())
}

fn authenticate_source(
    agent: &Agent,
    session: &mut Session,
    expected: PublicKey,
) -> Result<(), Failure> {
    exchange_hello(session, &agent.hello()?)?;
    if session.remote() != expected {
        return Err(
            "transfer connection authenticated a different node than the grant names".into(),
        );
    }
    Ok(())
}

fn expect_bytes(session: &mut Session, expected: u64) -> Result<(), Failure> {
    let (tag, length) = session.read_header()?;
    if tag == frame::ERROR {
        return Err(
            String::from_utf8_lossy(&session.read_body(length, FRAME_LIMIT)?)
                .into_owned()
                .into(),
        );
    }
    if tag != frame::OK || length != expected {
        return Err("unexpected byte count or response to package FETCH".into());
    }
    Ok(())
}

fn fabric_request(grant: &Grant, request: &Value) -> Result<(Vec<usize>, usize, u64), Failure> {
    super::fabric::topology(grant)?;
    let rank = |field: &str| -> Result<usize, Failure> {
        let value = request[field]
            .as_u64()
            .ok_or("fabric FETCH lacks rank or rail")?;
        usize::try_from(value).map_err(|_| "fabric FETCH rank overflows".into())
    };
    let source = rank("source_rank")?;
    if source != super::fabric::source_rank(grant) {
        return Err("fabric FETCH source differs from the grant's package source".into());
    }
    let path = super::fabric::topology(grant)?.path(source, rank("destination_rank")?)?;
    let rail = rank("rail")?;
    if rail >= 2 {
        return Err("fabric FETCH rail is outside the fabric capacity".into());
    }
    let name = field(request, "name")?;
    if !grant.package.files.iter().any(|file| file == name) {
        return Err("fabric FETCH file is outside the grant's package".into());
    }
    let size = request["size"]
        .as_u64()
        .ok_or("fabric FETCH lacks file size")?;
    let ranges = cache::ranges_from_json(size, request.get("ranges"))?;
    Ok((path, rail, cache::range_bytes(&ranges)))
}

/// Pull from the preceding rank on the payload path. Every connection stays on one physical link.
fn fabric_upstream(
    agent: &Agent,
    grant: &Grant,
    request: &Value,
    rank: usize,
) -> Result<Session, Failure> {
    let (path, rail, length) = fabric_request(grant, request)?;
    let index = path
        .iter()
        .position(|r| *r == rank)
        .ok_or("node is outside the transfer path")?;
    let previous = *path
        .get(index.checked_sub(1).ok_or("source has no upstream hop")?)
        .ok_or("missing upstream hop")?;
    let (local, remote) = super::fabric::topology(grant)?.hop(rank, previous, rail)?;
    let mut upstream = Session::connect_from(local, remote, &agent.key)?;
    authenticate_source(
        agent,
        &mut upstream,
        PublicKey::from_hex(&grant.members[previous].public_key)?,
    )?;
    upstream.write_frame(frame::FETCH_FABRIC, &serde_json::to_vec(request)?)?;
    expect_bytes(&mut upstream, length)?;
    Ok(upstream)
}

fn fetch_fabric(
    agent: &Agent,
    session: &mut Session,
    pinned: &mut Pinned,
    request: &Value,
) -> Result<(), Failure> {
    let grant = agent.grant(&request["grant"])?;
    let (path, rail, length) = fabric_request(&grant, request)?;
    let rank = grant
        .member(&agent.key.public())
        .ok_or("this relay is outside the grant")?
        .rank;
    let index = path
        .iter()
        .position(|r| *r == rank)
        .ok_or("this relay is outside the transfer path")?;
    let downstream = *path
        .get(index + 1)
        .ok_or("destination cannot relay its own FETCH")?;
    if session.remote() != PublicKey::from_hex(&grant.members[downstream].public_key)? {
        return Err("fabric FETCH did not arrive from the next member on its path".into());
    }
    let (from, to) = super::fabric::topology(&grant)?.hop(downstream, rank, rail)?;
    let (local, remote) = session.addresses()?;
    if local != std::net::SocketAddr::V4(to) || remote.ip() != std::net::IpAddr::V4(from) {
        return Err("fabric FETCH did not use the signed fabric interfaces".into());
    }
    session.transfer_timeout()?;
    if index == 0 {
        return fetch(agent, session, pinned, request);
    }
    let mut upstream = fabric_upstream(agent, &grant, request, rank)?;
    // One encrypted record at a time; TCP backpressure bounds memory at every hop.
    session.write_frame_from(frame::OK, length, &mut upstream.body())
}

/// Serve byte ranges of a package file to a member of the grant.
fn fetch(
    agent: &Agent,
    session: &mut Session,
    pinned: &mut Pinned,
    request: &Value,
) -> Result<(), Failure> {
    let grant = agent.grant(&request["grant"])?;
    if grant.member(&session.remote()).is_none() {
        return Err(format!(
            "{} is not a member of the grant",
            session.remote().fingerprint()
        )
        .into());
    }
    let name = field(request, "name")?;
    if !grant.package.files.iter().any(|f| f == name) {
        return Err(format!("{name} is not a file of the grant's package").into());
    }
    let mine = agent.key.public().to_hex();
    let path = match &grant.package.source {
        Some(source) if source.public_key == mine => Path::new(&source.path)
            .parent()
            .ok_or("source path has no directory")?
            .join(name),
        _ => {
            let complete = cache::entry(
                &agent.cache,
                "packages",
                &format!("{}/{name}", grant.complete_directory()),
            )?;
            if complete.is_file() {
                complete
            } else {
                let rank = grant
                    .member(&agent.key.public())
                    .map(|m| m.rank)
                    .ok_or("this node holds no copy of the package")?;
                cache::entry(
                    &agent.cache,
                    "packages",
                    &format!("{}/{name}", grant.cache_directory(rank)),
                )?
            }
        }
    };
    let file = File::open(&path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    pinned.pin(cache::unit(&agent.cache, &path));
    let size = file.metadata()?.len();
    let ranges = cache::ranges_from_json(size, request.get("ranges"))?;
    let mut reader = cache::RangeReader::new(&file, &ranges);
    session.write_frame_from(frame::OK, cache::range_bytes(&ranges), &mut reader)
}

/// Read a range of a node-local file: how a controller mirrors a package's header and index.
fn get(session: &mut Session, request: &Value) -> Result<(), Failure> {
    let path = Path::new(field(request, "path")?);
    if !path.is_absolute() {
        return Err("GET needs an absolute path".into());
    }
    let offset = request["offset"].as_u64().ok_or("request lacks offset")?;
    let nbytes = request["nbytes"].as_u64().ok_or("request lacks nbytes")?;
    if nbytes > 256 << 20 {
        return Err("GET reads at most 256 MiB".into());
    }
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = file.metadata()?.len();
    let ranges = [(offset, nbytes)];
    cache::validate_ranges(size, &ranges)?;
    let mut reader = cache::RangeReader::new(&file, &ranges);
    session.write_frame_from(frame::OK, nbytes, &mut reader)
}

/// A package's primary file size and its companions with their sizes.
fn stat(request: &Value) -> Result<Vec<u8>, Failure> {
    let path = Path::new(field(request, "path")?);
    if !path.is_absolute() {
        return Err("STAT needs an absolute path".into());
    }
    let size = package_file_size(path)?;
    let directory = path.parent().ok_or("path has no directory")?;
    let companions = flare_helios::runtime_package::package_companions(path)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|name| {
            let size = package_file_size(&directory.join(&name))?;
            Ok(json!({"name": name, "size": size}))
        })
        .collect::<Result<Vec<_>, Failure>>()?;
    Ok(json!({"size": size, "companions": companions})
        .to_string()
        .into_bytes())
}

fn package_file_size(path: &Path) -> Result<u64, Failure> {
    let info = fs::metadata(path)
        .map_err(|e| format!("cannot stat package file {}: {e}", path.display()))?;
    if !info.is_file() {
        return Err(format!(
            "package file must resolve to a regular file: {}",
            path.display()
        )
        .into());
    }
    Ok(info.len())
}

fn run(agent: &Arc<Agent>, request: &Value) -> Result<Arc<Group>, Failure> {
    let grant = agent.grant(&request["grant"])?;
    let rank = request["rank"].as_u64().ok_or("request lacks rank")? as usize;
    let mine = agent.key.public();
    let member = grant
        .rank(rank)
        .ok_or_else(|| format!("the grant has no rank {rank}"))?;
    if member.public_key != mine.to_hex() {
        return Err(format!("the grant names another node at rank {rank}").into());
    }
    if !grant.current(now()) {
        return Err("the grant has expired".into());
    }
    let mut pins = Pinned::new(&agent.pins);
    let binary = if request["binary"].is_null() {
        agent.executable.clone()
    } else {
        let binary_name = field(request, "binary")?;
        let binary = cache::entry(&agent.cache, "bin", binary_name)?;
        if !binary.is_file() {
            return Err(format!("executable {binary_name} is not cached").into());
        }
        pins.pin(binary.clone());
        binary
    };
    let publisher: Publisher = serde_json::from_value(request["publisher"].clone())?;
    publisher.validate().map_err(|e| e.to_string())?;
    let primary = grant.primary();
    let package = match &grant.package.source {
        Some(source) if source.public_key == mine.to_hex() => PathBuf::from(&source.path),
        _ => {
            let directory = if rank == 0 {
                grant.complete_directory()
            } else {
                grant.cache_directory(rank)
            };
            let path = cache::entry(&agent.cache, "packages", &format!("{directory}/{primary}"))?;
            pins.pin(cache::unit(&agent.cache, &path));
            path
        }
    };
    let local_package = rank == 0
        || grant.single_host
        || grant
            .package
            .source
            .as_ref()
            .is_some_and(|source| source.public_key == mine.to_hex());
    if local_package && !package.is_file() {
        return Err(format!("package {} is not on this node", package.display()).into());
    }
    let profile = agent
        .cache
        .join("publishers")
        .join(format!("{}.json", publisher.fingerprint()));
    fs::write(&profile, serde_json::to_vec(&vec![&publisher])?)?;
    let mut command = Command::new(&binary);
    command.arg("run");
    if local_package {
        command.arg(&package);
    }
    command.arg("--package-cache").arg(&agent.cache);
    if !grant.single_host {
        let port = grant
            .rendezvous
            .rsplit_once(':')
            .map(|(_, port)| port)
            .ok_or("grant rendezvous lacks a port")?;
        command
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--rendezvous")
            .arg(if rank == 0 {
                format!("0.0.0.0:{port}")
            } else {
                grant.rendezvous.clone()
            });
    }
    command.args(&grant.serving);
    if rank == 0 {
        command.args(&grant.rank_zero);
    }
    if grant.roce_ring && grant.roce_chain {
        return Err("grant selects both a ring and a chain".into());
    }
    if grant.roce_ring {
        command.arg("--roce-ring");
    }
    if grant.roce_chain {
        command.arg("--roce-chain");
    }
    // The controller accepted the publisher at its own terminal; the rank still verifies the
    // package and matches the exact key and namespace before loading.
    command
        .arg("--publisher-keys")
        .arg(&profile)
        .arg("--dangerously-approve-key")
        .arg(publisher.fingerprint())
        .arg("--dangerously-accept-publisher")
        .arg(&publisher.namespace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // The rank sees only its GPUs, in the grant's order; a member naming none takes the whole
    // node, as a rank started by hand would.
    let gpus = member.gpus.clone();
    if !gpus.is_empty() {
        let present = (agent.gpus)()?;
        for uuid in &gpus {
            if !present.iter().any(|g| &g.uuid == uuid) {
                return Err(format!("this node has no GPU {uuid}").into());
            }
        }
        command.env("CUDA_VISIBLE_DEVICES", gpu::visible_devices(&gpus));
        command.arg("--gpus").arg(
            (0..gpus.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    let mut groups = agent.groups.lock().unwrap();
    for current in groups.iter().filter(|g| g.running()) {
        let conflict = gpus.is_empty()
            || current.gpus.is_empty()
            || current.gpus.iter().any(|held| gpus.contains(held));
        if conflict {
            return Err(format!(
                "group {} ({}) runs on this node as rank {} and holds {}",
                current.id,
                current.name,
                current.rank,
                if current.gpus.is_empty() {
                    "every GPU".to_owned()
                } else {
                    current.gpus.join(",")
                }
            )
            .into());
        }
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", binary.display()))?;
    let group = Arc::new(Group {
        id: grant.group.clone(),
        name: grant.name.clone(),
        rank,
        package: package.display().to_string(),
        gpus,
        pid: child.id() as i32,
        metrics_url: (rank == 0).then(|| metrics_url(&grant.rank_zero)),
        started: Instant::now(),
        metrics: Mutex::new(None),
        log: Mutex::new(Log {
            lines: VecDeque::new(),
            bytes: 0,
            exit: None,
            subscribers: Vec::new(),
            pins: Some(pins),
        }),
    });
    groups.push(group.clone());
    prune_groups(&mut groups, Instant::now());
    drop(groups);
    eprintln!(
        "group {} ({}) rank {rank}: started pid {} on {}",
        group.id,
        group.name,
        group.pid,
        if group.gpus.is_empty() {
            "the node's GPUs".to_owned()
        } else {
            group.gpus.join(",")
        }
    );
    let mut forwarders = Vec::new();
    for output in [
        Box::new(child.stdout.take().unwrap()) as Box<dyn Read + Send>,
        Box::new(child.stderr.take().unwrap()),
    ] {
        let group = group.clone();
        forwarders.push(thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(Result::ok) {
                group.push(line);
            }
        }));
    }
    let finished = group.clone();
    thread::spawn(move || {
        let status = child.wait();
        for forwarder in forwarders {
            let _ = forwarder.join();
        }
        let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
        eprintln!(
            "group {} rank {}: exited with status {code}",
            finished.id, finished.rank
        );
        finished.finish(code);
    });
    Ok(group)
}

fn named_group(agent: &Agent, request: &Value) -> Result<Arc<Group>, Failure> {
    agent.find_group(request["group"].as_str())
}

fn attach(agent: &Agent, request: &Value) -> Result<Arc<Group>, Failure> {
    named_group(agent, request)
}

/// Send the group's retained output, then live output until it exits or the peer leaves.
fn stream_group(session: &mut Session, group: &Group) -> Result<(), Failure> {
    let (retained, exit, events) = group.subscribe();
    for line in retained {
        session.write_frame(frame::LOG, line.as_bytes())?;
    }
    if let Some(code) = exit {
        return session.write_frame(frame::EXIT, &code.to_le_bytes());
    }
    loop {
        match events.recv_timeout(Duration::from_secs(20)) {
            Ok(Event::Line(line)) => session.write_frame(frame::LOG, line.as_bytes())?,
            Ok(Event::Exit(code)) => return session.write_frame(frame::EXIT, &code.to_le_bytes()),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // An empty LOG frame keeps a silent attachment alive and detects a departed peer.
                session.write_frame(frame::LOG, b"")?;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let code = group.exit().unwrap_or(-1);
                return session.write_frame(frame::EXIT, &code.to_le_bytes());
            }
        }
    }
}

fn stop(agent: &Agent, request: &Value) -> Result<Vec<u8>, Failure> {
    let group = named_group(agent, request)?;
    let running = group.running();
    if running {
        // SAFETY: pid is this agent's child.
        unsafe { libc::kill(group.pid, libc::SIGTERM) };
        eprintln!("group {} rank {}: stop requested", group.id, group.rank);
    }
    Ok(json!({"stopped": running}).to_string().into_bytes())
}

/// Forget completed runs selected by exact launch identifier; never stop a live process.
fn remove(agent: &Agent, request: &Value) -> Result<Vec<u8>, Failure> {
    let id = field(request, "group")?;
    let mut groups = agent.groups.lock().unwrap();
    if groups.iter().any(|group| group.id == id && group.running()) {
        return Err(format!("group {id} is running; stop it before removing it").into());
    }
    let before = groups.len();
    groups.retain(|group| group.id != id);
    Ok(json!({"removed": before - groups.len()})
        .to_string()
        .into_bytes())
}

fn status(agent: &Agent, request: &Value) -> Result<Vec<u8>, Failure> {
    let groups = agent.groups();
    let describe = |g: &Group| {
        json!({
            "id": g.id,
            "name": g.name,
            "rank": g.rank,
            "package": g.package,
            "gpus": g.gpus,
            "pid": g.pid,
            "running": g.running(),
            "exit": g.exit(),
            "uptime_seconds": g.started.elapsed().as_secs(),
        })
    };
    let (gpus, gpus_error) = match (agent.gpus)() {
        Ok(gpus) => (
            gpus.iter()
                .map(|gpu| {
                    let holder = groups
                        .iter()
                        .find(|g| g.running() && g.gpus.contains(&gpu.uuid))
                        .map(|g| g.id.clone());
                    json!({
                        "index": gpu.index,
                        "uuid": gpu.uuid,
                        "name": gpu.name,
                        "revision": gpu.revision,
                        "memory_total": gpu.memory_total,
                        "memory_free": gpu.memory_free,
                        "group": holder,
                    })
                })
                .collect::<Vec<_>>(),
            Value::Null,
        ),
        Err(error) => (Vec::new(), error.to_string().into()),
    };
    let mut reply = json!({
        "name": agent.name,
        "public_key": agent.key.public().to_hex(),
        "fingerprint": agent.key.public().fingerprint(),
        "version": PROTOCOL_VERSION,
        "free_bytes": free_bytes(&agent.cache),
        "controllers_revision": agent.dir.controllers()?.map_or(0, |(_, s)| s.revision),
        "group": groups.last().map(|g| describe(g)),
        "groups": groups.iter().map(|g| describe(g)).collect::<Vec<_>>(),
        "gpus": gpus,
        "gpus_error": gpus_error,
        "packages": agent.packages(),
    });
    if let Some(toward) = request["toward"].as_str() {
        reply["address_toward"] = super::address_toward(toward)?.to_string().into();
    }
    Ok(reply.to_string().into_bytes())
}

// ---- install -----------------------------------------------------------------------------------

/// Install this executable as the host's agent service and prepare pairing.
pub fn install(listen: &str, join_code: Option<&str>, accept_first: bool) -> Result<(), Failure> {
    let dir = AgentDir::new(AgentDir::default_root());
    let key = dir.key()?;
    let home = super::home();
    let destination = home.join(".local/lib/flare-agent/flare");
    fs::create_dir_all(destination.parent().unwrap())?;
    let source = std::env::current_exe()?;
    let temporary = destination.with_extension("tmp");
    fs::copy(&source, &temporary)?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(0o755))?;
    fs::rename(&temporary, &destination)?;
    let unit_dir = home.join(".config/systemd/user");
    fs::create_dir_all(&unit_dir)?;
    fs::write(
        unit_dir.join("flare-agent.service"),
        format!(
            "[Unit]\nDescription=Flare agent: runs this host's rank of a orbit's serving group\nAfter=network-online.target\n\n[Service]\nExecStart=%h/.local/lib/flare-agent/flare agent --listen {listen}\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n"
        ),
    )?;
    let code = if accept_first {
        dir.set_accept_first()?;
        None
    } else {
        let code = match join_code {
            Some(code) => super::normalize_code(code),
            None => super::store::new_join_code()?,
        };
        if code.len() < 16 {
            return Err("a join code needs at least 16 characters".into());
        }
        dir.set_join_code(&code, 24 * 3600)?;
        Some(code)
    };
    let mut service = Vec::new();
    for arguments in [
        ["systemctl", "--user", "daemon-reload"].as_slice(),
        &["systemctl", "--user", "enable", "flare-agent"],
        &["systemctl", "--user", "restart", "flare-agent"],
    ] {
        match Command::new(arguments[0]).args(&arguments[1..]).status() {
            Ok(status) if status.success() => {}
            Ok(status) => service.push(format!("{} exited with {status}", arguments.join(" "))),
            Err(error) => service.push(format!("{}: {error}", arguments.join(" "))),
        }
    }
    if let Ok(user) = std::env::var("USER") {
        let _ = Command::new("loginctl")
            .args(["enable-linger", &user])
            .status();
    }
    let name = super::hostname();
    println!("flare agent installed at {}", destination.display());
    println!("node name:        {name}");
    println!("node fingerprint: {}", key.public().fingerprint());
    println!("listening on:     {listen}");
    match code {
        Some(code) => println!(
            "join code:        {code}\n                  (valid 24 hours; from the controller: flare orbit pair {name} --code {code})"
        ),
        None => println!("pairing:          accepts the first controller that connects"),
    }
    if service.is_empty() {
        println!(
            "service:          flare-agent enabled and running (journalctl --user -u flare-agent)"
        );
        Ok(())
    } else {
        Err(format!(
            "agent files are installed, but the service could not be started: {}",
            service.join("; ")
        )
        .into())
    }
}

pub fn reset() -> Result<(), Failure> {
    let dir = AgentDir::new(AgentDir::default_root());
    dir.reset()?;
    println!(
        "node {} is unpaired; install the agent again to accept a controller",
        super::hostname()
    );
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::orbit::grant::{ControllerKey, ControllerSet, Member, Package, Source};
    use std::net::TcpListener;

    /// A real agent using an explicit identity directory, private cache, and fabric listener.
    #[test]
    #[ignore = "multi-host hardware fixture; runs until stopped"]
    fn serve_ring_transfer_fixture() {
        let root = PathBuf::from(std::env::var("FLARE_RING_TEST_ROOT").expect("fixture directory"));
        let listen = std::env::var("FLARE_RING_TEST_LISTEN").expect("fixture listen address");
        assert!(root.is_absolute());
        let identity = PathBuf::from(
            std::env::var("FLARE_RING_TEST_AGENT_DIR").expect("agent identity directory"),
        );
        assert!(identity.is_absolute() && identity.join("controllers.toml").is_file());
        let mut agent = Agent::new(AgentDir::new(identity), root.join("cache")).unwrap();
        agent.executable =
            PathBuf::from(std::env::var("FLARE_RING_TEST_EXECUTABLE").expect("serving executable"));
        assert!(agent.executable.is_absolute() && agent.executable.is_file());
        serve_on(TcpListener::bind(listen).unwrap(), Arc::new(agent)).unwrap();
    }

    #[test]
    fn repository_downloads_require_pairing_and_resolve_on_the_node() {
        let harness = harness(73);
        let cache = harness.temp.path().join("hub");
        let commit = "0123456789012345678901234567890123456789";
        let snapshot = cache.join("models--owner--model/snapshots").join(commit);
        fs::create_dir_all(&snapshot).unwrap();
        let primary = snapshot.join("model.safetensors");
        fs::write(&primary, crate::hub::tests::fixture(&[])).unwrap();
        let request = serde_json::to_value(DownloadRequest {
            source: "owner/model".into(),
            options: crate::cli::HubOptions {
                revision: Some(commit.into()),
                offline: true,
                ..Default::default()
            },
            access_token: None,
            profiles: Vec::new(),
        })
        .unwrap();
        let mut unauthorized = connect(&harness, &harness.controller);
        assert!(unauthorized.request(frame::DOWNLOAD, &request).is_err());
        let mut session = pair_accept_first(&harness);
        // Offline acquisition resolves only the Flare cache and its source
        // records; an HF snapshot alone must not satisfy the request.
        let error = session
            .request(frame::DOWNLOAD, &request)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("not available in the Flare cache"),
            "{error}"
        );
        assert_eq!(primary, snapshot.join("model.safetensors"));
        let mut invalid = request;
        invalid["options"]["token_file"] = json!("/node/secret");
        assert!(session.request(frame::DOWNLOAD, &invalid).is_err());
    }

    #[test]
    fn offline_repository_download_resolves_a_real_signed_flare_entry() {
        use flare_helios::package::cache::{self, PackageSource, Root};

        let harness = harness(74);
        let (package, _profile_path, publisher, commit) =
            crate::hub::tests::signed_payload_fixture(harness.temp.path());
        let model_root = harness.temp.path().join("flare");
        let root = Root::open(&model_root).unwrap();
        let metadata = flare_helios::package::reader::Metadata::open(&package).unwrap();
        let selections = crate::package_cache::select_deployments(
            &metadata,
            std::slice::from_ref(&publisher),
            &[1],
        )
        .unwrap();
        let source = PackageSource::open(&package, std::slice::from_ref(&publisher), selections[0])
            .unwrap()
            .union(std::slice::from_ref(&publisher), &selections)
            .unwrap();
        let model_key = cache::model_key("owner/offline").unwrap();
        let expected_envelope = source.verified()[0].envelope_digest();
        let entry = root
            .install_package_staged(source, &model_key, "entry-tp1")
            .unwrap();
        let options = crate::cli::HubOptions {
            revision: Some(commit.clone()),
            model_cache: Some(model_root.clone()),
            cache_dir: Some(harness.temp.path().join("hub")),
            tensor_parallel: vec![1],
            offline: true,
            ..Default::default()
        };
        crate::hub::remember_download(
            &model_root,
            "owner/offline",
            &commit,
            &commit,
            "model.safetensors",
            &options,
            0,
            &entry,
            expected_envelope,
        )
        .unwrap();
        let request = serde_json::to_value(DownloadRequest {
            source: "owner/offline".into(),
            options,
            access_token: None,
            profiles: vec![publisher],
        })
        .unwrap();
        let mut session = pair_accept_first(&harness);
        let reply: Value =
            serde_json::from_slice(&session.request(frame::DOWNLOAD, &request).unwrap()).unwrap();
        assert_eq!(reply["path"], entry.directory().to_str().unwrap());
    }

    #[test]
    fn package_stat_rejects_a_nonregular_primary_before_reading_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let error = stat(&json!({"path": temp.path()})).unwrap_err().to_string();
        assert!(error.contains("must resolve to a regular file"), "{error}");
        assert!(
            error.contains(&temp.path().display().to_string()),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn package_file_sizes_follow_links_and_reject_invalid_targets() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("data.safetensors");
        fs::write(&file, b"payload").unwrap();
        let hardlink = temp.path().join("hardlink.safetensors");
        fs::hard_link(&file, &hardlink).unwrap();
        let link = temp.path().join("link.safetensors");
        symlink("data.safetensors", &link).unwrap();
        for path in [&file, &hardlink, &link] {
            assert_eq!(package_file_size(path).unwrap(), 7);
        }
        fs::remove_file(&link).unwrap();
        symlink(temp.path(), &link).unwrap();
        assert!(
            package_file_size(&link)
                .unwrap_err()
                .to_string()
                .contains("regular file")
        );
        fs::remove_file(&link).unwrap();
        symlink("missing.safetensors", &link).unwrap();
        assert!(package_file_size(&link).is_err());
        fs::remove_file(&link).unwrap();
        symlink("link.safetensors", &link).unwrap();
        assert!(package_file_size(&link).is_err());
    }

    pub type Verified = Arc<Mutex<Vec<(PathBuf, Option<usize>)>>>;

    /// An agent on a free port in a temporary directory, with its controller's key.
    pub struct Harness {
        pub address: String,
        pub agent: Arc<Agent>,
        pub controller: Key,
        pub temp: tempfile::TempDir,
        /// Packages the fake verifier accepted: the primary and the rank verified.
        pub verified: Verified,
    }

    pub fn harness(seed: u8) -> Harness {
        let temp = tempfile::tempdir().unwrap();
        let dir = AgentDir::new(temp.path().join("agent"));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let verified = Arc::new(Mutex::new(Vec::new()));
        let record = verified.clone();
        let mut agent = Agent::with_fakes(
            dir,
            temp.path().join("cache"),
            Box::new(move |primary, rank, publisher| {
                if !primary.is_file() {
                    return Err(format!("{} is not a file", primary.display()).into());
                }
                if publisher.namespace == "reject" {
                    return Err("stored-data content verification failed".into());
                }
                record.lock().unwrap().push((primary.to_owned(), rank));
                Ok(())
            }),
            Box::new(|| Ok(fake_gpus())),
        )
        .unwrap();
        agent.executable = temp.path().join("installed-flare");
        agent.fabric_endpoints = Some(b"loopback 127.0.0.1/8\n".to_vec());
        fs::write(&agent.executable, "#!/bin/sh\necho \"$0\"\necho \"$@\"\n").unwrap();
        fs::set_permissions(&agent.executable, fs::Permissions::from_mode(0o755)).unwrap();
        let agent = Arc::new(agent);
        {
            let agent = agent.clone();
            thread::spawn(move || serve_on(listener, agent));
        }
        Harness {
            address,
            agent,
            controller: Key::from_seed(&[seed; 32]).unwrap(),
            temp,
            verified,
        }
    }

    /// Two GPUs every test agent reports.
    pub fn fake_gpus() -> Vec<Gpu> {
        [
            "GPU-aaaaaaaa-0000-0000-0000-000000000000",
            "GPU-bbbbbbbb-0000-0000-0000-000000000000",
        ]
        .iter()
        .enumerate()
        .map(|(index, uuid)| Gpu {
            index: index as u32,
            uuid: (*uuid).into(),
            name: "Fake GPU".into(),
            revision: "gb10".into(),
            memory_total: 128 << 30,
            memory_free: 100 << 30,
            utilization: Some(7),
            memory_utilization: Some(63),
            temperature: Some(41),
            clock_mhz: None,
        })
        .collect()
    }

    pub fn controller_hello(key: &Key) -> Hello {
        Hello {
            version: PROTOCOL_VERSION,
            kind: "controller".into(),
            name: "test-controller".into(),
            public_key: key.public().to_hex(),
            controllers_revision: 0,
            free_bytes: 0,
            executable: None,
        }
    }

    pub fn connect(harness: &Harness, key: &Key) -> Session {
        let mut session = Session::connect(&harness.address, key).unwrap();
        exchange_hello(&mut session, &controller_hello(key)).unwrap();
        session
    }

    pub fn record(key: &Key) -> Signed {
        Signed::sign(
            key,
            &ControllerSet {
                revision: 1,
                keys: vec![ControllerKey {
                    name: "test-controller".into(),
                    public_key: key.public().to_hex(),
                }],
            },
        )
        .unwrap()
    }

    /// Pair through accept-first and return an authorized session.
    pub fn pair_accept_first(harness: &Harness) -> Session {
        harness.agent.dir.set_accept_first().unwrap();
        let mut session = connect(harness, &harness.controller);
        session
            .request(
                frame::PAIR,
                &json!({"controllers": record(&harness.controller)}),
            )
            .unwrap();
        session
    }

    pub fn put(
        session: &mut Session,
        kind: &str,
        name: &str,
        bytes: &[u8],
        ranges: Option<&[(u64, u64)]>,
    ) {
        let header =
            json!({"kind": kind, "name": name, "size": bytes.len(), "ranges": ranges}).to_string();
        let mut body = (header.len() as u32).to_le_bytes().to_vec();
        body.extend(header.as_bytes());
        for &(offset, nbytes) in ranges.unwrap_or(&[(0, bytes.len() as u64)]) {
            body.extend(&bytes[offset as usize..(offset + nbytes) as usize]);
        }
        session.write_frame(frame::PUT, &body).unwrap();
        session.expect(frame::OK).unwrap();
    }

    fn grant_for(harness: &Harness, rank: usize, source: Option<Source>) -> Signed {
        Signed::sign(
            &harness.controller,
            &Grant {
                group: "test-group".into(),
                name: "test".into(),
                package: Package {
                    export: "export".into(),
                    length: 13,
                    files: vec!["model.safetensors".into()],
                    source,
                },
                members: vec![Member {
                    rank,
                    name: harness.agent.name.clone(),
                    public_key: harness.agent.key.public().to_hex(),
                    address: harness.address.clone(),
                    gpus: vec![],
                }],
                rendezvous: "127.0.0.1:29800".into(),
                serving: vec!["--max-concurrency".into(), "2".into()],
                rank_zero: vec!["--port".into(), "8001".into()],
                roce_ring: false,
                transfer_fabric: None,
                roce_chain: false,
                single_host: false,
                issued: now(),
                expires: now() + 60,
            },
        )
        .unwrap()
    }

    fn publisher() -> Publisher {
        Publisher {
            name: "Test".into(),
            namespace: "test-publisher".into(),
            public_key: [1; 32],
            development: false,
        }
    }

    fn run_request(grant: &Signed, rank: usize, binary: &str) -> Value {
        json!({"grant": grant, "rank": rank, "binary": binary, "publisher": publisher()})
    }

    #[test]
    fn peer_launch_fetches_missing_shards_and_preserves_an_in_place_source() {
        for local_source in [false, true] {
            let harness = harness(84);
            let mut session = pair_accept_first(&harness);
            let path = harness.temp.path().join("model.safetensors");
            let source = local_source.then(|| {
                fs::write(&path, b"package bytes").unwrap();
                Source {
                    public_key: harness.agent.key.public().to_hex(),
                    path: path.display().to_string(),
                }
            });
            let grant = grant_for(&harness, 1, source);
            session
                .request(
                    frame::RUN,
                    &json!({"grant": grant, "rank": 1, "publisher": publisher()}),
                )
                .unwrap();
            let (lines, code) = collect(&mut session);
            assert_eq!(code, 0);
            let command = lines.iter().find(|line| line.starts_with("run ")).unwrap();
            if local_source {
                assert!(command.starts_with(&format!("run {} --package-cache ", path.display())));
            } else {
                assert!(command.starts_with("run --package-cache "));
                assert!(!path.exists());
            }
            assert!(command.contains("--rank 1 --rendezvous 127.0.0.1:29800"));
        }
    }

    pub fn collect(session: &mut Session) -> (Vec<String>, i32) {
        let mut lines = Vec::new();
        loop {
            match session.read_frame().unwrap() {
                (frame::LOG, line) if line.is_empty() => {}
                (frame::LOG, line) => lines.push(String::from_utf8(line).unwrap()),
                (frame::EXIT, code) => {
                    return (lines, i32::from_le_bytes(code.try_into().unwrap()));
                }
                (tag, body) => panic!("unexpected frame {tag}: {}", String::from_utf8_lossy(&body)),
            }
        }
    }

    #[test]
    fn hello_advertises_the_installed_executable_and_detects_agents_without_it() {
        let harness = harness(44);
        let hello = harness.agent.hello().unwrap();
        assert_eq!(
            hello.installed_executable().unwrap(),
            harness.agent.executable
        );
        let mut value = serde_json::to_value(&hello).unwrap();
        value.as_object_mut().unwrap().remove("executable");
        let without_executable: Hello = serde_json::from_value(value).unwrap();
        let error = without_executable
            .installed_executable()
            .unwrap_err()
            .to_string();
        assert!(error.contains("flare agent install"), "{error}");
        assert!(error.contains("--executable PATH"), "{error}");
    }

    #[test]
    fn run_defaults_to_the_installed_executable_only_when_no_override_is_named() {
        for binary in [None, Some(Value::Null)] {
            let harness = harness(45);
            let mut session = pair_accept_first(&harness);
            put(
                &mut session,
                "package",
                "export-13/model.safetensors",
                b"package bytes",
                None,
            );
            let grant = grant_for(&harness, 0, None);
            let mut request = run_request(&grant, 0, "missing-executable");
            let error = session
                .request(frame::RUN, &request)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("executable missing-executable is not cached"),
                "{error}"
            );
            let mut session = connect(&harness, &harness.controller);
            request["binary"] = json!(123);
            assert!(session.request(frame::RUN, &request).is_err());
            let mut session = connect(&harness, &harness.controller);
            if let Some(binary) = binary {
                request["binary"] = binary;
            } else {
                request.as_object_mut().unwrap().remove("binary");
            }
            session.request(frame::RUN, &request).unwrap();
            let (lines, exit) = collect(&mut session);
            assert_eq!(exit, 0);
            assert!(lines.contains(&harness.agent.executable.to_string_lossy().into_owned()));
            assert_eq!(
                fs::read_dir(harness.agent.cache.join("bin"))
                    .unwrap()
                    .count(),
                0
            );
        }
    }

    #[test]
    fn unpaired_nodes_refuse_requests_and_pair_through_their_join_code() {
        let harness = harness(1);
        let mut session = connect(&harness, &harness.controller);
        let refused = session
            .request(frame::STATUS, &json!({}))
            .err()
            .unwrap()
            .to_string();
        assert!(
            refused.contains("not an authorized controller"),
            "{refused}"
        );
        let mut session = connect(&harness, &harness.controller);
        let refused = session
            .request(
                frame::PAIR,
                &json!({"controllers": record(&harness.controller)}),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(refused.contains("accepts no pairing"), "{refused}");
        let mut session = connect(&harness, &harness.controller);
        session
            .write_frame(
                frame::CONTROLLERS,
                &serde_json::to_vec(&record(&harness.controller)).unwrap(),
            )
            .unwrap();
        assert!(
            session
                .expect(frame::OK)
                .err()
                .unwrap()
                .to_string()
                .contains("not an authorized")
        );

        harness
            .agent
            .dir
            .set_join_code("ABCD-EFGH-IJKL-MNOP", 60)
            .unwrap();
        let mut session = connect(&harness, &harness.controller);
        let wrong = hmac_sha256(&[0; 32], session.handshake_hash());
        let refused = session
            .request(
                frame::PAIR,
                &json!({"proof": hex(&wrong), "controllers": record(&harness.controller)}),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(refused.contains("proof rejected"), "{refused}");
        let other = Key::from_seed(&[9; 32]).unwrap();
        let mut session = connect(&harness, &harness.controller);
        assert!(
            session
                .request(frame::PAIR, &json!({"controllers": record(&other)}))
                .is_err()
        );

        let mut session = connect(&harness, &harness.controller);
        let key = crate::orbit::store::join_key("abcd efgh ijkl mnop");
        let proof = hmac_sha256(
            &key,
            &[b"controller".as_slice(), session.handshake_hash()].concat(),
        );
        let reply: Value = serde_json::from_slice(
            &session
                .request(
                    frame::PAIR,
                    &json!({"proof": hex(&proof), "controllers": record(&harness.controller)}),
                )
                .unwrap(),
        )
        .unwrap();
        let expected = hmac_sha256(
            &key,
            &[b"node".as_slice(), session.handshake_hash()].concat(),
        );
        assert_eq!(reply["proof"], hex(&expected));
        assert_eq!(reply["public_key"], harness.agent.key.public().to_hex());
        assert!(harness.agent.dir.join_code().unwrap().is_none());

        let status: Value = serde_json::from_slice(
            &session
                .request(frame::STATUS, &json!({"toward": "127.0.0.1"}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(status["controllers_revision"], 1);
        assert_eq!(status["address_toward"], "127.0.0.1");
        assert!(status["group"].is_null());
        let mut stranger = connect(&harness, &other);
        assert!(stranger.request(frame::STATUS, &json!({})).is_err());
        let mut again = connect(&harness, &harness.controller);
        let refused = again
            .request(
                frame::PAIR,
                &json!({"controllers": record(&harness.controller)}),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(refused.contains("is paired"), "{refused}");
    }

    #[test]
    fn accept_first_pairs_the_first_controller_only() {
        let harness = harness(2);
        pair_accept_first(&harness);
        assert!(!harness.agent.dir.accept_first());
        let other = Key::from_seed(&[8; 32]).unwrap();
        let mut late = connect(&harness, &other);
        assert!(
            late.request(frame::PAIR, &json!({"controllers": record(&other)}))
                .is_err()
        );
    }

    #[test]
    fn controllers_advance_through_signed_records_and_ranks_run_under_grants() {
        let harness = harness(3);
        let mut session = pair_accept_first(&harness);
        let second = Key::from_seed(&[4; 32]).unwrap();
        let revision2 = Signed::sign(
            &harness.controller,
            &ControllerSet {
                revision: 2,
                keys: vec![ControllerKey {
                    name: "second".into(),
                    public_key: second.public().to_hex(),
                }],
            },
        )
        .unwrap();
        session
            .write_frame(frame::CONTROLLERS, &serde_json::to_vec(&revision2).unwrap())
            .unwrap();
        session.expect(frame::OK).unwrap();
        assert!(session.request(frame::STATUS, &json!({})).is_err());
        let removed = Harness {
            address: harness.address.clone(),
            agent: harness.agent.clone(),
            controller: Key::from_seed(&[3; 32]).unwrap(),
            temp: tempfile::tempdir().unwrap(),
            verified: harness.verified.clone(),
        };
        let harness = Harness {
            controller: second,
            ..harness
        };
        let mut session = connect(&harness, &harness.controller);
        session.request(frame::STATUS, &json!({})).unwrap();

        let script = "#!/bin/sh\necho \"$@\"\necho err >&2\nexit 3\n";
        let name = cache::binary_name(script.as_bytes());
        put(&mut session, "binary", &name, script.as_bytes(), None);
        assert_eq!(
            session
                .request(
                    frame::HAVE,
                    &json!({"kind": "binary", "name": name, "size": script.len()})
                )
                .unwrap(),
            [1]
        );
        put(
            &mut session,
            "package",
            "export-13/model.safetensors",
            b"package bytes",
            None,
        );
        let grant = grant_for(&harness, 0, None);
        let wrong_rank = session
            .request(frame::RUN, &run_request(&grant, 1, &name))
            .err()
            .unwrap()
            .to_string();
        assert!(wrong_rank.contains("no rank 1"), "{wrong_rank}");
        let mut session = connect(&harness, &harness.controller);
        let stale = grant_for(&removed, 0, None);
        assert!(
            session
                .request(frame::RUN, &run_request(&stale, 0, &name))
                .is_err()
        );
        let mut session = connect(&harness, &harness.controller);
        let reply: Value = serde_json::from_slice(
            &session
                .request(frame::RUN, &run_request(&grant, 0, &name))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(reply["group"], "test-group");
        let (lines, code) = collect(&mut session);
        assert_eq!(code, 3);
        let command = lines.iter().find(|l| l.starts_with("run ")).unwrap();
        assert!(
            command.contains(
                "--rank 0 --rendezvous 0.0.0.0:29800 \
                 --max-concurrency 2 --port 8001 --publisher-keys"
            ),
            "{command}"
        );
        assert!(command.ends_with(&format!(
            "--dangerously-approve-key {} --dangerously-accept-publisher test-publisher",
            publisher().fingerprint()
        )));
        assert!(lines.contains(&"err".to_owned()));

        let mut session = connect(&harness, &harness.controller);
        session
            .request(frame::ATTACH, &json!({"group": "test-group"}))
            .unwrap();
        assert_eq!(collect(&mut session), (lines, 3));
        let mut session = connect(&harness, &harness.controller);
        let status: Value =
            serde_json::from_slice(&session.request(frame::STATUS, &json!({})).unwrap()).unwrap();
        assert_eq!(status["group"]["running"], false);
        assert_eq!(status["group"]["exit"], 3);
        let stopped: Value = serde_json::from_slice(
            &session
                .request(frame::STOP, &json!({"group": "test-group"}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stopped["stopped"], false);
    }

    #[test]
    fn running_groups_are_stopped_and_outlive_their_controller_session() {
        let harness = harness(5);
        let mut session = pair_accept_first(&harness);
        let script = "#!/bin/sh\necho started\ntrap 'echo stopped; exit 7' TERM\nwhile true; do sleep 0.1; done\n";
        let name = cache::binary_name(script.as_bytes());
        put(&mut session, "binary", &name, script.as_bytes(), None);
        put(
            &mut session,
            "package",
            "export-13-rank1/model.safetensors",
            b"package bytes",
            Some(&[(0, 5), (8, 5)]),
        );
        let grant = grant_for(&harness, 1, None);
        session
            .request(frame::RUN, &run_request(&grant, 1, &name))
            .unwrap();
        let first = loop {
            let (tag, body) = session.read_frame().unwrap();
            if tag == frame::LOG && !body.is_empty() {
                break String::from_utf8(body).unwrap();
            }
        };
        assert_eq!(first, "started");
        drop(session);
        thread::sleep(Duration::from_millis(300));
        let mut session = connect(&harness, &harness.controller);
        let status: Value =
            serde_json::from_slice(&session.request(frame::STATUS, &json!({})).unwrap()).unwrap();
        assert_eq!(status["group"]["running"], true);
        let busy = session
            .request(
                frame::RUN,
                &run_request(&grant_for(&harness, 1, None), 1, &name),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(busy.contains("runs on this node"), "{busy}");
        let mut attached = connect(&harness, &harness.controller);
        attached.request(frame::ATTACH, &json!({})).unwrap();
        let mut session = connect(&harness, &harness.controller);
        let stopped: Value =
            serde_json::from_slice(&session.request(frame::STOP, &json!({})).unwrap()).unwrap();
        assert_eq!(stopped["stopped"], true);
        let (lines, code) = collect(&mut attached);
        assert_eq!(code, 7);
        assert!(
            lines.contains(&"started".to_owned()) && lines.contains(&"stopped".to_owned()),
            "{lines:?}"
        );
        // The group's pins end with it, so a later package replaces it.
        put(
            &mut session,
            "package",
            "other-1/model.safetensors",
            b"x",
            None,
        );
        assert!(
            !harness
                .agent
                .cache
                .join("packages/export-13-rank1")
                .exists()
        );
    }

    fn grant_on_gpus(harness: &Harness, group: &str, gpus: &[&str]) -> Signed {
        Signed::sign(
            &harness.controller,
            &Grant {
                group: group.into(),
                name: group.into(),
                package: Package {
                    export: "export".into(),
                    length: 13,
                    files: vec!["model.safetensors".into()],
                    source: None,
                },
                members: vec![Member {
                    rank: 0,
                    name: harness.agent.name.clone(),
                    public_key: harness.agent.key.public().to_hex(),
                    address: harness.address.clone(),
                    gpus: gpus.iter().map(|g| (*g).to_owned()).collect(),
                }],
                rendezvous: "127.0.0.1:29800".into(),
                serving: vec![],
                rank_zero: vec![],
                roce_ring: false,
                transfer_fabric: None,
                roce_chain: false,
                single_host: false,
                issued: now(),
                expires: now() + 60,
            },
        )
        .unwrap()
    }

    #[test]
    fn groups_run_together_on_disjoint_gpus_and_status_lists_them() {
        let harness = harness(8);
        let mut session = pair_accept_first(&harness);
        let script = "#!/bin/sh\necho \"gpus=$CUDA_VISIBLE_DEVICES args=$*\"\ntrap 'exit 0' TERM\nwhile true; do sleep 0.1; done\n";
        let name = cache::binary_name(script.as_bytes());
        put(&mut session, "binary", &name, script.as_bytes(), None);
        put(
            &mut session,
            "package",
            "export-13/model.safetensors",
            b"package bytes",
            None,
        );
        let [a, b] = [
            "GPU-aaaaaaaa-0000-0000-0000-000000000000",
            "GPU-bbbbbbbb-0000-0000-0000-000000000000",
        ];
        let first = grant_on_gpus(&harness, "first", &[a]);
        let mut running_first = connect(&harness, &harness.controller);
        running_first
            .request(frame::RUN, &run_request(&first, 0, &name))
            .unwrap();
        let line = loop {
            let (tag, body) = running_first.read_frame().unwrap();
            if tag == frame::LOG && !body.is_empty() {
                break String::from_utf8(body).unwrap();
            }
        };
        assert!(
            line.starts_with(&format!("gpus={a} args=run ")) && line.ends_with("--gpus 0"),
            "{line}"
        );
        // A group on the held GPU, or on an unknown GPU, or on no GPU in particular, is refused.
        let mut session = connect(&harness, &harness.controller);
        let held = session
            .request(
                frame::RUN,
                &run_request(&grant_on_gpus(&harness, "clash", &[a]), 0, &name),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(held.contains("holds"), "{held}");
        let mut session = connect(&harness, &harness.controller);
        let unknown = session
            .request(
                frame::RUN,
                &run_request(
                    &grant_on_gpus(&harness, "where", &["GPU-cccccccc"]),
                    0,
                    &name,
                ),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(unknown.contains("no GPU"), "{unknown}");
        let mut session = connect(&harness, &harness.controller);
        assert!(
            session
                .request(
                    frame::RUN,
                    &run_request(&grant_on_gpus(&harness, "all", &[]), 0, &name)
                )
                .is_err()
        );
        // A group on the other GPU runs beside the first.
        let mut running_second = connect(&harness, &harness.controller);
        running_second
            .request(
                frame::RUN,
                &run_request(&grant_on_gpus(&harness, "second", &[b]), 0, &name),
            )
            .unwrap();
        let mut session = connect(&harness, &harness.controller);
        let status: Value =
            serde_json::from_slice(&session.request(frame::STATUS, &json!({})).unwrap()).unwrap();
        let groups = status["groups"].as_array().unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|g| g["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert!(groups.iter().all(|g| g["running"] == true));
        let gpus = status["gpus"].as_array().unwrap();
        assert_eq!(gpus[0]["group"], "first");
        assert_eq!(gpus[1]["group"], "second");
        assert_eq!(gpus[1]["revision"], "gb10");
        let packages = status["packages"].as_array().unwrap();
        assert_eq!(packages[0]["name"], "export-13");
        assert_eq!(packages[0]["verified"], false);
        assert_eq!(packages[0]["files"][0]["size"], 13);
        // With several groups, STOP and ATTACH must name one.
        let ambiguous = session
            .request(frame::STOP, &json!({}))
            .err()
            .unwrap()
            .to_string();
        assert!(ambiguous.contains("several groups"), "{ambiguous}");
        for group in ["first", "second"] {
            let mut session = connect(&harness, &harness.controller);
            let stopped: Value = serde_json::from_slice(
                &session
                    .request(frame::STOP, &json!({"group": group}))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(stopped["stopped"], true);
        }
        assert_eq!(collect(&mut running_first).1, 0);
        assert_eq!(collect(&mut running_second).1, 0);
        let mut session = connect(&harness, &harness.controller);
        let status: Value =
            serde_json::from_slice(&session.request(frame::STATUS, &json!({})).unwrap()).unwrap();
        assert!(
            status["gpus"]
                .as_array()
                .unwrap()
                .iter()
                .all(|g| g["group"].is_null())
        );
    }

    fn history_group(id: &str, started: Instant) -> Arc<Group> {
        Arc::new(Group {
            id: id.into(),
            name: "history-test".into(),
            rank: 0,
            package: "model.safetensors".into(),
            gpus: Vec::new(),
            pid: 0,
            metrics_url: None,
            started,
            log: Mutex::new(Log {
                lines: VecDeque::new(),
                bytes: 0,
                exit: None,
                subscribers: Vec::new(),
                pins: None,
            }),
            metrics: Mutex::new(None),
        })
    }

    #[test]
    fn history_retention_starts_at_exit_and_preserves_running_groups() {
        let started = Instant::now();
        let finished = started + SUCCESS_RETENTION * 2;
        let success = history_group("success", started);
        success.push("completed".into());
        success.log.lock().unwrap().exit = Some((0, finished));
        let failure = history_group("failure", started);
        failure.log.lock().unwrap().exit = Some((1, finished));
        let running = history_group("running", started);
        let mut groups = vec![success.clone(), failure, running];

        prune_groups(
            &mut groups,
            finished + SUCCESS_RETENTION - Duration::from_nanos(1),
        );
        assert_eq!(groups.len(), 3);
        prune_groups(&mut groups, finished + SUCCESS_RETENTION);
        assert_eq!(
            groups.iter().map(|g| g.id.as_str()).collect::<Vec<_>>(),
            ["failure", "running"]
        );
        // An attached session owns its group independently of the agent's history.
        let (lines, exit, _) = success.subscribe();
        assert_eq!(lines, ["completed"]);
        assert_eq!(exit, Some(0));
        prune_groups(&mut groups, finished + Duration::from_secs(365 * 86400));
        assert_eq!(
            groups.iter().map(|g| g.id.as_str()).collect::<Vec<_>>(),
            ["failure", "running"]
        );
    }

    #[test]
    fn completed_groups_are_count_bounded_without_another_launch() {
        let now = Instant::now();
        let mut groups = (0..RETAINED_GROUPS + 3)
            .map(|i| history_group(&format!("group-{i}"), now))
            .collect::<Vec<_>>();
        // No count limit applies while all the groups are running.
        prune_groups(&mut groups, now + SUCCESS_RETENTION);
        assert_eq!(groups.len(), RETAINED_GROUPS + 3);
        // The first launch remains active while every later launch finishes.
        for group in groups.iter().skip(1) {
            group.finish(0);
        }
        prune_groups(&mut groups, Instant::now());
        assert_eq!(groups.len(), RETAINED_GROUPS + 1);
        assert_eq!(groups[0].id, "group-0");
        assert!(groups[0].running());
        assert_eq!(groups[1].id, "group-3");
        for i in 0..RETAINED_GROUPS + 3 {
            let failure = history_group(&format!("failure-{i}"), now);
            failure.finish(1);
            groups.push(failure);
        }
        prune_groups(&mut groups, Instant::now());
        assert_eq!(
            groups
                .iter()
                .filter(|group| group.exit() == Some(1))
                .count(),
            RETAINED_GROUPS + 3
        );
        assert_eq!(
            groups
                .iter()
                .filter(|group| group.exit() == Some(0))
                .count(),
            RETAINED_GROUPS
        );
    }

    #[test]
    fn removing_history_requires_a_controller_and_preserves_live_groups() {
        let harness = harness(28);
        let now = Instant::now();
        let failed = history_group("failed", now);
        failed.push("failure detail".into());
        failed.finish(7);
        harness
            .agent
            .groups
            .lock()
            .unwrap()
            .extend([failed.clone(), history_group("live", now)]);
        // An unpaired controller cannot remove history.
        let mut stranger = connect(&harness, &harness.controller);
        assert!(
            stranger
                .request(frame::REMOVE, &json!({"group": "failed"}))
                .is_err()
        );
        let mut session = pair_accept_first(&harness);
        assert!(
            session
                .request(frame::REMOVE, &json!({"group": "live"}))
                .is_err()
        );
        assert_eq!(harness.agent.groups().len(), 2);
        let mut session = connect(&harness, &harness.controller);
        let removed: Value = serde_json::from_slice(
            &session
                .request(frame::REMOVE, &json!({"group": "failed"}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(removed["removed"], 1);
        assert_eq!(harness.agent.groups()[0].id, "live");
        assert_eq!(failed.subscribe().0, ["failure detail"]);
        // Retrying after another controller removed the same record is harmless.
        let removed: Value = serde_json::from_slice(
            &session
                .request(frame::REMOVE, &json!({"group": "failed"}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(removed["removed"], 0);
    }

    #[test]
    fn sampler_and_queries_remove_expired_history() {
        let harness = harness(27);
        let now = Instant::now();
        let expired = history_group("expired", now - SUCCESS_RETENTION * 2);
        expired.log.lock().unwrap().exit = Some((0, now - SUCCESS_RETENTION));
        harness.agent.groups.lock().unwrap().push(expired.clone());
        harness.agent.sample();
        assert!(harness.agent.groups.lock().unwrap().is_empty());

        harness.agent.groups.lock().unwrap().push(expired.clone());
        assert!(
            harness.agent.metrics()["groups"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        harness.agent.groups.lock().unwrap().push(expired.clone());
        let state: Value =
            serde_json::from_slice(&status(&harness.agent, &json!({})).unwrap()).unwrap();
        assert!(state["groups"].as_array().unwrap().is_empty());
        assert!(state["group"].is_null());
        harness.agent.groups.lock().unwrap().push(expired);
        assert!(harness.agent.find_group(Some("expired")).is_err());
    }

    #[test]
    fn metrics_urls_follow_rank_zero_bind_options() {
        assert_eq!(
            metrics_url(&[]),
            "http://127.0.0.1:8000/metrics?format=json"
        );
        assert_eq!(
            metrics_url(&[
                "--host".into(),
                "0.0.0.0".into(),
                "--port".into(),
                "8001".into()
            ]),
            "http://127.0.0.1:8001/metrics?format=json"
        );
        assert_eq!(
            metrics_url(&["--host".into(), "10.1.2.3".into()]),
            "http://10.1.2.3:8000/metrics?format=json"
        );
        assert_eq!(
            metrics_url(&["--host".into(), "::1".into(), "--port".into(), "9".into()]),
            "http://[::1]:9/metrics?format=json"
        );
    }

    #[test]
    fn metrics_urls_honor_bind_overrides() {
        let cases: &[(&[&str], &str)] = &[
            (
                &[
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8001",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
                "http://127.0.0.1:8000/metrics?format=json",
            ),
            (
                &[
                    "--host",
                    "10.1.2.3",
                    "--port",
                    "8001",
                    "--host=::1",
                    "--port=9000",
                ],
                "http://[::1]:9000/metrics?format=json",
            ),
            (
                &[
                    "--host=10.1.2.3",
                    "--port=8001",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
                "http://127.0.0.1:8000/metrics?format=json",
            ),
            (
                &["--host=::", "--port=9000"],
                "http://127.0.0.1:9000/metrics?format=json",
            ),
        ];
        for (options, expected) in cases {
            let options = options.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            assert_eq!(metrics_url(&options), *expected, "{options:?}");
        }
    }

    #[test]
    fn the_sampler_scrapes_rank_zero_metrics_and_reports_gpus_with_holders() {
        let harness = harness(9);
        let mut session = pair_accept_first(&harness);
        // A stand-in for rank zero's HTTP server: every request gets one JSON document.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buffer = [0u8; 1024];
                let _ = std::io::Read::read(&mut stream, &mut buffer);
                let body = r#"{"generated_tokens":1234,"active":2}"#;
                let _ = std::io::Write::write_all(
                    &mut stream,
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            }
        });
        let script = "#!/bin/sh\ntrap 'exit 0' TERM\nwhile true; do sleep 0.1; done\n";
        let name = cache::binary_name(script.as_bytes());
        put(&mut session, "binary", &name, script.as_bytes(), None);
        put(
            &mut session,
            "package",
            "export-13/model.safetensors",
            b"package bytes",
            None,
        );
        let a = "GPU-aaaaaaaa-0000-0000-0000-000000000000";
        let grant = Signed::sign(
            &harness.controller,
            &Grant {
                group: "serving".into(),
                name: "serving".into(),
                package: Package {
                    export: "export".into(),
                    length: 13,
                    files: vec!["model.safetensors".into()],
                    source: None,
                },
                members: vec![Member {
                    rank: 0,
                    name: harness.agent.name.clone(),
                    public_key: harness.agent.key.public().to_hex(),
                    address: harness.address.clone(),
                    gpus: vec![a.into()],
                }],
                rendezvous: "127.0.0.1:29800".into(),
                serving: vec![],
                rank_zero: vec![
                    "--host".into(),
                    "127.0.0.2".into(),
                    "--port".into(),
                    "0".into(),
                    "--host".into(),
                    "0.0.0.0".into(),
                    "--port".into(),
                    port.to_string(),
                ],
                roce_ring: false,
                transfer_fabric: None,
                roce_chain: false,
                single_host: false,
                issued: now(),
                expires: now() + 60,
            },
        )
        .unwrap();
        let mut running = connect(&harness, &harness.controller);
        running
            .request(frame::RUN, &run_request(&grant, 0, &name))
            .unwrap();
        harness.agent.sample();
        let mut session = connect(&harness, &harness.controller);
        let metrics: Value =
            serde_json::from_slice(&session.request(frame::METRICS, &json!({})).unwrap()).unwrap();
        assert!(metrics["at"].as_u64().unwrap() > 0);
        let gpus = metrics["gpus"].as_array().unwrap();
        assert_eq!(gpus[0]["utilization"], 7);
        assert_eq!(gpus[0]["group"], "serving");
        assert!(gpus[1]["group"].is_null());
        let group = &metrics["groups"][0];
        assert_eq!(group["name"], "serving");
        assert_eq!(group["metrics"]["generated_tokens"], 1234);
        assert!(group["metrics_at"].as_u64().unwrap() > 0);
        let mut session = connect(&harness, &harness.controller);
        session
            .request(frame::STOP, &json!({"group": "serving"}))
            .unwrap();
        assert_eq!(collect(&mut running).1, 0);
    }

    #[test]
    fn fabric_transfers_stream_ranges_directly_or_through_relays() {
        for ring in [false, true] {
            use crate::orbit::{fabric, store::Node};
            let hosts = (80..86)
                .map(|seed| Harness {
                    controller: Key::from_seed(&[80; 32]).unwrap(),
                    ..harness(seed)
                })
                .collect::<Vec<_>>();
            for host in &hosts {
                pair_accept_first(host);
            }
            let nodes = hosts
                .iter()
                .enumerate()
                .map(|(rank, h)| Node {
                    name: format!("rank{rank}"),
                    address: h.address.clone(),
                    public_key: h.agent.key.public().to_hex(),
                    fingerprint: h.agent.key.public().fingerprint(),
                })
                .collect::<Vec<_>>();
            let records = hosts
                .iter()
                .map(|h| {
                    connect(h, &h.controller)
                        .request(frame::FABRIC, &json!({}))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let bytes = (0..4_000_019).map(|i| (i % 251) as u8).collect::<Vec<_>>();
            let package = hosts[0].temp.path().join("model.safetensors");
            fs::write(&package, &bytes).unwrap();
            let mut grant = Grant {
                group: "ring".into(),
                name: "ring".into(),
                package: Package {
                    export: "ring".into(),
                    length: bytes.len() as u64,
                    files: vec!["model.safetensors".into()],
                    source: Some(Source {
                        public_key: nodes[0].public_key.clone(),
                        path: package.display().to_string(),
                    }),
                },
                members: nodes
                    .iter()
                    .enumerate()
                    .map(|(rank, node)| Member {
                        rank,
                        name: node.name.clone(),
                        public_key: node.public_key.clone(),
                        address: node.address.clone(),
                        gpus: vec![],
                    })
                    .collect(),
                rendezvous: String::new(),
                serving: vec![],
                rank_zero: vec![],
                roce_ring: ring,
                roce_chain: false,
                transfer_fabric: Some(fabric::plan(&nodes, &records, ring, false).unwrap()),
                single_host: false,
                issued: now(),
                expires: now() + 60,
            };
            let ranges = [(0u64, 100_001u64), (1_000_000, 2_999_999)];
            let request = |grant: &Grant, rank: usize| {
                json!({
                    "grant": Signed::sign(&hosts[0].controller, grant).unwrap(), "publisher": publisher(),
                    "source": {"public_key": nodes[0].public_key, "address": "127.0.0.1:1"},
                    "directory": format!("ring-{}-rank{rank}", bytes.len()),
                    "streams": ([0, 4, 8, 16, 32, 64][rank]),
                    "files": [{"name": "model.safetensors", "size": bytes.len(), "ranges": ranges}],
                })
            };
            for count in [json!(0), json!(65), json!("8")] {
                let mut invalid = request(&grant, 1);
                invalid["streams"] = count;
                let error = connect(&hosts[1], &hosts[1].controller)
                    .request(frame::TRANSFER, &invalid)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("transfer streams must be in 1..=64")
                );
            }
            // The unusable management address cannot carry payloads in either topology.
            thread::scope(|scope| {
                for rank in 1..hosts.len() {
                    let host = &hosts[rank];
                    let request = request(&grant, rank);
                    scope.spawn(move || {
                        let reply: Value = serde_json::from_slice(
                            &connect(host, &host.controller)
                                .request(frame::TRANSFER, &request)
                                .unwrap(),
                        )
                        .unwrap();
                        assert_eq!(reply["received"], cache::range_bytes(&ranges));
                        assert_eq!(reply["verified"], true);
                    });
                }
            });
            for rank in 1..hosts.len() {
                let directory = hosts[rank]
                    .agent
                    .cache
                    .join(format!("packages/ring-{}-rank{rank}", bytes.len()));
                let copy = fs::read(directory.join("model.safetensors")).unwrap();
                assert_eq!(copy.len(), bytes.len());
                for (offset, count) in ranges {
                    let span = offset as usize..(offset + count) as usize;
                    assert_eq!(copy[span.clone()], bytes[span]);
                }
                assert!(copy[100_001..1_000_000].iter().all(|&b| b == 0));
                assert!(directory.join(VERIFIED).exists());
                assert_eq!(hosts[rank].verified.lock().unwrap().len(), 1);
            }
            assert!(
                !hosts[1]
                    .agent
                    .cache
                    .join(format!("packages/ring-{}-rank2", bytes.len()))
                    .exists()
            );

            if !ring {
                continue;
            }

            // A node with the wrong identity cannot relay a member's request, even with its grant.
            let fetch = json!({"grant": Signed::sign(&hosts[0].controller, &grant).unwrap(),
            "source_rank": 0, "destination_rank": 2, "rail": 0,
            "name": "model.safetensors", "size": bytes.len(), "ranges": ranges});
            let mut stranger = connect(&hosts[1], &hosts[3].agent.key);
            assert!(
                stranger
                    .request(frame::FETCH_FABRIC, &fetch)
                    .unwrap_err()
                    .to_string()
                    .contains("next member")
            );

            // The right identity on an unsigned interface is rejected before forwarding.
            grant.transfer_fabric.as_mut().unwrap().links[1][2][0]
                .to
                .set_ip("127.0.0.2".parse().unwrap());
            grant.transfer_fabric.as_mut().unwrap().links[2][1][0]
                .from
                .set_ip("127.0.0.2".parse().unwrap());
            let mut fetch = fetch;
            fetch["grant"] =
                serde_json::to_value(Signed::sign(&hosts[0].controller, &grant).unwrap()).unwrap();
            let mut wrong_link = connect(&hosts[1], &hosts[2].agent.key);
            assert!(
                wrong_link
                    .request(frame::FETCH_FABRIC, &fetch)
                    .unwrap_err()
                    .to_string()
                    .contains("fabric interfaces")
            );

            // A dead next hop never falls back to the management address. Cache hits need no hop.
            grant.transfer_fabric = Some(fabric::plan(&nodes, &records, ring, false).unwrap());
            let closed = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = closed.local_addr().unwrap().port();
            drop(closed);
            grant.transfer_fabric.as_mut().unwrap().links[1][2][0]
                .from
                .set_port(port);
            grant.transfer_fabric.as_mut().unwrap().links[2][1][0]
                .to
                .set_port(port);
            let cached = request(&grant, 2);
            let reply: Value = serde_json::from_slice(
                &connect(&hosts[2], &hosts[2].controller)
                    .request(frame::TRANSFER, &cached)
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(reply["received"], 0);
            assert_eq!(reply["verified"], false);
            let mut cold = cached;
            cold["directory"] = "cold-ring-rank2".into();
            assert!(
                connect(&hosts[2], &hosts[2].controller)
                    .request(frame::TRANSFER, &cold)
                    .is_err()
            );
            assert!(
                !hosts[2]
                    .agent
                    .cache
                    .join("packages/cold-ring-rank2/.verified")
                    .exists()
            );
        }
    }

    #[test]
    fn nodes_fetch_package_ranges_from_a_source_node_under_a_grant() {
        let source = harness(6);
        let peer = Harness {
            controller: Key::from_seed(&[6; 32]).unwrap(),
            ..harness(7)
        };
        pair_accept_first(&source);
        pair_accept_first(&peer);
        let package = source.temp.path().join("held/model.safetensors");
        fs::create_dir_all(package.parent().unwrap()).unwrap();
        let bytes: Vec<u8> = (0..100_000u32).map(|i| (i % 253) as u8).collect();
        fs::write(&package, &bytes).unwrap();
        let grant = Signed::sign(
            &source.controller,
            &Grant {
                group: "g".into(),
                name: "g".into(),
                package: Package {
                    export: "export".into(),
                    length: bytes.len() as u64,
                    files: vec!["model.safetensors".into()],
                    source: Some(Source {
                        public_key: source.agent.key.public().to_hex(),
                        path: package.display().to_string(),
                    }),
                },
                members: vec![
                    Member {
                        rank: 0,
                        name: "source".into(),
                        public_key: source.agent.key.public().to_hex(),
                        address: source.address.clone(),
                        gpus: vec![],
                    },
                    Member {
                        rank: 1,
                        name: "peer".into(),
                        public_key: peer.agent.key.public().to_hex(),
                        address: peer.address.clone(),
                        gpus: vec![],
                    },
                ],
                rendezvous: "127.0.0.1:29800".into(),
                serving: vec![],
                rank_zero: vec![],
                roce_ring: false,
                transfer_fabric: None,
                roce_chain: false,
                single_host: false,
                issued: now(),
                expires: now() + 60,
            },
        )
        .unwrap();
        let ranges = [(0u64, 4096u64), (50_000, 10_000)];
        let mut session = connect(&peer, &peer.controller);
        let reply: Value = serde_json::from_slice(
            &session
                .request(
                    frame::TRANSFER,
                    &json!({
                        "grant": grant,
                        "publisher": publisher(),
                        "source": {"public_key": source.agent.key.public().to_hex(), "address": source.address},
                        "directory": "export-100000-rank1",
                        "files": [{"name": "model.safetensors", "size": bytes.len(), "ranges": ranges}],
                    }),
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(reply["received"], 14096);
        assert_eq!(reply["verified"], true);
        let marker = peer
            .agent
            .cache
            .join("packages/export-100000-rank1/.verified");
        assert!(marker.is_file());
        assert_eq!(
            peer.verified.lock().unwrap().as_slice(),
            [(marker.with_file_name("model.safetensors"), Some(1))]
        );
        // A repeated transfer of held files neither fetches nor verifies again.
        let request = json!({
            "grant": grant,
            "publisher": publisher(),
            "source": {"public_key": source.agent.key.public().to_hex(), "address": source.address},
            "directory": "export-100000-rank1",
            "files": [{"name": "model.safetensors", "size": bytes.len(), "ranges": ranges}],
        });
        let again: Value =
            serde_json::from_slice(&session.request(frame::TRANSFER, &request).unwrap()).unwrap();
        assert_eq!(
            (again["received"].as_u64(), again["verified"].as_bool()),
            (Some(0), Some(false))
        );
        assert_eq!(peer.verified.lock().unwrap().len(), 1);
        // Bytes that fail verification are removed with their directory.
        let mut rejecting = request.clone();
        rejecting["directory"] = "export-100000-rank2".into();
        rejecting["publisher"] = serde_json::to_value(Publisher {
            namespace: "reject".into(),
            ..publisher()
        })
        .unwrap();
        let mut session = connect(&peer, &peer.controller);
        let error = session
            .request(frame::TRANSFER, &rejecting)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("failed verification and was removed"),
            "{error}"
        );
        assert!(
            !peer
                .agent
                .cache
                .join("packages/export-100000-rank2")
                .exists()
        );
        let copy = fs::read(
            peer.agent
                .cache
                .join("packages/export-100000-rank1/model.safetensors"),
        )
        .unwrap();
        assert_eq!(copy.len(), bytes.len());
        for &(offset, nbytes) in &ranges {
            let span = offset as usize..(offset + nbytes) as usize;
            assert_eq!(copy[span.clone()], bytes[span]);
        }
        assert!(copy[4096..50_000].iter().all(|&b| b == 0));

        let stranger = Key::from_seed(&[9; 32]).unwrap();
        let mut session = Session::connect(&source.address, &stranger).unwrap();
        exchange_hello(&mut session, &controller_hello(&stranger)).unwrap();
        session
            .write_frame(
                frame::FETCH,
                &json!({"grant": grant, "name": "model.safetensors", "ranges": [[0, 10]]})
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();
        assert!(
            session
                .expect(frame::OK)
                .err()
                .unwrap()
                .to_string()
                .contains("not a member")
        );
        let mut session = Session::connect(&source.address, &peer.agent.key).unwrap();
        exchange_hello(&mut session, &peer.agent.hello().unwrap()).unwrap();
        session
            .write_frame(
                frame::FETCH,
                &json!({"grant": grant, "name": "other", "ranges": [[0, 10]]})
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();
        assert!(session.expect(frame::OK).is_err());

        let mut session = connect(&source, &source.controller);
        session
            .write_frame(
                frame::GET,
                &json!({"path": package, "offset": 50_000, "nbytes": 16})
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();
        let (tag, length) = session.read_header().unwrap();
        assert_eq!((tag, length), (frame::OK, 16));
        assert_eq!(session.read_body(16, 16).unwrap(), bytes[50_000..50_016]);
        assert!(
            session
                .request(frame::STAT, &json!({"path": "relative"}))
                .is_err()
        );
    }
}
