//! The controller side of orbit control: pairing nodes, distributing the controller set, and
//! launching, attaching to, and stopping serving groups. See docs/orbit.md.
//!
//! A launch verifies the package at the controller, where the operator accepts its publisher;
//! checks node capacity; signs a launch grant; deploys an explicit executable override; and
//! stages the complete source on rank zero when needed. RUN starts native serving ranks,
//! which distribute peer shards and establish their collective transport. The controller
//! prints each rank's output with a `[rank R name]` prefix.
//!
//! Stopping sends STOP to rank zero and waits for it to exit, so its coordinated shutdown drains
//! while every peer still participates in collectives. Peers still running 30 seconds after rank
//! zero exits receive STOP.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::net::{IpAddr, SocketAddr};
use std::os::unix::fs::{DirBuilderExt, FileExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use flare_helios::runtime_package::{self, AuthorizedPackage, Publisher};
use flare_helios::startup::StartupProgress;
use serde::Deserialize;
use serde_json::{Value, json};

use super::cache;
use super::grant::{Grant, Member, Package, Signed, Source};
#[cfg(test)]
use super::identity::Key;
use super::plan;
use super::session::{FRAME_LIMIT, Session};
use super::store::Orbit;
use super::store::{GroupSpec, Node};
use super::{Failure, frame, hex, now, random};

/// Rank zero's rendezvous port, on the address its host routes toward its peers.
pub const RENDEZVOUS_PORT: u16 = 29800;
/// How long a grant authorizes RUN after it is issued.
pub const GRANT_SECONDS: u64 = 600;
/// How long peers may keep running after rank zero exits before they receive STOP.
const PEER_GRACE: Duration = Duration::from_secs(30);
/// An agent sends an empty LOG frame after 20 silent seconds, so an attachment silent for longer
/// has lost its node.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(90);
/// The largest safetensors header Helios accepts in a package file.
const HEADER_LIMIT: u64 = 64 << 10;
const GIB: f64 = (1u64 << 30) as f64;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

/// Record SIGINT and SIGTERM in [`INTERRUPTED`] instead of terminating, so that waiting loops
/// can stop a group in order.
fn catch_interrupts() {
    INTERRUPTED.store(false, Ordering::Relaxed);
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, interrupt as *const () as libc::sighandler_t);
    }
}

fn interrupted() -> bool {
    INTERRUPTED.swap(false, Ordering::Relaxed)
}

/// What STATUS reports about a node.
#[derive(Clone, Debug, Deserialize)]
pub struct NodeStatus {
    pub fingerprint: String,
    pub free_bytes: u64,
    pub controllers_revision: u64,
    /// The group the node started most recently.
    pub group: Option<GroupStatus>,
    /// Every group the node runs or retains.
    #[serde(default)]
    pub groups: Vec<GroupStatus>,
    #[serde(default)]
    pub gpus: Vec<GpuStatus>,
    #[serde(default)]
    pub gpus_error: Option<String>,
    #[serde(default)]
    pub packages: Vec<PackageStatus>,
    /// The local address the node routes toward the address a request named as `toward`.
    #[serde(default)]
    pub address_toward: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GroupStatus {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub rank: usize,
    #[serde(default)]
    pub gpus: Vec<String>,
    pub running: bool,
    pub exit: Option<i32>,
    pub uptime_seconds: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GpuStatus {
    pub index: u32,
    pub uuid: String,
    pub name: String,
    pub revision: String,
    pub memory_total: u64,
    pub memory_free: u64,
    /// The running group holding the GPU.
    pub group: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PackageStatus {
    pub name: String,
    pub files: Vec<PackageFileStatus>,
    pub verified: bool,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
pub struct PackageFileStatus {
    pub name: String,
    pub size: u64,
}

impl NodeStatus {
    pub fn running(&self) -> impl Iterator<Item = &GroupStatus> {
        self.groups.iter().filter(|g| g.running)
    }
}

/// Options of `flare orbit run`.
pub struct RunOptions {
    /// The operator's name for the group, and the orbit file's `[[group]]` of that name when
    /// one exists; the cluster's name or the package's stem when absent.
    pub name: Option<String>,
    /// Save the effective launch configuration as a persistent named group.
    pub persistent: bool,
    /// Candidate nodes in rank order; empty leaves them to the cluster, the group, or the pool.
    pub nodes: Vec<String>,
    /// NVML indices of the GPUs each member uses on its node; empty lets placement pick.
    pub gpus: Vec<u32>,
    /// Print the placement and exit without starting anything.
    pub dry_run: bool,
    /// Serving options every rank loads with.
    pub serving: Vec<String>,
    /// Options for rank zero alone: the HTTP bind and request handling.
    pub rank_zero: Vec<String>,
    pub roce_ring: bool,
    pub transfer_streams: Option<u8>,
    pub roce_chain: bool,
    /// An executable to upload; absent means each node uses its installed agent executable.
    pub executable: Option<PathBuf>,
    /// Return once every rank has started instead of following the group.
    pub detach: bool,
    /// Every rank verifies the package's stored data before loading.
    pub verify_weights: bool,
    /// Package verification and publisher acceptance settings, as `flare run` takes them.
    pub run_args: crate::cli::Run,
}

/// The launch settings a grant carries.
#[derive(Clone, Debug, Default)]
pub struct LaunchOptions {
    pub name: String,
    /// The GPU UUIDs each member uses on its node, in rank order; an empty list for a member
    /// leaves the GPUs to the rank.
    pub placements: Vec<Vec<String>>,
    pub serving: Vec<String>,
    pub rank_zero: Vec<String>,
    pub roce_ring: bool,
    pub transfer_streams: Option<u8>,
    /// Place and verify the package on the members without sending the executable or
    /// starting ranks; the returned launch then has no rank sessions.
    pub stage_only: bool,
    /// Serve over a chain in rank order (exclusive with roce_ring).
    pub roce_chain: bool,
}

/// A `flare orbit run` package argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackageReference {
    Local(PathBuf),
    /// A package a orbit node holds at an absolute path.
    Node {
        node: String,
        path: String,
    },
}

/// Where a launch's package comes from.
#[derive(Clone, Debug)]
pub enum Holder {
    /// The controller holds the package; rank zero receives every file complete.
    Controller { primary: PathBuf },
    /// A node holds the package; every other node fetches from it.
    Node { node: Node, path: String },
}

/// One file of a package: the primary or a companion.
#[derive(Clone, Debug)]
pub struct PackageEntry {
    pub name: String,
    pub size: u64,
}

/// The byte ranges of one package file that a rank reads, at their offsets in the file.
#[derive(Clone, Debug)]
pub struct RankFile {
    pub name: String,
    pub size: u64,
    pub ranges: Vec<(u64, u64)>,
}

/// What a package needs from the GPUs that serve it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageNeeds {
    /// The hardware revision every GPU of the group must have.
    pub revision: String,
    /// True when the ranks run on separate hosts over RoCEnante; false for a single-host
    /// tensor-parallel package whose ranks run in one process over PCIe.
    pub multi_host: bool,
    /// Device bytes each rank bundle allocates before its KV cache, in rank order.
    pub rank_bytes: Vec<u64>,
    /// Bytes the serving contract reserves on every GPU beyond the allocations.
    pub reserve_bytes: u64,
}

/// A verified package and what each member reads of it: one member per host of a multi-host
/// package, or one member holding the complete package for a single-host one.
#[derive(Clone, Debug)]
pub struct ResolvedPackage {
    /// The export identity every rank's file records, or the primary's stem for a single-host
    /// package; with the primary's length it names cache directories.
    pub export: String,
    /// The primary file, then its companions.
    pub files: Vec<PackageEntry>,
    /// Each member's files and ranges, indexed by rank.
    pub ranks: Vec<Vec<RankFile>>,
    pub publisher: Publisher,
    pub holder: Holder,
    pub needs: PackageNeeds,
}

/// A started group: its grant and each rank's node with the session streaming its output, in
/// rank order.
pub struct Launch {
    pub grant: Grant,
    pub ranks: Vec<(Node, Session)>,
}

/// How a followed rank ended: its exit status, or why its output stopped arriving.
pub struct RankEnd {
    pub rank: usize,
    pub node: String,
    pub result: Result<i32, String>,
}

/// Orbit placement and lifecycle management use the shared paired-agent client.
/// Build one authenticated download request with public verification profiles.
/// Controller-local profile paths are parsed here and never sent to the node.
fn download_request(
    source: &str,
    revision: &str,
    options: &crate::cli::HubOptions,
    publisher_keys: &[PathBuf],
    trust_store: Option<&Path>,
    access_token: Option<String>,
) -> Result<super::agent::DownloadRequest, Failure> {
    let mut profiles =
        crate::trust::publishers(publisher_keys, true, crate::environment::home_directory)
            .map_err(|error| error.to_string())?;
    for remembered in
        crate::trust::trusted_profiles(trust_store).map_err(|error| error.to_string())?
    {
        // A fresh explicit profile for the same signer and namespace wins, including
        // a changed display name, matching trust::resolve_identity_with.
        if !profiles.iter().any(|profile| {
            profile.fingerprint() == remembered.fingerprint()
                && profile.namespace == remembered.namespace
        }) {
            profiles.push(remembered);
        }
    }
    Ok(super::agent::DownloadRequest {
        source: source.into(),
        // These paths describe the controller. The node must use its own cache
        // and must not receive a controller token-file path.
        options: crate::cli::HubOptions {
            revision: Some(revision.to_owned()),
            token_file: None,
            cache_dir: None,
            model_cache: None,
            ..options.clone()
        },
        access_token,
        profiles,
    })
}

pub struct Controller(crate::agent_client::Client);
impl std::ops::Deref for Controller {
    type Target = crate::agent_client::Client;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Controller {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Controller {
    pub fn open(orbit_path: Option<&Path>, key_path: Option<&Path>) -> Result<Self, Failure> {
        Ok(Self(crate::agent_client::Client::open(
            orbit_path, key_path,
        )?))
    }
    #[cfg(test)]
    pub fn new(key: Key, orbit: Orbit, orbit_path: PathBuf) -> Self {
        Self(crate::agent_client::Client::new(key, orbit, orbit_path))
    }
    /// The orbit file's nodes of a cluster, in rank order.
    pub fn cluster_nodes(&self, name: Option<&str>) -> Result<(String, Vec<Node>), Failure> {
        self.cluster(name)
    }

    /// The orbit file's nodes of a cluster, in rank order.
    fn cluster(&self, name: Option<&str>) -> Result<(String, Vec<Node>), Failure> {
        let (name, nodes) = self.orbit.cluster(name)?;
        Ok((name, nodes.into_iter().cloned().collect()))
    }

    // ---- status --------------------------------------------------------------------------------

    /// One node's STATUS; `toward` asks for the local address the node routes toward it.
    pub fn node_status(&self, node: &Node, toward: Option<&str>) -> Result<NodeStatus, Failure> {
        self.query_status(node, toward)
            .map_err(|e| at(&node.name, e))
    }

    fn query_status(&self, node: &Node, toward: Option<&str>) -> Result<NodeStatus, Failure> {
        let (mut session, _) = self.node_session(node)?;
        status_on(&mut session, toward)
    }

    /// Print every node of a cluster with its free cache space, controller-set revision, GPUs
    /// with their holders, and the groups it runs.
    pub fn status(&self, cluster: Option<&str>) -> Result<(), Failure> {
        let (name, nodes) = self.cluster(cluster)?;
        let results = parallel(&nodes, |_, node| self.query_status(node, None));
        let width = nodes.iter().map(|n| n.name.len()).max().unwrap_or(0).max(4);
        println!("cluster {name}");
        println!(
            "{:<width$}  {:<12}  {:>8}  {:>8}  GROUPS",
            "NODE", "FINGERPRINT", "FREE GiB", "REVISION"
        );
        for (node, result) in nodes.iter().zip(results) {
            match result {
                Ok(status) => {
                    let running: Vec<String> = status
                        .running()
                        .map(|g| format!("{} rank {}", g.name, g.rank))
                        .collect();
                    println!(
                        "{:<width$}  {:<12}  {:>8.1}  {:>8}  {}",
                        node.name,
                        status.fingerprint.get(..12).unwrap_or(&status.fingerprint),
                        status.free_bytes as f64 / GIB,
                        status.controllers_revision,
                        if running.is_empty() {
                            describe_group(status.group.as_ref())
                        } else {
                            running.join(", ")
                        }
                    );
                    for gpu in &status.gpus {
                        println!(
                            "{:<width$}  gpu {} {} ({}) {:.1}/{:.1} GiB free{}",
                            "",
                            gpu.index,
                            gpu.name,
                            gpu.revision,
                            gpu.memory_free as f64 / GIB,
                            gpu.memory_total as f64 / GIB,
                            gpu.group
                                .as_deref()
                                .map_or(String::new(), |g| format!(", held by {g}"))
                        );
                    }
                    if let Some(error) = &status.gpus_error {
                        println!("{:<width$}  gpus: {error}", "");
                    }
                    for package in &status.packages {
                        println!(
                            "{:<width$}  package {} {} files, {}, {}",
                            "",
                            package.name,
                            package.files.len(),
                            gib(package.bytes),
                            if package.verified {
                                "verified"
                            } else {
                                "unverified"
                            }
                        );
                    }
                }
                Err(error) => println!("{:<width$}  unreachable: {error}", node.name),
            }
        }
        Ok(())
    }

    /// Print every group on the cluster's nodes: name, identifier, node, rank, GPUs, state.
    pub fn ls(&self, cluster: Option<&str>) -> Result<(), Failure> {
        let (name, nodes) = self.cluster(cluster)?;
        let results = parallel(&nodes, |_, node| self.query_status(node, None));
        println!("cluster {name}");
        for spec in &self.orbit.group {
            if !spec.nodes.is_empty() && !nodes.iter().any(|node| spec.nodes.contains(&node.name)) {
                continue;
            }
            let state = saved_group_state(spec, &results);
            println!("persistent {}: {state}; {}", spec.name, spec.package);
        }
        println!(
            "{:<16}  {:<36}  {:<10}  {:>4}  {:<24}  STATE",
            "NAME", "GROUP", "NODE", "RANK", "GPUS"
        );
        for (node, result) in nodes.iter().zip(results) {
            match result {
                Ok(status) => {
                    for g in &status.groups {
                        let gpus: Vec<String> = g
                            .gpus
                            .iter()
                            .map(|uuid| {
                                status
                                    .gpus
                                    .iter()
                                    .find(|gpu| &gpu.uuid == uuid)
                                    .map_or_else(|| uuid.clone(), |gpu| gpu.index.to_string())
                            })
                            .collect();
                        println!(
                            "{:<16}  {:<36}  {:<10}  {:>4}  {:<24}  {}",
                            g.name,
                            g.id,
                            node.name,
                            g.rank,
                            if gpus.is_empty() {
                                "-".into()
                            } else {
                                gpus.join(",")
                            },
                            if g.running {
                                format!("running {}", uptime(g.uptime_seconds))
                            } else {
                                format!(
                                    "exited with status {}",
                                    g.exit.map_or_else(|| "unknown".into(), |c| c.to_string())
                                )
                            }
                        );
                    }
                }
                Err(error) => println!(
                    "{:<16}  {:<36}  {:<10}  unreachable: {error}",
                    "", "", node.name
                ),
            }
        }
        Ok(())
    }

    // ---- launching -----------------------------------------------------------------------------

    /// `flare orbit run`: launch a package on a cluster, then follow its output and stop the group
    /// on Ctrl-C, or, with `detach`, return once every rank has started.
    pub fn run(
        &mut self,
        package: Option<&str>,
        cluster: Option<&str>,
        mut options: RunOptions,
    ) -> Result<(), Failure> {
        // A [[group]] of the orbit file supplies what the command line leaves out: its package,
        // its serving and rank-zero options before the command line's, and
        // its node and GPU constraints.
        let spec = options
            .name
            .as_deref()
            .and_then(|name| self.orbit.group(name))
            .cloned();
        let package = match (package, &spec) {
            (Some(package), _) => package.to_owned(),
            (None, Some(spec)) => spec.package.clone(),
            (None, None) => {
                return Err(
                    "PACKAGE is required unless --name names a [[group]] of the orbit file".into(),
                );
            }
        };
        let (label, candidates): (String, Vec<String>) = if !options.nodes.is_empty() {
            ("nodes".into(), options.nodes.clone())
        } else if let Some(cluster) = cluster {
            let (name, nodes) = self.cluster(Some(cluster))?;
            (name, nodes.into_iter().map(|n| n.name).collect())
        } else if let Some(spec) = spec.as_ref().filter(|s| !s.nodes.is_empty()) {
            (spec.name.clone(), spec.nodes.clone())
        } else {
            ("orbit".into(), Vec::new())
        };
        let name = options.name.clone().unwrap_or_else(|| {
            if cluster.is_some() {
                label.clone()
            } else {
                default_group_name(&package)
            }
        });
        let mut serving = spec.as_ref().map_or_else(Vec::new, |s| s.serving.clone());
        serving.extend(options.serving.iter().cloned());
        let verify_weights = options.verify_weights
            || options.run_args.serving.verify_weights
            || serving.iter().any(|arg| arg == "--verify-weights");
        if verify_weights && !serving.iter().any(|a| a == "--verify-weights") {
            serving.push("--verify-weights".into());
        }
        let mut rank_zero = spec.as_ref().map_or_else(Vec::new, |s| s.rank_zero.clone());
        rank_zero.extend(options.rank_zero.iter().cloned());
        // An explicit shape on the command line replaces the saved one; otherwise the saved shape holds.
        let (roce_ring, roce_chain) = if options.roce_ring || options.roce_chain {
            (options.roce_ring, options.roce_chain)
        } else {
            spec.as_ref()
                .map_or((false, false), |s| (s.roce_ring, s.roce_chain))
        };
        let transfer_streams = options
            .transfer_streams
            .or_else(|| spec.as_ref().and_then(|s| s.transfer_streams));
        package_streams(transfer_streams)?;
        let gpus = if options.gpus.is_empty() {
            spec.as_ref().map_or_else(Vec::new, |s| s.gpus.clone())
        } else {
            options.gpus.clone()
        };

        if options.persistent
            && options
                .name
                .as_deref()
                .is_none_or(|name| name.trim().is_empty())
        {
            return Err("--persistent requires a nonempty --name".into());
        }
        if options.executable.is_none() {
            options.executable = spec.as_ref().and_then(|s| s.executable.clone());
        }
        if let Some(spec) = &spec {
            if options.run_args.serving.publisher_keys.is_empty() {
                options.run_args.serving.publisher_keys = spec.publisher_keys.clone();
            }
            if options.run_args.serving.trust_store.is_none() {
                options.run_args.serving.trust_store = spec.trust_store.clone();
            }
        }
        let executable = read_executable(options.executable.as_deref())?;
        let repository = crate::hub::is_repository(Path::new(&package));
        let resolved = if repository {
            None
        } else {
            Some(self.resolve_package(&package, &mut options.run_args, verify_weights)?)
        };

        // The banner circles while the nodes are read and the group placed; nothing else is
        // printed until it has completed a revolution.
        let banner = super::banner::Banner::start();

        // The pool: the candidates in order, or every node of the orbit that answers.
        let pool: Vec<Node> = if candidates.is_empty() {
            self.orbit.node.clone()
        } else {
            candidates
                .iter()
                .map(|name| {
                    self.orbit
                        .node(name)
                        .cloned()
                        .ok_or_else(|| format!("the orbit file has no node named {name}").into())
                })
                .collect::<Result<_, Failure>>()?
        };
        if pool.is_empty() {
            return Err("the orbit file lists no nodes; pair some first".into());
        }
        let results = parallel(&pool, |_, node| self.query_status(node, None));
        let mut nodes = Vec::new();
        let mut statuses = Vec::new();
        let mut skipped = Vec::new();
        for (node, result) in pool.into_iter().zip(results) {
            match result {
                Ok(status) => {
                    nodes.push(node);
                    statuses.push(status);
                }
                Err(error) if candidates.is_empty() => {
                    skipped.push(format!("{}: not a candidate: {error}", node.name))
                }
                Err(error) => return Err(at(&node.name, error)),
            }
        }
        if (options.persistent || spec.is_some())
            && statuses
                .iter()
                .any(|status| status.running().any(|group| group.name == name))
        {
            return Err(format!(
                "persistent group {name} is already running; stop it before starting it again"
            )
            .into());
        }
        if let Some(banner) = banner {
            banner.finish();
        }
        let resolved = match resolved {
            Some(resolved) => resolved,
            None => self.resolve_repository(
                &package,
                &nodes,
                &statuses,
                &gpus,
                &mut options.run_args,
                verify_weights,
            )?,
        };
        if repository {
            statuses = all(parallel(&nodes, |_, node| self.query_status(node, None)))?;
        }
        let names: Vec<String> = nodes.iter().map(|n| n.name.clone()).collect();
        let length = resolved.files[0].size;
        let needs = plan::Needs {
            revision: resolved.needs.revision.clone(),
            multi_host: resolved.needs.multi_host,
            rank_bytes: resolved.needs.rank_bytes.clone(),
            reserve_bytes: resolved.needs.reserve_bytes,
            complete_directory: format!("{}-{length}", resolved.export),
            rank_directories: (0..resolved.needs.rank_bytes.len())
                .map(|rank| format!("{}-{length}-rank{rank}", resolved.export))
                .collect(),
            source_node: match &resolved.holder {
                Holder::Node { node, .. } => Some(node.name.clone()),
                Holder::Controller { .. } => None,
            },
        };
        let constraints = plan::Constraints {
            nodes: candidates.clone(),
            gpus: gpus.clone(),
        };
        let placements = plan::plan(&needs, &names, &statuses, &constraints);
        for line in &skipped {
            eprintln!("{line}");
        }
        let placements = placements?;
        eprintln!(
            "placement of {} ({}, {}):\n{}",
            name,
            needs.revision,
            if needs.multi_host {
                format!("{} hosts", needs.rank_bytes.len())
            } else {
                format!("{} GPUs on one host", needs.rank_bytes.len())
            },
            plan::describe(&placements, &names, &statuses)
        );
        if options.dry_run {
            eprintln!("dry run: nothing started");
            return Ok(());
        }
        if options.persistent {
            let package = match &resolved.holder {
                Holder::Controller { primary } => primary.to_string_lossy().into_owned(),
                Holder::Node { node, path } => format!("{}:{path}", node.name),
            };
            let saved = GroupSpec {
                name: name.clone(),
                package,
                serving: serving.clone(),
                rank_zero: rank_zero.clone(),
                nodes: candidates.clone(),
                gpus,
                roce_ring,
                transfer_streams,
                roce_chain,
                executable: options
                    .executable
                    .as_deref()
                    .map(Path::canonicalize)
                    .transpose()?,
                publisher_keys: options
                    .run_args
                    .serving
                    .publisher_keys
                    .iter()
                    .map(|path| path.canonicalize())
                    .collect::<Result<_, _>>()?,
                trust_store: options
                    .run_args
                    .serving
                    .trust_store
                    .as_deref()
                    .map(absolute_path)
                    .transpose()?,
            };
            // Persist validated intent before RUN so a failed launch remains restartable.
            let mut orbit = Orbit::load(&self.orbit_path)?;
            orbit.save_group(saved);
            orbit.save(&self.orbit_path)?;
            self.orbit = orbit;
            eprintln!(
                "saved persistent group {name}; start it again with `flare orbit run --name {name}`"
            );
        }
        let members: Vec<Node> = placements.iter().map(|p| nodes[p.node].clone()).collect();
        let launch = self.launch(
            &label,
            &members,
            &resolved,
            executable.as_deref(),
            &LaunchOptions {
                name: name.clone(),
                placements: placements.iter().map(|p| p.gpus.clone()).collect(),
                serving,
                rank_zero: rank_zero.clone(),
                roce_ring,
                transfer_streams,
                stage_only: false,
                roce_chain,
            },
        )?;
        eprintln!(
            "group {} started; once loaded, rank zero on {} serves {}",
            launch.grant.group,
            members[0].name,
            serving_url(&rank_zero, &members[0].address)
        );
        if options.detach {
            eprintln!(
                "`flare orbit attach {name}` follows its output; `flare orbit stop {name}` stops it"
            );
            return Ok(());
        }
        self.supervise(launch)
    }

    /// `flare orbit push`: place a package's ranks on nodes, in rank order, or its complete
    /// copy on one node, verifying it there, without starting anything.
    pub fn push(
        &self,
        package: &str,
        to: &[String],
        run_args: &mut crate::cli::Run,
        transfer_streams: Option<u8>,
    ) -> Result<(), Failure> {
        package_streams(transfer_streams)?;
        let resolved = self.resolve_package(package, run_args, false)?;
        let world = resolved.ranks.len();
        let nodes: Vec<Node> = to
            .iter()
            .map(|name| {
                self.orbit
                    .node(name)
                    .cloned()
                    .ok_or_else(|| format!("the orbit file has no node named {name}").into())
            })
            .collect::<Result<_, Failure>>()?;
        let name = Path::new(package)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "package".into());
        let options = LaunchOptions {
            name: format!("push-{name}"),
            placements: vec![Vec::new(); world],
            stage_only: true,
            roce_ring: run_args.roce_ring,
            roce_chain: run_args.roce_chain,
            transfer_streams,
            ..LaunchOptions::default()
        };
        if nodes.len() == world {
            let launch = self.launch("push", &nodes, &resolved, None, &options)?;
            eprintln!(
                "{} staged on {}",
                launch.grant.name,
                nodes
                    .iter()
                    .enumerate()
                    .map(|(rank, n)| format!("{} (rank {rank})", n.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return Ok(());
        }
        if run_args.roce_ring || run_args.roce_chain {
            return Err(
                "--roce-ring and --roce-chain push require one target node per package rank, in rank order"
                    .into(),
            );
        }
        // Fewer or more nodes than ranks: each listed node gets the complete package, which
        // serves rank zero of a multi-host package or a single-host package entirely.
        let complete = ResolvedPackage {
            ranks: vec![
                resolved
                    .files
                    .iter()
                    .map(|f| RankFile {
                        name: f.name.clone(),
                        size: f.size,
                        ranges: vec![(0, f.size)],
                    })
                    .collect(),
            ],
            ..resolved.clone()
        };
        let options = LaunchOptions {
            placements: vec![Vec::new()],
            ..options
        };
        for node in &nodes {
            if let Holder::Node { node: holder, .. } = &resolved.holder
                && holder.public_key == node.public_key
            {
                eprintln!("{}: holds the package", node.name);
                continue;
            }
            self.launch(
                "push",
                std::slice::from_ref(node),
                &complete,
                None,
                &options,
            )?;
            eprintln!("{}: complete package staged", node.name);
        }
        Ok(())
    }

    /// Print what every node of a cluster holds in its package cache.
    pub fn packages(&self, cluster: Option<&str>) -> Result<(), Failure> {
        let (name, nodes) = self.cluster(cluster)?;
        let results = parallel(&nodes, |_, node| self.query_status(node, None));
        println!("cluster {name}");
        println!(
            "{:<10}  {:<52}  {:>5}  {:>10}  STATE",
            "NODE", "PACKAGE", "FILES", "ON DISK"
        );
        for (node, result) in nodes.iter().zip(results) {
            match result {
                Ok(status) if status.packages.is_empty() => {
                    println!("{:<10}  -", node.name)
                }
                Ok(status) => {
                    for package in &status.packages {
                        println!(
                            "{:<10}  {:<52}  {:>5}  {:>10}  {}",
                            node.name,
                            package.name,
                            package.files.len(),
                            gib(package.bytes),
                            if package.verified {
                                "verified"
                            } else {
                                "unverified"
                            }
                        );
                    }
                }
                Err(error) => println!("{:<10}  unreachable: {error}", node.name),
            }
        }
        Ok(())
    }

    /// Resolve and verify a package reference, obtaining publisher acceptance at this terminal,
    /// and compute every rank's byte ranges. A node-held package is verified from a local mirror
    /// of its header, signed index, and companion headers; payload bytes stay holes there, so its
    /// stored data is verified only by ranks that load with `--verify-weights`.
    pub fn resolve_package(
        &self,
        reference: &str,
        run_args: &mut crate::cli::Run,
        verify_weights: bool,
    ) -> Result<ResolvedPackage, Failure> {
        self.resolve_package_from(reference, run_args, verify_weights, None)
    }

    fn resolve_repository(
        &self,
        source: &str,
        nodes: &[Node],
        statuses: &[NodeStatus],
        gpus: &[u32],
        run_args: &mut crate::cli::Run,
        verify_weights: bool,
    ) -> Result<ResolvedPackage, Failure> {
        let options = &run_args.hub;
        let revision = repository_revision(source, options, statuses, gpus)?;
        let node = nodes
            .iter()
            .zip(statuses)
            .find(|(_, status)| {
                status.gpus.iter().any(|gpu| {
                    gpu.group.is_none() && (gpus.is_empty() || gpus.contains(&gpu.index))
                })
            })
            .map(|(node, _)| node)
            .ok_or("no candidate node has an available GPU for the repository package")?;
        let access_token = if options.offline {
            None
        } else {
            crate::hub::token(options).map_err(|e| e.to_string())?
        };
        let request = download_request(
            source,
            &revision,
            options,
            &run_args.serving.publisher_keys,
            run_args.serving.trust_store.as_deref(),
            access_token,
        )?;
        eprintln!(
            "{}: resolving and downloading {source} at revision {revision}",
            node.name
        );
        let (mut session, _) = self.node_session(node)?;
        let reply: Value = serde_json::from_slice(
            &session
                .request(frame::DOWNLOAD, &serde_json::to_value(request)?)
                .map_err(|e| at(&node.name, e))?,
        )?;
        let path = reply["path"]
            .as_str()
            .ok_or("DOWNLOAD reply lacks a path")?;
        let reference = format!("{}:{path}", node.name);
        let (id, _) = crate::hub::repository(source).map_err(|e| e.to_string())?;
        self.resolve_package_from(&reference, run_args, verify_weights, Some(id))
    }

    fn resolve_package_from(
        &self,
        reference: &str,
        run_args: &mut crate::cli::Run,
        verify_weights: bool,
        origin: Option<&str>,
    ) -> Result<ResolvedPackage, Failure> {
        // The controller computes every rank's ranges, so it verifies every rank's index.
        run_args.rank = None;
        match parse_package_reference(reference, |name| self.orbit.node(name).is_some()) {
            PackageReference::Local(path) => {
                let primary = path.canonicalize().map_err(|e| {
                    format!("cannot access package {reference}: {e}; a node-held package is NODE:/ABSOLUTE/PATH with NODE a orbit node")
                })?;
                run_args.serving.verify_weights = verify_weights;
                let (authorized, publisher) = verify(&primary, run_args, origin, false)?;
                describe(
                    &authorized,
                    &primary,
                    publisher,
                    Holder::Controller {
                        primary: primary.clone(),
                    },
                )
            }
            PackageReference::Node { node, path } => {
                let node = self
                    .orbit
                    .node(&node)
                    .ok_or_else(|| format!("the orbit file has no node named {node}"))?
                    .clone();
                let mirror = Mirror::create()?;
                let primary = self
                    .mirror_package(&node, &path, &mirror.0)
                    .map_err(|e| at(&node.name, e))?;
                run_args.serving.verify_weights = false;
                let (authorized, publisher) = verify(&primary, run_args, origin, true)?;
                describe(
                    &authorized,
                    &primary,
                    publisher,
                    Holder::Node { node, path },
                )
            }
        }
    }

    /// Mirror a node-held package into `directory`: every file at its full length, holding only
    /// the primary's safetensors header and signed index and each companion's header. Returns the
    /// mirrored primary's path.
    fn mirror_package(
        &self,
        node: &Node,
        path: &str,
        directory: &Path,
    ) -> Result<PathBuf, Failure> {
        let remote = Path::new(path);
        if !remote.is_absolute() {
            return Err(format!("{path}: a node-held package needs an absolute path").into());
        }
        let primary_name = remote
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("{path} names no file"))?;
        let (mut session, _) = self.node_session(node)?;
        let stat: Value =
            serde_json::from_slice(&session.request(frame::STAT, &json!({"path": path}))?)?;
        let size = stat["size"].as_u64().ok_or("STAT reply lacks size")?;
        let primary = directory.join(primary_name);
        mirror_file(&mut session, path, &primary, size, true)?;
        for companion in stat["companions"]
            .as_array()
            .ok_or("STAT reply lacks companions")?
        {
            let name = companion["name"]
                .as_str()
                .ok_or("STAT companion lacks name")?;
            let size = companion["size"]
                .as_u64()
                .ok_or("STAT companion lacks size")?;
            if !companion_path_is_valid(name) {
                return Err(format!("invalid companion name {name:?}").into());
            }
            let root = remote.parent().unwrap_or_else(|| Path::new("/"));
            let companion_path = root.join(name);
            let companion_path = companion_path.to_str().ok_or("non-UTF-8 companion path")?;
            let local = directory.join(name);
            if let Some(parent) = local.parent() {
                fs::create_dir_all(parent)?;
            }
            mirror_file(&mut session, companion_path, &local, size, false)?;
        }
        Ok(primary)
    }

    /// Start a resolved package on `nodes`, given in rank order: check each node's STATUS, issue
    /// the grant, place the executable and package, and send RUN to rank zero and then to the
    /// peers. Returns the grant and each rank's session, which streams the rank's output.
    pub fn launch(
        &self,
        cluster: &str,
        nodes: &[Node],
        package: &ResolvedPackage,
        executable: Option<&[u8]>,
        options: &LaunchOptions,
    ) -> Result<Launch, Failure> {
        let world = package.ranks.len();
        if nodes.len() != world {
            return Err(format!(
                "cluster {cluster} lists {} nodes; the package has {world} ranks",
                nodes.len()
            )
            .into());
        }
        if options.placements.len() != world {
            return Err(
                format!("{} placements for {world} ranks", options.placements.len()).into(),
            );
        }
        if (options.roce_ring || options.roce_chain)
            && !(package.needs.multi_host && (2..=16).contains(&world))
        {
            return Err(
                "--roce-ring and --roce-chain require a multi-host package with 2..16 ranks".into(),
            );
        }
        let streams = if world > 1 {
            package_streams(options.transfer_streams)?
        } else {
            1
        };

        // Every node must be free. Rank zero reports the address it routes toward its peers,
        // where they reach its rendezvous.
        let toward = nodes.get(1).unwrap_or(&nodes[0]).address.clone();
        let opened = parallel(nodes, |rank, node| {
            (|| -> Result<(Session, NodeStatus), Failure> {
                let (mut session, hello) = self.node_session(node)?;
                if executable.is_none() && !options.stage_only {
                    let installed = hello.installed_executable()?;
                    eprintln!(
                        "{}: using installed flare {}",
                        node.name,
                        installed.display()
                    );
                }
                let status = status_on(&mut session, (rank == 0).then_some(toward.as_str()))?;
                Ok((session, status))
            })()
            .map_err(|e| at(&node.name, e))
        });
        let (mut sessions, statuses): (Vec<Session>, Vec<NodeStatus>) =
            all(opened)?.into_iter().unzip();
        let placements = &options.placements;
        for (rank, placement) in placements
            .iter()
            .enumerate()
            .filter(|_| !options.stage_only)
        {
            for uuid in placement {
                match statuses[rank].gpus.iter().find(|g| &g.uuid == uuid) {
                    Some(gpu) => {
                        if let Some(holder) = &gpu.group {
                            return Err(at(
                                &nodes[rank].name,
                                format!("GPU {} is held by group {holder}", gpu.index).into(),
                            ));
                        }
                    }
                    None => return Err(at(&nodes[rank].name, format!("no GPU {uuid}").into())),
                }
            }
        }
        let rendezvous = if package.needs.multi_host {
            let routed = statuses[0].address_toward.as_deref().ok_or_else(|| {
                format!(
                    "{}: STATUS reported no address toward {toward}",
                    nodes[0].name
                )
            })?;
            SocketAddr::new(routed.parse::<IpAddr>()?, RENDEZVOUS_PORT).to_string()
        } else {
            String::new()
        };

        let issued = now();
        let transfer_fabric = if world > 1
            && (options.stage_only
                || matches!(&package.holder,
            Holder::Node { node, .. } if node.public_key != nodes[0].public_key))
        {
            let records = all(parallel_mut(&mut sessions, |rank, session| {
                session
                    .request(frame::FABRIC, &json!({}))
                    .map_err(|e| at(&nodes[rank].name, e))
            }))?;
            Some(super::fabric::plan(
                nodes,
                &records,
                options.roce_ring,
                options.roce_chain,
            )?)
        } else {
            None
        };
        let mut grant = Grant {
            group: group_id(&options.name, issued, &random(2)?),
            name: options.name.clone(),
            package: Package {
                export: package.export.clone(),
                length: package.files[0].size,
                files: package.files.iter().map(|f| f.name.clone()).collect(),
                source: match &package.holder {
                    Holder::Controller { .. } => None,
                    Holder::Node { node, path } => Some(Source {
                        public_key: node.public_key.clone(),
                        path: path.clone(),
                    }),
                },
            },
            members: nodes
                .iter()
                .enumerate()
                .map(|(rank, node)| Member {
                    rank,
                    name: node.name.clone(),
                    public_key: node.public_key.clone(),
                    address: node.address.clone(),
                    gpus: placements[rank].clone(),
                })
                .collect(),
            rendezvous,
            serving: options.serving.clone(),
            rank_zero: options.rank_zero.clone(),
            roce_ring: options.roce_ring,
            transfer_fabric,
            single_host: !package.needs.multi_host,
            roce_chain: options.roce_chain,
            issued,
            expires: issued + GRANT_SECONDS,
        };
        let transfer_grant = Signed::sign(&self.key, &grant)?;
        eprintln!(
            "group {}: {world} ranks, rendezvous {}",
            grant.group, grant.rendezvous
        );
        for (rank, node) in nodes.iter().enumerate() {
            eprintln!(
                "  rank {rank} on {}: {}",
                node.name,
                describe_gpus(&statuses[rank], &placements[rank])
            );
        }

        // Only an explicit override transfers an executable. With the package at the
        // controller, rank zero gets every file of it complete.
        let binary = executable.map(|bytes| (cache::binary_name(bytes), bytes));
        let complete = grant.complete_directory();
        all(parallel_mut(&mut sessions, |rank, session| {
            (|| -> Result<(), Failure> {
                let mut needed = 0;
                let send_binary = match &binary {
                    Some((name, bytes))
                        if !options.stage_only
                            && !have(session, "binary", name, bytes.len() as u64)? =>
                    {
                        needed += bytes.len() as u64;
                        Some((name, *bytes))
                    }
                    _ => None,
                };
                let mut sends = Vec::new();
                if let (0, Holder::Controller { primary }) = (rank, &package.holder) {
                    for file in &package.files {
                        let name = format!("{complete}/{}", file.name);
                        if !have(session, "package", &name, file.size)? {
                            needed += file.size;
                            let root = primary.parent().unwrap_or_else(|| Path::new("."));
                            sends.push((root.join(&file.name), name, file.size));
                        }
                    }
                }
                check_space(&statuses[rank], needed)?;
                if let Some((name, bytes)) = send_binary {
                    put(session, "binary", name, bytes.len() as u64, &mut &bytes[..])?;
                    eprintln!("{}: received flare executable {name}", nodes[rank].name);
                }
                for (path, name, size) in sends {
                    let mut source = File::open(&path)
                        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
                    eprintln!(
                        "{}: sending {} ({})",
                        nodes[rank].name,
                        path.display(),
                        gib(size)
                    );
                    put(session, "package", &name, size, &mut source)?;
                }
                Ok(())
            })()
            .map_err(|e| at(&nodes[rank].name, e))
        }))?;

        // Launch needs the complete source on rank zero. Prestaging also fetches peer ranges;
        // serving launches leave those ranges to the rank processes.
        let mut source = match &package.holder {
            Holder::Controller { .. } => nodes[0].clone(),
            Holder::Node { node, .. } => node.clone(),
        };
        // An outside source has no fabric links. Stage it once on rank zero before its peers fetch.
        if grant.transfer_fabric.is_some()
            && !nodes
                .iter()
                .any(|node| node.public_key == source.public_key)
        {
            let files = package
                .files
                .iter()
                .map(|f| RankFile {
                    name: f.name.clone(),
                    size: f.size,
                    ranges: vec![(0, f.size)],
                })
                .collect::<Vec<_>>();
            transfer(
                &mut sessions[0],
                &nodes[0],
                &statuses[0],
                &transfer_grant,
                &package.publisher,
                &source,
                &complete,
                &files,
                1,
            )?;
            source = nodes[0].clone();
        }
        all(parallel_mut(&mut sessions, |rank, session| {
            if (rank > 0 && !options.stage_only) || nodes[rank].public_key == source.public_key {
                return Ok(());
            }
            let (directory, files) = if rank == 0 {
                let files: Vec<RankFile> = package
                    .files
                    .iter()
                    .map(|f| RankFile {
                        name: f.name.clone(),
                        size: f.size,
                        ranges: vec![(0, f.size)],
                    })
                    .collect();
                (complete.clone(), files)
            } else {
                (grant.cache_directory(rank), package.ranks[rank].clone())
            };
            transfer(
                session,
                &nodes[rank],
                &statuses[rank],
                &transfer_grant,
                &package.publisher,
                &source,
                &directory,
                &files,
                streams,
            )
            .map_err(|e| at(&nodes[rank].name, e))
        }))?;

        if options.stage_only {
            return Ok(Launch {
                grant,
                ranks: Vec::new(),
            });
        }
        grant
            .serving
            .extend(["--transfer-streams".into(), streams.to_string()]);
        // Transfers can outlast a grant's launch window, so RUN carries a grant issued now.
        grant.issued = now();
        grant.expires = grant.issued + GRANT_SECONDS;
        let signed = Signed::sign(&self.key, &grant)?;
        let publisher = serde_json::to_value(&package.publisher)?;
        for rank in 0..world {
            let request = json!({
                "grant": signed,
                "rank": rank,
                "binary": binary.as_ref().map(|(name, _)| name),
                "publisher": publisher,
            });
            if let Err(error) = sessions[rank].request(frame::RUN, &request) {
                let error = at(&nodes[rank].name, error);
                if rank > 0 {
                    eprintln!("{error}; stopping the ranks already started");
                    let started: Vec<(usize, Node)> =
                        (0..rank).map(|r| (r, nodes[r].clone())).collect();
                    if let Err(stop) = self.stop_group(&grant.group, &started) {
                        eprintln!("{stop}");
                    }
                }
                return Err(error);
            }
        }
        Ok(Launch {
            grant,
            ranks: nodes.iter().cloned().zip(sessions).collect(),
        })
    }

    /// Print a launched group's output until every rank exits. The first Ctrl-C stops rank zero,
    /// a second stops every peer at once, and a third returns without waiting.
    fn supervise(&self, launch: Launch) -> Result<(), Failure> {
        let group = launch.grant.group;
        let nodes: Vec<Node> = launch.ranks.iter().map(|(node, _)| node.clone()).collect();
        let events = follow(
            launch
                .ranks
                .into_iter()
                .enumerate()
                .map(|(rank, (node, session))| (rank, node.name, session))
                .collect(),
        );
        catch_interrupts();
        let mut ended: BTreeMap<usize, Result<i32, String>> = BTreeMap::new();
        let mut rank_zero_exit: Option<Instant> = None;
        let mut interrupts = 0;
        let mut peers_stopped = false;
        while ended.len() < nodes.len() {
            match events.recv_timeout(Duration::from_millis(250)) {
                Ok(end) => {
                    report_end(&end);
                    if end.rank == 0 && end.result.is_ok() {
                        rank_zero_exit = Some(Instant::now());
                    }
                    ended.insert(end.rank, end.result);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if interrupted() {
                interrupts += 1;
                match interrupts {
                    1 => {
                        eprintln!(
                            "stopping group {group}: rank 0 drains first; Ctrl-C again stops every rank now"
                        );
                        self.request_stop(&nodes[0], &group);
                    }
                    2 => {
                        eprintln!(
                            "stopping every rank of group {group}; Ctrl-C again leaves without waiting"
                        );
                        for (rank, node) in nodes.iter().enumerate().skip(1) {
                            if !ended.contains_key(&rank) {
                                self.request_stop(node, &group);
                            }
                        }
                        peers_stopped = true;
                    }
                    _ => {
                        return Err(format!("left group {group} before every rank exited").into());
                    }
                }
            }
            if !peers_stopped && rank_zero_exit.is_some_and(|t| t.elapsed() >= PEER_GRACE) {
                peers_stopped = true;
                for (rank, node) in nodes.iter().enumerate().skip(1) {
                    if !ended.contains_key(&rank) {
                        eprintln!(
                            "rank {rank} on {} still runs 30 seconds after rank 0 exited; stopping it",
                            node.name
                        );
                        self.request_stop(node, &group);
                    }
                }
            }
        }
        let failed: Vec<String> = nodes
            .iter()
            .enumerate()
            .filter_map(|(rank, node)| match ended.get(&rank) {
                Some(Ok(0)) => None,
                Some(Ok(code)) => Some(format!(
                    "rank {rank} on {} exited with status {code}",
                    node.name
                )),
                Some(Err(error)) => Some(format!("rank {rank} on {}: {error}", node.name)),
                None => Some(format!("rank {rank} on {}: exit not observed", node.name)),
            })
            .collect();
        if failed.is_empty() {
            eprintln!("group {group} exited");
            Ok(())
        } else {
            Err(format!("group {group}: {}", failed.join("; ")).into())
        }
    }

    // ---- attaching and stopping ----------------------------------------------------------------

    /// Remove a named definition and its stopped runs, or one exact run without its definition.
    pub fn remove(&mut self, name: &str) -> Result<(), Failure> {
        // Inspect every paired node before deleting anything. An unreachable node may still run
        // this name, so partial visibility cannot authorize forgetting its saved definition.
        let statuses = all(parallel(&self.orbit.node, |_, node| {
            self.node_status(node, None)
        }))?;
        let mut runs = Vec::new();
        for (node, status) in self.orbit.node.iter().zip(&statuses) {
            for group in status
                .groups
                .iter()
                .filter(|group| group.id == name || group.name == name)
            {
                if group.running {
                    return Err(format!(
                        "group {} is running on {}; stop it before removing it",
                        group.id, node.name
                    )
                    .into());
                }
                runs.push((node.clone(), group.id.clone()));
            }
        }
        if runs.is_empty() && self.orbit.group(name).is_none() {
            return Err(format!("no saved group or retained run named {name}").into());
        }
        for (node, id) in &runs {
            let (mut session, _) = self.node_session(node)?;
            session.request(frame::REMOVE, &json!({"group": id}))?;
        }
        let mut orbit = Orbit::load(&self.orbit_path)?;
        let before = orbit.group.len();
        orbit.group.retain(|group| group.name != name);
        if before != orbit.group.len() {
            orbit.save(&self.orbit_path)?;
        }
        self.orbit = orbit;
        eprintln!("removed {name}; model packages remain cached");
        Ok(())
    }

    /// Follow the groups on a cluster's nodes, or the group named `group`: retained output,
    /// then live output until every rank exits. Ctrl-C detaches and leaves the groups running.
    pub fn attach(&self, cluster: Option<&str>, group: Option<&str>) -> Result<(), Failure> {
        let (name, nodes) = self.cluster(cluster)?;
        let attached = parallel(&nodes, |_, node| {
            (|| -> Result<Vec<(usize, String, Session)>, Failure> {
                let status = self.query_status(node, None)?;
                let mut ranks = Vec::new();
                for g in status
                    .groups
                    .iter()
                    .filter(|g| group.is_none_or(|name| g.name == name || g.id == name))
                {
                    let (mut session, _) = self.node_session(node)?;
                    session.request(frame::ATTACH, &json!({"group": g.id}))?;
                    ranks.push((g.rank, node.name.clone(), session));
                }
                Ok(ranks)
            })()
            .map_err(|e| at(&node.name, e))
        });
        let mut ranks = Vec::new();
        for result in attached {
            match result {
                Ok(found) => ranks.extend(found),
                Err(error) => eprintln!("{error}"),
            }
        }
        if ranks.is_empty() {
            return Err(format!("no node of cluster {name} has run a group").into());
        }
        let count = ranks.len();
        let events = follow(ranks);
        catch_interrupts();
        let mut ended = 0;
        while ended < count {
            match events.recv_timeout(Duration::from_millis(250)) {
                Ok(end) => {
                    report_end(&end);
                    ended += 1;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if interrupted() {
                        eprintln!("detached from cluster {name}; its group keeps running");
                        return Ok(());
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    }

    /// Stop the groups running on a cluster's nodes, or the group named `group`, rank zero
    /// first.
    pub fn stop(&self, cluster: Option<&str>, group: Option<&str>) -> Result<(), Failure> {
        let (name, nodes) = self.cluster(cluster)?;
        let mut groups: BTreeMap<String, Vec<(usize, Node)>> = BTreeMap::new();
        let results = parallel(&nodes, |_, node| self.node_status(node, None));
        for (node, result) in nodes.iter().zip(results) {
            match result {
                Ok(status) => {
                    for g in status
                        .running()
                        .filter(|g| group.is_none_or(|name| g.name == name || g.id == name))
                    {
                        groups
                            .entry(g.id.clone())
                            .or_default()
                            .push((g.rank, node.clone()));
                    }
                }
                Err(error) => eprintln!("{error}"),
            }
        }
        if groups.is_empty() {
            eprintln!("no group runs on cluster {name}");
            return Ok(());
        }
        for (group, mut members) in groups {
            members.sort_by_key(|(rank, _)| *rank);
            self.stop_group(&group, &members)?;
            eprintln!("group {group} stopped");
        }
        Ok(())
    }

    /// Stop a group given its members as (rank, node): STOP rank zero and wait for it to exit,
    /// then STOP the peers still running 30 seconds later. Ctrl-C while rank zero drains stops
    /// the peers at once.
    pub fn stop_group(&self, group: &str, members: &[(usize, Node)]) -> Result<(), Failure> {
        catch_interrupts();
        let peers: Vec<&Node> = members
            .iter()
            .filter(|(rank, _)| *rank != 0)
            .map(|(_, node)| node)
            .collect();
        let mut grace = Duration::ZERO;
        if let Some((_, zero)) = members.iter().find(|(rank, _)| *rank == 0) {
            let stopping = self.send_stop(zero, group).unwrap_or_else(|error| {
                eprintln!("{error}");
                false
            });
            if stopping {
                eprintln!(
                    "stopping rank 0 on {}; waiting for it to drain (Ctrl-C stops every rank now)",
                    zero.name
                );
                grace = PEER_GRACE;
                loop {
                    match self.group_running(zero, group) {
                        Ok(true) => {}
                        Ok(false) => {
                            eprintln!("rank 0 on {} exited", zero.name);
                            break;
                        }
                        Err(error) => {
                            eprintln!("{error}");
                            break;
                        }
                    }
                    if interrupted() {
                        grace = Duration::ZERO;
                        break;
                    }
                    thread::sleep(Duration::from_secs(1));
                }
            }
        }
        let running = self.wait_for_exit(group, &peers, grace);
        for node in &running {
            self.request_stop(node, group);
        }
        let remaining = self.wait_for_exit(group, &running, PEER_GRACE);
        if remaining.is_empty() {
            Ok(())
        } else {
            let names: Vec<&str> = remaining.iter().map(|n| n.name.as_str()).collect();
            Err(format!("group {group} still runs on {}", names.join(", ")).into())
        }
    }

    /// Poll `nodes` until none runs `group`, `limit` passes, or Ctrl-C; returns those still
    /// running. A node that cannot be queried counts as not running.
    fn wait_for_exit<'a>(&self, group: &str, nodes: &[&'a Node], limit: Duration) -> Vec<&'a Node> {
        let deadline = Instant::now() + limit;
        let mut running = nodes.to_vec();
        loop {
            running.retain(|node| {
                self.group_running(node, group).unwrap_or_else(|error| {
                    eprintln!("{error}");
                    false
                })
            });
            if running.is_empty() || Instant::now() >= deadline || interrupted() {
                return running;
            }
            thread::sleep(Duration::from_secs(1));
        }
    }

    fn group_running(&self, node: &Node, group: &str) -> Result<bool, Failure> {
        Ok(self
            .node_status(node, None)?
            .running()
            .any(|g| g.id == group))
    }

    /// Send STOP for `group` to a node; whether the node's rank was running.
    pub fn send_stop(&self, node: &Node, group: &str) -> Result<bool, Failure> {
        (|| -> Result<bool, Failure> {
            let (mut session, _) = self.node_session(node)?;
            let reply: Value =
                serde_json::from_slice(&session.request(frame::STOP, &json!({"group": group}))?)?;
            Ok(reply["stopped"].as_bool().unwrap_or(false))
        })()
        .map_err(|e| at(&node.name, e))
    }

    fn request_stop(&self, node: &Node, group: &str) {
        if let Err(error) = self.send_stop(node, group) {
            eprintln!("{error}");
        }
    }
}

// ---- requests ----------------------------------------------------------------------------------

fn status_on(session: &mut Session, toward: Option<&str>) -> Result<NodeStatus, Failure> {
    let request = match toward {
        Some(toward) => json!({"toward": toward}),
        None => json!({}),
    };
    Ok(serde_json::from_slice(
        &session.request(frame::STATUS, &request)?,
    )?)
}

/// Whether the node's cache holds the entry at `size` bytes; a held entry stays pinned for the
/// session.
fn have(session: &mut Session, kind: &str, name: &str, size: u64) -> Result<bool, Failure> {
    let reply = session.request(
        frame::HAVE,
        &json!({"kind": kind, "name": name, "size": size}),
    )?;
    Ok(reply == [1])
}

/// Store `size` bytes read from `source` as a complete cache entry on the node.
fn put(
    session: &mut Session,
    kind: &str,
    name: &str,
    size: u64,
    source: &mut dyn Read,
) -> Result<(), Failure> {
    let header = json!({"kind": kind, "name": name, "size": size}).to_string();
    let mut prefix = (header.len() as u32).to_le_bytes().to_vec();
    prefix.extend(header.as_bytes());
    let length = prefix.len() as u64 + size;
    let mut body = prefix.as_slice().chain(source);
    session.write_frame_from(frame::PUT, length, &mut body)?;
    session.expect(frame::OK)?;
    Ok(())
}

fn package_streams(selected: Option<u8>) -> Result<u8, Failure> {
    match selected {
        None => Ok(8),
        Some(streams @ 1..=64) => Ok(streams),
        Some(_) => Err("transfer streams must be in 1..=64".into()),
    }
}

/// Have the node fetch `files` from `source` into its cache under `directory`, skipping files it
/// already holds, and verify the directory's stored data unless it verified before.
#[allow(clippy::too_many_arguments)]
fn transfer(
    session: &mut Session,
    node: &Node,
    status: &NodeStatus,
    grant: &Signed,
    publisher: &Publisher,
    source: &Node,
    directory: &str,
    files: &[RankFile],
    streams: u8,
) -> Result<(), Failure> {
    let mut needed = 0;
    for file in files {
        if !have(
            session,
            "package",
            &format!("{directory}/{}", file.name),
            file.size,
        )? {
            needed += cache::range_bytes(&file.ranges);
        }
    }
    check_space(status, needed)?;
    if needed > 0 {
        eprintln!(
            "{}: fetching {} from {}, then verifying",
            node.name,
            gib(needed),
            source.name
        );
    }
    let request = json!({
        "grant": grant,
        "publisher": publisher,
        "source": {"public_key": source.public_key, "address": source.address},
        "directory": directory,
        "streams": streams,
        "files": files
            .iter()
            .map(|f| json!({"name": f.name, "size": f.size, "ranges": f.ranges}))
            .collect::<Vec<_>>(),
    });
    let reply: Value = serde_json::from_slice(&session.request(frame::TRANSFER, &request)?)?;
    let received = reply["received"].as_u64().unwrap_or(0);
    let verified = reply["verified"].as_bool().unwrap_or(false);
    match (received > 0, verified) {
        (true, _) => eprintln!("{}: received and verified {}", node.name, gib(received)),
        (false, true) => eprintln!("{}: verified the cached package", node.name),
        (false, false) => eprintln!("{}: holds the verified package", node.name),
    }
    Ok(())
}

fn check_space(status: &NodeStatus, needed: u64) -> Result<(), Failure> {
    if needed > status.free_bytes {
        return Err(format!(
            "{} free under the agent cache; this launch stores {} there",
            gib(status.free_bytes),
            gib(needed)
        )
        .into());
    }
    Ok(())
}

/// Read `nbytes` at `offset` of the node-local file `path` with GET.
fn get_bytes(
    session: &mut Session,
    path: &str,
    offset: u64,
    nbytes: u64,
) -> Result<Vec<u8>, Failure> {
    let mut bytes = Vec::with_capacity(nbytes as usize);
    let mut append = |chunk: &[u8], _: u64| -> Result<(), Failure> {
        bytes.extend_from_slice(chunk);
        Ok(())
    };
    get(session, path, offset, nbytes, &mut append)?;
    Ok(bytes)
}

/// Receives each chunk of a GET reply with its file offset.
type Sink<'a> = &'a mut dyn FnMut(&[u8], u64) -> Result<(), Failure>;

/// Read `nbytes` at `offset` of the node-local file `path` with GET, passing each chunk and its
/// file offset to `sink`.
fn get(
    session: &mut Session,
    path: &str,
    offset: u64,
    nbytes: u64,
    sink: Sink<'_>,
) -> Result<(), Failure> {
    session.write_frame(
        frame::GET,
        &serde_json::to_vec(&json!({"path": path, "offset": offset, "nbytes": nbytes}))?,
    )?;
    let (tag, length) = session.read_header()?;
    if tag == frame::ERROR {
        let message = session.read_body(length, FRAME_LIMIT)?;
        return Err(String::from_utf8_lossy(&message).into_owned().into());
    }
    if tag != frame::OK || length != nbytes {
        return Err(format!("unexpected reply to GET {path}").into());
    }
    let mut body = session.body();
    let mut buffer = vec![0u8; 1 << 20];
    let mut position = offset;
    loop {
        let count = body.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        sink(&buffer[..count], position)?;
        position += count as u64;
    }
    if position != offset + nbytes {
        return Err(format!("GET {path} ended early").into());
    }
    Ok(())
}

/// A companion name is a relative package path: nested components are allowed, traversal and
/// hidden entries are not.
fn companion_path_is_valid(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && !name.contains('\0')
        && name.split('/').all(|part| {
            !part.is_empty()
                && !part.starts_with('.')
                && part.len() <= 128
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        })
}

/// Create `local` at `size` bytes holding the safetensors header of the node-local file `remote`
/// and, for a primary, its signed index; every other byte is a hole.
fn mirror_file(
    session: &mut Session,
    remote: &str,
    local: &Path,
    size: u64,
    primary: bool,
) -> Result<(), Failure> {
    let file = File::create(local)?;
    file.set_len(size)?;
    if size < 8 {
        return Err(format!("{remote} is too short for a package file").into());
    }
    let prefix = get_bytes(session, remote, 0, 8)?;
    let header_size = u64::from_le_bytes(prefix[..8].try_into().unwrap());
    if header_size == 0 || header_size > HEADER_LIMIT || 8 + header_size > size {
        return Err(format!("{remote} has an invalid safetensors header size").into());
    }
    let header = get_bytes(session, remote, 8, header_size)?;
    file.write_all_at(&prefix, 0)?;
    file.write_all_at(&header, 8)?;
    if primary {
        let header: Value = serde_json::from_slice(&header)
            .map_err(|e| format!("{remote}: invalid safetensors header: {e}"))?;
        let ranges = mirrored_ranges(&header, size, 8 + header_size)
            .map_err(|e| format!("{remote}: {e}"))?;
        for (offset, length) in ranges {
            let mut write = |chunk: &[u8], position: u64| -> Result<(), Failure> {
                Ok(file.write_all_at(chunk, position)?)
            };
            get(session, remote, offset, length, &mut write)?;
        }
    }
    Ok(())
}

/// The primary byte ranges a controller mirrors for a node-held package: the signed index for
/// either canonical form, never payload bytes. `start` is the first data byte after the
/// safetensors header.
fn mirrored_ranges(header: &Value, size: u64, start: u64) -> Result<Vec<(u64, u64)>, String> {
    const MAX_INDEX: u64 = 256 << 20;
    let offsets = |name: &str| -> Option<(u64, u64)> {
        let offsets = header.get(name)?.get("data_offsets")?;
        Some((offsets.get(0)?.as_u64()?, offsets.get(1)?.as_u64()?))
    };
    let bounded = |name: &str| -> Result<Option<(u64, u64)>, String> {
        let Some((begin, end)) = offsets(name) else {
            return Ok(None);
        };
        if end <= begin {
            return Ok(None);
        }
        if end - begin > MAX_INDEX {
            return Err(format!("{name} index exceeds {MAX_INDEX} bytes"));
        }
        if start.checked_add(end).is_none_or(|value| value > size) {
            return Err(format!("{name} index exceeds the package file"));
        }
        Ok(Some((start + begin, end - begin)))
    };
    match header["__metadata__"]["format"].as_str() {
        Some(format) if matches!(format, "flare" | "flare-staging" | "flare-snapshot") => {
            let (begin, mut end) = offsets("flare.metadata").ok_or("not a Flare package")?;
            if format == "flare-snapshot" {
                let (layout_begin, layout_end) =
                    offsets("flare.layout").ok_or("snapshot lacks layout metadata")?;
                if layout_begin != end {
                    return Err("snapshot layout must follow its signed metadata".into());
                }
                end = layout_end;
            }
            if end <= begin || end - begin > MAX_INDEX {
                return Err("invalid model index range".into());
            }
            if start.checked_add(end).is_none_or(|value| value > size) {
                return Err("invalid model index range".into());
            }
            Ok(vec![(start + begin, end - begin)])
        }
        Some(flare_helios::package::container::PAYLOAD_FORMAT) => {
            let mut ranges = Vec::new();
            for name in ["flare.manifest", "flare.locations", "flare.derivations"] {
                if let Some(range) = bounded(name)? {
                    ranges.push(range);
                }
            }
            if ranges.is_empty() {
                return Err("not a Flare package".into());
            }
            Ok(ranges)
        }
        _ => Err("not a Flare package".into()),
    }
}

// ---- following output --------------------------------------------------------------------------

/// Print each session's LOG lines as `[rank R name] line` on stderr until its EXIT, on a thread
/// per rank; each rank's end arrives on the returned channel.
pub fn follow(ranks: Vec<(usize, String, Session)>) -> mpsc::Receiver<RankEnd> {
    let (sender, receiver) = mpsc::channel();
    for (rank, node, mut session) in ranks {
        let sender = sender.clone();
        thread::spawn(move || {
            let result = read_output(&mut session, rank, &node);
            let _ = sender.send(RankEnd { rank, node, result });
        });
    }
    receiver
}

fn read_output(session: &mut Session, rank: usize, node: &str) -> Result<i32, String> {
    session
        .set_read_timeout(Some(ATTACH_TIMEOUT))
        .map_err(|e| e.to_string())?;
    loop {
        match session
            .read_frame()
            .map_err(|e| format!("output lost: {e}"))?
        {
            (frame::LOG, line) if line.is_empty() => {}
            (frame::LOG, line) => {
                eprintln!("[rank {rank} {node}] {}", String::from_utf8_lossy(&line))
            }
            (frame::EXIT, code) => {
                return <[u8; 4]>::try_from(code.as_slice())
                    .map(i32::from_le_bytes)
                    .map_err(|_| String::from("malformed EXIT frame"));
            }
            (frame::ERROR, message) => return Err(String::from_utf8_lossy(&message).into_owned()),
            (tag, _) => return Err(format!("unexpected frame {tag}")),
        }
    }
}

fn report_end(end: &RankEnd) {
    match &end.result {
        Ok(code) => eprintln!("[rank {} {}] exited with status {code}", end.rank, end.node),
        Err(error) => eprintln!("[rank {} {}] {error}", end.rank, end.node),
    }
}

// ---- packages ----------------------------------------------------------------------------------

fn default_group_name(package: &str) -> String {
    if crate::hub::is_repository(Path::new(package)) {
        let (id, _) = crate::hub::repository(package).unwrap();
        return id.rsplit('/').next().unwrap().into();
    }
    Path::new(package)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "group".into())
}

/// Explicit revisions win; otherwise infer one hardware revision from the candidate nodes.
fn repository_revision(
    source: &str,
    options: &crate::cli::HubOptions,
    statuses: &[NodeStatus],
    gpus: &[u32],
) -> Result<String, Failure> {
    let (_, suffix) = crate::hub::repository(source).map_err(|e| e.to_string())?;
    if let Some(revision) = options.revision.as_deref().or(suffix) {
        return Ok(revision.into());
    }
    let revisions: std::collections::BTreeSet<_> = statuses
        .iter()
        .flat_map(|status| &status.gpus)
        .filter(|gpu| gpu.group.is_none() && (gpus.is_empty() || gpus.contains(&gpu.index)))
        .map(|gpu| gpu.revision.as_str())
        .collect();
    if revisions.is_empty() {
        return Err("candidate nodes have no available selected GPUs".into());
    }
    if revisions.len() > 1 {
        return Err("candidate GPUs do not identify one hardware revision; specify --revision or constrain --nodes/--gpus".into());
    }
    Ok(revisions.into_iter().next().unwrap().into())
}

/// Parse `NODE:/ABSOLUTE/PATH`, where NODE is a node name `is_node` accepts; anything else is a
/// local path.
pub fn parse_package_reference(text: &str, is_node: impl Fn(&str) -> bool) -> PackageReference {
    match text.split_once(':') {
        Some((node, path))
            if !node.is_empty()
                && !node.contains('/')
                && path.starts_with('/')
                && is_node(node) =>
        {
            PackageReference::Node {
                node: node.to_owned(),
                path: path.to_owned(),
            }
        }
        _ => PackageReference::Local(PathBuf::from(text)),
    }
}

/// Verify a package and obtain publisher acceptance as `flare run` does.
fn verify(
    primary: &Path,
    args: &crate::cli::Run,
    origin: Option<&str>,
    metadata_only: bool,
) -> Result<(AuthorizedPackage, Publisher), Failure> {
    let mut progress = StartupProgress::silent();
    let loaded = if metadata_only {
        crate::trust::load_metadata(primary, args, &mut progress, origin)
    } else {
        crate::trust::load(primary, args, &mut progress, origin)
    };
    loaded.map_err(|e| Failure::from(e.to_string()))
}

/// A verified package's files and every rank's byte ranges.
fn describe(
    authorized: &AuthorizedPackage,
    primary: &Path,
    publisher: Publisher,
    holder: Holder,
) -> Result<ResolvedPackage, Failure> {
    let rank_needs = runtime_package::rank_needs(authorized);
    let Some(first) = rank_needs.first() else {
        return Err(format!("{} holds no rank bundle", primary.display()).into());
    };
    if rank_needs
        .iter()
        .any(|r| r.architecture != first.architecture)
    {
        return Err(format!("{} mixes GPU architectures", primary.display()).into());
    }
    let primary_name = primary
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("the package path names no file")?
        .to_owned();
    let cluster = runtime_package::package_cluster(authorized);
    let needs = PackageNeeds {
        revision: super::gpu::capability_revision(first.architecture[0], first.architecture[1]),
        multi_host: cluster.is_some(),
        rank_bytes: rank_needs.iter().map(|r| r.device_bytes).collect(),
        reserve_bytes: runtime_package::reserve_bytes(authorized),
    };
    let (world, export) = match cluster {
        Some((world, export)) => (world, export),
        None => (
            1,
            primary_name
                .trim_end_matches(".safetensors")
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect(),
        ),
    };
    let mut files = vec![PackageEntry {
        name: primary_name.clone(),
        size: fs::metadata(primary)?.len(),
    }];
    let root = primary.parent().unwrap_or_else(|| Path::new("."));
    for name in runtime_package::package_companions(primary).map_err(|e| e.to_string())? {
        let size = fs::metadata(root.join(&name))?.len();
        files.push(PackageEntry { name, size });
    }
    let mut ranks = Vec::with_capacity(world);
    if needs.multi_host {
        for rank in 0..world {
            let mut rank_files = Vec::new();
            for file in runtime_package::rank_ranges(authorized, rank).map_err(|e| e.to_string())? {
                rank_files.push(RankFile {
                    name: if file.filename.is_empty() {
                        primary_name.clone()
                    } else {
                        file.filename
                    },
                    size: file.file.metadata()?.len(),
                    ranges: file.ranges,
                });
            }
            ranks.push(rank_files);
        }
    } else {
        // One member runs every rank from the complete package.
        ranks.push(
            files
                .iter()
                .map(|f| RankFile {
                    name: f.name.clone(),
                    size: f.size,
                    ranges: vec![(0, f.size)],
                })
                .collect(),
        );
    }
    Ok(ResolvedPackage {
        export,
        files,
        ranks,
        needs,
        publisher,
        holder,
    })
}

/// A private temporary directory, removed when dropped.
struct Mirror(PathBuf);

impl Mirror {
    fn create() -> Result<Self, Failure> {
        let path = std::env::temp_dir().join(format!(
            "flare-orbit-{}-{}",
            std::process::id(),
            hex(&random(4)?)
        ));
        fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Read only an explicit override; the controller's own binary is never sent implicitly.
fn read_executable(path: Option<&Path>) -> Result<Option<Vec<u8>>, Failure> {
    path.map(|path| {
        fs::read(path).map_err(|e| format!("cannot read executable {}: {e}", path.display()).into())
    })
    .transpose()
}

// ---- formatting --------------------------------------------------------------------------------

/// A group identifier: the cluster name, the UTC launch time, and `suffix` as hex.
pub fn group_id(cluster: &str, seconds: u64, suffix: &[u8]) -> String {
    format!("{cluster}-{}-{}", compact_utc(seconds), hex(suffix))
}

/// `YYYYMMDDTHHMMSSZ` for seconds since the Unix epoch.
pub fn compact_utc(seconds: u64) -> String {
    let (days, time) = (seconds / 86_400, seconds % 86_400);
    // The civil date of a day count since 1970-01-01, in 400-year eras that start on March 1 so
    // that leap days fall at the end of each year.
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        time / 3600,
        time % 3600 / 60,
        time % 60
    )
}

/// The URL rank zero serves, from its `--host` and `--port` options or `flare run`'s defaults;
/// an unspecified bind address is shown as its node's host.
pub fn serving_url(rank_zero: &[String], node_address: &str) -> String {
    let mut host = "127.0.0.1".to_owned();
    let mut port = "8000".to_owned();
    let mut arguments = rank_zero.iter();
    while let Some(argument) = arguments.next() {
        if let Some(value) = argument.strip_prefix("--host=") {
            host = value.to_owned();
        } else if let Some(value) = argument.strip_prefix("--port=") {
            port = value.to_owned();
        } else if argument == "--host" {
            if let Some(value) = arguments.next() {
                host = value.clone();
            }
        } else if argument == "--port"
            && let Some(value) = arguments.next()
        {
            port = value.clone();
        }
    }
    let host = match host.parse::<IpAddr>() {
        Ok(ip) if ip.is_unspecified() => node_address
            .rsplit_once(':')
            .map_or(node_address, |(host, _)| host)
            .to_owned(),
        Ok(IpAddr::V6(ip)) => format!("[{ip}]"),
        _ => host,
    };
    format!("http://{host}:{port}")
}

fn describe_gpus(status: &NodeStatus, uuids: &[String]) -> String {
    if uuids.is_empty() {
        return "the node's GPUs".into();
    }
    uuids
        .iter()
        .map(|uuid| {
            status
                .gpus
                .iter()
                .find(|g| &g.uuid == uuid)
                .map_or_else(|| uuid.clone(), |g| format!("gpu {} {}", g.index, g.name))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_group(group: Option<&GroupStatus>) -> String {
    match group {
        None => "-".into(),
        Some(g) if g.running => format!(
            "{} rank {} running {}",
            g.id,
            g.rank,
            uptime(g.uptime_seconds)
        ),
        Some(g) => format!(
            "{} rank {} exited with status {}",
            g.id,
            g.rank,
            g.exit
                .map_or_else(|| "unknown".into(), |code| code.to_string())
        ),
    }
}

fn uptime(seconds: u64) -> String {
    match seconds {
        s if s >= 86_400 => format!("{}d{:02}h", s / 86_400, s % 86_400 / 3600),
        s if s >= 3600 => format!("{}h{:02}m", s / 3600, s % 3600 / 60),
        s => format!("{}m{:02}s", s / 60, s % 60),
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

// ---- helpers -----------------------------------------------------------------------------------

fn absolute_path(path: &Path) -> Result<PathBuf, Failure> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

fn saved_group_state(spec: &GroupSpec, statuses: &[Result<NodeStatus, Failure>]) -> &'static str {
    if statuses
        .iter()
        .filter_map(|status| status.as_ref().ok())
        .any(|status| status.running().any(|group| group.name == spec.name))
    {
        "running"
    } else if statuses.iter().any(Result::is_err) {
        "unknown (node unreachable)"
    } else {
        "stopped"
    }
}

/// Prefix an error with the node it concerns.
fn at(node: &str, error: Failure) -> Failure {
    format!("{node}: {error}").into()
}

/// Every value, or one error listing every failure.
fn all<T>(results: Vec<Result<T, Failure>>) -> Result<Vec<T>, Failure> {
    let mut values = Vec::with_capacity(results.len());
    let mut errors = Vec::new();
    for result in results {
        match result {
            Ok(value) => values.push(value),
            Err(error) => errors.push(error.to_string()),
        }
    }
    if errors.is_empty() {
        Ok(values)
    } else {
        Err(errors.join("; ").into())
    }
}

/// Apply `work` to every item, each on its own thread; results keep the items' order.
fn parallel<T: Sync, R: Send>(items: &[T], work: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = items
            .iter()
            .enumerate()
            .map(|(index, item)| scope.spawn(move || work(index, item)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

/// [`parallel`] with exclusive access to each item.
fn parallel_mut<T: Send, R: Send>(
    items: &mut [T],
    work: impl Fn(usize, &mut T) -> R + Sync,
) -> Vec<R> {
    thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = items
            .iter_mut()
            .enumerate()
            .map(|(index, item)| scope.spawn(move || work(index, item)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::agent::tests::{harness, pair_accept_first};
    use std::sync::Arc;

    #[test]
    fn download_requests_carry_public_profiles_not_controller_paths() {
        use flare_helios::runtime_package::Publisher;
        use ring::signature::{Ed25519KeyPair, KeyPair};

        let temp = tempfile::tempdir().unwrap();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
        let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        let publisher = Publisher {
            name: "Controller".into(),
            namespace: "local-development".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: true,
        };
        let profile = temp.path().join("publisher.json");
        fs::write(
            &profile,
            serde_json::to_vec(&vec![publisher.clone()]).unwrap(),
        )
        .unwrap();
        let trusted_key =
            Ed25519KeyPair::from_pkcs8(
                Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                    .unwrap()
                    .as_ref(),
            )
            .unwrap();
        let trusted = Publisher {
            name: "Remembered".into(),
            namespace: "local-development".into(),
            public_key: trusted_key.public_key().as_ref().try_into().unwrap(),
            development: true,
        };
        let trust_store = temp.path().join("trust.toml");
        let public_key: String = trusted
            .public_key
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        fs::write(
            &trust_store,
            format!(
                "[[trusted]]\nfingerprint = \"{}\"\nnamespace = \"{}\"\npublic_key = \"{}\"\nname = \"{}\"\nidentity = \"profile\"\n",
                trusted.fingerprint(),
                trusted.namespace,
                public_key,
                trusted.name,
            ),
        )
        .unwrap();
        let controller_cache = temp.path().join("controller-cache");
        let controller_hf = temp.path().join("controller-hf");
        let options = crate::cli::HubOptions {
            cache_dir: Some(controller_hf.clone()),
            model_cache: Some(controller_cache.clone()),
            token_file: Some(temp.path().join("token")),
            ..Default::default()
        };
        let request = download_request(
            "owner/model",
            "gb202",
            &options,
            std::slice::from_ref(&profile),
            Some(&trust_store),
            None,
        )
        .unwrap();
        assert_eq!(request.profiles, vec![publisher.clone(), trusted]);
        let conflict_store = temp.path().join("conflicting-trust.toml");
        let conflict_key: String = publisher
            .public_key
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        fs::write(
            &conflict_store,
            format!(
                "[[trusted]]\nfingerprint = \"{}\"\nnamespace = \"{}\"\npublic_key = \"{}\"\nname = \"Remembered renamed\"\nidentity = \"profile\"\n",
                publisher.fingerprint(),
                publisher.namespace,
                conflict_key,
            ),
        )
        .unwrap();
        let conflict = download_request(
            "owner/model",
            "gb202",
            &options,
            std::slice::from_ref(&profile),
            Some(&conflict_store),
            None,
        )
        .unwrap();
        assert_eq!(
            conflict.profiles,
            vec![publisher.clone()],
            "fresh explicit profile must take precedence over a remembered entry"
        );
        assert!(request.options.cache_dir.is_none());
        assert!(request.options.model_cache.is_none());
        assert!(request.options.token_file.is_none());
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(!encoded.contains(&*profile.to_string_lossy()));
        assert!(!encoded.contains(&*controller_cache.to_string_lossy()));
        assert!(!encoded.contains(&*controller_hf.to_string_lossy()));
        let decoded: super::super::agent::DownloadRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.profiles.len(), 2);
        assert!(decoded.options.revision.as_deref() == Some("gb202"));
    }

    #[test]
    fn reading_an_executable_requires_an_explicit_path_on_every_controller_platform() {
        assert!(read_executable(None).unwrap().is_none());
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("flare");
        assert!(read_executable(Some(&path)).is_err());
        fs::write(&path, b"chosen executable").unwrap();
        assert_eq!(
            read_executable(Some(&path)).unwrap().unwrap(),
            b"chosen executable"
        );
    }

    #[test]
    fn launching_uses_each_nodes_installed_executable_without_uploading_a_binary() {
        check_launch(None);
    }

    #[test]
    fn launching_uploads_and_runs_an_explicit_executable_override() {
        check_launch(Some(
            b"#!/bin/sh\necho \"$0\"\necho \"$@\"\nif [ -f \"$2\" ]; then od -An -tx1 -j 40000 -N 4 \"$2\"; fi\nexit 0\n",
        ));
    }

    /// A controller over two in-process agents paired with its key, scripts as the executables,
    /// and a two-rank package whose "primary" is a plain file: rank 0 reads all of it, rank 1
    /// two ranges. Launching exercises PUT, RUN and output following, while staging exercises
    /// TRANSFER and FETCH. The rank scripts report arguments without loading a model.
    fn check_launch(executable: Option<&[u8]>) {
        let zero = harness(21);
        let one = crate::orbit::agent::tests::Harness {
            controller: Key::from_seed(&[21; 32]).unwrap(),
            ..harness(22)
        };
        let record = {
            let mut session = pair_accept_first(&zero);
            session.request(frame::STATUS, &json!({})).unwrap();
            pair_accept_first(&one);
            crate::orbit::agent::tests::record(&zero.controller)
        };
        let temp = tempfile::tempdir().unwrap();
        let node = |h: &crate::orbit::agent::tests::Harness, name: &str| Node {
            name: name.into(),
            fingerprint: h.agent.key.public().fingerprint(),
            public_key: h.agent.key.public().to_hex(),
            address: h.address.clone(),
        };
        let nodes = vec![node(&zero, "zero"), node(&one, "one")];
        let mut orbit = Orbit {
            controllers: Some(record),
            node: nodes.clone(),
            cluster: Default::default(),
            group: Vec::new(),
        };
        orbit.cluster.insert(
            "pair".into(),
            crate::orbit::store::Cluster {
                nodes: vec!["zero".into(), "one".into()],
            },
        );
        let mut controller = Controller::new(
            Key::from_seed(&[21; 32]).unwrap(),
            orbit,
            temp.path().join("orbit.toml"),
        );

        let primary = temp.path().join("model.safetensors");
        let bytes: Vec<u8> = (0..50_000u32).map(|i| (i % 241) as u8).collect();
        fs::write(&primary, &bytes).unwrap();
        let package = ResolvedPackage {
            export: "export".into(),
            files: vec![PackageEntry {
                name: "model.safetensors".into(),
                size: bytes.len() as u64,
            }],
            ranks: vec![
                vec![RankFile {
                    name: "model.safetensors".into(),
                    size: bytes.len() as u64,
                    ranges: vec![(0, bytes.len() as u64)],
                }],
                vec![RankFile {
                    name: "model.safetensors".into(),
                    size: bytes.len() as u64,
                    ranges: vec![(0, 4096), (40_000, 8_000)],
                }],
            ],
            publisher: Publisher {
                name: "Test".into(),
                namespace: "test-publisher".into(),
                public_key: [1; 32],
                development: false,
            },
            holder: Holder::Controller {
                primary: primary.clone(),
            },
            needs: PackageNeeds {
                revision: "gb10".into(),
                multi_host: true,
                rank_bytes: vec![1 << 30, 1 << 30],
                reserve_bytes: 0,
            },
        };
        let single_host = ResolvedPackage {
            ranks: vec![package.ranks[0].clone()],
            needs: PackageNeeds {
                multi_host: false,
                ..package.needs.clone()
            },
            ..package.clone()
        };
        let launch = controller
            .launch(
                "single",
                &nodes[..1],
                &single_host,
                executable,
                &LaunchOptions {
                    name: "single".into(),
                    placements: vec![vec![]],
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(launch.grant.single_host);
        assert!(launch.grant.rendezvous.is_empty());
        let events = follow(
            launch
                .ranks
                .into_iter()
                .map(|(node, session)| (0, node.name, session))
                .collect(),
        );
        assert_eq!(
            events
                .recv_timeout(Duration::from_secs(20))
                .unwrap()
                .result
                .unwrap(),
            0
        );
        controller.save().unwrap();
        controller.remove(&launch.grant.group).unwrap();

        let launch = controller
            .launch(
                "pair",
                &nodes,
                &package,
                executable,
                &LaunchOptions {
                    name: "pair".into(),
                    placements: vec![
                        vec!["GPU-aaaaaaaa-0000-0000-0000-000000000000".into()],
                        vec!["GPU-aaaaaaaa-0000-0000-0000-000000000000".into()],
                    ],
                    serving: vec![],
                    rank_zero: vec!["--port".into(), "8001".into()],
                    roce_ring: true,
                    transfer_streams: Some(8),
                    stage_only: false,
                    roce_chain: false,
                },
            )
            .unwrap();
        assert_eq!(launch.grant.members.len(), 2);
        assert_eq!(launch.grant.rendezvous, "127.0.0.1:29800");
        assert_eq!(launch.grant.name, "pair");
        assert!(
            launch
                .grant
                .members
                .iter()
                .all(|m| m.gpus == ["GPU-aaaaaaaa-0000-0000-0000-000000000000"])
        );
        let events = follow(
            launch
                .ranks
                .into_iter()
                .enumerate()
                .map(|(rank, (node, session))| (rank, node.name, session))
                .collect(),
        );
        let mut exits = BTreeMap::new();
        for _ in 0..2 {
            let end = events.recv_timeout(Duration::from_secs(20)).unwrap();
            exits.insert(end.rank, end.result.unwrap());
        }
        assert_eq!(exits, BTreeMap::from([(0, 0), (1, 0)]));

        // Rank zero holds the complete package. Peer processes fetch their own shards during
        // serving bootstrap; the controller starts them without a local package.
        let complete = zero.agent.cache.join(format!(
            "packages/{}/model.safetensors",
            launch.grant.complete_directory()
        ));
        assert_eq!(fs::read(&complete).unwrap(), bytes);
        let partial = one.agent.cache.join(format!(
            "packages/{}/model.safetensors",
            launch.grant.cache_directory(1)
        ));
        assert!(!partial.exists());
        for harness in [&zero, &one] {
            let selected = match executable {
                Some(bytes) => {
                    let binary = harness
                        .agent
                        .cache
                        .join("bin")
                        .join(cache::binary_name(bytes));
                    assert_eq!(fs::read(&binary).unwrap(), bytes);
                    binary
                }
                None => {
                    assert_eq!(
                        fs::read_dir(harness.agent.cache.join("bin"))
                            .unwrap()
                            .count(),
                        0
                    );
                    harness.temp.path().join("installed-flare")
                }
            };
            let mut session = crate::orbit::agent::tests::connect(harness, &harness.controller);
            session
                .request(frame::ATTACH, &json!({"group": launch.grant.group}))
                .unwrap();
            let (lines, exit) = crate::orbit::agent::tests::collect(&mut session);
            assert_eq!(exit, 0);
            if std::ptr::eq(harness, &one) {
                assert!(
                    lines
                        .iter()
                        .any(|line| line.starts_with("run --package-cache ")
                            && line.contains("--rank 1 --rendezvous 127.0.0.1:29800")),
                    "{lines:?}"
                );
            }
            assert!(
                lines.contains(&selected.to_string_lossy().into_owned()),
                "{lines:?}"
            );
        }

        controller.orbit.save_group(GroupSpec {
            name: "pair".into(),
            package: primary.to_string_lossy().into_owned(),
            ..GroupSpec::default()
        });
        controller.save().unwrap();
        controller.remove(&launch.grant.group).unwrap();
        assert!(controller.orbit.group("pair").is_some());
        for node in &nodes {
            assert!(
                controller
                    .node_status(node, None)
                    .unwrap()
                    .groups
                    .is_empty()
            );
        }
        controller.remove("pair").unwrap();
        assert!(
            Orbit::load(&controller.orbit_path)
                .unwrap()
                .group
                .is_empty()
        );
        assert!(complete.is_file());
        assert!(!partial.exists());

        // Staging places and verifies the package without executables or ranks.
        let staged = ResolvedPackage {
            export: "export2".into(),
            ..package.clone()
        };
        let launch = controller
            .launch(
                "push",
                &nodes,
                &staged,
                None,
                &LaunchOptions {
                    name: "push-model".into(),
                    placements: vec![vec![], vec![]],
                    stage_only: true,
                    ..LaunchOptions::default()
                },
            )
            .unwrap();
        assert!(launch.ranks.is_empty());
        assert_eq!(launch.grant.name, "push-model");
        assert_eq!(
            fs::read(zero.agent.cache.join(format!(
                "packages/{}/model.safetensors",
                launch.grant.complete_directory()
            )))
            .unwrap(),
            bytes
        );
        let staged_rank = one.agent.cache.join(format!(
            "packages/{}/model.safetensors",
            launch.grant.cache_directory(1)
        ));
        assert_eq!(
            fs::read(&staged_rank).unwrap()[40_000..48_000],
            bytes[40_000..48_000]
        );
        assert!(staged_rank.with_file_name(".verified").is_file());
        assert_eq!(one.verified.lock().unwrap().len(), 1);

        // A source outside the ring stages once on rank zero, which serves the peer over fabric.
        let outside = crate::orbit::agent::tests::Harness {
            controller: Key::from_seed(&[21; 32]).unwrap(),
            ..harness(23)
        };
        pair_accept_first(&outside);
        let external = ResolvedPackage {
            export: "external".into(),
            holder: Holder::Node {
                node: node(&outside, "outside"),
                path: primary.to_string_lossy().into_owned(),
            },
            ..package.clone()
        };
        let staged = controller
            .launch(
                "push",
                &nodes,
                &external,
                None,
                &LaunchOptions {
                    name: "external".into(),
                    placements: vec![vec![], vec![]],
                    stage_only: true,
                    roce_ring: true,
                    ..LaunchOptions::default()
                },
            )
            .unwrap();
        let complete = zero.agent.cache.join(format!(
            "packages/{}/model.safetensors",
            staged.grant.complete_directory()
        ));
        assert_eq!(fs::read(&complete).unwrap(), bytes);
        assert!(complete.with_file_name(".verified").is_file());
        let partial = one.agent.cache.join(format!(
            "packages/{}/model.safetensors",
            staged.grant.cache_directory(1)
        ));
        assert_eq!(
            fs::read(&partial).unwrap()[40_000..48_000],
            bytes[40_000..48_000]
        );
        assert!(partial.with_file_name(".verified").is_file());

        // A second launch is refused while a group runs; after exit it reuses the cache.
        let again = controller
            .launch(
                "pair",
                &nodes,
                &package,
                executable,
                &LaunchOptions {
                    name: "pair".into(),
                    placements: vec![vec![], vec![]],
                    ..LaunchOptions::default()
                },
            )
            .unwrap();
        let (zero_session, one_session): (Vec<_>, Vec<_>) =
            again.ranks.into_iter().partition(|(n, _)| n.name == "zero");
        drop(zero_session);
        drop(one_session);
        let _ = Arc::clone(&zero.agent);
    }

    #[test]
    fn default_group_names_preserve_repository_names_and_omit_revisions() {
        assert_eq!(
            default_group_name("owner/Qwen3.8-Flash-Next-FLARE"),
            "Qwen3.8-Flash-Next-FLARE"
        );
        assert_eq!(
            default_group_name("owner/Qwen3.8-Flash-Next-FLARE@release/gb10"),
            "Qwen3.8-Flash-Next-FLARE"
        );
        assert_eq!(default_group_name("/models/qwen3.8.safetensors"), "qwen3.8");
    }

    #[test]
    fn repository_revisions_follow_candidate_gpus_and_explicit_values() {
        let status = |revision: &str, index: u32, group: Option<&str>| {
            serde_json::from_value::<NodeStatus>(json!({
                "fingerprint": "node", "free_bytes": 0, "controllers_revision": 1,
                "gpus": [{"index": index, "uuid": "GPU", "name": "GPU", "revision": revision,
                    "memory_total": 0, "memory_free": 0, "group": group}]
            }))
            .unwrap()
        };
        let options = crate::cli::HubOptions::default();
        let sparks = vec![status("gb10", 0, None), status("gb10", 0, None)];
        assert_eq!(
            repository_revision("owner/model", &options, &sparks, &[]).unwrap(),
            "gb10"
        );
        assert_eq!(
            repository_revision("owner/model@release", &options, &[], &[]).unwrap(),
            "release"
        );
        let explicit = crate::cli::HubOptions {
            revision: Some("chosen".into()),
            ..Default::default()
        };
        assert_eq!(
            repository_revision("owner/model@release", &explicit, &[], &[]).unwrap(),
            "chosen"
        );
        let mixed = vec![status("gb10", 0, None), status("gb202", 1, None)];
        assert!(repository_revision("owner/model", &options, &mixed, &[]).is_err());
        assert_eq!(
            repository_revision("owner/model", &options, &mixed, &[0]).unwrap(),
            "gb10"
        );
        let occupied = vec![status("gb202", 0, Some("running")), status("gb10", 0, None)];
        assert_eq!(
            repository_revision("owner/model", &options, &occupied, &[]).unwrap(),
            "gb10"
        );
        assert!(repository_revision("owner/model", &options, &[], &[]).is_err());
    }

    #[test]
    fn launch_times_format_as_compact_utc_in_group_identifiers() {
        assert_eq!(compact_utc(0), "19700101T000000Z");
        assert_eq!(compact_utc(951_782_400), "20000229T000000Z");
        assert_eq!(compact_utc(1_792_423_800), "20261019T153000Z");
        assert_eq!(compact_utc(4_107_628_799), "21000301T235959Z");
        assert_eq!(
            group_id("sparks", 1_792_423_800, &[0x3f, 0x9c]),
            "sparks-20261019T153000Z-3f9c"
        );
    }

    #[test]
    fn package_references_name_an_orbit_node_and_an_absolute_path() {
        let is_node = |name: &str| name == "maxwell";
        assert_eq!(
            parse_package_reference("maxwell:/data/glm.safetensors", is_node),
            PackageReference::Node {
                node: "maxwell".into(),
                path: "/data/glm.safetensors".into(),
            }
        );
        for local in [
            "/data/glm.safetensors",
            "glm.safetensors",
            "hertz:/data/glm.safetensors",
            "maxwell:glm.safetensors",
            "./maxwell:/glm.safetensors",
        ] {
            assert_eq!(
                parse_package_reference(local, is_node),
                PackageReference::Local(local.into())
            );
        }
    }

    #[test]
    fn rank_zero_urls_follow_its_bind_options() {
        let options = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(serving_url(&[], "maxwell:29900"), "http://127.0.0.1:8000");
        assert_eq!(
            serving_url(
                &options(&["--host", "0.0.0.0", "--port=8001"]),
                "maxwell:29900"
            ),
            "http://maxwell:8001"
        );
        assert_eq!(
            serving_url(&options(&["--host=::"]), "[fd00::1]:29900"),
            "http://[fd00::1]:8000"
        );
        assert_eq!(
            serving_url(
                &options(&["--host", "10.0.0.2", "--port", "9000"]),
                "maxwell:29900"
            ),
            "http://10.0.0.2:9000"
        );
    }

    #[test]
    fn mirrored_ranges_follow_each_package_carrier() {
        let payload = json!({
            "__metadata__": {"format": "flare-payload"},
            "flare.manifest": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]},
            "flare.locations": {"dtype": "U8", "shape": [20], "data_offsets": [10, 30]},
            "flare.derivations": {"dtype": "U8", "shape": [5], "data_offsets": [30, 35]},
        });
        assert_eq!(
            mirrored_ranges(&payload, 1000, 8).unwrap(),
            vec![(8, 10), (18, 20), (38, 5)]
        );

        let packed = json!({
            "__metadata__": {"format": "flare"},
            "flare.metadata": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]},
        });
        assert_eq!(mirrored_ranges(&packed, 1000, 8).unwrap(), vec![(8, 16)]);

        let snapshot = json!({
            "__metadata__": {"format": "flare-snapshot"},
            "flare.metadata": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]},
            "flare.layout": {"dtype": "U8", "shape": [24], "data_offsets": [16, 40]},
        });
        assert_eq!(mirrored_ranges(&snapshot, 1000, 8).unwrap(), vec![(8, 40)]);

        let unknown = json!({"__metadata__": {"format": "other"}});
        assert!(mirrored_ranges(&unknown, 1000, 8).is_err());
        let empty = json!({"__metadata__": {"format": "flare-payload"}});
        assert!(mirrored_ranges(&empty, 1000, 8).is_err());
    }

    #[test]
    fn companion_paths_allow_nesting_but_not_traversal() {
        assert!(companion_path_is_valid(
            "weights/tp4/rank-0/layer.safetensors"
        ));
        assert!(companion_path_is_valid(
            "deployments/abc/rank-1/graphs.safetensors"
        ));
        assert!(!companion_path_is_valid(""));
        assert!(!companion_path_is_valid("/absolute.safetensors"));
        assert!(!companion_path_is_valid("../escape.safetensors"));
        assert!(!companion_path_is_valid("weights/../../escape.safetensors"));
        assert!(!companion_path_is_valid("weights//empty.safetensors"));
        assert!(!companion_path_is_valid(".hidden"));
    }

    #[test]
    fn mirror_package_reads_a_distributed_directory_package() {
        use flare_helios::package::{
            container::{Source, Writer, PAYLOAD_FORMAT},
            schema::{self, wire},
        };
        use prost::Message;

        let zero = harness(31);
        let record = {
            let mut session = pair_accept_first(&zero);
            session.request(frame::STATUS, &json!({})).unwrap();
            crate::orbit::agent::tests::record(&zero.controller)
        };
        let temp = tempfile::tempdir().unwrap();
        let node = Node {
            name: "zero".into(),
            fingerprint: zero.agent.key.public().fingerprint(),
            public_key: zero.agent.key.public().to_hex(),
            address: zero.address.clone(),
        };
        let nodes = vec![node.clone()];
        let mut orbit = Orbit {
            controllers: Some(record),
            node: nodes.clone(),
            cluster: Default::default(),
            group: Vec::new(),
        };
        orbit.cluster.insert(
            "pair".into(),
            crate::orbit::store::Cluster {
                nodes: vec!["zero".into()],
            },
        );
        let controller = Controller::new(
            Key::from_seed(&[31; 32]).unwrap(),
            orbit,
            temp.path().join("orbit.toml"),
        );

        let package = temp.path().join("package");
        let nested = package.join("weights/tp4/rank-0/layer.safetensors");
        fs::create_dir_all(nested.parent().unwrap()).unwrap();
        let payload = b"layer payload".to_vec();
        let mut companion = Writer::new(File::create(&nested).unwrap(), PAYLOAD_FORMAT).unwrap();
        companion
            .add("flare.payload", payload.len() as u64, Source::Bytes(payload))
            .unwrap();
        companion.finish().unwrap();

        let locations = wire::LocationSet {
            version: schema::FORMAT_VERSION,
            locations: vec![wire::PayloadLocation {
                payload_id: vec![7; 32],
                file: "weights/tp4/rank-0/layer.safetensors".into(),
                tensor: "flare.payload".into(),
            }],
            available_deployments: Vec::new(),
        };
        let primary = package.join("model.safetensors");
        let mut writer = Writer::new(File::create(&primary).unwrap(), PAYLOAD_FORMAT).unwrap();
        writer
            .add("flare.manifest", 8, Source::Bytes(b"manifest".to_vec()))
            .unwrap();
        writer
            .add(
                "flare.locations",
                locations.encoded_len() as u64,
                Source::Bytes(locations.encode_to_vec()),
            )
            .unwrap();
        writer.finish().unwrap();

        let mirror = temp.path().join("mirror");
        fs::create_dir_all(&mirror).unwrap();
        let mirrored = controller
            .mirror_package(&node, primary.to_str().unwrap(), &mirror)
            .unwrap();
        assert_eq!(
            flare_helios::runtime_package::package_companions(&mirrored).unwrap(),
            vec!["weights/tp4/rank-0/layer.safetensors"]
        );
        assert!(mirror
            .join("weights/tp4/rank-0/layer.safetensors")
            .is_file());
        assert_eq!(
            fs::metadata(mirror.join("weights/tp4/rank-0/layer.safetensors"))
                .unwrap()
                .len(),
            fs::metadata(&nested).unwrap().len()
        );
    }
}
