use flare_helios::execution::roce_fabric::Shape;
use std::{
    net::{IpAddr, SocketAddr},
    num::NonZeroUsize,
    path::PathBuf,
};

use clap::{Args, Parser, Subcommand, builder::PossibleValuesParser, builder::TypedValueParser};
use flare_helios::{
    engram::Storage,
    offload::{PrefixCacheOffload, Transport, parse_size},
    runtime_package::{KvCacheLimit, LoadOptions, MemoryLimit},
};

#[derive(Debug, Parser)]
#[command(
    name = "flare",
    version,
    about = "Serve prepared models on NVIDIA GPUs"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve a local package or download a prepared package from Hugging Face.
    Run(Box<Run>),
    /// Download a prepared package from the hardware's Hugging Face revision.
    Download(Download),
    /// Summarize a package's metadata, with optional memory, graph, and frontend details.
    Inspect(Inspect),
    #[command(hide = true)]
    HardwareRevision {
        #[arg(value_parser = clap::value_parser!(i32).range(1..))]
        gpu_count: i32,
    },
    /// Run this host's agent, which pairs with controllers and supervises serving ranks.
    Agent(AgentArgs),
    /// Control an orbit of agents from this machine.
    Orbit(OrbitArgs),
    #[cfg(unix)]
    #[command(hide = true)]
    CheckpointTransport {
        #[arg(long)]
        socket: PathBuf,
    },
    /// Manage remembered publisher trust.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
}

#[derive(Debug, Args, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct HubOptions {
    /// Hub revision; defaults to gb10 or gb202 detected from the selected GPUs.
    #[arg(long, value_name = "REVISION", help_heading = "Download options")]
    pub revision: Option<String>,

    /// Package filename; required when the revision contains several .safetensors files.
    #[arg(long, value_name = "FILE", help_heading = "Download options")]
    pub filename: Option<String>,

    /// Hugging Face hub cache [default: ~/.cache/huggingface/hub].
    #[arg(long, value_name = "PATH", help_heading = "Download options")]
    pub cache_dir: Option<PathBuf>,

    /// Hugging Face access token file; omit for public repositories.
    #[arg(long, value_name = "PATH", help_heading = "Download options")]
    pub token_file: Option<PathBuf>,

    /// Use only cached files, without contacting Hugging Face.
    #[arg(long, help_heading = "Download options")]
    pub offline: bool,
}

#[derive(Debug, Args)]
pub struct Inspect {
    /// Local prepared .safetensors package.
    #[arg(value_name = "PACKAGE")]
    pub package: PathBuf,

    /// Print the selected sections as JSON.
    #[arg(long)]
    pub json: bool,

    /// Disable terminal styling.
    #[arg(long, conflicts_with = "json")]
    pub plain: bool,

    /// Include allocation addresses, reservations, and execution buffer views.
    #[arg(long)]
    pub memory_map: bool,

    /// Include graph capacities, timings, and node counts for every selected rank.
    #[arg(long)]
    pub graphs: bool,

    /// Include tokenizer, parser, vision, asset, and auxiliary table details.
    #[arg(long)]
    pub frontend: bool,

    /// Include every detail section and the complete serving contract.
    #[arg(long)]
    pub all: bool,

    /// Read only primary-file metadata without checking signatures or opening companions.
    #[arg(long, conflicts_with_all = ["publisher_keys", "trust_store"])]
    pub unverified: bool,

    /// Verify only this rank of a multi-host package whose copy holds that rank's bytes alone.
    #[arg(long, value_name = "N")]
    pub rank: Option<usize>,

    /// Additional publisher verification profiles (JSON).
    /// Development packages default to ~/.config/flare/development-publisher.json.
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    pub publisher_keys: Vec<PathBuf>,

    /// Persistent trust file whose remembered keys also verify [default: ~/.config/flare/trust.toml].
    #[arg(long, value_name = "PATH")]
    pub trust_store: Option<PathBuf>,
}

#[derive(Debug, Args)]
#[command(args_override_self = true)]
pub struct Download {
    /// Hugging Face repository containing prepared Flare packages, optionally ORG/NAME@REVISION.
    #[arg(value_name = "REPOSITORY")]
    pub repository: String,

    /// GPUs used to detect the hardware revision; unnecessary with --revision.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        default_value = "0", value_parser = clap::value_parser!(i32).range(0..))]
    pub gpus: Vec<i32>,

    #[command(flatten)]
    pub hub: HubOptions,
}

#[derive(Debug, Args)]
pub struct AgentArgs {
    #[command(subcommand)]
    pub command: Option<AgentCommand>,

    /// Address and port to listen on.
    #[arg(long, value_name = "IP:PORT", default_value = "0.0.0.0:29900")]
    pub listen: String,
}

#[derive(Debug, Subcommand)]
pub enum AgentCommand {
    /// Pair this machine with resident agents for --hosts or Orbit launches.
    Pair {
        #[arg(required = true, value_name = "HOST")]
        hosts: Vec<String>,
        #[arg(long, value_name = "CODE")]
        code: Option<String>,
        #[arg(long, value_name = "PATH")]
        agents_file: Option<PathBuf>,
        #[arg(long, value_name = "PATH")]
        controller_key: Option<PathBuf>,
    },
    /// Install and restart this host's agent service, and prepare controller pairing.
    Install {
        /// One-time code a controller presents to pair.
        #[arg(long, value_name = "CODE")]
        join_code: Option<String>,
        /// Pair the first controller that connects, without a code.
        #[arg(long)]
        accept_first: bool,
        /// Address and port the installed agent listens on.
        #[arg(long, value_name = "IP:PORT", default_value = "0.0.0.0:29900")]
        listen: String,
    },
    /// Remove this host's authorized controllers and pairing codes; keep its key and cache.
    Reset,
}

#[derive(Debug, Args)]
pub struct OrbitArgs {
    #[command(subcommand)]
    pub command: OrbitCommand,

    /// The orbit file listing clusters and nodes [default: ~/.config/flare/orbit.toml].
    #[arg(long, value_name = "PATH", global = true)]
    pub orbit_file: Option<PathBuf>,

    /// This controller's private key [default: ~/.config/flare/orbit/controller.key].
    #[arg(long, value_name = "PATH", global = true)]
    pub key_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum OrbitCommand {
    /// Pair this controller with the agents on HOSTS.
    Pair {
        #[arg(required = true, value_name = "HOST")]
        hosts: Vec<String>,
        /// The join code shown when the agent was installed.
        #[arg(long, value_name = "CODE")]
        code: Option<String>,
    },
    /// Manage the controllers a paired agent trusts.
    Controllers {
        #[command(subcommand)]
        command: ControllersCommand,
    },
    /// Show each node's capacity, groups, and package cache.
    Status {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
    /// Launch a package across nodes or start a saved group.
    Run(Box<OrbitRun>),
    /// Show the live metrics dashboard for every group and GPU on the cluster's nodes.
    Dashboard {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
    /// List saved groups and retained runs on the cluster's nodes.
    Ls {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
    /// Follow retained and live rank output; Ctrl-C detaches without stopping ranks.
    Attach {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
        /// Only the group with this name or identifier.
        #[arg(value_name = "GROUP")]
        group: Option<String>,
    },
    /// Stop running groups.
    Stop {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
        /// Only the group with this name or identifier.
        #[arg(value_name = "GROUP")]
        group: Option<String>,
    },
    /// Remove a stopped group's saved definition and retained run history.
    Rm {
        /// Group name or exact launch identifier; a launch identifier keeps its saved definition.
        group: String,
    },
    /// Remove a node from the orbit file.
    Forget {
        #[arg(value_name = "NODE")]
        node: String,
    },
    /// Stage package data on nodes without starting ranks.
    Push(Box<OrbitPush>),
    /// List what every node's package cache holds.
    Packages {
        #[arg(long, value_name = "NAME")]
        cluster: Option<String>,
    },
}

#[derive(Debug, Args)]
pub struct OrbitPush {
    /// A local package path or NODE:/absolute/path on a node.
    #[arg(value_name = "PACKAGE")]
    pub package: String,

    /// Target nodes: one per rank in rank order, or any number to receive the complete package.
    #[arg(
        long,
        value_name = "NODE[,NODE...]",
        value_delimiter = ',',
        required = true
    )]
    pub to: Vec<String>,

    /// Stream package data through fabric interfaces in the target nodes' ring order.
    #[arg(long)]
    pub roce_ring: bool,

    /// Stream package data through fabric interfaces along the target nodes' chain order.
    #[arg(long, conflicts_with = "roce_ring")]
    pub roce_chain: bool,

    /// Parallel TCP streams per destination file over the fabric [default: 8].
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(1..=64))]
    pub transfer_streams: Option<u8>,

    #[command(flatten)]
    pub serving: ServingOptions,
}

impl OrbitPush {
    /// What package verification and publisher acceptance need, as `flare run` takes them.
    pub fn run_args(&self) -> Run {
        Run {
            package: Some(PathBuf::from(&self.package)),
            host: "127.0.0.1".parse().unwrap(),
            port: 8000,
            served_model_name: None,
            roce_ring: self.roce_ring,
            roce_chain: self.roce_chain,
            rendezvous: None,
            rank: None,
            hosts: Vec::new(),
            agents_file: None,
            controller_key: None,
            transfer_streams: self.transfer_streams,
            package_cache: None,
            stage_only: false,
            api_key_file: None,
            scheduler_policy: "balanced".into(),
            serving: self.serving.clone(),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum ControllersCommand {
    /// Authorize another controller on every paired node.
    Add {
        name: String,
        /// The controller's public key, in hex.
        key: String,
    },
    /// Revoke a controller on every paired node.
    Remove {
        /// The controller's public key, in hex.
        key: String,
    },
}

#[derive(Debug, Args)]
#[command(args_override_self = true)]
pub struct OrbitRun {
    /// A Hugging Face repository, local package path, or NODE:/absolute/path; omitted when --name names a
    /// [[group]] of the orbit file, which supplies it.
    #[arg(value_name = "PACKAGE")]
    pub package: Option<String>,

    /// Cluster whose nodes serve the ranks in order; without it placement chooses from every
    /// node of the orbit.
    #[arg(long, value_name = "NAME")]
    pub cluster: Option<String>,

    /// The group's name, used by attach, stop and ls [default: the cluster's name, or the
    /// package's file name].
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// Save this named group's launch configuration for reuse after stopping or rebooting.
    #[arg(long, requires = "name")]
    pub persistent: bool,

    /// Candidate nodes in rank order, instead of a cluster or the whole orbit.
    #[arg(long, value_name = "NODE[,NODE...]", value_delimiter = ',')]
    pub nodes: Vec<String>,

    /// Verify package metadata and print placement without starting ranks or saving a group.
    #[arg(long)]
    pub dry_run: bool,

    /// Return once rank processes start; the HTTP server may still be loading the model.
    #[arg(long)]
    pub detach: bool,

    /// Upload this Flare executable [default: saved override or each node's installed Flare].
    #[arg(long, value_name = "PATH")]
    pub executable: Option<PathBuf>,

    /// Experimental ring in rank order: only neighbor RoCE QPs connect;
    /// native CPU relays forward non-neighbor traffic. Also emulates a ring on a switch.
    #[arg(long, help_heading = "Multi-host options")]
    pub roce_ring: bool,

    /// Parallel TCP streams per destination file over the fabric [default: saved group or 8].
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(1..=64), help_heading = "Multi-host options")]
    pub transfer_streams: Option<u8>,

    /// Experimental chain in rank order: like --roce-ring, with no link between the first and
    /// last rank; hosts cabled directly as a chain.
    #[arg(long, conflicts_with = "roce_ring", help_heading = "Multi-host options")]
    pub roce_chain: bool,

    /// IP address for rank zero's HTTP server [default: saved group or 127.0.0.1].
    #[arg(long, value_name = "ADDR", help_heading = "Serving options")]
    pub host: Option<IpAddr>,

    /// Port for rank zero's HTTP server [default: saved group or 8000].
    #[arg(long, value_name = "N", help_heading = "Serving options")]
    pub port: Option<u16>,

    /// Replace the package's serving alias; the canonical model ID remains accepted.
    #[arg(long, value_name = "ALIAS", value_parser = served_model_name, help_heading = "Serving options")]
    pub served_model_name: Option<String>,

    /// Favor throughput or output cadence [default: saved group or balanced].
    #[arg(long, value_name = "MODE",
        value_parser = ["balanced", "throughput"], help_heading = "Serving options")]
    pub scheduler_policy: Option<String>,

    /// File on rank zero's node containing the serving API key; omit to disable authentication.
    #[arg(long, value_name = "PATH", help_heading = "Serving options")]
    pub api_key_file: Option<PathBuf>,

    #[command(flatten)]
    pub serving: ServingOptions,
}

impl OrbitRun {
    /// Serving options every rank loads with.
    pub fn rank_options(&self, matches: &clap::ArgMatches) -> Vec<String> {
        let mut options = self.serving.rank_options(Shape::from_flags(self.roce_ring, self.roce_chain));
        // Explicit defaults must override saved values, while omitted defaults must not.
        if self.serving.kv_cache_limit == KvCacheLimit::Memory
            && matches.value_source("kv_cache_limit")
                == Some(clap::parser::ValueSource::CommandLine)
        {
            options.extend(["--kv-cache-limit".into(), "memory".into()]);
        }
        options
    }

    /// Options only rank zero's HTTP server takes.
    pub fn rank_zero_options(&self) -> Vec<String> {
        let mut options = Vec::new();
        if let Some(host) = self.host {
            options.extend(["--host".into(), host.to_string()]);
        }
        if let Some(port) = self.port {
            options.extend(["--port".into(), port.to_string()]);
        }
        if let Some(name) = &self.served_model_name {
            options.extend(["--served-model-name".into(), name.clone()]);
        }
        if let Some(policy) = &self.scheduler_policy {
            options.extend(["--scheduler-policy".into(), policy.clone()]);
        }
        if let Some(path) = &self.api_key_file {
            options.extend(["--api-key-file".into(), path.display().to_string()]);
        }
        options
    }

    /// The equivalent single-host arguments, for publisher trust resolution of the package.
    pub fn run_args(&self) -> Run {
        Run {
            package: Some(PathBuf::from(self.package.as_deref().unwrap_or_default())),
            host: self.host.unwrap_or_else(|| "127.0.0.1".parse().unwrap()),
            port: self.port.unwrap_or(8000),
            served_model_name: self.served_model_name.clone(),
            roce_ring: self.roce_ring,
            roce_chain: self.roce_chain,
            rendezvous: None,
            rank: None,
            hosts: Vec::new(),
            agents_file: None,
            controller_key: None,
            transfer_streams: self.transfer_streams,
            package_cache: None,
            stage_only: false,
            api_key_file: self.api_key_file.clone(),
            scheduler_policy: self
                .scheduler_policy
                .clone()
                .unwrap_or_else(|| "balanced".into()),
            serving: self.serving.clone(),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum TrustCommand {
    /// Remove all remembered publisher identities for a signing key.
    Revoke {
        fingerprint: String,
        #[arg(long, value_name = "PATH")]
        trust_store: Option<PathBuf>,
    },
}

#[derive(Debug, Args)]
#[command(
    args_override_self = true,
    after_help = "Examples:\n  flare run model.safetensors --gpus 0 --host 127.0.0.1 --port 8000\n  flare run model.safetensors --embedding-table-storage ram --mtp-draft-tokens 3\n\nConcurrency and draft limits must fit the package's prepared capacities.\nOmitted concurrency and speculation options use package defaults."
)]
pub struct Run {
    /// Local prepared .safetensors package or Hugging Face ORG/NAME[@REVISION].
    #[arg(value_name = "PACKAGE", required_unless_present = "rank")]
    pub package: Option<PathBuf>,

    /// IP address for the HTTP server.
    #[arg(
        long,
        value_name = "ADDR",
        default_value = "127.0.0.1",
        help_heading = "Serving options"
    )]
    pub host: IpAddr,

    /// Port for the HTTP server.
    #[arg(
        long,
        value_name = "N",
        default_value = "8000",
        help_heading = "Serving options"
    )]
    pub port: u16,

    /// Replace the package's serving alias; the canonical model ID remains accepted.
    #[arg(long, value_name = "ALIAS", value_parser = served_model_name, help_heading = "Serving options")]
    pub served_model_name: Option<String>,

    /// Experimental ring in rank order: only neighbor RoCE QPs connect;
    /// native CPU relays forward non-neighbor traffic. Also emulates a ring on a switch.
    #[arg(long, help_heading = "Multi-host options")]
    pub roce_ring: bool,

    /// Experimental chain in rank order: like --roce-ring, with no link between the first and
    /// last rank; hosts cabled directly as a chain.
    #[arg(long, conflicts_with = "roce_ring", help_heading = "Multi-host options")]
    pub roce_chain: bool,

    /// Rank zero's rendezvous address: where rank zero listens, or on a peer, where it joins rank zero.
    #[arg(long, value_name = "HOST:PORT", help_heading = "Multi-host options")]
    pub rendezvous: Option<String>,

    /// This host's rank of a multi-host package.
    #[arg(long, value_name = "N", help_heading = "Multi-host options")]
    pub rank: Option<usize>,

    /// Start peer ranks through paired agents, in rank order; rank zero runs locally.
    #[arg(
        long,
        value_name = "HOST[,HOST...]",
        value_delimiter = ',',
        help_heading = "Multi-host options"
    )]
    pub hosts: Vec<String>,

    /// Paired agent registry [default: ~/.config/flare/orbit.toml].
    #[arg(
        long,
        value_name = "PATH",
        requires = "hosts",
        help_heading = "Multi-host options"
    )]
    pub agents_file: Option<PathBuf>,

    /// Paired controller key [default: ~/.config/flare/orbit/controller.key].
    #[arg(
        long,
        value_name = "PATH",
        requires = "hosts",
        help_heading = "Multi-host options"
    )]
    pub controller_key: Option<PathBuf>,

    /// Parallel CX7 TCP streams per peer for package distribution [default: 8].
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(1..=64), help_heading = "Multi-host options")]
    pub transfer_streams: Option<u8>,

    /// Cache root for received rank packages [default: ~/.cache/flare].
    #[arg(long, value_name = "PATH", help_heading = "Multi-host options")]
    pub package_cache: Option<PathBuf>,

    /// Distribute and verify rank packages, then exit before GPU initialization.
    #[arg(long, help_heading = "Multi-host options")]
    pub stage_only: bool,

    /// File containing the serving API key; omit to disable authentication.
    #[arg(long, value_name = "PATH", help_heading = "Serving options")]
    pub api_key_file: Option<PathBuf>,

    /// Favor token throughput or balance it with output cadence; both retain token-service fairness.
    #[arg(long, value_name = "MODE", default_value = "balanced",
        value_parser = ["balanced", "throughput"], help_heading = "Serving options")]
    pub scheduler_policy: String,

    #[command(flatten)]
    pub serving: ServingOptions,
}

/// Options that select a package's trust, loading, speculation and caching; every rank of a
/// multi-host package loads with the same values.
#[derive(Debug, Args, Clone)]
pub struct ServingOptions {
    /// Additional publisher verification profiles (JSON); does not grant execution trust.
    /// Development packages default to ~/.config/flare/development-publisher.json.
    #[arg(long, value_name = "PATH", action = clap::ArgAction::Append)]
    pub publisher_keys: Vec<PathBuf>,
    /// Verify stored allocation and table ranges before loading model GPU resources.
    #[arg(long)]
    pub verify_weights: bool,
    /// Permanently accept this exact signing key together with --dangerously-accept-publisher.
    #[arg(
        long,
        requires = "dangerously_accept_publisher",
        value_name = "FINGERPRINT"
    )]
    pub dangerously_approve_key: Option<String>,
    /// Exact Hugging Face publisher namespace acknowledged for unattended first use.
    #[arg(long, requires = "dangerously_approve_key", value_name = "NAMESPACE")]
    pub dangerously_accept_publisher: Option<String>,
    /// Persistent trust file [default: ~/.config/flare/trust.toml].
    #[arg(long, value_name = "PATH")]
    pub trust_store: Option<PathBuf>,

    #[command(flatten)]
    pub hub: HubOptions,

    /// GPUs in rank order, relative to CUDA_VISIBLE_DEVICES when set.
    /// Omitted: the first N visible GPUs, where N is the package's local rank count.
    #[arg(long, value_name = "N[,N...]", value_delimiter = ',', action = clap::ArgAction::Set,
        value_parser = clap::value_parser!(i32).range(0..),
        help_heading = "Serving options")]
    pub gpus: Vec<i32>,

    /// Limit KV capacity by available memory, one context, or context times concurrency.
    #[arg(
        long,
        default_value = "memory",
        value_name = "MODE",
        help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["concurrency", "context", "memory"])
            .map(|value| match value.as_str() {
                "concurrency" => KvCacheLimit::Concurrency,
                "context" => KvCacheLimit::Context,
                _ => KvCacheLimit::Memory,
            })
    )]
    pub kv_cache_limit: KvCacheLimit,

    /// Per-GPU serving memory: occupied size (30G), free reserve (-30G), or total fraction (0.8).
    /// Omitted: package serving reserve.
    #[arg(
        long,
        allow_hyphen_values = true,
        value_name = "LIMIT",
        help_heading = "Serving options"
    )]
    pub memory_limit: Option<MemoryLimit>,

    /// Maximum simultaneous requests [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Serving options")]
    pub max_concurrency: Option<NonZeroUsize>,

    /// Storage for auxiliary embedding tables (PLE/Engram) [default: disk].
    #[arg(long, value_name = "MODE", help_heading = "Serving options",
        value_parser = PossibleValuesParser::new(["disk", "ram"])
            .map(|value| if value == "disk" { Storage::Disk } else { Storage::Ram }))]
    pub embedding_table_storage: Option<Storage>,

    /// Per-host RAM budget for disk-backed embedding rows, including index, counters and miss
    /// workspace; 0 disables [default: 4GiB on discrete GPUs, 1GiB on unified-memory GPUs].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Serving options")]
    pub embedding_table_cache_size: Option<u64>,

    /// Retain embedding scales in host RAM while weights remain on disk; adds to the row-cache budget.
    #[arg(long, help_heading = "Serving options")]
    pub embedding_table_resident_scales: bool,

    /// Disk workers per embedding table; shares a fixed I/O budget [default: 4 on Windows, 1 elsewhere].
    #[arg(long, value_name = "N", help_heading = "Advanced serving options",
        value_parser = PossibleValuesParser::new(["1", "2", "4", "8"]).map(|value| value.parse::<usize>().unwrap()))]
    pub embedding_table_workers: Option<usize>,

    /// Maximum drafts per round; 0 disables speculation [default: package setting].
    #[arg(long, value_name = "N", help_heading = "Speculation options")]
    pub mtp_draft_tokens: Option<usize>,

    /// Advanced: fix the total verification-row budget, including pending inputs, using
    /// one exact captured packed graph per batch [default: adaptive within captured capacities].
    /// Requires adaptive prefix selection at the captured draft depth.
    #[arg(long, value_name = "N", help_heading = "Advanced speculation options")]
    pub mtp_verification_token_budget: Option<NonZeroUsize>,

    /// Advanced: retain all captured packed verification capacities up to N total rows,
    /// including pending inputs [default: all captured capacities]. Requires adaptive
    /// prefix selection at the captured draft depth.
    #[arg(
        long,
        value_name = "N",
        conflicts_with = "mtp_verification_token_budget",
        help_heading = "Advanced speculation options"
    )]
    pub mtp_max_verification_token_budget: Option<NonZeroUsize>,

    /// Adapt verification width to acceptance estimates and execution cost [default: package setting].
    #[arg(
        long,
        overrides_with = "no_mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub mtp_adaptive_verification: bool,

    /// Use the maximum width where captured, otherwise width one [default: package setting].
    #[arg(
        long,
        overrides_with = "mtp_adaptive_verification",
        help_heading = "Speculation options"
    )]
    pub no_mtp_adaptive_verification: bool,

    /// Exponent on split-proposal prefix-survival scores; below 1 favors longer verification [default: 1.0].
    #[arg(long, value_name = "ALPHA", value_parser = verification_alpha,
        help_heading = "Advanced speculation options")]
    pub mtp_adaptive_verification_alpha: Option<f64>,

    /// Host memory for prefix checkpoints and prepared inputs, e.g. 16G; 0 disables [default: disabled].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Prefix cache options")]
    pub prefix_cache_memory: Option<u64>,

    /// Directory for prefix checkpoints and prepared inputs on disk [default: disabled].
    #[arg(long, value_name = "DIR", help_heading = "Prefix cache options")]
    pub prefix_cache_disk: Option<PathBuf>,

    /// Disk retention limit, e.g. 100G; 0 disables [default: 10% of free space].
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Prefix cache options")]
    pub prefix_cache_disk_limit: Option<u64>,

    /// Transfer disk checkpoints through host staging or GPUDirect Storage [default: staged].
    #[arg(long, value_name = "MODE", help_heading = "Prefix cache options",
        value_parser = PossibleValuesParser::new(["staged", "direct"])
            .map(|value| if value == "staged" { Transport::Staged } else { Transport::Direct }))]
    pub prefix_cache_disk_transport: Option<Transport>,
}

fn served_model_name(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        Err("served model name must not be empty".into())
    } else {
        Ok(value.to_owned())
    }
}

fn verification_alpha(value: &str) -> Result<f64, String> {
    value.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| "adaptive verification alpha must be finite and greater than zero".into())
}

impl std::ops::Deref for Run {
    type Target = ServingOptions;

    fn deref(&self) -> &ServingOptions {
        &self.serving
    }
}

impl Run {
    pub fn bind_address(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.port)
    }

    /// Serving options every rank of a multi-host package must load with: they select the same
    /// graphs and capacities on every host.
    #[cfg(test)]
    pub fn rank_options(&self) -> Vec<String> {
        self.serving.rank_options(Shape::from_flags(self.roce_ring, self.roce_chain))
    }

    pub fn load_options(&self) -> LoadOptions {
        self.serving.load_options(self.rank)
    }
}

impl ServingOptions {
    pub fn select_gpus(&mut self, local_ranks: usize) {
        if self.gpus.is_empty() {
            self.gpus = (0..local_ranks as i32).collect();
        }
    }

    /// Serving options every rank of a multi-host package must load with.
    pub fn rank_options(&self, shape: Shape) -> Vec<String> {
        let mut options = Vec::new();
        if self.kv_cache_limit != KvCacheLimit::Memory {
            options.extend(["--kv-cache-limit".into(), self.kv_cache_limit.to_string()]);
        }
        if let Some(limit) = self.memory_limit {
            options.push(format!("--memory-limit={limit}"));
        }
        match shape {
            Shape::Mesh => {}
            Shape::Ring => options.push("--roce-ring".into()),
            Shape::Chain => options.push("--roce-chain".into()),
        }
        if self.verify_weights {
            options.push("--verify-weights".into());
        }
        if let Some(n) = self.max_concurrency {
            options.extend(["--max-concurrency".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_draft_tokens {
            options.extend(["--mtp-draft-tokens".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_verification_token_budget {
            options.extend(["--mtp-verification-token-budget".into(), n.to_string()]);
        }
        if let Some(n) = self.mtp_max_verification_token_budget {
            options.extend(["--mtp-max-verification-token-budget".into(), n.to_string()]);
        }
        if self.mtp_adaptive_verification {
            options.push("--mtp-adaptive-verification".into());
        }
        if self.no_mtp_adaptive_verification {
            options.push("--no-mtp-adaptive-verification".into());
        }
        if let Some(alpha) = self.mtp_adaptive_verification_alpha {
            options.extend(["--mtp-adaptive-verification-alpha".into(), alpha.to_string()]);
        }
        if let Some(storage) = self.embedding_table_storage {
            options.extend([
                "--embedding-table-storage".into(),
                if storage == Storage::Disk {
                    "disk"
                } else {
                    "ram"
                }
                .into(),
            ]);
        }
        if let Some(workers) = self.embedding_table_workers {
            options.extend(["--embedding-table-workers".into(), workers.to_string()]);
        }
        if let Some(bytes) = self.embedding_table_cache_size {
            options.extend([
                "--embedding-table-cache-size".into(),
                bytes.to_string(),
            ]);
        }
        if self.embedding_table_resident_scales {
            options.push("--embedding-table-resident-scales".into());
        }
        // Every rank keeps its own records under its host's directory; rank zero's store drives
        // all of them, so every rank needs the same settings.
        if let Some(bytes) = self.prefix_cache_memory {
            options.extend(["--prefix-cache-memory".into(), bytes.to_string()]);
        }
        if let Some(directory) = &self.prefix_cache_disk {
            options.extend(["--prefix-cache-disk".into(), directory.display().to_string()]);
        }
        if let Some(bytes) = self.prefix_cache_disk_limit {
            options.extend(["--prefix-cache-disk-limit".into(), bytes.to_string()]);
        }
        if let Some(transport) = self.prefix_cache_disk_transport {
            options.extend([
                "--prefix-cache-disk-transport".into(),
                match transport {
                    Transport::Staged => "staged",
                    Transport::Direct => "direct",
                }
                .into(),
            ]);
        }
        options
    }

    pub fn load_options(&self, cluster_rank: Option<usize>) -> LoadOptions {
        LoadOptions {
            kv_cache_limit: self.kv_cache_limit,
            memory_limit: self.memory_limit,
            embedding_table_storage: self.embedding_table_storage,
            embedding_table_cache_bytes: self.embedding_table_cache_size,
            embedding_table_resident_scales: self.embedding_table_resident_scales,
            embedding_table_workers: self.embedding_table_workers,
            max_concurrency: self.max_concurrency.map(NonZeroUsize::get),
            mtp_draft_tokens: self.mtp_draft_tokens,
            mtp_verification_token_budget: self
                .mtp_verification_token_budget
                .map(NonZeroUsize::get),
            mtp_max_verification_token_budget: self
                .mtp_max_verification_token_budget
                .map(NonZeroUsize::get),
            adaptive_verification: if self.mtp_adaptive_verification {
                Some(true)
            } else if self.no_mtp_adaptive_verification {
                Some(false)
            } else {
                None
            },
            device_labels: (!self.gpus.is_empty()).then(|| self.gpus.clone()),
            prefix_cache: PrefixCacheOffload {
                memory_bytes: self.prefix_cache_memory,
                disk: self.prefix_cache_disk.clone(),
                disk_limit: self.prefix_cache_disk_limit,
                transport: self.prefix_cache_disk_transport,
            },
            cluster_rank,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_orbit_options_preserve_saved_values_and_allow_explicit_defaults() {
        use clap::{CommandFactory, FromArgMatches};
        assert!(Cli::try_parse_from(["flare", "orbit", "run", "model", "--persistent"]).is_err());
        for explicit in [false, true] {
            let mut arguments = vec!["flare", "orbit", "run", "--name", "service", "--persistent"];
            if explicit {
                arguments.extend([
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--scheduler-policy",
                    "balanced",
                    "--mtp-adaptive-verification-alpha",
                    "1.0",
                    "--kv-cache-limit",
                    "memory",
                    "--embedding-table-cache-size",
                    "0",
                    "--gpus",
                    "0",
                ]);
            }
            let matches = Cli::command().try_get_matches_from(arguments).unwrap();
            let Command::Orbit(orbit) = Cli::from_arg_matches(&matches).unwrap().command else {
                panic!()
            };
            let OrbitCommand::Run(orbit) = orbit.command else {
                panic!()
            };
            assert!(orbit.persistent);
            let run_matches = matches
                .subcommand_matches("orbit")
                .unwrap()
                .subcommand_matches("run")
                .unwrap();
            let mut saved = vec![
                "flare".to_string(),
                "run".into(),
                "model".into(),
                "--host".into(),
                "0.0.0.0".into(),
                "--port".into(),
                "8001".into(),
                "--scheduler-policy".into(),
                "throughput".into(),
                "--mtp-adaptive-verification-alpha".into(),
                "0.5".into(),
                "--kv-cache-limit".into(),
                "context".into(),
                "--embedding-table-cache-size".into(),
                "1G".into(),
            ];
            saved.extend(orbit.rank_options(run_matches));
            saved.extend(orbit.rank_zero_options());
            let Command::Run(effective) = Cli::try_parse_from(saved).unwrap().command else {
                panic!()
            };
            assert_eq!(effective.port, if explicit { 8000 } else { 8001 });
            assert_eq!(effective.mtp_adaptive_verification_alpha, Some(if explicit { 1.0 } else { 0.5 }));
            assert_eq!(
                effective.host.to_string(),
                if explicit { "127.0.0.1" } else { "0.0.0.0" }
            );
            assert_eq!(
                effective.scheduler_policy,
                if explicit { "balanced" } else { "throughput" }
            );
            assert_eq!(
                effective.serving.kv_cache_limit,
                if explicit {
                    KvCacheLimit::Memory
                } else {
                    KvCacheLimit::Context
                }
            );
            assert_eq!(
                effective.serving.embedding_table_cache_size,
                Some(if explicit { 0 } else { 1 << 30 })
            );
            assert_eq!(
                run_matches.value_source("gpus") == Some(clap::parser::ValueSource::CommandLine),
                explicit
            );
        }
    }

    fn run(arguments: &[&str]) -> Run {
        let Command::Run(args) = Cli::try_parse_from(arguments).unwrap().command else {
            panic!("expected run")
        };
        *args
    }

    #[test]
    fn omitted_gpus_follow_package_ranks_and_explicit_selection_wins() {
        for ranks in [1, 2, 4, 8, 16] {
            let mut args = run(&["flare", "run", "model.safetensors"]);
            assert!(args.gpus.is_empty());
            args.serving.select_gpus(ranks);
            let expected: Vec<_> = (0..ranks as i32).collect();
            assert_eq!(args.gpus, expected);
            assert_eq!(args.load_options().device_labels, Some(expected));
        }
        for (flag, expected) in [("0", vec![0]), ("3,1", vec![3, 1])] {
            let mut args = run(&["flare", "run", "model.safetensors", "--gpus", flag]);
            args.serving.select_gpus(4);
            assert_eq!(args.gpus, expected);
        }
        let mut args = run(&["flare", "run", "model.safetensors"]);
        args.serving.select_gpus(2);
        assert_eq!(
            crate::environment::visibility(&args.gpus, Some("4,7,5")).unwrap(),
            "4,7"
        );
        assert!(crate::environment::visibility(&args.gpus, Some("4")).is_err());
    }

    #[test]
    fn embedding_workers_reach_loader_and_remote_arguments() {
        assert_eq!(
            run(&["flare", "run", "model.safetensors"])
                .load_options()
                .embedding_table_workers,
            None
        );
        for workers in ["1", "2", "4", "8"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--embedding-table-workers",
                workers,
            ]);
            assert_eq!(
                args.load_options().embedding_table_workers,
                Some(workers.parse().unwrap())
            );
            assert_eq!(
                args.load_options().embedding_workers(),
                workers.parse::<usize>().unwrap()
            );
            assert!(
                args.rank_options()
                    .windows(2)
                    .any(|pair| pair == ["--embedding-table-workers", workers])
            );
        }
        for workers in ["0", "3", "5"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--embedding-table-workers",
                    workers
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn omitted_limits_preserve_package_defaults() {
        let args = run(&["flare", "run", "model.safetensors"]);
        assert_eq!(args.bind_address().to_string(), "127.0.0.1:8000");
        assert_eq!(args.served_model_name, None);
        assert_eq!(args.scheduler_policy, "balanced");
        assert_eq!(args.mtp_adaptive_verification_alpha, None);
        let options = args.load_options();
        assert_eq!(options.device_labels, None);
        assert_eq!(options.embedding_table_storage, None);
        assert_eq!(options.embedding_table_cache_bytes, None);
        assert!(!args.rank_options().contains(&"--embedding-table-cache-size".to_owned()));
        assert_eq!(options.max_concurrency, None);
        assert_eq!(options.kv_cache_limit, KvCacheLimit::Memory);
        assert_eq!(options.memory_limit, None);
        assert_eq!(options.mtp_draft_tokens, None);
        assert_eq!(options.mtp_verification_token_budget, None);
        assert_eq!(options.mtp_max_verification_token_budget, None);
        assert_eq!(options.adaptive_verification, None);
        assert_eq!(options.prefix_cache, PrefixCacheOffload::default());
    }

    #[test]
    fn scheduling_preference_is_explicit_and_validated() {
        assert_eq!(
            run(&[
                "flare",
                "run",
                "model.safetensors",
                "--scheduler-policy",
                "throughput"
            ])
            .scheduler_policy,
            "throughput"
        );
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--scheduler-policy",
                "unknown"
            ])
            .is_err()
        );
    }

    #[test]
    fn manual_peers_and_agent_launches_parse_without_orbit_configuration() {
        let peer = run(&["flare", "run", "--rank", "2", "--rendezvous", "head:29800"]);
        assert!(peer.package.is_none());
        assert_eq!(peer.rank, Some(2));
        let head = run(&[
            "flare",
            "run",
            "model.safetensors",
            "--hosts",
            "peer1,peer2,peer3",
            "--transfer-streams",
            "16",
            "--stage-only",
        ]);
        assert_eq!(head.hosts, ["peer1", "peer2", "peer3"]);
        assert_eq!(head.transfer_streams, Some(16));
        assert!(head.stage_only);
        assert!(Cli::try_parse_from(["flare", "run"]).is_err());
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--transfer-streams",
                "0"
            ])
            .is_err()
        );
        assert!(matches!(
            Cli::parse_from(["flare", "agent", "pair", "peer1", "--code", "join-code"]).command,
            Command::Agent(AgentArgs {
                command: Some(AgentCommand::Pair { .. }),
                ..
            })
        ));
    }

    #[test]
    fn explicit_options_reach_the_loader() {
        let args = run(&[
            "flare",
            "run",
            "--gpus=1",
            "--gpus=2,0",
            "--host",
            "::1",
            "--port",
            "8001",
            "model.safetensors",
            "--api-key-file",
            "key.txt",
            "--max-concurrency",
            "3",
            "--mtp-draft-tokens=0",
            "--embedding-table-storage",
            "ram",
            "--prefix-cache-memory",
            "1.5GiB",
            "--prefix-cache-disk",
            "prefixes",
            "--prefix-cache-disk-limit",
            "0",
            "--prefix-cache-disk-transport",
            "direct",
        ]);
        assert_eq!(args.package, Some(PathBuf::from("model.safetensors")));
        assert_eq!(args.bind_address().to_string(), "[::1]:8001");
        assert_eq!(args.api_key_file, Some(PathBuf::from("key.txt")));
        let options = args.load_options();
        assert_eq!(options.device_labels, Some(vec![2, 0]));
        assert_eq!(options.embedding_table_storage, Some(Storage::Ram));
        assert_eq!(options.max_concurrency, Some(3));
        assert_eq!(options.mtp_draft_tokens, Some(0));
        assert_eq!(
            options.prefix_cache,
            PrefixCacheOffload {
                memory_bytes: Some(3 << 29),
                disk: Some(PathBuf::from("prefixes")),
                disk_limit: Some(0),
                transport: Some(Transport::Direct),
            }
        );
    }

    #[test]
    fn ring_prestaging_keeps_target_order_and_selects_fabric_distribution() {
        let parsed = Cli::try_parse_from([
            "flare",
            "orbit",
            "push",
            "model.safetensors",
            "--to",
            "zero,one,two,three",
            "--roce-ring",
            "--transfer-streams",
            "8",
        ])
        .unwrap();
        let Command::Orbit(OrbitArgs {
            command: OrbitCommand::Push(push),
            ..
        }) = parsed.command
        else {
            panic!("expected prestaging command");
        };
        assert_eq!(push.to, ["zero", "one", "two", "three"]);
        assert_eq!(push.transfer_streams, Some(8));
        assert!(push.run_args().roce_ring);
        let arguments = ["flare", "orbit", "push", "model.safetensors", "--to", "zero,one,two"];
        let parsed = Cli::try_parse_from([&arguments[..], &["--roce-chain"]].concat()).unwrap();
        let Command::Orbit(OrbitArgs {
            command: OrbitCommand::Push(push),
            ..
        }) = parsed.command
        else {
            panic!("expected prestaging command");
        };
        let run = push.run_args();
        assert!(run.roce_chain && !run.roce_ring);
        assert!(
            Cli::try_parse_from([&arguments[..], &["--roce-chain", "--roce-ring"]].concat()).is_err()
        );
    }

    #[test]
    fn transfer_stream_counts_are_bounded_and_stay_out_of_rank_arguments() {
        for count in ["1", "4", "8", "16", "32", "64"] {
            let parsed = Cli::try_parse_from([
                "flare",
                "orbit",
                "run",
                "model.safetensors",
                "--roce-ring",
                "--transfer-streams",
                count,
            ])
            .unwrap();
            let Command::Orbit(OrbitArgs {
                command: OrbitCommand::Run(run),
                ..
            }) = parsed.command
            else {
                panic!("expected orbit run");
            };
            assert_eq!(run.transfer_streams, Some(count.parse().unwrap()));
            assert!(
                !run.run_args()
                    .rank_options()
                    .iter()
                    .any(|s| s.contains("transfer-streams"))
            );
        }
        for command in ["run", "push"] {
            for count in ["0", "65", "256"] {
                let mut args = vec![
                    "flare",
                    "orbit",
                    command,
                    "m.safetensors",
                    "--roce-ring",
                    "--transfer-streams",
                    count,
                ];
                if command == "push" {
                    args.extend(["--to", "zero,one"]);
                }
                assert!(Cli::try_parse_from(args).is_err());
            }
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "orbit",
                "push",
                "m.safetensors",
                "--to",
                "zero,one",
                "--transfer-streams",
                "4"
            ])
            .is_ok()
        );
    }

    #[test]
    fn ring_flag_is_forwarded_to_every_agent_started_rank() {
        let args = run(&["flare", "run", "model.safetensors", "--roce-ring"]);
        assert!(args.roce_ring);
        assert_eq!(args.rank_options(), vec!["--roce-ring"]);
    }

    #[test]
    fn verification_token_budget_reaches_loader_and_peer_ranks() {
        for budget in ["24", "32"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--mtp-verification-token-budget",
                budget,
            ]);
            assert_eq!(
                args.load_options().mtp_verification_token_budget,
                Some(budget.parse().unwrap())
            );
            assert_eq!(
                args.rank_options(),
                vec!["--mtp-verification-token-budget", budget]
            );
        }
        for invalid in ["0", "-1", "unbounded"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--mtp-verification-token-budget",
                    invalid
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn maximum_verification_token_budget_reaches_loader_and_peer_ranks() {
        for budget in ["24", "32"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--mtp-max-verification-token-budget",
                budget,
            ]);
            assert_eq!(
                args.load_options().mtp_max_verification_token_budget,
                Some(budget.parse().unwrap())
            );
            assert_eq!(
                args.rank_options(),
                vec!["--mtp-max-verification-token-budget", budget]
            );
        }
        for invalid in ["0", "-1", "unbounded"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--mtp-max-verification-token-budget",
                    invalid
                ])
                .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--mtp-verification-token-budget",
                "24",
                "--mtp-max-verification-token-budget",
                "24"
            ])
            .is_err()
        );
    }

    #[test]
    fn adaptive_verification_alpha_is_validated_and_forwarded() {
        for value in ["0.5", "1.0", "2.0"] {
            let args = run(&["flare", "run", "model.safetensors", "--mtp-adaptive-verification-alpha", value]);
            assert_eq!(args.mtp_adaptive_verification_alpha, Some(value.parse().unwrap()));
            let mut forwarded = vec!["flare".to_owned(), "run".into(), "model.safetensors".into()];
            forwarded.extend(args.rank_options());
            let Command::Run(peer) = Cli::try_parse_from(forwarded).unwrap().command else { panic!() };
            assert_eq!(peer.mtp_adaptive_verification_alpha, args.mtp_adaptive_verification_alpha);
        }
        for value in ["0", "-0.5", "NaN", "inf", "-inf", "invalid"] {
            assert!(Cli::try_parse_from(["flare", "run", "model.safetensors",
                &format!("--mtp-adaptive-verification-alpha={value}")]).is_err());
        }
    }

    #[test]
    fn last_verification_flag_wins() {
        let enable = "--mtp-adaptive-verification";
        let disable = "--no-mtp-adaptive-verification";
        for (flags, expected) in [
            (vec![enable], true),
            (vec![disable], false),
            (vec![enable, disable], false),
            (vec![disable, enable], true),
        ] {
            let mut arguments = vec!["flare", "run", "model.safetensors"];
            arguments.extend(flags);
            assert_eq!(
                run(&arguments).load_options().adaptive_verification,
                Some(expected)
            );
        }
    }

    #[test]
    fn listener_flags_support_independent_overrides_and_ipv6() {
        for (flags, expected) in [
            (vec!["--host", "0.0.0.0"], "0.0.0.0:8000"),
            (vec!["--port", "8001"], "127.0.0.1:8001"),
            (vec!["--host", "::1", "--port", "8002"], "[::1]:8002"),
        ] {
            let mut arguments = vec!["flare", "run", "model.safetensors"];
            arguments.extend(flags);
            assert_eq!(run(&arguments).bind_address().to_string(), expected);
        }
    }

    #[test]
    fn served_model_name_accepts_aliases_and_rejects_blank_values() {
        for name in ["my-model", "organization/model", "local model"] {
            let args = run(&[
                "flare",
                "run",
                "model.safetensors",
                "--served-model-name",
                name,
            ]);
            assert_eq!(args.served_model_name.as_deref(), Some(name));
        }
        for name in ["", " ", "\t"] {
            assert!(
                Cli::try_parse_from([
                    "flare",
                    "run",
                    "model.safetensors",
                    "--served-model-name",
                    name
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn embedding_row_cache_accepts_a_binary_memory_budget() {
        for (size, bytes) in [("0", 0), ("1G", 1 << 30), ("4GiB", 4 << 30)] {
            let args = run(&["flare", "run", "model.safetensors", "--embedding-table-cache-size", size]);
            let options = args.load_options();
            assert_eq!(options.embedding_table_cache_bytes, Some(bytes));
            assert_eq!(options.embedding_table_storage, None);
            assert_eq!(args.rank_options(), ["--embedding-table-cache-size".to_owned(), bytes.to_string()]);
            let mut forwarded = vec!["flare".to_owned(), "run".into(), "model.safetensors".into()];
            forwarded.extend(args.rank_options());
            let Command::Run(peer) = Cli::try_parse_from(forwarded).unwrap().command else { panic!() };
            assert_eq!(peer.load_options().embedding_table_cache_bytes, Some(bytes));
        }
    }

    #[test]
    fn prefix_cache_settings_reach_every_rank() {
        let args = run(&[
            "flare",
            "run",
            "model.safetensors",
            "--prefix-cache-memory",
            "0",
            "--prefix-cache-disk",
            "/var/cache/flare",
            "--prefix-cache-disk-limit",
            "100G",
            "--prefix-cache-disk-transport",
            "staged",
        ]);
        let options = args.rank_options();
        assert_eq!(
            options,
            [
                "--prefix-cache-memory",
                "0",
                "--prefix-cache-disk",
                "/var/cache/flare",
                "--prefix-cache-disk-limit",
                "107374182400",
                "--prefix-cache-disk-transport",
                "staged",
            ]
        );
        let mut forwarded = vec!["flare", "run", "model.safetensors"];
        forwarded.extend(options.iter().map(String::as_str));
        assert_eq!(
            run(&forwarded).load_options().prefix_cache,
            args.load_options().prefix_cache
        );
        assert!(run(&["flare", "run", "model.safetensors"]).rank_options().is_empty());
    }

    #[test]
    fn resident_embedding_scales_are_forwarded_to_every_rank() {
        let args = run(&["flare", "run", "model.safetensors", "--embedding-table-resident-scales",
                         "--embedding-table-cache-size", "1G"]);
        assert!(args.load_options().embedding_table_resident_scales);
        assert_eq!(args.load_options().embedding_table_cache_bytes, Some(1 << 30));
        assert!(args.rank_options().contains(&"--embedding-table-resident-scales".to_owned()));
        assert!(!run(&["flare", "run", "model.safetensors"]).load_options().embedding_table_resident_scales);
    }

    #[test]
    fn memory_limits_accept_signed_values_and_reach_every_rank() {
        for value in ["30g", "-30g", "0.8"] {
            for flags in [
                vec!["--memory-limit".to_owned(), value.to_owned()],
                vec![format!("--memory-limit={value}")],
            ] {
                let mut arguments = vec![
                    "flare",
                    "run",
                    "model.safetensors",
                    "--kv-cache-limit",
                    "context",
                ];
                arguments.extend(flags.iter().map(String::as_str));
                let args = run(&arguments);
                assert_eq!(args.load_options().kv_cache_limit, KvCacheLimit::Context);
                assert_eq!(
                    args.load_options().memory_limit,
                    Some(value.parse().unwrap())
                );
                let forwarded = args.rank_options();
                let mut peer = vec!["flare", "run", "model.safetensors"];
                peer.extend(forwarded.iter().map(String::as_str));
                let peer = run(&peer).load_options();
                assert_eq!(peer.memory_limit, args.load_options().memory_limit);
                assert_eq!(peer.kv_cache_limit, KvCacheLimit::Context);
            }
        }
        for value in ["0", "-0.8", "2", "NaN", "1e-1", "-0G"] {
            assert!(
                Cli::try_parse_from(["flare", "run", "model.safetensors", "--memory-limit", value])
                    .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "flare",
                "run",
                "model.safetensors",
                "--kv-cache-limit",
                "unknown"
            ])
            .is_err()
        );
    }
}
