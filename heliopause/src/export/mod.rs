//! Preparation-time graph export and package publication.

pub(crate) mod archive;
pub(crate) mod binding;
pub(crate) mod blob;
pub(crate) mod capture;
pub(crate) mod conversion;
pub(crate) mod cubin;
pub(crate) mod io;
pub(crate) mod package;
pub(crate) mod protection;
pub(crate) mod reachability;
pub(crate) mod recorder;
pub(crate) mod source_identity;
#[cfg(unix)]
pub(crate) mod transfer;

use std::collections::BTreeMap;
use std::path::PathBuf;

pub(crate) use helios_core::runtime_package::wire;
use serde::{Deserialize, Serialize};

pub(crate) type ExportResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub(crate) type Result<T> = ExportResult<T>;

pub(crate) fn invalid(message: impl Into<String>) -> Box<dyn std::error::Error + Send + Sync> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Reference {
    pub allocation: String,
    #[serde(default)]
    pub offset: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Relocation {
    pub offset: u64,
    pub target: Reference,
    #[serde(default = "pointer")]
    pub kind: String,
}

fn pointer() -> String {
    "pointer".into()
}
fn device() -> String {
    "device".into()
}
fn snapshot() -> String {
    "snapshot".into()
}
fn alignment() -> u64 {
    256
}
fn one() -> usize {
    1
}
fn mtp() -> String {
    "mtp".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Allocation {
    pub name: String,
    pub address: u64,
    pub nbytes: u64,
    #[serde(default = "alignment")]
    pub alignment: u64,
    #[serde(default = "snapshot")]
    pub initialize: String,
    #[serde(default)]
    pub relocations: Vec<Relocation>,
    #[serde(default)]
    pub preserve_address: bool,
    #[serde(default = "device")]
    pub memory: String,
    #[serde(default)]
    pub snapshot_spans: Option<Vec<(u64, u64)>>,
    #[serde(default)]
    pub sizing: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Argument {
    pub size: usize,
    #[serde(default)]
    pub relocations: Vec<Relocation>,
    #[serde(default)]
    pub numeric: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Kernel {
    pub function: usize,
    pub module: String,
    pub arguments: Vec<Argument>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct GraphHandle {
    pub name: String,
    pub graph: usize,
    pub executable: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct FileSource {
    pub path: PathBuf,
    pub offset: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct FilePart {
    pub offset: u64,
    pub source: FileSource,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct ExportOptions {
    pub cache: Option<PathBuf>,
    #[serde(default)]
    pub no_weights: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CaptureRequest {
    pub path: PathBuf,
    #[serde(default)]
    pub cupti_path: Option<PathBuf>,
    pub context: usize,
    #[serde(default)]
    pub rank: usize,
    #[serde(default = "one")]
    pub world_size: usize,
    #[serde(default = "mtp")]
    pub proposal_source: String,
    #[serde(default)]
    pub publisher: Option<serde_json::Value>,
    #[serde(default = "alignment")]
    pub granularity: u64,
    #[serde(default)]
    pub peer_addresses: Vec<u64>,
    #[serde(default)]
    pub peer_mappings: Vec<(u64, u64)>,
    pub graphs: Vec<GraphHandle>,
    pub allocations: Vec<Allocation>,
    #[serde(default)]
    pub discover_allocations: bool,
    #[serde(default)]
    pub kernels: Vec<Kernel>,
    #[serde(default)]
    pub modules: BTreeMap<String, PathBuf>,
    #[serde(default)]
    pub reservations: Vec<(u64, u64)>,
    #[serde(default)]
    pub events: BTreeMap<String, usize>,
    #[serde(default)]
    pub numeric: Vec<(u64, u64)>,
    #[serde(default)]
    pub inactive: Vec<(u64, u64)>,
    #[serde(default)]
    pub roots: Vec<String>,
    #[serde(default)]
    pub parameter_layouts: BTreeMap<usize, Vec<(usize, bool)>>,
    #[serde(default)]
    pub zero_ranges: Vec<(u64, u64)>,
    #[serde(default)]
    pub device_address_floor: u64,
    pub execution: Option<serde_json::Value>,
    pub server: Option<serde_json::Value>,
    #[serde(default)]
    pub transports: Vec<serde_json::Value>,
    pub rebuild_weights: Option<serde_json::Value>,
    pub package_data: Option<serde_json::Value>,
    #[serde(default)]
    pub options: ExportOptions,
    #[serde(default)]
    pub sources: bool,
    pub shard_size: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RankOutput {
    pub index: wire::ModelIndex,
    pub payload: Vec<FilePart>,
    #[serde(default)]
    pub allocations: Vec<Allocation>,
    #[serde(default)]
    pub module_globals: Vec<(u64, u64, String)>,
    #[serde(default)]
    pub external: Vec<(u64, u64)>,
    #[serde(default)]
    pub staging: Option<PathBuf>,
}
