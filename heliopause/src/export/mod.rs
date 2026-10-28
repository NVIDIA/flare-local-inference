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
pub(crate) mod weight_payload;
pub(crate) mod weight_reuse;
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
    #[serde(default)]
    pub reuse_weights: Option<weight_reuse::Source>,
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
    /// Rank processes sharing this host's memory; omitted means the whole rank group.
    #[serde(default)]
    pub local_ranks: Option<usize>,
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
    /// Thin semantic weight declaration; resolved against retained allocations.
    #[serde(default)]
    pub weight_payload: Option<serde_json::Value>,
    pub package_data: Option<serde_json::Value>,
    #[serde(default)]
    pub options: ExportOptions,
    #[serde(default)]
    pub sources: bool,
}

impl CaptureRequest {
    fn local_rank_count(&self) -> ExportResult<usize> {
        let count = self.local_ranks.unwrap_or(self.world_size);
        if count == 0 || count > self.world_size {
            return Err(invalid("local preparation ranks must be positive and fit the rank group"));
        }
        Ok(count)
    }
}

/// One declared weight group with its bytes mapped into the rank's logical
/// payload concatenation. The publisher turns these into initializers.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct WeightPayloadGroup {
    pub name: String,
    pub storage: String,
    pub nbytes: u64,
    #[serde(default)]
    pub spans: Vec<WeightPayloadSourceSpan>,
    /// Shared cache object holding these bytes; empty when capture staged a file.
    #[serde(default)]
    pub payload_id: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WeightPayloadSourceSpan {
    pub allocation: String,
    pub destination_offset: u64,
    /// Offset of these bytes in the rank's logical payload concatenation.
    pub source_offset: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct WeightPayloadPiece {
    /// Raw checkpoint tensor byte offset.
    pub source_offset: u64,
    /// Retained owner byte offset.
    pub destination_offset: u64,
    /// Innermost contiguous byte run.
    pub contiguous_bytes: u64,
    pub extents: Vec<u64>,
    pub source_strides: Vec<u64>,
    pub destination_strides: Vec<u64>,
}

/// One logical checkpoint source of a prepared view.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WeightPayloadSource {
    pub tensor: String,
    /// One half-open `[begin, end)` range per source axis, in axis order.
    #[serde(default)]
    pub ranges: Vec<(u64, u64)>,
    #[serde(default)]
    pub concatenation_axis: Option<u32>,
    /// Affine raw-byte maps. Empty keeps the legacy single-source semantics.
    #[serde(default)]
    pub pieces: Vec<WeightPayloadPiece>,
    /// Empty names raw checkpoint bytes; a nonempty encoding names canonical
    /// encoded-tile coordinates without executable transform semantics.
    #[serde(default)]
    pub piece_encoding: String,
}

/// One declared view and the semantic group it belongs to.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WeightPayloadView {
    pub name: String,
    pub group: String,
    pub storage: String,
    #[serde(default)]
    pub sources: Vec<WeightPayloadSource>,
    #[serde(default)]
    pub packing: String,
    /// Structured packing geometry recorded by the producer, when declared.
    #[serde(default)]
    pub packing_parameters: Option<weight_payload::WeightPackingParameters>,
    #[serde(default)]
    pub statistics_domain: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct WeightPayloadSources {
    #[serde(default)]
    pub groups: Vec<WeightPayloadGroup>,
    #[serde(default)]
    pub initial_state: WeightPayloadGroup,
    /// Semantic group of every declared view, independent of retained coverage.
    #[serde(default)]
    pub views: Vec<WeightPayloadView>,
    /// Payloads that the receiving coordinator already holds; no sender file is required.
    #[serde(default)]
    pub cached: Vec<helios_core::package::schema::wire::PayloadDescriptor>,
    /// File-backed embedding table metadata rebound to this preparation's buffers.
    #[serde(default)]
    pub engram: Option<helios_core::package::schema::wire::Engram>,
}

/// Leases for captured cache objects, held from capture through publication and
/// rank transfer so collection cannot unlink a payload between phases.
#[derive(Clone, Default)]
pub(crate) struct PayloadLeases(pub Vec<crate::package::cache::PayloadObject>);

impl std::fmt::Debug for PayloadLeases {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "PayloadLeases({})", self.0.len())
    }
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
    /// Stable semantic weight groups resolved against retained byte sources.
    #[serde(default)]
    pub weight_payload: Option<WeightPayloadSources>,
    /// Platform identifier detected from the capture device.
    #[serde(default)]
    pub hardware: Option<String>,
    /// Live object leases; serialized rank sources carry none.
    #[serde(skip)]
    pub payload_leases: PayloadLeases,
}
