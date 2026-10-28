use std::collections::{HashMap, HashSet};

use crate::{BufferSpan, Error, ExecutionError, GraphBundle, Result};

pub const ABI_VERSION: u32 = 5;
pub const VECTOR_RECORD_FIELDS: usize = 24;
pub const MAX_OUTPUT_TOKENS: usize = 8;
pub const REQUEST_LANES: usize = 64;
pub const MAX_TOP_LOGPROBS: usize = 20;
pub const MAX_LOGPROB_TOKEN_IDS: usize = 128;
// Each token/logprob pair occupies two int32 words, starting with the chosen token.
pub const LOGPROB_PAIR_FIELDS: usize = 2;
pub const LOGPROB_FIELDS: usize = LOGPROB_PAIR_FIELDS * (1 + MAX_LOGPROB_TOKEN_IDS);
pub const PROMPT_FIELDS: usize = 3 + 2 * MAX_TOP_LOGPROBS;
pub const BATCH_CAPACITIES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    Bool,
    Uint8,
    Float8E4m3fn,
    Int32,
    Int64,
    Float32,
    Float16,
    Bfloat16,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::Uint8 | Self::Float8E4m3fn => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Int32 | Self::Float32 => 4,
            Self::Int64 => 8,
        }
    }
}

impl TryFrom<&str> for Dtype {
    type Error = crate::Error;
    fn try_from(value: &str) -> Result<Self> {
        match value {
            "bool" => Ok(Self::Bool),
            "uint8" => Ok(Self::Uint8),
            "float8_e4m3fn" => Ok(Self::Float8E4m3fn),
            "int32" => Ok(Self::Int32),
            "int64" => Ok(Self::Int64),
            "float32" => Ok(Self::Float32),
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::Bfloat16),
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!("unsupported buffer dtype {value:?}"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphKind {
    Init,
    Restore,
    Snapshot,
    Prefill,
    ScoredPrefill,
    Decode,
    Speculative,
    Mixed,
    Resume,
    Vision,
}

impl GraphKind {
    pub fn is_prefill(self) -> bool {
        matches!(self, Self::Prefill | Self::ScoredPrefill)
    }

    pub fn samples(self) -> bool {
        matches!(
            self,
            Self::Prefill
                | Self::ScoredPrefill
                | Self::Decode
                | Self::Speculative
                | Self::Mixed
                | Self::Resume
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingMode {
    General,
}

impl TryFrom<&str> for SamplingMode {
    type Error = crate::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "general" => Ok(Self::General),
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!("unsupported graph sampling mode {value:?}"),
            )),
        }
    }
}

impl TryFrom<&str> for GraphKind {
    type Error = crate::Error;
    fn try_from(value: &str) -> Result<Self> {
        match value {
            "init" => Ok(Self::Init),
            "restore" => Ok(Self::Restore),
            "snapshot" => Ok(Self::Snapshot),
            "prefill" => Ok(Self::Prefill),
            "scored_prefill" => Ok(Self::ScoredPrefill),
            "decode" => Ok(Self::Decode),
            "speculative" => Ok(Self::Speculative),
            "mixed" => Ok(Self::Mixed),
            "resume" => Ok(Self::Resume),
            "vision" => Ok(Self::Vision),
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!("unsupported graph kind {value:?}"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixAllocation {
    Host,
    Gpu,
}

#[derive(Clone, Debug)]
pub struct BufferDescriptor {
    pub name: String,
    pub address: usize,
    pub nbytes: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct InputBinding {
    pub buffer_name: String,
    pub host_offset: usize,
    pub nbytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(any(feature = "frontend", feature = "runtime-package"), derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(any(feature = "frontend", feature = "runtime-package"), serde(deny_unknown_fields))]
pub struct TunedBinding {
    pub buffer: String,
    pub tuning_graph: String,
}

#[derive(Clone, Debug)]
pub struct TuningGraphDescriptor {
    pub name: String,
    pub handle: usize,
    pub outputs: Vec<String>,
    pub status_buffer: String,
}

#[derive(Clone, Debug)]
pub struct GraphDescriptor {
    /// Confidence prefixes produced per request in the completion frame.
    pub confidence_width: usize,
    /// Proposal depth of a complete speculative round.
    pub draft_tokens: usize,
    pub name: String,
    pub kind: GraphKind,
    pub sampling_mode: SamplingMode,
    pub handle: usize,
    pub batch_capacity: usize,
    pub token_capacity: usize,
    /// Physical target-token rows; explicit for every speculative operation.
    pub packed_token_capacity: usize,
    /// Prompt capacity for a mixed graph; zero for every other operation.
    pub prefill_capacity: usize,
    /// Zero denotes an indeterminate duration.
    pub estimated_duration_ns: u64,
    pub input_bindings: Vec<InputBinding>,
    pub tuned_bindings: Vec<TunedBinding>,
}

impl GraphDescriptor {
    pub fn packed_tokens(&self) -> usize {
        if self.packed_token_capacity != 0 {
            return self.packed_token_capacity;
        }
        if self.kind == GraphKind::Mixed {
            self.prefill_capacity.saturating_add(
                self.batch_capacity
                    .saturating_sub(1)
                    .saturating_mul(self.token_capacity),
            )
        } else {
            self.batch_capacity.saturating_mul(self.token_capacity)
        }
    }
}

/// A check every completed frame passes before its results are published: a transport for
/// collectives between hosts that failed leaves the frame's results untrustworthy.
#[derive(Clone)]
pub struct CollectiveHealth(pub std::sync::Arc<dyn Fn() -> crate::Result<()> + Send + Sync>);

impl std::fmt::Debug for CollectiveHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CollectiveHealth")
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionManifest {
    pub engram: Option<crate::engram::Binding>,
    /// Present when this rank's graphs launch collectives over a transport between hosts.
    pub collectives: Option<CollectiveHealth>,
    /// This rank's prefix checkpoint storage, present after cache allocation.
    pub cache: Option<flare_selene::CacheGeometry>,
    /// Host prefix store resources shared by every rank, present when configured.
    pub offload: Option<crate::offload::Binding>,
    pub abi_version: u32,
    pub context: usize,
    pub device: i32,
    pub max_active: usize,
    pub input_frame_bytes: usize,
    pub buffers: Vec<BufferDescriptor>,
    pub graphs: Vec<GraphDescriptor>,
    pub tuning_graphs: Vec<TuningGraphDescriptor>,
}

impl ExecutionManifest {
    fn validate_tuning(&self) -> Result<()> {
        let host_inputs: HashSet<_> = self.graphs.iter().flat_map(|graph|
            graph.input_bindings.iter().map(|binding| binding.buffer_name.as_str())).collect();
        let mut names: HashSet<&str> = self.graphs.iter().map(|graph| graph.name.as_str()).collect();
        let mut producers = HashMap::new();
        let mut statuses = HashSet::new();
        for graph in &self.tuning_graphs {
            if graph.name.is_empty() || !names.insert(&graph.name) || graph.outputs.is_empty() {
                return Err("tuning graphs require unique names and persistent outputs".into());
            }
            let status = self.buffer(&graph.status_buffer)?;
            if status.dtype != Dtype::Int32 || status.shape != [1] || status.strides != [1]
                || !statuses.insert(graph.status_buffer.as_str()) {
                return Err("tuning status requires a unique int32[1] buffer".into());
            }
            for output in &graph.outputs {
                self.buffer(output)?;
                if producers.insert(output.as_str(), graph.name.as_str()).is_some() {
                    return Err("tuning outputs require exactly one producer".into());
                }
            }
        }
        for name in producers.keys().chain(statuses.iter()) {
            if host_inputs.contains(name) || matches!(*name, "completion_records" | "completion_logprobs" | "prompt_logprobs" | "proposal_confidence" | "completion_speculative_survival" | "readiness_timestamps") {
                return Err("tuning storage cannot overlap request transport storage".into());
            }
        }
        if statuses.iter().any(|name| producers.contains_key(name)) {
            return Err("tuning status cannot also be a tuning output".into());
        }
        for graph in &self.graphs {
            let mut bindings = HashSet::new();
            for binding in &graph.tuned_bindings {
                if !bindings.insert(&binding.buffer)
                    || producers.get(binding.buffer.as_str()) != Some(&binding.tuning_graph.as_str()) {
                    return Err("tuned bindings must name their exact persistent output producer".into());
                }
            }
        }
        Ok(())
    }

    pub(crate) fn referenced_tuning_graphs(&self) -> Vec<usize> {
        let referenced: HashSet<_> = self.graphs.iter().flat_map(|graph|
            graph.tuned_bindings.iter().map(|binding| binding.tuning_graph.as_str())).collect();
        self.tuning_graphs.iter().enumerate().filter_map(|(index, graph)|
            referenced.contains(graph.name.as_str()).then_some(index)).collect()
    }

    pub fn completion_fields(&self) -> usize {
        VECTOR_RECORD_FIELDS
    }

    pub fn request_capacity(&self) -> usize {
        self.max_active.next_power_of_two()
    }

    pub fn completion_bytes(&self) -> usize {
        self.request_capacity() * self.completion_fields() * 8
    }

    pub fn max_top_logprobs(&self) -> Result<usize> {
        let output = self
            .buffers
            .iter()
            .find(|b| b.name == "completion_logprobs");
        let flags = self.buffers.iter().find(|b| b.name == "logprob_requested");
        let ids = self.buffers.iter().find(|b| b.name == "logprob_token_ids");
        let counts = self
            .buffers
            .iter()
            .find(|b| b.name == "logprob_token_id_counts");
        if (ids.is_some() || counts.is_some()) && output.is_some() {
            if !ids.is_some_and(|b| {
                b.dtype == Dtype::Int32
                    && b.shape == [self.request_capacity(), MAX_LOGPROB_TOKEN_IDS]
            }) || !counts
                .is_some_and(|b| b.dtype == Dtype::Int32 && b.shape == [self.request_capacity()])
                || !output.is_some_and(|b| b.shape.last() == Some(&LOGPROB_FIELDS))
            {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    "generation logprobs require complete logprob_token_ids and logprob_token_id_counts inputs and widened output storage",
                ));
            }
        }
        match (output, flags) {
            (None, None) if ids.is_none() && counts.is_none() => Ok(0),
            (Some(output), Some(flags)) => {
                let fields = output.shape.last().copied().unwrap_or(0);
                let expected = [2, self.request_capacity(), MAX_OUTPUT_TOKENS, fields];
                if fields < LOGPROB_PAIR_FIELDS * 2
                    || fields
                        > if ids.is_some() {
                            LOGPROB_FIELDS
                        } else {
                            2 * (1 + MAX_TOP_LOGPROBS)
                        }
                    || fields % LOGPROB_PAIR_FIELDS != 0
                    || output.dtype != Dtype::Int32
                    || output.shape != expected
                {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!(
                            "completion_logprobs: expected Int32 shape {expected:?} with \
                             even last extent {}..={LOGPROB_FIELDS}; got {:?} shape {:?}",
                            LOGPROB_PAIR_FIELDS * 2,
                            output.dtype,
                            output.shape,
                        ),
                    ));
                }
                if flags.dtype != Dtype::Int32 || flags.shape != [self.request_capacity()] {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!(
                            "logprob_requested: expected Int32 shape {:?}; got {:?} shape {:?}",
                            [self.request_capacity()],
                            flags.dtype,
                            flags.shape,
                        ),
                    ));
                }
                Ok((fields / LOGPROB_PAIR_FIELDS - 1).min(MAX_TOP_LOGPROBS))
            }
            _ => Err(Error::execution(
                ExecutionError::Manifest,
                format!(
                    "generation logprobs missing buffer {}",
                    if output.is_none() {
                        "completion_logprobs"
                    } else {
                        "logprob_requested"
                    }
                ),
            )),
        }
    }

    pub fn logprob_token_id_capacity(&self) -> usize {
        self.buffers
            .iter()
            .find(|b| b.name == "logprob_token_ids")
            .map_or(0, |b| b.shape[1])
    }

    pub fn prompt_logprob_capacity(&self) -> Result<usize> {
        let targets = self.buffers.iter().find(|b| b.name == "prompt_target_tokens");
        let output = self.buffers.iter().find(|b| b.name == "prompt_logprobs");
        let scored = || self.graphs.iter().filter(|g| g.kind == GraphKind::ScoredPrefill);
        match (targets, output, scored().next().is_none()) {
            (None, None, true) => Ok(0),
            (Some(targets), Some(output), false) => {
                let capacity = targets.shape.first().copied().unwrap_or(0);
                if capacity == 0 || targets.shape != [capacity] || targets.dtype != Dtype::Int32
                    || output.shape != [2, capacity, PROMPT_FIELDS] || output.dtype != Dtype::Int32
                    || scored().any(|g| g.token_capacity > capacity)
                    || self.max_top_logprobs()? != MAX_TOP_LOGPROBS {
                    return Err("invalid prompt logprob buffer layout".into());
                }
                Ok(capacity)
            }
            _ => Err("prompt logprobs require scored graphs, targets, and output storage together".into()),
        }
    }

    pub fn buffer(&self, name: &str) -> Result<&BufferDescriptor> {
        self.buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| {
                Error::execution(ExecutionError::Manifest, format!("unknown buffer {name:?}"))
            })
    }

    pub fn validate(&self) -> Result<()> {
        if self.abi_version != ABI_VERSION {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "unsupported execution ABI version; re-export the model package",
            ));
        }
        if !(1..=REQUEST_LANES).contains(&self.max_active) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "max_active must be 1 through 64",
            ));
        }
        if self.input_frame_bytes == 0
            || !self.input_frame_bytes.is_multiple_of(8)
            || self.input_frame_bytes > isize::MAX as usize
        {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "input_frame_bytes must be a positive, addressable multiple of eight",
            ));
        }
        GraphBundle {
            context: self.context,
            device: self.device,
            graph_execs: self.graphs.iter().map(|graph| graph.handle)
                .chain(self.tuning_graphs.iter().map(|graph| graph.handle)).collect(),
            buffers: self
                .buffers
                .iter()
                .map(|buffer| BufferSpan {
                    address: buffer.address,
                    nbytes: buffer.nbytes,
                })
                .collect(),
        }
        .validate()?;
        let mut names = HashSet::new();
        for buffer in &self.buffers {
            if buffer.name.is_empty() || !names.insert(&buffer.name) {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    "buffer names must be nonempty and unique",
                ));
            }
            if buffer.shape.is_empty()
                || buffer.shape.contains(&0)
                || buffer.strides.len() != buffer.shape.len()
            {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("invalid shape/strides for {}", buffer.name),
                ));
            }
            let mut stride = 1usize;
            for (extent, actual) in buffer.shape.iter().zip(&buffer.strides).rev() {
                if *actual != stride {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!("buffer {} must be contiguous row-major", buffer.name),
                    ));
                }
                stride = stride.checked_mul(*extent).ok_or_else(|| {
                    Error::execution(ExecutionError::Manifest, "buffer shape overflow")
                })?;
            }
            if stride.checked_mul(buffer.dtype.size()) != Some(buffer.nbytes)
                || buffer.address % buffer.dtype.size() != 0
            {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!(
                        "buffer {} byte extent or alignment does not match dtype/shape",
                        buffer.name
                    ),
                ));
            }
        }
        let mut spans: Vec<_> = self
            .buffers
            .iter()
            .map(|buffer| (buffer.address, buffer.address + buffer.nbytes))
            .collect();
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "registered device buffer spans overlap",
            ));
        }
        for (name, dtype, shape) in [
            ("frame_index", Dtype::Int32, vec![1]),
            ("submission_sequence", Dtype::Int64, vec![1]),
            (
                "completion_records",
                Dtype::Int64,
                vec![2, self.request_capacity(), self.completion_fields()],
            ),
        ] {
            let buffer = self.buffer(name)?;
            if buffer.dtype != dtype || buffer.shape != shape {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("invalid required buffer layout: {name}"),
                ));
            }
        }
        let logprobs = self.max_top_logprobs()? != 0;
        self.prompt_logprob_capacity()?;
        if let Some(buffer) = self.buffers.iter().find(|buffer| buffer.name == "completion_speculative_survival") {
            if self.abi_version != ABI_VERSION || buffer.dtype != Dtype::Float32
                || buffer.shape != [2, self.request_capacity(), MAX_OUTPUT_TOKENS - 1] {
                return Err("completion_speculative_survival requires ABI5 and two float32 frames of seven prefixes per request slot".into());
            }
        }
        if let Some(buffer) = self.buffers.iter().find(|buffer| buffer.name == "proposal_confidence") {
            if buffer.dtype != Dtype::Float32 || buffer.shape.len() != 3
                || buffer.shape[0] != 2 || buffer.shape[1] != self.request_capacity()
                || !(1..MAX_OUTPUT_TOKENS).contains(&buffer.shape[2]) {
                return Err("proposal confidence requires two float32 frames indexed by request slot and draft prefix".into());
            }
        }
        if let Some(buffer) = self.buffers.iter().find(|buffer| buffer.name == "readiness_timestamps") {
            if buffer.dtype != Dtype::Int64 || buffer.shape.len() != 2 || buffer.shape[0] != 2
                || buffer.shape[1] == 0 || buffer.shape[1] % 3 != 0
                || buffer.shape[1] / 3 > super::MAX_READINESS_INTERVALS
                || buffer.strides != [buffer.shape[1], 1] {
                return Err("readiness timestamps require two contiguous int64 frames of three timestamps per interval".into());
            }
        }
        self.validate_tuning()?;
        let mut names = HashSet::new();
        let mut host_spans = HashMap::<&str, (usize, usize)>::new();
        for graph in &self.graphs {
            if graph.kind == GraphKind::Speculative {
                if graph.packed_token_capacity < graph.batch_capacity
                    || graph.packed_token_capacity > graph.batch_capacity.saturating_mul(graph.token_capacity)
                    || graph.draft_tokens > 7
                    || (graph.token_capacity == 1 && graph.draft_tokens != 0) {
                    return Err("speculative B/D/W/P geometry is invalid".into());
                }
            } else if graph.packed_token_capacity != 0 || graph.draft_tokens != 0 {
                return Err("draft depth and target rows require a speculative operation".into());
            }
            if graph.confidence_width > 0 {
                let confidence = self.buffer("proposal_confidence")?;
                if graph.kind != GraphKind::Speculative
                    || graph.confidence_width != graph.draft_tokens
                    || confidence.dtype != Dtype::Float32
                    || confidence.shape.len() != 3
                    || confidence.shape[0] != 2
                    || confidence.shape[1] != self.request_capacity()
                    || confidence.shape[2] < graph.confidence_width
                    || confidence.shape[2] > MAX_OUTPUT_TOKENS - 1 {
                    return Err("confidence output requires [2, request_slots, draft_capacity] float32 frames".into());
                }
            }
            if graph.name.is_empty() || !names.insert(&graph.name) {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    "graph names must be nonempty and unique",
                ));
            }
            let valid = match graph.kind {
                GraphKind::Mixed => {
                    self.abi_version == ABI_VERSION
                        && BATCH_CAPACITIES.contains(&graph.batch_capacity)
                        && (2..=self.request_capacity()).contains(&graph.batch_capacity)
                        && (1..=8).contains(&graph.token_capacity)
                        && graph.prefill_capacity >= 64
                        && graph.prefill_capacity.is_multiple_of(64)
                        && graph.packed_tokens() <= i32::MAX as usize
                        && graph
                            .prefill_capacity
                            .checked_add((graph.batch_capacity - 1) * graph.token_capacity)
                            .is_some()
                }
                GraphKind::Speculative => {
                    self.abi_version == ABI_VERSION
                        && BATCH_CAPACITIES.contains(&graph.batch_capacity)
                        && graph.batch_capacity <= self.request_capacity()
                        && (1..=8).contains(&graph.token_capacity)
                }
                GraphKind::Prefill | GraphKind::ScoredPrefill => {
                    graph.batch_capacity == 1
                        && ([1, 16].contains(&graph.token_capacity)
                            || (graph.token_capacity >= 64
                                && graph.token_capacity.is_multiple_of(64)))
                }
                GraphKind::Vision => graph.batch_capacity == 1 && graph.token_capacity > 0,
                _ => {
                    BATCH_CAPACITIES.contains(&graph.batch_capacity)
                        && graph.batch_capacity <= self.request_capacity()
                        && graph.token_capacity == 1
                }
            };
            if !valid || (graph.kind != GraphKind::Mixed && graph.prefill_capacity != 0) {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("unsupported graph capacities: {}", graph.name),
                ));
            }
            let mut bound = HashSet::new();
            for binding in &graph.input_bindings {
                if !bound.insert(binding.buffer_name.as_str())
                    || ["completion_records", "completion_logprobs", "prompt_logprobs", "readiness_timestamps",
                        "completion_speculative_survival"].contains(&binding.buffer_name.as_str())
                {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!("duplicate or output-only input binding in {}", graph.name),
                    ));
                }
                let buffer = self.buffer(&binding.buffer_name)?;
                let end = binding
                    .host_offset
                    .checked_add(binding.nbytes)
                    .ok_or_else(|| {
                        Error::execution(ExecutionError::Manifest, "host input span overflow")
                    })?;
                if binding.nbytes != buffer.nbytes
                    || binding.host_offset % buffer.dtype.size() != 0
                    || end > self.input_frame_bytes
                {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        format!("invalid host input binding: {}", binding.buffer_name),
                    ));
                }
                let span = (binding.host_offset, end);
                if let Some(previous) = host_spans.insert(&binding.buffer_name, span) {
                    if previous != span {
                        return Err(Error::execution(
                            ExecutionError::Manifest,
                            "one buffer has different host offsets across graphs",
                        ));
                    }
                }
            }
            if bound.contains("prompt_target_tokens") != (graph.kind == GraphKind::ScoredPrefill) {
                return Err("only scored prefill graphs bind prompt targets".into());
            }
            if logprobs && graph.kind.samples() {
                if !bound.contains("logprob_requested") {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        "sampling graphs require the generation logprob reporting flag",
                    ));
                }
                if self.logprob_token_id_capacity() != 0
                    && (!bound.contains("logprob_token_ids")
                        || !bound.contains("logprob_token_id_counts"))
                {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        "sampling graphs require explicit logprob token ID inputs",
                    ));
                }
            }
            if !bound.contains("frame_index") || !bound.contains("submission_sequence") {
                return Err(Error::execution(
                    ExecutionError::Manifest,
                    format!("graph {} lacks reserved input bindings", graph.name),
                ));
            }
            if graph.kind == GraphKind::Mixed {
                let row = self.buffer("prefill_row")?;
                if row.dtype != Dtype::Int32 || row.shape != [1] || !bound.contains("prefill_row") {
                    return Err(Error::execution(
                        ExecutionError::Manifest,
                        "mixed graphs require an int32[1] prefill_row input",
                    ));
                }
            }
        }
        let mut spans: Vec<_> = host_spans.values().copied().collect();
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::execution(
                ExecutionError::Manifest,
                "different input buffers overlap in host frame",
            ));
        }
        Ok(())
    }
}
