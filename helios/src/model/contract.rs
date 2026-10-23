//! Prepared storage, value sources, and frontend syntax independent of model identity.

use super::VisionCapabilities;
use crate::{
    Result,
    execution::{ExecutionManifest, GraphKind},
};

macro_rules! contract_enum {
    ($name:ident { $($variant:ident = $value:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(i32)]
        pub enum $name { $($variant = $value),+ }
        impl TryFrom<i32> for $name {
            type Error = crate::Error;
            fn try_from(value: i32) -> Result<Self> {
                match value {
                    $($value => Ok(Self::$variant),)+
                    _ => Err(concat!("unsupported ", stringify!($name)).into()),
                }
            }
        }
        #[cfg(any(feature = "frontend", feature = "runtime-package"))]
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
                serializer.serialize_i32(*self as i32)
            }
        }
        #[cfg(any(feature = "frontend", feature = "runtime-package"))]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
                let value = <i32 as serde::Deserialize>::deserialize(deserializer)?;
                Self::try_from(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

contract_enum!(HistoryKind {
    PagedKeyValue = 1, PagedLatent = 2, SlidingWindow = 3, RecurrentState = 4, ContinuationState = 5,
});
contract_enum!(ValueSource {
    Constant = 1, RequestSlot = 2, RequestGeneration = 3, LiveRows = 4,
    QueryLength = 5, SampleEnabled = 6, PromptToken = 7, PendingToken = 8,
    PromptAndPendingToken = 9, PromptPosition = 10, ImageFeatureIndex = 11,
    ImageLeftContext = 12, ImageRightContext = 13, PageTable = 14,
    LiveStateRows = 15, LiveStateRow = 16, PrefixStateRow = 17, SnapshotStateRow = 18,
    ConsumedTokens = 19, SnapshotConsumedTokens = 20, PrefillRow = 21,
    SamplingSeed = 22, SamplingParameters = 23, SamplingTopK = 24, MaxNewTokens = 25,
    EosTokenIds = 26, EosCount = 27, PromptHistogram = 28, GrammarMask = 29,
    GrammarEnabled = 30, CopyTokenIds = 31, CopyTokenCount = 32,
    CopyConsumedTokens = 33, CopyGeneration = 34, CopyGrammarMask = 35,
    CopyGrammarCount = 36, ImageTensor = 37, ImageFeatureSlot = 38, PromptTargetToken = 39, SpeculativePrefixLimit = 40,
});
contract_enum!(InputIndexing { Scalar = 1, SubmissionRow = 2, RequestRow = 3, PackedToken = 4, ActiveRow = 5 });
contract_enum!(StateAccessKind { Read = 1, Write = 2, ReadWrite = 3 });
contract_enum!(ToolParserKind { QwenCoder = 1, GlmArgumentTags = 2, Dsml = 3, Xtml = 4, MiniMaxXml = 5 });
contract_enum!(ReasoningParserKind { ThinkTags = 1, XtmlThink = 2, MmThink = 3 });
contract_enum!(ImagePreparationKind { Qwen = 1, Mimo = 2, Deepseek = 3, Moonvit = 4, Step = 5, Radio = 6, DeepseekRowMajor = 7, Glm = 8, PatchMerge = 9 });
contract_enum!(ImageLayoutKind { Qwen = 1, Mimo = 2, Deepseek = 3, Moonvit = 4, Step = 5, Radio = 6, DeepseekRowMajor = 7, Glm = 8, Bracketed = 9 });
contract_enum!(GraphOperation { Init = 1, Restore = 2, Snapshot = 3, Prefill = 4, Decode = 5, Resume = 6, Vision = 7, Speculative = 8, Mixed = 9, Proposal = 10, Verification = 11, ScoredPrefill = 12, PackedSpeculative = 13 });

impl TryFrom<GraphKind> for GraphOperation {
    type Error = crate::Error;

    fn try_from(kind: GraphKind) -> Result<Self> {
        Ok(match kind {
            GraphKind::Init => Self::Init,
            GraphKind::Restore => Self::Restore,
            GraphKind::Snapshot => Self::Snapshot,
            GraphKind::Prefill => Self::Prefill,
            GraphKind::ScoredPrefill => Self::ScoredPrefill,
            GraphKind::Decode => Self::Decode,
            GraphKind::Resume => Self::Resume,
            GraphKind::Vision => Self::Vision,
            GraphKind::Speculative => Self::Speculative,
            GraphKind::Mixed => Self::Mixed,
            GraphKind::Proposal => Self::Proposal,
            GraphKind::Verification => Self::Verification,
            GraphKind::PackedSpeculative => Self::PackedSpeculative,
        })
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct PreparedContract {
    pub version: u32,
    pub vocabulary: usize,
    pub input_vocabulary: usize,
    pub vision: VisionCapabilities,
    pub cache: CacheContract,
    pub graphs: Vec<GraphContract>,
    pub frontend: FrontendContract,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct CacheContract {
    pub page_tokens: usize,
    pub regions: Vec<HistoryRegion>,
    pub page_tables: Vec<PageTableView>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct HistoryRegion {
    pub kind: HistoryKind,
    pub storage: StateView,
    pub tokens_per_record: usize,
    pub window_tokens: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct PageTableView {
    pub buffer: String,
    pub page_offset: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ValueBinding {
    pub buffer: String,
    pub source: ValueSource,
    pub indexing: InputIndexing,
    pub value_offset: i64,
    pub padding: i64,
    pub tensor: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct StateView {
    pub buffer: String,
    pub offset: usize,
    pub stride: usize,
    pub nbytes: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct StateAccess {
    pub view: StateView,
    pub access: StateAccessKind,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct CompletionCursor {
    pub state: StateView,
    pub lag_tokens: usize,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct GraphContract {
    pub graph: String,
    pub kind: GraphOperation,
    pub inputs: Vec<ValueBinding>,
    pub state: Vec<StateAccess>,
    pub state_rows_per_request: usize,
    pub draft_cursor: Option<CompletionCursor>,
    pub embedding_tables: Vec<u32>,
    pub pending_token: Option<StateView>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ToolParser {
    pub kind: ToolParserKind,
    pub wrapping_newlines: u32,
    pub tag_separator: String,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ReasoningParser {
    pub kind: ReasoningParserKind,
    pub open: String,
    pub close: String,
    pub wrapping_newlines: u32,
    pub terminal_markers: Vec<String>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct PrefixBoundary {
    pub roles: Vec<String>,
    pub append_role: Option<String>,
    pub continue_final_message: bool,
    pub minimum_messages: usize,
    pub required_start: String,
    pub required_end: String,
    pub required_marker: String,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct FrontendContract {
    pub tools: Option<ToolParser>,
    pub reasoning: Option<ReasoningParser>,
    pub image_preparation: Option<ImagePreparationKind>,
    pub image_layout: Option<ImageLayoutKind>,
    pub rejected_tokens: Vec<u32>,
    pub position_dimensions: usize,
    pub assistant_marker: String,
    pub message_end_markers: Vec<String>,
    pub prefix_boundary: Option<PrefixBoundary>,
    pub image_tokens: Option<ImageTokenLayout>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ImageTokenLayout {
    pub feature_token: u32,
    pub span: Option<ImageSpan>,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "frontend", feature = "runtime-package"),
    serde(deny_unknown_fields)
)]
pub struct ImageSpan {
    pub start_token: u32,
    pub end_token: u32,
    pub max_tokens: usize,
    pub interior_tokens: Vec<u32>,
    pub unscoped_tokens: Vec<u32>,
}

impl PreparedContract {
    pub fn graph(&self, name: &str) -> Result<&GraphContract> {
        self.graphs
            .iter()
            .find(|g| g.graph == name)
            .ok_or_else(|| format!("graph {name} has no semantic contract").into())
    }

    pub fn validate(&self, manifest: &ExecutionManifest) -> Result<()> {
        if self.version != 1
            || self.vocabulary == 0
            || self.input_vocabulary < self.vocabulary
            || self.input_vocabulary > u32::MAX as usize
            || self.cache.page_tokens == 0
            || !matches!(self.frontend.position_dimensions, 0 | 3)
        {
            return Err("invalid prepared model contract version, vocabulary, or geometry".into());
        }
        self.vision.validate()?;
        self.frontend.validate()?;
        if self.vision.enabled() != self.frontend.image_layout.is_some()
            || self.frontend.image_layout.is_some() != self.frontend.image_preparation.is_some()
        {
            return Err("image processing and prepared vision capacities must agree".into());
        }
        for region in &self.cache.regions {
            region.storage.validate(manifest)?;
            if region.tokens_per_record == 0
                || (matches!(
                    region.kind,
                    HistoryKind::PagedKeyValue | HistoryKind::PagedLatent
                ) && (!self
                    .cache
                    .page_tokens
                    .is_multiple_of(region.tokens_per_record)
                    || region.window_tokens != 0))
                || (region.kind == HistoryKind::SlidingWindow && region.window_tokens == 0)
            {
                return Err("invalid attention history storage geometry".into());
            }
        }
        if self.vision.enabled() != self.frontend.image_tokens.is_some()
            || self.frontend.image_tokens.as_ref().is_some_and(|layout| {
                layout.feature_token as usize >= self.input_vocabulary
                    || layout.span.as_ref().is_some_and(|span| {
                        span.max_tokens == 0
                            || span.start_token == span.end_token
                            || [span.start_token, span.end_token]
                                .iter()
                                .chain(&span.interior_tokens)
                                .chain(&span.unscoped_tokens)
                                .any(|id| *id as usize >= self.input_vocabulary)
                    })
            })
        {
            return Err("invalid prepared image-token layout".into());
        }
        let mut views = std::collections::HashSet::new();
        for view in &self.cache.page_tables {
            if !views.insert(&view.buffer) {
                return Err("duplicate page-table view".into());
            }
        }
        let mut names = std::collections::HashSet::new();
        for graph in &self.graphs {
            if !names.insert(&graph.graph) || !(1..=8).contains(&graph.state_rows_per_request) {
                return Err(
                    "duplicate graph contract or invalid continuation-row requirement".into(),
                );
            }
        }
        for descriptor in &manifest.graphs {
            let graph = self.graph(&descriptor.name)?;
            if graph.kind != descriptor.kind.try_into()? {
                return Err("semantic graph operation differs from its execution catalog".into());
            }
            let targets: Vec<_> = graph.inputs.iter().filter(|b| b.source == ValueSource::PromptTargetToken).collect();
            if (descriptor.kind == GraphKind::ScoredPrefill && (targets.len() != 1
                || targets[0].indexing != InputIndexing::PackedToken || targets[0].value_offset != 0
                || targets[0].padding != -1 || targets[0].buffer != "prompt_target_tokens"))
                || (descriptor.kind != GraphKind::ScoredPrefill && !targets.is_empty()) {
                return Err("prompt targets belong only to scored prefill graphs".into());
            }
            let mut destinations = std::collections::HashSet::new();
            for input in &graph.inputs {
                if !destinations.insert(&input.buffer)
                    || !descriptor
                        .input_bindings
                        .iter()
                        .any(|b| b.buffer_name == input.buffer)
                {
                    return Err("semantic input must have one graph transport binding".into());
                }
                let buffer = manifest.buffer(&input.buffer)?;
                let mut stride = 1;
                if buffer
                    .shape
                    .iter()
                    .zip(&buffer.strides)
                    .rev()
                    .any(|(size, actual)| {
                        let mismatch = *actual != stride;
                        stride = stride.saturating_mul(*size);
                        mismatch
                    })
                {
                    return Err("host input bindings require contiguous prepared storage".into());
                }
            }
            let prefix: Vec<_> = graph.inputs.iter().filter(|input| input.source == ValueSource::SpeculativePrefixLimit).collect();
            if descriptor.kind == GraphKind::PackedSpeculative {
                if prefix.len() != 1 || graph.inputs.iter().any(|b| b.source == ValueSource::QueryLength)
                    || prefix[0].indexing != InputIndexing::SubmissionRow
                    || prefix[0].value_offset != 0 || prefix[0].padding != 0
                    || manifest.buffer(&prefix[0].buffer)?.dtype != crate::execution::Dtype::Int32 {
                    return Err("packed combined graph requires one int32 submission prefix ceiling and device-owned query lengths".into());
                }
            } else if !prefix.is_empty() {
                return Err("speculative prefix ceilings require a packed combined operation".into());
            }
            for access in &graph.state {
                access.view.validate(manifest)?;
            }
            if graph.inputs.iter().any(|b| {
                matches!(
                    b.source,
                    ValueSource::CopyTokenIds
                        | ValueSource::CopyTokenCount
                        | ValueSource::CopyConsumedTokens
                        | ValueSource::CopyGeneration
                        | ValueSource::CopyGrammarMask
                        | ValueSource::CopyGrammarCount
                ) && !matches!(descriptor.kind, GraphKind::Speculative | GraphKind::Mixed)
            }) {
                return Err("copy proposals require a verification operation".into());
            }
            if let Some(cursor) = &graph.draft_cursor {
                cursor.state.validate(manifest)?;
                if !graph.state.iter().any(|access| {
                    access.access != StateAccessKind::Read && access.view.contains(&cursor.state)
                }) {
                    return Err("history cursor requires a declared graph state update".into());
                }
            }
            if let Some(pending) = &graph.pending_token {
                pending.validate(manifest)?;
                if !graph
                    .state
                    .iter()
                    .any(|s| s.view.contains(pending) && s.access != StateAccessKind::Write)
                {
                    return Err("pending token requires a declared device-state read".into());
                }
            }
            if let Some(pending) = &graph.pending_token {
                for operation in [
                    GraphOperation::Init,
                    GraphOperation::Restore,
                    GraphOperation::Prefill,
                ] {
                    let mut producers = self
                        .graphs
                        .iter()
                        .filter(|producer| producer.kind == operation)
                        .peekable();
                    if producers.peek().is_none()
                        || producers.any(|producer| {
                            !producer.state.iter().any(|s| {
                                s.access != StateAccessKind::Read && s.view.contains(pending)
                            })
                        })
                    {
                        return Err(
                            "device pending-token input lacks an admission or sampling producer"
                                .into(),
                        );
                    }
                }
            }
            if matches!(
                descriptor.kind,
                GraphKind::Decode
                    | GraphKind::Speculative
                    | GraphKind::Proposal
                    | GraphKind::Verification
                    | GraphKind::PackedSpeculative
                    | GraphKind::Mixed
            ) && !graph.inputs.iter().any(|b| {
                matches!(
                    b.source,
                    ValueSource::PendingToken | ValueSource::PromptAndPendingToken
                )
            }) && graph.pending_token.is_none()
            {
                return Err("decode graph must declare its pending token source".into());
            }
        }
        Ok(())
    }
}

impl StateView {
    fn contains(&self, other: &Self) -> bool {
        self.buffer == other.buffer
            && self.stride == other.stride
            && self.offset <= other.offset
            && self
                .offset
                .checked_add(self.nbytes)
                .zip(other.offset.checked_add(other.nbytes))
                .is_some_and(|(outer, inner)| inner <= outer)
    }

    fn validate(&self, manifest: &ExecutionManifest) -> Result<()> {
        let buffer = manifest.buffer(&self.buffer)?;
        if self.nbytes == 0
            || buffer.shape.is_empty()
            || buffer
                .strides
                .first()
                .and_then(|stride| stride.checked_mul(buffer.dtype.size()))
                != Some(self.stride)
            || self
                .offset
                .checked_add(self.nbytes)
                .is_none_or(|end| end > self.stride)
            || buffer.shape[0]
                .checked_mul(self.stride)
                .is_none_or(|end| end > buffer.nbytes)
        {
            return Err("persistent state view exceeds its prepared storage".into());
        }
        Ok(())
    }
}

impl FrontendContract {
    fn validate(&self) -> Result<()> {
        if self.assistant_marker.is_empty() || self.message_end_markers.iter().any(String::is_empty)
        {
            return Err("assistant envelope requires nonempty delimiters".into());
        }
        if let Some(tool) = &self.tools {
            if tool.wrapping_newlines > 1
                || (tool.kind != ToolParserKind::QwenCoder && tool.wrapping_newlines != 0)
                || (tool.kind != ToolParserKind::Dsml && !tool.tag_separator.is_empty())
                || !matches!(tool.tag_separator.as_str(), "" | " ")
            {
                return Err("unsupported tool parser syntax parameters".into());
            }
        }
        if let Some(reasoning) = &self.reasoning {
            let (open, close) = match reasoning.kind {
                ReasoningParserKind::ThinkTags => ("<think>", "</think>"),
                ReasoningParserKind::XtmlThink => ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
                ReasoningParserKind::MmThink => ("<mm:think>", "</mm:think>"),
            };
            if reasoning.open != open
                || reasoning.close != close
                || (reasoning.kind == ReasoningParserKind::XtmlThink
                    && (reasoning.wrapping_newlines != 0
                        || !reasoning.terminal_markers.is_empty()
                        || self
                            .tools
                            .as_ref()
                            .is_none_or(|tool| tool.kind != ToolParserKind::Xtml)))
                || reasoning.wrapping_newlines > 1
                || reasoning
                    .terminal_markers
                    .iter()
                    .any(|marker| marker != "<tool_call>")
                || (!reasoning.terminal_markers.is_empty()
                    && self.tools.as_ref().is_none_or(|tool| {
                        !matches!(
                            tool.kind,
                            ToolParserKind::QwenCoder | ToolParserKind::GlmArgumentTags
                        )
                    }))
            {
                return Err("unsupported think-tag reasoning parameters".into());
            }
        }
        if self.prefix_boundary.as_ref().is_some_and(|p| {
            p.roles.is_empty()
                || p.roles.iter().any(String::is_empty)
                || p.append_role.as_ref().is_some_and(String::is_empty)
        }) {
            return Err("prefix boundary requires nonempty message roles".into());
        }
        Ok(())
    }
}
