//! Bounded live-request resources and scheduling with CUDA-fenced retirement.

mod cost;
mod learning;
mod policy;
mod refresh;
mod service;
mod utility;
#[cfg(feature = "server")]
pub(crate) use cost::observed_duration_ns;
pub use policy::{PolicyConfig, RequestAccounting};
use refresh::PlanOrigin;
pub mod speculation;
pub(crate) mod decoded_history;
pub mod split;
#[cfg(test)]
mod valuation;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

use crate::cache::{
    CheckpointId, CompletionUpdate, ConditioningEvent, EvictionTarget, Fence, Lookup, PageId,
    PrefixCheckpointCache, PrefixHistory, PrefixKey, Publication, PublishResult, SnapshotId,
    StateSnapshot,
};
use crate::execution::{CompletionRecord, GraphKind, REQUEST_LANES};
use crate::offload::{InboundTicket, PublishedCheckpoint};
use crate::{Result, Shutdown};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RequestHandle {
    pub slot: u32,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateSlotId(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct SchedulerLimits {
    pub max_active: usize,
    /// Physical request slots and buffer strides fixed by graph capture.
    pub request_slots: usize,
    pub max_sequence_tokens: usize,
    pub prefill_chunk_size: usize,
    pub pages: u32,
    /// Prefix-cache rows, separate from the fixed execution partition.
    pub snapshots: u32,
}

impl SchedulerLimits {
    pub fn request_capacity(&self) -> usize {
        self.request_slots
    }

    pub(crate) fn validate_prefill(&self) -> Result<()> {
        if self.prefill_chunk_size < 64
            || !self.prefill_chunk_size.is_multiple_of(64)
            || self.prefill_chunk_size > self.max_sequence_tokens
        {
            return Err(crate::Error::execution(
                crate::ExecutionError::Manifest,
                "prefill_chunk_size must be divisible by 64 and within 64..=context",
            ));
        }
        Ok(())
    }
}

/// Required token capacities for a validated, 64-aligned prefill chunk limit.
/// Prepared prefill token capacities: doubling from 128 to a step of at least 512 tokens and about an
/// eighth of the chunk, then growing by that step, then the chunk size itself. A partial chunk pads
/// by less than one step. Matches `ignition.components.capacity.prefill_capacities`.
pub fn prefill_capacities(chunk_size: usize) -> impl Iterator<Item = usize> {
    let step = 512.max(chunk_size / 8 / 512 * 512);
    std::iter::successors(Some(128usize), |capacity| capacity.checked_mul(2))
        .take_while(move |capacity| *capacity < chunk_size.min(step))
        .chain(
            std::iter::successors(Some(step), move |capacity| capacity.checked_add(step))
                .take_while(move |capacity| *capacity < chunk_size),
        )
        .chain(std::iter::once(chunk_size))
}

#[derive(Clone, Debug)]
pub struct SchedulerCapabilities {
    pub vocabulary: usize,
    /// Logical history tokens per appendable page, including compressed or sharded storage.
    /// Recurrent state and sliding windows occupy separately reserved continuation rows.
    pub page_tokens: usize,
    pub vision: crate::model::VisionCapabilities,
    pub draft_pages_per_request: usize,
    /// Decode consumes a token uploaded from a host-visible completion rather than GPU-owned state.
    /// The request cannot queue another decode before its preceding decode completes.
    pub completion_inputs: Vec<String>,
    /// Execution state rows each active request holds; None, the widest speculative graph's width.
    pub state_rows_per_request: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub cache_lookups: u64,
    pub cache_hits: u64,
    pub reused_tokens: u64,
    pub snapshot_publications: u64,
    pub snapshot_evictions: u64,
    pub message_checkpoints: u64,
    pub prompt_checkpoints: u64,
    pub active_requests: usize,
    pub free_slots: usize,
    pub retiring_slots: usize,
    pub free_pages: usize,
    pub free_snapshots: usize,
    pub snapshot_capacity: usize,
    pub snapshot_backing_pages: usize,
    pub cached_checkpoints: usize,
    pub cached_pages: usize,
    pub host_lookups: u64,
    pub host_hits: u64,
    pub host_reused_tokens: u64,
    /// Requests waiting in `Inbound` for a host restore; they count as active.
    pub inbound_waiting: usize,
    /// Growable requests finished early to free KV pages and requeued by the runtime.
    pub preemptions: u64,
}

#[derive(Clone)]
pub struct RequestSpec {
    pub prompt: Vec<u32>,
    pub message_boundaries: Vec<usize>,
    pub conditioning: Vec<ConditioningEvent>,
    pub max_new_tokens: u32,
    pub grammar_constrained: bool,
    pub host_stops: bool,
    pub prompt_logprobs: bool,
    /// Reserve pages for the prompt and one page of output, then grow them while the
    /// request decodes. When pages run out the scheduler may preempt such a request:
    /// it finishes through its terminal checkpoint and the runtime readmits its history.
    pub growable_output: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestPhase {
    Initialize,
    /// Waiting for a host-restored checkpoint; never selected by `plan`.
    Inbound,
    Restore,
    Vision,
    Prefill,
    Resume,
    Decode,
    Snapshot,
    Finishing,
}

#[derive(Clone, Copy, Debug)]
enum BoundaryKind {
    Message,
    Prompt,
    ConditionedWindow,
}
#[derive(Clone, Copy, Debug)]
struct BoundaryCapture {
    kind: BoundaryKind,
    snapshot: SnapshotId,
    consumed_tokens: u64,
    resume: RequestPhase,
}

#[derive(Clone, Debug)]
pub(crate) struct ConditionedInput {
    pub token_span: std::ops::Range<usize>,
    pub feature_range: std::ops::Range<usize>,
    pub patches: usize,
}

#[derive(Clone, Debug)]
struct ConditionedWindow {
    inputs: std::ops::Range<usize>,
    token_start: usize,
    prefill_end: usize,
    features: Vec<std::ops::Range<usize>>,
    patches: VecDeque<usize>,
}

pub struct LiveRequest {
    pub image_slot: Option<usize>,
    conditioned_features: Option<Vec<std::ops::Range<usize>>>,
    conditioned_inputs_ready: bool,
    conditioned_windows: VecDeque<ConditionedWindow>,
    window_snapshot: Option<SnapshotId>,
    window_snapshot_ready: bool,
    window_restore: bool,
    epoch: u64,
    acceptance: speculation::AcceptanceProfile,
    accounting: RequestAccounting,
    pub handle: RequestHandle,
    pub state_slot: StateSlotId,
    pub terminal_snapshot: Option<SnapshotId>,
    pub phase: RequestPhase,
    pub consumed_tokens: u64,
    pub generated_tokens: u32,
    pub prompt_tokens: usize,
    pub grammar_constrained: bool,
    pub error_code: Option<i64>,
    host_stops: bool,
    prompt_logprobs: bool,
    execution_limit: u64,
    history: PrefixHistory,
    tokens: Vec<u32>,
    prefix: Option<CheckpointId>,
    prefix_tokens: u64,
    admission_reused_tokens: u64,
    message_boundaries: Vec<usize>,
    message_snapshot: Option<SnapshotId>,
    prompt_snapshot: Option<SnapshotId>,
    boundary_capture: Option<BoundaryCapture>,
    prefix_snapshot: Option<SnapshotId>,
    shared_pages: Vec<PageId>,
    private_pages: Vec<PageId>,
    reserved_pages: Vec<PageId>,
    max_new_tokens: u32,
    scheduled_consumed: u64,
    last_use: Fence,
    pending: usize,
    terminal_snapshot_ready: bool,
    vision_patches: VecDeque<usize>,
    prefill_spans: Vec<std::ops::Range<usize>>,
    copy_proposal: Option<speculation::CopyProposal>,
    pub(crate) decoded_sequence: decoded_history::DecodedSequence,
    growable_output: bool,
    /// Finishing to free its pages; the runtime readmits its history instead of completing it.
    preempted: bool,
    admission_order: u64,
}

impl LiveRequest {
    fn prefill_kind(&self) -> GraphKind {
        if self.prompt_logprobs { GraphKind::ScoredPrefill } else { GraphKind::Prefill }
    }

    pub(crate) fn feature_index(&self, canonical: i32) -> Result<i64> {
        if canonical < 0 {
            return Ok(i64::from(canonical));
        }
        let Some(ranges) = &self.conditioned_features else {
            return Ok(i64::from(canonical));
        };
        let mut destination = 0usize;
        for range in ranges {
            if range.contains(&(canonical as usize)) {
                return i64::try_from(destination + canonical as usize - range.start)
                    .map_err(|_| "image feature index exceeds integer range".into());
            }
            destination += range.len();
        }
        Err("prefill references an image feature omitted from execution".into())
    }

    pub fn accounting(&self) -> &RequestAccounting {
        &self.accounting
    }

    pub fn acceptance(&self) -> &speculation::AcceptanceProfile {
        &self.acceptance
    }

    pub fn prefix(&self) -> Option<CheckpointId> {
        self.prefix
    }
    pub fn prefix_snapshot(&self) -> Option<SnapshotId> {
        if self.window_restore {
            self.window_snapshot
        } else {
            self.prefix_snapshot
        }
    }
    pub(crate) fn conditioned_input_end(&self) -> usize {
        self.conditioned_windows
            .front()
            .map_or(usize::MAX, |window| window.inputs.end)
    }
    pub(crate) fn conditioned_window_ready(&self) -> bool {
        self.window_snapshot_ready
    }
    fn following_conditioned_window(&self) -> Option<&ConditionedWindow> {
        let window = self.conditioned_windows.front()?;
        (self.vision_patches.is_empty() && self.consumed_tokens as usize == window.prefill_end)
            .then(|| self.conditioned_windows.get(1))
            .flatten()
    }
    fn prefill_limit(&self) -> usize {
        if let Some(window) = self.following_conditioned_window() {
            return window.prefill_end;
        }
        self.conditioned_windows
            .front()
            .map_or(self.prompt_tokens, |window| window.prefill_end)
    }
    fn prompt_phase(&self) -> RequestPhase {
        if !self.vision_patches.is_empty() {
            RequestPhase::Vision
        } else if self.consumed_tokens == self.prompt_tokens as u64 {
            RequestPhase::Resume
        } else {
            RequestPhase::Prefill
        }
    }
    pub fn reused_prompt_tokens(&self) -> u64 {
        self.admission_reused_tokens
    }
    pub fn snapshot_destination(&self) -> SnapshotId {
        self.boundary_capture
            .map(|c| c.snapshot)
            .or(self.terminal_snapshot)
            .expect("scheduled snapshot owns a destination")
    }
    pub fn snapshot_consumed(&self) -> u64 {
        self.boundary_capture
            .map_or(self.consumed_tokens, |c| c.consumed_tokens)
    }
    pub fn last_use(&self) -> Fence {
        self.last_use
    }
    pub fn preempted(&self) -> bool {
        self.preempted
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn copy_allowed(&self) -> bool {
        !self.grammar_constrained && !self.host_stops
    }
    pub fn host_stops(&self) -> bool {
        self.host_stops
    }
    pub fn copy_proposal(&self) -> Option<&speculation::CopyProposal> {
        self.copy_proposal
            .as_ref()
            .filter(|p| p.consumed == self.consumed_tokens)
    }
    pub fn pages(&self) -> impl Iterator<Item = PageId> + '_ {
        self.shared_pages.iter().chain(&self.private_pages).copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledLane {
    pub request: RequestHandle,
    pub start_position: u64,
    pub tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedStep {
    pub prefill_capacity: usize,
    /// Index in the active lane list; absent when only decode rows are populated.
    pub prefill_row: Option<usize>,
    pub speculative: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledStep {
    pub draft_tokens: usize,
    pub kind: GraphKind,
    pub batch_capacity: usize,
    pub token_capacity: usize,
    pub packed_token_capacity: usize,
    pub mixed: Option<MixedStep>,
    pub lanes: Vec<ScheduledLane>,
    /// Completion feedback must be processed before another action is submitted.
    pub requires_feedback: bool,
}

impl ScheduledStep {
    pub fn validate_mixed(&self) -> Result<()> {
        if self.kind == GraphKind::PackedSpeculative {
            if !(1..=7).contains(&self.draft_tokens)
                || self.draft_tokens + 1 > self.token_capacity
                || self.packed_token_capacity == 0 || !self.requires_feedback
                || self.lanes.is_empty() || self.lanes.len() > self.batch_capacity
                || !crate::execution::BATCH_CAPACITIES.contains(&self.batch_capacity)
                || !(1..=8).contains(&self.token_capacity)
                || self.lanes.iter().any(|lane| !(1..=self.draft_tokens + 1).contains(&lane.tokens))
                || self.lanes.iter().enumerate().any(|(i, a)| self.lanes[..i].iter().any(|b| a.request.slot == b.request.slot)) {
                return Err("packed combined lanes exceed their frozen proposal ceilings".into());
            }
        } else if self.draft_tokens != 0 {
            return Err("proposal depth belongs only to packed combined steps".into());
        }
        if self.packed_token_capacity != 0 && (!matches!(self.kind, GraphKind::Verification | GraphKind::PackedSpeculative)
            || self.packed_token_capacity < self.batch_capacity
            || self.packed_token_capacity > self.batch_capacity.saturating_mul(self.token_capacity)
            || (self.kind == GraphKind::Verification
                && self.lanes.iter().map(|lane| lane.tokens).sum::<usize>() > self.packed_token_capacity)) {
            return Err("verification lanes exceed their packed token capacity".into());
        }
        match &self.mixed {
            None if self.kind != GraphKind::Mixed => Ok(()),
            Some(m)
                if self.kind == GraphKind::Mixed
                    && self.batch_capacity >= 2
                    && crate::execution::BATCH_CAPACITIES.contains(&self.batch_capacity)
                    && !self.lanes.is_empty()
                    && self.lanes.len() <= self.batch_capacity
                    && m.prefill_capacity >= 64
                    && m.prefill_capacity.is_multiple_of(64)
                    && m.prefill_row.is_none_or(|row| row < self.lanes.len())
                    && (1..=8).contains(&self.token_capacity)
                    && (m.speculative || self.token_capacity == 1)
                    && self.lanes.iter().enumerate().all(|(row, lane)| {
                        lane.tokens > 0
                            && lane.tokens
                                <= if m.prefill_row == Some(row) {
                                    m.prefill_capacity
                                } else {
                                    self.token_capacity
                                }
                    }) =>
            {
                Ok(())
            }
            _ => Err("mixed step roles do not match its graph or lane capacities".into()),
        }
    }

    pub fn prefill_capacity(&self) -> usize {
        self.mixed.as_ref().map_or(0, |m| m.prefill_capacity)
    }

    pub fn lane_kind(&self, row: usize) -> GraphKind {
        match &self.mixed {
            Some(m) if m.prefill_row == Some(row) => GraphKind::Prefill,
            Some(m) if m.speculative => GraphKind::Speculative,
            Some(_) => GraphKind::Decode,
            None => self.kind,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CostFreshness {
    #[default]
    Unknown,
    Fresh,
    Stale,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimingOrigin {
    #[default]
    Discovery,
    Ordinary,
    Refresh,
    Acceptance,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SubmittedTiming {
    pub retained_ns: u64,
    pub observed_at_ns: Option<u64>,
    pub freshness: CostFreshness,
    pub origin: TimingOrigin,
    pub credit_ns: f64,
    pub commit_ns: u64,
    pub selected_at_ns: u64,
    pub refresh_episode: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct SelectedCost {
    retained_ns: u64,
    observed_at_ns: Option<u64>,
    freshness: CostFreshness,
    selected_at_ns: u64,
}

/// An immutable selection with frozen estimates. Commit rechecks request state,
/// feedback barriers, and refresh eligibility without rerunning selection.
#[derive(Clone, Debug)]
pub struct ScheduledPlan {
    owner: Arc<()>,
    step: ScheduledStep,
    epochs: Vec<u64>,
    estimated_duration_ns: u64,
    selection_value_rate: f64,
    selection_log_scale: f64,
    /// Physical verification rows selected before the proposal is sampled.
    verification_budget: Option<usize>,
    origin: PlanOrigin,
    selected_cost: SelectedCost,
}

impl ScheduledPlan {
    pub fn estimated_duration_ns(&self) -> u64 {
        self.estimated_duration_ns
    }
    pub fn selection_value_rate(&self) -> f64 {
        self.selection_value_rate
    }
    pub fn selection_log_scale(&self) -> f64 {
        self.selection_log_scale
    }
}

struct Submission {
    step: ScheduledStep,
    expected_finish_ns: u64,
    estimated_duration_ns: u64,
    participants: [Option<service::ServiceParticipant>; REQUEST_LANES],
    origin: PlanOrigin,
    timing: SubmittedTiming,
}
impl std::ops::Deref for Submission {
    type Target = ScheduledStep;
    fn deref(&self) -> &Self::Target {
        &self.step
    }
}

impl std::ops::Deref for ScheduledPlan {
    type Target = ScheduledStep;

    fn deref(&self) -> &Self::Target {
        &self.step
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedToken {
    pub request: RequestHandle,
    pub token: u32,
    pub finished: bool,
}

struct RetiredRequest {
    image_slot: Option<usize>,
    pages: Vec<PageId>,
    snapshots: [Option<SnapshotId>; 4],
    slot: usize,
    through: Fence,
}
struct Slot {
    generation: u64,
    request: Option<LiveRequest>,
    retiring: bool,
}

/// A validated request with its prefix history and GPU cache lookup, before any
/// reservation. The worker consults the host store between `plan_admission` and
/// `admit_planned`.
pub struct AdmissionPlan {
    history: PrefixHistory,
    lookup: Lookup,
}

impl AdmissionPlan {
    pub fn history(&self) -> &PrefixHistory {
        &self.history
    }
    /// Consumed count of the GPU checkpoint the lookup matched; zero on a miss.
    pub fn resident_tokens(&self) -> u64 {
        self.lookup.consumed_tokens
    }
}

/// A host record chain the worker chose to restore above the plan's GPU hit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InboundRequest {
    pub key: PrefixKey,
    pub consumed_tokens: u64,
    /// Complete pages at `consumed_tokens` minus complete pages at the plan's
    /// resident count.
    pub pages: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admitted {
    pub handle: RequestHandle,
    /// Destination ids of the restore this admission started; `None` when the
    /// request joined a restore already in flight or needs none.
    pub ticket: Option<InboundTicket>,
}

/// Scheduler-owned destination of one host-to-GPU restore. Its pages, snapshot,
/// and pin on the GPU ancestor outlive the requests waiting for it.
struct Ticket {
    parent: Option<CheckpointId>,
    pages: Vec<PageId>,
    snapshot: SnapshotId,
    history: PrefixHistory,
    waiters: Vec<RequestHandle>,
}

pub struct RequestScheduler {
    plan_owner: Arc<()>,
    limits: SchedulerLimits,
    capabilities: SchedulerCapabilities,
    identity: [u8; 32],
    cache: PrefixCheckpointCache,
    slots: Vec<Slot>,
    free_pages: BTreeSet<PageId>,
    free_image_slots: BTreeSet<usize>,
    free_snapshots: BTreeSet<SnapshotId>,
    snapshot_pages: Vec<std::ops::Range<u32>>,
    page_snapshot_refs: Vec<u32>,
    retired: Vec<RetiredRequest>,
    submissions: BTreeMap<Fence, Submission>,
    completed: Fence,
    submitted: Fence,
    now_ns: u64,
    policy: PolicyConfig,
    candidates: std::cell::RefCell<Vec<policy::Candidate>>,
    stats: SchedulerStats,
    speculative_widths: Vec<usize>,
    speculative_fixed_width: Option<usize>,
    speculative_catalog: [Vec<usize>; 7],
    prefill_catalog: Vec<usize>,
    scored_prefill_catalog: Vec<usize>,
    mixed_catalog: Vec<(usize, usize, usize)>,
    proposal_source: speculation::ProposalSource,
    decoded_history: Option<decoded_history::DecodedHistory>,
    split: Option<split::SplitSpeculation>,
    costs: cost::ActionCosts,
    refresh: refresh::RefreshState,
    tickets: HashMap<PrefixKey, Ticket>,
    track_publications: bool,
    publications: Vec<CheckpointId>,
    admissions: u64,
}

impl RequestScheduler {
    pub fn new(
        identity: [u8; 32],
        limits: SchedulerLimits,
        capabilities: SchedulerCapabilities,
    ) -> Result<Self> {
        limits.validate_prefill()?;
        capabilities.vision.validate()?;
        if capabilities.draft_pages_per_request > limits.pages as usize
            || capabilities.vocabulary == 0
            || capabilities.vocabulary > i32::MAX as usize
        {
            return Err("invalid model scheduling capabilities".into());
        }

        if !crate::execution::BATCH_CAPACITIES.contains(&limits.request_slots)
            || limits.max_active > limits.request_slots
            || !(1..=REQUEST_LANES).contains(&limits.max_active)
            || limits.max_sequence_tokens == 0
            || limits.pages == 0
            || limits.max_sequence_tokens > i64::MAX as usize
        {
            return Err("invalid live request capacities".into());
        }
        let cache = PrefixCheckpointCache::new(
            identity,
            limits.pages,
            limits.snapshots,
            capabilities.page_tokens,
        )?;
        Ok(Self {
            plan_owner: Arc::new(()),
            limits,
            free_image_slots: (0..capabilities.vision.feature_slots).collect(),
            capabilities,
            identity,
            cache,
            slots: (0..limits.max_active)
                .map(|_| Slot {
                    generation: 0,
                    request: None,
                    retiring: false,
                })
                .collect(),
            free_pages: (0..limits.pages).map(PageId).collect(),
            free_snapshots: (0..limits.snapshots).map(SnapshotId).collect(),
            snapshot_pages: Vec::new(),
            page_snapshot_refs: vec![0; limits.pages as usize],
            retired: Vec::new(),
            submissions: BTreeMap::new(),
            completed: Fence(0),
            submitted: Fence(0),
            now_ns: 0,
            policy: PolicyConfig::default(),
            candidates: policy::scratch(),
            stats: SchedulerStats::default(),
            speculative_widths: Vec::new(),
            speculative_fixed_width: None,
            speculative_catalog: Default::default(),
            prefill_catalog: prefill_capacities(limits.prefill_chunk_size).collect(),
            scored_prefill_catalog: Vec::new(),
            mixed_catalog: Vec::new(),
            proposal_source: speculation::ProposalSource::Mtp,
            decoded_history: None,
            split: None,
            costs: cost::ActionCosts::default(),
            refresh: refresh::RefreshState::default(),
            tickets: HashMap::new(),
            track_publications: false,
            publications: Vec::new(),
            admissions: 0,
        })
    }

    pub fn execution_width(&self) -> usize {
        self.speculative_catalog
            .iter()
            .flatten()
            .copied()
            .chain(self.mixed_catalog.iter().map(|(_, width, _)| *width))
            .max()
            .unwrap_or(1)
    }

    /// Execution state rows each active request holds, from its first state slot on.
    pub fn execution_rows(&self) -> usize {
        self.capabilities
            .state_rows_per_request
            .unwrap_or_else(|| self.execution_width())
    }

    pub fn prefill_graphs(&mut self, capacities: &[usize]) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("prefill graphs must be selected before admission".into());
        }
        let mut catalog = capacities.to_vec();
        catalog.sort_unstable();
        if catalog.last() != Some(&self.limits.prefill_chunk_size)
            || catalog.iter().any(|n| *n == 0 || !n.is_multiple_of(64))
            || catalog.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(
                "prefill graphs must be distinct aligned capacities covering the chunk size".into(),
            );
        }
        self.prefill_catalog = catalog;
        Ok(())
    }

    pub fn scored_prefill_graphs(&mut self, capacities: &[usize]) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("scored prefill graphs must be selected before admission".into());
        }
        let mut catalog = capacities.to_vec();
        catalog.sort_unstable();
        if !catalog.is_empty() && (catalog.last() != Some(&self.limits.prefill_chunk_size)
            || catalog.iter().any(|n| *n == 0 || !n.is_multiple_of(64))
            || catalog.windows(2).any(|p| p[0] == p[1])) {
            return Err("scored prefill capacities must cover the chunk size".into());
        }
        self.scored_prefill_catalog = catalog;
        Ok(())
    }

    pub fn snapshot_slot(&self, snapshot: SnapshotId) -> StateSlotId {
        StateSlotId((1 + self.limits.max_active * self.execution_rows()) as u32 + snapshot.0)
    }

    /// Each shape supplies (physical rows, decode width, prompt capacity).
    pub fn mixed_graphs(&mut self, shapes: &[(usize, usize, usize)]) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("mixed graphs must be selected before admission".into());
        }
        let mut catalog = shapes.to_vec();
        catalog.sort_unstable();
        if catalog.windows(2).any(|s| s[0] == s[1])
            || catalog.iter().any(|&(batch, width, prefill)| {
                batch < 2
                    || !crate::execution::BATCH_CAPACITIES.contains(&batch)
                    || batch > self.limits.max_active.next_power_of_two()
                    || prefill == 0
                    || !prefill.is_multiple_of(64)
                    || prefill > self.limits.prefill_chunk_size
                    || prefill.saturating_add(batch.saturating_sub(1).saturating_mul(width))
                        > i32::MAX as usize
                    || if self.speculative_widths.is_empty() {
                        width != 1
                    } else {
                        !self.speculative_widths.contains(&width)
                            || self.speculative_fixed_width.is_some_and(|w| w != width && width != 1)
                    }
            })
        {
            return Err("mixed graph shape is duplicated or outside runtime limits".into());
        }
        if let Some(fixed) = self.speculative_fixed_width {
            catalog.retain(|&(batch, width, _)| width == fixed
                || !shapes.iter().any(|&(b, w, _)| b == batch && w == fixed));
        }
        self.mixed_catalog = catalog;
        Ok(())
    }

    pub fn enable_speculation(&mut self, widths: &[usize]) -> Result<()> {
        speculation::validate_widths(widths)?;
        if self.slots.iter().any(|s| s.request.is_some())
            || self.submitted != Fence(0)
            || !self.mixed_catalog.is_empty()
        {
            return Err("speculation must be configured before mixed graphs and admission".into());
        }
        self.speculative_widths = widths.to_vec();
        self.refresh_speculative_catalog();
        Ok(())
    }
    pub fn fixed_speculative_width(&mut self, width: Option<usize>) -> Result<()> {
        if self.submitted != Fence(0)
            || !self.mixed_catalog.is_empty()
            || self.slots.iter().any(|s| s.request.is_some())
            || width.is_some_and(|w| !self.speculative_widths.contains(&w))
        {
            return Err(
                "fixed speculative width must be prepared and configured before admission".into(),
            );
        }
        self.speculative_fixed_width = width;
        self.refresh_speculative_catalog();
        Ok(())
    }
    fn refresh_speculative_catalog(&mut self) {
        self.speculative_catalog = speculation::graph_catalog(
            &self.speculative_widths,
            self.speculative_fixed_width,
            self.limits.max_active.next_power_of_two(),
        );
    }
    /// Restricts scheduling to graph shapes present in the loaded execution manifest.
    pub fn speculative_graphs(&mut self, shapes: &[(usize, usize)]) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("speculative graphs must be selected before admission".into());
        }
        let mut catalog: [Vec<usize>; 7] = Default::default();
        for &(batch, width) in shapes {
            if !crate::execution::BATCH_CAPACITIES.contains(&batch)
                || batch > self.limits.max_active.next_power_of_two()
                || !self.speculative_widths.contains(&width)
                || self
                    .speculative_fixed_width
                    .is_some_and(|fixed| fixed != width && width != 1)
            {
                return Err("loaded speculative shape is outside runtime limits".into());
            }
            let widths = &mut catalog[batch.trailing_zeros() as usize];
            if widths.contains(&width) {
                return Err("duplicate speculative graph shape".into());
            }
            widths.push(width);
        }
        for widths in
            &mut catalog[..self.limits.max_active.next_power_of_two().ilog2() as usize + 1]
        {
            widths.sort_unstable();
            if let Some(fixed) = self.speculative_fixed_width.filter(|fixed| widths.contains(fixed)) {
                widths.retain(|w| *w == fixed);
            }
            if !self.speculative_widths.is_empty() && widths.is_empty() {
                return Err("missing speculative graph for a runtime batch capacity".into());
            }
        }
        self.speculative_catalog = catalog;
        Ok(())
    }
    pub fn proposal_source(&mut self, source: speculation::ProposalSource) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("proposal source must be configured before admission".into());
        }
        self.proposal_source = source;
        Ok(())
    }
    pub fn set_copy_proposal(
        &mut self,
        handle: RequestHandle,
        proposal: speculation::CopyProposal,
    ) -> Result<()> {
        let vocabulary = self.capabilities.vocabulary;
        let words = vocabulary.div_ceil(32);
        let request = self.request_mut(handle)?;
        if request.phase != RequestPhase::Decode
            || request.pending != 0
            || proposal.consumed != request.consumed_tokens
            || proposal.count > 7
            || proposal.tokens[..proposal.count]
                .iter()
                .any(|id| *id < 0 || *id >= vocabulary as i64)
            || proposal.tokens[proposal.count..].iter().any(|id| *id != -1)
            || proposal.grammar_masks.len()
                != if request.grammar_constrained {
                    (proposal.count + 1) * words
                } else {
                    0
                }
            || proposal
                .grammar_masks
                .chunks(words)
                .any(|mask| mask.iter().all(|word| *word == 0))
            || (request.host_stops && proposal.count != 0)
        {
            return Err("invalid copy proposal for committed request state".into());
        }
        request.copy_proposal = Some(proposal);
        Ok(())
    }
    pub fn enable_history_drafts(&mut self) -> Result<()> {
        if self.submitted != Fence(0) || self.admissions != 0 {
            return Err("decoded history must be enabled before admission".into());
        }
        self.decoded_history.get_or_insert_with(decoded_history::DecodedHistory::new);
        Ok(())
    }

    pub fn history_proposal(&self, context: &[u32], maximum: usize) -> ([i64; 7], usize) {
        self.decoded_history.as_ref().map_or(([-1; 7], 0), |history| history.lookup(context, maximum))
    }

    pub(crate) fn resume_decoded_sequence(
        &mut self,
        handle: RequestHandle,
        sequence: decoded_history::DecodedSequence,
    ) -> Result<()> {
        let request = self.request_mut(handle)?;
        if request.generated_tokens != 0 || request.pending != 0 {
            return Err("decoded sequence must be restored before execution".into());
        }
        request.decoded_sequence = sequence;
        Ok(())
    }
    pub fn graph_durations(&mut self, graphs: &[crate::execution::GraphDescriptor]) -> Result<()> {
        if self.submitted != Fence(0) || self.slots.iter().any(|s| s.request.is_some()) {
            return Err("graph durations must be configured before admission".into());
        }
        self.costs = cost::ActionCosts::prepared(graphs)?;
        self.costs
            .advance_to(self.now_ns, self.policy.estimate_ttl_ns);
        // Each graph contributes at most one candidate per request slot.
        self.candidates = std::cell::RefCell::new(Vec::with_capacity(self.costs.len() * REQUEST_LANES));
        Ok(())
    }

    pub fn estimated_duration_ns(&self, step: &ScheduledStep) -> Option<u64> {
        self.costs.get(step)
    }

    pub fn observe_execution(
        &mut self,
        step: &ScheduledStep,
        timing: crate::execution::ExecutionTiming,
    ) {
        self.costs.observe(step, timing);
    }

    pub fn submitted_timing(&self, fence: Fence) -> Result<SubmittedTiming> {
        self.submissions.get(&fence).map(|submission| submission.timing)
            .ok_or_else(|| "unknown scheduled submission".into())
    }

    pub fn acceptance_credit_ns(&self) -> f64 {
        self.refresh.acceptance_credit_ns
    }

    pub fn refresh_credit_ns(&self) -> f64 {
        self.refresh.credit_ns
    }

    pub fn idle(&mut self) -> Result<()> {
        if !self.submissions.is_empty() {
            return Err("idle requires no submitted execution".into());
        }
        self.refresh.idle();
        Ok(())
    }

    pub fn plan_requires_feedback(&self, plan: &ScheduledPlan) -> bool {
        plan.requires_feedback || !self.costs.is_fresh(cost::Key::from_step(plan))
            || plan.origin.is_probe()
    }

    fn selected_cost(&self, key: cost::Key) -> SelectedCost {
        let (retained_ns, observed_at_ns, fresh) = self.costs.observation(key).unwrap_or((0, None, false));
        SelectedCost { retained_ns, observed_at_ns, selected_at_ns: self.now_ns,
            freshness: if observed_at_ns.is_none() { CostFreshness::Unknown }
                else if fresh { CostFreshness::Fresh } else { CostFreshness::Stale } }
    }

    fn refresh_ready(&self) -> bool {
        self.refresh.ready()
    }

    fn refresh_episode(&self) -> Option<&refresh::Episode> {
        self.refresh.episode()
    }

    fn reconcile_refresh(&mut self) {
        let mut keep = [false; REQUEST_LANES];
        if let Some(episode) = self.refresh.episode() {
            for (slot, owner) in episode.owners.iter().enumerate() {
                let Some(owner) = owner else { continue; };
                keep[slot] = self.request(owner.handle).is_ok() && (
                    self.submissions.contains_key(&owner.fence)
                    || self.split.as_ref().and_then(|split| split.retained[slot]).is_some_and(|proposal|
                        proposal.request == owner.handle && proposal.consumed == owner.consumed
                            && proposal.fence == owner.fence));
            }
        }
        self.refresh.retain_owners(&keep);
    }
    pub fn submitted_step(&self, fence: Fence) -> Result<&ScheduledStep> {
        self.submissions
            .get(&fence)
            .map(|submission| &submission.step)
            .ok_or_else(|| "unknown scheduled submission".into())
    }
    pub fn cache(&self) -> &PrefixCheckpointCache {
        &self.cache
    }
    pub fn stats(&self) -> SchedulerStats {
        let cache = self.cache.stats();
        SchedulerStats {
            active_requests: self
                .slots
                .iter()
                .filter(|slot| slot.request.is_some())
                .count(),
            free_slots: self
                .slots
                .iter()
                .filter(|slot| {
                    slot.request.is_none() && !slot.retiring && slot.generation < i64::MAX as u64
                })
                .count(),
            retiring_slots: self.slots.iter().filter(|slot| slot.retiring).count(),
            free_pages: self.free_pages.len(),
            free_snapshots: self.free_snapshots.len(),
            snapshot_capacity: self.limits.snapshots as usize,
            cached_checkpoints: cache.published_checkpoints,
            cached_pages: cache.cached_pages,
            inbound_waiting: if self.tickets.is_empty() {
                0
            } else {
                self.slots
                    .iter()
                    .filter(|slot| {
                        slot.request
                            .as_ref()
                            .is_some_and(|request| request.phase == RequestPhase::Inbound)
                    })
                    .count()
            },
            ..self.stats
        }
    }
    pub fn completed(&self) -> Fence {
        self.completed
    }
    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }
    pub fn free_snapshot_count(&self) -> usize {
        self.free_snapshots.len()
    }
    pub fn has_available_slot(&self) -> bool {
        self.slots.iter().any(|slot| {
            slot.request.is_none() && !slot.retiring && slot.generation < i64::MAX as u64
        })
    }
    pub fn has_available_image_slot(&self) -> bool {
        !self.free_image_slots.is_empty()
    }
    pub fn request(&self, handle: RequestHandle) -> Result<&LiveRequest> {
        self.slots
            .get(handle.slot as usize)
            .and_then(|slot| slot.request.as_ref())
            .filter(|request| request.handle == handle)
            .ok_or_else(|| "stale or retired request handle".into())
    }
    fn request_mut(&mut self, handle: RequestHandle) -> Result<&mut LiveRequest> {
        let request = self
            .slots
            .get_mut(handle.slot as usize)
            .and_then(|slot| slot.request.as_mut())
            .filter(|request| request.handle == handle)
            .ok_or("stale or retired request handle")?;
        request.epoch = request
            .epoch
            .checked_add(1)
            .ok_or("request epoch exhausted")?;
        Ok(request)
    }

    fn ensure_resources(&mut self, pages: usize, snapshots: usize) -> Result<()> {
        loop {
            let need_pages = pages.saturating_sub(self.free_pages.len());
            let need_snapshots =
                snapshots
                    .saturating_sub(self.free_snapshots.len())
                    .max(usize::from(
                        snapshots > 0 && self.available_snapshot(pages).is_none(),
                    ));
            if need_pages == 0 && need_snapshots == 0 {
                return Ok(());
            }
            // Defer admission while enough backing is already fenced for reclamation.
            if self.pending_resources(pages, snapshots) {
                return Err(crate::Error::admission(
                    crate::AdmissionError::Capacity,
                    "request resources are awaiting GPU retirement",
                ));
            }
            let evicted = self.cache.evict(EvictionTarget {
                pages: need_pages,
                snapshots: need_snapshots,
            });
            self.stats.snapshot_evictions += evicted.retired.len() as u64;
            let update = self.cache.advance_completed(self.completed)?;
            self.apply_completion(update);
            if evicted.retired.is_empty() {
                return Err(crate::Error::admission(
                    crate::AdmissionError::Capacity,
                    "live request resources are exhausted or awaiting GPU retirement",
                ));
            }
        }
    }

    fn draft_pages(&self) -> usize {
        if !self.speculative_widths.is_empty()
            && self.proposal_source == speculation::ProposalSource::Mtp
        {
            self.capabilities.draft_pages_per_request
        } else {
            0
        }
    }

    /// Pages beyond its private ones that `request` needs to commit work ending at `through`.
    fn pages_through(&self, request: &LiveRequest, through: u64) -> usize {
        (through.min(request.execution_limit) as usize)
            .div_ceil(self.capabilities.page_tokens)
            .saturating_sub(request.prefix_tokens as usize / self.capabilities.page_tokens)
            .saturating_add(self.draft_pages())
            .saturating_sub(request.private_pages.len())
    }

    /// Tokens a decoding request may consume before its next completion: two rounds
    /// at the widest verification, as two submissions may be in flight.
    fn decode_horizon(&self) -> u64 {
        2 * self.speculative_widths.last().copied().unwrap_or(1) as u64
    }

    /// Whether a request's reservation covers its next decode rounds; reservations of
    /// requests that are not growable always cover their whole output.
    pub(super) fn fits_next_rounds(&self, request: &LiveRequest) -> bool {
        !request.growable_output
            || self.pages_through(request, request.scheduled_consumed + self.decode_horizon())
                <= request.reserved_pages.len()
    }

    fn has_starved_request(&self) -> bool {
        self.slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .any(|request| {
                request.phase == RequestPhase::Decode
                    && !request.preempted
                    && !self.fits_next_rounds(request)
            })
    }

    /// Grows each decoding growable reservation a page beyond its next rounds, evicting
    /// unpinned prefix pages when the free list is short. If a request still cannot run,
    /// the newest growable decoder without work in flight is preempted.
    fn refill_reservations(&mut self) -> Result<()> {
        let ahead = self.decode_horizon() + self.capabilities.page_tokens as u64;
        let mut starved = false;
        for slot in 0..self.slots.len() {
            let want = match self.slots[slot].request.as_ref() {
                Some(request)
                    if request.growable_output
                        && !request.preempted
                        && request.phase == RequestPhase::Decode =>
                {
                    self.pages_through(request, request.scheduled_consumed + ahead)
                        .saturating_sub(request.reserved_pages.len())
                }
                _ => continue,
            };
            if want == 0 {
                continue;
            }
            while self.free_pages.len() < want {
                if self.pending_page_count() >= want {
                    break;
                }
                let evicted = self.cache.evict(EvictionTarget {
                    pages: want - self.free_pages.len(),
                    snapshots: 0,
                });
                self.stats.snapshot_evictions += evicted.retired.len() as u64;
                let update = self.cache.advance_completed(self.completed)?;
                self.apply_completion(update);
                if evicted.retired.is_empty() {
                    break;
                }
            }
            let take = want.min(self.free_pages.len());
            // Reserved pages are taken from the back, so new pages go behind the remaining ones.
            let mut pages: Vec<_> = (0..take)
                .map(|_| self.free_pages.pop_first().unwrap())
                .rev()
                .collect();
            let request = self.slots[slot].request.as_mut().unwrap();
            pages.append(&mut request.reserved_pages);
            request.reserved_pages = pages;
            let request = self.slots[slot].request.as_ref().unwrap();
            if self.fits_next_rounds(request) {
                continue;
            }
            let held = request.private_pages.len() + request.reserved_pages.len();
            let missing = self
                .pages_through(request, request.scheduled_consumed + self.decode_horizon())
                - request.reserved_pages.len();
            if held + missing <= self.limits.pages as usize {
                if self.pending_page_count() < missing {
                    starved = true;
                }
            } else if request.pending == 0 {
                // No other request's pages would let it continue: finish it at this length.
                let handle = request.handle;
                self.finish_early(handle, false)?;
            }
        }
        if starved {
            self.preempt_newest()?;
        }
        Ok(())
    }

    /// Finishes the newest growable decoder without work in flight through its terminal
    /// checkpoint, when backing allows one, so its pages return to the cache or the free
    /// list. The runtime readmits the request's history instead of completing it.
    fn preempt_newest(&mut self) -> Result<()> {
        let victim = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .filter(|request| {
                request.growable_output
                    && !request.preempted
                    && request.phase == RequestPhase::Decode
                    && request.pending == 0
                    && !self.owns_proposal(request)
            })
            .max_by_key(|request| request.admission_order)
            .map(|request| request.handle);
        let Some(handle) = victim else {
            return Ok(());
        };
        self.stats.preemptions += 1;
        self.finish_early(handle, true)
    }

    /// Ends a decoding request before its output limit. As a finished request does, it
    /// returns its unused reservation and retains its state in a terminal checkpoint when
    /// KV backing allows; a preempted request is readmitted by the runtime.
    fn finish_early(&mut self, handle: RequestHandle, preempted: bool) -> Result<()> {
        let request = self.request_mut(handle)?;
        request.preempted = preempted;
        request.phase = RequestPhase::Finishing;
        if request.consumed_tokens <= request.prefix_tokens || request.error_code.is_some() {
            return Ok(());
        }
        let unused = std::mem::take(&mut request.reserved_pages);
        self.free_pages.extend(unused);
        if let Some(snapshot) = self.retain_snapshot()? {
            let request = self.request_mut(handle)?;
            request.terminal_snapshot = Some(snapshot);
            request.phase = RequestPhase::Snapshot;
        }
        Ok(())
    }

    pub fn admit(&mut self, spec: RequestSpec) -> Result<RequestHandle> {
        let plan = self.plan_admission(&spec)?;
        Ok(self.admit_planned(spec, plan, None)?.handle)
    }

    /// Validates the request, builds its prefix history, and looks up the GPU
    /// cache without reserving anything.
    pub fn plan_admission(&mut self, spec: &RequestSpec) -> Result<AdmissionPlan> {
        if spec.prompt.is_empty()
            || spec.message_boundaries.first() == Some(&0)
            || spec.message_boundaries.last().is_some_and(|n| *n > spec.prompt.len())
            || spec.message_boundaries.windows(2).any(|pair| pair[0] >= pair[1])
            || (spec.max_new_tokens == 0 && !spec.prompt_logprobs)
            || spec
                .prompt
                .len()
                .checked_add(spec.max_new_tokens as usize)
                .is_none_or(|count| count > self.limits.max_sequence_tokens)
        {
            return Err(crate::Error::admission(
                crate::AdmissionError::Invalid,
                "request requires a nonempty prompt, positive output limit, and bounded total tokens",
            ));
        }
        self.free_slot()?;
        let mut history = PrefixHistory::new(self.identity);
        history.append(&spec.prompt, &spec.conditioning)?;
        if spec.prompt_logprobs && self.scored_prefill_catalog.is_empty() {
            return Err(crate::Error::admission(crate::AdmissionError::Invalid,
                "prompt logprobs are not available in this model package"));
        }
        let lookup = if spec.prompt_logprobs {
            Lookup { checkpoint: None, consumed_tokens: 0, pages: Vec::new(), snapshot: None }
        } else { self.cache.lookup(&history)? };
        Ok(AdmissionPlan { history, lookup })
    }

    fn free_slot(&self) -> Result<usize> {
        self.slots
            .iter()
            .position(|slot| {
                slot.request.is_none() && !slot.retiring && slot.generation < i64::MAX as u64
            })
            .ok_or_else(|| {
                crate::Error::admission(
                    crate::AdmissionError::Capacity,
                    "all request slots are active, retired, or generation-exhausted",
                )
            })
    }

    /// Reserves resources and creates the request described by `plan`.
    ///
    /// Without `inbound` the plan's GPU lookup becomes the request's prefix. With
    /// `inbound` at consumed count `c`, the request reserves as if it had hit a
    /// GPU prefix of depth `c`, starts in `Inbound`, and waits for the restore of
    /// `inbound.key`. A new restore takes `inbound.pages` page ids and one
    /// snapshot from the free lists, pins the plan's GPU checkpoint until
    /// `finish_inbound`, and is returned as the ticket for the worker to fill; a
    /// restore already in flight gains this request as a waiter and `ticket` is
    /// `None`. A failure leaves no reservation, ticket, or request behind.
    pub fn admit_planned(
        &mut self,
        spec: RequestSpec,
        plan: AdmissionPlan,
        inbound: Option<InboundRequest>,
    ) -> Result<Admitted> {
        let AdmissionPlan { history, lookup } = plan;
        // Lookups count once per admitted request, not per attempt of a waiting one.
        let (hit, reused) = (lookup.checkpoint.is_some(), lookup.consumed_tokens);
        let page_tokens = self.capabilities.page_tokens as u64;
        if let Some(inbound) = &inbound {
            let consumed = inbound.consumed_tokens;
            if inbound.key.consumed_tokens != consumed
                || consumed <= lookup.consumed_tokens
                || history.key_at(consumed) != Some(inbound.key)
                || inbound.pages as u64
                    != consumed / page_tokens - lookup.consumed_tokens / page_tokens
            {
                return Err(
                    "inbound checkpoint does not extend the admission plan's prefix history".into(),
                );
            }
        }
        let slot = self.free_slot()?;
        let consumed = inbound
            .as_ref()
            .map_or(lookup.consumed_tokens, |inbound| inbound.consumed_tokens);
        let creating = inbound
            .as_ref()
            .is_some_and(|inbound| !self.tickets.contains_key(&inbound.key));
        // The request pins its own prefix; a new ticket pins the ancestor instead.
        let pinned = if inbound.is_none() || creating {
            lookup.checkpoint
        } else {
            None
        };
        if let Some(id) = pinned {
            self.cache.pin(id)?;
        }
        let extra = self.speculative_widths.last().copied().unwrap_or(1) - 1;
        let execution_limit = (spec.prompt.len() + spec.max_new_tokens as usize)
            .saturating_add(extra)
            .min(self.limits.max_sequence_tokens);
        let draft_page = if !self.speculative_widths.is_empty()
            && self.proposal_source == speculation::ProposalSource::Mtp
        {
            self.capabilities.draft_pages_per_request
        } else {
            0
        };
        // A growable request reserves its prompt and one page of output; decoding grows it.
        let admission_limit = if spec.growable_output {
            execution_limit.min(spec.prompt.len() + self.capabilities.page_tokens)
        } else {
            execution_limit
        };
        let reserved_count = admission_limit.div_ceil(self.capabilities.page_tokens)
            - (consumed / page_tokens) as usize
            + draft_page;
        if admission_limit.div_ceil(self.capabilities.page_tokens) + draft_page
            > self.limits.pages as usize
        {
            if let Some(id) = pinned {
                self.cache.unpin(id)?;
            }
            return Err(crate::Error::admission(
                crate::AdmissionError::Context,
                "request needs more KV pages than this server holds",
            ));
        }
        let (ticket_pages, ticket_snapshots) = match &inbound {
            Some(inbound) if creating => (inbound.pages, 1),
            _ => (0, 0),
        };
        let execution_rows = self.execution_rows();
        let starving = spec.growable_output && self.has_starved_request();
        if let Err(error) = if starving {
            Err(crate::Error::admission(
                crate::AdmissionError::Capacity,
                "running requests are waiting for KV pages",
            ))
        } else {
            self.ensure_resources(reserved_count + ticket_pages, ticket_snapshots)
        } {
            if let Some(id) = pinned {
                self.cache.unpin(id)?;
            }
            return Err(error);
        }
        let generation = self.slots[slot].generation + 1;
        let handle = RequestHandle {
            slot: slot as u32,
            generation,
        };
        let mut ticket = None;
        if let Some(inbound) = &inbound {
            if creating {
                let snapshot = self
                    .reserve_snapshot(reserved_count + ticket_pages)
                    .unwrap();
                let pages: Vec<_> = (0..inbound.pages)
                    .map(|_| self.free_pages.pop_first().unwrap())
                    .collect();
                self.tickets.insert(
                    inbound.key,
                    Ticket {
                        parent: lookup.checkpoint,
                        pages: pages.clone(),
                        snapshot,
                        history: history.clone(),
                        waiters: vec![handle],
                    },
                );
                ticket = Some(InboundTicket {
                    key: inbound.key,
                    pages,
                    slot: self.snapshot_slot(snapshot),
                });
            } else {
                self.tickets
                    .get_mut(&inbound.key)
                    .unwrap()
                    .waiters
                    .push(handle);
            }
            self.stats.host_hits += 1;
            self.stats.host_reused_tokens += consumed - lookup.consumed_tokens;
        }
        let mut reserved_pages: Vec<_> = (0..reserved_count)
            .map(|_| self.free_pages.pop_first().unwrap())
            .collect();
        reserved_pages.reverse();
        let (phase, prefix, prefix_snapshot, shared_pages) = if inbound.is_some() {
            (RequestPhase::Inbound, None, None, Vec::new())
        } else if lookup.checkpoint.is_some() {
            (
                RequestPhase::Restore,
                lookup.checkpoint,
                lookup.snapshot,
                lookup.pages,
            )
        } else {
            (RequestPhase::Initialize, None, None, Vec::new())
        };
        self.slots[slot].generation = generation;
        self.slots[slot].request = Some(LiveRequest {
            image_slot: None,
            conditioned_features: None,
            conditioned_inputs_ready: true,
            conditioned_windows: VecDeque::new(),
            window_snapshot: None,
            window_snapshot_ready: false,
            window_restore: false,
            epoch: 0,
            acceptance: speculation::AcceptanceProfile::default(),
            accounting: self.accounting(),
            handle,
            state_slot: StateSlotId((1 + slot * execution_rows) as u32),
            terminal_snapshot: None,
            phase,
            consumed_tokens: consumed,
            generated_tokens: 0,
            prompt_tokens: spec.prompt.len(),
            grammar_constrained: spec.grammar_constrained,
            host_stops: spec.host_stops,
            prompt_logprobs: spec.prompt_logprobs,
            execution_limit: execution_limit as u64,
            error_code: None,
            copy_proposal: None,
            decoded_sequence: Default::default(),
            history,
            tokens: spec.prompt,
            prefix,
            prefix_tokens: consumed,
            admission_reused_tokens: consumed,
            message_boundaries: spec.message_boundaries,
            message_snapshot: None,
            prompt_snapshot: None,
            boundary_capture: None,
            prefix_snapshot,
            shared_pages,
            private_pages: Vec::new(),
            reserved_pages,
            max_new_tokens: spec.max_new_tokens,
            scheduled_consumed: consumed,
            last_use: Fence(0),
            pending: 0,
            terminal_snapshot_ready: false,
            vision_patches: VecDeque::new(),
            prefill_spans: Vec::new(),
            growable_output: spec.growable_output,
            preempted: false,
            admission_order: self.admissions,
        });
        self.admissions += 1;
        self.stats.cache_lookups += u64::from(!spec.prompt_logprobs);
        self.stats.cache_hits += u64::from(hit);
        self.stats.reused_tokens += reused;
        Ok(Admitted { handle, ticket })
    }

    /// Completes the restore for `key`.
    ///
    /// On success the ticket's pages and snapshot become a GPU checkpoint under
    /// the ticket's ancestor, ready at the completed fence, or return to the free
    /// lists when the cache already holds that key; every waiter is pinned to the
    /// resulting checkpoint and proceeds through `Restore` as a GPU prefix hit,
    /// and the returned list is empty. On failure the ticket's resources return
    /// to the free lists and every waiter is retired immediately, as a cancelled
    /// request that never executed; the returned handles identify them so the
    /// caller can admit them again.
    pub fn finish_inbound(&mut self, key: PrefixKey, success: bool) -> Result<Vec<RequestHandle>> {
        let ticket = self
            .tickets
            .remove(&key)
            .ok_or("no inbound restore is in flight for the supplied prefix key")?;
        if !success {
            return self.fail_ticket(ticket);
        }
        let proposal = Publication {
            parent: ticket.parent,
            consumed_tokens: key.consumed_tokens,
            tail_pages: ticket.pages.clone(),
            snapshot: StateSnapshot {
                id: ticket.snapshot,
                consumed_tokens: key.consumed_tokens,
            },
            ready_after: self.completed,
        };
        let result = match self.cache.publish(&ticket.history, &proposal) {
            Ok(result) => result,
            Err(error) => {
                self.fail_ticket(ticket)?;
                return Err(error);
            }
        };
        let (id, ready) = match result {
            PublishResult::Created { id, ready } => (id, ready),
            PublishResult::Existing { id, ready } => {
                self.free_pages.extend(ticket.pages.iter().copied());
                self.release_snapshot(ticket.snapshot);
                (id, ready)
            }
        };
        if let Some(parent) = ticket.parent {
            self.cache.unpin(parent)?;
        }
        if !ready {
            // The checkpoint becomes ready at a later fence; the waiters are
            // re-admitted rather than left waiting without a ticket.
            for handle in &ticket.waiters {
                self.retire(*handle, false)?;
            }
            return Ok(ticket.waiters);
        }
        if matches!(result, PublishResult::Created { .. }) {
            self.stats.snapshot_publications += 1;
            self.record_publication(id);
        }
        let snapshot = self.cache.checkpoint(id)?.snapshot();
        let shared_pages = self.cache.page_chain(id)?;
        for handle in ticket.waiters {
            self.request(handle)?;
            self.cache.pin(id)?;
            let request = self.request_mut(handle)?;
            request.prefix = Some(id);
            request.prefix_tokens = key.consumed_tokens;
            request.prefix_snapshot = Some(snapshot);
            request.shared_pages = shared_pages.clone();
            request.phase = RequestPhase::Restore;
        }
        Ok(Vec::new())
    }

    fn fail_ticket(&mut self, ticket: Ticket) -> Result<Vec<RequestHandle>> {
        self.free_pages.extend(ticket.pages);
        self.release_snapshot(ticket.snapshot);
        if let Some(parent) = ticket.parent {
            self.cache.unpin(parent)?;
        }
        for handle in &ticket.waiters {
            self.retire(*handle, false)?;
        }
        Ok(ticket.waiters)
    }

    /// Queue GPU-ready checkpoint notifications for an attached host store.
    pub fn enable_publications(&mut self) {
        self.track_publications = true;
    }

    /// Checkpoints that became GPU-ready since tracking was enabled or the
    /// previous call, in that order and each once, omitting retired checkpoints.
    pub fn take_publications(&mut self) -> Vec<PublishedCheckpoint> {
        std::mem::take(&mut self.publications)
            .into_iter()
            .filter_map(|id| self.publication(id))
            .collect()
    }

    pub(crate) fn publication(&self, id: CheckpointId) -> Option<PublishedCheckpoint> {
        let checkpoint = self.cache.checkpoint(id).ok()?;
        let parent = match checkpoint.parent() {
            Some(parent) => Some(self.cache.checkpoint(parent).ok()?.prefix()),
            None => None,
        };
        Some(PublishedCheckpoint {
            id,
            key: checkpoint.prefix(),
            parent,
            bucket: checkpoint.bucket(),
            chain_pages: self.cache.page_chain(id).ok()?,
            tail_pages: checkpoint.tail_pages().len(),
            slot: self.snapshot_slot(checkpoint.snapshot()),
        })
    }

    fn record_publication(&mut self, id: CheckpointId) {
        if !self.track_publications {
            return;
        }
        // Bound queued notifications when the store delays draining them.
        if self.publications.len() >= 2 * self.limits.snapshots as usize {
            self.publications
                .retain(|id| self.cache.checkpoint(*id).is_ok());
        }
        self.publications.push(id);
    }

    fn apply_completion(&mut self, update: CompletionUpdate) {
        self.stats.snapshot_publications += update.published.len() as u64;
        for id in update.published {
            self.record_publication(id);
        }
        self.free_pages.extend(update.reusable.pages);
        for snapshot in update.reusable.snapshots {
            self.release_snapshot(snapshot);
        }
    }

    /// Pins a GPU-ready checkpoint for the worker's copy-out; eviction and
    /// compaction skip it until `unpin_checkpoint`.
    pub fn pin_checkpoint(&mut self, id: CheckpointId) -> Result<()> {
        self.cache.pin(id)
    }

    pub fn unpin_checkpoint(&mut self, id: CheckpointId) -> Result<()> {
        self.cache.unpin(id)
    }

    pub fn record_host_lookup(&mut self) {
        self.stats.host_lookups += 1;
    }

    /// Keep model-declared bidirectional spans inside one prefill submission.
    pub fn set_prefill_spans(
        &mut self,
        handle: RequestHandle,
        spans: &[std::ops::Range<usize>],
    ) -> Result<()> {
        let capacity = self.limits.prefill_chunk_size;
        let request = self.request_mut(handle)?;
        if request.last_use != Fence(0)
            || !request.prefill_spans.is_empty()
            || spans.windows(2).any(|p| p[0].end > p[1].start)
            || spans.iter().any(|s| {
                s.start >= s.end
                    || s.end > request.prompt_tokens
                    || s.end - s.start > capacity
                    || (s.start < request.consumed_tokens as usize
                        && (request.consumed_tokens as usize) < s.end)
                    || request
                        .message_boundaries
                        .iter().any(|&n| s.start < n && n < s.end)
            })
        {
            return Err("invalid atomic prefill spans or prefix boundary".into());
        }
        request.prefill_spans.extend_from_slice(spans);
        Ok(())
    }

    pub(crate) fn conditioned_inputs_ready(
        &mut self,
        handle: RequestHandle,
        ready: bool,
    ) -> Result<()> {
        let request = self.request_mut(handle)?;
        request.conditioned_inputs_ready = ready;
        request.epoch += 1;
        Ok(())
    }

    /// Maps immutable prompt feature indices into the compact execution arena.
    #[cfg(test)]
    pub(crate) fn set_conditioned_features(
        &mut self,
        handle: RequestHandle,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<()> {
        let request = self.request_mut(handle)?;
        if request.last_use != Fence(0)
            || request.conditioned_features.is_some()
            || ranges.iter().any(|range| range.is_empty())
            || ranges.windows(2).any(|pair| pair[0].end > pair[1].start)
        {
            return Err("invalid conditioned feature execution ranges".into());
        }
        request.conditioned_features = Some(ranges.to_vec());
        Ok(())
    }

    pub(crate) fn set_conditioned_inputs(
        &mut self,
        handle: RequestHandle,
        inputs: &[ConditionedInput],
    ) -> Result<()> {
        let maximum = self
            .capabilities
            .vision
            .patch_capacities
            .last()
            .copied()
            .unwrap_or(0);
        let capacity = self.capabilities.vision.features_per_request;
        let merge = self.capabilities.vision.patches_per_feature;
        let request = self.request(handle)?;
        if request.last_use != Fence(0)
            || request.conditioned_features.is_some()
            || inputs.windows(2).any(|pair| {
                pair[0].token_span.end > pair[1].token_span.start
                    || pair[0].feature_range.end > pair[1].feature_range.start
            })
            || inputs.iter().any(|input| {
                input.token_span.is_empty()
                    || input.token_span.end > request.prompt_tokens
                    || input.token_span.end <= request.consumed_tokens as usize
                    || input.feature_range.is_empty()
                    || input.feature_range.len() > capacity
                    || input.patches == 0
                    || input.patches > maximum
                    || merge == 0
                    || input.patches % merge != 0
                    || input.feature_range.len() != input.patches / merge
            })
        {
            return Err("conditioned inputs violate prompt or prepared vision geometry".into());
        }
        let mut windows: VecDeque<ConditionedWindow> = VecDeque::new();
        let mut features = 0usize;
        for (index, input) in inputs.iter().enumerate() {
            if windows.is_empty() || features + input.feature_range.len() > capacity {
                if let Some(previous) = windows.back_mut() {
                    previous.prefill_end = input.token_span.start;
                }
                windows.push_back(ConditionedWindow {
                    inputs: index..index,
                    token_start: input.token_span.start,
                    prefill_end: request.prompt_tokens,
                    features: vec![],
                    patches: VecDeque::new(),
                });
                features = 0;
            }
            let window = windows.back_mut().unwrap();
            window.inputs.end = index + 1;
            window.features.push(input.feature_range.clone());
            window.patches.push_back(input.patches);
            features += input.feature_range.len();
        }
        if windows.iter().any(|window| request.prefill_spans.iter().any(|span| {
            (span.start < window.token_start && window.token_start < span.end)
                || (span.start < window.prefill_end && window.prefill_end < span.end)
        })) {
            return Err("conditioned input windows cannot split atomic prefill spans".into());
        }
        if handle
            .generation
            .checked_add(windows.len().saturating_sub(1) as u64)
            .is_none_or(|generation| generation > i64::MAX as u64)
        {
            return Err("request generation cannot cover its conditioned input windows".into());
        }
        if let Some(window) = windows.front() {
            self.set_vision_images(handle, &window.patches.iter().copied().collect::<Vec<_>>())?;
        }
        if windows.len() > 1 {
            // Unsubmitted page reservations may be rebound around the snapshot's backing pages.
            let pages = std::mem::take(&mut self.request_mut(handle)?.reserved_pages);
            let count = pages.len();
            self.free_pages.extend(pages);
            let resources = self.ensure_resources(count, 1);
            let snapshot = if resources.is_ok() {
                self.reserve_snapshot(count)
            } else {
                None
            };
            let mut reserved: Vec<_> = (0..count)
                .map(|_| self.free_pages.pop_first().unwrap())
                .collect();
            reserved.reverse();
            self.request_mut(handle)?.reserved_pages = reserved;
            if let Err(error) = resources {
                let cache = self.cache.stats();
                let mut prefix = self.request(handle)?.prefix;
                let mut own_checkpoints = 0;
                let mut exclusive_prefix = true;
                while let Some(id) = prefix {
                    let checkpoint = self.cache.checkpoint(id)?;
                    exclusive_prefix &= checkpoint.pin_count() == 1;
                    own_checkpoints += 1;
                    prefix = checkpoint.parent();
                }
                let quiescent = exclusive_prefix
                    && cache.published_checkpoints == own_checkpoints
                    && cache.pending_checkpoints == 0
                    && cache.retired_pages == 0
                    && cache.retired_snapshots == 0
                    && self.slots.iter().filter(|slot| slot.request.is_some()).count() == 1
                    && self.tickets.is_empty()
                    && self.submissions.is_empty()
                    && self.retired.is_empty();
                if quiescent {
                    return Err(crate::Error::admission(
                        crate::AdmissionError::Context,
                        "request pages and a conditioned input checkpoint exceed prepared storage",
                    ));
                }
                return Err(error);
            }
            self.request_mut(handle)?.window_snapshot =
                Some(snapshot.ok_or("conditioned input window lacks reserved snapshot storage")?);
        }
        let request = self.request_mut(handle)?;
        request.conditioned_features = Some(
            windows
                .front()
                .map_or_else(Vec::new, |window| window.features.clone()),
        );
        request.conditioned_inputs_ready = inputs.is_empty();
        request.conditioned_windows = windows;
        Ok(())
    }

    pub(crate) fn resume_conditioned_window(
        &mut self,
        handle: RequestHandle,
    ) -> Result<RequestHandle> {
        let request = self.request(handle)?;
        if !request.window_snapshot_ready
            || request.window_snapshot.is_none()
            || request.pending != 0
            || request.last_use > self.completed
            || request.generated_tokens != 0
            || request.conditioned_windows.len() < 2
            || request.boundary_capture.is_some()
        {
            return Err("conditioned window restore requires a completed private checkpoint before generation".into());
        }
        let generation = handle
            .generation
            .checked_add(1)
            .filter(|generation| *generation <= i64::MAX as u64)
            .ok_or("request generation exhausted")?;
        self.slots[handle.slot as usize].generation = generation;
        let next_handle = RequestHandle {
            slot: handle.slot,
            generation,
        };
        let request = self.request_mut(handle)?;
        request.handle = next_handle;
        request.conditioned_windows.pop_front();
        let window = request.conditioned_windows.front().unwrap();
        request.conditioned_features = Some(window.features.clone());
        request.vision_patches = window.patches.clone();
        request.conditioned_inputs_ready = false;
        request.window_snapshot_ready = false;
        request.window_restore = true;
        request.phase = RequestPhase::Restore;
        self.reconcile_refresh();
        Ok(next_handle)
    }

    pub fn set_vision_images(&mut self, handle: RequestHandle, patches: &[usize]) -> Result<()> {
        let vision = &self.capabilities.vision;
        let maximum = *vision
            .patch_capacities
            .last()
            .ok_or("vision is unsupported")?;
        let maximum_features = vision.features_per_request;
        let merge = vision.patches_per_feature;
        if patches.iter().try_fold(0usize, |total, count| total.checked_add(count / merge))
            .is_none_or(|total| total > maximum_features)
        {
            return Err("images exceed prepared feature storage".into());
        }
        let request = self.request_mut(handle)?;
        if request.last_use != Fence(0)
            || !request.vision_patches.is_empty()
            || request.image_slot.is_some()
            || patches.is_empty()
            || patches
                .iter()
                .any(|count| *count == 0 || *count > maximum || *count % merge != 0)
        {
            return Err("vision images must be assigned once before request initialization".into());
        }
        let slot = self.free_image_slots.pop_first().ok_or_else(|| crate::Error::admission(
            crate::AdmissionError::Capacity, "image feature slots are occupied or awaiting GPU retirement"))?;
        let request = self.request_mut(handle)?;
        request.image_slot = Some(slot);
        request.vision_patches.extend(patches.iter().copied());
        Ok(())
    }

    /// Plans one operation without submitting GPU work or reserving new pages.
    pub fn plan(&self) -> Option<ScheduledPlan> {
        self.select_plan()
    }

    fn speculative_tokens(&self, request: &LiveRequest, width: usize) -> usize {
        let context = self.limits.max_sequence_tokens - request.scheduled_consumed as usize;
        let output = (request.max_new_tokens - request.generated_tokens) as usize;
        let proposal = if request.host_stops
            || (request.grammar_constrained
                && self.proposal_source == speculation::ProposalSource::Mtp)
        {
            1
        } else if self.proposal_source == speculation::ProposalSource::Copy {
            request.copy_proposal().map_or(1, |p| p.count + 1)
        } else {
            width
        };
        width.min(context).min(output).min(proposal)
    }

    /// Records resource use before enqueueing the matching GPU operation.
    /// A CUDA submission failure requires stream drain before retirement can advance.
    pub fn commit(&mut self, plan: &ScheduledPlan, fence: Fence) -> Result<()> {
        if !Arc::ptr_eq(&plan.owner, &self.plan_owner) {
            return Err("execution plan belongs to another scheduler".into());
        }
        let step = &plan.step;
        step.validate_mixed()?;
        if (step.kind == GraphKind::Proposal && Some(step.token_capacity) != self.costs.proposal_width)
            || (step.kind == GraphKind::PackedSpeculative && Some(step.draft_tokens + 1) != self.costs.proposal_width) {
            return Err("proposal execution requires the full configured draft block".into());
        }
        if step.kind == GraphKind::Verification {
            if step.lanes.iter().any(|lane| {
                !self.request(lane.request).ok()
                    .and_then(|request| self.retained_proposal(request))
                    .is_some_and(|p| p.fence < fence
                        && p.width <= step.token_capacity && lane.tokens == p.width)
            }) {
                return Err(
                    "verification requires each request's retained proposal and token limit".into(),
                );
            }
        }
        if matches!(step.kind, GraphKind::Proposal | GraphKind::PackedSpeculative | GraphKind::Speculative | GraphKind::Mixed) && step.lanes.iter().any(|lane| {
            self.request(lane.request).ok()
                .and_then(|request| self.retained_proposal(request)).is_some()
        }) {
            return Err("request already owns a retained proposal".into());
        }
        if fence <= self.submitted || fence.0 > i64::MAX as u64 {
            return Err("submission fences must increase within int64 range".into());
        }
        let key = cost::Key::from_step(step);
        let observed_at_ns = self.costs.observation(key).and_then(|(_, observed, _)| observed);
        let owned = step.kind == GraphKind::Verification && self.refresh.episode().is_some_and(|episode|
            step.lanes.iter().any(|lane| self.request(lane.request).ok()
                .and_then(|request| self.retained_proposal(request))
                .is_some_and(|proposal| episode.owns(lane.request, proposal.consumed, proposal.fence))));
        let origin = match plan.origin {
            PlanOrigin::Ordinary | PlanOrigin::Discovery if owned =>
                self.refresh.episode().unwrap().continuation(),
            PlanOrigin::RefreshContinuation(_) | PlanOrigin::AcceptanceContinuation(_) if !owned =>
                return Err("refresh verification must consume an owned proposal".into()),
            PlanOrigin::Ordinary if observed_at_ns.is_none() => PlanOrigin::Discovery,
            origin => origin,
        };
        self.refresh.validate(origin)?;
        if origin == PlanOrigin::AcceptanceStart
            && (self.speculative_fixed_width.is_some() || step.token_capacity <= 1
                || !matches!(step.kind, GraphKind::Speculative | GraphKind::Proposal | GraphKind::PackedSpeculative | GraphKind::Mixed)) {
            return Err("acceptance learning requires an adaptive speculative choice".into());
        }
        if let PlanOrigin::RefreshStart(target) = origin {
            if self.costs.duration_key(target).is_none_or(|duration| duration == 0) || self.costs.is_fresh(target) {
                return Err("refresh target must have a stale measured cost".into());
            }
            if target != key && !(step.kind == GraphKind::Proposal && target.kind == GraphKind::Verification) {
                return Err("refresh must execute its physical target or an enabling proposal".into());
            }
        }
        let feedback = self.plan_requires_feedback(plan);
        if self.submissions.len() >= 2
            || (feedback && !self.submissions.is_empty())
            || self.submissions.values().any(|step| step.requires_feedback)
        {
            return Err("submission window or feedback barrier is full".into());
        }
        if !self.submission_window_available(plan.estimated_duration_ns) {
            return Err("in-flight execution requires completion feedback".into());
        }
        for (lane, epoch) in step.lanes.iter().zip(&plan.epochs) {
            if self.request(lane.request)?.epoch != *epoch {
                return Err("execution plan has stale request state".into());
            }
        }
        let mut needed = Vec::new();
        for lane in &step.lanes {
            let request = self.request(lane.request)?;
            let through = lane
                .start_position
                .checked_add(lane.tokens as u64)
                .ok_or("consumed token overflow")?;
            if through > self.limits.max_sequence_tokens as u64 {
                return Err("scheduled request exceeds sequence capacity".into());
            }
            let pages = self.pages_through(request, through);
            if pages > request.reserved_pages.len() {
                return Err("scheduled step exceeds the request's reserved page budget".into());
            }
            needed.push((lane.request, pages, through));
        }
        let participants = self.service_participants();
        let proposal_decision = (step.kind == GraphKind::Proposal).then(|| self.proposal_decision());
        let submitted_timing = SubmittedTiming {
            retained_ns: plan.selected_cost.retained_ns,
            observed_at_ns: plan.selected_cost.observed_at_ns,
            freshness: plan.selected_cost.freshness,
            origin: match origin {
                PlanOrigin::Ordinary => TimingOrigin::Ordinary,
                PlanOrigin::Discovery => TimingOrigin::Discovery,
                PlanOrigin::AcceptanceStart | PlanOrigin::AcceptanceContinuation(_) => TimingOrigin::Acceptance,
                _ => TimingOrigin::Refresh,
            },
            credit_ns: if origin.is_acceptance() { self.refresh.acceptance_credit_ns } else { self.refresh.credit_ns }, commit_ns: self.now_ns,
            selected_at_ns: plan.selected_cost.selected_at_ns,
            refresh_episode: match origin {
                PlanOrigin::RefreshStart(_) | PlanOrigin::AcceptanceStart => Some(fence.0),
                PlanOrigin::RefreshContinuation(id) | PlanOrigin::AcceptanceContinuation(id) => Some(id),
                _ => None,
            },
        };
        for (handle, pages, through) in needed {
            let request = self.request_mut(handle)?;
            let allocated: Vec<_> = (0..pages)
                .map(|_| request.reserved_pages.pop().unwrap())
                .collect();
            request.private_pages.extend(allocated);
            request.scheduled_consumed = through;
            request.last_use = fence;
            request.pending += 1;
            if let Some(prefix) = request.prefix {
                self.cache.mark_used(prefix, fence)?;
            }
        }
        self.submitted = fence;
        self.refresh.commit(origin, fence, step);
        self.costs.record_attempt(key);
        if let PlanOrigin::RefreshStart(target) = origin {
            self.costs.record_attempt(target);
        }
        if let Some(decision) = proposal_decision {
            self.split.as_mut().unwrap().pending_decision = Some((fence, decision));
        }
        if let Some(tokens) = plan.verification_budget {
            self.split.as_mut().unwrap().pending_budget = Some((fence, tokens));
        }
        if step.kind == GraphKind::Verification {
            for lane in &step.lanes {
                self.split.as_mut().unwrap().retained[lane.request.slot as usize] = None;
            }
        }
        let expected_finish_ns = self
            .now_ns
            .saturating_add(self.window_remaining().unwrap_or(0))
            .saturating_add(plan.estimated_duration_ns);
        self.submissions.insert(
            fence,
            Submission {
                step: ScheduledStep { requires_feedback: feedback, ..step.clone() },
                expected_finish_ns,
                estimated_duration_ns: plan.estimated_duration_ns,
                participants,
                origin,
                timing: submitted_timing,
            },
        );
        Ok(())
    }

    pub fn complete(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming,
    ) -> Result<Vec<GeneratedToken>> {
        self.complete_with_survival(fence, records, timing, None)
    }

    pub fn complete_with_survival(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming,
        survival: Option<&[[f32; 7]]>,
    ) -> Result<Vec<GeneratedToken>> {
        if self.submissions.get(&fence).is_some_and(|s| s.kind == GraphKind::PackedSpeculative) {
            return Err("packed combined completion requires its terminal confidence sidecar".into());
        }
        self.complete_inner(fence, records, timing, survival)
    }

    pub fn complete_packed_round(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming,
        confidence: &[[f32; 7]],
    ) -> Result<Vec<GeneratedToken>> {
        self.complete_packed_round_with_survival(fence, records, timing, confidence, None)
    }

    pub fn complete_packed_round_with_survival(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming,
        confidence: &[[f32; 7]],
        survival: Option<&[[f32; 7]]>,
    ) -> Result<Vec<GeneratedToken>> {
        let step = self.submissions.get(&fence)
            .ok_or("packed combined completion has no submission")?.step.clone();
        if step.kind != GraphKind::PackedSpeculative || confidence.len() < step.batch_capacity
            || step.lanes.iter().enumerate().any(|(row, _)| confidence[row][..step.draft_tokens]
                .iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))) {
            return Err("invalid packed combined confidence sidecar".into());
        }
        let generated = self.complete_inner(fence, records, timing, survival)?;
        self.observe_packed_forecasts(&step, records, confidence);
        Ok(generated)
    }

    fn complete_inner(
        &mut self,
        fence: Fence,
        records: &[CompletionRecord],
        timing: crate::execution::ExecutionTiming,
        survival: Option<&[[f32; 7]]>,
    ) -> Result<Vec<GeneratedToken>> {
        if records.len() < self.limits.request_capacity() || records.len() > REQUEST_LANES {
            return Err("completion does not cover the prepared request capacity".into());
        }
        if let Some(survival) = survival {
            crate::execution::validate_speculative_survival(survival, records)?;
        }
        if self.submissions.first_key_value().map(|(next, _)| *next) != Some(fence) {
            return Err("completion fence must match the oldest submitted step".into());
        }
        let step = self.submissions.get(&fence).unwrap();
        if records.iter().enumerate().any(|(slot, record)| {
            record.status != 0
                && !step
                    .lanes
                    .iter()
                    .any(|lane| lane.request.slot as usize == slot)
        }) {
            return Err(
                "completion contains an active physical slot outside the submitted lanes".into(),
            );
        }
        // Validate every live lane before committing any completion state.
        if step.kind == GraphKind::PackedSpeculative {
            let mut physical = 0usize;
            for lane in &step.lanes {
                let record = &records[lane.request.slot as usize];
                record.validate_vector()?;
                let vector = record.vector.ok_or("packed combined completion requires ABI3")?;
                if record.status != 0 && record.status != 3 {
                    let offered = usize::try_from(vector.offered_drafts)
                        .map_err(|_| "negative packed prefix offer")?;
                    if offered >= lane.tokens || vector.verification_width != step.token_capacity as i64 {
                        return Err("packed combined completion exceeds its frozen lane ceiling".into());
                    }
                    physical = physical.checked_add(offered + 1).ok_or("packed row count overflow")?;
                }
            }
            if physical > step.packed_token_capacity {
                return Err("packed combined completion exceeds its physical row budget".into());
            }
        }
        for (row, lane) in step.lanes.iter().enumerate() {
            let kind = step.lane_kind(row);
            records[lane.request.slot as usize].validate_vector()?;
            let Ok(request) = self.request(lane.request) else {
                continue;
            };
            if matches!(
                kind,
                GraphKind::Init
                    | GraphKind::Restore
                    | GraphKind::Snapshot
                    | GraphKind::Vision
                    | GraphKind::Proposal
            ) {
                let record = records[lane.request.slot as usize];
                if !(1..=3).contains(&record.status)
                    || record.generation != lane.request.generation as i64
                    || record.submission_sequence != fence.0 as i64
                    || (record.status != 3
                        && (record.consumed_tokens != request.consumed_tokens as i64
                            || record.generated_tokens != request.generated_tokens as i64
                            || record.token_id != -1
                            || record.vector.is_some_and(|v| v.emitted_count != 0)))
                {
                    return Err(
                        "state-transfer completion is inactive or has mismatched generation, sequence or counters".into(),
                    );
                }
                continue;
            }
            let record = records[lane.request.slot as usize];
            if let Some(survival) = survival {
                let observed = if matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative) && record.status != 3 {
                    let vector = record.vector.ok_or("speculative survival requires a completion vector")?;
                    let offered = (vector.offered_drafts as usize).min(lane.tokens.saturating_sub(1));
                    if vector.emitted_count > vector.accepted_drafts { offered }
                    else { (vector.accepted_drafts as usize).min(offered) }
                } else { 0 };
                if survival[lane.request.slot as usize].iter().enumerate()
                    .any(|(depth, p)| (*p != -1.0) != (depth < observed)) {
                    return Err("speculative survival does not match the observed logical prefix".into());
                }
            }
            let emitted = record.generated_tokens - request.generated_tokens as i64;
            let width = if matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative) {
                step.token_capacity as i64
            } else {
                1
            };
            if let Some(vector) = record.vector {
                if vector.emitted_count != emitted
                    || (matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative)
                        && (vector.verification_width != width
                            || record.consumed_tokens - request.consumed_tokens as i64 != emitted))
                {
                    return Err("token vector does not match committed request progress".into());
                }
            } else if matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative) {
                return Err("speculative completion requires ABI3 token vector".into());
            }
            if record.generation != lane.request.generation as i64
                || record.submission_sequence != fence.0 as i64
                || !(1..=3).contains(&record.status)
                || record.consumed_tokens < request.consumed_tokens as i64
                || record.consumed_tokens as u64 > lane.start_position + lane.tokens as u64
                || record.generated_tokens < request.generated_tokens as i64
                || record.generated_tokens > request.generated_tokens as i64 + width
                || record.generated_tokens > request.max_new_tokens as i64
                || (record.status == 1
                    && !matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative)
                    && record.consumed_tokens as u64 != lane.start_position + lane.tokens as u64)
                || (record.status == 1
                    && matches!(
                        kind,
                        GraphKind::Decode
                            | GraphKind::Speculative
                            | GraphKind::Verification
                            | GraphKind::PackedSpeculative
                            | GraphKind::Resume
                    )
                    && record.generated_tokens == request.generated_tokens as i64)
                || (record.status == 1
                    && matches!(
                        request.phase,
                        RequestPhase::Finishing | RequestPhase::Snapshot
                    ))
                || (record.generated_tokens > request.generated_tokens as i64
                    && u32::try_from(record.token_id).is_err())
            {
                return Err(
                    "completion counters, generation or token do not match the live request".into(),
                );
            }
        }
        self.refresh.validate_completion(step.origin, timing)?;
        let submission = self.submissions.remove(&fence).unwrap();
        self.refresh.complete(submission.origin, timing);
        self.observe_execution(&submission.step, timing);
        let step = submission.step;
        let now_ns = self.now_ns;
        let proposal_source = self.proposal_source;
        let mut useful_tokens = [0u64; REQUEST_LANES];
        let mut generated = Vec::new();
        let mut boundaries = Vec::new();
        let mut captures = Vec::new();
        let mut window_snapshots = Vec::new();
        for (row, lane) in step.lanes.iter().enumerate() {
            let kind = step.lane_kind(row);
            let Ok(request) = self.request_mut(lane.request) else {
                continue;
            };
            request.pending -= 1;
            if matches!(
                kind,
                GraphKind::Init
                    | GraphKind::Restore
                    | GraphKind::Snapshot
                    | GraphKind::Vision
                    | GraphKind::Proposal
            ) {
                let record = records[lane.request.slot as usize];
                if record.status == 3 {
                    request.error_code = Some(record.error_code);
                    request.terminal_snapshot_ready = false;
                    request.phase = RequestPhase::Finishing;
                } else if kind == GraphKind::Proposal {
                    request.scheduled_consumed = request.consumed_tokens;
                } else if kind == GraphKind::Snapshot {
                    if request.boundary_capture.is_some() {
                        boundaries.push(lane.request);
                    } else {
                        request.terminal_snapshot_ready = true;
                        request.phase = RequestPhase::Finishing;
                    }
                } else {
                    if kind == GraphKind::Vision {
                        request.vision_patches.pop_front();
                    }
                    if kind == GraphKind::Restore && request.window_restore {
                        request.window_restore = false;
                        if request.conditioned_windows.len() == 1 {
                            window_snapshots.extend(request.window_snapshot.take());
                        }
                    }
                    request.phase = request.prompt_phase();
                }
                continue;
            }
            let record = records[lane.request.slot as usize];
            let finished =
                record.status == 3 || ((record.status >= 2 || record.generated_tokens == request.max_new_tokens as i64)
                    && record.consumed_tokens >= request.prompt_tokens as i64);
            if matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative)
                && record.status != 3
            {
                let vector = record.vector.unwrap();
                // ABI3 may count padded proposals in the physical capture. Only
                // the lane's logical draft extent supplies acceptance observations.
                let offered = (vector.offered_drafts as usize).min(lane.tokens.saturating_sub(1));
                request.acceptance.observe(
                    proposal_source,
                    record.generated_tokens as u32,
                    offered,
                    (vector.accepted_drafts as usize).min(offered),
                    vector.emitted_count > vector.accepted_drafts,
                    survival.map(|rows| &rows[lane.request.slot as usize]),
                );
            }
            if record.status == 3 {
                request.error_code = Some(record.error_code);
            }
            let prefill = if kind.is_prefill() {
                record.consumed_tokens as u64 - request.consumed_tokens
            } else {
                0
            };
            let output = (record.generated_tokens - request.generated_tokens as i64) as u32;
            if record.status != 3 {
                request.accounting.credit(prefill, output, now_ns);
                useful_tokens[lane.request.slot as usize] = prefill + u64::from(output);
            }
            request.consumed_tokens = record.consumed_tokens as u64;
            if record.generated_tokens > request.generated_tokens as i64 {
                request.accounting.pending_publication_fence = Some(fence);
                let count = (record.generated_tokens - request.generated_tokens as i64) as usize;
                for index in 0..count {
                    let token = record.vector.map_or(record.token_id, |v| v.tokens[index]) as u32;
                    request.history.append(&[token], &[])?;
                    request.tokens.push(token);
                    request.generated_tokens += 1;
                    generated.push(GeneratedToken {
                        request: lane.request,
                        token,
                        finished: finished && index + 1 == count,
                    });
                }
            }
            request.phase = if finished {
                if request.consumed_tokens > request.prefix_tokens && request.error_code.is_none() {
                    RequestPhase::Snapshot
                } else {
                    RequestPhase::Finishing
                }
            } else if request.consumed_tokens < request.prompt_tokens as u64 {
                RequestPhase::Prefill
            } else if request.generated_tokens == 0 {
                RequestPhase::Resume
            } else {
                RequestPhase::Decode
            };
            if kind.is_prefill() && request.error_code.is_none() {
                let endpoint = request.consumed_tokens;
                if !finished
                    && request.conditioned_windows.len() > 1
                    && request.vision_patches.is_empty()
                    && request.conditioned_windows.front().is_some_and(|window| endpoint as usize == window.prefill_end)
                {
                    captures.push((
                        lane.request,
                        Some(BoundaryKind::ConditionedWindow),
                        endpoint,
                        RequestPhase::Restore,
                    ));
                    continue;
                }
                if !finished && request.consumed_tokens < request.prompt_tokens as u64 {
                    request.phase = request.prompt_phase();
                }
                let capture = if request.message_boundaries.binary_search(&(endpoint as usize)).is_ok()
                    && endpoint > request.prefix_tokens
                {
                    Some(BoundaryKind::Message)
                } else {
                    None
                }
                .or_else(|| {
                    (endpoint == request.prompt_tokens as u64
                        && endpoint > request.prefix_tokens
                        && request.message_boundaries.last() != Some(&request.prompt_tokens)
                        && request.max_new_tokens > 1)
                        .then_some(BoundaryKind::Prompt)
                });
                if let Some(kind) = capture {
                    captures.push((lane.request, Some(kind), endpoint, request.phase));
                }
            }
            if request.phase == RequestPhase::Snapshot
                && request.pending == 0
                && request.terminal_snapshot.is_none()
                && captures.last().is_none_or(|c| c.0 != lane.request)
            {
                captures.push((
                    lane.request,
                    None,
                    request.consumed_tokens,
                    RequestPhase::Finishing,
                ));
            }
            if request.pending == 0 {
                request.scheduled_consumed = request.consumed_tokens;
            }
        }
        if let Some(history) = &mut self.decoded_history {
            for event in &generated {
                let slot = event.request.slot as usize;
                if records[slot].status != 3 {
                    let request = self.slots[slot].request.as_mut().unwrap();
                    history.append(&mut request.decoded_sequence, event.token);
                }
            }
        }
        self.advance_retirement(fence)?;
        for snapshot in window_snapshots {
            self.release_snapshot(snapshot);
        }
        for handle in boundaries {
            self.publish_boundary(handle)?;
        }
        for (handle, kind, endpoint, resume) in captures {
            if matches!(resume, RequestPhase::Snapshot | RequestPhase::Finishing) {
                let request = self.request_mut(handle)?;
                if request.pending == 0 {
                    let unused = std::mem::take(&mut request.reserved_pages);
                    self.free_pages.extend(unused);
                }
            }
            let snapshot = if matches!(kind, Some(BoundaryKind::ConditionedWindow)) {
                self.request(handle)?.window_snapshot
            } else {
                self.retain_snapshot()?
            };
            let request = self.request_mut(handle)?;
            if let Some(snapshot) = snapshot {
                if let Some(kind) = kind {
                    match kind {
                        BoundaryKind::Message => request.message_snapshot = Some(snapshot),
                        BoundaryKind::Prompt => request.prompt_snapshot = Some(snapshot),
                        BoundaryKind::ConditionedWindow => {}
                    }
                    request.boundary_capture = Some(BoundaryCapture {
                        kind,
                        snapshot,
                        consumed_tokens: endpoint,
                        resume,
                    });
                } else {
                    request.terminal_snapshot = Some(snapshot);
                }
                request.phase = RequestPhase::Snapshot;
            } else if request.phase == RequestPhase::Snapshot {
                request.phase = RequestPhase::Finishing;
            }
        }
        self.observe_service(fence, &submission.participants, &useful_tokens, timing);
        self.refill_reservations()?;
        self.reconcile_refresh();
        Ok(generated)
    }

    fn publish_boundary(&mut self, handle: RequestHandle) -> Result<()> {
        let request = self.request(handle)?;
        let capture = request.boundary_capture.ok_or("missing boundary capture")?;
        if matches!(capture.kind, BoundaryKind::ConditionedWindow) {
            let request = self.request_mut(handle)?;
            request.boundary_capture = None;
            request.window_snapshot_ready = true;
            request.phase = RequestPhase::Restore;
            return Ok(());
        }
        let pages = capture.consumed_tokens as usize / self.capabilities.page_tokens
            - request.prefix_tokens as usize / self.capabilities.page_tokens;
        let proposal = Publication {
            parent: request.prefix,
            consumed_tokens: capture.consumed_tokens,
            tail_pages: request.private_pages[..pages].to_vec(),
            snapshot: StateSnapshot {
                id: capture.snapshot,
                consumed_tokens: capture.consumed_tokens,
            },
            ready_after: request.last_use,
        };
        let history = request.history.clone();
        let result = self.cache.publish(&history, &proposal)?;
        if let PublishResult::Created { id, ready } = result {
            if !ready {
                return Err("completed boundary copy was not GPU-ready".into());
            }
            self.record_publication(id);
            self.cache.pin(id)?;
            if let Some(parent) = self.request(handle)?.prefix {
                self.cache.unpin(parent)?;
            }
            let request = self.request_mut(handle)?;
            request
                .shared_pages
                .extend(request.private_pages.drain(..pages));
            request.prefix = Some(id);
            request.prefix_tokens = capture.consumed_tokens;
            request.prefix_snapshot = Some(capture.snapshot);
            self.stats.snapshot_publications += 1;
        } else {
            self.release_snapshot(capture.snapshot);
        }
        match capture.kind {
            BoundaryKind::Message => self.stats.message_checkpoints += 1,
            BoundaryKind::Prompt => self.stats.prompt_checkpoints += 1,
            BoundaryKind::ConditionedWindow => unreachable!(),
        }
        let request = self.request_mut(handle)?;
        match capture.kind {
            BoundaryKind::Message => request.message_snapshot = None,
            BoundaryKind::Prompt => request.prompt_snapshot = None,
            BoundaryKind::ConditionedWindow => unreachable!(),
        }
        request.boundary_capture = None;
        request.phase = if capture.resume == RequestPhase::Snapshot {
            RequestPhase::Finishing
        } else {
            capture.resume
        };
        Ok(())
    }

    fn advance_retirement(&mut self, fence: Fence) -> Result<()> {
        let update = self.cache.advance_completed(fence)?;
        self.apply_completion(update);
        self.completed = fence;
        let mut waiting = Vec::new();
        for retired in std::mem::take(&mut self.retired) {
            if retired.through <= fence {
                if let Some(slot) = retired.image_slot {
                    self.free_image_slots.insert(slot);
                }
                self.free_pages.extend(retired.pages);
                for snapshot in retired.snapshots.into_iter().flatten() {
                    self.release_snapshot(snapshot);
                }
                self.slots[retired.slot].retiring = false;
            } else {
                waiting.push(retired);
            }
        }
        self.retired = waiting;
        Ok(())
    }

    /// Cancels scheduling immediately; resources remain reserved through last GPU use.
    pub fn cancel(&mut self, handle: RequestHandle) -> Result<()> {
        self.retire(handle, false).map(|_| ())
    }

    /// Publishes an immutable terminal snapshot after all writes have completed.
    pub fn finish(&mut self, handle: RequestHandle) -> Result<Option<CheckpointId>> {
        let request = self.request(handle)?;
        if request.phase != RequestPhase::Finishing || request.last_use > self.completed {
            return Err("terminal checkpoint publication requires a finished request and completed GPU writes".into());
        }
        self.retire(handle, true)
    }

    /// Releases live ownership after the executor confirms both streams drained.
    /// Unconfirmed shutdown leaves every live resource and pending fence intact.
    /// Inbound tickets are dropped with their resources; the host store has
    /// drained its transfers before this point.
    pub fn release_after_shutdown(&mut self, shutdown: &Shutdown) -> Result<()> {
        if !shutdown.drained {
            return Err("cannot release request resources before confirmed CUDA drain".into());
        }
        for ticket in std::mem::take(&mut self.tickets).into_values() {
            self.fail_ticket(ticket)?;
        }
        let handles: Vec<_> = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref().map(|request| request.handle))
            .collect();
        for handle in handles {
            self.cancel(handle)?;
        }
        self.submissions.clear();
        if let Some(split) = &mut self.split {
            split.pending_decision = None;
            split.pending_budget = None;
        }
        self.refresh.abort();
        self.advance_retirement(self.submitted)
    }

    fn retire(&mut self, handle: RequestHandle, publish: bool) -> Result<Option<CheckpointId>> {
        if self.request(handle)?.phase == RequestPhase::Inbound {
            for ticket in self.tickets.values_mut() {
                ticket.waiters.retain(|waiter| *waiter != handle);
            }
        }
        let request = self.request(handle)?;
        let publication = (publish && request.terminal_snapshot_ready).then(|| Publication {
            parent: request.prefix,
            consumed_tokens: request.consumed_tokens,
            tail_pages: request.private_pages[..(request.consumed_tokens as usize
                / self.capabilities.page_tokens
                - request.prefix_tokens as usize / self.capabilities.page_tokens)]
                .to_vec(),
            snapshot: StateSnapshot {
                id: request
                    .terminal_snapshot
                    .expect("completed terminal snapshot owns a destination"),
                consumed_tokens: request.consumed_tokens,
            },
            ready_after: request.last_use,
        });
        let result = match &publication {
            Some(proposal) => {
                let history = request.history.clone();
                Some(self.cache.publish(&history, proposal)?)
            }
            None => None,
        };
        let mut request = self.slots[handle.slot as usize].request.take().unwrap();
        if let Some(split) = &mut self.split {
            split.retained[handle.slot as usize] = None;
        }
        let mut snapshot = request.terminal_snapshot;
        if matches!(result, Some(PublishResult::Created { .. })) {
            if let Some(PublishResult::Created { id, ready: true }) = result {
                self.stats.snapshot_publications += 1;
                self.record_publication(id);
            }
            request
                .private_pages
                .drain(..publication.as_ref().unwrap().tail_pages.len());
            snapshot = None;
        }
        if let Some(prefix) = request.prefix {
            self.cache.unpin(prefix)?;
        }
        self.slots[handle.slot as usize].retiring = true;
        request.private_pages.extend(request.reserved_pages);
        self.retired.push(RetiredRequest {
            image_slot: request.image_slot,
            pages: request.private_pages,
            snapshots: [
                snapshot,
                request.message_snapshot,
                request.prompt_snapshot,
                request.window_snapshot,
            ],
            slot: handle.slot as usize,
            through: request.last_use,
        });
        self.advance_retirement(self.completed)?;
        self.refill_reservations()?;
        self.reconcile_refresh();
        Ok(result.map(PublishResult::id).or(request.prefix))
    }
}

mod cache_memory;

#[cfg(test)]
mod boundary_tests;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod policy_tests;

#[cfg(test)]
mod vision_window_tests;
