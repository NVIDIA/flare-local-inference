//! Resolve prepared value bindings once and pack bounded host uploads for each submission.

use crate::{
    Result,
    execution::{Dtype, ExecutionManifest, GraphKind, REQUEST_LANES, Upload},
    model::{
        InputStep, ModelConfig,
        contract::*,
        inputs::{UploadBatch, append_pages},
    },
};

struct BoundInput {
    binding: ValueBinding,
    dtype: Dtype,
    width: usize,
    elements: usize,
    page_view: Option<usize>,
    cache_mask: bool,
}

struct BoundGraph {
    drafts: usize,
    semantics: GraphContract,
    inputs: Vec<BoundInput>,
    kind: GraphKind,
    batch: usize,
    tokens: usize,
    packed_tokens: usize,
    prefill: usize,
}

pub struct BoundContract {
    graphs: Vec<BoundGraph>,
}

pub struct Inputs {
    contract: BoundContract,
    staging: UploadBatch,
    pages: Vec<[(u64, usize); REQUEST_LANES]>,
    /// Grammar-mask byte ranges (buffer, offset, length) that already hold an unconstrained mask.
    unconstrained_masks: Vec<(String, usize, usize)>,
    integers: Vec<i64>,
    values: Vec<i64>,
}

impl Inputs {
    pub fn new(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<Self> {
        config.contract.validate(manifest)?;
        let graphs = manifest
            .graphs
            .iter()
            .map(|graph| {
                let semantics = config.contract.graph(&graph.name)?;
                if semantics
                    .draft_cursor
                    .as_ref()
                    .is_some_and(|cursor| cursor.lag_tokens > config.prepared_sequence_tokens)
                {
                    return Err("history cursor lag exceeds the prepared context".into());
                }
                let inputs = semantics
                    .inputs
                    .iter()
                    .map(|binding| {
                        let buffer = manifest.buffer(&binding.buffer)?;
                        let elements = buffer.nbytes / buffer.dtype.size();
                        #[cfg(feature = "images")]
                        if binding.source == ValueSource::ImageTensor {
                            let layout = config
                                .contract
                                .frontend
                                .image_layout
                                .ok_or("image tensor has no prepared layout")?;
                            let (dtype, minimum) = crate::frontend::images::packing::tensor_layout(
                                layout,
                                &binding.tensor,
                                graph.token_capacity,
                            )?;
                            if buffer.dtype != dtype || elements < minimum {
                                return Err(
                                    "image tensor type or extent differs from its layout".into()
                                );
                            }
                        }
                        let width = match binding.indexing {
                            InputIndexing::Scalar => elements,
                            _ => buffer.shape.get(1..).unwrap_or(&[]).iter().product(),
                        };
                        let page_view = config
                            .contract
                            .cache
                            .page_tables
                            .iter()
                            .position(|v| v.buffer == binding.buffer);
                        if binding.source == ValueSource::PageTable && page_view.is_none() {
                            return Err("page-table input lacks a declared view".into());
                        }
                        use InputIndexing::*;
                        use ValueSource::*;
                        let valid_indexing = match binding.source {
                            ImageTensor => {
                                graph.kind == GraphKind::Vision && binding.indexing == Scalar
                            }
                            LiveRows | PrefillRow => binding.indexing == Scalar,
                            ImageFeatureSlot => binding.indexing == Scalar && elements == 1
                                && matches!(graph.kind, GraphKind::Vision | GraphKind::Prefill | GraphKind::ScoredPrefill | GraphKind::Mixed),
                            PageTable | LiveStateRows => binding.indexing == RequestRow,
                            PromptToken | PromptTargetToken
                            | PromptAndPendingToken
                            | PromptPosition
                            | ImageFeatureIndex
                            | ImageLeftContext
                            | ImageRightContext => binding.indexing == PackedToken,
                            PendingToken => matches!(binding.indexing, SubmissionRow | PackedToken),
                            Constant => binding.indexing != PackedToken,
                            RequestSlot | RequestGeneration if graph.kind == GraphKind::Vision => {
                                binding.indexing == Scalar
                            }
                            _ => matches!(binding.indexing, SubmissionRow | RequestRow | ActiveRow),
                        };
                        if !valid_indexing {
                            return Err(
                                "semantic value source has incompatible row indexing".into()
                            );
                        }
                        if matches!(binding.indexing, SubmissionRow | RequestRow | ActiveRow)
                            && elements < config.limits.request_capacity() * width
                        {
                            return Err("row input cannot cover the prepared request slots".into());
                        }
                        if matches!(binding.source, SamplingParameters | ImageTensor | PageTable)
                            && binding.value_offset != 0
                        {
                            return Err("value offset requires an integer value source".into());
                        }
                        if binding.source == PageTable
                            && width
                                < config.prepared_sequence_tokens
                                    / config.contract.cache.page_tokens
                        {
                            return Err("page-table view cannot cover the prepared context".into());
                        }
                        if binding.indexing == PackedToken && graph.kind != GraphKind::Vision {
                            let rows = if graph.kind == GraphKind::Mixed {
                                graph.prefill_capacity
                                    + (graph.batch_capacity - 1) * graph.token_capacity
                            } else if graph.kind.is_prefill() {
                                graph.token_capacity
                            } else {
                                graph.packed_tokens()
                            };
                            if elements < rows * width {
                                return Err(
                                    "packed input cannot cover the graph token capacity".into()
                                );
                            }
                        }
                        let expected_width = match binding.source {
                            ValueSource::SamplingParameters => 6,
                            ValueSource::EosTokenIds | ValueSource::LiveStateRows => 8,
                            ValueSource::PromptHistogram => config.contract.vocabulary,
                            ValueSource::GrammarMask => config.contract.vocabulary.div_ceil(32),
                            ValueSource::CopyTokenIds => 7,
                            ValueSource::CopyGrammarMask => {
                                8 * config.contract.vocabulary.div_ceil(32)
                            }
                            ValueSource::PromptPosition => {
                                config.contract.frontend.position_dimensions
                            }
                            ValueSource::PageTable | ValueSource::ImageTensor => width,
                            _ => 1,
                        };
                        if width == 0
                            || width != expected_width
                            || (binding.source == ValueSource::PageTable
                                && buffer.dtype != Dtype::Int32)
                            || (matches!(
                                binding.source,
                                ValueSource::SamplingSeed
                                    | ValueSource::RequestGeneration
                                    | ValueSource::CopyGeneration
                            ) && buffer.dtype != Dtype::Int64)
                            || (binding.source == ValueSource::SamplingParameters
                                && buffer.dtype != Dtype::Float32)
                            || (binding.source != ValueSource::SamplingParameters
                                && binding.source != ValueSource::ImageTensor
                                && !matches!(buffer.dtype, Dtype::Int32 | Dtype::Int64))
                            || (binding.indexing != InputIndexing::Scalar
                                && elements < width * graph.batch_capacity)
                        {
                            return Err(format!(
                                "invalid shape or dtype for semantic input {}",
                                binding.buffer
                            )
                            .into());
                        }
                        Ok(BoundInput {
                            binding: binding.clone(),
                            dtype: buffer.dtype,
                            width,
                            elements,
                            page_view,
                            cache_mask: binding.source == GrammarMask
                                && buffer.dtype == Dtype::Int32
                                && binding.value_offset == 0
                                && matches!(binding.indexing, RequestRow | ActiveRow)
                                && !semantics.state.iter().any(|state|
                                    state.view.buffer == binding.buffer
                                        && state.access != StateAccessKind::Read),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(BoundGraph {
                    semantics: semantics.clone(),
                    inputs,
                    kind: graph.kind,
                    drafts: graph.draft_tokens,
                    batch: graph.batch_capacity,
                    tokens: graph.token_capacity,
                    packed_tokens: graph.packed_token_capacity,
                    prefill: graph.prefill_capacity,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let maximum = graphs
            .iter()
            .flat_map(|g| &g.inputs)
            .filter(|b| b.binding.source != ValueSource::ImageTensor)
            .map(|b| b.elements)
            .max()
            .unwrap_or(0);
        Ok(Self {
            contract: BoundContract { graphs },
            staging: UploadBatch::default(),
            pages: vec![[(0, 0); REQUEST_LANES]; config.contract.cache.page_tables.len()],
            unconstrained_masks: Vec::new(),
            integers: Vec::with_capacity(maximum),
            values: Vec::with_capacity(maximum),
        })
    }

    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.staging.uploads()
    }

    pub fn completion_contract(
        &self,
        step: &crate::scheduler::ScheduledStep,
    ) -> Result<&GraphContract> {
        self.contract
            .graphs
            .iter()
            .find(|g| {
                g.kind == step.kind
                    && g.drafts == step.draft_tokens
                    && g.batch == step.batch_capacity
                    && g.tokens == step.token_capacity
                    && g.packed_tokens == step.packed_token_capacity
                    && g.prefill == step.prefill_capacity()
            })
            .map(|g| &g.semantics)
            .ok_or_else(|| "completion has no prepared graph contract".into())
    }

    #[cfg(feature = "images")]
    pub fn pack_vision(
        &mut self,
        step: &crate::scheduler::ScheduledStep,
        config: &ModelConfig,
        handle: crate::scheduler::RequestHandle,
        image_slot: usize,
        images: &crate::frontend::images::ProcessedImages,
        image: usize,
        destination_feature_offset: usize,
    ) -> Result<()> {
        let graph = self
            .contract
            .graphs
            .iter()
            .find(|g| g.kind == GraphKind::Vision && g.tokens == step.token_capacity)
            .ok_or("vision submission lacks a prepared packing plan")?;
        let tensors = crate::frontend::images::packing::prepare(
            config,
            handle,
            images,
            image,
            destination_feature_offset,
        )?;
        let tensors = tensors.uploads();
        self.staging.clear();
        // Only retain cached bytes while this graph treats the same registered
        // buffer as a read-only mask input. Other bindings, graph state writes,
        // and graphs without that input may replace its contents.
        self.unconstrained_masks.retain(|(name, _, _)|
            graph.inputs.iter().any(|input| input.cache_mask && input.binding.buffer == *name));
        for input in &graph.inputs {
            match input.binding.source {
                ValueSource::RequestSlot => {
                    write_integers(&mut self.staging, input, 0, &[handle.slot as i64])?
                }
                ValueSource::RequestGeneration => {
                    write_integers(&mut self.staging, input, 0, &[handle.generation as i64])?
                }
                ValueSource::ImageFeatureSlot => {
                    if image_slot >= config.contract.vision.feature_slots {
                        return Err("image feature slot exceeds the prepared arena".into());
                    }
                    write_integers(&mut self.staging, input, 0, &[image_slot as i64])?
                }
                ValueSource::ImageTensor => {
                    let tensor = tensors
                        .iter()
                        .find(|t| t.buffer == input.binding.tensor)
                        .ok_or("image layout did not produce a required tensor")?;
                    if tensor.offset + tensor.bytes.len() > input.elements * input.dtype.size() {
                        return Err("image tensor exceeds prepared graph capacity".into());
                    }
                    self.staging.put(
                        &input.binding.buffer,
                        tensor.offset,
                        tensor.bytes.iter().copied(),
                    );
                }
                _ => return Err("invalid image graph value source".into()),
            }
        }
        Ok(())
    }

    pub fn pack(&mut self, frame: &InputStep<'_>) -> Result<()> {
        let result = self.pack_step(frame);
        if result.is_err() {
            // A failed pack has not uploaded any of the staged mask bytes.
            self.unconstrained_masks.clear();
        }
        result
    }

    fn pack_step(&mut self, frame: &InputStep<'_>) -> Result<()> {
        let graph = self
            .contract
            .graphs
            .iter()
            .find(|g| {
                g.kind == frame.step.kind
                    && g.drafts == frame.step.draft_tokens
                    && g.batch == frame.step.batch_capacity
                    && g.tokens == frame.step.token_capacity
                    && g.packed_tokens == frame.step.packed_token_capacity
                    && g.prefill == frame.step.prefill_capacity()
            })
            .ok_or("submission has no resolved packing plan")?;
        self.staging.clear();
        // Only retain cached bytes while this graph treats the same registered
        // buffer as a read-only mask input. Other bindings, graph state writes,
        // and graphs without that input may replace its contents.
        self.unconstrained_masks.retain(|(name, _, _)|
            graph.inputs.iter().any(|input| input.cache_mask && input.binding.buffer == *name));
        let mut history_proposals = [None; REQUEST_LANES];
        if frame.config.history_drafts
            && frame.config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Mtp
        {
            for (row, lane) in frame.step.lanes.iter().enumerate() {
                let live = frame.scheduler.request(lane.request)?;
                if frame.step.lane_kind(row) == GraphKind::Speculative && live.copy_allowed() {
                    history_proposals[row] = Some(frame.scheduler.history_proposal(
                        live.tokens(),
                        frame.step.token_capacity - 1,
                    ));
                }
            }
        }
        for input in &graph.inputs {
            let binding = &input.binding;
            if binding.source == ValueSource::PageTable {
                let view = input.page_view.unwrap();
                let offset = frame.config.contract.cache.page_tables[view].page_offset;
                for lane in &frame.step.lanes {
                    let live = frame.scheduler.request(lane.request)?;
                    let count = live.pages().count().saturating_sub(offset).min(input.width);
                    append_pages(
                        &mut self.staging,
                        &mut self.pages[view],
                        &binding.buffer,
                        live,
                        input.width,
                        offset..offset + count,
                    )?;
                }
                continue;
            }
            if binding.source == ValueSource::ImageTensor {
                return Err("vision tensors require an image submission".into());
            }
            if binding.indexing == InputIndexing::Scalar {
                self.integers.clear();
                self.integers.resize(
                    input.width,
                    match binding.source {
                        ValueSource::Constant => 0,
                        ValueSource::LiveRows => frame.step.lanes.len() as i64,
                        ValueSource::ImageFeatureSlot => {
                            match frame.step.lanes.iter().enumerate().find(|(row, _)|
                                frame.step.lane_kind(*row).is_prefill()) {
                                Some((_, lane)) => frame.scheduler.request(lane.request)?.image_slot.map_or(-1, |slot| slot as i64),
                                None => -1,
                            }
                        }
                        ValueSource::PrefillRow => frame
                            .step
                            .mixed
                            .as_ref()
                            .and_then(|m| m.prefill_row)
                            .map_or(-1, |r| r as i64),
                        _ => return Err("input source does not have scalar indexing".into()),
                    },
                );
                write_integers(&mut self.staging, input, 0, &self.integers)?;
                continue;
            }
            if binding.indexing == InputIndexing::SubmissionRow {
                self.integers.clear();
                self.integers.resize(
                    frame.config.limits.request_capacity() * input.width,
                    binding.padding,
                );
            }
            let mut token_offset = 0;
            for (row, lane) in frame.step.lanes.iter().enumerate() {
                let slot = lane.request.slot as usize;
                let admission = frame.admissions[slot]
                    .as_ref()
                    .ok_or("missing packing admission")?;
                let live = frame.scheduler.request(lane.request)?;
                let lane_kind = frame.step.lane_kind(row);
                let destination = match binding.indexing {
                    InputIndexing::SubmissionRow | InputIndexing::ActiveRow => row * input.width,
                    InputIndexing::RequestRow => slot * input.width,
                    InputIndexing::PackedToken => token_offset * input.width,
                    InputIndexing::Scalar => unreachable!(),
                };
                token_offset += lane.tokens;
                if binding.source == ValueSource::SamplingParameters {
                    self.staging.put(
                        &binding.buffer,
                        destination * 4,
                        admission
                            .sampling
                            .values()
                            .into_iter()
                            .flat_map(f32::to_le_bytes),
                    );
                    continue;
                }
                if binding.source == ValueSource::GrammarMask
                    && input.dtype == Dtype::Int32
                    && binding.value_offset == 0
                    && matches!(
                        binding.indexing,
                        InputIndexing::RequestRow | InputIndexing::ActiveRow
                    )
                {
                    // Grammar masks already contain packed i32 bit patterns.
                    // Preserve those bytes without widening and validating every word.
                    let count = admission
                        .grammar_mask
                        .as_ref()
                        .map_or(input.width, Vec::len);
                    if count == 0 {
                        continue;
                    }
                    if destination
                        .checked_add(count)
                        .is_none_or(|end| end > input.elements)
                    {
                        return Err("packed values exceed prepared input storage".into());
                    }
                    if let Some(mask) = &admission.grammar_mask {
                        // A constrained mask may overlap any unconstrained range of this buffer.
                        self.unconstrained_masks.retain(|(name, _, _)| *name != binding.buffer);
                        self.staging.i32(
                            &binding.buffer,
                            destination * 4,
                            mask.iter().copied(),
                        );
                    } else if !input.cache_mask || !self.unconstrained_masks.iter().any(|(name, offset, bytes)| {
                        *name == binding.buffer && *offset == destination * 4 && *bytes == count * 4
                    }) {
                        // Cache only rows whose graph contract preserves the uploaded mask.
                        self.staging.put(
                            &binding.buffer,
                            destination * 4,
                            std::iter::repeat_n(0xff, count * 4),
                        );
                        if input.cache_mask {
                            self.unconstrained_masks.push((binding.buffer.clone(), destination * 4, count * 4));
                        }
                    }
                    continue;
                }
                self.values.clear();
                let values = &mut self.values;
                use ValueSource::*;
                match binding.source {
                    Constant => values.resize(input.width, 0),
                    RequestSlot => values.push(slot as i64),
                    RequestGeneration => values.push(lane.request.generation as i64),
                    QueryLength | SpeculativePrefixLimit => values.push(lane.tokens as i64),
                    SampleEnabled => values.push(frame.samples[row] as i64),
                    PromptTargetToken => {
                        for position in lane.start_position as usize + 1..=lane.start_position as usize + lane.tokens {
                            let token = admission.prompt.get(position).copied();
                            let image = admission.data.is_image_position(position);
                            values.push(token.filter(|t| !image && (*t as usize) < frame.config.contract.vocabulary)
                                .map_or(-1, |t| t as i64));
                        }
                    }
                    PromptToken | PendingToken | PromptAndPendingToken => {
                        if lane_kind.is_prefill() && binding.source != PendingToken {
                            let start = usize::try_from(lane.start_position)
                                .map_err(|_| "prompt offset overflow")?;
                            let end = start
                                .checked_add(lane.tokens)
                                .ok_or("prompt extent overflow")?;
                            values.extend(
                                admission
                                    .prompt
                                    .get(start..end)
                                    .ok_or("prompt slice exceeds admission")?
                                    .iter()
                                    .map(|t| *t as i64),
                            );
                        } else if !lane_kind.is_prefill() && binding.source != PromptToken {
                            values.push(
                                *live
                                    .tokens()
                                    .get(live.consumed_tokens as usize)
                                    .ok_or("decode has no completed pending token")?
                                    as i64,
                            );
                        }
                    }
                    PromptPosition | ImageFeatureIndex | ImageLeftContext | ImageRightContext => {
                        if !lane_kind.is_prefill() {
                            continue;
                        }
                        let start = usize::try_from(lane.start_position)
                            .map_err(|_| "prompt offset overflow")?;
                        let end = start
                            .checked_add(lane.tokens)
                            .ok_or("prompt extent overflow")?;
                        match binding.source {
                            PromptPosition => values.extend(
                                admission
                                    .data
                                    .positions
                                    .get(start..end)
                                    .ok_or("prompt positions exceed admission")?
                                    .iter()
                                    .flatten(),
                            ),
                            ImageFeatureIndex => values.extend(
                                admission
                                    .data
                                    .vision_token_indices
                                    .get(start..end)
                                    .ok_or("image indices exceed admission")?
                                    .iter()
                                    .map(|v| {
                                        let index = frame.scheduler.request(lane.request)?.feature_index(*v)?;
                                        if index >= frame.config.contract.vision.features_per_request as i64 {
                                            return Err("image feature index exceeds prepared execution arena".into());
                                        }
                                        Ok(index)
                                    })
                                    .collect::<Result<Vec<_>>>()?,
                            ),
                            _ => {
                                values.resize(lane.tokens, 0);
                                for span in admission
                                    .data
                                    .spans
                                    .iter()
                                    .filter(|s| s.start < end && s.end > start)
                                {
                                    if span.start < start || span.end > end {
                                        return Err("prefill splits an atomic image span".into());
                                    }
                                    for position in span.clone() {
                                        values[position - start] =
                                            if binding.source == ImageLeftContext {
                                                position - span.start
                                            } else {
                                                span.end - position - 1
                                            } as i64;
                                    }
                                }
                            }
                        }
                    }
                    LiveStateRows => values.extend((0..input.width).map(|column| {
                        if column < frame.scheduler.execution_rows() {
                            live.state_slot.0 as i64 + column as i64
                        } else {
                            0
                        }
                    })),
                    LiveStateRow => values.push(live.state_slot.0 as i64),
                    PrefixStateRow => values.push(
                        frame
                            .scheduler
                            .snapshot_slot(
                                live.prefix_snapshot()
                                    .ok_or("restore has no prefix state")?,
                            )
                            .0 as i64,
                    ),
                    SnapshotStateRow => values
                        .push(frame.scheduler.snapshot_slot(live.snapshot_destination()).0 as i64),
                    ConsumedTokens => values.push(live.consumed_tokens as i64),
                    SnapshotConsumedTokens => values.push(live.snapshot_consumed() as i64),
                    SamplingSeed => values.push(admission.sampling.seed as i64),
                    SamplingTopK => values.push(admission.sampling.top_k as i64),
                    MaxNewTokens => values.push(admission.max_new_tokens as i64),
                    EosTokenIds => values.extend(
                        frame
                            .config
                            .eos_token_ids
                            .iter()
                            .map(|t| *t as i64)
                            .chain(std::iter::repeat(-1))
                            .take(input.width),
                    ),
                    EosCount => values.push(if admission.ignore_eos {
                        0
                    } else {
                        frame.config.eos_token_ids.len() as i64
                    }),
                    PromptHistogram => {
                        values.resize(input.width, 0);
                        for token in &admission.prompt {
                            if let Some(count) = values.get_mut(*token as usize) {
                                *count += 1;
                            }
                        }
                    }
                    GrammarMask => {
                        if let Some(mask) = &admission.grammar_mask {
                            values.extend(mask.iter().map(|v| *v as i64));
                        } else {
                            values.resize(input.width, -1);
                        }
                    }
                    GrammarEnabled => values.push(i64::from(admission.grammar_mask.is_some())),
                    CopyTokenIds | CopyTokenCount | CopyConsumedTokens | CopyGeneration
                    | CopyGrammarMask | CopyGrammarCount => {
                        let copy = if lane_kind != GraphKind::Speculative {
                            None
                        } else if frame.config.speculative_proposal_source
                            == crate::scheduler::speculation::ProposalSource::Copy
                        {
                            Some(
                                live.copy_proposal()
                                    .ok_or("copy verification lacks a proposal")?,
                            )
                        } else {
                            None
                        };
                        if copy.is_some_and(|proposal| proposal.count + 1 < lane.tokens) {
                            return Err(
                                "copy proposal is shorter than the verification input".into()
                            );
                        }
                        let history = history_proposals[row];
                        let count = copy
                            .map(|p| p.count)
                            .or_else(|| history.as_ref().map(|(_, n)| *n))
                            .unwrap_or(0);
                        match binding.source {
                            CopyTokenIds => values.extend(
                                copy.map(|p| p.tokens)
                                    .or_else(|| history.map(|(t, _)| t))
                                    .unwrap_or([-1; 7]),
                            ),
                            CopyTokenCount => values.push(count as i64),
                            CopyConsumedTokens => {
                                values.push(if copy.is_some() || history.is_some() {
                                    live.consumed_tokens as i64
                                } else {
                                    -1
                                })
                            }
                            CopyGeneration => values.push(if copy.is_some() || history.is_some() {
                                lane.request.generation as i64
                            } else {
                                0
                            }),
                            CopyGrammarMask => {
                                if let Some(copy) = copy {
                                    values.extend(copy.grammar_masks.iter().map(|v| *v as i64));
                                }
                            }
                            CopyGrammarCount => values.push(copy.map_or(0, |p| {
                                p.grammar_masks.len()
                                    / frame.config.contract.vocabulary.div_ceil(32)
                            }) as i64),
                            _ => unreachable!(),
                        }
                    }
                    _ => return Err("input source has incompatible indexing".into()),
                }
                if values.is_empty() {
                    continue;
                }
                if destination
                    .checked_add(values.len())
                    .is_none_or(|end| end > input.elements)
                {
                    return Err("packed values exceed prepared input storage".into());
                }
                if binding.indexing == InputIndexing::SubmissionRow {
                    if values.len() != input.width {
                        return Err("submission row has an invalid value count".into());
                    }
                    for (to, value) in self.integers[destination..destination + input.width]
                        .iter_mut()
                        .zip(values.iter())
                    {
                        *to = value
                            .checked_add(binding.value_offset)
                            .ok_or("input value overflow")?;
                    }
                } else {
                    write_integers(&mut self.staging, input, destination, &values)?;
                }
            }
            if binding.indexing == InputIndexing::SubmissionRow
                && binding.source != ValueSource::SamplingParameters
            {
                write_raw(&mut self.staging, input, 0, self.integers.iter().copied())?;
            }
        }
        Ok(())
    }
}

fn write_integers(
    staging: &mut UploadBatch,
    input: &BoundInput,
    offset: usize,
    values: &[i64],
) -> Result<()> {
    for value in values {
        let value = value
            .checked_add(input.binding.value_offset)
            .ok_or("input value overflow")?;
        if input.dtype == Dtype::Int32 && i32::try_from(value).is_err() {
            return Err("input value exceeds int32".into());
        }
    }
    write_raw(
        staging,
        input,
        offset,
        values.iter().map(|v| v + input.binding.value_offset),
    )
}

fn write_raw(
    staging: &mut UploadBatch,
    input: &BoundInput,
    offset: usize,
    values: impl Iterator<Item = i64> + Clone,
) -> Result<()> {
    match input.dtype {
        Dtype::Int64 => staging.i64(&input.binding.buffer, offset * 8, values),
        Dtype::Int32 => {
            if values.clone().any(|value| i32::try_from(value).is_err()) {
                return Err("input value exceeds int32".into());
            }
            staging.i32(&input.binding.buffer, offset * 4, values.map(|v| v as i32));
        }
        _ => return Err("integer source requires an integer input buffer".into()),
    }
    Ok(())
}
