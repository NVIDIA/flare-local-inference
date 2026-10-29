use std::{marker::PhantomData, rc::Rc, time::Instant};

use crate::{
    Error, ExecutionError, Failure, Result, Shutdown,
    cache::Fence,
    cuda::{BATCH_BYTES, BATCH_ENTRIES, Event, PinnedHost, Stream},
};

use super::{ExecutionManifest, GraphKind, MAX_OUTPUT_TOKENS, REQUEST_LANES, VECTOR_RECORD_FIELDS, ReadinessIntervals};

pub struct Upload<'a> {
    pub buffer: &'a str,
    pub offset: usize,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenVector {
    pub emitted_count: i64,
    pub offered_drafts: i64,
    pub accepted_drafts: i64,
    pub copy_offered_mask: i64,
    pub model_reserved: i64,
    pub draft_cache_consumed: i64,
    pub verification_width: i64,
    pub tokens: [i64; MAX_OUTPUT_TOKENS],
}
impl Default for TokenVector {
    fn default() -> Self {
        Self {
            emitted_count: 0,
            offered_drafts: 0,
            accepted_drafts: 0,
            copy_offered_mask: 0,
            model_reserved: -1,
            draft_cache_consumed: 0,
            verification_width: 1,
            tokens: [-1; MAX_OUTPUT_TOKENS],
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompletionRecord {
    pub generation: i64,
    pub submission_sequence: i64,
    pub consumed_tokens: i64,
    pub generated_tokens: i64,
    pub token_id: i64,
    pub status: i64,
    pub error_code: i64,
    pub reserved: i64,
    pub vector: Option<TokenVector>,
}

impl CompletionRecord {
    pub(crate) fn validate_vector(&self) -> Result<()> {
        if self.status != 0 {
            if let Some(v) = self.vector {
                if !(1..=8).contains(&v.verification_width)
                    || !(0..=v.verification_width).contains(&v.emitted_count)
                    || !(0..v.verification_width).contains(&v.offered_drafts)
                    || v.accepted_drafts < v.emitted_count.saturating_sub(1).max(0)
                    || v.accepted_drafts > v.emitted_count
                    || v.accepted_drafts > v.offered_drafts
                    || v.copy_offered_mask < 0
                    || v.copy_offered_mask >= (1 << v.offered_drafts)
                    || v.draft_cache_consumed < -1
                    || v.tokens[..v.emitted_count as usize]
                        .iter()
                        .any(|id| u32::try_from(*id).is_err())
                    || v.tokens[v.emitted_count as usize..]
                        .iter()
                        .any(|id| *id != -1)
                    || self.token_id
                        != if v.emitted_count == 0 {
                            -1
                        } else {
                            v.tokens[v.emitted_count as usize - 1]
                        }
                    || (self.status == 3 && v.emitted_count != 0)
                {
                    return Err(Error::execution(
                        ExecutionError::Completion,
                        "invalid bounded token-vector completion",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn fields(&self) -> Vec<i64> {
        let mut values = vec![
            self.generation,
            self.submission_sequence,
            self.consumed_tokens,
            self.generated_tokens,
            self.token_id,
            self.status,
            self.error_code,
            self.reserved,
        ];
        let v = self.vector.expect("completion serialization requires a token vector");
        values.extend([
            v.emitted_count,
            v.offered_drafts,
            v.accepted_drafts,
            v.copy_offered_mask,
            v.model_reserved,
            v.draft_cache_consumed,
            0,
            v.verification_width,
        ]);
        values.extend(v.tokens);
        values
    }
}
/// Inline completion storage bounded by the largest supported request capacity.
#[derive(Debug, Clone, Copy)]
pub struct CompletionRecords {
    records: [CompletionRecord; REQUEST_LANES],
    len: usize,
}

impl std::ops::Deref for CompletionRecords {
    type Target = [CompletionRecord];
    fn deref(&self) -> &Self::Target {
        &self.records[..self.len]
    }
}
impl std::ops::DerefMut for CompletionRecords {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.records[..self.len]
    }
}
impl<const N: usize> From<[CompletionRecord; N]> for CompletionRecords {
    fn from(values: [CompletionRecord; N]) -> Self {
        assert!(N <= REQUEST_LANES);
        let mut records = [CompletionRecord::default(); REQUEST_LANES];
        records[..N].copy_from_slice(&values);
        Self { records, len: N }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprobs {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromptLogprobs {
    pub value: TokenLogprobs,
    pub rank: u32,
}

#[derive(Debug)]
pub struct FrameCompletion {
    pub fence: Fence,
    pub graph_index: usize,
    pub timing: ExecutionTiming,
    pub records: CompletionRecords,
    pub logprobs: Option<Vec<[Option<TokenLogprobs>; MAX_OUTPUT_TOKENS]>>,
    pub prompt_logprobs: Option<Vec<Option<PromptLogprobs>>>,
    /// Confidence by proposal batch row; rows and columns outside the graph shape are zero.
    pub proposal_confidence: Option<[[f32; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES]>,
    /// Verification prefix survival by physical slot; -1 marks an unobserved prefix.
    pub speculative_survival: Option<[[f32; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES]>,
}

pub(crate) fn validate_speculative_survival(rows: &[[f32; 7]], records: &[CompletionRecord]) -> Result<()> {
    if rows.len() < records.len() || rows.len() > REQUEST_LANES {
        return Err("speculative survival does not cover completion slots".into());
    }
    for (slot, row) in rows.iter().enumerate() {
        let offered = records.get(slot).filter(|record| matches!(record.status, 1 | 2))
            .and_then(|record| record.vector).map_or(0, |vector| {
                if vector.emitted_count > vector.accepted_drafts { vector.offered_drafts }
                else { vector.accepted_drafts.min(vector.offered_drafts) }
            }).clamp(0, 7) as usize;
        let mut bound = 1.0;
        let mut missing = false;
        for (depth, &value) in row.iter().enumerate() {
            missing |= value == -1.0;
            if value != -1.0 && (missing || depth >= offered || !value.is_finite() || !(0.0..=bound).contains(&value)) {
                return Err(format!("invalid speculative survival at slot {slot}, depth {depth}: value {value}, preceding bound {bound}, observed extent {offered}, row {row:?}").into());
            }
            bound = value;
        }
    }
    Ok(())
}

fn reads_speculative_survival(graph: &super::GraphDescriptor) -> bool {
    graph.token_capacity > 1 && matches!(graph.kind, GraphKind::Speculative | GraphKind::Mixed)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionTiming {
    /// Local CUDA event intervals for up to sixteen ranks; absent ranks have no interval.
    pub gpu_ns: [Option<u64>; 16],
    /// Per-table minima of the readiness-through-barrier and barrier-only intervals.
    pub readiness: Option<ReadinessIntervals>,
    /// Host interval from dispatch through observation of every rank's completion.
    /// Includes input staging, coordination, device queueing and completion delivery.
    pub wall_ns: u64,
    /// Another frame was outstanding when this frame was dispatched.
    pub queued: bool,
}

impl ExecutionTiming {
    pub(crate) fn merge_rank(&mut self, rank: usize, peer: Self) -> Result<()> {
        self.gpu_ns[rank] = peer.gpu_ns[0];
        match (&mut self.readiness, peer.readiness) {
            (Some(local), Some(remote)) => local.merge(remote),
            (None, None) => Ok(()),
            _ => Err(Error::execution(ExecutionError::Completion, "rank readiness timestamp bindings differ")),
        }
    }
}

struct Pending {
    logprobs: u64,
    logprob_width: usize,
    logprob_token_ids: Vec<Vec<u32>>,
    prompt_rows: usize,
    fence: Fence,
    graph_index: usize,
    grammar: bool,
    dispatched: Instant,
    queued: bool,
}

struct Frame {
    input: PinnedHost,
    output: PinnedHost,
    confidence: Option<Readback>,
    survival: Option<Readback>,
    readiness: Option<Readback>,
    logprobs: Option<Readback>,
    prompt_logprobs: Option<Readback>,
    produced: Event,
    started: Event,
    transferred: Event,
    pending: Option<Pending>,
}

enum Readback {
    MappedHost { address: usize, nbytes: usize },
    DeviceCopy(PinnedHost),
}

impl Readback {
    fn new(stream: &Stream, address: usize, nbytes: usize) -> Result<Self> {
        Ok(match stream.mapped_host_address(address, nbytes)? {
            Some(address) => Self::MappedHost { address, nbytes },
            None => Self::DeviceCopy(stream.host_buffer(nbytes)?),
        })
    }

    unsafe fn enqueue_copy(&self, stream: &Stream, source: usize, nbytes: usize) -> Result<()> {
        match self {
            Self::MappedHost { .. } => Ok(()),
            // SAFETY: The caller retains the device source and this pinned
            // destination until the transfer stream completes.
            Self::DeviceCopy(host) => unsafe { stream.download(host, source, nbytes) },
        }
    }

    unsafe fn enqueue_pitched_copy(
        &self,
        stream: &Stream,
        source: usize,
        source_pitch: usize,
        width: usize,
        rows: usize,
    ) -> Result<()> {
        match self {
            Self::MappedHost { .. } => Ok(()),
            // SAFETY: The caller retains both allocations through the transfer event.
            Self::DeviceCopy(host) => unsafe {
                stream.download_2d(host, source, source_pitch, width, rows)
            },
        }
    }

    /// The caller has observed the frame's transfer event and retains the source
    /// allocation; proposal feedback prevents another graph from overwriting it.
    unsafe fn bytes(&self) -> &[u8] {
        match self {
            Self::MappedHost { address, nbytes } => {
                // SAFETY: The mapped allocation and producing graph obey the
                // executor's ownership and completion requirements above.
                unsafe { std::slice::from_raw_parts(*address as *const u8, *nbytes) }
            }
            Self::DeviceCopy(host) => host.bytes(),
        }
    }
}

struct InputCopyBatches {
    destinations: Vec<u64>,
    sources: Vec<u64>,
    sizes: Vec<usize>,
    bytes: usize,
}

impl InputCopyBatches {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.min(BATCH_ENTRIES);
        Self {
            destinations: Vec::with_capacity(capacity),
            sources: Vec::with_capacity(capacity),
            sizes: Vec::with_capacity(capacity),
            bytes: 0,
        }
    }

    fn clear(&mut self) {
        self.destinations.clear();
        self.sources.clear();
        self.sizes.clear();
        self.bytes = 0;
    }

    fn submit(
        &mut self,
        host: usize,
        copies: impl Iterator<Item = ResolvedUpload>,
        mut emit: impl FnMut(&[u64], &[u64], &[usize]) -> Result<()>,
    ) -> Result<()> {
        self.clear();
        for copy in copies {
            let mut offset = 0;
            while offset < copy.nbytes {
                if self.bytes == BATCH_BYTES || self.sizes.len() == BATCH_ENTRIES {
                    emit(&self.destinations, &self.sources, &self.sizes)?;
                    self.clear();
                }
                let nbytes = (copy.nbytes - offset).min(BATCH_BYTES - self.bytes);
                self.destinations.push((copy.device_address + offset) as u64);
                self.sources.push((host + copy.host_offset + offset) as u64);
                self.sizes.push(nbytes);
                self.bytes += nbytes;
                offset += nbytes;
            }
        }
        if !self.sizes.is_empty() {
            emit(&self.destinations, &self.sources, &self.sizes)?;
        }
        self.clear();
        Ok(())
    }
}

pub(super) fn collective_choices_overview(
    manifest: &ExecutionManifest,
    choices: &[Vec<(String, Vec<u8>)>],
) -> String {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Write;

    // Only this decoder assumes the b12x packed mode/CTA encoding.
    fn decode(bytes: &[u8]) -> (String, Option<u32>) {
        let Ok(word) = <[u8; 4]>::try_from(bytes) else {
            return (format!("raw={bytes:02x?}"), None);
        };
        let value = u32::from_ne_bytes(word);
        let name = match value >> 16 {
            0 => "Full".into(), 1 => "Payload RSAG".into(), 2 => "Bulk Push".into(),
            3 => "Bulk Pull".into(), 4 => "Pull".into(), id => format!("Mode {id}"),
        };
        (name, Some(value & 0xffff))
    }
    fn describe<'a>(values: impl Iterator<Item = &'a [u8]>) -> String {
        let mut modes: BTreeMap<String, BTreeSet<u32>> = BTreeMap::new();
        for bytes in values {
            let (name, ctas) = decode(bytes);
            modes.entry(name).or_default().extend(ctas);
        }
        modes.into_iter().map(|(name, ctas)| {
            if ctas.is_empty() { return name; }
            let counts = ctas.into_iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",");
            format!("{name} ({counts} CTAs)")
        }).collect::<Vec<_>>().join(", ")
    }
    let mut totals = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    for group in choices {
        for (_, bytes) in group {
            *totals.entry(decode(bytes).0).or_insert(0usize) += 1;
        }
        outputs.extend(group.iter().map(|(name, bytes)| (name.as_str(), bytes.as_slice())));
    }
    let mut text = format!("  Collective choices · device {} · diagnostic\n", manifest.device);
    let counts = totals.iter().map(|(name, count)| {
        format!("{name}: {count} binding{}", if *count == 1 { "" } else { "s" })
    }).collect::<Vec<_>>().join(" · ");
    writeln!(text, "  {counts}").unwrap();
    let mut graphs = manifest.graphs.iter().filter(|graph| !graph.tuned_bindings.is_empty()).collect::<Vec<_>>();
    graphs.sort_by_key(|graph| (graph.kind as u8, graph.batch_capacity, graph.token_capacity,
        graph.packed_tokens(), graph.prefill_capacity, graph.draft_tokens, graph.name.as_str()));
    for graph in graphs {
        let values = graph.tuned_bindings.iter().filter_map(|binding| outputs.get(binding.buffer.as_str()).copied());
        let selected = describe(values);
        writeln!(text, "  {} · {:?} b={} t={} rows={} drafts={} prefill={} · {}",
            graph.name, graph.kind, graph.batch_capacity, graph.token_capacity,
            graph.packed_tokens(), graph.draft_tokens, graph.prefill_capacity,
            if selected.is_empty() { "unavailable" } else { &selected }).unwrap();
    }
    text
}

struct Resources {
    model: Stream,
    copy: Stream,
    frames: [Frame; 2],
    engram: Option<crate::engram::Transfers>,
    input_copies: InputCopyBatches,
}

/// Thread-bound executor over one borrowed prepared model and its CUDA context.
pub struct GraphExecutor {
    manifest: ExecutionManifest,
    pub(super) copy_logprobs: bool,
    resources: Option<Resources>,
    resolved_uploads: Vec<ResolvedUpload>,
    startup_next: usize,
    startup_ready: bool,
    diagnostic_choices: Vec<Vec<(String, Vec<u8>)>>,
    next_sequence: u64,
    completed: Fence,
    poison: Option<Error>,
    shutdown: Option<Shutdown>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GraphExecutor {
    /// # Safety
    /// All manifest handles and allocations must belong to the given context,
    /// remain stable, and obey helios/manifest.md. The caller retains the owner
    /// until close confirms both streams drained. No other executor may mutate
    /// these inputs, graph executables, scratch or completion rows concurrently.
    pub unsafe fn new(manifest: ExecutionManifest) -> Result<Self> {
        let mut executor = unsafe { Self::new_unstarted(manifest)? };
        let started = Instant::now();
        for index in executor.manifest.referenced_tuning_graphs() {
            executor.run_tuning_graph(index)?;
        }
        executor.finish_startup()?;
        executor.report_startup(started);
        Ok(executor)
    }

    /// Construct rank-local streams without launching any collective graphs.
    pub(super) unsafe fn new_unstarted(manifest: ExecutionManifest) -> Result<Self> {
        manifest.validate()?;
        let upload_capacity = manifest
            .graphs
            .iter()
            .map(|graph| graph.input_bindings.len())
            .max()
            .unwrap_or(0)
            .saturating_mul(manifest.request_capacity());
        let model = Stream::open(manifest.context, manifest.device)?;
        let copy = Stream::open(manifest.context, manifest.device)?;
        let mut frame_index = 0;
        let mut frame = || -> Result<Frame> {
            let index = frame_index;
            frame_index += 1;
            Ok(Frame {
                input: model.host_buffer(manifest.input_frame_bytes)?,
                output: model.host_buffer(manifest.completion_bytes())?,
                logprobs: if manifest.max_top_logprobs()? != 0 {
                    let b = manifest.buffer("completion_logprobs")?;
                    Some(Readback::new(
                        &model, b.address + index * b.nbytes / 2, b.nbytes / 2,
                    )?)
                } else {
                    None
                },
                prompt_logprobs: if manifest.prompt_logprob_capacity()? != 0 {
                    let b = manifest.buffer("prompt_logprobs")?;
                    Some(Readback::new(
                        &model, b.address + index * b.nbytes / 2, b.nbytes / 2,
                    )?)
                } else {
                    None
                },
                confidence: manifest
                    .graphs
                    .iter()
                    .any(|graph| graph.confidence_width > 0)
                    .then(|| {
                        let buffer = manifest.buffer("proposal_confidence")?;
                        Readback::new(&model, buffer.address + index * buffer.nbytes / 2, buffer.nbytes / 2)
                    })
                    .transpose()?,
                survival: manifest.buffers.iter().find(|buffer| buffer.name == "completion_speculative_survival")
                    .map(|buffer| Readback::new(&model, buffer.address + index * buffer.nbytes / 2, buffer.nbytes / 2))
                    .transpose()?,
                readiness: manifest.buffers.iter().find(|buffer| buffer.name == "readiness_timestamps")
                    .map(|buffer| Readback::new(&model, buffer.address + index * buffer.nbytes / 2, buffer.nbytes / 2))
                    .transpose()?,
                produced: model.event_with_timing(true)?,
                started: model.event_with_timing(true)?,
                transferred: copy.event()?,
                pending: None,
            })
        };
        let frames = [frame()?, frame()?];
        let engram = manifest
            .engram
            .clone()
            .map(|binding| crate::engram::Transfers::new(binding, &manifest))
            .transpose()?;
        Ok(Self {
            manifest,
            copy_logprobs: true,
            resources: Some(Resources {
                model,
                copy,
                frames,
                engram,
                input_copies: InputCopyBatches::new(upload_capacity.saturating_add(2)),
            }),
            resolved_uploads: Vec::with_capacity(upload_capacity),
            startup_next: 0,
            startup_ready: false,
            diagnostic_choices: Vec::new(),
            next_sequence: 1,
            completed: Fence(0),
            poison: None,
            shutdown: None,
            _thread_bound: PhantomData,
        })
    }

    pub(super) fn run_tuning_graph(&mut self, index: usize) -> Result<()> {
        let order = self.manifest.referenced_tuning_graphs();
        if self.startup_ready || order.get(self.startup_next) != Some(&index) {
            return Err("startup tuning graph order differs from the manifest".into());
        }
        let graph = &self.manifest.tuning_graphs[index];
        let status = self.manifest.buffer(&graph.status_buffer)?;
        let resources = self.resources.as_ref().ok_or("executor is closed")?;
        let mut reset = resources.model.host_buffer(4)?;
        reset.bytes_mut().copy_from_slice(&(-1i32).to_ne_bytes());
        let readback = resources.model.host_buffer(4)?;
        let outputs = graph.outputs.iter().map(|name| {
            let buffer = self.manifest.buffer(name)?;
            Ok((buffer, resources.model.host_buffer(buffer.nbytes.min(64))?))
        }).collect::<Result<Vec<_>>>()?;
        let completed = resources.model.event()?;
        let started = Instant::now();
        let result = (|| -> Result<()> {
            // Pinned buffers remain alive until the event confirms all transfers.
            unsafe { resources.model.upload(status.address, &reset, 0, 4)?; }
            resources.model.launch(graph.handle)?;
            unsafe { resources.model.download(&readback, status.address, 4)?; }
            for (buffer, host) in &outputs {
                unsafe { resources.model.download(host, buffer.address, buffer.nbytes.min(64))?; }
            }
            resources.model.record(&completed)?;
            while !completed.completed()? {
                if started.elapsed() > std::time::Duration::from_secs(120) {
                    return Err("startup tuning graph completion deadline expired".into());
                }
                if let Some(health) = &self.manifest.collectives { (health.0)()?; }
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Record after any accepted commands so even a failed enqueue can drain.
            let drained = resources.model.record(&completed).and_then(|()| {
                loop {
                    if completed.completed()? { return Ok(()); }
                    if started.elapsed() > std::time::Duration::from_secs(120) {
                        return Err("startup CUDA drain deadline expired".into());
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            });
            if drained.is_err() {
                // Constructor failure cannot return an undrained ownership lease.
                eprintln!("startup tuning failed before CUDA drain: {error}; terminating the CUDA process");
                std::process::abort();
            }
            return Err(error);
        }
        let status = i32::from_ne_bytes(readback.bytes().try_into().unwrap());
        if status != 0 {
            return Err(format!("startup tuning graph {:?} failed with status {status}", graph.name).into());
        }
        self.diagnostic_choices.push(outputs.iter().map(|(buffer, host)| {
            (buffer.name.clone(), host.bytes().to_vec())
        }).collect());
        self.startup_next += 1;
        Ok(())
    }

    pub(super) fn report_startup(&self, started: Instant) {
        if self.startup_next != 0 {
            eprintln!("Startup tuning: {} graphs completed in {:.2}s", self.startup_next, started.elapsed().as_secs_f64());
            self.report_collective_choices();
        }
    }

    fn report_collective_choices(&self) {
        eprint!("{}", collective_choices_overview(&self.manifest, &self.diagnostic_choices));
    }

    pub(super) fn finish_startup(&mut self) -> Result<()> {
        if self.startup_next != self.manifest.referenced_tuning_graphs().len() {
            return Err("startup tuning graphs have not completed".into());
        }
        self.startup_ready = true;
        Ok(())
    }

    pub fn manifest(&self) -> &ExecutionManifest {
        &self.manifest
    }
    pub fn completed_fence(&self) -> Fence {
        self.completed
    }
    /// Fence to record on resources before the next successful submission attempt.
    /// A failed CUDA enqueue poisons the executor; shutdown must drain before reuse.
    pub fn next_fence(&self) -> Result<Fence> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                "executor is closed or failed",
                error.clone(),
            ))
            .into());
        }
        if self.resources.is_none() {
            return Err(Error::execution(
                ExecutionError::Closed,
                "executor is closed or failed",
            ));
        }
        if self.next_sequence > i64::MAX as u64 {
            return Err(Error::execution(
                ExecutionError::Capacity,
                "submission sequence exhausted",
            ));
        }
        Ok(Fence(self.next_sequence))
    }
    pub fn can_submit(&self, grammar_constrained: bool) -> bool {
        let Some(resources) = &self.resources else {
            return false;
        };
        self.startup_ready && self.poison.is_none()
            && resources.frames[self.next_sequence as usize % 2].pending.is_none()
            && !resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending.as_ref())
                .any(|pending| pending.grammar)
            && (!grammar_constrained
                || resources.frames.iter().all(|frame| frame.pending.is_none()))
    }
    pub fn outstanding(&self) -> usize {
        self.resources.as_ref().map_or(0, |resources| {
            resources
                .frames
                .iter()
                .filter(|frame| frame.pending.is_some())
                .count()
        })
    }

    /// Validate a coordinated submission without touching CUDA or advancing its fence.
    pub(super) fn validate_submission(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        let fence = self.next_fence()?;
        validate_uploads(&self.manifest, graph, uploads, &mut self.resolved_uploads)?;
        if !self.can_submit(grammar || !matches!(self.manifest.graphs[graph].kind,
            GraphKind::Decode | GraphKind::Speculative)) {
            return Ok(None);
        }
        if crate::engram::consumes(self.manifest.graphs[graph].kind) {
            if let Some(engram) = self.resources.as_ref().and_then(|resources| resources.engram.as_ref()) {
                engram.validate(fence.0, self.manifest.graphs[graph].packed_tokens(), uploads)?;
            }
        }
        Ok(Some(fence))
    }

    /// Retain allocations after a peer failure that may strand a GPU collective.
    pub(super) fn abandon(&mut self, error: Error) -> Shutdown {
        if let Some(resources) = self.resources.take() {
            std::mem::forget(resources);
        }
        let shutdown = Shutdown {
            drained: false,
            error: Some(error),
        };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }

    /// Returns None on frame or grammar backpressure without changing CUDA state.
    pub fn submit(
        &mut self,
        graph_index: usize,
        uploads: &[Upload<'_>],
        grammar_constrained: bool,
    ) -> Result<Option<Fence>> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                format!("executor failed: {error}"),
                error.clone(),
            ))
            .into());
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| Error::execution(ExecutionError::Closed, "executor is closed"))?;
        let graph =
            self.manifest.graphs.get(graph_index).ok_or_else(|| {
                Error::execution(ExecutionError::State, "graph index out of range")
            })?;
        let grammar_constrained = grammar_constrained
            || !matches!(graph.kind, GraphKind::Decode | GraphKind::Speculative);
        let mut resolved = validate_uploads(
            &self.manifest,
            graph_index,
            uploads,
            &mut self.resolved_uploads,
        )?;
        if (resources.frames.iter().any(|frame| frame.pending.is_some()) && grammar_constrained)
            || resources
                .frames
                .iter()
                .filter_map(|frame| frame.pending.as_ref())
                .any(|pending| pending.grammar)
        {
            return Ok(None);
        }
        let index = self.next_sequence as usize % 2;
        if resources.frames[index].pending.is_some() {
            return Ok(None);
        }
        if self.next_sequence > i64::MAX as u64 {
            return Err(Error::execution(
                ExecutionError::Capacity,
                "submission sequence exhausted",
            ));
        }
        if crate::engram::consumes(graph.kind) {
            if let Some(engram) = &resources.engram {
                engram.validate(self.next_sequence, graph.packed_tokens(), uploads)?;
            }
        }
        let fence = Fence(self.next_sequence);
        let queued = resources.frames.iter().any(|frame| frame.pending.is_some());
        let frame = &mut resources.frames[index];
        let dispatched = Instant::now();
        resolved.stage(frame.input.bytes_mut(), index as i32, fence.0 as i64);
        let logprobs = if self.copy_logprobs { resolved.logprobs } else { 0 };
        let logprob_width = resolved.logprob_width;
        let logprob_token_ids = if self.copy_logprobs {
            std::mem::take(&mut resolved.logprob_token_ids)
        } else {
            Vec::new()
        };
        let prompt_rows = if self.copy_logprobs && graph.kind == GraphKind::ScoredPrefill {
            let bytes = uploads.iter().filter(|u| u.buffer == "prompt_target_tokens")
                .map(|u| u.offset + u.bytes.len()).max().unwrap_or(0);
            if bytes == 0 || bytes % 4 != 0 || bytes / 4 > graph.token_capacity {
                return Err("scored prefill requires one target per live prompt row".into());
            }
            bytes / 4
        } else { 0 };
        frame.pending = Some(Pending {
            logprobs,
            logprob_width,
            logprob_token_ids,
            prompt_rows,
            fence,
            graph_index,
            grammar: grammar_constrained,
            dispatched,
            queued,
        });
        self.next_sequence += 1;
        let mut operation = || -> Result<()> {
            // Validated destinations are disjoint, so copies within a batch do
            // not depend on each other. Omitted input ranges remain untouched.
            resources.input_copies.submit(frame.input.address(), resolved.copies(), |dst, src, sizes| {
                // SAFETY: The manifest and pinned frame retain every validated
                // range. Pending frames remain immutable until completion, and
                // enqueue failures retain their owners until both streams drain.
                // Descriptor slices are consumed synchronously by the driver.
                unsafe { resources.model.copy_batch(dst, src, sizes) }
            })?;
            resources.model.record(&frame.started)?;
            if crate::engram::consumes(graph.kind) {
                if let Some(engram) = &resources.engram {
                    engram.enqueue(fence.0, graph.packed_tokens(), uploads, &frame.started)?;
                }
            }
            resources.model.launch(graph.handle)?;
            resources.model.record(&frame.produced)?;
            resources.copy.wait(&frame.produced)?;
            let address = self.manifest.buffer("completion_records")?.address
                + index * self.manifest.completion_bytes();
            // SAFETY: The manifest bounds both completion rows; index selects one
            // row in that retained allocation. The copy stream waits for the model
            // event, and the pinned output is not read or reused before transferred
            // completes. Shutdown drains both streams before releasing either owner.
            unsafe {
                resources.copy.download(
                    &frame.output,
                    address,
                    self.manifest.completion_bytes(),
                )?;
                if logprobs != 0 {
                    let b = self.manifest.buffer("completion_logprobs")?;
                    frame.logprobs.as_ref().unwrap().enqueue_pitched_copy(
                        &resources.copy, b.address + index * b.nbytes / 2,
                        b.shape[3] * 4, (2 + 2 * logprob_width) * 4,
                        self.manifest.request_capacity() * MAX_OUTPUT_TOKENS,
                    )?;
                }
                if prompt_rows != 0 {
                    let b = self.manifest.buffer("prompt_logprobs")?;
                    frame.prompt_logprobs.as_ref().unwrap().enqueue_copy(&resources.copy,
                        b.address + index * b.nbytes / 2, prompt_rows * super::PROMPT_FIELDS * 4)?;
                }
                if graph.confidence_width > 0 {
                    let confidence = self.manifest.buffer("proposal_confidence")?;
                    frame.confidence.as_ref().unwrap().enqueue_copy(
                        &resources.copy, confidence.address + index * confidence.nbytes / 2, confidence.nbytes / 2,
                    )?;
                }
                if reads_speculative_survival(graph) {
                    if let Some(readback) = &frame.survival {
                        let buffer = self.manifest.buffer("completion_speculative_survival")?;
                        readback.enqueue_copy(&resources.copy, buffer.address + index * buffer.nbytes / 2,
                                              buffer.nbytes / 2)?;
                    }
                }
                if crate::engram::consumes(graph.kind) {
                    if let Some(readback) = &frame.readiness {
                        let buffer = self.manifest.buffer("readiness_timestamps")?;
                        readback.enqueue_copy(&resources.copy, buffer.address + index * buffer.nbytes / 2,
                                              buffer.nbytes / 2)?;
                    }
                }
            }
            resources.copy.record(&frame.transferred)
        };
        if let Err(error) = operation() {
            self.poison = Some(error.clone());
            return Err(error);
        }
        Ok(Some(fence))
    }

    pub fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(error) = &self.poison {
            return Err(ExecutionError::Poisoned(Failure::caused_by(
                format!("executor failed: {error}"),
                error.clone(),
            ))
            .into());
        }
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| Error::execution(ExecutionError::Closed, "executor is closed"))?;
        let frame = resources
            .frames
            .iter_mut()
            .filter(|frame| frame.pending.is_some())
            .min_by_key(|frame| frame.pending.as_ref().unwrap().fence);
        let Some(frame) = frame else {
            return Ok(None);
        };
        match frame.transferred.completed() {
            Ok(false) => return Ok(None),
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
            Ok(true) => {}
        }
        if let Some(health) = &self.manifest.collectives {
            if let Err(error) = (health.0)() {
                self.poison = Some(error.clone());
                return Err(error);
            }
        }
        let pending = frame.pending.take().unwrap();
        if crate::engram::consumes(self.manifest.graphs[pending.graph_index].kind) {
            if let Some(engram) = &resources.engram {
                if let Err(error) = engram.finish(pending.fence.0) {
                    self.poison = Some(error.clone());
                    return Err(error);
                }
            }
        }
        self.completed = pending.fence;
        let records = match decode_records(
            frame.output.bytes(),
            pending.fence,
            self.manifest.request_capacity(),
            self.manifest.completion_fields(),
        ) {
            Ok(records) => records,
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        let logprobs = if pending.logprobs != 0 {
            // SAFETY: This frame's transfer event covers its sidecar; the source
            // and pinned destination remain owned until decoding completes.
            let readback = frame.logprobs.as_ref().unwrap();
            let fields = match readback {
                Readback::MappedHost { .. } => self.manifest.buffer("completion_logprobs")?.shape[3],
                Readback::DeviceCopy(_) => 2 + 2 * pending.logprob_width,
            };
            let bytes = unsafe { readback.bytes() };
            let nbytes = records.len() * MAX_OUTPUT_TOKENS * fields * size_of::<f32>();
            match decode_logprobs(
                &bytes[..nbytes],
                &records,
                self.manifest.max_top_logprobs()?,
                fields,
                pending.logprobs,
                &pending.logprob_token_ids,
            ) {
                Ok(values) => Some(values),
                Err(error) => {
                    self.poison = Some(error.clone());
                    return Err(error);
                }
            }
        } else {
            None
        };
        let prompt_logprobs = if pending.prompt_rows != 0 && !records.iter().any(|r| r.status == 3) {
            // SAFETY: The transfer event covers this frame; decoding precedes its release.
            let bytes = unsafe { frame.prompt_logprobs.as_ref().unwrap().bytes() };
            match decode_prompt_logprobs(bytes, pending.prompt_rows) {
                Ok(values) => Some(values),
                Err(error) => { self.poison = Some(error.clone()); return Err(error); }
            }
        } else { None };
        let elapsed_ns = match frame.produced.elapsed_ns(&frame.started) {
            Ok(value) => value,
            Err(error) => {
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        let readiness = if crate::engram::consumes(self.manifest.graphs[pending.graph_index].kind) {
            frame.readiness.as_ref().map(|readback| {
                // SAFETY: The frame's transfer event completed before decoding its retained readback.
                ReadinessIntervals::decode(unsafe { readback.bytes() })
            }).transpose()
        } else { Ok(None) };
        let readiness = match readiness {
            Ok(value) if value.is_none_or(|sample| sample.wait_ns() <= elapsed_ns) => value,
            value => {
                let error = value.err().unwrap_or_else(|| Error::execution(ExecutionError::Completion,
                    "readiness delay exceeds its enclosing CUDA interval"));
                self.poison = Some(error.clone());
                return Err(error);
            }
        };
        let speculative_survival = if reads_speculative_survival(&self.manifest.graphs[pending.graph_index]) {
            frame.survival.as_ref().map(|readback| {
                let mut rows = [[-1.0; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES];
                // SAFETY: The completion transfer protects this frame's retained feedback bytes.
                for (row, bytes) in rows.iter_mut().zip(unsafe { readback.bytes() }.chunks_exact(7 * 4)) {
                    for (value, bytes) in row.iter_mut().zip(bytes.chunks_exact(4)) {
                        *value = f32::from_ne_bytes(bytes.try_into().unwrap());
                    }
                }
                rows
            })
        } else { None };
        if let Some(rows) = &speculative_survival {
            if let Err(error) = validate_speculative_survival(rows, &records) {
                self.poison = Some(error.clone());
                return Err(error);
            }
        }
        Ok(Some(FrameCompletion {
            fence: pending.fence,
            graph_index: pending.graph_index,
            timing: ExecutionTiming {
                readiness,
                gpu_ns: [
                    Some(elapsed_ns),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                wall_ns: pending
                    .dispatched
                    .elapsed()
                    .as_nanos()
                    .min(u64::MAX as u128) as u64,
                queued: pending.queued,
            },
            records,
            logprobs,
            prompt_logprobs,
            speculative_survival,
            proposal_confidence: if self.manifest.graphs[pending.graph_index].confidence_width > 0
            {
                let mut confidence = [[0.0; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES];
                let graph = &self.manifest.graphs[pending.graph_index];
                for (row, bytes) in confidence
                    .iter_mut()
                    .take(graph.batch_capacity)
                    // SAFETY: The transfer event follows the producing graph;
                    // this frame remains reserved while its sidecar is decoded.
                    .zip(unsafe { frame.confidence.as_ref().unwrap().bytes() }
                        .chunks_exact(4 * self.manifest.buffer("proposal_confidence")?.shape[2]))
                {
                    for (value, bytes) in row.iter_mut().take(graph.confidence_width).zip(bytes.chunks_exact(4)) {
                        *value = f32::from_ne_bytes(bytes.try_into().unwrap());
                    }
                }
                Some(confidence)
            } else {
                None
            },
        }))
    }

    pub fn close(&mut self) -> Shutdown {
        if let Some(shutdown) = &self.shutdown {
            return shutdown.clone();
        }
        let resources = self.resources.take().unwrap();
        if let Some(engram) = &resources.engram {
            engram.cancel();
        }
        let compute = resources.model.synchronize().map_err(Error::drain);
        let copy = resources.copy.synchronize().map_err(Error::drain);
        let engram = resources
            .engram
            .as_ref()
            .map_or(Ok(()), |e| e.drain())
            .map_err(Error::drain);
        let drained = compute.is_ok() && copy.is_ok() && engram.is_ok();
        let error = compute
            .err()
            .or(copy.err())
            .or(engram.err())
            .or(self.poison.take());
        if drained {
            drop(resources);
        } else {
            std::mem::forget(resources);
        }
        let shutdown = Shutdown { drained, error };
        self.shutdown = Some(shutdown.clone());
        shutdown
    }
}

impl Drop for GraphExecutor {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResolvedUpload {
    pub(super) host_offset: usize,
    pub(super) device_address: usize,
    pub(super) nbytes: usize,
}

/// Borrowed descriptors are consumed synchronously; CUDA retains only frame bytes.
pub(super) struct ResolvedUploads<'uploads, 'bytes, 'storage> {
    uploads: &'uploads [Upload<'bytes>],
    bindings: &'storage [ResolvedUpload],
    frame_index: ResolvedUpload,
    submission_sequence: ResolvedUpload,
    logprobs: u64,
    logprob_width: usize,
    logprob_token_ids: Vec<Vec<u32>>,
}

impl ResolvedUploads<'_, '_, '_> {
    pub(super) fn stage(&self, frame: &mut [u8], index: i32, sequence: i64) {
        for (upload, binding) in self.uploads.iter().zip(self.bindings) {
            frame[binding.host_offset..binding.host_offset + binding.nbytes]
                .copy_from_slice(upload.bytes);
        }
        for (binding, bytes) in [
            (self.frame_index, index.to_le_bytes().as_slice()),
            (self.submission_sequence, sequence.to_le_bytes().as_slice()),
        ] {
            frame[binding.host_offset..binding.host_offset + binding.nbytes].copy_from_slice(bytes);
        }
    }

    pub(super) fn copies(&self) -> impl Iterator<Item = ResolvedUpload> + '_ {
        self.bindings
            .iter()
            .copied()
            .chain([self.frame_index, self.submission_sequence])
    }
}

/// Resolve one command against an already validated, immutable manifest.
/// Omitted inputs and disjoint partial uploads preserve resident device ranges.
/// Storage grows only to the fragment-count high-water; valid nonempty disjoint
/// fragments are bounded by input_frame_bytes. The executor reserves four uploads
/// per binding for physical request lanes before accepting any submissions.
pub(super) fn validate_uploads<'uploads, 'bytes, 'storage>(
    manifest: &ExecutionManifest,
    graph_index: usize,
    uploads: &'uploads [Upload<'bytes>],
    storage: &'storage mut Vec<ResolvedUpload>,
) -> Result<ResolvedUploads<'uploads, 'bytes, 'storage>> {
    storage.clear();
    let graph = manifest
        .graphs
        .get(graph_index)
        .ok_or_else(|| Error::execution(ExecutionError::Upload, "graph index out of range"))?;
    if graph.kind == GraphKind::ScoredPrefill {
        let bytes = uploads.iter().filter(|upload| upload.buffer == "prompt_target_tokens")
            .map(|upload| upload.offset.saturating_add(upload.bytes.len())).max().unwrap_or(0);
        if bytes == 0 || bytes % 4 != 0 || bytes / 4 > graph.token_capacity {
            return Err("scored prefill requires one target per live prompt row".into());
        }
    }
    if uploads.len() > manifest.input_frame_bytes {
        return Err(Error::execution(
            ExecutionError::Upload,
            "upload count exceeds the nonempty input-fragment bound",
        ));
    }
    let reserved = |name: &str| -> Result<ResolvedUpload> {
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == name)
            .ok_or_else(|| {
                Error::execution(
                    ExecutionError::Upload,
                    "graph lacks a reserved input binding",
                )
            })?;
        let buffer = manifest.buffer(name)?;
        Ok(ResolvedUpload {
            host_offset: binding.host_offset,
            device_address: buffer.address,
            nbytes: buffer.nbytes,
        })
    };
    let frame_index = reserved("frame_index")?;
    let submission_sequence = reserved("submission_sequence")?;
    for upload in uploads {
        if ["frame_index", "submission_sequence"].contains(&upload.buffer) {
            return Err(Error::execution(
                ExecutionError::Upload,
                "reserved executor inputs cannot be overridden",
            ));
        }
        let binding = graph
            .input_bindings
            .iter()
            .find(|binding| binding.buffer_name == upload.buffer)
            .ok_or_else(|| {
                Error::execution(ExecutionError::Upload, "upload is not bound by this graph")
            })?;
        let buffer = manifest.buffer(upload.buffer)?;
        if upload.bytes.is_empty()
            || upload.offset % buffer.dtype.size() != 0
            || upload.bytes.len() % buffer.dtype.size() != 0
            || upload
                .offset
                .checked_add(upload.bytes.len())
                .is_none_or(|end| end > binding.nbytes)
        {
            return Err(Error::execution(
                ExecutionError::Upload,
                format!("unaligned or out-of-range upload: {}", upload.buffer),
            ));
        }
        let resolved = ResolvedUpload {
            host_offset: binding.host_offset + upload.offset,
            device_address: buffer.address + upload.offset,
            nbytes: upload.bytes.len(),
        };
        // Manifest validation makes different buffers' host spans disjoint.
        if storage.iter().any(|prior| {
            prior.host_offset < resolved.host_offset + resolved.nbytes
                && resolved.host_offset < prior.host_offset + prior.nbytes
        }) {
            return Err(Error::execution(
                ExecutionError::Upload,
                "uploads overlap within one input buffer",
            ));
        }
        storage.push(resolved);
    }
    let mut logprobs = 0u64;
    if graph.kind.samples() {
        for upload in uploads.iter().filter(|u| u.buffer == "logprob_requested") {
            for (i, bytes) in upload.bytes.chunks_exact(size_of::<i32>()).enumerate() {
                if i32::from_ne_bytes(bytes.try_into().unwrap()) != 0 {
                    logprobs |= 1 << (upload.offset / size_of::<i32>() + i);
                }
            }
        }
    }
    let mut logprob_width = if logprobs != 0 {
        (manifest.buffer("completion_logprobs")?.shape[3] / 2 - 1).min(super::MAX_TOP_LOGPROBS)
    } else {
        0
    };
    let mut logprob_token_ids = Vec::new();
    if graph.kind.samples() && manifest.logprob_token_id_capacity() != 0 {
        let capacity = manifest.request_capacity();
        if !uploads.iter().any(|u| {
            u.buffer == "logprob_requested" && u.offset == 0 && u.bytes.len() == capacity * 4
        }) {
            return Err(
                "sampling requires all generation logprob gates on every submission".into(),
            );
        }
        let counts = uploads
            .iter()
            .find(|u| {
                u.buffer == "logprob_token_id_counts"
                    && u.offset == 0
                    && u.bytes.len() == capacity * 4
            })
            .ok_or("sampling requires all logprob token ID counts on every submission")?;
        for (slot, count) in counts.bytes.chunks_exact(4).enumerate() {
            let count = i32::from_ne_bytes(count.try_into().unwrap());
            if !(0..=super::MAX_LOGPROB_TOKEN_IDS as i32).contains(&count)
                || (count != 0 && logprobs & (1 << slot) == 0)
            {
                return Err("invalid logprob token ID count or request gate".into());
            }
            if count == 0 {
                continue;
            }
            if logprob_token_ids.is_empty() {
                logprob_token_ids.resize_with(capacity, Vec::new);
            }
            let ids = uploads
                .iter()
                .find(|u| {
                    u.buffer == "logprob_token_ids"
                        && u.offset == slot * super::MAX_LOGPROB_TOKEN_IDS * 4
                        && u.bytes.len() == count as usize * 4
                })
                .ok_or("missing requested logprob token ID prefix")?;
            for id in ids.bytes.chunks_exact(4) {
                let id = i32::from_ne_bytes(id.try_into().unwrap());
                if id < 0 || logprob_token_ids[slot].contains(&(id as u32)) {
                    return Err("invalid or duplicate requested logprob token ID".into());
                }
                logprob_token_ids[slot].push(id as u32);
            }
            logprob_width = logprob_width.max(count as usize);
        }
    }
    Ok(ResolvedUploads {
        uploads,
        bindings: storage,
        frame_index,
        submission_sequence,
        logprobs,
        logprob_width,
        logprob_token_ids,
    })
}

pub(super) fn decode_records(
    bytes: &[u8],
    fence: Fence,
    capacity: usize,
    fields: usize,
) -> Result<CompletionRecords> {
    if !(1..=REQUEST_LANES).contains(&capacity)
        || fields != VECTOR_RECORD_FIELDS
        || bytes.len() != capacity * fields * 8
    {
        return Err(Error::execution(
            ExecutionError::Completion,
            "completion extent mismatch",
        ));
    }
    let mut records = CompletionRecords {
        records: [CompletionRecord::default(); REQUEST_LANES],
        len: capacity,
    };
    for (record, bytes) in records.iter_mut().zip(bytes.chunks_exact(VECTOR_RECORD_FIELDS * 8)) {
        let mut fields = [0i64; VECTOR_RECORD_FIELDS];
        for (field, bytes) in fields.iter_mut().zip(bytes.chunks_exact(8)) {
            *field = i64::from_le_bytes(bytes.try_into().unwrap());
        }
        *record = CompletionRecord {
            generation: fields[0],
            submission_sequence: fields[1],
            consumed_tokens: fields[2],
            generated_tokens: fields[3],
            token_id: fields[4],
            status: fields[5],
            error_code: fields[6],
            reserved: fields[7],
            vector: Some(TokenVector {
                emitted_count: fields[8],
                offered_drafts: fields[9],
                accepted_drafts: fields[10],
                copy_offered_mask: fields[11],
                model_reserved: fields[12],
                draft_cache_consumed: fields[13],
                verification_width: fields[15],
                tokens: fields[16..24].try_into().unwrap(),
            }),
        };
        if record.status != 0 && fields[14] != 0 {
            return Err(Error::execution(
                ExecutionError::Completion,
                "reserved vector completion field must be zero",
            ));
        }
        record.validate_vector()?;
        if !(0..=3).contains(&record.status)
            || record.reserved != 0
            || (record.status != 0
                && (record.generation <= 0
                    || record.submission_sequence != fence.0 as i64
                    || record.consumed_tokens < 0
                    || record.generated_tokens < 0
                    || record.token_id < -1
                    || (record.status != 3 && record.error_code != 0)))
        {
            return Err(Error::execution(
                ExecutionError::Completion,
                "invalid or stale device completion record",
            ));
        }
    }
    Ok(records)
}

pub(super) fn decode_logprobs(
    bytes: &[u8],
    records: &[CompletionRecord],
    k: usize,
    fields: usize,
    requested: u64,
    selections: &[Vec<u32>],
) -> Result<Vec<[Option<TokenLogprobs>; MAX_OUTPUT_TOKENS]>> {
    if !(1..=super::MAX_TOP_LOGPROBS).contains(&k)
        || fields < 2 + 2 * k
        || fields > super::LOGPROB_FIELDS
        || fields % 2 != 0
        || bytes.len() != records.len() * MAX_OUTPUT_TOKENS * fields * size_of::<i32>()
    {
        return Err(Error::execution(
            ExecutionError::Completion,
            "generation logprob frame extent mismatch",
        ));
    }
    let mut result: Vec<[Option<TokenLogprobs>; MAX_OUTPUT_TOKENS]> = (0..records.len())
        .map(|_| std::array::from_fn(|_| None))
        .collect();
    for (slot, record) in records
        .iter()
        .enumerate()
        .filter(|(slot, r)| matches!(r.status, 1 | 2) && requested & (1 << slot) != 0)
    {
        let count = record
            .vector
            .map_or(usize::from(record.token_id >= 0), |v| {
                v.emitted_count as usize
            });
        if count > MAX_OUTPUT_TOKENS {
            return Err("generation logprob emitted count exceeds the frame".into());
        }
        for position in 0..count {
            let start = (slot * MAX_OUTPUT_TOKENS + position) * fields * size_of::<i32>();
            let int = |at: usize| {
                let offset = start + at * size_of::<i32>();
                i32::from_ne_bytes(bytes[offset..offset + size_of::<i32>()].try_into().unwrap())
            };
            let float = |at| f32::from_bits(int(at) as u32);
            let expected = record
                .vector
                .map_or(record.token_id, |v| v.tokens[position]);
            if int(0) < 0 || i64::from(int(0)) != expected || float(1).is_nan() {
                return Err("generated token is missing its raw logprob record".into());
            }
            let ids = selections.get(slot).map_or(&[][..], Vec::as_slice);
            let count = if ids.is_empty() { k } else { ids.len() };
            if count > super::MAX_LOGPROB_TOKEN_IDS || 2 + 2 * count > fields {
                return Err("generation logprob selection exceeds the copied width".into());
            }
            let mut value = TokenLogprobs {
                token: int(0) as u32,
                logprob: float(1),
                top: Vec::with_capacity(count + usize::from(!ids.is_empty())),
            };
            for i in 0..count {
                let id = int(super::LOGPROB_PAIR_FIELDS * (1 + i));
                if id < 0 && ids.is_empty() {
                    break;
                }
                if id < 0 || (!ids.is_empty() && id as u32 != ids[i]) {
                    return Err(
                        "generation logprob ID does not match the requested selection".into(),
                    );
                }
                let lp = float(super::LOGPROB_PAIR_FIELDS * (1 + i) + 1);
                if lp.is_nan() {
                    return Err(Error::execution(
                        ExecutionError::Completion,
                        "invalid generation logprob alternative",
                    ));
                }
                if ids.is_empty() && i > 0 {
                    let (previous, p) = value.top[i - 1];
                    if lp > p || (lp == p && id as u32 <= previous) {
                        return Err(Error::execution(
                            ExecutionError::Completion,
                            "generation logprob alternatives are not ordered",
                        ));
                    }
                }
                value.top.push((id as u32, lp));
            }
            if !ids.is_empty() && !ids.contains(&value.token) {
                value.top.push((value.token, value.logprob));
            }
            result[slot][position] = Some(value);
        }
    }
    Ok(result)
}

pub(super) fn decode_prompt_logprobs(bytes: &[u8], rows: usize) -> Result<Vec<Option<PromptLogprobs>>> {
    let fields = super::PROMPT_FIELDS;
    if bytes.len() < rows * fields * 4 {
        return Err("prompt logprob frame extent mismatch".into());
    }
    bytes[..rows * fields * 4].chunks_exact(fields * 4).map(|row| {
        let int = |i: usize| i32::from_ne_bytes(row[i * 4..(i + 1) * 4].try_into().unwrap());
        let float = |i: usize| f32::from_bits(int(i) as u32);
        if int(0) == -1 {
            if int(2) != 0 || !float(1).is_nan() {
                return Err("invalid prompt logprob guard".into());
            }
            return Ok(None);
        }
        if int(0) < 0 || int(2) <= 0 || float(1).is_nan() || float(1) > 0. {
            return Err("invalid prompt logprob result".into());
        }
        let mut value = TokenLogprobs {
            token: int(0) as u32, logprob: float(1),
            top: Vec::with_capacity(super::MAX_TOP_LOGPROBS),
        };
        for i in 0..super::MAX_TOP_LOGPROBS {
            let token = int(3 + 2 * i);
            if token == -1 {
                break;
            }
            let lp = float(4 + 2 * i);
            if token < 0 || lp.is_nan() || lp > 0. {
                return Err("invalid prompt alternative".into());
            }
            if i > 0 {
                let (previous, previous_lp) = value.top[i - 1];
                if lp > previous_lp || (lp == previous_lp && token as u32 <= previous) {
                    return Err("prompt alternatives are not ordered".into());
                }
            }
            value.top.push((token as u32, lp));
        }
        Ok(Some(PromptLogprobs { value, rank: int(2) as u32 }))
    }).collect()
}

#[cfg(test)]
mod input_copy_tests {
    use super::*;

    #[test]
    fn input_batches_preserve_every_partial_range_across_both_driver_limits() {
        let host = 0x10000;
        let mut copies = vec![ResolvedUpload {
            host_offset: 128, device_address: 0x20000000, nbytes: BATCH_BYTES + 16,
        }];
        copies.extend((0..BATCH_ENTRIES + 3).map(|i| ResolvedUpload {
            host_offset: BATCH_BYTES + 1024 + i * 8,
            device_address: 0x40000000 + i * 16,
            nbytes: 8,
        }));
        let mut batches = InputCopyBatches::new(4);
        let mut recorded = Vec::new();
        batches.submit(host, copies.iter().copied(), |dst, src, sizes| {
            assert!(sizes.len() <= BATCH_ENTRIES);
            assert!(sizes.iter().sum::<usize>() <= BATCH_BYTES);
            assert!(sizes.iter().all(|&size| size > 0));
            recorded.extend(dst.iter().zip(src).zip(sizes).map(|((&d, &s), &n)| (d, s, n)));
            Ok(())
        }).unwrap();
        let mut actual = recorded.into_iter();
        for copy in copies {
            let mut offset = 0;
            while offset < copy.nbytes {
                let (dst, src, nbytes) = actual.next().unwrap();
                assert_eq!(dst, (copy.device_address + offset) as u64);
                assert_eq!(src, (host + copy.host_offset + offset) as u64);
                assert!(nbytes <= copy.nbytes - offset);
                offset += nbytes;
            }
        }
        assert!(actual.next().is_none());
        assert!(batches.sizes.is_empty());
    }

    #[test]
    fn failed_batch_stops_submission_and_does_not_retain_descriptors_for_reuse() {
        let mut batches = InputCopyBatches::new(4);
        let copy = ResolvedUpload { host_offset: 16, device_address: 4096, nbytes: BATCH_BYTES + 8 };
        let mut calls = 0;
        assert!(batches.submit(8192, std::iter::once(copy), |_, _, _| {
            calls += 1;
            Err("copy failed".into())
        }).is_err());
        assert_eq!(calls, 1);
        batches.submit(8192, std::iter::once(ResolvedUpload { nbytes: 8, ..copy }), |dst, src, sizes| {
            assert_eq!(dst, &[4096]);
            assert_eq!(src, &[8208]);
            assert_eq!(sizes, &[8]);
            Ok(())
        }).unwrap();
        batches.submit(8192, std::iter::empty(), |_, _, _| panic!("empty input must not enqueue")).unwrap();
    }
}

#[cfg(all(test, feature = "runtime-package"))]
mod confidence_tests {
    use super::*;
    use crate::runtime_package::driver::{CopyEndpoint, CopyParams, Driver};

    #[test]
    #[ignore = "requires a CUDA device selected through CUDA_VISIBLE_DEVICES"]
    fn mapped_and_device_confidence_survive_repeated_graph_replay() {
        const BYTES: usize = 160;
        let driver = Driver::load().unwrap();
        let mut context = std::ptr::null_mut();
        let (mut source, mut device_output) = (0, 0);
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            let mut device = 0;
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            driver
                .check(
                    "cuCtxCreate_v2",
                    (driver.context_create)(&mut context, 0, device),
                )
                .unwrap();
            driver
                .check("cuMemAlloc_v2", (driver.allocate)(&mut source, BYTES))
                .unwrap();
            driver
                .check(
                    "cuMemAlloc_v2",
                    (driver.allocate)(&mut device_output, BYTES),
                )
                .unwrap();
        }
        {
            let model = Stream::open(context as usize, 0).unwrap();
            let copy = Stream::open(context as usize, 0).unwrap();
            let mapped = model.mapped_host_buffer(BYTES).unwrap();
            let mapped_address = model.mapped_address(&mapped).unwrap();
            let mut input = model.host_buffer(BYTES).unwrap();
            for address in [mapped_address, device_output as usize] {
                let confidence = Readback::new(&model, address, BYTES).unwrap();
                assert_eq!(
                    matches!(confidence, Readback::MappedHost { .. }),
                    address == mapped_address
                );
                let mut graph = std::ptr::null_mut();
                let mut executable = std::ptr::null_mut();
                unsafe {
                    driver
                        .check("cuGraphCreate", (driver.graph_create)(&mut graph, 0))
                        .unwrap();
                    let endpoint = |address| CopyEndpoint {
                        memory_type: 2,
                        device: address,
                        pitch: BYTES,
                        height: 1,
                        ..CopyEndpoint::default()
                    };
                    let params = CopyParams {
                        source: endpoint(source),
                        destination: endpoint(address as u64),
                        width: BYTES,
                        height: 1,
                        depth: 1,
                    };
                    let mut node = std::ptr::null_mut();
                    driver
                        .check(
                            "cuGraphAddMemcpyNode",
                            (driver.graph_copy)(
                                &mut node,
                                graph,
                                std::ptr::null(),
                                0,
                                &params,
                                context,
                            ),
                        )
                        .unwrap();
                    driver
                        .check(
                            "cuGraphInstantiateWithFlags",
                            (driver.graph_instantiate)(&mut executable, graph, 0),
                        )
                        .unwrap();
                }
                let produced = model.event().unwrap();
                let transferred = copy.event().unwrap();
                for generation in 1..=16 {
                    let values: Vec<u8> = (0..BYTES / 4)
                        .flat_map(|i| ((generation * 100 + i) as f32).to_ne_bytes())
                        .collect();
                    input.bytes_mut().copy_from_slice(&values);
                    unsafe {
                        model.upload(source as usize, &input, 0, BYTES).unwrap();
                    }
                    model.launch(executable as usize).unwrap();
                    model.record(&produced).unwrap();
                    copy.wait(&produced).unwrap();
                    unsafe {
                        confidence.enqueue_copy(&copy, address, BYTES).unwrap();
                    }
                    copy.record(&transferred).unwrap();
                    while !transferred.completed().unwrap() {
                        std::thread::yield_now();
                    }
                    // The transfer fence covers the mapped graph output as well
                    // as the copied device output before either is read or reused.
                    assert_eq!(unsafe { confidence.bytes() }, values);
                }
                copy.synchronize().unwrap();
                model.synchronize().unwrap();
                unsafe {
                    driver
                        .check(
                            "cuGraphExecDestroy",
                            (driver.executable_destroy)(executable),
                        )
                        .unwrap();
                    driver
                        .check("cuGraphDestroy", (driver.graph_destroy)(graph))
                        .unwrap();
                }
            }
        }
        unsafe {
            driver.check("cuMemFree_v2", (driver.free)(source)).unwrap();
            driver
                .check("cuMemFree_v2", (driver.free)(device_output))
                .unwrap();
            driver
                .check("cuCtxDestroy_v2", (driver.context_destroy)(context))
                .unwrap();
        }
    }
}
