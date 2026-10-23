use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::{
    AdmissionError, Error, ExecutionError, Failure, Result, Shutdown,
    cache::ConditioningEvent,
    execution::{ExecutionManifest, GraphKind, RankExecutor, SamplingMode},
    model::{ModelConfig, SamplingParameters},
    scheduler::{
        RequestHandle, RequestPhase, RequestScheduler, RequestSpec, ScheduledPlan, ScheduledStep,
    },
};

pub struct Prompt {
    pub tokens: Vec<u32>,
    /// Sorted token counts of exact rendered message prefixes.
    pub message_boundaries: Vec<usize>,
    pub data: crate::model::PromptData,
    pub conditioned_inputs: Vec<crate::frontend::images::PreparedImageInput>,
    pub conditioning: Vec<ConditioningEvent>,
    pub sampling: SamplingParameters,
    pub max_new_tokens: u32,
    pub stop_sequences: Vec<Vec<u32>>,
    pub stop_strings: Vec<String>,
    pub grammar: Option<crate::grammar::RequestGrammar>,
    /// Generate through end-of-sequence tokens until the output limit, as vLLM's `ignore_eos`.
    pub ignore_eos: bool,
    pub logprobs: bool,
    pub logprob_token_ids: Vec<u32>,
    pub prompt_logprobs: bool,
}

impl Prompt {
    fn has_host_stops(&self) -> bool {
        !self.stop_sequences.is_empty() || !self.stop_strings.is_empty()
    }
}

#[derive(Clone, Debug)]
pub enum GenerationEvent {
    Token(u32),
    LogprobToken(crate::execution::TokenLogprobs),
    Protocol(Vec<crate::output::Piece>),
    PromptLogprobs { start: usize, tokens: Vec<u32>, values: Vec<Option<crate::execution::PromptLogprobs>> },
    ObservedToken {
        token: u32,
        timestamp_ns: Option<u64>,
    },
    Finished {
        prompt_tokens: usize,
        generated_tokens: u32,
        reused_prompt_tokens: u64,
        reason: &'static str,
        termination: crate::output::Termination,
        stop_sequence: Option<String>,
    },
    AdmissionRejected(Error),
    Error(Error),
}

pub struct Generation {
    pub events: async_mpsc::Receiver<GenerationEvent>,
    admission: oneshot::Receiver<Result<()>>,
    /// Completes before stop matching or decoding can hide the first token.
    first_token: oneshot::Receiver<Result<()>>,
    cancelled: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<Error>>>,
    pub(super) enqueue_ns: Option<u64>,
}

/// A buffered group waits for one admission. Waiting for every choice would
/// leave admitted choices' bounded output queues undrained.
pub(super) async fn await_admission(generations: &mut [Generation]) -> Result<()> {
    use std::{future::Future, pin::Pin, task::Poll};
    std::future::poll_fn(|cx| {
        let mut admitted = generations.is_empty();
        for generation in generations.iter_mut() {
            if let Some(error) = generation
                .failure
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
            {
                return Poll::Ready(Err(error));
            }
            match Pin::new(&mut generation.admission).poll(cx) {
                Poll::Ready(Ok(Ok(()))) => admitted = true,
                Poll::Ready(Ok(Err(error))) => return Poll::Ready(Err(error)),
                Poll::Ready(Err(_)) => return Poll::Ready(Err(generation.failure())),
                Poll::Pending => {}
            }
        }
        if admitted {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    })
    .await
}

impl Generation {
    pub(super) async fn wait_for_first_token(&mut self) -> Result<()> {
        (&mut self.first_token)
            .await
            .unwrap_or_else(|_| Err(self.failure()))
    }

    pub fn failure(&self) -> Error {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap_or_else(|| {
                Error::execution(
                    ExecutionError::Worker,
                    "model worker closed without a terminal completion",
                )
            })
    }
}
impl Drop for Generation {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

struct Incoming {
    _capacity: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    arrival_ns: u64,
    publications: std::cell::Cell<u64>,
    prompt: Prompt,
    events: async_mpsc::Sender<GenerationEvent>,
    admission: Mutex<Option<oneshot::Sender<Result<()>>>>,
    first_token: std::cell::Cell<Option<oneshot::Sender<Result<()>>>>,
    cancelled: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<Error>>>,
    metrics: Arc<super::metrics::Metrics>,
    observation: Option<bool>,
    /// Consult the host prefix store at admission; cleared after a failed restore.
    host_lookup: bool,
    /// Admitted once already, so the admission counter is not advanced again.
    readmitted: bool,
    prompt_score_end: usize,
    prompt_scored: usize,
    /// Retry capacity admission only after the scheduler's resources change.
    blocked: Option<(crate::cache::Fence, usize, usize, usize)>,
}
impl Incoming {
    fn prompt_score_range(&self, start: usize, tokens: usize) -> std::ops::Range<usize> {
        let first = (if start == 0 { 0 } else { start + 1 }).max(self.prompt_scored);
        first..(start + tokens + 1).min(self.prompt_score_end)
    }

    fn notify_admission(&self, result: Result<()>) {
        if let Some(sender) = self
            .admission
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = sender.send(result);
        }
    }
    fn fail(&self, error: Error) {
        self.finish_error(error, GenerationEvent::Error);
    }
    fn reject_admission(&self, error: Error) {
        self.finish_error(error, GenerationEvent::AdmissionRejected);
    }
    fn finish_error(&self, error: Error, event: fn(Error) -> GenerationEvent) {
        let mut failure = self.failure.lock().unwrap_or_else(|e| e.into_inner());
        if failure.is_some() {
            return;
        }
        *failure = Some(error.clone());
        self.notify_admission(Err(error.clone()));
        if let Some(sender) = self.first_token.take() {
            let _ = sender.send(Err(error.clone()));
        }
        self.metrics.failed.fetch_add(1, Ordering::Relaxed);
        let _ = self.events.try_send(event(error));
    }
    fn emit_token(&self, token: u32) {
        self.emit(GenerationEvent::Token(token));
    }
    fn observe_token(&self, token: u32) {
        if let Some(sender) = self.first_token.take() {
            let _ = sender.send(Ok(()));
        }
        if let Some(timestamps) = self.observation {
            let timestamp_ns = timestamps.then(super::observation::monotonic_ns);
            self.emit(GenerationEvent::ObservedToken {
                token,
                timestamp_ns,
            });
        }
    }
    fn emit(&self, event: GenerationEvent) {
        let publishes = matches!(
            &event,
            GenerationEvent::Token(_) | GenerationEvent::LogprobToken(_) | GenerationEvent::Protocol(_)
        );
        if self.events.try_send(event).is_err() {
            self.fail(Error::execution(
                ExecutionError::Capacity,
                "generation consumer exceeded the bounded output queue",
            ));
            self.cancelled.store(true, Ordering::Release);
        } else if publishes {
            self.publications
                .set(self.publications.get().saturating_add(1));
        }
    }
}

struct Active<'a> {
    request: RequestHandle,
    incoming: Incoming,
    last_token: Option<u32>,
    next_image: usize,
    selected_images: Vec<crate::frontend::images::PreparedImageInput>,
    prepared_image: Option<crate::frontend::images::ProcessedImages>,
    stops: super::stops::TokenStops,
    text_stops: Option<super::stops::TextStops<'a>>,
    carried: Carried,
    held_logprobs: VecDeque<Option<crate::execution::TokenLogprobs>>,
}

/// Output accounting and decoded tail retained across admissions.
#[derive(Clone, Copy, Default)]
struct Carried {
    decoded_sequence: crate::scheduler::decoded_history::DecodedSequence,
    prompt_tokens: Option<usize>,
    reused_prompt_tokens: Option<u64>,
    generated_tokens: u32,
    /// The request's own sampling seed, from which each continuation's seed derives.
    seed: Option<u64>,
}

/// The sampling seed of a preempted request's continuation after `generated` tokens.
/// Readmission restarts the device's random streams, so the continuation draws from
/// a stream that depends on the seed and the preemption point instead of repeating
/// the first one; the same preemption point gives the same continuation.
fn continuation_seed(seed: u64, generated: u32) -> u64 {
    // SplitMix64 over the seed and the generated-token count.
    let mut value = seed ^ u64::from(generated).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The output state of a preempted request whose history waits for readmission.
struct Parked<'a> {
    last_token: Option<u32>,
    stops: super::stops::TokenStops,
    text_stops: Option<super::stops::TextStops<'a>>,
    carried: Carried,
    held_logprobs: VecDeque<Option<crate::execution::TokenLogprobs>>,
}

impl Active<'_> {
    fn emit_output(
        &mut self,
        token: u32,
        eos: &[u32],
        logprob: Option<crate::execution::TokenLogprobs>,
    ) -> Result<bool> {
        if let Some(stops) = &mut self.text_stops {
            let silent = eos.contains(&token);
            let (pieces, matched) = if let Some(value) = logprob {
                stops.push_logprob(token, value, silent)?
            } else {
                if silent {
                    return Ok(false);
                }
                stops.push(token)?
            };
            if !pieces.is_empty() {
                self.incoming.emit(GenerationEvent::Protocol(pieces));
            }
            Ok(matched)
        } else {
            if self.incoming.prompt.logprobs {
                self.held_logprobs.push_back(logprob);
            }
            let (visible, matched) = self.stops.push(token);
            for token in visible {
                let value = self.held_logprobs.pop_front().flatten();
                self.incoming.emit(match value {
                    Some(value) => GenerationEvent::LogprobToken(value),
                    None => GenerationEvent::Token(token),
                });
            }
            Ok(matched)
        }
    }
}

pub struct ModelRuntime {
    sender: mpsc::SyncSender<Vec<Incoming>>,
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    config: Arc<ModelConfig>,
    event_capacity: usize,
    capacity: Arc<tokio::sync::Semaphore>,
    thread: Mutex<Option<JoinHandle<Shutdown>>>,
    shutdown: Mutex<Option<Shutdown>>,
    finished: Arc<(Mutex<bool>, Condvar)>,
    metrics: Arc<super::metrics::Metrics>,
    input_store: Option<crate::offload::input::InputStore>,
}

impl ModelRuntime {
    /// # Safety
    /// The caller retains every prepared graph/context/allocation until close
    /// confirms drain, and grants this runtime exclusive graph mutation access.
    pub unsafe fn start(
        manifest: ExecutionManifest,
        config: ModelConfig,
        queue_capacity: usize,
        event_capacity: usize,
        frontend: Arc<crate::frontend::Frontend>,
        peers: Option<crate::execution::RankPeers>,
        policy: crate::scheduler::PolicyConfig,
    ) -> Result<Self> {
        crate::model::catalog::validate(&config, &manifest)?;
        super::observation::initialize_clock();
        if queue_capacity == 0 || queue_capacity > 1024 || !(2..=4096).contains(&event_capacity) {
            return Err("invalid native request/event queue capacity".into());
        }
        let capacity = Arc::new(tokio::sync::Semaphore::new(
            queue_capacity + config.limits.max_active,
        ));
        let input_store = manifest
            .offload
            .as_ref()
            .and_then(|binding| binding.inputs());
        let worker_input_store = input_store.clone();
        let config = Arc::new(config);
        let (sender, receiver) = mpsc::sync_channel(queue_capacity);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker_health = healthy.clone();
        let worker_config = config.clone();
        let finished = Arc::new((Mutex::new(false), Condvar::new()));
        let worker_finished = finished.clone();
        let mut metrics =
            super::metrics::Metrics::new(
                config.limits.pages,
                config.contract.cache.page_tokens,
                config.decode_weight_bytes,
            );
        metrics.engram = manifest
            .engram
            .as_ref()
            .map(|binding| binding.shared.stats());
        let metrics = Arc::new(metrics);
        metrics.scheduler_trace.configure(&manifest.graphs);
        let worker_metrics = metrics.clone();
        let thread = thread::Builder::new()
            .name("flare-model".into())
            .spawn(move || {
                if matches!(peers, Some(crate::execution::RankPeers::Processes(_)))
                    && manifest.offload.as_ref().is_some_and(|binding| !binding.sharded())
                {
                    let error = Error::execution(
                        ExecutionError::Manifest,
                        "the host prefix store requires every rank in one process",
                    );
                    let _ = ready_tx.send(Err(error.clone()));
                    return Shutdown {
                        drained: true,
                        error: Some(error),
                    };
                }
                // SAFETY: start's caller retains all prepared resources through
                // confirmed drain. This thread exclusively owns the executor;
                // construction binds both streams to the manifest's context.
                let mut executor = match unsafe { RankExecutor::new(manifest, peers) } {
                    Ok(executor) => executor,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.clone()));
                        return Shutdown {
                            drained: true,
                            error: Some(error),
                        };
                    }
                };
                let mut scheduler = match RequestScheduler::new(
                    worker_config.identity,
                    worker_config.limits,
                    crate::model::scheduler_capabilities(&worker_config),
                ) {
                    Ok(scheduler) => scheduler,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.clone()));
                        let mut shutdown = executor.close();
                        shutdown.error = Some(error);
                        return shutdown;
                    }
                };
                if let Err(error) = scheduler
                    .graph_durations(&executor.manifest().graphs)
                    .and_then(|()| scheduler.configure_policy(policy))
                    .and_then(|()| match &executor.manifest().cache {
                        Some(cache) if !cache.snapshot_pages.is_empty() => {
                            scheduler.share_snapshot_pages(&cache.snapshot_pages)
                        }
                        _ => Ok(()),
                    })
                    .and_then(|()| scheduler.enable_speculation(&worker_config.speculative_widths))
                    .and_then(|()| {
                        scheduler.proposal_source(worker_config.speculative_proposal_source)
                    })
                    .and_then(|()| {
                        if worker_config.history_drafts
                            || worker_config.speculative_proposal_source == crate::scheduler::speculation::ProposalSource::Copy
                        {
                            scheduler.enable_history_drafts()
                        } else {
                            Ok(())
                        }
                    })
                    .and_then(|()| {
                        scheduler.fixed_speculative_width(worker_config.speculative_fixed_width)
                    })
                    .and_then(|()| {
                        scheduler.mixed_graphs(
                            &executor
                                .manifest()
                                .graphs
                                .iter()
                                .filter(|g| g.kind == GraphKind::Mixed)
                                .map(|g| (g.batch_capacity, g.token_capacity, g.prefill_capacity))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .and_then(|()| {
                        scheduler.scored_prefill_graphs(&executor.manifest().graphs.iter()
                            .filter(|g| g.kind == GraphKind::ScoredPrefill).map(|g| g.token_capacity).collect::<Vec<_>>())?;
                        scheduler.prefill_graphs(
                            &executor
                                .manifest()
                                .graphs
                                .iter()
                                .filter(|graph| graph.kind == crate::execution::GraphKind::Prefill)
                                .map(|graph| graph.token_capacity)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .and_then(|()| {
                        scheduler.speculative_graphs(
                            &executor
                                .manifest()
                                .graphs
                                .iter()
                                .filter(|graph| {
                                    matches!(
                                        graph.kind,
                                        GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative
                                    )
                                })
                                .flat_map(|graph| {
                                    let first = if graph.packed_token_capacity == 0 { graph.token_capacity } else { 1 };
                                    let last = graph.token_capacity.min(graph.packed_tokens() / graph.batch_capacity);
                                    (first..=last).map(move |width| (graph.batch_capacity, width))
                                })
                                .collect::<std::collections::BTreeSet<_>>()
                                .into_iter().collect::<Vec<_>>(),
                        )
                    })
                    .and_then(|()| {
                        if let Some(width) = executor
                            .manifest()
                            .graphs
                            .iter()
                            .filter(|g| matches!(g.kind, GraphKind::Proposal | GraphKind::PackedSpeculative))
                            .map(|g| g.token_capacity)
                            .max()
                        {
                            scheduler.split_speculation(width)?;
                        }
                        Ok(())
                    })
                {
                    let _ = ready_tx.send(Err(error.clone()));
                    let mut shutdown = executor.close();
                    shutdown.error = Some(error);
                    return shutdown;
                }
                // The host prefix store opens its transfer threads on every
                // rank's context, which the loader prepared for this process.
                let store = match executor
                    .manifest()
                    .offload
                    .as_ref()
                    .and_then(|binding| binding.take())
                    .map(crate::offload::open)
                    .transpose()
                {
                    Ok(store) => store,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.clone()));
                        let mut shutdown = executor.close();
                        shutdown.error = Some(error);
                        return shutdown;
                    }
                };
                let mut worker = Worker::new(
                    &worker_config,
                    &frontend,
                    &worker_stop,
                    &worker_metrics,
                    scheduler,
                    receiver,
                    queue_capacity,
                );
                worker.attach_store(store);
                if worker.config.contract.vision.enabled() {
                    worker.image_preparation = Some(ImagePreparation::new(worker_input_store));
                }
                worker_metrics.scheduler(worker.scheduler.stats());
                worker_health.store(true, Ordering::Release);
                let _ = ready_tx.send(Ok(()));
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker.run(&mut executor)
                }));
                worker_health.store(false, Ordering::Release);
                let error = match run {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error),
                    Err(_) => Some(Error::execution(
                        ExecutionError::Worker,
                        "native model worker panicked",
                    )),
                };
                let final_error = error.clone().unwrap_or_else(|| {
                    Error::execution(ExecutionError::Closed, "model runtime stopped")
                });
                worker.fail_all(&final_error);
                let mut shutdown = executor.close();
                worker.close_store(&mut shutdown);
                if shutdown.drained {
                    let _ = worker.scheduler.release_after_shutdown(&shutdown);
                    worker_metrics.scheduler(worker.scheduler.stats());
                }
                worker_metrics.pending.store(0, Ordering::Relaxed);
                if shutdown.error.is_none() {
                    shutdown.error = error;
                }
                *worker_finished.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
                worker_finished.1.notify_all();
                shutdown
            })
            .map_err(|error| {
                Error::Execution(ExecutionError::Worker(Failure::caused_by(
                    format!("cannot start native model thread: {error}"),
                    error,
                )))
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender,
                stop,
                healthy,
                config,
                event_capacity,
                input_store,
                capacity,
                thread: Mutex::new(Some(thread)),
                shutdown: Mutex::new(None),
                finished,
                metrics,
            }),
            result => {
                let returned = thread.join().ok().and_then(|shutdown| shutdown.error);
                Err(match result {
                    Ok(Err(error)) => error,
                    Err(disconnected) => super::readiness_failure(
                        "model worker exited before readiness",
                        disconnected,
                        returned,
                    ),
                    Ok(Ok(())) => unreachable!("ready success handled above"),
                })
            }
        }
    }

    pub(super) fn input_store(&self) -> Option<&crate::offload::input::InputStore> {
        self.input_store.as_ref()
    }

    pub fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) && !self.stop.load(Ordering::Acquire)
    }
    pub(super) fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub(super) fn metrics(&self) -> &super::metrics::Metrics {
        &self.metrics
    }
    pub(super) fn metrics_handle(&self) -> Arc<super::metrics::Metrics> {
        self.metrics.clone()
    }
    pub fn wait(&self, timeout: std::time::Duration) -> bool {
        let finished = self.finished.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .finished
            .1
            .wait_timeout_while(finished, timeout, |finished| !*finished);
        self.healthy()
    }
    pub fn generate(&self, prompt: Prompt) -> Result<Generation> {
        self.enqueue(prompt, None)
    }

    pub(super) fn generate_observed(&self, prompt: Prompt, timestamps: bool) -> Result<Generation> {
        self.enqueue(prompt, Some(timestamps))
    }

    fn enqueue(&self, prompt: Prompt, observation: Option<bool>) -> Result<Generation> {
        self.enqueue_many(vec![prompt], observation)
            .map(|mut group| group.remove(0))
    }

    pub fn generate_many(&self, prompts: Vec<Prompt>) -> Result<Vec<Generation>> {
        self.enqueue_many(prompts, None)
    }

    pub(super) fn generate_many_observed(&self, prompts: Vec<Prompt>, timestamps: bool) -> Result<Vec<Generation>> {
        self.enqueue_many(prompts, Some(timestamps))
    }

    fn enqueue_many(
        &self,
        prompts: Vec<Prompt>,
        observation: Option<bool>,
    ) -> Result<Vec<Generation>> {
        if !self.healthy() {
            return Err(Error::admission(
                AdmissionError::Unavailable,
                "prepared model runtime is unavailable",
            ));
        }
        let count = u32::try_from(prompts.len())
            .ok()
            .filter(|n| *n > 0)
            .ok_or("generation group must be nonempty and bounded")?;
        let capacity = Arc::new(
            self.capacity
                .clone()
                .try_acquire_many_owned(count)
                .map_err(|_| {
                    Error::admission(
                        AdmissionError::Capacity,
                        "native generation capacity is full",
                    )
                })?,
        );
        let mut incoming_group = Vec::with_capacity(prompts.len());
        let mut generations = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            prompt
                .sampling
                .validate()
                .map_err(|error| error.admission_context(AdmissionError::Invalid))?;
            super::stops::TokenStops::new(
                prompt.stop_sequences.clone(),
                self.config.contract.vocabulary,
            )?;
            super::stops::validate_strings(
                &prompt.stop_strings,
                !prompt.stop_sequences.is_empty(),
                prompt.grammar.is_some(),
            )?;
            crate::model::admission::validate_prompt(&self.config, &prompt)
                .map_err(|error| error.admission_context(AdmissionError::Invalid))?;
            let (events, receiver) = async_mpsc::channel(self.event_capacity);
            let (admission, admitted) = oneshot::channel();
            let (first_token, first_token_ready) = oneshot::channel();
            let cancelled = Arc::new(AtomicBool::new(false));
            let failure = Arc::new(Mutex::new(None));
            let incoming = Incoming {
                _capacity: Some(capacity.clone()),
                arrival_ns: super::observation::monotonic_ns(),
                publications: Default::default(),
                prompt_score_end: if prompt.prompt_logprobs { prompt.tokens.len() } else { 0 },
                prompt_scored: 0,
                prompt,
                events,
                admission: Mutex::new(Some(admission)),
                first_token: std::cell::Cell::new(Some(first_token)),
                cancelled: cancelled.clone(),
                failure: failure.clone(),
                metrics: self.metrics.clone(),
                observation,
                host_lookup: true,
                readmitted: false,
                blocked: None,
            };
            let enqueue_ns = observation
                .filter(|timestamps| *timestamps)
                .map(|_| super::observation::monotonic_ns());
            incoming_group.push(incoming);
            generations.push(Generation {
                events: receiver,
                admission: admitted,
                first_token: first_token_ready,
                cancelled,
                failure,
                enqueue_ns,
            });
        }
        self.sender
            .try_send(incoming_group)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => {
                    Error::admission(AdmissionError::Capacity, "native request queue is full")
                }
                mpsc::TrySendError::Disconnected(_) => Error::admission(
                    AdmissionError::Unavailable,
                    "native model worker disconnected",
                ),
            })?;
        Ok(generations)
    }
    pub fn close(&self) -> Shutdown {
        // Join serialization precedes the cached result lock; request callbacks never use either.
        let mut thread = self.thread.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(shutdown) = self
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return shutdown;
        }
        self.stop.store(true, Ordering::Release);
        let shutdown = thread
            .take()
            .map(|thread| {
                thread.join().unwrap_or_else(|_| Shutdown {
                    drained: false,
                    error: Some(Error::execution(
                        ExecutionError::Worker,
                        "model thread panicked outside its drain boundary",
                    )),
                })
            })
            .unwrap_or(Shutdown {
                drained: false,
                error: Some(Error::execution(
                    ExecutionError::Closed,
                    "missing model worker shutdown state",
                )),
            });
        *self.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for ModelRuntime {
    fn drop(&mut self) {
        self.close();
    }
}

type ImagePreparationJob = (
    RequestHandle,
    crate::frontend::images::PreparedImageInput,
    Arc<AtomicBool>,
);
type ImagePreparationResult = (
    RequestHandle,
    Result<Option<crate::frontend::images::ProcessedImages>>,
);

struct ImagePreparation {
    sender: Option<mpsc::SyncSender<ImagePreparationJob>>,
    receiver: Option<mpsc::Receiver<ImagePreparationResult>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ImagePreparation {
    fn new(store: Option<crate::offload::input::InputStore>) -> Self {
        let (sender, jobs) = mpsc::sync_channel::<ImagePreparationJob>(1);
        let (results, receiver) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let thread = thread::spawn(move || {
            while let Ok((handle, image, cancelled)) = jobs.recv() {
                if worker_stop.load(Ordering::Acquire) {
                    break;
                }
                let prepared = if cancelled.load(Ordering::Acquire) {
                    Ok(None)
                } else {
                    image.materialize_cached(store.as_ref()).map(Some)
                };
                if results.send((handle, prepared)).is_err() {
                    break;
                }
            }
        });
        Self {
            sender: Some(sender),
            receiver: Some(receiver),
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for ImagePreparation {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.sender.take();
        self.receiver.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Worker<'a> {
    config: &'a ModelConfig,
    frontend: &'a crate::frontend::Frontend,
    stop: &'a AtomicBool,
    metrics: &'a super::metrics::Metrics,
    receiver: mpsc::Receiver<Vec<Incoming>>,
    scheduler: RequestScheduler,
    pending: VecDeque<Incoming>,
    /// Preempted requests, readmitted before new ones.
    parked: VecDeque<(Incoming, Parked<'a>)>,
    active: Vec<Option<Active<'a>>>,
    admissions: Vec<Option<crate::model::Admission>>,
    inputs: Option<crate::model::Inputs>,
    queue_capacity: usize,
    store: Option<Box<dyn crate::offload::PrefixStore>>,
    /// Prefill tokens per second measured from completed prefill frames, which
    /// the host-store fetch policy compares with transfer time.
    prefill_rate: f64,
    image_preparation: Option<ImagePreparation>,
    preparing_image: Option<RequestHandle>,
    image_preparation_cursor: usize,
    prefill_started: VecDeque<(crate::cache::Fence, std::time::Instant, u64)>,
}

impl<'a> Worker<'a> {
    fn new(
        config: &'a ModelConfig,
        frontend: &'a crate::frontend::Frontend,
        stop: &'a AtomicBool,
        metrics: &'a super::metrics::Metrics,
        scheduler: RequestScheduler,
        receiver: mpsc::Receiver<Vec<Incoming>>,
        queue_capacity: usize,
    ) -> Self {
        Self {
            config,
            frontend,
            stop,
            metrics,
            scheduler,
            receiver,
            queue_capacity,
            store: None,
            pending: VecDeque::with_capacity(queue_capacity),
            parked: VecDeque::new(),
            active: (0..config.limits.max_active).map(|_| None).collect(),
            admissions: (0..config.limits.max_active).map(|_| None).collect(),
            inputs: None,
            prefill_rate: crate::offload::INITIAL_PREFILL_RATE,
            prefill_started: VecDeque::new(),
            image_preparation: None,
            preparing_image: None,
            image_preparation_cursor: 0,
        }
    }

    fn run(&mut self, executor: &mut RankExecutor) -> Result<()> {
        // Input construction stays inside the worker's unwind and drain boundary.
        self.inputs = Some(crate::model::Inputs::new(self.config, executor.manifest())?);
        while !self.stop.load(Ordering::Acquire) {
            let std::ops::ControlFlow::Continue(mut progress) = self.admit()? else {
                return Ok(());
            };
            loop {
                let completion = match executor.poll() {
                    Ok(None) => break,
                    Ok(Some(completion)) => completion,
                    Err(error) => return Err(error),
                };
                let kind = executor.manifest().graphs[completion.graph_index].kind;
                self.complete(kind, completion)?;
                progress = true;
            }
            progress |= self.publish()?;
            progress |= self.service_store()?;
            let (step, planned) = self.plan(executor.outstanding())?;
            progress |= planned;
            if let Some(step) = step {
                progress |= self.submit(executor, &step)?;
            }
            if progress {
                self.metrics.scheduler(self.scheduler.stats());
                self.metrics
                    .pending
                    .store(self.pending.len() as u64, Ordering::Relaxed);
            }
            self.metrics.scheduler_trace.details(|| {
                use super::metrics::scheduler::{RequestDetail, SchedulerDetail};
                SchedulerDetail {
                    at_ns: super::observation::monotonic_ns(),
                    outstanding: executor.outstanding(),
                    requests: std::array::from_fn(|slot| {
                        let active = self.active.get(slot)?.as_ref()?;
                        let live = self.scheduler.request(active.request).ok()?;
                        let accounting = live.accounting();
                        Some(RequestDetail {
                            handle: live.handle,
                            phase: live.phase,
                            consumed: live.consumed_tokens,
                            prompt: live.prompt_tokens,
                            generated: live.generated_tokens,
                            last_progress_ns: accounting.last_progress_ns,
                            last_output_ns: accounting.last_output_ns,
                            published_deliveries: accounting.published_deliveries,
                        })
                    }),
                }
            });
            if !progress {
                if executor.outstanding() > 0 {
                    std::hint::spin_loop();
                    thread::yield_now();
                } else {
                    self.scheduler.idle()?;
                    match self
                        .receiver
                        .recv_timeout(std::time::Duration::from_millis(2))
                    {
                        Ok(request) => self.pending.extend(request),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                    }
                }
            }
        }
        Ok(())
    }

    fn attach_store(&mut self, store: Option<Box<dyn crate::offload::PrefixStore>>) {
        if store.is_some() {
            self.scheduler.enable_publications();
        }
        self.store = store;
    }

    fn close_store(&mut self, shutdown: &mut Shutdown) {
        let Some(store) = self.store.take() else {
            return;
        };
        // Retain registered slabs if either drain fails, including an unwind
        // inside the store's close operation.
        let mut store = std::mem::ManuallyDrop::new(store);
        if !shutdown.drained {
            return;
        }
        shutdown.drained = false;
        match store.close() {
            Ok(()) => {
                shutdown.drained = true;
                drop(std::mem::ManuallyDrop::into_inner(store));
            }
            Err(error) => {
                shutdown.error.get_or_insert(error);
            }
        }
    }

    fn complete_image_preparation(&mut self) -> Result<bool> {
        let mut progress = false;
        loop {
            let Some(worker) = self.image_preparation.as_ref() else {
                break;
            };
            let result = match worker.receiver.as_ref().unwrap().try_recv() {
                Ok(result) => result,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err("image preparation worker disconnected".into());
                }
            };
            let (handle, prepared) = result;
            let slot = handle.slot as usize;
            let Some(active) = self.active[slot]
                .as_mut()
                .filter(|active| active.request == handle)
            else {
                self.preparing_image = None;
                continue;
            };
            match prepared {
                Ok(Some(images)) => {
                    active.prepared_image = Some(images);
                    self.scheduler.conditioned_inputs_ready(handle, true)?;
                }
                Ok(None) => {
                    self.preparing_image = None;
                }
                Err(error) => {
                    self.preparing_image = None;
                    self.scheduler.cancel(handle)?;
                    let active = self.active[slot].take().unwrap();
                    self.admissions[slot] = None;
                    active.incoming.fail(error);
                }
            }
            progress = true;
        }
        Ok(progress)
    }

    fn admit(&mut self) -> Result<std::ops::ControlFlow<(), bool>> {
        self.scheduler
            .advance_to(super::observation::monotonic_ns())?;
        let mut progress = self.complete_image_preparation()?;
        self.pending.retain(|request| {
            let cancelled = request.cancelled.load(Ordering::Acquire);
            if cancelled {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                progress = true;
            }
            !cancelled
        });
        while self.pending.len() < self.queue_capacity {
            match self.receiver.try_recv() {
                Ok(request) => {
                    self.pending.extend(request);
                    progress = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Ok(std::ops::ControlFlow::Break(()));
                }
            }
        }
        self.parked.retain(|(incoming, _)| {
            let cancelled = incoming.cancelled.load(Ordering::Acquire);
            if cancelled {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                progress = true;
            }
            !cancelled
        });
        for slot in 0..self.active.len() {
            if self.active[slot]
                .as_ref()
                .is_some_and(|a| a.incoming.cancelled.load(Ordering::Acquire))
            {
                let active = self.active[slot].take().unwrap();
                if active.prepared_image.is_some() {
                    self.preparing_image = None;
                }
                self.scheduler.cancel(active.request)?;
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                self.admissions[slot] = None;
                progress = true;
            }
        }
        while self.scheduler.has_available_slot() {
            let stats = self.scheduler.stats();
            let resources = (
                self.scheduler.completed(),
                stats.free_pages,
                stats.free_snapshots,
                stats.free_slots,
            );
            let eligible = |incoming: &Incoming| incoming.blocked != Some(resources);
            let parked = self
                .parked
                .iter()
                .position(|(incoming, _)| eligible(incoming));
            let (mut incoming, parked, position) = match parked {
                Some(index) => {
                    let (incoming, parked) = self.parked.remove(index).unwrap();
                    (incoming, Some(parked), index)
                }
                None => {
                    let Some(index) = self.pending.iter().position(eligible) else {
                        break;
                    };
                    (self.pending.remove(index).unwrap(), None, index)
                }
            };
            if incoming.cancelled.load(Ordering::Acquire) {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if incoming.prompt.logprobs && self.config.max_top_logprobs == 0 {
                incoming.reject_admission("logprobs are not available in this model package".into());
                continue;
            }
            let spec = RequestSpec {
                message_boundaries: incoming.prompt.message_boundaries.clone(),
                prompt: incoming.prompt.tokens.clone(),
                conditioning: incoming.prompt.conditioning.clone(),
                max_new_tokens: incoming.prompt.max_new_tokens,
                grammar_constrained: incoming.prompt.grammar.is_some(),
                host_stops: incoming.prompt.has_host_stops(),
                prompt_logprobs: incoming.prompt_scored < incoming.prompt_score_end,
                growable_output: true,
            };
            self.scheduler
                .advance_to(super::observation::monotonic_ns())?;
            let admission = if let Some(store) =
                self.store.as_mut().filter(|_| incoming.host_lookup && !spec.prompt_logprobs)
            {
                admit_with_host_store(&mut self.scheduler, store.as_mut(), spec, self.prefill_rate)
            } else {
                self.scheduler.admit(spec)
            };
            let handle = match admission {
                Ok(handle) => handle,
                // Pages are short: wait for running requests to finish or yield theirs.
                Err(Error::Admission(AdmissionError::Capacity(_))) => {
                    incoming.blocked = Some(resources);
                    match parked {
                        Some(parked) => self.parked.insert(position, (incoming, parked)),
                        None => self.pending.insert(position, incoming),
                    }
                    continue;
                }
                Err(error) => {
                    incoming.reject_admission(error);
                    continue;
                }
            };
            incoming.blocked = None;
            self.scheduler.set_arrival(handle, incoming.arrival_ns)?;
            let slot = handle.slot as usize;
            if let Err(error) = self
                .scheduler
                .set_prefill_spans(handle, &incoming.prompt.data.spans)
            {
                self.scheduler.cancel(handle)?;
                incoming.fail(error);
                continue;
            }
            let consumed = self.scheduler.request(handle)?.consumed_tokens as usize;
            for image in incoming
                .prompt
                .conditioned_inputs
                .iter()
                .filter(|image| image.token_span.end <= consumed)
            {
                image.release_decoded();
            }
            let selected_images: Vec<_> = incoming
                .prompt
                .conditioned_inputs
                .iter()
                .filter(|input| input.token_span.end > consumed)
                .cloned()
                .collect();
            let inputs = selected_images
                .iter()
                .map(|image| crate::scheduler::ConditionedInput {
                    token_span: image.token_span.clone(),
                    feature_range: image.feature_range.clone(),
                    patches: image.geometry.patches,
                })
                .collect::<Vec<_>>();
            if let Err(error) = self.scheduler.set_conditioned_inputs(handle, &inputs) {
                self.scheduler.cancel(handle)?;
                if matches!(error, Error::Admission(AdmissionError::Capacity(_))) {
                    incoming.blocked = Some(resources);
                    match parked {
                        Some(parked) => self.parked.insert(position, (incoming, parked)),
                        None => self.pending.insert(position, incoming),
                    }
                } else {
                    incoming.fail(error);
                }
                continue;
            }
            self.admissions[slot] = Some(crate::model::Admission::from_prompt(
                self.config,
                handle,
                &incoming.prompt,
                incoming.prompt.ignore_eos,
            )?);
            let parked = match parked {
                Some(parked) => parked,
                None => Parked {
                    stops: super::stops::TokenStops::new(
                        incoming.prompt.stop_sequences.clone(),
                        self.config.contract.vocabulary,
                    )?,
                    text_stops: if incoming.prompt.stop_strings.is_empty() {
                        None
                    } else {
                        let mut stops = super::stops::TextStops::new(self.frontend, incoming.prompt.stop_strings.clone())?;
                        if incoming.prompt.logprobs {
                            stops.track(self.frontend);
                        }
                        Some(stops)
                    },
                    last_token: None,
                    carried: Carried::default(),
                    held_logprobs: VecDeque::new(),
                },
            };
            if !incoming.readmitted {
                self.metrics.admitted.fetch_add(1, Ordering::Relaxed);
            }
            self.scheduler.resume_decoded_sequence(handle, parked.carried.decoded_sequence)?;
            incoming.notify_admission(Ok(()));
            self.active[slot] = Some(Active {
                request: handle,
                stops: parked.stops,
                text_stops: parked.text_stops,
                incoming,
                last_token: parked.last_token,
                next_image: 0,
                selected_images,
                prepared_image: None,
                carried: parked.carried,
                held_logprobs: parked.held_logprobs,
            });
            progress = true;
        }
        if self.preparing_image.is_none() {
            if let Some(preparation) = &self.image_preparation {
                for offset in 0..self.active.len() {
                    let slot = (self.image_preparation_cursor + offset) % self.active.len();
                    let Some(active) = &self.active[slot] else {
                        continue;
                    };
                    if self.scheduler.request(active.request)?.phase == RequestPhase::Inbound {
                        continue;
                    }
                    if active.next_image
                        >= self
                            .scheduler
                            .request(active.request)?
                            .conditioned_input_end()
                    {
                        continue;
                    }
                    let Some(image) = active.selected_images.get(active.next_image) else {
                        continue;
                    };
                    let job = (
                        active.request,
                        image.clone(),
                        active.incoming.cancelled.clone(),
                    );
                    match preparation.sender.as_ref().unwrap().try_send(job) {
                        Ok(()) => {
                            self.preparing_image = Some(active.request);
                            self.image_preparation_cursor = (slot + 1) % self.active.len();
                            progress = true;
                            break;
                        }
                        Err(mpsc::TrySendError::Full(_)) => break,
                        Err(mpsc::TrySendError::Disconnected(_)) => {
                            return Err("image preparation worker disconnected".into());
                        }
                    }
                }
            }
        }
        Ok(std::ops::ControlFlow::Continue(progress))
    }

    fn complete(
        &mut self,
        kind: GraphKind,
        mut completion: crate::execution::FrameCompletion,
    ) -> Result<()> {
        self.scheduler
            .advance_to(super::observation::monotonic_ns())?;
        let measured_step = self.scheduler.submitted_step(completion.fence)?.clone();
        let selection = self.scheduler.submitted_timing(completion.fence)?;
        if measured_step.kind != kind {
            return Err("completion graph kind differs from submitted step".into());
        }
        for (row, lane) in measured_step.lanes.iter().enumerate() {
            let record = &completion.records[lane.request.slot as usize];
            if record.status != 0 {
                crate::model::validate_completion(
                    self.config,
                    self.inputs
                        .as_ref()
                        .unwrap()
                        .completion_contract(&measured_step)?,
                    measured_step.lane_kind(row),
                    record,
                )?;
            }
        }
        let mut prefill = 0;
        let mut accepted = 0;
        let mut draft_rounds = 0;
        let mut verification_drafts = 0;
        let mut verification_rounds = 0;
        if matches!(
            kind,
            GraphKind::Prefill
                | GraphKind::ScoredPrefill
                | GraphKind::Decode
                | GraphKind::Speculative
                | GraphKind::Verification
                | GraphKind::PackedSpeculative
                | GraphKind::Mixed
        ) {
            let step = self.scheduler.submitted_step(completion.fence)?;
            for (row, lane) in step.lanes.iter().enumerate() {
                let kind = step.lane_kind(row);
                let Ok(live) = self.scheduler.request(lane.request) else {
                    continue;
                };
                let record = &completion.records[lane.request.slot as usize];
                if !matches!(record.status, 1 | 2) {
                    continue;
                }
                if kind.is_prefill() {
                    prefill += (record.consumed_tokens.max(0) as u64)
                        .saturating_sub(live.consumed_tokens)
                        .min(lane.tokens as u64);
                } else if record.generated_tokens > live.generated_tokens as i64 {
                    verification_rounds += 1;
                    if let Some(vector) = record.vector {
                        let offered =
                            (vector.offered_drafts as usize).min(lane.tokens.saturating_sub(1));
                        verification_drafts += offered as u64;
                        if offered > 0 {
                            accepted += (vector.accepted_drafts as usize).min(offered) as u64;
                            draft_rounds += 1;
                        }
                    }
                }
            }
        }
        let generated = if kind == GraphKind::Proposal {
            let confidence = completion
                .proposal_confidence
                .as_ref()
                .ok_or("proposal completion lacks confidence")?;
            self.scheduler.complete_proposal(
                completion.fence,
                &completion.records,
                completion.timing,
                &confidence[..measured_step.batch_capacity],
            )?
        } else if kind == GraphKind::PackedSpeculative {
            let confidence = completion.proposal_confidence.as_ref()
                .ok_or("packed combined completion lacks confidence")?;
            self.scheduler.complete_packed_round_with_survival(completion.fence, &completion.records,
                completion.timing, &confidence[..measured_step.batch_capacity],
                completion.speculative_survival.as_ref().map(|rows| rows.as_slice()))?
        } else {
            self.scheduler
                .complete_with_survival(completion.fence, &completion.records, completion.timing,
                    completion.speculative_survival.as_ref().map(|rows| rows.as_slice()))?
        };
        self.metrics
            .progress(prefill, generated.len(), accepted, draft_rounds,
                verification_drafts, verification_rounds);
        self.metrics.replay_completed(completion.timing);
        if prefill > 0 && self.store.is_some() {
            self.observe_prefill_rate(completion.fence, prefill);
        }
        self.metrics.scheduler_trace.completed(
            super::observation::monotonic_ns(),
            completion.fence,
            completion.graph_index,
            &measured_step,
            completion.timing,
            selection,
            if matches!(selection.origin, crate::scheduler::TimingOrigin::Acceptance) {
                self.scheduler.acceptance_credit_ns()
            } else { self.scheduler.refresh_credit_ns() },
        );
        if kind == GraphKind::Proposal {
            self.metrics.proposal_completed(measured_step.batch_capacity, measured_step.token_capacity,
                completion.timing.gpu_ns.into_iter().flatten().max().unwrap_or(0));
        }
        if matches!(kind, GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative) {
            let nanos = completion
                .timing
                .gpu_ns
                .into_iter()
                .flatten()
                .max()
                .unwrap_or(0);
            let step = &measured_step;
            if kind == GraphKind::PackedSpeculative {
                self.metrics.packed_speculative_completed(step.batch_capacity, step.token_capacity,
                    step.draft_tokens, nanos, &completion.records);
            } else { self.metrics.speculative_completed(
                kind,
                step.batch_capacity,
                step.token_capacity,
                nanos,
                &completion.records,
            ); }
        }
        if kind == GraphKind::ScoredPrefill {
            let lane = &measured_step.lanes[0];
            let slot = lane.request.slot as usize;
            if completion.records[slot].status != 3
                && let Some(request) = self.active[slot].as_mut().filter(|r| r.request == lane.request)
            {
                let prompt = &request.incoming.prompt;
                let range = request.incoming.prompt_score_range(lane.start_position as usize, lane.tokens);
                let (first, end) = (range.start, range.end);
                let payload = completion.prompt_logprobs.as_ref().ok_or("scored prefill lacks results")?;
                let values = (first..end).map(|position| {
                    if position == 0 || prompt.data.is_image_position(position) {
                        return Ok(None);
                    }
                    let value = payload.get(position - lane.start_position as usize - 1)
                        .and_then(Option::as_ref).ok_or("scored prefill omitted a prompt token")?;
                    if value.value.token != prompt.tokens[position] || value.rank as usize > self.config.contract.vocabulary
                        || value.value.top.iter().any(|(t, _)| *t as usize >= self.config.contract.vocabulary) {
                        return Err(crate::Error::from("scored prefill returned an invalid token or rank"));
                    }
                    Ok(Some(value.clone()))
                }).collect::<Result<Vec<_>>>()?;
                if first < end {
                    request.incoming.prompt_scored = end;
                    if request.incoming.events.try_send(GenerationEvent::PromptLogprobs {
                        start: first, tokens: prompt.tokens[first..end].to_vec(), values,
                    }).is_err() {
                        request.incoming.cancelled.store(true, Ordering::Release);
                    }
                }
            }
        }
        let mut stopped = Vec::new();
        let mut positions = [0usize; crate::execution::REQUEST_LANES];
        for token in generated {
            let slot = token.request.slot as usize;
            let position = positions[slot];
            positions[slot] += 1;
            if stopped.contains(&token.request) {
                continue;
            }
            if let Some(request) = self.active[token.request.slot as usize]
                .as_mut()
                .filter(|a| a.request == token.request)
            {
                request.incoming.observe_token(token.token);
                request.last_token = Some(token.token);
                if let Some(grammar) = &mut request.incoming.prompt.grammar {
                    if let Err(error) = grammar.advance(token.token) {
                        request.incoming.fail(
                            ExecutionError::Completion(Failure::caused_by(
                                format!("grammar advancement failed: {error}"),
                                error,
                            ))
                            .into(),
                        );
                        stopped.push(token.request);
                        continue;
                    }
                    self.admissions[token.request.slot as usize]
                        .as_mut()
                        .unwrap()
                        .grammar_updated(grammar.mask());
                }
                let before = request.incoming.publications.get();
                let terminating_eos = !request.incoming.prompt.ignore_eos
                    && self.config.eos_token_ids.contains(&token.token);
                let logprob = if request.incoming.prompt.logprobs && !terminating_eos {
                    match completion.logprobs.as_mut().and_then(|v| v[slot][position].take()) {
                        Some(value) if value.token == token.token
                            && value.top.iter().all(|(id, _)| (*id as usize) < self.config.contract.vocabulary) => Some(value),
                        _ => {
                            request.incoming.fail(Error::execution(
                                ExecutionError::Completion,
                                "generated token is missing its raw logprob record",
                            ));
                            stopped.push(token.request);
                            continue;
                        }
                    }
                } else {
                    None
                };
                let stop = match request.emit_output(token.token, &self.config.eos_token_ids, logprob) {
                    Ok(matched) => matched,
                    Err(error) => {
                        request.incoming.fail(error);
                        stopped.push(token.request);
                        continue;
                    }
                };
                if request.incoming.publications.get() != before {
                    self.scheduler
                        .advance_to(super::observation::monotonic_ns())?;
                    self.scheduler
                        .published_output(token.request, completion.fence)?;
                }
                if stop {
                    self.metrics.completed.fetch_add(1, Ordering::Relaxed);
                    let live = self.scheduler.request(token.request)?;
                    if request
                        .incoming
                        .events
                        .try_send(GenerationEvent::Finished {
                            prompt_tokens: request
                                .carried
                                .prompt_tokens
                                .unwrap_or(live.prompt_tokens),
                            generated_tokens: request.carried.generated_tokens
                                + live.generated_tokens,
                            reused_prompt_tokens: request
                                .carried
                                .reused_prompt_tokens
                                .unwrap_or(live.reused_prompt_tokens()),
                            reason: "stop",
                            termination: crate::output::Termination::UserStop,
                            stop_sequence: request
                                .text_stops
                                .as_ref()
                                .and_then(|s| s.matched().map(str::to_owned)),
                        })
                        .is_err()
                    {
                        request
                            .incoming
                            .fail("generation consumer exceeded the bounded output queue".into());
                    }
                    stopped.push(token.request);
                }
            }
        }
        self.retire(&stopped)?;
        Ok(())
    }

    fn publish(&mut self) -> Result<bool> {
        self.scheduler
            .advance_to(super::observation::monotonic_ns())?;
        let mut progress = false;
        for slot in 0..self.active.len() {
            if let Some(active) = &mut self.active[slot] {
                if self
                    .scheduler
                    .request(active.request)?
                    .conditioned_window_ready()
                {
                    active.request = self.scheduler.resume_conditioned_window(active.request)?;
                    self.admissions[slot]
                        .as_mut()
                        .ok_or("conditioned window lacks admission")?
                        .handle = active.request;
                    progress = true;
                }
            }
            if self.park_if_preempted(slot)? {
                progress = true;
                continue;
            }
            if let Some(request) = &mut self.active[slot] {
                let live = self.scheduler.request(request.request)?;
                if live.phase == RequestPhase::Finishing
                    && live.last_use() <= self.scheduler.completed()
                {
                    if let Some(error) = live.error_code {
                        request
                            .incoming
                            .fail(ExecutionError::Device { code: error }.into());
                    } else {
                        let user_stop = if let Some(stops) = &mut request.text_stops {
                            let eos = request
                                .last_token
                                .filter(|token| !request.incoming.prompt.ignore_eos && self.config.eos_token_ids.contains(token));
                            match stops.finish(eos) {
                                Ok((pieces, matched)) => {
                                    if !pieces.is_empty() {
                                        request.incoming.emit(GenerationEvent::Protocol(pieces));
                                    }
                                    matched
                                }
                                Err(error) => {
                                    request.incoming.fail(error);
                                    self.scheduler.cancel(request.request)?;
                                    self.active[slot] = None;
                                    self.admissions[slot] = None;
                                    progress = true;
                                    continue;
                                }
                            }
                        } else {
                            false
                        };
                        self.metrics.completed.fetch_add(1, Ordering::Relaxed);
                        for token in request.stops.finish() {
                            if request
                                .incoming
                                .events
                                .try_send(match request.held_logprobs.pop_front().flatten() {
                                    Some(value) => GenerationEvent::LogprobToken(value),
                                    None => GenerationEvent::Token(token),
                                })
                                .is_err()
                            {
                                request.incoming.fail(
                                    "generation consumer exceeded the bounded output queue".into(),
                                );
                            }
                        }
                        let reason = if user_stop
                            || request
                                .last_token
                                .is_some_and(|token| self.config.eos_token_ids.contains(&token))
                        {
                            "stop"
                        } else {
                            "length"
                        };
                        if request
                            .incoming
                            .events
                            .try_send(GenerationEvent::Finished {
                                prompt_tokens: request
                                    .carried
                                    .prompt_tokens
                                    .unwrap_or(live.prompt_tokens),
                                generated_tokens: request.carried.generated_tokens
                                    + live.generated_tokens,
                                reused_prompt_tokens: request
                                    .carried
                                    .reused_prompt_tokens
                                    .unwrap_or(live.reused_prompt_tokens()),
                                reason,
                                stop_sequence: request
                                    .text_stops
                                    .as_ref()
                                    .and_then(|s| s.matched().map(str::to_owned)),
                                termination: if user_stop {
                                    crate::output::Termination::UserStop
                                } else if reason == "stop" {
                                    crate::output::Termination::Eos
                                } else {
                                    crate::output::Termination::OutputLimit
                                },
                            })
                            .is_err()
                        {
                            request.incoming.fail(
                                "generation consumer exceeded the bounded output queue".into(),
                            );
                        }
                        if user_stop {
                            self.scheduler.cancel(request.request)?;
                            self.active[slot] = None;
                            self.admissions[slot] = None;
                            progress = true;
                            continue;
                        }
                    }
                    self.scheduler.finish(request.request)?;
                    self.active[slot] = None;
                    self.admissions[slot] = None;
                    progress = true;
                }
            }
        }
        Ok(progress)
    }

    /// Parks a preempted request once its terminal checkpoint is published: its history
    /// becomes the prompt, its output limit drops by what it generated, and its output
    /// state and counters carry over to the next admission.
    fn park_if_preempted(&mut self, slot: usize) -> Result<bool> {
        let Some(handle) = self.active[slot].as_ref().map(|active| active.request) else {
            return Ok(false);
        };
        let live = self.scheduler.request(handle)?;
        if !live.preempted()
            || live.phase != RequestPhase::Finishing
            || live.error_code.is_some()
            || live.last_use() > self.scheduler.completed()
        {
            return Ok(false);
        }
        let tokens = live.tokens().to_vec();
        let decoded_sequence = live.decoded_sequence;
        let (generated, prompt_tokens, reused) = (
            live.generated_tokens,
            live.prompt_tokens,
            live.reused_prompt_tokens(),
        );
        self.scheduler.finish(handle)?;
        let active = self.active[slot].take().unwrap();
        self.admissions[slot] = None;
        let mut incoming = active.incoming;
        crate::model::prompt_metadata(&mut incoming.prompt.data)
            .extend_text(incoming.prompt.tokens.len(), tokens.len())?;
        incoming.prompt.tokens = tokens;
        incoming.prompt.max_new_tokens -= generated;
        incoming.readmitted = true;
        let seed = active.carried.seed.unwrap_or(incoming.prompt.sampling.seed);
        let generated_tokens = active.carried.generated_tokens + generated;
        incoming.prompt.sampling.seed = continuation_seed(seed, generated_tokens);
        let parked = Parked {
            last_token: active.last_token,
            stops: active.stops,
            text_stops: active.text_stops,
            held_logprobs: active.held_logprobs,
            carried: Carried {
                decoded_sequence,
                prompt_tokens: Some(active.carried.prompt_tokens.unwrap_or(prompt_tokens)),
                reused_prompt_tokens: Some(active.carried.reused_prompt_tokens.unwrap_or(reused)),
                generated_tokens,
                seed: Some(seed),
            },
        };
        self.parked.push_back((incoming, parked));
        Ok(true)
    }

    fn retire(&mut self, stopped: &[RequestHandle]) -> Result<()> {
        cancel_stopped(
            stopped,
            &mut self.scheduler,
            &mut self.active,
            &mut self.admissions,
        )
    }

    fn plan(&mut self, outstanding: usize) -> Result<(Option<ScheduledPlan>, bool)> {
        self.scheduler
            .advance_to(super::observation::monotonic_ns())?;
        let mut progress = false;
        if self.config.speculative_proposal_source
            == crate::scheduler::speculation::ProposalSource::Copy
            && outstanding == 0
        {
            for slot in 0..self.active.len() {
                let Some(request) = self.active[slot].as_ref() else {
                    continue;
                };
                let live = self.scheduler.request(request.request)?;
                if live.phase != RequestPhase::Decode || live.copy_proposal().is_some() {
                    continue;
                }
                let proposal = prepare_copy_proposal(
                    self.scheduler.history_proposal(live.tokens(), if live.host_stops() { 0 } else { 7 }),
                    live.consumed_tokens,
                    request.incoming.prompt.grammar.as_ref(),
                    &self.config.eos_token_ids,
                );
                match proposal {
                    Ok(proposal) => self
                        .scheduler
                        .set_copy_proposal(request.request, proposal)?,
                    Err(error) => {
                        request.incoming.fail(
                            ExecutionError::State(Failure::caused_by(
                                format!("copy grammar preparation failed: {error}"),
                                error,
                            ))
                            .into(),
                        );
                        self.scheduler.cancel(request.request)?;
                        self.active[slot] = None;
                        self.admissions[slot] = None;
                        progress = true;
                    }
                }
            }
        }
        Ok((self.scheduler.plan(), progress))
    }

    fn reserve_and_pack(&mut self, step: &ScheduledPlan, fence: crate::cache::Fence) -> Result<()> {
        self.scheduler.commit(step, fence)?;
        let inputs = self
            .inputs
            .as_mut()
            .expect("worker inputs initialized before submission");
        if step.kind == GraphKind::Vision {
            let request = self.active[step.lanes[0].request.slot as usize]
                .as_mut()
                .ok_or("missing active vision request")?;
            inputs.pack_vision(
                step,
                self.config,
                request.request,
                self.scheduler.request(request.request)?.image_slot
                    .ok_or("vision request lacks an image feature slot")?,
                request
                    .prepared_image
                    .as_ref()
                    .ok_or("missing selected vision inputs")?,
                0,
                usize::try_from(
                    self.scheduler.request(request.request)?.feature_index(
                        i32::try_from(
                            request.selected_images[request.next_image]
                                .feature_range
                                .start,
                        )
                        .map_err(|_| "conditioned feature index exceeds integer range")?,
                    )?,
                )
                .map_err(|_| "conditioned feature destination is negative")?,
            )?;
            request.prepared_image = None;
            request.next_image += 1;
            self.preparing_image = None;
            self.scheduler
                .conditioned_inputs_ready(request.request, false)?;
        } else {
            crate::model::pack_inputs(
                inputs,
                self.config,
                &self.scheduler,
                step,
                &self.admissions,
            )?;
        }
        Ok(())
    }

    fn submit(&mut self, executor: &mut RankExecutor, step: &ScheduledPlan) -> Result<bool> {
        self.scheduler
            .advance_to(super::observation::monotonic_ns())?;
        if !executor.can_submit(self.scheduler.plan_requires_feedback(step))
            || !self
                .scheduler
                .submission_window_available(step.estimated_duration_ns())
        {
            return Ok(false);
        }
        validate_sampling_admissions(step, &self.admissions)?;
        let graph = crate::model::catalog::select_shape_with_drafts(
            self.config,
            executor.manifest(),
            step.kind,
            step.batch_capacity,
            step.token_capacity,
            step.prefill_capacity(),
            step.packed_token_capacity,
            step.draft_tokens,
            SamplingMode::General,
        )?;
        let fence = executor.next_fence()?;
        self.reserve_and_pack(step, fence)?;
        let feedback = self.scheduler.submitted_step(fence)?.requires_feedback;
        if step
            .lanes
            .iter()
            .enumerate()
            .any(|(row, _)| step.lane_kind(row).is_prefill())
            && self.store.is_some()
        {
            if self.prefill_started.len() >= 8 {
                self.prefill_started.pop_front();
            }
            self.prefill_started.push_back((
                fence,
                std::time::Instant::now(),
                step.lanes
                    .iter()
                    .enumerate()
                    .filter(|(row, _)| step.lane_kind(*row).is_prefill())
                    .map(|(_, lane)| lane.tokens as u64)
                    .sum(),
            ));
        }
        let mut flags = [0u8; crate::execution::REQUEST_LANES * size_of::<i32>()];
        let mut counts = [0u8; crate::execution::REQUEST_LANES * size_of::<i32>()];
        let mut selections = Vec::new();
        for lane in &step.lanes {
            let slot = lane.request.slot as usize;
            if let Some(active) = &self.active[slot] {
                let start = slot * size_of::<i32>();
                let end = start + size_of::<i32>();
                if active.incoming.prompt.logprobs {
                    flags[start..end].copy_from_slice(&1i32.to_ne_bytes());
                }
                let ids = &active.incoming.prompt.logprob_token_ids;
                counts[start..end].copy_from_slice(&(ids.len() as i32).to_ne_bytes());
                if !ids.is_empty() {
                    selections.push((
                        slot,
                        ids.iter()
                            .flat_map(|id| id.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    ));
                }
            }
        }
        let mut uploads = crate::model::Inputs::uploads(
            self.inputs
                .as_ref()
                .expect("worker inputs initialized before submission"),
        );
        if self.config.max_top_logprobs != 0 {
            uploads.push(crate::execution::Upload {
                buffer: "logprob_requested",
                offset: 0,
                bytes: &flags[..self.config.limits.request_capacity() * size_of::<i32>()],
            });
        }
        if self.config.logprob_token_id_capacity != 0 {
            uploads.push(crate::execution::Upload {
                buffer: "logprob_token_id_counts",
                offset: 0,
                bytes: &counts[..self.config.limits.request_capacity() * size_of::<i32>()],
            });
            for (slot, bytes) in &selections {
                uploads.push(crate::execution::Upload {
                    buffer: "logprob_token_ids",
                    offset: slot * crate::execution::MAX_LOGPROB_TOKEN_IDS * size_of::<i32>(),
                    bytes,
                });
            }
        }
        if executor.submit(
            graph,
            &uploads,
            feedback,
        )? != Some(fence)
        {
            return Err("executor refused a reserved scheduler submission".into());
        }
        observe_submission(self.metrics, &self.scheduler, step, executor.outstanding())?;
        self.metrics
            .scheduler_trace
            .submitted(super::observation::monotonic_ns(), graph, step, feedback);
        Ok(true)
    }

    /// Copy newly ready checkpoints out, apply finished host transfers, and
    /// re-admit the waiters of a failed restore without the host store.
    fn service_store(&mut self) -> Result<bool> {
        self.scheduler
            .advance_to(super::observation::monotonic_ns())?;
        let Some(store) = self.store.as_mut() else {
            return Ok(false);
        };
        let mut progress = false;
        for publication in self.scheduler.take_publications() {
            progress = true;
            // The checkpoint may have been evicted between readiness and this drain.
            if self.scheduler.pin_checkpoint(publication.id).is_err() {
                continue;
            }
            if !matches!(store.store(&publication), Ok(true)) {
                self.scheduler.unpin_checkpoint(publication.id)?;
            }
        }
        let mut events = Vec::new();
        store.poll(&mut events)?;
        let mut requeued = Vec::new();
        for event in events {
            progress = true;
            match event {
                crate::offload::StoreEvent::Stored { checkpoint, .. } => {
                    // The pin taken before `store` is released; a checkpoint retired
                    // meanwhile has no pin left and is ignored.
                    let _ = self.scheduler.unpin_checkpoint(checkpoint);
                }
                crate::offload::StoreEvent::Fetched { key, result } => {
                    requeued.extend(self.scheduler.finish_inbound(key, result.is_ok())?);
                }
            }
        }
        // Store gauges change without events, so they are refreshed every pass.
        self.metrics.store(store.stats());
        for handle in requeued {
            self.requeue(handle);
        }
        Ok(progress)
    }

    /// Return a request whose restore failed to the front of the queue; its next
    /// admission skips the host store.
    fn requeue(&mut self, handle: RequestHandle) {
        let slot = handle.slot as usize;
        let Some(active) = self.active[slot].take_if(|active| active.request == handle) else {
            return;
        };
        self.admissions[slot] = None;
        if active.prepared_image.is_some() {
            self.preparing_image = None;
        }
        let mut incoming = active.incoming;
        incoming.host_lookup = false;
        incoming.readmitted = true;
        self.parked.push_front((
            incoming,
            Parked {
                last_token: active.last_token,
                stops: active.stops,
                text_stops: active.text_stops,
                carried: active.carried,
                held_logprobs: active.held_logprobs,
            },
        ));
    }

    fn observe_prefill_rate(&mut self, fence: crate::cache::Fence, tokens: u64) {
        let Some(position) = self
            .prefill_started
            .iter()
            .position(|(started, _, _)| *started == fence)
        else {
            return;
        };
        let (_, instant, _) = self.prefill_started.remove(position).unwrap();
        let seconds = instant.elapsed().as_secs_f64();
        if tokens > 0 && seconds > 0.0 {
            self.prefill_rate = 0.8 * self.prefill_rate + 0.2 * (tokens as f64 / seconds);
        }
    }

    fn fail_all(&self, error: &Error) {
        for request in self.active.iter().flatten() {
            request.incoming.fail(error.clone());
        }
        for (request, _) in &self.parked {
            request.fail(error.clone());
        }
        for request in &self.pending {
            request.fail(error.clone());
        }
        while let Ok(group) = self.receiver.try_recv() {
            for request in group {
                request.fail(error.clone());
            }
        }
    }
}

/// Plan the admission, look the history up in the host store above the GPU
/// checkpoint, and start a restore when the transfer beats recomputation.
fn admit_with_host_store(
    scheduler: &mut RequestScheduler,
    store: &mut dyn crate::offload::PrefixStore,
    spec: RequestSpec,
    prefill_rate: f64,
) -> Result<RequestHandle> {
    let plan = scheduler.plan_admission(&spec)?;
    let hit = store
        .lookup(plan.history(), plan.resident_tokens())
        .filter(|hit| crate::offload::worth_fetching(hit, prefill_rate));
    let Some(hit) = hit else {
        let handle = scheduler.admit_planned(spec, plan, None)?.handle;
        // Lookups count once per admitted request, not per attempt of a waiting one.
        scheduler.record_host_lookup();
        return Ok(handle);
    };
    let inbound = crate::scheduler::InboundRequest {
        key: hit.key,
        consumed_tokens: hit.consumed_tokens,
        pages: hit.pages,
    };
    let admitted = match scheduler.admit_planned(spec.clone(), plan, Some(inbound)) {
        Ok(admitted) => admitted,
        Err(Error::Admission(AdmissionError::Capacity(_))) => {
            let handle = scheduler.admit(spec)?;
            scheduler.record_host_lookup();
            return Ok(handle);
        }
        Err(error) => return Err(error),
    };
    scheduler.record_host_lookup();
    let Some(ticket) = admitted.ticket else {
        return Ok(admitted.handle);
    };
    let key = ticket.key;
    if store.fetch(hit, ticket).is_err() {
        // No event follows a rejected fetch; this request is the ticket's only
        // waiter, so retire it and admit the request again without the store.
        scheduler.finish_inbound(key, false)?;
        return scheduler.admit(spec);
    }
    Ok(admitted.handle)
}

fn cancel_stopped(
    stopped: &[RequestHandle],
    scheduler: &mut RequestScheduler,
    active: &mut [Option<Active<'_>>],
    admissions: &mut [Option<crate::model::Admission>],
) -> Result<()> {
    for (index, &handle) in stopped.iter().enumerate() {
        // Several tokens from one completion can encounter the same poisoned grammar.
        if stopped[..index].contains(&handle) {
            continue;
        }
        scheduler.cancel(handle)?;
        active[handle.slot as usize] = None;
        admissions[handle.slot as usize] = None;
    }
    Ok(())
}

fn prepare_copy_proposal(
    (mut ids, mut count): ([i64; 7], usize),
    consumed: u64,
    grammar: Option<&crate::grammar::RequestGrammar>,
    eos_ids: &[u32],
) -> Result<crate::scheduler::speculation::CopyProposal> {
    if let Some(end) = ids[..count]
        .iter()
        .position(|id| eos_ids.contains(&(*id as u32)))
    {
        count = end;
    }
    let grammar_masks = if let Some(grammar) = grammar {
        let tokens: Vec<_> = ids[..count].iter().map(|id| *id as u32).collect();
        let (legal, masks) = grammar.copy_masks(&tokens)?;
        count = legal;
        masks
    } else {
        Vec::new()
    };
    ids[count..].fill(-1);
    Ok(crate::scheduler::speculation::CopyProposal {
        consumed,
        tokens: ids,
        count,
        grammar_masks,
    })
}

fn validate_sampling_admissions(
    step: &ScheduledStep,
    admissions: &[Option<crate::model::Admission>],
) -> Result<()> {
    if !step.kind.samples() {
        return Ok(());
    }
    if step.lanes.is_empty() {
        return Err("sampling submission has no request lanes".into());
    }
    for lane in &step.lanes {
        admissions
            .get(lane.request.slot as usize)
            .and_then(Option::as_ref)
            .filter(|admission| admission.handle == lane.request)
            .ok_or("sampling submission lacks its matching admission")?;
    }
    Ok(())
}

fn observe_submission(
    metrics: &super::metrics::Metrics,
    scheduler: &RequestScheduler,
    step: &ScheduledStep,
    outstanding: usize,
) -> Result<()> {
    metrics.launched(step.kind);
    let (mut constrained, mut unconstrained) = (false, false);
    for lane in &step.lanes {
        if scheduler.request(lane.request)?.grammar_constrained {
            constrained = true;
        } else {
            unconstrained = true;
        }
    }
    if constrained {
        metrics.constrained_submission(
            outstanding,
            matches!(step.kind, GraphKind::Decode | GraphKind::Speculative) && unconstrained,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Admission, PromptData};
    use crate::scheduler::SchedulerCapabilities;

    #[test]
    fn general_sampler_handles_mixed_positive_temperatures_with_matching_epochs() {
        use crate::scheduler::ScheduledLane;

        let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|slot| {
            Some(Admission {
                handle: RequestHandle {
                    slot: slot as u32,
                    generation: 17 + slot as u64,
                },
                sampling: SamplingParameters {
                    temperature: if slot == 0 { 1. } else { 0.5 },
                    top_k: 1,
                    ..Default::default()
                },
                max_new_tokens: 16,
                prompt: vec![7],
                data: PromptData {
                    positions: vec![[0; 3]],
                    vision_token_indices: vec![-1],
                    ..Default::default()
                },
                grammar_mask: None,
                ignore_eos: false,
                prompt_logprobs: false,
            })
        });
        let lane = |slot: usize| ScheduledLane {
            request: RequestHandle {
                slot: slot as u32,
                generation: 17 + slot as u64,
            },
            start_position: 64,
            tokens: 1,
        };
        for kind in [
            GraphKind::Prefill,
            GraphKind::Decode,
            GraphKind::Resume,
            GraphKind::Speculative,
        ] {
            let mut step = ScheduledStep { draft_tokens: 0, packed_token_capacity: 0,
                kind,
                batch_capacity: 4,
                token_capacity: 1,
                mixed: None,
                lanes: vec![lane(3), lane(1)],
                requires_feedback: true,
            };
            validate_sampling_admissions(&step, &admissions).unwrap();
            let before = step.clone();
            step.lanes.push(lane(0));
            validate_sampling_admissions(&step, &admissions).unwrap();
            step.lanes.remove(2);
            assert_eq!(step, before);
            step.lanes.reverse();
            validate_sampling_admissions(&step, &admissions).unwrap();
            step.lanes[0].request.generation += 1;
            assert!(matches!(
                validate_sampling_admissions(&step, &admissions).unwrap_err(),
                Error::Execution(ExecutionError::State(_))
            ));
        }
        let mut step = ScheduledStep { draft_tokens: 0, packed_token_capacity: 0,
            kind: GraphKind::Decode,
            batch_capacity: 1,
            token_capacity: 1,
            mixed: None,
            lanes: vec![lane(0)],
            requires_feedback: false,
        };
        validate_sampling_admissions(&step, &admissions).unwrap();
        admissions[0] = None;
        assert!(validate_sampling_admissions(&step, &admissions).is_err());
        for kind in [
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Vision,
        ] {
            step.kind = kind;
            validate_sampling_admissions(&step, &admissions).unwrap();
        }
    }

    #[test]
    fn copy_preparation_trims_eos_and_preserves_empty_proposals() {
        let copy = prepare_copy_proposal(([11, 12, 13, -1, -1, -1, -1], 3), 11, None, &[12]).unwrap();
        assert_eq!(copy.count, 1);
        assert_eq!(copy.tokens, [11, -1, -1, -1, -1, -1, -1]);
        assert_eq!(copy.consumed, 11);
        assert!(copy.grammar_masks.is_empty());
        let empty = prepare_copy_proposal(([-1; 7], 0), 11, None, &[12]).unwrap();
        assert_eq!((empty.count, empty.tokens), (0, [-1; 7]));
    }

    #[test]
    fn grammar_metrics_count_submitted_lanes_and_ignore_unconstrained_pipeline_depth() {
        use crate::{cache::Fence, execution::CompletionRecord, scheduler::SchedulerLimits};

        fn submit(
            scheduler: &mut RequestScheduler,
            metrics: &super::super::metrics::Metrics,
            sequence: u64,
            outstanding: usize,
            kind: GraphKind,
        ) -> ScheduledPlan {
            let step = scheduler.phase_plan(kind, None);
            let fence = Fence(sequence * 2);
            scheduler.commit(&step, fence).unwrap();
            observe_submission(metrics, scheduler, &step, outstanding).unwrap();
            let mut records = [CompletionRecord::default(); 4];
            for lane in &step.lanes {
                let live = scheduler.request(lane.request).unwrap();
                let samples = matches!(step.kind, GraphKind::Prefill | GraphKind::Decode);
                records[lane.request.slot as usize] = CompletionRecord {
                    generation: lane.request.generation as i64,
                    submission_sequence: fence.0 as i64,
                    consumed_tokens: (lane.start_position + lane.tokens as u64) as i64,
                    generated_tokens: live.generated_tokens as i64 + i64::from(samples),
                    token_id: if samples { 17 } else { -1 },
                    status: 1,
                    ..CompletionRecord::default()
                };
            }
            scheduler
                .complete(fence, &records, Default::default())
                .unwrap();
            if step.kind.is_prefill() {
                let capture = scheduler.phase_plan(GraphKind::Snapshot, None);
                assert_eq!(capture.kind, GraphKind::Snapshot);
                let copy_fence = Fence(fence.0 + 1);
                scheduler.commit(&capture, copy_fence).unwrap();
                let mut copies = [CompletionRecord::default(); 4];
                for lane in &capture.lanes {
                    let live = scheduler.request(lane.request).unwrap();
                    copies[lane.request.slot as usize] = CompletionRecord {
                        generation: lane.request.generation as i64,
                        submission_sequence: copy_fence.0 as i64,
                        consumed_tokens: live.consumed_tokens as i64,
                        generated_tokens: live.generated_tokens as i64,
                        token_id: -1,
                        status: 1,
                        ..Default::default()
                    };
                }
                scheduler
                    .complete(copy_fence, &copies, Default::default())
                    .unwrap();
            }
            step
        }

        let metrics = super::super::metrics::Metrics::default();
        let mut scheduler = RequestScheduler::new(
            [5; 32],
            SchedulerLimits {
                request_slots: 4,
                max_active: 4,
                max_sequence_tokens: 64,
                prefill_chunk_size: 64,
                pages: 4,
                snapshots: 40,
            },
            SchedulerCapabilities {
                vocabulary: 248320,
                page_tokens: 64,
                vision: crate::model::VisionCapabilities {
                    feature_slots: 4,
                    features_per_request: 16384,
                    max_images: 4,
                    patches_per_feature: 4,
                    patch_capacities: vec![256, 1024, 4096, 16384],
                },

                draft_pages_per_request: 1,
                completion_inputs: vec![],

                state_rows_per_request: None,
            },
        )
        .unwrap();
        scheduler
            .measured_graph_durations(&crate::scheduler::tests::duration_graphs(4))
            .unwrap();
        let spec = |token, grammar_constrained| RequestSpec {
            message_boundaries: vec![],
            prompt: vec![token],
            conditioning: vec![],
            max_new_tokens: 20,
            grammar_constrained,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        };
        let constrained = scheduler.admit(spec(7, true)).unwrap();
        assert_eq!(
            submit(&mut scheduler, &metrics, 1, 1, GraphKind::Init).kind,
            GraphKind::Init
        );
        assert_eq!(
            submit(&mut scheduler, &metrics, 2, 1, GraphKind::Prefill).kind,
            GraphKind::Prefill
        );
        let only_constrained = submit(&mut scheduler, &metrics, 3, 1, GraphKind::Decode);
        assert_eq!(only_constrained.kind, GraphKind::Decode);
        assert_eq!(only_constrained.lanes.len(), 1);
        assert!(
            metrics
                .render()
                .contains("flare_mixed_grammar_decode_steps_total 0\n")
        );

        for token in [8, 9] {
            scheduler.admit(spec(token, false)).unwrap();
        }
        assert_eq!(
            submit(&mut scheduler, &metrics, 4, 1, GraphKind::Init).kind,
            GraphKind::Init
        );
        assert_eq!(
            submit(&mut scheduler, &metrics, 5, 1, GraphKind::Prefill).kind,
            GraphKind::Prefill
        );
        assert_eq!(
            submit(&mut scheduler, &metrics, 6, 1, GraphKind::Prefill).kind,
            GraphKind::Prefill
        );
        let mixed = submit(&mut scheduler, &metrics, 7, 1, GraphKind::Decode);
        assert_eq!(mixed.kind, GraphKind::Decode);
        assert_eq!(mixed.lanes.len(), 3);
        assert!(
            metrics
                .render()
                .contains("flare_mixed_grammar_decode_steps_total 1\n")
        );
        scheduler.cancel(constrained).unwrap();
        let unconstrained = submit(&mut scheduler, &metrics, 8, 2, GraphKind::Decode);
        assert_eq!(unconstrained.kind, GraphKind::Decode);
        assert_eq!(unconstrained.lanes.len(), 2);
        assert!(!unconstrained.requires_feedback);
        let text = metrics.render();
        assert!(text.contains("flare_mixed_grammar_decode_steps_total 1\n"));
        assert!(text.contains("flare_constrained_max_inflight 1\n"));
        assert!(text.contains("flare_graph_launches_total{kind=\"decode\"} 3\n"));
    }

    fn stream(capacity: usize) -> (Incoming, Generation) {
        let (events, receiver) = async_mpsc::channel(capacity);
        let (admission, admitted) = oneshot::channel();
        let (first_token, first_token_ready) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        let incoming = Incoming {
            _capacity: None,
            arrival_ns: super::super::observation::monotonic_ns(),
            publications: Default::default(),
            prompt: Prompt {
                tokens: vec![7],
                message_boundaries: vec![],
                data: PromptData {
                    positions: vec![[0; 3]],
                    vision_token_indices: vec![-1],
                    ..Default::default()
                },
                conditioned_inputs: vec![],
                conditioning: vec![],
                sampling: SamplingParameters::default(),
                max_new_tokens: 10,
                stop_sequences: vec![],
                stop_strings: vec![],
                logprobs: false,
                logprob_token_ids: vec![],
                prompt_logprobs: false,
                ignore_eos: false,
                grammar: None,
            },
            events,
            admission: Mutex::new(Some(admission)),
            first_token: std::cell::Cell::new(Some(first_token)),
            cancelled: cancelled.clone(),
            failure: failure.clone(),
            metrics: Arc::new(super::super::metrics::Metrics::default()),
            observation: None,
            host_lookup: true,
            readmitted: false,
            prompt_score_end: 0,
            prompt_scored: 0,
            blocked: None,
        };
        (
            incoming,
            Generation {
                events: receiver,
                admission: admitted,
                first_token: first_token_ready,
                cancelled,
                failure,
                enqueue_ns: None,
            },
        )
    }

    #[test]
    fn readmission_omits_published_scores_and_preserves_the_original_prompt_end() {
        let (mut incoming, _generation) = stream(2);
        incoming.prompt_score_end = 8193;
        assert_eq!(incoming.prompt_score_range(0, 4096), 0..4097);
        incoming.prompt_scored = 4097;
        incoming.readmitted = true;
        assert!(incoming.prompt_score_range(0, 4096).is_empty());
        assert_eq!(incoming.prompt_score_range(4096, 4096), 4097..8193);
        incoming.prompt_scored = 8193;
        assert!(incoming.prompt_score_range(8192, 4096).is_empty());
    }

    #[test]
    fn only_successful_output_events_count_as_publication() {
        let (mut incoming, mut generation) = stream(2);
        incoming.observation = Some(false);
        incoming.observe_token(7);
        assert_eq!(incoming.publications.get(), 0);
        incoming.emit_token(7);
        assert_eq!(incoming.publications.get(), 1);
        incoming.emit_token(8);
        assert_eq!(incoming.publications.get(), 1);
        assert!(incoming.cancelled.load(Ordering::Acquire));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken { .. })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(7))
        ));
    }

    #[tokio::test]
    async fn admission_waits_for_one_choice_and_preserves_later_failures() {
        let (first, first_generation) = stream(2);
        let (second, second_generation) = stream(2);
        let mut generations = [first_generation, second_generation];
        let mut admission = Box::pin(await_admission(&mut generations));
        assert!(futures_util::poll!(&mut admission).is_pending());
        first.notify_admission(Ok(()));
        admission.await.unwrap();
        second.reject_admission(Error::admission(AdmissionError::Capacity, "full"));
        assert!(matches!(
            generations[1].events.recv().await,
            Some(GenerationEvent::AdmissionRejected(Error::Admission(
                AdmissionError::Capacity(_)
            )))
        ));
        first.fail(Error::execution(ExecutionError::Worker, "stopped"));
        assert!(matches!(
            generations[0].events.recv().await,
            Some(GenerationEvent::Error(Error::Execution(
                ExecutionError::Worker(_)
            )))
        ));
    }

    #[tokio::test]
    async fn admission_reports_rejection_and_worker_disconnect_before_output() {
        let (incoming, generation) = stream(2);
        incoming.reject_admission(Error::admission(AdmissionError::Context, "too long"));
        assert!(matches!(
            await_admission(&mut [generation]).await,
            Err(Error::Admission(AdmissionError::Context(_)))
        ));
        let (incoming, generation) = stream(2);
        drop(incoming);
        assert!(matches!(
            await_admission(&mut [generation]).await,
            Err(Error::Execution(ExecutionError::Worker(_)))
        ));
        let (incoming, generation) = stream(2);
        incoming.notify_admission(Ok(()));
        incoming.fail(Error::execution(ExecutionError::Worker, "stopped"));
        assert!(matches!(
            await_admission(&mut [generation]).await,
            Err(Error::Execution(ExecutionError::Worker(_)))
        ));
        await_admission(&mut []).await.unwrap();
    }

    fn retirement_scheduler() -> RequestScheduler {
        let mut scheduler = RequestScheduler::new(
            [3; 32],
            crate::scheduler::SchedulerLimits {
                request_slots: 2,
                max_active: 2,
                max_sequence_tokens: 64,
                prefill_chunk_size: 64,
                pages: 4,
                snapshots: 40,
            },
            SchedulerCapabilities {
                vocabulary: 6,
                page_tokens: 64,
                vision: crate::model::testing::metadata().vision,

                draft_pages_per_request: 0,
                completion_inputs: vec![],

                state_rows_per_request: None,
            },
        )
        .unwrap();
        scheduler
            .measured_graph_durations(&crate::scheduler::tests::duration_graphs(2))
            .unwrap();
        scheduler
    }

    #[test]
    fn continuation_seeds_depend_on_the_seed_and_the_preemption_point() {
        assert_eq!(continuation_seed(7, 100), continuation_seed(7, 100));
        assert_ne!(continuation_seed(7, 100), continuation_seed(7, 101));
        assert_ne!(continuation_seed(7, 100), continuation_seed(8, 100));
        assert_ne!(continuation_seed(7, 100), 7);
    }

    fn retirement_spec(grammar_constrained: bool) -> RequestSpec {
        RequestSpec {
            message_boundaries: vec![],
            prompt: vec![0],
            conditioning: vec![],
            max_new_tokens: 10,
            grammar_constrained,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        }
    }

    #[test]
    fn poisoned_grammar_vector_retires_once_and_preserves_other_request() {
        use crate::{
            cache::Fence,
            execution::{CompletionRecord, TokenVector},
        };

        let vocab: tokenizers::models::bpe::Vocab = ["a", "b", "c", "d", "x"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
        let mut tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer
            .add_special_tokens([tokenizers::AddedToken::from("<eos>", true)])
            .unwrap();
        let compiler = crate::grammar::GrammarCompiler::new(&tokenizer, 6, &[5]).unwrap();
        let mut grammar = compiler.compile_lark("start: \"abcd\"").unwrap();
        grammar.advance(0).unwrap();

        let mut scheduler = retirement_scheduler();
        scheduler.enable_speculation(&[1, 4]).unwrap();
        scheduler.fixed_speculative_width(Some(4)).unwrap();
        scheduler
            .proposal_source(crate::scheduler::speculation::ProposalSource::Copy)
            .unwrap();
        let failed = scheduler.admit(retirement_spec(true)).unwrap();
        for (sequence, expected) in [
            (1, GraphKind::Init),
            (2, GraphKind::Prefill),
            (3, GraphKind::Snapshot),
        ] {
            let step = scheduler.plan().unwrap();
            assert_eq!(step.kind, expected);
            scheduler.commit(&step, Fence(sequence)).unwrap();
            let sampled = expected != GraphKind::Init;
            let mut records = [CompletionRecord::default(); 4];
            records[failed.slot as usize] = CompletionRecord {
                generation: failed.generation as i64,
                submission_sequence: sequence as i64,
                consumed_tokens: i64::from(sampled),
                generated_tokens: i64::from(sampled),
                token_id: if expected == GraphKind::Prefill {
                    0
                } else {
                    -1
                },
                status: 1,
                ..Default::default()
            };
            scheduler
                .complete(Fence(sequence), &records, Default::default())
                .unwrap();
        }
        let (count, grammar_masks) = grammar.copy_masks(&[1, 2, 3]).unwrap();
        assert_eq!(count, 3);
        scheduler
            .set_copy_proposal(
                failed,
                crate::scheduler::speculation::CopyProposal {
                    consumed: 1,
                    tokens: [1, 2, 3, -1, -1, -1, -1],
                    count,
                    grammar_masks,
                },
            )
            .unwrap();
        let step = scheduler.plan().unwrap();
        assert_eq!(
            (step.kind, step.token_capacity),
            (GraphKind::Speculative, 4)
        );
        scheduler.commit(&step, Fence(4)).unwrap();
        let other = scheduler.admit(retirement_spec(false)).unwrap();
        let other_pages = scheduler
            .request(other)
            .unwrap()
            .pages()
            .collect::<Vec<_>>();
        let mut records = [CompletionRecord::default(); 4];
        records[failed.slot as usize] = CompletionRecord {
            generation: failed.generation as i64,
            submission_sequence: 4,
            consumed_tokens: 4,
            generated_tokens: 4,
            token_id: 2,
            status: 1,
            vector: Some(TokenVector {
                emitted_count: 3,
                offered_drafts: 3,
                accepted_drafts: 2,
                draft_cache_consumed: -1,
                verification_width: 4,
                tokens: [4, 1, 2, -1, -1, -1, -1, -1],
                ..Default::default()
            }),
            ..Default::default()
        };
        let generated = scheduler
            .complete(Fence(4), &records, Default::default())
            .unwrap();
        assert_eq!(
            generated
                .iter()
                .map(|token| token.token)
                .collect::<Vec<_>>(),
            [4, 1, 2]
        );
        let (mut incoming, mut generation) = stream(16);
        incoming.observation = Some(false);
        incoming.prompt.grammar = Some(grammar);
        let failure_metrics = incoming.metrics.clone();
        let (other_incoming, mut other_generation) = stream(16);
        let make_active = |handle, incoming| Active {
            request: handle,
            incoming,
            last_token: None,
            next_image: 0,
            selected_images: vec![],
            prepared_image: None,
            stops: super::super::stops::TokenStops::new(vec![], 6).unwrap(),
            text_stops: None,
            held_logprobs: VecDeque::new(),
            carried: Carried::default(),
        };
        let mut active: [Option<Active<'_>>; 4] = std::array::from_fn(|_| None);
        active[failed.slot as usize] = Some(make_active(failed, incoming));
        active[other.slot as usize] = Some(make_active(other, other_incoming));
        let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
        for handle in [failed, other] {
            admissions[handle.slot as usize] = Some(Admission {
                handle,
                sampling: SamplingParameters::default(),
                max_new_tokens: 10,
                prompt: vec![0],
                data: PromptData {
                    positions: vec![[0; 3]],
                    vision_token_indices: vec![-1],
                    ..Default::default()
                },
                grammar_mask: None,
                ignore_eos: false,
                prompt_logprobs: false,
            });
        }
        let mut stopped = Vec::new();
        for token in generated {
            let request = active[token.request.slot as usize].as_mut().unwrap();
            request.incoming.observe_token(token.token);
            let error = request
                .incoming
                .prompt
                .grammar
                .as_mut()
                .unwrap()
                .advance(token.token)
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "token 4 is forbidden by the grammar mask"
            );
            request.incoming.fail(
                ExecutionError::Completion(Failure::caused_by(
                    format!("grammar advancement failed: {error}"),
                    error,
                ))
                .into(),
            );
            stopped.push(token.request);
        }
        assert_eq!(stopped, [failed; 3]);
        assert_eq!(failure_metrics.failed.load(Ordering::Relaxed), 1);
        cancel_stopped(&stopped, &mut scheduler, &mut active, &mut admissions).unwrap();
        assert!(scheduler.request(failed).is_err());
        assert!(active[failed.slot as usize].is_none());
        assert!(admissions[failed.slot as usize].is_none());
        assert_eq!(scheduler.stats().active_requests, 1);
        assert_eq!(
            scheduler
                .request(other)
                .unwrap()
                .pages()
                .collect::<Vec<_>>(),
            other_pages
        );
        assert_eq!(scheduler.request(other).unwrap().generated_tokens, 0);
        assert_eq!(active[other.slot as usize].as_ref().unwrap().request, other);
        assert_eq!(
            admissions[other.slot as usize].as_ref().unwrap().handle,
            other
        );
        assert_eq!(scheduler.plan().unwrap().lanes[0].request, other);
        assert!(matches!(
            other_generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        let mut observed = Vec::new();
        let mut errors = 0;
        while let Ok(event) = generation.events.try_recv() {
            match event {
                GenerationEvent::ObservedToken {
                    token,
                    timestamp_ns: None,
                } => observed.push(token),
                GenerationEvent::Error(Error::Execution(ExecutionError::Completion(_))) => {
                    errors += 1
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert_eq!(observed, [4, 1, 2]);
        assert_eq!(errors, 1);
    }

    #[test]
    fn duplicate_retirement_keeps_resources_until_pending_fence_completes() {
        use crate::{cache::Fence, execution::CompletionRecord};

        let mut scheduler = retirement_scheduler();
        let handle = scheduler.admit(retirement_spec(false)).unwrap();
        let step = scheduler.plan().unwrap();
        scheduler.commit(&step, Fence(1)).unwrap();
        let free_pages = scheduler.free_page_count();
        let free_snapshots = scheduler.free_snapshot_count();
        let mut active: [Option<Active<'_>>; 4] = std::array::from_fn(|_| None);
        let mut admissions: [_; 4] = std::array::from_fn(|_| None);
        cancel_stopped(
            &[handle, handle],
            &mut scheduler,
            &mut active,
            &mut admissions,
        )
        .unwrap();
        assert_eq!(scheduler.stats().retiring_slots, 1);
        assert_eq!(scheduler.free_page_count(), free_pages);
        assert_eq!(scheduler.free_snapshot_count(), free_snapshots);
        let mut records = [CompletionRecord::default(); 4];
        records[handle.slot as usize] = CompletionRecord {
            generation: handle.generation as i64,
            submission_sequence: 1,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
        assert!(
            scheduler
                .complete(Fence(1), &records, Default::default())
                .unwrap()
                .is_empty()
        );
        assert_eq!(scheduler.stats().retiring_slots, 0);
        assert_eq!(scheduler.free_page_count(), 4);
        assert_eq!(scheduler.free_snapshot_count(), 40);
        let reused = scheduler.admit(retirement_spec(false)).unwrap();
        assert_eq!(reused.slot, handle.slot);
        assert!(reused.generation > handle.generation);
    }

    #[test]
    fn retirement_deduplication_preserves_generation_and_first_occurrence_order() {
        let mut scheduler = retirement_scheduler();
        let first = scheduler.admit(retirement_spec(false)).unwrap();
        let second = scheduler.admit(retirement_spec(false)).unwrap();
        let different_epoch = RequestHandle {
            generation: first.generation + 1,
            ..first
        };
        let mut active: [Option<Active<'_>>; 4] = std::array::from_fn(|_| None);
        let mut admissions: [_; 4] = std::array::from_fn(|_| None);
        let error = cancel_stopped(
            &[first, first, different_epoch, second],
            &mut scheduler,
            &mut active,
            &mut admissions,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Execution(ExecutionError::State(_))));
        assert!(scheduler.request(first).is_err());
        assert!(scheduler.request(second).is_ok());
    }

    #[test]
    fn string_and_token_stops_both_require_the_host_stop_scheduler_barrier() {
        let (mut incoming, _) = stream(4);
        assert!(!incoming.prompt.has_host_stops());
        incoming.prompt.stop_strings.push("END".into());
        assert!(incoming.prompt.has_host_stops());
        incoming.prompt.stop_strings.clear();
        incoming.prompt.stop_sequences.push(vec![1, 2]);
        assert!(incoming.prompt.has_host_stops());
    }

    #[test]
    fn worker_string_stop_keeps_raw_final_token_and_suppresses_subtoken_suffix() {
        let frontend = super::super::stops::string_tests::frontend();
        let (mut incoming, mut generation) = stream(16);
        incoming.observation = Some(false);
        incoming.prompt.stop_strings = vec!["END".into()];
        let mut request = Active {
            request: RequestHandle {
                slot: 0,
                generation: 1,
            },
            incoming,
            last_token: None,
            next_image: 0,
            selected_images: vec![],
            prepared_image: None,
            stops: super::super::stops::TokenStops::new(vec![], 248320).unwrap(),
            held_logprobs: VecDeque::new(),
            text_stops: Some(
                super::super::stops::TextStops::new(&frontend, vec!["END".into()]).unwrap(),
            ),
            carried: Carried::default(),
        };
        request.incoming.observe_token(0);
        assert!(!request.emit_output(0, &[248046], None).unwrap());
        request.incoming.observe_token(1);
        assert!(request.emit_output(1, &[248046], None).unwrap());
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 0,
                timestamp_ns: None
            })
        ));
        assert!(
            matches!(generation.events.try_recv(), Ok(GenerationEvent::Protocol(pieces)) if pieces == vec![crate::output::Piece::Token(0),crate::output::Piece::Text("before".into())])
        );
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 1,
                timestamp_ns: None
            })
        ));
        assert!(generation.events.try_recv().is_err());
        assert_eq!(
            request.text_stops.as_mut().unwrap().finish(None).unwrap(),
            (vec![], true)
        );
    }

    #[test]
    fn saturated_output_cancels_without_blocking_and_preserves_first_failure() {
        let (incoming, mut generation) = stream(2);
        incoming.emit_token(11);
        incoming.emit_token(12);
        assert!(!incoming.cancelled.load(Ordering::Acquire));
        incoming.emit_token(13);
        incoming.emit_token(14);
        incoming.fail("a later shutdown error".into());
        assert!(incoming.cancelled.load(Ordering::Acquire));
        assert_eq!(incoming.metrics.failed.load(Ordering::Relaxed), 1);
        assert_eq!(
            generation.failure().to_string(),
            "generation consumer exceeded the bounded output queue"
        );
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(11))
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(12))
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn observations_keep_raw_stop_and_eos_ids_without_changing_visible_tokens() {
        let (mut incoming, mut generation) = stream(16);
        incoming.observe_token(9);
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        incoming.observation = Some(false);
        let mut stops =
            super::super::stops::TokenStops::new(vec![vec![10, 248046]], 248320).unwrap();
        for token in [9, 10, 248046] {
            incoming.observe_token(token);
            let (visible, _) = stops.push(token);
            for value in visible {
                incoming.emit_token(value);
            }
        }
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 9,
                timestamp_ns: None
            })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Token(9))
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 10,
                timestamp_ns: None
            })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::ObservedToken {
                token: 248046,
                timestamp_ns: None
            })
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        assert!(!incoming.cancelled.load(Ordering::Acquire));
    }

    #[test]
    fn observation_timestamps_are_monotonic_and_backpressure_remains_bounded() {
        let (mut incoming, mut generation) = stream(2);
        incoming.observation = Some(true);
        let start = super::super::observation::monotonic_ns();
        incoming.observe_token(248044);
        incoming.observe_token(17);
        incoming.observe_token(18);
        assert!(incoming.cancelled.load(Ordering::Acquire));
        let mut previous = start;
        for expected in [248044, 17] {
            let GenerationEvent::ObservedToken {
                token,
                timestamp_ns: Some(timestamp),
            } = generation.events.try_recv().unwrap()
            else {
                panic!("missing raw observation")
            };
            assert_eq!(token, expected);
            assert!(timestamp >= previous);
            previous = timestamp;
        }
        assert!(matches!(
            generation.failure(),
            Error::Execution(ExecutionError::Capacity(_))
        ));
    }

    #[test]
    fn disconnected_output_marks_cancellation_and_retains_failure_for_the_reader() {
        let (incoming, generation) = stream(2);
        let failure = generation.failure.clone();
        drop(generation);
        assert!(incoming.cancelled.load(Ordering::Acquire));
        incoming.emit_token(11);
        assert_eq!(incoming.metrics.failed.load(Ordering::Relaxed), 1);
        assert!(matches!(
            failure.lock().unwrap().as_ref().unwrap(),
            Error::Execution(ExecutionError::Capacity(_))
        ));
    }

    #[test]
    fn resource_rejection_is_distinct_from_execution_failure_and_retains_first_error() {
        let mut scheduler = RequestScheduler::new(
            [7; 32],
            crate::scheduler::SchedulerLimits {
                request_slots: 2,
                max_active: 2,
                max_sequence_tokens: 64,
                prefill_chunk_size: 64,
                pages: 1,
                snapshots: 2,
            },
            SchedulerCapabilities {
                vocabulary: 248320,
                page_tokens: 64,
                vision: crate::model::VisionCapabilities {
                    feature_slots: 4,
                    features_per_request: 16384,
                    max_images: 4,
                    patches_per_feature: 4,
                    patch_capacities: vec![256, 1024, 4096, 16384],
                },

                draft_pages_per_request: 1,
                completion_inputs: vec![],

                state_rows_per_request: None,
            },
        )
        .unwrap();
        let spec = || RequestSpec {
            message_boundaries: vec![],
            prompt: vec![7],
            conditioning: vec![],
            max_new_tokens: 10,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        };
        let active = scheduler.admit(spec()).unwrap();
        let rejected = scheduler.admit(spec()).unwrap_err();
        let (incoming, mut generation) = stream(2);
        incoming.reject_admission(rejected.clone());
        incoming.fail("later runtime error".into());
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::AdmissionRejected(message)) if matches!(message, Error::Admission(AdmissionError::Capacity(_))) && message.to_string() == rejected.to_string()
        ));
        assert!(matches!(
            generation.events.try_recv(),
            Err(async_mpsc::error::TryRecvError::Empty)
        ));
        assert!(matches!(
            generation.failure(),
            Error::Admission(AdmissionError::Capacity(_))
        ));
        assert_eq!(generation.failure().to_string(), rejected.to_string());
        assert_eq!(incoming.metrics.failed.load(Ordering::Relaxed), 1);
        assert_eq!(scheduler.stats().active_requests, 1);
        assert_eq!(scheduler.free_page_count(), 0);
        assert_eq!(scheduler.free_snapshot_count(), 2);

        let (execution, mut generation) = stream(2);
        execution.fail(rejected.clone());
        assert!(matches!(
            generation.events.try_recv(),
            Ok(GenerationEvent::Error(message)) if matches!(message, Error::Admission(AdmissionError::Capacity(_))) && message.to_string() == rejected.to_string()
        ));
        scheduler.cancel(active).unwrap();
        assert_eq!(scheduler.free_page_count(), 1);
        assert_eq!(scheduler.free_snapshot_count(), 2);
    }
}

#[cfg(test)]
#[path = "runtime/phase_tests.rs"]
mod phase_tests;

#[cfg(all(test, feature = "runtime-package"))]
mod prompt_tests;
