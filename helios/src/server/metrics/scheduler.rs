use std::{
    collections::VecDeque,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use crate::{
    cache::Fence,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind, REQUEST_LANES},
    scheduler::{RequestHandle, RequestPhase, ScheduledPlan, ScheduledStep, SubmittedTiming},
};

pub(in crate::server) const TRACE_CAPACITY: usize = 1024;

pub(in crate::server) struct GraphLabel {
    pub name: String,
    pub kind: GraphKind,
    pub batch: usize,
    pub tokens: usize,
    pub prefill: usize,
    pub packed: usize,
}

#[derive(Clone, Copy)]
pub(in crate::server) struct GraphSample {
    pub at_ns: u64,
    pub fence: Fence,
    pub graph: usize,
    pub gpu_ns: Option<u64>,
    pub wall_ns: u64,
    pub queued: bool,
    pub selection: SubmittedTiming,
    pub credit_after_ns: f64,
    pub generations: [u64; REQUEST_LANES],
}

#[derive(Clone, Copy)]
pub(in crate::server) struct Dispatch {
    pub at_ns: u64,
    pub graph: usize,
    pub requests: usize,
    pub estimate_ns: u64,
    pub feedback: bool,
}

#[derive(Clone, Copy)]
pub(in crate::server) struct RequestDetail {
    pub handle: RequestHandle,
    pub phase: RequestPhase,
    pub consumed: u64,
    pub prompt: usize,
    pub generated: u32,
    pub last_progress_ns: u64,
    pub last_output_ns: Option<u64>,
    pub published_deliveries: u64,
}

#[derive(Clone, Copy)]
pub(in crate::server) struct SchedulerDetail {
    pub at_ns: u64,
    pub outstanding: usize,
    pub requests: [Option<RequestDetail>; REQUEST_LANES],
}

impl Default for SchedulerDetail {
    fn default() -> Self {
        Self {
            at_ns: 0,
            outstanding: 0,
            requests: [None; REQUEST_LANES],
        }
    }
}

struct TraceBuffer {
    samples: VecDeque<GraphSample>,
    detail: SchedulerDetail,
    dispatch: Option<Dispatch>,
}

#[derive(Default)]
pub(in crate::server) struct SchedulerMetrics {
    pub catalog: OnceLock<Vec<GraphLabel>>,
    buffer: OnceLock<Mutex<TraceBuffer>>,
    enabled: AtomicBool,
    pub detail_requested: AtomicBool,
    pub dropped: AtomicU64,
    pub execution_ns: [AtomicU64; 4],
    pub executions: [AtomicU64; 4],
}

impl SchedulerMetrics {
    pub fn configure(&self, graphs: &[GraphDescriptor]) {
        let _ = self.catalog.set(
            graphs
                .iter()
                .map(|g| GraphLabel {
                    name: g.name.clone(),
                    kind: g.kind,
                    batch: g.batch_capacity,
                    tokens: g.token_capacity,
                    prefill: g.prefill_capacity,
                    packed: g.packed_token_capacity,
                })
                .collect(),
        );
    }

    pub fn start(&self) {
        self.buffer.get_or_init(|| {
            Mutex::new(TraceBuffer {
                samples: VecDeque::with_capacity(TRACE_CAPACITY),
                detail: SchedulerDetail::default(),
                dispatch: None,
            })
        });
        self.detail_requested.store(true, Ordering::Relaxed);
        self.enabled.store(true, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.detail_requested.store(false, Ordering::Relaxed);
    }

    fn write(&self, update: impl FnOnce(&mut TraceBuffer)) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let Some(buffer) = self.buffer.get() else {
            return;
        };
        if let Ok(mut trace) = buffer.try_lock() {
            update(&mut trace);
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn submitted(&self, at_ns: u64, graph: usize, plan: &ScheduledPlan, feedback: bool) {
        self.write(|trace| {
            trace.dispatch = Some(Dispatch {
                at_ns,
                graph,
                requests: plan.lanes.len(),
                estimate_ns: plan.estimated_duration_ns(),
                feedback,
            })
        });
    }

    pub fn completed(
        &self,
        at_ns: u64,
        fence: Fence,
        graph: usize,
        step: &ScheduledStep,
        timing: ExecutionTiming,
        selection: SubmittedTiming,
        credit_after_ns: f64,
    ) {
        use crate::scheduler::TimingOrigin;
        let reason = match selection.origin {
            TimingOrigin::Ordinary => 0, TimingOrigin::Discovery => 1,
            TimingOrigin::Refresh => 2, TimingOrigin::Acceptance => 3,
        };
        let duration = crate::scheduler::observed_duration_ns(timing)
            .unwrap_or(if timing.queued { 0 } else { timing.wall_ns });
        self.execution_ns[reason].fetch_add(duration, Ordering::Relaxed);
        self.executions[reason].fetch_add(1, Ordering::Relaxed);
        self.write(|trace| {
            let mut generations = [0; REQUEST_LANES];
            for lane in &step.lanes {
                generations[lane.request.slot as usize] = lane.request.generation;
            }
            if trace.samples.len() == TRACE_CAPACITY {
                trace.samples.pop_front();
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            trace.samples.push_back(GraphSample {
                at_ns,
                fence,
                graph,
                generations,
                wall_ns: timing.wall_ns,
                queued: timing.queued,
                selection,
                credit_after_ns,
                gpu_ns: timing
                    .gpu_ns
                    .into_iter()
                    .flatten()
                    .max()
                    .filter(|ns| *ns > 0),
            });
        });
    }

    pub fn details(&self, detail: impl FnOnce() -> SchedulerDetail) {
        if !self.detail_requested.load(Ordering::Relaxed)
            || !self.detail_requested.swap(false, Ordering::Relaxed)
        {
            return;
        }
        self.write(|trace| trace.detail = detail());
    }

    pub fn drain(
        &self,
        samples: &mut VecDeque<GraphSample>,
    ) -> (SchedulerDetail, Option<Dispatch>) {
        samples.clear();
        let mut trace = self
            .buffer
            .get()
            .unwrap()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::mem::swap(samples, &mut trace.samples);
        let result = (trace.detail, trace.dispatch);
        self.detail_requested.store(true, Ordering::Relaxed);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step() -> ScheduledStep {
        ScheduledStep { draft_tokens: 0, packed_token_capacity: 0,
            kind: GraphKind::Decode,
            batch_capacity: 1,
            token_capacity: 1,
            mixed: None,
            requires_feedback: false,
            lanes: vec![crate::scheduler::ScheduledLane {
                request: RequestHandle {
                    slot: 7,
                    generation: 11,
                },
                start_position: 10,
                tokens: 1,
            }],
        }
    }

    #[test]
    fn trace_is_opt_in_bounded_and_never_waits_for_the_reader() {
        let metrics = SchedulerMetrics::default();
        let step = step();
        let mut timing = ExecutionTiming::default();
        timing.gpu_ns[0] = Some(10);
        timing.gpu_ns[15] = Some(25);
        metrics.completed(0, Fence(1), 3, &step, timing, SubmittedTiming::default(), 0.0);
        assert!(metrics.buffer.get().is_none());
        metrics.start();
        let lock = metrics.buffer.get().unwrap().lock().unwrap();
        metrics.completed(0, Fence(1), 3, &step, timing, SubmittedTiming::default(), 0.0);
        assert_eq!(metrics.dropped.load(Ordering::Relaxed), 1);
        drop(lock);
        for at in 0..TRACE_CAPACITY + 3 {
            let selection = SubmittedTiming { retained_ns: 100 + at as u64, ..Default::default() };
            metrics.completed(at as u64, Fence(at as u64 + 1), 3, &step, timing, selection, -(at as f64));
        }
        let mut samples = VecDeque::with_capacity(TRACE_CAPACITY);
        metrics.drain(&mut samples);
        assert_eq!(samples.len(), TRACE_CAPACITY);
        assert_eq!(samples.front().unwrap().at_ns, 3);
        assert_eq!(metrics.dropped.load(Ordering::Relaxed), 4);
        assert!(
            samples
                .iter()
                .all(|s| s.gpu_ns == Some(25) && s.generations[7] == 11 && s.generations[0] == 0)
        );
        assert!(samples.iter().all(|sample| {
            sample.fence.0 == sample.at_ns + 1
                && sample.selection.retained_ns == sample.at_ns + 100
                && sample.credit_after_ns == -(sample.at_ns as f64)
        }));
        metrics.stop();
        metrics.completed(5000, Fence(5001), 3, &step, timing, SubmittedTiming::default(), 0.0);
        metrics.drain(&mut samples);
        assert!(samples.is_empty());
    }

    #[test]
    fn selection_counters_include_raw_probe_cost_without_enabling_trace() {
        use crate::scheduler::TimingOrigin;
        let metrics = SchedulerMetrics::default();
        let step = step();
        for (index, origin) in [TimingOrigin::Ordinary, TimingOrigin::Discovery,
            TimingOrigin::Refresh, TimingOrigin::Acceptance].into_iter().enumerate() {
            let mut timing = ExecutionTiming { wall_ns: 100, ..Default::default() };
            timing.gpu_ns[0] = Some(60);
            metrics.completed(0, Fence(1), 0, &step, timing, SubmittedTiming { origin, ..Default::default() }, 0.);
            timing.queued = true;
            metrics.completed(0, Fence(2), 0, &step, timing, SubmittedTiming { origin, ..Default::default() }, 0.);
            assert_eq!(metrics.execution_ns[index].load(Ordering::Relaxed), 160);
            assert_eq!(metrics.executions[index].load(Ordering::Relaxed), 2);
        }
        assert!(metrics.buffer.get().is_none());
    }

    #[test]
    fn completion_context_survives_trace_activation_and_unrelated_dispatches() {
        use crate::scheduler::{CostFreshness, TimingOrigin};
        let metrics = SchedulerMetrics::default();
        let step = step();
        let first = SubmittedTiming {
            retained_ns: 40,
            observed_at_ns: Some(10),
            freshness: CostFreshness::Stale,
            origin: TimingOrigin::Refresh,
            refresh_episode: Some(1),
            credit_ns: 5.0,
            selected_at_ns: 90,
            commit_ns: 100,
        };
        metrics.start();
        metrics.write(|trace| {
            trace.dispatch = Some(Dispatch {
                at_ns: 80, graph: 9, requests: 1, estimate_ns: 999, feedback: false,
            });
        });
        let mut samples = VecDeque::with_capacity(TRACE_CAPACITY);
        metrics.drain(&mut samples);
        let timing = ExecutionTiming { wall_ns: 60, ..Default::default() };
        metrics.completed(160, Fence(1), 3, &step, timing, first, -55.0);
        let (_, dispatch) = metrics.drain(&mut samples);
        assert_eq!(dispatch.unwrap().graph, 9);
        let sample = samples.front().unwrap();
        assert_eq!((sample.fence, sample.graph, sample.selection.retained_ns), (Fence(1), 3, 40));
        assert!(matches!(sample.selection.origin, TimingOrigin::Refresh));
        assert!(matches!(sample.selection.freshness, CostFreshness::Stale));
        assert_eq!(sample.selection.observed_at_ns, Some(10));
        assert_eq!((sample.selection.commit_ns, sample.selection.credit_ns), (100, 5.0));
        assert_eq!((sample.gpu_ns, sample.wall_ns, sample.credit_after_ns), (None, 60, -55.0));
        metrics.stop();
        metrics.start();
        let mut timing = ExecutionTiming { wall_ns: 500, queued: true, ..Default::default() };
        timing.gpu_ns[0] = Some(20);
        metrics.completed(700, Fence(2), 9, &step, timing, SubmittedTiming {
            retained_ns: 18, observed_at_ns: Some(170), freshness: CostFreshness::Fresh,
            origin: TimingOrigin::Ordinary, credit_ns: -55.0, commit_ns: 200,
            selected_at_ns: 190,
            refresh_episode: None,
        }, -54.0);
        metrics.drain(&mut samples);
        assert_eq!(samples.len(), 1);
        let sample = samples.front().unwrap();
        assert_eq!((sample.fence, sample.graph, sample.selection.retained_ns), (Fence(2), 9, 18));
        assert_eq!((sample.gpu_ns, sample.wall_ns, sample.queued), (Some(20), 500, true));
    }

    #[test]
    fn request_detail_is_published_only_when_requested() {
        let metrics = SchedulerMetrics::default();
        metrics.details(|| panic!("disabled telemetry must not inspect requests"));
        metrics.start();
        metrics.details(|| SchedulerDetail {
            at_ns: 42,
            outstanding: 2,
            ..Default::default()
        });
        metrics.details(|| panic!("snapshot already supplied"));
        let mut samples = VecDeque::with_capacity(TRACE_CAPACITY);
        let (detail, _) = metrics.drain(&mut samples);
        assert_eq!((detail.at_ns, detail.outstanding), (42, 2));
        metrics.details(|| SchedulerDetail {
            at_ns: 57,
            ..Default::default()
        });
        assert_eq!(metrics.drain(&mut samples).0.at_ns, 57);
    }
}
