use std::{
    fmt::Write,
    sync::atomic::{AtomicU64, Ordering},
};

pub(super) mod scheduler;

pub(super) fn input_metrics(
    stats: crate::offload::input::InputStats,
    preparation: crate::frontend::images::ImagePreparationCounts,
) -> Vec<(&'static str, &'static str, &'static str, u64)> {
    vec![
        ("input_decoded_images_total", "counter", "Images decoded from encoded request sources.", preparation.decoded_images),
        ("input_prepared_images_total", "counter", "Images resized and converted into vision input tensors.", preparation.prepared_images),
        ("input_decoded_source_bytes", "gauge", "Decoded source bytes retained for one-use image preparation.", preparation.decoded_source_bytes),
        ("input_decoded_source_capacity", "gauge", "Maximum decoded source bytes retained across requests.", preparation.decoded_source_capacity),
        ("input_cache_metadata_hits_total", "counter", "Coalesced source-description lookups served by host retention.", stats.metadata_hits),
        ("input_cache_metadata_misses_total", "counter", "Coalesced source-description lookups requiring decoding.", stats.metadata_misses),
        ("input_cache_payload_hits_total", "counter", "Coalesced prepared-input lookups served by host retention.", stats.payload_hits),
        ("input_cache_payload_misses_total", "counter", "Coalesced prepared-input lookups requiring preprocessing.", stats.payload_misses),
        ("input_cache_memory_hits_total", "counter", "Prepared-input records read from host memory.", stats.memory_hits),
        ("input_cache_disk_hits_total", "counter", "Prepared-input records read from disk.", stats.disk_hits),
        ("input_cache_failures_total", "counter", "Failed prepared-input reads or writes.", stats.failures),
        ("input_cache_stored_total", "counter", "Prepared-input records committed to disk.", stats.stored),
        ("input_cache_skipped_total", "counter", "Prepared-input jobs skipped because a resource bound was reached.", stats.skipped),
        ("input_cache_queued", "gauge", "Outstanding prepared-input retention jobs.", stats.queued as u64),
        ("input_cache_pending_writes", "gauge", "Prepared-input writes waiting for shared disk capacity.", stats.pending_writes as u64),
        ("input_cache_staging_bytes", "gauge", "Prepared-input bytes held by queued writes and disk-read leases.", stats.staging_bytes),
        ("input_cache_staging_limit", "gauge", "Prepared-input serialization and disk-read window capacity.", stats.staging_limit),
        ("input_cache_index_bytes", "gauge", "Host bytes charged to the prepared-input disk index.", stats.index_bytes),
        ("input_cache_memory_limit", "gauge", "Host memory capacity reserved for prepared-input retention.", stats.memory_limit),
        ("input_cache_memory_records", "gauge", "Prepared-input records retained in host memory.", stats.memory_records as u64),
        ("input_cache_memory_bytes", "gauge", "Host bytes charged to prepared-input retention and active leases.", stats.memory_bytes),
        ("input_cache_disk_records", "gauge", "Prepared-input records retained on disk.", stats.disk_records as u64),
        ("input_cache_disk_bytes", "gauge", "Prepared-input record bytes retained on disk.", stats.disk_bytes),
    ]
}

use serde::{Deserialize, Serialize};

use crate::{
    execution::{CompletionRecord, GraphKind},
    offload::StoreStats,
    scheduler::SchedulerStats,
};

#[derive(Default)]
struct RoundMetrics {
    rounds: AtomicU64,
    gpu_ns: AtomicU64,
    emitted: AtomicU64,
    offered: AtomicU64,
    accepted: AtomicU64,
}

#[derive(Default)]
struct ProposalMetrics {
    rounds: AtomicU64,
    gpu_ns: AtomicU64,
}

/// Point-in-time serving counters and gauges for dashboards.
///
/// `GET /metrics?format=json` serializes this structure with its field names as
/// keys. Totals count from server start; `active`, `pending`, the page fields,
/// and the host record fields are instantaneous gauges. Fields absent from a
/// serialized snapshot deserialize as zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Snapshot {
    /// Prompt tokens consumed by completed prefill work, excluding padding and prefix reuse.
    pub prefill_tokens: u64,
    /// Committed generated tokens, including EOS and tokens withheld by text stops.
    pub generated_tokens: u64,
    /// Draft positions run by the learned drafter; zero for models without one.
    pub draft_steps: u64,
    /// Completed speculative or verification graphs, including terminal no-op rounds.
    pub verify_steps: u64,
    /// Draft positions allocated for progressing request rounds, including rejected drafts.
    pub verification_drafts: u64,
    /// Progressing decode or verification request rounds, including zero-draft rounds.
    pub verification_rounds: u64,
    /// Accepted draft tokens from progressing request rounds, excluding the bonus token.
    pub accepted_drafts: u64,
    /// Progressing request rounds that offered at least one draft.
    pub draft_rounds: u64,
    /// Physical model slots with a live request.
    pub active: u64,
    /// Requests waiting inside the model scheduler, excluding its ingress channel.
    pub pending: u64,
    /// Requests whose generation completed successfully.
    pub completed: u64,
    /// Requests failed by admission, execution, grammar, or output backpressure.
    pub failed: u64,
    /// Sum of maximum rank-local CUDA intervals of completed graphs, including readiness waits.
    pub replay_gpu_ns: u64,
    /// Sum of exposed embedding readiness delay measured by instrumented graphs.
    pub replay_readiness_wait_ns: u64,
    /// Completed graphs with prepared readiness timing outputs.
    pub replay_readiness_samples: u64,
    /// Tokens covered by checkpoints returned from GPU prefix lookups.
    pub reused_tokens: u64,
    /// KV pages owned by immutable checkpoints.
    pub cached_pages: u64,
    /// KV pages reusable after completed GPU fences.
    pub free_pages: u64,
    /// Pages in the per-rank KV pool.
    pub total_pages: u64,
    /// Logical history tokens per KV page.
    pub page_tokens: u64,
    /// Host prefix store lookups whose chain admission fetched.
    pub host_hits: u64,
    /// Tokens covered by host prefix store chains beyond the GPU checkpoint.
    pub host_reused_tokens: u64,
    /// Checkpoint records resident in the host store's registered memory.
    pub host_memory_records: u64,
    /// Checkpoint records in the host store's record directory.
    pub host_disk_records: u64,
    /// Bytes copied from GPU checkpoints to the host store.
    pub host_bytes_out: u64,
    /// Bytes copied from the host store into GPU pages and rows.
    pub host_bytes_in: u64,
    /// Bytes of weights one decode step reads for one request, as the package declares; zero
    /// when it declares none. Times verification steps per second, the weight bandwidth consumed.
    #[serde(default)]
    pub decode_weight_bytes: u64,
}

impl Snapshot {
    /// JSON object of every field plus `learned_drafter`, which reports whether
    /// the model has a learned drafter; `draft_steps` is zero without one.
    pub(super) fn json(&self, learned_drafter: bool) -> serde_json::Value {
        let mut value = serde_json::to_value(self).unwrap_or_default();
        if let Some(fields) = value.as_object_mut() {
            fields.insert("learned_drafter".into(), learned_drafter.into());
        }
        value
    }
}

/// Response body formats of `GET /metrics`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    /// Prometheus text exposition format 0.0.4 of every Helios counter and gauge.
    Prometheus,
    /// The [`Snapshot`] fields as one JSON object.
    Json,
}

impl Format {
    /// Selects the format from the `format` query parameter (`json` or
    /// `prometheus`), else from whichever of `application/json` and
    /// `text/plain` the `Accept` header ranks higher, else Prometheus text.
    /// Equal quality values keep the earlier media range.
    pub(super) fn negotiate(query: Option<&str>, accept: Option<&str>) -> Result<Self, String> {
        if let Some(format) = query
            .into_iter()
            .flat_map(|query| query.split('&'))
            .find_map(|pair| pair.strip_prefix("format="))
        {
            return match format {
                "json" => Ok(Self::Json),
                "prometheus" => Ok(Self::Prometheus),
                _ => Err(format!(
                    "unsupported metrics format {format:?}; use json or prometheus"
                )),
            };
        }
        let mut selected = (0.0, Self::Prometheus);
        for range in accept.unwrap_or_default().split(',') {
            let mut parameters = range.split(';');
            let media = parameters.next().unwrap_or_default().trim();
            let format = if media.eq_ignore_ascii_case("application/json") {
                Self::Json
            } else if media.eq_ignore_ascii_case("text/plain") {
                Self::Prometheus
            } else {
                continue;
            };
            let quality = parameters
                .find_map(|parameter| {
                    let (name, value) = parameter.split_once('=')?;
                    name.trim()
                        .eq_ignore_ascii_case("q")
                        .then(|| value.trim().parse::<f32>().ok())
                        .flatten()
                })
                .unwrap_or(1.0);
            if quality > selected.0 {
                selected = (quality, format);
            }
        }
        Ok(selected.1)
    }
}

#[derive(Default)]
pub(super) struct Metrics {
    pub scheduler_trace: scheduler::SchedulerMetrics,
    total_pages: u64,
    page_tokens: u64,
    decode_weight_bytes: u64,
    pub engram: Option<std::sync::Arc<crate::engram::Stats>>,
    prefill_tokens: AtomicU64,
    generated_tokens: AtomicU64,
    accepted_drafts: AtomicU64,
    draft_rounds: AtomicU64,
    verification_drafts: AtomicU64,
    verification_rounds: AtomicU64,
    pub admitted: AtomicU64,
    pub http_requests: [AtomicU64; 5],
    pub completed: AtomicU64,
    pub cancelled: AtomicU64,
    pub failed: AtomicU64,
    replay_gpu_ns: AtomicU64,
    replay_readiness_wait_ns: AtomicU64,
    replay_readiness_samples: AtomicU64,
    launches: [AtomicU64; 12],
    sampling_launches: [AtomicU64; 5],
    speculative: [[RoundMetrics; 8]; 7],
    proposals: [[ProposalMetrics; 7]; 7],
    draft_sources: [[[AtomicU64; 2]; 2]; 7],
    mixed_grammar_decode_steps: AtomicU64,
    constrained_max_inflight: AtomicU64,
    cache_lookups: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    reused_tokens: AtomicU64,
    snapshot_publications: AtomicU64,
    snapshot_evictions: AtomicU64,
    preemptions: AtomicU64,
    message_checkpoints: AtomicU64,
    prompt_checkpoints: AtomicU64,
    active_requests: AtomicU64,
    free_slots: AtomicU64,
    retiring_slots: AtomicU64,
    free_pages: AtomicU64,
    free_snapshots: AtomicU64,
    snapshot_backing_pages: AtomicU64,
    cached_checkpoints: AtomicU64,
    cached_pages: AtomicU64,
    pub pending: AtomicU64,
    host_lookups: AtomicU64,
    host_hits: AtomicU64,
    host_reused_tokens: AtomicU64,
    host_stored: AtomicU64,
    host_store_skipped: AtomicU64,
    host_store_failed: AtomicU64,
    host_fetched: AtomicU64,
    host_fetch_failed: AtomicU64,
    host_bytes_out: AtomicU64,
    host_bytes_in: AtomicU64,
    host_outbound_in_flight: AtomicU64,
    host_inbound_in_flight: AtomicU64,
    host_memory_records: AtomicU64,
    host_memory_bytes: AtomicU64,
    host_disk_records: AtomicU64,
    host_disk_bytes: AtomicU64,
    host_outbound_rate: AtomicU64,
    host_inbound_rate: AtomicU64,
    host_inbound_waiting: AtomicU64,
    learned_draft_steps: AtomicU64,
}

impl Metrics {
    pub fn new(total_pages: u32, page_tokens: usize, decode_weight_bytes: u64) -> Self {
        Self {
            total_pages: total_pages as u64,
            page_tokens: page_tokens as u64,
            decode_weight_bytes,
            ..Self::default()
        }
    }

    pub fn progress(&self, prefill: u64, generated: usize, accepted: u64, rounds: u64,
        verification_drafts: u64, verification_rounds: u64) {
        for (counter, value) in [
            (&self.prefill_tokens, prefill),
            (&self.generated_tokens, generated as u64),
            (&self.accepted_drafts, accepted),
            (&self.draft_rounds, rounds),
            (&self.verification_drafts, verification_drafts),
            (&self.verification_rounds, verification_rounds),
        ] {
            if value != 0 {
                counter.fetch_add(value, Ordering::Relaxed);
            }
        }
    }

    pub fn snapshot(&self, learned_drafter: bool) -> Snapshot {
        let mut snapshot = Snapshot {
            draft_steps: if learned_drafter { self.learned_draft_steps.load(Ordering::Relaxed) } else { 0 },
            prefill_tokens: self.prefill_tokens.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            accepted_drafts: self.accepted_drafts.load(Ordering::Relaxed),
            draft_rounds: self.draft_rounds.load(Ordering::Relaxed),
            verification_drafts: self.verification_drafts.load(Ordering::Relaxed),
            verification_rounds: self.verification_rounds.load(Ordering::Relaxed),
            active: self.active_requests.load(Ordering::Relaxed),
            pending: self.pending.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            replay_gpu_ns: self.replay_gpu_ns.load(Ordering::Relaxed),
            replay_readiness_wait_ns: self.replay_readiness_wait_ns.load(Ordering::Relaxed),
            replay_readiness_samples: self.replay_readiness_samples.load(Ordering::Relaxed),
            reused_tokens: self.reused_tokens.load(Ordering::Relaxed),
            cached_pages: self.cached_pages.load(Ordering::Relaxed),
            free_pages: self.free_pages.load(Ordering::Relaxed),
            total_pages: self.total_pages,
            page_tokens: self.page_tokens,
            host_hits: self.host_hits.load(Ordering::Relaxed),
            host_reused_tokens: self.host_reused_tokens.load(Ordering::Relaxed),
            host_memory_records: self.host_memory_records.load(Ordering::Relaxed),
            host_disk_records: self.host_disk_records.load(Ordering::Relaxed),
            host_bytes_out: self.host_bytes_out.load(Ordering::Relaxed),
            host_bytes_in: self.host_bytes_in.load(Ordering::Relaxed),
            decode_weight_bytes: self.decode_weight_bytes,
            ..Snapshot::default()
        };
        for widths in &self.speculative {
            for counts in widths {
                let rounds = counts.rounds.load(Ordering::Relaxed);
                snapshot.verify_steps += rounds;
            }
        }
        snapshot
    }

    pub fn replay_completed(&self, timing: crate::execution::ExecutionTiming) {
        let raw = timing.gpu_ns.into_iter().flatten().max().unwrap_or(0);
        self.replay_gpu_ns.fetch_add(raw, Ordering::Relaxed);
        if let Some(intervals) = timing.readiness {
            self.replay_readiness_wait_ns.fetch_add(intervals.wait_ns(), Ordering::Relaxed);
            self.replay_readiness_samples.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn launched(&self, kind: GraphKind) {
        let index = match kind {
            GraphKind::Init => 0,
            GraphKind::Restore => 1,
            GraphKind::Resume => 2,
            GraphKind::Prefill | GraphKind::ScoredPrefill => 3,
            GraphKind::Decode => 4,
            GraphKind::Vision => 5,
            GraphKind::Snapshot => 6,
            GraphKind::Speculative => 7,
            GraphKind::Mixed => 8,
            GraphKind::Proposal => 9,
            GraphKind::Verification => 10,
            GraphKind::PackedSpeculative => 11,
        };
        self.launches[index].fetch_add(1, Ordering::Relaxed);
        let sampling_index = match kind {
            GraphKind::Prefill | GraphKind::ScoredPrefill => 0,
            GraphKind::Decode => 1,
            GraphKind::Resume => 2,
            GraphKind::Speculative | GraphKind::Verification | GraphKind::PackedSpeculative => 3,
            GraphKind::Mixed => 4,
            _ => return,
        };
        self.sampling_launches[sampling_index].fetch_add(1, Ordering::Relaxed);
    }

    pub fn speculative_completed(
        &self,
        kind: GraphKind,
        batch: usize,
        width: usize,
        elapsed_ns: u64,
        records: &[CompletionRecord],
    ) {
        if kind == GraphKind::Speculative {
            self.learned_draft_steps.fetch_add(width.saturating_sub(1) as u64, Ordering::Relaxed);
        }
        let counters = &self.speculative[batch.trailing_zeros() as usize][width - 1];
        counters.rounds.fetch_add(1, Ordering::Relaxed);
        counters.gpu_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        for record in records.iter().filter(|r| r.status != 0) {
            let Some(v) = record.vector else {
                continue;
            };
            counters
                .emitted
                .fetch_add(v.emitted_count as u64, Ordering::Relaxed);
            counters
                .offered
                .fetch_add(v.offered_drafts as u64, Ordering::Relaxed);
            counters
                .accepted
                .fetch_add(v.accepted_drafts as u64, Ordering::Relaxed);
            for depth in 0..v.offered_drafts as usize {
                let source = usize::from(v.copy_offered_mask & (1 << depth) != 0);
                self.draft_sources[depth][source][0].fetch_add(1, Ordering::Relaxed);
                if depth < v.accepted_drafts as usize {
                    self.draft_sources[depth][source][1].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn packed_speculative_completed(&self, batch: usize, width: usize, drafts: usize,
        elapsed_ns: u64, records: &[CompletionRecord]) {
        self.learned_draft_steps.fetch_add(drafts as u64, Ordering::Relaxed);
        self.speculative_completed(GraphKind::PackedSpeculative, batch, width, elapsed_ns, records);
    }

    pub fn proposal_completed(&self, batch: usize, width: usize, elapsed_ns: u64) {
        self.learned_draft_steps.fetch_add(width.saturating_sub(1) as u64, Ordering::Relaxed);
        let counters = &self.proposals[batch.trailing_zeros() as usize][width - 2];
        counters.rounds.fetch_add(1, Ordering::Relaxed);
        counters.gpu_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
    }

    pub fn constrained_submission(&self, outstanding: usize, mixed_decode: bool) {
        self.constrained_max_inflight
            .fetch_max(outstanding as u64, Ordering::Relaxed);
        if mixed_decode {
            self.mixed_grammar_decode_steps
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn scheduler(&self, stats: SchedulerStats) {
        if stats.host_lookups != 0 || stats.host_hits != 0 {
            for (target, value) in [
                (&self.host_lookups, stats.host_lookups),
                (&self.host_hits, stats.host_hits),
                (&self.host_reused_tokens, stats.host_reused_tokens),
                (&self.host_inbound_waiting, stats.inbound_waiting as u64),
            ] {
                target.store(value, Ordering::Relaxed);
            }
        }
        for (target, value) in [
            (&self.cache_lookups, stats.cache_lookups),
            (&self.cache_hits, stats.cache_hits),
            (&self.cache_misses, stats.cache_lookups - stats.cache_hits),
            (&self.reused_tokens, stats.reused_tokens),
            (&self.snapshot_publications, stats.snapshot_publications),
            (&self.snapshot_evictions, stats.snapshot_evictions),
            (&self.preemptions, stats.preemptions),
            (&self.message_checkpoints, stats.message_checkpoints),
            (&self.prompt_checkpoints, stats.prompt_checkpoints),
            (&self.active_requests, stats.active_requests as u64),
            (&self.free_slots, stats.free_slots as u64),
            (&self.retiring_slots, stats.retiring_slots as u64),
            (&self.free_pages, stats.free_pages as u64),
            (&self.free_snapshots, stats.free_snapshots as u64),
            (
                &self.snapshot_backing_pages,
                stats.snapshot_backing_pages as u64,
            ),
            (&self.cached_checkpoints, stats.cached_checkpoints as u64),
            (&self.cached_pages, stats.cached_pages as u64),
        ] {
            target.store(value, Ordering::Relaxed);
        }
    }

    /// Host prefix store counters and gauges, written from the model thread.
    /// Lookups, hits, and reused tokens come from the scheduler, which applies
    /// the fetch policy.
    pub fn store(&self, stats: StoreStats) {
        for (target, value) in [
            (&self.host_stored, stats.stored),
            (&self.host_store_skipped, stats.store_skipped),
            (&self.host_store_failed, stats.store_failed),
            (&self.host_fetched, stats.fetched),
            (&self.host_fetch_failed, stats.fetch_failed),
            (&self.host_bytes_out, stats.bytes_out),
            (&self.host_bytes_in, stats.bytes_in),
            (
                &self.host_outbound_in_flight,
                stats.outbound_in_flight as u64,
            ),
            (&self.host_inbound_in_flight, stats.inbound_in_flight as u64),
            (&self.host_memory_records, stats.memory_records as u64),
            (&self.host_memory_bytes, stats.memory_bytes),
            (&self.host_disk_records, stats.disk_records as u64),
            (&self.host_disk_bytes, stats.disk_bytes),
            (
                &self.host_outbound_rate,
                stats.outbound_rate.max(0.0) as u64,
            ),
            (&self.host_inbound_rate, stats.inbound_rate.max(0.0) as u64),
        ] {
            target.store(value, Ordering::Relaxed);
        }
    }

    pub fn render(&self) -> String {
        let mut text = String::with_capacity(4096);
        for (name, help, counts) in [
            ("execution_nanoseconds_total", "Raw execution time by scheduler selection reason, including readiness stalls and isolated host handoff.", &self.scheduler_trace.execution_ns),
            ("executions_total", "Completed physical executions by scheduler selection reason; split proposal and verification each count once.", &self.scheduler_trace.executions),
        ] {
            let _ = writeln!(text, "# HELP flare_scheduler_{name} {help}\n# TYPE flare_scheduler_{name} counter");
            for (reason, count) in ["ordinary", "discovery", "timing_refresh", "acceptance_learning"].into_iter().zip(counts) {
                let _ = writeln!(text, "flare_scheduler_{name}{{reason=\"{reason}\"}} {}", count.load(Ordering::Relaxed));
            }
        }
        for (name, help, count) in [
            ("replay_gpu_nanoseconds_total", "Sum of maximum rank-local CUDA durations of completed graphs, including readiness waits.", &self.replay_gpu_ns),
            ("replay_readiness_wait_nanoseconds_total", "Exposed embedding readiness delay measured across ranks; includes RAM and disk delivery.", &self.replay_readiness_wait_ns),
            ("replay_readiness_samples_total", "Completed graphs with prepared readiness timing outputs.", &self.replay_readiness_samples),
        ] {
            let _ = writeln!(text, "# HELP flare_{name} {help}\n# TYPE flare_{name} counter\nflare_{name} {}",
                             count.load(Ordering::Relaxed));
        }
        text.push_str("# HELP flare_http_requests_total HTTP generation or token-count requests by API, before request validation.\n# TYPE flare_http_requests_total counter\n");
        for (api, count) in [
            "chat",
            "completions",
            "messages",
            "responses",
            "count_tokens",
        ]
        .into_iter()
        .zip(&self.http_requests)
        {
            let _ = writeln!(
                &mut text,
                "flare_http_requests_total{{api=\"{api}\"}} {}",
                count.load(Ordering::Relaxed)
            );
        }

        if let Some(stats) = &self.engram {
            stats.render(&mut text);
        }
        text.push_str("# HELP flare_sampling_graph_launches_total Successful native submissions by captured sampler mode; includes consumption-only prefill.\n# TYPE flare_sampling_graph_launches_total counter\n");
        for (kind, count) in ["prefill", "decode", "resume", "speculative", "mixed"]
            .into_iter()
            .zip(&self.sampling_launches)
        {
            let _ = writeln!(
                text,
                "flare_sampling_graph_launches_total{{kind=\"{kind}\",mode=\"general\"}} {}",
                count.load(Ordering::Relaxed)
            );
        }
        for (metric, help) in [
            (
                "rounds_total",
                "Completed combined speculative or verification graphs, including terminal no-op rounds.",
            ),
            (
                "gpu_nanoseconds_total",
                "CUDA event duration of combined speculative or verification graphs; excludes separate proposal graphs and transport.",
            ),
            (
                "emitted_tokens_total",
                "Raw emitted tokens in completed speculative records, including lanes cancelled after submission.",
            ),
            (
                "offered_drafts_total",
                "Drafts offered in completed speculative records.",
            ),
            (
                "accepted_drafts_total",
                "Accepted draft tokens after EOS and output-budget clipping.",
            ),
        ] {
            let _ = writeln!(
                text,
                "# HELP flare_speculative_{metric} {help}\n# TYPE flare_speculative_{metric} counter"
            );
            for (batch, widths) in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .zip(&self.speculative)
            {
                for (width, counts) in (1..=8).zip(widths) {
                    let value = match metric {
                        "rounds_total" => &counts.rounds,
                        "gpu_nanoseconds_total" => &counts.gpu_ns,
                        "emitted_tokens_total" => &counts.emitted,
                        "offered_drafts_total" => &counts.offered,
                        _ => &counts.accepted,
                    }
                    .load(Ordering::Relaxed);
                    let _ = writeln!(
                        text,
                        "flare_speculative_{metric}{{batch=\"{batch}\",width=\"{width}\"}} {value}"
                    );
                }
            }
        }
        for (metric, help) in [
            ("rounds_total", "Completed separate proposal graphs by prepared batch and draft depth."),
            ("gpu_nanoseconds_total", "CUDA event duration of separate proposal graphs; excludes transport and verification."),
        ] {
            let _ = writeln!(text, "# HELP flare_proposal_{metric} {help}\n# TYPE flare_proposal_{metric} counter");
            for (batch, depths) in crate::execution::BATCH_CAPACITIES.into_iter().zip(&self.proposals) {
                for (depth, counts) in (1..=7).zip(depths) {
                    let value = (if metric == "rounds_total" { &counts.rounds } else { &counts.gpu_ns }).load(Ordering::Relaxed);
                    let _ = writeln!(text, "flare_proposal_{metric}{{batch=\"{batch}\",depth=\"{depth}\"}} {value}");
                }
            }
        }
        text.push_str("# HELP flare_speculative_drafts_total Draft outcomes by one-based depth and actual proposal source; accepted counts are clipped at EOS/output budget.\n# TYPE flare_speculative_drafts_total counter\n");
        for (index, sources) in self.draft_sources.iter().enumerate() {
            let depth = index + 1;
            for (source, outcomes) in ["mtp", "copy"].into_iter().zip(sources) {
                for (outcome, count) in ["offered", "accepted"].into_iter().zip(outcomes) {
                    let value = count.load(Ordering::Relaxed);
                    let _ = writeln!(
                        text,
                        "flare_speculative_drafts_total{{depth=\"{depth}\",source=\"{source}\",outcome=\"{outcome}\"}} {value}"
                    );
                }
            }
        }
        for (name, help, value) in [
            (
                "prefill_tokens_total",
                "Prompt tokens consumed by successful completed prefill work, excluding padding and prefix reuse.",
                &self.prefill_tokens,
            ),
            (
                "generated_tokens_total",
                "Committed generated tokens, including EOS and tokens subsequently withheld by text stops.",
                &self.generated_tokens,
            ),
            (
                "accepted_draft_tokens_total",
                "Accepted draft tokens from progressing live request rounds, excluding the bonus token.",
                &self.accepted_drafts,
            ),
            (
                "draft_request_rounds_total",
                "Progressing live request rounds offering at least one draft; denominator for mean acceptance length.",
                &self.draft_rounds,
            ),
            (
                "verification_draft_tokens_total",
                "Draft positions allocated for progressing live request rounds, including rejected drafts and excluding padding.",
                &self.verification_drafts,
            ),
            (
                "verification_request_rounds_total",
                "Progressing live decode or verification request rounds, including zero-draft rounds; denominator for mean planned verification depth.",
                &self.verification_rounds,
            ),
            (
                "requests_admitted_total",
                "Requests admitted into a physical model slot.",
                &self.admitted,
            ),
            (
                "requests_completed_total",
                "Requests whose model generation completed successfully, including explicit token stops.",
                &self.completed,
            ),
            (
                "requests_cancelled_total",
                "Queued or active model requests cancelled before completion.",
                &self.cancelled,
            ),
            (
                "requests_failed_total",
                "Model requests failed by admission, execution, grammar, or output backpressure.",
                &self.failed,
            ),
            (
                "mixed_grammar_decode_steps_total",
                "Successful decode submissions containing both grammar-constrained and unconstrained request lanes.",
                &self.mixed_grammar_decode_steps,
            ),
            (
                "cache_lookups_total",
                "Prefix lookups during model admission, including admissions rejected for resource pressure.",
                &self.cache_lookups,
            ),
            (
                "cache_hits_total",
                "Prefix lookups returning a GPU-ready checkpoint.",
                &self.cache_hits,
            ),
            (
                "cache_reused_tokens_total",
                "Tokens covered by checkpoints returned from prefix lookups.",
                &self.reused_tokens,
            ),
            (
                "snapshot_publications_total",
                "Unique checkpoints made visible after completed GPU writes.",
                &self.snapshot_publications,
            ),
            (
                "message_checkpoints_total",
                "Exact rendered message endpoints retained after completed snapshot copies.",
                &self.message_checkpoints,
            ),
            (
                "prompt_checkpoints_total",
                "Exact prompt endpoints retained after completed snapshot copies.",
                &self.prompt_checkpoints,
            ),
            (
                "snapshot_evictions_total",
                "Checkpoints logically evicted; GPU retirement may still be pending.",
                &self.snapshot_evictions,
            ),
            (
                "preemptions_total",
                "Requests that yielded their KV pages under pressure and were readmitted.",
                &self.preemptions,
            ),
            (
                "host_cache_lookups_total",
                "Host prefix store lookups above the GPU checkpoint during admission.",
                &self.host_lookups,
            ),
            (
                "host_cache_hits_total",
                "Host prefix store lookups whose chain the admission fetched.",
                &self.host_hits,
            ),
            (
                "host_cache_reused_tokens_total",
                "Tokens covered by host prefix store chains beyond the GPU checkpoint.",
                &self.host_reused_tokens,
            ),
            (
                "host_cache_stored_total",
                "Published checkpoints whose copy to the host store completed.",
                &self.host_stored,
            ),
            (
                "host_cache_store_skipped_total",
                "Published checkpoints the host store did not copy: already present, in flight, or no free slabs.",
                &self.host_store_skipped,
            ),
            (
                "host_cache_store_failed_total",
                "Host store copy-outs that failed after draining their transfers.",
                &self.host_store_failed,
            ),
            (
                "host_cache_fetched_total",
                "Host store chains restored into GPU pages and rows.",
                &self.host_fetched,
            ),
            (
                "host_cache_fetch_failed_total",
                "Host store restores that failed; their requests were re-admitted without the host store.",
                &self.host_fetch_failed,
            ),
            (
                "host_cache_bytes_out_total",
                "Bytes copied from GPU checkpoints to the host store.",
                &self.host_bytes_out,
            ),
            (
                "host_cache_bytes_in_total",
                "Bytes copied from the host store into GPU pages and rows.",
                &self.host_bytes_in,
            ),
        ] {
            metric(
                &mut text,
                name,
                "counter",
                help,
                value.load(Ordering::Relaxed),
            );
        }
        metric(
            &mut text,
            "cache_misses_total",
            "counter",
            "Prefix lookups without a GPU-ready checkpoint.",
            self.cache_misses.load(Ordering::Relaxed),
        );
        text.push_str("# HELP flare_graph_launches_total Successful native CUDA graph submissions.\n# TYPE flare_graph_launches_total counter\n");
        for (kind, count) in [
            "init",
            "restore",
            "resume",
            "prefill",
            "decode",
            "vision",
            "snapshot",
            "speculative",
            "mixed",
            "proposal",
            "verification",
            "packed_speculative",
        ]
        .into_iter()
        .zip(&self.launches)
        {
            let _ = writeln!(
                text,
                "flare_graph_launches_total{{kind=\"{kind}\"}} {}",
                count.load(Ordering::Relaxed)
            );
        }
        for (name, help, value) in [
            (
                "constrained_max_inflight",
                "Maximum outstanding executor frames immediately after a successful grammar-constrained graph submission; zero until the first such submission.",
                &self.constrained_max_inflight,
            ),
            (
                "active_requests",
                "Physical model slots with a live request.",
                &self.active_requests,
            ),
            (
                "free_request_slots",
                "Request slots available for admission.",
                &self.free_slots,
            ),
            (
                "retiring_request_slots",
                "Cancelled slots waiting for GPU completion.",
                &self.retiring_slots,
            ),
            (
                "free_kv_pages",
                "KV pages reusable after completed GPU fences.",
                &self.free_pages,
            ),
            (
                "free_snapshots",
                "Unowned snapshot slots; shared backing may be held by KV pages.",
                &self.free_snapshots,
            ),
            (
                "snapshot_backing_pages",
                "KV page ids excluded by owned snapshots, including pending GPU retirement.",
                &self.snapshot_backing_pages,
            ),
            (
                "cached_checkpoints",
                "GPU-ready immutable checkpoints.",
                &self.cached_checkpoints,
            ),
            (
                "cached_kv_pages",
                "Pages owned by immutable checkpoints.",
                &self.cached_pages,
            ),
            (
                "pending_requests",
                "Requests waiting inside the model scheduler, excluding its ingress channel.",
                &self.pending,
            ),
            (
                "host_cache_outbound_in_flight",
                "Host store copy-outs in progress.",
                &self.host_outbound_in_flight,
            ),
            (
                "host_cache_inbound_in_flight",
                "Host store restores in progress.",
                &self.host_inbound_in_flight,
            ),
            (
                "host_cache_inbound_waiting",
                "Admitted requests waiting for a host store restore.",
                &self.host_inbound_waiting,
            ),
            (
                "host_cache_memory_records",
                "Checkpoint records resident in the host store's registered memory.",
                &self.host_memory_records,
            ),
            (
                "host_cache_memory_bytes",
                "Bytes of records resident in the host store's registered memory.",
                &self.host_memory_bytes,
            ),
            (
                "host_cache_disk_records",
                "Checkpoint records in the host store's record directory.",
                &self.host_disk_records,
            ),
            (
                "host_cache_disk_bytes",
                "Bytes of record files in the host store's record directory.",
                &self.host_disk_bytes,
            ),
            (
                "host_cache_outbound_bytes_per_second",
                "Measured copy-out rate of the host store's most recent transfers.",
                &self.host_outbound_rate,
            ),
            (
                "host_cache_inbound_bytes_per_second",
                "Measured restore rate of the host store's most recent transfers.",
                &self.host_inbound_rate,
            ),
        ] {
            metric(
                &mut text,
                name,
                "gauge",
                help,
                value.load(Ordering::Relaxed),
            );
        }
        for (name, help, value) in [
            (
                "total_kv_pages",
                "Total pages in the per-rank KV pool before checkpoint retention.",
                self.total_pages,
            ),
            (
                "kv_page_tokens",
                "Logical history tokens per KV page, including compressed storage.",
                self.page_tokens,
            ),
            (
                "decode_weight_bytes",
                "Bytes of weights one decode step reads for one request, as the package declares; zero when undeclared.",
                self.decode_weight_bytes,
            ),
            (
                "total_kv_tokens",
                "Logical token capacity of the per-rank KV pool before checkpoint retention.",
                self.total_pages.saturating_mul(self.page_tokens),
            ),
            (
                "free_kv_tokens",
                "Logical token capacity of KV pages reusable after completed GPU fences.",
                self.free_pages
                    .load(Ordering::Relaxed)
                    .saturating_mul(self.page_tokens),
            ),
            (
                "cached_kv_tokens",
                "Logical token capacity of pages owned by immutable checkpoints.",
                self.cached_pages
                    .load(Ordering::Relaxed)
                    .saturating_mul(self.page_tokens),
            ),
        ] {
            metric(&mut text, name, "gauge", help, value);
        }
        text
    }
}

pub(super) fn metric(text: &mut String, name: &str, kind: &str, help: &str, value: u64) {
    let _ = writeln!(
        text,
        "# HELP flare_{name} {help}\n# TYPE flare_{name} {kind}\nflare_{name} {value}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_metrics_retain_raw_gpu_time_and_count_only_measured_waits() {
        let metrics = Metrics::default();
        let mut timing = crate::execution::ExecutionTiming::default();
        timing.gpu_ns[0] = Some(10_000);
        timing.gpu_ns[1] = Some(9_000);
        metrics.replay_completed(timing);
        let bytes = [1000u64, 3000, 3100].into_iter().flat_map(u64::to_ne_bytes).collect::<Vec<_>>();
        timing.readiness = Some(crate::execution::ReadinessIntervals::decode(&bytes).unwrap());
        metrics.replay_completed(timing);
        let snapshot = metrics.snapshot(false);
        assert_eq!(snapshot.replay_gpu_ns, 20_000);
        assert_eq!(snapshot.replay_readiness_wait_ns, 2_000);
        assert_eq!(snapshot.replay_readiness_samples, 1);
        let text = metrics.render();
        assert!(text.contains("flare_replay_gpu_nanoseconds_total 20000\n"));
        assert!(text.contains("flare_replay_readiness_wait_nanoseconds_total 2000\n"));
        assert!(text.contains("flare_replay_readiness_samples_total 1\n"));
    }

    #[test]
    fn scheduler_host_gauges_reset_after_inbound_requests_finish() {
        let metrics = Metrics::default();
        let mut stats = SchedulerStats::default();
        metrics.scheduler(stats);
        assert_eq!(metrics.host_inbound_waiting.load(Ordering::Relaxed), 0);
        stats.host_lookups = 1;
        stats.host_hits = 1;
        stats.host_reused_tokens = 64;
        stats.inbound_waiting = 1;
        metrics.scheduler(stats);
        assert_eq!(metrics.host_inbound_waiting.load(Ordering::Relaxed), 1);
        stats.inbound_waiting = 0;
        metrics.scheduler(stats);
        assert_eq!(metrics.host_inbound_waiting.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.host_lookups.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.host_hits.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.host_reused_tokens.load(Ordering::Relaxed), 64);
    }

    #[test]
    fn speculative_metrics_preserve_depth_source_and_clipped_progress() {
        let metrics = Metrics::default();
        let mut records = [CompletionRecord::default(); 4];
        records[2] = CompletionRecord {
            status: 2,
            vector: Some(crate::execution::TokenVector {
                verification_width: 8,
                emitted_count: 2,
                offered_drafts: 7,
                accepted_drafts: 2,
                copy_offered_mask: 0b101,
                ..Default::default()
            }),
            ..Default::default()
        };
        metrics.speculative_completed(GraphKind::Speculative, 1, 8, 1000, &records);
        for learned in [false, true] {
            let snapshot = metrics.snapshot(learned);
            assert_eq!(snapshot.verify_steps, 1);
        }
        let text = metrics.render();
        for line in [
            "flare_speculative_rounds_total{batch=\"1\",width=\"8\"} 1",
            "flare_speculative_gpu_nanoseconds_total{batch=\"1\",width=\"8\"} 1000",
            "flare_speculative_emitted_tokens_total{batch=\"1\",width=\"8\"} 2",
            "flare_speculative_offered_drafts_total{batch=\"1\",width=\"8\"} 7",
            "flare_speculative_drafts_total{depth=\"1\",source=\"copy\",outcome=\"accepted\"} 1",
            "flare_speculative_drafts_total{depth=\"2\",source=\"mtp\",outcome=\"accepted\"} 1",
            "flare_speculative_drafts_total{depth=\"3\",source=\"copy\",outcome=\"offered\"} 1",
            "flare_speculative_drafts_total{depth=\"3\",source=\"copy\",outcome=\"accepted\"} 0",
        ] {
            assert!(text.lines().any(|value| value == line), "{line}");
        }
    }

    #[test]
    fn separate_proposals_keep_their_shape_and_cost_after_narrow_verification() {
        let metrics = Metrics::default();
        metrics.proposal_completed(4, 8, 700);
        metrics.speculative_completed(GraphKind::Verification, 2, 3, 300, &[]);
        metrics.speculative_completed(GraphKind::Verification, 2, 1, 100, &[]);
        let snapshot = metrics.snapshot(true);
        assert_eq!(snapshot.draft_steps, 7);
        assert_eq!(snapshot.verify_steps, 2);
        let text = metrics.render();
        assert!(text.contains("flare_proposal_rounds_total{batch=\"4\",depth=\"7\"} 1"));
        assert!(text.contains("flare_proposal_gpu_nanoseconds_total{batch=\"4\",depth=\"7\"} 700"));
        assert!(text.contains("flare_speculative_gpu_nanoseconds_total{batch=\"2\",width=\"3\"} 300"));
    }

    #[test]
    fn snapshot_json_round_trips_beside_the_prometheus_counters() {
        let metrics = Metrics::new(100, 128, 0);
        metrics.progress(40, 3, 2, 1, 5, 2);
        metrics.scheduler(SchedulerStats {
            active_requests: 2,
            free_pages: 10,
            cached_pages: 3,
            ..SchedulerStats::default()
        });
        let snapshot = metrics.snapshot(true);
        let json = snapshot.json(true);
        assert_eq!(json["generated_tokens"], 3);
        assert_eq!(json["verification_drafts"], 5);
        assert_eq!(json["verification_rounds"], 2);
        assert_eq!(json["free_pages"], 10);
        assert_eq!(json["learned_drafter"], true);
        assert_eq!(serde_json::from_value::<Snapshot>(json).unwrap(), snapshot);
        assert_eq!(
            serde_json::from_str::<Snapshot>(r#"{"generated_tokens":5}"#).unwrap(),
            Snapshot {
                generated_tokens: 5,
                ..Snapshot::default()
            }
        );
        let text = metrics.render();
        for line in [
            "flare_generated_tokens_total 3",
            "flare_verification_draft_tokens_total 5",
            "flare_verification_request_rounds_total 2",
            "flare_prefill_tokens_total 40",
            "flare_active_requests 2",
            "flare_free_kv_pages 10",
            "flare_total_kv_pages 100",
        ] {
            assert!(text.lines().any(|value| value == line), "{line}");
        }
    }

    #[test]
    fn metrics_format_follows_query_then_ranked_accept_then_prometheus() {
        for (query, accept, format) in [
            (None, None, Format::Prometheus),
            (None, Some("*/*"), Format::Prometheus),
            (None, Some("application/json"), Format::Json),
            (
                None,
                Some("application/json, text/plain, */*"),
                Format::Json,
            ),
            (
                None,
                Some("text/plain, application/json"),
                Format::Prometheus,
            ),
            (
                None,
                Some("text/plain;q=0.5, Application/JSON; Q=0.9"),
                Format::Json,
            ),
            (None, Some("application/json;q=0"), Format::Prometheus),
            (
                None,
                Some(
                    "application/openmetrics-text;version=1.0.0;q=0.5,text/plain;version=0.0.4;q=0.3,*/*;q=0.2",
                ),
                Format::Prometheus,
            ),
            (Some("format=json"), Some("text/plain"), Format::Json),
            (
                Some("x=1&format=prometheus"),
                Some("application/json"),
                Format::Prometheus,
            ),
        ] {
            assert_eq!(
                Format::negotiate(query, accept),
                Ok(format),
                "{query:?} {accept:?}"
            );
        }
        assert!(Format::negotiate(Some("format=yaml"), None).is_err());
    }

    #[test]
    fn counters_expose_fixed_names_without_request_payloads() {
        let metrics = Metrics::new(100, 128, 0);
        metrics.scheduler(SchedulerStats {
            cache_lookups: 5,
            cache_hits: 2,
            reused_tokens: 192,
            free_pages: 10,
            cached_pages: 3,
            host_lookups: 4,
            host_hits: 1,
            host_reused_tokens: 640,
            inbound_waiting: 1,
            ..SchedulerStats::default()
        });
        metrics.launched(GraphKind::Restore);
        metrics.launched(GraphKind::Decode);
        metrics.launched(GraphKind::Decode);
        metrics.store(StoreStats {
            bytes_out: 1 << 20,
            memory_records: 3,
            outbound_rate: 2.5e9,
            ..StoreStats::default()
        });
        let text = metrics.render();
        for (name, value) in [
            ("total_kv_pages", 100),
            ("kv_page_tokens", 128),
            ("total_kv_tokens", 12800),
            ("free_kv_pages", 10),
            ("free_kv_tokens", 1280),
            ("cached_kv_pages", 3),
            ("cached_kv_tokens", 384),
        ] {
            assert!(text.contains(&format!("# TYPE flare_{name} gauge\n")));
            assert!(text.contains(&format!("\nflare_{name} {value}\n")));
        }
        assert!(text.contains("flare_host_cache_lookups_total 4\n"));
        assert!(text.contains("flare_host_cache_hits_total 1\n"));
        assert!(text.contains("flare_host_cache_reused_tokens_total 640\n"));
        assert!(text.contains("flare_host_cache_inbound_waiting 1\n"));
        assert!(text.contains("flare_host_cache_bytes_out_total 1048576\n"));
        assert!(text.contains("flare_host_cache_memory_records 3\n"));
        assert!(text.contains("flare_host_cache_outbound_bytes_per_second 2500000000\n"));
        assert!(text.contains("flare_cache_misses_total 3\n"));
        assert!(text.contains("flare_cache_reused_tokens_total 192\n"));
        assert!(text.contains("flare_graph_launches_total{kind=\"restore\"} 1\n"));
        assert!(!text.contains("mode=\"greedy\""));
        assert!(
            text.contains(
                "flare_sampling_graph_launches_total{kind=\"decode\",mode=\"general\"} 2\n"
            )
        );
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("flare_graph_launches_total{"))
                .count(),
            12
        );
    }
}
