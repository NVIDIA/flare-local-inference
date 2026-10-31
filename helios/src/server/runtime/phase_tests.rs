/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::{
    cache::Fence,
    execution::{CompletionRecord, FrameCompletion},
    frontend::{Asset, Frontend, FrontendAssets},
    model::PromptData,
    scheduler::SchedulerLimits,
};
use sha2::{Digest, Sha256};
use std::ops::ControlFlow;
use std::sync::atomic::AtomicUsize;

struct Fixture {
    config: ModelConfig,
    frontend: Frontend,
    stop: AtomicBool,
    metrics: Arc<super::super::metrics::Metrics>,
}

impl Fixture {
    fn new() -> Self {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "flare-worker-phase-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let vocabulary: tokenizers::models::bpe::Vocab = ["a", "b", "c", "d", "x"]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_owned(), id as u32))
            .collect();
        let mut tokenizer = tokenizers::Tokenizer::new(
            tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocabulary, vec![])
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer
            .add_special_tokens([tokenizers::AddedToken::from("<eos>", true)])
            .unwrap();
        let asset = |name: &str, bytes: &[u8]| {
            let path = directory.join(name);
            std::fs::write(&path, bytes).unwrap();
            Asset {
                range: None,
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            }
        };
        let frontend = Frontend::load(
            FrontendAssets {
                tokenizer: asset(
                    "tokenizer.json",
                    tokenizer.to_string(false).unwrap().as_bytes(),
                ),
                tokens: Default::default(),
                chat_template: Some(asset("chat.jinja", b"{% mark %}{{ messages[0].content }}")),
            },
            Default::default(),
        )
        .unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        Self {
            config: ModelConfig {
                max_top_logprobs: 0,
                logprob_token_id_capacity: 0,
                prepared_sequence_tokens: 128,
                contract: crate::model::testing::metadata(),
                identity: [23; 32],
                model_id: "qwen-worker-phases".into(),
                limits: SchedulerLimits {
                    request_slots: 4,
                    max_active: 4,
                    max_sequence_tokens: 128,
                    prefill_chunk_size: 64,
                    pages: 16,
                    snapshots: 16,
                },
                eos_token_ids: vec![5],
                sampling_ranges: Default::default(),
                sampling_defaults: SamplingParameters::default(),
                max_new_tokens: 16,
                speculative_widths: vec![],
                speculative_fixed_width: None,
                speculative_token_capacity: 512,
                decode_weight_bytes: 0,
                history_drafts: false,
                speculative_proposal_source: Default::default(),
            },
            frontend,
            stop: AtomicBool::new(false),
            metrics: Arc::new(Default::default()),
        }
    }

    fn worker(&self, receiver: mpsc::Receiver<Vec<Incoming>>) -> Worker<'_> {
        self.worker_for(receiver)
    }

    fn worker_for(&self, receiver: mpsc::Receiver<Vec<Incoming>>) -> Worker<'_> {
        let manifest = crate::model::testing::manifest(&self.config);
        let mut scheduler = RequestScheduler::new(
            self.config.identity,
            self.config.limits,
            crate::model::scheduler_capabilities_with_manifest(&self.config, &manifest).unwrap(),
        ).unwrap();
        let mut graphs = manifest.graphs.clone();
        for graph in &mut graphs {
            graph.estimated_duration_ns = 1000;
        }
        scheduler.measured_graph_durations(&graphs).unwrap();
        if self.config.history_drafts
            || self.config.speculative_proposal_source
                == crate::scheduler::speculation::ProposalSource::Copy
        {
            scheduler.enable_history_drafts().unwrap();
        }
        let mut worker = Worker::new(
            &self.config,
            &self.frontend,
            &self.stop,
            &self.metrics,
            scheduler,
            receiver,
            4,
        );
        worker.inputs = Some(
            crate::model::Inputs::new(&self.config, &crate::model::testing::manifest(&self.config))
                .unwrap(),
        );
        worker
    }

    fn incoming(&self, tokens: Vec<u32>, limit: u32) -> (Incoming, Generation) {
        let length = tokens.len();
        self.incoming_for(
            tokens,
            limit,
            PromptData {
                positions: (0..length).map(|n| [n as i64; 3]).collect(),
                vision_token_indices: vec![-1; length],
                ..Default::default()
            },
        )
    }

    fn prompt_for(tokens: Vec<u32>, limit: u32, data: PromptData) -> Prompt {
        Prompt {
            tokens,
            message_boundaries: vec![],
            data,
            conditioned_inputs: vec![],
            conditioning: vec![],
            sampling: SamplingParameters::default(),
            max_new_tokens: limit,
            stop_sequences: vec![],
            stop_strings: vec![],
            logprobs: false,
            logprob_token_ids: vec![],
            prompt_logprobs: false,
            ignore_eos: false,
            grammar: None,
        }
    }

    fn incoming_for(
        &self,
        tokens: Vec<u32>,
        limit: u32,
        data: crate::model::PromptData,
    ) -> (Incoming, Generation) {
        let (events, receiver) = async_mpsc::channel(64);
        let (admission, admitted) = oneshot::channel();
        let (first_token, first_token_ready) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None));
        (
            Incoming {
                _capacity: None,
                arrival_ns: super::super::observation::monotonic_ns(),
                publications: Default::default(),
                prompt: Self::prompt_for(tokens, limit, data),
                events,
                admission: Mutex::new(Some(admission)),
                first_token: std::cell::Cell::new(Some(first_token)),
                cancelled: cancelled.clone(),
                failure: failure.clone(),
                metrics: self.metrics.clone(),
                observation: Some(false),
                host_lookup: true,
                readmitted: false,
                prompt_score_end: 0,
                prompt_scored: 0,
                blocked: None,
                pool_lookup: None,
                pool_lookup_started: None,
                host_lookup_outcome: None,
                pool_plan: None,
            },
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
}

#[test]
fn image_admission_queues_excess_images_and_admits_text() {
    use crate::frontend::images::{
        ImageDescription, ImageGeometry, ImageInput, ImageLimits, ImageProcessor, ImageSettings,
        ImageSource, PreparedImageInput, RgbImage,
    };
    let mut fixture = Fixture::new();
    fixture.config.contract.vision = crate::model::VisionCapabilities {
        feature_slots: 1,
        features_per_request: 64,
        max_images: 1,
        patches_per_feature: 4,
        patch_capacities: vec![256],
    };
    let vision_contract: crate::model::contract::PreparedContract =
        serde_json::from_str(include_str!("../../../tests/support/contracts/qwen38.json")).unwrap();
    fixture.config.contract.frontend = vision_contract.frontend;
    let (sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let image = || {
        let (mut incoming, generation) = fixture.incoming(vec![0, 1], 1);
        let geometry = ImageGeometry {
            height: 32,
            width: 32,
            grid_thw: [1, 16, 16],
            patches: 256,
            tokens: 64,
        };
        incoming.prompt.conditioned_inputs = vec![PreparedImageInput::new(
            ImageInput {
                description: ImageDescription {
                    identity: [7; 32],
                    width: 32,
                    height: 32,
                    geometry,
                },
                source: ImageSource::Decoded(Arc::new(
                    RgbImage::new(32, 32, vec![0; 32 * 32 * 3]).unwrap(),
                )),
            },
            0..2,
            0..64,
            Arc::new(
                ImageProcessor::load(ImageSettings::Moonvit {}, ImageLimits::default()).unwrap(),
            ),
        )];
        (incoming, generation)
    };
    let (first, first_generation) = image();
    let (waiting, waiting_generation) = image();
    let (text, text_generation) = fixture.incoming(vec![2, 3], 1);
    sender.send(vec![first, waiting, text]).unwrap();
    assert!(matches!(
        worker.admit().unwrap(),
        ControlFlow::Continue(true)
    ));
    assert_eq!(worker.pending.len(), 1);
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert!(
        worker
            .active
            .iter()
            .flatten()
            .any(|r| r.incoming.prompt.conditioned_inputs.is_empty())
    );
    assert!(!worker.scheduler.has_available_image_slot());
    first_generation.cancelled.store(true, Ordering::Release);
    assert!(matches!(
        worker.admit().unwrap(),
        ControlFlow::Continue(true)
    ));
    assert!(worker.pending.is_empty());
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert!(!worker.scheduler.has_available_image_slot());
    waiting_generation.cancelled.store(true, Ordering::Release);
    assert!(matches!(
        worker.admit().unwrap(),
        ControlFlow::Continue(true)
    ));
    assert!(worker.scheduler.has_available_image_slot());
    assert_eq!(worker.active.iter().flatten().count(), 1);
    drop(text_generation);
}

fn planned(worker: &mut Worker<'_>, kind: GraphKind, sequence: u64) -> ScheduledPlan {
    let (step, _) = worker.plan(0).unwrap();
    let step = step.unwrap();
    assert_eq!(step.kind, kind);
    worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
    step
}

fn completion(
    worker: &Worker<'_>,
    step: &ScheduledStep,
    sequence: u64,
    sampled: Option<u32>,
) -> FrameCompletion {
    let mut records = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let live = worker.scheduler.request(lane.request).unwrap();
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: sequence as i64,
            consumed_tokens: (lane.start_position + lane.tokens as u64) as i64,
            generated_tokens: live.generated_tokens as i64 + i64::from(sampled.is_some()),
            token_id: sampled.map_or(-1, i64::from),
            status: 1,
            ..Default::default()
        };
    }
    FrameCompletion {
        logprobs: None,
        prompt_logprobs: None,
        proposal_confidence: None,
        speculative_survival: None,
        fence: Fence(sequence),
        graph_index: 0,
        timing: Default::default(),
        records: records.into(),
    }
}

fn advance(worker: &mut Worker<'_>, kind: GraphKind, sequence: u64, token: Option<u32>) {
    let step = planned(worker, kind, sequence);
    let frame = completion(worker, &step, sequence, token);
    worker.complete(kind, frame).unwrap();
}

#[test]
fn all_lanes_validate_before_scheduler_or_observation_mutation() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (first, mut first_generation) = fixture.incoming(vec![0], 8);
    let (second, mut second_generation) = fixture.incoming(vec![1], 8);
    sender.send(vec![first]).unwrap();
    sender.send(vec![second]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let step = planned(&mut worker, GraphKind::Init, 1);
    assert_eq!(step.lanes.len(), 2);
    let mut frame = completion(&worker, &step, 1, None);
    frame.records[step.lanes[1].request.slot as usize].token_id =
        fixture.config.contract.vocabulary as i64;
    assert!(matches!(
        worker.complete(GraphKind::Init, frame),
        Err(Error::Execution(ExecutionError::Completion(_)))
    ));
    assert_eq!(worker.scheduler.completed(), Fence(0));
    assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(1000));
    assert_eq!(worker.scheduler.submitted_step(Fence(1)).unwrap(), &*step);
    for lane in &step.lanes {
        let live = worker.scheduler.request(lane.request).unwrap();
        assert_eq!(live.phase, RequestPhase::Initialize);
        assert_eq!((live.consumed_tokens, live.generated_tokens), (0, 0));
    }
    assert!(first_generation.events.try_recv().is_err());
    assert!(second_generation.events.try_recv().is_err());
    worker
        .complete(GraphKind::Init, completion(&worker, &step, 1, None))
        .unwrap();
    assert_eq!(worker.scheduler.completed(), Fence(1));
    assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(1000));
}

#[test]
fn worker_observes_prefill_decode_and_prerequisite_timings() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 13], 8);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    for (kind, fence, token) in [
        (GraphKind::Init, 1, None),
        (GraphKind::Prefill, 2, Some(1)),
        (GraphKind::Snapshot, 3, None),
        (GraphKind::Decode, 4, Some(2)),
    ] {
        let step = planned(&mut worker, kind, fence);
        let mut frame = completion(&worker, &step, fence, token);
        frame.timing.gpu_ns = [
            Some(100),
            Some(200),
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
        ];
        frame.timing.wall_ns = 350;
        worker.complete(kind, frame).unwrap();
        assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(619));
        let metrics = fixture.metrics.snapshot(false);
        assert_eq!(metrics.verification_drafts, 0);
        assert_eq!(
            metrics.verification_rounds,
            u64::from(kind == GraphKind::Decode)
        );
    }
}

#[test]
fn worker_discovers_indeterminate_costs_and_records_actual_progress() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 13], 8);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker
        .scheduler
        .graph_durations(&crate::scheduler::tests::duration_graphs(4))
        .unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let h = worker.active[0].as_ref().unwrap().request;
    for (kind, fence, token) in [
        (GraphKind::Init, 1, None),
        (GraphKind::Prefill, 2, Some(1)),
        (GraphKind::Snapshot, 3, None),
        (GraphKind::Decode, 4, Some(2)),
    ] {
        let step = planned(&mut worker, kind, fence);
        assert_eq!(step.estimated_duration_ns(), 0);
        assert!(step.requires_feedback);
        let mut frame = completion(&worker, &step, fence, token);
        frame.timing.gpu_ns = [
            Some(100),
            Some(200),
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
        ];
        worker.complete(kind, frame).unwrap();
        assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(200));
    }
    let accounting = worker.scheduler.request(h).unwrap().accounting();
    assert_eq!(accounting.committed_prefill_tokens, 13);
    assert_eq!(accounting.committed_output_tokens, 2);
}

#[test]
fn output_clock_advances_only_after_stop_buffering_releases_tokens() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming(vec![0], 8);
    let arrival = incoming.arrival_ns;
    incoming.prompt.stop_sequences = vec![vec![1, 2]];
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let h = worker.active[0].as_ref().unwrap().request;
    assert_eq!(
        worker.scheduler.request(h).unwrap().accounting().arrival_ns,
        arrival
    );
    advance(&mut worker, GraphKind::Init, 1, None);
    advance(&mut worker, GraphKind::Prefill, 2, Some(1));
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .published_deliveries,
        0
    );
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_ns,
        None
    );
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::ObservedToken { token: 1, .. })
    ));
    assert!(generation.events.try_recv().is_err());
    advance(&mut worker, GraphKind::Snapshot, 3, None);
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_ns,
        None
    );
    advance(&mut worker, GraphKind::Decode, 4, Some(3));
    let published = worker
        .scheduler
        .request(h)
        .unwrap()
        .accounting()
        .last_output_ns;
    assert!(published.is_some());
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .published_deliveries,
        1
    );
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_fence,
        Some(Fence(4))
    );
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::ObservedToken { token: 3, .. })
    ));
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::Token(1))
    ));
    assert!(matches!(
        generation.events.try_recv(),
        Ok(GenerationEvent::Token(3))
    ));
    advance(&mut worker, GraphKind::Decode, 5, Some(1));
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .published_deliveries,
        1
    );
    assert_eq!(
        worker
            .scheduler
            .request(h)
            .unwrap()
            .accounting()
            .last_output_ns,
        published
    );
}

#[test]
fn packing_sees_reserved_pages_and_pack_failure_retains_submission_until_drain() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 65], 8);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    advance(&mut worker, GraphKind::Init, 1, None);
    let handle = worker.active[0].as_ref().unwrap().request;
    assert_eq!(worker.scheduler.request(handle).unwrap().pages().count(), 0);
    let step = planned(&mut worker, GraphKind::Prefill, 2);
    let pages: Vec<_> = worker.scheduler.request(handle).unwrap().pages().collect();
    assert_eq!(pages.len(), 1);
    let uploads = crate::model::Inputs::uploads(worker.inputs.as_ref().unwrap());
    let page_upload = uploads
        .iter()
        .find(|upload| upload.buffer == "page_tables")
        .unwrap();
    assert_eq!(page_upload.offset, 0);
    assert_eq!(page_upload.bytes, (pages[0].0 as i32).to_le_bytes());
    worker
        .complete(GraphKind::Prefill, completion(&worker, &step, 2, None))
        .unwrap();
    let step = worker.plan(0).unwrap().0.unwrap();
    assert_eq!(
        (step.kind, step.lanes[0].start_position),
        (GraphKind::Prefill, 64)
    );
    worker.admissions[handle.slot as usize] = None;
    assert!(worker.reserve_and_pack(&step, Fence(3)).is_err());
    assert_eq!(worker.scheduler.submitted_step(Fence(3)).unwrap(), &*step);
    assert_eq!(worker.scheduler.request(handle).unwrap().pages().count(), 2);
    assert_eq!(
        worker.scheduler.request(handle).unwrap().last_use(),
        Fence(3)
    );
    let free = worker.scheduler.free_page_count();
    worker.retire(&[handle]).unwrap();
    assert_eq!(worker.scheduler.free_page_count(), free);
    worker
        .scheduler
        .release_after_shutdown(&Shutdown {
            drained: true,
            error: None,
        })
        .unwrap();
    assert_eq!(
        worker.scheduler.free_page_count(),
        fixture.config.limits.pages as usize
    );
}

#[test]
fn exact_system_and_prompt_checkpoints_publish_only_after_snapshot_completion() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming(vec![0; 13], 2);
    incoming.prompt.message_boundaries = vec![10];
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    advance(&mut worker, GraphKind::Init, 1, None);
    advance(&mut worker, GraphKind::Prefill, 2, None);
    assert_eq!(
        worker.scheduler.request(handle).unwrap().consumed_tokens,
        10
    );
    assert_eq!(worker.scheduler.stats().message_checkpoints, 0);
    assert!(!worker.publish().unwrap());
    let snapshot = planned(&mut worker, GraphKind::Snapshot, 3);
    let uploads = crate::model::Inputs::uploads(worker.inputs.as_ref().unwrap());
    let expected = uploads
        .iter()
        .find(|u| u.buffer == "state_expected_consumed")
        .unwrap();
    assert_eq!(&expected.bytes[..8], &10i64.to_le_bytes());
    assert_eq!(worker.scheduler.stats().message_checkpoints, 0);
    worker
        .complete(GraphKind::Snapshot, completion(&worker, &snapshot, 3, None))
        .unwrap();
    assert_eq!(worker.scheduler.stats().message_checkpoints, 1);
    assert_eq!(
        worker
            .scheduler
            .request(handle)
            .unwrap()
            .reused_prompt_tokens(),
        0
    );
    advance(&mut worker, GraphKind::Prefill, 4, Some(1));
    assert_eq!(
        worker.scheduler.request(handle).unwrap().consumed_tokens,
        13
    );
    assert_eq!(worker.scheduler.stats().prompt_checkpoints, 0);
    assert!(!worker.publish().unwrap());
    advance(&mut worker, GraphKind::Snapshot, 5, None);
    assert_eq!(worker.scheduler.stats().prompt_checkpoints, 1);
    advance(&mut worker, GraphKind::Decode, 6, Some(2));
    assert!(!worker.publish().unwrap());
    advance(&mut worker, GraphKind::Snapshot, 7, None);
    assert!(worker.publish().unwrap());
    assert!(!worker.publish().unwrap());
    assert_eq!(worker.scheduler.stats().snapshot_publications, 3);
    let metrics = fixture.metrics.snapshot(false);
    assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 2));
    let mut finished = 0;
    while let Ok(event) = generation.events.try_recv() {
        if let GenerationEvent::Finished {
            generated_tokens,
            reused_prompt_tokens,
            reason,
            ..
        } = event
        {
            assert_eq!(
                (generated_tokens, reused_prompt_tokens, reason),
                (2, 0, "length")
            );
            finished += 1;
        }
    }
    assert_eq!(finished, 1);
    let (repeat, _generation) = fixture.incoming(vec![0; 13], 2);
    sender.send(vec![repeat]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let live = worker
        .scheduler
        .request(worker.active[0].as_ref().unwrap().request)
        .unwrap();
    assert_eq!(
        (live.phase, live.reused_prompt_tokens()),
        (RequestPhase::Restore, 13)
    );
    advance(&mut worker, GraphKind::Restore, 8, None);
    advance(&mut worker, GraphKind::Resume, 9, Some(1));
    let metrics = fixture.metrics.snapshot(false);
    assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 3));
}

#[test]
fn ignored_eos_at_the_output_limit_finishes_for_length() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming(vec![0], 2);
    incoming.prompt.ignore_eos = true;
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    advance(&mut worker, GraphKind::Init, 1, None);
    advance(&mut worker, GraphKind::Prefill, 2, Some(5));
    advance(&mut worker, GraphKind::Snapshot, 3, None);
    advance(&mut worker, GraphKind::Decode, 4, Some(5));
    advance(&mut worker, GraphKind::Snapshot, 5, None);
    assert!(worker.publish().unwrap());
    let mut finished = Vec::new();
    while let Ok(event) = generation.events.try_recv() {
        if let GenerationEvent::Finished {
            generated_tokens,
            reason,
            termination,
            ..
        } = event
        {
            finished.push((generated_tokens, reason, termination));
        }
    }
    assert_eq!(
        finished,
        vec![(2, "length", crate::output::Termination::OutputLimit)]
    );
}

#[test]
fn disabled_host_store_preserves_decode_prefix_reuse_and_cancellation() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, mut generation) = fixture.incoming(vec![0; 13], 1);
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert!(worker.store.is_none());
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    for (kind, fence, token) in [
        (GraphKind::Init, 1, None),
        (GraphKind::Prefill, 2, Some(1)),
        (GraphKind::Snapshot, 3, None),
    ] {
        advance(&mut worker, kind, fence, token);
        assert!(!worker.service_store().unwrap());
        assert!(worker.scheduler.take_publications().is_empty());
    }
    assert!(worker.publish().unwrap());
    assert!(worker.scheduler.take_publications().is_empty());
    assert!(matches!(
        generation.events.try_recv().unwrap(),
        GenerationEvent::ObservedToken { token: 1, .. }
    ));
    let mut finished = false;
    while let Ok(event) = generation.events.try_recv() {
        match event {
            GenerationEvent::Finished {
                generated_tokens: 1,
                reused_prompt_tokens: 0,
                ..
            } => finished = true,
            GenerationEvent::Error(error) => panic!("{error}"),
            _ => {}
        }
    }
    assert!(finished);
    let (incoming, mut generation) = fixture.incoming(vec![0; 13], 2);
    sender.send(vec![incoming]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    let request = worker.scheduler.request(handle).unwrap();
    assert_eq!(
        (request.phase, request.reused_prompt_tokens()),
        (RequestPhase::Restore, 13)
    );
    assert!(matches!(
        generation.first_token.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    advance(&mut worker, GraphKind::Restore, 4, None);
    assert!(matches!(
        generation.first_token.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    advance(&mut worker, GraphKind::Resume, 5, Some(2));
    assert!(generation.first_token.try_recv().unwrap().is_ok());
    assert!(matches!(
        generation.events.try_recv().unwrap(),
        GenerationEvent::ObservedToken { token: 2, .. }
    ));
    generation.cancelled.store(true, Ordering::Release);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.scheduler.request(handle).is_err());
    assert!(!worker.service_store().unwrap());
    assert_eq!(fixture.metrics.cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(worker.prefill_started.capacity(), 0);
    assert_eq!(worker.prefill_rate, flare_selene::INITIAL_PREFILL_RATE);
    let stats = worker.scheduler.stats();
    assert_eq!((stats.cache_hits, stats.reused_tokens), (1, 13));
    assert_eq!(
        (stats.host_lookups, stats.host_hits, stats.inbound_waiting),
        (0, 0, 0)
    );
    let metrics = fixture.metrics.snapshot(false);
    assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 2));
    assert_eq!(
        (metrics.host_memory_records, metrics.host_disk_records),
        (0, 0)
    );
    assert_eq!((metrics.host_bytes_out, metrics.host_bytes_in), (0, 0));
    let mut shutdown = Shutdown {
        drained: true,
        error: None,
    };
    worker.close_store(&mut shutdown);
    assert!(shutdown.drained);
    assert!(shutdown.error.is_none());
}

#[test]
fn speculative_dashboard_counts_live_rounds_and_batched_forward_passes() {
    for copy in [false, true] {
        let mut fixture = Fixture::new();
        fixture.config.speculative_widths = vec![1, 4];
        fixture.config.speculative_fixed_width = Some(4);
        fixture.config.history_drafts = copy;
        if copy {
            fixture.config.speculative_proposal_source =
                crate::scheduler::speculation::ProposalSource::Copy;
            for graph in &mut fixture.config.contract.graphs {
                graph.draft_cursor = None;
            }
        }
        let (sender, receiver) = mpsc::channel();
        let (incoming, _generation) = fixture.incoming(vec![0; 13], 8);
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        worker.scheduler.enable_speculation(&[1, 4]).unwrap();
        worker.scheduler.fixed_speculative_width(Some(4)).unwrap();
        worker
            .scheduler
            .proposal_source(fixture.config.speculative_proposal_source)
            .unwrap();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(&mut worker, GraphKind::Prefill, 2, Some(0));
        advance(&mut worker, GraphKind::Snapshot, 3, None);
        if copy {
            let handle = worker.active[0].as_ref().unwrap().request;
            worker
                .scheduler
                .set_copy_proposal(
                    handle,
                    crate::scheduler::speculation::CopyProposal {
                        consumed: 13,
                        tokens: [0, 0, 0, -1, -1, -1, -1],
                        count: 3,
                        grammar_masks: vec![],
                    },
                )
                .unwrap();
        }
        let step = planned(&mut worker, GraphKind::Speculative, 4);
        assert_eq!(step.token_capacity, 4);
        let mut frame = completion(&worker, &step, 4, None);
        frame.timing.gpu_ns[0] = Some(1000);
        frame.timing.wall_ns = 1500;
        let record = &mut frame.records[step.lanes[0].request.slot as usize];
        record.consumed_tokens = 16;
        record.generated_tokens = 4;
        record.token_id = 2;
        record.vector = Some(crate::execution::TokenVector {
            emitted_count: 3,
            offered_drafts: 3,
            accepted_drafts: 2,
            copy_offered_mask: if copy { 7 } else { 0 },
            verification_width: 4,
            draft_cache_consumed: if copy { -1 } else { 15 },
            tokens: [1, 1, 2, -1, -1, -1, -1, -1],
            ..Default::default()
        });
        let mut invalid = frame.records;
        invalid[step.lanes[0].request.slot as usize].generated_tokens = 20;
        assert!(
            worker
                .complete(
                    GraphKind::Speculative,
                    FrameCompletion {
                        logprobs: None,
                        prompt_logprobs: None,
                        proposal_confidence: None,
                        speculative_survival: None,
                        records: invalid,
                        ..frame
                    }
                )
                .is_err()
        );
        assert_eq!(fixture.metrics.snapshot(!copy).generated_tokens, 1);
        assert_eq!(fixture.metrics.snapshot(!copy).verification_rounds, 0);
        assert_eq!(fixture.metrics.snapshot(!copy).verification_drafts, 0);
        worker.complete(GraphKind::Speculative, frame).unwrap();
        let profile = worker
            .scheduler
            .request(step.lanes[0].request)
            .unwrap()
            .acceptance();
        let source = fixture.config.speculative_proposal_source;
        assert!(profile.prefix_probability(source, 0).unwrap() > 0.5);
        assert!(profile.prefix_probability(source, 2).unwrap() < 0.125);
        assert_eq!(worker.scheduler.estimated_duration_ns(&step), Some(1063));
        let metrics = fixture.metrics.snapshot(!copy);
        assert_eq!((metrics.prefill_tokens, metrics.generated_tokens), (13, 4));
        assert_eq!((metrics.accepted_drafts, metrics.draft_rounds), (2, 1));
        assert_eq!(
            (metrics.verification_drafts, metrics.verification_rounds),
            (3, 1)
        );
        assert_eq!(metrics.verify_steps, 1);
        assert_eq!(metrics.draft_steps, if copy { 0 } else { 3 });
    }
}

#[test]
fn packed_dashboard_depth_uses_allocated_prefixes_and_live_request_count() {
    use crate::model::contract::{GraphOperation, ValueSource};
    let mut fixture = Fixture::new();
    fixture.config.speculative_widths = (1..=8).collect();
    let mut manifest = crate::model::testing::manifest(&fixture.config);
    let name = "Speculative-4-8";
    let graph = manifest.graphs.iter_mut().find(|g| g.name == name).unwrap();
    graph.kind = GraphKind::Speculative;
    graph.draft_tokens = 7;
    graph.confidence_width = 7;
    graph.packed_token_capacity = 9;
    let contract = fixture
        .config
        .contract
        .graphs
        .iter_mut()
        .find(|g| g.graph == name)
        .unwrap();
    contract.kind = GraphOperation::Speculative;
    for input in &mut contract.inputs {
        if input.source == ValueSource::QueryLength {
            input.source = ValueSource::SpeculativePrefixLimit;
        }
    }
    let mut scheduler = RequestScheduler::new(
        fixture.config.identity,
        fixture.config.limits,
        crate::model::scheduler_capabilities_with_manifest(&fixture.config, &manifest).unwrap(),
    )
    .unwrap();
    let graphs: Vec<_> = manifest
        .graphs
        .iter()
        .filter(|g| g.kind != GraphKind::Speculative || g.token_capacity == 1 || g.name == name)
        .cloned()
        .map(|mut g| {
            g.estimated_duration_ns = if g.name == name { 10_000 } else {
                match g.kind {
                    GraphKind::Speculative | GraphKind::Decode => 100_000,
                    _ => 100,
                }
            };
            g
        })
        .collect();
    scheduler.measured_graph_durations(&graphs).unwrap();
    scheduler
        .enable_speculation(&(1..=8).collect::<Vec<_>>())
        .unwrap();
    let (sender, receiver) = mpsc::channel();
    let mut generations = Vec::new();
    for token in 0..3 {
        let (incoming, generation) = fixture.incoming(vec![token; 13], 16);
        sender.send(vec![incoming]).unwrap();
        generations.push(generation);
    }
    let mut worker = Worker::new(
        &fixture.config,
        &fixture.frontend,
        &fixture.stop,
        &fixture.metrics,
        scheduler,
        receiver,
        4,
    );
    worker.inputs = Some(crate::model::Inputs::new(&fixture.config, &manifest).unwrap());
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let mut sequence = 0;
    let step = loop {
        sequence += 1;
        assert!(sequence < 20);
        let (plan, _) = worker.plan(0).unwrap();
        let step = plan.unwrap();
        worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
        if step.kind == GraphKind::Speculative {
            break step;
        }
        let frame = completion(
            &worker,
            &step,
            sequence,
            step.kind.is_prefill().then_some(0),
        );
        worker.complete(step.kind, frame).unwrap();
    };
    assert_eq!(
        (step.batch_capacity, step.token_capacity, step.lanes.len()),
        (4, 8, 3)
    );
    let mut frame = completion(&worker, &step, sequence, Some(1));
    frame.proposal_confidence = Some([[0.5; 7]; crate::execution::REQUEST_LANES]);
    for (lane, offered) in step.lanes.iter().zip([5, 1, 0]) {
        let record = &mut frame.records[lane.request.slot as usize];
        record.consumed_tokens = lane.start_position as i64 + 1;
        record.vector = Some(crate::execution::TokenVector {
            emitted_count: 1,
            offered_drafts: offered,
            accepted_drafts: 0,
            verification_width: 8,
            draft_cache_consumed: lane.start_position as i64,
            tokens: [1, -1, -1, -1, -1, -1, -1, -1],
            ..Default::default()
        });
    }
    worker
        .complete(GraphKind::Speculative, frame)
        .unwrap();
    let metrics = fixture.metrics.snapshot(true);
    assert_eq!(
        (metrics.verification_drafts, metrics.verification_rounds),
        (6, 3)
    );
    assert_eq!((metrics.accepted_drafts, metrics.draft_rounds), (0, 2));
    assert_eq!((metrics.draft_steps, metrics.verify_steps), (7, 1));
    let start = std::time::Instant::now();
    let mut history = super::super::dashboard::History::new(start, Default::default());
    history.sample(start + std::time::Duration::from_secs(1), metrics);
    assert_eq!(history.rates().depth, Some(2.0));
    assert_eq!(history.rates().acceptance, Some(0.0));
}

#[test]
fn shared_decoded_history_reaches_copy_and_mtp_uploads_after_response_completion() {
    use crate::model::contract::{GraphOperation, InputIndexing, ValueSource};
    use crate::scheduler::speculation::ProposalSource;
    for source in [ProposalSource::Copy, ProposalSource::Mtp] {
        let mut fixture = Fixture::new();
        fixture.config.speculative_widths = vec![1, 8];
        fixture.config.speculative_fixed_width = Some(8);
        fixture.config.speculative_proposal_source = source;
        fixture.config.history_drafts = source == ProposalSource::Mtp;
        for graph in &mut fixture.config.contract.graphs {
            if graph.kind != GraphOperation::Speculative {
                continue;
            }
            if source == ProposalSource::Copy {
                graph.draft_cursor = None;
            }
            for (name, value) in [
                ("copy_ids", ValueSource::CopyTokenIds),
                ("copy_count", ValueSource::CopyTokenCount),
                ("copy_consumed", ValueSource::CopyConsumedTokens),
                ("copy_generation", ValueSource::CopyGeneration),
                ("copy_masks", ValueSource::CopyGrammarMask),
                ("copy_mask_count", ValueSource::CopyGrammarCount),
            ] {
                graph.inputs.push(crate::model::testing::binding(
                    name,
                    value,
                    InputIndexing::RequestRow,
                    0,
                    0,
                ));
            }
        }
        let (sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        worker.scheduler.enable_speculation(&[1, 8]).unwrap();
        worker.scheduler.fixed_speculative_width(Some(8)).unwrap();
        worker.scheduler.proposal_source(source).unwrap();
        let (incoming, _first_output) = fixture.incoming(vec![0, 1, 2, 3], 6);
        sender.send(vec![incoming]).unwrap();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        assert_eq!(worker.scheduler.history_proposal(&[0, 1, 2, 3], 7).1, 0);
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(&mut worker, GraphKind::Prefill, 2, Some(0));
        advance(&mut worker, GraphKind::Snapshot, 3, None);
        for (index, token) in [1, 2, 3, 4, 2].into_iter().enumerate() {
            let sequence = index as u64 + 4;
            let step = planned(&mut worker, GraphKind::Speculative, sequence);
            let live = worker.scheduler.request(step.lanes[0].request).unwrap();
            let mut frame = completion(&worker, &step, sequence, Some(token));
            let record = &mut frame.records[live.handle.slot as usize];
            record.consumed_tokens = live.consumed_tokens as i64 + 1;
            record.vector = Some(crate::execution::TokenVector {
                verification_width: step.token_capacity as i64,
                offered_drafts: step.lanes[0].tokens as i64 - 1,
                emitted_count: 1,
                accepted_drafts: 0,
                draft_cache_consumed: if source == ProposalSource::Copy {
                    -1
                } else {
                    record.consumed_tokens - 1
                },
                tokens: [i64::from(token), -1, -1, -1, -1, -1, -1, -1],
                ..Default::default()
            });
            worker.complete(GraphKind::Speculative, frame).unwrap();
        }
        advance(&mut worker, GraphKind::Snapshot, 9, None);
        worker.publish().unwrap();
        assert!(worker.active.iter().all(Option::is_none));
        let (incoming, _second_output) = fixture.incoming(vec![4, 0, 1, 2], 16);
        sender.send(vec![incoming]).unwrap();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        advance(&mut worker, GraphKind::Init, 10, None);
        advance(&mut worker, GraphKind::Prefill, 11, Some(3));
        advance(&mut worker, GraphKind::Snapshot, 12, None);
        let step = planned(&mut worker, GraphKind::Speculative, 13);
        let live = worker.scheduler.request(step.lanes[0].request).unwrap();
        let uploads = worker.inputs.as_ref().unwrap().uploads();
        let bytes = |name| uploads.iter().find(|u| u.buffer == name).unwrap().bytes;
        assert_eq!(
            i32::from_le_bytes(bytes("copy_count").try_into().unwrap()),
            2
        );
        assert_eq!(
            bytes("copy_ids")
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
            [4, 2, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            i64::from_le_bytes(bytes("copy_consumed").try_into().unwrap()),
            live.consumed_tokens as i64
        );
        assert_eq!(
            i64::from_le_bytes(bytes("copy_generation").try_into().unwrap()),
            live.handle.generation as i64
        );
    }
}

#[test]
fn observations_precede_grammar_failures_and_token_stop_completion() {
    for invalid in [false, true] {
        let fixture = Fixture::new();
        let compiler = crate::grammar::GrammarCompiler::new(
            fixture.frontend.tokenizer(),
            fixture.config.contract.vocabulary,
            &[5],
            &[],
        )
        .unwrap();
        let (sender, receiver) = mpsc::channel();
        let (mut incoming, mut generation) = fixture.incoming(vec![0], 8);
        if invalid {
            incoming.prompt.grammar = Some(compiler.compile_lark("start: \"a\"").unwrap());
        } else {
            incoming.prompt.stop_sequences = vec![vec![0]];
        }
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(
            &mut worker,
            GraphKind::Prefill,
            2,
            Some(if invalid { 4 } else { 0 }),
        );
        assert!(
            matches!(generation.events.try_recv().unwrap(), GenerationEvent::ObservedToken { token, timestamp_ns: None } if token == if invalid { 4 } else { 0 })
        );
        match generation.events.try_recv().unwrap() {
            GenerationEvent::Error(Error::Execution(ExecutionError::Completion(_))) if invalid => {
                assert_eq!(fixture.metrics.failed.load(Ordering::Relaxed), 1);
            }
            GenerationEvent::Finished {
                reason: "stop",
                termination: crate::output::Termination::UserStop,
                ..
            } if !invalid => {
                assert_eq!(fixture.metrics.completed.load(Ordering::Relaxed), 1);
            }
            event => panic!("unexpected completion event: {event:?}"),
        }
        assert!(generation.events.try_recv().is_err());
        assert_eq!(worker.scheduler.stats().active_requests, 0);
        assert!(worker.active.iter().all(Option::is_none));
        assert!(worker.admissions.iter().all(Option::is_none));
    }
}

#[test]
fn fatal_broadcast_preserves_first_failure_and_reaches_pending_and_ingress() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (first, first_generation) = fixture.incoming(vec![0], 8);
    first.fail(Error::execution(
        ExecutionError::Completion,
        "first completion failure",
    ));
    sender.send(vec![first]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let (pending, pending_generation) = fixture.incoming(vec![1], 8);
    worker.pending.push_back(pending);
    let (queued, queued_generation) = fixture.incoming(vec![2], 8);
    sender.send(vec![queued]).unwrap();
    worker.fail_all(&Error::execution(ExecutionError::Worker, "worker failed"));
    assert_eq!(
        first_generation.failure().to_string(),
        "first completion failure"
    );
    assert!(matches!(
        first_generation.failure(),
        Error::Execution(ExecutionError::Completion(_))
    ));
    for generation in [pending_generation, queued_generation] {
        assert_eq!(generation.failure().to_string(), "worker failed");
        assert!(matches!(
            generation.failure(),
            Error::Execution(ExecutionError::Worker(_))
        ));
    }
    assert_eq!(fixture.metrics.failed.load(Ordering::Relaxed), 3);
    assert_eq!(worker.scheduler.stats().active_requests, 1);
}

/// A host store that answers lookups from a script and reports completions on demand.
struct FakeStore {
    copies: std::collections::HashSet<crate::cache::PrefixKey>,
    hit: Option<flare_selene::HostHit>,
    reject_fetch: bool,
    store_outcomes: VecDeque<flare_selene::StoreOutcome>,
    progress: u64,
    stored: Vec<flare_selene::PublishedCheckpoint>,
    fetched: Vec<(flare_selene::HostHit, flare_selene::InboundTicket)>,
    events: VecDeque<flare_selene::StoreEvent>,
    close_error: Option<Error>,
    poll_error: Option<Error>,
    close_panics: bool,
    close_calls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    /// Shares prefixes through a pool: lookups begin at arrival and evictions are reported.
    pooled: bool,
    /// Token counts of the histories `begin_lookup` received.
    begun: Vec<u64>,
    reused: Vec<crate::cache::PrefixKey>,
    /// Whether each `lookup_for` call carried a ticket.
    tickets_seen: Vec<bool>,
    /// Tickets dropped without an answer.
    withdrawn: Arc<AtomicUsize>,
    exports: Vec<(flare_selene::ExportId, flare_selene::ExportChain)>,
    refused: Vec<(flare_selene::ExportId, flare_selene::ExportRefusal)>,
    reject_export: bool,
    evictions_lost: usize,
    /// Tickets stay unanswered instead of answering with nothing.
    pending_tickets: bool,
    /// Each eviction notice with the number of `store` calls made before it.
    evicted: Vec<(
        crate::cache::PrefixKey,
        Option<flare_selene::PublishedCheckpoint>,
        usize,
    )>,
}

impl FakeStore {
    fn new(hit: Option<flare_selene::HostHit>) -> Self {
        Self {
            hit,
            copies: Default::default(),
            reject_fetch: false,
            store_outcomes: VecDeque::new(),
            progress: 0,
            stored: Vec::new(),
            fetched: Vec::new(),
            events: VecDeque::new(),
            close_error: None,
            poll_error: None,
            close_panics: false,
            close_calls: Arc::new(AtomicUsize::new(0)),
            drops: Arc::new(AtomicUsize::new(0)),
            pooled: false,
            begun: Vec::new(),
            reused: Vec::new(),
            tickets_seen: Vec::new(),
            withdrawn: Arc::new(AtomicUsize::new(0)),
            exports: Vec::new(),
            refused: Vec::new(),
            reject_export: false,
            evictions_lost: 0,
            pending_tickets: false,
            evicted: Vec::new(),
        }
    }
}

impl flare_selene::HostStore for FakeStore {
    fn reused(&mut self, key: crate::cache::PrefixKey) {
        self.reused.push(key);
    }
    fn pooled(&self) -> bool {
        self.pooled
    }
    fn begin_lookup(
        &mut self,
        history: &dyn flare_selene::History,
    ) -> Option<flare_selene::pool::LookupTicket> {
        self.begun.push(history.token_count());
        let withdrawn = self.withdrawn.clone();
        Some(flare_selene::pool::LookupTicket::watched(
            if self.pending_tickets {
                flare_selene::pool::LookupState::Pending
            } else {
                flare_selene::pool::LookupState::None
            },
            move || {
                withdrawn.fetch_add(1, Ordering::Relaxed);
            },
        ))
    }
    fn lookup_for(
        &mut self,
        ticket: Option<&flare_selene::pool::LookupTicket>,
        history: &dyn flare_selene::History,
        resident: u64,
        before: u64,
    ) -> Option<flare_selene::HostHit> {
        self.tickets_seen.push(ticket.is_some());
        self.lookup(history, resident, before)
    }
    fn export_pinned(
        &mut self,
        export: flare_selene::ExportId,
        chain: flare_selene::ExportChain,
    ) -> flare_selene::Result<()> {
        if self.reject_export {
            return Err("export rejected".into());
        }
        self.exports.push((export, chain));
        Ok(())
    }
    fn export_refused(
        &mut self,
        export: flare_selene::ExportId,
        reason: flare_selene::ExportRefusal,
    ) {
        self.refused.push((export, reason));
    }
    fn evicted(
        &mut self,
        key: crate::cache::PrefixKey,
        heir: Option<&flare_selene::PublishedCheckpoint>,
    ) {
        self.evicted
            .push((key, heir.cloned(), self.stored.len()));
    }
    fn evictions_lost(&mut self) {
        self.evictions_lost += 1;
    }
    fn lookup(
        &mut self,
        history: &dyn flare_selene::History,
        resident: u64,
        before: u64,
    ) -> Option<flare_selene::HostHit> {
        self.hit.clone().filter(|hit| {
            history.key_at(hit.consumed_tokens) == Some(hit.key)
                && hit.consumed_tokens > resident
                && hit.consumed_tokens <= history.token_count()
                && hit.consumed_tokens < before
        })
    }
    fn store(
        &mut self,
        checkpoint: &flare_selene::PublishedCheckpoint,
    ) -> flare_selene::Result<flare_selene::StoreOutcome> {
        self.stored.push(checkpoint.clone());
        let outcome = self
            .store_outcomes
            .pop_front()
            .unwrap_or(flare_selene::StoreOutcome::Present);
        if outcome == flare_selene::StoreOutcome::Present {
            self.copies.insert(checkpoint.key);
        }
        Ok(outcome)
    }
    fn fetch(
        &mut self,
        hit: flare_selene::HostHit,
        ticket: flare_selene::InboundTicket,
    ) -> flare_selene::Result<()> {
        if self.reject_fetch {
            return Err("fetch rejected".into());
        }
        self.fetched.push((hit, ticket));
        Ok(())
    }
    fn poll(&mut self, events: &mut Vec<flare_selene::StoreEvent>) -> flare_selene::Result<()> {
        if let Some(error) = &self.poll_error {
            return Err(error.clone().into());
        }
        for event in self.events.drain(..) {
            if let flare_selene::StoreEvent::Stored { key, result: Ok(_), .. } = &event {
                self.copies.insert(*key);
            }
            events.push(event);
        }
        Ok(())
    }
    fn stats(&self) -> flare_selene::StoreStats {
        flare_selene::StoreStats {
            progress: self.progress,
            ..Default::default()
        }
    }
    fn close(&mut self) -> flare_selene::Result<()> {
        self.close_calls.fetch_add(1, Ordering::Relaxed);
        assert!(!self.close_panics, "transfer drain panicked");
        self.close_error.clone().map_or(Ok(()), |error| Err(error.into()))
    }
}

impl Drop for FakeStore {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn shutdown_releases_the_store_after_both_drains_complete() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let store = FakeStore::new(None);
    let close_calls = store.close_calls.clone();
    let drops = store.drops.clone();
    worker.attach_store(Some(Box::new(store)));
    let mut shutdown = Shutdown {
        drained: true,
        error: None,
    };
    worker.close_store(&mut shutdown);
    assert!(shutdown.drained);
    assert!(shutdown.error.is_none());
    assert!(worker.store.is_none());
    assert_eq!(close_calls.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    worker.close_store(&mut shutdown);
    drop(worker);
    assert_eq!(close_calls.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn shutdown_retains_the_store_when_its_drain_fails() {
    for previous_error in [None, Some(Error::from("model failed"))] {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.close_error = Some("transfer drain failed".into());
        let close_calls = store.close_calls.clone();
        let drops = store.drops.clone();
        worker.attach_store(Some(Box::new(store)));
        let expected_error = previous_error
            .clone()
            .unwrap_or_else(|| "transfer drain failed".into());
        let mut shutdown = Shutdown {
            drained: true,
            error: previous_error,
        };
        worker.close_store(&mut shutdown);
        assert!(!shutdown.drained);
        assert_eq!(
            shutdown.error.unwrap().to_string(),
            expected_error.to_string()
        );
        assert!(worker.store.is_none());
        drop(worker);
        assert_eq!(close_calls.load(Ordering::Relaxed), 1);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn shutdown_retains_the_store_without_closing_when_the_executor_did_not_drain() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let store = FakeStore::new(None);
    let close_calls = store.close_calls.clone();
    let drops = store.drops.clone();
    worker.attach_store(Some(Box::new(store)));
    let mut shutdown = Shutdown {
        drained: false,
        error: Some("model drain failed".into()),
    };
    worker.close_store(&mut shutdown);
    assert!(!shutdown.drained);
    assert_eq!(shutdown.error.unwrap().to_string(), "model drain failed");
    assert!(worker.store.is_none());
    drop(worker);
    assert_eq!(close_calls.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
}

#[test]
fn shutdown_retains_the_store_when_its_drain_panics() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.close_panics = true;
    let close_calls = store.close_calls.clone();
    let drops = store.drops.clone();
    worker.attach_store(Some(Box::new(store)));
    let mut shutdown = Shutdown {
        drained: true,
        error: None,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker.close_store(&mut shutdown);
    }));
    assert!(result.is_err());
    assert!(!shutdown.drained);
    assert!(worker.store.is_none());
    drop(worker);
    assert_eq!(close_calls.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
}

/// A store that cannot account for its transfers stops the worker, so ids and
/// pins held by those transfers are never reused or released.
#[test]
fn a_store_that_cannot_report_completions_stops_the_worker() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.poll_error = Some("rank 1's prefix record connection failed".into());
    worker.attach_store(Some(Box::new(store)));
    let error = worker.service_store().unwrap_err();
    assert!(
        error.to_string().contains("prefix record connection"),
        "{error}"
    );
    assert!(
        worker.store.is_some(),
        "the worker keeps the store it cannot drain"
    );
}

fn host_hit(history: &crate::cache::PrefixHistory, consumed: u64) -> flare_selene::HostHit {
    let key = history.key_at(consumed).unwrap();
    flare_selene::HostHit {
        key,
        consumed_tokens: consumed,
        resident_tokens: 0,
        pages: (consumed / 64) as usize,
        bytes: 1 << 20,
        tier: flare_selene::Tier::Dram,
        estimated_seconds: 0.0,
        chain: flare_selene::index::Match {
            key,
            consumed_tokens: consumed,
            resident_tokens: 0,
            nodes: vec![key],
            skip_pages: 0,
            pages: (consumed / 64) as usize,
        },
    }
}

fn fake_store<'w>(worker: &'w mut Worker<'_>) -> &'w mut FakeStore {
    let store = worker.store.as_mut().unwrap().as_mut() as *mut dyn flare_selene::HostStore;
    // SAFETY: The worker holds the only store and it was constructed as a FakeStore.
    unsafe { &mut *(store as *mut FakeStore) }
}

#[test]
fn a_gpu_hit_refreshes_the_host_record_at_admission() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let checkpoint = unpinned_checkpoint(&mut worker, 0, 64);
    let key = worker.scheduler.publication(checkpoint).unwrap().key;
    worker.attach_store(Some(Box::new(FakeStore::new(None))));
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    sender.send(vec![incoming]).unwrap();
    worker.admit().unwrap();
    assert!(worker.active[0].is_some());
    assert_eq!(fake_store(&mut worker).reused, [key]);
}

#[test]
fn host_hit_holds_the_request_inbound_until_the_restore_completes() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(&history, 70))))));
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    let live = worker.scheduler.request(handle).unwrap();
    assert_eq!(
        (
            live.phase,
            live.consumed_tokens,
            live.reused_prompt_tokens()
        ),
        (RequestPhase::Inbound, 70, 70)
    );
    assert!(worker.plan(0).unwrap().0.is_none());
    assert_eq!(worker.scheduler.stats().inbound_waiting, 1);
    assert_eq!(
        (
            worker.scheduler.stats().host_lookups,
            worker.scheduler.stats().host_hits
        ),
        (1, 1)
    );
    let (hit, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
    assert_eq!((ticket.key, ticket.pages.len()), (hit.key, 1));
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Fetched {
            key: ticket.key,
            result: Ok(1 << 20),
        });
    assert!(worker.service_store().unwrap());
    let live = worker.scheduler.request(handle).unwrap();
    assert_eq!(
        (live.phase, live.pages().count()),
        (RequestPhase::Restore, 1)
    );
    let restored = live.prefix().unwrap();
    // Polling the restore publishes it without an eager copy-out.
    assert_eq!(fake_store(&mut worker).stored.len(), 1);
    let publication = fake_store(&mut worker).stored[0].clone();
    assert_eq!(
        (
            publication.id,
            publication.chain_pages.len(),
            publication.tail_pages
        ),
        (restored, 1, 1)
    );
    assert_eq!(publication.key, history.key_at(70).unwrap());
    assert_eq!(pins(&worker, restored), 1);
    assert_eq!(fake_store(&mut worker).stored.len(), 1);
    advance(&mut worker, GraphKind::Restore, 1, None);
    advance(&mut worker, GraphKind::Resume, 2, Some(1));
    assert_eq!(
        worker.scheduler.request(handle).unwrap().phase,
        RequestPhase::Decode
    );
    assert_eq!(fixture.metrics.snapshot(false).prefill_tokens, 0);
}

#[test]
fn host_hit_recomputes_when_snapshot_backing_would_prevent_admission() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker
        .scheduler
        .share_snapshot_pages(&vec![0..16; 16])
        .unwrap();
    worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(&history, 70))))));
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    assert_eq!(
        worker.scheduler.request(handle).unwrap().phase,
        RequestPhase::Initialize
    );
    assert_eq!(worker.scheduler.free_page_count(), 14);
    assert_eq!(worker.scheduler.free_snapshot_count(), 16);
    assert!(fake_store(&mut worker).fetched.is_empty());
}

#[test]
fn failed_restores_readmit_the_request_without_the_host_store() {
    for reject_at_fetch in [false, true] {
        let fixture = Fixture::new();
        let (sender, receiver) = mpsc::channel();
        let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
        let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
        history.append(&incoming.prompt.tokens, &[]).unwrap();
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(Some(host_hit(&history, 70)));
        store.reject_fetch = reject_at_fetch;
        worker.attach_store(Some(Box::new(store)));
        let free_pages = worker.scheduler.free_page_count();
        let free_snapshots = worker.scheduler.free_snapshot_count();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        if !reject_at_fetch {
            let handle = worker.active[0].as_ref().unwrap().request;
            assert_eq!(
                worker.scheduler.request(handle).unwrap().phase,
                RequestPhase::Inbound
            );
            let (_, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
            fake_store(&mut worker)
                .events
                .push_back(flare_selene::StoreEvent::Fetched {
                    key: ticket.key,
                    result: Err("transfer failed".into()),
                });
            assert!(worker.service_store().unwrap());
            assert!(worker.active[0].is_none());
            assert_eq!(worker.parked.len(), 1);
            assert!(!worker.parked[0].0.host_lookup);
            assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        }
        let handle = worker.active[0].as_ref().unwrap().request;
        let live = worker.scheduler.request(handle).unwrap();
        assert_eq!(
            (
                live.phase,
                live.consumed_tokens,
                live.reused_prompt_tokens()
            ),
            (RequestPhase::Initialize, 0, 0)
        );
        assert_eq!(fixture.metrics.admitted.load(Ordering::Relaxed), 1);
        assert_eq!(worker.scheduler.stats().inbound_waiting, 0);
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(&mut worker, GraphKind::Prefill, 2, None);
        advance(&mut worker, GraphKind::Prefill, 3, Some(1));
        assert_eq!(
            worker.scheduler.request(handle).unwrap().consumed_tokens,
            70
        );
        worker.retire(&[handle]).unwrap();
        worker
            .scheduler
            .release_after_shutdown(&Shutdown {
                drained: true,
                error: None,
            })
            .unwrap();
        assert_eq!(worker.scheduler.free_page_count(), free_pages);
        assert_eq!(worker.scheduler.free_snapshot_count(), free_snapshots);
    }
}

/// A worker over a pooled fake store whose scripted hit restores a 70-token prompt,
/// leaving a GPU-ready one-page checkpoint. Returns the sender that keeps the
/// worker's channel open, the request's handle, the checkpoint and its key.
fn worker_with_checkpoint(
    fixture: &Fixture,
    pooled: bool,
) -> (
    Worker<'_>,
    mpsc::Sender<Vec<Incoming>>,
    RequestHandle,
    crate::cache::CheckpointId,
    crate::cache::PrefixKey,
) {
    let (sender, receiver) = mpsc::channel();
    let (incoming, generation) = fixture.incoming(vec![0; 70], 4);
    std::mem::forget(generation);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(Some(host_hit(&history, 70)));
    store.pooled = pooled;
    worker.attach_store(Some(Box::new(store)));
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    let (_, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Fetched {
            key: ticket.key,
            result: Ok(1 << 20),
        });
    assert!(worker.service_store().unwrap());
    let id = worker.scheduler.request(handle).unwrap().prefix().unwrap();
    (worker, sender, handle, id, history.key_at(70).unwrap())
}

fn pins(worker: &Worker<'_>, id: crate::cache::CheckpointId) -> u64 {
    worker.scheduler.cache().checkpoint(id).unwrap().pin_count()
}

fn export_request(
    export: u64,
    key: crate::cache::PrefixKey,
    generation: u64,
    begin_tokens: u64,
) -> flare_selene::StoreEvent {
    flare_selene::StoreEvent::ExportRequested {
        export: flare_selene::ExportId(export),
        key,
        begin_tokens,
        want_row: true,
        reserved_bytes: 1 << 20,
    }
}

fn export_ended(export: u64) -> flare_selene::StoreEvent {
    flare_selene::StoreEvent::ExportEnded {
        export: flare_selene::ExportId(export),
    }
}

#[test]
fn an_export_pins_its_checkpoint_until_it_ends_and_releases_once() {
    let fixture = Fixture::new();
    let (mut worker, _sender, _handle, id, key) = worker_with_checkpoint(&fixture, true);
    let base = pins(&worker, id);
    fake_store(&mut worker)
        .events
        .push_back(export_request(7, key, id.value(), 0));
    worker.service_store().unwrap();
    // The answer comes in the iteration of the request, after the pin.
    assert_eq!(pins(&worker, id), base + 1);
    let store = fake_store(&mut worker);
    assert!(store.refused.is_empty());
    let (export, chain) = store.exports.pop().unwrap();
    assert_eq!(export, flare_selene::ExportId(7));
    assert_eq!(chain.nodes.len(), 1);
    assert_eq!(chain.nodes[0].key, key);
    assert!(chain.row.is_some());
    assert_eq!(worker.exports.len(), 1);
    // A duplicate end for the same export finds nothing: one unpin.
    fake_store(&mut worker).events.push_back(export_ended(7));
    fake_store(&mut worker).events.push_back(export_ended(7));
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), base);
    assert!(worker.exports.is_empty());
}

#[test]
fn an_export_without_an_end_keeps_its_pin() {
    // A quarantined export never reports an end: its pin must outlive everything else.
    let fixture = Fixture::new();
    let (mut worker, _sender, handle, id, key) = worker_with_checkpoint(&fixture, true);
    let store = fake_store(&mut worker);
    store
        .events
        .push_back(export_request(1, key, id.value(), 0));
    worker.service_store().unwrap();
    worker.retire(&[handle]).unwrap();
    for _ in 0..3 {
        worker.service_store().unwrap();
    }
    // The request's pin is gone; the export's remains.
    assert_eq!(pins(&worker, id), 1);
    assert_eq!(worker.exports.len(), 1);
}

#[test]
fn exports_that_cannot_be_served_are_refused_without_a_pin() {
    use flare_selene::ExportRefusal::*;
    let fixture = Fixture::new();
    let (mut worker, _sender, _handle, id, key) = worker_with_checkpoint(&fixture, true);
    let base = pins(&worker, id);
    let mut other = crate::cache::PrefixHistory::new(fixture.config.identity);
    other.append(&[9; 70], &[]).unwrap();
    let store = fake_store(&mut worker);
    store
        .events
        .push_back(export_request(1, other.key_at(70).unwrap(), id.value(), 0));
    store
        .events
        .push_back(export_request(3, key, id.value(), 10_000));
    store
        .events
        .push_back(export_request(4, key, id.value(), 0));
    store
        .events
        .push_back(export_request(4, key, id.value(), 0));
    worker.service_store().unwrap();
    let store = fake_store(&mut worker);
    let refused: Vec<_> = store.refused.iter().map(|(e, r)| (e.0, *r)).collect();
    assert_eq!(
        refused,
        [
            (1, NotFound),
            (3, ChainGap),
            (4, Busy)
        ]
    );
    assert_eq!(store.exports.len(), 1);
    assert_eq!(pins(&worker, id), base + 1);
}

#[test]
fn a_store_that_rejects_a_pinned_export_gets_the_pin_back() {
    let fixture = Fixture::new();
    let (mut worker, _sender, _handle, id, key) = worker_with_checkpoint(&fixture, true);
    let base = pins(&worker, id);
    let store = fake_store(&mut worker);
    store.reject_export = true;
    store
        .events
        .push_back(export_request(1, key, id.value(), 0));
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), base);
    assert!(worker.exports.is_empty());
}

#[test]
fn an_export_is_refused_when_the_next_admission_would_no_longer_fit() {
    let mut fixture = Fixture::new();
    fixture.config.limits.pages = 4;
    fixture.config.limits.max_sequence_tokens = 192;
    let (mut worker, _sender, _handle, id, key) = worker_with_checkpoint(&fixture, true);
    let base = pins(&worker, id);
    // With nothing queued the export fits.
    fake_store(&mut worker)
        .events
        .push_back(export_request(1, key, id.value(), 0));
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), base + 1);
    fake_store(&mut worker).events.push_back(export_ended(1));
    worker.service_store().unwrap();
    // A queued three-page request would not fit, once the checkpoint's page is protected.
    let (queued, _generation) = fixture.incoming(vec![9; 150], 4);
    std::mem::forget(_generation);
    worker.pending.push_back(queued);
    fake_store(&mut worker)
        .events
        .push_back(export_request(2, key, id.value(), 0));
    worker.service_store().unwrap();
    assert_eq!(
        fake_store(&mut worker)
            .refused
            .last()
            .map(|(e, r)| (e.0, *r)),
        Some((2, flare_selene::ExportRefusal::NoHeadroom))
    );
    assert_eq!(pins(&worker, id), base);
}

#[test]
fn only_pooled_stores_hear_of_evictions() {
    for pooled in [true, false] {
        let mut fixture = Fixture::new();
        fixture.config.limits.pages = 3;
        fixture.config.limits.max_sequence_tokens = 192;
        let (mut worker, _sender, handle, _id, key) = worker_with_checkpoint(&fixture, pooled);
        worker.retire(&[handle]).unwrap();
        let publication = fake_store(&mut worker).stored[0].clone();
        let (queued, generation) = fixture.incoming(vec![9; 150], 4);
        std::mem::forget(generation);
        worker.pending.push_back(queued);
        worker.admit().unwrap();
        worker.service_store().unwrap();
        let evicted = &fake_store(&mut worker).evicted;
        assert_eq!(evicted.len(), usize::from(pooled));
        if pooled {
            assert_eq!(evicted[0].0, key);
            assert!(evicted[0].1.is_none());
        }
        assert_eq!(publication.key, key);
    }
}

#[test]
fn a_pooled_lookup_starts_at_arrival_and_ends_at_admission() {
    let fixture = Fixture::new();
    let (sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.pooled = true;
    let withdrawn = store.withdrawn.clone();
    worker.attach_store(Some(Box::new(store)));
    // Five requests arrive for four slots; the fifth keeps waiting.
    let mut generations = Vec::new();
    let mut group = Vec::new();
    for fill in 0..5 {
        let (incoming, generation) = fixture.incoming(vec![fill; 70], 4);
        generations.push(generation);
        group.push(incoming);
    }
    sender.send(group).unwrap();
    worker.admit().unwrap();
    assert_eq!(fake_store(&mut worker).begun, [70; 5]);
    // Admission consulted the ticket and then released it.
    assert_eq!(fake_store(&mut worker).tickets_seen, [true; 4]);
    assert_eq!(withdrawn.load(Ordering::Relaxed), 4);
    assert!(
        worker
            .active
            .iter()
            .flatten()
            .all(|a| a.incoming.pool_lookup.is_none())
    );
    assert_eq!(worker.pending.len(), 1);
    assert!(worker.pending[0].pool_lookup.is_some());
    // Cancelling the waiting request withdraws its lookup with it.
    generations[4].cancelled.store(true, Ordering::Release);
    worker.admit().unwrap();
    assert!(worker.pending.is_empty());
    assert_eq!(withdrawn.load(Ordering::Relaxed), 5);
}

#[test]
fn a_full_gpu_hit_starts_no_pool_lookup_or_admission_hold() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let checkpoint = unpinned_checkpoint(&mut worker, 0, 70);
    let key = worker.scheduler.publication(checkpoint).unwrap().key;
    let mut store = FakeStore::new(None);
    store.pooled = true;
    store.pending_tickets = true;
    worker.attach_store(Some(Box::new(store)));
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    worker.enqueue(vec![incoming]);
    assert!(fake_store(&mut worker).begun.is_empty());
    assert!(worker.pending[0].pool_lookup.is_none());
    assert_eq!(worker.pending[0].pool_plan.as_ref().unwrap().history().token_count(), 70);
    worker.admit().unwrap();
    assert!(worker.active[0].is_some());
    assert!(fake_store(&mut worker).tickets_seen.is_empty());
    assert_eq!(fake_store(&mut worker).reused, [key]);
}

#[test]
fn gpu_residency_is_rechecked_using_the_queued_history() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.pooled = true;
    worker.attach_store(Some(Box::new(store)));
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    worker.enqueue(vec![incoming]);
    assert_eq!(worker.pending[0].pool_plan.as_ref().unwrap().resident_tokens(), 0);
    let checkpoint = unpinned_checkpoint(&mut worker, 0, 70);
    worker.admit().unwrap();
    let handle = worker.active[0].as_ref().unwrap().request;
    assert_eq!(worker.scheduler.request(handle).unwrap().prefix(), Some(checkpoint));
    assert!(fake_store(&mut worker).tickets_seen.is_empty());
}

#[test]
fn late_pool_replies_retry_all_capacity_blocked_requests() {
    use flare_selene::pool::{Answer, LookupState};
    let mut fixture = Fixture::new();
    fixture.config.limits.pages = 3;
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let (first, _first_generation) = fixture.incoming(vec![0; 70], 4);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&first.prompt.tokens, &[]).unwrap();
    let hit = host_hit(&history, 70);
    let mut store = FakeStore::new(Some(hit.clone()));
    store.pooled = true;
    worker.attach_store(Some(Box::new(store)));
    worker.enqueue(vec![first]);
    let _ = worker.admit().unwrap();
    assert_eq!(fake_store(&mut worker).fetched.len(), 1);
    fake_store(&mut worker).hit = None;
    fake_store(&mut worker).pending_tickets = true;
    let (requests, _generations): (Vec<_>, Vec<_>) =
        (0..3).map(|_| fixture.incoming(vec![0; 70], 4)).unzip();
    worker.enqueue(requests);
    for incoming in &mut worker.pending {
        incoming.pool_lookup_started =
            Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    }
    let _ = worker.admit().unwrap();
    assert_eq!(worker.pending.len(), 3);
    let stats = worker.scheduler.stats();
    let resources = (
        worker.scheduler.completed(),
        stats.free_pages,
        stats.free_snapshots,
        stats.free_slots,
    );
    for incoming in &worker.pending {
        assert_eq!(incoming.blocked, Some(resources));
        incoming
            .pool_lookup
            .as_ref()
            .unwrap()
            .set_state(LookupState::Ready(Arc::new(Answer { candidates: vec![] })));
    }
    fake_store(&mut worker).hit = Some(hit);
    let attempts = fake_store(&mut worker).tickets_seen.len();
    let _ = worker.admit().unwrap();
    assert_eq!(fake_store(&mut worker).tickets_seen.len(), attempts + 3);
    assert_eq!(worker.pending.len(), 2);
    assert!(
        worker
            .pending
            .iter()
            .all(|incoming| incoming.pool_lookup_started.is_none())
    );
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert_eq!(fake_store(&mut worker).fetched.len(), 1);
}

#[test]
fn a_local_store_builds_no_lookup_and_a_failed_restore_starts_none() {
    let fixture = Fixture::new();
    let (mut worker, _sender, _handle, _id, _key) = worker_with_checkpoint(&fixture, false);
    assert!(fake_store(&mut worker).begun.is_empty());
    assert_eq!(fake_store(&mut worker).tickets_seen, [false]);

    // A pooled request whose restore fails is readmitted without the store, and
    // without a new lookup.
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(Some(host_hit(&history, 70)));
    store.pooled = true;
    worker.attach_store(Some(Box::new(store)));
    worker.admit().unwrap();
    let (_, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Fetched {
            key: ticket.key,
            result: Err("transfer failed".into()),
        });
    worker.service_store().unwrap();
    assert_eq!(worker.parked.len(), 1);
    assert!(worker.parked[0].0.pool_lookup.is_none());
    assert_eq!(fake_store(&mut worker).begun, [70]);
}

fn complete_pool_answer(worker: &mut Worker<'_>, slot: usize, key: crate::cache::PrefixKey) {
    use flare_selene::{
        levels::Level,
        plan,
        pool::{Answer, LookupState, PoolCandidate, SourceRef, Tier},
    };
    let answer = Answer {
        candidates: vec![PoolCandidate {
            plan: plan::Candidate {
                level: Level::HostNvme,
                source: plan::SourceId(1),
                key,
                begin_tokens: 0,
                skip_pages: 0,
                nodes: vec![plan::ChainNode {
                    key,
                    tail_pages: (key.consumed_tokens / 64) as usize,
                }],
            },
            source: SourceRef {
                instance: vec![1; 16],
                session_incarnation: 1,
            },
            tier: Tier::Nvme,
        }],
    };
    worker.active[slot]
        .as_ref()
        .unwrap()
        .incoming
        .pool_lookup
        .as_ref()
        .unwrap()
        .set_state(LookupState::Ready(Arc::new(answer)));
}

#[test]
fn a_late_pool_answer_retries_admission_only_before_prefill_and_after_drain() {
    use std::time::{Duration, Instant};
    for phase in 0..5 {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.pooled = true;
        store.pending_tickets = true;
        worker.attach_store(Some(Box::new(store)));
        let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
        worker.enqueue(vec![incoming]);
        let hit = host_hit(worker.pending[0].pool_plan.as_ref().unwrap().history(), 70);
        worker.pending[0].pool_lookup_started = Some(Instant::now() - Duration::from_secs(1));
        let _ = worker.admit().unwrap();
        let handle = worker.active[0].as_ref().unwrap().request;
        let mut in_flight = None;
        if phase > 0 {
            let step = planned(&mut worker, GraphKind::Init, 1);
            if phase == 1 {
                in_flight = Some(step);
            } else {
                worker
                    .complete(GraphKind::Init, completion(&worker, &step, 1, None))
                    .unwrap();
            }
        }
        if phase >= 3 {
            let step = planned(&mut worker, GraphKind::Prefill, 2);
            if phase == 3 {
                in_flight = Some(step);
            } else {
                worker
                    .complete(GraphKind::Prefill, completion(&worker, &step, 2, None))
                    .unwrap();
            }
        }
        complete_pool_answer(&mut worker, 0, hit.key);
        fake_store(&mut worker).hit = Some(hit);
        if let Some(step) = in_flight {
            assert!(
                !worker.retry_late_pool_answers().unwrap(),
                "in-flight slot stays owned"
            );
            assert_eq!(worker.active[0].as_ref().unwrap().request, handle);
            worker
                .complete(
                    step.kind,
                    completion(&worker, &step, if phase == 1 { 1 } else { 2 }, None),
                )
                .unwrap();
        }
        if phase < 3 {
            let (step, progress) = worker.plan(0).unwrap();
            assert!(step.is_none() && progress);
            assert!(worker.scheduler.request(handle).is_err());
            let _ = worker.admit().unwrap();
            let next = worker.active[0].as_ref().unwrap().request;
            assert_ne!(next, handle);
            assert_eq!(
                worker.scheduler.request(next).unwrap().phase,
                RequestPhase::Inbound
            );
            assert_eq!(fake_store(&mut worker).fetched.len(), 1);
            assert_eq!(fake_store(&mut worker).begun, [70]);
            assert_eq!(fixture.metrics.admitted.load(Ordering::Relaxed), 1);
            let (_, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
            fake_store(&mut worker)
                .events
                .push_back(flare_selene::StoreEvent::Fetched {
                    key: ticket.key,
                    result: Ok(1 << 20),
                });
            worker.service_store().unwrap();
            let fence = if phase == 0 { 1 } else { 2 };
            advance(&mut worker, GraphKind::Restore, fence, None);
            advance(&mut worker, GraphKind::Resume, fence + 1, Some(1));
            assert_eq!(fixture.metrics.snapshot(false).prefill_tokens, 0);
        } else {
            assert!(!worker.retry_late_pool_answers().unwrap());
            assert_eq!(worker.active[0].as_ref().unwrap().request, handle);
            assert!(
                worker.active[0]
                    .as_ref()
                    .unwrap()
                    .incoming
                    .pool_lookup
                    .is_none()
            );
            assert!(fake_store(&mut worker).fetched.is_empty());
        }
        drop(worker);
        let text = fixture.metrics.render();
        for (outcome, count) in [
            ("no_match", u64::from(phase >= 3)),
            ("admitted", u64::from(phase < 3)),
        ] {
            let line = format!("flare_host_cache_lookup_outcomes_total{{reason=\"{outcome}\"}} {count}\n");
            assert!(text.contains(&line), "missing {line}");
        }
    }
}

#[test]
fn an_idle_worker_holds_a_pending_pool_lookup_until_answer_or_deadline() {
    use flare_selene::pool::{Answer, LookupState};
    use std::time::{Duration, Instant};

    for answer in [
        Some(LookupState::Ready(Arc::new(Answer { candidates: vec![] }))),
        Some(LookupState::None),
        None,
    ] {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let (incoming, _generation) = fixture.incoming(vec![0; 70], 4);
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.pooled = true;
        store.pending_tickets = true;
        let withdrawn = store.withdrawn.clone();
        worker.attach_store(Some(Box::new(store)));
        worker.enqueue(vec![incoming]);
        // Hold the observation fixed even if the test thread is descheduled.
        let started = Instant::now() + Duration::from_secs(60);
        worker.pending[0].pool_lookup_started = Some(started);
        let grace = worker.pending[0].pool_lookup.as_ref().unwrap().grace();
        assert_eq!(grace, Duration::from_millis(2));
        assert!(worker.pending[0].awaits_pool(started + grace - Duration::from_nanos(1)));
        assert!(!worker.pending[0].awaits_pool(started + grace));
        worker.admit().unwrap();
        assert!(worker.active.iter().all(Option::is_none));
        assert_eq!(worker.pending.len(), 1);
        assert_eq!(worker.idle_wait(), Duration::from_micros(100));
        assert_eq!(withdrawn.load(Ordering::Relaxed), 0);

        let pending = answer.is_none();
        if let Some(answer) = answer {
            worker.pending[0]
                .pool_lookup
                .as_ref()
                .unwrap()
                .set_state(answer);
        } else {
            worker.pending[0].pool_lookup_started = Some(Instant::now() - grace);
        }
        worker.admit().unwrap();
        assert!(worker.active[0].is_some());
        assert!(worker.pending.is_empty());
        assert_eq!(worker.idle_wait(), Duration::from_millis(2));
        assert_eq!(withdrawn.load(Ordering::Relaxed), usize::from(!pending));
        if pending {
            assert!(worker.active[0].as_ref().unwrap().incoming.pool_lookup.is_some());
            advance(&mut worker, GraphKind::Init, 1, None);
            let step = planned(&mut worker, GraphKind::Prefill, 2);
            let _ = worker.plan(1).unwrap();
            assert_eq!(withdrawn.load(Ordering::Relaxed), 1);
            assert!(worker.active[0].as_ref().unwrap().incoming.pool_lookup.is_none());
            worker
                .complete(GraphKind::Prefill, completion(&worker, &step, 2, None))
                .unwrap();
        }
    }
}

#[test]
fn a_pending_pool_lookup_does_not_hold_other_requests() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.pooled = true;
    store.pending_tickets = true;
    worker.attach_store(Some(Box::new(store)));
    let (waiting, _waiting_generation) = fixture.incoming(vec![0; 70], 4);
    worker.enqueue(vec![waiting]);
    worker.pending[0].pool_lookup_started =
        Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
    let (ready, _ready_generation) = fixture.incoming(vec![1; 70], 4);
    worker.enqueue(vec![ready]);
    worker.pending[1].pool_lookup.as_ref().unwrap().set_state(flare_selene::pool::LookupState::None);
    worker.admit().unwrap();
    assert!(worker.active[0].is_some());
    assert_eq!(worker.pending.len(), 1);
    assert!(worker.pending[0].pool_lookup.is_some());
}

fn group_runtime(
    fixture: &Fixture,
    capacity: usize,
) -> (ModelRuntime, mpsc::Receiver<Vec<Incoming>>) {
    let (sender, receiver) = mpsc::sync_channel(2);
    (
        ModelRuntime {
            sender,
            stop: Arc::new(AtomicBool::new(false)),
            healthy: Arc::new(AtomicBool::new(true)),
            config: Arc::new(fixture.config.clone()),
            event_capacity: 16,
            capacity: Arc::new(tokio::sync::Semaphore::new(capacity)),
            thread: Mutex::new(None),
            shutdown: Mutex::new(None),
            finished: Arc::new((Mutex::new(false), Condvar::new())),
            metrics: fixture.metrics.clone(),
            input_store: None,
        },
        receiver,
    )
}
#[test]
fn grouped_submission_validates_all_children_before_enqueue_and_releases_capacity() {
    let fixture = Fixture::new();
    let (runtime, receiver) = group_runtime(&fixture, 4);
    let prompt = || {
        Fixture::prompt_for(
            vec![1, 2],
            4,
            PromptData {
                positions: vec![[0; 3], [1; 3]],
                vision_token_indices: vec![-1; 2],
                ..Default::default()
            },
        )
    };
    let good = prompt();
    let mut bad = prompt();
    bad.tokens.clear();
    assert!(runtime.generate_many(vec![good, bad]).is_err());
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert_eq!(runtime.capacity.available_permits(), 4);
    let prompts = (0..3).map(|_| prompt()).collect();
    let generations = runtime.generate_many(prompts).unwrap();
    let group = receiver.try_recv().unwrap();
    assert_eq!(group.len(), 3);
    assert_eq!(runtime.capacity.available_permits(), 1);
    let prompts = (0..2).map(|_| prompt()).collect();
    assert!(runtime.generate_many(prompts).is_err());
    assert!(matches!(
        receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(generations);
    assert!(group.iter().all(|r| r.cancelled.load(Ordering::Acquire)));
    drop(group);
    assert_eq!(runtime.capacity.available_permits(), 4);
}

fn exercise_text_preemption(
    metadata: crate::model::PreparedContract,
    data: impl Fn(usize) -> crate::model::PromptData,
    evict_checkpoint: bool,
) {
    let mut fixture = Fixture::new();
    fixture.config.history_drafts = true;
    fixture.config.contract = metadata;
    fixture.config.prepared_sequence_tokens = 4096;
    fixture.config.limits.max_sequence_tokens = 4096;
    fixture.config.limits.pages = 6;
    fixture.config.limits.max_active = 2;
    let page_tokens = crate::model::scheduler_capabilities(&fixture.config).page_tokens;
    let budget = (page_tokens * 4) as u32;
    let (sender, receiver) = mpsc::channel();
    let (first, mut first_generation) = fixture.incoming_for(vec![0; 60], budget, data(60));
    let (second, mut second_generation) = fixture.incoming_for(vec![2; 60], budget, data(60));
    sender.send(vec![first, second]).unwrap();
    let mut worker = fixture.worker_for(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert_eq!(worker.scheduler.free_page_count(), 2);
    let older = worker.active[0].as_ref().unwrap().request;
    let mut sequence = 0;
    while worker.parked.is_empty() {
        sequence += 1;
        assert!(sequence < (page_tokens * 8) as u64, "no request was parked");
        let step = worker
            .plan(0)
            .unwrap()
            .0
            .expect("page pressure must make progress");
        worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
        let sample = match step.kind {
            GraphKind::Decode | GraphKind::Resume => Some(1),
            GraphKind::Prefill => {
                let lane = &step.lanes[0];
                let live = worker.scheduler.request(lane.request).unwrap();
                (lane.start_position as usize + lane.tokens == live.prompt_tokens).then_some(1)
            }
            _ => None,
        };
        let frame = completion(&worker, &step, sequence, sample);
        worker.complete(step.kind, frame).unwrap();
        worker.publish().unwrap();
        for generation in [&mut first_generation, &mut second_generation] {
            while let Ok(event) = generation.events.try_recv() {
                assert!(!matches!(
                    event,
                    GenerationEvent::Error(_) | GenerationEvent::AdmissionRejected(_)
                ));
            }
        }
    }
    assert!(worker.scheduler.stats().preemptions > 0);
    let (incoming, parked) = worker.parked.front_mut().unwrap();
    assert!(parked.carried.generated_tokens > 0);
    let decoded_sequence = parked.carried.decoded_sequence;
    assert_ne!(decoded_sequence, Default::default());
    let history_len = incoming.prompt.tokens.len();
    assert_eq!(history_len, 60 + parked.carried.generated_tokens as usize);
    assert_eq!(
        incoming.prompt.max_new_tokens + parked.carried.generated_tokens,
        budget
    );
    crate::model::validate_prompt_data(
        &fixture.config,
        &incoming.prompt.tokens,
        &incoming.prompt.data,
    )
    .unwrap();
    let metadata = crate::model::prompt_metadata(&mut incoming.prompt.data);
    if let Some(positions) = metadata.positions {
        assert_eq!(positions.last(), Some(&[history_len as i64 - 1; 3]));
    }
    if let Some(indices) = metadata.vision_token_indices {
        assert_eq!(indices, &vec![-1; history_len]);
    }
    worker.retire(&[older]).unwrap();
    if evict_checkpoint {
        let pressure = worker
            .scheduler
            .admit(RequestSpec {
                prompt: vec![3; page_tokens * 5],
                max_new_tokens: 1,
                growable_output: false,
                message_boundaries: vec![],
                conditioning: vec![],
                grammar_constrained: false,
                host_stops: false,
                prompt_logprobs: false,
            })
            .unwrap();
        worker.scheduler.cancel(pressure).unwrap();
    }
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.parked.is_empty());
    let resumed = worker.active.iter().flatten().next().unwrap().request;
    let live = worker.scheduler.request(resumed).unwrap();
    assert_eq!(live.decoded_sequence, decoded_sequence);
    if evict_checkpoint {
        assert!(live.consumed_tokens < history_len as u64 - 1);
    } else {
        assert_eq!(live.phase, RequestPhase::Restore);
        assert_eq!(live.consumed_tokens, history_len as u64 - 1);
    }
    let mut recomputed = false;
    for _ in 0..history_len / 64 + 8 {
        sequence += 1;
        let step = worker.plan(0).unwrap().0.unwrap();
        recomputed |= step.kind == GraphKind::Prefill
            && step.lanes[0].start_position < history_len as u64 - 1;
        worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
        let sample = match step.kind {
            GraphKind::Decode | GraphKind::Resume => Some(1),
            GraphKind::Prefill => {
                let lane = &step.lanes[0];
                (lane.start_position as usize + lane.tokens == history_len).then_some(1)
            }
            _ => None,
        };
        let frame = completion(&worker, &step, sequence, sample);
        worker.complete(step.kind, frame).unwrap();
        worker.publish().unwrap();
        if worker.scheduler.request(resumed).unwrap().generated_tokens > 0 {
            assert_eq!(recomputed, evict_checkpoint);
            assert!(
                worker.active[resumed.slot as usize]
                    .as_ref()
                    .unwrap()
                    .carried
                    .generated_tokens
                    > 0
            );
            return;
        }
    }
    panic!("readmitted request did not generate output");
}

#[test]
fn text_preemption_restores_or_recomputes_every_attention_contract() {
    for page_tokens in [64, 128, 256] {
        for evict in [false, true] {
            let mut metadata = crate::model::testing::metadata();
            metadata.cache.page_tokens = page_tokens;
            exercise_text_preemption(
                metadata,
                |n| PromptData {
                    positions: (0..n).map(|p| [p as i64; 3]).collect(),
                    vision_token_indices: vec![-1; n],
                    ..Default::default()
                },
                evict,
            );
        }
    }
}

#[test]
fn text_and_conditioned_admission_reserve_prompt_and_initial_output_pages() {
    let mut fixture = Fixture::new();
    fixture.config.prepared_sequence_tokens = 2048;
    fixture.config.limits.max_sequence_tokens = 2048;
    fixture.config.limits.pages = 4;
    let (sender, receiver) = mpsc::channel();
    let (first, _first_generation) = fixture.incoming(vec![0; 60], 1024);
    let (second, _second_generation) = fixture.incoming(vec![1; 60], 1024);
    let (third, mut third_generation) = fixture.incoming(vec![2; 60], 1024);
    sender.send(vec![first, second, third]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.active.iter().flatten().count(), 2);
    assert_eq!(worker.pending.len(), 1);
    assert!(third_generation.events.try_recv().is_err());
    let first = worker.active[0].as_ref().unwrap().request;
    worker.retire(&[first]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.pending.is_empty());
    assert_eq!(worker.active.iter().flatten().count(), 2);

    let (sender, receiver) = mpsc::channel();
    let (mut conditioned, mut generation) = fixture.incoming(vec![0; 60], 1024);
    conditioned.prompt.conditioning.push(ConditioningEvent {
        position: 0,
        identity: [3; 32],
    });
    sender.send(vec![conditioned]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.active.iter().flatten().count(), 1);
    assert_eq!(worker.scheduler.free_page_count(), 2);
    assert!(generation.events.try_recv().is_err());
    assert!(generation.admission.try_recv().unwrap().is_ok());
}

#[test]
fn prompt_reservations_wait_for_capacity_without_blocking_smaller_requests() {
    for page_tokens in [64, 128, 256] {
        let mut fixture = Fixture::new();
        fixture.config.contract.cache.page_tokens = page_tokens;
        fixture.config.prepared_sequence_tokens = 2048;
        fixture.config.limits.max_sequence_tokens = 2048;
        fixture.config.limits.pages = 4;
        let (sender, receiver) = mpsc::channel();
        let (first, mut first_generation) = fixture.incoming(vec![0; 60], 1024);
        let (mut waiting, mut waiting_generation) =
            fixture.incoming(vec![1; page_tokens * 2 - 1], (page_tokens * 2) as u32);
        waiting.prompt.conditioning.push(ConditioningEvent {
            position: 0,
            identity: [3; 32],
        });
        let (small, mut small_generation) = fixture.incoming(vec![2], 1);
        sender.send(vec![first, waiting, small]).unwrap();
        let mut worker = fixture.worker(receiver);
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        assert_eq!(worker.active.iter().flatten().count(), 2);
        assert_eq!(worker.pending.len(), 1);
        assert_eq!(worker.scheduler.free_page_count(), 1);
        assert!(first_generation.admission.try_recv().unwrap().is_ok());
        assert!(small_generation.admission.try_recv().unwrap().is_ok());
        assert!(matches!(
            waiting_generation.admission.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(waiting_generation.events.try_recv().is_err());
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(false));
        let first = worker.active[0].as_ref().unwrap().request;
        worker.retire(&[first]).unwrap();
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        assert!(worker.pending.is_empty());
        assert!(waiting_generation.admission.try_recv().unwrap().is_ok());
        assert_eq!(worker.scheduler.free_page_count(), 0);
        assert_eq!(worker.metrics.failed.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn cancelling_a_capacity_waiter_releases_it_without_admission() {
    let mut fixture = Fixture::new();
    fixture.config.limits.pages = 2;
    let (sender, receiver) = mpsc::channel();
    let (first, _first_generation) = fixture.incoming(vec![0; 60], 60);
    let (waiting, waiting_generation) = fixture.incoming(vec![1], 1);
    sender.send(vec![first, waiting]).unwrap();
    let mut worker = fixture.worker(receiver);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.pending.len(), 1);
    drop(waiting_generation);
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.pending.is_empty());
    assert_eq!(worker.metrics.admitted.load(Ordering::Relaxed), 1);
    assert_eq!(worker.metrics.cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(worker.metrics.failed.load(Ordering::Relaxed), 0);
}

/// HTTP state over an idle model runtime whose requests arrive on `sender`.
fn http_state(
    sender: mpsc::SyncSender<Vec<Incoming>>,
    api_key: Option<String>,
) -> Arc<super::super::HttpState> {
    use super::super::HttpState;
    use tokio::sync::Semaphore;
    let mut fixture = Fixture::new();
    fixture.config.contract.vocabulary = 6;
    fixture.config.contract.input_vocabulary = 6;
    fixture.config.sampling_defaults.top_k = -1;
    let grammar = crate::grammar::GrammarCompiler::new(
        fixture.frontend.tokenizer(),
        6,
        &fixture.config.eos_token_ids,
        &[],
    )
    .unwrap();
    let model = Arc::new(ModelRuntime {
        sender,
        stop: Arc::new(AtomicBool::new(false)),
        healthy: Arc::new(AtomicBool::new(true)),
        config: Arc::new(fixture.config),
        event_capacity: 64,
        capacity: Arc::new(Semaphore::new(4)),
        thread: Mutex::new(None),
        shutdown: Mutex::new(Some(Shutdown {
            drained: true,
            error: None,
        })),
        finished: Arc::new((Mutex::new(true), Condvar::new())),
        metrics: fixture.metrics,
        input_store: None,
    });
    Arc::new(HttpState {
        model,
        frontend: Arc::new(fixture.frontend),
        images: None,
        grammar: Arc::new(grammar),
        frontend_slots: Arc::new(Semaphore::new(1)),
        ingress_slots: Arc::new(Semaphore::new(4)),
        max_body_bytes: 65536,
        frontend_capacity: 1,
        ingress_capacity: 4,
        ids: Default::default(),
        api_key,
        context_hint: String::new(),
        capacity_hint: String::new(),
        started: std::time::Instant::now(),
        generated: Default::default(),
    })
}

async fn http_generation(
    stream: bool,
    choices: usize,
) -> (
    tokio::task::JoinHandle<axum::response::Response>,
    Vec<Incoming>,
) {
    http_generation_request(
        "/v1/completions",
        serde_json::json!({
            "model": "qwen-worker-phases", "prompt": "a", "max_tokens": 2,
            "stream": stream, "n": choices,
        }),
    )
    .await
}

async fn http_generation_request(
    path: &str,
    body: serde_json::Value,
) -> (
    tokio::task::JoinHandle<axum::response::Response>,
    Vec<Incoming>,
) {
    use super::super::http;
    use axum::{Extension, extract::State};
    let (sender, receiver) = mpsc::sync_channel(4);
    let state = http_state(sender, None);
    let ingress = Arc::new(state.ingress_slots.clone().acquire_owned().await.unwrap());
    let request = axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    let response = tokio::spawn(http::generate(State(state), Extension(ingress), request));
    let incoming = tokio::task::spawn_blocking(move || {
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap()
    })
    .await
    .unwrap();
    (response, incoming)
}

#[tokio::test]
async fn http_admission_rejections_use_json_or_terminal_stream_errors() {
    use axum::http::StatusCode;
    for stream in [false, true] {
        for (error, status, code) in [
            (
                Error::admission(AdmissionError::Capacity, "full"),
                StatusCode::SERVICE_UNAVAILABLE,
                "capacity_exhausted",
            ),
            (
                Error::admission(AdmissionError::Unavailable, "stopped"),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_unavailable",
            ),
            (
                Error::admission(AdmissionError::Context, "too long"),
                StatusCode::BAD_REQUEST,
                "context_length_exceeded",
            ),
        ] {
            let (mut response, incoming) = http_generation(stream, 1).await;
            let streaming_response = if stream {
                Some(
                    tokio::time::timeout(std::time::Duration::from_secs(5), &mut response)
                        .await
                        .unwrap()
                        .unwrap(),
                )
            } else {
                assert!(futures_util::poll!(&mut response).is_pending());
                None
            };
            incoming[0].reject_admission(error);
            let response = match streaming_response {
                Some(response) => response,
                None => response.await.unwrap(),
            };
            assert_eq!(
                response.status(),
                if stream { StatusCode::OK } else { status }
            );
            assert_eq!(
                response.headers()["content-type"],
                if stream {
                    "text/event-stream"
                } else {
                    "application/json"
                }
            );
            assert!(
                response.headers()["x-request-id"]
                    .to_str()
                    .unwrap()
                    .starts_with("cmpl-")
            );
            let bytes = axum::body::to_bytes(response.into_body(), 65536)
                .await
                .unwrap();
            let body = std::str::from_utf8(&bytes).unwrap();
            let body: serde_json::Value = if stream {
                assert!(!body.contains("[DONE]"), "{body}");
                serde_json::from_str(
                    body.lines()
                        .find_map(|line| line.strip_prefix("data: "))
                        .unwrap(),
                )
                .unwrap()
            } else {
                serde_json::from_str(body).unwrap()
            };
            assert_eq!(body["error"]["code"], code);
            assert!(body["error"]["param"].is_null());
            assert!(incoming[0].cancelled.load(Ordering::Acquire));
        }
    }
}

#[tokio::test]
async fn http_stream_headers_precede_admission_and_late_failure_has_no_success_terminal() {
    let (response, incoming) = http_generation(true, 2).await;
    let response = tokio::time::timeout(std::time::Duration::from_secs(5), response)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert!(!incoming[1].cancelled.load(Ordering::Acquire));
    incoming[0].notify_admission(Ok(()));
    incoming[0].emit_token(1);
    incoming[1].reject_admission(Error::admission(
        AdmissionError::Unavailable,
        "worker stopped",
    ));
    let bytes = axum::body::to_bytes(response.into_body(), 65536)
        .await
        .unwrap();
    let body = std::str::from_utf8(&bytes).unwrap();
    assert!(body.contains("\"text\":\"b\""), "{body}");
    assert!(body.contains("\"code\":\"server_unavailable\""), "{body}");
    assert!(!body.contains("[DONE]"), "{body}");
    assert!(!body.contains("\"finish_reason\":\"stop\""), "{body}");
    assert!(incoming.iter().all(|i| i.cancelled.load(Ordering::Acquire)));
}

#[tokio::test]
async fn chat_stream_role_waits_for_final_prefill_even_when_stop_matching_hides_tokens() {
    use futures_util::StreamExt;
    for (hidden, eos) in [(false, false), (true, false), (false, true)] {
        let mut request = serde_json::json!({
            "messages": [{"role": "user", "content": "a".repeat(70)}],
            "max_tokens": 4, "stream": true,
        });
        if hidden {
            request["stop_token_sequences"] = serde_json::json!([[1, 2]]);
        }
        let (response, mut incoming) =
            http_generation_request("/v1/chat/completions", request).await;
        let response = tokio::time::timeout(std::time::Duration::from_secs(5), response)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        assert!(futures_util::poll!(body.next()).is_pending());

        let fixture = Fixture::new();
        // The fixture's graph bindings include an image-index view for text rows.
        incoming[0].prompt.data.vision_token_indices = vec![-1; incoming[0].prompt.tokens.len()];
        let (sender, receiver) = mpsc::channel();
        sender.send(incoming).unwrap();
        let mut worker = fixture.worker(receiver);
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        assert!(futures_util::poll!(body.next()).is_pending());
        advance(&mut worker, GraphKind::Init, 1, None);
        advance(&mut worker, GraphKind::Prefill, 2, None);
        assert!(futures_util::poll!(body.next()).is_pending());
        let step = planned(&mut worker, GraphKind::Prefill, 3);
        let mut frame = completion(&worker, &step, 3, Some(if eos { 5 } else { 1 }));
        if eos {
            frame.records[0].status = 2;
        }
        worker.complete(GraphKind::Prefill, frame).unwrap();

        let role = tokio::time::timeout(std::time::Duration::from_secs(5), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let role: serde_json::Value = serde_json::from_str(
            std::str::from_utf8(&role)
                .unwrap()
                .trim()
                .strip_prefix("data: ")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            role["choices"][0]["delta"],
            serde_json::json!({"role":"assistant","content":""})
        );
        if hidden || eos {
            assert!(futures_util::poll!(body.next()).is_pending());
        } else {
            let text = body.next().await.unwrap().unwrap();
            assert!(
                std::str::from_utf8(&text)
                    .unwrap()
                    .contains("\"content\":\"b\"")
            );
        }
        let cancelled = worker.active[0]
            .as_ref()
            .unwrap()
            .incoming
            .cancelled
            .clone();
        drop(body);
        assert!(cancelled.load(Ordering::Acquire));
    }
}

#[tokio::test]
async fn chat_stream_failure_before_first_token_emits_only_error() {
    for admission in [false, true] {
        let (response, incoming) = http_generation_request(
            "/v1/chat/completions",
            serde_json::json!({
                "messages": [{"role": "user", "content": "a"}],
                "max_tokens": 2, "stream": true,
            }),
        )
        .await;
        let response = response.await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        if admission {
            incoming[0].reject_admission(Error::admission(
                AdmissionError::Unavailable,
                "worker stopped",
            ));
        } else {
            incoming[0].notify_admission(Ok(()));
            incoming[0].fail(Error::execution(ExecutionError::Worker, "worker stopped"));
        }
        let bytes = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            axum::body::to_bytes(response.into_body(), 65536),
        )
        .await
        .unwrap()
        .unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(body.contains("worker stopped"), "{body}");
        assert_eq!(
            body.lines()
                .filter(|line| line.starts_with("data:"))
                .count(),
            1
        );
        assert!(!body.contains("\"role\""), "{body}");
        assert!(!body.contains("[DONE]"), "{body}");
        assert!(incoming[0].cancelled.load(Ordering::Acquire));
    }
}

#[tokio::test]
async fn chat_stream_disconnect_before_first_token_cancels_generation() {
    use futures_util::StreamExt;
    let (response, incoming) = http_generation_request(
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "a"}], "max_tokens": 2, "stream": true,
        }),
    )
    .await;
    let mut body = response.await.unwrap().into_body().into_data_stream();
    assert!(futures_util::poll!(body.next()).is_pending());
    drop(body);
    assert!(incoming[0].cancelled.load(Ordering::Acquire));
}

#[tokio::test]
async fn chat_stream_flushes_headers_over_tcp_before_admission_or_tokens() {
    use std::io::{Read, Write};
    let (sender, receiver) = mpsc::sync_channel(4);
    let state = http_state(sender, None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = super::super::router(state, &[], 65536);
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let client = tokio::task::spawn_blocking(move || {
        let mut socket = std::net::TcpStream::connect(address).unwrap();
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "a"}], "max_tokens": 2, "stream": true,
        })
        .to_string();
        write!(socket, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut headers = Vec::new();
        while !headers.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            headers.push(byte[0]);
        }
        let headers = std::str::from_utf8(&headers).unwrap();
        assert!(headers.starts_with("HTTP/1.1 200 OK\r\n"), "{headers}");
        assert!(headers.contains("text/event-stream"), "{headers}");
        socket
            .set_read_timeout(Some(std::time::Duration::from_millis(100)))
            .unwrap();
        let error = socket.read(&mut [0]).unwrap_err();
        assert!(matches!(
            error.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ));
        socket
    });
    let incoming = tokio::task::spawn_blocking(move || {
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap()
    })
    .await
    .unwrap();
    let mut socket = client.await.unwrap();
    incoming[0].notify_admission(Ok(()));
    incoming[0].observe_token(1);
    incoming[0].emit_token(1);
    tokio::task::spawn_blocking(move || {
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut body = String::new();
        while !body.contains("\"content\":\"b\"") {
            let mut bytes = [0; 4096];
            let length = socket.read(&mut bytes).unwrap();
            assert_ne!(length, 0, "{body}");
            body.push_str(std::str::from_utf8(&bytes[..length]).unwrap());
        }
        let role = body.find("\"role\":\"assistant\"").unwrap();
        assert!(role < body.find("\"content\":\"b\"").unwrap(), "{body}");
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn http_disconnect_while_waiting_cancels_queued_generation() {
    let (response, incoming) = http_generation(true, 1).await;
    drop(response.await.unwrap());
    assert!(incoming[0].cancelled.load(Ordering::Acquire));
}

#[tokio::test]
async fn ollama_discovery_uses_bearer_auth_without_generation_admission() {
    use std::io::{Read, Write};
    let (sender, receiver) = mpsc::sync_channel(4);
    let state = http_state(sender, Some("serving-key".into()));
    let permits = state
        .ingress_slots
        .clone()
        .acquire_many_owned(4)
        .await
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = super::super::router(state, &["discovery-model".into()], 1024);
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let result = tokio::task::spawn_blocking(move || {
        for (method, path, body) in [
            ("GET", "/api/tags", ""),
            ("POST", "/api/show", r#"{"model":"discovery-model"}"#),
        ] {
            for (auth, expected) in [
                ("", 401),
                ("Authorization: Bearer wrong-key\r\n", 401),
                ("x-api-key: serving-key\r\n", 401),
                ("Authorization: Bearer serving-key\r\n", 200),
            ] {
                let mut stream = std::net::TcpStream::connect(address).unwrap();
                stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
                write!(stream,
                    "{method} {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len(),
                ).unwrap();
                let mut response = String::new();
                stream.read_to_string(&mut response).unwrap();
                let (head, body) = response.split_once("\r\n\r\n").unwrap();
                let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
                assert_eq!(status, expected, "{path}: {body}");
                let value: serde_json::Value = serde_json::from_str(body).unwrap();
                if status == 200 {
                    if path == "/api/tags" {
                        assert_eq!(value["models"][0]["model"], "discovery-model");
                    } else {
                        assert_eq!(value["model_info"]["general.name"], "discovery-model");
                    }
                }
            }
        }
        assert!(receiver.try_recv().is_err());
    }).await;
    server.abort();
    drop(permits);
    result.unwrap();
}

#[tokio::test]
async fn metrics_route_serves_prometheus_or_json_without_the_api_key() {
    use std::io::{Read, Write};
    let (sender, _receiver) = mpsc::sync_channel(4);
    let state = http_state(sender, Some("serving-key".into()));
    state.model.metrics().progress(10, 3, 0, 0, 0, 0);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = super::super::router(state, &[], 65536);
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let get = |path: &'static str, accept: Option<&'static str>| {
        tokio::task::spawn_blocking(move || {
            let mut stream = std::net::TcpStream::connect(address).unwrap();
            let accept = accept.map_or_else(String::new, |accept| format!("Accept: {accept}\r\n"));
            write!(
                stream,
                "GET {path} HTTP/1.1\r\nHost: localhost\r\n{accept}Connection: close\r\n\r\n"
            )
            .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            let (head, body) = response.split_once("\r\n\r\n").unwrap();
            let status: u16 = head.split(' ').nth(1).unwrap().parse().unwrap();
            let content_type = head
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-type: ")
                        .map(str::to_owned)
                })
                .unwrap_or_default();
            (status, content_type, body.to_owned())
        })
    };
    let (status, content_type, body) = get("/metrics", None).await.unwrap();
    assert_eq!(status, 200);
    assert!(
        content_type.starts_with("text/plain; version=0.0.4"),
        "{content_type}"
    );
    assert!(
        body.contains("\nflare_generated_tokens_total 3\n"),
        "{body}"
    );
    assert!(body.contains("\nflare_prefill_tokens_total 10\n"));
    for (path, accept) in [
        ("/metrics", Some("application/json")),
        ("/metrics?format=json", Some("text/plain")),
    ] {
        let (status, content_type, body) = get(path, accept).await.unwrap();
        assert_eq!(
            (status, content_type.as_str()),
            (200, "application/json"),
            "{path}"
        );
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["learned_drafter"], false);
        let snapshot: super::super::Snapshot = serde_json::from_value(value).unwrap();
        assert_eq!(
            (snapshot.prefill_tokens, snapshot.generated_tokens),
            (10, 3)
        );
    }
    assert_eq!(get("/metrics?format=yaml", None).await.unwrap().0, 400);
    assert_eq!(get("/health", None).await.unwrap().0, 200);
    assert_eq!(get("/v1/models", None).await.unwrap().0, 401);
    server.abort();
}

fn described_image(
    span: std::ops::Range<usize>,
    features: std::ops::Range<usize>,
) -> crate::frontend::images::PreparedImageInput {
    use crate::frontend::images::{
        ImageDescription, ImageEncoding, ImageGeometry, ImageInput, ImageLimits, ImageProcessor,
        ImageSettings, PreparedImageInput,
    };
    let geometry = ImageGeometry {
        height: 112,
        width: 112,
        grid_thw: [1, 8, 8],
        patches: 64,
        tokens: 16,
    };
    let processor =
        Arc::new(ImageProcessor::load(ImageSettings::Moonvit {}, ImageLimits::default()).unwrap());
    PreparedImageInput::new(
        ImageInput {
            description: ImageDescription {
                identity: [span.start as u8; 32],
                width: 112,
                height: 112,
                geometry,
            },
            source: processor.encoded_source(
                ImageEncoding::Png,
                Arc::from(&b"invalid PNG must stay unprocessed on a prefix hit"[..]),
                None,
            ),
        },
        span,
        features,
        processor,
    )
}

#[test]
fn host_prefix_selects_whole_required_images_and_compacts_feature_indices() {
    for (consumed, expected) in [(18, 2), (26, 1), (55, 1), (56, 0), (100, 0)] {
        let mut fixture = Fixture::new();
        fixture.config.contract.vision = crate::model::VisionCapabilities {
            feature_slots: 1,
            features_per_request: 64,
            max_images: 4,
            patches_per_feature: 4,
            patch_capacities: vec![256],
        };
        fixture.config.contract.frontend =
            serde_json::from_str::<crate::model::contract::PreparedContract>(include_str!(
                "../../../tests/support/contracts/qwen38.json"
            ))
            .unwrap()
            .frontend;
        let (sender, receiver) = mpsc::channel();
        let (mut incoming, _generation) = fixture.incoming(vec![0; 100], 4);
        incoming.prompt.conditioned_inputs = vec![
            described_image(10..26, 0..16),
            described_image(40..56, 16..32),
        ];
        let canonical = incoming.prompt.data.vision_token_indices.clone();
        let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
        history.append(&incoming.prompt.tokens, &[]).unwrap();
        sender.send(vec![incoming]).unwrap();
        let mut worker = fixture.worker(receiver);
        if expected == 0 {
            let occupied = worker
                .scheduler
                .admit(RequestSpec {
                    prompt: vec![1; 2],
                    message_boundaries: vec![],
                    conditioning: vec![],
                    max_new_tokens: 1,
                    grammar_constrained: false,
                    host_stops: false,
                    prompt_logprobs: false,
                    growable_output: true,
                })
                .unwrap();
            worker.scheduler.set_vision_images(occupied, &[64]).unwrap();
            assert!(!worker.scheduler.has_available_image_slot());
        }
        worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(
            &history, consumed,
        ))))));
        assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
        let active = worker.active.iter().flatten().next().unwrap();
        let live = worker.scheduler.request(active.request).unwrap();
        assert_eq!(live.consumed_tokens, consumed);
        assert_eq!(active.selected_images.len(), expected);
        assert_eq!(live.image_slot.is_some(), expected != 0);
        assert_eq!(active.incoming.prompt.data.vision_token_indices, canonical);
        if expected == 1 {
            assert_eq!(active.selected_images[0].feature_range, 16..32);
            assert_eq!(live.feature_index(16).unwrap(), 0);
            assert_eq!(live.feature_index(31).unwrap(), 15);
        }
        if expected == 0 {
            assert!(active.prepared_image.is_none());
            assert!(worker.image_preparation.is_none());
        }
    }
}

#[test]
fn failed_host_restore_replans_all_images_from_the_shallower_prefix() {
    let mut fixture = Fixture::new();
    fixture.config.contract.vision = crate::model::VisionCapabilities {
        feature_slots: 1,
        features_per_request: 64,
        max_images: 4,
        patches_per_feature: 4,
        patch_capacities: vec![256],
    };
    fixture.config.contract.frontend =
        serde_json::from_str::<crate::model::contract::PreparedContract>(include_str!(
            "../../../tests/support/contracts/qwen38.json"
        ))
        .unwrap()
        .frontend;
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, _generation) = fixture.incoming(vec![0; 100], 4);
    incoming.prompt.conditioned_inputs = vec![
        described_image(10..26, 0..16),
        described_image(40..56, 16..32),
    ];
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    sender.send(vec![incoming]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(
        &history, 100,
    ))))));
    let _ = worker.admit().unwrap();
    assert_eq!(worker.active[0].as_ref().unwrap().selected_images.len(), 0);
    worker.active[0].as_mut().unwrap().carried = Carried {
        decoded_sequence: Default::default(),
        prompt_tokens: Some(90),
        reused_prompt_tokens: Some(56),
        generated_tokens: 7,
        seed: Some(99),
    };
    worker.active[0].as_mut().unwrap().last_token = Some(2);
    let (_, ticket) = fake_store(&mut worker).fetched.pop().unwrap();
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Fetched {
            key: ticket.key,
            result: Err("restore transfer failed".into()),
        });
    worker.service_store().unwrap();
    let _ = worker.admit().unwrap();
    let active = worker.active.iter().flatten().next().unwrap();
    assert_eq!(
        worker
            .scheduler
            .request(active.request)
            .unwrap()
            .consumed_tokens,
        0
    );
    assert_eq!(active.selected_images.len(), 2);
    assert_eq!(active.next_image, 0);
    assert!(active.incoming.readmitted);
    assert!(!active.incoming.host_lookup);
    assert_eq!(active.carried.prompt_tokens, Some(90));
    assert_eq!(active.carried.reused_prompt_tokens, Some(56));
    assert_eq!(active.carried.generated_tokens, 7);
    assert_eq!(active.carried.seed, Some(99));
    assert_eq!(active.last_token, Some(2));
}

#[test]
fn image_materialization_failure_retires_only_its_request() {
    let mut fixture = Fixture::new();
    fixture.config.contract.vision = crate::model::VisionCapabilities {
        feature_slots: 1,
        features_per_request: 64,
        max_images: 4,
        patches_per_feature: 4,
        patch_capacities: vec![256],
    };
    fixture.config.contract.frontend =
        serde_json::from_str::<crate::model::contract::PreparedContract>(include_str!(
            "../../../tests/support/contracts/qwen38.json"
        ))
        .unwrap()
        .frontend;
    let (sender, receiver) = mpsc::channel();
    let (mut image, mut image_generation) = fixture.incoming(vec![0; 30], 4);
    image.prompt.conditioned_inputs = vec![described_image(10..26, 0..16)];
    let (text, _text_generation) = fixture.incoming(vec![1; 30], 4);
    sender.send(vec![image, text]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.image_preparation = Some(ImagePreparation::new(None));
    let _ = worker.admit().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while worker.active.iter().flatten().count() != 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "image preparation did not return its failure"
        );
        let _ = worker.admit().unwrap();
        std::thread::yield_now();
    }
    let text = worker.active.iter().flatten().next().unwrap();
    assert!(text.incoming.prompt.conditioned_inputs.is_empty());
    assert!(worker.scheduler.has_available_image_slot());
    assert!(matches!(
        image_generation.events.try_recv().unwrap(),
        GenerationEvent::Error(_)
    ));
    assert_eq!(worker.metrics.failed.load(Ordering::Relaxed), 1);
}

#[test]
fn prepared_image_window_stays_global_until_pack_or_cancellation() {
    use crate::frontend::images::{
        ImageInput, ImageLimits, ImageProcessor, ImageSettings, ImageSource, PreparedImageInput,
        RgbImage,
    };
    let mut fixture = Fixture::new();
    fixture.config.contract.vision = crate::model::VisionCapabilities {
        feature_slots: 2,
        features_per_request: 64,
        max_images: 4,
        patches_per_feature: 4,
        patch_capacities: vec![256],
    };
    fixture.config.contract.frontend = serde_json::from_str::<crate::model::PreparedContract>(
        include_str!("../../../tests/support/contracts/qwen38.json"),
    )
    .unwrap()
    .frontend;
    let processor =
        Arc::new(ImageProcessor::load(ImageSettings::Moonvit {}, ImageLimits::default()).unwrap());
    let pixels = Arc::new(RgbImage::new(112, 112, vec![42; 112 * 112 * 3]).unwrap());
    let descriptor = PreparedImageInput::new(
        ImageInput {
            description: processor.describe_image(&pixels).unwrap(),
            source: ImageSource::Decoded(pixels),
        },
        10..26,
        0..16,
        processor.clone(),
    );
    let (sender, receiver) = mpsc::channel();
    let (mut first, first_generation) = fixture.incoming(vec![0; 30], 4);
    first.prompt.conditioned_inputs = vec![descriptor.clone()];
    let (mut second, _second_generation) = fixture.incoming(vec![1; 30], 4);
    second.prompt.conditioned_inputs = vec![descriptor];
    sender.send(vec![first, second]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.image_preparation = Some(ImagePreparation::new(None));
    let _ = worker.admit().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while worker.active[0].as_ref().unwrap().prepared_image.is_none() {
        assert!(std::time::Instant::now() < deadline);
        let _ = worker.admit().unwrap();
        std::thread::yield_now();
    }
    for _ in 0..10 {
        let _ = worker.admit().unwrap();
    }
    assert_eq!(processor.preparation_counts().prepared_images, 1);
    assert!(worker.active[1].as_ref().unwrap().prepared_image.is_none());
    first_generation.cancelled.store(true, Ordering::Release);
    while worker.active[1].as_ref().unwrap().prepared_image.is_none() {
        assert!(std::time::Instant::now() < deadline);
        let _ = worker.admit().unwrap();
        std::thread::yield_now();
    }
    assert!(worker.active[0].is_none());
    assert_eq!(processor.preparation_counts().prepared_images, 2);
    assert_eq!(
        worker.preparing_image,
        Some(worker.active[1].as_ref().unwrap().request)
    );
}

#[test]
fn worker_restores_conditioned_windows_with_fresh_generation_and_unchanged_sampling() {
    use crate::execution::{BufferDescriptor, Dtype, GraphDescriptor, InputBinding, SamplingMode};
    use crate::frontend::images::{
        ImageInput, ImageLimits, ImageProcessor, ImageSettings, ImageSource, PreparedImageInput,
        QwenImageSettings, RgbImage,
    };
    use crate::model::contract::{GraphContract, GraphOperation, InputIndexing, ValueSource};
    let mut fixture = Fixture::new();
    fixture.config.limits.max_sequence_tokens = 512;
    fixture.config.prepared_sequence_tokens = 512;
    fixture.config.contract.vision = crate::model::VisionCapabilities {
        feature_slots: 1,
        features_per_request: 1,
        max_images: 4,
        patches_per_feature: 4,
        patch_capacities: vec![256],
    };
    fixture.config.contract.frontend = serde_json::from_str::<crate::model::PreparedContract>(
        include_str!("../../../tests/support/contracts/qwen38.json"),
    )
    .unwrap()
    .frontend;
    let mut manifest = crate::model::testing::manifest(&fixture.config);
    let mut bindings = vec![];
    let mut semantic = vec![];
    for (index, (name, source, dtype)) in [
        ("window_vision_slot", ValueSource::RequestSlot, Dtype::Int64),
        (
            "window_vision_generation",
            ValueSource::RequestGeneration,
            Dtype::Int64,
        ),
        (
            "window_vision_feature_slot",
            ValueSource::ImageFeatureSlot,
            Dtype::Int32,
        ),
        (
            "window_vision_feature_offset",
            ValueSource::ImageTensor,
            Dtype::Int32,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let mut binding = crate::model::testing::binding(name, source, InputIndexing::Scalar, 0, 0);
        if source == ValueSource::ImageTensor {
            binding.tensor = "vision_feature_offset".into();
        }
        semantic.push(binding);
        let nbytes = dtype.size();
        manifest.buffers.push(BufferDescriptor {
            name: name.into(),
            address: 1_000_000 + index * 16,
            nbytes,
            dtype,
            shape: vec![1],
            strides: vec![1],
        });
        bindings.push(InputBinding {
            buffer_name: name.into(),
            host_offset: manifest.input_frame_bytes,
            nbytes,
        });
        manifest.input_frame_bytes += 16;
    }
    fixture.config.contract.graphs.push(GraphContract { tuned_bindings: vec![],
        graph: "Vision-1-256".into(),
        kind: GraphOperation::Vision,
        inputs: semantic,
        state: vec![],
        state_rows_per_request: 1,
        draft_cursor: None,
        embedding_tables: vec![],
        pending_token: None,
    });
    manifest.graphs.push(GraphDescriptor { tuned_bindings: vec![],
        confidence_width: 0,
        draft_tokens: 0,
        name: "Vision-1-256".into(),
        kind: GraphKind::Vision,
        handle: 1,
        batch_capacity: 1,
        token_capacity: 256,
        prefill_capacity: 0,
        estimated_duration_ns: 1000,
        input_bindings: bindings,
        packed_token_capacity: 0,
        sampling_mode: SamplingMode::General,
    });
    let processor = Arc::new(
        ImageProcessor::load(
            ImageSettings::Qwen(QwenImageSettings {
                min_pixels: 1024,
                max_pixels: 1024,
                position_side: 48,
                image_token_id: 248056,
                video_token_id: 248057,
            }),
            ImageLimits {
                patches_per_feature: 4,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let mut tokens = vec![0; 100];
    let mut data = PromptData {
        positions: (0..100).map(|n| [n; 3]).collect(),
        vision_token_indices: vec![-1; 100],
        ..Default::default()
    };
    let mut images = vec![];
    for (index, position) in [10, 40, 70].into_iter().enumerate() {
        let source = Arc::new(RgbImage::new(32, 32, vec![index as u8; 32 * 32 * 3]).unwrap());
        images.push(PreparedImageInput::new(
            ImageInput {
                description: processor.describe_image(&source).unwrap(),
                source: ImageSource::Decoded(source),
            },
            position..position + 1,
            index..index + 1,
            processor.clone(),
        ));
        tokens[position] = 248056;
        data.vision_token_indices[position] = index as i32;
    }
    let (sender, receiver) = mpsc::channel();
    let (mut incoming, mut generation) = fixture.incoming_for(tokens, 1, data);
    incoming.prompt.conditioned_inputs = images;
    incoming.prompt.sampling.seed = 1234567;
    sender.send(vec![incoming]).unwrap();
    let mut scheduler = RequestScheduler::new(
        fixture.config.identity,
        fixture.config.limits,
        crate::model::scheduler_capabilities(&fixture.config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&manifest.graphs)
        .unwrap();
    let mut worker = Worker::new(
        &fixture.config,
        &fixture.frontend,
        &fixture.stop,
        &fixture.metrics,
        scheduler,
        receiver,
        4,
    );
    worker.inputs = Some(crate::model::Inputs::new(&fixture.config, &manifest).unwrap());
    worker.image_preparation = Some(ImagePreparation::new(None));
    let _ = worker.admit().unwrap();
    let original = worker.active[0].as_ref().unwrap().request;
    let mut sequence = 0;
    let mut rollovers = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let _ = worker.admit().unwrap();
        assert!(
            std::time::Instant::now() < deadline,
            "conditioned worker stopped progressing"
        );
        let Some(step) = worker.scheduler.plan() else {
            std::thread::yield_now();
            continue;
        };
        sequence += 1;
        worker.reserve_and_pack(&step, Fence(sequence)).unwrap();
        if step.kind == GraphKind::Vision {
            let uploads = worker.inputs.as_ref().unwrap().uploads();
            assert_eq!(
                uploads
                    .iter()
                    .find(|upload| upload.buffer == "window_vision_feature_offset")
                    .unwrap()
                    .bytes,
                0i32.to_le_bytes()
            );
        }
        let sampled = (step.kind == GraphKind::Prefill
            && step.lanes[0].start_position + step.lanes[0].tokens as u64 == 100)
            .then_some(1);
        let frame = completion(&worker, &step, sequence, sampled);
        worker.complete(step.kind, frame).unwrap();
        let ready = worker.active[0].as_ref().is_some_and(|active| {
            worker
                .scheduler
                .request(active.request)
                .unwrap()
                .conditioned_window_ready()
        });
        worker.publish().unwrap();
        if ready {
            rollovers += 1;
            let active = worker.active[0].as_ref().unwrap();
            assert_eq!(active.request.generation, original.generation + rollovers);
            assert_eq!(
                worker.admissions[0].as_ref().unwrap().handle,
                active.request
            );
            assert_eq!(active.incoming.prompt.sampling.seed, 1234567);
            assert_eq!(active.carried.generated_tokens, 0);
            assert_eq!(
                worker
                    .scheduler
                    .request(active.request)
                    .unwrap()
                    .reused_prompt_tokens(),
                0
            );
            assert!(generation.events.try_recv().is_err());
        }
        if worker.active[0].is_none() {
            break;
        }
    }
    assert_eq!(rollovers, 2);
    assert_eq!(processor.preparation_counts().prepared_images, 3);
    assert_eq!(fixture.metrics.admitted.load(Ordering::Relaxed), 1);
    assert_eq!(fixture.metrics.snapshot(false).prefill_tokens, 100);
}

#[test]
fn zero_output_budget_requires_prompt_scoring_at_both_admission_checks() {
    let fixture = Fixture::new();
    let (mut incoming, _generation) = fixture.incoming(vec![0, 1], 0);
    let config = &fixture.config;
    incoming.prompt.max_new_tokens = 0;
    assert!(crate::model::admission::validate_prompt(config, &incoming.prompt).is_err());
    incoming.prompt.prompt_logprobs = true;
    crate::model::admission::validate_prompt(config, &incoming.prompt).unwrap();
    let mut admission = crate::model::Admission::from_prompt(
        config,
        RequestHandle {
            slot: 0,
            generation: 1,
        },
        &incoming.prompt,
        false,
    )
    .unwrap();
    admission.prompt_logprobs = false;
    assert!(admission.validate(config).is_err());
}

#[test]
fn completed_copyout_retries_capacity_waiters_without_a_new_model_fence() {
    let mut fixture = Fixture::new();
    fixture.config.limits.pages = 2;
    let (sender, receiver) = mpsc::channel();
    let (incoming, _generation) = fixture.incoming(vec![0; 70], 1);
    let mut history = crate::cache::PrefixHistory::new(fixture.config.identity);
    history.append(&incoming.prompt.tokens, &[]).unwrap();
    let mut worker = fixture.worker(receiver);
    worker.attach_store(Some(Box::new(FakeStore::new(Some(host_hit(&history, 64))))));
    sender.send(vec![incoming]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    let handle = worker.active[0].as_ref().unwrap().request;
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Fetched {
            key: history.key_at(64).unwrap(),
            result: Ok(4096),
        });
    fake_store(&mut worker)
        .store_outcomes
        .push_back(flare_selene::StoreOutcome::Queued);
    worker.service_store().unwrap();
    let publication = fake_store(&mut worker).stored[0].clone();
    worker.scheduler.cancel(handle).unwrap();
    worker.active[0] = None;
    worker.service_store().unwrap();
    assert_eq!(fake_store(&mut worker).stored.len(), 1);
    let (incoming, _other_generation) = fixture.incoming(vec![1; 64], 1);
    sender.send(vec![incoming]).unwrap();
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert_eq!(worker.pending.len(), 1);
    assert!(worker.pending[0].blocked.is_some());
    let fence = worker.scheduler.completed();
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Stored {
            checkpoint: publication.id,
            key: publication.key,
            result: Ok(4096),
        });
    worker.service_store().unwrap();
    assert_eq!(worker.scheduler.completed(), fence);
    assert!(worker.pending[0].blocked.is_none());
    assert_eq!(worker.admit().unwrap(), ControlFlow::Continue(true));
    assert!(worker.pending.is_empty());
    assert_eq!(worker.scheduler.stats().active_requests, 1);
}

#[test]
fn native_prompt_rejects_invalid_requested_logprob_ids() {
    let mut fixture = Fixture::new();
    let (mut incoming, _generation) = fixture.incoming(vec![0, 1], 1);
    incoming.prompt.logprobs = true;
    incoming.prompt.logprob_token_ids = vec![0, 1];
    let validate = |config: &ModelConfig, prompt: &Prompt| {
        crate::model::admission::validate_prompt(config, prompt)
    };
    assert!(
        validate(&fixture.config, &incoming.prompt)
            .unwrap_err()
            .to_string()
            .contains("re-export")
    );
    fixture.config.logprob_token_id_capacity = crate::execution::MAX_LOGPROB_TOKEN_IDS;
    validate(&fixture.config, &incoming.prompt).unwrap();
    for ids in [
        vec![0, 0],
        vec![fixture.config.contract.vocabulary as u32],
        vec![0; 129],
    ] {
        incoming.prompt.logprob_token_ids = ids;
        assert!(validate(&fixture.config, &incoming.prompt).is_err());
    }
    incoming.prompt.logprob_token_ids = vec![0];
    incoming.prompt.logprobs = false;
    assert!(validate(&fixture.config, &incoming.prompt).is_err());
}

/// Create a ready checkpoint through the scheduler's inbound publication path,
/// then release its request so eviction policy can consider it.
fn unpinned_checkpoint(
    worker: &mut Worker<'_>,
    token: u32,
    end: usize,
) -> crate::cache::CheckpointId {
    let spec = RequestSpec {
        prompt: vec![token; end + 1],
        message_boundaries: vec![],
        conditioning: vec![],
        max_new_tokens: 1,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: true,
    };
    let plan = worker.scheduler.plan_admission(&spec).unwrap();
    let key = plan.history().key_at(end as u64).unwrap();
    let pages = (end as u64 - plan.resident_tokens()) as usize / 64;
    let admitted = worker
        .scheduler
        .admit_planned(
            spec,
            plan,
            Some(crate::scheduler::InboundRequest {
                key,
                consumed_tokens: end as u64,
                pages,
            }),
        )
        .unwrap();
    worker.scheduler.finish_inbound(key, true).unwrap();
    let id = worker
        .scheduler
        .request(admitted.handle)
        .unwrap()
        .prefix()
        .unwrap();
    worker.scheduler.cancel(admitted.handle).unwrap();
    id
}

fn complete_store(worker: &mut Worker<'_>, id: crate::cache::CheckpointId, success: bool) {
    let key = worker.scheduler.cache().checkpoint(id).unwrap().prefix();
    fake_store(worker)
        .events
        .push_back(flare_selene::StoreEvent::Stored {
            checkpoint: id,
            key,
            result: if success {
                Ok(1)
            } else {
                Err("copy failed".into())
            },
        });
}

#[test]
fn queued_publication_copies_release_one_pin_on_success_or_failure() {
    for success in [false, true] {
        let mut fixture = Fixture::new();
        fixture.config.limits.pages = 128;
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store
            .store_outcomes
            .push_back(flare_selene::StoreOutcome::Queued);
        worker.attach_store(Some(Box::new(store)));
        let id = unpinned_checkpoint(&mut worker, 0, 64);
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 1);
        assert!(!has_copy(&mut worker, id));
        complete_store(&mut worker, id, success);
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 0);
        assert_eq!(
            has_copy(&mut worker, id),
            success
        );
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 0);
    }
}

#[test]
fn admission_proceeds_while_a_publication_copy_is_in_flight() {
    use flare_selene::StoreOutcome::*;
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.store_outcomes.push_back(Queued);
    worker.attach_store(Some(Box::new(store)));
    let id = unpinned_checkpoint(&mut worker, 0, 64);
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), 1);
    let (incoming, generation) = fixture.incoming(vec![1; 70], 4);
    worker.pending.push_back(incoming);
    let _ = worker.admit().unwrap();
    assert_eq!(worker.scheduler.stats().active_requests, 1);
    assert_eq!(pins(&worker, id), 1);
    complete_store(&mut worker, id, true);
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), 0);
    drop(generation);
}

#[test]
fn publication_copy_and_export_release_their_own_pins_in_either_completion_order() {
    use flare_selene::StoreOutcome::*;
    for export_first in [false, true] {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.pooled = true;
        store.store_outcomes.push_back(Queued);
        worker.attach_store(Some(Box::new(store)));
        let id = unpinned_checkpoint(&mut worker, 0, 64);
        worker.service_store().unwrap();
        let key = worker.scheduler.cache().checkpoint(id).unwrap().prefix();
        fake_store(&mut worker)
            .events
            .push_back(export_request(1, key, id.value(), 0));
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 2);
        if export_first {
            fake_store(&mut worker).events.push_back(export_ended(1));
        } else {
            complete_store(&mut worker, id, true);
        }
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 1);
        if export_first {
            complete_store(&mut worker, id, true);
        } else {
            fake_store(&mut worker).events.push_back(export_ended(1));
        }
        fake_store(&mut worker).events.push_back(export_ended(1));
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 0);
        assert!(has_copy(&mut worker, id));
        assert_eq!(fake_store(&mut worker).stored.len(), 1);
    }
}

#[test]
fn publication_copy_pins_survive_ancestor_compaction() {
    use flare_selene::StoreOutcome::*;
    let mut fixture = Fixture::new();
    fixture.config.limits.max_sequence_tokens = 256;
    fixture.config.limits.snapshots = 2;
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store.pooled = true;
    worker.attach_store(Some(Box::new(store)));
    let a = unpinned_checkpoint(&mut worker, 0, 64);
    let b = unpinned_checkpoint(&mut worker, 0, 128);
    fake_store(&mut worker)
        .store_outcomes
        .extend([Present, Queued]);
    worker.service_store().unwrap();
    let original = fake_store(&mut worker).stored[1].clone();
    assert_eq!(original.id, b);
    assert_eq!(pins(&worker, b), 1);
    let (incoming, generation) = fixture.incoming(vec![1; 70], 4);
    let spec = RequestSpec {
        prompt: incoming.prompt.tokens.clone(),
        message_boundaries: vec![],
        conditioning: vec![],
        max_new_tokens: 4,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: true,
    };
    let plan = worker.scheduler.plan_admission(&spec).unwrap();
    let key = plan.history().key_at(64).unwrap();
    worker
        .scheduler
        .admit_planned(
            spec,
            plan,
            Some(crate::scheduler::InboundRequest {
                key,
                consumed_tokens: 64,
                pages: 1,
            }),
        )
        .unwrap();
    assert!(worker.scheduler.cache().checkpoint(a).is_err());
    assert_eq!(pins(&worker, b), 1);
    assert_eq!(
        worker.scheduler.cache().page_chain(b).unwrap(),
        original.chain_pages
    );
    complete_store(&mut worker, b, true);
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, b), 0);
    let notice = &fake_store(&mut worker).evicted[0];
    assert_eq!(notice.1.as_ref().unwrap().id, b);
    drop(generation);
}

#[test]
fn a_present_publication_takes_no_pin() {
    let fixture = Fixture::new();
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store
        .store_outcomes
        .push_back(flare_selene::StoreOutcome::Present);
    worker.attach_store(Some(Box::new(store)));
    let id = unpinned_checkpoint(&mut worker, 0, 64);
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), 0);
    assert!(has_copy(&mut worker, id));
    assert_eq!(fake_store(&mut worker).stored.len(), 1);
}

fn has_copy(worker: &mut Worker<'_>, id: crate::cache::CheckpointId) -> bool {
    let key = worker.scheduler.cache().checkpoint(id).unwrap().prefix();
    fake_store(worker).copies.contains(&key)
}

#[test]
fn losing_a_host_copy_does_not_recopy_a_gpu_checkpoint() {
    use flare_selene::StoreOutcome;
    for outcome in [StoreOutcome::Present, StoreOutcome::Queued] {
        let fixture = Fixture::new();
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.store_outcomes.push_back(outcome);
        worker.attach_store(Some(Box::new(store)));
        let id = unpinned_checkpoint(&mut worker, 0, 64);
        worker.service_store().unwrap();
        if outcome == StoreOutcome::Queued {
            complete_store(&mut worker, id, true);
            worker.service_store().unwrap();
        }
        assert!(has_copy(&mut worker, id));
        fake_store(&mut worker).copies.clear();
        assert!(!worker.service_store().unwrap());
        assert!(!worker.service_store().unwrap());
        assert_eq!(fake_store(&mut worker).stored.len(), 1);
        assert_eq!(pins(&worker, id), 0);
    }
}

#[test]
fn publications_copy_immediately_even_without_gpu_pressure() {
    for pooled in [false, true] {
        let mut fixture = Fixture::new();
        fixture.config.limits.pages = 256;
        let (_sender, receiver) = mpsc::channel();
        let mut worker = fixture.worker(receiver);
        let mut store = FakeStore::new(None);
        store.pooled = pooled;
        store
            .store_outcomes
            .push_back(flare_selene::StoreOutcome::Queued);
        worker.attach_store(Some(Box::new(store)));
        let id = unpinned_checkpoint(&mut worker, 0, 64);
        worker.service_store().unwrap();
        assert_eq!(fake_store(&mut worker).stored[0].id, id);
        assert_eq!(pins(&worker, id), 1);
        assert!(!worker.service_store().unwrap());
        assert_eq!(fake_store(&mut worker).stored.len(), 1);
        complete_store(&mut worker, id, true);
        worker.service_store().unwrap();
        assert_eq!(pins(&worker, id), 0);
    }
}

#[test]
fn publication_retries_wait_for_progress_and_abandon_evicted_ids() {
    use flare_selene::{StoreOutcome::*, StoreReason};
    let mut fixture = Fixture::new();
    fixture.config.limits.pages = 2;
    let (_sender, receiver) = mpsc::channel();
    let mut worker = fixture.worker(receiver);
    let mut store = FakeStore::new(None);
    store
        .store_outcomes
        .extend([Retry(StoreReason::Slabs), Queued]);
    worker.attach_store(Some(Box::new(store)));
    let id = unpinned_checkpoint(&mut worker, 0, 64);
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), 0);
    assert_eq!(worker.deferred_stores, [id]);
    assert!(!worker.service_store().unwrap());
    assert_eq!(fake_store(&mut worker).stored.len(), 1);
    fake_store(&mut worker).progress += 1;
    worker.service_store().unwrap();
    assert!(worker.deferred_stores.is_empty());
    assert_eq!(pins(&worker, id), 1);
    assert_eq!(
        fixture.metrics.host_retry_success.load(Ordering::Relaxed),
        1
    );
    fake_store(&mut worker)
        .events
        .push_back(flare_selene::StoreEvent::Deferred { checkpoint: id });
    worker.service_store().unwrap();
    assert_eq!(pins(&worker, id), 0);
    assert_eq!(worker.deferred_stores, [id]);
    let spec = RequestSpec {
        prompt: vec![1; 70],
        message_boundaries: vec![],
        conditioning: vec![],
        max_new_tokens: 1,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: true,
    };
    worker.scheduler.admit(spec).unwrap();
    worker.service_store().unwrap();
    assert!(worker.deferred_stores.is_empty());
    assert_eq!(
        fixture.metrics.host_retry_abandoned.load(Ordering::Relaxed),
        1
    );
}
