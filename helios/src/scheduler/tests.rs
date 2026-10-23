use super::*;

pub(super) fn measured_costs(graphs: &[crate::execution::GraphDescriptor]) -> cost::ActionCosts {
    let mut costs = cost::ActionCosts::prepared(graphs).unwrap();
    for graph in graphs {
        costs.observe(
            &ScheduledStep { draft_tokens: graph.draft_tokens, packed_token_capacity: graph.packed_token_capacity,
                kind: graph.kind,
                batch_capacity: graph.batch_capacity,
                token_capacity: graph.token_capacity,
                mixed: (graph.kind == GraphKind::Mixed).then_some(MixedStep {
                    prefill_capacity: graph.prefill_capacity,
                    prefill_row: None,
                    speculative: true,
                }),
                lanes: vec![],
                requires_feedback: false,
            },
            crate::execution::ExecutionTiming {
                gpu_ns: [
                    Some(graph.estimated_duration_ns),
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
                ..Default::default()
            },
        );
    }
    costs
}

impl RequestScheduler {
    pub(crate) fn measured_graph_durations(
        &mut self,
        graphs: &[crate::execution::GraphDescriptor],
    ) -> Result<()> {
        self.graph_durations(graphs)?;
        self.costs = measured_costs(graphs);
        self.costs
            .advance_to(self.now_ns, self.policy.estimate_ttl_ns);
        Ok(())
    }
}

#[test]
fn admission_and_completion_cover_configured_concurrency() {
    for active in [1, 3, 5, 9, 33, 64] {
        let mut scheduler = scheduler(active, 512, 128);
        let handles = (0..active)
            .map(|_| scheduler.admit(spec(65, 32)).unwrap())
            .collect::<Vec<_>>();
        assert!(scheduler.admit(spec(65, 32)).is_err());
        let step = scheduler.plan().unwrap();
        assert_eq!(step.kind, GraphKind::Init);
        assert_eq!(step.lanes.len(), active);
        assert_eq!(step.batch_capacity, active.next_power_of_two());
        scheduler.commit(&step, Fence(1)).unwrap();
        let mut records = vec![CompletionRecord::default(); active.next_power_of_two()];
        for handle in &handles {
            records[handle.slot as usize] = CompletionRecord {
                generation: handle.generation as i64,
                submission_sequence: 1,
                status: 1,
                token_id: -1,
                ..Default::default()
            };
        }
        assert!(
            scheduler
                .complete(Fence(1), &records[..active - 1], Default::default())
                .is_err()
        );
        scheduler
            .complete(Fence(1), &records, Default::default())
            .unwrap();
        for handle in handles {
            assert_eq!(
                scheduler.request(handle).unwrap().phase,
                RequestPhase::Prefill
            );
        }
    }
}

#[test]
fn verification_catalog_covers_all_prepared_concurrencies_through_64() {
    let catalog = speculation::graph_catalog(&[1, 2, 3, 4], None, 64);
    assert!(catalog.iter().all(|widths| widths == &[1, 2, 3, 4]));
    let fixed = speculation::graph_catalog(&[1, 2, 3, 4], Some(4), 64);
    assert!(fixed.iter().all(|widths| widths == &[4]));
}

pub(super) fn scheduler(active: usize, pages: u32, snapshots: u32) -> RequestScheduler {
    scheduler_with_geometry(active, pages, snapshots, 64, 4)
}

fn scheduler_with_geometry(active: usize, pages: u32, snapshots: u32, page_tokens: usize, image_slots: usize) -> RequestScheduler {
    let mut scheduler = RequestScheduler::new(
        [9; 32],
        SchedulerLimits {
            request_slots: active.next_power_of_two(),
            max_active: active,
            max_sequence_tokens: 2048,
            prefill_chunk_size: 512,
            pages,
            snapshots,
        },
        SchedulerCapabilities {
            vocabulary: 248320,
            page_tokens,
            vision: crate::model::VisionCapabilities {
                feature_slots: image_slots,
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
        .measured_graph_durations(&duration_graphs(active))
        .unwrap();
    scheduler
}
pub(crate) fn duration_graphs(active: usize) -> Vec<crate::execution::GraphDescriptor> {
    let mut graphs = Vec::new();
    for batch in crate::execution::BATCH_CAPACITIES
        .into_iter()
        .filter(|b| *b <= active.next_power_of_two())
    {
        for kind in [
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Resume,
            GraphKind::Decode,
            GraphKind::Speculative,
        ] {
            for tokens in 1..=if kind == GraphKind::Speculative { 8 } else { 1 } {
                graphs.push(crate::execution::GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0,
                    name: format!("{kind:?}-{batch}-{tokens}"),
                    kind,
                    sampling_mode: crate::execution::SamplingMode::General,
                    handle: 1,
                    batch_capacity: batch,
                    token_capacity: tokens,
                    prefill_capacity: 0,
                    estimated_duration_ns: 1000,
                    input_bindings: vec![],
                });
            }
        }
    }
    for (kind, capacities) in [
        (GraphKind::Prefill, (64..=8192).step_by(64).collect()),
        (GraphKind::Vision, vec![8, 16, 256, 1024, 4096, 16384]),
    ] {
        for tokens in capacities {
            graphs.push(crate::execution::GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0,
                name: format!("{kind:?}-{tokens}"),
                kind,
                sampling_mode: crate::execution::SamplingMode::General,
                handle: 1,
                batch_capacity: 1,
                token_capacity: tokens,
                prefill_capacity: 0,
                estimated_duration_ns: 1000,
                input_bindings: vec![],
            });
        }
    }
    graphs
        .iter_mut()
        .filter(|g| g.kind == GraphKind::Snapshot)
        .for_each(|g| g.estimated_duration_ns = 1);
    graphs
        .iter_mut()
        .filter(|g| g.kind == GraphKind::Speculative)
        .for_each(|g| g.estimated_duration_ns = 1_000_000);
    graphs
}

pub(super) fn spec(tokens: usize, max_new: u32) -> RequestSpec {
    RequestSpec {
        message_boundaries: vec![],
        prompt: (0..tokens as u32).collect(),
        conditioning: vec![],
        max_new_tokens: max_new,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: false,
    }
}
fn records(
    handle: RequestHandle,
    fence: u64,
    consumed: i64,
    generated: i64,
    token: i64,
    status: i64,
) -> [CompletionRecord; 4] {
    let mut records = [CompletionRecord::default(); 4];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: fence as i64,
        consumed_tokens: consumed,
        generated_tokens: generated,
        token_id: token,
        status,
        error_code: 0,
        reserved: 0,
        vector: None,
    };
    records
}
fn init(scheduler: &mut RequestScheduler, fence: u64) {
    let step = scheduler.plan().unwrap();
    assert!(matches!(step.kind, GraphKind::Init | GraphKind::Restore));
    scheduler.commit(&step, Fence(fence)).unwrap();
    let completions = state_records(scheduler, &step, fence);
    scheduler
        .complete(Fence(fence), &completions, Default::default())
        .unwrap();
}

#[test]
fn vision_image_count_is_bounded_by_shared_feature_storage() {
    let mut scheduler = scheduler(1, 8, 2);
    scheduler.capabilities.vision.features_per_request = 5 * 64;
    let handle = scheduler.admit(spec(65, 2)).unwrap();
    assert!(scheduler.set_vision_images(handle, &[256; 6]).is_err());
    scheduler.set_vision_images(handle, &[256; 5]).unwrap();
    init(&mut scheduler, 1);
    for fence in 2..7 {
        let step = scheduler.plan().unwrap();
        assert_eq!((step.kind, step.token_capacity), (GraphKind::Vision, 256));
        scheduler.commit(&step, Fence(fence)).unwrap();
        scheduler.complete(Fence(fence), &records(handle, fence, 0, 0, -1, 1), Default::default()).unwrap();
    }
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Prefill);
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.has_available_image_slot());
}

#[test]
fn vision_images_run_after_admission_before_prefill_and_cancel_waits_for_copy() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(65, 2)).unwrap();
    scheduler.set_vision_images(handle, &[256, 1152]).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!((step.kind, step.token_capacity), (GraphKind::Vision, 256));
    scheduler.commit(&step, Fence(2)).unwrap();
    assert!(scheduler.plan().is_none());
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 0, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!((step.kind, step.token_capacity), (GraphKind::Vision, 4096));
    scheduler.commit(&step, Fence(3)).unwrap();
    scheduler
        .complete(
            Fence(3),
            &records(handle, 3, 0, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Prefill);
    scheduler.cancel(handle).unwrap();
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    scheduler.set_vision_images(handle, &[256]).unwrap();
    init(&mut scheduler, 4);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(5)).unwrap();
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.admit(spec(64, 1)).is_err());
    scheduler
        .complete(
            Fence(5),
            &records(handle, 5, 0, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    assert!(scheduler.admit(spec(64, 1)).is_ok());
}

fn freeze(scheduler: &mut RequestScheduler, handle: RequestHandle, fence: u64) {
    assert!(scheduler.finish(handle).is_err());
    let step = scheduler.phase_plan(GraphKind::Snapshot, Some(handle));
    assert_eq!(step.kind, GraphKind::Snapshot);
    scheduler.commit(&step, Fence(fence)).unwrap();
    assert!(scheduler.finish(handle).is_err());
    let completions = state_records(scheduler, &step, fence);
    scheduler
        .complete(Fence(fence), &completions, Default::default())
        .unwrap();
}

fn state_records(
    scheduler: &RequestScheduler,
    step: &ScheduledStep,
    fence: u64,
) -> [CompletionRecord; 4] {
    let mut output = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        let slot = lane.request.slot as usize;
        output[slot] = records(
            lane.request,
            fence,
            request.consumed_tokens as i64,
            request.generated_tokens as i64,
            -1,
            1,
        )[slot];
    }
    output
}

#[test]
fn inactive_state_completion_never_initializes_or_publishes_a_checkpoint() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(1)).unwrap();
    assert!(
        scheduler
            .complete(
                Fence(1),
                &[CompletionRecord::default(); 4],
                Default::default()
            )
            .is_err()
    );
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Initialize
    );
    let mut corrupt = state_records(&scheduler, &step, 1);
    corrupt[handle.slot as usize].consumed_tokens = 64;
    assert!(
        scheduler
            .complete(Fence(1), &corrupt, Default::default())
            .is_err()
    );
    let mut corrupt = state_records(&scheduler, &step, 1);
    corrupt[handle.slot as usize].token_id = 7;
    assert!(
        scheduler
            .complete(Fence(1), &corrupt, Default::default())
            .is_err()
    );
    scheduler
        .complete(
            Fence(1),
            &state_records(&scheduler, &step, 1),
            Default::default(),
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 1, 88, 2),
            Default::default(),
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(3)).unwrap();
    assert!(
        scheduler
            .complete(
                Fence(3),
                &[CompletionRecord::default(); 4],
                Default::default()
            )
            .is_err()
    );
    assert!(!scheduler.request(handle).unwrap().terminal_snapshot_ready);
    assert!(scheduler.finish(handle).is_err());
    scheduler
        .complete(
            Fence(3),
            &state_records(&scheduler, &step, 3),
            Default::default(),
        )
        .unwrap();
    assert!(scheduler.finish(handle).unwrap().is_some());
}

#[test]
fn prefill_buckets_mask_capacity_and_allocate_only_live_pages() {
    let mut scheduler = scheduler(1, 32, 2);
    let handle = scheduler.admit(spec(529, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!(
        (step.kind, step.token_capacity, step.lanes[0].tokens),
        (GraphKind::Prefill, 512, 512)
    );
    scheduler.commit(&step, Fence(2)).unwrap();
    assert_eq!(scheduler.free_page_count(), 23);
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 512, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!((step.token_capacity, step.lanes[0].tokens), (128, 17));
    scheduler.commit(&step, Fence(3)).unwrap();
    assert_eq!(scheduler.request(handle).unwrap().pages().count(), 9);
    scheduler
        .complete(
            Fence(3),
            &records(handle, 3, 529, 1, 77, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 4);
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Decode
    );
}

#[test]
fn prefill_uses_only_declared_capacities() {
    for (capacities, expected) in [(vec![128, 512], 512), (vec![128, 384, 512], 384)] {
        let mut scheduler = scheduler(1, 32, 2);
        scheduler.prefill_graphs(&capacities).unwrap();
        scheduler.admit(spec(300, 3)).unwrap();
        init(&mut scheduler, 1);
        let step = scheduler.plan().unwrap();
        assert_eq!(
            (step.kind, step.token_capacity, step.lanes[0].tokens),
            (GraphKind::Prefill, expected, 300)
        );
    }
    for capacities in [
        vec![],
        vec![128],
        vec![128, 513],
        vec![128, 128, 512],
        vec![0, 512],
    ] {
        assert!(scheduler(1, 32, 2).prefill_graphs(&capacities).is_err());
    }
}

#[test]
fn configured_prefill_chunks_and_tails_preserve_alignment_and_live_page_extents() {
    for (chunk_size, tail, tail_capacity) in
        [(4096, 17, 128), (3072, 2050, 2560), (8192, 513, 1024)]
    {
        let mut scheduler = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                request_slots: 1,
                max_active: 1,
                max_sequence_tokens: 32768,
                prefill_chunk_size: chunk_size,
                pages: 512,
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
        scheduler
            .measured_graph_durations(&duration_graphs(1))
            .unwrap();
        let handle = scheduler.admit(spec(chunk_size + tail, 3)).unwrap();
        init(&mut scheduler, 1);
        let step = scheduler.plan().unwrap();
        assert_eq!(
            (step.token_capacity, step.lanes[0].tokens),
            (chunk_size, chunk_size)
        );
        assert_eq!(step.lanes[0].start_position, 0);
        scheduler.commit(&step, Fence(2)).unwrap();
        assert_eq!(
            scheduler.request(handle).unwrap().pages().count(),
            chunk_size / 64
        );
        scheduler
            .complete(
                Fence(2),
                &records(handle, 2, chunk_size as i64, 0, -1, 1),
                Default::default(),
            )
            .unwrap();
        let step = scheduler.plan().unwrap();
        assert_eq!(
            (step.token_capacity, step.lanes[0].tokens),
            (tail_capacity, tail)
        );
        assert_eq!(step.lanes[0].start_position, chunk_size as u64);
        scheduler.commit(&step, Fence(3)).unwrap();
        assert_eq!(
            scheduler.request(handle).unwrap().pages().count(),
            (chunk_size + tail).div_ceil(64)
        );
        scheduler
            .complete(
                Fence(3),
                &records(handle, 3, (chunk_size + tail) as i64, 1, 77, 1),
                Default::default(),
            )
            .unwrap();
        retain_prompt(&mut scheduler, 5);
        assert_eq!(
            scheduler.request(handle).unwrap().phase,
            RequestPhase::Decode
        );
    }
}

#[test]
fn invalid_prefill_limits_reject_before_request_resources_are_created() {
    for chunk_size in [0, 1, 16, 63, 65, 4095, 4160, usize::MAX] {
        let result = RequestScheduler::new(
            [9; 32],
            SchedulerLimits {
                request_slots: 1,
                max_active: 1,
                max_sequence_tokens: 4096,
                prefill_chunk_size: chunk_size,
                pages: 64,
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
        );
        assert!(matches!(
            result.err().unwrap(),
            crate::Error::Execution(crate::ExecutionError::Manifest(_))
        ));
    }
}

#[test]
fn cancellation_delays_slot_page_and_snapshot_reuse_until_gpu_fence() {
    let mut scheduler = scheduler(1, 2, 2);
    let handle = scheduler.admit(spec(64, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_page_count(), 0);
    assert_eq!(scheduler.free_snapshot_count(), 2);
    assert!(scheduler.admit(spec(64, 3)).is_err());
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 1, 10, 1),
            Default::default(),
        )
        .unwrap();
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (2, 2)
    );
    let reused = scheduler.admit(spec(64, 3)).unwrap();
    assert_eq!(reused.slot, handle.slot);
    assert!(reused.generation > handle.generation);
    assert!(scheduler.cancel(handle).is_err());
}

#[test]
fn cancellation_waits_for_both_submissions_and_reused_slot_rejects_old_epoch() {
    let mut scheduler = scheduler(1, 4, 2);
    let old = scheduler.admit(spec(4, 8)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(old, 2, 4, 1, 77, 1), Default::default())
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    for fence in [4, 5] {
        let step = scheduler.plan().unwrap();
        scheduler.commit(&step, Fence(fence)).unwrap();
    }
    scheduler.cancel(old).unwrap();
    assert_eq!(
        (
            scheduler.stats().active_requests,
            scheduler.stats().retiring_slots
        ),
        (0, 1)
    );
    scheduler
        .complete(Fence(4), &records(old, 4, 5, 2, 78, 1), Default::default())
        .unwrap();
    assert_eq!(scheduler.stats().free_slots, 0);
    assert_eq!(scheduler.free_snapshot_count(), 1);
    assert!(scheduler.admit(spec(4, 8)).is_err());
    scheduler
        .complete(Fence(5), &records(old, 5, 6, 3, 79, 1), Default::default())
        .unwrap();
    assert_eq!(
        (
            scheduler.stats().free_slots,
            scheduler.free_page_count(),
            scheduler.free_snapshot_count()
        ),
        (1, 4, 1)
    );
    let reused = scheduler.admit(spec(4, 8)).unwrap();
    assert_eq!(reused.slot, old.slot);
    assert!(reused.generation > old.generation);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(6)).unwrap();
    assert!(
        scheduler
            .complete(Fence(6), &records(old, 6, 0, 0, -1, 1), Default::default())
            .is_err()
    );
    assert_eq!(
        scheduler.request(reused).unwrap().phase,
        RequestPhase::Restore
    );
    scheduler
        .complete(
            Fence(6),
            &records(reused, 6, 4, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    scheduler.cancel(reused).unwrap();
}

#[test]
fn terminal_snapshot_reclaims_unused_output_reservations_after_gpu_completion() {
    let mut scheduler = scheduler(1, 16, 1);
    scheduler.share_snapshot_pages(&[8..16]).unwrap();
    let handle = scheduler.admit(spec(64, 64 * 15)).unwrap();
    assert_eq!(scheduler.free_page_count(), 0);
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    assert_eq!(
        scheduler.request(handle).unwrap().private_pages,
        [PageId(0)]
    );
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 1, 88, 2),
            Default::default(),
        )
        .unwrap();
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Snapshot
    );
    assert_eq!(scheduler.stats().snapshot_backing_pages, 8);
    freeze(&mut scheduler, handle, 3);
    scheduler.finish(handle).unwrap().unwrap();
    scheduler.ensure_resources(16, 0).unwrap();
    assert_eq!(free(&scheduler), (16, 1));
}

#[test]
fn snapshot_pressure_waits_for_cancelled_restore_before_reusing_storage() {
    let mut scheduler = scheduler(2, 4, 2);
    scheduler.share_snapshot_pages(&[2..4, 2..4]).unwrap();
    let unrelated = || {
        let mut request = spec(65, 1);
        request.prompt[0] = 999;
        request
    };
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(seed, 2, 64, 1, 88, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, seed, 3);
    let checkpoint = scheduler.finish(seed).unwrap().unwrap();
    let restored = scheduler.admit(spec(64, 1)).unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Restore);
    scheduler.commit(&step, Fence(4)).unwrap();
    assert!(scheduler.admit(unrelated()).is_err());
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        1
    );
    scheduler.cancel(restored).unwrap();
    assert!(scheduler.admit(unrelated()).is_err());
    assert_eq!(scheduler.stats().snapshot_evictions, 1);
    assert!(scheduler.cache().checkpoint(checkpoint).is_err());
    assert_eq!(scheduler.free_snapshot_count(), 1);
    scheduler
        .complete(
            Fence(4),
            &records(restored, 4, 64, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    assert_eq!(
        (
            scheduler.stats().free_slots,
            scheduler.free_page_count(),
            scheduler.free_snapshot_count()
        ),
        (2, 4, 2)
    );
    let fresh = scheduler.admit(unrelated()).unwrap();
    assert_eq!(
        scheduler.request(fresh).unwrap().phase,
        RequestPhase::Initialize
    );
    scheduler.cancel(fresh).unwrap();
}

#[test]
fn terminal_checkpoint_full_prompt_hit_restores_and_resumes_without_prefill() {
    let mut scheduler = scheduler(2, 8, 4);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let prefill = scheduler.plan().unwrap();
    scheduler.commit(&prefill, Fence(2)).unwrap();
    assert_eq!(scheduler.stats().snapshot_publications, 0);
    assert!(scheduler.finish(handle).is_err());
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 1, 99, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, handle, 3);
    let checkpoint = scheduler.finish(handle).unwrap().unwrap();
    assert_eq!(scheduler.stats().snapshot_publications, 1);
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        0
    );
    let first = scheduler.admit(spec(64, 2)).unwrap();
    let second = scheduler.admit(spec(64, 2)).unwrap();
    let stats = scheduler.stats();
    assert_eq!(
        (stats.cache_lookups, stats.cache_hits, stats.reused_tokens),
        (3, 2, 128)
    );
    assert_eq!(
        (
            stats.active_requests,
            stats.free_slots,
            stats.cached_checkpoints
        ),
        (2, 0, 1)
    );
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        2
    );
    assert_eq!(
        scheduler
            .request(first)
            .unwrap()
            .pages()
            .collect::<Vec<_>>(),
        scheduler
            .request(second)
            .unwrap()
            .pages()
            .collect::<Vec<_>>()
    );
    init(&mut scheduler, 4);
    let resume = scheduler.plan().unwrap();
    assert_eq!(resume.kind, GraphKind::Resume);
    assert_eq!(resume.batch_capacity, 2);
    scheduler.cancel(first).unwrap();
    scheduler.cancel(second).unwrap();
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        0
    );
}

#[test]
fn appended_conversation_outlives_snapshot_capacity_without_losing_prefix_pages() {
    let mut scheduler = scheduler(1, 32, 3);
    for turn in 1..=12 {
        let tokens = turn * 64;
        let handle = scheduler.admit(spec(tokens, 1)).unwrap_or_else(|error| {
            panic!(
                "turn {turn} failed admission: {error}; {:?}",
                scheduler.stats()
            )
        });
        assert_eq!(
            scheduler.request(handle).unwrap().consumed_tokens,
            (tokens - 64) as u64
        );
        let fence = (turn as u64 - 1) * 3 + 1;
        init(&mut scheduler, fence);
        let step = scheduler.plan().unwrap();
        assert_eq!(step.kind, GraphKind::Prefill);
        assert_eq!(step.lanes[0].tokens, 64);
        scheduler.commit(&step, Fence(fence + 1)).unwrap();
        scheduler
            .complete(
                Fence(fence + 1),
                &records(handle, fence + 1, tokens as i64, 1, 77, 2),
                Default::default(),
            )
            .unwrap();
        freeze(&mut scheduler, handle, fence + 2);
        let checkpoint = scheduler.finish(handle).unwrap().unwrap();
        let pages = scheduler.cache().page_chain(checkpoint).unwrap();
        assert_eq!(pages.len(), turn);
        assert_eq!(pages.iter().copied().collect::<BTreeSet<_>>().len(), turn);
        assert_eq!(scheduler.stats().cached_pages, turn);
        assert_eq!(scheduler.free_page_count(), 32 - turn);
    }
    scheduler.ensure_resources(32, 3).unwrap();
    assert_eq!(scheduler.free_page_count(), 32);
    assert_eq!(scheduler.free_snapshot_count(), 3);
}

#[test]
fn branching_requests_own_distinct_tail_pages_and_terminal_snapshots() {
    let mut scheduler = scheduler(2, 8, 5);
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(seed, 2, 64, 1, 77, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, seed, 3);
    scheduler.finish(seed).unwrap();
    let first = scheduler.admit(spec(65, 4)).unwrap();
    let second = scheduler.admit(spec(65, 4)).unwrap();
    init(&mut scheduler, 4);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(5)).unwrap();
    scheduler
        .complete(
            Fence(5),
            &records(first, 5, 65, 1, 88, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 6);
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Prefill);
    assert_eq!(step.lanes[0].request, second);
    scheduler.commit(&step, Fence(7)).unwrap();
    scheduler
        .complete(
            Fence(7),
            &records(second, 7, 65, 1, 90, 1),
            Default::default(),
        )
        .unwrap();
    let first = scheduler.request(first).unwrap();
    let second = scheduler.request(second).unwrap();
    assert_eq!(first.shared_pages, second.shared_pages);
    assert_ne!(first.private_pages, second.private_pages);
    assert!(first.terminal_snapshot.is_none() && second.terminal_snapshot.is_none());
}

#[test]
fn host_uploaded_decode_waits_for_its_previous_sample() {
    let mut scheduler = scheduler(1, 8, 2);
    scheduler.capabilities.completion_inputs = vec!["token_ids".into()];
    let handle = scheduler.admit(spec(63, 3)).unwrap();
    init(&mut scheduler, 1);
    let prefill = scheduler.plan().unwrap();
    scheduler.commit(&prefill, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 63, 1, 88, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 3);

    let first = scheduler.plan().unwrap();
    assert_eq!(first.kind, GraphKind::Decode);
    scheduler.commit(&first, Fence(4)).unwrap();
    assert!(scheduler.plan().is_none());
    assert_eq!(scheduler.request(handle).unwrap().tokens()[63], 88);

    scheduler
        .complete(
            Fence(4),
            &records(handle, 4, 64, 2, 89, 1),
            Default::default(),
        )
        .unwrap();
    let second = scheduler.plan().unwrap();
    assert_eq!(second.kind, GraphKind::Decode);
    assert_eq!(second.lanes[0].start_position, 64);
    assert_eq!(scheduler.request(handle).unwrap().tokens()[64], 89);
    scheduler.commit(&second, Fence(5)).unwrap();
    assert!(scheduler.plan().is_none());
    scheduler
        .complete(
            Fence(5),
            &records(handle, 5, 65, 3, 90, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, handle, 6);
    scheduler.finish(handle).unwrap();
}

#[test]
fn decode_allows_two_steps_and_duplicate_finished_record_emits_once() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(63, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 63, 1, 88, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    for fence in [4, 5] {
        let step = scheduler.plan().unwrap();
        scheduler.commit(&step, Fence(fence)).unwrap();
    }
    assert!(scheduler.plan().is_none());
    let generated = scheduler
        .complete(
            Fence(4),
            &records(handle, 4, 64, 2, 89, 2),
            Default::default(),
        )
        .unwrap();
    assert_eq!(generated.len(), 1);
    assert!(scheduler.finish(handle).is_err());
    let generated = scheduler
        .complete(
            Fence(5),
            &records(handle, 5, 64, 2, 89, 2),
            Default::default(),
        )
        .unwrap();
    assert!(generated.is_empty());
    assert_eq!(scheduler.request(handle).unwrap().tokens().len(), 65);
    freeze(&mut scheduler, handle, 6);
    scheduler.finish(handle).unwrap();
}

#[test]
fn grammar_barrier_and_bad_completion_do_not_advance_request_state() {
    let mut scheduler = scheduler(1, 8, 2);
    let mut request = spec(65, 3);
    request.grammar_constrained = true;
    let handle = scheduler.admit(request).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    let bad = records(handle, 2, 66, 1, 88, 1);
    assert!(
        scheduler
            .complete(Fence(2), &bad, Default::default())
            .is_err()
    );
    assert_eq!(scheduler.request(handle).unwrap().consumed_tokens, 0);
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 65, 1, 88, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(4)).unwrap();
    assert!(scheduler.plan().is_none());
    scheduler
        .complete(
            Fence(4),
            &records(handle, 4, 66, 2, 89, 1),
            Default::default(),
        )
        .unwrap();
}

#[test]
fn resource_exhaustion_rolls_back_admission_and_page_reservation() {
    let mut scheduler = scheduler(2, 1, 2);
    assert!(scheduler.admit(spec(65, 3)).is_err());
    assert_eq!(scheduler.free_page_count(), 1);
    assert_eq!(scheduler.free_snapshot_count(), 2);
    let handle = scheduler.admit(spec(60, 3)).unwrap();
    assert!(scheduler.admit(spec(10, 3)).is_err());
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    assert_eq!(scheduler.free_page_count(), 0);
    assert_eq!(scheduler.request(handle).unwrap().last_use(), Fence(2));
    scheduler.cancel(handle).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 60, 1, 8, 1),
            Default::default(),
        )
        .unwrap();
    assert_eq!(scheduler.free_snapshot_count(), 2);
}

#[test]
fn uncertain_drain_keeps_ownership_and_confirmed_failure_reclaims_submissions() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(64, 3)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    assert!(
        scheduler
            .release_after_shutdown(&Shutdown {
                drained: false,
                error: Some("device error".into())
            })
            .is_err()
    );
    assert!(scheduler.request(handle).is_ok());
    assert_eq!(scheduler.free_page_count(), 6);
    scheduler
        .release_after_shutdown(&Shutdown {
            drained: true,
            error: Some("graph failed".into()),
        })
        .unwrap();
    assert!(scheduler.request(handle).is_err());
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (8, 2)
    );
    assert!(scheduler.admit(spec(64, 3)).is_ok());
}

#[test]
fn concurrent_identical_terminal_publications_release_duplicate_resources() {
    let mut scheduler = scheduler(2, 8, 4);
    let first = scheduler.admit(spec(64, 1)).unwrap();
    let second = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let mut checkpoint = None;
    for (handle, fence) in [(first, 2), (second, 4)] {
        let step = scheduler.plan().unwrap();
        assert_eq!(step.lanes[0].request, handle);
        scheduler.commit(&step, Fence(fence)).unwrap();
        scheduler
            .complete(
                Fence(fence),
                &records(handle, fence, 64, 1, 88, 2),
                Default::default(),
            )
            .unwrap();
        freeze(&mut scheduler, handle, fence + 1);
        let result = scheduler.finish(handle).unwrap();
        if checkpoint.is_none() {
            checkpoint = result;
        } else {
            assert_eq!(checkpoint, result);
        }
    }
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (7, 3)
    );
    assert_eq!(scheduler.stats().snapshot_publications, 1);
    let mut unrelated = spec(64, 1);
    unrelated.prompt[0] = 999;
    scheduler.ensure_resources(8, 1).unwrap();
    assert_eq!(scheduler.stats().snapshot_evictions, 1);
    assert_eq!(scheduler.stats().cached_checkpoints, 0);
    scheduler.admit(unrelated).unwrap();
}

#[test]
fn failed_terminal_snapshot_never_publishes_uninitialized_storage() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 1, 88, 2),
            Default::default(),
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Snapshot);
    scheduler.commit(&step, Fence(4)).unwrap();
    let mut error = records(handle, 4, 64, 1, -1, 3);
    error[handle.slot as usize].error_code = 6;
    scheduler
        .complete(Fence(4), &error, Default::default())
        .unwrap();
    assert_eq!(scheduler.request(handle).unwrap().error_code, Some(6));
    assert_eq!(scheduler.finish(handle).unwrap(), None);
    assert_eq!(
        (scheduler.free_page_count(), scheduler.free_snapshot_count()),
        (8, 2)
    );
}

fn speculative_scheduler(prompt: usize, budget: u32) -> (RequestScheduler, RequestHandle) {
    let mut s = scheduler(1, 64, 4);
    s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    let mut graphs = duration_graphs(1);
    graphs
        .iter_mut()
        .filter(|g| g.kind == GraphKind::Speculative && g.token_capacity == 8)
        .for_each(|g| g.estimated_duration_ns = 1);
    s.measured_graph_durations(&graphs).unwrap();
    let h = s.admit(spec(prompt, budget)).unwrap();
    init(&mut s, 1);
    let step = s.plan().unwrap();
    s.commit(&step, Fence(2)).unwrap();
    s.complete(
        Fence(2),
        &records(h, 2, prompt as i64, 1, 900, 1),
        Default::default(),
    )
    .unwrap();
    retain_prompt(&mut s, 3);
    (s, h)
}
fn retain_prompt(s: &mut RequestScheduler, fence: u64) {
    let step = s.phase_plan(GraphKind::Snapshot, None);
    assert_eq!(step.kind, GraphKind::Snapshot);
    assert!(
        step.lanes
            .iter()
            .all(|l| s.request(l.request).unwrap().boundary_capture.is_some())
    );
    let pages = step
        .lanes
        .iter()
        .map(|l| s.request(l.request).unwrap().pages().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let reuse = step
        .lanes
        .iter()
        .map(|l| s.request(l.request).unwrap().reused_prompt_tokens())
        .collect::<Vec<_>>();
    s.commit(&step, Fence(fence)).unwrap();
    s.complete(
        Fence(fence),
        &state_records(s, &step, fence),
        Default::default(),
    )
    .unwrap();
    for (i, l) in step.lanes.iter().enumerate() {
        assert_eq!(
            s.request(l.request).unwrap().pages().collect::<Vec<_>>(),
            pages[i]
        );
        assert_eq!(
            s.request(l.request).unwrap().reused_prompt_tokens(),
            reuse[i]
        );
    }
}

fn vector_records(
    h: RequestHandle,
    fence: u64,
    consumed: i64,
    generated: i64,
    tokens: &[i64],
    width: i64,
    status: i64,
) -> [CompletionRecord; 4] {
    let mut row = records(
        h,
        fence,
        consumed,
        generated,
        tokens.last().copied().unwrap_or(-1),
        status,
    );
    let mut v = crate::execution::TokenVector {
        emitted_count: tokens.len() as i64,
        offered_drafts: if tokens.is_empty() { 0 } else { width - 1 },
        accepted_drafts: (tokens.len() as i64).min(width - 1),
        draft_cache_consumed: consumed - 1,
        verification_width: width,
        ..Default::default()
    };
    v.tokens[..tokens.len()].copy_from_slice(tokens);
    row[h.slot as usize].vector = Some(v);
    row
}
#[test]
fn speculative_every_committed_prefix_updates_counts_and_tokens_once() {
    for count in 1..=8 {
        let (mut s, h) = speculative_scheduler(65, 32);
        let step = s.plan().unwrap();
        assert_eq!(step.kind, GraphKind::Speculative);
        assert_eq!(step.token_capacity, 8);
        s.commit(&step, Fence(4)).unwrap();
        let ids = (10..10 + count as i64).collect::<Vec<_>>();
        let events = s
            .complete(
                Fence(4),
                &vector_records(h, 4, 65 + count as i64, 1 + count as i64, &ids, 8, 1),
                Default::default(),
            )
            .unwrap();
        assert_eq!(
            events.iter().map(|t| t.token).collect::<Vec<_>>(),
            ids.iter().map(|t| *t as u32).collect::<Vec<_>>()
        );
        assert_eq!(s.request(h).unwrap().consumed_tokens, 65 + count as u64);
        assert_eq!(s.request(h).unwrap().generated_tokens, 1 + count as u32);
        assert_eq!(s.request(h).unwrap().scheduled_consumed, 65 + count as u64);
    }
}
#[test]
fn speculative_rounds_wait_for_feedback_and_use_actual_counts() {
    let (mut s, h) = speculative_scheduler(65, 32);
    let first = s.plan().unwrap();
    assert!(first.requires_feedback);
    s.commit(&first, Fence(4)).unwrap();
    assert!(s.plan().is_none());
    assert!(s.commit(&first, Fence(5)).is_err());
    s.complete(
        Fence(4),
        &vector_records(h, 4, 67, 3, &[10, 11], 8, 1),
        Default::default(),
    )
    .unwrap();
    let mut graphs = duration_graphs(1);
    graphs
        .iter_mut()
        .filter(|g| g.kind == GraphKind::Speculative && g.token_capacity == 1)
        .for_each(|g| g.estimated_duration_ns = 1);
    s.costs = measured_costs(&graphs);
    let second = s.plan().unwrap();
    assert_eq!(second.token_capacity, 1);
    assert_eq!(second.lanes[0].start_position, 67);
    s.commit(&second, Fence(5)).unwrap();
    s.complete(
        Fence(5),
        &vector_records(h, 5, 68, 4, &[12], 1, 1),
        Default::default(),
    )
    .unwrap();
    assert_eq!(s.request(h).unwrap().scheduled_consumed, 68);
    assert_eq!(s.request(h).unwrap().tokens().last(), Some(&12));
}
#[test]
fn speculative_output_limit_masks_rows_without_queueing_terminal_work() {
    let (mut s, h) = speculative_scheduler(60, 2);
    let mut graphs = duration_graphs(1);
    graphs.retain(|graph| graph.kind != GraphKind::Speculative || graph.token_capacity == 8);
    s.costs = measured_costs(&graphs);
    let first = s.plan().unwrap();
    assert_eq!(first.token_capacity, 8);
    assert_eq!(first.lanes[0].tokens, 1);
    s.commit(&first, Fence(4)).unwrap();
    assert_eq!(s.request(h).unwrap().pages().count(), 2);
    assert!(s.plan().is_none());
    let out = s
        .complete(
            Fence(4),
            &vector_records(h, 4, 61, 2, &[7], 8, 2),
            Default::default(),
        )
        .unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].finished);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Snapshot);
    freeze(&mut s, h, 5);
    s.finish(h).unwrap();
    assert_eq!(
        (
            s.stats().free_pages,
            s.stats().free_snapshots,
            s.stats().free_slots
        ),
        (64, 2, 1)
    );
}
#[test]
fn speculative_bad_vector_is_transactional_and_cancel_waits_for_round_fence() {
    let (mut s, h) = speculative_scheduler(65, 32);
    let first = s.plan().unwrap();
    s.commit(&first, Fence(4)).unwrap();
    let bad = vector_records(h, 4, 68, 3, &[10, 11], 8, 1);
    assert!(s.complete(Fence(4), &bad, Default::default()).is_err());
    assert_eq!(s.request(h).unwrap().generated_tokens, 1);
    s.cancel(h).unwrap();
    assert!(s.admit(spec(3, 2)).is_err());
    s.complete(
        Fence(4),
        &vector_records(h, 4, 67, 3, &[10, 11], 8, 1),
        Default::default(),
    )
    .unwrap();
    let next = s.admit(spec(3, 2)).unwrap();
    assert_eq!(next.slot, h.slot);
    assert_eq!(next.generation, h.generation + 1);
}
#[test]
fn fixed_graph_masks_grammar_and_host_stops_to_one_token() {
    for grammar in [false, true] {
        let (mut s, h) = speculative_scheduler(65, 32);
        let mut graphs = duration_graphs(1);
        graphs.retain(|graph| graph.kind != GraphKind::Speculative || graph.token_capacity == 8);
        s.costs = measured_costs(&graphs);
        s.speculative_fixed_width = Some(8);
        s.request_mut(h).unwrap().grammar_constrained = grammar;
        s.request_mut(h).unwrap().host_stops = !grammar;
        let step = s.plan().unwrap();
        assert_eq!(step.token_capacity, 8);
        assert_eq!(step.lanes[0].tokens, 1);
        assert!(step.requires_feedback);
        s.commit(&step, Fence(4)).unwrap();
        assert!(s.plan().is_none());
        s.complete(
            Fence(4),
            &vector_records(h, 4, 66, 2, &[10], 8, 1),
            Default::default(),
        )
        .unwrap();
        assert!(s.plan().is_some());
    }
}
#[test]
fn speculative_width_catalog_is_validated() {
    use super::speculation::validate_widths;
    for bad in [vec![2], vec![1, 9], vec![1, 4, 2]] {
        assert!(validate_widths(&bad).is_err());
    }
}

#[test]
fn shared_drafts_learn_committed_outputs_without_prompt_backfill_or_slot_joining() {
    let mut s = scheduler(1, 64, 4);
    s.enable_history_drafts().unwrap();
    s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    s.fixed_speculative_width(Some(8)).unwrap();
    let h = s.admit(spec(65, 32)).unwrap();
    assert_eq!(s.history_proposal(&[0, 1, 2, 3], 7).1, 0);
    init(&mut s, 1);
    let step = s.plan().unwrap();
    s.commit(&step, Fence(2)).unwrap();
    s.complete(Fence(2), &records(h, 2, 65, 1, 10, 1), Default::default()).unwrap();
    retain_prompt(&mut s, 3);
    let step = s.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Speculative);
    s.commit(&step, Fence(4)).unwrap();
    let mut completion = vector_records(h, 4, 70, 6, &[11, 12, 13, 90, 91], 8, 1);
    completion[0].vector.as_mut().unwrap().accepted_drafts = 4;
    let mut invalid = completion;
    invalid[0].generation += 1;
    assert!(s.complete(Fence(4), &invalid, Default::default()).is_err());
    assert_eq!(s.history_proposal(&[10, 11, 12, 13], 7).1, 0);
    s.complete(Fence(4), &completion, Default::default()).unwrap();
    assert_eq!(s.history_proposal(&[10, 11, 12, 13], 7), ([90, 91, -1, -1, -1, -1, -1], 2));
    assert_eq!(s.history_proposal(&[0, 1, 2, 3], 7).1, 0);
    assert_eq!(s.history_proposal(&[12, 13, 90, 91], 7).1, 0);
    s.cancel(h).unwrap();
    let next = s.admit(spec(66, 16)).unwrap();
    assert_eq!(next.slot, h.slot);
    assert_ne!(next.generation, h.generation);
    assert_eq!(s.history_proposal(&[10, 11, 12, 13], 7).0[0], 90);
    init(&mut s, 5);
    let step = s.plan().unwrap();
    s.commit(&step, Fence(6)).unwrap();
    s.complete(Fence(6), &records(next, 6, 66, 1, 92, 1), Default::default()).unwrap();
    retain_prompt(&mut s, 7);
    assert_eq!(s.history_proposal(&[12, 13, 90, 91], 7).1, 0);
    let step = s.plan().unwrap();
    s.commit(&step, Fence(8)).unwrap();
    let mut failed = vector_records(next, 8, 66, 1, &[], 8, 3);
    failed[0].error_code = 1;
    s.complete(Fence(8), &failed, Default::default()).unwrap();
    assert_eq!(s.history_proposal(&[1, 2, 3, 4], 7).1, 0);
}

#[test]
fn draft_boundary_page_remains_private_across_published_prefix_branches() {
    let mut scheduler = scheduler(2, 16, 4);
    scheduler.enable_speculation(&[1, 2, 4, 8]).unwrap();
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    let initial = scheduler.request(seed).unwrap().pages().collect::<Vec<_>>();
    assert_eq!(initial.len(), 2);
    scheduler
        .complete(
            Fence(2),
            &records(seed, 2, 64, 1, 77, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, seed, 3);
    scheduler.finish(seed).unwrap();
    assert_eq!(scheduler.stats().cached_pages, 1);
    assert_eq!(scheduler.free_page_count(), 15);
    let first = scheduler.admit(spec(64, 4)).unwrap();
    let second = scheduler.admit(spec(64, 4)).unwrap();
    init(&mut scheduler, 4);
    let a = scheduler
        .request(first)
        .unwrap()
        .pages()
        .collect::<Vec<_>>();
    let b = scheduler
        .request(second)
        .unwrap()
        .pages()
        .collect::<Vec<_>>();
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(a[0], initial[0]);
    assert_eq!(b[0], initial[0]);
    assert_ne!(a[1], b[1]);
    assert_ne!(a[1], a[0]);
    assert_ne!(b[1], b[0]);
    scheduler.cancel(first).unwrap();
    scheduler.cancel(second).unwrap();
    assert_eq!(scheduler.free_page_count(), 15);
}

#[test]
fn fixed_width_respects_output_limit_at_context_boundary() {
    let mut scheduler = scheduler(1, 16, 4);
    scheduler.limits.max_sequence_tokens = 128;
    scheduler.enable_speculation(&[1, 2, 4, 8]).unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let h = scheduler.admit(spec(125, 3)).unwrap();
    init(&mut scheduler, 1);
    let prefill = scheduler.plan().unwrap();
    scheduler.commit(&prefill, Fence(2)).unwrap();
    scheduler
        .complete(Fence(2), &records(h, 2, 125, 1, 9, 1), Default::default())
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    let first = scheduler.plan().unwrap();
    assert_eq!(first.token_capacity, 8);
    assert_eq!(first.lanes[0].tokens, 2);
    scheduler.commit(&first, Fence(4)).unwrap();
    assert_eq!(scheduler.request(h).unwrap().pages().count(), 3);
    assert!(scheduler.plan().is_none());
    scheduler
        .complete(
            Fence(4),
            &vector_records(h, 4, 127, 3, &[10, 11], 8, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, h, 5);
    scheduler.finish(h).unwrap();
    assert_eq!(scheduler.free_page_count(), 15);
}

#[test]
fn execution_plan_survives_unrelated_arrival_and_cost_observation() {
    let mut s = scheduler(2, 64, 8);
    let first = s.admit(spec(65, 8)).unwrap();
    let plan = s.plan().unwrap();
    let second = s.admit(spec(129, 8)).unwrap();
    assert_eq!(s.plan().unwrap().lanes.len(), 2);
    s.observe_execution(
        &plan,
        crate::execution::ExecutionTiming {
            readiness: None,
            gpu_ns: [
                Some(20),
                Some(30),
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
            wall_ns: 50,
            queued: false,
        },
    );
    s.commit(&plan, Fence(1)).unwrap();
    assert_eq!(s.request(first).unwrap().pending, 1);
    assert_eq!(s.request(second).unwrap().pending, 0);
    let completion = state_records(&s, &plan, 1);
    s.complete(Fence(1), &completion, Default::default())
        .unwrap();
    assert_eq!(s.request(first).unwrap().phase, RequestPhase::Prefill);
    assert_eq!(s.request(second).unwrap().phase, RequestPhase::Initialize);
}

#[test]
fn execution_plan_rejects_owner_epoch_replay_and_reused_generation() {
    let mut s = scheduler(1, 64, 8);
    let first = s.admit(spec(65, 8)).unwrap();
    let plan = s.plan().unwrap();
    let mut other = scheduler(1, 64, 8);
    other.admit(spec(65, 8)).unwrap();
    let before = other.stats();
    assert!(other.commit(&plan, Fence(1)).is_err());
    assert_eq!(other.stats(), before);
    s.set_vision_images(first, &[256]).unwrap();
    let before = s.stats();
    assert!(s.commit(&plan, Fence(1)).is_err());
    assert_eq!(s.stats(), before);
    let revised = s.plan().unwrap();
    s.commit(&revised, Fence(1)).unwrap();
    assert!(s.commit(&revised, Fence(2)).is_err());
    let completion = state_records(&s, &revised, 1);
    s.complete(Fence(1), &completion, Default::default())
        .unwrap();
    assert!(s.commit(&revised, Fence(2)).is_err());
    s.cancel(first).unwrap();
    let second = s.admit(spec(65, 8)).unwrap();
    assert_ne!(first.generation, second.generation);
    assert!(s.commit(&plan, Fence(2)).is_err());
    assert_eq!(s.request(second).unwrap().pending, 0);
}

#[test]
fn loaded_catalog_cannot_change_under_a_plan() {
    let mut s = scheduler(1, 64, 8);
    s.admit(spec(65, 8)).unwrap();
    let plan = s.plan().unwrap();
    assert!(s.prefill_graphs(&[64, 128, 512]).is_err());
    assert!(s.enable_speculation(&[1, 2]).is_err());
    s.commit(&plan, Fence(1)).unwrap();
}

#[test]
fn acceptance_keeps_request_program_and_censored_prefixes_separate() {
    use speculation::{AcceptanceProfile, ProposalSource::{Copy, Mtp}};
    let mut a = AcceptanceProfile::default();
    let b = AcceptanceProfile::default();
    a.observe(Mtp, 2, 5, 2, false, None);
    assert!(a.prefix_probability(Mtp, 0).unwrap() > 0.5);
    assert!(a.prefix_probability(Mtp, 1).unwrap() > 0.25);
    assert_eq!(a.prefix_probability(Copy, 0), Some(0.5));
    assert_eq!(a.prefix_probability(Mtp, 2), Some(0.125));
    a.observe(Copy, 2, 5, 2, true, None);
    assert!(a.prefix_probability(Copy, 2).unwrap() < 0.125);
    assert_eq!(a.prefix_probability(Mtp, 2), Some(0.125));
    assert!(a.prefix_probability(Copy, 3).unwrap() < 0.0625);
    assert_eq!(b.prefix_probability(Mtp, 0), Some(0.5));
    assert_eq!(b.expected_tokens(Mtp, 0), 0.0);
    assert_eq!(b.expected_tokens(Mtp, 3), 1.75);
}

#[test]
fn physically_offered_padding_is_not_an_acceptance_observation() {
    use speculation::ProposalSource::Mtp;
    for remaining in [1, 2] {
        let (mut s, h) = speculative_scheduler(65, remaining + 1);
        let mut graphs = duration_graphs(1);
        graphs.retain(|graph| graph.kind != GraphKind::Speculative || graph.token_capacity == 8);
        s.costs = measured_costs(&graphs);
        let plan = s.plan().unwrap();
        assert_eq!(plan.token_capacity, 8);
        assert_eq!(plan.lanes[0].tokens, remaining as usize);
        s.commit(&plan, Fence(4)).unwrap();
        let tokens = vec![10; remaining as usize];
        let mut completion = vector_records(
            h,
            4,
            65 + remaining as i64,
            1 + remaining as i64,
            &tokens,
            8,
            2,
        );
        completion[0].vector.as_mut().unwrap().accepted_drafts = remaining as i64;
        s.complete(Fence(4), &completion, Default::default())
            .unwrap();
        let profile = s.request(h).unwrap().acceptance();
        if remaining == 1 {
            assert_eq!(profile.rounds(), 0);
            assert_eq!(profile.prefix_probability(Mtp, 0), Some(0.5));
        } else {
            assert!(profile.prefix_probability(Mtp, 0).unwrap() > 0.5);
            assert_eq!(profile.prefix_probability(Mtp, 1), Some(0.25));
        }
    }
}

#[test]
fn completion_learns_only_proven_acceptance_and_is_transactional() {
    use speculation::ProposalSource::Mtp;
    for (accepted, status) in [(1, 2), (0, 2), (0, 1)] {
        let (mut s, h) = speculative_scheduler(65, 8);
        let plan = s.plan().unwrap();
        s.commit(&plan, Fence(4)).unwrap();
        let mut completion = vector_records(h, 4, 66, 2, &[10], 8, status);
        completion[0].vector.as_mut().unwrap().accepted_drafts = accepted;
        let mut invalid = completion;
        invalid[0].generation += 1;
        assert!(s.complete(Fence(4), &invalid, Default::default()).is_err());
        assert_eq!(s.request(h).unwrap().acceptance().rounds(), 0);
        s.complete(Fence(4), &completion, Default::default())
            .unwrap();
        let profile = s.request(h).unwrap().acceptance();
        assert_eq!(profile.rounds(), 1);
        if accepted == 1 {
            assert!(profile.prefix_probability(Mtp, 0).unwrap() > 0.5);
            assert_eq!(profile.prefix_probability(Mtp, 1), Some(0.25));
        } else {
            assert!(profile.prefix_probability(Mtp, 0).unwrap() < 0.5);
            assert!(profile.prefix_probability(Mtp, 1).unwrap() < 0.25);
        }
    }
}

#[test]
fn graph_duration_median_uses_runtime_observations_and_excludes_queue_wait() {
    use crate::execution::ExecutionTiming;
    let mut s = scheduler(2, 64, 8);
    let graphs = duration_graphs(2);
    s.admit(spec(65, 8)).unwrap();
    assert!(s.measured_graph_durations(&graphs).is_err());
    let plan = s.plan().unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(1000));
    for queued in [false, true] {
        s.observe_execution(
            &plan,
            ExecutionTiming {
                readiness: None,
                gpu_ns: [
                    Some(1100),
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
                    Some(1800),
                ],
                wall_ns: 100_000,
                queued,
            },
        );
    }
    assert_eq!(s.estimated_duration_ns(&plan), Some(14075));
    let mut same_graph = (*plan).clone();
    same_graph.lanes[0].start_position = 32769;
    same_graph.lanes[0].tokens = 1;
    assert_eq!(s.estimated_duration_ns(&same_graph), Some(14075));
    same_graph.batch_capacity = 2;
    assert_eq!(s.estimated_duration_ns(&same_graph), Some(1000));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            readiness: None,
            gpu_ns: [None; 16],
            wall_ns: 999_999,
            queued: false,
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(14075));
}

#[test]
fn readiness_correction_changes_shape_cost_without_changing_handoff_or_service_time() {
    use crate::execution::{ExecutionTiming, ReadinessIntervals};
    let mut costs = cost::ActionCosts::prepared(&duration_graphs(1)).unwrap();
    let step = ScheduledStep { draft_tokens: 0, kind: GraphKind::Decode, batch_capacity: 1, token_capacity: 1,
        packed_token_capacity: 0, mixed: None, lanes: vec![], requires_feedback: false };
    let timestamps = [1000u64, 3000, 3100].into_iter().flat_map(u64::to_ne_bytes).collect::<Vec<_>>();
    let mut gpu_ns = [None; 16];
    gpu_ns[0] = Some(10_000);
    gpu_ns[1] = Some(9_000);
    let mut timing = ExecutionTiming { gpu_ns, readiness: Some(ReadinessIntervals::decode(&timestamps).unwrap()),
        wall_ns: 12_000, queued: false };
    costs.observe(&step, timing);
    assert_eq!(costs.get(&step), Some(10_000));
    assert_eq!(cost::observed_duration_ns(timing), Some(12_000));
    timing.queued = true;
    timing.wall_ns = 100_000;
    costs.observe(&step, timing);
    assert_eq!(costs.get(&step), Some(10_000));
    assert_eq!(cost::observed_duration_ns(timing), Some(10_000));
    assert_eq!(timing.gpu_ns[0], Some(10_000));
}

#[test]
fn graph_duration_configuration_accepts_indeterminate_and_rejects_duplicates_atomically() {
    let mut s = scheduler(1, 64, 8);
    let mut graphs = duration_graphs(1);
    graphs[0].estimated_duration_ns = 0;
    s.graph_durations(&graphs).unwrap();
    graphs.push(graphs[0].clone());
    assert!(s.graph_durations(&graphs).is_err());
    s.admit(spec(65, 8)).unwrap();
    assert_eq!(s.estimated_duration_ns(&s.plan().unwrap()), Some(0));
}

#[test]
fn bad_intermediate_vector_token_does_not_partially_commit() {
    let (mut s, h) = speculative_scheduler(65, 32);
    let step = s.plan().unwrap();
    s.commit(&step, Fence(4)).unwrap();
    let mut bad = vector_records(h, 4, 67, 3, &[10, 11], 8, 1);
    bad[0].vector.as_mut().unwrap().tokens[0] = -2;
    assert!(s.complete(Fence(4), &bad, Default::default()).is_err());
    assert_eq!(s.request(h).unwrap().consumed_tokens, 65);
    assert_eq!(s.request(h).unwrap().generated_tokens, 1);
    assert_eq!(s.request(h).unwrap().tokens().len(), 66);
    s.complete(
        Fence(4),
        &vector_records(h, 4, 67, 3, &[10, 11], 8, 1),
        Default::default(),
    )
    .unwrap();
}

#[test]
fn fixed_and_adaptive_widths_are_independent_of_batch_capacity() {
    for active in [1, 2, 3, 4] {
        let mut s = scheduler(active, 64, 8);
        let widths = (1..=8).collect::<Vec<_>>();
        s.enable_speculation(&widths).unwrap();
        s.fixed_speculative_width(Some(7)).unwrap();
        let handles = (0..active)
            .map(|_| s.admit(spec(65, 32)).unwrap())
            .collect::<Vec<_>>();
        init(&mut s, 1);
        for (index, h) in handles.iter().enumerate() {
            let step = s.plan().unwrap();
            assert_eq!(step.kind, GraphKind::Prefill);
            let fence = 2 + 2 * index as u64;
            s.commit(&step, Fence(fence)).unwrap();
            s.complete(
                Fence(fence),
                &records(*h, fence, 65, 1, 9, 1),
                Default::default(),
            )
            .unwrap();
            retain_prompt(&mut s, fence + 1);
        }
        let step = s.plan().unwrap();
        let batch = active.next_power_of_two();
        let width = 7;
        assert_eq!((step.batch_capacity, step.token_capacity), (batch, width));
        assert!(step.requires_feedback);
        s.speculative_fixed_width = None;
        s.refresh_speculative_catalog();
        for width in 1..=8 {
            let mut graphs = duration_graphs(active);
            graphs
                .iter_mut()
                .filter(|g| g.kind == GraphKind::Speculative && g.token_capacity == width)
                .for_each(|g| g.estimated_duration_ns = 1);
            s.costs = measured_costs(&graphs);
            assert_eq!(s.plan().unwrap().token_capacity, width);
        }
    }
}

fn copied(consumed: u64, count: usize, grammar: bool) -> speculation::CopyProposal {
    let mut tokens = [-1; 7];
    for (index, token) in tokens[..count].iter_mut().enumerate() {
        *token = index as i64 + 10;
    }
    speculation::CopyProposal {
        consumed,
        tokens,
        count,
        grammar_masks: if grammar {
            vec![-1; (count + 1) * 248320usize.div_ceil(32)]
        } else {
            vec![]
        },
    }
}

#[test]
fn fixed_copy_graph_masks_each_available_chain_independently() {
    for (available, grammar, expected) in
        [(0, false, 1), (2, false, 3), (7, false, 8), (2, true, 3)]
    {
        let mut s = scheduler(2, 32, 4);
        s.enable_speculation(&speculation::WIDTHS).unwrap();
        s.proposal_source(speculation::ProposalSource::Copy)
            .unwrap();
        s.fixed_speculative_width(Some(8)).unwrap();
        let mut request = spec(65, 32);
        request.grammar_constrained = grammar;
        let a = s.admit(request).unwrap();
        let b = s.admit(spec(65, 32)).unwrap();
        init(&mut s, 1);
        for (i, h) in [a, b].into_iter().enumerate() {
            let step = s.plan().unwrap();
            s.commit(&step, Fence(2 + 2 * i as u64)).unwrap();
            s.complete(
                Fence(2 + 2 * i as u64),
                &records(h, 2 + 2 * i as u64, 65, 1, 9, 1),
                Default::default(),
            )
            .unwrap();
            retain_prompt(&mut s, 3 + 2 * i as u64);
        }
        s.set_copy_proposal(a, copied(65, available, grammar))
            .unwrap();
        s.set_copy_proposal(b, copied(65, 7, false)).unwrap();
        let step = s.plan().unwrap();
        assert_eq!((step.batch_capacity, step.token_capacity), (2, 8));
        assert_eq!(step.lanes[0].tokens, expected);
        assert_eq!(step.lanes[1].tokens, 8);
        assert!(step.requires_feedback);
        s.commit(&step, Fence(6)).unwrap();
        assert!(s.plan().is_none());
        let mut complete = vector_records(a, 6, 66, 2, &[40], 8, 1);
        let other = vector_records(b, 6, 66, 2, &[41], 8, 1);
        complete[b.slot as usize] = other[b.slot as usize];
        for h in [a, b] {
            complete[h.slot as usize]
                .vector
                .as_mut()
                .unwrap()
                .draft_cache_consumed = -1;
        }
        let events = s.complete(Fence(6), &complete, Default::default()).unwrap();
        assert_eq!(events.len(), 2);
        assert!(s.request(a).unwrap().copy_proposal().is_none());
        assert_eq!(s.plan().unwrap().token_capacity, 8);
        assert_eq!(s.plan().unwrap().lanes[0].tokens, 1);
        assert!(s.set_copy_proposal(a, copied(65, 7, grammar)).is_err());
        s.cancel(a).unwrap();
        s.cancel(b).unwrap();
        assert_eq!((s.free_page_count(), s.free_snapshot_count()), (31, 3));
    }
}

#[test]
fn copy_source_uses_no_draft_page_and_fixed_graph_masks_host_stops() {
    let mut s = scheduler(1, 2, 2);
    s.limits.max_sequence_tokens = 128;
    s.enable_speculation(&speculation::WIDTHS).unwrap();
    s.proposal_source(speculation::ProposalSource::Copy)
        .unwrap();
    s.fixed_speculative_width(Some(8)).unwrap();
    let mut request = spec(64, 64);
    request.host_stops = true;
    let h = s.admit(request).unwrap();
    init(&mut s, 1);
    let prefill = s.plan().unwrap();
    s.commit(&prefill, Fence(2)).unwrap();
    s.complete(Fence(2), &records(h, 2, 64, 1, 9, 1), Default::default())
        .unwrap();
    retain_prompt(&mut s, 3);
    assert_eq!(s.request(h).unwrap().pages().count(), 1);
    assert!(s.set_copy_proposal(h, copied(64, 1, false)).is_err());
    s.set_copy_proposal(h, copied(64, 0, false)).unwrap();
    assert_eq!(s.plan().unwrap().token_capacity, 8);
    assert_eq!(s.plan().unwrap().lanes[0].tokens, 1);
    s.cancel(h).unwrap();
    assert_eq!(s.free_page_count(), 1);
    // MTP requires a private draft page in addition to these two target pages.
    let mut mtp = scheduler(1, 2, 2);
    mtp.limits.max_sequence_tokens = 128;
    mtp.enable_speculation(&speculation::WIDTHS).unwrap();
    assert!(mtp.admit(spec(64, 64)).is_err());
}

#[test]
fn declared_model_capabilities_drive_state_slots_vision_buckets_and_copy_validation() {
    let capabilities = SchedulerCapabilities {
        vocabulary: 32,
        page_tokens: 64,
        vision: crate::model::VisionCapabilities {
            feature_slots: 4,
            features_per_request: 16,
            max_images: 2,
            patches_per_feature: 2,
            patch_capacities: vec![8, 32],
        },
        draft_pages_per_request: 1,
        completion_inputs: vec![],
        state_rows_per_request: None,
    };
    let limits = SchedulerLimits {
        request_slots: 1,
        max_active: 1,
        max_sequence_tokens: 2048,
        prefill_chunk_size: 512,
        pages: 8,
        snapshots: 2,
    };
    let mut s = RequestScheduler::new([1; 32], limits, capabilities.clone()).unwrap();
    s.measured_graph_durations(&duration_graphs(1)).unwrap();
    s.enable_speculation(&[1, 2]).unwrap();
    let handle = s
        .admit(RequestSpec {
            message_boundaries: vec![],
            prompt: vec![7; 4],
            conditioning: vec![],
            max_new_tokens: 8,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .unwrap();
    assert_eq!(s.request(handle).unwrap().state_slot, StateSlotId(1));
    assert!(s.set_vision_images(handle, &[16, 16, 4]).is_err());
    assert!(s.set_vision_images(handle, &[3]).is_err());
    s.set_vision_images(handle, &[4]).unwrap();
    init(&mut s, 1);
    let step = s.plan().unwrap();
    assert_eq!((step.kind, step.token_capacity), (GraphKind::Vision, 8));
    s.commit(&step, Fence(2)).unwrap();
    s.complete(
        Fence(2),
        &records(handle, 2, 0, 0, -1, 1),
        Default::default(),
    )
    .unwrap();
    let step = s.plan().unwrap();
    s.commit(&step, Fence(3)).unwrap();
    s.complete(
        Fence(3),
        &records(handle, 3, 4, 1, 7, 1),
        Default::default(),
    )
    .unwrap();
    retain_prompt(&mut s, 4);
    let mut proposal = speculation::CopyProposal {
        consumed: 4,
        tokens: [32, -1, -1, -1, -1, -1, -1],
        count: 1,
        grammar_masks: vec![],
    };
    assert!(s.set_copy_proposal(handle, proposal.clone()).is_err());
    proposal.tokens[0] = 31;
    s.set_copy_proposal(handle, proposal).unwrap();
    for bad in [
        SchedulerCapabilities {
            vocabulary: 0,
            ..capabilities.clone()
        },
        SchedulerCapabilities {
            draft_pages_per_request: 9,
            completion_inputs: vec![],
            state_rows_per_request: None,
            ..capabilities
        },
    ] {
        assert!(RequestScheduler::new([1; 32], limits, bad).is_err());
    }
}

#[test]
fn scheduler_uses_only_loaded_verification_shapes() {
    for active in [1, 2, 3, 4] {
        let mut s = scheduler(active, 64, 8);
        s.enable_speculation(&[1, 2, 3, 4]).unwrap();
        let shapes: Vec<_> = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (4, 1)]
            .into_iter()
            .filter(|(batch, _)| *batch <= active.next_power_of_two())
            .collect();
        s.speculative_graphs(&shapes).unwrap();
        let handles = (0..active)
            .map(|_| s.admit(spec(65, 32)).unwrap())
            .collect::<Vec<_>>();
        init(&mut s, 1);
        for (index, h) in handles.iter().enumerate() {
            let step = s.plan().unwrap();
            let fence = 2 + 2 * index as u64;
            s.commit(&step, Fence(fence)).unwrap();
            s.complete(
                Fence(fence),
                &records(*h, fence, 65, 1, 9, 1),
                Default::default(),
            )
            .unwrap();
            retain_prompt(&mut s, fence + 1);
        }
        let step = s.plan().unwrap();
        assert!(shapes.contains(&(step.batch_capacity, step.token_capacity)));
        assert_eq!(step.kind, GraphKind::Speculative);
        s.commit(&step, Fence(12)).unwrap();
        let mut complete = [CompletionRecord::default(); 4];
        for h in &handles {
            complete[h.slot as usize] =
                vector_records(*h, 12, 66, 2, &[10], step.token_capacity as i64, 1)
                    [h.slot as usize];
        }
        s.complete(Fence(12), &complete, Default::default())
            .unwrap();
        for h in handles.iter().skip(1) {
            s.cancel(*h).unwrap();
        }
        let step = s.plan().unwrap();
        assert!(shapes.contains(&(step.batch_capacity, step.token_capacity)));
        assert_eq!(s.request(handles[0]).unwrap().consumed_tokens, 66);
    }
}

#[test]
fn prefill_capacities_step_after_doubling_and_end_at_the_chunk() {
    let capacities = |chunk| super::prefill_capacities(chunk).collect::<Vec<_>>();
    assert_eq!(capacities(64), [64]);
    assert_eq!(capacities(1000), [128, 256, 512, 1000]);
    assert_eq!(
        capacities(4096),
        [128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    );
    assert_eq!(
        capacities(8192),
        [
            128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192
        ]
    );
}

fn history(tokens: usize) -> PrefixHistory {
    let mut history = PrefixHistory::new([9; 32]);
    history
        .append(&(0..tokens as u32).collect::<Vec<_>>(), &[])
        .unwrap();
    history
}
fn free(scheduler: &RequestScheduler) -> (usize, usize) {
    (scheduler.free_page_count(), scheduler.free_snapshot_count())
}
fn host_scheduler(active: usize, pages: u32, snapshots: u32) -> RequestScheduler {
    let mut scheduler = scheduler(active, pages, snapshots);
    scheduler.enable_publications();
    scheduler
}
fn inbound(scheduler: &mut RequestScheduler, spec: RequestSpec, consumed: u64) -> Admitted {
    let plan = scheduler.plan_admission(&spec).unwrap();
    let key = plan.history().key_at(consumed).unwrap();
    let pages = (consumed / 64 - plan.resident_tokens() / 64) as usize;
    scheduler
        .admit_planned(
            spec,
            plan,
            Some(InboundRequest {
                key,
                consumed_tokens: consumed,
                pages,
            }),
        )
        .unwrap()
}
/// Publishes a 64-token prompt as a terminal checkpoint through the ordinary path.
fn seed_checkpoint(scheduler: &mut RequestScheduler, fence: u64) -> CheckpointId {
    let seed = scheduler.admit(spec(64, 1)).unwrap();
    init(scheduler, fence);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(fence + 1)).unwrap();
    scheduler
        .complete(
            Fence(fence + 1),
            &records(seed, fence + 1, 64, 1, 88, 2),
            Default::default(),
        )
        .unwrap();
    freeze(scheduler, seed, fence + 2);
    scheduler.finish(seed).unwrap().unwrap()
}

#[test]
fn disabled_publication_tracking_keeps_gpu_reuse_without_allocating_a_notification_queue() {
    let mut scheduler = scheduler(1, 4, 2);
    for round in 0..32 {
        let checkpoint = seed_checkpoint(&mut scheduler, round * 3 + 1);
        assert_eq!(scheduler.stats().snapshot_publications, round + 1);
        assert_eq!(scheduler.publications.capacity(), 0);
        assert!(scheduler.take_publications().is_empty());
        let hit = scheduler.admit(spec(64, 2)).unwrap();
        let request = scheduler.request(hit).unwrap();
        assert_eq!(request.phase, RequestPhase::Restore);
        assert_eq!(request.prefix(), Some(checkpoint));
        assert_eq!(request.reused_prompt_tokens(), 64);
        scheduler.cancel(hit).unwrap();
        scheduler.ensure_resources(4, 2).unwrap();
        assert_eq!(free(&scheduler), (4, 2));
        assert_eq!(scheduler.stats().snapshot_evictions, round + 1);
    }
    let stats = scheduler.stats();
    assert_eq!((stats.cache_hits, stats.reused_tokens), (32, 32 * 64));
    assert_eq!(
        (
            stats.host_lookups,
            stats.host_hits,
            stats.host_reused_tokens
        ),
        (0, 0, 0)
    );
    assert!(scheduler.tickets.is_empty());
    assert_eq!(scheduler.tickets.capacity(), 0);
}

#[test]
fn inbound_request_waits_unscheduled_then_restores_like_a_gpu_prefix_hit() {
    let mut scheduler = host_scheduler(2, 8, 4);
    let admitted = inbound(&mut scheduler, spec(65, 2), 64);
    let handle = admitted.handle;
    let ticket = admitted.ticket.unwrap();
    let key = history(65).key_at(64).unwrap();
    assert_eq!(ticket.key, key);
    assert_eq!(ticket.pages.len(), 1);
    assert_eq!(ticket.slot, scheduler.snapshot_slot(SnapshotId(0)));
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.phase, RequestPhase::Inbound);
    assert_eq!(request.consumed_tokens, 64);
    assert_eq!(request.reused_prompt_tokens(), 64);
    assert!(request.prefix().is_none());
    assert!(request.prefix_snapshot().is_none());
    assert_eq!(request.pages().count(), 0);
    assert!(scheduler.plan().is_none());
    let stats = scheduler.stats();
    assert_eq!(
        (
            stats.active_requests,
            stats.inbound_waiting,
            stats.free_slots
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (stats.cache_lookups, stats.cache_hits, stats.reused_tokens),
        (1, 0, 0)
    );
    assert_eq!((stats.host_hits, stats.host_reused_tokens), (1, 64));
    // The ticket holds one page and one snapshot; the request holds one reserved
    // page; checkpoint retention reserves no destination yet.
    assert_eq!(free(&scheduler), (6, 3));
    assert!(scheduler.take_publications().is_empty());

    assert!(scheduler.finish_inbound(key, true).unwrap().is_empty());
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.phase, RequestPhase::Restore);
    let restored = request.prefix().unwrap();
    assert_eq!(
        request
            .prefix_snapshot()
            .map(|s| scheduler.snapshot_slot(s)),
        Some(ticket.slot)
    );
    assert_eq!(request.pages().collect::<Vec<_>>(), ticket.pages);
    assert_eq!(request.consumed_tokens, 64);
    assert_eq!(request.reused_prompt_tokens(), 64);
    let checkpoint = scheduler.cache().checkpoint(restored).unwrap();
    assert_eq!(checkpoint.prefix(), key);
    assert!(checkpoint.ready());
    assert_eq!(checkpoint.tail_pages(), &ticket.pages[..]);
    assert_eq!(checkpoint.pin_count(), 1);
    let stats = scheduler.stats();
    assert_eq!(
        (
            stats.inbound_waiting,
            stats.snapshot_publications,
            stats.cached_checkpoints
        ),
        (0, 1, 1)
    );
    assert_eq!(free(&scheduler), (6, 3));
    assert_eq!(
        scheduler.take_publications(),
        vec![PublishedCheckpoint {
            id: restored,
            key,
            parent: None,
            bucket: key.digest,
            chain_pages: ticket.pages.clone(),
            tail_pages: 1,
            slot: ticket.slot,
        }]
    );
    assert!(scheduler.take_publications().is_empty());

    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!(
        (
            step.kind,
            step.lanes[0].start_position,
            step.lanes[0].tokens
        ),
        (GraphKind::Prefill, 64, 1)
    );
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 65, 1, 77, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Decode
    );
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Decode);
    scheduler.commit(&step, Fence(4)).unwrap();
    scheduler
        .complete(
            Fence(4),
            &records(handle, 4, 66, 2, 78, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, handle, 5);
    let terminal = scheduler.finish(handle).unwrap().unwrap();
    let published = scheduler.take_publications();
    assert_eq!(published.len(), 2);
    let prompt = &published[0];
    assert_eq!(prompt.key, history(65).key_at(65).unwrap());
    assert_eq!(prompt.parent, Some(key));
    assert_eq!(prompt.bucket, key.digest);
    assert_eq!(prompt.chain_pages, ticket.pages);
    assert_eq!(prompt.tail_pages, 0);
    assert_eq!(prompt.slot, scheduler.snapshot_slot(SnapshotId(1)));
    let end = &published[1];
    assert_eq!(end.id, terminal);
    assert_eq!(
        end.key,
        scheduler.cache().checkpoint(terminal).unwrap().prefix()
    );
    assert_eq!(end.key.consumed_tokens, 66);
    assert_eq!(end.parent, Some(prompt.key));
    assert_eq!(end.bucket, key.digest);
    assert_eq!(end.chain_pages, ticket.pages);
    assert_eq!(end.tail_pages, 0);
    assert_eq!(end.slot, scheduler.snapshot_slot(SnapshotId(2)));
    // Three checkpoints own one page and three snapshots; evicting them frees every id.
    assert_eq!(free(&scheduler), (7, 1));
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
    assert!(scheduler.take_publications().is_empty());
}

#[test]
fn inbound_full_prompt_hit_resumes_without_prefill() {
    let mut scheduler = host_scheduler(1, 4, 2);
    let admitted = inbound(&mut scheduler, spec(64, 2), 64);
    let handle = admitted.handle;
    let ticket = admitted.ticket.unwrap();
    assert!(scheduler.request(handle).unwrap().prompt_snapshot.is_none());
    assert_eq!(free(&scheduler), (2, 1));
    scheduler.finish_inbound(ticket.key, true).unwrap();
    let restored = scheduler.request(handle).unwrap().prefix().unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!(
        (step.kind, step.lanes[0].start_position),
        (GraphKind::Resume, 64)
    );
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 1, 77, 1),
            Default::default(),
        )
        .unwrap();
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Decode
    );
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Decode);
    scheduler.commit(&step, Fence(3)).unwrap();
    scheduler
        .complete(
            Fence(3),
            &records(handle, 3, 65, 2, 78, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, handle, 4);
    let terminal = scheduler.finish(handle).unwrap().unwrap();
    assert_eq!(
        scheduler
            .take_publications()
            .iter()
            .map(|p| (p.id, p.parent, p.tail_pages, p.chain_pages.clone()))
            .collect::<Vec<_>>(),
        [
            (restored, None, 1, ticket.pages.clone()),
            (terminal, Some(ticket.key), 0, ticket.pages.clone())
        ]
    );
    assert_eq!(free(&scheduler), (3, 0));
    scheduler.ensure_resources(4, 2).unwrap();
    assert_eq!(free(&scheduler), (4, 2));
}

#[test]
fn inbound_restore_above_a_resident_checkpoint_pins_the_ancestor_until_completion() {
    let mut scheduler = host_scheduler(2, 8, 4);
    let ancestor = seed_checkpoint(&mut scheduler, 1);
    let ancestor_key = history(64).key_at(64).unwrap();
    let ancestor_pages = scheduler.cache().page_chain(ancestor).unwrap();
    assert_eq!(
        scheduler
            .take_publications()
            .iter()
            .map(|p| p.id)
            .collect::<Vec<_>>(),
        [ancestor]
    );
    assert_eq!(free(&scheduler), (7, 3));
    let request = spec(130, 2);
    let plan = scheduler.plan_admission(&request).unwrap();
    assert_eq!(plan.resident_tokens(), 64);
    let key = plan.history().key_at(128).unwrap();
    let admitted = scheduler
        .admit_planned(
            request,
            plan,
            Some(InboundRequest {
                key,
                consumed_tokens: 128,
                pages: 1,
            }),
        )
        .unwrap();
    let handle = admitted.handle;
    let ticket = admitted.ticket.unwrap();
    assert_eq!(ticket.pages.len(), 1);
    let stats = scheduler.stats();
    assert_eq!(
        (stats.cache_lookups, stats.cache_hits, stats.reused_tokens),
        (2, 1, 64)
    );
    assert_eq!((stats.host_hits, stats.host_reused_tokens), (1, 64));
    // Ticket: one page and one snapshot. Request: one page beyond the two
    // complete pages. Checkpoint retention has no reserved destination.
    assert_eq!(free(&scheduler), (5, 2));
    assert_eq!(
        scheduler.cache().checkpoint(ancestor).unwrap().pin_count(),
        1
    );
    assert!(scheduler.ensure_resources(8, 4).is_err());
    assert!(scheduler.cache().checkpoint(ancestor).is_ok());
    assert_eq!(scheduler.stats().snapshot_evictions, 0);
    assert_eq!(free(&scheduler), (5, 2));

    scheduler.finish_inbound(key, true).unwrap();
    let request = scheduler.request(handle).unwrap();
    let restored = request.prefix().unwrap();
    let mut chain = ancestor_pages.clone();
    chain.extend(&ticket.pages);
    assert_eq!(request.pages().collect::<Vec<_>>(), chain);
    assert_eq!(request.consumed_tokens, 128);
    assert_eq!(request.reused_prompt_tokens(), 128);
    assert_eq!(request.phase, RequestPhase::Restore);
    let checkpoint = scheduler.cache().checkpoint(restored).unwrap();
    assert_eq!(checkpoint.parent(), Some(ancestor));
    assert_eq!(checkpoint.tail_pages(), &ticket.pages[..]);
    assert_eq!(checkpoint.pin_count(), 1);
    // The ticket's pin is released; the waiter's pin on the child reaches the ancestor.
    assert_eq!(
        scheduler.cache().checkpoint(ancestor).unwrap().pin_count(),
        1
    );
    assert_eq!(
        scheduler.take_publications(),
        vec![PublishedCheckpoint {
            id: restored,
            key,
            parent: Some(ancestor_key),
            bucket: key.digest,
            chain_pages: chain.clone(),
            tail_pages: 1,
            slot: ticket.slot,
        }]
    );
    init(&mut scheduler, 4);
    let step = scheduler.plan().unwrap();
    assert_eq!(
        (
            step.kind,
            step.lanes[0].start_position,
            step.lanes[0].tokens
        ),
        (GraphKind::Prefill, 128, 2)
    );
    scheduler.commit(&step, Fence(5)).unwrap();
    scheduler
        .complete(
            Fence(5),
            &records(handle, 5, 130, 1, 77, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 6);
    let published = scheduler.take_publications();
    assert_eq!(published.len(), 1);
    assert_eq!(
        (
            published[0].key,
            published[0].parent,
            published[0].bucket,
            published[0].tail_pages,
            &published[0].chain_pages
        ),
        (
            history(130).key_at(130).unwrap(),
            Some(key),
            key.digest,
            0,
            &chain
        )
    );
    scheduler.cancel(handle).unwrap();
    assert_eq!(
        scheduler.cache().checkpoint(ancestor).unwrap().pin_count(),
        0
    );
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
}

#[test]
fn finish_inbound_attaches_to_an_existing_checkpoint_and_frees_the_ticket() {
    let mut scheduler = host_scheduler(2, 8, 4);
    let cold = scheduler.admit(spec(64, 1)).unwrap();
    let admitted = inbound(&mut scheduler, spec(64, 2), 64);
    let ticket = admitted.ticket.unwrap();
    // The cold request prefills the same prefix and publishes it first; the
    // inbound request never appears in a step.
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!(step.lanes.len(), 1);
    assert_eq!(step.lanes[0].request, cold);
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(cold, 2, 64, 1, 88, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, cold, 3);
    let existing = scheduler.finish(cold).unwrap().unwrap();
    assert_eq!(
        scheduler.cache().checkpoint(existing).unwrap().prefix(),
        ticket.key
    );
    assert_eq!(
        scheduler
            .take_publications()
            .iter()
            .map(|p| p.id)
            .collect::<Vec<_>>(),
        [existing]
    );
    let before = free(&scheduler);
    assert!(
        scheduler
            .finish_inbound(ticket.key, true)
            .unwrap()
            .is_empty()
    );
    assert_eq!(free(&scheduler), (before.0 + 1, before.1 + 1));
    let request = scheduler.request(admitted.handle).unwrap();
    assert_eq!(request.phase, RequestPhase::Restore);
    assert_eq!(request.prefix(), Some(existing));
    assert_eq!(
        request.prefix_snapshot(),
        Some(scheduler.cache().checkpoint(existing).unwrap().snapshot())
    );
    assert_eq!(
        request.pages().collect::<Vec<_>>(),
        scheduler.cache().page_chain(existing).unwrap()
    );
    assert_eq!(
        scheduler.cache().checkpoint(existing).unwrap().pin_count(),
        1
    );
    assert_eq!(scheduler.stats().cached_checkpoints, 1);
    assert!(scheduler.take_publications().is_empty());
    assert!(scheduler.finish_inbound(ticket.key, true).is_err());
    init(&mut scheduler, 4);
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Resume);
    scheduler.cancel(admitted.handle).unwrap();
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
}

#[test]
fn inbound_alias_backing_survives_cancellation_until_transfer_completion() {
    for success in [false, true] {
        let mut scheduler = host_scheduler(2, 16, 2);
        scheduler.share_snapshot_pages(&[8..12, 10..14]).unwrap();
        let first = inbound(&mut scheduler, spec(65, 2), 64);
        let ticket = first.ticket.unwrap();
        let second = inbound(&mut scheduler, spec(66, 2), 64);
        assert!(second.ticket.is_none());
        assert_eq!(free(&scheduler), (9, 1));
        scheduler.cancel(first.handle).unwrap();
        scheduler.cancel(second.handle).unwrap();
        assert_eq!(free(&scheduler), (11, 1));
        assert_eq!(scheduler.stats().snapshot_backing_pages, 4);
        scheduler.finish_inbound(ticket.key, success).unwrap();
        if success {
            assert_eq!(scheduler.stats().snapshot_backing_pages, 4);
            scheduler.ensure_resources(16, 0).unwrap();
        }
        assert_eq!(free(&scheduler), (16, 2));
        assert_eq!(scheduler.stats().snapshot_backing_pages, 0);
    }
}

#[test]
fn failed_inbound_frees_the_ticket_and_returns_waiters_for_readmission() {
    let mut scheduler = host_scheduler(2, 8, 6);
    let first = inbound(&mut scheduler, spec(65, 2), 64);
    let ticket = first.ticket.unwrap();
    let second = inbound(&mut scheduler, spec(66, 2), 64);
    assert!(second.ticket.is_none());
    assert_eq!(scheduler.stats().inbound_waiting, 2);
    assert_eq!(scheduler.stats().free_slots, 0);
    // Ticket: one page and one snapshot. Each waiter holds one page.
    assert_eq!(free(&scheduler), (5, 5));
    let returned = scheduler.finish_inbound(ticket.key, false).unwrap();
    assert_eq!(returned, [first.handle, second.handle]);
    assert!(scheduler.request(first.handle).is_err());
    assert!(scheduler.request(second.handle).is_err());
    let stats = scheduler.stats();
    assert_eq!(
        (
            stats.active_requests,
            stats.inbound_waiting,
            stats.free_slots,
            stats.retiring_slots,
            stats.cached_checkpoints
        ),
        (0, 0, 2, 0, 0)
    );
    assert_eq!(free(&scheduler), (8, 6));
    assert!(scheduler.finish_inbound(ticket.key, false).is_err());
    assert!(scheduler.take_publications().is_empty());
    let again = scheduler.admit(spec(65, 2)).unwrap();
    assert_eq!(
        scheduler.request(again).unwrap().phase,
        RequestPhase::Initialize
    );
}

#[test]
fn waiters_sharing_one_ticket_restore_together_and_both_complete() {
    let mut scheduler = host_scheduler(2, 8, 4);
    let first = inbound(&mut scheduler, spec(65, 1), 64);
    let ticket = first.ticket.unwrap();
    let second = inbound(&mut scheduler, spec(66, 1), 64);
    assert!(second.ticket.is_none());
    let stats = scheduler.stats();
    assert_eq!(
        (
            stats.host_hits,
            stats.host_reused_tokens,
            stats.inbound_waiting
        ),
        (2, 128, 2)
    );
    assert_eq!(free(&scheduler), (5, 3));
    assert!(scheduler.plan().is_none());
    scheduler.finish_inbound(ticket.key, true).unwrap();
    let restored = scheduler.request(first.handle).unwrap().prefix().unwrap();
    assert_eq!(
        scheduler.request(second.handle).unwrap().prefix(),
        Some(restored)
    );
    assert_eq!(
        scheduler.cache().checkpoint(restored).unwrap().pin_count(),
        2
    );
    for handle in [first.handle, second.handle] {
        let request = scheduler.request(handle).unwrap();
        assert_eq!(request.phase, RequestPhase::Restore);
        assert_eq!(request.pages().collect::<Vec<_>>(), ticket.pages);
        assert_eq!(request.consumed_tokens, 64);
    }
    init(&mut scheduler, 1);
    let step = scheduler.phase_plan(GraphKind::Prefill, Some(first.handle));
    assert_eq!(
        (step.kind, step.lanes[0].request),
        (GraphKind::Prefill, first.handle)
    );
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(first.handle, 2, 65, 1, 77, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, first.handle, 3);
    let first_terminal = scheduler.finish(first.handle).unwrap().unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!(
        (step.kind, step.lanes[0].request, step.lanes[0].tokens),
        (GraphKind::Prefill, second.handle, 2)
    );
    scheduler.commit(&step, Fence(4)).unwrap();
    scheduler
        .complete(
            Fence(4),
            &records(second.handle, 4, 66, 1, 78, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, second.handle, 5);
    let second_terminal = scheduler.finish(second.handle).unwrap().unwrap();
    assert_ne!(first_terminal, second_terminal);
    for terminal in [first_terminal, second_terminal] {
        assert_eq!(
            scheduler.cache().checkpoint(terminal).unwrap().parent(),
            Some(restored)
        );
    }
    assert_eq!(
        scheduler.cache().checkpoint(restored).unwrap().pin_count(),
        0
    );
    assert_eq!(
        scheduler
            .take_publications()
            .iter()
            .map(|p| (p.id, p.parent))
            .collect::<Vec<_>>(),
        [
            (restored, None),
            (first_terminal, Some(ticket.key)),
            (second_terminal, Some(ticket.key))
        ]
    );
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
}

#[test]
fn cancelled_inbound_waiter_leaves_ticket_resources_until_finish_inbound() {
    let mut scheduler = host_scheduler(2, 8, 4);
    let admitted = inbound(&mut scheduler, spec(65, 2), 64);
    let ticket = admitted.ticket.unwrap();
    scheduler.cancel(admitted.handle).unwrap();
    assert!(scheduler.request(admitted.handle).is_err());
    let stats = scheduler.stats();
    assert_eq!(
        (
            stats.active_requests,
            stats.inbound_waiting,
            stats.free_slots,
            stats.retiring_slots
        ),
        (0, 0, 2, 0)
    );
    assert_eq!(free(&scheduler), (7, 3));
    assert!(
        scheduler
            .finish_inbound(ticket.key, true)
            .unwrap()
            .is_empty()
    );
    let published = scheduler.take_publications();
    assert_eq!(published.len(), 1);
    let restored = published[0].id;
    assert_eq!(published[0].chain_pages, ticket.pages);
    assert_eq!(
        scheduler.cache().checkpoint(restored).unwrap().pin_count(),
        0
    );
    assert_eq!(free(&scheduler), (7, 3));
    let hit = scheduler.admit(spec(65, 2)).unwrap();
    let request = scheduler.request(hit).unwrap();
    assert_eq!(request.prefix(), Some(restored));
    assert_eq!(request.phase, RequestPhase::Restore);
    assert_eq!(request.pages().collect::<Vec<_>>(), ticket.pages);
    scheduler.cancel(hit).unwrap();
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));

    let admitted = inbound(&mut scheduler, spec(65, 2), 64);
    let ticket = admitted.ticket.unwrap();
    scheduler.cancel(admitted.handle).unwrap();
    assert_eq!(free(&scheduler), (7, 3));
    assert!(
        scheduler
            .finish_inbound(ticket.key, false)
            .unwrap()
            .is_empty()
    );
    assert_eq!(free(&scheduler), (8, 4));
    assert_eq!(scheduler.stats().cached_checkpoints, 0);
}

#[test]
fn invalid_inbound_requests_reserve_nothing_and_planned_admission_matches_admit() {
    let mut scheduler = host_scheduler(1, 8, 4);
    let request = spec(65, 2);
    let key = history(65).key_at(64).unwrap();
    let root = history(65).key_at(0).unwrap();
    for inbound in [
        InboundRequest {
            key,
            consumed_tokens: 65,
            pages: 1,
        },
        InboundRequest {
            key: PrefixKey {
                consumed_tokens: 64,
                digest: [0; 16],
            },
            consumed_tokens: 64,
            pages: 1,
        },
        InboundRequest {
            key,
            consumed_tokens: 64,
            pages: 2,
        },
        InboundRequest {
            key: root,
            consumed_tokens: 0,
            pages: 0,
        },
    ] {
        let plan = scheduler.plan_admission(&request).unwrap();
        assert!(
            scheduler
                .admit_planned(request.clone(), plan, Some(inbound))
                .is_err()
        );
        assert_eq!(free(&scheduler), (8, 4));
        assert_eq!(scheduler.stats().active_requests, 0);
    }
    assert_eq!(scheduler.stats().host_hits, 0);
    let plan = scheduler.plan_admission(&request).unwrap();
    assert_eq!(plan.resident_tokens(), 0);
    // Lookups count once per admitted request; the failed attempts above count none.
    assert_eq!(scheduler.stats().cache_lookups, 0);
    let admitted = scheduler
        .admit_planned(request.clone(), plan, None)
        .unwrap();
    assert!(admitted.ticket.is_none());
    assert_eq!(scheduler.stats().cache_lookups, 1);
    let live = scheduler.request(admitted.handle).unwrap();
    assert_eq!(live.phase, RequestPhase::Initialize);
    assert_eq!(live.consumed_tokens, 0);
    assert!(scheduler.plan_admission(&request).is_err());
    scheduler.cancel(admitted.handle).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
}

#[test]
fn pinned_checkpoints_survive_eviction_pressure_until_unpinned() {
    let mut scheduler = host_scheduler(1, 4, 2);
    let checkpoint = seed_checkpoint(&mut scheduler, 1);
    assert_eq!(free(&scheduler), (3, 1));
    assert!(scheduler.unpin_checkpoint(checkpoint).is_err());
    scheduler.pin_checkpoint(checkpoint).unwrap();
    assert_eq!(
        scheduler
            .cache()
            .checkpoint(checkpoint)
            .unwrap()
            .pin_count(),
        1
    );
    assert!(scheduler.ensure_resources(4, 2).is_err());
    assert!(scheduler.cache().checkpoint(checkpoint).is_ok());
    assert_eq!(scheduler.stats().snapshot_evictions, 0);
    scheduler.unpin_checkpoint(checkpoint).unwrap();
    scheduler.ensure_resources(4, 2).unwrap();
    assert!(scheduler.cache().checkpoint(checkpoint).is_err());
    assert_eq!(free(&scheduler), (4, 2));
    assert!(scheduler.pin_checkpoint(checkpoint).is_err());
    // A checkpoint retired before its publication is taken is omitted.
    assert!(scheduler.take_publications().is_empty());
    scheduler.record_host_lookup();
    scheduler.record_host_lookup();
    assert_eq!(scheduler.stats().host_lookups, 2);
}

#[test]
fn take_publications_reports_boundary_and_terminal_checkpoints_once() {
    let mut scheduler = host_scheduler(1, 8, 4);
    let handle = scheduler
        .admit(RequestSpec {
            message_boundaries: vec![64],
            ..spec(130, 2)
        })
        .unwrap();
    let history = history(130);
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!((step.kind, step.lanes[0].tokens), (GraphKind::Prefill, 64));
    scheduler.commit(&step, Fence(2)).unwrap();
    scheduler
        .complete(
            Fence(2),
            &records(handle, 2, 64, 0, -1, 1),
            Default::default(),
        )
        .unwrap();
    assert!(scheduler.take_publications().is_empty());
    let first_page = scheduler
        .request(handle)
        .unwrap()
        .pages()
        .collect::<Vec<_>>();
    assert_eq!(first_page.len(), 1);
    retain_prompt(&mut scheduler, 3);
    let system = scheduler.request(handle).unwrap().prefix().unwrap();
    assert_eq!(
        scheduler.take_publications(),
        vec![PublishedCheckpoint {
            id: system,
            key: history.key_at(64).unwrap(),
            parent: None,
            bucket: history.key_at(64).unwrap().digest,
            chain_pages: first_page.clone(),
            tail_pages: 1,
            slot: scheduler.snapshot_slot(scheduler.cache().checkpoint(system).unwrap().snapshot()),
        }]
    );
    let step = scheduler.plan().unwrap();
    assert_eq!(
        (
            step.kind,
            step.lanes[0].start_position,
            step.lanes[0].tokens
        ),
        (GraphKind::Prefill, 64, 66)
    );
    scheduler.commit(&step, Fence(4)).unwrap();
    scheduler
        .complete(
            Fence(4),
            &records(handle, 4, 130, 1, 77, 1),
            Default::default(),
        )
        .unwrap();
    retain_prompt(&mut scheduler, 5);
    let prompt = scheduler.request(handle).unwrap().prefix().unwrap();
    let pages = scheduler
        .request(handle)
        .unwrap()
        .pages()
        .collect::<Vec<_>>();
    assert_eq!(pages.len(), 3);
    assert_eq!(pages[0], first_page[0]);
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Decode);
    scheduler.commit(&step, Fence(6)).unwrap();
    scheduler
        .complete(
            Fence(6),
            &records(handle, 6, 131, 2, 78, 2),
            Default::default(),
        )
        .unwrap();
    freeze(&mut scheduler, handle, 7);
    let terminal = scheduler.finish(handle).unwrap().unwrap();
    let published = scheduler.take_publications();
    assert_eq!(
        published,
        vec![
            PublishedCheckpoint {
                id: prompt,
                key: history.key_at(130).unwrap(),
                parent: Some(history.key_at(64).unwrap()),
                bucket: history.key_at(128).unwrap().digest,
                chain_pages: pages[..2].to_vec(),
                tail_pages: 1,
                slot: scheduler
                    .snapshot_slot(scheduler.cache().checkpoint(prompt).unwrap().snapshot()),
            },
            PublishedCheckpoint {
                id: terminal,
                key: scheduler.cache().checkpoint(terminal).unwrap().prefix(),
                parent: Some(history.key_at(130).unwrap()),
                bucket: history.key_at(128).unwrap().digest,
                chain_pages: pages[..2].to_vec(),
                tail_pages: 0,
                slot: scheduler
                    .snapshot_slot(scheduler.cache().checkpoint(terminal).unwrap().snapshot()),
            },
        ]
    );
    assert_eq!(published[1].key.consumed_tokens, 131);
    assert_eq!(published[1].tail(), &[]);
    assert!(scheduler.take_publications().is_empty());
    assert_eq!(free(&scheduler), (6, 1));
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
}

#[test]
fn shutdown_release_drops_inbound_tickets_and_their_ancestor_pins() {
    let mut scheduler = host_scheduler(2, 8, 4);
    let ancestor = seed_checkpoint(&mut scheduler, 1);
    let admitted = inbound(&mut scheduler, spec(130, 2), 128);
    let ticket = admitted.ticket.unwrap();
    assert_eq!(free(&scheduler), (5, 2));
    assert_eq!(
        scheduler.cache().checkpoint(ancestor).unwrap().pin_count(),
        1
    );
    scheduler
        .release_after_shutdown(&Shutdown {
            drained: true,
            error: None,
        })
        .unwrap();
    assert!(scheduler.request(admitted.handle).is_err());
    assert!(scheduler.finish_inbound(ticket.key, true).is_err());
    assert_eq!(
        scheduler.cache().checkpoint(ancestor).unwrap().pin_count(),
        0
    );
    assert_eq!(free(&scheduler), (7, 3));
    scheduler.ensure_resources(8, 4).unwrap();
    assert_eq!(free(&scheduler), (8, 4));
}

/// Plans, commits, and completes one step: state transfers complete as copies, and every
/// other lane consumes its tokens and samples one more.
fn sample_round(scheduler: &mut RequestScheduler, fence: u64) -> ScheduledStep {
    let plan = scheduler.plan().unwrap();
    let step = plan.step.clone();
    scheduler.commit(&plan, Fence(fence)).unwrap();
    if matches!(
        step.kind,
        GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot | GraphKind::Vision
    ) {
        let completions = state_records(scheduler, &step, fence);
        scheduler
            .complete(Fence(fence), &completions, Default::default())
            .unwrap();
        return step;
    }
    let mut output = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        output[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: (lane.start_position + lane.tokens as u64) as i64,
            generated_tokens: request.generated_tokens as i64 + 1,
            token_id: 100,
            status: 1,
            ..Default::default()
        };
    }
    scheduler
        .complete(Fence(fence), &output, Default::default())
        .unwrap();
    step
}

fn growable(tokens: usize, max_new: u32) -> RequestSpec {
    RequestSpec {
        growable_output: true,
        ..spec(tokens, max_new)
    }
}

#[test]
fn growable_requests_reserve_a_page_of_output_and_grow_while_decoding() {
    let mut scheduler = scheduler(2, 4, 4);
    let handle = scheduler.admit(growable(60, 400)).unwrap();
    // The prompt and one page of output, not the eight pages its output limit spans.
    assert_eq!(scheduler.free_page_count(), 2);
    init(&mut scheduler, 1);
    let mut fence = 2;
    while scheduler.request(handle).unwrap().consumed_tokens < 70 {
        sample_round(&mut scheduler, fence);
        fence += 1;
    }
    // Two rounds and a page ahead of its position, the request took a third page.
    assert_eq!(scheduler.free_page_count(), 1);
    // A request without an output limit waits for pages instead of reserving them.
    assert!(scheduler.admit(growable(60, 400)).is_err());
    assert_eq!(scheduler.stats().preemptions, 0);
}

#[test]
fn starved_growable_request_preempts_the_newest_and_resumes_from_its_checkpoint() {
    let mut scheduler = scheduler(2, 6, 4);
    let older = scheduler.admit(growable(60, 400)).unwrap();
    let newer = scheduler.admit(growable(60, 400)).unwrap();
    assert_eq!(scheduler.free_page_count(), 2);
    let mut fence = 1;
    while !scheduler.request(newer).unwrap().preempted() {
        sample_round(&mut scheduler, fence);
        fence += 1;
        assert!(fence < 1000, "no request ran out of pages");
    }
    // The pool cannot hold both histories; the newest growable request yields its pages.
    assert!(!scheduler.request(older).unwrap().preempted());
    let request = scheduler.request(newer).unwrap();
    assert_eq!(request.phase, RequestPhase::Snapshot);
    assert_eq!(scheduler.stats().preemptions, 1);
    let tokens = request.tokens().to_vec();
    let remaining = 400 - request.generated_tokens;
    freeze(&mut scheduler, newer, fence);
    let checkpoint = scheduler.finish(newer).unwrap().unwrap();
    scheduler.cancel(older).unwrap();
    // Readmitting the history restores the published checkpoint instead of recomputing it.
    let resumed = scheduler
        .admit(RequestSpec {
            prompt: tokens,
            growable_output: true,
            ..spec(1, remaining)
        })
        .unwrap();
    let request = scheduler.request(resumed).unwrap();
    assert_eq!(request.phase, RequestPhase::Restore);
    assert_eq!(request.prefix(), Some(checkpoint));
}

#[test]
fn growable_request_longer_than_the_pool_finishes_instead_of_yielding() {
    let mut scheduler = scheduler(1, 4, 4);
    let handle = scheduler.admit(growable(60, 400)).unwrap();
    let mut fence = 1;
    while scheduler.request(handle).unwrap().phase != RequestPhase::Finishing {
        sample_round(&mut scheduler, fence);
        fence += 1;
        assert!(fence < 1000, "the request never finished");
    }
    // Four pages hold 256 positions; it ends there, not through repeated preemption.
    let request = scheduler.request(handle).unwrap();
    assert!(!request.preempted() && request.consumed_tokens + 2 > 4 * 64);
    assert_eq!(scheduler.stats().preemptions, 0);
    scheduler.finish(handle).unwrap();
    // Admission rejects a prompt that the whole pool cannot hold instead of queueing it.
    let error = scheduler.admit(growable(4 * 64, 10)).unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Admission(crate::AdmissionError::Context(_))
    ));
}

#[test]
fn requests_that_cannot_grow_keep_their_reservation_and_are_never_preempted() {
    let mut scheduler = scheduler(2, 8, 4);
    let reserved = scheduler.admit(spec(60, 190)).unwrap();
    assert_eq!(scheduler.free_page_count(), 8 - 4);
    let grower = scheduler.admit(growable(60, 400)).unwrap();
    assert_eq!(scheduler.free_page_count(), 2);
    let mut fence = 1;
    while !scheduler.request(grower).unwrap().preempted() {
        sample_round(&mut scheduler, fence);
        fence += 1;
        assert!(fence < 1000, "the growable request never ran out of pages");
    }
    assert!(!scheduler.request(reserved).unwrap().preempted());
    assert_eq!(scheduler.stats().preemptions, 1);
}

#[test]
fn image_slots_are_independent_and_retire_after_the_last_gpu_use() {
    for page_tokens in [64, 128, 256] {
        let mut scheduler = scheduler_with_geometry(4, 64, 8, page_tokens, 1);
        let text = scheduler.admit(spec(8, 2)).unwrap();
        let image = scheduler.admit(spec(8, 2)).unwrap();
        scheduler.set_vision_images(image, &[256]).unwrap();
        assert_eq!(image.slot, 1);
        assert_eq!(scheduler.request(image).unwrap().image_slot, Some(0));
        assert_eq!(scheduler.request(text).unwrap().image_slot, None);
        let waiting = scheduler.admit(spec(8, 2)).unwrap();
        assert!(matches!(scheduler.set_vision_images(waiting, &[256]),
                         Err(crate::Error::Admission(crate::AdmissionError::Capacity(_)))));
        init(&mut scheduler, 1);
        scheduler.cancel(text).unwrap();
        scheduler.cancel(waiting).unwrap();
        let step = scheduler.plan().unwrap();
        assert_eq!(step.kind, GraphKind::Vision);
        scheduler.commit(&step, Fence(2)).unwrap();
        scheduler.cancel(image).unwrap();
        assert!(!scheduler.has_available_image_slot());
        scheduler.complete(Fence(2), &records(image, 2, 0, 0, -1, 1), Default::default()).unwrap();
        assert!(scheduler.has_available_image_slot());
        let replacement = scheduler.admit(spec(8, 2)).unwrap();
        scheduler.set_vision_images(replacement, &[256]).unwrap();
        assert_eq!(scheduler.request(replacement).unwrap().image_slot, Some(0));
        assert!(scheduler.set_vision_images(replacement, &[256]).is_err());
        scheduler.cancel(replacement).unwrap();
        assert!(scheduler.has_available_image_slot());
    }
}

#[test]
fn conditioned_feature_ranges_compact_without_changing_canonical_indices() {
    let mut scheduler = scheduler(1, 8, 2);
    let handle = scheduler.admit(spec(65, 2)).unwrap();
    scheduler
        .set_conditioned_features(handle, &[128..192, 256..320])
        .unwrap();
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.feature_index(-1).unwrap(), -1);
    assert_eq!(request.feature_index(128).unwrap(), 0);
    assert_eq!(request.feature_index(191).unwrap(), 63);
    assert_eq!(request.feature_index(256).unwrap(), 64);
    assert_eq!(request.feature_index(319).unwrap(), 127);
    assert!(request.feature_index(127).is_err());
    assert!(request.feature_index(192).is_err());
    assert!(
        scheduler
            .set_conditioned_features(handle, &[0..64])
            .is_err()
    );
}

#[test]
fn image_preparation_wait_does_not_block_unrelated_prefill() {
    let mut scheduler = scheduler(2, 16, 4);
    let image = scheduler.admit(spec(65, 2)).unwrap();
    scheduler.set_vision_images(image, &[256]).unwrap();
    scheduler.conditioned_inputs_ready(image, false).unwrap();
    let text = scheduler.admit(spec(66, 2)).unwrap();
    init(&mut scheduler, 1);
    let step = scheduler.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Prefill);
    assert!(step.lanes.iter().all(|lane| lane.request == text));
    scheduler.cancel(text).unwrap();
    assert!(scheduler.plan().is_none());
    scheduler.conditioned_inputs_ready(image, true).unwrap();
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Vision);
}

#[test]
fn scored_requests_bypass_prefix_lookup_and_keep_their_own_prefill_catalog() {
    let mut scheduler = scheduler(4, 128, 16);
    let mut scored = spec(700, 0);
    scored.prompt_logprobs = true;
    assert!(scheduler.admit(scored.clone()).is_err());
    scheduler.scored_prefill_graphs(&[128, 512]).unwrap();
    let mut graphs = duration_graphs(4);
    let scored_graphs: Vec<_> = graphs.iter().filter(|g| g.kind == GraphKind::Prefill
        && [128, 512].contains(&g.token_capacity)).cloned().map(|mut graph| {
            graph.kind = GraphKind::ScoredPrefill;
            graph.name = format!("scored-{}", graph.name);
            graph
        }).collect();
    graphs.extend(scored_graphs);
    scheduler.measured_graph_durations(&graphs).unwrap();
    let before = scheduler.stats().cache_lookups;
    let handle = scheduler.admit(scored).unwrap();
    assert_eq!(scheduler.stats().cache_lookups, before);
    assert_eq!(scheduler.request(handle).unwrap().reused_prompt_tokens(), 0);
    init(&mut scheduler, 1);
    let mut fence = 2;
    let mut consumed = 0;
    while consumed < 700 {
        let step = scheduler.plan().unwrap();
        assert_eq!(step.kind, GraphKind::ScoredPrefill);
        assert!([128, 512].contains(&step.token_capacity));
        consumed += step.lanes[0].tokens;
        scheduler.commit(&step, Fence(fence)).unwrap();
        scheduler.complete(Fence(fence), &records(handle, fence, consumed as i64, 0, -1,
            if consumed == 700 { 2 } else { 1 }), Default::default()).unwrap();
        fence += 1;
    }
    assert_eq!(scheduler.request(handle).unwrap().generated_tokens, 0);
    assert_eq!(scheduler.request(handle).unwrap().phase, RequestPhase::Snapshot);
    let snapshot = scheduler.plan().unwrap();
    scheduler.commit(&snapshot, Fence(fence)).unwrap();
    let done = state_records(&scheduler, &snapshot, fence);
    scheduler.complete(Fence(fence), &done, Default::default()).unwrap();
    scheduler.finish(handle).unwrap();
    assert!(scheduler.stats().snapshot_publications > 0);
    let mut again = spec(700, 1);
    again.prompt_logprobs = true;
    let scored = scheduler.plan_admission(&again).unwrap();
    assert!(scored.lookup.checkpoint.is_none());
    again.prompt_logprobs = false;
    let ordinary = scheduler.plan_admission(&again).unwrap();
    assert!(ordinary.lookup.checkpoint.is_some());
}
