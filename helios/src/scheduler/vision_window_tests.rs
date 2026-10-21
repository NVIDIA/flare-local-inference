use super::*;
use crate::execution::ExecutionTiming;

fn scheduler(page_tokens: usize, rows: usize) -> RequestScheduler {
    scheduler_with_active(page_tokens, rows, 1)
}

fn scheduler_with_active(page_tokens: usize, rows: usize, active: usize) -> RequestScheduler {
    scheduler_with_resources(page_tokens, rows, active, 16)
}

fn scheduler_with_resources(
    page_tokens: usize,
    rows: usize,
    active: usize,
    snapshots: u32,
) -> RequestScheduler {
    scheduler_with_arenas(page_tokens, rows, active, snapshots, active)
}

fn scheduler_with_arenas(
    page_tokens: usize,
    rows: usize,
    active: usize,
    snapshots: u32,
    arenas: usize,
) -> RequestScheduler {
    let mut scheduler = RequestScheduler::new(
        [19; 32],
        SchedulerLimits {
            request_slots: active.next_power_of_two(),
            max_active: active,
            max_sequence_tokens: 65_536,
            prefill_chunk_size: 4096,
            pages: (65_536 / page_tokens * active) as u32,
            snapshots,
        },
        SchedulerCapabilities {
            vocabulary: 248_320,
            page_tokens,
            vision: crate::model::VisionCapabilities {
                feature_slots: arenas,
                features_per_request: 16_384,
                max_images: 4,
                patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16_384],
            },
            draft_pages_per_request: 1,
            completion_inputs: vec![],
            state_rows_per_request: Some(rows),
        },
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&tests::duration_graphs(active))
        .unwrap();
    scheduler
}

fn inputs(count: usize) -> Vec<ConditionedInput> {
    inputs_with_size(count, 4096)
}

fn inputs_with_size(count: usize, size: usize) -> Vec<ConditionedInput> {
    (0..count)
        .map(|image| ConditionedInput {
            token_span: 17 + image * size..17 + (image + 1) * size,
            feature_range: image * size..(image + 1) * size,
            patches: size * 4,
        })
        .collect()
}

fn specification(inputs: &[ConditionedInput]) -> RequestSpec {
    let mut spec = tests::spec(inputs.last().unwrap().token_span.end + 23, 1);
    spec.conditioning = inputs
        .iter()
        .enumerate()
        .map(|(image, input)| ConditioningEvent {
            position: input.token_span.start as u64,
            identity: [image as u8; 32],
        })
        .collect();
    spec
}

fn admit(scheduler: &mut RequestScheduler, inputs: &[ConditionedInput]) -> RequestHandle {
    let handle = scheduler.admit(specification(inputs)).unwrap();
    scheduler.set_conditioned_inputs(handle, inputs).unwrap();
    handle
}

fn commit_step(
    scheduler: &mut RequestScheduler,
    handle: RequestHandle,
) -> (ScheduledPlan, Fence, Vec<CompletionRecord>) {
    if scheduler.request(handle).unwrap().phase == RequestPhase::Vision {
        scheduler.conditioned_inputs_ready(handle, true).unwrap();
    }
    let step = scheduler
        .plan()
        .expect("conditioned request has executable work");
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&step, fence).unwrap();
    let mut records = vec![CompletionRecord::default(); scheduler.limits.request_slots];
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        let prefill_end = lane.start_position + lane.tokens as u64;
        let emitted = usize::from(
            request.max_new_tokens > 0
                && ((step.kind.is_prefill() && prefill_end == request.prompt_tokens as u64)
                    || matches!(step.kind, GraphKind::Decode | GraphKind::Resume)),
        );
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: match step.kind {
                kind if kind.is_prefill() => prefill_end,
                GraphKind::Decode => request.consumed_tokens + 1,
                _ => request.consumed_tokens,
            } as i64,
            generated_tokens: request.generated_tokens as i64 + emitted as i64,
            token_id: if emitted == 1 { 900 } else { -1 },
            status: if emitted == 1 && request.generated_tokens + 1 == request.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
    }
    (step, fence, records)
}

fn step(scheduler: &mut RequestScheduler, handle: RequestHandle) -> ScheduledPlan {
    let (step, fence, records) = commit_step(scheduler, handle);
    scheduler
        .advance_to(scheduler.now_ns + step.estimated_duration_ns())
        .unwrap();
    let mut timing = ExecutionTiming::default();
    timing.gpu_ns[0] = Some(step.estimated_duration_ns());
    timing.wall_ns = step.estimated_duration_ns();
    let generated = scheduler.complete(fence, &records, timing).unwrap();
    for token in generated {
        scheduler.published_output(token.request, fence).unwrap();
    }
    step
}

fn advance_to_window_snapshot(scheduler: &mut RequestScheduler, handle: RequestHandle) {
    for _ in 0..100 {
        let request = scheduler.request(handle).unwrap();
        if request.phase == RequestPhase::Snapshot
            && request.consumed_tokens < request.prompt_tokens as u64
        {
            return;
        }
        step(scheduler, handle);
    }
    panic!("private window checkpoint was not reached");
}

#[test]
fn working_windows_follow_the_prefill_horizon_without_expanding_feature_storage() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let images = inputs(5);
    let owner = admit(&mut scheduler, &images);
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.conditioned_input_end(), 1,
        "a large physical arena must not force a waiter to sit through several prefill horizons");
    assert_eq!(request.conditioned_windows.len(), 5);
    assert_eq!(request.conditioned_windows.front().unwrap().prefill_end, images[1].token_span.start);
    assert_eq!(scheduler.capabilities.vision.features_per_request, 16_384);
    assert_eq!(scheduler.capabilities.vision.feature_slots, 1);
}

#[test]
fn an_image_request_waits_for_a_working_arena_without_failing_admission() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let owner = admit(&mut scheduler, &inputs(5));
    let images = inputs(1);
    let waiter = scheduler.admit(specification(&images)).unwrap();
    assert!(scheduler.set_conditioned_inputs(waiter, &images).is_ok(),
        "an otherwise valid request must wait for feature storage, not fail admission");
    assert_eq!(scheduler.request(owner).unwrap().image_slot, Some(0));
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
    assert_eq!(scheduler.stats().active_requests, 2);
}

#[test]
fn a_completed_window_hands_its_arena_to_a_waiter_before_snapshot_and_owner_reacquisition() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let images = inputs(5);
    let owner = admit(&mut scheduler, &images);
    let waiter = admit(&mut scheduler, &inputs(1));
    for _ in 0..100 {
        let (executed, fence, records) = commit_step(&mut scheduler, owner);
        let last_consumer = executed.kind.is_prefill()
            && executed.lanes.iter().any(|lane| lane.request == owner
                && lane.start_position + lane.tokens as u64 >= images[0].token_span.end as u64);
        assert_eq!(scheduler.request(owner).unwrap().image_slot, Some(0));
        assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
        assert!(scheduler.resume_conditioned_window(owner).is_err());
        scheduler.complete(fence, &records, Default::default()).unwrap();
        if last_consumer {
            break;
        }
    }
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.consumed_tokens, images[0].token_span.end as u64);
    assert_eq!(request.phase, RequestPhase::Snapshot);
    assert!(!request.conditioned_window_ready());
    assert_eq!(request.image_slot, None,
        "the last image consumer returns its arena before the private snapshot");
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
    assert!(scheduler.resume_conditioned_window(owner).is_err());
    let (snapshot, fence, records) = commit_step(&mut scheduler, owner);
    assert_eq!(snapshot.kind, GraphKind::Snapshot);
    assert_eq!(scheduler.request(owner).unwrap().image_slot, None);
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
    assert!(scheduler.resume_conditioned_window(owner).is_err());
    scheduler.complete(fence, &records, ExecutionTiming::default()).unwrap();
    assert_eq!(scheduler.request(owner).unwrap().image_slot, None);
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
    let restored_owner = scheduler.resume_conditioned_window(owner).unwrap();
    assert_eq!(restored_owner.generation, owner.generation + 1);
    assert_eq!(scheduler.request(restored_owner).unwrap().phase, RequestPhase::Restore);
    assert_eq!(scheduler.request(restored_owner).unwrap().image_slot, None,
        "returning owners must queue behind image requests already waiting");
    assert_eq!(scheduler.image_waiters, VecDeque::from([restored_owner]));
}

#[test]
fn the_final_window_releases_after_its_last_image_consumer_before_text_tail_and_decode() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let image = inputs(1);
    let mut spec = specification(&image);
    spec.prompt.extend(std::iter::repeat_n(7, 8192));
    spec.message_boundaries = vec![spec.prompt.len()];
    spec.max_new_tokens = 8;
    let owner = scheduler.admit(spec).unwrap();
    scheduler.set_conditioned_inputs(owner, &image).unwrap();
    let waiter = admit(&mut scheduler, &inputs(1));
    for _ in 0..100 {
        step(&mut scheduler, owner);
        let request = scheduler.request(owner).unwrap();
        if request.consumed_tokens >= image[0].token_span.end as u64 {
            assert!(request.consumed_tokens < request.prompt_tokens as u64);
            assert_eq!(request.phase, RequestPhase::Prefill);
            assert_eq!(request.image_slot, None,
                "completed image consumers must not pin feature storage through a text tail or decode");
            assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
            return;
        }
    }
    panic!("the last image consumer made no progress");
}

#[test]
fn an_intermediate_window_returns_features_before_its_intervening_text_tail() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let mut images = inputs(2);
    images[1].token_span.start += 8192;
    images[1].token_span.end += 8192;
    let owner = admit(&mut scheduler, &images);
    let waiter = admit(&mut scheduler, &inputs(1));
    assert_eq!(scheduler.request(owner).unwrap().conditioned_windows.len(), 2);
    for _ in 0..100 {
        step(&mut scheduler, owner);
        let request = scheduler.request(owner).unwrap();
        if request.consumed_tokens >= images[0].token_span.end as u64 {
            assert!(request.consumed_tokens < images[1].token_span.start as u64);
            assert_eq!(request.phase, RequestPhase::Prefill);
            assert!(!request.conditioned_window_ready());
            assert_eq!(request.image_slot, None,
                "an image-free text gap must not retain an intermediate window's feature lease");
            assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
            return;
        }
    }
    panic!("the intermediate window's image consumer did not finish");
}

#[test]
fn a_cpu_ready_waiter_cannot_run_vision_without_an_arena_lease() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let owner = admit(&mut scheduler, &inputs(5));
    let waiter = admit(&mut scheduler, &inputs(1));
    let initialized = step(&mut scheduler, owner);
    assert_eq!(initialized.kind, GraphKind::Init);
    assert_eq!(scheduler.request(owner).unwrap().phase, RequestPhase::Vision);
    assert_eq!(scheduler.request(waiter).unwrap().phase, RequestPhase::Vision);
    scheduler.conditioned_inputs_ready(waiter, true).unwrap();
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
    assert!(scheduler.plan().is_none(),
        "CPU readiness cannot authorize vision writes into another request's arena");
    assert_eq!(scheduler.request(waiter).unwrap().vision_patches.len(), 1);
    assert_eq!(scheduler.request(waiter).unwrap().consumed_tokens, 0);

    scheduler.cancel(owner).unwrap();
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
    let vision = step(&mut scheduler, waiter);
    assert_eq!(vision.kind, GraphKind::Vision);
    assert_eq!(vision.lanes[0].request, waiter);
    assert!(scheduler.request(waiter).unwrap().vision_patches.is_empty());
}

#[test]
fn cancelled_arena_waiters_are_removed_before_fifo_handoff_and_slot_reuse() {
    let mut scheduler = scheduler_with_arenas(128, 4, 3, 16, 1);
    let owner = admit(&mut scheduler, &inputs(5));
    let cancelled = admit(&mut scheduler, &inputs(1));
    let rightful = admit(&mut scheduler, &inputs(1));
    assert_eq!(step(&mut scheduler, owner).kind, GraphKind::Init);
    assert_eq!(scheduler.image_waiters, VecDeque::from([cancelled, rightful]));
    scheduler.conditioned_inputs_ready(rightful, true).unwrap();
    scheduler.cancel(cancelled).unwrap();
    assert_eq!(scheduler.image_waiters, VecDeque::from([rightful]));

    let mut previous = cancelled;
    for _ in 0..32 {
        let replacement = admit(&mut scheduler, &inputs(1));
        assert_eq!(replacement.slot, cancelled.slot);
        assert!(replacement.generation > previous.generation);
        assert_eq!(scheduler.image_waiters, VecDeque::from([rightful, replacement]));
        let initialized = step(&mut scheduler, replacement);
        assert_eq!(initialized.kind, GraphKind::Init);
        assert_eq!(initialized.lanes[0].request, replacement);
        scheduler.conditioned_inputs_ready(replacement, true).unwrap();
        assert!(scheduler.plan().is_none());
        scheduler.cancel(replacement).unwrap();
        assert!(scheduler.request(replacement).is_err());
        assert_eq!(scheduler.image_waiters, VecDeque::from([rightful]),
            "cancelled generations must not accumulate in the arena queue");
        previous = replacement;
    }

    advance_to_window_snapshot(&mut scheduler, owner);
    assert_eq!(scheduler.request(owner).unwrap().image_slot, None);
    assert_eq!(scheduler.request(rightful).unwrap().image_slot, Some(0));
    assert!(scheduler.image_waiters.is_empty());
    let mut vision = step(&mut scheduler, rightful);
    if vision.kind == GraphKind::Snapshot {
        assert_eq!(vision.lanes[0].request, owner);
        assert_eq!(scheduler.request(rightful).unwrap().image_slot, Some(0));
        vision = step(&mut scheduler, rightful);
    }
    assert_eq!(vision.kind, GraphKind::Vision);
    assert_eq!(vision.lanes[0].request, rightful);
}

#[test]
fn owner_cancellation_fences_vision_prefill_and_snapshot_resources_after_arena_handoff() {
    for (phase, kind) in [
        (RequestPhase::Vision, GraphKind::Vision),
        (RequestPhase::Prefill, GraphKind::Prefill),
        (RequestPhase::Snapshot, GraphKind::Snapshot),
    ] {
        let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
        let free_pages = scheduler.free_page_count();
        let free_snapshots = scheduler.free_snapshot_count();
        let owner = admit(&mut scheduler, &inputs(5));
        let waiter = admit(&mut scheduler, &inputs(1));
        assert_eq!(step(&mut scheduler, owner).kind, GraphKind::Init);
        for _ in 0..100 {
            if scheduler.request(owner).unwrap().phase == phase {
                break;
            }
            step(&mut scheduler, owner);
        }
        assert_eq!(scheduler.request(owner).unwrap().phase, phase);
        if kind == GraphKind::Snapshot {
            assert!(matches!(scheduler.request(owner).unwrap().boundary_capture.unwrap().kind,
                BoundaryKind::ConditionedWindow));
        }
        let (pending, fence, records) = commit_step(&mut scheduler, owner);
        assert_eq!(pending.kind, kind);
        assert_eq!(pending.lanes[0].request, owner);
        assert!(scheduler.completed() < fence);
        let request = scheduler.request(owner).unwrap();
        let state = request.state_slot;
        let pages: Vec<_> = request.pages().chain(request.reserved_pages.iter().copied()).collect();
        let snapshot = request.window_snapshot.unwrap();
        let retained_pages = scheduler.free_page_count();
        let retained_snapshots = scheduler.free_snapshot_count();
        let owns_arena = kind != GraphKind::Snapshot;
        assert_eq!(request.image_slot, owns_arena.then_some(0));
        scheduler.conditioned_inputs_ready(waiter, true).unwrap();
        scheduler.cancel(owner).unwrap();
        assert!(scheduler.request(owner).is_err());
        assert_eq!(scheduler.stats().retiring_slots, 1);
        assert_eq!(scheduler.stats().free_slots, 0);
        assert!(scheduler.slots[owner.slot as usize].retiring);
        assert_ne!(scheduler.request(waiter).unwrap().state_slot, state);
        assert_eq!(scheduler.free_page_count(), retained_pages);
        assert_eq!(scheduler.free_snapshot_count(), retained_snapshots);
        let retired = &scheduler.retired[0];
        assert_eq!(retired.slot, owner.slot as usize);
        assert_eq!(retired.through, fence);
        assert_eq!(retired.pages, pages);
        assert!(retired.snapshots.contains(&Some(snapshot)));
        assert_eq!(retired.image_slot, owns_arena.then_some(0));
        assert!(scheduler.free_image_slots.is_empty());
        if owns_arena {
            assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
            assert_eq!(scheduler.image_waiters, VecDeque::from([waiter]));
        } else {
            assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0),
                "a snapshot has no raw-feature consumer, so cancellation cannot reclaim the waiter's lease");
            assert!(scheduler.image_waiters.is_empty());
        }
        assert!(scheduler.plan().is_none(), "{kind:?} still fences the cancelled request's state");

        assert!(scheduler.complete(fence, &records, Default::default()).unwrap().is_empty());
        assert_eq!(scheduler.completed(), fence);
        assert_eq!(scheduler.stats().retiring_slots, 0);
        assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
        assert!(scheduler.free_image_slots.is_empty());
        assert!(scheduler.image_waiters.is_empty());
        let vision = step(&mut scheduler, waiter);
        assert_eq!(vision.kind, GraphKind::Vision);
        assert_eq!(vision.lanes[0].request, waiter);
        scheduler.cancel(waiter).unwrap();
        assert_eq!(scheduler.free_image_slots.len(), 1);
        assert_eq!(scheduler.free_page_count(), free_pages);
        assert_eq!(scheduler.free_snapshot_count(), free_snapshots);
        assert_eq!(scheduler.stats().snapshot_publications, 0);
    }
}

#[test]
fn a_message_snapshot_keeps_features_needed_later_in_the_same_window() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let images = inputs_with_size(2, 1024);
    let message_end = images[0].token_span.end;
    let mut spec = specification(&images);
    spec.prompt.extend(std::iter::repeat_n(7, 8192));
    spec.message_boundaries = vec![message_end, spec.prompt.len()];
    let owner = scheduler.admit(spec).unwrap();
    scheduler.set_conditioned_inputs(owner, &images).unwrap();
    let waiter = admit(&mut scheduler, &inputs(1));
    advance_to_window_snapshot(&mut scheduler, owner);
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.consumed_tokens, message_end as u64);
    assert_eq!(request.conditioned_windows.len(), 1);
    assert!(matches!(request.boundary_capture.unwrap().kind, BoundaryKind::Message));
    assert!(request.vision_patches.is_empty());
    assert_eq!(request.feature_index(images[1].feature_range.start as i32).unwrap(), 1024);
    let (snapshot, fence, records) = commit_step(&mut scheduler, owner);
    assert_eq!(snapshot.kind, GraphKind::Snapshot);
    assert_eq!(scheduler.request(owner).unwrap().image_slot, Some(0));
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
    scheduler.complete(fence, &records, Default::default()).unwrap();
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.phase, RequestPhase::Prefill);
    assert!(!request.conditioned_window_ready());
    assert_eq!(request.image_slot, Some(0),
        "a public message checkpoint does not end the feature window");
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
    assert_eq!(scheduler.image_waiters, VecDeque::from([waiter]));
    assert_eq!(scheduler.stats().message_checkpoints, 1);

    let (consumer, fence, records) = commit_step(&mut scheduler, owner);
    assert_eq!(consumer.kind, GraphKind::Prefill);
    assert_eq!(consumer.lanes[0].request, owner);
    let consumer_end = consumer.lanes[0].start_position + consumer.lanes[0].tokens as u64;
    assert!(consumer_end >= images[1].token_span.end as u64);
    assert!(consumer_end < scheduler.request(owner).unwrap().prompt_tokens as u64);
    assert_eq!(scheduler.request(owner).unwrap().image_slot, Some(0));
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
    assert!(scheduler.complete(fence, &records, Default::default()).unwrap().is_empty());
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.consumed_tokens, consumer_end);
    assert!(request.consumed_tokens < request.prompt_tokens as u64);
    assert_eq!(request.phase, RequestPhase::Prefill);
    assert_eq!(request.image_slot, None);
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
}

#[test]
fn positive_output_scored_prefill_hands_off_windows_restores_and_releases_the_last_consumer() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    let capacities: Vec<_> = prefill_capacities(scheduler.limits.prefill_chunk_size).collect();
    scheduler.scored_prefill_graphs(&capacities).unwrap();
    let mut graphs = tests::duration_graphs(2);
    let scored: Vec<_> = graphs.iter().filter(|graph| graph.kind == GraphKind::Prefill)
        .cloned().map(|mut graph| {
            graph.kind = GraphKind::ScoredPrefill;
            graph.name = format!("scored-{}", graph.name);
            graph
        }).collect();
    graphs.extend(scored);
    scheduler.measured_graph_durations(&graphs).unwrap();
    let images = inputs(2);
    let mut spec = specification(&images);
    spec.prompt.extend(std::iter::repeat_n(7, 8192));
    spec.message_boundaries = vec![spec.prompt.len()];
    spec.prompt_logprobs = true;
    spec.max_new_tokens = 3;
    let owner = scheduler.admit(spec).unwrap();
    scheduler.set_conditioned_inputs(owner, &images).unwrap();
    let waiter = admit(&mut scheduler, &inputs(1));
    let mut scored_steps = 0;
    for _ in 0..100 {
        if scheduler.request(owner).unwrap().phase == RequestPhase::Snapshot {
            break;
        }
        let executed = step(&mut scheduler, owner);
        if executed.kind.is_prefill() {
            assert_eq!(executed.kind, GraphKind::ScoredPrefill);
            assert_eq!(executed.lanes[0].request, owner);
            scored_steps += 1;
        }
        assert_eq!(scheduler.request(owner).unwrap().generated_tokens, 0);
    }
    assert!(scored_steps > 0);
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.consumed_tokens, images[1].token_span.start as u64);
    assert!(matches!(request.boundary_capture.unwrap().kind, BoundaryKind::ConditionedWindow));
    let state = request.state_slot;
    let pages: Vec<_> = request.pages().collect();
    let snapshot = request.window_snapshot.unwrap();
    let (checkpoint, fence, records) = commit_step(&mut scheduler, owner);
    assert_eq!(checkpoint.kind, GraphKind::Snapshot);
    assert_eq!(scheduler.request(owner).unwrap().image_slot, None);
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
    assert!(scheduler.resume_conditioned_window(owner).is_err());
    scheduler.complete(fence, &records, Default::default()).unwrap();
    assert_eq!(scheduler.request(owner).unwrap().image_slot, None);
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
    assert_eq!(scheduler.stats().snapshot_publications, 0);

    let restored = scheduler.resume_conditioned_window(owner).unwrap();
    assert_eq!(restored.slot, owner.slot);
    assert_eq!(restored.generation, owner.generation + 1);
    assert!(scheduler.request(owner).is_err());
    assert!(scheduler.cancel(owner).is_err());
    assert!(scheduler.conditioned_inputs_ready(owner, true).is_err());
    let request = scheduler.request(restored).unwrap();
    assert_eq!(request.state_slot, state);
    assert_eq!(request.pages().collect::<Vec<_>>(), pages);
    assert_eq!(request.prefix_snapshot(), Some(snapshot));
    assert_eq!(request.image_slot, None);
    assert_eq!(request.feature_index(images[1].feature_range.start as i32).unwrap(), 0);
    assert!(request.feature_index(0).is_err());
    let (restore, fence, mut records) = commit_step(&mut scheduler, restored);
    assert_eq!(restore.kind, GraphKind::Restore);
    assert_eq!(restore.lanes[0].request, restored);
    records[restored.slot as usize].generation = owner.generation as i64;
    assert!(scheduler.complete(fence, &records, Default::default()).is_err());
    assert_eq!(scheduler.request(restored).unwrap().phase, RequestPhase::Restore);
    records[restored.slot as usize].generation = restored.generation as i64;
    scheduler.complete(fence, &records, Default::default()).unwrap();
    assert_eq!(scheduler.request(restored).unwrap().phase, RequestPhase::Vision);
    assert_eq!(scheduler.request(restored).unwrap().window_snapshot, None);
    scheduler.conditioned_inputs_ready(restored, true).unwrap();
    assert!(scheduler.plan().is_none(), "restoration cannot bypass the waiter's feature lease");

    for _ in 0..10 {
        if scheduler.request(waiter).unwrap().image_slot.is_none() {
            break;
        }
        step(&mut scheduler, waiter);
    }
    assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
    assert_eq!(scheduler.request(restored).unwrap().image_slot, Some(0));
    scheduler.cancel(waiter).unwrap();
    let next_waiter = admit(&mut scheduler, &inputs(1));
    assert_eq!(scheduler.request(next_waiter).unwrap().image_slot, None);
    for _ in 0..100 {
        if scheduler.request(restored).unwrap().consumed_tokens >= images[1].token_span.end as u64 {
            break;
        }
        let (executed, fence, records) = commit_step(&mut scheduler, restored);
        if executed.kind.is_prefill() {
            assert_eq!(executed.kind, GraphKind::ScoredPrefill);
            assert_eq!(executed.lanes[0].request, restored);
        }
        assert_eq!(scheduler.request(restored).unwrap().image_slot, Some(0));
        assert_eq!(scheduler.request(next_waiter).unwrap().image_slot, None,
            "the scored image consumer still owns the arena until completion");
        assert!(scheduler.complete(fence, &records, Default::default()).unwrap().is_empty());
    }
    let request = scheduler.request(restored).unwrap();
    assert!(request.consumed_tokens >= images[1].token_span.end as u64);
    assert!(request.consumed_tokens < request.prompt_tokens as u64);
    assert_eq!(request.generated_tokens, 0);
    assert_eq!(request.phase, RequestPhase::Prefill);
    assert_eq!(request.image_slot, None);
    assert_eq!(scheduler.request(next_waiter).unwrap().image_slot, Some(0));
    let mut tail_scored_steps = 0;
    let mut decode_steps = 0;
    for _ in 0..100 {
        if scheduler.request(restored).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        let executed = step(&mut scheduler, restored);
        if executed.kind.is_prefill() {
            assert_eq!(executed.kind, GraphKind::ScoredPrefill);
            tail_scored_steps += 1;
        }
        decode_steps += usize::from(executed.kind == GraphKind::Decode);
        assert_eq!(scheduler.request(restored).unwrap().image_slot, None);
        assert_eq!(scheduler.request(next_waiter).unwrap().image_slot, Some(0));
    }
    let completed = scheduler.request(restored).unwrap();
    assert_eq!(completed.phase, RequestPhase::Finishing);
    assert_eq!(completed.consumed_tokens, completed.prompt_tokens as u64 + 2);
    assert_eq!(completed.generated_tokens, 3);
    assert!(tail_scored_steps > 0);
    assert_eq!(decode_steps, 2);
    assert!(scheduler.finish(restored).unwrap().is_some());
}

#[test]
fn cold_history_exceeding_two_arenas_keeps_pages_state_and_user_reuse_across_windows() {
    for (page_tokens, rows) in [(64, 1), (128, 4), (1024, 8)] {
        let mut scheduler = scheduler(page_tokens, rows);
        let images = inputs(12);
        let mut handle = admit(&mut scheduler, &images);
        assert_eq!(scheduler.request(handle).unwrap().conditioned_windows.len(), 12);
        let state = scheduler.request(handle).unwrap().state_slot;
        let image_slot = scheduler.request(handle).unwrap().image_slot;
        let initial_lookups = scheduler.stats().cache_lookups;
        let mut renewals = 0;
        let mut restores = 0;
        let mut vision_jobs = 0;
        let mut sampled = 0;
        for _ in 0..200 {
            let request = scheduler.request(handle).unwrap();
            if request.phase == RequestPhase::Finishing {
                break;
            }
            if request.conditioned_window_ready() {
                let old_handle = handle;
                let consumed = request.consumed_tokens;
                assert_ne!(
                    consumed % page_tokens as u64,
                    0,
                    "checkpoint must exercise a partial page"
                );
                let pages: Vec<_> = request.pages().collect();
                let reserved = request.reserved_pages.clone();
                let key = request.history.key_at(consumed).unwrap();
                let snapshot = request.window_snapshot.unwrap();
                let free = scheduler.free_snapshot_count();
                assert!(
                    scheduler.plan().is_none(),
                    "private checkpoint waits for explicit epoch renewal"
                );
                handle = scheduler.resume_conditioned_window(handle).unwrap();
                renewals += 1;
                assert_eq!(handle.slot, old_handle.slot);
                assert_eq!(handle.generation, old_handle.generation + 1);
                assert!(scheduler.request(old_handle).is_err());
                assert!(
                    scheduler
                        .conditioned_inputs_ready(old_handle, true)
                        .is_err()
                );
                let request = scheduler.request(handle).unwrap();
                assert_eq!(request.state_slot, state);
                assert_eq!(request.image_slot, image_slot);
                assert_eq!(request.pages().collect::<Vec<_>>(), pages);
                assert_eq!(request.reserved_pages, reserved);
                assert_eq!(request.history.key_at(consumed).unwrap(), key);
                assert_eq!(request.prefix_snapshot(), Some(snapshot));
                assert_eq!(request.generated_tokens, 0);
                assert_eq!(scheduler.free_snapshot_count(), free);
                assert_eq!(
                    request.feature_index((renewals * 4096) as i32).unwrap(),
                    0
                );
                assert!(
                    request
                        .feature_index(((renewals - 1) * 4096) as i32)
                        .is_err()
                );
            }
            let before = scheduler.request(handle).unwrap().generated_tokens;
            let executed = step(&mut scheduler, handle);
            vision_jobs += usize::from(executed.kind == GraphKind::Vision);
            restores += usize::from(executed.kind == GraphKind::Restore);
            sampled += (scheduler.request(handle).unwrap().generated_tokens - before) as usize;
            assert_eq!(scheduler.stats().cache_lookups, initial_lookups);
            assert_eq!(scheduler.stats().reused_tokens, 0);
            assert_eq!(scheduler.stats().snapshot_publications, 0);
        }
        assert_eq!(renewals, 11);
        assert_eq!(restores, 11);
        assert_eq!(vision_jobs, images.len());
        assert_eq!(
            sampled, 1,
            "window endpoints cannot produce an output token"
        );
        assert_eq!(
            scheduler.request(handle).unwrap().phase,
            RequestPhase::Finishing
        );
        assert_eq!(scheduler.request(handle).unwrap().window_snapshot, None);
        assert!(scheduler.finish(handle).unwrap().is_some());
        assert_eq!(scheduler.stats().snapshot_publications, 1);
    }
}

#[test]
fn private_snapshot_cancellation_retires_its_row_and_pages_but_not_the_arena_after_the_fence() {
    let mut scheduler = scheduler(64, 4);
    let initial_snapshots = scheduler.free_snapshot_count();
    let initial_pages = scheduler.free_page_count();
    let handle = admit(&mut scheduler, &inputs(5));
    advance_to_window_snapshot(&mut scheduler, handle);
    let (_, fence, records) = commit_step(&mut scheduler, handle);
    let retained = scheduler.free_snapshot_count();
    let retained_pages = scheduler.free_page_count();
    let request = scheduler.request(handle).unwrap();
    let snapshot = request.window_snapshot.unwrap();
    let pages: Vec<_> = request.pages().chain(request.reserved_pages.iter().copied()).collect();
    assert_eq!(request.image_slot, None);
    assert_eq!(scheduler.free_image_slots.len(), 1);
    assert!(retained < initial_snapshots);
    assert!(retained_pages < initial_pages);
    assert!(scheduler.resume_conditioned_window(handle).is_err());
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_snapshot_count(), retained);
    assert_eq!(scheduler.free_page_count(), retained_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
    assert_eq!(scheduler.stats().retiring_slots, 1);
    assert_eq!(scheduler.stats().free_slots, 0);
    let retired = &scheduler.retired[0];
    assert_eq!(retired.slot, handle.slot as usize);
    assert_eq!(retired.through, fence);
    assert_eq!(retired.pages, pages);
    assert!(retired.snapshots.contains(&Some(snapshot)));
    assert_eq!(retired.image_slot, None);
    scheduler
        .complete(fence, &records, Default::default())
        .unwrap();
    assert_eq!(scheduler.free_snapshot_count(), initial_snapshots);
    assert_eq!(scheduler.free_page_count(), initial_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
    assert_eq!(scheduler.stats().retiring_slots, 0);
    assert_eq!(scheduler.stats().snapshot_publications, 0);
}

#[test]
fn restore_rejects_a_completion_from_the_previous_execution_epoch() {
    let mut scheduler = scheduler(64, 1);
    let handle = admit(&mut scheduler, &inputs(5));
    advance_to_window_snapshot(&mut scheduler, handle);
    step(&mut scheduler, handle);
    let next = scheduler.resume_conditioned_window(handle).unwrap();
    let (executed, fence, mut records) = commit_step(&mut scheduler, next);
    assert_eq!(executed.kind, GraphKind::Restore);
    records[next.slot as usize].generation = handle.generation as i64;
    assert!(
        scheduler
            .complete(fence, &records, Default::default())
            .is_err()
    );
    assert_eq!(
        scheduler.request(next).unwrap().phase,
        RequestPhase::Restore
    );
    records[next.slot as usize].generation = next.generation as i64;
    scheduler
        .complete(fence, &records, Default::default())
        .unwrap();
    assert_eq!(scheduler.request(next).unwrap().phase, RequestPhase::Vision);
}

#[test]
fn cancellation_during_private_restore_keeps_the_snapshot_until_its_fence() {
    let mut scheduler = scheduler(128, 4);
    let free_snapshots = scheduler.free_snapshot_count();
    let free_pages = scheduler.free_page_count();
    let handle = admit(&mut scheduler, &inputs(5));
    advance_to_window_snapshot(&mut scheduler, handle);
    step(&mut scheduler, handle);
    let next = scheduler.resume_conditioned_window(handle).unwrap();
    let (_, fence, records) = commit_step(&mut scheduler, next);
    let retained = scheduler.free_snapshot_count();
    assert!(scheduler.cancel(handle).is_err());
    assert!(scheduler.request(next).is_ok());
    scheduler.cancel(next).unwrap();
    assert_eq!(scheduler.free_snapshot_count(), retained);
    assert_eq!(scheduler.free_image_slots.len(), 0);
    scheduler
        .complete(fence, &records, Default::default())
        .unwrap();
    assert_eq!(scheduler.free_snapshot_count(), free_snapshots);
    assert_eq!(scheduler.free_page_count(), free_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
    assert_eq!(scheduler.stats().snapshot_publications, 0);
}

#[test]
fn window_boundaries_keep_bidirectional_image_spans_inside_one_prefill() {
    let mut scheduler = scheduler(64, 1);
    let mut images = inputs(9);
    for (index, input) in images.iter_mut().enumerate() {
        input.token_span.start += 5 * index;
        input.token_span.end += 5 * index;
    }
    let mut handle = admit(&mut scheduler, &images);
    let spans: Vec<_> = images
        .iter()
        .map(|input| input.token_span.clone())
        .collect();
    scheduler.set_prefill_spans(handle, &spans).unwrap();
    let mut renewals = 0;
    for _ in 0..200 {
        if scheduler.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        if scheduler
            .request(handle)
            .unwrap()
            .conditioned_window_ready()
        {
            let consumed = scheduler.request(handle).unwrap().consumed_tokens as usize;
            assert!(
                spans
                    .iter()
                    .all(|span| !(span.start < consumed && consumed < span.end))
            );
            handle = scheduler.resume_conditioned_window(handle).unwrap();
            renewals += 1;
        }
        let executed = step(&mut scheduler, handle);
        if executed.kind == GraphKind::Prefill {
            let lane = &executed.lanes[0];
            let start = lane.start_position as usize;
            let end = start + lane.tokens;
            for span in spans
                .iter()
                .filter(|span| span.start < end && span.end > start)
            {
                assert!(
                    start <= span.start && span.end <= end,
                    "prefill {start}..{end} splits image {span:?}"
                );
            }
        }
    }
    assert_eq!(renewals, 8);
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Finishing
    );
}

#[test]
fn an_atomic_span_crossing_the_soft_horizon_keeps_both_images_in_the_first_window() {
    let mut scheduler = scheduler(128, 4);
    let images = inputs_with_size(12, 512);
    let atomic = images[7].token_span.start..images[8].token_span.end;
    assert_eq!(atomic.len(), 1024);
    let mut handle = scheduler.admit(specification(&images)).unwrap();
    scheduler.set_prefill_spans(handle, &[atomic.clone()]).unwrap();
    scheduler.set_conditioned_inputs(handle, &images).unwrap();
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.conditioned_windows.len(), 2);
    assert_eq!(request.conditioned_input_end(), 9);
    let first = request.conditioned_windows.front().unwrap();
    assert_eq!(first.inputs, 0..9);
    assert_eq!(first.token_end, images[8].token_span.end);
    assert_eq!(first.prefill_end, images[9].token_span.start);
    assert_eq!(first.features.iter().map(|range| range.len()).sum::<usize>(), 4608);
    assert!(first.token_end - first.token_start > scheduler.limits.prefill_chunk_size);
    assert_eq!(scheduler.capabilities.vision.features_per_request, 16_384);
    let mut vision_jobs = 0;
    let mut atomic_consumers = 0;
    let mut renewals = 0;
    let mut restores = 0;
    for _ in 0..100 {
        let request = scheduler.request(handle).unwrap();
        if request.phase == RequestPhase::Finishing {
            break;
        }
        if request.conditioned_window_ready() {
            assert_eq!(request.consumed_tokens, images[9].token_span.start as u64);
            handle = scheduler.resume_conditioned_window(handle).unwrap();
            renewals += 1;
        }
        let executed = step(&mut scheduler, handle);
        vision_jobs += usize::from(executed.kind == GraphKind::Vision);
        restores += usize::from(executed.kind == GraphKind::Restore);
        if executed.kind.is_prefill() {
            let lane = &executed.lanes[0];
            let start = lane.start_position as usize;
            let end = start + lane.tokens;
            assert!(lane.tokens <= 4096);
            if start < atomic.end && atomic.start < end {
                assert!(start <= atomic.start && atomic.end <= end,
                    "prefill {start}..{end} split atomic span {atomic:?}");
                atomic_consumers += 1;
            }
        }
    }
    assert_eq!(vision_jobs, 12);
    assert_eq!(atomic_consumers, 1);
    assert_eq!(renewals, 1);
    assert_eq!(restores, 1);
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.phase, RequestPhase::Finishing);
    assert_eq!(request.generated_tokens, 1);
    assert_eq!(request.consumed_tokens, request.prompt_tokens as u64);
    assert_eq!(scheduler.stats().reused_tokens, 0);
    assert_eq!(scheduler.stats().snapshot_publications, 0);
}

#[test]
fn an_image_larger_than_the_prefill_horizon_is_admitted_whole_without_resizing_the_arena() {
    let mut scheduler = scheduler_with_arenas(128, 4, 2, 16, 1);
    scheduler.capabilities.vision.patch_capacities.push(32_768);
    scheduler.capabilities.vision.validate().unwrap();
    let mut graphs = tests::duration_graphs(2);
    let mut large_vision = graphs.iter().find(|graph|
        graph.kind == GraphKind::Vision && graph.token_capacity == 16_384).unwrap().clone();
    large_vision.name = "large-image-vision".into();
    large_vision.token_capacity = 32_768;
    graphs.push(large_vision);
    scheduler.measured_graph_durations(&graphs).unwrap();
    let images = inputs_with_size(1, 8192);
    let mut spec = specification(&images);
    spec.prompt.extend(std::iter::repeat_n(7, 8192));
    spec.message_boundaries = vec![spec.prompt.len()];
    let owner = scheduler.admit(spec).unwrap();
    scheduler.set_conditioned_inputs(owner, &images).unwrap();
    let waiter = admit(&mut scheduler, &inputs(1));
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.conditioned_windows.len(), 1);
    assert_eq!(request.conditioned_input_end(), 1);
    assert_eq!(request.conditioned_windows.front().unwrap().features, vec![0..8192]);
    assert_eq!(request.conditioned_windows.front().unwrap().token_end, 8209);
    assert_eq!(request.vision_patches, VecDeque::from([32_768]));
    assert_eq!(request.window_snapshot, None);
    assert_eq!(request.feature_index(8191).unwrap(), 8191);
    let mut vision_jobs = 0;
    let mut image_prefills = 0;
    let mut released = false;
    for _ in 0..100 {
        if scheduler.request(owner).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        let (executed, fence, records) = commit_step(&mut scheduler, owner);
        assert_ne!(executed.kind, GraphKind::Restore);
        if executed.kind == GraphKind::Vision {
            assert_eq!(executed.token_capacity, 32_768);
            assert_eq!(executed.lanes[0].request, owner);
            vision_jobs += 1;
        }
        let request = scheduler.request(owner).unwrap();
        if request.consumed_tokens < images[0].token_span.end as u64 {
            assert_eq!(request.image_slot, Some(0));
            assert_eq!(scheduler.request(waiter).unwrap().image_slot, None);
            if executed.kind.is_prefill() {
                assert!(executed.lanes[0].tokens <= 4096);
                image_prefills += 1;
            }
        }
        let generated = scheduler.complete(fence, &records, Default::default()).unwrap();
        for token in generated {
            scheduler.published_output(token.request, fence).unwrap();
        }
        let request = scheduler.request(owner).unwrap();
        if request.consumed_tokens >= images[0].token_span.end as u64 {
            if !released {
                assert!(request.consumed_tokens < request.prompt_tokens as u64);
                released = true;
            }
            assert_eq!(request.image_slot, None);
            assert_eq!(scheduler.request(waiter).unwrap().image_slot, Some(0));
        }
        assert_eq!(request.conditioned_windows.len(), 1);
        assert_eq!(scheduler.capabilities.vision.features_per_request, 16_384);
        assert_eq!(scheduler.capabilities.vision.feature_slots, 1);
    }
    assert_eq!(vision_jobs, 1);
    assert!(image_prefills > 1);
    assert!(released);
    let request = scheduler.request(owner).unwrap();
    assert_eq!(request.phase, RequestPhase::Finishing);
    assert_eq!(request.consumed_tokens, request.prompt_tokens as u64);
    assert_eq!(request.generated_tokens, 1);
}

#[test]
fn partial_image_prefix_reuse_preserves_canonical_indices_and_only_counts_initial_reuse() {
    let mut scheduler = scheduler(64, 4);
    let images = inputs(12);
    let endpoint = images[4].token_span.start + 512;
    let mut specification = specification(&images);
    specification.message_boundaries.push(endpoint);
    let mut handle = scheduler.admit(specification.clone()).unwrap();
    scheduler.set_conditioned_inputs(handle, &images).unwrap();
    for _ in 0..150 {
        if scheduler.request(handle).unwrap().prefix_tokens == endpoint as u64 {
            break;
        }
        if scheduler
            .request(handle)
            .unwrap()
            .conditioned_window_ready()
        {
            handle = scheduler.resume_conditioned_window(handle).unwrap();
        }
        step(&mut scheduler, handle);
    }
    assert_eq!(
        scheduler.request(handle).unwrap().prefix_tokens,
        endpoint as u64
    );
    scheduler.cancel(handle).unwrap();
    handle = scheduler.admit(specification).unwrap();
    assert_eq!(
        scheduler.request(handle).unwrap().consumed_tokens,
        endpoint as u64
    );
    scheduler
        .set_conditioned_inputs(handle, &images[4..])
        .unwrap();
    assert_eq!(
        scheduler
            .request(handle)
            .unwrap()
            .feature_index(16_384 + 512)
            .unwrap(),
        512
    );
    assert!(scheduler.request(handle).unwrap().feature_index(0).is_err());
    let expected_reuse = scheduler.stats().reused_tokens;
    assert_eq!(expected_reuse, endpoint as u64);
    let mut renewals = 0;
    for _ in 0..200 {
        if scheduler.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        if scheduler
            .request(handle)
            .unwrap()
            .conditioned_window_ready()
        {
            handle = scheduler.resume_conditioned_window(handle).unwrap();
            renewals += 1;
        }
        step(&mut scheduler, handle);
        assert_eq!(scheduler.stats().reused_tokens, expected_reuse);
        assert_eq!(
            scheduler.request(handle).unwrap().reused_prompt_tokens(),
            endpoint as u64
        );
    }
    assert_eq!(renewals, 7);
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Finishing
    );
}

#[test]
fn vision_windows_make_progress_while_another_request_keeps_decoding() {
    let mut scheduler = scheduler_with_active(64, 1, 2);
    let mut durations = tests::duration_graphs(2);
    for graph in &mut durations {
        graph.estimated_duration_ns = 1_000_000;
    }
    scheduler.measured_graph_durations(&durations).unwrap();
    let decoder = scheduler.admit(tests::spec(32, 512)).unwrap();
    for _ in 0..10 {
        if scheduler.request(decoder).unwrap().phase == RequestPhase::Decode {
            break;
        }
        step(&mut scheduler, decoder);
    }
    assert_eq!(
        scheduler.request(decoder).unwrap().phase,
        RequestPhase::Decode
    );
    let mut visual = admit(&mut scheduler, &inputs(5));
    let mut vision_jobs = 0;
    let mut renewals = 0;
    for _ in 0..511 {
        if scheduler.request(visual).unwrap().generated_tokens == 1 {
            break;
        }
        if scheduler
            .request(visual)
            .unwrap()
            .conditioned_window_ready()
        {
            visual = scheduler.resume_conditioned_window(visual).unwrap();
            renewals += 1;
        }
        let selected = step(&mut scheduler, visual);
        if selected.kind == GraphKind::Vision {
            assert!(
                selected.selection_value_rate() > 0.0,
                "vision must retain the value of its following prefill"
            );
            vision_jobs += 1;
        }
        if selected.kind == GraphKind::Snapshot
            && scheduler.request(visual).unwrap().consumed_tokens
                < scheduler.request(visual).unwrap().prompt_tokens as u64
        {
            assert!(selected.selection_value_rate() > 0.0);
        }
        assert_eq!(
            scheduler.request(decoder).unwrap().phase,
            RequestPhase::Decode,
            "visual phase {:?}, consumed {}, vision jobs {}, renewals {}, elapsed {}ns",
            scheduler.request(visual).unwrap().phase,
            scheduler.request(visual).unwrap().consumed_tokens,
            vision_jobs,
            renewals,
            scheduler.now_ns
        );
    }
    assert_eq!(
        vision_jobs,
        5,
        "visual phase {:?}, consumed {}, decoder output {}, elapsed {}ns",
        scheduler.request(visual).unwrap().phase,
        scheduler.request(visual).unwrap().consumed_tokens,
        scheduler.request(decoder).unwrap().generated_tokens,
        scheduler.now_ns
    );
    assert_eq!(renewals, 4);
    let completed = scheduler.request(visual).unwrap();
    assert_eq!(completed.consumed_tokens, completed.prompt_tokens as u64);
    assert_eq!(completed.generated_tokens, 1);
    assert!(matches!(
        completed.phase,
        RequestPhase::Snapshot | RequestPhase::Finishing
    ));
}

#[test]
fn unavailable_cold_window_row_is_rejected_without_leaking_page_reservations() {
    let mut scheduler = scheduler_with_resources(64, 1, 1, 0);
    let available_pages = scheduler.free_page_count();
    let images = inputs(5);
    let handle = scheduler.admit(specification(&images)).unwrap();
    let reserved = scheduler.request(handle).unwrap().reserved_pages.len();
    let error = scheduler
        .set_conditioned_inputs(handle, &images)
        .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Admission(crate::AdmissionError::Context(_))
    ));
    assert_eq!(
        scheduler.request(handle).unwrap().reserved_pages.len(),
        reserved
    );
    assert_eq!(scheduler.request(handle).unwrap().last_use(), Fence(0));
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_page_count(), available_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
}

#[test]
fn an_external_checkpoint_pin_keeps_window_storage_failure_retryable() {
    let mut scheduler = scheduler_with_resources(64, 1, 1, 1);
    let text = scheduler.admit(tests::spec(32, 1)).unwrap();
    for _ in 0..10 {
        if scheduler.request(text).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        step(&mut scheduler, text);
    }
    let checkpoint = scheduler.finish(text).unwrap().unwrap();
    scheduler.cache.pin(checkpoint).unwrap();
    let images = inputs(5);
    let handle = scheduler.admit(specification(&images)).unwrap();
    let error = scheduler
        .set_conditioned_inputs(handle, &images)
        .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Admission(crate::AdmissionError::Capacity(_))
    ));
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.cache.checkpoint(checkpoint).is_ok());
    scheduler.cache.unpin(checkpoint).unwrap();
    let handle = scheduler.admit(specification(&images)).unwrap();
    scheduler.set_conditioned_inputs(handle, &images).unwrap();
    assert!(scheduler.request(handle).unwrap().window_snapshot.is_some());
    assert!(scheduler.cache.checkpoint(checkpoint).is_err());
    scheduler.cancel(handle).unwrap();
}
