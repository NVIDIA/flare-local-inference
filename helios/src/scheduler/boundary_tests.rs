use super::*;

fn scheduler(snapshots: u32, mtp: bool) -> RequestScheduler {
    scheduler_with_pages(snapshots, mtp, 64)
}

fn scheduler_with_pages(snapshots: u32, mtp: bool, page_tokens: usize) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [9; 32],
        SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 8192,
            prefill_chunk_size: 512,
            pages: 256,
            snapshots,
        },
        SchedulerCapabilities {
            vocabulary: 248320,
            page_tokens,
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
    s.measured_graph_durations(&super::tests::duration_graphs(4))
        .unwrap();
    if mtp {
        s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    }
    s
}
fn spec(system: Option<usize>, prompt: usize, output: u32) -> RequestSpec {
    RequestSpec {
        message_boundaries: system.into_iter().collect(),
        prompt: (0..prompt as u32).collect(),
        conditioning: vec![],
        max_new_tokens: output,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: false,
    }
}
fn run_step(s: &mut RequestScheduler) -> ScheduledPlan {
    let step = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let r = s.request(lane.request).unwrap();
        let end = lane.start_position + lane.tokens as u64;
        let emitted = match step.kind {
            GraphKind::Prefill => usize::from(end == r.prompt_tokens as u64),
            GraphKind::Decode | GraphKind::Resume => 1,
            GraphKind::Speculative => step
                .token_capacity
                .min((r.max_new_tokens - r.generated_tokens) as usize),
            _ => 0,
        };
        let consumed = if step.kind == GraphKind::Speculative {
            r.consumed_tokens + emitted as u64
        } else {
            end
        };
        let generated = r.generated_tokens + emitted as u32;
        let token = if emitted > 0 { 900 } else { -1 };
        let mut record = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: token,
            status: if emitted > 0 && generated == r.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
        if step.kind == GraphKind::Speculative {
            let mut vector = crate::execution::TokenVector {
                emitted_count: emitted as i64,
                offered_drafts: (step.token_capacity - 1) as i64,
                accepted_drafts: emitted.min(step.token_capacity - 1) as i64,
                verification_width: step.token_capacity as i64,
                draft_cache_consumed: consumed as i64 - 1,
                ..Default::default()
            };
            vector.tokens[..emitted].fill(token);
            record.vector = Some(vector);
        }
        records[lane.request.slot as usize] = record;
    }
    s.complete(fence, &records, Default::default()).unwrap();
    step
}
fn stop_at_capture(s: &mut RequestScheduler, handle: RequestHandle, endpoint: u64) {
    for _ in 0..200 {
        let r = s.request(handle).unwrap();
        if r.phase == RequestPhase::Snapshot && r.snapshot_consumed() == endpoint {
            return;
        }
        run_step(s);
    }
    panic!("endpoint not scheduled");
}
fn finish(s: &mut RequestScheduler, h: RequestHandle) -> CheckpointId {
    for _ in 0..250 {
        if s.request(h).unwrap().phase == RequestPhase::Finishing {
            return s.finish(h).unwrap().unwrap();
        }
        run_step(s);
    }
    panic!("request failed to finish");
}

#[test]
fn duplicate_finished_boundary_needs_no_terminal_snapshot() {
    let mut s = scheduler(2, false);
    let a = s.admit(spec(Some(70), 70, 1)).unwrap();
    let b = s.admit(spec(Some(70), 70, 1)).unwrap();
    let mut pending = vec![a, b];
    for _ in 0..20 {
        run_step(&mut s);
        let finished: Vec<_> = pending
            .iter()
            .copied()
            .filter(|&h| s.request(h).unwrap().phase == RequestPhase::Finishing)
            .collect();
        for h in finished {
            s.finish(h).unwrap();
            pending.retain(|&other| other != h);
        }
        if pending.is_empty() {
            break;
        }
    }
    assert!(pending.is_empty());
    assert_eq!(s.stats().snapshot_publications, 1);
    assert_eq!(s.free_snapshot_count(), 1);
    assert_eq!(s.free_page_count(), 255);
}

#[test]
fn requests_finish_when_no_checkpoint_storage_is_available() {
    let mut s = scheduler(0, false);
    let handle = s.admit(spec(Some(10), 70, 3)).unwrap();
    for _ in 0..20 {
        if s.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        assert_ne!(run_step(&mut s).kind, GraphKind::Snapshot);
    }
    assert_eq!(s.request(handle).unwrap().generated_tokens, 3);
    assert_eq!(s.finish(handle).unwrap(), None);
    assert_eq!(s.free_page_count(), 256);
}

#[test]
fn exact_system_prompt_terminal_endpoints_and_partial_restore_pages() {
    for page_tokens in [64, 256] {
        for system in [1, 2, 3, 10, 15, 63, 64, 65, 66, 67, 127, 255, 256, 257, 511] {
            for mtp in [false, true] {
                let mut s = scheduler_with_pages(8, mtp, page_tokens);
                let prompt = system + 67;
                let h = s.admit(spec(Some(system), prompt, 3)).unwrap();
                stop_at_capture(&mut s, h, system as u64);
                let prior = s.request(h).unwrap().pages().collect::<Vec<_>>();
                assert_eq!(s.request(h).unwrap().generated_tokens, 0);
                assert_eq!(
                    s.request(h).unwrap().snapshot_destination(),
                    s.request(h).unwrap().message_snapshot.unwrap()
                );
                run_step(&mut s);
                let r = s.request(h).unwrap();
                assert_eq!(r.prefix_tokens, system as u64);
                assert_eq!(r.shared_pages.len(), system / page_tokens);
                assert_eq!(r.pages().collect::<Vec<_>>(), prior);
                assert_eq!(r.reused_prompt_tokens(), 0);
                let continuation = s.plan().unwrap();
                assert_eq!(continuation.kind, GraphKind::Prefill);
                assert_eq!(continuation.lanes[0].start_position, system as u64);
                assert_eq!(continuation.lanes[0].tokens, 67);
                stop_at_capture(&mut s, h, prompt as u64);
                run_step(&mut s);
                let terminal = finish(&mut s, h);
                assert_eq!(
                    s.cache
                        .checkpoint(terminal)
                        .unwrap()
                        .prefix()
                        .consumed_tokens,
                    (prompt + 2) as u64
                );
                assert_eq!(
                    s.cache.page_chain(terminal).unwrap().len(),
                    (prompt + 2) / page_tokens
                );
                assert_eq!(
                    s.cache.lookup(&history(system)).unwrap().consumed_tokens,
                    system as u64
                );
                assert_eq!(
                    s.cache.lookup(&history(prompt)).unwrap().consumed_tokens,
                    prompt as u64
                );
                let mut branch = spec(Some(system), prompt, 3);
                branch.prompt[system] = 777;
                let a = s.admit(branch.clone()).unwrap();
                let b = s.admit(branch).unwrap();
                assert_eq!(s.request(a).unwrap().consumed_tokens, system as u64);
                run_step(&mut s);
                let a = s.request(a).unwrap();
                let b = s.request(b).unwrap();
                assert_eq!(a.phase, RequestPhase::Prefill);
                assert_eq!(a.shared_pages, b.shared_pages);
                if system % page_tokens != 0 || mtp {
                    assert_ne!(a.private_pages[0], b.private_pages[0]);
                }
                assert_eq!(
                    a.pages().count(),
                    system.div_ceil(page_tokens) + usize::from(mtp)
                );
                if mtp && system % page_tokens >= 2 {
                    let target_full = system / page_tokens;
                    let a_pages = a.pages().collect::<Vec<_>>();
                    let b_pages = b.pages().collect::<Vec<_>>();
                    assert_ne!(a_pages[target_full], b_pages[target_full]);
                    assert_ne!(a_pages[target_full + 1], b_pages[target_full + 1]);
                }
            }
        }
    }
}

#[test]
fn prompt_6459_retains_exact_prompt_and_terminal_with_immutable_admission_reuse() {
    let mut s = scheduler(8, false);
    let h = s.admit(spec(Some(10), 6459, 144)).unwrap();
    let terminal = finish(&mut s, h);
    assert_eq!(
        s.cache
            .checkpoint(terminal)
            .unwrap()
            .prefix()
            .consumed_tokens,
        6602
    );
    assert_eq!(
        s.cache.lookup(&history(6459)).unwrap().consumed_tokens,
        6459
    );
    let h = s.admit(spec(Some(10), 6459, 144)).unwrap();
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 6459);
    assert_eq!(run_step(&mut s).kind, GraphKind::Restore);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Resume);
    run_step(&mut s);
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 6459);
}

#[test]
fn boundary_snapshots_are_optional_and_cancel_behind_fence() {
    let mut s = scheduler(2, false);
    let h = s.admit(spec(Some(10), 200, 3)).unwrap();
    assert_eq!(s.free_snapshot_count(), 2);
    assert_eq!(s.stats().active_requests, 1);
    finish(&mut s, h);
    let mut s = scheduler(3, false);
    let h = s.admit(spec(Some(10), 200, 3)).unwrap();
    assert_eq!(s.free_snapshot_count(), 3);
    stop_at_capture(&mut s, h, 10);
    let step = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    s.cancel(h).unwrap();
    assert_eq!(s.free_snapshot_count(), 2);
    let mut record = [CompletionRecord::default(); 4];
    record[h.slot as usize] = CompletionRecord {
        generation: h.generation as i64,
        submission_sequence: fence.0 as i64,
        consumed_tokens: 10,
        token_id: -1,
        status: 1,
        ..Default::default()
    };
    s.complete(fence, &record, Default::default()).unwrap();
    assert_eq!(s.free_snapshot_count(), 3);
    assert_eq!(s.free_page_count(), 256);
    assert_eq!(s.cache.stats().published_checkpoints, 0);
}

#[test]
fn duplicate_boundary_publication_does_not_remap_live_partial_pages() {
    let mut s = scheduler(6, false);
    let a = s.admit(spec(Some(10), 200, 2)).unwrap();
    let b = s.admit(spec(Some(10), 200, 2)).unwrap();
    stop_at_capture(&mut s, a, 10);
    run_step(&mut s);
    stop_at_capture(&mut s, b, 10);
    let pages = s.request(b).unwrap().pages().collect::<Vec<_>>();
    run_step(&mut s);
    assert_eq!(s.request(b).unwrap().pages().collect::<Vec<_>>(), pages);
    assert_eq!(s.request(b).unwrap().prefix_tokens, 0);
    assert!(s.request(a).unwrap().prefix_tokens >= 10);
    assert_eq!(s.cache.lookup(&history(10)).unwrap().consumed_tokens, 10);
    assert!(s.request(b).unwrap().message_snapshot.is_none());
    assert_eq!(s.free_snapshot_count(), 3);
}

#[test]
fn failed_boundary_copy_is_not_published_and_all_reservations_retire() {
    let mut s = scheduler(3, false);
    let h = s.admit(spec(Some(3), 70, 3)).unwrap();
    stop_at_capture(&mut s, h, 3);
    let step = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    records[h.slot as usize] = CompletionRecord {
        generation: h.generation as i64,
        submission_sequence: fence.0 as i64,
        consumed_tokens: 3,
        token_id: -1,
        status: 3,
        error_code: 9,
        ..Default::default()
    };
    s.complete(fence, &records, Default::default()).unwrap();
    assert_eq!(s.finish(h).unwrap(), None);
    assert_eq!(s.free_snapshot_count(), 3);
    assert_eq!(s.free_page_count(), 256);
}

#[test]
fn identical_prompt_and_terminal_endpoint_requires_one_reservation() {
    let mut s = scheduler(1, false);
    let h = s.admit(spec(None, 3, 1)).unwrap();
    let id = finish(&mut s, h);
    assert_eq!(s.cache.checkpoint(id).unwrap().prefix().consumed_tokens, 3);
    assert!(s.cache.page_chain(id).unwrap().is_empty());
}

fn history(n: usize) -> PrefixHistory {
    let mut h = PrefixHistory::new([9; 32]);
    h.append(&(0..n as u32).collect::<Vec<_>>(), &[]).unwrap();
    h
}

#[test]
fn runtime_concurrency_and_loaded_width_assign_fixed_execution_rows() {
    let template = scheduler(256, false);
    for active in [1usize, 4, 32] {
        for width in [1usize, 3, 6] {
            let mut s = RequestScheduler::new(
                [9; 32],
                SchedulerLimits {
                    max_active: active,
                    request_slots: 32,
                    ..template.limits
                },
                template.capabilities.clone(),
            )
            .unwrap();
            s.enable_speculation(&[1, 2, 3, 4, 5, 6]).unwrap();
            let shapes: Vec<_> = crate::execution::BATCH_CAPACITIES
                .into_iter()
                .filter(|batch| *batch <= active.next_power_of_two())
                .flat_map(|batch| (1..=width).map(move |width| (batch, width)))
                .collect();
            s.speculative_graphs(&shapes).unwrap();
            let mut handles = Vec::new();
            let mut owned = BTreeSet::new();
            for _ in 0..active {
                let handle = s.admit(spec(None, 64, 1)).unwrap();
                let request = s.request(handle).unwrap();
                assert_eq!(
                    request.state_slot.0 as usize,
                    1 + handle.slot as usize * width
                );
                for column in 0..width {
                    assert!(owned.insert(request.state_slot.0 + column as u32));
                }
                assert!(request.terminal_snapshot.is_none());
                handles.push(handle);
            }
            assert_eq!(s.free_snapshot_count(), 256);
            for handle in handles {
                s.cancel(handle).unwrap();
            }
            assert_eq!(s.free_snapshot_count(), 256);
        }
    }
}

#[test]
fn atomic_prefill_spans_keep_bidirectional_inputs_together() {
    let mut s = scheduler(24, false);
    let h = s.admit(spec(None, 1537, 1)).unwrap();
    s.set_prefill_spans(h, &[400..780, 810..1190]).unwrap();
    let mut chunks = Vec::new();
    while s.request(h).unwrap().phase != RequestPhase::Finishing {
        let step = run_step(&mut s);
        if step.kind == GraphKind::Prefill {
            let lane = &step.lanes[0];
            chunks.push((lane.start_position, lane.tokens));
        }
    }
    assert_eq!(chunks, [(0, 400), (400, 410), (810, 512), (1322, 215)]);
    s.finish(h).unwrap();
    let reused = s.admit(spec(None, 1537, 1)).unwrap();
    s.set_prefill_spans(reused, &[400..780, 810..1190]).unwrap();
    assert!(s.request(reused).unwrap().reused_prompt_tokens() >= 1537);
}

#[test]
fn atomic_prefill_spans_reject_invalid_geometry_before_execution() {
    for spans in [
        vec![10..10],
        vec![20..19],
        vec![0..513],
        vec![900..1100],
        vec![20..100, 90..110],
        vec![200..300, 10..20],
    ] {
        let mut s = scheduler(8, false);
        let h = s.admit(spec(None, 1024, 1)).unwrap();
        assert!(s.set_prefill_spans(h, &spans).is_err());
        assert_eq!(s.plan().unwrap().kind, GraphKind::Init);
    }
    let mut s = scheduler(8, false);
    let h = s.admit(spec(Some(512), 1024, 1)).unwrap();
    assert!(s.set_prefill_spans(h, &[400..700]).is_err());
    s.set_prefill_spans(h, &[512..700]).unwrap();
    run_step(&mut s);
    assert!(s.set_prefill_spans(h, &[800..900]).is_err());
}

#[test]
fn empty_atomic_spans_preserve_text_chunking() {
    let mut s = scheduler(16, false);
    let h = s.admit(spec(None, 1100, 1)).unwrap();
    s.set_prefill_spans(h, &[]).unwrap();
    let mut chunks = Vec::new();
    while s.request(h).unwrap().phase != RequestPhase::Finishing {
        let step = run_step(&mut s);
        if step.kind == GraphKind::Prefill {
            chunks.push(step.lanes[0].tokens);
        }
    }
    assert_eq!(chunks, [512, 512, 76]);
}

#[test]
fn message_and_prompt_checkpoints_evict_complete_page_chains_under_pressure() {
    for capacity in [2usize, 3, 16, 17] {
        let mut s = scheduler(capacity as u32, false);
        let turns = capacity + 3;
        let mut lengths = Vec::new();
        for turn in 0..turns {
            let length = 192 + 64 * turn;
            let mut request = spec(Some(length - 128), length, 1);
            request.prompt[0] = (turn + 1) as u32;
            let handle = s.admit(request).unwrap();
            while s.request(handle).unwrap().phase != RequestPhase::Finishing {
                run_step(&mut s);
            }
            s.finish(handle).unwrap();
            lengths.push(length);
        }
        let pairs = capacity / 2;
        let mut pages: usize = lengths[lengths.len() - pairs..].iter().map(|n| n / 64).sum();
        if capacity % 2 != 0 {
            pages += lengths[lengths.len() - pairs - 1] / 64;
        }
        assert_eq!(s.stats().cached_checkpoints, capacity);
        assert_eq!(s.stats().cached_pages, pages);
        assert_eq!(s.stats().snapshot_publications, (2 * turns) as u64);
    }
}

#[test]
fn every_message_endpoint_can_restore_after_a_changed_following_turn() {
    let endpoints = vec![1, 7, 63, 64, 65, 129, 300, 514, 700];
    for page_tokens in [64, 128, 256, 2048] {
        for mtp in [false, true] {
            let mut s = scheduler_with_pages(16, mtp, page_tokens);
            let mut request = spec(None, 901, 3);
            request.message_boundaries = endpoints.clone();
            let handle = s.admit(request).unwrap();
            for &end in &endpoints {
                stop_at_capture(&mut s, handle, end as u64);
                assert_eq!(s.request(handle).unwrap().generated_tokens, 0);
                run_step(&mut s);
                assert_eq!(s.request(handle).unwrap().prefix_tokens, end as u64);
            }
            finish(&mut s, handle);
            assert_eq!(s.stats().message_checkpoints, endpoints.len() as u64);
            for &end in &endpoints {
                let mut branch = spec(None, end + 10, 1);
                branch.prompt[end..].fill(1000);
                branch.message_boundaries = endpoints.iter().copied().filter(|&n| n <= end).collect();
                let branch = s.admit(branch).unwrap();
                assert_eq!(s.request(branch).unwrap().reused_prompt_tokens(), end as u64);
                assert_eq!(run_step(&mut s).kind, GraphKind::Restore);
                let step = s.plan().unwrap();
                assert_eq!(step.kind, GraphKind::Prefill);
                assert_eq!(step.lanes[0].start_position, end as u64);
                assert_eq!(step.lanes[0].tokens, 10);
                s.cancel(branch).unwrap();
            }
        }
    }
}

#[test]
fn many_message_boundaries_make_progress_with_bounded_snapshot_storage() {
    let mut s = scheduler(2, false);
    let mut request = spec(None, 100, 3);
    request.message_boundaries = (1..100).collect();
    let handle = s.admit(request).unwrap();
    for _ in 0..250 {
        if s.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        run_step(&mut s);
    }
    assert_eq!(s.request(handle).unwrap().phase, RequestPhase::Finishing);
    assert_eq!(s.request(handle).unwrap().generated_tokens, 3);
    s.finish(handle).unwrap();
    assert!(s.stats().cached_checkpoints <= 2);
}

#[test]
fn message_boundaries_must_be_ordered_unique_and_within_the_prompt() {
    for endpoints in [vec![0], vec![2, 1], vec![1, 1], vec![101]] {
        let mut s = scheduler(4, false);
        let mut request = spec(None, 100, 3);
        request.message_boundaries = endpoints;
        assert!(s.admit(request).is_err());
        assert_eq!(s.free_page_count(), 256);
    }
}
