use super::*;
use super::{policy::Candidate, utility::ValueRate};
use crate::execution::{ExecutionTiming, SamplingMode, TokenVector};

const MS: u64 = 1_000_000;

#[test]
fn padding_pruning_keeps_three_requests_on_batch_four_and_two_requests_on_batch_two() {
    let mut s = engine(8, false);
    let handles = (0..3).map(|id| add(&mut s, id, 64, 4096)).collect::<Vec<_>>();
    for handle in handles {
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.consumed_tokens = 64;
        request.scheduled_consumed = 64;
        request.generated_tokens = 1;
    }
    let (values, age, _) = s.lane_values(s.window_remaining().unwrap());
    let mut candidates = Vec::new();
    s.candidates(&mut candidates, &values, age);
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 8 && c.mask.count_ones() == 3));
    let count = s.prune_padding(&mut candidates);
    candidates.truncate(count);
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 4 && c.mask.count_ones() == 3));
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 2 && c.mask.count_ones() == 2));
    assert!(!candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 8));
}

#[test]
fn faster_padded_estimates_do_not_displace_equivalent_unpadded_work() {
    let mut s = engine(8, false);
    let h = add(&mut s, 1, 64, 128);
    ready(&mut s, &[h]);
    set_cost(&mut s, GraphKind::Decode, 1, 1, 10 * MS);
    for batch in [2, 4, 8] {
        set_cost(&mut s, GraphKind::Decode, batch, 1, MS);
    }
    let plan = s.plan().unwrap();
    assert_eq!((plan.kind, plan.batch_capacity, plan.lanes.len()), (GraphKind::Decode, 1, 1));
}

#[test]
fn equal_counts_from_different_cohorts_and_different_logical_prefixes_are_not_pruned() {
    let mut s = engine(4, true);
    let handles = (0..2).map(|id| add(&mut s, id, 64, 128)).collect::<Vec<_>>();
    ready(&mut s, &handles);
    let candidate = Candidate { kind: GraphKind::Speculative, mask: 1 << handles[0].slot,
        batch: 1, capacity: 1, packed_token_capacity: 0, prefill_capacity: 0, prefill_slot: None,
        work: 0, duration_ns: MS, rate: ValueRate::new(MS, 1.0, 0.0), feedback: true,
        verification_budget: None };
    assert!(!s.dominates_padding(candidate, Candidate { batch: 4, capacity: 4, ..candidate }));
    assert!(!s.dominates_padding(candidate, Candidate { batch: 4, mask: 1 << handles[1].slot, ..candidate }));
}

fn engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits {
            max_active: active,
            request_slots: active.next_power_of_two(),
            max_sequence_tokens: 8192,
            prefill_chunk_size: 512,
            pages: 1024,
            snapshots: 256,
        },
        SchedulerCapabilities {
            vocabulary: 1024,
            page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            GraphKind::Prefill => (g.token_capacity as u64 / 64 + 1) * MS,
            GraphKind::Decode => MS + (g.batch_capacity as u64 - 1) * MS / 4,
            GraphKind::Speculative => {
                MS + (g.batch_capacity as u64 - 1) * MS / 4 + (g.token_capacity as u64 - 1) * MS / 2
            }
            _ => MS / 10,
        };
    }
    s.measured_graph_durations(&graphs).unwrap();
    if speculative {
        s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    }
    s
}

fn add(s: &mut RequestScheduler, id: u32, prompt: usize, output: u32) -> RequestHandle {
    s.admit(RequestSpec {
        prompt: vec![id; prompt],
        message_boundaries: vec![],
        conditioning: vec![],
        max_new_tokens: output,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: false,
    })
    .unwrap()
}

fn complete(
    s: &mut RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
    publish: bool,
) {
    complete_timed(s, plan, fence, accepted, publish, Default::default());
}

fn completion_records(
    s: &RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
) -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    for (row, lane) in plan.lanes.iter().enumerate() {
        let kind = plan.lane_kind(row);
        let Ok(r) = s.request(lane.request) else {
            continue;
        };
        let count = match kind {
            GraphKind::Prefill => {
                usize::from(r.consumed_tokens + lane.tokens as u64 == r.prompt_tokens as u64)
            }
            GraphKind::Resume | GraphKind::Decode => 1,
            GraphKind::Speculative => lane.tokens.min(accepted + 1),
            _ => 0,
        };
        let consumed = if kind == GraphKind::Speculative {
            r.consumed_tokens + count as u64
        } else {
            lane.start_position + lane.tokens as u64
        };
        let generated = r.generated_tokens + count as u32;
        let mut record = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: if count == 0 { -1 } else { 99 },
            status: if count > 0 && generated == r.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
        if kind == GraphKind::Speculative {
            let mut tokens = [-1; 8];
            tokens[..count].fill(99);
            record.vector = Some(TokenVector {
                emitted_count: count as i64,
                offered_drafts: (plan.token_capacity - 1) as i64,
                accepted_drafts: (count - 1) as i64,
                verification_width: plan.token_capacity as i64,
                draft_cache_consumed: -1,
                tokens,
                ..Default::default()
            });
        }
        records[lane.request.slot as usize] = record;
    }
    records
}

fn complete_timed(
    s: &mut RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
    publish: bool,
    timing: ExecutionTiming,
) {
    let records = completion_records(s, plan, fence, accepted);
    let output = s.complete(fence, &records, timing).unwrap();
    if publish {
        for token in output {
            s.published_output(token.request, fence).unwrap();
        }
    }
    let finished: Vec<_> = s
        .slots
        .iter()
        .filter_map(|slot| slot.request.as_ref())
        .filter(|r| r.phase == RequestPhase::Finishing && r.pending == 0)
        .map(|r| r.handle)
        .collect();
    for h in finished {
        s.finish(h).unwrap();
    }
}

fn step(s: &mut RequestScheduler, accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("runnable trace must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + plan.estimated_duration_ns())
        .unwrap();
    complete_timed(
        s,
        &plan,
        fence,
        accepted,
        true,
        gpu_time(plan.estimated_duration_ns()),
    );
    plan
}

fn ready(s: &mut RequestScheduler, handles: &[RequestHandle]) {
    for _ in 0..200 {
        if handles
            .iter()
            .all(|h| s.request(*h).unwrap().phase == RequestPhase::Decode)
        {
            return;
        }
        step(s, 0);
    }
    panic!("requests did not reach decode");
}

#[test]
fn a_speculative_burst_counts_one_published_delivery() {
    let mut s = engine(1, true);
    s.fixed_speculative_width(Some(8)).unwrap();
    let h = add(&mut s, 1, 65, 64);
    ready(&mut s, &[h]);
    let deliveries = s.request(h).unwrap().accounting.published_deliveries;
    let output = s.request(h).unwrap().generated_tokens;
    step(&mut s, 7);
    let request = s.request(h).unwrap();
    assert_eq!(request.generated_tokens, output + 8);
    assert_eq!(request.accounting.published_deliveries, deliveries + 1);
    assert_eq!(request.accounting.last_output_fence, Some(s.completed));
}

#[test]
fn publication_requires_a_matching_output_completion() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 65, 64);
    assert!(s.published_output(h, Fence(0)).is_err());
    step(&mut s, 0);
    assert!(s.published_output(h, s.completed).is_err());
    ready(&mut s, &[h]);
    let earlier = s.request(h).unwrap().accounting.last_output_fence.unwrap();
    step(&mut s, 1);
    let before = *s.request(h).unwrap().accounting();
    assert!(s.published_output(h, earlier).is_err());
    assert!(s.published_output(h, Fence(s.submitted.0 + 1)).is_err());
    assert_eq!(
        s.request(h).unwrap().accounting.last_output_ns,
        before.last_output_ns
    );
    assert_eq!(
        s.request(h).unwrap().accounting.published_deliveries,
        before.published_deliveries
    );
}

#[test]
fn large_cohort_search_covers_high_slots_without_growing_scratch() {
    let mut s = engine(64, true);
    let handles: Vec<_> = (1..=64).map(|id| add(&mut s, id, 1, 64)).collect();
    for h in &handles {
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 1;
        r.scheduled_consumed = 1;
        r.generated_tokens = 1;
    }
    s.advance_to(100 * MS).unwrap();
    for h in &handles[..63] {
        s.request_mut(*h).unwrap().accounting.last_output_ns = Some(s.now_ns);
    }
    let capacity = s.candidates.borrow().capacity();
    let plan = s.plan().unwrap();
    assert!(plan.lanes.iter().any(|l| l.request == handles[63]));
    assert!(plan.selection_value_rate().is_finite());
    assert_eq!(s.candidates.borrow().capacity(), capacity);
    assert!(plan.lanes.iter().all(|l| l.tokens <= plan.token_capacity));
}

#[test]
fn prefix_restore_earns_no_credit_for_reused_prompt_tokens() {
    let mut s = engine(1, false);
    let seed = add(&mut s, 1, 65, 1);
    while s.request(seed).is_ok() {
        step(&mut s, 0);
    }
    let h = add(&mut s, 1, 65, 32);
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 65);
    assert_eq!(s.request(h).unwrap().accounting.committed_prefill_tokens, 0);
    let restore = step(&mut s, 0);
    assert_eq!(restore.kind, GraphKind::Restore);
    assert_eq!(s.request(h).unwrap().accounting.committed_output_tokens, 0);
    let resume = step(&mut s, 0);
    assert_eq!(resume.kind, GraphKind::Resume);
    let r = s.request(h).unwrap();
    assert_eq!(r.accounting.committed_output_tokens, 1);
}

#[test]
fn a_plan_keeps_its_duration_when_the_graph_estimate_changes() {
    let mut s = engine(1, false);
    add(&mut s, 1, 1, 32);
    let plan = s.plan().unwrap();
    let estimate = plan.estimated_duration_ns();
    let selected = plan.selected_cost;
    s.advance_to(1).unwrap();
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(80 * MS),
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
    assert_ne!(s.estimated_duration_ns(&plan), Some(estimate));
    assert_eq!(plan.estimated_duration_ns(), estimate);
    s.commit(&plan, Fence(1)).unwrap();
    let timing = s.submitted_timing(Fence(1)).unwrap();
    assert_eq!(timing.retained_ns, selected.retained_ns);
    assert_eq!(timing.observed_at_ns, selected.observed_at_ns);
    assert_eq!(timing.freshness, selected.freshness);
    assert_eq!(timing.selected_at_ns, 0);
    assert_eq!(timing.commit_ns, 1);
    assert_ne!(timing.retained_ns, s.estimated_duration_ns(&plan).unwrap());
    assert_eq!(s.window_remaining(), Some(estimate));
}

#[test]
fn atomic_prefill_respects_spans_without_a_duration_cap() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 256, 32);
    s.set_prefill_spans(h, &[0..256]).unwrap();
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(
        (plan.kind, plan.token_capacity, plan.lanes[0].tokens),
        (GraphKind::Prefill, 256, 256)
    );
    step(&mut s, 0);
    assert_eq!(s.request(h).unwrap().consumed_tokens, 256);
}

#[test]
fn short_prefill_does_not_explore_padding_after_the_smallest_shape_is_measured() {
    let mut s = cold_engine(1, false);
    s.prefill_graphs(&[64, 128, 512]).unwrap();
    let h = add(&mut s, 1, 130, 32);
    s.request_mut(h).unwrap().message_boundaries = vec![17, 82];
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Init);
    let first = measured_step(&mut s, MS, 0);
    assert_eq!((first.token_capacity, first.lanes[0].tokens), (64, 17));
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Snapshot);
    let second = measured_step(&mut s, 2 * MS, 0);
    assert_eq!((second.token_capacity, second.lanes[0].tokens), (128, 65));
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Snapshot);
    let tail = s.plan().unwrap();
    assert_eq!((tail.token_capacity, tail.lanes[0].tokens), (64, 48));
}

#[test]
fn indeterminate_and_equal_costs_naturally_choose_full_useful_prefill() {
    for known in [false, true] {
        let mut s = cold_engine(1, false);
        if known {
            let mut graphs = super::tests::duration_graphs(1);
            for graph in &mut graphs {
                graph.estimated_duration_ns = MS;
            }
            s.measured_graph_durations(&graphs).unwrap();
        }
        let h = add(&mut s, 1, 4096, 64);
        measured_step(&mut s, MS / 10, 0);
        let plan = measured_step(&mut s, 5 * MS, 0);
        assert_eq!(plan.kind, GraphKind::Prefill);
        assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (512, 512));
        let credit = s.request(h).unwrap().accounting();
        assert_eq!(credit.committed_prefill_tokens, 512);
        assert_eq!(credit.committed_output_tokens, 0);
    }
}

#[test]
fn indeterminate_and_equal_costs_naturally_choose_full_useful_speculation() {
    for known in [false, true] {
        for (limit, width) in [(64, 8), (3, 2)] {
            let mut s = cold_engine(1, true);
            if known {
                let mut graphs = super::tests::duration_graphs(1);
                for graph in &mut graphs {
                    graph.estimated_duration_ns = MS;
                }
                s.measured_graph_durations(&graphs).unwrap();
            }
            add(&mut s, 1, 1, limit);
            for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
                assert_eq!(measured_step(&mut s, MS / 10, 0).kind, kind);
            }
            let plan = s.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Speculative);
            assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (width, width));
        }
    }
}

#[test]
fn indeterminate_execution_occupies_the_submission_window_until_completion() {
    let mut s = cold_engine(2, false);
    add(&mut s, 1, 1, 64);
    let plan = s.plan().unwrap();
    assert_eq!(plan.estimated_duration_ns(), 0);
    assert!(plan.requires_feedback);
    s.commit(&plan, Fence(1)).unwrap();
    add(&mut s, 2, 1, 64);
    assert!(!s.submission_window_available(0));
    assert!(!s.submission_window_available(MS));
    assert!(s.plan().is_none());
    s.advance_to(MS).unwrap();
    complete(&mut s, &plan, Fence(1), 0, true);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(MS),
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
    assert_eq!(s.estimated_duration_ns(&plan), Some(MS));
    assert!(s.plan().is_some());
}

#[test]
fn stale_graph_cost_retains_its_last_valid_observation_and_resets_the_median() {
    let mut s = cold_engine(1, false);
    add(&mut s, 1, 1, 64);
    let plan = s.plan().unwrap();
    let ttl = s.policy.estimate_ttl_ns;
    s.advance_to(10 * MS).unwrap();
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(MS),
                Some(2 * MS),
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
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.advance_to(10 * MS + ttl - 1).unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(0),
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
    s.advance_to(10 * MS + ttl).unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    assert!(!s.costs.is_fresh(cost::Key::from_step(&plan)));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            wall_ns: MS,
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(8 * MS),
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
    assert_eq!(s.estimated_duration_ns(&plan), Some(8 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(16 * MS),
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
    assert_eq!(s.estimated_duration_ns(&plan), Some(12 * MS));
}

#[test]
fn funded_stale_graph_recheck_recovers_a_poisoned_observation() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 1, 64);
    ready(&mut s, &[h]);
    let mut graphs = super::tests::duration_graphs(1);
    for graph in &mut graphs {
        graph.estimated_duration_ns = if graph.token_capacity == 8 {
            2 * MS
        } else {
            4 * MS
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let plan = s.plan().unwrap();
    assert_eq!(plan.token_capacity, 8);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(1_000 * MS),
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
    assert_eq!(s.estimated_duration_ns(&plan), Some(501_000_000));
    assert_ne!(s.plan().unwrap().token_capacity, 8);
    let ttl = s.policy.estimate_ttl_ns;
    s.advance_to(ttl / 2).unwrap();
    for width in [1, 2, 4] {
        let mut other = (*plan).clone();
        other.token_capacity = width;
        s.observe_execution(
            &other,
            ExecutionTiming {
                gpu_ns: [
                    Some(4 * MS),
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
    s.advance_to(ttl).unwrap();
    assert_ne!(s.plan().unwrap().token_capacity, 8);
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let retry = measured_step(&mut s, 2 * MS, 7);
    assert_eq!(
        (retry.token_capacity, retry.estimated_duration_ns()),
        (8, 501_000_000)
    );
    assert!(matches!(retry.origin, PlanOrigin::RefreshStart(_)));
    assert_eq!(s.estimated_duration_ns(&retry), Some(2 * MS));
}

#[test]
fn neglected_stale_shapes_require_funding_while_the_winner_stays_fresh() {
    let mut s = cold_engine(1, true);
    s.configure_policy(PolicyConfig {
        estimate_ttl_ns: 20 * MS,
        ..s.policy
    })
    .unwrap();
    let h = add(&mut s, 1, 1, 1024);
    for _ in 0..3 {
        measured_step(&mut s, MS / 10, 0);
    }
    let mut widths = Vec::new();
    for _ in 0..4 {
        widths.push(measured_step(&mut s, MS, 7).token_capacity);
    }
    assert_eq!(widths, [8, 4, 2, 1]);
    for _ in 0..15 {
        assert_eq!(measured_step(&mut s, MS, 7).token_capacity, 8);
    }
    for _ in 0..10 {
        let plan = measured_step(&mut s, MS, 7);
        assert_eq!(plan.token_capacity, 8);
        assert_eq!(plan.origin, PlanOrigin::Ordinary);
    }
    let mut retried = Vec::new();
    for _ in 0..3 {
        s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
        let plan = measured_step(&mut s, MS, 7);
        if matches!(plan.origin, PlanOrigin::RefreshStart(_)) {
            retried.push(plan.token_capacity);
        }
    }
    assert!(retried.contains(&4) && retried.contains(&2) && retried.contains(&1));
    assert!(!retried.contains(&8));
    assert!(s.request(h).unwrap().accounting.committed_output_tokens > 0);
}

fn gpu_time(ns: u64) -> ExecutionTiming {
    let mut gpu_ns = [None; 16];
    gpu_ns[0] = Some(ns);
    ExecutionTiming {
        gpu_ns,
        ..Default::default()
    }
}

fn price_close(actual: utility::ServicePrice, expected: utility::ServicePrice) {
    assert!((actual.log_price(1.0) - expected.log_price(1.0)).abs() < 1e-12);
}

#[test]
fn service_uses_validated_progress_and_serial_execution_time_once() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 4096, 32);
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Prefill);
    let rate = s.service_rates(s.request(h).unwrap()).0.unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let mut expected = s.request(h).unwrap().accounting.service.tokens.unwrap();
    expected.observe(
        500.0,
        plan.lanes[0].tokens as f64,
        rate,
        s.policy.adaptation_ms,
    );
    let mut timing = gpu_time(2 * MS);
    timing.gpu_ns[1] = Some(3 * MS);
    timing.wall_ns = 500 * MS;
    s.advance_to(s.now_ns + timing.wall_ns).unwrap();
    complete_timed(&mut s, &plan, fence, 0, true, timing);
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        expected,
    );
    s.advance_to(s.now_ns + 100 * MS).unwrap();
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        expected,
    );
}

#[test]
fn pipelined_frames_use_ordered_completion_deltas_and_one_delivery_per_frame() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    let (token_rate, delivery_rate) = s.service_rates(s.request(h).unwrap());
    let first = s.plan().unwrap();
    let a = Fence(s.submitted.0 + 1);
    s.commit(&first, a).unwrap();
    let second = s.plan().unwrap();
    let b = Fence(a.0 + 1);
    s.commit(&second, b).unwrap();
    let before = s.request(h).unwrap().accounting;
    let mut tokens = before.service.tokens.unwrap();
    let mut deliveries = before.service.deliveries.unwrap();
    for (plan, fence, ms) in [(&first, a, 2), (&second, b, 3)] {
        tokens.observe(ms as f64, 1.0, token_rate.unwrap(), s.policy.adaptation_ms);
        deliveries.observe(
            ms as f64,
            1.0,
            delivery_rate.unwrap(),
            s.policy.adaptation_ms,
        );
        complete_timed(&mut s, plan, fence, 0, true, gpu_time(ms * MS));
        let after = s.request(h).unwrap().accounting;
        price_close(after.service.tokens.unwrap(), tokens);
        price_close(after.service.deliveries.unwrap(), deliveries);
    }
    assert_eq!(
        s.request(h).unwrap().accounting.published_deliveries,
        before.published_deliveries + 2
    );
}

#[test]
fn speculative_token_volume_and_visible_burst_service_have_separate_prices() {
    let mut s = engine(1, true);
    s.fixed_speculative_width(Some(8)).unwrap();
    let h = add(&mut s, 1, 65, 64);
    ready(&mut s, &[h]);
    let (token_rate, delivery_rate) = s.service_rates(s.request(h).unwrap());
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let before = s.request(h).unwrap().accounting.service;
    let mut token_price = before.tokens.unwrap();
    token_price.observe(30.0, 8.0, token_rate.unwrap(), s.policy.adaptation_ms);
    let mut hidden_price = before.deliveries.unwrap();
    hidden_price.observe(30.0, 0.0, delivery_rate.unwrap(), s.policy.adaptation_ms);
    complete_timed(&mut s, &plan, fence, 7, false, gpu_time(30 * MS));
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        token_price,
    );
    price_close(
        s.request(h).unwrap().accounting.service.deliveries.unwrap(),
        hidden_price,
    );
    let mut published_price = before.deliveries.unwrap();
    published_price.observe(30.0, 1.0, delivery_rate.unwrap(), s.policy.adaptation_ms);
    for _ in 0..8 {
        s.published_output(h, fence).unwrap();
        price_close(
            s.request(h).unwrap().accounting.service.deliveries.unwrap(),
            published_price,
        );
    }
}

#[test]
fn invalid_completion_changes_neither_prices_nor_cost_estimates() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let before = s.request(h).unwrap().accounting.service;
    let cost = s.estimated_duration_ns(&plan);
    let mut records = completion_records(&s, &plan, fence, 0);
    records[h.slot as usize].generation += 1;
    assert!(s.complete(fence, &records, gpu_time(100 * MS)).is_err());
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        before.tokens.unwrap(),
    );
    price_close(
        s.request(h).unwrap().accounting.service.deliveries.unwrap(),
        before.deliveries.unwrap(),
    );
    assert_eq!(s.estimated_duration_ns(&plan), cost);
    assert_eq!(s.submitted_step(fence).unwrap(), &plan.step);
}

#[test]
fn missing_device_timing_does_not_substitute_host_or_estimated_time() {
    for gpu_ns in [None, Some(0)] {
        let mut s = engine(1, false);
        let h = add(&mut s, 1, 65, 32);
        ready(&mut s, &[h]);
        let plan = s.plan().unwrap();
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let before = s.request(h).unwrap().accounting.service;
        let mut timing = gpu_time(0);
        timing.gpu_ns[0] = gpu_ns;
        timing.wall_ns = 500 * MS;
        complete_timed(&mut s, &plan, fence, 0, true, timing);
        price_close(
            s.request(h).unwrap().accounting.service.tokens.unwrap(),
            before.tokens.unwrap(),
        );
        price_close(
            s.request(h).unwrap().accounting.service.deliveries.unwrap(),
            before.deliveries.unwrap(),
        );
    }
}

#[test]
fn an_arrival_during_a_graph_is_not_charged_for_unavailable_service() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 32);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let arrival = add(&mut s, 2, 65, 32);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(10 * MS));
    assert!(
        s.request(arrival)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_none()
    );
}

#[test]
fn service_snapshot_does_not_update_a_reused_request_slot() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 32);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let cancelled = add(&mut s, 2, 65, 32);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    assert!(
        s.request(cancelled)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_some()
    );
    s.cancel(cancelled).unwrap();
    let replacement = add(&mut s, 3, 65, 32);
    assert_eq!(replacement.slot, cancelled.slot);
    assert_ne!(replacement.generation, cancelled.generation);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(10 * MS));
    assert!(
        s.request(replacement)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_none()
    );
}

#[test]
fn stale_costs_remain_available_as_service_normalizers() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    assert!(s.service_rates(s.request(h).unwrap()).0.is_some());
    let rates = s.service_rates(s.request(h).unwrap());
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(s.service_rates(s.request(h).unwrap()), rates);
}

#[test]
fn unserved_prices_grow_with_device_time_and_inbound_waiters_are_excluded() {
    let mut s = engine(3, false);
    let first = add(&mut s, 1, 65, 64);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let waiting = add(&mut s, 2, 4096, 64);
    let inbound = add(&mut s, 3, 4096, 64);
    s.request_mut(inbound).unwrap().phase = RequestPhase::Inbound;
    let rate = s.service_rates(s.request(waiting).unwrap()).0.unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let mut expected = s
        .request(waiting)
        .unwrap()
        .accounting
        .service
        .tokens
        .unwrap();
    expected.observe(50.0, 0.0, rate, s.policy.adaptation_ms);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(50 * MS));
    price_close(
        s.request(waiting)
            .unwrap()
            .accounting
            .service
            .tokens
            .unwrap(),
        expected,
    );
    assert!(
        s.request(inbound)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_none()
    );
    assert!(
        s.request(inbound)
            .unwrap()
            .accounting
            .service
            .deliveries
            .is_none()
    );
}

#[test]
fn price_updates_preserve_an_unrelated_immutable_plan() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 64);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let second = add(&mut s, 2, 4096, 64);
    s.request_mut(second).unwrap().phase = RequestPhase::Prefill;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let other = s.phase_plan(GraphKind::Prefill, Some(second));
    assert_eq!(other.lanes.len(), 1);
    assert_eq!(other.lanes[0].request, second);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(MS));
    s.commit(&other, Fence(fence.0 + 1)).unwrap();
}

#[test]
fn late_publication_cannot_rewind_prices_past_another_completion() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 64);
    ready(&mut s, &[h]);
    let first = s.plan().unwrap();
    let a = Fence(s.submitted.0 + 1);
    s.commit(&first, a).unwrap();
    complete_timed(&mut s, &first, a, 0, false, gpu_time(MS));
    let second = s.plan().unwrap();
    let b = Fence(a.0 + 1);
    s.commit(&second, b).unwrap();
    complete_timed(&mut s, &second, b, 0, false, gpu_time(MS));
    let before = s.request(h).unwrap().accounting.service.deliveries.unwrap();
    assert!(s.published_output(h, a).is_err());
    price_close(
        s.request(h).unwrap().accounting.service.deliveries.unwrap(),
        before,
    );
    s.published_output(h, b).unwrap();
}

#[test]
fn first_published_output_initializes_delivery_history_and_preserves_token_price() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 64);
    step(&mut s, 0);
    let prefill = s.plan().unwrap();
    let a = Fence(s.submitted.0 + 1);
    s.commit(&prefill, a).unwrap();
    let mut tokens = s.request(h).unwrap().accounting.service.tokens.unwrap();
    let rate = s.service_rates(s.request(h).unwrap()).0.unwrap();
    tokens.observe(3.0, 66.0, rate, s.policy.adaptation_ms);
    complete_timed(&mut s, &prefill, a, 0, true, gpu_time(3 * MS));
    let service = s.request(h).unwrap().accounting.service;
    price_close(service.tokens.unwrap(), tokens);
    let deliveries = utility::ServicePrice::initial(1.0 / 3.0);
    price_close(service.deliveries.unwrap(), deliveries);
    let snapshot = s.plan().unwrap();
    assert_eq!(snapshot.kind, GraphKind::Snapshot);
    s.commit(&snapshot, Fence(a.0 + 1)).unwrap();
    let service = s.request(h).unwrap().accounting.service;
    price_close(service.tokens.unwrap(), tokens);
    price_close(service.deliveries.unwrap(), deliveries);
}

fn cold_engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = engine(active, speculative);
    s.graph_durations(&super::tests::duration_graphs(active))
        .unwrap();
    s
}

fn measured_step(s: &mut RequestScheduler, duration_ns: u64, accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("runnable request must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + duration_ns).unwrap();
    complete_timed(s, &plan, fence, accepted, true, gpu_time(duration_ns));
    plan
}

fn interactive_engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [17; 32],
        SchedulerLimits {
            max_active: active,
            request_slots: active.next_power_of_two(),
            max_sequence_tokens: 65536,
            prefill_chunk_size: 4096,
            pages: 32768,
            snapshots: 256,
        },
        SchedulerCapabilities {
            vocabulary: 1024,
            page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    s.prefill_graphs(&[64, 128, 256, 512, 1024, 2048, 4096])
        .unwrap();
    if speculative {
        s.enable_speculation(&[1, 3, 8]).unwrap();
    }
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            GraphKind::Prefill => 2 * MS + g.token_capacity as u64 * MS / 64,
            GraphKind::Decode => (15 + 2 * (g.batch_capacity as u64 - 1)) * MS,
            GraphKind::Speculative => {
                let solo = match g.token_capacity {
                    1 => 15 * MS,
                    3 => 17_300_000,
                    8 => 30 * MS,
                    n => (15 + 2 * n as u64) * MS,
                };
                solo + 2 * (g.batch_capacity as u64 - 1) * MS
            }
            _ => MS / 10,
        };
    }
    s.measured_graph_durations(&graphs).unwrap();
    s
}

fn set_cost(s: &mut RequestScheduler, kind: GraphKind, batch: usize, capacity: usize, nanos: u64) {
    let step = ScheduledStep { packed_token_capacity: 0,
        kind,
        batch_capacity: batch,
        token_capacity: capacity,
        mixed: None,
        lanes: Vec::new(),
        requires_feedback: false,
    };
    s.costs.set_measured(&step, nanos);
}

#[test]
fn one_two_and_sixteen_long_prefills_use_full_efficient_chunks() {
    for count in [1, 2, 16] {
        let mut s = interactive_engine(count, true);
        let handles: Vec<_> = (1..=count as u32)
            .map(|id| add(&mut s, id, 32768, 128))
            .collect();
        let mut chunks = 0;
        while chunks < count * 3 {
            let plan = step(&mut s, 0);
            if plan.kind == GraphKind::Prefill {
                assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (4096, 4096));
                chunks += 1;
            }
            assert!(
                handles
                    .iter()
                    .all(|h| s.request(*h).unwrap().generated_tokens == 0)
            );
        }
        assert!(
            handles
                .iter()
                .all(|h| s.request(*h).unwrap().consumed_tokens > 0)
        );
    }
}

#[test]
fn small_prefill_tails_choose_the_best_effective_value_rate() {
    for (prompt, capacity) in [(64, 64), (511, 512), (4095, 4096)] {
        let mut s = interactive_engine(1, false);
        add(&mut s, 1, prompt, 128);
        assert_eq!(step(&mut s, 0).kind, GraphKind::Init);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Prefill);
        assert_eq!(
            (plan.token_capacity, plan.lanes[0].tokens),
            (capacity, prompt)
        );
    }
}

#[test]
fn equivalent_prefill_tails_use_the_smaller_graph_despite_a_faster_padded_estimate() {
    let mut s = interactive_engine(1, false);
    set_cost(&mut s, GraphKind::Prefill, 1, 4096, MS);
    add(&mut s, 1, 64, 128);
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (64, 64));
}

#[test]
fn efficient_thirty_millisecond_speculation_beats_seventeen_millisecond_rounds() {
    let mut s = interactive_engine(1, true);
    let h = add(&mut s, 1, 64, 2048);
    ready(&mut s, &[h]);
    for _ in 0..128 {
        s.request_mut(h).unwrap().acceptance.observe(7, 7, true, 0);
    }
    let plan = s.plan().unwrap();
    assert_eq!(
        (plan.kind, plan.token_capacity, plan.estimated_duration_ns()),
        (GraphKind::Speculative, 8, 30 * MS)
    );
    for _ in 0..256 {
        s.request_mut(h).unwrap().acceptance.observe(7, 0, true, 0);
    }
    assert!(s.plan().unwrap().token_capacity < 8);
}

#[test]
fn equivalent_decode_work_uses_the_smaller_batch_despite_a_faster_padded_estimate() {
    let mut s = interactive_engine(4, false);
    let h = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[h]);
    set_cost(&mut s, GraphKind::Decode, 4, 1, 10 * MS);
    let plan = s.plan().unwrap();
    assert_eq!((plan.batch_capacity, plan.lanes.len()), (1, 1));
    assert_eq!(plan.lanes[0].request, h);
}

#[test]
fn speculative_cohorts_are_selected_by_expected_lane_value() {
    let mut s = interactive_engine(8, true);
    let handles: Vec<_> = (1..=8).map(|id| add(&mut s, id, 64, 2048)).collect();
    for (index, h) in handles.iter().enumerate() {
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 64;
        r.scheduled_consumed = 64;
        r.generated_tokens = 1;
        for _ in 0..128 {
            r.acceptance
                .observe(7, if index >= 6 { 7 } else { 0 }, true, 0);
        }
    }
    for batch in [1, 2, 4, 8] {
        for width in [1, 3, 8] {
            set_cost(&mut s, GraphKind::Speculative, batch, width, 200 * MS);
        }
    }
    set_cost(&mut s, GraphKind::Speculative, 2, 8, 30 * MS);
    let plan = s.plan().unwrap();
    assert_eq!((plan.batch_capacity, plan.token_capacity), (2, 8));
    assert_eq!(
        plan.lanes.iter().map(|l| l.request).collect::<Vec<_>>(),
        handles[6..]
    );
}

#[test]
fn two_long_decodes_keep_both_streams_flowing() {
    for accepted in [0, 7] {
        let mut s = interactive_engine(2, true);
        let handles = [add(&mut s, 1, 64, 2048), add(&mut s, 2, 64, 2048)];
        let mut last = [None; 2];
        let mut gaps = [0; 2];
        for _ in 0..200 {
            let plan = step(&mut s, accepted);
            if plan.kind == GraphKind::Speculative {
                for lane in &plan.lanes {
                    let i = handles.iter().position(|h| *h == lane.request).unwrap();
                    if let Some(t) = last[i] {
                        gaps[i] = gaps[i].max(s.now_ns - t);
                    }
                    last[i] = Some(s.now_ns);
                }
            }
        }
        assert!(
            handles
                .iter()
                .all(|h| s.request(*h).unwrap().generated_tokens > 100)
        );
        assert!(gaps.iter().all(|g| *g <= 2 * 32 * MS), "gaps={gaps:?}");
    }
}

#[test]
fn long_prefill_and_low_acceptance_decode_both_make_progress() {
    let mut s = interactive_engine(2, true);
    let decode = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[decode]);
    for _ in 0..128 {
        s.request_mut(decode)
            .unwrap()
            .acceptance
            .observe(7, 0, true, 0);
    }
    let prefill = add(&mut s, 2, 32768, 128);
    let mut last = s.now_ns;
    let mut gap = 0;
    let mut prefill_chunks = 0;
    for _ in 0..400 {
        let plan = step(&mut s, 0);
        if plan.lanes.iter().any(|l| l.request == decode) && plan.kind == GraphKind::Speculative {
            gap = gap.max(s.now_ns - last);
            last = s.now_ns;
        }
        if plan.kind == GraphKind::Prefill {
            prefill_chunks += 1;
        }
        if s.request(prefill).is_ok_and(|r| r.consumed_tokens >= 32768) {
            break;
        }
    }
    assert!(s.request(prefill).unwrap().consumed_tokens >= 32768);
    assert!(s.request(decode).unwrap().generated_tokens > 1);
    gap = gap.max(s.now_ns - last);
    assert!(prefill_chunks >= 8);
    assert!(gap < 300 * MS, "output gap={} ms", gap / MS);
}

#[test]
fn two_prefill_frames_can_exceed_twenty_milliseconds_without_a_time_window_cap() {
    let mut s = interactive_engine(2, false);
    add(&mut s, 1, 32768, 128);
    add(&mut s, 2, 32768, 128);
    step(&mut s, 0);
    let first = s.plan().unwrap();
    assert_eq!(
        (first.kind, first.token_capacity),
        (GraphKind::Prefill, 4096)
    );
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    let second = s.plan().unwrap();
    assert_eq!(
        (second.kind, second.token_capacity),
        (GraphKind::Prefill, 4096)
    );
    assert_ne!(first.lanes[0].request, second.lanes[0].request);
    s.commit(&second, Fence(fence.0 + 1)).unwrap();
    assert!(s.plan().is_none());
    assert!(first.estimated_duration_ns() + second.estimated_duration_ns() > 20 * MS);
}

#[test]
fn decode_waiting_selects_a_smaller_prefill_at_the_efficiency_knee() {
    let mut s = interactive_engine(5, true);
    s.prefill_graphs(&[64, 512, 4096]).unwrap();
    s.configure_policy(PolicyConfig {
        output_token_value: 16.0,
        token_fairness: 0.0,
        delivery_fairness: 0.0,
        decode_age_price: 0.04,
        ..s.policy
    })
    .unwrap();
    set_cost(&mut s, GraphKind::Prefill, 1, 512, 8 * MS);
    set_cost(&mut s, GraphKind::Prefill, 1, 4096, 62 * MS);
    let prefill = add(&mut s, 1, 32768, 128);
    step(&mut s, 0);
    let decodes: Vec<_> = (2..=5).map(|id| add(&mut s, id, 64, 2048)).collect();
    for h in &decodes {
        let now = s.now_ns;
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 64;
        r.scheduled_consumed = 64;
        r.generated_tokens = 1;
        r.accounting.last_output_ns = Some(now);
        for _ in 0..128 {
            r.acceptance.observe(7, 7, true, 0);
        }
    }
    let smaller = s.plan().unwrap();
    assert_eq!(
        (smaller.kind, smaller.token_capacity),
        (GraphKind::Prefill, 512)
    );
    assert_eq!(smaller.lanes[0].request, prefill);
    s.advance_to(s.now_ns + 1000 * MS).unwrap();
    assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
    for h in decodes {
        s.cancel(h).unwrap();
    }
    let larger = s.plan().unwrap();
    assert_eq!(
        (larger.kind, larger.token_capacity),
        (GraphKind::Prefill, 4096)
    );
}

#[test]
fn throughput_terminal_snapshot_completes_before_a_competing_long_prompt() {
    for token_fairness in [0.0, PolicyConfig::throughput().token_fairness] {
        let mut s = interactive_engine(2, false);
        s.configure_policy(PolicyConfig { token_fairness, ..PolicyConfig::throughput() }).unwrap();
        let short = add(&mut s, 1, 64, 1);
        step(&mut s, 0);
        step(&mut s, 0);
        assert_eq!(s.request(short).unwrap().phase, RequestPhase::Snapshot);
        let long = add(&mut s, 2, 32768, 128);
        let start = s.now_ns;
        while s.request(short).is_ok() && s.now_ns - start < 1000 * MS {
            step(&mut s, 0);
        }
        assert!(s.request(short).is_err(), "a generated answer must not wait for unrelated prefill");
        assert!(s.request(long).unwrap().consumed_tokens < 32768,
            "terminal delivery needs useful value even without delivery fairness or age pricing");
    }
}

#[test]
fn terminal_snapshot_completes_while_another_long_prompt_is_still_prefilling() {
    let mut s = interactive_engine(2, false);
    let short = add(&mut s, 1, 64, 1);
    step(&mut s, 0);
    step(&mut s, 0);
    assert_eq!(s.request(short).unwrap().phase, RequestPhase::Snapshot);
    let long = add(&mut s, 2, 32768, 128);
    let start = s.now_ns;
    while s.request(short).is_ok() && s.now_ns - start < 1000 * MS {
        step(&mut s, 0);
    }
    assert!(s.request(short).is_err());
    assert!(s.request(long).unwrap().consumed_tokens < 32768);
}

#[test]
fn five_and_sixteen_streams_receive_repeated_output_service() {
    for (count, accepted, known_acceptance) in [5, 16].into_iter().flat_map(|n| {
        [0, 3, 7]
            .into_iter()
            .flat_map(move |a| [false, true].map(|k| (n, a, k)))
    }) {
        let mut s = interactive_engine(count, true);
        let handles: Vec<_> = (1..=count as u32)
            .map(|id| add(&mut s, id, 64, 4096))
            .collect();
        if known_acceptance {
            for &h in &handles {
                for _ in 0..128 {
                    s.request_mut(h)
                        .unwrap()
                        .acceptance
                        .observe(7, accepted, true, 0);
                }
            }
        }
        let mut first = vec![None; count];
        let mut last = vec![None; count];
        let mut gap = vec![0; count];
        for _ in 0..150 {
            let plan = step(&mut s, accepted);
            if matches!(plan.kind, GraphKind::Prefill | GraphKind::Speculative) {
                for lane in &plan.lanes {
                    let i = handles.iter().position(|h| *h == lane.request).unwrap();
                    first[i].get_or_insert(s.now_ns);
                    if let Some(at) = last[i] {
                        gap[i] = gap[i].max(s.now_ns - at);
                    }
                    last[i] = Some(s.now_ns);
                }
            }
        }
        for (i, at) in last.iter().enumerate() {
            gap[i] = gap[i].max(s.now_ns - at.unwrap_or(0));
        }
        assert!(
            first.iter().all(|t| t.is_some_and(|t| t < 100 * MS)),
            "count={count} accepted={accepted} known={known_acceptance} first={first:?}"
        );
        assert!(
            gap.iter().all(|g| *g < 150 * MS),
            "count={count} accepted={accepted} known={known_acceptance} gaps={gap:?}"
        );
        let before: Vec<_> = handles
            .iter()
            .map(|h| s.request(*h).unwrap().generated_tokens)
            .collect();
        for _ in 0..100 {
            step(&mut s, accepted);
        }
        let progress: Vec<_> = handles
            .iter()
            .enumerate()
            .map(|(i, h)| s.request(*h).unwrap().generated_tokens - before[i])
            .collect();
        assert!(
            *progress.iter().min().unwrap() >= 50,
            "progress={progress:?}"
        );
        assert!(
            progress.iter().max().unwrap() - progress.iter().min().unwrap() <= 8,
            "count={count} accepted={accepted} known={known_acceptance} progress={progress:?}"
        );
    }
}

#[test]
fn delivery_reference_uses_batch_throughput_and_excludes_prefill_only_requests() {
    let mut s = interactive_engine(16, true);
    let handles: Vec<_> = (1..=2).map(|id| add(&mut s, id, 64, 4096)).collect();
    ready(&mut s, &handles);
    let references = |s: &RequestScheduler| {
        let rates = std::array::from_fn(|slot| {
            s.slots
                .get(slot)
                .and_then(|s| s.request.as_ref())
                .and_then(|r| s.service_rates(r).1)
        });
        s.delivery_reference_intervals(&rates)
    };
    assert_eq!(references(&s)[handles[0].slot as usize], Some(17.0));
    for id in 3..=16 {
        add(&mut s, id, 32768, 128);
    }
    assert_eq!(references(&s)[handles[0].slot as usize], Some(17.0));
    set_cost(&mut s, GraphKind::Speculative, 16, 1, MS);
    assert_eq!(references(&s)[handles[0].slot as usize], Some(1.0));
    let retained = references(&s);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(references(&s), retained);
}

#[test]
fn snapshot_enabling_a_shared_decode_batch_competes_by_the_complete_sequence_cost() {
    let mut s = interactive_engine(16, true);
    let handles: Vec<_> = (1..=16).map(|id| add(&mut s, id, 64, 4096)).collect();
    for &h in &handles {
        for _ in 0..128 {
            s.request_mut(h).unwrap().acceptance.observe(7, 7, true, 0);
        }
    }
    for _ in 0..200 {
        let decoding = handles
            .iter()
            .filter(|h| s.request(**h).unwrap().phase == RequestPhase::Decode)
            .count();
        let copying = handles
            .iter()
            .filter(|h| s.request(**h).unwrap().phase == RequestPhase::Snapshot)
            .count();
        if decoding == 15 && copying == 1 {
            break;
        }
        step(&mut s, 7);
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.lanes.len(), 1);
    assert_eq!(plan.estimated_duration_ns(), MS / 10);
    let handle = plan.lanes[0].request;
    let tokens = s.request(handle).unwrap().generated_tokens;
    for batch in crate::execution::BATCH_CAPACITIES
        .into_iter()
        .take_while(|b| *b <= 16)
    {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, 1000 * MS);
    }
    assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
    for batch in crate::execution::BATCH_CAPACITIES
        .into_iter()
        .take_while(|b| *b <= 16)
    {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, MS / 10);
    }
    assert_eq!(step(&mut s, 7).kind, GraphKind::Snapshot);
    assert_eq!(s.request(handle).unwrap().generated_tokens, tokens);
    let batch = s.plan().unwrap();
    assert_eq!(batch.kind, GraphKind::Speculative);
    assert_eq!(batch.lanes.len(), 16);
}

#[test]
fn equally_costly_snapshot_can_finish_one_request_and_enable_another() {
    let mut s = interactive_engine(2, true);
    let terminal = add(&mut s, 1, 64, 1);
    let continuing = add(&mut s, 2, 64, 4096);
    step(&mut s, 7);
    for handle in [terminal, continuing] {
        let plan = s.phase_plan(GraphKind::Prefill, Some(handle));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        s.advance_to(s.now_ns + plan.estimated_duration_ns())
            .unwrap();
        complete_timed(
            &mut s,
            &plan,
            fence,
            7,
            true,
            gpu_time(plan.estimated_duration_ns()),
        );
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.lanes.len(), 2);
    step(&mut s, 7);
    assert!(s.request(terminal).is_err());
    assert_eq!(s.request(continuing).unwrap().phase, RequestPhase::Decode);
    assert_eq!(s.request(continuing).unwrap().generated_tokens, 1);
}

#[test]
fn prerequisite_cohort_accounts_for_ready_batch_mates() {
    let mut s = interactive_engine(16, true);
    s.configure_policy(PolicyConfig {
        token_fairness: 0.0,
        delivery_fairness: 0.0,
        decode_age_price: 0.0,
        ..Default::default()
    })
    .unwrap();
    let execute = |s: &mut RequestScheduler, kind, handle| {
        let plan = s.phase_plan(kind, Some(handle));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        s.advance_to(s.now_ns + plan.estimated_duration_ns())
            .unwrap();
        complete_timed(
            s,
            &plan,
            fence,
            7,
            true,
            gpu_time(plan.estimated_duration_ns()),
        );
    };
    let ready_handles: Vec<_> = (1..=14).map(|id| add(&mut s, id, 64, 4096)).collect();
    for &handle in &ready_handles {
        for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
            execute(&mut s, kind, handle);
        }
    }
    for &handle in &ready_handles {
        for _ in 0..128 {
            s.request_mut(handle)
                .unwrap()
                .acceptance
                .observe(7, 7, true, 0);
        }
    }
    let decode = add(&mut s, 15, 64, 4096);
    let prefill = s
        .admit(RequestSpec {
            prompt: vec![16; 32768],
            message_boundaries: vec![512],
            conditioning: vec![],
            max_new_tokens: 128,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .unwrap();
    for handle in [decode, prefill] {
        for kind in [GraphKind::Init, GraphKind::Prefill] {
            execute(&mut s, kind, handle);
        }
    }
    for batch in [2, 4, 8, 16] {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, 1000 * MS);
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.batch_capacity, 1);
    assert_eq!(plan.lanes[0].request, decode);
}

#[test]
fn invalid_utility_configuration_is_rejected_before_admission() {
    for policy in [
        PolicyConfig {
            output_token_value: f64::INFINITY,
            ..Default::default()
        },
        PolicyConfig {
            token_fairness: -1.0,
            ..Default::default()
        },
        PolicyConfig {
            delivery_fairness: f64::NAN,
            ..Default::default()
        },
        PolicyConfig {
            decode_age_price: -1.0,
            ..Default::default()
        },
        PolicyConfig {
            adaptation_ms: 0.0,
            ..Default::default()
        },
        PolicyConfig {
            estimate_ttl_ns: 0,
            ..Default::default()
        },
    ] {
        assert!(engine(1, false).configure_policy(policy).is_err());
    }
}

fn mixed_engine(
    active: usize,
    speculative: bool,
    shapes: &[(usize, usize, usize, u64)],
) -> RequestScheduler {
    let mut s = interactive_engine(active, speculative);
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = s
            .costs
            .duration(g.kind, g.batch_capacity, g.token_capacity)
            .unwrap();
    }
    for &(batch, width, prefill, nanos) in shapes {
        graphs.push(crate::execution::GraphDescriptor { packed_token_capacity: 0,
            name: format!("mixed-{batch}-{prefill}-{width}"),
            kind: GraphKind::Mixed,
            sampling_mode: SamplingMode::General,
            handle: graphs.len() + 1,
            batch_capacity: batch,
            token_capacity: width,
            prefill_capacity: prefill,
            estimated_duration_ns: nanos,
            input_bindings: vec![],
        });
    }
    s.measured_graph_durations(&graphs).unwrap();
    s.mixed_graphs(
        &shapes
            .iter()
            .map(|&(b, w, p, _)| (b, w, p))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    s
}

fn throughput_only(s: &mut RequestScheduler) {
    s.configure_policy(PolicyConfig {
        token_fairness: 0.0,
        delivery_fairness: 0.0,
        decode_age_price: 0.0,
        ..Default::default()
    })
    .unwrap();
}

fn initialize(s: &mut RequestScheduler, handle: RequestHandle) {
    let plan = s.phase_plan(GraphKind::Init, Some(handle));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    complete(s, &plan, fence, 0, true);
}

#[test]
fn mixed_prefill_exploration_requires_additional_useful_tokens() {
    for prompt_tokens in [0, 17, 512] {
        let mut s = mixed_engine(2, false, &[(2, 1, 64, MS), (2, 1, 512, 0)]);
        throughput_only(&mut s);
        let decoder = add(&mut s, 1, 64, 1024);
        ready(&mut s, &[decoder]);
        if prompt_tokens > 0 {
            let prompt = add(&mut s, 2, prompt_tokens, 128);
            initialize(&mut s, prompt);
        }
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Mixed);
        assert_eq!(plan.prefill_capacity(), if prompt_tokens > 64 { 512 } else { 64 });
    }
}

#[test]
fn mixed_value_selects_an_efficiency_knee_and_a_fitting_prompt_tail() {
    for prompt_tokens in [64, 8192] {
        let mut s = mixed_engine(
            2,
            false,
            &[
                (2, 1, 64, 2 * MS),
                (2, 1, 512, 8 * MS),
                (2, 1, 4096, 80 * MS),
            ],
        );
        throughput_only(&mut s);
        let d = add(&mut s, 1, 64, 1024);
        ready(&mut s, &[d]);
        let p = add(&mut s, 2, prompt_tokens, 128);
        initialize(&mut s, p);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Mixed);
        assert_eq!(
            plan.prefill_capacity(),
            if prompt_tokens == 64 { 64 } else { 512 }
        );
        assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, Some(1));
        assert_eq!(plan.lanes[0].request, d);
        assert_eq!(plan.lanes[0].tokens, 1);
        assert_eq!(plan.lanes[1].request, p);
        assert_eq!(plan.lanes[1].tokens, plan.prefill_capacity());
        let expected =
            (plan.lanes[1].tokens as f64 + 64.0 + if prompt_tokens == 64 { 64.0 } else { 0.0 })
                / (plan.estimated_duration_ns() as f64 / 1e6);
        assert!(
            (plan.selection_value_rate() * plan.selection_log_scale().exp() - expected).abs()
                < 1e-10
        );
    }
}

#[test]
fn mixed_shapes_can_win_with_either_phase_unoccupied() {
    let mut s = mixed_engine(2, false, &[(2, 1, 4096, MS)]);
    throughput_only(&mut s);
    let p = add(&mut s, 1, 8192, 1024);
    initialize(&mut s, p);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.lanes.len(), 1);
    assert_eq!(plan.lanes[0].tokens, 4096);
    assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, Some(0));
    ready(&mut s, &[p]);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.lanes.len(), 1);
    assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, None);
    assert_eq!(plan.lane_kind(0), GraphKind::Decode);
}

#[test]
fn mixed_prefill_and_verification_commit_distinct_progress_and_reject_bad_records_atomically() {
    let mut s = mixed_engine(2, true, &[(2, 8, 512, MS)]);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 8192, 128);
    initialize(&mut s, p);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.token_capacity, 8);
    let consumed = s.request(d).unwrap().consumed_tokens;
    let output = s.request(d).unwrap().generated_tokens;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    assert!(s.plan().is_none());
    let records = completion_records(&s, &plan, fence, 0);
    let mut broken = records.clone();
    broken[p.slot as usize].consumed_tokens -= 1;
    assert!(s.complete(fence, &broken, gpu_time(200 * MS)).is_err());
    assert_eq!(s.request(d).unwrap().consumed_tokens, consumed);
    assert_eq!(s.request(p).unwrap().consumed_tokens, 0);
    assert_eq!(s.estimated_duration_ns(&plan), Some(MS));
    broken = records;
    broken[d.slot as usize]
        .vector
        .as_mut()
        .unwrap()
        .verification_width = 1;
    assert!(s.complete(fence, &broken, gpu_time(200 * MS)).is_err());
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(2 * MS));
    assert_eq!(s.request(p).unwrap().consumed_tokens, 512);
    assert_eq!(
        s.request(p).unwrap().accounting.committed_prefill_tokens,
        512
    );
    assert_eq!(s.request(p).unwrap().generated_tokens, 0);
    assert_eq!(s.request(d).unwrap().consumed_tokens, consumed + 1);
    assert_eq!(s.request(d).unwrap().generated_tokens, output + 1);
    assert_eq!(
        s.request(d).unwrap().accounting.committed_prefill_tokens,
        64
    );
    assert_eq!(s.request(d).unwrap().pending, 0);
}

#[test]
fn mixed_prompt_tail_publishes_first_output_and_preserves_its_snapshot_boundary() {
    let mut s = mixed_engine(2, true, &[(2, 8, 64, MS)]);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 64, 128);
    initialize(&mut s, p);
    let output = s.request(d).unwrap().generated_tokens;
    let plan = step(&mut s, 3);
    assert_eq!(plan.kind, GraphKind::Mixed);
    let request = s.request(p).unwrap();
    assert_eq!(request.generated_tokens, 1);
    assert_eq!(request.accounting.published_deliveries, 1);
    assert_eq!(request.phase, RequestPhase::Snapshot);
    assert_eq!(
        request.boundary_capture.unwrap().resume,
        RequestPhase::Decode
    );
    assert_eq!(s.request(d).unwrap().generated_tokens, output + 4);
    ready(&mut s, &[p, d]);
    assert!(s.plan().unwrap().lanes.iter().all(|l| l.tokens <= 8));
}

#[test]
fn mixed_cost_observations_and_expiry_are_specific_to_both_token_capacities() {
    let mut s = mixed_engine(
        2,
        true,
        &[(2, 1, 64, MS), (2, 8, 64, 2 * MS), (2, 8, 512, 3 * MS)],
    );
    let make = |width, prefill| ScheduledStep { packed_token_capacity: 0,
        kind: GraphKind::Mixed,
        batch_capacity: 2,
        token_capacity: width,
        mixed: Some(MixedStep {
            prefill_capacity: prefill,
            prefill_row: None,
            speculative: true,
        }),
        lanes: vec![],
        requires_feedback: true,
    };
    s.advance_to(10 * MS).unwrap();
    s.observe_execution(&make(8, 64), gpu_time(10 * MS));
    assert_eq!(s.estimated_duration_ns(&make(1, 64)), Some(MS));
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(6 * MS));
    assert_eq!(s.estimated_duration_ns(&make(8, 512)), Some(3 * MS));
    s.advance_to(s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(s.estimated_duration_ns(&make(8, 512)), Some(3 * MS));
    assert!(!s.costs.is_fresh(cost::Key::from_step(&make(8, 512))));
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(6 * MS));
    s.advance_to(s.policy.estimate_ttl_ns + 10 * MS).unwrap();
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(6 * MS));
    s.observe_execution(&make(8, 64), gpu_time(MS));
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(MS));
}

#[test]
fn mixed_and_separate_selection_matches_exhaustive_useful_work_per_time() {
    #[derive(Clone, Copy)]
    struct Alternative { kind: u8, mask: u64, prompt: usize, batch: usize, prefill: usize, rate: f64 }
    for seed in 0..16usize {
        let shapes: Vec<_> = [2, 4, 8]
            .into_iter()
            .flat_map(|batch| {
                [64, 512, 4096].into_iter().map(move |p| {
                    (
                        batch,
                        1,
                        p,
                        (3 + (seed * 7 + batch * 3 + p / 64) % 79) as u64 * MS,
                    )
                })
            })
            .collect();
        let mut s = mixed_engine(6, false, &shapes);
        throughput_only(&mut s);
        for slot in 0..6 {
            let prompt = if slot < 2 {
                [64, 129, 320, 900][(seed + slot) % 4]
            } else {
                64
            };
            let h = add(&mut s, slot as u32 + 1, prompt, 1024);
            let r = s.request_mut(h).unwrap();
            if slot < 2 {
                r.phase = RequestPhase::Prefill;
            } else {
                r.phase = RequestPhase::Decode;
                r.consumed_tokens = 64;
                r.scheduled_consumed = 64;
                r.generated_tokens = 1;
            }
        }
        let mut alternatives = Vec::<Alternative>::new();
        let mut consider = |batch: usize, pcap: usize, ns: u64, allow_p: bool, allow_d: bool| {
            let mut best = [None::<Alternative>; 2];
            for mask in 1u64..64 {
                let ps = (mask & 3).count_ones() as usize;
                let ds = (mask & !3).count_ones() as usize;
                if ps > 1 || ps + ds > batch || (!allow_p && ps > 0) || (!allow_d && ds > 0) {
                    continue;
                }
                let mut value = 64.0 * ds as f64;
                let mut prompt_work = 0;
                for slot in 0..2 {
                    if mask & (1 << slot) == 0 {
                        continue;
                    }
                    let prompt = s.slots[slot].request.as_ref().unwrap().prompt_tokens;
                    prompt_work = prompt.min(pcap);
                    value += prompt.min(pcap) as f64 + if prompt <= pcap { 64.0 } else { 0.0 };
                }
                let alternative = Alternative { kind: if allow_p && allow_d { 2 } else { u8::from(allow_p) },
                    mask, prompt: prompt_work, batch, prefill: pcap, rate: value * 1e6 / ns as f64 };
                let entry = &mut best[usize::from(ps != 0)];
                if entry.is_none_or(|old| alternative.rate > old.rate
                    || (alternative.rate == old.rate && (mask.count_ones() > old.mask.count_ones()
                        || (mask.count_ones() == old.mask.count_ones() && mask < old.mask)))) {
                    *entry = Some(alternative);
                }
            }
            alternatives.extend(best.into_iter().flatten());
        };
        for &p in &s.prefill_catalog {
            consider(
                1,
                p,
                s.costs.duration(GraphKind::Prefill, 1, p).unwrap(),
                true,
                false,
            );
        }
        for b in [1, 2, 4, 8] {
            consider(
                b,
                0,
                s.costs.duration(GraphKind::Decode, b, 1).unwrap(),
                false,
                true,
            );
        }
        for &(b, _, p, t) in &shapes {
            consider(b, p, t, true, true);
        }
        let expected = alternatives.iter().filter(|candidate| !alternatives.iter().any(|other| {
            other.kind == candidate.kind && other.mask == candidate.mask && other.prompt == candidate.prompt
                && other.batch <= candidate.batch && other.prefill <= candidate.prefill
                && (other.batch < candidate.batch || other.prefill < candidate.prefill)
        })).map(|candidate| candidate.rate).fold(0.0f64, f64::max);
        let plan = s.plan().unwrap();
        let actual = plan.selection_value_rate() * plan.selection_log_scale().exp();
        assert!(
            (actual - expected).abs() < 1e-9,
            "seed {seed}: {actual} != {expected}; {plan:?}"
        );
        assert_eq!(
            plan.lanes
                .iter()
                .enumerate()
                .filter(|(row, _)| plan.lane_kind(*row) == GraphKind::Prefill)
                .count(),
            usize::from(
                plan.kind == GraphKind::Prefill
                    || plan.mixed.as_ref().is_some_and(|m| m.prefill_row.is_some())
            )
        );
    }
}

#[test]
fn mixed_catalog_completes_common_concurrent_workloads_without_decode_starvation() {
    for (prefills, decoders, accepted) in [
        (2, 0, 0),
        (1, 1, 0),
        (1, 1, 7),
        (1, 4, 3),
        (1, 5, 3),
        (0, 2, 0),
        (0, 2, 7),
        (0, 16, 3),
    ] {
        let active: usize = prefills + decoders;
        let mut shapes = Vec::new();
        for b in crate::execution::BATCH_CAPACITIES
            .into_iter()
            .filter(|b| *b >= 2 && *b <= active.next_power_of_two())
        {
            for p in [64, 512, 4096] {
                for w in [1, 3, 8] {
                    let decode = match w {
                        1 => 15 * MS,
                        3 => 17_300_000,
                        _ => 30 * MS,
                    } + 2 * (b as u64 - 2) * MS;
                    shapes.push((b, w, p, (2 * MS + p as u64 * MS / 64 + decode) * 4 / 5));
                }
            }
        }
        let mut s = mixed_engine(active, true, &shapes);
        let handles: Vec<_> = (0..active)
            .map(|i| {
                add(
                    &mut s,
                    i as u32 + 1,
                    if i < prefills { 32768 } else { 64 },
                    128,
                )
            })
            .collect();
        let mut maximum_gap = 0;
        let mut launches = 0;
        while s.slots.iter().any(|slot| slot.request.is_some()) && launches < 20_000 {
            step(&mut s, accepted);
            launches += 1;
            for &h in &handles {
                if let Ok(r) = s.request(h) {
                    if r.generated_tokens < r.max_new_tokens {
                        if let Some(last) = r.accounting.last_output_ns {
                            maximum_gap = maximum_gap.max(s.now_ns - last);
                        }
                    }
                }
            }
        }
        assert_eq!(
            s.slots.iter().filter(|slot| slot.request.is_some()).count(),
            0,
            "prefills={prefills}, decoders={decoders}, accepted={accepted}"
        );
        assert!(
            maximum_gap < 1000 * MS,
            "prefills={prefills}, decoders={decoders}, accepted={accepted}, gap={maximum_gap}"
        );
    }
}

#[test]
fn mixed_prerequisite_values_a_new_prompt_with_ready_decoders() {
    let mut s = mixed_engine(2, false, &[(2, 1, 512, MS)]);
    throughput_only(&mut s);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 8192, 128);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Init);
    assert_eq!(plan.lanes[0].request, p);
    let expected = (512.0 + 64.0) / 1.1;
    assert!(
        (plan.selection_value_rate() * plan.selection_log_scale().exp() - expected).abs() < 1e-9
    );
    assert_eq!(plan.estimated_duration_ns(), MS / 10);
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.lanes.len(), 2);
}

#[test]
fn mixed_verification_width_reserves_state_even_when_separate_graphs_are_narrower() {
    let mut s = interactive_engine(2, true);
    s.speculative_graphs(&[(1, 1), (2, 1)]).unwrap();
    assert_eq!(s.execution_width(), 1);
    s.mixed_graphs(&[(2, 8, 512)]).unwrap();
    assert_eq!(s.execution_width(), 8);
    assert_eq!(s.execution_rows(), 8);
    assert!(s.fixed_speculative_width(Some(1)).is_err());
    assert!(s.enable_speculation(&[]).is_err());
    let first = add(&mut s, 1, 65, 128);
    let second = add(&mut s, 2, 65, 128);
    assert_eq!(
        s.request(second).unwrap().state_slot.0 - s.request(first).unwrap().state_slot.0,
        8
    );
}

#[test]
fn stale_mixed_catalog_preserves_retained_useful_value_selection() {
    let mut s = mixed_engine(
        4,
        true,
        &[(4, 1, 64, MS), (4, 3, 512, MS), (4, 8, 4096, MS)],
    );
    let decoders: Vec<_> = (1..=3).map(|id| add(&mut s, id, 64, 1024)).collect();
    ready(&mut s, &decoders);
    let p = add(&mut s, 4, 8192, 128);
    initialize(&mut s, p);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(
        (
            plan.batch_capacity,
            plan.prefill_capacity(),
            plan.token_capacity
        ),
        (4, 4096, 8)
    );
    assert_eq!(plan.estimated_duration_ns(), MS);
    assert_eq!(plan.origin, PlanOrigin::Ordinary);
    assert_eq!(plan.lanes.len(), 4);
    assert_eq!(plan.lanes.last().unwrap().tokens, 4096);
}

#[test]
fn mixed_cancellation_keeps_other_rows_live_and_retires_pages_only_after_completion() {
    let mut s = mixed_engine(2, true, &[(2, 8, 512, MS)]);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 8192, 128);
    initialize(&mut s, p);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 3);
    let free = s.free_pages.len();
    let output = s.request(d).unwrap().generated_tokens;
    s.cancel(p).unwrap();
    assert_eq!(s.free_pages.len(), free);
    assert!(
        s.admit(RequestSpec {
            prompt: vec![3; 64],
            message_boundaries: vec![],
            conditioning: vec![],
            max_new_tokens: 128,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .is_err()
    );
    let generated = s.complete(fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(generated.len(), 4);
    assert!(generated.iter().all(|t| t.request == d));
    assert_eq!(s.request(d).unwrap().generated_tokens, output + 4);
    assert!(s.free_pages.len() > free);
    let replacement = add(&mut s, 3, 64, 128);
    assert_eq!(replacement.slot, p.slot);
    assert_ne!(replacement.generation, p.generation);
    assert_eq!(
        s.request(replacement)
            .unwrap()
            .accounting
            .committed_prefill_tokens,
        0
    );
}

#[test]
fn retained_cost_keys_keep_packed_mixed_and_ordinary_age_independent() {
    let base = super::tests::duration_graphs(2)[0].clone();
    let keys = [
        cost::Key { kind: GraphKind::Decode, batch: 2, tokens: 1, prefill: 0, packed: 0 },
        cost::Key { kind: GraphKind::Mixed, batch: 2, tokens: 8, prefill: 64, packed: 0 },
        cost::Key { kind: GraphKind::Verification, batch: 2, tokens: 8, prefill: 0, packed: 10 },
    ];
    let graphs: Vec<_> = keys.iter().enumerate().map(|(index, key)| crate::execution::GraphDescriptor {
        name: index.to_string(), kind: key.kind, batch_capacity: key.batch, token_capacity: key.tokens,
        prefill_capacity: key.prefill, packed_token_capacity: key.packed, ..base.clone()
    }).collect();
    let mut costs = cost::ActionCosts::prepared(&graphs).unwrap();
    let make = |key: cost::Key| ScheduledStep { kind: key.kind, batch_capacity: key.batch,
        token_capacity: key.tokens, packed_token_capacity: key.packed,
        mixed: (key.kind == GraphKind::Mixed).then_some(MixedStep {
            prefill_capacity: key.prefill, prefill_row: None, speculative: true,
        }), lanes: vec![], requires_feedback: true };
    for &key in &keys {
        let mut timing = gpu_time(8 * MS);
        timing.wall_ns = 12 * MS;
        costs.observe(&make(key), timing);
    }
    costs.advance_to(30_000 * MS, 30_000 * MS);
    assert_eq!(costs.stale_keys().count(), 3);
    for &key in &keys {
        assert_eq!(costs.duration_key(key), Some(12 * MS));
        assert!(!costs.is_fresh(key));
        assert_eq!(costs.observation(key), Some((12 * MS, Some(0), false)));
    }
    assert_eq!(costs.packed_shapes().next().unwrap().nanos, 12 * MS);
    costs.record_attempt(keys[2]);
    assert_eq!(costs.refresh_order(keys[2]), 30_000 * MS);
    assert_eq!(costs.refresh_order(keys[0]), 0);
    let mut timing = gpu_time(2 * MS);
    timing.wall_ns = 5 * MS;
    costs.observe(&make(keys[1]), timing);
    assert_eq!(costs.duration_key(keys[1]), Some(5 * MS));
    assert!(costs.is_fresh(keys[1]));
    assert_eq!(costs.duration_key(keys[2]), Some(12 * MS));
    assert_eq!(costs.stale_keys().count(), 2);
}

#[test]
fn refresh_skips_infeasible_ages_and_resolves_the_oldest_feasible_utility_group() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[h]);
    let key = |width| cost::Key { kind: GraphKind::Speculative, batch: 1,
        tokens: width, prefill: 0, packed: 0 };
    let infeasible = cost::Key { kind: GraphKind::Prefill, batch: 1,
        tokens: 64, prefill: 0, packed: 0 };
    let base = s.now_ns;
    s.advance_to(base + MS).unwrap();
    for width in [2, 4] { set_cost(&mut s, GraphKind::Speculative, 1, width, 100 * MS); }
    s.advance_to(base + 2 * MS).unwrap();
    set_cost(&mut s, GraphKind::Speculative, 1, 1, MS);
    s.advance_to(base + s.policy.estimate_ttl_ns + 2 * MS).unwrap();
    let others: Vec<_> = s.costs.stale_keys()
        .filter(|candidate| ![infeasible, key(1), key(2), key(4)].contains(candidate))
        .map(|candidate| (candidate, s.costs.duration_key(candidate).unwrap())).collect();
    for (candidate, nanos) in others {
        set_cost(&mut s, candidate.kind, candidate.batch, candidate.tokens, nanos);
    }
    set_cost(&mut s, GraphKind::Speculative, 1, 8, MS / 10);
    assert_eq!(s.plan().unwrap().token_capacity, 8);
    assert!(s.costs.refresh_order(infeasible) < s.costs.refresh_order(key(2)));
    assert!(s.costs.refresh_order(key(2)) < s.costs.refresh_order(key(1)));
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::RefreshStart(key(4)));
    s.costs.record_attempt(key(4));
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::RefreshStart(key(2)));
}

#[test]
fn refresh_commit_rechecks_credit_and_freezes_physical_cost_context() {
    let mut s = engine(2, false);
    let a = add(&mut s, 1, 64, 4096);
    let b = add(&mut s, 2, 64, 4096);
    ready(&mut s, &[a, b]);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let mut first = s.phase_plan(GraphKind::Decode, Some(a));
    let key = cost::Key::from_step(&first);
    first.origin = PlanOrigin::RefreshStart(key);
    let mut saved = s.phase_plan(GraphKind::Decode, Some(b));
    saved.origin = PlanOrigin::RefreshStart(key);
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    let context = s.submitted_timing(fence).unwrap();
    assert_eq!(context.retained_ns, MS);
    assert_eq!(context.freshness, CostFreshness::Stale);
    assert_eq!(context.origin, TimingOrigin::Refresh);
    assert_eq!(context.credit_ns, refresh::CREDIT_CAP_NS);
    assert!(s.submitted_step(fence).unwrap().requires_feedback);
    let mut timing = gpu_time(8 * MS);
    timing.gpu_ns[15] = Some(9 * MS);
    timing.wall_ns = 12 * MS;
    s.advance_to(s.now_ns + timing.wall_ns).unwrap();
    complete_timed(&mut s, &first, fence, 0, false, timing);
    assert_eq!(s.refresh_credit_ns(), -7_000_000.0);
    assert!(s.refresh_episode().is_none());
    let order = s.costs.refresh_order(key);
    let pending = s.request(b).unwrap().pending;
    assert!(s.commit(&saved, Fence(fence.0 + 1)).is_err());
    assert_eq!(s.costs.refresh_order(key), order);
    assert_eq!(s.request(b).unwrap().pending, pending);
    assert_eq!(s.refresh_credit_ns(), -7_000_000.0);
    assert_eq!(s.submitted, fence);
}

#[test]
fn invalid_refresh_timing_pays_isolated_wall_time_once_without_refreshing_cost() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 1, 4096);
    ready(&mut s, &[h]);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let mut plan = s.phase_plan(GraphKind::Decode, Some(h));
    let key = cost::Key::from_step(&plan);
    plan.origin = PlanOrigin::RefreshStart(key);
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 0);
    let before = s.costs.observation(key);
    let timing = ExecutionTiming { wall_ns: 100 * MS, queued: true, ..Default::default() };
    assert!(s.complete(fence, &records, timing).is_err());
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);
    assert!(s.submitted_step(fence).is_ok());
    s.advance_to(s.now_ns + timing.wall_ns).unwrap();
    s.complete(fence, &records, ExecutionTiming { queued: false, ..timing }).unwrap();
    assert_eq!(s.refresh_credit_ns(), -95_000_000.0);
    assert_eq!(s.costs.observation(key), before);
    assert!(s.complete(fence, &records, timing).is_err());
    assert_eq!(s.refresh_credit_ns(), -95_000_000.0);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), -95_000_000.0);
}

#[test]
fn ordinary_income_excludes_queued_wall_wait_and_idle_discards_only_positive_credit() {
    let mut s = engine(2, false);
    let a = add(&mut s, 1, 64, 4096);
    let b = add(&mut s, 2, 64, 4096);
    ready(&mut s, &[a, b]);
    s.refresh.credit_ns = 0.0;
    let first = s.phase_plan(GraphKind::Decode, Some(a));
    let second = s.phase_plan(GraphKind::Decode, Some(b));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    s.commit(&second, Fence(fence.0 + 1)).unwrap();
    assert!(s.idle().is_err());
    let frozen = s.submitted_timing(Fence(fence.0 + 1)).unwrap();
    let mut timing = gpu_time(10 * MS);
    timing.wall_ns = 50 * MS;
    complete_timed(&mut s, &first, fence, 0, false, timing);
    assert_eq!(s.submitted_timing(Fence(fence.0 + 1)).unwrap().retained_ns, frozen.retained_ns);
    timing.gpu_ns[15] = Some(20 * MS);
    timing.wall_ns = 200 * MS;
    timing.queued = true;
    complete_timed(&mut s, &second, Fence(fence.0 + 1), 0, false, timing);
    assert!((s.refresh_credit_ns() - 70_000_000.0 * 0.002 / 0.998).abs() < 1e-8);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), 0.0);
}

#[test]
fn saved_fresh_plans_gain_a_physical_staleness_feedback_barrier_at_commit() {
    let mut s = engine(2, false);
    let a = add(&mut s, 1, 64, 4096);
    let b = add(&mut s, 2, 64, 4096);
    ready(&mut s, &[a, b]);
    let first = s.phase_plan(GraphKind::Decode, Some(a));
    let second = s.phase_plan(GraphKind::Decode, Some(b));
    assert!(!first.requires_feedback);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert!(s.plan_requires_feedback(&first));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    assert!(s.submitted_step(fence).unwrap().requires_feedback);
    assert_eq!(s.submitted_timing(fence).unwrap().freshness, CostFreshness::Fresh);
    let order = s.costs.refresh_order(cost::Key::from_step(&second));
    let pending = s.request(b).unwrap().pending;
    assert!(s.commit(&second, Fence(fence.0 + 1)).is_err());
    assert_eq!(s.request(b).unwrap().pending, pending);
    assert_eq!(s.costs.refresh_order(cost::Key::from_step(&second)), order);
}

#[test]
fn proposal_episode_owns_generations_and_pays_all_verification_captures_in_debt() {
    let h = |slot, generation| RequestHandle { slot, generation };
    let step = |kind, handles: &[RequestHandle]| ScheduledStep { kind, batch_capacity: 2,
        token_capacity: 8, packed_token_capacity: 0, mixed: None, requires_feedback: true,
        lanes: handles.iter().map(|&request| ScheduledLane { request, start_position: 65, tokens: 8 }).collect() };
    let p = step(GraphKind::Proposal, &[h(0, 1), h(1, 1)]);
    let target = cost::Key { kind: GraphKind::Verification, batch: 2, tokens: 8, prefill: 0, packed: 0 };
    let start = PlanOrigin::RefreshStart(target);
    let mut budget = refresh::RefreshState::default();
    budget.credit_ns = refresh::CREDIT_CAP_NS;
    budget.validate(start).unwrap();
    budget.commit(start, Fence(7), &p);
    assert!(budget.episode().unwrap().owns(h(0, 1), 65, Fence(7)));
    assert!(!budget.episode().unwrap().owns(h(0, 2), 65, Fence(7)));
    assert!(!budget.episode().unwrap().owns(h(0, 1), 66, Fence(7)));
    assert!(!budget.episode().unwrap().owns(h(0, 1), 65, Fence(8)));
    budget.complete(start, gpu_time(10 * MS));
    assert!(!budget.ready());
    let continuation = PlanOrigin::RefreshContinuation(7);
    for (slot, duration) in [(0, 20 * MS), (1, 30 * MS)] {
        budget.validate(continuation).unwrap();
        budget.commit(continuation, Fence(8 + slot as u64), &step(GraphKind::Verification, &[h(slot, 1)]));
        budget.complete(continuation, gpu_time(duration));
    }
    assert!(budget.episode().is_none());
    assert_eq!(budget.credit_ns, -55_000_000.0);
    assert!(!budget.ready());
    budget.idle();
    assert_eq!(budget.credit_ns, -55_000_000.0);
    budget.abort();
    assert_eq!(budget.credit_ns, -55_000_000.0);
    budget.credit_ns = refresh::CREDIT_CAP_NS;
    budget.commit(start, Fence(10), &p);
    budget.abort();
    assert_eq!(budget.credit_ns, 0.0);
    assert!(budget.episode().is_none());
}

#[test]
fn initial_unknown_graph_completions_neither_earn_nor_spend_refresh_credit() {
    let mut s = cold_engine(1, false);
    add(&mut s, 1, 64, 64);
    for _ in 0..3 {
        let plan = measured_step(&mut s, 10 * MS, 0);
        assert_eq!(plan.origin, PlanOrigin::Discovery);
        assert_eq!(s.refresh_credit_ns(), 0.0);
        assert!(s.refresh_episode().is_none());
    }
}

#[test]
fn refresh_funding_requires_earned_credit_and_accounts_for_repeated_attempts_and_idle() {
    let mut s = engine(2, false);
    let h = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[h]);
    set_cost(&mut s, GraphKind::Decode, 1, 1, 10 * MS);
    set_cost(&mut s, GraphKind::Decode, 2, 1, 100 * MS);
    s.idle().unwrap();
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let ordinary = |s: &mut RequestScheduler, duration| {
        let plan = measured_step(s, duration, 0);
        assert_eq!(plan.origin, PlanOrigin::Ordinary);
        assert_eq!((plan.kind, plan.batch_capacity), (GraphKind::Decode, 1));
    };
    ordinary(&mut s, 10 * MS);
    assert!(s.refresh_credit_ns() > 0.0);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), 0.0);

    let mut ordinary_ns = 10 * MS;
    for _ in 0..249 {
        ordinary(&mut s, 10 * MS);
        ordinary_ns += 10 * MS;
    }
    ordinary(&mut s, 4 * MS);
    ordinary_ns += 4 * MS;
    assert!(s.refresh_credit_ns() < refresh::CREDIT_CAP_NS);
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::Ordinary);
    ordinary(&mut s, MS);
    ordinary_ns += MS;
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);

    let first = s.plan().unwrap();
    let key = cost::Key::from_step(&first);
    assert_eq!(first.origin, PlanOrigin::RefreshStart(key));
    assert_eq!((first.kind, first.batch_capacity), (GraphKind::Decode, 2));
    let before = s.costs.observation(key);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    s.advance_to(s.now_ns + 8 * MS).unwrap();
    complete_timed(&mut s, &first, fence, 0, false,
        ExecutionTiming { wall_ns: 8 * MS, ..Default::default() });
    assert_eq!(s.costs.observation(key), before);
    assert_eq!(s.refresh_credit_ns(), -3_000_000.0);
    assert!(s.refresh_episode().is_none());
    let mut refresh_ns = 8 * MS;
    assert!(refresh_ns as f64 <= ordinary_ns as f64 * 0.002 / 0.998 + (8 * MS) as f64);
    s.idle().unwrap();
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(s.refresh_credit_ns(), -3_000_000.0);

    for _ in 0..399 {
        ordinary(&mut s, 10 * MS);
        ordinary_ns += 10 * MS;
    }
    assert!(s.refresh_credit_ns() < refresh::CREDIT_CAP_NS);
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::Ordinary);
    ordinary(&mut s, 2 * MS);
    ordinary_ns += 2 * MS;
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);
    let second = measured_step(&mut s, 20 * MS, 0);
    assert_eq!(second.origin, PlanOrigin::RefreshStart(key));
    refresh_ns += 20 * MS;
    assert_eq!(s.refresh_credit_ns(), -15_000_000.0);
    assert!(s.refresh_episode().is_none());
    assert!(refresh_ns as f64 <= ordinary_ns as f64 * 0.002 / 0.998 + (20 * MS) as f64);
    assert!((s.refresh_credit_ns()
        - ((ordinary_ns - 10 * MS) as f64 * 0.002 / 0.998 - refresh_ns as f64)).abs() < 1e-6);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), -15_000_000.0);
}

#[test]
fn cancelling_all_inflight_proposal_owners_keeps_the_charge_and_releases_reused_slots() {
    let mut s = engine(2, false);
    s.enable_speculation(&[1, 2]).unwrap();
    let mut graphs = super::tests::duration_graphs(2);
    graphs.retain(|graph| graph.kind != GraphKind::Speculative || graph.token_capacity == 1);
    let base = graphs[0].clone();
    for batch in [1, 2] {
        for (kind, width) in [(GraphKind::Proposal, 2),
            (GraphKind::Verification, 1), (GraphKind::Verification, 2)] {
            graphs.push(crate::execution::GraphDescriptor { name: format!("{kind:?}-{batch}-{width}"),
                kind, batch_capacity: batch, token_capacity: width,
                estimated_duration_ns: 100 * MS, ..base.clone() });
        }
    }
    s.speculative_graphs(&[(1, 1), (1, 2), (2, 1), (2, 2)]).unwrap();
    s.measured_graph_durations(&graphs).unwrap();
    s.split_speculation(2).unwrap();
    let owners = [add(&mut s, 1, 64, 4096), add(&mut s, 2, 64, 4096)];
    ready(&mut s, &owners);
    let target = cost::Key { kind: GraphKind::Proposal, batch: 2, tokens: 2, prefill: 0, packed: 0 };
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let others: Vec<_> = s.costs.stale_keys().filter(|key| *key != target)
        .map(|key| (key, s.costs.duration_key(key).unwrap())).collect();
    for (key, nanos) in others {
        set_cost(&mut s, key.kind, key.batch, key.tokens, nanos);
    }
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let plan = s.plan().unwrap();
    assert_eq!(plan.origin, PlanOrigin::RefreshStart(target));
    assert_eq!(plan.lanes.len(), 2);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    for owner in owners { s.cancel(owner).unwrap(); }
    assert_eq!(s.refresh_episode().unwrap().id, fence.0);
    assert_eq!(s.refresh_episode().unwrap().owned_mask(), 0);
    assert!(s.admit(super::tests::spec(64, 4096)).is_err());
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);
    assert!(s.idle().is_err());

    let records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    s.advance_to(s.now_ns + 8 * MS).unwrap();
    assert!(s.complete_proposal(fence, &records, gpu_time(8 * MS), &[[1.0; 7]; 2]).unwrap().is_empty());
    assert!(s.refresh_episode().is_none());
    assert_eq!(s.refresh_credit_ns(), -3_000_000.0);
    let replacements = [add(&mut s, 3, 64, 4096), add(&mut s, 4, 64, 4096)];
    for (owner, replacement) in owners.into_iter().zip(replacements) {
        assert_eq!(replacement.slot, owner.slot);
        assert_ne!(replacement.generation, owner.generation);
        assert!(!s.owns_proposal(s.request(replacement).unwrap()));
        assert_eq!(s.request(replacement).unwrap().pending, 0);
    }
    assert!(s.split.as_ref().unwrap().retained.iter().all(Option::is_none));
    assert!(s.complete_proposal(fence, &records, gpu_time(8 * MS), &[[1.0; 7]; 2]).is_err());
    assert_eq!(s.refresh_credit_ns(), -3_000_000.0);
    assert!(!matches!(s.plan().unwrap().origin, PlanOrigin::RefreshContinuation(_)));
}
