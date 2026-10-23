use super::contract::*;
use super::testing::manifest;
use super::*;
use crate::{
    cache::Fence,
    execution::{CompletionRecord, GraphKind},
    scheduler::{RequestScheduler, RequestSpec, SchedulerLimits},
};

#[test]
fn prompt_scoring_recognizes_features_without_protected_image_spans() {
    let data = PromptData { positions: vec![], vision_token_indices: vec![-1, 0, 1, -1, -1, -1],
        spans: vec![4..5] };
    assert!(!data.is_image_position(0));
    assert!(data.is_image_position(1));
    assert!(data.is_image_position(2));
    assert!(!data.is_image_position(3));
    assert!(data.is_image_position(4));
    assert!(!data.is_image_position(5));
    assert!(!data.is_image_position(6));
}

pub(super) fn config() -> ModelConfig {
    ModelConfig {
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 2048,
        contract: super::testing::metadata(),
        identity: [1; 32],
        model_id: "qwen38".into(),
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 2048,
            prefill_chunk_size: 512,
            pages: 64,
            snapshots: 40,
        },
        eos_token_ids: vec![248046, 248044],
        sampling_ranges: Default::default(),
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 256,
        speculative_widths: Vec::new(),
        speculative_fixed_width: None,
        speculative_token_capacity: 512,
        decode_weight_bytes: 0,
        history_drafts: true,
        speculative_proposal_source: Default::default(),
    }
}

#[test]
fn admission_histogram_seed_and_partial_page_uploads_preserve_native_semantics() {
    let config = config();
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&self::manifest(&config).graphs)
        .unwrap();
    let handle = scheduler
        .admit(RequestSpec {
            message_boundaries: vec![],
            prompt: vec![7; 65],
            conditioning: vec![],
            max_new_tokens: 4,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[handle.slot as usize] = Some(Admission {
        handle,
        sampling: SamplingParameters {
            seed: u64::MAX,
            temperature: 1.,
            ..Default::default()
        },
        max_new_tokens: 4,
        prompt: vec![7; 65],
        data: PromptData {
            positions: (0..65).map(|p| [p; 3]).collect(),
            vision_token_indices: vec![-1; 65],
            ..Default::default()
        },
        grammar_mask: None,
        ignore_eos: false,
        prompt_logprobs: false,
    });
    admissions[0].as_ref().unwrap().validate(&config).unwrap();
    let mut inputs = Inputs::new(&config, &manifest(&config)).unwrap();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(1)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    assert_state_bindings(&inputs, &scheduler, &step);
    let uploads = inputs.uploads();
    let seed = uploads
        .iter()
        .find(|u| u.buffer == "admission_seed_bits")
        .unwrap();
    assert_eq!(seed.bytes, &[255; 8]);
    let histogram = uploads
        .iter()
        .find(|u| u.buffer == "admission_histogram")
        .unwrap();
    assert_eq!(
        i32::from_le_bytes(histogram.bytes[28..32].try_into().unwrap()),
        65
    );
    for ignore_eos in [false, true, false] {
        admissions[0].as_mut().unwrap().ignore_eos = ignore_eos;
        crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
        assert_state_bindings(&inputs, &scheduler, &step);
        let uploads = inputs.uploads();
        let count = uploads
            .iter()
            .find(|upload| upload.buffer == "admission_eos_count")
            .unwrap();
        assert_eq!(
            i32::from_le_bytes(count.bytes.try_into().unwrap()),
            if ignore_eos { 0 } else { 2 }
        );
        let ids = uploads
            .iter()
            .find(|upload| upload.buffer == "admission_eos_token_ids")
            .unwrap();
        assert_eq!(
            i64::from_le_bytes(ids.bytes[..8].try_into().unwrap()),
            248046
        );
        assert_eq!(config.eos_token_ids, [248046, 248044]);
    }
    let mut initialized = [CompletionRecord::default(); 4];
    initialized[0] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 1,
        status: 1,
        token_id: -1,
        ..Default::default()
    };
    scheduler
        .complete(Fence(1), &initialized, Default::default())
        .unwrap();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    assert_state_bindings(&inputs, &scheduler, &step);
    assert_eq!(
        inputs
            .uploads()
            .iter()
            .find(|u| u.buffer == "page_tables")
            .unwrap()
            .bytes
            .len(),
        8
    );
    let mut records = [CompletionRecord::default(); 4];
    records[0] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 2,
        consumed_tokens: 65,
        generated_tokens: 1,
        token_id: 9,
        status: 1,
        ..Default::default()
    };
    scheduler
        .complete(Fence(2), &records, Default::default())
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(4)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    assert_state_bindings(&inputs, &scheduler, &step);
    assert!(!inputs.uploads().iter().any(|u| u.buffer == "page_tables"));
    assert!(
        !inputs
            .uploads()
            .iter()
            .any(|u| u.buffer == "token_ids" || u.buffer == "admission_histogram")
    );
}

#[test]
fn large_prefill_uploads_preserve_token_position_and_image_indices_without_tail_padding() {
    let mut config = config();
    let image_contract: PreparedContract =
        serde_json::from_str(include_str!("../../tests/support/contracts/qwen38.json")).unwrap();
    config.contract.vision = image_contract.vision;
    config.contract.frontend = image_contract.frontend;
    config.limits.max_sequence_tokens = 8192;
    config.prepared_sequence_tokens = 8192;
    config.limits.prefill_chunk_size = 4096;
    config.limits.pages = 128;
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&self::manifest(&config).graphs)
        .unwrap();
    let prompt: Vec<u32> = (0..4113).collect();
    let handle = scheduler
        .admit(RequestSpec {
            message_boundaries: vec![],
            prompt: prompt.clone(),
            conditioning: vec![],
            max_new_tokens: 4,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[handle.slot as usize] = Some(Admission {
        handle,
        sampling: SamplingParameters::default(),
        max_new_tokens: 4,
        data: PromptData {
            positions: prompt.iter().map(|&token| [token as i64; 3]).collect(),
            vision_token_indices: prompt
                .iter()
                .map(|&token| if token % 2 == 0 { token as i32 } else { -1 })
                .collect(),
            ..Default::default()
        },
        prompt,
        grammar_mask: None,
        ignore_eos: false,
        prompt_logprobs: false,
    });
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(1)).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 1,
        status: 1,
        token_id: -1,
        ..Default::default()
    };
    scheduler
        .complete(Fence(1), &records, Default::default())
        .unwrap();
    let mut inputs = Inputs::new(&config, &manifest(&config)).unwrap();
    for (sequence, start, count, capacity) in [(2, 0, 4096, 4096), (3, 4096, 17, 4096)] {
        let step = scheduler.plan().unwrap();
        assert_eq!(
            (step.lanes[0].tokens, step.token_capacity),
            (count, capacity)
        );
        scheduler.commit(&step, Fence(sequence)).unwrap();
        crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
        assert_state_bindings(&inputs, &scheduler, &step);
        let uploads = inputs.uploads();
        let bytes = |name| {
            uploads
                .iter()
                .find(|upload| upload.buffer == name)
                .unwrap()
                .bytes
        };
        let tokens: Vec<u8> = (start..start + count)
            .flat_map(|id| (id as i64).to_le_bytes())
            .collect();
        let positions: Vec<u8> = (start..start + count)
            .flat_map(|id| [id as i64; 3])
            .flat_map(i64::to_le_bytes)
            .collect();
        let image_indices: Vec<u8> = (start..start + count)
            .map(|id| if id % 2 == 0 { id as i32 } else { -1 })
            .flat_map(i32::to_le_bytes)
            .collect();
        assert_eq!(bytes("token_ids"), tokens);
        assert_eq!(bytes("prefill_positions"), positions);
        assert_eq!(bytes("vision_token_indices"), image_indices);
        assert_eq!(bytes("vision_feature_slot"), (-1i32).to_le_bytes());
        records[handle.slot as usize] = CompletionRecord {
            generation: handle.generation as i64,
            submission_sequence: sequence as i64,
            consumed_tokens: (start + count) as i64,
            generated_tokens: i64::from(sequence == 3),
            token_id: if sequence == 3 { 77 } else { -1 },
            status: 1,
            ..Default::default()
        };
        scheduler
            .complete(Fence(sequence), &records, Default::default())
            .unwrap();
    }
}

#[test]
fn admission_rejects_nonfinite_parameters_and_inconsistent_positions() {
    assert!(
        SamplingParameters {
            temperature: f32::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        SamplingParameters {
            top_p: f32::from_bits(1),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        SamplingParameters {
            temperature: 1.,
            top_k: -1,
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
}

fn assert_state_bindings(
    inputs: &Inputs,
    scheduler: &RequestScheduler,
    step: &crate::scheduler::ScheduledStep,
) {
    let uploads = inputs.uploads();
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        let binding = uploads
            .iter()
            .find(|upload| {
                upload.buffer == "request_state_rows"
                    && upload.offset == lane.request.slot as usize * 64
            })
            .unwrap();
        let rows: Vec<_> = binding
            .bytes
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[0], request.state_slot.0 as i64);
        for column in 0..scheduler.execution_rows() {
            assert_eq!(rows[column], request.state_slot.0 as i64 + column as i64);
        }
        assert!(
            rows[scheduler.execution_rows()..]
                .iter()
                .all(|row| *row == 0)
        );
    }
}

fn retain_prompt(s: &mut RequestScheduler, fence: u64) {
    let step = s.plan().unwrap();
    assert_eq!(step.kind, GraphKind::Snapshot);
    s.commit(&step, Fence(fence)).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    for lane in &step.lanes {
        let request = s.request(lane.request).unwrap();
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence as i64,
            consumed_tokens: request.consumed_tokens as i64,
            generated_tokens: request.generated_tokens as i64,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
    }
    s.complete(Fence(fence), &records, Default::default())
        .unwrap();
}

#[test]
fn completion_frontiers_follow_declared_history_cursor() {
    let config = config();
    let mut contract = config
        .contract
        .graphs
        .iter()
        .find(|g| g.kind == GraphOperation::Speculative)
        .unwrap()
        .clone();
    let mut record = CompletionRecord {
        consumed_tokens: 65,
        token_id: 7,
        status: 1,
        vector: Some(crate::execution::TokenVector {
            emitted_count: 1,
            tokens: [7, -1, -1, -1, -1, -1, -1, -1],
            draft_cache_consumed: 64,
            ..Default::default()
        }),
        ..Default::default()
    };
    validate_completion(&config, &contract, GraphKind::Speculative, &record).unwrap();
    record.vector.as_mut().unwrap().draft_cache_consumed = 65;
    assert!(validate_completion(&config, &contract, GraphKind::Speculative, &record).is_err());
    contract.draft_cursor.as_mut().unwrap().lag_tokens = 0;
    validate_completion(&config, &contract, GraphKind::Speculative, &record).unwrap();
    contract.draft_cursor = None;
    assert!(validate_completion(&config, &contract, GraphKind::Speculative, &record).is_err());
    record.vector.as_mut().unwrap().draft_cache_consumed = -1;
    validate_completion(&config, &contract, GraphKind::Speculative, &record).unwrap();
    record.vector.as_mut().unwrap().tokens[0] = config.contract.vocabulary as i64;
    assert!(validate_completion(&config, &contract, GraphKind::Speculative, &record).is_err());
}

#[test]
fn invalid_sources_and_storage_views_reject_before_packing() {
    let config = config();
    let manifest = manifest(&config);
    for mutate in [
        (|c: &mut PreparedContract| c.graphs[0].inputs[0].source = ValueSource::PendingToken)
            as fn(&mut PreparedContract),
        |c| c.graphs[0].inputs[0].indexing = InputIndexing::Scalar,
        |c| c.graphs[0].state[0].view.nbytes = 33,
        |c| c.graphs[0].state[0].view.stride = 64,
        |c| c.graphs[0].state_rows_per_request = 0,
        |c| c.graphs[0].kind = GraphOperation::Vision,
        |c| c.graphs[0].inputs[0].buffer = "absent".into(),
        |c| c.cache.page_tables.clear(),
        |c| c.graphs[0].state.clear(),
    ] {
        let mut bad = config.clone();
        mutate(&mut bad.contract);
        assert!(Inputs::new(&bad, &manifest).is_err());
    }
    let mut missing_producer = config.clone();
    for graph in &mut missing_producer.contract.graphs {
        if graph.kind == GraphOperation::Init {
            graph.state.retain(|s| s.view.buffer != "pending");
        }
    }
    assert!(Inputs::new(&missing_producer, &manifest).is_err());
    for strides in [vec![], vec![usize::MAX]] {
        let mut malformed = manifest.clone();
        malformed
            .buffers
            .iter_mut()
            .find(|buffer| buffer.name == "pending")
            .unwrap()
            .strides = strides;
        assert!(Inputs::new(&config, &malformed).is_err());
    }
}

#[cfg(feature = "server")]
#[test]
fn enum_values_and_unknown_contract_fields_fail_closed() {
    let contract = testing::metadata();
    let original = serde_json::to_value(&contract).unwrap();
    for value in [0, -1, 999] {
        let mut invalid = original.clone();
        invalid["graphs"][0]["inputs"][0]["source"] = value.into();
        assert!(serde_json::from_value::<PreparedContract>(invalid).is_err());
        let mut invalid = original.clone();
        invalid["graphs"][0]["kind"] = value.into();
        assert!(serde_json::from_value::<PreparedContract>(invalid).is_err());
    }
    let mut invalid = original;
    invalid["supports_page_growth"] = true.into();
    assert!(serde_json::from_value::<PreparedContract>(invalid).is_err());
}

#[test]
fn host_feedback_is_derived_from_inputs_without_model_identity() {
    let mut config = config();
    assert!(scheduler_capabilities(&config).completion_inputs.is_empty());
    for graph in &mut config.contract.graphs {
        if graph.kind == GraphOperation::Decode {
            graph.inputs.push(testing::binding(
                "sampled_ids",
                ValueSource::PendingToken,
                InputIndexing::SubmissionRow,
                0,
                0,
            ));
            graph.pending_token = None;
        }
    }
    assert_eq!(
        scheduler_capabilities(&config).completion_inputs,
        ["sampled_ids"]
    );
    config.model_id = "unregistered-family".into();
    assert_eq!(
        scheduler_capabilities(&config).completion_inputs,
        ["sampled_ids"]
    );
}

#[test]
fn packing_follows_value_bindings_after_buffers_and_model_are_renamed() {
    let config = config();
    let manifest = manifest(&config);
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&manifest.graphs)
        .unwrap();
    let handle = scheduler
        .admit(RequestSpec {
            message_boundaries: vec![],
            prompt: vec![7; 65],
            conditioning: vec![],
            max_new_tokens: 4,
            grammar_constrained: true,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[handle.slot as usize] = Some(Admission {
        handle,
        sampling: SamplingParameters {
            seed: u64::MAX,
            ..Default::default()
        },
        max_new_tokens: 4,
        prompt: vec![7; 65],
        data: PromptData {
            positions: (0..65).map(|p| [p; 3]).collect(),
            ..Default::default()
        },
        grammar_mask: Some(vec![i32::MIN; config.contract.vocabulary.div_ceil(32)]),
        ignore_eos: false,
        prompt_logprobs: false,
    });
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(1)).unwrap();
    let mut original = Inputs::new(&config, &manifest).unwrap();
    pack_inputs(&mut original, &config, &scheduler, &step, &admissions).unwrap();
    let rename = |name: &str| format!("prepared_{name}");
    let mut changed = config.clone();
    changed.model_id = "independent-recipe".into();
    for graph in &mut changed.contract.graphs {
        for input in &mut graph.inputs {
            input.buffer = rename(&input.buffer);
        }
        for state in &mut graph.state {
            state.view.buffer = rename(&state.view.buffer);
        }
        if let Some(cursor) = &mut graph.draft_cursor {
            cursor.state.buffer = rename(&cursor.state.buffer);
        }
        if let Some(pending) = &mut graph.pending_token {
            pending.buffer = rename(&pending.buffer);
        }
    }
    for view in &mut changed.contract.cache.page_tables {
        view.buffer = rename(&view.buffer);
    }
    for region in &mut changed.contract.cache.regions {
        region.storage.buffer = rename(&region.storage.buffer);
    }
    let mut changed_manifest = manifest.clone();
    let reserved = |name: &str| {
        matches!(
            name,
            "frame_index" | "submission_sequence" | "completion_records"
        )
    };
    for buffer in &mut changed_manifest.buffers {
        if !reserved(&buffer.name) {
            buffer.name = rename(&buffer.name);
        }
    }
    for graph in &mut changed_manifest.graphs {
        for binding in &mut graph.input_bindings {
            if !reserved(&binding.buffer_name) {
                binding.buffer_name = rename(&binding.buffer_name);
            }
        }
    }
    let mut renamed = Inputs::new(&changed, &changed_manifest).unwrap();
    pack_inputs(&mut renamed, &changed, &scheduler, &step, &admissions).unwrap();
    let expected: Vec<_> = original
        .uploads()
        .iter()
        .map(|u| (rename(u.buffer), u.offset, u.bytes.to_vec()))
        .collect();
    let actual: Vec<_> = renamed
        .uploads()
        .iter()
        .map(|u| (u.buffer.to_owned(), u.offset, u.bytes.to_vec()))
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn copied_proposals_require_concrete_graph_inputs() {
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    let mut manifest = manifest(&config);
    manifest
        .graphs
        .retain(|graph| graph.kind != GraphKind::Prefill || graph.batch_capacity == 1);
    for (index, graph) in manifest.graphs.iter_mut().enumerate() {
        graph.handle = index + 1;
    }
    let error = catalog::validate(&config, &manifest)
        .unwrap_err()
        .to_string();
    assert!(error.contains("copied proposals require"), "{error}");
    config.history_drafts = false;
    config.speculative_proposal_source = crate::scheduler::speculation::ProposalSource::Copy;
    let error = catalog::validate(&config, &manifest)
        .unwrap_err()
        .to_string();
    assert!(error.contains("copied proposals require"), "{error}");
}

#[test]
fn copy_staging_uses_physical_slots_stochastic_defaults_and_bonus_mask() {
    use crate::scheduler::speculation::{CopyProposal, ProposalSource};
    let mut config = config();
    config.speculative_widths = (1..=8).collect();
    config.speculative_fixed_width = Some(8);
    config.speculative_proposal_source = ProposalSource::Copy;
    add_copy_inputs(&mut config);
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&self::manifest(&config).graphs)
        .unwrap();
    scheduler
        .enable_speculation(&config.speculative_widths)
        .unwrap();
    scheduler.proposal_source(ProposalSource::Copy).unwrap();
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let handles = (0..3)
        .map(|slot| {
            scheduler
                .admit(RequestSpec {
                    message_boundaries: vec![],
                    prompt: vec![7; 65],
                    conditioning: vec![],
                    max_new_tokens: 16,
                    grammar_constrained: slot == 2,
                    host_stops: false,
                    prompt_logprobs: false,
                    growable_output: false,
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(1)).unwrap();
    let mut completed = [CompletionRecord::default(); 4];
    for h in &handles {
        completed[h.slot as usize] = CompletionRecord {
            generation: h.generation as i64,
            submission_sequence: 1,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
    }
    scheduler
        .complete(Fence(1), &completed, Default::default())
        .unwrap();
    for h in &handles[..2] {
        scheduler.cancel(*h).unwrap();
    }
    let handle = handles[2];
    assert_eq!(handle.slot, 2);
    let words = config.contract.vocabulary.div_ceil(32);
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    admissions[2] = Some(Admission {
        handle,
        sampling: SamplingParameters::default(),
        max_new_tokens: 16,
        prompt: vec![7; 65],
        data: PromptData {
            positions: (0..65).map(|p| [p; 3]).collect(),
            vision_token_indices: vec![-1; 65],
            ..Default::default()
        },
        grammar_mask: Some(vec![1; words]),
        ignore_eos: false,
        prompt_logprobs: false,
    });
    let mut inputs = Inputs::new(&config, &manifest(&config)).unwrap();
    let step = scheduler.plan().unwrap();
    scheduler.commit(&step, Fence(2)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    assert!(
        !inputs
            .uploads()
            .iter()
            .any(|u| u.buffer == "draft_page_tables")
    );
    completed.fill(CompletionRecord::default());
    completed[2] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 2,
        consumed_tokens: 65,
        generated_tokens: 1,
        token_id: 7,
        status: 1,
        ..Default::default()
    };
    scheduler
        .complete(Fence(2), &completed, Default::default())
        .unwrap();
    retain_prompt(&mut scheduler, 3);
    let masks = [vec![1; words], vec![2; words], vec![4; words]].concat();
    scheduler
        .set_copy_proposal(
            handle,
            CopyProposal {
                consumed: 65,
                tokens: [8, 9, -1, -1, -1, -1, -1],
                count: 2,
                grammar_masks: masks.clone(),
            },
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!(step.token_capacity, 8);
    assert_eq!(step.lanes[0].tokens, 3);
    scheduler.commit(&step, Fence(4)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    let uploads = inputs.uploads();
    let bytes = |name| uploads.iter().find(|u| u.buffer == name).unwrap().bytes;
    let i64s = |name| {
        bytes(name)
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let i32s = |name| {
        bytes(name)
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    assert_eq!(i64s("copy_generation"), [handle.generation as i64]);
    assert_eq!(i64s("copy_expected_consumed"), [65]);
    assert_eq!(i32s("query_lengths"), [3, 0, 0, 0]);
    assert_eq!(i32s("copy_token_count"), [2]);
    assert_eq!(i32s("copy_grammar_mask_count"), [3]);
    assert_eq!(&i64s("copy_token_ids")[..], &[8, 9, -1, -1, -1, -1, -1]);
    assert_eq!(i32s("copy_grammar_mask"), masks);
    assert_eq!(
        uploads
            .iter()
            .find(|u| u.buffer == "copy_grammar_mask")
            .unwrap()
            .offset,
        2 * 8 * words * 4
    );
    assert!(!uploads.iter().any(|u| u.buffer == "draft_page_tables"));
    completed[2] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 4,
        consumed_tokens: 66,
        generated_tokens: 2,
        token_id: 10,
        status: 1,
        vector: Some(crate::execution::TokenVector {
            emitted_count: 1,
            offered_drafts: 2,
            accepted_drafts: 0,
            copy_offered_mask: 3,
            draft_cache_consumed: -1,
            verification_width: 8,
            tokens: [10, -1, -1, -1, -1, -1, -1, -1],
            ..Default::default()
        }),
        ..Default::default()
    };
    scheduler
        .complete(Fence(4), &completed, Default::default())
        .unwrap();
    scheduler
        .set_copy_proposal(
            handle,
            CopyProposal {
                consumed: 66,
                tokens: [-1; 7],
                count: 0,
                grammar_masks: vec![16; words],
            },
        )
        .unwrap();
    let step = scheduler.plan().unwrap();
    assert_eq!(step.token_capacity, 8);
    assert_eq!(step.lanes[0].tokens, 1);
    scheduler.commit(&step, Fence(5)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    let uploads = inputs.uploads();
    let counts = uploads
        .iter()
        .find(|u| u.buffer == "copy_grammar_mask_count")
        .unwrap()
        .bytes;
    assert_eq!(i32::from_le_bytes(counts[..4].try_into().unwrap()), 1);
    let mask = uploads
        .iter()
        .find(|u| u.buffer == "copy_grammar_mask")
        .unwrap();
    assert_eq!(mask.bytes.len(), words * 4);
    assert!(
        mask.bytes
            .chunks_exact(4)
            .all(|b| i32::from_le_bytes(b.try_into().unwrap()) == 16)
    );
}

fn add_copy_inputs(config: &mut ModelConfig) {
    use InputIndexing::RequestRow;
    use ValueSource::*;
    for graph in &mut config.contract.graphs {
        if graph.kind != GraphOperation::Speculative {
            continue;
        }
        graph.draft_cursor = None;
        for (name, source) in [
            ("copy_token_ids", CopyTokenIds),
            ("copy_token_count", CopyTokenCount),
            ("copy_expected_consumed", CopyConsumedTokens),
            ("copy_generation", CopyGeneration),
            ("copy_grammar_mask", CopyGrammarMask),
            ("copy_grammar_mask_count", CopyGrammarCount),
        ] {
            graph
                .inputs
                .push(testing::binding(name, source, RequestRow, 0, 0));
        }
    }
}

#[test]
fn mixed_native_uploads_pack_prompt_after_the_decode_extent_and_sample_only_eligible_rows() {
    for width in [1, 3] {
        let mut config = config();
        if width > 1 {
            config.speculative_widths = vec![1, 3];
        }
        let m = mixed_manifest(&mut config, width);
        let mut scheduler = RequestScheduler::new(
            config.identity,
            config.limits,
            scheduler_capabilities(&config),
        )
        .unwrap();
        scheduler
            .enable_speculation(&config.speculative_widths)
            .unwrap();
        scheduler
            .mixed_graphs(&[(2, width, 64), (2, width, 128)])
            .unwrap();
        scheduler.measured_graph_durations(&m.graphs).unwrap();
        let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
        let mut admit = |scheduler: &mut RequestScheduler, count: usize, token: u32| {
            let prompt = vec![token; count];
            let h = scheduler
                .admit(RequestSpec {
                    prompt: prompt.clone(),
                    message_boundaries: vec![],
                    conditioning: vec![],
                    max_new_tokens: 16,
                    grammar_constrained: false,
                    host_stops: false,
                    prompt_logprobs: false,
                    growable_output: false,
                })
                .unwrap();
            admissions[h.slot as usize] = Some(Admission {
                handle: h,
                sampling: SamplingParameters::default(),
                max_new_tokens: 16,
                prompt,
                data: PromptData {
                    positions: (0..count).map(|p| [p as i64; 3]).collect(),
                    vision_token_indices: vec![-1; count],
                    ..Default::default()
                },
                grammar_mask: None,
                ignore_eos: false,
                prompt_logprobs: false,
            });
            h
        };
        let d = admit(&mut scheduler, 65, 7);
        for (fence, kind, consumed, generated, token) in [
            (1, GraphKind::Init, 0, 0, -1),
            (2, GraphKind::Prefill, 65, 1, 7),
        ] {
            let plan = scheduler.phase_plan(kind, Some(d));
            scheduler.commit(&plan, Fence(fence)).unwrap();
            let mut records = [CompletionRecord::default(); 4];
            records[d.slot as usize] = CompletionRecord {
                generation: d.generation as i64,
                submission_sequence: fence as i64,
                consumed_tokens: consumed,
                generated_tokens: generated,
                token_id: token,
                status: 1,
                ..Default::default()
            };
            scheduler
                .complete(Fence(fence), &records, Default::default())
                .unwrap();
        }
        retain_prompt(&mut scheduler, 3);
        let p = admit(&mut scheduler, 129, 11);
        let init = scheduler.phase_plan(GraphKind::Init, Some(p));
        scheduler.commit(&init, Fence(4)).unwrap();
        let mut records = [CompletionRecord::default(); 4];
        records[p.slot as usize] = CompletionRecord {
            generation: p.generation as i64,
            submission_sequence: 4,
            token_id: -1,
            status: 1,
            ..Default::default()
        };
        scheduler
            .complete(Fence(4), &records, Default::default())
            .unwrap();
        let plan = scheduler.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Mixed);
        assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, Some(1));
        scheduler.commit(&plan, Fence(5)).unwrap();
        let mut inputs = Inputs::new(&config, &m).unwrap();
        crate::model::pack_inputs(&mut inputs, &config, &scheduler, &plan, &admissions).unwrap();
        let uploads = inputs.uploads();
        let integers = |name: &str| {
            uploads
                .iter()
                .find(|u| u.buffer == name)
                .unwrap()
                .bytes
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(integers("prefill_row"), [1]);
        assert_eq!(integers("vision_feature_slot"), [-1]);
        assert_eq!(integers("query_lengths"), [width as i32, 64, 0, 0]);
        assert_eq!(integers("sample_enabled"), [1, 0, 0, 0]);
        assert_eq!(
            uploads
                .iter()
                .filter(|u| u.buffer == "query_lengths")
                .count(),
            1
        );
        for (name, bytes_per_token) in [
            ("token_ids", 8),
            ("prefill_positions", 24),
            ("vision_token_indices", 4),
        ] {
            let upload = uploads.iter().find(|u| u.buffer == name).unwrap();
            assert_eq!(upload.offset, width * bytes_per_token);
            assert_eq!(upload.bytes.len(), 64 * bytes_per_token);
        }
        let tokens = uploads.iter().find(|u| u.buffer == "token_ids").unwrap();
        assert!(
            tokens
                .bytes
                .chunks_exact(8)
                .all(|v| i64::from_le_bytes(v.try_into().unwrap()) == 11)
        );
        let mut wrong = (*plan).clone();
        wrong.mixed.as_mut().unwrap().prefill_row = Some(0);
        assert!(
            crate::model::pack_inputs(&mut inputs, &config, &scheduler, &wrong, &admissions)
                .is_err()
        );
    }
}

fn mixed_manifest(config: &mut ModelConfig, width: usize) -> crate::execution::ExecutionManifest {
    use crate::execution::{BufferDescriptor, Dtype, GraphDescriptor, InputBinding, SamplingMode};
    let mut manifest = manifest(config);
    let address = manifest
        .buffers
        .iter()
        .map(|b| b.address + b.nbytes)
        .max()
        .unwrap()
        .next_multiple_of(8);
    manifest.buffers.push(BufferDescriptor {
        name: "prefill_row".into(),
        address,
        nbytes: 4,
        dtype: Dtype::Int32,
        shape: vec![1],
        strides: vec![1],
    });
    let mut bindings = manifest
        .graphs
        .iter()
        .find(|g| g.kind == GraphKind::Prefill)
        .unwrap()
        .input_bindings
        .clone();
    bindings.push(InputBinding {
        buffer_name: "prefill_row".into(),
        host_offset: manifest.input_frame_bytes,
        nbytes: 4,
    });
    manifest.input_frame_bytes += 8;
    let prefill = config
        .contract
        .graphs
        .iter()
        .find(|g| g.kind == GraphOperation::Prefill)
        .unwrap()
        .clone();
    let decode = config
        .contract
        .graphs
        .iter()
        .find(|g| g.kind == GraphOperation::Decode)
        .unwrap()
        .clone();
    for capacity in [64, 128] {
        let name = format!("mixed-{capacity}");
        let mut semantics = prefill.clone();
        semantics.graph = name.clone();
        semantics.kind = GraphOperation::Mixed;
        semantics.pending_token = decode.pending_token.clone();
        semantics.inputs.push(testing::binding(
            "prefill_row",
            ValueSource::PrefillRow,
            InputIndexing::Scalar,
            0,
            -1,
        ));
        config.contract.graphs.push(semantics);
        manifest.graphs.push(GraphDescriptor { draft_tokens: 0,
            packed_token_capacity: 0,
            name,
            kind: GraphKind::Mixed,
            sampling_mode: SamplingMode::General,
            handle: manifest.graphs.len() + 1,
            batch_capacity: 2,
            token_capacity: width,
            prefill_capacity: capacity,
            estimated_duration_ns: if capacity == 64 { 1 } else { 10_000_000 },
            input_bindings: bindings.clone(),
        });
    }
    manifest
}

#[test]
fn state_transfer_and_resume_upload_fixture() {
    let config = config();
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&self::manifest(&config).graphs)
        .unwrap();
    let mut inputs = Inputs::new(&config, &manifest(&config)).unwrap();
    for cycle in 0..2 {
        let prompt = vec![7; 64];
        let handle = scheduler
            .admit(RequestSpec {
                message_boundaries: vec![],
                prompt: prompt.clone(),
                conditioning: vec![],
                max_new_tokens: 2,
                grammar_constrained: false,
                host_stops: false,
                prompt_logprobs: false,
                growable_output: false,
            })
            .unwrap();
        let mut admissions: [_; 4] = std::array::from_fn(|_| None);
        admissions[handle.slot as usize] = Some(Admission {
            handle,
            sampling: SamplingParameters::default(),
            max_new_tokens: 2,
            prompt,
            data: PromptData {
                positions: (0..64).map(|i| [i, i + 1, i + 2]).collect(),
                vision_token_indices: vec![-1; 64],
                ..Default::default()
            },
            grammar_mask: None,
            ignore_eos: false,
            prompt_logprobs: false,
        });
        for (offset, kind, consumed, generated, token, status) in if cycle == 0 {
            vec![
                (1, GraphKind::Init, 0, 0, -1, 1),
                (2, GraphKind::Prefill, 64, 1, 7, 1),
                (3, GraphKind::Snapshot, 64, 1, -1, 1),
                (4, GraphKind::Decode, 65, 2, 8, 2),
                (5, GraphKind::Snapshot, 65, 2, -1, 1),
            ]
        } else {
            vec![
                (1, GraphKind::Restore, 64, 0, -1, 1),
                (2, GraphKind::Resume, 64, 1, 7, 1),
                (3, GraphKind::Decode, 65, 2, 8, 2),
                (4, GraphKind::Snapshot, 65, 2, -1, 1),
            ]
        } {
            let fence = Fence(cycle * 5 + offset);
            let step = scheduler.plan().unwrap();
            assert_eq!(step.kind, kind);
            scheduler.commit(&step, fence).unwrap();
            crate::model::pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions)
                .unwrap();
            assert_state_bindings(&inputs, &scheduler, &step);
            assert!(!inputs.uploads().is_empty());
            if matches!(
                kind,
                GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot
            ) {
                let uploads = inputs.uploads();
                let values = |name| {
                    uploads
                        .iter()
                        .find(|u| u.buffer == name)
                        .unwrap()
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(values("state_expected_consumed"), [consumed, -1, -1, -1]);
                if kind == GraphKind::Snapshot {
                    assert_eq!(values("state_source_slots"), [1, 0, 0, 0]);
                    assert_eq!(
                        values("state_destination_slots")[0],
                        scheduler
                            .snapshot_slot(
                                scheduler.request(handle).unwrap().snapshot_destination()
                            )
                            .0 as i64
                    );
                }
            }
            let mut records = [CompletionRecord::default(); 4];
            records[handle.slot as usize] = CompletionRecord {
                generation: handle.generation as i64,
                submission_sequence: fence.0 as i64,
                consumed_tokens: consumed,
                generated_tokens: generated,
                token_id: token,
                status,
                ..Default::default()
            };
            scheduler
                .complete(fence, &records, Default::default())
                .unwrap();
        }
        scheduler.finish(handle).unwrap();
    }
}

#[cfg(feature = "images")]
#[test]
fn vision_upload_selects_source_image_independently_of_destination_feature_offset() {
    use crate::frontend::images::{ImageGeometry, ProcessedImages};
    let mut config = config();
    config.contract =
        serde_json::from_str(include_str!("../../tests/support/contracts/qwen38.json")).unwrap();
    let images = ProcessedImages {
        pixel_values: (0..8 * 1536).map(|i| (i as f32 - 71.) / 32.).collect(),
        geometry: (0..2)
            .map(|_| ImageGeometry {
                height: 32,
                width: 32,
                grid_thw: [1, 2, 2],
                patches: 4,
                tokens: 1,
            })
            .collect(),
        position_ids: (0..8).map(|i| [i, i + 1]).collect(),
        interpolation_indices: (0..8).map(|i| [i, i + 1, i + 2, i + 3]).collect(),
        interpolation_weights: vec![[0.25; 4]; 8],
        cu_seqlens: vec![0, 4, 8],
    };
    for (image, destination) in [(0, 0), (1, 1), (1, 0)] {
        let staging = crate::frontend::images::packing::prepare(
            &config,
            crate::scheduler::RequestHandle {
                slot: 3,
                generation: 19,
            },
            &images,
            image,
            destination,
        )
        .unwrap();
        let uploads = staging.uploads();
        assert_eq!(
            uploads
                .iter()
                .find(|u| u.buffer == "vision_feature_offset")
                .unwrap()
                .bytes,
            (destination as i32).to_le_bytes()
        );
        let pixels = uploads
            .iter()
            .find(|u| u.buffer == "vision_pixel_values")
            .unwrap();
        assert_eq!(
            pixels.bytes,
            images.pixel_values[image * 4 * 1536..(image + 1) * 4 * 1536]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn retained_proposal_catalog_requires_prefix_coverage_and_proposal_free_decode() {
    let mut config = config();
    config.limits.max_active = 1;
    config.limits.request_slots = 1;
    config.history_drafts = false;
    config.speculative_widths = (1..=8).collect();
    config
        .contract
        .graphs
        .retain(|g| g.graph.split('-').nth(1) == Some("1"));
    let mut manifest = manifest(&config);
    let mut proposal = manifest
        .graphs
        .iter()
        .find(|g| g.kind == GraphKind::Speculative && g.token_capacity == 8)
        .unwrap()
        .clone();
    let mut semantics = config.contract.graph(&proposal.name).unwrap().clone();
    proposal.name = "Proposal-1-8".into();
    proposal.kind = GraphKind::Proposal;
    semantics.graph = proposal.name.clone();
    semantics.kind = GraphOperation::Proposal;
    semantics.draft_cursor = None;
    for state in &mut semantics.state {
        state.access = StateAccessKind::Read;
    }
    let mut decode = manifest
        .graphs
        .iter()
        .find(|g| g.kind == GraphKind::Speculative && g.token_capacity == 1)
        .unwrap()
        .clone();
    let mut decode_contract = config.contract.graph(&decode.name).unwrap().clone();
    decode.name = "Speculative-free-1".into();
    decode_contract.graph = decode.name.clone();
    for graph in &mut manifest.graphs {
        if graph.kind == GraphKind::Speculative {
            graph.kind = GraphKind::Verification;
        }
    }
    for graph in &mut config.contract.graphs {
        if graph.kind == GraphOperation::Speculative {
            graph.kind = GraphOperation::Verification;
        }
    }
    manifest.graphs.push(decode);
    config.contract.graphs.push(decode_contract);
    manifest.graphs.push(proposal);
    config.contract.graphs.push(semantics);
    for (index, graph) in manifest.graphs.iter_mut().enumerate() {
        graph.handle = index + 1;
    }
    let address = manifest
        .buffers
        .iter()
        .map(|b| b.address + b.nbytes)
        .max()
        .unwrap()
        .next_multiple_of(4096);
    manifest.buffers.push(crate::execution::BufferDescriptor {
        name: "proposal_confidence".into(),
        address,
        nbytes: 28,
        dtype: crate::execution::Dtype::Float32,
        shape: vec![1, 7],
        strides: vec![7, 1],
    });
    catalog::validate(&config, &manifest).unwrap();
    // Packages with only the maximum proposal depth remain valid. Additional
    // captured depths use the same ownership and verification contract.
    let mut shallow = manifest
        .graphs
        .iter()
        .find(|g| g.kind == GraphKind::Proposal)
        .unwrap()
        .clone();
    let mut shallow_contract = config.contract.graph(&shallow.name).unwrap().clone();
    shallow.name = "Proposal-1-2".into();
    shallow.token_capacity = 2;
    shallow.handle = manifest.graphs.len() + 1;
    shallow_contract.graph = shallow.name.clone();
    manifest.graphs.insert(0, shallow);
    config.contract.graphs.push(shallow_contract);
    catalog::validate(&config, &manifest).unwrap();
    assert_eq!(
        manifest.graphs[catalog::select(
            &config,
            &manifest,
            GraphKind::Proposal,
            1,
            2,
            0,
            crate::execution::SamplingMode::General
        )
        .unwrap()]
        .token_capacity,
        2
    );
    let mut missing = manifest.clone();
    missing
        .graphs
        .retain(|g| g.kind != GraphKind::Verification || g.token_capacity != 3);
    assert!(catalog::validate(&config, &missing).is_err());
    let mut multiple = config.clone();
    multiple.limits.max_active = 2;
    assert!(catalog::validate(&multiple, &manifest).is_err());
    let mut packed = manifest.clone();
    for graph in &mut packed.graphs {
        if graph.kind == GraphKind::Verification {
            graph.packed_token_capacity = graph.token_capacity;
            graph.token_capacity = 8;
        }
    }
    catalog::validate(&config, &packed).unwrap();
    let index = catalog::select_shape(
        &config,
        &packed,
        GraphKind::Verification,
        1,
        8,
        0,
        3,
        crate::execution::SamplingMode::General,
    )
    .unwrap();
    assert_eq!(packed.graphs[index].packed_tokens(), 3);
    packed.graphs[index].packed_token_capacity = 9;
    assert!(catalog::validate(&config, &packed).is_err());
    let mut coexist = manifest.clone();
    let template = coexist.graphs.iter().find(|g|
        g.kind == GraphKind::Verification && g.token_capacity == 8).unwrap().clone();
    let prefix_address = coexist.buffers.iter().map(|b| b.address + b.nbytes).max().unwrap().next_multiple_of(8);
    coexist.buffers.push(crate::execution::BufferDescriptor { name: "speculative_prefix_limits".into(),
        address: prefix_address, nbytes: 4, dtype: crate::execution::Dtype::Int32,
        shape: vec![1], strides: vec![1] });
    for (kind, operation, name) in [
        (GraphKind::Speculative, GraphOperation::Speculative, "joined"),
        (GraphKind::PackedSpeculative, GraphOperation::PackedSpeculative, "packed-round"),
    ] {
        let mut graph = template.clone();
        let mut semantics = config.contract.graph(&template.name).unwrap().clone();
        graph.name = name.into(); graph.kind = kind; graph.handle = coexist.graphs.len() + 1;
        semantics.graph = name.into(); semantics.kind = operation;
        if kind == GraphKind::PackedSpeculative {
            graph.draft_tokens = 5; graph.packed_token_capacity = 4;
            let input = graph.input_bindings.iter_mut().find(|b| b.buffer_name == "query_lengths").unwrap();
            input.buffer_name = "speculative_prefix_limits".into(); input.host_offset = coexist.input_frame_bytes;
            coexist.input_frame_bytes += 8;
            let input = semantics.inputs.iter_mut().find(|b| b.source == ValueSource::QueryLength).unwrap();
            input.buffer = "speculative_prefix_limits".into(); input.source = ValueSource::SpeculativePrefixLimit;
        }
        coexist.graphs.push(graph); config.contract.graphs.push(semantics);
    }
    catalog::validate(&config, &coexist).unwrap();
    config.speculative_token_capacity = 0;
    assert!(catalog::validate(&config, &coexist).is_err());
}

fn grammar_packing_fixture() -> (
    ModelConfig,
    crate::execution::ExecutionManifest,
    RequestScheduler,
    Vec<Option<Admission>>,
    crate::scheduler::ScheduledStep,
) {
    let mut config = config();
    config.limits.request_slots = 8;
    config.limits.max_active = 8;
    config.limits.pages = 256;
    config.contract.vocabulary = 129280;
    config.contract.input_vocabulary = 129280;
    let mut template = config
        .contract
        .graphs
        .iter()
        .find(|g| g.graph == "Speculative-4-6")
        .unwrap()
        .clone();
    template.graph = "Speculative-8-6".into();
    template.inputs.push(super::testing::binding(
        "pending_input_ids",
        ValueSource::PendingToken,
        InputIndexing::SubmissionRow,
        0,
        0,
    ));
    config.contract.graphs.push(template.clone());
    let mut manifest = manifest(&config);
    let descriptor = manifest
        .graphs
        .iter()
        .find(|g| g.name == template.graph)
        .unwrap()
        .clone();
    for (kind, operation, name) in [
        (GraphKind::Proposal, GraphOperation::Proposal, "proposal8x6"),
        (
            GraphKind::Verification,
            GraphOperation::Verification,
            "verification8x6",
        ),
    ] {
        let mut contract = template.clone();
        contract.graph = name.into();
        contract.kind = operation;
        config.contract.graphs.push(contract);
        let mut graph = descriptor.clone();
        graph.name = name.into();
        graph.kind = kind;
        manifest.graphs.push(graph);
    }
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    let mut admissions = Vec::new();
    let mut lanes = Vec::new();
    for _ in 0..8 {
        let handle = scheduler
            .admit(RequestSpec {
                message_boundaries: vec![],
                prompt: vec![7; 66],
                conditioning: vec![],
                max_new_tokens: 512,
                grammar_constrained: false,
                host_stops: false,
                prompt_logprobs: false,
                growable_output: false,
            })
            .unwrap();
        admissions.push(Some(Admission {
            handle,
            sampling: SamplingParameters::default(),
            max_new_tokens: 512,
            prompt: vec![7; 66],
            data: PromptData::default(),
            grammar_mask: None,
            ignore_eos: false,
            prompt_logprobs: false,
        }));
        lanes.push(crate::scheduler::ScheduledLane {
            request: handle,
            start_position: 0,
            tokens: 3,
        });
    }
    let step = crate::scheduler::ScheduledStep { draft_tokens: 0,
        kind: GraphKind::Proposal,
        batch_capacity: 8,
        token_capacity: 6,
        packed_token_capacity: 0,
        mixed: None,
        requires_feedback: true,
        lanes,
    };
    (config, manifest, scheduler, admissions, step)
}

#[test]
fn grammar_mask_packing_preserves_bits_row_layout_and_transitions() {
    use InputIndexing::*;
    let (config, manifest, scheduler, mut admissions, mut step) = grammar_packing_fixture();
    step.lanes = [6, 1, 3].map(|slot| step.lanes[slot].clone()).into();
    let words = config.contract.vocabulary.div_ceil(32);
    for indexing in [RequestRow, ActiveRow, SubmissionRow] {
        let mut config = config.clone();
        for graph in &mut config.contract.graphs {
            for binding in &mut graph.inputs {
                if binding.buffer == "sampling_grammar_mask" {
                    binding.indexing = indexing;
                }
            }
        }
        for kind in [GraphKind::Proposal, GraphKind::Verification] {
            step.kind = kind;
            let mut inputs = Inputs::new(&config, &manifest).unwrap();
            // Graph inputs persist between submissions; uploads only change what they cover.
            let mut actual = vec![0xa5; 8 * words * 4];
            let mut expected =
                vec![if indexing == SubmissionRow { 0 } else { 0xa5 }; actual.len()];
            for round in 0..6 {
                if indexing == SubmissionRow {
                    // Submission rows are repacked in full with every step.
                    actual.fill(0xa5);
                    expected.fill(0);
                }
                for (slot, admission) in admissions.iter_mut().enumerate() {
                    admission.as_mut().unwrap().grammar_mask = match (round, slot) {
                        (0, 1) | (1, 6) => Some(
                            (0..words)
                                .map(|i| [i32::MIN, i32::MAX, 0, -1][i % 4])
                                .collect(),
                        ),
                        (0, 6) => Some(vec![0; words]),
                        (1, 3) => Some(vec![1; words]),
                        (3, 1) => Some(vec![]),
                        _ => None,
                    };
                }
                pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
                let mut uploaded = 0;
                for upload in inputs
                    .uploads()
                    .into_iter()
                    .filter(|u| u.buffer == "sampling_grammar_mask")
                {
                    assert!(!upload.bytes.is_empty());
                    uploaded += upload.bytes.len();
                    actual[upload.offset..upload.offset + upload.bytes.len()]
                        .copy_from_slice(upload.bytes);
                }
                if round == 5 && indexing != SubmissionRow {
                    // A repeated unconstrained submission leaves the resident masks in place.
                    assert_eq!(uploaded, 0, "{indexing:?} {kind:?} round {round}");
                }
                for (row, lane) in step.lanes.iter().enumerate() {
                    let slot = lane.request.slot as usize;
                    let destination = if indexing == RequestRow { slot } else { row };
                    let mask = admissions[slot].as_ref().unwrap().grammar_mask.as_ref();
                    // Empty masks leave the row untouched, including its padding.
                    if mask.is_some_and(Vec::is_empty) {
                        continue;
                    }
                    for word in 0..words {
                        let value = mask.map_or(-1, |m| m[word]);
                        let offset = (destination * words + word) * 4;
                        expected[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
                assert_eq!(actual, expected, "{indexing:?} {kind:?} round {round}");
            }
        }
    }
}

#[test]
fn grammar_mask_cache_survives_alternating_graph_indexing() {
    for indexing in [InputIndexing::RequestRow, InputIndexing::ActiveRow] {
        let (mut config, manifest, scheduler, mut admissions, mut step) = grammar_packing_fixture();
        for graph in &mut config.contract.graphs {
            for binding in &mut graph.inputs {
                if binding.buffer == "sampling_grammar_mask" {
                    binding.indexing = if graph.kind == GraphOperation::Verification {
                        InputIndexing::SubmissionRow
                    } else { indexing };
                }
            }
        }
        step.lanes = [6, 1, 3].map(|slot| step.lanes[slot].clone()).into();
        let words = config.contract.vocabulary.div_ceil(32);
        let mut actual = vec![0xa5; 8 * words * 4];
        let mut inputs = Inputs::new(&config, &manifest).unwrap();
        for kind in [GraphKind::Proposal, GraphKind::Verification, GraphKind::Proposal] {
            step.kind = kind;
            for admission in admissions.iter_mut().flatten() {
                admission.grammar_mask = (kind == GraphKind::Verification).then(|| vec![0; words]);
            }
            pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
            for upload in inputs.uploads().into_iter().filter(|u| u.buffer == "sampling_grammar_mask") {
                actual[upload.offset..upload.offset + upload.bytes.len()].copy_from_slice(upload.bytes);
            }
            if kind == GraphKind::Proposal {
                for (row, lane) in step.lanes.iter().enumerate() {
                    let destination = if indexing == InputIndexing::RequestRow { lane.request.slot as usize } else { row };
                    assert!(actual[destination * words * 4..(destination + 1) * words * 4].iter().all(|&b| b == 0xff),
                        "{indexing:?}: graph alternation must restore unconstrained mask bytes");
                }
            }
        }
    }
}

#[test]
fn grammar_mask_cache_invalidates_other_bindings_and_graph_writes() {
    for mode in ["missing-input", "offset-input", "state-write"] {
        let (mut config, manifest, scheduler, admissions, mut step) = grammar_packing_fixture();
        let words = config.contract.vocabulary.div_ceil(32);
        let graph = config.contract.graphs.iter_mut().find(|g| g.kind == GraphOperation::Verification).unwrap();
        if mode == "missing-input" {
            graph.inputs.retain(|b| b.buffer != "sampling_grammar_mask");
        } else if mode == "offset-input" {
            graph.inputs.iter_mut().find(|b| b.buffer == "sampling_grammar_mask").unwrap().value_offset = 1;
        } else {
            graph.state.push(StateAccess {
                view: StateView { buffer: "sampling_grammar_mask".into(), offset: 0, stride: words * 4, nbytes: words * 4 },
                access: StateAccessKind::Write,
            });
        }
        step.lanes = vec![step.lanes[1].clone()];
        let mut inputs = Inputs::new(&config, &manifest).unwrap();
        let mut actual = vec![0xa5; 8 * words * 4];
        for kind in [GraphKind::Proposal, GraphKind::Verification, GraphKind::Verification, GraphKind::Proposal] {
            step.kind = kind;
            pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
            for upload in inputs.uploads().into_iter().filter(|u| u.buffer == "sampling_grammar_mask") {
                actual[upload.offset..upload.offset + upload.bytes.len()].copy_from_slice(upload.bytes);
            }
            if kind == GraphKind::Proposal || mode == "state-write" {
                assert!(actual[words * 4..2 * words * 4].iter().all(|&b| b == 0xff), "{mode}: missing unconstrained upload");
            }
            if kind == GraphKind::Verification {
                // Model execution may overwrite a destination not preserved by
                // its semantic input contract; reproduce that device write.
                actual.fill(0);
            }
        }
    }
}

#[test]
fn grammar_mask_cache_discards_unsubmitted_rows_after_pack_error() {
    let (config, manifest, scheduler, mut admissions, mut step) = grammar_packing_fixture();
    let words = config.contract.vocabulary.div_ceil(32);
    step.lanes = [0, 7].map(|slot| step.lanes[slot].clone()).into();
    let mut inputs = Inputs::new(&config, &manifest).unwrap();
    admissions[7].as_mut().unwrap().grammar_mask = Some(vec![0; words + 1]);
    assert!(pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).is_err());
    admissions[7].as_mut().unwrap().grammar_mask = None;
    pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    let uploaded: usize = inputs.uploads().iter().filter(|u| u.buffer == "sampling_grammar_mask").map(|u| u.bytes.len()).sum();
    assert_eq!(uploaded, 2 * words * 4, "the failed pack uploaded neither row");
    pack_inputs(&mut inputs, &config, &scheduler, &step, &admissions).unwrap();
    assert!(!inputs.uploads().iter().any(|u| u.buffer == "sampling_grammar_mask"));
}

#[test]
fn prefill_compacts_execution_feature_indices_and_preserves_admission_metadata() {
    let mut config = config();
    let image_contract: PreparedContract =
        serde_json::from_str(include_str!("../../tests/support/contracts/qwen38.json")).unwrap();
    config.contract.vision = image_contract.vision;
    config.contract.frontend = image_contract.frontend;
    let mut scheduler = RequestScheduler::new(
        config.identity,
        config.limits,
        scheduler_capabilities(&config),
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&manifest(&config).graphs)
        .unwrap();
    let handle = scheduler
        .admit(RequestSpec {
            prompt: vec![7; 4],
            message_boundaries: vec![],
            conditioning: vec![],
            max_new_tokens: 1,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: true,
        })
        .unwrap();
    scheduler
        .set_conditioned_features(handle, &[128..192])
        .unwrap();
    let canonical = vec![-1, 128, 129, -1];
    let mut admissions = std::array::from_fn::<_, 4, _>(|_| None);
    admissions[handle.slot as usize] = Some(Admission {
        handle,
        sampling: SamplingParameters::default(),
        max_new_tokens: 1,
        prompt: vec![7; 4],
        data: PromptData {
            positions: vec![[0; 3]; 4],
            vision_token_indices: canonical.clone(),
            ..Default::default()
        },
        grammar_mask: None,
        ignore_eos: false,
        prompt_logprobs: false,
    });
    let init = scheduler.plan().unwrap();
    scheduler.commit(&init, Fence(1)).unwrap();
    let mut records = [CompletionRecord::default(); 4];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64,
        submission_sequence: 1,
        status: 1,
        token_id: -1,
        ..Default::default()
    };
    scheduler
        .complete(Fence(1), &records, Default::default())
        .unwrap();
    let prefill = scheduler.plan().unwrap();
    assert_eq!(prefill.kind, GraphKind::Prefill);
    scheduler.commit(&prefill, Fence(2)).unwrap();
    let mut inputs = Inputs::new(&config, &manifest(&config)).unwrap();
    crate::model::pack_inputs(&mut inputs, &config, &scheduler, &prefill, &admissions).unwrap();
    let uploads = inputs.uploads();
    let indices = uploads
        .iter()
        .find(|upload| upload.buffer == "vision_token_indices")
        .unwrap();
    assert_eq!(
        indices.bytes,
        [-1i32, 0, 1, -1]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        admissions[handle.slot as usize]
            .as_ref()
            .unwrap()
            .data
            .vision_token_indices,
        canonical
    );
}
include!("packed_round_tests.rs");
