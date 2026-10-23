fn packed_round_manifest(config: &mut ModelConfig) -> crate::execution::ExecutionManifest {
    use crate::execution::{BufferDescriptor, Dtype, InputBinding};
    config.history_drafts = false;
    config.speculative_widths = (1..=8).collect();
    let mut result = manifest(config);
    result.graphs.retain(|g| (g.kind != GraphKind::Prefill || g.batch_capacity == 1)
        && (g.kind != GraphKind::Speculative || [1, 8].contains(&g.token_capacity)));
    let names: std::collections::HashSet<_> = result.graphs.iter().map(|g| g.name.clone()).collect();
    config.contract.graphs.retain(|g| names.contains(&g.graph));
    let address = result.buffers.iter().map(|b| b.address + b.nbytes).max().unwrap().next_multiple_of(8);
    let bytes = config.limits.request_capacity() * 4;
    result.buffers.push(BufferDescriptor { name: "speculative_prefix_limits".into(), address,
        nbytes: bytes, dtype: Dtype::Int32, shape: vec![config.limits.request_capacity()], strides: vec![1] });
    result.buffers.push(BufferDescriptor { name: "proposal_confidence".into(), address: address + bytes,
        nbytes: config.limits.request_capacity() * 28, dtype: Dtype::Float32,
        shape: vec![config.limits.request_capacity(), 7], strides: vec![7, 1] });
    for graph in &mut result.graphs {
        if graph.kind != GraphKind::Speculative || graph.token_capacity == 1 { continue; }
        graph.kind = GraphKind::PackedSpeculative;
        graph.draft_tokens = 7;
        graph.packed_token_capacity = graph.batch_capacity * 3;
        graph.input_bindings.retain(|b| b.buffer_name != "query_lengths");
        graph.input_bindings.push(InputBinding { buffer_name: "speculative_prefix_limits".into(),
            host_offset: result.input_frame_bytes, nbytes: bytes });
        let semantics = config.contract.graphs.iter_mut().find(|g| g.graph == graph.name).unwrap();
        semantics.kind = GraphOperation::PackedSpeculative;
        let binding = semantics.inputs.iter_mut().find(|b| b.source == ValueSource::QueryLength).unwrap();
        binding.source = ValueSource::SpeculativePrefixLimit;
        binding.buffer = "speculative_prefix_limits".into();
    }
    result.input_frame_bytes += bytes;
    for (index, graph) in result.graphs.iter_mut().enumerate() { graph.handle = index + 1; }
    result
}

#[test]
fn packed_combined_contract_separates_immutable_ceilings_from_device_query_lengths() {
    let mut config = config();
    let value = packed_round_manifest(&mut config);
    catalog::validate(&config, &value).unwrap();
    Inputs::new(&config, &value).unwrap();
    for mutation in 0..5 {
        let mut invalid = config.clone();
        let graph = invalid.contract.graphs.iter_mut().find(|g| g.kind == GraphOperation::PackedSpeculative).unwrap();
        let prefix = graph.inputs.iter_mut().find(|b| b.source == ValueSource::SpeculativePrefixLimit).unwrap();
        match mutation {
            0 => prefix.source = ValueSource::QueryLength,
            1 => prefix.indexing = InputIndexing::RequestRow,
            2 => prefix.padding = 1,
            3 => prefix.value_offset = 1,
            _ => { let extra = prefix.clone(); graph.inputs.push(extra); }
        }
        assert!(Inputs::new(&invalid, &value).is_err());
    }
    let mut invalid = value.clone();
    invalid.buffers.iter_mut().find(|b| b.name == "speculative_prefix_limits").unwrap().dtype = crate::execution::Dtype::Int64;
    assert!(Inputs::new(&config, &invalid).is_err());
}

#[test]
fn packed_combined_budget_keeps_width_one_batches_without_a_useful_packed_shape() {
    let mut config = config();
    let mut value = packed_round_manifest(&mut config);
    config.speculative_widths = (1..=6).collect();
    config.speculative_token_capacity = 6;
    for graph in &mut value.graphs {
        if graph.kind == GraphKind::PackedSpeculative {
            graph.token_capacity = 6;
            graph.draft_tokens = 5;
        }
    }
    value.graphs.retain(|g| g.kind != GraphKind::PackedSpeculative || g.packed_token_capacity <= 6);
    let names: std::collections::HashSet<_> = value.graphs.iter().map(|g| g.name.clone()).collect();
    config.contract.graphs.retain(|g| names.contains(&g.graph));
    catalog::validate(&config, &value).unwrap();
    Inputs::new(&config, &value).unwrap();
    assert!(!value.graphs.iter().any(|g| g.kind == GraphKind::PackedSpeculative && g.batch_capacity == 4));
    let mut partial = value.clone();
    let mut partial_config = config.clone();
    let template = partial.graphs.iter().find(|g| g.kind == GraphKind::PackedSpeculative && g.batch_capacity == 2).unwrap().clone();
    let mut graph = template.clone();
    graph.name = "packed-partial-four".into();
    graph.batch_capacity = 4;
    graph.packed_token_capacity = 5;
    graph.handle = partial.graphs.len() + 1;
    let mut semantics = partial_config.contract.graph(&template.name).unwrap().clone();
    semantics.graph = graph.name.clone();
    partial_config.contract.graphs.push(semantics);
    partial.graphs.push(graph);
    catalog::validate(&partial_config, &partial).unwrap();
    Inputs::new(&partial_config, &partial).unwrap();
    for (kind, batch) in [(GraphKind::PackedSpeculative, 2), (GraphKind::Speculative, 4)] {
        let mut invalid = value.clone();
        let omitted = invalid.graphs.iter().find(|g| g.kind == kind && g.batch_capacity == batch).unwrap().name.clone();
        invalid.graphs.retain(|g| g.name != omitted);
        let mut invalid_config = config.clone();
        invalid_config.contract.graphs.retain(|g| g.graph != omitted);
        assert!(catalog::validate(&invalid_config, &invalid).is_err(), "missing {kind:?} batch {batch}");
    }
}

#[test]
fn packed_combined_packing_preserves_sparse_slot_keys_and_ceiling_padding() {
    let mut config = config();
    let manifest = packed_round_manifest(&mut config);
    let mut scheduler = RequestScheduler::new(config.identity, config.limits, scheduler_capabilities(&config)).unwrap();
    scheduler.enable_speculation(&config.speculative_widths).unwrap();
    scheduler.split_speculation(8).unwrap();
    scheduler.measured_graph_durations(&manifest.graphs).unwrap();
    let mut admissions: [Option<Admission>; 4] = std::array::from_fn(|_| None);
    let mut handles = Vec::new();
    for _ in 0..4 {
        let prompt = vec![7; 64];
        let h = scheduler.admit(RequestSpec { prompt_logprobs: false, message_boundaries: vec![], prompt: prompt.clone(), conditioning: vec![],
            max_new_tokens: 64, grammar_constrained: false, host_stops: false, growable_output: false }).unwrap();
        admissions[h.slot as usize] = Some(Admission { prompt_logprobs: false, handle: h, sampling: SamplingParameters::default(),
            max_new_tokens: 64, prompt, data: PromptData { positions: (0..64).map(|i| [i; 3]).collect(),
                vision_token_indices: vec![-1; 64], ..Default::default() }, grammar_mask: None, ignore_eos: true });
        handles.push(h);
    }
    scheduler.cancel(handles[1]).unwrap();admissions[handles[1].slot as usize] = None;
    let mut inputs = Inputs::new(&config, &manifest).unwrap();
    for sequence in 1..32 {
        let plan = scheduler.plan().unwrap();
        scheduler.commit(&plan, Fence(sequence)).unwrap();
        pack_inputs(&mut inputs, &config, &scheduler, &plan, &admissions).unwrap();
        if plan.kind == GraphKind::PackedSpeculative {
            let uploads = inputs.uploads();
            assert!(!uploads.iter().any(|u| u.buffer == "query_lengths"));
            let words = |name: &str| uploads.iter().find(|u| u.buffer == name).unwrap().bytes
                .chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            let slots = words("row_slot_ids");
            assert_eq!(&slots[..plan.lanes.len()], &[0, 2, 3]);
            assert_eq!(slots[3], -1);
            let ceilings = words("speculative_prefix_limits");
            assert_eq!(&ceilings[..plan.lanes.len()], &plan.lanes.iter().map(|l| l.tokens as i32).collect::<Vec<_>>());
            assert_eq!(ceilings[3], 0);
            assert!(plan.lanes.iter().map(|l| l.tokens).sum::<usize>() > plan.packed_token_capacity,
                "host reservations cover per-lane worst cases, not sampled allocation");
            return;
        }
        let mut records = [CompletionRecord::default(); 4];
        for lane in &plan.lanes {
            let live = scheduler.request(lane.request).unwrap();
            let samples = plan.kind == GraphKind::Prefill;
            records[lane.request.slot as usize] = CompletionRecord { generation: lane.request.generation as i64,
                submission_sequence: sequence as i64,
                consumed_tokens: if samples { 64 } else { live.consumed_tokens as i64 },
                generated_tokens: if samples { 1 } else { live.generated_tokens as i64 },
                token_id: if samples { 7 } else { -1 }, status: 1, ..Default::default() };
        }
        scheduler.complete(Fence(sequence), &records, Default::default()).unwrap();
    }
    panic!("lifecycle did not reach packed combined decoding");
}
