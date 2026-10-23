//! Host-only fixtures for transport, scheduling, and contract validation.

use super::{ModelConfig, VisionCapabilities, contract::*};
use crate::execution::{
    BufferDescriptor, Dtype, ExecutionManifest, GraphDescriptor, GraphKind, InputBinding,
    SamplingMode,
};

pub fn binding(
    buffer: &str,
    source: ValueSource,
    indexing: InputIndexing,
    value_offset: i64,
    padding: i64,
) -> ValueBinding {
    ValueBinding {
        buffer: buffer.into(),
        source,
        indexing,
        value_offset,
        padding,
        tensor: String::new(),
    }
}

pub fn metadata() -> PreparedContract {
    use InputIndexing::*;
    use ValueSource::*;
    let row = StateView {
        buffer: "state_pool".into(),
        offset: 0,
        stride: 32,
        nbytes: 32,
    };
    let pending = StateView {
        buffer: "pending".into(),
        offset: 0,
        stride: 8,
        nbytes: 8,
    };
    let mut graphs = Vec::new();
    for batch in [1, 2, 4] {
        for kind in [
            GraphKind::Init,
            GraphKind::Restore,
            GraphKind::Snapshot,
            GraphKind::Resume,
            GraphKind::Decode,
            GraphKind::Speculative,
            GraphKind::Prefill,
        ] {
            for width in 1..=if kind == GraphKind::Speculative { 8 } else { 1 } {
                let mut inputs = vec![binding(
                    "request_state_rows",
                    LiveStateRows,
                    RequestRow,
                    0,
                    0,
                )];
                if matches!(kind, GraphKind::Init | GraphKind::Restore) {
                    for (name, source) in [
                        ("admission_slots", RequestSlot),
                        ("admission_generations", RequestGeneration),
                        ("admission_seed_bits", SamplingSeed),
                        ("admission_parameters", SamplingParameters),
                        ("admission_top_k", SamplingTopK),
                        ("admission_max_new_tokens", MaxNewTokens),
                        ("admission_eos_token_ids", EosTokenIds),
                        ("admission_eos_count", EosCount),
                        ("admission_histogram", PromptHistogram),
                        ("admission_grammar_mask", GrammarMask),
                        ("admission_grammar_enabled", GrammarEnabled),
                    ] {
                        inputs.push(binding(
                            name,
                            source,
                            if name == "admission_slots" || name == "admission_generations" {
                                SubmissionRow
                            } else {
                                ActiveRow
                            },
                            0,
                            if source == RequestSlot { -1 } else { 0 },
                        ));
                    }
                    inputs.push(binding("admission_count", LiveRows, Scalar, 0, 0));
                } else {
                    inputs.extend([
                        binding("row_slot_ids", RequestSlot, SubmissionRow, 0, -1),
                        binding("row_generations", RequestGeneration, SubmissionRow, 0, 0),
                        binding("live_rows", LiveRows, Scalar, 0, 0),
                    ]);
                }
                if matches!(
                    kind,
                    GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot
                ) {
                    inputs.extend([
                        binding("state_request_slots", RequestSlot, SubmissionRow, 0, -1),
                        binding(
                            "state_source_slots",
                            if kind == GraphKind::Restore {
                                PrefixStateRow
                            } else {
                                LiveStateRow
                            },
                            SubmissionRow,
                            0,
                            0,
                        ),
                        binding(
                            "state_destination_slots",
                            if kind == GraphKind::Snapshot {
                                SnapshotStateRow
                            } else {
                                LiveStateRow
                            },
                            SubmissionRow,
                            0,
                            0,
                        ),
                        binding(
                            "state_expected_consumed",
                            if kind == GraphKind::Snapshot {
                                SnapshotConsumedTokens
                            } else {
                                ConsumedTokens
                            },
                            SubmissionRow,
                            0,
                            -1,
                        ),
                        binding(
                            "state_operation",
                            Constant,
                            SubmissionRow,
                            i64::from(kind == GraphKind::Init),
                            0,
                        ),
                        binding(
                            "state_qsa_slots",
                            if kind == GraphKind::Snapshot {
                                Constant
                            } else {
                                RequestSlot
                            },
                            SubmissionRow,
                            if kind == GraphKind::Snapshot { -1 } else { 1 },
                            -1,
                        ),
                        binding(
                            "state_accepted_tokens",
                            Constant,
                            SubmissionRow,
                            i64::from(kind != GraphKind::Snapshot),
                            0,
                        ),
                        binding("state_num_ops", LiveRows, Scalar, 0, 0),
                    ]);
                }
                if kind != GraphKind::Init {
                    inputs.push(binding("page_tables", PageTable, RequestRow, 0, 0));
                }
                if kind.samples() {
                    inputs.extend([
                        binding("sample_enabled", SampleEnabled, SubmissionRow, 0, 0),
                        binding("sampling_grammar_mask", GrammarMask, RequestRow, 0, 0),
                    ]);
                }
                if matches!(kind, GraphKind::Prefill | GraphKind::Speculative) {
                    inputs.push(binding("query_lengths", QueryLength, SubmissionRow, 0, 0));
                }
                if kind == GraphKind::Prefill {
                    inputs.extend([
                        binding("token_ids", PromptToken, PackedToken, 0, 0),
                        binding("prefill_positions", PromptPosition, PackedToken, 0, 0),
                        binding("vision_token_indices", ImageFeatureIndex, PackedToken, 0, 0),
                        binding("vision_feature_slot", ImageFeatureSlot, Scalar, 0, -1),
                    ]);
                }
                graphs.push(GraphContract {
                    graph: format!("{kind:?}-{batch}-{width}"),
                    kind: kind.try_into().unwrap(),
                    inputs,
                    state: vec![
                        StateAccess {
                            view: row.clone(),
                            access: StateAccessKind::ReadWrite,
                        },
                        StateAccess {
                            view: pending.clone(),
                            access: StateAccessKind::ReadWrite,
                        },
                    ],
                    state_rows_per_request: 1,
                    draft_cursor: (kind == GraphKind::Speculative).then(|| CompletionCursor {
                        state: row.clone(),
                        lag_tokens: 1,
                    }),
                    embedding_tables: vec![],
                    pending_token: matches!(kind, GraphKind::Decode | GraphKind::Speculative)
                        .then(|| pending.clone()),
                });
            }
        }
    }
    PreparedContract {
        version: 1,
        vocabulary: 248320,
        input_vocabulary: 248320,
        vision: VisionCapabilities::disabled(),
        cache: CacheContract {
            page_tokens: 64,
            regions: vec![],
            page_tables: vec![PageTableView {
                buffer: "page_tables".into(),
                page_offset: 0,
            }],
        },
        graphs,
        frontend: FrontendContract {
            tools: None,
            reasoning: None,
            image_preparation: None,
            image_layout: None,
            rejected_tokens: vec![],
            position_dimensions: 3,
            assistant_marker: "<assistant>".into(),
            message_end_markers: vec![],
            prefix_boundary: None,
            image_tokens: None,
        },
    }
}

pub fn manifest(config: &ModelConfig) -> ExecutionManifest {
    let mut specs = std::collections::BTreeMap::new();
    let slots = config.limits.request_capacity();
    for graph in &config.contract.graphs {
        for b in &graph.inputs {
            use ValueSource::*;
            let width = match b.source {
                SamplingParameters => 6,
                EosTokenIds | LiveStateRows => 8,
                PromptHistogram => config.contract.vocabulary,
                GrammarMask => config.contract.vocabulary.div_ceil(32),
                PageTable => config.prepared_sequence_tokens / config.contract.cache.page_tokens,
                CopyTokenIds => 7,
                CopyGrammarMask => 8 * config.contract.vocabulary.div_ceil(32),
                PromptPosition => 3,
                _ => 1,
            };
            let shape = match b.indexing {
                InputIndexing::Scalar => vec![1],
                InputIndexing::PackedToken => {
                    if width == 1 {
                        vec![config.limits.prefill_chunk_size]
                    } else {
                        vec![config.limits.prefill_chunk_size, width]
                    }
                }
                _ => {
                    if width == 1 {
                        vec![slots]
                    } else {
                        vec![slots, width]
                    }
                }
            };
            let dtype = match b.source {
                SamplingParameters => Dtype::Float32,
                RequestSlot if b.buffer == "row_slot_ids" || b.buffer == "admission_slots" => {
                    Dtype::Int32
                }
                CopyTokenCount | CopyGrammarMask | CopyGrammarCount | QueryLength | LiveRows
                | SampleEnabled | SamplingTopK | EosCount | PromptHistogram | GrammarMask
                | GrammarEnabled | ImageFeatureIndex | ImageFeatureSlot | PageTable => Dtype::Int32,
                Constant
                    if matches!(
                        b.buffer.as_str(),
                        "state_operation" | "state_accepted_tokens"
                    ) =>
                {
                    Dtype::Int32
                }
                _ => Dtype::Int64,
            };
            specs.insert(b.buffer.clone(), (dtype, shape));
        }
    }
    specs.extend([
        ("frame_index".into(), (Dtype::Int32, vec![1])),
        ("submission_sequence".into(), (Dtype::Int64, vec![1])),
        (
            "completion_records".into(),
            (Dtype::Int64, vec![2, slots, 24]),
        ),
        ("state_pool".into(), (Dtype::Uint8, vec![1024, 32])),
        ("pending".into(), (Dtype::Int64, vec![slots])),
    ]);
    let (mut address, mut host) = (0x10000, 0);
    let mut offsets = std::collections::BTreeMap::new();
    let buffers = specs
        .into_iter()
        .map(|(name, (dtype, shape))| {
            let nbytes = shape.iter().product::<usize>() * dtype.size();
            let mut stride = 1;
            let mut strides: Vec<_> = shape
                .iter()
                .rev()
                .map(|n| {
                    let s = stride;
                    stride *= n;
                    s
                })
                .collect();
            strides.reverse();
            let result = BufferDescriptor {
                name: name.clone(),
                address,
                nbytes,
                dtype,
                shape,
                strides,
            };
            offsets.insert(name, (host, nbytes));
            address += nbytes.next_multiple_of(8);
            host += nbytes.next_multiple_of(8);
            result
        })
        .collect();
    let graphs = config
        .contract
        .graphs
        .iter()
        .map(|g| {
            let fields: Vec<_> = g.graph.split('-').collect();
            let kind = match g.kind {
                GraphOperation::Init => GraphKind::Init,
                GraphOperation::Restore => GraphKind::Restore,
                GraphOperation::Snapshot => GraphKind::Snapshot,
                GraphOperation::Prefill => GraphKind::Prefill,
                GraphOperation::ScoredPrefill => GraphKind::ScoredPrefill,
                GraphOperation::Decode => GraphKind::Decode,
                GraphOperation::Resume => GraphKind::Resume,
                GraphOperation::Speculative => GraphKind::Speculative,
                _ => unreachable!(),
            };
            GraphDescriptor { draft_tokens: 0, packed_token_capacity: 0,
                name: g.graph.clone(),
                kind,
                sampling_mode: SamplingMode::General,
                handle: 1,
                batch_capacity: fields[1].parse().unwrap(),
                token_capacity: if kind == GraphKind::Prefill {
                    config.limits.prefill_chunk_size
                } else {
                    fields[2].parse().unwrap()
                },
                prefill_capacity: 0,
                estimated_duration_ns: 1000,
                input_bindings: ["frame_index", "submission_sequence"]
                    .into_iter()
                    .chain(g.inputs.iter().map(|b| b.buffer.as_str()))
                    .map(|name| InputBinding {
                        buffer_name: name.into(),
                        host_offset: offsets[name].0,
                        nbytes: offsets[name].1,
                    })
                    .collect(),
            }
        })
        .collect();
    ExecutionManifest {
        abi_version: 3,
        context: 1,
        device: 0,
        max_active: slots,
        input_frame_bytes: host,
        buffers,
        graphs,
        cache: None,
        collectives: None,
        engram: None,
        offload: None,
    }
}
