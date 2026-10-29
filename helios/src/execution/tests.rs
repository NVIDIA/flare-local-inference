use super::*;

#[test]
fn optional_readiness_output_preserves_uninstrumented_manifests_and_rejects_invalid_layouts() {
    let mut value = manifest();
    value.validate().unwrap();
    value.buffers.push(BufferDescriptor { name: "readiness_timestamps".into(), address: 0x10000,
        nbytes: 96, dtype: Dtype::Int64, shape: vec![2, 6], strides: vec![6, 1] });
    value.validate().unwrap();
    value.buffers.last_mut().unwrap().shape = vec![2, 2, 3];
    value.buffers.last_mut().unwrap().strides = vec![6, 3, 1];
    assert!(value.validate().is_err());
    value.buffers.last_mut().unwrap().shape = vec![2, 6];
    value.buffers.last_mut().unwrap().strides = vec![6, 1];
    value.buffers.last_mut().unwrap().dtype = Dtype::Float32;
    assert!(value.validate().is_err());
}

#[test]
fn optional_speculative_survival_validates_frame_layout() {
    let mut value = manifest();
    value.abi_version = ABI_VERSION;
    value.buffers[2].shape = vec![2, 4, 24];
    value.buffers[2].strides = vec![96, 24, 1];
    value.buffers[2].nbytes = 1536;
    value.validate().unwrap();
    value.buffers.push(BufferDescriptor { name: "completion_speculative_survival".into(),
        address: 0x10000, nbytes: 224, dtype: Dtype::Float32,
        shape: vec![2, 4, 7], strides: vec![28, 7, 1] });
    value.validate().unwrap();
    for (shape, dtype) in [(vec![1, 8, 7], Dtype::Float32), (vec![2, 2, 14], Dtype::Float32),
        (vec![2, 4, 7], Dtype::Int32)] {
        let mut invalid = value.clone();
        let buffer = invalid.buffers.last_mut().unwrap();
        buffer.shape = shape;
        buffer.dtype = dtype;
        assert!(invalid.validate().is_err());
    }
    value.graphs[0].input_bindings.push(InputBinding {
        buffer_name: "completion_speculative_survival".into(), host_offset: 0, nbytes: 224 });
    assert!(value.validate().is_err());
}

#[test]
fn fp8_cache_buffers_use_single_byte_elements() {
    let mut value = manifest();
    let dtype = Dtype::try_from("float8_e4m3fn").unwrap();
    assert_eq!(dtype, Dtype::Float8E4m3fn);
    assert_eq!(dtype.size(), 1);
    value.buffers.push(BufferDescriptor {
        name: "kv_pages".into(),
        address: 0x10000,
        nbytes: 1024,
        dtype,
        shape: vec![4, 64, 4],
        strides: vec![256, 4, 1],
    });
    value.validate().unwrap();
    value.buffers.last_mut().unwrap().nbytes *= 2;
    assert!(value.validate().is_err());
}

#[test]
fn tensor_parallel_capacity_rejects_more_than_sixteen_ranks() {
    let pairs: Vec<_> = (0..16)
        .map(|_| {
            let (local, peer) = crate::execution::rank_stream::test_pair().unwrap();
            (RankStream::from(local), peer)
        })
        .collect();
    let (sockets, _peers): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let result = unsafe { TensorParallelExecutor::new(manifest(), sockets) };
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("at most sixteen")
    );
    let result = unsafe { threaded::ThreadedExecutor::new(manifest(), vec![manifest(); 16]) };
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("at most sixteen")
    );
}

#[test]
fn completion_wire_decodes_every_configured_lane() {
    for capacity in [1, 2, 8, 64] {
        for fields in [24] {
            let mut bytes = vec![0u8; capacity * fields * 8];
            let last = (capacity - 1) * fields * 8;
            for (field, value) in [(0, 7i64), (1, 9), (4, -1), (5, 1)] {
                bytes[last + field * 8..last + (field + 1) * 8]
                    .copy_from_slice(&value.to_le_bytes());
            }
            if fields == 24 {
                for (field, value) in [(12, -1i64), (15, 1)] {
                    bytes[last + field * 8..last + (field + 1) * 8]
                        .copy_from_slice(&value.to_le_bytes());
                }
                for field in 16..24 {
                    bytes[last + field * 8..last + (field + 1) * 8]
                        .copy_from_slice(&(-1i64).to_le_bytes());
                }
            }
            let records = transport::decode_records(&bytes, Fence(9), capacity, fields).unwrap();
            assert_eq!(records.len(), capacity);
            assert_eq!(records[capacity - 1].generation, 7);
            assert!(
                transport::decode_records(&bytes[..bytes.len() - 8], Fence(9), capacity, fields)
                    .is_err()
            );
        }
    }
}
use crate::cache::Fence;
use crate::{Error, ExecutionError};

#[test]
fn collective_overview_uses_graph_geometry_and_individual_bindings() {
    let mut value = manifest();
    let mut packed = value.graphs[0].clone();
    packed.name = "arbitrary-target-name".into();
    packed.kind = GraphKind::Speculative;
    packed.batch_capacity = 4;
    packed.token_capacity = 6;
    packed.packed_token_capacity = 16;
    packed.draft_tokens = 5;
    packed.tuned_bindings = vec![TunedBinding { buffer: "target-choice".into(), tuning_graph: "choose".into() }];
    let mut mixed = packed.clone();
    mixed.name = "arbitrary-mixed-name".into();
    mixed.kind = GraphKind::Mixed;
    mixed.token_capacity = 1;
    mixed.packed_token_capacity = 0;
    mixed.prefill_capacity = 128;
    mixed.draft_tokens = 0;
    mixed.tuned_bindings[0].buffer = "mixed-choice".into();
    value.graphs[0].name = "arbitrary-decode-name".into();
    value.graphs[0].tuned_bindings = packed.tuned_bindings.clone();
    value.graphs.extend([packed, mixed]);
    let choices = vec![vec![
        ("target-choice".into(), ((5u32 << 16) | 1).to_ne_bytes().to_vec()),
        ("mixed-choice".into(), ((6u32 << 16) | 64).to_ne_bytes().to_vec()),
    ]];
    let text = super::transport::collective_choices_overview(&value, &choices);
    assert!(text.contains("Mode 5: 1 binding · Mode 6: 1 binding\n"));
    assert!(text.contains("arbitrary-decode-name · Decode b=4 t=1 rows=4 drafts=0 prefill=0 · Mode 5 (1 CTAs)\n"));
    assert!(text.contains("arbitrary-target-name · Speculative b=4 t=6 rows=16 drafts=5 prefill=0 · Mode 5 (1 CTAs)\n"));
    assert!(text.contains("arbitrary-mixed-name · Mixed b=4 t=1 rows=131 drafts=0 prefill=128 · Mode 6 (64 CTAs)\n"));
}

pub(super) fn manifest() -> ExecutionManifest {
    let buffers = vec![
        BufferDescriptor {
            name: "frame_index".into(),
            address: 0x1000,
            nbytes: 4,
            dtype: Dtype::Int32,
            shape: vec![1],
            strides: vec![1],
        },
        BufferDescriptor {
            name: "submission_sequence".into(),
            address: 0x2000,
            nbytes: 8,
            dtype: Dtype::Int64,
            shape: vec![1],
            strides: vec![1],
        },
        BufferDescriptor {
            name: "completion_records".into(),
            address: 0x3000,
            nbytes: 1536,
            dtype: Dtype::Int64,
            shape: vec![2, 4, 24],
            strides: vec![96, 24, 1],
        },
        BufferDescriptor {
            name: "tokens".into(),
            address: 0x4000,
            nbytes: 32,
            dtype: Dtype::Int64,
            shape: vec![4],
            strides: vec![1],
        },
    ];
    let graph = GraphDescriptor { tuned_bindings: vec![], confidence_width: 0, draft_tokens: 0, packed_token_capacity: 0,
        name: "decode4".into(),
        kind: GraphKind::Decode,
        sampling_mode: SamplingMode::General,
        handle: 0x5000,
        batch_capacity: 4,
        token_capacity: 1,
        prefill_capacity: 0,
        estimated_duration_ns: 1_000_000,
        input_bindings: vec![
            InputBinding {
                buffer_name: "frame_index".into(),
                host_offset: 0,
                nbytes: 4,
            },
            InputBinding {
                buffer_name: "submission_sequence".into(),
                host_offset: 8,
                nbytes: 8,
            },
            InputBinding {
                buffer_name: "tokens".into(),
                host_offset: 16,
                nbytes: 32,
            },
        ],
    };
    ExecutionManifest { tuning_graphs: vec![],
        engram: None,
        collectives: None,
        cache: None,
        offload: None,
        abi_version: ABI_VERSION,
        context: 1,
        device: 0,
        max_active: 4,
        input_frame_bytes: 48,
        buffers,
        graphs: vec![graph],
    }
}

#[test]
fn execution_abi_and_unsupported_sampling_modes_reject_before_cuda() {
    let mut value = manifest();
    value.abi_version = 1;
    assert!(matches!(
        value.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    value.abi_version = ABI_VERSION;
    value.validate().unwrap();
    assert!(SamplingMode::try_from("greedy").is_err());
    assert!(SamplingMode::try_from("automatic").is_err());
}

#[test]
fn required_completion_and_input_layouts_are_checked_before_cuda() {
    let original = manifest();
    original.validate().unwrap();
    let mut changed = original.clone();
    changed.buffers[2].shape = vec![4, 2, 8];
    changed.buffers[2].strides = vec![16, 8, 1];
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    changed = original.clone();
    changed.buffers[3].address = 0x3008;
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    changed = original.clone();
    changed.buffers[3].strides = vec![2];
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    changed = original.clone();
    changed.buffers[3].address = usize::MAX - 8;
    assert!(changed.validate().is_err());
    changed = original;
    changed.graphs[0]
        .input_bindings
        .retain(|binding| binding.buffer_name != "frame_index");
    assert!(matches!(
        changed.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
}

#[test]
fn graph_variants_share_one_nonoverlapping_input_layout() {
    let mut manifest = manifest();
    let mut graph = manifest.graphs[0].clone();
    graph.name = "decode2".into();
    graph.handle += 1;
    graph.batch_capacity = 2;
    manifest.graphs.push(graph);
    manifest.validate().unwrap();
    manifest.graphs[1].input_bindings[2].host_offset = 8;
    assert!(matches!(
        manifest.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
    manifest.graphs.pop();
    manifest.graphs[0].input_bindings[2].host_offset = 8;
    assert!(matches!(
        manifest.validate().unwrap_err(),
        Error::Execution(ExecutionError::Manifest(_))
    ));
}

#[test]
fn prefill_transport_alignment_is_independent_of_model_context_limit() {
    let mut manifest = manifest();
    manifest.graphs[0].kind = GraphKind::Prefill;
    manifest.graphs[0].batch_capacity = 1;
    for capacity in [1, 16, 64, 128, 256, 512, 3072, 4096, 8192, 262144, 262208] {
        manifest.graphs[0].token_capacity = capacity;
        manifest.validate().unwrap();
    }
    for capacity in [0, 2, 15, 32, 63, 65, 513, 4095, usize::MAX] {
        manifest.graphs[0].token_capacity = capacity;
        assert!(manifest.validate().is_err());
    }
}

#[test]
fn partial_uploads_reject_overlaps_reserved_inputs_and_misalignment() {
    let manifest = manifest();
    let mut storage = Vec::new();
    transport::validate_uploads(
        &manifest,
        0,
        &[Upload {
            buffer: "tokens",
            offset: 8,
            bytes: &[1; 8],
        }],
        &mut storage,
    )
    .unwrap();
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[Upload {
                buffer: "tokens",
                offset: 1,
                bytes: &[1; 8]
            }],
            &mut storage,
        )
        .is_err()
    );
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[Upload {
                buffer: "tokens",
                offset: 32,
                bytes: &[1; 8]
            }],
            &mut storage,
        )
        .is_err()
    );
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[Upload {
                buffer: "frame_index",
                offset: 0,
                bytes: &[0; 4]
            }],
            &mut storage,
        )
        .is_err()
    );
    assert!(
        transport::validate_uploads(
            &manifest,
            0,
            &[
                Upload {
                    buffer: "tokens",
                    offset: 0,
                    bytes: &[1; 16]
                },
                Upload {
                    buffer: "tokens",
                    offset: 8,
                    bytes: &[2; 8]
                }
            ],
            &mut storage,
        )
        .is_err()
    );
}

#[test]
fn upload_and_completion_failures_keep_distinct_domains() {
    let manifest = manifest();
    let mut storage = Vec::new();
    let uploads = [Upload {
        buffer: "tokens",
        offset: 1,
        bytes: &[1; 8],
    }];
    let error = match transport::validate_uploads(&manifest, 0, &uploads, &mut storage) {
        Err(error) => error,
        Ok(_) => panic!("misaligned upload was accepted"),
    };
    assert!(matches!(error, Error::Execution(ExecutionError::Upload(_))));
    assert_eq!(
        error.to_string(),
        "unaligned or out-of-range upload: tokens"
    );
    assert!(storage.is_empty());

    let error = transport::decode_records(&[0; 8], Fence(1), 4, 8).unwrap_err();
    assert!(matches!(
        error,
        Error::Execution(ExecutionError::Completion(_))
    ));
    assert_eq!(error.to_string(), "completion extent mismatch");
}

#[test]
fn physical_lane_records_reject_stale_sequences_and_impossible_status() {
    let mut records = [[0i64; VECTOR_RECORD_FIELDS]; 4];
    records[3][..8].copy_from_slice(&[7, 9, 65, 2, 88, 1, 0, 0]);
    records[3][8] = 1;
    records[3][12] = -1;
    records[3][15] = 1;
    records[3][16..].fill(-1);
    records[3][16] = 88;
    let bytes = |records: [[i64; VECTOR_RECORD_FIELDS]; 4]| {
        records.into_iter().flatten().flat_map(i64::to_le_bytes).collect::<Vec<_>>()
    };
    let decoded = transport::decode_records(&bytes(records), Fence(9), 4, VECTOR_RECORD_FIELDS).unwrap();
    assert_eq!(decoded[3].generation, 7);
    assert_eq!(decoded[3].token_id, 88);
    assert_eq!(decoded[0].status, 0);
    assert!(transport::decode_records(&bytes(records), Fence(10), 4, VECTOR_RECORD_FIELDS).is_err());
    records[3][0] = 0;
    assert!(transport::decode_records(&bytes(records), Fence(9), 4, VECTOR_RECORD_FIELDS).is_err());
    records[3][0] = 7;
    records[3][5] = 4;
    assert!(transport::decode_records(&bytes(records), Fence(9), 4, VECTOR_RECORD_FIELDS).is_err());
    assert!(transport::decode_records(&[0; 4 * 8 * 8], Fence(9), 4, 8).is_err());
}

#[test]
fn speculative_requires_vector_storage_and_explicit_target_geometry() {
    let mut manifest = manifest();
    manifest.abi_version = ABI_VERSION;
    manifest.buffers[2].shape = vec![2, 4, 24];
    manifest.buffers[2].strides = vec![96, 24, 1];
    manifest.buffers[2].nbytes = 1536;
    manifest.graphs[0].kind = GraphKind::Speculative;
    for batch in [1, 2, 4] {
        manifest.graphs[0].batch_capacity = batch;
        for depth in 0..=7 {
            manifest.graphs[0].draft_tokens = depth;
            for width in 1..=depth + 1 {
                manifest.graphs[0].token_capacity = width;
                if width == 1 && depth != 0 {
                    assert!(manifest.validate().is_err());
                    continue;
                }
                for target in batch..=batch * width {
                    manifest.graphs[0].packed_token_capacity = target;
                    manifest.validate().unwrap();
                }
            }
        }
    }
    let mut invalid = manifest.clone();
    invalid.graphs[0].packed_token_capacity = 0;
    assert!(invalid.validate().is_err());
    let mut invalid = manifest.clone();
    invalid.graphs[0].packed_token_capacity = 33;
    assert!(invalid.validate().is_err());
    let mut invalid = manifest.clone();
    invalid.graphs[0].token_capacity = 9;
    assert!(invalid.validate().is_err());
    manifest.abi_version = 3;
    assert!(manifest.validate().is_err());
}

#[test]
fn speculative_operation_consumes_embeddings_and_rejects_removed_graph_names() {
    assert_eq!(GraphKind::try_from("speculative").unwrap(), GraphKind::Speculative);
    assert!(crate::engram::consumes(GraphKind::Speculative));
    assert!(crate::model::contract::GraphOperation::try_from(GraphKind::Speculative).is_ok());
    for name in ["proposal", "verification", "packed_speculative"] {
        assert!(GraphKind::try_from(name).is_err());
    }
}

#[test]
fn vector_parser_rejects_counts_tails_and_stale_epochs_without_cuda() {
    let mut vector = TokenVector {
        emitted_count: 3,
        offered_drafts: 7,
        accepted_drafts: 3,
        copy_offered_mask: 3,
        model_reserved: 65,
        draft_cache_consumed: 66,
        verification_width: 8,
        ..Default::default()
    };
    vector.tokens[..3].copy_from_slice(&[40, 41, 42]);
    let good = CompletionRecord {
        generation: 3,
        submission_sequence: 9,
        consumed_tokens: 67,
        generated_tokens: 4,
        token_id: 42,
        status: 2,
        vector: Some(vector),
        ..Default::default()
    };
    let encoded = |row: CompletionRecord| {
        let mut bytes = vec![0; 4 * VECTOR_RECORD_FIELDS * 8];
        for (dst, field) in bytes[..VECTOR_RECORD_FIELDS * 8]
            .chunks_exact_mut(8)
            .zip(row.fields())
        {
            dst.copy_from_slice(&field.to_le_bytes());
        }
        bytes
    };
    assert_eq!(
        transport::decode_records(&encoded(good), Fence(9), 4, VECTOR_RECORD_FIELDS).unwrap()[0],
        good
    );
    for change in 0..8 {
        let mut row = good;
        let v = row.vector.as_mut().unwrap();
        match change {
            0 => v.emitted_count = 9,
            1 => v.tokens[3] = 1,
            2 => v.accepted_drafts = 1,
            3 => v.offered_drafts = 8,
            4 => v.copy_offered_mask = 128,
            5 => v.draft_cache_consumed = -2,
            6 => row.token_id = 40,
            _ => row.submission_sequence = 8,
        }
        assert!(
            transport::decode_records(&encoded(row), Fence(9), 4, VECTOR_RECORD_FIELDS).is_err(),
            "change{change}"
        );
    }
}

#[test]
fn resolved_uploads_reject_unknown_missing_duplicate_and_invalid_ranges() {
    let manifest = manifest();
    manifest.validate().unwrap();
    let mut storage = Vec::new();
    let valid = Upload {
        buffer: "tokens",
        offset: 8,
        bytes: &[1; 8],
    };
    for uploads in [
        vec![Upload {
            buffer: "missing",
            offset: 0,
            bytes: &[0; 8],
        }],
        vec![Upload {
            buffer: "completion_records",
            offset: 0,
            bytes: &[0; 8],
        }],
        vec![Upload {
            buffer: "submission_sequence",
            offset: 0,
            bytes: &[0; 8],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &[],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &[0; 7],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: usize::MAX - 7,
            bytes: &[0; 16],
        }],
        vec![Upload {
            buffer: "tokens",
            offset: 24,
            bytes: &[0; 16],
        }],
        vec![
            Upload {
                buffer: "tokens",
                offset: 8,
                bytes: &[0; 8],
            },
            Upload {
                buffer: "tokens",
                offset: 8,
                bytes: &[0; 8],
            },
        ],
        vec![
            Upload {
                buffer: "tokens",
                offset: 0,
                bytes: &[0; 16],
            },
            Upload {
                buffer: "tokens",
                offset: 8,
                bytes: &[0; 16],
            },
        ],
    ] {
        assert!(transport::validate_uploads(&manifest, 0, &uploads, &mut storage).is_err());
    }
    assert!(transport::validate_uploads(&manifest, 1, &[], &mut storage).is_err());
    let mut missing = manifest.clone();
    missing.buffers.retain(|buffer| buffer.name != "tokens");
    assert!(transport::validate_uploads(&missing, 0, &[valid], &mut storage).is_err());
    missing = manifest.clone();
    missing.graphs[0]
        .input_bindings
        .retain(|binding| binding.buffer_name != "frame_index");
    assert!(transport::validate_uploads(&missing, 0, &[], &mut storage).is_err());

    // Failure can leave scratch descriptors, but the next validation clears them.
    let accepted = [Upload {
        buffer: "tokens",
        offset: 24,
        bytes: &[5; 8],
    }];
    let resolved = transport::validate_uploads(&manifest, 0, &accepted, &mut storage).unwrap();
    assert_eq!(
        resolved.copies().collect::<Vec<_>>(),
        vec![
            transport::ResolvedUpload {
                host_offset: 40,
                device_address: 0x4018,
                nbytes: 8
            },
            transport::ResolvedUpload {
                host_offset: 0,
                device_address: 0x1000,
                nbytes: 4
            },
            transport::ResolvedUpload {
                host_offset: 8,
                device_address: 0x2000,
                nbytes: 8
            },
        ]
    );
}

#[test]
fn failed_validation_never_changes_a_staged_frame() {
    let manifest = manifest();
    let mut storage = Vec::new();
    let mut frame = vec![0xA5; manifest.input_frame_bytes];
    let before = frame.clone();
    let uploads = [
        Upload {
            buffer: "tokens",
            offset: 0,
            bytes: &[1; 8],
        },
        Upload {
            buffer: "tokens",
            offset: 4,
            bytes: &[2; 8],
        },
    ];
    if let Ok(resolved) = transport::validate_uploads(&manifest, 0, &uploads, &mut storage) {
        resolved.stage(&mut frame, 0, 1);
        panic!("invalid command reached frame staging");
    }
    assert_eq!(frame, before);
    // An omitted upload is valid; only executor-owned input words are staged.
    let resolved = transport::validate_uploads(&manifest, 0, &[], &mut storage).unwrap();
    resolved.stage(&mut frame, 1, 2);
    assert_eq!(&frame[..4], &1i32.to_le_bytes());
    assert_eq!(&frame[8..16], &2i64.to_le_bytes());
    assert_eq!(&frame[4..8], &before[4..8]);
    assert_eq!(&frame[16..], &before[16..]);
    assert_eq!(resolved.copies().count(), 2);
}

#[test]
fn reusable_resolution_preserves_two_frames_and_exact_partial_copy_ranges() {
    let manifest = manifest();
    manifest.validate().unwrap();
    let mut storage = Vec::with_capacity(12);
    let pointer = storage.as_ptr();
    let capacity = storage.capacity();
    let mut frames = [
        vec![0xA5; manifest.input_frame_bytes],
        vec![0x5A; manifest.input_frame_bytes],
    ];
    let mut device = vec![0xCC; 0x4020];
    let first = {
        let bytes = 0x0123_4567_89AB_CDEFi64.to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 8,
            bytes: &bytes,
        }];
        let resolved = transport::validate_uploads(&manifest, 0, &uploads, &mut storage).unwrap();
        resolved.stage(&mut frames[0], 0, 10);
        let copies = resolved.copies().collect::<Vec<_>>();
        assert_eq!(&frames[0][24..32], &bytes);
        copies
    };
    let staged = frames[0].clone();
    {
        let bytes = (-77i64).to_le_bytes();
        let uploads = [Upload {
            buffer: "tokens",
            offset: 24,
            bytes: &bytes,
        }];
        let resolved = transport::validate_uploads(&manifest, 0, &uploads, &mut storage).unwrap();
        resolved.stage(&mut frames[1], 1, 11);
        assert_eq!(&frames[1][40..48], &bytes);
    }
    assert_eq!(frames[0], staged);
    assert_eq!(storage.as_ptr(), pointer);
    assert_eq!(storage.capacity(), capacity);
    // The first pinned frame remains sufficient after source bytes and descriptors
    // were replaced. This CPU transfer model checks the actual resolved ranges.
    for binding in first {
        device[binding.device_address..binding.device_address + binding.nbytes]
            .copy_from_slice(&frames[0][binding.host_offset..binding.host_offset + binding.nbytes]);
    }
    assert_eq!(&device[0x1000..0x1004], &0i32.to_le_bytes());
    assert_eq!(&device[0x2000..0x2008], &10i64.to_le_bytes());
    assert_eq!(
        &device[0x4008..0x4010],
        &0x0123_4567_89AB_CDEFi64.to_le_bytes()
    );
    assert_eq!(&device[0x4000..0x4008], &[0xCC; 8]);
    assert_eq!(&device[0x4010..0x4020], &[0xCC; 16]);
    let uploads = [Upload {
        buffer: "tokens",
        offset: 16,
        bytes: &[0x71; 8],
    }];
    let resolved = transport::validate_uploads(&manifest, 0, &uploads, &mut storage).unwrap();
    resolved.stage(&mut frames[0], 0, 12);
    for binding in resolved.copies() {
        device[binding.device_address..binding.device_address + binding.nbytes]
            .copy_from_slice(&frames[0][binding.host_offset..binding.host_offset + binding.nbytes]);
    }
    assert_eq!(
        &device[0x4008..0x4010],
        &0x0123_4567_89AB_CDEFi64.to_le_bytes()
    );
    assert_eq!(&device[0x4010..0x4018], &[0x71; 8]);
    assert_eq!(&device[0x4018..0x4020], &[0xCC; 8]);
    assert_eq!(&device[0x2000..0x2008], &12i64.to_le_bytes());
}

#[test]
fn fragmented_upload_descriptors_reuse_their_highwater_capacity() {
    let manifest = manifest();
    let mut storage = Vec::with_capacity(1);
    let uploads = [24, 0, 16, 8].map(|offset| Upload {
        buffer: "tokens",
        offset,
        bytes: &[0x31; 8],
    });
    let mut frame = vec![0xA5; manifest.input_frame_bytes];
    transport::validate_uploads(&manifest, 0, &uploads, &mut storage)
        .unwrap()
        .stage(&mut frame, 0, 1);
    let pointer = storage.as_ptr();
    let capacity = storage.capacity();
    assert!(capacity >= 4);
    assert_eq!(&frame[16..48], &[0x31; 32]);
    for sequence in 2..258 {
        let count = sequence as usize % 5;
        let resolved =
            transport::validate_uploads(&manifest, 0, &uploads[..count], &mut storage).unwrap();
        assert_eq!(resolved.copies().count(), count + 2);
        resolved.stage(&mut frame, 0, sequence);
        assert_eq!(storage.as_ptr(), pointer);
        assert_eq!(storage.capacity(), capacity);
    }
}

#[test]
fn mixed_manifest_requires_independent_prompt_capacity_and_a_role_input() {
    let mut m = manifest();
    m.abi_version = ABI_VERSION;
    m.buffers[2].shape = vec![2, 4, 24];
    m.buffers[2].strides = vec![96, 24, 1];
    m.buffers[2].nbytes = 1536;
    m.graphs[0].kind = GraphKind::Mixed;
    m.graphs[0].prefill_capacity = 512;
    m.graphs[0].token_capacity = 8;
    assert!(m.validate().is_err());
    m.buffers.push(BufferDescriptor {
        name: "prefill_row".into(),
        address: 0x6000,
        nbytes: 4,
        dtype: Dtype::Int32,
        shape: vec![1],
        strides: vec![1],
    });
    let offset = m.input_frame_bytes;
    m.input_frame_bytes += 8;
    m.graphs[0].input_bindings.push(InputBinding {
        buffer_name: "prefill_row".into(),
        host_offset: offset,
        nbytes: 4,
    });
    m.validate().unwrap();
    assert_eq!(m.graphs[0].packed_tokens(), 536);
    for (batch, width, prefill) in [
        (1, 8, 512),
        (4, 9, 512),
        (4, 8, 0),
        (4, 8, 65),
        (4, 8, 1usize << 31),
    ] {
        let mut bad = m.clone();
        bad.graphs[0].batch_capacity = batch;
        bad.graphs[0].token_capacity = width;
        bad.graphs[0].prefill_capacity = prefill;
        assert!(bad.validate().is_err());
    }
    m.graphs[0].kind = GraphKind::Speculative;
    assert!(m.validate().is_err());
}

pub(super) fn with_logprobs(k: usize) -> ExecutionManifest {
    let mut value = manifest();
    let fields = LOGPROB_PAIR_FIELDS * (1 + k);
    value.buffers.push(BufferDescriptor {
        name: "completion_logprobs".into(), address: 0x10000,
        nbytes: 2 * 4 * MAX_OUTPUT_TOKENS * fields * size_of::<i32>(),
        dtype: Dtype::Int32, shape: vec![2, 4, MAX_OUTPUT_TOKENS, fields],
        strides: vec![4 * MAX_OUTPUT_TOKENS * fields, MAX_OUTPUT_TOKENS * fields, fields, 1],
    });
    value.buffers.push(BufferDescriptor {
        name: "logprob_requested".into(), address: if k == MAX_LOGPROB_TOKEN_IDS { 0x30000 } else { 0x20000 },
        nbytes: 16, dtype: Dtype::Int32, shape: vec![4], strides: vec![1],
    });
    value.graphs[0].input_bindings.push(InputBinding {
        buffer_name: "logprob_requested".into(),
        host_offset: value.input_frame_bytes, nbytes: 16,
    });
    value.input_frame_bytes += 16;
    if k == MAX_LOGPROB_TOKEN_IDS {
        for (name, address, shape, strides) in [
            ("logprob_token_ids", 0x40000, vec![4, k], vec![k, 1]),
            ("logprob_token_id_counts", 0x50000, vec![4], vec![1]),
        ] {
            let nbytes = shape.iter().product::<usize>() * 4;
            value.buffers.push(BufferDescriptor {
                name: name.into(),
                address,
                nbytes,
                dtype: Dtype::Int32,
                shape,
                strides,
            });
            value.graphs[0].input_bindings.push(InputBinding {
                buffer_name: name.into(),
                host_offset: value.input_frame_bytes,
                nbytes,
            });
            value.input_frame_bytes += nbytes;
        }
    }
    value
}

#[test]
fn old_execution_manifests_serve_without_logprob_outputs_and_partial_declarations_fail() {
    let old = manifest();
    old.validate().unwrap();
    assert_eq!(old.max_top_logprobs().unwrap(), 0);
    for k in (1..=MAX_TOP_LOGPROBS).chain([MAX_LOGPROB_TOKEN_IDS]) {
        let current = with_logprobs(k);
        current.validate().unwrap();
        assert_eq!(current.max_top_logprobs().unwrap(), k.min(MAX_TOP_LOGPROBS));
        for name in ["completion_logprobs", "logprob_requested"] {
            let mut partial = current.clone();
            partial.buffers.retain(|b| b.name != name);
            let error = partial.max_top_logprobs().unwrap_err();
            assert!(matches!(&error, Error::Execution(ExecutionError::Manifest(_))));
            assert!(error.to_string().contains(name));
            assert!(partial.validate().is_err());
        }
    }
    for k in [0, MAX_TOP_LOGPROBS + 1] {
        assert!(with_logprobs(k).validate().is_err());
    }
    let mut missing = with_logprobs(MAX_TOP_LOGPROBS);
    missing.graphs[0].input_bindings.pop();
    assert!(missing.validate().is_err());
    for field in 0..4 {
        let mut malformed = with_logprobs(MAX_TOP_LOGPROBS);
        let output = malformed.buffers.iter_mut()
            .find(|b| b.name == "completion_logprobs").unwrap();
        output.shape[field] += 1;
        let error = malformed.max_top_logprobs().unwrap_err();
        assert!(matches!(&error, Error::Execution(ExecutionError::Manifest(_))));
        let message = error.to_string();
        assert!(message.contains("completion_logprobs"));
        assert!(message.contains("expected Int32 shape") && message.contains("got Int32 shape"));
        assert!(malformed.validate().is_err());
    }
}

#[test]
fn logprob_readback_uses_only_emitted_positions_and_validates_chosen_ids() {
    let mut records = [CompletionRecord::default(); 4];
    records[2] = CompletionRecord {
        generation: 9, submission_sequence: 7, consumed_tokens: 3,
        generated_tokens: 2, token_id: 8, status: 1,
        vector: Some(TokenVector {
            emitted_count: 2, offered_drafts: 1, accepted_drafts: 1, verification_width: 2,
            tokens: [7, 8, -1, -1, -1, -1, -1, -1], ..Default::default()
        }),
        ..Default::default()
    };
    for k in 1..=MAX_TOP_LOGPROBS {
        let fields = 2 * (1 + k);
        let mut words = vec![-1i32; 4 * MAX_OUTPUT_TOKENS * fields];
        for (position, token) in [7, 8].into_iter().enumerate() {
            let start = (2 * MAX_OUTPUT_TOKENS + position) * fields;
            let row = &mut words[start..start + fields];
            row[0] = token;
            row[1] = (-3f32).to_bits() as i32;
            row[2] = 5;
            row[3] = (-2f32).to_bits() as i32;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let decoded = transport::decode_logprobs(&bytes, &records, k, fields, 1 << 2, &[]).unwrap();
        assert!(decoded[0].iter().all(Option::is_none));
        assert_eq!(decoded[2][0].as_ref().unwrap().token, 7);
        assert_eq!(decoded[2][1].as_ref().unwrap().logprob, -3.);
        assert_eq!(decoded[2][0].as_ref().unwrap().top.len(), 1);
        assert!(decoded[2][2..].iter().all(Option::is_none));
        records[2].vector.as_mut().unwrap().tokens[0] = 4;
        assert!(transport::decode_logprobs(&bytes, &records, k, fields, 1 << 2, &[]).is_err());
        records[2].vector.as_mut().unwrap().tokens[0] = 7;
    }
}

#[test]
fn prompt_payload_guards_ranks_order_and_frame_extents_are_checked() {
    let row = |token: i32, lp: f32, rank: i32| {
        let mut words = vec![-1i32; PROMPT_FIELDS];
        words[..3].copy_from_slice(&[token, lp.to_bits() as i32, rank]);
        for i in 0..MAX_TOP_LOGPROBS {
            words[3 + 2 * i] = i as i32;
            words[4 + 2 * i] = (-((i + 1) as f32)).to_bits() as i32;
        }
        words
    };
    let guard = row(-1, f32::NAN, 0);
    let actual = row(30, -12., 43);
    let bytes: Vec<_> = guard.iter().chain(&actual).flat_map(|v| v.to_ne_bytes()).collect();
    let result = transport::decode_prompt_logprobs(&bytes, 2).unwrap();
    assert!(result[0].is_none());
    let value = result[1].as_ref().unwrap();
    assert_eq!(value.rank, 43);
    assert_eq!(value.value.token, 30);
    assert_eq!(value.value.top[19], (19, -20.));
    assert!(transport::decode_prompt_logprobs(&bytes[..bytes.len() - 1], 2).is_err());
    assert_eq!(transport::decode_prompt_logprobs(&bytes, 1).unwrap(), vec![None]);
    for bad in [row(30, f32::NAN, 1), row(30, -1., 0), row(-1, -1., 0)] {
        let bytes: Vec<_> = bad.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert!(transport::decode_prompt_logprobs(&bytes, 1).is_err());
    }
}

#[test]
fn older_manifests_have_no_prompt_storage_and_scoring_requires_a_complete_layout() {
    let mut value = manifest();
    assert_eq!(value.prompt_logprob_capacity().unwrap(), 0);
    value.validate().unwrap();
    value.buffers.push(BufferDescriptor { name: "prompt_target_tokens".into(), address: 0x100000,
        nbytes: 512 * 4, dtype: Dtype::Int32, shape: vec![512], strides: vec![1] });
    assert!(value.validate().is_err());
}

#[test]
fn prompt_catalog_storage_and_bindings_are_validated_together() {
    let mut value = with_logprobs(MAX_TOP_LOGPROBS);
    value.buffers.push(BufferDescriptor { name: "prompt_target_tokens".into(), address: 0x300000,
        nbytes: 512 * 4, dtype: Dtype::Int32, shape: vec![512], strides: vec![1] });
    value.buffers.push(BufferDescriptor { name: "prompt_logprobs".into(), address: 0x400000,
        nbytes: 2 * 512 * PROMPT_FIELDS * 4, dtype: Dtype::Int32, shape: vec![2, 512, PROMPT_FIELDS],
        strides: vec![512 * PROMPT_FIELDS, PROMPT_FIELDS, 1] });
    value.graphs[0].kind = GraphKind::ScoredPrefill;
    value.graphs[0].batch_capacity = 1;
    value.graphs[0].token_capacity = 512;
    value.graphs[0].input_bindings.push(InputBinding { buffer_name: "prompt_target_tokens".into(),
        host_offset: value.input_frame_bytes, nbytes: 512 * 4 });
    value.input_frame_bytes += 512 * 4;
    value.validate().unwrap();
    assert_eq!(value.prompt_logprob_capacity().unwrap(), 512);
    let mut narrow = value.clone();
    let output = narrow
        .buffers
        .iter_mut()
        .find(|b| b.name == "completion_logprobs")
        .unwrap();
    output.shape[3] = 4;
    output.strides = vec![4 * MAX_OUTPUT_TOKENS * 4, MAX_OUTPUT_TOKENS * 4, 4, 1];
    output.nbytes = 2 * 4 * MAX_OUTPUT_TOKENS * 4 * 4;
    assert_eq!(narrow.max_top_logprobs().unwrap(), 1);
    assert!(narrow.validate().is_err());
    for name in ["prompt_logprobs", "prompt_target_tokens"] {
        let mut partial = value.clone();
        partial.buffers.retain(|b| b.name != name);
        assert!(partial.validate().is_err());
    }
    let mut malformed = value.clone();
    malformed.graphs[0].token_capacity = 1024;
    assert!(malformed.validate().is_err());
    let mut missing = value.clone();
    missing.graphs[0].input_bindings.retain(|b| b.buffer_name != "prompt_target_tokens");
    assert!(missing.validate().is_err());
    let mut ordinary = value;
    ordinary.graphs[0].kind = GraphKind::Prefill;
    assert!(ordinary.validate().is_err());
}

#[test]
fn packed_combined_peer_agreement_checks_the_complete_committed_vector() {
    let mut vector = TokenVector { verification_width: 6, ..Default::default() };
    vector.offered_drafts = 1;
    let record = CompletionRecord { generation: 1, submission_sequence: 1, consumed_tokens: 65,
        token_id: -1, status: 1, vector: Some(vector), ..Default::default() };
    let completion = || FrameCompletion { speculative_survival: None, prompt_logprobs: None, fence: crate::cache::Fence(1), graph_index: 0,
        records: [record; 4].into(), timing: Default::default(), logprobs: None, proposal_confidence: None };
    for mutation in 0..5 {
        let root = completion(); let mut peer = completion();
        match mutation {
            0 => peer.records[3].vector.as_mut().unwrap().offered_drafts = 2,
            1 => peer.records[3].vector.as_mut().unwrap().accepted_drafts = 1,
            2 => peer.records[3].vector.as_mut().unwrap().verification_width = 5,
            3 => peer.records[3].generated_tokens = 1,
            _ => peer.records[3].token_id = 100,
        }
        assert!(parallel::agree(&root, &peer).is_ok(), "legacy agreement contract remains unchanged");
        assert!(parallel::agree_speculative(&root, &peer).is_err());
    }
    parallel::agree_speculative(&completion(), &completion()).unwrap();
}

#[test]
fn packed_combined_peer_agreement_ignores_only_undefined_inactive_vector_payload() {
    let active = CompletionRecord { generation: 1, submission_sequence: 1, consumed_tokens: 65,
        token_id: -1, status: 1, vector: Some(TokenVector { verification_width: 6, ..Default::default() }),
        ..Default::default() };
    let mut root = FrameCompletion { speculative_survival: None, prompt_logprobs: None, fence: crate::cache::Fence(1), graph_index: 0,
        records: [active; 4].into(), timing: Default::default(), logprobs: None, proposal_confidence: None };
    root.records[3].status = 0;
    let mut peer_records = root.records;
    peer_records[3].generated_tokens = 17;
    peer_records[3].token_id = 400;
    peer_records[3].vector.as_mut().unwrap().offered_drafts = 6;
    peer_records[3].vector.as_mut().unwrap().verification_width = 7;
    peer_records[3].vector.as_mut().unwrap().tokens[7] = 999;
    // Both byte images satisfy the transport ABI; inactive payload is not state.
    for records in [&root.records, &peer_records] {
        let bytes: Vec<_> = records.iter().flat_map(|r| r.fields()).flat_map(i64::to_le_bytes).collect();
        transport::decode_records(&bytes, crate::cache::Fence(1), 4, 24).unwrap();
    }
    let mut peer = FrameCompletion { speculative_survival: None, prompt_logprobs: None, records: peer_records, fence: root.fence, graph_index: root.graph_index,
        timing: Default::default(), logprobs: None, proposal_confidence: None };
    parallel::agree_speculative(&root, &peer).unwrap();
    peer.records[3].generation += 1;
    assert!(parallel::agree_speculative(&root, &peer).is_err(), "base epoch checks remain unchanged");
}

#[test]
fn explicit_logprob_decoding_preserves_order_appends_sample_and_honors_pitch() {
    for count in [1, 20, 21, 127, 128] {
        for fields in [2 + 2 * count.max(MAX_TOP_LOGPROBS), LOGPROB_FIELDS] {
            let mut records = [CompletionRecord::default(); 2];
            records[1].status = 1;
            records[1].token_id = 999;
            let ids: Vec<u32> = (0..count as u32).rev().collect();
            let selections = [vec![], ids.clone()];
            let mut words = vec![-777i32; 2 * MAX_OUTPUT_TOKENS * fields];
            let start = MAX_OUTPUT_TOKENS * fields;
            words[start] = 999;
            words[start + 1] = (-4f32).to_bits() as i32;
            for (i, id) in ids.iter().enumerate() {
                words[start + 2 + 2 * i] = *id as i32;
                words[start + 3 + 2 * i] = (-2f32).to_bits() as i32;
            }
            let decode = |words: &[i32], records: &[CompletionRecord]| {
                let bytes: Vec<_> = words.iter().flat_map(|v| v.to_ne_bytes()).collect();
                transport::decode_logprobs(
                    &bytes,
                    records,
                    MAX_TOP_LOGPROBS,
                    fields,
                    2,
                    &selections,
                )
            };
            let result = decode(&words, &records).unwrap();
            let value = result[1][0].as_ref().unwrap();
            assert_eq!(value.top.len(), count + 1);
            assert_eq!(
                value.top[..count].iter().map(|p| p.0).collect::<Vec<_>>(),
                ids
            );
            assert_eq!(value.top[count], (999, -4.));
            assert!(result[1][1..].iter().all(Option::is_none));
            words[start] = ids[0] as i32;
            records[1].token_id = i64::from(ids[0]);
            assert_eq!(
                decode(&words, &records).unwrap()[1][0]
                    .as_ref()
                    .unwrap()
                    .top
                    .len(),
                count
            );
            words[start + 2] = 999;
            assert!(decode(&words, &records).is_err());
        }
    }
    let current = with_logprobs(MAX_LOGPROB_TOKEN_IDS);
    for name in ["logprob_token_ids", "logprob_token_id_counts"] {
        let mut partial = current.clone();
        partial.buffers.retain(|buffer| buffer.name != name);
        assert!(partial.validate().is_err());
        let mut missing = current.clone();
        missing.graphs[0]
            .input_bindings
            .retain(|binding| binding.buffer_name != name);
        assert!(missing.validate().is_err());
    }
}

#[test]
fn confidence_readback_is_declared_per_graph_and_bounded_by_frame_capacity() {
    let mut manifest = manifest();
    let graph = &mut manifest.graphs[0];
    graph.kind = GraphKind::Speculative;
    graph.draft_tokens = 3;
    graph.token_capacity = 2;
    graph.packed_token_capacity = 5;
    manifest.validate().unwrap();
    manifest.graphs[0].confidence_width = 3;
    assert!(manifest.validate().is_err());
    manifest.buffers.push(BufferDescriptor {
        name: "proposal_confidence".into(), address: 0x20000,
        nbytes: 2 * 4 * 7 * 4, dtype: Dtype::Float32,
        shape: vec![2, 4, 7], strides: vec![28, 7, 1],
    });
    manifest.validate().unwrap();
    let mut no_readback = manifest.graphs[0].clone();
    no_readback.name = "speculative_without_confidence".into();
    no_readback.handle += 1;
    no_readback.confidence_width = 0;
    manifest.graphs.push(no_readback);
    manifest.validate().unwrap();
    let mut invalid = manifest.clone();
    invalid.graphs[0].confidence_width = 4;
    assert!(invalid.validate().is_err());
    let mut invalid = manifest.clone();
    let buffer = invalid.buffers.last_mut().unwrap();
    buffer.shape = vec![2, 4, 2];
    buffer.strides = vec![8, 2, 1];
    buffer.nbytes = 2 * 4 * 2 * 4;
    assert!(invalid.validate().is_err());
}

#[test]
fn scheduler_width_one_round_has_no_proposal_work() {
    use crate::scheduler::{ScheduledLane, ScheduledStep, RequestHandle};
    let mut step = ScheduledStep {
        kind: GraphKind::Speculative, batch_capacity: 1, token_capacity: 1,
        draft_tokens: 0, packed_token_capacity: 1, mixed: None, requires_feedback: false,
        lanes: vec![ScheduledLane { request: RequestHandle { slot: 0, generation: 1 }, start_position: 64, tokens: 1 }],
    };
    step.validate_mixed().unwrap();
    step.draft_tokens = 1;
    assert!(step.validate_mixed().is_err());
    step.token_capacity = 2;
    step.draft_tokens = 7;
    step.validate_mixed().unwrap();
}

#[test]
#[should_panic(expected = "completion serialization requires a token vector")]
fn completion_serialization_rejects_missing_vector_storage() {
    CompletionRecord::default().fields();
}

pub(super) fn tuning_manifest() -> ExecutionManifest {
    let mut value = manifest();
    for (name, address, dtype, nbytes) in [
        ("choice", 0x10000, Dtype::Int64, 8),
        ("tuning_status", 0x10010, Dtype::Int32, 4),
    ] {
        value.buffers.push(BufferDescriptor { name: name.into(), address, dtype, nbytes,
            shape: vec![1], strides: vec![1] });
    }
    value.tuning_graphs.push(TuningGraphDescriptor { name: "select".into(), handle: 0x90000,
        outputs: vec!["choice".into()], status_buffer: "tuning_status".into() });
    value.graphs[0].tuned_bindings.push(TunedBinding { buffer: "choice".into(), tuning_graph: "select".into() });
    value
}

#[test]
fn tuning_outputs_have_exact_producers_and_no_host_writers() {
    let valid = tuning_manifest();
    valid.validate().unwrap();
    for mutation in 0..7 {
        let mut value = valid.clone();
        match mutation {
            0 => value.graphs[0].tuned_bindings[0].tuning_graph = "missing".into(),
            1 => value.graphs[0].tuned_bindings[0].buffer = "missing".into(),
            2 => value.tuning_graphs[0].outputs.push("choice".into()),
            3 => value.tuning_graphs[0].status_buffer = "choice".into(),
            4 => value.tuning_graphs[0].outputs = vec!["frame_index".into()],
            5 => value.tuning_graphs[0].handle = value.graphs[0].handle,
            _ => { let duplicate = value.graphs[0].tuned_bindings[0].clone(); value.graphs[0].tuned_bindings.push(duplicate); },
        }
        assert!(value.validate().is_err(), "mutation {mutation}");
    }
}

#[test]
fn tuning_order_is_manifest_order_and_deduplicates_consumers() {
    let mut value = tuning_manifest();
    value.graphs.push(value.graphs[0].clone());
    assert_eq!(value.referenced_tuning_graphs(), vec![0]);
    value.graphs.iter_mut().for_each(|graph| graph.tuned_bindings.clear());
    assert!(value.referenced_tuning_graphs().is_empty());
}
