use std::io::Write;
use std::time::Duration;

use super::*;
use crate::execution::{CompletionRecord, Dtype};

#[test]
fn logical_layout_excludes_process_addresses_and_detects_contract_changes() {
    let manifest = super::super::tests::manifest();
    let expected = layout(&manifest);
    let mut peer = manifest.clone();
    peer.context += 10;
    peer.device += 1;
    for buffer in &mut peer.buffers {
        buffer.address += 0x10000;
    }
    peer.graphs[0].handle += 0x10000;
    assert_eq!(expected, layout(&peer));
    peer.buffers[3].dtype = Dtype::Float32;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.graphs[0].token_capacity += 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.graphs[0].prefill_capacity = 64;
    assert_ne!(expected, layout(&peer));
    peer = manifest;
    peer.graphs[0].input_bindings[2].host_offset += 8;
    assert_ne!(expected, layout(&peer));
}

#[test]
fn rank_layout_requires_equal_cache_capacities_and_snapshot_exclusions() {
    let mut manifest = super::super::tests::manifest();
    let mut cache = flare_selene::testing::geometry(64, &[("kv", 4096)], 65536);
    cache.snapshot_pages = vec![100..120, 115..135];
    manifest.cache = Some(cache);
    let expected = layout(&manifest);
    let mut peer = manifest.clone();
    let cache = peer.cache.as_mut().unwrap();
    cache.row.address += 0x10000;
    cache.planes[0].address += 0x20000;
    assert_eq!(expected, layout(&peer));
    peer.cache.as_mut().unwrap().snapshot_pages[0].start += 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.cache.as_mut().unwrap().page_capacity -= 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.cache.as_mut().unwrap().row_capacity -= 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest;
    peer.cache = None;
    assert_ne!(expected, layout(&peer));
}

#[test]
fn fragmented_completion_remains_pending_until_all_bytes_arrive() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut channel = Channel::new(receiver).unwrap();
    channel.nonblocking(true).unwrap();
    sender.write_all(&[COMPLETE, 3]).unwrap();
    assert!(channel.poll().unwrap().is_none());
    sender.write_all(&[0, 0, 0, 10]).unwrap();
    assert!(channel.poll().unwrap().is_none());
    sender.write_all(&[11, 12]).unwrap();
    assert_eq!(channel.poll().unwrap(), Some((COMPLETE, vec![10, 11, 12])));
    drop(sender);
    assert!(
        channel
            .poll()
            .unwrap_err()
            .to_string()
            .contains("disconnected")
    );
}

#[test]
fn packet_bounds_and_peer_errors_are_reported() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut channel = Channel::new(receiver).unwrap();
    sender.write_all(&[COMPLETE, 255, 255, 255, 255]).unwrap();
    assert!(
        channel
            .poll()
            .unwrap_err()
            .to_string()
            .contains("oversized")
    );
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut sender = Channel::new(sender).unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    sender.send(FAILED, b"rank fence differs").unwrap();
    assert!(
        receiver
            .receive(READY)
            .unwrap_err()
            .to_string()
            .contains("rank fence differs")
    );
}

#[cfg(unix)]
fn assert_blocking(channel: &Channel) {
    use std::os::fd::AsRawFd;
    let fd = match &channel.stream {
        RankStream::Unix(stream) => stream.as_raw_fd(),
        RankStream::Tcp(stream) => stream.as_raw_fd(),
    };
    // SAFETY: F_GETFL reads flags from the live socket without retaining its descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    assert!(flags >= 0);
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}

#[test]
fn receive_preserves_control_tags_on_both_transports() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    for (sender, receiver) in [crate::execution::rank_stream::test_pair().unwrap(),
                               (client.into(), server.into())] {
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        for tag in [READY, COMMIT, PREPARE, DRAIN] {
            sender.send(tag, &[tag]).unwrap();
            assert_eq!(receiver.receive(tag).unwrap(), vec![tag]);
            #[cfg(unix)]
            assert_blocking(&receiver);
        }
    }
}

#[test]
fn nonblocking_poll_preserves_fragment_for_receive_and_idle_wait() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    sender.write_all(&[PREPARE, 3, 0]).unwrap();
    receiver.nonblocking(true).unwrap();
    assert!(receiver.poll().unwrap().is_none());
    receiver.nonblocking(false).unwrap();
    #[cfg(unix)]
    assert_blocking(&receiver);
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        sender.write_all(&[0, 0, 10, 11, 12]).unwrap();
        let mut sender = Channel::new(sender).unwrap();
        sender.send(DRAIN, &[]).unwrap();
    });
    assert_eq!(receiver.receive(PREPARE).unwrap(), vec![10, 11, 12]);
    assert_eq!(receiver.wait(true).unwrap(), (DRAIN, vec![]));
    #[cfg(unix)]
    assert_blocking(&receiver);
    writer.join().unwrap();
}

#[test]
fn receive_reports_wrong_tags_peer_errors_and_disconnects() {
    for tag in [COMMIT, FAILED] {
        let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        sender.send(tag, b"rank failed").unwrap();
        let error = receiver.receive(READY).unwrap_err().to_string();
        assert!(error.contains(if tag == FAILED { "rank failed" } else { "received 4" }), "{error}");
        #[cfg(unix)]
        assert_blocking(&receiver);
    }
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    drop(sender);
    assert!(receiver.receive(READY).unwrap_err().to_string().contains("disconnected"));
    #[cfg(unix)]
    assert_blocking(&receiver);
}

#[test]
fn upload_decoder_rejects_unbounded_or_truncated_fragments() {
    let manifest = super::super::tests::manifest();
    let mut bytes = Vec::new();
    number(&mut bytes, usize::MAX);
    let mut remaining = manifest.input_frame_bytes;
    assert!(uploads(&manifest, &mut Reader(&bytes), &mut remaining).is_err());
    bytes.clear();
    for number_value in [1, 3, 8, 8] {
        number(&mut bytes, number_value);
    }
    bytes.extend_from_slice(&7_i64.to_le_bytes());
    let mut remaining = manifest.input_frame_bytes;
    let decoded = uploads(&manifest, &mut Reader(&bytes), &mut remaining).unwrap();
    assert_eq!(decoded[0].buffer, "tokens");
    assert_eq!(decoded[0].offset, 8);
    assert_eq!(decoded[0].bytes, 7_i64.to_le_bytes());
    let mut remaining = manifest.input_frame_bytes;
    assert!(uploads(&manifest, &mut Reader(&bytes[..bytes.len() - 1]), &mut remaining).is_err());
}

#[test]
fn prepare_packets_preserve_large_and_partial_uploads_on_both_transports() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    let mut bytes = vec![0x5a; 80 * 1024 * 1024];
    bytes[..8].copy_from_slice(&1u64.to_le_bytes());
    let end = bytes.len();
    bytes[end - 8..].copy_from_slice(&2u64.to_le_bytes());
    let mut manifest = super::super::tests::manifest();
    let buffer = &mut manifest.buffers[3];
    buffer.nbytes = bytes.len() + 16;
    buffer.shape = vec![buffer.nbytes / 8];
    manifest.graphs[0].input_bindings[2].nbytes = buffer.nbytes;
    manifest.input_frame_bytes = 16 + buffer.nbytes;
    manifest.validate().unwrap();
    assert_eq!(u64::from_le_bytes(layout(&manifest)[..8].try_into().unwrap()), 5);
    for (sender, receiver) in [crate::execution::rank_stream::test_pair().unwrap(),
                               (client.into(), server.into())] {
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        for parts in [1, 2] {
            std::thread::scope(|scope| {
                let writer = scope.spawn(|| {
                    let uploads: Vec<_> = bytes.chunks(bytes.len() / parts).enumerate()
                        .map(|(index, bytes)| Upload {
                            buffer: "tokens", offset: 8 + index * 80 * 1024 * 1024 / parts, bytes,
                        }).collect();
                    let mut packets = 0;
                    wire::prepare(&manifest, Fence(7), 0, true, &uploads, |tag, data| {
                        assert!(data.len() <= 64 * 1024 * 1024);
                        assert_eq!(tag, if packets == 0 { PREPARE_MORE } else { PREPARE });
                        packets += 1;
                        sender.send(tag, data)
                    }).unwrap();
                    assert_eq!(packets, 2);
                    sender.send(DRAIN, &[]).unwrap();
                });
                let (tag, packet) = receiver.wait(false).unwrap();
                let (fence, graph, grammar, owned) =
                    receive_prepare(&mut receiver, &manifest, tag, packet).unwrap();
                assert_eq!((fence, graph, grammar), (Fence(7), 0, true));
                assert!(owned.len() > parts);
                let mut offset = 8;
                for upload in &owned {
                    assert_eq!(upload.buffer, "tokens");
                    assert_eq!(upload.offset, offset);
                    assert_eq!(upload.bytes.len() % 8, 0);
                    assert_eq!(upload.bytes, bytes[offset - 8..offset - 8 + upload.bytes.len()]);
                    offset += upload.bytes.len();
                }
                assert_eq!(offset, bytes.len() + 8);
                let borrowed: Vec<_> = owned.iter().map(|upload| Upload {
                    buffer: &upload.buffer, offset: upload.offset, bytes: &upload.bytes,
                }).collect();
                super::super::transport::validate_uploads(&manifest, graph, &borrowed, &mut Vec::new()).unwrap();
                assert!(receiver.receive(DRAIN).unwrap().is_empty());
                writer.join().unwrap();
            });
        }
    }
}

#[test]
fn small_prepare_preserves_one_packet_and_upload_extents() {
    let manifest = super::super::tests::manifest();
    let bytes = 7_i64.to_le_bytes();
    let input = [Upload { buffer: "tokens", offset: 8, bytes: &bytes }];
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut sender = Channel::new(sender).unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    wire::prepare(&manifest, Fence(3), 0, false, &input, |tag, data| {
        assert_eq!(tag, PREPARE);
        sender.send(tag, data)
    }).unwrap();
    let (tag, packet) = receiver.wait(false).unwrap();
    let (fence, graph, grammar, owned) = receive_prepare(&mut receiver, &manifest, tag, packet).unwrap();
    assert_eq!((fence, graph, grammar), (Fence(3), 0, false));
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].offset, 8);
    assert_eq!(owned[0].bytes, bytes);
}

#[test]
fn prepare_continuations_reject_identity_changes_wrong_tags_and_frame_overflow() {
    let manifest = super::super::tests::manifest();
    let packet = |fence, graph, grammar, size: usize| {
        let mut bytes = Vec::new();
        for value in [fence, graph, grammar, 1, 3, 0, size] {
            number(&mut bytes, value);
        }
        bytes.resize(bytes.len() + size, 0);
        bytes
    };
    for (tag, next, expected) in [
        (PREPARE, packet(8, 0, 0, 8), "identity differs"),
        (PREPARE, packet(7, 1, 0, 8), "identity differs"),
        (PREPARE, packet(7, 0, 1, 8), "identity differs"),
        (PREPARE, packet(7, 0, 2, 8), "invalid grammar"),
        (COMMIT, vec![], "expected rank prepare"),
        (PREPARE, packet(7, 0, 0, 32), "input-frame extent"),
    ] {
        let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        sender.send(tag, &next).unwrap();
        let error = receive_prepare(&mut receiver, &manifest, PREPARE_MORE, packet(7, 0, 0, 24))
            .err().unwrap().to_string();
        assert!(error.contains(expected), "{error}");
    }
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    drop(sender);
    let error = receive_prepare(&mut receiver, &manifest, PREPARE_MORE, packet(7, 0, 0, 8))
        .err().unwrap().to_string();
    assert!(error.contains("disconnected"), "{error}");
    let mut empty = Vec::new();
    for value in [7, 0, 0, 0] { number(&mut empty, value); }
    let error = receive_prepare(&mut receiver, &manifest, PREPARE_MORE, empty).err().unwrap().to_string();
    assert!(error.contains("empty rank prepare"), "{error}");
}

fn completed(status: i64) -> FrameCompletion {
    let record = CompletionRecord {
        generation: 7,
        submission_sequence: 3,
        consumed_tokens: 65,
        generated_tokens: 0,
        token_id: -1,
        status,
        error_code: 0,
        reserved: 0,
        vector: Some(super::super::TokenVector::default()),
    };
    FrameCompletion {
        logprobs: None,
        prompt_logprobs: None,
        proposal_confidence: None,
        speculative_survival: None,
        fence: Fence(3),
        graph_index: 2,
        timing: Default::default(),
        records: [record; 4].into(),
    }
}

#[test]
fn peers_agree_on_request_consumption_without_sampling() {
    let mut root = completed(2);
    root.records[0].generated_tokens = 2;
    root.records[0].token_id = 19;
    let mut peer = completed(1);
    agree(&root, &peer).unwrap();
    peer.records[0].consumed_tokens -= 1;
    let error = agree(&root, &peer).unwrap_err().to_string();
    assert!(error.contains("graph 2, fence 3, lane 0"), "{error}");
    assert!(error.contains("consumed_tokens: 65"), "{error}");
    assert!(error.contains("consumed_tokens: 64"), "{error}");
    peer = completed(1);
    peer.records[0].generation += 1;
    assert!(agree(&root, &peer).is_err());
    peer = completed(3);
    assert!(agree(&root, &peer).is_err());
    root = completed(3);
    agree(&root, &peer).unwrap();
}

#[test]
fn completion_wire_preserves_the_execution_abi() {
    let mut root = completed(1);
    root.timing.gpu_ns[0] = Some(1250);
    root.timing.wall_ns = 2345;
    root.timing.queued = true;
    root.timing.readiness = Some(super::super::ReadinessIntervals::decode(
        &[100u64, 110, 111, 200, 210, 215].into_iter().flat_map(u64::to_ne_bytes).collect::<Vec<_>>()
    ).unwrap());
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    assert_eq!(reader.number().unwrap(), 3);
    assert_eq!(reader.number().unwrap(), 2);
    assert_eq!(timing(&mut reader).unwrap(), root.timing);
    assert_eq!(proposal_confidence(&mut reader).unwrap(), root.proposal_confidence);
    assert_eq!(speculative_survival(&mut reader).unwrap(), root.speculative_survival);
    let records = super::super::transport::decode_records(reader.0, Fence(3), 4, super::super::VECTOR_RECORD_FIELDS).unwrap();
    for (expected, actual) in root.records.iter().zip(records.iter()) {
        assert_eq!(expected.fields(), actual.fields());
    }
}

#[test]
fn speculative_survival_is_optional_and_validated_per_rank_without_numeric_consensus() {
    let mut root = completed(1);
    for record in root.records.iter_mut() {
        record.vector = Some(crate::execution::TokenVector {
            emitted_count: 1, offered_drafts: 3, accepted_drafts: 0, verification_width: 4,
            ..Default::default()
        });
    }
    let mut values = [[-1.; 7]; 64];
    values[..4].fill([0.75, 0.5, 0.25, -1., -1., -1., -1.]);
    root.speculative_survival = Some(values);
    let mut peer = completed(1);
    peer.records = root.records.clone();
    assert!(agree(&root, &peer).is_err());
    peer.speculative_survival = Some(values);
    peer.speculative_survival.as_mut().unwrap()[1][1] = 0.6;
    agree(&root, &peer).unwrap();
    assert_eq!(root.speculative_survival, Some(values));
    for (slot, depth, invalid) in [(0, 0, f32::NAN), (0, 0, 1.1), (0, 0, -0.1),
        (0, 1, 0.8), (0, 1, -1.), (0, 2, -1.), (0, 3, 0.1), (4, 0, 0.1)] {
        peer.speculative_survival = Some(values);
        peer.speculative_survival.as_mut().unwrap()[slot][depth] = invalid;
        assert!(agree(&root, &peer).is_err(), "slot {slot}, depth {depth}, value {invalid}");
    }
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    reader.number().unwrap();
    reader.number().unwrap();
    timing(&mut reader).unwrap();
    proposal_confidence(&mut reader).unwrap();
    let offset = bytes.len() - reader.0.len();
    assert_eq!(speculative_survival(&mut reader).unwrap(), Some(values));
    for length in [0, 1, 7, 8, 9, 8 + 64 * 7 * 4 - 1] {
        assert!(speculative_survival(&mut Reader(&bytes[offset..offset + length])).is_err());
    }
}

#[test]
fn rank_timing_rejects_truncation_and_preserves_zero_duration() {
    for gpu in [None, Some(0), Some(731)] {
        let mut root = completed(1);
        root.timing.gpu_ns[0] = gpu;
        root.timing.wall_ns = 1234;
        let bytes = completion(&root);
        for length in 0..40 {
            assert!(timing(&mut Reader(&bytes[16..16 + length])).is_err());
        }
        assert_eq!(timing(&mut Reader(&bytes[16..])).unwrap(), root.timing);
    }
    let mut invalid = vec![];
    for value in [2, 0, 1234, 0] {
        number(&mut invalid, value);
    }
    assert!(timing(&mut Reader(&invalid)).is_err());
}

#[test]
fn rank_completion_requires_every_record() {
    let root = completed(1);
    let mut peer = completed(1);
    peer.records = [peer.records[0]; 3].into();
    assert!(agree(&root, &peer).is_err());
}

#[test]
fn proposal_confidence_survives_rank_transport_without_numeric_consensus() {
    let mut root = completed(1);
    root.proposal_confidence = Some([[0.5; 7]; 64]);
    let mut peer = completed(1);
    assert!(agree(&root, &peer).is_err());
    peer.proposal_confidence = root.proposal_confidence;
    agree(&root, &peer).unwrap();
    peer.proposal_confidence.as_mut().unwrap()[3][2] = 0.6;
    agree(&root, &peer).unwrap();
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    reader.number().unwrap();
    reader.number().unwrap();
    timing(&mut reader).unwrap();
    assert_eq!(proposal_confidence(&mut reader).unwrap(), root.proposal_confidence);
    assert_eq!(speculative_survival(&mut reader).unwrap(), root.speculative_survival);
    assert_eq!(reader.0.len(), 4 * super::super::VECTOR_RECORD_FIELDS * 8);
}

#[test]
fn proposal_confidence_validates_each_rank_without_changing_root_values() {
    let mut root = completed(1);
    let mut peer = completed(1);
    root.proposal_confidence = Some([[0.0; 7]; 64]);
    peer.proposal_confidence = root.proposal_confidence;
    root.proposal_confidence.as_mut().unwrap()[1][2] = f32::from_bits(0x3f79148d);
    peer.proposal_confidence.as_mut().unwrap()[1][2] = f32::from_bits(0x3f79149e);
    agree(&root, &peer).unwrap();
    peer.proposal_confidence.as_mut().unwrap()[1][2] = 0.973;
    let authoritative = root.proposal_confidence;
    agree(&root, &peer).unwrap();
    assert_eq!(root.proposal_confidence, authoritative);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.000001, 1.000001] {
        root.proposal_confidence = authoritative;
        peer.proposal_confidence = authoritative;
        root.proposal_confidence.as_mut().unwrap()[1][2] = invalid;
        assert!(agree(&root, &peer).is_err());
        root.proposal_confidence = authoritative;
        peer.proposal_confidence.as_mut().unwrap()[1][2] = invalid;
        assert!(agree(&root, &peer).is_err());
    }
}

#[test]
fn rank_zero_confidence_is_authoritative_without_relaxing_request_agreement() {
    let mut root = completed(1);
    let mut peer = completed(1);
    root.proposal_confidence = Some([[0.5; 7]; 64]);
    peer.proposal_confidence = root.proposal_confidence;
    root.proposal_confidence.as_mut().unwrap()[2][2] = 0.51028365;
    peer.proposal_confidence.as_mut().unwrap()[2][2] = 0.5102996;
    let authoritative = root.proposal_confidence;
    agree(&root, &peer).unwrap();
    assert_eq!(root.proposal_confidence, authoritative);
    peer.records[0].consumed_tokens += 1;
    assert!(agree(&root, &peer).is_err());
}

#[test]
fn speculative_process_schema_rejects_a_different_proposal_depth() {
    let mut root = crate::execution::tests::manifest();
    let graph = &mut root.graphs[0];
    graph.kind = crate::execution::GraphKind::Speculative;
    graph.token_capacity = 6;
    graph.packed_token_capacity = 8;
    graph.draft_tokens = 1;
    let mut peer = root.clone();
    assert_eq!(layout(&root), layout(&peer));
    peer.graphs[0].draft_tokens = 3;
    assert_ne!(layout(&root), layout(&peer), "different proposal contracts must fail before any rank launch");
}

#[test]
fn logprob_uploads_budget_count_and_prefix_wire_bytes_separately() {
    let manifest = super::super::tests::with_logprobs(crate::execution::MAX_LOGPROB_TOKEN_IDS);
    let measure = |uploads: &[Upload<'_>]| {
        let mut size = 0;
        wire::prepare(&manifest, Fence(1), 0, false, uploads, |_, bytes| {
            size += bytes.len();
            Ok(())
        })
        .unwrap();
        size
    };
    let empty = measure(&[]);
    let counts = [0u8; 16];
    assert_eq!(
        measure(&[Upload {
            buffer: "logprob_token_id_counts",
            offset: 0,
            bytes: &counts
        }]) - empty,
        16 + 24
    );
    for count in [1, 20, 21, 127, 128] {
        let ids: Vec<_> = (0..count as i32).flat_map(i32::to_ne_bytes).collect();
        let additional = measure(&[
            Upload {
                buffer: "logprob_token_id_counts",
                offset: 0,
                bytes: &counts,
            },
            Upload {
                buffer: "logprob_token_ids",
                offset: 0,
                bytes: &ids,
            },
        ]) - empty;
        assert_eq!(additional, 16 + 24 + 4 * count + 24);
        eprintln!(
            "C=4 N={count}: H2D={} remote_peer={additional}",
            16 + 4 * count
        );
    }
}

#[test]
fn two_frame_window_retains_preparing_and_unconsumed_completions() {
    let mut manifest = crate::execution::tests::manifest();
    manifest.graphs[0].kind = crate::execution::GraphKind::Decode;
    let frame = |sequence| Pending {
        fence: Fence(sequence), graph: 0, grammar: false, uploads: Vec::new(),
        dispatched: Instant::now(), committed: false, queued: sequence > 1,
        ready: vec![false; 3], completions: (0..4).map(|_| None).collect(),
    };
    let mut pending = VecDeque::new();
    assert!(window_available(&pending, false, &manifest));
    pending.push_back(frame(1));
    assert!(!window_available(&pending, false, &manifest));
    for rank in 0..3 { ready_frame(&mut pending, rank, Fence(1), 0).unwrap(); }
    assert!(pending[0].ready.iter().all(|ready| *ready));
    pending[0].committed = true;
    assert!(window_available(&pending, false, &manifest));
    assert!(!window_available(&pending, true, &manifest));
    pending.push_back(frame(2));
    for rank in 0..3 { ready_frame(&mut pending, rank, Fence(2), 0).unwrap(); }
    pending[1].committed = true;
    assert!(!window_available(&pending, false, &manifest));
    for rank in 0..3 {
        assert!(completion_frame(&mut pending, rank, Fence(2), 0).is_err());
        let mut value = completed(1);
        value.graph_index = 0;
        value.fence = Fence(1);
        for record in value.records.iter_mut() { record.submission_sequence = 1; }
        completion_frame(&mut pending, rank, Fence(1), 0).unwrap().completions[rank + 1] = Some(value);
    }
    pending[0].completions[0] = Some(completed(1));
    assert!(!window_available(&pending, false, &manifest));
    pending.pop_front();
    assert!(window_available(&pending, false, &manifest));
    assert!(ready_frame(&mut pending, 0, Fence(2), 0).is_err());
    assert!(ready_frame(&mut pending, 0, Fence(3), 0).is_err());
    assert!(completion_frame(&mut pending, 0, Fence(1), 0).is_err());
    assert!(completion_frame(&mut pending, 0, Fence(2), 1).is_err());
    completion_frame(&mut pending, 0, Fence(2), 0).unwrap();
}

#[test]
fn rank_commit_identity_rejects_truncation_and_trailing_bytes() {
    let bytes = identity(Fence(7), 3);
    assert_eq!(read_identity(&bytes).unwrap(), (Fence(7), 3));
    assert!(read_identity(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(read_identity(&trailing).is_err());
}
