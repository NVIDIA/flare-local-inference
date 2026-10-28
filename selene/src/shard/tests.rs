
    use super::*;
use std::net::TcpListener;
    use crate::testing::{StoppedCopier, geometry};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(super) fn directory(test: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("flare-shard-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// A staged transport without a staging pool: its disk thread runs truncates and unlinks.
    fn staged() -> Staged {
        let transfers = StoppedCopier::shared(&[geometry(64, &[("kv", 4096)], 4096)]).unwrap();
        Staged::new(
            None,
            transfers,
            DiskThread::start("flare-shard-test").unwrap(),
        )
    }

    pub(super) fn connected() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (client, listener.accept().unwrap().0)
    }

    fn record_file(directory: &Path, tokens: u64) -> PathBuf {
        let path = directory.join(record::file_name(&key(tokens, 0x5a)));
        std::fs::write(&path, vec![1u8; 8192]).unwrap();
        path
    }

    /// Poll until `count` events arrived, failing after a few seconds.
    fn events(transport: &mut Sharded, count: usize) -> Vec<TransportEvent> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut events = Vec::new();
        while events.len() < count {
            assert!(Instant::now() < deadline, "only {events:?} arrived");
            transport.poll(&mut events).unwrap();
            std::thread::sleep(IDLE);
        }
        events
    }

    /// Poll for a while and require that nothing was reported.
    fn quiet(transport: &mut Sharded) {
        let mut events = Vec::new();
        for _ in 0..50 {
            transport.poll(&mut events).unwrap();
            std::thread::sleep(IDLE);
        }
        assert!(events.is_empty(), "{events:?}");
    }

    fn reply(peer: &mut Link) -> (u8, Vec<u8>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(frame) = peer.receive().unwrap() {
                return frame;
            }
            assert!(Instant::now() < deadline, "no frame arrived");
            std::thread::sleep(IDLE);
        }
    }

    #[test]
    fn rank_zero_reports_a_job_after_every_rank_and_unlinks_partial_stores() {
        let directory = directory("zero");
        let (zero, peer) = connected();
        let mut peer = Link::new(peer).unwrap();
        let mut transport = Sharded::new(staged(), vec![Link::new(zero).unwrap()]).unwrap();

        // Rank zero's unlink finishes first; the job waits for the peer.
        let first = record_file(&directory, 128);
        transport.unlink(JobId(1), first.clone()).unwrap();
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        assert_eq!((tag, reader.u64().unwrap()), (UNLINK, 1));
        assert_eq!(reader.text().unwrap(), name(&first).unwrap());
        quiet(&mut transport);
        assert!(!first.exists());
        peer.send(&encode_event(Kind::Unlinked, JobId(1), &Ok(0)))
            .unwrap();
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Unlinked {
                id: JobId(1),
                result: Ok(())
            }]
        ));

        // A peer's failure fails the job and names the rank.
        let second = record_file(&directory, 256);
        transport.truncate(JobId(2), second.clone(), 4096).unwrap();
        assert_eq!(reply(&mut peer).0, TRUNCATE);
        peer.send(&encode_event(
            Kind::Truncated,
            JobId(2),
            &Err(invalid("disk full")),
        ))
        .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::Truncated {
                    id: JobId(2),
                    result: Err(error),
                },
            ] => {
                assert!(error.to_string().contains("rank 1: "), "{error}");
                assert!(error.to_string().contains("disk full"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(std::fs::metadata(&second).unwrap().len(), 4096);

        // Rank zero wrote a record the peer failed to write: rank zero removes its file and
        // only then reports the store as failed. The peer, which wrote nothing, is not asked.
        let third = record_file(&directory, 384);
        transport.pending.insert(
            (Kind::Stored, JobId(3)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Ok(8192)), None],
                path: Some(third.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(3), 1, Err(invalid("no space")));
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::StoreInvalidated {
                    id: JobId(3),
                    error,
                },
            ] => {
                assert!(error.to_string().contains("no space"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert!(!third.exists());

        // The peer wrote a record rank zero failed to write: the peer is asked to remove it,
        // with a job id outside the disk store's, and the failure waits for its answer.
        let fourth = record_file(&directory, 512);
        transport.pending.insert(
            (Kind::Stored, JobId(4)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("short write"))), None],
                path: Some(fourth.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(4), 1, Ok(8192));
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        let internal = reader.u64().unwrap();
        assert_eq!(tag, UNLINK);
        assert!(internal >= INTERNAL_JOBS);
        assert_eq!(reader.text().unwrap(), name(&fourth).unwrap());
        quiet(&mut transport);
        peer.send(&encode_event(Kind::Unlinked, JobId(internal), &Ok(0)))
            .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::StoreInvalidated {
                    id: JobId(4),
                    error,
                },
            ] => {
                assert!(error.to_string().contains("short write"), "{error}");
            }
            other => panic!("{other:?}"),
        }

        // A cleanup unlink that itself fails leaves a file the index does not
        // charge for, so the failure names the rank that kept it.
        let sixth = record_file(&directory, 576);
        transport.pending.insert(
            (Kind::Stored, JobId(6)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("no space"))), None],
                path: Some(sixth.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(6), 1, Ok(8192));
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        let internal = reader.u64().unwrap();
        assert_eq!(tag, UNLINK);
        peer.send(&encode_event(
            Kind::Unlinked,
            JobId(internal),
            &Err(invalid("directory is read-only")),
        ))
        .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::StoreInvalidated {
                    id: JobId(6),
                    error,
                },
            ] => {
                let error = error.to_string();
                assert!(error.contains("no space"), "{error}");
                assert!(error.contains("cleanup unlink failed"), "{error}");
                assert!(error.contains("rank 1"), "{error}");
            }
            other => panic!("{other:?}"),
        }

        // When every write fails, the previous files remain usable.
        let seventh = record_file(&directory, 608);
        transport.pending.insert(
            (Kind::Stored, JobId(7)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("write failed"))), None],
                path: Some(seventh.clone()),
                cleanup: None,
                started: Instant::now(),
            },
        );
        transport.record(Kind::Stored, JobId(7), 1, Err(invalid("disk full")));
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Stored {
                id: JobId(7),
                result: Err(_)
            }]
        ));
        assert!(seventh.exists());

        // A broken connection fails every job still waiting on that rank.
        let fifth = record_file(&directory, 640);
        transport.unlink(JobId(5), fifth).unwrap();
        assert_eq!(reply(&mut peer).0, UNLINK);
        drop(peer);
        // The peer may still be copying into the ids of the jobs it already
        // received, so the transport stops instead of reporting them failed.
        let mut reported = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        let error = loop {
            assert!(
                Instant::now() < deadline,
                "the transport never noticed the broken link"
            );
            match transport.poll(&mut reported) {
                Ok(()) => std::thread::sleep(IDLE),
                Err(error) => break error,
            }
        };
        assert!(error.to_string().contains("rank 1"), "{error}");
        assert!(reported.is_empty(), "{reported:?}");
        assert!(
            transport
                .unlink(JobId(6), record_file(&directory, 768))
                .is_err()
        );
        // A poisoned transport never confirms its drain, so its staging and
        // transfer threads stay alive with it.
        assert!(transport.close().is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn peer_service_mirrors_jobs_into_its_own_directory_until_closed() {
        let zero_directory = directory("mirror-zero");
        let peer_directory = directory("mirror-peer");
        let (zero, peer) = connected();
        let lock = crate::testing::directory_lock(&peer_directory).unwrap();
        let service_directory = peer_directory.clone();
        let service = std::thread::spawn(move || {
            let mut peer = Peer {
                staged: staged(),
                directory: service_directory,
                fingerprint: [3; 16],
                geometry: vec![geometry(64, &[("kv", 4096)], 4096)],
                link: Link::new(peer).unwrap(),
                _lock: lock,
            };
            let outcome = peer.run();
            let shutdown = peer.shutdown();
            if shutdown.is_err() {
                std::mem::forget(peer);
            }
            outcome.and(shutdown)
        });
        let mut transport = Sharded::new(staged(), vec![Link::new(zero).unwrap()]).unwrap();
        let (ours, theirs) = (
            record_file(&zero_directory, 128),
            record_file(&peer_directory, 128),
        );
        let (kept_ours, kept_theirs) = (
            record_file(&zero_directory, 256),
            record_file(&peer_directory, 256),
        );
        transport.unlink(JobId(1), ours.clone()).unwrap();
        transport
            .truncate(JobId(2), kept_ours.clone(), 4096)
            .unwrap();
        let mut events = events(&mut transport, 2);
        events.sort_by_key(|event| Kind::of(event).1);
        assert!(matches!(
            events.as_slice(),
            [
                TransportEvent::Unlinked {
                    id: JobId(1),
                    result: Ok(())
                },
                TransportEvent::Truncated {
                    id: JobId(2),
                    result: Ok(())
                },
            ]
        ));
        assert!(!ours.exists() && !theirs.exists());
        assert_eq!(std::fs::metadata(&kept_ours).unwrap().len(), 4096);
        assert_eq!(std::fs::metadata(&kept_theirs).unwrap().len(), 4096);

        transport.close().unwrap();
        service.join().unwrap().unwrap();
        std::fs::remove_dir_all(zero_directory).unwrap();
        std::fs::remove_dir_all(peer_directory).unwrap();
    }

    fn key(tokens: u64, seed: u8) -> PrefixKey {
        PrefixKey {
            consumed_tokens: tokens,
            digest: [seed; 16],
        }
    }

    fn scanned(key: PrefixKey, parent: Option<PrefixKey>, row: bool, fingerprint: u8) -> Scanned {
        let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
        let mut header = Header::new(
            &geometry,
            [7; 32],
            [fingerprint; 16],
            key,
            parent,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        header.row_checksum = fingerprint as u64;
        Scanned {
            bytes: if row {
                header.length_with_row()
            } else {
                header.row_segment
            },
            path: PathBuf::from(record::file_name(&key)),
            header,
            row,
            modified: SystemTime::now(),
        }
    }

    #[test]
    fn an_unconfirmed_drain_is_reported_so_the_peer_keeps_its_staging() {
        let directory = directory("drain");
        let lock = crate::testing::directory_lock(&directory).unwrap();
        let (zero, peer) = connected();
        let mut peer = Peer {
            staged: staged(),
            directory,
            fingerprint: [3; 16],
            geometry: vec![geometry(64, &[("kv", 4096)], 4096)],
            link: Link::new(peer).unwrap(),
            _lock: lock,
        };
        // The link failed, so the peer cannot report that it drained. The
        // caller must retain the transport rather than release its staging
        // under copies that may still run.
        peer.link.fail(invalid("rank zero closed the connection"));
        assert!(peer.shutdown().is_err());
        drop(zero);
    }

    #[test]
    fn a_job_past_its_deadline_stops_the_transport() {
        let (zero, _peer) = connected();
        let mut transport = Sharded::new(staged(), vec![Link::new(zero).unwrap()]).unwrap();
        transport.pending.insert(
            (Kind::Unlinked, JobId(7)),
            Pending {
                kind: Kind::Unlinked,
                results: vec![Some(Ok(0)), None],
                path: None,
                cleanup: None,
                started: Instant::now() - JOB_TIMEOUT - Duration::from_secs(1),
            },
        );
        transport.expire();
        let mut events = Vec::new();
        let error = transport.poll(&mut events).unwrap_err();
        assert!(error.to_string().contains("did not complete"), "{error}");
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn a_peer_rejects_a_header_from_another_layout() {
        let directory = directory("layout");
        let lock = crate::testing::directory_lock(&directory).unwrap();
        let (zero, peer) = connected();
        let peer = Peer {
            staged: staged(),
            directory,
            fingerprint: [3; 16],
            geometry: vec![geometry(64, &[("kv", 4096)], 4096)],
            link: Link::new(peer).unwrap(),
            _lock: lock,
        };
        let own = Header::new(
            &peer.geometry,
            [7; 32],
            [1; 16],
            key(128, 0x5a),
            None,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        assert!(peer.own(own).is_ok());
        let wider = vec![geometry(64, &[("kv", 8192)], 8192)];
        let foreign = Header::new(
            &wider,
            [7; 32],
            [1; 16],
            key(128, 0x5a),
            None,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        assert!(peer.own(foreign).is_err());
        drop(zero);
    }

    #[test]
    fn a_peer_that_stops_reading_fails_the_job_instead_of_blocking() {
        let (zero, peer) = connected();
        let mut link = Link::new(zero).unwrap();
        // The peer never reads, so the socket buffer and then the queue fill.
        let frame = vec![0u8; 1 << 20];
        let started = Instant::now();
        let mut failure = None;
        for _ in 0..(BACKLOG * 4) {
            if let Err(error) = link.enqueue(&frame) {
                failure = Some(error);
                break;
            }
        }
        assert!(started.elapsed() < Duration::from_secs(5));
        let failure = failure.expect("the queue accepted more than its backlog");
        assert!(failure.to_string().contains("backlog"), "{failure}");
        drop(peer);
    }

    #[test]
    fn agreement_keeps_records_every_rank_holds_and_drops_their_orphans() {
        let (a, b, c, d) = (key(128, 1), key(256, 2), key(384, 3), key(128, 4));
        let zero = vec![
            scanned(a, None, true, 0),
            scanned(b, Some(a), true, 0),
            scanned(c, Some(b), true, 0),
            scanned(d, None, true, 0),
        ];
        // Rank one lacks b's row and holds no d; rank two lacks d's row too.
        let one = vec![
            scanned(a, None, true, 1),
            scanned(b, Some(a), false, 1),
            scanned(c, Some(b), true, 1),
        ];
        let two = vec![
            scanned(a, None, true, 2),
            scanned(b, Some(a), true, 2),
            scanned(c, Some(b), true, 2),
            scanned(d, None, false, 2),
        ];
        let peers = [one, two]
            .iter()
            .map(|records| decode_summary(&summary(records).unwrap()[5..]).unwrap())
            .collect::<Vec<_>>();
        let agreed = agree(&zero, &peers).unwrap();
        assert_eq!(agreed, HashMap::from([(a, true), (b, false), (c, true)]));
        let decoded = decode_agreed(&encode_agreed(&agreed)[5..]).unwrap();
        assert_eq!(decoded, agreed);

        // A peer whose header for `a` differs drops `a` and, with it, every descendant.
        let mut other = scanned(a, None, true, 1);
        other.header.bucket = [9; 16];
        let peers = [decode_summary(&summary(&[other]).unwrap()[5..]).unwrap()];
        assert!(agree(&zero, &peers).unwrap().is_empty());
    }

    #[test]
    fn events_and_peer_file_names_round_trip_and_reject_paths() {
        let event = TransportEvent::StoreInvalidated {
            id: JobId(11),
            error: invalid("partial store"),
        };
        let (kind, id, result) = Kind::of(&event);
        let frame = encode_event(kind, id, &result);
        let (kind, id, result) = decode_event(&frame[5..]).unwrap();
        assert!(matches!(
            kind.event(id, result),
            TransportEvent::StoreInvalidated { id: JobId(11), .. }
        ));
        for result in [Ok(42u64), Err(invalid("disk full"))] {
            let frame = encode_event(Kind::Fetched, JobId(9), &result);
            let (kind, id, decoded) = decode_event(&frame[5..]).unwrap();
            assert_eq!((kind, id), (Kind::Fetched, JobId(9)));
            match (result, decoded) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(a), Err(b)) => assert!(b.to_string().contains(&a.to_string())),
                other => panic!("{other:?}"),
            }
        }
        let directory = Path::new("/records");
        let key = key(4096, 0xab);
        let plain = record::file_name(&key);
        assert_eq!(file(directory, &plain).unwrap(), directory.join(&plain));
        let temporary = record::temporary_name(&key);
        assert_eq!(
            file(directory, &temporary).unwrap(),
            directory.join(&temporary)
        );
        for name in [
            "../etc/passwd",
            "/abs.ckpt",
            "x.ckpt",
            ".tmp",
            "",
            &format!("a/{plain}"),
        ] {
            assert!(file(directory, name).is_err(), "{name:?}");
        }
    }

#[test]
fn space_pressure_survives_rank_events_and_aggregation() {
    for kind in [ErrorKind::StorageFull, ErrorKind::QuotaExceeded] {
        let frame = encode_event(
            Kind::Stored,
            JobId(7),
            &Err(Error::carrying(std::io::Error::from(kind))),
        );
        let (kind, id, result) = decode_event(&frame[5..]).unwrap();
        assert!(result.as_ref().unwrap_err().no_space());
        let mut transport = Sharded::new(staged(), Vec::new()).unwrap();
        transport.finish(
            id,
            Pending {
                kind,
                results: vec![Some(result)],
                path: None,
                cleanup: None,
                started: Instant::now(),
            },
        );
        assert!(
            matches!(&transport.events[..], [TransportEvent::Stored { result: Err(error), .. }] if error.no_space())
        );
        transport.close().unwrap();
    }
}
