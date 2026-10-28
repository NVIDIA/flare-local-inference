//! Pool protocol, the manager's own store: Put, and claims on the manager's records.

mod support;
use support::HostClient;

use selene_manager::{Key, proto::v2};
use std::time::Duration;
use support::{
    Host, content_for, descriptor, key,
    pool::{self, Damage, Ident, PAGE_BYTES, Pool, PutResult, Session},
};

fn host() -> Host {
    let mut host = Host::dram_only(8 * 1024 * 1024);
    host.config.limits.pin_deadline_ms = 1000;
    host.config.limits.progress_timeout_ms = 3000;
    host
}

async fn open(pool: &mut Pool, name: &str) -> Session {
    let mut session = pool.open(&Ident::new(name)).await;
    session.snapshot(vec![]).await;
    session
}

async fn store_chain(writer: &mut Session, first_id: u64, tokens: &[u64]) {
    let mut parent: Option<Key> = None;
    for (index, token) in tokens.iter().enumerate() {
        let record = key(*token);
        match writer
            .put_record(first_id + index as u64, record, parent, Damage::default())
            .await
        {
            PutResult::Stored(bytes) => assert!(bytes > 0),
            other => panic!("put failed: {other:?}"),
        }
        parent = Some(record);
    }
}

/// The bytes the manager must serve for a stored record.
fn expected(tokens: &[u64], upto: usize, begin: u64, with_row: bool) -> Vec<Vec<u8>> {
    let mut parent: Option<Key> = None;
    let mut windows = Vec::new();
    for (index, token) in tokens.iter().enumerate().take(upto + 1) {
        let record = key(*token);
        let (header, bytes, _) = content_for(
            &descriptor(),
            record,
            parent,
            parent.map_or(0, |p| (p.consumed_tokens / 64) as u32),
        );
        let start_page = begin / 64;
        let node_start = parent.map_or(0, |parent| parent.consumed_tokens / 64);
        let first = start_page
            .saturating_sub(node_start)
            .min(u64::from(header.tail_pages)) as usize;
        let mut pages = Vec::new();
        for page in first..header.tail_pages as usize {
            let at = header.page_segment as usize + page * PAGE_BYTES as usize;
            pages.extend_from_slice(&bytes[at..at + PAGE_BYTES as usize]);
        }
        if with_row && index == upto {
            pages.extend_from_slice(&bytes[header.row_segment as usize..]);
        }
        windows.push(pages);
        parent = Some(record);
    }
    windows
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn u1_and_c3_a_put_is_stored_indexed_and_served_to_a_claim() {
    let mut host = host();
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "writer").await;
    let tokens = [1024u64, 2048];
    store_chain(&mut writer, 1, &tokens).await;
    let mut reader = open(&mut pool, "reader").await;
    let candidate = reader.manager_candidate(1, 2048).await;
    assert_eq!(candidate.level, v2::Level::HostDram as i32);
    assert!(candidate.source.as_ref().unwrap().instance_id.is_empty());
    assert_eq!(candidate.nodes.len(), 2);
    reader.claim(reader.claim_for(1, &candidate, 0, true)).await;
    let granted = reader.claim_granted(1).await;
    assert_eq!(granted.total_pages, 32);
    assert!(granted.row);
    assert!(!granted.retained);
    // The windows alias the manager's bytes and carry the persisted checksums.
    let mut got: Vec<Vec<u8>> = vec![Vec::new(), Vec::new()];
    let mut pages_seen = 0u64;
    let mut ready_count = 0;
    while pages_seen < 32 || ready_count == 0 || !got[1].len().is_multiple_of(1) {
        let ready = reader.window_ready(1).await;
        let window = ready.window.unwrap();
        let bytes = reader.shm.as_ref().unwrap().read(&window);
        for (page, sum) in ready.page_xxh3.iter().enumerate() {
            assert_eq!(
                *sum,
                pool::xxh3(&bytes[page * PAGE_BYTES as usize..(page + 1) * PAGE_BYTES as usize])
            );
        }
        got[window.record as usize].extend_from_slice(&bytes);
        pages_seen += u64::from(window.page_count);
        ready_count += 1;
        reader.released(1, window.index).await;
        if pages_seen >= 32 {
            break;
        }
    }
    let expect = expected(&tokens, 1, 0, true);
    assert_eq!(got, expect);
    reader.barrier().await;
    reader.copy_result(1, v2::CopyOutcome::CopyOk).await;
    let terminal = reader.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Copied as i32,
        "{terminal:?}"
    );
    reader.fence(1, v2::FenceOutcome::Drained).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn u2_u8_and_a9_present_queue_and_invalid_metadata() {
    let mut host = host();
    host.config.limits.max_open_puts = 1;
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "writer").await;
    // U2: the same record again is PRESENT, with no transfer.
    store_chain(&mut writer, 1, &[1024]).await;
    match writer
        .put_record(2, key(1024), None, Damage::default())
        .await
    {
        PutResult::Refused(refused) => assert_eq!(refused.reason, v2::Reason::Present as i32),
        other => panic!("{other:?}"),
    }
    // A9: metadata that names another key is refused and nothing is indexed.
    match writer
        .put_record(
            3,
            key(2048),
            Some(key(1024)),
            Damage {
                wrong_key: true,
                ..Damage::default()
            },
        )
        .await
    {
        PutResult::Refused(refused) => assert_eq!(refused.reason, v2::Reason::Invalid as i32),
        other => panic!("{other:?}"),
    }
    let mut reader = open(&mut pool, "reader").await;
    reader.manager_candidate(900, 1024).await;
    let reply = reader
        .lookup_request(v2::LookupRequest {
            lookup_id: 1,
            boundaries: vec![v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            }],
            before_tokens: 5000,
            resident_tokens: 0,
        })
        .await;
    let keys: Vec<u64> = reply
        .candidates
        .iter()
        .map(|c| c.key.as_ref().unwrap().consumed_tokens)
        .collect();
    assert_eq!(keys, vec![1024], "the invalid put was not indexed");
    // A corrupted page is caught by the manager's own check.
    match writer
        .put_record(
            4,
            key(2048),
            Some(key(1024)),
            Damage {
                corrupt_page: Some(0),
                ..Damage::default()
            },
        )
        .await
    {
        PutResult::Refused(refused) => assert_eq!(refused.reason, v2::Reason::Invalid as i32),
        other => panic!("{other:?}"),
    }
    // U8: a put already open fills the writer's one slot.
    match writer
        .put_record(
            5,
            key(3072),
            None,
            Damage {
                cancel_after_accept: true,
                ..Damage::default()
            },
        )
        .await
    {
        PutResult::Refused(_) => {}
        other => panic!("{other:?}"),
    }
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            put_id: 6,
            key: Some(pool::pkey(4096)),
            chain_pages: 64,
            row: true,
            has_published_parent: false,
            published_parent: None,
            bucket: vec![4; 16],
            published_tail_pages: 64,
            mode: v2::CopyMode::SharedWindows as i32,
        }))
        .await;
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            put_id: 7,
            key: Some(pool::pkey(5120)),
            chain_pages: 80,
            row: true,
            has_published_parent: false,
            published_parent: None,
            bucket: vec![4; 16],
            published_tail_pages: 80,
            mode: v2::CopyMode::SharedWindows as i32,
        }))
        .await;
    let refused = writer
        .take(|m| match m {
            selene_manager::proto::v2::server_message::M::PutRefused(refused)
                if refused.put_id == 7 =>
            {
                Some(refused.clone())
            }
            _ => None,
        })
        .await;
    assert_eq!(refused.reason, v2::Reason::Queue as i32);
}

// ---------------------------------------------------------------- recovered records (NVMe)

fn rooted(label: &str) -> (std::path::PathBuf, selene_manager::store::Config) {
    let base = std::env::temp_dir().join(format!("selene-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    let mut config = selene_manager::store::Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        8 * 1024 * 1024,
        64 * 1024 * 1024,
    );
    config.name = support::instance_name(label);
    (base, config)
}

/// A first manager life that stores `tokens` durably and stops without deleting its store.
async fn first_life(config: &selene_manager::store::Config, tokens: &[u64]) {
    let first = config.clone();
    let mut host = Host::open_config(first);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "writer").await;
    store_chain(&mut writer, 1, tokens).await;
    host.durable(tokens.len());
    host.stop();
    host.base = std::env::temp_dir().join(format!("selene-unused-{}", std::process::id()));
}

async fn read_claim(
    reader: &mut Session,
    candidate: &v2::Candidate,
    id: u64,
) -> (
    Vec<(v2::WindowSpec, Vec<u8>, v2::WindowReady)>,
    v2::ClaimGranted,
) {
    reader.claim(reader.claim_for(id, candidate, 0, true)).await;
    let granted = reader.claim_granted(id).await;
    let mut windows = Vec::new();
    let mut pages = 0;
    while pages < granted.total_pages || windows.is_empty() {
        let ready = reader.window_ready(id).await;
        let window = ready.window.unwrap();
        let bytes = reader.shm.as_ref().unwrap().read(&window);
        pages += u64::from(window.page_count);
        reader.released(id, window.index).await;
        windows.push((window, bytes, ready));
    }
    (windows, granted)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c4_a_recovered_record_is_served_from_nvme_with_its_persisted_checksums() {
    let (base, config) = rooted("c4");
    let tokens = [1024u64, 2048];
    first_life(&config, &tokens).await;
    let second = config.clone();
    let mut host = Host::open_config(second);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut reader = open(&mut pool, "reader").await;
    // Recovered records start in the disk tier.
    let candidate = reader.manager_candidate(1, 2048).await;
    assert_eq!(candidate.level, v2::Level::HostNvme as i32);
    assert_eq!(candidate.tier, v2::Tier::Nvme as i32);
    let (windows, granted) = read_claim(&mut reader, &candidate, 1).await;
    assert_eq!(granted.total_pages, 32);
    let mut got: Vec<Vec<u8>> = vec![Vec::new(), Vec::new()];
    for (window, bytes, ready) in &windows {
        for (page, sum) in ready.page_xxh3.iter().enumerate() {
            assert_eq!(*sum, pool::xxh3(&bytes[page * 4096..(page + 1) * 4096]));
        }
        got[window.record as usize].extend_from_slice(bytes);
    }
    assert_eq!(got, expected(&tokens, 1, 0, true));
    reader.barrier().await;
    reader.copy_result(1, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(1).await.state,
        v2::ClaimState::Copied as i32
    );
    reader.fence(1, v2::FenceOutcome::Drained).await;

    host.stop();
    host.base = std::env::temp_dir().join(format!("selene-unused2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_requesters_stream_that_blips_keeps_the_manager_windows_until_it_reports() {
    let (base, config) = rooted("blip");
    let tokens = [1024u64];
    first_life(&config, &tokens).await;
    let second = config.clone();
    let mut host = Host::open_config(second);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let ident = Ident::new("reader");
    let mut reader = pool.open(&ident).await;
    reader.snapshot(vec![]).await;
    let candidate = reader.manager_candidate(1, 1024).await;
    reader.claim(reader.claim_for(1, &candidate, 0, true)).await;
    reader.claim_granted(1).await;
    // Windows arrive but are never released; the stream drops and the process lives on.
    let ready = reader.window_ready(1).await;
    let length = ready.window.unwrap().length;
    let mut watcher = open(&mut pool, "watcher").await;
    assert_eq!(host.gauge("pool_quarantined_bytes"), 0);
    drop(reader);
    let mut quarantined = 0;
    for _ in 0..100 {
        quarantined = host.gauge("pool_quarantined_bytes");
        if quarantined > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(
        quarantined, length,
        "leased to a process that is still alive"
    );
    watcher.barrier().await;
    assert_eq!(host.gauge("pool_quarantined_bytes"), length);
    let mut again = pool.open(&ident).await;
    again.snapshot(vec![]).await;
    again.fence(1, v2::FenceOutcome::Drained).await;
    for _ in 0..100 {
        if host.gauge("pool_quarantined_bytes") == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(host.gauge("pool_quarantined_bytes"), 0);
    host.stop();
    host.base = std::env::temp_dir().join(format!("selene-unused3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a8_damage_at_rest_never_reaches_a_restore() {
    let (base, config) = rooted("a8");
    first_life(&config, &[1024]).await;
    let second = config.clone();
    let mut host = Host::open_config(second);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut reader = open(&mut pool, "reader").await;
    let candidate = reader.manager_candidate(1, 1024).await;
    // A bit flips at rest after recovery and before the record is first read. The
    // manager serves the checksums it persisted, so the requester can tell.
    let mut scoped = descriptor();
    scoped.pool_name = config.name.clone();
    let file = config
        .root
        .as_ref()
        .unwrap()
        .join(scoped.name())
        .join(selene_manager::record::file_name(&key(1024)));
    {
        use std::os::unix::fs::FileExt;
        let handle = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file)
            .unwrap();
        let at = 4096 + 8192 + 10;
        let mut byte = [0u8; 1];
        handle.read_exact_at(&mut byte, at).unwrap();
        byte[0] ^= 0xff;
        handle.write_all_at(&byte, at).unwrap();
    }
    // The store reads whole records back with their checksums: damage found on the way
    // fails the claim before any window carries it. A copy of it that did get out would
    // carry the persisted checksums, so the requester's own check would catch it too.
    reader.claim(reader.claim_for(1, &candidate, 0, true)).await;
    let granted = reader.claim_granted(1).await;
    assert_eq!(granted.total_pages, 16);
    let mut served_damage = false;
    let terminal = loop {
        enum Next {
            Ready(v2::WindowReady),
            Terminal(v2::ClaimTerminal),
        }
        let next = reader
            .take(|m| match m {
                v2::server_message::M::WindowReady(ready) => Some(Next::Ready(ready.clone())),
                v2::server_message::M::ClaimTerminal(terminal) => Some(Next::Terminal(*terminal)),
                _ => None,
            })
            .await;
        match next {
            Next::Ready(ready) => {
                let window = ready.window.unwrap();
                let bytes = reader.shm.as_ref().unwrap().read(&window);
                for (page, sum) in ready.page_xxh3.iter().enumerate() {
                    if *sum != pool::xxh3(&bytes[page * 4096..(page + 1) * 4096]) {
                        served_damage = true;
                    }
                }
                reader.released(1, window.index).await;
            }
            Next::Terminal(terminal) => break terminal,
        }
        if served_damage {
            reader.copy_result(1, v2::CopyOutcome::CopyChecksum).await;
        }
    };
    assert_ne!(terminal.state, v2::ClaimState::Copied as i32);
    host.stop();
    host.base = std::env::temp_dir().join(format!("selene-unused-a8-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn u4_a_cancelled_put_keeps_its_unfilled_windows_until_the_writers_fence() {
    let mut host = host();
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "writer").await;
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            put_id: 1,
            key: Some(pool::pkey(1024)),
            chain_pages: 16,
            row: true,
            has_published_parent: false,
            published_parent: None,
            bucket: vec![4; 16],
            published_tail_pages: 16,
            mode: v2::CopyMode::SharedWindows as i32,
        }))
        .await;
    writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(accept) if accept.put_id == 1 => Some(()),
            _ => None,
        })
        .await;
    let grant = writer
        .take(|m| match m {
            v2::server_message::M::PutGrant(grant) if grant.put_id == 1 => Some(*grant),
            _ => None,
        })
        .await;
    assert!(grant.window.unwrap().length > 0);
    host.wait(|host| {
        host.gauge("memory_allocated_bytes") + host.gauge("staging_allocated_bytes") > 0
    });
    let held = host.gauge("memory_allocated_bytes") + host.gauge("staging_allocated_bytes");
    assert!(held > 0, "a granted window is memory in use");
    writer
        .send(v2::client_message::M::PutCancel(v2::PutCancel {
            put_id: 1,
        }))
        .await;
    writer.barrier().await;
    assert_eq!(
        host.gauge("memory_allocated_bytes") + host.gauge("staging_allocated_bytes"),
        held,
        "the writer may still be writing the window"
    );
    writer
        .send(v2::client_message::M::FenceAck(v2::FenceAck {
            subject: Some(v2::fence_ack::Subject::PutId(1)),
            outcome: v2::FenceOutcome::Drained as i32,
        }))
        .await;
    let mut settled = false;
    for _ in 0..100 {
        if host.gauge("memory_allocated_bytes") + host.gauge("staging_allocated_bytes") == 0 {
            settled = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(settled, "the fence releases the window");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_nvme_limit_bounds_the_reservations_it_reports_however_many_puts_pass_it() {
    // A limit of 1 MiB, and records of about 70 KiB: fifteen fill it, and forty are offered.
    let limit: u64 = 1 << 20;
    let mut host = Host::with_storage(256 * 1024, limit);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut writer = open(&mut pool, "writer").await;
    let (mut stored, mut refused) = (0, 0);
    for index in 0..40u64 {
        match writer
            .put_record(index + 1, key(1024 + 64 * index), None, Damage::default())
            .await
        {
            PutResult::Stored(_) => stored += 1,
            PutResult::Refused(_) => refused += 1,
        }
        // The gauge follows the store's clock; the peak is kept at every admission.
        assert!(
            host.gauge("nvme_reserved_bytes") <= limit,
            "reserved {} after put {index}",
            host.gauge("nvme_reserved_bytes")
        );
    }
    assert!(stored >= 10, "{stored} stored, {refused} refused");
    host.wait(|host| host.gauge("open_ops") == 0);
    let peak = host.gauge("nvme_reserved_peak_bytes");
    assert!(
        peak > limit / 2 && peak <= limit,
        "the reservations peaked at {peak} bytes against a limit of {limit}"
    );
    assert!(host.gauge("nvme_reserved_bytes") <= peak);
    assert_eq!(host.gauge("pool_quarantined_bytes"), 0);
}
