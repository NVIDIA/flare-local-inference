//! Pool protocol, instances on other hosts: a put and a claim whose bytes move on the
//! data channel, over TCP and over TLS, and what happens when the instance misbehaves,
//! stalls or dies in the middle of a transfer.

mod support;

use selene_manager::{Key, proto::v2};
use std::time::Duration;
use support::{
    channel::{self, Ended, Security},
    content_for, descriptor, key,
    net::Net,
    pool::{self, Damage, Ident, PAGE_BYTES, Pool, PutResult, Session},
};
use tokio::io::AsyncReadExt;

async fn open(pool: &mut Pool, name: &str) -> Session {
    let mut session = pool.open(&Ident::new(name)).await;
    assert_eq!(session.opened.status, v2::OpenStatus::OpenOk as i32);
    session.snapshot(vec![]).await;
    session
}

/// A manager whose claims expire quickly when the instance stops answering.
fn net() -> Net {
    Net::start_with(
        |_| {},
        |store| {
            store.limits.progress_timeout_ms = 600;
        },
    )
}

async fn store_chain(writer: &mut Session, security: &Security, first_id: u64, tokens: &[u64]) {
    let mut parent: Option<Key> = None;
    for (index, token) in tokens.iter().enumerate() {
        let record = key(*token);
        match channel::put_record(
            writer,
            security,
            first_id + index as u64,
            record,
            parent,
            Damage::default(),
        )
        .await
        {
            PutResult::Stored(bytes) => assert!(bytes > 0),
            other => panic!("put failed: {other:?}"),
        }
        parent = Some(record);
    }
}

/// The bytes the manager must serve for a stored chain, record by record.
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

fn data_claim(reader: &Session, id: u64, candidate: &v2::Candidate, begin: u64) -> v2::Claim {
    let mut claim = reader.claim_for(id, candidate, begin, true);
    claim.mode = v2::CopyMode::DataChannel as i32;
    claim
}

/// Restore `candidate` over the channel and return what arrived, per record.
async fn restore(
    reader: &mut Session,
    security: &Security,
    id: u64,
    candidate: &v2::Candidate,
    records: usize,
) -> (Vec<Vec<u8>>, v2::ClaimGranted) {
    reader.claim(data_claim(reader, id, candidate, 0)).await;
    let granted = reader.claim_granted(id).await;
    let reference = granted.channel.clone().expect("a data-channel grant");
    assert_eq!(reference.transport, v2::DataTransport::Tcp as i32);
    assert_eq!(reference.token.len(), 32);
    assert!(
        granted.window_credits > 0 && !granted.retained,
        "a host-tier source streams"
    );
    let mut conn = channel::dial(&reference, security).await;
    let (windows, ended) = channel::read_windows(&mut conn).await;
    assert_eq!(ended, Ended::End);
    let mut got = vec![Vec::new(); records];
    let mut pages = 0u64;
    for (next, received) in windows.iter().enumerate() {
        assert_eq!(
            received.window.shape.index, next as u32,
            "windows arrive in order"
        );
        let shape = received.window.shape;
        for (page, sum) in received.window.page_xxh3.iter().enumerate() {
            let at = page * PAGE_BYTES as usize;
            assert_eq!(
                *sum,
                pool::xxh3(&received.payload[at..at + PAGE_BYTES as usize]),
                "the stored checksum is of the bytes sent"
            );
        }
        got[shape.record as usize].extend_from_slice(&received.payload);
        pages += u64::from(shape.page_count);
    }
    assert_eq!(pages, granted.total_pages);
    (got, granted)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d1_a_remote_instance_stores_and_restores_a_chain_over_tcp() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    let tokens = [1024u64, 2048];
    store_chain(&mut writer, &Security::Plain, 1, &tokens).await;
    let mut reader = open(&mut tcp, "reader").await;
    let candidate = reader.manager_candidate(1, 2048).await;
    assert_eq!(candidate.level, v2::Level::RemoteNvme as i32);
    let (got, granted) = restore(&mut reader, &Security::Plain, 1, &candidate, 2).await;
    assert!(granted.row);
    assert_eq!(got, expected(&tokens, 1, 0, true));
    reader.copy_result(1, v2::CopyOutcome::CopyOk).await;
    let terminal = reader.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Copied as i32,
        "{terminal:?}"
    );
    reader.fence(1, v2::FenceOutcome::Drained).await;
    // The windows came back: a second restore of the same record needs them again.
    let (again, _) = restore(&mut reader, &Security::Plain, 2, &candidate, 2).await;
    assert_eq!(again, got);
    reader.copy_result(2, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(2).await.state,
        v2::ClaimState::Copied as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d1_a_restore_may_start_mid_record() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    let tokens = [1024u64, 2048];
    store_chain(&mut writer, &Security::Plain, 1, &tokens).await;
    let mut reader = open(&mut tcp, "reader").await;
    let candidate = reader.manager_candidate(1, 2048).await;
    // The requester already holds 640 tokens (ten pages): the first record is entered at its tenth page.
    reader.claim(data_claim(&reader, 1, &candidate, 640)).await;
    let granted = reader.claim_granted(1).await;
    assert_eq!(granted.records[0].first_page, 10);
    let mut conn = channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    let (windows, ended) = channel::read_windows(&mut conn).await;
    assert_eq!(ended, Ended::End);
    let mut got = vec![Vec::new(); 2];
    for received in &windows {
        got[received.window.shape.record as usize].extend_from_slice(&received.payload);
    }
    assert_eq!(got, expected(&tokens, 1, 640, true));
    reader.copy_result(1, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(1).await.state,
        v2::ClaimState::Copied as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d3_an_instance_that_never_connects_expires_and_its_windows_return() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    let tokens = [1024u64, 2048];
    store_chain(&mut writer, &Security::Plain, 1, &tokens).await;
    let mut reader = open(&mut tcp, "reader").await;
    let candidate = reader.manager_candidate(1, 2048).await;
    reader.claim(data_claim(&reader, 1, &candidate, 0)).await;
    let granted = reader.claim_granted(1).await;
    let reference = granted.channel.unwrap();
    let terminal = reader.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    assert_eq!(terminal.reason, v2::ClaimReason::NoProgress as i32);
    // The token died with the claim.
    let mut late = channel::dial(&reference, &Security::Plain).await;
    let (windows, ended) = channel::read_windows(&mut late).await;
    assert!(windows.is_empty() && ended == Ended::Broken);
    reader.fence(1, v2::FenceOutcome::Drained).await;
    // Nothing is quarantined and the same record restores.
    let (got, _) = restore(&mut reader, &Security::Plain, 2, &candidate, 2).await;
    assert_eq!(got, expected(&tokens, 1, 0, true));
    reader.copy_result(2, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(2).await.state,
        v2::ClaimState::Copied as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d4_a_peer_that_dies_mid_transfer_fails_the_claim_and_frees_the_manager() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    let tokens = [1024u64, 2048];
    store_chain(&mut writer, &Security::Plain, 1, &tokens).await;
    let mut reader = open(&mut tcp, "reader").await;
    let candidate = reader.manager_candidate(1, 2048).await;
    reader.claim(data_claim(&reader, 1, &candidate, 0)).await;
    let granted = reader.claim_granted(1).await;
    let mut conn = channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    // Read a few bytes of the first window and die.
    let mut some = [0u8; 100];
    conn.read_exact(&mut some).await.unwrap();
    drop(conn);
    let terminal = reader.claim_terminal(1).await;
    assert!(
        terminal.state == v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    reader.fence(1, v2::FenceOutcome::Drained).await;
    let (got, _) = restore(&mut reader, &Security::Plain, 2, &candidate, 2).await;
    assert_eq!(got, expected(&tokens, 1, 0, true));
    reader.copy_result(2, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        reader.claim_terminal(2).await.state,
        v2::ClaimState::Copied as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d5_a_claim_released_before_the_copy_stops_its_transfer() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    store_chain(&mut writer, &Security::Plain, 1, &[1024]).await;
    let mut reader = open(&mut tcp, "reader").await;
    let candidate = reader.manager_candidate(1, 1024).await;
    reader.claim(data_claim(&reader, 1, &candidate, 0)).await;
    let granted = reader.claim_granted(1).await;
    let mut conn = channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    reader
        .send(v2::client_message::M::ClaimRelease(v2::ClaimRelease {
            claim_id: 1,
        }))
        .await;
    let terminal = reader.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::Cancelled as i32);
    assert!(terminal.fence_required);
    // The manager closes the connection: reading ends.
    let mut sink = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), conn.read_to_end(&mut sink))
        .await
        .expect("the manager closed the connection");
    reader.fence(1, v2::FenceOutcome::Drained).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d6_a_put_whose_writer_dies_mid_window_fails_and_the_same_record_stores_after() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    // Begin a put, take its grant, send half of the first window and die.
    writer
        .send(v2::client_message::M::PutBegin(v2::PutBegin {
            put_id: 1,
            key: Some(pool::pkey(2048)),
            chain_pages: 32,
            row: true,
            has_published_parent: false,
            published_parent: None,
            bucket: vec![4; 16],
            published_tail_pages: 32,
            mode: v2::CopyMode::DataChannel as i32,
        }))
        .await;
    let accept = writer
        .take(|m| match m {
            v2::server_message::M::PutAccept(accept) if accept.put_id == 1 => Some(accept.clone()),
            _ => None,
        })
        .await;
    let mut conn = channel::dial(accept.channel.as_ref().unwrap(), &Security::Plain).await;
    let frame = flare_selene::shared::frame::ToInstance::read(&mut conn, channel::MAX_PAGES)
        .await
        .unwrap();
    let flare_selene::shared::frame::ToInstance::Grant(shape) = frame else {
        panic!("expected a grant");
    };
    flare_selene::shared::frame::ToManager::Filled {
        index: shape.index,
        row_xxh3: None,
        page_xxh3: vec![0; shape.page_count as usize],
    }
    .write(&mut conn)
    .await
    .unwrap();
    tokio::io::AsyncWriteExt::write_all(&mut conn, &vec![1u8; shape.length as usize / 2])
        .await
        .unwrap();
    drop(conn);
    let refused = writer
        .take(|m| match m {
            v2::server_message::M::PutRefused(refused) if refused.put_id == 1 => {
                Some(refused.clone())
            }
            _ => None,
        })
        .await;
    assert_eq!(refused.reason, v2::Reason::Io as i32, "{}", refused.detail);
    // Nothing was indexed and the writer may try again.
    match channel::put_record(
        &mut writer,
        &Security::Plain,
        2,
        key(2048),
        None,
        Damage::default(),
    )
    .await
    {
        PutResult::Stored(bytes) => assert!(bytes > 0),
        other => panic!("{other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d7_a_corrupt_put_and_a_cancelled_put_over_the_channel() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    match channel::put_record(
        &mut writer,
        &Security::Plain,
        1,
        key(1024),
        None,
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
    match channel::put_record(
        &mut writer,
        &Security::Plain,
        2,
        key(1024),
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
    // Both ended; the record stores cleanly.
    match channel::put_record(
        &mut writer,
        &Security::Plain,
        3,
        key(1024),
        None,
        Damage::default(),
    )
    .await
    {
        PutResult::Stored(bytes) => assert!(bytes > 0),
        other => panic!("{other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d8_a_session_lost_mid_claim_ends_the_transfer() {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut writer = open(&mut tcp, "writer").await;
    store_chain(&mut writer, &Security::Plain, 1, &[1024]).await;
    let mut reader = open(&mut tcp, "reader").await;
    let candidate = reader.manager_candidate(1, 1024).await;
    reader.claim(data_claim(&reader, 1, &candidate, 0)).await;
    let granted = reader.claim_granted(1).await;
    let mut conn = channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    // The control stream goes away (the instance's process died): the data connection closes.
    drop(reader);
    let mut sink = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), conn.read_to_end(&mut sink))
        .await
        .expect("the manager ended the transfer");
    let mut other = open(&mut tcp, "other").await;
    other.claim(data_claim(&other, 1, &candidate, 0)).await;
    let granted = other.claim_granted(1).await;
    let mut conn = channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    let (windows, ended) = channel::read_windows(&mut conn).await;
    assert_eq!(ended, Ended::End);
    assert!(!windows.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d9_the_data_channel_needs_a_listener() {
    // A manager with no network listener serves only shared windows.
    let mut host = support::Host::dram_only(8 * 1024 * 1024);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut session = open(&mut pool, "local").await;
    let mut claim = v2::Claim {
        claim_id: 1,
        source: Some(v2::SourceRef::default()),
        tier: v2::Tier::Dram as i32,
        key: Some(pool::pkey(1024)),
        begin_tokens: 0,
        want_row: true,
        mode: v2::CopyMode::DataChannel as i32,
        deadline_ms: 0,
    };
    session.claim(claim.clone()).await;
    assert_eq!(
        session.claim_terminal(1).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
    claim.claim_id = 2;
    let _ = claim;
    match channel::put_record(
        &mut session,
        &Security::Plain,
        1,
        key(1024),
        None,
        Damage::default(),
    )
    .await
    {
        PutResult::Refused(refused) => {
            assert_eq!(refused.reason, v2::Reason::Invalid as i32);
            assert!(
                refused.detail.contains("data channel"),
                "{}",
                refused.detail
            );
        }
        other => panic!("{other:?}"),
    }
}

// ---- RDMA (experimental, behind the `rdma` feature) ---------------------------------

#[cfg(feature = "rdma")]
mod over_rdma {
    use super::*;
    use support::rdma::Instance;

    fn rdma_net() -> Net {
        Net::start_with(
            |config| config.rdma = true,
            |store| store.limits.progress_timeout_ms = 600,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn r1_a_chain_is_stored_and_restored_over_rdma() {
        let net = rdma_net();
        let mut tcp = net.tcp().await;
        assert!(
            tcp.hello
                .data_transports
                .contains(&(v2::DataTransport::Rdma as i32))
        );
        let instance = Instance::new();
        let mut writer = open(&mut tcp, "writer").await;
        let tokens = [1024u64, 2048];
        let mut parent: Option<Key> = None;
        for (index, token) in tokens.iter().enumerate() {
            let record = key(*token);
            match channel::put_record(
                &mut writer,
                &Security::Plain,
                1 + index as u64,
                record,
                parent,
                Damage::default(),
            )
            .await
            {
                PutResult::Stored(bytes) => assert!(bytes > 0),
                other => panic!("{other:?}"),
            }
            parent = Some(record);
        }
        let mut reader = open(&mut tcp, "reader").await;
        let candidate = reader.manager_candidate(1, 2048).await;
        reader.claim(data_claim(&reader, 1, &candidate, 0)).await;
        let granted = reader.claim_granted(1).await;
        let reference = granted.channel.clone().unwrap();
        assert_eq!(reference.transport, v2::DataTransport::Rdma as i32);
        let mut dialed = instance.dial(&reference).await;
        let (windows, ended) = instance.read_windows(&mut dialed).await;
        assert!(ended);
        let mut got = vec![Vec::new(); 2];
        for received in &windows {
            for (page, sum) in received.window.page_xxh3.iter().enumerate() {
                let at = page * PAGE_BYTES as usize;
                assert_eq!(
                    *sum,
                    pool::xxh3(&received.payload[at..at + PAGE_BYTES as usize])
                );
            }
            got[received.window.shape.record as usize].extend_from_slice(&received.payload);
        }
        assert_eq!(got, expected(&tokens, 1, 0, true));
        reader.copy_result(1, v2::CopyOutcome::CopyOk).await;
        assert_eq!(
            reader.claim_terminal(1).await.state,
            v2::ClaimState::Copied as i32
        );
        reader.fence(1, v2::FenceOutcome::Drained).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn r2_an_instance_that_dies_mid_transfer_frees_the_manager() {
        let net = rdma_net();
        let mut tcp = net.tcp().await;
        let instance = Instance::new();
        let mut writer = open(&mut tcp, "writer").await;
        match channel::put_record(
            &mut writer,
            &Security::Plain,
            1,
            key(2048),
            None,
            Damage::default(),
        )
        .await
        {
            PutResult::Stored(_) => {}
            other => panic!("{other:?}"),
        }
        let mut reader = open(&mut tcp, "reader").await;
        let candidate = reader.manager_candidate(1, 2048).await;
        reader.claim(data_claim(&reader, 1, &candidate, 0)).await;
        let granted = reader.claim_granted(1).await;
        // Connect, lend a slot and die before reading anything: queue pair and socket both go.
        let dialed = instance.dial(granted.channel.as_ref().unwrap()).await;
        drop(dialed);
        let terminal = reader.claim_terminal(1).await;
        assert!(
            terminal.state == v2::ClaimState::Failed as i32,
            "{terminal:?}"
        );
        reader.fence(1, v2::FenceOutcome::Drained).await;
        // The windows came back and the manager's regions are not quarantined: it serves again.
        reader.claim(data_claim(&reader, 2, &candidate, 0)).await;
        let granted = reader.claim_granted(2).await;
        let mut dialed = instance.dial(granted.channel.as_ref().unwrap()).await;
        let (windows, ended) = instance.read_windows(&mut dialed).await;
        assert!(ended && !windows.is_empty());
        reader.copy_result(2, v2::CopyOutcome::CopyOk).await;
        assert_eq!(
            reader.claim_terminal(2).await.state,
            v2::ClaimState::Copied as i32
        );
    }
}
