//! Pool protocol, copies between instances that are not on the manager's host: a source
//! reached over TCP writes the windows of an export to the manager on the data channel,
//! and the manager relays them to the requester, over its own data channel or through
//! shared windows. Both ends are scripted; the manager is real.

mod support;

use flare_selene::shared::frame::{ToInstance, ToManager};
use selene_manager::proto::v2;
use std::time::Duration;
use support::{
    channel::{self, Conn, Ended, MAX_PAGES, Security},
    net::Net,
    pool::{
        self, Ident, PAGE_BYTES, PAGE_TOKENS, Pool, Session, chain, page_bytes, records_for,
        row_bytes, saved,
    },
};
use tokio::{io::AsyncWriteExt, time::timeout};

const SEED: u8 = 17;
const TOKENS: [u64; 2] = [1024, 2048];

/// A manager whose claims expire quickly when a party stops answering, and whose windows
/// are small, so a chain takes several.
fn net() -> Net {
    Net::start_with(
        |_| {},
        |store| {
            store.limits.pin_deadline_ms = 600;
            store.limits.progress_timeout_ms = 600;
            store.limits.max_window_bytes = 8 * PAGE_BYTES;
        },
    )
}

struct World {
    net: Net,
    tcp: Pool,
    source: Session,
    requester: Session,
    remote_requester: bool,
}

async fn world(tier: v2::Tier, remote_requester: bool) -> World {
    let net = net();
    let mut tcp = net.tcp().await;
    let mut source = tcp.open(&Ident::new("src")).await;
    assert_eq!(
        source.opened.export_slots, 2,
        "a session without shared memory serves exports"
    );
    assert!(source.shm.is_none());
    source.snapshot(vec![]).await;
    source
        .notices(chain(&TOKENS, tier).into_iter().map(saved).collect())
        .await;
    let mut requester = if remote_requester {
        tcp.open(&Ident::new("req")).await
    } else {
        net.unix().await.open(&Ident::new("req")).await
    };
    requester.snapshot(vec![]).await;
    World {
        net,
        tcp,
        source,
        requester,
        remote_requester,
    }
}

impl World {
    async fn claim(&mut self, id: u64) -> v2::Claim {
        let reply = self.requester.lookup(900 + id, TOKENS[1], 0).await;
        let candidate = reply
            .candidates
            .into_iter()
            .find(|c| c.key.as_ref().unwrap().consumed_tokens == TOKENS[1])
            .expect("a candidate for the endpoint");
        assert!(
            candidate
                .source
                .as_ref()
                .is_some_and(|s| !s.instance_id.is_empty()),
            "the source is the other instance"
        );
        let mut claim = self.requester.claim_for(id, &candidate, 0, true);
        if self.remote_requester {
            claim.mode = v2::CopyMode::DataChannel as i32;
        }
        self.requester.claim(claim.clone()).await;
        claim
    }

    /// The source pins the whole chain as an honest source would.
    async fn pin(&mut self) -> (v2::ExportRequest, Vec<v2::Record>) {
        let request = self.source.export_request().await;
        assert!(
            request.channel.is_some(),
            "the export names the channel its windows go to"
        );
        let records = records_for(
            &TOKENS,
            request.begin_tokens / PAGE_TOKENS,
            true,
        );
        self.source.pin_ok(&request, records.clone(), true).await;
        (request, records)
    }
}

/// The bytes an honest source writes into a window.
fn content(records: &[v2::Record], record: u32, first_page: u32, pages: u32, row: bool) -> Vec<u8> {
    let rec = &records[record as usize];
    let node = TOKENS
        .iter()
        .position(|t| *t == rec.key.as_ref().unwrap().consumed_tokens)
        .unwrap();
    let start = if node == 0 {
        0
    } else {
        TOKENS[node - 1] / PAGE_TOKENS
    };
    let mut bytes = Vec::new();
    for page in 0..u64::from(pages) {
        bytes.extend(page_bytes(SEED, start + u64::from(first_page) + page));
    }
    if row {
        bytes.extend(row_bytes(SEED, TOKENS[node]));
    }
    bytes
}

/// What a source does with the grants on its channel.
#[derive(Clone, Copy, Default)]
struct Source {
    /// Fill this many windows, then stop (return with the connection still open).
    stop_after: Option<usize>,
}

/// Dial the export's channel and fill the grants as an honest source would, until every
/// page and the row are written. Returns the connection and the windows filled.
async fn serve(
    request: &v2::ExportRequest,
    records: &[v2::Record],
    behaviour: Source,
) -> (Conn, usize) {
    let mut conn = channel::dial(request.channel.as_ref().unwrap(), &Security::Plain).await;
    let total: u64 = records.iter().map(|r| u64::from(r.pages)).sum();
    let (mut pages, mut row, mut served) = (0u64, false, 0usize);
    while pages < total || !row {
        if behaviour.stop_after == Some(served) {
            break;
        }
        let frame = match timeout(
            Duration::from_secs(10),
            ToInstance::read(&mut conn, MAX_PAGES),
        )
        .await
        {
            Ok(Ok(frame)) => frame,
            _ => break,
        };
        let ToInstance::Grant(shape) = frame else {
            panic!("an export's channel carries grants, not {frame:?}");
        };
        let bytes = content(
            records,
            shape.record,
            shape.first_page,
            shape.page_count,
            shape.row,
        );
        assert_eq!(bytes.len() as u64, shape.length);
        let hashes = (0..shape.page_count as usize)
            .map(|p| pool::xxh3(&bytes[p * PAGE_BYTES as usize..(p + 1) * PAGE_BYTES as usize]))
            .collect();
        let row_hash = shape
            .row
            .then(|| pool::xxh3(&bytes[shape.page_count as usize * PAGE_BYTES as usize..]));
        let sent = async {
            ToManager::Filled {
                index: shape.index,
                row_xxh3: row_hash,
                page_xxh3: hashes,
            }
            .write(&mut conn)
            .await?;
            conn.write_all(&bytes).await?;
            conn.flush().await
        };
        if sent.await.is_err() {
            // The manager closed the source leg mid-window.
            break;
        }
        pages += u64::from(shape.page_count);
        row |= shape.row;
        served += 1;
    }
    (conn, served)
}

/// Read the windows of a claim from a remote requester's channel and check them.
async fn read_remote(
    granted: &v2::ClaimGranted,
    records: &[v2::Record],
) -> (usize, channel::Ended) {
    let mut conn = channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    let (windows, ended) = channel::read_windows(&mut conn).await;
    for received in &windows {
        let s = received.window.shape;
        assert_eq!(
            received.payload,
            content(records, s.record, s.first_page, s.page_count, s.row),
            "window {} bytes",
            s.index
        );
        for (page, sum) in received.window.page_xxh3.iter().enumerate() {
            let at = page * PAGE_BYTES as usize;
            assert_eq!(
                *sum,
                pool::xxh3(&received.payload[at..at + PAGE_BYTES as usize])
            );
        }
    }
    (windows.len(), ended)
}

/// Read the windows of a claim from shared memory until every page and the row came.
async fn read_shared(requester: &mut Session, claim: u64, records: &[v2::Record]) -> usize {
    let total: u64 = records.iter().map(|r| u64::from(r.pages)).sum();
    let (mut pages, mut row, mut count) = (0u64, false, 0usize);
    while pages < total || !row {
        let ready = requester.window_ready(claim).await;
        let window = ready.window.unwrap();
        let got = requester.shm.as_ref().unwrap().read(&window);
        assert_eq!(
            got,
            content(
                records,
                window.record,
                window.first_page,
                window.page_count,
                window.row
            ),
            "window {} bytes",
            window.index
        );
        requester.released(claim, window.index).await;
        pages += u64::from(window.page_count);
        row |= window.row;
        count += 1;
    }
    count
}

async fn relay_one(w: &mut World, id: u64) {
    w.claim(id).await;
    let (request, records) = w.pin().await;
    let remote = w.remote_requester;
    let granted = w.requester.claim_granted(id).await;
    assert_eq!(granted.channel.is_some(), remote);
    let source = serve(&request, &records, Source::default());
    let requester = async {
        if remote {
            let (windows, ended) = read_remote(&granted, &records).await;
            assert_eq!(ended, Ended::End);
            windows
        } else {
            read_shared(&mut w.requester, id, &records).await
        }
    };
    let ((_conn, served), windows) = tokio::join!(source, requester);
    assert!(served > 2, "the chain takes several windows ({served})");
    assert_eq!(served, windows, "every window was relayed");
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.requester.copy_result(id, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        w.requester.claim_terminal(id).await.state,
        v2::ClaimState::Copied as i32
    );
    if remote {
        w.requester.fence(id, v2::FenceOutcome::Drained).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r4_a_source_that_dies_mid_relay_fails_the_claim_and_frees_the_manager() {
    let mut w = world(v2::Tier::Dram, true).await;
    w.claim(1).await;
    let (request, records) = w.pin().await;
    let granted = w.requester.claim_granted(1).await;
    // The source fills one window and dies: its connection and its session.
    let (conn, served) = serve(
        &request,
        &records,
        Source {
            stop_after: Some(1),
        },
    )
    .await;
    assert_eq!(served, 1);
    drop(conn);
    let (_, ended) = read_remote(&granted, &records).await;
    assert_ne!(ended, Ended::End, "the transfer did not complete");
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportFailed,
            v2::FenceOutcome::Drained,
        )
        .await;
    // The manager serves the same record again from the same source.
    relay_one(&mut w, 2).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r5_a_requester_that_dies_mid_relay_cancels_the_export_and_stops_the_source_leg() {
    let mut w = world(v2::Tier::Dram, true).await;
    w.claim(1).await;
    let (request, records) = w.pin().await;
    let granted = w.requester.claim_granted(1).await;
    let mut requester_conn =
        channel::dial(granted.channel.as_ref().unwrap(), &Security::Plain).await;
    let source = serve(&request, &records, Source::default());
    let requester = async {
        // Read a few bytes of the first window and die.
        use tokio::io::AsyncReadExt;
        let mut some = [0u8; 100];
        requester_conn.read_exact(&mut some).await.unwrap();
        drop(requester_conn);
    };
    let _ = tokio::join!(source, requester);
    let terminal = w.requester.claim_terminal(1).await;
    assert!(
        terminal.state == v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    // The source is told to stop and releases what it pinned.
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, request.export_id);
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    relay_one(&mut w, 2).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r6_a_stalled_source_leg_expires_the_claim_and_the_manager_closes_it() {
    let mut w = world(v2::Tier::Dram, false).await;
    w.claim(1).await;
    let (request, records) = w.pin().await;
    let _granted = w.requester.claim_granted(1).await;
    // The source connects and fills one window, then goes quiet with the connection open.
    let (mut conn, served) = serve(
        &request,
        &records,
        Source {
            stop_after: Some(1),
        },
    )
    .await;
    assert_eq!(served, 1);
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    assert_eq!(terminal.reason, v2::ClaimReason::NoProgress as i32);
    // The manager stopped the leg: the quiet source finds its connection closed.
    // Grants already sent are still in the socket; then the connection ends.
    let ended = timeout(Duration::from_secs(5), async {
        loop {
            match ToInstance::read_or_end(&mut conn, MAX_PAGES).await {
                Ok(Some(ToInstance::Grant(_))) => continue,
                other => break other,
            }
        }
    })
    .await
    .expect("the manager closed the stalled leg");
    assert!(
        !matches!(ended, Ok(Some(_))),
        "the stalled leg ended with {ended:?}"
    );
    // A window the requester held is released, then everything is fenced.
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.source.export_cancel().await;
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    relay_one(&mut w, 2).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r7_a_source_that_never_connects_expires_the_claim_and_returns_the_windows() {
    let mut w = world(v2::Tier::Gpu, true).await;
    w.claim(1).await;
    let (request, _records) = w.pin().await;
    let _granted = w.requester.claim_granted(1).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.source.export_cancel().await;
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    relay_one(&mut w, 2).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r8_a_source_session_lost_mid_relay_fails_the_claim() {
    let mut w = world(v2::Tier::Dram, true).await;
    w.claim(1).await;
    let (request, records) = w.pin().await;
    let granted = w.requester.claim_granted(1).await;
    let (conn, _) = serve(
        &request,
        &records,
        Source {
            stop_after: Some(1),
        },
    )
    .await;
    // The whole session goes: the manager learns by the session, not by the export.
    let World {
        net,
        tcp,
        source,
        mut requester,
        ..
    } = w;
    drop(source);
    let _keep = tcp;
    let terminal = requester.claim_terminal(1).await;
    assert_eq!(
        terminal.state,
        v2::ClaimState::Failed as i32,
        "{terminal:?}"
    );
    assert_eq!(terminal.reason, v2::ClaimReason::SourceLost as i32);
    drop(conn);
    let _ = read_remote(&granted, &records).await;
    requester.fence(1, v2::FenceOutcome::Drained).await;
    drop(net);
}
