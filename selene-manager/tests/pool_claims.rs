//! Pool protocol, claims: a requester, a source and the manager between them.
//! Both ends are scripted over the wire; the manager is real.

mod support;

use selene_manager::proto::v2::{self, server_message::M};
use std::time::Duration;
use support::{
    Host,
    pool::{
        self, Ident, PAGE_BYTES, PAGE_TOKENS, Pool, ROW_BYTES, Session, chain, page_bytes,
        records_for, row_bytes, saved,
    },
};

const SEED: u8 = 11;

fn host() -> Host {
    let mut host = Host::dram_only(2 * 1024 * 1024);
    host.config.limits.pin_deadline_ms = 300;
    host.config.limits.progress_timeout_ms = 600;
    host
}

struct World {
    host: Host,
    pool: Pool,
    source: Session,
    requester: Session,
    tokens: Vec<u64>,
    source_ident: Ident,
    requester_ident: Ident,
}

async fn world(tier: v2::Tier, tokens: &[u64]) -> World {
    world_with(tier, tokens, |_| {}).await
}

async fn world_with(tier: v2::Tier, tokens: &[u64], tweak: impl FnOnce(&mut Host)) -> World {
    let mut host = host();
    tweak(&mut host);
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let source_ident = Ident::new("src");
    let requester_ident = Ident::new("req");
    let mut source = pool.open(&source_ident).await;
    source.snapshot(vec![]).await;
    source
        .notices(chain(tokens, tier).into_iter().map(saved).collect())
        .await;
    let mut requester = pool.open(&requester_ident).await;
    requester.snapshot(vec![]).await;
    World {
        host,
        pool,
        source,
        requester,
        tokens: tokens.to_vec(),
        source_ident,
        requester_ident,
    }
}

impl World {
    async fn candidate(&mut self, id: u64, upto: u64) -> v2::Candidate {
        let reply = self.requester.lookup(id, upto, 0).await;
        reply
            .candidates
            .into_iter()
            .find(|c| c.key.as_ref().unwrap().consumed_tokens == upto)
            .expect("a candidate for the endpoint")
    }

    async fn claim(&mut self, id: u64, upto: u64, begin: u64, want_row: bool) -> v2::Claim {
        let candidate = self.candidate(900 + id, upto).await;
        let claim = self.requester.claim_for(id, &candidate, begin, want_row);
        self.requester.claim(claim.clone()).await;
        claim
    }

    fn staging_used(&self) -> u64 {
        self.host.gauge("staging_allocated_bytes")
    }

    async fn staging_now(&self) -> u64 {
        self.host.wait(|host| host.gauge("staging_allocated_bytes") > 0);
        self.staging_used()
    }

    async fn staging_settles_at(&self, bytes: u64) {
        self.host.wait(|host| host.gauge("staging_allocated_bytes") == bytes);
    }

    /// The source pins the whole chain as an honest source would.
    async fn pin(&mut self, request: &v2::ExportRequest) -> Vec<v2::Record> {
        let endpoint = request.key.as_ref().unwrap().consumed_tokens;
        let upto: Vec<u64> = self
            .tokens
            .iter()
            .copied()
            .filter(|t| *t <= endpoint)
            .collect();
        let records = records_for(
            &upto,
            request.begin_tokens / PAGE_TOKENS,
            request.want_row,
        );
        self.source
            .pin_ok(request, records.clone(), request.want_row)
            .await;
        records
    }

    /// The bytes an honest source writes into a window.
    fn content(&self, records: &[v2::Record], window: &v2::WindowSpec) -> Vec<u8> {
        let record = &records[window.record as usize];
        let node = self
            .tokens
            .iter()
            .position(|t| *t == record.key.as_ref().unwrap().consumed_tokens)
            .unwrap();
        let start = if node == 0 {
            0
        } else {
            self.tokens[node - 1] / PAGE_TOKENS
        };
        let mut bytes = Vec::new();
        for page in 0..u64::from(window.page_count) {
            bytes.extend(page_bytes(
                SEED,
                start + u64::from(window.first_page) + page,
            ));
        }
        if window.row {
            bytes.extend(row_bytes(SEED, self.tokens[node]));
        }
        bytes
    }

    /// Runs one claim to COPIED with an honest source and requester.
    async fn copy_all(
        &mut self,
        claim_id: u64,
        request: &v2::ExportRequest,
        records: &[v2::Record],
    ) {
        let fill = self.source.export_fill().await;
        self.fill_and_read(claim_id, request, records, fill.windows)
            .await;
        self.finish(claim_id, request).await;
    }

    async fn fill_and_read(
        &mut self,
        claim_id: u64,
        request: &v2::ExportRequest,
        records: &[v2::Record],
        windows: Vec<v2::WindowSpec>,
    ) {
        for window in &windows {
            let bytes = self.content(records, window);
            self.source.filled(request.export_id, window, &bytes).await;
        }
        for _ in &windows {
            let ready = self.requester.window_ready(claim_id).await;
            let window = ready.window.unwrap();
            let got = self.requester.shm.as_ref().unwrap().read(&window);
            let expected = self.content(records, &window);
            assert_eq!(got, expected, "window {} bytes", window.index);
            for (page, sum) in ready.page_xxh3.iter().enumerate() {
                assert_eq!(
                    *sum,
                    pool::xxh3(
                        &expected[page * PAGE_BYTES as usize..(page + 1) * PAGE_BYTES as usize]
                    )
                );
            }
            self.requester.released(claim_id, window.index).await;
        }
    }

    async fn finish(&mut self, claim_id: u64, request: &v2::ExportRequest) {
        self.source
            .export_end(
                request.export_id,
                v2::ExportEndStatus::ExportComplete,
                v2::FenceOutcome::Drained,
            )
            .await;
        self.source.barrier().await;
        self.requester
            .copy_result(claim_id, v2::CopyOutcome::CopyOk)
            .await;
        let terminal = self.requester.claim_terminal(claim_id).await;
        assert_eq!(
            terminal.state,
            v2::ClaimState::Copied as i32,
            "{terminal:?}"
        );
        assert!(terminal.fence_required);
        self.requester
            .fence(claim_id, v2::FenceOutcome::Drained)
            .await;
    }
}

// ---------------------------------------------------------------- happy paths

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c1_a_gpu_chain_is_copied_through_retained_windows() {
    let mut w = world(v2::Tier::Gpu, &[1024, 2048]).await;
    w.claim(1, 2048, 0, true).await;
    let request = w.source.export_request().await;
    assert_eq!(request.claim_id, 1);
    assert_eq!(request.tier, v2::Tier::Gpu as i32);
    assert_eq!(
        request.reserved_bytes,
        32 * PAGE_BYTES + ROW_BYTES,
        "the whole export is reserved"
    );
    assert_eq!(request.session_incarnation, w.source.sid);
    // Nothing is granted before the pin is acknowledged.
    assert!(
        w.requester
            .never(Duration::from_millis(150), |m| matches!(
                m,
                M::ClaimGranted(_)
            )
            .then_some(()))
            .await
    );
    let records = w.pin(&request).await;
    let granted = w.requester.claim_granted(1).await;
    assert!(granted.retained);
    assert_eq!(granted.total_pages, 32);
    assert!(granted.row);
    w.copy_all(1, &request, &records).await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c1b_small_windows_cut_records_and_carry_the_row_last() {
    let mut w = world_with(v2::Tier::Gpu, &[1024, 2048], |host| {
        host.config.limits.max_window_bytes = 5 * PAGE_BYTES;
    })
    .await;
    w.claim(1, 2048, 0, true).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    w.requester.claim_granted(1).await;
    let fill = w.source.export_fill().await;
    // 16 + 16 pages in windows of five pages; the row rides with the last short window.
    assert!(
        fill.windows
            .iter()
            .all(|window| window.length <= 5 * PAGE_BYTES)
    );
    assert_eq!(fill.windows.iter().filter(|window| window.row).count(), 1);
    assert!(fill.windows.last().unwrap().row);
    let total: u64 = fill.windows.iter().map(|window| window.length).sum();
    assert_eq!(total, 32 * PAGE_BYTES + ROW_BYTES);
    // Windows tile the reserved block without overlap.
    let mut spans: Vec<(u64, u64)> = fill
        .windows
        .iter()
        .map(|w| (w.offset, w.offset + w.length))
        .collect();
    spans.sort_unstable();
    assert!(spans.windows(2).all(|pair| pair[0].1 <= pair[1].0));
    w.fill_and_read(1, &request, &records, fill.windows).await;
    w.finish(1, &request).await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a10_a_split_that_is_not_page_aligned_follows_flooring() {
    let mut w = world(v2::Tier::Dram, &[1024, 2048, 3072]).await;
    // The requester already holds 1500 tokens: pages from floor(1500/64) = 23.
    let candidate = w.candidate(1, 3072).await;
    let claim = w.requester.claim_for(1, &candidate, 1500, true);
    w.requester.claim(claim).await;
    let request = w.source.export_request().await;
    assert_eq!(request.begin_tokens, 1500);
    let records = w.pin(&request).await;
    // Node 1024 (pages 0..16) is gone, node 2048 (16..32) starts at page 23: first_page 7.
    assert_eq!(records.len(), 2);
    assert_eq!((records[0].first_page, records[0].pages), (7, 9));
    assert_eq!((records[1].first_page, records[1].pages), (0, 16));
    let granted = w.requester.claim_granted(1).await;
    assert_eq!(granted.total_pages, 25);
    let fill = w.source.export_fill().await;
    w.fill_and_read(1, &request, &records, fill.windows).await;
    w.finish(1, &request).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a11_a_row_only_tail_is_one_row_window() {
    let mut w = world(v2::Tier::Dram, &[1024, 2048]).await;
    let candidate = w.candidate(1, 2048).await;
    // The requester holds everything up to the endpoint's own page: only the row is missing.
    let claim = w.requester.claim_for(1, &candidate, 2048, true);
    w.requester.claim(claim).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    assert_eq!(records.len(), 1);
    assert_eq!(
        (records[0].first_page, records[0].pages, records[0].row),
        (16, 0, true)
    );
    let granted = w.requester.claim_granted(1).await;
    assert_eq!(granted.total_pages, 0);
    let fill = w.source.export_fill().await;
    assert_eq!(fill.windows.len(), 1);
    assert_eq!((fill.windows[0].page_count, fill.windows[0].row), (0, true));
    w.fill_and_read(1, &request, &records, fill.windows).await;
    w.finish(1, &request).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c2_a_host_tier_claim_larger_than_staging_streams_with_credits() {
    // 2100 pages need more than the 8 MiB staging region.
    let tokens = [2100 * PAGE_TOKENS];
    let mut w = world_with(v2::Tier::Dram, &tokens, |host| {
        host.config.staging = 8 * 1024 * 1024;
        host.config.limits.max_window_bytes = 1024 * 1024;
        host.config.limits.progress_timeout_ms = 5000;
    })
    .await;
    w.claim(1, tokens[0], 0, true).await;
    let request = w.source.export_request().await;
    assert_eq!(request.reserved_bytes, 0, "a host-tier source streams");
    let records = w.pin(&request).await;
    let granted = w.requester.claim_granted(1).await;
    assert!(!granted.retained);
    assert_eq!(granted.window_credits, 2);
    let mut seen = 0usize;
    let mut outstanding = 0usize;
    let mut fill = w.source.export_fill().await;
    loop {
        assert!(fill.windows.len() <= 2);
        for window in &fill.windows {
            outstanding += 1;
            assert!(
                outstanding <= 2,
                "never more than window_credits unreleased"
            );
            let bytes = w.content(&records, window);
            w.source.filled(request.export_id, window, &bytes).await;
        }
        for _ in &fill.windows {
            let ready = w.requester.window_ready(1).await;
            let window = ready.window.unwrap();
            assert_eq!(
                w.requester.shm.as_ref().unwrap().read(&window),
                w.content(&records, &window)
            );
            w.requester.released(1, window.index).await;
            outstanding -= 1;
            seen += 1;
        }
        if seen >= 33 {
            break;
        }
        fill = w.source.export_fill().await;
    }
    assert_eq!(
        seen, 33,
        "2100 pages and the row in windows of at most 64 pages"
    );
    w.finish(1, &request).await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_large_slow_transfer_completes_but_a_stalled_source_is_bounded() {
    let tokens = [16384 * PAGE_TOKENS];
    let mut w = world_with(v2::Tier::Nvme, &tokens, |host| {
        host.config.staging = 8 * 1024 * 1024;
        host.config.limits.max_window_bytes = 256 * 1024;
    })
    .await;
    let candidate = w.candidate(1, tokens[0]).await;
    let mut claim = w.requester.claim_for(1, &candidate, 0, true);
    claim.deadline_ms = 10_000;
    w.requester.claim(claim).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    w.requester.claim_granted(1).await;
    let started = std::time::Instant::now();
    let mut seen = 0;
    let mut bytes = 0;
    while bytes < 16384 * PAGE_BYTES + ROW_BYTES {
        let fill = w.source.export_fill().await;
        let length = fill.windows.iter().map(|window| window.length).sum::<u64>();
        tokio::time::sleep(Duration::from_secs_f64(length as f64 / (5 << 20) as f64)).await;
        bytes += length;
        seen += fill.windows.len();
        w.fill_and_read(1, &request, &records, fill.windows).await;
    }
    assert!(seen > 250);
    assert!(started.elapsed() > Duration::from_secs(12));
    w.finish(1, &request).await;
    w.staging_settles_at(0).await;

    w.claim(2, tokens[0], 0, true).await;
    let request = w.source.export_request().await;
    w.pin(&request).await;
    w.requester.claim_granted(2).await;
    let stalled = std::time::Instant::now();
    let terminal = w.requester.claim_terminal(2).await;
    assert_eq!(terminal.reason, v2::ClaimReason::NoProgress as i32);
    assert!(stalled.elapsed() < Duration::from_secs(2));
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, request.export_id);
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportFailed,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.requester.fence(2, v2::FenceOutcome::Drained).await;
    w.staging_settles_at(0).await;
}

// ---------------------------------------------------------------- refusals

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c5_source_refusals_fail_the_claim_and_free_the_slot() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let reasons = [
        v2::ClaimReason::NotFound,
        v2::ClaimReason::ChainGap,
        v2::ClaimReason::NoHeadroom,
        v2::ClaimReason::SourceBusy,
    ];
    for (index, reason) in reasons.iter().enumerate() {
        let id = index as u64 + 1;
        w.claim(id, 1024, 0, true).await;
        let request = w.source.export_request().await;
        w.source.pin_refused(&request, *reason).await;
        let terminal = w.requester.claim_terminal(id).await;
        assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
        assert_eq!(terminal.reason, *reason as i32);
        assert!(!terminal.fence_required, "nothing was granted");
    }
    // Five refusals in a row never exhausted the export slots.
    w.claim(10, 1024, 0, true).await;
    w.source.export_request().await;
    w.staging_settles_at(w.staging_used()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c6_no_pin_answer_times_out_and_a_late_pin_is_released() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    w.claim(1, 1024, 0, true).await;
    let request = w.source.export_request().await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::SourceTimeout as i32);
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, request.export_id);
    // The pin completes after the deadline: it is released, never granted.
    w.pin(&request).await;
    let again = w.source.export_cancel().await;
    assert_eq!(again.export_id, request.export_id);
    assert!(
        w.requester
            .never(Duration::from_millis(150), |m| matches!(
                m,
                M::ClaimGranted(_)
            )
            .then_some(()))
            .await
    );
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c7_and_c8_a_release_that_crosses_the_pin_ack_never_grants() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    // C7: released before the source answers; the source drops the request.
    w.claim(1, 1024, 0, true).await;
    let first = w.source.export_request().await;
    w.requester.send(pool_release(1)).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::Cancelled as i32);
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, first.export_id);
    w.source
        .pin_refused(&first, v2::ClaimReason::Cancelled)
        .await;
    // C8: the pin acknowledgment was already on its way.
    w.claim(2, 1024, 0, true).await;
    let second = w.source.export_request().await;
    w.requester.send(pool_release(2)).await;
    w.requester.claim_terminal(2).await;
    w.pin(&second).await;
    assert!(
        w.requester
            .never(Duration::from_millis(150), |m| matches!(
                m,
                M::ClaimGranted(_)
            )
            .then_some(()))
            .await,
        "a terminal claim is never granted"
    );
    // The manager asks the source to release the late pin; the source obliges.
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, second.export_id);
    w.source
        .export_end(
            second.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.staging_settles_at(0).await;
}

fn pool_release(claim_id: u64) -> v2::client_message::M {
    v2::client_message::M::ClaimRelease(v2::ClaimRelease { claim_id })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c14_to_c19_invalid_unknown_and_replayed_claims() {
    let mut w = world(v2::Tier::Dram, &[1024, 2048]).await;
    // C14: releasing an id the manager never saw.
    w.requester.send(pool_release(77)).await;
    let terminal = w.requester.claim_terminal(77).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::UnknownClaim as i32);
    // C19: begin beyond the endpoint.
    let candidate = w.candidate(1, 2048).await;
    let mut claim = w.requester.claim_for(10, &candidate, 4096, true);
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(10).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
    // A non-increasing id is a replay.
    claim.claim_id = 5;
    claim.begin_tokens = 0;
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(5).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
    // An unspecified mode, and a mode the session cannot use.
    claim.claim_id = 11;
    claim.mode = v2::CopyMode::Unspecified as i32;
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(11).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
    claim.claim_id = 12;
    claim.mode = v2::CopyMode::DataChannel as i32;
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(12).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
    // C18: a stale source session, an unknown instance, a missing entry.
    claim.claim_id = 13;
    claim.mode = v2::CopyMode::SharedWindows as i32;
    claim.source.as_mut().unwrap().session_incarnation += 100;
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(13).await.reason,
        v2::ClaimReason::StaleSource as i32
    );
    claim.claim_id = 14;
    claim.source.as_mut().unwrap().session_incarnation -= 100;
    claim.source.as_mut().unwrap().instance_id = vec![9; 16];
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(14).await.reason,
        v2::ClaimReason::UnknownSource as i32
    );
    claim.claim_id = 15;
    claim.source.as_mut().unwrap().instance_id =
        candidate.source.as_ref().unwrap().instance_id.clone();
    claim.key = Some(pool::pkey(4096));
    w.requester.claim(claim.clone()).await;
    assert_eq!(
        w.requester.claim_terminal(15).await.reason,
        v2::ClaimReason::UnknownSource as i32
    );
    // A reused id cannot start another lifecycle, even after refusal.
    w.requester.claim(claim).await;
    assert_eq!(
        w.requester.claim_terminal(15).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn x1_claims_in_flight_and_export_slots_are_limited() {
    let mut w = world(v2::Tier::Gpu, &[1024, 2048, 3072]).await;
    let candidate = w.candidate(900, 3072).await;
    for id in 1..=2 {
        w.claim(id, 1024 * id, 0, true).await;
    }
    w.requester
        .claim(w.requester.claim_for(3, &candidate, 0, true))
        .await;
    let terminal = w.requester.claim_terminal(3).await;
    assert_eq!(terminal.reason, v2::ClaimReason::ClaimLimit as i32);
    assert!(!terminal.fence_required);
    // A second requester is limited by the source's export slots instead.
    let mut other = w.pool.open(&Ident::new("req2")).await;
    other.snapshot(vec![]).await;
    let reply = other.lookup(1, 3072, 0).await;
    assert!(
        !reply.candidates.is_empty(),
        "a busy source is still offered"
    );
    // A candidate obtained before the slots filled still loses the claim race safely.
    other.claim(other.claim_for(1, &candidate, 0, true)).await;
    let terminal = other.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::SourceLimit as i32);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lookups_keep_busy_sources_in_order_with_other_peers_and_the_manager() {
    let mut w = world_with(v2::Tier::Gpu, &[1024], |host| {
        host.config.limits.pin_deadline_ms = 10_000;
    })
    .await;
    let mut writer = w.pool.open(&Ident::new("writer")).await;
    writer.snapshot(vec![]).await;
    assert!(matches!(
        writer
            .put_record(1, pool::key(1024), None, pool::Damage::default())
            .await,
        pool::PutResult::Stored(_)
    ));
    writer.manager_candidate(1, 1024).await;
    let mut peer_ident = Ident::new("peer");
    peer_ident.export_slots = 1;
    let mut peer = w.pool.open(&peer_ident).await;
    peer.snapshot(chain(&[1024], v2::Tier::Dram)).await;
    let mut reader = w.pool.open(&Ident::new("reader")).await;
    reader.snapshot(vec![]).await;
    for id in 1..=2 {
        w.claim(id, 1024, 0, true).await;
    }
    w.requester.barrier().await;
    // Include the manager record's bucket and the peers' bucket in the same history.
    let lookup = |id| v2::LookupRequest {
        lookup_id: id,
        boundaries: vec![
            v2::PrefixKey {
                consumed_tokens: 0,
                digest: vec![4; 16],
            },
            pool::pkey(1024),
        ],
        before_tokens: 1025,
        resident_tokens: 0,
    };
    let reply = reader.lookup_request(lookup(1)).await;
    assert_eq!(reply.candidates.len(), 3);
    assert_eq!(reply.candidates[0].tier, v2::Tier::Gpu as i32);
    assert_eq!(
        reply.candidates[1].source.as_ref().unwrap().instance_id,
        peer_ident.instance_id
    );
    assert!(
        reply.candidates[2]
            .source
            .as_ref()
            .unwrap()
            .instance_id
            .is_empty()
    );
    let candidate = &reply.candidates[1];
    reader.claim(reader.claim_for(1, candidate, 0, true)).await;
    let export = peer.export_request().await;

    let reply = reader.lookup_request(lookup(2)).await;
    assert_eq!(reply.candidates.len(), 3);
    let candidate = &reply.candidates[2];
    assert!(candidate.source.as_ref().unwrap().instance_id.is_empty());
    reader.claim(reader.claim_for(2, candidate, 0, true)).await;
    let grant = reader.claim_granted(2).await;
    assert_eq!(grant.total_pages, 1024 / PAGE_TOKENS);
    assert!(grant.row);
    peer.pin_refused(&export, v2::ClaimReason::SourceBusy).await;
    assert_eq!(
        reader.claim_terminal(1).await.reason,
        v2::ClaimReason::SourceBusy as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c20_a_gpu_export_that_cannot_be_reserved_fails_before_any_request() {
    let tokens = [2100 * PAGE_TOKENS];
    let mut w = world_with(v2::Tier::Gpu, &tokens, |host| {
        host.config.staging = 8 * 1024 * 1024;
    })
    .await;
    w.claim(1, tokens[0], 0, true).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::NoWindows as i32);
    assert!(
        w.source
            .never(Duration::from_millis(150), |m| matches!(
                m,
                M::ExportRequest(_)
            )
            .then_some(()))
            .await,
        "nothing was asked of the source"
    );
}

// ---------------------------------------------------------------- terminal races

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c13_duplicates_change_nothing() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    w.claim(1, 1024, 0, true).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    // A duplicate PinAck is ignored.
    w.pin(&request).await;
    let granted = w.requester.claim_granted(1).await;
    assert!(
        w.requester
            .never(Duration::from_millis(100), |m| matches!(
                m,
                M::ClaimGranted(_)
            )
            .then_some(()))
            .await,
        "one grant only"
    );
    let fill = w.source.export_fill().await;
    w.fill_and_read(1, &request, &records, fill.windows.clone())
        .await;
    // Duplicate release and duplicate fill are harmless.
    w.requester.released(1, 0).await;
    let bytes = w.content(&records, &fill.windows[0]);
    w.source
        .filled(request.export_id, &fill.windows[0], &bytes)
        .await;
    w.finish(1, &request).await;
    for _ in 0..2 {
        w.requester.copy_result(1, v2::CopyOutcome::CopyOk).await;
        w.requester.fence(1, v2::FenceOutcome::Drained).await;
        w.source
            .export_end(
                request.export_id,
                v2::ExportEndStatus::ExportComplete,
                v2::FenceOutcome::Drained,
            )
            .await;
    }
    assert_eq!(granted.records.len(), 1);
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c15_copy_ok_with_an_unreleased_window_is_a_copy_error() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    w.claim(1, 1024, 0, true).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    w.requester.claim_granted(1).await;
    let fill = w.source.export_fill().await;
    let bytes = w.content(&records, &fill.windows[0]);
    w.source
        .filled(request.export_id, &fill.windows[0], &bytes)
        .await;
    w.requester.window_ready(1).await;
    // The requester says OK without having released the window.
    w.requester.copy_result(1, v2::CopyOutcome::CopyOk).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::CopyError as i32);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c16_deadline_and_no_progress_expire_a_claim() {
    let mut w = world(v2::Tier::Gpu, &[1024, 2048]).await;
    // No progress: granted, but no window is ever filled.
    w.claim(1, 1024, 0, true).await;
    let request = w.source.export_request().await;
    w.pin(&request).await;
    w.requester.claim_granted(1).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::NoProgress as i32);
    assert!(terminal.fence_required);
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, request.export_id);
    // The whole-claim deadline, clamped by the manager.
    let candidate = w.candidate(30, 2048).await;
    let mut claim = w.requester.claim_for(2, &candidate, 0, true);
    claim.deadline_ms = 100;
    w.requester.claim(claim).await;
    let _second = w.source.export_request().await;
    let terminal = w.requester.claim_terminal(2).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::Deadline as i32);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c17_a_failed_fill_fails_the_claim() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    w.claim(1, 1024, 0, true).await;
    let request = w.source.export_request().await;
    w.pin(&request).await;
    w.requester.claim_granted(1).await;
    let fill = w.source.export_fill().await;
    w.source
        .send(v2::client_message::M::WindowFilled(v2::WindowFilled {
            export_id: request.export_id,
            index: fill.windows[0].index,
            status: v2::FillStatus::FillFailed as i32,
            page_xxh3: vec![],
            row_xxh3: 0,
            has_row: false,
            ..Default::default()
        }))
        .await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::CopyError as i32);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a6_a_premature_or_partial_copy_ok_is_refused() {
    let mut w = world_with(v2::Tier::Dram, &[2100 * PAGE_TOKENS], |host| {
        host.config.staging = 8 * 1024 * 1024;
        host.config.limits.max_window_bytes = 1024 * 1024;
        host.config.limits.progress_timeout_ms = 5000;
    })
    .await;
    w.claim(1, 2100 * PAGE_TOKENS, 0, true).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    w.requester.claim_granted(1).await;
    let fill = w.source.export_fill().await;
    // Only the windows issued so far are filled and released, later ones are not issued yet.
    w.fill_and_read(1, &request, &records, fill.windows).await;
    w.requester.copy_result(1, v2::CopyOutcome::CopyOk).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::CopyError as i32);
}

// ---------------------------------------------------------------- fencing

/// A claim whose source filled `filled` windows and left the rest in its hands.
struct Partial {
    request: v2::ExportRequest,
    records: Vec<v2::Record>,
    windows: Vec<v2::WindowSpec>,
}

async fn granted(w: &mut World, id: u64, upto: u64) -> Partial {
    w.claim(id, upto, 0, true).await;
    let request = w.source.export_request().await;
    let records = w.pin(&request).await;
    w.requester.claim_granted(id).await;
    let fill = w.source.export_fill().await;
    Partial {
        request,
        records,
        windows: fill.windows,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f1_the_window_order_is_enforced() {
    let mut w = world_with(v2::Tier::Gpu, &[1024], |host| {
        host.config.limits.max_window_bytes = 4 * PAGE_BYTES;
    })
    .await;
    let partial = granted(&mut w, 1, 1024).await;
    assert!(partial.windows.len() > 1);
    // Released before it was filled: a protocol error closes the session.
    w.requester.released(1, 0).await;
    let closing = w.requester.closed().await.expect("closing");
    assert_eq!(
        closing.reason,
        v2::ClosingReason::ClosingProtocolViolation as i32
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f2_a_window_the_source_may_still_write_waits_for_its_end() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let partial = granted(&mut w, 1, 1024).await;
    let held = w.staging_now().await;
    assert!(held >= partial.windows[0].length);
    w.requester.send(pool_release(1)).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::Cancelled as i32);
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.requester.barrier().await;
    assert_eq!(
        w.staging_used(),
        held,
        "the source has not said it stopped writing"
    );
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, partial.request.export_id);
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f3_a_window_the_requester_may_still_read_waits_for_its_fence() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let partial = granted(&mut w, 1, 1024).await;
    let bytes = w.content(&partial.records, &partial.windows[0]);
    w.source
        .filled(partial.request.export_id, &partial.windows[0], &bytes)
        .await;
    w.requester.window_ready(1).await;
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    let held = w.staging_now().await;
    w.requester.send(pool_release(1)).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert!(terminal.fence_required);
    w.requester.barrier().await;
    assert_eq!(w.staging_used(), held, "the requester may still be reading");
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn f4_a_quarantined_party_keeps_its_windows_out_of_service() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let partial = granted(&mut w, 1, 1024).await;
    let bytes = w.content(&partial.records, &partial.windows[0]);
    w.source
        .filled(partial.request.export_id, &partial.windows[0], &bytes)
        .await;
    w.requester.window_ready(1).await;
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.requester.send(pool_release(1)).await;
    w.requester.claim_terminal(1).await;
    assert_eq!(w.host.gauge("pool_quarantined_bytes"), 0);
    w.requester.fence(1, v2::FenceOutcome::Quarantined).await;
    w.requester.barrier().await;
    let held = w.staging_now().await;
    assert!(held > 0);
    w.host
        .wait(|host| host.gauge("pool_quarantined_bytes") == partial.windows[0].length);
    // The process is still alive, so the windows stay out of service.
    w.requester.barrier().await;
    assert_eq!(w.staging_used(), held);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a3_a_quarantined_export_keeps_its_slot() {
    let mut w = world(v2::Tier::Gpu, &[1024, 2048, 3072]).await;
    let partial = granted(&mut w, 1, 1024).await;
    let bytes = w.content(&partial.records, &partial.windows[0]);
    w.source
        .filled(partial.request.export_id, &partial.windows[0], &bytes)
        .await;
    w.requester.window_ready(1).await;
    w.requester.released(1, 0).await;
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Quarantined,
        )
        .await;
    w.requester.copy_result(1, v2::CopyOutcome::CopyOk).await;
    assert_eq!(
        w.requester.claim_terminal(1).await.state,
        v2::ClaimState::Copied as i32
    );
    // One of the source's two slots is still taken.
    let candidate = w.candidate(903, 3072).await;
    let second = granted(&mut w, 2, 2048).await;
    w.requester
        .claim(w.requester.claim_for(3, &candidate, 0, true))
        .await;
    let terminal = w.requester.claim_terminal(3).await;
    assert_eq!(terminal.reason, v2::ClaimReason::SourceLimit as i32);
    drop(second);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a1_and_a2_goodbye_clears_only_the_senders_obligations() {
    // A1: the requester says goodbye while the source may still be writing.
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let partial = granted(&mut w, 1, 1024).await;
    let held = w.staging_now().await;
    w.requester.goodbye(true).await;
    let mut requester =
        std::mem::replace(&mut w.requester, w.pool.open(&Ident::new("req-b")).await);
    drop(requester.tx.clone());
    requester.ended = true;
    drop(requester);
    w.requester.barrier().await;
    assert_eq!(w.staging_used(), held, "the source's write is still open");
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.staging_settles_at(0).await;

    // A2: the source says goodbye while the requester may still be reading.
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let partial = granted(&mut w, 1, 1024).await;
    let bytes = w.content(&partial.records, &partial.windows[0]);
    w.source
        .filled(partial.request.export_id, &partial.windows[0], &bytes)
        .await;
    w.requester.window_ready(1).await;
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    let held = w.staging_now().await;
    w.source.goodbye(true).await;
    let source = std::mem::replace(&mut w.source, w.pool.open(&Ident::new("src-b")).await);
    drop(source);
    w.requester.barrier().await;
    assert_eq!(w.staging_used(), held, "the requester may still be reading");
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r11_a_source_blip_quarantines_until_the_export_end_arrives_on_a_new_session() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let source_ident = w.source_ident.clone();
    let partial = granted(&mut w, 1, 1024).await;
    let held = w.staging_now().await;
    // The stream drops, the process lives on.
    let old = std::mem::replace(&mut w.source, w.pool.open(&source_ident).await);
    drop(old);
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::SourceLost as i32);
    w.source.barrier().await;
    assert_eq!(
        w.staging_used(),
        held,
        "quarantined until the source reports"
    );
    // The new session of the same process reports for the old export.
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.requester.fence(1, v2::FenceOutcome::Drained).await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn s4_and_s7_a_new_process_attests_nothing_for_its_predecessor() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    let old_ident = w.requester_ident.clone();
    let partial = granted(&mut w, 1, 1024).await;
    let bytes = w.content(&partial.records, &partial.windows[0]);
    w.source
        .filled(partial.request.export_id, &partial.windows[0], &bytes)
        .await;
    w.requester.window_ready(1).await;
    w.source
        .export_end(
            partial.request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    let held = w.staging_now().await;
    // Another process claims the same instance id: the older one is fenced.
    let new_ident = old_ident.restarted();
    let mut fresh = w.pool.open(&new_ident).await;
    let closing = w.requester.closed().await.expect("closing");
    assert_eq!(closing.reason, v2::ClosingReason::ClosingSuperseded as i32);
    fresh.snapshot(vec![]).await;
    // Its acknowledgment of a claim id it never made resolves nothing.
    fresh.fence(1, v2::FenceOutcome::Drained).await;
    fresh.barrier().await;
    assert_eq!(
        w.staging_used(),
        held,
        "nothing attested for the old process"
    );
    // S7: its own counters start afresh and are accepted.
    let candidate = {
        let reply = fresh.lookup(1, 1024, 0).await;
        reply.candidates[0].clone()
    };
    let claim = fresh.claim_for(1, &candidate, 0, true);
    fresh.claim(claim).await;
    let request = w.source.export_request().await;
    assert_eq!(request.claim_id, 1);
    assert_ne!(request.export_id, partial.request.export_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn s5_and_s6_the_predecessors_entries_are_hidden_until_the_new_snapshot() {
    let mut w = world(v2::Tier::Dram, &[1024, 2048]).await;
    let src_ident = w.source_ident.clone();
    let mut third = w.pool.open(&Ident::new("third")).await;
    third.snapshot(vec![]).await;
    assert_eq!(third.lookup(1, 2048, 0).await.candidates.len(), 2);
    let candidate = w.candidate(2, 2048).await;
    // A newer session of the source opens; the old entries are not offered at once.
    let mut newer = w.pool.open(&src_ident.restarted()).await;
    let _ = w.source.closed().await;
    assert!(third.lookup(3, 2048, 0).await.candidates.is_empty());
    // A claim naming the predecessor's session is stale.
    w.requester
        .claim(w.requester.claim_for(1, &candidate, 0, true))
        .await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::StaleSource as i32);
    // S6: the new session never finishes its snapshot and ends: nothing stale is offered.
    drop(newer);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(third.lookup(4, 2048, 0).await.candidates.is_empty());
    // A later successor installs and its set alone is visible.
    newer = w.pool.open(&src_ident.restarted()).await;
    newer.snapshot(chain(&[4096], v2::Tier::Dram)).await;
    let reply = third.lookup(5, 4096, 0).await;
    assert_eq!(reply.candidates.len(), 1);
}

// ---------------------------------------------------------------- real process death

/// The other end of a claim, run in a separate process so that it can really die.
#[test]
fn child_peer() {
    let Ok(role) = std::env::var("SELENE_CHILD_ROLE") else {
        return;
    };
    let socket = std::path::PathBuf::from(std::env::var("SELENE_CHILD_SOCKET").unwrap());
    let flag = std::path::PathBuf::from(std::env::var("SELENE_CHILD_FLAG").unwrap());
    let tokens: Vec<u64> = vec![1024];
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let mut pool = Pool::connect(&socket).await;
            match role.as_str() {
                "requester" => {
                    let mut session = pool.open(&Ident::new("req")).await;
                    session.snapshot(vec![]).await;
                    let reply = session.lookup(1, 1024, 0).await;
                    let candidate = reply.candidates[0].clone();
                    session
                        .claim(session.claim_for(1, &candidate, 0, true))
                        .await;
                    session.claim_granted(1).await;
                    session.window_ready(1).await;
                    std::fs::write(&flag, b"ready").unwrap();
                }
                "reconnecter" => {
                    // Opens a session and does nothing else; after the manager restarted and
                    // `go` exists it opens another, as a client reconnecting does.
                    let mut session = pool.open(&Ident::new("idle")).await;
                    session.snapshot(vec![]).await;
                    std::fs::write(&flag, b"ready").unwrap();
                    let go = flag.with_extension("go");
                    while !go.exists() {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    drop(session);
                    drop(pool);
                    let mut pool = Pool::connect(&socket).await;
                    // Only the session: no snapshot, no claim, no put.
                    let _session = pool.open(&Ident::new("idle")).await;
                    std::fs::write(flag.with_extension("back"), b"back").unwrap();
                    tokio::time::sleep(Duration::from_secs(30)).await;
                }
                "source" => {
                    let mut session = pool.open(&Ident::new("src")).await;
                    session.snapshot(vec![]).await;
                    session
                        .notices(
                            chain(&tokens, v2::Tier::Gpu)
                                .into_iter()
                                .map(saved)
                                .collect(),
                        )
                        .await;
                    let request = session.export_request().await;
                    let records = records_for(&tokens, 0, true);
                    session.pin_ok(&request, records, true).await;
                    session.export_fill().await;
                    std::fs::write(&flag, b"ready").unwrap();
                }
                other => panic!("unknown role {other}"),
            }
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
}

fn spawn_child(
    role: &str,
    socket: &std::path::Path,
    flag: &std::path::Path,
) -> std::process::Child {
    std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_peer", "--nocapture"])
        .env("SELENE_CHILD_ROLE", role)
        .env("SELENE_CHILD_SOCKET", socket)
        .env("SELENE_CHILD_FLAG", flag)
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap()
}

fn flag_path(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("selene-flag-{}-{name}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn wait_for_file(path: &std::path::Path) {
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while !path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "the child never signalled"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_manager_releases_its_regions_when_the_previous_runs_processes_reopen_a_session()
 {
    let mut host = host();
    host.run();
    let flag = flag_path("reconnecter");
    struct Kill(std::process::Child);
    impl Drop for Kill {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let child = Kill(spawn_child("reconnecter", &host.config.socket, &flag));
    wait_for_file(&flag);
    host.stop();
    // A crash leaves the holders file as the live manager last wrote it; a graceful stop has
    // emptied it, so write what a crash would have left: the child, and nobody else.
    let stat = std::fs::read_to_string(format!("/proc/{}/stat", child.0.id())).unwrap();
    let start = stat[stat.rfind(')').unwrap() + 1..]
        .split_whitespace()
        .nth(19)
        .unwrap()
        .to_owned();
    std::fs::write(
        host.shm_directory().join("holders"),
        format!("{} {start}\n", child.0.id()),
    )
    .unwrap();
    host.run();
    assert_eq!(host.gauge("pool_regions_held"), 1, "the child is a holder");
    // Nothing the child does but reopen its session ends the hold: it asks for no window.
    std::fs::write(flag.with_extension("go"), b"go").unwrap();
    wait_for_file(&flag.with_extension("back"));
    host.wait(|host| host.gauge("pool_regions_held") == 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r7_and_r8_a_killed_or_hung_requester_leaves_nothing_behind() {
    let mut host = host();
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let mut source = pool.open(&Ident::new("src")).await;
    source.snapshot(vec![]).await;
    let tokens = [1024u64];
    source
        .notices(
            chain(&tokens, v2::Tier::Gpu)
                .into_iter()
                .map(saved)
                .collect(),
        )
        .await;
    let flag = flag_path("r7");
    let mut child = spawn_child("requester", &host.config.socket, &flag);
    let request = source.export_request().await;
    let records = records_for(&tokens, 0, true);
    source.pin_ok(&request, records.clone(), true).await;
    let fill = source.export_fill().await;
    let content: Vec<u8> = (0..16)
        .flat_map(|page| page_bytes(SEED, page))
        .chain(row_bytes(SEED, 1024))
        .collect();
    source
        .filled(request.export_id, &fill.windows[0], &content)
        .await;
    wait_for_file(&flag);
    source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportComplete,
            v2::FenceOutcome::Drained,
        )
        .await;
    source.barrier().await;
    host.wait(|host| host.gauge("staging_allocated_bytes") > 0);
    let held = host.gauge("staging_allocated_bytes");
    assert!(held > 0);

    // R8: the process hangs. The claim expires, but its window stays out of service.
    unsafe { libc::kill(child.id() as i32, libc::SIGSTOP) };
    let cancel = source.export_cancel().await;
    assert_eq!(cancel.export_id, request.export_id);
    source.barrier().await;
    assert_eq!(
        host.gauge("staging_allocated_bytes"),
        held,
        "a hung reader keeps its window"
    );

    // R7: it dies. Its window comes back without anyone's acknowledgment.
    child.kill().unwrap();
    child.wait().unwrap();
    for _ in 0..100 {
        if host.gauge("staging_allocated_bytes") == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(host.gauge("staging_allocated_bytes"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn r10_a_killed_source_fails_its_claims_and_frees_its_windows() {
    let mut host = host();
    host.run();
    let mut pool = Pool::connect(&host.config.socket).await;
    let flag = flag_path("r10");
    let mut child = spawn_child("source", &host.config.socket, &flag);
    let mut requester = pool.open(&Ident::new("req")).await;
    requester.snapshot(vec![]).await;
    let mut candidate = None;
    for id in 1..400 {
        let reply = requester.lookup(id, 1024, 0).await;
        if let Some(found) = reply.candidates.first() {
            candidate = Some(found.clone());
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let candidate = candidate.expect("the child source announced its chain");
    requester
        .claim(requester.claim_for(1, &candidate, 0, true))
        .await;
    requester.claim_granted(1).await;
    wait_for_file(&flag);
    host.wait(|host| host.gauge("staging_allocated_bytes") > 0);
    assert!(host.gauge("staging_allocated_bytes") > 0);
    child.kill().unwrap();
    child.wait().unwrap();
    let terminal = requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::SourceLost as i32);
    for _ in 0..100 {
        if host.gauge("staging_allocated_bytes") == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(
        host.gauge("staging_allocated_bytes"),
        0,
        "its write windows are reclaimed after exit"
    );
    // Its entries are gone too.
    let reply = requester.lookup(500, 1024, 0).await;
    assert!(reply.candidates.is_empty());
}

// ---------------------------------------------------------------- deadlines and sizes

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn x5_a_deadline_is_clamped_to_the_managers_limit() {
    let mut w = world_with(v2::Tier::Gpu, &[1024], |host| {
        host.config.limits.max_claim_deadline_ms = 400;
        host.config.limits.pin_deadline_ms = 10_000;
    })
    .await;
    let candidate = w.candidate(1, 1024).await;
    let mut claim = w.requester.claim_for(1, &candidate, 0, true);
    claim.deadline_ms = 600_000;
    let began = std::time::Instant::now();
    w.requester.claim(claim).await;
    w.source.export_request().await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::Deadline as i32);
    assert!(
        began.elapsed() < Duration::from_secs(5),
        "{:?}",
        began.elapsed()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a21_a_pin_larger_than_the_reservation_fails_no_windows() {
    let mut w = world(v2::Tier::Gpu, &[1024]).await;
    w.claim(1, 1024, 0, true).await;
    let request = w.source.export_request().await;
    // The chain grew: the source reports more pages than were reserved.
    let records = vec![v2::Record {
        key: Some(pool::pkey(1024)),
        first_page: 0,
        pages: 40,
        row: true,
    }];
    w.source.pin_ok(&request, records, true).await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.reason, v2::ClaimReason::NoWindows as i32);
    let cancel = w.source.export_cancel().await;
    assert_eq!(cancel.export_id, request.export_id);
    w.source
        .export_end(
            request.export_id,
            v2::ExportEndStatus::ExportReleased,
            v2::FenceOutcome::Drained,
        )
        .await;
    w.staging_settles_at(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_source_lost_between_lookup_and_claim_is_a_clean_miss() {
    let mut w = world(v2::Tier::Dram, &[1024]).await;
    let candidate = w.candidate(1, 1024).await;
    // A resync after loss of the source's records invalidates the lookup hint.
    w.source.snapshot(vec![]).await;
    w.requester
        .claim(w.requester.claim_for(1, &candidate, 0, true))
        .await;
    let terminal = w.requester.claim_terminal(1).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    assert_eq!(terminal.reason, v2::ClaimReason::UnknownSource as i32);
    assert!(!terminal.fence_required);
    assert!(w.requester.lookup(2, 1024, 0).await.candidates.is_empty());
    assert_eq!(w.staging_used(), 0);
    assert!(
        w.source
            .never(Duration::from_millis(50), |message| {
                matches!(message, M::ExportRequest(_)).then_some(())
            })
            .await
    );
}
