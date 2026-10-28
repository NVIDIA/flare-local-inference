//! The L0 engine tests' vocabulary (store, put, fetch, get) over the pool protocol.
//!
//! A [`Client`] is one `selene.v2` session in the scope of a record descriptor. A put
//! is `PutBegin` to `Stored`; a fetch is a lookup of the manager's own entry followed
//! by a claim on it. A restore carries pages and the row only, so what a put returns
//! and a fetch yields is a record's payload: its pages, then its row. The header and
//! checksum table are the manager's to validate on commit and are never served.

use super::{
    content_for,
    pool::{self, Ident, Pool, Session},
};
use selene_manager::{
    Key,
    pool::key_to,
    proto::v2::{self, client_message as c, server_message::M},
    record::{self, Descriptor},
    store::Config,
};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static CLIENTS: AtomicU64 = AtomicU64::new(0);

/// How long a put or claim may wait for the manager's next message. The tests that
/// pause the manager's disk work hold a put open for several seconds.
const WAIT: Duration = Duration::from_secs(30);
/// The bucket digest `content_for` writes into every record.
pub const BUCKET: [u8; 16] = [4; 16];

pub fn scope_of(descriptor: &Descriptor) -> v2::Scope {
    v2::Scope {
        model_identity: descriptor.identity.to_vec(),
        layout_fingerprint: descriptor.fingerprint.to_vec(),
        geometry: Some(v2::Geometry {
            page_tokens: descriptor.page_tokens,
            ranks: descriptor
                .ranks
                .iter()
                .map(|rank| v2::RankPlanes {
                    plane_bytes: rank.planes.clone(),
                    row_bytes: rank.row_bytes,
                })
                .collect(),
            page_record_bytes: descriptor.page_bytes(),
            row_record_bytes: descriptor.row_bytes(),
        }),
    }
}

fn key_of(key: &v2::PrefixKey) -> Key {
    Key {
        consumed_tokens: key.consumed_tokens,
        digest: key.digest.as_slice().try_into().unwrap(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelAt {
    None,
    Accept,
    Grant,
    Filled,
    Commit,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PutOptions {
    pub cancel: Option<CancelAt>,
    pub cancel_after_fills: Option<u32>,
    pub abort_after_commit: bool,
    /// Commit a header-only fragment of the metadata after the first window.
    pub early_commit: bool,
    /// Never acknowledge the first window, then commit.
    pub skip_filled: bool,
    pub corrupt_page: Option<usize>,
    pub corrupt_row: bool,
    pub wrong_key: bool,
    pub keep_windows: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PutOutcome {
    /// The record's payload.
    Stored(Vec<u8>),
    /// `PutRefused.reason`, numbered as `v2::Reason`.
    Refused(u8),
    Cancelled,
    /// The writer went away right after committing.
    Abandoned,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GetOptions {
    /// `Grant`: release the claim as soon as the first window is ready.
    pub cancel: Option<CancelAt>,
    /// With a cancel: send no fence, as a process that is still holding its windows.
    pub leave_leased: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum GetOutcome {
    /// One buffer per record the claim covered: its pages from the claim's first page,
    /// then, for the last record, the row.
    Done(Vec<Vec<u8>>),
    Refused(v2::ClaimReason),
    Cancelled,
}

#[derive(Debug)]
pub enum PutStart {
    Accepted(v2::PutAccept),
    Refused(u8),
}

pub struct GetSpec {
    pub keys: Vec<Key>,
    pub skip: u32,
    pub pages: u32,
    pub options: GetOptions,
}

impl GetSpec {
    pub fn new(keys: &[Key], skip: u32, pages: u32) -> Self {
        Self {
            keys: keys.to_vec(),
            skip,
            pages,
            options: GetOptions::default(),
        }
    }
}

/// A claim whose windows the client is holding.
pub struct Held {
    pub id: u64,
    pub granted: v2::ClaimGranted,
    pub windows: Vec<v2::WindowReady>,
}

pub struct Client {
    pub pool: Pool,
    pub session: Session,
    pub descriptor: Descriptor,
    next_put: u64,
    next_claim: u64,
    next_lookup: u64,
}

struct Run {
    id: u64,
    pages: u32,
    options: GetOptions,
    layout: Vec<(u32, u32)>,
    records: Vec<Vec<u8>>,
    expected: u64,
    received: u64,
    cancelled: bool,
    abandoned: bool,
    damaged: bool,
    finished: bool,
    outcome: Option<GetOutcome>,
}

impl Run {
    fn new(id: u64, spec: &GetSpec) -> Self {
        Self {
            id,
            pages: spec.pages,
            options: spec.options,
            layout: Vec::new(),
            records: Vec::new(),
            expected: 0,
            received: 0,
            cancelled: false,
            abandoned: false,
            damaged: false,
            finished: false,
            outcome: None,
        }
    }

    fn refused(reason: v2::ClaimReason) -> Self {
        let mut run = Self::new(0, &GetSpec::new(&[], 0, 0));
        run.outcome = Some(GetOutcome::Refused(reason));
        run
    }
}

enum Event {
    Granted(v2::ClaimGranted),
    Ready(v2::WindowReady),
    Terminal(v2::ClaimTerminal),
}

/// The bytes a claim moves: pages then, on the last record, the row.
pub fn expected_bytes(granted: &v2::ClaimGranted, page_bytes: u64, row_bytes: u64) -> u64 {
    granted
        .records
        .iter()
        .map(|record| u64::from(record.pages) * page_bytes + if record.row { row_bytes } else { 0 })
        .sum()
}

impl Client {
    pub async fn connect(
        config: &Config,
        mut descriptor: Descriptor,
    ) -> std::result::Result<Self, String> {
        let serial = CLIENTS.fetch_add(1, Ordering::Relaxed) + 1;
        let mut pool = Pool::connect(&config.socket).await;
        descriptor.pool_name = pool.hello.pool_name.clone();
        let mut ident = Ident::new(&format!("l0-{serial}"));
        let mut id = [0u8; 16];
        id[..4].copy_from_slice(&std::process::id().to_le_bytes());
        id[4..12].copy_from_slice(&serial.to_le_bytes());
        ident.instance_id = id;
        ident.scope = scope_of(&descriptor);
        ident.claim_slots = 4;
        let mut session = pool.open(&ident).await;
        if session.opened.status != v2::OpenStatus::OpenOk as i32 {
            return Err(format!("session refused: {:?}", session.opened.status));
        }
        session.snapshot(vec![]).await;
        Ok(Self {
            pool,
            session,
            descriptor,
            next_put: 0,
            next_claim: 0,
            next_lookup: 0,
        })
    }

    pub fn content(
        &self,
        key: Key,
        parent: Option<Key>,
        first: u32,
    ) -> (flare_selene::shared::record::Header, Vec<u8>, Vec<u64>) {
        content_for(&self.descriptor, key, parent, first)
    }

    async fn next<T>(&mut self, mut pick: impl FnMut(&M) -> Option<T>) -> T {
        match self.session.take_within(WAIT, &mut pick).await {
            Some(found) => found,
            None => panic!(
                "the manager sent nothing expected (stream ended: {}); unmatched: {:?}",
                self.session.ended, self.session.backlog
            ),
        }
    }

    // ------------------------------------------------------------------ lookup

    /// The manager's own entries visible to this scope, deepest first, at most the
    /// manager's candidate limit.
    pub async fn records(&mut self) -> Vec<v2::Candidate> {
        self.lookup_before(u64::MAX).await
    }

    async fn lookup_before(&mut self, before_tokens: u64) -> Vec<v2::Candidate> {
        self.next_lookup += 1;
        let reply = self
            .session
            .lookup_request(v2::LookupRequest {
                lookup_id: self.next_lookup,
                boundaries: vec![v2::PrefixKey {
                    consumed_tokens: 0,
                    digest: BUCKET.to_vec(),
                }],
                before_tokens,
                resident_tokens: 0,
            })
            .await;
        reply.candidates
    }

    /// The manager's entry ending at exactly `tokens`, if it offers one now.
    pub async fn candidate(&mut self, tokens: u64) -> Option<v2::Candidate> {
        self.lookup_before(tokens + 1)
            .await
            .into_iter()
            .find(|candidate| candidate.key.as_ref().unwrap().consumed_tokens == tokens)
    }

    /// Polls until the manager offers the entry ending at `tokens`: the index follows
    /// the store by a moment.
    pub async fn wait_candidate(&mut self, tokens: u64, within: Duration) -> Option<v2::Candidate> {
        let deadline = Instant::now() + within;
        loop {
            if let Some(found) = self.candidate(tokens).await {
                return Some(found);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Polls until the manager no longer offers the entry ending at `tokens`.
    pub async fn wait_absent(&mut self, tokens: u64, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        loop {
            if self.candidate(tokens).await.is_none() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Polls until the manager offers exactly `count` entries.
    pub async fn wait_records(&mut self, count: usize) -> Vec<v2::Candidate> {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let records = self.records().await;
            if records.len() == count {
                return records;
            }
            assert!(
                Instant::now() < deadline,
                "the manager offers {} entries, not {count}",
                records.len()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    pub async fn goodbye(&self) {
        self.session.goodbye(true).await;
    }

    // ------------------------------------------------------------------ put

    fn begin(&self, id: u64, key: Key, parent: Option<Key>) -> v2::PutBegin {
        let page_tokens = self.descriptor.page_tokens;
        let chain_pages = key.consumed_tokens / page_tokens;
        v2::PutBegin {
            put_id: id,
            key: Some(key_to(key)),
            chain_pages,
            row: true,
            has_published_parent: parent.is_some(),
            published_parent: parent.map(key_to),
            bucket: BUCKET.to_vec(),
            published_tail_pages: chain_pages
                - parent.map_or(0, |parent| parent.consumed_tokens / page_tokens),
            mode: v2::CopyMode::SharedWindows as i32,
        }
    }

    /// Sends `PutBegin` and returns the put id without waiting for the answer.
    pub async fn send_begin(&mut self, key: Key, parent: Option<Key>) -> u64 {
        self.next_put += 1;
        let id = self.next_put;
        let begin = self.begin(id, key, parent);
        self.session.send(c::M::PutBegin(begin)).await;
        id
    }

    /// The manager's first answer to a put: accepted, or refused with a reason.
    pub async fn first_answer(&mut self, id: u64) -> PutStart {
        self.next(|m| match m {
            M::PutAccept(accept) if accept.put_id == id => Some(PutStart::Accepted(accept.clone())),
            M::PutRefused(refused) if refused.put_id == id => {
                Some(PutStart::Refused(refused.reason as u8))
            }
            _ => None,
        })
        .await
    }

    pub async fn start_put(&mut self, key: Key, parent: Option<Key>) -> (u64, PutStart) {
        let id = self.send_begin(key, parent).await;
        let start = self.first_answer(id).await;
        (id, start)
    }

    pub async fn cancel_put(&self, id: u64) {
        self.session
            .send(c::M::PutCancel(v2::PutCancel { put_id: id }))
            .await;
    }

    pub async fn fence_put(&self, id: u64) {
        self.session
            .send(c::M::FenceAck(v2::FenceAck {
                subject: Some(v2::fence_ack::Subject::PutId(id)),
                outcome: v2::FenceOutcome::Drained as i32,
            }))
            .await;
    }

    /// Cancels a put and fences it: this writer is not touching its windows.
    pub async fn abandon_put(&self, id: u64) {
        self.cancel_put(id).await;
        self.fence_put(id).await;
    }

    pub async fn store(&mut self, key: Key, parent: Option<Key>) -> Vec<u8> {
        match self.put(key, parent, PutOptions::default()).await {
            PutOutcome::Stored(bytes) => bytes,
            other => panic!("store failed for {key:?} parent {parent:?}: {other:?}"),
        }
    }

    pub async fn put(&mut self, key: Key, parent: Option<Key>, options: PutOptions) -> PutOutcome {
        self.next_put += 1;
        match super::put::put(
            &mut self.session,
            &self.descriptor,
            self.next_put,
            key,
            parent,
            options,
            None,
        )
        .await
        {
            super::put::Outcome::Stored { payload, .. } => PutOutcome::Stored(payload),
            super::put::Outcome::Refused(reason) => PutOutcome::Refused(reason.reason as u8),
            super::put::Outcome::Cancelled => PutOutcome::Cancelled,
            super::put::Outcome::Abandoned => PutOutcome::Abandoned,
        }
    }

    // ------------------------------------------------------------------ get

    pub async fn fetch(&mut self, keys: &[Key], skip: u32, pages: u32) -> Vec<Vec<u8>> {
        match self.get(keys, skip, pages, GetOptions::default()).await {
            GetOutcome::Done(records) => records,
            other => panic!("fetch failed: {other:?}"),
        }
    }

    /// Restores the chain ending at the last key: `skip` pages are already held and
    /// `pages` are expected after them.
    pub async fn get(
        &mut self,
        keys: &[Key],
        skip: u32,
        pages: u32,
        options: GetOptions,
    ) -> GetOutcome {
        let mut spec = GetSpec::new(keys, skip, pages);
        spec.options = options;
        self.get_many(vec![spec]).await.remove(0)
    }

    pub async fn fetch_two(
        &mut self,
        left: (Vec<Key>, u32, u32),
        right: (Vec<Key>, u32, u32),
    ) -> (GetOutcome, GetOutcome) {
        let mut outcomes = self
            .get_many(vec![
                GetSpec::new(&left.0, left.1, left.2),
                GetSpec::new(&right.0, right.1, right.2),
            ])
            .await;
        let right = outcomes.pop().unwrap();
        (outcomes.pop().unwrap(), right)
    }

    /// Run several claims at once on this session.
    pub async fn get_many(&mut self, specs: Vec<GetSpec>) -> Vec<GetOutcome> {
        self.claim_all(&specs).await
    }

    async fn claim_all(&mut self, specs: &[GetSpec]) -> Vec<GetOutcome> {
        let page_bytes = self.descriptor.page_bytes();
        let row_bytes = self.descriptor.row_bytes();
        let mut runs = Vec::new();
        for spec in specs {
            let endpoint = spec.keys.last().unwrap().consumed_tokens;
            let Some(candidate) = self.wait_candidate(endpoint, Duration::from_secs(5)).await
            else {
                runs.push(Run::refused(v2::ClaimReason::UnknownSource));
                continue;
            };
            self.next_claim += 1;
            let id = self.next_claim;
            let begin = u64::from(spec.skip) * self.descriptor.page_tokens;
            let claim = self.session.claim_for(id, &candidate, begin, true);
            self.session.claim(claim).await;
            runs.push(Run::new(id, spec));
        }
        while runs.iter().any(|run| run.outcome.is_none()) {
            let ids: Vec<u64> = runs
                .iter()
                .filter(|run| run.outcome.is_none())
                .map(|run| run.id)
                .collect();
            let event = self
                .next(|m| match m {
                    M::ClaimGranted(granted) if ids.contains(&granted.claim_id) => {
                        Some(Event::Granted(granted.clone()))
                    }
                    M::WindowReady(ready) if ids.contains(&ready.claim_id) => {
                        Some(Event::Ready(ready.clone()))
                    }
                    M::ClaimTerminal(terminal) if ids.contains(&terminal.claim_id) => {
                        Some(Event::Terminal(*terminal))
                    }
                    _ => None,
                })
                .await;
            let id = match &event {
                Event::Granted(granted) => granted.claim_id,
                Event::Ready(ready) => ready.claim_id,
                Event::Terminal(terminal) => terminal.claim_id,
            };
            let position = runs.iter().position(|run| run.id == id).unwrap();
            let mut run = std::mem::replace(
                &mut runs[position],
                Run::refused(v2::ClaimReason::UnknownClaim),
            );
            match event {
                Event::Granted(granted) => {
                    run.expected = expected_bytes(&granted, page_bytes, row_bytes);
                    run.layout = granted
                        .records
                        .iter()
                        .map(|record| (record.first_page, record.pages))
                        .collect();
                    run.records = granted
                        .records
                        .iter()
                        .map(|record| {
                            vec![
                                0u8;
                                (u64::from(record.pages) * page_bytes
                                    + if record.row { row_bytes } else { 0 })
                                    as usize
                            ]
                        })
                        .collect();
                    if granted.total_pages != u64::from(run.pages) {
                        run.abandoned = true;
                        self.session
                            .copy_result(id, v2::CopyOutcome::CopyAbandoned)
                            .await;
                    }
                }
                Event::Ready(ready) => self.receive(&mut run, ready).await,
                Event::Terminal(terminal) => {
                    if terminal.fence_required && !(run.cancelled && run.options.leave_leased) {
                        self.session.fence(id, v2::FenceOutcome::Drained).await;
                    }
                    let reason = v2::ClaimReason::try_from(terminal.reason)
                        .unwrap_or(v2::ClaimReason::Unspecified);
                    run.outcome = Some(if run.cancelled {
                        GetOutcome::Cancelled
                    } else if run.abandoned {
                        GetOutcome::Refused(v2::ClaimReason::InvalidClaim)
                    } else if terminal.state == v2::ClaimState::Copied as i32 {
                        GetOutcome::Done(std::mem::take(&mut run.records))
                    } else {
                        GetOutcome::Refused(reason)
                    });
                }
            }
            runs[position] = run;
        }
        runs.into_iter().map(|run| run.outcome.unwrap()).collect()
    }

    /// Copies one ready window out of the shared region and releases it.
    async fn receive(&mut self, run: &mut Run, ready: v2::WindowReady) {
        let id = run.id;
        if run.cancelled || run.abandoned || run.damaged {
            return;
        }
        if run.options.cancel == Some(CancelAt::Grant) {
            run.cancelled = true;
            self.session
                .send(c::M::ClaimRelease(v2::ClaimRelease { claim_id: id }))
                .await;
            return;
        }
        let window = ready.window.unwrap();
        let bytes = self.session.shm.as_ref().unwrap().read(&window);
        let page = self.descriptor.page_bytes() as usize;
        let mut sound = ready.page_xxh3.len() == window.page_count as usize;
        for (index, sum) in ready.page_xxh3.iter().enumerate() {
            sound &= *sum == pool::xxh3(&bytes[index * page..(index + 1) * page]);
        }
        if ready.has_row {
            sound &= ready.row_xxh3 == pool::xxh3(&bytes[window.page_count as usize * page..]);
        }
        if !sound {
            run.damaged = true;
            self.session
                .copy_result(id, v2::CopyOutcome::CopyChecksum)
                .await;
            return;
        }
        let (first, pages) = run.layout[window.record as usize];
        let at = if window.page_count > 0 {
            (window.first_page - first) as usize * page
        } else {
            pages as usize * page
        };
        run.records[window.record as usize][at..at + bytes.len()].copy_from_slice(&bytes);
        run.received += window.length;
        self.session.released(id, window.index).await;
        if run.received == run.expected && !run.finished {
            run.finished = true;
            self.session.copy_result(id, v2::CopyOutcome::CopyOk).await;
        }
    }

    // ------------------------------------------------------------------ leases

    /// Claims a chain and keeps whatever windows the manager issues, releasing none.
    pub async fn hold(&mut self, keys: &[Key], skip: u32, pages: u32) -> Held {
        let endpoint = keys.last().unwrap().consumed_tokens;
        let candidate = self
            .wait_candidate(endpoint, Duration::from_secs(5))
            .await
            .expect("the manager offers the record");
        self.next_claim += 1;
        let id = self.next_claim;
        let begin = u64::from(skip) * self.descriptor.page_tokens;
        let claim = self.session.claim_for(id, &candidate, begin, true);
        self.session.claim(claim).await;
        let granted = self.session.claim_granted(id).await;
        assert_eq!(granted.total_pages, u64::from(pages));
        let mut windows = Vec::new();
        let mut quiet = |m: &M| match m {
            M::WindowReady(ready) if ready.claim_id == id => Some(ready.clone()),
            _ => None,
        };
        while let Some(ready) = self
            .session
            .take_within(Duration::from_millis(100), &mut quiet)
            .await
        {
            windows.push(ready);
        }
        Held {
            id,
            granted,
            windows,
        }
    }

    /// Releases what was held and carries the claim to its end.
    pub async fn finish(&mut self, held: Held) {
        let id = held.id;
        let expected = expected_bytes(
            &held.granted,
            self.descriptor.page_bytes(),
            self.descriptor.row_bytes(),
        );
        let mut received = 0;
        for ready in &held.windows {
            let window = ready.window.as_ref().unwrap();
            received += window.length;
            self.session.released(id, window.index).await;
        }
        while received < expected {
            let ready = self.session.window_ready(id).await;
            let window = ready.window.unwrap();
            received += window.length;
            self.session.released(id, window.index).await;
        }
        self.session.copy_result(id, v2::CopyOutcome::CopyOk).await;
        let terminal = self.session.claim_terminal(id).await;
        assert_eq!(
            terminal.state,
            v2::ClaimState::Copied as i32,
            "{terminal:?}"
        );
        self.session.fence(id, v2::FenceOutcome::Drained).await;
    }

    /// Releases the claim without releasing or fencing its windows.
    pub async fn drop_held(&mut self, held: &Held) -> v2::ClaimTerminal {
        self.session
            .send(c::M::ClaimRelease(v2::ClaimRelease { claim_id: held.id }))
            .await;
        self.session.claim_terminal(held.id).await
    }
}
