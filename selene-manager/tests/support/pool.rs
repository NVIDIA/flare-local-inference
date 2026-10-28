#![allow(dead_code)]
//! A minimal wire-level client for the pool protocol. It speaks `selene.v2` to a
//! real manager; there is no model of the manager's state machine in here.

use selene_manager::{
    Key,
    proto::v2::{
        self, ClientMessage, Header, ServerMessage, client_message as c, pool_client::PoolClient,
        server_message::M,
    },
};
use std::{
    collections::VecDeque,
    fs::{self, File},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    Request, Streaming,
    transport::{Channel, Endpoint, Uri},
};

pub const PAGE_TOKENS: u64 = 64;
pub const PAGE_BYTES: u64 = 4096;
pub const ROW_BYTES: u64 = 4096;

pub fn key(tokens: u64) -> Key {
    Key {
        consumed_tokens: tokens,
        digest: (u128::from(tokens) * 7919 + 1).to_le_bytes(),
    }
}

pub fn pkey(tokens: u64) -> v2::PrefixKey {
    let key = key(tokens);
    v2::PrefixKey {
        consumed_tokens: key.consumed_tokens,
        digest: key.digest.to_vec(),
    }
}

/// The bucket of a record: the digest of the 1024-token boundary at or below it.
pub fn bucket_of(tokens: u64) -> Vec<u8> {
    key(tokens / 1024 * 1024).digest.to_vec()
}

/// The boundary keys of a history of `tokens` tokens.
pub fn boundaries(tokens: u64) -> Vec<v2::PrefixKey> {
    (0..=tokens / 1024).map(|n| pkey(n * 1024)).collect()
}

pub fn scope_for(model: u8) -> v2::Scope {
    v2::Scope {
        model_identity: vec![model; 32],
        layout_fingerprint: vec![3; 16],
        geometry: Some(v2::Geometry {
            page_tokens: PAGE_TOKENS,
            ranks: vec![v2::RankPlanes {
                plane_bytes: vec![4000],
                row_bytes: 1000,
            }],
            page_record_bytes: PAGE_BYTES,
            row_record_bytes: ROW_BYTES,
        }),
    }
}

pub fn scope() -> v2::Scope {
    scope_for(7)
}

pub fn entry(
    tokens: u64,
    parent_tokens: Option<u64>,
    tier: v2::Tier,
) -> v2::IndexEntry {
    let parent_pages = parent_tokens.map_or(0, |parent| parent / PAGE_TOKENS);
    v2::IndexEntry {
        key: Some(pkey(tokens)),
        has_parent: parent_tokens.is_some(),
        parent: parent_tokens.map(pkey),
        bucket: bucket_of(tokens),
        tail_pages: (tokens / PAGE_TOKENS - parent_pages) as u32,
        row: true,
        tier: tier as i32,
        bytes: 1000,
    }
}

/// A chain of entries at the given token counts, each the child of the previous.
pub fn chain(tokens: &[u64], tier: v2::Tier) -> Vec<v2::IndexEntry> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            entry(
                *token,
                index.checked_sub(1).map(|previous| tokens[previous]),
                tier,
            )
        })
        .collect()
}

pub fn saved(entry: v2::IndexEntry) -> v2::Notice {
    v2::Notice {
        kind: Some(v2::notice::Kind::Saved(v2::Saved { entry: Some(entry) })),
    }
}

pub fn evicted(tokens: u64, tier: v2::Tier) -> v2::Notice {
    v2::Notice {
        kind: Some(v2::notice::Kind::Evicted(v2::Evicted {
            key: Some(pkey(tokens)),
            tier: tier as i32,
            kind: v2::EvictKind::Removed as i32,
            heir: None,
        })),
    }
}

pub fn spilled(to: v2::IndexEntry, from_tier: v2::Tier) -> v2::Notice {
    v2::Notice {
        kind: Some(v2::notice::Kind::Spilled(v2::Spilled {
            to: Some(to),
            from_tier: from_tier as i32,
        })),
    }
}

pub fn default_order() -> Vec<i32> {
    vec![
        v2::Level::NeighbourGpu as i32,
        v2::Level::HostDram as i32,
        v2::Level::HostNvme as i32,
        v2::Level::RemoteNvme as i32,
    ]
}

pub use flare_selene::shared::record::checksum as xxh3;

/// A client's identity for one process.
#[derive(Clone)]
pub struct Ident {
    pub instance_id: [u8; 16],
    pub nonce: [u8; 16],
    pub name: String,
    pub scope: v2::Scope,
    pub export_slots: u32,
    pub claim_slots: u32,
}

impl Ident {
    pub fn new(name: &str) -> Self {
        let mut instance_id = [0u8; 16];
        for (slot, byte) in instance_id.iter_mut().zip(name.bytes().cycle()) {
            *slot = byte;
        }
        let mut nonce = [0u8; 16];
        use std::io::Read;
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut nonce))
            .unwrap();
        Self {
            instance_id,
            nonce,
            name: name.into(),
            scope: scope(),
            export_slots: 2,
            claim_slots: 2,
        }
    }

    /// The same instance as a new process.
    pub fn restarted(&self) -> Self {
        let mut next = self.clone();
        let mut nonce = [0u8; 16];
        use std::io::Read;
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut nonce))
            .unwrap();
        next.nonce = nonce;
        next
    }
}

/// The unary side and the shared-memory view.
pub struct Pool {
    pub client: PoolClient<Channel>,
    pub hello: v2::HelloReply,
    pub socket: PathBuf,
}

pub async fn channel(socket: &Path) -> Channel {
    let path = socket.to_path_buf();
    Endpoint::try_from("http://[::]:0")
        .unwrap()
        .connect_with_connector(tower::service_fn(move |_: Uri| {
            let path = path.clone();
            async move {
                let stream = tokio::net::UnixStream::connect(path).await?;
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
            }
        }))
        .await
        .expect("connect to the manager")
}

impl Pool {
    pub async fn connect(socket: &Path) -> Self {
        Self::over(channel(socket).await, socket).await
    }

    /// The unary and session calls over a channel the caller made (for example TCP).
    pub async fn over(channel: Channel, socket: &Path) -> Self {
        let mut client = PoolClient::new(channel);
        let hello = client
            .hello(Request::new(v2::HelloRequest { protocol: 2 }))
            .await
            .expect("hello")
            .into_inner();
        Self {
            client,
            hello,
            socket: socket.to_path_buf(),
        }
    }

    pub async fn hello_again(&mut self) -> v2::HelloReply {
        self.client
            .hello(Request::new(v2::HelloRequest { protocol: 2 }))
            .await
            .expect("hello")
            .into_inner()
    }

    /// Opens a session, returning it and the manager's `SessionOpened`.
    pub async fn open(&mut self, ident: &Ident) -> Session {
        self.open_with(ident, self.hello.manager_incarnation).await
    }

    pub async fn open_with(&mut self, ident: &Ident, incarnation: u64) -> Session {
        let (tx, rx) = mpsc::channel(256);
        tx.send(ClientMessage {
            header: None,
            m: Some(c::M::Open(v2::OpenSession {
                manager_incarnation: incarnation,
                instance_id: ident.instance_id.to_vec(),
                scope: Some(ident.scope.clone()),
                process_nonce: ident.nonce.to_vec(),
                export_slots: ident.export_slots,
                claim_slots: ident.claim_slots,
                ..Default::default()
            })),
        })
        .await
        .unwrap();
        let mut stream = self
            .client
            .session(Request::new(ReceiverStream::new(rx)))
            .await
            .expect("session")
            .into_inner();
        let first = tokio::time::timeout(Duration::from_secs(5), stream.message())
            .await
            .expect("opened in time")
            .expect("stream")
            .expect("opened");
        let Some(M::Opened(opened)) = first.m else {
            panic!("first message was not `opened`");
        };
        let shm = self.hello.shared_memory.as_ref().map(|shm| Shm {
            arena: fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&shm.arena_path)
                .unwrap(),
            staging: fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&shm.staging_path)
                .unwrap(),
        });
        Session {
            tx,
            rx: stream,
            incarnation: self.hello.manager_incarnation,
            sid: opened.session_incarnation,
            opened,
            barrier_id: u64::MAX,
            backlog: VecDeque::new(),
            ended: false,
            shm,
        }
    }
}

pub struct Shm {
    pub arena: File,
    pub staging: File,
}

impl Shm {
    pub fn file(&self, region: i32) -> &File {
        if region == v2::Region::Arena as i32 {
            &self.arena
        } else {
            &self.staging
        }
    }

    pub fn write(&self, window: &v2::WindowSpec, bytes: &[u8]) {
        assert_eq!(bytes.len() as u64, window.length);
        self.file(window.region)
            .write_all_at(bytes, window.offset)
            .unwrap();
    }

    pub fn read(&self, window: &v2::WindowSpec) -> Vec<u8> {
        let mut bytes = vec![0u8; window.length as usize];
        self.file(window.region)
            .read_exact_at(&mut bytes, window.offset)
            .unwrap();
        bytes
    }
}

pub struct Session {
    pub tx: mpsc::Sender<ClientMessage>,
    pub rx: Streaming<ServerMessage>,
    pub incarnation: u64,
    pub sid: u64,
    pub opened: v2::SessionOpened,
    pub barrier_id: u64,
    pub backlog: VecDeque<M>,
    pub ended: bool,
    pub shm: Option<Shm>,
}

impl Session {
    pub fn header(&self) -> Header {
        Header {
            manager_incarnation: self.incarnation,
            session_incarnation: self.sid,
        }
    }

    pub async fn send(&self, m: c::M) {
        let _ = self
            .tx
            .send(ClientMessage {
                header: Some(self.header()),
                m: Some(m),
            })
            .await;
    }

    /// Sends with a chosen header (stale incarnations).
    pub async fn send_with(&self, header: Header, m: c::M) {
        let _ = self
            .tx
            .send(ClientMessage {
                header: Some(header),
                m: Some(m),
            })
            .await;
    }

    /// The next message matching `pick`; others wait in the backlog.
    pub async fn take<T>(&mut self, mut pick: impl FnMut(&M) -> Option<T>) -> T {
        match self.take_within(Duration::from_secs(5), &mut pick).await {
            Some(found) => found,
            None => panic!(
                "the expected message did not arrive (stream ended: {}); unmatched: {:?}",
                self.ended, self.backlog
            ),
        }
    }

    pub async fn take_within<T>(
        &mut self,
        wait: Duration,
        pick: &mut impl FnMut(&M) -> Option<T>,
    ) -> Option<T> {
        for index in 0..self.backlog.len() {
            if let Some(found) = pick(&self.backlog[index]) {
                self.backlog.remove(index);
                return Some(found);
            }
        }
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, self.rx.message()).await {
                Ok(Ok(Some(message))) => {
                    let Some(m) = message.m else { continue };
                    if let Some(found) = pick(&m) {
                        return Some(found);
                    }
                    self.backlog.push_back(m);
                }
                Ok(Ok(None)) | Ok(Err(_)) => {
                    self.ended = true;
                    return None;
                }
                Err(_) => return None,
            }
        }
    }

    /// Whether any message matching `pick` arrives within `wait`.
    pub async fn never<T>(
        &mut self,
        wait: Duration,
        mut pick: impl FnMut(&M) -> Option<T>,
    ) -> bool {
        self.take_within(wait, &mut pick).await.is_none()
    }

    /// Waits for the stream to end, returning any `SessionClosing` seen first.
    pub async fn closed(&mut self) -> Option<v2::SessionClosing> {
        let mut reason = None;
        for index in 0..self.backlog.len() {
            if let M::Closing(closing) = &self.backlog[index] {
                reason = Some(closing.clone());
            }
        }
        loop {
            match tokio::time::timeout(Duration::from_secs(5), self.rx.message()).await {
                Ok(Ok(Some(message))) => {
                    if let Some(M::Closing(closing)) = message.m {
                        reason = Some(closing);
                    }
                }
                Ok(Ok(None)) | Ok(Err(_)) => {
                    self.ended = true;
                    return reason;
                }
                Err(_) => panic!("the stream did not end"),
            }
        }
    }

    // ------------------------------------------------------------ index

    pub async fn snapshot(&mut self, entries: Vec<v2::IndexEntry>) {
        self.send(c::M::ResyncBegin(v2::ResyncBegin {})).await;
        for chunk in entries.chunks(512) {
            self.send(c::M::ResyncChunk(v2::ResyncChunk {
                entries: chunk.to_vec(),
            }))
            .await;
        }
        self.send(c::M::ResyncEnd(v2::ResyncEnd {})).await;
        self.take(|m| matches!(m, M::ResyncAck(_)).then_some(()))
            .await;
    }

    /// An ordered lookup reply confirms that prior messages were processed.
    pub async fn barrier(&mut self) {
        self.barrier_id -= 1;
        self.lookup_request(v2::LookupRequest {
            lookup_id: self.barrier_id,
            boundaries: vec![],
            before_tokens: 0,
            resident_tokens: 0,
        })
        .await;
    }

    pub async fn notices(&mut self, notices: Vec<v2::Notice>) {
        self.send(c::M::Notices(v2::NoticeBatch { notices })).await;
        self.barrier().await;
    }

    pub async fn resync_required(&mut self) -> v2::ResyncRequired {
        self.take(|m| match m {
            M::ResyncRequired(required) => Some(*required),
            _ => None,
        })
        .await
    }

    // ------------------------------------------------------------ lookup

    pub async fn lookup(
        &mut self,
        id: u64,
        history_tokens: u64,
        resident_tokens: u64,
    ) -> v2::LookupReply {
        self.lookup_request(v2::LookupRequest {
            lookup_id: id,
            boundaries: boundaries(history_tokens),
            before_tokens: history_tokens + 1,
            resident_tokens,
        })
        .await
    }

    pub async fn lookup_request(&mut self, request: v2::LookupRequest) -> v2::LookupReply {
        let id = request.lookup_id;
        self.send(c::M::Lookup(request)).await;
        self.take(|m| match m {
            M::LookupReply(reply) if reply.lookup_id == id => Some(reply.clone()),
            _ => None,
        })
        .await
    }

    // ------------------------------------------------------------ claims

    pub fn claim_for(
        &self,
        id: u64,
        candidate: &v2::Candidate,
        begin_tokens: u64,
        want_row: bool,
    ) -> v2::Claim {
        v2::Claim {
            claim_id: id,
            source: candidate.source.clone(),
            tier: candidate.tier,
            key: candidate.key.clone(),
            begin_tokens,
            want_row,
            mode: v2::CopyMode::SharedWindows as i32,
            deadline_ms: 0,
        }
    }

    pub async fn claim(&self, claim: v2::Claim) {
        self.send(c::M::Claim(claim)).await;
    }

    pub async fn claim_terminal(&mut self, id: u64) -> v2::ClaimTerminal {
        self.take(|m| match m {
            M::ClaimTerminal(terminal) if terminal.claim_id == id => Some(*terminal),
            _ => None,
        })
        .await
    }

    pub async fn claim_granted(&mut self, id: u64) -> v2::ClaimGranted {
        self.take(|m| match m {
            M::ClaimGranted(granted) if granted.claim_id == id => Some(granted.clone()),
            _ => None,
        })
        .await
    }

    pub async fn window_ready(&mut self, id: u64) -> v2::WindowReady {
        self.take(|m| match m {
            M::WindowReady(ready) if ready.claim_id == id => Some(ready.clone()),
            _ => None,
        })
        .await
    }

    pub async fn export_request(&mut self) -> v2::ExportRequest {
        self.take(|m| match m {
            M::ExportRequest(request) => Some(request.clone()),
            _ => None,
        })
        .await
    }

    pub async fn export_fill(&mut self) -> v2::ExportFill {
        self.take(|m| match m {
            M::ExportFill(fill) => Some(fill.clone()),
            _ => None,
        })
        .await
    }

    pub async fn export_cancel(&mut self) -> v2::ExportCancel {
        self.take(|m| match m {
            M::ExportCancel(cancel) => Some(*cancel),
            _ => None,
        })
        .await
    }

    pub async fn pin_ok(&self, request: &v2::ExportRequest, records: Vec<v2::Record>, row: bool) {
        let total = records.iter().map(|record| u64::from(record.pages)).sum();
        self.send(c::M::PinAck(v2::PinAck {
            export_id: request.export_id,
            result: v2::PinResult::Pinned as i32,
            refusal: 0,
            records,
            total_pages: total,
            row,
        }))
        .await;
    }

    pub async fn pin_refused(&self, request: &v2::ExportRequest, reason: v2::ClaimReason) {
        self.send(c::M::PinAck(v2::PinAck {
            export_id: request.export_id,
            result: v2::PinResult::Refused as i32,
            refusal: reason as i32,
            records: vec![],
            total_pages: 0,
            row: false,
        }))
        .await;
    }

    pub async fn filled(&self, export_id: u64, window: &v2::WindowSpec, bytes: &[u8]) {
        if let Some(shm) = &self.shm {
            shm.write(window, bytes);
        }
        let mut page_xxh3 = Vec::new();
        for page in 0..window.page_count as usize {
            page_xxh3.push(xxh3(
                &bytes[page * PAGE_BYTES as usize..(page + 1) * PAGE_BYTES as usize],
            ));
        }
        let (row_xxh3, has_row) = if window.row {
            let pages = window.page_count as usize * PAGE_BYTES as usize;
            (xxh3(&bytes[pages..]), true)
        } else {
            (0, false)
        };
        self.send(c::M::WindowFilled(v2::WindowFilled {
            export_id,
            index: window.index,
            status: v2::FillStatus::FillOk as i32,
            page_xxh3,
            row_xxh3,
            has_row,
            ..Default::default()
        }))
        .await;
    }

    pub async fn export_end(
        &self,
        export_id: u64,
        status: v2::ExportEndStatus,
        fence: v2::FenceOutcome,
    ) {
        self.send(c::M::ExportEnd(v2::ExportEnd {
            export_id,
            status: status as i32,
            fence: fence as i32,
        }))
        .await;
    }

    pub async fn released(&self, claim_id: u64, index: u32) {
        self.send(c::M::WindowReleased(v2::WindowReleased { claim_id, index }))
            .await;
    }

    pub async fn copy_result(&self, claim_id: u64, outcome: v2::CopyOutcome) {
        self.send(c::M::CopyResult(v2::CopyResult {
            claim_id,
            outcome: outcome as i32,
        }))
        .await;
    }

    pub async fn fence(&self, claim_id: u64, outcome: v2::FenceOutcome) {
        self.send(c::M::FenceAck(v2::FenceAck {
            subject: Some(v2::fence_ack::Subject::ClaimId(claim_id)),
            outcome: outcome as i32,
        }))
        .await;
    }

    pub async fn goodbye(&self, quiesced: bool) {
        self.send(c::M::Goodbye(v2::Goodbye { quiesced })).await;
    }
}

/// The bytes of the page with the given global page number, deterministic by source.
pub fn page_bytes(seed: u8, page: u64) -> Vec<u8> {
    (0..PAGE_BYTES as usize)
        .map(|offset| (page as usize * 31 + offset + seed as usize) as u8)
        .collect()
}

pub fn row_bytes(seed: u8, endpoint_tokens: u64) -> Vec<u8> {
    (0..ROW_BYTES as usize)
        .map(|offset| (endpoint_tokens as usize * 17 + offset + seed as usize) as u8)
        .collect()
}

/// The records a source would pin for a chain, given the physical start page.
pub fn records_for(
    tokens: &[u64],
    start_page: u64,
    want_row: bool,
) -> Vec<v2::Record> {
    let mut records = Vec::new();
    let mut previous_end = 0;
    for (index, token) in tokens.iter().enumerate() {
        let end = token / PAGE_TOKENS;
        let first = start_page
            .saturating_sub(previous_end)
            .min(end - previous_end);
        let pages = end - previous_end - first;
        let last = index + 1 == tokens.len();
        if end > start_page || last {
            records.push(v2::Record {
                key: Some(pkey(*token)),
                first_page: first as u32,
                pages: pages as u32,
                row: last && want_row,
            });
        }
        previous_end = end;
    }
    records
}

/// The outcome of a scripted put.
#[derive(Debug)]
pub enum PutResult {
    Stored(u64),
    Refused(v2::PutRefused),
}

/// Deliberate damage for the put tests.
#[derive(Clone, Copy, Default)]
pub struct Damage {
    /// Flip a bit in this record-relative page of the payload before checksumming.
    pub corrupt_page: Option<usize>,
    /// Commit metadata whose header names another key.
    pub wrong_key: bool,
    /// Stop after the accept and cancel.
    pub cancel_after_accept: bool,
}

impl Session {
    /// Writes one record into the manager's store through shared windows.
    pub async fn put_record(
        &mut self,
        put_id: u64,
        key: Key,
        parent: Option<Key>,
        damage: Damage,
    ) -> PutResult {
        super::put::put_record(self, None, put_id, key, parent, damage).await
    }

    /// Waits until the manager's own entry for `tokens` shows up in lookups.
    pub async fn manager_candidate(&mut self, id: u64, tokens: u64) -> v2::Candidate {
        for round in 0..200 {
            let reply = self
                .lookup_request(v2::LookupRequest {
                    lookup_id: id * 1000 + round,
                    boundaries: vec![v2::PrefixKey {
                        consumed_tokens: 0,
                        digest: vec![4; 16],
                    }],
                    before_tokens: tokens + 1,
                    resident_tokens: 0,
                })
                .await;
            if let Some(found) = reply
                .candidates
                .iter()
                .find(|c| c.key.as_ref().unwrap().consumed_tokens == tokens)
            {
                return found.clone();
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the manager never offered {tokens}");
    }
}
