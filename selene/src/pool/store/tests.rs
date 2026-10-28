use std::sync::{Arc, Mutex};

use super::*;
use crate::{
    CheckpointId, Layout, PageId, Retirement, Retiring, StateSlotId,
    levels::Ranking,
    pool::{
        Answer,
        shared::{FetchFailure, PutRefusal, PutSource},
    },
    testing::{TestHistory, geometry},
};

const PT: u64 = 16;

fn layout() -> Layout {
    Layout::new(&[geometry(PT, &[("k", 4096)], 4096)]).unwrap()
}

fn history() -> TestHistory {
    let mut history = TestHistory::new([7; 32]);
    history
        .append(&(0..2000).collect::<Vec<u32>>(), &[])
        .unwrap();
    history
}

fn identity() -> Identity {
    Identity {
        model: [7; 32],
        layout: [1; 16],
    }
}

// ---- fakes ----

#[derive(Default)]
struct LocalState {
    hit: Option<HostHit>,
    fetches: Vec<(HostHit, InboundTicket)>,
    events: Vec<StoreEvent>,
    reject_fetch: bool,
    pinned: Vec<(Tier, PrefixKey)>,
    refuse_pin: Option<ExportRefusal>,
    announced: bool,
    /// What the last local tier offers for a move, coldest first.
    retiring: Vec<Retiring>,
    /// How each offered record's move ended, in order.
    retired: Vec<(PrefixKey, Retirement)>,
    spill: Option<bool>,
    cascaded: bool,
    /// Keys the local tiers were asked to copy, and the answer given.
    stored: Vec<PrefixKey>,
    store_outcome: Option<StoreOutcome>,
    copies_in_flight: usize,
}

#[derive(Clone, Default)]
struct FakeLocal(Arc<Mutex<LocalState>>);

impl HostStore for FakeLocal {
    fn lookup(&mut self, _: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
        self.0
            .lock()
            .unwrap()
            .hit
            .clone()
            .filter(|hit| hit.resident_tokens == resident && hit.consumed_tokens < before)
    }
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        let mut state = self.0.lock().unwrap();
        state.stored.push(checkpoint.key);
        Ok(state.store_outcome.unwrap_or(StoreOutcome::Queued))
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        let mut state = self.0.lock().unwrap();
        if state.reject_fetch {
            return Err("rejected".into());
        }
        state.fetches.push((hit, ticket));
        Ok(())
    }
    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        events.append(&mut self.0.lock().unwrap().events);
        Ok(())
    }
    fn stats(&self) -> StoreStats {
        StoreStats {
            outbound_in_flight: self.0.lock().unwrap().copies_in_flight,
            ..StoreStats::default()
        }
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl crate::store::TierStore for FakeLocal {
    fn announce_to(&mut self, _: Announcer) {
        self.0.lock().unwrap().announced = true;
    }
    fn pin_export(
        &mut self,
        tier: Tier,
        key: PrefixKey,
        _: u64,
        _: bool,
    ) -> std::result::Result<Box<dyn HostPin>, ExportRefusal> {
        let mut state = self.0.lock().unwrap();
        state.pinned.push((tier, key));
        match state.refuse_pin {
            Some(reason) => Err(reason),
            None => Ok(Box::new(NoPin(key))),
        }
    }
    fn cascade(&mut self) {
        self.0.lock().unwrap().cascaded = true;
    }
    fn set_spill(&mut self, enabled: bool) {
        self.0.lock().unwrap().spill = Some(enabled);
    }
    fn take_retiring(&mut self) -> Vec<Retiring> {
        std::mem::take(&mut self.0.lock().unwrap().retiring)
    }
    fn retired(&mut self, key: PrefixKey, _: Tier, how: Retirement) {
        self.0.lock().unwrap().retired.push((key, how));
    }
}

struct NoPin(PrefixKey);
impl HostPin for NoPin {
    fn chain(&self) -> &crate::pool::HostChain {
        // The chain a put asks the pin for: a single record of 10 pages.
        static CHAIN: std::sync::OnceLock<crate::pool::HostChain> = std::sync::OnceLock::new();
        CHAIN.get_or_init(|| crate::pool::HostChain {
            records: vec![crate::pool::HostRecord {
                key: self.0,
                first_page: 0,
                pages: 10,
                row: true,
            }],
            total_pages: 10,
            row: true,
        })
    }
    fn read(
        &mut self,
        _: &crate::pool::HostWindow,
        _: &mut [u8],
    ) -> Result<crate::pool::Checksums> {
        unreachable!()
    }
}

#[derive(Default)]
struct PoolState {
    host_pinned: Vec<ExportId>,
    refused: Vec<(ExportId, ExportRefusal)>,
    answer: Option<Arc<Answer>>,
    residents: Vec<u64>,
    fetches: Vec<PoolFetch>,
    reject: bool,
    events: Vec<PoolEvent>,
    saved: Vec<IndexEntry>,
    evicted: Vec<(PrefixKey, Tier, Option<IndexEntry>)>,
    lost: usize,
    ready: bool,
    /// Puts accepted, by id; the test ends them with a [`PoolEvent::Put`].
    puts: Vec<(u64, PutRequest)>,
    /// Refuse every put as a pool whose command queue is full does.
    put_error: bool,
}

#[derive(Clone)]
struct FakePool(Arc<Mutex<PoolState>>, Announcer);

impl PoolPort for FakePool {
    fn begin_lookup(&self, _history: &dyn History, resident: u64) -> LookupTicket {
        let mut state = self.0.lock().unwrap();
        state.residents.push(resident);
        match state.answer.clone() {
            Some(answer) => LookupTicket::watched(LookupState::Ready(answer), || {}),
            None => LookupTicket::watched(LookupState::Pending, || {}),
        }
    }
    fn fetch(&mut self, fetch: PoolFetch) -> Result<u64> {
        let mut state = self.0.lock().unwrap();
        if state.reject {
            return Err("the pool is not ready".into());
        }
        state.fetches.push(fetch);
        Ok(state.fetches.len() as u64)
    }
    fn poll(&mut self, events: &mut Vec<PoolEvent>) -> Result<()> {
        events.append(&mut self.0.lock().unwrap().events);
        Ok(())
    }
    fn saved(&self, entry: IndexEntry) {
        self.0.lock().unwrap().saved.push(entry);
    }
    fn evicted(&self, key: PrefixKey, tier: Tier, heir: Option<IndexEntry>) {
        self.0.lock().unwrap().evicted.push((key, tier, heir));
    }
    fn evictions_lost(&self) {
        self.0.lock().unwrap().lost += 1;
    }
    fn put(&mut self, request: PutRequest) -> Result<u64> {
        let mut state = self.0.lock().unwrap();
        if !state.ready {
            return Err("the fake pool takes no puts".into());
        }
        if state.put_error {
            return Err("the pool's command queue is full".into());
        }
        let id = state.puts.len() as u64 + 1;
        state.puts.push((id, request));
        Ok(id)
    }
    fn cancel_put(&mut self, _: u64) {}
    fn announcer(&self) -> Announcer {
        self.1.clone()
    }
    fn host_pinned(&mut self, export: ExportId, _: Box<dyn HostPin>) {
        self.0.lock().unwrap().host_pinned.push(export);
    }
    fn export_pinned(&mut self, _: ExportId, _: ExportChain) {}
    fn export_refused(&mut self, export: ExportId, reason: ExportRefusal) {
        self.0.lock().unwrap().refused.push((export, reason));
    }
    fn ready(&self) -> bool {
        self.0.lock().unwrap().ready
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Rig {
    store: PooledStore<FakeLocal, FakePool>,
    local: FakeLocal,
    pool: Arc<Mutex<PoolState>>,
    history: TestHistory,
}

fn rig() -> Rig {
    rig_with(Ranking::for_test(
        &[
            (Level::NeighbourGpu, 10.0),
            (Level::HostDram, 10.0),
            (Level::HostNvme, 100.0),
            (Level::RemoteNvme, 200.0),
        ],
    ))
}

fn rig_with(ranking: Ranking) -> Rig {
    let local = FakeLocal::default();
    let pool = Arc::new(Mutex::new(PoolState::default()));
    let mut rig = Rig {
        store: PooledStore::new(
            local.clone(),
            FakePool(
                pool.clone(),
                Announcer::new(Arc::new(tokio::sync::Notify::new())),
            ),
            identity(),
            &layout(),
        ),
        local,
        pool,
        history: history(),
    };
    rig.store.rates = ranking;
    rig
}

impl Rig {
    fn key(&self, tokens: u64) -> PrefixKey {
        self.history.key_at(tokens).unwrap()
    }

    /// A candidate whose records end at `ends`, the first starting at page `first_start`.
    fn candidate(
        &self,
        level: Level,
        source: u8,
        begin: u64,
        first_start: u64,
        ends: &[u64],
        _us: u32,
    ) -> PoolCandidate {
        let mut previous = first_start;
        let nodes: Vec<plan::ChainNode> = ends
            .iter()
            .map(|end| {
                let page = end / PT;
                let node = plan::ChainNode {
                    key: self.key(*end),
                    tail_pages: (page - previous) as usize,
                };
                previous = page;
                node
            })
            .collect();
        PoolCandidate {
            plan: plan::Candidate {
                level,
                source: SourceId(u64::from(source)),
                key: nodes.last().unwrap().key,
                begin_tokens: begin,
                skip_pages: (begin / PT - first_start) as usize,
                nodes,
            },
            source: SourceRef {
                instance: vec![source],
                session_incarnation: 1,
            },
            tier: if level == Level::HostNvme {
                Tier::Nvme
            } else {
                Tier::Dram
            },
        }
    }

    fn answer(&self, candidates: Vec<PoolCandidate>) {
        self.pool.lock().unwrap().answer = Some(Arc::new(Answer { candidates }));
    }

    fn local_hit(&self, end: u64, tier: Tier) {
        let key = self.key(end);
        self.local.0.lock().unwrap().hit = Some(HostHit {
            key,
            consumed_tokens: end,
            resident_tokens: 0,
            pages: (end / PT) as usize,
            bytes: 0,
            tier,
            estimated_seconds: 0.001,
            chain: Match {
                key,
                consumed_tokens: end,
                resident_tokens: 0,
                nodes: vec![key],
                skip_pages: 0,
                pages: (end / PT) as usize,
            },
        });
    }

    fn lookup(&mut self) -> Option<HostHit> {
        let ticket = PoolPort::begin_lookup(&self.store.pool, &self.history, 0);
        let history = self.history.clone();
        self.store.lookup_for(Some(&ticket), &history, 0, u64::MAX)
    }

    fn ticket(&self, hit: &HostHit) -> InboundTicket {
        InboundTicket {
            key: hit.key,
            pages: (0..hit.pages as u32).map(PageId).collect(),
            slot: StateSlotId(3),
        }
    }

    fn poll(&mut self) -> Vec<StoreEvent> {
        let mut events = Vec::new();
        self.store.poll(&mut events).unwrap();
        events
    }
}

fn fetched(events: &[StoreEvent]) -> Vec<(PrefixKey, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            StoreEvent::Fetched { key, result } => Some((*key, result.is_ok())),
            _ => None,
        })
        .collect()
}

fn failure(source: (SourceRef, Tier), no_headroom: bool) -> FetchFailure {
    FetchFailure {
        error: "claim refused".into(),
        source: Some(source),
        no_headroom,
    }
}

// ---- single-source restores ----

#[test]
fn only_successful_pool_restores_refine_the_level_bandwidth() {
    for success in [false, true] {
        let mut rig = rig();
        rig.answer(vec![rig.candidate(Level::HostDram, 1, 0, 0, &[800], 0)]);
        let hit = rig.lookup().unwrap();
        let ticket = rig.ticket(&hit);
        let before = rig.store.rates.clone();
        rig.store.fetch(hit, ticket).unwrap();
        assert_eq!(rig.store.rates, before);
        rig.pool.lock().unwrap().events.push(PoolEvent::Fetched {
            id: 1,
            result: if success {
                Ok(4096)
            } else {
                Err(failure(
                    (
                        SourceRef {
                            instance: vec![1],
                            session_incarnation: 1,
                        },
                        Tier::Dram,
                    ),
                    false,
                ))
            },
        });
        assert_eq!(fetched(&rig.poll()), [(rig.key(800), success)]);
        assert_eq!(rig.store.rates != before, success);
    }
}

#[test]
fn a_longer_pool_copy_is_the_only_source_even_when_a_local_prefix_is_closer() {
    let mut rig = rig();
    rig.local_hit(800, Tier::Dram);
    rig.answer(vec![rig.candidate(
        Level::RemoteNvme,
        2,
        0,
        0,
        &[640, 1600],
        0,
    )]);
    let hit = rig.lookup().unwrap();
    assert_eq!((hit.key, hit.pages), (rig.key(1600), 100));
    let ticket = rig.ticket(&hit);
    rig.store.fetch(hit, ticket.clone()).unwrap();
    assert!(rig.local.0.lock().unwrap().fetches.is_empty());
    let state = rig.pool.lock().unwrap();
    let fetch = &state.fetches[0];
    assert_eq!(fetch.part.pages, ticket.pages);
    assert_eq!(fetch.part.slot, Some(ticket.slot));
    assert!(fetch.part.part.row);
}

#[test]
fn a_longer_local_copy_wins_over_a_closer_pool_copy() {
    let mut rig = rig();
    rig.local_hit(1600, Tier::Nvme);
    rig.answer(vec![rig.candidate(Level::HostDram, 2, 0, 0, &[800], 0)]);
    let hit = rig.lookup().unwrap();
    let ticket = rig.ticket(&hit);
    rig.store.fetch(hit, ticket).unwrap();
    assert_eq!(rig.local.0.lock().unwrap().fetches.len(), 1);
    assert!(rig.pool.lock().unwrap().fetches.is_empty());
}

#[test]
fn a_failed_pool_source_is_not_replanned_and_reports_once_after_drain() {
    let mut rig = rig();
    rig.answer(vec![
        rig.candidate(Level::HostDram, 1, 0, 0, &[1600], 0),
        rig.candidate(Level::HostNvme, 2, 0, 0, &[1600], 0),
    ]);
    let hit = rig.lookup().unwrap();
    let ticket = rig.ticket(&hit);
    rig.store.fetch(hit, ticket).unwrap();
    rig.pool.lock().unwrap().events.push(PoolEvent::Fetched {
        id: 1,
        result: Err(failure(
            (
                SourceRef {
                    instance: vec![1],
                    session_incarnation: 1,
                },
                Tier::Dram,
            ),
            false,
        )),
    });
    assert_eq!(fetched(&rig.poll()), [(rig.key(1600), false)]);
    assert!(rig.poll().is_empty());
    assert_eq!(rig.pool.lock().unwrap().fetches.len(), 1);
}

#[test]
fn polling_expires_headroom_exclusions_without_a_lookup() {
    let mut rig = rig();
    let expired = (SourceRef { instance: vec![1], session_incarnation: 1 }, Tier::Dram);
    let active = (SourceRef { instance: vec![2], session_incarnation: 2 }, Tier::Gpu);
    rig.store.skip.insert(expired.clone(), Instant::now() - SKIP);
    rig.store.skip.insert(active.clone(), Instant::now() + SKIP);
    rig.poll();
    assert!(!rig.store.skip.contains_key(&expired));
    assert!(rig.store.skip.contains_key(&active));
}

#[test]
fn a_source_without_headroom_is_skipped_on_the_next_lookup() {
    let mut rig = rig();
    let candidate = rig.candidate(Level::HostDram, 1, 0, 0, &[1600], 0);
    let source = (candidate.source.clone(), candidate.tier);
    rig.answer(vec![candidate]);
    let hit = rig.lookup().unwrap();
    let ticket = rig.ticket(&hit);
    rig.store.fetch(hit, ticket).unwrap();
    rig.pool.lock().unwrap().events.push(PoolEvent::Fetched {
        id: 1,
        result: Err(failure(source, true)),
    });
    assert_eq!(fetched(&rig.poll()), [(rig.key(1600), false)]);
    assert!(rig.lookup().is_none());
}

#[test]
fn a_failed_pool_restore_falls_back_to_the_local_copy_of_the_same_endpoint() {
    let mut rig = rig();
    rig.local_hit(1600, Tier::Nvme);
    let candidate = rig.candidate(Level::HostDram, 1, 0, 0, &[1600], 0);
    let refused = (candidate.source.clone(), candidate.tier);
    rig.answer(vec![candidate]);
    let hit = rig.lookup().unwrap();
    assert_eq!(hit.key, rig.key(1600));
    let ticket = rig.ticket(&hit);
    rig.store.fetch(hit, ticket).unwrap();
    assert!(rig.local.0.lock().unwrap().fetches.is_empty());
    rig.pool.lock().unwrap().events.push(PoolEvent::Fetched {
        id: 1,
        result: Err(failure(refused, false)),
    });
    assert!(
        fetched(&rig.poll()).is_empty(),
        "the failure is not reported"
    );
    assert_eq!(
        rig.local.0.lock().unwrap().fetches.len(),
        1,
        "the local copy is fetched"
    );
    rig.local
        .0
        .lock()
        .unwrap()
        .events
        .push(StoreEvent::Fetched {
            key: rig.key(1600),
            result: Ok(1),
        });
    assert_eq!(fetched(&rig.poll()), [(rig.key(1600), true)]);
}

#[test]
fn without_an_answer_the_local_lookup_is_used() {
    let mut rig = rig();
    rig.local_hit(800, Tier::Nvme);
    let history = rig.history.clone();
    assert_eq!(
        rig.store
            .lookup_for(None, &history, 0, u64::MAX)
            .unwrap()
            .key,
        rig.key(800)
    );
    let pending = LookupTicket::watched(LookupState::Pending, || {});
    assert_eq!(
        rig.store
            .lookup_for(Some(&pending), &history, 0, u64::MAX)
            .unwrap()
            .key,
        rig.key(800)
    );
    // And no pool restore is started when the answer holds nothing usable.
    rig.answer(Vec::new());
    assert_eq!(rig.lookup().unwrap().key, rig.key(800));
}

// ---- failure ----

#[test]
fn a_pool_that_refuses_the_fetch_leaves_no_event_behind() {
    let mut rig = rig();
    rig.answer(vec![rig.candidate(Level::HostDram, 1, 0, 0, &[800], 0)]);
    let hit = rig.lookup().unwrap();
    let ticket = rig.ticket(&hit);
    rig.pool.lock().unwrap().reject = true;
    assert!(rig.store.fetch(hit, ticket).is_err());
    assert!(rig.poll().is_empty());
}

// ---- announcements and events ----

#[test]
fn gpu_publications_are_announced_and_their_evictions_reported() {
    let mut rig = rig();
    let key = rig.key(160);
    let checkpoint = PublishedCheckpoint {
        id: CheckpointId::new(42),
        key,
        parent: Some(rig.key(64)),
        bucket: [4; 16],
        chain_pages: vec![PageId(1), PageId(2)],
        tail_pages: 2,
        slot: StateSlotId(1),
    };
    assert_eq!(rig.store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
    assert_eq!(rig.copies(), [checkpoint.key]);
    let saved = rig.pool.lock().unwrap().saved.clone();
    assert_eq!(saved.len(), 1);
    assert_eq!((saved[0].tier, saved[0].tail_pages), (Tier::Gpu, 2));
    assert_eq!(saved[0].parent, Some(rig.key(64)));
    let mut heir = checkpoint.clone();
    heir.id = CheckpointId::new(43);
    rig.store.evicted(key, Some(&heir));
    rig.store.evicted(key, None);
    rig.store.evictions_lost();
    let state = rig.pool.lock().unwrap();
    assert_eq!(state.evicted.len(), 2);
    assert_eq!(state.evicted[0].2.as_ref().map(|h| h.key), Some(heir.key));
    assert_eq!(state.lost, 1);
}

#[test]
fn queued_lookups_do_not_trim_local_prefixes_and_export_events_reach_the_worker() {
    let mut rig = rig();
    rig.local_hit(800, Tier::Dram);
    let history = rig.history.clone();
    assert!(rig.store.begin_lookup(&history).is_some());
    let resident = rig.pool.lock().unwrap().residents[0];
    assert_eq!(resident, 0);
    let export = ExportId(5);
    rig.pool
        .lock()
        .unwrap()
        .events
        .push(PoolEvent::Export(super::super::shared::ExportRequest {
            export,
            key: rig.key(800),
            begin_tokens: 0,
            want_row: true,
            deadline: Instant::now(),
            reserved_bytes: 1,
            tier: Tier::Gpu,
        }));
    rig.pool
        .lock()
        .unwrap()
        .events
        .push(PoolEvent::ExportEnded { export });
    let events = rig.poll();
    assert!(matches!(events[0], StoreEvent::ExportRequested { export: e, .. } if e == export));
    assert!(matches!(events[1], StoreEvent::ExportEnded { export: e } if e == export));
}

#[test]
fn host_tier_exports_are_pinned_by_the_store_and_never_reach_the_worker() {
    let mut rig = rig();
    assert!(
        rig.local.0.lock().unwrap().announced,
        "the store announces its tiers"
    );
    let key = rig.key(800);
    let request = |export: u64, tier: Tier| {
        PoolEvent::Export(super::super::shared::ExportRequest {
            export: ExportId(export),
            tier,
            key,
            begin_tokens: 0,
            want_row: true,
            deadline: Instant::now(),
            reserved_bytes: 0,
        })
    };
    let events = vec![request(1, Tier::Dram), request(2, Tier::Nvme)];
    rig.pool.lock().unwrap().events.extend(events);
    assert!(rig.poll().is_empty());
    {
        let state = rig.pool.lock().unwrap();
        assert_eq!(state.host_pinned, [ExportId(1), ExportId(2)]);
    }
    assert_eq!(rig.local.0.lock().unwrap().pinned[1].0, Tier::Nvme);
    rig.local.0.lock().unwrap().refuse_pin = Some(ExportRefusal::NotFound);
    let refused = request(3, Tier::Dram);
    rig.pool.lock().unwrap().events.push(refused);
    assert!(rig.poll().is_empty());
    assert_eq!(
        rig.pool.lock().unwrap().refused,
        [(ExportId(3), ExportRefusal::NotFound)]
    );
}

#[test]
fn a_candidate_that_disagrees_with_the_history_is_dropped_and_the_others_kept() {
    let mut rig = rig();
    let mut forged = rig.candidate(Level::HostDram, 1, 0, 0, &[1200], 0);
    forged.plan.nodes[0].key.digest[0] ^= 1;
    forged.plan.key.digest[0] ^= 1;
    let good = rig.candidate(Level::HostNvme, 2, 0, 0, &[800], 0);
    rig.answer(vec![forged, good]);
    let hit = rig.lookup().unwrap();
    assert_eq!(hit.key, rig.key(800));
}

#[test]
fn a_candidate_starting_after_the_resident_prefix_never_stitches_to_a_local_prefix() {
    let mut rig = rig();
    // Nothing local and nothing resident: a candidate that begins at 800 leaves a hole.
    rig.answer(vec![rig.candidate(Level::HostNvme, 2, 800, 0, &[1600], 0)]);
    assert!(rig.lookup().is_none());
    // The local prefix is usable; the pool source still leaves a hole.
    rig.local_hit(800, Tier::Dram);
    let hit = rig.lookup().unwrap();
    assert_eq!(hit.key, rig.key(800));
}

// ---- the eviction cascade ----

impl Rig {
    fn retiring(&self, page: u64) -> Retiring {
        Retiring {
            tier: Tier::Nvme,
            key: self.key(page * PT),
            parent: (page > 1).then(|| self.key((page - 1) * PT)),
            bucket: [page as u8; 16],
            tail_pages: 1,
            row: true,
        }
    }

    fn offer(&self, records: &[Retiring]) {
        self.local
            .0
            .lock()
            .unwrap()
            .retiring
            .extend(records.iter().cloned());
    }

    fn ready(&self, ready: bool) {
        let mut pool = self.pool.lock().unwrap();
        pool.ready = ready;
        pool.events.push(PoolEvent::Status(if ready {
            crate::pool::shared::PoolStatus::Ready
        } else {
            crate::pool::shared::PoolStatus::Local
        }));
    }

    /// The ids of the puts started and not yet answered, with their keys.
    fn started(&self) -> Vec<(u64, PrefixKey)> {
        self.pool
            .lock()
            .unwrap()
            .puts
            .iter()
            .map(|(id, request)| (*id, request.key))
            .collect()
    }

    fn end(&self, id: u64, outcome: PutOutcome) {
        self.pool
            .lock()
            .unwrap()
            .events
            .push(PoolEvent::Put { id, outcome });
    }

    fn retired(&self) -> Vec<(PrefixKey, Retirement)> {
        std::mem::take(&mut self.local.0.lock().unwrap().retired)
    }
}

#[test]
fn a_retiring_record_is_put_and_released_only_when_the_manager_commits_it() {
    let mut rig = rig();
    assert!(rig.local.0.lock().unwrap().cascaded);
    rig.ready(true);
    let record = rig.retiring(5);
    rig.offer(std::slice::from_ref(&record));
    rig.poll();
    assert_eq!(rig.local.0.lock().unwrap().spill, Some(true));
    // The put describes the record and its pinned chain.
    {
        let state = rig.pool.lock().unwrap();
        let [(_, request)] = &state.puts[..] else {
            panic!("one put expected");
        };
        assert_eq!(
            (request.key, request.parent, request.bucket),
            (record.key, record.parent, record.bucket)
        );
        assert_eq!(
            (request.chain_pages, request.tail_pages, request.row),
            (10, 1, true)
        );
        assert!(matches!(request.source, PutSource::Host(_)));
    }
    assert_eq!(
        rig.local.0.lock().unwrap().pinned,
        [(Tier::Nvme, record.key)]
    );
    // Until the manager answers, the source keeps the record.
    rig.poll();
    assert!(rig.retired().is_empty());
    let (id, _) = rig.started()[0];
    rig.end(id, PutOutcome::Stored { bytes: 1 });
    rig.poll();
    assert_eq!(rig.retired(), [(record.key, Retirement::Moved)]);
}

#[test]
fn a_record_the_manager_already_holds_is_released_without_another_copy() {
    let mut rig = rig();
    rig.ready(true);
    let record = rig.retiring(3);
    rig.offer(std::slice::from_ref(&record));
    rig.poll();
    let (id, _) = rig.started()[0];
    rig.end(id, PutOutcome::Present);
    rig.poll();
    assert_eq!(rig.retired(), [(record.key, Retirement::Moved)]);
}

#[test]
fn refusals_keep_records_local_and_retry_only_after_progress() {
    for refusal in [
        PutRefusal::Budget,
        PutRefusal::Queue,
        PutRefusal::NotReady,
        PutRefusal::Oversize,
    ] {
        let mut rig = rig();
        rig.ready(true);
        let record = rig.retiring(4);
        rig.offer(std::slice::from_ref(&record));
        rig.poll();
        let (id, _) = rig.started()[0];
        rig.end(id, PutOutcome::Refused(refusal));
        rig.poll();
        assert_eq!(rig.retired(), [(record.key, Retirement::Failed)]);
        rig.poll();
        assert_eq!(rig.started().len(), 1);
        rig.offer(std::slice::from_ref(&record));
        rig.poll();
        assert_eq!(
            rig.started().len(),
            1,
            "unchanged pressure must not resubmit"
        );
        // Held through the backoff, not failed.
        assert!(rig.retired().is_empty());
        rig.ready(true);
        rig.poll();
        assert_eq!(rig.started().len(), 2);
    }
}

#[test]
fn repeated_failures_keep_the_record_local() {
    for failure in [
        PutOutcome::Failed("the session was lost".into()),
        PutOutcome::Cancelled,
        PutOutcome::Refused(PutRefusal::Io),
        PutOutcome::Refused(PutRefusal::Invalid),
        PutOutcome::Refused(PutRefusal::Inflight),
    ] {
        let mut rig = rig();
        rig.ready(true);
        let record = rig.retiring(6);
        rig.offer(std::slice::from_ref(&record));
        rig.poll();
        rig.end(rig.started()[0].0, failure);
        rig.poll();
        assert_eq!(rig.retired(), [(record.key, Retirement::Failed)]);
        // A manager event permits another selection.
        rig.ready(true);
        rig.offer(std::slice::from_ref(&record));
        rig.poll();
        let started = rig.started();
        assert_eq!(started.len(), 2);
        rig.end(started[1].0, PutOutcome::Failed("again".into()));
        rig.poll();
        assert_eq!(rig.retired(), [(record.key, Retirement::Failed)]);
    }
    // Success and later failure preserve the same retention rule.
    let mut rig = rig();
    rig.ready(true);
    let record = rig.retiring(6);
    rig.offer(std::slice::from_ref(&record));
    rig.poll();
    rig.end(rig.started()[0].0, PutOutcome::Cancelled);
    rig.poll();
    rig.ready(true);
    rig.offer(std::slice::from_ref(&record));
    rig.poll();
    rig.end(rig.started()[1].0, PutOutcome::Present);
    rig.poll();
    rig.offer(std::slice::from_ref(&record));
    rig.poll();
    rig.end(rig.started()[2].0, PutOutcome::Cancelled);
    rig.poll();
    assert_eq!(
        rig.retired(),
        [
            (record.key, Retirement::Failed),
            (record.key, Retirement::Moved),
            (record.key, Retirement::Failed)
        ]
    );
}

#[test]
fn a_source_that_cannot_be_pinned_counts_as_a_failed_write() {
    let mut rig = rig();
    rig.ready(true);
    rig.local.0.lock().unwrap().refuse_pin = Some(ExportRefusal::ChainGap);
    let record = rig.retiring(2);
    for expected in [Retirement::Failed, Retirement::Failed] {
        rig.ready(true);
        rig.offer(std::slice::from_ref(&record));
        rig.poll();
        assert_eq!(rig.retired(), [(record.key, expected)]);
    }
    assert!(rig.started().is_empty());
}

#[test]
fn an_unavailable_manager_releases_selection_and_the_tier_stops_spilling() {
    let mut rig = rig();
    // Not ready: selection is released without removing local records.
    let records: Vec<_> = (1..=3).map(|n| rig.retiring(n)).collect();
    rig.offer(&records);
    rig.poll();
    assert_eq!(rig.local.0.lock().unwrap().spill, Some(false));
    assert!(rig.started().is_empty());
    assert_eq!(
        rig.retired(),
        records
            .iter()
            .map(|r| (r.key, Retirement::Failed))
            .collect::<Vec<_>>()
    );
    // A pool whose command queue refuses the put does the same.
    rig.ready(true);
    rig.pool.lock().unwrap().put_error = true;
    rig.offer(&records[..1]);
    rig.poll();
    assert_eq!(rig.retired(), [(records[0].key, Retirement::Failed)]);
    assert_eq!(rig.local.0.lock().unwrap().spill, Some(false));
    // Manager loss releases waiting selections, while active moves remain held.
    rig.pool.lock().unwrap().put_error = false;
    rig.ready(true);
    let many: Vec<_> = (1..=10).map(|n| rig.retiring(n)).collect();
    rig.offer(&many);
    rig.poll();
    assert_eq!(rig.started().len(), 4);
    rig.ready(false);
    rig.poll();
    assert_eq!(rig.retired().len(), 6);
    assert_eq!(rig.store.cascade.in_progress(), 4);
    rig.end(rig.started()[0].0, PutOutcome::Cancelled);
    rig.poll();
    assert_eq!(rig.retired(), [(many[0].key, Retirement::Failed)]);
    assert_eq!(rig.store.cascade.in_progress(), 3);
}

#[test]
fn a_failure_during_a_burst_drops_only_the_failed_record() {
    let mut rig = rig();
    rig.ready(true);
    let records: Vec<_> = (1..=8).map(|n| rig.retiring(n)).collect();
    rig.offer(&records);
    rig.poll();
    assert_eq!(rig.started().len(), 4);
    rig.end(rig.started()[0].0, PutOutcome::Failed("lost".into()));
    rig.poll();
    assert_eq!(rig.retired(), [(records[0].key, Retirement::Failed)]);
    assert_eq!(rig.store.cascade.in_progress(), 7);
    // The backoff ends on the next manager event and the queue resumes.
    rig.ready(true);
    rig.poll();
    assert_eq!(rig.started().len(), 5);
}

#[test]
fn a_full_queue_stops_selection_and_keeps_every_record_local() {
    use super::super::cascade::{MOVES_MAX, PENDING_MAX};
    let mut rig = rig();
    rig.ready(true);
    let total = MOVES_MAX + PENDING_MAX + 3;
    let records: Vec<_> = (1..=total as u64).map(|n| rig.retiring(n)).collect();
    rig.offer(&records);
    rig.poll();
    assert_eq!(
        rig.started()
            .iter()
            .map(|(_, key)| *key)
            .collect::<Vec<_>>(),
        records[..MOVES_MAX]
            .iter()
            .map(|r| r.key)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        rig.retired(),
        records[MOVES_MAX + PENDING_MAX..]
            .iter()
            .map(|r| (r.key, Retirement::Failed))
            .collect::<Vec<_>>()
    );
    assert_eq!(rig.store.cascade.in_progress(), MOVES_MAX + PENDING_MAX);
    // As moves end, the waiting records start in order.
    let first = rig.started()[0];
    rig.end(first.0, PutOutcome::Stored { bytes: 1 });
    rig.poll();
    assert_eq!(rig.started().len(), MOVES_MAX + 1);
    assert_eq!(rig.started()[MOVES_MAX].1, records[MOVES_MAX].key);
}

#[test]
fn puts_the_cascade_did_not_start_are_left_for_their_owner() {
    let mut rig = rig();
    rig.ready(true);
    rig.end(900, PutOutcome::Present);
    rig.poll();
    let puts = rig.store.take_puts();
    assert!(matches!(puts.as_slice(), [(900, PutOutcome::Present)]));
    assert!(rig.retired().is_empty());
}

// ---- selective write-through at publication ----

impl Rig {
    fn publication(&self, end: u64) -> PublishedCheckpoint {
        let pages = (end / PT) as usize;
        PublishedCheckpoint {
            id: CheckpointId::new(5),
            key: self.key(end),
            parent: None,
            bucket: [3; 16],
            chain_pages: (0..pages as u32).map(PageId).collect(),
            tail_pages: pages,
            slot: StateSlotId(1),
        }
    }

    fn copies(&self) -> Vec<PrefixKey> {
        self.local.0.lock().unwrap().stored.clone()
    }

    fn announced_on_gpu(&self) -> usize {
        self.pool
            .lock()
            .unwrap()
            .saved
            .iter()
            .filter(|entry| entry.tier == Tier::Gpu)
            .count()
    }
}

#[test]
fn a_publication_is_announced_and_copied_to_the_first_host_tier() {
    let mut rig = rig();
    let checkpoint = rig.publication(800);
    assert_eq!(rig.store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
    assert_eq!(rig.copies(), [checkpoint.key]);
    assert_eq!(rig.announced_on_gpu(), 1, "announced once");
}

#[test]
fn publication_copies_row_only_records_even_when_recompute_is_cheaper() {
    let mut rig = rig();
    let parent = rig.publication(800);
    let mut child = rig.publication(800);
    child.key = rig.key(807);
    child.parent = Some(parent.key);
    child.tail_pages = 0;
    assert_eq!(rig.store.store(&child).unwrap(), StoreOutcome::Queued);
    assert_eq!(rig.copies(), [child.key]);
    assert_eq!(rig.store.stats().store_skipped, 0);
}

#[test]
fn a_pool_hint_never_substitutes_for_a_committed_local_copy() {
    let mut rig = rig();
    rig.answer(vec![rig.candidate(Level::HostDram, 2, 0, 0, &[800], 0)]);
    rig.lookup();
    let checkpoint = rig.publication(800);
    assert_eq!(rig.store.store(&checkpoint).unwrap(), StoreOutcome::Queued);
    assert_eq!(rig.copies(), [checkpoint.key]);
}

#[test]
fn a_full_copy_queue_tells_the_worker_to_retry_and_the_tiers_answers_pass_through() {
    let mut rig = rig();
    let checkpoint = rig.publication(800);
    rig.local.0.lock().unwrap().copies_in_flight = 16;
    assert_eq!(
        rig.store.store(&checkpoint).unwrap(),
        StoreOutcome::Retry(StoreReason::Queue)
    );
    assert!(rig.copies().is_empty());
    rig.local.0.lock().unwrap().copies_in_flight = 0;
    for inner in [
        StoreOutcome::Retry(StoreReason::Queue),
        StoreOutcome::Retry(StoreReason::Slabs),
        StoreOutcome::Retry(StoreReason::Inflight),
        StoreOutcome::Retry(StoreReason::Budget),
        StoreOutcome::Present,
        StoreOutcome::Oversize,
        StoreOutcome::Queued,
    ] {
        rig.local.0.lock().unwrap().store_outcome = Some(inner);
        assert_eq!(rig.store.store(&checkpoint).unwrap(), inner);
    }
}
