//! A host store attached to a pool.
//!
//! [`PooledStore`] wraps the local host store and a [`PoolClient`]. Local
//! behaviour is the wrapped store's. On top of it: GPU holdings are announced
//! and their evictions reported, pool lookups run while a request queues, a
//! restore may take its pages from the pool, and exports reach the model
//! worker through `poll`.
//!
//! A restore uses one source. A failed pool copy may use this host's own copy
//! of the same endpoint, after the pool's writes have drained.

use crate::shared::Tier;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::{
    Error, ExportChain, ExportId, ExportRefusal, History, HostHit, HostStore, InboundTicket,
    PrefixKey, PublishedCheckpoint, Result, StoreEvent, StoreOutcome, StoreReason, StoreStats,
    index::Match,
    levels::{Level, Ranking},
    plan::{self, Identity, Plan, Request, SourceId},
};

use super::{
    announce::{Announcer, IndexEntry},
    cascade::Cascade,
    client::PoolClient,
    exporter::HostPin,
    lookup::{LookupState, LookupTicket},
    shared::{FetchPart, PoolEvent, PoolFetch, PutOutcome, PutRequest},
    types::{PoolCandidate, SourceRef},
};

/// What the store uses of the pool client. [`PoolClient`] is the implementation;
/// tests of the store's planning and sequencing substitute a recording one.
pub trait PoolPort {
    fn begin_lookup(&self, history: &dyn History, resident: u64) -> LookupTicket;
    fn fetch(&mut self, fetch: PoolFetch) -> Result<u64>;
    fn poll(&mut self, events: &mut Vec<PoolEvent>) -> Result<()>;
    fn saved(&self, entry: IndexEntry);
    fn evicted(&self, key: PrefixKey, tier: Tier, heir: Option<IndexEntry>);
    fn evictions_lost(&self);
    fn put(&mut self, request: PutRequest) -> Result<u64>;
    fn cancel_put(&mut self, id: u64);
    fn announcer(&self) -> Announcer;
    fn host_pinned(&mut self, export: ExportId, pin: Box<dyn HostPin>);
    fn export_pinned(&mut self, export: ExportId, chain: ExportChain);
    fn export_refused(&mut self, export: ExportId, reason: ExportRefusal);
    /// Whether a put can be accepted now: the session is open and the manager ready.
    fn ready(&self) -> bool;
    fn close(&mut self) -> Result<()>;
}

impl PoolPort for PoolClient {
    fn begin_lookup(&self, history: &dyn History, resident: u64) -> LookupTicket {
        PoolClient::begin_lookup(self, history, resident)
    }
    fn fetch(&mut self, fetch: PoolFetch) -> Result<u64> {
        PoolClient::fetch(self, fetch)
    }
    fn poll(&mut self, events: &mut Vec<PoolEvent>) -> Result<()> {
        PoolClient::poll(self, events)
    }
    fn saved(&self, entry: IndexEntry) {
        PoolClient::saved(self, entry)
    }
    fn evicted(&self, key: PrefixKey, tier: Tier, heir: Option<IndexEntry>) {
        PoolClient::evicted(self, key, tier, heir)
    }
    fn evictions_lost(&self) {
        PoolClient::evictions_lost(self)
    }
    fn put(&mut self, request: PutRequest) -> Result<u64> {
        PoolClient::put(self, request)
    }
    fn cancel_put(&mut self, id: u64) {
        PoolClient::cancel_put(self, id)
    }
    fn announcer(&self) -> Announcer {
        PoolClient::announcer(self)
    }
    fn host_pinned(&mut self, export: ExportId, pin: Box<dyn HostPin>) {
        PoolClient::host_pinned(self, export, pin)
    }
    fn export_pinned(&mut self, export: ExportId, chain: ExportChain) {
        PoolClient::export_pinned(self, export, chain)
    }
    fn export_refused(&mut self, export: ExportId, reason: ExportRefusal) {
        PoolClient::export_refused(self, export, reason)
    }
    fn ready(&self) -> bool {
        PoolClient::status(self) == super::shared::PoolStatus::Ready
    }
    fn close(&mut self) -> Result<()> {
        PoolClient::close(self)
    }
}

/// The planner's name for this instance's own tiers.
const LOCAL: SourceId = SourceId(0);
/// A source that lacked headroom is not tried again for this long.
const SKIP: Duration = Duration::from_secs(1);
/// Plans remembered between a lookup and its fetch.
const PLANS: usize = 256;
/// Copies out of the GPU in flight beyond which a publication is declined.
const COPY_QUEUE: usize = 16;
struct Planned {
    plan: Plan,
    local: Option<HostHit>,
    candidate: PoolCandidate,
}
struct Restore {
    ticket: InboundTicket,
    local: Option<HostHit>,
    id: Option<u64>,
    level: Level,
    bytes: u64,
    started: Instant,
}

pub struct PooledStore<S: crate::store::TierStore, P: PoolPort = PoolClient> {
    inner: S,
    pool: P,
    identity: Identity,
    page_tokens: u64,
    page_record_bytes: u64,
    row_record_bytes: u64,
    plans: HashMap<(PrefixKey, u64), Planned>,
    restores: HashMap<PrefixKey, Restore>,
    rates: Ranking,
    skip: HashMap<(SourceRef, Tier), Instant>,
    inner_events: Vec<StoreEvent>,
    pool_events: Vec<PoolEvent>,
    /// How the puts started through this store ended, until collected.
    puts: Vec<(u64, PutOutcome)>,
    /// Moves of the last local tier's records to the manager.
    cascade: Cascade,
    /// Restores that were wholly from the pool, which the local tiers did not count.
    pool_fetched: u64,
    /// Restores that failed in the pool part.
    pool_fetch_failed: u64,
    /// Bytes the pool parts copied into the device.
    pool_bytes_in: u64,
}

#[cfg(any(test, feature = "test-support"))]
impl<S: crate::store::TierStore> PooledStore<S, PoolClient> {
    /// Attach `inner` to the pool named by `config`. Starts the pool client's
    /// thread; the store is usable at once and shares nothing until the
    /// session is ready. `copier` is a page copier dedicated to the pool.
    pub fn connect(
        inner: S,
        config: super::client::PoolConfig,
        copier: crate::SharedCopier,
        registrar: std::sync::Arc<dyn crate::HostRegistrar + Send + Sync>,
    ) -> Result<Self> {
        let layout = copier.lock().unwrap().layout().clone();
        let identity = Identity {
            model: config.scope.model_identity,
            layout: config.scope.layout_fingerprint,
        };
        let pool = PoolClient::start(
            config,
            None,
            copier,
            registrar,
            super::Roots::System,
            super::STARTUP_WAIT,
        )?;
        let store = Self::new(inner, pool, identity, &layout);
        Ok(store)
    }
}

impl<S: crate::store::TierStore, P: PoolPort> PooledStore<S, P> {
    pub fn new(mut inner: S, pool: P, identity: Identity, layout: &crate::Layout) -> Self {
        inner.announce_to(pool.announcer());
        inner.cascade();
        Self {
            inner,
            pool,
            identity,
            page_tokens: layout.page_tokens,
            page_record_bytes: layout.page_record_bytes,
            row_record_bytes: layout.row_record_bytes,
            plans: HashMap::new(),
            restores: HashMap::new(),
            rates: Ranking::default(),
            skip: HashMap::new(),
            inner_events: Vec::new(),
            pool_events: Vec::new(),
            puts: Vec::new(),
            cascade: Cascade::new(layout),
            pool_fetched: 0,
            pool_fetch_failed: 0,
            pool_bytes_in: 0,
        }
    }

    /// The puts that ended since the last call.
    #[cfg(any(test, feature = "test-support"))]
    pub fn take_puts(&mut self) -> Vec<(u64, PutOutcome)> {
        std::mem::take(&mut self.puts)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn pool(&self) -> &P {
        &self.pool
    }

    // ---- publication -----------------------------------------------------------

    /// Queue a copy into the first host tier, bounded by the copy queue.
    fn copy_out(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        if self.inner.stats().outbound_in_flight >= COPY_QUEUE {
            return Ok(StoreOutcome::Retry(StoreReason::Queue));
        }
        self.inner.store(checkpoint)
    }

    // ---- lookup ---------------------------------------------------------------

    /// The local hit as a planner candidate.
    fn local_candidate(&self, hit: &HostHit) -> Option<plan::Candidate> {
        let pt = self.page_tokens;
        let mut nodes = Vec::with_capacity(hit.chain.nodes.len());
        let mut previous: Option<u64> = None;
        for (index, key) in hit.chain.nodes.iter().enumerate() {
            let end = key.consumed_tokens / pt;
            let tail = match previous {
                None => hit.chain.skip_pages as u64 + end.checked_sub(hit.resident_tokens / pt)?,
                Some(before) => end.checked_sub(before)?,
            };
            nodes.push(plan::ChainNode {
                key: *key,
                tail_pages: tail as usize,
            });
            previous = Some(end);
            let _ = index;
        }
        Some(plan::Candidate {
            level: match hit.tier {
                Tier::Gpu => return None,
                Tier::Dram => Level::HostDram,
                Tier::Nvme => Level::HostNvme,
            },
            source: LOCAL,
            key: hit.key,
            begin_tokens: hit.resident_tokens,
            skip_pages: hit.chain.skip_pages,
            nodes,
        })
    }

    fn usable(&self, now: Instant, candidate: &PoolCandidate) -> bool {
        self.skip
            .get(&(candidate.source.clone(), candidate.tier))
            .is_none_or(|until| now >= *until)
    }

    // ---- restores ---------------------------------------------------------------

    fn fetch_local(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        self.inner.fetch(hit, ticket)
    }

    fn fail(&mut self, key: PrefixKey, error: Error, events: &mut Vec<StoreEvent>) {
        events.push(StoreEvent::Fetched {
            key,
            result: Err(error),
        });
    }
}

impl<S: crate::store::TierStore, P: PoolPort> PooledStore<S, P> {
    fn pool_finished(
        &mut self,
        id: u64,
        result: std::result::Result<u64, super::shared::FetchFailure>,
        events: &mut Vec<StoreEvent>,
    ) {
        let Some(key) = self
            .restores
            .iter()
            .find(|(_, r)| r.id == Some(id))
            .map(|(key, _)| *key)
        else {
            return;
        };
        let mut restore = self.restores.remove(&key).expect("found");
        match result {
            Ok(bytes) => {
                if !restore.ticket.pages.is_empty() {
                    self.rates
                        .observe(restore.level, restore.bytes, restore.started.elapsed());
                }
                self.pool_fetched += 1;
                self.pool_bytes_in += bytes;
                events.push(StoreEvent::Fetched {
                    key,
                    result: Ok(bytes),
                });
            }
            Err(failure) => {
                self.pool_fetch_failed += 1;
                if failure.no_headroom
                    && let Some(source) = failure.source
                {
                    self.skip.insert(source, Instant::now() + SKIP);
                }
                if let Some(hit) = restore.local.take().filter(|hit| hit.key == key)
                    && self.fetch_local(hit, restore.ticket.clone()).is_ok()
                {
                    restore.id = None;
                    self.restores.insert(key, restore);
                    return;
                }
                self.fail(key, failure.error, events);
            }
        }
    }
}

impl<S: crate::store::TierStore, P: PoolPort> HostStore for PooledStore<S, P> {
    fn reused(&mut self, key: PrefixKey) {
        self.inner.reused(key);
    }

    fn pooled(&self) -> bool {
        true
    }
    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit> {
        self.inner.lookup(history, resident, before)
    }

    fn lookup_for(
        &mut self,
        ticket: Option<&LookupTicket>,
        history: &dyn History,
        resident: u64,
        before: u64,
    ) -> Option<HostHit> {
        let local = self.inner.lookup(history, resident, before);
        let Some(LookupState::Ready(answer)) = ticket.map(LookupTicket::state) else {
            return local;
        };
        let now = Instant::now();
        let pool: Vec<_> = answer
            .candidates
            .iter()
            .filter(|c| self.usable(now, c))
            .cloned()
            .collect();
        let mut candidates: Vec<_> = pool.iter().map(|c| c.plan.clone()).collect();
        candidates.extend(local.as_ref().and_then(|hit| self.local_candidate(hit)));
        let plan = plan::plan(
            &Request {
                history,
                identity: &self.identity,
                page_tokens: self.page_tokens,
                resident,
                before,
            },
            &candidates,
        )?;
        if plan.part.source == LOCAL {
            return local;
        }
        plan.validate(history, self.page_tokens).ok()?;
        let candidate = pool
            .into_iter()
            .find(|c| c.plan.source == plan.part.source && c.plan.key == plan.endpoint())?;
        let bytes = plan.pages() as u64 * self.page_record_bytes + self.row_record_bytes;
        let hit = HostHit {
            key: plan.endpoint(),
            consumed_tokens: plan.endpoint().consumed_tokens,
            resident_tokens: resident,
            pages: plan.pages(),
            bytes,
            tier: Tier::Dram,
            estimated_seconds: self.rates.part_seconds(plan.part.level, bytes),
            chain: Match {
                key: plan.endpoint(),
                consumed_tokens: plan.endpoint().consumed_tokens,
                resident_tokens: resident,
                nodes: plan.part.chain.clone(),
                skip_pages: plan.part.first_page,
                pages: plan.pages(),
            },
        };
        if self.plans.len() >= PLANS {
            self.plans.clear();
        }
        self.plans.insert(
            (hit.key, resident),
            Planned {
                plan,
                local,
                candidate,
            },
        );
        Some(hit)
    }

    fn begin_lookup(&mut self, history: &dyn History) -> Option<LookupTicket> {
        Some(self.pool.begin_lookup(history, 0))
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()> {
        let Some(planned) = self.plans.remove(&(hit.key, hit.resident_tokens)) else {
            return self.fetch_local(hit, ticket);
        };
        self.plans.retain(|(key, _), _| *key != hit.key);
        if ticket.key != hit.key || ticket.pages.len() != planned.plan.pages() {
            return Err(Error::new("the ticket does not match the planned restore"));
        }
        let level = planned.plan.part.level;
        let bytes = hit.bytes;
        let started = Instant::now();
        let id = self.pool.fetch(PoolFetch {
            id: 0,
            page_tokens: self.page_tokens,
            part: FetchPart {
                candidate: planned.candidate,
                part: planned.plan.part,
                pages: ticket.pages.clone(),
                slot: Some(ticket.slot),
            },
        })?;
        self.restores.insert(
            hit.key,
            Restore {
                ticket,
                local: planned.local,
                id: Some(id),
                level,
                bytes,
                started,
            },
        );
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()> {
        let now = Instant::now();
        self.skip.retain(|_, until| *until > now);
        let mut inner_events = std::mem::take(&mut self.inner_events);
        let inner = self.inner.poll(&mut inner_events);
        for event in inner_events.drain(..) {
            if let StoreEvent::Fetched { key, .. } = &event {
                if self.restores.get(key).is_some_and(|r| r.id.is_none()) {
                    self.restores.remove(key);
                }
            }
            events.push(event);
        }
        self.inner_events = inner_events;
        let mut pool_events = std::mem::take(&mut self.pool_events);
        let pool = self.pool.poll(&mut pool_events);
        if !pool_events.is_empty() {
            self.cascade.progress();
        }
        for event in pool_events.drain(..) {
            match event {
                PoolEvent::Export(request) if request.tier == Tier::Gpu => {
                    events.push(StoreEvent::ExportRequested {
                        export: request.export,
                        key: request.key,
                        begin_tokens: request.begin_tokens,
                        want_row: request.want_row,
                        reserved_bytes: request.reserved_bytes,
                    })
                }
                PoolEvent::Export(request) => match self.inner.pin_export(
                    request.tier,
                    request.key,
                    request.begin_tokens,
                    request.want_row,
                ) {
                    Ok(pin) => self.pool.host_pinned(request.export, pin),
                    Err(reason) => self.pool.export_refused(request.export, reason),
                },
                PoolEvent::ExportEnded { export } => {
                    events.push(StoreEvent::ExportEnded { export })
                }
                PoolEvent::Fetched { id, result } => self.pool_finished(id, result, events),
                PoolEvent::Put { id, outcome } => {
                    if !self.cascade.ended(&mut self.inner, id, &outcome) {
                        self.puts.push((id, outcome));
                    }
                }
                PoolEvent::Status(_) => {}
            }
        }
        self.pool_events = pool_events;
        self.cascade.pump(&mut self.inner, &mut self.pool);
        inner.and(pool)
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome> {
        // Announce the GPU holding before starting its host copy.
        PoolPort::saved(
            &self.pool,
            IndexEntry {
                key: checkpoint.key,
                parent: checkpoint.parent,
                bucket: checkpoint.bucket,
                tail_pages: checkpoint.tail_pages as u32,
                row: true,
                tier: Tier::Gpu,
                bytes: checkpoint.tail_pages as u64 * self.page_record_bytes
                    + self.row_record_bytes,
            },
        );
        self.copy_out(checkpoint)
    }

    fn stats(&self) -> StoreStats {
        let mut stats = self.inner.stats();
        stats.fetched += self.pool_fetched;
        stats.fetch_failed += self.pool_fetch_failed;
        stats.bytes_in += self.pool_bytes_in;
        stats
    }

    fn close(&mut self) -> Result<()> {
        self.inner.drain_on_shutdown();
        let pool = PoolPort::close(&mut self.pool);
        let inner = self.inner.close();
        pool.and(inner)
    }

    fn export_pinned(&mut self, export: ExportId, chain: ExportChain) -> Result<()> {
        PoolPort::export_pinned(&mut self.pool, export, chain);
        Ok(())
    }

    fn export_refused(&mut self, export: ExportId, reason: ExportRefusal) {
        PoolPort::export_refused(&mut self.pool, export, reason);
    }

    fn evictions_lost(&mut self) {
        PoolPort::evictions_lost(&self.pool);
    }

    fn evicted(&mut self, key: PrefixKey, heir: Option<&PublishedCheckpoint>) {
        PoolPort::evicted(
            &self.pool,
            key,
            Tier::Gpu,
            heir.map(|heir| IndexEntry {
                key: heir.key,
                parent: heir.parent,
                bucket: heir.bucket,
                tail_pages: heir.tail_pages as u32,
                row: true,
                tier: Tier::Gpu,
                bytes: heir.tail_pages as u64 * self.page_record_bytes + self.row_record_bytes,
            }),
        );
    }
}

#[cfg(test)]
mod tests;
