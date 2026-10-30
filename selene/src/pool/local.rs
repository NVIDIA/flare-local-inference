use crate::store::TierStore;
use crate::{
    ExportRefusal, HostHit, HostStore, InboundTicket, PrefixKey, PublishedCheckpoint, StoreOutcome,
    StoreStats,
};

/// The local tiers beneath a pooled store. Without any, nothing is retained
/// locally and every restore comes from the pool.
pub(crate) struct Tiers(Option<Box<dyn TierStore>>);

impl Tiers {
    pub(crate) fn new(tiers: Option<Box<dyn TierStore>>) -> Self {
        Self(tiers)
    }
}

impl HostStore for Tiers {
    fn reused(&mut self, key: PrefixKey) {
        if let Some(tiers) = &mut self.0 {
            tiers.reused(key);
        }
    }

    fn lookup(
        &mut self,
        history: &dyn crate::History,
        resident: u64,
        before: u64,
    ) -> Option<HostHit> {
        self.0.as_mut()?.lookup(history, resident, before)
    }

    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> crate::Result<StoreOutcome> {
        match &mut self.0 {
            Some(tiers) => tiers.store(checkpoint),
            None => Ok(StoreOutcome::Present),
        }
    }

    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> crate::Result<()> {
        match &mut self.0 {
            Some(tiers) => tiers.fetch(hit, ticket),
            None => Err(crate::Error::new("no local tier holds this record")),
        }
    }

    fn poll(&mut self, events: &mut Vec<crate::StoreEvent>) -> crate::Result<()> {
        match &mut self.0 {
            Some(tiers) => tiers.poll(events),
            None => Ok(()),
        }
    }

    fn stats(&self) -> StoreStats {
        self.0
            .as_ref()
            .map(|tiers| tiers.stats())
            .unwrap_or_default()
    }

    fn close(&mut self) -> crate::Result<()> {
        match &mut self.0 {
            Some(tiers) => tiers.close(),
            None => Ok(()),
        }
    }

    // The hooks a pooled store drives on its tiers. Each must reach them: a default here
    // would leave the tiers announcing nothing, never exporting and never spilling.
}

impl crate::store::TierStore for Tiers {
    fn drain_on_shutdown(&mut self) {
        if let Some(tiers) = &mut self.0 {
            tiers.drain_on_shutdown();
        }
    }
    fn announce_to(&mut self, announcer: crate::pool::Announcer) {
        if let Some(tiers) = &mut self.0 {
            tiers.announce_to(announcer);
        }
    }
    fn pin_export(
        &mut self,
        tier: crate::pool::Tier,
        key: PrefixKey,
        begin_tokens: u64,
        want_row: bool,
    ) -> std::result::Result<Box<dyn crate::pool::HostPin>, ExportRefusal> {
        match &mut self.0 {
            Some(tiers) => tiers.pin_export(tier, key, begin_tokens, want_row),
            None => Err(ExportRefusal::NotFound),
        }
    }
    fn set_spill(&mut self, enabled: bool) {
        if let Some(tiers) = &mut self.0 {
            tiers.set_spill(enabled);
        }
    }
    fn take_retiring(&mut self) -> Vec<crate::Retiring> {
        self.0
            .as_mut()
            .map(|tiers| tiers.take_retiring())
            .unwrap_or_default()
    }
    fn retired(&mut self, key: PrefixKey, tier: crate::pool::Tier, how: crate::Retirement) {
        if let Some(tiers) = &mut self.0 {
            tiers.retired(key, tier, how);
        }
    }
}

#[cfg(test)]
mod tests;
