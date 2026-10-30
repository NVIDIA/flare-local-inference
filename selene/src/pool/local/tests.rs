use super::*;
use crate::store::TierStore;
use std::sync::{Arc, Mutex};

/// Records which pool hooks reached it.
#[derive(Clone, Default)]
struct Hooks(Arc<Mutex<Vec<String>>>);

impl HostStore for Hooks {
    fn lookup(&mut self, _: &dyn crate::History, _: u64, _: u64) -> Option<HostHit> {
        None
    }
    fn store(&mut self, _: &PublishedCheckpoint) -> crate::Result<StoreOutcome> {
        Ok(StoreOutcome::Present)
    }

    fn fetch(&mut self, _: HostHit, _: InboundTicket) -> crate::Result<()> {
        Ok(())
    }
    fn poll(&mut self, _: &mut Vec<crate::StoreEvent>) -> crate::Result<()> {
        Ok(())
    }
    fn stats(&self) -> StoreStats {
        StoreStats::default()
    }
    fn close(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

impl crate::store::TierStore for Hooks {
    fn drain_on_shutdown(&mut self) {
        self.0.lock().unwrap().push("drain".into());
    }
    fn pin_export(
        &mut self,
        tier: crate::pool::Tier,
        _: PrefixKey,
        _: u64,
        _: bool,
    ) -> std::result::Result<Box<dyn crate::pool::HostPin>, ExportRefusal> {
        self.0.lock().unwrap().push(format!("pin {tier:?}"));
        Err(ExportRefusal::ChainGap)
    }
    fn set_spill(&mut self, enabled: bool) {
        self.0.lock().unwrap().push(format!("spill {enabled}"));
    }
    fn take_retiring(&mut self) -> Vec<crate::Retiring> {
        self.0.lock().unwrap().push("take_retiring".into());
        Vec::new()
    }
    fn retired(&mut self, _: PrefixKey, tier: crate::pool::Tier, how: crate::Retirement) {
        self.0
            .lock()
            .unwrap()
            .push(format!("retired {tier:?} {how:?}"));
    }
}

#[test]
fn the_tiers_beneath_a_pooled_store_receive_every_pool_hook() {
    let hooks = Hooks::default();
    let mut tiers = Tiers::new(Some(Box::new(hooks.clone())));
    let key = PrefixKey {
        consumed_tokens: 64,
        digest: [1; 16],
    };
    tiers.drain_on_shutdown();
    tiers.set_spill(true);
    assert!(tiers.take_retiring().is_empty());
    tiers.retired(key, crate::pool::Tier::Dram, crate::Retirement::Moved);
    assert!(matches!(
        tiers.pin_export(crate::pool::Tier::Dram, key, 0, true),
        Err(ExportRefusal::ChainGap)
    ));
    assert_eq!(
        *hooks.0.lock().unwrap(),
        [
            "drain",
            "spill true",
            "take_retiring",
            "retired Dram Moved",
            "pin Dram"
        ]
    );
    // With no tiers there is nothing to reach.
    let mut none = Tiers::new(None);
    assert!(none.take_retiring().is_empty());
    assert!(matches!(
        none.pin_export(crate::pool::Tier::Dram, key, 0, true),
        Err(ExportRefusal::NotFound)
    ));
}
