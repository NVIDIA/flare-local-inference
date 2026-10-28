use flare_selene::{CacheGeometry, Result, offload::*, pool::PoolOptions, slab};

fn pool(address: &str) -> PoolOptions {
    PoolOptions { address: address.into(), ca: None }
}

#[test]
fn startup_releases_registration_and_charges_when_a_peer_rejects_the_pool() {
    use flare_selene::pool::{Budget, PoolBuilder};
    use flare_selene::testing::{TrackingRegistrar, geometry};
    use std::sync::Arc;

    let mut manager = selene_manager::testing::Manager::new();
    manager.run();
    let options = pool(&manager.address());
    let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
    let registrar = Arc::new(TrackingRegistrar::default());
    let mut startup = Startup::default();
    let (summary, charges) = startup
        .start_pool(
            PoolBuilder {
                options: &options,
                identity: [3; 32],
                geometry: &geometry,
                gpu_uuids: vec![[1; 16]],
                registration_context: 0,
                budget: Budget {
                    host: Ok(u64::MAX),
                    devices: vec![(u64::MAX, false), (u64::MAX, true)],
                },
                peers: None,
                disk: false,
            },
            registrar.clone(),
            |approved| {
                assert!(approved);
                assert_eq!(registrar.calls.lock().unwrap().len(), 2);
                assert_eq!(*registrar.drops.lock().unwrap(), 0);
                Ok(false)
            },
        )
        .unwrap();
    assert_eq!(summary, "local tiers only");
    assert_eq!(charges, [0, 0]);
    assert_eq!(*registrar.drops.lock().unwrap(), 2);
}

#[test]
fn startup_keeps_registration_through_attachment_and_store_drain() {
    use flare_selene::pool::{Budget, PoolBuilder};
    use flare_selene::testing::{StoppedCopier, TrackingRegistrar, geometry};
    use std::sync::Arc;
    struct Copiers;
    impl CopierFactory for Copiers {
        fn mapped(
            &self,
            _: &[(usize, i32)],
            geometry: Vec<CacheGeometry>,
            _: &slab::SlabPool,
        ) -> Result<flare_selene::SharedCopier> {
            StoppedCopier::shared(&geometry)
        }
        fn unmapped(
            &self,
            _: &[(usize, i32)],
            geometry: Vec<CacheGeometry>,
        ) -> Result<Box<dyn flare_selene::PageCopier>> {
            Ok(Box::new(StoppedCopier::new(&geometry)?))
        }
    }
    let mut manager = selene_manager::testing::Manager::new();
    manager.run();
    let options = PrefixCacheOffload {
        pool: Some(pool(&manager.address())),
        ..Default::default()
    };
    let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
    let registrar = Arc::new(TrackingRegistrar::default());
    let mut startup = Startup::default();
    assert!(startup.binding(0).is_none());
    let (_, charges) = startup
        .start_pool(
            PoolBuilder {
                options: options.pool.as_ref().unwrap(),
                identity: [3; 32],
                geometry: &geometry,
                gpu_uuids: vec![[1; 16]],
                registration_context: 0,
                budget: Budget {
                    host: Ok(u64::MAX),
                    devices: vec![(u64::MAX, false)],
                },
                peers: None,
                disk: false,
            },
            registrar.clone(),
            |approved| {
                assert!(approved);
                Ok(true)
            },
        )
        .unwrap();
    assert!(charges[0] > 0);
    let prepared = prepare(
        &options,
        std::path::Path::new("model.safetensors"),
        [3; 32],
        geometry,
        vec![(0, 0)],
        &flare_selene::testing::IdentityRegistrar,
    )
    .unwrap();
    startup.attach(prepared, &Copiers).unwrap();
    let binding = startup.binding(0).unwrap();
    assert_eq!(*registrar.drops.lock().unwrap(), 0);
    let mut store = open(binding.take().unwrap(), &Copiers).unwrap();
    startup.reset();
    assert_eq!(*registrar.drops.lock().unwrap(), 0);
    store.close().unwrap();
    assert_eq!(*registrar.drops.lock().unwrap(), 2);
}
