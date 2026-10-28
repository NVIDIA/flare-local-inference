use flare_selene::{CacheGeometry, Result, pool::{Budget, PoolBuilder, PoolPeers, PreparedPool}};
use flare_selene::{
    History,
    pool::{PoolOptions, PoolStatus},
    testing::{TestHistory, TrackingRegistrar as Registrar},
};
use selene_manager::testing::{Manager, wait};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
fn unlimited() -> Budget {
    Budget {
        host: Ok(u64::MAX),
        devices: vec![(u64::MAX, false)],
    }
}
fn start_with_registrar(
    options: &PoolOptions,
    identity: [u8; 32],
    geometry: &[CacheGeometry],
    gpu_uuids: Vec<[u8; 16]>,
    registration_context: usize,
    budget: Budget,
    registrar: Arc<Registrar>,
    peers: Option<PoolPeers>,
    disk: bool,
) -> Result<(PreparedPool, Vec<u64>)> {
    PoolBuilder {
        options,
        identity,
        geometry,
        gpu_uuids,
        registration_context,
        budget,
        peers,
        disk,
    }
    .start(registrar)
}
struct NoCopies;
impl flare_selene::offload::CopierFactory for NoCopies {
    fn mapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
        _: &flare_selene::slab::SlabPool,
    ) -> Result<flare_selene::SharedCopier> {
        panic!("no copy engine should be constructed")
    }
    fn unmapped(
        &self,
        _: &[(usize, i32)],
        _: Vec<CacheGeometry>,
    ) -> Result<Box<dyn flare_selene::PageCopier>> {
        panic!("no copy engine should be constructed")
    }
}
fn start(manager: &Manager, registrar: Arc<Registrar>, budget: Budget) -> (PreparedPool, Vec<u64>) {
    let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
    start_with_registrar(
        &PoolOptions { address: manager.address(), ca: None },
        [3; 32],
        &geometry,
        vec![[1; 16]],
        0,
        budget,
        registrar,
        None,
        false,
    )
    .unwrap()
}
#[test]
fn over_budget_startup_registers_nothing_and_serves_local_only() {
    let mut manager = Manager::new();
    manager.run();
    for budget in [
        Budget {
            host: Ok(1),
            ..unlimited()
        },
        Budget {
            devices: vec![(1, false)],
            ..unlimited()
        },
    ] {
        let registrar = Arc::new(Registrar::default());
        let (prepared, charges) = start(&manager, registrar.clone(), budget);
        wait("startup", || {
            matches!(
                prepared.client.as_ref().unwrap().status(),
                PoolStatus::Disabled(_)
            )
        });
        let PoolStatus::Disabled(reason) = prepared.client.as_ref().unwrap().status() else {
            unreachable!()
        };
        assert!(
            reason.contains("arena") && reason.contains("staging") && reason.contains("budget"),
            "{reason}"
        );
        assert_eq!(charges, [0]);
        assert!(
            prepared
                .client
                .as_ref()
                .unwrap()
                .registered_regions()
                .is_empty()
        );
        assert!(registrar.calls.lock().unwrap().is_empty());
        let mut store = prepared.attach(None).unwrap();
        let mut history = TestHistory::new([3; 32]);
        history.append(&[7; 128], &[]).unwrap();
        assert!(store.lookup(&history, 0, u64::MAX).is_none());
        let checkpoint = flare_selene::PublishedCheckpoint {
            id: flare_selene::CheckpointId::new(1),
            key: history.key_at(128).unwrap(),
            parent: None,
            bucket: [0; 16],
            chain_pages: vec![flare_selene::PageId(1), flare_selene::PageId(2)],
            tail_pages: 2,
            slot: flare_selene::StateSlotId(1),
        };
        store.store(&checkpoint).unwrap();
        store.close().unwrap();
    }
}

#[test]
fn budget_rejection_emits_one_warning_per_instance() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "over_budget_startup_registers_nothing_and_serves_local_only",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        stderr
            .lines()
            .filter(|line| line.starts_with("warning:"))
            .count(),
        2,
        "{stderr}"
    );
}

#[test]
fn an_absent_manager_times_out_before_local_readiness_and_never_attaches_late() {
    let mut manager = Manager::new();
    let registrar = Arc::new(Registrar::default());
    let started = Instant::now();
    let (prepared, charges) = start(&manager, registrar.clone(), unlimited());
    assert!(started.elapsed() >= flare_selene::pool::STARTUP_WAIT);
    assert!(started.elapsed() < flare_selene::pool::STARTUP_WAIT + Duration::from_secs(5));
    assert_eq!(charges, [0]);
    assert!(matches!(
        prepared.client.as_ref().unwrap().status(),
        PoolStatus::Disabled(_)
    ));
    manager.run();
    assert!(matches!(
        prepared.client.as_ref().unwrap().status(),
        PoolStatus::Disabled(_)
    ));
    assert!(registrar.calls.lock().unwrap().is_empty());
    let mut store = prepared.attach(None).unwrap();
    store.poll(&mut Vec::new()).unwrap();
    store.close().unwrap();
}

#[test]
fn startup_wait_precedes_readiness_and_restart_never_registers_again() {
    let mut manager = Manager::new();
    let registrar = Arc::new(Registrar::default());
    let options = PoolOptions { address: manager.address(), ca: None };
    let registered = registrar.clone();
    let (ready, readiness) = std::sync::mpsc::channel();
    let startup = thread::spawn(move || {
        let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
        let result = start_with_registrar(
            &options,
            [3; 32],
            &geometry,
            vec![[1; 16]],
            0,
            unlimited(),
            registered,
            None,
            false,
        )
        .unwrap();
        ready.send(()).unwrap();
        result
    });
    assert!(registrar.calls.lock().unwrap().is_empty());
    manager.run();
    readiness.recv_timeout(Duration::from_secs(5)).unwrap();
    let (mut prepared, charges) = startup.join().unwrap();
    assert_eq!(charges, [(16 << 20) / 512]);
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    let changed = vec![flare_selene::testing::geometry(128, &[("kv", 4096)], 4096)];
    assert!(
        prepared
            .start_copies(&[], changed, &NoCopies)
            .unwrap_err()
            .to_string()
            .contains("layouts disagree")
    );
    wait("startup", || prepared.client.as_ref().unwrap().status() == PoolStatus::Ready);
    manager.stop();
    wait("the old session to disconnect", || prepared.client.as_ref().unwrap().status() != PoolStatus::Ready);
    manager.run();
    wait("startup", || {
        prepared.client.as_ref().unwrap().status() == PoolStatus::Ready
    });
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    manager.stop();
    manager.config.staging *= 2;
    manager.run();
    wait("startup", || {
        matches!(
            prepared.client.as_ref().unwrap().status(),
            PoolStatus::Disabled(_)
        )
    });
    assert_eq!(registrar.calls.lock().unwrap().len(), 2);
    assert_eq!(*registrar.drops.lock().unwrap(), 0);
    prepared.client.as_mut().unwrap().close().unwrap();
    assert_eq!(*registrar.drops.lock().unwrap(), 2);
    let mut store = prepared.attach(None).unwrap();
    let mut history = TestHistory::new([3; 32]);
    history.append(&[7; 128], &[]).unwrap();
    assert!(store.lookup(&history, 0, u64::MAX).is_none());
    store.poll(&mut Vec::new()).unwrap();
    store.close().unwrap();
}
