use super::*;
use flare_selene::CacheGeometry;
use flare_selene::pool::{PoolPeers, PreparedPool};
use crate::offload::PoolOptions;
use flare_selene::{pool::PoolStatus, testing::TrackingRegistrar as Registrar};
use selene_manager::testing::{Manager, wait};
use std::{sync::Arc, thread};
fn unlimited() -> Budget {
    Budget {
        host: Ok(u64::MAX),
        devices: vec![(u64::MAX, false)],
    }
}
#[test]
fn peer_registration_budget_disables_the_whole_ranked_instance() {
    use super::super::shard::{PoolOffer, pool_peer, pool_rank_zero};
    use crate::execution::RankStream;
    use std::net::{TcpListener, TcpStream};

    let mut manager = Manager::new();
    manager.run();
    for rejection in 0..3 {
        let reject = rejection != 2;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let peer_stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let zero_stream = listener.accept().unwrap().0;
        let options = PoolOptions { address: manager.address(), ca: None };
        let peer_registrar = Arc::new(Registrar::default());
        let registered = peer_registrar.clone();
        let offer = |rank| PoolOffer {
            enabled: true,
            identity: [3; 32],
            layout: [4; 16],
            rank,
            world: 2,
            gpu_uuid: [rank as u8 + 1; 16],
        };
        let peer = thread::spawn(move || {
            let peers = pool_peer(offer(1), &mut RankStream::Tcp(peer_stream)).unwrap();
            let budget = match rejection {
                0 => Budget {
                    host: Ok(1),
                    ..unlimited()
                },
                1 => Budget {
                    devices: vec![(1, false)],
                    ..unlimited()
                },
                _ => unlimited(),
            };
            start_with_registrar(
                &options,
                [3; 32],
                &[flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
                vec![[2; 16]],
                0,
                budget,
                &StartupProgress::silent(),
                registered,
                peers,
                false,
            )
            .unwrap()
        });
        let peers = pool_rank_zero(offer(0), &mut [RankStream::Tcp(zero_stream)]).unwrap();
        let (mut zero, _) = start_with_registrar(
            &PoolOptions { address: manager.address(), ca: None },
            [3; 32],
            &[flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
            vec![[1; 16]],
            0,
            unlimited(),
            &StartupProgress::silent(),
            Arc::new(Registrar::default()),
            peers,
            false,
        )
        .unwrap();
        let (peer, charges) = peer.join().unwrap();
        assert!(peer.agent.is_some() && peer.client.is_none() && peer.records.is_none());
        if reject {
            assert_eq!(charges, [0]);
            assert!(!peer.approved());
            assert!(peer_registrar.calls.lock().unwrap().is_empty());
            wait("startup", || {
                matches!(
                    zero.client.as_ref().unwrap().status(),
                    PoolStatus::Disabled(_)
                )
            });
            zero.close().unwrap();
        } else {
            assert!(peer.approved());
            wait("startup", || zero.client.as_ref().unwrap().status() == PoolStatus::Ready);
            wait("startup", || peer.agent.as_ref().unwrap().status() == PoolStatus::Ready);
            assert_eq!(
                zero.identity.layout,
                flare_selene::geometry::fingerprint(&vec![
                    flare_selene::testing::geometry(
                        64,
                        &[("kv", 4096)],
                        4096
                    );
                    2
                ])
            );
            zero.close().unwrap();
        }
        let options = PoolOptions { address: manager.address(), ca: None };
        let options = super::super::PrefixCacheOffload {
            pool: Some(options),
            ..Default::default()
        };
        let mut prepared = super::super::prepare_with_inputs(
            &options,
            std::path::Path::new("model.safetensors"),
            [3; 32],
            vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
            vec![(0, 0)],
            0,
            Some((1, 2)),
        )
        .unwrap();
        let super::super::Preparation::Store(resources) = &mut prepared else {
            panic!("no pool resources")
        };
        resources.pool = Some(peer);
        let shared = prepared.into_shared().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let zero = listener.accept().unwrap().0;
        let service = thread::spawn(move || {
            super::super::shard::join_peer(Some(&shared), &mut RankStream::Tcp(stream), 1, 2)
                .unwrap()
                .unwrap()
        });
        assert!(
            super::super::shard::join_rank_zero(None, &mut [RankStream::Tcp(zero)])
                .unwrap()
                .is_none()
        );
        service.join().unwrap().join().unwrap();
    }
}

fn start_with_registrar(
    options: &PoolOptions,
    identity: [u8; 32],
    geometry: &[CacheGeometry],
    gpu_uuids: Vec<[u8; 16]>,
    registration_context: usize,
    budget: Budget,
    _: &StartupProgress,
    registrar: Arc<Registrar>,
    peers: Option<PoolPeers>,
    disk: bool,
) -> flare_selene::Result<(PreparedPool, Vec<u64>)> {
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
