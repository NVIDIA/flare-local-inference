/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ranks of one model on separate hosts: rendezvous at rank zero and connect their RoCE transport.

pub use flare_helios::execution::roce_fabric::Shape;
use std::sync::Arc;

use flare_helios::{
    execution::{CollectiveHealth, RankStream},
    runtime_package::{ClusterRank, NativeModel},
    startup::StartupProgress,
};

type Failure = Box<dyn std::error::Error>;

const NO_TRANSPORT: &str = "this flare build has no RoCE transport for multi-host packages; on Linux x86_64, rebuild with --features roce";

/// Reject unsupported serving before loading resources or sizing caches.
pub fn require_transport() -> Result<(), Failure> {
    if cfg!(all(
        target_os = "linux",
        any(target_arch = "aarch64", feature = "roce")
    )) {
        Ok(())
    } else {
        Err(NO_TRANSPORT.into())
    }
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
pub use flare_helios::execution::roce::RoceTransport;

#[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
pub struct RoceTransport;

/// A rank's connection to the rest of its cluster; the transport must outlive every replay.
pub struct Joined<S> {
    pub streams: S,
    pub transport: Arc<RoceTransport>,
    /// A peer's host prefix record service, which rank zero drives until it closes its store.
    pub records: Option<flare_helios::offload::shard::Service>,
}

/// The check every completed frame passes: a failed transport poisons the rank.
pub fn health(transport: &Arc<RoceTransport>) -> CollectiveHealth {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        let transport = transport.clone();
        CollectiveHealth(Arc::new(move || transport.health()))
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = transport;
        unreachable!("no transport without RoCE support")
    }
}

/// Open this rank's transport over the fabric every rank discovered, from their endpoint lists in
/// rank order.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn open(
    owner: &NativeModel,
    fabric: &flare_helios::execution::roce_fabric::Fabric,
    endpoints: &[Vec<u8>],
    shape: Shape,
) -> Result<RoceTransport, Failure> {
    use flare_helios::execution::roce_fabric;

    let spec = owner
        .roce(0)
        .ok_or("multi-host package rank has no RoCE transport")?;
    let endpoints = endpoints
        .iter()
        .map(|bytes| roce_fabric::decode(bytes))
        .collect::<flare_helios::Result<Vec<_>>>()?;
    let topology = fabric.topology(&endpoints, spec.rank, shape)?;
    // SAFETY: The package restored the region at its recorded address; owner outlives the
    // transport, and no graph runs before every rank connected.
    Ok(unsafe { RoceTransport::open(spec, &topology)? })
}

#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn connected(transport: &RoceTransport) -> String {
    format!(
        "{} rails per {} over {}",
        transport.topology().rails,
        match transport.topology().shape {
            Shape::Mesh => "peer",
            Shape::Ring => "neighbor (experimental CPU relay ring)",
            Shape::Chain => "neighbor (experimental CPU relay chain)",
        },
        transport.topology().hcas.join(",")
    )
}

/// The transport shape a rank was started with, as the ranks exchange it before planning routes.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn shape_record(shape: Shape) -> Vec<u8> {
    shape.name().as_bytes().to_vec()
}

/// Every rank must select the same shape: a chain rank plans no route to the far end that a ring
/// rank would wait for.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
fn agree_shape(records: &[Vec<u8>], shape: Shape) -> Result<(), Failure> {
    let local = shape_record(shape);
    if let Some(rank) = records.iter().position(|record| *record != local) {
        return Err(format!(
            "rank {rank} selected RoCE transport shape {} where this rank selected {}",
            String::from_utf8_lossy(&records[rank]),
            String::from_utf8_lossy(&local)
        )
        .into());
    }
    Ok(())
}

/// A rank's admitted connections to the rest of its cluster, made before it loads.
pub enum Membership {
    /// Rank zero's connection to each peer, ranks 1 onward.
    Zero(Vec<RankStream>),
    /// A peer's connection to rank zero.
    Peer(RankStream),
}

impl flare_helios::runtime_package::CacheAgreement for Membership {
    fn caches(&mut self, local: (u64, u64)) -> flare_helios::Result<(u64, u64)> {
        use flare_helios::execution::rendezvous;
        match self {
            Membership::Zero(peers) => rendezvous::agree_capacity(peers, local),
            Membership::Peer(stream) => rendezvous::propose_capacity(stream, local),
        }
    }

    fn pool(
        &mut self,
        offer: flare_helios::offload::shard::PoolOffer,
    ) -> flare_helios::Result<Option<flare_helios::offload::shard::PoolPeers>> {
        use flare_helios::offload::shard;
        match self {
            Membership::Zero(peers) => Ok(shard::pool_rank_zero(offer, peers)?),
            Membership::Peer(stream) => Ok(shard::pool_peer(offer, stream)?),
        }
    }

    fn pool_approved(&mut self, approved: bool, world: usize) -> flare_helios::Result<bool> {
        use flare_helios::execution::rendezvous;
        let records = match self {
            Membership::Zero(peers) => rendezvous::share_records(peers, vec![approved as u8])?,
            Membership::Peer(stream) => {
                rendezvous::exchange_record(stream, &[approved as u8], world)?
            }
        };
        Ok(records.iter().all(|record| record.as_slice() == [1]))
    }
}

/// Rank zero: exchange transport records with every admitted peer and wait until every rank's
/// transport is connected.
pub fn rank_zero(
    owner: &NativeModel,
    streams: Vec<RankStream>,
    progress: &StartupProgress,
    shape: Shape,
) -> Result<Joined<Vec<RankStream>>, Failure> {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::{rendezvous, roce_fabric};
        let mut streams = streams;

        let phase = progress.phase("Connect RoCE transport", None);
        // Every rank learns every rank's devices and the shape, then plans the same routes from them.
        let fabric = roce_fabric::discover()?;
        let shapes = rendezvous::share_records(&mut streams, shape_record(shape))?;
        agree_shape(&shapes, shape)?;
        let endpoints =
            rendezvous::share_records(&mut streams, roce_fabric::encode(&fabric.endpoints))?;
        let mut transport = open(owner, &fabric, &endpoints, shape)?;
        let records = rendezvous::share_records(&mut streams, transport.local_record()?)?;
        transport.connect(&records)?;
        rendezvous::await_connected(&mut streams)?;
        phase.finish(connected(&transport));
        let phase = progress.phase("Join host prefix stores", None);
        match flare_helios::offload::shard::join_rank_zero(owner.offload(), &mut streams)? {
            Some(summary) => phase.finish(summary),
            None => phase.finish("none"),
        }
        Ok(Joined {
            streams,
            transport: Arc::new(transport),
            records: None,
        })
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = (owner, streams, progress, shape);
        Err(NO_TRANSPORT.into())
    }
}

/// A peer rank: exchange transport records with rank zero over its admitted stream, and connect.
pub fn peer(
    owner: &NativeModel,
    cluster: &ClusterRank,
    stream: RankStream,
    progress: &StartupProgress,
    shape: Shape,
) -> Result<Joined<RankStream>, Failure> {
    #[cfg(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))]
    {
        use flare_helios::execution::{rendezvous, roce_fabric};
        let mut stream = stream;

        let phase = progress.phase("Connect RoCE transport", None);
        let fabric = roce_fabric::discover()?;
        let shapes = rendezvous::exchange_record(&mut stream, &shape_record(shape), cluster.world)?;
        agree_shape(&shapes, shape)?;
        let endpoints = rendezvous::exchange_record(
            &mut stream,
            &roce_fabric::encode(&fabric.endpoints),
            cluster.world,
        )?;
        let mut transport = open(owner, &fabric, &endpoints, shape)?;
        let records =
            rendezvous::exchange_record(&mut stream, &transport.local_record()?, cluster.world)?;
        transport.connect(&records)?;
        rendezvous::report_connected(&mut stream)?;
        phase.finish(connected(&transport));
        let phase = progress.phase("Join host prefix stores", None);
        let records = flare_helios::offload::shard::join_peer(
            owner.offload(),
            &mut stream,
            cluster.rank,
            cluster.world,
        )?;
        phase.finish(match &records {
            Some(service) => service.summary.clone(),
            None => "none".to_owned(),
        });
        Ok(Joined {
            streams: stream,
            transport: Arc::new(transport),
            records,
        })
    }
    #[cfg(not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce"))))]
    {
        let _ = (owner, cluster, stream, progress, shape);
        Err(NO_TRANSPORT.into())
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use flare_helios::runtime_package::CacheAgreement;
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn pool_approval_is_unanimous_and_waits_for_every_rank() {
        for peer_approved in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let zero = listener.accept().unwrap().0;
            let (ready, started) = std::sync::mpsc::channel();
            let zero = std::thread::spawn(move || {
                let approved = Membership::Zero(vec![RankStream::Tcp(zero)])
                    .pool_approved(true, 2)
                    .unwrap();
                ready.send(approved).unwrap();
                approved
            });
            assert!(
                started
                    .recv_timeout(std::time::Duration::from_millis(50))
                    .is_err()
            );
            let approved = Membership::Peer(RankStream::Tcp(peer))
                .pool_approved(peer_approved, 2)
                .unwrap();
            assert_eq!(approved, peer_approved);
            assert_eq!(started.recv().unwrap(), peer_approved);
            assert_eq!(zero.join().unwrap(), peer_approved);
        }
    }
}

#[cfg(all(
    test,
    not(all(target_os = "linux", any(target_arch = "aarch64", feature = "roce")))
))]
mod transport_tests {
    use super::*;

    #[test]
    fn unsupported_build_rejects_multi_host_serving() {
        assert!(
            require_transport()
                .unwrap_err()
                .to_string()
                .contains("--features roce")
        );
    }
}
