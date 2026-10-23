//! Ranks of one model on separate hosts: rendezvous at rank zero and connect their RoCE transport.

use std::sync::Arc;

use flare_helios::{
    execution::{CollectiveHealth, RankStream},
    runtime_package::{ClusterRank, NativeModel},
    startup::StartupProgress,
};

type Failure = Box<dyn std::error::Error>;

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub use flare_helios::execution::roce::RoceTransport;

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
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
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let transport = transport.clone();
        CollectiveHealth(Arc::new(move || transport.health()))
    }
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = transport;
        unreachable!("no transport without RoCE support")
    }
}

/// Open this rank's transport over the fabric every rank discovered, from their endpoint lists in
/// rank order.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn open(
    owner: &NativeModel,
    fabric: &flare_helios::execution::roce_fabric::Fabric,
    endpoints: &[Vec<u8>],
    ring: bool,
) -> Result<RoceTransport, Failure> {
    use flare_helios::execution::roce_fabric;

    let spec = owner
        .roce(0)
        .ok_or("multi-host package rank has no RoCE transport")?;
    let endpoints = endpoints
        .iter()
        .map(|bytes| roce_fabric::decode(bytes))
        .collect::<flare_helios::Result<Vec<_>>>()?;
    let topology = if ring {
        fabric.ring_topology(&endpoints, spec.rank)?
    } else {
        fabric.topology(&endpoints, spec.rank)?
    };
    // SAFETY: The package restored the region at its recorded address; owner outlives the
    // transport, and no graph runs before every rank connected.
    Ok(unsafe { RoceTransport::open(spec, &topology)? })
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn connected(transport: &RoceTransport) -> String {
    format!(
        "{} rails per {} over {}",
        transport.topology().rails,
        if transport.topology().ring {
            "neighbor (experimental CPU relay ring)"
        } else {
            "peer"
        },
        transport.topology().hcas.join(",")
    )
}

/// A rank's admitted connections to the rest of its cluster, made before it loads.
pub enum Membership {
    /// Rank zero's connection to each peer, ranks 1 onward.
    Zero(Vec<RankStream>),
    /// A peer's connection to rank zero.
    Peer(RankStream),
}

impl Membership {
    /// The continuation rows and KV pages every rank allocates: the fewest any rank can back.
    pub fn agree(&mut self, local: (u64, u64)) -> flare_helios::Result<(u64, u64)> {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            use flare_helios::execution::rendezvous;

            match self {
                Membership::Zero(peers) => rendezvous::agree_capacity(peers, local),
                Membership::Peer(stream) => rendezvous::propose_capacity(stream, local),
            }
        }
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        {
            let _ = local;
            unreachable!("no membership without RoCE support")
        }
    }
}

/// Rank zero: exchange transport records with every admitted peer and wait until every rank's
/// transport is connected.
pub fn rank_zero(
    owner: &NativeModel,
    mut streams: Vec<RankStream>,
    progress: &StartupProgress,
    ring: bool,
) -> Result<Joined<Vec<RankStream>>, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::{rendezvous, roce_fabric};

        let phase = progress.phase("Connect RoCE transport", None);
        // Every rank learns every rank's devices, then plans the same routes from them.
        let fabric = roce_fabric::discover()?;
        let endpoints =
            rendezvous::share_records(&mut streams, roce_fabric::encode(&fabric.endpoints))?;
        let mut transport = open(owner, &fabric, &endpoints, ring)?;
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
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = (owner, streams, progress, ring);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}

/// A peer rank: exchange transport records with rank zero over its admitted stream, and connect.
pub fn peer(
    owner: &NativeModel,
    cluster: &ClusterRank,
    mut stream: RankStream,
    progress: &StartupProgress,
    ring: bool,
) -> Result<Joined<RankStream>, Failure> {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        use flare_helios::execution::{rendezvous, roce_fabric};

        let phase = progress.phase("Connect RoCE transport", None);
        let fabric = roce_fabric::discover()?;
        let endpoints = rendezvous::exchange_record(
            &mut stream,
            &roce_fabric::encode(&fabric.endpoints),
            cluster.world,
        )?;
        let mut transport = open(owner, &fabric, &endpoints, ring)?;
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
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        let _ = (owner, cluster, stream, progress, ring);
        Err("this flare build has no RoCE transport for multi-host packages".into())
    }
}
