//! Rank-stream glue for Selene's record agreement and service.
mod pool;
use super::Shared;
use crate::{
    Result,
    execution::{RankStream, rendezvous},
};
pub use flare_selene::{offload::sharded::Service, shard::Sharded};
pub use pool::{PoolOffer, PoolPeers, pool_peer, pool_rank_zero};
use std::net::SocketAddr;

pub fn join_rank_zero(store: Option<&Shared>, peers: &mut [RankStream]) -> Result<Option<String>> {
    let addresses = peers
        .iter()
        .map(|peer| match peer {
            RankStream::Tcp(stream) => Ok((stream.peer_addr()?, stream.local_addr()?)),
            _ => Err(std::io::Error::other(
                "a multi-host prefix store requires TCP rank connections",
            )),
        })
        .collect::<std::io::Result<Vec<(SocketAddr, SocketAddr)>>>()
        .map_err(|error| {
            flare_selene::record::io_failure("inspect a rendezvous connection", error)
        })?;
    Ok(flare_selene::offload::sharded::join_rank_zero(
        store,
        &addresses,
        |offers| Ok(rendezvous::share_records_each(peers, offers)?),
    )?)
}

pub fn join_peer(
    store: Option<&Shared>,
    stream: &mut RankStream,
    rank: usize,
    world: usize,
) -> Result<Option<Service>> {
    Ok(flare_selene::offload::sharded::join_peer(
        store,
        rank,
        &super::transfer::Copiers,
        |record| Ok(rendezvous::exchange_record(stream, record, world)?),
    )?)
}

#[cfg(test)]
use std::net::{TcpListener, TcpStream};
#[cfg(test)]
mod tests;
