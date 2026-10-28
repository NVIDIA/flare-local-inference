use crate::execution::RankStream;
use flare_selene::{Result, record};
pub use flare_selene::pool::{PoolOffer, PoolPeers};

pub fn pool_rank_zero(offer: PoolOffer, peers: &mut [RankStream]) -> Result<Option<PoolPeers>> {
    use crate::execution::rendezvous;
    let records = rendezvous::share_records(peers, offer.encode())?;
    // Disabled ranks do not need TCP addresses or a pool connection.
    let addresses = if offer.enabled {
        peers
            .iter()
            .map(|peer| match peer {
                RankStream::Tcp(stream) => Ok((stream.peer_addr()?, stream.local_addr()?)),
                _ => Err(std::io::Error::other(
                    "a pool requires TCP rank connections",
                )),
            })
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|e| record::io_failure("inspect pool rank connections", e))?
    } else {
        Vec::new()
    };
    flare_selene::pool::pool_rank_zero(offer, &records, &addresses, |offers| {
        rendezvous::share_records_each(peers, offers)?;
        Ok(())
    })
}

pub fn pool_peer(offer: PoolOffer, stream: &mut RankStream) -> Result<Option<PoolPeers>> {
    use crate::execution::rendezvous;
    let records = rendezvous::exchange_record(stream, &offer.encode(), offer.world)?;
    let world = offer.world;
    flare_selene::pool::pool_peer(offer, &records, || {
        Ok(rendezvous::exchange_record(stream, &[], world)?)
    })
}
