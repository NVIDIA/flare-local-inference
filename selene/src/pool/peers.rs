use crate::{
    Result,
    geometry::invalid,
    record,
    shard::{IDLE, Link, Reader, Writer, nonce},
};
use std::{
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream},
    time::{Duration, Instant},
};
const HELLO: u8 = 1;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

const POOL: u8 = 11;

/// Pool agreement before any rank sizes its cache. Disk agreement happens later.
#[derive(Clone)]
pub struct PoolOffer {
    pub enabled: bool,
    pub identity: [u8; 32],
    pub layout: [u8; 16],
    pub rank: usize,
    pub world: usize,
    pub gpu_uuid: [u8; 16],
}

/// Dedicated pool connections, in rank order, and the whole instance's GPU identity.
pub struct PoolPeers {
    pub links: Vec<Link>,
    pub gpu_uuids: Vec<[u8; 16]>,
    pub rank: usize,
    pub listener: Option<(TcpListener, [u8; 16])>,
}

impl PoolOffer {
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = Writer::new(POOL);
        writer.u8(self.enabled as u8);
        if self.enabled {
            writer
                .bytes(&self.identity)
                .bytes(&self.layout)
                .u32(self.rank as u32)
                .u32(self.world as u32)
                .bytes(&self.gpu_uuid);
        }
        writer.finish()[5..].to_vec()
    }

    fn agreement(&self, records: &[Vec<u8>]) -> Result<Option<Vec<[u8; 16]>>> {
        if records.len() != self.world || records.get(self.rank) != Some(&self.encode()) {
            return Err(invalid("pool rendezvous describes a different world"));
        }
        let mut uuids = Vec::new();
        for (rank, record) in records.iter().enumerate() {
            let mut reader = Reader(record);
            let enabled = reader.u8()?;
            if enabled > 1 || (enabled == 1) != self.enabled {
                return Err(invalid(format!(
                    "rank {rank} selected different prefix pool settings; configure a pool on every rank or none"
                )));
            }
            if enabled == 1 {
                if reader.take(32)? != self.identity
                    || reader.take(16)? != self.layout
                    || reader.u32()? as usize != rank
                    || reader.u32()? as usize != self.world
                {
                    return Err(invalid(format!(
                        "rank {rank} describes a different pool identity, cache layout or rank binding"
                    )));
                }
                uuids.push(reader.take(16)?.try_into().unwrap());
            }
            reader.end()?;
        }
        Ok(self.enabled.then_some(uuids))
    }
}

/// Rank zero's pool-only rendezvous, before registration and cache sizing.
pub fn pool_rank_zero(
    offer: PoolOffer,
    records: &[Vec<u8>],
    addresses: &[(SocketAddr, SocketAddr)],
    send: impl FnOnce(Vec<Vec<u8>>) -> Result<()>,
) -> Result<Option<PoolPeers>> {
    let Some(gpu_uuids) = offer.agreement(records)? else {
        return Ok(None);
    };
    let wildcard = match addresses.first() {
        Some((_, local)) if local.is_ipv4() => {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0)
        }
        Some(_) => SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0),
        None => return Err(invalid("a multi-host pool needs a peer rank")),
    };
    let listener =
        TcpListener::bind(wildcard).map_err(|e| record::io_failure("listen for pool ranks", e))?;
    let port = listener
        .local_addr()
        .map_err(|e| record::io_failure("inspect pool listener", e))?
        .port();
    let token = nonce()?;
    let mut offers = vec![Vec::new()];
    for (_, local) in addresses {
        let mut writer = Writer::new(POOL);
        writer
            .bytes(&token)
            .text(&SocketAddr::new(local.ip(), port).to_string());
        offers.push(writer.finish()[5..].to_vec());
    }
    send(offers)?;
    listener
        .set_nonblocking(true)
        .map_err(|e| record::io_failure("configure pool listener", e))?;
    let mut links: Vec<Option<Link>> = (1..offer.world).map(|_| None).collect();
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    while links.iter().any(Option::is_none) {
        if Instant::now() >= deadline {
            return Err(invalid("peer ranks did not connect their pool links"));
        }
        let (stream, address) = match listener.accept() {
            Ok(pair) => pair,
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(IDLE);
                continue;
            }
            Err(e) => return Err(record::io_failure("accept a pool rank", e)),
        };
        let mut link = Link::new(stream)?;
        let Ok(hello) = link.expect(HELLO, HANDSHAKE_TIMEOUT) else {
            continue;
        };
        let mut reader = Reader(&hello);
        let (Ok(offered), Ok(rank)) = (reader.take(16), reader.u32()) else {
            continue;
        };
        let rank = rank as usize;
        if offered != token
            || !(1..offer.world).contains(&rank)
            || links[rank - 1].is_some()
            || address.ip() != addresses[rank - 1].0.ip()
            || reader.end().is_err()
        {
            continue;
        }
        links[rank - 1] = Some(link);
    }
    Ok(Some(PoolPeers {
        links: links.into_iter().map(Option::unwrap).collect(),
        gpu_uuids,
        rank: 0,
        listener: Some((listener, token)),
    }))
}

/// A peer's matching pool rendezvous. Selene exchanges session secrets and attachment bindings
/// over this link, using the agreed whole-instance identity.
pub fn pool_peer(
    offer: PoolOffer,
    records: &[Vec<u8>],
    receive: impl FnOnce() -> Result<Vec<Vec<u8>>>,
) -> Result<Option<PoolPeers>> {
    let Some(gpu_uuids) = offer.agreement(records)? else {
        return Ok(None);
    };
    let records = receive()?;
    let mut reader = Reader(&records[0]);
    let token = reader.take(16)?;
    let address: SocketAddr = reader
        .text()?
        .parse()
        .map_err(|_| invalid("malformed pool rank address"))?;
    reader.end()?;
    let connected = TcpStream::connect_timeout(&address, CONNECT_TIMEOUT)
        .map_err(|e| record::io_failure("connect a pool rank", e))?;
    let mut link = Link::new(connected)?;
    let mut hello = Writer::new(HELLO);
    hello.bytes(token).u32(offer.rank as u32);
    link.send(&hello.finish())?;
    Ok(Some(PoolPeers {
        links: vec![link],
        gpu_uuids,
        rank: offer.rank,
        listener: None,
    }))
}

#[cfg(test)]
mod tests;
