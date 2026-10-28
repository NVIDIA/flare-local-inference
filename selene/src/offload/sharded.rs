//! Disk records of a model whose ranks run as separate processes on separate hosts.
//!
//! Each host keeps the record files of its own rank in its own record directory. A file holds
//! one rank's planes and row in the ordinary record format, so every host's files have the same
//! names and lengths. Rank zero's [`DiskStore`](super::disk_store::DiskStore) indexes, sizes,
//! evicts, and plans every record; [`Sharded`](crate::shard::Sharded) runs each of its transport jobs on rank zero's own
//! [`Staged`] transport and on every peer's, through a dedicated connection to each peer, and
//! reports the job only after every rank finished it. A fetch therefore completes only once every
//! rank's GPU holds its pages and row. A store that some rank failed removes the file from the
//! ranks that wrote it before the failure is reported, so no host keeps a record the index lacks.
//!
//! Before serving, the ranks agree on the records they scanned: rank zero keeps a record only
//! when every rank holds a file with the same header and keeps its row only when every rank
//! holds the row. Every rank then deletes or truncates its files to match, so the index rank zero
//! builds describes every host's directory.

use super::{Prepared, Shared};
use crate::{
    PrefixKey, Result,
    disk::DiskThread,
    disk_store::Staged,
    geometry::invalid,
    record,
    shard::{
        AGREED, HANDSHAKE_TIMEOUT, HELLO, IDLE, Link, Peer, Reader, SUMMARY, Writer, agree,
        decode_agreed, decode_summary, encode_agreed, nonce, serve, summary,
    },
};
use std::{
    collections::HashMap,
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream},
    sync::mpsc::Receiver,
    thread::JoinHandle,
    time::{Duration, Instant},
};
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);
const JOB_TIMEOUT: Duration = Duration::from_secs(120);
// ---------------------------------------------------------------------------------------------
// A peer's record service

/// A peer rank's record service: its staged transport, driven by rank zero over one connection.
pub struct Service {
    thread: Option<JoinHandle<Result<()>>>,
    finished: Option<Receiver<Result<()>>>,
    pool: Option<crate::pool::PreparedPool>,
    /// What this rank joined with, for the startup report.
    pub summary: String,
}

impl Service {
    /// Wait for rank zero to close the service, or for its connection to end.
    ///
    /// A rank zero that stays alive without closing cannot hold this rank's
    /// shutdown open: after [`JOB_TIMEOUT`] the service is reported as not
    /// drained, and its resources are retained by the caller.
    pub fn join(self) -> Result<()> {
        self.join_timeout(JOB_TIMEOUT)
    }

    fn join_timeout(mut self, timeout: Duration) -> Result<()> {
        // The agent drains before the record service can release its directory and transfers.
        if let Some(pool) = &mut self.pool {
            if let Err(error) = pool.close() {
                std::mem::forget(self);
                return Err(invalid(error.to_string()));
            }
        }
        let result = match &self.finished {
            None => Ok(()),
            Some(finished) => match finished.recv_timeout(timeout) {
                Ok(result) => result,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    std::mem::forget(self);
                    return Err(invalid(
                        "rank zero did not close the prefix record service in time",
                    ));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    Err(invalid("the prefix record service ended without a result"))
                }
            },
        };
        if let Some(thread) = self.thread.take() {
            // The service reports its result before it returns.
            let _ = thread.join();
        }
        result
    }
}

/// Delete the files of records outside the agreement and truncate rows it does not keep.
fn retain(prepared: &mut Prepared, agreed: &HashMap<PrefixKey, bool>) -> Result<()> {
    let mut kept = Vec::with_capacity(agreed.len());
    for mut record in std::mem::take(&mut prepared.scanned) {
        match agreed.get(&record.header.key) {
            None => match std::fs::remove_file(&record.path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(record::io_failure(
                        format!("delete record file {}", record.path.display()),
                        error,
                    ));
                }
            },
            Some(&row) => {
                if record.row && !row {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&record.path)
                        .and_then(|file| file.set_len(record.header.row_segment))
                        .map_err(|error| {
                            record::io_failure(
                                format!("truncate record file {}", record.path.display()),
                                error,
                            )
                        })?;
                    record.row = false;
                    record.bytes = record.header.row_segment;
                }
                kept.push(record);
            }
        }
    }
    prepared
        .disk_budget
        .set_checkpoint(kept.iter().map(|record| record.bytes).sum());
    prepared.scanned = kept;
    Ok(())
}

/// Whether this rank serves a disk store its peers can mirror; a memory tier keeps records in
/// one process's memory, which other hosts cannot reach.
fn mirrored(prepared: &Prepared) -> Result<bool> {
    if prepared.memory_pool.is_some() {
        return Err(invalid(
            "a multi-host package keeps host prefix records on disk only; set --prefix-cache-memory 0",
        ));
    }
    Ok(prepared.directory_lock.is_some())
}

/// Rank zero, after its peers joined and before serving: agree with every peer on whether a disk
/// store exists, connect a record connection to each, and agree on the scanned records. Returns
/// a summary for the startup report when the ranks share a store.
pub fn join_rank_zero(
    store: Option<&Shared>,
    addresses: &[(SocketAddr, SocketAddr)],
    exchange: impl FnOnce(Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>>,
) -> Result<Option<String>> {
    let mut prepared = store.map(Shared::lock);
    let mut prepared = prepared.as_mut().and_then(|guard| guard.as_mut());
    let pool_listener = prepared
        .as_mut()
        .and_then(|prepared| prepared.pool.as_mut())
        .and_then(crate::pool::PreparedPool::take_listener);
    let enabled = match &prepared {
        Some(prepared) => mirrored(prepared)?,
        None => false,
    };
    // Each peer's own view of this rank: the local address of the rendezvous
    // connection that peer opened, so a multi-homed rank zero is reachable
    // from every peer on the interface that peer already uses.
    let remote: Vec<SocketAddr> = addresses.iter().map(|(remote, _)| *remote).collect();
    let local: Vec<SocketAddr> = addresses.iter().map(|(_, local)| *local).collect();
    let (pool_listener, token) = match pool_listener {
        Some((listener, token)) => (Some(listener), token),
        None => (None, nonce()?),
    };
    let listener = if enabled && pool_listener.is_none() {
        let wildcard = match local.first().map(|local| local.is_ipv4()) {
            Some(true) => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            Some(false) => SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0),
            None => return Err(invalid("a multi-host prefix store needs a peer rank")),
        };
        Some(
            TcpListener::bind(wildcard)
                .map_err(|error| record::io_failure("listen for prefix record peers", error))?,
        )
    } else {
        pool_listener
    };
    let listener = listener.filter(|_| enabled);
    let world = addresses.len() + 1;
    let layout = prepared
        .as_ref()
        .map(|prepared| crate::geometry::fingerprint(&prepared.geometry));
    let limit = prepared
        .as_ref()
        .map(|prepared| prepared.resolved.disk_limit);
    let mut offers = vec![Vec::new()];
    for peer_local in &local {
        let mut offer = vec![enabled as u8];
        if let Some(listener) = &listener {
            let port = listener
                .local_addr()
                .map_err(|error| record::io_failure("inspect the prefix record listener", error))?
                .port();
            let address = SocketAddr::new(peer_local.ip(), port);
            offer.extend(token);
            offer.extend(layout.unwrap_or_default());
            offer.extend(limit.unwrap_or_default().to_le_bytes());
            offer.extend((address.to_string().len() as u32).to_le_bytes());
            offer.extend(address.to_string().as_bytes());
        }
        offers.push(offer);
    }
    let replies = exchange(offers)?;
    let mut peer_enabled = Vec::with_capacity(addresses.len());
    let mut limits = Vec::with_capacity(addresses.len());
    for (index, reply) in replies[1..].iter().enumerate() {
        let enabled = reply.first() == Some(&1);
        peer_enabled.push(enabled);
        if enabled {
            let mut reader = Reader(&reply[1..]);
            let peer_layout = reader.take(16)?;
            let peer_limit = reader.u64()?;
            reader.end()?;
            if let Some(layout) = layout
                && peer_layout != layout
            {
                return Err(invalid(format!(
                    "rank {} describes a different cache layout than rank zero; every rank of a multi-host package must have the same plane and row sizes",
                    index + 1
                )));
            }
            limits.push(peer_limit);
        }
    }
    if let Some(rank) = peer_enabled.iter().position(|&peer| peer != enabled) {
        return Err(invalid(format!(
            "rank {} {} a disk prefix store and rank zero {}; give every rank the same --prefix-cache-disk settings",
            rank + 1,
            if enabled { "lacks" } else { "has" },
            if enabled { "has one" } else { "does not" },
        )));
    }
    let (Some(listener), Some(prepared)) = (listener, prepared) else {
        return Ok(None);
    };
    // The limit applies to every host's directory, so the smallest one bounds
    // the records every host can hold.
    if let Some(minimum) = limits.iter().copied().min() {
        prepared.resolved.disk_limit = prepared.resolved.disk_limit.min(minimum);
        prepared.disk_budget.set_limit(prepared.resolved.disk_limit);
    }
    let mut links: Vec<Option<Link>> = (1..world).map(|_| None).collect();
    listener
        .set_nonblocking(true)
        .map_err(|error| record::io_failure("configure the prefix record listener", error))?;
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    while links.iter().any(Option::is_none) {
        if Instant::now() >= deadline {
            return Err(invalid(
                "peer ranks did not connect their prefix record connections",
            ));
        }
        let stream = match listener.accept() {
            Ok((stream, address)) => (stream, address),
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(IDLE);
                continue;
            }
            Err(error) => {
                return Err(record::io_failure("accept a prefix record peer", error));
            }
        };
        let (stream, address) = stream;
        let mut link = Link::new(stream)?;
        let Ok(hello) = link.expect(HELLO, HANDSHAKE_TIMEOUT) else {
            continue;
        };
        let mut reader = Reader(&hello);
        let (Ok(offered), Ok(rank), Ok(peer_layout)) =
            (reader.take(16), reader.u32(), reader.take(16))
        else {
            continue;
        };
        let rank = rank as usize;
        if offered != token
            || !(1..world).contains(&rank)
            || links[rank - 1].is_some()
            || address.ip() != remote[rank - 1].ip()
            || Some(peer_layout) != layout.as_ref().map(|layout| layout.as_slice())
        {
            continue;
        }
        links[rank - 1] = Some(link);
    }
    let mut links: Vec<Link> = links.into_iter().map(Option::unwrap).collect();
    let summaries = links
        .iter_mut()
        .map(|link| decode_summary(&link.expect(SUMMARY, REPLY_TIMEOUT)?))
        .collect::<Result<Vec<_>>>()?;
    let agreed = agree(&prepared.scanned, &summaries)?;
    let frame = encode_agreed(&agreed);
    for link in &mut links {
        link.send(&frame)?;
    }
    let scanned = prepared.scanned.len();
    retain(prepared, &agreed)?;
    prepared.shards = links;
    Ok(Some(format!(
        "{} of {scanned} records on every rank, limit {}",
        prepared.scanned.len(),
        super::gibibytes(prepared.resolved.disk_limit)
    )))
}

/// A peer, at the point where rank zero runs [`join_rank_zero`]: connect this rank's record
/// connection, apply the scan agreement, and start the service rank zero drives.
pub fn join_peer(
    store: Option<&Shared>,
    rank: usize,
    factory: &dyn super::CopierFactory,
    exchange: impl FnOnce(&[u8]) -> Result<Vec<Vec<u8>>>,
) -> Result<Option<Service>> {
    // Read what the offer needs, then release the guard: taking the prepared
    // resources below locks the same shared cell.
    let (enabled, layout, limit) = match store.map(Shared::lock).as_ref().and_then(|g| g.as_ref()) {
        Some(prepared) => (
            mirrored(prepared)?,
            crate::geometry::fingerprint(&prepared.geometry),
            prepared.resolved.disk_limit,
        ),
        None => (false, [0; 16], 0),
    };
    let mut record = vec![enabled as u8];
    if enabled {
        record.extend(layout);
        record.extend(limit.to_le_bytes());
    }
    let records = exchange(&record)?;
    let offer = &records[0];
    let zero = offer.first() == Some(&1);
    if zero != enabled {
        return Err(invalid(format!(
            "rank zero {} a disk prefix store and rank {rank} {}; give every rank the same --prefix-cache-disk settings",
            if zero { "has" } else { "lacks" },
            if enabled { "has one" } else { "does not" },
        )));
    }
    if !enabled {
        let pool = store
            .and_then(Shared::take)
            .and_then(|mut prepared| prepared.pool.take());
        return Ok(pool.map(|pool| Service {
            thread: None,
            finished: None,
            pool: Some(pool),
            summary: "pool rank agent".into(),
        }));
    }
    if records[1..].iter().any(|record| record.first() != Some(&1)) {
        return Err(invalid(
            "another peer rank lacks a disk prefix store; give every rank the same --prefix-cache-disk settings",
        ));
    }
    let mut reader = Reader(&offer[1..]);
    let token = reader.take(16)?;
    let zero_layout = reader.take(16)?;
    let zero_limit = reader.u64()?;
    let length = reader.u32()? as usize;
    let address = std::str::from_utf8(reader.take(length)?)
        .map_err(|_| invalid("rank zero's prefix record address is malformed"))?;
    reader.end()?;
    if zero_layout != layout {
        return Err(invalid(format!(
            "rank {rank} describes a different cache layout than rank zero; every rank of a multi-host package must have the same plane and row sizes"
        )));
    }
    let limit = zero_limit.min(limit);
    let connected = TcpStream::connect_timeout(
        &address
            .parse()
            .map_err(|_| invalid("rank zero's prefix record address is malformed"))?,
        CONNECT_TIMEOUT,
    )
    .map_err(|error| {
        record::io_failure(
            format!("connect to rank zero's prefix records at {address}"),
            error,
        )
    })?;
    let mut link = Link::new(connected)?;
    let mut hello = Writer::new(HELLO);
    hello.bytes(token).u32(rank as u32).bytes(&layout);
    link.send(&hello.finish())?;
    let shared = store.ok_or_else(|| invalid("a peer's prefix store vanished"))?;
    let mut prepared = shared
        .take()
        .ok_or_else(|| invalid("a peer's prefix store was already opened"))?;
    // The smallest limit among the ranks bounds what every host can hold.
    if limit < prepared.resolved.disk_limit {
        prepared.resolved.disk_limit = limit;
        prepared.disk_budget.set_limit(limit);
    }
    // Start this rank's record transport before reporting the scan. A failure
    // to open the transfer threads, the disk thread, or the record directory
    // then fails the join instead of interrupting serving, when rank zero has
    // already treated this rank as joined.
    let pool = prepared
        .staging_pool
        .as_ref()
        .ok_or_else(|| invalid("a peer's prefix record service needs a staging pool"))?;
    let transfers = factory.mapped(&prepared.contexts, prepared.geometry.clone(), pool)?;
    let staged = Staged::new(
        prepared.staging_pool.take(),
        transfers,
        DiskThread::start("flare-offload-disk")?,
    );
    link.send(&summary(&prepared.scanned)?)?;
    let agreed = decode_agreed(&link.expect(AGREED, REPLY_TIMEOUT)?)?;
    retain(&mut prepared, &agreed)?;
    let summary = format!(
        "serving rank zero, limit {}",
        super::gibibytes(prepared.resolved.disk_limit)
    );
    let directory = prepared
        .resolved
        .directory
        .clone()
        .ok_or_else(|| invalid("a peer's prefix record service needs a record directory"))?;
    let lock = prepared
        .directory_lock
        .take()
        .ok_or_else(|| invalid("a peer's prefix record service needs its directory lock"))?;
    let peer = Peer {
        staged,
        directory,
        fingerprint: prepared.fingerprint,
        geometry: prepared.geometry.clone(),
        link,
        _lock: lock,
    };
    let (finished, outcome) = std::sync::mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("flare-offload-peer".into())
        .spawn(move || {
            let result = serve(peer);
            let _ = finished.send(result.clone());
            result
        })
        .map_err(|error| invalid(format!("cannot start the prefix record service: {error}")))?;
    Ok(Some(Service {
        thread: Some(thread),
        finished: Some(outcome),
        pool: prepared.pool.take(),
        summary,
    }))
}

#[cfg(test)]
mod tests;
