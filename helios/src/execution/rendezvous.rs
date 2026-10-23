//! Rank zero's door for a package whose ranks run on separate hosts.
//!
//! Every host runs `flare run` on its own rank's file. Rank zero listens on the rendezvous address
//! and each peer knocks there with its rank, the world size and its file's export identifier; rank
//! zero admits every rank once and only from the same export. Ranks join before loading. The
//! admitted connection first carries the cache capacity every rank allocates, then the transport
//! records the ranks exchange, then confirmation that every peer connected its transport, and
//! finally the coordinated execution protocol.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use super::RankStream;
use crate::{Error, ExecutionError, Result};

// Version 3 negotiates capacity before loading and exchanges discovered fabric endpoints.
const MAGIC: &[u8; 8] = b"FLARERV3";
const ADMITTED: u8 = 0;
const REFUSED: u8 = 1;
const CONNECTED: &[u8] = b"connected";
// Transport records and refusal reasons are small; a larger frame is a protocol error.
const FRAME_LIMIT: usize = 1 << 20;
const HELLO_TIMEOUT: Duration = Duration::from_secs(30);

fn failure(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Worker, message)
}

fn io(context: &str, error: std::io::Error) -> Error {
    failure(format!("{context}: {error}"))
}

fn write_frame(stream: &mut impl Write, bytes: &[u8]) -> Result<()> {
    stream
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .and_then(|_| stream.write_all(bytes))
        .and_then(|_| stream.flush())
        .map_err(|e| io("rendezvous write failed", e))
}

fn read_frame(stream: &mut impl Read) -> Result<Vec<u8>> {
    let mut length = [0u8; 8];
    stream
        .read_exact(&mut length)
        .map_err(|e| io("rendezvous read failed", e))?;
    let length = u64::from_le_bytes(length) as usize;
    if length > FRAME_LIMIT {
        return Err(failure("rendezvous frame exceeds its limit"));
    }
    let mut bytes = vec![0u8; length];
    stream
        .read_exact(&mut bytes)
        .map_err(|e| io("rendezvous read failed", e))?;
    Ok(bytes)
}

fn hello(rank: usize, world: usize, export_id: &str) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend((world as u32).to_le_bytes());
    bytes.extend((rank as u32).to_le_bytes());
    bytes.extend(export_id.as_bytes());
    bytes
}

fn parse_hello(bytes: &[u8]) -> Option<(usize, usize, &str)> {
    let rest = bytes.strip_prefix(MAGIC.as_slice())?;
    if rest.len() < 8 {
        return None;
    }
    let world = u32::from_le_bytes(rest[..4].try_into().ok()?) as usize;
    let rank = u32::from_le_bytes(rest[4..8].try_into().ok()?) as usize;
    Some((rank, world, std::str::from_utf8(&rest[8..]).ok()?))
}

/// Admit ranks 1..world at `address`; returns their streams in rank order.
pub fn accept(
    address: &str,
    world: usize,
    export_id: &str,
    timeout: Duration,
    mut report: impl FnMut(usize),
) -> Result<Vec<RankStream>> {
    let listener =
        TcpListener::bind(address).map_err(|e| io(&format!("cannot listen on {address}"), e))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| io("rendezvous listener", e))?;
    let deadline = Instant::now() + timeout;
    let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
    let mut admitted = 0;
    while admitted + 1 < world {
        let (mut stream, origin) = match listener.accept() {
            Ok(connection) => connection,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    let missing: Vec<_> = (1..world).filter(|r| peers[*r].is_none()).collect();
                    return Err(failure(format!(
                        "ranks {missing:?} did not join within {} s",
                        timeout.as_secs()
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(error) => return Err(io("rendezvous accept failed", error)),
        };
        stream
            .set_nonblocking(false)
            .and_then(|_| stream.set_read_timeout(Some(HELLO_TIMEOUT)))
            .and_then(|_| stream.set_nodelay(true))
            .map_err(|e| io("rendezvous connection", e))?;
        let Ok(bytes) = read_frame(&mut stream) else {
            continue;
        };
        let refusal = match parse_hello(&bytes) {
            None if bytes.starts_with(b"FLARERV") => {
                Some("incompatible rendezvous protocol; rebuild Flare on every host".to_owned())
            }
            None => Some("not a flare rank".to_owned()),
            Some((_, peer_world, _)) if peer_world != world => Some(format!(
                "its file has {peer_world} ranks, rank zero's has {world}"
            )),
            Some((_, _, peer_export)) if peer_export != export_id => {
                Some("its file comes from a different export than rank zero's".to_owned())
            }
            Some((rank, _, _)) if rank == 0 || rank >= world => {
                Some(format!("rank {rank} is invalid"))
            }
            Some((rank, _, _)) if peers[rank].is_some() => {
                Some(format!("rank {rank} already joined"))
            }
            Some(_) => None,
        };
        match refusal {
            Some(reason) => {
                eprintln!("Refused a rank from {origin}: {reason}");
                let mut reply = vec![REFUSED];
                reply.extend(reason.as_bytes());
                let _ = write_frame(&mut stream, &reply);
            }
            None => {
                let (rank, _, _) = parse_hello(&bytes).unwrap();
                write_frame(&mut stream, &[ADMITTED])?;
                stream
                    .set_read_timeout(None)
                    .map_err(|e| io("rendezvous connection", e))?;
                peers[rank] = Some(stream);
                admitted += 1;
                report(rank);
            }
        }
    }
    Ok(peers
        .into_iter()
        .skip(1)
        .map(|stream| RankStream::from(stream.unwrap()))
        .collect())
}

/// Knock on rank zero's rendezvous until admitted or `timeout` passes.
pub fn knock(
    address: &str,
    rank: usize,
    world: usize,
    export_id: &str,
    timeout: Duration,
) -> Result<RankStream> {
    let deadline = Instant::now() + timeout;
    let targets: Vec<SocketAddr> = address
        .to_socket_addrs()
        .map_err(|e| io(&format!("cannot resolve {address}"), e))?
        .collect();
    loop {
        let attempt = targets
            .iter()
            .find_map(|target| TcpStream::connect_timeout(target, Duration::from_secs(5)).ok());
        if let Some(mut stream) = attempt {
            stream
                .set_nodelay(true)
                .and_then(|_| stream.set_read_timeout(Some(HELLO_TIMEOUT)))
                .map_err(|e| io("rendezvous connection", e))?;
            write_frame(&mut stream, &hello(rank, world, export_id))?;
            let reply = read_frame(&mut stream)?;
            return match reply.split_first() {
                Some((&ADMITTED, [])) => {
                    stream
                        .set_read_timeout(None)
                        .map_err(|e| io("rendezvous connection", e))?;
                    Ok(RankStream::from(stream))
                }
                Some((&REFUSED, reason)) => Err(failure(format!(
                    "rank zero refused rank {rank}: {}",
                    String::from_utf8_lossy(reason)
                ))),
                _ => Err(failure("rank zero's rendezvous reply is malformed")),
            };
        }
        if Instant::now() >= deadline {
            return Err(failure(format!(
                "rank zero did not answer at {address} within {} s",
                timeout.as_secs()
            )));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn encode_capacity((rows, pages): (u64, u64)) -> Vec<u8> {
    [rows.to_le_bytes(), pages.to_le_bytes()].concat()
}

fn decode_capacity(bytes: &[u8]) -> Result<(u64, u64)> {
    if bytes.len() != 16 {
        return Err(failure("rendezvous cache capacity is malformed"));
    }
    let word = |range: std::ops::Range<usize>| u64::from_le_bytes(bytes[range].try_into().unwrap());
    let capacity = (word(0..8), word(8..16));
    if capacity.0 == 0 || capacity.1 == 0 {
        return Err(failure("rank cache capacity must be positive"));
    }
    Ok(capacity)
}

/// Rank zero: agree on the continuation rows and KV pages every rank allocates, the smallest any
/// rank can back, and send them to each peer. Ranks size from their own free memory, but every
/// rank must hold the same rows and pages: rank zero's scheduler addresses all of them.
pub fn agree_capacity(peers: &mut [RankStream], local: (u64, u64)) -> Result<(u64, u64)> {
    if local.0 == 0 || local.1 == 0 {
        return Err(failure("rank cache capacity must be positive"));
    }
    let mut agreed = local;
    for peer in peers.iter_mut() {
        let (rows, pages) = decode_capacity(&read_frame(peer)?)?;
        agreed = (agreed.0.min(rows), agreed.1.min(pages));
    }
    for peer in peers.iter_mut() {
        write_frame(peer, &encode_capacity(agreed))?;
    }
    Ok(agreed)
}

/// A peer: propose the rows and pages this rank can back and receive what every rank allocates.
pub fn propose_capacity(stream: &mut RankStream, local: (u64, u64)) -> Result<(u64, u64)> {
    if local.0 == 0 || local.1 == 0 {
        return Err(failure("rank cache capacity must be positive"));
    }
    write_frame(stream, &encode_capacity(local))?;
    let agreed = decode_capacity(&read_frame(stream)?)?;
    if agreed.0 > local.0 || agreed.1 > local.1 {
        return Err(failure(
            "rank zero agreed on more cache than this rank can back",
        ));
    }
    Ok(agreed)
}

/// Rank zero: collect every peer's transport record and send all of them, in rank order, to each.
pub fn share_records(peers: &mut [RankStream], local: Vec<u8>) -> Result<Vec<Vec<u8>>> {
    let mut records = vec![local];
    for peer in peers.iter_mut() {
        records.push(read_frame(peer)?);
    }
    let encoded = encode(&records);
    for peer in peers.iter_mut() {
        write_frame(peer, &encoded)?;
    }
    Ok(records)
}

/// Rank zero: like [`share_records`], with a different rank-zero record for each peer.
///
/// `local` holds one entry per rank: entry zero is rank zero's own record, and
/// entry `i + 1` is the record sent to peer `i`. A record that names an address
/// can therefore name the address that peer reaches rank zero on.
pub fn share_records_each(peers: &mut [RankStream], local: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
    if local.len() != peers.len() + 1 {
        return Err(failure("rank zero prepared records for a different world"));
    }
    let mut records = vec![local[0].clone()];
    for peer in peers.iter_mut() {
        records.push(read_frame(peer)?);
    }
    for (index, peer) in peers.iter_mut().enumerate() {
        let mut for_peer = records.clone();
        for_peer[0] = local[index + 1].clone();
        write_frame(peer, &encode(&for_peer))?;
    }
    Ok(records)
}

/// A peer: send this rank's transport record and receive every rank's.
pub fn exchange_record(
    stream: &mut RankStream,
    local: &[u8],
    world: usize,
) -> Result<Vec<Vec<u8>>> {
    write_frame(stream, local)?;
    let records = decode(&read_frame(stream)?)?;
    if records.len() != world {
        return Err(failure(
            "rank zero sent transport records for a different world",
        ));
    }
    Ok(records)
}

/// A peer: confirm that its transport is connected; no graph runs until every rank did.
pub fn report_connected(stream: &mut RankStream) -> Result<()> {
    write_frame(stream, CONNECTED)
}

/// Rank zero: wait for every peer's transport.
pub fn await_connected(peers: &mut [RankStream]) -> Result<()> {
    for (index, peer) in peers.iter_mut().enumerate() {
        if read_frame(peer)? != CONNECTED {
            return Err(failure(format!(
                "rank {} did not connect its transport",
                index + 1
            )));
        }
    }
    Ok(())
}

fn encode(records: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for record in records {
        bytes.extend((record.len() as u64).to_le_bytes());
        bytes.extend(record);
    }
    bytes
}

fn decode(mut bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut records = Vec::new();
    while !bytes.is_empty() {
        let (length, rest) = bytes
            .split_first_chunk::<8>()
            .ok_or_else(|| failure("truncated transport records"))?;
        let length = u64::from_le_bytes(*length) as usize;
        if rest.len() < length {
            return Err(failure("truncated transport records"));
        }
        records.push(rest[..length].to_vec());
        bytes = &rest[length..];
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_join_once_from_one_export_and_exchange_records() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        drop(listener);
        let zero = {
            let address = address.clone();
            std::thread::spawn(move || {
                let mut peers =
                    accept(&address, 3, "export-a", Duration::from_secs(20), |_| {}).unwrap();
                let capacity = agree_capacity(&mut peers, (54, 4096)).unwrap();
                let records = share_records(&mut peers, b"zero".to_vec()).unwrap();
                await_connected(&mut peers).unwrap();
                (capacity, records)
            })
        };
        // A rank from another export is refused, then the real ranks join out of order.
        match knock(&address, 1, 3, "export-b", Duration::from_secs(20)) {
            Err(error) => assert!(error.to_string().contains("different export"), "{error}"),
            Ok(_) => panic!("a rank from another export joined"),
        }
        let joins: Vec<_> = [2usize, 1]
            .into_iter()
            .map(|rank| {
                let address = address.clone();
                std::thread::spawn(move || {
                    let mut stream =
                        knock(&address, rank, 3, "export-a", Duration::from_secs(20)).unwrap();
                    // Every rank allocates the fewest rows and pages any rank can back.
                    let local = [(0, 0), (50, 4352), (60, 3840)][rank];
                    let capacity = propose_capacity(&mut stream, local).unwrap();
                    let records =
                        exchange_record(&mut stream, format!("rank{rank}").as_bytes(), 3).unwrap();
                    report_connected(&mut stream).unwrap();
                    (capacity, records)
                })
            })
            .collect();
        let expected: Vec<Vec<u8>> = vec![b"zero".to_vec(), b"rank1".to_vec(), b"rank2".to_vec()];
        assert_eq!(zero.join().unwrap(), ((50, 3840), expected.clone()));
        for join in joins {
            assert_eq!(join.join().unwrap(), ((50, 3840), expected.clone()));
        }
    }

    #[test]
    fn a_peer_refuses_more_cache_than_it_can_back() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        drop(listener);
        let zero = {
            let address = address.clone();
            std::thread::spawn(move || {
                let mut peers =
                    accept(&address, 2, "export", Duration::from_secs(20), |_| {}).unwrap();
                // A corrupted rank zero answers with more than the peer proposed.
                let _ = read_frame(&mut peers[0]).unwrap();
                write_frame(&mut peers[0], &encode_capacity((64, 4096))).unwrap();
            })
        };
        let mut stream = knock(&address, 1, 2, "export", Duration::from_secs(20)).unwrap();
        let error = propose_capacity(&mut stream, (54, 4096)).unwrap_err();
        assert!(error.to_string().contains("more cache"), "{error}");
        zero.join().unwrap();
    }

    #[test]
    fn cache_negotiation_rejects_malformed_or_invalid_limits() {
        use super::super::rank_stream::test_pair;

        for capacity in [(81, 100), (80, 101), (0, 100), (80, 0)] {
            let (mut root, mut peer) = test_pair().unwrap();
            let sender = std::thread::spawn(move || {
                assert_eq!(
                    decode_capacity(&read_frame(&mut root).unwrap()).unwrap(),
                    (80, 100)
                );
                write_frame(&mut root, &encode_capacity(capacity)).unwrap();
            });
            assert!(propose_capacity(&mut peer, (80, 100)).is_err());
            sender.join().unwrap();
        }
        for capacity in [(0, 100), (80, 0)] {
            assert!(agree_capacity(&mut [], capacity).is_err());
            let (_root, mut peer) = test_pair().unwrap();
            assert!(propose_capacity(&mut peer, capacity).is_err());
        }
        assert!(decode_capacity(&[0; 15]).is_err());
        assert!(decode_capacity(&[0; 17]).is_err());
    }

    #[test]
    fn rendezvous_version_matches_the_capacity_and_fabric_exchange() {
        assert_eq!(
            parse_hello(&hello(1, 2, "export-a")),
            Some((1, 2, "export-a"))
        );
        for version in [b"FLARERV1", b"FLARERV2"] {
            let mut old = hello(1, 2, "export-a");
            old[..8].copy_from_slice(version);
            assert!(parse_hello(&old).is_none());
        }
    }
}
