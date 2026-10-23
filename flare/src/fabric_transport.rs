//! Rank-owned TCP channels shared by preparation and serving package distribution.
//! Ignition authenticates checkpoint frames; serving validates received package signatures.

use std::collections::HashSet;
use std::io::{self, BufRead, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::Duration;

use crate::fabric::{self, Failure, Topology};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
struct Config {
    rank: usize,
    world: usize,
    ring: bool,
    #[serde(default)]
    chain: bool,
    streams: usize,
    source_port: u16,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub record: Vec<u8>,
    pub port: u16,
}

/// A rank's in-process relay, retained until every package receiver has drained.
pub struct Relay {
    config: Config,
    topology: Topology,
    stopping: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<Result<(), Failure>>>,
}

impl Relay {
    pub fn start(
        rank: usize,
        streams: usize,
        topology: Topology,
        listener: TcpListener,
        source_port: u16,
    ) -> Result<Self, Failure> {
        topology.validate()?;
        let config = Config {
            rank,
            world: topology.links.len(),
            ring: topology.ring,
            chain: topology.chain,
            streams,
            source_port,
        };
        if rank >= config.world || !(1..=64).contains(&streams) || (rank == 0) != (source_port != 0)
        {
            return Err("invalid fabric relay rank or capacity".into());
        }
        listener.set_nonblocking(true)?;
        let stopping = Arc::new(AtomicBool::new(false));
        let (owned_config, owned_topology, stop) =
            (config.clone(), topology.clone(), stopping.clone());
        let worker =
            thread::spawn(move || serve(&owned_config, &owned_topology, listener, None, &stop));
        Ok(Self {
            config,
            topology,
            stopping,
            worker: Some(worker),
        })
    }

    pub fn connect(&self, stream: usize) -> Result<TcpStream, Failure> {
        if self.config.rank == 0 || stream >= self.config.streams {
            return Err("invalid fabric destination or stream".into());
        }
        upstream(
            &self.config,
            &self.topology,
            self.config.rank,
            stream,
            &Mutex::new(Connections::default()),
        )
    }

    pub fn finish(mut self) -> Result<(), Failure> {
        self.stopping.store(true, Ordering::Release);
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| "fabric relay panicked")?
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn read_json<T: serde::de::DeserializeOwned>(input: &mut impl BufRead) -> Result<T, Failure> {
    let mut bytes = Vec::new();
    input.take(65537).read_until(b'\n', &mut bytes)?;
    if bytes.len() > 65536 || bytes.last() != Some(&b'\n') {
        return Err("invalid preparation transport configuration".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn reply(value: impl Serialize) -> Result<(), Failure> {
    println!("{}", serde_json::to_string(&value)?);
    io::stdout().flush()?;
    Ok(())
}

#[derive(Default)]
struct Connections {
    tcp: Vec<TcpStream>,
    local: Vec<UnixStream>,
}

impl Connections {
    fn stop(&self) {
        for socket in &self.tcp {
            let _ = socket.shutdown(Shutdown::Both);
        }
        for socket in &self.local {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
    fn tcp(&mut self, socket: &TcpStream) -> io::Result<()> {
        self.tcp.push(socket.try_clone()?);
        Ok(())
    }
}

/// On EOF or error, close both directions before joining the other blocked copier.
fn bridge<A: Read + Write + Send, B: Read + Write + Send>(
    mut left: A,
    mut right: B,
    mut left_copy: A,
    mut right_copy: B,
    stop: impl FnOnce(),
) -> Result<(), Failure> {
    thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel();
        let second = sender.clone();
        let a = scope.spawn(move || {
            let _ = sender.send(io::copy(&mut left, &mut right));
        });
        let b = scope.spawn(move || {
            let _ = second.send(io::copy(&mut right_copy, &mut left_copy));
        });
        let result = receiver
            .recv()
            .map_err(|_| "preparation transfer worker stopped")?;
        stop();
        a.join()
            .map_err(|_| "preparation transfer worker panicked")?;
        b.join()
            .map_err(|_| "preparation transfer worker panicked")?;
        result?;
        Ok(())
    })
}

fn upstream(
    config: &Config,
    topology: &Topology,
    destination: usize,
    stream: usize,
    held: &Mutex<Connections>,
) -> Result<TcpStream, Failure> {
    let socket = if config.rank == 0 {
        let socket = TcpStream::connect((Ipv4Addr::LOCALHOST, config.source_port))?;
        held.lock().unwrap().tcp(&socket)?;
        socket
    } else {
        let path = topology.path(0, destination)?;
        let index = path
            .iter()
            .position(|&r| r == config.rank)
            .ok_or("rank is outside the transfer path")?;
        let previous = *path
            .get(index.checked_sub(1).ok_or("source has no upstream")?)
            .ok_or("missing upstream")?;
        let (local, remote) = topology.hop(config.rank, previous, stream)?;
        let socket = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None)?;
        socket.bind(&SocketAddr::from((local, 0)).into())?;
        held.lock().unwrap().tcp(&socket.try_clone()?.into())?;
        socket.connect_timeout(&SocketAddr::V4(remote).into(), Duration::from_secs(10))?;
        let socket: TcpStream = socket.into();
        socket
    };
    socket.set_nodelay(true)?;
    socket.set_write_timeout(Some(Duration::from_secs(30)))?;
    (&socket).write_all(&(destination as u32).to_le_bytes())?;
    (&socket).write_all(&(stream as u32).to_le_bytes())?;
    socket.set_write_timeout(None)?;
    Ok(socket)
}

fn header(socket: &mut impl Read, config: &Config) -> Result<(usize, usize), Failure> {
    let mut bytes = [0; 8];
    socket.read_exact(&mut bytes)?;
    let destination = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let stream = u32::from_le_bytes(bytes[4..].try_into().unwrap()) as usize;
    if destination == 0 || destination >= config.world || stream >= config.streams {
        return Err("preparation channel is outside its rank and stream slots".into());
    }
    Ok((destination, stream))
}

fn relay(
    mut socket: TcpStream,
    config: &Config,
    topology: &Topology,
    held: &Mutex<Connections>,
    slots: &Mutex<HashSet<(usize, usize)>>,
) -> Result<(), Failure> {
    socket.set_read_timeout(Some(Duration::from_secs(30)))?;
    let (destination, stream) = header(&mut socket, config)?;
    let path = topology.path(0, destination)?;
    let index = path
        .iter()
        .position(|&r| r == config.rank)
        .ok_or("rank is outside the transfer path")?;
    let downstream = *path
        .get(index + 1)
        .ok_or("destination cannot relay its own transfer")?;
    let (from, to) = topology.hop(downstream, config.rank, stream)?;
    if socket.local_addr()? != SocketAddr::V4(to) || socket.peer_addr()?.ip() != from {
        return Err("preparation channel did not use its fabric endpoints".into());
    }
    if !slots.lock().unwrap().insert((destination, stream)) {
        return Err("duplicate preparation channel".into());
    }
    socket.set_read_timeout(None)?;
    let upstream = upstream(config, topology, destination, stream, held)?;
    bridge(&socket, &upstream, &socket, &upstream, || {
        let _ = socket.shutdown(Shutdown::Both);
        let _ = upstream.shutdown(Shutdown::Both);
    })
}

fn local(
    mut socket: UnixStream,
    config: &Config,
    topology: &Topology,
    held: &Mutex<Connections>,
    slots: &Mutex<HashSet<(usize, usize)>>,
) -> Result<(), Failure> {
    socket.set_read_timeout(Some(Duration::from_secs(30)))?;
    let (destination, stream) = header(&mut socket, config)?;
    if destination != config.rank || !slots.lock().unwrap().insert((destination, stream)) {
        return Err("invalid or duplicate local preparation channel".into());
    }
    socket.set_read_timeout(None)?;
    let upstream = upstream(config, topology, destination, stream, held)?;
    bridge(&socket, &upstream, &socket, &upstream, || {
        let _ = socket.shutdown(Shutdown::Both);
        let _ = upstream.shutdown(Shutdown::Both);
    })
}

pub fn run(path: &Path) -> Result<(), Failure> {
    let mut input = io::BufReader::new(io::stdin());
    let config: Config = read_json(&mut input)?;
    if !(2..=16).contains(&config.world)
        || config.rank >= config.world
        || !(1..=64).contains(&config.streams)
        || (config.rank == 0) != (config.source_port != 0)
    {
        return Err("invalid preparation transport rank or capacity".into());
    }
    let record = fabric::discover()?;
    let network = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0))?;
    let local_listener = UnixListener::bind(path)?;
    reply(Endpoint {
        record,
        port: network.local_addr()?.port(),
    })?;
    let endpoints: Vec<Endpoint> = read_json(&mut input)?;
    if endpoints.len() != config.world {
        return Err("fabric rank count differs".into());
    }
    let ports = endpoints.iter().map(|e| e.port).collect::<Vec<_>>();
    let records = endpoints.into_iter().map(|e| e.record).collect::<Vec<_>>();
    let topology = Topology::plan_shape(&records, &ports, config.ring, config.chain)?;
    network.set_nonblocking(true)?;
    local_listener.set_nonblocking(true)?;
    reply(serde_json::json!({"ready": true}))?;
    let stopping = Arc::new(AtomicBool::new(false));
    let parent_stopped = stopping.clone();
    thread::spawn(move || {
        let _ = input.read(&mut [0; 1]);
        parent_stopped.store(true, Ordering::Release);
    });
    let result = serve(&config, &topology, network, Some(local_listener), &stopping);
    let _ = std::fs::remove_file(path);
    result
}

fn serve(
    config: &Config,
    topology: &Topology,
    network: TcpListener,
    local_listener: Option<UnixListener>,
    stopping: &AtomicBool,
) -> Result<(), Failure> {
    let held = Mutex::new(Connections::default());
    let slots = Mutex::new(HashSet::new());
    thread::scope(|scope| -> Result<(), Failure> {
        let (sender, receiver) = mpsc::channel();
        let mut workers = Vec::new();
        let mut error = None;
        while !stopping.load(Ordering::Acquire) {
            if let Ok(Err(failure)) = receiver.try_recv() {
                error = Some(failure);
                break;
            }
            // At most one local or relayed channel per destination and stream, plus one
            // pending handshake. Retained shutdown handles have the same finite bound.
            if workers.len() >= config.world * config.streams + 1 {
                error = Some("preparation channel capacity exceeded".into());
                break;
            }
            match network.accept() {
                Ok((socket, _)) => {
                    if let Err(e) = held.lock().unwrap().tcp(&socket) {
                        error = Some(e.into());
                        break;
                    }
                    let (config, topology, held, slots, sender) =
                        (&config, &topology, &held, &slots, sender.clone());
                    workers.push(scope.spawn(move || {
                        let _ = sender.send(relay(socket, config, topology, held, slots));
                    }));
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => {
                    error = Some(e.into());
                    break;
                }
            }
            if let Some(local_listener) = &local_listener {
                match local_listener.accept() {
                    Ok((socket, _)) => {
                        match socket.try_clone() {
                            Ok(copy) => held.lock().unwrap().local.push(copy),
                            Err(e) => {
                                error = Some(e.into());
                                break;
                            }
                        }
                        let (config, topology, held, slots, sender) =
                            (&config, &topology, &held, &slots, sender.clone());
                        workers.push(scope.spawn(move || {
                            let _ = sender.send(local(socket, config, topology, held, slots));
                        }));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(e) => {
                        error = Some(e.into());
                        break;
                    }
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
        held.lock().unwrap().stop();
        for worker in workers {
            worker.join().map_err(|_| "preparation relay panicked")?;
        }
        error.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_direct_and_relayed_channels_preserve_both_directions_and_drain() {
        for (world, ring, chain) in [
            (2, false, false),
            (4, false, false),
            (4, true, false),
            (4, false, true),
        ] {
            let temporary = tempfile::tempdir().unwrap();
            let source = TcpListener::bind("127.0.0.1:0").unwrap();
            let source_port = source.local_addr().unwrap().port();
            let network = (0..world)
                .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
                .collect::<Vec<_>>();
            let ports = network
                .iter()
                .map(|s| s.local_addr().unwrap().port())
                .collect::<Vec<_>>();
            let records = vec![b"a 127.0.0.1/8\n".to_vec(); world];
            let topology = Topology::plan_shape(&records, &ports, ring, chain).unwrap();
            let stopping = AtomicBool::new(false);
            let streams = 4;
            thread::scope(|scope| {
                let server = scope.spawn(|| {
                    thread::scope(|scope| {
                        for _ in 0..(world - 1) * streams {
                            let (mut socket, _) = source.accept().unwrap();
                            scope.spawn(move || {
                                socket
                                    .set_read_timeout(Some(Duration::from_secs(10)))
                                    .unwrap();
                                let mut header = [0; 8];
                                socket.read_exact(&mut header).unwrap();
                                let mut bytes = vec![0; 1 << 20];
                                socket.read_exact(&mut bytes).unwrap();
                                assert!(bytes.iter().all(|b| *b == header[0] + header[4]));
                                socket.write_all(&bytes).unwrap();
                                socket.read_exact(&mut [0; 1]).unwrap();
                            });
                        }
                    });
                });
                let mut relays = Vec::new();
                for (rank, listener) in network.into_iter().enumerate() {
                    let local =
                        UnixListener::bind(temporary.path().join(rank.to_string())).unwrap();
                    listener.set_nonblocking(true).unwrap();
                    local.set_nonblocking(true).unwrap();
                    let (topology, stopping) = (&topology, &stopping);
                    relays.push(scope.spawn(move || {
                        serve(
                            &Config {
                                rank,
                                world,
                                ring,
                                chain,
                                streams,
                                source_port: if rank == 0 { source_port } else { 0 },
                            },
                            topology,
                            listener,
                            Some(local),
                            stopping,
                        )
                    }));
                }
                let mut clients = Vec::new();
                for rank in 1..world {
                    for stream in 0..streams {
                        let path = temporary.path().join(rank.to_string());
                        clients.push(scope.spawn(move || {
                            let mut socket = UnixStream::connect(path).unwrap();
                            socket
                                .set_read_timeout(Some(Duration::from_secs(10)))
                                .unwrap();
                            socket.write_all(&(rank as u32).to_le_bytes()).unwrap();
                            socket.write_all(&(stream as u32).to_le_bytes()).unwrap();
                            let bytes = vec![(rank + stream) as u8; 1 << 20];
                            socket.write_all(&bytes).unwrap();
                            let mut response = vec![0; bytes.len()];
                            socket.read_exact(&mut response).unwrap();
                            assert_eq!(response, bytes);
                            socket.write_all(&[1]).unwrap();
                        }));
                    }
                }
                for client in clients {
                    client.join().unwrap();
                }
                server.join().unwrap();
                stopping.store(true, Ordering::Release);
                for relay in relays {
                    relay.join().unwrap().unwrap();
                }
            });
        }
    }
    #[test]
    fn rejected_channel_aborts_blocked_siblings_before_workers_exit() {
        let temporary = tempfile::tempdir().unwrap();
        let source = TcpListener::bind("127.0.0.1:0").unwrap();
        let network = TcpListener::bind("127.0.0.1:0").unwrap();
        let local_path = temporary.path().join("channels.sock");
        let listener = UnixListener::bind(&local_path).unwrap();
        network.set_nonblocking(true).unwrap();
        listener.set_nonblocking(true).unwrap();
        let topology = Topology::plan(
            &vec![b"a 127.0.0.1/8\n".to_vec(); 2],
            &[
                source.local_addr().unwrap().port(),
                network.local_addr().unwrap().port(),
            ],
            false,
        )
        .unwrap();
        let worker = thread::spawn(move || {
            serve(
                &Config {
                    rank: 1,
                    world: 2,
                    ring: false,
                    chain: false,
                    streams: 2,
                    source_port: 0,
                },
                &topology,
                network,
                Some(listener),
                &AtomicBool::new(false),
            )
        });
        let mut active = UnixStream::connect(&local_path).unwrap();
        active
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        active.write_all(&[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let (mut upstream, _) = source.accept().unwrap();
        upstream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        upstream.read_exact(&mut [0; 8]).unwrap();
        let mut rejected = UnixStream::connect(&local_path).unwrap();
        rejected.write_all(&[0; 8]).unwrap();
        assert_eq!(active.read(&mut [0; 1]).unwrap(), 0);
        assert_eq!(upstream.read(&mut [0; 1]).unwrap(), 0);
        assert!(
            worker
                .join()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("rank and stream slots")
        );
    }
}
