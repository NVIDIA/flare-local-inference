//! Launch explicitly ordered peer agents while rank zero serves in this process.

use crate::{
    agent_client::Client,
    cli::Run,
    fabric::Failure,
    orbit::{
        self, frame,
        grant::{Grant, Member, Package, Signed},
        session::Session,
        store::Node,
    },
};
use flare_helios::execution::roce_fabric::Shape;
use flare_helios::runtime_package::{self, AuthorizedPackage, Publisher};
use serde_json::{Value, json};
use std::{
    fs::File,
    net::{IpAddr, SocketAddr, TcpStream},
    path::Path,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub struct Peers {
    client: Client,
    group: String,
    nodes: Vec<Node>,
    sockets: Vec<TcpStream>,
    workers: Vec<thread::JoinHandle<()>>,
    ended: mpsc::Receiver<Result<(), String>>,
    remaining: usize,
}
impl Peers {
    pub fn start(
        args: &mut Run,
        package: &AuthorizedPackage,
        publisher: &Publisher,
        path: &Path,
    ) -> Result<Self, Failure> {
        let (world, export) = runtime_package::package_cluster(package)
            .ok_or("--hosts requires a multi-host package")?;
        if args.rank.is_some_and(|rank| rank != 0) || args.hosts.len() + 1 != world {
            return Err(format!(
                "--hosts requires {} peer agents in rank order; rank zero runs locally",
                world - 1
            )
            .into());
        }
        let client = Client::open(args.agents_file.as_deref(), args.controller_key.as_deref())?;
        let nodes = client.peers(&args.hosts)?;
        let mut sessions = Vec::new();
        let mut members = vec![Member {
            rank: 0,
            name: orbit::hostname(),
            public_key: client.key.public().to_hex(),
            address: String::new(),
            gpus: Vec::new(),
        }];
        for (index, node) in nodes.iter().enumerate() {
            let (mut session, hello) = client.node_session(node)?;
            hello.installed_executable()?;
            let status: Value =
                serde_json::from_slice(&session.request(frame::STATUS, &json!({}))?)?;
            let mut gpus = Vec::new();
            for &index in &args.gpus {
                let gpu = status["gpus"]
                    .as_array()
                    .and_then(|gpus| {
                        gpus.iter()
                            .find(|g| g["index"].as_i64() == Some(index as i64))
                    })
                    .ok_or_else(|| format!("{} has no GPU {index}", node.name))?;
                if !gpu["group"].is_null() {
                    return Err(format!("{} GPU {index} is in use", node.name).into());
                }
                gpus.push(
                    gpu["uuid"]
                        .as_str()
                        .ok_or("agent GPU lacks UUID")?
                        .to_owned(),
                );
            }
            members.push(Member {
                rank: index + 1,
                name: node.name.clone(),
                public_key: node.public_key.clone(),
                address: node.address.clone(),
                gpus,
            });
            sessions.push(session);
        }
        let local = sessions[0].addresses()?.0.ip();
        let rendezvous = match args.rendezvous.as_deref() {
            Some(address) => {
                let (host, port) = address
                    .rsplit_once(':')
                    .ok_or("rendezvous requires HOST:PORT")?;
                if host.parse::<IpAddr>().is_ok_and(|ip| ip.is_unspecified()) {
                    SocketAddr::new(local, port.parse()?).to_string()
                } else {
                    address.to_owned()
                }
            }
            None => SocketAddr::new(local, 29800).to_string(),
        };
        args.rendezvous = Some(rendezvous.clone());
        args.rank = Some(0);
        let issued = orbit::now();
        let group = format!("manual-{}-{}", issued, orbit::hex(&orbit::random(4)?));
        let mut serving = args
            .serving
            .rank_options(Shape::from_flags(args.roce_ring, args.roce_chain));
        if args.hub.offline {
            serving.push("--offline".into());
        }
        if args.stage_only {
            serving.push("--stage-only".into());
        }
        let grant = Grant {
            group: group.clone(),
            name: String::new(),
            package: Package {
                export,
                length: File::open(path)?.metadata()?.len(),
                files: vec![
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .ok_or("package filename must be UTF-8")?
                        .to_owned(),
                ],
                source: None,
            },
            members,
            rendezvous,
            serving,
            rank_zero: Vec::new(),
            roce_ring: args.roce_ring,
            roce_chain: args.roce_chain,
            transfer_fabric: None,
            single_host: false,
            issued,
            expires: issued + 600,
        };
        let signed = Signed::sign(&client.key, &grant)?;
        let (sender, ended) = mpsc::channel();
        let mut peers = Self {
            client,
            group,
            nodes: Vec::new(),
            sockets: Vec::new(),
            workers: Vec::new(),
            ended,
            remaining: 0,
        };
        for (index, (node, mut session)) in nodes.into_iter().zip(sessions).enumerate() {
            let rank = index + 1;
            session.request(
                frame::RUN,
                &json!({"grant": signed, "rank": rank, "publisher": publisher}),
            )?;
            peers.nodes.push(node.clone());
            peers.sockets.push(session.try_clone_stream()?);
            let sender = sender.clone();
            eprintln!("Started rank {rank} on {}", node.name);
            peers.workers.push(thread::spawn(move || {
                let result = output(&mut session, rank, &node.name);
                if result.is_err() {
                    crate::STOP.store(true, std::sync::atomic::Ordering::Release);
                }
                let _ = sender.send(result);
            }));
            peers.remaining += 1;
        }
        Ok(peers)
    }

    /// The local rank has drained; require every agent to observe a successful peer exit.
    pub fn finish(mut self) -> Result<(), Failure> {
        let deadline = Instant::now() + Duration::from_secs(30);
        while self.remaining > 0 {
            let result = self
                .ended
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|_| "peer agents did not report exit after rank zero drained")?;
            self.remaining -= 1;
            result?;
        }
        for worker in self.workers.drain(..) {
            worker.join().map_err(|_| "agent output worker panicked")?;
        }
        self.nodes.clear();
        Ok(())
    }
}
impl Drop for Peers {
    fn drop(&mut self) {
        for node in &self.nodes {
            let result = self.client.node_session(node).and_then(|(mut session, _)| {
                session.request(frame::STOP, &json!({"group": self.group}))
            });
            if let Err(e) = result {
                eprintln!("{}: could not stop peer: {e}", node.name);
            }
        }
        for socket in &self.sockets {
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
fn output(session: &mut Session, rank: usize, name: &str) -> Result<(), String> {
    session
        .set_read_timeout(Some(Duration::from_secs(90)))
        .map_err(|e| e.to_string())?;
    loop {
        match session
            .read_frame()
            .map_err(|e| format!("rank {rank} on {name}: {e}"))?
        {
            (frame::LOG, line) if line.is_empty() => {}
            (frame::LOG, line) => {
                eprintln!("[rank {rank} {name}] {}", String::from_utf8_lossy(&line))
            }
            (frame::EXIT, code) => {
                let code = <[u8; 4]>::try_from(code.as_slice())
                    .map(i32::from_le_bytes)
                    .map_err(|_| "malformed agent exit status")?;
                return if code == 0 {
                    Ok(())
                } else {
                    Err(format!("rank {rank} on {name} exited with status {code}"))
                };
            }
            (frame::ERROR, message) => return Err(String::from_utf8_lossy(&message).into_owned()),
            (tag, _) => return Err(format!("unexpected agent frame {tag}")),
        }
    }
}
