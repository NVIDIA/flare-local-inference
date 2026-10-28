//! A multi-process instance as real processes on one host: a source, rank 0 and the agents of
//! the peer ranks are separate OS processes (the `rank_node` example), the manager runs in
//! this one. Ranks are killed in the middle of a restore.
//!
//! Run from `selene-manager/` with `cargo test --features test-support`.

use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{Receiver, channel},
    thread,
    time::{Duration, Instant},
};

use selene_manager::testing::{Manager, free_address};

fn start_manager() -> Manager {
    let mut manager = Manager::new();
    manager.config.memory = 16 << 20;
    manager.config.staging = 16 << 20;
    manager.config.metrics = Some(free_address());
    manager.run();
    manager
}

struct Node {
    child: Child,
    input: ChildStdin,
    lines: Receiver<String>,
    name: String,
}

fn example(name: &str) -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    path.pop();
    path.join("examples").join(name)
}

impl Node {
    fn spawn(name: &str, args: &[&str]) -> Self {
        let mut child = Command::new(example("rank_node"))
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn rank_node");
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (tx, lines) = channel();
        thread::spawn(move || {
            for line in BufReader::new(output).lines().map_while(|l| l.ok()) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        Self {
            child,
            input,
            lines,
            name: name.into(),
        }
    }

    fn send(&mut self, line: &str) {
        let _ = writeln!(self.input, "{line}");
        let _ = self.input.flush();
    }

    /// The next line that starts with `prefix`, within `seconds`.
    fn expect(&mut self, prefix: &str, seconds: u64) -> String {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) if line.starts_with(prefix) => return line,
                Ok(_) => {}
                Err(_) => panic!("{} said no `{prefix}` in {seconds}s", self.name),
            }
        }
    }

    /// Whether a line starting with `prefix` arrives within `seconds`.
    fn says(&mut self, prefix: &str, seconds: u64) -> bool {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) if line.starts_with(prefix) => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
        }
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.kill();
    }
}

/// A cluster: the source, rank 0 and `world - 1` agents, all running.
struct Cluster {
    manager: Manager,
    source: Node,
    rank0: Node,
    agents: Vec<Node>,
}

fn cluster(world: usize, holds: &[bool]) -> Cluster {
    let manager = start_manager();
    let socket = manager.config.socket.to_str().unwrap().to_string();
    let world_text = world.to_string();
    let mut source = Node::spawn(
        "source",
        &["source", "--socket", &socket, "--world", &world_text],
    );
    let mut rank0 = Node::spawn(
        "rank 0",
        &["rank0", "--socket", &socket, "--world", &world_text],
    );
    let port = rank0.expect("link-port ", 15)["link-port ".len()..].to_string();
    let link = format!("127.0.0.1:{port}");
    let mut agents = Vec::new();
    for rank in 1..world {
        let hold = holds.get(rank - 1).copied().unwrap_or(false).to_string();
        agents.push(Node::spawn(
            &format!("agent {rank}"),
            &[
                "agent",
                "--socket",
                &socket,
                "--world",
                &world_text,
                "--rank",
                &rank.to_string(),
                "--link",
                &link,
                "--hold-copy",
                &hold,
            ],
        ));
    }
    source.expect("ready", 20);
    rank0.expect("ready", 20);
    for agent in &mut agents {
        agent.expect("ready", 20);
    }
    Cluster {
        manager,
        source,
        rank0,
        agents,
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_killed_peer_stops_rank_zero_and_quarantined_windows_wait_for_process_exit() {
    // Kill after submission while the peer holds its completion.
    let mut cluster = cluster(2, &[true]);
    cluster.rank0.send("restore");
    cluster.agents[0].expect("copy pending", 20);
    cluster.agents[0].kill();
    assert!(
        cluster.rank0.says("poll err", 30),
        "rank 0 did not stop when a rank died with a copy outstanding"
    );
    // No result was ever given for pages the dead rank may have been writing.
    assert!(!cluster.rank0.says("fetched ok", 1));
    // Rank zero cannot prove its peer drained and has quarantined the whole claim.
    wait_for("the claim to be quarantined", || {
        cluster.manager.gauge("pool_quarantined_bytes") > 0
    });
    cluster.rank0.kill();
    wait_for("all quarantined participants to exit", || {
        cluster.manager.gauge("pool_quarantined_bytes") == 0
    });
    cluster.source.send("quit");
}

#[test]
fn a_killed_rank_zero_lets_the_agents_drain_attest_and_exit() {
    let mut cluster = cluster(2, &[true]);
    cluster.rank0.send("restore");
    cluster.agents[0].expect("copy pending", 20);
    cluster.rank0.kill();
    cluster.agents[0].send("release");
    // The agent sees its link close, finishes the copy it started, attests to the manager and
    // stops.
    assert!(
        cluster.agents[0].says("agent disabled", 30),
        "the agent did not stop when rank 0 died"
    );
    cluster.agents[0].expect("closed ", 30);
    wait_for("the windows to be returned", || {
        cluster.manager.gauge("pool_quarantined_bytes") == 0
    });
    cluster.source.send("quit");
}
