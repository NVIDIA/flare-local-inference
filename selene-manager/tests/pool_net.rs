//! Pool protocol over the network listener: sessions without shared memory, TLS, RDMA
//! and the startup warnings. What moves bytes over the data channel is in `pool_channel`.

mod support;

use selene_manager::proto::v2::{self, client_message as c, server_message::M};
use std::{
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use support::{
    net::{Net, free_address},
    pool::{self, Ident, Pool, Session, chain, saved},
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint};

async fn open(pool: &mut Pool, name: &str) -> Session {
    let mut session = pool.open(&Ident::new(name)).await;
    assert_eq!(session.opened.status, v2::OpenStatus::OpenOk as i32);
    session.snapshot(vec![]).await;
    session
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn h6_a_tcp_session_has_no_shared_memory_and_serves_exports_over_the_data_channel() {
    let net = Net::start(|_| {});
    let unix = net.unix().await;
    assert!(unix.hello.shared_memory.is_some());
    let mut tcp = net.tcp().await;
    assert!(
        tcp.hello.shared_memory.is_none(),
        "no shared memory over TCP"
    );
    assert_eq!(tcp.hello.encryption, v2::Encryption::Plaintext as i32);
    assert_eq!(
        tcp.hello.data_transports,
        vec![v2::DataTransport::Tcp as i32]
    );
    let session = tcp.open(&Ident::new("remote")).await;
    assert_eq!(
        session.opened.export_slots, 2,
        "an instance without shared memory is a source whose windows are relayed"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l4_and_a22_remote_sessions_see_the_managers_records_as_level_five_and_instances_through_the_relay()
 {
    let net = Net::start(|_| {});
    let mut unix = net.unix().await;
    // A same-host source with shared memory, and a record in the manager's store.
    let mut source = open(&mut unix, "source").await;
    source
        .notices(
            chain(&[1024], v2::Tier::Dram)
                .into_iter()
                .map(saved)
                .collect(),
        )
        .await;
    let mut writer = open(&mut unix, "writer").await;
    let record = support::key(2048);
    match writer
        .put_record(1, support::key(1024), None, pool::Damage::default())
        .await
    {
        pool::PutResult::Stored(_) => {}
        other => panic!("{other:?}"),
    }
    let _ = record;
    // Same host: the manager's DRAM is level 3.
    let mut near = open(&mut unix, "near").await;
    let local = near.manager_candidate(1, 1024).await;
    assert_eq!(local.level, v2::Level::HostDram as i32);
    // Another host: every manager record is level 5, and an instance's records are offered
    // too, relayed through the manager.
    let mut tcp = net.tcp().await;
    let mut far = open(&mut tcp, "far").await;
    let remote = far.manager_candidate(2, 1024).await;
    assert_eq!(remote.level, v2::Level::RemoteNvme as i32);
    assert!(remote.source.as_ref().unwrap().instance_id.is_empty());
    let reply = far.lookup(3, 1024, 0).await;
    assert!(
        reply.candidates.iter().any(|candidate| candidate
            .source
            .as_ref()
            .is_some_and(|source| !source.instance_id.is_empty())),
        "the same-host instance's record is offered to the remote session"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_remote_session_cannot_use_shared_windows() {
    let net = Net::start(|_| {});
    let mut tcp = net.tcp().await;
    let mut far = open(&mut tcp, "far").await;
    let claim = v2::Claim {
        claim_id: 1,
        source: Some(v2::SourceRef::default()),
        tier: v2::Tier::Dram as i32,
        key: Some(pool::pkey(1024)),
        begin_tokens: 0,
        want_row: true,
        mode: v2::CopyMode::SharedWindows as i32,
        deadline_ms: 0,
    };
    far.claim(claim).await;
    assert_eq!(
        far.claim_terminal(1).await.reason,
        v2::ClaimReason::InvalidClaim as i32
    );
    // The same for a put, with the reason in the detail.
    far.send(c::M::PutBegin(v2::PutBegin {
        put_id: 1,
        key: Some(pool::pkey(1024)),
        chain_pages: 16,
        row: true,
        has_published_parent: false,
        published_parent: None,
        bucket: vec![4; 16],
        published_tail_pages: 16,
        mode: v2::CopyMode::SharedWindows as i32,
    }))
    .await;
    let refused = far
        .take(|m| match m {
            M::PutRefused(refused) if refused.put_id == 1 => Some(refused.clone()),
            _ => None,
        })
        .await;
    assert_eq!(refused.reason, v2::Reason::Invalid as i32);
    assert!(
        refused.detail.contains("shared windows"),
        "{}",
        refused.detail
    );
}

/// A self-signed certificate for `localhost`, as PEM files in `dir`.
fn certificate(dir: &std::path::Path) -> (PathBuf, PathBuf, String) {
    let generated = selene_manager::testing::certificate();
    let cert = dir.join("server.pem");
    let key = dir.join("server-key.pem");
    let pem = generated.cert.pem();
    std::fs::write(&cert, &pem).unwrap();
    std::fs::write(&key, generated.signing_key.serialize_pem()).unwrap();
    (cert, key, pem)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn h4_a_tls_manager_declares_tls_and_offers_no_rdma() {
    let dir = std::env::temp_dir().join(format!("selene-tls-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cert, key, pem) = certificate(&dir);
    let net = Net::start(|config| {
        config.tls_cert = Some(cert.clone());
        config.tls_key = Some(key.clone());
    });
    // A stalled TLS handshake must not block another client's session.
    let _idle = tokio::net::TcpStream::connect(net.address).await.unwrap();
    let channel = Endpoint::from_shared(format!("https://localhost:{}", net.address.port()))
        .unwrap()
        .connect_timeout(Duration::from_secs(2))
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(pem.clone()))
                .domain_name("localhost"),
        )
        .unwrap()
        .connect()
        .await
        .expect("a client with the manager's CA connects");
    let mut tls = Pool::over(channel, &net.socket).await;
    assert_eq!(tls.hello.encryption, v2::Encryption::Tls as i32);
    assert!(
        !tls.hello
            .data_transports
            .contains(&(v2::DataTransport::Rdma as i32)),
        "RDMA is never offered by a TLS manager"
    );
    let session = tls.open(&Ident::new("far")).await;
    assert_eq!(session.opened.status, v2::OpenStatus::OpenOk as i32);
    // A client that does not hold the manager's CA is refused before any call.
    let other = selene_manager::testing::certificate();
    let refused = Endpoint::from_shared(format!("https://localhost:{}", net.address.port()))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(other.cert.pem()))
                .domain_name("localhost"),
        )
        .unwrap()
        .connect()
        .await;
    assert!(
        refused.is_err(),
        "a manager whose certificate the client does not trust"
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn selene(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_selene"))
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap()
}

#[test]
fn rdma_with_tls_is_refused_at_startup_with_the_reason() {
    let dir = std::env::temp_dir().join(format!("selene-rdma-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cert, key, _) = certificate(&dir);
    let socket = dir.join("run/m.sock");
    let output = selene(&[
        "--socket",
        socket.to_str().unwrap(),
        "--memory",
        "8M",
        "--staging",
        "8M",
        "--listen",
        "127.0.0.1:0",
        "--rdma",
        "--tls-cert",
        cert.to_str().unwrap(),
        "--tls-key",
        key.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    if cfg!(feature = "rdma") {
        assert!(text.contains("not encrypted"), "{text}");
    } else {
        assert!(text.contains("`rdma` feature"), "{text}");
    }
    assert!(!socket.exists(), "nothing was started");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn startup_names_each_option_that_widens_the_exposure() {
    let dir = std::env::temp_dir().join(format!("selene-warn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let socket = dir.join("run/m.sock");
    let address = free_address();
    let name = support::instance_name("warn");
    let _shm = support::ShmCleanup::new(&name, &socket);
    let child = Command::new(env!("CARGO_BIN_EXE_selene"))
        .args([
            "--name",
            &name,
            "--socket",
            socket.to_str().unwrap(),
            "--memory",
            "8M",
            "--staging",
            "8M",
            "--listen",
            &address.to_string(),
            "--listen-data",
            &free_address().to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !socket.exists() {
        assert!(Instant::now() < deadline, "the manager never started");
        thread::sleep(Duration::from_millis(20));
    }
    // The warning is printed while the manager starts up; stop it only once it has been
    // seen (or give up with what was printed), not when the socket first appears.
    let mut child = child;
    let stderr = child.stderr.take().unwrap();
    let (lines, seen) = std::sync::mpsc::channel();
    thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stderr)
            .lines()
            .map_while(|l| l.ok())
        {
            if lines.send(line).is_err() {
                return;
            }
        }
    });
    let mut text = String::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !(text.contains("warning: network listening enabled on") && text.contains("plaintext"))
        && Instant::now() < deadline
    {
        if let Ok(line) = seen.recv_timeout(Duration::from_millis(100)) {
            text.push_str(&line);
            text.push('\n');
        }
    }
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let _ = child.wait();
    while let Ok(line) = seen.recv_timeout(Duration::from_millis(200)) {
        text.push_str(&line);
        text.push('\n');
    }
    assert!(
        text.contains("warning: network listening enabled on") && text.contains("plaintext"),
        "{text}"
    );
    assert!(!text.contains("RDMA"), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}
