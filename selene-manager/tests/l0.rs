mod support;
use selene_manager::{
    control, disk,
    proto::v2::{self, client_message as c, server_message::M},
    record,
    store::Config,
};
use std::{
    fs,
    net::TcpListener,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use support::*;

const CHILD: &str = "SELENE_CHILD_BASE";
const CHILD_POINT: &str = "SELENE_CHILD_POINT";
const CHILD_DUMP: &str = "SELENE_CHILD_DUMP";
const CHILD_QUARANTINE: &str = "SELENE_CHILD_QUARANTINE";

/// Polls until the manager's entry for `tokens` is offered from `tier`.
async fn wait_tier(client: &mut Client, tokens: u64, tier: v2::Tier) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if client
            .candidate(tokens)
            .await
            .is_some_and(|candidate| candidate.tier == tier as i32)
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the entry for {tokens} never moved to {tier:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// selene.v2 pushes no Room: a put refused for budget is simply tried again, and is
/// admitted once the budget has refilled or the shortage cleared. A debt of several
/// megabytes takes about fifteen seconds to repay.
async fn store_when_room(
    client: &mut Client,
    key: selene_manager::Key,
    parent: Option<selene_manager::Key>,
) -> Vec<u8> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match client.put(key, parent, PutOptions::default()).await {
            PutOutcome::Stored(bytes) => return bytes,
            PutOutcome::Refused(reason) if reason == v2::Reason::Budget as u8 => {
                assert!(Instant::now() < deadline, "the put was never admitted");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            other => panic!("unexpected put outcome: {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_03_accept_shapes_and_present_inflight() {
    let mut host = Host::with_storage(4 * 1024 * 1024, 4 * 1024 * 1024);
    host.run();
    let mut a = host.client().await;
    let mut b = host.client().await;
    // A second writer sees INFLIGHT while the first put is open.
    let (open, start) = a.start_put(key(128), None).await;
    assert!(matches!(start, PutStart::Accepted(_)), "{start:?}");
    assert_eq!(
        b.put(key(128), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Inflight as u8)
    );
    a.abandon_put(open).await;
    host.wait(|host| host.gauge("open_ops") == 0);
    let bytes = b.store(key(128), None).await;
    assert!(!bytes.is_empty());
    assert_eq!(
        b.put(key(128), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Present as u8)
    );
    let mut c = host.client().await;
    let mut open = Vec::new();
    for tokens in [640, 768, 896, 1024] {
        let (id, start) = c.start_put(key(tokens), None).await;
        assert!(
            matches!(start, PutStart::Accepted(_)),
            "{tokens}: {start:?}"
        );
        open.push(id);
    }
    assert_eq!(
        c.put(key(1152), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Queue as u8)
    );
    for id in open {
        c.abandon_put(id).await;
    }
    host.wait(|host| host.gauge("open_ops") == 0);
    b.store(key(256), Some(key(128))).await;
    b.store(key(512), None).await;
    let mut fresh = host.client().await;
    for (index, tokens) in [3u64, 5].into_iter().enumerate() {
        let bytes = fresh.store(key(tokens), (index > 0).then(|| key(3))).await;
        assert!(!bytes.is_empty());
    }
    // Two row-only records: a restore supplies the endpoint's row and nothing for the
    // parent, which has no page and whose row is not wanted.
    let data = fresh.fetch(&[key(3), key(5)], 0, 0).await;
    assert_eq!(data.len(), 1);
    assert_eq!(data[0].len() as u64, descriptor().row_bytes());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_04_invalid_checksum_short_payload_and_offsets() {
    let mut host = Host::with_storage(4 * 1024 * 1024, 4 * 1024 * 1024);
    host.run();
    for (options, tokens) in [
        (
            PutOptions {
                corrupt_page: Some(0),
                ..PutOptions::default()
            },
            128,
        ),
        (
            PutOptions {
                early_commit: true,
                ..PutOptions::default()
            },
            128 * 64,
        ),
        (
            PutOptions {
                corrupt_row: true,
                ..PutOptions::default()
            },
            256,
        ),
        (
            PutOptions {
                skip_filled: true,
                ..PutOptions::default()
            },
            192 * 64,
        ),
    ] {
        let mut client = host.client().await;
        let outcome = client.put(key(tokens), None, options).await;
        assert!(
            matches!(outcome, PutOutcome::Refused(reason) if reason == v2::Reason::Invalid as u8),
            "{tokens}: {outcome:?}"
        );
    }
    host.wait(|host| host.gauge("nvme_reserved_bytes") == 0);
    assert_eq!(host.gauge("staging_allocated_bytes"), 0);
    // Nothing the refused puts carried was indexed.
    let mut check = host.client().await;
    assert!(check.records().await.is_empty());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_05_parent_first_chain_closed_durability() {
    let base = base("durable");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4 * 1024 * 1024,
        4 * 1024 * 1024,
    );
    config.name = instance_name("durable");
    config.faults.pause_at = Some("parent_write".into());
    let signal = base.join("gate");
    config.faults.signal = Some(signal.clone());
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    client.store(key(128), None).await;
    wait(|| signal.exists());
    client.store(key(256), Some(key(128))).await;
    let namespace = client.descriptor.name();
    assert!(
        !host
            .config
            .root
            .as_ref()
            .unwrap()
            .join(&namespace)
            .join(record::file_name(&key(256)))
            .exists()
    );
    assert_eq!(host.gauge("nvme_bytes_total"), 0);
    fs::remove_file(&signal).unwrap();
    wait(|| signal.exists());
    assert!(
        host.config
            .root
            .as_ref()
            .unwrap()
            .join(&namespace)
            .join(record::file_name(&key(128)))
            .exists()
    );
    fs::remove_file(signal).unwrap();
    host.durable(2);
    // The catalog's durable flag is not on the wire: both records being on disk, the
    // parent having been written first, is the chain-durable state.
    for value in [key(128), key(256)] {
        assert!(
            host.config
                .root
                .as_ref()
                .unwrap()
                .join(&namespace)
                .join(record::file_name(&value))
                .exists(),
            "both records must be chain-durable"
        );
    }
}
#[test]
fn l0_06_multiprocess_kill_recovery() {
    if let Ok(base) = std::env::var(CHILD) {
        let base = PathBuf::from(base);
        let point = std::env::var(CHILD_POINT).unwrap_or_default();
        let memory = if point == "parent_write" {
            4 * 1024 * 1024
        } else {
            4096
        };
        let mut config = Config::new(
            base.join("records"),
            base.join("run/m.sock"),
            memory,
            64 * 1024 * 1024,
        );
        config.name = instance_name("child");
        config.faults.pause_at = Some(point.clone());
        config.faults.signal = Some(base.join("gate"));
        if let Ok(dump) = std::env::var(CHILD_DUMP) {
            let store = selene_manager::store::Store::open(config).unwrap_or_else(|error| {
                panic!("the snapshot child could not open the store: {error}")
            });
            let snapshot = store
                .namespace_id()
                .and_then(|id| store.snapshot(id))
                .map(|entries| entries.len())
                .unwrap_or(0);
            let disk_bytes: u64 = store
                .metrics_text()
                .lines()
                .filter(|line| line.starts_with("selene_bytes{") && line.contains("tier=\"disk\""))
                .map(|line| line.rsplit(' ').next().unwrap().parse::<u64>().unwrap())
                .sum();
            fs::write(dump, format!("{snapshot} {disk_bytes}")).unwrap();
            loop {
                std::thread::park();
            }
        }
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let mut host = Host::open_config(config);
            host.run();
            let mut client = host.client().await;
            if point == "data" {
                let _ = client
                    .put(
                        key(128),
                        None,
                        PutOptions {
                            cancel: Some(CancelAt::Filled),
                            ..PutOptions::default()
                        },
                    )
                    .await;
            } else {
                client.store(key(128), None).await;
            }
            loop {
                std::thread::park();
            }
        });
    }
    for point in ["data", "before_rename", "after_rename", "parent_write"] {
        let base = base(point);
        let config = Config::new(
            base.join("records"),
            base.join("run/m.sock"),
            4096,
            64 * 1024 * 1024,
        );
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "l0_06_multiprocess_kill_recovery", "--nocapture"])
            .env(CHILD, &base)
            .env(CHILD_POINT, point)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        wait(|| base.join("gate").exists());
        child.kill().unwrap();
        let status = child.wait().unwrap();
        remove_children_shm(child.id());
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        let lock = disk::RootLock::acquire(&config.root.as_ref().unwrap().clone()).unwrap();
        let recovered = recovered_keys(config.root.as_ref().unwrap());
        assert_eq!(
            recovered,
            if point == "after_rename" {
                vec![key(128)]
            } else {
                vec![]
            },
            "{point}"
        );
        assert!(
            fs::read_dir(config.root.as_ref().unwrap())
                .unwrap()
                .map(|namespace| namespace.unwrap().path())
                .filter(|path| path.is_dir())
                .flat_map(|path| fs::read_dir(path).unwrap())
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".tmp"))
        );
        drop(lock);
        let _ = fs::remove_file(base.join("gate"));
        let dump = base.join("dump");
        let mut restarted = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "l0_06_multiprocess_kill_recovery", "--nocapture"])
            .env(CHILD, &base)
            .env(CHILD_DUMP, &dump)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        // `exists` can be true before the child has written the snapshot, so wait for a
        // value the file actually parses to.
        let mut snapshot = None;
        wait(|| {
            snapshot = fs::read_to_string(&dump)
                .ok()
                .and_then(|text| {
                    let (count, bytes) = text.split_once(' ')?;
                    Some((count.parse::<usize>().ok()?, bytes.parse::<u64>().ok()?))
                });
            snapshot.is_some()
        });
        let persisted = if recovered.is_empty() {
            0
        } else {
            content_for(&descriptor(), key(128), None, 0).0.length_with_row()
        };
        assert_eq!(snapshot.unwrap(), (recovered.len(), persisted));
        restarted.kill().unwrap();
        restarted.wait().unwrap();
        remove_children_shm(restarted.id());
        fs::remove_dir_all(base).unwrap();
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_07_streaming_and_nvme_oversize() {
    let mut host = Host::with_storage(65536, 2 * 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    let bytes = client.store(key(128 * 64), None).await;
    host.durable(1);
    assert_eq!(host.gauge("memory_allocated_bytes"), 0);
    let data = client.fetch(&[key(128 * 64)], 0, 128).await;
    assert_eq!(data[0], bytes);
    assert_eq!(
        client
            .put(key(1024 * 64), None, PutOptions::default())
            .await,
        PutOutcome::Refused(v2::Reason::Oversize as u8)
    );
    drop(client);
    host.stop();
    host.run();
    let mut client = host.client().await;
    assert_eq!(client.wait_records(1).await.len(), 1);
    // The 64 pages already held are not sent again.
    let held = 64 * descriptor().page_bytes() as usize;
    let data = client.fetch(&[key(128 * 64)], 64, 64).await;
    assert_eq!(data[0], bytes[held..]);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_08_refill_and_split_units() {
    let mut refill = Host::with_storage(65536, 1024 * 1024);
    refill.run();
    let mut client = refill.client().await;
    client.store(key(128), None).await;
    refill.durable(1);
    drop(client);
    refill.stop();
    refill.run();
    let mut client = refill.client().await;
    let candidate = client
        .wait_candidate(128, Duration::from_secs(15))
        .await
        .expect("the recovered record is offered");
    assert_eq!(candidate.tier, v2::Tier::Nvme as i32);
    client.fetch(&[key(128)], 0, 2).await;
    refill.wait(|host| host.gauge("memory_allocated_bytes") > 0);
    wait_tier(&mut client, 128, v2::Tier::Dram).await;

    let mut split = Host::with_storage(4096, 32 * 1024 * 1024);
    split.run();
    let mut geometry = descriptor();
    geometry.ranks[0].planes = vec![4 * 1024 * 1024 + 4096];
    let mut client = split.client_with(geometry).await.unwrap();
    let expected = client.store(key(128), None).await;
    let actual = client.fetch(&[key(128)], 0, 2).await;
    assert_eq!(actual[0], expected);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_09_disk_failure_is_refused_io() {
    let base = base("full");
    fs::create_dir_all(&base).unwrap();
    let full = base.join("full");
    fs::write(&full, b"full").unwrap();
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        64 * 1024 * 1024,
    );
    config.name = instance_name("full");
    config.faults.disk_full = Some(full.clone());
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    // The record does not fit the tiny arena, so it streams through staging.
    assert_eq!(
        client.put(key(512 * 64), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Io as u8)
    );
    host.wait(|host| host.gauge("nvme_errors_total") > 0);
    // A single-window stream must also get its IO refusal: the failing chunk is
    // the last pending user, so removing the operation first would lose it.
    let mut single = host.client().await;
    assert_eq!(
        single.put(key(64 * 64), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Io as u8)
    );
    fs::remove_file(full).unwrap();
    let bytes = client.store(key(128), None).await;
    assert!(!bytes.is_empty());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_10_cancel_drains_and_staging_gauge() {
    let base = base("cancel");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        4 * 1024 * 1024,
    );
    config.name = instance_name("cancel");
    config.faults.delay_ms = 300;
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    // Streaming put: two pages plus the row, one window of 12288 bytes.
    assert_eq!(
        client
            .put(
                key(128),
                None,
                PutOptions {
                    cancel: Some(CancelAt::Filled),
                    ..PutOptions::default()
                },
            )
            .await,
        PutOutcome::Cancelled
    );
    // The fixed hold for the two-page record (2 x a 16 KiB window) lives
    // until its delayed disk write completes, then drains.
    host.wait(|host| host.gauge("staging_allocated_bytes") > 0);
    assert_eq!(host.gauge("staging_allocated_bytes"), 32768);
    host.wait(|host| host.gauge("staging_allocated_bytes") == 0);
    host.wait(|host| host.gauge("nvme_reserved_bytes") == 0);
    // A restore is not blocked by a draining put.
    let stored = client.store(key(128 * 64), None).await;
    let mut getter = host.client().await;
    let start = Instant::now();
    let data = getter.fetch(&[key(128 * 64)], 0, 128).await;
    assert!(start.elapsed() < Duration::from_millis(500));
    assert_eq!(data[0], stored);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_12_round_robin_and_dead_client_cleanup() {
    let mut host = Host::with_storage(8 * 1024 * 1024, 64 * 1024 * 1024);
    host.run();
    let mut a = host.client().await;
    let mut client = host.client().await;
    let mut previous = None;
    for index in 1..=160u64 {
        let value = key(index * 64);
        client.store(value, previous).await;
        previous = Some(value);
    }
    host.durable(160);
    let keys: Vec<_> = (1..=160).map(|index| key(index * 64)).collect();
    // Four requesters that stop in the middle of a 160-record restore.
    for _ in 0..4 {
        let mut dead = host.client().await;
        let held = dead.hold(&keys, 0, 160).await;
        assert!(!held.windows.is_empty());
        dead.goodbye().await;
    }
    let start = Instant::now();
    a.fetch(&[key(64)], 0, 1).await;
    assert!(start.elapsed() < Duration::from_millis(500));
    host.wait(|host| host.gauge("leases") == 0 && host.gauge("cleanup_records") == 0);
    // A release for a claim nobody made changes nothing for the session or its live claim.
    let mut unknown = host.client().await;
    let held = unknown.hold(&[key(64)], 0, 1).await;
    unknown
        .session
        .send(c::M::WindowReleased(v2::WindowReleased {
            claim_id: 999_999,
            index: 0,
        }))
        .await;
    unknown.session.barrier().await;
    let start = Instant::now();
    a.fetch(&[key(64)], 0, 1).await;
    assert!(start.elapsed() < Duration::from_millis(500));
    unknown.finish(held).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_18_restart_recovers_every_committed_record_without_a_descriptor() {
    let mut host = Host::with_storage(64 * 1024, 4 * 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    let mut expected = Vec::new();
    for tokens in [128, 256, 512] {
        expected.push((key(tokens), client.store(key(tokens), None).await));
    }
    let mut other = descriptor();
    other.identity = [9; 32];
    let mut second = host.client_with(other.clone()).await.unwrap();
    let second_bytes = second.store(key(64), None).await;
    host.durable(4);
    let root = host.config.root.clone().unwrap();
    let directory = root.join(client.descriptor.name());
    assert!(!directory.join("descriptor").exists());
    for (key, _) in &expected {
        let bytes = fs::read(directory.join(record::file_name(key))).unwrap();
        assert_eq!(&bytes[..8], b"FLARECKP");
    }
    let reported = |host: &Host| -> u64 {
        host.metrics()
            .lines()
            .filter(|line| line.starts_with("selene_bytes{") && line.contains("tier=\"disk\""))
            .map(|line| line.rsplit(' ').next().unwrap().parse::<u64>().unwrap())
            .sum()
    };
    let persisted: u64 = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .flat_map(|path| fs::read_dir(path).unwrap())
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "ckpt"))
        .map(|entry| entry.metadata().unwrap().len())
        .sum();
    assert_eq!(reported(&host), persisted);
    drop(client);
    drop(second);
    host.stop();
    let broken = directory.join(record::file_name(&key(1024)));
    fs::write(&broken, b"broken").unwrap();
    fs::write(directory.join(".interrupted.tmp"), b"partial").unwrap();
    host.run();
    assert_eq!(reported(&host), persisted);
    let mut client = host.client().await;
    assert_eq!(client.records().await.len(), expected.len());
    for (key, bytes) in expected {
        assert_eq!(
            client
                .fetch(&[key], 0, (key.consumed_tokens / 64) as u32)
                .await[0],
            bytes
        );
    }
    let mut second = host.client_with(other).await.unwrap();
    assert_eq!(second.records().await.len(), 1);
    assert_eq!(second.fetch(&[key(64)], 0, 1).await[0], second_bytes);
    assert!(!broken.exists());
    assert!(!directory.join(".interrupted.tmp").exists());
    assert_eq!(reported(&host), persisted);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_19_two_managers_one_name() {
    let mut host = Host::with_storage(1024 * 1024, 1024 * 1024);
    host.run();
    let base = base("second");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        1024 * 1024,
        1024 * 1024,
    );
    config.name = host.config.name.clone();
    let second = Host::open_config(config);
    let error = control::serve(second.config.clone(), Arc::new(AtomicBool::new(false)))
        .unwrap_err()
        .to_string();
    assert!(error.contains(&host.config.name), "{error}");
    assert!(
        error.contains(&host.config.socket.display().to_string()),
        "{error}"
    );
    assert!(
        error.contains(&format!("pid {}", std::process::id())),
        "{error}"
    );
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_21_live_disconnected_client_quarantine() {
    if let Ok(socket) = std::env::var(CHILD_QUARANTINE) {
        let mut config = Config::dram_only(PathBuf::from(&socket), 32768);
        config.name = instance_name("quarantine-child");
        let mut client = Client::connect(&config, descriptor()).await.unwrap();
        let _ = client
            .get(
                &[key(128)],
                0,
                2,
                GetOptions {
                    cancel: Some(CancelAt::Grant),
                    leave_leased: true,
                },
            )
            .await;
        drop(client);
        fs::write(PathBuf::from(&socket).with_extension("ready"), b"ready").unwrap();
        loop {
            std::thread::park();
        }
    }
    let mut host = Host::dram_only(32768);
    host.run();
    let mut client = host.client().await;
    client.store(key(128), None).await;
    assert!(host.gauge("memory_allocated_bytes") > 0);
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "l0_21_live_disconnected_client_quarantine",
            "--nocapture",
        ])
        .env(CHILD_QUARANTINE, &host.config.socket)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let ready = host.config.socket.with_extension("ready");
    wait(|| ready.exists());
    // The child's stream is gone but the process lives: the lease is quarantined.
    host.wait(|host| host.gauge("memory_allocated_bytes") > 0);
    let mut probe = host.client().await;
    assert_eq!(
        probe.put(key(256), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Budget as u8)
    );
    child.kill().unwrap();
    child.wait().unwrap();
    remove_children_shm(child.id());
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match probe.put(key(256), None, PutOptions::default()).await {
            PutOutcome::Stored(_) => break,
            PutOutcome::Refused(reason) if reason == v2::Reason::Budget as u8 => {
                assert!(Instant::now() < deadline, "quarantine was not reclaimed");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            other => panic!("unexpected put outcome after exit: {other:?}"),
        }
    }
    fs::remove_file(ready).ok();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_23_commit_waits_for_durability() {
    let base = base("commit");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        64 * 1024 * 1024,
    );
    config.name = instance_name("commit");
    config.faults.pause_at = Some("before_rename".into());
    let signal = base.join("gate");
    config.faults.signal = Some(signal.clone());
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    let stored = Arc::new(AtomicBool::new(false));
    let flag = stored.clone();
    let task = tokio::spawn(async move {
        let outcome = client.put(key(128), None, PutOptions::default()).await;
        flag.store(true, Ordering::Release);
        outcome
    });
    wait(|| signal.exists());
    assert!(!stored.load(Ordering::Acquire), "Stored before durability");
    assert_eq!(host.durable_records(), 0);
    fs::remove_file(signal).unwrap();
    assert!(matches!(task.await.unwrap(), PutOutcome::Stored(_)));
    host.durable(1);
}
#[test]
fn l0_24_signal_handling() {
    let base = base("signal");
    fs::create_dir_all(base.join("run")).unwrap();
    let socket = base.join("run/manager.sock");
    let name = instance_name("signal");
    let _shm = support::ShmCleanup::new(&name, &socket);
    let mut child = Command::new(env!("CARGO_BIN_EXE_selene"))
        .args([
            "--name",
            &name,
            "--socket",
            socket.to_str().unwrap(),
            "--memory",
            "8M",
            "--staging",
            "8M",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    wait(|| socket.exists());
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "{status:?}");
    assert!(!socket.exists());
    let _ = fs::remove_dir_all(base);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_25_verify_before_window_reuse() {
    let mut host = Host::with_storage(4096, 4 * 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    let outcome = client
        .put(
            key(128),
            None,
            PutOptions {
                corrupt_page: Some(0),
                ..PutOptions::default()
            },
        )
        .await;
    assert_eq!(outcome, PutOutcome::Refused(v2::Reason::Invalid as u8));
    host.wait(|host| host.gauge("staging_allocated_bytes") == 0);
    let mut fresh = host.client().await;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match fresh.put(key(128), None, PutOptions::default()).await {
            PutOutcome::Stored(_) => break,
            PutOutcome::Refused(reason) if reason == v2::Reason::Inflight as u8 => {
                assert!(Instant::now() < deadline, "inflight did not drain");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_27_no_progress_timeout_ends_the_call() {
    let mut host = Host::with_storage(1024 * 1024, 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    client.store(key(128), None).await;
    let (id, start) = client.start_put(key(256), None).await;
    assert!(matches!(start, PutStart::Accepted(_)), "{start:?}");
    // The writer stalls; the manager ends the put when progress stops.
    let begun = Instant::now();
    let refused = client
        .session
        .take_within(Duration::from_secs(8), &mut |m| match m {
            M::PutRefused(refused) if refused.put_id == id => Some(refused.clone()),
            _ => None,
        })
        .await
        .expect("the put did not end on the no-progress timer");
    assert_eq!(refused.reason, v2::Reason::Invalid as i32);
    assert!(begun.elapsed() >= Duration::from_secs(2));
    assert!(begun.elapsed() < Duration::from_secs(8));
    client.fence_put(id).await;
    let mut fresh = host.client().await;
    assert!(!fresh.fetch(&[key(128)], 0, 2).await[0].is_empty());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_30_committed_detached_put_finishes() {
    let base = base("detached");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        64 * 1024 * 1024,
    );
    config.name = instance_name("detached");
    config.faults.delay_ms = 100;
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    let outcome = client
        .put(
            key(512 * 64),
            None,
            PutOptions {
                abort_after_commit: true,
                ..PutOptions::default()
            },
        )
        .await;
    assert_eq!(outcome, PutOutcome::Abandoned);
    drop(client);
    host.wait(|host| host.durable_records() == 1);
    let mut check = host.client().await;
    assert!(
        check
            .wait_candidate(512 * 64, Duration::from_secs(5))
            .await
            .is_some()
    );
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_31_timed_out_publication_finishes_instead_of_orphaning() {
    let base = base("timeout-publish");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        64 * 1024 * 1024,
    );
    config.name = instance_name("timeout-publish");
    config.faults.pause_at = Some("before_rename".into());
    let signal = base.join("gate");
    config.faults.signal = Some(signal.clone());
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    let task = tokio::spawn(async move { client.put(key(128), None, PutOptions::default()).await });
    let deadline = Instant::now() + Duration::from_secs(10);
    while !signal.exists() {
        assert!(
            Instant::now() < deadline,
            "publication never reached before_rename"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let outcome = tokio::time::timeout(Duration::from_secs(10), task)
        .await
        .expect("the publication did not time out")
        .unwrap();
    assert_eq!(host.durable_records(), 0);
    fs::remove_file(&signal).unwrap();
    assert_eq!(outcome, PutOutcome::Refused(v2::Reason::Invalid as u8));
    let deadline = Instant::now() + Duration::from_secs(10);
    while host.durable_records() != 1 {
        assert!(Instant::now() < deadline, "publication did not complete");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut check = host.client().await;
    assert!(
        check
            .wait_candidate(128, Duration::from_secs(5))
            .await
            .is_some()
    );
    // Recovery must find exactly the record the index announced.
    drop(check);
    host.stop();
    host.run();
    let mut after = host.client().await;
    let records = after.wait_records(1).await;
    assert_eq!(records[0].key.as_ref().unwrap().consumed_tokens, 128);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_34_concurrent_nvme_gets_share_connection_credits() {
    let mut host = Host::with_storage(4096, 64 * 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    let one = client.store(key(64 * 64), None).await;
    let two = client.store(key(128 * 64), None).await;
    let (left, right) = client
        .fetch_two((vec![key(64 * 64)], 0, 64), (vec![key(128 * 64)], 0, 128))
        .await;
    match (left, right) {
        (GetOutcome::Done(records), GetOutcome::Done(other)) => {
            assert_eq!(records[0], one);
            assert_eq!(other[0], two);
        }
        other => panic!("concurrent restores failed: {other:?}"),
    }
    host.wait(|host| {
        host.gauge("leases") == 0
            && host.gauge("cleanup_records") == 0
            && host.gauge("open_ops") == 0
    });
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_40_detached_finalization_uses_the_put_namespace() {
    let base = base("detached-ns");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        64 * 1024 * 1024,
    );
    config.name = instance_name("detached-ns");
    config.faults.delay_ms = 50;
    let mut host = Host::open_config(config);
    host.run();
    let mut first = host.client().await;
    let mut other_descriptor = descriptor();
    other_descriptor.fingerprint = [9; 16];
    let other = host.client_with(other_descriptor.clone()).await.unwrap();
    // Parent in the first namespace, then a child whose writer goes away
    // after commit. Finalization must use the put's namespace, not an
    // arbitrary one.
    first.store(key(64), None).await;
    host.wait(|host| host.durable_records() == 1);
    let outcome = first
        .put(
            key(576 * 64),
            Some(key(64)),
            PutOptions {
                abort_after_commit: true,
                ..PutOptions::default()
            },
        )
        .await;
    assert_eq!(outcome, PutOutcome::Abandoned);
    drop(first);
    host.wait(|host| host.durable_records() == 2);
    drop(other);
    host.stop();
    host.run();
    let mut first = host.client().await;
    let records = first.wait_records(2).await;
    let tokens: Vec<u64> = records
        .iter()
        .map(|record| record.key.as_ref().unwrap().consumed_tokens)
        .collect();
    assert!(tokens.contains(&64));
    assert!(tokens.contains(&(576 * 64)));
    let mut other = host.client_with(other_descriptor.clone()).await.unwrap();
    assert!(other.records().await.is_empty());
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_42_cancel_before_dispatch_withdraws_the_put() {
    let mut host = Host::with_storage(1024 * 1024, 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    let id = client.send_begin(key(128), None).await;
    client.cancel_put(id).await;
    client.fence_put(id).await;
    host.wait(|host| host.gauge("open_ops") == 0);
    let mut fresh = host.client().await;
    // The withdrawn put must not leave INFLIGHT behind.
    let _ = fresh.store(key(128), None).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_43_failed_get_drains_release_every_pin() {
    let mut host = Host::with_storage(4096, 64 * 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    client.store(key(128), None).await;
    client.store(key(256), Some(key(128))).await;
    host.durable(2);
    let directory = host
        .config
        .root
        .as_ref()
        .unwrap()
        .join(client.descriptor.name());
    for value in [key(128), key(256)] {
        fs::write(directory.join(record::file_name(&value)), b"").unwrap();
    }
    // Two streamed records mean two reads are in flight; the first failure
    // starts the drain, and the second completion must decrement it.
    let outcome = client
        .get(&[key(128), key(256)], 0, 4, GetOptions::default())
        .await;
    assert_eq!(
        outcome,
        GetOutcome::Refused(v2::ClaimReason::CopyError),
        "{outcome:?}"
    );
    host.wait(|host| host.gauge("cleanup_records") == 0 && host.gauge("leases") == 0);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_52_hold_slots_stay_busy_until_writes_drain() {
    let base = base("hold-drain");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        4096,
        4 * 1024 * 1024,
    );
    config.name = instance_name("hold-drain");
    let (gate, entered, release) = selene_manager::disk::IoGate::new();
    config.faults.io_gate = Some(gate);
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    let record = key(192 * 64);
    let (id, start) = client.start_put(record, None).await;
    let PutStart::Accepted(accept) = start else {
        panic!("put refused")
    };
    let first = accept.first_page as u32;
    let (header, bytes, _) = content_for(&client.descriptor, record, None, first);
    let page_bytes = header.page_record_bytes() as usize;
    for grant in 0..3 {
        let window = client
            .session
            .take(|m| match m {
                M::PutGrant(g) if g.put_id == id => g.window,
                _ => None,
            })
            .await;
        let start =
            header.page_segment as usize + (window.first_page - first) as usize * page_bytes;
        client
            .session
            .shm
            .as_ref()
            .unwrap()
            .write(&window, &bytes[start..start + window.length as usize]);
        let pages = (window.first_page - first) as usize;
        let page_xxh3 = (pages..pages + window.page_count as usize)
            .map(|page| {
                let at = header.page_segment as usize + page * page_bytes;
                flare_selene::shared::record::checksum(&bytes[at..at + page_bytes])
            })
            .collect();
        client
            .session
            .send(c::M::PutFilled(v2::PutFilled {
                put_id: id,
                index: window.index,
                page_xxh3,
                row_xxh3: if window.row {
                    flare_selene::shared::record::checksum(&bytes[header.row_segment as usize..])
                } else {
                    0
                },
                has_row: window.row,
                ..Default::default()
            }))
            .await;
        if grant == 1 {
            // The actor has processed both fills while the first write still owns its slot.
            client.session.barrier().await;
            entered.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(
                !client
                    .session
                    .backlog
                    .iter()
                    .any(|m| matches!(m, M::PutGrant(g) if g.put_id == id)),
                "a third window was granted before a write drained"
            );
            release.send(()).unwrap();
        }
    }
    client
        .session
        .send(c::M::PutCommit(v2::PutCommit {
            put_id: id,
            metadata_offset: 0,
            metadata: bytes[..header.page_segment as usize].to_vec(),
        }))
        .await;
    client
        .session
        .take(|m| match m {
            M::Stored(stored) if stored.put_id == id => Some(()),
            _ => None,
        })
        .await;
    let fetched = client.fetch(&[record], 0, 192).await;
    assert_eq!(fetched[0], bytes[header.page_segment as usize..]);
    host.wait(|host| host.gauge("open_ops") == 0);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_53_room_reconsiders_the_arena_after_a_lease_releases() {
    // 16 MiB arena, 8 MiB staging. An 8 MiB page record needs a 16 MiB
    // arena extent but its streaming hold would be 16 MiB. While the
    // resident record is leased the put is refused; after the lease
    // releases, the arena must be reconsidered and the put must complete.
    let base = base("room-arena-retry");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        16 * 1024 * 1024,
        64 * 1024 * 1024,
    );
    config.name = instance_name("room-arena-retry");
    config.staging = 8 * 1024 * 1024;
    let mut host = Host::open_config(config);
    host.run();
    let mut geometry = descriptor();
    geometry.ranks[0].planes = vec![8 * 1024 * 1024];
    let mut resident = host.client_with(geometry.clone()).await.unwrap();
    let stored = resident.store(key(64), None).await;
    host.durable(1);
    // A second namespace (same geometry) for the put that has to wait.
    let mut other = geometry;
    other.identity = [9; 32];
    other.fingerprint = [4; 16];
    let mut waiter = host.client_with(other).await.unwrap();
    // Lease the resident record and hold the windows (the 8 MiB page and
    // the row arrive as separate windows).
    let held = resident.hold(&[key(64)], 0, 1).await;
    assert!(!held.windows.is_empty(), "lease granted no window");
    assert_eq!(
        waiter.put(key(64), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Budget as u8)
    );
    assert_eq!(
        waiter.put(key(64), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Budget as u8),
        "the arena record was still leased"
    );
    resident.finish(held).await;
    let stored_again = store_when_room(&mut waiter, key(64), None).await;
    let fetched = waiter.fetch(&[key(64)], 0, 1).await;
    assert_eq!(fetched[0], stored_again);
    assert!(!stored.is_empty());
    host.wait(|host| host.gauge("open_ops") == 0);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_54_get_sources_survive_eviction_pressure() {
    // Diagnostic: a restore's delivered bytes must be the bytes the manager
    // verified when it read the record. Hold leases on every arena record,
    // force a streamed put, release the leases so the read refills the
    // arena, and apply put pressure while the slow read is in flight. Then
    // repeat with a record too large for the arena so the header and the
    // payload windows are read at different times.
    let base = base("get-pressure");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        32 * 1024,
        4 * 1024 * 1024,
    );
    config.name = instance_name("get-pressure");
    let (gate, entered, release) = selene_manager::disk::IoGate::new();
    config.faults.read_gate = Some(gate.clone());
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    let first = client.store(key(64), None).await;
    let second = client.store(key(128), Some(key(64))).await;
    host.durable(2);
    // The second record alone: its parent's page is already held.
    let one = client.hold(&[key(64)], 0, 1).await;
    let two = client.hold(&[key(128)], 1, 1).await;
    assert!(!one.windows.is_empty() && !two.windows.is_empty());
    // Both arena records are pinned, so this three-page root streams to disk.
    let streamed = client.store(key(192), None).await;
    client.finish(one).await;
    client.finish(two).await;
    // The arena is now free for the read to verify and refill the record.
    let mut reader = host.client().await;
    let mut pressure = host.client().await;
    let streamed_key = [key(192)];
    let fetch = reader.fetch(&streamed_key, 0, 3);
    let put = async {
        let entered = tokio::task::spawn_blocking(move || {
            entered
                .recv_timeout(Duration::from_secs(5))
                .expect("the restore never reached disk");
            entered
        })
        .await
        .unwrap();
        let put = tokio::spawn(async move {
            let bytes = pressure.store(key(256), None).await;
            (pressure, bytes)
        });
        host.wait(|host| host.gauge("open_ops") >= 2);
        release.send(()).unwrap();
        let (pressure, bytes) = put.await.unwrap();
        (pressure, bytes, entered)
    };
    let (fetched, (returned, pressed, entered)) = tokio::join!(fetch, put);
    pressure = returned;
    assert_eq!(fetched[0], streamed);
    assert!(!pressed.is_empty());
    assert!(!first.is_empty() && !second.is_empty());

    // 128 pages do not fit the arena: the header is read once and each
    // payload window later, while another put adds eviction pressure.
    let large = client.store(key(128 * 64), None).await;
    gate.arm();
    let large_key = [key(128 * 64)];
    let fetch = reader.fetch(&large_key, 0, 128);
    let put = async {
        tokio::task::spawn_blocking(move || {
            entered
                .recv_timeout(Duration::from_secs(5))
                .expect("the restore never reached disk");
        })
        .await
        .unwrap();
        let put = tokio::spawn(async move {
            let bytes = pressure.store(key(384), None).await;
            (pressure, bytes)
        });
        host.wait(|host| host.gauge("open_ops") >= 2);
        release.send(()).unwrap();
        put.await.unwrap()
    };
    let (fetched, (returned, pressed)) = tokio::join!(fetch, put);
    drop(returned);
    assert_eq!(fetched[0], large);
    assert!(!pressed.is_empty());
    host.wait(|host| host.gauge("open_ops") == 0);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_55_cancelled_get_lease_keeps_its_bytes() {
    // Diagnostic: after a claim is cancelled its tree pins are released, but
    // the requester still holds the granted window. Eviction pressure must not
    // free or reuse that arena extent before the requester reads it.
    let mut host = Host::with_storage(32 * 1024, 4 * 1024 * 1024);
    host.run();
    let mut client = host.client().await;
    let stored = client.store(key(64), None).await;
    host.durable(1);
    let held = client.hold(&[key(64)], 0, 1).await;
    let window = held.windows[0].window.unwrap();
    assert!(window.page_count > 0 && window.first_page == 0);
    // Cancelled; the window is neither released nor fenced.
    let terminal = client.drop_held(&held).await;
    assert_eq!(terminal.state, v2::ClaimState::Failed as i32);
    // Pressure: a 32 KiB root cannot fit beside the leased 16 KiB extent, so
    // eviction must either skip the lease or corrupt the requester's window.
    let pressed = client.store(key(128), None).await;
    assert!(!pressed.is_empty());
    let bytes = client.session.shm.as_ref().unwrap().read(&window);
    assert_eq!(bytes, stored[..window.length as usize]);
    host.wait(|host| host.gauge("open_ops") == 0);
}
#[test]
fn l0_57_sigkill_never_blocks_a_restart_on_the_same_instance() {
    let base = base("sigkill-restart");
    fs::create_dir_all(base.join("run")).unwrap();
    let socket = base.join("run/manager.sock");
    let name = instance_name("sigkill-restart");
    let _shm = support::ShmCleanup::new(&name, &socket);
    let spawn = |stderr: &str| {
        Command::new(env!("CARGO_BIN_EXE_selene"))
            .args([
                "--name",
                &name,
                "--socket",
                socket.to_str().unwrap(),
                "--memory",
                "8M",
                "--staging",
                "8M",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::from(fs::File::create(base.join(stderr)).unwrap()))
            .spawn()
            .unwrap()
    };

    let mut first = spawn("first.err");
    wait(|| socket.exists());
    unsafe {
        libc::kill(first.id() as i32, libc::SIGKILL);
    }
    let status = first.wait().unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(libc::SIGKILL));
    // The killed manager leaves its socket, lock and region files behind. The
    // flock alone decides ownership: the directory's existence must not refuse
    // the restart, and the stale socket must be replaced.
    assert!(
        socket.exists(),
        "the killed manager leaves its socket behind"
    );
    let mut config = Config::dram_only(socket.clone(), 8 * 1024 * 1024);
    config.name = name.clone();
    let directory = config.shm_directory().unwrap();
    let before = |file: &str| {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(directory.join(file)).unwrap().ino()
    };
    let (arena, staging) = (before("arena"), before("staging"));
    fs::write(directory.join("generation.tmp"), b"debris").unwrap();
    let mut restarted = spawn("second.err");
    wait(|| std::os::unix::net::UnixStream::connect(&socket).is_ok());
    assert!(
        restarted.try_wait().unwrap().is_none(),
        "the restarted manager must stay up"
    );
    unsafe {
        libc::kill(restarted.id() as i32, libc::SIGTERM);
    }
    let status = restarted.wait().unwrap();
    assert!(status.success(), "{status:?}");
    assert!(!socket.exists());
    let log = fs::read_to_string(base.join("second.err")).unwrap();
    assert!(
        log.contains("reclaimed stale instance") && log.contains("(1 files)"),
        "missing reclaim log: {log}"
    );
    // Only unfinished metadata was reclaimed; the region files were reused.
    assert!(log.contains("reused its shared regions"), "{log}");
    assert!(!directory.join("generation.tmp").exists());
    assert_eq!((before("arena"), before("staging")), (arena, staging));
    let _ = fs::remove_dir_all(base);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_60_restore_is_not_refused_when_the_nvme_budget_is_full() {
    // One streamed record and two arena records fill the durable budget close
    // to --nvme. A restore window is transient staging traffic, not a durable
    // reservation, so the disk read must still be served.
    let base = base("restore-budget");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        1024 * 1024,
        3_200_000,
    );
    config.name = instance_name("restore-budget");
    let mut host = Host::open_config(config);
    host.run();
    let mut client = host.client().await;
    let streamed = client.store(key(512 * 64), None).await;
    client.store(key(64 * 64), None).await;
    client.store(key(64 * 64 + 1), None).await;
    host.durable(3);
    assert!(
        host.gauge("nvme_reserved_bytes") > 3_000_000,
        "the durable budget must be nearly full: {}",
        host.gauge("nvme_reserved_bytes")
    );
    assert_eq!(client.fetch(&[key(512 * 64)], 0, 512).await[0], streamed);
}
#[test]
fn l0_62_socket_directory_modes() {
    // The manager sets a mode only on a directory it created. An existing
    // directory keeps its mode: pointing the socket into /tmp or /run must not
    // change a host directory, and a manager started as root must not either.
    use std::os::unix::fs::PermissionsExt;
    let base = base("socket-mode");
    fs::create_dir_all(&base).unwrap();
    fs::set_permissions(&base, fs::Permissions::from_mode(0o755)).unwrap();
    let command = |name: &str, socket: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_selene"))
            .args([
                "--name",
                name,
                "--socket",
                socket.to_str().unwrap(),
                "--memory",
                "8M",
                "--staging",
                "8M",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    };
    let socket = base.join("manager.sock");
    let name = instance_name("socket-mode");
    let _shm = support::ShmCleanup::new(&name, &socket);
    let mut child = command(&name, &socket);
    wait(|| socket.exists());
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    assert!(child.wait().unwrap().success());
    let mode = fs::metadata(&base).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755, "an existing socket directory keeps its mode");
    // A directory the manager creates gets 0770.
    let directory = base.join("run");
    let socket = directory.join("manager.sock");
    let mut child = command(&name, &socket);
    wait(|| socket.exists());
    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }
    assert!(child.wait().unwrap().success());
    let mode = fs::metadata(&directory).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o770, "a created socket directory gets 0770");
    let _ = fs::remove_dir_all(base);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn l0_63_a_paused_put_keeps_its_window_past_the_deadline() {
    // A put that stalls past the no-progress deadline is failed, but its
    // granted window stays leased: the writer may still be writing into it.
    // A later put cannot take the extent, and bytes written after the deadline
    // are intact.
    let mut host = Host::dram_only(32 * 1024);
    host.run();
    let mut client = host.client().await;
    let (id, start) = client.start_put(key(3 * 64), None).await;
    assert!(matches!(start, PutStart::Accepted(_)), "{start:?}");
    let grant = client
        .session
        .take(|m| match m {
            M::PutGrant(grant) if grant.put_id == id => Some(*grant),
            _ => None,
        })
        .await;
    let window = grant.window.unwrap();
    // Stall until the manager ends the put for no progress.
    let refused = client
        .session
        .take_within(Duration::from_secs(10), &mut |m| match m {
            M::PutRefused(refused) if refused.put_id == id => Some(refused.clone()),
            _ => None,
        })
        .await
        .expect("the no-progress deadline did not fire");
    assert_eq!(refused.reason, v2::Reason::Invalid as i32, "{refused:?}");
    // Resume writing into the window; the extent must not be reusable.
    let pattern: Vec<u8> = (0..window.length as usize)
        .map(|index| (index % 251) as u8)
        .collect();
    client
        .session
        .shm
        .as_ref()
        .unwrap()
        .write(&window, &pattern);
    let mut other = host.client().await;
    assert_eq!(
        other.put(key(64), None, PutOptions::default()).await,
        PutOutcome::Refused(v2::Reason::Budget as u8),
        "the stalled put's extent must stay leased"
    );
    assert_eq!(client.session.shm.as_ref().unwrap().read(&window), pattern);
}

/// A startup failure after the store opened must release the instance and root locks and
/// leave no socket, so the same manager can start again on the same paths immediately.
#[test]
fn l0_67_a_failed_start_releases_the_instance_and_root() {
    let base = base("failed-start");
    let mut config = Config::new(
        base.join("records"),
        base.join("run/m.sock"),
        65536,
        64 * 1024 * 1024,
    );
    config.name = instance_name("failed-start");
    let squatter = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut failed = config.clone();
    failed.metrics = Some(squatter.local_addr().unwrap());
    let error = control::run(failed, Arc::new(AtomicBool::new(false)))
        .expect_err("a taken metrics port must fail the start");
    assert_eq!(
        error.kind(),
        std::io::ErrorKind::AddrInUse,
        "the metrics refusal is reported: {error}"
    );
    assert!(!config.socket.exists(), "a failed start leaves no socket");
    drop(squatter);
    // The instance, lock and root are free again: the same manager starts at once.
    config.metrics = None;
    let mut host = Host::open_config(config);
    host.config.metrics = None;
    host.try_start()
        .expect("the previous attempt released its locks");
    host.stop();
    let _ = fs::remove_dir_all(base);
}
/// Certificate errors are found before the manager opens its store or any listener, so a
/// failed start leaves no locks, socket or accepting data port behind.
#[test]
fn l0_68_a_tls_error_starts_nothing() {
    let base = base("tls-error");
    fs::create_dir_all(&base).unwrap();
    let socket = base.join("run/m.sock");
    let mut local = selene_manager::config::Config::defaults();
    local.name = instance_name("tls-error");
    local.socket = socket.clone();
    local.socket_explicit = true;
    local.root = Some(base.join("records"));
    local.memory = 65536;
    local.staging = 8 << 20;
    local.nvme = 64 << 20;
    local.tls_cert = Some(base.join("missing-cert.pem"));
    local.tls_key = Some(base.join("missing-key.pem"));
    local.listen = Some("127.0.0.1:0".parse().unwrap());
    let error = control::serve(local, Arc::new(AtomicBool::new(false)))
        .expect_err("a missing certificate must fail the start");
    assert_eq!(
        error.kind(),
        std::io::ErrorKind::NotFound,
        "the certificate error is reported: {error}"
    );
    assert!(!socket.exists(), "nothing was created");
    // A key that does not belong to the certificate is only caught by building the full
    // server configuration, which now happens before anything starts.
    let good = selene_manager::testing::certificate();
    let other = selene_manager::testing::certificate();
    let cert = base.join("cert.pem");
    let key = base.join("key.pem");
    fs::write(&cert, good.cert.pem()).unwrap();
    fs::write(&key, other.signing_key.serialize_pem()).unwrap();
    let mut local = selene_manager::config::Config::defaults();
    local.name = instance_name("tls-mismatch");
    local.socket = socket.clone();
    local.socket_explicit = true;
    local.root = Some(base.join("records"));
    local.memory = 65536;
    local.staging = 8 << 20;
    local.nvme = 64 << 20;
    local.tls_cert = Some(cert);
    local.tls_key = Some(key);
    local.listen = Some("127.0.0.1:0".parse().unwrap());
    let error = control::serve(local, Arc::new(AtomicBool::new(false)))
        .expect_err("a key that does not match the certificate must fail the start");
    assert!(
        error.to_string().contains("TLS"),
        "the certificate error is named: {error}"
    );
    assert!(!socket.exists(), "nothing was created");
    let _ = fs::remove_dir_all(base);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_pool_admits_concurrent_large_puts() {
    let mut host = Host::with_storage(8 * 1024 * 1024, 1800 * 1024);
    host.run();
    let mut writer = host.client().await;
    for (root, leaf) in [(64, 4096), (128, 4160), (192, 4224)] {
        writer.store(key(root), None).await;
        writer.store(key(leaf), Some(key(root))).await;
    }
    host.durable(6);
    let mut a = host.client().await;
    let mut b = host.client().await;
    let mut c = host.client().await;
    let (a, b, c) = tokio::join!(
        a.put(key(4097), None, PutOptions::default()),
        b.put(key(4098), None, PutOptions::default()),
        c.put(key(4099), None, PutOptions::default()),
    );
    for outcome in [a, b, c] {
        assert!(matches!(outcome, PutOutcome::Stored(_)), "{outcome:?}");
    }
    for tokens in [4096, 4160, 4224] {
        assert!(
            writer.candidate(tokens).await.is_none(),
            "old leaf survived"
        );
    }
    for tokens in [4097, 4098, 4099] {
        assert!(
            writer.candidate(tokens).await.is_some(),
            "new put is missing"
        );
    }
    assert_eq!(host.gauge("retry_budget_total"), 0);
}

#[test]
fn removed_shared_region_stops_the_manager_cleanly() {
    let base = base("removed-region");
    let mut config = Config::dram_only(base.join("run/m.sock"), 64 * 1024);
    config.name = instance_name("removed-region");
    config.staging = TEST_STAGING as u64;
    let socket = config.socket.clone();
    let directory = config.shm_directory().unwrap();
    let _cleanup = ShmCleanup::new(&config.name, &socket);
    let stop = Arc::new(AtomicBool::new(false));
    let manager_stop = stop.clone();
    let manager = std::thread::spawn(move || control::run(config, manager_stop));
    wait(|| socket.exists());
    fs::remove_file(directory.join("staging")).unwrap();
    wait(|| stop.load(Ordering::Acquire));
    let error = manager.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("RemoveIPC"), "{error}");
    assert!(!socket.exists(), "manager still accepts connections");
    fs::remove_dir_all(base).unwrap();
}
