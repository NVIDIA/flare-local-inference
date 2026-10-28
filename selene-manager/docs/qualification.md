# Qualification

The `selene` and `selene-manager` tests cover store rules at unit level, races and fencing with a scripted
wire client, and happy paths with the real client and manager. Real processes
cover SIGKILL. GPU integration remains in Helios.

## Test ladder

Each rung uses the real component below it and adds one class of risk. All of them run on a
Linux CPU host against a real manager; the pool tests drive it with a minimal wire-level
client (`tests/support/pool.rs`) and never model the manager's state machine.

| Suite | What it proves |
|---|---|
| Unit (`src/**`) | Modules in isolation: the index and chain walks, window planning, configuration, the record format, the arena, the disk worker, the store's admission and eviction rules, the data-channel modules, and the store-claim completion order. |
| `tests/pool_l0.rs` | Hello, sessions and supersede, notices, content keys, live resync, lookup, recovery readiness, limits, shutdown drain. |
| `tests/pool_claims.rs` | Claims between instances: retained and streamed copies byte for byte through the shared staging region, every refusal, timeouts, races, fencing and quarantine, Goodbye, stream blips, process replacement, and a real peer process that is killed or hung. |
| `tests/pool_store.rs` | Put into the manager's store and claims on its records: persisted checksums, recovered NVMe records, damage at rest, a requester whose stream drops, put cancel and fence. |
| `tests/pool_net.rs` | The network listener: sessions without shared memory, level-5 candidates, refusal of shared windows, TLS declared and RDMA withheld, the startup warnings and the refusal of RDMA with TLS. |
| `tests/pool_channel.rs` | Instances on other hosts: a chain put and restored over TCP byte for byte (mid-record starts, persisted checksums), an instance that never connects, dies mid-window or mid-fill, a released claim, a lost session, corrupt and cancelled puts, no listener; with the `rdma` feature the same over RDMA on a queue pair per transfer. |
| `tests/pool_relay.rs` | Relay faults between instances that share no memory with the manager: a source or a requester that dies mid-relay, a stalled source leg, a source that never connects, a source whose session is lost; every case ends with nothing quarantined and the same record relayable again. |
| `tests/client.rs`, `tests/client/` | Real client and manager: shared-memory, TCP, TLS and RDMA restores, restart, relay, multi-rank copies and faults, local retention and the opt-in shutdown drain. |
| `tests/rank_processes.rs` | SIGKILL during a multi-rank copy: quarantine until process exit, and peer drain after rank zero dies. |
| `tests/l0.rs` | The store engine through the pool protocol: chain durability, crash and recovery, streaming and oversize records, budget and cancel, directory modes, failed-start cleanup, signal handling, and kill/restart. |

## Running

Run from the Flare repository root, once for each of the two crates, `selene` and `selene-manager`.

```sh
for crate in selene selene-manager; do
  cargo test --manifest-path $crate/Cargo.toml --release --locked --features test-support
  cargo test --manifest-path $crate/Cargo.toml --release --locked --features test-support,rdma
  cargo fmt --manifest-path $crate/Cargo.toml -- --check
  cargo clippy --manifest-path $crate/Cargo.toml --all-targets --features test-support --locked -- -D warnings
done
```

Set `SELENE_NVME_SCRATCH` to a disposable directory on a local NVMe file system for the tests that
need real direct I/O. The tests that start managers need room in `/dev/shm`; in a container pass
`--shm-size=4g`. The `rdma` feature needs libibverbs and a CMake toolchain at build time (see
`Dockerfile.rdma`). RDMA is experimental and is not a correctness gate.

## Limitations

- Data-channel tests run over loopback (and Soft-RoCE for RDMA), so they exercise the framing,
  the fencing and the faults but not a real network's latency or loss; the two-host runs are a
  separate qualification.
- A peer killed mid-transfer is a connection that drops (and a queue pair that is destroyed); a
  half-open connection is bounded by the 20 s write and read limits and the claim's progress timeout.
- The tests cover the CPU data path and shared-memory windows. They do not include GPU transfer,
  scheduling or model execution time.
