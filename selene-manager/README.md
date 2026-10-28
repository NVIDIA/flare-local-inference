# selene-manager

The Selene manager (binary `selene`): a Linux, CPU-only service that lets Flare instances
share KV-cache records. Instances announce what they hold; the manager answers lookups from
that index, brokers every copy between instances (it never moves a byte through a message), keeps
its own store of records in DRAM and on local NVMe for instances to spill into, and owns the
shared-memory windows the copies go through. It never accesses GPU memory and holds no document
content in its index.

Flare copies each published GPU checkpoint into its first configured host tier,
in both local and pool modes. Pooled instances announce the GPU holding at
publication and cascade retained records under host pressure: DRAM to local
NVMe, then the manager. A handover drops the source only after the destination
commits. Refusals, unavailable destinations and repeated failures keep records
local and retry after progress; see [host retention](../docs/prefix-cache-offload.md#215-sharing-through-a-pool). GPU eviction does not initiate a late copy.
The [shutdown drain](../docs/prefix-cache-offload.md#25-transfers-slabs-and-record-files)
is opt-in and off by default; ordinary shutdown still drains accesses before releasing memory.
A GPU cache hit refreshes an existing local host record's recency, but cannot
recreate an evicted record or prevent checkpoint compaction.

The protocol is `selene.v2` ([selene.proto](../selene/proto/selene.proto)); the state machines and
fencing rules are in [PROTOCOL.md](PROTOCOL.md). Running it is in
[docs/operating.md](docs/operating.md), serving other hosts in
[docs/network-operations.md](docs/network-operations.md), and what the tests prove in
[docs/qualification.md](docs/qualification.md).

## Build and run

Use Rust 1.88 or newer (validated with 1.94.1 on Linux). Serving is async on tokio and speaks
gRPC through tonic and prost over a Unix socket; the crate also depends on `flare-selene` (by
path, for shared record, index and wire types), `flare-eos` for host file I/O, and `blake2`.

Run these commands from the `selene-manager/` directory.

```sh
cargo build --release --locked
mkdir -p /run/selene /var/lib/selene
target/release/selene \
  --socket /run/selene/pool.sock \
  --root /var/lib/selene \
  --memory 8G --nvme 256G --staging 512M \
  --metrics 127.0.0.1:9108
```

Run with the same effective uid and gid as the instances. A missing socket directory is created
with permissions 0770, an existing one keeps its mode and ownership, and the socket itself is
0660. One process exclusively locks each root, and startup refuses to replace a live socket. A
stale socket from a killed manager is removed once startup confirms nothing is listening. The NVMe
file system must support 4096-byte O_DIRECT reads and writes, rename and file/directory sync. Do
not use a network file system.

Size options accept bytes with binary K/M/G/T suffixes.

| Option | Default |
|---|---|
| `--name` | `default`; names the shared-memory instance and the default socket |
| `--socket` | `/run/selene/<name>.sock` |
| `--root` | none; records stay in DRAM until evicted |
| `--memory` | 1 GiB |
| `--nvme` | 0 without `--root`; required with `--root` |
| `--staging` | 4 GiB (minimum 8 MiB) |
| `--listen` | none; serve instances on other hosts at ADDR:PORT |
| `--listen-data` | the host of `--listen`, next port; where those instances copy bytes |
| `--tls-cert`, `--tls-key` | none; TLS for `--listen` |
| `--rdma` | off; experimental, needs the `rdma` feature, not with TLS |
| `--metrics` | disabled; when set, HTTP `GET /metrics` |

See [address forms and startup warnings](docs/network-operations.md#addresses-and-startup-warnings)
for network connection settings and diagnostics.

The arena and staging are preallocated files under `/dev/shm/selene/<name>`, so the shared-memory
mount must hold at least `--memory + --staging`. Budgets cover record and window extents, not total
process RSS.

## Docker and systemd

```sh
docker build -f selene-manager/Dockerfile -t selene .   # from the Flare repository root
docker run --rm --name selene \
  --user "$(id -u):$(id -g)" --pid=host --network=host --ipc=host \
  --security-opt seccomp=selene-manager/deploy/seccomp-io-uring.json \
  -v /run/selene:/run/selene \
  -v /var/lib/selene:/var/lib/selene \
  selene --root /var/lib/selene --nvme 256G --memory 8G \
  --metrics 127.0.0.1:9108
```

Use a system account for the manager and local instances, or set
`RemoveIPC=no` in `/etc/systemd/logind.conf`. Otherwise logind can delete their
shared-memory files at logout, including when they run in containers. The
manager then drains and exits with a `RemoveIPC` error; restart the manager
and instances after fixing the host configuration.

Create `/run/selene` and the root directory with ownership matching the runtime uid/gid. The
manager and every instance must share `/dev/shm/selene`: run them with `--ipc=host` as above, or
bind-mount the same directory at `/dev/shm/selene` in each container. Share the socket directory
and a PID namespace with the instances so the peer pid matches `SO_PEERCRED`. No GPU flags, CUDA
libraries or privileges are required, and the image has no CUDA dependency.

The disk threads use io_uring, which Docker's default seccomp profile blocks. Run the container
with `deploy/seccomp-io-uring.json`: Docker's default profile with `io_uring_setup`,
`io_uring_enter` and `io_uring_register` added to the allowed syscalls (diff it against moby's
`default.json` to review). Disabling seccomp is not needed. Without the profile the manager exits
at startup with an error naming io_uring.

`deploy/` has a compose file for one host (`compose.yaml`), one that also serves other hosts over
TLS (`compose-network.yaml`), and a systemd unit (`selene.service`). `Dockerfile.rdma` builds the
experimental RDMA variant. Configure clients using the
[address forms](docs/network-operations.md#addresses-and-startup-warnings).

## Source layout

| Path | What it holds |
|---|---|
| `src/pool/` | The pool actor: sessions, the announcement index, lookup, claims and fences, put, readiness. |
| `src/store.rs`, `src/disk.rs`, `src/record.rs`, `src/arena.rs` | The manager's store library: DRAM arena, NVMe records, eviction, durability, recovery. |
| `src/local.rs` | Shared-memory regions, windows and process-exit (pidfd) reclaim. |
| `src/control.rs`, `src/config.rs`, `src/main.rs` | Startup, listeners, shutdown, flags. |
| `src/data/channel.rs` | The data channel instances without shared memory copy through: the listener, the per-transfer handlers and their fencing, over the stream and RDMA. |
| `src/data/ibv.rs`, `../selene/src/shared/rdma.rs` | Registered regions, shared verbs setup and the write-only RDMA channel. |
| `src/regions.rs` | The store's two regions as the data channel reads and writes them. |

## Tests

Run on Linux. See [qualification](docs/qualification.md) for the test ladder,
commands and hardware requirements. The `selene` and `selene-manager` crates include client/manager tests
and the process tests for SIGKILL; GPU integration is in Helios.
