# Operating selene-manager

This runbook covers the deployment model, sizing, refusal reasons, metrics, restarts and
upgrades for `selene`. The wire contract is [PROTOCOL.md](../PROTOCOL.md) and
[selene.proto](../../selene/proto/selene.proto); building, testing and containers are in
[README.md](../README.md); remote hosts are in [network-operations.md](network-operations.md).

## Deployment model

- **One manager per pool.** A pool is the set of Flare instances that share records. Run one
  `selene` for it, on a host with local NVMe if you want records to outlive the instances.
  Instances do not talk to each other; they talk to the manager, which brokers every copy.
- **The manager owns its NVMe directory exclusively.** `--root` is locked
  (`<root>/.lock`): a second manager on the same directory refuses to start. Do not put the
  directory on a network file system, share it between managers, or edit it while the manager runs.
- **Instances configure only their private tiers plus the pool address.** Each Flare instance
  keeps its own DRAM and NVMe settings for its private tiers and adds
  `--prefix-cache-pool <address>`; see
  [address forms](network-operations.md#addresses-and-startup-warnings). The manager's `--name` supplies the identity salt. Everything shared,
  the index, holds, windows, the manager's own store and the network, lives in the manager.
- **Never put the manager behind a load balancer.** A session is stateful: the index entries an
  instance announced, the claims in flight and the windows granted to it all live in the one
  manager process it is connected to. A balancer that spreads instances over several managers
  splits the pool, and one that moves a stream between them loses its claims. Give each pool one
  address.
- **Same host, same identity.** The manager and every instance that uses shared memory must
  run with the same effective uid and gid, share the socket directory and `/dev/shm/selene` at
  identical paths, and (in containers) share a PID namespace so peer credentials are real.
  Instances that cannot meet this still work over `--listen`, without shared memory.

## Sizing

The service is CPU-only and never touches GPU memory. Three budgets are configured
separately and shared by every scope: `--memory` (the DRAM arena), `--staging` (windows) and
`--nvme` (durable records).

`--memory` is the arena for the manager's own records served without touching NVMe. Size it for
the hot working set, not for everything. Allocation uses power-of-two size classes, so a record
consumes its length rounded up and usable capacity can be materially below the mapping size.
A record whose rounded extent exceeds the whole arena is not dropped (with `--root` it streams
to NVMe) but it will never be served from DRAM; without `--root` it can never be admitted and is
refused `OVERSIZE`.

`--staging` holds the windows of copies between instances and the reads of NVMe records. A
**GPU export is accepted only after staging for the whole export has been reserved** (one
block, rounded up to the allocator's size class), so the source's GPU pin lasts only for its
device-to-host copy: with too little free staging the claim fails `NO_WINDOWS` at once and
nothing is pinned. Host-tier exports that do not fit stream with two windows at a time. Provision
staging for the largest export you expect to run concurrently times `export_slots` (default 2 per
instance), plus the quarantined bytes below. The minimum is 8 MiB and the default 4 GiB.

One export of a long context needs staging for all of its pages and its row: a Qwen prefix of
about 33,000 tokens did not fit the former 256 MiB default and was refused `NO_WINDOWS` on every
request, so the requester recomputed it. When the first session of a scope arrives the manager
logs, once for that scope, how many tokens of context one export can hold in the configured
staging and warns if that is under 262,144 tokens (Flare's usual per-request limit; the manager
does not know a model's real maximum). The warning names the staging size that would hold it.
Size `--staging` from that line and from your longest contexts.

`--nvme` bounds record reservations: the rounded arena extent for an in-memory
record, or the padded file plus the two-window hold for a streamed put. It must exceed the
largest record an instance can put, or the put is refused `OVERSIZE`. Keep it above `--memory`:
DRAM eviction needs durability, so a full NVMe budget stops DRAM from making room.

The arena and staging are preallocated files in `/dev/shm/selene/<name>`, so the shared-memory mount
must provide at least `--memory + --staging` for each manager. A container's default 64 MiB
`/dev/shm` is far too small; use `--shm-size` or the host IPC namespace.

The files are fixed (`arena`, `staging`, and a `generation` and `holders` file beside them) and
persist across restarts. A restarted manager with the same `--memory` and `--staging` reopens the
same files at the same offsets, so instances that registered them with CUDA at startup keep
working; a killed manager never blocks a restart, and debris from a crash is reclaimed. If a size
changed or a file is unusable, the manager recreates them under a new generation and logs that
running instances must restart to use the pool: until they do, they run on local tiers only. After
a crash the manager keeps the regions out of service until each process that had a session has
reconnected or exited (PROTOCOL.md 8.6). The files outlive a stopped manager and keep their
memory: remove `/dev/shm/selene/<name>` to give it back.

Protect these files from systemd-logind `RemoveIPC`, which can unlink a normal
user's shared memory when their last login session ends, including files used
by containers with that uid. Run the manager and local instances under a
system account, preferably through the supplied systemd service. If a normal
login uid is required, set `RemoveIPC=no` in `/etc/systemd/logind.conf` and
apply the logind configuration according to the host's service procedures.
The manager checks the region files once per second. If one disappears or is
replaced, it drains and exits with an error naming `RemoveIPC`. Fix the host
configuration, then restart the manager and instances so they map the new
regions; it never recreates regions underneath live copies.

Pool names isolate key namespaces. Instances of one pool map its shared arena and
staging files. Processes running under the same uid can read other pools' files.
Give each trust boundary its own uid (with distinct manager names) or its own host.

### The manager's network card is the pool's aggregate bandwidth

Every copy between hosts passes through this manager: a restore that crosses the network
leaves the source's host, is assembled in the manager's staging and reaches the requester over
the manager's data listener (a source or requester on the manager's own host uses shared
memory and does not touch the card). The manager's NIC therefore bounds the pool's total
cross-host copy rate, summed over every instance and every rank, in both directions. A
multi-process instance (tensor-parallel ranks that are separate processes) multiplies the
number of copies in flight by its rank count, but moves the same bytes: each rank sends or
receives only its own slice, so a record of N bytes crosses the card once whatever the number
of ranks. Size the card (or bond several) for the aggregate you expect, put the manager on the
host whose card has the most headroom, and watch `selene_pool_channel_bytes_total`. Hashing for
multi-process instances (verifying a source's slices and computing the stored checksums, or the
reverse) runs on up to eight threads of the manager's host, overlapped with the network.

### Instance DRAM rows

A pooled instance keeps a retiring record's local rows pinned until its put
finishes. A slow manager can therefore exhaust the instance's rows and skip
new host publications. Rows take one quarter of the local DRAM budget while
preserving minimum page windows. Flare warns at startup when this leaves fewer than
`--max-concurrency` rows. Increase `--prefix-cache-memory`
or reduce concurrency; this threshold is a sizing warning, not a guarantee
that an arbitrarily long persistence backlog will fit.

## Multi-process instances

An instance whose tensor-parallel ranks are separate processes (possibly on separate hosts)
is one instance of the pool. Rank 0 opens the only session; every other rank runs an agent
that attaches to the manager with the `Rank` stream and moves only its own slice of each
window, in the manager's shared memory when it is on the manager's host and rank 0 is too, and
on its own data channel otherwise. Records at rest are the ordinary composite records: an
instance of this kind and a single-process instance of the same shape share them.

- Claims, exports and puts of such an instance are refused (`RANKS_DETACHED`, put
  `PUT_NOT_READY`) until every rank is attached.
- A rank that was replaced keeps its obligations: the manager reuses memory a rank could still
  touch only after that rank (identified by its own process, not by its number) attests it
  stopped, says goodbye quiesced, or exits. `selene_pool_quarantined_bytes` includes ranks.
- When rank 0 dies, the agents stay attached only to drain; they cannot be granted anything.
- Each peer has one persistent `RankLink` to rank 0 for control, jobs and completions,
  outside the manager. It is as trusted as the instance's execution rendezvous.

## Startup

See [network startup warnings and TLS rules](network-operations.md#addresses-and-startup-warnings).
A build without the `rdma` feature refuses `--rdma`. Configuration and TLS
validation run before the store or any listener opens.

Startup order is the root lock, NVMe recovery, shared-memory reclaim, then the socket. A second
manager on the same root or the same `--name` exits at once; a stale socket is replaced only after
confirming nothing is listening.

## Start the manager before the instance

An instance attaches to the pool once, at startup, and never later. Registering the manager's
shared memory with the GPU consumes device memory, which a serving instance must not do, so Flare
registers the manager's `arena` and `staging` files before it admits requests and only then. An
instance that starts while the manager is down waits for it up to 10 seconds
(`flare_selene::pool::STARTUP_WAIT`), then runs on its local tiers only, prints one warning, and does
not attach when the manager comes up afterwards: restart it to use the pool. The same holds for an
instance whose manager's certificate it does not trust, or whose protocol, encryption or scope
differs, and for one running when the manager recreates its regions (a size changed): it keeps
serving on its local tiers and prints one warning.

Order the manager first. With compose, give the manager a health check and make the instance wait
for it:

```yaml
services:
  manager:
    image: ${MANAGER_IMAGE:-selene:local}
    # ... the options of deploy/compose.yaml ...
    healthcheck:
      test: ["CMD-SHELL", "test -S /run/selene/pool.sock"]
      interval: 1s
      retries: 30
  flare:
    image: ${FLARE_IMAGE}
    depends_on:
      manager:
        condition: service_healthy
```

With systemd, give the instance's unit `Requires=selene.service` and `After=selene.service`. Restarting
the manager alone is safe for running instances as long as its `--memory` and `--staging` are
unchanged; changing either means restarting the instances too.

## Readiness and restarts

The manager is `READY` when its own disk recovery completes. It serves lookups,
claims and puts without waiting for previous instances to reconnect. Until recovery
finishes, requests answer `NOT_READY`; sessions and announcements remain available.
Sources reconnect and resnapshot independently, so their hints may temporarily miss.

- Records the manager held only in DRAM are lost with it; NVMe records are recovered and start in the
  disk tier. A put is complete (and the writer may drop its copy) only at `Stored`.
- `SIGTERM` or `SIGINT` drains: new sessions, lookups, claims and puts are refused, claims in
  flight run to a terminal state (bounded by their deadlines), then sessions are closed with
  `SHUTDOWN`.

## Refusal reasons

Claims are refused with `FAILED` and a reason; puts with `PutRefused`. Instances treat every
refusal as a normal outcome and move to another candidate, another tier, or recompute.

| Reason | Meaning | Operator action |
|---|---|---|
| `NOT_READY` | The manager is recovering or draining. | None; check `selene_pool_state`. |
| `CLAIM_LIMIT` | The requester already has its claim slots in flight (default 2). | None; limits do not scale with concurrency. |
| `SOURCE_LIMIT` / `SOURCE_BUSY` | The source already serves its export slots (default 2). | None. |
| `UNKNOWN_SOURCE` / `STALE_SOURCE` | The named source or entry is gone, or the source reconnected. | None; the index catches up. |
| `NO_WINDOWS` | Staging cannot hold the export, or too much of it is quarantined. | Raise `--staging`; look at `selene_pool_quarantined_bytes`. |
| `NOT_FOUND`, `CHAIN_GAP`, `NO_HEADROOM` | The source no longer has it, or has no room to pin it. | None. |
| `SOURCE_TIMEOUT` | The source did not answer within `pin_deadline_ms` (default 250). | Look at the source's load. |
| `SOURCE_LOST` / `REQUESTER_LOST` | The other side's stream ended. | Look at the instance that went. |
| `RANKS_DETACHED` / `RANK_LOST` | A rank of a multi-process instance is not attached, or its attachment ended during the transfer. | Look at that rank's process. |
| `COPY_ERROR` / `CHECKSUM` | A read, write or checksum failed, or the copy was not exact. | Alert on repeats; check the device. |
| `DEADLINE` / `NO_PROGRESS` | The claim ran past its deadline or stopped making progress (3s). | Look at copy throughput. |
| put `PRESENT` | Already stored and committed: success, nothing transferred. | None. |
| put `BUDGET` | Temporary arena, staging or durable-budget pressure: the writer keeps its local record and releases its selection. | Retry after manager progress or reconnect; add capacity if sustained. |
| put `OVERSIZE` | Can never fit in this configuration. | Raise the budget or add `--root`. |
| put `QUEUE` | The writer has `max_open_puts` (4) open. | None. |
| put `INVALID` | Malformed, metadata that does not match the put, or a mode not allowed. | Alert: client and manager disagree. |
| put `IO` | A device error. | Check the NVMe and the file system. |
| put `PUT_NOT_READY` | Not ready (see Readiness). | None. |

## Quarantine

Memory is reused only when every process that could still touch it has said it stopped, or is
confirmed gone. If a process cannot say so (it reports `QUARANTINED`, or its stream ended while the
process lives and it has not reported on a new session), the windows it held stay out of service
until it exits, or until the manager does. `selene_pool_quarantined_bytes` and `ReadinessInfo` show
the total; a nonzero value that does not fall means an instance is hung, not a leak.

## Metrics

`GET /metrics` on `--metrics HOST:PORT` returns Prometheus text (`selene_` prefix, content type
`text/plain; version=0.0.4`); only that path is served.

| Series | Meaning | Alert on |
|---|---|---|
| `selene_pool_state`, `selene_pool_ready` | 1 recovering, 3 ready, 4 draining; and 1 when ready. | Recovery that does not finish. |
| `selene_pool_sessions` | Live instance sessions. | Dropping to zero while instances serve. |
| `selene_pool_index_entries` | Entries announced to the index. | Unbounded growth. |
| `selene_pool_claims_in_flight`, `selene_pool_claim_records`, `selene_pool_lookups_queued` | Work in flight and records kept for fences. | Records that never fall. |
| `selene_pool_claims_total{state,reason}`, `selene_pool_claims_refused_total{reason}`, `selene_pool_lookups_total{status}` | Outcomes. | `COPY_ERROR`, `CHECKSUM`, `NO_WINDOWS` rising. |
| `selene_pool_puts_open` | Puts in flight. | |
| `selene_pool_quarantined_bytes` | Window bytes out of service. | Anything sustained. |
| `selene_puts_total{outcome}` | `ok`, `error`, `present`, `queue`, `budget`, `inflight`, `oversize` for the store's puts. | Any `error`; `budget` growing faster than `ok`. |
| `selene_gets_total{outcome}` | `ok`, `error` for the store's reads. | Any `error`. |
| `selene_nvme_errors_total` | NVMe read and write failures. | Any increase. |
| `selene_nvme_bytes_total`, `selene_nvme_seconds_total` | Durable bytes written and disk time. | Disk time growing faster than bytes. |
| `selene_memory_allocated_bytes`, `selene_staging_allocated_bytes` | Arena and staging in use. | Near `--memory` / `--staging`. |
| `selene_nvme_reserved_bytes`, `selene_nvme_reserved_peak_bytes` | What `--nvme` bounds, now and its high-water mark since start: every record's padded file or rounded arena extent, what each unfinished put will leave on disk, and the staging hold of each unfinished streamed put. Never above `--nvme`. Windows leased for claims and exports are not in it; `selene_staging_allocated_bytes` shows them with the holds, and `--staging` bounds them. | Near `--nvme`: puts are refused `BUDGET` and DRAM eviction stops. |
| `selene_evictions_total`, `selene_compactions_total` | DRAM evictions and row compactions. | Hot set not staying resident. |
| `selene_hits_total{namespace,tier}`, `selene_records{namespace,tier}`, `selene_bytes{namespace,tier}` | Hits and inventory of the manager's own records. | Hit rate collapse. |

## Recovery on disk

Recovery per scope removes leftover `.*.tmp` files, validates every `.ckpt` header, layout,
name and length, sorts parents before children, and deletes orphaned
descendants. A record that is corrupt in a detectable way is removed; one that cannot be read
because of an I/O error fails startup without deletion, so an operator can inspect or move it. If
the disk fills during a write the manager keeps the non-durable DRAM copies and retries every
100 ms; puts are refused `BUDGET` until space returns. Never free space by deleting files under a
running manager.

## Upgrades and clearing the cache

Stop the manager, replace the binary and start it with the same `--root`, `--name` and `--socket`.
Durable records are validated and reused; instances reconnect, resync and refill the DRAM tier from
their traffic. The protocol version must match the instances: a manager that does not speak the
instance's version answers `Hello` with `VERSION_MISMATCH` and the instance runs on its local tiers
until restarted. `--name`, `--socket`, `--listen`, `--listen-data` and the TLS files are restart-only.

The root holds one opaque 64-hex directory per scope plus `.lock`. Remove a scope
directory to drop that model's durable records, or all of them to clear the pool, only while the
manager is stopped. A changed model identity, layout fingerprint or pool name is a different scope, so
the old directory is safe to delete and cannot be reused by accident.

## Troubleshooting

| Symptom | Check |
|---|---|
| Manager exits at startup | Another manager on the same root or `--name`; a root that cannot do `O_DIRECT`; `the store cannot start its disk threads` (io_uring blocked: run with `--security-opt seccomp=selene-manager/deploy/seccomp-io-uring.json`, Docker's default profile plus the three io_uring calls; disabling seccomp is not needed); a certificate and key that do not match; `--rdma` with TLS. |
| Instances run on local tiers only | The pool address, socket permissions, uid/gid equality, `/dev/shm/selene` mounted at the same path in both containers, `selene_pool_ready`. |
| `NO_WINDOWS` | `--staging`; `selene_pool_quarantined_bytes`; many large exports at once. |
| Quarantined bytes do not fall | A hung instance (it will clear when the process exits). |

### Duplicate puts and stale lookup hints

`Present` means the complete record is already committed. Duplicate put logs
report cumulative totals at 1, 2, 4, 8, and subsequent doublings instead of
printing once per duplicate; every put still receives its own reply.

A restore warning ending in `UnknownSource` means that its lookup hint no
longer names an indexed source record. A source disconnect, manager restart,
resync, or eviction between lookup and claim can cause this race. Lookup does
not pin records. The manager refuses the claim before granting windows or
requesting an export, and the instance falls back to an equivalent local
copy or recomputes after draining. An occasional warning during restarts is
expected; repeated warnings on a quiet pool warrant checking source sessions
and record retention.
