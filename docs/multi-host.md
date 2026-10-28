# Multi-host serving

Flare can serve one prepared model across several DGX Sparks. Each host runs
one tensor-parallel rank; rank zero serves the HTTP API and coordinates its
peers. The package fixes the rank count and GPU architecture.

Serving ranks own package distribution and collective rendezvous. Choose how to
launch them:

| Mode | Command | Process ownership |
| --- | --- | --- |
| Manual | `flare run PACKAGE --rank 0 --rendezvous ZERO:PORT`, then `flare run --rank R --rendezvous ZERO:PORT` on each peer | Operator starts every rank |
| Agent-assisted | `flare run PACKAGE --hosts PEER1,PEER2,...` on rank zero | Local rank zero starts ordered peers through paired agents |
| Orbit | `flare orbit run PACKAGE` | Orbit places and supervises every rank through agents |

All three modes use the same native package receiver and CX7 transport. Orbit
adds placement, saved groups, detached operation, and monitoring. Its controller
needs no model GPU and can have a different CPU architecture from the nodes.

Graph input uploads use rank-control packets of at most 64 MiB. Larger input
sets are split on tensor-element boundaries, with the total bytes bounded by
the prepared input-frame extent. Each peer validates the complete upload set
before acknowledging readiness; graph replay begins only after every rank is
ready. Small input sets use one packet. Serving builds must agree on the rank
control protocol version; changing packet framing does not require package
re-export.

This guide covers setup and operation. The [Orbit reference](orbit.md) defines
placement, persistent groups, caching, and the protocol. To create a package,
use [Ignition preparation](#prepare-the-package); serving an existing package
needs only Flare, the driver, and the package.

## Requirements

- A multi-host package matching the serving GPUs and number of ranks. A
  single-host TP4 package needs four GPUs in one host; Orbit cannot turn it
  into a package for four separate Sparks.
- Compatible Linux, NVIDIA drivers, and Flare serving builds on the nodes.
  Each executable must support the package's graph contract and RoCE proxy ABI.
  [Debian packages](debian.md) provide the native CLI and agent; Ignition and
  the CUDA toolkit are needed only for preparation.
- Active RoCEv2 ports with IPv4 GIDs and reachable peer endpoints. Model
  collectives use RoCEnante over the Sparks' ConnectX-7 ports.
- Free GPU memory and disk space on each selected node. Spark shares host and
  GPU memory; see [DGX Spark preparation](model-preparation.md#dgx-spark).

### Prefix cache pool

A multi-host instance can use a Selene prefix cache pool (see
[prefix cache offload](prefix-cache-offload.md)).

- Pass the same `--prefix-cache-pool` setting to every rank, or to none.
- Local retention is disk only; `--prefix-cache-memory` is rejected.
- If any rank's registration is refused, every rank uses its local tiers.
- The pool and record links connect to rank zero's ephemeral TCP port and carry
  session secrets in plaintext, so secure the cluster fabric.
- Build with `rdma` only if the manager offers RDMA.
- Restarting with wiped local disks restores nothing unless records had already
  cascaded to the manager; see
  [sharing through a pool](prefix-cache-offload.md#215-sharing-through-a-pool).

### Network connections

| Connection | Destination | Port |
| --- | --- | --- |
| Agent control and package transfer | Every agent, reachable from the controller and source/peer nodes | TCP 29900 by default |
| Serving package payloads | Serving ranks' CX7 interfaces | Dynamically allocated TCP listener per rank |
| Serving rendezvous | Rank zero, reachable from every peer | TCP 29800 for Orbit launches |
| Host prefix record service | Rank zero, reachable from every peer | Dynamically allocated TCP listener per launch; peers connect during startup when a disk prefix cache is configured |
| Prefix pool and record links | Rank zero, reachable from every peer | Dynamically allocated TCP listener per launch, on the rendezvous interface |
| HTTP API | Rank zero, reachable from clients | TCP 8000 in this guide |
| Ignition preparation rendezvous | Preparation rank zero | The selected port; examples use TCP 29630 |
| Ignition preparation coordination | Preparation hosts | Additional dynamically allocated TCPStore and Gloo ports |
| Ignition checkpoint and export payloads | Preparation hosts' CX7 interfaces | Dynamically allocated TCP listener per rank |

Agent control uses encrypted TCP to the addresses recorded at pairing. Serving
ranks distribute package payloads through dynamically allocated TCP listeners on
the discovered CX7 interfaces. Direct mode connects to each peer; `--roce-ring`
relays non-neighbor data through the ring, and `--roce-chain` along a chain. Serving distribution uses plain TCP
on the dedicated cluster fabric, with package signature verification and
publisher acceptance before GPU loading. It requires no route keys or agents.
Model collectives use RDMA. Agent-based prestaging with `orbit push` uses the
agents' existing authenticated sessions and signed grants; those agents must
also listen on their fabric addresses. Flare does not configure interfaces,
addresses, routes, switches, or firewalls.

The serving ranks select up to two rails per peer from active devices. The
package does not pin the physical fabric used during preparation. For direct
cabling and neighbor forwarding, see [experimental rings](#experimental-rings).

## Manual serving

Rank zero holds the complete primary and companion files. Start a two-host
package as follows, using the same serving options on both hosts:

```sh
# Rank zero on maxwell
flare run /data/model.safetensors --rank 0 --rendezvous maxwell:29800

# Rank one on hertz; the package is fetched from rank zero
flare run --rank 1 --rendezvous maxwell:29800
```

For a ring, add `--roce-ring` on rank zero and start every peer with its physical
rank number. Rank zero sends the routing mode to peers. `--transfer-streams N`
on rank zero selects 1..64 TCP streams per peer, default 8, across available CX7
rails. A peer with a complete local rank copy can supply its package path;
Flare verifies that its export matches rank zero's.

Received files contain only the rank's required ranges. Flare verifies their
stored data, reuses verified cache entries, and keeps relays alive until every
rank has finished receiving. `--model-cache PATH` selects a receiver's cache
root; `--package-cache PATH` is the compatible fallback when `--model-cache` is
absent. The default is `paths.package_cache`, otherwise
`~/.cache/flare/packages`. `--stage-only` on rank zero distributes and
verifies packages, then exits all ranks before GPU loading. Manual ranks apply
normal publisher trust independently; unattended first use requires the exact
publisher key and namespace acceptance flags described in the CLI reference.

## Launch peers through agents

Install an agent on each peer and pair it from the head node:

```sh
flare agent pair hertz --code CODE_FROM_HERTZ
flare run /data/model.safetensors --hosts hertz
```

`--hosts` lists peers only, in rank order; the local process is rank zero.
Names or addresses must match paired agent records. The head node needs no
agent, cluster definition, or saved group. Peer processes use the executable
running their agents. The default rendezvous is the head node's address toward
the first peer on port 29800; `--rendezvous HOST:PORT` overrides it.

Pairing shares the node registry and controller key with Orbit, so existing
pairings work in both modes. `--agents-file PATH` and `--controller-key PATH`
select those files when their default paths are unsuitable. The local process
forwards peer output and owns the launch lifetime; after local serving drains,
it waits for peer exits. Use Orbit for detached groups.

## One-time setup: the Flare agent

### Install on every serving host

Run as the account that owns the model files. With the Debian package:

```sh
/usr/bin/flare agent install
```

With a source build, use `./flare/target/release/flare agent install` instead.
The installer copies that executable to `~/.local/lib/flare-agent/flare`,
enables and restarts the `flare-agent` systemd user service, and requests login
lingering. It prints the node name, fingerprint, and a one-time pairing code
valid for 24 hours. Check the service with:

```sh
systemctl --user status flare-agent
journalctl --user -u flare-agent
```

`--listen IP:PORT` changes the default `0.0.0.0:29900` listener. For a trusted
network, `flare agent install --accept-first` allows the first controller to
pair without a code. It does not reset existing pairings.

### Pair from the controller

Default installations generate a different code on each host. Pair each host
with the code it printed; substitute the actual codes in this example:

```sh
flare orbit pair maxwell --code CODE_FROM_MAXWELL
flare orbit pair hertz --code CODE_FROM_HERTZ
```

For four hosts, pair `faraday` and `ampere` in the same way. A single
`flare orbit pair maxwell hertz ... --code CODE` works only when every host was
installed with the same explicit `--join-code CODE`. Hosts installed with
`--accept-first` pair without `--code`.

Compare the reported fingerprints with those printed on the hosts. Pairing
records node addresses and public keys in `~/.config/flare/orbit.toml` and
creates the controller's private key at
`~/.config/flare/orbit/controller.key`. An authorized controller can run code
as the agent account; keep its private key private.

### Define a cluster

Add a named list to the controller's orbit file, using the names pairing
recorded. This example uses two Sparks:

```toml
[cluster.sparks]
nodes = ["maxwell", "hertz"]
```

For a TP4 package, list four suitable nodes in the intended rank order.
A cluster is a candidate list: when it has exactly one node per rank, its first
node is rank zero. Longer lists can skip nodes that cannot serve a rank.
Always inspect the printed placement.

```sh
flare orbit status --cluster sparks
```

`status`, `ls`, `packages`, `dashboard`, `attach`, and `stop` default to the only
configured cluster. `run` instead considers all paired nodes unless `--cluster`,
`--nodes`, or a saved group's node list constrains placement.

To authorize another controller, use `flare orbit controllers add NAME PUBLIC_KEY`
from an authorized controller. Supply its hex public key, not its private key
or a key filename. See [controller-set management](orbit.md#controller-set).

## Package sources

Orbit accepts a Hugging Face repository `ORG/NAME[@REVISION]`, a local primary
path, or `NODE:/absolute/path`, where NODE is a paired node's name. Repository
packages download on a candidate node, keeping the full package off the
controller. The source node reads its package in place. Orbit stages the complete package
on rank zero when needed; serving ranks then distribute peer shards.

For a package published on Hugging Face, launch from its repository:

```sh
flare orbit run ORG/MODEL --revision gb10
```

Replace `ORG/MODEL` with the package repository. If the revision contains several
primary files, select the multi-host package with `--filename FILE`. `gb10` selects the Spark hardware revision even from
a controller without a GPU. Without an explicit revision, Orbit infers one
from available candidate GPUs and refuses ambiguous mixed hardware.
`--offline` and `--cache-dir` apply to the downloading node; `--token-file`
is read on the controller and its token travels through the encrypted session.
To download on a particular source host before launching, run `flare download`
there and pass its returned primary path as `NODE:/absolute/path`.

Downloads reuse the Hugging Face cache. Keep the primary and its companion
filenames and symlink targets intact. To inspect the package's rank count and
hardware contract on the source host, run `flare inspect PRIMARY --plain`.

### Checkpoint and package storage

| Artifact | Preparation rank zero / source host | Peer host |
| --- | --- | --- |
| Source checkpoint | Complete checkpoint in its hub cache or local directory | Requests only the source ranges its loader needs during preparation |
| Export data | Collects rank manifests and requests payload files absent from its cache | Prepares and retains its rank's payload blobs locally |
| Serving package | Complete package on serving rank zero; an external source may also retain a complete copy | Required rank payloads and shared data |
| Serving executable | Installed agent binary by default | Installed agent binary by default |

Exports put metadata in the primary and all payloads in subsequent files bounded
by the `64G` shard limit. Identical payloads share storage through hardlinks.
Snapshot exports and legacy sparse packages have different allocation patterns;
see [package caching](orbit.md#package-caching).

Budget the checkpoint, previously retained blobs, missing payloads from a new
export, and temporary staging. Deduplication reduces repeated writes but does
not eliminate preparation memory or all disk reads. `--no-weights` requires
matching cached payloads. Output and package cache must be on the same filesystem
for hardlinks; if `/data` is a separate mount, configure `[paths] package_cache`
on that filesystem before using the export example below. See
[package storage](cli-design.md#package-snapshots).

## Prepare the package

Skip this section when serving a prepared package. Preparation uses Ignition
processes started directly on each host. The Flare agent does not run Ignition.
Use matching Flare, Ignition, and locked FlashInfer sources on every preparation
host; source-identity mismatches are rejected. Recipe-supported rank counts and
model options are listed in [model preparation](model-preparation.md).

Provision the same preparation token on every host before starting. Set
`[preparation] checkpoint_token_file` in `~/.config/flare/config.toml`, or use
its default `~/.config/flare/agent.token`. The file needs identical contents of
at least 32 random bytes and mode `600` on each host. This token is separate
from Orbit identities. Checkpoint and export frames use that token for
authentication without encryption. Payloads travel through rank-owned helpers on the CX7 interfaces,
using direct peer connections or ring relays selected by `--roce-ring`.

The example prepares GLM-5.3 Flash for eight concurrent requests on two Sparks.
Run from each checkout root.
On `maxwell`, whose preparation address is `10.200.0.12`, publish to its model
cache. Replace `CACHE_ENTRY` in the conversion command with the printed entry
path after preparation completes:

```sh
.venv/bin/ignition export glm-5.3-flash --ranks 2 \
  --rendezvous 10.200.0.12:29630 --gpus 0 --max-concurrency 8
.venv/bin/ignition convert CACHE_ENTRY --form packed --output /data/flare-runtime/glm53-spark-tp2.safetensors
```

On `hertz`:

```sh
.venv/bin/ignition join 10.200.0.12:29630 --rank 1 --ranks 2
```

For four hosts, use `--ranks 4` and start peer ranks 1, 2, and 3. Rank zero
admits each peer once, serves checkpoint metadata and selected byte ranges,
and coordinates preparation. Peers do not create full checkpoint shards.
Keep checkpoint source files unchanged during loading.

For a physical ring, add `--roce-ring` to rank zero's `export` or `debug-serve` command;
for hosts cabled as a chain with no link between the first and last rank, add
`--roce-chain` instead. The peers' `--rank` values define physical order. Preparation does not
require Orbit agents, pairing, a cluster file, or controller keys. Every rank
starts one native helper, exchanges its discovered fabric endpoints through
the rendezvous, and keeps the helper alive while dependent transfers run.
All hosts need matching Flare executables and preparation sources. Ignition
finds Flare on `PATH` or at `flare/target/release/flare`; check for stale installed
executables. Direct connections between two Sparks and switched fabrics use
peer-to-peer CX7 TCP without `--roce-ring`.

Checkpoint loading uses eight TCP streams per peer by default. Select 1 through
64 with `--transfer-streams N`. Streams alternate between available rails.
Direct topology sends each peer its requested ranges directly. Ring topology uses the shortest path from rank
zero; equal-length paths follow increasing rank numbers, wrapping at the rank count.
Each hop binds and checks the exchanged fabric IPs. A failed fabric channel ends
preparation instead of retrying through management.
Admission metadata and coordination use the rendezvous connection.
Export retains the authenticated fabric channels after checkpoint reads drain:
peers send missing prepared payload files back through the same direct connections
or ring hops and rails. Rank zero receives from all peers concurrently, with one file per stream
and bounded 4 MiB frames. Each received file is hashed and published atomically.
On failure, sibling connections are aborted and workers are joined; incomplete
files are never published. Export manifests and cache requests use the
rendezvous connection.

Prefetch retains at most one 4 MiB payload per stream plus bounded transport
buffers. The native weight loader copies data in order, preserving packed rows,
strides, and conversions. Peers retain metadata and one reusable 4 MiB
memory-backed file, passed to the loader by file descriptor, without writing
checkpoint weights to disk. The preparation report records the checkpoint
transport, stream count, payload bytes, and the interval from the first payload
read to the last; this interval includes intervening loading work.
Export reports missing payload bytes, stream count, receive duration, and
throughput separately for each peer. Cache hits skip bulk payload transfer.

Each rank loads its weights and prepares its graphs. Forward compilation is
shared across ranks, and compiled artifacts go to the ranks that need them.
Peers send export manifests; rank zero requests missing payload files,
assembles the package, signs it, and zstd-compresses executable resources. Temporary
staging is removed after use; reusable cached blobs remain.

GLM-5.3 Flash supports `--no-context-parallel` to replicate attention pages and
`--no-stream-overlap` to serialize concurrent branches. These are preparation
choices that change the captured graphs. Use the recipe-specific guidance for
other models.

## Serve

From the controller:

```sh
flare orbit run maxwell:/data/flare-runtime/glm53-spark-tp2.safetensors \
  --cluster sparks --name glm53 --gpus 0 --host 0.0.0.0
```

`--dry-run` prints placement without starting ranks or saving a persistent group.
It still reads the package metadata, verifies its signatures, and queries nodes.
The package fixes the rank count; the candidate list needs enough eligible nodes.
`--nodes maxwell,hertz` supplies an ordered list directly.

Each rank uses its node's installed agent executable. `--executable PATH`
uploads a controller-local override that must run on all selected nodes. Saved
groups retain an explicit override. A controller with a different CPU
architecture needs no node binary for the default path, provided the agents
advertise support for it.

The controller verifies the signed index and obtains publisher acceptance.
Rank zero receives a complete package if it does not already hold the source;
peers fetch only their required data. Fetched data is verified and cached.
Every serving rank checks its signatures; `--verify-weights` also digests stored
data during loading.

Rank zero listens for peers on TCP 29800 at the address it routes toward the
first peer. Ranks agree on the package and capacities, load their resources,
and connect RoCE. The HTTP API runs on rank zero's host, not on the controller.
The example binds all interfaces on Maxwell:

```sh
curl -fsS http://maxwell:8000/health
```

`--prefix-cache-disk DIR --prefix-cache-disk-limit SIZE --prefix-cache-memory 0`
keep published prefix checkpoints on every host's disk. Each rank writes its
own share of a record under `DIR` on its host; rank zero indexes, sizes, and
evicts every record and drives each peer's copies. Give every rank the same
settings: startup fails when one rank has a store and another does not. A
memory tier cannot span hosts, so a nonzero `--prefix-cache-memory` is
rejected.

The effective limit is the smallest limit among the ranks, because every host
must hold the same records. When the limit is omitted, each host derives one
tenth of its own free space and the ranks use the smallest of those values; the
startup line reports the limit each rank uses.

A rank's record connection carries every store, fetch, truncate, and unlink. If
it fails while a job is outstanding, the ranks stop serving rather than
continue with a cache they can no longer account for, and the waiting request
fails. Restart the group after a record-connection failure.

The startup phase `Join host prefix stores` reports how many records every rank
holds. In `/metrics`, `flare_host_cache_stored_total` counts copy-outs,
`flare_host_cache_hits_total` counts restores, and
`flare_host_cache_reused_tokens_total` counts the tokens those restores
covered. See
[prefix cache offload](prefix-cache-offload.md#214-ranks-on-separate-hosts).

A printed URL or successful `--detach` means rank processes started; wait for
a successful health response before sending inference requests. The default
HTTP bind is `127.0.0.1` on rank zero, so remote clients need an appropriate
`--host` value.

## Manage groups

Use names or exact launch IDs from `ls` to select runs:

```sh
flare orbit ls --cluster sparks
flare orbit dashboard --cluster sparks
flare orbit attach glm53 --cluster sparks
flare orbit stop glm53 --cluster sparks
```

Ctrl-C during attached `orbit run` stops that group. Ctrl-C during `orbit attach`
only detaches. A disconnected controller leaves ranks running. `stop` signals
rank zero first so it can drain collectives, then stops peers still running
30 seconds after rank zero exits. Omitting the group from `stop` selects every
running group in the chosen cluster.

Ordinary launches are ephemeral. Agents retain successful run history for five
minutes after exit, capped at sixteen successful records per agent. Failures
remain until removal or agent restart. Logs and metrics are held in memory;
restart loses them. Each agent retains about 64 KiB of output per rank.

To save a configuration for later manual restart:

```sh
flare orbit run maxwell:/data/flare-runtime/glm53-spark-tp2.safetensors \
  --cluster sparks --name glm53 --persistent --detach \
  --gpus 0 --host 0.0.0.0
flare orbit stop glm53 --cluster sparks
flare orbit run --name glm53 --detach
```

Stop any existing `glm53` run before creating the saved group. Persistent
means the controller retains the launch definition; it does not automatically
restart ranks after failure or reboot. `flare orbit rm glm53` removes a stopped
saved group and its retained history. Removing an exact launch ID keeps the
saved definition. Neither operation deletes model data. See
[persistent and ephemeral groups](orbit.md#persistent-and-ephemeral-groups).

## Update agents

Stop groups before replacing their agents. Update Flare on each host, then run
`/usr/bin/flare agent install` for a Debian installation, or the equivalent
command from the selected source build. This updates the executable used for
both the agent and default serving, and restarts the service. Preserve a custom
listener by passing `--listen IP:PORT` again. Existing node keys, authorized
controllers, and caches remain; re-pairing is unnecessary unless the node was
reset.

APT upgrades alone do not replace the agent's private executable copy or restart
its service. The systemd service may restart a failed agent, but agents do not
restore the rank processes it previously supervised. After updating all nodes,
check `orbit status` and restart saved groups explicitly.

## Failures

| Error or symptom | Check |
| --- | --- |
| Agent unreachable | Service state, recorded address, route, and TCP listener; both controllers and peers must reach the agent. |
| Unauthorized controller | Use an already authorized controller to add the key, or pair an unpaired node. |
| Join code rejected | Use that host's code before expiry and before its first successful use. |
| Agent protocol mismatch | Update controller and agents to compatible protocol versions. |
| Agent does not advertise an installed executable | Update the agent with `flare agent install`, or supply a node-compatible `--executable PATH`. |
| No eligible GPU or GPU held by a group | Inspect `orbit status`; stop the owning group or select other candidates. |
| `cannot access package` | Check the local primary or `NODE:/absolute/path`; repository inputs use `ORG/NAME[@REVISION]`. |
| `/health` unavailable after `--detach` | Follow startup logs; check rank-zero bind address, port, and model-loading progress. |
| Preparation authentication failed | Identical private checkpoint-token contents on every preparation host. |
| Checkpoint source changed | Keep source checkpoint files unchanged through loading. |
| Rank identity or graph I/O schema differs | Matching export, prepared world size, serving options, and compatible executables. |
| Package RoCE proxy ABI mismatch | Align preparation and serving dependencies; re-export when the executable cannot load that ABI. |
| Missing GID or rail/queue-pair error | Active ports, IPv4 RoCEv2 GIDs, reachable endpoints, and the intended rank order. |

Inspect rank output and `journalctl --user -u flare-agent` before restarting a
failed group. A collective failure invalidates the serving run; the agent does
not retry it automatically.

## Experimental rings

`--roce-ring` connects neighbors in selected rank order and uses native CPU relays
for other destinations. Transport supports 2 through 16 ranks, subject to each
recipe's supported sizes. Four-Spark transport checks pass; physical ring cabling
and full-model ring serving remain unqualified. `--roce-chain` uses the same relays
over a chain with open ends. See [RoCE ring transport](roce-ring.md).
